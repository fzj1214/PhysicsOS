from __future__ import annotations

import itertools
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np

from physicsos.paths import resolve_workspace_path
from physicsos.schemas.case_runtime import ConvergenceStudyInput, ConvergenceStudyOutput, ExecuteCaseInput
from physicsos.runtime.artifacts import artifact, digest, write_json
from physicsos.runtime.domain import prepare_domain
from physicsos.runtime.execution import execute_case, load_domain
from physicsos.runtime.verification import _cells, comparison_error, evaluate_reference, probes, sample_solution, solution_context


def _characteristic_size(run, workspace: Path) -> float:
    config, _ = solution_context(run, workspace)
    if config["domain"]["requirements"]["representation"] == "mesh":
        points, cells, _ = _cells(config)
        maximum = max(float(np.linalg.norm(points[cells[:, a]] - points[cells[:, b]], axis=1).max()) for a, b in itertools.combinations(range(cells.shape[1]), 2))
        return maximum
    import json
    grid = json.loads(Path(config["domain_artifacts"]["background_grid"]).read_text())
    return max(float(np.diff(grid["axes"][name]).max()) for name in "xyz")


def _freeze(source: Path, target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    for helper in source.parent.glob("*.py"):
        shutil.copyfile(helper, target / helper.name)
    return target / source.name


def convergence_study(input: ConvergenceStudyInput, workspace: Path) -> ConvergenceStudyOutput:
    directory = workspace / "cases" / input.case_id / "studies" / uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    runs = []
    rows = []
    status = "uncertain"
    message = "Independent reference is missing; recorded actual reruns only."
    observed_order = None
    points = weights = None
    try:
        base = load_domain(input.prepared_domain, workspace)
        if base.case_id != input.case_id or (input.axis == "mesh") != (base.requirements.representation == "mesh"):
            raise ValueError("Study axis or case does not match the prepared domain.")
        source = resolve_workspace_path(input.kernel_uri or f"cases/{input.case_id}/taps/kernel.py", workspace=workspace)
        kernel = _freeze(source, directory / "kernel")
        kernel_hash = digest(kernel)
        reference = _freeze(resolve_workspace_path(input.reference_uri, workspace=workspace), directory / "reference") if input.reference_uri else None
        if reference and digest(reference) == kernel_hash:
            raise ValueError("Kernel and independent reference must be distinct implementations.")
        frozen_hashes = {str(path): digest(path) for path in directory.rglob("*.py")}
        for level, value in enumerate(input.refinements):
            recipe = base.recipe.model_copy(deep=True)
            recipe.force_remesh = True
            if input.axis == "mesh":
                recipe.requirements.mesh_policy.target_element_size = value
            else:
                recipe.requirements.grid_resolution = [int(value)] * 3
            prepared = prepare_domain(recipe, workspace)
            if prepared.domain.status != "ready":
                raise ValueError(f"Refinement {level} is {prepared.domain.status}: {prepared.domain.required_actions}")
            executed = execute_case(ExecuteCaseInput(case_id=input.case_id, prepared_domain=prepared.manifest, kernel_uri=str(kernel), controls=input.controls, field_name=input.field_name, timeout_seconds=input.timeout_seconds), workspace)
            runs.append(executed.manifest)
            if executed.run.result.status != "success":
                status = "failed"
                raise ValueError(f"Refinement {level} failed to execute: {executed.run.errors}")
            if points is None:
                points, weights = probes(executed.run, workspace, input.max_samples)
                np.save(directory / "probe_points.npy", points)
                np.save(directory / "probe_weights.npy", weights)
            actual = sample_solution(executed.run, workspace, points)
            np.save(directory / f"sampled_solution_{level}.npy", actual)
            row = {"level": level, "requested_refinement": value, "h": _characteristic_size(executed.run, workspace), "domain_id": prepared.domain.id, "run_id": executed.run.id, "kernel_sha256": executed.run.kernel_sha256}
            if reference:
                config, _ = solution_context(executed.run, workspace)
                expected = evaluate_reference(reference, input.reference_function, points, config, directory / f"reference-evaluation-{level}", input.timeout_seconds)
                error, scale = comparison_error(actual, expected, weights)
                row.update(error=error, reference_rms=scale)
            rows.append(row)
        if any(not Path(path).is_file() or digest(Path(path)) != checksum for path, checksum in frozen_hashes.items()):
            raise ValueError("Frozen study implementations changed during reruns.")
        sizes = [row["h"] for row in rows]
        if not all(b < a for a, b in zip(sizes, sizes[1:])):
            raise ValueError("Actual discretization sizes did not decrease; no convergence order can be inferred.")
        if reference:
            errors = np.asarray([row["error"] for row in rows])
            if np.all(errors <= input.error_tolerance):
                status = "verified"
                message = "All independent sample errors are below tolerance; order is unresolved at this accuracy floor."
            elif np.any(errors <= np.finfo(float).eps):
                message = "Mixed zero and nonzero errors prevent a stable order estimate."
            else:
                observed_order = float(np.polyfit(np.log(sizes), np.log(errors), 1)[0])
                monotone = np.all(np.diff(errors) <= input.error_tolerance)
                passed = monotone and abs(observed_order - input.expected_order) <= input.rate_tolerance
                status = "verified" if passed else "failed"
                message = "Observed refinement order matches the declared method." if passed else "Actual field errors do not satisfy the declared convergence requirement."
    except Exception as exc:
        message = str(exc)
    path = directory / "report.json"
    write_json(path, {
        "schema_version": "physicsos.runtime_convergence.v1", "status": status, "passes": status == "verified",
        "axis": input.axis, "rows": rows, "runs": [item.model_dump(mode="json") for item in runs],
        "expected_order": input.expected_order, "observed_order": observed_order,
        "rate_tolerance": input.rate_tolerance, "error_tolerance": input.error_tolerance,
        "max_samples": input.max_samples,
        "message": message, "norm": "weighted RMS at common physical-domain probes",
        "generated_from_actual_runs": True,
    })
    return ConvergenceStudyOutput(status=status, runs=runs, report=artifact(path, "runtime_convergence", workspace))
