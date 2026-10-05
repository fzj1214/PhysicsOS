"""Check actual fields; comparisons never consume solver-reported errors."""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import sys
from uuid import uuid4

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from physicsos.paths import resolve_workspace_path
from physicsos.schemas.case_runtime import CaseRun, VerifyCaseInput, VerifyCaseOutput
from physicsos.verification.base import AggregateVerificationReport, VerificationResult, VerificationStatus
from physicsos.runtime.artifacts import artifact, checked_path, digest, write_json
from physicsos.runtime.execution import load_domain, run_process


def load_run(reference, workspace: Path) -> CaseRun:
    run = CaseRun.model_validate_json(checked_path(reference, workspace).read_text(encoding="utf-8"))
    for item in run.artifacts.values():
        checked_path(item, workspace)
    for value, checksum in run.input_hashes.items():
        path = Path(value)
        if not path.is_file() or digest(path) != checksum:
            raise ValueError("A run input changed after execution.")
    return run


def solution_context(run: CaseRun, workspace: Path):
    load_domain(run.domain_manifest, workspace)
    config = json.loads(checked_path(run.artifacts["context"], workspace).read_text(encoding="utf-8"))
    values = np.load(checked_path(run.artifacts["solution"], workspace), allow_pickle=False)
    if not np.isfinite(values).all():
        raise ValueError("Solution contains nonfinite values.")
    return config, values


def _cells(config: dict):
    with np.load(config["domain_artifacts"]["mesh_arrays"], allow_pickle=False) as raw:
        points = raw["points"]
        dimension = config["domain"]["requirements"]["dimension"]
        cell_type = {1: "line", 2: "triangle", 3: "tetra"}[dimension]
        if config["domain"].get("checked_cell_types") != [cell_type]:
            raise ValueError("This sampler requires a uniformly supported linear simplex field; no partial mixed-element verification is allowed.")
        if cell_type not in raw:
            raise ValueError(f"Sampling provider for this element type is unavailable; expected linear {cell_type}.")
        cells = raw[cell_type]
    return points, cells, dimension


def probes(run: CaseRun, workspace: Path, maximum: int):
    config, _ = solution_context(run, workspace)
    if config["domain"]["requirements"]["representation"] == "mesh":
        points, cells, dimension = _cells(config)
        selected = np.linspace(0, len(cells) - 1, min(maximum, len(cells)), dtype=int)
        vertices = points[cells[selected]]
        # Irrational barycentric weights avoid coarse centroids becoming
        # fine-mesh nodes under common integer refinement ratios.
        barycentric = np.sqrt(np.arange(1, dimension + 2, dtype=float))
        barycentric /= barycentric.sum()
        centers = np.einsum("i,nij->nj", barycentric, vertices)
        if dimension == 3:
            weights = np.abs(np.linalg.det(vertices[:, 1:] - vertices[:, :1])) / 6
        elif dimension == 2:
            weights = np.linalg.norm(np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]), axis=1) / 2
        else:
            weights = np.linalg.norm(vertices[:, 1] - vertices[:, 0], axis=1)
        return centers, weights
    grid = json.loads(Path(config["domain_artifacts"]["background_grid"]).read_text())
    axes = [np.asarray(grid["axes"][name]) for name in "xyz"]
    occupancy = np.load(config["domain_artifacts"]["occupancy"], allow_pickle=False)
    active = np.ones(tuple(size - 1 for size in occupancy.shape), dtype=bool)
    import itertools
    for corner in itertools.product((0, 1), repeat=3):
        active &= occupancy[tuple(slice(offset, offset + size - 1) for offset, size in zip(corner, occupancy.shape))] > 0
    candidates = np.argwhere(active)
    if not len(candidates):
        raise ValueError("No complete physical-domain grid cells are available for comparison.")
    selected = candidates[np.linspace(0, len(candidates) - 1, min(maximum, len(candidates)), dtype=int)]
    # One-third-cell probes stay off nodes under dyadic refinement; otherwise
    # coarse midpoints become fine nodes and can hide interpolation error.
    points = np.column_stack([((2 * axes[i][:-1] + axes[i][1:]) / 3)[selected[:, i]] for i in range(3)])
    import trimesh
    from physicsos.backends.surface_mesh import read_mesh_surface
    vertices, faces = read_mesh_surface(Path(config["domain_artifacts"]["source_asset"]))
    membership = trimesh.Trimesh(vertices=vertices, faces=faces, process=False).contains(points)
    if config["domain"]["requirements"]["domain_side"] == "exterior":
        membership = ~membership
    points = points[membership]
    if not len(points):
        raise ValueError("Grid probes do not lie in the actual physical domain.")
    return points, np.full(len(points), np.prod([np.diff(axis).mean() for axis in axes]))


def sample_solution(run: CaseRun, workspace: Path, query: np.ndarray) -> np.ndarray:
    config, values = solution_context(run, workspace)
    if config["domain"]["requirements"]["representation"] == "background_grid":
        import trimesh
        from physicsos.backends.surface_mesh import read_mesh_surface
        vertices, faces = read_mesh_surface(Path(config["domain_artifacts"]["source_asset"]))
        inside = trimesh.Trimesh(vertices=vertices, faces=faces, process=False).contains(query)
        if config["domain"]["requirements"]["domain_side"] == "exterior":
            inside = ~inside
        if not inside.all():
            raise ValueError("A comparison point is outside the actual physical grid domain.")
        grid = json.loads(Path(config["domain_artifacts"]["background_grid"]).read_text())
        axes = [np.asarray(grid["axes"][name]) for name in "xyz"]
        return RegularGridInterpolator(axes, values, bounds_error=True)(query)
    points, cells, dimension = _cells(config)
    vertices = points[cells]
    lower, upper = vertices.min(axis=1), vertices.max(axis=1)
    tolerance = max(float(np.ptp(points, axis=0).max()) * 1e-9, np.finfo(float).eps)
    sampled = []
    for point in query:
        candidates = np.flatnonzero(((point >= lower - tolerance) & (point <= upper + tolerance)).all(axis=1))
        found = None
        for cell_id in candidates:
            basis = (vertices[cell_id, 1:] - vertices[cell_id, 0]).T
            barycentric, _, _, _ = np.linalg.lstsq(basis, point - vertices[cell_id, 0], rcond=None)
            weights = np.concatenate([[1 - barycentric.sum()], barycentric])
            if np.linalg.norm(basis @ barycentric - (point - vertices[cell_id, 0])) <= tolerance and weights.min() >= -tolerance:
                found = np.tensordot(weights, values[cells[cell_id]], axes=1)
                break
        if found is None:
            raise ValueError("A comparison point is outside the actual mesh; no triangulation across holes is allowed.")
        sampled.append(found)
    return np.asarray(sampled)


def evaluate_reference(source: Path, function: str, points: np.ndarray, config: dict, directory: Path, timeout: int) -> np.ndarray:
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = directory / "reference.py"
    for helper in source.parent.glob("*.py"):
        shutil.copyfile(helper, directory / helper.name)
    shutil.copyfile(source, snapshot)
    hashes = {str(path): digest(path) for path in directory.glob("*.py")}
    np.save(directory / "points.npy", points, allow_pickle=False)
    write_json(directory / "context.json", config)
    command = [sys.executable, str(Path(__file__).with_name("kernel_worker.py")), "--kernel", str(snapshot), "--entrypoint", function, "--config", str(directory / "context.json"), "--points", str(directory / "points.npy"), "--values", str(directory / "values.npy"), "--response", str(directory / "response.json")]
    process = run_process(command, directory, timeout)
    write_json(directory / "execution_log.json", process)
    response_path = directory / "response.json"
    response = json.loads(response_path.read_text()) if response_path.exists() else {}
    if process["returncode"] != 0 or not response.get("ok"):
        raise ValueError(f"Independent reference evaluation failed: {response.get('error') or process['stderr'][-1000:]}")
    if any(not Path(path).is_file() or digest(Path(path)) != value for path, value in hashes.items()):
        raise ValueError("Reference code changed during evaluation.")
    values = np.load(directory / "values.npy", allow_pickle=False)
    if len(values) != len(points) or not np.isfinite(values).all():
        raise ValueError("Independent reference has invalid shape or values.")
    return values


def comparison_error(actual: np.ndarray, expected: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    if actual.shape != expected.shape:
        raise ValueError("Reference and solution field shapes differ.")
    error = np.abs(actual - expected).reshape(len(weights), -1)
    scale = np.abs(expected).reshape(len(weights), -1)
    rms = math.sqrt(float(np.average((error * error).mean(axis=1), weights=weights)))
    reference_rms = math.sqrt(float(np.average((scale * scale).mean(axis=1), weights=weights)))
    return rms, reference_rms


def verify_case(input: VerifyCaseInput, workspace: Path) -> VerifyCaseOutput:
    manifest_path = resolve_workspace_path(input.run_manifest.uri, workspace=workspace)
    run = CaseRun.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    evidence_error = None
    try:
        load_run(input.run_manifest, workspace)
    except (OSError, ValueError, KeyError) as exc:
        evidence_error = str(exc)
    path = manifest_path.parent / "verification" / uuid4().hex / "report.json"
    metrics = {}
    details = {"run_id": run.id, "domain_id": run.domain_id, "kernel_sha256": run.kernel_sha256}
    details.update(reference_function=input.reference_function, field_name=run.field_name, max_samples=input.max_samples)
    status = VerificationStatus.UNCERTAIN
    message = "An independent reference is required to verify field accuracy."
    if evidence_error:
        status, message = VerificationStatus.FAILED, "Run evidence changed or is missing: " + evidence_error
    elif run.result.status != "success":
        status, message = VerificationStatus.FAILED, "; ".join(run.errors) or "Kernel execution failed."
    elif input.reference_uri:
        try:
            source = resolve_workspace_path(input.reference_uri, workspace=workspace)
            reference_hash = digest(source)
            if reference_hash == run.kernel_sha256:
                raise ValueError("The solver kernel cannot serve as its own independent reference.")
            config, _ = solution_context(run, workspace)
            if input.comparison_probes:
                with np.load(checked_path(input.comparison_probes, workspace), allow_pickle=False) as supplied:
                    points, weights = supplied["points"], supplied["weights"]
                if points.ndim != 2 or points.shape[1] != 3 or weights.shape != (len(points),) or not 1 <= len(points) <= input.max_samples or not np.isfinite(points).all() or not np.isfinite(weights).all() or not (weights > 0).all():
                    raise ValueError("Comparison probes require finite (N, 3) points and positive aligned weights within the sampling budget.")
                details["comparison_probes"] = input.comparison_probes.model_dump(mode="json")
            else:
                points, weights = probes(run, workspace, input.max_samples)
            expected = evaluate_reference(source, input.reference_function, points, config, path.parent / "reference", input.timeout_seconds)
            actual = sample_solution(run, workspace, points)
            load_run(input.run_manifest, workspace)
            if input.comparison_probes:
                checked_path(input.comparison_probes, workspace)
            error, scale = comparison_error(actual, expected, weights)
            metrics = {"weighted_sample_rms_error": error, "reference_rms": scale, "relative_sample_error": error / max(scale, input.absolute_tolerance), "samples": float(len(points))}
            details.update(reference_sha256=reference_hash, norm="weighted physical-domain probe RMS, not an exact global L2 integral", absolute_tolerance=input.absolute_tolerance, relative_tolerance=input.relative_tolerance)
            passed = error <= input.absolute_tolerance or error <= input.relative_tolerance * scale
            status = VerificationStatus.VERIFIED if passed else VerificationStatus.FAILED
            message = "Independent field comparison passed." if passed else "Field error exceeds the configured tolerance."
        except Exception as exc:
            message = str(exc)
            details["missing_evidence"] = message
    check = VerificationResult(verifier_name="IndependentFieldReference", status=status, message=message, metrics=metrics, details=details)
    report = AggregateVerificationReport(problem_id=run.case_id, result_id=run.result.id, overall_status=status, individual_results={check.verifier_name: check}, passed_checks=int(status == VerificationStatus.VERIFIED), failed_checks=int(status == VerificationStatus.FAILED), uncertain_checks=int(status == VerificationStatus.UNCERTAIN), failure_mode=("runtime_error" if run.result.status != "success" or evidence_error else "high_error") if status == VerificationStatus.FAILED else None)
    write_json(path, report)
    return VerifyCaseOutput(report=report, artifact=artifact(path, "runtime_verification", workspace))
