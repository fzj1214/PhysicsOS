from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
from uuid import uuid4

import numpy as np

from physicsos.paths import resolve_workspace_path
from physicsos.schemas.case_runtime import CaseRun, ExecuteCaseInput, ExecuteCaseOutput, PreparedDomain
from physicsos.schemas.common import Provenance, RuntimeStats
from physicsos.schemas.solver import FieldDataRef, SolverResult
from physicsos.runtime.artifacts import artifact, checked_path, digest, write_json


def load_domain(reference, workspace: Path) -> PreparedDomain:
    domain = PreparedDomain.model_validate_json(checked_path(reference, workspace).read_text(encoding="utf-8"))
    if domain.status != "ready":
        raise ValueError(f"Domain is {domain.status}: {'; '.join(domain.required_actions)}")
    for item in domain.artifacts.values():
        checked_path(item, workspace)
    return domain


def run_process(command: list[str], cwd: Path, timeout: int, env: dict | None = None) -> dict:
    started = perf_counter()
    try:
        completed = subprocess.run(command, cwd=cwd, env=env, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=timeout, check=False)
        result = {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr, "timed_out": False}
    except subprocess.TimeoutExpired as exc:
        decode = lambda value: value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""
        result = {"returncode": 124, "stdout": decode(exc.stdout), "stderr": decode(exc.stderr), "timed_out": True}
    result["wall_time_seconds"] = perf_counter() - started
    return result


def execute_case(input: ExecuteCaseInput, workspace: Path) -> ExecuteCaseOutput:
    run_id = uuid4().hex
    directory = workspace / "cases" / input.case_id / "runs" / run_id
    working_root = directory / "workspace"
    working_case = working_root / "cases" / input.case_id
    output = working_case / "taps"
    output.mkdir(parents=True, exist_ok=False)
    context_path = directory / "context.json"
    log_path = directory / "execution_log.json"
    errors = []
    hashes = {}
    references = {}
    domain_id = ""
    kernel_hash = ""
    process = {"returncode": 1, "wall_time_seconds": 0, "stderr": "Execution was not started."}
    try:
        domain = load_domain(input.prepared_domain, workspace)
        domain_id = domain.id
        if domain.case_id != input.case_id:
            raise ValueError("Prepared domain belongs to a different case.")
        kernel_source = resolve_workspace_path(input.kernel_uri or f"cases/{input.case_id}/taps/kernel.py", workspace=workspace)
        if not kernel_source.is_file():
            raise ValueError("The case-local kernel has not been implemented.")
        original_case = workspace / "cases" / input.case_id
        # Copy implementation/resources, never old solver results or caches.
        for section in ("problem", "context", "references", "materials", "pseudopotentials"):
            source = original_case / section
            if source.is_dir():
                shutil.copytree(source, working_case / section)
        if kernel_source.parent.is_dir():
            for source in kernel_source.parent.glob("*.py"):
                shutil.copyfile(source, output / source.name)
        kernel = output / "kernel.py"
        shutil.copyfile(kernel_source, kernel)
        kernel_hash = digest(kernel)
        paths = {}
        for name, reference in domain.artifacts.items():
            source = checked_path(reference, workspace)
            relative = source.relative_to(workspace)
            target = working_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            paths[name] = str(target)
        for name, path in paths.items():
            source = Path(path)
            # Familiar geometry filenames remain usable by existing kernels.
            target = working_case / "geometry" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        geometry_context = domain.geometry.model_dump(mode="json")
        if "source_asset" in paths:
            geometry_context["source"]["uri"] = paths["source_asset"]
        config = {
            **input.controls, "case_id": input.case_id, "run_id": run_id,
            "case_dir": str(working_case), "output_dir": str(output),
            "controls": input.controls,
            "domain": {"id": domain.id, "requirements": domain.requirements.model_dump(mode="json"), "geometry": geometry_context, "artifacts": paths, "checked_cell_types": domain.mesh.quality.checked_cell_types if domain.mesh else []},
            "domain_artifacts": paths,
        }
        write_json(context_path, config)
        for path in working_root.rglob("*"):
            if path.is_file():
                hashes[str(path)] = digest(path)
        hashes[str(context_path)] = digest(context_path)
        from physicsos.tools.taps_tools import _ensure_workspace_path_shim
        shim = _ensure_workspace_path_shim(working_case)
        env = {**os.environ, "PHYSICSOS_WORKSPACE": str(working_root), "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        env["PYTHONPATH"] = str(shim) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        response = directory / "worker_response.json"
        command = [sys.executable, str(Path(__file__).with_name("kernel_worker.py")), "--kernel", str(kernel), "--config", str(context_path), "--response", str(response)]
        process = run_process(command, working_case, input.timeout_seconds, env)
        if not response.is_file():
            raise ValueError(f"Kernel execution failed: {process['stderr'][-2000:]}")
        returned = json.loads(response.read_text())
        if not returned.get("ok"):
            raise ValueError(returned.get("error", "Kernel entrypoint failed."))
        if process["returncode"] != 0:
            raise ValueError(f"Kernel process exited {process['returncode']}.")
        for path, checksum in hashes.items():
            if not Path(path).is_file() or digest(Path(path)) != checksum:
                raise ValueError(f"Kernel modified a versioned input: {path}")
        load_domain(input.prepared_domain, workspace)
        solution = output / "solution.npy"
        values = np.load(solution, allow_pickle=False)
        if values.dtype.kind not in "fci" or not values.size or not np.isfinite(values).all():
            raise ValueError("Solution must contain finite numeric field values.")
        if domain.requirements.representation == "mesh":
            with np.load(paths["mesh_arrays"], allow_pickle=False) as mesh:
                if values.shape[0] != len(mesh["points"]):
                    raise ValueError("Solution node count does not match the prepared mesh.")
        elif list(values.shape[:3]) != domain.requirements.grid_resolution:
            raise ValueError("Solution shape does not match the prepared background grid.")
        for name, filename in [("solution", "solution.npy"), ("residual_history", "residual_history.json"), ("runtime_metadata", "runtime_metadata.json")]:
            path = output / filename
            if not path.is_file():
                raise ValueError(f"Required kernel output is missing: {filename}")
            if path.suffix == ".json":
                json.loads(path.read_text(encoding="utf-8"))
            references[name] = artifact(path, name, workspace)
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    write_json(log_path, {"run_id": run_id, "domain_id": domain_id, "kernel_sha256": kernel_hash, "process": process, "errors": errors})
    references["execution_log"] = artifact(log_path, "case_execution_log", workspace)
    if context_path.exists():
        references["context"] = artifact(context_path, "runtime_context", workspace)
    result = SolverResult(
        id=f"result:{input.case_id}:{run_id}", problem_id=input.case_id, backend="case_local_kernel",
        status="failed" if errors else "success", artifacts=list(references.values()),
        fields=[FieldDataRef(field=input.field_name, uri=references["solution"].uri, format="npy", location="node")] if "solution" in references else [],
        runtime=RuntimeStats(wall_time_seconds=process["wall_time_seconds"]),
        provenance=Provenance(created_by="CaseRuntime", metadata={"run_id": run_id, "domain_id": domain_id, "kernel_sha256": kernel_hash}),
    )
    run = CaseRun(id=run_id, case_id=input.case_id, domain_id=domain_id, domain_manifest=input.prepared_domain, kernel_sha256=kernel_hash, controls=input.controls, field_name=input.field_name, result=result, artifacts=references, input_hashes=hashes, working_case_dir=str(working_case), errors=errors)
    path = directory / "manifest.json"
    write_json(path, run)
    return ExecuteCaseOutput(run=run, manifest=artifact(path, "case_run", workspace))
