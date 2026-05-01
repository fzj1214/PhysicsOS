from __future__ import annotations

import json
from pathlib import Path
from typing import Literal
from urllib import request

from pydantic import Field

from physicsos.backends.catalog import DEFAULT_BACKENDS
from physicsos.backends.surrogate_runtime import route_surrogate, run_surrogate_scaffold
from physicsos.config import project_root
from physicsos.schemas.common import ArtifactRef, ComputeBudget, Provenance, StrictBaseModel
from physicsos.schemas.problem import PhysicsProblem
from physicsos.schemas.solver import HybridPolicy, PreparedSolverCase, SolverDecision, SolverPolicy, SolverResult, SupportScore


class EstimateSolverSupportInput(StrictBaseModel):
    problem: PhysicsProblem
    candidate_backends: list[str] = Field(default_factory=list)


class EstimateSolverSupportOutput(StrictBaseModel):
    scores: list[SupportScore]


def estimate_solver_support(input: EstimateSolverSupportInput) -> EstimateSolverSupportOutput:
    """Score registered open-source solver backends for a PhysicsProblem."""
    requested = set(input.candidate_backends)
    scores: list[SupportScore] = []
    for backend in DEFAULT_BACKENDS:
        if requested and backend.name not in requested:
            continue
        supported = input.problem.domain in backend.domains or "custom" in backend.domains
        scores.append(
            SupportScore(
                backend=backend.name,
                score=0.8 if supported else 0.15,
                supported=supported,
                reasons=[backend.role] if supported else [f"Backend domains {backend.domains} do not match {input.problem.domain}."],
                risks=[] if supported else ["domain_mismatch"],
            )
        )
    return EstimateSolverSupportOutput(scores=scores)


class RouteSolverBackendInput(StrictBaseModel):
    problem: PhysicsProblem
    support_scores: list[SupportScore]
    policy: SolverPolicy = Field(default_factory=SolverPolicy)


class RouteSolverBackendOutput(StrictBaseModel):
    decision: SolverDecision


def route_solver_backend(input: RouteSolverBackendInput) -> RouteSolverBackendOutput:
    """Select surrogate, full solver, hybrid, warm-start, or corrector mode."""
    supported = [score for score in input.support_scores if score.supported]
    if not supported:
        selected = "custom_python"
        reason = "No domain-specific backend matched; route to custom PDE scaffold."
    else:
        selected = max(supported, key=lambda score: score.score).backend
        reason = "Selected highest support score among open-source backends."
    mode = "full_solver" if input.policy.force_full_solver else "hybrid"
    return RouteSolverBackendOutput(
        decision=SolverDecision(
            selected_backend=selected,
            candidate_backends=input.support_scores,
            mode=mode,
            reason=reason,
        )
    )


class RunSurrogateSolverInput(StrictBaseModel):
    problem: PhysicsProblem
    backend: str
    checkpoint: str | None = None


class RunSurrogateSolverOutput(StrictBaseModel):
    result: SolverResult


def run_surrogate_solver(input: RunSurrogateSolverInput) -> RunSurrogateSolverOutput:
    """Run a neural operator or surrogate backend through the surrogate runtime."""
    decision = route_surrogate(input.problem, mode="fast_solver")
    if input.backend:
        decision = decision.model_copy(update={"selected_model_id": input.backend})
    result = run_surrogate_scaffold(input.problem, decision)
    return RunSurrogateSolverOutput(result=result)


class RunFullSolverInput(StrictBaseModel):
    problem: PhysicsProblem
    backend: str
    budget: ComputeBudget = Field(default_factory=ComputeBudget)
    approval_token: str | None = None
    service_base_url: str | None = None


class PrepareFullSolverCaseInput(StrictBaseModel):
    problem: PhysicsProblem
    backend: str
    budget: ComputeBudget = Field(default_factory=ComputeBudget)
    service_base_url: str | None = None


class PrepareFullSolverCaseOutput(StrictBaseModel):
    prepared: PreparedSolverCase
    runner_manifest: ArtifactRef
    requires_approval: bool = True


class SubmitFullSolverJobInput(StrictBaseModel):
    runner_manifest: ArtifactRef
    mode: Literal["dry_run", "http"] = "dry_run"
    approval_token: str | None = None
    service_base_url: str | None = None


class SubmitFullSolverJobOutput(StrictBaseModel):
    result: SolverResult
    runner_response: ArtifactRef
    submitted: bool = False


def _backend_info(backend_name: str):
    return next((backend for backend in DEFAULT_BACKENDS if backend.name == backend_name), None)


def _solver_workspace(problem_id: str, backend: str) -> Path:
    path = project_root() / "scratch" / problem_id.replace(":", "_") / "solver_fallback" / backend
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_full_solver_case(input: PrepareFullSolverCaseInput) -> PrepareFullSolverCaseOutput:
    """Prepare a sandboxed full-solver service manifest without executing external CLI."""
    backend = _backend_info(input.backend)
    workspace = _solver_workspace(input.problem.id, input.backend)
    manifest_path = workspace / "solver_job_manifest.json"
    manifest = {
        "schema_version": "physicsos.full_solver_job.v1",
        "problem_id": input.problem.id,
        "backend": input.backend,
        "backend_command": backend.command if backend is not None else None,
        "python_integration": backend.python_integration if backend is not None else "custom",
        "role": backend.role if backend is not None else "custom external solver",
        "service": {
            "base_url": input.service_base_url,
            "mode": "prepare_only",
            "requires_approval_token": True,
        },
        "budget": input.budget.model_dump(mode="json"),
        "inputs": {
            "domain": input.problem.domain,
            "geometry_id": input.problem.geometry.id,
            "mesh_id": input.problem.mesh.id if input.problem.mesh is not None else None,
            "operators": [operator.equation_class for operator in input.problem.operators],
            "fields": [field.name for field in input.problem.fields],
            "boundary_condition_count": len(input.problem.boundary_conditions),
            "material_count": len(input.problem.materials),
        },
        "execution_policy": {
            "sandboxed_workspace": str(workspace),
            "network_access": "runner_service_only",
            "external_process_execution": "disabled_until_approved",
            "artifact_collection": ["stdout", "stderr", "native_outputs", "converted_fields", "residual_report"],
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    artifact = ArtifactRef(
        uri=str(Path(manifest_path)),
        kind="full_solver_runner_manifest",
        format="json",
        description="Prepared sandbox/service manifest for full-solver fallback.",
    )
    prepared = PreparedSolverCase(problem_id=input.problem.id, backend=input.backend, workspace_uri=str(workspace), artifacts=[artifact])
    return PrepareFullSolverCaseOutput(prepared=prepared, runner_manifest=artifact)


def _load_manifest(artifact: ArtifactRef) -> dict:
    if artifact.kind != "full_solver_runner_manifest":
        raise ValueError(f"Expected full_solver_runner_manifest artifact, got {artifact.kind}.")
    try:
        return json.loads(Path(artifact.uri).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read solver runner manifest: {artifact.uri}") from exc


def _runner_response_path(manifest: dict, mode: str) -> Path:
    workspace = manifest.get("execution_policy", {}).get("sandboxed_workspace")
    if workspace:
        path = Path(str(workspace))
    else:
        path = _solver_workspace(str(manifest["problem_id"]), str(manifest["backend"]))
    path.mkdir(parents=True, exist_ok=True)
    safe_mode = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in mode)
    return path / f"runner_response_{safe_mode}.json"


def _write_runner_response(manifest: dict, payload: dict) -> ArtifactRef:
    path = _runner_response_path(manifest, str(payload.get("mode", "runner")))
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return ArtifactRef(uri=str(Path(path)), kind="full_solver_runner_response", format="json")


def _http_submit_job(manifest: dict, service_base_url: str, approval_token: str) -> dict:
    endpoint = service_base_url.rstrip("/") + "/api/physicsos/jobs"
    body = json.dumps(manifest).encode("utf-8")
    req = request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {approval_token}",
        },
    )
    with request.urlopen(req, timeout=30) as response:
        response_body = response.read().decode("utf-8")
    try:
        parsed = json.loads(response_body)
    except json.JSONDecodeError:
        parsed = {"raw_response": response_body}
    return {"endpoint": endpoint, "status": "submitted", "response": parsed}


def submit_full_solver_job(input: SubmitFullSolverJobInput) -> SubmitFullSolverJobOutput:
    """Submit a prepared full-solver manifest to a controlled runner adapter."""
    manifest = _load_manifest(input.runner_manifest)
    backend = str(manifest["backend"])
    problem_id = str(manifest["problem_id"])
    if input.mode == "dry_run":
        payload = {
            "mode": "dry_run",
            "status": "validated",
            "submitted": False,
            "message": "Manifest is valid; no external solver service or CLI was invoked.",
            "manifest": manifest,
        }
        response_artifact = _write_runner_response(manifest, payload)
        result = SolverResult(
            id=f"result:full:{problem_id.replace(':', '_')}:{backend}:dry_run",
            problem_id=problem_id,
            backend=backend,
            status="needs_review",
            scalar_outputs={"message": payload["message"], "runner_mode": "dry_run"},
            artifacts=[input.runner_manifest, response_artifact],
            provenance=Provenance(created_by="submit_full_solver_job", source="dry_run"),
        )
        return SubmitFullSolverJobOutput(result=result, runner_response=response_artifact, submitted=False)

    service_base_url = input.service_base_url or manifest.get("service", {}).get("base_url")
    if not service_base_url:
        raise ValueError("HTTP runner mode requires service_base_url in input or manifest.")
    if not input.approval_token:
        raise PermissionError("HTTP runner mode requires approval_token.")
    payload = _http_submit_job(manifest, service_base_url, input.approval_token)
    response_artifact = _write_runner_response(manifest, payload)
    result = SolverResult(
        id=f"result:full:{problem_id.replace(':', '_')}:{backend}:submitted",
        problem_id=problem_id,
        backend=backend,
        status="partial",
        scalar_outputs={"message": "Full solver job submitted to runner service.", "runner_mode": "http"},
        artifacts=[input.runner_manifest, response_artifact],
        provenance=Provenance(created_by="submit_full_solver_job", source="http_runner"),
    )
    return SubmitFullSolverJobOutput(result=result, runner_response=response_artifact, submitted=True)


class RunFullSolverOutput(StrictBaseModel):
    result: SolverResult


def run_full_solver(input: RunFullSolverInput) -> RunFullSolverOutput:
    """Submit a trusted open-source full solver job to an external runner service."""
    if input.approval_token is None:
        raise PermissionError("run_full_solver requires a foamvm CLI token.")
    prepared = prepare_full_solver_case(
        PrepareFullSolverCaseInput(
            problem=input.problem,
            backend=input.backend,
            budget=input.budget,
            service_base_url=input.service_base_url,
        )
    )
    result = submit_full_solver_job(
        SubmitFullSolverJobInput(
            runner_manifest=prepared.runner_manifest,
            mode="http",
            approval_token=input.approval_token,
            service_base_url=input.service_base_url,
        )
    ).result
    return RunFullSolverOutput(result=result)


class RunHybridSolverInput(StrictBaseModel):
    problem: PhysicsProblem
    surrogate_backend: str
    full_backend: str
    hybrid_policy: HybridPolicy = Field(default_factory=HybridPolicy)


class RunHybridSolverOutput(StrictBaseModel):
    result: SolverResult
    stages: list[SolverResult] = Field(default_factory=list)


def run_hybrid_solver(input: RunHybridSolverInput) -> RunHybridSolverOutput:
    """Use surrogate as fast solver, warm start, or corrector around a full solver."""
    surrogate = run_surrogate_solver(RunSurrogateSolverInput(problem=input.problem, backend=input.surrogate_backend)).result
    return RunHybridSolverOutput(result=surrogate, stages=[surrogate])


for _tool, _input, _output, _approval in [
    (estimate_solver_support, EstimateSolverSupportInput, EstimateSolverSupportOutput, False),
    (route_solver_backend, RouteSolverBackendInput, RouteSolverBackendOutput, False),
    (run_surrogate_solver, RunSurrogateSolverInput, RunSurrogateSolverOutput, False),
    (prepare_full_solver_case, PrepareFullSolverCaseInput, PrepareFullSolverCaseOutput, False),
    (submit_full_solver_job, SubmitFullSolverJobInput, SubmitFullSolverJobOutput, True),
    (run_full_solver, RunFullSolverInput, RunFullSolverOutput, True),
    (run_hybrid_solver, RunHybridSolverInput, RunHybridSolverOutput, False),
]:
    _tool.input_model = _input
    _tool.output_model = _output
    _tool.side_effects = "may create solver artifacts"
    _tool.requires_approval = _approval
