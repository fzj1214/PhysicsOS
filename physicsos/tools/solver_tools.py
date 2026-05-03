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


class PrepareOpenFOAMRunnerManifestInput(StrictBaseModel):
    case_bundle: ArtifactRef
    solver: Literal["simpleFoam", "icoFoam"] = "simpleFoam"
    budget: ComputeBudget = Field(default_factory=ComputeBudget)
    service_base_url: str | None = None


class PrepareOpenFOAMRunnerManifestOutput(StrictBaseModel):
    runner_manifest: ArtifactRef
    warnings: list[str] = Field(default_factory=list)


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


def _load_taps_case_bundle(artifact: ArtifactRef) -> dict:
    if artifact.kind != "taps_backend_case_bundle":
        raise ValueError(f"Expected taps_backend_case_bundle artifact, got {artifact.kind}.")
    try:
        payload = json.loads(Path(artifact.uri).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read TAPS backend case bundle: {artifact.uri}") from exc
    if payload.get("schema_version") != "physicsos.taps_backend_case_bundle.v1":
        raise ValueError(f"Unsupported TAPS backend case bundle schema: {payload.get('schema_version')}")
    return payload


def _openfoam_file(path: str, content: str) -> dict[str, str]:
    return {"path": path, "content": content}


def _foam_header(class_name: str, object_name: str) -> str:
    return (
        "FoamFile\n"
        "{\n"
        "    version     2.0;\n"
        "    format      ascii;\n"
        f"    class       {class_name};\n"
        f"    object      {object_name};\n"
        "}\n"
    )


def _of_scalar(value: object, default: float) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _bundle_coefficient(bundle: dict, names: set[str], default: float) -> float:
    normalized = {name.lower() for name in names}
    plan = bundle.get("backend_preparation_plan") if isinstance(bundle.get("backend_preparation_plan"), dict) else {}
    coefficient_map = plan.get("coefficient_map") if isinstance(plan.get("coefficient_map"), list) else bundle.get("coefficient_binding", [])
    if isinstance(coefficient_map, list):
        for item in coefficient_map:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").lower()
            if name in normalized and "value" in item:
                return _of_scalar(item.get("value"), default)
    return default


def _bundle_boundary_roles(bundle: dict) -> set[str]:
    boundaries = bundle.get("boundary_binding", [])
    roles: set[str] = set()
    if isinstance(boundaries, list):
        for item in boundaries:
            if isinstance(item, dict):
                roles.add(str(item.get("kind") or item.get("backend_tag") or "").lower())
                roles.add(str(item.get("region_id") or "").lower())
    return roles


def _openfoam_channel_case_files(*, nu: float, solver: str) -> list[dict[str, str]]:
    end_time = "50" if solver == "simpleFoam" else "0.5"
    delta_t = "1" if solver == "simpleFoam" else "0.005"
    return [
        _openfoam_file(
            "system/blockMeshDict",
            _foam_header("dictionary", "blockMeshDict")
            + """
convertToMeters 1;
vertices
(
    (0 0 0)
    (1 0 0)
    (1 1 0)
    (0 1 0)
    (0 0 0.05)
    (1 0 0.05)
    (1 1 0.05)
    (0 1 0.05)
);
blocks
(
    hex (0 1 2 3 4 5 6 7) (20 20 1) simpleGrading (1 1 1)
);
edges ();
boundary
(
    inlet  { type patch; faces ((0 4 7 3)); }
    outlet { type patch; faces ((1 2 6 5)); }
    walls  { type wall; faces ((0 1 5 4) (3 7 6 2)); }
    frontAndBack { type empty; faces ((0 3 2 1) (4 5 6 7)); }
);
mergePatchPairs ();
""".lstrip(),
        ),
        _openfoam_file(
            "0/U",
            _foam_header("volVectorField", "U")
            + """
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);
boundaryField
{
    inlet        { type fixedValue; value uniform (1 0 0); }
    outlet       { type zeroGradient; }
    walls        { type noSlip; }
    frontAndBack { type empty; }
}
""".lstrip(),
        ),
        _openfoam_file(
            "0/p",
            _foam_header("volScalarField", "p")
            + """
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    inlet        { type zeroGradient; }
    outlet       { type fixedValue; value uniform 0; }
    walls        { type zeroGradient; }
    frontAndBack { type empty; }
}
""".lstrip(),
        ),
        _openfoam_file(
            "constant/transportProperties",
            _foam_header("dictionary", "transportProperties")
            + f"""
transportModel  Newtonian;
nu              [0 2 -1 0 0 0 0] {nu:.12g};
""".lstrip(),
        ),
        _openfoam_file(
            "constant/turbulenceProperties",
            _foam_header("dictionary", "turbulenceProperties")
            + """
simulationType  laminar;
""".lstrip(),
        ),
        _openfoam_file(
            "system/controlDict",
            _foam_header("dictionary", "controlDict")
            + f"""
application     {solver};
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {end_time};
deltaT          {delta_t};
writeControl    timeStep;
writeInterval   1;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
""".lstrip(),
        ),
        _openfoam_file(
            "system/fvSchemes",
            _foam_header("dictionary", "fvSchemes")
            + """
ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; }
divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes    { default corrected; }
""".lstrip(),
        ),
        _openfoam_file(
            "system/fvSolution",
            _foam_header("dictionary", "fvSolution")
            + """
solvers
{
    p { solver PCG; preconditioner DIC; tolerance 1e-06; relTol 0.05; }
    U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-05; relTol 0.1; }
}
SIMPLE
{
    nNonOrthogonalCorrectors 0;
}
relaxationFactors
{
    fields { p 0.3; }
    equations { U 0.7; }
}
""".lstrip(),
        ),
    ]


def _validate_openfoam_case_files(case_files: list[dict[str, str]]) -> None:
    safe_path = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._/-")
    required = {
        "system/blockMeshDict",
        "0/U",
        "0/p",
        "constant/transportProperties",
        "constant/turbulenceProperties",
        "system/controlDict",
        "system/fvSchemes",
        "system/fvSolution",
    }
    paths = {file.get("path", "") for file in case_files}
    missing = sorted(required - paths)
    if missing:
        raise ValueError(f"OpenFOAM case_files missing required files: {missing}")
    for file in case_files:
        path = file.get("path", "")
        if not path or path.startswith("/") or ".." in path or any(char not in safe_path for char in path):
            raise ValueError(f"Unsafe OpenFOAM case file path: {path}")
        if not isinstance(file.get("content"), str):
            raise ValueError(f"OpenFOAM case file content must be a string: {path}")


def prepare_openfoam_runner_manifest(input: PrepareOpenFOAMRunnerManifestInput) -> PrepareOpenFOAMRunnerManifestOutput:
    """Convert a reviewed TAPS backend case bundle into foamvm's OpenFOAM runner manifest."""
    bundle = _load_taps_case_bundle(input.case_bundle)
    if str(bundle.get("backend", "")).lower() != "openfoam":
        raise ValueError("OpenFOAM runner manifest adapter requires bundle.backend=openfoam.")
    approval_gate = bundle.get("approval_gate") if isinstance(bundle.get("approval_gate"), dict) else {}
    if approval_gate.get("execute_external_solver") is not False:
        raise ValueError("Case bundle approval gate must not pre-enable external solver execution.")
    mesh_export = bundle.get("mesh_export") if isinstance(bundle.get("mesh_export"), dict) else {}
    if mesh_export.get("required") and mesh_export.get("provided"):
        raise NotImplementedError("polyMesh/gmshToFoam runner manifests are not enabled yet; first adapter supports blockMesh-only channel cases.")
    roles = _bundle_boundary_roles(bundle)
    if not ({"inlet", "outlet"} <= roles and ("wall" in roles or "walls" in roles)):
        raise ValueError("OpenFOAM first adapter requires inlet, outlet, and wall boundary roles.")
    mu = _bundle_coefficient(bundle, {"dynamic_viscosity", "mu", "viscosity"}, 1.0)
    rho = _bundle_coefficient(bundle, {"density", "rho"}, 1.0)
    nu = mu / rho if abs(rho) > 1e-12 else mu
    case_files = _openfoam_channel_case_files(nu=nu, solver=input.solver)
    _validate_openfoam_case_files(case_files)
    problem_id = str(bundle.get("problem_id") or "problem:openfoam")
    workspace = _solver_workspace(problem_id, "openfoam")
    manifest = {
        "schema_version": "physicsos.full_solver_job.v1",
        "problem_id": problem_id,
        "backend": "openfoam",
        "backend_command": input.solver,
        "service": {
            "base_url": input.service_base_url,
            "mode": "prepare_only",
            "requires_approval_token": True,
        },
        "budget": input.budget.model_dump(mode="json"),
        "inputs": {
            "source_case_bundle": input.case_bundle.uri,
            "source_schema_version": bundle.get("schema_version"),
            "adapter": "physicsos.openfoam.blockmesh_channel.v1",
        },
        "openfoam": {
            "solver": input.solver,
            "mesh_mode": "blockMesh",
            "case_files": case_files,
        },
        "execution_policy": {
            "sandboxed_workspace": str(workspace),
            "network_access": "runner_service_only",
            "external_process_execution": "disabled_until_approved",
            "artifact_collection": ["log.blockMesh", f"log.{input.solver}", "VTK.tar.gz", "native_outputs"],
        },
    }
    path = workspace / "openfoam_runner_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    artifact = ArtifactRef(
        uri=str(path),
        kind="full_solver_runner_manifest",
        format="json",
        description="foamvm-compatible OpenFOAM full-solver runner manifest.",
    )
    return PrepareOpenFOAMRunnerManifestOutput(runner_manifest=artifact)


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
    (prepare_openfoam_runner_manifest, PrepareOpenFOAMRunnerManifestInput, PrepareOpenFOAMRunnerManifestOutput, False),
    (submit_full_solver_job, SubmitFullSolverJobInput, SubmitFullSolverJobOutput, True),
    (run_full_solver, RunFullSolverInput, RunFullSolverOutput, True),
    (run_hybrid_solver, RunHybridSolverInput, RunHybridSolverOutput, False),
]:
    _tool.input_model = _input
    _tool.output_model = _output
    _tool.side_effects = "may create solver artifacts"
    _tool.requires_approval = _approval
