from __future__ import annotations

from pydantic import Field

from physicsos.schemas.boundary import BoundaryConditionSpec
from physicsos.schemas.common import ArtifactRef, StrictBaseModel
from physicsos.schemas.geometry import BoundaryRole, GeometrySpec
from physicsos.schemas.problem import PhysicsProblem


class BuildPhysicsProblemInput(StrictBaseModel):
    user_request: str
    geometry: GeometrySpec | None = None
    uploaded_artifacts: list[ArtifactRef] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class BuildPhysicsProblemOutput(StrictBaseModel):
    problem: PhysicsProblem | None
    missing_inputs: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


def build_physics_problem(input: BuildPhysicsProblemInput) -> BuildPhysicsProblemOutput:
    """Legacy deterministic NL builder disabled for paper-replication paths."""
    return BuildPhysicsProblemOutput(
        problem=None,
        missing_inputs=["legacy_build_physics_problem_disabled"],
        assumptions=[
            "build_physics_problem() is disabled as a deterministic natural-language builder.",
            "Use analysis-file-agent to write case-local problem_statement.md and open_questions.md.",
        ],
    )


class ValidatePhysicsProblemInput(StrictBaseModel):
    problem: PhysicsProblem


class ValidatePhysicsProblemOutput(StrictBaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CanonicalPhysicsProblemInput(StrictBaseModel):
    problem: PhysicsProblem


class CanonicalPhysicsProblemOutput(StrictBaseModel):
    problem: PhysicsProblem
    status: str = "ready"
    role_assignments: dict[str, BoundaryRole] = Field(default_factory=dict)
    missing_boundary_roles: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _canonical_boundary_role(region_id: str) -> BoundaryRole | None:
    lowered = region_id.lower().replace(" ", "").replace("-", "_")
    pieces = [piece for piece in lowered.replace(":", "_").replace("=", "_").split("_") if piece]
    if lowered in {"x=0", "x_min", "xmin", "x0", "left", "boundary:x_min", "boundary:left"} or "left" in pieces or "x0" in pieces or ("x" in pieces and "0" in pieces):
        return "x_min"
    if lowered in {"x=l", "x=1", "x_max", "xmax", "x1", "right", "boundary:x_max", "boundary:right"} or "right" in pieces or "x1" in pieces or ("x" in pieces and "1" in pieces):
        return "x_max"
    if lowered in {"y=0", "y_min", "ymin", "y0", "bottom", "boundary:y_min", "boundary:bottom"} or "bottom" in pieces or "y0" in pieces:
        return "y_min"
    if lowered in {"y=1", "y_max", "ymax", "y1", "top", "boundary:y_max", "boundary:top"} or "top" in pieces or "y1" in pieces:
        return "y_max"
    if lowered in {"z=0", "z_min", "zmin", "z0", "front", "boundary:z_min", "boundary:front"} or "front" in pieces or "z0" in pieces:
        return "z_min"
    if lowered in {"z=1", "z_max", "zmax", "z1", "back", "boundary:z_max", "boundary:back"} or "back" in pieces or "z1" in pieces:
        return "z_max"
    if "inlet" in pieces:
        return "inlet"
    if "outlet" in pieces:
        return "outlet"
    if "wall" in pieces:
        return "wall"
    if "symmetry" in pieces:
        return "symmetry"
    if "farfield" in pieces:
        return "farfield"
    return None


def _geometry_boundary_role(problem: PhysicsProblem, region_id: str) -> BoundaryRole | None:
    for boundary in problem.geometry.boundaries:
        if boundary.id == region_id and boundary.role is not None:
            return boundary.role
    return None


def _is_imported_geometry(problem: PhysicsProblem) -> bool:
    return problem.geometry.source.kind in {"cad_step", "cad_iges", "stl", "mesh_file"}


def _is_solver_critical_boundary(boundary: BoundaryConditionSpec) -> bool:
    return boundary.kind in {"dirichlet", "neumann", "robin", "wall", "inlet", "outlet", "symmetry", "farfield"}


def canonicalize_physics_problem(input: CanonicalPhysicsProblemInput) -> CanonicalPhysicsProblemOutput:
    """Assign canonical boundary roles before solver planning."""
    problem = input.problem
    assignments: dict[str, BoundaryRole] = {}
    missing: list[str] = []
    warnings: list[str] = []
    canonical_boundaries: list[BoundaryConditionSpec] = []
    for boundary in problem.boundary_conditions:
        role = boundary.boundary_role or _geometry_boundary_role(problem, boundary.region_id) or _canonical_boundary_role(boundary.region_id)
        if role is None:
            if _is_solver_critical_boundary(boundary):
                missing.append(boundary.id)
            canonical_boundaries.append(boundary)
            continue
        assignments[boundary.id] = role
        if boundary.boundary_role != role:
            source = "confirmed geometry" if _geometry_boundary_role(problem, boundary.region_id) == role else "deterministic canonicalizer"
            warnings.append(f"Assigned boundary_role={role} to {boundary.id} from {source}.")
            canonical_boundaries.append(boundary.model_copy(update={"boundary_role": role}))
        else:
            canonical_boundaries.append(boundary)
    status = "needs_user_input" if missing and _is_imported_geometry(problem) else "ready"
    if missing and not _is_imported_geometry(problem):
        warnings.append("Some solver-critical boundaries lack canonical roles; downstream executable validators may reject unsupported kernels.")
    canonical_problem = problem.model_copy(update={"boundary_conditions": canonical_boundaries})
    return CanonicalPhysicsProblemOutput(
        problem=canonical_problem,
        status=status,
        role_assignments=assignments,
        missing_boundary_roles=missing,
        warnings=warnings,
    )


def validate_physics_problem(input: ValidatePhysicsProblemInput) -> ValidatePhysicsProblemOutput:
    """Validate that a PhysicsProblem has the minimum inputs required before solving."""
    problem = input.problem
    errors: list[str] = []
    warnings: list[str] = []
    if not problem.operators:
        errors.append("PhysicsProblem.operators is required.")
    if not problem.fields:
        errors.append("PhysicsProblem.fields is required.")
    if not problem.targets:
        errors.append("PhysicsProblem.targets is required.")
    if not problem.boundary_conditions:
        errors.append("At least one boundary condition is required for continuum solvers.")
    if not problem.materials:
        warnings.append("No materials specified; only abstract/custom solvers may support this problem.")
    if problem.mesh is None:
        warnings.append("No mesh specified; geometry-embedding-agent must generate the geometry data needed by the paper-style TAPS route.")
    return ValidatePhysicsProblemOutput(valid=not errors, errors=errors, warnings=warnings)


canonicalize_physics_problem.input_model = CanonicalPhysicsProblemInput
canonicalize_physics_problem.output_model = CanonicalPhysicsProblemOutput
canonicalize_physics_problem.side_effects = "none"

validate_physics_problem.input_model = ValidatePhysicsProblemInput
validate_physics_problem.output_model = ValidatePhysicsProblemOutput
validate_physics_problem.side_effects = "none"
