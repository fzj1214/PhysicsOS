from __future__ import annotations

from pydantic import Field

from physicsos.schemas.common import ArtifactRef, Provenance, StrictBaseModel, TargetSpec, UserIntent
from physicsos.schemas.geometry import GeometrySource, GeometrySpec
from physicsos.schemas.operators import FieldSpec, OperatorSpec
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
    """Convert user intent and artifacts into the minimal PhysicsProblem IR."""
    geometry = input.geometry
    if geometry is None:
        geometry = GeometrySpec(
            id="geometry:text-placeholder",
            source=GeometrySource(kind="text"),
            dimension=3,
        )

    problem = PhysicsProblem(
        id="problem:draft",
        user_intent=UserIntent(raw_request=input.user_request),
        domain="custom",
        geometry=geometry,
        fields=[FieldSpec(name="u", kind="scalar")],
        operators=[
            OperatorSpec(
                id="operator:placeholder",
                name="unspecified_operator",
                domain="custom",
                equation_class="unspecified",
                form="strong",
                fields_out=["u"],
                assumptions=["placeholder problem; requires physics specialization before solve"],
            )
        ],
        materials=[],
        boundary_conditions=[],
        targets=[TargetSpec(name="default_observation")],
        provenance=Provenance(created_by="build_physics_problem", source="user_request"),
    )
    return BuildPhysicsProblemOutput(
        problem=problem,
        missing_inputs=["operator", "materials", "boundary_conditions"],
        assumptions=input.assumptions + ["Draft placeholder created; not solver-ready."],
    )


class ValidatePhysicsProblemInput(StrictBaseModel):
    problem: PhysicsProblem


class ValidatePhysicsProblemOutput(StrictBaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


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
        warnings.append("No mesh specified; geometry-mesh-agent must generate one or choose a mesh-free backend.")
    return ValidatePhysicsProblemOutput(valid=not errors, errors=errors, warnings=warnings)


build_physics_problem.input_model = BuildPhysicsProblemInput
build_physics_problem.output_model = BuildPhysicsProblemOutput
build_physics_problem.side_effects = "none"

validate_physics_problem.input_model = ValidatePhysicsProblemInput
validate_physics_problem.output_model = ValidatePhysicsProblemOutput
validate_physics_problem.side_effects = "none"

