from __future__ import annotations

from pydantic import Field

from physicsos.schemas.boundary import BoundaryConditionSpec, InitialConditionSpec
from physicsos.schemas.common import ArtifactRef, Provenance, StrictBaseModel, TargetSpec, UserIntent
from physicsos.schemas.common import ParameterSpec
from physicsos.schemas.geometry import GeometrySource, GeometrySpec
from physicsos.schemas.materials import MaterialProperty, MaterialSpec
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
    request_lower = input.user_request.lower()
    geometry = input.geometry
    if geometry is None:
        geometry = GeometrySpec(
            id="geometry:text-placeholder",
            source=GeometrySource(kind="text"),
            dimension=1 if any(token in request_lower for token in ["1d", "1-d", "一维"]) else 3,
        )

    heat_terms = ["heat", "thermal", "diffusion", "热传导", "导热", "传热", "稳态热", "steady heat"]
    steady_terms = ["steady", "stationary", "稳态", "定常"]
    if any(term in request_lower for term in heat_terms):
        is_steady = any(term in request_lower for term in steady_terms)
        operator_name = "Steady heat conduction" if is_steady else "Transient heat equation"
        problem = PhysicsProblem(
            id="problem:draft-thermal-1d" if geometry.dimension == 1 else "problem:draft-thermal",
            user_intent=UserIntent(raw_request=input.user_request),
            domain="thermal",
            geometry=geometry,
            fields=[FieldSpec(name="T", kind="scalar", units="K", location="node")],
            operators=[
                OperatorSpec(
                    id="operator:heat",
                    name=operator_name,
                    domain="thermal",
                    equation_class="heat",
                    form="weak",
                    fields_in=["T", "k"],
                    fields_out=["T"],
                    differential_terms=[
                        {
                            "expression": "-div(k grad(T)) = q" if is_steady else "dT/dt - div(alpha grad(T)) = q",
                            "order": 2,
                            "fields": ["T"],
                        }
                    ],
                    assumptions=[
                        "1D generated geometry inferred from natural language" if geometry.dimension == 1 else "Generated geometry inferred from natural language",
                        "Default Dirichlet temperatures added so the typed workflow can run; user-provided values should override them.",
                    ],
                )
            ],
            materials=[
                MaterialSpec(
                    id="material:generic-thermal",
                    name="Generic thermal solid",
                    phase="solid",
                    properties=[
                        MaterialProperty(name="thermal_conductivity", value=1.0, units="W/(m*K)"),
                        MaterialProperty(name="thermal_diffusivity", value=0.05, units="m^2/s"),
                    ],
                )
            ],
            boundary_conditions=[
                BoundaryConditionSpec(id="bc:left", region_id="x=0", field="T", kind="dirichlet", value=300.0, units="K", source="inferred"),
                BoundaryConditionSpec(id="bc:right", region_id="x=L", field="T", kind="dirichlet", value=350.0, units="K", source="inferred"),
            ],
            initial_conditions=[] if is_steady else [
                InitialConditionSpec(id="ic:uniform", field="T", value=300.0, units="K")
            ],
            parameters=[
                ParameterSpec(name="k", value=1.0, units="W/(m*K)", description="default thermal conductivity"),
            ],
            targets=[TargetSpec(name="temperature_field", field="T", objective="observe")],
            provenance=Provenance(created_by="build_physics_problem", source="user_request"),
        )
        assumptions = input.assumptions + [
            "Built a typed thermal conduction problem from natural language.",
            "Default material and boundary values are inferred placeholders.",
        ]
        missing_inputs = ["geometry_details", "material_properties", "boundary_values"] if not is_steady else [
            "geometry_details",
            "material_properties",
            "boundary_values",
            "heat_source",
        ]
        return BuildPhysicsProblemOutput(problem=problem, missing_inputs=missing_inputs, assumptions=assumptions)

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
