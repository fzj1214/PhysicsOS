from __future__ import annotations

from typing import Literal

from pydantic import Field

from physicsos.schemas.common import ArtifactRef, StrictBaseModel


class TAPSAxisSpec(StrictBaseModel):
    name: str
    kind: Literal["space", "parameter", "time", "geometry"]
    min_value: float | None = None
    max_value: float | None = None
    points: int | None = None
    units: str | None = None


class TAPSBasisConfig(StrictBaseModel):
    tensor_rank: int = 8
    patch_size: int = 3
    dilation: float = 1.0
    reproducing_order: int = 2
    quadrature_order: int = 3


class TAPSNonlinearConfig(StrictBaseModel):
    method: Literal["picard", "newton", "fixed_point"] = "picard"
    max_iterations: int = 50
    tolerance: float = 1e-10
    damping: float = 0.8
    linear_reaction: float = 1.0
    cubic_reaction: float = 0.1
    coupling_strength: float = 0.25


class TAPSGeometryEncodingSpec(StrictBaseModel):
    kind: Literal[
        "sdf",
        "occupancy_mask",
        "surface_point_cloud",
        "volume_point_cloud",
        "mesh_graph",
        "boundary_graph",
        "laplacian_eigenbasis",
        "multi_resolution_grid",
        "parametric_shape_vector",
    ]
    uri: str
    resolution: list[int] | None = None
    target_backend: str | None = None
    boundary_policy: Literal["dirichlet_zero", "ignore_inactive", "embedded_boundary"] = "dirichlet_zero"


class TAPSCoefficientSpec(StrictBaseModel):
    name: str
    value: float | int | str | list[float] | dict
    role: Literal["material", "source", "boundary", "operator", "solver"] = "operator"
    units: str | None = None
    region_ids: list[str] = Field(default_factory=list)
    source: Literal["physics_problem", "knowledge_agent", "user", "template", "default"] = "physics_problem"


class TAPSBoundaryConditionSpec(StrictBaseModel):
    id: str
    region_id: str
    field: str
    kind: str
    value: float | int | str | list[float] | dict
    units: str | None = None
    source: Literal["physics_problem", "knowledge_agent", "user", "template", "default"] = "physics_problem"


class TAPSEquationTerm(StrictBaseModel):
    id: str
    role: Literal[
        "time_derivative",
        "mass",
        "diffusion",
        "advection",
        "reaction",
        "source",
        "constraint",
        "boundary",
        "interface",
        "constitutive",
        "custom",
    ]
    expression: str
    fields: list[str] = Field(default_factory=list)
    coefficients: list[str] = Field(default_factory=list)
    integration_domain: str | None = None
    assumptions: list[str] = Field(default_factory=list)


class TAPSWeakFormSpec(StrictBaseModel):
    family: str
    strong_form: str | None = None
    trial_fields: list[str] = Field(default_factory=list)
    test_functions: list[str] = Field(default_factory=list)
    terms: list[TAPSEquationTerm] = Field(default_factory=list)
    boundary_terms: list[TAPSEquationTerm] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    residual_expression: str | None = None
    source: Literal["physics_problem", "knowledge_agent", "user", "template", "hybrid"] = "physics_problem"


class TAPSCompilationPlan(StrictBaseModel):
    problem_id: str
    status: Literal["ready", "needs_knowledge", "needs_user_input", "unsupported"]
    equation_family: str
    unknown_fields: list[str] = Field(default_factory=list)
    axes: list[TAPSAxisSpec] = Field(default_factory=list)
    weak_form: TAPSWeakFormSpec | None = None
    required_knowledge_queries: list[str] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    recommended_next_action: Literal[
        "compile_taps_problem",
        "ask_knowledge_agent",
        "ask_user",
        "author_runtime_extension",
        "fallback_solver",
    ]


class TAPSRuntimeExtensionSpec(StrictBaseModel):
    id: str
    problem_id: str
    purpose: str
    language: Literal["python", "ufl", "sympy", "jax", "torch", "custom"] = "python"
    entrypoint: str
    artifact: ArtifactRef
    required_inputs: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    safety_status: Literal["draft", "requires_review", "approved_for_local_run"] = "draft"
    notes: list[str] = Field(default_factory=list)


class TAPSProblem(StrictBaseModel):
    id: str
    problem_id: str
    axes: list[TAPSAxisSpec] = Field(default_factory=list)
    operator_weak_form: str | None = None
    weak_form: TAPSWeakFormSpec | None = None
    compilation_status: Literal["compiled", "knowledge_assisted", "scaffold", "unsupported"] = "scaffold"
    basis: TAPSBasisConfig = Field(default_factory=TAPSBasisConfig)
    nonlinear: TAPSNonlinearConfig = Field(default_factory=TAPSNonlinearConfig)
    geometry_encodings: list[TAPSGeometryEncodingSpec] = Field(default_factory=list)
    coefficients: list[TAPSCoefficientSpec] = Field(default_factory=list)
    boundary_conditions: list[TAPSBoundaryConditionSpec] = Field(default_factory=list)
    slabs: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class TAPSSupportScore(StrictBaseModel):
    score: float
    supported: bool
    reasons: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class TAPSResidualReport(StrictBaseModel):
    residuals: dict[str, float] = Field(default_factory=dict)
    rank: int
    converged: bool
    recommended_action: Literal["accept", "increase_rank", "refine_axes", "split_slab", "fallback"]


class TAPSResultArtifacts(StrictBaseModel):
    factor_matrices: list[ArtifactRef] = Field(default_factory=list)
    reconstruction_metadata: ArtifactRef | None = None
    residual_history: ArtifactRef | None = None
