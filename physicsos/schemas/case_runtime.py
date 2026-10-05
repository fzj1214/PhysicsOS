"""Contracts joining asset preparation, case-local kernels and verification."""
from __future__ import annotations

from typing import Any, Literal
import math

from pydantic import Field, model_validator

from physicsos.schemas.common import ArtifactRef, StrictBaseModel
from physicsos.schemas.geometry import BoundaryRole, GeometryMeshContract, GeometrySpec
from physicsos.schemas.geometry_repair import GeometryRepairOptions
from physicsos.schemas.mesh import MeshPolicy, MeshSpec
from physicsos.schemas.operators import PhysicsSpec
from physicsos.schemas.solver import SolverResult
from physicsos.verification.base import AggregateVerificationReport


class DomainRequirements(StrictBaseModel):
    representation: Literal["mesh", "background_grid"] = "mesh"
    dimension: Literal[1, 2, 3] = 3
    backend: str = "case_kernel"
    mesh_policy: MeshPolicy = Field(default_factory=MeshPolicy)
    grid_resolution: list[int] = Field(default_factory=lambda: [17, 17, 17])
    domain_side: Literal["interior", "exterior"] = "interior"
    bounds_min: list[float] | None = None
    bounds_max: list[float] | None = None
    whole_boundary_role: BoundaryRole | None = None
    boundary_roles: dict[str, BoundaryRole] = Field(default_factory=dict)
    required_boundary_roles: list[BoundaryRole] = Field(default_factory=list)
    required_quality_metrics: list[str] = Field(default_factory=lambda: ["min_jacobian", "max_skewness", "aspect_ratio_p95"])
    max_aspect_ratio_p95: float = Field(default=25, gt=1, allow_inf_nan=False)
    max_skewness: float = Field(default=.95, gt=0, lt=1)


class PrepareDomainInput(StrictBaseModel):
    case_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
    geometry: GeometrySpec
    physics: PhysicsSpec = Field(default_factory=lambda: PhysicsSpec(domains=["custom"]))
    requirements: DomainRequirements = Field(default_factory=DomainRequirements)
    repair: Literal["auto", "never", "always"] = "auto"
    repair_options: GeometryRepairOptions = Field(default_factory=GeometryRepairOptions)
    max_meshing_attempts: int = Field(default=2, ge=1, le=5)
    force_remesh: bool = False


class PreparedDomain(StrictBaseModel):
    schema_version: Literal["physicsos.prepared_domain.v1"] = "physicsos.prepared_domain.v1"
    id: str
    case_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
    status: Literal["ready", "needs_input", "needs_review", "backend_unavailable", "failed"]
    requirements: DomainRequirements
    geometry: GeometrySpec
    mesh: MeshSpec | None = None
    contract: GeometryMeshContract | None = None
    artifacts: dict[str, ArtifactRef] = Field(default_factory=dict)
    checks: dict[str, bool] = Field(default_factory=dict)
    actions: list[dict[str, Any]] = Field(default_factory=list)
    required_actions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    recipe: PrepareDomainInput

    @model_validator(mode="after")
    def readiness_requires_evidence(self):
        if self.status == "ready":
            if not self.checks or not all(self.checks.values()) or self.required_actions:
                raise ValueError("Ready domains require every preparation check to pass.")
            if self.contract is None or self.contract.semantic.unresolved_bindings:
                raise ValueError("Ready domains require bound geometry semantics.")
            if self.requirements.representation == "mesh" and (self.mesh is None or not self.mesh.quality.passes or self.mesh.quality.checked_elements <= 0):
                raise ValueError("Ready mesh domains require actual element-quality evidence.")
        return self


class PrepareDomainOutput(StrictBaseModel):
    domain: PreparedDomain
    manifest: ArtifactRef


class ExecuteCaseInput(StrictBaseModel):
    case_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
    prepared_domain: ArtifactRef
    kernel_uri: str | None = None
    controls: dict[str, Any] = Field(default_factory=dict)
    field_name: str = "u"
    timeout_seconds: int = Field(default=60, ge=1, le=86400)

    @model_validator(mode="after")
    def reserved_controls(self):
        reserved = {"case_dir", "output_dir", "domain", "domain_artifacts", "run_id", "case_id", "controls"}
        if reserved & self.controls.keys():
            raise ValueError("Solver controls cannot override managed runtime context.")
        return self


class CaseRun(StrictBaseModel):
    schema_version: Literal["physicsos.case_run.v1"] = "physicsos.case_run.v1"
    id: str
    case_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
    domain_id: str
    domain_manifest: ArtifactRef
    kernel_sha256: str
    controls: dict[str, Any]
    field_name: str
    result: SolverResult
    artifacts: dict[str, ArtifactRef] = Field(default_factory=dict)
    input_hashes: dict[str, str] = Field(default_factory=dict)
    working_case_dir: str
    errors: list[str] = Field(default_factory=list)


class ExecuteCaseOutput(StrictBaseModel):
    run: CaseRun
    manifest: ArtifactRef


class VerifyCaseInput(StrictBaseModel):
    run_manifest: ArtifactRef
    comparison_probes: ArtifactRef | None = None
    reference_uri: str | None = None
    reference_function: str = "exact_solution"
    relative_tolerance: float = Field(default=.05, gt=0, allow_inf_nan=False)
    absolute_tolerance: float = Field(default=1e-10, gt=0, allow_inf_nan=False)
    max_samples: int = Field(default=256, ge=8, le=10000)
    timeout_seconds: int = Field(default=60, ge=1, le=86400)


class VerifyCaseOutput(StrictBaseModel):
    report: AggregateVerificationReport
    artifact: ArtifactRef


class ConvergenceStudyInput(StrictBaseModel):
    case_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
    prepared_domain: ArtifactRef
    kernel_uri: str | None = None
    reference_uri: str | None = None
    reference_function: str = "exact_solution"
    refinements: list[float] = Field(default_factory=lambda: [.4, .2, .1], min_length=3)
    axis: Literal["mesh", "grid"] = "mesh"
    controls: dict[str, Any] = Field(default_factory=dict)
    field_name: str = "u"
    expected_order: float = Field(default=2, gt=0, allow_inf_nan=False)
    rate_tolerance: float = Field(default=.5, gt=0)
    error_tolerance: float = Field(default=1e-8, gt=0)
    max_samples: int = Field(default=256, ge=8, le=10000)
    timeout_seconds: int = Field(default=60, ge=1, le=86400)

    @model_validator(mode="after")
    def valid_refinements(self):
        if any(not math.isfinite(value) or value <= 0 for value in self.refinements):
            raise ValueError("Refinements must be finite and positive.")
        if self.axis == "mesh" and not all(b < a for a, b in zip(self.refinements, self.refinements[1:])):
            raise ValueError("Mesh sizes must decrease strictly.")
        if self.axis == "grid" and (any(value < 3 or value > 128 or value != int(value) for value in self.refinements) or not all(b > a for a, b in zip(self.refinements, self.refinements[1:]))):
            raise ValueError("Grid resolutions must be increasing integers in [3, 128].")
        return self


class ConvergenceStudyOutput(StrictBaseModel):
    status: Literal["verified", "failed", "uncertain"]
    runs: list[ArtifactRef] = Field(default_factory=list)
    report: ArtifactRef


class SearchRuntimeHistoryInput(StrictBaseModel):
    physics_domain: str | None = None
    regime: str | None = None
    dimension: int | None = None
    representation: Literal["mesh", "background_grid"] | None = None
    case_id: str | None = None
    status: Literal["verified", "failed", "uncertain"] | None = None
    top_k: int = Field(default=10, ge=1, le=100)


class SearchRuntimeHistoryOutput(StrictBaseModel):
    attempts: list[dict[str, Any]] = Field(default_factory=list)
    matched_attempts: int = 0
    outcome_counts: dict[str, int] = Field(default_factory=dict)
