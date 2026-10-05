"""Versioned strategies and independent benchmark contracts for runtime RSI."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from physicsos.schemas.case_runtime import PrepareDomainInput
from physicsos.schemas.common import ArtifactRef, StrictBaseModel
from physicsos.schemas.geometry_repair import GeometryRepairOptions
from physicsos.schemas.mesh import MeshPolicy
from physicsos.schemas.operators import PhysicsDomain
from physicsos.verification.base import ConfidenceScore

_ID = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$"


class StrategyScope(StrictBaseModel):
    problem_family: str = Field(min_length=1, max_length=160)
    physics_domains: list[PhysicsDomain] = Field(min_length=1)
    regime: str | None = None
    dimension: Literal[1, 2, 3]
    representation: Literal["mesh", "background_grid"]
    domain_side: Literal["interior", "exterior"] = "interior"

    @field_validator("physics_domains")
    @classmethod
    def canonical_domains(cls, value):
        return sorted(set(value))


class StrategySpec(StrictBaseModel):
    name: str = Field(pattern=_ID)
    scope: StrategyScope
    description: str = Field(min_length=1)
    guidance: str = Field(min_length=1)
    mesh_policy: MeshPolicy | None = None
    repair: Literal["auto", "never", "always"] | None = None
    repair_options: GeometryRepairOptions | None = None
    solver_controls: dict[str, Any] = Field(default_factory=dict)
    kernel_uri: str | None = None
    builder_uri: str | None = None
    parent: ArtifactRef | None = None

    @model_validator(mode="after")
    def one_implementation(self):
        if self.kernel_uri and self.builder_uri:
            raise ValueError("Choose a reusable kernel or a build_case_kernel(config) provider.")
        reserved = {"case_dir", "output_dir", "domain", "domain_artifacts", "run_id", "case_id", "controls"}
        if reserved & self.solver_controls.keys():
            raise ValueError("Strategy controls cannot override runtime context.")
        return self


class StrategyRevision(StrictBaseModel):
    schema_version: Literal["physicsos.rsi_strategy.v1"] = "physicsos.rsi_strategy.v1"
    id: str
    spec: StrategySpec
    artifacts: dict[str, ArtifactRef] = Field(default_factory=dict)


class RefinementCheck(StrictBaseModel):
    refinements: list[float] = Field(min_length=3, max_length=6)
    expected_order: float = Field(default=2, gt=0, allow_inf_nan=False)
    rate_tolerance: float = Field(default=.5, gt=0, allow_inf_nan=False)
    error_tolerance: float = Field(default=1e-8, gt=0, allow_inf_nan=False)


class BenchmarkCase(StrictBaseModel):
    id: str = Field(pattern=_ID)
    split: Literal["development", "holdout"]
    prepare: PrepareDomainInput
    controls: dict[str, Any] = Field(default_factory=dict)
    tunable_controls: list[str] = Field(default_factory=list)
    kernel_uri: str | None = None
    reference_uri: str
    reference_function: str = "exact_solution"
    field_name: str = "u"
    relative_tolerance: float = Field(default=.05, gt=0, allow_inf_nan=False)
    absolute_tolerance: float = Field(default=1e-10, gt=0, allow_inf_nan=False)
    max_samples: int = Field(default=128, ge=8, le=10000)
    convergence: RefinementCheck | None = None


class PromotionPolicy(StrictBaseModel):
    min_development_cases: int = Field(default=1, ge=1, le=32)
    min_holdout_cases: int = Field(default=3, ge=1, le=32)
    require_convergence: bool = True
    max_error_regression: float = Field(default=.05, ge=0, allow_inf_nan=False)
    min_error_improvement: float = Field(default=.01, gt=0, allow_inf_nan=False)
    rollback_failure_limit: int = Field(default=1, ge=1, le=10)


class BenchmarkSuiteInput(StrictBaseModel):
    name: str = Field(pattern=_ID)
    scope: StrategyScope
    benchmarks: list[BenchmarkCase] = Field(min_length=2, max_length=32)
    policy: PromotionPolicy = Field(default_factory=PromotionPolicy)

    @model_validator(mode="after")
    def distinct_case_ids(self):
        if len({case.id for case in self.benchmarks}) != len(self.benchmarks):
            raise ValueError("Benchmark IDs must be distinct.")
        return self


class BenchmarkSuite(StrictBaseModel):
    schema_version: Literal["physicsos.rsi_suite.v1"] = "physicsos.rsi_suite.v1"
    id: str
    spec: BenchmarkSuiteInput
    problem_identities: dict[str, str]
    holdout_fingerprint: str
    artifacts: dict[str, ArtifactRef] = Field(default_factory=dict)
    resource_directories: dict[str, str] = Field(default_factory=dict)


class RegisterStrategyOutput(StrictBaseModel):
    strategy: StrategyRevision
    manifest: ArtifactRef


class RegisterSuiteOutput(StrictBaseModel):
    suite: BenchmarkSuite
    manifest: ArtifactRef


class RevisionProviderSpec(StrictBaseModel):
    name: str = Field(pattern=_ID)
    description: str = Field(min_length=1)
    python_uri: str
    entrypoint: str = Field(default="revise_strategy", pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")


class RevisionProvider(StrictBaseModel):
    schema_version: Literal["physicsos.rsi_revision_provider.v1"] = "physicsos.rsi_revision_provider.v1"
    id: str
    spec: RevisionProviderSpec
    artifacts: dict[str, ArtifactRef] = Field(default_factory=dict)


class RegisterRevisionProviderOutput(StrictBaseModel):
    provider: RevisionProvider
    manifest: ArtifactRef


class StrategyPatch(StrictBaseModel):
    name: str | None = Field(default=None, pattern=_ID)
    description: str | None = Field(default=None, min_length=1)
    guidance: str | None = Field(default=None, min_length=1)
    mesh_policy: MeshPolicy | None = None
    repair: Literal["auto", "never", "always"] | None = None
    repair_options: GeometryRepairOptions | None = None
    solver_controls: dict[str, Any] | None = None


class RevisionProposal(StrictBaseModel):
    stop: bool = False
    rationale: str = Field(min_length=1)
    patch: StrategyPatch = Field(default_factory=StrategyPatch)
    kernel_source: str | None = None
    builder_source: str | None = None

    @model_validator(mode="after")
    def one_implementation(self):
        if self.kernel_source is not None and self.builder_source is not None:
            raise ValueError("A revision can supply kernel_source or builder_source, not both.")
        return self


class ImproveStrategiesInput(StrictBaseModel):
    suite: ArtifactRef
    revision_provider: ArtifactRef
    initial_strategy: ArtifactRef | None = None
    max_revisions: int = Field(default=3, ge=1, le=8)
    max_stagnant_revisions: int = Field(default=2, ge=1, le=8)
    max_kernel_runs: int = Field(default=128, ge=1, le=512)
    timeout_seconds: int = Field(default=60, ge=1, le=3600)
    development_error_target: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    include_holdout: bool = True
    auto_promote: bool = True


class ImproveStrategiesOutput(StrictBaseModel):
    status: Literal["promoted", "eligible", "rejected", "needs_holdout", "blocked", "budget_exhausted", "no_improvement"]
    selected_strategy: ArtifactRef | None = None
    revisions: list[ArtifactRef] = Field(default_factory=list)
    evaluations: list[ArtifactRef] = Field(default_factory=list)
    final_evaluation: ArtifactRef | None = None
    report: ArtifactRef
    reserved_kernel_runs: int = 0
    provider_calls: int = 0
    reasons: list[str] = Field(default_factory=list)


class EvaluateStrategiesInput(StrictBaseModel):
    suite: ArtifactRef
    candidates: list[ArtifactRef] = Field(min_length=1, max_length=8)
    baseline: ArtifactRef | None = None
    include_holdout: bool = True
    auto_promote: bool = True
    max_kernel_runs: int = Field(default=128, ge=1, le=512)
    timeout_seconds: int = Field(default=60, ge=1, le=3600)


class BenchmarkOutcome(StrictBaseModel):
    benchmark_id: str
    split: Literal["development", "holdout", "production"]
    problem_identity: str
    strategy: ArtifactRef
    status: Literal["verified", "failed", "uncertain"]
    stage: Literal["preparation", "generation", "execution", "verification", "convergence", "complete"]
    evidence: dict[str, ArtifactRef] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    diagnostics: list[str] = Field(default_factory=list)
    suggested_actions: list[str] = Field(default_factory=list)


class EvaluateStrategiesOutput(StrictBaseModel):
    status: Literal["promoted", "eligible", "rejected", "needs_holdout", "blocked"]
    selected_strategy: ArtifactRef | None = None
    report: ArtifactRef
    reasons: list[str] = Field(default_factory=list)
    activation_id: str | None = None


class PromoteStrategyInput(StrictBaseModel):
    evaluation: ArtifactRef


class PromotionOutput(StrictBaseModel):
    status: Literal["promoted", "blocked"]
    activation_id: str | None = None
    generation: int | None = None
    reasons: list[str] = Field(default_factory=list)
    report: ArtifactRef


class RollbackStrategyInput(StrictBaseModel):
    scope: StrategyScope
    expected_generation: int = Field(ge=0)
    reason: str = Field(min_length=1)


class RollbackOutput(StrictBaseModel):
    status: Literal["rolled_back", "blocked"]
    active_strategy: ArtifactRef | None = None
    generation: int | None = None
    report: ArtifactRef


class AssessCapabilityInput(StrictBaseModel):
    scope: StrategyScope
    strategy: ArtifactRef | None = None
    relative_tolerance: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    absolute_tolerance: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    require_convergence: bool | None = None
    expected_order: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    rate_tolerance: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    error_tolerance: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class CapabilityEstimate(StrictBaseModel):
    scope: StrategyScope
    strategy: ArtifactRef | None = None
    confidence: ConfidenceScore
    independent_problems: int
    verified: int
    failed: int
    uncertain: int
    success_rate: float | None
    predicted_success_probability: float
    success_interval: list[float]
    production_brier_score: float | None = None
    invalid_evidence: int = 0
    failure_patterns: dict[str, int] = Field(default_factory=dict)
    failure_evidence: list[ArtifactRef] = Field(default_factory=list)
    suggested_actions: list[str] = Field(default_factory=list)
    comparison_contract: dict[str, Any] = Field(default_factory=dict)
    guidance: str | None = None
    active_generation: int = 0
    recommendation: str
    report: ArtifactRef


class BindCaseStrategyInput(StrictBaseModel):
    case_id: str = Field(pattern=_ID)
    assessment: AssessCapabilityInput


class BindCaseStrategyOutput(StrictBaseModel):
    case_id: str
    status: Literal["available", "needs_strategy", "needs_review"]
    binding: ArtifactRef
    context: ArtifactRef
    guidance: ArtifactRef
    capability: CapabilityEstimate
    warnings: list[str] = Field(default_factory=list)


class SolveWithStrategyInput(StrictBaseModel):
    problem_family: str = Field(min_length=1, max_length=160)
    prepare: PrepareDomainInput
    strategy: ArtifactRef | None = None
    controls: dict[str, Any] = Field(default_factory=dict)
    tunable_controls: list[str] = Field(default_factory=list)
    kernel_uri: str | None = None
    reference_uri: str
    reference_function: str = "exact_solution"
    field_name: str = "u"
    relative_tolerance: float = Field(default=.05, gt=0, allow_inf_nan=False)
    absolute_tolerance: float = Field(default=1e-10, gt=0, allow_inf_nan=False)
    max_samples: int = Field(default=128, ge=8, le=10000)
    convergence: RefinementCheck | None = None
    timeout_seconds: int = Field(default=60, ge=1, le=3600)


class SolveWithStrategyOutput(StrictBaseModel):
    status: Literal["verified", "failed", "uncertain", "needs_strategy"]
    outcome: BenchmarkOutcome | None = None
    rollback: RollbackOutput | None = None
    report: ArtifactRef
