from __future__ import annotations

from typing import Literal

from pydantic import Field

from physicsos.schemas.common import ArtifactRef, Provenance, RuntimeStats, StrictBaseModel


class FieldDataRef(StrictBaseModel):
    field: str
    uri: str
    format: str
    location: str | None = None
    units: str | None = None


class SolverResult(StrictBaseModel):
    id: str
    problem_id: str
    backend: str
    status: Literal["success", "failed", "partial", "needs_review"]
    fields: list[FieldDataRef] = Field(default_factory=list)
    scalar_outputs: dict[str, float | int | str] = Field(default_factory=dict)
    residuals: dict[str, float] = Field(default_factory=dict)
    uncertainty: dict[str, float] = Field(default_factory=dict)
    runtime: RuntimeStats = Field(default_factory=RuntimeStats)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    provenance: Provenance


class SupportScore(StrictBaseModel):
    backend: str
    score: float
    supported: bool
    reasons: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    required_missing_inputs: list[str] = Field(default_factory=list)


class CostEstimate(StrictBaseModel):
    backend: str
    expected_runtime_seconds: float | None = None
    expected_cpu_cores: int | None = None
    expected_gpus: int | None = None
    expected_memory_gb: float | None = None
    notes: list[str] = Field(default_factory=list)


class PreparedSolverCase(StrictBaseModel):
    problem_id: str
    backend: str
    workspace_uri: str
    artifacts: list[ArtifactRef] = Field(default_factory=list)


class SolverPolicy(StrictBaseModel):
    prefer_open_source: bool = True
    allow_private_plugins: bool = False
    max_expected_error: float | None = None
    prefer_surrogate: bool = True
    force_full_solver: bool = False


class HybridPolicy(StrictBaseModel):
    mode: Literal["warm_start", "corrector", "surrogate_then_full", "adaptive"] = "adaptive"
    accept_surrogate_if_verified: bool = True


class SolverDecision(StrictBaseModel):
    selected_backend: str
    candidate_backends: list[SupportScore]
    mode: Literal["surrogate_only", "full_solver", "hybrid", "warm_start", "corrector"]
    reason: str
    expected_error: float | None = None
    expected_runtime_seconds: float | None = None

