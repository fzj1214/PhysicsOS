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


class TAPSGeometrySeparabilityAssessment(StrictBaseModel):
    status: Literal["ready_for_paper_taps", "needs_geometry_embedding", "needs_review"]
    can_use_background_grid: bool
    missing_artifacts: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class TAPSProblem(StrictBaseModel):
    id: str
    case_id: str
    problem_statement_uri: str
    derivation_uri: str
    implementation_notes_uri: str
    geometry_embedding_uri: str | None = None
    route: Literal["paper_reproduction"] = "paper_reproduction"
    axes: list[TAPSAxisSpec] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class TAPSRuntimeExtensionSpec(StrictBaseModel):
    id: str
    case_id: str
    purpose: str
    entrypoint: str
    artifact: ArtifactRef
    required_inputs: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    safety_status: Literal["draft", "requires_review", "approved_for_local_run"] = "draft"
    notes: list[str] = Field(default_factory=list)


class TAPSSupportScore(StrictBaseModel):
    score: float
    supported: bool
    reasons: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class TAPSResidualReport(StrictBaseModel):
    residuals: dict[str, float] = Field(default_factory=dict)
    converged: bool
    recommended_action: Literal["accept", "increase_rank", "refine_axes", "split_slab", "verify", "fallback"]
