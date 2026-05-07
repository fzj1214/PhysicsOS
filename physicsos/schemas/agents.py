from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from physicsos.schemas.common import ArtifactRef, StrictBaseModel
from physicsos.schemas.knowledge import KnowledgeContext
from physicsos.schemas.postprocess import PostprocessResult
from physicsos.schemas.solver import SolverResult
from physicsos.schemas.taps import TAPSGeometrySeparabilityAssessment, TAPSProblem, TAPSResidualReport, TAPSSupportScore
from physicsos.schemas.verification import VerificationReport
from physicsos.tools.memory_tools import CaseMemoryContext, StoreCaseResultOutput


AgentStatus = Literal[
    "ready",
    "complete",
    "needs_user_input",
    "needs_knowledge",
    "needs_geometry_embedding",
    "needs_verification",
    "failed",
]


class AgentHandoff(StrictBaseModel):
    """Machine-readable handoff envelope between paper-style PhysicsOS agents."""

    agent_name: str
    status: AgentStatus
    case_id: str
    summary: str
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    recommended_next_agent: str | None = None
    recommended_next_action: str | None = None


class AnalysisFileAgentOutput(StrictBaseModel):
    handoff: AgentHandoff
    problem_statement: ArtifactRef | None = None
    problem_json: ArtifactRef | None = None
    open_questions: ArtifactRef | None = None


class GeometryEmbeddingAgentOutput(StrictBaseModel):
    handoff: AgentHandoff
    embedding: ArtifactRef | None = None
    embedding_notes: ArtifactRef | None = None
    assessment: TAPSGeometrySeparabilityAssessment | None = None


class KnowledgeAgentInput(StrictBaseModel):
    case_id: str
    query: str
    local_top_k: int = 4
    arxiv_max_results: int = 0
    use_deepsearch: bool = False
    case_memory_context: CaseMemoryContext | None = None


class KnowledgeAgentOutput(StrictBaseModel):
    handoff: AgentHandoff
    context: KnowledgeContext


class TAPSDerivationAgentOutput(StrictBaseModel):
    handoff: AgentHandoff
    derivation_prompt: ArtifactRef | None = None
    derivation: ArtifactRef | None = None
    implementation_notes: ArtifactRef | None = None
    support: TAPSSupportScore | None = None


class TAPSImplementationAgentOutput(StrictBaseModel):
    handoff: AgentHandoff
    taps_problem: TAPSProblem | None = None
    kernel: ArtifactRef | None = None
    execution_plan: ArtifactRef | None = None
    static_review: ArtifactRef | None = None
    result: SolverResult | None = None


class VerificationAgentOutput(StrictBaseModel):
    handoff: AgentHandoff
    report: VerificationReport | None = None
    taps_residual: TAPSResidualReport | None = None


class PostprocessAgentOutput(StrictBaseModel):
    handoff: AgentHandoff
    result: PostprocessResult


class CaseMemoryAgentInput(StrictBaseModel):
    case_id: str
    result: SolverResult
    verification: VerificationReport
    postprocess: PostprocessResult
    case_memory_context: CaseMemoryContext | None = None


class CaseMemoryAgentOutput(StrictBaseModel):
    handoff: AgentHandoff
    stored: StoreCaseResultOutput


class PhysicsOSCaseState(StrictBaseModel):
    case_id: str
    current_stage: str | None = None
    analysis: AnalysisFileAgentOutput | None = None
    geometry: GeometryEmbeddingAgentOutput | None = None
    knowledge: KnowledgeAgentOutput | None = None
    derivation: TAPSDerivationAgentOutput | None = None
    implementation: TAPSImplementationAgentOutput | None = None
    verification: VerificationAgentOutput | None = None
    postprocess: PostprocessAgentOutput | None = None
    case_memory: CaseMemoryAgentOutput | None = None
    events: list[dict[str, Any]] = Field(default_factory=list)
