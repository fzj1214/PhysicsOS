from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from physicsos.schemas.common import ArtifactRef, StrictBaseModel
from physicsos.schemas.contracts import ContractReviewReport, PhysicsProblemContract
from physicsos.schemas.geometry import GeometryEncoding, GeometrySpec
from physicsos.schemas.knowledge import KnowledgeContext
from physicsos.schemas.mesh import MeshQualityReport, MeshSpec
from physicsos.schemas.postprocess import PostprocessResult
from physicsos.schemas.problem import PhysicsProblem
from physicsos.schemas.solver import SolverDecision, SolverResult
from physicsos.schemas.taps import TAPSCompilationPlan, TAPSProblem, TAPSResidualReport, TAPSSupportScore
from physicsos.schemas.verification import VerificationReport
from physicsos.tools.memory_tools import CaseMemoryContext, StoreCaseResultOutput


AgentStatus = Literal[
    "ready",
    "complete",
    "needs_user_input",
    "needs_knowledge",
    "fallback_required",
    "failed",
]


class AgentHandoff(StrictBaseModel):
    """Machine-readable handoff envelope between PhysicsOS core agents."""

    agent_name: str
    status: AgentStatus
    problem_id: str
    summary: str
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    recommended_next_agent: str | None = None
    recommended_next_action: str | None = None


class ValidationRetryContext(StrictBaseModel):
    agent_name: str = "validate_physics_problem"
    stage: str = "problem_validation"
    attempt: int
    problem: PhysicsProblem
    input_context: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    retry_instruction: str


class GeometryMeshAgentInput(StrictBaseModel):
    problem: PhysicsProblem
    requested_encodings: list[str] = Field(default_factory=list)
    target_backends: list[str] = Field(default_factory=list)
    case_memory_context: CaseMemoryContext | None = None


class GeometryMeshAgentOutput(StrictBaseModel):
    handoff: AgentHandoff
    geometry: GeometrySpec
    mesh: MeshSpec | None = None
    encodings: list[GeometryEncoding] = Field(default_factory=list)
    quality: MeshQualityReport | None = None


class KnowledgeAgentInput(StrictBaseModel):
    problem: PhysicsProblem
    query: str
    local_top_k: int = 4
    arxiv_max_results: int = 0
    use_deepsearch: bool = False
    case_memory_context: CaseMemoryContext | None = None


class KnowledgeAgentOutput(StrictBaseModel):
    handoff: AgentHandoff
    context: KnowledgeContext


class TAPSAgentInput(StrictBaseModel):
    problem: PhysicsProblem
    problem_contract: PhysicsProblemContract | None = None
    knowledge_context: KnowledgeContext | None = None
    case_memory_context: CaseMemoryContext | None = None
    tensor_rank: int = 8
    max_wall_time_seconds: float = 120.0


class TAPSAgentOutput(StrictBaseModel):
    handoff: AgentHandoff
    support: TAPSSupportScore
    compilation_plan: TAPSCompilationPlan | None = None
    taps_problem: TAPSProblem | None = None
    contract_review: ContractReviewReport | None = None
    result: SolverResult | None = None
    residual: TAPSResidualReport | None = None


class SolverAgentInput(StrictBaseModel):
    problem: PhysicsProblem
    taps_handoff: AgentHandoff | None = None
    case_memory_context: CaseMemoryContext | None = None
    force_full_solver: bool = False


class SolverAgentOutput(StrictBaseModel):
    handoff: AgentHandoff
    decision: SolverDecision | None = None
    result: SolverResult


class VerificationAgentInput(StrictBaseModel):
    problem: PhysicsProblem
    result: SolverResult
    taps_problem: TAPSProblem | None = None
    case_memory_context: CaseMemoryContext | None = None


class VerificationAgentOutput(StrictBaseModel):
    handoff: AgentHandoff
    report: VerificationReport
    taps_residual: TAPSResidualReport | None = None


class PostprocessAgentInput(StrictBaseModel):
    problem: PhysicsProblem
    result: SolverResult
    verification: VerificationReport
    case_memory_context: CaseMemoryContext | None = None


class PostprocessAgentOutput(StrictBaseModel):
    handoff: AgentHandoff
    result: PostprocessResult


class CaseMemoryAgentInput(StrictBaseModel):
    problem: PhysicsProblem
    result: SolverResult
    verification: VerificationReport
    postprocess: PostprocessResult
    case_memory_context: CaseMemoryContext | None = None


class CaseMemoryAgentOutput(StrictBaseModel):
    handoff: AgentHandoff
    stored: StoreCaseResultOutput


class PhysicsOSWorkflowState(StrictBaseModel):
    problem: PhysicsProblem
    run_id: str | None = None
    problem_contract: PhysicsProblemContract | None = None
    case_memory_context: CaseMemoryContext | None = None
    geometry: GeometryMeshAgentOutput | None = None
    knowledge: KnowledgeAgentOutput | None = None
    taps: TAPSAgentOutput | None = None
    solver: SolverAgentOutput | None = None
    verification: VerificationAgentOutput | None = None
    postprocess: PostprocessAgentOutput | None = None
    case_memory: CaseMemoryAgentOutput | None = None
    validation_attempts: list[ValidationRetryContext] = Field(default_factory=list)
