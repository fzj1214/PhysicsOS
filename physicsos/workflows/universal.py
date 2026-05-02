from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import Field, ValidationError

from physicsos.events import PhysicsOSEvent, emit_physicsos_event
from physicsos.schemas.common import StrictBaseModel
from physicsos.schemas.agents import (
    AgentHandoff,
    CaseMemoryAgentInput,
    CaseMemoryAgentOutput,
    GeometryMeshAgentInput,
    GeometryMeshAgentOutput,
    KnowledgeAgentInput,
    KnowledgeAgentOutput,
    PostprocessAgentInput,
    PostprocessAgentOutput,
    SolverAgentInput,
    SolverAgentOutput,
    TAPSAgentInput,
    TAPSAgentOutput,
    VerificationAgentInput,
    VerificationAgentOutput,
    PhysicsOSWorkflowState,
    ValidationRetryContext,
)
from physicsos.schemas.knowledge import KnowledgeContext
from physicsos.schemas.mesh import MeshSpec
from physicsos.schemas.postprocess import PostprocessResult
from physicsos.schemas.problem import PhysicsProblem
from physicsos.schemas.solver import SolverResult
from physicsos.schemas.taps import TAPSProblem, TAPSResidualReport, TAPSBasisConfig
from physicsos.schemas.verification import VerificationReport
from physicsos.schemas.operators import PhysicsSpec
from physicsos.tools.geometry_tools import GenerateGeometryEncodingInput, GenerateMeshInput, LabelRegionsInput, generate_geometry_encoding, generate_mesh, label_regions
from physicsos.tools.knowledge_tools import BuildKnowledgeContextInput, build_knowledge_context
from physicsos.tools.memory_tools import (
    AppendCaseMemoryEventInput,
    CaseMemoryContext,
    CaseMemoryEvent,
    SearchCaseMemoryInput,
    StoreCaseResultInput,
    StoreCaseResultOutput,
    append_case_memory_event,
    build_case_memory_context,
    store_case_result,
)
from physicsos.tools.postprocess_tools import (
    ExtractKPIsInput,
    GenerateVisualizationsInput,
    WriteSimulationReportInput,
    extract_kpis,
    generate_visualizations,
    write_simulation_report,
)
from physicsos.tools.problem_tools import ValidatePhysicsProblemInput, validate_physics_problem
from physicsos.tools.solver_tools import RunSurrogateSolverInput, run_surrogate_solver
from physicsos.tools.taps_tools import (
    BuildTAPSProblemInput,
    EstimateTAPSResidualInput,
    EstimateTAPSSupportInput,
    RunTAPSBackendInput,
    build_taps_problem,
    estimate_taps_residual,
    estimate_taps_support,
    run_taps_backend,
)
from physicsos.tools.verification_tools import (
    CheckConservationLawsInput,
    EstimateUncertaintyInput,
    ValidateSelectedSlicesInput,
    check_conservation_laws,
    estimate_uncertainty,
    validate_selected_slices,
)
from physicsos.workflows.taps_thermal import build_default_thermal_problem

AgentOutputT = TypeVar("AgentOutputT")


class WorkflowStep(StrictBaseModel):
    name: str
    status: str
    summary: str


class PhysicsOSWorkflowResult(StrictBaseModel):
    state: PhysicsOSWorkflowState
    run_id: str
    problem: PhysicsProblem
    case_memory_context: CaseMemoryContext | None = None
    geometry: GeometryMeshAgentOutput | None = None
    knowledge: KnowledgeAgentOutput | None = None
    taps: TAPSAgentOutput | None = None
    solver: SolverAgentOutput | None = None
    verification_agent: VerificationAgentOutput | None = None
    postprocess_agent: PostprocessAgentOutput | None = None
    case_memory: CaseMemoryAgentOutput | None = None
    knowledge_context: KnowledgeContext | None = None
    taps_problem: TAPSProblem | None = None
    taps_residual: TAPSResidualReport | None = None
    solver_result: SolverResult | None = None
    verification: VerificationReport | None = None
    postprocess: PostprocessResult | None = None
    case_store: StoreCaseResultOutput | None = None
    validation_attempts: list[ValidationRetryContext] = Field(default_factory=list)
    trace: list[WorkflowStep] = Field(default_factory=list)
    events: list[PhysicsOSEvent] = Field(default_factory=list)


def _handoff(
    *,
    agent_name: str,
    status: str,
    problem_id: str,
    summary: str,
    recommended_next_agent: str | None = None,
    recommended_next_action: str | None = None,
) -> AgentHandoff:
    return AgentHandoff(
        agent_name=agent_name,
        status=status,  # type: ignore[arg-type]
        problem_id=problem_id,
        summary=summary,
        recommended_next_agent=recommended_next_agent,
        recommended_next_action=recommended_next_action,
    )


def _emit_workflow_event(
    events: list[PhysicsOSEvent],
    *,
    run_id: str,
    case_id: str,
    event: str,
    stage: str,
    status: str,
    summary: str,
    payload: dict[str, Any] | None = None,
) -> PhysicsOSEvent:
    physicsos_event = PhysicsOSEvent(
        run_id=run_id,
        case_id=case_id,
        event=event,  # type: ignore[arg-type]
        stage=stage,
        status=status,
        summary=summary,
        payload=payload or {},
    )
    events.append(physicsos_event)
    emit_physicsos_event(physicsos_event)
    return physicsos_event


def _append_case_event(
    *,
    run_id: str,
    case_id: str,
    stage: str,
    event: str,
    summary: str,
    payload: dict[str, Any] | None = None,
) -> None:
    append_case_memory_event(
        AppendCaseMemoryEventInput(
            event=CaseMemoryEvent(
                run_id=run_id,
                case_id=case_id,
                stage=stage,
                event=event,
                summary=summary,
                payload=payload or {},
            )
        )
    )


def _verify_taps_result(problem: PhysicsProblem, taps_problem: TAPSProblem, result: SolverResult) -> tuple[TAPSResidualReport, VerificationReport]:
    residual = estimate_taps_residual(
        EstimateTAPSResidualInput(problem=problem, taps_problem=taps_problem, result=result)
    ).report
    conservation = check_conservation_laws(CheckConservationLawsInput(problem=problem, result=result))
    slices = validate_selected_slices(ValidateSelectedSlicesInput(problem=problem, result=result))
    uncertainty = estimate_uncertainty(EstimateUncertaintyInput(problem=problem, result=result, method="residual_proxy"))
    conservation_skipped_only = bool(conservation.skipped_quantities) and not conservation.conservation_errors
    conservation_acceptable = conservation.passes or conservation_skipped_only
    verification_passed = residual.converged and conservation_acceptable and slices.passes
    status = "accepted" if residual.converged else "needs_full_solver"
    warnings: list[str] = []
    if conservation_skipped_only:
        warnings.append(
            "Conservation check skipped declared quantities without backend-reported imbalance metrics: "
            + ", ".join(conservation.skipped_quantities)
        )
    if residual.converged and not conservation_acceptable:
        status = "needs_full_solver"
    if residual.converged and conservation_skipped_only:
        status = "accepted_with_warnings"
    if residual.converged and conservation_acceptable and not slices.passes:
        status = "accepted_with_warnings"
        warnings.append("Selected-slice validation did not fully pass.")
    action = "accept" if verification_passed else "run_full_solver"
    explanation = "TAPS result accepted by residual, conservation, and selected-slice checks."
    if warnings:
        explanation = "TAPS result accepted with warnings: " + " ".join(warnings)
    elif not verification_passed:
        failed = []
        if not residual.converged:
            failed.append("residual check failed")
        if not conservation_acceptable:
            failed.append("conservation check failed")
        if not slices.passes:
            failed.append("selected-slice validation failed")
        explanation = "TAPS result requires follow-up: " + "; ".join(failed) + "."
    verification = VerificationReport(
        problem_id=problem.id,
        result_id=result.id,
        status=status,
        residuals=residual.residuals,
        conservation_errors=conservation.conservation_errors,
        uncertainty=uncertainty.uncertainty,
        ood_score=0.0 if residual.converged else 0.5,
        warnings=warnings,
        recommended_next_action=action,
        explanation=explanation,
    )
    return residual, verification


def _verify_fallback_result(problem: PhysicsProblem, result: SolverResult) -> VerificationReport:
    return VerificationReport(
        problem_id=problem.id,
        result_id=result.id,
        status="needs_full_solver",
        residuals=result.residuals,
        uncertainty=result.uncertainty,
        ood_score=1.0,
        recommended_next_action="run_full_solver",
        explanation="Surrogate fallback executed, but no trusted residual verifier is connected for this case yet.",
    )


def _model_context(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return {"agent_input": value.model_dump(mode="json")}
    return {"agent_input": repr(value)}


def _run_agent_with_retries(
    *,
    agent_name: str,
    stage: str,
    problem: PhysicsProblem,
    agent_input: Any,
    runner: Callable[[Any], AgentOutputT],
    state: PhysicsOSWorkflowState,
    max_validation_attempts: int,
    events: list[PhysicsOSEvent] | None = None,
    run_id: str | None = None,
) -> AgentOutputT | None:
    max_attempts = max(1, max_validation_attempts)
    for attempt in range(1, max_attempts + 1):
        try:
            return runner(agent_input)
        except (ValidationError, ValueError, RuntimeError) as exc:
            final_attempt = attempt == max_attempts
            retry = ValidationRetryContext(
                agent_name=agent_name,
                stage=stage,
                attempt=attempt,
                problem=problem,
                input_context=_model_context(agent_input),
                errors=[str(exc)],
                retry_instruction=(
                    "Typed agent call failed after all retry attempts. Return this context to the main agent "
                    "so it can repair the typed input or select another workflow branch."
                    if final_attempt
                    else "Typed agent call failed. Rebuild or repair the typed input/output contract and retry this same stage."
                ),
            )
            state.validation_attempts.append(retry)
            if events is not None and run_id is not None:
                _emit_workflow_event(
                    events,
                    run_id=run_id,
                    case_id=problem.id,
                    event="validation.retry",
                    stage=stage,
                    status="retry_exhausted" if final_attempt else "retrying",
                    summary=f"{agent_name} typed call failed on attempt {attempt}/{max_attempts}: {exc}",
                    payload=retry.model_dump(mode="json"),
                )
                _append_case_event(
                    run_id=run_id,
                    case_id=problem.id,
                    stage=stage,
                    event="validation_retry",
                    summary=f"{agent_name} retry {attempt}/{max_attempts}: {exc}",
                    payload=retry.model_dump(mode="json"),
                )
    return None


def _run_geometry_mesh_agent(input: GeometryMeshAgentInput) -> GeometryMeshAgentOutput:
    problem = input.problem
    geometry = problem.geometry
    artifacts = []
    if not geometry.boundaries or not geometry.regions:
        labeled = label_regions(LabelRegionsInput(geometry=geometry, physics_domain=problem.domain)).geometry
        geometry = labeled

    mesh = problem.mesh
    quality = mesh.quality if mesh is not None else None
    if mesh is None and geometry.dimension > 1:
        mesh = generate_mesh(
            GenerateMeshInput(
                geometry=geometry,
                physics=PhysicsSpec(domains=[problem.domain]),
                target_backends=input.target_backends,
            )
        ).mesh
        quality = mesh.quality

    encodings = list(geometry.encodings)
    requested = input.requested_encodings
    if requested:
        generated = generate_geometry_encoding(
            GenerateGeometryEncodingInput(geometry=geometry, mesh=mesh, encodings=requested)
        )
        encodings.extend(generated.encodings)
        artifacts.extend(generated.artifacts)
        geometry = geometry.model_copy(update={"encodings": encodings})

    return GeometryMeshAgentOutput(
        handoff=_handoff(
            agent_name="geometry-mesh-agent",
            status="complete",
            problem_id=problem.id,
            summary=f"geometry={geometry.id}; mesh={'ready' if mesh is not None else 'not_required_or_not_generated'}; encodings={len(encodings)}",
            recommended_next_agent="taps-agent",
            recommended_next_action="validate_and_compile_taps_problem",
        ).model_copy(update={"artifacts": artifacts}),
        geometry=geometry,
        mesh=mesh,
        encodings=encodings,
        quality=quality,
    )


def _run_knowledge_agent(input: KnowledgeAgentInput) -> KnowledgeAgentOutput:
    context = build_knowledge_context(
        BuildKnowledgeContextInput(
            query=input.query,
            local_top_k=input.local_top_k,
            arxiv_max_results=input.arxiv_max_results,
            use_deepsearch=input.use_deepsearch,
        )
    ).context
    return KnowledgeAgentOutput(
        handoff=_handoff(
            agent_name="knowledge-agent",
            status="complete",
            problem_id=input.problem.id,
            summary=f"Retrieved {len(context.chunks)} local chunks and {len(context.papers)} papers.",
            recommended_next_agent="geometry-mesh-agent",
            recommended_next_action="prepare_geometry_mesh_and_encodings",
        ),
        context=context,
    )


def _validate_problem_with_retries(
    problem: PhysicsProblem,
    *,
    max_validation_attempts: int,
    stage: str,
    input_context: dict,
) -> tuple[bool, list[ValidationRetryContext], WorkflowStep]:
    attempts: list[ValidationRetryContext] = []
    max_attempts = max(1, max_validation_attempts)
    validation = None
    for attempt in range(1, max_attempts + 1):
        validation = validate_physics_problem(ValidatePhysicsProblemInput(problem=problem))
        if validation.valid:
            return (
                True,
                attempts,
                WorkflowStep(
                    name=stage,
                    status="accepted",
                    summary="; ".join(validation.warnings) or "Problem is solver-ready.",
                ),
            )
        retry_instruction = (
            "Repair the PhysicsProblem using the provided typed input_context, then retry validation. "
            "Do not continue to TAPS or solver execution until validation passes."
        )
        if attempt == max_attempts:
            retry_instruction = (
                "Validation attempts exhausted. Return this typed context to the main agent so it can repair "
                "the PhysicsProblem or ask the user for the missing inputs."
            )
        attempts.append(
            ValidationRetryContext(
                agent_name="validate_physics_problem",
                stage=stage,
                attempt=attempt,
                problem=problem,
                input_context=input_context,
                errors=validation.errors,
                warnings=validation.warnings,
                retry_instruction=retry_instruction,
            )
        )
    assert validation is not None
    return (
        False,
        attempts,
        WorkflowStep(
            name=stage,
            status="retry_exhausted",
            summary="; ".join(validation.errors + validation.warnings) or "Problem validation failed.",
        ),
    )


def _run_taps_agent(input: TAPSAgentInput) -> TAPSAgentOutput:
    support = estimate_taps_support(EstimateTAPSSupportInput(problem=input.problem)).support
    if not support.supported:
        return TAPSAgentOutput(
            handoff=_handoff(
                agent_name="taps-agent",
                status="fallback_required",
                problem_id=input.problem.id,
                summary=f"TAPS support score={support.score}; risks={', '.join(support.risks) or 'none'}.",
                recommended_next_agent="solver-agent",
                recommended_next_action="route_solver_backend",
            ),
            support=support,
        )

    taps_problem = build_taps_problem(
        BuildTAPSProblemInput(
            problem=input.problem,
            basis=TAPSBasisConfig(tensor_rank=input.tensor_rank, reproducing_order=2),
        )
    ).taps_problem
    solver_result = run_taps_backend(RunTAPSBackendInput(problem=input.problem, taps_problem=taps_problem)).result
    residual = estimate_taps_residual(
        EstimateTAPSResidualInput(problem=input.problem, taps_problem=taps_problem, result=solver_result)
    ).report
    return TAPSAgentOutput(
        handoff=_handoff(
            agent_name="taps-agent",
            status="complete" if solver_result.status in {"success", "needs_review"} else "failed",
            problem_id=input.problem.id,
            summary=f"backend={solver_result.backend}; action={residual.recommended_action}",
            recommended_next_agent="verification-agent",
            recommended_next_action="verify_result",
        ),
        support=support,
        taps_problem=taps_problem,
        result=solver_result,
        residual=residual,
    )


def _run_solver_agent(input: SolverAgentInput) -> SolverAgentOutput:
    solver_result = run_surrogate_solver(RunSurrogateSolverInput(problem=input.problem, backend="")).result
    return SolverAgentOutput(
        handoff=_handoff(
            agent_name="solver-agent",
            status="complete",
            problem_id=input.problem.id,
            summary=f"backend={solver_result.backend}",
            recommended_next_agent="verification-agent",
            recommended_next_action="verify_result",
        ),
        result=solver_result,
    )


def _run_verification_agent(input: VerificationAgentInput) -> VerificationAgentOutput:
    if input.taps_problem is not None:
        taps_residual, report = _verify_taps_result(input.problem, input.taps_problem, input.result)
    else:
        taps_residual = None
        report = _verify_fallback_result(input.problem, input.result)
    return VerificationAgentOutput(
        handoff=_handoff(
            agent_name="verification-agent",
            status="complete",
            problem_id=input.problem.id,
            summary=report.explanation,
            recommended_next_agent="postprocess-agent",
            recommended_next_action="postprocess_result",
        ),
        report=report,
        taps_residual=taps_residual,
    )


def _run_postprocess_agent(input: PostprocessAgentInput) -> PostprocessAgentOutput:
    kpis = extract_kpis(ExtractKPIsInput(problem=input.problem, result=input.result))
    visualizations = generate_visualizations(GenerateVisualizationsInput(problem=input.problem, result=input.result))
    postprocess = PostprocessResult(
        problem_id=input.problem.id,
        result_id=input.result.id,
        kpis=kpis.kpis,
        units=kpis.units,
        visualizations=visualizations.artifacts,
        recommendations=[input.verification.explanation],
    )
    report = write_simulation_report(
        WriteSimulationReportInput(
            problem=input.problem,
            result=input.result,
            verification=input.verification,
            postprocess=postprocess,
        )
    )
    postprocess.report = report.report
    if report.manifest is not None:
        postprocess.visualizations.append(report.manifest)
    return PostprocessAgentOutput(
        handoff=_handoff(
            agent_name="postprocess-agent",
            status="complete",
            problem_id=input.problem.id,
            summary=f"Report written to {report.report.uri}.",
            recommended_next_agent="case-memory",
            recommended_next_action="store_case_result",
        ),
        result=postprocess,
    )


def _run_case_memory_agent(input: CaseMemoryAgentInput) -> CaseMemoryAgentOutput:
    stored = store_case_result(
        StoreCaseResultInput(
            problem=input.problem,
            result=input.result,
            verification=input.verification,
            postprocess=input.postprocess,
        )
    )
    return CaseMemoryAgentOutput(
        handoff=_handoff(
            agent_name="case-memory",
            status="complete",
            problem_id=input.problem.id,
            summary=f"Stored case {stored.case_id} with features {', '.join(stored.indexed_features)}.",
            recommended_next_action="done",
        ),
        stored=stored,
    )


def run_physicsos_workflow(
    problem: PhysicsProblem | None = None,
    *,
    use_knowledge: bool = True,
    arxiv_max_results: int = 0,
    use_deepsearch: bool = False,
    taps_rank: int = 8,
    max_validation_attempts: int = 2,
) -> PhysicsOSWorkflowResult:
    """Run the local PhysicsOS orchestration loop.

    This deterministic workflow mirrors the intended DeepAgents tool sequence and
    gives tests a stable way to validate the architecture without calling an LLM.
    """
    trace: list[WorkflowStep] = []
    events: list[PhysicsOSEvent] = []
    run_id = f"workflow:{uuid4().hex}"
    problem = problem or build_default_thermal_problem()
    state = PhysicsOSWorkflowState(problem=problem, run_id=run_id)
    trace.append(WorkflowStep(name="problem", status="ready", summary=f"Using PhysicsProblem {problem.id}."))
    _emit_workflow_event(
        events,
        run_id=run_id,
        case_id=problem.id,
        event="workflow.started",
        stage="problem",
        status="ready",
        summary=f"Using PhysicsProblem {problem.id}.",
    )

    case_memory_context = build_case_memory_context(SearchCaseMemoryInput(problem=problem, top_k=3))
    state.case_memory_context = case_memory_context
    _emit_workflow_event(
        events,
        run_id=run_id,
        case_id=problem.id,
        event="case_memory.hit",
        stage="case-memory",
        status="complete",
        summary=f"Found {len(case_memory_context.hits)} similar prior cases.",
        payload=case_memory_context.model_dump(mode="json"),
    )
    _append_case_event(
        run_id=run_id,
        case_id=problem.id,
        stage="problem",
        event="workflow_problem_ready",
        summary=f"Workflow started for {problem.id}.",
        payload={"domain": problem.domain, "similar_cases": len(case_memory_context.hits)},
    )

    knowledge = None
    knowledge_context = None
    if use_knowledge:
        knowledge_input = KnowledgeAgentInput(
            problem=problem,
            query=f"{problem.domain} {' '.join(operator.equation_class for operator in problem.operators)} verification TAPS",
            arxiv_max_results=arxiv_max_results,
            use_deepsearch=use_deepsearch,
            case_memory_context=case_memory_context,
        )
        _emit_workflow_event(
            events,
            run_id=run_id,
            case_id=problem.id,
            event="agent.started",
            stage="knowledge",
            status="started",
            summary="knowledge-agent started.",
        )
        knowledge = _run_agent_with_retries(
            agent_name="knowledge-agent",
            stage="knowledge-agent",
            problem=problem,
            agent_input=knowledge_input,
            runner=_run_knowledge_agent,
            state=state,
            max_validation_attempts=max_validation_attempts,
            events=events,
            run_id=run_id,
        )
        if knowledge is None:
            return PhysicsOSWorkflowResult(
                state=state,
                run_id=run_id,
                problem=problem,
                case_memory_context=case_memory_context,
                validation_attempts=state.validation_attempts,
                trace=trace
                + [
                    WorkflowStep(
                        name="knowledge-agent",
                        status="retry_exhausted",
                        summary="knowledge-agent failed typed validation after retries.",
                    )
                ],
                events=events,
            )
        knowledge_context = knowledge.context
        state.knowledge = knowledge
        trace.append(
            WorkflowStep(name="knowledge-agent", status=knowledge.handoff.status, summary=knowledge.handoff.summary)
        )
        _emit_workflow_event(
            events,
            run_id=run_id,
            case_id=problem.id,
            event="agent.output",
            stage="knowledge",
            status=knowledge.handoff.status,
            summary=knowledge.handoff.summary,
            payload={"chunks": len(knowledge.context.chunks), "papers": len(knowledge.context.papers)},
        )
        _append_case_event(
            run_id=run_id,
            case_id=problem.id,
            stage="knowledge",
            event="agent_output",
            summary=knowledge.handoff.summary,
            payload={"chunks": len(knowledge.context.chunks), "papers": len(knowledge.context.papers)},
        )

    geometry_input = GeometryMeshAgentInput(
        problem=problem,
        requested_encodings=["mesh_graph"] if problem.geometry.dimension > 1 else [],
        target_backends=["taps"],
        case_memory_context=case_memory_context,
    )
    _emit_workflow_event(
        events,
        run_id=run_id,
        case_id=problem.id,
        event="agent.started",
        stage="geometry",
        status="started",
        summary="geometry-mesh-agent started.",
    )
    geometry = _run_agent_with_retries(
        agent_name="geometry-mesh-agent",
        stage="geometry-mesh-agent",
        problem=problem,
        agent_input=geometry_input,
        runner=_run_geometry_mesh_agent,
        state=state,
        max_validation_attempts=max_validation_attempts,
        events=events,
        run_id=run_id,
    )
    if geometry is None:
        return PhysicsOSWorkflowResult(
            state=state,
            run_id=run_id,
            problem=problem,
            case_memory_context=case_memory_context,
            knowledge=knowledge,
            knowledge_context=knowledge_context,
            validation_attempts=state.validation_attempts,
            trace=trace
            + [
                WorkflowStep(
                    name="geometry-mesh-agent",
                    status="retry_exhausted",
                    summary="geometry-mesh-agent failed typed validation after retries.",
                )
            ],
            events=events,
        )
    state.geometry = geometry
    problem = problem.model_copy(update={"geometry": geometry.geometry, "mesh": geometry.mesh})
    state.problem = problem
    trace.append(
        WorkflowStep(
            name="geometry-mesh-agent",
            status=geometry.handoff.status,
            summary=geometry.handoff.summary,
        )
    )
    _emit_workflow_event(
        events,
        run_id=run_id,
        case_id=problem.id,
        event="agent.output",
        stage="geometry",
        status=geometry.handoff.status,
        summary=geometry.handoff.summary,
        payload={"mesh_ready": geometry.mesh is not None, "encodings": len(geometry.encodings)},
    )
    _append_case_event(
        run_id=run_id,
        case_id=problem.id,
        stage="geometry",
        event="agent_output",
        summary=geometry.handoff.summary,
        payload={"mesh_ready": geometry.mesh is not None, "encodings": len(geometry.encodings)},
    )

    validation_ok, validation_attempts, validation_step = _validate_problem_with_retries(
        problem,
        max_validation_attempts=max_validation_attempts,
        stage="validate_physics_problem",
        input_context={
            "workflow_order": [
                "problem",
                "knowledge-agent",
                "geometry-mesh-agent",
                "validate_physics_problem",
                "taps-agent",
                "verification-agent",
                "postprocess-agent",
                "case-memory",
            ],
            "knowledge_available": knowledge is not None,
            "geometry_available": True,
        },
    )
    state.validation_attempts.extend(validation_attempts)
    trace.append(validation_step)
    _emit_workflow_event(
        events,
        run_id=run_id,
        case_id=problem.id,
        event="validation.retry" if validation_attempts else "agent.output",
        stage="validate",
        status=validation_step.status,
        summary=validation_step.summary,
        payload={"attempts": [attempt.model_dump(mode="json") for attempt in validation_attempts]},
    )
    if not validation_ok:
        return PhysicsOSWorkflowResult(
            state=state,
            run_id=run_id,
            problem=problem,
            case_memory_context=case_memory_context,
            geometry=geometry,
            knowledge=knowledge,
            knowledge_context=knowledge_context,
            validation_attempts=state.validation_attempts,
            trace=trace,
            events=events,
        )

    taps_input = TAPSAgentInput(
        problem=problem,
        knowledge_context=knowledge_context,
        case_memory_context=case_memory_context,
        tensor_rank=taps_rank,
    )
    _emit_workflow_event(
        events,
        run_id=run_id,
        case_id=problem.id,
        event="agent.started",
        stage="taps",
        status="started",
        summary="taps-agent started.",
    )
    taps = _run_agent_with_retries(
        agent_name="taps-agent",
        stage="taps-agent",
        problem=problem,
        agent_input=taps_input,
        runner=_run_taps_agent,
        state=state,
        max_validation_attempts=max_validation_attempts,
        events=events,
        run_id=run_id,
    )
    if taps is None:
        return PhysicsOSWorkflowResult(
            state=state,
            run_id=run_id,
            problem=problem,
            case_memory_context=case_memory_context,
            geometry=geometry,
            knowledge=knowledge,
            knowledge_context=knowledge_context,
            validation_attempts=state.validation_attempts,
            trace=trace
            + [
                WorkflowStep(
                    name="taps-agent",
                    status="retry_exhausted",
                    summary="taps-agent failed typed validation after retries.",
                )
            ],
            events=events,
        )
    state.taps = taps
    support = taps.support
    trace.append(
        WorkflowStep(
            name="taps-agent.support",
            status="supported" if support.supported else "fallback",
            summary=f"score={support.score}; risks={', '.join(support.risks) or 'none'}",
        )
    )
    _emit_workflow_event(
        events,
        run_id=run_id,
        case_id=problem.id,
        event="agent.output",
        stage="taps",
        status="supported" if support.supported else "fallback",
        summary=f"score={support.score}; risks={', '.join(support.risks) or 'none'}",
        payload=taps.model_dump(mode="json"),
    )
    _append_case_event(
        run_id=run_id,
        case_id=problem.id,
        stage="taps",
        event="support_check",
        summary=f"TAPS support score={support.score}.",
        payload={"supported": support.supported, "risks": support.risks},
    )

    taps_problem = None
    taps_residual = None
    if support.supported:
        if taps.taps_problem is None or taps.result is None:
            raise RuntimeError("taps-agent returned supported status without taps_problem/result")
        taps_problem = taps.taps_problem
        solver_result = taps.result
        verification_input = VerificationAgentInput(
            problem=problem,
            result=solver_result,
            taps_problem=taps_problem,
            case_memory_context=case_memory_context,
        )
        verification_agent = _run_agent_with_retries(
            agent_name="verification-agent",
            stage="verification-agent",
            problem=problem,
            agent_input=verification_input,
            runner=_run_verification_agent,
            state=state,
            max_validation_attempts=max_validation_attempts,
            events=events,
            run_id=run_id,
        )
        if verification_agent is None:
            return PhysicsOSWorkflowResult(
                state=state,
                run_id=run_id,
                problem=problem,
                case_memory_context=case_memory_context,
                geometry=geometry,
                knowledge=knowledge,
                taps=taps,
                solver_result=solver_result,
                knowledge_context=knowledge_context,
                taps_problem=taps_problem,
                validation_attempts=state.validation_attempts,
                trace=trace
                + [
                    WorkflowStep(
                        name="verification-agent",
                        status="retry_exhausted",
                        summary="verification-agent failed typed validation after retries.",
                    )
                ],
                events=events,
            )
        state.verification = verification_agent
        taps_residual = verification_agent.taps_residual
        verification = verification_agent.report
        trace.append(
            WorkflowStep(
                name="taps-agent.solve",
                status=solver_result.status,
                summary=f"backend={solver_result.backend}; action={taps_residual.recommended_action}",
            )
        )
        _emit_workflow_event(
            events,
            run_id=run_id,
            case_id=problem.id,
            event="agent.output",
            stage="taps.solve",
            status=solver_result.status,
            summary=f"backend={solver_result.backend}; action={taps_residual.recommended_action}",
            payload=solver_result.model_dump(mode="json"),
        )
    else:
        solver_input = SolverAgentInput(
            problem=problem,
            taps_handoff=taps.handoff,
            case_memory_context=case_memory_context,
        )
        solver = _run_agent_with_retries(
            agent_name="solver-agent",
            stage="solver-agent",
            problem=problem,
            agent_input=solver_input,
            runner=_run_solver_agent,
            state=state,
            max_validation_attempts=max_validation_attempts,
            events=events,
            run_id=run_id,
        )
        if solver is None:
            return PhysicsOSWorkflowResult(
                state=state,
                run_id=run_id,
                problem=problem,
                case_memory_context=case_memory_context,
                geometry=geometry,
                knowledge=knowledge,
                taps=taps,
                knowledge_context=knowledge_context,
                validation_attempts=state.validation_attempts,
                trace=trace
                + [
                    WorkflowStep(
                        name="solver-agent",
                        status="retry_exhausted",
                        summary="solver-agent failed typed validation after retries.",
                    )
                ],
                events=events,
            )
        state.solver = solver
        solver_result = solver.result
        verification_input = VerificationAgentInput(
            problem=problem,
            result=solver_result,
            case_memory_context=case_memory_context,
        )
        verification_agent = _run_agent_with_retries(
            agent_name="verification-agent",
            stage="verification-agent",
            problem=problem,
            agent_input=verification_input,
            runner=_run_verification_agent,
            state=state,
            max_validation_attempts=max_validation_attempts,
            events=events,
            run_id=run_id,
        )
        if verification_agent is None:
            return PhysicsOSWorkflowResult(
                state=state,
                run_id=run_id,
                problem=problem,
                case_memory_context=case_memory_context,
                geometry=geometry,
                knowledge=knowledge,
                taps=taps,
                solver=solver,
                solver_result=solver_result,
                knowledge_context=knowledge_context,
                validation_attempts=state.validation_attempts,
                trace=trace
                + [
                    WorkflowStep(
                        name="verification-agent",
                        status="retry_exhausted",
                        summary="verification-agent failed typed validation after retries.",
                    )
                ],
                events=events,
            )
        state.verification = verification_agent
        verification = verification_agent.report
        trace.append(
            WorkflowStep(
                name="solver-agent.surrogate_fallback",
                status=solver_result.status,
                summary=f"backend={solver_result.backend}; action={verification.recommended_next_action}",
            )
        )
        _emit_workflow_event(
            events,
            run_id=run_id,
            case_id=problem.id,
            event="agent.output",
            stage="solver",
            status=solver_result.status,
            summary=f"backend={solver_result.backend}; action={verification.recommended_next_action}",
            payload=solver_result.model_dump(mode="json"),
        )
    if support.supported:
        solver = SolverAgentOutput(
            handoff=taps.handoff,
            result=solver_result,
        )
        state.solver = solver

    _emit_workflow_event(
        events,
        run_id=run_id,
        case_id=problem.id,
        event="agent.output",
        stage="verification",
        status=verification.status,
        summary=verification.explanation,
        payload=verification.model_dump(mode="json"),
    )
    _append_case_event(
        run_id=run_id,
        case_id=problem.id,
        stage="verification",
        event="agent_output",
        summary=verification.explanation,
        payload={"status": verification.status, "recommended_next_action": verification.recommended_next_action},
    )

    postprocess_input = PostprocessAgentInput(
        problem=problem,
        result=solver_result,
        verification=verification,
        case_memory_context=case_memory_context,
    )
    postprocess_agent = _run_agent_with_retries(
        agent_name="postprocess-agent",
        stage="postprocess-agent",
        problem=problem,
        agent_input=postprocess_input,
        runner=_run_postprocess_agent,
        state=state,
        max_validation_attempts=max_validation_attempts,
        events=events,
        run_id=run_id,
    )
    if postprocess_agent is None:
        return PhysicsOSWorkflowResult(
            state=state,
            run_id=run_id,
            problem=problem,
            case_memory_context=case_memory_context,
            geometry=geometry,
            knowledge=knowledge,
            taps=taps,
            solver=solver,
            verification_agent=verification_agent,
            knowledge_context=knowledge_context,
            taps_problem=taps_problem,
            taps_residual=taps_residual,
            solver_result=solver_result,
            verification=verification,
            validation_attempts=state.validation_attempts,
            trace=trace
            + [
                WorkflowStep(
                    name="postprocess-agent",
                    status="retry_exhausted",
                    summary="postprocess-agent failed typed validation after retries.",
                )
            ],
            events=events,
        )
    state.postprocess = postprocess_agent
    postprocess = postprocess_agent.result
    trace.append(
        WorkflowStep(
            name="postprocess-agent",
            status=postprocess_agent.handoff.status,
            summary=postprocess_agent.handoff.summary,
        )
    )
    _emit_workflow_event(
        events,
        run_id=run_id,
        case_id=problem.id,
        event="agent.output",
        stage="postprocess",
        status=postprocess_agent.handoff.status,
        summary=postprocess_agent.handoff.summary,
        payload=postprocess.model_dump(mode="json"),
    )

    case_memory_input = CaseMemoryAgentInput(
        problem=problem,
        result=solver_result,
        verification=verification,
        postprocess=postprocess,
        case_memory_context=case_memory_context,
    )
    case_memory = _run_agent_with_retries(
        agent_name="case-memory",
        stage="case-memory",
        problem=problem,
        agent_input=case_memory_input,
        runner=_run_case_memory_agent,
        state=state,
        max_validation_attempts=max_validation_attempts,
        events=events,
        run_id=run_id,
    )
    if case_memory is None:
        return PhysicsOSWorkflowResult(
            state=state,
            run_id=run_id,
            problem=problem,
            case_memory_context=case_memory_context,
            geometry=geometry,
            knowledge=knowledge,
            taps=taps,
            solver=solver,
            verification_agent=verification_agent,
            postprocess_agent=postprocess_agent,
            knowledge_context=knowledge_context,
            taps_problem=taps_problem,
            taps_residual=taps_residual,
            solver_result=solver_result,
            verification=verification,
            postprocess=postprocess,
            validation_attempts=state.validation_attempts,
            trace=trace
            + [
                WorkflowStep(
                    name="case-memory",
                    status="retry_exhausted",
                    summary="case-memory failed typed validation after retries.",
                )
            ],
            events=events,
        )
    state.case_memory = case_memory
    case_store = case_memory.stored
    trace.append(
        WorkflowStep(
            name="case-memory",
            status=case_memory.handoff.status,
            summary=case_memory.handoff.summary,
        )
    )
    _emit_workflow_event(
        events,
        run_id=run_id,
        case_id=problem.id,
        event="workflow.completed",
        stage="case-memory",
        status=case_memory.handoff.status,
        summary=case_memory.handoff.summary,
        payload=case_store.model_dump(mode="json"),
    )
    _append_case_event(
        run_id=run_id,
        case_id=problem.id,
        stage="case-memory",
        event="commit_final_case",
        summary=case_memory.handoff.summary,
        payload=case_store.model_dump(mode="json"),
    )

    return PhysicsOSWorkflowResult(
        state=state,
        run_id=run_id,
        problem=problem,
        case_memory_context=case_memory_context,
        geometry=geometry,
        knowledge=knowledge,
        taps=taps,
        solver=solver,
        verification_agent=verification_agent,
        postprocess_agent=postprocess_agent,
        case_memory=case_memory,
        knowledge_context=knowledge_context,
        taps_problem=taps_problem,
        taps_residual=taps_residual,
        solver_result=solver_result,
        verification=verification,
        postprocess=postprocess,
        case_store=case_store,
        validation_attempts=state.validation_attempts,
        trace=trace,
        events=events,
    )
