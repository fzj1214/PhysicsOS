from __future__ import annotations

from pydantic import Field

from physicsos.schemas.common import StrictBaseModel
from physicsos.schemas.knowledge import KnowledgeContext
from physicsos.schemas.postprocess import PostprocessResult
from physicsos.schemas.problem import PhysicsProblem
from physicsos.schemas.solver import SolverResult
from physicsos.schemas.taps import TAPSProblem, TAPSResidualReport, TAPSBasisConfig
from physicsos.schemas.verification import VerificationReport
from physicsos.tools.knowledge_tools import BuildKnowledgeContextInput, build_knowledge_context
from physicsos.tools.memory_tools import StoreCaseResultInput, StoreCaseResultOutput, store_case_result
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


class WorkflowStep(StrictBaseModel):
    name: str
    status: str
    summary: str


class PhysicsOSWorkflowResult(StrictBaseModel):
    problem: PhysicsProblem
    knowledge_context: KnowledgeContext | None = None
    taps_problem: TAPSProblem | None = None
    taps_residual: TAPSResidualReport | None = None
    solver_result: SolverResult
    verification: VerificationReport
    postprocess: PostprocessResult
    case_store: StoreCaseResultOutput
    trace: list[WorkflowStep] = Field(default_factory=list)


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


def run_physicsos_workflow(
    problem: PhysicsProblem | None = None,
    *,
    use_knowledge: bool = True,
    arxiv_max_results: int = 0,
    use_deepsearch: bool = False,
    taps_rank: int = 8,
) -> PhysicsOSWorkflowResult:
    """Run the local PhysicsOS orchestration loop.

    This deterministic workflow mirrors the intended DeepAgents tool sequence and
    gives tests a stable way to validate the architecture without calling an LLM.
    """
    trace: list[WorkflowStep] = []
    problem = problem or build_default_thermal_problem()
    trace.append(WorkflowStep(name="problem", status="ready", summary=f"Using PhysicsProblem {problem.id}."))

    knowledge_context = None
    if use_knowledge:
        knowledge_context = build_knowledge_context(
            BuildKnowledgeContextInput(
                query=f"{problem.domain} {' '.join(operator.equation_class for operator in problem.operators)} verification TAPS",
                local_top_k=4,
                arxiv_max_results=arxiv_max_results,
                use_deepsearch=use_deepsearch,
            )
        ).context
        trace.append(
            WorkflowStep(
                name="knowledge-agent",
                status="complete",
                summary=f"Retrieved {len(knowledge_context.chunks)} local chunks and {len(knowledge_context.papers)} papers.",
            )
        )

    validation = validate_physics_problem(ValidatePhysicsProblemInput(problem=problem))
    trace.append(
        WorkflowStep(
            name="validate_physics_problem",
            status="accepted" if validation.valid else "rejected",
            summary="; ".join(validation.errors + validation.warnings) or "Problem is solver-ready.",
        )
    )
    if not validation.valid:
        raise ValueError(f"PhysicsProblem is not solver-ready: {validation.errors}")

    support = estimate_taps_support(EstimateTAPSSupportInput(problem=problem)).support
    trace.append(
        WorkflowStep(
            name="taps-agent.support",
            status="supported" if support.supported else "fallback",
            summary=f"score={support.score}; risks={', '.join(support.risks) or 'none'}",
        )
    )

    taps_problem = None
    taps_residual = None
    if support.supported:
        taps_problem = build_taps_problem(
            BuildTAPSProblemInput(problem=problem, basis=TAPSBasisConfig(tensor_rank=taps_rank, reproducing_order=2))
        ).taps_problem
        solver_result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result
        taps_residual, verification = _verify_taps_result(problem, taps_problem, solver_result)
        trace.append(
            WorkflowStep(
                name="taps-agent.solve",
                status=solver_result.status,
                summary=f"backend={solver_result.backend}; action={taps_residual.recommended_action}",
            )
        )
    else:
        solver_result = run_surrogate_solver(RunSurrogateSolverInput(problem=problem, backend="")).result
        verification = _verify_fallback_result(problem, solver_result)
        trace.append(
            WorkflowStep(
                name="solver-agent.surrogate_fallback",
                status=solver_result.status,
                summary=f"backend={solver_result.backend}; action={verification.recommended_next_action}",
            )
        )

    kpis = extract_kpis(ExtractKPIsInput(problem=problem, result=solver_result))
    visualizations = generate_visualizations(GenerateVisualizationsInput(problem=problem, result=solver_result))
    postprocess = PostprocessResult(
        problem_id=problem.id,
        result_id=solver_result.id,
        kpis=kpis.kpis,
        units=kpis.units,
        visualizations=visualizations.artifacts,
        recommendations=[verification.explanation],
    )
    report = write_simulation_report(
        WriteSimulationReportInput(problem=problem, result=solver_result, verification=verification, postprocess=postprocess)
    )
    postprocess.report = report.report
    if report.manifest is not None:
        postprocess.visualizations.append(report.manifest)
    trace.append(WorkflowStep(name="postprocess-agent", status="complete", summary=f"Report written to {report.report.uri}."))

    case_store = store_case_result(StoreCaseResultInput(problem=problem, result=solver_result, verification=verification, postprocess=postprocess))
    trace.append(
        WorkflowStep(
            name="case-memory",
            status="complete",
            summary=f"Stored case {case_store.case_id} with features {', '.join(case_store.indexed_features)}.",
        )
    )

    return PhysicsOSWorkflowResult(
        problem=problem,
        knowledge_context=knowledge_context,
        taps_problem=taps_problem,
        taps_residual=taps_residual,
        solver_result=solver_result,
        verification=verification,
        postprocess=postprocess,
        case_store=case_store,
        trace=trace,
    )
