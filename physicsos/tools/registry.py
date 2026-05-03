from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from physicsos.tools.catalog_tools import (
    list_operator_templates,
    list_postprocess_templates,
    list_solver_backends,
    list_verification_rules,
    recommend_runtime_stack,
)
from physicsos.tools.geometry_tools import (
    apply_boundary_labels,
    apply_boundary_labeling_artifact,
    assess_mesh_quality,
    create_boundary_labeling_artifact,
    create_geometry_labeler_viewer,
    export_backend_mesh,
    generate_geometry_encoding,
    generate_mesh,
    import_geometry,
    label_regions,
    plan_geometry_mesh,
    plan_geometry_mesh_structured,
    prepare_mesh_conversion_job,
    repair_geometry,
    submit_mesh_conversion_job,
)
from physicsos.tools.knowledge_tools import build_knowledge_context, ingest_knowledge_document, run_deepsearch, search_arxiv, search_knowledge_base
from physicsos.tools.memory_tools import append_case_memory_event, read_case_memory_events, search_case_memory, store_case_result
from physicsos.tools.postprocess_tools import (
    extract_kpis,
    generate_visualizations,
    plan_postprocess,
    plan_postprocess_structured,
    write_simulation_report,
)
from physicsos.tools.problem_tools import build_physics_problem, canonicalize_physics_problem, validate_physics_problem
from physicsos.tools.solver_tools import estimate_solver_support, prepare_full_solver_case, prepare_openfoam_runner_manifest, route_solver_backend, run_full_solver, run_hybrid_solver, run_surrogate_solver, submit_full_solver_job
from physicsos.tools.surrogate_tools import (
    estimate_surrogate_support,
    list_available_surrogates,
    route_surrogate_model,
    run_surrogate_inference,
)
from physicsos.tools.taps_tools import author_taps_runtime_extension, build_taps_problem, estimate_taps_residual, estimate_taps_support, export_taps_backend_bridge, formulate_taps_equation, plan_backend_preparation, plan_backend_preparation_structured, plan_numerical_solve, plan_numerical_solve_structured, plan_taps_adaptive_fallback, prepare_taps_backend_case_bundle, run_taps_backend, validate_backend_preparation_plan, validate_numerical_solve_plan, validate_taps_ir
from physicsos.tools.verification_tools import check_boundary_condition_application, check_conservation_laws, compute_physics_residuals, detect_ood_case, estimate_uncertainty, validate_selected_slices
from physicsos.tools.workflow_tools import run_typed_physicsos_workflow


@dataclass(frozen=True)
class ToolSpec:
    name: str
    function: Callable[..., Any]
    input_model: type[BaseModel] | None
    output_model: type[BaseModel] | None
    side_effects: str
    requires_approval: bool = False


MAIN_AGENT_TOOLS = [
    build_physics_problem,
    canonicalize_physics_problem,
    validate_physics_problem,
    run_typed_physicsos_workflow,
    list_operator_templates,
    list_solver_backends,
    list_verification_rules,
    list_postprocess_templates,
    recommend_runtime_stack,
]

GEOMETRY_MESH_TOOLS = [
    import_geometry,
    repair_geometry,
    label_regions,
    plan_geometry_mesh,
    plan_geometry_mesh_structured,
    apply_boundary_labels,
    create_boundary_labeling_artifact,
    apply_boundary_labeling_artifact,
    create_geometry_labeler_viewer,
    generate_geometry_encoding,
    generate_mesh,
    export_backend_mesh,
    prepare_mesh_conversion_job,
    submit_mesh_conversion_job,
    assess_mesh_quality,
]

TAPS_TOOLS = [
    validate_physics_problem,
    canonicalize_physics_problem,
    estimate_solver_support,
    estimate_taps_support,
    build_knowledge_context,
    search_knowledge_base,
    formulate_taps_equation,
    build_taps_problem,
    validate_taps_ir,
    plan_numerical_solve,
    plan_numerical_solve_structured,
    validate_numerical_solve_plan,
    plan_backend_preparation,
    plan_backend_preparation_structured,
    validate_backend_preparation_plan,
    export_taps_backend_bridge,
    plan_taps_adaptive_fallback,
    prepare_taps_backend_case_bundle,
    author_taps_runtime_extension,
    run_taps_backend,
    estimate_taps_residual,
]

SOLVER_TOOLS = [
    validate_physics_problem,
    list_available_surrogates,
    estimate_surrogate_support,
    route_surrogate_model,
    run_surrogate_inference,
    estimate_solver_support,
    route_solver_backend,
    prepare_openfoam_runner_manifest,
    run_surrogate_solver,
    prepare_full_solver_case,
    submit_full_solver_job,
    run_full_solver,
    run_hybrid_solver,
]

VERIFICATION_TOOLS = [
    compute_physics_residuals,
    check_boundary_condition_application,
    check_conservation_laws,
    validate_selected_slices,
    estimate_uncertainty,
    detect_ood_case,
]

POSTPROCESS_TOOLS = [
    extract_kpis,
    plan_postprocess,
    plan_postprocess_structured,
    generate_visualizations,
    write_simulation_report,
]

KNOWLEDGE_TOOLS = [
    search_arxiv,
    run_deepsearch,
    ingest_knowledge_document,
    search_knowledge_base,
    build_knowledge_context,
    search_case_memory,
    append_case_memory_event,
    read_case_memory_events,
    store_case_result,
]

SHARED_KNOWLEDGE_TOOLS = [
    search_knowledge_base,
    build_knowledge_context,
    search_case_memory,
    append_case_memory_event,
    read_case_memory_events,
]

def _unique_tools(*groups: list[Callable[..., Any]]) -> list[Callable[..., Any]]:
    seen: set[str] = set()
    tools: list[Callable[..., Any]] = []
    for group in groups:
        for tool in group:
            name = tool.__name__
            if name not in seen:
                tools.append(tool)
                seen.add(name)
    return tools


SUBAGENT_TOOL_GROUPS = {
    "geometry-mesh-agent": _unique_tools(MAIN_AGENT_TOOLS, SHARED_KNOWLEDGE_TOOLS, GEOMETRY_MESH_TOOLS),
    "taps-agent": _unique_tools(MAIN_AGENT_TOOLS, SHARED_KNOWLEDGE_TOOLS, TAPS_TOOLS),
    "solver-agent": _unique_tools(MAIN_AGENT_TOOLS, SHARED_KNOWLEDGE_TOOLS, SOLVER_TOOLS),
    "verification-agent": _unique_tools(MAIN_AGENT_TOOLS, SHARED_KNOWLEDGE_TOOLS, VERIFICATION_TOOLS),
    "postprocess-agent": _unique_tools(MAIN_AGENT_TOOLS, SHARED_KNOWLEDGE_TOOLS, POSTPROCESS_TOOLS),
    "knowledge-agent": _unique_tools(MAIN_AGENT_TOOLS, KNOWLEDGE_TOOLS),
}


PHYSICSOS_TOOLS = _unique_tools(
    MAIN_AGENT_TOOLS,
    GEOMETRY_MESH_TOOLS,
    TAPS_TOOLS,
    SOLVER_TOOLS,
    VERIFICATION_TOOLS,
    POSTPROCESS_TOOLS,
    KNOWLEDGE_TOOLS,
)

TOOL_REGISTRY: dict[str, ToolSpec] = {
    tool.__name__: ToolSpec(
        name=tool.__name__,
        function=tool,
        input_model=getattr(tool, "input_model", None),
        output_model=getattr(tool, "output_model", None),
        side_effects=getattr(tool, "side_effects", "none"),
        requires_approval=getattr(tool, "requires_approval", False),
    )
    for tool in PHYSICSOS_TOOLS
}
