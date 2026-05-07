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
from physicsos.tools.case_tools import build_paper_context_window, build_taps_derivation_prompt, create_case_workspace, load_taps_case_references, update_case_stage_status
from physicsos.tools.geometry_embedding_tools import (
    build_geometry_embedding,
    execute_gmsh_distance_field,
    generate_background_grid,
    generate_gmsh_distance_field,
    generate_primitive_geometry,
    import_stl_geometry,
    prepare_geometry_analysis_files,
    voxelize_geometry,
)
from physicsos.tools.geometry_tools import (
    apply_boundary_labels,
    apply_boundary_labeling_artifact,
    assess_mesh_quality,
    build_geometry_mesh_contract,
    create_boundary_labeling_artifact,
    create_geometry_labeler_viewer,
    export_backend_mesh,
    generate_boundary_region_candidates,
    generate_geometry_encoding,
    generate_mesh,
    import_geometry,
    label_regions,
    mesh_semantics_gate,
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
from physicsos.tools.problem_tools import canonicalize_physics_problem, validate_physics_problem
from physicsos.tools.taps_tools import assess_taps_geometry_separability, author_taps_runtime_extension, build_taps_problem, compile_taps_kernel, estimate_taps_residual, estimate_taps_support, estimate_taps_support_structured, execute_taps_kernel, formulate_taps_equation, prepare_taps_backend_case_bundle, review_generated_taps_kernel, static_check_generated_kernel
from physicsos.tools.verification_tools import check_boundary_condition_application, check_conservation_laws, compute_physics_residuals, detect_ood_case, estimate_uncertainty, validate_selected_slices
from physicsos.tools.verification_chain_tools import (
    execute_convergence_code,
    execute_exact_sol_code,
    generate_convergence_code,
    generate_exact_sol_code,
    plot_result,
)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    function: Callable[..., Any]
    input_model: type[BaseModel] | None
    output_model: type[BaseModel] | None
    side_effects: str
    requires_approval: bool = False


PHYSICSOS_PROBLEM_PREP_TOOLS = [
    canonicalize_physics_problem,
    validate_physics_problem,
    recommend_runtime_stack,
]

PHYSICSOS_CATALOG_TOOLS = [
    list_operator_templates,
    list_solver_backends,
    list_verification_rules,
    list_postprocess_templates,
]

PHYSICSOS_CASE_TOOLS = [
    create_case_workspace,
    update_case_stage_status,
    load_taps_case_references,
    build_paper_context_window,
    build_taps_derivation_prompt,
]

GEOMETRY_EMBEDDING_TOOLS = [
    prepare_geometry_analysis_files,
    generate_primitive_geometry,
    import_stl_geometry,
    generate_gmsh_distance_field,
    execute_gmsh_distance_field,
    generate_background_grid,
    voxelize_geometry,
    build_geometry_embedding,
]

GEOMETRY_MESH_TOOLS = [
    *GEOMETRY_EMBEDDING_TOOLS,
    import_geometry,
    repair_geometry,
    label_regions,
    build_geometry_mesh_contract,
    mesh_semantics_gate,
    plan_geometry_mesh,
    plan_geometry_mesh_structured,
    apply_boundary_labels,
    generate_boundary_region_candidates,
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
    estimate_taps_support,
    estimate_taps_support_structured,
    build_knowledge_context,
    search_knowledge_base,
    assess_taps_geometry_separability,
    formulate_taps_equation,
    build_taps_problem,
    prepare_taps_backend_case_bundle,
    author_taps_runtime_extension,
    compile_taps_kernel,
    static_check_generated_kernel,
    review_generated_taps_kernel,
    execute_taps_kernel,
    estimate_taps_residual,
]

VERIFICATION_TOOLS = [
    generate_exact_sol_code,
    execute_exact_sol_code,
    generate_convergence_code,
    execute_convergence_code,
    plot_result,
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


DEEPAGENTS_MAIN_BRIDGE_TOOLS = _unique_tools(
    PHYSICSOS_CASE_TOOLS,
    PHYSICSOS_CATALOG_TOOLS,
)

PHYSICSOS_WORKFLOW_NODE_CAPABILITIES = _unique_tools(
    GEOMETRY_MESH_TOOLS,
    TAPS_TOOLS,
    VERIFICATION_TOOLS,
    POSTPROCESS_TOOLS,
    KNOWLEDGE_TOOLS,
)

DEEPAGENTS_SUBAGENT_TOOL_GROUPS = {
    "analysis-file-agent": _unique_tools(DEEPAGENTS_MAIN_BRIDGE_TOOLS, SHARED_KNOWLEDGE_TOOLS),
    "geometry-embedding-agent": _unique_tools(DEEPAGENTS_MAIN_BRIDGE_TOOLS, SHARED_KNOWLEDGE_TOOLS, GEOMETRY_MESH_TOOLS),
    "taps-derivation-agent": _unique_tools(DEEPAGENTS_MAIN_BRIDGE_TOOLS, SHARED_KNOWLEDGE_TOOLS, TAPS_TOOLS),
    "taps-implementation-agent": _unique_tools(DEEPAGENTS_MAIN_BRIDGE_TOOLS, SHARED_KNOWLEDGE_TOOLS, TAPS_TOOLS),
    "verification-agent": _unique_tools(DEEPAGENTS_MAIN_BRIDGE_TOOLS, SHARED_KNOWLEDGE_TOOLS, VERIFICATION_TOOLS),
    "postprocess-agent": _unique_tools(DEEPAGENTS_MAIN_BRIDGE_TOOLS, SHARED_KNOWLEDGE_TOOLS, POSTPROCESS_TOOLS),
    "knowledge-agent": _unique_tools(DEEPAGENTS_MAIN_BRIDGE_TOOLS, KNOWLEDGE_TOOLS),
}


PHYSICSOS_REGISTRY_TOOLS = _unique_tools(
    DEEPAGENTS_MAIN_BRIDGE_TOOLS,
    PHYSICSOS_PROBLEM_PREP_TOOLS,
    PHYSICSOS_WORKFLOW_NODE_CAPABILITIES,
)

# Compatibility aliases for callers that import the aggregate tool surfaces.
MAIN_AGENT_TOOLS = DEEPAGENTS_MAIN_BRIDGE_TOOLS
SUBAGENT_TOOL_GROUPS = DEEPAGENTS_SUBAGENT_TOOL_GROUPS
PHYSICSOS_TOOLS = PHYSICSOS_REGISTRY_TOOLS

TOOL_REGISTRY: dict[str, ToolSpec] = {
    tool.__name__: ToolSpec(
        name=tool.__name__,
        function=tool,
        input_model=getattr(tool, "input_model", None),
        output_model=getattr(tool, "output_model", None),
        side_effects=getattr(tool, "side_effects", "none"),
        requires_approval=getattr(tool, "requires_approval", False),
    )
    for tool in PHYSICSOS_REGISTRY_TOOLS
}
