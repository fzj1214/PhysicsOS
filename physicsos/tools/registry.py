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
    assess_mesh_quality,
    generate_geometry_encoding,
    generate_mesh,
    import_geometry,
    label_regions,
    repair_geometry,
)
from physicsos.tools.knowledge_tools import build_knowledge_context, ingest_knowledge_document, run_deepsearch, search_arxiv, search_knowledge_base
from physicsos.tools.memory_tools import search_case_memory, store_case_result
from physicsos.tools.postprocess_tools import extract_kpis, generate_visualizations, write_simulation_report
from physicsos.tools.problem_tools import build_physics_problem, validate_physics_problem
from physicsos.tools.solver_tools import estimate_solver_support, prepare_full_solver_case, route_solver_backend, run_full_solver, run_hybrid_solver, run_surrogate_solver, submit_full_solver_job
from physicsos.tools.surrogate_tools import (
    estimate_surrogate_support,
    list_available_surrogates,
    route_surrogate_model,
    run_surrogate_inference,
)
from physicsos.tools.taps_tools import author_taps_runtime_extension, build_taps_problem, estimate_taps_residual, estimate_taps_support, formulate_taps_equation, run_taps_backend
from physicsos.tools.verification_tools import check_conservation_laws, compute_physics_residuals, detect_ood_case, estimate_uncertainty, validate_selected_slices


@dataclass(frozen=True)
class ToolSpec:
    name: str
    function: Callable[..., Any]
    input_model: type[BaseModel] | None
    output_model: type[BaseModel] | None
    side_effects: str
    requires_approval: bool = False


PHYSICSOS_TOOLS = [
    build_physics_problem,
    validate_physics_problem,
    list_operator_templates,
    list_solver_backends,
    list_verification_rules,
    list_postprocess_templates,
    recommend_runtime_stack,
    import_geometry,
    repair_geometry,
    label_regions,
    apply_boundary_labels,
    generate_geometry_encoding,
    generate_mesh,
    assess_mesh_quality,
    estimate_solver_support,
    route_solver_backend,
    estimate_taps_support,
    formulate_taps_equation,
    build_taps_problem,
    author_taps_runtime_extension,
    run_taps_backend,
    estimate_taps_residual,
    list_available_surrogates,
    estimate_surrogate_support,
    route_surrogate_model,
    run_surrogate_inference,
    run_surrogate_solver,
    prepare_full_solver_case,
    submit_full_solver_job,
    run_full_solver,
    run_hybrid_solver,
    compute_physics_residuals,
    check_conservation_laws,
    validate_selected_slices,
    estimate_uncertainty,
    detect_ood_case,
    extract_kpis,
    generate_visualizations,
    write_simulation_report,
    search_arxiv,
    run_deepsearch,
    ingest_knowledge_document,
    search_knowledge_base,
    build_knowledge_context,
    search_case_memory,
    store_case_result,
]

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
