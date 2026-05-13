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
    prepare_mesh_conversion_job,
    repair_geometry,
    submit_mesh_conversion_job,
)
from physicsos.tools.knowledge_tools import build_knowledge_context, ingest_knowledge_document, run_deepsearch, search_arxiv, search_knowledge_base
from physicsos.tools.ks_dft_verification_tools import (
    check_ks_band_dos_provenance,
    check_ks_charge_conservation,
    check_ks_hamiltonian_evidence,
    check_ks_material_artifact_usage,
    check_ks_molecular_context_evidence,
    check_ks_orthonormality,
    check_ks_poisson_residual,
    check_ks_rank_grid_kpoint_convergence,
    check_ks_scf_residual,
)
from physicsos.tools.ks_dft_taps_tools import compile_ks_dft_taps_kernel, plan_lrdm_scf_acceleration, prepare_gamma_only_ks_dft_taps_kernel, prepare_ks_dft_multik_integration_policy, prepare_ks_dft_task_assumptions, prepare_ks_dft_xc_policy, prepare_toy_ks_dft_taps_kernel, prepare_verified_ks_dft_band_dos_preflight
from physicsos.tools.materials_tools import (
    analyze_spacegroup,
    build_taps_kpoint_axis,
    compare_crystal_structures,
    compute_reciprocal_lattice,
    generate_pymatgen_highsymm_kpath,
    generate_seekpath_kpath,
    generate_structure_parameter_axis,
    generate_uniform_kmesh,
    make_supercell_structure,
    map_site_properties,
    parse_molecular_structure,
    parse_material_structure,
    prepare_ks_dft_molecular_context,
    prepare_ks_dft_taps_material_context,
    prepare_molecular_taps_scaling_policy,
    reduce_irreducible_kpoints,
    reduce_lattice_cell,
    review_ks_dft_material_context,
    sample_kpath_segments,
    standardize_crystal_structure,
    validate_material_structure,
    write_material_structure,
)
from physicsos.tools.memory_tools import append_case_memory_event, read_case_memory_events, search_case_memory, store_case_result
from physicsos.tools.postprocess_tools import (
    extract_kpis,
    generate_visualizations,
    plan_postprocess,
    write_simulation_report,
)
from physicsos.tools.problem_tools import canonicalize_physics_problem, validate_physics_problem
from physicsos.tools.pseudopotential_tools import index_vasp_paw_pbe_library, select_pseudopotentials_for_structure, validate_local_pseudopotential_artifact, validate_nonlocal_projector_artifact
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

MATERIALS_TOOLS = [
    parse_material_structure,
    write_material_structure,
    validate_material_structure,
    analyze_spacegroup,
    standardize_crystal_structure,
    compare_crystal_structures,
    reduce_lattice_cell,
    compute_reciprocal_lattice,
    generate_uniform_kmesh,
    reduce_irreducible_kpoints,
    generate_seekpath_kpath,
    generate_pymatgen_highsymm_kpath,
    sample_kpath_segments,
    build_taps_kpoint_axis,
    make_supercell_structure,
    generate_structure_parameter_axis,
    map_site_properties,
    parse_molecular_structure,
    prepare_ks_dft_molecular_context,
    prepare_molecular_taps_scaling_policy,
    prepare_ks_dft_taps_material_context,
    review_ks_dft_material_context,
]

PSEUDOPOTENTIAL_TOOLS = [
    index_vasp_paw_pbe_library,
    select_pseudopotentials_for_structure,
    validate_local_pseudopotential_artifact,
    validate_nonlocal_projector_artifact,
]

KS_DFT_VERIFICATION_TOOLS = [
    check_ks_charge_conservation,
    check_ks_orthonormality,
    check_ks_scf_residual,
    check_ks_poisson_residual,
    check_ks_rank_grid_kpoint_convergence,
    check_ks_hamiltonian_evidence,
    check_ks_band_dos_provenance,
    check_ks_material_artifact_usage,
    check_ks_molecular_context_evidence,
]

KS_DFT_TAPS_TOOLS = [
    compile_ks_dft_taps_kernel,
    prepare_toy_ks_dft_taps_kernel,
    prepare_gamma_only_ks_dft_taps_kernel,
    prepare_ks_dft_multik_integration_policy,
    prepare_verified_ks_dft_band_dos_preflight,
    plan_lrdm_scf_acceleration,
    prepare_ks_dft_xc_policy,
    prepare_ks_dft_task_assumptions,
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
)

PHYSICSOS_WORKFLOW_NODE_CAPABILITIES = _unique_tools(
    GEOMETRY_MESH_TOOLS,
    TAPS_TOOLS,
    VERIFICATION_TOOLS,
    POSTPROCESS_TOOLS,
    KNOWLEDGE_TOOLS,
)

DEEPAGENTS_SUBAGENT_TOOL_GROUPS = {
    "analysis-file-agent": _unique_tools(
        [update_case_stage_status],
        [search_knowledge_base, build_knowledge_context, search_case_memory, read_case_memory_events],
    ),
    "geometry-embedding-agent": _unique_tools(
        [update_case_stage_status],
        GEOMETRY_EMBEDDING_TOOLS,
        [search_knowledge_base, build_knowledge_context],
    ),
    "materials-preprocess-agent": _unique_tools(
        [update_case_stage_status],
        MATERIALS_TOOLS,
        PSEUDOPOTENTIAL_TOOLS,
        [search_knowledge_base, build_knowledge_context],
    ),
    "ks-dft-analysis-agent": _unique_tools(
        [update_case_stage_status],
        [
            review_ks_dft_material_context,
            prepare_ks_dft_molecular_context,
            prepare_molecular_taps_scaling_policy,
            select_pseudopotentials_for_structure,
            build_knowledge_context,
            search_knowledge_base,
        ],
    ),
    "ks-dft-taps-derivation-agent": _unique_tools(
        [update_case_stage_status, build_taps_derivation_prompt],
        [
            review_ks_dft_material_context,
            prepare_ks_dft_molecular_context,
            prepare_molecular_taps_scaling_policy,
            select_pseudopotentials_for_structure,
            build_knowledge_context,
            search_knowledge_base,
            validate_local_pseudopotential_artifact,
            validate_nonlocal_projector_artifact,
            prepare_ks_dft_xc_policy,
            prepare_ks_dft_task_assumptions,
            formulate_taps_equation,
            build_taps_problem,
        ],
    ),
    "ks-dft-taps-implementation-agent": _unique_tools(
        [update_case_stage_status],
        [
            review_ks_dft_material_context,
            prepare_molecular_taps_scaling_policy,
            build_knowledge_context,
            formulate_taps_equation,
            build_taps_problem,
            select_pseudopotentials_for_structure,
            validate_local_pseudopotential_artifact,
            validate_nonlocal_projector_artifact,
            compile_ks_dft_taps_kernel,
            prepare_ks_dft_multik_integration_policy,
            prepare_verified_ks_dft_band_dos_preflight,
            plan_lrdm_scf_acceleration,
            prepare_ks_dft_xc_policy,
            prepare_ks_dft_task_assumptions,
            compile_taps_kernel,
            static_check_generated_kernel,
            review_generated_taps_kernel,
            execute_taps_kernel,
        ],
    ),
    "ks-dft-verification-agent": _unique_tools(
        [update_case_stage_status],
        [
            review_ks_dft_material_context,
            prepare_ks_dft_molecular_context,
            check_ks_charge_conservation,
            check_ks_orthonormality,
            check_ks_scf_residual,
            check_ks_poisson_residual,
            check_ks_rank_grid_kpoint_convergence,
            check_ks_hamiltonian_evidence,
            check_ks_band_dos_provenance,
            check_ks_material_artifact_usage,
            check_ks_molecular_context_evidence,
            prepare_ks_dft_multik_integration_policy,
            prepare_verified_ks_dft_band_dos_preflight,
            plan_lrdm_scf_acceleration,
        ],
    ),
    "taps-derivation-agent": _unique_tools(
        [update_case_stage_status, build_taps_derivation_prompt],
        [
            build_knowledge_context,
            search_knowledge_base,
            assess_taps_geometry_separability,
            formulate_taps_equation,
            build_taps_problem,
        ],
    ),
    "taps-implementation-agent": _unique_tools(
        [update_case_stage_status],
        [
            build_knowledge_context,
            formulate_taps_equation,
            build_taps_problem,
            compile_taps_kernel,
            static_check_generated_kernel,
            review_generated_taps_kernel,
            execute_taps_kernel,
        ],
    ),
    "verification-agent": _unique_tools(
        [update_case_stage_status],
        [
            generate_exact_sol_code,
            execute_exact_sol_code,
            generate_convergence_code,
            execute_convergence_code,
            plot_result,
        ],
    ),
    "postprocess-agent": _unique_tools(
        [update_case_stage_status],
        [
            extract_kpis,
            generate_visualizations,
            write_simulation_report,
        ],
    ),
    "knowledge-agent": _unique_tools(
        [
            load_taps_case_references,
            build_paper_context_window,
            build_knowledge_context,
            search_knowledge_base,
            search_case_memory,
            append_case_memory_event,
            read_case_memory_events,
            store_case_result,
        ],
        [search_arxiv, run_deepsearch, ingest_knowledge_document],
    ),
}


PHYSICSOS_REGISTRY_TOOLS = _unique_tools(
    DEEPAGENTS_MAIN_BRIDGE_TOOLS,
    PHYSICSOS_PROBLEM_PREP_TOOLS,
    PHYSICSOS_CATALOG_TOOLS,
    MATERIALS_TOOLS,
    PSEUDOPOTENTIAL_TOOLS,
    KS_DFT_TAPS_TOOLS,
    KS_DFT_VERIFICATION_TOOLS,
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
