from __future__ import annotations

from physicsos.agents.prompts import (
    GEOMETRY_MESH_AGENT_PROMPT,
    KNOWLEDGE_AGENT_PROMPT,
    POSTPROCESS_AGENT_PROMPT,
    SOLVER_AGENT_PROMPT,
    TAPS_AGENT_PROMPT,
    VERIFICATION_AGENT_PROMPT,
)
from physicsos.tools.geometry_tools import (
    apply_boundary_labels,
    assess_mesh_quality,
    export_backend_mesh,
    generate_geometry_encoding,
    generate_mesh,
    import_geometry,
    label_regions,
    repair_geometry,
)
from physicsos.tools.knowledge_tools import build_knowledge_context, ingest_knowledge_document, run_deepsearch, search_arxiv, search_knowledge_base
from physicsos.tools.memory_tools import search_case_memory, store_case_result
from physicsos.tools.postprocess_tools import extract_kpis, generate_visualizations, write_simulation_report
from physicsos.tools.problem_tools import validate_physics_problem
from physicsos.tools.solver_tools import estimate_solver_support, prepare_full_solver_case, route_solver_backend, run_full_solver, run_hybrid_solver, run_surrogate_solver, submit_full_solver_job
from physicsos.tools.surrogate_tools import estimate_surrogate_support, list_available_surrogates, route_surrogate_model, run_surrogate_inference
from physicsos.tools.taps_tools import author_taps_runtime_extension, build_taps_problem, estimate_taps_residual, estimate_taps_support, formulate_taps_equation, run_taps_backend
from physicsos.tools.verification_tools import check_conservation_laws, compute_physics_residuals, detect_ood_case, estimate_uncertainty, validate_selected_slices


SUBAGENTS = [
    {
        "name": "geometry-mesh-agent",
        "description": "Build GeometrySpec and MeshSpec from CAD/STL/STEP/CIF/POSCAR/molecular/text inputs.",
        "system_prompt": GEOMETRY_MESH_AGENT_PROMPT,
        "tools": [
            import_geometry,
            repair_geometry,
            label_regions,
            apply_boundary_labels,
            generate_geometry_encoding,
            generate_mesh,
            export_backend_mesh,
            assess_mesh_quality,
        ],
    },
    {
        "name": "taps-agent",
        "description": "Compile and run TAPS-first equation-driven surrogate solves for explicit parameterized PDEs.",
        "system_prompt": TAPS_AGENT_PROMPT,
        "tools": [
            validate_physics_problem,
            estimate_taps_support,
            build_knowledge_context,
            search_knowledge_base,
            formulate_taps_equation,
            build_taps_problem,
            author_taps_runtime_extension,
            run_taps_backend,
            estimate_taps_residual,
        ],
    },
    {
        "name": "solver-agent",
        "description": "Select and run non-TAPS surrogate, full, or hybrid fallback solver backends for a validated PhysicsProblem.",
        "system_prompt": SOLVER_AGENT_PROMPT,
        "tools": [
            validate_physics_problem,
            list_available_surrogates,
            estimate_surrogate_support,
            route_surrogate_model,
            run_surrogate_inference,
            estimate_solver_support,
            route_solver_backend,
            run_surrogate_solver,
            prepare_full_solver_case,
            submit_full_solver_job,
            run_full_solver,
            run_hybrid_solver,
        ],
    },
    {
        "name": "verification-agent",
        "description": "Check residuals, uncertainty, conservation, OOD risk, and recommended next actions.",
        "system_prompt": VERIFICATION_AGENT_PROMPT,
        "tools": [
            compute_physics_residuals,
            check_conservation_laws,
            validate_selected_slices,
            estimate_uncertainty,
            detect_ood_case,
        ],
    },
    {
        "name": "postprocess-agent",
        "description": "Extract KPIs, visualizations, reports, and optimization suggestions from solver results.",
        "system_prompt": POSTPROCESS_AGENT_PROMPT,
        "tools": [extract_kpis, generate_visualizations, write_simulation_report],
    },
    {
        "name": "knowledge-agent",
        "description": "Retrieve local knowledge, arXiv papers, DeepSearch reports, prior cases, materials, operator templates, validation rules, and paper notes.",
        "system_prompt": KNOWLEDGE_AGENT_PROMPT,
        "tools": [
            search_knowledge_base,
            build_knowledge_context,
            search_arxiv,
            run_deepsearch,
            ingest_knowledge_document,
            search_case_memory,
            store_case_result,
        ],
    },
]
