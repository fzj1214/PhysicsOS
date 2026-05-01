import json

import pytest

from physicsos.schemas.common import ComputeBudget, Provenance
from physicsos.schemas.geometry import GeometryEntity, GeometrySource, GeometrySpec
from physicsos.schemas.geometry import GeometryEncoding
from physicsos.schemas.materials import MaterialProperty, MaterialSpec
from physicsos.schemas.mesh import MeshPolicy
from physicsos.schemas.operators import FieldSpec, OperatorSpec
from physicsos.schemas.operators import PhysicsSpec
from physicsos.schemas.problem import PhysicsProblem
from physicsos.schemas.solver import SolverPolicy
from physicsos.tools.registry import TOOL_REGISTRY
from physicsos.tools.solver_tools import (
    EstimateSolverSupportInput,
    PrepareFullSolverCaseInput,
    RouteSolverBackendInput,
    SubmitFullSolverJobInput,
    estimate_solver_support,
    prepare_full_solver_case,
    route_solver_backend,
    submit_full_solver_job,
)
from physicsos.tools.surrogate_tools import RouteSurrogateModelInput, RunSurrogateInferenceInput, route_surrogate_model, run_surrogate_inference
from physicsos.tools.taps_tools import (
    BuildTAPSProblemInput,
    EstimateTAPSResidualInput,
    FormulateTAPSEquationInput,
    RunTAPSBackendInput,
    AuthorTAPSRuntimeExtensionInput,
    author_taps_runtime_extension,
    build_taps_problem,
    estimate_taps_residual,
    formulate_taps_equation,
    run_taps_backend,
)
from physicsos.tools.verification_tools import (
    CheckConservationLawsInput,
    ComputePhysicsResidualsInput,
    DetectOODCaseInput,
    EstimateUncertaintyInput,
    ValidateSelectedSlicesInput,
    check_conservation_laws,
    compute_physics_residuals,
    detect_ood_case,
    estimate_uncertainty,
    validate_selected_slices,
)
from physicsos.agents.runtime import DeepAgentsRuntimeConfig, build_runtime_kwargs
from physicsos.backends.knowledge_base import search_knowledge, upsert_document
from physicsos.schemas.knowledge import KnowledgeSource
from physicsos.tools.knowledge_tools import BuildKnowledgeContextInput, build_knowledge_context
from physicsos.tools.memory_tools import SearchCaseMemoryInput, StoreCaseResultInput, search_case_memory, store_case_result
from physicsos.tools.catalog_tools import (
    ListOperatorTemplatesInput,
    ListSolverBackendsInput,
    RecommendRuntimeStackInput,
    list_operator_templates,
    list_solver_backends,
    recommend_runtime_stack,
)
from physicsos.tools.geometry_tools import (
    AssessMeshQualityInput,
    ApplyBoundaryLabelsInput,
    ApplyBoundaryLabelingArtifactInput,
    BoundaryLabelAssignment,
    CreateBoundaryLabelingArtifactInput,
    CreateGeometryLabelerViewerInput,
    ExportBackendMeshInput,
    GenerateGeometryEncodingInput,
    GenerateMeshInput,
    ImportGeometryInput,
    LabelRegionsInput,
    PrepareMeshConversionJobInput,
    SubmitMeshConversionJobInput,
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
    prepare_mesh_conversion_job,
    submit_mesh_conversion_job,
)
from physicsos.workflows import run_physicsos_workflow, run_taps_thermal_workflow
from physicsos.backends.taps_generic import (
    _assemble_triangle_elasticity_stiffness,
    _assemble_triangle_nedelec_curl_curl,
    _assemble_triangle_stiffness,
)


def _minimal_fluid_problem() -> PhysicsProblem:
    geometry = GeometrySpec(
        id="geometry:test",
        source=GeometrySource(kind="generated"),
        dimension=3,
    )
    return PhysicsProblem(
        id="problem:test",
        user_intent={"raw_request": "simulate simple flow"},
        domain="fluid",
        geometry=geometry,
        fields=[
            FieldSpec(name="U", kind="vector", units="m/s"),
            FieldSpec(name="p", kind="scalar", units="Pa"),
        ],
        operators=[
            OperatorSpec(
                id="operator:ns",
                name="Navier-Stokes",
                domain="fluid",
                equation_class="navier_stokes",
                form="strong",
                fields_in=["U", "p"],
                fields_out=["U", "p"],
            )
        ],
        materials=[],
        boundary_conditions=[],
        targets=[{"name": "pressure_drop", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )


def test_tool_registry_has_core_tools() -> None:
    assert "build_physics_problem" in TOOL_REGISTRY
    assert "estimate_solver_support" in TOOL_REGISTRY
    assert "check_conservation_laws" in TOOL_REGISTRY
    assert "validate_selected_slices" in TOOL_REGISTRY
    assert "prepare_full_solver_case" in TOOL_REGISTRY
    assert "submit_full_solver_job" in TOOL_REGISTRY
    assert "apply_boundary_labels" in TOOL_REGISTRY
    assert "create_boundary_labeling_artifact" in TOOL_REGISTRY
    assert "apply_boundary_labeling_artifact" in TOOL_REGISTRY
    assert "create_geometry_labeler_viewer" in TOOL_REGISTRY
    assert "export_backend_mesh" in TOOL_REGISTRY
    assert "prepare_mesh_conversion_job" in TOOL_REGISTRY
    assert "submit_mesh_conversion_job" in TOOL_REGISTRY
    assert "run_full_solver" in TOOL_REGISTRY
    assert "list_operator_templates" in TOOL_REGISTRY
    assert "recommend_runtime_stack" in TOOL_REGISTRY
    assert TOOL_REGISTRY["submit_full_solver_job"].requires_approval is True
    assert TOOL_REGISTRY["run_full_solver"].requires_approval is True
    assert TOOL_REGISTRY["submit_mesh_conversion_job"].requires_approval is True


def test_solver_routing_prefers_open_source_cfd_backend() -> None:
    problem = _minimal_fluid_problem()
    support = estimate_solver_support(EstimateSolverSupportInput(problem=problem))
    route = route_solver_backend(
        RouteSolverBackendInput(
            problem=problem,
            support_scores=support.scores,
            policy=SolverPolicy(force_full_solver=True),
        )
    )
    assert route.decision.selected_backend in {"openfoam", "su2"}
    assert route.decision.mode == "full_solver"


def test_stage3_runtime_registries_expose_operator_and_backend_catalogs() -> None:
    problem = _minimal_fluid_problem()
    operators = list_operator_templates(ListOperatorTemplatesInput(domain="fluid"))
    backends = list_solver_backends(ListSolverBackendsInput(domain="fluid", requires_remote_service=True))
    stack = recommend_runtime_stack(RecommendRuntimeStackInput(problem=problem))
    assert any(template.equation_class == "navier_stokes" for template in operators.templates)
    assert {backend.id for backend in backends.backends} >= {"openfoam", "su2"}
    assert stack.verification_rules
    assert stack.postprocess_templates
    assert "remote_full_solver" in stack.recommended_order


def test_full_solver_fallback_prepares_sandbox_manifest() -> None:
    problem = _minimal_fluid_problem()
    output = prepare_full_solver_case(
        PrepareFullSolverCaseInput(
            problem=problem,
            backend="openfoam",
            budget=ComputeBudget(max_wall_time_seconds=120.0, max_cpu_cores=4),
            service_base_url="http://solver-runner.local",
        )
    )
    manifest = json.loads(open(output.runner_manifest.uri, encoding="utf-8").read())
    assert output.requires_approval
    assert output.prepared.backend == "openfoam"
    assert output.runner_manifest.kind == "full_solver_runner_manifest"
    assert manifest["schema_version"] == "physicsos.full_solver_job.v1"
    assert manifest["execution_policy"]["external_process_execution"] == "disabled_until_approved"
    assert manifest["service"]["requires_approval_token"] is True


def test_full_solver_runner_adapter_supports_dry_run_and_requires_remote_http() -> None:
    problem = _minimal_fluid_problem()
    prepared = prepare_full_solver_case(PrepareFullSolverCaseInput(problem=problem, backend="openfoam")).runner_manifest
    dry_run = submit_full_solver_job(SubmitFullSolverJobInput(runner_manifest=prepared, mode="dry_run"))
    dry_payload = json.loads(open(dry_run.runner_response.uri, encoding="utf-8").read())
    assert dry_run.submitted is False
    assert dry_run.result.scalar_outputs["runner_mode"] == "dry_run"
    assert dry_payload["message"].endswith("no external solver service or CLI was invoked.")
    with pytest.raises(ValueError, match="HTTP runner mode requires service_base_url"):
        submit_full_solver_job(SubmitFullSolverJobInput(runner_manifest=prepared, mode="http", approval_token="test-token"))
    with pytest.raises(PermissionError, match="HTTP runner mode requires approval_token"):
        submit_full_solver_job(
            SubmitFullSolverJobInput(
                runner_manifest=prepared,
                mode="http",
                service_base_url="https://foamvm.vercel.app",
            )
        )


def test_surrogate_runtime_routes_to_neural_operator_scaffold() -> None:
    problem = _minimal_fluid_problem()
    route = route_surrogate_model(RouteSurrogateModelInput(problem=problem))
    assert route.decision.selected_model_id != "none"
    result = run_surrogate_inference(RunSurrogateInferenceInput(problem=problem, decision=route.decision))
    assert result.result.status == "needs_review"
    assert result.result.backend == route.decision.selected_model_id


def test_surrogate_adapter_generates_io_bundles_for_downloaded_checkpoint() -> None:
    problem = _minimal_fluid_problem()
    problem.geometry.encodings.append(GeometryEncoding(kind="multi_resolution_grid", uri="scratch/grid.json"))
    route = route_surrogate_model(RouteSurrogateModelInput(problem=problem))
    result = run_surrogate_inference(RunSurrogateInferenceInput(problem=problem, decision=route.decision))
    if route.decision.selected_model_id.startswith("polymathic"):
        artifact_kinds = {artifact.kind for artifact in result.result.artifacts}
        assert "surrogate_input_bundle" in artifact_kinds
        assert "surrogate_output_bundle" in artifact_kinds


def test_deepagents_runtime_kwargs_can_be_built_without_optional_deps() -> None:
    kwargs = build_runtime_kwargs(
        DeepAgentsRuntimeConfig(
            enable_filesystem_backend=False,
            enable_memory_store=False,
            enable_checkpointer=False,
        )
    )
    assert kwargs["name"] == "physicsos-main"
    assert "interrupt_on" in kwargs


def test_deepagents_graph_can_be_created_with_model_object() -> None:
    fake_models = pytest.importorskip("langchain_core.language_models.fake_chat_models")
    pytest.importorskip("deepagents")
    from physicsos.agents import create_physicsos_agent

    model = fake_models.FakeListChatModel(responses=["ok"])
    agent = create_physicsos_agent(
        model=model,
        runtime=DeepAgentsRuntimeConfig(
            enable_filesystem_backend=False,
            enable_memory_store=False,
            enable_checkpointer=False,
        ),
    )
    assert hasattr(agent, "invoke")


def test_local_knowledge_base_search(tmp_path) -> None:
    db = tmp_path / "kb.sqlite"
    source = KnowledgeSource(id="manual:test", kind="manual", title="TAPS note")
    chunks = upsert_document(source, "TAPS uses tensor decomposition and Galerkin weak forms.", db_path=db)
    assert chunks == 1
    results = search_knowledge("TAPS Galerkin", db_path=db)
    assert results
    assert results[0].source.title == "TAPS note"


def test_knowledge_context_can_run_without_network() -> None:
    context = build_knowledge_context(
        BuildKnowledgeContextInput(query="TAPS heat equation", local_top_k=2, arxiv_max_results=0)
    ).context
    assert context.query == "TAPS heat equation"
    assert context.papers == []


def test_case_memory_stores_and_retrieves_similar_cases(tmp_path) -> None:
    memory_path = tmp_path / "case_memory.jsonl"
    result = run_physicsos_workflow(use_knowledge=False, taps_rank=8)
    stored = store_case_result(
        StoreCaseResultInput(
            problem=result.problem,
            result=result.solver_result,
            verification=result.verification,
            postprocess=result.postprocess,
            memory_uri=str(memory_path),
            dataset_tags=["test"],
        )
    )
    assert stored.stored
    assert memory_path.exists()
    assert {"domain", "operators", "geometry", "solver_backend"} <= set(stored.indexed_features)

    hits = search_case_memory(
        SearchCaseMemoryInput(
            problem=result.problem,
            top_k=3,
            filters={"domain": result.problem.domain, "backend": result.solver_result.backend},
            memory_uri=str(memory_path),
        )
    )
    assert hits.searched_records == 1
    assert hits.cases
    assert hits.cases[0].case_id == result.problem.id
    assert hits.cases[0].score > 0.5
    assert hits.cases[0].backend == result.solver_result.backend
    assert hits.cases[0].verification_status == result.verification.status


def test_taps_thermal_workflow_writes_artifacts() -> None:
    result = run_taps_thermal_workflow(rank=8, use_knowledge=False)
    assert result.result.backend == "taps:thermal_1d"
    assert result.residual.residuals
    assert result.verification.status in {"accepted", "accepted_with_warnings"}
    assert {artifact.kind for artifact in result.result.artifacts} >= {
        "taps_factor_matrices",
        "taps_reconstruction_metadata",
        "taps_residual_history",
    }


def test_universal_workflow_runs_taps_first_loop() -> None:
    result = run_physicsos_workflow(use_knowledge=False, taps_rank=8)
    assert result.solver_result.backend == "taps:thermal_1d"
    assert result.taps_problem is not None
    assert result.taps_residual is not None
    assert result.verification.status == "accepted"
    assert result.verification.recommended_next_action == "accept"
    assert result.postprocess.report is not None
    assert any(artifact.kind == "visualization:residual_summary" for artifact in result.postprocess.visualizations)
    assert any(artifact.kind == "simulation_report_manifest" for artifact in result.postprocess.visualizations)
    report_text = open(result.postprocess.report.uri, encoding="utf-8").read()
    assert "## Executive Summary" in report_text
    assert "## Verification Appendix" in report_text
    assert "## Artifact Manifest" in report_text
    assert result.verification.uncertainty
    assert result.trace[-1].name == "case-memory"


def test_geometry_mesh_tools_report_real_backend_availability() -> None:
    geometry = import_geometry(ImportGeometryInput(source=GeometrySource(kind="generated"))).geometry
    result = generate_mesh(
        GenerateMeshInput(
            geometry=geometry,
            physics=PhysicsSpec(domains=["thermal"]),
            mesh_policy=MeshPolicy(target_element_size=0.25),
            target_backends=["fenicsx"],
        )
    )
    assert result.mesh.solver_compatibility == ["fenicsx"]
    if result.mesh.quality.passes:
        assert result.mesh.files
        assert result.mesh.elements.total is not None and result.mesh.elements.total > 0
    else:
        assert any("gmsh" in issue.lower() for issue in result.mesh.quality.issues)


def test_taps_agent_formulates_non_template_custom_operator() -> None:
    problem = _minimal_fluid_problem()
    problem.operators[0].differential_terms.append(
        {"expression": "int_Omega v_i * (rho * U_j * grad_j U_i + grad_i p) dOmega", "order": 1, "fields": ["U", "p"]}
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    assert plan.weak_form is not None
    assert plan.weak_form.terms
    assert plan.weak_form.terms[0].role == "custom"
    assert plan.recommended_next_action in {"compile_taps_problem", "ask_knowledge_agent"}


def test_taps_agent_requests_knowledge_for_under_specified_problem() -> None:
    geometry = GeometrySpec(id="geometry:unknown", source=GeometrySource(kind="text"), dimension=3)
    problem = PhysicsProblem(
        id="problem:unknown",
        user_intent={"raw_request": "solve a new coupled physical phenomenon from a paper"},
        domain="custom",
        geometry=geometry,
        fields=[FieldSpec(name="u", kind="scalar")],
        operators=[
            OperatorSpec(
                id="operator:unknown",
                name="unknown",
                domain="custom",
                equation_class="unspecified",
                form="strong",
                fields_out=["u"],
            )
        ],
        materials=[],
        boundary_conditions=[],
        targets=[{"name": "field", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    assert plan.status == "needs_knowledge"
    assert plan.required_knowledge_queries


def test_taps_problem_carries_general_weak_form_ir() -> None:
    geometry = GeometrySpec(id="geometry:poisson", source=GeometrySource(kind="generated"), dimension=2)
    problem = PhysicsProblem(
        id="problem:poisson",
        user_intent={"raw_request": "solve a Poisson equation on a 2D domain"},
        domain="custom",
        geometry=geometry,
        fields=[FieldSpec(name="u", kind="scalar")],
        operators=[
            OperatorSpec(
                id="operator:poisson",
                name="Poisson",
                domain="custom",
                equation_class="poisson",
                form="weak",
                fields_out=["u"],
            )
        ],
        materials=[],
        boundary_conditions=[{"id": "bc:u", "region_id": "boundary", "field": "u", "kind": "dirichlet", "value": 0.0}],
        targets=[{"name": "field", "field": "u", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    assert taps_problem.weak_form is not None
    assert taps_problem.weak_form.family == "poisson"
    assert taps_problem.compilation_status == "compiled"


def test_taps_generic_scalar_assembler_executes_1d_poisson() -> None:
    geometry = GeometrySpec(id="geometry:line", source=GeometrySource(kind="generated"), dimension=1)
    problem = PhysicsProblem(
        id="problem:poisson-1d",
        user_intent={"raw_request": "solve a 1D Poisson equation with homogeneous Dirichlet endpoints"},
        domain="custom",
        geometry=geometry,
        fields=[FieldSpec(name="u", kind="scalar")],
        operators=[
            OperatorSpec(
                id="operator:poisson",
                name="Poisson",
                domain="custom",
                equation_class="poisson",
                form="weak",
                fields_out=["u"],
            )
        ],
        materials=[],
        boundary_conditions=[
            {"id": "bc:left", "region_id": "x=0", "field": "u", "kind": "dirichlet", "value": 0.0},
            {"id": "bc:right", "region_id": "x=1", "field": "u", "kind": "dirichlet", "value": 0.0},
        ],
        targets=[{"name": "field", "field": "u", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result
    residual = estimate_taps_residual(EstimateTAPSResidualInput(problem=problem, taps_problem=taps_problem, result=result)).report
    assert result.backend.startswith("taps:generic_scalar_elliptic_1d:poisson")
    assert result.status == "success"
    assert residual.converged
    assert residual.rank == taps_problem.basis.tensor_rank
    assert {artifact.kind for artifact in result.artifacts} >= {
        "taps_assembled_operator",
        "taps_solution_field",
        "taps_residual_history",
    }


def test_taps_generic_scalar_assembler_executes_2d_poisson() -> None:
    geometry = GeometrySpec(id="geometry:square", source=GeometrySource(kind="generated"), dimension=2)
    problem = PhysicsProblem(
        id="problem:poisson-2d",
        user_intent={"raw_request": "solve a 2D Poisson equation on a square with homogeneous Dirichlet boundaries"},
        domain="custom",
        geometry=geometry,
        fields=[FieldSpec(name="u", kind="scalar")],
        operators=[
            OperatorSpec(
                id="operator:poisson",
                name="Poisson",
                domain="custom",
                equation_class="poisson",
                form="weak",
                fields_out=["u"],
            )
        ],
        materials=[],
        boundary_conditions=[{"id": "bc:u", "region_id": "boundary", "field": "u", "kind": "dirichlet", "value": 0.0}],
        targets=[{"name": "field", "field": "u", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result
    residual = estimate_taps_residual(EstimateTAPSResidualInput(problem=problem, taps_problem=taps_problem, result=result)).report
    assert result.backend.startswith("taps:generic_scalar_elliptic_2d:poisson")
    assert result.status == "success"
    assert residual.converged
    assert residual.rank == taps_problem.basis.tensor_rank
    assert result.scalar_outputs["tensor_rank"] == taps_problem.basis.tensor_rank
    assert {artifact.kind for artifact in result.artifacts} >= {
        "taps_assembled_operator",
        "taps_solution_field",
        "taps_residual_history",
    }


def test_taps_nonlinear_reaction_diffusion_executes_1d() -> None:
    geometry = GeometrySpec(id="geometry:line-rd", source=GeometrySource(kind="generated"), dimension=1)
    problem = PhysicsProblem(
        id="problem:reaction-diffusion-1d",
        user_intent={"raw_request": "solve a 1D nonlinear reaction diffusion equation"},
        domain="custom",
        geometry=geometry,
        fields=[FieldSpec(name="u", kind="scalar")],
        operators=[
            OperatorSpec(
                id="operator:rd",
                name="Reaction diffusion",
                domain="custom",
                equation_class="reaction_diffusion",
                form="weak",
                fields_out=["u"],
            )
        ],
        materials=[],
        boundary_conditions=[
            {"id": "bc:left", "region_id": "x=0", "field": "u", "kind": "dirichlet", "value": 0.0},
            {"id": "bc:right", "region_id": "x=1", "field": "u", "kind": "dirichlet", "value": 0.0},
        ],
        targets=[{"name": "field", "field": "u", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result
    residual = estimate_taps_residual(EstimateTAPSResidualInput(problem=problem, taps_problem=taps_problem, result=result)).report
    assert result.backend == "taps:nonlinear_reaction_diffusion_1d"
    assert result.status == "success"
    assert residual.converged
    assert "normalized_nonlinear_residual" in result.residuals
    assert "nonlinear_iterations" in result.residuals
    assert {artifact.kind for artifact in result.artifacts} >= {
        "taps_nonlinear_operator",
        "taps_solution_field",
        "taps_iteration_history",
    }


def test_taps_nonlinear_reaction_diffusion_executes_2d() -> None:
    geometry = GeometrySpec(id="geometry:square-rd", source=GeometrySource(kind="generated"), dimension=2)
    problem = PhysicsProblem(
        id="problem:reaction-diffusion-2d",
        user_intent={"raw_request": "solve a 2D nonlinear reaction diffusion equation"},
        domain="custom",
        geometry=geometry,
        fields=[FieldSpec(name="u", kind="scalar")],
        operators=[
            OperatorSpec(
                id="operator:rd",
                name="Reaction diffusion",
                domain="custom",
                equation_class="reaction_diffusion",
                form="weak",
                fields_out=["u"],
            )
        ],
        materials=[],
        boundary_conditions=[{"id": "bc:u", "region_id": "boundary", "field": "u", "kind": "dirichlet", "value": 0.0}],
        targets=[{"name": "field", "field": "u", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    for axis in plan.axes:
        axis.points = 24
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result
    residual = estimate_taps_residual(EstimateTAPSResidualInput(problem=problem, taps_problem=taps_problem, result=result)).report
    assert result.backend == "taps:nonlinear_reaction_diffusion_2d"
    assert result.status == "success"
    assert residual.converged
    assert "normalized_nonlinear_residual" in result.residuals
    assert "nonlinear_iterations" in result.residuals
    assert {artifact.kind for artifact in result.artifacts} >= {
        "taps_nonlinear_operator",
        "taps_solution_field",
        "taps_iteration_history",
    }


def test_taps_coupled_reaction_diffusion_executes_2d() -> None:
    geometry = GeometrySpec(id="geometry:square-coupled-rd", source=GeometrySource(kind="generated"), dimension=2)
    problem = PhysicsProblem(
        id="problem:coupled-reaction-diffusion-2d",
        user_intent={"raw_request": "solve a 2D two-field coupled reaction diffusion equation"},
        domain="custom",
        geometry=geometry,
        fields=[FieldSpec(name="u", kind="scalar"), FieldSpec(name="v", kind="scalar")],
        operators=[
            OperatorSpec(
                id="operator:coupled-rd",
                name="Coupled reaction diffusion",
                domain="custom",
                equation_class="coupled_reaction_diffusion",
                form="weak",
                fields_out=["u", "v"],
            )
        ],
        materials=[],
        boundary_conditions=[
            {"id": "bc:u", "region_id": "boundary", "field": "u", "kind": "dirichlet", "value": 0.0},
            {"id": "bc:v", "region_id": "boundary", "field": "v", "kind": "dirichlet", "value": 0.0},
        ],
        targets=[{"name": "fields", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    for axis in plan.axes:
        axis.points = 24
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result
    residual = estimate_taps_residual(EstimateTAPSResidualInput(problem=problem, taps_problem=taps_problem, result=result)).report
    assert result.backend == "taps:coupled_reaction_diffusion_2d"
    assert result.status == "success"
    assert result.scalar_outputs["field_count"] == 2
    assert residual.converged
    assert "normalized_coupled_residual" in result.residuals
    assert {artifact.kind for artifact in result.artifacts} >= {
        "taps_coupled_operator",
        "taps_coupled_solution_fields",
        "taps_iteration_history",
    }


def test_geometry_encoded_taps_consumes_occupancy_mask() -> None:
    geometry = GeometrySpec(id="geometry:masked-square", source=GeometrySource(kind="generated"), dimension=2)
    encoding_output = generate_geometry_encoding(
        GenerateGeometryEncodingInput(geometry=geometry, encodings=["occupancy_mask"], resolutions=[[24, 24]])
    )
    geometry.encodings.extend(encoding_output.encodings)
    problem = PhysicsProblem(
        id="problem:geometry-encoded-poisson-2d",
        user_intent={"raw_request": "solve Poisson on a geometry-encoded square mask"},
        domain="custom",
        geometry=geometry,
        fields=[FieldSpec(name="u", kind="scalar")],
        operators=[
            OperatorSpec(
                id="operator:poisson",
                name="Poisson",
                domain="custom",
                equation_class="poisson",
                form="weak",
                fields_out=["u"],
            )
        ],
        materials=[],
        boundary_conditions=[{"id": "bc:u", "region_id": "boundary", "field": "u", "kind": "dirichlet", "value": 0.0}],
        targets=[{"name": "field", "field": "u", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result
    assert taps_problem.geometry_encodings
    assert taps_problem.geometry_encodings[0].kind == "occupancy_mask"
    assert result.status == "success"
    assert result.residuals["active_cell_fraction"] == 1.0
    assert any(artifact.kind == "taps_assembled_operator" for artifact in result.artifacts)


def test_geometry_encoded_taps_handles_hole_mask() -> None:
    geometry = GeometrySpec(
        id="geometry:hole-square",
        source=GeometrySource(kind="generated"),
        dimension=2,
        entities=[GeometryEntity(id="entity:hole", kind="region", label="central_hole")],
    )
    encoding_output = generate_geometry_encoding(
        GenerateGeometryEncodingInput(geometry=geometry, encodings=["occupancy_mask"], resolutions=[[32, 32]])
    )
    geometry.encodings.extend(encoding_output.encodings)
    problem = PhysicsProblem(
        id="problem:geometry-encoded-hole-poisson-2d",
        user_intent={"raw_request": "solve Poisson on a square domain with a central circular hole mask"},
        domain="custom",
        geometry=geometry,
        fields=[FieldSpec(name="u", kind="scalar")],
        operators=[
            OperatorSpec(
                id="operator:poisson",
                name="Poisson",
                domain="custom",
                equation_class="poisson",
                form="weak",
                fields_out=["u"],
            )
        ],
        materials=[],
        boundary_conditions=[{"id": "bc:u", "region_id": "boundary", "field": "u", "kind": "dirichlet", "value": 0.0}],
        targets=[{"name": "field", "field": "u", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result
    residual = estimate_taps_residual(EstimateTAPSResidualInput(problem=problem, taps_problem=taps_problem, result=result)).report
    assert result.backend.startswith("taps:generic_scalar_elliptic_2d")
    assert result.status == "success"
    assert residual.converged
    assert 0.0 < result.residuals["active_cell_fraction"] < 1.0
    assert result.residuals["masked_relaxation_iterations"] > 0
    assert any(artifact.kind == "taps_assembled_operator" for artifact in result.artifacts)


def test_geometry_agent_labels_generated_boundaries_conservatively() -> None:
    geometry = GeometrySpec(id="geometry:generated-channel", source=GeometrySource(kind="generated"), dimension=2)
    output = label_regions(LabelRegionsInput(geometry=geometry, physics_domain="fluid"))
    assert output.geometry.regions[0].id == "region:domain"
    labels = {boundary.label: boundary.kind for boundary in output.geometry.boundaries}
    assert labels == {
        "x_min": "inlet",
        "x_max": "outlet",
        "y_min": "wall",
        "y_max": "wall",
    }
    assert all(boundary.confidence < 1.0 for boundary in output.geometry.boundaries)
    assert not output.unresolved_regions


def test_geometry_agent_applies_explicit_boundary_labels() -> None:
    geometry = GeometrySpec(
        id="geometry:imported-shell",
        source=GeometrySource(kind="stl", uri="dummy.stl"),
        dimension=2,
        entities=[GeometryEntity(id="entity:2:1", kind="surface", label="unnamed_surface")],
        quality={"passes": False, "unresolved_regions": ["boundary_labels"], "issues": ["Boundary labels are unresolved; user/CAD physical groups are required."]},
    )
    output = apply_boundary_labels(
        ApplyBoundaryLabelsInput(
            geometry=geometry,
            assignments=[
                BoundaryLabelAssignment(
                    entity_ids=["entity:2:1"],
                    boundary_id="boundary:wall",
                    label="wall",
                    kind="wall",
                    confidence=1.0,
                )
            ],
            source="user",
        )
    )
    assert output.applied == ["boundary:wall"]
    assert output.geometry.boundaries[0].kind == "wall"
    assert output.geometry.boundaries[0].confidence == 1.0
    assert output.geometry.quality is not None
    assert "boundary_labels" not in output.geometry.quality.unresolved_regions
    assert output.geometry.transforms[-1].description.startswith("Applied 1 explicit boundary label")


def test_geometry_agent_assesses_triangle_mesh_quality() -> None:
    geometry = GeometrySpec(id="geometry:quality-square", source=GeometrySource(kind="generated"), dimension=2)
    mesh = generate_mesh(
        GenerateMeshInput(
            geometry=geometry,
            physics=PhysicsSpec(domains=["custom"]),
            mesh_policy=MeshPolicy(target_element_size=0.35),
            target_backends=["taps"],
        )
    ).mesh
    output = assess_mesh_quality(AssessMeshQualityInput(mesh=mesh, physics=PhysicsSpec(domains=["custom"]), backend="taps"))
    assert output.recommended_action == "accept"
    assert output.report.passes
    assert output.report.min_jacobian is not None and output.report.min_jacobian > 0.0
    assert output.report.aspect_ratio_p95 is not None and output.report.aspect_ratio_p95 >= 1.0
    assert output.report.max_skewness is not None and 0.0 <= output.report.max_skewness < 1.0


def test_mesh_graph_taps_solves_fem_poisson() -> None:
    geometry = GeometrySpec(id="geometry:mesh-graph-square", source=GeometrySource(kind="generated"), dimension=2)
    mesh = generate_mesh(
        GenerateMeshInput(
            geometry=geometry,
            physics=PhysicsSpec(domains=["custom"]),
            mesh_policy=MeshPolicy(target_element_size=0.35),
            target_backends=["taps"],
        )
    ).mesh
    encoding_output = generate_geometry_encoding(
        GenerateGeometryEncodingInput(geometry=geometry, mesh=mesh, encodings=["mesh_graph"])
    )
    graph_payload = json.loads(open(encoding_output.artifacts[0].uri, encoding="utf-8").read())
    assert set(graph_payload["boundary_node_sets"]) >= {"x_min", "x_max", "y_min", "y_max"}
    assert set(graph_payload["boundary_edge_sets"]) >= {"boundary:x_min", "boundary:x_max", "boundary:y_min", "boundary:y_max"}
    assert graph_payload["boundary_edge_sets"]["boundary:x_min"] == graph_payload["boundary_edge_sets"]["x_min"]
    physical_names = {group["name"] for group in graph_payload["physical_boundary_groups"]}
    assert {"x_min", "x_max", "y_min", "y_max"} <= physical_names
    assert graph_payload["boundary_nodes"]
    geometry.encodings.extend(encoding_output.encodings)
    problem = PhysicsProblem(
        id="problem:mesh-graph-poisson-2d",
        user_intent={"raw_request": "solve Poisson on a Gmsh mesh graph"},
        domain="custom",
        geometry=geometry,
        mesh=mesh,
        fields=[FieldSpec(name="u", kind="scalar")],
        operators=[
            OperatorSpec(
                id="operator:poisson",
                name="Poisson",
                domain="custom",
                equation_class="poisson",
                form="weak",
                fields_out=["u"],
            )
        ],
        materials=[],
        boundary_conditions=[{"id": "bc:u", "region_id": "boundary", "field": "u", "kind": "dirichlet", "value": 0.0}],
        targets=[{"name": "field", "field": "u", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result
    residual = estimate_taps_residual(EstimateTAPSResidualInput(problem=problem, taps_problem=taps_problem, result=result)).report
    assert result.backend.startswith("taps:mesh_fem_poisson")
    assert result.status == "success"
    assert residual.converged
    assert result.residuals["fem_nodes"] > 0
    assert result.residuals["fem_triangles"] > 0
    assert result.residuals["fem_nonzeros"] > 0
    assert {artifact.kind for artifact in result.artifacts} >= {
        "taps_mesh_fem_operator",
        "taps_mesh_fem_solution_field",
        "taps_iteration_history",
    }


def test_mesh_graph_taps_solves_p2_fem_poisson_from_second_order_gmsh_mesh() -> None:
    geometry = GeometrySpec(id="geometry:mesh-p2-square", source=GeometrySource(kind="generated"), dimension=2)
    mesh = generate_mesh(
        GenerateMeshInput(
            geometry=geometry,
            physics=PhysicsSpec(domains=["custom"]),
            mesh_policy=MeshPolicy(target_element_size=0.45, element_order=2),
            target_backends=["taps"],
        )
    ).mesh
    assert any("triangle 6" in cell_type.lower() for cell_type in mesh.topology.cell_types)
    encoding_output = generate_geometry_encoding(
        GenerateGeometryEncodingInput(geometry=geometry, mesh=mesh, encodings=["mesh_graph"])
    )
    graph_payload = json.loads(open(encoding_output.artifacts[0].uri, encoding="utf-8").read())
    assert any(len(cell) == 6 for block in graph_payload["cell_blocks"] for cell in block["cells"] if "triangle" in block["type"].lower())
    geometry.encodings.extend(encoding_output.encodings)
    problem = PhysicsProblem(
        id="problem:mesh-p2-poisson-2d",
        user_intent={"raw_request": "solve Poisson on a second-order Gmsh triangle mesh"},
        domain="custom",
        geometry=geometry,
        mesh=mesh,
        fields=[FieldSpec(name="u", kind="scalar")],
        operators=[
            OperatorSpec(
                id="operator:poisson",
                name="Poisson",
                domain="custom",
                equation_class="poisson",
                form="weak",
                fields_out=["u"],
            )
        ],
        materials=[],
        boundary_conditions=[{"id": "bc:u", "region_id": "boundary", "field": "u", "kind": "dirichlet", "value": 0.0}],
        targets=[{"name": "field", "field": "u", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result
    assert result.status == "success"
    assert result.residuals["fem_basis_order"] == 2.0
    assert result.residuals["fem_nonzeros"] > result.residuals["fem_triangles"] * 6


def test_mesh_graph_taps_solves_em_curl_curl_nedelec() -> None:
    geometry = GeometrySpec(id="geometry:mesh-em-square", source=GeometrySource(kind="generated"), dimension=2)
    mesh = generate_mesh(
        GenerateMeshInput(
            geometry=geometry,
            physics=PhysicsSpec(domains=["electromagnetic"]),
            mesh_policy=MeshPolicy(target_element_size=0.35),
            target_backends=["taps"],
        )
    ).mesh
    encoding_output = generate_geometry_encoding(
        GenerateGeometryEncodingInput(geometry=geometry, mesh=mesh, encodings=["mesh_graph"])
    )
    geometry.encodings.extend(encoding_output.encodings)
    problem = PhysicsProblem(
        id="problem:mesh-em-curl-curl-2d",
        user_intent={"raw_request": "solve a 2D out-of-plane electromagnetic curl-curl problem on a square mesh"},
        domain="electromagnetic",
        geometry=geometry,
        mesh=mesh,
        fields=[FieldSpec(name="E", kind="vector")],
        operators=[
            OperatorSpec(
                id="operator:maxwell",
                name="Maxwell curl-curl",
                domain="electromagnetic",
                equation_class="maxwell",
                form="weak",
                fields_out=["E"],
            )
        ],
        materials=[
            MaterialSpec(
                id="material:dielectric",
                name="dielectric",
                phase="custom",
                properties=[
                    MaterialProperty(name="relative_permittivity", value=2.5),
                    MaterialProperty(name="relative_permeability", value=1.0),
                    MaterialProperty(name="wave_number", value=0.4),
                    MaterialProperty(name="current_source", value=0.75),
                ],
            )
        ],
        boundary_conditions=[{"id": "bc:e", "region_id": "boundary", "field": "E_t", "kind": "dirichlet", "value": 0.0}],
        targets=[{"name": "field", "field": "E", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result
    residual = estimate_taps_residual(EstimateTAPSResidualInput(problem=problem, taps_problem=taps_problem, result=result)).report
    assert taps_problem.weak_form is not None
    assert taps_problem.weak_form.family == "maxwell"
    assert taps_problem.boundary_conditions[0].field == "E_t"
    assert any(term.id.endswith(":curl_curl") for term in taps_problem.weak_form.terms)
    assert result.backend.startswith("taps:mesh_fem_em_curl_curl")
    assert result.status == "success"
    assert residual.converged
    assert result.residuals["fem_nodes"] > 0
    assert {artifact.kind for artifact in result.artifacts} >= {
        "taps_mesh_fem_em_curl_curl_operator",
        "taps_mesh_fem_em_field",
        "taps_iteration_history",
    }
    operator_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_mesh_fem_em_curl_curl_operator")
    operator_payload = json.loads(open(operator_artifact.uri, encoding="utf-8").read())
    assert operator_payload["type"] == "triangle_nedelec_order1_em_curl_curl"
    assert operator_payload["assembly"] == "nedelec_first_kind_edge_element_hcurl"
    assert operator_payload["edge_dof_count"] > 0
    assert operator_payload["boundary_edge_count"] > 0
    assert operator_payload["material"]["relative_permittivity"] == pytest.approx(2.5)
    assert operator_payload["material"]["wave_number"] == pytest.approx(0.4)
    assert operator_payload["material"]["source_amplitude"] == pytest.approx(0.75)
    assert operator_payload["hcurl_scaffold"]["edge_dofs_required"] is True
    assert operator_payload["hcurl_scaffold"]["status"] == "nedelec_order1_edge_element"
    assert operator_payload["hcurl_scaffold"]["boundary_condition"] == "pec_tangential_zero"


def test_mesh_graph_taps_solves_em_curl_curl_nedelec_order2() -> None:
    geometry = GeometrySpec(id="geometry:mesh-em-p2-square", source=GeometrySource(kind="generated"), dimension=2)
    mesh = generate_mesh(
        GenerateMeshInput(
            geometry=geometry,
            physics=PhysicsSpec(domains=["electromagnetic"]),
            mesh_policy=MeshPolicy(target_element_size=0.55, element_order=2),
            target_backends=["taps"],
        )
    ).mesh
    encoding_output = generate_geometry_encoding(
        GenerateGeometryEncodingInput(geometry=geometry, mesh=mesh, encodings=["mesh_graph"])
    )
    geometry.encodings.extend(encoding_output.encodings)
    problem = PhysicsProblem(
        id="problem:mesh-em-p2-curl-curl-2d",
        user_intent={"raw_request": "solve a second-order 2D electromagnetic curl-curl problem on a square mesh"},
        domain="electromagnetic",
        geometry=geometry,
        mesh=mesh,
        fields=[FieldSpec(name="E", kind="vector")],
        operators=[
            OperatorSpec(
                id="operator:maxwell",
                name="Maxwell curl-curl",
                domain="electromagnetic",
                equation_class="maxwell",
                form="weak",
                fields_out=["E"],
            )
        ],
        materials=[
            MaterialSpec(
                id="material:air-p2",
                name="air",
                phase="gas",
                properties=[
                    MaterialProperty(name="relative_permittivity", value=1.0),
                    MaterialProperty(name="relative_permeability", value=1.0),
                    MaterialProperty(name="wave_number", value=0.4),
                    MaterialProperty(name="current_source", value=0.25),
                ],
            )
        ],
        boundary_conditions=[{"id": "bc:e", "region_id": "boundary", "field": "E_t", "kind": "dirichlet", "value": 0.0}],
        targets=[{"name": "field", "field": "E", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result
    assert result.status == "success"
    assert result.residuals["fem_basis_order"] == 2.0
    operator_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_mesh_fem_em_curl_curl_operator")
    operator_payload = json.loads(open(operator_artifact.uri, encoding="utf-8").read())
    assert operator_payload["type"] == "triangle_nedelec_order2_em_curl_curl"
    assert operator_payload["basis_order"] == 2
    assert operator_payload["hcurl_scaffold"]["status"] == "nedelec_order2_hierarchical_scaffold"
    assert operator_payload["hcurl_scaffold"]["high_order_boundary_dofs"] is True
    assert operator_payload["dof_count"] > operator_payload["edge_dof_count"] > 0
    assert operator_payload["cell_interior_dof_count"] > 0
    assert operator_payload["boundary_dof_count"] >= 2 * operator_payload["boundary_edge_count"]
    assert any(element["basis"] == "nedelec_first_kind_order2_hierarchical_scaffold_triangle" for element in operator_payload["elements"])
    solution_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_mesh_fem_em_field")
    solution_payload = json.loads(open(solution_artifact.uri, encoding="utf-8").read())
    assert len(solution_payload["values"]) == operator_payload["dof_count"]


def test_mesh_graph_taps_em_order2_boundary_policy_selects_high_order_dofs() -> None:
    geometry = GeometrySpec(id="geometry:mesh-em-p2-port-xmin-square", source=GeometrySource(kind="generated"), dimension=2)
    mesh = generate_mesh(
        GenerateMeshInput(
            geometry=geometry,
            physics=PhysicsSpec(domains=["electromagnetic"]),
            mesh_policy=MeshPolicy(target_element_size=0.6, element_order=2),
            target_backends=["taps"],
        )
    ).mesh
    encoding_output = generate_geometry_encoding(
        GenerateGeometryEncodingInput(geometry=geometry, mesh=mesh, encodings=["mesh_graph"])
    )
    graph_payload = json.loads(open(encoding_output.artifacts[0].uri, encoding="utf-8").read())
    assert graph_payload["boundary_edge_sets"]["boundary:x_min"]
    geometry.encodings.extend(encoding_output.encodings)
    problem = PhysicsProblem(
        id="problem:mesh-em-p2-port-xmin-curl-curl-2d",
        user_intent={"raw_request": "solve a second-order curl-curl EM problem with a port only on x_min"},
        domain="electromagnetic",
        geometry=geometry,
        mesh=mesh,
        fields=[FieldSpec(name="E", kind="vector")],
        operators=[
            OperatorSpec(
                id="operator:maxwell",
                name="Maxwell curl-curl",
                domain="electromagnetic",
                equation_class="maxwell",
                form="weak",
                fields_out=["E"],
            )
        ],
        materials=[
            MaterialSpec(
                id="material:air-p2-port-xmin",
                name="air",
                phase="gas",
                properties=[
                    MaterialProperty(name="relative_permittivity", value=1.0),
                    MaterialProperty(name="relative_permeability", value=1.0),
                    MaterialProperty(name="wave_number", value=0.5),
                    MaterialProperty(name="current_source", value=0.0),
                ],
            )
        ],
        boundary_conditions=[
            {
                "id": "bc:p2-port-xmin",
                "region_id": "boundary:x_min",
                "field": "E_t",
                "kind": "custom",
                "value": {"kind": "port", "impedance": 1.0, "amplitude": 1.0},
            }
        ],
        targets=[{"name": "field", "field": "E", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result
    assert result.status == "success"
    operator_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_mesh_fem_em_curl_curl_operator")
    operator_payload = json.loads(open(operator_artifact.uri, encoding="utf-8").read())
    assert operator_payload["basis_order"] == 2
    assert operator_payload["hcurl_scaffold"]["boundary_condition"] == "port"
    assert operator_payload["active_boundary_edges"] == graph_payload["boundary_edge_sets"]["boundary:x_min"]
    assert operator_payload["active_boundary_geometric_edge_count"] == 1
    assert operator_payload["active_boundary_dof_count"] == 2 * operator_payload["active_boundary_geometric_edge_count"]
    assert operator_payload["active_boundary_edge_count"] >= operator_payload["active_boundary_geometric_edge_count"]
    active_dof_entities = [operator_payload["dofs"][index] for index in operator_payload["active_boundary_dofs"]]
    assert all(entity["kind"] == "edge_moment" for entity in active_dof_entities)


def test_mesh_graph_taps_em_curl_curl_supports_natural_boundary_policy() -> None:
    geometry = GeometrySpec(id="geometry:mesh-em-natural-square", source=GeometrySource(kind="generated"), dimension=2)
    mesh = generate_mesh(
        GenerateMeshInput(
            geometry=geometry,
            physics=PhysicsSpec(domains=["electromagnetic"]),
            mesh_policy=MeshPolicy(target_element_size=0.45),
            target_backends=["taps"],
        )
    ).mesh
    encoding_output = generate_geometry_encoding(
        GenerateGeometryEncodingInput(geometry=geometry, mesh=mesh, encodings=["mesh_graph"])
    )
    geometry.encodings.extend(encoding_output.encodings)
    problem = PhysicsProblem(
        id="problem:mesh-em-natural-curl-curl-2d",
        user_intent={"raw_request": "solve a 2D electromagnetic curl-curl problem with natural farfield boundary"},
        domain="electromagnetic",
        geometry=geometry,
        mesh=mesh,
        fields=[FieldSpec(name="E", kind="vector")],
        operators=[
            OperatorSpec(
                id="operator:maxwell",
                name="Maxwell curl-curl",
                domain="electromagnetic",
                equation_class="maxwell",
                form="weak",
                fields_out=["E"],
            )
        ],
        materials=[
            MaterialSpec(
                id="material:air",
                name="air",
                phase="gas",
                properties=[
                    MaterialProperty(name="relative_permittivity", value=1.0),
                    MaterialProperty(name="relative_permeability", value=1.0),
                    MaterialProperty(name="wave_number", value=0.5),
                    MaterialProperty(name="current_source", value=0.2),
                ],
            )
        ],
        boundary_conditions=[{"id": "bc:farfield", "region_id": "boundary", "field": "E_t", "kind": "farfield", "value": 0.0}],
        targets=[{"name": "field", "field": "E", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result
    assert result.status == "success"
    operator_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_mesh_fem_em_curl_curl_operator")
    operator_payload = json.loads(open(operator_artifact.uri, encoding="utf-8").read())
    assert operator_payload["hcurl_scaffold"]["boundary_condition"] == "natural"
    assert operator_payload["boundary_edge_count"] == 0


def test_mesh_graph_taps_em_curl_curl_supports_complex_frequency_coefficients() -> None:
    geometry = GeometrySpec(id="geometry:mesh-em-complex-square", source=GeometrySource(kind="generated"), dimension=2)
    mesh = generate_mesh(
        GenerateMeshInput(
            geometry=geometry,
            physics=PhysicsSpec(domains=["electromagnetic"]),
            mesh_policy=MeshPolicy(target_element_size=0.55),
            target_backends=["taps"],
        )
    ).mesh
    encoding_output = generate_geometry_encoding(
        GenerateGeometryEncodingInput(geometry=geometry, mesh=mesh, encodings=["mesh_graph"])
    )
    geometry.encodings.extend(encoding_output.encodings)
    problem = PhysicsProblem(
        id="problem:mesh-em-complex-curl-curl-2d",
        user_intent={"raw_request": "solve a lossy frequency-domain electromagnetic curl-curl problem"},
        domain="electromagnetic",
        geometry=geometry,
        mesh=mesh,
        fields=[FieldSpec(name="E", kind="vector")],
        operators=[
            OperatorSpec(
                id="operator:maxwell",
                name="Maxwell curl-curl",
                domain="electromagnetic",
                equation_class="maxwell",
                form="weak",
                fields_out=["E"],
            )
        ],
        materials=[
            MaterialSpec(
                id="material:lossy-dielectric",
                name="lossy dielectric",
                phase="custom",
                properties=[
                    MaterialProperty(name="relative_permittivity", value=[2.5, -0.15]),
                    MaterialProperty(name="relative_permeability", value=1.0),
                    MaterialProperty(name="wave_number", value=[0.45, 0.02]),
                    MaterialProperty(name="current_source", value=[0.5, 0.1]),
                ],
            )
        ],
        boundary_conditions=[{"id": "bc:e", "region_id": "boundary", "field": "E_t", "kind": "dirichlet", "value": 0.0}],
        targets=[{"name": "field", "field": "E", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result
    assert result.status == "success"
    operator_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_mesh_fem_em_curl_curl_operator")
    operator_payload = json.loads(open(operator_artifact.uri, encoding="utf-8").read())
    assert operator_payload["material"]["complex_frequency_domain"] is True
    assert operator_payload["material"]["relative_permittivity"] == pytest.approx([2.5, -0.15])
    assert operator_payload["material"]["wave_number"] == pytest.approx([0.45, 0.02])
    assert operator_payload["rhs"]
    assert any(isinstance(value, list) and abs(value[1]) > 0.0 for value in operator_payload["rhs"])
    solution_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_mesh_fem_em_field")
    solution_payload = json.loads(open(solution_artifact.uri, encoding="utf-8").read())
    assert any(isinstance(value, list) for value in solution_payload["values"])


def test_mesh_graph_taps_em_curl_curl_supports_absorbing_and_port_boundaries() -> None:
    geometry = GeometrySpec(id="geometry:mesh-em-port-square", source=GeometrySource(kind="generated"), dimension=2)
    mesh = generate_mesh(
        GenerateMeshInput(
            geometry=geometry,
            physics=PhysicsSpec(domains=["electromagnetic"]),
            mesh_policy=MeshPolicy(target_element_size=0.5),
            target_backends=["taps"],
        )
    ).mesh
    encoding_output = generate_geometry_encoding(
        GenerateGeometryEncodingInput(geometry=geometry, mesh=mesh, encodings=["mesh_graph"])
    )
    geometry.encodings.extend(encoding_output.encodings)
    problem = PhysicsProblem(
        id="problem:mesh-em-port-curl-curl-2d",
        user_intent={"raw_request": "solve a curl-curl EM problem with an impedance port boundary"},
        domain="electromagnetic",
        geometry=geometry,
        mesh=mesh,
        fields=[FieldSpec(name="E", kind="vector")],
        operators=[
            OperatorSpec(
                id="operator:maxwell",
                name="Maxwell curl-curl",
                domain="electromagnetic",
                equation_class="maxwell",
                form="weak",
                fields_out=["E"],
            )
        ],
        materials=[
            MaterialSpec(
                id="material:air-port",
                name="air",
                phase="gas",
                properties=[
                    MaterialProperty(name="relative_permittivity", value=1.0),
                    MaterialProperty(name="relative_permeability", value=1.0),
                    MaterialProperty(name="wave_number", value=0.55),
                    MaterialProperty(name="current_source", value=0.1),
                ],
            )
        ],
        boundary_conditions=[
            {
                "id": "bc:port",
                "region_id": "boundary",
                "field": "E_t",
                "kind": "custom",
                "value": {"kind": "port", "impedance": [0.8, 0.2], "amplitude": [1.0, -0.25]},
            }
        ],
        targets=[{"name": "field", "field": "E", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result
    assert result.status == "success"
    operator_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_mesh_fem_em_curl_curl_operator")
    operator_payload = json.loads(open(operator_artifact.uri, encoding="utf-8").read())
    assert operator_payload["hcurl_scaffold"]["boundary_condition"] == "port"
    assert operator_payload["boundary_edge_count"] == 0
    assert operator_payload["material"]["boundary_impedance"] == pytest.approx([0.8, 0.2])
    assert operator_payload["material"]["port_amplitude"] == pytest.approx([1.0, -0.25])
    assert operator_payload["material"]["complex_frequency_domain"] is True
    assert any(isinstance(value, list) and abs(value[1]) > 0.0 for value in operator_payload["rhs"])


def test_mesh_graph_taps_em_boundary_policy_selects_region_specific_edges() -> None:
    geometry = GeometrySpec(id="geometry:mesh-em-port-xmin-square", source=GeometrySource(kind="generated"), dimension=2)
    mesh = generate_mesh(
        GenerateMeshInput(
            geometry=geometry,
            physics=PhysicsSpec(domains=["electromagnetic"]),
            mesh_policy=MeshPolicy(target_element_size=0.5),
            target_backends=["taps"],
        )
    ).mesh
    encoding_output = generate_geometry_encoding(
        GenerateGeometryEncodingInput(geometry=geometry, mesh=mesh, encodings=["mesh_graph"])
    )
    graph_payload = json.loads(open(encoding_output.artifacts[0].uri, encoding="utf-8").read())
    assert graph_payload["boundary_edge_sets"]["x_min"]
    assert graph_payload["boundary_edge_sets"]["boundary:x_min"] == graph_payload["boundary_edge_sets"]["x_min"]
    assert len(graph_payload["boundary_edge_sets"]["x_min"]) < len(graph_payload["boundary_edge_sets"]["boundary"])
    geometry.encodings.extend(encoding_output.encodings)
    problem = PhysicsProblem(
        id="problem:mesh-em-port-xmin-curl-curl-2d",
        user_intent={"raw_request": "solve a curl-curl EM problem with a port only on x_min"},
        domain="electromagnetic",
        geometry=geometry,
        mesh=mesh,
        fields=[FieldSpec(name="E", kind="vector")],
        operators=[
            OperatorSpec(
                id="operator:maxwell",
                name="Maxwell curl-curl",
                domain="electromagnetic",
                equation_class="maxwell",
                form="weak",
                fields_out=["E"],
            )
        ],
        materials=[
            MaterialSpec(
                id="material:air-port-xmin",
                name="air",
                phase="gas",
                properties=[
                    MaterialProperty(name="relative_permittivity", value=1.0),
                    MaterialProperty(name="relative_permeability", value=1.0),
                    MaterialProperty(name="wave_number", value=0.55),
                    MaterialProperty(name="current_source", value=0.0),
                ],
            )
        ],
        boundary_conditions=[
            {
                "id": "bc:port-xmin",
                "region_id": "boundary:x_min",
                "field": "E_t",
                "kind": "custom",
                "value": {"kind": "port", "impedance": 1.0, "amplitude": 1.0},
            }
        ],
        targets=[{"name": "field", "field": "E", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result
    assert result.status == "success"
    operator_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_mesh_fem_em_curl_curl_operator")
    operator_payload = json.loads(open(operator_artifact.uri, encoding="utf-8").read())
    assert operator_payload["hcurl_scaffold"]["boundary_condition"] == "port"
    assert operator_payload["active_boundary_edges"] == graph_payload["boundary_edge_sets"]["boundary:x_min"]
    assert operator_payload["active_boundary_edge_count"] == len(graph_payload["boundary_edge_sets"]["boundary:x_min"])
    assert operator_payload["active_boundary_edge_count"] < len(graph_payload["boundary_edge_sets"]["boundary"])


def test_imported_geo_physical_curve_labels_drive_em_boundary_selection(tmp_path) -> None:
    geo_path = tmp_path / "physical_square.geo"
    geo_path.write_text(
        "\n".join(
            [
                'SetFactory("Built-in");',
                "lc = 0.25;",
                "Point(1) = {0, 0, 0, lc};",
                "Point(2) = {1, 0, 0, lc};",
                "Point(3) = {1, 1, 0, lc};",
                "Point(4) = {0, 1, 0, lc};",
                "Line(1) = {1, 2};",
                "Line(2) = {2, 3};",
                "Line(3) = {3, 4};",
                "Line(4) = {4, 1};",
                "Curve Loop(1) = {1, 2, 3, 4};",
                "Plane Surface(1) = {1};",
                'Physical Surface("domain") = {1};',
                'Physical Curve("wall_bottom") = {1};',
                'Physical Curve("wall_right") = {2};',
                'Physical Curve("wall_top") = {3};',
                'Physical Curve("port_left") = {4};',
            ]
        ),
        encoding="utf-8",
    )
    imported = import_geometry(ImportGeometryInput(source=GeometrySource(kind="mesh_file", uri=str(geo_path)))).geometry
    assert imported.dimension == 2
    assert any(region.label == "domain" for region in imported.regions)
    assert any(boundary.label == "port_left" for boundary in imported.boundaries)
    mesh = generate_mesh(
        GenerateMeshInput(
            geometry=imported,
            physics=PhysicsSpec(domains=["electromagnetic"]),
            mesh_policy=MeshPolicy(target_element_size=0.25),
            target_backends=["taps"],
        )
    ).mesh
    encoding_output = generate_geometry_encoding(
        GenerateGeometryEncodingInput(geometry=imported, mesh=mesh, encodings=["mesh_graph"])
    )
    graph_payload = json.loads(open(encoding_output.artifacts[0].uri, encoding="utf-8").read())
    assert graph_payload["boundary_edge_sets"]["boundary:port_left"]
    assert graph_payload["boundary_edge_sets"]["boundary:port_left"] == graph_payload["boundary_edge_sets"]["port_left"]
    assert len(graph_payload["boundary_edge_sets"]["boundary:port_left"]) < len(graph_payload["boundary_edge_sets"]["boundary"])
    port_group = next(group for group in graph_payload["physical_boundary_groups"] if group["name"] == "port_left")
    assert port_group["edge_ids"] == graph_payload["boundary_edge_sets"]["boundary:port_left"]
    assert port_group["solver_native"]["openfoam_patch"] == "port_left"
    assert port_group["solver_native"]["su2_marker"] == "port_left"

    imported.encodings.extend(encoding_output.encodings)
    problem = PhysicsProblem(
        id="problem:imported-geo-port-left-curl-curl-2d",
        user_intent={"raw_request": "solve a curl-curl EM problem with a named imported CAD port boundary"},
        domain="electromagnetic",
        geometry=imported,
        mesh=mesh,
        fields=[FieldSpec(name="E", kind="vector")],
        operators=[
            OperatorSpec(
                id="operator:maxwell",
                name="Maxwell curl-curl",
                domain="electromagnetic",
                equation_class="maxwell",
                form="weak",
                fields_out=["E"],
            )
        ],
        materials=[
            MaterialSpec(
                id="material:air-imported-port",
                name="air",
                phase="gas",
                properties=[
                    MaterialProperty(name="relative_permittivity", value=1.0),
                    MaterialProperty(name="relative_permeability", value=1.0),
                    MaterialProperty(name="wave_number", value=0.5),
                    MaterialProperty(name="current_source", value=0.0),
                ],
            )
        ],
        boundary_conditions=[
            {
                "id": "bc:imported-port-left",
                "region_id": "boundary:port_left",
                "field": "E_t",
                "kind": "custom",
                "value": {"kind": "port", "impedance": 1.0, "amplitude": 1.0},
            }
        ],
        targets=[{"name": "field", "field": "E", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result
    assert result.status == "success"
    operator_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_mesh_fem_em_curl_curl_operator")
    operator_payload = json.loads(open(operator_artifact.uri, encoding="utf-8").read())
    assert operator_payload["hcurl_scaffold"]["boundary_condition"] == "port"
    assert operator_payload["active_boundary_edges"] == graph_payload["boundary_edge_sets"]["boundary:port_left"]


def test_backend_mesh_export_manifest_preserves_physical_boundary_groups(tmp_path) -> None:
    geo_path = tmp_path / "export_square.geo"
    geo_path.write_text(
        "\n".join(
            [
                'SetFactory("Built-in");',
                "lc = 0.3;",
                "Point(1) = {0, 0, 0, lc};",
                "Point(2) = {1, 0, 0, lc};",
                "Point(3) = {1, 1, 0, lc};",
                "Point(4) = {0, 1, 0, lc};",
                "Line(1) = {1, 2};",
                "Line(2) = {2, 3};",
                "Line(3) = {3, 4};",
                "Line(4) = {4, 1};",
                "Curve Loop(1) = {1, 2, 3, 4};",
                "Plane Surface(1) = {1};",
                'Physical Surface("domain") = {1};',
                'Physical Curve("inlet") = {4};',
                'Physical Curve("wall") = {1, 2, 3};',
            ]
        ),
        encoding="utf-8",
    )
    geometry = import_geometry(ImportGeometryInput(source=GeometrySource(kind="mesh_file", uri=str(geo_path)))).geometry
    mesh = generate_mesh(
        GenerateMeshInput(
            geometry=geometry,
            physics=PhysicsSpec(domains=["fluid"]),
            mesh_policy=MeshPolicy(target_element_size=0.3),
            target_backends=["openfoam"],
        )
    ).mesh
    encoding_output = generate_geometry_encoding(
        GenerateGeometryEncodingInput(geometry=geometry, mesh=mesh, encodings=["mesh_graph"])
    )
    geometry.encodings.extend(encoding_output.encodings)
    exported = export_backend_mesh(
        ExportBackendMeshInput(
            geometry=geometry,
            mesh=mesh,
            backend="openfoam",
            geometry_encoding=encoding_output.encodings[0],
        )
    )
    manifest = json.loads(open(exported.manifest.uri, encoding="utf-8").read())
    assert manifest["schema_version"] == "physicsos.backend_mesh_export.v1"
    assert manifest["execution_policy"]["local_tool_invocation"] is False
    assert manifest["target"]["target"] == "constant/polyMesh"
    patches = {item["backend_name"]: item for item in manifest["boundary_exports"]}
    assert {"inlet", "wall"} <= set(patches)
    assert patches["inlet"]["solver_native"]["openfoam_patch"] == "inlet"
    assert patches["wall"]["edge_ids"]
    assert exported.warnings == []


def test_3d_gmsh_physical_surfaces_export_as_solver_face_groups(tmp_path) -> None:
    geo_path = tmp_path / "box_surfaces.geo"
    geo_path.write_text(
        "\n".join(
            [
                'SetFactory("OpenCASCADE");',
                "Box(1) = {0, 0, 0, 1, 1, 1};",
                "Mesh.CharacteristicLengthMin = 0.6;",
                "Mesh.CharacteristicLengthMax = 0.6;",
                "Physical Volume(\"fluid\") = {1};",
                "eps = 1e-6;",
                "inlet[] = Surface In BoundingBox{-eps, -eps, -eps, eps, 1 + eps, 1 + eps};",
                "outlet[] = Surface In BoundingBox{1 - eps, -eps, -eps, 1 + eps, 1 + eps, 1 + eps};",
                "walls[] = Surface In BoundingBox{-eps, -eps, -eps, 1 + eps, 1 + eps, 1 + eps};",
                "walls[] -= inlet[];",
                "walls[] -= outlet[];",
                "Physical Surface(\"inlet\") = inlet[];",
                "Physical Surface(\"outlet\") = outlet[];",
                "Physical Surface(\"wall\") = walls[];",
            ]
        ),
        encoding="utf-8",
    )
    geometry = import_geometry(ImportGeometryInput(source=GeometrySource(kind="mesh_file", uri=str(geo_path)))).geometry
    assert geometry.dimension == 3
    assert {"inlet", "outlet", "wall"} <= {boundary.label for boundary in geometry.boundaries}
    mesh = generate_mesh(
        GenerateMeshInput(
            geometry=geometry,
            physics=PhysicsSpec(domains=["fluid"]),
            mesh_policy=MeshPolicy(target_element_size=0.6),
            target_backends=["openfoam"],
        )
    ).mesh
    encoding_output = generate_geometry_encoding(
        GenerateGeometryEncodingInput(geometry=geometry, mesh=mesh, encodings=["mesh_graph"])
    )
    graph_payload = json.loads(open(encoding_output.artifacts[0].uri, encoding="utf-8").read())
    assert graph_payload["faces"]
    assert graph_payload["boundary_face_sets"]["boundary:inlet"]
    assert graph_payload["boundary_face_sets"]["boundary:outlet"]
    assert graph_payload["boundary_face_sets"]["boundary:wall"]
    assert graph_payload["physical_boundary_groups"]

    geometry.encodings.extend(encoding_output.encodings)
    exported = export_backend_mesh(
        ExportBackendMeshInput(
            geometry=geometry,
            mesh=mesh,
            backend="openfoam",
            geometry_encoding=encoding_output.encodings[0],
        )
    )
    manifest = json.loads(open(exported.manifest.uri, encoding="utf-8").read())
    patches = {item["backend_name"]: item for item in manifest["boundary_exports"]}
    assert {"inlet", "outlet", "wall"} <= set(patches)
    assert patches["inlet"]["dimension"] == 2
    assert patches["inlet"]["face_ids"]
    assert patches["outlet"]["face_ids"]
    assert patches["wall"]["face_ids"]
    assert exported.warnings == []


def test_mesh_conversion_runner_manifest_inlines_exported_msh_and_dry_runs(tmp_path) -> None:
    geo_path = tmp_path / "conversion_box.geo"
    geo_path.write_text(
        "\n".join(
            [
                'SetFactory("OpenCASCADE");',
                "Box(1) = {0, 0, 0, 1, 1, 1};",
                "Mesh.CharacteristicLengthMin = 0.8;",
                "Mesh.CharacteristicLengthMax = 0.8;",
                "Physical Volume(\"fluid\") = {1};",
                "eps = 1e-6;",
                "inlet[] = Surface In BoundingBox{-eps, -eps, -eps, eps, 1 + eps, 1 + eps};",
                "Physical Surface(\"inlet\") = inlet[];",
            ]
        ),
        encoding="utf-8",
    )
    geometry = import_geometry(ImportGeometryInput(source=GeometrySource(kind="mesh_file", uri=str(geo_path)))).geometry
    mesh = generate_mesh(
        GenerateMeshInput(
            geometry=geometry,
            physics=PhysicsSpec(domains=["fluid"]),
            mesh_policy=MeshPolicy(target_element_size=0.8),
            target_backends=["openfoam"],
        )
    ).mesh
    encoding_output = generate_geometry_encoding(
        GenerateGeometryEncodingInput(geometry=geometry, mesh=mesh, encodings=["mesh_graph"])
    )
    export = export_backend_mesh(
        ExportBackendMeshInput(
            geometry=geometry,
            mesh=mesh,
            backend="openfoam",
            geometry_encoding=encoding_output.encodings[0],
        )
    )
    prepared = prepare_mesh_conversion_job(
        PrepareMeshConversionJobInput(
            mesh_export_manifest=export.manifest,
            service_base_url="https://foamvm.vercel.app",
        )
    )
    manifest = json.loads(open(prepared.runner_manifest.uri, encoding="utf-8").read())
    assert manifest["schema_version"] == "physicsos.mesh_conversion_job.v1"
    assert manifest["job_type"] == "mesh_conversion"
    assert manifest["backend"] == "openfoam"
    assert manifest["inputs"]["source_mesh_file"]["content_base64"]
    assert manifest["conversion_plan"]["allowed_converters"] == ["gmshToFoam", "meshio"]
    assert manifest["execution_policy"]["local_external_process_execution"] is False
    assert prepared.warnings == []

    dry_run = submit_mesh_conversion_job(SubmitMeshConversionJobInput(runner_manifest=prepared.runner_manifest, mode="dry_run"))
    response = json.loads(open(dry_run.runner_response.uri, encoding="utf-8").read())
    assert dry_run.submitted is False
    assert dry_run.status == "validated"
    assert response["message"].endswith("no external conversion service or CLI was invoked.")


def test_boundary_labeling_artifact_requires_confirmation_before_apply(tmp_path) -> None:
    geo_path = tmp_path / "labeling_box.geo"
    geo_path.write_text(
        "\n".join(
            [
                'SetFactory("OpenCASCADE");',
                "Box(1) = {0, 0, 0, 1, 1, 1};",
                "Mesh.CharacteristicLengthMin = 0.7;",
                "Mesh.CharacteristicLengthMax = 0.7;",
                "Physical Volume(\"domain\") = {1};",
                "eps = 1e-6;",
                "inlet[] = Surface In BoundingBox{-eps, -eps, -eps, eps, 1 + eps, 1 + eps};",
                "Physical Surface(\"inlet\") = inlet[];",
            ]
        ),
        encoding="utf-8",
    )
    geometry = import_geometry(ImportGeometryInput(source=GeometrySource(kind="mesh_file", uri=str(geo_path)))).geometry
    mesh = generate_mesh(
        GenerateMeshInput(
            geometry=geometry,
            physics=PhysicsSpec(domains=["fluid"]),
            mesh_policy=MeshPolicy(target_element_size=0.7),
            target_backends=["openfoam"],
        )
    ).mesh
    encoding_output = generate_geometry_encoding(
        GenerateGeometryEncodingInput(geometry=geometry, mesh=mesh, encodings=["mesh_graph"])
    )
    labeling = create_boundary_labeling_artifact(
        CreateBoundaryLabelingArtifactInput(geometry=geometry, geometry_encoding=encoding_output.encodings[0])
    )
    payload = json.loads(open(labeling.artifact.uri, encoding="utf-8").read())
    assert payload["policy"]["weak_suggestions_require_confirmation"] is True
    assert payload["policy"]["solver_export_uses_confirmed_labels_only"] is True
    assert payload["viewer_geometry"]["points"]
    assert payload["viewer_geometry"]["faces"]
    assert payload["suggested_boundary_labels"]
    assert payload["confirmed_boundary_labels"] == []
    viewer = create_geometry_labeler_viewer(CreateGeometryLabelerViewerInput(labeling_artifact=labeling.artifact)).viewer
    viewer_text = open(viewer.uri, encoding="utf-8").read()
    assert viewer.kind == "geometry_labeler_viewer"
    assert "PhysicsOS standalone tool" in viewer_text
    assert "physicsos.boundary_labeling.v1" in viewer_text

    applied_empty = apply_boundary_labeling_artifact(
        ApplyBoundaryLabelingArtifactInput(geometry=GeometrySpec(id="geometry:empty-labels", source=geometry.source, dimension=3), labeling_artifact=labeling.artifact)
    )
    assert applied_empty.applied == []

    target_id = next(group["id"] for group in payload["selectable_groups"] if group["name"] == "inlet")
    payload["confirmed_boundary_labels"] = [
        {
            "target_ids": [target_id],
            "boundary_id": "boundary:confirmed_inlet",
            "label": "confirmed_inlet",
            "kind": "inlet",
            "confidence": 1.0,
            "confirmed_by": "user",
        }
    ]
    open(labeling.artifact.uri, "w", encoding="utf-8").write(json.dumps(payload, indent=2))
    applied = apply_boundary_labeling_artifact(
        ApplyBoundaryLabelingArtifactInput(
            geometry=GeometrySpec(id="geometry:confirmed-labels", source=geometry.source, dimension=3),
            labeling_artifact=labeling.artifact,
        )
    )
    assert applied.applied == ["boundary:confirmed_inlet"]
    assert applied.geometry.boundaries[0].kind == "inlet"
    assert applied.geometry.boundaries[0].confidence == 1.0
    assert applied.geometry.entities[0].id == target_id


def test_triangle_p1_assembler_uses_cell_gradients() -> None:
    stiffness, lumped_mass, total_area, elements = _assemble_triangle_stiffness(
        points=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        triangles=[[0, 1, 2]],
    )
    assert total_area == pytest.approx(0.5)
    assert lumped_mass == pytest.approx([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    for actual, expected in zip(elements[0]["grad_phi"], [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(
        elements[0]["local_stiffness"],
        [
            [1.0, -0.5, -0.5],
            [-0.5, 0.5, 0.0],
            [-0.5, 0.0, 0.5],
        ],
    ):
        assert actual == pytest.approx(expected)
    assert stiffness[0][0] == pytest.approx(1.0)
    assert stiffness[0][1] == pytest.approx(-0.5)
    assert stiffness[0][2] == pytest.approx(-0.5)


def test_triangle_nedelec_curl_curl_assembler_uses_edge_dofs() -> None:
    stiffness, edges, total_area, elements = _assemble_triangle_nedelec_curl_curl(
        points=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        triangles=[[0, 1, 2]],
        curl_weight=1.0,
        mass_weight=0.25,
    )
    assert total_area == pytest.approx(0.5)
    assert edges == [(0, 1), (1, 2), (0, 2)]
    assert len(stiffness) == 3
    assert elements[0]["basis"] == "nedelec_first_kind_order1_triangle"
    assert elements[0]["orientation_signs"] == pytest.approx([1.0, 1.0, -1.0])
    local = elements[0]["local_matrix"]
    for i in range(3):
        for j in range(3):
            assert local[i][j] == pytest.approx(local[j][i])
            assert stiffness[i][j] == pytest.approx(stiffness[j][i])
    assert all(stiffness[i][i].real > 0.0 and abs(stiffness[i][i].imag) <= 1e-14 for i in range(3))


def test_triangle_nedelec_order2_scaffold_uses_edge_moments_and_cell_dofs() -> None:
    stiffness, dofs, total_area, elements = _assemble_triangle_nedelec_curl_curl(
        points=[
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.0],
            [0.5, 0.5],
            [0.0, 0.5],
        ],
        triangles=[[0, 1, 2, 3, 4, 5]],
        curl_weight=1.0,
        mass_weight=0.25,
    )
    assert total_area == pytest.approx(0.5)
    assert len(dofs) == 8
    assert sum(1 for dof in dofs if dof["kind"] == "edge_moment") == 6
    assert sum(1 for dof in dofs if dof["kind"] == "cell_interior") == 2
    assert len(stiffness) == 8
    element = elements[0]
    assert element["basis"] == "nedelec_first_kind_order2_hierarchical_scaffold_triangle"
    assert element["edge_moment_dofs_per_edge"] == 2
    assert element["cell_interior_dofs"] == 2
    assert len(element["dofs"]) == 8
    local = element["local_matrix"]
    for i in range(8):
        assert stiffness[i][i].real > 0.0
        for j in range(8):
            assert local[i][j] == pytest.approx(local[j][i])
            assert stiffness[i][j] == pytest.approx(stiffness[j][i])


def test_triangle_p2_assembler_uses_quadratic_cell_gradients() -> None:
    stiffness, lumped_mass, total_area, elements = _assemble_triangle_stiffness(
        points=[
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.0],
            [0.5, 0.5],
            [0.0, 0.5],
        ],
        triangles=[[0, 1, 2, 3, 4, 5]],
    )
    assert total_area == pytest.approx(0.5)
    assert elements[0]["basis"] == "p2_triangle"
    assert len(elements[0]["local_stiffness"]) == 6
    assert len(elements[0]["quadrature"]) == 3
    assert lumped_mass[0] == pytest.approx(0.0, abs=1e-12)
    assert lumped_mass[3] == pytest.approx(1.0 / 6.0)
    for row in elements[0]["local_stiffness"]:
        assert sum(row) == pytest.approx(0.0, abs=1e-12)
    for i in range(6):
        for j in range(6):
            assert stiffness[i][j] == pytest.approx(stiffness[j][i])
    assert all(stiffness[i][i] > 0.0 for i in range(6))


def test_triangle_p3_assembler_uses_generic_lagrange_basis() -> None:
    points = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0 / 3.0, 0.0],
        [2.0 / 3.0, 0.0],
        [2.0 / 3.0, 1.0 / 3.0],
        [1.0 / 3.0, 2.0 / 3.0],
        [0.0, 2.0 / 3.0],
        [0.0, 1.0 / 3.0],
        [1.0 / 3.0, 1.0 / 3.0],
    ]
    stiffness, lumped_mass, total_area, elements = _assemble_triangle_stiffness(
        points=points,
        triangles=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
    )
    assert total_area == pytest.approx(0.5)
    assert elements[0]["basis"] == "p3_triangle"
    assert elements[0]["lagrange_order"] == 3
    assert len(elements[0]["local_stiffness"]) == 10
    assert len(elements[0]["quadrature"]) == 7
    assert sum(lumped_mass) == pytest.approx(0.5)
    for row in elements[0]["local_stiffness"]:
        assert sum(row) == pytest.approx(0.0, abs=1e-10)
    for i in range(10):
        for j in range(10):
            assert stiffness[i][j] == pytest.approx(stiffness[j][i], abs=1e-10)
    assert all(stiffness[i][i] > 0.0 for i in range(10))


def test_triangle_p1_elasticity_element_has_rigid_body_modes() -> None:
    stiffness, lumped_mass, total_area, elements = _assemble_triangle_elasticity_stiffness(
        points=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        triangles=[[0, 1, 2]],
    )
    assert total_area == pytest.approx(0.5)
    assert lumped_mass == pytest.approx([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    assert len(stiffness) == 6
    assert elements[0]["basis"] == "p1_vector_triangle"
    local = elements[0]["local_stiffness"]
    assert len(local) == 6
    for i in range(6):
        for j in range(6):
            assert local[i][j] == pytest.approx(local[j][i])

    rigid_x = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    rigid_y = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    rigid_rotation = [0.0, 0.0, 0.0, 1.0, -1.0, 0.0]
    for mode in [rigid_x, rigid_y, rigid_rotation]:
        internal_force = [sum(local[i][j] * mode[j] for j in range(6)) for i in range(6)]
        assert internal_force == pytest.approx([0.0] * 6, abs=1e-12)


def test_triangle_p2_elasticity_element_has_rigid_body_modes() -> None:
    points = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.5, 0.5],
        [0.0, 0.5],
    ]
    stiffness, lumped_mass, total_area, elements = _assemble_triangle_elasticity_stiffness(
        points=points,
        triangles=[[0, 1, 2, 3, 4, 5]],
    )
    assert total_area == pytest.approx(0.5)
    assert sum(lumped_mass) == pytest.approx(0.5)
    assert len(stiffness) == 12
    assert elements[0]["basis"] == "p2_vector_triangle"
    assert elements[0]["lagrange_order"] == 2
    local = elements[0]["local_stiffness"]
    assert len(local) == 12
    for i in range(12):
        for j in range(12):
            assert local[i][j] == pytest.approx(local[j][i], abs=1e-12)

    rigid_x = []
    rigid_y = []
    rigid_rotation = []
    for x, y in points:
        rigid_x.extend([1.0, 0.0])
        rigid_y.extend([0.0, 1.0])
        rigid_rotation.extend([-y, x])
    for mode in [rigid_x, rigid_y, rigid_rotation]:
        internal_force = [sum(local[i][j] * mode[j] for j in range(12)) for i in range(12)]
        assert internal_force == pytest.approx([0.0] * 12, abs=1e-10)


def test_triangle_p3_elasticity_element_has_rigid_body_modes() -> None:
    points = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0 / 3.0, 0.0],
        [2.0 / 3.0, 0.0],
        [2.0 / 3.0, 1.0 / 3.0],
        [1.0 / 3.0, 2.0 / 3.0],
        [0.0, 2.0 / 3.0],
        [0.0, 1.0 / 3.0],
        [1.0 / 3.0, 1.0 / 3.0],
    ]
    stiffness, lumped_mass, total_area, elements = _assemble_triangle_elasticity_stiffness(
        points=points,
        triangles=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
    )
    assert total_area == pytest.approx(0.5)
    assert sum(lumped_mass) == pytest.approx(0.5)
    assert len(stiffness) == 20
    assert elements[0]["basis"] == "p3_vector_triangle"
    assert elements[0]["lagrange_order"] == 3
    local = elements[0]["local_stiffness"]
    assert len(local) == 20
    assert len(elements[0]["quadrature"]) == 7
    for i in range(20):
        for j in range(20):
            assert local[i][j] == pytest.approx(local[j][i], abs=1e-10)

    rigid_x = []
    rigid_y = []
    rigid_rotation = []
    for x, y in points:
        rigid_x.extend([1.0, 0.0])
        rigid_y.extend([0.0, 1.0])
        rigid_rotation.extend([-y, x])
    for mode in [rigid_x, rigid_y, rigid_rotation]:
        internal_force = [sum(local[i][j] * mode[j] for j in range(20)) for i in range(20)]
        assert internal_force == pytest.approx([0.0] * 20, abs=1e-9)


def test_mesh_graph_taps_solves_linear_elasticity() -> None:
    geometry = GeometrySpec(id="geometry:mesh-elasticity-square", source=GeometrySource(kind="generated"), dimension=2)
    mesh = generate_mesh(
        GenerateMeshInput(
            geometry=geometry,
            physics=PhysicsSpec(domains=["solid"]),
            mesh_policy=MeshPolicy(target_element_size=0.25),
            target_backends=["taps"],
        )
    ).mesh
    encoding_output = generate_geometry_encoding(
        GenerateGeometryEncodingInput(geometry=geometry, mesh=mesh, encodings=["mesh_graph"])
    )
    geometry.encodings.extend(encoding_output.encodings)
    problem = PhysicsProblem(
        id="problem:mesh-elasticity-2d",
        user_intent={"raw_request": "solve small-strain 2D linear elasticity on a square mesh"},
        domain="solid",
        geometry=geometry,
        mesh=mesh,
        fields=[FieldSpec(name="u", kind="vector")],
        operators=[
            OperatorSpec(
                id="operator:elasticity",
                name="Linear elasticity",
                domain="solid",
                equation_class="linear_elasticity",
                form="weak",
                fields_out=["u"],
                source_terms=[{"expression": "body_force", "units": "N/m^3"}],
            )
        ],
        materials=[
            MaterialSpec(
                id="material:test-solid",
                name="test solid",
                phase="solid",
                properties=[
                    MaterialProperty(name="young_modulus", value=12.0, units="Pa"),
                    MaterialProperty(name="poisson_ratio", value=0.25),
                    MaterialProperty(name="constitutive_model", value="plane_strain"),
                    MaterialProperty(name="body_force", value=[0.0, -2.0], units="N/m^3"),
                ],
            )
        ],
        boundary_conditions=[{"id": "bc:u", "region_id": "boundary", "field": "u", "kind": "dirichlet", "value": [0.0, 0.0]}],
        targets=[{"name": "displacement", "field": "u", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result
    residual = estimate_taps_residual(EstimateTAPSResidualInput(problem=problem, taps_problem=taps_problem, result=result)).report
    assert taps_problem.weak_form is not None
    assert taps_problem.weak_form.family == "linear_elasticity"
    assert {coefficient.name for coefficient in taps_problem.coefficients} >= {
        "young_modulus",
        "poisson_ratio",
        "constitutive_model",
        "body_force",
    }
    assert result.backend.startswith("taps:mesh_fem_linear_elasticity")
    assert result.status == "success"
    assert residual.converged
    assert result.residuals["fem_dofs"] == pytest.approx(2.0 * result.residuals["fem_nodes"])
    assert {artifact.kind for artifact in result.artifacts} >= {
        "taps_mesh_fem_elasticity_operator",
        "taps_mesh_fem_displacement_field",
        "taps_iteration_history",
    }
    operator_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_mesh_fem_elasticity_operator")
    operator_payload = json.loads(open(operator_artifact.uri, encoding="utf-8").read())
    assert operator_payload["material"]["young_modulus"] == pytest.approx(12.0)
    assert operator_payload["material"]["poisson_ratio"] == pytest.approx(0.25)
    assert operator_payload["material"]["constitutive_model"] == "plane_strain"
    assert operator_payload["material"]["body_force"] == pytest.approx([0.0, -2.0])


def test_mesh_graph_taps_solves_p2_linear_elasticity() -> None:
    geometry = GeometrySpec(id="geometry:mesh-p2-elasticity-square", source=GeometrySource(kind="generated"), dimension=2)
    mesh = generate_mesh(
        GenerateMeshInput(
            geometry=geometry,
            physics=PhysicsSpec(domains=["solid"]),
            mesh_policy=MeshPolicy(target_element_size=0.45, element_order=2),
            target_backends=["taps"],
        )
    ).mesh
    assert any("triangle 6" in cell_type.lower() for cell_type in mesh.topology.cell_types)
    encoding_output = generate_geometry_encoding(
        GenerateGeometryEncodingInput(geometry=geometry, mesh=mesh, encodings=["mesh_graph"])
    )
    geometry.encodings.extend(encoding_output.encodings)
    problem = PhysicsProblem(
        id="problem:mesh-p2-elasticity-2d",
        user_intent={"raw_request": "solve second-order small-strain 2D linear elasticity on a square mesh"},
        domain="solid",
        geometry=geometry,
        mesh=mesh,
        fields=[FieldSpec(name="u", kind="vector")],
        operators=[
            OperatorSpec(
                id="operator:elasticity",
                name="Linear elasticity",
                domain="solid",
                equation_class="linear_elasticity",
                form="weak",
                fields_out=["u"],
            )
        ],
        materials=[
            MaterialSpec(
                id="material:p2-solid",
                name="P2 test solid",
                phase="solid",
                properties=[
                    MaterialProperty(name="young_modulus", value=5.0),
                    MaterialProperty(name="poisson_ratio", value=0.2),
                    MaterialProperty(name="body_force", value=[0.0, -1.0]),
                ],
            )
        ],
        boundary_conditions=[{"id": "bc:u", "region_id": "boundary", "field": "u", "kind": "dirichlet", "value": [0.0, 0.0]}],
        targets=[{"name": "displacement", "field": "u", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result
    assert result.backend.startswith("taps:mesh_fem_linear_elasticity")
    assert result.status == "success"
    assert result.residuals["fem_basis_order"] == 2.0
    assert result.residuals["fem_dofs"] == pytest.approx(2.0 * result.residuals["fem_nodes"])
    operator_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_mesh_fem_elasticity_operator")
    operator_payload = json.loads(open(operator_artifact.uri, encoding="utf-8").read())
    assert operator_payload["type"] == "triangle_p2_fem_linear_elasticity"
    assert operator_payload["basis_order"] == 2
    assert any(element["basis"] == "p2_vector_triangle" for element in operator_payload["elements"])


def test_mesh_graph_taps_solves_p3_linear_elasticity() -> None:
    geometry = GeometrySpec(id="geometry:mesh-p3-elasticity-square", source=GeometrySource(kind="generated"), dimension=2)
    mesh = generate_mesh(
        GenerateMeshInput(
            geometry=geometry,
            physics=PhysicsSpec(domains=["solid"]),
            mesh_policy=MeshPolicy(target_element_size=0.6, element_order=3),
            target_backends=["taps"],
        )
    ).mesh
    assert any("triangle 10" in cell_type.lower() for cell_type in mesh.topology.cell_types)
    encoding_output = generate_geometry_encoding(
        GenerateGeometryEncodingInput(geometry=geometry, mesh=mesh, encodings=["mesh_graph"])
    )
    graph_payload = json.loads(open(encoding_output.artifacts[0].uri, encoding="utf-8").read())
    assert any(len(cell) == 10 for block in graph_payload["cell_blocks"] for cell in block["cells"] if "triangle" in block["type"].lower())
    geometry.encodings.extend(encoding_output.encodings)
    problem = PhysicsProblem(
        id="problem:mesh-p3-elasticity-2d",
        user_intent={"raw_request": "solve third-order small-strain 2D linear elasticity on a square mesh"},
        domain="solid",
        geometry=geometry,
        mesh=mesh,
        fields=[FieldSpec(name="u", kind="vector")],
        operators=[
            OperatorSpec(
                id="operator:elasticity",
                name="Linear elasticity",
                domain="solid",
                equation_class="linear_elasticity",
                form="weak",
                fields_out=["u"],
            )
        ],
        materials=[
            MaterialSpec(
                id="material:p3-solid",
                name="P3 test solid",
                phase="solid",
                properties=[
                    MaterialProperty(name="young_modulus", value=3.0),
                    MaterialProperty(name="poisson_ratio", value=0.2),
                    MaterialProperty(name="body_force", value=[0.0, -0.5]),
                ],
            )
        ],
        boundary_conditions=[{"id": "bc:u", "region_id": "boundary", "field": "u", "kind": "dirichlet", "value": [0.0, 0.0]}],
        targets=[{"name": "displacement", "field": "u", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result
    assert result.backend.startswith("taps:mesh_fem_linear_elasticity")
    assert result.status == "success"
    assert result.residuals["fem_basis_order"] == 3.0
    operator_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_mesh_fem_elasticity_operator")
    operator_payload = json.loads(open(operator_artifact.uri, encoding="utf-8").read())
    assert operator_payload["type"] == "triangle_p3_fem_linear_elasticity"
    assert operator_payload["basis_order"] == 3
    assert any(element["basis"] == "p3_vector_triangle" for element in operator_payload["elements"])


def test_verification_tools_use_backend_residuals_and_ood_heuristics() -> None:
    problem = PhysicsProblem(
        id="problem:verification-poisson",
        user_intent={"raw_request": "verify a simple Poisson solve"},
        domain="custom",
        geometry=GeometrySpec(
            id="geometry:verification-square",
            source=GeometrySource(kind="generated"),
            dimension=2,
            encodings=[
                {
                    "kind": "occupancy_mask",
                    "uri": "scratch/mock/occupancy_mask.json",
                    "resolution": [8, 8],
                    "target_backend": "taps",
                }
            ],
        ),
        fields=[FieldSpec(name="u", kind="scalar")],
        operators=[
            OperatorSpec(
                id="operator:poisson",
                name="Poisson",
                domain="custom",
                equation_class="poisson",
                form="weak",
                fields_out=["u"],
            )
        ],
        materials=[],
        boundary_conditions=[{"id": "bc:u", "region_id": "boundary", "field": "u", "kind": "dirichlet", "value": 0.0}],
        targets=[{"name": "field", "field": "u", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result
    residuals = compute_physics_residuals(ComputePhysicsResidualsInput(problem=problem, result=result))
    conservation = check_conservation_laws(CheckConservationLawsInput(problem=problem, result=result))
    slices = validate_selected_slices(ValidateSelectedSlicesInput(problem=problem, result=result))
    uncertainty = estimate_uncertainty(EstimateUncertaintyInput(problem=problem, result=result, method="residual_proxy"))
    ood = detect_ood_case(DetectOODCaseInput(problem=problem))
    assert residuals.passes
    assert residuals.normalized_residuals
    assert residuals.artifact is not None
    assert conservation.passes
    assert conservation.artifact.kind == "verification:conservation_laws"
    assert slices.passes
    assert slices.slice_names
    assert slices.artifact.kind == "verification:selected_slices"
    assert 0.0 < uncertainty.confidence <= 1.0
    assert ood.ood_score < 0.5


def test_taps_agent_can_author_reviewed_runtime_extension() -> None:
    problem = _minimal_fluid_problem()
    output = author_taps_runtime_extension(
        AuthorTAPSRuntimeExtensionInput(
            problem=problem,
            purpose="Prototype a missing custom weak-form assembler for this case.",
            entrypoint="custom_solver",
            code="def solve(problem, taps_problem):\n    raise NotImplementedError('draft extension')\n",
        )
    )
    assert output.extension.safety_status == "requires_review"
    assert output.extension.artifact.kind == "taps_runtime_extension"
