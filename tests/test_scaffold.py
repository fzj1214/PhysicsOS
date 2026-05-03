import json
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from physicsos.cli import (
    BANNER,
    _ensure_deepagents_physicsos_config,
    _interactive,
    _launch_deepagents_cli,
    _patch_deepagents_allow_blocking,
    _patch_deepagents_physicsos_tools,
    _patch_deepagents_physicsos_noninteractive_events,
    _patch_deepagents_physicsos_tui_events,
    _patch_deepagents_workspace_paths,
    _physicsos_banner,
    _physicsos_agent_prompt,
    main as cli_main,
)
from physicsos.config import load_config, physicsos_home, runtime_paths
from physicsos.events import PhysicsOSEvent, PhysicsOSEventRenderer, read_physicsos_events, wrap_tool_for_events
from physicsos.agents.prompts import PHYSICSOS_SYSTEM_PROMPT
from physicsos.agents.structured import CoreAgentLLMConfig, call_structured_agent, create_openai_structured_client, structured_agent_event_context
from physicsos.backends import geometry_mesh as geometry_mesh_backend
from physicsos.paths import from_agent_path, to_agent_path
from physicsos.schemas.agents import GeometryMeshAgentInput, TAPSAgentOutput
from physicsos.schemas.boundary import BoundaryConditionSpec, InitialConditionSpec
from physicsos.schemas.common import ArtifactRef, ComputeBudget, Provenance, StrictBaseModel
from physicsos.schemas.contracts import build_physics_problem_contract, review_problem_to_taps_contract
from physicsos.schemas.geometry import BoundaryRegionSpec, GeometryEntity, GeometrySource, GeometrySpec
from physicsos.schemas.geometry import GeometryEncoding
from physicsos.schemas.materials import MaterialProperty, MaterialSpec
from physicsos.schemas.mesh import MeshPolicy
from physicsos.schemas.operators import FieldSpec, OperatorSpec
from physicsos.schemas.operators import PhysicsSpec
from physicsos.schemas.operators import NondimensionalNumber
from physicsos.schemas.problem import PhysicsProblem
from physicsos.schemas.solver import SolverPolicy, SolverResult
from physicsos.schemas.verification import VerificationReport
from physicsos.tools.registry import MAIN_AGENT_TOOLS, SUBAGENT_TOOL_GROUPS, TOOL_REGISTRY
from physicsos.tools.workflow_tools import (
    RunTypedPhysicsOSWorkflowInput,
    build_physics_problem_structured,
    run_typed_physicsos_workflow,
)
from physicsos.tools.problem_tools import CanonicalPhysicsProblemInput, build_physics_problem, canonicalize_physics_problem, BuildPhysicsProblemInput
from physicsos.tools.postprocess_tools import (
    GenerateVisualizationsInput,
    PostprocessPlanInput,
    generate_visualizations,
    plan_postprocess,
    plan_postprocess_structured,
)
from physicsos.tools.solver_tools import (
    EstimateSolverSupportInput,
    PrepareFullSolverCaseInput,
    PrepareOpenFOAMRunnerManifestInput,
    RouteSolverBackendInput,
    SubmitFullSolverJobInput,
    estimate_solver_support,
    prepare_full_solver_case,
    prepare_openfoam_runner_manifest,
    route_solver_backend,
    submit_full_solver_job,
)
from physicsos.tools.surrogate_tools import RouteSurrogateModelInput, RunSurrogateInferenceInput, route_surrogate_model, run_surrogate_inference
from physicsos.tools.taps_tools import (
    BuildTAPSProblemInput,
    EstimateTAPSResidualInput,
    EstimateTAPSSupportInput,
    ExportTAPSBackendBridgeInput,
    FormulateTAPSEquationInput,
    NumericalSolvePlanInput,
    PlanBackendPreparationInput,
    PlanTAPSAdaptiveFallbackInput,
    PrepareTAPSBackendCaseBundleInput,
    RunTAPSBackendInput,
    ValidateNumericalSolvePlanInput,
    ValidateTAPSIRInput,
    AuthorTAPSRuntimeExtensionInput,
    author_taps_runtime_extension,
    build_taps_problem,
    estimate_taps_residual,
    estimate_taps_support,
    export_taps_backend_bridge,
    formulate_taps_equation,
    formulate_taps_equation_structured,
    plan_backend_preparation,
    plan_backend_preparation_structured,
    plan_numerical_solve,
    plan_numerical_solve_structured,
    plan_taps_adaptive_fallback,
    prepare_taps_backend_case_bundle,
    run_taps_backend,
    validate_numerical_solve_plan,
    validate_taps_ir,
)
from physicsos.tools.verification_tools import (
    CheckBoundaryConditionApplicationInput,
    CheckConservationLawsInput,
    ComputePhysicsResidualsInput,
    DetectOODCaseInput,
    EstimateUncertaintyInput,
    ValidateSelectedSlicesInput,
    check_boundary_condition_application,
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
from physicsos.tools.memory_tools import (
    AppendCaseMemoryEventInput,
    CaseMemoryEvent,
    ReadCaseMemoryEventsInput,
    SearchCaseMemoryInput,
    StoreCaseResultInput,
    append_case_memory_event,
    read_case_memory_events,
    search_case_memory,
    store_case_result,
)
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
    GeometryMeshPlanInput,
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
    plan_geometry_mesh,
    plan_geometry_mesh_structured,
    prepare_mesh_conversion_job,
    submit_mesh_conversion_job,
)
from physicsos.workflows import run_physicsos_workflow, run_taps_thermal_workflow
from physicsos.workflows.taps_thermal import build_default_thermal_problem
from physicsos.workflows.universal import _run_geometry_mesh_agent
from physicsos.backends.taps_generic import (
    _assemble_tetra_elasticity_stiffness,
    _assemble_tetra_nedelec_curl_curl,
    _assemble_tetra_raviart_thomas_div,
    _assemble_tetra_stiffness,
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


def _postprocess_plan_payload() -> dict:
    return {
        "visualization_plan": [{"kind": "plot", "fields": [], "description": "Residual and scalar output summary."}],
        "report_sections": ["Executive Summary", "Verification Appendix", "Artifact Manifest"],
        "figure_captions": {},
        "recommendations": ["Review verification result and generated report."],
        "warnings": [],
        "assumptions": ["fake structured postprocess plan"],
    }


def test_tool_registry_has_core_tools() -> None:
    assert "build_physics_problem" in TOOL_REGISTRY
    assert "estimate_solver_support" in TOOL_REGISTRY
    assert "check_conservation_laws" in TOOL_REGISTRY
    assert "validate_selected_slices" in TOOL_REGISTRY
    assert "prepare_full_solver_case" in TOOL_REGISTRY
    assert "prepare_openfoam_runner_manifest" in TOOL_REGISTRY
    assert "submit_full_solver_job" in TOOL_REGISTRY
    assert "apply_boundary_labels" in TOOL_REGISTRY
    assert "create_boundary_labeling_artifact" in TOOL_REGISTRY
    assert "apply_boundary_labeling_artifact" in TOOL_REGISTRY
    assert "create_geometry_labeler_viewer" in TOOL_REGISTRY
    assert "plan_geometry_mesh" in TOOL_REGISTRY
    assert "plan_geometry_mesh_structured" in TOOL_REGISTRY
    assert "plan_postprocess" in TOOL_REGISTRY
    assert "plan_postprocess_structured" in TOOL_REGISTRY
    assert "export_backend_mesh" in TOOL_REGISTRY
    assert "prepare_mesh_conversion_job" in TOOL_REGISTRY
    assert "submit_mesh_conversion_job" in TOOL_REGISTRY
    assert "validate_taps_ir" in TOOL_REGISTRY
    assert "export_taps_backend_bridge" in TOOL_REGISTRY
    assert "plan_numerical_solve" in TOOL_REGISTRY
    assert "plan_numerical_solve_structured" in TOOL_REGISTRY
    assert "validate_numerical_solve_plan" in TOOL_REGISTRY
    assert "plan_backend_preparation" in TOOL_REGISTRY
    assert "plan_backend_preparation_structured" in TOOL_REGISTRY
    assert "validate_backend_preparation_plan" in TOOL_REGISTRY
    assert "plan_taps_adaptive_fallback" in TOOL_REGISTRY
    assert "prepare_taps_backend_case_bundle" in TOOL_REGISTRY
    assert "run_full_solver" in TOOL_REGISTRY
    assert "list_operator_templates" in TOOL_REGISTRY
    assert "recommend_runtime_stack" in TOOL_REGISTRY
    assert "append_case_memory_event" in TOOL_REGISTRY
    assert "read_case_memory_events" in TOOL_REGISTRY
    assert TOOL_REGISTRY["submit_full_solver_job"].requires_approval is True
    assert TOOL_REGISTRY["run_full_solver"].requires_approval is True
    assert TOOL_REGISTRY["submit_mesh_conversion_job"].requires_approval is True


def test_subagent_tool_groups_are_scoped_and_registered() -> None:
    main_tool_names = {tool.__name__ for tool in MAIN_AGENT_TOOLS}
    assert "build_physics_problem" in main_tool_names
    assert "run_typed_physicsos_workflow" in main_tool_names
    assert "run_taps_backend" not in main_tool_names
    for agent_name, tools in SUBAGENT_TOOL_GROUPS.items():
        assert tools, agent_name
        tool_names = {tool.__name__ for tool in tools}
        assert main_tool_names <= tool_names
        assert {
            "search_knowledge_base",
            "build_knowledge_context",
            "search_case_memory",
            "append_case_memory_event",
            "read_case_memory_events",
        } <= tool_names
        for tool in tools:
            assert tool.__name__ in TOOL_REGISTRY
    assert "run_taps_backend" in {tool.__name__ for tool in SUBAGENT_TOOL_GROUPS["taps-agent"]}
    assert "run_taps_backend" not in {tool.__name__ for tool in SUBAGENT_TOOL_GROUPS["postprocess-agent"]}
    assert "generate_mesh" in {tool.__name__ for tool in SUBAGENT_TOOL_GROUPS["geometry-mesh-agent"]}
    assert "run_full_solver" in {tool.__name__ for tool in SUBAGENT_TOOL_GROUPS["solver-agent"]}
    assert "route_solver_backend" in {tool.__name__ for tool in SUBAGENT_TOOL_GROUPS["solver-agent"]}
    assert "route_solver_backend" not in {tool.__name__ for tool in SUBAGENT_TOOL_GROUPS["taps-agent"]}


def test_agent_paths_are_forward_slash_and_round_trip(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    native = workspace / "scratch" / "case-1" / "result.json"
    agent_path = to_agent_path(native, workspace=workspace)
    assert agent_path == "/workspace/scratch/case-1/result.json"
    assert "\\" not in agent_path
    assert from_agent_path(agent_path, workspace=workspace) == native
    assert from_agent_path("/workspace", workspace=workspace) == workspace
    assert to_agent_path("https://example.com/a\\b", workspace=workspace) == "https://example.com/a\\b"


def test_deepagents_cli_server_graph_uses_scoped_physicsos_tools(tmp_path) -> None:
    pytest.importorskip("deepagents_cli")
    _patch_deepagents_allow_blocking()
    _patch_deepagents_physicsos_tools()

    import deepagents_cli.server as cli_server
    import deepagents_cli.server_manager as server_manager

    server_manager._scaffold_workspace(tmp_path)
    command = cli_server._build_server_cmd(tmp_path / "langgraph.json", host="127.0.0.1", port=2024)
    source = (tmp_path / "server_graph.py").read_text(encoding="utf-8")
    assert "--allow-blocking" in command
    assert "MAIN_AGENT_TOOLS" in source
    assert "SUBAGENT_TOOL_GROUPS" in source
    assert "load_scoped_subagents" in source
    assert "_patch_deepagents_workspace_paths" in source
    assert "PHYSICSOS_TOOLS" not in source


def test_deepagents_cli_uses_workspace_virtual_paths(tmp_path) -> None:
    pytest.importorskip("deepagents_cli")
    _patch_deepagents_workspace_paths()

    import deepagents_cli.agent as cli_agent
    from deepagents.backends import LocalShellBackend

    prompt = cli_agent.get_system_prompt(assistant_id="physicsos", cwd=tmp_path)
    assert "/workspace/scratch/result.png" in prompt
    assert "D:\\..." in prompt
    assert "filesystem tools require `/workspace/...` paths" in prompt

    backend = LocalShellBackend(root_dir=tmp_path, inherit_env=True)
    composite = __import__("deepagents.backends").backends.CompositeBackend(default=backend, routes={})

    class DummyAgent:
        pass

    def fake_create_cli_agent(*args, **kwargs):
        return DummyAgent(), composite

    mp = pytest.MonkeyPatch()
    mp.setattr(cli_agent, "create_cli_agent", fake_create_cli_agent)
    try:
        _patch_deepagents_workspace_paths()
        _, patched_backend = cli_agent.create_cli_agent()
        write_result = patched_backend.write("/workspace/output/plot.txt", "ok")
        assert write_result.error is None
        assert (tmp_path / "output" / "plot.txt").read_text(encoding="utf-8") == "ok"
    finally:
        mp.undo()


def test_tool_event_payloads_expose_workspace_paths(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    native = workspace / "scratch" / "plot.png"
    native.parent.mkdir()
    native.write_bytes(b"png")

    monkeypatch.setenv("PHYSICSOS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(workspace))
    monkeypatch.setenv("PHYSICSOS_EVENT_LOG", str(tmp_path / "events.jsonl"))

    class Output(StrictBaseModel):
        artifact: ArtifactRef

    def make_artifact() -> Output:
        return Output(artifact=ArtifactRef(uri=str(native), kind="image", format="png"))

    wrapped = wrap_tool_for_events(make_artifact)
    result = wrapped()
    assert result.artifact.uri == str(native)

    events = read_physicsos_events(tmp_path / "events.jsonl")
    output = [event for event in events if event.event == "tool.output"][-1].payload["output"]
    assert output["artifact"]["uri"] == "/workspace/scratch/plot.png"
    assert output["artifact"]["native_uri"] == str(native)


def test_physicsos_prompts_make_typed_workflow_canonical() -> None:
    prompt = _physicsos_agent_prompt()
    assert "run_typed_physicsos_workflow" in PHYSICSOS_SYSTEM_PROMPT
    assert "run_typed_physicsos_workflow" in prompt
    assert "Do not call DeepAgents `task`" in prompt
    assert "typed workflow owns those stages" in prompt
    assert "Prefer TAPS-first reasoning through the typed workflow" in prompt


def test_deepagents_tui_stream_renders_physicsos_custom_events() -> None:
    pytest.importorskip("deepagents_cli")
    import deepagents_cli.textual_adapter as textual_adapter

    _patch_deepagents_physicsos_tui_events()
    assert getattr(textual_adapter.execute_task_textual, "_physicsos_tui_events") is True
    assert getattr(textual_adapter.execute_task_textual, "_physicsos_stream_modes") == ("messages", "updates", "custom")
    constants = set(textual_adapter.execute_task_textual.__code__.co_consts)
    names = set(textual_adapter.execute_task_textual.__code__.co_names)
    assert "custom" in constants
    assert "PhysicsOSEventRenderer" in names
    assert "collect_physicsos_events" in names
    assert "AppMessage" in textual_adapter.execute_task_textual.__globals__


def test_deepagents_noninteractive_stream_renders_physicsos_custom_events() -> None:
    pytest.importorskip("deepagents_cli")
    import deepagents_cli.non_interactive as non_interactive

    _patch_deepagents_physicsos_noninteractive_events()
    assert getattr(non_interactive._stream_agent, "_physicsos_noninteractive_stream_modes") == (
        "messages",
        "updates",
        "custom",
    )
    assert getattr(non_interactive._process_stream_chunk, "_physicsos_noninteractive_events") is True


def test_cli_without_args_starts_interactive_welcome(capsys) -> None:
    with patch("builtins.input", side_effect=["exit"]):
        assert cli_main(["legacy-repl"]) == 0
    output = capsys.readouterr().out
    assert BANNER == "PhysicsOS\nPhysicsOS"
    assert output.count("PhysicsOS") >= 2
    assert "TAPS-first physics simulation agent" in output
    assert "Commands" in output
    assert "paths" in output


def test_cli_defaults_to_official_deepagents_cli(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_cli_main():
        captured["argv"] = list(__import__("sys").argv)

    monkeypatch.setenv("PHYSICSOS_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("PHYSICSOS_OPENAI_BASE_URL", "https://api.tu-zi.com/v1")
    monkeypatch.setenv("PHYSICSOS_OPENAI_MODEL", "gpt-5.4")
    monkeypatch.delenv("PYTHONUTF8", raising=False)
    monkeypatch.delenv("PYTHONIOENCODING", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    with patch("deepagents_cli.cli_main", side_effect=fake_cli_main):
        assert _launch_deepagents_cli(["--message", "hello"]) == 0
    assert captured["argv"][:3] == ["deepagents", "--model", "openai:gpt-5.4"]
    assert "--model-params" in captured["argv"]
    assert "--agent" in captured["argv"]
    assert "physicsos" in captured["argv"]
    assert __import__("os").environ["OPENAI_API_KEY"] == "test-key"
    assert __import__("os").environ["PYTHONUTF8"] == "1"
    assert __import__("os").environ["PYTHONIOENCODING"] == "utf-8"
    assert (tmp_path / ".deepagents" / "physicsos" / "AGENTS.md").exists()
    assert (tmp_path / ".deepagents" / "physicsos" / "agents" / "taps-agent" / "AGENTS.md").exists()
    assert "PhysicsOS" in _physicsos_banner()
    assert "██████" in _physicsos_banner()


def test_deepagents_env_keeps_explicit_python_encoding(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_cli_main():
        captured["argv"] = list(__import__("sys").argv)

    monkeypatch.setenv("PYTHONUTF8", "0")
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8:replace")
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    with patch("deepagents_cli.cli_main", side_effect=fake_cli_main):
        assert _launch_deepagents_cli(["--message", "hello"]) == 0

    assert captured["argv"][0] == "deepagents"
    assert __import__("os").environ["PYTHONUTF8"] == "0"
    assert __import__("os").environ["PYTHONIOENCODING"] == "utf-8:replace"


def test_physicsos_config_json_is_created_and_used(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PHYSICSOS_HOME", str(tmp_path / "physicsos-home"))
    config = load_config()
    path = tmp_path / "physicsos-home" / "config.json"
    assert path.exists()
    assert config["model"]["name"] == "gpt-5.4"
    assert config["model"]["base_url"] == "https://api.tu-zi.com/v1"


def test_physicsos_config_recovers_unescaped_windows_paths(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        '{\n'
        '  "model": {"api_key": "sk-test", "name": "gpt-test"},\n'
        '  "storage": {"home": "C:\\Users\\Name\\.physicsos"}\n'
        '}\n',
        encoding="utf-8",
    )
    config = load_config(path)
    assert config["model"]["api_key"] == "sk-test"
    assert config["storage"]["home"].endswith("\\.physicsos")


def test_openai_structured_client_uses_config_without_leaking_key(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model": {
                    "name": "gpt-test",
                    "api_key": "sk-secret123456",
                    "base_url": "https://example.invalid/v1",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PHYSICSOS_CONFIG", str(config_path))

    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)

            class Message:
                content = '{"value": 3}'

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    class FakeOpenAI:
        def __init__(self, api_key, base_url):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    client = create_openai_structured_client()

    class Input(StrictBaseModel):
        prompt: str

    class Output(StrictBaseModel):
        value: int

    result = call_structured_agent(
        agent_name="real-client-test",
        input_model=Input(prompt="x"),
        output_model=Output,
        system_prompt="Return value.",
        client=client,
        config=CoreAgentLLMConfig(model="gpt-test", max_structured_attempts=1),
    )
    assert result.output is not None
    assert result.output.value == 3
    assert captured["model"] == "gpt-test"
    assert captured["base_url"] == "https://example.invalid/v1"
    assert getattr(client, "api_key") == "sk-s...3456"


def test_cli_paths_prints_runtime_storage(capsys, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PHYSICSOS_HOME", str(tmp_path / "physicsos-home"))
    assert cli_main(["paths"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["home"] == str(tmp_path / "physicsos-home")
    assert payload["config_json"].endswith("config.json")
    assert payload["cloud_config"].endswith("config.json")
    assert payload["case_memory"].endswith("data\\case_memory.jsonl") or payload["case_memory"].endswith("data/case_memory.jsonl")
    assert payload["knowledge_base"].endswith("physicsos_knowledge.sqlite")


def test_cli_runner_download_commands_call_cloud_client(capsys, monkeypatch, tmp_path) -> None:
    class FakeFoamVMClient:
        @classmethod
        def from_config(cls):
            return cls()

        def download_artifact(self, job_id, artifact_id, output_dir="."):
            return {"job_id": job_id, "artifact_id": artifact_id, "path": str(Path(output_dir) / "VTK.tar.gz")}

        def download_all_artifacts(self, job_id, output_dir="."):
            return {"job_id": job_id, "downloaded": [{"path": str(Path(output_dir) / "VTK.tar.gz")}]}

    monkeypatch.setattr("physicsos.cli.FoamVMClient", FakeFoamVMClient)
    assert cli_main(["runner", "download", "job:1", "artifact:1", "--output-dir", str(tmp_path)]) == 0
    single = json.loads(capsys.readouterr().out)
    assert cli_main(["runner", "download-all", "job:1", "--output-dir", str(tmp_path)]) == 0
    all_payload = json.loads(capsys.readouterr().out)
    assert single["artifact_id"] == "artifact:1"
    assert single["path"] == str(tmp_path / "VTK.tar.gz")
    assert all_payload["downloaded"][0]["path"] == str(tmp_path / "VTK.tar.gz")


def test_foamvm_client_downloads_relative_artifact_urls(monkeypatch, tmp_path) -> None:
    from physicsos.cloud.foamvm_client import FoamVMClient

    calls = []

    class FakeFoamVMClient(FoamVMClient):
        def job_artifacts(self, job_id):
            return {
                "artifacts": [
                    {
                        "id": "artifact:1",
                        "filename": "VTK.tar.gz",
                        "url": "/api/files?id=artifact%3A1",
                    }
                ]
            }

        def _request_bytes(self, path_or_url):
            calls.append(path_or_url)
            return b"vtk-bytes", None

    client = FakeFoamVMClient(runner_url="https://foamvm.example", access_token="token:test")
    result = client.download_artifact("job:1", "artifact:1", output_dir=tmp_path)
    assert calls == ["/api/files?id=artifact%3A1"]
    assert result["path"] == str(tmp_path / "VTK.tar.gz")
    assert (tmp_path / "VTK.tar.gz").read_bytes() == b"vtk-bytes"


def test_foamvm_client_downloads_all_artifacts(tmp_path) -> None:
    from physicsos.cloud.foamvm_client import FoamVMClient

    class FakeFoamVMClient(FoamVMClient):
        def job_artifacts(self, job_id):
            return {
            "artifacts": [
                {
                    "id": "artifact:1",
                    "filename": "VTK.tar.gz",
                    "url": "/api/files?id=artifact%3A1",
                },
                {
                    "id": "artifact:2",
                    "filename": "log.simpleFoam",
                    "url": "/api/files?id=artifact%3A2",
                },
            ]
        }

        def _request_bytes(self, path_or_url):
            return path_or_url.encode(), None

    result = FakeFoamVMClient(runner_url="https://foamvm.example", access_token="token:test").download_all_artifacts(
        "job:1", output_dir=tmp_path
    )
    assert [item["filename"] for item in result["downloaded"]] == ["VTK.tar.gz", "log.simpleFoam"]
    assert (tmp_path / "VTK.tar.gz").exists()
    assert (tmp_path / "log.simpleFoam").exists()


def test_interactive_cli_routes_natural_language_to_agent(capsys, monkeypatch, tmp_path) -> None:
    class FakeAgent:
        def __init__(self) -> None:
            self.calls = []

        def invoke(self, payload):
            self.calls.append(payload)
            return {
                "messages": [{"role": "assistant", "content": "agent response"}],
                "physicsos_events": [
                    {
                        "run_id": "run:test",
                        "case_id": "problem:test",
                        "event": "agent.output",
                        "stage": "taps",
                        "status": "complete",
                        "summary": "backend=taps:thermal_1d",
                    }
                ],
            }

    fake_agent = FakeAgent()
    monkeypatch.setenv("PHYSICSOS_HOME", str(tmp_path / "physicsos-home"))
    with patch("builtins.input", side_effect=["solve heat equation on a rod", "/exit"]):
        assert _interactive(agent=fake_agent) == 0
    output = capsys.readouterr().out
    assert "[taps] backend=taps:thermal_1d" in output
    assert "agent response" in output
    assert fake_agent.calls[0]["messages"][0]["content"] == "solve heat equation on a rod"
    assert (tmp_path / "physicsos-home" / "history.jsonl").exists()


def test_physicsos_home_uses_environment_override(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PHYSICSOS_HOME", str(tmp_path / "home"))
    assert physicsos_home() == tmp_path / "home"
    assert runtime_paths().knowledge_base == tmp_path / "home" / "data" / "knowledge" / "physicsos_knowledge.sqlite"
    assert runtime_paths().config_json == tmp_path / "home" / "config.json"


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


def test_deepagents_runtime_uses_backend_factory() -> None:
    pytest.importorskip("deepagents")
    kwargs = build_runtime_kwargs(
        DeepAgentsRuntimeConfig(
            enable_filesystem_backend=True,
            enable_memory_store=False,
            enable_checkpointer=False,
        )
    )
    assert callable(kwargs["backend"])
    backend = kwargs["backend"](object())
    assert backend is not None


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


def test_case_memory_events_are_shared_append_only_records(tmp_path) -> None:
    events_path = tmp_path / "case_memory_events.jsonl"
    event = CaseMemoryEvent(
        run_id="run:test",
        case_id="problem:test",
        stage="geometry",
        event="agent_output",
        summary="mesh ready",
        payload={"mesh_ready": True},
    )
    appended = append_case_memory_event(AppendCaseMemoryEventInput(event=event, events_uri=str(events_path)))
    append_case_memory_event(AppendCaseMemoryEventInput(event=event, events_uri=str(events_path)))
    read = read_case_memory_events(ReadCaseMemoryEventsInput(case_id="problem:test", events_uri=str(events_path)))
    assert appended.appended
    assert len(read.events) == 1
    assert read.events[0].payload["mesh_ready"] is True


def test_physicsos_event_renderer_outputs_compact_stage_lines() -> None:
    event = PhysicsOSEvent(
        run_id="run:test",
        case_id="problem:test",
        event="agent.output",
        stage="taps",
        status="complete",
        summary="backend=taps:thermal_1d",
    )
    assert PhysicsOSEventRenderer().render(event) == "[taps] backend=taps:thermal_1d"


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
    assert result.run_id.startswith("workflow:")
    assert result.case_memory_context is not None
    assert result.solver_result.backend == "taps:thermal_1d"
    assert result.geometry is not None
    assert result.taps_problem is not None
    assert result.taps_residual is not None
    assert result.taps is not None
    assert result.taps.handoff.agent_name == "taps-agent"
    assert result.taps.result == result.solver_result
    assert result.solver.handoff.agent_name == "taps-agent"
    assert result.verification_agent.report == result.verification
    assert result.postprocess_agent.result == result.postprocess
    assert result.case_memory.stored == result.case_store
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
    assert [step.name for step in result.trace[:3]] == ["problem", "geometry-mesh-agent", "validate_physics_problem"]
    assert result.trace[-1].name == "case-memory"
    assert result.state.geometry == result.geometry
    assert result.state.taps == result.taps
    assert result.state.solver == result.solver
    event_names = [event.event for event in result.events]
    assert event_names[0] == "workflow.started"
    assert "case_memory.hit" in event_names
    assert "workflow.completed" in event_names
    rendered = PhysicsOSEventRenderer().render_many(result.events)
    assert any(line.startswith("[taps]") for line in rendered)
    event_log_name = "events-" + "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in result.run_id) + ".jsonl"
    event_log = read_physicsos_events(runtime_paths().sessions / event_log_name)
    assert any(event.event == "workflow.completed" for event in event_log)


def test_core_workflow_agent_outputs_are_strict_pydantic_contracts() -> None:
    result = run_physicsos_workflow(use_knowledge=False, taps_rank=8)
    payload = result.taps.model_dump(mode="json")
    assert TAPSAgentOutput.model_validate(payload).handoff.agent_name == "taps-agent"
    assert result.state.problem_contract is not None
    assert result.taps.contract_review is not None
    assert result.taps.contract_review.status == "accepted"
    assert result.contract_review == result.taps.contract_review
    with pytest.raises(ValidationError):
        TAPSAgentOutput.model_validate({"support": payload["support"]})


def test_structured_agent_retries_until_pydantic_output_validates() -> None:
    calls = []

    class Output(StrictBaseModel):
        value: int

    class Input(StrictBaseModel):
        prompt: str

    def fake_client(request):
        calls.append(request)
        if len(calls) == 1:
            return "not-json"
        if len(calls) == 2:
            return {"value": "bad"}
        return {"value": 7}

    result = call_structured_agent(
        agent_name="test-agent",
        input_model=Input(prompt="return a value"),
        output_model=Output,
        system_prompt="Return structured output.",
        client=fake_client,
        config=CoreAgentLLMConfig(max_structured_attempts=3),
    )
    assert result.status == "accepted"
    assert result.output is not None
    assert result.output.value == 7
    assert len(result.attempts) == 3
    assert result.attempts[0].validation_errors
    assert result.attempts[1].validation_errors


def test_structured_agent_writes_attempt_events_and_artifacts(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    event_log = tmp_path / "events.jsonl"
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(workspace))
    monkeypatch.setenv("PHYSICSOS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PHYSICSOS_EVENT_LOG", str(event_log))
    events = []
    calls = []

    class Output(StrictBaseModel):
        value: int

    class Input(StrictBaseModel):
        prompt: str

    def fake_client(request):
        calls.append(request)
        if len(calls) == 1:
            return {"value": "bad"}
        return {"value": 7}

    with structured_agent_event_context(run_id="workflow:test", case_id="problem:test", events=events):
        result = call_structured_agent(
            agent_name="test-structured-agent",
            input_model=Input(prompt="return a value"),
            output_model=Output,
            system_prompt="Return structured output.",
            client=fake_client,
            config=CoreAgentLLMConfig(max_structured_attempts=2),
        )

    assert result.output is not None
    assert [event.status for event in events] == ["retrying", "accepted"]
    assert [event.event for event in events] == ["validation.retry", "agent.output"]
    assert all(event.artifacts for event in events)
    assert all(event.artifacts[0].uri.startswith("/workspace/scratch/structured_agents/") for event in events)
    native_artifact = from_agent_path(events[-1].artifacts[0].uri, workspace=workspace)
    payload = json.loads(native_artifact.read_text(encoding="utf-8"))
    assert payload["raw_response"]
    assert payload["parsed"]["value"] == 7
    logged_events = read_physicsos_events(event_log)
    assert any(event.stage == "test-structured-agent" for event in logged_events)


def test_call_structured_agent_semantic_validator_retries_in_single_attempt_stream(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(workspace))
    monkeypatch.setenv("PHYSICSOS_HOME", str(tmp_path / "home"))
    events = []
    calls = []

    class Output(StrictBaseModel):
        value: int

    class Input(StrictBaseModel):
        prompt: str

    def fake_client(request):
        calls.append(request)
        if len(calls) == 1:
            return {"value": -1}
        return {"value": 3}

    with structured_agent_event_context(run_id="workflow:semantic", case_id="problem:test", events=events):
        result = call_structured_agent(
            agent_name="semantic-agent",
            input_model=Input(prompt="return a positive value"),
            output_model=Output,
            system_prompt="Return structured output.",
            client=fake_client,
            config=CoreAgentLLMConfig(max_structured_attempts=3),
            semantic_validator=lambda output: ["value must be positive"] if output.value <= 0 else [],
            semantic_feedback_builder=lambda output, errors: [
                json.dumps({"invalid_output": output.model_dump(mode="json"), "errors": errors})
            ],
        )

    assert result.output is not None
    assert result.output.value == 3
    assert [event.payload["attempt"] for event in events] == [1, 2]
    assert [event.payload["max_attempts"] for event in events] == [3, 3]
    assert [event.event for event in events] == ["validation.retry", "agent.output"]
    assert events[0].payload["validation_errors"] == ["value must be positive"]
    assert json.loads(calls[1]["validation_feedback"][0])["invalid_output"] == {"value": -1}


def test_llm_build_physics_problem_falls_back_without_client(monkeypatch) -> None:
    monkeypatch.setenv("PHYSICSOS_CORE_AGENTS_MODE", "llm")
    output = run_typed_physicsos_workflow(
        RunTypedPhysicsOSWorkflowInput(user_request="simulate one-dimensional steady heat conduction", use_knowledge=False)
    )
    assert output.workflow is not None
    assert output.build.problem is not None
    assert any("deterministic fallback" in assumption for assumption in output.build.assumptions)


def test_llm_build_physics_problem_falls_back_after_structured_failures(monkeypatch) -> None:
    monkeypatch.setenv("PHYSICSOS_CORE_AGENTS_MODE", "llm")
    calls = []

    def broken_client(request):
        calls.append(request)
        return {"problem": {"not": "a PhysicsProblem"}, "missing_inputs": [], "assumptions": []}

    output = run_typed_physicsos_workflow(
        RunTypedPhysicsOSWorkflowInput(
            user_request="simulate one-dimensional steady heat conduction",
            use_knowledge=False,
            core_agents_mode="llm",
        ),
        structured_client=broken_client,
    )
    assert output.workflow is not None
    assert output.build.problem is not None
    agent_names = [call["agent_name"] for call in calls]
    assert agent_names.count("build-physics-problem-agent") == 3
    assert agent_names.count("taps-formulation-agent") == 3
    assert any("Structured LLM problem extraction failed" in assumption for assumption in output.build.assumptions)


def test_llm_build_physics_problem_uses_structured_client(monkeypatch) -> None:
    monkeypatch.setenv("PHYSICSOS_CORE_AGENTS_MODE", "llm")
    problem = build_default_thermal_problem()

    def fake_client(_request):
        return {"problem": problem.model_dump(mode="json"), "missing_inputs": [], "assumptions": ["llm extracted typed problem"]}

    built = build_physics_problem_structured(
        BuildPhysicsProblemInput(user_request="simulate a rod with fixed end temperatures"),
        client=fake_client,
    )
    assert built.problem is not None
    assert built.problem.id == problem.id
    assert built.assumptions == ["llm extracted typed problem"]

    output = run_typed_physicsos_workflow(
        RunTypedPhysicsOSWorkflowInput(
            user_request="simulate a rod with fixed end temperatures",
            use_knowledge=False,
            core_agents_mode="llm",
        ),
        structured_client=fake_client,
    )
    assert output.workflow is not None
    assert output.workflow.problem.id == problem.id
    assert output.workflow.taps is not None
    assert output.workflow.taps.contract_review is not None
    assert output.workflow.taps.contract_review.status == "accepted"


def test_taps_contract_review_rejects_boundary_value_drift() -> None:
    problem = build_default_thermal_problem()
    contract = build_physics_problem_contract(problem)
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    assert review_problem_to_taps_contract(contract, taps_problem, plan).status == "accepted"

    changed = taps_problem.boundary_conditions[0].model_copy(update={"value": 999.0})
    drifted = taps_problem.model_copy(update={"boundary_conditions": [changed, *taps_problem.boundary_conditions[1:]]})
    review = review_problem_to_taps_contract(contract, drifted, plan)
    assert review.status == "needs_retry"
    assert any("boundary" in error.lower() for error in review.errors)


def test_taps_contract_review_accepts_operator_family_synonyms() -> None:
    problem = build_default_thermal_problem()
    contract = build_physics_problem_contract(problem)
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    synonym_weak_form = taps_problem.weak_form.model_copy(update={"family": "Galerkin weak form of 1D steady diffusion/heat conduction"})
    synonym_taps_problem = taps_problem.model_copy(update={"weak_form": synonym_weak_form})
    review = review_problem_to_taps_contract(contract, synonym_taps_problem, plan)
    assert review.status == "accepted"
    assert review.reviewed_items["operator_family_preserved"] is True


def test_llm_taps_formulation_uses_structured_client_for_custom_smooth_pde(monkeypatch) -> None:
    monkeypatch.setenv("PHYSICSOS_CORE_AGENTS_MODE", "llm")
    problem = PhysicsProblem(
        id="problem:llm-custom-smooth",
        user_intent={"raw_request": "simulate a smooth steady Cahn-Hilliard phase-field on a square"},
        domain="custom",
        geometry=GeometrySpec(id="geometry:llm-square", source=GeometrySource(kind="generated"), dimension=2),
        fields=[FieldSpec(name="c", kind="scalar", location="node"), FieldSpec(name="mu", kind="scalar", location="node")],
        operators=[
            OperatorSpec(
                id="operator:cahn_hilliard",
                name="Steady Cahn-Hilliard",
                domain="custom",
                equation_class="cahn_hilliard",
                form="weak",
                fields_out=["c", "mu"],
            )
        ],
        materials=[MaterialSpec(id="material:mixture", name="mixture", phase="mixture", properties=[MaterialProperty(name="mobility", value=1.0), MaterialProperty(name="epsilon", value=0.02)])],
        boundary_conditions=[
            BoundaryConditionSpec(id="bc:left", region_id="boundary:x_min", field="c", kind="neumann", value=0.0),
            BoundaryConditionSpec(id="bc:right", region_id="boundary:x_max", field="c", kind="neumann", value=0.0),
        ],
        targets=[{"name": "phase_field", "field": "c", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    calls = []

    def fake_client(request):
        calls.append(request)
        if request["agent_name"] == "build-physics-problem-agent":
            return {"problem": problem.model_dump(mode="json"), "missing_inputs": [], "assumptions": ["llm extracted custom smooth PDE"]}
        if request["agent_name"] == "geometry-mesh-planning-agent":
            return {
                "mesh_policy": {"strategy": "unstructured", "target_element_size": 0.2, "element_order": 1},
                "requested_encodings": ["mesh_graph"],
                "target_backends": ["taps"],
                "require_boundary_confirmation": False,
                "boundary_confidence_threshold": 0.7,
                "assumptions": ["geometry plan"],
                "warnings": [],
            }
        if request["agent_name"] == "postprocess-planning-agent":
            return _postprocess_plan_payload()
        if request["agent_name"] == "numerical-solve-planning-agent":
            return {
                "problem_id": problem.id,
                "status": "fallback_required",
                "solver_family": "cahn_hilliard",
                "backend_target": "taps",
                "field_bindings": {"primary": "c", "secondary": "mu"},
                "discretization": {"dimension": 2, "node_counts": {"x": 32, "y": 32}, "element_order": 1, "quadrature_order": 2},
                "coefficient_bindings": [{"name": "diffusion", "role": "diffusion", "value": 1.0, "source_name": "fallback"}],
                "source_bindings": [{"name": "zero_source", "value": 0.0, "expression": "0"}],
                "boundary_condition_bindings": [
                    {"id": "bc:left", "region_id": "boundary:x_min", "field": "c", "kind": "neumann", "value": 0.0},
                    {"id": "bc:right", "region_id": "boundary:x_max", "field": "c", "kind": "neumann", "value": 0.0},
                ],
                "expected_artifacts": ["taps_weak_form_ir"],
                "validation_checks": ["unsupported_local_kernel"],
                "fallback_decision": "author_runtime_extension",
                "assumptions": ["No local Cahn-Hilliard numerical kernel selected."],
                "unsupported_reasons": ["No deterministic Cahn-Hilliard kernel is connected."],
            }
        assert request["agent_name"] == "taps-formulation-agent"
        return {
            "plan": {
                "problem_id": problem.id,
                "status": "ready",
                "equation_family": "cahn_hilliard",
                "unknown_fields": ["c", "mu"],
                "axes": [
                    {"name": "x", "kind": "space", "min_value": 0.0, "max_value": 1.0, "points": 32},
                    {"name": "y", "kind": "space", "min_value": 0.0, "max_value": 1.0, "points": 32},
                ],
                "weak_form": {
                    "family": "cahn_hilliard",
                    "strong_form": "M grad(mu) = 0 with algebraic chemical potential closure",
                    "trial_fields": ["c", "mu"],
                    "test_functions": ["v_c", "v_mu"],
                    "terms": [
                        {
                            "id": "operator:cahn_hilliard:mobility",
                            "role": "custom",
                            "expression": "int_Omega M flux_operator(v_c, mu) dOmega",
                            "fields": ["c", "mu"],
                            "coefficients": ["mobility"],
                        },
                        {
                            "id": "operator:cahn_hilliard:chemical_potential",
                            "role": "custom",
                            "expression": "int_Omega v_mu chemical_potential_closure(c, mu, epsilon) dOmega",
                            "fields": ["c", "mu"],
                            "coefficients": ["epsilon"],
                        },
                    ],
                    "boundary_terms": [],
                    "constraints": [],
                    "residual_expression": "Find c and mu such that the steady Cahn-Hilliard mixed weak form vanishes.",
                    "source": "hybrid",
                },
                "required_knowledge_queries": [],
                "missing_inputs": [],
                "assumptions": ["LLM formulated a smooth stable mixed fourth-order phase-field weak form; no local TAPS kernel is connected yet."],
                "risks": ["Requires mixed nonlinear phase-field assembly or a backend bridge before trusted local execution."],
                "recommended_next_action": "author_runtime_extension",
            }
        }

    output = run_typed_physicsos_workflow(
        RunTypedPhysicsOSWorkflowInput(
            user_request="simulate a smooth steady Cahn-Hilliard phase-field on a square",
            use_knowledge=False,
            core_agents_mode="llm",
            taps_max_wall_time_seconds=5.0,
        ),
        structured_client=fake_client,
    )
    assert output.workflow is not None
    assert output.workflow.taps is not None
    assert output.workflow.taps.compilation_plan is not None
    assert output.workflow.taps.numerical_plan is not None
    assert output.workflow.taps.compilation_plan.equation_family == "cahn_hilliard"
    assert output.workflow.taps.numerical_plan.solver_family == "cahn_hilliard"
    assert output.workflow.taps.contract_review is not None
    assert output.workflow.taps.contract_review.status == "accepted"
    assert output.workflow.solver_result is not None
    assert output.workflow.solver_result.status == "needs_review"
    assert [call["agent_name"] for call in calls] == [
        "build-physics-problem-agent",
        "geometry-mesh-planning-agent",
        "taps-formulation-agent",
        "numerical-solve-planning-agent",
        "postprocess-planning-agent",
    ]


def test_llm_numerical_plan_drives_tetra_elasticity_workflow(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PHYSICSOS_CORE_AGENTS_MODE", "llm")
    mesh_graph_path = tmp_path / "tetra_elasticity_mesh_graph.json"
    mesh_graph_path.write_text(
        json.dumps(
            {
                "type": "mesh_graph",
                "source_mesh": "hand-built-tetra-star",
                "node_count": 5,
                "points": [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.25, 0.25, 0.25],
                ],
                "edges": [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [0, 4], [1, 4], [2, 4], [3, 4]],
                "boundary_nodes": [0, 1, 2, 3],
                "cell_blocks": [
                    {"type": "tetra", "cells": [[0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4]]}
                ],
            }
        ),
        encoding="utf-8",
    )
    problem = PhysicsProblem(
        id="problem:llm-tetra-elasticity-3d",
        user_intent={"raw_request": "solve 3D linear elasticity on a tetrahedral mesh"},
        domain="solid",
        geometry=GeometrySpec(
            id="geometry:llm-tetra-elasticity",
            source=GeometrySource(kind="generated"),
            dimension=3,
            encodings=[GeometryEncoding(kind="mesh_graph", uri=str(mesh_graph_path), target_backend="taps")],
        ),
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
                id="material:llm-solid",
                name="llm solid",
                phase="solid",
                properties=[
                    MaterialProperty(name="young_modulus", value=8.0),
                    MaterialProperty(name="poisson_ratio", value=0.25),
                    MaterialProperty(name="body_force", value=[0.0, 0.0, -1.0]),
                ],
            )
        ],
        boundary_conditions=[BoundaryConditionSpec(id="bc:u", region_id="boundary", field="u", kind="dirichlet", value=[0.0, 0.0, 0.0])],
        targets=[{"name": "displacement", "field": "u", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    calls = []

    def fake_client(request):
        calls.append(request)
        if request["agent_name"] == "build-physics-problem-agent":
            return {"problem": problem.model_dump(mode="json"), "missing_inputs": [], "assumptions": ["llm extracted 3d elasticity"]}
        if request["agent_name"] == "geometry-mesh-planning-agent":
            return {
                "mesh_policy": {"strategy": "reuse", "target_element_size": 0.25, "element_order": 1},
                "requested_encodings": ["mesh_graph"],
                "target_backends": ["taps"],
                "require_boundary_confirmation": False,
                "boundary_confidence_threshold": 0.7,
                "assumptions": ["mesh graph already provided"],
                "warnings": [],
            }
        if request["agent_name"] == "taps-formulation-agent":
            return {"plan": formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan.model_dump(mode="json")}
        if request["agent_name"] == "numerical-solve-planning-agent":
            return {
                "problem_id": problem.id,
                "status": "ready",
                "solver_family": "mesh_fem_linear_elasticity",
                "backend_target": "taps",
                "field_bindings": {"primary": "u"},
                "discretization": {"dimension": 3, "node_counts": {"x": 64, "y": 64, "z": 64}, "element_order": 1, "quadrature_order": 2},
                "coefficient_bindings": [
                    {"name": "diffusion", "role": "diffusion", "value": 1.0, "source_name": "default"},
                    {"name": "young_modulus", "role": "operator", "value": 8.0, "source_name": "young_modulus"},
                    {"name": "poisson_ratio", "role": "operator", "value": 0.25, "source_name": "poisson_ratio"},
                    {"name": "body_force", "role": "source", "value": [0.0, 0.0, -1.0], "source_name": "body_force"},
                    {"name": "max_iterations", "role": "solver", "value": 20000, "source_name": "llm"},
                    {"name": "tolerance", "role": "solver", "value": 1e-8, "source_name": "llm"},
                ],
                "source_bindings": [{"name": "zero_source", "value": 0.0, "expression": "0"}],
                "boundary_condition_bindings": [
                    {"id": "bc:u", "region_id": "boundary", "field": "u", "kind": "dirichlet", "value": [0.0, 0.0, 0.0]}
                ],
                "expected_artifacts": ["taps_mesh_fem_elasticity_operator", "taps_mesh_fem_displacement_field", "taps_iteration_history"],
                "validation_checks": ["mesh_graph_present", "coefficient_bound", "linear_residual"],
                "assumptions": ["llm selected 3d tetra vector fem"],
            }
        if request["agent_name"] == "postprocess-planning-agent":
            return _postprocess_plan_payload()
        raise AssertionError(request["agent_name"])

    output = run_typed_physicsos_workflow(
        RunTypedPhysicsOSWorkflowInput(
            user_request="solve 3D linear elasticity on a tetrahedral mesh",
            use_knowledge=False,
            core_agents_mode="llm",
            taps_max_wall_time_seconds=20.0,
        ),
        structured_client=fake_client,
    )

    assert output.workflow is not None
    assert output.workflow.taps is not None
    assert output.workflow.taps.numerical_plan is not None
    assert output.workflow.taps.numerical_plan.solver_family == "mesh_fem_linear_elasticity"
    assert output.workflow.solver_result is not None
    assert output.workflow.solver_result.status == "success"
    assert output.workflow.solver_result.residuals["fem_tetrahedra"] == 4.0
    assert any(call["agent_name"] == "numerical-solve-planning-agent" for call in calls)


def test_llm_numerical_plan_drives_tetra10_poisson_workflow(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PHYSICSOS_CORE_AGENTS_MODE", "llm")
    mesh_graph_path = tmp_path / "tetra10_mesh_graph.json"
    mesh_graph_path.write_text(
        json.dumps(
            {
                "type": "mesh_graph",
                "source_mesh": "hand-built-tetra10",
                "node_count": 10,
                "points": [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.5, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.0, 0.5, 0.0],
                    [0.0, 0.0, 0.5],
                    [0.5, 0.0, 0.5],
                    [0.0, 0.5, 0.5],
                ],
                "edges": [[0, 1], [1, 2], [0, 2], [0, 3], [1, 3], [2, 3]],
                "boundary_nodes": [0, 1, 2, 3],
                "cell_blocks": [{"type": "tetra10", "cells": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]}],
            }
        ),
        encoding="utf-8",
    )
    problem = PhysicsProblem(
        id="problem:llm-tetra10-poisson-3d",
        user_intent={"raw_request": "solve 3D Poisson on a second-order tetrahedral mesh"},
        domain="custom",
        geometry=GeometrySpec(
            id="geometry:llm-tetra10-poisson",
            source=GeometrySource(kind="generated"),
            dimension=3,
            encodings=[GeometryEncoding(kind="mesh_graph", uri=str(mesh_graph_path), target_backend="taps")],
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
        boundary_conditions=[BoundaryConditionSpec(id="bc:u", region_id="boundary", field="u", kind="dirichlet", value=0.0)],
        targets=[{"name": "field", "field": "u", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    calls = []

    def fake_client(request):
        calls.append(request)
        if request["agent_name"] == "build-physics-problem-agent":
            return {"problem": problem.model_dump(mode="json"), "missing_inputs": [], "assumptions": ["llm extracted tetra10 poisson"]}
        if request["agent_name"] == "geometry-mesh-planning-agent":
            return {
                "mesh_policy": {"strategy": "reuse", "target_element_size": 0.25, "element_order": 2},
                "requested_encodings": ["mesh_graph"],
                "target_backends": ["taps"],
                "require_boundary_confirmation": False,
                "boundary_confidence_threshold": 0.7,
                "assumptions": ["mesh graph already provided"],
                "warnings": [],
            }
        if request["agent_name"] == "taps-formulation-agent":
            return {"plan": formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan.model_dump(mode="json")}
        if request["agent_name"] == "numerical-solve-planning-agent":
            return {
                "problem_id": problem.id,
                "status": "ready",
                "solver_family": "mesh_fem_poisson",
                "backend_target": "taps",
                "field_bindings": {"primary": "u"},
                "discretization": {"dimension": 3, "node_counts": {"x": 64, "y": 64, "z": 64}, "element_order": 2, "quadrature_order": 2},
                "coefficient_bindings": [{"name": "diffusion", "role": "diffusion", "value": 1.0, "source_name": "llm"}],
                "source_bindings": [{"name": "zero_source", "value": 0.0, "expression": "0"}],
                "boundary_condition_bindings": [
                    {"id": "bc:u", "region_id": "boundary", "field": "u", "kind": "dirichlet", "value": 0.0}
                ],
                "expected_artifacts": ["taps_mesh_fem_operator", "taps_mesh_fem_solution_field", "taps_iteration_history"],
                "validation_checks": ["mesh_graph_present", "coefficient_bound", "linear_residual"],
                "assumptions": ["llm selected second-order tetra scalar FEM"],
            }
        if request["agent_name"] == "postprocess-planning-agent":
            return _postprocess_plan_payload()
        raise AssertionError(request["agent_name"])

    output = run_typed_physicsos_workflow(
        RunTypedPhysicsOSWorkflowInput(
            user_request="solve 3D Poisson on a second-order tetrahedral mesh",
            use_knowledge=False,
            core_agents_mode="llm",
            taps_max_wall_time_seconds=20.0,
        ),
        structured_client=fake_client,
    )

    assert output.workflow is not None
    assert output.workflow.taps is not None
    assert output.workflow.taps.numerical_plan is not None
    assert output.workflow.taps.numerical_plan.solver_family == "mesh_fem_poisson"
    assert output.workflow.taps.numerical_plan.discretization.element_order == 2
    assert output.workflow.solver_result is not None
    assert output.workflow.solver_result.status == "success"
    assert output.workflow.solver_result.residuals["fem_basis_order"] == 2.0
    assert any(call["agent_name"] == "numerical-solve-planning-agent" for call in calls)


def test_llm_invalid_numerical_plan_stops_before_taps_execution(monkeypatch) -> None:
    monkeypatch.setenv("PHYSICSOS_CORE_AGENTS_MODE", "llm")
    problem = build_default_thermal_problem()
    calls = []

    def fake_client(request):
        calls.append(request)
        if request["agent_name"] == "build-physics-problem-agent":
            return {"problem": problem.model_dump(mode="json"), "missing_inputs": [], "assumptions": ["llm extracted heat"]}
        if request["agent_name"] == "geometry-mesh-planning-agent":
            return {
                "mesh_policy": {"strategy": "structured", "target_element_size": 0.1, "element_order": 1},
                "requested_encodings": [],
                "target_backends": ["taps"],
                "require_boundary_confirmation": False,
                "boundary_confidence_threshold": 0.7,
                "assumptions": [],
                "warnings": [],
            }
        if request["agent_name"] == "taps-formulation-agent":
            return {"plan": formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan.model_dump(mode="json")}
        if request["agent_name"] == "numerical-solve-planning-agent":
            return {
                "problem_id": problem.id,
                "status": "ready",
                "solver_family": "scalar_elliptic_1d",
                "backend_target": "taps",
                "field_bindings": {"primary": "T"},
                "discretization": {"dimension": 1, "node_counts": {"x": 64}, "element_order": 1, "quadrature_order": 2},
                "coefficient_bindings": [{"name": "diffusion", "role": "diffusion", "value": 1.0, "source_name": "llm"}],
                "source_bindings": [{"name": "zero_source", "value": 0.0, "expression": "0"}],
                "boundary_condition_bindings": [
                    {"id": "bc:left", "region_id": "x=0", "field": "T", "kind": "dirichlet", "value": 999.0}
                ],
                "expected_artifacts": [],
                "validation_checks": [],
                "assumptions": ["invalid drift"],
            }
        raise AssertionError(request["agent_name"])

    output = run_typed_physicsos_workflow(
        RunTypedPhysicsOSWorkflowInput(
            user_request="simulate one-dimensional steady heat conduction",
            use_knowledge=False,
            core_agents_mode="llm",
        ),
        structured_client=fake_client,
    )

    assert output.workflow is not None
    assert output.workflow.taps is not None
    assert output.workflow.taps.handoff.status == "failed"
    assert output.workflow.taps.result is None
    assert output.workflow.solver_result is None
    assert any("Numerical solve plan validation failed" in step.summary for step in output.workflow.trace)


def test_numerical_solve_planner_structured_contract_retry(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("PHYSICSOS_HOME", str(tmp_path / "home"))
    problem = build_default_thermal_problem()
    compilation_plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=compilation_plan)).taps_problem
    problem_contract = build_physics_problem_contract(problem)
    valid_plan = plan_numerical_solve(
        NumericalSolvePlanInput(
            problem=problem,
            taps_problem=taps_problem,
            compilation_plan=compilation_plan,
            problem_contract=problem_contract,
        )
    )
    calls = []

    def fake_client(request):
        calls.append(request)
        if len(calls) == 1:
            invalid_plan = valid_plan.model_dump(mode="json")
            invalid_plan["boundary_condition_bindings"] = [
                {"id": "bc:left", "region_id": "x=0", "field": "T", "kind": "dirichlet", "value": 999.0}
            ]
            return invalid_plan
        return valid_plan.model_dump(mode="json")

    events = []
    with structured_agent_event_context(run_id="run:numerical-plan", case_id=problem.id, events=events):
        output = plan_numerical_solve_structured(
            NumericalSolvePlanInput(
                problem=problem,
                taps_problem=taps_problem,
                compilation_plan=compilation_plan,
                problem_contract=problem_contract,
            ),
            client=fake_client,
            config=CoreAgentLLMConfig(mode="llm", max_structured_attempts=2),
        )

    assert len(calls) == 2
    assert output.solver_family == valid_plan.solver_family
    assert [event.payload["attempt"] for event in events] == [1, 2]
    assert [event.payload["max_attempts"] for event in events] == [2, 2]
    assert events[0].event == "validation.retry"
    feedback = json.loads(calls[1]["validation_feedback"][0])
    assert "invalid_plan" in feedback
    assert "problem_contract" in feedback
    assert feedback["errors"]
    assert "Repair only the invalid NumericalSolvePlanOutput" in feedback["instruction"]


def test_llm_taps_formulation_falls_back_after_validation_failures() -> None:
    problem = build_default_thermal_problem()
    calls = []

    def broken_client(request):
        calls.append(request)
        return {"plan": {"problem_id": problem.id, "status": "ready"}}

    output = formulate_taps_equation_structured(
        FormulateTAPSEquationInput(problem=problem, problem_contract=build_physics_problem_contract(problem)),
        client=broken_client,
        config=CoreAgentLLMConfig(mode="llm", max_structured_attempts=2),
    )
    assert len(calls) == 2
    assert output.plan.problem_id == problem.id
    assert any("deterministic formulation fallback" in assumption for assumption in output.plan.assumptions)


def test_llm_taps_contract_review_rejects_formulation_drift(monkeypatch) -> None:
    monkeypatch.setenv("PHYSICSOS_CORE_AGENTS_MODE", "llm")
    problem = build_default_thermal_problem()

    def fake_client(request):
        if request["agent_name"] == "build-physics-problem-agent":
            return {"problem": problem.model_dump(mode="json"), "missing_inputs": [], "assumptions": []}
        if request["agent_name"] == "geometry-mesh-planning-agent":
            return {
                "mesh_policy": {"strategy": "structured", "element_order": 1},
                "requested_encodings": [],
                "target_backends": ["taps"],
                "require_boundary_confirmation": False,
                "boundary_confidence_threshold": 0.7,
                "assumptions": [],
                "warnings": [],
            }
        return {
            "plan": {
                "problem_id": problem.id,
                "status": "ready",
                "equation_family": "poisson",
                "unknown_fields": ["u"],
                "axes": [{"name": "x", "kind": "space", "min_value": 0.0, "max_value": 1.0, "points": 16}],
                "weak_form": {
                    "family": "poisson",
                    "strong_form": "-div(grad(u)) = f",
                    "trial_fields": ["u"],
                    "test_functions": ["v_u"],
                    "terms": [
                        {
                            "id": "operator:poisson:diffusion",
                            "role": "diffusion",
                            "expression": "int_Omega grad(v_u) dot grad(u) dOmega",
                            "fields": ["u"],
                            "coefficients": [],
                        }
                    ],
                    "boundary_terms": [],
                    "constraints": [],
                    "source": "hybrid",
                },
                "required_knowledge_queries": [],
                "missing_inputs": [],
                "assumptions": ["bad drift"],
                "risks": [],
                "recommended_next_action": "compile_taps_problem",
            }
        }

    output = run_typed_physicsos_workflow(
        RunTypedPhysicsOSWorkflowInput(
            user_request="simulate one-dimensional steady heat conduction",
            use_knowledge=False,
            core_agents_mode="llm",
        ),
        structured_client=fake_client,
    )
    assert output.workflow is not None
    assert output.workflow.taps is not None
    assert output.workflow.taps.handoff.status == "failed"
    assert output.workflow.taps.contract_review is not None
    assert output.workflow.taps.contract_review.status == "needs_retry"
    assert output.workflow.solver_result is None


def test_workflow_sets_solver_envelope_for_llm_taps_result_even_when_support_score_is_low(monkeypatch) -> None:
    monkeypatch.setenv("PHYSICSOS_CORE_AGENTS_MODE", "llm")
    problem = build_default_thermal_problem()

    def fake_client(request):
        if request["agent_name"] == "build-physics-problem-agent":
            return {"problem": problem.model_dump(mode="json"), "missing_inputs": [], "assumptions": []}
        if request["agent_name"] == "geometry-mesh-planning-agent":
            return {
                "mesh_policy": {"strategy": "structured", "element_order": 1},
                "requested_encodings": [],
                "target_backends": ["taps"],
                "require_boundary_confirmation": False,
                "boundary_confidence_threshold": 0.7,
                "assumptions": [],
                "warnings": [],
            }
        if request["agent_name"] == "postprocess-planning-agent":
            return _postprocess_plan_payload()
        return formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).model_dump(mode="json")

    output = run_typed_physicsos_workflow(
        RunTypedPhysicsOSWorkflowInput(
            user_request="simulate one-dimensional steady heat conduction",
            use_knowledge=False,
            core_agents_mode="llm",
        ),
        structured_client=fake_client,
    )
    assert output.workflow is not None
    assert output.workflow.taps is not None
    assert output.workflow.solver is not None
    assert output.workflow.solver.result == output.workflow.solver_result
    assert output.workflow.case_memory is not None


def test_workflow_events_include_structured_taps_attempt(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PHYSICSOS_CORE_AGENTS_MODE", "llm")
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("PHYSICSOS_HOME", str(tmp_path / "home"))
    problem = build_default_thermal_problem()

    def fake_client(request):
        if request["agent_name"] == "build-physics-problem-agent":
            return {"problem": problem.model_dump(mode="json"), "missing_inputs": [], "assumptions": []}
        if request["agent_name"] == "geometry-mesh-planning-agent":
            return {
                "mesh_policy": {"strategy": "structured", "element_order": 1},
                "requested_encodings": [],
                "target_backends": ["taps"],
                "require_boundary_confirmation": False,
                "boundary_confidence_threshold": 0.7,
                "assumptions": [],
                "warnings": [],
            }
        if request["agent_name"] == "postprocess-planning-agent":
            return _postprocess_plan_payload()
        return formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).model_dump(mode="json")

    output = run_typed_physicsos_workflow(
        RunTypedPhysicsOSWorkflowInput(
            user_request="simulate one-dimensional steady heat conduction",
            use_knowledge=False,
            core_agents_mode="llm",
        ),
        structured_client=fake_client,
    )
    assert output.workflow is not None
    structured_events = [event for event in output.workflow.events if event.stage == "taps-formulation-agent"]
    assert structured_events
    assert structured_events[-1].event == "agent.output"
    assert structured_events[-1].artifacts[0].kind == "structured_agent_attempt"
    assert structured_events[-1].display["raw_response_in_artifact"] is True
    rendered = PhysicsOSEventRenderer().render_many(structured_events)
    assert any("[taps-formulation-agent]" in line for line in rendered)


def test_natural_language_entry_logs_build_and_taps_structured_attempts_to_same_run(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PHYSICSOS_CORE_AGENTS_MODE", "llm")
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("PHYSICSOS_HOME", str(tmp_path / "home"))
    problem = build_default_thermal_problem()

    def fake_client(request):
        if request["agent_name"] == "build-physics-problem-agent":
            return {"problem": problem.model_dump(mode="json"), "missing_inputs": [], "assumptions": []}
        if request["agent_name"] == "postprocess-planning-agent":
            return _postprocess_plan_payload()
        return formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).model_dump(mode="json")

    output = run_typed_physicsos_workflow(
        RunTypedPhysicsOSWorkflowInput(
            user_request="simulate one-dimensional steady heat conduction",
            use_knowledge=False,
            core_agents_mode="llm",
        ),
        structured_client=fake_client,
    )
    assert output.workflow is not None
    event_log_name = "events-" + "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in output.workflow.run_id) + ".jsonl"
    event_log = read_physicsos_events(runtime_paths().sessions / event_log_name)
    stages = [event.stage for event in event_log if event.artifacts and event.artifacts[0].kind == "structured_agent_attempt"]
    assert "build-physics-problem-agent" in stages
    assert "taps-formulation-agent" in stages


def test_geometry_mesh_planner_structured_retry_and_fallback(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("PHYSICSOS_HOME", str(tmp_path / "home"))
    problem = build_default_thermal_problem().model_copy(
        update={"geometry": GeometrySpec(id="geometry:plate", source=GeometrySource(kind="generated"), dimension=2)}
    )
    calls = []

    def fake_client(request):
        calls.append(request)
        if len(calls) == 1:
            return {"mesh_policy": {"strategy": "invalid"}}
        return {
            "mesh_policy": {"strategy": "unstructured", "target_element_size": 0.05, "element_order": 2},
            "requested_encodings": ["mesh_graph", "sdf"],
            "target_backends": ["taps"],
            "require_boundary_confirmation": False,
            "boundary_confidence_threshold": 0.7,
            "assumptions": ["structured geometry plan"],
            "warnings": [],
        }

    events = []
    with structured_agent_event_context(run_id="workflow:geometry-test", case_id=problem.id, events=events):
        plan = plan_geometry_mesh_structured(
            GeometryMeshPlanInput(problem=problem, requested_encodings=["mesh_graph"], target_backends=["taps"]),
            client=fake_client,
            config=CoreAgentLLMConfig(mode="llm", max_structured_attempts=2),
        )

    assert plan.mesh_policy.target_element_size == 0.05
    assert plan.mesh_policy.element_order == 2
    assert plan.requested_encodings == ["mesh_graph", "sdf"]
    assert [event.status for event in events] == ["retrying", "accepted"]
    assert events[0].stage == "geometry-mesh-planning-agent"


def test_postprocess_planner_structured_retry_and_fallback(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("PHYSICSOS_HOME", str(tmp_path / "home"))
    problem = build_default_thermal_problem()
    result = SolverResult(
        id="result:postprocess",
        problem_id=problem.id,
        backend="taps:test",
        status="success",
        scalar_outputs={"max_temperature": 1.0},
        residuals={"l2": 0.0},
        provenance=Provenance(created_by="test"),
    )
    verification = VerificationReport(
        problem_id=problem.id,
        result_id=result.id,
        status="accepted",
        residuals={"l2": 0.0},
        recommended_next_action="accept",
        explanation="Verification accepted.",
    )
    calls = []

    def fake_client(request):
        calls.append(request)
        if len(calls) == 1:
            return {"visualization_plan": [{"kind": "not_a_kind"}]}
        return _postprocess_plan_payload()

    events = []
    with structured_agent_event_context(run_id="run:postprocess", case_id=problem.id, events=events):
        plan = plan_postprocess_structured(
            PostprocessPlanInput(problem=problem, result=result, verification=verification),
            client=fake_client,
            config=CoreAgentLLMConfig(mode="llm", max_structured_attempts=2),
        )

    assert len(calls) == 2
    assert plan.recommendations == ["Review verification result and generated report."]
    assert [event.stage for event in events] == ["postprocess-planning-agent", "postprocess-planning-agent"]
    assert events[0].event == "validation.retry"
    assert events[-1].event == "agent.output"


def test_backend_preparation_planner_structured_semantic_retry(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("PHYSICSOS_HOME", str(tmp_path / "home"))
    problem = _minimal_fluid_problem()
    problem.geometry = GeometrySpec(id="geometry:fluid-2d", source=GeometrySource(kind="generated"), dimension=2)
    problem.materials = [
        MaterialSpec(
            id="material:fluid",
            name="fluid",
            phase="liquid",
            properties=[
                MaterialProperty(name="dynamic_viscosity", value=1.0),
                MaterialProperty(name="density", value=1.0),
                MaterialProperty(name="pressure_drop", value=1.0),
            ],
        )
    ]
    problem.boundary_conditions = [
        BoundaryConditionSpec(id="bc:wall", region_id="wall", field="U", kind="wall", value=[0.0, 0.0]),
        BoundaryConditionSpec(id="bc:inlet", region_id="inlet", field="U", kind="inlet", value=[1.0, 0.0]),
        BoundaryConditionSpec(id="bc:outlet", region_id="outlet", field="p", kind="outlet", value=0.0),
    ]
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    calls = []

    def payload(execute: bool) -> dict:
        return {
            "problem_id": problem.id,
            "status": "needs_inputs",
            "target_backend": "openfoam",
            "backend_family": "openfoam",
            "field_space_plan": [{"family": "incompressible_navier_stokes_mesh", "fields": ["U", "p"], "space": "mixed_velocity_pressure"}],
            "mesh_export": {"required": True, "provided": False, "expected_kind": "backend_mesh_export_manifest"},
            "coefficient_map": [{"name": coefficient.name, "role": coefficient.role, "region_ids": coefficient.region_ids} for coefficient in taps_problem.coefficients],
            "boundary_tag_map": [
                {"id": boundary.id, "field": boundary.field, "kind": boundary.kind, "region_id": boundary.region_id, "backend_tag": boundary.region_id}
                for boundary in problem.boundary_conditions
            ],
            "stabilization_policy": {"required_review": True, "selected": "review_required_before_execution"},
            "solver_controls": {"external_execution_enabled": execute, "review_required": True},
            "dependency_checks": [{"name": "openfoam_runner_available", "command": "foamVersion", "required": False}],
            "approval_gate": {"execute_external_solver": execute, "requires_user_approval": True, "requires_dependency_checks": True, "requires_mesh_export_manifest": True},
            "expected_artifacts": ["taps_backend_bridge_manifest", "taps_backend_case_bundle"],
            "validation_checks": ["approval_gate_blocks_execution"],
            "assumptions": [],
            "warnings": [],
            "unsupported_reasons": [],
        }

    def fake_client(request):
        calls.append(request)
        return payload(execute=len(calls) == 1)

    events = []
    with structured_agent_event_context(run_id="run:backend-prep", case_id=problem.id, events=events):
        output = plan_backend_preparation_structured(
            PlanBackendPreparationInput(problem=problem, taps_problem=taps_problem, backend="openfoam"),
            client=fake_client,
            config=CoreAgentLLMConfig(mode="llm", max_structured_attempts=2),
        )

    assert len(calls) == 2
    assert output.backend_family == "openfoam"
    assert output.approval_gate["execute_external_solver"] is False
    assert output.solver_controls["external_execution_enabled"] is False
    assert [event.stage for event in events] == ["backend-preparation-planning-agent", "backend-preparation-planning-agent"]
    assert [event.payload["attempt"] for event in events] == [1, 2]
    assert [event.payload["max_attempts"] for event in events] == [2, 2]
    assert [event.event for event in events] == ["validation.retry", "agent.output"]
    assert calls[1]["validation_feedback"]
    feedback = json.loads(calls[1]["validation_feedback"][0])
    assert "invalid_plan" in feedback
    assert feedback["errors"]
    assert "Do not enable external execution" in feedback["instruction"]


def test_geometry_mesh_planner_requests_boundary_confirmation_for_imported_geometry() -> None:
    problem = build_default_thermal_problem().model_copy(
        update={"geometry": GeometrySpec(id="geometry:imported", source=GeometrySource(kind="mesh_file", uri="part.msh"), dimension=2)}
    )
    plan = plan_geometry_mesh(GeometryMeshPlanInput(problem=problem, target_backends=["fenicsx"]))
    assert plan.require_boundary_confirmation is True
    assert "mesh_graph" in plan.requested_encodings


def test_geometry_mesh_handoff_includes_boundary_labeler_viewer(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("PHYSICSOS_HOME", str(tmp_path / "home"))
    geometry = GeometrySpec(id="geometry:imported-box", source=GeometrySource(kind="mesh_file", uri="box.msh"), dimension=3)
    problem = build_default_thermal_problem().model_copy(update={"geometry": geometry})
    output = _run_geometry_mesh_agent(
        GeometryMeshAgentInput(problem=problem, requested_encodings=[], target_backends=["openfoam"])
    )
    artifact_kinds = {artifact.kind for artifact in output.handoff.artifacts}
    viewer = next(artifact for artifact in output.handoff.artifacts if artifact.kind == "geometry_labeler_viewer")
    labeling = next(artifact for artifact in output.handoff.artifacts if artifact.kind == "boundary_labeling_artifact")
    assert output.handoff.status == "needs_user_input"
    assert output.handoff.recommended_next_action == "confirm_boundary_labels"
    assert {"boundary_labeling_artifact", "geometry_labeler_viewer"} <= artifact_kinds
    assert Path(viewer.uri).exists()
    assert Path(labeling.uri).exists()
    assert "geometry_labeler_viewer artifact" in output.handoff.summary


def test_workflow_events_include_structured_geometry_attempt(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PHYSICSOS_CORE_AGENTS_MODE", "llm")
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("PHYSICSOS_HOME", str(tmp_path / "home"))
    problem = build_default_thermal_problem()

    def fake_client(request):
        agent_name = request["agent_name"]
        if agent_name == "build-physics-problem-agent":
            return {"problem": problem.model_dump(mode="json"), "missing_inputs": [], "assumptions": []}
        if agent_name == "geometry-mesh-planning-agent":
            return {
                "mesh_policy": {"strategy": "structured", "element_order": 1},
                "requested_encodings": [],
                "target_backends": ["taps"],
                "require_boundary_confirmation": False,
                "boundary_confidence_threshold": 0.7,
                "assumptions": ["no mesh required for 1D"],
                "warnings": [],
            }
        if agent_name == "postprocess-planning-agent":
            return _postprocess_plan_payload()
        return formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).model_dump(mode="json")

    output = run_typed_physicsos_workflow(
        RunTypedPhysicsOSWorkflowInput(
            user_request="simulate one-dimensional steady heat conduction",
            use_knowledge=False,
            core_agents_mode="llm",
        ),
        structured_client=fake_client,
    )
    assert output.workflow is not None
    stages = [event.stage for event in output.workflow.events if event.artifacts and event.artifacts[0].kind == "structured_agent_attempt"]
    assert "geometry-mesh-planning-agent" in stages


def test_workflow_events_include_structured_postprocess_attempt(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PHYSICSOS_CORE_AGENTS_MODE", "llm")
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("PHYSICSOS_HOME", str(tmp_path / "home"))
    problem = build_default_thermal_problem()

    def fake_client(request):
        agent_name = request["agent_name"]
        if agent_name == "build-physics-problem-agent":
            return {"problem": problem.model_dump(mode="json"), "missing_inputs": [], "assumptions": []}
        if agent_name == "postprocess-planning-agent":
            return _postprocess_plan_payload()
        return formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).model_dump(mode="json")

    output = run_typed_physicsos_workflow(
        RunTypedPhysicsOSWorkflowInput(
            user_request="simulate one-dimensional steady heat conduction",
            use_knowledge=False,
            core_agents_mode="llm",
        ),
        structured_client=fake_client,
    )
    assert output.workflow is not None
    structured_events = [event for event in output.workflow.events if event.stage == "postprocess-planning-agent"]
    assert structured_events
    assert structured_events[-1].event == "agent.output"
    assert structured_events[-1].artifacts[0].kind == "structured_agent_attempt"
    assert output.workflow.postprocess is not None
    assert "Review verification result and generated report." in output.workflow.postprocess.recommendations


def test_universal_workflow_orders_knowledge_before_geometry_and_taps() -> None:
    result = run_physicsos_workflow(use_knowledge=True, arxiv_max_results=0, taps_rank=8)
    step_names = [step.name for step in result.trace]
    assert step_names.index("knowledge-agent") < step_names.index("geometry-mesh-agent")
    assert step_names.index("geometry-mesh-agent") < step_names.index("validate_physics_problem")
    assert step_names.index("validate_physics_problem") < step_names.index("taps-agent.support")
    assert result.knowledge is not None
    assert result.geometry is not None
    assert result.knowledge.handoff.recommended_next_agent == "geometry-mesh-agent"
    assert result.geometry.handoff.recommended_next_agent == "taps-agent"


def test_universal_workflow_returns_typed_retry_context_on_validation_failure() -> None:
    problem = _minimal_fluid_problem()
    result = run_physicsos_workflow(problem=problem, use_knowledge=False, max_validation_attempts=2)
    assert result.solver_result is None
    assert result.taps is None
    assert result.validation_attempts
    assert len(result.validation_attempts) == 2
    assert result.state.validation_attempts == result.validation_attempts
    assert result.trace[-1].status == "retry_exhausted"
    assert "boundary condition" in " ".join(result.validation_attempts[-1].errors).lower()
    assert result.validation_attempts[-1].input_context["geometry_available"] is True


def test_core_workflow_retries_typed_subagent_failures(monkeypatch) -> None:
    def broken_taps(_input):
        raise ValueError("typed taps output failed validation")

    monkeypatch.setattr("physicsos.workflows.universal._run_taps_agent", broken_taps)
    result = run_physicsos_workflow(use_knowledge=False, max_validation_attempts=2)
    assert result.solver_result is None
    assert result.taps is None
    assert result.trace[-1].name == "taps-agent"
    assert result.trace[-1].status == "retry_exhausted"
    assert [attempt.agent_name for attempt in result.validation_attempts] == ["taps-agent", "taps-agent"]
    assert "typed taps output failed validation" in result.validation_attempts[-1].errors[0]
    assert "agent_input" in result.validation_attempts[-1].input_context


def test_core_workflow_retries_taps_wall_time_budget(monkeypatch) -> None:
    def timed_out_taps(input):
        raise TimeoutError(f"TAPS backend exceeded wall-time budget before solve: {input.max_wall_time_seconds}")

    monkeypatch.setattr("physicsos.workflows.universal._run_taps_agent", timed_out_taps)
    result = run_physicsos_workflow(use_knowledge=False, max_validation_attempts=2, taps_max_wall_time_seconds=0.01)
    assert result.taps is None
    assert result.trace[-1].status == "retry_exhausted"
    assert [attempt.agent_name for attempt in result.validation_attempts] == ["taps-agent", "taps-agent"]
    assert "wall-time budget" in result.validation_attempts[-1].errors[0]
    assert result.validation_attempts[-1].input_context["agent_input"]["max_wall_time_seconds"] == 0.01


def test_run_taps_backend_wall_time_budget_uses_subprocess(monkeypatch) -> None:
    output = run_typed_physicsos_workflow(
        RunTypedPhysicsOSWorkflowInput(
            user_request="simulate one-dimensional steady heat conduction",
            use_knowledge=False,
            max_validation_attempts=1,
            taps_max_wall_time_seconds=30.0,
        )
    )
    assert output.workflow is not None
    taps_input = RunTAPSBackendInput(problem=output.workflow.problem, taps_problem=output.workflow.taps_problem)
    calls = []

    expected_output = object()

    def fake_subprocess(input):
        calls.append(input)
        return expected_output

    monkeypatch.setattr("physicsos.tools.taps_tools._run_taps_backend_subprocess", fake_subprocess)
    result = run_taps_backend(taps_input)
    assert result is expected_output
    assert calls[0].budget.max_wall_time_seconds == 120.0


def test_natural_language_entry_runs_typed_workflow_for_1d_heat_conduction() -> None:
    output = run_typed_physicsos_workflow(
        RunTypedPhysicsOSWorkflowInput(
            user_request="帮我模拟一维稳定热传导",
            use_knowledge=False,
            max_validation_attempts=2,
        )
    )
    assert output.build.problem is not None
    assert output.initial_validation is not None
    assert output.workflow is not None
    assert output.workflow.problem.domain == "thermal"
    assert output.workflow.problem.geometry.dimension == 1
    assert output.workflow.validation_attempts == []
    assert output.workflow.solver_result is not None


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


def test_geometry_mesh_backend_uses_subprocess_from_worker_threads(monkeypatch) -> None:
    class WorkerThread:
        pass

    geometry = GeometrySpec(id="geometry:worker", source=GeometrySource(kind="generated"), dimension=2)
    payloads = []

    def fake_subprocess(payload, timeout_seconds=120.0):
        payloads.append(payload)
        return {
            "ok": True,
            "mesh": {
                "id": "mesh:geometry:worker",
                "kind": "unstructured",
                "dimension": 2,
                "topology": {"cell_types": ["triangle"], "node_count": 3, "cell_count": 1},
                "elements": {"total": 1, "by_type": {"triangle": 1}},
                "regions": [],
                "boundaries": [],
                "quality": {"passes": True, "issues": []},
                "files": [],
                "solver_compatibility": ["taps"],
            },
            "artifacts": [],
        }

    main = object()
    monkeypatch.setattr(geometry_mesh_backend.threading, "current_thread", lambda: WorkerThread())
    monkeypatch.setattr(geometry_mesh_backend.threading, "main_thread", lambda: main)
    monkeypatch.setattr(geometry_mesh_backend, "_run_backend_subprocess", fake_subprocess)

    mesh, artifacts = geometry_mesh_backend.generate_mesh_backend(geometry, ["taps"], 0.2)
    assert artifacts == []
    assert mesh.quality.passes is True
    assert payloads[0]["action"] == "generate_mesh"


def test_taps_agent_formulates_non_template_custom_operator() -> None:
    problem = _minimal_fluid_problem()
    problem.operators[0].differential_terms.append(
        {"expression": "int_Omega v_i * (rho * U_j * grad_j U_i + grad_i p) dOmega", "order": 1, "fields": ["U", "p"]}
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    assert plan.weak_form is not None
    assert plan.weak_form.terms
    assert {term.role for term in plan.weak_form.terms} >= {"diffusion", "constraint", "advection"}
    assert plan.recommended_next_action in {"compile_taps_problem", "ask_knowledge_agent"}


def test_taps_executes_low_re_navier_stokes_as_stokes_simplification() -> None:
    problem = _minimal_fluid_problem()
    problem.geometry = GeometrySpec(id="geometry:fluid-2d", source=GeometrySource(kind="generated"), dimension=2)
    problem.operators[0].nondimensional_numbers.append(NondimensionalNumber(name="Re", value=0.1))
    problem.operators[0].conserved_quantities.extend(["mass", "momentum"])
    problem.materials = [
        MaterialSpec(
            id="material:fluid",
            name="Low-Re Newtonian fluid",
            phase="liquid",
            properties=[
                MaterialProperty(name="dynamic_viscosity", value=1.0, units="Pa*s"),
                MaterialProperty(name="density", value=1.0, units="kg/m^3"),
                MaterialProperty(name="pressure_drop", value=1.0, units="Pa"),
            ],
        )
    ]
    problem.boundary_conditions = [
        BoundaryConditionSpec(id="bc:wall", region_id="walls", field="U", kind="wall", value=[0.0, 0.0]),
        BoundaryConditionSpec(id="bc:inlet", region_id="x_min", field="p", kind="inlet", value=1.0),
        BoundaryConditionSpec(id="bc:outlet", region_id="x_max", field="p", kind="outlet", value=0.0),
    ]

    support = estimate_taps_support(EstimateTAPSSupportInput(problem=problem)).support
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result

    assert support.supported is True
    assert plan.weak_form is not None
    assert plan.weak_form.family == "navier_stokes"
    assert {term.role for term in plan.weak_form.terms} >= {"diffusion", "constraint"}
    assert all(term.role != "advection" for term in plan.weak_form.terms)
    assert result.backend.startswith("taps:stokes_channel_2d")
    assert result.status == "success"
    assert result.scalar_outputs["full_navier_stokes_supported"] == 0
    assert result.scalar_outputs["simplification"] == "steady_low_re_stokes_no_convection"
    assert result.scalar_outputs["numerical_plan_solver_family"] == "incompressible_stokes_channel_2d"
    assert result.residuals["normalized_stokes_residual"] == 0.0
    assert any(artifact.kind == "taps_stokes_solution_field" for artifact in result.artifacts)
    operator_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_stokes_operator")
    operator_payload = json.loads(open(operator_artifact.uri, encoding="utf-8").read())
    assert operator_payload["numerical_plan_solver_family"] == "incompressible_stokes_channel_2d"
    assert operator_payload["coefficient_values_applied"]["dynamic_viscosity"] == pytest.approx(1.0)


def test_full_navier_stokes_compiles_but_requires_nonlinear_taps_extension() -> None:
    problem = _minimal_fluid_problem()
    problem.geometry = GeometrySpec(id="geometry:fluid-2d", source=GeometrySource(kind="generated"), dimension=2)
    problem.operators[0].nondimensional_numbers.append(NondimensionalNumber(name="Re", value=100.0))
    problem.materials = [
        MaterialSpec(
            id="material:fluid",
            name="Newtonian fluid",
            phase="liquid",
            properties=[MaterialProperty(name="dynamic_viscosity", value=1.0), MaterialProperty(name="density", value=1.0)],
        )
    ]
    problem.boundary_conditions = [
        BoundaryConditionSpec(id="bc:wall", region_id="walls", field="U", kind="wall", value=[0.0, 0.0])
    ]

    support = estimate_taps_support(EstimateTAPSSupportInput(problem=problem)).support
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    validation = validate_taps_ir(ValidateTAPSIRInput(problem=problem, taps_problem=taps_problem))
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result

    assert support.supported is False
    assert plan.weak_form is not None
    assert any(term.role == "advection" for term in plan.weak_form.terms)
    assert validation.fallback_recommended is True
    assert "no executable TAPS IR block mapping" in " ".join(validation.warnings)
    assert result.status == "needs_review"
    assert result.scalar_outputs["message"].startswith("TAPSProblem compiled, but no executable TAPS kernel")


def test_taps_executes_oseen_linearized_navier_stokes_with_frozen_velocity() -> None:
    problem = _minimal_fluid_problem()
    problem.geometry = GeometrySpec(id="geometry:fluid-2d", source=GeometrySource(kind="generated"), dimension=2)
    problem.operators[0].nondimensional_numbers.append(NondimensionalNumber(name="Re", value=25.0))
    problem.materials = [
        MaterialSpec(
            id="material:fluid",
            name="Linearized Newtonian fluid",
            phase="liquid",
            properties=[
                MaterialProperty(name="dynamic_viscosity", value=1.0, units="Pa*s"),
                MaterialProperty(name="density", value=1.0, units="kg/m^3"),
                MaterialProperty(name="pressure_drop", value=1.0, units="Pa"),
                MaterialProperty(name="frozen_velocity", value=[0.25, 0.0], units="m/s"),
            ],
        )
    ]
    problem.boundary_conditions = [
        BoundaryConditionSpec(id="bc:wall", region_id="walls", field="U", kind="wall", value=[0.0, 0.0]),
        BoundaryConditionSpec(id="bc:inlet", region_id="x_min", field="p", kind="inlet", value=1.0),
        BoundaryConditionSpec(id="bc:outlet", region_id="x_max", field="p", kind="outlet", value=0.0),
    ]

    support = estimate_taps_support(EstimateTAPSSupportInput(problem=problem)).support
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    validation = validate_taps_ir(ValidateTAPSIRInput(problem=problem, taps_problem=taps_problem))
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result

    assert support.supported is True
    assert plan.weak_form is not None
    assert any(term.role == "advection" and "U_bar" in term.expression for term in plan.weak_form.terms)
    assert validation.fallback_recommended is False
    assert result.status == "success"
    assert result.backend.startswith("taps:oseen_channel_2d")
    assert result.scalar_outputs["simplification"] == "linearized_oseen_frozen_convection"
    assert result.scalar_outputs["full_navier_stokes_supported"] == 0
    assert result.scalar_outputs["numerical_plan_solver_family"] == "incompressible_oseen_channel_2d"
    assert result.residuals["normalized_oseen_residual"] == 0.0
    assert any(artifact.kind == "taps_oseen_solution_field" for artifact in result.artifacts)
    operator_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_oseen_operator")
    operator_payload = json.loads(open(operator_artifact.uri, encoding="utf-8").read())
    assert operator_payload["numerical_plan_solver_family"] == "incompressible_oseen_channel_2d"
    assert operator_payload["coefficient_values_applied"]["frozen_velocity"] == pytest.approx([0.25, 0.0])


def test_taps_executes_laminar_channel_navier_stokes_picard_kernel() -> None:
    problem = _minimal_fluid_problem()
    problem.geometry = GeometrySpec(id="geometry:fluid-2d", source=GeometrySource(kind="generated"), dimension=2)
    problem.operators[0].nondimensional_numbers.append(NondimensionalNumber(name="Re", value=50.0))
    problem.materials = [
        MaterialSpec(
            id="material:fluid",
            name="Laminar Newtonian fluid",
            phase="liquid",
            properties=[
                MaterialProperty(name="dynamic_viscosity", value=1.0, units="Pa*s"),
                MaterialProperty(name="density", value=1.0, units="kg/m^3"),
                MaterialProperty(name="pressure_drop", value=0.2, units="Pa"),
            ],
        )
    ]
    problem.boundary_conditions = [
        BoundaryConditionSpec(id="bc:wall", region_id="walls", field="U", kind="wall", value=[0.0, 0.0]),
        BoundaryConditionSpec(id="bc:inlet", region_id="x_min", field="U", kind="inlet", value=[1.0, 0.0]),
        BoundaryConditionSpec(id="bc:outlet", region_id="x_max", field="p", kind="outlet", value=0.0),
    ]

    support = estimate_taps_support(EstimateTAPSSupportInput(problem=problem)).support
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    validation = validate_taps_ir(ValidateTAPSIRInput(problem=problem, taps_problem=taps_problem))
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result

    assert support.supported is True
    assert plan.weak_form is not None
    assert any(term.role == "advection" for term in plan.weak_form.terms)
    assert validation.fallback_recommended is False
    assert result.status == "success"
    assert result.backend.startswith("taps:navier_stokes_channel_2d")
    assert result.scalar_outputs["full_navier_stokes_supported"] == 1
    assert result.scalar_outputs["support_scope"] == "restricted_steady_laminar_2d_channel"
    assert result.scalar_outputs["numerical_plan_solver_family"] == "incompressible_navier_stokes_channel_2d"
    assert result.residuals["normalized_nonlinear_residual"] <= 1.5e-10
    assert result.residuals["nonlinear_iterations"] >= 2.0
    assert any(artifact.kind == "taps_navier_stokes_solution_field" for artifact in result.artifacts)
    operator_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_navier_stokes_operator")
    operator_payload = json.loads(open(operator_artifact.uri, encoding="utf-8").read())
    assert operator_payload["numerical_plan_solver_family"] == "incompressible_navier_stokes_channel_2d"
    assert operator_payload["solver_controls_applied"]["support_scope"] == "restricted_steady_laminar_2d_channel"


def test_taps_rejects_high_re_navier_stokes_without_supported_phase3_scope() -> None:
    problem = _minimal_fluid_problem()
    problem.geometry = GeometrySpec(id="geometry:fluid-2d", source=GeometrySource(kind="generated"), dimension=2)
    problem.operators[0].nondimensional_numbers.append(NondimensionalNumber(name="Re", value=10000.0))
    problem.materials = [
        MaterialSpec(
            id="material:fluid",
            name="High-Re Newtonian fluid",
            phase="liquid",
            properties=[
                MaterialProperty(name="dynamic_viscosity", value=1.0),
                MaterialProperty(name="density", value=1.0),
                MaterialProperty(name="pressure_drop", value=1.0),
            ],
        )
    ]
    problem.boundary_conditions = [
        BoundaryConditionSpec(id="bc:wall", region_id="walls", field="U", kind="wall", value=[0.0, 0.0]),
        BoundaryConditionSpec(id="bc:inlet", region_id="x_min", field="U", kind="inlet", value=[1.0, 0.0]),
        BoundaryConditionSpec(id="bc:outlet", region_id="x_max", field="p", kind="outlet", value=0.0),
    ]

    support = estimate_taps_support(EstimateTAPSSupportInput(problem=problem)).support
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    validation = validate_taps_ir(ValidateTAPSIRInput(problem=problem, taps_problem=taps_problem))
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result

    assert support.supported is False
    assert validation.fallback_recommended is True
    assert result.status == "needs_review"
    assert result.scalar_outputs["message"].startswith("TAPSProblem compiled, but no executable TAPS kernel")


def test_mesh_navier_stokes_exports_reviewed_backend_bridge_without_local_execution(tmp_path) -> None:
    mesh_graph_path = tmp_path / "fluid_mesh_graph.json"
    mesh_graph_path.write_text(
        json.dumps(
            {
                "type": "mesh_graph",
                "nodes": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                "edges": [[0, 1], [1, 2], [2, 3], [3, 0]],
                "cells": [{"type": "triangle", "nodes": [0, 1, 2]}, {"type": "triangle", "nodes": [0, 2, 3]}],
                "node_count": 4,
                "boundary_edge_sets": {"boundary:inlet": [3], "boundary:outlet": [1], "boundary:wall": [0, 2]},
                "physical_boundary_groups": [
                    {"name": "inlet", "dimension": 1, "edge_ids": [3], "solver_native": {"openfoam_patch": "inlet", "su2_marker": "inlet"}},
                    {"name": "outlet", "dimension": 1, "edge_ids": [1], "solver_native": {"openfoam_patch": "outlet", "su2_marker": "outlet"}},
                    {"name": "wall", "dimension": 1, "edge_ids": [0, 2], "solver_native": {"openfoam_patch": "wall", "su2_marker": "wall"}},
                ],
            }
        ),
        encoding="utf-8",
    )
    geometry = GeometrySpec(
        id="geometry:fluid-mesh",
        source=GeometrySource(kind="generated"),
        dimension=2,
        encodings=[GeometryEncoding(kind="mesh_graph", uri=str(mesh_graph_path), target_backend="taps")],
    )
    problem = _minimal_fluid_problem()
    problem.geometry = geometry
    problem.operators[0].nondimensional_numbers.append(NondimensionalNumber(name="Re", value=1000.0))
    problem.materials = [
        MaterialSpec(
            id="material:fluid",
            name="Mesh Newtonian fluid",
            phase="liquid",
            properties=[
                MaterialProperty(name="dynamic_viscosity", value=1.0),
                MaterialProperty(name="density", value=1.0),
                MaterialProperty(name="pressure_drop", value=1.0),
            ],
        )
    ]
    problem.boundary_conditions = [
        BoundaryConditionSpec(id="bc:wall", region_id="wall", field="U", kind="wall", value=[0.0, 0.0]),
        BoundaryConditionSpec(id="bc:inlet", region_id="inlet", field="U", kind="inlet", value=[1.0, 0.0]),
        BoundaryConditionSpec(id="bc:outlet", region_id="outlet", field="p", kind="outlet", value=0.0),
    ]

    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    validation = validate_taps_ir(ValidateTAPSIRInput(problem=problem, taps_problem=taps_problem))
    fallback = plan_taps_adaptive_fallback(
        PlanTAPSAdaptiveFallbackInput(problem=problem, taps_problem=taps_problem, preferred_backend="openfoam")
    )
    preparation_plan = plan_backend_preparation(
        PlanBackendPreparationInput(problem=problem, taps_problem=taps_problem, backend="openfoam")
    )
    bridge = export_taps_backend_bridge(ExportTAPSBackendBridgeInput(problem=problem, taps_problem=taps_problem, backend="openfoam"))
    bundle = prepare_taps_backend_case_bundle(
        PrepareTAPSBackendCaseBundleInput(problem=problem, taps_problem=taps_problem, backend="openfoam")
    )
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result

    assert validation.valid
    assert validation.fallback_recommended is True
    assert any(check["name"] == "mesh_navier_stokes_backend_bridge_available" and check["passes"] for check in validation.checks)
    assert fallback.decision["mode"] == "export_backend_bridge"
    assert preparation_plan.status == "needs_inputs"
    assert preparation_plan.approval_gate["execute_external_solver"] is False
    assert preparation_plan.stabilization_policy["required_review"] is True
    assert result.status == "needs_review"
    bridge_payload = json.loads(open(bridge.manifest.uri, encoding="utf-8").read())
    assert any(block["family"] == "incompressible_navier_stokes_mesh" for block in bridge_payload["blocks"])
    assert bridge_payload["fallback_policy"]["preferred_full_solver_targets"] == ["openfoam", "su2"]
    assert bridge_payload["mesh_requirements"]["required_boundary_roles"] == ["inlet", "outlet", "wall"]
    assert bridge_payload["backend_preparation_plan"]["target_backend"] == "openfoam"
    assert bridge_payload["backend_preparation_plan"]["stabilization_policy"]["required_review"] is True
    assert bridge_payload["backend_preparation_validation"]["valid"] is True
    assert bridge.draft_artifact is not None
    assert "simpleFoam" in open(bridge.draft_artifact.uri, encoding="utf-8").read()
    bundle_payload = json.loads(open(bundle.bundle.uri, encoding="utf-8").read())
    assert bundle_payload["mesh_export"]["required"] is True
    assert bundle_payload["approval_gate"]["execute_external_solver"] is False
    assert bundle_payload["backend_preparation_plan"]["approval_gate"]["execute_external_solver"] is False
    assert bundle_payload["stabilization_policy"]["required_review"] is True
    assert {item["kind"] for item in bundle_payload["boundary_binding"]} >= {"inlet", "outlet", "wall"}
    assert any(check["name"] == "openfoam_runner_available" for check in bundle_payload["dependency_checks"])
    openfoam_runner = prepare_openfoam_runner_manifest(
        PrepareOpenFOAMRunnerManifestInput(case_bundle=bundle.bundle, solver="simpleFoam")
    )
    runner_manifest = json.loads(open(openfoam_runner.runner_manifest.uri, encoding="utf-8").read())
    assert runner_manifest["schema_version"] == "physicsos.full_solver_job.v1"
    assert runner_manifest["backend"] == "openfoam"
    assert runner_manifest["openfoam"]["solver"] == "simpleFoam"
    assert runner_manifest["openfoam"]["mesh_mode"] == "blockMesh"
    case_paths = {file["path"] for file in runner_manifest["openfoam"]["case_files"]}
    assert {
        "system/blockMeshDict",
        "0/U",
        "0/p",
        "constant/transportProperties",
        "constant/turbulenceProperties",
        "system/fvSchemes",
        "system/fvSolution",
    } <= case_paths
    case_contents = {file["path"]: file["content"] for file in runner_manifest["openfoam"]["case_files"]}
    assert "writeInterval   1;" in case_contents["system/controlDict"]
    assert "residualControl" not in case_contents["system/fvSolution"]
    assert "simulationType  laminar;" in case_contents["constant/turbulenceProperties"]
    assert "div((nuEff*dev2(T(grad(U))))) Gauss linear;" in case_contents["system/fvSchemes"]
    dry_run = submit_full_solver_job(SubmitFullSolverJobInput(runner_manifest=openfoam_runner.runner_manifest, mode="dry_run"))
    assert dry_run.submitted is False
    assert dry_run.result.status == "needs_review"


def test_taps_executes_custom_scalar_elliptic_weak_form_ir() -> None:
    geometry = GeometrySpec(id="geometry:custom-line", source=GeometrySource(kind="generated"), dimension=1)
    problem = PhysicsProblem(
        id="problem:custom-scalar-elliptic-1d",
        user_intent={"raw_request": "solve a custom scalar weak form with grad(v) k grad(phi) and source f"},
        domain="custom",
        geometry=geometry,
        fields=[FieldSpec(name="phi", kind="scalar")],
        operators=[
            OperatorSpec(
                id="operator:custom-elliptic",
                name="Custom scalar elliptic equation",
                domain="custom",
                equation_class="custom",
                form="weak",
                fields_out=["phi"],
                differential_terms=[
                    {
                        "expression": "int_Omega grad(v_phi) dot k grad(phi) dOmega",
                        "order": 2,
                        "fields": ["phi"],
                    }
                ],
                source_terms=[{"expression": "int_Omega v_phi f dOmega"}],
            )
        ],
        materials=[MaterialSpec(id="material:custom", name="custom", phase="solid", properties=[MaterialProperty(name="k", value=1.0)])],
        boundary_conditions=[
            {"id": "bc:left", "region_id": "x=0", "field": "phi", "kind": "dirichlet", "value": 0.0},
            {"id": "bc:right", "region_id": "x=1", "field": "phi", "kind": "dirichlet", "value": 0.0},
        ],
        targets=[{"name": "field", "field": "phi", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    solve_plan = plan_numerical_solve(NumericalSolvePlanInput(problem=problem, taps_problem=taps_problem))
    validation = validate_numerical_solve_plan(
        ValidateNumericalSolvePlanInput(problem=problem, taps_problem=taps_problem, plan=solve_plan)
    )
    assert validation.valid
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result
    assert result.scalar_outputs["weak_form_ir_blocks"] == 1.0
    operator_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_assembled_operator")
    operator_payload = json.loads(open(operator_artifact.uri, encoding="utf-8").read())
    assert operator_payload["weak_form_blocks"]["operator_family"] == "scalar_elliptic"
    assert {block["role"] for block in operator_payload["weak_form_blocks"]["blocks"]} >= {"diffusion", "source"}


def test_numerical_solve_plan_preserves_nonzero_dirichlet_boundaries() -> None:
    built = BuildPhysicsProblemInput(
        user_request="simulate 1D steady heat conduction in a rod with T(0)=300 K and T(1)=350 K"
    )
    problem = build_physics_problem(built).problem
    assert problem is not None
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    solve_plan = plan_numerical_solve(NumericalSolvePlanInput(problem=problem, taps_problem=taps_problem))
    validation = validate_numerical_solve_plan(
        ValidateNumericalSolvePlanInput(problem=problem, taps_problem=taps_problem, plan=solve_plan)
    )
    assert validation.valid
    assert solve_plan.solver_family == "scalar_elliptic_1d"
    assert [(bc.region_id, bc.value) for bc in solve_plan.boundary_condition_bindings] == [("x=0", 300.0), ("x=L", 350.0)]


def test_taps_1d_steady_heat_applies_nonzero_dirichlet_boundaries() -> None:
    problem = build_physics_problem(
        BuildPhysicsProblemInput(user_request="simulate 1D steady heat conduction in a rod with T(0)=300 K and T(1)=350 K")
    ).problem
    assert problem is not None
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result
    solution_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_solution_field")
    solution = json.loads(open(solution_artifact.uri, encoding="utf-8").read())
    values = solution["values"]
    middle = values[len(values) // 2]
    assert result.status == "success"
    assert result.scalar_outputs["boundary_condition_error"] == 0.0
    assert values[0] == 300.0
    assert values[-1] == 350.0
    assert middle == pytest.approx(325.0, abs=1.0)
    assert solution["boundary_values_applied"] == {"left": 300.0, "right": 350.0}


def test_taps_1d_steady_heat_accepts_llm_boundary_ids() -> None:
    problem = build_physics_problem(
        BuildPhysicsProblemInput(user_request="simulate 1D steady heat conduction in a rod with T(0)=300 K and T(1)=350 K")
    ).problem
    assert problem is not None
    problem = problem.model_copy(
        update={
            "boundary_conditions": [
                condition.model_copy(update={"region_id": "bnd_x0" if condition.value == 300.0 else "bnd_x1"})
                for condition in problem.boundary_conditions
            ]
        }
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result
    solution_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_solution_field")
    solution = json.loads(open(solution_artifact.uri, encoding="utf-8").read())

    assert result.status == "success"
    assert solution["boundary_values_applied"] == {"left": 300.0, "right": 350.0}
    assert solution["values"][0] == 300.0
    assert solution["values"][-1] == 350.0


def test_canonicalize_physics_problem_assigns_explicit_roles_for_opaque_boundary_ids() -> None:
    problem = build_physics_problem(
        BuildPhysicsProblemInput(user_request="simulate 1D steady heat conduction in a rod with T(0)=300 K and T(1)=350 K")
    ).problem
    assert problem is not None
    problem = problem.model_copy(
        update={
            "boundary_conditions": [
                BoundaryConditionSpec(
                    id="bc:hot",
                    region_id="surface_hot_end",
                    boundary_role="x_min",
                    field="T",
                    kind="dirichlet",
                    value=300.0,
                    units="K",
                ),
                BoundaryConditionSpec(
                    id="bc:cold",
                    region_id="surface_cold_end",
                    boundary_role="x_max",
                    field="T",
                    kind="dirichlet",
                    value=350.0,
                    units="K",
                ),
            ]
        }
    )

    canonical = canonicalize_physics_problem(CanonicalPhysicsProblemInput(problem=problem))
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=canonical.problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=canonical.problem, compilation_plan=plan)).taps_problem
    solve_plan = plan_numerical_solve(NumericalSolvePlanInput(problem=canonical.problem, taps_problem=taps_problem))
    validation = validate_numerical_solve_plan(
        ValidateNumericalSolvePlanInput(problem=canonical.problem, taps_problem=taps_problem, plan=solve_plan)
    )
    result = run_taps_backend(RunTAPSBackendInput(problem=canonical.problem, taps_problem=taps_problem, numerical_plan=solve_plan)).result
    solution_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_solution_field")
    solution = json.loads(open(solution_artifact.uri, encoding="utf-8").read())

    assert canonical.status == "ready"
    assert canonical.role_assignments == {"bc:hot": "x_min", "bc:cold": "x_max"}
    assert validation.valid
    assert result.status == "success"
    assert solution["boundary_values_applied"] == {"left": 300.0, "right": 350.0}


def test_workflow_stops_imported_geometry_with_missing_solver_critical_boundary_roles(monkeypatch) -> None:
    geometry = GeometrySpec(id="geometry:imported-opaque", source=GeometrySource(kind="mesh_file", uri="opaque.msh"), dimension=3)
    problem = build_physics_problem(
        BuildPhysicsProblemInput(
            user_request="simulate heat on imported mesh",
            geometry=geometry,
        )
    ).problem
    assert problem is not None
    problem = problem.model_copy(
        update={
            "boundary_conditions": [
                BoundaryConditionSpec(id="bc:hot", region_id="surface_17", field="T", kind="dirichlet", value=300.0),
                BoundaryConditionSpec(id="bc:cold", region_id="surface_42", field="T", kind="dirichlet", value=350.0),
            ]
        }
    )

    def fake_client(request):
        assert request["agent_name"] == "build-physics-problem-agent"
        return {"problem": problem.model_dump(mode="json"), "missing_inputs": [], "assumptions": ["opaque imported groups"]}

    output = run_typed_physicsos_workflow(
        RunTypedPhysicsOSWorkflowInput(
            user_request="simulate heat on imported mesh",
            use_knowledge=False,
            core_agents_mode="llm",
        ),
        structured_client=fake_client,
    )

    assert output.workflow is None
    assert output.canonicalization is not None
    assert output.canonicalization.status == "needs_user_input"
    assert output.canonicalization.missing_boundary_roles == ["bc:hot", "bc:cold"]


def test_verification_rejects_solver_artifact_with_wrong_boundary_values(tmp_path) -> None:
    problem = build_physics_problem(
        BuildPhysicsProblemInput(user_request="simulate 1D steady heat conduction in a rod with T(0)=300 K and T(1)=350 K")
    ).problem
    assert problem is not None
    solution_path = tmp_path / "solution_field.json"
    solution_path.write_text(
        json.dumps(
            {
                "field": "T",
                "axis": "x",
                "x": [0.0, 1.0],
                "values": [0.0, 0.0],
                "boundary_values_applied": {"left": 0.0, "right": 0.0},
            }
        ),
        encoding="utf-8",
    )
    result = SolverResult(
        id=f"result:taps:{problem.id}",
        problem_id=problem.id,
        backend="taps:weak_ir_scalar_elliptic_1d:test",
        status="success",
        residuals={"normalized_linear_residual": 0.0, "boundary_condition_error": 0.0},
        artifacts=[ArtifactRef(uri=str(solution_path), kind="taps_solution_field", format="json")],
        provenance=Provenance(created_by="test"),
    )

    check = check_boundary_condition_application(CheckBoundaryConditionApplicationInput(problem=problem, result=result))

    assert not check.passes
    assert check.errors == {"bc:left": 300.0, "bc:right": 350.0}


def test_verification_checks_boundary_values_by_canonical_role_and_boundary_id(tmp_path) -> None:
    problem = build_physics_problem(
        BuildPhysicsProblemInput(user_request="simulate 1D steady heat conduction in a rod with T(0)=300 K and T(1)=350 K")
    ).problem
    assert problem is not None
    problem = problem.model_copy(
        update={
            "boundary_conditions": [
                BoundaryConditionSpec(id="bc:hot", region_id="surface_hot", boundary_role="x_min", field="T", kind="dirichlet", value=300.0),
                BoundaryConditionSpec(id="bc:cold", region_id="surface_cold", boundary_role="x_max", field="T", kind="dirichlet", value=350.0),
            ]
        }
    )
    solution_path = tmp_path / "solution_field_roles.json"
    solution_path.write_text(
        json.dumps(
            {
                "field": "T",
                "values": [300.0, 349.0],
                "boundary_values_applied": {"x_min": 300.0, "bc:cold": 349.0},
            }
        ),
        encoding="utf-8",
    )
    result = SolverResult(
        id=f"result:taps:{problem.id}",
        problem_id=problem.id,
        backend="taps:test",
        status="success",
        residuals={"normalized_linear_residual": 0.0},
        artifacts=[ArtifactRef(uri=str(solution_path), kind="taps_solution_field", format="json")],
        provenance=Provenance(created_by="test"),
    )

    check = check_boundary_condition_application(CheckBoundaryConditionApplicationInput(problem=problem, result=result))

    assert not check.passes
    assert check.checked_boundaries == ["bc:hot", "bc:cold"]
    assert check.errors == {"bc:cold": 1.0}


def test_taps_executes_custom_transient_diffusion_weak_form_ir() -> None:
    geometry = GeometrySpec(id="geometry:custom-transient-line", source=GeometrySource(kind="generated"), dimension=1)
    problem = PhysicsProblem(
        id="problem:custom-transient-diffusion-1d",
        user_intent={"raw_request": "solve a custom transient diffusion weak form"},
        domain="custom",
        geometry=geometry,
        fields=[FieldSpec(name="T", kind="scalar")],
        operators=[
            OperatorSpec(
                id="operator:custom-transient",
                name="Custom transient diffusion",
                domain="custom",
                equation_class="custom",
                form="weak",
                fields_out=["T"],
                differential_terms=[
                    {"expression": "int_Omega v dT/dt dOmega", "order": 1, "fields": ["T"]},
                    {"expression": "int_Omega grad(v) dot alpha grad(T) dOmega", "order": 2, "fields": ["T"]},
                ],
                source_terms=[{"expression": "0"}],
            )
        ],
        materials=[MaterialSpec(id="material:thermal-custom", name="thermal custom", phase="solid", properties=[MaterialProperty(name="alpha", value=0.05)])],
        boundary_conditions=[
            {"id": "bc:left", "region_id": "x=0", "field": "T", "kind": "dirichlet", "value": 0.0},
            {"id": "bc:right", "region_id": "x=1", "field": "T", "kind": "dirichlet", "value": 0.0},
        ],
        initial_conditions=[InitialConditionSpec(id="ic:T", field="T", value={"expression": "sin(pi*x)", "language": "text"})],
        targets=[{"name": "field", "field": "T", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    for axis in plan.axes:
        if axis.kind in {"space", "time"}:
            axis.points = 16
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    validation = validate_taps_ir(ValidateTAPSIRInput(problem=problem, taps_problem=taps_problem))
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result
    bridge = export_taps_backend_bridge(ExportTAPSBackendBridgeInput(problem=problem, taps_problem=taps_problem, backend="fenicsx"))
    fallback = plan_taps_adaptive_fallback(PlanTAPSAdaptiveFallbackInput(problem=problem, taps_problem=taps_problem))
    bundle = prepare_taps_backend_case_bundle(PrepareTAPSBackendCaseBundleInput(problem=problem, taps_problem=taps_problem, backend="fenicsx"))
    assert validation.valid
    assert any(check["name"] == "executable_block_mapping_connected" and check["passes"] for check in validation.checks)
    assert not validation.fallback_recommended
    assert fallback.decision["mode"] == "run_taps_backend"
    assert result.status == "success"
    assert result.backend == "taps:weak_ir_transient_diffusion_1d:custom"
    assert result.scalar_outputs["weak_form_ir_blocks"] == 1.0
    assert result.scalar_outputs["numerical_plan_solver_family"] == "transient_diffusion_1d"
    metadata_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_reconstruction_metadata")
    metadata_payload = json.loads(open(metadata_artifact.uri, encoding="utf-8").read())
    assert metadata_payload["weak_form_blocks"]["operator_family"] == "transient_diffusion"
    assert metadata_payload["field"] == "T"
    assert metadata_payload["initial_condition_bindings"][0]["id"] == "ic:T"
    assert metadata_payload["coefficient_values_applied"]["rank"] == 8
    bridge_payload = json.loads(open(bridge.manifest.uri, encoding="utf-8").read())
    assert bridge_payload["schema_version"] == "physicsos.taps_backend_bridge.v1"
    assert {"family": "transient_diffusion", "space": "H1", "time_integrator": "implicit_euler_or_crank_nicolson"} in bridge_payload["blocks"]
    assert bridge.draft_artifact is not None
    assert "Crank" in open(bridge.draft_artifact.uri, encoding="utf-8").read()
    bundle_payload = json.loads(open(bundle.bundle.uri, encoding="utf-8").read())
    assert bundle_payload["schema_version"] == "physicsos.taps_backend_case_bundle.v1"
    assert bundle_payload["approval_gate"]["execute_external_solver"] is False
    assert bundle_payload["approval_gate"]["requires_user_approval"] is True
    assert any(check["name"] == "python_import_dolfinx" for check in bundle_payload["dependency_checks"])


def test_taps_compiles_strong_form_and_boundary_weak_terms() -> None:
    geometry = GeometrySpec(id="geometry:strong-line", source=GeometrySource(kind="generated"), dimension=1)
    problem = PhysicsProblem(
        id="problem:strong-form-diffusion-1d",
        user_intent={"raw_request": "compile a strong-form diffusion equation with a Neumann weak boundary term"},
        domain="custom",
        geometry=geometry,
        fields=[FieldSpec(name="u", kind="scalar")],
        operators=[
            OperatorSpec(
                id="operator:strong-diffusion",
                name="Strong diffusion",
                domain="custom",
                equation_class="custom",
                form="strong",
                fields_out=["u"],
                differential_terms=[{"expression": "-div(k grad(u)) = f", "order": 2, "fields": ["u"]}],
                source_terms=[{"expression": "int_Omega v_u f dOmega"}],
            )
        ],
        materials=[MaterialSpec(id="material:strong", name="strong", phase="solid", properties=[MaterialProperty(name="k", value=1.0)])],
        boundary_conditions=[
            {"id": "bc:left", "region_id": "x=0", "field": "u", "kind": "dirichlet", "value": 0.0},
            {"id": "bc:right_flux", "region_id": "x=1", "field": "u", "kind": "neumann", "value": 0.0},
        ],
        targets=[{"name": "field", "field": "u", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    assert taps_problem.weak_form is not None
    assert taps_problem.weak_form.terms[0].role == "diffusion"
    assert "grad(v_u)" in taps_problem.weak_form.terms[0].expression
    assert taps_problem.weak_form.boundary_terms
    assert taps_problem.weak_form.boundary_terms[0].integration_domain == "x=1"
    solve_plan = plan_numerical_solve(NumericalSolvePlanInput(problem=problem, taps_problem=taps_problem))
    validation = validate_numerical_solve_plan(
        ValidateNumericalSolvePlanInput(problem=problem, taps_problem=taps_problem, plan=solve_plan)
    )
    assert not validation.valid
    assert any("requires canonical Dirichlet boundary_role values x_min and x_max" in error for error in validation.errors)


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


def test_taps_adaptive_fallback_exports_bridge_for_unsupported_ir() -> None:
    geometry = GeometrySpec(id="geometry:unsupported-custom", source=GeometrySource(kind="generated"), dimension=1)
    problem = PhysicsProblem(
        id="problem:unsupported-custom-ir",
        user_intent={"raw_request": "compile an unsupported algebraic custom weak form"},
        domain="custom",
        geometry=geometry,
        fields=[FieldSpec(name="q", kind="scalar")],
        operators=[
            OperatorSpec(
                id="operator:unsupported",
                name="Unsupported custom weak form",
                domain="custom",
                equation_class="custom",
                form="weak",
                fields_out=["q"],
                differential_terms=[{"expression": "int_Omega v_q custom_memory_kernel(q) dOmega", "fields": ["q"]}],
            )
        ],
        materials=[],
        boundary_conditions=[],
        targets=[{"name": "field", "field": "q", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    validation = validate_taps_ir(ValidateTAPSIRInput(problem=problem, taps_problem=taps_problem))
    fallback = plan_taps_adaptive_fallback(PlanTAPSAdaptiveFallbackInput(problem=problem, taps_problem=taps_problem, preferred_backend="mfem"))
    bridge = export_taps_backend_bridge(ExportTAPSBackendBridgeInput(problem=problem, taps_problem=taps_problem, backend="mfem"))
    assert validation.valid
    assert validation.fallback_recommended
    assert validation.recommended_action == "author_runtime_extension_or_export_full_solver"
    assert fallback.decision["mode"] == "export_backend_bridge"
    assert fallback.decision["execute_external_solver"] is False
    assert bridge.draft_artifact is not None
    assert "PyMFEM" in open(bridge.draft_artifact.uri, encoding="utf-8").read()


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
    visualizations = generate_visualizations(GenerateVisualizationsInput(problem=problem, result=result))
    assert any(artifact.kind == "visualization:solution_line" and artifact.format == "png" for artifact in visualizations.artifacts)


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
    assert result.scalar_outputs["numerical_plan_solver_family"] == "scalar_elliptic_2d"
    assert {artifact.kind for artifact in result.artifacts} >= {
        "taps_assembled_operator",
        "taps_solution_field",
        "taps_residual_history",
    }
    solution_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_solution_field")
    solution_payload = json.loads(open(solution_artifact.uri, encoding="utf-8").read())
    assert solution_payload["field"] == "u"
    assert solution_payload["boundary_values_applied"] == {"left": 0.0, "right": 0.0, "bottom": 0.0, "top": 0.0}
    assert solution_payload["coefficient_values_applied"]["diffusion"] == 1.0
    assert "boundary_condition_error" in solution_payload["residual_checks"]
    visualizations = generate_visualizations(GenerateVisualizationsInput(problem=problem, result=result))
    assert any(artifact.kind == "visualization:solution_heatmap" and artifact.format == "png" for artifact in visualizations.artifacts)


def test_taps_2d_scalar_elliptic_applies_nonzero_dirichlet_boundary_lifting() -> None:
    geometry = GeometrySpec(id="geometry:square-nonzero-bc", source=GeometrySource(kind="generated"), dimension=2)
    problem = PhysicsProblem(
        id="problem:poisson-2d-nonzero-bc",
        user_intent={"raw_request": "solve 2D Poisson with nonzero Dirichlet boundary"},
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
        boundary_conditions=[{"id": "bc:u", "region_id": "boundary", "field": "u", "kind": "dirichlet", "value": 5.0}],
        targets=[{"name": "field", "field": "u", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    solve_plan = plan_numerical_solve(NumericalSolvePlanInput(problem=problem, taps_problem=taps_problem))
    validation = validate_numerical_solve_plan(
        ValidateNumericalSolvePlanInput(problem=problem, taps_problem=taps_problem, plan=solve_plan)
    )
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result
    solution_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_solution_field")
    solution_payload = json.loads(open(solution_artifact.uri, encoding="utf-8").read())
    values = solution_payload["values"]
    assert validation.valid
    assert result.status == "success"
    assert result.scalar_outputs["boundary_condition_error"] == 0.0
    assert solution_payload["boundary_values_applied"] == {"left": 5.0, "right": 5.0, "bottom": 5.0, "top": 5.0}
    assert all(value == 5.0 for value in values[0])
    assert all(value == 5.0 for value in values[-1])
    assert all(row[0] == 5.0 and row[-1] == 5.0 for row in values)


def test_taps_generic_scalar_assembler_executes_3d_heat_with_endpoint_roles() -> None:
    geometry = GeometrySpec(id="geometry:cylinder-structured-3d", source=GeometrySource(kind="generated"), dimension=3)
    problem = PhysicsProblem(
        id="problem:heat-3d-structured",
        user_intent={"raw_request": "solve 3D steady heat conduction with x_min=300 K and x_max=350 K"},
        domain="thermal",
        geometry=geometry,
        fields=[FieldSpec(name="T", kind="scalar", units="K")],
        operators=[
            OperatorSpec(
                id="operator:heat-3d",
                name="Steady heat conduction",
                domain="thermal",
                equation_class="elliptic PDE",
                form="weak",
                fields_out=["T"],
                differential_terms=[{"expression": "div(k grad(T)) = 0", "order": 2, "fields": ["T"]}],
            )
        ],
        materials=[MaterialSpec(id="material:solid", name="solid", phase="solid", properties=[MaterialProperty(name="thermal_conductivity", value=15.0)])],
        boundary_conditions=[
            {"id": "bc:xmin", "region_id": "bnd_hot_min", "boundary_role": "x_min", "field": "T", "kind": "dirichlet", "value": 300.0, "units": "K"},
            {"id": "bc:xmax", "region_id": "bnd_hot_max", "boundary_role": "x_max", "field": "T", "kind": "dirichlet", "value": 350.0, "units": "K"},
        ],
        targets=[{"name": "temperature", "field": "T", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    for axis in plan.axes:
        if axis.kind == "space":
            axis.points = 8
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result
    residual = estimate_taps_residual(EstimateTAPSResidualInput(problem=problem, taps_problem=taps_problem, result=result)).report
    solution_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_solution_field")
    solution_payload = json.loads(open(solution_artifact.uri, encoding="utf-8").read())
    bc_check = check_boundary_condition_application(CheckBoundaryConditionApplicationInput(problem=problem, result=result))

    assert result.backend.startswith("taps:weak_ir_scalar_elliptic_3d")
    assert result.status == "success"
    assert residual.converged
    assert result.scalar_outputs["numerical_plan_solver_family"] == "scalar_elliptic_3d"
    assert solution_payload["boundary_values_applied"]["left"] == 300.0
    assert solution_payload["boundary_values_applied"]["right"] == 350.0
    assert solution_payload["values"][0][0][0] == 300.0
    assert solution_payload["values"][-1][0][0] == 350.0
    assert bc_check.passes


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
    assert result.scalar_outputs["numerical_plan_solver_family"] == "nonlinear_reaction_diffusion_1d"
    assert "normalized_nonlinear_residual" in result.residuals
    assert "nonlinear_iterations" in result.residuals
    assert {artifact.kind for artifact in result.artifacts} >= {
        "taps_nonlinear_operator",
        "taps_solution_field",
        "taps_iteration_history",
    }
    solution_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_solution_field")
    solution_payload = json.loads(open(solution_artifact.uri, encoding="utf-8").read())
    assert solution_payload["coefficient_values_applied"]["diffusion"] == 1.0
    assert solution_payload["solver_controls_applied"]["damping"] == 0.8


def test_taps_executes_custom_nonlinear_reaction_diffusion_weak_form_ir() -> None:
    geometry = GeometrySpec(id="geometry:custom-rd-line", source=GeometrySource(kind="generated"), dimension=1)
    problem = PhysicsProblem(
        id="problem:custom-nonlinear-rd-1d",
        user_intent={"raw_request": "solve a custom scalar nonlinear reaction diffusion weak form"},
        domain="custom",
        geometry=geometry,
        fields=[FieldSpec(name="u", kind="scalar")],
        operators=[
            OperatorSpec(
                id="operator:custom-rd",
                name="Custom nonlinear reaction diffusion",
                domain="custom",
                equation_class="custom",
                form="weak",
                fields_out=["u"],
                differential_terms=[
                    {"expression": "int_Omega grad(v) dot D grad(u) dOmega", "order": 2, "fields": ["u"]},
                    {"expression": "- int_Omega v R(u) dOmega with nonlinear cubic u^3 reaction", "order": 0, "fields": ["u"]},
                ],
                source_terms=[{"expression": "int_Omega v f dOmega"}],
            )
        ],
        materials=[MaterialSpec(id="material:custom-rd", name="custom rd", phase="solid", properties=[MaterialProperty(name="D", value=1.0)])],
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
    assert result.status == "success"
    assert result.backend == "taps:weak_ir_nonlinear_reaction_diffusion_1d:custom"
    assert result.scalar_outputs["weak_form_ir_blocks"] == 1.0
    assert "normalized_nonlinear_residual" in result.residuals
    operator_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_nonlinear_operator")
    operator_payload = json.loads(open(operator_artifact.uri, encoding="utf-8").read())
    assert operator_payload["weak_form_blocks"]["operator_family"] == "nonlinear_reaction_diffusion"
    assert {block["role"] for block in operator_payload["weak_form_blocks"]["blocks"]} >= {"diffusion", "nonlinear_reaction", "source"}


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
    assert result.scalar_outputs["numerical_plan_solver_family"] == "nonlinear_reaction_diffusion_2d"
    assert "normalized_nonlinear_residual" in result.residuals
    assert "nonlinear_iterations" in result.residuals
    assert {artifact.kind for artifact in result.artifacts} >= {
        "taps_nonlinear_operator",
        "taps_solution_field",
        "taps_iteration_history",
    }
    solution_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_solution_field")
    solution_payload = json.loads(open(solution_artifact.uri, encoding="utf-8").read())
    assert solution_payload["coefficient_values_applied"]["linear_reaction"] == 1.0
    assert solution_payload["solver_controls_applied"]["max_iterations"] == 50


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
    assert result.scalar_outputs["numerical_plan_solver_family"] == "coupled_reaction_diffusion_2d"
    assert residual.converged
    assert "normalized_coupled_residual" in result.residuals
    assert {artifact.kind for artifact in result.artifacts} >= {
        "taps_coupled_operator",
        "taps_coupled_solution_fields",
        "taps_iteration_history",
    }
    operator_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_coupled_operator")
    operator_payload = json.loads(open(operator_artifact.uri, encoding="utf-8").read())
    assert operator_payload["numerical_plan_solver_family"] == "coupled_reaction_diffusion_2d"
    assert operator_payload["fields"] == ["u", "v"]
    assert operator_payload["solver_controls_applied"]["max_iterations"] == 50


def test_taps_executes_custom_coupled_field_weak_form_ir() -> None:
    geometry = GeometrySpec(id="geometry:custom-coupled-rd-square", source=GeometrySource(kind="generated"), dimension=2)
    problem = PhysicsProblem(
        id="problem:custom-coupled-rd-2d",
        user_intent={"raw_request": "solve a custom two-field coupled reaction diffusion weak form"},
        domain="custom",
        geometry=geometry,
        fields=[FieldSpec(name="u", kind="scalar"), FieldSpec(name="v", kind="scalar")],
        operators=[
            OperatorSpec(
                id="operator:custom-coupled-rd",
                name="Custom coupled reaction diffusion",
                domain="custom",
                equation_class="custom",
                form="weak",
                fields_out=["u", "v"],
                differential_terms=[
                    {"expression": "sum_i int_Omega grad(w_i) dot D_i grad(field_i) dOmega", "order": 2, "fields": ["u", "v"]},
                    {"expression": "int_Omega kappa (u - v) (w_u - w_v) dOmega", "order": 0, "fields": ["u", "v"]},
                    {"expression": "- sum_i int_Omega w_i R_i(u, v) dOmega with nonlinear reactions", "order": 0, "fields": ["u", "v"]},
                ],
                source_terms=[{"expression": "rhs f_u f_v"}],
            )
        ],
        materials=[MaterialSpec(id="material:custom-coupled", name="custom coupled", phase="solid", properties=[MaterialProperty(name="D", value=1.0)])],
        boundary_conditions=[
            {"id": "bc:u", "region_id": "boundary", "field": "u", "kind": "dirichlet", "value": 0.0},
            {"id": "bc:v", "region_id": "boundary", "field": "v", "kind": "dirichlet", "value": 0.0},
        ],
        targets=[{"name": "fields", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    for axis in plan.axes:
        axis.points = 20
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result
    assert result.status == "success"
    assert result.backend == "taps:weak_ir_coupled_reaction_diffusion_2d:custom"
    assert result.scalar_outputs["weak_form_ir_blocks"] == 1.0
    operator_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_coupled_operator")
    operator_payload = json.loads(open(operator_artifact.uri, encoding="utf-8").read())
    assert operator_payload["weak_form_blocks"]["operator_family"] == "coupled_reaction_diffusion"
    assert operator_payload["weak_form_blocks"]["subspace_solver"] == "block_gauss_seidel_picard"
    assert {block["role"] for block in operator_payload["weak_form_blocks"]["blocks"]} >= {
        "field_diffusion",
        "coupling_operator",
        "field_reaction",
        "source",
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


def test_tetra_p1_fem_assembly_unit_tetra() -> None:
    points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    stiffness, lumped_mass, total_volume, elements = _assemble_tetra_stiffness(points, [[0, 1, 2, 3]])

    assert total_volume == pytest.approx(1.0 / 6.0)
    assert lumped_mass == pytest.approx([1.0 / 24.0] * 4)
    assert elements[0]["basis"] == "p1_tetra"
    expected_gradients = [[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    for actual, expected in zip(elements[0]["grad_phi"], expected_gradients):
        assert actual == pytest.approx(expected)
    dense = [[stiffness[row].get(col, 0.0) for col in range(4)] for row in range(4)]
    assert dense[0][0] == pytest.approx(0.5)
    assert dense[0][1] == pytest.approx(-1.0 / 6.0)
    assert dense[1][1] == pytest.approx(1.0 / 6.0)


def test_tetra_p2_scalar_assembly_uses_tetra10_nodes() -> None:
    points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ]
    stiffness, lumped_mass, total_volume, elements = _assemble_tetra_stiffness(points, [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])

    assert total_volume == pytest.approx(1.0 / 6.0)
    assert sum(lumped_mass) == pytest.approx(1.0 / 6.0)
    assert len(stiffness) == 10
    assert elements[0]["basis"] == "p2_tetra"
    assert elements[0]["geometry"] == "isoparametric_quadratic_tetra"
    assert len(elements[0]["local_stiffness"]) == 10
    assert len(elements[0]["quadrature"]) == 4
    assert len(elements[0]["quadrature"][0]["grad_phi"]) == 10
    local = elements[0]["local_stiffness"]
    for i in range(10):
        for j in range(10):
            assert local[i][j] == pytest.approx(local[j][i], abs=1e-12)
        assert sum(local[i]) == pytest.approx(0.0, abs=1e-12)


def test_tetra_p2_isoparametric_geometry_uses_curved_mid_edge_nodes() -> None:
    straight_points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ]
    curved_points = [point[:] for point in straight_points]
    curved_points[8] = [0.65, 0.0, 0.65]

    _, _, straight_volume, straight_elements = _assemble_tetra_stiffness(straight_points, [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    _, _, curved_volume, curved_elements = _assemble_tetra_stiffness(curved_points, [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])

    assert straight_elements[0]["geometry"] == "isoparametric_quadratic_tetra"
    assert curved_elements[0]["geometry"] == "isoparametric_quadratic_tetra"
    assert straight_volume == pytest.approx(1.0 / 6.0)
    assert curved_volume != pytest.approx(straight_volume)
    assert [record["det_j"] for record in curved_elements[0]["quadrature"]] != pytest.approx(
        [record["det_j"] for record in straight_elements[0]["quadrature"]]
    )


def test_tetra10_poisson_flags_inverted_curved_element_quality(tmp_path: Path) -> None:
    mesh_graph_path = tmp_path / "tetra10_inverted_mesh_graph.json"
    mesh_graph_path.write_text(
        json.dumps(
            {
                "type": "mesh_graph",
                "source_mesh": "hand-built-inverted-tetra10",
                "node_count": 10,
                "points": [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.5, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.0, 0.5, 0.0],
                    [0.0, 0.0, 0.5],
                    [-3.0, 0.0, 1.0],
                    [0.0, 0.5, 0.5],
                ],
                "edges": [[0, 1], [1, 2], [0, 2], [0, 3], [1, 3], [2, 3]],
                "boundary_nodes": [0, 1, 2, 3],
                "cell_blocks": [{"type": "tetra10", "cells": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]}],
            }
        ),
        encoding="utf-8",
    )
    geometry = GeometrySpec(
        id="geometry:tetra10-inverted-mesh-graph",
        source=GeometrySource(kind="generated"),
        dimension=3,
        encodings=[GeometryEncoding(kind="mesh_graph", uri=str(mesh_graph_path), target_backend="taps")],
    )
    problem = PhysicsProblem(
        id="problem:mesh-tetra10-inverted-poisson-3d",
        user_intent={"raw_request": "solve 3D Poisson on an inverted second-order tetrahedral mesh graph"},
        domain="custom",
        geometry=geometry,
        fields=[FieldSpec(name="u", kind="scalar")],
        operators=[OperatorSpec(id="operator:poisson", name="Poisson", domain="custom", equation_class="poisson", form="weak", fields_out=["u"])],
        materials=[],
        boundary_conditions=[BoundaryConditionSpec(id="bc:u", region_id="boundary", field="u", kind="dirichlet", value=0.0)],
        targets=[{"name": "field", "field": "u", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result
    operator_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_mesh_fem_operator")
    operator_payload = json.loads(Path(operator_artifact.uri).read_text(encoding="utf-8"))

    assert result.status == "needs_review"
    assert result.residuals["mesh_quality_passes"] == 0.0
    assert operator_payload["mesh_quality"]["passes"] is False
    assert any("Jacobian determinant changes sign" in issue for issue in operator_payload["mesh_quality"]["issues"])


def test_mesh_graph_taps_solves_tetra_fem_poisson(tmp_path: Path) -> None:
    mesh_graph_path = tmp_path / "tetra_mesh_graph.json"
    mesh_graph_path.write_text(
        json.dumps(
            {
                "type": "mesh_graph",
                "source_mesh": "hand-built-tetra-star",
                "node_count": 5,
                "points": [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.25, 0.25, 0.25],
                ],
                "edges": [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [0, 4], [1, 4], [2, 4], [3, 4]],
                "boundary_nodes": [0, 1, 2, 3],
                "cell_blocks": [
                    {
                        "type": "tetra",
                        "cells": [[0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    geometry = GeometrySpec(
        id="geometry:tetra-mesh-graph",
        source=GeometrySource(kind="generated"),
        dimension=3,
        encodings=[GeometryEncoding(kind="mesh_graph", uri=str(mesh_graph_path), target_backend="taps")],
    )
    problem = PhysicsProblem(
        id="problem:mesh-tetra-poisson-3d",
        user_intent={"raw_request": "solve 3D Poisson on a tetrahedral mesh graph"},
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

    assert result.backend.startswith("taps:mesh_fem_poisson")
    assert result.status == "success"
    assert result.residuals["fem_tetrahedra"] == 4
    assert result.residuals["fem_triangles"] == 0
    operator_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_mesh_fem_operator")
    operator_payload = json.loads(Path(operator_artifact.uri).read_text(encoding="utf-8"))
    assert operator_payload["type"] == "tetra_p1_fem_poisson"
    assert operator_payload["total_volume"] == pytest.approx(1.0 / 6.0)


def test_mesh_graph_taps_solves_tetra10_fem_poisson(tmp_path: Path) -> None:
    mesh_graph_path = tmp_path / "tetra10_mesh_graph.json"
    mesh_graph_path.write_text(
        json.dumps(
            {
                "type": "mesh_graph",
                "source_mesh": "hand-built-tetra10",
                "node_count": 10,
                "points": [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.5, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.0, 0.5, 0.0],
                    [0.0, 0.0, 0.5],
                    [0.5, 0.0, 0.5],
                    [0.0, 0.5, 0.5],
                ],
                "edges": [[0, 1], [1, 2], [0, 2], [0, 3], [1, 3], [2, 3]],
                "boundary_nodes": [0, 1, 2, 3],
                "cell_blocks": [{"type": "tetra10", "cells": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]}],
            }
        ),
        encoding="utf-8",
    )
    geometry = GeometrySpec(
        id="geometry:tetra10-mesh-graph",
        source=GeometrySource(kind="generated"),
        dimension=3,
        encodings=[GeometryEncoding(kind="mesh_graph", uri=str(mesh_graph_path), target_backend="taps")],
    )
    problem = PhysicsProblem(
        id="problem:mesh-tetra10-poisson-3d",
        user_intent={"raw_request": "solve 3D Poisson on a second-order tetrahedral mesh graph"},
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

    assert result.backend.startswith("taps:mesh_fem_poisson")
    assert result.status == "success"
    assert result.residuals["fem_tetrahedra"] == 1
    assert result.residuals["fem_basis_order"] == 2
    operator_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_mesh_fem_operator")
    operator_payload = json.loads(Path(operator_artifact.uri).read_text(encoding="utf-8"))
    assert operator_payload["type"] == "tetra_p2_fem_poisson"
    assert any(element["basis"] == "p2_tetra" for element in operator_payload["elements"])


def test_mesh_graph_tetra_fem_poisson_applies_nonzero_dirichlet_roles(tmp_path: Path) -> None:
    mesh_graph_path = tmp_path / "tetra_bc_mesh_graph.json"
    mesh_graph_path.write_text(
        json.dumps(
            {
                "type": "mesh_graph",
                "source_mesh": "hand-built-tetra-star-bc",
                "node_count": 5,
                "points": [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.25, 0.25, 0.25],
                ],
                "edges": [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [0, 4], [1, 4], [2, 4], [3, 4]],
                "boundary_nodes": [0, 1, 2, 3],
                "boundary_node_sets": {"x_min": [0, 2, 3], "x_max": [1]},
                "cell_blocks": [
                    {"type": "tetra", "cells": [[0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4]]}
                ],
            }
        ),
        encoding="utf-8",
    )
    geometry = GeometrySpec(
        id="geometry:tetra-bc-mesh-graph",
        source=GeometrySource(kind="generated"),
        dimension=3,
        encodings=[GeometryEncoding(kind="mesh_graph", uri=str(mesh_graph_path), target_backend="taps")],
    )
    problem = PhysicsProblem(
        id="problem:mesh-tetra-poisson-bc-3d",
        user_intent={"raw_request": "solve 3D Poisson on a tetra mesh with x_min=2 and x_max=5"},
        domain="custom",
        geometry=geometry,
        fields=[FieldSpec(name="u", kind="scalar")],
        operators=[OperatorSpec(id="operator:poisson", name="Poisson", domain="custom", equation_class="poisson", form="weak", fields_out=["u"])],
        materials=[],
        boundary_conditions=[
            BoundaryConditionSpec(id="bc:xmin", region_id="x_min", boundary_role="x_min", field="u", kind="dirichlet", value=2.0),
            BoundaryConditionSpec(id="bc:xmax", region_id="x_max", boundary_role="x_max", field="u", kind="dirichlet", value=5.0),
        ],
        targets=[{"name": "field", "field": "u", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    numerical_plan = plan_numerical_solve(NumericalSolvePlanInput(problem=problem, taps_problem=taps_problem))
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem, numerical_plan=numerical_plan)).result

    assert result.status == "success"
    solution_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_mesh_fem_solution_field")
    solution_payload = json.loads(Path(solution_artifact.uri).read_text(encoding="utf-8"))
    assert solution_payload["values"][0] == pytest.approx(2.0)
    assert solution_payload["values"][2] == pytest.approx(2.0)
    assert solution_payload["values"][3] == pytest.approx(2.0)
    assert solution_payload["values"][1] == pytest.approx(5.0)
    assert solution_payload["boundary_values_applied"] == {"0": 2.0, "1": 5.0, "2": 2.0, "3": 2.0}


def test_mesh_graph_fem_poisson_applies_neumann_and_robin_boundary_terms() -> None:
    geometry = GeometrySpec(id="geometry:mesh-poisson-mixed-bc-square", source=GeometrySource(kind="generated"), dimension=2)
    mesh = generate_mesh(
        GenerateMeshInput(
            geometry=geometry,
            physics=PhysicsSpec(domains=["custom"]),
            mesh_policy=MeshPolicy(target_element_size=0.5),
            target_backends=["taps"],
        )
    ).mesh
    encoding_output = generate_geometry_encoding(
        GenerateGeometryEncodingInput(geometry=geometry, mesh=mesh, encodings=["mesh_graph"])
    )
    geometry.encodings.extend(encoding_output.encodings)
    problem = PhysicsProblem(
        id="problem:mesh-poisson-mixed-bc-2d",
        user_intent={"raw_request": "solve mesh FEM Poisson with Dirichlet, Neumann, and Robin boundaries"},
        domain="custom",
        geometry=geometry,
        mesh=mesh,
        fields=[FieldSpec(name="u", kind="scalar")],
        operators=[OperatorSpec(id="operator:poisson", name="Poisson", domain="custom", equation_class="poisson", form="weak", fields_out=["u"])],
        materials=[],
        boundary_conditions=[
            BoundaryConditionSpec(id="bc:left", region_id="x_min", boundary_role="x_min", field="u", kind="dirichlet", value=0.0),
            BoundaryConditionSpec(id="bc:right_flux", region_id="x_max", boundary_role="x_max", field="u", kind="neumann", value=2.0),
            BoundaryConditionSpec(id="bc:top_robin", region_id="y_max", boundary_role="y_max", field="u", kind="robin", value={"h": 3.0, "r": 1.5}),
        ],
        targets=[{"name": "field", "field": "u", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result
    operator_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_mesh_fem_operator")
    operator_payload = json.loads(Path(operator_artifact.uri).read_text(encoding="utf-8"))
    solution_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_mesh_fem_solution_field")
    solution_payload = json.loads(Path(solution_artifact.uri).read_text(encoding="utf-8"))

    assert result.status in {"success", "needs_review"}
    applied = operator_payload["boundary_weak_terms_applied"]
    assert {item["id"] for item in applied} == {"bc:right_flux", "bc:top_robin"}
    assert next(item for item in applied if item["id"] == "bc:right_flux")["value"] == pytest.approx(2.0)
    assert next(item for item in applied if item["id"] == "bc:top_robin")["coefficient"] == pytest.approx(3.0)
    assert solution_payload["boundary_weak_terms_applied"] == applied


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
    assert result.scalar_outputs["numerical_plan_solver_family"] == "mesh_fem_em_curl_curl"
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
    assert operator_payload["numerical_plan_solver_family"] == "mesh_fem_em_curl_curl"
    assert operator_payload["assembly"] == "nedelec_first_kind_edge_element_hcurl"
    assert operator_payload["edge_dof_count"] > 0
    assert operator_payload["boundary_edge_count"] > 0
    assert operator_payload["material"]["relative_permittivity"] == pytest.approx(2.5)
    assert operator_payload["material"]["wave_number"] == pytest.approx(0.4)
    assert operator_payload["material"]["source_amplitude"] == pytest.approx(0.75)
    assert operator_payload["coefficient_values_applied"]["relative_permittivity"] == pytest.approx(2.5)
    assert operator_payload["solver_controls_applied"]["max_iterations"] == 5000
    assert operator_payload["hcurl_scaffold"]["edge_dofs_required"] is True
    assert operator_payload["hcurl_scaffold"]["status"] == "nedelec_order1_edge_element"
    assert operator_payload["hcurl_scaffold"]["boundary_condition"] == "pec_tangential_zero"


def test_mesh_graph_taps_executes_custom_hcurl_curl_curl_weak_form_ir() -> None:
    geometry = GeometrySpec(id="geometry:custom-hcurl-square", source=GeometrySource(kind="generated"), dimension=2)
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
        id="problem:custom-hcurl-curl-curl-2d",
        user_intent={"raw_request": "solve a custom H(curl) curl-curl weak form on a square mesh"},
        domain="custom",
        geometry=geometry,
        mesh=mesh,
        fields=[FieldSpec(name="E", kind="vector")],
        operators=[
            OperatorSpec(
                id="operator:custom-hcurl",
                name="Custom Hcurl curl-curl weak form",
                domain="custom",
                equation_class="custom",
                form="weak",
                fields_out=["E"],
                differential_terms=[
                    {
                        "expression": "int_Omega mu^-1 curl(v) dot curl(E) dOmega",
                        "order": 2,
                        "fields": ["E"],
                    },
                    {
                        "expression": "- int_Omega k0^2 epsilon v dot E dOmega",
                        "order": 0,
                        "fields": ["E"],
                    },
                ],
                source_terms=[{"expression": "int_Omega v dot J dOmega"}],
            )
        ],
        materials=[
            MaterialSpec(
                id="material:custom-em",
                name="custom em",
                phase="custom",
                properties=[
                    MaterialProperty(name="relative_permittivity", value=1.5),
                    MaterialProperty(name="relative_permeability", value=1.0),
                    MaterialProperty(name="wave_number", value=0.35),
                    MaterialProperty(name="current_source", value=0.5),
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
    assert result.backend == "taps:weak_ir_em_curl_curl:custom"
    assert result.scalar_outputs["weak_form_ir_blocks"] == 1.0
    operator_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_mesh_fem_em_curl_curl_operator")
    operator_payload = json.loads(open(operator_artifact.uri, encoding="utf-8").read())
    assert operator_payload["weak_form_blocks"]["operator_family"] == "hcurl_curl_curl"
    assert {block["role"] for block in operator_payload["weak_form_blocks"]["blocks"]} >= {"curl_curl", "mass", "source"}
    assert operator_payload["hcurl_scaffold"]["edge_dofs_required"] is True


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


def test_tetra_nedelec_order1_assembly_unit_tetra() -> None:
    points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    stiffness, edges, faces, total_volume, elements = _assemble_tetra_nedelec_curl_curl(points, [[0, 1, 2, 3]])

    assert total_volume == pytest.approx(1.0 / 6.0)
    assert len(edges) == 6
    assert len(faces) == 4
    assert len(stiffness) == 6
    assert elements[0]["basis"] == "nedelec_first_kind_order1_tetra"
    assert len(elements[0]["quadrature"]) == 4
    assert len(elements[0]["curl_basis"]) == 6
    local = elements[0]["local_matrix"]
    for i in range(6):
        for j in range(6):
            assert local[i][j] == pytest.approx(local[j][i], abs=1e-12)
    assert all(row for row in stiffness)


def test_mesh_graph_taps_solves_tetra_em_curl_curl(tmp_path: Path) -> None:
    mesh_graph_path = tmp_path / "tetra_em_mesh_graph.json"
    mesh_graph_path.write_text(
        json.dumps(
            {
                "type": "mesh_graph",
                "source_mesh": "hand-built-tetra-em",
                "node_count": 5,
                "points": [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.25, 0.25, 0.25],
                ],
                "edges": [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [0, 4], [1, 4], [2, 4], [3, 4]],
                "boundary_nodes": [0, 1, 2, 3],
                "cell_blocks": [
                    {"type": "tetra", "cells": [[0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4]]}
                ],
            }
        ),
        encoding="utf-8",
    )
    geometry = GeometrySpec(
        id="geometry:tetra-em-mesh-graph",
        source=GeometrySource(kind="generated"),
        dimension=3,
        encodings=[GeometryEncoding(kind="mesh_graph", uri=str(mesh_graph_path), target_backend="taps")],
    )
    problem = PhysicsProblem(
        id="problem:mesh-tetra-em-curl-curl-3d",
        user_intent={"raw_request": "solve a 3D electromagnetic curl-curl problem on a tetrahedral mesh graph"},
        domain="electromagnetic",
        geometry=geometry,
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
                id="material:tetra-em",
                name="tetra em",
                phase="custom",
                properties=[
                    MaterialProperty(name="relative_permittivity", value=1.5),
                    MaterialProperty(name="relative_permeability", value=1.0),
                    MaterialProperty(name="wave_number", value=0.25),
                    MaterialProperty(name="current_source", value=0.5),
                ],
            )
        ],
        boundary_conditions=[BoundaryConditionSpec(id="bc:e", region_id="boundary", field="E_t", kind="dirichlet", value=0.0)],
        targets=[{"name": "field", "field": "E", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result
    residual = estimate_taps_residual(EstimateTAPSResidualInput(problem=problem, taps_problem=taps_problem, result=result)).report
    operator_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_mesh_fem_em_curl_curl_operator")
    operator_payload = json.loads(open(operator_artifact.uri, encoding="utf-8").read())
    solution_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_mesh_fem_em_field")
    solution_payload = json.loads(open(solution_artifact.uri, encoding="utf-8").read())

    assert result.backend.startswith("taps:mesh_fem_em_curl_curl")
    assert result.status == "success"
    assert residual.converged
    assert result.residuals["fem_tetrahedra"] == 4.0
    assert operator_payload["type"] == "tetra_nedelec_order1_em_curl_curl"
    assert operator_payload["element_shape"] == "tetra"
    assert operator_payload["tetra_count"] == 4
    assert operator_payload["triangle_count"] == 0
    assert operator_payload["basis_order"] == 1
    assert operator_payload["assembly"] == "nedelec_first_kind_edge_element_hcurl"
    assert operator_payload["edge_dof_count"] == operator_payload["dof_count"]
    assert operator_payload["active_boundary_face_count"] >= 1
    assert operator_payload["faces"]
    assert any(element["basis"] == "nedelec_first_kind_order1_tetra" for element in operator_payload["elements"])
    assert len(solution_payload["values"]) == operator_payload["dof_count"]
    assert solution_payload["field_kind"] == "hcurl_edge_field"


def test_tetra_nedelec_order2_scaffold_has_edge_face_and_cell_dofs() -> None:
    points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ]
    stiffness, dofs, faces, total_volume, elements = _assemble_tetra_nedelec_curl_curl(points, [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])

    assert total_volume == pytest.approx(1.0 / 6.0)
    assert len(dofs) == 17
    assert len(faces) == 4
    assert sum(1 for dof in dofs if dof["kind"] == "edge_moment") == 12
    assert sum(1 for dof in dofs if dof["kind"] == "face_tangent") == 4
    assert sum(1 for dof in dofs if dof["kind"] == "cell_interior") == 1
    assert elements[0]["basis"] == "nedelec_first_kind_order2_hierarchical_scaffold_tetra"
    assert len(elements[0]["quadrature"]) == 4
    local = elements[0]["local_matrix"]
    for i in range(len(local)):
        for j in range(len(local)):
            assert local[i][j] == pytest.approx(local[j][i], abs=1e-12)
    assert len(stiffness) == len(dofs)


def test_tetra_raviart_thomas_rt0_div_scaffold_unit_tetra() -> None:
    points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    stiffness, dofs, total_volume, elements = _assemble_tetra_raviart_thomas_div(points, [[0, 1, 2, 3]])

    assert total_volume == pytest.approx(1.0 / 6.0)
    assert len(dofs) == 4
    assert all(dof["kind"] == "face_flux" for dof in dofs)
    assert elements[0]["basis"] == "raviart_thomas_order0_tetra"
    assert len(elements[0]["divergence_basis"]) == 4
    local = elements[0]["local_matrix"]
    for i in range(4):
        for j in range(4):
            assert local[i][j] == pytest.approx(local[j][i], abs=1e-12)
    assert all(row for row in stiffness)


def test_mesh_graph_taps_solves_tetra_hdiv_div_scaffold(tmp_path: Path) -> None:
    mesh_graph_path = tmp_path / "tetra_hdiv_mesh_graph.json"
    mesh_graph_path.write_text(
        json.dumps(
            {
                "type": "mesh_graph",
                "source_mesh": "hand-built-hdiv-tetra",
                "node_count": 4,
                "points": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "edges": [[0, 1], [1, 2], [0, 2], [0, 3], [1, 3], [2, 3]],
                "boundary_nodes": [0, 1, 2, 3],
                "cell_blocks": [{"type": "tetra", "cells": [[0, 1, 2, 3]]}],
            }
        ),
        encoding="utf-8",
    )
    geometry = GeometrySpec(
        id="geometry:tetra-hdiv-mesh-graph",
        source=GeometrySource(kind="generated"),
        dimension=3,
        encodings=[GeometryEncoding(kind="mesh_graph", uri=str(mesh_graph_path), target_backend="taps")],
    )
    problem = PhysicsProblem(
        id="problem:mesh-tetra-hdiv-darcy-3d",
        user_intent={"raw_request": "solve a 3D Darcy flux problem on a tetrahedral mesh"},
        domain="custom",
        geometry=geometry,
        fields=[FieldSpec(name="q", kind="vector")],
        operators=[
            OperatorSpec(
                id="operator:darcy",
                name="Darcy mixed flux",
                domain="custom",
                equation_class="darcy",
                form="weak",
                fields_out=["q"],
            )
        ],
        materials=[MaterialSpec(id="material:porous", name="porous", phase="custom", properties=[MaterialProperty(name="permeability", value=2.0)])],
        boundary_conditions=[BoundaryConditionSpec(id="bc:flux", region_id="boundary", field="q", kind="neumann", value=0.0)],
        targets=[{"name": "flux", "field": "q", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    numerical_plan = plan_numerical_solve(NumericalSolvePlanInput(problem=problem, taps_problem=taps_problem))
    validation = validate_numerical_solve_plan(
        ValidateNumericalSolvePlanInput(problem=problem, taps_problem=taps_problem, plan=numerical_plan)
    )
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem, numerical_plan=numerical_plan)).result

    assert numerical_plan.solver_family == "mesh_fem_hdiv_div"
    assert validation.valid
    assert result.backend.startswith("taps:mesh_fem_hdiv_div")
    assert result.status == "success"
    assert result.residuals["fem_tetrahedra"] == 1.0
    assert result.residuals["fem_face_dofs"] == 4.0
    assert result.residuals["hdiv_scaffold"] == 1.0
    operator_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_mesh_fem_hdiv_operator")
    operator_payload = json.loads(Path(operator_artifact.uri).read_text(encoding="utf-8"))
    assert operator_payload["type"] == "tetra_raviart_thomas_order0_hdiv_div_scaffold"
    assert operator_payload["hdiv_scaffold"]["status"] == "raviart_thomas_order0_tetra_scaffold"
    assert operator_payload["material"]["permeability"] == pytest.approx(2.0)


def test_mesh_graph_taps_solves_tetra10_em_curl_curl_order2_scaffold(tmp_path: Path) -> None:
    mesh_graph_path = tmp_path / "tetra10_em_mesh_graph.json"
    mesh_graph_path.write_text(
        json.dumps(
            {
                "type": "mesh_graph",
                "source_mesh": "hand-built-tetra10-em",
                "node_count": 10,
                "points": [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.5, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.0, 0.5, 0.0],
                    [0.0, 0.0, 0.5],
                    [0.5, 0.0, 0.5],
                    [0.0, 0.5, 0.5],
                ],
                "edges": [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
                "boundary_nodes": [0, 1, 2, 3],
                "cell_blocks": [{"type": "tetra10", "cells": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]}],
            }
        ),
        encoding="utf-8",
    )
    geometry = GeometrySpec(
        id="geometry:tetra10-em-mesh-graph",
        source=GeometrySource(kind="generated"),
        dimension=3,
        encodings=[GeometryEncoding(kind="mesh_graph", uri=str(mesh_graph_path), target_backend="taps")],
    )
    problem = PhysicsProblem(
        id="problem:mesh-tetra10-em-curl-curl-3d",
        user_intent={"raw_request": "solve a second-order 3D electromagnetic curl-curl problem on a tetrahedral mesh graph"},
        domain="electromagnetic",
        geometry=geometry,
        fields=[FieldSpec(name="E", kind="vector")],
        operators=[OperatorSpec(id="operator:maxwell", name="Maxwell curl-curl", domain="electromagnetic", equation_class="maxwell", form="weak", fields_out=["E"])],
        materials=[
            MaterialSpec(
                id="material:tetra10-em",
                name="tetra10 em",
                phase="custom",
                properties=[
                    MaterialProperty(name="relative_permittivity", value=1.2),
                    MaterialProperty(name="relative_permeability", value=1.0),
                    MaterialProperty(name="wave_number", value=0.2),
                    MaterialProperty(name="current_source", value=0.25),
                ],
            )
        ],
        boundary_conditions=[BoundaryConditionSpec(id="bc:e", region_id="boundary", field="E_t", kind="dirichlet", value=0.0)],
        targets=[{"name": "field", "field": "E", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result
    operator_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_mesh_fem_em_curl_curl_operator")
    operator_payload = json.loads(Path(operator_artifact.uri).read_text(encoding="utf-8"))

    assert result.status == "success"
    assert result.residuals["fem_basis_order"] == 2.0
    assert operator_payload["type"] == "tetra_nedelec_order2_em_curl_curl"
    assert operator_payload["element_shape"] == "tetra"
    assert operator_payload["face_tangent_dof_count"] == 4
    assert operator_payload["cell_interior_dof_count"] == 1
    assert operator_payload["hcurl_scaffold"]["face_tangent_dofs"] == 4
    assert any(element["basis"] == "nedelec_first_kind_order2_hierarchical_scaffold_tetra" for element in operator_payload["elements"])


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


def test_cli_applies_confirmed_boundary_labels_with_roles(tmp_path, capsys) -> None:
    geometry_path = tmp_path / "geometry.json"
    labeling_path = tmp_path / "boundary_labeling_artifact.json"
    output_path = tmp_path / "geometry.confirmed.json"
    geometry = GeometrySpec(
        id="geometry:cli-labels",
        source=GeometrySource(kind="mesh_file", uri="box.msh"),
        dimension=3,
    )
    geometry_path.write_text(geometry.model_dump_json(indent=2), encoding="utf-8")
    labeling_path.write_text(
        json.dumps(
            {
                "schema_version": "physicsos.boundary_labeling.v1",
                "geometry_id": geometry.id,
                "confirmed_boundary_labels": [
                    {
                        "target_ids": ["face:left"],
                        "boundary_id": "boundary:left",
                        "label": "left end",
                        "kind": "surface",
                        "role": "x_min",
                        "confidence": 1.0,
                        "confirmed_by": "user",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    code = cli_main(["geometry", "apply-boundary-labels", str(geometry_path), str(labeling_path), "--output", str(output_path)])
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    confirmed = GeometrySpec.model_validate_json(output_path.read_text(encoding="utf-8"))

    assert code == 0
    assert payload["applied"] == ["boundary:left"]
    assert payload["boundary_count"] == 1
    assert confirmed.boundaries[0].role == "x_min"
    assert confirmed.boundaries[0].entity_ids == ["face:left"]


def test_cli_resumes_workflow_with_confirmed_geometry(tmp_path, capsys) -> None:
    problem = build_default_thermal_problem()
    confirmed_geometry = problem.geometry.model_copy(
        update={
            "id": "geometry:confirmed-workflow",
            "boundaries": [
                BoundaryRegionSpec(id="boundary:x_min", label="left", kind="surface", role="x_min", confidence=1.0),
                BoundaryRegionSpec(id="boundary:x_max", label="right", kind="surface", role="x_max", confidence=1.0),
            ],
        }
    )
    problem_path = tmp_path / "problem.json"
    geometry_path = tmp_path / "geometry.confirmed.json"
    output_path = tmp_path / "workflow_result.json"
    problem_path.write_text(problem.model_dump_json(indent=2), encoding="utf-8")
    geometry_path.write_text(confirmed_geometry.model_dump_json(indent=2), encoding="utf-8")

    code = cli_main(
        [
            "workflow",
            "resume-confirmed-geometry",
            str(problem_path),
            str(geometry_path),
            "--output",
            str(output_path),
            "--taps-max-wall-time-seconds",
            "30",
        ]
    )
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)

    assert code == 0
    assert output_path.exists()
    assert payload["geometry_id"] == "geometry:confirmed-workflow"
    assert payload["verification_status"] in {"accepted", "accepted_with_warnings"}
    assert any(step["name"] == "taps-agent.solve" for step in payload["trace"])


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


def test_tetra_p1_elasticity_element_has_3d_rigid_body_modes() -> None:
    points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    stiffness, lumped_mass, total_volume, elements = _assemble_tetra_elasticity_stiffness(
        points=points,
        tetrahedra=[[0, 1, 2, 3]],
        young_modulus=2.0,
        poisson_ratio=0.25,
    )
    assert total_volume == pytest.approx(1.0 / 6.0)
    assert lumped_mass == pytest.approx([1.0 / 24.0] * 4)
    assert len(stiffness) == 12
    assert elements[0]["basis"] == "p1_vector_tetra"
    local = elements[0]["local_stiffness"]
    assert len(local) == 12
    for i in range(12):
        for j in range(12):
            assert local[i][j] == pytest.approx(local[j][i], abs=1e-12)

    modes: list[list[float]] = []
    for axis in range(3):
        mode = []
        for _ in points:
            mode.extend([1.0 if component == axis else 0.0 for component in range(3)])
        modes.append(mode)
    rotation_x = []
    rotation_y = []
    rotation_z = []
    for x, y, z in points:
        rotation_x.extend([0.0, -z, y])
        rotation_y.extend([z, 0.0, -x])
        rotation_z.extend([-y, x, 0.0])
    modes.extend([rotation_x, rotation_y, rotation_z])
    for mode in modes:
        internal_force = [sum(local[i][j] * mode[j] for j in range(12)) for i in range(12)]
        assert internal_force == pytest.approx([0.0] * 12, abs=1e-12)


def test_tetra_p2_elasticity_element_has_3d_rigid_body_modes() -> None:
    points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ]
    stiffness, lumped_mass, total_volume, elements = _assemble_tetra_elasticity_stiffness(
        points=points,
        tetrahedra=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
        young_modulus=2.0,
        poisson_ratio=0.25,
    )
    assert total_volume == pytest.approx(1.0 / 6.0)
    assert sum(lumped_mass) == pytest.approx(1.0 / 6.0)
    assert len(stiffness) == 30
    assert elements[0]["basis"] == "p2_vector_tetra"
    assert elements[0]["geometry"] == "isoparametric_quadratic_tetra"
    local = elements[0]["local_stiffness"]
    assert len(local) == 30
    for i in range(30):
        for j in range(30):
            assert local[i][j] == pytest.approx(local[j][i], abs=1e-10)

    modes: list[list[float]] = []
    for axis in range(3):
        mode = []
        for _ in points:
            mode.extend([1.0 if component == axis else 0.0 for component in range(3)])
        modes.append(mode)
    rotation_x = []
    rotation_y = []
    rotation_z = []
    for x, y, z in points:
        rotation_x.extend([0.0, -z, y])
        rotation_y.extend([z, 0.0, -x])
        rotation_z.extend([-y, x, 0.0])
    modes.extend([rotation_x, rotation_y, rotation_z])
    for mode in modes:
        internal_force = [sum(local[i][j] * mode[j] for j in range(30)) for i in range(30)]
        assert internal_force == pytest.approx([0.0] * 30, abs=1e-9)


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
    assert result.scalar_outputs["numerical_plan_solver_family"] == "mesh_fem_linear_elasticity"
    assert residual.converged
    assert result.residuals["fem_dofs"] == pytest.approx(2.0 * result.residuals["fem_nodes"])
    assert {artifact.kind for artifact in result.artifacts} >= {
        "taps_mesh_fem_elasticity_operator",
        "taps_mesh_fem_displacement_field",
        "taps_iteration_history",
    }
    operator_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_mesh_fem_elasticity_operator")
    operator_payload = json.loads(open(operator_artifact.uri, encoding="utf-8").read())
    assert operator_payload["numerical_plan_solver_family"] == "mesh_fem_linear_elasticity"
    assert operator_payload["material"]["young_modulus"] == pytest.approx(12.0)
    assert operator_payload["material"]["poisson_ratio"] == pytest.approx(0.25)
    assert operator_payload["material"]["constitutive_model"] == "plane_strain"
    assert operator_payload["material"]["body_force"] == pytest.approx([0.0, -2.0])
    assert operator_payload["coefficient_values_applied"]["young_modulus"] == pytest.approx(12.0)
    assert operator_payload["solver_controls_applied"]["max_iterations"] == 20000


def test_mesh_graph_taps_executes_custom_vector_elasticity_weak_form_ir() -> None:
    geometry = GeometrySpec(id="geometry:custom-vector-elasticity-square", source=GeometrySource(kind="generated"), dimension=2)
    mesh = generate_mesh(
        GenerateMeshInput(
            geometry=geometry,
            physics=PhysicsSpec(domains=["solid"]),
            mesh_policy=MeshPolicy(target_element_size=0.35),
            target_backends=["taps"],
        )
    ).mesh
    encoding_output = generate_geometry_encoding(
        GenerateGeometryEncodingInput(geometry=geometry, mesh=mesh, encodings=["mesh_graph"])
    )
    geometry.encodings.extend(encoding_output.encodings)
    problem = PhysicsProblem(
        id="problem:custom-vector-elasticity-2d",
        user_intent={"raw_request": "solve a custom vector weak form using epsilon(v)^T C epsilon(u) and body force"},
        domain="custom",
        geometry=geometry,
        mesh=mesh,
        fields=[FieldSpec(name="u", kind="vector")],
        operators=[
            OperatorSpec(
                id="operator:custom-vector-elasticity",
                name="Custom vector elasticity weak form",
                domain="custom",
                equation_class="custom",
                form="weak",
                fields_out=["u"],
                differential_terms=[
                    {
                        "expression": "int_Omega epsilon(v)^T C(E, nu) epsilon(u) dOmega",
                        "order": 2,
                        "fields": ["u"],
                    }
                ],
                source_terms=[{"expression": "unit_gravity_y", "units": "N/m^3"}],
            )
        ],
        materials=[
            MaterialSpec(
                id="material:custom-vector-solid",
                name="custom vector solid",
                phase="solid",
                properties=[
                    MaterialProperty(name="young_modulus", value=10.0),
                    MaterialProperty(name="poisson_ratio", value=0.2),
                    MaterialProperty(name="constitutive_model", value="plane_stress"),
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
    assert result.status == "success"
    assert result.backend == "taps:weak_ir_linear_elasticity:custom"
    assert result.scalar_outputs["weak_form_ir_blocks"] == 1.0
    operator_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_mesh_fem_elasticity_operator")
    operator_payload = json.loads(open(operator_artifact.uri, encoding="utf-8").read())
    assert operator_payload["weak_form_blocks"]["operator_family"] == "vector_linear_elasticity"
    assert {block["role"] for block in operator_payload["weak_form_blocks"]["blocks"]} >= {"strain_energy", "body_force"}


def test_mesh_graph_taps_solves_tetra_linear_elasticity(tmp_path: Path) -> None:
    mesh_graph_path = tmp_path / "tetra_elasticity_mesh_graph.json"
    mesh_graph_path.write_text(
        json.dumps(
            {
                "type": "mesh_graph",
                "source_mesh": "hand-built-tetra-star",
                "node_count": 5,
                "points": [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.25, 0.25, 0.25],
                ],
                "edges": [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [0, 4], [1, 4], [2, 4], [3, 4]],
                "boundary_nodes": [0, 1, 2, 3],
                "cell_blocks": [
                    {
                        "type": "tetra",
                        "cells": [[0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    geometry = GeometrySpec(
        id="geometry:tetra-elasticity-mesh-graph",
        source=GeometrySource(kind="generated"),
        dimension=3,
        encodings=[GeometryEncoding(kind="mesh_graph", uri=str(mesh_graph_path), target_backend="taps")],
    )
    problem = PhysicsProblem(
        id="problem:mesh-tetra-elasticity-3d",
        user_intent={"raw_request": "solve 3D small-strain linear elasticity on a tetrahedral mesh graph"},
        domain="solid",
        geometry=geometry,
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
                id="material:tetra-solid",
                name="tetra solid",
                phase="solid",
                properties=[
                    MaterialProperty(name="young_modulus", value=8.0),
                    MaterialProperty(name="poisson_ratio", value=0.25),
                    MaterialProperty(name="body_force", value=[0.0, 0.0, -1.0]),
                ],
            )
        ],
        boundary_conditions=[{"id": "bc:u", "region_id": "boundary", "field": "u", "kind": "dirichlet", "value": [0.0, 0.0, 0.0]}],
        targets=[{"name": "displacement", "field": "u", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result

    assert result.backend.startswith("taps:mesh_fem_linear_elasticity")
    assert result.status == "success"
    assert result.residuals["fem_tetrahedra"] == 4
    assert result.residuals["fem_triangles"] == 0
    assert result.residuals["fem_dofs"] == pytest.approx(3.0 * result.residuals["fem_nodes"])
    operator_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_mesh_fem_elasticity_operator")
    operator_payload = json.loads(Path(operator_artifact.uri).read_text(encoding="utf-8"))
    assert operator_payload["type"] == "tetra_p1_fem_linear_elasticity"
    assert operator_payload["material"]["constitutive_model"] == "isotropic_3d"
    assert operator_payload["material"]["body_force"] == pytest.approx([0.0, 0.0, -1.0])
    solution_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_mesh_fem_displacement_field")
    solution_payload = json.loads(Path(solution_artifact.uri).read_text(encoding="utf-8"))
    assert solution_payload["components"] == ["ux", "uy", "uz"]
    assert all(len(value) == 3 for value in solution_payload["values"])


def test_mesh_graph_taps_solves_tetra10_linear_elasticity(tmp_path: Path) -> None:
    mesh_graph_path = tmp_path / "tetra10_elasticity_mesh_graph.json"
    mesh_graph_path.write_text(
        json.dumps(
            {
                "type": "mesh_graph",
                "source_mesh": "hand-built-tetra10",
                "node_count": 10,
                "points": [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.5, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.0, 0.5, 0.0],
                    [0.0, 0.0, 0.5],
                    [0.5, 0.0, 0.5],
                    [0.0, 0.5, 0.5],
                ],
                "edges": [[0, 1], [1, 2], [0, 2], [0, 3], [1, 3], [2, 3]],
                "boundary_nodes": [0, 1, 2, 3],
                "cell_blocks": [{"type": "tetra10", "cells": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]}],
            }
        ),
        encoding="utf-8",
    )
    geometry = GeometrySpec(
        id="geometry:tetra10-elasticity-mesh-graph",
        source=GeometrySource(kind="generated"),
        dimension=3,
        encodings=[GeometryEncoding(kind="mesh_graph", uri=str(mesh_graph_path), target_backend="taps")],
    )
    problem = PhysicsProblem(
        id="problem:mesh-tetra10-elasticity-3d",
        user_intent={"raw_request": "solve second-order 3D small-strain linear elasticity on a tetrahedral mesh graph"},
        domain="solid",
        geometry=geometry,
        fields=[FieldSpec(name="u", kind="vector")],
        operators=[OperatorSpec(id="operator:elasticity", name="Linear elasticity", domain="solid", equation_class="linear_elasticity", form="weak", fields_out=["u"])],
        materials=[
            MaterialSpec(
                id="material:tetra10-solid",
                name="tetra10 solid",
                phase="solid",
                properties=[
                    MaterialProperty(name="young_modulus", value=8.0),
                    MaterialProperty(name="poisson_ratio", value=0.25),
                    MaterialProperty(name="body_force", value=[0.0, 0.0, -1.0]),
                ],
            )
        ],
        boundary_conditions=[{"id": "bc:u", "region_id": "boundary", "field": "u", "kind": "dirichlet", "value": [0.0, 0.0, 0.0]}],
        targets=[{"name": "displacement", "field": "u", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result

    assert result.backend.startswith("taps:mesh_fem_linear_elasticity")
    assert result.status == "success"
    assert result.residuals["fem_tetrahedra"] == 1
    assert result.residuals["fem_basis_order"] == 2
    assert result.residuals["fem_dofs"] == pytest.approx(30.0)
    operator_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_mesh_fem_elasticity_operator")
    operator_payload = json.loads(Path(operator_artifact.uri).read_text(encoding="utf-8"))
    assert operator_payload["type"] == "tetra_p2_fem_linear_elasticity"
    assert any(element["basis"] == "p2_vector_tetra" for element in operator_payload["elements"])


def test_mesh_graph_tetra_elasticity_applies_nonzero_dirichlet_roles(tmp_path: Path) -> None:
    mesh_graph_path = tmp_path / "tetra_elasticity_bc_mesh_graph.json"
    mesh_graph_path.write_text(
        json.dumps(
            {
                "type": "mesh_graph",
                "source_mesh": "hand-built-tetra-star-bc",
                "node_count": 5,
                "points": [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.25, 0.25, 0.25],
                ],
                "edges": [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [0, 4], [1, 4], [2, 4], [3, 4]],
                "boundary_nodes": [0, 1, 2, 3],
                "boundary_node_sets": {"x_min": [0, 2, 3], "x_max": [1]},
                "cell_blocks": [
                    {"type": "tetra", "cells": [[0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4]]}
                ],
            }
        ),
        encoding="utf-8",
    )
    geometry = GeometrySpec(
        id="geometry:tetra-elasticity-bc-mesh-graph",
        source=GeometrySource(kind="generated"),
        dimension=3,
        encodings=[GeometryEncoding(kind="mesh_graph", uri=str(mesh_graph_path), target_backend="taps")],
    )
    problem = PhysicsProblem(
        id="problem:mesh-tetra-elasticity-bc-3d",
        user_intent={"raw_request": "solve 3D elasticity with nonzero displacement on x_max"},
        domain="solid",
        geometry=geometry,
        fields=[FieldSpec(name="u", kind="vector")],
        operators=[OperatorSpec(id="operator:elasticity", name="Linear elasticity", domain="solid", equation_class="linear_elasticity", form="weak", fields_out=["u"])],
        materials=[MaterialSpec(id="material:bc-solid", name="bc solid", phase="solid", properties=[MaterialProperty(name="young_modulus", value=8.0), MaterialProperty(name="poisson_ratio", value=0.25)])],
        boundary_conditions=[
            BoundaryConditionSpec(id="bc:xmin", region_id="x_min", boundary_role="x_min", field="u", kind="dirichlet", value=[0.0, 0.0, 0.0]),
            BoundaryConditionSpec(id="bc:xmax", region_id="x_max", boundary_role="x_max", field="u", kind="dirichlet", value=[0.1, 0.0, 0.0]),
        ],
        targets=[{"name": "displacement", "field": "u", "objective": "observe"}],
        provenance=Provenance(created_by="test"),
    )
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    numerical_plan = plan_numerical_solve(NumericalSolvePlanInput(problem=problem, taps_problem=taps_problem))
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem, numerical_plan=numerical_plan)).result

    assert result.status == "success"
    solution_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "taps_mesh_fem_displacement_field")
    solution_payload = json.loads(Path(solution_artifact.uri).read_text(encoding="utf-8"))
    assert solution_payload["values"][0] == pytest.approx([0.0, 0.0, 0.0])
    assert solution_payload["values"][2] == pytest.approx([0.0, 0.0, 0.0])
    assert solution_payload["values"][3] == pytest.approx([0.0, 0.0, 0.0])
    assert solution_payload["values"][1] == pytest.approx([0.1, 0.0, 0.0])
    assert solution_payload["boundary_values_applied"]["1"] == pytest.approx([0.1, 0.0, 0.0])


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
