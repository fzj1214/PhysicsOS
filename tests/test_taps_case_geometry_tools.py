from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from physicsos.tools.case_tools import (
    BuildPaperContextWindowInput,
    BuildTAPSDerivationPromptInput,
    CreateCaseWorkspaceInput,
    LoadTAPSCaseReferencesInput,
    UpdateCaseStageStatusInput,
    build_paper_context_window,
    build_taps_derivation_prompt,
    create_case_workspace,
    load_taps_case_references,
    update_case_stage_status,
)
from physicsos.tools.geometry_embedding_tools import (
    BuildGeometryEmbeddingInput,
    ExecuteGmshDistanceFieldInput,
    GenerateBackgroundGridInput,
    GenerateGmshDistanceFieldInput,
    GeneratePrimitiveGeometryInput,
    ImportSTLGeometryInput,
    PrepareGeometryAnalysisFilesInput,
    VoxelizeGeometryInput,
    build_geometry_embedding,
    execute_gmsh_distance_field,
    generate_background_grid,
    generate_gmsh_distance_field,
    generate_primitive_geometry,
    import_stl_geometry,
    prepare_geometry_analysis_files,
    voxelize_geometry,
)
from physicsos.tools.registry import DEEPAGENTS_MAIN_BRIDGE_TOOLS, DEEPAGENTS_SUBAGENT_TOOL_GROUPS


ASCII_TETRA_STL = """solid tetra
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 1 0 0
    vertex 0 1 0
  endloop
endfacet
facet normal 0 -1 0
  outer loop
    vertex 0 0 0
    vertex 0 0 1
    vertex 1 0 0
  endloop
endfacet
facet normal 1 1 1
  outer loop
    vertex 1 0 0
    vertex 0 0 1
    vertex 0 1 0
  endloop
endfacet
facet normal -1 0 0
  outer loop
    vertex 0 0 0
    vertex 0 1 0
    vertex 0 0 1
  endloop
endfacet
endsolid tetra
"""


def test_case_references_and_derivation_prompt(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(tmp_path))

    case = create_case_workspace(CreateCaseWorkspaceInput(case_id="prompt-case", user_request="solve diffusion"))
    references = load_taps_case_references(LoadTAPSCaseReferencesInput(case_id=case.case_id))
    prompt = build_taps_derivation_prompt(BuildTAPSDerivationPromptInput(case_id=case.case_id))

    assert case.case_dir == "/workspace/cases/prompt-case"
    assert len(references.references) == 5
    assert prompt.prompt.uri == "/workspace/cases/prompt-case/taps/derivation_prompt.md"
    assert (tmp_path / "cases" / "prompt-case" / "references" / "taps_template_eq5.md").exists()
    prompt_text = (tmp_path / "cases" / "prompt-case" / "taps" / "derivation_prompt.md").read_text(encoding="utf-8")
    assert "## 1. Role-playing" in prompt_text
    assert "## 2. Few-shot prompt" in prompt_text
    assert "## 3. Constraints" in prompt_text
    assert "## 4. Chain-of-thought derivation requirements" in prompt_text
    assert "## 5. Formatting guidelines" in prompt_text
    assert "/workspace/cases/prompt-case/context/context_window.md" in prompt_text
    assert "Do not jump directly to the final matrix form." in prompt_text


def test_paper_context_window_packages_four_modules(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(tmp_path))

    case = create_case_workspace(CreateCaseWorkspaceInput(case_id="context-case", user_request="solve parametric diffusion"))
    problem_dir = tmp_path / "cases" / case.case_id / "problem"
    (problem_dir / "problem_statement.md").write_text("Solve a parametric diffusion equation with TAPS.\n", encoding="utf-8")
    load_taps_case_references(LoadTAPSCaseReferencesInput(case_id=case.case_id))

    context = build_paper_context_window(
        BuildPaperContextWindowInput(
            case_id=case.case_id,
            user_prompt="solve parametric diffusion",
            include_geometry_embedding=False,
        )
    )

    assert context.context_window.uri == "/workspace/cases/context-case/context/context_window.md"
    assert context.manifest.uri == "/workspace/cases/context-case/context/context_window.json"

    text = (tmp_path / "cases" / "context-case" / "context" / "context_window.md").read_text(encoding="utf-8")
    payload = json.loads((tmp_path / "cases" / "context-case" / "context" / "context_window.json").read_text(encoding="utf-8"))

    assert "analysis files" in payload["modules"]
    assert "tools" in payload["modules"]
    assert "online/local resources" in payload["modules"]
    assert "context window" in payload["modules"]
    assert payload["not_a_workflow_engine"] is True
    assert "Paper Context Window" in text
    assert "/workspace/cases/context-case/references/taps_template_eq5.md" in text
    assert "few-shot/CoT TAPS derivation" in text
    assert "not a fixed numerical solver" in text


def test_main_bridge_exposes_context_window_builder() -> None:
    tool_names = {tool.__name__ for tool in DEEPAGENTS_MAIN_BRIDGE_TOOLS}

    assert "build_paper_context_window" in tool_names


def test_update_case_stage_status_maintains_visible_plan(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(tmp_path))

    case = create_case_workspace(CreateCaseWorkspaceInput(case_id="stage-case"))
    updated = update_case_stage_status(
        UpdateCaseStageStatusInput(
            case_id=case.case_id,
            stage="ANALYSIS_FILES",
            status="done",
            note="problem statement prepared",
        )
    )

    plan_text = (tmp_path / "cases" / "stage-case" / "execution_plan.md").read_text(encoding="utf-8")
    manifest = json.loads((tmp_path / "cases" / "stage-case" / "manifest.json").read_text(encoding="utf-8"))

    assert updated.current_stage == "GEOMETRY_EMBEDDING"
    assert updated.completed_stages == ["ANALYSIS_FILES"]
    assert "- [done] ANALYSIS_FILES" in plan_text
    assert "- [todo] GEOMETRY_EMBEDDING" in plan_text
    assert "`ANALYSIS_FILES` -> `done`: problem statement prepared" in plan_text
    assert manifest["current_stage"] == "GEOMETRY_EMBEDDING"
    assert manifest["stage_status"]["ANALYSIS_FILES"] == "done"


def test_update_case_stage_status_rejects_workspace_as_recoverable_warning(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(tmp_path))

    case = create_case_workspace(CreateCaseWorkspaceInput(case_id="bad-stage-case"))
    updated = update_case_stage_status(
        UpdateCaseStageStatusInput(
            case_id=case.case_id,
            stage="workspace",
            status="done",
            note="workspace created",
        )
    )

    manifest = json.loads((tmp_path / "cases" / "bad-stage-case" / "manifest.json").read_text(encoding="utf-8"))

    assert updated.current_stage == "ANALYSIS_FILES"
    assert updated.completed_stages == []
    assert updated.warnings
    assert "Unknown case stage `workspace`" in updated.warnings[0]
    assert manifest["current_stage"] == "ANALYSIS_FILES"
    assert manifest["stage_status"]["ANALYSIS_FILES"] == "todo"
    assert "workspace" not in manifest["stage_status"]


def test_main_bridge_exposes_stage_status_tool() -> None:
    tool_names = {tool.__name__ for tool in DEEPAGENTS_MAIN_BRIDGE_TOOLS}

    assert "update_case_stage_status" in tool_names


def test_stl_to_geometry_embedding_minimal_route(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(tmp_path))
    stl_path = tmp_path / "tetra.stl"
    stl_path.write_text(ASCII_TETRA_STL, encoding="utf-8")

    case = create_case_workspace(CreateCaseWorkspaceInput(case_id="stl-case"))
    imported = import_stl_geometry(ImportSTLGeometryInput(case_id=case.case_id, source_uri="/workspace/tetra.stl"))
    gmsh = generate_gmsh_distance_field(GenerateGmshDistanceFieldInput(case_id=case.case_id))
    grid = generate_background_grid(
        GenerateBackgroundGridInput(
            case_id=case.case_id,
            bounds_min=imported.summary.bounds_min,
            bounds_max=imported.summary.bounds_max,
            resolution=[5, 5, 5],
        )
    )
    voxelized = voxelize_geometry(VoxelizeGeometryInput(case_id=case.case_id))
    embedding = build_geometry_embedding(BuildGeometryEmbeddingInput(case_id=case.case_id))

    assert imported.summary.triangle_count == 4
    assert gmsh.gmsh_geo.uri == "/workspace/cases/stl-case/geometry/gmsh_model.geo"
    assert grid.background_grid.uri == "/workspace/cases/stl-case/geometry/background_grid.json"
    assert voxelized.sdf.uri == "/workspace/cases/stl-case/geometry/sdf.npy"
    assert voxelized.quality.uri == "/workspace/cases/stl-case/geometry/sdf_quality.json"
    assert embedding.warnings == []
    assert embedding.handoff_notes.uri == "/workspace/cases/stl-case/geometry/taps_geometry_handoff.md"

    sdf = np.load(tmp_path / "cases" / "stl-case" / "geometry" / "sdf.npy")
    occupancy = np.load(tmp_path / "cases" / "stl-case" / "geometry" / "occupancy.npy")
    payload = json.loads((tmp_path / "cases" / "stl-case" / "geometry" / "embedding.json").read_text(encoding="utf-8"))
    quality = json.loads((tmp_path / "cases" / "stl-case" / "geometry" / "sdf_quality.json").read_text(encoding="utf-8"))
    handoff_text = (tmp_path / "cases" / "stl-case" / "geometry" / "taps_geometry_handoff.md").read_text(encoding="utf-8")

    assert sdf.shape == (5, 5, 5)
    assert occupancy.shape == (5, 5, 5)
    assert payload["method"] == "immersed_boundary_ife_taps"
    assert payload["paper_route_role"] == "geometry_analysis_file_extension"
    assert "agent_handoff" in payload
    assert payload["sdf_convention"] == "phi(x) <= 0 is inside the STL domain"
    assert quality["schema_version"] == "physicsos.sdf_quality.v1"
    assert quality["production_ready"] is False
    assert "Derivation-agent handoff" in handoff_text
    assert "Implementation-agent handoff" in handoff_text
    assert "Verification-agent handoff" in handoff_text


def test_gmsh_distance_field_execution_has_logged_fallback(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(tmp_path))
    stl_path = tmp_path / "tetra.stl"
    stl_path.write_text(ASCII_TETRA_STL, encoding="utf-8")

    case = create_case_workspace(CreateCaseWorkspaceInput(case_id="gmsh-sdf-case"))
    imported = import_stl_geometry(ImportSTLGeometryInput(case_id=case.case_id, source_uri="/workspace/tetra.stl"))
    generate_background_grid(
        GenerateBackgroundGridInput(
            case_id=case.case_id,
            bounds_min=imported.summary.bounds_min,
            bounds_max=imported.summary.bounds_max,
            resolution=[4, 4, 4],
        )
    )

    sampled = execute_gmsh_distance_field(
        ExecuteGmshDistanceFieldInput(case_id=case.case_id, timeout_seconds=1, fallback_to_vertex_sdf=True)
    )

    assert sampled.status in {"success", "fallback"}
    assert sampled.sdf.uri == "/workspace/cases/gmsh-sdf-case/geometry/gmsh_sdf.npy"
    assert (tmp_path / "cases" / "gmsh-sdf-case" / "geometry" / "gmsh_distance_execution_log.json").exists()

    sdf = np.load(tmp_path / "cases" / "gmsh-sdf-case" / "geometry" / "gmsh_sdf.npy")
    manifest = json.loads((tmp_path / "cases" / "gmsh-sdf-case" / "geometry" / "gmsh_sampled_sdf.json").read_text(encoding="utf-8"))

    assert sdf.shape == (4, 4, 4)
    assert manifest["schema_version"] == "physicsos.gmsh_sampled_sdf.v1"
    assert manifest["status"] == sampled.status


def test_geometry_embedding_agent_exposes_gmsh_distance_execution() -> None:
    tool_names = {tool.__name__ for tool in DEEPAGENTS_SUBAGENT_TOOL_GROUPS["geometry-embedding-agent"]}

    assert "execute_gmsh_distance_field" in tool_names


def test_natural_language_primitive_geometry_enters_taps_prompt(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(tmp_path))

    case = create_case_workspace(CreateCaseWorkspaceInput(case_id="nl-geometry-case"))
    generated = generate_primitive_geometry(
        GeneratePrimitiveGeometryInput(
            case_id=case.case_id,
            description="a rectangular box 2 by 1 by 0.5 meters",
            primitive="auto",
        )
    )
    generate_background_grid(
        GenerateBackgroundGridInput(
            case_id=case.case_id,
            bounds_min=generated.summary.bounds_min,
            bounds_max=generated.summary.bounds_max,
            resolution=[5, 5, 5],
        )
    )
    voxelize_geometry(VoxelizeGeometryInput(case_id=case.case_id))
    embedding = build_geometry_embedding(BuildGeometryEmbeddingInput(case_id=case.case_id))
    load_taps_case_references(LoadTAPSCaseReferencesInput(case_id=case.case_id))
    prompt = build_taps_derivation_prompt(BuildTAPSDerivationPromptInput(case_id=case.case_id))

    assert generated.stl.uri == "/workspace/cases/nl-geometry-case/geometry/input.stl"
    assert generated.summary.triangle_count == 12
    assert embedding.warnings == []
    assert prompt.prompt.uri == "/workspace/cases/nl-geometry-case/taps/derivation_prompt.md"

    request = json.loads((tmp_path / "cases" / "nl-geometry-case" / "geometry" / "generated_geometry.json").read_text(encoding="utf-8"))
    prompt_text = (tmp_path / "cases" / "nl-geometry-case" / "taps" / "derivation_prompt.md").read_text(encoding="utf-8")

    assert request["primitive"] == "box"
    assert request["parameters"] == {"length": 2.0, "width": 1.0, "height": 0.5}
    assert "Geometry embedding notes" in prompt_text


def test_prepare_geometry_analysis_files_one_step_for_natural_language(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(tmp_path))

    case = create_case_workspace(CreateCaseWorkspaceInput(case_id="geometry-analysis-case"))
    prepared = prepare_geometry_analysis_files(
        PrepareGeometryAnalysisFilesInput(
            case_id=case.case_id,
            description="sphere radius 0.75 meters",
            grid_resolution=[5, 5, 5],
            run_gmsh_sampler=False,
        )
    )
    load_taps_case_references(LoadTAPSCaseReferencesInput(case_id=case.case_id))
    problem_dir = tmp_path / "cases" / case.case_id / "problem"
    (problem_dir / "problem_statement.md").write_text("Solve diffusion on the generated sphere with immersed-boundary TAPS.\n", encoding="utf-8")
    build_paper_context_window(BuildPaperContextWindowInput(case_id=case.case_id, include_geometry_embedding=True))
    prompt = build_taps_derivation_prompt(
        BuildTAPSDerivationPromptInput(
            case_id=case.case_id,
            geometry_embedding_uri=prepared.derivation_context.uri,
        )
    )

    assert prepared.source_mode == "generated_primitive"
    assert prepared.summary.triangle_count > 12
    assert prepared.derivation_context.uri == "/workspace/cases/geometry-analysis-case/geometry/taps_geometry_context.md"
    assert prepared.embedding.uri == "/workspace/cases/geometry-analysis-case/geometry/embedding.json"
    assert (tmp_path / "cases" / "geometry-analysis-case" / "geometry" / "taps_geometry_handoff.md").exists()
    assert (tmp_path / "cases" / "geometry-analysis-case" / "geometry" / "occupancy.npy").exists()

    context_text = (tmp_path / "cases" / "geometry-analysis-case" / "geometry" / "taps_geometry_context.md").read_text(encoding="utf-8")
    prompt_text = (tmp_path / "cases" / "geometry-analysis-case" / "taps" / "derivation_prompt.md").read_text(encoding="utf-8")

    assert "This file is an analysis-file input" in context_text
    assert "PhysicsOS geometry extension to the paper route only" in context_text
    assert "does not define a separate solver workflow" in context_text
    assert "chi(x)=H(-phi(x))" in context_text
    assert "taps_geometry_handoff.md" in (tmp_path / "cases" / "geometry-analysis-case" / "context" / "context_window.md").read_text(encoding="utf-8")
    assert prepared.derivation_context.uri in prompt_text
    assert prompt.warnings == []


def test_geometry_embedding_agent_exposes_one_step_analysis_tool() -> None:
    tool_names = {tool.__name__ for tool in DEEPAGENTS_SUBAGENT_TOOL_GROUPS["geometry-embedding-agent"]}

    assert "prepare_geometry_analysis_files" in tool_names


def test_geometry_tool_does_not_hard_code_composite_natural_language_parser() -> None:
    import inspect
    import physicsos.tools.geometry_embedding_tools as module

    source = inspect.getsource(module)

    assert "_parse_composite_parameters" not in source
    assert "box_with_cylindrical_hole" not in source
    assert '"composite"' not in source
