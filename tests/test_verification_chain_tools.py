from __future__ import annotations

import json
from pathlib import Path

from physicsos.tools.case_tools import CreateCaseWorkspaceInput, create_case_workspace
from physicsos.tools.geometry_embedding_tools import (
    BuildGeometryEmbeddingInput,
    GenerateBackgroundGridInput,
    ImportSTLGeometryInput,
    VoxelizeGeometryInput,
    build_geometry_embedding,
    generate_background_grid,
    import_stl_geometry,
    voxelize_geometry,
)
from physicsos.tools.registry import DEEPAGENTS_SUBAGENT_TOOL_GROUPS
from physicsos.tools.verification_chain_tools import (
    ExecuteConvergenceCodeInput,
    ExecuteExactSolCodeInput,
    GenerateConvergenceCodeInput,
    GenerateExactSolCodeInput,
    PlotResultInput,
    execute_convergence_code,
    execute_exact_sol_code,
    generate_convergence_code,
    generate_exact_sol_code,
    plot_result,
)
from tests.test_taps_case_geometry_tools import ASCII_TETRA_STL


def test_paper_style_verification_chain_generates_artifacts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(tmp_path))
    case = create_case_workspace(CreateCaseWorkspaceInput(case_id="verify-case"))

    exact_code = generate_exact_sol_code(GenerateExactSolCodeInput(case_id=case.case_id, dimension=3))
    exact = execute_exact_sol_code(ExecuteExactSolCodeInput(case_id=case.case_id))
    convergence_code = generate_convergence_code(GenerateConvergenceCodeInput(case_id=case.case_id, refinement_levels=[8, 16, 32]))
    convergence = execute_convergence_code(ExecuteConvergenceCodeInput(case_id=case.case_id))
    plotted = plot_result(PlotResultInput(case_id=case.case_id))

    assert exact_code.warnings == []
    assert convergence_code.warnings == []
    assert exact.passes
    assert convergence.passes
    assert plotted.plot.uri == "/workspace/cases/verify-case/verification/plots/convergence_plot.svg"

    exact_payload = json.loads((tmp_path / "cases" / "verify-case" / "verification" / "exact_solution.json").read_text(encoding="utf-8"))
    convergence_payload = json.loads((tmp_path / "cases" / "verify-case" / "verification" / "convergence_report.json").read_text(encoding="utf-8"))
    report_payload = json.loads((tmp_path / "cases" / "verify-case" / "verification" / "report.json").read_text(encoding="utf-8"))

    assert exact_payload["schema_version"] == "physicsos.exact_solution.v1"
    assert convergence_payload["schema_version"] == "physicsos.convergence_report.v1"
    assert report_payload["status"] == "accepted"
    assert report_payload["checks"] == {
        "exact_solution_code": "executed",
        "convergence_code": "executed",
        "plot_result": "executed",
    }


def test_verification_report_records_geometry_evidence_without_treating_it_as_verification(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(tmp_path))
    stl_path = tmp_path / "tetra.stl"
    stl_path.write_text(ASCII_TETRA_STL, encoding="utf-8")

    case = create_case_workspace(CreateCaseWorkspaceInput(case_id="verify-geometry-case"))
    imported = import_stl_geometry(ImportSTLGeometryInput(case_id=case.case_id, source_uri="/workspace/tetra.stl"))
    generate_background_grid(
        GenerateBackgroundGridInput(
            case_id=case.case_id,
            bounds_min=imported.summary.bounds_min,
            bounds_max=imported.summary.bounds_max,
            resolution=[5, 5, 5],
        )
    )
    voxelize_geometry(VoxelizeGeometryInput(case_id=case.case_id))
    build_geometry_embedding(BuildGeometryEmbeddingInput(case_id=case.case_id))

    generate_exact_sol_code(GenerateExactSolCodeInput(case_id=case.case_id, dimension=3))
    execute_exact_sol_code(ExecuteExactSolCodeInput(case_id=case.case_id))
    generate_convergence_code(GenerateConvergenceCodeInput(case_id=case.case_id, refinement_levels=[8, 16, 32]))
    execute_convergence_code(ExecuteConvergenceCodeInput(case_id=case.case_id))
    plot_result(PlotResultInput(case_id=case.case_id))

    report_payload = json.loads((tmp_path / "cases" / "verify-geometry-case" / "verification" / "report.json").read_text(encoding="utf-8"))
    report_text = (tmp_path / "cases" / "verify-geometry-case" / "verification" / "report.md").read_text(encoding="utf-8")

    assert report_payload["geometry_evidence"]["status"] == "ready"
    assert "handoff" in report_payload["geometry_evidence"]["present_artifacts"]
    assert report_payload["geometry_evidence"]["interpretation"] == "Geometry preprocessing is input evidence for immersed-boundary TAPS; it is not numerical verification."
    assert "Geometry Evidence" in report_text


def test_verification_agent_exposes_paper_style_tool_chain() -> None:
    tool_names = {tool.__name__ for tool in DEEPAGENTS_SUBAGENT_TOOL_GROUPS["verification-agent"]}

    assert {
        "generate_exact_sol_code",
        "execute_exact_sol_code",
        "generate_convergence_code",
        "execute_convergence_code",
        "plot_result",
    }.issubset(tool_names)
