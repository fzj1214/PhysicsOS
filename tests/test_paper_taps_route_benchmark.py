from __future__ import annotations

import json
from pathlib import Path

from physicsos.tools.case_tools import (
    BuildTAPSDerivationPromptInput,
    CreateCaseWorkspaceInput,
    LoadTAPSCaseReferencesInput,
    build_taps_derivation_prompt,
    create_case_workspace,
    load_taps_case_references,
)
from physicsos.tools.geometry_embedding_tools import (
    BuildGeometryEmbeddingInput,
    GenerateBackgroundGridInput,
    GenerateGmshDistanceFieldInput,
    ImportSTLGeometryInput,
    VoxelizeGeometryInput,
    build_geometry_embedding,
    generate_background_grid,
    generate_gmsh_distance_field,
    import_stl_geometry,
    voxelize_geometry,
)
from physicsos.tools.taps_tools import (
    CompileTAPSKernelInput,
    ExecuteTAPSKernelInput,
    StaticCheckGeneratedKernelInput,
    compile_taps_kernel,
    execute_taps_kernel,
    static_check_generated_kernel,
)
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


def test_small_3d_stl_paper_taps_route_builds_prompt_package(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(tmp_path))
    stl_path = tmp_path / "tetra.stl"
    stl_path.write_text(ASCII_TETRA_STL, encoding="utf-8")

    case = create_case_workspace(
        CreateCaseWorkspaceInput(
            case_id="paper-route-benchmark",
            user_request="solve Poisson diffusion on an STL tetrahedron with immersed-boundary TAPS",
        )
    )

    problem_dir = tmp_path / "cases" / case.case_id / "problem"
    problem_dir.mkdir(parents=True, exist_ok=True)
    (problem_dir / "problem_statement.md").write_text(
        "\n".join(
            [
                "# Problem Statement",
                "",
                "Solve `-div(chi grad u) = chi` on the STL tetrahedron.",
                "Use zero Dirichlet boundary constraints on the immersed boundary.",
                "Represent the geometry through `phi(x)` and `chi(x)=H(-phi(x))` on a Cartesian background grid.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    imported = import_stl_geometry(ImportSTLGeometryInput(case_id=case.case_id, source_uri="/workspace/tetra.stl"))
    generate_gmsh_distance_field(GenerateGmshDistanceFieldInput(case_id=case.case_id))
    generate_background_grid(
        GenerateBackgroundGridInput(
            case_id=case.case_id,
            bounds_min=imported.summary.bounds_min,
            bounds_max=imported.summary.bounds_max,
            resolution=[6, 6, 6],
        )
    )
    voxelize_geometry(VoxelizeGeometryInput(case_id=case.case_id))
    build_geometry_embedding(BuildGeometryEmbeddingInput(case_id=case.case_id, boundary_constraint_policy="penalty"))

    load_taps_case_references(LoadTAPSCaseReferencesInput(case_id=case.case_id))
    build_taps_derivation_prompt(BuildTAPSDerivationPromptInput(case_id=case.case_id))

    taps_dir = tmp_path / "cases" / case.case_id / "taps"
    (taps_dir / "derivation.md").write_text(
        "\n".join(
            [
                "# TAPS Derivation",
                "",
                "Weak form: integrate `chi(x) grad(v) . grad(u)` over `Omega_bg`.",
                "The STL boundary is represented by `phi(x)=0`; `chi(x)=H(-phi(x))` activates the physical domain.",
                "Penalty boundary terms impose zero Dirichlet data on immersed boundary samples.",
                "Cut-cell quadrature candidates are read from `cut_cells.npy`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (taps_dir / "implementation_notes.md").write_text(
        "Generated code should load SDF, occupancy, boundary samples, normals, and cut cells as case-local inputs.\n",
        encoding="utf-8",
    )

    compiled = compile_taps_kernel(
        CompileTAPSKernelInput(case_id=case.case_id, max_iterations=60, tolerance=1e-5, boundary_penalty=13.0)
    )
    checked = static_check_generated_kernel(StaticCheckGeneratedKernelInput(case_id=case.case_id))
    executed = execute_taps_kernel(ExecuteTAPSKernelInput(case_id=case.case_id, timeout_seconds=30))

    exact_code = generate_exact_sol_code(GenerateExactSolCodeInput(case_id=case.case_id, dimension=3))
    exact = execute_exact_sol_code(ExecuteExactSolCodeInput(case_id=case.case_id))
    convergence_code = generate_convergence_code(GenerateConvergenceCodeInput(case_id=case.case_id, refinement_levels=[6, 8, 10]))
    convergence = execute_convergence_code(ExecuteConvergenceCodeInput(case_id=case.case_id))
    plotted = plot_result(PlotResultInput(case_id=case.case_id))

    assert compiled.implementation_manifest.uri == f"/workspace/cases/{case.case_id}/taps/implementation_manifest.json"
    assert checked.passes
    assert not executed.passes
    assert exact_code.warnings == []
    assert exact.passes
    assert convergence_code.warnings == []
    assert convergence.passes
    assert plotted.report_json.uri == f"/workspace/cases/{case.case_id}/verification/report.json"

    case_dir = tmp_path / "cases" / case.case_id
    manifest = json.loads((case_dir / "taps" / "implementation_manifest.json").read_text(encoding="utf-8"))
    kernel_log = json.loads((case_dir / "taps" / "kernel_execution_log.json").read_text(encoding="utf-8"))
    verification = json.loads((case_dir / "verification" / "report.json").read_text(encoding="utf-8"))

    assert manifest["route"] == "paper_prompt_engineering_case_kernel"
    assert "one-shot TAPS derivation reference" in "\n".join(manifest["implementation_agent_tasks"])
    assert kernel_log["returncode"] != 0
    assert "not a built-in numerical solver" in kernel_log["stdout"]
    assert verification["status"] == "accepted"
