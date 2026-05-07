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
from physicsos.tools.taps_tools import (
    CompileTAPSKernelInput,
    ExecuteTAPSKernelInput,
    ReviewGeneratedTAPSKernelInput,
    StaticCheckGeneratedKernelInput,
    compile_taps_kernel,
    execute_taps_kernel,
    review_generated_taps_kernel,
    static_check_generated_kernel,
)
from tests.test_taps_case_geometry_tools import ASCII_TETRA_STL


def test_compile_taps_kernel_creates_prompt_package_not_fixed_solver(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(tmp_path))
    stl_path = tmp_path / "tetra.stl"
    stl_path.write_text(ASCII_TETRA_STL, encoding="utf-8")

    case = create_case_workspace(CreateCaseWorkspaceInput(case_id="kernel-case"))
    imported = import_stl_geometry(ImportSTLGeometryInput(case_id=case.case_id, source_uri="/workspace/tetra.stl"))
    generate_background_grid(
        GenerateBackgroundGridInput(
            case_id=case.case_id,
            bounds_min=imported.summary.bounds_min,
            bounds_max=imported.summary.bounds_max,
            resolution=[6, 6, 6],
        )
    )
    voxelize_geometry(VoxelizeGeometryInput(case_id=case.case_id))
    build_geometry_embedding(BuildGeometryEmbeddingInput(case_id=case.case_id))
    taps_dir = tmp_path / "cases" / "kernel-case" / "taps"
    taps_dir.mkdir(parents=True, exist_ok=True)
    (taps_dir / "derivation.md").write_text("Weak form includes chi(x), phi(x), penalty boundary terms, and cut-cell quadrature.", encoding="utf-8")
    (taps_dir / "implementation_notes.md").write_text("Use occupancy, SDF, boundary samples, normals, and penalty enforcement.", encoding="utf-8")

    compiled = compile_taps_kernel(CompileTAPSKernelInput(case_id=case.case_id, max_iterations=40, tolerance=1e-5, boundary_penalty=11.0))
    static_check = static_check_generated_kernel(StaticCheckGeneratedKernelInput(case_id=case.case_id))
    review = review_generated_taps_kernel(ReviewGeneratedTAPSKernelInput(case_id=case.case_id))
    executed = execute_taps_kernel(ExecuteTAPSKernelInput(case_id=case.case_id, timeout_seconds=30))

    assert compiled.kernel.uri == "/workspace/cases/kernel-case/taps/kernel.py"
    assert compiled.implementation_manifest.uri == "/workspace/cases/kernel-case/taps/implementation_manifest.json"
    assert static_check.passes
    assert not review.passes
    assert "scaffold_replaced" in review.missing_requirements
    assert not executed.passes

    kernel_log = json.loads((tmp_path / "cases" / "kernel-case" / "taps" / "kernel_execution_log.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "cases" / "kernel-case" / "taps" / "implementation_manifest.json").read_text(encoding="utf-8"))
    prompt_text = (tmp_path / "cases" / "kernel-case" / "taps" / "implementation_prompt.md").read_text(encoding="utf-8")
    review_spec = json.loads((tmp_path / "cases" / "kernel-case" / "taps" / "kernel_review_spec.json").read_text(encoding="utf-8"))

    assert kernel_log["returncode"] != 0
    assert "intentionally not a built-in numerical solver" in kernel_log["stdout"]
    assert manifest["not_ir"] is True
    assert manifest["source_artifacts"]["context_window"]["path"] == "/workspace/cases/kernel-case/context/context_window.md"
    assert manifest["source_artifacts"]["geometry_handoff"]["path"] == "/workspace/cases/kernel-case/geometry/taps_geometry_handoff.md"
    assert manifest["detected_derivation_features"]["geometry_characteristic_terms"] is True
    assert manifest["detected_derivation_features"]["boundary_constraint_terms"] is True
    assert review_spec["schema_version"] == "physicsos.taps_kernel_review_spec.v1"
    assert "Context window" in prompt_text
    assert "Geometry handoff" in prompt_text
    assert "one-shot TAPS reference" in prompt_text
    assert "do not develop the new code from scratch" in prompt_text
    assert "taps_verification_workflow.md" in prompt_text
    assert not (tmp_path / "cases" / "kernel-case" / "taps" / "solution.npy").exists()


def test_execute_taps_kernel_runs_agent_generated_case_local_code(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(tmp_path))
    case = create_case_workspace(CreateCaseWorkspaceInput(case_id="agent-generated-kernel"))
    taps_dir = tmp_path / "cases" / case.case_id / "taps"
    taps_dir.mkdir(parents=True, exist_ok=True)
    (taps_dir / "kernel.py").write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import json",
                "from pathlib import Path",
                "import numpy as np",
                "",
                "def run_case(config=None):",
                "    case_dir = Path(__file__).resolve().parents[1]",
                "    taps_dir = case_dir / 'taps'",
                "    # Generated TAPS code keeps subspace matrix structure from derivation.md.",
                "    subspace_matrix = np.eye(3)",
                "    u = np.array([0.0, 1.0, 0.0])",
                "    np.save(taps_dir / 'solution.npy', u)",
                "    (taps_dir / 'residual_history.json').write_text(json.dumps([{'iteration': 1, 'relative_update': 0.0}]), encoding='utf-8')",
                "    (taps_dir / 'runtime_metadata.json').write_text(json.dumps({'status': 'success', 'method': 'agent_generated_fixture'}), encoding='utf-8')",
                "    (taps_dir / 'solution_summary.json').write_text(json.dumps({'shape': [3], 'max': 1.0}), encoding='utf-8')",
                "    return {'status': 'success'}",
                "",
                "if __name__ == '__main__':",
                "    print(json.dumps(run_case()))",
                "",
            ]
        ),
        encoding="utf-8",
    )

    review = review_generated_taps_kernel(ReviewGeneratedTAPSKernelInput(case_id=case.case_id))
    executed = execute_taps_kernel(ExecuteTAPSKernelInput(case_id=case.case_id, timeout_seconds=30))

    assert review.passes
    assert executed.passes
    assert executed.result.backend == "paper_taps_case_kernel"
    assert executed.result.residuals["final_relative_update"] == 0.0


def test_taps_implementation_agent_exposes_kernel_tools() -> None:
    tool_names = {tool.__name__ for tool in DEEPAGENTS_SUBAGENT_TOOL_GROUPS["taps-implementation-agent"]}

    assert {"compile_taps_kernel", "static_check_generated_kernel", "review_generated_taps_kernel", "execute_taps_kernel"}.issubset(tool_names)


def test_review_generated_taps_kernel_uses_case_local_spec(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(tmp_path))
    case = create_case_workspace(CreateCaseWorkspaceInput(case_id="custom-review-spec"))
    taps_dir = tmp_path / "cases" / case.case_id / "taps"
    taps_dir.mkdir(parents=True, exist_ok=True)
    (taps_dir / "kernel.py").write_text("def run_case(config=None):\n    return {'status': 'custom'}\n", encoding="utf-8")
    (taps_dir / "kernel_review_spec.json").write_text(
        json.dumps(
            {
                "schema_version": "physicsos.taps_kernel_review_spec.v1",
                "checks": [
                    {
                        "id": "custom_marker",
                        "description": "Case-local spec can require arbitrary generated-code evidence.",
                        "severity": "error",
                        "contains_any": ["CUSTOM_REVIEW_MARKER"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    failed = review_generated_taps_kernel(ReviewGeneratedTAPSKernelInput(case_id=case.case_id))
    assert not failed.passes
    assert failed.missing_requirements == ["custom_marker"]

    (taps_dir / "kernel.py").write_text(
        "def run_case(config=None):\n    CUSTOM_REVIEW_MARKER = True\n    return {'status': 'custom'}\n",
        encoding="utf-8",
    )
    passed = review_generated_taps_kernel(ReviewGeneratedTAPSKernelInput(case_id=case.case_id))
    assert passed.passes
