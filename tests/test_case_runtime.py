from __future__ import annotations

import json
from pathlib import Path

import meshio
import numpy as np
import pytest

from physicsos.runtime import CaseRuntime
from physicsos.runtime.artifacts import checked_path
from physicsos.schemas.case_runtime import ConvergenceStudyInput, DomainRequirements, ExecuteCaseInput, PrepareDomainInput, VerifyCaseInput
from physicsos.schemas.geometry import GeometryEntity, GeometrySource, GeometrySpec, RegionSpec
from physicsos.schemas.mesh import MeshPolicy
from physicsos.verification.base import VerificationStatus


# A real P1 FEM kernel used only as a regression fixture. The runtime contains
# no PDE-specific assembly code and can execute other generated kernels.
FEM_KERNEL = '''from pathlib import Path
import json
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

def run_case(config):
    with np.load(config["domain_artifacts"]["mesh_arrays"]) as mesh:
        points, cells = mesh["points"], mesh["tetra"]
    with np.load(config["domain_artifacts"]["boundary_nodes"]) as boundary:
        fixed = boundary["wall"]
    k = config["controls"].get("k", 1.0)
    rows, cols, entries = [], [], []
    rhs = np.zeros(len(points))
    for cell in cells:
        xyz = points[cell]
        volume = abs(np.linalg.det(xyz[1:] - xyz[0])) / 6
        gradients = np.linalg.inv(np.column_stack([np.ones(4), xyz]))[1:, :]
        local = k * volume * gradients.T @ gradients
        for i, a in enumerate(cell):
            rhs[a] -= 6 * volume / 4
            for j, b in enumerate(cell):
                rows.append(a); cols.append(b); entries.append(local[i, j])
    matrix = coo_matrix((entries, (rows, cols)), shape=(len(points), len(points))).tocsr()
    u = (points * points).sum(axis=1) / k
    free = np.flatnonzero(~fixed)
    constrained = np.flatnonzero(fixed)
    u[free] = spsolve(matrix[free][:, free], rhs[free] - matrix[free][:, constrained] @ u[constrained])
    output = Path(config["output_dir"])
    np.save(output / "solution.npy", u)
    (output / "residual_history.json").write_text(json.dumps([{"residual": float(np.linalg.norm((matrix @ u - rhs)[free]))}]))
    (output / "runtime_metadata.json").write_text(json.dumps({"method": "P1 FEM", "k": k}))
    return {"status": "success"}
'''

REFERENCE = '''import numpy as np
def exact_solution(points, config):
    return (points * points).sum(axis=1) / config["controls"].get("k", 1.0)
'''


@pytest.fixture
def simulation(tmp_path, monkeypatch):
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(tmp_path))
    case = tmp_path / "cases" / "fem"
    (case / "taps").mkdir(parents=True)
    (case / "verification").mkdir()
    (case / "taps" / "kernel.py").write_text(FEM_KERNEL)
    reference = case / "verification" / "reference.py"
    reference.write_text(REFERENCE)
    geometry = GeometrySpec(id="box", source=GeometrySource(kind="generated"), dimension=3, entities=[GeometryEntity(id="box", kind="solid", metadata={"primitive": "box", "length": 1., "width": 1., "height": 1.})])
    runtime = CaseRuntime(tmp_path)
    prepared = runtime.prepare(PrepareDomainInput(case_id="fem", geometry=geometry, requirements=DomainRequirements(mesh_policy=MeshPolicy(target_element_size=.4), whole_boundary_role="wall", required_boundary_roles=["wall"])))
    assert prepared.domain.status == "ready", prepared.domain.required_actions
    return runtime, prepared, reference


def test_actual_volume_preparation_and_fem_execution(simulation):
    runtime, prepared, reference = simulation
    assert prepared.domain.mesh.quality.checked_dimension == 3
    assert prepared.domain.mesh.quality.checked_elements > 0
    assert prepared.domain.mesh.quality.min_jacobian > 0
    executed = runtime.execute(ExecuteCaseInput(case_id="fem", prepared_domain=prepared.manifest))
    assert executed.run.result.status == "success", executed.run.errors
    verified = runtime.verify(VerifyCaseInput(run_manifest=executed.manifest, reference_uri=str(reference), relative_tolerance=.15))
    assert verified.report.overall_status == VerificationStatus.VERIFIED
    metric = verified.report.individual_results["IndependentFieldReference"].metrics
    assert metric["weighted_sample_rms_error"] > 0
    assert metric["relative_sample_error"] < .15


def test_runs_preserve_prior_results_and_apply_controls(simulation):
    runtime, prepared, _ = simulation
    first = runtime.execute(ExecuteCaseInput(case_id="fem", prepared_domain=prepared.manifest, controls={"k": 1.}))
    second = runtime.execute(ExecuteCaseInput(case_id="fem", prepared_domain=prepared.manifest, controls={"k": 2.}))
    assert first.run.result.status == second.run.result.status == "success"
    assert first.run.id != second.run.id
    a = np.load(checked_path(first.run.artifacts["solution"], runtime.workspace))
    b = np.load(checked_path(second.run.artifacts["solution"], runtime.workspace))
    assert np.allclose(a, b * 2)
    assert checked_path(first.run.artifacts["solution"], runtime.workspace).exists()


def test_changed_prepared_data_blocks_execution(simulation):
    runtime, prepared, _ = simulation
    path = checked_path(prepared.domain.artifacts["mesh_arrays"], runtime.workspace)
    path.write_bytes(b"corrupted mesh")
    result = runtime.execute(ExecuteCaseInput(case_id="fem", prepared_domain=prepared.manifest))
    assert result.run.result.status == "failed"
    assert "version changed" in result.run.errors[0]


def test_missing_reference_is_uncertain_and_saved(simulation):
    runtime, prepared, _ = simulation
    executed = runtime.execute(ExecuteCaseInput(case_id="fem", prepared_domain=prepared.manifest))
    result = runtime.verify(VerifyCaseInput(run_manifest=executed.manifest))
    assert result.report.overall_status == VerificationStatus.UNCERTAIN
    records = [json.loads(line) for line in (runtime.workspace / "data" / "runtime_attempts.jsonl").read_text().splitlines()]
    assert records[-1]["run_id"] == executed.run.id
    assert records[-1]["verification"]["overall_status"] == "uncertain"


def test_solver_reported_zero_residual_cannot_hide_wrong_field(simulation):
    runtime, prepared, reference = simulation
    kernel = runtime.workspace / "cases" / "fem" / "taps" / "kernel.py"
    kernel.write_text(FEM_KERNEL.replace('np.save(output / "solution.npy", u)', 'np.save(output / "solution.npy", u * 0)'))
    run = runtime.execute(ExecuteCaseInput(case_id="fem", prepared_domain=prepared.manifest))
    result = runtime.verify(VerifyCaseInput(run_manifest=run.manifest, reference_uri=str(reference)))
    assert result.report.overall_status == VerificationStatus.FAILED


def test_real_mesh_refinement_study(simulation):
    runtime, prepared, reference = simulation
    study = runtime.convergence(ConvergenceStudyInput(case_id="fem", prepared_domain=prepared.manifest, reference_uri=str(reference), refinements=[.45, .30, .20], rate_tolerance=1., max_samples=64))
    report = json.loads(checked_path(study.report, runtime.workspace).read_text())
    assert study.status == "verified", report
    assert len(study.runs) == 3
    assert len({row["domain_id"] for row in report["rows"]}) == 3
    assert len({row["kernel_sha256"] for row in report["rows"]}) == 1
    assert report["rows"][-1]["error"] < report["rows"][0]["error"]
    assert report["observed_order"] > 1
    assert report["generated_from_actual_runs"]


def test_missing_boundary_intent_blocks_ready_state(tmp_path, monkeypatch):
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(tmp_path))
    geometry = GeometrySpec(id="box", source=GeometrySource(kind="generated"), dimension=3, entities=[GeometryEntity(id="box", kind="solid", metadata={"length": 1., "width": 1., "height": 1.})])
    runtime = CaseRuntime(tmp_path)
    result = runtime.prepare(PrepareDomainInput(case_id="boundary", geometry=geometry, requirements=DomainRequirements(required_boundary_roles=["inlet"], mesh_policy=MeshPolicy(target_element_size=.5))))
    assert result.domain.status == "needs_input"
    assert "bind_boundary_role:inlet" in result.domain.required_actions


def test_mesh_quality_rejects_inverted_tetra(tmp_path, monkeypatch):
    from physicsos.backends.mesh_quality import assess_mesh_backend
    from physicsos.schemas.common import ArtifactRef
    from physicsos.schemas.mesh import MeshSpec
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(tmp_path))
    path = tmp_path / "inverted.msh"
    meshio.write(path, meshio.Mesh(np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]), [("tetra", [[0,2,1,3]])]), file_format="gmsh22", binary=False)
    quality = assess_mesh_backend(MeshSpec(id="bad", kind="volume", dimension=3, files=[ArtifactRef(uri=str(path), kind="mesh_file", format="msh")]))
    assert not quality.passes
    assert quality.min_jacobian < 0


@pytest.mark.parametrize(("policy", "action"), [
    (MeshPolicy(boundary_layer=True), "boundary_layer"),
    (MeshPolicy(refinement_regions=["near-wall"]), "local_refinement"),
    (MeshPolicy(strategy="structured"), "structured"),
    (MeshPolicy(strategy="adaptive"), "adaptive"),
    (MeshPolicy(strategy="solver_native"), "solver_native"),
])
def test_unimplemented_mesh_policy_cannot_prepare_ready_domain(tmp_path, policy, action):
    geometry = GeometrySpec(id="box", source=GeometrySource(kind="generated"), dimension=3, entities=[GeometryEntity(id="box", kind="solid", metadata={"length": 1., "width": 1., "height": 1.})])
    result = CaseRuntime(tmp_path).prepare(PrepareDomainInput(case_id="policy", geometry=geometry, requirements=DomainRequirements(mesh_policy=policy, whole_boundary_role="wall")))
    assert result.domain.status == "needs_review"
    assert not result.domain.checks["mesh_policy"]
    assert f"provide_mesh_provider_for:{action}" in result.domain.required_actions
    assert not any(item["action"] == "generate_mesh" for item in result.domain.actions)


@pytest.fixture
def material_mesh(tmp_path, monkeypatch):
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(tmp_path))
    path = tmp_path / "materials.msh"
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 0., -1.]])
    cells = [("tetra", [[0, 1, 2, 3], [0, 2, 1, 4]]), ("triangle", [[0, 1, 3], [1, 2, 3], [2, 0, 3], [0, 2, 4], [2, 1, 4], [1, 0, 4], [0, 2, 1]])]
    meshio.write(path, meshio.Mesh(points, cells, cell_data={"gmsh:physical": [np.array([11, 12]), np.array([21] * 6 + [22])], "gmsh:geometrical": [np.array([1, 2]), np.arange(3, 10)]}, field_data={"steel": np.array([11, 3]), "rubber": np.array([12, 3]), "wall": np.array([21, 2]), "contact": np.array([22, 2])}), file_format="gmsh22", binary=False)
    return path


def test_source_mesh_reuse_preserves_materials_and_internal_interface(tmp_path, material_mesh):
    geometry = GeometrySpec(id="materials", source=GeometrySource(kind="mesh_file", uri=str(material_mesh)), dimension=3, regions=[RegionSpec(id=name, label=name, kind="material") for name in ("steel", "rubber")])
    result = CaseRuntime(tmp_path).prepare(PrepareDomainInput(case_id="materials", geometry=geometry, requirements=DomainRequirements(boundary_roles={"contact": "interface"}, required_boundary_roles=["wall", "interface"])))
    assert result.domain.status == "ready", result.domain.required_actions
    assert result.domain.mesh.quality.checked_element_orders == [1]
    assert {region.label: region.kind for region in result.domain.geometry.regions} == {"steel": "material", "rubber": "material"}
    assert len(next(boundary for boundary in result.domain.geometry.boundaries if boundary.role == "wall").entity_ids) == 6
    raw = meshio.read(checked_path(result.domain.artifacts["mesh"], tmp_path))
    assert raw.field_data["contact"].tolist() == [22, 2]
    assert np.concatenate([data for block, data in zip(raw.cells, raw.cell_data["gmsh:physical"]) if block.dim == 3]).tolist() == [11, 12]


@pytest.mark.parametrize("operation", ["remesh", "repair", "grid"])
def test_surface_conversion_cannot_drop_undeclared_material_partition(tmp_path, material_mesh, operation):
    # The raw mesh tags must be inspected even when GeometrySpec is unlabelled.
    geometry = GeometrySpec(id="materials", source=GeometrySource(kind="mesh_file", uri=str(material_mesh)), dimension=3)
    result = CaseRuntime(tmp_path).prepare(PrepareDomainInput(case_id="materials", geometry=geometry, force_remesh=operation == "remesh", repair="always" if operation == "repair" else "auto", requirements=DomainRequirements(representation="background_grid" if operation == "grid" else "mesh", whole_boundary_role="wall")))
    assert result.domain.status == "needs_review"
    assert not result.domain.checks["region_preservation"]
    assert result.domain.required_actions == ["provide_interface_preserving_meshing_or_repair_provider"]
    assert not any(item["action"] in {"generate_mesh", "repair_surface"} for item in result.domain.actions)


def test_surface_conversion_detects_internal_interface_without_material_names(tmp_path, material_mesh):
    raw = meshio.read(material_mesh)
    for block, tags in zip(raw.cells, raw.cell_data["gmsh:physical"]):
        if block.dim == 3:
            tags[:] = 11
    raw.field_data.pop("rubber")
    path = tmp_path / "single_material.msh"
    meshio.write(path, raw, file_format="gmsh22", binary=False)
    geometry = GeometrySpec(id="contact", source=GeometrySource(kind="mesh_file", uri=str(path)), dimension=3)
    result = CaseRuntime(tmp_path).prepare(PrepareDomainInput(case_id="contact", geometry=geometry, force_remesh=True))
    assert result.domain.status == "needs_review"
    assert not result.domain.checks["region_preservation"]


def test_whole_boundary_binding_cannot_overwrite_internal_interface(tmp_path, material_mesh):
    geometry = GeometrySpec(id="materials", source=GeometrySource(kind="mesh_file", uri=str(material_mesh)), dimension=3)
    result = CaseRuntime(tmp_path).prepare(PrepareDomainInput(case_id="materials", geometry=geometry, requirements=DomainRequirements(whole_boundary_role="wall")))
    assert result.domain.status == "needs_review"
    assert not result.domain.checks["boundary_preservation"]
    assert result.domain.required_actions == ["bind_exterior_boundary_groups_without_overwriting_internal_interfaces"]


def test_reused_linear_mesh_cannot_satisfy_quadratic_requirement(simulation):
    runtime, prepared, _ = simulation
    geometry = prepared.domain.geometry.model_copy(deep=True)
    geometry.source = GeometrySource(kind="mesh_file", uri=str(checked_path(prepared.domain.artifacts["mesh"], runtime.workspace)))
    result = runtime.prepare(PrepareDomainInput(case_id="order", geometry=geometry, requirements=DomainRequirements(mesh_policy=MeshPolicy(element_order=2), whole_boundary_role="wall"), max_meshing_attempts=1))
    assert result.domain.status == "needs_review"
    quality_check = next(item for item in result.domain.actions if item["action"] == "check_mesh_quality")
    assert quality_check["element_orders"] == [1]
    assert any("requested order 2" in issue for issue in quality_check["issues"])


def test_runtime_repair_uses_explicit_workspace(simulation):
    from physicsos.schemas.geometry_repair import GeometryRepairOptions
    runtime, prepared, _ = simulation
    isolated = CaseRuntime(runtime.workspace / "isolated")
    geometry = prepared.domain.geometry.model_copy(deep=True)
    geometry.source = GeometrySource(kind="mesh_file", uri=str(checked_path(prepared.domain.artifacts["mesh"], runtime.workspace)))
    result = isolated.prepare(PrepareDomainInput(case_id="repair", geometry=geometry, repair="always", repair_options=GeometryRepairOptions(python_executable=str(isolated.workspace / "missing-python"))))
    assert result.domain.status == "backend_unavailable", result.domain.required_actions
    report = checked_path(result.domain.artifacts["repair_report"], isolated.workspace)
    assert report.is_relative_to(isolated.workspace)
    assert json.loads(report.read_text())["status"] == "backend_unavailable"


def test_background_grid_uses_final_geometry_and_real_reruns(tmp_path, monkeypatch):
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(tmp_path))
    runtime = CaseRuntime(tmp_path)
    case = tmp_path / "cases" / "grid"
    (case / "taps").mkdir(parents=True)
    (case / "verification").mkdir()
    kernel = case / "taps" / "kernel.py"
    kernel.write_text('''from pathlib import Path
import json
import numpy as np
def run_case(config):
    grid = json.loads(Path(config["domain_artifacts"]["background_grid"]).read_text())
    x, y, z = np.meshgrid(*[grid["axes"][name] for name in "xyz"], indexing="ij")
    output = Path(config["output_dir"])
    np.save(output / "solution.npy", x*x + y*y + z*z)
    (output / "runtime_metadata.json").write_text('{"method":"analytic nodal field"}')
    (output / "residual_history.json").write_text('[]')
    return {}
''')
    reference = case / "verification" / "reference.py"
    reference.write_text(REFERENCE)
    geometry = GeometrySpec(id="box", source=GeometrySource(kind="generated"), dimension=3, entities=[GeometryEntity(id="box", kind="solid", metadata={"length": 1., "width": 1., "height": 1.})])
    prepared = runtime.prepare(PrepareDomainInput(case_id="grid", geometry=geometry, requirements=DomainRequirements(representation="background_grid", grid_resolution=[9,9,9], whole_boundary_role="wall", required_boundary_roles=["wall"], mesh_policy=MeshPolicy(target_element_size=.4))))
    assert prepared.domain.status == "ready", prepared.domain.required_actions
    sdf = np.load(checked_path(prepared.domain.artifacts["sdf"], tmp_path))
    assert sdf[4,4,4] < 0
    study = runtime.convergence(ConvergenceStudyInput(case_id="grid", prepared_domain=prepared.manifest, reference_uri=str(reference), axis="grid", refinements=[9,17,33], max_samples=32))
    report = json.loads(checked_path(study.report, tmp_path).read_text())
    assert study.status == "verified", report
    assert report["observed_order"] == pytest.approx(2, abs=.1)


def test_input_mutation_cannot_be_reported_as_success(simulation):
    runtime, prepared, _ = simulation
    kernel = runtime.workspace / "cases" / "fem" / "taps" / "kernel.py"
    kernel.write_text(FEM_KERNEL.replace('return {"status": "success"}', 'Path(config["domain_artifacts"]["mesh_arrays"]).write_bytes(b"changed input")\n    return {"status": "success"}'))
    result = runtime.execute(ExecuteCaseInput(case_id="fem", prepared_domain=prepared.manifest))
    assert result.run.result.status == "failed"
    assert "modified a versioned input" in result.run.errors[0]


def test_new_verification_preserves_prior_report(simulation):
    runtime, prepared, reference = simulation
    executed = runtime.execute(ExecuteCaseInput(case_id="fem", prepared_domain=prepared.manifest))
    first = runtime.verify(VerifyCaseInput(run_manifest=executed.manifest, reference_uri=str(reference), relative_tolerance=.15))
    second = runtime.verify(VerifyCaseInput(run_manifest=executed.manifest))
    assert first.artifact.uri != second.artifact.uri
    assert checked_path(first.artifact, runtime.workspace).is_file()


def test_changed_run_output_is_failed_evidence(simulation):
    runtime, prepared, reference = simulation
    executed = runtime.execute(ExecuteCaseInput(case_id="fem", prepared_domain=prepared.manifest))
    checked_path(executed.run.artifacts["solution"], runtime.workspace).write_bytes(b"corrupted output")
    result = runtime.verify(VerifyCaseInput(run_manifest=executed.manifest, reference_uri=str(reference)))
    assert result.report.overall_status == VerificationStatus.FAILED
    assert "evidence changed" in result.report.individual_results["IndependentFieldReference"].message


def test_runtime_history_counts_distinct_attempts(simulation):
    from physicsos.schemas.case_runtime import SearchRuntimeHistoryInput
    runtime, prepared, reference = simulation
    executed = runtime.execute(ExecuteCaseInput(case_id="fem", prepared_domain=prepared.manifest))
    runtime.verify(VerifyCaseInput(run_manifest=executed.manifest, reference_uri=str(reference), relative_tolerance=.15))
    runtime.verify(VerifyCaseInput(run_manifest=executed.manifest, reference_uri=str(reference), relative_tolerance=.15))
    history = runtime.history(SearchRuntimeHistoryInput(dimension=3, representation="mesh"))
    assert history.matched_attempts == 1
    assert history.outcome_counts["verified"] == 1
    assert history.attempts[0]["kernel_sha256"] == executed.run.kernel_sha256


def test_unprepared_legacy_execution_cannot_reuse_old_solution(tmp_path, monkeypatch):
    from physicsos.tools.taps_tools import ExecuteTAPSKernelInput, execute_taps_kernel
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(tmp_path))
    directory = tmp_path / "cases" / "legacy" / "taps"
    directory.mkdir(parents=True)
    (directory / "kernel.py").write_text('def run_case(config):\n    return {"status": "success"}\n')
    np.save(directory / "solution.npy", [1.])
    (directory / "runtime_metadata.json").write_text('{}')
    (directory / "residual_history.json").write_text('[]')
    output = execute_taps_kernel(ExecuteTAPSKernelInput(case_id="legacy"))
    assert not output.passes
    assert not (directory / "solution.npy").exists()


def test_old_verification_chain_cannot_accept_a_case_without_solving(tmp_path, monkeypatch):
    from physicsos.tools.verification_chain_tools import GenerateConvergenceCodeInput, generate_convergence_code, ExecuteConvergenceCodeInput, execute_convergence_code
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(tmp_path))
    generate_convergence_code(GenerateConvergenceCodeInput(case_id="unsolved"))
    result = execute_convergence_code(ExecuteConvergenceCodeInput(case_id="unsolved"))
    assert not result.passes
    payload = json.loads(Path(tmp_path / "cases" / "unsolved" / "verification" / "convergence_report.json").read_text())
    assert payload["status"] == "uncertain"
    assert payload["rows"] == []


def test_empty_verifiers_and_missing_balances_are_not_verified():
    import runpy
    from physicsos.verification.pipeline import VerificationPipeline
    from physicsos.verification.conservation import ConservationChecker
    from physicsos.verification.base import AggregateVerificationReport
    from pydantic import ValidationError
    helpers = runpy.run_path(str(Path(__file__).parent / "verification" / "test_verification_framework.py"))
    problem, result = helpers["make_simple_problem"](), helpers["make_simple_result"]()
    assert VerificationPipeline().verify(problem, result).overall_status == VerificationStatus.UNCERTAIN
    result.residuals = {}
    assert ConservationChecker().verify(problem, result).status == VerificationStatus.UNCERTAIN
    with pytest.raises(ValidationError):
        AggregateVerificationReport(problem_id=problem.id, result_id=result.id, overall_status="verified", failed_checks=1)
