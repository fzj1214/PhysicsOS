from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess

import meshio
import numpy as np
from pydantic import ValidationError
import pytest

from physicsos.backends import geometry_repair, pamo_worker
from physicsos.backends.surface_mesh import read_mesh_surface, surface_metrics, write_surface_stl
from physicsos.paths import resolve_workspace_path
from physicsos.schemas.geometry import (
    BoundaryRegionSpec, GeometryEncoding, GeometryEntity, GeometrySource, GeometrySpec, RegionSpec,
)
from physicsos.tools.geometry_tools import RepairGeometryInput, _triangle_quality, repair_geometry


VERTICES = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
FACES = np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])


@pytest.fixture
def asset(tmp_path, monkeypatch):
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PHYSICSOS_PAMO_EXECUTOR", raising=False)
    monkeypatch.delenv("PHYSICSOS_PAMO_PYTHON", raising=False)
    path = tmp_path / "original asset.stl"
    write_surface_stl(path, VERTICES, FACES)
    return GeometrySpec(
        id="geometry:test-asset", source=GeometrySource(kind="stl", uri=str(path)), dimension=3,
        entities=[GeometryEntity(id="old-face", kind="surface")],
        regions=[RegionSpec(id="solid", label="solid", kind="solid", entity_ids=["old-face"])],
        boundaries=[BoundaryRegionSpec(id="inlet", label="inlet", kind="inlet", role="inlet", entity_ids=["old-face"], confidence=1.0)],
        encodings=[GeometryEncoding(kind="sdf", uri="/workspace/old_sdf.npy")],
    )


def respond(command, **kwargs):
    """Worker protocol fixture; all adopted geometry is checked by host code."""
    directory = Path(kwargs["cwd"])
    request = json.loads((directory / "request.json").read_text())
    surface = directory / "repaired.stl"
    write_surface_stl(surface, VERTICES, FACES)
    np.savez(directory / "repaired.npz", vertices=VERTICES, faces=FACES)
    payload = {
        "job_id": request["job_id"], "input_sha256": request["input_sha256"], "status": "repaired",
        "quality": {"passes": True, "watertight": True, "manifold": True, "self_intersections": 0},
        "deviation": {"relative_sampled_max_distance": 0.0},
        "output_surface": "repaired.stl", "output_mesh": "repaired.npz",
        "output_sha256": hashlib.sha256(surface.read_bytes()).hexdigest(),
    }
    (directory / "response.json").write_text(json.dumps(payload))
    return subprocess.CompletedProcess(command, 0, "worker completed", "")


def test_checked_repair_preserves_source_and_invalidates_old_bindings(asset, monkeypatch):
    original = Path(asset.source.uri).read_bytes()
    monkeypatch.setattr(geometry_repair.subprocess, "run", respond)
    result = repair_geometry(RepairGeometryInput(geometry=asset, case_id="asset-repair"))
    assert result.status == "repaired"
    assert result.geometry.quality.passes
    assert result.geometry.source.uri != asset.source.uri
    assert Path(asset.source.uri).read_bytes() == original
    assert result.geometry.encodings == []
    assert result.invalidated_encodings[0].uri == "/workspace/old_sdf.npy"
    assert result.geometry.regions[0].entity_ids == []
    assert result.geometry.boundaries[0].entity_ids == []
    assert result.geometry.boundaries[0].role == "inlet"
    assert result.geometry.boundaries[0].confidence == 0
    assert result.requires_boundary_relabel
    assert asset.boundaries[0].entity_ids == ["old-face"]
    assert asset.encodings
    report = json.loads(resolve_workspace_path(result.repair_report.uri).read_text())
    assert report["after"]["watertight"]
    assert "cases/asset-repair/geometry/repairs" in result.repair_report.uri


def test_repeated_repairs_get_distinct_artifacts(asset, monkeypatch):
    monkeypatch.setattr(geometry_repair.subprocess, "run", respond)
    first = repair_geometry(RepairGeometryInput(geometry=asset))
    second = repair_geometry(RepairGeometryInput(geometry=asset))
    assert first.repair_report.uri != second.repair_report.uri
    assert resolve_workspace_path(first.repaired_surface.uri).is_file()


@pytest.mark.parametrize("defect", ["wrong_job", "wrong_checksum", "unknown_intersections", "missing_distance", "excessive_distance", "outside_job", "nonmanifold", "nonzero_exit", "mismatched_stl"])
def test_invalid_worker_evidence_cannot_activate_geometry(asset, monkeypatch, defect):
    def broken(command, **kwargs):
        completed = respond(command, **kwargs)
        directory = Path(kwargs["cwd"])
        response = directory / "response.json"
        payload = json.loads(response.read_text())
        if defect == "wrong_job":
            payload["job_id"] = "another-job"
        elif defect == "wrong_checksum":
            payload["output_sha256"] = "wrong"
        elif defect == "unknown_intersections":
            payload["quality"]["self_intersections"] = None
        elif defect == "missing_distance":
            payload["deviation"] = {}
        elif defect == "excessive_distance":
            payload["deviation"]["relative_sampled_max_distance"] = 1.0
        elif defect == "outside_job":
            payload["output_surface"] = "../another-output.stl"
        elif defect == "nonmanifold":
            np.savez(directory / "repaired.npz", vertices=VERTICES, faces=np.concatenate([FACES, FACES[:1]]))
        elif defect == "nonzero_exit":
            completed.returncode = 1
        elif defect == "mismatched_stl":
            surface = directory / "repaired.stl"
            write_surface_stl(surface, VERTICES + [0, 0, 10], FACES)
            payload["output_sha256"] = hashlib.sha256(surface.read_bytes()).hexdigest()
        response.write_text(json.dumps(payload))
        return completed
    monkeypatch.setattr(geometry_repair.subprocess, "run", broken)
    result = repair_geometry(RepairGeometryInput(geometry=asset))
    assert result.status == "failed"
    assert not result.geometry.quality.passes
    assert result.geometry.source == asset.source
    assert result.geometry.encodings == asset.encodings


def test_needs_review_keeps_candidate_without_activating_it(asset, monkeypatch):
    def review(command, **kwargs):
        completed = respond(command, **kwargs)
        path = Path(kwargs["cwd"]) / "response.json"
        payload = json.loads(path.read_text())
        payload["status"] = "needs_review"
        payload["quality"] = {"passes": False, "issues": ["Feature deviation exceeds tolerance"]}
        path.write_text(json.dumps(payload))
        return completed
    monkeypatch.setattr(geometry_repair.subprocess, "run", review)
    result = repair_geometry(RepairGeometryInput(geometry=asset))
    assert result.status == "needs_review"
    assert result.repaired_surface is not None
    assert result.geometry.source == asset.source
    assert not result.geometry.quality.passes


def test_missing_python_environment_has_explicit_status(asset):
    result = repair_geometry(RepairGeometryInput(geometry=asset, python_executable=str(Path(asset.source.uri).parent / "no-such-python")))
    assert result.status == "backend_unavailable"
    assert not result.geometry.quality.passes
    assert resolve_workspace_path(result.repair_report.uri).exists()


def test_missing_source_does_not_launch_worker(asset, monkeypatch):
    asset.source.uri = "/workspace/missing.stl"
    monkeypatch.setattr(geometry_repair.subprocess, "run", lambda *a, **kw: pytest.fail("worker must not run"))
    assert repair_geometry(RepairGeometryInput(geometry=asset)).status == "failed"


def test_changed_source_revision_does_not_launch_worker(asset, monkeypatch):
    asset.source.checksum = "0" * 64
    monkeypatch.setattr(geometry_repair.subprocess, "run", lambda *a, **kw: pytest.fail("worker must not run"))
    result = repair_geometry(RepairGeometryInput(geometry=asset))
    assert result.status == "failed"
    assert "changed" in result.warnings[0]


def test_docker_mounts_job_and_passes_only_container_paths(asset, monkeypatch):
    seen = []
    def docker(command, **kwargs):
        seen.append(command)
        request = json.loads((Path(kwargs["cwd"]) / "request.json").read_text())
        assert request["input_path"] == "source.stl"
        return respond(command, **kwargs)
    monkeypatch.setattr(geometry_repair.subprocess, "run", docker)
    result = repair_geometry(RepairGeometryInput(geometry=asset, executor="docker", docker_image="test-pamo:revision"))
    command = seen[0]
    assert result.status == "repaired"
    assert command[:3] == ["docker", "run", "--rm"]
    assert command[command.index("--gpus") + 1] == "all"
    assert command[command.index("--volume") + 1].endswith(":/job")
    assert command[-5:] == ["test-pamo:revision", "--request", "/job/request.json", "--response", "/job/response.json"]


def test_timeout_stops_named_docker_container(asset, monkeypatch):
    commands = []
    def timed_out(command, **kwargs):
        commands.append(command)
        if command[1] == "run":
            raise subprocess.TimeoutExpired(command, 1, output=b"partial")
        return subprocess.CompletedProcess(command, 0, "", "")
    monkeypatch.setattr(geometry_repair.subprocess, "run", timed_out)
    result = repair_geometry(RepairGeometryInput(geometry=asset, executor="docker", timeout_seconds=1))
    assert result.status == "failed"
    assert commands[1][:4] == ["docker", "stop", "--time", "1"]
    assert commands[1][-1] == commands[0][commands[0].index("--name") + 1]
    log = json.loads(resolve_workspace_path(result.execution_log.uri).read_text())
    assert log["timed_out"] and log["stdout"] == "partial"


def test_surface_metrics_detect_vertex_nonmanifoldness():
    vertices = np.concatenate([VERTICES, -VERTICES[1:]])
    second_shell = np.asarray([[0, 5, 4], [0, 4, 6], [4, 5, 6], [5, 0, 6]])
    metrics = surface_metrics(vertices, np.concatenate([FACES, second_shell]))
    assert metrics["watertight"]
    assert not metrics["manifold"]
    assert metrics["nonmanifold_vertices"] == 1


def test_triangle_quality_is_invariant_to_3d_orientation():
    xy = _triangle_quality([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [0, 1, 2])
    yz = _triangle_quality([[0, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 1, 2])
    assert yz == pytest.approx(xy)


def test_extract_volume_boundary_excludes_internal_faces(tmp_path):
    vertices = np.concatenate([VERTICES, [[0, 0, -1]]])
    path = tmp_path / "volume.vtu"
    meshio.write(path, meshio.Mesh(vertices, [("tetra", [[0, 1, 2, 3], [0, 2, 1, 4]])]))
    points, faces = read_mesh_surface(path)
    assert len(faces) == 6
    metrics = surface_metrics(points, faces)
    assert metrics["watertight"] and metrics["manifold"] and metrics["winding_consistent"]


@pytest.mark.parametrize("binary", [False, True])
def test_stl_codec_preserves_surface_and_topology(tmp_path, binary):
    path = tmp_path / "surface.stl"
    if binary:
        meshio.write(path, meshio.Mesh(VERTICES, [("triangle", FACES)]), binary=True)
    else:
        write_surface_stl(path, VERTICES, FACES)
    vertices, faces = read_mesh_surface(path)
    assert np.array_equal(vertices[faces], VERTICES[FACES])
    assert surface_metrics(vertices, faces)["watertight"]


def test_worker_records_unavailable_cuda(asset, monkeypatch):
    def unavailable():
        raise pamo_worker.BackendUnavailable("NVIDIA CUDA unavailable")
    monkeypatch.setattr(pamo_worker, "_cuda_backend", unavailable)
    path = Path(asset.source.uri)
    report = pamo_worker.process_request({
        "schema_version": "physicsos.pamo_request.v1", "job_id": "cpu-only",
        "input_path": path.name, "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }, path.parent)
    assert report["status"] == "backend_unavailable"
    assert not report["quality"]["passes"]


def test_worker_normalizes_coordinates_and_passes_explicit_min_verts(monkeypatch):
    calls = []
    class Array:
        def __init__(self, value):
            self.value = value
        def cuda(self):
            return self.value
    class Optimizer:
        R = 128
        def __init__(self, mesh, **kwargs):
            calls.append(kwargs)
        def run(self, points, faces, **kwargs):
            calls.append(kwargs)
            assert points.min() >= 0 and points.max() <= 1
            return points, faces
    mesh_type = lambda **kw: SimpleNamespace(**kw)
    monkeypatch.setattr(pamo_worker, "_cuda_backend", lambda: (SimpleNamespace(from_numpy=Array), Optimizer, SimpleNamespace(Trimesh=mesh_type)))
    original = VERTICES * 2 + 1e9
    vertices, faces, parameters = pamo_worker.optimize_surface(original, FACES, {"repair_policy": "conservative"})
    assert np.array_equal(vertices, original)
    assert calls == [{"use_stage1": True, "use_stage3": True}, {"ratio": 1.0, "min_verts": 0}]
    assert parameters["effective_resolution"] == 128


def test_unknown_intersections_cannot_pass_worker_quality():
    metrics = surface_metrics(VERTICES, FACES)
    options = {"max_aspect_ratio_p95": 25., "max_skewness": .95, "max_relative_surface_distance": .01}
    report = pamo_worker.quality_report(metrics, None, {"relative_sampled_max_distance": 0.}, options)
    assert not report["passes"]


def test_mutually_exclusive_targets_and_finite_tolerance(asset):
    with pytest.raises(ValidationError):
        RepairGeometryInput(geometry=asset, target_faces=100, simplification_ratio=.5)
    with pytest.raises(ValidationError):
        RepairGeometryInput(geometry=asset, max_relative_surface_distance=float("inf"))


def test_geometry_agent_has_repair_and_quality_tools():
    from physicsos.tools.registry import DEEPAGENTS_SUBAGENT_TOOL_GROUPS
    names = {tool.__name__ for tool in DEEPAGENTS_SUBAGENT_TOOL_GROUPS["geometry-embedding-agent"]}
    assert {"repair_geometry", "generate_mesh", "assess_mesh_quality", "mesh_semantics_gate"} <= names


def test_actual_cad_asset_can_be_tessellated_and_checked_without_gpu(tmp_path):
    import gmsh
    path = tmp_path / "box.step"
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.addBox(0, 0, 0, 2, 1, 1)
        gmsh.model.occ.synchronize()
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    vertices, faces = pamo_worker._cad_surface(path, 0.4)
    metrics = surface_metrics(vertices, faces)
    assert metrics["face_count"] > 0
    assert metrics["watertight"] and metrics["manifold"] and metrics["winding_consistent"]


def test_sampled_distance_measures_real_surface_displacement():
    pytest.importorskip("trimesh")
    pytest.importorskip("rtree")
    same = pamo_worker.sampled_surface_distance(VERTICES, FACES, VERTICES, FACES, 128)
    moved = pamo_worker.sampled_surface_distance(VERTICES, FACES, VERTICES + [0, 0, .2], FACES, 128)
    assert same["sampled_max_distance"] < 1e-12
    assert moved["relative_sampled_max_distance"] > .05
    assert not same["is_exact_hausdorff_bound"]


def test_worker_files_parse_as_python_310():
    import ast
    for module in (pamo_worker,):
        ast.parse(Path(module.__file__).read_text(), feature_version=(3, 10))
    from physicsos.backends import surface_mesh
    ast.parse(Path(surface_mesh.__file__).read_text(), feature_version=(3, 10))
