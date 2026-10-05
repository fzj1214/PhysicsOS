"""Standalone PaMO job worker, compatible with Python 3.10 and CUDA.

Run as a file in a separate environment, or in runners/pamo's image.
Only this worker imports PaMO, torch, trimesh, Gmsh, and PyMeshLab.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import numpy as np

if __package__:
    from .surface_mesh import read_mesh_surface, surface_metrics, validate_surface_arrays, write_surface_stl
else:
    from surface_mesh import read_mesh_surface, surface_metrics, validate_surface_arrays, write_surface_stl


class BackendUnavailable(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _hash_stream(handle)


def _hash_stream(handle) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _job_path(directory: Path, value: str) -> Path:
    path = (directory / value).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("Worker paths must remain inside the job directory.")
    return path


def _cuda_backend():
    try:
        import torch
        if not torch.cuda.is_available():
            raise BackendUnavailable("PaMO requires an NVIDIA CUDA device; configure a CUDA Python environment or executor='docker'.")
        from pamo import PaMO
        import trimesh
        return torch, PaMO, trimesh
    except (ImportError, OSError) as exc:
        raise BackendUnavailable(f"PaMO CUDA environment is unavailable: {exc}") from exc


def _cad_surface(path: Path, element_size: float | None):
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        if element_size is not None:
            gmsh.option.setNumber("Mesh.MeshSizeMin", element_size)
            gmsh.option.setNumber("Mesh.MeshSizeMax", element_size)
        gmsh.model.mesh.generate(2)
        tags, coordinates, _ = gmsh.model.mesh.getNodes()
        vertices = np.asarray(coordinates).reshape(-1, 3)
        index = {int(tag): i for i, tag in enumerate(tags)}
        types, _, nodes = gmsh.model.mesh.getElements(2)
        faces = []
        for element_type, node_tags in zip(types, nodes):
            name, _, _, node_count, _, primary_count = gmsh.model.mesh.getElementProperties(element_type)
            for cell in np.asarray(node_tags).reshape(-1, node_count):
                corners = [index[int(tag)] for tag in cell[:primary_count]]
                if name.startswith("Triangle"):
                    faces.append(corners[:3])
                elif name.startswith("Quadrilateral"):
                    faces.extend([[corners[0], corners[1], corners[2]], [corners[0], corners[2], corners[3]]])
        return validate_surface_arrays(vertices, np.asarray(faces, dtype=np.int64))
    finally:
        gmsh.finalize()


def load_surface(path: Path, source_kind: str, options: dict):
    if source_kind in {"cad_step", "cad_iges"}:
        return _cad_surface(path, options.get("surface_element_size"))
    if path.suffix.lower() in {".msh", ".vtk", ".vtu", ".xdmf", ".mesh", ".inp"}:
        return read_mesh_surface(path)
    import trimesh
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Input must contain a triangle surface; point clouds cannot be repaired by PaMO.")
    return validate_surface_arrays(mesh.vertices, mesh.faces)


def optimize_surface(vertices, faces, options: dict):
    torch, PaMO, trimesh = _cuda_backend()
    # Normalize in float64 before GPU float32 conversion, then restore the
    # original coordinates. This avoids losing small features at large offsets.
    origin = vertices.min(axis=0)
    scale = float(np.ptp(vertices, axis=0).max())
    if scale <= 0:
        raise ValueError("Input surface has zero spatial extent.")
    normalized = (vertices - origin) / scale
    mesh = trimesh.Trimesh(vertices=normalized, faces=faces, process=False)
    policies = {"conservative": 1.0, "balanced": 0.5, "aggressive": 0.1}
    ratio = options.get("simplification_ratio") or policies[options["repair_policy"]]
    if options.get("target_faces") is not None:
        ratio = min(1.0, options["target_faces"] / len(faces))
    optimizer = PaMO(mesh, use_stage1=True, use_stage3=True)
    repaired_vertices, repaired_faces = optimizer.run(
        torch.from_numpy(np.asarray(normalized, dtype=np.float32)).cuda(),
        torch.from_numpy(np.asarray(faces, dtype=np.int32)).cuda(),
        ratio=float(ratio), min_verts=0,
    )
    repaired_vertices = np.asarray(repaired_vertices, dtype=np.float64) * scale + origin
    repaired_vertices, repaired_faces = validate_surface_arrays(repaired_vertices, repaired_faces)
    parameters = {
        **options, "effective_ratio": float(ratio), "min_verts": 0,
        "use_stage1": True, "use_stage3": True,
        "effective_resolution": int(optimizer.R),
        "normalization_origin": origin.tolist(), "normalization_scale": scale,
    }
    return repaired_vertices, repaired_faces, parameters


def intersection_count(vertices, faces) -> tuple[int | None, str | None]:
    try:
        import pymeshlab
        mesh_set = pymeshlab.MeshSet()
        mesh_set.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces.astype(np.int32)))
        mesh_set.apply_filter("compute_selection_by_self_intersections_per_face")
        return int(mesh_set.current_mesh().selected_face_number()), None
    except Exception as exc:
        return None, f"Independent self-intersection check unavailable: {exc}"


def _samples(vertices, faces, count: int, seed: int):
    rng = np.random.default_rng(seed)
    triangles = vertices[faces]
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    if float(area.sum()) <= 0:
        raise ValueError("Cannot sample a surface with zero area.")
    selected = triangles[rng.choice(len(faces), count, p=area / area.sum())]
    barycentric = rng.random((count, 2))
    folded = barycentric.sum(axis=1) > 1
    barycentric[folded] = 1 - barycentric[folded]
    samples = selected[:, 0] + barycentric[:, :1] * (selected[:, 1] - selected[:, 0]) + barycentric[:, 1:] * (selected[:, 2] - selected[:, 0])
    vertex_ids = rng.choice(len(vertices), min(count, len(vertices)), replace=False)
    return np.concatenate([samples, vertices[vertex_ids]])


def sampled_surface_distance(original_vertices, original_faces, vertices, faces, sample_count: int) -> dict:
    import trimesh
    original = trimesh.Trimesh(vertices=original_vertices, faces=original_faces, process=False)
    repaired = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    _, forward, _ = trimesh.proximity.closest_point(repaired, _samples(original_vertices, original_faces, sample_count, 0))
    _, reverse, _ = trimesh.proximity.closest_point(original, _samples(vertices, faces, sample_count, 1))
    diagonal = float(np.linalg.norm(np.ptp(original_vertices, axis=0)))
    maximum = float(max(forward.max(), reverse.max()))
    return {
        "method": "deterministic_bidirectional_surface_sampling",
        "samples_per_direction": sample_count,
        "sampled_max_distance": maximum, "relative_sampled_max_distance": maximum / diagonal,
        "mean_distance": float((forward.mean() + reverse.mean()) / 2),
        "is_exact_hausdorff_bound": False,
    }


def quality_report(metrics: dict, intersections: int | None, deviation: dict, options: dict) -> dict:
    issues = []
    for key, description in [("watertight", "Surface is not watertight."), ("manifold", "Surface is not manifold."), ("winding_consistent", "Surface winding is inconsistent.")]:
        if not metrics[key]:
            issues.append(description)
    if intersections is None:
        issues.append("Self-intersections were not independently checked.")
    elif intersections:
        issues.append(f"Detected {intersections} self-intersecting faces.")
    if metrics["aspect_ratio_p95"] > options["max_aspect_ratio_p95"]:
        issues.append("Surface aspect ratio exceeds the configured limit.")
    if metrics["max_skewness"] > options["max_skewness"]:
        issues.append("Surface skewness exceeds the configured limit.")
    distance = deviation.get("relative_sampled_max_distance")
    if distance is None or not np.isfinite(distance):
        issues.append("Surface deviation could not be measured.")
    elif distance > options["max_relative_surface_distance"]:
        issues.append("Surface deviation exceeds the configured limit.")
    return {
        "watertight": metrics["watertight"], "manifold": metrics["manifold"],
        "self_intersections": intersections, "passes": not issues,
        "unresolved_regions": [], "issues": issues,
    }


def process_request(request: dict, directory: Path) -> dict:
    report = {
        "schema_version": "physicsos.geometry_repair.v1", "job_id": request["job_id"],
        "input_sha256": request["input_sha256"], "status": "failed",
        "quality": {"passes": False, "issues": []}, "warnings": [],
    }
    try:
        if request.get("schema_version") != "physicsos.pamo_request.v1":
            raise ValueError("Unsupported PaMO request schema.")
        source = _job_path(directory, request["input_path"])
        if _sha256(source) != request["input_sha256"]:
            raise ValueError("Input snapshot checksum does not match the request.")
        _cuda_backend()
        options = request["options"]
        original_vertices, original_faces = load_surface(source, request["source_kind"], options)
        report["before"] = surface_metrics(original_vertices, original_faces)
        vertices, faces, parameters = optimize_surface(original_vertices, original_faces, options)
        report["parameters"] = parameters
        report["after"] = surface_metrics(vertices, faces)
        intersections, warning = intersection_count(vertices, faces)
        if warning:
            report["warnings"].append(warning)
        try:
            deviation = sampled_surface_distance(original_vertices, original_faces, vertices, faces, options["validation_samples"])
        except Exception as exc:
            deviation = {"error": str(exc)}
        report["deviation"] = deviation
        report["quality"] = quality_report(report["after"], intersections, deviation, options)
        report["status"] = "repaired" if report["quality"]["passes"] else "needs_review"
        output = directory / "repaired.stl"
        write_surface_stl(output, vertices, faces)
        np.savez_compressed(directory / "repaired.npz", vertices=vertices, faces=faces)
        report.update(output_surface=output.name, output_mesh="repaired.npz", output_sha256=_sha256(output))
        try:
            version = importlib.metadata.version("pamo")
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
        report["backend_version"] = os.environ.get("PAMO_REVISION") or version
        report["warnings"].append("Surface deviation is sampled, not an exact Hausdorff bound; physical boundary labels must be rebound after repair.")
    except BackendUnavailable as exc:
        report.update(status="backend_unavailable", error=str(exc))
        report["quality"]["issues"] = [str(exc)]
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        report["quality"]["passes"] = False
        report["quality"]["issues"] = [report["error"]]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an isolated PhysicsOS PaMO repair job.")
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    args = parser.parse_args(argv)
    request = json.loads(args.request.read_text(encoding="utf-8"))
    report = process_request(request, args.request.resolve().parent)
    args.response.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return 0 if report["status"] in {"repaired", "needs_review"} else 3 if report["status"] == "backend_unavailable" else 1


if __name__ == "__main__":
    sys.exit(main())
