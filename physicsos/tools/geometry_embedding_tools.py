from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import struct
from pathlib import Path
from shutil import copyfile
from typing import Literal

import numpy as np
from pydantic import Field

from physicsos.config import runtime_paths
from physicsos.paths import resolve_workspace_path, to_agent_path
from physicsos.schemas.common import ArtifactRef, StrictBaseModel
from physicsos.tools.case_tools import _append_event, _artifact, _case_dir, _workspace


class STLGeometrySummary(StrictBaseModel):
    triangle_count: int
    bounds_min: list[float]
    bounds_max: list[float]
    units: str = "m"
    warnings: list[str] = Field(default_factory=list)


class ImportSTLGeometryInput(StrictBaseModel):
    case_id: str
    source_uri: str
    units: str = "m"
    target_filename: str = "input.stl"


class ImportSTLGeometryOutput(StrictBaseModel):
    stl: ArtifactRef
    summary: STLGeometrySummary
    summary_artifact: ArtifactRef


class GeneratePrimitiveGeometryInput(StrictBaseModel):
    case_id: str
    description: str
    primitive: Literal["auto", "box", "sphere", "cylinder"] = "auto"
    units: str = "m"
    resolution: int = 24
    target_filename: str = "input.stl"


class GeneratePrimitiveGeometryOutput(StrictBaseModel):
    stl: ArtifactRef
    summary: STLGeometrySummary
    summary_artifact: ArtifactRef
    request_artifact: ArtifactRef
    warnings: list[str] = Field(default_factory=list)


class PrepareGeometryAnalysisFilesInput(StrictBaseModel):
    case_id: str
    source_uri: str | None = None
    description: str | None = None
    primitive: Literal["auto", "box", "sphere", "cylinder"] = "auto"
    units: str = "m"
    grid_resolution: list[int] = Field(default_factory=lambda: [24, 24, 24])
    run_gmsh_sampler: bool = False
    gmsh_timeout_seconds: int = 10
    boundary_constraint_policy: Literal["penalty", "nitsche", "ife_enrichment"] = "penalty"


class PrepareGeometryAnalysisFilesOutput(StrictBaseModel):
    source_mode: Literal["stl", "generated_primitive"]
    stl: ArtifactRef
    summary: STLGeometrySummary
    background_grid: ArtifactRef
    embedding: ArtifactRef
    embedding_notes: ArtifactRef
    derivation_context: ArtifactRef
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _read_stl_triangles(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) >= 84:
        triangle_count = struct.unpack("<I", data[80:84])[0]
        expected = 84 + triangle_count * 50
        if expected == len(data):
            triangles = np.empty((triangle_count, 3, 3), dtype=float)
            offset = 84
            for index in range(triangle_count):
                offset += 12
                coords = struct.unpack("<9f", data[offset : offset + 36])
                triangles[index] = np.array(coords, dtype=float).reshape(3, 3)
                offset += 38
            return triangles

    vertices: list[list[float]] = []
    for raw_line in data.decode("utf-8", errors="ignore").splitlines():
        parts = raw_line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            try:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                continue
    if len(vertices) < 3:
        return np.empty((0, 3, 3), dtype=float)
    usable = len(vertices) - (len(vertices) % 3)
    return np.array(vertices[:usable], dtype=float).reshape((-1, 3, 3))


def _stl_summary(triangles: np.ndarray, units: str) -> STLGeometrySummary:
    warnings: list[str] = []
    if triangles.size == 0:
        warnings.append("STL parser found no triangles.")
        return STLGeometrySummary(triangle_count=0, bounds_min=[0.0, 0.0, 0.0], bounds_max=[0.0, 0.0, 0.0], units=units, warnings=warnings)
    points = triangles.reshape((-1, 3))
    return STLGeometrySummary(
        triangle_count=int(triangles.shape[0]),
        bounds_min=[float(value) for value in points.min(axis=0)],
        bounds_max=[float(value) for value in points.max(axis=0)],
        units=units,
        warnings=warnings,
    )


def generate_primitive_geometry(input: GeneratePrimitiveGeometryInput) -> GeneratePrimitiveGeometryOutput:
    """Generate a simple case-local STL from natural-language geometry text."""
    case_dir = _case_dir(input.case_id)
    geometry_dir = case_dir / "geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    primitive, parameters, warnings = _parse_primitive_description(input.description, input.primitive)
    triangles = _primitive_triangles(primitive, parameters, max(8, min(96, input.resolution)))
    stl_path = geometry_dir / input.target_filename
    stl_path.write_text(_ascii_stl(primitive, triangles), encoding="utf-8")
    summary = _stl_summary(triangles, input.units)
    summary_path = geometry_dir / "stl_summary.json"
    summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    request_payload = {
        "schema_version": "physicsos.generated_geometry_request.v1",
        "description": input.description,
        "primitive": primitive,
        "parameters": parameters,
        "units": input.units,
        "stl": to_agent_path(stl_path, workspace=_workspace()),
        "warnings": warnings,
    }
    request_path = geometry_dir / "generated_geometry.json"
    request_path.write_text(json.dumps(request_payload, indent=2), encoding="utf-8")
    _append_event(case_dir, "primitive_geometry_generated", {"primitive": primitive, "triangle_count": summary.triangle_count})
    return GeneratePrimitiveGeometryOutput(
        stl=_artifact(stl_path, "stl_geometry", "Generated case-local STL geometry."),
        summary=summary,
        summary_artifact=_artifact(summary_path, "stl_summary", "Parsed STL bounds and triangle count."),
        request_artifact=_artifact(request_path, "generated_geometry_request", "Natural-language geometry generation record."),
        warnings=warnings,
    )


def prepare_geometry_analysis_files(input: PrepareGeometryAnalysisFilesInput) -> PrepareGeometryAnalysisFilesOutput:
    """Prepare all geometry analysis files consumed by the paper-style TAPS prompt."""
    case_dir = _case_dir(input.case_id)
    warnings: list[str] = []
    if input.source_uri:
        source_mode: Literal["stl", "generated_primitive"] = "stl"
        imported = import_stl_geometry(ImportSTLGeometryInput(case_id=input.case_id, source_uri=input.source_uri, units=input.units))
        stl_artifact = imported.stl
        summary = imported.summary
        artifacts: list[ArtifactRef] = [imported.stl, imported.summary_artifact]
    elif input.description:
        source_mode = "generated_primitive"
        generated = generate_primitive_geometry(
            GeneratePrimitiveGeometryInput(
                case_id=input.case_id,
                description=input.description,
                primitive=input.primitive,
                units=input.units,
            )
        )
        stl_artifact = generated.stl
        summary = generated.summary
        warnings.extend(generated.warnings)
        artifacts = [generated.stl, generated.summary_artifact, generated.request_artifact]
    else:
        raise ValueError("Either source_uri or description is required for geometry analysis files.")

    gmsh = generate_gmsh_distance_field(GenerateGmshDistanceFieldInput(case_id=input.case_id, stl_uri=stl_artifact.uri))
    grid = generate_background_grid(
        GenerateBackgroundGridInput(
            case_id=input.case_id,
            bounds_min=summary.bounds_min,
            bounds_max=summary.bounds_max,
            resolution=input.grid_resolution,
        )
    )
    artifacts.extend([gmsh.gmsh_geo, gmsh.distance_field_manifest, grid.background_grid])
    warnings.extend(gmsh.warnings)
    if input.run_gmsh_sampler:
        sampled = execute_gmsh_distance_field(
            ExecuteGmshDistanceFieldInput(
                case_id=input.case_id,
                stl_uri=stl_artifact.uri,
                background_grid_uri=grid.background_grid.uri,
                timeout_seconds=input.gmsh_timeout_seconds,
                fallback_to_vertex_sdf=True,
            )
        )
        artifacts.extend([sampled.sdf, sampled.manifest, sampled.execution_log])
        warnings.extend(sampled.warnings)
    voxelized = voxelize_geometry(VoxelizeGeometryInput(case_id=input.case_id, stl_uri=stl_artifact.uri, background_grid_uri=grid.background_grid.uri))
    embedding = build_geometry_embedding(
        BuildGeometryEmbeddingInput(case_id=input.case_id, boundary_constraint_policy=input.boundary_constraint_policy)
    )
    artifacts.extend([voxelized.sdf, voxelized.occupancy, voxelized.boundary_samples, voxelized.normals, voxelized.cut_cells, voxelized.quality, embedding.embedding, embedding.embedding_notes, embedding.handoff_notes])
    warnings.extend(voxelized.warnings)
    warnings.extend(embedding.warnings)
    context = _write_geometry_derivation_context(input.case_id, source_mode, summary, warnings)
    _append_event(case_dir, "geometry_analysis_files_prepared", {"source_mode": source_mode})
    return PrepareGeometryAnalysisFilesOutput(
        source_mode=source_mode,
        stl=stl_artifact,
        summary=summary,
        background_grid=grid.background_grid,
        embedding=embedding.embedding,
        embedding_notes=embedding.embedding_notes,
        derivation_context=context,
        artifacts=artifacts,
        warnings=warnings,
    )


def _write_geometry_derivation_context(case_id: str, source_mode: str, summary: STLGeometrySummary, warnings: list[str]) -> ArtifactRef:
    geometry_dir = _case_dir(case_id) / "geometry"
    path = geometry_dir / "taps_geometry_context.md"
    lines = [
        "# TAPS Geometry Context",
        "",
        "This file is an analysis-file input for the paper-style TAPS derivation prompt.",
        "It is a PhysicsOS geometry extension to the paper route only; it does not define a separate solver workflow.",
        "",
        f"- Source mode: `{source_mode}`",
        f"- Units: `{summary.units}`",
        f"- Triangle count: `{summary.triangle_count}`",
        f"- Bounds min: `{summary.bounds_min}`",
        f"- Bounds max: `{summary.bounds_max}`",
        "",
        "Required derivation usage:",
        "- Treat `phi(x)` as the signed distance / level-set field from `geometry/sdf.npy` or `geometry/gmsh_sdf.npy` when available.",
        "- Treat `chi(x)=H(-phi(x))` as the physical-domain characteristic function from `geometry/occupancy.npy`.",
        "- Use `boundary_samples.npy` and `normals.npy` for immersed boundary terms.",
        "- Use `cut_cells.npy` to identify near-boundary cells for corrected quadrature or IFE enrichment.",
        "- Read `sdf_quality.json` before trusting SDF/occupancy artifacts; use it to report whether production SDF generation is still needed.",
        "- Keep these geometry terms as coefficients/boundary terms in the TAPS derivation; do not solve the PDE in the geometry module.",
        "- Leave PDE derivation, case-local implementation, and Fig. 7 verification to the TAPS derivation, implementation, and verification agents.",
        "",
    ]
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return _artifact(path, "taps_geometry_context", "Geometry analysis file for TAPS derivation prompt.")


def _parse_primitive_description(description: str, primitive: str) -> tuple[str, dict[str, float], list[str]]:
    text = description.lower()
    numbers = [float(item) for item in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", text)]
    warnings: list[str] = []
    selected = primitive
    if selected == "auto":
        if "sphere" in text or "ball" in text:
            selected = "sphere"
        elif "cylinder" in text or "pipe" in text:
            selected = "cylinder"
        else:
            selected = "box"
    if selected == "sphere":
        radius = _value_after(text, ("radius", "r"), numbers[0] if numbers else 0.5)
        if "diameter" in text and numbers:
            radius = _value_after(text, ("diameter",), numbers[0]) / 2.0
        return "sphere", {"radius": max(radius, 1e-9)}, warnings
    if selected == "cylinder":
        radius = _value_after(text, ("radius", "r"), numbers[0] if numbers else 0.5)
        height = _value_after(text, ("height", "length", "h"), numbers[1] if len(numbers) > 1 else 1.0)
        if "diameter" in text and numbers:
            radius = _value_after(text, ("diameter",), numbers[0]) / 2.0
        return "cylinder", {"radius": max(radius, 1e-9), "height": max(height, 1e-9)}, warnings
    if len(numbers) >= 3:
        length, width, height = numbers[:3]
    elif len(numbers) == 1:
        length = width = height = numbers[0]
    else:
        length = width = height = 1.0
        warnings.append("No explicit dimensions found; generated a unit box.")
    return "box", {"length": max(length, 1e-9), "width": max(width, 1e-9), "height": max(height, 1e-9)}, warnings


def _value_after(text: str, names: tuple[str, ...], fallback: float) -> float:
    for name in names:
        match = re.search(rf"{name}\s*[:=]?\s*([-+]?(?:\d*\.\d+|\d+))", text)
        if match:
            return float(match.group(1))
    return fallback


def _primitive_triangles(primitive: str, parameters: dict[str, float], resolution: int) -> np.ndarray:
    if primitive == "sphere":
        return _sphere_triangles(parameters["radius"], resolution)
    if primitive == "cylinder":
        return _cylinder_triangles(parameters["radius"], parameters["height"], resolution)
    return _box_triangles(parameters["length"], parameters["width"], parameters["height"])


def _box_triangles(length: float, width: float, height: float) -> np.ndarray:
    x0, x1 = 0.0, length
    y0, y1 = 0.0, width
    z0, z1 = 0.0, height
    v = {
        "000": [x0, y0, z0],
        "100": [x1, y0, z0],
        "110": [x1, y1, z0],
        "010": [x0, y1, z0],
        "001": [x0, y0, z1],
        "101": [x1, y0, z1],
        "111": [x1, y1, z1],
        "011": [x0, y1, z1],
    }
    faces = [
        ("000", "110", "100"), ("000", "010", "110"),
        ("001", "101", "111"), ("001", "111", "011"),
        ("000", "100", "101"), ("000", "101", "001"),
        ("010", "011", "111"), ("010", "111", "110"),
        ("000", "001", "011"), ("000", "011", "010"),
        ("100", "110", "111"), ("100", "111", "101"),
    ]
    return np.array([[v[a], v[b], v[c]] for a, b, c in faces], dtype=float)


def _sphere_triangles(radius: float, resolution: int) -> np.ndarray:
    lat_steps = max(4, resolution // 2)
    lon_steps = max(8, resolution)
    triangles: list[list[list[float]]] = []
    for i in range(lat_steps):
        theta0 = math.pi * i / lat_steps
        theta1 = math.pi * (i + 1) / lat_steps
        for j in range(lon_steps):
            phi0 = 2.0 * math.pi * j / lon_steps
            phi1 = 2.0 * math.pi * (j + 1) / lon_steps
            p00 = _sphere_point(radius, theta0, phi0)
            p01 = _sphere_point(radius, theta0, phi1)
            p10 = _sphere_point(radius, theta1, phi0)
            p11 = _sphere_point(radius, theta1, phi1)
            if i > 0:
                triangles.append([p00, p10, p01])
            if i < lat_steps - 1:
                triangles.append([p01, p10, p11])
    return np.array(triangles, dtype=float) + np.array([radius, radius, radius])


def _sphere_point(radius: float, theta: float, phi: float) -> list[float]:
    return [
        radius * math.sin(theta) * math.cos(phi),
        radius * math.sin(theta) * math.sin(phi),
        radius * math.cos(theta),
    ]


def _cylinder_triangles(radius: float, height: float, resolution: int) -> np.ndarray:
    steps = max(8, resolution)
    bottom = np.array([radius, radius, 0.0])
    top = np.array([radius, radius, height])
    triangles: list[list[list[float]]] = []
    for i in range(steps):
        a0 = 2.0 * math.pi * i / steps
        a1 = 2.0 * math.pi * (i + 1) / steps
        p0 = np.array([radius + radius * math.cos(a0), radius + radius * math.sin(a0), 0.0])
        p1 = np.array([radius + radius * math.cos(a1), radius + radius * math.sin(a1), 0.0])
        q0 = p0 + np.array([0.0, 0.0, height])
        q1 = p1 + np.array([0.0, 0.0, height])
        triangles.append([p0.tolist(), p1.tolist(), q0.tolist()])
        triangles.append([p1.tolist(), q1.tolist(), q0.tolist()])
        triangles.append([bottom.tolist(), p0.tolist(), p1.tolist()])
        triangles.append([top.tolist(), q1.tolist(), q0.tolist()])
    return np.array(triangles, dtype=float)


def _ascii_stl(name: str, triangles: np.ndarray) -> str:
    lines = [f"solid {name}"]
    for triangle in triangles:
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
        lines.append(f"facet normal {normal[0]:.12g} {normal[1]:.12g} {normal[2]:.12g}")
        lines.append("  outer loop")
        for vertex in triangle:
            lines.append(f"    vertex {vertex[0]:.12g} {vertex[1]:.12g} {vertex[2]:.12g}")
        lines.append("  endloop")
        lines.append("endfacet")
    lines.append(f"endsolid {name}")
    return "\n".join(lines) + "\n"


def import_stl_geometry(input: ImportSTLGeometryInput) -> ImportSTLGeometryOutput:
    """Copy an STL into the case geometry directory and write a parse summary."""
    case_dir = _case_dir(input.case_id)
    geometry_dir = case_dir / "geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    source = resolve_workspace_path(input.source_uri, workspace=runtime_paths().workspace)
    if not source.exists():
        raise FileNotFoundError(f"STL file does not exist: {input.source_uri}")
    target = geometry_dir / input.target_filename
    if source.resolve() != target.resolve():
        copyfile(source, target)
    triangles = _read_stl_triangles(target)
    summary = _stl_summary(triangles, input.units)
    summary_path = geometry_dir / "stl_summary.json"
    summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    _append_event(case_dir, "stl_geometry_imported", {"source": input.source_uri, "triangle_count": summary.triangle_count})
    return ImportSTLGeometryOutput(
        stl=_artifact(target, "stl_geometry", "Case-local STL geometry."),
        summary=summary,
        summary_artifact=_artifact(summary_path, "stl_summary", "Parsed STL bounds and triangle count."),
    )


class GenerateGmshDistanceFieldInput(StrictBaseModel):
    case_id: str
    stl_uri: str | None = None
    distance_sampling: int = 100


class GenerateGmshDistanceFieldOutput(StrictBaseModel):
    gmsh_geo: ArtifactRef
    distance_field_manifest: ArtifactRef
    warnings: list[str] = Field(default_factory=list)


def generate_gmsh_distance_field(input: GenerateGmshDistanceFieldInput) -> GenerateGmshDistanceFieldOutput:
    """Write Gmsh preprocessing files for STL distance-field construction."""
    case_dir = _case_dir(input.case_id)
    geometry_dir = case_dir / "geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    stl_uri = input.stl_uri or f"/workspace/cases/{input.case_id}/geometry/input.stl"
    stl_path = resolve_workspace_path(stl_uri, workspace=runtime_paths().workspace)
    geo_path = geometry_dir / "gmsh_model.geo"
    geo_path.write_text(
        "\n".join(
            [
                'SetFactory("OpenCASCADE");',
                f'Merge "{stl_path.as_posix()}";',
                "Field[1] = Distance;",
                "Field[1].FacesList = {1};",
                f"Field[1].Sampling = {max(1, input.distance_sampling)};",
                "Background Field = 1;",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "physicsos.gmsh_distance_field.v1",
        "source_stl": to_agent_path(stl_path, workspace=_workspace()),
        "gmsh_geo": to_agent_path(geo_path, workspace=_workspace()),
        "distance_field": "Gmsh Distance field declaration for downstream SDF/geometry-parameter preprocessing.",
        "execution_policy": "prepared_only; no external gmsh process was invoked by this tool",
    }
    manifest_path = geometry_dir / "gmsh_distance_field.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    warnings = [] if stl_path.exists() else [f"STL path does not exist yet: {stl_uri}"]
    _append_event(case_dir, "gmsh_distance_field_prepared", {"gmsh_geo": to_agent_path(geo_path, workspace=_workspace())})
    return GenerateGmshDistanceFieldOutput(
        gmsh_geo=_artifact(geo_path, "gmsh_geo", "Gmsh STL distance-field preprocessing file."),
        distance_field_manifest=_artifact(manifest_path, "gmsh_distance_field_manifest", "Gmsh distance-field preprocessing manifest."),
        warnings=warnings,
    )


class ExecuteGmshDistanceFieldInput(StrictBaseModel):
    case_id: str
    stl_uri: str | None = None
    background_grid_uri: str | None = None
    timeout_seconds: int = 30
    fallback_to_vertex_sdf: bool = True


class ExecuteGmshDistanceFieldOutput(StrictBaseModel):
    sdf: ArtifactRef
    manifest: ArtifactRef
    execution_log: ArtifactRef
    status: Literal["success", "fallback", "failed"]
    warnings: list[str] = Field(default_factory=list)


def execute_gmsh_distance_field(input: ExecuteGmshDistanceFieldInput) -> ExecuteGmshDistanceFieldOutput:
    """Sample a Gmsh-backed STL distance field on the Cartesian background grid.

    The preferred path runs a short case-local Python script using the `gmsh`
    module. If Gmsh is unavailable or times out, the tool can fall back to the
    existing vertex-distance SDF so downstream TAPS artifacts remain explicit.
    """
    case_dir = _case_dir(input.case_id)
    geometry_dir = case_dir / "geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    stl_uri = input.stl_uri or f"/workspace/cases/{input.case_id}/geometry/input.stl"
    grid_uri = input.background_grid_uri or f"/workspace/cases/{input.case_id}/geometry/background_grid.json"
    stl_path = resolve_workspace_path(stl_uri, workspace=runtime_paths().workspace)
    grid_path = resolve_workspace_path(grid_uri, workspace=runtime_paths().workspace)
    script_path = geometry_dir / "gmsh_distance_sampler.py"
    output_path = geometry_dir / "gmsh_sdf.npy"
    script_path.write_text(_gmsh_distance_sampler_script(), encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    try:
        completed = subprocess.run(
            [sys.executable, str(script_path), str(stl_path), str(grid_path), str(output_path)],
            cwd=str(geometry_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=max(1, input.timeout_seconds),
        )
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        completed = subprocess.CompletedProcess(
            args=[sys.executable, str(script_path), str(stl_path), str(grid_path), str(output_path)],
            returncode=124,
            stdout=exc.stdout or "",
            stderr=exc.stderr or f"Timed out after {input.timeout_seconds}s.",
        )
        timed_out = True
    log_payload = {
        "schema_version": "physicsos.gmsh_distance_execution_log.v1",
        "script": to_agent_path(script_path, workspace=_workspace()),
        "stl": to_agent_path(stl_path, workspace=_workspace()) if stl_path.exists() else stl_uri,
        "background_grid": to_agent_path(grid_path, workspace=_workspace()) if grid_path.exists() else grid_uri,
        "returncode": completed.returncode,
        "timed_out": timed_out,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    log_path = geometry_dir / "gmsh_distance_execution_log.json"
    log_path.write_text(json.dumps(log_payload, indent=2), encoding="utf-8")
    warnings: list[str] = []
    status: Literal["success", "fallback", "failed"]
    if completed.returncode == 0 and output_path.exists():
        status = "success"
    elif input.fallback_to_vertex_sdf:
        triangles = _read_stl_triangles(stl_path)
        points, shape, _grid_payload = _load_background_grid(grid_path)
        distance = _distance_to_vertices(points, triangles)
        inside = _inside_stl(points, triangles)
        sdf = np.where(inside, -distance, distance).reshape(shape)
        np.save(output_path, sdf)
        status = "fallback"
        warnings.append("Gmsh distance sampling failed or timed out; wrote vertex-distance fallback to gmsh_sdf.npy.")
    else:
        status = "failed"
        warnings.append("Gmsh distance sampling failed and fallback_to_vertex_sdf=false.")
    manifest_payload = {
        "schema_version": "physicsos.gmsh_sampled_sdf.v1",
        "case_id": input.case_id,
        "status": status,
        "sdf": to_agent_path(output_path, workspace=_workspace()) if output_path.exists() else None,
        "source_stl": to_agent_path(stl_path, workspace=_workspace()) if stl_path.exists() else stl_uri,
        "background_grid": to_agent_path(grid_path, workspace=_workspace()) if grid_path.exists() else grid_uri,
        "execution_log": to_agent_path(log_path, workspace=_workspace()),
        "preferred_method": "gmsh.model.getClosestPoint sampled on background_grid.json",
        "fallback_method": "signed vertex-distance approximation using STL ray-cast inside/outside",
        "warnings": warnings,
    }
    manifest_path = geometry_dir / "gmsh_sampled_sdf.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    _append_event(case_dir, "gmsh_distance_field_executed", {"status": status})
    return ExecuteGmshDistanceFieldOutput(
        sdf=_artifact(output_path, "gmsh_sampled_sdf", "Sampled signed distance field from Gmsh or explicit fallback."),
        manifest=_artifact(manifest_path, "gmsh_sampled_sdf_manifest", "Gmsh sampled SDF execution manifest."),
        execution_log=_artifact(log_path, "gmsh_distance_execution_log", "Execution log for Gmsh distance sampling."),
        status=status,
        warnings=warnings,
    )


def _gmsh_distance_sampler_script() -> str:
    return r'''from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


def _load_points(grid_path: Path):
    payload = json.loads(grid_path.read_text(encoding="utf-8"))
    axes = payload["axes"]
    x = np.array(axes["x"], dtype=float)
    y = np.array(axes["y"], dtype=float)
    z = np.array(axes["z"], dtype=float)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]), (len(x), len(y), len(z))


def _closest_distance(gmsh, entities, point):
    best = math.inf
    coords = [float(point[0]), float(point[1]), float(point[2])]
    for dim, tag in entities:
        if dim != 2:
            continue
        try:
            closest, _param = gmsh.model.getClosestPoint(dim, tag, coords)
        except Exception:
            continue
        diff = np.array(closest, dtype=float) - point
        best = min(best, float(np.linalg.norm(diff)))
    return best


def main():
    stl_path = Path(sys.argv[1])
    grid_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.merge(str(stl_path))
        try:
            gmsh.model.mesh.classifySurfaces(40.0 * math.pi / 180.0, True, True, math.pi)
            gmsh.model.mesh.createGeometry()
        except Exception:
            pass
        entities = gmsh.model.getEntities(2)
        if not entities:
            entities = gmsh.model.getEntities()
        points, shape = _load_points(grid_path)
        sdf = np.empty(points.shape[0], dtype=float)
        for index, point in enumerate(points):
            sdf[index] = _closest_distance(gmsh, entities, point)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, sdf.reshape(shape))
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
'''


class GenerateBackgroundGridInput(StrictBaseModel):
    case_id: str
    bounds_min: list[float]
    bounds_max: list[float]
    resolution: list[int] = Field(default_factory=lambda: [24, 24, 24])
    padding_fraction: float = 0.05


class GenerateBackgroundGridOutput(StrictBaseModel):
    background_grid: ArtifactRef
    axes: dict[str, list[float]]
    warnings: list[str] = Field(default_factory=list)


def _padded_bounds(bounds_min: list[float], bounds_max: list[float], padding_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    lower = np.array(bounds_min, dtype=float)
    upper = np.array(bounds_max, dtype=float)
    span = np.maximum(upper - lower, 1.0)
    padding = max(0.0, padding_fraction) * span
    return lower - padding, upper + padding


def generate_background_grid(input: GenerateBackgroundGridInput) -> GenerateBackgroundGridOutput:
    """Generate a Cartesian background grid descriptor for IBM/IFE TAPS."""
    if len(input.bounds_min) != 3 or len(input.bounds_max) != 3 or len(input.resolution) != 3:
        raise ValueError("bounds_min, bounds_max, and resolution must each have length 3.")
    resolution = [max(2, min(128, int(value))) for value in input.resolution]
    lower, upper = _padded_bounds(input.bounds_min, input.bounds_max, input.padding_fraction)
    axes = {
        "x": [float(value) for value in np.linspace(lower[0], upper[0], resolution[0])],
        "y": [float(value) for value in np.linspace(lower[1], upper[1], resolution[1])],
        "z": [float(value) for value in np.linspace(lower[2], upper[2], resolution[2])],
    }
    payload = {
        "schema_version": "physicsos.background_grid.v1",
        "bounds_min": [float(value) for value in lower],
        "bounds_max": [float(value) for value in upper],
        "resolution": resolution,
        "axis_names": ["x", "y", "z"],
        "axes": axes,
        "purpose": "Cartesian background grid for immersed-boundary / IFE TAPS.",
    }
    case_dir = _case_dir(input.case_id)
    path = case_dir / "geometry" / "background_grid.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _append_event(case_dir, "background_grid_generated", {"resolution": resolution})
    return GenerateBackgroundGridOutput(
        background_grid=_artifact(path, "background_grid", "Cartesian background grid for IBM/IFE TAPS."),
        axes=axes,
    )


class VoxelizeGeometryInput(StrictBaseModel):
    case_id: str
    stl_uri: str | None = None
    background_grid_uri: str | None = None


class VoxelizeGeometryOutput(StrictBaseModel):
    sdf: ArtifactRef
    occupancy: ArtifactRef
    boundary_samples: ArtifactRef
    normals: ArtifactRef
    cut_cells: ArtifactRef
    quality: ArtifactRef
    warnings: list[str] = Field(default_factory=list)


def _distance_to_vertices(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    vertices = triangles.reshape((-1, 3))
    if vertices.size == 0:
        return np.full(points.shape[0], math.inf)
    distances = np.full(points.shape[0], math.inf)
    chunk = 2048
    for start in range(0, points.shape[0], chunk):
        segment = points[start : start + chunk]
        diff = segment[:, None, :] - vertices[None, :, :]
        distances[start : start + chunk] = np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))
    return distances


def _ray_intersects_triangle(point: np.ndarray, triangle: np.ndarray) -> bool:
    epsilon = 1e-12
    direction = np.array([1.0, 0.0, 0.0])
    v0, v1, v2 = triangle
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(direction, edge2)
    a = float(np.dot(edge1, h))
    if -epsilon < a < epsilon:
        return False
    f = 1.0 / a
    s = point - v0
    u = f * float(np.dot(s, h))
    if u < 0.0 or u > 1.0:
        return False
    q = np.cross(s, edge1)
    v = f * float(np.dot(direction, q))
    if v < 0.0 or u + v > 1.0:
        return False
    t = f * float(np.dot(edge2, q))
    return t > epsilon


def _inside_stl(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    inside = np.zeros(points.shape[0], dtype=bool)
    if triangles.shape[0] == 0:
        return inside
    for index, point in enumerate(points):
        hits = sum(1 for triangle in triangles if _ray_intersects_triangle(point, triangle))
        inside[index] = hits % 2 == 1
    return inside


def _load_background_grid(path: Path) -> tuple[np.ndarray, tuple[int, int, int], dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    axes = payload.get("axes")
    if not isinstance(axes, dict):
        raise ValueError("background_grid.json is missing axes.")
    x = np.array(axes["x"], dtype=float)
    y = np.array(axes["y"], dtype=float)
    z = np.array(axes["z"], dtype=float)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    return points, (len(x), len(y), len(z)), payload


def voxelize_geometry(input: VoxelizeGeometryInput) -> VoxelizeGeometryOutput:
    """Generate SDF, occupancy, boundary samples, normals, and cut-cell artifacts."""
    case_dir = _case_dir(input.case_id)
    geometry_dir = case_dir / "geometry"
    stl_uri = input.stl_uri or f"/workspace/cases/{input.case_id}/geometry/input.stl"
    grid_uri = input.background_grid_uri or f"/workspace/cases/{input.case_id}/geometry/background_grid.json"
    stl_path = resolve_workspace_path(stl_uri, workspace=runtime_paths().workspace)
    grid_path = resolve_workspace_path(grid_uri, workspace=runtime_paths().workspace)
    triangles = _read_stl_triangles(stl_path)
    points, shape, grid_payload = _load_background_grid(grid_path)
    distance = _distance_to_vertices(points, triangles)
    inside = _inside_stl(points, triangles)
    sdf = np.where(inside, -distance, distance).reshape(shape)
    occupancy = inside.reshape(shape).astype(np.uint8)
    centroids = triangles.mean(axis=1) if triangles.size else np.empty((0, 3), dtype=float)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]) if triangles.size else np.empty((0, 3), dtype=float)
    normal_norms = np.linalg.norm(normals, axis=1) if normals.size else np.empty((0,), dtype=float)
    valid = normal_norms > 0
    if normals.size:
        normals[valid] = normals[valid] / normal_norms[valid, None]
    cut_cells = np.argwhere(np.abs(sdf) <= _grid_spacing(grid_payload) * 1.5)

    sdf_path = geometry_dir / "sdf.npy"
    occupancy_path = geometry_dir / "occupancy.npy"
    samples_path = geometry_dir / "boundary_samples.npy"
    normals_path = geometry_dir / "normals.npy"
    cut_cells_path = geometry_dir / "cut_cells.npy"
    quality_path = geometry_dir / "sdf_quality.json"
    np.save(sdf_path, sdf)
    np.save(occupancy_path, occupancy)
    np.save(samples_path, centroids)
    np.save(normals_path, normals)
    np.save(cut_cells_path, cut_cells)
    spacing = _grid_spacing(grid_payload)
    quality_payload = _sdf_quality_payload(
        method="vertex_distance_ray_cast",
        shape=shape,
        sdf=sdf,
        occupancy=occupancy,
        normals=normals,
        normal_norms=normal_norms,
        cut_cells=cut_cells,
        spacing=spacing,
        triangles=triangles,
    )
    quality_path.write_text(json.dumps(quality_payload, indent=2), encoding="utf-8")
    warnings = [
        "SDF is a lightweight vertex-distance approximation; use Gmsh/OpenVDB distance generation for production accuracy."
    ]
    _append_event(case_dir, "geometry_voxelized", {"grid_shape": list(shape), "cut_cell_count": int(cut_cells.shape[0])})
    return VoxelizeGeometryOutput(
        sdf=_artifact(sdf_path, "sdf", "Signed distance field on the Cartesian background grid."),
        occupancy=_artifact(occupancy_path, "occupancy_mask", "Voxel occupancy mask for chi(x)."),
        boundary_samples=_artifact(samples_path, "boundary_samples", "STL triangle centroids for boundary coupling."),
        normals=_artifact(normals_path, "surface_normals", "STL triangle normals for boundary terms."),
        cut_cells=_artifact(cut_cells_path, "cut_cells", "Near-boundary grid indices for cut-cell quadrature."),
        quality=_artifact(quality_path, "sdf_quality", "SDF and voxelization quality metrics for geometry handoff."),
        warnings=warnings,
    )


def _sdf_quality_payload(
    *,
    method: str,
    shape: tuple[int, int, int],
    sdf: np.ndarray,
    occupancy: np.ndarray,
    normals: np.ndarray,
    normal_norms: np.ndarray,
    cut_cells: np.ndarray,
    spacing: float,
    triangles: np.ndarray,
) -> dict[str, object]:
    total = int(occupancy.size)
    inside = int(occupancy.sum())
    outside = total - inside
    finite_sdf = sdf[np.isfinite(sdf)]
    zero_normals = int(np.sum(normal_norms <= 0)) if normal_norms.size else 0
    warnings: list[str] = []
    if total == 0 or finite_sdf.size == 0:
        warnings.append("No finite SDF samples were generated.")
    if inside == 0:
        warnings.append("No occupied grid samples detected; check STL closure, grid bounds, or ray-cast direction.")
    if outside == 0:
        warnings.append("All grid samples are occupied; grid bounds may not include exterior padding.")
    if zero_normals:
        warnings.append(f"{zero_normals} degenerate STL triangle normals were detected.")
    if cut_cells.size == 0:
        warnings.append("No cut-cell candidates detected at the current grid resolution.")
    return {
        "schema_version": "physicsos.sdf_quality.v1",
        "method": method,
        "production_ready": False,
        "recommended_production_methods": ["Gmsh sampled distance field", "OpenVDB signed distance field", "robust winding-number SDF"],
        "grid_shape": list(shape),
        "grid_spacing_min": float(spacing),
        "triangle_count": int(triangles.shape[0]),
        "sample_count": total,
        "inside_count": inside,
        "outside_count": outside,
        "occupancy_fraction": float(inside / total) if total else 0.0,
        "sdf_min": float(np.min(finite_sdf)) if finite_sdf.size else None,
        "sdf_max": float(np.max(finite_sdf)) if finite_sdf.size else None,
        "cut_cell_count": int(cut_cells.shape[0]),
        "normal_count": int(normals.shape[0]),
        "degenerate_normal_count": zero_normals,
        "warnings": warnings,
    }


def _grid_spacing(grid_payload: dict[str, object]) -> float:
    axes = grid_payload.get("axes")
    if not isinstance(axes, dict):
        return 1.0
    spacings: list[float] = []
    for name in ("x", "y", "z"):
        values = axes.get(name)
        if isinstance(values, list) and len(values) > 1:
            spacings.append(abs(float(values[1]) - float(values[0])))
    return min(spacings) if spacings else 1.0


class BuildGeometryEmbeddingInput(StrictBaseModel):
    case_id: str
    boundary_constraint_policy: Literal["penalty", "nitsche", "ife_enrichment"] = "penalty"
    quadrature_policy: str = "cut_cell_candidates_from_sdf_band"


class BuildGeometryEmbeddingOutput(StrictBaseModel):
    embedding: ArtifactRef
    embedding_notes: ArtifactRef
    handoff_notes: ArtifactRef
    warnings: list[str] = Field(default_factory=list)


def build_geometry_embedding(input: BuildGeometryEmbeddingInput) -> BuildGeometryEmbeddingOutput:
    """Assemble the reviewable IBM/IFE geometry embedding contract."""
    case_dir = _case_dir(input.case_id)
    geometry_dir = case_dir / "geometry"
    artifact_paths = {
        "source_stl": geometry_dir / "input.stl",
        "gmsh_model": geometry_dir / "gmsh_model.geo",
        "background_grid": geometry_dir / "background_grid.json",
        "sdf": geometry_dir / "sdf.npy",
        "sdf_quality": geometry_dir / "sdf_quality.json",
        "occupancy": geometry_dir / "occupancy.npy",
        "boundary_samples": geometry_dir / "boundary_samples.npy",
        "normals": geometry_dir / "normals.npy",
        "cut_cells": geometry_dir / "cut_cells.npy",
        "gmsh_distance_field": geometry_dir / "gmsh_distance_field.json",
    }
    missing = [name for name, path in artifact_paths.items() if not path.exists() and name not in {"gmsh_model", "gmsh_distance_field"}]
    payload = {
        "schema_version": "physicsos.geometry_embedding.v1",
        "geometry_id": input.case_id,
        "method": "immersed_boundary_ife_taps",
        "paper_route_role": "geometry_analysis_file_extension",
        "sdf_convention": "phi(x) <= 0 is inside the STL domain",
        "chi_definition": "chi(x) = H(-phi(x)); occupancy.npy stores the discrete indicator.",
        "artifacts": {name: to_agent_path(path, workspace=_workspace()) for name, path in artifact_paths.items() if path.exists()},
        "parameter_axes": [
            {
                "name": "alpha_g",
                "description": "Optional geometry parameter axis from SDF/Gmsh distance-field snapshots.",
                "status": "prepared_not_fitted",
            }
        ],
        "quadrature_policy": input.quadrature_policy,
        "boundary_constraint_policy": input.boundary_constraint_policy,
        "weak_form_coupling": [
            "Integrate volume residuals over Omega_bg with chi-weighted coefficients.",
            "Apply Dirichlet constraints through penalty/Nitsche/IFE terms on phi-derived boundary samples.",
            "Use normals.npy for Neumann and Nitsche consistency terms.",
            "Use cut_cells.npy to trigger corrected quadrature or enrichment near immersed boundaries.",
        ],
        "agent_handoff": {
            "context_window": f"/workspace/cases/{input.case_id}/geometry/taps_geometry_handoff.md",
            "taps_derivation_agent": [
                "Treat geometry as embedded-domain coefficients in the Galerkin weak form, not as a separate solver.",
                "Introduce chi-weighted volume integrals on Omega_bg.",
                "Introduce phi-derived boundary terms for Dirichlet/Neumann constraints.",
                "Define geometry matrices and any geometry parameter axis before subspace iteration.",
            ],
            "taps_implementation_agent": [
                "Load background_grid.json, sdf.npy or gmsh_sdf.npy, occupancy.npy, boundary_samples.npy, normals.npy, and cut_cells.npy from the case geometry directory.",
                "Read sdf_quality.json and surface warnings in runtime_metadata.json or verification metadata.",
                "Validate array shapes and grid alignment before assembling geometry-weighted matrices.",
                "Keep all geometry coupling traceable to derivation.md and implementation_notes.md.",
                "Fail clearly if required geometry arrays are missing or inconsistent.",
            ],
            "verification_agent": [
                "Report which geometry artifacts were consumed by the generated kernel.",
                "Check that boundary samples/normals and occupancy/SDF are present for immersed-boundary verification.",
                "Report SDF quality status and whether the case used fallback vertex-distance SDF or production Gmsh/OpenVDB-style SDF.",
                "For manufactured solutions, state whether errors are measured on Omega_bg or the chi-weighted physical domain.",
                "Report missing geometry evidence explicitly instead of treating geometry preprocessing as verification.",
            ],
        },
        "missing_required_artifacts": missing,
    }
    embedding_path = geometry_dir / "embedding.json"
    embedding_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    notes_path = geometry_dir / "geometry_embedding.md"
    handoff_path = geometry_dir / "taps_geometry_handoff.md"
    notes_path.write_text(
        "\n".join(
            [
                "# Geometry Embedding",
                "",
                "Route: STL/CAD -> Gmsh preprocessing -> Cartesian background grid -> SDF/voxel occupancy -> IBM/IFE TAPS weak-form coupling.",
                "",
                "Scope: this is the PhysicsOS geometry extension to the paper's prompt-engineered TAPS workflow. It supplies analysis files for derivation; it is not a PDE solver, not a TAPS implementation, and not verification evidence.",
                "",
                "SDF convention: `phi(x) <= 0` is inside the physical domain.",
                "",
                "Characteristic function: `chi(x) = H(-phi(x))` and `occupancy.npy` stores the discrete indicator.",
                "",
                f"Boundary constraint policy: `{input.boundary_constraint_policy}`.",
                "",
                "Weak-form coupling notes:",
                "- Volume integrals are chi-weighted on the background grid.",
                "- Boundary terms use phi-derived samples and normals.",
                "- Cut-cell indices mark where corrected quadrature or IFE enrichment is needed.",
                "- Geometry parameters can be introduced as SDF or Gmsh distance-field snapshots.",
                "",
                "Missing required artifacts: " + (", ".join(missing) if missing else "none"),
                "",
            ]
        ),
        encoding="utf-8",
    )
    handoff_path.write_text(
        "\n".join(
            [
                "# TAPS Geometry Handoff",
                "",
                "This is the geometry module's handoff into the paper-style TAPS loop. It maximizes STL/CAD usefulness while keeping the non-geometry framework aligned with the paper: prompt-engineered derivation, case-local implementation, Fig. 7 verification, and revision.",
                "",
                "Scope:",
                "- Geometry supplies analysis files and coefficients for the TAPS prompt context.",
                "- Geometry does not solve the PDE, generate the TAPS kernel, or verify numerical accuracy.",
                "- Gmsh is used for geometry and distance-field preprocessing only.",
                "",
                "Artifacts for context window:",
                f"- Embedding contract: `/workspace/cases/{input.case_id}/geometry/embedding.json`",
                f"- Geometry notes: `/workspace/cases/{input.case_id}/geometry/geometry_embedding.md`",
                f"- Geometry derivation context: `/workspace/cases/{input.case_id}/geometry/taps_geometry_context.md`",
                "- Background grid: `geometry/background_grid.json`",
                "- Level set / SDF: `geometry/sdf.npy` or `geometry/gmsh_sdf.npy`",
                "- SDF quality report: `geometry/sdf_quality.json`",
                "- Characteristic function: `geometry/occupancy.npy`",
                "- Boundary samples: `geometry/boundary_samples.npy`",
                "- Boundary normals: `geometry/normals.npy`",
                "- Cut cells: `geometry/cut_cells.npy`",
                "",
                "Derivation-agent handoff:",
                "- Embed the STL domain in `Omega_bg` using `phi(x)` and `chi(x)=H(-phi(x))`.",
                "- Add chi-weighted volume terms to the weak form.",
                "- Add boundary constraint terms using phi-derived boundary samples and normals.",
                "- Define geometry matrices before they appear in subspace iterations.",
                "- If geometry is parameterized, introduce an explicit geometry axis from SDF/Gmsh distance snapshots.",
                "",
                "Implementation-agent handoff:",
                "- Load the geometry artifacts case-locally from `/workspace/cases/<case_id>/geometry/`.",
                "- Validate SDF/occupancy/background-grid shape consistency before assembly.",
                "- Read `sdf_quality.json` and propagate quality warnings into runtime metadata.",
                "- Assemble geometry-weighted matrices exactly as derived; do not invent geometry operators outside derivation.md.",
                "- If required geometry artifacts are missing, fail clearly and request geometry regeneration.",
                "",
                "Verification-agent handoff:",
                "- Record which geometry artifacts the generated kernel consumed.",
                "- Check that boundary samples/normals exist for boundary-condition verification.",
                "- Report SDF quality status and missing geometry evidence.",
                "- State whether relative L2 error is evaluated on the full background grid or chi-weighted physical domain.",
                "- Treat geometry preprocessing as input evidence only, not as numerical verification.",
                "",
                "Missing required artifacts: " + (", ".join(missing) if missing else "none"),
                "",
            ]
        ),
        encoding="utf-8",
    )
    _append_event(case_dir, "geometry_embedding_built", {"missing_required_artifacts": missing})
    return BuildGeometryEmbeddingOutput(
        embedding=_artifact(embedding_path, "geometry_embedding", "IBM/IFE TAPS geometry embedding contract."),
        embedding_notes=_artifact(notes_path, "geometry_embedding_notes", "Human-readable geometry embedding notes."),
        handoff_notes=_artifact(handoff_path, "taps_geometry_handoff", "Geometry handoff into derivation, implementation, and verification agents."),
        warnings=[f"Missing required geometry artifact: {name}" for name in missing],
    )


for _tool, _input, _output in [
    (prepare_geometry_analysis_files, PrepareGeometryAnalysisFilesInput, PrepareGeometryAnalysisFilesOutput),
    (generate_primitive_geometry, GeneratePrimitiveGeometryInput, GeneratePrimitiveGeometryOutput),
    (import_stl_geometry, ImportSTLGeometryInput, ImportSTLGeometryOutput),
    (generate_gmsh_distance_field, GenerateGmshDistanceFieldInput, GenerateGmshDistanceFieldOutput),
    (execute_gmsh_distance_field, ExecuteGmshDistanceFieldInput, ExecuteGmshDistanceFieldOutput),
    (generate_background_grid, GenerateBackgroundGridInput, GenerateBackgroundGridOutput),
    (voxelize_geometry, VoxelizeGeometryInput, VoxelizeGeometryOutput),
    (build_geometry_embedding, BuildGeometryEmbeddingInput, BuildGeometryEmbeddingOutput),
]:
    _tool.input_model = _input
    _tool.output_model = _output
    _tool.side_effects = "workspace artifacts only"
    _tool.requires_approval = False
