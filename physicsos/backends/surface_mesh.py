"""CPU surface checks shared by PhysicsOS and the standalone CUDA worker.

This module deliberately has no PhysicsOS imports: the worker can run in
PaMO's Python 3.10 environment without installing the agent stack.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import struct

import numpy as np


def validate_surface_arrays(vertices, faces) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float64)
    raw_faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("Surface vertices must be a nonempty (N, 3) array.")
    if raw_faces.ndim != 2 or raw_faces.shape[1] != 3 or not len(raw_faces):
        raise ValueError("Surface faces must be a nonempty (M, 3) array.")
    if not np.isfinite(vertices).all():
        raise ValueError("Surface contains nonfinite coordinates.")
    if raw_faces.dtype.kind not in "iu":
        raise ValueError("Surface indices must be integers.")
    faces = raw_faces.astype(np.int64)
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Surface contains out-of-range vertex indices.")
    return vertices, faces


def surface_metrics(vertices, faces) -> dict[str, object]:
    vertices, faces = validate_surface_arrays(vertices, faces)
    triangles = vertices[faces]
    extent = np.ptp(vertices, axis=0)
    diagonal = float(np.linalg.norm(extent))
    doubled_areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1,
    )
    lengths = np.stack([
        np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
        np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
        np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
    ], axis=1)
    eps = max(diagonal * diagonal * 1e-14, np.finfo(float).tiny)
    degenerate = doubled_areas <= eps
    ratios = lengths.max(axis=1) / np.maximum(lengths.min(axis=1), np.finfo(float).tiny)
    shape_quality = 2 * np.sqrt(3) * doubled_areas / np.maximum(
        (lengths * lengths).sum(axis=1), np.finfo(float).tiny,
    )
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    directions = np.where(edges[:, 0] < edges[:, 1], 1, -1)
    orientation_sum = np.bincount(inverse, weights=directions, minlength=len(counts))
    duplicates = len(faces) - len(np.unique(np.sort(faces, axis=1), axis=0))

    # Edge incidence alone misses two closed shells touching at one vertex.
    # A manifold vertex has one connected link (a cycle or a boundary path).
    links = defaultdict(list)
    for a, b, c in faces.tolist():
        links[a].append((b, c))
        links[b].append((c, a))
        links[c].append((a, b))
    nonmanifold_vertices = 0
    for link_edges in links.values():
        adjacency = defaultdict(set)
        degree = Counter()
        for a, b in link_edges:
            adjacency[a].add(b)
            adjacency[b].add(a)
            degree[a] += 1
            degree[b] += 1
        visited = set()
        pending = [next(iter(adjacency))]
        while pending:
            node = pending.pop()
            if node not in visited:
                visited.add(node)
                pending.extend(adjacency[node] - visited)
        boundary_ends = sum(value == 1 for value in degree.values())
        if len(visited) != len(adjacency) or any(value > 2 for value in degree.values()) or boundary_ends not in {0, 2}:
            nonmanifold_vertices += 1

    boundary_edges = int(np.count_nonzero(counts == 1))
    nonmanifold_edges = int(np.count_nonzero(counts > 2))
    return {
        "vertex_count": len(vertices), "face_count": len(faces),
        "bounds_min": vertices.min(axis=0).tolist(), "bounds_max": vertices.max(axis=0).tolist(),
        "bbox_diagonal": diagonal, "boundary_edges": boundary_edges,
        "nonmanifold_edges": nonmanifold_edges, "nonmanifold_vertices": nonmanifold_vertices,
        "duplicate_faces": int(duplicates), "degenerate_faces": int(degenerate.sum()),
        "watertight": bool(boundary_edges == 0 and nonmanifold_edges == 0 and not degenerate.any()),
        "manifold": bool(nonmanifold_edges == 0 and nonmanifold_vertices == 0 and duplicates == 0 and not degenerate.any()),
        "winding_consistent": bool(np.all(orientation_sum[counts == 2] == 0)),
        "min_doubled_area": float(doubled_areas.min()),
        "aspect_ratio_p95": float(np.percentile(ratios, 95)),
        "max_skewness": float(np.max(1 - np.clip(shape_quality, 0, 1))),
    }


_VOLUME_FACES = {
    "tetra": [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)],
    "hexahedron": [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)],
    "wedge": [(0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)],
    "pyramid": [(0, 3, 2, 1), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)],
}


def has_internal_surface_elements(mesh) -> bool:
    """Detect explicitly meshed interfaces from volume connectivity, not names."""
    incidence = Counter()
    for block in mesh.cells:
        kind = next((kind for kind in _VOLUME_FACES if block.type.startswith(kind)), None)
        if kind:
            for face in _VOLUME_FACES[kind]:
                incidence.update(tuple(sorted(map(int, row))) for row in block.data[:, face])
    for block in mesh.cells:
        corners = 3 if block.type.startswith("triangle") else 4 if block.type.startswith("quad") else None
        if corners and any(incidence[tuple(sorted(map(int, row)))] > 1 for row in block.data[:, :corners]):
            return True
    return False


def read_mesh_surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a meshio mesh; extract exterior faces when volume cells exist."""
    if path.suffix.lower() == ".stl":
        return read_stl_surface(path)
    import meshio

    mesh = meshio.read(path)
    vertices = np.asarray(mesh.points[:, :3], dtype=np.float64)
    volume_blocks = [(block, kind) for block in mesh.cells for kind in _VOLUME_FACES if block.type.startswith(kind)]
    polygons = []
    if volume_blocks:
        exterior = {}
        incidence = Counter()
        for block, kind in volume_blocks:
            for cell in block.data:
                corner_count = max(max(face) for face in _VOLUME_FACES[kind]) + 1
                center = vertices[cell[:corner_count]].mean(axis=0)
                for local_face in _VOLUME_FACES[kind]:
                    face = [int(cell[index]) for index in local_face]
                    points = vertices[face]
                    normal = np.cross(points[1] - points[0], points[2] - points[0])
                    if np.dot(normal, points.mean(axis=0) - center) < 0:
                        face.reverse()
                    key = tuple(sorted(face))
                    incidence[key] += 1
                    exterior[key] = face
        polygons = [face for key, face in exterior.items() if incidence[key] == 1]
    else:
        for block in mesh.cells:
            if block.type.startswith("triangle"):
                polygons.extend(block.data[:, :3].tolist())
            elif block.type.startswith("quad"):
                polygons.extend(block.data[:, :4].tolist())
    faces = [triangle for face in polygons for triangle in ([face] if len(face) == 3 else [[face[0], face[1], face[2]], [face[0], face[2], face[3]]])]
    return validate_surface_arrays(vertices, np.asarray(faces, dtype=np.int64))


def read_stl_surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Decode binary or full-precision ASCII STL without uint32 size overflow."""
    with path.open("rb") as handle:
        header = handle.read(84)
        count = struct.unpack("<I", header[80:84])[0] if len(header) == 84 else 0
        if len(header) == 84 and 84 + count * 50 == path.stat().st_size:
            dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
            triangles = np.fromfile(handle, dtype=dtype, count=count)["vertices"].astype(np.float64)
        else:
            handle.seek(0)
            rows = []
            for line in handle:
                tokens = line.decode("ascii").split()
                if tokens and tokens[0].lower() == "vertex":
                    if len(tokens) != 4:
                        raise ValueError("Invalid STL vertex record.")
                    rows.append([float(value) for value in tokens[1:]])
            if not rows or len(rows) % 3:
                raise ValueError("STL must contain complete triangles.")
            triangles = np.asarray(rows, dtype=np.float64).reshape(-1, 3, 3)
    vertices, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    return validate_surface_arrays(vertices, inverse.reshape(-1, 3))


def write_surface_stl(path: Path, vertices, faces) -> None:
    """Write ASCII STL with full coordinate precision, without binary float32 loss."""
    vertices, faces = validate_surface_arrays(vertices, faces)
    with path.open("w", encoding="ascii") as handle:
        handle.write("solid physicsos_repaired\n")
        for triangle in vertices[faces]:
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            norm = np.linalg.norm(normal)
            if norm:
                normal /= norm
            handle.write("  facet normal " + " ".join(format(float(v), ".17g") for v in normal) + "\n    outer loop\n")
            for point in triangle:
                handle.write("      vertex " + " ".join(format(float(v), ".17g") for v in point) + "\n")
            handle.write("    endloop\n  endfacet\n")
        handle.write("endsolid physicsos_repaired\n")
