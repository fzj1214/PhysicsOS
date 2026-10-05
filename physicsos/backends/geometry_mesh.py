from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from typing import Any

from physicsos.config import project_root
from physicsos.paths import resolve_workspace_path
from physicsos.schemas.common import ArtifactRef
from physicsos.schemas.geometry import (
    BoundaryRegionSpec,
    GeometryEntity,
    GeometryQualityReport,
    GeometrySource,
    GeometrySpec,
    RegionSpec,
)
from physicsos.schemas.mesh import ElementStats, MeshQualityReport, MeshSpec, MeshTopology


def _safe(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


def _workspace(geometry_id: str) -> Path:
    path = project_root() / "scratch" / _safe(geometry_id) / "geometry_mesh"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _import_optional(module_name: str) -> Any | None:
    try:
        return __import__(module_name)
    except ImportError:
        return None


def _gmsh_requires_subprocess() -> bool:
    """Gmsh may install signal handlers; isolate it outside Textual worker threads."""
    return threading.current_thread() is not threading.main_thread()


def _run_backend_subprocess(payload: dict[str, Any], timeout_seconds: float = 120.0) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
        input_path = Path(handle.name)
        json.dump(payload, handle)
    output_path = input_path.with_suffix(".out.json")
    script = (
        "import json, traceback\n"
        "from pathlib import Path\n"
        "from physicsos.backends.geometry_mesh import _run_backend_payload\n"
        f"input_path = Path({str(input_path)!r})\n"
        f"output_path = Path({str(output_path)!r})\n"
        "payload = json.loads(input_path.read_text(encoding='utf-8'))\n"
        "try:\n"
        "    result = _run_backend_payload(payload)\n"
        "except Exception as exc:\n"
        "    result = {'ok': False, 'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()}\n"
        "output_path.write_text(json.dumps(result), encoding='utf-8')\n"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(project_root()),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        if completed.returncode != 0 and not output_path.exists():
            stderr = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
            return {"ok": False, "error": f"geometry subprocess failed: {stderr}"}
        if not output_path.exists():
            return {"ok": False, "error": "geometry subprocess did not write an output payload."}
        result = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            return {"ok": False, "error": "geometry subprocess returned a non-object payload."}
        return result
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"geometry subprocess exceeded {timeout_seconds:.0f}s timeout."}
    finally:
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


def _run_backend_payload(payload: dict[str, Any]) -> dict[str, Any]:
    action = payload.get("action")
    if action == "mesh_quality":
        from physicsos.backends.mesh_quality import assess_mesh_main
        quality = assess_mesh_main(MeshSpec.model_validate(payload["mesh"]))
        return {"ok": True, "quality": quality.model_dump(mode="json")}
    if action == "bind_whole_boundary":
        _bind_whole_boundary_main(Path(payload["path"]), payload["role"], payload["dimension"])
        return {"ok": True}
    if action == "import_geometry":
        geometry, artifacts = _import_geometry_backend_main(
            GeometrySource.model_validate(payload["source"]),
            target_units=str(payload.get("target_units") or "SI"),
        )
        return {
            "ok": True,
            "geometry": geometry.model_dump(mode="json"),
            "artifacts": [artifact.model_dump(mode="json") for artifact in artifacts],
        }
    if action == "generate_mesh":
        mesh, artifacts = _generate_mesh_backend_main(
            GeometrySpec.model_validate(payload["geometry"]),
            target_backends=list(payload.get("target_backends") or []),
            target_element_size=payload.get("target_element_size"),
            element_order=int(payload.get("element_order") or 1),
            output_dir=payload.get("output_dir"),
            mesh_dimension=payload.get("mesh_dimension"),
        )
        return {
            "ok": True,
            "mesh": mesh.model_dump(mode="json"),
            "artifacts": [artifact.model_dump(mode="json") for artifact in artifacts],
        }
    return {"ok": False, "error": f"unknown geometry backend action: {action!r}"}


def _source_path(source: GeometrySource) -> Path | None:
    if source.uri is None:
        return None
    return resolve_workspace_path(source.uri, workspace=project_root())


def _physical_group_labels(gmsh: Any) -> dict[tuple[int, int], list[tuple[int, str]]]:
    labels: dict[tuple[int, int], list[tuple[int, str]]] = {}
    for dim, physical_tag in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(dim, physical_tag) or f"physical_{dim}_{physical_tag}"
        for entity_tag in gmsh.model.getEntitiesForPhysicalGroup(dim, physical_tag):
            labels.setdefault((int(dim), int(entity_tag)), []).append((int(physical_tag), name))
    return labels


def _boundary_kind_from_label(label: str) -> str:
    lowered = label.lower()
    if "inlet" in lowered:
        return "inlet"
    if "outlet" in lowered:
        return "outlet"
    if "wall" in lowered:
        return "wall"
    if "symmetry" in lowered:
        return "symmetry"
    if "periodic" in lowered:
        return "periodic"
    if "interface" in lowered:
        return "interface"
    if "farfield" in lowered or "far_field" in lowered:
        return "farfield"
    return "surface"


def _region_kind_from_label(label: str) -> str:
    lowered = label.lower()
    if "fluid" in lowered:
        return "fluid"
    if "solid" in lowered:
        return "solid"
    if "void" in lowered or "hole" in lowered:
        return "void"
    if "interface" in lowered:
        return "interface"
    if "periodic" in lowered:
        return "periodic_cell"
    return "custom"


def _gmsh_entities(gmsh: Any) -> tuple[list[GeometryEntity], list[RegionSpec], list[BoundaryRegionSpec], int]:
    raw_entities = gmsh.model.getEntities()
    physical_labels = _physical_group_labels(gmsh)
    entities: list[GeometryEntity] = []
    regions: list[RegionSpec] = []
    boundaries: list[BoundaryRegionSpec] = []
    dimension = max((int(dim) for dim, _ in raw_entities), default=0)
    for dim, tag in raw_entities:
        entity_dim = int(dim)
        entity_id = f"entity:{dim}:{tag}"
        kind = {0: "point", 1: "curve", 2: "surface", 3: "solid"}.get(entity_dim, "region")
        group_labels = physical_labels.get((entity_dim, int(tag)), [])
        entity_label = group_labels[0][1] if group_labels else f"{kind}_{tag}"
        entities.append(
            GeometryEntity(
                id=entity_id,
                kind=kind,
                label=entity_label,
                metadata={
                    "gmsh_dim": entity_dim,
                    "gmsh_tag": int(tag),
                    "physical_groups": ",".join(label for _, label in group_labels),
                },
            )
        )
        if entity_dim == dimension and entity_dim > 0:
            if group_labels:
                for physical_tag, label in group_labels:
                    regions.append(
                        RegionSpec(
                            id=f"region:physical:{physical_tag}",
                            label=label,
                            kind=_region_kind_from_label(label),  # type: ignore[arg-type]
                            entity_ids=[entity_id],
                        )
                    )
            else:
                region_label = {1: "curve_domain", 2: "surface_domain", 3: "volume"}.get(entity_dim, "domain")
                regions.append(RegionSpec(id=f"region:{tag}", label=f"{region_label}_{tag}", kind="custom", entity_ids=[entity_id]))
        elif entity_dim == dimension - 1:
            if group_labels:
                for physical_tag, label in group_labels:
                    boundaries.append(
                        BoundaryRegionSpec(
                            id=f"boundary:physical:{physical_tag}",
                            label=label,
                            kind=_boundary_kind_from_label(label),  # type: ignore[arg-type]
                            entity_ids=[entity_id],
                            confidence=1.0,
                        )
                    )
            else:
                boundary_label = {0: "point", 1: "curve", 2: "surface"}.get(entity_dim, "boundary")
                boundaries.append(
                    BoundaryRegionSpec(id=f"boundary:{tag}", label=f"{boundary_label}_{tag}", kind="surface", entity_ids=[entity_id])
                )
    if not regions and dimension == 2:
        regions.append(RegionSpec(id="region:surface-domain", label="surface_domain", kind="custom"))
    return entities, regions, boundaries, dimension


def import_geometry_backend(source: GeometrySource, target_units: str = "SI") -> tuple[GeometrySpec, list[ArtifactRef]]:
    if source.kind != "generated" and _gmsh_requires_subprocess():
        result = _run_backend_subprocess(
            {
                "action": "import_geometry",
                "source": source.model_dump(mode="json"),
                "target_units": target_units,
            }
        )
        if result.get("ok"):
            return (
                GeometrySpec.model_validate(result["geometry"]),
                [ArtifactRef.model_validate(artifact) for artifact in result.get("artifacts", [])],
            )
        geometry = GeometrySpec(
            id=f"geometry:{_safe(source.kind)}",
            source=source,
            dimension=3,
            quality=GeometryQualityReport(passes=False, issues=[str(result.get("error") or "geometry subprocess failed")]),
        )
        return geometry, []
    return _import_geometry_backend_main(source, target_units=target_units)


def _import_geometry_backend_main(source: GeometrySource, target_units: str = "SI") -> tuple[GeometrySpec, list[ArtifactRef]]:
    """Import geometry through gmsh when available, otherwise return explicit capability status."""
    geometry_id = f"geometry:{_safe(source.kind)}"
    artifacts: list[ArtifactRef] = []
    path = _source_path(source)

    if source.kind == "generated":
        geometry = GeometrySpec(
            id=geometry_id,
            source=source,
            dimension=3,
            quality=GeometryQualityReport(passes=True, issues=["Generated geometry placeholder; concrete primitive is chosen at mesh time."]),
        )
        return geometry, artifacts

    if path is not None and path.exists():
        artifacts.append(ArtifactRef(uri=str(path), kind="geometry_source", format=path.suffix.lstrip(".") or None))

    gmsh = _import_optional("gmsh")
    if gmsh is None:
        geometry = GeometrySpec(
            id=geometry_id,
            source=source,
            dimension=3,
            quality=GeometryQualityReport(
                passes=False,
                issues=["gmsh Python package is not installed; CAD/STL geometry import is unavailable."],
            ),
        )
        return geometry, artifacts
    if path is None or not path.exists():
        geometry = GeometrySpec(
            id=geometry_id,
            source=source,
            dimension=3,
            quality=GeometryQualityReport(passes=False, issues=[f"Geometry source path not found: {source.uri}"]),
        )
        return geometry, artifacts

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))
        entities, regions, boundaries, dimension = _gmsh_entities(gmsh)
    finally:
        gmsh.finalize()

    geometry = GeometrySpec(
        id=geometry_id,
        source=source,
        dimension=dimension,  # type: ignore[arg-type]
        entities=entities,
        regions=regions,
        boundaries=boundaries,
        quality=GeometryQualityReport(passes=True),
    )
    return geometry, artifacts


def _build_generated_geometry(gmsh: Any, dimension: int, element_size: float | None, geometry: GeometrySpec) -> None:
    lc = element_size or 0.1
    parameters = {key: value for entity in geometry.entities for key, value in entity.metadata.items()}
    def length(name: str) -> float:
        value = parameters.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Generated geometry requires an explicit positive {name}.")
        return float(value)
    if dimension == 1:
        p1 = gmsh.model.occ.addPoint(0.0, 0.0, 0.0, lc)
        p2 = gmsh.model.occ.addPoint(length("length"), 0.0, 0.0, lc)
        line = gmsh.model.occ.addLine(p1, p2)
        gmsh.model.occ.synchronize()
        tag = gmsh.model.addPhysicalGroup(1, [line])
        gmsh.model.setPhysicalName(1, tag, "domain")
    elif dimension == 2:
        extent_x, extent_y = length("length"), length("width")
        surface = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, extent_x, extent_y)
        gmsh.model.occ.synchronize()
        domain_tag = gmsh.model.addPhysicalGroup(2, [surface])
        gmsh.model.setPhysicalName(2, domain_tag, "domain")
        boundary = gmsh.model.getBoundary([(2, surface)], oriented=False, recursive=False)
        grouped_curves: dict[str, list[int]] = {"x_min": [], "x_max": [], "y_min": [], "y_max": []}
        for dim, tag in boundary:
            if dim != 1:
                continue
            xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(dim, tag)
            midpoint_x = 0.5 * (xmin + xmax)
            midpoint_y = 0.5 * (ymin + ymax)
            if abs(midpoint_x) <= 1e-8:
                grouped_curves["x_min"].append(tag)
            elif abs(midpoint_x - extent_x) <= 1e-8:
                grouped_curves["x_max"].append(tag)
            elif abs(midpoint_y) <= 1e-8:
                grouped_curves["y_min"].append(tag)
            elif abs(midpoint_y - extent_y) <= 1e-8:
                grouped_curves["y_max"].append(tag)
        for name, curve_tags in grouped_curves.items():
            if not curve_tags:
                continue
            physical_tag = gmsh.model.addPhysicalGroup(1, curve_tags)
            gmsh.model.setPhysicalName(1, physical_tag, name)
    else:
        primitive = parameters.get("primitive", "box")
        if primitive == "sphere":
            volume = gmsh.model.occ.addSphere(0, 0, 0, length("radius"))
        elif primitive == "cylinder":
            volume = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, length("height"), length("radius"))
        elif primitive == "box":
            volume = gmsh.model.occ.addBox(0, 0, 0, length("length"), length("width"), length("height"))
        else:
            raise ValueError("Provide a concrete CAD/mesh asset for this generated primitive.")
        gmsh.model.occ.synchronize()
        tag = gmsh.model.addPhysicalGroup(3, [volume])
        gmsh.model.setPhysicalName(3, tag, "domain")
        if primitive == "box":
            extents = [length("length"), length("width"), length("height")]
            for dim, face in gmsh.model.getBoundary([(3, volume)], oriented=False):
                bounds = gmsh.model.getBoundingBox(dim, face)
                for axis, name in enumerate("xyz"):
                    midpoint = (bounds[axis] + bounds[axis + 3]) / 2
                    role = f"{name}_min" if abs(midpoint) < 1e-8 else f"{name}_max" if abs(midpoint - extents[axis]) < 1e-8 else None
                    if role:
                        physical = gmsh.model.addPhysicalGroup(2, [face])
                        gmsh.model.setPhysicalName(2, physical, role)
                        break


def _surface_volume(gmsh: Any, path: Path, target_element_size: float | None = None) -> None:
    """Build volumes from closed surface components, retaining nested cavities."""
    import numpy as np
    import trimesh
    from physicsos.backends.surface_mesh import read_mesh_surface, surface_metrics, write_surface_stl
    vertices, faces = read_mesh_surface(path)
    metrics = surface_metrics(vertices, faces)
    if not metrics["watertight"] or not metrics["manifold"]:
        raise ValueError("Volume meshing requires a closed manifold surface; repair the asset first.")
    source = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    components = list(source.split(only_watertight=False))
    if path.suffix.lower() == ".stl":
        gmsh.merge(str(path))
    else:
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as temporary:
            component_path = Path(temporary.name)
        try:
            write_surface_stl(component_path, vertices, faces)
            gmsh.merge(str(component_path))
        finally:
            component_path.unlink(missing_ok=True)
    # Preserve the checked surface mesh instead of forcing global UV
    # parametrization, which can fail indefinitely on tiny closed soups.
    if target_element_size is not None:
        edges = vertices[faces[:, [1, 2, 0]]] - vertices[faces]
        maximum = float(np.linalg.norm(edges, axis=2).max())
        levels = max(0, math.ceil(math.log2(maximum / target_element_size)))
        if levels > 8 or len(faces) * 4 ** levels > 2000000:
            raise ValueError("Requested surface refinement exceeds the preparation budget.")
        for _ in range(levels):
            gmsh.model.mesh.refine()
    gmsh.model.mesh.createTopology(makeSimplyConnected=False, exportDiscrete=True)
    groups = [[] for _ in components]
    for _, tag in gmsh.model.getEntities(2):
        _, coordinates, _ = gmsh.model.mesh.getNodes(2, tag, includeBoundary=True)
        if not len(coordinates):
            raise ValueError("A surface patch has no coordinates for component binding.")
        point = np.asarray(coordinates).reshape(-1, 3)[0]
        distances = [float(trimesh.proximity.closest_point(component, np.asarray([point]))[1][0]) for component in components]
        component_id = int(np.argmin(distances))
        groups[component_id].append(tag)
    loop_tags = [gmsh.model.geo.addSurfaceLoop(tags) for tags in groups]
    # Containment is evaluated on original closed components, never their boxes.
    parents = []
    for i, component in enumerate(components):
        candidates = [j for j, other in enumerate(components) if i != j and abs(other.volume) > abs(component.volume) and bool(other.contains(np.asarray([component.vertices[0]]))[0])]
        parents.append(min(candidates, key=lambda j: abs(components[j].volume)) if candidates else None)
    def depth(index):
        seen = set()
        while parents[index] is not None:
            if index in seen:
                raise ValueError("Surface containment is cyclic.")
            seen.add(index)
            index = parents[index]
        return len(seen)
    for i in range(len(components)):
        if depth(i) % 2 == 0:
            holes = [loop_tags[j] for j in range(len(components)) if parents[j] == i]
            gmsh.model.geo.addVolume([loop_tags[i], *holes])
    gmsh.model.geo.synchronize()


def _attach_physical_groups(gmsh: Any, geometry: GeometrySpec) -> None:
    bindings = [*geometry.regions, *geometry.boundaries] if geometry.source.kind in {"cad_step", "cad_iges", "generated"} or (geometry.source.uri or "").endswith(".geo") else []
    for region in bindings:
        selected = [entity for entity in geometry.entities if entity.id in region.entity_ids and "gmsh_tag" in entity.metadata]
        dimensions = {int(entity.metadata["gmsh_dim"]) for entity in selected}
        for dimension in dimensions:
            tags = [int(entity.metadata["gmsh_tag"]) for entity in selected if int(entity.metadata["gmsh_dim"]) == dimension]
            existing = [(dim, tag) for dim, tag in gmsh.model.getPhysicalGroups(dimension) if gmsh.model.getPhysicalName(dim, tag) == region.label]
            if tags and not existing:
                physical = gmsh.model.addPhysicalGroup(dimension, tags)
                gmsh.model.setPhysicalName(dimension, physical, region.label)
    for dimension, name in [(geometry.dimension, "domain"), (geometry.dimension - 1, "boundary")]:
        covered = {int(entity) for dim, tag in gmsh.model.getPhysicalGroups(dimension) for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, tag)}
        missing = [tag for _, tag in gmsh.model.getEntities(dimension) if tag not in covered]
        if missing:
            physical = gmsh.model.addPhysicalGroup(dimension, missing)
            gmsh.model.setPhysicalName(dimension, physical, name)


def _mesh_counts_from_gmsh(gmsh: Any) -> tuple[MeshTopology, ElementStats]:
    _, node_coords, _ = gmsh.model.mesh.getNodes()
    element_types, _, element_node_tags = gmsh.model.mesh.getElements()
    by_type: dict[str, int] = {}
    total = 0
    for element_type, nodes in zip(element_types, element_node_tags):
        name = gmsh.model.mesh.getElementProperties(int(element_type))[0]
        node_count_per_element = gmsh.model.mesh.getElementProperties(int(element_type))[3]
        count = int(len(nodes) / node_count_per_element) if node_count_per_element else 0
        by_type[name] = by_type.get(name, 0) + count
        total += count
    return (
        MeshTopology(cell_types=sorted(by_type), node_count=int(len(node_coords) / 3), cell_count=total),
        ElementStats(total=total, by_type=by_type),
    )


def _convert_with_meshio(msh_path: Path) -> ArtifactRef | None:
    meshio = _import_optional("meshio")
    if meshio is None:
        return None
    vtu_path = msh_path.with_suffix(".vtu")
    mesh = meshio.read(msh_path)
    # Gmsh-specific cell sets can be inconsistent across mixed-dimensional
    # blocks; strip them for a solver-neutral visualization artifact.
    mesh.cell_sets = {}
    try:
        meshio.write(vtu_path, mesh)
    except (KeyError, ValueError):
        # Some meshio/VTK versions cannot write high-order Gmsh cells such as
        # triangle10. Keep the source .msh artifact; mesh_graph encoding can
        # still consume it directly.
        return None
    return ArtifactRef(uri=str(vtu_path), kind="mesh_file", format="vtu", description="meshio-converted visualization mesh")


def generate_mesh_backend(
    geometry: GeometrySpec,
    target_backends: list[str],
    target_element_size: float | None,
    element_order: int = 1,
    output_dir: str | Path | None = None,
    mesh_dimension: int | None = None,
) -> tuple[MeshSpec, list[ArtifactRef]]:
    if _gmsh_requires_subprocess():
        result = _run_backend_subprocess(
            {
                "action": "generate_mesh",
                "geometry": geometry.model_dump(mode="json"),
                "target_backends": target_backends,
                "target_element_size": target_element_size,
                "element_order": element_order,
                "output_dir": str(output_dir) if output_dir is not None else None,
                "mesh_dimension": mesh_dimension,
            }
        )
        if result.get("ok"):
            return (
                MeshSpec.model_validate(result["mesh"]),
                [ArtifactRef.model_validate(artifact) for artifact in result.get("artifacts", [])],
            )
        mesh = MeshSpec(
            id=f"mesh:{geometry.id}",
            kind="unstructured",
            dimension=geometry.dimension,
            regions=geometry.regions,
            boundaries=geometry.boundaries,
            quality=MeshQualityReport(passes=False, issues=[str(result.get("error") or "geometry subprocess failed")]),
            solver_compatibility=target_backends,
        )
        return mesh, []
    return _generate_mesh_backend_main(
        geometry,
        target_backends=target_backends,
        target_element_size=target_element_size,
        element_order=element_order,
        output_dir=output_dir,
        mesh_dimension=mesh_dimension,
    )


def _generate_mesh_backend_main(
    geometry: GeometrySpec,
    target_backends: list[str],
    target_element_size: float | None,
    element_order: int = 1,
    output_dir: str | Path | None = None,
    mesh_dimension: int | None = None,
) -> tuple[MeshSpec, list[ArtifactRef]]:
    """Generate a solver-neutral mesh with gmsh and optionally convert it with meshio."""
    artifacts: list[ArtifactRef] = []
    mesh_dimension = geometry.dimension if mesh_dimension is None else mesh_dimension
    gmsh = _import_optional("gmsh")
    if gmsh is None:
        mesh = MeshSpec(
            id=f"mesh:{geometry.id}",
            kind="unstructured",
            dimension=geometry.dimension,
            regions=geometry.regions,
            boundaries=geometry.boundaries,
            quality=MeshQualityReport(
                passes=False,
                issues=["gmsh Python package is not installed; real mesh generation is unavailable."],
            ),
            solver_compatibility=target_backends,
        )
        return mesh, artifacts

    output_dir = Path(output_dir) if output_dir is not None else _workspace(geometry.id)
    output_dir.mkdir(parents=True, exist_ok=True)
    msh_path = output_dir / "mesh.msh"
    source_path = _source_path(geometry.source)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(_safe(geometry.id))
        if geometry.source.kind == "generated" or source_path is None:
            if geometry.source.kind != "generated":
                raise ValueError("A concrete source asset is required for meshing.")
            _build_generated_geometry(gmsh, geometry.dimension, target_element_size, geometry)
        elif source_path.exists():
            if geometry.dimension == 3 and geometry.source.kind in {"stl", "mesh_file"} and source_path.suffix.lower() != ".geo":
                _surface_volume(gmsh, source_path, target_element_size)
            else:
                gmsh.open(str(source_path))
        else:
            raise FileNotFoundError(f"Geometry source path not found: {geometry.source.uri}")
        _attach_physical_groups(gmsh, geometry)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        if target_element_size is not None:
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", target_element_size)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", target_element_size)
        gmsh.model.mesh.generate(mesh_dimension)
        if element_order > 1:
            gmsh.model.mesh.setOrder(element_order)
        gmsh.write(str(msh_path))
        topology, elements = _mesh_counts_from_gmsh(gmsh)
        _, regions, boundaries, _ = _gmsh_entities(gmsh)
    finally:
        gmsh.finalize()

    artifacts.append(ArtifactRef(uri=str(msh_path), kind="mesh_file", format="msh", description="Gmsh mesh"))
    converted = _convert_with_meshio(msh_path)
    if converted is not None:
        artifacts.append(converted)

    mesh = MeshSpec(
        id=f"mesh:{geometry.id}",
        kind="unstructured",
        dimension=mesh_dimension,
        topology=topology,
        elements=elements,
        regions=regions,
        boundaries=boundaries,
        quality=MeshQualityReport(passes=elements.total is not None and elements.total > 0),
        files=artifacts,
        solver_compatibility=target_backends,
    )
    return mesh, artifacts


def bind_whole_boundary(path: Path, role: str, dimension: int) -> None:
    if _gmsh_requires_subprocess():
        result = _run_backend_subprocess({"action": "bind_whole_boundary", "path": str(path), "role": role, "dimension": dimension})
        if not result.get("ok"):
            raise ValueError(result.get("error", "Could not bind physical boundary groups."))
    else:
        _bind_whole_boundary_main(path, role, dimension)


def _bind_whole_boundary_main(path: Path, role: str, dimension: int) -> None:
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))
        tags = [tag for _, tag in gmsh.model.getEntities(dimension - 1)]
        if not tags:
            raise ValueError("No boundary entities are available for the declared whole-boundary role.")
        gmsh.model.removePhysicalGroups(gmsh.model.getPhysicalGroups(dimension - 1))
        physical = gmsh.model.addPhysicalGroup(dimension - 1, tags)
        gmsh.model.setPhysicalName(dimension - 1, physical, role)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
