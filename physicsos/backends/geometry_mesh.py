from __future__ import annotations

from pathlib import Path
from typing import Any

from physicsos.config import project_root
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


def _source_path(source: GeometrySource) -> Path | None:
    if source.uri is None:
        return None
    path = Path(source.uri)
    if not path.is_absolute():
        path = project_root() / path
    return path


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


def _build_generated_geometry(gmsh: Any, dimension: int, element_size: float | None) -> None:
    lc = element_size or 0.1
    if dimension == 1:
        p1 = gmsh.model.occ.addPoint(0.0, 0.0, 0.0, lc)
        p2 = gmsh.model.occ.addPoint(1.0, 0.0, 0.0, lc)
        line = gmsh.model.occ.addLine(p1, p2)
        gmsh.model.occ.synchronize()
        tag = gmsh.model.addPhysicalGroup(1, [line])
        gmsh.model.setPhysicalName(1, tag, "domain")
    elif dimension == 2:
        surface = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, 1.0, 1.0)
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
            elif abs(midpoint_x - 1.0) <= 1e-8:
                grouped_curves["x_max"].append(tag)
            elif abs(midpoint_y) <= 1e-8:
                grouped_curves["y_min"].append(tag)
            elif abs(midpoint_y - 1.0) <= 1e-8:
                grouped_curves["y_max"].append(tag)
        for name, curve_tags in grouped_curves.items():
            if not curve_tags:
                continue
            physical_tag = gmsh.model.addPhysicalGroup(1, curve_tags)
            gmsh.model.setPhysicalName(1, physical_tag, name)
    else:
        volume = gmsh.model.occ.addBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        gmsh.model.occ.synchronize()
        tag = gmsh.model.addPhysicalGroup(3, [volume])
        gmsh.model.setPhysicalName(3, tag, "domain")


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
) -> tuple[MeshSpec, list[ArtifactRef]]:
    """Generate a solver-neutral mesh with gmsh and optionally convert it with meshio."""
    artifacts: list[ArtifactRef] = []
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

    output_dir = _workspace(geometry.id)
    msh_path = output_dir / "mesh.msh"
    source_path = _source_path(geometry.source)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(_safe(geometry.id))
        if geometry.source.kind == "generated" or source_path is None:
            _build_generated_geometry(gmsh, geometry.dimension, target_element_size)
        elif source_path.exists():
            gmsh.open(str(source_path))
        else:
            raise FileNotFoundError(f"Geometry source path not found: {geometry.source.uri}")
        if target_element_size is not None:
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", target_element_size)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", target_element_size)
        gmsh.model.mesh.generate(geometry.dimension)
        if element_order > 1:
            gmsh.model.mesh.setOrder(element_order)
        gmsh.write(str(msh_path))
        topology, elements = _mesh_counts_from_gmsh(gmsh)
    finally:
        gmsh.finalize()

    artifacts.append(ArtifactRef(uri=str(msh_path), kind="mesh_file", format="msh", description="Gmsh mesh"))
    converted = _convert_with_meshio(msh_path)
    if converted is not None:
        artifacts.append(converted)

    mesh = MeshSpec(
        id=f"mesh:{geometry.id}",
        kind="unstructured",
        dimension=geometry.dimension,
        topology=topology,
        elements=elements,
        regions=geometry.regions,
        boundaries=geometry.boundaries,
        quality=MeshQualityReport(passes=elements.total is not None and elements.total > 0),
        files=artifacts,
        solver_compatibility=target_backends,
    )
    return mesh, artifacts
