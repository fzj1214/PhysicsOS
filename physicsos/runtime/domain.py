"""Prepare actual asset geometry against a kernel's discretization requirements."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
from uuid import uuid4

import meshio
import numpy as np

from physicsos.backends.geometry_mesh import _run_backend_subprocess
from physicsos.backends.geometry_repair import run_pamo_repair
from physicsos.backends.surface_mesh import has_internal_surface_elements, read_mesh_surface, surface_metrics, write_surface_stl
from physicsos.paths import resolve_workspace_path
from physicsos.schemas.common import ArtifactRef
from physicsos.schemas.geometry import BoundaryRegionSpec, GeometryEncoding, GeometryEntity, GeometrySpec, RegionSpec
from physicsos.schemas.geometry_repair import RepairGeometryInput
from physicsos.schemas.mesh import ElementStats, MeshSpec, MeshTopology
from physicsos.schemas.case_runtime import PreparedDomain, PrepareDomainInput, PrepareDomainOutput
from physicsos.tools.geometry_tools import BuildGeometryMeshContractInput, MeshSemanticsGateInput, build_geometry_mesh_contract, mesh_semantics_gate
from physicsos.runtime.artifacts import artifact, digest, write_json


_ROLES = {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "inlet", "outlet", "wall", "symmetry", "farfield", "interface", "periodic", "custom"}


class _PreparationNeedsReview(ValueError):
    def __init__(self, check: str, actions: list[str]):
        super().__init__("; ".join(actions))
        self.check = check
        self.actions = actions


def _check_mesh_policy(requirements) -> None:
    policy = requirements.mesh_policy
    unsupported = set()
    if policy.strategy not in {"auto", "unstructured"}:
        unsupported.add(policy.strategy)
    if policy.boundary_layer:
        unsupported.add("boundary_layer")
    if policy.refinement_regions:
        unsupported.add("local_refinement")
    if unsupported:
        raise _PreparationNeedsReview("mesh_policy", [f"provide_mesh_provider_for:{name}" for name in sorted(unsupported)])


def _check_surface_conversion(geometry: GeometrySpec) -> None:
    """Exterior-surface conversion cannot carry internal material partitions."""
    regions = {region.label for region in geometry.regions}
    interfaces = any(boundary.kind == "interface" or boundary.role == "interface" for boundary in geometry.boundaries)
    interfaces |= any(region.kind == "interface" for region in geometry.regions)
    if geometry.source.uri and geometry.source.kind == "mesh_file" and Path(geometry.source.uri).suffix.lower() != ".geo":
        raw = meshio.read(geometry.source.uri)
        interfaces |= has_internal_surface_elements(raw)
        tags = raw.cell_data.get("gmsh:physical", [])
        top_tags = {int(tag) for block, values in zip(raw.cells, tags) if block.dim == geometry.dimension for tag in np.unique(values) if tag > 0}
        if len(top_tags) > 1:
            regions.update(f"physical:{tag}" for tag in top_tags)
        for name, tag_dim in raw.field_data.items():
            if int(tag_dim[1]) == geometry.dimension - 1 and "interface" in name.lower():
                interfaces = True
        # Other mesh formats can carry material IDs without Gmsh field names.
        for name, blocks in raw.cell_data.items():
            if name != "gmsh:physical" and not any(word in name.lower() for word in ("material", "region")):
                continue
            values = {str(value) for block, data in zip(raw.cells, blocks) if block.dim == geometry.dimension for value in np.unique(data)}
            if len(values) > 1:
                regions.update(values)
    elif geometry.source.uri and geometry.source.kind in {"cad_step", "cad_iges"}:
        result = _run_backend_subprocess({"action": "import_geometry", "source": geometry.source.model_dump(mode="json")})
        if not result.get("ok"):
            raise ValueError(result.get("error", "Could not inspect CAD regions before surface conversion."))
        imported = GeometrySpec.model_validate(result["geometry"])
        if imported.quality is None or not imported.quality.passes:
            raise ValueError("CAD region inventory could not be established before surface conversion.")
        if len({region.id for region in imported.regions}) > 1:
            regions.update(region.label for region in imported.regions)
    if len(regions) > 1 or interfaces:
        raise _PreparationNeedsReview("region_preservation", ["provide_interface_preserving_meshing_or_repair_provider"])


def _native_mesh(geometry: GeometrySpec, requirement, output_dir: Path) -> MeshSpec:
    if geometry.source.kind in {"stl", "mesh_file"} and not (geometry.source.uri or "").lower().endswith(".geo"):
        _check_surface_conversion(geometry)
    result = _run_backend_subprocess({
        "action": "generate_mesh", "geometry": geometry.model_dump(mode="json"),
        "target_backends": [requirement.backend],
        "target_element_size": requirement.mesh_policy.target_element_size,
        "element_order": requirement.mesh_policy.element_order, "output_dir": str(output_dir),
        "mesh_dimension": 2 if requirement.representation == "background_grid" else requirement.dimension,
    })
    if not result.get("ok"):
        raise ValueError(result.get("error", "Mesh generation failed."))
    return MeshSpec.model_validate(result["mesh"])


def _quality(mesh: MeshSpec):
    from physicsos.schemas.mesh import MeshQualityReport
    result = _run_backend_subprocess({"action": "mesh_quality", "mesh": mesh.model_dump(mode="json")})
    if not result.get("ok"):
        raise ValueError(result.get("error", "Mesh quality evaluation failed."))
    return MeshQualityReport.model_validate(result["quality"])


def _repair_surface(geometry, domain, input, directory, workspace):
    _check_surface_conversion(geometry)
    if not geometry.source.uri:
        surface_requirements = domain.requirements.model_copy(deep=True)
        surface_requirements.representation = "background_grid"
        proxy = _native_mesh(geometry, surface_requirements, directory / "repair-input")
        vertices, faces = read_mesh_surface(Path(next(item.uri for item in proxy.files if item.format == "msh")))
        source = directory / "repair_input.stl"
        write_surface_stl(source, vertices, faces)
        geometry = geometry.model_copy(deep=True)
        geometry.source.kind = "stl"
        geometry.source.uri = str(source)
        geometry.source.checksum = digest(source)
    result = run_pamo_repair(RepairGeometryInput(geometry=geometry, **input.repair_options.model_dump(exclude={"case_id"}), case_id=input.case_id), workspace=workspace)
    domain.actions.append({"action": "repair_surface", "status": result.status, "report": result.repair_report.uri})
    domain.artifacts["repair_report"] = artifact(resolve_workspace_path(result.repair_report.uri, workspace=workspace), "geometry_repair_report", workspace)
    if result.status != "repaired":
        domain.status = "backend_unavailable" if result.status == "backend_unavailable" else "needs_review"
        domain.required_actions = ["resolve_surface_repair", *result.geometry.quality.issues]
        return None
    geometry = result.geometry
    source = resolve_workspace_path(geometry.source.uri, workspace=workspace)
    shutil.copyfile(source, directory / "repaired_asset.stl")
    geometry.source.uri = str(directory / "repaired_asset.stl")
    domain.artifacts["source_asset"] = artifact(Path(geometry.source.uri), "geometry_source", workspace)
    return geometry


def _mesh_geometry(mesh: MeshSpec, original: GeometrySpec, requirements) -> GeometrySpec:
    path = next(item.uri for item in mesh.files if item.format == "msh")
    if requirements.whole_boundary_role:
        if has_internal_surface_elements(meshio.read(path)):
            raise _PreparationNeedsReview("boundary_preservation", ["bind_exterior_boundary_groups_without_overwriting_internal_interfaces"])
        result = _run_backend_subprocess({"action": "bind_whole_boundary", "path": path, "role": requirements.whole_boundary_role, "dimension": original.dimension})
        if not result.get("ok"):
            raise ValueError(result.get("error", "Boundary binding failed."))
    result = _run_backend_subprocess({"action": "import_geometry", "source": {"kind": "mesh_file", "uri": path}})
    if not result.get("ok"):
        raise ValueError(result.get("error", "Mesh semantics could not be read."))
    geometry = GeometrySpec.model_validate(result["geometry"])
    geometry.id = original.id
    geometry.dimension = original.dimension
    geometry.source = original.source.model_copy(deep=True)
    geometry.coordinate_system = original.coordinate_system.model_copy(deep=True)
    geometry.transforms = list(original.transforms)
    # Physical groups may span multiple entities; preserve one semantic region.
    merged = {}
    prior_roles = {boundary.label: boundary.role for boundary in original.boundaries if boundary.confidence >= .7 and boundary.role}
    for boundary in geometry.boundaries:
        role = requirements.boundary_roles.get(boundary.label) or prior_roles.get(boundary.label) or (boundary.label if boundary.label in _ROLES else None)
        boundary.role = role
        if boundary.id in merged:
            merged[boundary.id].entity_ids.extend(boundary.entity_ids)
        else:
            merged[boundary.id] = boundary
    geometry.boundaries = list(merged.values())
    prior_regions = {region.label: region for region in original.regions}
    merged_regions = {}
    for region in geometry.regions:
        if region.label in prior_regions:
            region.kind = prior_regions[region.label].kind
        if region.id in merged_regions:
            merged_regions[region.id].entity_ids.extend(region.entity_ids)
        else:
            merged_regions[region.id] = region
    geometry.regions = list(merged_regions.values())
    if original.source.kind == "generated" and geometry.entities:
        parameters = {key: value for entity in original.entities for key, value in entity.metadata.items()}
        geometry.entities[0].metadata.update(parameters)
    mesh.boundaries = geometry.boundaries
    mesh.regions = geometry.regions
    return geometry


def _mesh_artifacts(mesh: MeshSpec, geometry: GeometrySpec, directory: Path, workspace: Path) -> dict[str, ArtifactRef]:
    path = Path(next(item.uri for item in mesh.files if item.format == "msh"))
    raw = meshio.read(path)
    arrays = {"points": raw.points[:, :3]}
    for block in raw.cells:
        arrays[block.type] = np.concatenate([arrays[block.type], block.data]) if block.type in arrays else block.data
    np.savez_compressed(directory / "mesh_arrays.npz", **arrays)
    masks = {}
    physical = raw.cell_data.get("gmsh:physical", [])
    role_map = {boundary.label: boundary.role for boundary in geometry.boundaries}
    groups = []
    for name, tag_dim in raw.field_data.items():
        tag, dimension = map(int, tag_dim)
        if dimension != mesh.dimension - 1:
            continue
        node_ids = set()
        for block, tags in zip(raw.cells, physical):
            if block.dim == dimension:
                node_ids.update(block.data[np.asarray(tags) == tag].ravel().tolist())
        mask = np.zeros(len(raw.points), dtype=bool)
        if node_ids:
            mask[list(node_ids)] = True
        masks[name if name not in {"file", "allow_pickle"} else "boundary_" + name] = mask
        if role_map.get(name):
            role = role_map[name]
            masks[role] = masks.get(role, np.zeros_like(mask)) | mask
        groups.append({"name": name, "tag": tag, "dimension": dimension, "node_ids": sorted(node_ids), "role": role_map.get(name)})
    np.savez_compressed(directory / "boundary_nodes.npz", **masks)
    references = {
        "mesh": artifact(path, "simulation_mesh", workspace),
        "mesh_arrays": artifact(directory / "mesh_arrays.npz", "mesh_arrays", workspace),
        "boundary_nodes": artifact(directory / "boundary_nodes.npz", "boundary_nodes", workspace),
    }
    layout = directory / "mesh_layout.json"
    write_json(layout, {"type": "mesh_graph", "dimension": mesh.dimension, "node_count": len(raw.points), "cell_types": sorted(arrays.keys() - {"points"}), "physical_boundary_groups": groups, "arrays": references["mesh_arrays"].uri, "source_mesh": references["mesh"].uri})
    references["mesh_layout"] = artifact(layout, "mesh_layout", workspace)
    geometry.encodings = [GeometryEncoding(kind="mesh_graph", uri=references["mesh_layout"].uri, target_backend="case_kernel")]
    return references


def _grid_artifacts(geometry: GeometrySpec, requirements, directory: Path, workspace: Path, proxy_mesh: MeshSpec, source_regions: list[RegionSpec]) -> tuple[dict[str, ArtifactRef], dict[str, bool]]:
    import trimesh
    source = resolve_workspace_path(geometry.source.uri, workspace=workspace)
    vertices, faces = read_mesh_surface(source)
    metrics = surface_metrics(vertices, faces)
    if not metrics["watertight"] or not metrics["manifold"] or not metrics["winding_consistent"]:
        raise ValueError("A signed-distance domain requires a closed, consistently oriented manifold surface.")
    resolution = requirements.grid_resolution
    if requirements.dimension != 3 or len(resolution) != 3 or any(isinstance(value, bool) or value < 3 or value > 128 for value in resolution):
        raise ValueError("Background-grid requirements need three resolutions in [3, 128].")
    lower = np.asarray(requirements.bounds_min if requirements.bounds_min is not None else vertices.min(axis=0))
    upper = np.asarray(requirements.bounds_max if requirements.bounds_max is not None else vertices.max(axis=0))
    if lower.shape != (3,) or upper.shape != (3,) or not np.isfinite([lower, upper]).all() or not (upper > lower).all():
        raise ValueError("Background-grid bounds are invalid.")
    if requirements.domain_side == "exterior" and (requirements.bounds_min is None or requirements.bounds_max is None or not (lower < vertices.min(axis=0)).all() or not (upper > vertices.max(axis=0)).all()):
        raise ValueError("Exterior domains require explicit enclosing bounds with a positive margin.")
    axes = [np.linspace(lower[i], upper[i], resolution[i]) for i in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    surface = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    values = np.concatenate([-trimesh.proximity.signed_distance(surface, chunk) for chunk in np.array_split(points, max(1, len(points) // 2048))])
    if requirements.domain_side == "exterior":
        values = -values
    sdf = values.reshape(resolution)
    occupancy = (sdf <= 0).astype(np.uint8)
    spacing = float(max(np.diff(axis).max() for axis in axes))
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    if requirements.domain_side == "exterior":
        normals = -normals
    def face_key(vertices):
        return tuple(sorted(tuple(float(value) for value in point) for point in vertices))
    raw = meshio.read(next(item.uri for item in proxy_mesh.files if item.format == "msh"))
    tags_to_names = {int(tag_dim[0]): name for name, tag_dim in raw.field_data.items() if int(tag_dim[1]) == 2}
    face_labels = {}
    for block, tags in zip(raw.cells, raw.cell_data.get("gmsh:physical", [])):
        if block.type.startswith("triangle"):
            for face, tag in zip(block.data[:, :3], tags):
                face_labels[face_key(raw.points[face, :3])] = tags_to_names.get(int(tag), "boundary")
    labels = np.asarray([face_labels.get(face_key(triangle), "boundary") for triangle in triangles])
    samples = triangles.mean(axis=1)
    masks = {}
    boundaries = []
    entities = []
    role_map = {boundary.label: boundary.role for boundary in geometry.boundaries}
    for label in sorted(set(labels)):
        name = f"body:{label}" if requirements.domain_side == "exterior" else label
        role = requirements.boundary_roles.get(name) or requirements.boundary_roles.get(label) or role_map.get(label)
        if role is None and label in _ROLES:
            role = label
        masks[name] = labels == label
        identifier = f"boundary:{name}"
        entities.append(GeometryEntity(id=identifier, kind="surface", label=name, metadata={"sample_count": int(masks[name].sum())}))
        boundaries.append(BoundaryRegionSpec(id=identifier, label=name, kind="surface", role=role, entity_ids=[identifier], confidence=1))
        if role:
            masks[role] = masks.get(role, np.zeros(len(samples), dtype=bool)) | masks[name]
    if requirements.domain_side == "exterior":
        for axis, axis_name in enumerate("xyz"):
            others = [i for i in range(3) if i != axis]
            coordinates = np.meshgrid(axes[others[0]], axes[others[1]], indexing="ij")
            for side, coordinate in (("min", lower[axis]), ("max", upper[axis])):
                name = f"outer:{axis_name}_{side}"
                plane = np.zeros((coordinates[0].size, 3))
                plane[:, axis] = coordinate
                plane[:, others[0]] = coordinates[0].ravel()
                plane[:, others[1]] = coordinates[1].ravel()
                offset = len(samples)
                samples = np.concatenate([samples, plane])
                normal = np.zeros_like(plane)
                normal[:, axis] = -1 if side == "min" else 1
                normals = np.concatenate([normals, normal])
                for key in list(masks):
                    masks[key] = np.pad(masks[key], (0, len(plane)))
                masks[name] = np.arange(len(samples)) >= offset
                role = requirements.boundary_roles.get(name) or requirements.boundary_roles.get(f"{axis_name}_{side}") or f"{axis_name}_{side}"
                masks[role] = masks.get(role, np.zeros(len(samples), dtype=bool)) | masks[name]
                identifier = f"boundary:{name}"
                entities.append(GeometryEntity(id=identifier, kind="surface", label=name, metadata={"coordinate": float(coordinate)}))
                boundaries.append(BoundaryRegionSpec(id=identifier, label=name, kind="surface", role=role, entity_ids=[identifier], confidence=1))
    np.savez_compressed(directory / "boundary_groups.npz", **{("boundary_" + key if key in {"file", "allow_pickle"} else key): value for key, value in masks.items()})
    region_id = "region:computational-domain"
    region = source_regions[0] if len(source_regions) == 1 and requirements.domain_side == "interior" else None
    label = region.label if region else "computational domain"
    geometry.entities = [GeometryEntity(id=region_id, kind="region", label=label, metadata={"domain_side": requirements.domain_side}), *entities]
    geometry.regions = [RegionSpec(id=region_id, label=label, kind=region.kind if region else "custom", entity_ids=[region_id])]
    geometry.boundaries = boundaries
    for name, array in {"sdf": sdf, "occupancy": occupancy, "boundary_samples": samples, "normals": normals, "cut_cells": np.argwhere(np.abs(sdf) <= 1.5 * spacing)}.items():
        np.save(directory / f"{name}.npy", array)
    write_json(directory / "background_grid.json", {"schema_version": "physicsos.background_grid.v1", "axes": dict(zip("xyz", [axis.tolist() for axis in axes])), "resolution": resolution, "bounds_min": lower.tolist(), "bounds_max": upper.tolist(), "domain_side": requirements.domain_side})
    checks = {"surface_topology": True, "array_alignment": bool(np.isfinite(sdf).all() and occupancy.any()), "grid_bounds": True}
    write_json(directory / "sdf_quality.json", {"method": "trimesh_triangle_distance", "passes": all(checks.values()), "source_mesh_metrics": metrics, "spacing": spacing, "domain_side": requirements.domain_side, "self_intersections_checked": geometry.quality is not None and geometry.quality.self_intersections == 0})
    references = {name: artifact(directory / f"{name}.npy", name, workspace) for name in ("sdf", "occupancy", "boundary_samples", "normals", "cut_cells")}
    references.update({name: artifact(directory / f"{name}.json", name, workspace) for name in ("background_grid", "sdf_quality")})
    references["boundary_groups"] = artifact(directory / "boundary_groups.npz", "boundary_groups", workspace)
    embedding = directory / "embedding.json"
    write_json(embedding, {"schema_version": "physicsos.geometry_embedding.v1", "sdf_convention": "phi <= 0 in the computational domain", "artifacts": {name: reference.uri for name, reference in references.items()}, "domain_side": requirements.domain_side, "source_sha256": digest(source)})
    references["embedding"] = artifact(embedding, "geometry_embedding", workspace)
    geometry.encodings = [GeometryEncoding(kind="sdf", uri=references["sdf"].uri, resolution=resolution, target_backend="taps"), GeometryEncoding(kind="structured_axes", uri=references["background_grid"].uri, resolution=resolution, target_backend="taps")]
    return references, checks


def prepare_domain(input: PrepareDomainInput, workspace: Path) -> PrepareDomainOutput:
    revision = uuid4().hex
    directory = workspace / "cases" / input.case_id / "domains" / revision
    directory.mkdir(parents=True, exist_ok=False)
    geometry = input.geometry.model_copy(deep=True)
    geometry.id = f"{input.case_id}-{revision}"
    geometry.dimension = input.requirements.dimension
    domain = PreparedDomain(id=revision, case_id=input.case_id, status="failed", requirements=input.requirements.model_copy(deep=True), geometry=geometry, recipe=input.model_copy(deep=True))
    request_path = directory / "request.json"
    write_json(request_path, input)
    domain.artifacts["preparation_request"] = artifact(request_path, "preparation_request", workspace)
    try:
        if geometry.source.uri:
            source = resolve_workspace_path(geometry.source.uri, workspace=workspace)
            target = directory / ("source_asset" + source.suffix)
            shutil.copyfile(source, target)
            if geometry.source.checksum and geometry.source.checksum.removeprefix("sha256:") != digest(target):
                raise ValueError("Source asset version changed.")
            geometry.source.uri = str(target)
            geometry.source.checksum = digest(target)
            if geometry.source.kind == "generated":
                geometry.source.kind = "stl" if target.suffix.lower() == ".stl" else "mesh_file"
            domain.artifacts["source_asset"] = artifact(target, "geometry_source", workspace)
            domain.artifacts["recipe_asset"] = domain.artifacts["source_asset"]
        elif geometry.source.kind != "generated":
            raise ValueError("A concrete source asset is required.")
        domain.checks["source_defined"] = True
        _check_mesh_policy(domain.requirements)
        domain.checks["mesh_policy"] = True
        if domain.requirements.representation == "background_grid":
            _check_surface_conversion(geometry)
        if input.requirements.representation == "mesh" and input.requirements.domain_side == "exterior":
            domain.status = "needs_input"
            domain.required_actions = ["provide_an_explicit_exterior_computational_domain_asset_or_use_an_enclosed_background_grid"]
        else:
            mesh = None
            repaired = False
            mesh_passed = False
            if input.repair == "always":
                geometry = _repair_surface(geometry, domain, input, directory, workspace)
                if geometry is None:
                    path = directory / "manifest.json"
                    write_json(path, domain)
                    return PrepareDomainOutput(domain=domain, manifest=artifact(path, "prepared_domain", workspace))
                repaired = True
            for attempt in range(input.max_meshing_attempts):
                attempt_dir = directory / f"mesh-{attempt}"
                missing = []
                mesh_passed = False
                try:
                    # Reuse is permitted only for the actual source mesh,
                    # and never during an explicit refinement study.
                    can_reuse = domain.requirements.representation == "mesh" and not input.force_remesh and attempt == 0 and geometry.source.kind == "mesh_file" and geometry.source.uri and domain.requirements.mesh_policy.target_element_size is None
                    if can_reuse:
                        raw = meshio.read(geometry.source.uri)
                        can_reuse = any(block.dim == geometry.dimension for block in raw.cells)
                    if can_reuse:
                        attempt_dir.mkdir()
                        mesh_path = attempt_dir / "mesh.msh"
                        shutil.copyfile(geometry.source.uri, mesh_path) if Path(geometry.source.uri).suffix.lower() == ".msh" else meshio.write(mesh_path, raw, file_format="gmsh22", binary=False)
                        mesh = MeshSpec(id=f"mesh:{revision}", kind="volume" if geometry.dimension == 3 else "surface", dimension=geometry.dimension, files=[ArtifactRef(uri=str(mesh_path), kind="mesh_file", format="msh")], topology=MeshTopology(cell_types=[block.type for block in raw.cells], node_count=len(raw.points), cell_count=sum(len(block.data) for block in raw.cells)), elements=ElementStats(total=sum(len(block.data) for block in raw.cells)))
                        domain.actions.append({"action": "reuse_source_mesh", "attempt": attempt})
                    else:
                        mesh = _native_mesh(geometry, domain.requirements, attempt_dir)
                        domain.actions.append({"action": "generate_mesh", "attempt": attempt, "target_element_size": domain.requirements.mesh_policy.target_element_size})
                    geometry = _mesh_geometry(mesh, geometry, domain.requirements)
                    quality = _quality(mesh)
                    missing = [name for name in domain.requirements.required_quality_metrics if not hasattr(quality, name) or getattr(quality, name) is None]
                    if missing:
                        quality.passes = False
                        quality.issues.extend(f"Required quality metric unavailable: {name}" for name in missing)
                    if quality.checked_element_orders != [domain.requirements.mesh_policy.element_order]:
                        quality.passes = False
                        quality.issues.append(f"Element orders {quality.checked_element_orders} do not match requested order {domain.requirements.mesh_policy.element_order}.")
                    if quality.aspect_ratio_p95 is not None and quality.aspect_ratio_p95 > domain.requirements.max_aspect_ratio_p95:
                        quality.passes = False
                        quality.issues.append("Aspect ratio exceeds kernel requirements.")
                    if quality.max_skewness is not None and quality.max_skewness > domain.requirements.max_skewness:
                        quality.passes = False
                        quality.issues.append("Skewness exceeds kernel requirements.")
                    mesh.quality = quality
                    domain.actions.append({"action": "check_mesh_quality", "attempt": attempt, "passes": quality.passes, "issues": quality.issues, "element_orders": quality.checked_element_orders})
                    if quality.passes:
                        mesh_passed = True
                        break
                    mesh_error = "; ".join(quality.issues)
                except _PreparationNeedsReview:
                    raise
                except Exception as exc:
                    mesh_error = str(exc)
                    domain.actions.append({"action": "meshing_failed", "attempt": attempt, "error": mesh_error})
                if missing:
                    domain.status = "needs_review"
                    domain.required_actions = ["provide_required_quality_evidence", *missing]
                    break
                if attempt + 1 >= input.max_meshing_attempts:
                    domain.status = "needs_review"
                    domain.required_actions = ["repair_or_remesh", mesh_error]
                    break
                if input.repair != "never" and not repaired and (mesh is None or domain.requirements.representation == "background_grid"):
                    repaired = True
                    repaired_geometry = _repair_surface(geometry, domain, input, directory, workspace)
                    if repaired_geometry is None:
                        break
                    geometry = repaired_geometry
                else:
                    domain.required_actions = ["repair_or_remesh", mesh_error]
                    domain.status = "needs_review"
                    if mesh is not None:
                        path = Path(next(item.uri for item in mesh.files if item.format == "msh"))
                        extent = float(np.ptp(meshio.read(path).points, axis=0).max())
                        domain.requirements.mesh_policy.target_element_size = (domain.requirements.mesh_policy.target_element_size or extent / 5) / 2
            if mesh is not None and mesh_passed:
                domain.required_actions = []
                replay_geometry = geometry.model_copy(deep=True)
                if domain.requirements.representation == "mesh":
                    domain.mesh = mesh
                    domain.artifacts.update(_mesh_artifacts(mesh, geometry, directory, workspace))
                    mesh.files = [artifact(resolve_workspace_path(item.uri, workspace=workspace), item.kind, workspace) for item in mesh.files]
                    domain.checks["mesh_quality"] = True
                else:
                    # The surface proxy's groups are not computational volumes.
                    # Replay source intent, not derived proxy region names.
                    replay_geometry.regions = [region.model_copy(deep=True) for region in input.geometry.regions]
                    native = Path(next(item.uri for item in mesh.files if item.format == "msh"))
                    vertices, faces = read_mesh_surface(native)
                    surface = directory / "grid_source.stl"
                    write_surface_stl(surface, vertices, faces)
                    geometry.source.uri = str(surface)
                    geometry.source.kind = "stl"
                    geometry.source.checksum = digest(surface)
                    domain.artifacts["source_asset"] = artifact(surface, "geometry_source", workspace)
                    domain.artifacts["surface_proxy_mesh"] = artifact(native, "surface_proxy_mesh", workspace)
                    references, checks = _grid_artifacts(geometry, domain.requirements, directory, workspace, mesh, input.geometry.regions)
                    domain.artifacts.update(references)
                    domain.checks.update(checks)
                domain.contract = build_geometry_mesh_contract(BuildGeometryMeshContractInput(geometry=geometry, mesh=domain.mesh)).contract
                gate = mesh_semantics_gate(MeshSemanticsGateInput(geometry=geometry, mesh=domain.mesh, contract=domain.contract))
                roles = {boundary.role for boundary in geometry.boundaries}
                missing_roles = set(domain.requirements.required_boundary_roles) - roles
                boundary_reference = domain.artifacts.get("boundary_nodes") or domain.artifacts.get("boundary_groups")
                with np.load(resolve_workspace_path(boundary_reference.uri, workspace=workspace), allow_pickle=False) if boundary_reference else _empty_context() as masks:
                    if masks is not None:
                        missing_roles.update(role for role in domain.requirements.required_boundary_roles if role not in masks or not masks[role].any())
                domain.checks["boundary_semantics"] = gate.passes and not missing_roles
                # In an exterior grid, source-body regions are outside the
                # computational domain and remain in the replay recipe.
                expected_regions = set() if domain.requirements.representation == "background_grid" and domain.requirements.domain_side == "exterior" else {region.label for region in input.geometry.regions}
                missing_regions = expected_regions - {region.label for region in geometry.regions}
                domain.checks["region_semantics"] = not missing_regions
                domain.required_actions = [*gate.required_actions, *(f"bind_boundary_role:{role}" for role in sorted(missing_roles)), *(f"bind_region:{name}" for name in sorted(missing_regions))]
                domain.warnings.extend(gate.warnings)
                domain.status = "ready" if all(domain.checks.values()) and not domain.required_actions else "needs_input"
                domain.geometry = geometry
                # Refinements replay the frozen, adopted source and its intent.
                domain.recipe.geometry = replay_geometry
                domain.recipe.geometry.encodings = []
                domain.recipe.requirements = domain.requirements.model_copy(deep=True)
                domain.recipe.repair = "never"
    except _PreparationNeedsReview as exc:
        domain.status = "needs_review"
        domain.checks[exc.check] = False
        domain.required_actions = exc.actions
    except Exception as exc:
        domain.status = "failed"
        domain.required_actions = [str(exc)]
        domain.checks["preparation_complete"] = False
    path = directory / "manifest.json"
    write_json(path, domain)
    return PrepareDomainOutput(domain=domain, manifest=artifact(path, "prepared_domain", workspace))


def _empty_context():
    from contextlib import nullcontext
    return nullcontext(None)
