from __future__ import annotations

import base64
import json
import math
from pathlib import Path
from typing import Literal
from urllib import request

from pydantic import Field

from physicsos.agents.structured import CoreAgentLLMConfig, StructuredLLMClient, call_structured_agent
from physicsos.backends.geometry_mesh import generate_mesh_backend, import_geometry_backend
from physicsos.config import project_root
from physicsos.schemas.common import ArtifactRef, StrictBaseModel
from physicsos.schemas.contracts import PhysicsProblemContract
from physicsos.schemas.geometry import BoundaryRegionSpec, BoundaryRole, GeometryEncoding, GeometryEntity, GeometryQualityReport, GeometrySource, GeometrySpec, GeometryTransform, RegionSpec
from physicsos.schemas.knowledge import KnowledgeContext
from physicsos.schemas.mesh import MeshPolicy, MeshQualityReport, MeshSpec
from physicsos.schemas.operators import PhysicsDomain, PhysicsSpec
from physicsos.schemas.problem import PhysicsProblem
from physicsos.tools.memory_tools import CaseMemoryContext


class ImportGeometryInput(StrictBaseModel):
    source: GeometrySource
    target_units: str = "SI"


class ImportGeometryOutput(StrictBaseModel):
    geometry: GeometrySpec
    artifacts: list[ArtifactRef] = Field(default_factory=list)


def import_geometry(input: ImportGeometryInput) -> ImportGeometryOutput:
    """Import CAD/mesh/material geometry into GeometrySpec."""
    geometry, artifacts = import_geometry_backend(input.source, target_units=input.target_units)
    return ImportGeometryOutput(geometry=geometry, artifacts=artifacts)


class RepairGeometryInput(StrictBaseModel):
    geometry: GeometrySpec
    repair_policy: str = "conservative"


class RepairGeometryOutput(StrictBaseModel):
    geometry: GeometrySpec
    changes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def repair_geometry(input: RepairGeometryInput) -> RepairGeometryOutput:
    """Repair invalid, non-manifold, open, or self-intersecting geometry."""
    geometry = input.geometry.model_copy(update={"quality": GeometryQualityReport(passes=True)})
    return RepairGeometryOutput(geometry=geometry, changes=["No-op scaffold repair."])


class LabelRegionsInput(StrictBaseModel):
    geometry: GeometrySpec
    physics_domain: PhysicsDomain
    hints: list[str] = Field(default_factory=list)


class LabelRegionsOutput(StrictBaseModel):
    geometry: GeometrySpec
    confidence_by_region: dict[str, float] = Field(default_factory=dict)
    unresolved_regions: list[str] = Field(default_factory=list)


class GeometryMeshPlanInput(StrictBaseModel):
    problem: PhysicsProblem
    problem_contract: PhysicsProblemContract | None = None
    requested_encodings: list[str] = Field(default_factory=list)
    target_backends: list[str] = Field(default_factory=list)
    knowledge_context: KnowledgeContext | None = None
    case_memory_context: CaseMemoryContext | None = None


class GeometryMeshPlanOutput(StrictBaseModel):
    mesh_policy: MeshPolicy = Field(default_factory=MeshPolicy)
    requested_encodings: list[str] = Field(default_factory=list)
    target_backends: list[str] = Field(default_factory=list)
    require_boundary_confirmation: bool = False
    boundary_confidence_threshold: float = 0.7
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


GEOMETRY_MESH_PLAN_SYSTEM_PROMPT = """You are the PhysicsOS geometry-mesh planning agent.
Return only a JSON object matching GeometryMeshPlanOutput.

Task:
- Preserve the locked PhysicsProblemContract when provided.
- Infer region and boundary semantics, mesh policy, backend targets, and required encodings.
- Do not claim that a mesh exists or directly write mesh files.
- Mark require_boundary_confirmation=true for imported/CAD/mesh geometries with ambiguous or low-confidence solver-critical boundary labels.
- Prefer mesh_graph encodings for 2D/3D TAPS/FEM workflows.
"""


def plan_geometry_mesh(input: GeometryMeshPlanInput) -> GeometryMeshPlanOutput:
    """Deterministic geometry/mesh planning fallback."""
    geometry = input.problem.geometry
    encodings = list(input.requested_encodings)
    if geometry.dimension > 1 and "mesh_graph" not in encodings:
        encodings.append("mesh_graph")
    boundary_confidence = min((boundary.confidence for boundary in geometry.boundaries), default=1.0)
    imported_geometry = geometry.source.kind in {"cad_step", "cad_iges", "stl", "mesh_file"}
    require_confirmation = imported_geometry and (not geometry.boundaries or boundary_confidence < 0.7)
    mesh_policy = MeshPolicy(
        strategy="unstructured" if geometry.dimension > 1 else "structured",
        target_element_size=0.2 if geometry.dimension > 1 else None,
        element_order=1,
        boundary_layer=input.problem.domain == "fluid",
    )
    return GeometryMeshPlanOutput(
        mesh_policy=mesh_policy,
        requested_encodings=encodings,
        target_backends=input.target_backends,
        require_boundary_confirmation=require_confirmation,
        assumptions=["Deterministic geometry-mesh fallback selected conservative mesh and encoding defaults."],
        warnings=["Imported geometry needs confirmed boundary labels before solver export."] if require_confirmation else [],
    )


def plan_geometry_mesh_structured(
    input: GeometryMeshPlanInput,
    *,
    client: StructuredLLMClient,
    config: CoreAgentLLMConfig | None = None,
) -> GeometryMeshPlanOutput:
    """LLM-backed geometry/mesh planning with strict Pydantic validation."""
    result = call_structured_agent(
        agent_name="geometry-mesh-planning-agent",
        input_model=input,
        output_model=GeometryMeshPlanOutput,
        system_prompt=GEOMETRY_MESH_PLAN_SYSTEM_PROMPT,
        client=client,
        config=config,
    )
    if result.output is not None:
        return result.output
    fallback = plan_geometry_mesh(input)
    return fallback.model_copy(
        update={
            "assumptions": [
                *fallback.assumptions,
                "Structured LLM geometry-mesh planning failed validation; deterministic fallback was used.",
                result.error or "Structured geometry-mesh planner returned no validated output.",
            ]
        }
    )


def _default_boundary_kind(label: str, physics_domain: PhysicsDomain) -> str:
    if physics_domain == "fluid":
        if label in {"x_min", "left"}:
            return "inlet"
        if label in {"x_max", "right"}:
            return "outlet"
        return "wall"
    if physics_domain == "electromagnetic":
        return "farfield"
    return "surface"


def _generated_boundary_labels(dimension: int) -> list[tuple[str, str]]:
    if dimension == 1:
        return [("x_min", "point"), ("x_max", "point")]
    if dimension == 2:
        return [("x_min", "curve"), ("x_max", "curve"), ("y_min", "curve"), ("y_max", "curve")]
    if dimension == 3:
        return [
            ("x_min", "surface"),
            ("x_max", "surface"),
            ("y_min", "surface"),
            ("y_max", "surface"),
            ("z_min", "surface"),
            ("z_max", "surface"),
        ]
    return []


def label_regions(input: LabelRegionsInput) -> LabelRegionsOutput:
    """Infer physical regions and boundary labels."""
    geometry = input.geometry.model_copy(deep=True)
    confidence_by_region: dict[str, float] = {}
    unresolved_regions: list[str] = []

    if not geometry.regions and geometry.dimension > 0:
        region_kind = "fluid" if input.physics_domain == "fluid" else "solid" if input.physics_domain in {"solid", "thermal"} else "custom"
        geometry.regions.append(RegionSpec(id="region:domain", label="domain", kind=region_kind))
        confidence_by_region["region:domain"] = 0.75 if geometry.source.kind == "generated" else 0.45
    for region in geometry.regions:
        confidence_by_region.setdefault(region.id, 1.0 if region.entity_ids else 0.75)

    if not geometry.boundaries and geometry.source.kind == "generated":
        for label, entity_kind in _generated_boundary_labels(geometry.dimension):
            entity_id = f"entity:boundary:{label}"
            if all(entity.id != entity_id for entity in geometry.entities):
                geometry.entities.append(GeometryEntity(id=entity_id, kind=entity_kind, label=label))  # type: ignore[arg-type]
            boundary_kind = _default_boundary_kind(label, input.physics_domain)
            geometry.boundaries.append(
                BoundaryRegionSpec(
                    id=f"boundary:{label}",
                    label=label,
                    kind=boundary_kind,  # type: ignore[arg-type]
                    entity_ids=[entity_id],
                    confidence=0.70,
                )
            )
            confidence_by_region[f"boundary:{label}"] = 0.70
    elif not geometry.boundaries:
        unresolved_regions.append("boundary_labels")
        quality = geometry.quality or GeometryQualityReport()
        geometry.quality = quality.model_copy(
            update={
                "passes": False,
                "unresolved_regions": sorted(set([*quality.unresolved_regions, "boundary_labels"])),
                "issues": sorted(set([*quality.issues, "Boundary labels are unresolved; user/CAD physical groups are required."])),
            }
        )
    else:
        for boundary in geometry.boundaries:
            confidence_by_region.setdefault(boundary.id, boundary.confidence)

    return LabelRegionsOutput(geometry=geometry, confidence_by_region=confidence_by_region, unresolved_regions=unresolved_regions)


class BoundaryLabelAssignment(StrictBaseModel):
    entity_ids: list[str] = Field(default_factory=list)
    boundary_id: str
    label: str
    kind: Literal["inlet", "outlet", "wall", "symmetry", "periodic", "interface", "farfield", "surface", "custom"] = "custom"
    role: BoundaryRole | None = None
    confidence: float = 1.0


class ApplyBoundaryLabelsInput(StrictBaseModel):
    geometry: GeometrySpec
    assignments: list[BoundaryLabelAssignment]
    replace_existing: bool = False
    source: Literal["user", "cad_physical_group", "knowledge_agent", "script"] = "user"


class ApplyBoundaryLabelsOutput(StrictBaseModel):
    geometry: GeometrySpec
    applied: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def apply_boundary_labels(input: ApplyBoundaryLabelsInput) -> ApplyBoundaryLabelsOutput:
    """Apply explicit user/CAD/knowledge-agent boundary labels to GeometrySpec."""
    geometry = input.geometry.model_copy(deep=True)
    warnings: list[str] = []
    applied: list[str] = []
    existing_entity_ids = {entity.id for entity in geometry.entities}
    if input.replace_existing:
        geometry.boundaries = []
    by_id = {boundary.id: boundary for boundary in geometry.boundaries}
    for assignment in input.assignments:
        missing = [entity_id for entity_id in assignment.entity_ids if entity_id not in existing_entity_ids]
        if missing:
            warnings.append(f"Assignment {assignment.boundary_id} references unknown entities: {', '.join(missing)}")
        confidence = max(0.0, min(1.0, assignment.confidence))
        boundary = BoundaryRegionSpec(
            id=assignment.boundary_id,
            label=assignment.label,
            kind=assignment.kind,
            entity_ids=[entity_id for entity_id in assignment.entity_ids if entity_id in existing_entity_ids],
            role=assignment.role,
            confidence=confidence,
        )
        if boundary.id in by_id:
            geometry.boundaries = [boundary if item.id == boundary.id else item for item in geometry.boundaries]
        else:
            geometry.boundaries.append(boundary)
        by_id[boundary.id] = boundary
        applied.append(boundary.id)
    unresolved = [item for item in (geometry.quality.unresolved_regions if geometry.quality else []) if item != "boundary_labels"]
    quality = geometry.quality or GeometryQualityReport()
    issues = [issue for issue in quality.issues if "Boundary labels are unresolved" not in issue]
    geometry.quality = quality.model_copy(update={"unresolved_regions": unresolved, "issues": issues, "passes": not unresolved and not issues})
    geometry.transforms.append(
        GeometryTransform(kind="custom", description=f"Applied {len(applied)} explicit boundary label assignments from {input.source}.")
    )
    return ApplyBoundaryLabelsOutput(geometry=geometry, applied=applied, warnings=warnings)


class BoundaryLabelCandidate(StrictBaseModel):
    target_ids: list[str] = Field(default_factory=list)
    boundary_id: str
    label: str
    kind: Literal["inlet", "outlet", "wall", "symmetry", "periodic", "interface", "farfield", "surface", "custom"] = "custom"
    role: BoundaryRole | None = None
    confidence: float = 0.0
    reason: str | None = None
    requires_confirmation: bool = True


class ConfirmedBoundaryLabel(StrictBaseModel):
    target_ids: list[str] = Field(default_factory=list)
    boundary_id: str
    label: str
    kind: Literal["inlet", "outlet", "wall", "symmetry", "periodic", "interface", "farfield", "surface", "custom"] = "custom"
    role: BoundaryRole | None = None
    confidence: float = 1.0
    confirmed_by: str = "user"


class CreateBoundaryLabelingArtifactInput(StrictBaseModel):
    geometry: GeometrySpec
    geometry_encoding: GeometryEncoding | None = None
    include_weak_suggestions: bool = True


class CreateBoundaryLabelingArtifactOutput(StrictBaseModel):
    artifact: ArtifactRef
    selectable_groups: list[dict[str, object]] = Field(default_factory=list)
    suggestions: list[BoundaryLabelCandidate] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _weak_kind_from_group_name(name: str, dimension: int) -> tuple[str, float, str] | None:
    lowered = name.lower()
    if any(token in lowered for token in ["inlet", "outlet", "wall", "symmetry", "periodic", "farfield", "port"]):
        if "inlet" in lowered:
            kind = "inlet"
        elif "outlet" in lowered:
            kind = "outlet"
        elif "wall" in lowered:
            kind = "wall"
        elif "symmetry" in lowered:
            kind = "symmetry"
        elif "periodic" in lowered:
            kind = "periodic"
        elif "farfield" in lowered:
            kind = "farfield"
        else:
            kind = "custom"
        if "port" in lowered:
            kind = "custom"
        return kind, 0.65, "Existing group name contains a common boundary-condition keyword."
    if dimension >= 2 and lowered in {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}:
        return "surface", 0.25, "Axis-aligned bounding-box group; physical meaning requires user confirmation."
    return None


def create_boundary_labeling_artifact(input: CreateBoundaryLabelingArtifactInput) -> CreateBoundaryLabelingArtifactOutput:
    """Create a human-confirmable boundary labeling artifact for a mesh/CAD viewer."""
    output_dir = project_root() / "scratch" / input.geometry.id.replace(":", "_") / "boundary_labeling"
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_payload = _mesh_graph_payload(input.geometry, input.geometry_encoding)
    selectable_groups: list[dict[str, object]] = []
    warnings: list[str] = []
    suggestions: list[BoundaryLabelCandidate] = []

    if graph_payload is not None:
        for raw_group in graph_payload.get("physical_boundary_groups", []):
            if not isinstance(raw_group, dict):
                continue
            name = str(raw_group.get("name") or "boundary")
            dimension = int(raw_group.get("dimension") or (2 if raw_group.get("face_ids") else 1))
            target_id = f"mesh_graph:physical:{raw_group.get('tag', name)}"
            selectable = {
                "id": target_id,
                "source": "mesh_graph_physical_group",
                "name": name,
                "dimension": dimension,
                "edge_ids": raw_group.get("edge_ids", []),
                "face_ids": raw_group.get("face_ids", []),
                "node_ids": raw_group.get("node_ids", []),
                "solver_native": raw_group.get("solver_native", {}),
            }
            selectable_groups.append(selectable)
            weak = _weak_kind_from_group_name(name, dimension)
            if input.include_weak_suggestions and weak is not None:
                kind, confidence, reason = weak
                suggestions.append(
                    BoundaryLabelCandidate(
                        target_ids=[target_id],
                        boundary_id=f"boundary:{name}",
                        label=name,
                        kind=kind,  # type: ignore[arg-type]
                        role=None,
                        confidence=confidence,
                        reason=reason,
                        requires_confirmation=True,
                    )
                )
    else:
        warnings.append("No mesh_graph encoding is available; labeling artifact only includes GeometrySpec boundaries.")

    for boundary in input.geometry.boundaries:
        target_ids = boundary.entity_ids or [boundary.id]
        selectable_groups.append(
            {
                "id": boundary.id,
                "source": "geometry_boundary",
                "name": boundary.label,
                "dimension": max(0, input.geometry.dimension - 1),
                "entity_ids": boundary.entity_ids,
                "confidence": boundary.confidence,
            }
        )
        if input.include_weak_suggestions and boundary.confidence < 1.0:
            suggestions.append(
                BoundaryLabelCandidate(
                    target_ids=target_ids,
                    boundary_id=boundary.id,
                    label=boundary.label,
                    kind=boundary.kind,
                    role=boundary.role,
                    confidence=boundary.confidence,
                    reason="GeometrySpec contains a non-confirmed boundary label.",
                    requires_confirmation=True,
                )
            )

    artifact_payload = {
        "schema_version": "physicsos.boundary_labeling.v1",
        "geometry_id": input.geometry.id,
        "source": "geometry_mesh_agent",
        "policy": {
            "weak_suggestions_require_confirmation": True,
            "solver_export_uses_confirmed_labels_only": True,
        },
        "viewer_geometry": (
            {
                "points": graph_payload.get("points", []),
                "edges": graph_payload.get("edges", []),
                "faces": graph_payload.get("faces", []),
                "boundary_edge_sets": graph_payload.get("boundary_edge_sets", {}),
                "boundary_face_sets": graph_payload.get("boundary_face_sets", {}),
                "bbox": graph_payload.get("bbox", {}),
            }
            if graph_payload is not None
            else {"points": [], "edges": [], "faces": [], "boundary_edge_sets": {}, "boundary_face_sets": {}, "bbox": {}}
        ),
        "selectable_groups": selectable_groups,
        "suggested_boundary_labels": [suggestion.model_dump() for suggestion in suggestions],
        "confirmed_boundary_labels": [],
        "warnings": warnings,
    }
    path = output_dir / "boundary_labeling_artifact.json"
    path.write_text(json.dumps(artifact_payload, indent=2), encoding="utf-8")
    artifact = ArtifactRef(
        uri=str(path),
        kind="boundary_labeling_artifact",
        format="json",
        description="Human-confirmable boundary labeling artifact for viewer/CLI workflows.",
    )
    return CreateBoundaryLabelingArtifactOutput(
        artifact=artifact,
        selectable_groups=selectable_groups,
        suggestions=suggestions,
        warnings=warnings,
    )


class ApplyBoundaryLabelingArtifactInput(StrictBaseModel):
    geometry: GeometrySpec
    labeling_artifact: ArtifactRef
    replace_existing: bool = False


class ApplyBoundaryLabelingArtifactOutput(StrictBaseModel):
    geometry: GeometrySpec
    applied: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def apply_boundary_labeling_artifact(input: ApplyBoundaryLabelingArtifactInput) -> ApplyBoundaryLabelingArtifactOutput:
    """Apply only confirmed boundary labels from a viewer/CLI labeling artifact."""
    payload = _read_json_artifact(input.labeling_artifact.uri)
    if payload is None:
        return ApplyBoundaryLabelingArtifactOutput(geometry=input.geometry, warnings=["Boundary labeling artifact is missing or invalid JSON."])
    raw_confirmed = payload.get("confirmed_boundary_labels", [])
    if not isinstance(raw_confirmed, list):
        return ApplyBoundaryLabelingArtifactOutput(geometry=input.geometry, warnings=["confirmed_boundary_labels must be a list."])

    geometry = input.geometry.model_copy(deep=True)
    if input.replace_existing:
        geometry.boundaries = []
    existing_entity_ids = {entity.id for entity in geometry.entities}
    assignments: list[BoundaryLabelAssignment] = []
    warnings: list[str] = []
    for raw_label in raw_confirmed:
        if not isinstance(raw_label, dict):
            warnings.append("Skipping malformed confirmed label entry.")
            continue
        confirmed = ConfirmedBoundaryLabel.model_validate(raw_label)
        entity_ids: list[str] = []
        for target_id in confirmed.target_ids:
            entity_id = target_id
            if entity_id not in existing_entity_ids:
                entity_kind = "surface" if geometry.dimension >= 3 else "curve" if geometry.dimension == 2 else "point"
                geometry.entities.append(GeometryEntity(id=entity_id, kind=entity_kind, label=confirmed.label))  # type: ignore[arg-type]
                existing_entity_ids.add(entity_id)
            entity_ids.append(entity_id)
        assignments.append(
            BoundaryLabelAssignment(
                entity_ids=entity_ids,
                boundary_id=confirmed.boundary_id,
                label=confirmed.label,
                kind=confirmed.kind,
                role=confirmed.role,
                confidence=max(0.0, min(1.0, confirmed.confidence)),
            )
        )
    applied = apply_boundary_labels(
        ApplyBoundaryLabelsInput(
            geometry=geometry,
            assignments=assignments,
            replace_existing=False,
            source="user",
        )
    )
    applied.geometry.transforms.append(
        GeometryTransform(kind="custom", description=f"Applied confirmed boundary labels from {input.labeling_artifact.uri}.")
    )
    return ApplyBoundaryLabelingArtifactOutput(
        geometry=applied.geometry,
        applied=applied.applied,
        warnings=[*warnings, *applied.warnings],
    )


class CreateGeometryLabelerViewerInput(StrictBaseModel):
    labeling_artifact: ArtifactRef
    title: str = "PhysicsOS Geometry Labeler"


class CreateGeometryLabelerViewerOutput(StrictBaseModel):
    viewer: ArtifactRef
    warnings: list[str] = Field(default_factory=list)


def _geometry_labeler_html(title: str, artifact_json: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{ color-scheme: dark; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    body {{ margin: 0; background: radial-gradient(circle at 20% 0%, rgba(34,211,238,.18), transparent 32%), #020617; color: #e2e8f0; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 28px; }}
    header, section {{ border: 1px solid rgba(255,255,255,.1); background: rgba(255,255,255,.04); border-radius: 28px; padding: 22px; }}
    h1 {{ margin: 8px 0 0; font-size: clamp(30px, 5vw, 54px); line-height: 1; }}
    .eyebrow {{ color: #67e8f9; font-size: 12px; letter-spacing: .28em; text-transform: uppercase; }}
    .grid {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(320px, .72fr); gap: 18px; margin-top: 18px; }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
    svg {{ width: 100%; height: auto; border-radius: 24px; background: radial-gradient(circle at top, rgba(34,211,238,.16), transparent 45%), #020617; }}
    button, select, input {{ border-radius: 999px; border: 1px solid rgba(255,255,255,.12); background: rgba(15,23,42,.9); color: #e2e8f0; padding: 10px 12px; }}
    button {{ cursor: pointer; }}
    button.primary {{ background: #67e8f9; color: #020617; border: 0; font-weight: 700; }}
    .groups {{ display: grid; gap: 8px; max-height: 300px; overflow: auto; }}
    .group {{ width: 100%; text-align: left; border-radius: 16px; }}
    .group.active {{ border-color: #67e8f9; background: rgba(34,211,238,.14); }}
    label {{ display: grid; gap: 6px; color: #94a3b8; font-size: 12px; letter-spacing: .16em; text-transform: uppercase; }}
    pre, textarea {{ box-sizing: border-box; width: 100%; min-height: 260px; overflow: auto; border: 1px solid rgba(255,255,255,.1); border-radius: 20px; background: rgba(2,6,23,.78); color: #cbd5e1; padding: 14px; font-size: 12px; line-height: 1.55; }}
    .controls {{ display: grid; gap: 12px; margin-top: 12px; }}
    .two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
  </style>
</head>
<body>
  <main>
    <header>
      <div class="eyebrow">PhysicsOS standalone tool</div>
      <h1>{title}</h1>
      <p>Rotate the mesh/facet artifact, select a face/edge group, confirm its physical label, then use the output JSON with <code>apply_boundary_labeling_artifact</code>.</p>
    </header>
    <div class="grid">
      <section>
        <svg id="viewer" viewBox="0 0 560 560"></svg>
        <div class="two">
          <label>Yaw <input id="yaw" type="range" min="-3.14" max="3.14" step="0.01" value="-0.65" /></label>
          <label>Pitch <input id="pitch" type="range" min="-1.4" max="1.4" step="0.01" value="0.45" /></label>
        </div>
      </section>
      <section>
        <h2>Selectable groups</h2>
        <div class="groups" id="groups"></div>
        <div class="controls">
          <label>Label <input id="label" value="boundary" /></label>
          <label>Kind
            <select id="kind">
              <option>inlet</option><option>outlet</option><option>wall</option><option>symmetry</option><option>periodic</option><option>interface</option><option>farfield</option><option selected>surface</option><option>custom</option>
            </select>
          </label>
          <button class="primary" id="confirm">Confirm selected group</button>
        </div>
      </section>
    </div>
    <section style="margin-top:18px">
      <h2>Confirmed artifact output</h2>
      <pre id="output"></pre>
    </section>
    <section style="margin-top:18px">
      <h2>Embedded input artifact</h2>
      <textarea id="input"></textarea>
    </section>
  </main>
  <script id="artifact" type="application/json">{artifact_json}</script>
  <script>
    const artifact = JSON.parse(document.getElementById('artifact').textContent);
    document.getElementById('input').value = JSON.stringify(artifact, null, 2);
    let selectedId = artifact.selectable_groups?.[0]?.id || null;
    let confirmed = artifact.confirmed_boundary_labels || [];
    const svg = document.getElementById('viewer');
    const yaw = document.getElementById('yaw');
    const pitch = document.getElementById('pitch');
    const label = document.getElementById('label');
    const kind = document.getElementById('kind');

    function inferKind(name) {{
      const s = String(name || '').toLowerCase();
      if (s.includes('inlet')) return 'inlet';
      if (s.includes('outlet')) return 'outlet';
      if (s.includes('wall')) return 'wall';
      if (s.includes('symmetry')) return 'symmetry';
      if (s.includes('periodic')) return 'periodic';
      if (s.includes('farfield')) return 'farfield';
      return 'surface';
    }}
    function points2d(points, yawValue, pitchValue) {{
      if (!points.length) return [];
      const min = [0,1,2].map(a => Math.min(...points.map(p => p[a] || 0)));
      const max = [0,1,2].map(a => Math.max(...points.map(p => p[a] || 0)));
      const center = min.map((v,a) => (v + max[a]) / 2);
      const span = Math.max(...max.map((v,a) => v - min[a]), 1e-6);
      const cy = Math.cos(yawValue), sy = Math.sin(yawValue), cp = Math.cos(pitchValue), sp = Math.sin(pitchValue);
      return points.map(p => {{
        const x0 = ((p[0] || 0) - center[0]) / span, y0 = ((p[1] || 0) - center[1]) / span, z0 = ((p[2] || 0) - center[2]) / span;
        const x1 = cy * x0 + sy * z0, z1 = -sy * x0 + cy * z0, y1 = cp * y0 - sp * z1, z2 = sp * y0 + cp * z1;
        return {{ x: 280 + x1 * 370, y: 280 - y1 * 370, z: z2 }};
      }});
    }}
    function selectGroup(id) {{
      selectedId = id;
      const group = artifact.selectable_groups.find(g => g.id === id);
      label.value = group?.name || 'boundary';
      kind.value = inferKind(group?.name);
      renderGroups();
      render();
    }}
    function renderGroups() {{
      const root = document.getElementById('groups');
      root.innerHTML = '';
      for (const group of artifact.selectable_groups || []) {{
        const button = document.createElement('button');
        button.className = 'group' + (group.id === selectedId ? ' active' : '');
        button.innerHTML = `<strong>${{group.name}}</strong><br/><small>${{group.id}}</small><br/><small>faces ${{group.face_ids?.length || 0}} · edges ${{group.edge_ids?.length || 0}}</small>`;
        button.onclick = () => selectGroup(group.id);
        root.appendChild(button);
      }}
    }}
    function render() {{
      const geom = artifact.viewer_geometry || {{}};
      const points = geom.points || [], faces = geom.faces || [], edges = geom.edges || [];
      const projected = points2d(points, Number(yaw.value), Number(pitch.value));
      const groupByFace = new Map();
      for (const group of artifact.selectable_groups || []) for (const faceId of group.face_ids || []) groupByFace.set(faceId, group);
      svg.innerHTML = '';
      faces.map((face, index) => ({{ face, index, depth: face.reduce((s,i) => s + (projected[i]?.z || 0), 0) / Math.max(face.length, 1), group: groupByFace.get(index) }}))
        .sort((a,b) => a.depth - b.depth)
        .forEach(item => {{
          const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
          polygon.setAttribute('points', item.face.map(i => `${{projected[i]?.x || 0}},${{projected[i]?.y || 0}}`).join(' '));
          polygon.setAttribute('fill', item.group?.id === selectedId ? 'rgba(34,211,238,.45)' : item.group ? 'rgba(148,163,184,.18)' : 'rgba(71,85,105,.12)');
          polygon.setAttribute('stroke', item.group?.id === selectedId ? 'rgb(103,232,249)' : 'rgba(255,255,255,.18)');
          polygon.setAttribute('stroke-width', item.group?.id === selectedId ? '3' : '1');
          polygon.style.cursor = item.group ? 'pointer' : 'default';
          if (item.group) polygon.onclick = () => selectGroup(item.group.id);
          svg.appendChild(polygon);
        }});
      for (const edge of edges) {{
        const a = projected[edge[0]], b = projected[edge[1]];
        if (!a || !b) continue;
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', a.x); line.setAttribute('y1', a.y); line.setAttribute('x2', b.x); line.setAttribute('y2', b.y);
        line.setAttribute('stroke', 'rgba(226,232,240,.35)'); line.setAttribute('stroke-width', '1.2');
        svg.appendChild(line);
      }}
    }}
    function renderOutput() {{
      document.getElementById('output').textContent = JSON.stringify({{ ...artifact, confirmed_boundary_labels: confirmed }}, null, 2);
    }}
    document.getElementById('confirm').onclick = () => {{
      if (!selectedId) return;
      const next = {{ target_ids: [selectedId], boundary_id: `boundary:${{label.value}}`, label: label.value, kind: kind.value, confidence: 1, confirmed_by: 'user' }};
      confirmed = confirmed.filter(item => !item.target_ids.includes(selectedId)).concat([next]);
      renderOutput();
    }};
    yaw.oninput = render; pitch.oninput = render;
    if (selectedId) selectGroup(selectedId);
    renderGroups(); render(); renderOutput();
  </script>
</body>
</html>
"""


def create_geometry_labeler_viewer(input: CreateGeometryLabelerViewerInput) -> CreateGeometryLabelerViewerOutput:
    """Create a standalone HTML viewer for confirming boundary labeling artifacts."""
    payload = _read_json_artifact(input.labeling_artifact.uri)
    warnings: list[str] = []
    if payload is None:
        return CreateGeometryLabelerViewerOutput(
            viewer=ArtifactRef(uri=input.labeling_artifact.uri, kind="geometry_labeler_viewer", format="html"),
            warnings=["Boundary labeling artifact is missing or invalid JSON; no viewer was created."],
        )
    if payload.get("schema_version") != "physicsos.boundary_labeling.v1":
        warnings.append(f"Unexpected labeling artifact schema: {payload.get('schema_version')}")
    source_path = Path(input.labeling_artifact.uri)
    output_dir = source_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    viewer_path = output_dir / "geometry_labeler_viewer.html"
    artifact_json = json.dumps(payload).replace("</", "<\\/")
    viewer_path.write_text(_geometry_labeler_html(input.title, artifact_json), encoding="utf-8")
    return CreateGeometryLabelerViewerOutput(
        viewer=ArtifactRef(
            uri=str(viewer_path),
            kind="geometry_labeler_viewer",
            format="html",
            description="Standalone browser UI for confirming PhysicsOS boundary labels.",
        ),
        warnings=warnings,
    )


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def _triangle_quality(points: list[list[float]], cell: list[int]) -> tuple[float, float, float] | None:
    if len(cell) < 3:
        return None
    p0, p1, p2 = (points[int(cell[0])], points[int(cell[1])], points[int(cell[2])])
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    jacobian = abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
    area = jacobian / 2.0
    if area <= 0.0:
        return None
    lengths = [
        math.dist(p0[:3], p1[:3]),
        math.dist(p1[:3], p2[:3]),
        math.dist(p2[:3], p0[:3]),
    ]
    shortest = min(lengths)
    if shortest <= 0.0:
        return None
    aspect_ratio = max(lengths) / shortest
    shape_quality = 4.0 * math.sqrt(3.0) * area / (sum(length * length for length in lengths) + 1e-30)
    skewness_proxy = 1.0 - max(0.0, min(1.0, shape_quality))
    return jacobian, aspect_ratio, skewness_proxy


def _mesh_quality_from_msh(mesh: MeshSpec) -> MeshQualityReport | None:
    msh_artifact = next((artifact for artifact in mesh.files if artifact.format == "msh"), None)
    if msh_artifact is None:
        return None
    try:
        import meshio
    except ImportError:
        return MeshQualityReport(passes=False, issues=["meshio is not installed; mesh quality could not be recomputed from .msh."])

    raw_mesh = meshio.read(msh_artifact.uri)
    points = [[float(coord) for coord in point[:3]] for point in raw_mesh.points]
    jacobians: list[float] = []
    aspect_ratios: list[float] = []
    skewness: list[float] = []
    unsupported_cell_types: set[str] = set()
    for block in raw_mesh.cells:
        if "triangle" not in block.type.lower():
            unsupported_cell_types.add(block.type)
            continue
        for raw_cell in block.data.tolist():
            quality = _triangle_quality(points, [int(node) for node in raw_cell])
            if quality is None:
                jacobians.append(0.0)
                continue
            jacobian, aspect_ratio, skewness_proxy = quality
            jacobians.append(jacobian)
            aspect_ratios.append(aspect_ratio)
            skewness.append(skewness_proxy)

    issues: list[str] = []
    min_jacobian = min(jacobians) if jacobians else None
    max_skewness = max(skewness) if skewness else None
    aspect_ratio_p95 = _percentile(aspect_ratios, 0.95)
    if min_jacobian is None:
        issues.append("No triangle cells found for current local mesh quality evaluator.")
    elif min_jacobian <= 1e-14:
        issues.append("Degenerate triangle cell detected.")
    if aspect_ratio_p95 is not None and aspect_ratio_p95 > 25.0:
        issues.append(f"High triangle aspect ratio p95={aspect_ratio_p95:.3g}.")
    if max_skewness is not None and max_skewness > 0.95:
        issues.append(f"High triangle skewness proxy max={max_skewness:.3g}.")
    if unsupported_cell_types and not jacobians:
        issues.append(f"Unsupported cell types for local evaluator: {', '.join(sorted(unsupported_cell_types))}.")
    passes = not issues
    return MeshQualityReport(
        min_jacobian=min_jacobian,
        max_skewness=max_skewness,
        aspect_ratio_p95=aspect_ratio_p95,
        passes=passes,
        issues=issues,
    )


class GenerateGeometryEncodingInput(StrictBaseModel):
    geometry: GeometrySpec
    mesh: MeshSpec | None = None
    encodings: list[str]
    resolutions: list[list[int]] = Field(default_factory=list)


class GenerateGeometryEncodingOutput(StrictBaseModel):
    encodings: list[GeometryEncoding]
    artifacts: list[ArtifactRef] = Field(default_factory=list)


def generate_geometry_encoding(input: GenerateGeometryEncodingInput) -> GenerateGeometryEncodingOutput:
    """Generate SDF, masks, graph, point cloud, or multiresolution grid encodings."""
    output_dir = project_root() / "scratch" / input.geometry.id.replace(":", "_") / "geometry_encodings"
    output_dir.mkdir(parents=True, exist_ok=True)
    encodings: list[GeometryEncoding] = []
    artifacts: list[ArtifactRef] = []
    for index, kind in enumerate(input.encodings):
        resolution = input.resolutions[index] if index < len(input.resolutions) else [32, 32]
        path = output_dir / f"{kind}.json"
        if kind == "occupancy_mask":
            nx = resolution[0] if resolution else 32
            ny = resolution[1] if len(resolution) > 1 else nx
            has_hole = any("hole" in entity.label.lower() for entity in input.geometry.entities if entity.label) or any(
                "hole" in region.label.lower() for region in input.geometry.regions
            )
            hole_center = [0.5, 0.5]
            hole_radius = 0.2
            mask: list[list[int]] = []
            for i in range(nx):
                x = i / (nx - 1) if nx > 1 else 0.0
                row: list[int] = []
                for j in range(ny):
                    y = j / (ny - 1) if ny > 1 else 0.0
                    in_hole = has_hole and ((x - hole_center[0]) ** 2 + (y - hole_center[1]) ** 2 <= hole_radius * hole_radius)
                    row.append(0 if in_hole else 1)
                mask.append(row)
            payload = {
                "type": "occupancy_mask",
                "geometry_id": input.geometry.id,
                "resolution": [nx, ny],
                "axes": {
                    "x": [i / (nx - 1) if nx > 1 else 0.0 for i in range(nx)],
                    "y": [j / (ny - 1) if ny > 1 else 0.0 for j in range(ny)],
                },
                "mask": mask,
                "active_value": 1,
                "boundary_policy": "dirichlet_zero",
                "holes": [{"shape": "circle", "center": hole_center, "radius": hole_radius}] if has_hole else [],
                "description": "Generated active-domain mask for geometry-encoded TAPS.",
            }
        elif kind == "sdf":
            nx = resolution[0] if resolution else 32
            ny = resolution[1] if len(resolution) > 1 else nx
            values = []
            for i in range(nx):
                x = i / (nx - 1) if nx > 1 else 0.0
                row = []
                for j in range(ny):
                    y = j / (ny - 1) if ny > 1 else 0.0
                    row.append(min(x, y, 1.0 - x, 1.0 - y))
                values.append(row)
            payload = {
                "type": "sdf",
                "geometry_id": input.geometry.id,
                "resolution": [nx, ny],
                "values": values,
                "description": "Signed-distance-like box interior distance for unit-square TAPS geometry encoding.",
            }
        elif kind == "mesh_graph":
            msh_artifact = None
            if input.mesh is not None:
                msh_artifact = next((artifact for artifact in input.mesh.files if artifact.format == "msh"), None)
            if msh_artifact is None:
                payload = {
                    "type": "mesh_graph",
                    "geometry_id": input.geometry.id,
                    "node_count": 0,
                    "edge_count": 0,
                    "description": "No .msh artifact was provided; mesh_graph could not be generated.",
                }
            else:
                try:
                    import meshio
                except ImportError:
                    payload = {
                        "type": "mesh_graph",
                        "geometry_id": input.geometry.id,
                        "source_mesh": msh_artifact.uri,
                        "node_count": 0,
                        "edge_count": 0,
                        "description": "meshio is not installed; mesh_graph could not be generated.",
                    }
                else:
                    mesh = meshio.read(msh_artifact.uri)
                    points = [[float(coord) for coord in point[:3]] for point in mesh.points]
                    edges: set[tuple[int, int]] = set()
                    faces: set[tuple[int, ...]] = set()
                    cell_blocks = []
                    for block in mesh.cells:
                        cells = [[int(node) for node in cell] for cell in block.data.tolist()]
                        cell_blocks.append({"type": block.type, "cells": cells})
                        for cell in cells:
                            for a_index, a in enumerate(cell):
                                for b in cell[a_index + 1 :]:
                                    edge = (a, b) if a < b else (b, a)
                                    edges.add(edge)
                            if block.type.startswith("triangle") or block.type.startswith("quad"):
                                vertices = cell[:3] if block.type.startswith("triangle") else cell[:4]
                                faces.add(tuple(sorted(vertices)))
                    mins = [min(point[axis] for point in points) for axis in range(3)] if points else [0.0, 0.0, 0.0]
                    maxs = [max(point[axis] for point in points) for axis in range(3)] if points else [0.0, 0.0, 0.0]
                    tolerance = 1e-10
                    boundary_node_sets = {
                        "x_min": [index for index, point in enumerate(points) if abs(point[0] - mins[0]) <= tolerance],
                        "x_max": [index for index, point in enumerate(points) if abs(point[0] - maxs[0]) <= tolerance],
                        "y_min": [index for index, point in enumerate(points) if abs(point[1] - mins[1]) <= tolerance],
                        "y_max": [index for index, point in enumerate(points) if abs(point[1] - maxs[1]) <= tolerance],
                    }
                    if input.geometry.dimension == 3:
                        boundary_node_sets["z_min"] = [index for index, point in enumerate(points) if abs(point[2] - mins[2]) <= tolerance]
                        boundary_node_sets["z_max"] = [index for index, point in enumerate(points) if abs(point[2] - maxs[2]) <= tolerance]
                    boundary_nodes = sorted({node for nodes in boundary_node_sets.values() for node in nodes})
                    boundary_edge_sets: dict[str, list[int]] = {}
                    sorted_edges = sorted(edges)
                    edge_index = {edge: index for index, edge in enumerate(sorted_edges)}
                    for name, nodes in boundary_node_sets.items():
                        node_set = set(nodes)
                        boundary_edge_sets[name] = [
                            index for index, (a, b) in enumerate(sorted_edges) if a in node_set and b in node_set
                        ]
                    boundary_edge_sets["boundary"] = sorted(
                        {edge for edge_ids in boundary_edge_sets.values() for edge in edge_ids}
                    )
                    sorted_faces = sorted(faces)
                    face_index = {face: index for index, face in enumerate(sorted_faces)}
                    boundary_face_sets: dict[str, list[int]] = {}
                    if input.geometry.dimension == 3:
                        for name, nodes in boundary_node_sets.items():
                            node_set = set(nodes)
                            boundary_face_sets[name] = [
                                index for index, face in enumerate(sorted_faces) if all(node in node_set for node in face)
                            ]
                        boundary_face_sets["boundary"] = sorted(
                            {face for face_ids in boundary_face_sets.values() for face in face_ids}
                        )
                    field_names_by_tag = {
                        int(values[0]): name
                        for name, values in getattr(mesh, "field_data", {}).items()
                        if len(values) >= 2 and int(values[1]) in {1, 2}
                    }
                    field_dims_by_tag = {
                        int(values[0]): int(values[1])
                        for values in getattr(mesh, "field_data", {}).values()
                        if len(values) >= 2 and int(values[1]) in {1, 2}
                    }
                    physical_boundary_groups: dict[int, dict[str, object]] = {}
                    physical_cell_data = getattr(mesh, "cell_data_dict", {}).get("gmsh:physical", {})
                    physical_offsets: dict[str, int] = {}
                    for block in mesh.cells:
                        is_line_boundary = block.type.startswith("line")
                        is_surface_boundary = input.geometry.dimension == 3 and (
                            block.type.startswith("triangle") or block.type.startswith("quad")
                        )
                        if not is_line_boundary and not is_surface_boundary:
                            continue
                        physical_tags = physical_cell_data.get(block.type)
                        if physical_tags is None:
                            continue
                        offset = physical_offsets.get(block.type, 0)
                        block_tags = physical_tags.tolist()[offset : offset + len(block.data)]
                        physical_offsets[block.type] = offset + len(block.data)
                        for cell, tag in zip(block.data.tolist(), block_tags):
                            if is_line_boundary and len(cell) < 2:
                                continue
                            if is_surface_boundary and len(cell) < 3:
                                continue
                            edge_id: int | None = None
                            face_id: int | None = None
                            edge_nodes: list[int] = []
                            face_nodes: list[int] = []
                            if is_line_boundary:
                                a = int(cell[0])
                                b = int(cell[-1])
                                edge = (a, b) if a < b else (b, a)
                                if edge not in edge_index:
                                    continue
                                edge_id = edge_index[edge]
                                edge_nodes = [a, b]
                            else:
                                vertices = [int(node) for node in (cell[:3] if block.type.startswith("triangle") else cell[:4])]
                                face = tuple(sorted(vertices))
                                if face not in face_index:
                                    continue
                                face_id = face_index[face]
                                face_nodes = vertices
                            physical_tag = int(tag)
                            physical_dim = field_dims_by_tag.get(physical_tag)
                            if physical_dim is not None and ((is_line_boundary and physical_dim != 1) or (is_surface_boundary and physical_dim != 2)):
                                continue
                            physical_name = field_names_by_tag.get(physical_tag) or f"physical_{physical_tag}"
                            group = physical_boundary_groups.setdefault(
                                physical_tag,
                                {
                                    "tag": physical_tag,
                                    "dimension": 1 if is_line_boundary else 2,
                                    "name": physical_name,
                                    "aliases": [f"physical:{physical_tag}", f"boundary:physical:{physical_tag}", physical_name, f"boundary:{physical_name}"],
                                    "edge_ids": [],
                                    "face_ids": [],
                                    "node_ids": [],
                                    "solver_native": {
                                        "gmsh_physical_tag": physical_tag,
                                        "gmsh_physical_name": physical_name,
                                        "openfoam_patch": physical_name,
                                        "su2_marker": physical_name,
                                        "fenicsx_facet_tag": physical_tag,
                                    },
                                },
                            )
                            names = {
                                f"physical:{physical_tag}",
                                f"boundary:physical:{physical_tag}",
                            }
                            if physical_name:
                                names.add(physical_name)
                                names.add(f"boundary:{physical_name}")
                            for name in names:
                                if edge_id is not None:
                                    boundary_edge_sets.setdefault(name, []).append(edge_id)
                                if face_id is not None:
                                    boundary_face_sets.setdefault(name, []).append(face_id)
                                boundary_node_sets.setdefault(name, [])
                                boundary_node_sets[name] = sorted(set([*boundary_node_sets[name], *edge_nodes, *face_nodes]))
                            if edge_id is not None:
                                group["edge_ids"] = sorted(set([*group["edge_ids"], edge_id]))  # type: ignore[index]
                            if face_id is not None:
                                group["face_ids"] = sorted(set([*group["face_ids"], face_id]))  # type: ignore[index]
                            group["node_ids"] = sorted(set([*group["node_ids"], *edge_nodes, *face_nodes]))  # type: ignore[index]
                    boundary_edge_sets = {name: sorted(set(edge_ids)) for name, edge_ids in boundary_edge_sets.items()}
                    boundary_face_sets = {name: sorted(set(face_ids)) for name, face_ids in boundary_face_sets.items()}
                    payload = {
                        "type": "mesh_graph",
                        "geometry_id": input.geometry.id,
                        "source_mesh": msh_artifact.uri,
                        "points": points,
                        "edges": [[a, b] for a, b in sorted_edges],
                        "faces": [list(face) for face in sorted_faces],
                        "boundary_nodes": boundary_nodes,
                        "boundary_node_sets": boundary_node_sets,
                        "boundary_edge_sets": boundary_edge_sets,
                        "boundary_face_sets": boundary_face_sets,
                        "physical_boundary_groups": sorted(physical_boundary_groups.values(), key=lambda group: str(group["name"])),
                        "bbox": {"min": mins, "max": maxs},
                        "cell_blocks": cell_blocks,
                        "node_count": len(points),
                        "edge_count": len(edges),
                        "description": "Graph encoding derived from Gmsh mesh connectivity with bbox boundary node labels.",
                    }
        else:
            payload = {
                "type": kind,
                "geometry_id": input.geometry.id,
                "resolution": resolution,
                "description": "Placeholder encoding metadata; concrete encoder not connected yet.",
            }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        encoding = GeometryEncoding(kind=kind, uri=str(Path(path)), resolution=resolution, target_backend="taps")  # type: ignore[arg-type]
        encodings.append(encoding)
        artifacts.append(ArtifactRef(uri=str(path), kind=f"geometry_encoding:{kind}", format="json"))
    return GenerateGeometryEncodingOutput(encodings=encodings, artifacts=artifacts)


class GenerateMeshInput(StrictBaseModel):
    geometry: GeometrySpec
    physics: PhysicsSpec
    mesh_policy: MeshPolicy = Field(default_factory=MeshPolicy)
    target_backends: list[str] = Field(default_factory=list)


class GenerateMeshOutput(StrictBaseModel):
    mesh: MeshSpec
    artifacts: list[ArtifactRef] = Field(default_factory=list)


def generate_mesh(input: GenerateMeshInput) -> GenerateMeshOutput:
    """Generate solver-compatible mesh from GeometrySpec."""
    mesh, artifacts = generate_mesh_backend(
        input.geometry,
        target_backends=input.target_backends,
        target_element_size=input.mesh_policy.target_element_size,
        element_order=input.mesh_policy.element_order,
    )
    return GenerateMeshOutput(mesh=mesh, artifacts=artifacts)


class ExportBackendMeshInput(StrictBaseModel):
    geometry: GeometrySpec
    mesh: MeshSpec
    backend: Literal["openfoam", "su2", "fenicsx", "mfem", "taps", "generic"]
    geometry_encoding: GeometryEncoding | None = None
    include_solver_native_hints: bool = True


class ExportBackendMeshOutput(StrictBaseModel):
    manifest: ArtifactRef
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    boundary_exports: list[dict[str, object]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _read_json_artifact(uri: str) -> dict[str, object] | None:
    path = Path(uri)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _mesh_graph_payload(geometry: GeometrySpec, explicit_encoding: GeometryEncoding | None) -> dict[str, object] | None:
    candidates = [explicit_encoding] if explicit_encoding is not None else []
    candidates.extend(encoding for encoding in geometry.encodings if encoding.kind == "mesh_graph")
    for encoding in candidates:
        if encoding is None:
            continue
        payload = _read_json_artifact(encoding.uri)
        if payload and payload.get("type") == "mesh_graph":
            return payload
    return None


def _backend_boundary_name(group: dict[str, object], backend: str) -> str:
    solver_native = group.get("solver_native")
    if isinstance(solver_native, dict):
        if backend == "openfoam" and isinstance(solver_native.get("openfoam_patch"), str):
            return solver_native["openfoam_patch"]
        if backend == "su2" and isinstance(solver_native.get("su2_marker"), str):
            return solver_native["su2_marker"]
        if backend in {"fenicsx", "mfem"} and solver_native.get("fenicsx_facet_tag") is not None:
            return str(solver_native["fenicsx_facet_tag"])
    name = group.get("name")
    return str(name) if name is not None else "boundary"


def _boundary_exports_from_graph(payload: dict[str, object], backend: str, include_hints: bool) -> list[dict[str, object]]:
    raw_groups = payload.get("physical_boundary_groups")
    if not isinstance(raw_groups, list):
        return []
    exports: list[dict[str, object]] = []
    for raw_group in raw_groups:
        if not isinstance(raw_group, dict):
            continue
        name = str(raw_group.get("name") or "boundary")
        edge_ids = raw_group.get("edge_ids") if isinstance(raw_group.get("edge_ids"), list) else []
        face_ids = raw_group.get("face_ids") if isinstance(raw_group.get("face_ids"), list) else []
        node_ids = raw_group.get("node_ids") if isinstance(raw_group.get("node_ids"), list) else []
        export: dict[str, object] = {
            "source_name": name,
            "backend_name": _backend_boundary_name(raw_group, backend),
            "gmsh_physical_tag": raw_group.get("tag"),
            "dimension": raw_group.get("dimension"),
            "edge_ids": [int(edge_id) for edge_id in edge_ids if isinstance(edge_id, int)],
            "face_ids": [int(face_id) for face_id in face_ids if isinstance(face_id, int)],
            "node_ids": [int(node_id) for node_id in node_ids if isinstance(node_id, int)],
        }
        if include_hints and isinstance(raw_group.get("solver_native"), dict):
            export["solver_native"] = raw_group["solver_native"]
        exports.append(export)
    return sorted(exports, key=lambda item: str(item["source_name"]))


def _boundary_exports_from_geometry(geometry: GeometrySpec, backend: str) -> list[dict[str, object]]:
    exports: list[dict[str, object]] = []
    for boundary in geometry.boundaries:
        name = boundary.label or boundary.id.removeprefix("boundary:")
        if backend in {"fenicsx", "mfem"} and boundary.id.startswith("boundary:physical:"):
            backend_name = boundary.id.removeprefix("boundary:physical:")
        else:
            backend_name = name
        exports.append(
            {
                "source_name": name,
                "backend_name": backend_name,
                "boundary_id": boundary.id,
                "entity_ids": boundary.entity_ids,
                "confidence": boundary.confidence,
            }
        )
    return exports


def export_backend_mesh(input: ExportBackendMeshInput) -> ExportBackendMeshOutput:
    """Export a solver-facing mesh manifest with physical boundary mappings."""
    output_dir = project_root() / "scratch" / input.geometry.id.replace(":", "_") / "backend_mesh_exports"
    output_dir.mkdir(parents=True, exist_ok=True)
    source_mesh = next((artifact for artifact in input.mesh.files if artifact.format == "msh"), None)
    graph_payload = _mesh_graph_payload(input.geometry, input.geometry_encoding)
    boundary_exports = (
        _boundary_exports_from_graph(graph_payload, input.backend, input.include_solver_native_hints)
        if graph_payload is not None
        else _boundary_exports_from_geometry(input.geometry, input.backend)
    )
    warnings: list[str] = []
    if source_mesh is None:
        warnings.append("No Gmsh .msh artifact is attached to MeshSpec; backend runner must supply or regenerate mesh.")
    if not boundary_exports:
        warnings.append("No physical boundary groups are available; backend runner may need user/CAD boundary labels.")

    mesh_formats = {
        "openfoam": {"target": "constant/polyMesh", "conversion": "gmshToFoam or meshio-to-OpenFOAM runner step"},
        "su2": {"target": ".su2", "conversion": "meshio or SU2-compatible mesh conversion runner step"},
        "fenicsx": {"target": ".xdmf/.h5 with facet tags", "conversion": "meshio XDMF writer with gmsh:physical facet data"},
        "mfem": {"target": ".mesh", "conversion": "meshio/MFEM conversion runner step"},
        "taps": {"target": "mesh_graph.json", "conversion": "PhysicsOS mesh_graph encoding"},
        "generic": {"target": ".msh + boundary manifest", "conversion": "no solver-specific conversion requested"},
    }
    manifest_payload = {
        "schema_version": "physicsos.backend_mesh_export.v1",
        "geometry_id": input.geometry.id,
        "mesh_id": input.mesh.id,
        "backend": input.backend,
        "source_mesh": source_mesh.model_dump() if source_mesh is not None else None,
        "source_mesh_graph": graph_payload.get("source_mesh") if graph_payload else None,
        "target": mesh_formats[input.backend],
        "execution_policy": {
            "external_conversion_execution": "disabled_until_runner_approval",
            "local_tool_invocation": False,
        },
        "boundary_exports": boundary_exports,
        "regions": [region.model_dump() for region in input.mesh.regions],
        "warnings": warnings,
    }
    manifest_path = output_dir / f"{input.backend}_mesh_export_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    manifest = ArtifactRef(
        uri=str(manifest_path),
        kind="backend_mesh_export_manifest",
        format="json",
        description=f"{input.backend} mesh export manifest with physical boundary mappings",
    )
    return ExportBackendMeshOutput(manifest=manifest, artifacts=[manifest], boundary_exports=boundary_exports, warnings=warnings)


class PrepareMeshConversionJobInput(StrictBaseModel):
    mesh_export_manifest: ArtifactRef
    service_base_url: str | None = None
    inline_source_mesh: bool = True
    max_inline_bytes: int = 5_000_000


class PrepareMeshConversionJobOutput(StrictBaseModel):
    runner_manifest: ArtifactRef
    requires_approval: bool = True
    warnings: list[str] = Field(default_factory=list)


class SubmitMeshConversionJobInput(StrictBaseModel):
    runner_manifest: ArtifactRef
    mode: Literal["dry_run", "http"] = "dry_run"
    approval_token: str | None = None
    service_base_url: str | None = None


class SubmitMeshConversionJobOutput(StrictBaseModel):
    runner_response: ArtifactRef
    submitted: bool = False
    status: Literal["validated", "submitted"] = "validated"
    warnings: list[str] = Field(default_factory=list)


def _load_mesh_export_manifest(artifact: ArtifactRef) -> dict[str, object]:
    if artifact.kind != "backend_mesh_export_manifest":
        raise ValueError(f"Expected backend_mesh_export_manifest artifact, got {artifact.kind}.")
    payload = _read_json_artifact(artifact.uri)
    if payload is None:
        raise ValueError(f"Could not read backend mesh export manifest: {artifact.uri}")
    if payload.get("schema_version") != "physicsos.backend_mesh_export.v1":
        raise ValueError(f"Unsupported backend mesh export manifest schema: {payload.get('schema_version')}")
    return payload


def _inline_source_mesh(source_mesh: object, max_inline_bytes: int) -> tuple[dict[str, object] | None, str | None]:
    if not isinstance(source_mesh, dict):
        return None, "Mesh export manifest does not include a source_mesh artifact."
    uri = source_mesh.get("uri")
    if not isinstance(uri, str):
        return None, "source_mesh.uri is missing; runner must receive the mesh by another channel."
    path = Path(uri)
    if not path.exists():
        return None, f"source mesh path is not readable from this workspace: {uri}"
    size = path.stat().st_size
    if size > max_inline_bytes:
        return None, f"source mesh is {size} bytes, larger than max_inline_bytes={max_inline_bytes}; upload by artifact channel instead."
    return (
        {
            "path": path.name,
            "format": source_mesh.get("format"),
            "kind": source_mesh.get("kind"),
            "size": size,
            "content_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
        },
        None,
    )


def _mesh_conversion_workspace(export_manifest: dict[str, object]) -> Path:
    geometry_id = str(export_manifest.get("geometry_id") or "geometry")
    backend = str(export_manifest.get("backend") or "generic")
    path = project_root() / "scratch" / geometry_id.replace(":", "_") / "mesh_conversion" / backend
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_mesh_conversion_job(input: PrepareMeshConversionJobInput) -> PrepareMeshConversionJobOutput:
    """Prepare a runner-side mesh conversion job from a backend mesh export manifest."""
    export_manifest = _load_mesh_export_manifest(input.mesh_export_manifest)
    workspace = _mesh_conversion_workspace(export_manifest)
    warnings = list(export_manifest.get("warnings", [])) if isinstance(export_manifest.get("warnings"), list) else []
    source_mesh_file = None
    if input.inline_source_mesh:
        source_mesh_file, warning = _inline_source_mesh(export_manifest.get("source_mesh"), input.max_inline_bytes)
        if warning is not None:
            warnings.append(warning)

    backend = str(export_manifest.get("backend") or "generic")
    runner_manifest = {
        "schema_version": "physicsos.mesh_conversion_job.v1",
        "job_type": "mesh_conversion",
        "geometry_id": export_manifest.get("geometry_id"),
        "mesh_id": export_manifest.get("mesh_id"),
        "backend": backend,
        "service": {
            "base_url": input.service_base_url,
            "mode": "prepare_only",
            "requires_approval_token": True,
        },
        "inputs": {
            "mesh_export_manifest": export_manifest,
            "source_mesh_file": source_mesh_file,
            "boundary_exports": export_manifest.get("boundary_exports", []),
        },
        "conversion_plan": {
            "target": export_manifest.get("target", {}),
            "backend": backend,
            "runner_required": True,
            "allowed_converters": {
                "openfoam": ["gmshToFoam", "meshio"],
                "su2": ["meshio"],
                "fenicsx": ["meshio-xdmf"],
                "mfem": ["meshio-mfem"],
                "taps": ["physicsos-mesh-graph"],
                "generic": ["copy-msh-and-boundary-manifest"],
            }.get(backend, ["meshio"]),
        },
        "execution_policy": {
            "sandboxed_workspace": str(workspace),
            "local_external_process_execution": False,
            "runner_external_process_execution": "requires_approval_token",
            "network_access": "runner_service_only",
            "artifact_collection": ["converted_mesh", "boundary_mapping", "conversion_log"],
        },
        "warnings": warnings,
    }
    path = workspace / "mesh_conversion_runner_manifest.json"
    path.write_text(json.dumps(runner_manifest, indent=2), encoding="utf-8")
    artifact = ArtifactRef(
        uri=str(path),
        kind="mesh_conversion_runner_manifest",
        format="json",
        description="Prepared runner-side mesh conversion manifest.",
    )
    return PrepareMeshConversionJobOutput(runner_manifest=artifact, warnings=warnings)


def _load_mesh_conversion_manifest(artifact: ArtifactRef) -> dict[str, object]:
    if artifact.kind != "mesh_conversion_runner_manifest":
        raise ValueError(f"Expected mesh_conversion_runner_manifest artifact, got {artifact.kind}.")
    payload = _read_json_artifact(artifact.uri)
    if payload is None:
        raise ValueError(f"Could not read mesh conversion runner manifest: {artifact.uri}")
    if payload.get("schema_version") != "physicsos.mesh_conversion_job.v1":
        raise ValueError(f"Unsupported mesh conversion runner manifest schema: {payload.get('schema_version')}")
    return payload


def _mesh_conversion_response_path(manifest: dict[str, object], mode: str) -> Path:
    workspace = manifest.get("execution_policy", {}).get("sandboxed_workspace") if isinstance(manifest.get("execution_policy"), dict) else None
    path = Path(str(workspace)) if workspace else _mesh_conversion_workspace({"geometry_id": manifest.get("geometry_id"), "backend": manifest.get("backend")})
    path.mkdir(parents=True, exist_ok=True)
    safe_mode = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in mode)
    return path / f"mesh_conversion_response_{safe_mode}.json"


def _write_mesh_conversion_response(manifest: dict[str, object], payload: dict[str, object]) -> ArtifactRef:
    path = _mesh_conversion_response_path(manifest, str(payload.get("mode", "runner")))
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return ArtifactRef(uri=str(path), kind="mesh_conversion_runner_response", format="json")


def _http_submit_mesh_conversion(manifest: dict[str, object], service_base_url: str, approval_token: str) -> dict[str, object]:
    endpoint = service_base_url.rstrip("/") + "/api/physicsos/jobs"
    body = json.dumps(manifest).encode("utf-8")
    req = request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {approval_token}",
        },
    )
    with request.urlopen(req, timeout=30) as response:
        text = response.read().decode("utf-8")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = {"raw_response": text}
    return {"endpoint": endpoint, "status": "submitted", "response": parsed}


def submit_mesh_conversion_job(input: SubmitMeshConversionJobInput) -> SubmitMeshConversionJobOutput:
    """Submit a mesh conversion runner manifest, or validate it in dry-run mode."""
    manifest = _load_mesh_conversion_manifest(input.runner_manifest)
    warnings = list(manifest.get("warnings", [])) if isinstance(manifest.get("warnings"), list) else []
    if input.mode == "dry_run":
        payload = {
            "mode": "dry_run",
            "status": "validated",
            "submitted": False,
            "message": "Mesh conversion manifest is valid; no external conversion service or CLI was invoked.",
            "manifest": manifest,
        }
        response = _write_mesh_conversion_response(manifest, payload)
        return SubmitMeshConversionJobOutput(runner_response=response, submitted=False, status="validated", warnings=warnings)

    service_base_url = input.service_base_url or (
        manifest.get("service", {}).get("base_url") if isinstance(manifest.get("service"), dict) else None
    )
    if not service_base_url:
        raise ValueError("HTTP mesh conversion mode requires service_base_url in input or manifest.")
    if not input.approval_token:
        raise PermissionError("HTTP mesh conversion mode requires approval_token.")
    payload = _http_submit_mesh_conversion(manifest, str(service_base_url), input.approval_token)
    response = _write_mesh_conversion_response(manifest, payload)
    return SubmitMeshConversionJobOutput(runner_response=response, submitted=True, status="submitted", warnings=warnings)


class AssessMeshQualityInput(StrictBaseModel):
    mesh: MeshSpec
    physics: PhysicsSpec
    backend: str | None = None


class AssessMeshQualityOutput(StrictBaseModel):
    report: MeshQualityReport
    recommended_action: str


def assess_mesh_quality(input: AssessMeshQualityInput) -> AssessMeshQualityOutput:
    """Evaluate mesh quality for selected physics and backend."""
    computed = _mesh_quality_from_msh(input.mesh)
    report = computed or input.mesh.quality
    issues = list(report.issues)
    if input.backend in {"taps", "taps:mesh_fem_poisson"} and not any("triangle" in cell_type.lower() for cell_type in input.mesh.topology.cell_types):
        issues.append("TAPS mesh FEM path expects triangle cells in the current local assembler.")
    passes = report.passes and not issues
    report = report.model_copy(update={"passes": passes, "issues": sorted(set(issues))})
    if passes:
        action = "accept"
    elif report.min_jacobian is not None and report.min_jacobian <= 1e-14:
        action = "remesh"
    elif report.aspect_ratio_p95 is not None or report.max_skewness is not None:
        action = "refine_or_remesh"
    else:
        action = "inspect_mesh"
    return AssessMeshQualityOutput(report=report, recommended_action=action)


for _tool, _input, _output in [
    (import_geometry, ImportGeometryInput, ImportGeometryOutput),
    (repair_geometry, RepairGeometryInput, RepairGeometryOutput),
    (label_regions, LabelRegionsInput, LabelRegionsOutput),
    (plan_geometry_mesh, GeometryMeshPlanInput, GeometryMeshPlanOutput),
    (plan_geometry_mesh_structured, GeometryMeshPlanInput, GeometryMeshPlanOutput),
    (apply_boundary_labels, ApplyBoundaryLabelsInput, ApplyBoundaryLabelsOutput),
    (create_boundary_labeling_artifact, CreateBoundaryLabelingArtifactInput, CreateBoundaryLabelingArtifactOutput),
    (apply_boundary_labeling_artifact, ApplyBoundaryLabelingArtifactInput, ApplyBoundaryLabelingArtifactOutput),
    (create_geometry_labeler_viewer, CreateGeometryLabelerViewerInput, CreateGeometryLabelerViewerOutput),
    (generate_geometry_encoding, GenerateGeometryEncodingInput, GenerateGeometryEncodingOutput),
    (generate_mesh, GenerateMeshInput, GenerateMeshOutput),
    (export_backend_mesh, ExportBackendMeshInput, ExportBackendMeshOutput),
    (prepare_mesh_conversion_job, PrepareMeshConversionJobInput, PrepareMeshConversionJobOutput),
    (submit_mesh_conversion_job, SubmitMeshConversionJobInput, SubmitMeshConversionJobOutput),
    (assess_mesh_quality, AssessMeshQualityInput, AssessMeshQualityOutput),
]:
    _tool.input_model = _input
    _tool.output_model = _output
    _tool.side_effects = "workspace artifacts only"
    _tool.requires_approval = _tool.__name__ == "submit_mesh_conversion_job"
