from __future__ import annotations

from typing import Literal

from pydantic import Field

from physicsos.schemas.common import ArtifactRef, StrictBaseModel

BoundaryRole = Literal[
    "x_min",
    "x_max",
    "y_min",
    "y_max",
    "z_min",
    "z_max",
    "side_wall",
    "inlet",
    "outlet",
    "wall",
    "symmetry",
    "farfield",
    "interface",
    "periodic",
    "custom",
]


class CoordinateSystem(StrictBaseModel):
    kind: Literal["cartesian", "cylindrical", "spherical", "crystal", "custom"] = "cartesian"
    units: str = "m"


class GeometrySource(StrictBaseModel):
    kind: Literal[
        "text",
        "cad_step",
        "cad_iges",
        "stl",
        "mesh_file",
        "cif",
        "poscar",
        "molecular_graph",
        "image",
        "generated",
    ]
    uri: str | None = None
    checksum: str | None = None


class GeometryEntity(StrictBaseModel):
    id: str
    kind: Literal["point", "curve", "surface", "solid", "cell", "atom", "molecule", "region"]
    label: str | None = None
    artifact: ArtifactRef | None = None
    metadata: dict[str, str | float | int | bool] = Field(default_factory=dict)


class RegionSpec(StrictBaseModel):
    id: str
    label: str
    kind: Literal["fluid", "solid", "void", "material", "interface", "periodic_cell", "custom"]
    entity_ids: list[str] = Field(default_factory=list)


class BoundaryRegionSpec(StrictBaseModel):
    id: str
    label: str
    kind: Literal[
        "inlet",
        "outlet",
        "wall",
        "symmetry",
        "periodic",
        "interface",
        "farfield",
        "surface",
        "custom",
    ]
    entity_ids: list[str] = Field(default_factory=list)
    role: BoundaryRole | None = None
    confidence: float = 1.0


class GeometryTransform(StrictBaseModel):
    kind: Literal["scale", "translate", "rotate", "repair", "boolean", "unit_conversion", "custom"]
    description: str


class GeometryEncoding(StrictBaseModel):
    kind: Literal[
        "structured_axes",
        "sdf",
        "occupancy_mask",
        "surface_point_cloud",
        "volume_point_cloud",
        "mesh_graph",
        "boundary_graph",
        "laplacian_eigenbasis",
        "multi_resolution_grid",
        "parametric_shape_vector",
    ]
    uri: str
    resolution: list[int] | None = None
    feature_names: list[str] = Field(default_factory=list)
    target_backend: str | None = None


class GeometrySemanticRegion(StrictBaseModel):
    id: str
    label: str
    kind: Literal["domain", "subdomain", "material", "interface", "boundary", "source_support", "initial_slice", "custom"]
    role: BoundaryRole | None = None
    entity_ids: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    source: Literal["user", "cad_physical_group", "mesh_tag", "generated", "inferred", "unknown"] = "unknown"
    metadata: dict[str, str | float | int | bool] = Field(default_factory=dict)


class GeometrySemanticContract(StrictBaseModel):
    geometry_id: str
    dimension: Literal[0, 1, 2, 3]
    coordinate_system: CoordinateSystem = Field(default_factory=CoordinateSystem)
    domains: list[GeometrySemanticRegion] = Field(default_factory=list)
    subdomains: list[GeometrySemanticRegion] = Field(default_factory=list)
    boundaries: list[GeometrySemanticRegion] = Field(default_factory=list)
    interfaces: list[GeometrySemanticRegion] = Field(default_factory=list)
    source_supports: list[GeometrySemanticRegion] = Field(default_factory=list)
    unresolved_bindings: list[str] = Field(default_factory=list)
    min_confidence: float = 1.0
    provenance: dict[str, str | float | int | bool] = Field(default_factory=dict)


class GeometryNumericalEncoding(StrictBaseModel):
    kind: Literal[
        "structured_axes",
        "mesh_graph",
        "sdf",
        "occupancy_mask",
        "boundary_graph",
        "laplacian_eigenbasis",
        "multi_resolution_grid",
        "parametric_shape_vector",
        "nurbs_mapping",
        "separated_geometry_operator",
    ]
    uri: str | None = None
    target_backend: str | None = None
    axis_names: list[str] = Field(default_factory=list)
    resolution: list[int] | None = None
    quality: dict[str, float | int | bool | str] = Field(default_factory=dict)
    metadata: dict[str, str | float | int | bool] = Field(default_factory=dict)


class GeometryMeshContract(StrictBaseModel):
    semantic: GeometrySemanticContract
    numerical_encodings: list[GeometryNumericalEncoding] = Field(default_factory=list)
    mesh_export_manifest: ArtifactRef | None = None
    warnings: list[str] = Field(default_factory=list)


class GeometryQualityReport(StrictBaseModel):
    watertight: bool | None = None
    manifold: bool | None = None
    self_intersections: int | None = None
    unresolved_regions: list[str] = Field(default_factory=list)
    passes: bool = True
    issues: list[str] = Field(default_factory=list)


class GeometrySpec(StrictBaseModel):
    id: str
    source: GeometrySource
    dimension: Literal[0, 1, 2, 3]
    coordinate_system: CoordinateSystem = Field(default_factory=CoordinateSystem)
    entities: list[GeometryEntity] = Field(default_factory=list)
    regions: list[RegionSpec] = Field(default_factory=list)
    boundaries: list[BoundaryRegionSpec] = Field(default_factory=list)
    transforms: list[GeometryTransform] = Field(default_factory=list)
    encodings: list[GeometryEncoding] = Field(default_factory=list)
    quality: GeometryQualityReport | None = None
