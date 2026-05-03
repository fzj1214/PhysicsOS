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
