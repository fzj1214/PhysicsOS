from __future__ import annotations

from typing import Literal

from pydantic import Field

from physicsos.schemas.common import ArtifactRef, StrictBaseModel
from physicsos.schemas.geometry import BoundaryRegionSpec, RegionSpec


class MeshTopology(StrictBaseModel):
    cell_types: list[str] = Field(default_factory=list)
    node_count: int | None = None
    cell_count: int | None = None


class ElementStats(StrictBaseModel):
    total: int | None = None
    by_type: dict[str, int] = Field(default_factory=dict)


class MeshQualityReport(StrictBaseModel):
    min_jacobian: float | None = None
    max_skewness: float | None = None
    max_nonorthogonality: float | None = None
    aspect_ratio_p95: float | None = None
    boundary_layer_quality: dict[str, float] = Field(default_factory=dict)
    passes: bool = True
    issues: list[str] = Field(default_factory=list)


class MeshPolicy(StrictBaseModel):
    strategy: Literal["auto", "structured", "unstructured", "boundary_layer", "adaptive", "solver_native"] = "auto"
    target_element_size: float | None = None
    element_order: int = 1
    boundary_layer: bool = False
    refinement_regions: list[str] = Field(default_factory=list)


class MeshSpec(StrictBaseModel):
    id: str
    kind: Literal["structured", "unstructured", "hybrid", "surface", "volume", "particle", "k_space", "none"]
    dimension: Literal[0, 1, 2, 3]
    topology: MeshTopology = Field(default_factory=MeshTopology)
    elements: ElementStats = Field(default_factory=ElementStats)
    regions: list[RegionSpec] = Field(default_factory=list)
    boundaries: list[BoundaryRegionSpec] = Field(default_factory=list)
    quality: MeshQualityReport = Field(default_factory=MeshQualityReport)
    files: list[ArtifactRef] = Field(default_factory=list)
    solver_compatibility: list[str] = Field(default_factory=list)
