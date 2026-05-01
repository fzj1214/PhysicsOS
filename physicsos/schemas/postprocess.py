from __future__ import annotations

from typing import Literal

from pydantic import Field

from physicsos.schemas.common import ArtifactRef, StrictBaseModel


class VisualizationSpec(StrictBaseModel):
    kind: Literal["contour", "streamline", "isosurface", "plot", "band_structure", "dos", "stress_map", "custom"]
    fields: list[str] = Field(default_factory=list)
    description: str | None = None


class PostprocessResult(StrictBaseModel):
    problem_id: str
    result_id: str
    kpis: dict[str, float | int | str] = Field(default_factory=dict)
    units: dict[str, str] = Field(default_factory=dict)
    visualizations: list[ArtifactRef] = Field(default_factory=list)
    report: ArtifactRef | None = None
    recommendations: list[str] = Field(default_factory=list)

