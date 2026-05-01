from __future__ import annotations

from typing import Literal

from pydantic import Field

from physicsos.schemas.common import StrictBaseModel


class MaterialProperty(StrictBaseModel):
    name: str
    value: float | int | str | list[float]
    units: str | None = None
    temperature_dependence: str | None = None


class MaterialSpec(StrictBaseModel):
    id: str
    name: str
    phase: Literal["solid", "liquid", "gas", "plasma", "crystal", "molecule", "mixture", "custom"]
    region_ids: list[str] = Field(default_factory=list)
    properties: list[MaterialProperty] = Field(default_factory=list)

