from __future__ import annotations

from typing import Literal

from physicsos.schemas.common import ExpressionRef, ScalarVectorTensor, StrictBaseModel
from physicsos.schemas.geometry import BoundaryRole


class BoundaryConditionSpec(StrictBaseModel):
    id: str
    region_id: str
    boundary_role: BoundaryRole | None = None
    field: str
    kind: Literal[
        "dirichlet",
        "neumann",
        "robin",
        "periodic",
        "symmetry",
        "wall",
        "inlet",
        "outlet",
        "interface",
        "farfield",
        "initial",
        "custom",
    ]
    value: ScalarVectorTensor | ExpressionRef | dict
    units: str | None = None
    confidence: float = 1.0
    source: Literal["user", "inferred", "template", "retrieved"] = "user"


class InitialConditionSpec(StrictBaseModel):
    id: str
    field: str
    value: ScalarVectorTensor | ExpressionRef
    units: str | None = None
