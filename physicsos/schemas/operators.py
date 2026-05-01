from __future__ import annotations

from typing import Literal

from pydantic import Field

from physicsos.schemas.common import StrictBaseModel

PhysicsDomain = Literal[
    "fluid",
    "thermal",
    "solid",
    "electromagnetic",
    "acoustic",
    "molecular",
    "quantum",
    "multiphysics",
    "custom",
]


class FieldSpec(StrictBaseModel):
    name: str
    kind: Literal["scalar", "vector", "tensor"]
    units: str | None = None
    location: Literal["cell", "node", "face", "particle", "global", "k_point"] = "cell"


class DifferentialTerm(StrictBaseModel):
    expression: str
    order: int | None = None
    fields: list[str] = Field(default_factory=list)


class SourceTerm(StrictBaseModel):
    expression: str
    units: str | None = None


class CouplingSpec(StrictBaseModel):
    source_operator_id: str
    target_operator_id: str
    description: str


class NondimensionalNumber(StrictBaseModel):
    name: str
    value: float


class OperatorSpec(StrictBaseModel):
    id: str
    name: str
    domain: PhysicsDomain
    equation_class: str
    form: Literal["strong", "weak", "integral", "discrete", "learned"]
    fields_in: list[str] = Field(default_factory=list)
    fields_out: list[str] = Field(default_factory=list)
    conserved_quantities: list[str] = Field(default_factory=list)
    differential_terms: list[DifferentialTerm] = Field(default_factory=list)
    source_terms: list[SourceTerm] = Field(default_factory=list)
    coupling: list[CouplingSpec] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    nondimensional_numbers: list[NondimensionalNumber] = Field(default_factory=list)


class PhysicsSpec(StrictBaseModel):
    domains: list[PhysicsDomain]
    regime: str | None = None
    steady: bool | None = None
    coupled: bool = False

