from __future__ import annotations

from pydantic import Field

from physicsos.schemas.boundary import BoundaryConditionSpec, InitialConditionSpec
from physicsos.schemas.common import ConstraintSpec, ParameterSpec, Provenance, StrictBaseModel, TargetSpec, UserIntent
from physicsos.schemas.geometry import GeometrySpec
from physicsos.schemas.materials import MaterialSpec
from physicsos.schemas.mesh import MeshSpec
from physicsos.schemas.operators import FieldSpec, OperatorSpec, PhysicsDomain
from physicsos.schemas.verification import VerificationPolicy


class PhysicsProblem(StrictBaseModel):
    id: str
    user_intent: UserIntent
    domain: PhysicsDomain
    geometry: GeometrySpec
    mesh: MeshSpec | None = None
    fields: list[FieldSpec]
    operators: list[OperatorSpec]
    materials: list[MaterialSpec]
    boundary_conditions: list[BoundaryConditionSpec]
    initial_conditions: list[InitialConditionSpec] = Field(default_factory=list)
    parameters: list[ParameterSpec] = Field(default_factory=list)
    targets: list[TargetSpec]
    constraints: list[ConstraintSpec] = Field(default_factory=list)
    verification_policy: VerificationPolicy = Field(default_factory=VerificationPolicy)
    provenance: Provenance

