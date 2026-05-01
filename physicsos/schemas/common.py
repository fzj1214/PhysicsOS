from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictBaseModel(BaseModel):
    """Base model for public PhysicsOS contracts."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, protected_namespaces=())


class ArtifactRef(StrictBaseModel):
    uri: str
    kind: str
    format: str | None = None
    checksum: str | None = None
    description: str | None = None


class Provenance(StrictBaseModel):
    created_by: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: str | None = None
    version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimeStats(StrictBaseModel):
    wall_time_seconds: float | None = None
    cpu_time_seconds: float | None = None
    gpu_time_seconds: float | None = None
    mpi_ranks: int | None = None
    peak_memory_gb: float | None = None


class ComputeBudget(StrictBaseModel):
    max_wall_time_seconds: float | None = None
    max_cpu_cores: int | None = None
    max_gpus: int | None = None
    max_memory_gb: float | None = None
    queue: str | None = None


class UserIntent(StrictBaseModel):
    raw_request: str
    objective: str | None = None
    constraints: list[str] = Field(default_factory=list)


class ParameterSpec(StrictBaseModel):
    name: str
    value: float | int | str | bool
    units: str | None = None
    description: str | None = None


class TargetSpec(StrictBaseModel):
    name: str
    field: str | None = None
    statistic: str | None = None
    units: str | None = None
    objective: Literal["observe", "minimize", "maximize", "match"] = "observe"


class ConstraintSpec(StrictBaseModel):
    name: str
    expression: str
    tolerance: float | None = None


class ExpressionRef(StrictBaseModel):
    expression: str
    language: Literal["text", "python", "sympy", "ufl", "openfoam", "custom"] = "text"


ScalarVectorTensor = float | int | list[float] | list[list[float]]
