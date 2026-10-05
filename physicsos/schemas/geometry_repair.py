from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from physicsos.schemas.common import ArtifactRef, StrictBaseModel
from physicsos.schemas.geometry import GeometryQualityReport, GeometrySpec


RepairStatus = Literal["repaired", "needs_review", "backend_unavailable", "failed"]


class GeometryRepairOptions(StrictBaseModel):
    repair_policy: Literal["conservative", "balanced", "aggressive"] = "conservative"
    executor: Literal["auto", "python", "docker"] = "auto"
    python_executable: str | None = None
    docker_image: str | None = None
    case_id: str | None = None
    timeout_seconds: int = Field(default=600, ge=1, le=86400)
    simplification_ratio: float | None = Field(default=None, gt=0, le=1)
    target_faces: int | None = Field(default=None, ge=4)
    max_relative_surface_distance: float = Field(default=0.01, gt=0, allow_inf_nan=False)
    max_aspect_ratio_p95: float = Field(default=25.0, gt=1, allow_inf_nan=False)
    max_skewness: float = Field(default=0.95, gt=0, lt=1)
    validation_samples: int = Field(default=2048, ge=128, le=100000)
    surface_element_size: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def exclusive_face_target(self):
        if self.target_faces is not None and self.simplification_ratio is not None:
            raise ValueError("Specify target_faces or simplification_ratio, not both.")
        return self


class RepairGeometryInput(GeometryRepairOptions):
    geometry: GeometrySpec


class GeometryRepairReport(StrictBaseModel):
    schema_version: Literal["physicsos.geometry_repair.v1"] = "physicsos.geometry_repair.v1"
    job_id: str
    status: RepairStatus
    input_sha256: str
    quality: GeometryQualityReport
    parameters: dict[str, Any] = Field(default_factory=dict)
    before: dict[str, Any] = Field(default_factory=dict)
    after: dict[str, Any] = Field(default_factory=dict)
    deviation: dict[str, Any] = Field(default_factory=dict)
    output_surface: str | None = None
    output_mesh: str | None = None
    output_sha256: str | None = None
    backend_version: str | None = None
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)


class RepairGeometryOutput(StrictBaseModel):
    geometry: GeometrySpec
    status: RepairStatus
    repair_report: ArtifactRef
    execution_log: ArtifactRef
    repaired_surface: ArtifactRef | None = None
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    invalidated_encodings: list[ArtifactRef] = Field(default_factory=list)
    requires_boundary_relabel: bool = False
    changes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
