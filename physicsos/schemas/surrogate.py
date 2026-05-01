from __future__ import annotations

from typing import Literal

from pydantic import Field

from physicsos.schemas.common import ArtifactRef, StrictBaseModel


SurrogateFamily = Literal[
    "grid_neural_operator",
    "tensorized_neural_operator",
    "geometry_informed_neural_operator",
    "mesh_graph_operator",
    "manifold_operator",
    "pinn_corrector",
    "taps",
    "rom",
    "unet_surrogate",
    "foundation_surrogate",
    "custom",
]


class SurrogateModelSpec(StrictBaseModel):
    id: str
    name: str
    family: SurrogateFamily
    domains: list[str]
    operator_families: list[str] = Field(default_factory=list)
    geometry_encodings: list[str] = Field(default_factory=list)
    mesh_kinds: list[str] = Field(default_factory=list)
    steady: bool | None = None
    supports_transient: bool = False
    checkpoint: ArtifactRef | None = None
    runner: Literal["scaffold", "torchscript", "torch_state_dict", "safetensors", "external"] = "scaffold"
    input_adapter: str | None = None
    output_adapter: str | None = None
    training_dataset: str | None = None
    expected_fields: list[str] = Field(default_factory=list)
    output_fields: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class SurrogateSupportScore(StrictBaseModel):
    model_id: str
    score: float
    supported: bool
    reasons: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class SurrogateDecision(StrictBaseModel):
    selected_model_id: str
    candidate_models: list[SurrogateSupportScore]
    mode: Literal["fast_solver", "warm_start", "corrector", "reject"]
    reason: str
    required_geometry_encodings: list[str] = Field(default_factory=list)
    expected_uncertainty: float | None = None


class SurrogateTensorSpec(StrictBaseModel):
    name: str
    uri: str
    format: Literal["npy", "npz", "pt", "pth", "safetensors", "json"]
    shape: list[int] | None = None
    dtype: str | None = None
    semantic: str | None = None


class SurrogateInputBundle(StrictBaseModel):
    problem_id: str
    tensors: list[SurrogateTensorSpec] = Field(default_factory=list)
    metadata: dict[str, str | int | float | bool | list[int] | list[str]] = Field(default_factory=dict)


class SurrogateOutputBundle(StrictBaseModel):
    problem_id: str
    model_id: str
    tensors: list[SurrogateTensorSpec] = Field(default_factory=list)
    metadata_uri: str
    warnings: list[str] = Field(default_factory=list)
