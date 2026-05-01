from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from physicsos.backends.surrogate_registry import resolve_path
from physicsos.schemas.problem import PhysicsProblem
from physicsos.schemas.surrogate import SurrogateInputBundle, SurrogateModelSpec, SurrogateOutputBundle, SurrogateTensorSpec


def safe_path_component(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


class SurrogateAdapter(ABC):
    name: str

    @abstractmethod
    def build_input(self, problem: PhysicsProblem, model: SurrogateModelSpec) -> SurrogateInputBundle:
        ...

    @abstractmethod
    def parse_output(self, problem: PhysicsProblem, model: SurrogateModelSpec, raw_output_uri: str) -> SurrogateOutputBundle:
        ...


def _model_dir(model: SurrogateModelSpec) -> Path:
    if model.checkpoint is None:
        return resolve_path("models")
    checkpoint = resolve_path(model.checkpoint.uri)
    return checkpoint if checkpoint.is_dir() else checkpoint.parent


def _read_config(model: SurrogateModelSpec) -> dict[str, Any]:
    config_path = _model_dir(model) / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class TheWellGridAdapter(SurrogateAdapter):
    name = "the_well_grid"

    def build_input(self, problem: PhysicsProblem, model: SurrogateModelSpec) -> SurrogateInputBundle:
        config = _read_config(model)
        expected_shape = [
            1,
            int(config.get("dim_in", 0)),
            *[int(value) for value in config.get("spatial_resolution", [])],
        ]
        metadata = {
            "adapter": self.name,
            "source": "the_well.benchmark.models",
            "dim_in": int(config.get("dim_in", 0)),
            "dim_out": int(config.get("dim_out", 0)),
            "n_spatial_dims": int(config.get("n_spatial_dims", 0)),
            "spatial_resolution": config.get("spatial_resolution", []),
            "note": "Provide a normalized tensor matching The Well dataset preprocessing.",
        }
        tensor = SurrogateTensorSpec(
            name="input_tensor",
            uri=f"scratch/{safe_path_component(problem.id)}/{safe_path_component(model.id)}/input_tensor.npy",
            format="npy",
            shape=expected_shape,
            dtype="float32",
            semantic="The Well normalized model input",
        )
        return SurrogateInputBundle(problem_id=problem.id, tensors=[tensor], metadata=metadata)

    def parse_output(self, problem: PhysicsProblem, model: SurrogateModelSpec, raw_output_uri: str) -> SurrogateOutputBundle:
        config = _read_config(model)
        expected_shape = [
            1,
            int(config.get("dim_out", 0)),
            *[int(value) for value in config.get("spatial_resolution", [])],
        ]
        metadata_path = resolve_path(f"scratch/{safe_path_component(problem.id)}/{safe_path_component(model.id)}/output_metadata.json")
        payload = {
            "adapter": self.name,
            "model_id": model.id,
            "raw_output_uri": raw_output_uri,
            "expected_output_shape": expected_shape,
            "output_fields": model.output_fields,
        }
        _write_json(metadata_path, payload)
        tensor = SurrogateTensorSpec(
            name="output_tensor",
            uri=raw_output_uri,
            format="npy",
            shape=expected_shape,
            dtype="float32",
            semantic="The Well normalized model output",
        )
        return SurrogateOutputBundle(
            problem_id=problem.id,
            model_id=model.id,
            tensors=[tensor],
            metadata_uri=str(metadata_path),
            warnings=["Output remains normalized unless a dataset-specific denormalizer is provided."],
        )


class TheWellUNetAdapter(TheWellGridAdapter):
    name = "the_well_unet"


class TAPSAdapter(SurrogateAdapter):
    name = "taps_apriori"

    def build_input(self, problem: PhysicsProblem, model: SurrogateModelSpec) -> SurrogateInputBundle:
        metadata = {
            "adapter": self.name,
            "required_inputs": ["operator_weak_form", "parameter_axes", "basis_config"],
            "note": "TAPS is an equation-driven a priori surrogate; no public pretrained checkpoint is required.",
        }
        return SurrogateInputBundle(problem_id=problem.id, tensors=[], metadata=metadata)

    def parse_output(self, problem: PhysicsProblem, model: SurrogateModelSpec, raw_output_uri: str) -> SurrogateOutputBundle:
        metadata_path = resolve_path(f"scratch/{safe_path_component(problem.id)}/{safe_path_component(model.id)}/taps_output_metadata.json")
        _write_json(
            metadata_path,
            {
                "adapter": self.name,
                "raw_output_uri": raw_output_uri,
                "note": "TAPS output should contain reduced basis coefficients and reconstruction metadata.",
            },
        )
        return SurrogateOutputBundle(problem_id=problem.id, model_id=model.id, metadata_uri=str(metadata_path))


ADAPTERS: dict[str, SurrogateAdapter] = {
    "the_well_grid": TheWellGridAdapter(),
    "the_well_unet": TheWellUNetAdapter(),
    "taps_apriori": TAPSAdapter(),
}


def get_adapter(name: str | None, model: SurrogateModelSpec) -> SurrogateAdapter:
    if name and name in ADAPTERS:
        return ADAPTERS[name]
    if model.family in {"grid_neural_operator", "tensorized_neural_operator"}:
        return ADAPTERS["the_well_grid"]
    if model.family == "unet_surrogate":
        return ADAPTERS["the_well_unet"]
    if model.family == "taps":
        return ADAPTERS["taps_apriori"]
    raise KeyError(f"No surrogate adapter registered for model {model.id} family {model.family}.")


def list_adapters() -> list[str]:
    return sorted(ADAPTERS)
