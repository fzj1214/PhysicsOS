from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import TypeAdapter

from physicsos.backends.surrogate_catalog import SURROGATE_MODELS
from physicsos.schemas.surrogate import SurrogateModelSpec

_MODEL_LIST_ADAPTER = TypeAdapter(list[SurrogateModelSpec])


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(uri: str) -> Path:
    path = Path(uri)
    if path.is_absolute():
        return path
    return _repo_root() / path


def load_surrogate_registry(path: str | os.PathLike[str]) -> tuple[SurrogateModelSpec, ...]:
    registry_path = resolve_path(str(path))
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    models = _MODEL_LIST_ADAPTER.validate_python(data.get("models", []))
    return tuple(models)


def configured_surrogate_models() -> tuple[SurrogateModelSpec, ...]:
    registry = os.getenv("PHYSICSOS_SURROGATE_REGISTRY", "configs/surrogates.local.json")
    if not registry:
        return SURROGATE_MODELS

    registry_path = resolve_path(registry)
    if not registry_path.exists():
        return SURROGATE_MODELS

    local_models = load_surrogate_registry(registry_path)
    by_id = {model.id: model for model in SURROGATE_MODELS}
    by_id.update({model.id: model for model in local_models})
    return tuple(by_id.values())


def checkpoint_exists(model: SurrogateModelSpec) -> bool:
    if model.checkpoint is None:
        return False
    return resolve_path(model.checkpoint.uri).exists()
