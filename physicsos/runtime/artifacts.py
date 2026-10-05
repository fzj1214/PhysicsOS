from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

from physicsos.paths import resolve_workspace_path, to_agent_path
from physicsos.schemas.common import ArtifactRef


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def artifact(path: Path, kind: str, workspace: Path) -> ArtifactRef:
    return ArtifactRef(uri=to_agent_path(path, workspace=workspace), kind=kind, format=path.suffix.lstrip("."), checksum=digest(path))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=path.name + ".", suffix=".tmp", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def checked_path(reference: ArtifactRef, workspace: Path) -> Path:
    path = resolve_workspace_path(reference.uri, workspace=workspace)
    if not path.is_file():
        raise ValueError(f"Required artifact is missing: {reference.uri}")
    if reference.checksum is None or digest(path) != reference.checksum.removeprefix("sha256:"):
        raise ValueError(f"Artifact version changed or has no checksum: {reference.uri}")
    return path
