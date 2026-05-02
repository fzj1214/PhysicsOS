from __future__ import annotations

from pathlib import Path, PurePosixPath
from urllib.parse import urlparse


def is_remote_uri(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https", "s3", "gs", "blob"}


def to_agent_path(path: str | Path, *, workspace: str | Path | None = None) -> str:
    """Return a stable forward-slash path for prompts and tool outputs."""
    value = str(path)
    if is_remote_uri(value) or value.startswith("file://"):
        return value

    local = Path(value)
    if workspace is not None:
        root = Path(workspace).resolve()
        try:
            local = local.resolve().relative_to(root)
        except (OSError, ValueError):
            pass

    return PurePosixPath(*local.parts).as_posix()


def from_agent_path(path: str | Path, *, workspace: str | Path | None = None) -> Path:
    """Convert an agent-facing forward-slash path back to a local Path."""
    value = str(path)
    if value.startswith("file://"):
        return Path(urlparse(value).path)
    local = Path(*PurePosixPath(value).parts)
    if workspace is not None and not local.is_absolute():
        return Path(workspace) / local
    return local
