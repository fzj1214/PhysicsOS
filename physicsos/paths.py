from __future__ import annotations

from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

AGENT_WORKSPACE_PREFIX = "/workspace"


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
            relative = local.resolve().relative_to(root)
            return f"{AGENT_WORKSPACE_PREFIX}/{PurePosixPath(*relative.parts).as_posix()}"
        except (OSError, ValueError):
            pass

    return PurePosixPath(*local.parts).as_posix()


def from_agent_path(path: str | Path, *, workspace: str | Path | None = None) -> Path:
    """Convert an agent-facing forward-slash path back to a local Path."""
    value = str(path)
    if value.startswith("file://"):
        return Path(urlparse(value).path)
    if value == AGENT_WORKSPACE_PREFIX:
        return Path(workspace) if workspace is not None else Path.cwd()
    if value.startswith(f"{AGENT_WORKSPACE_PREFIX}/"):
        suffix = value.removeprefix(f"{AGENT_WORKSPACE_PREFIX}/")
        root = Path(workspace) if workspace is not None else Path.cwd()
        return root / Path(*PurePosixPath(suffix).parts)
    local = Path(*PurePosixPath(value).parts)
    if workspace is not None and not local.is_absolute():
        return Path(workspace) / local
    return local
