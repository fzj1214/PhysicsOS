from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimePaths:
    home: Path
    workspace: Path
    cloud_config: Path
    sessions: Path
    history: Path
    scratch: Path
    case_memory: Path
    knowledge_base: Path


def physicsos_home() -> Path:
    override = os.environ.get("PHYSICSOS_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".physicsos"


def project_root() -> Path:
    if os.environ.get("PHYSICSOS_HOME"):
        return physicsos_home()
    package_parent = Path(__file__).resolve().parents[1]
    if (package_parent / "pyproject.toml").exists() and (package_parent / "physicsos").is_dir():
        return package_parent
    return physicsos_home()


def runtime_paths() -> RuntimePaths:
    workspace = project_root()
    home = physicsos_home()
    return RuntimePaths(
        home=home,
        workspace=workspace,
        cloud_config=home / "config.toml",
        sessions=workspace / "sessions",
        history=workspace / "history.jsonl",
        scratch=workspace / "scratch",
        case_memory=workspace / "data" / "case_memory.jsonl",
        knowledge_base=workspace / "data" / "knowledge" / "physicsos_knowledge.sqlite",
    )


def load_env_file(path: str | Path | None = None) -> None:
    if path is None and (Path.cwd() / ".env").exists():
        env_path = Path.cwd() / ".env"
    else:
        env_path = Path(path) if path is not None else project_root() / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
