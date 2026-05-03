from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from pathlib import Path


@dataclass(frozen=True)
class RuntimePaths:
    home: Path
    workspace: Path
    config_json: Path
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
    workspace_override = os.environ.get("PHYSICSOS_WORKSPACE")
    if workspace_override:
        return Path(workspace_override).expanduser()
    if os.environ.get("PHYSICSOS_HOME"):
        return physicsos_home()
    package_parent = Path(__file__).resolve().parents[1]
    if (package_parent / "pyproject.toml").exists() and (package_parent / "physicsos").is_dir():
        return package_parent
    return physicsos_home()


def runtime_paths() -> RuntimePaths:
    workspace = project_root()
    home = physicsos_home()
    config_json = home / "config.json"
    return RuntimePaths(
        home=home,
        workspace=workspace,
        config_json=config_json,
        cloud_config=config_json,
        sessions=workspace / "sessions",
        history=workspace / "history.jsonl",
        scratch=workspace / "scratch",
        case_memory=workspace / "data" / "case_memory.jsonl",
        knowledge_base=workspace / "data" / "knowledge" / "physicsos_knowledge.sqlite",
    )


def default_config() -> dict[str, Any]:
    return {
        "model": {
            "provider": "openai",
            "name": "gpt-5.4",
            "api_key": "",
            "base_url": "https://api.tu-zi.com/v1",
            "use_responses_api": False,
        },
        "cloud": {
            "runner_url": "https://foamvm.vercel.app",
            "access_token": "",
        },
        "storage": {
            "home": str(physicsos_home()),
            "scratch": "scratch",
            "case_memory": "data/case_memory.jsonl",
            "knowledge_base": "data/knowledge/physicsos_knowledge.sqlite",
        },
        "ui": {
            "launcher": "deepagents-cli",
            "agent": "physicsos",
            "banner": "physicsos",
        },
        "core_agents": {
            "mode": "llm",
            "max_structured_attempts": 3,
            "prompt_version": "v1",
        },
    }


def _merge_defaults(value: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    for key, item in value.items():
        if isinstance(item, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_defaults(item, merged[key])
        else:
            merged[key] = item
    return merged


def config_path() -> Path:
    override = os.environ.get("PHYSICSOS_CONFIG")
    if override:
        return Path(override).expanduser()
    return runtime_paths().config_json


def _loads_config_json(text: str, target: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        # Some hand-written Windows configs contain unescaped local path
        # backslashes such as "\.physicsos". JSON only allows a narrow set of
        # escape sequences, so repair those path-like escapes before failing.
        import re

        repaired = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)
        try:
            loaded = json.loads(repaired)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid PhysicsOS config JSON: {target}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"PhysicsOS config must be a JSON object: {target}")
    return loaded


def load_config(path: str | Path | None = None, *, create: bool = True) -> dict[str, Any]:
    target = Path(path).expanduser() if path is not None else config_path()
    if not target.exists():
        config = default_config()
        if create:
            save_config(config, target)
        return config
    loaded = _loads_config_json(target.read_text(encoding="utf-8"), target)
    return _merge_defaults(loaded, default_config())


def save_config(config: dict[str, Any], path: str | Path | None = None) -> Path:
    target = Path(path).expanduser() if path is not None else config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_merge_defaults(config, default_config()), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


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
