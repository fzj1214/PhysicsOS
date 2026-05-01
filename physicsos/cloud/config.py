from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


DEFAULT_RUNNER_URL = "https://foamvm.vercel.app"


@dataclass(frozen=True)
class CloudConfig:
    runner_url: str = DEFAULT_RUNNER_URL
    access_token: str | None = None


def config_path() -> Path:
    override = os.environ.get("PHYSICSOS_CONFIG")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".physicsos" / "config.toml"


def load_cloud_config(path: Path | None = None) -> CloudConfig:
    target = path or config_path()
    if not target.exists():
        return CloudConfig(
            runner_url=os.environ.get("PHYSICSOS_RUNNER_URL", DEFAULT_RUNNER_URL),
            access_token=os.environ.get("PHYSICSOS_ACCESS_TOKEN"),
        )

    data = tomllib.loads(target.read_text(encoding="utf-8"))
    cloud = data.get("cloud", {})
    return CloudConfig(
        runner_url=str(cloud.get("runner_url") or os.environ.get("PHYSICSOS_RUNNER_URL") or DEFAULT_RUNNER_URL),
        access_token=cloud.get("access_token") or os.environ.get("PHYSICSOS_ACCESS_TOKEN"),
    )


def save_cloud_config(config: CloudConfig, path: Path | None = None) -> Path:
    target = path or config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    token = config.access_token or ""
    content = (
        "[cloud]\n"
        f'runner_url = "{config.runner_url}"\n'
        f'access_token = "{token}"\n'
    )
    target.write_text(content, encoding="utf-8")
    return target
