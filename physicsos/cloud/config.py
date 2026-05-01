from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from physicsos.config import config_path as physicsos_config_path
from physicsos.config import load_config, save_config


DEFAULT_RUNNER_URL = "https://foamvm.vercel.app"


@dataclass(frozen=True)
class CloudConfig:
    runner_url: str = DEFAULT_RUNNER_URL
    access_token: str | None = None


def config_path() -> Path:
    override = os.environ.get("PHYSICSOS_CONFIG")
    if override:
        return Path(override).expanduser()
    return physicsos_config_path()


def load_cloud_config(path: Path | None = None) -> CloudConfig:
    target = path or config_path()
    data = load_config(target)
    cloud = data.get("cloud", {})
    return CloudConfig(
        runner_url=str(cloud.get("runner_url") or os.environ.get("PHYSICSOS_RUNNER_URL") or DEFAULT_RUNNER_URL),
        access_token=cloud.get("access_token") or os.environ.get("PHYSICSOS_ACCESS_TOKEN"),
    )


def save_cloud_config(config: CloudConfig, path: Path | None = None) -> Path:
    target = path or config_path()
    data = load_config(target)
    data["cloud"] = {
        **data.get("cloud", {}),
        "runner_url": config.runner_url,
        "access_token": config.access_token or "",
    }
    return save_config(data, target)
