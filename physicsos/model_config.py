"""Shared model configuration for the launcher, settings UI, and agent tools."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from urllib.parse import urlsplit

from physicsos.config import default_config, load_config, save_config


@dataclass(frozen=True)
class ModelSettings:
    name: str
    base_url: str
    api_key: str = field(repr=False)
    use_responses_api: bool = False

    @property
    def spec(self) -> str:
        return f"openai:{self.name}"

    @property
    def params(self) -> dict[str, object]:
        # Keep credentials out of CLI arguments and persisted conversation context.
        return {"base_url": self.base_url, "use_responses_api": self.use_responses_api}

    def validate(self, *, require_model: bool = True) -> None:
        if require_model and not self.name.strip():
            raise ValueError("请填写模型名称 / Model ID is required.")
        url = urlsplit(self.base_url)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("API 地址应为 http(s)://主机/v1，例如 https://api.openai.com/v1。")
        if not self.api_key.strip():
            raise ValueError("请填写 API Key；本地无鉴权服务可填写 local。")


def model_settings() -> ModelSettings:
    config = load_config(create=False)
    model = config.get("model", {})
    defaults = default_config()["model"]
    responses = os.getenv("PHYSICSOS_OPENAI_USE_RESPONSES_API", os.getenv("PHYSICSOS_STRUCTURED_USE_RESPONSES_API"))
    return ModelSettings(
        name=str(os.getenv("PHYSICSOS_OPENAI_MODEL") or model.get("name") or defaults["name"]),
        base_url=str(os.getenv("PHYSICSOS_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL") or model.get("base_url") or defaults["base_url"]).rstrip("/"),
        api_key=str(os.getenv("PHYSICSOS_OPENAI_API_KEY") or os.getenv("DEEPAGENTS_CLI_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or model.get("api_key") or ""),
        use_responses_api=(responses.lower() in {"1", "true", "yes", "on"}) if responses is not None else bool(model.get("use_responses_api", False)),
    )


def uses_openai_model(spec: str | None) -> bool:
    if not spec:
        return True
    if ":" in spec:
        return spec.startswith("openai:")
    from deepagents_cli.config import detect_provider

    return (detect_provider(spec) or "openai") == "openai"


def save_model_settings(model: ModelSettings) -> None:
    model.validate()
    config = load_config(create=False)
    config["model"].update(
        provider="openai", name=model.name, base_url=model.base_url,
        api_key=model.api_key, use_responses_api=model.use_responses_api,
    )
    save_config(config)


def apply_model_environment(model: ModelSettings) -> None:
    """Apply explicit settings to this process and subsequently spawned servers."""
    for name in ("PHYSICSOS_OPENAI_API_KEY", "DEEPAGENTS_CLI_OPENAI_API_KEY", "OPENAI_API_KEY"):
        os.environ[name] = model.api_key
    for name in ("PHYSICSOS_OPENAI_BASE_URL", "OPENAI_BASE_URL"):
        os.environ[name] = model.base_url
    os.environ["PHYSICSOS_OPENAI_MODEL"] = model.name
    os.environ["PHYSICSOS_OPENAI_USE_RESPONSES_API"] = str(model.use_responses_api).lower()
