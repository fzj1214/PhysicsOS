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


@dataclass(frozen=True)
class DeepSearchSettings:
    """Configuration for the DeepSearch literature-synthesis model.

    DeepSearch often runs on a different provider or endpoint than the main
    agent model, so it carries its own key and base URL. Leaving the key blank
    falls back to the main model credentials.
    """

    enabled: bool = True
    name: str = "gemini-2.5-pro-deepsearch"
    base_url: str = ""
    api_key: str = field(default="", repr=False)
    reuse_main_model: bool = True

    def validate(self, *, main: ModelSettings | None = None) -> None:
        if not self.enabled:
            return
        if not self.name.strip():
            raise ValueError("请填写 DeepSearch 模型名称 / Model ID。")
        if self.base_url.strip():
            url = urlsplit(self.base_url)
            if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
                raise ValueError("DeepSearch API 地址应为 http(s)://主机/v1。")
        if self.reuse_main_model:
            return
        # A key for the main endpoint must not be silently sent to a different one.
        inherits_main_endpoint = not self.base_url.strip() or (main is not None and self.base_url.strip() == main.base_url)
        if not self.api_key.strip() and not (inherits_main_endpoint and main and main.api_key.strip()):
            raise ValueError("DeepSearch 指向其他服务时，请单独填写它的 API Key。")


def main_model_settings() -> ModelSettings:
    """The main agent model, ignoring DeepSearch-specific overrides."""
    return model_settings()


def resolve_deepsearch(model: DeepSearchSettings | None = None, main: ModelSettings | None = None) -> DeepSearchSettings:
    """Merge saved DeepSearch settings with environment overrides and main-model fallback."""
    config = load_config(create=False).get("deepsearch", {})
    saved = model or DeepSearchSettings(
        enabled=bool(config.get("enabled", True)),
        name=str(config.get("name") or "gemini-2.5-pro-deepsearch"),
        base_url=str(config.get("base_url") or ""),
        api_key=str(config.get("api_key") or ""),
        reuse_main_model=bool(config.get("reuse_main_model", True)),
    )
    main = main or model_settings()
    env_key = os.getenv("PHYSICSOS_DEEPSEARCH_API_KEY")
    env_url = os.getenv("PHYSICSOS_DEEPSEARCH_BASE_URL")
    env_model = os.getenv("PHYSICSOS_DEEPSEARCH_MODEL")
    # Only inherit the main endpoint when the user asked to share it, or when no
    # separate endpoint was configured at all.
    shares_main = saved.reuse_main_model or not saved.base_url.strip()
    resolved = DeepSearchSettings(
        enabled=saved.enabled,
        name=env_model or saved.name or main.name,
        base_url=(env_url or saved.base_url or (main.base_url if shares_main else "")).rstrip("/"),
        api_key=env_key or saved.api_key or (main.api_key if shares_main else ""),
        reuse_main_model=saved.reuse_main_model,
    )
    return resolved


def deepsearch_settings() -> DeepSearchSettings:
    return resolve_deepsearch()


def save_deepsearch_settings(deepsearch: DeepSearchSettings, main: ModelSettings | None = None) -> None:
    deepsearch.validate(main=main)
    config = load_config(create=False)
    config.setdefault("deepsearch", {}).update(
        enabled=deepsearch.enabled, name=deepsearch.name, base_url=deepsearch.base_url,
        api_key=deepsearch.api_key, reuse_main_model=deepsearch.reuse_main_model,
    )
    save_config(config)


def apply_deepsearch_environment(deepsearch: DeepSearchSettings) -> None:
    """Publish DeepSearch credentials to this process for the knowledge tools."""
    os.environ["PHYSICSOS_DEEPSEARCH_MODEL"] = deepsearch.name
    if deepsearch.base_url:
        os.environ["PHYSICSOS_DEEPSEARCH_BASE_URL"] = deepsearch.base_url
    if deepsearch.api_key:
        os.environ["PHYSICSOS_DEEPSEARCH_API_KEY"] = deepsearch.api_key
