from __future__ import annotations

import os

from physicsos.config import load_config, load_env_file


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def create_openai_compatible_model():
    """Create a LangChain ChatOpenAI model for OpenAI-compatible providers.

    Required environment:
      PHYSICSOS_OPENAI_API_KEY

    Optional environment:
      PHYSICSOS_OPENAI_BASE_URL
      PHYSICSOS_OPENAI_MODEL
      PHYSICSOS_OPENAI_USE_RESPONSES_API
    """
    load_env_file()
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError("Install optional agents dependencies with `pip install -e .[agents]`.") from exc

    config = load_config()
    model_config = config.get("model", {})

    api_key = os.getenv("PHYSICSOS_OPENAI_API_KEY") or model_config.get("api_key")
    if not api_key:
        raise RuntimeError("Set PHYSICSOS_OPENAI_API_KEY or model.api_key in ~/.physicsos/config.json.")

    base_url = os.getenv("PHYSICSOS_OPENAI_BASE_URL") or model_config.get("base_url") or "https://api.tu-zi.com/v1"
    model = os.getenv("PHYSICSOS_OPENAI_MODEL") or model_config.get("name") or "gpt-5.4"
    use_responses_api = _env_bool("PHYSICSOS_OPENAI_USE_RESPONSES_API", bool(model_config.get("use_responses_api", False)))

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        use_responses_api=use_responses_api,
    )
