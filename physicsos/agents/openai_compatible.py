from __future__ import annotations

from physicsos.config import load_env_file
from physicsos.model_config import model_settings


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

    model = model_settings()
    if not model.api_key:
        raise RuntimeError("Run `physicsos config` to configure your model and API Key.")
    return ChatOpenAI(model=model.name, api_key=model.api_key, **model.params)
