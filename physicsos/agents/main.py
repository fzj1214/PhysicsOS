from __future__ import annotations

from typing import Any

from physicsos.agents.prompts import PHYSICSOS_SYSTEM_PROMPT
from physicsos.agents.runtime import DeepAgentsRuntimeConfig, build_runtime_kwargs
from physicsos.agents.subagents import SUBAGENTS
from physicsos.tools.registry import DEEPAGENTS_MAIN_BRIDGE_TOOLS


def create_physicsos_agent(model: Any = "openai:gpt-5.4", runtime: DeepAgentsRuntimeConfig | None = None, **kwargs: Any) -> Any:
    """Create the DeepAgents main-agent interface for PhysicsOS.

    This is not the LangGraph/PhysicsOS workflow constructor. It creates the
    flexible DeepAgents main agent with a small catalog bridge and the
    paper-style TAPS subagents. Legacy typed workflow entry points are not
    exposed on the default DeepAgents tool surface.
    """
    try:
        from deepagents import create_deep_agent
    except ImportError as exc:
        raise RuntimeError("Install PhysicsOS with the 'agents' extra to create the DeepAgents runtime.") from exc

    runtime_config = runtime or DeepAgentsRuntimeConfig(model=model if isinstance(model, str) else "custom")
    runtime_kwargs = build_runtime_kwargs(runtime_config)
    options: dict[str, Any] = {
        "model": model if not isinstance(model, str) else runtime_config.model,
        "tools": DEEPAGENTS_MAIN_BRIDGE_TOOLS,
        "subagents": SUBAGENTS,
        "system_prompt": PHYSICSOS_SYSTEM_PROMPT,
    }
    options.update(runtime_kwargs)
    options.update(kwargs)
    return create_deep_agent(**options)


def assert_deepagents_runtime_available() -> None:
    try:
        import deepagents  # noqa: F401
        import langgraph  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Install optional dependencies with `pip install -e .[agents]`.") from exc
