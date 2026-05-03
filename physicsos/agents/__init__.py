"""DeepAgents integration for PhysicsOS."""

from physicsos.agents.openai_compatible import create_openai_compatible_model
from physicsos.agents.runtime import DeepAgentsRuntimeConfig
from physicsos.agents.structured import create_openai_structured_client

__all__ = [
    "DeepAgentsRuntimeConfig",
    "create_openai_compatible_model",
    "create_openai_structured_client",
    "create_physicsos_agent",
]


def create_physicsos_agent(*args, **kwargs):
    from physicsos.agents.main import create_physicsos_agent as _create_physicsos_agent

    return _create_physicsos_agent(*args, **kwargs)
