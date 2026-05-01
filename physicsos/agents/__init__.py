"""DeepAgents integration for PhysicsOS."""

from physicsos.agents.main import create_physicsos_agent
from physicsos.agents.openai_compatible import create_openai_compatible_model
from physicsos.agents.runtime import DeepAgentsRuntimeConfig

__all__ = ["DeepAgentsRuntimeConfig", "create_openai_compatible_model", "create_physicsos_agent"]
