from __future__ import annotations

from physicsos.cli import LOCAL_COMMANDS
from physicsos.tools.registry import DEEPAGENTS_MAIN_BRIDGE_TOOLS, TOOL_REGISTRY


def _tool_names(tools: list[object]) -> set[str]:
    return {getattr(tool, "__name__", "") for tool in tools}


def test_deepagents_main_bridge_does_not_expose_legacy_typed_workflow() -> None:
    assert "run_typed_physicsos_workflow" not in _tool_names(DEEPAGENTS_MAIN_BRIDGE_TOOLS)
    assert "run_typed_physicsos_workflow" not in TOOL_REGISTRY
    assert "run_taps_backend" not in TOOL_REGISTRY


def test_cli_does_not_expose_old_workflow_command() -> None:
    assert "workflow" not in LOCAL_COMMANDS
