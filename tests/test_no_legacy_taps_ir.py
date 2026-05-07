from __future__ import annotations

import importlib.util
from pathlib import Path

import physicsos.schemas.taps as taps_schema
from physicsos.tools.registry import DEEPAGENTS_SUBAGENT_TOOL_GROUPS, TOOL_REGISTRY


def test_legacy_taps_ir_schema_names_are_removed() -> None:
    removed = {
        "TAPSIntegralTermIR",
        "TAPSFormulationIR",
        "SeparatedGeometryOperator",
        "SeparatedOperatorTermIR",
        "SeparatedOperatorIR",
        "TAPSExecutionPlan",
    }

    assert removed.isdisjoint(set(dir(taps_schema)))


def test_legacy_taps_ir_tools_are_not_registered() -> None:
    removed = {
        "formulate_taps_ir",
        "formulate_taps_ir_structured",
        "validate_taps_ir",
        "propose_separated_operator_structured",
    }
    registered = set(TOOL_REGISTRY)
    taps_tools = {tool.__name__ for tool in DEEPAGENTS_SUBAGENT_TOOL_GROUPS["taps-derivation-agent"]}

    assert removed.isdisjoint(registered)
    assert removed.isdisjoint(taps_tools)


def test_legacy_taps_ir_backends_are_removed() -> None:
    assert importlib.util.find_spec("physicsos.backends.taps_execution") is None
    assert importlib.util.find_spec("physicsos.backends.taps_generic") is None
    assert importlib.util.find_spec("physicsos.backends.taps_thermal") is None
    assert not (Path(__file__).parents[1] / "physicsos" / "backends" / "taps_execution.py").exists()
