from __future__ import annotations

import asyncio

import pytest

from physicsos.agents.prompts import PHYSICSOS_SYSTEM_PROMPT
from physicsos.agents.subagents import SUBAGENTS
from physicsos.cli import (
    _is_retryable_agent_error,
    _patch_deepagents_physicsos_agent_config,
    _retry_async_agent_call,
)


def test_create_cli_agent_patch_injects_physicsos_prompt_and_runtime_subagents(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("deepagents_cli")

    import deepagents_cli.agent as cli_agent

    captured_cli: dict[str, object] = {}
    captured_deep: dict[str, object] = {}

    def fake_create_cli_agent(*args: object, **kwargs: object):
        captured_cli["args"] = args
        captured_cli["kwargs"] = kwargs
        return object(), object()

    def fake_create_deep_agent(*args: object, **kwargs: object):
        captured_deep["args"] = args
        captured_deep["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(cli_agent, "create_cli_agent", fake_create_cli_agent)
    monkeypatch.setattr(cli_agent, "create_deep_agent", fake_create_deep_agent)

    _patch_deepagents_physicsos_agent_config()
    cli_agent.create_cli_agent("model", "physicsos", system_prompt="base prompt")
    cli_agent.create_deep_agent(model="model", subagents=[{"name": "remote"}])

    cli_kwargs = captured_cli["kwargs"]
    assert isinstance(cli_kwargs, dict)
    assert PHYSICSOS_SYSTEM_PROMPT in str(cli_kwargs["system_prompt"])
    assert "base prompt" in str(cli_kwargs["system_prompt"])

    deep_kwargs = captured_deep["kwargs"]
    assert isinstance(deep_kwargs, dict)
    injected = deep_kwargs["subagents"]
    assert isinstance(injected, list)
    assert injected[0] == {"name": "remote"}
    assert injected[1:] == SUBAGENTS


def test_retry_async_agent_call_retries_retryable_errors_five_times() -> None:
    attempts = 0

    class APIConnectionError(Exception):
        pass

    async def fail() -> None:
        nonlocal attempts
        attempts += 1
        raise APIConnectionError("An internal error occurred")

    with pytest.raises(APIConnectionError):
        asyncio.run(_retry_async_agent_call(fail, base_delay_seconds=0))

    assert attempts == 5


def test_retry_async_agent_call_does_not_retry_non_retryable_errors() -> None:
    attempts = 0

    async def fail() -> None:
        nonlocal attempts
        attempts += 1
        raise ValueError("bad request")

    with pytest.raises(ValueError):
        asyncio.run(_retry_async_agent_call(fail, base_delay_seconds=0))

    assert attempts == 1
    assert not _is_retryable_agent_error(ValueError("bad request"))
