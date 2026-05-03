from __future__ import annotations

import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Callable
from typing import Any, Generic, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

from physicsos.events import PhysicsOSEvent, emit_physicsos_event
from physicsos.paths import to_agent_path
from physicsos.schemas.common import ArtifactRef, StrictBaseModel
from physicsos.config import load_config, runtime_paths


OutputT = TypeVar("OutputT", bound=BaseModel)


class CoreAgentLLMConfig(StrictBaseModel):
    mode: str = "llm"
    model: str | None = None
    max_structured_attempts: int = 3
    prompt_version: str = "v1"


class StructuredAgentAttempt(StrictBaseModel):
    attempt: int
    raw_response: str
    parsed: dict[str, Any] | None = None
    validation_errors: list[str] = Field(default_factory=list)


class StructuredAgentResult(StrictBaseModel, Generic[OutputT]):
    agent_name: str
    status: str
    output: OutputT | None = None
    attempts: list[StructuredAgentAttempt] = Field(default_factory=list)
    error: str | None = None


StructuredLLMClient = Callable[[dict[str, Any]], str | dict[str, Any] | BaseModel]


class StructuredAgentEventContext(StrictBaseModel):
    run_id: str | None = None
    case_id: str | None = None


_STRUCTURED_AGENT_CONTEXT: ContextVar[StructuredAgentEventContext | None] = ContextVar("physicsos_structured_agent_context", default=None)
_STRUCTURED_AGENT_EVENTS: ContextVar[list[PhysicsOSEvent] | None] = ContextVar("physicsos_structured_agent_events", default=None)


@contextmanager
def structured_agent_event_context(
    *,
    run_id: str | None = None,
    case_id: str | None = None,
    events: list[PhysicsOSEvent] | None = None,
):
    context_token = _STRUCTURED_AGENT_CONTEXT.set(StructuredAgentEventContext(run_id=run_id, case_id=case_id))
    events_token = _STRUCTURED_AGENT_EVENTS.set(events)
    try:
        yield
    finally:
        _STRUCTURED_AGENT_EVENTS.reset(events_token)
        _STRUCTURED_AGENT_CONTEXT.reset(context_token)


def load_core_agent_config() -> CoreAgentLLMConfig:
    config = load_config(create=False)
    model_config = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    core_config = config.get("core_agents", {}) if isinstance(config.get("core_agents"), dict) else {}
    return CoreAgentLLMConfig(
        mode=os.environ.get("PHYSICSOS_CORE_AGENTS_MODE", str(core_config.get("mode", "llm"))),
        model=os.environ.get("PHYSICSOS_CORE_AGENT_MODEL", str(model_config.get("name") or "")) or None,
        max_structured_attempts=int(os.environ.get("PHYSICSOS_CORE_AGENT_MAX_ATTEMPTS", core_config.get("max_structured_attempts", 3))),
        prompt_version=str(core_config.get("prompt_version", "v1")),
    )


def _redacted_config_value(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    if len(text) <= 8:
        return "***"
    return f"{text[:4]}...{text[-4:]}"


def create_openai_structured_client() -> StructuredLLMClient:
    """Create an OpenAI-compatible structured JSON client from PhysicsOS config."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the openai package to use real structured core agents.") from exc

    config = load_config()
    model_config = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    api_key = os.environ.get("PHYSICSOS_OPENAI_API_KEY") or str(model_config.get("api_key") or "")
    if not api_key:
        raise RuntimeError("Set PHYSICSOS_OPENAI_API_KEY or model.api_key in ~/.physicsos/config.json.")
    base_url = os.environ.get("PHYSICSOS_OPENAI_BASE_URL") or str(model_config.get("base_url") or "https://api.tu-zi.com/v1")
    default_model = os.environ.get("PHYSICSOS_CORE_AGENT_MODEL") or os.environ.get("PHYSICSOS_OPENAI_MODEL") or str(model_config.get("name") or "gpt-5.4")
    client = OpenAI(api_key=api_key, base_url=base_url)

    def invoke(request: dict[str, Any]) -> str:
        model = str(request.get("model") or default_model)
        payload = {
            "agent_name": request.get("agent_name"),
            "input_schema": request.get("input_schema"),
            "output_schema": request.get("output_schema"),
            "input": request.get("input"),
            "validation_feedback": request.get("validation_feedback") or [],
        }
        messages = [
            {
                "role": "system",
                "content": (
                    str(request.get("system_prompt") or "")
                    + "\nReturn exactly one JSON object matching the output_schema. Do not wrap it in Markdown."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""

    invoke.provider = "openai-compatible"  # type: ignore[attr-defined]
    invoke.model = default_model  # type: ignore[attr-defined]
    invoke.base_url = base_url  # type: ignore[attr-defined]
    invoke.api_key = _redacted_config_value(api_key)  # type: ignore[attr-defined]
    return invoke


def _safe_path_component(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)[:120] or "unknown"


def _structured_attempt_artifact(
    *,
    call_id: str,
    request: dict[str, Any],
    attempt: StructuredAgentAttempt,
    status: str,
) -> ArtifactRef:
    context = _STRUCTURED_AGENT_CONTEXT.get()
    run_id = (context.run_id if context is not None else None) or os.getenv("PHYSICSOS_RUN_ID") or f"structured:{call_id}"
    agent_name = str(request.get("agent_name") or "structured-agent")
    output_dir = runtime_paths().scratch / "structured_agents" / _safe_path_component(run_id) / _safe_path_component(agent_name) / call_id
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"attempt-{attempt.attempt}.json"
    payload = {
        "call_id": call_id,
        "agent_name": agent_name,
        "status": status,
        "attempt": attempt.attempt,
        "model": request.get("model"),
        "prompt_version": request.get("prompt_version"),
        "system_prompt": request.get("system_prompt"),
        "input_schema": request.get("input_schema"),
        "output_schema": request.get("output_schema"),
        "input": request.get("input"),
        "validation_feedback": request.get("validation_feedback") or [],
        "raw_response": attempt.raw_response,
        "parsed": attempt.parsed,
        "validation_errors": attempt.validation_errors,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return ArtifactRef(
        uri=to_agent_path(path, workspace=runtime_paths().workspace),
        kind="structured_agent_attempt",
        format="json",
        description=f"{agent_name} structured attempt {attempt.attempt}",
    )


def _emit_structured_attempt_event(
    *,
    request: dict[str, Any],
    attempt: StructuredAgentAttempt,
    max_attempts: int,
    status: str,
    event_type: str,
    artifact: ArtifactRef,
) -> None:
    context = _STRUCTURED_AGENT_CONTEXT.get()
    run_id = (context.run_id if context is not None else None) or os.getenv("PHYSICSOS_RUN_ID") or f"structured:{uuid4().hex}"
    case_id = (context.case_id if context is not None else None) or os.getenv("PHYSICSOS_CASE_ID")
    agent_name = str(request.get("agent_name") or "structured-agent")
    parsed = attempt.parsed if isinstance(attempt.parsed, dict) else {}
    validation_errors = attempt.validation_errors
    summary = f"{agent_name} structured attempt {attempt.attempt}/{max_attempts}: {status}"
    if validation_errors:
        summary += f"; {validation_errors[0]}"
    event = PhysicsOSEvent(
        run_id=run_id,
        case_id=case_id,
        event=event_type,  # type: ignore[arg-type]
        stage=agent_name,
        status=status,
        summary=summary,
        payload={
            "agent_name": agent_name,
            "attempt": attempt.attempt,
            "max_attempts": max_attempts,
            "model": request.get("model"),
            "prompt_version": request.get("prompt_version"),
            "output_schema": (request.get("output_schema") or {}).get("title") if isinstance(request.get("output_schema"), dict) else None,
            "parsed_keys": sorted(parsed.keys()),
            "validation_errors": validation_errors,
            "artifact": artifact.model_dump(mode="json"),
        },
        artifacts=[artifact],
        display={"collapsed": True, "raw_response_in_artifact": True},
    )
    collector = _STRUCTURED_AGENT_EVENTS.get()
    if collector is not None:
        collector.append(event)
    emit_physicsos_event(event)


def _coerce_raw_response(raw: str | dict[str, Any] | BaseModel) -> tuple[str, Any]:
    if isinstance(raw, BaseModel):
        payload = raw.model_dump(mode="json")
        return json.dumps(payload, ensure_ascii=False), payload
    if isinstance(raw, dict):
        return json.dumps(raw, ensure_ascii=False), raw
    text = str(raw)
    try:
        return text, json.loads(text)
    except json.JSONDecodeError:
        return text, None


def call_structured_agent(
    *,
    agent_name: str,
    input_model: BaseModel,
    output_model: type[OutputT],
    system_prompt: str,
    client: StructuredLLMClient,
    config: CoreAgentLLMConfig | None = None,
) -> StructuredAgentResult[OutputT]:
    """Call an LLM-like client and only return validated Pydantic output."""
    cfg = config or CoreAgentLLMConfig()
    attempts: list[StructuredAgentAttempt] = []
    validation_feedback: list[str] = []
    max_attempts = max(1, cfg.max_structured_attempts)
    call_id = uuid4().hex
    last_request: dict[str, Any] | None = None
    for attempt_index in range(1, max_attempts + 1):
        request = {
            "agent_name": agent_name,
            "model": cfg.model,
            "prompt_version": cfg.prompt_version,
            "system_prompt": system_prompt,
            "input_schema": input_model.__class__.model_json_schema(),
            "output_schema": output_model.model_json_schema(),
            "input": input_model.model_dump(mode="json"),
            "validation_feedback": validation_feedback,
        }
        last_request = request
        try:
            raw = client(request)
        except Exception as exc:
            error = f"Structured LLM client call failed: {exc.__class__.__name__}: {exc}"
            attempt = StructuredAgentAttempt(attempt=attempt_index, raw_response="", validation_errors=[error])
            attempts.append(attempt)
            status = "retrying" if attempt_index < max_attempts else "retry_exhausted"
            artifact = _structured_attempt_artifact(call_id=call_id, request=request, attempt=attempt, status=status)
            _emit_structured_attempt_event(
                request=request,
                attempt=attempt,
                max_attempts=max_attempts,
                status=status,
                event_type="validation.retry",
                artifact=artifact,
            )
            validation_feedback = [error]
            continue
        raw_text, payload = _coerce_raw_response(raw)
        if payload is None:
            error = "LLM response was not valid JSON."
            attempt = StructuredAgentAttempt(attempt=attempt_index, raw_response=raw_text, validation_errors=[error])
            attempts.append(attempt)
            status = "retrying" if attempt_index < max_attempts else "retry_exhausted"
            artifact = _structured_attempt_artifact(call_id=call_id, request=request, attempt=attempt, status=status)
            _emit_structured_attempt_event(
                request=request,
                attempt=attempt,
                max_attempts=max_attempts,
                status=status,
                event_type="validation.retry",
                artifact=artifact,
            )
            validation_feedback = [error]
            continue
        try:
            output = output_model.model_validate(payload)
        except ValidationError as exc:
            errors = [str(err) for err in exc.errors()]
            attempt = StructuredAgentAttempt(
                attempt=attempt_index,
                raw_response=raw_text,
                parsed=payload if isinstance(payload, dict) else {"value": payload},
                validation_errors=errors,
            )
            attempts.append(attempt)
            status = "retrying" if attempt_index < max_attempts else "retry_exhausted"
            artifact = _structured_attempt_artifact(call_id=call_id, request=request, attempt=attempt, status=status)
            _emit_structured_attempt_event(
                request=request,
                attempt=attempt,
                max_attempts=max_attempts,
                status=status,
                event_type="validation.retry",
                artifact=artifact,
            )
            validation_feedback = errors
            continue
        attempt = StructuredAgentAttempt(
            attempt=attempt_index,
            raw_response=raw_text,
            parsed=output.model_dump(mode="json"),
        )
        attempts.append(attempt)
        artifact = _structured_attempt_artifact(call_id=call_id, request=request, attempt=attempt, status="accepted")
        _emit_structured_attempt_event(
            request=request,
            attempt=attempt,
            max_attempts=max_attempts,
            status="accepted",
            event_type="agent.output",
            artifact=artifact,
        )
        return StructuredAgentResult(agent_name=agent_name, status="accepted", output=output, attempts=attempts)

    return StructuredAgentResult(
        agent_name=agent_name,
        status="retry_exhausted",
        attempts=attempts,
        error="Structured agent output failed validation.",
    )
