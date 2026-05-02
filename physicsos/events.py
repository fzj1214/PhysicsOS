from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from functools import wraps
import inspect
import json
import os
from pathlib import Path
from typing import Any, Literal, TypeVar
from uuid import uuid4

from pydantic import Field

from physicsos.config import runtime_paths
from physicsos.schemas.common import ArtifactRef, StrictBaseModel


PhysicsOSEventType = Literal[
    "workflow.started",
    "agent.started",
    "agent.output",
    "tool.started",
    "tool.output",
    "validation.retry",
    "artifact.created",
    "case_memory.hit",
    "case_memory.event",
    "workflow.completed",
    "workflow.failed",
]


class PhysicsOSEvent(StrictBaseModel):
    run_id: str
    case_id: str | None = None
    event: PhysicsOSEventType
    stage: str | None = None
    status: str = "info"
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    display: dict[str, Any] = Field(default_factory=dict)


class PhysicsOSEventRenderer:
    """Compact renderer shared by CLI surfaces and tests."""

    def render(self, event: PhysicsOSEvent, *, verbose: bool = False) -> str:
        stage = event.stage or event.event
        prefix = f"[{stage}]"
        line = f"{prefix} {event.summary}"
        if event.status not in {"", "info", "complete"}:
            line = f"{line} ({event.status})"
        if event.artifacts:
            artifact_uris = ", ".join(artifact.uri for artifact in event.artifacts[:3])
            line = f"{line} -> {artifact_uris}"
        if verbose and event.payload:
            payload = json.dumps(event.payload, ensure_ascii=False, sort_keys=True)
            line = f"{line}\n{payload}"
        return line

    def render_many(self, events: Iterable[PhysicsOSEvent], *, verbose: bool = False) -> list[str]:
        return [self.render(event, verbose=verbose) for event in events]


def default_event_log_path(run_id: str | None = None) -> Path:
    run = run_id or uuid4().hex
    safe_run = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in run)
    return runtime_paths().sessions / f"events-{safe_run}.jsonl"


def append_physicsos_event(event: PhysicsOSEvent, path: str | Path | None = None) -> Path:
    target = Path(path).expanduser() if path is not None else default_event_log_path(event.run_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(event.model_dump_json() + "\n")
    return target


def read_physicsos_events(path: str | Path) -> list[PhysicsOSEvent]:
    target = Path(path).expanduser()
    if not target.exists():
        return []
    events: list[PhysicsOSEvent] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(PhysicsOSEvent.model_validate_json(line))
        except ValueError:
            continue
    return events


def emit_physicsos_event(event: PhysicsOSEvent, path: str | Path | None = None) -> Path | None:
    target = append_physicsos_event(event, path or os.getenv("PHYSICSOS_EVENT_LOG"))
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
        writer({"physicsos_event": event.model_dump(mode="json")})
    except Exception:
        pass
    return target


ToolT = TypeVar("ToolT", bound=Callable[..., Any])


def wrap_tool_for_events(tool: ToolT, *, run_id: str | None = None) -> ToolT:
    if getattr(tool, "_physicsos_event_wrapped", False):
        return tool

    @wraps(tool)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        effective_run_id = run_id or os.getenv("PHYSICSOS_RUN_ID") or uuid4().hex
        stage = getattr(tool, "__name__", "tool")
        emit_physicsos_event(
            PhysicsOSEvent(
                run_id=effective_run_id,
                event="tool.started",
                stage=stage,
                summary=f"{stage} started.",
            )
        )
        try:
            result = tool(*args, **kwargs)
        except Exception as exc:
            emit_physicsos_event(
                PhysicsOSEvent(
                    run_id=effective_run_id,
                    event="workflow.failed",
                    stage=stage,
                    status="failed",
                    summary=f"{stage} failed: {exc}",
                    payload={"error": str(exc)},
                )
            )
            raise
        payload: dict[str, Any] = {}
        if hasattr(result, "model_dump"):
            payload = {"output": result.model_dump(mode="json")}
        emit_physicsos_event(
            PhysicsOSEvent(
                run_id=effective_run_id,
                event="tool.output",
                stage=stage,
                status="complete",
                summary=f"{stage} completed.",
                payload=payload,
            )
        )
        return result

    wrapped.__signature__ = inspect.signature(tool)  # type: ignore[attr-defined]
    for attr in ("input_model", "output_model", "side_effects", "requires_approval"):
        if hasattr(tool, attr):
            setattr(wrapped, attr, getattr(tool, attr))
    wrapped._physicsos_event_wrapped = True  # type: ignore[attr-defined]
    return wrapped  # type: ignore[return-value]


def wrap_tools_for_events(tools: Iterable[ToolT], *, run_id: str | None = None) -> list[ToolT]:
    return [wrap_tool_for_events(tool, run_id=run_id) for tool in tools]


def collect_physicsos_events(value: object) -> list[PhysicsOSEvent]:
    if isinstance(value, PhysicsOSEvent):
        return [value]
    if isinstance(value, dict):
        raw_events = value.get("physicsos_events")
        if raw_events is None and isinstance(value.get("workflow"), dict):
            raw_events = value["workflow"].get("events")
        if raw_events is None and "physicsos_event" in value:
            raw_events = [value["physicsos_event"]]
        if isinstance(raw_events, list):
            events: list[PhysicsOSEvent] = []
            for item in raw_events:
                try:
                    events.append(PhysicsOSEvent.model_validate(item))
                except ValueError:
                    continue
            return events
    return []
