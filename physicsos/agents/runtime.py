from __future__ import annotations

from typing import Any

from pydantic import Field

from physicsos.schemas.common import StrictBaseModel


def _instantiate_backend(backend_type: type, runtime: object | None = None, **kwargs: Any) -> Any:
    """Instantiate DeepAgents backends across 0.4/0.5 constructor changes."""
    try:
        return backend_type(**kwargs)
    except TypeError:
        if runtime is None:
            raise
        return backend_type(runtime, **kwargs)


def _build_filesystem_backend(runtime: object | None = None) -> Any:
    try:
        from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
    except ImportError as exc:
        raise RuntimeError("deepagents is required for enable_filesystem_backend=True") from exc

    return CompositeBackend(
        default=_instantiate_backend(StateBackend, runtime),
        routes={
            "/cases/": _instantiate_backend(StoreBackend, runtime),
            "/datasets/": _instantiate_backend(StoreBackend, runtime),
            "/models/": _instantiate_backend(StoreBackend, runtime),
            "/reports/": _instantiate_backend(StoreBackend, runtime),
            "/scratch/": _instantiate_backend(StateBackend, runtime),
        },
    )


class DeepAgentsRuntimeConfig(StrictBaseModel):
    model: str = "openai:gpt-5.4"
    enable_filesystem_backend: bool = True
    enable_memory_store: bool = True
    enable_checkpointer: bool = True
    interrupt_full_solver: bool = True
    interrupt_hpc: bool = True
    interrupt_deletes: bool = True
    name: str = "physicsos-main"
    extra: dict[str, Any] = Field(default_factory=dict)


def build_runtime_kwargs(config: DeepAgentsRuntimeConfig) -> dict[str, Any]:
    """Build DeepAgents runtime kwargs with optional LangGraph persistence.

    Imports happen lazily so the core package remains usable without installing
    the `agents` optional dependency.
    """
    kwargs: dict[str, Any] = {"name": config.name}

    if config.enable_memory_store:
        try:
            from langgraph.store.memory import InMemoryStore
        except ImportError as exc:
            raise RuntimeError("langgraph is required for enable_memory_store=True") from exc
        kwargs["store"] = InMemoryStore()

    if config.enable_checkpointer:
        try:
            from langgraph.checkpoint.memory import MemorySaver
        except ImportError as exc:
            raise RuntimeError("langgraph is required for enable_checkpointer=True") from exc
        kwargs["checkpointer"] = MemorySaver()

    if config.enable_filesystem_backend:
        kwargs["backend"] = _build_filesystem_backend

    interrupt_on: dict[str, Any] = {}
    if config.interrupt_full_solver:
        interrupt_on["run_full_solver"] = {"allowed_decisions": ["approve", "edit", "reject"]}
    if config.interrupt_hpc:
        interrupt_on["submit_hpc_job"] = {"allowed_decisions": ["approve", "edit", "reject"]}
    if config.interrupt_deletes:
        interrupt_on["delete_case_artifacts"] = {"allowed_decisions": ["approve", "reject"]}
    if interrupt_on:
        kwargs["interrupt_on"] = interrupt_on

    kwargs.update(config.extra)
    return kwargs
