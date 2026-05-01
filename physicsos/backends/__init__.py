"""Solver backend interfaces and adapters."""

from physicsos.backends.solver_base import SolverBackend
from physicsos.backends.surrogate_adapters import list_adapters
from physicsos.backends.surrogate_runtime import list_surrogate_models, route_surrogate, run_surrogate_scaffold

__all__ = ["SolverBackend", "list_adapters", "list_surrogate_models", "route_surrogate", "run_surrogate_scaffold"]
