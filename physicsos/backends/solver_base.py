from __future__ import annotations

from typing import Literal, Protocol

from physicsos.schemas.common import ArtifactRef
from physicsos.schemas.problem import PhysicsProblem
from physicsos.schemas.solver import CostEstimate, PreparedSolverCase, SolverResult, SupportScore


class SolverBackend(Protocol):
    name: str
    family: Literal[
        "surrogate",
        "neural_operator",
        "fem",
        "fvm",
        "md",
        "dft",
        "rom",
        "legacy",
        "chemistry",
        "particle_transport",
        "custom",
    ]

    def supports(self, problem: PhysicsProblem) -> SupportScore:
        ...

    def estimate_cost(self, problem: PhysicsProblem) -> CostEstimate:
        ...

    def prepare(self, problem: PhysicsProblem) -> PreparedSolverCase:
        ...

    def solve(self, prepared: PreparedSolverCase) -> SolverResult:
        ...

    def collect_artifacts(self, result: SolverResult) -> list[ArtifactRef]:
        ...

