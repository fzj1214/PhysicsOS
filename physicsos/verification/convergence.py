"""Legacy convergence gate. Actual refinement studies use CaseRuntime."""
from __future__ import annotations

from typing import Any
from pydantic import Field
from physicsos.schemas.common import StrictBaseModel
from physicsos.schemas.problem import PhysicsProblem
from physicsos.schemas.solver import SolverResult
from physicsos.verification.base import Verifier, VerificationResult, VerificationStatus


class ConvergenceReport(StrictBaseModel):
    mesh_sizes: list[float]
    errors: list[float]
    convergence_rate: float | None
    expected_rate: float
    extrapolated_error: float | None
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)


class ConvergenceChecker(Verifier):
    def __init__(self, refinement_levels: int = 3, refinement_factor: float = 2, rate_tolerance: float = .5):
        self.refinement_levels = refinement_levels
        self.refinement_factor = refinement_factor
        self.rate_tol = rate_tolerance

    def verify(self, problem: PhysicsProblem, result: SolverResult) -> VerificationResult:
        return VerificationResult(
            verifier_name=self.name,
            status=VerificationStatus.FAILED if result.status == "failed" else VerificationStatus.UNCERTAIN,
            # Kept for old callers; explicitly not an observed rate.
            metrics={"convergence_rate": 0.0},
            message="No independently executed refinement study is attached; use CaseRuntime.convergence.",
            details={"convergence_rate_known": False, "solver_status": result.status},
        )

    def required_data(self) -> list[str]:
        return ["field_values", "mesh", "convergence_history"]

    def mesh_convergence_study(self, problem: PhysicsProblem, refinement_levels: int | None = None, refinement_factor: float | None = None) -> ConvergenceReport:
        raise NotImplementedError("Use CaseRuntime.convergence with a prepared domain and a case-local kernel; synthetic errors are not evidence.")
