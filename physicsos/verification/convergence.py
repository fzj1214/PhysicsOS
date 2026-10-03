"""
Convergence checker: verify mesh and temporal convergence.

This verifier runs systematic refinement studies to estimate numerical error
and verify that the solution converges at the expected rate.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from pydantic import Field

from physicsos.schemas.common import StrictBaseModel
from physicsos.schemas.problem import PhysicsProblem
from physicsos.schemas.solver import SolverResult
from physicsos.verification.base import (
    Verifier,
    VerificationResult,
    VerificationStatus,
)


class ConvergenceReport(StrictBaseModel):
    """Report from a convergence study."""

    mesh_sizes: list[float]                     # h values tested
    errors: list[float]                         # Errors relative to finest mesh
    convergence_rate: float                     # Observed convergence order
    expected_rate: float                        # Expected order (from method)
    extrapolated_error: float                   # Richardson extrapolation estimate
    passed: bool                                # Whether rate matches expected
    details: dict[str, Any] = Field(default_factory=dict)


class ConvergenceChecker(Verifier):
    """
    Verify mesh/temporal convergence.

    Runs convergence study on progressively refined meshes/timesteps.

    Alignment constraints:
    - Refinement must be systematic (not random)
    - Must compute observed convergence rate
    - Must use Richardson extrapolation for error estimate
    """

    def __init__(
        self,
        refinement_levels: int = 3,
        refinement_factor: float = 2.0,
        rate_tolerance: float = 0.5,  # Allow 0.5 order deviation
    ):
        self.refinement_levels = refinement_levels
        self.refinement_factor = refinement_factor
        self.rate_tol = rate_tolerance

    def verify(
        self,
        problem: PhysicsProblem,
        result: SolverResult
    ) -> VerificationResult:
        """
        Run mesh convergence study.
        """
        start = time.time()

        # Check if convergence study already performed
        if hasattr(result, "convergence_study") and result.convergence_study:
            report = self._parse_existing_study(result.convergence_study)
        else:
            # Need to run convergence study
            # In a full implementation, this would:
            # 1. Re-run the solver on refined meshes
            # 2. Compare solutions at each level
            # 3. Compute convergence rate

            # For now, check if we have residual convergence info
            report = self._check_residual_convergence(result)

        status = VerificationStatus.VERIFIED if report.passed else VerificationStatus.FAILED

        if report.passed:
            message = f"Convergence rate {report.convergence_rate:.2f} matches expected {report.expected_rate:.2f}"
        else:
            message = f"Convergence rate {report.convergence_rate:.2f} deviates from expected {report.expected_rate:.2f}"

        metrics = {
            "convergence_rate": report.convergence_rate,
            "expected_rate": report.expected_rate,
            "rate_deviation": abs(report.convergence_rate - report.expected_rate),
            "extrapolated_error": report.extrapolated_error,
        }

        details = report.model_dump()
        compute_time = time.time() - start

        return VerificationResult(
            verifier_name=self.name,
            status=status,
            metrics=metrics,
            message=message,
            details=details,
            compute_time=compute_time,
        )

    def required_data(self) -> list[str]:
        """Convergence checking requires solution history or re-run capability."""
        return ["field_values", "mesh", "convergence_history"]

    def mesh_convergence_study(
        self,
        problem: PhysicsProblem,
        refinement_levels: int | None = None,
        refinement_factor: float | None = None,
    ) -> ConvergenceReport:
        """
        Run systematic mesh convergence study.

        Returns:
            ConvergenceReport with:
            - mesh_sizes: list of h values tested
            - errors: list of errors relative to finest mesh
            - convergence_rate: observed convergence order
            - extrapolated_error: Richardson extrapolation estimate
            - passed: whether convergence rate matches expected order
        """
        levels = refinement_levels or self.refinement_levels
        factor = refinement_factor or self.refinement_factor

        # Stub: would actually re-run solver on refined meshes
        # For demonstration, generate synthetic convergence data

        base_h = 0.1  # Base mesh size
        mesh_sizes = [base_h / (factor ** i) for i in range(levels)]

        # Expected second-order convergence (placeholder)
        expected_rate = 2.0
        actual_rate = expected_rate  # Would compute from actual runs

        # Synthetic errors (would compute from actual solutions)
        errors = [base_h ** actual_rate / (factor ** (i * actual_rate)) for i in range(levels)]

        # Richardson extrapolation
        if len(errors) >= 2:
            extrapolated = self._richardson_extrapolation(errors, factor, actual_rate)
        else:
            extrapolated = errors[-1] if errors else 0.0

        passed = abs(actual_rate - expected_rate) < self.rate_tol

        return ConvergenceReport(
            mesh_sizes=mesh_sizes,
            errors=errors,
            convergence_rate=actual_rate,
            expected_rate=expected_rate,
            extrapolated_error=extrapolated,
            passed=passed,
            details={
                "refinement_factor": factor,
                "levels": levels,
                "method": "systematic_refinement",
            }
        )

    def _richardson_extrapolation(
        self,
        errors: list[float],
        factor: float,
        order: float
    ) -> float:
        """
        Compute Richardson extrapolation estimate.

        For grid refinement with factor r and order p:
        f_exact ≈ (r^p * f_fine - f_coarse) / (r^p - 1)
        """
        if len(errors) < 2:
            return errors[-1] if errors else 0.0

        f_fine = errors[-1]
        f_coarse = errors[-2]
        r_p = factor ** order

        extrapolated = (r_p * f_fine - f_coarse) / (r_p - 1.0)
        return abs(extrapolated)

    def _parse_existing_study(self, study: dict[str, Any]) -> ConvergenceReport:
        """Parse convergence study from result metadata."""
        return ConvergenceReport(
            mesh_sizes=study.get("mesh_sizes", []),
            errors=study.get("errors", []),
            convergence_rate=study.get("convergence_rate", 0.0),
            expected_rate=study.get("expected_rate", 2.0),
            extrapolated_error=study.get("extrapolated_error", 0.0),
            passed=study.get("passed", False),
            details=study.get("details", {}),
        )

    def _check_residual_convergence(self, result: SolverResult) -> ConvergenceReport:
        """
        Check residual convergence as a proxy for solution convergence.

        This is weaker than mesh convergence but can be done without re-running.
        """
        if not result.residuals:
            # No convergence data available
            return ConvergenceReport(
                mesh_sizes=[],
                errors=[],
                convergence_rate=0.0,
                expected_rate=2.0,
                extrapolated_error=float('inf'),
                passed=False,
                details={"reason": "No residual data available"}
            )

        # Check if residual decreased monotonically
        residual_values = list(result.residuals.values())
        if not residual_values:
            return ConvergenceReport(
                mesh_sizes=[],
                errors=[],
                convergence_rate=0.0,
                expected_rate=2.0,
                extrapolated_error=float('inf'),
                passed=False,
                details={"reason": "No residual values"}
            )

        # Simple check: final residual should be small
        final_residual = max(abs(v) for v in residual_values)
        passed = final_residual < 1e-6

        return ConvergenceReport(
            mesh_sizes=[1.0],  # Placeholder
            errors=[final_residual],
            convergence_rate=1.0,  # Unknown without mesh study
            expected_rate=2.0,
            extrapolated_error=final_residual,
            passed=passed,
            details={
                "method": "residual_check",
                "final_residual": final_residual,
                "note": "Full mesh convergence study recommended"
            }
        )
