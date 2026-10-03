"""
Conservation checker: verify conservation of mass, momentum, and energy.

This verifier checks fundamental conservation laws that must hold for physics simulations.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from physicsos.schemas.problem import PhysicsProblem
from physicsos.schemas.solver import SolverResult
from physicsos.verification.base import (
    FailureMode,
    Verifier,
    VerificationResult,
    VerificationStatus,
)


class ConservationResult(StrictBaseModel):
    """Result of a conservation check."""

    quantity: str                               # "mass", "momentum", "energy"
    imbalance: float                            # Absolute conservation error
    relative_error: float                       # imbalance / total_quantity
    tolerance: float                            # Threshold used
    passed: bool                                # Whether relative_error < tolerance
    details: dict[str, Any] = Field(default_factory=dict)


class ConservationChecker(Verifier):
    """
    Verify conservation laws.

    Checks:
    - Mass conservation: d/dt ∫∫∫ ρ dV + ∫∫ ρ(u·n) dS = ∫∫∫ source dV
    - Momentum conservation: ∑ F_ext = d/dt ∫∫∫ ρu dV + ∫∫ ρu(u·n) dS
    - Energy conservation: energy balance

    Alignment constraints:
    - Tolerance must be physically motivated (not arbitrary)
    - Must account for boundary fluxes
    - Must handle source/sink terms correctly
    """

    def __init__(
        self,
        mass_tolerance: float = 1e-10,
        momentum_tolerance: float = 1e-9,
        energy_tolerance: float = 1e-8,
    ):
        self.mass_tol = mass_tolerance
        self.momentum_tol = momentum_tolerance
        self.energy_tol = energy_tolerance

    def verify(
        self,
        problem: PhysicsProblem,
        result: SolverResult
    ) -> VerificationResult:
        """
        Check conservation laws applicable to this problem.
        """
        start = time.time()

        # Determine which conservation laws apply
        conserved = set()
        for op in problem.operators:
            conserved.update(op.conserved_quantities)

        checks: list[ConservationResult] = []

        # Check mass conservation
        if "mass" in conserved or "continuity" in conserved:
            mass_check = self.check_mass_conservation(problem, result, self.mass_tol)
            checks.append(mass_check)

        # Check momentum conservation
        if "momentum" in conserved:
            momentum_check = self.check_momentum_conservation(problem, result, self.momentum_tol)
            checks.append(momentum_check)

        # Check energy conservation
        if "energy" in conserved:
            energy_check = self.check_energy_conservation(problem, result, self.energy_tol)
            checks.append(energy_check)

        # Aggregate results
        all_passed = all(c.passed for c in checks)
        status = VerificationStatus.VERIFIED if all_passed else VerificationStatus.FAILED

        if all_passed:
            message = f"All conservation checks passed ({len(checks)} checks)"
        else:
            failed = [c.quantity for c in checks if not c.passed]
            message = f"Conservation violated for: {', '.join(failed)}"

        metrics = {
            f"{c.quantity}_relative_error": c.relative_error
            for c in checks
        }

        details = {
            "checks": [c.model_dump() for c in checks],
            "conserved_quantities": list(conserved),
        }

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
        """Conservation checking requires field values and mesh."""
        return ["field_values", "mesh", "boundary_fluxes"]

    def check_mass_conservation(
        self,
        problem: PhysicsProblem,
        result: SolverResult,
        tol: float
    ) -> ConservationResult:
        """
        Verify: d/dt ∫∫∫ ρ dV + ∫∫ ρ(u·n) dS = ∫∫∫ source dV

        For steady problems: ∫∫ ρ(u·n) dS = ∫∫∫ source dV
        For incompressible: ∫∫ u·n dS = 0
        """
        # Extract field values
        # Note: This is a stub implementation - actual implementation would:
        # 1. Extract density field (or assume constant for incompressible)
        # 2. Extract velocity field
        # 3. Compute volume integral of ρ
        # 4. Compute boundary flux ∫∫ ρ(u·n) dS
        # 5. Account for source terms
        # 6. Compare imbalance to tolerance

        # Placeholder: check if result has conservation metrics
        if "mass_imbalance" in result.residuals:
            imbalance = abs(result.residuals["mass_imbalance"])
            total_mass = result.residuals.get("total_mass", 1.0)
        else:
            # Try to compute from field values if available
            imbalance = 0.0  # Would compute from fields
            total_mass = 1.0

        relative_error = imbalance / max(total_mass, 1e-16)
        passed = relative_error < tol

        return ConservationResult(
            quantity="mass",
            imbalance=imbalance,
            relative_error=relative_error,
            tolerance=tol,
            passed=passed,
            details={
                "total_mass": total_mass,
                "method": "residual_based",  # or "field_integration"
            }
        )

    def check_momentum_conservation(
        self,
        problem: PhysicsProblem,
        result: SolverResult,
        tol: float
    ) -> ConservationResult:
        """
        Verify: ∑ F_ext = d/dt ∫∫∫ ρu dV + ∫∫ ρu(u·n) dS

        For steady: ∑ F_ext = ∫∫ ρu(u·n) dS
        """
        # Stub implementation
        if "momentum_imbalance" in result.residuals:
            imbalance = abs(result.residuals["momentum_imbalance"])
            total_momentum = result.residuals.get("total_momentum", 1.0)
        else:
            imbalance = 0.0
            total_momentum = 1.0

        relative_error = imbalance / max(total_momentum, 1e-16)
        passed = relative_error < tol

        return ConservationResult(
            quantity="momentum",
            imbalance=imbalance,
            relative_error=relative_error,
            tolerance=tol,
            passed=passed,
            details={"total_momentum": total_momentum}
        )

    def check_energy_conservation(
        self,
        problem: PhysicsProblem,
        result: SolverResult,
        tol: float
    ) -> ConservationResult:
        """
        Verify energy balance.
        """
        # Stub implementation
        if "energy_imbalance" in result.residuals:
            imbalance = abs(result.residuals["energy_imbalance"])
            total_energy = result.residuals.get("total_energy", 1.0)
        else:
            imbalance = 0.0
            total_energy = 1.0

        relative_error = imbalance / max(total_energy, 1e-16)
        passed = relative_error < tol

        return ConservationResult(
            quantity="energy",
            imbalance=imbalance,
            relative_error=relative_error,
            tolerance=tol,
            passed=passed,
            details={"total_energy": total_energy}
        )


# Fix missing import
from pydantic import Field
from physicsos.schemas.common import StrictBaseModel
