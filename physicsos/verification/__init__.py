"""
Independent Verification Framework for PhysicsOS.

This module provides verification methods that are independent of the code generator,
ensuring trustworthy validation of physics simulations.
"""

from physicsos.verification.base import (
    Verifier,
    VerificationResult,
    VerificationStatus,
    ConfidenceScore,
)
from physicsos.verification.conservation import ConservationChecker
from physicsos.verification.convergence import ConvergenceChecker
from physicsos.verification.pipeline import VerificationPipeline

__all__ = [
    "Verifier",
    "VerificationResult",
    "VerificationStatus",
    "ConfidenceScore",
    "ConservationChecker",
    "ConvergenceChecker",
    "VerificationPipeline",
]
