"""
Base interfaces and data structures for independent verification.

All verification methods must implement the Verifier interface to ensure
consistency and independence from code generation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import Field, model_validator

from physicsos.schemas.common import StrictBaseModel
from physicsos.schemas.problem import PhysicsProblem
from physicsos.schemas.solver import SolverResult


class VerificationStatus(str, Enum):
    """Status of verification checks."""

    UNVERIFIED = "unverified"      # Not yet checked
    VERIFIED = "verified"          # Passed all checks
    VALIDATED = "validated"        # Also compared with experimental data
    FAILED = "failed"              # Failed verification
    UNCERTAIN = "uncertain"        # Verification inconclusive


class ConfidenceScore(str, Enum):
    """System's confidence in its ability to solve this problem."""

    UNKNOWN = "unknown"      # No similar cases in knowledge base
    LOW = "low"              # High failure rate in similar cases
    MEDIUM = "medium"        # Some successful cases, some failures
    HIGH = "high"            # Consistently successful in this regime


class VerificationResult(StrictBaseModel):
    """Result of a single verification check."""

    verifier_name: str                          # Which verifier produced this
    status: VerificationStatus                  # VERIFIED, FAILED, UNCERTAIN

    # Quantitative metrics
    metrics: dict[str, float] = Field(default_factory=dict)

    # Diagnostic information
    message: str                                # Human-readable summary
    details: dict[str, Any] = Field(default_factory=dict)

    # Metadata
    timestamp: datetime = Field(default_factory=lambda: datetime.now())
    compute_time: float = 0.0                   # Seconds

    @model_validator(mode="after")
    def validate_diagnostic(self):
        """Alignment constraint: FAILED status must include actionable diagnostic."""
        if self.status == VerificationStatus.FAILED:
            if not self.message or not self.details:
                raise ValueError("Failed verification must include a diagnostic and details.")
        return self


class AggregateVerificationReport(StrictBaseModel):
    """
    Aggregate report from all verification checks.

    This extends the existing VerificationReport with RSI-specific fields.
    """

    # Core fields
    problem_id: str
    result_id: str
    overall_status: VerificationStatus

    # Individual verification results
    individual_results: dict[str, VerificationResult] = Field(default_factory=dict)

    # Summary statistics
    passed_checks: int = 0
    failed_checks: int = 0
    uncertain_checks: int = 0

    # RSI-specific fields
    confidence: ConfidenceScore = ConfidenceScore.UNKNOWN
    failure_mode: str | None = None             # If failed, what went wrong
    similar_cases: list[str] = Field(default_factory=list)  # Case IDs

    # Metadata
    timestamp: datetime = Field(default_factory=lambda: datetime.now())
    total_compute_time: float = 0.0

    @model_validator(mode="after")
    def validate_status(self):
        """Alignment constraint: overall_status is VERIFIED only if all checks passed."""
        if self.overall_status == VerificationStatus.VERIFIED:
            if self.passed_checks <= 0 or self.failed_checks or self.uncertain_checks or not self.individual_results:
                raise ValueError("VERIFIED requires actual passed checks and no failed or uncertain checks.")
        return self


class Verifier(ABC):
    """
    Base interface for all verification methods.

    All verifiers must be independent of the code generator to ensure trustworthy validation.
    """

    @abstractmethod
    def verify(
        self,
        problem: PhysicsProblem,
        result: SolverResult
    ) -> VerificationResult:
        """
        Verify a solution against the problem specification.

        Args:
            problem: The original problem specification
            result: The numerical solution to verify

        Returns:
            VerificationResult with pass/fail status and diagnostics

        Alignment constraints:
        - Must NOT use the same model that generated the solution
        - Must return quantitative metrics (not just pass/fail)
        - Must be deterministic (same inputs → same output)
        """
        pass

    @abstractmethod
    def required_data(self) -> list[str]:
        """
        Declare what data this verifier needs from the solution.

        Examples: ["field_values", "mesh", "residual_history"]

        Alignment constraint:
        - Solution must provide all required data, or verification fails early
        """
        pass

    @property
    def name(self) -> str:
        """Return the verifier's name."""
        return self.__class__.__name__


class CapabilityAssessment(StrictBaseModel):
    """
    System's self-assessment of its capability to solve a problem.

    Used for epistemic humility: the system must recognize when it doesn't know something.
    """

    confidence: ConfidenceScore
    reasoning: str                              # Why this confidence level
    similar_cases: list[str] = Field(default_factory=list)  # Case IDs
    recommendation: str                         # What the user should expect

    # Quantitative backing
    success_rate: float | None = None           # Of similar cases
    total_similar_cases: int = 0


class FailureMode(str, Enum):
    """Taxonomy of verification failure modes."""

    # Conservation violations
    MASS_NOT_CONSERVED = "mass_not_conserved"
    MOMENTUM_NOT_CONSERVED = "momentum_not_conserved"
    ENERGY_NOT_CONSERVED = "energy_not_conserved"

    # Convergence failures
    DIVERGED = "diverged"
    OSCILLATORY = "oscillatory"
    STALLED = "stalled"

    # Stability issues
    NUMERICAL_INSTABILITY = "numerical_instability"
    CFL_VIOLATION = "cfl_violation"

    # Accuracy issues
    LOW_CONVERGENCE_RATE = "low_convergence_rate"
    HIGH_ERROR = "high_error"
    NONPHYSICAL_SOLUTION = "nonphysical_solution"

    # Code errors
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    DEPENDENCY_ERROR = "dependency_error"

    # Unknown
    UNKNOWN = "unknown"


class FailureRecord(StrictBaseModel):
    """
    Detailed record of a verification failure for learning.
    """

    # Identity
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now())

    # Link to case
    case_uuid: str

    # Problem context
    problem: PhysicsProblem
    generated_code: str

    # Failure classification
    failure_mode: FailureMode
    severity: str = "high"                      # "critical", "high", "medium", "low"

    # Diagnosis
    diagnostic: str                             # Root cause analysis
    verification_logs: dict[str, Any] = Field(default_factory=dict)

    # Learning
    pattern_embedding: list[float] | None = None  # For clustering
    similar_failures: list[str] = Field(default_factory=list)

    # Resolution tracking
    attempted_fixes: list[str] = Field(default_factory=list)
    resolution: str | None = None               # How it was eventually fixed
    resolved_at: datetime | None = None

    def __post_init__(self):
        """Alignment constraint: if resolved, resolution must be set."""
        if self.resolved_at is not None:
            assert self.resolution is not None, \
                "Resolved failures must include resolution description"
