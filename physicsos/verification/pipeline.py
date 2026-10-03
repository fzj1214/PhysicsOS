"""
Verification pipeline: orchestrate multiple verifiers and aggregate results.

This module coordinates independent verification checks and produces an aggregate report.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import Field

from physicsos.schemas.common import StrictBaseModel
from physicsos.schemas.problem import PhysicsProblem
from physicsos.schemas.solver import SolverResult
from physicsos.verification.base import (
    AggregateVerificationReport,
    CapabilityAssessment,
    ConfidenceScore,
    FailureMode,
    FailureRecord,
    Verifier,
    VerificationResult,
    VerificationStatus,
)


class VerificationPipeline:
    """
    Orchestrate multiple independent verifiers.

    Usage:
        pipeline = VerificationPipeline()
        pipeline.register(ConservationChecker())
        pipeline.register(ConvergenceChecker())

        report = pipeline.verify(problem, result)
    """

    def __init__(self):
        self.verifiers: list[Verifier] = []

    def register(self, verifier: Verifier):
        """Register a verifier in the pipeline."""
        self.verifiers.append(verifier)

    def verify(
        self,
        problem: PhysicsProblem,
        result: SolverResult,
        confidence: ConfidenceScore = ConfidenceScore.UNKNOWN,
    ) -> AggregateVerificationReport:
        """
        Run all registered verifiers and aggregate results.

        Args:
            problem: The physics problem specification
            result: The solver result to verify
            confidence: Pre-assessed confidence (from capability assessment)

        Returns:
            AggregateVerificationReport with overall status and individual results
        """
        start = time.time()

        individual_results: dict[str, VerificationResult] = {}
        passed = 0
        failed = 0
        uncertain = 0

        for verifier in self.verifiers:
            try:
                result_obj = verifier.verify(problem, result)
                individual_results[verifier.name] = result_obj

                if result_obj.status == VerificationStatus.VERIFIED:
                    passed += 1
                elif result_obj.status == VerificationStatus.FAILED:
                    failed += 1
                else:
                    uncertain += 1
            except Exception as e:
                # Verifier raised an exception - treat as uncertain
                individual_results[verifier.name] = VerificationResult(
                    verifier_name=verifier.name,
                    status=VerificationStatus.UNCERTAIN,
                    message=f"Verifier raised exception: {str(e)}",
                    details={"exception": str(e)},
                    metrics={},
                )
                uncertain += 1

        # Determine overall status
        if failed > 0:
            overall_status = VerificationStatus.FAILED
            failure_mode = self._determine_failure_mode(individual_results)
        elif uncertain > 0:
            overall_status = VerificationStatus.UNCERTAIN
            failure_mode = None
        else:
            overall_status = VerificationStatus.VERIFIED
            failure_mode = None

        total_compute_time = time.time() - start

        return AggregateVerificationReport(
            problem_id=problem.id,
            result_id=result.id,
            overall_status=overall_status,
            individual_results=individual_results,
            passed_checks=passed,
            failed_checks=failed,
            uncertain_checks=uncertain,
            confidence=confidence,
            failure_mode=failure_mode,
            total_compute_time=total_compute_time,
        )

    def _determine_failure_mode(
        self,
        results: dict[str, VerificationResult]
    ) -> str:
        """
        Determine the primary failure mode from failed checks.
        """
        failed_verifiers = [
            name for name, result in results.items()
            if result.status == VerificationStatus.FAILED
        ]

        # Map verifier names to failure modes
        if "ConservationChecker" in failed_verifiers:
            return FailureMode.MASS_NOT_CONSERVED.value  # Could be more specific
        elif "ConvergenceChecker" in failed_verifiers:
            return FailureMode.LOW_CONVERGENCE_RATE.value
        elif any("Stability" in name for name in failed_verifiers):
            return FailureMode.NUMERICAL_INSTABILITY.value
        else:
            return FailureMode.UNKNOWN.value


class SelfDiagnostic:
    """
    Assess the system's own capability for a given problem.

    Used for epistemic humility: the system must recognize when it doesn't know something.
    """

    def __init__(self, knowledge_base: Any = None):
        """
        Args:
            knowledge_base: Reference to the case memory / model catalog
        """
        self.kb = knowledge_base

    def assess_capability(
        self,
        problem: PhysicsProblem
    ) -> CapabilityAssessment:
        """
        Evaluate the system's confidence in solving this problem.

        Returns:
            CapabilityAssessment with:
            - confidence: ConfidenceScore (UNKNOWN/LOW/MEDIUM/HIGH)
            - reasoning: Why this confidence level
            - similar_cases: Links to knowledge base
            - recommendation: What the user should expect

        Alignment constraints:
        - If no similar verified cases exist → UNKNOWN confidence
        - If similar cases have high failure rate → LOW confidence + warning
        - Must explicitly state "I don't know" rather than hallucinating confidence
        """
        if self.kb is None:
            # No knowledge base available
            return CapabilityAssessment(
                confidence=ConfidenceScore.UNKNOWN,
                reasoning="No historical case data available",
                similar_cases=[],
                recommendation="Proceeding with exploratory verification. Success not guaranteed.",
                success_rate=None,
                total_similar_cases=0,
            )

        # Query knowledge base for similar cases
        similar_cases = self._find_similar_cases(problem)

        if not similar_cases:
            return CapabilityAssessment(
                confidence=ConfidenceScore.UNKNOWN,
                reasoning="No similar problems found in knowledge base",
                similar_cases=[],
                recommendation=(
                    "I haven't encountered a problem like this before. "
                    "I'll generate a solution and run extensive verification, "
                    "but cannot guarantee success. Consider this exploratory."
                ),
                success_rate=None,
                total_similar_cases=0,
            )

        # Compute success rate
        verified_count = sum(
            1 for case in similar_cases
            if case.get("verification_status") == "verified"
        )
        total = len(similar_cases)
        success_rate = verified_count / total

        # Determine confidence
        if success_rate > 0.9:
            confidence = ConfidenceScore.HIGH
            recommendation = (
                f"This problem is similar to {total} previously solved cases "
                f"({success_rate:.0%} success rate). Proceeding with high confidence."
            )
        elif success_rate >= 0.6:
            confidence = ConfidenceScore.MEDIUM
            recommendation = (
                f"This problem shares features with {total} previous cases "
                f"({success_rate:.0%} success rate). Will generate solution and verify carefully."
            )
        elif success_rate >= 0.3:
            confidence = ConfidenceScore.LOW
            recommendation = (
                f"This problem type has succeeded in only {verified_count} of {total} previous attempts. "
                "Generating multiple candidate solutions for verification."
            )
        else:
            confidence = ConfidenceScore.LOW
            recommendation = (
                f"Warning: Similar problems have a low success rate ({success_rate:.0%}). "
                "Will attempt solution with multiple strategies and thorough verification."
            )

        case_ids = [case.get("id", "unknown") for case in similar_cases[:5]]

        return CapabilityAssessment(
            confidence=confidence,
            reasoning=f"Based on {total} similar cases with {success_rate:.0%} success rate",
            similar_cases=case_ids,
            recommendation=recommendation,
            success_rate=success_rate,
            total_similar_cases=total,
        )

    def _find_similar_cases(self, problem: PhysicsProblem) -> list[dict[str, Any]]:
        """
        Query knowledge base for similar problems.

        Placeholder: actual implementation would use the search_case_memory tool.
        """
        # Stub: would call search_case_memory
        return []


class FailureAnalyzer:
    """
    Diagnose verification failures to enable learning.
    """

    def diagnose_failure(
        self,
        problem: PhysicsProblem,
        result: SolverResult,
        verification: AggregateVerificationReport,
    ) -> FailureRecord:
        """
        Analyze a verification failure to determine root cause.

        Returns:
            FailureRecord with diagnostic information for learning
        """
        # Extract failure mode from verification report
        failure_mode_str = verification.failure_mode or "unknown"
        try:
            failure_mode = FailureMode(failure_mode_str)
        except ValueError:
            failure_mode = FailureMode.UNKNOWN

        # Generate diagnostic message
        failed_checks = [
            name for name, result_obj in verification.individual_results.items()
            if result_obj.status == VerificationStatus.FAILED
        ]

        diagnostic = f"Verification failed: {', '.join(failed_checks)}. "

        # Add specific diagnostics from failed checks
        for name in failed_checks:
            check_result = verification.individual_results[name]
            diagnostic += f"{name}: {check_result.message}. "

        # Create failure record
        record = FailureRecord(
            case_uuid=problem.id,
            problem=problem,
            generated_code=result.script or "",
            failure_mode=failure_mode,
            severity="high" if verification.confidence == ConfidenceScore.HIGH else "medium",
            diagnostic=diagnostic,
            verification_logs={
                "individual_results": {
                    name: result_obj.model_dump()
                    for name, result_obj in verification.individual_results.items()
                },
            },
        )

        return record
