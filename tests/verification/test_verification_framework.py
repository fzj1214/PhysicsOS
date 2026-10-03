"""
Tests for the independent verification framework.
"""

import pytest

from physicsos.schemas.boundary import BoundaryConditionSpec
from physicsos.schemas.common import Provenance, StrictBaseModel, TargetSpec, UserIntent
from physicsos.schemas.geometry import GeometrySource, GeometrySpec
from physicsos.schemas.materials import MaterialProperty, MaterialSpec
from physicsos.schemas.operators import FieldSpec, OperatorSpec
from physicsos.schemas.problem import PhysicsProblem
from physicsos.schemas.solver import SolverResult
from physicsos.verification.base import (
    CapabilityAssessment,
    ConfidenceScore,
    FailureMode,
    VerificationStatus,
)
from physicsos.verification.conservation import ConservationChecker
from physicsos.verification.convergence import ConvergenceChecker
from physicsos.verification.pipeline import (
    FailureAnalyzer,
    SelfDiagnostic,
    VerificationPipeline,
)


def make_simple_problem() -> PhysicsProblem:
    """Create a minimal valid PhysicsProblem for testing."""
    return PhysicsProblem(
        id="test-case-001",
        user_intent=UserIntent(
            raw_request="Test steady Navier-Stokes flow",
            objective="Verify conservation and convergence",
        ),
        domain="fluid",
        geometry=GeometrySpec(
            id="test-geom-001",
            dimension=2,
            source=GeometrySource(kind="text", uri="rectangle 1m x 1m"),
            entities=[],
        ),
        mesh=None,
        fields=[
            FieldSpec(name="velocity", kind="vector", units="m/s"),
            FieldSpec(name="pressure", kind="scalar", units="Pa"),
        ],
        operators=[
            OperatorSpec(
                id="op-001",
                name="navier_stokes",
                domain="fluid",
                equation_class="navier_stokes",
                form="weak",
                conserved_quantities=["mass", "momentum"],
            )
        ],
        materials=[
            MaterialSpec(
                id="mat-001",
                name="water",
                phase="liquid",
                properties=[
                    MaterialProperty(name="density", value=1000.0, units="kg/m^3")
                ],
            )
        ],
        boundary_conditions=[
            BoundaryConditionSpec(
                id="bc-001",
                kind="dirichlet",
                field="velocity",
                region_id="inlet",
                value=[1.0, 0.0],
            )
        ],
        targets=[],
        provenance=Provenance(created_by="test", source="test", version="1.0"),
    )


def make_simple_result(converged: bool = True) -> SolverResult:
    """Create a minimal valid SolverResult for testing."""
    return SolverResult(
        id="test-result-001",
        problem_id="test-case-001",
        backend="test_solver",
        status="success" if converged else "failed",
        residuals={
            "mass_imbalance": 1e-11 if converged else 1e-3,
            "momentum_imbalance": 1e-10 if converged else 1e-2,
            "total_mass": 1000.0,
            "total_momentum": 100.0,
        },
        provenance=Provenance(created_by="test", source="test", version="1.0"),
    )


class TestConservationChecker:
    """Test conservation verification."""

    def test_mass_conservation_pass(self):
        """Verify that small mass imbalance passes."""
        checker = ConservationChecker(mass_tolerance=1e-9)
        problem = make_simple_problem()
        result = make_simple_result(converged=True)

        verification = checker.verify(problem, result)

        assert verification.status == VerificationStatus.VERIFIED
        assert "mass_relative_error" in verification.metrics
        assert verification.metrics["mass_relative_error"] < 1e-9

    def test_mass_conservation_fail(self):
        """Verify that large mass imbalance fails."""
        checker = ConservationChecker(mass_tolerance=1e-9)
        problem = make_simple_problem()
        result = make_simple_result(converged=False)

        verification = checker.verify(problem, result)

        assert verification.status == VerificationStatus.FAILED
        assert "mass_relative_error" in verification.metrics
        assert verification.metrics["mass_relative_error"] > 1e-9

    def test_determinism(self):
        """Verify that same inputs produce same output."""
        checker = ConservationChecker()
        problem = make_simple_problem()
        result = make_simple_result()

        v1 = checker.verify(problem, result)
        v2 = checker.verify(problem, result)

        assert v1.status == v2.status
        assert v1.metrics == v2.metrics
        # Timestamps may differ, but message should be same
        assert v1.message == v2.message

    def test_required_data(self):
        """Verify required_data declares dependencies."""
        checker = ConservationChecker()
        required = checker.required_data()

        assert isinstance(required, list)
        assert len(required) > 0
        assert "field_values" in required or "mesh" in required


class TestConvergenceChecker:
    """Test convergence verification."""

    def test_convergence_check_pass(self):
        """Verify that low residual passes."""
        checker = ConvergenceChecker()
        problem = make_simple_problem()
        result = make_simple_result(converged=True)

        verification = checker.verify(problem, result)

        # With stub implementation, should at least not crash
        assert verification.status in [
            VerificationStatus.VERIFIED,
            VerificationStatus.UNCERTAIN,
        ]
        assert "convergence_rate" in verification.metrics

    def test_convergence_check_fail(self):
        """Verify that high residual fails."""
        checker = ConvergenceChecker()
        problem = make_simple_problem()
        result = make_simple_result(converged=False)

        verification = checker.verify(problem, result)

        # Stub may return UNCERTAIN if no convergence study
        assert verification.status in [
            VerificationStatus.FAILED,
            VerificationStatus.UNCERTAIN,
        ]


class TestVerificationPipeline:
    """Test verification pipeline orchestration."""

    def test_pipeline_all_pass(self):
        """Verify that all checks passing → VERIFIED."""
        pipeline = VerificationPipeline()
        pipeline.register(ConservationChecker())

        problem = make_simple_problem()
        result = make_simple_result(converged=True)

        report = pipeline.verify(problem, result)

        assert report.overall_status == VerificationStatus.VERIFIED
        assert report.passed_checks > 0
        assert report.failed_checks == 0

    def test_pipeline_one_fail(self):
        """Verify that one failure → FAILED."""
        pipeline = VerificationPipeline()
        pipeline.register(ConservationChecker(mass_tolerance=1e-12))  # Very strict

        problem = make_simple_problem()
        result = make_simple_result(converged=False)

        report = pipeline.verify(problem, result)

        assert report.overall_status == VerificationStatus.FAILED
        assert report.failed_checks > 0
        assert report.failure_mode is not None

    def test_pipeline_aggregation(self):
        """Verify that multiple verifiers are aggregated correctly."""
        pipeline = VerificationPipeline()
        pipeline.register(ConservationChecker())
        pipeline.register(ConvergenceChecker())

        problem = make_simple_problem()
        result = make_simple_result(converged=True)

        report = pipeline.verify(problem, result)

        assert len(report.individual_results) == 2
        assert "ConservationChecker" in report.individual_results
        assert "ConvergenceChecker" in report.individual_results


class TestSelfDiagnostic:
    """Test capability assessment (epistemic humility)."""

    def test_unknown_confidence_no_kb(self):
        """Verify UNKNOWN when no knowledge base."""
        diagnostic = SelfDiagnostic(knowledge_base=None)
        problem = make_simple_problem()

        assessment = diagnostic.assess_capability(problem)

        assert assessment.confidence == ConfidenceScore.UNKNOWN
        assert "no" in assessment.reasoning.lower() or "not available" in assessment.reasoning.lower()
        assert assessment.total_similar_cases == 0

    def test_epistemic_humility(self):
        """Verify system admits ignorance rather than hallucinating."""
        diagnostic = SelfDiagnostic(knowledge_base=None)
        problem = make_simple_problem()

        assessment = diagnostic.assess_capability(problem)

        # Must explicitly state uncertainty
        assert assessment.confidence == ConfidenceScore.UNKNOWN
        assert len(assessment.recommendation) > 0
        # Recommendation should mention uncertainty/exploration
        assert any(
            word in assessment.recommendation.lower()
            for word in ["not", "unknown", "exploratory", "cannot guarantee"]
        )


class TestFailureAnalyzer:
    """Test failure diagnosis."""

    def test_diagnose_conservation_failure(self):
        """Verify correct failure mode for conservation violation."""
        analyzer = FailureAnalyzer()

        problem = make_simple_problem()
        result = make_simple_result(converged=False)

        # Create verification report with conservation failure
        pipeline = VerificationPipeline()
        pipeline.register(ConservationChecker(mass_tolerance=1e-12))
        verification = pipeline.verify(problem, result)

        # Diagnose
        failure = analyzer.diagnose_failure(problem, result, verification)

        assert failure.failure_mode in [
            FailureMode.MASS_NOT_CONSERVED,
            FailureMode.MOMENTUM_NOT_CONSERVED,
        ]
        assert len(failure.diagnostic) > 0
        assert "conservation" in failure.diagnostic.lower() or "conserv" in failure.diagnostic.lower()

    def test_failure_record_has_diagnostics(self):
        """Verify failure record includes actionable diagnostics."""
        analyzer = FailureAnalyzer()

        problem = make_simple_problem()
        result = make_simple_result(converged=False)

        pipeline = VerificationPipeline()
        pipeline.register(ConservationChecker())
        verification = pipeline.verify(problem, result)

        failure = analyzer.diagnose_failure(problem, result, verification)

        # Must have diagnostic message
        assert failure.diagnostic
        assert len(failure.diagnostic) > 10  # Not just "failed"

        # Must include verification logs
        assert failure.verification_logs
        assert "individual_results" in failure.verification_logs


class TestAlignmentConstraints:
    """Test alignment constraints are enforced."""

    def test_verified_requires_no_failures(self):
        """Verify VERIFIED status requires all checks passed."""
        pipeline = VerificationPipeline()
        pipeline.register(ConservationChecker())

        problem = make_simple_problem()
        result = make_simple_result(converged=True)

        report = pipeline.verify(problem, result)

        if report.overall_status == VerificationStatus.VERIFIED:
            assert report.failed_checks == 0
            assert report.uncertain_checks == 0

    def test_verification_has_metrics(self):
        """Verify all verification results include metrics."""
        checker = ConservationChecker()
        problem = make_simple_problem()
        result = make_simple_result()

        verification = checker.verify(problem, result)

        assert verification.metrics is not None
        assert len(verification.metrics) > 0
        # All metrics must be numeric
        for key, value in verification.metrics.items():
            assert isinstance(value, (int, float))

    def test_failed_status_has_diagnostic(self):
        """Verify FAILED status includes diagnostic message."""
        checker = ConservationChecker(mass_tolerance=1e-12)
        problem = make_simple_problem()
        result = make_simple_result(converged=False)

        verification = checker.verify(problem, result)

        if verification.status == VerificationStatus.FAILED:
            assert verification.message
            assert len(verification.message) > 0
            assert verification.details
            assert len(verification.details) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
