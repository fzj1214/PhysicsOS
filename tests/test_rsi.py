from __future__ import annotations

import json
from pathlib import Path

import pytest

from physicsos.rsi import RSIRuntime
from physicsos.runtime.artifacts import checked_path
from physicsos.schemas.case_runtime import DomainRequirements, PrepareDomainInput
from physicsos.schemas.geometry import GeometryEntity, GeometrySource, GeometrySpec
from physicsos.schemas.mesh import MeshPolicy
from physicsos.schemas.operators import PhysicsSpec
from physicsos.schemas.rsi import (
    AssessCapabilityInput, BenchmarkCase, BenchmarkSuiteInput, EvaluateStrategiesInput,
    PromoteStrategyInput, PromotionPolicy, RefinementCheck, RollbackStrategyInput,
    SolveWithStrategyInput, StrategyScope, StrategySpec,
)


# Genuine 1D P1 FEM assembly for -k*u''=-2. The runtime has no PDE fixture code.
KERNEL = '''from pathlib import Path
import json
import numpy as np

def run_case(config):
    with np.load(config["domain_artifacts"]["mesh_arrays"]) as mesh:
        points, cells = mesh["points"], mesh["line"]
    with np.load(config["domain_artifacts"]["boundary_nodes"]) as masks:
        boundary = masks["wall"]
    k = config["controls"].get("k", 1.)
    matrix = np.zeros((len(points), len(points)))
    rhs = np.zeros(len(points))
    for cell in cells:
        h = np.linalg.norm(points[cell[1]] - points[cell[0]])
        matrix[np.ix_(cell, cell)] += k / h * np.array([[1., -1.], [-1., 1.]])
        rhs[cell] -= h
    u = points[:, 0] ** 2 / k
    free, fixed = np.flatnonzero(~boundary), np.flatnonzero(boundary)
    u[free] = np.linalg.solve(matrix[np.ix_(free, free)], rhs[free] - matrix[np.ix_(free, fixed)] @ u[fixed])
    if config["controls"].get("break_at", 1e10) < points[:, 0].max():
        u *= 0
    output = Path(config["output_dir"])
    np.save(output / "solution.npy", u)
    (output / "residual_history.json").write_text(json.dumps([{"residual": float(np.linalg.norm((matrix @ u - rhs)[free]))}]))
    (output / "runtime_metadata.json").write_text('{"method":"P1 FEM"}')
    return {"status":"success"}
'''

REFERENCE = '''import numpy as np
def exact_solution(points, config):
    return np.square(points[:, 0]) / config["controls"].get("k", 1.)
'''


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(tmp_path))
    scope = StrategyScope(problem_family="poisson-dirichlet", physics_domains=["thermal"], regime="steady-conduction", dimension=1, representation="mesh")
    kernel = tmp_path / "implementations" / "kernel.py"
    kernel.parent.mkdir()
    kernel.write_text(KERNEL)
    reference = tmp_path / "references" / "reference.py"
    reference.parent.mkdir()
    reference.write_text(REFERENCE)
    return RSIRuntime(tmp_path), scope, kernel, reference


def benchmark(scope, kernel, reference, identifier, length, split, convergence=None):
    geometry = GeometrySpec(id=identifier, source=GeometrySource(kind="generated"), dimension=1,
                            entities=[GeometryEntity(id=identifier, kind="curve", metadata={"length": length})])
    prepare = PrepareDomainInput(case_id=identifier, geometry=geometry,
                                physics=PhysicsSpec(domains=scope.physics_domains, regime=scope.regime, steady=True),
                                requirements=DomainRequirements(dimension=1, whole_boundary_role="wall", required_boundary_roles=["wall"], mesh_policy=MeshPolicy(target_element_size=.35)))
    return BenchmarkCase(id=identifier, split=split, prepare=prepare, kernel_uri=str(kernel), reference_uri=str(reference),
                         controls={"k": 1.}, tunable_controls=["break_at"], relative_tolerance=.3, convergence=convergence)


def suite(setup, offset=0., convergence=None, require_convergence=False):
    runtime, scope, kernel, reference = setup
    cases = [benchmark(scope, kernel, reference, f"case-{index}", 1. + .17 * index + offset,
                       "development" if index == 0 else "holdout", convergence) for index in range(4)]
    registered = runtime.register_suite(BenchmarkSuiteInput(name="fem-suite", scope=scope, benchmarks=cases,
                                                            policy=PromotionPolicy(require_convergence=require_convergence, min_error_improvement=.005)))
    return registered


def strategy(setup, name="fine", size=.12, **kwargs):
    runtime, scope, kernel, _ = setup
    return runtime.register_strategy(StrategySpec(name=name, scope=scope, description="Mesh strategy regression fixture",
                                                  guidance="Assemble the P1 system on the current domain and verify independently.",
                                                  mesh_policy=MeshPolicy(target_element_size=size), kernel_uri=str(kernel), **kwargs))


def test_actual_evaluation_selects_and_promotes_shared_strategy(setup):
    runtime, scope, _, _ = setup
    registered = suite(setup)
    coarse, fine = strategy(setup, "coarse", .4), strategy(setup)
    result = runtime.evaluate(EvaluateStrategiesInput(suite=registered.manifest, candidates=[coarse.manifest, fine.manifest]))
    assert result.status == "promoted", result.reasons
    assert result.selected_strategy == fine.manifest
    assert runtime.store.active(scope)["strategy"] == fine.manifest
    report = json.loads(checked_path(result.report, runtime.workspace).read_text())
    outcomes = [runtime._valid_outcome(type(result.report).model_validate(item)) for item in report["outcomes"]]
    assert len(outcomes) == 5
    dev = [outcome for outcome in outcomes if outcome.split == "development"]
    assert dev[0].evidence["comparison_probes"] == dev[1].evidence["comparison_probes"]
    estimate = runtime.assess(AssessCapabilityInput(scope=scope))
    assert estimate.independent_problems == 3
    assert estimate.verified == 3
    assert estimate.confidence.value == "low"
    assert estimate.success_interval[0] < .5


def test_renamed_problems_cannot_leak_into_holdout_or_inflate_coverage(setup):
    runtime, scope, kernel, reference = setup
    cases = [benchmark(scope, kernel, reference, f"duplicate-{i}", 1., "development" if i == 0 else "holdout") for i in range(4)]
    with pytest.raises(ValueError, match="distinct physical problems"):
        runtime.register_suite(BenchmarkSuiteInput(name="duplicates", scope=scope, benchmarks=cases, policy=PromotionPolicy(require_convergence=False)))


def test_development_results_cannot_promote_or_raise_capability(setup):
    runtime, scope, _, _ = setup
    registered, candidate = suite(setup), strategy(setup)
    result = runtime.evaluate(EvaluateStrategiesInput(suite=registered.manifest, candidates=[candidate.manifest], include_holdout=False))
    assert result.status == "needs_holdout"
    assert runtime.store.active(scope)["strategy"] is None
    assert runtime.promote(PromoteStrategyInput(evaluation=result.report)).status == "blocked"
    estimate = runtime.assess(AssessCapabilityInput(scope=scope, strategy=candidate.manifest))
    assert estimate.independent_problems == 0
    assert estimate.confidence.value == "unknown"


def test_holdout_is_not_a_reusable_candidate_selection_set(setup):
    runtime, _, _, _ = setup
    registered, candidate = suite(setup), strategy(setup)
    first = runtime.evaluate(EvaluateStrategiesInput(suite=registered.manifest, candidates=[candidate.manifest], auto_promote=False))
    assert first.status == "eligible", first.reasons
    # A second registration with new IDs/paths but the same problems is exposed.
    alias = suite(setup)
    second = runtime.evaluate(EvaluateStrategiesInput(suite=alias.manifest, candidates=[candidate.manifest], auto_promote=False))
    assert second.status == "blocked"
    assert "already been exposed" in second.reasons[0]


def test_promotion_rechecks_evidence_and_active_generation(setup):
    runtime, scope, _, _ = setup
    registered, candidate = suite(setup), strategy(setup)
    result = runtime.evaluate(EvaluateStrategiesInput(suite=registered.manifest, candidates=[candidate.manifest], auto_promote=False))
    assert result.status == "eligible"
    promotion = runtime.promote(PromoteStrategyInput(evaluation=result.report))
    assert promotion.status == "promoted"
    runtime.rollback(RollbackStrategyInput(scope=scope, expected_generation=promotion.generation, reason="Exercise stale evidence guard."))
    repeated = runtime.promote(PromoteStrategyInput(evaluation=result.report))
    assert repeated.status == "blocked"
    assert "changed after this evaluation" in repeated.reasons[0]


def test_fresh_holdout_improvement_promotes_and_production_failure_rolls_back(setup):
    runtime, scope, kernel, reference = setup
    coarse = strategy(setup, "coarse", .4)
    first = runtime.evaluate(EvaluateStrategiesInput(suite=suite(setup).manifest, candidates=[coarse.manifest]))
    assert first.status == "promoted", first.reasons
    fine = strategy(setup, solver_controls={"break_at": 2.})
    second = runtime.evaluate(EvaluateStrategiesInput(suite=suite(setup, offset=.1).manifest, candidates=[fine.manifest]))
    assert second.status == "promoted", second.reasons
    production = benchmark(scope, kernel, reference, "production", 2.3, "development")
    solved = runtime.solve(SolveWithStrategyInput(problem_family=scope.problem_family, prepare=production.prepare,
                                                kernel_uri=str(kernel), reference_uri=str(reference), controls=production.controls,
                                                tunable_controls=production.tunable_controls, relative_tolerance=.3))
    assert solved.status == "failed"
    assert solved.rollback.status == "rolled_back"
    assert runtime.store.active(scope)["strategy"] == coarse.manifest
    estimate = runtime.assess(AssessCapabilityInput(scope=scope, strategy=fine.manifest))
    assert estimate.failed == 1
    assert estimate.production_brier_score is not None


def test_small_budget_stops_before_any_candidate_solver(setup):
    runtime, _, _, _ = setup
    result = runtime.evaluate(EvaluateStrategiesInput(suite=suite(setup).manifest, candidates=[strategy(setup).manifest], max_kernel_runs=1))
    assert result.status == "blocked"
    report = json.loads(checked_path(result.report, runtime.workspace).read_text())
    assert report["outcomes"] == []


def test_strategy_cannot_change_physical_material_coefficient(setup):
    runtime, _, _, _ = setup
    candidate = strategy(setup, solver_controls={"k": 100.})
    result = runtime.evaluate(EvaluateStrategiesInput(suite=suite(setup).manifest, candidates=[candidate.manifest]))
    assert result.status == "rejected"
    report = json.loads(checked_path(result.report, runtime.workspace).read_text())
    outcome = runtime._valid_outcome(type(result.report).model_validate(report["outcomes"][0]))
    assert "physical or undeclared controls" in outcome.diagnostics[0]


def test_real_refinement_is_required_by_default(setup):
    runtime, scope, _, _ = setup
    registered = suite(setup, convergence=RefinementCheck(refinements=[.43, .23, .13], rate_tolerance=1.5), require_convergence=True)
    result = runtime.evaluate(EvaluateStrategiesInput(suite=registered.manifest, candidates=[strategy(setup).manifest]))
    assert result.status == "promoted", result.reasons
    report = json.loads(checked_path(result.report, runtime.workspace).read_text())
    for item in report["outcomes"]:
        outcome = runtime._valid_outcome(type(result.report).model_validate(item))
        study = json.loads(checked_path(outcome.evidence["convergence"], runtime.workspace).read_text())
        assert len(study["rows"]) == 3
        assert study["generated_from_actual_runs"]


def test_builder_generates_case_local_kernels_against_prepared_domains(setup):
    runtime, scope, _, _ = setup
    source = runtime.workspace / "builders" / "builder.py"
    source.parent.mkdir()
    source.write_text("def build_case_kernel(config):\n    assert config['domain']['status'] == 'ready'\n    assert 'current domain' in config['guidance']\n    return {'kernel_source': " + repr(KERNEL) + "}\n")
    candidate = runtime.register_strategy(StrategySpec(name="builder", scope=scope, description="Reusable generator",
                                                       guidance="Reassemble the current domain.", builder_uri=str(source),
                                                       mesh_policy=MeshPolicy(target_element_size=.12)))
    source.write_text("raise RuntimeError('Source changed after snapshot')")
    result = runtime.evaluate(EvaluateStrategiesInput(suite=suite(setup).manifest, candidates=[candidate.manifest]))
    assert result.status == "promoted", result.reasons
    report = json.loads(checked_path(result.report, runtime.workspace).read_text())
    for item in report["outcomes"]:
        outcome = runtime._valid_outcome(type(result.report).model_validate(item))
        assert "generation_log" in outcome.evidence


def test_changed_numerical_evidence_blocks_promotion_and_calibration_credit(setup):
    runtime, scope, _, _ = setup
    registered, candidate = suite(setup), strategy(setup)
    result = runtime.evaluate(EvaluateStrategiesInput(suite=registered.manifest, candidates=[candidate.manifest], auto_promote=False))
    assert result.status == "eligible"
    report = json.loads(checked_path(result.report, runtime.workspace).read_text())
    outcome = runtime._valid_outcome(type(result.report).model_validate(report["outcomes"][-1]))
    run = json.loads(checked_path(outcome.evidence["run"], runtime.workspace).read_text())
    checked_path(type(result.report).model_validate(run["artifacts"]["solution"]), runtime.workspace).write_bytes(b"corrupted")
    promoted = runtime.promote(PromoteStrategyInput(evaluation=result.report))
    assert promoted.status == "blocked"
    estimate = runtime.assess(AssessCapabilityInput(scope=scope, strategy=candidate.manifest))
    assert estimate.verified == 2
    assert estimate.uncertain == 1
    assert estimate.invalid_evidence > 0


def test_partial_holdout_overlap_is_exposed_even_if_suite_fingerprint_changes(setup):
    runtime, _, _, _ = setup
    candidate = strategy(setup)
    first = runtime.evaluate(EvaluateStrategiesInput(suite=suite(setup).manifest, candidates=[candidate.manifest], auto_promote=False))
    assert first.status == "eligible"
    alias = suite(setup, offset=.17)
    second = runtime.evaluate(EvaluateStrategiesInput(suite=alias.manifest, candidates=[candidate.manifest], auto_promote=False))
    assert second.status == "blocked"
    assert "already been exposed" in second.reasons[0]


def test_failed_development_evidence_produces_revision_guidance(setup):
    runtime, scope, _, _ = setup
    candidate = strategy(setup, solver_controls={"k": 100.})
    runtime.evaluate(EvaluateStrategiesInput(suite=suite(setup).manifest, candidates=[candidate.manifest]))
    estimate = runtime.assess(AssessCapabilityInput(scope=scope, strategy=candidate.manifest))
    assert estimate.independent_problems == 0
    assert estimate.failure_patterns["development:preparation"] == 1
    assert estimate.failure_evidence
    assert estimate.suggested_actions


def test_looser_accuracy_evidence_cannot_raise_confidence_for_stricter_target(setup):
    runtime, scope, _, _ = setup
    candidate = strategy(setup)
    result = runtime.evaluate(EvaluateStrategiesInput(suite=suite(setup).manifest, candidates=[candidate.manifest]))
    assert result.status == "promoted"
    estimate = runtime.assess(AssessCapabilityInput(scope=scope, relative_tolerance=1e-5))
    assert estimate.verified == 0
    assert estimate.uncertain == 3
    assert estimate.confidence.value == "unknown"
    assert estimate.comparison_contract["relative_tolerance"] == 1e-5


def test_candidate_cannot_relax_geometry_fidelity_limits(setup):
    from physicsos.schemas.geometry_repair import GeometryRepairOptions
    runtime, _, _, _ = setup
    candidate = strategy(setup, repair_options=GeometryRepairOptions(max_relative_surface_distance=.5))
    result = runtime.evaluate(EvaluateStrategiesInput(suite=suite(setup).manifest, candidates=[candidate.manifest]))
    assert result.status == "rejected"
    report = json.loads(checked_path(result.report, runtime.workspace).read_text())
    outcome = runtime._valid_outcome(type(result.report).model_validate(report["outcomes"][0]))
    assert "cannot weaken" in outcome.diagnostics[0]


def test_prior_development_problem_cannot_be_relabelled_as_fresh_holdout(setup):
    runtime, scope, kernel, reference = setup
    candidate = strategy(setup)
    first = runtime.evaluate(EvaluateStrategiesInput(suite=suite(setup).manifest, candidates=[candidate.manifest], include_holdout=False))
    assert first.status == "needs_holdout"
    cases = [benchmark(scope, kernel, reference, "new-dev", .77, "development")]
    cases.extend(benchmark(scope, kernel, reference, f"new-hold-{i}", length, "holdout") for i, length in enumerate([1., 1.9, 2.1]))
    alias = runtime.register_suite(BenchmarkSuiteInput(name="role-switch", scope=scope, benchmarks=cases, policy=PromotionPolicy(require_convergence=False)))
    second = runtime.evaluate(EvaluateStrategiesInput(suite=alias.manifest, candidates=[candidate.manifest]))
    assert second.status == "blocked"
    assert "development or holdout" in second.reasons[0]
    assert json.loads(checked_path(second.report, runtime.workspace).read_text())["outcomes"] == []


def test_existing_history_migrates_development_exposure(setup):
    from physicsos.schemas.common import ArtifactRef
    runtime, scope, _, _ = setup
    reference = ArtifactRef(uri="historical.json", kind="rsi_outcome", checksum="historical")
    runtime.store.observe(scope, reference, "previous-development", "development", reference)
    with runtime.store.connection() as connection:
        connection.execute("DROP TABLE development_problems")
        connection.execute("PRAGMA user_version=0")
    with pytest.raises(ValueError, match="already been exposed"):
        runtime.store.check_holdout(["previous-development"])


def test_rollback_clears_default_if_predecessor_qualification_is_corrupted(setup):
    runtime, scope, _, _ = setup
    coarse, fine = strategy(setup, "coarse", .4), strategy(setup)
    first = runtime.evaluate(EvaluateStrategiesInput(suite=suite(setup).manifest, candidates=[coarse.manifest]))
    assert first.status == "promoted"
    second = runtime.evaluate(EvaluateStrategiesInput(suite=suite(setup, offset=.1).manifest, candidates=[fine.manifest]))
    assert second.status == "promoted", second.reasons
    checked_path(first.report, runtime.workspace).write_bytes(b"lost numerical qualification")
    reverted = runtime.rollback(RollbackStrategyInput(scope=scope, expected_generation=runtime.store.active(scope)["generation"], reason="Exercise qualification invalidation."))
    assert reverted.status == "rolled_back"
    assert reverted.active_strategy is None
    assert runtime.store.active(scope)["strategy"] is None


def test_qualification_rejects_copied_numeric_evidence_for_different_problems(setup):
    runtime, _, _, _ = setup
    registered, candidate = suite(setup), strategy(setup)
    evaluated = runtime.evaluate(EvaluateStrategiesInput(suite=registered.manifest, candidates=[candidate.manifest], auto_promote=False))
    assert evaluated.status == "eligible"
    report_path = checked_path(evaluated.report, runtime.workspace)
    report = json.loads(report_path.read_text())
    original = [runtime._valid_outcome(type(evaluated.report).model_validate(value)) for value in report["outcomes"]]
    held = next(item for item in original if item.split == "holdout")
    forged = [value for value, outcome in zip(report["outcomes"], original) if outcome.split == "development"]
    for case in registered.suite.spec.benchmarks:
        if case.split == "holdout":
            copied = held.model_copy(deep=True)
            copied.benchmark_id = case.id
            copied.problem_identity = registered.suite.problem_identities[case.id]
            reference = runtime._write(report_path.parent / (case.id + "-copy.json"), copied, "rsi_outcome")
            forged.append(reference.model_dump(mode="json"))
    report["outcomes"] = forged
    fake_report = runtime._write(report_path.parent / "copied-report.json", report, "rsi_evaluation")
    result = runtime.promote(PromoteStrategyInput(evaluation=fake_report))
    assert result.status == "blocked"
    assert "prepared physical problem differs" in result.reasons[0]


def test_verification_cannot_be_attached_to_another_valid_run(setup):
    runtime, _, _, _ = setup
    evaluated = runtime.evaluate(EvaluateStrategiesInput(suite=suite(setup).manifest, candidates=[strategy(setup).manifest], auto_promote=False))
    report_path = checked_path(evaluated.report, runtime.workspace)
    rows = json.loads(report_path.read_text())["outcomes"]
    first = runtime._valid_outcome(type(evaluated.report).model_validate(rows[0]))
    second = runtime._valid_outcome(type(evaluated.report).model_validate(rows[1]))
    first.evidence["verification"] = second.evidence["verification"]
    copied = runtime._write(report_path.parent / "different-verification.json", first, "rsi_outcome")
    with pytest.raises(ValueError, match="different run"):
        runtime._valid_outcome(copied)


def test_case_context_refreshes_learned_guidance_after_rollback(setup):
    from physicsos.schemas.rsi import BindCaseStrategyInput
    from physicsos.tools.case_tools import BuildPaperContextWindowInput, BuildTAPSDerivationPromptInput, build_paper_context_window, build_taps_derivation_prompt
    runtime, scope, _, _ = setup
    candidate = strategy(setup)
    promoted = runtime.evaluate(EvaluateStrategiesInput(suite=suite(setup).manifest, candidates=[candidate.manifest]))
    assert promoted.status == "promoted"
    bound = runtime.bind_context(BindCaseStrategyInput(case_id="agent-case", assessment=AssessCapabilityInput(scope=scope)))
    assert bound.status == "available"
    built = build_paper_context_window(BuildPaperContextWindowInput(case_id="agent-case"))
    assert candidate.strategy.spec.guidance in checked_path(bound.guidance, runtime.workspace).read_text()
    # Context tools return legacy file references without checksums.
    context_path = runtime.workspace / "cases" / "agent-case" / "context" / "context_window.md"
    assert candidate.strategy.spec.guidance in context_path.read_text()
    reverted = runtime.rollback(RollbackStrategyInput(scope=scope, expected_generation=runtime.store.active(scope)["generation"], reason="Refresh bound guidance."))
    assert reverted.status == "rolled_back"
    build_paper_context_window(BuildPaperContextWindowInput(case_id="agent-case"))
    build_taps_derivation_prompt(BuildTAPSDerivationPromptInput(case_id="agent-case"))
    assert candidate.strategy.spec.guidance not in context_path.read_text()
    pointer = json.loads((context_path.parent / "rsi_strategy.json").read_text())
    assert pointer["status"] == "needs_strategy"
    assert pointer["strategy"] is None
    assert pointer["active_generation"] == reverted.generation
    assert "needs_strategy" in (runtime.workspace / "cases" / "agent-case" / "taps" / "derivation_prompt.md").read_text()


def test_corrupt_binding_clears_cached_guidance(setup):
    from physicsos.schemas.rsi import BindCaseStrategyInput
    from physicsos.tools.case_tools import BuildPaperContextWindowInput, build_paper_context_window
    runtime, scope, _, _ = setup
    runtime.bind_context(BindCaseStrategyInput(case_id="binding", assessment=AssessCapabilityInput(scope=scope)))
    directory = runtime.workspace / "cases" / "binding" / "context"
    (directory / "rsi_strategy.md").write_text("stale strategy guidance")
    (directory / "rsi_binding.json").write_text("corrupted")
    result = build_paper_context_window(BuildPaperContextWindowInput(case_id="binding"))
    assert any("needs review" in warning for warning in result.warnings)
    assert "stale strategy guidance" not in (directory / "context_window.md").read_text()
    assert json.loads((directory / "rsi_strategy.json").read_text())["status"] == "needs_review"


def revision_provider(setup, code):
    from physicsos.schemas.rsi import RevisionProviderSpec
    runtime, _, _, _ = setup
    source = runtime.workspace / "revision_provider" / "provider.py"
    source.parent.mkdir(exist_ok=True)
    source.write_text(code)
    return runtime.register_revision_provider(RevisionProviderSpec(name="revision-provider", description="Development-driven revision fixture", python_uri=str(source)))


def test_campaign_fixes_real_kernel_failure_and_promotes_without_holdout_feedback(setup):
    from physicsos.schemas.rsi import ImproveStrategiesInput
    runtime, scope, kernel, _ = setup
    kernel.write_text(KERNEL.replace('np.save(output / "solution.npy", u)', 'np.save(output / "solution.npy", u * 0)'))
    initial = strategy(setup)
    provider = revision_provider(setup, '''def revise_strategy(config):
    assert all(item["split"] == "development" for item in config["development_outcomes"])
    assert all(item["split"] == "development" for item in config["development_cases"])
    assert all("reference_uri" not in item for item in config["development_cases"])
    assert any(item["status"] == "failed" for item in config["development_outcomes"])
    source = config["implementation"]["source"].replace('np.save(output / "solution.npy", u * 0)', 'np.save(output / "solution.npy", u)')
    return {"rationale":"Restore the assembled field; independent comparison exposed zero output.", "kernel_source":source}
''')
    result = runtime.improve(ImproveStrategiesInput(suite=suite(setup).manifest, initial_strategy=initial.manifest, revision_provider=provider.manifest, max_kernel_runs=16))
    assert result.status == "promoted", result.reasons
    assert result.provider_calls == 1
    assert len(result.revisions) == 1
    assert runtime.store.active(scope)["strategy"] == result.selected_strategy
    revised = runtime._strategy(result.selected_strategy)
    assert revised.spec.parent == initial.manifest
    assert result.reserved_kernel_runs <= 16
    report = json.loads(checked_path(result.report, runtime.workspace).read_text())
    assert not report["holdout_feedback_used_for_revision"]
    assert result.final_evaluation is not None
    first = json.loads(checked_path(result.evaluations[0], runtime.workspace).read_text())
    revised_report = json.loads(checked_path(result.evaluations[1], runtime.workspace).read_text())
    original = runtime._valid_outcome(type(result.report).model_validate(first["outcomes"][0]))
    later = [runtime._valid_outcome(type(result.report).model_validate(ref)) for ref in revised_report["outcomes"]]
    assert all(item.evidence["comparison_probes"] == original.evidence["comparison_probes"] for item in later)


def test_campaign_budget_reserves_final_evidence_before_calling_provider(setup):
    from physicsos.schemas.rsi import ImproveStrategiesInput
    runtime, _, _, _ = setup
    provider = revision_provider(setup, "def revise_strategy(config):\n    raise AssertionError('Provider must not run')\n")
    result = runtime.improve(ImproveStrategiesInput(suite=suite(setup).manifest, initial_strategy=strategy(setup).manifest, revision_provider=provider.manifest, max_kernel_runs=1))
    assert result.status == "budget_exhausted"
    assert result.provider_calls == result.reserved_kernel_runs == 0
    assert result.evaluations == []


def test_campaign_stagnation_stops_without_exposing_holdouts(setup):
    from physicsos.schemas.rsi import ImproveStrategiesInput
    runtime, _, kernel, _ = setup
    kernel.write_text(KERNEL.replace('np.save(output / "solution.npy", u)', 'np.save(output / "solution.npy", u * 0)'))
    initial = strategy(setup)
    registered = suite(setup)
    provider = revision_provider(setup, "def revise_strategy(config):\n    return {'rationale':'No implementation change available.'}\n")
    result = runtime.improve(ImproveStrategiesInput(suite=registered.manifest, initial_strategy=initial.manifest, revision_provider=provider.manifest, max_revisions=5, max_stagnant_revisions=2))
    assert result.status == "rejected", result.reasons
    assert result.provider_calls == 2
    assert len(result.revisions) == 2
    assert result.final_evaluation is None
    assert runtime.store.holdout_use(registered.suite.holdout_fingerprint) is None
    report = json.loads(checked_path(result.report, runtime.workspace).read_text())
    assert report["stop_reason"] == "stagnation_limit"


def test_revision_provider_cannot_patch_frozen_scope_or_benchmark_threshold(setup):
    from physicsos.schemas.rsi import ImproveStrategiesInput
    runtime, _, _, _ = setup
    provider = revision_provider(setup, "def revise_strategy(config):\n    return {'rationale':'Attempt to weaken comparison.', 'patch':{'relative_tolerance':100.}}\n")
    result = runtime.improve(ImproveStrategiesInput(suite=suite(setup).manifest, initial_strategy=strategy(setup).manifest, revision_provider=provider.manifest))
    assert result.status == "blocked"
    assert result.revisions == []
    assert result.final_evaluation is None
    assert "relative_tolerance" in result.reasons[0]


def test_provider_stop_retains_development_artifacts_and_needs_holdout(setup):
    from physicsos.schemas.rsi import ImproveStrategiesInput
    runtime, _, _, _ = setup
    provider = revision_provider(setup, "def revise_strategy(config):\n    return {'stop':True,'rationale':'The supplied implementation needs no further changes.'}\n")
    result = runtime.improve(ImproveStrategiesInput(suite=suite(setup).manifest, initial_strategy=strategy(setup).manifest, revision_provider=provider.manifest, include_holdout=False))
    assert result.status == "needs_holdout"
    assert result.provider_calls == 1
    assert len(result.evaluations) == 1
    assert result.final_evaluation is None


def test_campaign_rejects_provider_request_mutation(setup):
    from physicsos.schemas.rsi import ImproveStrategiesInput
    runtime, _, _, _ = setup
    provider = revision_provider(setup, "from pathlib import Path\ndef revise_strategy(config):\n    Path('context.json').write_text('{}')\n    return {'rationale':'Attempted request mutation.'}\n")
    result = runtime.improve(ImproveStrategiesInput(suite=suite(setup).manifest, initial_strategy=strategy(setup).manifest, revision_provider=provider.manifest))
    assert result.status == "blocked"
    assert "changed its frozen" in result.reasons[0]


def test_campaign_improves_incumbent_and_keeps_default_if_final_holdouts_fail(setup):
    from physicsos.schemas.rsi import ImproveStrategiesInput
    runtime, scope, _, _ = setup
    coarse = strategy(setup, "coarse", .4)
    bootstrap = runtime.evaluate(EvaluateStrategiesInput(suite=suite(setup).manifest, candidates=[coarse.manifest]))
    assert bootstrap.status == "promoted"
    provider = revision_provider(setup, '''def revise_strategy(config):
    return {"rationale":"Refine development mesh; regression fixture fails on unseen longer shapes.", "patch":{"mesh_policy":{"target_element_size":.12},"solver_controls":{"break_at":1.23}}}
''')
    registered = suite(setup, offset=.1)
    result = runtime.improve(ImproveStrategiesInput(suite=registered.manifest, revision_provider=provider.manifest, max_revisions=3))
    assert result.status == "rejected", result.reasons
    assert result.provider_calls == 1
    assert result.final_evaluation is not None
    assert runtime.store.active(scope)["strategy"] == coarse.manifest
    assert runtime.store.holdout_use(registered.suite.holdout_fingerprint) is not None


def test_campaign_budget_can_finish_qualified_seed_without_starting_another_revision(setup):
    from physicsos.schemas.rsi import ImproveStrategiesInput
    runtime, _, _, _ = setup
    provider = revision_provider(setup, "def revise_strategy(config):\n    raise AssertionError('Final evidence was reserved')\n")
    result = runtime.improve(ImproveStrategiesInput(suite=suite(setup).manifest, initial_strategy=strategy(setup).manifest, revision_provider=provider.manifest, max_kernel_runs=5))
    assert result.status == "promoted", result.reasons
    assert result.provider_calls == 0
    assert result.reserved_kernel_runs == 5


def test_provider_snapshots_survive_source_edits_and_campaign_lineage_is_fixed(setup):
    from physicsos.schemas.rsi import ImproveStrategiesInput
    runtime, _, _, _ = setup
    provider = revision_provider(setup, "def revise_strategy(config):\n    return {'rationale':'Refine the globally shared mesh policy.', 'patch':{'mesh_policy':{'target_element_size':.1}}}\n")
    (runtime.workspace / "revision_provider" / "provider.py").write_text("raise AssertionError('Authoring file changed')")
    initial = strategy(setup, "initial", .4)
    result = runtime.improve(ImproveStrategiesInput(suite=suite(setup).manifest, initial_strategy=initial.manifest, revision_provider=provider.manifest))
    assert result.status == "promoted", result.reasons
    assert runtime._strategy(result.selected_strategy).spec.parent == initial.manifest


def test_registry_change_before_final_validation_stops_campaign(setup, monkeypatch):
    from physicsos.schemas.rsi import ImproveStrategiesInput
    runtime, scope, _, _ = setup
    provider = revision_provider(setup, "def revise_strategy(config):\n    return {'rationale':'Refine mesh.', 'patch':{'mesh_policy':{'target_element_size':.1}}}\n")
    original = runtime.store.active
    calls = 0
    def changed(request_scope):
        nonlocal calls
        calls += 1
        state = original(request_scope)
        if calls >= 3:
            state["generation"] = state["generation"] + 1
        return state
    monkeypatch.setattr(runtime.store, "active", changed)
    result = runtime.improve(ImproveStrategiesInput(suite=suite(setup).manifest, initial_strategy=strategy(setup).manifest, revision_provider=provider.manifest))
    assert result.status == "blocked"
    assert result.provider_calls == 0
    assert result.final_evaluation is None
    assert original(scope)["strategy"] is None
