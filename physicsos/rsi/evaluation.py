"""Development selection followed by one-use, frozen holdout evidence."""
from __future__ import annotations

import json
import math
import sqlite3

from physicsos.runtime.artifacts import checked_path
from physicsos.schemas.common import ArtifactRef
from physicsos.schemas.rsi import EvaluateStrategiesInput, EvaluateStrategiesOutput, PromoteStrategyInput


def _score(outcomes) -> tuple:
    verified = sum(item.status == "verified" for item in outcomes)
    failed = sum(item.status == "failed" for item in outcomes)
    errors = [item.metrics.get("error_ratio", math.inf) for item in outcomes]
    return (-verified, failed, sum(errors) / len(errors))


def eligibility(suite, selected: ArtifactRef, baseline: ArtifactRef | None, outcomes) -> list[str]:
    reasons = []
    candidates = {item.benchmark_id: item for item in outcomes if item.strategy.checksum == selected.checksum}
    incumbents = {item.benchmark_id: item for item in outcomes if baseline and item.strategy.checksum == baseline.checksum}
    for case in suite.spec.benchmarks:
        outcome = candidates.get(case.id)
        if outcome is None or outcome.status != "verified":
            reasons.append(f"{case.split}:{case.id} lacks verified candidate evidence.")
        elif suite.spec.policy.require_convergence and "convergence" not in outcome.evidence:
            reasons.append(f"{case.id} lacks required actual refinement evidence.")
    if baseline:
        holdout = [case for case in suite.spec.benchmarks if case.split == "holdout"]
        gains = []
        expanded = False
        for case in holdout:
            candidate, incumbent = candidates.get(case.id), incumbents.get(case.id)
            if incumbent is None:
                reasons.append(f"{case.id} lacks baseline evidence.")
                continue
            if candidate is None or candidate.status != "verified":
                continue
            if incumbent.status != "verified":
                # Inconclusive baseline evidence cannot prove an improvement.
                expanded |= incumbent.status == "failed"
                continue
            if candidate.evidence.get("comparison_probes") != incumbent.evidence.get("comparison_probes"):
                reasons.append(f"{case.id} did not compare identical physical probes.")
                continue
            gain = incumbent.metrics["error_ratio"] - candidate.metrics["error_ratio"]
            gains.append(gain)
            if gain < -suite.spec.policy.max_error_regression:
                reasons.append(f"{case.id} regressed beyond the permitted error budget.")
        if not expanded and (len(gains) != len(holdout) or sum(gains) / len(gains) < suite.spec.policy.min_error_improvement):
            reasons.append("Holdout evidence does not demonstrate the required accuracy or coverage improvement.")
    return reasons


def evaluate_strategies(runtime, input: EvaluateStrategiesInput, *, probe_directory=None, expected_generation: int | None = None) -> EvaluateStrategiesOutput:
    directory = runtime._directory("evaluations")
    evaluation_id = directory.name
    outcomes, references = [], []
    selected = None
    baseline = None
    selection_ref = None
    suite = None
    active = {"generation": 0}
    status = "blocked"
    reasons = []
    used_runs = 0
    try:
        suite = runtime._suite(input.suite)
        active = runtime.store.active(suite.spec.scope)
        if expected_generation is not None and active["generation"] != expected_generation:
            raise ValueError("The active strategy changed during this revision campaign.")
        baseline = input.baseline or active["strategy"]
        candidates = list({item.checksum: item for item in input.candidates}.values())
        if any(item.checksum is None for item in input.candidates):
            raise ValueError("Candidate revisions need checksums.")
        strategies = ([baseline] if baseline else []) + candidates
        strategies = list({item.checksum: item for item in strategies}.values())
        for reference in strategies:
            if runtime._strategy(reference).spec.scope != suite.spec.scope:
                raise ValueError("Every candidate and baseline must match the exact suite scope.")
        if active["strategy"] and baseline.checksum != active["strategy"].checksum:
            raise ValueError("Compare against the active strategy before replacing it.")
        development = [case for case in suite.spec.benchmarks if case.split == "development"]
        holdout = [case for case in suite.spec.benchmarks if case.split == "holdout"]
        cost = lambda case: 1 + (len(case.convergence.refinements) if case.convergence else 0)
        required_runs = len(strategies) * sum(map(cost, development))
        if input.include_holdout:
            required_runs += (2 if baseline else 1) * sum(map(cost, holdout))
        if required_runs > input.max_kernel_runs:
            raise ValueError(f"The complete evaluation needs up to {required_runs} kernel runs; budget is {input.max_kernel_runs}.")
        if input.include_holdout:
            runtime.store.check_holdout([suite.problem_identities[case.id] for case in holdout])
        runtime.store.expose_development([suite.problem_identities[case.id] for case in development], evaluation_id)

        def evaluate(reference, cases):
            nonlocal used_runs
            results = []
            for case in cases:
                used_runs += cost(case)
                outcome, evidence = runtime._run_benchmark(reference, case, suite.problem_identities[case.id], case.split,
                                                          directory / "attempts" / reference.checksum / case.id,
                                                          (probe_directory or directory / "probes") / (case.id + ".npz"), input.timeout_seconds,
                                                          suite.resource_directories.get(case.id))
                outcomes.append(outcome)
                references.append(evidence)
                results.append(outcome)
                runtime._observe(suite.spec.scope, outcome, evidence)
            return results

        scores = {}
        for reference in strategies:
            results = evaluate(reference, development)
            scores[reference.checksum] = _score(results)
        eligible = [reference for reference in candidates if all(item.status == "verified" for item in outcomes if item.strategy.checksum == reference.checksum)]
        if not eligible:
            status = "rejected"
            reasons = ["No candidate passed every development benchmark."]
        else:
            selected = min(eligible, key=lambda item: (scores[item.checksum], item.checksum))
            selection_ref = runtime._write(directory / "selection.json", {
                "evaluation_id": evaluation_id, "suite": input.suite.model_dump(mode="json"),
                "selected": selected.model_dump(mode="json"), "baseline": baseline.model_dump(mode="json") if baseline else None,
                "expected_generation": active["generation"],
                "candidates": [item.model_dump(mode="json") for item in candidates],
                "development_outcomes": [item.model_dump(mode="json") for item in references],
            }, "rsi_selection")
            if not input.include_holdout:
                status, reasons = "needs_holdout", ["Selection used development data only; promotion needs fresh holdout evidence."]
            else:
                try:
                    runtime.store.consume_holdout(suite.holdout_fingerprint, evaluation_id, selection_ref,
                                                  [suite.problem_identities[case.id] for case in holdout])
                except sqlite3.IntegrityError:
                    raise ValueError("These holdout problems have already been exposed by another evaluation; register fresh holdout problems.") from None
                if baseline:
                    evaluate(baseline, holdout)
                if not baseline or baseline.checksum != selected.checksum:
                    evaluate(selected, holdout)
                reasons = eligibility(suite, selected, baseline, outcomes)
                status = "rejected" if reasons else "eligible"
        runtime._suite(input.suite)
        for reference in strategies:
            runtime._strategy(reference)
    except Exception as exc:
        status, reasons = "blocked", [str(exc)]
    report = runtime._write(directory / "report.json", {
        "schema_version": "physicsos.rsi_evaluation.v1", "id": evaluation_id, "status": status,
        "suite": input.suite.model_dump(mode="json"), "expected_generation": active["generation"],
        "selected": selected.model_dump(mode="json") if selected else None,
        "baseline": baseline.model_dump(mode="json") if baseline else None,
        "selection": selection_ref.model_dump(mode="json") if selection_ref else None,
        "outcomes": [item.model_dump(mode="json") for item in references],
        "reasons": reasons, "reserved_kernel_runs": used_runs,
        "selection_metric": "verified coverage, then independent error / fixed acceptance budget",
        "wall_time_used_for_promotion": False,
    }, "rsi_evaluation")
    activation_id = None
    if status == "eligible" and input.auto_promote:
        promotion = runtime.promote(PromoteStrategyInput(evaluation=report))
        status, activation_id = promotion.status, promotion.activation_id
        if status == "blocked":
            reasons = promotion.reasons
    return EvaluateStrategiesOutput(status=status, selected_strategy=selected, report=report, reasons=reasons, activation_id=activation_id)


def validate_promotion(runtime, reference: ArtifactRef):
    path = checked_path(reference, runtime.workspace)
    if reference.kind != "rsi_evaluation" or not path.is_relative_to(runtime.store.root / "evaluations"):
        raise ValueError("Promotion requires a saved RSI evaluation.")
    report = json.loads(path.read_text())
    if report["status"] != "eligible":
        raise ValueError("This evaluation is not eligible for promotion.")
    suite = runtime._suite(ArtifactRef.model_validate(report["suite"]))
    selected = ArtifactRef.model_validate(report["selected"])
    baseline = ArtifactRef.model_validate(report["baseline"]) if report["baseline"] else None
    runtime._strategy(selected)
    selection_ref = ArtifactRef.model_validate(report["selection"])
    use = runtime.store.holdout_use(suite.holdout_fingerprint)
    if use is None or use["evaluation_id"] != report["id"] or ArtifactRef.model_validate_json(use["selection"]) != selection_ref:
        raise ValueError("The evaluation has no matching frozen holdout selection.")
    selection = json.loads(checked_path(selection_ref, runtime.workspace).read_text())
    if any(selection[key] != report[key] for key in ("selected", "baseline", "expected_generation", "suite")):
        raise ValueError("Promotion inputs changed after development selection.")
    development_refs = [ArtifactRef.model_validate(item) for item in selection["development_outcomes"]]
    report_refs = [ArtifactRef.model_validate(item) for item in report["outcomes"]]
    if any(item not in report_refs for item in development_refs):
        raise ValueError("Frozen development outcomes were replaced after selection.")
    outcomes = [runtime._valid_outcome(ArtifactRef.model_validate(item)) for item in report["outcomes"]]
    cases = {case.id: case for case in suite.spec.benchmarks}
    seen = set()
    for outcome in outcomes:
        key = (outcome.strategy.checksum, outcome.benchmark_id)
        if key in seen or outcome.benchmark_id not in cases:
            raise ValueError("Duplicate or unknown benchmark evidence cannot qualify a strategy.")
        seen.add(key)
        case = cases[outcome.benchmark_id]
        if outcome.split != case.split or outcome.problem_identity != suite.problem_identities[case.id]:
            raise ValueError("Benchmark split or physical identity changed.")
        runtime._check_benchmark_evidence(outcome, case, suite.spec.scope)
        if outcome.status == "verified":
            verification = json.loads(checked_path(outcome.evidence["verification"], runtime.workspace).read_text())
            check = verification["individual_results"]["IndependentFieldReference"]
            if check["details"]["relative_tolerance"] != case.relative_tolerance or check["details"]["absolute_tolerance"] != case.absolute_tolerance:
                raise ValueError("A candidate weakened the frozen verification thresholds.")
    reasons = eligibility(suite, selected, baseline, outcomes)
    if reasons:
        raise ValueError("; ".join(reasons))
    return report, suite, selected
