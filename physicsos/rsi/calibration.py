"""Small-sample capability estimates from independent runtime evidence."""
from __future__ import annotations

import json
import math

from physicsos.runtime.artifacts import checked_path
from physicsos.schemas.common import ArtifactRef
from physicsos.schemas.rsi import AssessCapabilityInput, CapabilityEstimate
from physicsos.verification.base import ConfidenceScore


def wilson_interval(successes: int, total: int) -> list[float]:
    if total == 0:
        return [0., 1.]
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0., center - radius), min(1., center + radius)]


def assess_capability(runtime, input: AssessCapabilityInput) -> CapabilityEstimate:
    directory = runtime._directory("assessments")
    active = runtime.store.active(input.scope)
    strategy = input.strategy or active["strategy"]
    policy = runtime.store.policy_for(input.scope, strategy) if strategy else {}
    contract = {key: getattr(input, key) if getattr(input, key) is not None else policy.get(key)
                for key in ("relative_tolerance", "absolute_tolerance", "require_convergence", "expected_order", "rate_tolerance", "error_tolerance")}
    if any(getattr(input, key) is not None for key in ("expected_order", "rate_tolerance", "error_tolerance")):
        contract["require_convergence"] = True
    guidance = None
    invalid = 0
    latest = {}
    failures = {}
    if strategy:
        try:
            revision = runtime._strategy(strategy)
            if revision.spec.scope != input.scope:
                raise ValueError("The strategy does not cover the requested scope.")
            guidance = revision.spec.guidance
        except (ValueError, OSError):
            invalid += 1
        if guidance is not None:
            for observation in runtime.store.observations(input.scope, strategy):
                # Candidate selection successes are training evidence, not
                # an independent estimate of future capability.
                try:
                    outcome = runtime._valid_outcome(ArtifactRef.model_validate_json(observation["evidence"]))
                    if outcome.problem_identity != observation["identity"] or outcome.strategy.checksum != strategy.checksum or outcome.split != observation["split"]:
                        raise ValueError("Observation identity does not match its evidence.")
                    status, stage = outcome.status, outcome.stage
                    if status == "verified":
                        verification = json.loads(checked_path(outcome.evidence["verification"], runtime.workspace).read_text())
                        details = verification["individual_results"]["IndependentFieldReference"]["details"]
                        sufficient = all(contract[key] is None or details[key] <= contract[key] for key in ("relative_tolerance", "absolute_tolerance"))
                        if contract["require_convergence"]:
                            sufficient &= "convergence" in outcome.evidence
                            if "convergence" in outcome.evidence:
                                study = json.loads(checked_path(outcome.evidence["convergence"], runtime.workspace).read_text())
                                sufficient &= contract["expected_order"] is None or study["expected_order"] == contract["expected_order"]
                                sufficient &= all(contract[key] is None or study.get(key, float("inf")) <= contract[key] for key in ("rate_tolerance", "error_tolerance"))
                        if not sufficient:
                            status, stage = "uncertain", "verification_contract"
                    if outcome.status != "verified":
                        failures[(observation["identity"], observation["split"])] = (outcome, observation)
                    else:
                        failures.pop((observation["identity"], observation["split"]), None)
                    if observation["split"] != "development":
                        latest[observation["identity"]] = (status, stage, observation)
                except (ValueError, OSError, KeyError):
                    invalid += 1
                    if observation["split"] != "development":
                        latest[observation["identity"]] = ("uncertain", "invalid_evidence", observation)
    counts = {status: sum(item[0] == status for item in latest.values()) for status in ("verified", "failed", "uncertain")}
    total = len(latest)
    interval = wilson_interval(counts["verified"], total)
    probability = (counts["verified"] + 1) / (total + 2)
    if total == 0 or counts["verified"] + counts["failed"] == 0:
        confidence = ConfidenceScore.UNKNOWN
    elif total >= 20 and interval[0] >= .8:
        confidence = ConfidenceScore.HIGH
    elif total >= 5 and interval[0] >= .5:
        confidence = ConfidenceScore.MEDIUM
    else:
        confidence = ConfidenceScore.LOW
    patterns, losses = {}, []
    actions = set()
    examples = []
    for outcome, observation in failures.values():
        stage = outcome.split + ":" + outcome.stage
        patterns[stage] = patterns.get(stage, 0) + 1
        actions.update(outcome.suggested_actions)
        examples.append(ArtifactRef.model_validate_json(observation["evidence"]))
    for status, stage, observation in latest.values():
        if stage in {"invalid_evidence", "verification_contract"}:
            patterns[stage] = patterns.get(stage, 0) + 1
        if observation["split"] == "production" and observation["assessment"]:
            try:
                assessment = json.loads(checked_path(ArtifactRef.model_validate_json(observation["assessment"]), runtime.workspace).read_text())
                predicted = assessment["predicted_success_probability"]
                if assessment["scope"] != input.scope.model_dump(mode="json") or assessment["strategy"]["checksum"] != strategy.checksum or not 0 <= predicted <= 1:
                    raise ValueError("Prediction does not belong to this scope and strategy.")
                losses.append((predicted - int(status == "verified")) ** 2)
            except (ValueError, OSError, KeyError, TypeError):
                invalid += 1
    recommendation = ("No independent evidence covers this scope; generate candidates and evaluate a fresh suite."
                      if total == 0 else f"{counts['verified']} of {total} distinct holdout/production problems verified. Continue independent checks; the interval describes observed verification success, not unrestricted physics capability.")
    payload = dict(scope=input.scope, strategy=strategy, confidence=confidence,
                   independent_problems=total, **counts, success_rate=counts["verified"] / total if total else None,
                   predicted_success_probability=probability, success_interval=interval,
                   production_brier_score=sum(losses) / len(losses) if losses else None,
                   invalid_evidence=invalid, failure_patterns=patterns, failure_evidence=examples[-10:],
                   suggested_actions=sorted(actions), guidance=guidance,
                   comparison_contract=contract,
                   active_generation=active["generation"], recommendation=recommendation)
    reference = runtime._write(directory / "report.json", {
        **{key: value.model_dump(mode="json") if hasattr(value, "model_dump") else [item.model_dump(mode="json") for item in value] if key == "failure_evidence" else value for key, value in payload.items()},
        "interval_method": "95% Wilson score interval", "probability_method": "Beta(1,1) smoothed verification-pass frequency",
        "development_evidence_used": False,
    }, "rsi_capability")
    return CapabilityEstimate(**payload, report=reference)
