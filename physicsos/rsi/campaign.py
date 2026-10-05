"""Budgeted strategy revision using development evidence and one final holdout."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import shutil
import sys

from physicsos.paths import resolve_workspace_path
from physicsos.runtime.artifacts import checked_path, digest, write_json
from physicsos.runtime.execution import run_process
from physicsos.schemas.common import ArtifactRef
from physicsos.schemas.rsi import (
    EvaluateStrategiesInput, ImproveStrategiesInput, ImproveStrategiesOutput,
    RevisionProposal, StrategySpec,
)
from physicsos.rsi.evaluation import _score, evaluate_strategies


class _BudgetExhausted(ValueError):
    pass


def _read_outcomes(runtime, evaluation: ArtifactRef):
    report = json.loads(checked_path(evaluation, runtime.workspace).read_text())
    outcomes = [runtime._valid_outcome(ArtifactRef.model_validate(value)) for value in report["outcomes"]]
    if any(outcome.split != "development" for outcome in outcomes):
        raise ValueError("Revision feedback must contain development evidence only.")
    return report, outcomes


def _development_data(suite):
    return [case.model_dump(mode="json", exclude={"reference_uri", "kernel_uri"})
            for case in suite.spec.benchmarks if case.split == "development"]


def _implementation(runtime, revision):
    uri = revision.spec.builder_uri or revision.spec.kernel_uri
    if uri is None:
        return {"kind": "case_local", "source": None, "helpers": {}}
    source = resolve_workspace_path(uri, workspace=runtime.workspace)
    return {"kind": "builder" if revision.spec.builder_uri else "kernel",
            "filename": source.name, "source": source.read_text(),
            "helpers": {path.name: path.read_text() for path in source.parent.glob("*.py") if path != source}}


def _revise(runtime, provider_ref, parent_ref, suite_ref, suite, outcomes, round_index, budget, directory, timeout):
    provider = runtime._revision_provider(provider_ref)
    parent = runtime._strategy(parent_ref)
    directory.mkdir(parents=True)
    source = resolve_workspace_path(provider.spec.python_uri, workspace=runtime.workspace)
    for helper in source.parent.glob("*.py"):
        shutil.copyfile(helper, directory / helper.name)
    context = {
        "schema_version": "physicsos.rsi_revision_context.v1", "round": round_index,
        "scope": suite.spec.scope.model_dump(mode="json"),
        "parent_strategy": parent.spec.model_dump(mode="json"),
        "implementation": _implementation(runtime, parent),
        "development_cases": _development_data(suite),
        "development_outcomes": [outcome.model_dump(mode="json") for outcome in outcomes],
        "remaining_budget": budget,
        "constraints": ["Keep the physics scope, physical data and benchmark acceptance criteria fixed.",
                        "Use only declared numerical controls; preserve geometry quality and boundary requirements.",
                        "Return a reusable strategy patch and optional run_case/config or build_case_kernel implementation.",
                        "Only development feedback is available; final holdout evidence is not revision input."],
    }
    write_json(directory / "context.json", context)
    frozen = {str(path): digest(path) for path in [*directory.glob("*.py"), directory / "context.json"]}
    command = [sys.executable, str(Path(__file__).parents[1] / "runtime" / "kernel_worker.py"),
               "--kernel", str(directory / source.name), "--entrypoint", provider.spec.entrypoint,
               "--config", str(directory / "context.json"), "--response", str(directory / "response.json")]
    process = run_process(command, directory, timeout)
    runtime._write(directory / "execution_log.json", process, "rsi_revision_log")
    response_path = directory / "response.json"
    response = json.loads(response_path.read_text()) if response_path.is_file() else {}
    if process["returncode"] != 0 or not response.get("ok"):
        raise ValueError("Revision provider failed: " + str(response.get("error") or process["stderr"][-1000:]))
    if any(not Path(path).is_file() or digest(Path(path)) != checksum for path, checksum in frozen.items()):
        raise ValueError("Revision provider changed its frozen implementation or request.")
    runtime._revision_provider(provider_ref)
    runtime._strategy(parent_ref)
    runtime._suite(suite_ref)
    proposal = RevisionProposal.model_validate(response["result"])
    if proposal.stop:
        return proposal, None
    spec = parent.spec.model_dump(mode="json")
    patch = proposal.patch.model_dump(mode="json", exclude_unset=True)
    spec.update(patch)
    spec["name"] = patch.get("name") or parent.spec.name[:72] + f"-r{round_index}"
    spec["parent"] = parent_ref.model_dump(mode="json")
    if spec["solver_controls"] is None:
        raise ValueError("A solver_controls patch must be a dictionary.")
    generated = proposal.kernel_source if proposal.kernel_source is not None else proposal.builder_source
    if generated is not None:
        entrypoint = "run_case" if proposal.kernel_source is not None else "build_case_kernel"
        if not any(isinstance(node, ast.FunctionDef) and node.name == entrypoint for node in ast.parse(generated).body):
            raise ValueError(f"Revision source must define {entrypoint}(config).")
        implementation_dir = directory / "implementation"
        implementation_dir.mkdir()
        prior_uri = parent.spec.builder_uri or parent.spec.kernel_uri
        if prior_uri:
            for helper in resolve_workspace_path(prior_uri, workspace=runtime.workspace).parent.glob("*.py"):
                shutil.copyfile(helper, implementation_dir / helper.name)
        path = implementation_dir / ("kernel.py" if entrypoint == "run_case" else "builder.py")
        path.write_text(generated, encoding="utf-8")
        spec["kernel_uri"] = str(path) if entrypoint == "run_case" else None
        spec["builder_uri"] = str(path) if entrypoint == "build_case_kernel" else None
    return proposal, runtime.register_strategy(StrategySpec.model_validate(spec)).manifest


def improve_strategies(runtime, input: ImproveStrategiesInput) -> ImproveStrategiesOutput:
    directory = runtime._directory("campaigns")
    evaluations, revisions, provider_evidence = [], [], []
    selected = final = None
    provider_calls = used_runs = 0
    status = "blocked"
    reasons = []
    terminal = "not_started"
    expected_generation = None
    suite = None
    try:
        suite = runtime._suite(input.suite)
        provider = runtime._revision_provider(input.revision_provider)
        active = runtime.store.active(suite.spec.scope)
        expected_generation = active["generation"]
        incumbent = active["strategy"]
        initial = input.initial_strategy or incumbent
        if initial is None:
            raise ValueError("An empty scope needs an initial reusable strategy revision.")
        if runtime._strategy(initial).spec.scope != suite.spec.scope:
            raise ValueError("Initial strategy does not match the frozen suite scope.")
        development = [case for case in suite.spec.benchmarks if case.split == "development"]
        holdout = [case for case in suite.spec.benchmarks if case.split == "holdout"]
        case_cost = lambda case: 1 + (len(case.convergence.refinements) if case.convergence else 0)
        development_cost = sum(map(case_cost, development))
        holdout_cost = sum(map(case_cost, holdout))
        final_reservation = (2 if incumbent else 1) * (development_cost + holdout_cost) if input.include_holdout else 0
        if input.include_holdout:
            runtime.store.check_holdout([suite.problem_identities[case.id] for case in holdout])
        anchor = incumbent or initial

        def check_state():
            if runtime.store.active(suite.spec.scope)["generation"] != expected_generation:
                raise ValueError("The active strategy changed during this revision campaign.")
            runtime._suite(input.suite)
            runtime._revision_provider(input.revision_provider)

        def evaluate(candidates, reserve_final=True, include_holdout=False):
            nonlocal used_runs
            check_state()
            groups = {reference.checksum for reference in candidates}
            baseline = incumbent if include_holdout else anchor
            if baseline:
                groups.add(baseline.checksum)
            cost = len(groups) * development_cost + ((2 if baseline else 1) * holdout_cost if include_holdout else 0)
            reserve = final_reservation if reserve_final else 0
            if used_runs + cost + reserve > input.max_kernel_runs:
                raise _BudgetExhausted(f"Remaining budget cannot cover {cost} evaluation runs plus {reserve} reserved final runs.")
            evaluated = evaluate_strategies(runtime, EvaluateStrategiesInput(
                suite=input.suite, candidates=candidates, baseline=baseline,
                include_holdout=include_holdout, auto_promote=input.auto_promote if include_holdout else False,
                max_kernel_runs=cost, timeout_seconds=input.timeout_seconds,
            ), probe_directory=directory / "probes", expected_generation=expected_generation)
            report = json.loads(checked_path(evaluated.report, runtime.workspace).read_text())
            used_runs += report["reserved_kernel_runs"]
            evaluations.append(evaluated.report)
            if evaluated.status == "blocked":
                raise ValueError("; ".join(evaluated.reasons))
            return evaluated

        first = evaluate([initial])
        _, outcomes = _read_outcomes(runtime, first.report)
        best = initial
        by_revision = lambda ref, rows: [row for row in rows if row.strategy.checksum == ref.checksum]
        best_outcomes = by_revision(best, outcomes)
        anchor_outcomes = by_revision(anchor, outcomes)
        best_score, anchor_score = _score(best_outcomes), _score(anchor_outcomes)
        feedback = best_outcomes
        stagnant = 0
        terminal = "revision_limit"
        for round_index in range(1, input.max_revisions + 1):
            passed = all(outcome.status == "verified" for outcome in best_outcomes)
            progressed = best_score < anchor_score or (incumbent is None and best.checksum != initial.checksum)
            target_met = input.development_error_target is None or all(outcome.metrics.get("error_ratio", float("inf")) <= input.development_error_target for outcome in best_outcomes)
            if passed and progressed and target_met:
                terminal = "development_goal_met"
                break
            check_state()
            next_cost = (len({anchor.checksum, best.checksum}) + 1) * development_cost
            if used_runs + next_cost + final_reservation > input.max_kernel_runs:
                terminal = "revision_budget_exhausted"
                break
            provider_calls += 1
            round_directory = directory / "revisions" / str(round_index)
            try:
                proposal, child = _revise(runtime, input.revision_provider, best, input.suite, suite, feedback, round_index,
                                          {"kernel_runs": input.max_kernel_runs - used_runs - final_reservation,
                                           "revisions": input.max_revisions - round_index}, round_directory, input.timeout_seconds)
            finally:
                for path in round_directory.glob("*.json"):
                    provider_evidence.append(runtime._write(directory / "provider_records" / (str(round_index) + "-" + path.name), json.loads(path.read_text()), "rsi_revision_evidence"))
            if proposal.stop:
                terminal = "provider_stopped"
                reasons.append(proposal.rationale)
                break
            revisions.append(child)
            evaluated = evaluate(list({ref.checksum: ref for ref in (best, child)}.values()))
            _, outcomes = _read_outcomes(runtime, evaluated.report)
            child_outcomes = by_revision(child, outcomes)
            current_best_outcomes = by_revision(best, outcomes)
            child_score, previous_score = _score(child_outcomes), _score(current_best_outcomes)
            feedback = child_outcomes
            if child_score < previous_score:
                best, best_outcomes, best_score = child, child_outcomes, child_score
                stagnant = 0
            else:
                best_outcomes, best_score = current_best_outcomes, previous_score
                stagnant += 1
            if stagnant >= input.max_stagnant_revisions:
                terminal = "stagnation_limit"
                break
        selected = best
        passed = all(outcome.status == "verified" for outcome in best_outcomes)
        improved = best_score < anchor_score
        if not passed:
            status = "budget_exhausted" if terminal == "revision_budget_exhausted" else "rejected"
            reasons.append("No revision passed every development requirement.")
        elif incumbent and not improved:
            status = "no_improvement"
            reasons.append("Development evidence did not improve on the current default; holdouts were retained.")
        elif not input.include_holdout:
            status = "needs_holdout"
            reasons.append("The selected revision has development evidence; promotion needs fresh holdouts.")
        else:
            # No revision callback is invoked after exposing the final holdouts.
            completed = evaluate([best], reserve_final=False, include_holdout=True)
            final, status = completed.report, completed.status
            reasons.extend(completed.reasons)
        runtime._revision_provider(input.revision_provider)
        runtime._suite(input.suite)
    except _BudgetExhausted as exc:
        status, reasons, terminal = "budget_exhausted", [str(exc)], "evaluation_budget_exhausted"
    except Exception as exc:
        status, reasons = "blocked", [f"{type(exc).__name__}: {exc}"]
    report = runtime._write(directory / "report.json", {
        "schema_version": "physicsos.rsi_campaign.v1", "status": status, "stop_reason": terminal,
        "request": input.model_dump(mode="json"), "expected_generation": expected_generation,
        "selected_strategy": selected.model_dump(mode="json") if selected else None,
        "revisions": [ref.model_dump(mode="json") for ref in revisions],
        "evaluations": [ref.model_dump(mode="json") for ref in evaluations],
        "provider_evidence": [ref.model_dump(mode="json") for ref in provider_evidence],
        "final_evaluation": final.model_dump(mode="json") if final else None,
        "reserved_kernel_runs": used_runs, "provider_calls": provider_calls,
        "reasons": reasons, "holdout_feedback_used_for_revision": False,
    }, "rsi_campaign")
    return ImproveStrategiesOutput(status=status, selected_strategy=selected, revisions=revisions,
                                   evaluations=evaluations, final_evaluation=final, report=report,
                                   reserved_kernel_runs=used_runs, provider_calls=provider_calls, reasons=reasons)
