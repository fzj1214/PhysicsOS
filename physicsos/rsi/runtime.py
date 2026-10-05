"""Reuse CaseRuntime to test, adopt and monitor transferable strategy revisions."""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import shutil
import sys
from uuid import uuid4

import numpy as np

from physicsos.config import project_root
from physicsos.paths import resolve_workspace_path
from physicsos.runtime import CaseRuntime
from physicsos.runtime.artifacts import artifact, checked_path, digest, write_json
from physicsos.runtime.execution import load_domain, run_process
from physicsos.runtime.verification import load_run, probes
from physicsos.schemas.case_runtime import ConvergenceStudyInput, ExecuteCaseInput, PrepareDomainInput, PreparedDomain, VerifyCaseInput
from physicsos.schemas.common import ArtifactRef
from physicsos.schemas.rsi import (
    AssessCapabilityInput, BenchmarkCase, BenchmarkOutcome, BenchmarkSuite,
    BenchmarkSuiteInput, CapabilityEstimate, EvaluateStrategiesInput,
    EvaluateStrategiesOutput, PromoteStrategyInput, PromotionOutput,
    RegisterStrategyOutput, RegisterSuiteOutput, RollbackOutput,
    RollbackStrategyInput, SolveWithStrategyInput, SolveWithStrategyOutput,
    StrategyRevision, StrategyScope, StrategySpec,
)
from physicsos.verification.base import AggregateVerificationReport, ConfidenceScore
from physicsos.rsi.store import StrategyStore, canonical_hash, scope_key


class _NeedsInput(ValueError):
    pass


def problem_scope(family: str, prepare: PrepareDomainInput) -> StrategyScope:
    return StrategyScope(problem_family=family, physics_domains=prepare.physics.domains,
                         regime=prepare.physics.regime, dimension=prepare.requirements.dimension,
                         representation=prepare.requirements.representation,
                         domain_side=prepare.requirements.domain_side)


def problem_identity(scope: StrategyScope, case: BenchmarkCase) -> str:
    geometry = case.prepare.geometry
    # Discretization, paths, case IDs and verification tolerances are not new
    # physical problems. A copied benchmark cannot inflate the evidence count.
    shape = {
        "source_kind": geometry.source.kind, "source_sha256": geometry.source.checksum,
        "coordinates": geometry.coordinate_system.model_dump(mode="json"),
        "entities": [{"kind": item.kind, "label": item.label, "metadata": item.metadata} for item in geometry.entities],
        "regions": [{"label": item.label, "kind": item.kind} for item in geometry.regions],
        "boundaries": [{"label": item.label, "kind": item.kind, "role": item.role} for item in geometry.boundaries],
        "transforms": [item.model_dump(mode="json") for item in geometry.transforms],
    }
    requirements = case.prepare.requirements
    def normalized(value):
        if isinstance(value, float):
            return float(format(value, ".15g"))
        if isinstance(value, list):
            return [normalized(item) for item in value]
        if isinstance(value, dict):
            return {key: normalized(item) for key, item in value.items()}
        return value
    return canonical_hash(normalized({"scope": scope.model_dump(mode="json"), "geometry": shape,
                           "physics": case.prepare.physics.model_dump(mode="json"),
                           "boundary_roles": requirements.boundary_roles,
                           "whole_boundary_role": requirements.whole_boundary_role,
                           "required_boundary_roles": sorted(requirements.required_boundary_roles),
                           "bounds_min": requirements.bounds_min, "bounds_max": requirements.bounds_max,
                           "controls": {key: value for key, value in case.controls.items() if key not in case.tunable_controls}}))


class RSIRuntime:
    def __init__(self, workspace: str | Path | None = None):
        self.workspace = Path(workspace or project_root()).resolve()
        self.runtime = CaseRuntime(self.workspace)
        self.store = StrategyStore(self.workspace)

    def _directory(self, category: str) -> Path:
        path = self.store.root / category / uuid4().hex
        path.mkdir(parents=True, exist_ok=False)
        return path

    def _write(self, path: Path, payload, kind: str) -> ArtifactRef:
        write_json(path, payload)
        return artifact(path, kind, self.workspace)

    def _copy_file(self, source: Path, target: Path, artifacts: dict, name: str) -> str:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        reference = artifact(target, "rsi_frozen_input", self.workspace)
        artifacts[name] = reference
        return reference.uri

    def _copy_python(self, uri: str, target: Path, artifacts: dict, prefix: str) -> str:
        source = resolve_workspace_path(uri, workspace=self.workspace)
        if not source.is_file() or source.suffix != ".py":
            raise ValueError(f"A Python implementation file is required: {uri}")
        for helper in source.parent.glob("*.py"):
            self._copy_file(helper, target / helper.name, artifacts, f"{prefix}:{helper.name}")
        return artifacts[f"{prefix}:{source.name}"].uri

    def _freeze_case(self, case: BenchmarkCase, directory: Path, artifacts: dict) -> tuple[BenchmarkCase, str]:
        frozen = case.model_copy(deep=True)
        frozen.prepare.geometry.encodings = []
        source = frozen.prepare.geometry.source
        if source.uri:
            original = resolve_workspace_path(source.uri, workspace=self.workspace)
            if source.checksum and digest(original) != source.checksum.removeprefix("sha256:"):
                raise ValueError("Benchmark source geometry changed before registration.")
            source.uri = self._copy_file(original, directory / ("geometry" + original.suffix), artifacts, f"{case.id}:geometry")
            source.checksum = artifacts[f"{case.id}:geometry"].checksum
        default_kernel = f"cases/{case.prepare.case_id}/taps/kernel.py"
        if case.kernel_uri or resolve_workspace_path(default_kernel, workspace=self.workspace).is_file():
            frozen.kernel_uri = self._copy_python(case.kernel_uri or default_kernel, directory / "kernel", artifacts, f"{case.id}:kernel")
        frozen.reference_uri = self._copy_python(case.reference_uri, directory / "reference", artifacts, f"{case.id}:reference")
        resources = directory / "resources"
        for section in ("problem", "context", "references", "materials", "pseudopotentials"):
            original = self.workspace / "cases" / case.prepare.case_id / section
            if original.is_dir():
                for item in original.rglob("*"):
                    if item.is_file():
                        self._copy_file(item, resources / section / item.relative_to(original), artifacts, f"{case.id}:resources:{section}/{item.relative_to(original).as_posix()}")
        return frozen, str(resources)

    def register_strategy(self, input: StrategySpec) -> RegisterStrategyOutput:
        directory = self._directory("strategies")
        spec = input.model_copy(deep=True)
        artifacts = {}
        if spec.parent:
            parent = self._strategy(spec.parent)
            if parent.spec.scope != spec.scope:
                raise ValueError("A strategy revision cannot change its parent's scope.")
        if spec.kernel_uri:
            spec.kernel_uri = self._copy_python(spec.kernel_uri, directory / "kernel", artifacts, "kernel")
        if spec.builder_uri:
            spec.builder_uri = self._copy_python(spec.builder_uri, directory / "builder", artifacts, "builder")
        revision = StrategyRevision(id=directory.name, spec=spec, artifacts=artifacts)
        reference = self._write(directory / "manifest.json", revision, "rsi_strategy")
        return RegisterStrategyOutput(strategy=revision, manifest=reference)

    def register_revision_provider(self, input):
        from physicsos.schemas.rsi import RevisionProvider, RegisterRevisionProviderOutput
        directory = self._directory("revision_providers")
        artifacts = {}
        spec = input.model_copy(deep=True)
        spec.python_uri = self._copy_python(spec.python_uri, directory / "provider", artifacts, "provider")
        provider = RevisionProvider(id=directory.name, spec=spec, artifacts=artifacts)
        reference = self._write(directory / "manifest.json", provider, "rsi_revision_provider")
        return RegisterRevisionProviderOutput(provider=provider, manifest=reference)

    def register_suite(self, input: BenchmarkSuiteInput) -> RegisterSuiteOutput:
        directory = self._directory("suites")
        spec = input.model_copy(deep=True)
        artifacts, resources, identities = {}, {}, {}
        for index, case in enumerate(spec.benchmarks):
            if problem_scope(spec.scope.problem_family, case.prepare) != spec.scope:
                raise ValueError(f"Benchmark {case.id} is outside the suite scope.")
            if case.convergence:
                ConvergenceStudyInput(case_id=case.prepare.case_id, prepared_domain=ArtifactRef(uri="unused", kind="unused"), axis="mesh" if spec.scope.representation == "mesh" else "grid", **case.convergence.model_dump())
            frozen, resource_dir = self._freeze_case(case, directory / case.id, artifacts)
            spec.benchmarks[index] = frozen
            resources[case.id] = resource_dir
            identities[case.id] = problem_identity(spec.scope, frozen)
        if len(set(identities.values())) != len(identities):
            raise ValueError("Development and holdout benchmarks must be distinct physical problems; renamed copies are not independent evidence.")
        for split, minimum in (("development", spec.policy.min_development_cases), ("holdout", spec.policy.min_holdout_cases)):
            if sum(case.split == split for case in spec.benchmarks) < minimum:
                raise ValueError(f"The suite needs at least {minimum} distinct {split} problems.")
        if spec.policy.require_convergence and any(case.convergence is None for case in spec.benchmarks):
            raise ValueError("This promotion policy requires an actual refinement study for every benchmark.")
        if spec.policy.require_convergence and len({case.convergence.expected_order for case in spec.benchmarks}) != 1:
            raise ValueError("Freeze one declared method order within a strategy scope.")
        holdout = sorted(identities[case.id] for case in spec.benchmarks if case.split == "holdout")
        suite = BenchmarkSuite(id=directory.name, spec=spec, problem_identities=identities,
                               holdout_fingerprint=canonical_hash({"scope": scope_key(spec.scope), "problems": holdout}),
                               artifacts=artifacts, resource_directories=resources)
        reference = self._write(directory / "manifest.json", suite, "rsi_suite")
        return RegisterSuiteOutput(suite=suite, manifest=reference)

    def _load(self, reference: ArtifactRef, model, kind: str, category: str):
        path = checked_path(reference, self.workspace)
        if reference.kind != kind or not path.is_relative_to(self.store.root / category):
            raise ValueError(f"Use a registered {kind} artifact from this workspace.")
        value = model.model_validate_json(path.read_text())
        for item in value.artifacts.values():
            checked_path(item, self.workspace)
        return value

    def _strategy(self, reference: ArtifactRef) -> StrategyRevision:
        return self._load(reference, StrategyRevision, "rsi_strategy", "strategies")

    def _suite(self, reference: ArtifactRef) -> BenchmarkSuite:
        return self._load(reference, BenchmarkSuite, "rsi_suite", "suites")

    def _revision_provider(self, reference: ArtifactRef):
        from physicsos.schemas.rsi import RevisionProvider
        return self._load(reference, RevisionProvider, "rsi_revision_provider", "revision_providers")

    def _apply(self, strategy: StrategyRevision, benchmark: BenchmarkCase, case_id: str):
        if problem_scope(strategy.spec.scope.problem_family, benchmark.prepare) != strategy.spec.scope:
            raise _NeedsInput("Strategy scope does not match this problem.")
        unknown_controls = set(strategy.spec.solver_controls) - set(benchmark.tunable_controls)
        if unknown_controls:
            raise _NeedsInput("Strategy cannot override physical or undeclared controls: " + ", ".join(sorted(unknown_controls)))
        prepared = benchmark.prepare.model_copy(deep=True)
        prepared.case_id = case_id
        if strategy.spec.mesh_policy:
            prepared.requirements.mesh_policy = strategy.spec.mesh_policy.model_copy(deep=True)
            prepared.force_remesh = True
        if strategy.spec.repair:
            prepared.repair = strategy.spec.repair
        if strategy.spec.repair_options:
            for name in ("max_relative_surface_distance", "max_aspect_ratio_p95", "max_skewness"):
                if getattr(strategy.spec.repair_options, name) > getattr(prepared.repair_options, name):
                    raise _NeedsInput("Strategy cannot weaken the geometry repair acceptance limit: " + name)
            if strategy.spec.repair_options.validation_samples < prepared.repair_options.validation_samples:
                raise _NeedsInput("Strategy cannot reduce independent geometry validation samples.")
            prepared.repair_options = strategy.spec.repair_options.model_copy(deep=True)
        controls = {**benchmark.controls, **strategy.spec.solver_controls}
        ExecuteCaseInput(case_id=case_id, prepared_domain=ArtifactRef(uri="unused", kind="unused"), controls=controls)
        return prepared, controls

    def _kernel(self, strategy: StrategyRevision, benchmark: BenchmarkCase, prepared, controls: dict, directory: Path, timeout: int) -> tuple[str, ArtifactRef | None]:
        if not strategy.spec.builder_uri:
            return strategy.spec.kernel_uri or benchmark.kernel_uri, None
        generation = directory / "generation"
        generation.mkdir(parents=True)
        source = resolve_workspace_path(strategy.spec.builder_uri, workspace=self.workspace)
        for helper in source.parent.glob("*.py"):
            shutil.copyfile(helper, generation / helper.name)
        config = {"problem": benchmark.prepare.model_dump(mode="json"), "guidance": strategy.spec.guidance,
                  "controls": controls, "domain": prepared.domain.model_dump(mode="json"),
                  "domain_artifacts": {name: str(checked_path(item, self.workspace)) for name, item in prepared.domain.artifacts.items()}}
        write_json(generation / "context.json", config)
        before = {str(path): digest(path) for path in [*generation.glob("*.py"), generation / "context.json"]}
        command = [sys.executable, str(Path(__file__).parents[1] / "runtime" / "kernel_worker.py"), "--kernel", str(generation / source.name), "--entrypoint", "build_case_kernel", "--config", str(generation / "context.json"), "--response", str(generation / "response.json")]
        process = run_process(command, generation, timeout)
        log = self._write(generation / "execution_log.json", process, "rsi_generation_log")
        response = json.loads((generation / "response.json").read_text()) if (generation / "response.json").exists() else {}
        if process["returncode"] != 0 or not response.get("ok"):
            raise ValueError("Kernel builder failed: " + str(response.get("error") or process["stderr"][-1000:]))
        if any(not Path(path).is_file() or digest(Path(path)) != checksum for path, checksum in before.items()):
            raise ValueError("Kernel builder changed its frozen implementation.")
        code = response.get("result", {}).get("kernel_source")
        if not isinstance(code, str) or not any(isinstance(node, ast.FunctionDef) and node.name == "run_case" for node in ast.parse(code).body):
            raise ValueError("build_case_kernel(config) must return kernel_source with run_case(config).")
        kernel = generation / "generated_kernel.py"
        kernel.write_text(code, encoding="utf-8")
        return str(kernel), log

    def _run_benchmark(self, strategy_ref: ArtifactRef, benchmark: BenchmarkCase, identity: str, split: str, directory: Path, probe_file: Path | None, timeout: int, resources: str | None = None, production_case: bool = False, require_convergence: bool = False) -> tuple[BenchmarkOutcome, ArtifactRef]:
        case_id = benchmark.prepare.case_id if production_case else "rsi-" + uuid4().hex
        outcome = BenchmarkOutcome(benchmark_id=benchmark.id, split=split, problem_identity=identity,
                                   strategy=strategy_ref, status="uncertain", stage="preparation")
        try:
            strategy = self._strategy(strategy_ref)
            request, controls = self._apply(strategy, benchmark, case_id)
            request.repair_options.timeout_seconds = min(request.repair_options.timeout_seconds, timeout)
            if resources and Path(resources).is_dir() and not production_case:
                shutil.copytree(resources, self.workspace / "cases" / case_id, dirs_exist_ok=True)
            prepared = self.runtime.prepare(request)
            outcome.evidence["prepared_domain"] = prepared.manifest
            if prepared.domain.status != "ready":
                outcome.status = "failed" if prepared.domain.status == "failed" else "uncertain"
                outcome.diagnostics = prepared.domain.required_actions
                outcome.suggested_actions = prepared.domain.required_actions
            else:
                outcome.stage = "generation"
                kernel, generation_log = self._kernel(strategy, benchmark, prepared, controls, directory, timeout)
                if generation_log:
                    outcome.evidence["generation_log"] = generation_log
                    outcome.evidence["generation_request"] = artifact(directory / "generation" / "context.json", "rsi_generation_request", self.workspace)
                    outcome.evidence["generation_response"] = artifact(directory / "generation" / "response.json", "rsi_generation_response", self.workspace)
                    outcome.evidence["generated_kernel"] = artifact(Path(kernel), "rsi_generated_kernel", self.workspace)
                outcome.stage = "execution"
                executed = self.runtime.execute(ExecuteCaseInput(case_id=case_id, prepared_domain=prepared.manifest,
                                                                kernel_uri=kernel, controls=controls,
                                                                field_name=benchmark.field_name, timeout_seconds=timeout))
                outcome.evidence["run"] = executed.manifest
                outcome.metrics["wall_time_seconds"] = executed.run.result.runtime.wall_time_seconds or 0.
                comparison = None
                if executed.run.result.status == "success" and probe_file is not None:
                    if not probe_file.exists():
                        points, weights = probes(executed.run, self.workspace, benchmark.max_samples)
                        probe_file.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(probe_file, points=points, weights=weights)
                    comparison = artifact(probe_file, "rsi_comparison_probes", self.workspace)
                    outcome.evidence["comparison_probes"] = comparison
                verified = self.runtime.verify(VerifyCaseInput(run_manifest=executed.manifest, reference_uri=benchmark.reference_uri,
                                                              reference_function=benchmark.reference_function, comparison_probes=comparison,
                                                              relative_tolerance=benchmark.relative_tolerance, absolute_tolerance=benchmark.absolute_tolerance,
                                                              max_samples=benchmark.max_samples, timeout_seconds=timeout))
                outcome.evidence["verification"] = verified.artifact
                outcome.status = verified.report.overall_status.value
                check = verified.report.individual_results["IndependentFieldReference"]
                outcome.metrics.update(check.metrics)
                if "weighted_sample_rms_error" in check.metrics:
                    outcome.metrics["error_ratio"] = check.metrics["weighted_sample_rms_error"] / max(benchmark.absolute_tolerance, benchmark.relative_tolerance * check.metrics["reference_rms"])
                if executed.run.result.status != "success":
                    outcome.diagnostics = executed.run.errors
                    outcome.suggested_actions = ["revise_case_kernel_or_solver_controls"]
                else:
                    outcome.stage = "verification"
                    if outcome.status != "verified":
                        outcome.diagnostics = [check.message]
                        outcome.suggested_actions = ["inspect_independent_field_error_and_boundary_application"] if outcome.status == "failed" else ["provide_missing_verification_or_sampling_evidence"]
                    elif benchmark.convergence:
                        outcome.stage = "convergence"
                        study = self.runtime.convergence(ConvergenceStudyInput(case_id=case_id, prepared_domain=prepared.manifest,
                                                                            kernel_uri=kernel, reference_uri=benchmark.reference_uri,
                                                                            reference_function=benchmark.reference_function, controls=controls,
                                                                            field_name=benchmark.field_name, max_samples=benchmark.max_samples,
                                                                            axis="mesh" if prepared.domain.requirements.representation == "mesh" else "grid",
                                                                            timeout_seconds=timeout, **benchmark.convergence.model_dump()))
                        outcome.evidence["convergence"] = study.report
                        outcome.status = study.status
                        report = json.loads(checked_path(study.report, self.workspace).read_text())
                        if report["observed_order"] is not None:
                            outcome.metrics["observed_order"] = report["observed_order"]
                        if outcome.status != "verified":
                            outcome.diagnostics = [report["message"]]
                            outcome.suggested_actions = ["revise_discretization_or_method_and_rerun_refinement"]
                    elif require_convergence:
                        outcome.stage, outcome.status = "convergence", "uncertain"
                        outcome.diagnostics = ["The active promotion policy requires an actual refinement study."]
                        outcome.suggested_actions = ["supply_convergence_requirements"]
                    if outcome.status == "verified":
                        outcome.stage = "complete"
            self._strategy(strategy_ref)
        except _NeedsInput as exc:
            outcome.status = "uncertain"
            outcome.diagnostics = [str(exc)]
            outcome.suggested_actions = ["bind_compatible_problem_and_numerical_control_contract"]
        except Exception as exc:
            outcome.status = "failed"
            outcome.diagnostics = [f"{type(exc).__name__}: {exc}"]
            outcome.suggested_actions = ["resolve_" + outcome.stage + "_failure"]
        for name in ("execution_log", "response"):
            path = directory / "generation" / (name + ".json")
            if path.is_file():
                key = "generation_log" if name == "execution_log" else "generation_response"
                outcome.evidence.setdefault(key, artifact(path, "rsi_" + key, self.workspace))
        reference = self._write(directory / "outcome.json", outcome, "rsi_outcome")
        return outcome, reference

    def _valid_outcome(self, reference: ArtifactRef) -> BenchmarkOutcome:
        outcome = BenchmarkOutcome.model_validate_json(checked_path(reference, self.workspace).read_text())
        self._strategy(outcome.strategy)
        for item in outcome.evidence.values():
            checked_path(item, self.workspace)
        run = load_run(outcome.evidence["run"], self.workspace) if "run" in outcome.evidence else None
        if run and run.domain_manifest != outcome.evidence.get("prepared_domain"):
            raise ValueError("Run evidence belongs to a different prepared domain.")
        verification = None
        if "verification" in outcome.evidence:
            verification = AggregateVerificationReport.model_validate_json(checked_path(outcome.evidence["verification"], self.workspace).read_text())
            if run is None or verification.problem_id != run.case_id or verification.result_id != run.result.id:
                raise ValueError("Verification evidence belongs to a different run.")
            check = verification.individual_results["IndependentFieldReference"]
            if any(check.details.get(key) != value for key, value in {"run_id": run.id, "domain_id": run.domain_id, "kernel_sha256": run.kernel_sha256}.items()):
                raise ValueError("Verification identity does not match the executed kernel/domain.")
        if outcome.status == "verified":
            load_domain(outcome.evidence["prepared_domain"], self.workspace)
            if run is None or run.result.status != "success" or verification is None or verification.overall_status.value != "verified" or outcome.stage != "complete" or "error_ratio" not in outcome.metrics:
                raise ValueError("A successful strategy needs actual independent field evidence.")
            check = verification.individual_results["IndependentFieldReference"]
            ratio = check.metrics["weighted_sample_rms_error"] / max(check.details["absolute_tolerance"], check.details["relative_tolerance"] * check.metrics["reference_rms"])
            if outcome.metrics["error_ratio"] != ratio or not math.isfinite(ratio):
                raise ValueError("Strategy scoring does not match independent verification metrics.")
            if "comparison_probes" in outcome.evidence and check.details.get("comparison_probes") != outcome.evidence["comparison_probes"].model_dump(mode="json"):
                raise ValueError("The verification did not use the frozen comparison probes.")
            if "convergence" in outcome.evidence:
                study = json.loads(checked_path(outcome.evidence["convergence"], self.workspace).read_text())
                if study["status"] != "verified" or not study["generated_from_actual_runs"] or len(study["rows"]) < 3:
                    raise ValueError("Actual refinement evidence is missing or no longer passes.")
                for run_ref in study["runs"]:
                    run = load_run(ArtifactRef.model_validate(run_ref), self.workspace)
                    load_domain(run.domain_manifest, self.workspace)
        return outcome

    def _check_benchmark_evidence(self, outcome: BenchmarkOutcome, benchmark: BenchmarkCase, scope: StrategyScope):
        """Bind physical inputs, controls, implementations and criteria to a case."""
        if "prepared_domain" not in outcome.evidence:
            if outcome.status == "verified":
                raise ValueError("Benchmark has no prepared physical-input evidence.")
            return
        domain = PreparedDomain.model_validate_json(checked_path(outcome.evidence["prepared_domain"], self.workspace).read_text())
        request_ref = domain.artifacts.get("preparation_request")
        if request_ref is None:
            raise ValueError("Qualification needs the original preparation request; reevaluate this benchmark.")
        actual = PrepareDomainInput.model_validate_json(checked_path(request_ref, self.workspace).read_text())
        revision = self._strategy(outcome.strategy)
        if revision.spec.scope != scope:
            raise ValueError("Benchmark strategy scope changed.")
        expected, controls = self._apply(revision, benchmark, domain.case_id)
        expected.repair_options.timeout_seconds = actual.repair_options.timeout_seconds
        if actual != expected:
            raise ValueError("The prepared physical problem differs from the frozen benchmark request.")
        if "run" not in outcome.evidence:
            return
        run = load_run(outcome.evidence["run"], self.workspace)
        if run.controls != controls or run.field_name != benchmark.field_name:
            raise ValueError("Executed controls or field differ from the frozen benchmark.")
        source = revision.spec.kernel_uri or benchmark.kernel_uri
        if revision.spec.builder_uri and run.result.status == "success":
            kernel = checked_path(outcome.evidence["generated_kernel"], self.workspace)
            response = json.loads(checked_path(outcome.evidence["generation_response"], self.workspace).read_text())
            generation = json.loads(checked_path(outcome.evidence["generation_request"], self.workspace).read_text())
            if not response.get("ok") or response["result"].get("kernel_source") != kernel.read_text() or generation["problem"] != benchmark.prepare.model_dump(mode="json") or generation["controls"] != controls:
                raise ValueError("Generated kernel is not bound to this benchmark's generation evidence.")
        elif source:
            kernel = resolve_workspace_path(source, workspace=self.workspace)
        elif run.result.status == "success":
            raise ValueError("The benchmark has no frozen implementation provider.")
        else:
            return
        if run.result.status == "success" and run.kernel_sha256 != digest(kernel):
            raise ValueError("Executed implementation differs from the frozen strategy/provider output.")
        if outcome.status == "verified":
            verification = AggregateVerificationReport.model_validate_json(checked_path(outcome.evidence["verification"], self.workspace).read_text())
            details = verification.individual_results["IndependentFieldReference"].details
            if details["reference_sha256"] != digest(resolve_workspace_path(benchmark.reference_uri, workspace=self.workspace)):
                raise ValueError("Verification used a different independent reference.")
            if details.get("reference_function") != benchmark.reference_function or details.get("field_name") != benchmark.field_name or details.get("max_samples") != benchmark.max_samples:
                raise ValueError("Reference function, field or sampling budget differs from the benchmark.")
            if benchmark.convergence:
                study = json.loads(checked_path(outcome.evidence["convergence"], self.workspace).read_text())
                if any(study.get(key) != value for key, value in benchmark.convergence.model_dump(exclude={"refinements"}).items()):
                    raise ValueError("Refinement acceptance criteria differ from the frozen benchmark.")
                if [row["requested_refinement"] for row in study["rows"]] != benchmark.convergence.refinements:
                    raise ValueError("Actual refinement requests differ from the benchmark.")

    def _observe(self, scope: StrategyScope, outcome: BenchmarkOutcome, reference: ArtifactRef, activation_id: str | None = None, assessment: ArtifactRef | None = None):
        self.store.observe(scope, outcome.strategy, outcome.problem_identity, outcome.split, reference, activation_id, assessment)

    def evaluate(self, input: EvaluateStrategiesInput) -> EvaluateStrategiesOutput:
        from physicsos.rsi.evaluation import evaluate_strategies
        return evaluate_strategies(self, input)

    def improve(self, input):
        from physicsos.rsi.campaign import improve_strategies
        return improve_strategies(self, input)

    def promote(self, input: PromoteStrategyInput) -> PromotionOutput:
        from physicsos.rsi.evaluation import validate_promotion
        directory = self._directory("transitions")
        result = {"status": "blocked", "reasons": []}
        try:
            report, suite, selected = validate_promotion(self, input.evaluation)
            policy = suite.spec.policy.model_dump(mode="json")
            policy.update(relative_tolerance=min(case.relative_tolerance for case in suite.spec.benchmarks),
                          absolute_tolerance=min(case.absolute_tolerance for case in suite.spec.benchmarks))
            if policy["require_convergence"]:
                policy.update(expected_order=suite.spec.benchmarks[0].convergence.expected_order,
                              rate_tolerance=min(case.convergence.rate_tolerance for case in suite.spec.benchmarks),
                              error_tolerance=min(case.convergence.error_tolerance for case in suite.spec.benchmarks))
            transition = self.store.promote(suite.spec.scope, report["expected_generation"], selected,
                                            input.evaluation, policy)
            result.update(status="promoted", **transition)
        except Exception as exc:
            result["reasons"] = [str(exc)]
        reference = self._write(directory / "promotion.json", {**result, "evaluation": input.evaluation.model_dump(mode="json")}, "rsi_promotion")
        return PromotionOutput(**result, report=reference)

    def rollback(self, input: RollbackStrategyInput) -> RollbackOutput:
        from physicsos.rsi.evaluation import validate_promotion
        directory = self._directory("transitions")
        status, previous, generation = "blocked", None, None
        notes = []
        try:
            active = self.store.active(input.scope)
            record = self.store.activation(active["activation_id"]) if active["activation_id"] else None
            if record is None:
                raise ValueError("There is no active promotion to roll back.")
            if record["previous_strategy"]:
                candidate = ArtifactRef.model_validate_json(record["previous_strategy"])
                try:
                    self._strategy(candidate)
                    predecessor = self.store.activation(record["previous_activation"]) if record["previous_activation"] else None
                    if predecessor is None or predecessor["status"] == "rolled_back":
                        raise ValueError("The predecessor has no current qualification record.")
                    _, suite, qualified = validate_promotion(self, ArtifactRef.model_validate_json(predecessor["evaluation"]))
                    if qualified != candidate or suite.spec.scope != input.scope:
                        raise ValueError("Predecessor qualification belongs to a different strategy/scope.")
                    previous = candidate
                except (OSError, ValueError, KeyError) as exc:
                    notes.append("Previous strategy evidence is invalid; cleared the default: " + str(exc))
            transition = self.store.rollback(input.scope, input.expected_generation, previous)
            generation, status = transition["generation"], "rolled_back"
        except Exception as exc:
            notes.append(str(exc))
            current = self.store.active(input.scope)
            previous, generation = current["strategy"], current["generation"]
        reference = self._write(directory / "rollback.json", {"status": status, "scope": input.scope.model_dump(mode="json"),
                                                               "reason": input.reason, "notes": notes, "generation": generation,
                                                               "active_strategy": previous.model_dump(mode="json") if previous else None}, "rsi_rollback")
        return RollbackOutput(status=status, active_strategy=previous, generation=generation, report=reference)

    def assess(self, input: AssessCapabilityInput) -> CapabilityEstimate:
        from physicsos.rsi.calibration import assess_capability
        return assess_capability(self, input)

    def bind_context(self, input):
        from physicsos.rsi.context import bind_case_context
        return bind_case_context(self, input)

    def solve(self, input: SolveWithStrategyInput) -> SolveWithStrategyOutput:
        directory = self._directory("executions")
        scope = problem_scope(input.problem_family, input.prepare)
        active = self.store.active(scope)
        selected = input.strategy or active["strategy"]
        if selected is None:
            report = self._write(directory / "report.json", {"status": "needs_strategy", "scope": scope.model_dump(mode="json"), "message": "No promoted strategy exists for this exact scope; author and evaluate a candidate first."}, "rsi_execution")
            return SolveWithStrategyOutput(status="needs_strategy", report=report)
        try:
            revision = self._strategy(selected)
            if revision.spec.scope != scope:
                raise ValueError("The selected strategy does not cover this problem scope.")
        except (ValueError, OSError) as exc:
            rollback = None
            if active["strategy"] and selected.checksum == active["strategy"].checksum:
                rollback = self.rollback(RollbackStrategyInput(scope=scope, expected_generation=active["generation"], reason="Active strategy evidence is invalid: " + str(exc)))
            report = self._write(directory / "report.json", {"status": "needs_strategy", "message": str(exc),
                                                            "rollback": rollback.model_dump(mode="json") if rollback else None}, "rsi_execution")
            return SolveWithStrategyOutput(status="needs_strategy", rollback=rollback, report=report)
        refinement_contract = input.convergence.model_dump(exclude={"refinements"}) if input.convergence else {}
        estimate = self.assess(AssessCapabilityInput(scope=scope, strategy=selected,
                                                   relative_tolerance=input.relative_tolerance, absolute_tolerance=input.absolute_tolerance,
                                                   require_convergence=True if input.convergence else None,
                                                   **refinement_contract))
        artifacts = {}
        benchmark = BenchmarkCase(id=input.prepare.case_id, split="development", prepare=input.prepare,
                                  controls=input.controls, tunable_controls=input.tunable_controls,
                                  kernel_uri=input.kernel_uri, reference_uri=input.reference_uri,
                                  reference_function=input.reference_function, field_name=input.field_name,
                                  relative_tolerance=input.relative_tolerance, absolute_tolerance=input.absolute_tolerance,
                                  max_samples=input.max_samples, convergence=input.convergence)
        try:
            benchmark, _ = self._freeze_case(benchmark, directory / "inputs", artifacts)
        except (ValueError, OSError) as exc:
            report = self._write(directory / "report.json", {"status": "uncertain", "stage": "input_snapshot", "message": str(exc)}, "rsi_execution")
            return SolveWithStrategyOutput(status="uncertain", report=report)
        identity = problem_identity(scope, benchmark)
        activation_id = active["activation_id"] if active["strategy"] and selected.checksum == active["strategy"].checksum else None
        outcome, reference = self._run_benchmark(selected, benchmark, identity, "production", directory / "attempt", None,
                                                input.timeout_seconds, production_case=True,
                                                require_convergence=bool(activation_id and json.loads(self.store.activation(activation_id)["policy"])["require_convergence"]))
        self._observe(scope, outcome, reference, activation_id, estimate.report)
        rollback = None
        if activation_id and outcome.status == "failed":
            record = self.store.activation(activation_id)
            latest = {}
            for observation in self.store.observations(scope, selected):
                if observation["split"] == "production" and observation["activation_id"] == activation_id:
                    try:
                        evidence = self._valid_outcome(ArtifactRef.model_validate_json(observation["evidence"]))
                        latest[observation["identity"]] = evidence.status
                    except (ValueError, OSError, KeyError):
                        latest[observation["identity"]] = "uncertain"
            failures = sum(status == "failed" for status in latest.values())
            if failures >= json.loads(record["policy"])["rollback_failure_limit"]:
                rollback = self.rollback(RollbackStrategyInput(scope=scope, expected_generation=active["generation"], reason=f"{failures} distinct production problems failed under activation {activation_id}."))
        report = self._write(directory / "report.json", {"status": outcome.status, "outcome": reference.model_dump(mode="json"),
                                                        "assessment": estimate.report.model_dump(mode="json"),
                                                        "frozen_inputs": {name: item.model_dump(mode="json") for name, item in artifacts.items()},
                                                        "activation_id": activation_id, "rollback": rollback.model_dump(mode="json") if rollback else None}, "rsi_execution")
        return SolveWithStrategyOutput(status=outcome.status, outcome=outcome, rollback=rollback, report=report)
