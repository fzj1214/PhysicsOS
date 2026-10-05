from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from physicsos.config import project_root
from physicsos.paths import resolve_workspace_path
from physicsos.schemas.case_runtime import ConvergenceStudyInput, ExecuteCaseInput, PrepareDomainInput, VerifyCaseInput
from physicsos.schemas.case_runtime import PreparedDomain, SearchRuntimeHistoryInput, SearchRuntimeHistoryOutput
from physicsos.schemas.common import ArtifactRef
from physicsos.runtime.artifacts import checked_path, write_json
from physicsos.runtime.convergence import convergence_study
from physicsos.runtime.domain import prepare_domain
from physicsos.runtime.execution import execute_case
from physicsos.runtime.execution import load_domain
from physicsos.runtime.verification import verify_case


class CaseRuntime:
    """One preparation/execution path shared by agents and verification reruns."""

    def __init__(self, workspace: str | Path | None = None):
        self.workspace = Path(workspace or project_root()).resolve()

    def _event(self, case_id: str, stage: str, payload: dict):
        path = self.workspace / "cases" / case_id / "runtime_events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event_id": uuid4().hex, "stage": stage, **payload}, ensure_ascii=False, allow_nan=False) + "\n")

    def prepare(self, input: PrepareDomainInput):
        result = prepare_domain(input, self.workspace)
        geometry_dir = self.workspace / "cases" / input.case_id / "geometry"
        write_json(geometry_dir / "prepared_domain.json", {"domain_id": result.domain.id, "status": result.domain.status, "manifest": result.manifest.model_dump(mode="json"), "required_actions": result.domain.required_actions})
        handoff = geometry_dir / "taps_geometry_handoff.md"
        handoff.write_text("# Prepared simulation domain\n\n" + f"Status: {result.domain.status}\nDomain manifest: {result.manifest.uri}\nRepresentation: {result.domain.requirements.representation}\n\n" + "Kernel entrypoint: `run_case(config)`. Consume `config['domain_artifacts']`, reassemble for its discretization, and write only to `config['output_dir']`.\n\n" + "\n".join(f"- {name}: {item.uri}" for name, item in result.domain.artifacts.items()), encoding="utf-8")
        self._event(input.case_id, "domain_preparation", {"domain_id": result.domain.id, "status": result.domain.status, "manifest": result.manifest.model_dump(mode="json"), "required_actions": result.domain.required_actions})
        return result

    def execute(self, input: ExecuteCaseInput):
        result = execute_case(input, self.workspace)
        self._event(input.case_id, "kernel_execution", {"run_id": result.run.id, "domain_id": result.run.domain_id, "status": result.run.result.status, "manifest": result.manifest.model_dump(mode="json"), "errors": result.run.errors})
        return result

    def verify(self, input: VerifyCaseInput):
        result = verify_case(input, self.workspace)
        path = resolve_workspace_path(input.run_manifest.uri, workspace=self.workspace)
        run = json.loads(path.read_text())
        self._event(run["case_id"], "independent_verification", {"run_id": run["id"], "status": result.report.overall_status.value, "report": result.artifact.model_dump(mode="json")})
        # Append attempts, including failures/unknowns; never replace history by case id.
        memory = self.workspace / "data" / "runtime_attempts.jsonl"
        memory.parent.mkdir(parents=True, exist_ok=True)
        with memory.open("a", encoding="utf-8") as stream:
            try:
                domain = PreparedDomain.model_validate_json(checked_path(ArtifactRef.model_validate(run["domain_manifest"]), self.workspace).read_text())
                features = {"physics_domains": domain.recipe.physics.domains, "regime": domain.recipe.physics.regime, "dimension": domain.requirements.dimension, "representation": domain.requirements.representation}
            except (OSError, ValueError):
                features = {}
            stream.write(json.dumps({"case_id": run["case_id"], "run_id": run["id"], "domain_id": run["domain_id"], "kernel_sha256": run["kernel_sha256"], "controls": run["controls"], "features": features, "verification": result.report.model_dump(mode="json"), "verification_artifact": result.artifact.model_dump(mode="json"), "run_manifest": input.run_manifest.model_dump(mode="json")}, ensure_ascii=False) + "\n")
        return result

    def history(self, input: SearchRuntimeHistoryInput):
        path = self.workspace / "data" / "runtime_attempts.jsonl"
        attempts = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                features = record.get("features", {})
                if input.case_id and record["case_id"] != input.case_id:
                    continue
                if input.physics_domain and input.physics_domain not in features.get("physics_domains", []):
                    continue
                if any(value is not None and features.get(key) != value for key, value in {"regime": input.regime, "dimension": input.dimension, "representation": input.representation}.items()):
                    continue
                attempts.append(record)
        # Re-verifying one run cannot inflate its historical success count.
        unique = {record["run_id"]: record for record in attempts}
        counts = {status: sum(record["verification"]["overall_status"] == status for record in unique.values()) for status in ("verified", "failed", "uncertain")}
        selected = [record for record in unique.values() if input.status is None or record["verification"]["overall_status"] == input.status]
        return SearchRuntimeHistoryOutput(attempts=selected[-input.top_k:][::-1], matched_attempts=len(unique), outcome_counts=counts)

    def convergence(self, input: ConvergenceStudyInput):
        result = convergence_study(input, self.workspace)
        self._event(input.case_id, "convergence_study", {"status": result.status, "report": result.report.model_dump(mode="json"), "runs": [item.model_dump(mode="json") for item in result.runs]})
        return result
