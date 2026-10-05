"""Versioned learned-strategy context shared by case agents and runtime tools."""
from __future__ import annotations

import json
from uuid import uuid4

from physicsos.runtime.artifacts import artifact, checked_path, write_json
from physicsos.schemas.case_runtime import PreparedDomain
from physicsos.schemas.common import ArtifactRef
from physicsos.schemas.rsi import BindCaseStrategyInput, BindCaseStrategyOutput


def bind_case_context(runtime, input: BindCaseStrategyInput) -> BindCaseStrategyOutput:
    from physicsos.rsi.runtime import problem_scope
    directory = runtime.workspace / "cases" / input.case_id / "context"
    snapshot = directory / "rsi" / uuid4().hex
    snapshot.mkdir(parents=True, exist_ok=False)
    binding_path = directory / "rsi_binding.json"
    write_json(binding_path, input)
    estimate = runtime.assess(input.assessment)
    status = "needs_strategy"
    strategy = None
    warnings = []
    if estimate.strategy:
        try:
            strategy = runtime._strategy(estimate.strategy)
            if strategy.spec.scope != input.assessment.scope:
                raise ValueError("Learned strategy does not cover the declared case scope.")
            status = "available"
        except (ValueError, OSError) as exc:
            status = "needs_review"
            warnings.append(str(exc))
    pointer = runtime.workspace / "cases" / input.case_id / "geometry" / "prepared_domain.json"
    if pointer.exists():
        try:
            reference = ArtifactRef.model_validate(json.loads(pointer.read_text())["manifest"])
            prepared = PreparedDomain.model_validate_json(checked_path(reference, runtime.workspace).read_text())
            if problem_scope(input.assessment.scope.problem_family, prepared.recipe) != input.assessment.scope:
                raise ValueError("Prepared case physics/dimension/representation differs from the bound RSI scope.")
        except (ValueError, OSError, KeyError) as exc:
            status = "needs_review"
            warnings.append(str(exc))
    applicable = strategy if status == "available" else None
    payload = {
        "schema_version": "physicsos.rsi_case_context.v1", "case_id": input.case_id,
        "status": status, "scope": input.assessment.scope.model_dump(mode="json"),
        "selection_mode": "pinned" if input.assessment.strategy else "current_default",
        "active_generation": estimate.active_generation,
        "strategy": estimate.strategy.model_dump(mode="json") if applicable else None,
        "spec": applicable.spec.model_dump(mode="json") if applicable else None,
        "capability": estimate.model_dump(mode="json"), "warnings": warnings,
    }
    context = runtime._write(snapshot / "context.json", payload, "rsi_case_context")
    lines = ["# Learned runtime strategy", "", f"Status: {status}",
             f"Problem family: {input.assessment.scope.problem_family}",
             f"Active generation: {estimate.active_generation}",
             f"Confidence: {estimate.confidence.value}",
             f"Independent verification outcomes: {estimate.verified}/{estimate.independent_problems}", "",
             f"Capability evidence: {estimate.report.uri}", f"Strategy context: {context.uri}", ""]
    if applicable:
        lines.extend(["Generation guidance", "", applicable.spec.guidance, "",
                      "Preparation / numerical controls", "",
                      json.dumps({"mesh_policy": applicable.spec.mesh_policy.model_dump(mode="json") if applicable.spec.mesh_policy else None,
                                  "repair": applicable.spec.repair, "solver_controls": applicable.spec.solver_controls}, indent=2), "",
                      "Apply this revision with solve_with_rsi_strategy. Preserve the case's physical controls, boundary/quality requirements and independent reference/refinement checks.", ""])
    lines.extend(["Observed revision actions", "", *estimate.suggested_actions,
                  "", estimate.recommendation, "", *warnings])
    guidance_path = snapshot / "guidance.md"
    guidance_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    guidance = artifact(guidance_path, "rsi_case_guidance", runtime.workspace)
    write_json(directory / "rsi_strategy.json", {**payload, "snapshot": context.model_dump(mode="json"), "guidance": guidance.model_dump(mode="json")})
    write_json(directory / "rsi_context_pointer.json", {"context": context.model_dump(mode="json"), "guidance": guidance.model_dump(mode="json")})
    (directory / "rsi_strategy.md").write_text(guidance_path.read_text(), encoding="utf-8")
    return BindCaseStrategyOutput(case_id=input.case_id, status=status, binding=artifact(binding_path, "rsi_case_binding", runtime.workspace),
                                  context=context, guidance=guidance, capability=estimate, warnings=warnings)
