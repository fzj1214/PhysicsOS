from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Literal

import numpy as np
from pydantic import Field

from physicsos.config import runtime_paths
from physicsos.paths import resolve_workspace_path, to_agent_path
from physicsos.schemas.common import ArtifactRef, Provenance, StrictBaseModel
from physicsos.schemas.problem import PhysicsProblem
from physicsos.schemas.solver import SolverResult
from physicsos.schemas.taps import (
    TAPSGeometrySeparabilityAssessment,
    TAPSProblem,
    TAPSResidualReport,
    TAPSRuntimeExtensionSpec,
    TAPSSupportScore,
)


def _workspace() -> Path:
    return runtime_paths().workspace


def _case_dir(case_id: str) -> Path:
    return _workspace() / "cases" / case_id


def _artifact(path: Path, kind: str, description: str | None = None) -> ArtifactRef:
    return ArtifactRef(
        uri=to_agent_path(path, workspace=_workspace()),
        kind=kind,
        format=path.suffix.removeprefix(".") or None,
        description=description,
    )


def _ensure_workspace_path_shim(case_dir: Path) -> Path:
    shim_dir = case_dir / ".physicsos_runtime"
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim_path = shim_dir / "sitecustomize.py"
    shim_path.write_text(
        r'''
from __future__ import annotations

import builtins
import os
from pathlib import Path, PosixPath, PurePosixPath, WindowsPath

_WORKSPACE = os.environ.get("PHYSICSOS_WORKSPACE")
_PREFIX = "/workspace"


def _translate(value):
    if not _WORKSPACE:
        return value
    text = os.fspath(value)
    normalized = text.replace("\\", "/")
    if normalized == _PREFIX:
        return _WORKSPACE
    if normalized.startswith(_PREFIX + "/"):
        suffix = normalized[len(_PREFIX) + 1 :]
        return str(Path(_WORKSPACE).joinpath(*PurePosixPath(suffix).parts))
    return value


_original_open = builtins.open


def open(file, *args, **kwargs):
    if isinstance(file, (str, os.PathLike)):
        file = _translate(file)
    return _original_open(file, *args, **kwargs)


builtins.open = open


_original_path_new = Path.__new__


def _path_new(cls, *args, **kwargs):
    if args and isinstance(args[0], (str, os.PathLike)):
        args = (_translate(args[0]), *args[1:])
    return _original_path_new(cls, *args, **kwargs)


Path.__new__ = staticmethod(_path_new)
PosixPath.__new__ = staticmethod(_path_new)
WindowsPath.__new__ = staticmethod(_path_new)


_original_path_open = Path.open


def _path_open(self, *args, **kwargs):
    translated = _translate(self)
    if translated is not self:
        return _original_open(translated, *args, **kwargs)
    return _original_path_open(self, *args, **kwargs)


Path.open = _path_open
PosixPath.open = _path_open
WindowsPath.open = _path_open
'''.lstrip(),
        encoding="utf-8",
    )
    return shim_dir


def _read_json(path_or_uri: str | Path) -> dict[str, object]:
    path = resolve_workspace_path(path_or_uri, workspace=_workspace())
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text_if_exists(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _artifact_status(case_id: str, relative_path: str) -> dict[str, object]:
    path = _case_dir(case_id) / relative_path
    item: dict[str, object] = {
        "path": f"/workspace/cases/{case_id}/{relative_path}",
        "exists": path.exists(),
    }
    if path.exists():
        item["bytes"] = path.stat().st_size
    return item


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


class EstimateTAPSSupportInput(StrictBaseModel):
    problem: PhysicsProblem | None = None
    problem_statement_uri: str | None = None
    geometry_embedding_uri: str | None = None


class EstimateTAPSSupportOutput(StrictBaseModel):
    support: TAPSSupportScore


def estimate_taps_support(input: EstimateTAPSSupportInput) -> EstimateTAPSSupportOutput:
    """Estimate whether the paper-style TAPS route has enough artifacts to proceed."""
    reasons: list[str] = []
    risks: list[str] = []
    score = 0.25
    if input.problem is not None:
        score += 0.25
        reasons.append("Typed PhysicsProblem is available as context.")
        if input.problem.fields:
            score += 0.10
        else:
            risks.append("Problem fields are not specified.")
        if input.problem.boundary_conditions:
            score += 0.10
        else:
            risks.append("Boundary conditions are not specified.")
    if input.problem_statement_uri:
        score += 0.20
        reasons.append("Case-local problem statement is available.")
    if input.geometry_embedding_uri:
        try:
            embedding = _read_json(input.geometry_embedding_uri)
            if embedding.get("schema_version") == "physicsos.geometry_embedding.v1":
                score += 0.20
                reasons.append("Geometry embedding artifact is available.")
        except (OSError, json.JSONDecodeError, ValueError):
            risks.append("Geometry embedding URI could not be read.")
    supported = score >= 0.55
    if not reasons:
        reasons.append("TAPS support needs case artifacts from the paper-style route.")
    return EstimateTAPSSupportOutput(
        support=TAPSSupportScore(score=min(1.0, round(score, 4)), supported=supported, reasons=reasons, risks=risks)
    )


def estimate_taps_support_structured(input: EstimateTAPSSupportInput, **_: object) -> EstimateTAPSSupportOutput:
    """Compatibility wrapper; no structured IR generation is performed."""
    return estimate_taps_support(input)


class AssessTAPSGeometrySeparabilityInput(StrictBaseModel):
    case_id: str | None = None
    geometry_embedding_uri: str | None = None


class AssessTAPSGeometrySeparabilityOutput(StrictBaseModel):
    assessment: TAPSGeometrySeparabilityAssessment


def assess_taps_geometry_separability(input: AssessTAPSGeometrySeparabilityInput) -> AssessTAPSGeometrySeparabilityOutput:
    """Assess whether geometry artifacts are ready for derivation/code generation."""
    uri = input.geometry_embedding_uri
    if uri is None and input.case_id is not None:
        uri = f"/workspace/cases/{input.case_id}/geometry/embedding.json"
    reasons: list[str] = []
    warnings: list[str] = []
    can_execute = False
    missing: list[str] = []
    if uri is None:
        missing.append("geometry/embedding.json")
        warnings.append("No geometry embedding URI or case_id was provided.")
    else:
        try:
            payload = _read_json(uri)
            missing = [str(item) for item in payload.get("missing_required_artifacts", [])] if isinstance(payload, dict) else []
            can_execute = not missing
            reasons.append("Geometry embedding was read from case artifacts.")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            missing.append("geometry/embedding.json")
            warnings.append(f"Could not read geometry embedding: {exc}")
    status: Literal["ready_for_paper_taps", "needs_geometry_embedding", "needs_review"] = (
        "ready_for_paper_taps" if can_execute else "needs_geometry_embedding"
    )
    return AssessTAPSGeometrySeparabilityOutput(
        assessment=TAPSGeometrySeparabilityAssessment(
            status=status,
            can_use_background_grid=can_execute,
            missing_artifacts=missing,
            reasons=reasons or ["Geometry embedding is not ready."],
            warnings=warnings,
        )
    )


class FormulateTAPSEquationInput(StrictBaseModel):
    case_id: str
    derivation_uri: str | None = None
    implementation_notes_uri: str | None = None


class FormulateTAPSEquationOutput(StrictBaseModel):
    equation_summary: ArtifactRef
    warnings: list[str] = Field(default_factory=list)


def formulate_taps_equation(input: FormulateTAPSEquationInput) -> FormulateTAPSEquationOutput:
    """Summarize derivation artifacts for implementation without creating a typed IR."""
    case_dir = _case_dir(input.case_id)
    taps_dir = case_dir / "taps"
    taps_dir.mkdir(parents=True, exist_ok=True)
    derivation_uri = input.derivation_uri or f"/workspace/cases/{input.case_id}/taps/derivation.md"
    notes_uri = input.implementation_notes_uri or f"/workspace/cases/{input.case_id}/taps/implementation_notes.md"
    warnings: list[str] = []
    sections: list[str] = ["# TAPS Equation Summary", ""]
    for title, uri in [("Derivation", derivation_uri), ("Implementation Notes", notes_uri)]:
        path = resolve_workspace_path(uri, workspace=_workspace())
        if path.exists():
            text = path.read_text(encoding="utf-8")
            sections.extend([f"## {title}", "", text[:4000], ""])
        else:
            warnings.append(f"Missing artifact: {uri}")
    path = taps_dir / "equation_summary.md"
    path.write_text("\n".join(sections), encoding="utf-8")
    return FormulateTAPSEquationOutput(
        equation_summary=_artifact(path, "taps_equation_summary", "Paper-style derivation summary for code generation."),
        warnings=warnings,
    )


class BuildTAPSProblemInput(StrictBaseModel):
    case_id: str
    problem_statement_uri: str | None = None
    derivation_uri: str | None = None
    geometry_embedding_uri: str | None = None


class BuildTAPSProblemOutput(StrictBaseModel):
    taps_problem: TAPSProblem
    manifest: ArtifactRef


def build_taps_problem(input: BuildTAPSProblemInput) -> BuildTAPSProblemOutput:
    """Build a lightweight manifest for the paper-style TAPS case."""
    case_dir = _case_dir(input.case_id)
    taps_dir = case_dir / "taps"
    taps_dir.mkdir(parents=True, exist_ok=True)
    taps_problem = TAPSProblem(
        id=f"taps:{input.case_id}",
        case_id=input.case_id,
        problem_statement_uri=input.problem_statement_uri or f"/workspace/cases/{input.case_id}/problem/problem_statement.md",
        derivation_uri=input.derivation_uri or f"/workspace/cases/{input.case_id}/taps/derivation.md",
        implementation_notes_uri=f"/workspace/cases/{input.case_id}/taps/implementation_notes.md",
        geometry_embedding_uri=input.geometry_embedding_uri or f"/workspace/cases/{input.case_id}/geometry/embedding.json",
        route="paper_reproduction",
    )
    path = taps_dir / "taps_problem.json"
    path.write_text(taps_problem.model_dump_json(indent=2), encoding="utf-8")
    return BuildTAPSProblemOutput(taps_problem=taps_problem, manifest=_artifact(path, "taps_problem_manifest"))


class PrepareTAPSBackendCaseBundleInput(StrictBaseModel):
    case_id: str


class PrepareTAPSBackendCaseBundleOutput(StrictBaseModel):
    bundle_manifest: ArtifactRef
    missing_artifacts: list[str] = Field(default_factory=list)


def prepare_taps_backend_case_bundle(input: PrepareTAPSBackendCaseBundleInput) -> PrepareTAPSBackendCaseBundleOutput:
    """Collect case artifacts needed by taps-implementation-agent."""
    case_dir = _case_dir(input.case_id)
    expected = [
        "problem/problem_statement.md",
        "geometry/embedding.json",
        "references/taps_template_eq5.md",
        "references/taps_matrix_definitions.md",
        "references/taps_cot_outline.md",
        "taps/derivation.md",
        "taps/implementation_notes.md",
    ]
    missing = [item for item in expected if not (case_dir / item).exists()]
    payload = {
        "schema_version": "physicsos.taps_case_bundle.v1",
        "case_id": input.case_id,
        "route": "paper_reproduction",
        "artifacts": {item: f"/workspace/cases/{input.case_id}/{item}" for item in expected if item not in missing},
        "missing_artifacts": missing,
    }
    path = case_dir / "taps" / "case_bundle.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return PrepareTAPSBackendCaseBundleOutput(bundle_manifest=_artifact(path, "taps_case_bundle"), missing_artifacts=missing)


class AuthorTAPSRuntimeExtensionInput(StrictBaseModel):
    case_id: str
    purpose: str = "case-local TAPS kernel generated from derivation.md"
    language: Literal["python"] = "python"


class AuthorTAPSRuntimeExtensionOutput(StrictBaseModel):
    extension: TAPSRuntimeExtensionSpec


def author_taps_runtime_extension(input: AuthorTAPSRuntimeExtensionInput) -> AuthorTAPSRuntimeExtensionOutput:
    """Create a case-local kernel scaffold for the implementation agent to edit."""
    taps_dir = _case_dir(input.case_id) / "taps"
    taps_dir.mkdir(parents=True, exist_ok=True)
    kernel_path = taps_dir / "kernel.py"
    if not kernel_path.exists():
        kernel_path.write_text(
            "\n".join(
                [
                    "from __future__ import annotations",
                    "",
                    "def run_case(config: dict | None = None) -> dict:",
                    "    \"\"\"Case-local TAPS kernel scaffold generated from paper-style artifacts.\"\"\"",
                    "    return {\"status\": \"not_implemented\", \"message\": \"taps-implementation-agent must fill this kernel from derivation.md\"}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    extension = TAPSRuntimeExtensionSpec(
        id=f"extension:{input.case_id}",
        case_id=input.case_id,
        purpose=input.purpose,
        entrypoint="run_case",
        artifact=_artifact(kernel_path, "taps_kernel", "Case-local TAPS kernel scaffold."),
        required_inputs=["derivation.md", "implementation_notes.md", "geometry/embedding.json"],
        expected_outputs=["solution", "residual_history", "runtime_metadata"],
        safety_status="draft",
    )
    return AuthorTAPSRuntimeExtensionOutput(extension=extension)


class CompileTAPSKernelInput(StrictBaseModel):
    case_id: str
    overwrite: bool = True
    max_iterations: int = 250
    tolerance: float = 1e-6
    boundary_penalty: float = 25.0


class CompileTAPSKernelOutput(StrictBaseModel):
    kernel: ArtifactRef
    execution_plan: ArtifactRef
    implementation_manifest: ArtifactRef
    static_review: ArtifactRef
    warnings: list[str] = Field(default_factory=list)


def compile_taps_kernel(input: CompileTAPSKernelInput) -> CompileTAPSKernelOutput:
    """Package paper-style artifacts for implementation-agent code generation."""
    case_dir = _case_dir(input.case_id)
    taps_dir = case_dir / "taps"
    taps_dir.mkdir(parents=True, exist_ok=True)
    kernel_path = taps_dir / "kernel.py"
    warnings: list[str] = []
    derivation = _read_text_if_exists(taps_dir / "derivation.md")
    implementation_notes = _read_text_if_exists(taps_dir / "implementation_notes.md")
    geometry_embedding_path = case_dir / "geometry" / "embedding.json"
    geometry_embedding: dict[str, object] = {}
    if geometry_embedding_path.exists():
        try:
            geometry_embedding = json.loads(geometry_embedding_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            warnings.append(f"geometry/embedding.json could not be parsed: {exc}")
    else:
        warnings.append("geometry/embedding.json is missing; implementation prompt will require the agent to resolve geometry inputs.")
    if not derivation:
        warnings.append("taps/derivation.md is missing; implementation prompt will ask the agent to derive or request it first.")
    if not implementation_notes:
        warnings.append("taps/implementation_notes.md is missing; implementation prompt will ask the agent to create it from derivation.md.")

    requested_policy = str(geometry_embedding.get("boundary_constraint_policy") or "penalty")
    has_geometry_terms = _contains_any(derivation + "\n" + implementation_notes, ("chi", "phi", "sdf", "level-set", "level set", "occupancy"))
    has_boundary_terms = _contains_any(derivation + "\n" + implementation_notes, ("nitsche", "penalty", "boundary", "dirichlet", "neumann", "ife"))
    if kernel_path.exists() and not input.overwrite:
        warnings.append("kernel.py already exists and overwrite=false; existing kernel was kept.")
    else:
        kernel_path.write_text(_implementation_agent_kernel_scaffold(), encoding="utf-8")

    execution_plan = {
        "schema_version": "physicsos.taps_execution_plan.v1",
        "case_id": input.case_id,
        "route": "paper_prompt_engineering_case_kernel",
        "kernel": f"/workspace/cases/{input.case_id}/taps/kernel.py",
        "inputs": {
            "context_window": f"/workspace/cases/{input.case_id}/context/context_window.md",
            "background_grid": f"/workspace/cases/{input.case_id}/geometry/background_grid.json",
            "occupancy": f"/workspace/cases/{input.case_id}/geometry/occupancy.npy",
            "sdf": f"/workspace/cases/{input.case_id}/geometry/sdf.npy",
            "boundary_samples": f"/workspace/cases/{input.case_id}/geometry/boundary_samples.npy",
            "normals": f"/workspace/cases/{input.case_id}/geometry/normals.npy",
            "cut_cells": f"/workspace/cases/{input.case_id}/geometry/cut_cells.npy",
            "geometry_embedding": f"/workspace/cases/{input.case_id}/geometry/embedding.json",
            "derivation": f"/workspace/cases/{input.case_id}/taps/derivation.md",
            "implementation_notes": f"/workspace/cases/{input.case_id}/taps/implementation_notes.md",
            "implementation_prompt": f"/workspace/cases/{input.case_id}/taps/implementation_prompt.md",
        },
        "outputs": {
            "solution": f"/workspace/cases/{input.case_id}/taps/solution.npy",
            "solution_summary": f"/workspace/cases/{input.case_id}/taps/solution_summary.json",
            "residual_history": f"/workspace/cases/{input.case_id}/taps/residual_history.json",
            "runtime_metadata": f"/workspace/cases/{input.case_id}/taps/runtime_metadata.json",
        },
        "agent_implementation_contract": {
            "method": "paper_prompt_engineering",
            "one_shot_examples": [
                "/workspace/cases/{case_id}/references/taps_template_eq5.md",
                "/workspace/cases/{case_id}/references/taps_matrix_definitions.md",
                "/workspace/cases/{case_id}/references/taps_cot_outline.md",
            ],
            "required_generated_code_behavior": [
                "implement the derivation in taps/derivation.md",
                "preserve C-HiDeNN-TD/TAPS subspace iteration structure from the reference example",
                "load only case-local artifacts",
                "write solution, residual_history, runtime_metadata, and solution_summary artifacts",
                "raise a clear error instead of fabricating missing physics or verification evidence",
            ],
        },
    }
    plan_path = taps_dir / "execution_plan.json"
    plan_path.write_text(json.dumps(execution_plan, indent=2), encoding="utf-8")
    prompt_path = taps_dir / "implementation_prompt.md"
    prompt_path.write_text(
        _render_implementation_prompt(input.case_id, requested_policy, has_geometry_terms, has_boundary_terms),
        encoding="utf-8",
    )
    review_spec = _default_kernel_review_spec(input.case_id, has_geometry_terms)
    review_spec_path = taps_dir / "kernel_review_spec.json"
    review_spec_path.write_text(json.dumps(review_spec, indent=2), encoding="utf-8")
    implementation_manifest = {
        "schema_version": "physicsos.taps_implementation_manifest.v1",
        "case_id": input.case_id,
        "route": "paper_prompt_engineering_case_kernel",
        "not_ir": True,
        "source_artifacts": {
            "problem_statement": _artifact_status(input.case_id, "problem/problem_statement.md"),
            "context_window": _artifact_status(input.case_id, "context/context_window.md"),
            "geometry_embedding": _artifact_status(input.case_id, "geometry/embedding.json"),
            "geometry_handoff": _artifact_status(input.case_id, "geometry/taps_geometry_handoff.md"),
            "sdf_quality": _artifact_status(input.case_id, "geometry/sdf_quality.json"),
            "derivation": _artifact_status(input.case_id, "taps/derivation.md"),
            "implementation_notes": _artifact_status(input.case_id, "taps/implementation_notes.md"),
            "taps_template": _artifact_status(input.case_id, "references/taps_template_eq5.md"),
            "matrix_definitions": _artifact_status(input.case_id, "references/taps_matrix_definitions.md"),
            "cot_outline": _artifact_status(input.case_id, "references/taps_cot_outline.md"),
            "ibm_ife_notes": _artifact_status(input.case_id, "references/ibm_ife_geometry_embedding.md"),
            "kernel_review_spec": _artifact_status(input.case_id, "taps/kernel_review_spec.json"),
        },
        "detected_derivation_features": {
            "geometry_characteristic_terms": has_geometry_terms,
            "boundary_constraint_terms": has_boundary_terms,
            "requested_boundary_constraint_policy": requested_policy,
        },
        "implementation_agent_tasks": [
            "read context/context_window.md",
            "read implementation_prompt.md",
            "use the one-shot TAPS derivation reference as an example, not as a hard-coded solver",
            "write the actual case-local kernel.py for the current derivation",
            "run static_check_generated_kernel, review_generated_taps_kernel, execute_taps_kernel, and verification-agent tools",
        ],
        "scaffold_behavior": [
            "kernel.py intentionally raises NotImplementedError until taps-implementation-agent fills it",
            "PhysicsOS does not hard-code the one-shot example as the target solver",
            "execution evidence must come from generated case-local code and verification artifacts",
        ],
    }
    manifest_path = taps_dir / "implementation_manifest.json"
    manifest_path.write_text(json.dumps(implementation_manifest, indent=2), encoding="utf-8")
    review_payload = {
        "schema_version": "physicsos.taps_static_review.v1",
        "case_id": input.case_id,
        "status": "generated",
        "notes": [
            "Implementation package was generated from case-local paper-style artifacts.",
            "kernel.py is a scaffold for the taps-implementation-agent, not a fixed built-in numerical solver.",
            "The one-shot TAPS example is treated as a reference pattern for prompt engineering.",
            "Actual numerical evidence must be produced by generated case-local code and then verified.",
        ],
        "warnings": warnings,
    }
    review_path = taps_dir / "static_review.md"
    review_path.write_text(_render_static_review(review_payload), encoding="utf-8")
    return CompileTAPSKernelOutput(
        kernel=_artifact(kernel_path, "taps_kernel_scaffold", "Case-local TAPS kernel scaffold for implementation-agent editing."),
        execution_plan=_artifact(plan_path, "taps_execution_plan", "Case-local TAPS execution plan."),
        implementation_manifest=_artifact(manifest_path, "taps_implementation_manifest", "Audit trail for case-local TAPS kernel generation."),
        static_review=_artifact(review_path, "taps_static_review", "Static review notes for generated kernel."),
        warnings=warnings,
    )


def _default_kernel_review_spec(case_id: str, has_geometry_terms: bool) -> dict[str, object]:
    checks: list[dict[str, object]] = [
        {
            "id": "entrypoint",
            "description": "Generated code exposes the case-local execution entry point.",
            "severity": "error",
            "contains_all": ["def run_case"],
        },
        {
            "id": "scaffold_replaced",
            "description": "Generated code replaced the initial scaffold.",
            "severity": "error",
            "absent_any": ["NotImplementedError", "intentionally not a built-in numerical solver"],
        },
        {
            "id": "solution_artifact",
            "description": "Generated code writes or documents a solution artifact.",
            "severity": "error",
            "contains_any": ["solution.npy", "solution"],
        },
        {
            "id": "residual_history_artifact",
            "description": "Generated code writes residual history for verification.",
            "severity": "error",
            "contains_any": ["residual_history.json", "residual_history"],
        },
        {
            "id": "runtime_metadata_artifact",
            "description": "Generated code writes runtime metadata for auditability.",
            "severity": "error",
            "contains_any": ["runtime_metadata.json", "runtime_metadata"],
        },
        {
            "id": "paper_taps_structure",
            "description": "Generated code keeps visible paper-style TAPS implementation structure.",
            "severity": "error",
            "contains_any": ["subspace", "C-HiDeNN", "chide", "tensor", "Kronecker", "matrix", "weak form"],
        },
        {
            "id": "solution_summary_artifact",
            "description": "Generated code writes a solution summary for downstream reporting.",
            "severity": "warning",
            "contains_any": ["solution_summary.json", "solution_summary"],
        },
        {
            "id": "case_local_context",
            "description": "Generated code visibly references case-local derivation or prompt artifacts.",
            "severity": "warning",
            "contains_any": ["derivation.md", "implementation_notes.md", "implementation_prompt.md", "embedding.json"],
        },
        {
            "id": "no_template_copy_markers",
            "description": "Generated code should not copy prompt/example marker text as implementation.",
            "severity": "warning",
            "absent_any": ["hard-code the example", "one-shot example as the target"],
        },
    ]
    if has_geometry_terms:
        checks.append(
            {
                "id": "geometry_terms",
                "description": "Generated code accounts for geometry embedding terms requested by the derivation.",
                "severity": "warning",
                "contains_any": ["phi", "chi", "sdf", "occupancy", "boundary_samples", "normals", "cut_cells", "embedding.json"],
            }
        )
    return {
        "schema_version": "physicsos.taps_kernel_review_spec.v1",
        "case_id": case_id,
        "description": "Case-local review criteria for generated TAPS kernel. This artifact may be edited by the agent; the review tool only interprets it.",
        "checks": checks,
    }


def _render_static_review(payload: dict[str, object]) -> str:
    lines = ["# TAPS Kernel Static Review", ""]
    lines.append(f"- Status: `{payload['status']}`")
    lines.append(f"- Case: `{payload['case_id']}`")
    lines.extend(["", "## Notes"])
    lines.extend(f"- {note}" for note in payload.get("notes", []) if isinstance(note, str))
    warnings = payload.get("warnings", [])
    if isinstance(warnings, list) and warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in warnings if isinstance(warning, str))
    return "\n".join(lines) + "\n"


def _render_implementation_prompt(case_id: str, requested_policy: str, has_geometry_terms: bool, has_boundary_terms: bool) -> str:
    return f"""# TAPS Implementation Prompt

Role-playing:
You are a computational mechanics expert implementing the code step of the paper's TAPS workflow.

Task:
Write `/workspace/cases/{case_id}/taps/kernel.py` for the current case based on the generated mathematical derivation. Follow the paper's implementation strategy: do not develop the new code from scratch; use the complete 1D S-P-T TAPS template as a code-translation pattern, then replace the PDE-specific equations, matrices, axes, coefficients, boundary terms, and geometry terms for the current derivation.

Inputs:
- Context window: `/workspace/cases/{case_id}/context/context_window.md`
- Derivation: `/workspace/cases/{case_id}/taps/derivation.md`
- Implementation notes: `/workspace/cases/{case_id}/taps/implementation_notes.md`
- Template example: `/workspace/cases/{case_id}/references/taps_template_eq5.md`
- Verification template: `/workspace/cases/{case_id}/references/taps_verification_workflow.md`
- Matrix definitions: `/workspace/cases/{case_id}/references/taps_matrix_definitions.md`
- CoT outline: `/workspace/cases/{case_id}/references/taps_cot_outline.md`
- Geometry embedding: `/workspace/cases/{case_id}/geometry/embedding.json`
- Geometry handoff: `/workspace/cases/{case_id}/geometry/taps_geometry_handoff.md`
- SDF quality report: `/workspace/cases/{case_id}/geometry/sdf_quality.json`
- Background grid and geometry arrays under `/workspace/cases/{case_id}/geometry/`

Use the one-shot TAPS reference as an implementation pattern, not as the target solver.

Required implementation behavior:
1. Read the context window first so the implementation sees the paper's four modules: analysis files, tools, local resources, and examples.
2. Preserve the derivation's C-HiDeNN-TD/TAPS subspace iteration structure.
3. Translate the weak form and matrix definitions into executable case-local code.
4. Use the template as a one-shot implementation example only; replace all problem-specific PDE, coefficient, parameter, axis, matrix, source, boundary, and geometry parts.
5. If STL/geometry embedding is present, load `phi`, `chi`, boundary samples, normals, and cut-cell metadata as coefficients or boundary terms.
6. If `taps_geometry_handoff.md` exists, follow it for geometry artifact loading, shape validation, SDF quality reporting, and verification metadata.
7. Write `solution.npy` or another documented solution artifact, `solution_summary.json`, `residual_history.json`, and `runtime_metadata.json`.
8. If a required derivation or physics input is missing, raise a clear error. Do not fabricate a numerical answer.
9. Leave enough structure for the Fig. 7 verification tools to generate exact solution code, convergence code, execute, and plot.
10. Keep generated numerical choices traceable to the derivation or implementation notes; if the derivation does not define a matrix/operator, stop and request a derivation revision.

Detected derivation features:
- Geometry characteristic terms present: `{has_geometry_terms}`
- Boundary terms present: `{has_boundary_terms}`
- Requested boundary constraint policy: `{requested_policy}`

Output:
- Replace the scaffold in `/workspace/cases/{case_id}/taps/kernel.py`.
- Keep the entry point `run_case(config: dict | None = None) -> dict`.
- Make the script executable as `python taps/kernel.py` from the case directory.
"""


def _implementation_agent_kernel_scaffold() -> str:
    return r'''from __future__ import annotations

import json
from pathlib import Path


def run_case(config: dict | None = None) -> dict:
    case_dir = Path((config or {}).get("case_dir") or Path(__file__).resolve().parents[1])
    prompt = case_dir / "taps" / "implementation_prompt.md"
    raise NotImplementedError(
        "This scaffold is intentionally not a built-in numerical solver. "
        "taps-implementation-agent must read context_window.md, implementation_prompt.md, derivation.md, "
        "and the one-shot references, then replace kernel.py with generated case-local TAPS code. "
        f"Expected prompt: {prompt}"
    )


if __name__ == "__main__":
    try:
        result = run_case()
        print(json.dumps(result, indent=2))
    except Exception as exc:
        print(json.dumps({"status": "not_implemented", "error": str(exc)}, indent=2))
        raise
'''


class StaticCheckGeneratedKernelInput(StrictBaseModel):
    case_id: str
    kernel_uri: str | None = None


class StaticCheckGeneratedKernelOutput(StrictBaseModel):
    static_check: ArtifactRef
    passes: bool


def static_check_generated_kernel(input: StaticCheckGeneratedKernelInput) -> StaticCheckGeneratedKernelOutput:
    """Run py_compile on the generated case-local TAPS kernel."""
    kernel_path = resolve_workspace_path(input.kernel_uri or f"/workspace/cases/{input.case_id}/taps/kernel.py", workspace=_workspace())
    completed = subprocess.run(
        [sys.executable, "-m", "py_compile", str(kernel_path)],
        cwd=str(_workspace()),
        capture_output=True,
        text=True,
        timeout=30,
    )
    payload = {
        "schema_version": "physicsos.taps_kernel_static_check.v1",
        "kernel": to_agent_path(kernel_path, workspace=_workspace()),
        "returncode": completed.returncode,
        "passes": completed.returncode == 0,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    path = _case_dir(input.case_id) / "taps" / "kernel_static_check.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return StaticCheckGeneratedKernelOutput(static_check=_artifact(path, "taps_kernel_static_check"), passes=completed.returncode == 0)


class ReviewGeneratedTAPSKernelInput(StrictBaseModel):
    case_id: str
    kernel_uri: str | None = None
    review_spec_uri: str | None = None


class ReviewGeneratedTAPSKernelOutput(StrictBaseModel):
    review_json: ArtifactRef
    review_markdown: ArtifactRef
    passes: bool
    missing_requirements: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def review_generated_taps_kernel(input: ReviewGeneratedTAPSKernelInput) -> ReviewGeneratedTAPSKernelOutput:
    """Review generated case-local code using the case-local kernel_review_spec.json."""
    case_dir = _case_dir(input.case_id)
    taps_dir = case_dir / "taps"
    taps_dir.mkdir(parents=True, exist_ok=True)
    kernel_path = resolve_workspace_path(input.kernel_uri or f"/workspace/cases/{input.case_id}/taps/kernel.py", workspace=_workspace())
    spec_path = resolve_workspace_path(input.review_spec_uri or f"/workspace/cases/{input.case_id}/taps/kernel_review_spec.json", workspace=_workspace())
    dft_spec_path = _case_dir(input.case_id) / "taps" / "ks_dft_kernel_review_spec.json"
    if input.review_spec_uri is None and not spec_path.exists() and dft_spec_path.exists():
        spec_path = dft_spec_path
    text = kernel_path.read_text(encoding="utf-8") if kernel_path.exists() else ""
    missing: list[str] = []
    warnings: list[str] = []
    if spec_path.exists():
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    else:
        spec = _default_kernel_review_spec(input.case_id, has_geometry_terms=False)
        warnings.append("kernel_review_spec.json was missing; used the default review spec.")
    check_results = _evaluate_kernel_review_spec(text, spec)
    for result in check_results:
        if not result["passes"]:
            if result["severity"] == "error":
                missing.append(str(result["id"]))
            else:
                warnings.append(f"{result['id']}: {result['description']}")
    payload = {
        "schema_version": "physicsos.taps_generated_kernel_review.v1",
        "case_id": input.case_id,
        "kernel": to_agent_path(kernel_path, workspace=_workspace()) if kernel_path.exists() else str(input.kernel_uri),
        "review_spec": to_agent_path(spec_path, workspace=_workspace()) if spec_path.exists() else str(input.review_spec_uri),
        "passes": not missing,
        "missing_requirements": missing,
        "warnings": warnings,
        "check_results": check_results,
    }
    json_path = taps_dir / "generated_kernel_review.json"
    md_path = taps_dir / "generated_kernel_review.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_render_generated_kernel_review(payload), encoding="utf-8")
    return ReviewGeneratedTAPSKernelOutput(
        review_json=_artifact(json_path, "taps_generated_kernel_review", "Machine-readable generated-kernel review."),
        review_markdown=_artifact(md_path, "taps_generated_kernel_review", "Markdown generated-kernel review."),
        passes=not missing,
        missing_requirements=missing,
        warnings=warnings,
    )


def _evaluate_kernel_review_spec(text: str, spec: dict[str, object]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    checks = spec.get("checks", [])
    if not isinstance(checks, list):
        return [
            {
                "id": "invalid_review_spec",
                "description": "kernel_review_spec.json must contain a list field named checks.",
                "severity": "error",
                "passes": False,
            }
        ]
    for raw in checks:
        if not isinstance(raw, dict):
            continue
        check_id = str(raw.get("id") or "unnamed_check")
        description = str(raw.get("description") or "")
        severity = str(raw.get("severity") or "error")
        contains_all = _string_list(raw.get("contains_all"))
        contains_any = _string_list(raw.get("contains_any"))
        absent_any = _string_list(raw.get("absent_any"))
        passes = True
        failed_conditions: list[str] = []
        if contains_all and not all(_text_contains(text, item) for item in contains_all):
            passes = False
            failed_conditions.append("contains_all")
        if contains_any and not any(_text_contains(text, item) for item in contains_any):
            passes = False
            failed_conditions.append("contains_any")
        if absent_any and any(_text_contains(text, item) for item in absent_any):
            passes = False
            failed_conditions.append("absent_any")
        results.append(
            {
                "id": check_id,
                "description": description,
                "severity": "warning" if severity == "warning" else "error",
                "passes": passes,
                "failed_conditions": failed_conditions,
            }
        )
    return results


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item]


def _text_contains(text: str, pattern: str) -> bool:
    return pattern.lower() in text.lower()


def _render_generated_kernel_review(payload: dict[str, object]) -> str:
    lines = [
        "# Generated TAPS Kernel Review",
        "",
        f"- Case: `{payload['case_id']}`",
        f"- Passes: `{payload['passes']}`",
        f"- Kernel: `{payload['kernel']}`",
        "",
        "## Reviewed Requirements",
    ]
    check_results = payload.get("check_results", [])
    if isinstance(check_results, list):
        for item in check_results:
            if isinstance(item, dict):
                lines.append(f"- `{item.get('id')}`: passes=`{item.get('passes')}` severity=`{item.get('severity')}`")
    missing = payload.get("missing_requirements", [])
    if isinstance(missing, list) and missing:
        lines.extend(["", "## Missing Requirements"])
        lines.extend(f"- `{item}`" for item in missing if isinstance(item, str))
    warnings = payload.get("warnings", [])
    if isinstance(warnings, list) and warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {item}" for item in warnings if isinstance(item, str))
    return "\n".join(lines) + "\n"


class ExecuteTAPSKernelInput(StrictBaseModel):
    case_id: str
    kernel_uri: str | None = None
    timeout_seconds: int = 60


class ExecuteTAPSKernelOutput(StrictBaseModel):
    result: SolverResult
    solution: ArtifactRef
    residual_history: ArtifactRef
    runtime_metadata: ArtifactRef
    execution_log: ArtifactRef
    passes: bool


def execute_taps_kernel(input: ExecuteTAPSKernelInput) -> ExecuteTAPSKernelOutput:
    """Execute the generated case-local TAPS kernel in a subprocess."""
    case_dir = _case_dir(input.case_id)
    taps_dir = case_dir / "taps"
    taps_dir.mkdir(parents=True, exist_ok=True)
    kernel_path = resolve_workspace_path(input.kernel_uri or f"/workspace/cases/{input.case_id}/taps/kernel.py", workspace=_workspace())
    shim_dir = _ensure_workspace_path_shim(case_dir)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PHYSICSOS_WORKSPACE"] = str(_workspace())
    env["PYTHONPATH"] = str(shim_dir) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    started = perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, str(kernel_path)],
            cwd=str(case_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=max(1, input.timeout_seconds),
        )
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        completed = subprocess.CompletedProcess(args=[sys.executable, str(kernel_path)], returncode=124, stdout=exc.stdout or "", stderr=exc.stderr or "Timed out")
        timed_out = True
    log_payload = {
        "schema_version": "physicsos.taps_kernel_execution_log.v1",
        "kernel": to_agent_path(kernel_path, workspace=_workspace()),
        "workspace_path_shim": to_agent_path(shim_dir / "sitecustomize.py", workspace=_workspace()),
        "returncode": completed.returncode,
        "timed_out": timed_out,
        "wall_time_seconds": perf_counter() - started,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    log_path = taps_dir / "kernel_execution_log.json"
    log_path.write_text(json.dumps(log_payload, indent=2), encoding="utf-8")
    solution_path = taps_dir / "solution.npy"
    residual_path = taps_dir / "residual_history.json"
    metadata_path = taps_dir / "runtime_metadata.json"
    passes = completed.returncode == 0 and solution_path.exists() and residual_path.exists() and metadata_path.exists()
    result = SolverResult(
        id=f"result:{input.case_id}",
        problem_id=input.case_id,
        backend="paper_taps_case_kernel",
        status="success" if passes else "failed",
        scalar_outputs={"case_id": input.case_id},
        residuals=_residuals_from_history(residual_path),
        artifacts=[
            _artifact(solution_path, "taps_solution", "Numpy solution field from generated TAPS kernel."),
            _artifact(residual_path, "taps_residual_history"),
            _artifact(metadata_path, "taps_runtime_metadata"),
            _artifact(log_path, "taps_kernel_execution_log"),
        ],
        provenance=Provenance(created_by="execute_taps_kernel", source="paper_taps_case_kernel"),
    )
    return ExecuteTAPSKernelOutput(
        result=result,
        solution=_artifact(solution_path, "taps_solution"),
        residual_history=_artifact(residual_path, "taps_residual_history"),
        runtime_metadata=_artifact(metadata_path, "taps_runtime_metadata"),
        execution_log=_artifact(log_path, "taps_kernel_execution_log"),
        passes=passes,
    )


def _residuals_from_history(path: Path) -> dict[str, float]:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(rows, list) or not rows:
        return {}
    last = rows[-1]
    if not isinstance(last, dict):
        return {}
    value = last.get("relative_update")
    return {"final_relative_update": float(value)} if isinstance(value, (float, int)) else {}


class EstimateTAPSResidualInput(StrictBaseModel):
    case_id: str
    verification_report_uri: str | None = None


class EstimateTAPSResidualOutput(StrictBaseModel):
    report: TAPSResidualReport


def estimate_taps_residual(input: EstimateTAPSResidualInput) -> EstimateTAPSResidualOutput:
    """Read verification/report.json as the source of trust, not a typed IR residual."""
    uri = input.verification_report_uri or f"/workspace/cases/{input.case_id}/verification/report.json"
    residuals: dict[str, float] = {}
    converged = False
    try:
        payload = _read_json(uri)
        converged = bool(payload.get("passes") or payload.get("status") == "accepted")
        observed = payload.get("observed_order")
        if isinstance(observed, (float, int)):
            residuals["observed_order"] = float(observed)
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    return EstimateTAPSResidualOutput(
        report=TAPSResidualReport(residuals=residuals, converged=converged, recommended_action="accept" if converged else "verify")
    )


for _tool, _input, _output in [
    (estimate_taps_support, EstimateTAPSSupportInput, EstimateTAPSSupportOutput),
    (estimate_taps_support_structured, EstimateTAPSSupportInput, EstimateTAPSSupportOutput),
    (assess_taps_geometry_separability, AssessTAPSGeometrySeparabilityInput, AssessTAPSGeometrySeparabilityOutput),
    (formulate_taps_equation, FormulateTAPSEquationInput, FormulateTAPSEquationOutput),
    (build_taps_problem, BuildTAPSProblemInput, BuildTAPSProblemOutput),
    (prepare_taps_backend_case_bundle, PrepareTAPSBackendCaseBundleInput, PrepareTAPSBackendCaseBundleOutput),
    (author_taps_runtime_extension, AuthorTAPSRuntimeExtensionInput, AuthorTAPSRuntimeExtensionOutput),
    (compile_taps_kernel, CompileTAPSKernelInput, CompileTAPSKernelOutput),
    (static_check_generated_kernel, StaticCheckGeneratedKernelInput, StaticCheckGeneratedKernelOutput),
    (review_generated_taps_kernel, ReviewGeneratedTAPSKernelInput, ReviewGeneratedTAPSKernelOutput),
    (execute_taps_kernel, ExecuteTAPSKernelInput, ExecuteTAPSKernelOutput),
    (estimate_taps_residual, EstimateTAPSResidualInput, EstimateTAPSResidualOutput),
]:
    _tool.input_model = _input
    _tool.output_model = _output
    _tool.side_effects = "workspace artifacts only"
    _tool.requires_approval = False
