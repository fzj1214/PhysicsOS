from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from shutil import copyfile
from typing import Literal

from pydantic import Field

from physicsos.config import runtime_paths
from physicsos.paths import to_agent_path
from physicsos.schemas.common import ArtifactRef, StrictBaseModel


CASE_STAGE_ORDER = [
    "ANALYSIS_FILES",
    "GEOMETRY_EMBEDDING",
    "CONTEXT_REFERENCES",
    "CONTEXT_WINDOW",
    "TAPS_DERIVATION",
    "CODE_IMPLEMENTATION",
    "FIG7_VERIFICATION",
    "REVISION_OR_REPORT",
]

REFERENCE_FILENAMES = [
    "taps_template_eq5.md",
    "taps_matrix_definitions.md",
    "taps_cot_outline.md",
    "taps_verification_workflow.md",
    "ibm_ife_geometry_embedding.md",
]

KS_DFT_REFERENCE_FILENAMES = [
    "materials_tool_contract.md",
    "ks_dft_formula_notes.md",
    "ks_tensor_basis_notes.md",
    "chefsi_notes.md",
    "lrdm_scf_notes.md",
]


class CreateCaseWorkspaceInput(StrictBaseModel):
    case_id: str | None = None
    user_request: str | None = None
    overwrite_manifest: bool = False


class CreateCaseWorkspaceOutput(StrictBaseModel):
    case_id: str
    case_dir: str
    manifest: ArtifactRef
    execution_plan: ArtifactRef
    events: ArtifactRef
    created_directories: list[str] = Field(default_factory=list)


class ContextWindowArtifact(StrictBaseModel):
    section: str
    path: str
    exists: bool = False
    description: str | None = None


class UpdateCaseStageStatusInput(StrictBaseModel):
    case_id: str
    stage: str = Field(
        description=(
            "Paper-loop stage to update. Use one of: "
            "ANALYSIS_FILES, GEOMETRY_EMBEDDING, CONTEXT_REFERENCES, "
            "CONTEXT_WINDOW, TAPS_DERIVATION, CODE_IMPLEMENTATION, "
            "FIG7_VERIFICATION, REVISION_OR_REPORT. Do not use workspace; "
            "case workspace creation is handled by create_case_workspace."
        ),
        json_schema_extra={"enum": CASE_STAGE_ORDER},
    )
    status: Literal["done", "todo"]
    note: str | None = None


class UpdateCaseStageStatusOutput(StrictBaseModel):
    execution_plan: ArtifactRef
    manifest: ArtifactRef
    completed_stages: list[str] = Field(default_factory=list)
    todo_stages: list[str] = Field(default_factory=list)
    current_stage: str | None = None
    warnings: list[str] = Field(default_factory=list)


def _safe_case_id(value: str | None) -> str:
    if value:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
        if safe:
            return safe[:96]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"case-{stamp}"


def _workspace() -> Path:
    path = runtime_paths().workspace
    path.mkdir(parents=True, exist_ok=True)
    return path


def _case_dir(case_id: str) -> Path:
    return _workspace() / "cases" / _safe_case_id(case_id)


def _artifact(path: Path, kind: str, description: str | None = None) -> ArtifactRef:
    return ArtifactRef(
        uri=to_agent_path(path, workspace=_workspace()),
        kind=kind,
        format=path.suffix.removeprefix(".") or None,
        description=description,
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _append_event(case_dir: Path, event: str, payload: dict[str, object] | None = None) -> None:
    item = {"event": event, "timestamp": datetime.now(UTC).isoformat()}
    if payload:
        item.update(payload)
    events = case_dir / "events.jsonl"
    events.parent.mkdir(parents=True, exist_ok=True)
    with events.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def create_case_workspace(input: CreateCaseWorkspaceInput) -> CreateCaseWorkspaceOutput:
    """Create the paper-style PhysicsOS case workspace and stage plan."""
    case_id = _safe_case_id(input.case_id)
    case_dir = _case_dir(case_id)
    directories = [
        case_dir / "problem",
        case_dir / "geometry",
        case_dir / "references",
        case_dir / "context",
        case_dir / "taps",
        case_dir / "verification" / "plots",
        case_dir / "report" / "figures",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    manifest_path = case_dir / "manifest.json"
    if input.overwrite_manifest or not manifest_path.exists():
        _write_json(
            manifest_path,
            {
                "schema_version": "physicsos.case_manifest.v1",
                "case_id": case_id,
                "created_at": datetime.now(UTC).isoformat(),
                "user_request": input.user_request,
                "stage_order": CASE_STAGE_ORDER,
                "current_stage": "ANALYSIS_FILES",
                "route": "paper_taps_prompt_engineering",
            },
        )

    plan_path = case_dir / "execution_plan.md"
    if input.overwrite_manifest or not plan_path.exists():
        plan_lines = [
            f"# Execution Plan: {case_id}",
            "",
            "Default route: reproduce paper 2509.11447v1 with analysis files, tools, references, context-window examples, TAPS derivation, case-local implementation, and Fig. 7 verification.",
            "",
        ]
        plan_lines.extend(f"- [todo] {stage}" for stage in CASE_STAGE_ORDER)
        plan_path.write_text("\n".join(plan_lines) + "\n", encoding="utf-8")

    events_path = case_dir / "events.jsonl"
    events_path.touch(exist_ok=True)
    _append_event(case_dir, "case_workspace_ready", {"case_id": case_id})

    return CreateCaseWorkspaceOutput(
        case_id=case_id,
        case_dir=to_agent_path(case_dir, workspace=_workspace()),
        manifest=_artifact(manifest_path, "case_manifest", "PhysicsOS case manifest."),
        execution_plan=_artifact(plan_path, "execution_plan", "Visible PhysicsOS stage plan."),
        events=_artifact(events_path, "case_events", "Append-only PhysicsOS case event log."),
        created_directories=[to_agent_path(path, workspace=_workspace()) for path in directories],
    )


def update_case_stage_status(input: UpdateCaseStageStatusInput) -> UpdateCaseStageStatusOutput:
    """Update the visible paper-loop stage checklist for a case.

    This is an audit/status tool only. It does not orchestrate subagents or
    execute any TAPS, geometry, or verification work.
    """
    case_id = _safe_case_id(input.case_id)
    case_dir = _case_dir(case_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    plan_path = case_dir / "execution_plan.md"
    manifest_path = case_dir / "manifest.json"

    statuses = _read_plan_statuses(plan_path)
    if input.stage not in CASE_STAGE_ORDER:
        warning = (
            f"Unknown case stage `{input.stage}`. Expected one of: {', '.join(CASE_STAGE_ORDER)}. "
            "Use `create_case_workspace` for workspace creation; do not mark `workspace` as a stage."
        )
        current_stage = next((stage for stage in CASE_STAGE_ORDER if statuses.get(stage) != "done"), None)
        completed = [stage for stage in CASE_STAGE_ORDER if statuses.get(stage) == "done"]
        todo = [stage for stage in CASE_STAGE_ORDER if statuses.get(stage) != "done"]
        if not plan_path.exists():
            _write_stage_plan(plan_path, case_id, statuses)
        manifest = _read_manifest(manifest_path, case_id)
        manifest.update(
            {
                "updated_at": datetime.now(UTC).isoformat(),
                "stage_status": statuses,
                "current_stage": current_stage,
                "last_stage_warning": warning,
            }
        )
        _write_json(manifest_path, manifest)
        _append_event(case_dir, "case_stage_status_warning", {"stage": input.stage, "status": input.status, "warning": warning})
        return UpdateCaseStageStatusOutput(
            execution_plan=_artifact(plan_path, "execution_plan", "Visible PhysicsOS stage plan."),
            manifest=_artifact(manifest_path, "case_manifest", "PhysicsOS case manifest."),
            completed_stages=completed,
            todo_stages=todo,
            current_stage=current_stage,
            warnings=[warning],
        )

    statuses[input.stage] = input.status
    current_stage = next((stage for stage in CASE_STAGE_ORDER if statuses.get(stage) != "done"), None)
    completed = [stage for stage in CASE_STAGE_ORDER if statuses.get(stage) == "done"]
    todo = [stage for stage in CASE_STAGE_ORDER if statuses.get(stage) != "done"]

    plan_lines = _stage_plan_lines(case_id, statuses)
    if input.note:
        existing_notes = _read_plan_notes(plan_path)
        existing_notes.append(f"- {datetime.now(UTC).isoformat()} `{input.stage}` -> `{input.status}`: {input.note}")
        plan_lines.extend(["", "## Notes", *existing_notes])
    elif plan_path.exists():
        existing_notes = _read_plan_notes(plan_path)
        if existing_notes:
            plan_lines.extend(["", "## Notes", *existing_notes])
    plan_path.write_text("\n".join(plan_lines) + "\n", encoding="utf-8")

    manifest = _read_manifest(manifest_path, case_id)
    manifest.update(
        {
            "updated_at": datetime.now(UTC).isoformat(),
            "stage_status": statuses,
            "current_stage": current_stage,
        }
    )
    _write_json(manifest_path, manifest)
    _append_event(case_dir, "case_stage_status_updated", {"stage": input.stage, "status": input.status, "current_stage": current_stage})

    return UpdateCaseStageStatusOutput(
        execution_plan=_artifact(plan_path, "execution_plan", "Visible PhysicsOS stage plan."),
        manifest=_artifact(manifest_path, "case_manifest", "PhysicsOS case manifest."),
        completed_stages=completed,
        todo_stages=todo,
        current_stage=current_stage,
    )


def _stage_plan_lines(case_id: str, statuses: dict[str, str]) -> list[str]:
    plan_lines = [
        f"# Execution Plan: {case_id}",
        "",
        "Default route: reproduce paper 2509.11447v1 with analysis files, tools, references, context-window examples, TAPS derivation, case-local implementation, and Fig. 7 verification.",
        "",
    ]
    plan_lines.extend(f"- [{statuses.get(stage, 'todo')}] {stage}" for stage in CASE_STAGE_ORDER)
    return plan_lines


def _write_stage_plan(plan_path: Path, case_id: str, statuses: dict[str, str]) -> None:
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text("\n".join(_stage_plan_lines(case_id, statuses)) + "\n", encoding="utf-8")


def _read_plan_statuses(plan_path: Path) -> dict[str, str]:
    statuses = {stage: "todo" for stage in CASE_STAGE_ORDER}
    if not plan_path.exists():
        return statuses
    for line in plan_path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"- \[(done|todo)\] ([A-Z0-9_]+)\s*$", line)
        if match and match.group(2) in statuses:
            statuses[match.group(2)] = match.group(1)
    return statuses


def _read_plan_notes(plan_path: Path) -> list[str]:
    if not plan_path.exists():
        return []
    lines = plan_path.read_text(encoding="utf-8").splitlines()
    try:
        start = lines.index("## Notes") + 1
    except ValueError:
        return []
    return [line for line in lines[start:] if line.strip()]


def _read_manifest(manifest_path: Path, case_id: str) -> dict[str, object]:
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
    return {
        "schema_version": "physicsos.case_manifest.v1",
        "case_id": case_id,
        "created_at": datetime.now(UTC).isoformat(),
        "stage_order": CASE_STAGE_ORDER,
        "route": "paper_taps_prompt_engineering",
    }


class LoadTAPSCaseReferencesInput(StrictBaseModel):
    case_id: str
    include_geometry_embedding: bool = True
    include_ks_dft: bool = False


class LoadTAPSCaseReferencesOutput(StrictBaseModel):
    references_dir: str
    references: list[ArtifactRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _reference_source_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "docs" / "knowledge_seed" / "references"


def load_taps_case_references(input: LoadTAPSCaseReferencesInput) -> LoadTAPSCaseReferencesOutput:
    """Copy paper-style TAPS prompt references into a case workspace."""
    case_dir = _case_dir(input.case_id)
    references_dir = case_dir / "references"
    references_dir.mkdir(parents=True, exist_ok=True)
    source_dir = _reference_source_dir()
    filenames = list(REFERENCE_FILENAMES)
    if not input.include_geometry_embedding:
        filenames.remove("ibm_ife_geometry_embedding.md")
    if input.include_ks_dft:
        filenames.extend(KS_DFT_REFERENCE_FILENAMES)

    artifacts: list[ArtifactRef] = []
    warnings: list[str] = []
    for filename in filenames:
        source = source_dir / filename
        target = references_dir / filename
        if source.exists():
            copyfile(source, target)
        else:
            warnings.append(f"Reference source is missing: {source}")
            target.write_text(f"# {filename}\n\nReference source missing in this installation.\n", encoding="utf-8")
        artifacts.append(_artifact(target, "taps_reference", f"TAPS reference file {filename}."))
    _append_event(case_dir, "taps_references_loaded", {"count": len(artifacts)})
    return LoadTAPSCaseReferencesOutput(
        references_dir=to_agent_path(references_dir, workspace=_workspace()),
        references=artifacts,
        warnings=warnings,
    )


class BuildTAPSDerivationPromptInput(StrictBaseModel):
    case_id: str
    problem_statement_uri: str | None = None
    geometry_embedding_uri: str | None = None


class BuildTAPSDerivationPromptOutput(StrictBaseModel):
    prompt: ArtifactRef
    required_inputs: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def build_taps_derivation_prompt(input: BuildTAPSDerivationPromptInput) -> BuildTAPSDerivationPromptOutput:
    """Build the Fig. 4 / Appendix D style TAPS derivation prompt from case files."""
    case_dir = _case_dir(input.case_id)
    refs = case_dir / "references"
    problem_statement = input.problem_statement_uri or "/".join(["/workspace", "cases", input.case_id, "problem", "problem_statement.md"])
    geometry_embedding = input.geometry_embedding_uri or "/".join(["/workspace", "cases", input.case_id, "geometry", "geometry_embedding.md"])
    context_window = "/".join(["/workspace", "cases", input.case_id, "context", "context_window.md"])
    required = [
        context_window,
        problem_statement,
        f"/workspace/cases/{input.case_id}/references/taps_template_eq5.md",
        f"/workspace/cases/{input.case_id}/references/taps_matrix_definitions.md",
        f"/workspace/cases/{input.case_id}/references/taps_cot_outline.md",
    ]
    warnings = []
    for item in required:
        path = _workspace() / Path(*item.removeprefix("/workspace/").split("/")) if item.startswith("/workspace/") else Path(item)
        if not path.exists():
            warnings.append(f"Input may be missing: {item}")

    prompt_text = f"""# TAPS Derivation Prompt

This prompt follows the Appendix D structure in paper 2509.11447v1. It has five parts: role-playing, few-shot prompt, constraints, chain-of-thought derivation requirements, and formatting guidelines.

## 1. Role-playing

You are a computational mechanics expert tasked with making targeted corrections to a mathematical derivation.

Task:
Create the mathematical derivation for the given problem statement based on the template example. The target is a TAPS / C-HiDeNN-TD derivation that can be translated into case-local implementation code and verified using the Fig. 7 workflow.

## 2. Few-shot prompt

Use the complete derivation template the way the paper uses the Eq. 5 example: as an in-context demonstration of structure and symbolic style.

- Context window: `{context_window}`
- Template example: `/workspace/cases/{input.case_id}/references/taps_template_eq5.md`
- Problem statement: `{problem_statement}`
- Matrix definition: `/workspace/cases/{input.case_id}/references/taps_matrix_definitions.md`
- CoT derivation outline: `/workspace/cases/{input.case_id}/references/taps_cot_outline.md`
- Geometry embedding notes: `{geometry_embedding}`  # PhysicsOS extension; use only when geometry exists
- Geometry handoff: `/workspace/cases/{input.case_id}/geometry/taps_geometry_handoff.md`  # PhysicsOS extension; use only when geometry exists

Required actions:
1. Start with the complete template derivation.
2. Replace only the parts required by the new PDE, coefficients, parameters, boundary conditions, and geometry.
3. Preserve all valid TAPS structure from the reference example.
4. Derive every subspace iteration step by step.

## 3. Constraints

- Use only matrix symbols from `taps_matrix_definitions.md`, unless a new matrix is mathematically required; if a new matrix is required, define it before use.
- The Eq. 5 template is not the target problem and must not be hard-coded.
- Do not replace the TAPS/C-HiDeNN-TD route with POD, neural operators, FEM-only code, or a full-solver workflow.
- Geometry embedding is the only PhysicsOS extension to the paper route. Treat STL/Gmsh artifacts as analysis files that provide `phi(x)`, `chi(x)`, boundary samples, normals, cut cells, and possible geometry parameter axes.
- When `/geometry/taps_geometry_handoff.md` exists, use it to connect geometry artifacts to derivation, implementation, and verification responsibilities.
- Gmsh is a geometry preprocessor here, not a PDE solver.

## 4. Chain-of-thought derivation requirements

The visible derivation must contain the inspectable mathematical reasoning described in Fig. 5 and Fig. 6:

1. Start from the strong form and define all independent variables, fields, parameters, source terms, boundary conditions, and initial conditions.
2. Derive the weak form.
3. Insert the C-HiDeNN-TD trial function into the weak form.
4. Insert the corresponding test-function variation for the current subspace.
5. Separate axis-dependent factors.
6. Define all stiffness, mass, derivative, coefficient, source, boundary, and geometry matrices before using them.
7. Derive the final matrix system for the current subspace iteration.
8. Repeat the derivation for each spatial, temporal, material-parameter, and geometry-embedding axis.
9. State the update order and stopping criterion for subspace iteration.
10. Do not jump directly to the final matrix form.

For STL/3D geometry, explicitly show how `phi(x)`, `chi(x)=H(-phi(x))`, normals, boundary samples, cut-cell quadrature, and geometry parameters enter the Galerkin weak form and the axis matrices.

## 5. Formatting guidelines

- Write `/workspace/cases/{input.case_id}/taps/derivation.md`.
- Write `/workspace/cases/{input.case_id}/taps/implementation_notes.md`.
- Use Markdown with standard LaTeX math formatting.
- Preserve the derivation structure from the template where it remains valid.
- Include implementation notes that tell the implementation agent how to translate the final subspace matrix equations into case-local code.
- End with a short checklist of generated equations, matrices, required artifacts, and unresolved assumptions.
"""
    prompt_path = case_dir / "taps" / "derivation_prompt.md"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt_text, encoding="utf-8")
    _append_event(case_dir, "taps_derivation_prompt_built", {"prompt": to_agent_path(prompt_path, workspace=_workspace())})
    return BuildTAPSDerivationPromptOutput(
        prompt=_artifact(prompt_path, "taps_derivation_prompt", "Paper-style TAPS derivation prompt."),
        required_inputs=required,
        warnings=warnings,
    )


class BuildPaperContextWindowInput(StrictBaseModel):
    case_id: str
    user_prompt: str | None = None
    include_geometry_embedding: bool = True
    include_ks_dft_materials: bool = False


class BuildPaperContextWindowOutput(StrictBaseModel):
    context_window: ArtifactRef
    manifest: ArtifactRef
    sections: list[ContextWindowArtifact] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _context_artifact(section: str, path: Path, description: str) -> ContextWindowArtifact:
    return ContextWindowArtifact(
        section=section,
        path=to_agent_path(path, workspace=_workspace()),
        exists=path.exists(),
        description=description,
    )


def build_paper_context_window(input: BuildPaperContextWindowInput) -> BuildPaperContextWindowOutput:
    """Assemble the paper's context-window module as case-local prompt context.

    This tool only packages pointers and short guidance for the main agent and
    subagents. It does not orchestrate the TAPS route or execute solver logic.
    """
    case_dir = _case_dir(input.case_id)
    context_dir = case_dir / "context"
    context_dir.mkdir(parents=True, exist_ok=True)

    references = [
        case_dir / "references" / "taps_template_eq5.md",
        case_dir / "references" / "taps_matrix_definitions.md",
        case_dir / "references" / "taps_cot_outline.md",
        case_dir / "references" / "taps_verification_workflow.md",
    ]
    if input.include_geometry_embedding:
        references.append(case_dir / "references" / "ibm_ife_geometry_embedding.md")
    if input.include_ks_dft_materials:
        references.extend(case_dir / "references" / filename for filename in KS_DFT_REFERENCE_FILENAMES)

    sections = [
        _context_artifact("analysis files", case_dir / "problem" / "problem_statement.md", "Current PDE/problem statement prepared from user inputs."),
        _context_artifact("analysis files", case_dir / "problem" / "problem.json", "Structured problem facts when available."),
        _context_artifact("analysis files", case_dir / "problem" / "open_questions.md", "Missing inputs and assumptions requiring user or agent resolution."),
    ]
    if input.include_geometry_embedding:
        sections.extend(
            [
                _context_artifact("analysis files", case_dir / "geometry" / "geometry_embedding.md", "STL/Gmsh immersed-boundary notes for the derivation prompt."),
                _context_artifact("analysis files", case_dir / "geometry" / "taps_geometry_context.md", "Geometry analysis-file input containing phi, chi, normals, samples, and cut-cell guidance."),
                _context_artifact("analysis files", case_dir / "geometry" / "taps_geometry_handoff.md", "Geometry handoff describing how derivation, implementation, and verification agents should consume STL/CAD embedding artifacts."),
                _context_artifact("analysis files", case_dir / "geometry" / "sdf_quality.json", "SDF/voxelization quality report; required before trusting fallback geometry artifacts."),
            ]
        )
    if input.include_ks_dft_materials:
        sections.extend(
            [
                _context_artifact("analysis files", case_dir / "materials" / "ks_dft_material_context.md", "Material-tool contract for KS-DFT-TAPS derivation; fixes standardized structure, reciprocal lattice, symmetry, and k-point artifacts."),
                _context_artifact("analysis files", case_dir / "materials" / "ks_dft_material_context.json", "Machine-readable KS-DFT material context."),
                _context_artifact("analysis files", case_dir / "materials" / "structure_standardized.json", "pymatgen-standardized structure; the only structure source for KS-DFT-TAPS derivation."),
                _context_artifact("analysis files", case_dir / "materials" / "symmetry_dataset.json", "spglib/pymatgen symmetry dataset; do not recompute in prompt."),
                _context_artifact("analysis files", case_dir / "materials" / "reciprocal_lattice.json", "Reciprocal lattice with explicit convention."),
                _context_artifact("analysis files", case_dir / "materials" / "kmesh.json", "Uniform k-point mesh policy from deterministic materials tools."),
                _context_artifact("analysis files", case_dir / "materials" / "irreducible_kpoints.json", "Irreducible k-points and weights from spglib."),
                _context_artifact("analysis files", case_dir / "materials" / "kpath_seekpath.json", "SeekPath high-symmetry line path for post-SCF band path use."),
                _context_artifact("analysis files", case_dir / "materials" / "molecule.json", "Molecule/cluster Cartesian coordinates with explicit charge and multiplicity."),
                _context_artifact("analysis files", case_dir / "materials" / "ks_dft_molecular_context.md", "Molecular KS-DFT context with open-boundary/vacuum-box policy gates."),
                _context_artifact("analysis files", case_dir / "materials" / "ks_dft_molecular_context.json", "Machine-readable molecular KS-DFT context."),
                _context_artifact("analysis files", case_dir / "taps" / "molecular_taps_scaling_policy.json", "LLM-selectable large-molecule TAPS scaling strategy contract."),
                _context_artifact("analysis files", case_dir / "problem" / "ks_dft_problem.json", "KS-DFT-TAPS problem specification."),
                _context_artifact("analysis files", case_dir / "problem" / "ks_dft_open_questions.md", "Missing KS-DFT assumptions and material inputs."),
            ]
        )
    sections.extend(
        _context_artifact("online/local resources", reference, "Case-local prompt/reference resource copied from the PhysicsOS knowledge seed.")
        for reference in references
    )
    sections.extend(
        [
            _context_artifact("context examples", case_dir / "references" / "taps_template_eq5.md", "Few-shot TAPS derivation template based on the paper's Eq. 5 example."),
            _context_artifact("context examples", case_dir / "references" / "taps_cot_outline.md", "Step-by-step derivation outline matching the paper's CoT prompting rationale."),
            _context_artifact("tools", case_dir / "taps" / "derivation_prompt.md", "Prompt produced for the derivation agent."),
            _context_artifact("tools", case_dir / "taps" / "implementation_prompt.md", "Prompt package produced for the implementation agent."),
            _context_artifact("tools", case_dir / "verification" / "report.md", "Fig. 7 verification report target."),
        ]
    )

    warnings = [f"Context input is not present yet: {item.path}" for item in sections if not item.exists]
    manifest_payload = {
        "schema_version": "physicsos.paper_context_window.v1",
        "case_id": input.case_id,
        "created_at": datetime.now(UTC).isoformat(),
        "route": "paper_taps_prompt_engineering",
        "purpose": "Case-local context window for the paper-style CAE agent modules.",
        "not_a_workflow_engine": True,
        "modules": ["analysis files", "tools", "online/local resources", "context window"],
        "user_prompt": input.user_prompt,
        "sections": [section.model_dump() for section in sections],
        "warnings": warnings,
    }

    manifest_path = context_dir / "context_window.json"
    _write_json(manifest_path, manifest_payload)

    lines = [
        "# Paper Context Window",
        "",
        "Persona:",
        "You are a professional CAE agent specializing in paper-style TAPS data-free MOR development.",
        "",
        "Purpose:",
        "Assemble the current context window described in paper 2509.11447v1: analysis files, tools, online/local resources, and context examples. This file guides agent prompting; it is not a fixed numerical solver and not a LangGraph workflow.",
        "",
        "Default route:",
        "analysis files -> few-shot/CoT TAPS derivation -> case-local code implementation -> Fig. 7 verification -> revise/report",
        "",
        "Paper constraints:",
        "- Use the Eq. 5 TAPS derivation as a few-shot/CoT example, not as hard-coded execution logic.",
        "- Derive weak form, C-HiDeNN-TD approximation, axis matrices, and subspace iterations step by step.",
        "- Generate case-local implementation code from the derivation and prompt package.",
        "- Verify using exact/manufactured solution generation, convergence-study generation, execution, and plotting.",
        "- Human inspection remains expected for generated derivations and code.",
        "",
    ]
    if input.user_prompt:
        lines.extend(["User prompt:", input.user_prompt, ""])

    grouped: dict[str, list[ContextWindowArtifact]] = {}
    for section in sections:
        grouped.setdefault(section.section, []).append(section)
    for name in ["analysis files", "tools", "online/local resources", "context examples"]:
        lines.extend([name.title(), ""])
        for artifact in grouped.get(name, []):
            status = "present" if artifact.exists else "missing"
            lines.append(f"- [{status}] `{artifact.path}` - {artifact.description}")
        lines.append("")

    if input.include_geometry_embedding:
        lines.extend(
            [
                "PhysicsOS Geometry Extension",
                "",
                "For STL/CAD cases, geometry embedding is an analysis-file extension before TAPS derivation: STL or generated primitive -> Gmsh preprocessing -> Cartesian background grid -> SDF/occupancy/boundary samples/normals/cut cells -> SDF quality report -> geometry notes -> geometry handoff. Gmsh is not used as the PDE solver.",
                "",
                "The geometry handoff is intentionally strong: it tells the derivation agent how geometry enters the weak form, the implementation agent which artifacts to load and validate, and the verification agent which geometry evidence to report. It still remains inside the paper-style TAPS prompt-engineering loop.",
                "",
            ]
        )
    if input.include_ks_dft_materials:
        lines.extend(
            [
                "KS-DFT-TAPS Materials Extension",
                "",
                "For KS-DFT-TAPS cases, materials preprocessing is a deterministic tool layer before derivation. For crystals, use pymatgen/seekpath/spglib artifacts for structure, symmetry, reciprocal lattice, irreducible k-points, and high-symmetry paths. For molecules/clusters, use `molecule.json`, `ks_dft_molecular_context.json`, and `molecular_taps_scaling_policy.json` for charge, multiplicity, boundary policy, and large-system strategy selection. The derivation agent must not invent or recompute those quantities.",
                "",
                "Required KS-DFT hard rules:",
                "- Read `/workspace/cases/{}/materials/ks_dft_material_context.md` for crystal cases or `/workspace/cases/{}/materials/ks_dft_molecular_context.md` for molecule/cluster cases before deriving.".format(input.case_id, input.case_id),
                "- For crystals, use `structure_standardized.json` as the only structure source.",
                "- For crystals, use `kmesh.json` and `irreducible_kpoints.json` for Brillouin-zone integration.",
                "- For crystals, use `kpath_seekpath.json` only for line-mode band path after SCF verification.",
                "- For molecules/clusters, do not silently reuse crystal kmesh/kpath logic; finalize open-boundary or explicit vacuum-box policy first.",
                "- If required material artifacts are missing, request `materials-preprocess-agent`.",
                "",
            ]
        )
    if warnings:
        lines.extend(["Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")

    context_path = context_dir / "context_window.md"
    context_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _append_event(case_dir, "paper_context_window_built", {"context": to_agent_path(context_path, workspace=_workspace()), "warnings": len(warnings)})

    return BuildPaperContextWindowOutput(
        context_window=_artifact(context_path, "paper_context_window", "Case-local context window for paper-style TAPS prompting."),
        manifest=_artifact(manifest_path, "paper_context_window_manifest", "Machine-readable context-window manifest."),
        sections=sections,
        warnings=warnings,
    )


for _tool, _input, _output in [
    (create_case_workspace, CreateCaseWorkspaceInput, CreateCaseWorkspaceOutput),
    (update_case_stage_status, UpdateCaseStageStatusInput, UpdateCaseStageStatusOutput),
    (load_taps_case_references, LoadTAPSCaseReferencesInput, LoadTAPSCaseReferencesOutput),
    (build_taps_derivation_prompt, BuildTAPSDerivationPromptInput, BuildTAPSDerivationPromptOutput),
    (build_paper_context_window, BuildPaperContextWindowInput, BuildPaperContextWindowOutput),
]:
    _tool.input_model = _input
    _tool.output_model = _output
    _tool.side_effects = "workspace artifacts only"
    _tool.requires_approval = False
