from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Literal

from pydantic import Field

from physicsos.agents.structured import CoreAgentLLMConfig, StructuredLLMClient, call_structured_agent
from physicsos.config import project_root
from physicsos.schemas.common import ArtifactRef, StrictBaseModel
from physicsos.schemas.contracts import PhysicsProblemContract
from physicsos.schemas.knowledge import KnowledgeContext
from physicsos.schemas.postprocess import PostprocessResult, VisualizationSpec
from physicsos.schemas.problem import PhysicsProblem
from physicsos.schemas.solver import SolverResult
from physicsos.schemas.verification import VerificationReport
from physicsos.tools.memory_tools import CaseMemoryContext


class ExtractKPIsInput(StrictBaseModel):
    problem: PhysicsProblem
    result: SolverResult


class ExtractKPIsOutput(StrictBaseModel):
    kpis: dict[str, float | int | str] = Field(default_factory=dict)
    units: dict[str, str] = Field(default_factory=dict)


def extract_kpis(input: ExtractKPIsInput) -> ExtractKPIsOutput:
    """Extract engineering KPIs from solver outputs."""
    return ExtractKPIsOutput(kpis=input.result.scalar_outputs, units={})


class GenerateVisualizationsInput(StrictBaseModel):
    problem: PhysicsProblem
    result: SolverResult
    visualization_plan: list[VisualizationSpec] = Field(default_factory=list)


class GenerateVisualizationsOutput(StrictBaseModel):
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PostprocessPlanInput(StrictBaseModel):
    problem: PhysicsProblem
    problem_contract: PhysicsProblemContract | None = None
    result: SolverResult
    verification: VerificationReport
    available_artifacts: list[ArtifactRef] = Field(default_factory=list)
    knowledge_context: KnowledgeContext | None = None
    case_memory_context: CaseMemoryContext | None = None


class PostprocessPlanOutput(StrictBaseModel):
    visualization_plan: list[VisualizationSpec] = Field(default_factory=list)
    report_sections: list[str] = Field(default_factory=list)
    figure_captions: dict[str, str] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


POSTPROCESS_PLAN_SYSTEM_PROMPT = """You are the PhysicsOS postprocess planning agent.
Return only a JSON object matching PostprocessPlanOutput.

Task:
- Preserve the locked PhysicsProblemContract when provided.
- Select figures, KPIs, report sections, captions, and recommendations from existing typed solver, verification, and artifact data.
- Do not invent artifact paths, solver fields, or numeric values.
- If a desired figure needs a missing field or artifact, record a warning instead of pretending it can be plotted.
- The deterministic postprocess tools will read artifacts, render figures, and write the report manifest.
"""


def plan_postprocess(input: PostprocessPlanInput) -> PostprocessPlanOutput:
    """Deterministic postprocess planning fallback."""
    visualization_plan: list[VisualizationSpec] = []
    artifact_kinds = {artifact.kind for artifact in [*input.available_artifacts, *input.result.artifacts]}
    fields = [field.name for field in input.problem.fields]
    if any("solution" in kind for kind in artifact_kinds):
        if input.problem.domain == "fluid" and any(field.lower() in {"u", "velocity"} for field in fields):
            visualization_plan.append(
                VisualizationSpec(
                    kind="streamline",
                    fields=fields,
                    description="Velocity field quiver or streamline visualization from solver output.",
                )
            )
            visualization_plan.append(
                VisualizationSpec(
                    kind="contour",
                    fields=[field for field in fields if field.lower() in {"p", "pressure"}] or fields[:1],
                    description="Pressure or scalar field contour from solver output.",
                )
            )
        elif input.problem.geometry.dimension > 1:
            visualization_plan.append(
                VisualizationSpec(kind="contour", fields=fields[:1], description="2D scalar field heatmap from solver output.")
            )
        else:
            visualization_plan.append(
                VisualizationSpec(kind="plot", fields=fields[:1], description="1D scalar field line plot from solver output.")
            )
    visualization_plan.append(
        VisualizationSpec(kind="plot", fields=[], description="Residual and scalar output summary.")
    )
    report_sections = [
        "Executive Summary",
        "Problem Specification",
        "Solver Provenance",
        "KPIs and Scalar Outputs",
        "Embedded Visualizations",
        "Verification Appendix",
        "Artifact Manifest",
        "Recommended Next Actions",
    ]
    recommendations = [input.verification.explanation or input.verification.recommended_next_action]
    if input.verification.status != "accepted":
        recommendations.append(f"Follow verification recommendation: {input.verification.recommended_next_action}.")
    return PostprocessPlanOutput(
        visualization_plan=visualization_plan,
        report_sections=report_sections,
        recommendations=recommendations,
        assumptions=["Deterministic postprocess fallback selected figures from solver artifacts and geometry dimension."],
    )


def plan_postprocess_structured(
    input: PostprocessPlanInput,
    *,
    client: StructuredLLMClient,
    config: CoreAgentLLMConfig | None = None,
) -> PostprocessPlanOutput:
    """LLM-backed postprocess planning with strict Pydantic validation."""
    result = call_structured_agent(
        agent_name="postprocess-planning-agent",
        input_model=input,
        output_model=PostprocessPlanOutput,
        system_prompt=POSTPROCESS_PLAN_SYSTEM_PROMPT,
        client=client,
        config=config,
    )
    if result.output is not None:
        return result.output
    fallback = plan_postprocess(input)
    return fallback.model_copy(
        update={
            "assumptions": [
                *fallback.assumptions,
                "Structured LLM postprocess planning failed validation; deterministic fallback was used.",
                result.error or "Structured postprocess planner returned no validated output.",
            ]
        }
    )


def generate_visualizations(input: GenerateVisualizationsInput) -> GenerateVisualizationsOutput:
    """Generate lightweight local visualization artifacts from solver outputs."""
    output_dir = project_root() / "scratch" / input.problem.id.replace(":", "_") / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[ArtifactRef] = []

    summary_path = output_dir / "residual_summary.json"
    summary_payload = {
        "problem_id": input.problem.id,
        "result_id": input.result.id,
        "backend": input.result.backend,
        "status": input.result.status,
        "residuals": input.result.residuals,
        "uncertainty": input.result.uncertainty,
        "scalar_outputs": input.result.scalar_outputs,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    artifacts.append(ArtifactRef(uri=str(Path(summary_path)), kind="visualization:residual_summary", format="json"))

    warnings: list[str] = []
    solution_payload = _load_solution_payload(input.result)
    mpl_artifacts, mpl_warnings = _generate_matplotlib_figures(input, output_dir, solution_payload)
    artifacts.extend(mpl_artifacts)
    warnings.extend(mpl_warnings)
    values = _extract_plot_values(solution_payload) if solution_payload is not None else []
    if values and not any(artifact.kind.startswith("visualization:solution") and artifact.format == "png" for artifact in artifacts):
        svg_path = output_dir / "solution_preview.svg"
        svg_path.write_text(_render_line_svg(values, title=f"{input.result.backend} solution preview"), encoding="utf-8")
        artifacts.append(ArtifactRef(uri=str(Path(svg_path)), kind="visualization:solution_preview", format="svg"))

    return GenerateVisualizationsOutput(artifacts=artifacts, warnings=warnings)


def _generate_matplotlib_figures(
    input: GenerateVisualizationsInput,
    output_dir: Path,
    solution_payload: dict | None,
) -> tuple[list[ArtifactRef], list[str]]:
    if solution_payload is None:
        return [], []
    values = _extract_plot_values(solution_payload)
    matrix = _extract_matrix(solution_payload)
    vector = _extract_vector_field(solution_payload)
    warnings: list[str] = []
    if not values and not matrix and vector is None:
        return [], warnings
    payload_path = output_dir / "matplotlib_payload.json"
    result_path = output_dir / "matplotlib_result.json"
    script_path = output_dir / "render_matplotlib.py"
    payload = {
        "backend": input.result.backend,
        "field_name": _first_field_name(input, default="solution"),
        "values": values,
        "matrix": matrix,
        "vector": {"u": vector[0], "v": vector[1]} if vector is not None else None,
        "output_dir": str(output_dir),
    }
    if vector is not None:
        pass
    elif input.problem.domain == "fluid":
        warnings.append("No plottable 2D vector field was found for fluid quiver visualization.")
    payload_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    script_path.write_text(_matplotlib_renderer_script(), encoding="utf-8")
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    mpl_config_dir = project_root() / "scratch" / "_matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    env.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    try:
        completed = subprocess.run(
            [sys.executable, str(script_path), str(payload_path), str(result_path)],
            cwd=str(project_root()),
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], [*warnings, f"matplotlib subprocess unavailable; used SVG/JSON fallback: {exc}"]
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()[-1:] or [str(completed.returncode)]
        return [], [*warnings, f"matplotlib subprocess failed; used SVG/JSON fallback: {detail[0]}"]
    try:
        raw_artifacts = json.loads(result_path.read_text(encoding="utf-8")).get("artifacts", [])
    except (OSError, json.JSONDecodeError) as exc:
        return [], [*warnings, f"matplotlib result artifact was invalid; used SVG/JSON fallback: {exc}"]
    artifacts = [
        ArtifactRef(uri=str(item["uri"]), kind=str(item["kind"]), format=str(item.get("format") or "png"))
        for item in raw_artifacts
        if isinstance(item, dict) and item.get("uri") and item.get("kind")
    ]
    return artifacts, warnings


def _matplotlib_renderer_script() -> str:
    return r'''
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
result_path = Path(sys.argv[2])
output_dir = Path(payload["output_dir"])
artifacts = []

values = payload.get("values") or []
if values:
    path = output_dir / "solution_line.png"
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=140)
    ax.plot(range(len(values)), values, color="#0f766e", linewidth=2.0)
    ax.set_title(f"{payload.get('backend')} solution")
    ax.set_xlabel("sample")
    ax.set_ylabel(payload.get("field_name") or "value")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    artifacts.append({"uri": str(path), "kind": "visualization:solution_line", "format": "png"})

matrix = payload.get("matrix") or []
if matrix:
    path = output_dir / "solution_heatmap.png"
    fig, ax = plt.subplots(figsize=(5.4, 4.8), dpi=140)
    image = ax.imshow(matrix, origin="lower", cmap="viridis", aspect="auto")
    ax.set_title(f"{payload.get('field_name') or 'solution'} heatmap")
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    fig.colorbar(image, ax=ax, shrink=0.82)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    artifacts.append({"uri": str(path), "kind": "visualization:solution_heatmap", "format": "png"})

vector = payload.get("vector")
if isinstance(vector, dict) and vector.get("u") and vector.get("v"):
    u = vector["u"]
    v = vector["v"]
    path = output_dir / "velocity_quiver.png"
    fig, ax = plt.subplots(figsize=(5.6, 4.8), dpi=140)
    step = max(1, len(u) // 24)
    xs = list(range(0, len(u[0]) if u and u[0] else 0, step))
    ys = list(range(0, len(u), step))
    uu = [[u[y][x] for x in xs] for y in ys]
    vv = [[v[y][x] for x in xs] for y in ys]
    ax.quiver(xs, ys, uu, vv, color="#0f766e")
    ax.set_title("velocity field")
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    artifacts.append({"uri": str(path), "kind": "visualization:velocity_quiver", "format": "png"})

result_path.write_text(json.dumps({"artifacts": artifacts}, ensure_ascii=False), encoding="utf-8")
'''


def _load_solution_payload(result: SolverResult) -> dict | None:
    for artifact in result.artifacts:
        if artifact.format != "json" or "solution" not in artifact.kind:
            continue
        try:
            return json.loads(Path(artifact.uri).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return None


def _flatten_numbers(value: object) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (float, int)):
        return [float(value)]
    if isinstance(value, list):
        numbers: list[float] = []
        for item in value:
            numbers.extend(_flatten_numbers(item))
        return numbers
    if isinstance(value, dict):
        numbers: list[float] = []
        for item in value.values():
            numbers.extend(_flatten_numbers(item))
        return numbers
    return []


def _extract_plot_values(payload: dict | None) -> list[float]:
    if payload is None:
        return []
    if "values" in payload:
        values = payload["values"]
        if isinstance(values, list) and values and isinstance(values[0], list):
            return _flatten_numbers(values[len(values) // 2])
        return _flatten_numbers(values)
    if "fields" in payload and isinstance(payload["fields"], dict):
        first_field = next(iter(payload["fields"].values()), None)
        values = first_field
        if isinstance(values, list) and values and isinstance(values[0], list):
            return _flatten_numbers(values[len(values) // 2])
        return _flatten_numbers(values)
    return []


def _extract_matrix(payload: dict | None) -> list[list[float]]:
    if payload is None:
        return []
    candidates = []
    if "values" in payload:
        candidates.append(payload["values"])
    if "fields" in payload and isinstance(payload["fields"], dict):
        candidates.extend(payload["fields"].values())
    for candidate in candidates:
        if isinstance(candidate, list) and candidate and isinstance(candidate[0], list):
            matrix = [[float(value) for value in row if isinstance(value, (float, int)) and math.isfinite(float(value))] for row in candidate]
            if matrix and all(matrix[0] and len(row) == len(matrix[0]) for row in matrix):
                return matrix
    return []


def _extract_vector_field(payload: dict | None) -> tuple[list[list[float]], list[list[float]]] | None:
    if payload is None:
        return None
    fields = payload.get("fields")
    if not isinstance(fields, dict):
        return None
    candidates = [
        ("u", "v"),
        ("U_x", "U_y"),
        ("velocity_x", "velocity_y"),
    ]
    for x_name, y_name in candidates:
        if x_name in fields and y_name in fields:
            u = _matrix_from_value(fields[x_name])
            v = _matrix_from_value(fields[y_name])
            if u and v and len(u) == len(v) and len(u[0]) == len(v[0]):
                return u, v
    velocity = fields.get("U") or fields.get("velocity")
    if isinstance(velocity, list) and velocity and isinstance(velocity[0], list):
        u_rows: list[list[float]] = []
        v_rows: list[list[float]] = []
        for row in velocity:
            if not isinstance(row, list):
                return None
            u_row: list[float] = []
            v_row: list[float] = []
            for item in row:
                if not isinstance(item, list) or len(item) < 2:
                    return None
                u_row.append(float(item[0]))
                v_row.append(float(item[1]))
            u_rows.append(u_row)
            v_rows.append(v_row)
        return u_rows, v_rows
    return None


def _matrix_from_value(value: object) -> list[list[float]]:
    if not isinstance(value, list) or not value or not isinstance(value[0], list):
        return []
    matrix = []
    for row in value:
        if not isinstance(row, list):
            return []
        matrix.append([float(item) for item in row if isinstance(item, (float, int)) and math.isfinite(float(item))])
    return matrix if matrix and all(matrix[0] and len(row) == len(matrix[0]) for row in matrix) else []


def _first_field_name(input: GenerateVisualizationsInput, *, default: str) -> str:
    return input.problem.fields[0].name if input.problem.fields else default


def _render_line_svg(values: list[float], title: str) -> str:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        finite_values = [0.0]
    width = 720
    height = 280
    padding = 36
    min_value = min(finite_values)
    max_value = max(finite_values)
    span = max(max_value - min_value, 1e-12)
    if len(finite_values) == 1:
        points = [(padding, height / 2)]
    else:
        points = []
        for index, value in enumerate(finite_values):
            x = padding + index * (width - 2 * padding) / (len(finite_values) - 1)
            y = height - padding - (value - min_value) * (height - 2 * padding) / span
            points.append((x, y))
    polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="280" viewBox="0 0 720 280">',
            '<rect width="720" height="280" fill="#fbfaf5"/>',
            f'<text x="36" y="28" font-family="Georgia, serif" font-size="16" fill="#1f2933">{title}</text>',
            f'<text x="36" y="260" font-family="Georgia, serif" font-size="12" fill="#52616b">min={min_value:.4g}, max={max_value:.4g}, n={len(finite_values)}</text>',
            f'<polyline fill="none" stroke="#0f766e" stroke-width="2.5" points="{polyline}"/>',
            '</svg>',
        ]
    )


class WriteSimulationReportInput(StrictBaseModel):
    problem: PhysicsProblem
    result: SolverResult
    verification: VerificationReport
    postprocess: PostprocessResult | None = None
    format: Literal["markdown", "pdf", "html", "pptx"] = "markdown"


class WriteSimulationReportOutput(StrictBaseModel):
    report: ArtifactRef
    manifest: ArtifactRef | None = None


def write_simulation_report(input: WriteSimulationReportInput) -> WriteSimulationReportOutput:
    """Generate a local simulation report artifact."""
    report_dir = project_root() / "scratch" / input.problem.id.replace(":", "_") / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    suffix = "md" if input.format == "markdown" else input.format
    report_path = report_dir / f"simulation_report.{suffix}"
    manifest_path = report_dir / "report_manifest.json"
    manifest_payload = _build_report_manifest(input)
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    if input.format == "markdown":
        report_path.write_text(_render_markdown_report(input, manifest_payload, manifest_path), encoding="utf-8")
    else:
        report_path.write_text(
            f"Renderer for {input.format} is not connected yet. Markdown content should be generated first.\n",
            encoding="utf-8",
        )
    report = ArtifactRef(
        uri=str(Path(report_path)),
        kind="simulation_report",
        format=input.format,
        description="Local PhysicsOS simulation report.",
    )
    manifest = ArtifactRef(
        uri=str(Path(manifest_path)),
        kind="simulation_report_manifest",
        format="json",
        description="Machine-readable report manifest with artifacts, verification appendix, and provenance.",
    )
    return WriteSimulationReportOutput(report=report, manifest=manifest)


def _build_report_manifest(input: WriteSimulationReportInput) -> dict:
    postprocess = input.postprocess
    artifact_records = []
    for artifact in [*input.result.artifacts, *((postprocess.visualizations if postprocess else []))]:
        artifact_records.append(
            {
                "kind": artifact.kind,
                "format": artifact.format,
                "uri": artifact.uri,
                "description": artifact.description,
                "exists": Path(artifact.uri).exists(),
            }
        )
    return {
        "schema_version": "physicsos.report_manifest.v1",
        "problem": {
            "id": input.problem.id,
            "domain": input.problem.domain,
            "geometry_id": input.problem.geometry.id,
            "geometry_dimension": input.problem.geometry.dimension,
            "mesh_id": input.problem.mesh.id if input.problem.mesh is not None else None,
            "operators": [operator.model_dump(mode="json") for operator in input.problem.operators],
            "fields": [field.model_dump(mode="json") for field in input.problem.fields],
            "boundary_conditions": [bc.model_dump(mode="json") for bc in input.problem.boundary_conditions],
            "targets": [target.model_dump(mode="json") for target in input.problem.targets],
        },
        "solver": {
            "result_id": input.result.id,
            "backend": input.result.backend,
            "status": input.result.status,
            "runtime": input.result.runtime.model_dump(mode="json"),
            "provenance": input.result.provenance.model_dump(mode="json"),
            "scalar_outputs": input.result.scalar_outputs,
            "residuals": input.result.residuals,
            "uncertainty": input.result.uncertainty,
        },
        "verification": {
            "status": input.verification.status,
            "recommended_next_action": input.verification.recommended_next_action,
            "explanation": input.verification.explanation,
            "residuals": input.verification.residuals,
            "conservation_errors": input.verification.conservation_errors,
            "uncertainty": input.verification.uncertainty,
            "ood_score": input.verification.ood_score,
            "nearest_reference_cases": input.verification.nearest_reference_cases,
            "warnings": input.verification.warnings,
        },
        "postprocess": {
            "kpis": postprocess.kpis if postprocess else {},
            "units": postprocess.units if postprocess else {},
            "recommendations": postprocess.recommendations if postprocess else [],
        },
        "artifacts": artifact_records,
    }


def _render_markdown_report(input: WriteSimulationReportInput, manifest: dict, manifest_path: Path) -> str:
    lines = [
        f"# Simulation Report: {input.problem.id}",
        "",
        "## Executive Summary",
        f"- Domain: `{input.problem.domain}`",
        f"- Solver backend: `{input.result.backend}`",
        f"- Solver status: `{input.result.status}`",
        f"- Verification status: `{input.verification.status}`",
        f"- Recommended next action: `{input.verification.recommended_next_action}`",
        f"- Explanation: {input.verification.explanation}",
        "",
        "## Problem Specification",
        f"- Raw request: {input.problem.user_intent.raw_request}",
        f"- Geometry: `{input.problem.geometry.id}` ({input.problem.geometry.dimension}D, source={input.problem.geometry.source.kind})",
        f"- Mesh: `{input.problem.mesh.id}`" if input.problem.mesh is not None else "- Mesh: not provided",
        f"- Fields: {', '.join(f'{field.name}:{field.kind}' for field in input.problem.fields) or 'none'}",
        f"- Operators: {', '.join(operator.equation_class for operator in input.problem.operators) or 'none'}",
        f"- Boundary conditions: {len(input.problem.boundary_conditions)}",
        f"- Targets: {', '.join(target.name for target in input.problem.targets) or 'none'}",
        "",
        "## Solver Provenance",
        f"- Result id: `{input.result.id}`",
        f"- Created by: `{input.result.provenance.created_by}`",
        f"- Source: `{input.result.provenance.source or 'n/a'}`",
        f"- Version: `{input.result.provenance.version or 'n/a'}`",
        f"- Wall time: {input.result.runtime.wall_time_seconds if input.result.runtime.wall_time_seconds is not None else 'n/a'}",
        "",
        "## KPIs and Scalar Outputs",
    ]
    kpis = input.postprocess.kpis if input.postprocess is not None else {}
    units = input.postprocess.units if input.postprocess is not None else {}
    scalar_keys = sorted(set(input.result.scalar_outputs) | set(kpis))
    if scalar_keys:
        lines.extend(["| Metric | Value | Units |", "| --- | ---: | --- |"])
        for key in scalar_keys:
            value = kpis.get(key, input.result.scalar_outputs.get(key, ""))
            lines.append(f"| `{key}` | {value} | {units.get(key, '')} |")
    else:
        lines.append("- No scalar outputs were reported.")

    lines.extend(["", "## Embedded Visualizations"])
    visualizations = input.postprocess.visualizations if input.postprocess is not None else []
    if visualizations:
        for artifact in visualizations:
            path = Path(artifact.uri)
            if artifact.format in {"png", "svg", "jpg", "jpeg", "webp"}:
                lines.append(f"![{artifact.kind}]({path.as_posix()})")
            elif artifact.format == "json":
                lines.append(f"- `{artifact.kind}`: `{artifact.uri}`")
                preview = _json_preview(path)
                if preview:
                    lines.extend(["", "```json", preview, "```"])
            else:
                lines.append(f"- `{artifact.kind}`: `{artifact.uri}`")
    else:
        lines.append("- No visualization artifacts were generated.")

    lines.extend(["", "## Verification Appendix"])
    lines.extend(_markdown_metric_table("Residuals", input.verification.residuals))
    lines.extend(_markdown_metric_table("Conservation Errors", input.verification.conservation_errors))
    lines.extend(_markdown_metric_table("Uncertainty", input.verification.uncertainty))
    lines.append(f"- OOD score: {input.verification.ood_score}")
    if input.verification.warnings:
        lines.extend(["", "### Warnings"])
        for warning in input.verification.warnings:
            lines.append(f"- {warning}")
    if input.verification.nearest_reference_cases:
        lines.append(f"- Nearest reference cases: {', '.join(input.verification.nearest_reference_cases)}")

    lines.extend(["", "## Artifact Manifest"])
    lines.append(f"- Machine-readable manifest: `{manifest_path}`")
    if manifest["artifacts"]:
        lines.extend(["| Kind | Format | Exists | URI |", "| --- | --- | --- | --- |"])
        for artifact in manifest["artifacts"]:
            lines.append(f"| `{artifact['kind']}` | {artifact['format'] or ''} | {artifact['exists']} | `{artifact['uri']}` |")
    else:
        lines.append("- No artifacts were recorded.")

    recommendations = input.postprocess.recommendations if input.postprocess is not None else []
    lines.extend(["", "## Recommended Next Actions"])
    if recommendations:
        for item in recommendations:
            lines.append(f"- {item}")
    else:
        lines.append(f"- {input.verification.recommended_next_action}")
    return "\n".join(lines) + "\n"


def _markdown_metric_table(title: str, values: dict[str, float]) -> list[str]:
    lines = ["", f"### {title}"]
    if not values:
        lines.append("- Not reported.")
        return lines
    lines.extend(["| Metric | Value |", "| --- | ---: |"])
    for key, value in values.items():
        lines.append(f"| `{key}` | {value:.6g} |")
    return lines


def _json_preview(path: Path, max_chars: int = 1200) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return json.dumps(payload, indent=2)[:max_chars]


for _tool, _input, _output in [
    (extract_kpis, ExtractKPIsInput, ExtractKPIsOutput),
    (plan_postprocess, PostprocessPlanInput, PostprocessPlanOutput),
    (plan_postprocess_structured, PostprocessPlanInput, PostprocessPlanOutput),
    (generate_visualizations, GenerateVisualizationsInput, GenerateVisualizationsOutput),
    (write_simulation_report, WriteSimulationReportInput, WriteSimulationReportOutput),
]:
    _tool.input_model = _input
    _tool.output_model = _output
    _tool.side_effects = "workspace artifacts only"
