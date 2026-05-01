from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

from pydantic import Field

from physicsos.config import project_root
from physicsos.schemas.common import ArtifactRef, StrictBaseModel
from physicsos.schemas.postprocess import PostprocessResult, VisualizationSpec
from physicsos.schemas.problem import PhysicsProblem
from physicsos.schemas.solver import SolverResult
from physicsos.schemas.verification import VerificationReport


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

    solution_payload = _load_solution_payload(input.result)
    values = _extract_plot_values(solution_payload) if solution_payload is not None else []
    if values:
        svg_path = output_dir / "solution_preview.svg"
        svg_path.write_text(_render_line_svg(values, title=f"{input.result.backend} solution preview"), encoding="utf-8")
        artifacts.append(ArtifactRef(uri=str(Path(svg_path)), kind="visualization:solution_preview", format="svg"))

    return GenerateVisualizationsOutput(artifacts=artifacts)


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
            rel = path.name
            if artifact.format == "svg":
                lines.append(f"![{artifact.kind}]({path.as_posix()})")
            else:
                lines.append(f"- `{artifact.kind}`: `{artifact.uri}`")
                preview = _json_preview(path)
                if preview:
                    lines.extend(["", "```json", preview, "```"])
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
    (generate_visualizations, GenerateVisualizationsInput, GenerateVisualizationsOutput),
    (write_simulation_report, WriteSimulationReportInput, WriteSimulationReportOutput),
]:
    _tool.input_model = _input
    _tool.output_model = _output
    _tool.side_effects = "workspace artifacts only"
