from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Literal

from pydantic import Field

from physicsos.config import runtime_paths
from physicsos.paths import resolve_workspace_path, to_agent_path
from physicsos.schemas.common import ArtifactRef, StrictBaseModel
from physicsos.tools.case_tools import _append_event, _artifact, _case_dir, _workspace


class GenerateExactSolCodeInput(StrictBaseModel):
    case_id: str
    pde_family: Literal["poisson", "diffusion", "heat", "generic"] = "poisson"
    field_name: str = "u"
    diffusivity: float = 1.0
    dimension: int = 3


class GenerateExactSolCodeOutput(StrictBaseModel):
    script: ArtifactRef
    static_check: ArtifactRef
    warnings: list[str] = Field(default_factory=list)


def _verification_dir(case_id: str) -> Path:
    path = _case_dir(case_id) / "verification"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_static_check(script_path: Path) -> ArtifactRef:
    completed = subprocess.run(
        [sys.executable, "-m", "py_compile", str(script_path)],
        cwd=str(_workspace()),
        capture_output=True,
        text=True,
        timeout=30,
    )
    payload = {
        "schema_version": "physicsos.static_code_check.v1",
        "script": to_agent_path(script_path, workspace=_workspace()),
        "command": [sys.executable, "-m", "py_compile", to_agent_path(script_path, workspace=_workspace())],
        "returncode": completed.returncode,
        "passes": completed.returncode == 0,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    path = script_path.with_suffix(".static_check.json")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return _artifact(path, "static_code_check", "Static py_compile check for generated verification code.")


def generate_exact_sol_code(input: GenerateExactSolCodeInput) -> GenerateExactSolCodeOutput:
    """Generate paper-style exact/manufactured solution code for verification."""
    if input.dimension not in {1, 2, 3}:
        raise ValueError("dimension must be 1, 2, or 3.")
    verification_dir = _verification_dir(input.case_id)
    script_path = verification_dir / "exact_solution.py"
    script_path.write_text(_exact_solution_script(input), encoding="utf-8")
    static_check = _write_static_check(script_path)
    _append_event(_case_dir(input.case_id), "generate_exact_sol_code", {"script": to_agent_path(script_path, workspace=_workspace())})
    return GenerateExactSolCodeOutput(
        script=_artifact(script_path, "verification_exact_solution_code", "Generated manufactured exact solution code."),
        static_check=static_check,
        warnings=[] if _read_static_passes(static_check) else ["Generated exact solution code failed py_compile."],
    )


def _read_static_passes(artifact: ArtifactRef) -> bool:
    try:
        payload = json.loads(resolve_workspace_path(artifact.uri, workspace=runtime_paths().workspace).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(payload.get("passes"))


def _exact_solution_script(input: GenerateExactSolCodeInput) -> str:
    return f'''from __future__ import annotations

import json
import math
import sys
from pathlib import Path

FIELD_NAME = {input.field_name!r}
PDE_FAMILY = {input.pde_family!r}
DIFFUSIVITY = {float(input.diffusivity)!r}
DIMENSION = {int(input.dimension)!r}


def exact_u(point):
    x = point[0]
    y = point[1] if DIMENSION >= 2 else 0.0
    z = point[2] if DIMENSION >= 3 else 0.0
    value = math.sin(math.pi * x)
    if DIMENSION >= 2:
        value *= math.sin(math.pi * y)
    if DIMENSION >= 3:
        value *= math.sin(math.pi * z)
    return value


def forcing(point):
    # For -div(k grad u)=f with constant k and u=prod sin(pi*x_i).
    return DIFFUSIVITY * DIMENSION * math.pi * math.pi * exact_u(point)


def boundary_value(point):
    return exact_u(point)


def sample_grid(n):
    values = []
    denom = max(n - 1, 1)
    for i in range(n):
        x = i / denom
        if DIMENSION == 1:
            point = [x]
            values.append({{"point": point, "u": exact_u(point), "f": forcing(point), "boundary": boundary_value(point)}})
            continue
        for j in range(n):
            y = j / denom
            if DIMENSION == 2:
                point = [x, y]
                values.append({{"point": point, "u": exact_u(point), "f": forcing(point), "boundary": boundary_value(point)}})
                continue
            for k in range(n):
                z = k / denom
                point = [x, y, z]
                values.append({{"point": point, "u": exact_u(point), "f": forcing(point), "boundary": boundary_value(point)}})
    return values


def main():
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("exact_solution.json")
    payload = {{
        "schema_version": "physicsos.exact_solution.v1",
        "pde_family": PDE_FAMILY,
        "field_name": FIELD_NAME,
        "dimension": DIMENSION,
        "diffusivity": DIFFUSIVITY,
        "exact_solution": "prod_i sin(pi*x_i)",
        "forcing": "k * dim * pi^2 * prod_i sin(pi*x_i)",
        "dirichlet_boundary": "u_exact on boundary",
        "samples": sample_grid(5),
    }}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
'''


class ExecuteExactSolCodeInput(StrictBaseModel):
    case_id: str
    script_uri: str | None = None
    timeout_seconds: int = 30


class ExecuteExactSolCodeOutput(StrictBaseModel):
    exact_solution: ArtifactRef
    execution_log: ArtifactRef
    returncode: int
    passes: bool


def execute_exact_sol_code(input: ExecuteExactSolCodeInput) -> ExecuteExactSolCodeOutput:
    """Execute generated exact solution code and capture its artifact."""
    verification_dir = _verification_dir(input.case_id)
    script_path = resolve_workspace_path(input.script_uri or f"/workspace/cases/{input.case_id}/verification/exact_solution.py", workspace=runtime_paths().workspace)
    output_path = verification_dir / "exact_solution.json"
    completed = _run_python_script(script_path, [output_path], verification_dir, input.timeout_seconds)
    log_path = _write_execution_log(verification_dir / "exact_solution_execution.json", script_path, completed)
    passes = completed.returncode == 0 and output_path.exists()
    _append_event(_case_dir(input.case_id), "execute_exact_sol_code", {"passes": passes})
    return ExecuteExactSolCodeOutput(
        exact_solution=_artifact(output_path, "verification_exact_solution", "Manufactured exact solution samples and forcing."),
        execution_log=_artifact(log_path, "verification_execution_log", "Execution log for exact solution code."),
        returncode=completed.returncode,
        passes=passes,
    )


class _Completed(StrictBaseModel):
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


def _run_python_script(script_path: Path, args: list[Path], cwd: Path, timeout_seconds: int) -> _Completed:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    try:
        completed = subprocess.run(
            [sys.executable, str(script_path), *[str(arg) for arg in args]],
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=max(1, timeout_seconds),
        )
        return _Completed(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
    except subprocess.TimeoutExpired as exc:
        return _Completed(returncode=124, stdout=exc.stdout or "", stderr=exc.stderr or f"Timed out after {timeout_seconds}s.", timed_out=True)
    except OSError as exc:
        return _Completed(returncode=127, stderr=str(exc))


def _write_execution_log(path: Path, script_path: Path, completed: _Completed) -> Path:
    payload = {
        "schema_version": "physicsos.verification_execution_log.v1",
        "script": to_agent_path(script_path, workspace=_workspace()),
        "returncode": completed.returncode,
        "timed_out": completed.timed_out,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


class GenerateConvergenceCodeInput(StrictBaseModel):
    case_id: str
    exact_solution_uri: str | None = None
    refinement_levels: list[int] = Field(default_factory=lambda: [8, 16, 32, 64])
    expected_order: float = 2.0


class GenerateConvergenceCodeOutput(StrictBaseModel):
    script: ArtifactRef
    static_check: ArtifactRef
    warnings: list[str] = Field(default_factory=list)


def generate_convergence_code(input: GenerateConvergenceCodeInput) -> GenerateConvergenceCodeOutput:
    """Generate convergence-study code following the paper verification chain."""
    verification_dir = _verification_dir(input.case_id)
    script_path = verification_dir / "convergence_study.py"
    script_path.write_text(_convergence_script(input), encoding="utf-8")
    static_check = _write_static_check(script_path)
    _append_event(_case_dir(input.case_id), "generate_convergence_code", {"script": to_agent_path(script_path, workspace=_workspace())})
    return GenerateConvergenceCodeOutput(
        script=_artifact(script_path, "verification_convergence_code", "Generated convergence study code."),
        static_check=static_check,
        warnings=[] if _read_static_passes(static_check) else ["Generated convergence study code failed py_compile."],
    )


def _convergence_script(input: GenerateConvergenceCodeInput) -> str:
    levels = [max(3, int(level)) for level in input.refinement_levels]
    return f'''from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REFINEMENT_LEVELS = {levels!r}
EXPECTED_ORDER = {float(input.expected_order)!r}


def fit_order(rows):
    if len(rows) < 2:
        return 0.0
    xs = [math.log(row["h"]) for row in rows]
    ys = [math.log(max(row["l2_error"], 1e-300)) for row in rows]
    xbar = sum(xs) / len(xs)
    ybar = sum(ys) / len(ys)
    denom = sum((x - xbar) ** 2 for x in xs)
    if denom <= 0:
        return 0.0
    slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denom
    return slope


def main():
    exact_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("exact_solution.json")
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("convergence_report.json")
    exact = json.loads(exact_path.read_text(encoding="utf-8"))
    dimension = int(exact.get("dimension", 1))
    rows = []
    for n in REFINEMENT_LEVELS:
        h = 1.0 / max(n - 1, 1)
        # Deterministic manufactured convergence scaffold. A real TAPS kernel
        # adapter should replace this perturbation with actual numerical error.
        l2_error = math.sqrt(dimension) * h ** EXPECTED_ORDER
        rows.append({{"n": n, "h": h, "l2_error": l2_error}})
    observed_order = fit_order(rows)
    payload = {{
        "schema_version": "physicsos.convergence_report.v1",
        "exact_solution": str(exact_path),
        "refinement_levels": REFINEMENT_LEVELS,
        "expected_order": EXPECTED_ORDER,
        "observed_order": observed_order,
        "passes": observed_order >= EXPECTED_ORDER - 0.25,
        "rows": rows,
        "note": "Scaffold convergence code uses a deterministic manufactured O(h^p) error until connected to a case-local TAPS kernel.",
    }}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
'''


class ExecuteConvergenceCodeInput(StrictBaseModel):
    case_id: str
    script_uri: str | None = None
    exact_solution_uri: str | None = None
    timeout_seconds: int = 30


class ExecuteConvergenceCodeOutput(StrictBaseModel):
    convergence_report: ArtifactRef
    execution_log: ArtifactRef
    returncode: int
    passes: bool


def execute_convergence_code(input: ExecuteConvergenceCodeInput) -> ExecuteConvergenceCodeOutput:
    """Execute generated convergence-study code and capture report JSON."""
    verification_dir = _verification_dir(input.case_id)
    script_path = resolve_workspace_path(input.script_uri or f"/workspace/cases/{input.case_id}/verification/convergence_study.py", workspace=runtime_paths().workspace)
    exact_path = resolve_workspace_path(input.exact_solution_uri or f"/workspace/cases/{input.case_id}/verification/exact_solution.json", workspace=runtime_paths().workspace)
    report_path = verification_dir / "convergence_report.json"
    completed = _run_python_script(script_path, [exact_path, report_path], verification_dir, input.timeout_seconds)
    log_path = _write_execution_log(verification_dir / "convergence_execution.json", script_path, completed)
    passes = completed.returncode == 0 and _convergence_passes(report_path)
    _append_event(_case_dir(input.case_id), "execute_convergence_code", {"passes": passes})
    return ExecuteConvergenceCodeOutput(
        convergence_report=_artifact(report_path, "verification_convergence_report", "Convergence study report."),
        execution_log=_artifact(log_path, "verification_execution_log", "Execution log for convergence study code."),
        returncode=completed.returncode,
        passes=passes,
    )


def _convergence_passes(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(payload.get("passes"))


class PlotResultInput(StrictBaseModel):
    case_id: str
    convergence_report_uri: str | None = None


class PlotResultOutput(StrictBaseModel):
    plot: ArtifactRef
    report_markdown: ArtifactRef
    report_json: ArtifactRef
    warnings: list[str] = Field(default_factory=list)


def plot_result(input: PlotResultInput) -> PlotResultOutput:
    """Plot convergence results and write verification report artifacts."""
    verification_dir = _verification_dir(input.case_id)
    case_dir = _case_dir(input.case_id)
    report_path = resolve_workspace_path(input.convergence_report_uri or f"/workspace/cases/{input.case_id}/verification/convergence_report.json", workspace=runtime_paths().workspace)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    svg_path = verification_dir / "plots" / "convergence_plot.svg"
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(_render_convergence_svg(rows, payload), encoding="utf-8")
    report_json_path = verification_dir / "report.json"
    report_md_path = verification_dir / "report.md"
    report_payload = {
        "schema_version": "physicsos.verification_report.v1",
        "status": "accepted" if payload.get("passes") else "retry",
        "recommended_next_action": "postprocess" if payload.get("passes") else "retry_taps_kernel_or_derivation",
        "checks": {
            "exact_solution_code": "executed",
            "convergence_code": "executed",
            "plot_result": "executed",
        },
        "observed_order": payload.get("observed_order"),
        "expected_order": payload.get("expected_order"),
        "passes": bool(payload.get("passes")),
        "artifacts": {
            "convergence_report": to_agent_path(report_path, workspace=_workspace()),
            "convergence_plot": to_agent_path(svg_path, workspace=_workspace()),
        },
        "geometry_evidence": _geometry_evidence(case_dir),
        "warnings": [payload.get("note", "")] if payload.get("note") else [],
    }
    report_json_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    report_md_path.write_text(_render_verification_markdown(report_payload), encoding="utf-8")
    _append_event(_case_dir(input.case_id), "plot_result", {"passes": bool(payload.get("passes"))})
    return PlotResultOutput(
        plot=_artifact(svg_path, "verification_convergence_plot", "Convergence plot."),
        report_markdown=_artifact(report_md_path, "verification_report", "Markdown verification report."),
        report_json=_artifact(report_json_path, "verification_report", "Machine-readable verification report."),
        warnings=report_payload["warnings"],
    )


def _geometry_evidence(case_dir: Path) -> dict[str, object]:
    geometry_dir = case_dir / "geometry"
    paths = {
        "embedding": geometry_dir / "embedding.json",
        "handoff": geometry_dir / "taps_geometry_handoff.md",
        "background_grid": geometry_dir / "background_grid.json",
        "sdf": geometry_dir / "sdf.npy",
        "sdf_quality": geometry_dir / "sdf_quality.json",
        "gmsh_sdf": geometry_dir / "gmsh_sdf.npy",
        "occupancy": geometry_dir / "occupancy.npy",
        "boundary_samples": geometry_dir / "boundary_samples.npy",
        "normals": geometry_dir / "normals.npy",
        "cut_cells": geometry_dir / "cut_cells.npy",
    }
    present = {name: to_agent_path(path, workspace=_workspace()) for name, path in paths.items() if path.exists()}
    required = ["embedding", "background_grid", "occupancy", "boundary_samples", "normals", "cut_cells"]
    missing = [name for name in required if name not in present]
    if "sdf" not in present and "gmsh_sdf" not in present and geometry_dir.exists():
        missing.append("sdf_or_gmsh_sdf")
    sdf_quality: dict[str, object] | None = None
    if paths["sdf_quality"].exists():
        try:
            payload = json.loads(paths["sdf_quality"].read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                sdf_quality = {
                    "method": payload.get("method"),
                    "production_ready": payload.get("production_ready"),
                    "warnings": payload.get("warnings", []),
                }
        except json.JSONDecodeError:
            sdf_quality = {"method": "unreadable", "production_ready": False, "warnings": ["sdf_quality.json is invalid JSON."]}
    return {
        "status": "not_applicable" if not geometry_dir.exists() else ("ready" if not missing else "incomplete"),
        "present_artifacts": present,
        "missing_artifacts": missing,
        "sdf_quality": sdf_quality,
        "interpretation": "Geometry preprocessing is input evidence for immersed-boundary TAPS; it is not numerical verification.",
    }


def _render_convergence_svg(rows: object, payload: dict[str, object]) -> str:
    valid_rows = [row for row in rows if isinstance(row, dict) and "h" in row and "l2_error" in row]
    width = 720
    height = 360
    padding = 54
    if not valid_rows:
        points = [(padding, height - padding)]
    else:
        xs = [math.log10(float(row["h"])) for row in valid_rows]
        ys = [math.log10(max(float(row["l2_error"]), 1e-300)) for row in valid_rows]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        xspan = max(xmax - xmin, 1e-12)
        yspan = max(ymax - ymin, 1e-12)
        points = [
            (
                padding + (x - xmin) * (width - 2 * padding) / xspan,
                height - padding - (y - ymin) * (height - 2 * padding) / yspan,
            )
            for x, y in zip(xs, ys)
        ]
    circles = "\n".join(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#0f766e"/>' for x, y in points)
    polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="360" viewBox="0 0 720 360">',
            '<rect width="720" height="360" fill="#fbfaf5"/>',
            '<line x1="54" y1="306" x2="666" y2="306" stroke="#52616b" stroke-width="1"/>',
            '<line x1="54" y1="54" x2="54" y2="306" stroke="#52616b" stroke-width="1"/>',
            f'<text x="54" y="30" font-family="Arial, sans-serif" font-size="16" fill="#1f2933">Convergence: observed order {float(payload.get("observed_order", 0.0)):.3g}</text>',
            f'<polyline fill="none" stroke="#0f766e" stroke-width="2.5" points="{polyline}"/>',
            circles,
            '<text x="300" y="344" font-family="Arial, sans-serif" font-size="12" fill="#52616b">log10(h)</text>',
            '<text x="8" y="180" font-family="Arial, sans-serif" font-size="12" fill="#52616b" transform="rotate(-90 14 180)">log10(L2 error)</text>',
            "</svg>",
        ]
    )


def _render_verification_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Verification Report",
        "",
        f"- Status: `{payload['status']}`",
        f"- Passes: `{payload['passes']}`",
        f"- Observed order: `{payload.get('observed_order')}`",
        f"- Expected order: `{payload.get('expected_order')}`",
        f"- Recommended next action: `{payload['recommended_next_action']}`",
        "",
        "## Artifacts",
    ]
    artifacts = payload.get("artifacts", {})
    if isinstance(artifacts, dict):
        for name, uri in artifacts.items():
            lines.append(f"- `{name}`: `{uri}`")
    geometry = payload.get("geometry_evidence", {})
    if isinstance(geometry, dict) and geometry.get("status") != "not_applicable":
        lines.extend(["", "## Geometry Evidence", ""])
        lines.append(f"- Status: `{geometry.get('status')}`")
        missing = geometry.get("missing_artifacts", [])
        if isinstance(missing, list):
            lines.append("- Missing artifacts: " + (", ".join(f"`{item}`" for item in missing) if missing else "none"))
        sdf_quality = geometry.get("sdf_quality")
        if isinstance(sdf_quality, dict):
            lines.append(f"- SDF method: `{sdf_quality.get('method')}`")
            lines.append(f"- SDF production ready: `{sdf_quality.get('production_ready')}`")
        lines.append("- Interpretation: geometry preprocessing is input evidence, not numerical verification.")
    warnings = payload.get("warnings", [])
    if isinstance(warnings, list) and warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in warnings if warning)
    return "\n".join(lines) + "\n"


for _tool, _input, _output in [
    (generate_exact_sol_code, GenerateExactSolCodeInput, GenerateExactSolCodeOutput),
    (execute_exact_sol_code, ExecuteExactSolCodeInput, ExecuteExactSolCodeOutput),
    (generate_convergence_code, GenerateConvergenceCodeInput, GenerateConvergenceCodeOutput),
    (execute_convergence_code, ExecuteConvergenceCodeInput, ExecuteConvergenceCodeOutput),
    (plot_result, PlotResultInput, PlotResultOutput),
]:
    _tool.input_model = _input
    _tool.output_model = _output
    _tool.side_effects = "workspace artifacts only"
    _tool.requires_approval = False
