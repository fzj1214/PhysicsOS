from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

from pydantic import Field

from physicsos.config import project_root
from physicsos.paths import resolve_workspace_path
from physicsos.schemas.common import ArtifactRef
from physicsos.schemas.problem import PhysicsProblem
from physicsos.schemas.solver import SolverResult
from physicsos.schemas.common import StrictBaseModel


class ComputePhysicsResidualsInput(StrictBaseModel):
    problem: PhysicsProblem
    result: SolverResult


class ComputePhysicsResidualsOutput(StrictBaseModel):
    residuals: dict[str, float] = Field(default_factory=dict)
    normalized_residuals: dict[str, float] = Field(default_factory=dict)
    passes: bool
    artifact: ArtifactRef | None = None


def _residual_threshold(problem: PhysicsProblem) -> float:
    policy = problem.verification_policy
    if policy is not None and policy.residual_tolerance is not None:
        return policy.residual_tolerance
    return 1e-5


def _is_residual_key(key: str) -> bool:
    lowered = key.lower()
    return "residual" in lowered or "error" in lowered or "relative_update" in lowered


def compute_physics_residuals(input: ComputePhysicsResidualsInput) -> ComputePhysicsResidualsOutput:
    """Compute PDE/operator residuals from backend-reported verification metrics."""
    heat_payload = _heat_1d_payload(input.result)
    if heat_payload is not None:
        residuals = _heat_1d_residual_from_payload(heat_payload) or {}
        normalized = {key: value for key, value in residuals.items() if key.startswith("normalized")}
        threshold = _residual_threshold(input.problem)
        passes = bool(normalized) and all(abs(value) <= threshold for value in normalized.values())
        if input.result.status not in {"success", "partial"}:
            passes = False
        artifact = _write_verification_artifact(
            input.problem.id,
            "physics_residuals",
            {
                "residuals": residuals,
                "normalized_residuals": normalized,
                "threshold": threshold,
                "passes": passes,
                "source": "independent_heat_1d_solution_artifact",
            },
        )
        return ComputePhysicsResidualsOutput(passes=passes, residuals=residuals, normalized_residuals=normalized, artifact=artifact)
    residuals = {key: float(value) for key, value in input.result.residuals.items() if _is_residual_key(key)}
    normalized = {
        key: value
        for key, value in residuals.items()
        if key.lower().startswith("normalized") or "relative" in key.lower() or "l2" in key.lower()
    }
    if not normalized and residuals:
        scale = max(abs(value) for value in residuals.values()) + 1e-12
        normalized = {f"normalized_{key}": abs(value) / scale for key, value in residuals.items()}
    threshold = _residual_threshold(input.problem)
    passes = bool(normalized) and all(abs(value) <= threshold for value in normalized.values())
    if input.result.status not in {"success", "partial"}:
        passes = False
    artifact = _write_verification_artifact(
        input.problem.id,
        "physics_residuals",
        {"residuals": residuals, "normalized_residuals": normalized, "threshold": threshold, "passes": passes},
    )
    return ComputePhysicsResidualsOutput(passes=passes, residuals=residuals, normalized_residuals=normalized, artifact=artifact)


class CheckConservationLawsInput(StrictBaseModel):
    problem: PhysicsProblem
    result: SolverResult


class CheckConservationLawsOutput(StrictBaseModel):
    conservation_errors: dict[str, float] = Field(default_factory=dict)
    checked_quantities: list[str] = Field(default_factory=list)
    skipped_quantities: list[str] = Field(default_factory=list)
    passes: bool
    artifact: ArtifactRef


def _write_verification_artifact(problem_id: str, name: str, payload: dict) -> ArtifactRef:
    output_dir = project_root() / "scratch" / problem_id.replace(":", "_") / "verification"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return ArtifactRef(uri=str(Path(path)), kind=f"verification:{name}", format="json")


def _float_metric(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _metric_lookup(result: SolverResult, quantity: str) -> float | None:
    candidates = [
        f"{quantity}_conservation_error",
        f"conservation_error_{quantity}",
        f"{quantity}_imbalance",
        f"{quantity}_residual",
    ]
    combined: dict[str, object] = {**result.scalar_outputs, **result.residuals}
    lowered = {key.lower(): value for key, value in combined.items()}
    for candidate in candidates:
        value = lowered.get(candidate.lower())
        parsed = _float_metric(value)
        if parsed is not None:
            return abs(parsed)
    return None


def check_conservation_laws(input: CheckConservationLawsInput) -> CheckConservationLawsOutput:
    """Check declared conserved quantities against backend-reported imbalance metrics."""
    quantities = sorted({quantity for operator in input.problem.operators for quantity in operator.conserved_quantities})
    errors: dict[str, float] = {}
    skipped: list[str] = []
    for quantity in quantities:
        value = _metric_lookup(input.result, quantity)
        if value is None:
            skipped.append(quantity)
        else:
            errors[quantity] = value
    tolerance = input.problem.verification_policy.conservation_tolerance or 1e-6
    passes = all(value <= tolerance for value in errors.values()) and (bool(errors) or not quantities)
    payload = {
        "checked_quantities": list(errors),
        "skipped_quantities": skipped,
        "conservation_errors": errors,
        "tolerance": tolerance,
        "passes": passes,
        "note": "Quantities without backend imbalance metrics are skipped, not accepted as verified.",
    }
    artifact = _write_verification_artifact(input.problem.id, "conservation_laws", payload)
    return CheckConservationLawsOutput(
        conservation_errors=errors,
        checked_quantities=list(errors),
        skipped_quantities=skipped,
        passes=passes,
        artifact=artifact,
    )


class ValidateSelectedSlicesInput(StrictBaseModel):
    problem: PhysicsProblem
    result: SolverResult
    max_points_per_slice: int = 8


class ValidateSelectedSlicesOutput(StrictBaseModel):
    slice_metrics: dict[str, float | int | str] = Field(default_factory=dict)
    slice_names: list[str] = Field(default_factory=list)
    passes: bool
    artifact: ArtifactRef


class CheckBoundaryConditionApplicationInput(StrictBaseModel):
    problem: PhysicsProblem
    result: SolverResult


class CheckBoundaryConditionApplicationOutput(StrictBaseModel):
    checked_boundaries: list[str] = Field(default_factory=list)
    errors: dict[str, float] = Field(default_factory=dict)
    missing_boundaries: list[str] = Field(default_factory=list)
    passes: bool
    artifact: ArtifactRef


def _load_json_artifact(uri: str) -> dict | None:
    try:
        return json.loads(resolve_workspace_path(uri, workspace=project_root()).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _solution_payload(result: SolverResult) -> dict | None:
    preferred = [
        artifact
        for artifact in result.artifacts
        if artifact.format == "json" and ("solution" in artifact.kind or "reconstruction_metadata" in artifact.kind)
    ]
    for artifact in preferred:
        payload = _load_json_artifact(artifact.uri)
        if payload is not None:
            return payload
    taps_payload = _taps_separated_solution_payload(result)
    if taps_payload is not None:
        return taps_payload
    return None


def _artifact_payload_by_kind(result: SolverResult, kind: str) -> dict | list | None:
    artifact = next((artifact for artifact in result.artifacts if artifact.kind == kind and artifact.format == "json"), None)
    if artifact is None:
        return None
    return _load_json_artifact(artifact.uri)


def _taps_separated_solution_payload(result: SolverResult) -> dict | None:
    if result.backend != "taps:separated_galerkin":
        return None
    factors = _artifact_payload_by_kind(result, "taps_factors")
    axis_operators = _artifact_payload_by_kind(result, "taps_axis_operators")
    samples = _artifact_payload_by_kind(result, "taps_reconstruction_samples")
    if not isinstance(factors, dict) or not isinstance(axis_operators, dict):
        return None
    terms = factors.get("terms")
    axis_order = factors.get("axis_order")
    if not isinstance(terms, list) or not terms or not isinstance(axis_order, list) or not axis_order:
        return None
    primary_axis = str(axis_order[0])
    axis_payload = axis_operators.get(primary_axis)
    if not isinstance(axis_payload, dict):
        return None
    nodes = axis_payload.get("nodes")
    first_term = terms[0]
    if not isinstance(first_term, dict):
        return None
    term_factors = first_term.get("factors")
    if not isinstance(term_factors, dict):
        return None
    values = term_factors.get(primary_axis)
    if not isinstance(nodes, list) or not isinstance(values, list):
        return None
    try:
        points = [float(value) for value in nodes]
        field_values = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    if len(points) != len(field_values):
        return None
    boundary_values_applied = {}
    if field_values:
        boundary_values_applied = {
            "x_min": field_values[0],
            "left": field_values[0],
            "x_max": field_values[-1],
            "right": field_values[-1],
        }
    return {
        "schema_version": "physicsos.solution.v1",
        "backend_id": "taps:separated_galerkin",
        "field": "primary",
        "axis_order": [str(axis) for axis in axis_order],
        "points": points,
        "values": field_values,
        "boundary_values_applied": boundary_values_applied,
        "reconstruction_samples": samples if isinstance(samples, list) else [],
        "source_artifacts": {
            "taps_factors": True,
            "taps_axis_operators": True,
            "taps_reconstruction_samples": isinstance(samples, list),
        },
    }


def _heat_1d_payload(result: SolverResult) -> dict | None:
    payload = _solution_payload(result)
    if payload is None or payload.get("schema_version") != "physicsos.solution.v1":
        return None
    if payload.get("backend_id") != "fdm_heat_1d":
        return None
    if not all(key in payload for key in ("x", "t", "values")):
        return None
    return payload


def _heat_1d_residual_from_payload(payload: dict) -> dict[str, float] | None:
    try:
        x = [float(value) for value in payload["x"]]
        t = [float(value) for value in payload["t"]]
        values = [[float(item) for item in row] for row in payload["values"]]
        alpha = float(payload["coefficient_values_applied"]["thermal_diffusivity"])
        method = str(payload.get("solver_controls_applied", {}).get("method") or "implicit_euler")
    except (KeyError, TypeError, ValueError):
        return None
    if len(x) < 3 or len(t) < 2 or len(values) != len(t):
        return None
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    if dx <= 0 or dt <= 0:
        return None
    residual_sq = 0.0
    scale_sq = 0.0
    count = 0
    for n in range(len(values) - 1):
        old = values[n]
        new = values[n + 1]
        if len(old) != len(x) or len(new) != len(x):
            return None
        for i in range(1, len(x) - 1):
            time_term = (new[i] - old[i]) / dt
            laplace_new = (new[i - 1] - 2.0 * new[i] + new[i + 1]) / (dx * dx)
            if method == "crank_nicolson":
                laplace_old = (old[i - 1] - 2.0 * old[i] + old[i + 1]) / (dx * dx)
                diffusion_term = alpha * 0.5 * (laplace_new + laplace_old)
            else:
                diffusion_term = alpha * laplace_new
            residual = time_term - diffusion_term
            residual_sq += residual * residual
            scale_sq += time_term * time_term + diffusion_term * diffusion_term
            count += 1
    return {
        "rms_pde_residual": math.sqrt(residual_sq / max(count, 1)),
        "normalized_pde_residual": math.sqrt(residual_sq) / (math.sqrt(scale_sq) + 1e-30),
    }


def _boundary_role_from_id(region_id: str) -> str | None:
    lowered = region_id.lower().replace(" ", "").replace("-", "_")
    pieces = [piece for piece in lowered.replace(":", "_").split("_") if piece]
    if (
        lowered in {"x=0", "x_min", "xmin", "x0", "left", "boundary:x_min", "boundary:left"}
        or lowered.endswith(":x_min")
        or "left" in pieces
        or "x0" in pieces
        or ("x" in pieces and "0" in pieces)
    ):
        return "x_min"
    if (
        lowered in {"x=l", "x=1", "x_max", "xmax", "x1", "right", "boundary:x_max", "boundary:right"}
        or lowered.endswith(":x_max")
        or "right" in pieces
        or "x1" in pieces
        or ("x" in pieces and "1" in pieces)
    ):
        return "x_max"
    return None


def _canonical_role(problem: PhysicsProblem, region_id: str, explicit_role: str | None = None) -> str | None:
    if explicit_role is not None:
        return explicit_role
    for boundary in problem.geometry.boundaries:
        if boundary.id == region_id and boundary.role is not None:
            return boundary.role
    return _boundary_role_from_id(region_id)


def _boundary_artifact_key(role: str) -> str | None:
    return {
        "x_min": "left",
        "x_max": "right",
        "y_min": "bottom",
        "y_max": "top",
        "z_min": "front",
        "z_max": "back",
    }.get(role)


def _boundary_artifact_keys(boundary_id: str, region_id: str, role: str | None) -> list[str]:
    keys = [boundary_id, region_id]
    if role:
        keys.append(role)
        legacy = _boundary_artifact_key(role)
        if legacy:
            keys.append(legacy)
    seen: set[str] = set()
    ordered: list[str] = []
    for key in keys:
        if key and key not in seen:
            ordered.append(key)
            seen.add(key)
    return ordered


def _numeric_error(actual: object, expected: object) -> float | None:
    expected_values = _flatten_numbers(expected)
    actual_values = _flatten_numbers(actual)
    if not expected_values or len(expected_values) != len(actual_values):
        return None
    return max(abs(actual_value - expected_value) for actual_value, expected_value in zip(actual_values, expected_values))


def check_boundary_condition_application(input: CheckBoundaryConditionApplicationInput) -> CheckBoundaryConditionApplicationOutput:
    """Compare applied solver boundary values against the locked PhysicsProblem."""
    payload = _solution_payload(input.result) or {}
    applied = payload.get("boundary_values_applied")
    checked: list[str] = []
    errors: dict[str, float] = {}
    missing: list[str] = []
    heat_payload = _heat_1d_payload(input.result)
    if heat_payload is not None:
        try:
            rows = heat_payload["values"]
            by_role = {"x_min": [row[0] for row in rows], "x_max": [row[-1] for row in rows]}
        except (KeyError, TypeError, IndexError):
            by_role = {}
        for boundary in input.problem.boundary_conditions:
            if boundary.kind != "dirichlet":
                continue
            role = _canonical_role(input.problem, boundary.region_id, boundary.boundary_role)
            actual_values = by_role.get(role or "")
            expected = _flatten_numbers(boundary.value)
            if not actual_values or not expected:
                missing.append(boundary.id)
                continue
            checked.append(boundary.id)
            error = max(abs(float(actual) - expected[0]) for actual in actual_values)
            if error > 1e-9:
                errors[boundary.id] = error
    elif not isinstance(applied, dict):
        missing = [boundary.id for boundary in input.problem.boundary_conditions if boundary.kind == "dirichlet"]
    else:
        for boundary in input.problem.boundary_conditions:
            if boundary.kind != "dirichlet":
                continue
            role = _canonical_role(input.problem, boundary.region_id, boundary.boundary_role)
            key = next((candidate for candidate in _boundary_artifact_keys(boundary.id, boundary.region_id, role) if candidate in applied), None)
            if key is None:
                missing.append(boundary.id)
                continue
            error = _numeric_error(applied.get(key), boundary.value)
            if error is None:
                missing.append(boundary.id)
                continue
            checked.append(boundary.id)
            if error > 1e-9:
                errors[boundary.id] = error
    passes = not errors and not missing and bool(checked)
    artifact = _write_verification_artifact(
        input.problem.id,
        "boundary_condition_application",
        {
            "result_id": input.result.id,
            "backend": input.result.backend,
            "applied_boundary_values": applied if isinstance(applied, dict) else None,
            "checked_boundaries": checked,
            "errors": errors,
            "missing_boundaries": missing,
            "passes": passes,
        },
    )
    return CheckBoundaryConditionApplicationOutput(
        checked_boundaries=checked,
        errors=errors,
        missing_boundaries=missing,
        passes=passes,
        artifact=artifact,
    )


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


def _finite_stats(values: list[float]) -> dict[str, float | int]:
    finite = [value for value in values if math.isfinite(value)]
    if not values:
        return {"count": 0, "finite_count": 0}
    return {
        "count": len(values),
        "finite_count": len(finite),
        "min": min(finite) if finite else float("nan"),
        "max": max(finite) if finite else float("nan"),
        "mean": sum(finite) / len(finite) if finite else float("nan"),
    }


def _sample(values: list[float], max_points: int) -> list[float]:
    if len(values) <= max_points:
        return values
    step = max(1, (len(values) - 1) // (max_points - 1))
    sampled = [values[index] for index in range(0, len(values), step)][: max_points - 1]
    sampled.append(values[-1])
    return sampled


def validate_selected_slices(input: ValidateSelectedSlicesInput) -> ValidateSelectedSlicesOutput:
    """Validate representative field slices for finite values and write a slice summary artifact."""
    payload = _solution_payload(input.result)
    slices: dict[str, dict] = {}
    metrics: dict[str, float | int | str] = {}
    if payload is None:
        metrics["status"] = "missing_solution_artifact"
    elif "fields" in payload and isinstance(payload["fields"], dict):
        for field, values in payload["fields"].items():
            flattened = _flatten_numbers(values)
            stats = _finite_stats(flattened)
            slices[f"field:{field}:global"] = {**stats, "sample": _sample(flattened, input.max_points_per_slice)}
    elif "points" in payload and "values" in payload:
        values = _flatten_numbers(payload["values"])
        stats = _finite_stats(values)
        slices["mesh_nodes:global"] = {**stats, "sample": _sample(values, input.max_points_per_slice)}
    elif "values" in payload and isinstance(payload["values"], list):
        values = payload["values"]
        flattened = _flatten_numbers(values)
        stats = _finite_stats(flattened)
        slices["field:global"] = {**stats, "sample": _sample(flattened, input.max_points_per_slice)}
        if values and isinstance(values[0], list):
            mid_i = len(values) // 2
            mid_j = len(values[0]) // 2 if values[0] else 0
            row = _flatten_numbers(values[mid_i])
            column = _flatten_numbers([row_values[mid_j] for row_values in values if isinstance(row_values, list) and len(row_values) > mid_j])
            slices["field:mid_x_row"] = {**_finite_stats(row), "sample": _sample(row, input.max_points_per_slice)}
            slices["field:mid_y_column"] = {**_finite_stats(column), "sample": _sample(column, input.max_points_per_slice)}
    else:
        numbers = _flatten_numbers(payload)
        slices["metadata:numeric"] = {**_finite_stats(numbers), "sample": _sample(numbers, input.max_points_per_slice)}

    for name, stats in slices.items():
        metrics[f"{name}:finite_fraction"] = float(stats.get("finite_count", 0)) / (float(stats.get("count", 0)) + 1e-12)
    passes = bool(slices) and all(value >= 0.999999 for key, value in metrics.items() if key.endswith(":finite_fraction"))
    artifact = _write_verification_artifact(
        input.problem.id,
        "selected_slices",
        {"result_id": input.result.id, "backend": input.result.backend, "slices": slices, "metrics": metrics, "passes": passes},
    )
    return ValidateSelectedSlicesOutput(slice_metrics=metrics, slice_names=list(slices), passes=passes, artifact=artifact)


class EstimateUncertaintyInput(StrictBaseModel):
    problem: PhysicsProblem
    result: SolverResult
    method: Literal["ensemble", "dropout", "conformal", "residual_proxy", "backend_reported"] = "backend_reported"


class EstimateUncertaintyOutput(StrictBaseModel):
    uncertainty: dict[str, float] = Field(default_factory=dict)
    confidence: float


def estimate_uncertainty(input: EstimateUncertaintyInput) -> EstimateUncertaintyOutput:
    """Estimate predictive uncertainty for fields and KPIs."""
    if input.method == "backend_reported" and input.result.uncertainty:
        max_uncertainty = max(abs(value) for value in input.result.uncertainty.values())
        confidence = max(0.0, min(1.0, 1.0 / (1.0 + max_uncertainty)))
        return EstimateUncertaintyOutput(uncertainty=input.result.uncertainty, confidence=confidence)
    if input.method in {"residual_proxy", "backend_reported"}:
        residual_values = [abs(value) for key, value in input.result.residuals.items() if _is_residual_key(key)]
        proxy = max(residual_values) if residual_values else 1.0
        uncertainty = {"residual_proxy": proxy}
        confidence = max(0.0, min(1.0, 1.0 / (1.0 + proxy)))
        return EstimateUncertaintyOutput(uncertainty=uncertainty, confidence=confidence)
    return EstimateUncertaintyOutput(uncertainty=input.result.uncertainty, confidence=0.5 if input.result.uncertainty else 0.0)


class DetectOODCaseInput(StrictBaseModel):
    problem: PhysicsProblem
    reference_scope: Literal["model_training_set", "case_memory", "both"] = "both"


class DetectOODCaseOutput(StrictBaseModel):
    ood_score: float
    reasons: list[str] = Field(default_factory=list)
    nearest_cases: list[str] = Field(default_factory=list)


def detect_ood_case(input: DetectOODCaseInput) -> DetectOODCaseOutput:
    """Detect out-of-distribution geometry, regime, material, or boundary conditions."""
    problem = input.problem
    score = 0.05
    reasons: list[str] = []
    if problem.domain == "custom":
        score += 0.10
        reasons.append("custom physics domain has no fixed training distribution.")
    if not problem.geometry.boundaries and not problem.boundary_conditions:
        score += 0.30
        reasons.append("boundary labels and boundary conditions are missing.")
    elif not problem.geometry.boundaries:
        score += 0.10
        reasons.append("geometry boundary labels are not attached to GeometrySpec.")
    if problem.geometry.source.kind in {"text", "image"}:
        score += 0.20
        reasons.append(f"geometry source kind '{problem.geometry.source.kind}' needs reconstruction before trusted solve.")
    if problem.domain in {"fluid", "thermal", "solid", "electromagnetic"} and not problem.materials:
        score += 0.15
        reasons.append("material properties are missing for a standard physics domain.")
    if problem.domain == "fluid" and not any(operator.nondimensional_numbers for operator in problem.operators):
        score += 0.15
        reasons.append("fluid regime has no nondimensional numbers such as Reynolds/Mach/Grashof.")
    if not problem.geometry.encodings:
        score += 0.05
        reasons.append("no solver/surrogate-ready geometry encoding is attached.")
    if any(operator.equation_class in {"unknown", "custom"} for operator in problem.operators):
        score += 0.20
        reasons.append("operator equation class is underspecified.")
    if not reasons:
        reasons.append("case is in-distribution for local deterministic checks.")
    return DetectOODCaseOutput(ood_score=min(1.0, score), reasons=reasons, nearest_cases=[])


for _tool, _input, _output in [
    (compute_physics_residuals, ComputePhysicsResidualsInput, ComputePhysicsResidualsOutput),
    (check_conservation_laws, CheckConservationLawsInput, CheckConservationLawsOutput),
    (validate_selected_slices, ValidateSelectedSlicesInput, ValidateSelectedSlicesOutput),
    (check_boundary_condition_application, CheckBoundaryConditionApplicationInput, CheckBoundaryConditionApplicationOutput),
    (estimate_uncertainty, EstimateUncertaintyInput, EstimateUncertaintyOutput),
    (detect_ood_case, DetectOODCaseInput, DetectOODCaseOutput),
]:
    _tool.input_model = _input
    _tool.output_model = _output
    _tool.side_effects = "none"
