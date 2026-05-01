from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

from pydantic import Field

from physicsos.config import project_root
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


def _load_json_artifact(uri: str) -> dict | None:
    try:
        return json.loads(Path(uri).read_text(encoding="utf-8"))
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
    (estimate_uncertainty, EstimateUncertaintyInput, EstimateUncertaintyOutput),
    (detect_ood_case, DetectOODCaseInput, DetectOODCaseOutput),
]:
    _tool.input_model = _input
    _tool.output_model = _output
    _tool.side_effects = "none"
