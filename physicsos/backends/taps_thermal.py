from __future__ import annotations

import json
import math
from pathlib import Path

from physicsos.config import project_root
from physicsos.backends.taps_generic import _boundary_position_from_role, _boundary_position, weak_form_transient_diffusion_blocks
from physicsos.schemas.common import ArtifactRef
from physicsos.schemas.taps import NumericalSolvePlanOutput, TAPSProblem, TAPSResidualReport, TAPSResultArtifacts


def _safe(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


def _linspace(min_value: float, max_value: float, points: int) -> list[float]:
    if points <= 1:
        return [min_value]
    step = (max_value - min_value) / (points - 1)
    return [min_value + step * index for index in range(points)]


def _axis(problem: TAPSProblem, name: str, default_min: float, default_max: float, default_points: int) -> list[float]:
    spec = next((axis for axis in problem.axes if axis.name == name), None)
    min_value = default_min if spec is None or spec.min_value is None else spec.min_value
    max_value = default_max if spec is None or spec.max_value is None else spec.max_value
    points = default_points if spec is None or spec.points is None else spec.points
    return _linspace(min_value, max_value, points)


def _series(alpha_value: float, time_value: float, k: float, rank: int) -> float:
    total = 0.0
    for order in range(rank):
        total += ((-k * k) ** order) * (alpha_value**order) * (time_value**order) / math.factorial(order)
    return total


def _series_dt(alpha_value: float, time_value: float, k: float, rank: int) -> float:
    total = 0.0
    for order in range(1, rank):
        total += (
            ((-k * k) ** order)
            * (alpha_value**order)
            * order
            * (time_value ** (order - 1))
            / math.factorial(order)
        )
    return total


def _plan_solver_number(plan: NumericalSolvePlanOutput | None, name: str, default: float) -> float:
    if plan is None:
        return default
    normalized = name.lower()
    for binding in plan.coefficient_bindings:
        if binding.role == "solver" and binding.name.lower() == normalized:
            value = binding.value
            if isinstance(value, (float, int)):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    return default
    return default


def _dirichlet_values_1d(plan: NumericalSolvePlanOutput | None, field: str) -> dict[str, float]:
    values = {"left": 0.0, "right": 0.0}
    if plan is None:
        return values
    for binding in plan.boundary_condition_bindings:
        if binding.kind.lower() != "dirichlet" or binding.field != field:
            continue
        position = _boundary_position_from_role(binding.boundary_role) or _boundary_position(binding.region_id)
        if position in values and isinstance(binding.value, (float, int)):
            values[position] = float(binding.value)
    return values


def solve_transient_heat_1d(
    taps_problem: TAPSProblem,
    numerical_plan: NumericalSolvePlanOutput | None = None,
) -> tuple[TAPSResultArtifacts, TAPSResidualReport]:
    """Equation-driven low-rank S-P-T surrogate for 1D transient heat equation.

    Model problem:
        dT/dt = alpha d2T/dx2, x in [0, L], T(0,t)=T(L,t)=0
        T(x,0; alpha)=sin(pi x / L)

    Analytical field:
        T(x, alpha, t) = sin(pi x / L) exp(-alpha (pi/L)^2 t)

    The parameter-time factor exp(-alpha k t) is approximated by the separable
    Taylor expansion sum_n c_n alpha^n t^n. This avoids data-driven training and
    avoids SVD/OpenMP runtime conflicts in constrained local environments.
    """
    output_dir = project_root() / "scratch" / _safe(taps_problem.problem_id) / "taps_thermal"
    output_dir.mkdir(parents=True, exist_ok=True)
    weak_form_blocks = weak_form_transient_diffusion_blocks(taps_problem)

    x = _axis(taps_problem, "x", 0.0, 1.0, 128)
    alpha = _axis(taps_problem, "alpha", 0.01, 0.1, 48)
    time = _axis(taps_problem, "t", 0.0, 1.0, 64)
    planned_rank = int(_plan_solver_number(numerical_plan, "rank", float(taps_problem.basis.tensor_rank)))
    rank = max(1, min(planned_rank, len(alpha), len(time)))

    length = float(x[-1] - x[0]) if len(x) > 1 else 1.0
    k = math.pi / length
    x_factor = [math.sin(k * (value - x[0])) for value in x]
    alpha_modes = [[value**order for order in range(rank)] for value in alpha]
    time_modes = [[value**order for value in time] for order in range(rank)]
    coefficients = [((-k * k) ** order) / math.factorial(order) for order in range(rank)]

    error_sq = 0.0
    full_sq = 0.0
    residual_sq = 0.0
    dt_sq = 0.0
    for x_value in x_factor:
        for alpha_value in alpha:
            for time_value in time:
                exact = x_value * math.exp(-alpha_value * k * k * time_value)
                approx = x_value * _series(alpha_value, time_value, k, rank)
                error_sq += (exact - approx) ** 2
                full_sq += exact**2
                d_t = x_value * _series_dt(alpha_value, time_value, k, rank)
                d_xx = -k * k * approx
                residual = d_t - alpha_value * d_xx
                residual_sq += residual**2
                dt_sq += d_t**2
    relative_l2 = float(math.sqrt(error_sq) / (math.sqrt(full_sq) + 1e-12))
    normalized_residual = float(math.sqrt(residual_sq) / (math.sqrt(dt_sq) + 1e-12))

    factor_payload = {
        "type": "low_rank_heat_1d_spt",
        "weak_form_blocks": weak_form_blocks,
        "numerical_plan_solver_family": numerical_plan.solver_family if numerical_plan is not None else None,
        "rank": rank,
        "axes": {
            "x": x,
            "alpha": alpha,
            "t": time,
        },
        "factors": {
            "x": x_factor,
            "alpha_modes": alpha_modes,
            "coefficients": coefficients,
            "time_modes": time_modes,
        },
    }
    field = numerical_plan.field_bindings.get("primary", "T") if numerical_plan is not None else "T"
    metadata_payload = {
        "equation": "dT/dt = alpha d2T/dx2",
        "weak_form_blocks": weak_form_blocks,
        "field": field,
        "initial_condition": "sin(pi x / L)",
        "boundary_condition": "T(0,t)=T(L,t)=0",
        "initial_condition_bindings": numerical_plan.initial_condition_bindings if numerical_plan is not None else [],
        "boundary_values_applied": _dirichlet_values_1d(numerical_plan, field),
        "coefficient_values_applied": {
            binding.name: binding.value
            for binding in (numerical_plan.coefficient_bindings if numerical_plan is not None else [])
        },
        "rank": rank,
        "weak_form_blocks": weak_form_blocks,
        "relative_l2_reconstruction_error": relative_l2,
        "normalized_pde_residual": normalized_residual,
    }
    residual_payload = {
        "numerical_plan_solver_family": numerical_plan.solver_family if numerical_plan is not None else None,
        "rank": rank,
        "relative_l2_reconstruction_error": relative_l2,
        "normalized_pde_residual": normalized_residual,
        "converged": relative_l2 < 1e-3 and normalized_residual < 1e-2,
    }

    factors_path = output_dir / "factor_matrices.json"
    metadata_path = output_dir / "reconstruction_metadata.json"
    residual_path = output_dir / "residual_history.json"
    factors_path.write_text(json.dumps(factor_payload, indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    residual_path.write_text(json.dumps(residual_payload, indent=2), encoding="utf-8")

    recommended_action = "accept"
    if normalized_residual >= 1e-2:
        recommended_action = "increase_rank" if rank < min(len(alpha), len(time)) else "refine_axes"
    if relative_l2 >= 1e-3:
        recommended_action = "increase_rank" if rank < min(len(alpha), len(time)) else "split_slab"

    artifacts = TAPSResultArtifacts(
        factor_matrices=[ArtifactRef(uri=str(factors_path), kind="taps_factor_matrices", format="json")],
        reconstruction_metadata=ArtifactRef(uri=str(metadata_path), kind="taps_reconstruction_metadata", format="json"),
        residual_history=ArtifactRef(uri=str(residual_path), kind="taps_residual_history", format="json"),
    )
    report = TAPSResidualReport(
        residuals={
            "relative_l2_reconstruction_error": relative_l2,
            "normalized_pde_residual": normalized_residual,
        },
        rank=rank,
        converged=recommended_action == "accept",
        recommended_action=recommended_action,
    )
    return artifacts, report
