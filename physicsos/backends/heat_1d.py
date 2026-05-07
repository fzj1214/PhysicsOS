from __future__ import annotations

import json
import math
from pathlib import Path
from time import perf_counter

from physicsos.config import project_root
from physicsos.schemas.common import ArtifactRef, Provenance, RuntimeStats
from physicsos.schemas.problem import PhysicsProblem
from physicsos.schemas.solver import SolverResult


def _safe(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


def _numeric(value: object) -> float | None:
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


def _parameter(problem: PhysicsProblem, names: set[str], default: float | None = None) -> float | None:
    normalized = {name.lower() for name in names}
    for parameter in problem.parameters:
        if parameter.name.lower() in normalized:
            value = _numeric(parameter.value)
            if value is not None:
                return value
    return default


def _material_property(problem: PhysicsProblem, names: set[str]) -> tuple[float | None, str | None, str | None]:
    normalized = {name.lower() for name in names}
    for material in problem.materials:
        for prop in material.properties:
            if prop.name.lower() in normalized:
                value = _numeric(prop.value)
                if value is not None:
                    return value, prop.name, prop.units
    return None, None, None


def heat_1d_supports(problem: PhysicsProblem) -> bool:
    equation_classes = {operator.equation_class.lower() for operator in problem.operators}
    has_numeric_initial = any(ic.field == "T" and _numeric(ic.value) is not None for ic in problem.initial_conditions)
    try:
        _alpha(problem)
    except ValueError:
        has_alpha = False
    else:
        has_alpha = True
    try:
        _length(problem)
        _final_time(problem)
    except ValueError:
        has_time_and_length = False
    else:
        has_time_and_length = True
    return (
        problem.domain == "thermal"
        and problem.geometry.dimension == 1
        and bool(equation_classes & {"heat", "diffusion", "thermal_diffusion"})
        and has_numeric_initial
        and has_alpha
        and has_time_and_length
        and len([bc for bc in problem.boundary_conditions if bc.kind == "dirichlet"]) >= 2
    )


def _length(problem: PhysicsProblem) -> float:
    parameter_value = _parameter(problem, {"length", "l"}, None)
    if parameter_value is not None and parameter_value > 0:
        return parameter_value
    for entity in problem.geometry.entities:
        value = entity.metadata.get("length")
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    raise ValueError("1D heat solver requires explicit rod length in parameters or geometry entity metadata.")


def _final_time(problem: PhysicsProblem) -> float:
    value = _parameter(problem, {"final_time", "end_time", "t_final", "simulation_time", "time"}, None)
    if value is not None and value > 0:
        return value
    raise ValueError("1D heat solver requires explicit final_time/simulation_time parameter.")


def _alpha(problem: PhysicsProblem) -> tuple[float, dict[str, object]]:
    direct, source, units = _material_property(problem, {"thermal_diffusivity", "alpha", "diffusivity", "d"})
    if direct is not None and direct > 0:
        return direct, {"thermal_diffusivity": direct, "source_name": source, "units": units}
    k, k_name, k_units = _material_property(problem, {"thermal_conductivity", "k"})
    rho, rho_name, rho_units = _material_property(problem, {"density", "rho"})
    cp, cp_name, cp_units = _material_property(problem, {"specific_heat", "heat_capacity", "cp"})
    if k is not None and rho is not None and cp is not None and rho > 0 and cp > 0:
        alpha = k / (rho * cp)
        return alpha, {
            "thermal_diffusivity": alpha,
            "derived_from": {
                "thermal_conductivity": {"value": k, "source_name": k_name, "units": k_units},
                "density": {"value": rho, "source_name": rho_name, "units": rho_units},
                "specific_heat": {"value": cp, "source_name": cp_name, "units": cp_units},
            },
        }
    raise ValueError("1D heat solver requires thermal_diffusivity or thermal_conductivity+density+specific_heat.")


def _role(boundary_id: str, region_id: str, boundary_role: str | None) -> str | None:
    if boundary_role in {"x_min", "x_max"}:
        return boundary_role
    lowered = f"{boundary_id} {region_id}".lower().replace("=", "_").replace("-", "_")
    pieces = [piece for piece in lowered.replace(":", "_").split("_") if piece]
    if any(piece in pieces for piece in ["left", "x0"]) or ("x" in pieces and "0" in pieces):
        return "x_min"
    if any(piece in pieces for piece in ["right", "xl", "x1"]) or ("x" in pieces and ("l" in pieces or "1" in pieces)):
        return "x_max"
    return None


def _dirichlet_temperatures(problem: PhysicsProblem) -> tuple[float, float, dict[str, float]]:
    values: dict[str, float] = {}
    for bc in problem.boundary_conditions:
        if bc.kind != "dirichlet" or bc.field != "T":
            continue
        value = _numeric(bc.value)
        if value is None:
            continue
        role = _role(bc.id, bc.region_id, bc.boundary_role)
        if role == "x_min":
            values["left"] = value
        elif role == "x_max":
            values["right"] = value
    if "left" not in values or "right" not in values:
        raise ValueError("1D heat solver requires left/right Dirichlet temperatures.")
    return values["left"], values["right"], values


def _initial_temperature(problem: PhysicsProblem) -> float:
    for ic in problem.initial_conditions:
        if ic.field == "T":
            value = _numeric(ic.value)
            if value is not None:
                return value
    raise ValueError("1D heat solver requires a numeric uniform initial temperature.")


def _linspace(start: float, stop: float, count: int) -> list[float]:
    if count <= 1:
        return [start]
    step = (stop - start) / (count - 1)
    return [start + step * index for index in range(count)]


def _solve_tridiagonal(lower: list[float], diag: list[float], upper: list[float], rhs: list[float]) -> list[float]:
    n = len(diag)
    c = upper[:]
    d = rhs[:]
    b = diag[:]
    for i in range(1, n):
        factor = lower[i - 1] / b[i - 1]
        b[i] -= factor * c[i - 1]
        d[i] -= factor * d[i - 1]
    x = [0.0 for _ in range(n)]
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x


def _pde_residual(values: list[list[float]], alpha: float, dx: float, dt: float) -> tuple[float, float]:
    residual_sq = 0.0
    scale_sq = 0.0
    count = 0
    for n in range(len(values) - 1):
        old = values[n]
        new = values[n + 1]
        for i in range(1, len(new) - 1):
            time_term = (new[i] - old[i]) / dt
            laplace_new = (new[i - 1] - 2.0 * new[i] + new[i + 1]) / (dx * dx)
            residual = time_term - alpha * laplace_new
            residual_sq += residual * residual
            scale_sq += time_term * time_term + (alpha * laplace_new) * (alpha * laplace_new)
            count += 1
    rms = math.sqrt(residual_sq / max(count, 1))
    normalized = math.sqrt(residual_sq) / (math.sqrt(scale_sq) + 1e-30)
    return rms, normalized


def run_heat_1d_solver(problem: PhysicsProblem, *, nx: int = 101, nt: int | None = None) -> SolverResult:
    """Run a production 1D implicit-Euler transient heat solve for numeric rod cases."""
    started = perf_counter()
    if not heat_1d_supports(problem):
        raise ValueError("Problem is outside fdm_heat_1d support scope.")
    length = _length(problem)
    final_time = _final_time(problem)
    alpha, coefficient_values = _alpha(problem)
    left, right, boundary_values = _dirichlet_temperatures(problem)
    initial = _initial_temperature(problem)
    nx = max(11, int(nx))
    dx = length / (nx - 1)
    if nt is None:
        # Accuracy-oriented default. Implicit Euler is unconditionally stable,
        # but smaller dt keeps the verification residual and transient curve useful.
        nt = max(20, min(2000, int(math.ceil(final_time / max(0.01, 0.2 * dx * dx / max(alpha, 1e-30))))))
    nt = max(1, int(nt))
    dt = final_time / nt
    x = _linspace(0.0, length, nx)
    time = _linspace(0.0, final_time, nt + 1)
    current = [initial for _ in range(nx)]
    current[0] = left
    current[-1] = right
    values = [current[:]]
    interior = nx - 2
    r = alpha * dt / (dx * dx)
    lower = [-r for _ in range(max(0, interior - 1))]
    diag = [1.0 + 2.0 * r for _ in range(interior)]
    upper = [-r for _ in range(max(0, interior - 1))]
    for _ in range(nt):
        rhs = current[1:-1]
        if rhs:
            rhs[0] += r * left
            rhs[-1] += r * right
            interior_solution = _solve_tridiagonal(lower, diag, upper, rhs)
            current = [left, *interior_solution, right]
        values.append(current[:])
    rms_residual, normalized_residual = _pde_residual(values, alpha, dx, dt)
    boundary_error = max(abs(row[0] - left) for row in values) + max(abs(row[-1] - right) for row in values)
    initial_error = max(abs(values[0][i] - initial) for i in range(1, nx - 1))
    min_allowed = min(left, right, initial)
    max_allowed = max(left, right, initial)
    flattened = [value for row in values for value in row]
    range_violation = max(0.0, min_allowed - min(flattened), max(flattened) - max_allowed)
    output_dir = project_root() / "scratch" / _safe(problem.id) / "fdm_heat_1d"
    output_dir.mkdir(parents=True, exist_ok=True)
    solution_path = output_dir / "solution.json"
    residual_path = output_dir / "residual_check.json"
    solution_payload = {
        "schema_version": "physicsos.solution.v1",
        "problem_id": problem.id,
        "backend_id": "fdm_heat_1d",
        "field": "T",
        "x": x,
        "t": time,
        "values": values,
        "units": {"x": "m", "t": "s", "T": problem.fields[0].units or "K"},
        "boundary_values_applied": {"left": left, "right": right, "x_min": left, "x_max": right},
        "initial_values_applied": {"T": initial, "uniform": initial},
        "coefficient_values_applied": coefficient_values,
        "solver_controls_applied": {"method": "implicit_euler", "nx": nx, "nt": nt, "dx": dx, "dt": dt},
    }
    residual_payload = {
        "schema_version": "physicsos.verification.heat_1d.v1",
        "problem_id": problem.id,
        "backend_id": "fdm_heat_1d",
        "rms_pde_residual": rms_residual,
        "normalized_pde_residual": normalized_residual,
        "boundary_condition_error": boundary_error,
        "initial_condition_error": initial_error,
        "range_violation": range_violation,
        "passes": normalized_residual <= 1e-8 and boundary_error <= 1e-12 and initial_error <= 1e-12 and range_violation <= 1e-12,
    }
    solution_path.write_text(json.dumps(solution_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    residual_path.write_text(json.dumps(residual_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts = [
        ArtifactRef(uri=str(Path(solution_path)), kind="solution:heat_1d", format="json", description="Canonical 1D heat solution artifact."),
        ArtifactRef(uri=str(Path(residual_path)), kind="verification:heat_1d_residual", format="json"),
    ]
    return SolverResult(
        id=f"result:{_safe(problem.id)}:fdm_heat_1d",
        problem_id=problem.id,
        backend="fdm_heat_1d",
        status="success" if residual_payload["passes"] else "needs_review",
        scalar_outputs={
            "capability_status": "production",
            "support_scope": "1D transient heat/diffusion with numeric uniform IC, numeric left/right Dirichlet BCs, and scalar thermal diffusivity",
            "verification_methods": "independent_discrete_residual,boundary_values_from_solution,initial_condition_from_solution,maximum_principle_range",
            "max_temperature": max(flattened),
            "min_temperature": min(flattened),
            "final_center_temperature": values[-1][nx // 2],
            "material_source_confidence": 0.8,
        },
        residuals={
            "rms_pde_residual": rms_residual,
            "normalized_pde_residual": normalized_residual,
            "boundary_condition_error": boundary_error,
            "initial_condition_error": initial_error,
            "range_violation": range_violation,
        },
        runtime=RuntimeStats(wall_time_seconds=perf_counter() - started),
        artifacts=artifacts,
        provenance=Provenance(created_by="run_heat_1d_solver", source="fdm_heat_1d"),
    )
