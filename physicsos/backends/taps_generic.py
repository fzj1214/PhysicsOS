from __future__ import annotations

import json
import math
from pathlib import Path
from functools import lru_cache

from physicsos.config import project_root
from physicsos.schemas.common import ArtifactRef
from physicsos.schemas.taps import TAPSProblem, TAPSResidualReport, TAPSResultArtifacts


SUPPORTED_FAMILIES = {
    "poisson",
    "diffusion",
    "thermal_diffusion",
    "reaction_diffusion",
    "coupled_reaction_diffusion",
    "helmholtz",
}


def _safe(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


def _linspace(min_value: float, max_value: float, points: int) -> list[float]:
    if points <= 1:
        return [min_value]
    step = (max_value - min_value) / (points - 1)
    return [min_value + step * index for index in range(points)]


def _space_axes(problem: TAPSProblem) -> list[str]:
    return [axis.name for axis in problem.axes if axis.kind == "space"]


def _axis_values(problem: TAPSProblem, name: str, default_points: int = 64) -> list[float]:
    axis = next((candidate for candidate in problem.axes if candidate.name == name), None)
    if axis is None:
        return _linspace(0.0, 1.0, default_points)
    return _linspace(axis.min_value if axis.min_value is not None else 0.0, axis.max_value if axis.max_value is not None else 1.0, axis.points or default_points)


def _load_occupancy_mask(problem: TAPSProblem, nx: int, ny: int) -> list[list[int]] | None:
    encoding = next((item for item in problem.geometry_encodings if item.kind == "occupancy_mask"), None)
    if encoding is None:
        return None
    path = Path(encoding.uri)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_mask = payload.get("mask")
    if not isinstance(raw_mask, list) or not raw_mask:
        return None
    source_nx = len(raw_mask)
    source_ny = len(raw_mask[0]) if isinstance(raw_mask[0], list) else 0
    if source_nx == 0 or source_ny == 0:
        return None
    mask: list[list[int]] = []
    for i in range(nx):
        source_i = min(source_nx - 1, round(i * (source_nx - 1) / max(1, nx - 1)))
        row: list[int] = []
        for j in range(ny):
            source_j = min(source_ny - 1, round(j * (source_ny - 1) / max(1, ny - 1)))
            row.append(1 if raw_mask[source_i][source_j] else 0)
        mask.append(row)
    return mask


def _load_mesh_graph(problem: TAPSProblem) -> dict | None:
    encoding = next((item for item in problem.geometry_encodings if item.kind == "mesh_graph"), None)
    if encoding is None:
        return None
    path = Path(encoding.uri)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("node_count", 0) <= 0 or not payload.get("edges"):
        return None
    return payload


def _material_number(problem: TAPSProblem, names: set[str], default: float) -> float:
    if problem.weak_form is None:
        return default
    coefficient_value = _coefficient_number(problem, names, default)
    if coefficient_value != default:
        return coefficient_value
    for term in problem.weak_form.terms:
        for coefficient in term.coefficients:
            lowered = coefficient.lower()
            if lowered in names:
                return default
    return default


def _coefficient_number(problem: TAPSProblem, names: set[str], default: float) -> float:
    normalized = {name.lower() for name in names}
    for coefficient in problem.coefficients:
        if coefficient.name.lower() not in normalized:
            continue
        if isinstance(coefficient.value, (float, int)):
            return float(coefficient.value)
        if isinstance(coefficient.value, str):
            try:
                return float(coefficient.value)
            except ValueError:
                continue
    return default


def _coefficient_string(problem: TAPSProblem, names: set[str], default: str) -> str:
    normalized = {name.lower() for name in names}
    for coefficient in problem.coefficients:
        if coefficient.name.lower() in normalized and isinstance(coefficient.value, str):
            return coefficient.value
    return default


def _coefficient_vector(problem: TAPSProblem, names: set[str], default: list[float]) -> list[float]:
    normalized = {name.lower() for name in names}
    for coefficient in problem.coefficients:
        if coefficient.name.lower() not in normalized:
            continue
        if isinstance(coefficient.value, list):
            values = [float(value) for value in coefficient.value]
            if values:
                return values
        if isinstance(coefficient.value, str):
            stripped = coefficient.value.strip().strip("[]()")
            pieces = [piece.strip() for piece in stripped.replace(";", ",").split(",") if piece.strip()]
            try:
                values = [float(piece) for piece in pieces]
            except ValueError:
                continue
            if values:
                return values
    return default


def _coefficient_complex(problem: TAPSProblem, names: set[str], default: complex) -> complex:
    normalized = {name.lower() for name in names}
    for coefficient in problem.coefficients:
        if coefficient.name.lower() not in normalized:
            continue
        value = coefficient.value
        if isinstance(value, (float, int)):
            return complex(float(value), 0.0)
        if isinstance(value, list) and value:
            real = float(value[0])
            imag = float(value[1]) if len(value) > 1 else 0.0
            return complex(real, imag)
        if isinstance(value, dict):
            real = float(value.get("real", value.get("re", 0.0)))
            imag = float(value.get("imag", value.get("im", 0.0)))
            return complex(real, imag)
        if isinstance(value, str):
            stripped = value.strip().lower().replace("i", "j")
            try:
                return complex(stripped)
            except ValueError:
                pieces = [piece.strip() for piece in stripped.strip("[]()").replace(";", ",").split(",") if piece.strip()]
                if pieces:
                    try:
                        real = float(pieces[0])
                        imag = float(pieces[1]) if len(pieces) > 1 else 0.0
                        return complex(real, imag)
                    except ValueError:
                        continue
    return default


def _json_number(value: float | complex) -> float | list[float]:
    if isinstance(value, complex):
        if abs(value.imag) <= 1e-14:
            return value.real
        return [value.real, value.imag]
    return value


def _json_vector(values: list[float] | list[complex]) -> list[float | list[float]]:
    return [_json_number(value) for value in values]


def _json_sparse_rows(rows: list[dict[int, float]] | list[dict[int, complex]]) -> list[dict[str, float | list[float]]]:
    return [{str(col): _json_number(value) for col, value in row.items()} for row in rows]


def _is_zero_boundary_value(value: object) -> bool:
    if isinstance(value, (float, int)):
        return abs(float(value)) <= 1e-14
    if isinstance(value, list):
        try:
            return all(abs(float(item)) <= 1e-14 for item in value)
        except (TypeError, ValueError):
            return False
    if isinstance(value, str):
        try:
            return abs(float(value)) <= 1e-14
        except ValueError:
            return value.strip().lower() in {"0", "zero", "pec", "tangential_zero"}
    return False


def _em_tangential_boundary_policy(problem: TAPSProblem) -> str:
    if not problem.boundary_conditions:
        return "pec_tangential_zero_default"
    natural_kinds = {"neumann", "robin", "symmetry", "farfield", "outlet", "interface"}
    for boundary in problem.boundary_conditions:
        field = boundary.field.lower()
        kind = boundary.kind.lower()
        value_kind = boundary.value.get("kind", "").lower() if isinstance(boundary.value, dict) else ""
        if field in {"e", "e_t", "et", "tangential_e", "electric_field"} and (
            value_kind in {"absorbing", "impedance", "port"} or kind in {"robin", "custom"}
        ):
            return value_kind or "impedance"
        if field in {"e", "e_t", "et", "tangential_e", "electric_field"} and kind == "dirichlet" and _is_zero_boundary_value(boundary.value):
            return "pec_tangential_zero"
        if field in {"e", "e_t", "et", "tangential_e", "electric_field"} and kind in natural_kinds:
            return "natural"
    return "pec_tangential_zero_default"


def _em_boundary_parameters(problem: TAPSProblem) -> dict[str, complex]:
    params = {"impedance": 0.0 + 0.0j, "port_amplitude": 0.0 + 0.0j}
    for boundary in problem.boundary_conditions:
        field = boundary.field.lower()
        if field not in {"e", "e_t", "et", "tangential_e", "electric_field"} or not isinstance(boundary.value, dict):
            continue
        value = boundary.value
        if "impedance" in value:
            raw = value["impedance"]
            if isinstance(raw, list) and raw:
                params["impedance"] = complex(float(raw[0]), float(raw[1]) if len(raw) > 1 else 0.0)
            elif isinstance(raw, dict):
                params["impedance"] = complex(float(raw.get("real", raw.get("re", 0.0))), float(raw.get("imag", raw.get("im", 0.0))))
            else:
                params["impedance"] = complex(raw) if isinstance(raw, str) else complex(float(raw), 0.0)
        if "amplitude" in value:
            raw = value["amplitude"]
            if isinstance(raw, list) and raw:
                params["port_amplitude"] = complex(float(raw[0]), float(raw[1]) if len(raw) > 1 else 0.0)
            elif isinstance(raw, dict):
                params["port_amplitude"] = complex(float(raw.get("real", raw.get("re", 0.0))), float(raw.get("imag", raw.get("im", 0.0))))
            else:
                params["port_amplitude"] = complex(raw) if isinstance(raw, str) else complex(float(raw), 0.0)
    return params


def _edge_ids_for_boundary_region(graph: dict, edge_list: list[tuple[int, int]], boundary_nodes: set[int], region_id: str) -> set[int]:
    boundary_edge_sets = graph.get("boundary_edge_sets", {})
    candidates = [
        region_id,
        region_id.replace("region:", "boundary:"),
        region_id.replace("boundary:", ""),
    ]
    for candidate in candidates:
        if candidate in boundary_edge_sets:
            return {int(edge_id) for edge_id in boundary_edge_sets[candidate]}
    if region_id in {"boundary", "all_boundaries", "external_boundary"}:
        return {index for index, (a, b) in enumerate(edge_list) if a in boundary_nodes and b in boundary_nodes}
    if region_id in {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}:
        return {int(edge_id) for edge_id in boundary_edge_sets.get(region_id, [])}
    return {index for index, (a, b) in enumerate(edge_list) if a in boundary_nodes and b in boundary_nodes}


def _em_boundary_edge_ids_for_policy(
    problem: TAPSProblem,
    graph: dict,
    edge_list: list[tuple[int, int]],
    boundary_nodes: set[int],
    policy: str,
) -> set[int]:
    if not problem.boundary_conditions:
        return {index for index, (a, b) in enumerate(edge_list) if a in boundary_nodes and b in boundary_nodes}
    selected: set[int] = set()
    for boundary in problem.boundary_conditions:
        field = boundary.field.lower()
        if field not in {"e", "e_t", "et", "tangential_e", "electric_field"}:
            continue
        kind = boundary.kind.lower()
        value_kind = boundary.value.get("kind", "").lower() if isinstance(boundary.value, dict) else ""
        include = False
        if policy.startswith("pec") and kind == "dirichlet" and _is_zero_boundary_value(boundary.value):
            include = True
        elif policy in {"absorbing", "impedance", "port"} and (value_kind == policy or kind in {"robin", "custom"}):
            include = True
        if include:
            selected.update(_edge_ids_for_boundary_region(graph, edge_list, boundary_nodes, boundary.region_id))
    return selected


def _graph_edge_tuples(graph: dict) -> list[tuple[int, int]]:
    return [
        (int(edge[0]), int(edge[1])) if int(edge[0]) < int(edge[1]) else (int(edge[1]), int(edge[0]))
        for edge in graph.get("edges", [])
        if isinstance(edge, list) and len(edge) >= 2
    ]


def _edge_tuples_for_boundary_region(
    graph: dict,
    geometric_edges: list[tuple[int, int]],
    boundary_nodes: set[int],
    region_id: str,
) -> set[tuple[int, int]]:
    graph_edges = _graph_edge_tuples(graph)
    boundary_edge_sets = graph.get("boundary_edge_sets", {})
    candidates = [
        region_id,
        region_id.replace("region:", "boundary:"),
        region_id.replace("boundary:", ""),
    ]
    for candidate in candidates:
        if candidate in boundary_edge_sets:
            selected: set[tuple[int, int]] = set()
            for edge_id in boundary_edge_sets[candidate]:
                index = int(edge_id)
                if 0 <= index < len(graph_edges):
                    selected.add(graph_edges[index])
            boundary_node_sets = graph.get("boundary_node_sets", {})
            if candidate in boundary_node_sets:
                node_set = {int(node) for node in boundary_node_sets[candidate]}
                selected.update(edge for edge in geometric_edges if edge[0] in node_set and edge[1] in node_set)
            return selected
    if region_id in {"boundary", "all_boundaries", "external_boundary"}:
        return {edge for edge in geometric_edges if edge[0] in boundary_nodes and edge[1] in boundary_nodes}
    if region_id in {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}:
        return {
            graph_edges[int(edge_id)]
            for edge_id in boundary_edge_sets.get(region_id, [])
            if 0 <= int(edge_id) < len(graph_edges)
        }
    return {edge for edge in geometric_edges if edge[0] in boundary_nodes and edge[1] in boundary_nodes}


def _em_boundary_edge_tuples_for_policy(
    problem: TAPSProblem,
    graph: dict,
    geometric_edges: list[tuple[int, int]],
    boundary_nodes: set[int],
    policy: str,
) -> set[tuple[int, int]]:
    if not problem.boundary_conditions:
        return {edge for edge in geometric_edges if edge[0] in boundary_nodes and edge[1] in boundary_nodes}
    selected: set[tuple[int, int]] = set()
    for boundary in problem.boundary_conditions:
        field = boundary.field.lower()
        if field not in {"e", "e_t", "et", "tangential_e", "electric_field"}:
            continue
        kind = boundary.kind.lower()
        value_kind = boundary.value.get("kind", "").lower() if isinstance(boundary.value, dict) else ""
        include = False
        if policy.startswith("pec") and kind == "dirichlet" and _is_zero_boundary_value(boundary.value):
            include = True
        elif policy in {"absorbing", "impedance", "port"} and (value_kind == policy or kind in {"robin", "custom"}):
            include = True
        if include:
            selected.update(_edge_tuples_for_boundary_region(graph, geometric_edges, boundary_nodes, boundary.region_id))
    return selected


def _nedelec_geometric_edges(dof_entities: list) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for entity in dof_entities:
        edge: tuple[int, int] | None = None
        if isinstance(entity, tuple) and len(entity) >= 2:
            edge = (int(entity[0]), int(entity[1]))
        elif isinstance(entity, dict) and entity.get("kind") == "edge_moment" and isinstance(entity.get("edge"), list):
            raw_edge = entity["edge"]
            edge = (int(raw_edge[0]), int(raw_edge[1]))
        if edge is None:
            continue
        edge = edge if edge[0] < edge[1] else (edge[1], edge[0])
        if edge not in seen:
            seen.add(edge)
            edges.append(edge)
    return edges


def _nedelec_dof_ids_for_edges(dof_entities: list, selected_edges: set[tuple[int, int]]) -> set[int]:
    selected = {edge if edge[0] < edge[1] else (edge[1], edge[0]) for edge in selected_edges}
    dof_ids: set[int] = set()
    for index, entity in enumerate(dof_entities):
        edge: tuple[int, int] | None = None
        if isinstance(entity, tuple) and len(entity) >= 2:
            edge = (int(entity[0]), int(entity[1]))
        elif isinstance(entity, dict) and entity.get("kind") == "edge_moment" and isinstance(entity.get("edge"), list):
            raw_edge = entity["edge"]
            edge = (int(raw_edge[0]), int(raw_edge[1]))
        if edge is None:
            continue
        edge = edge if edge[0] < edge[1] else (edge[1], edge[0])
        if edge in selected:
            dof_ids.add(index)
    return dof_ids


def _nedelec_edges_for_dof_ids(dof_entities: list, dof_ids: set[int]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for index, entity in enumerate(dof_entities):
        if index not in dof_ids:
            continue
        edge: tuple[int, int] | None = None
        if isinstance(entity, tuple) and len(entity) >= 2:
            edge = (int(entity[0]), int(entity[1]))
        elif isinstance(entity, dict) and entity.get("kind") == "edge_moment" and isinstance(entity.get("edge"), list):
            raw_edge = entity["edge"]
            edge = (int(raw_edge[0]), int(raw_edge[1]))
        if edge is None:
            continue
        edges.add(edge if edge[0] < edge[1] else (edge[1], edge[0]))
    return edges


def _family(problem: TAPSProblem) -> str:
    if problem.weak_form is not None:
        return problem.weak_form.family.lower()
    return (problem.operator_weak_form or "custom").lower()


def _mode_pairs(rank: int, max_x_mode: int, max_y_mode: int) -> list[tuple[int, int, float]]:
    pairs: list[tuple[int, int, float]] = []
    for mode_sum in range(2, max_x_mode + max_y_mode + 1):
        for mx in range(1, min(max_x_mode, mode_sum - 1) + 1):
            my = mode_sum - mx
            if 1 <= my <= max_y_mode:
                coefficient = 1.0 / float(mx * mx + my * my)
                pairs.append((mx, my, coefficient))
            if len(pairs) >= rank:
                return pairs
    return pairs


def _source_modes_1d(rank: int, max_mode: int) -> list[tuple[int, float]]:
    return [(mode, 1.0 / float(mode * mode)) for mode in range(1, min(rank, max_mode) + 1)]


def _reaction_value(family: str) -> float:
    if family == "reaction_diffusion":
        return 1.0
    if family == "helmholtz":
        # Use a non-resonant wavenumber proxy for the first executable kernel.
        return -(0.5 * math.pi) ** 2
    return 0.0


def _thomas_solve(lower: list[float], diag: list[float], upper: list[float], rhs: list[float]) -> list[float]:
    n = len(diag)
    if n == 0:
        return []
    c = upper[:]
    d = diag[:]
    b = rhs[:]
    for i in range(1, n):
        if abs(d[i - 1]) < 1e-14:
            raise ZeroDivisionError("Singular tridiagonal system in generic TAPS assembler.")
        factor = lower[i - 1] / d[i - 1]
        d[i] -= factor * c[i - 1]
        b[i] -= factor * b[i - 1]
    solution = [0.0 for _ in range(n)]
    if abs(d[-1]) < 1e-14:
        raise ZeroDivisionError("Singular tridiagonal system in generic TAPS assembler.")
    solution[-1] = b[-1] / d[-1]
    for i in range(n - 2, -1, -1):
        solution[i] = (b[i] - c[i] * solution[i + 1]) / d[i]
    return solution


def _residual_norm(lower: list[float], diag: list[float], upper: list[float], rhs: list[float], solution: list[float]) -> float:
    residual_sq = 0.0
    rhs_sq = 0.0
    for i, value in enumerate(solution):
        ax = diag[i] * value
        if i > 0:
            ax += lower[i - 1] * solution[i - 1]
        if i < len(solution) - 1:
            ax += upper[i] * solution[i + 1]
        residual = ax - rhs[i]
        residual_sq += residual * residual
        rhs_sq += rhs[i] * rhs[i]
    return math.sqrt(residual_sq) / (math.sqrt(rhs_sq) + 1e-12)


def _nonlinear_residual_norm_1d(
    lower: list[float],
    base_diag: list[float],
    cubic: float,
    rhs: list[float],
    solution: list[float],
) -> float:
    residual_sq = 0.0
    rhs_sq = 0.0
    for i, value in enumerate(solution):
        ax = base_diag[i] * value + cubic * value * value * value
        if i > 0:
            ax += lower[i - 1] * solution[i - 1]
        if i < len(solution) - 1:
            ax += lower[i] * solution[i + 1]
        residual = ax - rhs[i]
        residual_sq += residual * residual
        rhs_sq += rhs[i] * rhs[i]
    return math.sqrt(residual_sq) / (math.sqrt(rhs_sq) + 1e-12)


def _residual_norm_2d(
    x: list[float],
    y: list[float],
    solution: list[list[float]],
    rhs: list[list[float]],
    diffusion: float,
    reaction: float,
    active_mask: list[list[int]] | None = None,
) -> float:
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    cx = diffusion / (dx * dx)
    cy = diffusion / (dy * dy)
    residual_sq = 0.0
    rhs_sq = 0.0
    for i in range(1, len(x) - 1):
        for j in range(1, len(y) - 1):
            if active_mask is not None and not active_mask[i][j]:
                continue
            left = solution[i - 1][j] if active_mask is None or active_mask[i - 1][j] else 0.0
            right = solution[i + 1][j] if active_mask is None or active_mask[i + 1][j] else 0.0
            down = solution[i][j - 1] if active_mask is None or active_mask[i][j - 1] else 0.0
            up = solution[i][j + 1] if active_mask is None or active_mask[i][j + 1] else 0.0
            au = (
                (2.0 * cx + 2.0 * cy + reaction) * solution[i][j]
                - cx * (left + right)
                - cy * (down + up)
            )
            residual = au - rhs[i][j]
            residual_sq += residual * residual
            rhs_sq += rhs[i][j] * rhs[i][j]
    return math.sqrt(residual_sq) / (math.sqrt(rhs_sq) + 1e-12)


def _masked_relaxation_2d(
    x: list[float],
    y: list[float],
    rhs: list[list[float]],
    diffusion: float,
    reaction: float,
    active_mask: list[list[int]],
    max_iterations: int = 5000,
    tolerance: float = 1e-10,
) -> tuple[list[list[float]], list[dict[str, float | int]]]:
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    cx = diffusion / (dx * dx)
    cy = diffusion / (dy * dy)
    denom = 2.0 * cx + 2.0 * cy + reaction
    solution = [[0.0 for _ in y] for _ in x]
    history: list[dict[str, float | int]] = []
    for iteration in range(1, max_iterations + 1):
        update_sq = 0.0
        norm_sq = 0.0
        for i in range(1, len(x) - 1):
            for j in range(1, len(y) - 1):
                if not active_mask[i][j]:
                    continue
                left = solution[i - 1][j] if active_mask[i - 1][j] else 0.0
                right = solution[i + 1][j] if active_mask[i + 1][j] else 0.0
                down = solution[i][j - 1] if active_mask[i][j - 1] else 0.0
                up = solution[i][j + 1] if active_mask[i][j + 1] else 0.0
                new_value = (rhs[i][j] + cx * (left + right) + cy * (down + up)) / denom
                update_sq += (new_value - solution[i][j]) ** 2
                norm_sq += new_value * new_value
                solution[i][j] = new_value
        residual = _residual_norm_2d(x, y, solution, rhs, diffusion, reaction, active_mask=active_mask)
        update_norm = math.sqrt(update_sq) / (math.sqrt(norm_sq) + 1e-12)
        if iteration == 1 or iteration % 25 == 0 or residual < tolerance or update_norm < tolerance:
            history.append({"iteration": iteration, "normalized_linear_residual": residual, "relative_update": update_norm})
        if residual < tolerance or update_norm < tolerance:
            break
    return solution, history


def _nonlinear_residual_norm_2d(
    x: list[float],
    y: list[float],
    solution: list[list[float]],
    rhs: list[list[float]],
    diffusion: float,
    linear_reaction: float,
    cubic_reaction: float,
) -> float:
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    cx = diffusion / (dx * dx)
    cy = diffusion / (dy * dy)
    residual_sq = 0.0
    rhs_sq = 0.0
    for i in range(1, len(x) - 1):
        for j in range(1, len(y) - 1):
            value = solution[i][j]
            au = (
                (2.0 * cx + 2.0 * cy + linear_reaction) * value
                - cx * (solution[i - 1][j] + solution[i + 1][j])
                - cy * (solution[i][j - 1] + solution[i][j + 1])
                + cubic_reaction * value * value * value
            )
            residual = au - rhs[i][j]
            residual_sq += residual * residual
            rhs_sq += rhs[i][j] * rhs[i][j]
    return math.sqrt(residual_sq) / (math.sqrt(rhs_sq) + 1e-12)


def _coupled_residual_norm_2d(
    x: list[float],
    y: list[float],
    u: list[list[float]],
    v: list[list[float]],
    rhs_u: list[list[float]],
    rhs_v: list[list[float]],
    diffusion: float,
    linear_reaction: float,
    cubic_reaction: float,
    coupling: float,
) -> float:
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    cx = diffusion / (dx * dx)
    cy = diffusion / (dy * dy)
    residual_sq = 0.0
    rhs_sq = 0.0
    for i in range(1, len(x) - 1):
        for j in range(1, len(y) - 1):
            u_value = u[i][j]
            v_value = v[i][j]
            au = (
                (2.0 * cx + 2.0 * cy + linear_reaction + coupling) * u_value
                - cx * (u[i - 1][j] + u[i + 1][j])
                - cy * (u[i][j - 1] + u[i][j + 1])
                + cubic_reaction * u_value * u_value * u_value
                - coupling * v_value
            )
            av = (
                (2.0 * cx + 2.0 * cy + linear_reaction + coupling) * v_value
                - cx * (v[i - 1][j] + v[i + 1][j])
                - cy * (v[i][j - 1] + v[i][j + 1])
                + cubic_reaction * v_value * v_value * v_value
                - coupling * u_value
            )
            ru = au - rhs_u[i][j]
            rv = av - rhs_v[i][j]
            residual_sq += ru * ru + rv * rv
            rhs_sq += rhs_u[i][j] * rhs_u[i][j] + rhs_v[i][j] * rhs_v[i][j]
    return math.sqrt(residual_sq) / (math.sqrt(rhs_sq) + 1e-12)


def _graph_residual_norm(
    node_count: int,
    edges: list[list[int]],
    boundary_nodes: set[int],
    rhs: list[float],
    solution: list[float],
) -> float:
    residual_sq = 0.0
    rhs_sq = 0.0
    adjacency = [[] for _ in range(node_count)]
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    for node in range(node_count):
        if node in boundary_nodes:
            residual = solution[node]
            residual_sq += residual * residual
            continue
        degree = len(adjacency[node])
        au = degree * solution[node] - sum(solution[neighbor] for neighbor in adjacency[node])
        residual = au - rhs[node]
        residual_sq += residual * residual
        rhs_sq += rhs[node] * rhs[node]
    return math.sqrt(residual_sq) / (math.sqrt(rhs_sq) + 1e-12)


def _triangle_cells(graph: dict) -> list[list[int]]:
    triangles: list[list[int]] = []
    for block in graph.get("cell_blocks", []):
        block_type = str(block.get("type", "")).lower()
        if "triangle" not in block_type:
            continue
        for cell in block.get("cells", []):
            if len(cell) >= 3:
                triangles.append([int(node) for node in cell])
    return triangles


def _p2_shape_values_and_gradients(r: float, s: float) -> tuple[list[float], list[list[float]]]:
    l1 = 1.0 - r - s
    l2 = r
    l3 = s
    values = [
        l1 * (2.0 * l1 - 1.0),
        l2 * (2.0 * l2 - 1.0),
        l3 * (2.0 * l3 - 1.0),
        4.0 * l1 * l2,
        4.0 * l2 * l3,
        4.0 * l3 * l1,
    ]
    gradients_ref = [
        [-(4.0 * l1 - 1.0), -(4.0 * l1 - 1.0)],
        [4.0 * l2 - 1.0, 0.0],
        [0.0, 4.0 * l3 - 1.0],
        [4.0 * (l1 - l2), -4.0 * l2],
        [4.0 * l3, 4.0 * l2],
        [-4.0 * l3, 4.0 * (l1 - l3)],
    ]
    return values, gradients_ref


def _lagrange_nodes(order: int) -> list[tuple[int, int, int]]:
    if order < 1:
        raise ValueError("Triangle Lagrange order must be >= 1.")
    nodes: list[tuple[int, int, int]] = [(order, 0, 0), (0, order, 0), (0, 0, order)]
    for i in range(order - 1, 0, -1):
        nodes.append((i, order - i, 0))
    for j in range(order - 1, 0, -1):
        nodes.append((0, j, order - j))
    for k in range(order - 1, 0, -1):
        nodes.append((order - k, 0, k))
    for i in range(1, order):
        for j in range(1, order - i):
            nodes.append((i, j, order - i - j))
    return nodes


def _monomial_powers(order: int) -> list[tuple[int, int]]:
    return [(a, b) for degree in range(order + 1) for a in range(degree, -1, -1) for b in [degree - a]]


def _invert_matrix(matrix: list[list[float]]) -> list[list[float]]:
    size = len(matrix)
    augmented = [row[:] + [1.0 if i == j else 0.0 for j in range(size)] for i, row in enumerate(matrix)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-14:
            raise ValueError("Singular Vandermonde matrix for triangle Lagrange basis.")
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        scale = augmented[col][col]
        augmented[col] = [value / scale for value in augmented[col]]
        for row in range(size):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor == 0.0:
                continue
            augmented[row] = [value - factor * augmented[col][idx] for idx, value in enumerate(augmented[row])]
    return [row[size:] for row in augmented]


@lru_cache(maxsize=8)
def _lagrange_basis_coefficients(order: int) -> tuple[tuple[float, ...], ...]:
    powers = _monomial_powers(order)
    rows = []
    for _, j, k in _lagrange_nodes(order):
        r = j / order
        s = k / order
        rows.append([(r ** a) * (s ** b) for a, b in powers])
    inverse = _invert_matrix(rows)
    # Columns of V^{-1} are nodal basis coefficients.
    return tuple(tuple(inverse[col][basis] for col in range(len(powers))) for basis in range(len(powers)))


def _lagrange_shape_values_and_gradients(order: int, r: float, s: float) -> tuple[list[float], list[list[float]]]:
    powers = _monomial_powers(order)
    coefficients = _lagrange_basis_coefficients(order)
    values: list[float] = []
    gradients: list[list[float]] = []
    for basis_coefficients in coefficients:
        value = 0.0
        dr = 0.0
        ds = 0.0
        for coefficient, (a, b) in zip(basis_coefficients, powers):
            value += coefficient * (r ** a) * (s ** b)
            if a:
                dr += coefficient * a * (r ** (a - 1)) * (s ** b)
            if b:
                ds += coefficient * b * (r ** a) * (s ** (b - 1))
        values.append(value)
        gradients.append([dr, ds])
    return values, gradients


def _triangle_quadrature(order: int) -> list[tuple[float, float, float]]:
    if order <= 2:
        return [(1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0), (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0), (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0)]
    return [
        (1.0 / 3.0, 1.0 / 3.0, 0.1125),
        (0.059715871789770, 0.470142064105115, 0.066197076394253),
        (0.470142064105115, 0.059715871789770, 0.066197076394253),
        (0.470142064105115, 0.470142064105115, 0.066197076394253),
        (0.797426985353087, 0.101286507323456, 0.062969590272414),
        (0.101286507323456, 0.797426985353087, 0.062969590272414),
        (0.101286507323456, 0.101286507323456, 0.062969590272414),
    ]


def _assemble_triangle_lagrange_element(
    points: list[list[float]], nodes: list[int], order: int
) -> tuple[float, list[list[float]], list[float], list[dict]] | None:
    n0, n1, n2 = nodes[:3]
    x0, y0 = float(points[n0][0]), float(points[n0][1])
    x1, y1 = float(points[n1][0]), float(points[n1][1])
    x2, y2 = float(points[n2][0]), float(points[n2][1])
    j00 = x1 - x0
    j01 = x2 - x0
    j10 = y1 - y0
    j11 = y2 - y0
    det_j = j00 * j11 - j01 * j10
    area = abs(det_j) / 2.0
    if area <= 1e-14:
        return None
    inv_det = 1.0 / det_j
    jt_inv = [[j11 * inv_det, -j10 * inv_det], [-j01 * inv_det, j00 * inv_det]]
    basis_size = (order + 1) * (order + 2) // 2
    local_stiffness = [[0.0 for _ in range(basis_size)] for _ in range(basis_size)]
    local_mass = [0.0 for _ in range(basis_size)]
    quadrature_records: list[dict] = []
    for r, s, weight in _triangle_quadrature(order):
        values, gradients_ref = _lagrange_shape_values_and_gradients(order, r, s)
        gradients = [
            [
                jt_inv[0][0] * grad[0] + jt_inv[0][1] * grad[1],
                jt_inv[1][0] * grad[0] + jt_inv[1][1] * grad[1],
            ]
            for grad in gradients_ref
        ]
        scaled_weight = abs(det_j) * weight
        for i in range(basis_size):
            local_mass[i] += values[i] * scaled_weight
            for j in range(basis_size):
                local_stiffness[i][j] += scaled_weight * (
                    gradients[i][0] * gradients[j][0] + gradients[i][1] * gradients[j][1]
                )
        quadrature_records.append({"r": r, "s": s, "weight": weight, "grad_phi": gradients})
    return area, local_stiffness, local_mass, quadrature_records


def _assemble_triangle_p2_element(
    points: list[list[float]], nodes: list[int]
) -> tuple[float, list[list[float]], list[float], list[dict]] | None:
    n0, n1, n2 = nodes[:3]
    x0, y0 = float(points[n0][0]), float(points[n0][1])
    x1, y1 = float(points[n1][0]), float(points[n1][1])
    x2, y2 = float(points[n2][0]), float(points[n2][1])
    j00 = x1 - x0
    j01 = x2 - x0
    j10 = y1 - y0
    j11 = y2 - y0
    det_j = j00 * j11 - j01 * j10
    area = abs(det_j) / 2.0
    if area <= 1e-14:
        return None
    inv_det = 1.0 / det_j
    # J^{-T}: maps reference shape gradients to physical gradients.
    jt_inv = [[j11 * inv_det, -j10 * inv_det], [-j01 * inv_det, j00 * inv_det]]
    quadrature = [(1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0), (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0), (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0)]
    local_stiffness = [[0.0 for _ in range(6)] for _ in range(6)]
    local_mass = [0.0 for _ in range(6)]
    quadrature_records: list[dict] = []
    for r, s, weight in quadrature:
        values, gradients_ref = _p2_shape_values_and_gradients(r, s)
        gradients = [
            [
                jt_inv[0][0] * grad[0] + jt_inv[0][1] * grad[1],
                jt_inv[1][0] * grad[0] + jt_inv[1][1] * grad[1],
            ]
            for grad in gradients_ref
        ]
        scaled_weight = abs(det_j) * weight
        for i in range(6):
            local_mass[i] += scaled_weight * values[i]
            for j in range(6):
                local_stiffness[i][j] += scaled_weight * (
                    gradients[i][0] * gradients[j][0] + gradients[i][1] * gradients[j][1]
                )
        quadrature_records.append({"r": r, "s": s, "weight": weight, "grad_phi": gradients})
    return area, local_stiffness, local_mass, quadrature_records


def _assemble_triangle_stiffness(
    points: list[list[float]], triangles: list[list[int]]
) -> tuple[list[dict[int, float]], list[float], float, list[dict]]:
    node_count = len(points)
    stiffness: list[dict[int, float]] = [dict() for _ in range(node_count)]
    lumped_mass = [0.0 for _ in range(node_count)]
    element_records: list[dict] = []
    total_area = 0.0
    for triangle in triangles:
        if len(triangle) >= 10:
            nodes = triangle[:10]
            assembled = _assemble_triangle_lagrange_element(points, nodes, order=3)
            if assembled is None:
                continue
            area, local_stiffness, local_mass, quadrature_records = assembled
            for local_i, global_i in enumerate(nodes):
                lumped_mass[global_i] += local_mass[local_i]
                for local_j, global_j in enumerate(nodes):
                    value = local_stiffness[local_i][local_j]
                    stiffness[global_i][global_j] = stiffness[global_i].get(global_j, 0.0) + value
            element_records.append(
                {
                    "nodes": nodes,
                    "area": area,
                    "basis": "p3_triangle",
                    "lagrange_order": 3,
                    "quadrature": quadrature_records,
                    "local_stiffness": local_stiffness,
                    "local_mass": local_mass,
                }
            )
            total_area += area
            continue

        if len(triangle) >= 6:
            nodes = triangle[:6]
            assembled = _assemble_triangle_p2_element(points, nodes)
            if assembled is None:
                continue
            area, local_stiffness, local_mass, quadrature_records = assembled
            for local_i, global_i in enumerate(nodes):
                lumped_mass[global_i] += local_mass[local_i]
                for local_j, global_j in enumerate(nodes):
                    value = local_stiffness[local_i][local_j]
                    stiffness[global_i][global_j] = stiffness[global_i].get(global_j, 0.0) + value
            element_records.append(
                {
                    "nodes": nodes,
                    "area": area,
                    "basis": "p2_triangle",
                    "quadrature": quadrature_records,
                    "local_stiffness": local_stiffness,
                    "local_mass": local_mass,
                }
            )
            total_area += area
            continue

        n0, n1, n2 = triangle[:3]
        x0, y0 = float(points[n0][0]), float(points[n0][1])
        x1, y1 = float(points[n1][0]), float(points[n1][1])
        x2, y2 = float(points[n2][0]), float(points[n2][1])
        signed_area_2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        area = abs(signed_area_2) / 2.0
        if area <= 1e-14:
            continue
        b = [y1 - y2, y2 - y0, y0 - y1]
        c = [x2 - x1, x0 - x2, x1 - x0]
        nodes = [n0, n1, n2]
        gradients = [[b[index] / (2.0 * area), c[index] / (2.0 * area)] for index in range(3)]
        local_stiffness: list[list[float]] = []
        for local_i, global_i in enumerate(nodes):
            lumped_mass[global_i] += area / 3.0
            local_row = []
            for local_j, global_j in enumerate(nodes):
                value = (b[local_i] * b[local_j] + c[local_i] * c[local_j]) / (4.0 * area)
                local_row.append(value)
                stiffness[global_i][global_j] = stiffness[global_i].get(global_j, 0.0) + value
            local_stiffness.append(local_row)
        element_records.append(
            {
                "nodes": nodes,
                "area": area,
                "basis": "p1_triangle",
                "grad_phi": gradients,
                "local_stiffness": local_stiffness,
            }
        )
        total_area += area
    return stiffness, lumped_mass, total_area, element_records


def _elasticity_constitutive_matrix(young_modulus: float, poisson_ratio: float, model: str = "plane_stress") -> list[list[float]]:
    if young_modulus <= 0.0:
        raise ValueError("Young's modulus must be positive.")
    if not -1.0 < poisson_ratio < 0.5:
        raise ValueError("Poisson ratio must be in (-1, 0.5).")
    if model == "plane_strain":
        scale = young_modulus / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
        return [
            [scale * (1.0 - poisson_ratio), scale * poisson_ratio, 0.0],
            [scale * poisson_ratio, scale * (1.0 - poisson_ratio), 0.0],
            [0.0, 0.0, scale * (0.5 - poisson_ratio)],
        ]
    scale = young_modulus / (1.0 - poisson_ratio * poisson_ratio)
    return [
        [scale, scale * poisson_ratio, 0.0],
        [scale * poisson_ratio, scale, 0.0],
        [0.0, 0.0, scale * (1.0 - poisson_ratio) / 2.0],
    ]


def _assemble_triangle_elasticity_element(
    points: list[list[float]],
    nodes: list[int],
    young_modulus: float = 1.0,
    poisson_ratio: float = 0.3,
    model: str = "plane_stress",
) -> tuple[float, list[list[float]], list[list[float]], list[list[float]]] | None:
    n0, n1, n2 = nodes[:3]
    x0, y0 = float(points[n0][0]), float(points[n0][1])
    x1, y1 = float(points[n1][0]), float(points[n1][1])
    x2, y2 = float(points[n2][0]), float(points[n2][1])
    signed_area_2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    area = abs(signed_area_2) / 2.0
    if area <= 1e-14:
        return None
    b = [y1 - y2, y2 - y0, y0 - y1]
    c = [x2 - x1, x0 - x2, x1 - x0]
    gradients = [[b[index] / (2.0 * area), c[index] / (2.0 * area)] for index in range(3)]
    b_matrix = [[0.0 for _ in range(6)] for _ in range(3)]
    for local_index, (dphidx, dphidy) in enumerate(gradients):
        ux = 2 * local_index
        uy = ux + 1
        b_matrix[0][ux] = dphidx
        b_matrix[1][uy] = dphidy
        b_matrix[2][ux] = dphidy
        b_matrix[2][uy] = dphidx
    constitutive = _elasticity_constitutive_matrix(young_modulus, poisson_ratio, model=model)
    db = [
        [sum(constitutive[row][k] * b_matrix[k][col] for k in range(3)) for col in range(6)]
        for row in range(3)
    ]
    local_stiffness = [
        [area * sum(b_matrix[k][i] * db[k][j] for k in range(3)) for j in range(6)]
        for i in range(6)
    ]
    return area, local_stiffness, gradients, b_matrix


def _assemble_triangle_lagrange_elasticity_element(
    points: list[list[float]],
    nodes: list[int],
    order: int,
    young_modulus: float = 1.0,
    poisson_ratio: float = 0.3,
    model: str = "plane_stress",
) -> tuple[float, list[list[float]], list[float], list[dict]] | None:
    n0, n1, n2 = nodes[:3]
    x0, y0 = float(points[n0][0]), float(points[n0][1])
    x1, y1 = float(points[n1][0]), float(points[n1][1])
    x2, y2 = float(points[n2][0]), float(points[n2][1])
    j00 = x1 - x0
    j01 = x2 - x0
    j10 = y1 - y0
    j11 = y2 - y0
    det_j = j00 * j11 - j01 * j10
    area = abs(det_j) / 2.0
    if area <= 1e-14:
        return None
    inv_det = 1.0 / det_j
    jt_inv = [[j11 * inv_det, -j10 * inv_det], [-j01 * inv_det, j00 * inv_det]]
    basis_size = (order + 1) * (order + 2) // 2
    local_stiffness = [[0.0 for _ in range(2 * basis_size)] for _ in range(2 * basis_size)]
    local_mass = [0.0 for _ in range(basis_size)]
    constitutive = _elasticity_constitutive_matrix(young_modulus, poisson_ratio, model=model)
    quadrature_records: list[dict] = []
    for r, s, weight in _triangle_quadrature(order):
        values, gradients_ref = _lagrange_shape_values_and_gradients(order, r, s)
        gradients = [
            [
                jt_inv[0][0] * grad[0] + jt_inv[0][1] * grad[1],
                jt_inv[1][0] * grad[0] + jt_inv[1][1] * grad[1],
            ]
            for grad in gradients_ref
        ]
        b_matrix = [[0.0 for _ in range(2 * basis_size)] for _ in range(3)]
        for local_index, (dphidx, dphidy) in enumerate(gradients):
            ux = 2 * local_index
            uy = ux + 1
            b_matrix[0][ux] = dphidx
            b_matrix[1][uy] = dphidy
            b_matrix[2][ux] = dphidy
            b_matrix[2][uy] = dphidx
        db = [
            [sum(constitutive[row][k] * b_matrix[k][col] for k in range(3)) for col in range(2 * basis_size)]
            for row in range(3)
        ]
        scaled_weight = abs(det_j) * weight
        for local_i in range(basis_size):
            local_mass[local_i] += scaled_weight * values[local_i]
        for i in range(2 * basis_size):
            for j in range(2 * basis_size):
                local_stiffness[i][j] += scaled_weight * sum(b_matrix[k][i] * db[k][j] for k in range(3))
        quadrature_records.append(
            {
                "r": r,
                "s": s,
                "weight": weight,
                "grad_phi": gradients,
                "strain_displacement_matrix": b_matrix,
            }
        )
    return area, local_stiffness, local_mass, quadrature_records


def _assemble_triangle_elasticity_stiffness(
    points: list[list[float]],
    triangles: list[list[int]],
    young_modulus: float = 1.0,
    poisson_ratio: float = 0.3,
    model: str = "plane_stress",
) -> tuple[list[dict[int, float]], list[float], float, list[dict]]:
    dof_count = 2 * len(points)
    stiffness: list[dict[int, float]] = [dict() for _ in range(dof_count)]
    lumped_mass = [0.0 for _ in range(len(points))]
    element_records: list[dict] = []
    total_area = 0.0
    for triangle in triangles:
        if len(triangle) >= 10:
            nodes = triangle[:10]
            assembled = _assemble_triangle_lagrange_elasticity_element(
                points,
                nodes,
                order=3,
                young_modulus=young_modulus,
                poisson_ratio=poisson_ratio,
                model=model,
            )
            if assembled is None:
                continue
            area, local_stiffness, local_mass, quadrature_records = assembled
            dofs = [dof for node in nodes for dof in (2 * node, 2 * node + 1)]
            for local_i, global_i in enumerate(nodes):
                lumped_mass[global_i] += local_mass[local_i]
            for local_i, global_i in enumerate(dofs):
                for local_j, global_j in enumerate(dofs):
                    value = local_stiffness[local_i][local_j]
                    stiffness[global_i][global_j] = stiffness[global_i].get(global_j, 0.0) + value
            element_records.append(
                {
                    "nodes": nodes,
                    "dofs": dofs,
                    "area": area,
                    "basis": "p3_vector_triangle",
                    "lagrange_order": 3,
                    "constitutive_model": model,
                    "young_modulus": young_modulus,
                    "poisson_ratio": poisson_ratio,
                    "quadrature": quadrature_records,
                    "local_stiffness": local_stiffness,
                    "local_mass": local_mass,
                }
            )
            total_area += area
            continue

        if len(triangle) >= 6:
            nodes = triangle[:6]
            assembled = _assemble_triangle_lagrange_elasticity_element(
                points,
                nodes,
                order=2,
                young_modulus=young_modulus,
                poisson_ratio=poisson_ratio,
                model=model,
            )
            if assembled is None:
                continue
            area, local_stiffness, local_mass, quadrature_records = assembled
            dofs = [dof for node in nodes for dof in (2 * node, 2 * node + 1)]
            for local_i, global_i in enumerate(nodes):
                lumped_mass[global_i] += local_mass[local_i]
            for local_i, global_i in enumerate(dofs):
                for local_j, global_j in enumerate(dofs):
                    value = local_stiffness[local_i][local_j]
                    stiffness[global_i][global_j] = stiffness[global_i].get(global_j, 0.0) + value
            element_records.append(
                {
                    "nodes": nodes,
                    "dofs": dofs,
                    "area": area,
                    "basis": "p2_vector_triangle",
                    "lagrange_order": 2,
                    "constitutive_model": model,
                    "young_modulus": young_modulus,
                    "poisson_ratio": poisson_ratio,
                    "quadrature": quadrature_records,
                    "local_stiffness": local_stiffness,
                    "local_mass": local_mass,
                }
            )
            total_area += area
            continue

        nodes = triangle[:3]
        if len(nodes) < 3:
            continue
        assembled = _assemble_triangle_elasticity_element(
            points,
            nodes,
            young_modulus=young_modulus,
            poisson_ratio=poisson_ratio,
            model=model,
        )
        if assembled is None:
            continue
        area, local_stiffness, gradients, b_matrix = assembled
        dofs = [dof for node in nodes for dof in (2 * node, 2 * node + 1)]
        for node in nodes:
            lumped_mass[node] += area / 3.0
        for local_i, global_i in enumerate(dofs):
            for local_j, global_j in enumerate(dofs):
                value = local_stiffness[local_i][local_j]
                stiffness[global_i][global_j] = stiffness[global_i].get(global_j, 0.0) + value
        element_records.append(
            {
                "nodes": nodes,
                "dofs": dofs,
                "area": area,
                "basis": "p1_vector_triangle",
                "constitutive_model": model,
                "young_modulus": young_modulus,
                "poisson_ratio": poisson_ratio,
                "grad_phi": gradients,
                "strain_displacement_matrix": b_matrix,
                "local_stiffness": local_stiffness,
            }
        )
        total_area += area
    return stiffness, lumped_mass, total_area, element_records


def _triangle_geometry(points: list[list[float]], nodes: list[int]) -> tuple[float, list[list[float]]] | None:
    n0, n1, n2 = nodes[:3]
    x0, y0 = float(points[n0][0]), float(points[n0][1])
    x1, y1 = float(points[n1][0]), float(points[n1][1])
    x2, y2 = float(points[n2][0]), float(points[n2][1])
    signed_area_2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    area = abs(signed_area_2) / 2.0
    if area <= 1e-14:
        return None
    b = [y1 - y2, y2 - y0, y0 - y1]
    c = [x2 - x1, x0 - x2, x1 - x0]
    gradients = [[b[index] / (2.0 * area), c[index] / (2.0 * area)] for index in range(3)]
    return area, gradients


def _assemble_triangle_nedelec_curl_curl(
    points: list[list[float]],
    triangles: list[list[int]],
    curl_weight: float | complex = 1.0,
    mass_weight: float | complex = 1.0,
) -> tuple[list[dict[int, float | complex]], list, float, list[dict]]:
    if triangles and any(len(triangle) >= 6 for triangle in triangles):
        return _assemble_triangle_nedelec_curl_curl_order2(points, triangles, curl_weight=curl_weight, mass_weight=mass_weight)

    edge_index: dict[tuple[int, int], int] = {}
    edge_list: list[tuple[int, int]] = []
    for triangle in triangles:
        vertices = triangle[:3]
        for a, b in [(vertices[0], vertices[1]), (vertices[1], vertices[2]), (vertices[2], vertices[0])]:
            edge = (a, b) if a < b else (b, a)
            if edge not in edge_index:
                edge_index[edge] = len(edge_list)
                edge_list.append(edge)

    stiffness: list[dict[int, float | complex]] = [dict() for _ in edge_list]
    total_area = 0.0
    element_records: list[dict] = []
    local_edge_pairs = [(0, 1), (1, 2), (2, 0)]
    for triangle in triangles:
        vertices = triangle[:3]
        geometry = _triangle_geometry(points, vertices)
        if geometry is None:
            continue
        area, grad_lambda = geometry
        local_edges: list[int] = []
        orientation_signs: list[float] = []
        for local_a, local_b in local_edge_pairs:
            a = vertices[local_a]
            b = vertices[local_b]
            edge = (a, b) if a < b else (b, a)
            local_edges.append(edge_index[edge])
            orientation_signs.append(1.0 if (a, b) == edge else -1.0)

        curls: list[float] = []
        for local_a, local_b in local_edge_pairs:
            ga = grad_lambda[local_a]
            gb = grad_lambda[local_b]
            curls.append(2.0 * (ga[0] * gb[1] - ga[1] * gb[0]))

        local_matrix = [[0.0 + 0.0j for _ in range(3)] for _ in range(3)]
        quadrature_records: list[dict] = []
        for r, s, weight in _triangle_quadrature(1):
            lambdas = [1.0 - r - s, r, s]
            nedelec_values = []
            for local_a, local_b in local_edge_pairs:
                value = [
                    lambdas[local_a] * grad_lambda[local_b][0] - lambdas[local_b] * grad_lambda[local_a][0],
                    lambdas[local_a] * grad_lambda[local_b][1] - lambdas[local_b] * grad_lambda[local_a][1],
                ]
                nedelec_values.append(value)
            scaled_weight = 2.0 * area * weight
            for i in range(3):
                for j in range(3):
                    mass = nedelec_values[i][0] * nedelec_values[j][0] + nedelec_values[i][1] * nedelec_values[j][1]
                    local_matrix[i][j] += orientation_signs[i] * orientation_signs[j] * (
                        scaled_weight * mass_weight * mass
                    )
            quadrature_records.append({"r": r, "s": s, "weight": weight, "nedelec_basis": nedelec_values})

        for i in range(3):
            for j in range(3):
                local_matrix[i][j] += orientation_signs[i] * orientation_signs[j] * area * curl_weight * curls[i] * curls[j]
        for local_i, global_i in enumerate(local_edges):
            for local_j, global_j in enumerate(local_edges):
                value = local_matrix[local_i][local_j]
                stiffness[global_i][global_j] = stiffness[global_i].get(global_j, 0.0) + value
        element_records.append(
            {
                "vertices": vertices,
                "edges": local_edges,
                "oriented_vertex_pairs": [(vertices[a], vertices[b]) for a, b in local_edge_pairs],
                "orientation_signs": orientation_signs,
                "area": area,
                "basis": "nedelec_first_kind_order1_triangle",
                "curl_basis": curls,
                "local_matrix": [[_json_number(value) for value in row] for row in local_matrix],
                "quadrature": quadrature_records,
            }
        )
        total_area += area
    return stiffness, edge_list, total_area, element_records


def _nedelec_p1_value_and_curl(
    lambdas: list[float],
    grad_lambda: list[list[float]],
    local_a: int,
    local_b: int,
) -> tuple[list[float], float]:
    value = [
        lambdas[local_a] * grad_lambda[local_b][0] - lambdas[local_b] * grad_lambda[local_a][0],
        lambdas[local_a] * grad_lambda[local_b][1] - lambdas[local_b] * grad_lambda[local_a][1],
    ]
    ga = grad_lambda[local_a]
    gb = grad_lambda[local_b]
    curl = 2.0 * (ga[0] * gb[1] - ga[1] * gb[0])
    return value, curl


def _scaled_nedelec_value_and_curl(
    scalar_value: float,
    scalar_gradient: list[float],
    base_value: list[float],
    base_curl: float,
) -> tuple[list[float], float]:
    value = [scalar_value * base_value[0], scalar_value * base_value[1]]
    curl = scalar_gradient[0] * base_value[1] - scalar_gradient[1] * base_value[0] + scalar_value * base_curl
    return value, curl


def _assemble_triangle_nedelec_curl_curl_order2(
    points: list[list[float]],
    triangles: list[list[int]],
    curl_weight: float | complex = 1.0,
    mass_weight: float | complex = 1.0,
) -> tuple[list[dict[int, float | complex]], list[dict], float, list[dict]]:
    """Assemble a hierarchical second-order H(curl) scaffold on triangles.

    The basis uses two edge moment DOFs per geometric edge and two interior
    bubble-like DOFs per cell. This is intentionally explicit and artifact-rich
    so the next promotion step can replace the scaffold basis with a formally
    complete high-order Nedelec family without changing global DOF plumbing.
    """
    local_edge_pairs = [(0, 1), (1, 2), (2, 0)]
    edge_dof_index: dict[tuple[int, int, int], int] = {}
    dofs: list[dict] = []
    cell_interior_dofs: dict[tuple[int, int], int] = {}
    for cell_id, triangle in enumerate(triangles):
        vertices = triangle[:3]
        if len(vertices) < 3:
            continue
        for local_a, local_b in local_edge_pairs:
            a = vertices[local_a]
            b = vertices[local_b]
            edge = (a, b) if a < b else (b, a)
            for moment in range(2):
                key = (edge[0], edge[1], moment)
                if key not in edge_dof_index:
                    edge_dof_index[key] = len(dofs)
                    dofs.append({"kind": "edge_moment", "edge": [edge[0], edge[1]], "moment": moment})
        for interior in range(2):
            cell_interior_dofs[(cell_id, interior)] = len(dofs)
            dofs.append({"kind": "cell_interior", "cell_id": cell_id, "moment": interior, "vertices": vertices})

    stiffness: list[dict[int, float | complex]] = [dict() for _ in dofs]
    total_area = 0.0
    element_records: list[dict] = []
    for cell_id, triangle in enumerate(triangles):
        vertices = triangle[:3]
        geometry = _triangle_geometry(points, vertices)
        if geometry is None:
            continue
        area, grad_lambda = geometry
        local_dofs: list[int] = []
        local_basis_descriptors: list[dict] = []
        orientation_signs: list[float] = []
        for edge_number, (local_a, local_b) in enumerate(local_edge_pairs):
            a = vertices[local_a]
            b = vertices[local_b]
            edge = (a, b) if a < b else (b, a)
            sign = 1.0 if (a, b) == edge else -1.0
            for moment in range(2):
                local_dofs.append(edge_dof_index[(edge[0], edge[1], moment)])
                orientation_signs.append(sign)
                local_basis_descriptors.append(
                    {"kind": "edge_moment", "edge_number": edge_number, "lambda_scale": local_a if moment == 0 else local_b}
                )
        for interior in range(2):
            local_dofs.append(cell_interior_dofs[(cell_id, interior)])
            orientation_signs.append(1.0)
            local_basis_descriptors.append({"kind": "cell_interior", "interior": interior})

        local_size = len(local_dofs)
        local_matrix = [[0.0 + 0.0j for _ in range(local_size)] for _ in range(local_size)]
        quadrature_records: list[dict] = []
        for r, s, weight in _triangle_quadrature(2):
            lambdas = [1.0 - r - s, r, s]
            basis_values: list[list[float]] = []
            basis_curls: list[float] = []
            p1_values: list[list[float]] = []
            p1_curls: list[float] = []
            for local_a, local_b in local_edge_pairs:
                value, curl = _nedelec_p1_value_and_curl(lambdas, grad_lambda, local_a, local_b)
                p1_values.append(value)
                p1_curls.append(curl)
            for edge_number, (local_a, local_b) in enumerate(local_edge_pairs):
                for scalar_index in (local_a, local_b):
                    value, curl = _scaled_nedelec_value_and_curl(
                        lambdas[scalar_index],
                        grad_lambda[scalar_index],
                        p1_values[edge_number],
                        p1_curls[edge_number],
                    )
                    basis_values.append(value)
                    basis_curls.append(curl)
            bubble = lambdas[0] * lambdas[1] * lambdas[2]
            bubble_grad = [
                lambdas[1] * lambdas[2] * grad_lambda[0][0]
                + lambdas[0] * lambdas[2] * grad_lambda[1][0]
                + lambdas[0] * lambdas[1] * grad_lambda[2][0],
                lambdas[1] * lambdas[2] * grad_lambda[0][1]
                + lambdas[0] * lambdas[2] * grad_lambda[1][1]
                + lambdas[0] * lambdas[1] * grad_lambda[2][1],
            ]
            for interior_base in (0, 1):
                value, curl = _scaled_nedelec_value_and_curl(
                    bubble,
                    bubble_grad,
                    p1_values[interior_base],
                    p1_curls[interior_base],
                )
                basis_values.append(value)
                basis_curls.append(curl)
            scaled_weight = 2.0 * area * weight
            for i in range(local_size):
                for j in range(local_size):
                    mass = basis_values[i][0] * basis_values[j][0] + basis_values[i][1] * basis_values[j][1]
                    local_matrix[i][j] += orientation_signs[i] * orientation_signs[j] * (
                        scaled_weight * (mass_weight * mass + curl_weight * basis_curls[i] * basis_curls[j])
                    )
            quadrature_records.append(
                {
                    "r": r,
                    "s": s,
                    "weight": weight,
                    "basis_values": basis_values,
                    "curl_basis": basis_curls,
                }
            )
        for local_i, global_i in enumerate(local_dofs):
            for local_j, global_j in enumerate(local_dofs):
                value = local_matrix[local_i][local_j]
                stiffness[global_i][global_j] = stiffness[global_i].get(global_j, 0.0) + value
        element_records.append(
            {
                "vertices": vertices,
                "dofs": local_dofs,
                "area": area,
                "basis": "nedelec_first_kind_order2_hierarchical_scaffold_triangle",
                "edge_moment_dofs_per_edge": 2,
                "cell_interior_dofs": 2,
                "orientation_signs": orientation_signs,
                "local_basis": local_basis_descriptors,
                "local_matrix": [[_json_number(value) for value in row] for row in local_matrix],
                "quadrature": quadrature_records,
            }
        )
        total_area += area
    return stiffness, dofs, total_area, element_records


def _sparse_residual_norm(
    stiffness: list[dict[int, float]],
    rhs: list[float],
    solution: list[float],
    boundary_nodes: set[int],
) -> float:
    residual_sq = 0.0
    rhs_sq = 0.0
    for row, entries in enumerate(stiffness):
        if row in boundary_nodes:
            residual = solution[row]
            residual_sq += residual * residual
            continue
        au = sum(value * solution[col] for col, value in entries.items())
        residual = au - rhs[row]
        residual_sq += residual * residual
        rhs_sq += rhs[row] * rhs[row]
    return math.sqrt(residual_sq) / (math.sqrt(rhs_sq) + 1e-12)


def _jacobi_sparse_solve(
    stiffness: list[dict[int, float]],
    rhs: list[float],
    boundary_nodes: set[int],
    max_iterations: int = 10000,
    tolerance: float = 1e-8,
) -> tuple[list[float], list[dict[str, float | int]]]:
    solution = [0.0 for _ in rhs]
    diagonal = [row.get(index, 0.0) for index, row in enumerate(stiffness)]
    history: list[dict[str, float | int]] = []
    for iteration in range(1, max_iterations + 1):
        update_sq = 0.0
        norm_sq = 0.0
        previous = solution[:]
        for row, entries in enumerate(stiffness):
            if row in boundary_nodes:
                solution[row] = 0.0
                continue
            diag = diagonal[row]
            if abs(diag) < 1e-14:
                continue
            off_diag = sum(value * previous[col] for col, value in entries.items() if col != row)
            new_value = (rhs[row] - off_diag) / diag
            # Mild damping keeps the pure-Python baseline robust across coarse meshes.
            new_value = 0.8 * new_value + 0.2 * previous[row]
            update_sq += (new_value - previous[row]) ** 2
            norm_sq += new_value * new_value
            solution[row] = new_value
        residual = _sparse_residual_norm(stiffness, rhs, solution, boundary_nodes)
        update_norm = math.sqrt(update_sq) / (math.sqrt(norm_sq) + 1e-12)
        if iteration == 1 or iteration % 25 == 0 or residual < tolerance or update_norm < 1e-10:
            history.append({"iteration": iteration, "normalized_fem_residual": residual, "relative_update": update_norm})
        if residual < tolerance or update_norm < 1e-10:
            break
    return solution, history


def _sparse_matvec(stiffness: list[dict[int, float]], vector: list[float], boundary_nodes: set[int]) -> list[float]:
    output = [0.0 for _ in vector]
    for row, entries in enumerate(stiffness):
        if row in boundary_nodes:
            output[row] = vector[row]
        else:
            output[row] = sum(value * vector[col] for col, value in entries.items())
    return output


def _cg_sparse_solve(
    stiffness: list[dict[int, float]],
    rhs: list[float],
    boundary_nodes: set[int],
    max_iterations: int = 5000,
    tolerance: float = 1e-8,
) -> tuple[list[float], list[dict[str, float | int]]]:
    solution = [0.0 for _ in rhs]
    rhs_effective = [0.0 if index in boundary_nodes else value for index, value in enumerate(rhs)]
    residual = rhs_effective[:]
    direction = residual[:]
    residual_sq = sum(value * value for value in residual)
    rhs_norm = math.sqrt(sum(value * value for value in rhs_effective)) + 1e-12
    history: list[dict[str, float | int]] = []
    if math.sqrt(residual_sq) / rhs_norm < tolerance:
        return solution, [{"iteration": 0, "normalized_fem_residual": 0.0, "relative_update": 0.0}]
    for iteration in range(1, max_iterations + 1):
        ad = _sparse_matvec(stiffness, direction, boundary_nodes)
        denom = sum(direction[index] * ad[index] for index in range(len(direction)))
        if abs(denom) < 1e-24:
            break
        alpha = residual_sq / denom
        update_sq = 0.0
        solution_sq = 0.0
        for index in range(len(solution)):
            if index in boundary_nodes:
                solution[index] = 0.0
                continue
            delta = alpha * direction[index]
            solution[index] += delta
            residual[index] -= alpha * ad[index]
            update_sq += delta * delta
            solution_sq += solution[index] * solution[index]
        new_residual_sq = sum(value * value for value in residual)
        normalized = math.sqrt(new_residual_sq) / rhs_norm
        update_norm = math.sqrt(update_sq) / (math.sqrt(solution_sq) + 1e-12)
        if iteration == 1 or iteration % 25 == 0 or normalized < tolerance or update_norm < 1e-10:
            history.append({"iteration": iteration, "normalized_fem_residual": normalized, "relative_update": update_norm})
        if normalized < tolerance or update_norm < 1e-10:
            break
        beta = new_residual_sq / (residual_sq + 1e-30)
        direction = [residual[index] + beta * direction[index] for index in range(len(direction))]
        for index in boundary_nodes:
            direction[index] = 0.0
            residual[index] = 0.0
        residual_sq = new_residual_sq
    return solution, history


def _complex_sparse_residual_norm(
    stiffness: list[dict[int, complex]],
    rhs: list[complex],
    solution: list[complex],
    boundary_nodes: set[int],
) -> float:
    residual_sq = 0.0
    rhs_sq = 0.0
    for row, entries in enumerate(stiffness):
        if row in boundary_nodes:
            residual = solution[row]
        else:
            residual = sum(value * solution[col] for col, value in entries.items()) - rhs[row]
            rhs_sq += abs(rhs[row]) ** 2
        residual_sq += abs(residual) ** 2
    return math.sqrt(residual_sq) / (math.sqrt(rhs_sq) + 1e-12)


def _dense_complex_solve(
    stiffness: list[dict[int, complex]],
    rhs: list[complex],
    boundary_nodes: set[int],
) -> tuple[list[complex], list[dict[str, float | int]]]:
    size = len(rhs)
    matrix = [[0.0j for _ in range(size)] for _ in range(size)]
    vector = rhs[:]
    for row, entries in enumerate(stiffness):
        if row in boundary_nodes:
            matrix[row][row] = 1.0 + 0.0j
            vector[row] = 0.0j
            continue
        for col, value in entries.items():
            matrix[row][col] = value
    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(matrix[row][col]))
        if abs(matrix[pivot][col]) < 1e-18:
            continue
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        vector[col], vector[pivot] = vector[pivot], vector[col]
        pivot_value = matrix[col][col]
        matrix[col] = [value / pivot_value for value in matrix[col]]
        vector[col] /= pivot_value
        for row in range(size):
            if row == col:
                continue
            factor = matrix[row][col]
            if abs(factor) <= 1e-24:
                continue
            matrix[row] = [value - factor * matrix[col][idx] for idx, value in enumerate(matrix[row])]
            vector[row] -= factor * vector[col]
    for index in boundary_nodes:
        vector[index] = 0.0j
    residual = _complex_sparse_residual_norm(stiffness, rhs, vector, boundary_nodes)
    return vector, [{"iteration": 1, "normalized_fem_residual": residual, "relative_update": 0.0}]


def solve_mesh_fem_linear_elasticity(taps_problem: TAPSProblem) -> tuple[TAPSResultArtifacts, TAPSResidualReport]:
    """Assemble and solve a P1 triangle 2D linear-elasticity system from mesh_graph."""
    graph = _load_mesh_graph(taps_problem)
    if graph is None:
        raise ValueError("mesh_graph encoding is required for mesh FEM linear-elasticity solver.")
    points = graph["points"]
    triangles = _triangle_cells(graph)
    if not triangles:
        raise ValueError("mesh_graph contains no triangle cells for FEM assembly.")
    output_dir = project_root() / "scratch" / _safe(taps_problem.problem_id) / "taps_generic"
    output_dir.mkdir(parents=True, exist_ok=True)

    young_modulus = _coefficient_number(taps_problem, {"e", "young_modulus", "youngs_modulus", "young's_modulus"}, 1.0)
    poisson_ratio = _coefficient_number(taps_problem, {"nu", "poisson_ratio", "poissons_ratio", "poisson's_ratio"}, 0.3)
    constitutive_model = _coefficient_string(taps_problem, {"constitutive_model", "stress_model"}, "plane_stress").lower()
    if constitutive_model not in {"plane_stress", "plane_strain"}:
        constitutive_model = "plane_stress"
    stiffness, lumped_mass, total_area, element_records = _assemble_triangle_elasticity_stiffness(
        points,
        triangles,
        young_modulus=young_modulus,
        poisson_ratio=poisson_ratio,
        model=constitutive_model,
    )
    basis_order = max((int(str(element.get("basis", "p1_vector_triangle")).split("_", 1)[0].replace("p", "")) for element in element_records), default=1)
    boundary_nodes = set(int(node) for node in graph.get("boundary_nodes", []))
    boundary_dofs = {2 * node + component for node in boundary_nodes for component in (0, 1)}
    body_force = _coefficient_vector(taps_problem, {"body_force", "b", "gravity"}, [0.0, -1.0])
    if len(body_force) == 1:
        body_force = [0.0, body_force[0]]
    elif len(body_force) < 2:
        body_force = [0.0, -1.0]
    rhs = [0.0 for _ in range(2 * len(points))]
    for node, mass in enumerate(lumped_mass):
        if node in boundary_nodes:
            continue
        rhs[2 * node] = mass * body_force[0]
        rhs[2 * node + 1] = mass * body_force[1]
    solution, history = _jacobi_sparse_solve(stiffness, rhs, boundary_dofs, max_iterations=20000, tolerance=1e-8)
    final_residual = _sparse_residual_norm(stiffness, rhs, solution, boundary_dofs)
    final_update = float(history[-1]["relative_update"]) if history else float("inf")
    converged = final_residual < 1e-8 or final_update < 1e-10

    nonzero_entries = sum(len(row) for row in stiffness)
    displacement = [[solution[2 * node], solution[2 * node + 1]] for node in range(len(points))]
    operator_payload = {
        "type": f"triangle_p{basis_order}_fem_linear_elasticity",
        "basis_order": basis_order,
        "assembly": "constant_strain_triangle_galerkin",
        "operator": "int epsilon(v)^T C epsilon(u) dOmega = int v dot b dOmega",
        "source_mesh": graph.get("source_mesh"),
        "node_count": len(points),
        "triangle_count": len(triangles),
        "dof_count": len(solution),
        "boundary_node_count": len(boundary_nodes),
        "boundary_dof_count": len(boundary_dofs),
        "total_area": total_area,
        "nonzero_entries": nonzero_entries,
        "material": {
            "young_modulus": young_modulus,
            "poisson_ratio": poisson_ratio,
            "constitutive_model": constitutive_model,
            "body_force": body_force[:2],
        },
        "elements": element_records,
        "stiffness_rows": [{str(col): value for col, value in row.items()} for row in stiffness],
        "rhs": rhs,
    }
    solution_payload = {
        "field": taps_problem.weak_form.trial_fields[0] if taps_problem.weak_form and taps_problem.weak_form.trial_fields else "u",
        "field_kind": "vector",
        "components": ["ux", "uy"],
        "points": points,
        "values": displacement,
    }
    residual_payload = {
        "family": "linear_elasticity",
        "normalized_fem_residual": final_residual,
        "relative_update": final_update,
        "iterations": int(history[-1]["iteration"]) if history else 0,
        "iteration_history": history,
        "converged": converged,
    }
    operator_path = output_dir / "mesh_fem_elasticity_operator.json"
    solution_path = output_dir / "mesh_fem_elasticity_displacement.json"
    residual_path = output_dir / "mesh_fem_elasticity_iteration_history.json"
    operator_path.write_text(json.dumps(operator_payload, indent=2), encoding="utf-8")
    solution_path.write_text(json.dumps(solution_payload, indent=2), encoding="utf-8")
    residual_path.write_text(json.dumps(residual_payload, indent=2), encoding="utf-8")
    artifacts = TAPSResultArtifacts(
        factor_matrices=[ArtifactRef(uri=str(operator_path), kind="taps_mesh_fem_elasticity_operator", format="json")],
        reconstruction_metadata=ArtifactRef(uri=str(solution_path), kind="taps_mesh_fem_displacement_field", format="json"),
        residual_history=ArtifactRef(uri=str(residual_path), kind="taps_iteration_history", format="json"),
    )
    return (
        artifacts,
        TAPSResidualReport(
            residuals={
                "normalized_fem_residual": final_residual,
                "relative_update": final_update,
                "fem_nodes": float(len(points)),
                "fem_triangles": float(len(triangles)),
                "fem_dofs": float(len(solution)),
                "fem_nonzeros": float(nonzero_entries),
                "fem_basis_order": float(basis_order),
            },
            rank=taps_problem.basis.tensor_rank,
            converged=converged,
            recommended_action="accept" if converged else "refine_axes",
        ),
    )


def solve_mesh_fem_poisson(taps_problem: TAPSProblem) -> tuple[TAPSResultArtifacts, TAPSResidualReport]:
    """Assemble and solve a P1/P2 triangle FEM-like Poisson system from mesh_graph."""
    graph = _load_mesh_graph(taps_problem)
    if graph is None:
        raise ValueError("mesh_graph encoding is required for mesh FEM Poisson solver.")
    points = graph["points"]
    triangles = _triangle_cells(graph)
    if not triangles:
        raise ValueError("mesh_graph contains no triangle cells for FEM assembly.")
    output_dir = project_root() / "scratch" / _safe(taps_problem.problem_id) / "taps_generic"
    output_dir.mkdir(parents=True, exist_ok=True)

    stiffness, lumped_mass, total_area, element_records = _assemble_triangle_stiffness(points, triangles)
    basis_order = max((int(str(element.get("basis", "p1_triangle")).split("_", 1)[0].replace("p", "")) for element in element_records), default=1)
    boundary_nodes = set(int(node) for node in graph.get("boundary_nodes", []))
    rhs = []
    for index, point in enumerate(points):
        if index in boundary_nodes:
            rhs.append(0.0)
        else:
            x = float(point[0])
            y = float(point[1])
            rhs.append(lumped_mass[index] * math.sin(math.pi * x) * math.sin(math.pi * y))
    solution, history = _jacobi_sparse_solve(stiffness, rhs, boundary_nodes)
    final_residual = _sparse_residual_norm(stiffness, rhs, solution, boundary_nodes)
    final_update = float(history[-1]["relative_update"]) if history else float("inf")
    converged = final_residual < 1e-8 or final_update < 1e-10

    nonzero_entries = sum(len(row) for row in stiffness)
    operator_payload = {
        "type": f"triangle_p{basis_order}_fem_poisson",
        "basis_order": basis_order,
        "assembly": "cell_gradient_galerkin",
        "operator": "int grad(v) dot grad(u) dOmega = int v f dOmega",
        "source_mesh": graph.get("source_mesh"),
        "node_count": len(points),
        "triangle_count": len(triangles),
        "boundary_node_count": len(boundary_nodes),
        "total_area": total_area,
        "nonzero_entries": nonzero_entries,
        "elements": element_records,
        "stiffness_rows": [{str(col): value for col, value in row.items()} for row in stiffness],
        "rhs": rhs,
    }
    solution_payload = {
        "field": taps_problem.weak_form.trial_fields[0] if taps_problem.weak_form and taps_problem.weak_form.trial_fields else "u",
        "points": points,
        "values": solution,
    }
    residual_payload = {
        "family": "poisson",
        "normalized_fem_residual": final_residual,
        "relative_update": final_update,
        "iterations": int(history[-1]["iteration"]) if history else 0,
        "iteration_history": history,
        "converged": converged,
    }
    operator_path = output_dir / "mesh_fem_operator.json"
    solution_path = output_dir / "mesh_fem_solution_field.json"
    residual_path = output_dir / "mesh_fem_iteration_history.json"
    operator_path.write_text(json.dumps(operator_payload, indent=2), encoding="utf-8")
    solution_path.write_text(json.dumps(solution_payload, indent=2), encoding="utf-8")
    residual_path.write_text(json.dumps(residual_payload, indent=2), encoding="utf-8")
    artifacts = TAPSResultArtifacts(
        factor_matrices=[ArtifactRef(uri=str(operator_path), kind="taps_mesh_fem_operator", format="json")],
        reconstruction_metadata=ArtifactRef(uri=str(solution_path), kind="taps_mesh_fem_solution_field", format="json"),
        residual_history=ArtifactRef(uri=str(residual_path), kind="taps_iteration_history", format="json"),
    )
    return (
        artifacts,
        TAPSResidualReport(
            residuals={
                "normalized_fem_residual": final_residual,
                "relative_update": final_update,
                "fem_nodes": float(len(points)),
                "fem_triangles": float(len(triangles)),
                "fem_nonzeros": float(nonzero_entries),
                "fem_basis_order": float(basis_order),
            },
            rank=taps_problem.basis.tensor_rank,
            converged=converged,
            recommended_action="accept" if converged else "refine_axes",
        ),
    )


def solve_mesh_fem_em_curl_curl(taps_problem: TAPSProblem) -> tuple[TAPSResultArtifacts, TAPSResidualReport]:
    """Solve a 2D electromagnetic curl-curl problem with first-order Nedelec edge elements."""
    graph = _load_mesh_graph(taps_problem)
    if graph is None:
        raise ValueError("mesh_graph encoding is required for mesh FEM EM curl-curl solver.")
    points = graph["points"]
    triangles = _triangle_cells(graph)
    if not triangles:
        raise ValueError("mesh_graph contains no triangle cells for FEM assembly.")
    output_dir = project_root() / "scratch" / _safe(taps_problem.problem_id) / "taps_generic"
    output_dir.mkdir(parents=True, exist_ok=True)

    mu_r = _coefficient_complex(taps_problem, {"mu_r", "relative_permeability", "permeability"}, 1.0 + 0.0j)
    eps_r = _coefficient_complex(taps_problem, {"eps_r", "epsilon_r", "relative_permittivity", "permittivity"}, 1.0 + 0.0j)
    wave_number = _coefficient_complex(taps_problem, {"k0", "k", "wave_number", "wavenumber"}, 0.5 + 0.0j)
    source_amplitude = _coefficient_complex(taps_problem, {"source", "current_source", "jz"}, 1.0 + 0.0j)
    is_complex_frequency_domain = any(abs(value.imag) > 1e-14 for value in [mu_r, eps_r, wave_number, source_amplitude])
    curl_weight = 1.0 / (mu_r if abs(mu_r) > 1e-12 else 1e-12)
    mass_weight = wave_number * wave_number * eps_r

    stiffness, dof_entities, total_area, element_records = _assemble_triangle_nedelec_curl_curl(
        points,
        triangles,
        curl_weight=curl_weight,
        mass_weight=mass_weight,
    )
    basis_order = 2 if any("order2" in str(element.get("basis", "")) for element in element_records) else 1
    geometric_edges = _nedelec_geometric_edges(dof_entities)
    boundary_nodes = set(int(node) for node in graph.get("boundary_nodes", []))
    boundary_policy = _em_tangential_boundary_policy(taps_problem)
    boundary_parameters = _em_boundary_parameters(taps_problem)
    active_boundary_edge_tuples = _em_boundary_edge_tuples_for_policy(taps_problem, graph, geometric_edges, boundary_nodes, boundary_policy)
    active_boundary_dofs = _nedelec_dof_ids_for_edges(dof_entities, active_boundary_edge_tuples)
    active_boundary_geometric_edges = _nedelec_edges_for_dof_ids(dof_entities, active_boundary_dofs)
    active_boundary_edges = _em_boundary_edge_ids_for_policy(taps_problem, graph, geometric_edges, boundary_nodes, boundary_policy)
    if boundary_policy.startswith("pec"):
        boundary_dofs = active_boundary_dofs
    else:
        boundary_dofs = set()
    rhs = [0.0j if is_complex_frequency_domain else 0.0 for _ in dof_entities]
    for dof_id, dof_entity in enumerate(dof_entities):
        if dof_id in boundary_dofs:
            continue
        if isinstance(dof_entity, tuple) and len(dof_entity) >= 2:
            a, b = int(dof_entity[0]), int(dof_entity[1])
            moment_scale = 1.0
        elif isinstance(dof_entity, dict) and dof_entity.get("kind") == "edge_moment":
            raw_edge = dof_entity.get("edge", [0, 0])
            a, b = int(raw_edge[0]), int(raw_edge[1])
            moment_scale = 1.0 / float(int(dof_entity.get("moment", 0)) + 1)
        elif isinstance(dof_entity, dict) and dof_entity.get("kind") == "cell_interior":
            raw_vertices = dof_entity.get("vertices", [0, 0, 0])
            coords = [points[int(node)] for node in raw_vertices[:3]]
            midpoint_x = sum(float(point[0]) for point in coords) / 3.0
            midpoint_y = sum(float(point[1]) for point in coords) / 3.0
            rhs[dof_id] = 0.1 * source_amplitude * math.sin(math.pi * midpoint_x) * math.sin(math.pi * midpoint_y)
            continue
        else:
            continue
        ax, ay = float(points[a][0]), float(points[a][1])
        bx, by = float(points[b][0]), float(points[b][1])
        length = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
        midpoint_x = 0.5 * (ax + bx)
        midpoint_y = 0.5 * (ay + by)
        rhs[dof_id] = moment_scale * source_amplitude * length * math.sin(math.pi * midpoint_x) * math.sin(math.pi * midpoint_y)
        if boundary_policy in {"absorbing", "impedance", "port"} and dof_id in active_boundary_dofs:
            impedance = boundary_parameters["impedance"] if abs(boundary_parameters["impedance"]) > 0.0 else 1.0 + 0.0j
            stiffness[dof_id][dof_id] = stiffness[dof_id].get(dof_id, 0.0) + impedance * length
            if boundary_policy == "port":
                rhs[dof_id] += moment_scale * boundary_parameters["port_amplitude"] * length
                is_complex_frequency_domain = is_complex_frequency_domain or abs(boundary_parameters["port_amplitude"].imag) > 1e-14
            is_complex_frequency_domain = is_complex_frequency_domain or abs(impedance.imag) > 1e-14
    if is_complex_frequency_domain:
        complex_stiffness = [{col: complex(value) for col, value in row.items()} for row in stiffness]
        complex_rhs = [complex(value) for value in rhs]
        solution, history = _dense_complex_solve(complex_stiffness, complex_rhs, boundary_dofs)
        final_residual = _complex_sparse_residual_norm(complex_stiffness, complex_rhs, solution, boundary_dofs)
    else:
        real_stiffness = [{col: float(value.real if isinstance(value, complex) else value) for col, value in row.items()} for row in stiffness]
        real_rhs = [float(value.real if isinstance(value, complex) else value) for value in rhs]
        solution, history = _cg_sparse_solve(real_stiffness, real_rhs, boundary_dofs, max_iterations=5000, tolerance=1e-8)
        final_residual = _sparse_residual_norm(real_stiffness, real_rhs, solution, boundary_dofs)
    final_update = float(history[-1]["relative_update"]) if history else float("inf")
    converged = final_residual < 1e-8 or final_update < 1e-10

    nonzero_entries = sum(len(row) for row in stiffness)
    edge_dof_count = sum(
        1 for entity in dof_entities if isinstance(entity, tuple) or (isinstance(entity, dict) and entity.get("kind") == "edge_moment")
    )
    cell_interior_dof_count = sum(1 for entity in dof_entities if isinstance(entity, dict) and entity.get("kind") == "cell_interior")
    boundary_edge_tuples = {
        tuple(entity["edge"])
        for index, entity in enumerate(dof_entities)
        if index in boundary_dofs and isinstance(entity, dict) and entity.get("kind") == "edge_moment"
    }
    dof_payload = [
        {"kind": "edge", "edge": [int(entity[0]), int(entity[1])]} if isinstance(entity, tuple) else entity
        for entity in dof_entities
    ]
    operator_payload = {
        "type": f"triangle_nedelec_order{basis_order}_em_curl_curl",
        "basis_order": basis_order,
        "assembly": "nedelec_first_kind_edge_element_hcurl",
        "operator": "int mu^-1 curl(v) curl(E) dOmega + int k0^2 eps v dot E dOmega = int v dot J dOmega",
        "source_mesh": graph.get("source_mesh"),
        "node_count": len(points),
        "dof_count": len(dof_entities),
        "edge_dof_count": edge_dof_count,
        "cell_interior_dof_count": cell_interior_dof_count,
        "triangle_count": len(triangles),
        "boundary_edge_count": len(boundary_dofs) if basis_order == 1 else len(boundary_edge_tuples),
        "boundary_dof_count": len(boundary_dofs),
        "active_boundary_edge_count": len(active_boundary_edges),
        "active_boundary_geometric_edge_count": len(active_boundary_geometric_edges),
        "active_boundary_dof_count": len(active_boundary_dofs),
        "total_area": total_area,
        "nonzero_entries": nonzero_entries,
        "material": {
            "relative_permeability": _json_number(mu_r),
            "relative_permittivity": _json_number(eps_r),
            "wave_number": _json_number(wave_number),
            "source_amplitude": _json_number(source_amplitude),
            "boundary_impedance": _json_number(boundary_parameters["impedance"]),
            "port_amplitude": _json_number(boundary_parameters["port_amplitude"]),
            "complex_frequency_domain": is_complex_frequency_domain,
        },
        "hcurl_scaffold": {
            "status": "nedelec_order1_edge_element" if basis_order == 1 else "nedelec_order2_hierarchical_scaffold",
            "tangential_continuity": "global edge DOFs with orientation signs",
            "boundary_condition": boundary_policy,
            "edge_dofs_required": True,
            "high_order_boundary_dofs": basis_order > 1,
        },
        "edges": [[a, b] for a, b in geometric_edges],
        "dofs": dof_payload,
        "boundary_dofs": sorted(boundary_dofs),
        "active_boundary_edges": sorted(active_boundary_edges),
        "active_boundary_geometric_edges": [[a, b] for a, b in sorted(active_boundary_geometric_edges)],
        "active_boundary_dofs": sorted(active_boundary_dofs),
        "elements": element_records,
        "stiffness_rows": _json_sparse_rows(stiffness),
        "rhs": _json_vector(rhs),
    }
    solution_payload = {
        "field": taps_problem.weak_form.trial_fields[0] if taps_problem.weak_form and taps_problem.weak_form.trial_fields else "E",
        "field_kind": "hcurl_edge_field",
        "components": ["tangential_edge_dof"] if basis_order == 1 else ["hcurl_dof"],
        "points": points,
        "edges": [[a, b] for a, b in geometric_edges],
        "dofs": dof_payload,
        "values": _json_vector(solution),
    }
    residual_payload = {
        "family": "maxwell",
        "normalized_fem_residual": final_residual,
        "relative_update": final_update,
        "iterations": int(history[-1]["iteration"]) if history else 0,
        "iteration_history": history,
        "converged": converged,
    }
    operator_path = output_dir / "mesh_fem_em_curl_curl_operator.json"
    solution_path = output_dir / "mesh_fem_em_ez_field.json"
    residual_path = output_dir / "mesh_fem_em_iteration_history.json"
    operator_path.write_text(json.dumps(operator_payload, indent=2), encoding="utf-8")
    solution_path.write_text(json.dumps(solution_payload, indent=2), encoding="utf-8")
    residual_path.write_text(json.dumps(residual_payload, indent=2), encoding="utf-8")
    artifacts = TAPSResultArtifacts(
        factor_matrices=[ArtifactRef(uri=str(operator_path), kind="taps_mesh_fem_em_curl_curl_operator", format="json")],
        reconstruction_metadata=ArtifactRef(uri=str(solution_path), kind="taps_mesh_fem_em_field", format="json"),
        residual_history=ArtifactRef(uri=str(residual_path), kind="taps_iteration_history", format="json"),
    )
    return (
        artifacts,
        TAPSResidualReport(
            residuals={
                "normalized_fem_residual": final_residual,
                "relative_update": final_update,
                "fem_nodes": float(len(points)),
                "fem_edge_dofs": float(edge_dof_count),
                "fem_dofs": float(len(dof_entities)),
                "fem_triangles": float(len(triangles)),
                "fem_nonzeros": float(nonzero_entries),
                "fem_basis_order": float(basis_order),
            },
            rank=taps_problem.basis.tensor_rank,
            converged=converged,
            recommended_action="accept" if converged else "refine_axes",
        ),
    )


def solve_graph_poisson(taps_problem: TAPSProblem) -> tuple[TAPSResultArtifacts, TAPSResidualReport]:
    """Solve graph-Laplacian Poisson on a mesh_graph encoding."""
    graph = _load_mesh_graph(taps_problem)
    if graph is None:
        raise ValueError("mesh_graph encoding is required for graph Poisson solver.")
    output_dir = project_root() / "scratch" / _safe(taps_problem.problem_id) / "taps_generic"
    output_dir.mkdir(parents=True, exist_ok=True)

    points = graph["points"]
    edges = graph["edges"]
    node_count = int(graph["node_count"])
    boundary_nodes = set(int(node) for node in graph.get("boundary_nodes", []))
    adjacency = [[] for _ in range(node_count)]
    for raw_a, raw_b in edges:
        a = int(raw_a)
        b = int(raw_b)
        adjacency[a].append(b)
        adjacency[b].append(a)

    rhs = []
    for index, point in enumerate(points):
        if index in boundary_nodes:
            rhs.append(0.0)
        else:
            x = float(point[0])
            y = float(point[1])
            rhs.append(math.sin(math.pi * x) * math.sin(math.pi * y))
    solution = [0.0 for _ in range(node_count)]
    history: list[dict[str, float | int]] = []
    for iteration in range(1, 5000 + 1):
        update_sq = 0.0
        norm_sq = 0.0
        for node in range(node_count):
            if node in boundary_nodes:
                continue
            degree = len(adjacency[node])
            if degree == 0:
                continue
            new_value = (rhs[node] + sum(solution[neighbor] for neighbor in adjacency[node])) / degree
            update_sq += (new_value - solution[node]) ** 2
            norm_sq += new_value * new_value
            solution[node] = new_value
        residual = _graph_residual_norm(node_count, edges, boundary_nodes, rhs, solution)
        update_norm = math.sqrt(update_sq) / (math.sqrt(norm_sq) + 1e-12)
        if iteration == 1 or iteration % 25 == 0 or residual < 1e-8 or update_norm < 1e-10:
            history.append({"iteration": iteration, "normalized_graph_residual": residual, "relative_update": update_norm})
        if residual < 1e-8 or update_norm < 1e-10:
            break

    final_residual = _graph_residual_norm(node_count, edges, boundary_nodes, rhs, solution)
    final_update = float(history[-1]["relative_update"]) if history else float("inf")
    converged = final_residual < 1e-8 or final_update < 1e-10
    operator_payload = {
        "type": "graph_laplacian_poisson",
        "operator": "L_graph u = f with Dirichlet boundary nodes",
        "source_mesh": graph.get("source_mesh"),
        "node_count": node_count,
        "edge_count": len(edges),
        "boundary_node_count": len(boundary_nodes),
        "edges": edges,
    }
    solution_payload = {
        "field": taps_problem.weak_form.trial_fields[0] if taps_problem.weak_form and taps_problem.weak_form.trial_fields else "u",
        "points": points,
        "values": solution,
    }
    residual_payload = {
        "family": "poisson",
        "normalized_graph_residual": final_residual,
        "relative_update": final_update,
        "iterations": int(history[-1]["iteration"]) if history else 0,
        "iteration_history": history,
        "converged": converged,
    }
    operator_path = output_dir / "graph_operator.json"
    solution_path = output_dir / "graph_solution_field.json"
    residual_path = output_dir / "graph_iteration_history.json"
    operator_path.write_text(json.dumps(operator_payload, indent=2), encoding="utf-8")
    solution_path.write_text(json.dumps(solution_payload, indent=2), encoding="utf-8")
    residual_path.write_text(json.dumps(residual_payload, indent=2), encoding="utf-8")
    artifacts = TAPSResultArtifacts(
        factor_matrices=[ArtifactRef(uri=str(operator_path), kind="taps_graph_operator", format="json")],
        reconstruction_metadata=ArtifactRef(uri=str(solution_path), kind="taps_graph_solution_field", format="json"),
        residual_history=ArtifactRef(uri=str(residual_path), kind="taps_iteration_history", format="json"),
    )
    return (
        artifacts,
        TAPSResidualReport(
            residuals={
                "normalized_graph_residual": final_residual,
                "relative_update": final_update,
                "graph_nodes": float(node_count),
                "graph_edges": float(len(edges)),
            },
            rank=taps_problem.basis.tensor_rank,
            converged=converged,
            recommended_action="accept" if converged else "refine_axes",
        ),
    )


def solve_scalar_elliptic_1d(taps_problem: TAPSProblem) -> tuple[TAPSResultArtifacts, TAPSResidualReport]:
    """Execute first generic TAPS weak-form kernel for scalar 1D linear PDEs.

    The kernel assembles the model operator:

        -d/dx(k du/dx) + c u = f

    with homogeneous Dirichlet endpoints. Poisson/diffusion use c=0,
    reaction-diffusion uses c>0, and Helmholtz uses a non-resonant c<0 proxy.
    This is deliberately small and pure Python so it can run inside agent tests.
    """
    family = _family(taps_problem)
    output_dir = project_root() / "scratch" / _safe(taps_problem.problem_id) / "taps_generic"
    output_dir.mkdir(parents=True, exist_ok=True)

    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"Generic TAPS scalar assembler does not support family={family!r}.")
    axes = _space_axes(taps_problem)
    if len(axes) != 1:
        raise ValueError(f"Generic TAPS scalar assembler currently executes exactly one space axis, got {axes}.")

    x = _axis_values(taps_problem, axes[0], default_points=64)
    if len(x) < 3:
        raise ValueError("Generic TAPS scalar assembler requires at least three grid points.")
    dx = x[1] - x[0]
    if dx <= 0.0:
        raise ValueError("Generic TAPS scalar assembler requires an increasing space axis.")

    diffusion = _material_number(taps_problem, {"k", "d", "diffusivity", "thermal_diffusivity"}, 1.0)
    reaction = _reaction_value(family)
    interior = x[1:-1]
    n = len(interior)
    lower = [-(diffusion / (dx * dx)) for _ in range(max(0, n - 1))]
    diag = [(2.0 * diffusion / (dx * dx)) + reaction for _ in range(n)]
    upper = [-(diffusion / (dx * dx)) for _ in range(max(0, n - 1))]
    source_modes = _source_modes_1d(taps_problem.basis.tensor_rank, len(x) - 2)
    rhs = [
        sum(coefficient * math.sin(mode * math.pi * value) for mode, coefficient in source_modes)
        for value in interior
    ]
    solution_interior = _thomas_solve(lower, diag, upper, rhs)
    solution = [0.0, *solution_interior, 0.0]
    normalized_residual = _residual_norm(lower, diag, upper, rhs, solution_interior)

    matrix_payload = {
        "type": "tridiagonal_scalar_elliptic_1d",
        "family": family,
        "operator": "-d/dx(k du/dx) + c u = f",
        "axis": {"name": axes[0], "values": x},
        "coefficients": {"diffusion": diffusion, "reaction": reaction},
        "source_modes": [{"mode": mode, "coefficient": coefficient} for mode, coefficient in source_modes],
        "tridiagonal": {"lower": lower, "diag": diag, "upper": upper, "rhs": rhs},
    }
    solution_payload = {
        "field": taps_problem.weak_form.trial_fields[0] if taps_problem.weak_form and taps_problem.weak_form.trial_fields else "u",
        "axis": axes[0],
        "x": x,
        "values": solution,
    }
    residual_payload = {
        "family": family,
        "normalized_linear_residual": normalized_residual,
        "separable_rank": len(source_modes),
        "converged": normalized_residual < 1e-10,
    }

    matrix_path = output_dir / "assembled_operator.json"
    solution_path = output_dir / "solution_field.json"
    residual_path = output_dir / "residual_history.json"
    matrix_path.write_text(json.dumps(matrix_payload, indent=2), encoding="utf-8")
    solution_path.write_text(json.dumps(solution_payload, indent=2), encoding="utf-8")
    residual_path.write_text(json.dumps(residual_payload, indent=2), encoding="utf-8")

    artifacts = TAPSResultArtifacts(
        factor_matrices=[ArtifactRef(uri=str(matrix_path), kind="taps_assembled_operator", format="json")],
        reconstruction_metadata=ArtifactRef(uri=str(solution_path), kind="taps_solution_field", format="json"),
        residual_history=ArtifactRef(uri=str(residual_path), kind="taps_residual_history", format="json"),
    )
    report = TAPSResidualReport(
        residuals={"normalized_linear_residual": normalized_residual},
        rank=len(source_modes),
        converged=normalized_residual < 1e-10,
        recommended_action="accept" if normalized_residual < 1e-10 else "refine_axes",
    )
    return artifacts, report


def solve_reaction_diffusion_nonlinear_1d(taps_problem: TAPSProblem) -> tuple[TAPSResultArtifacts, TAPSResidualReport]:
    """Solve a scalar 1D nonlinear reaction-diffusion model with Picard iteration.

    Model:
        -D u'' + beta u + gamma u^3 = f
    """
    output_dir = project_root() / "scratch" / _safe(taps_problem.problem_id) / "taps_generic"
    output_dir.mkdir(parents=True, exist_ok=True)
    axes = _space_axes(taps_problem)
    if len(axes) != 1:
        raise ValueError(f"1D nonlinear reaction-diffusion requires exactly one space axis, got {axes}.")

    x = _axis_values(taps_problem, axes[0], default_points=64)
    dx = x[1] - x[0]
    diffusion = _material_number(taps_problem, {"d", "diffusivity", "thermal_diffusivity"}, 1.0)
    beta = taps_problem.nonlinear.linear_reaction
    gamma = taps_problem.nonlinear.cubic_reaction
    interior = x[1:-1]
    n = len(interior)
    lower = [-(diffusion / (dx * dx)) for _ in range(max(0, n - 1))]
    base_diag = [(2.0 * diffusion / (dx * dx)) + beta for _ in range(n)]
    source_modes = _source_modes_1d(taps_problem.basis.tensor_rank, len(x) - 2)
    rhs = [
        0.2 * sum(coefficient * math.sin(mode * math.pi * value) for mode, coefficient in source_modes)
        for value in interior
    ]
    solution = _thomas_solve(lower, base_diag, lower, rhs)
    history: list[dict[str, float | int]] = []
    for iteration in range(1, taps_problem.nonlinear.max_iterations + 1):
        diag = [base_diag[i] + gamma * solution[i] * solution[i] for i in range(n)]
        candidate = _thomas_solve(lower, diag, lower, rhs)
        damped = [
            taps_problem.nonlinear.damping * candidate[i] + (1.0 - taps_problem.nonlinear.damping) * solution[i]
            for i in range(n)
        ]
        update_sq = sum((damped[i] - solution[i]) ** 2 for i in range(n))
        norm_sq = sum(damped[i] ** 2 for i in range(n))
        solution = damped
        residual = _nonlinear_residual_norm_1d(lower, base_diag, gamma, rhs, solution)
        update_norm = math.sqrt(update_sq) / (math.sqrt(norm_sq) + 1e-12)
        history.append({"iteration": iteration, "normalized_nonlinear_residual": residual, "relative_update": update_norm})
        if residual < taps_problem.nonlinear.tolerance or update_norm < taps_problem.nonlinear.tolerance:
            break

    full_solution = [0.0, *solution, 0.0]
    final_residual = _nonlinear_residual_norm_1d(lower, base_diag, gamma, rhs, solution)
    final_update = float(history[-1]["relative_update"]) if history else float("inf")
    converged = final_residual < max(1e-8, taps_problem.nonlinear.tolerance * 100.0)

    operator_payload = {
        "type": "nonlinear_reaction_diffusion_1d",
        "operator": "-D u'' + beta u + gamma u^3 = f",
        "axis": {"name": axes[0], "values": x},
        "coefficients": {"diffusion": diffusion, "linear_reaction": beta, "cubic_reaction": gamma},
        "source_modes": [{"mode": mode, "coefficient": coefficient} for mode, coefficient in source_modes],
        "method": taps_problem.nonlinear.method,
    }
    solution_payload = {
        "field": taps_problem.weak_form.trial_fields[0] if taps_problem.weak_form and taps_problem.weak_form.trial_fields else "u",
        "axis": axes[0],
        "x": x,
        "values": full_solution,
    }
    residual_payload = {
        "family": "reaction_diffusion",
        "normalized_nonlinear_residual": final_residual,
        "relative_update": final_update,
        "iterations": len(history),
        "iteration_history": history,
        "separable_rank": len(source_modes),
        "converged": converged,
    }
    operator_path = output_dir / "nonlinear_operator_1d.json"
    solution_path = output_dir / "nonlinear_solution_field_1d.json"
    residual_path = output_dir / "nonlinear_iteration_history_1d.json"
    operator_path.write_text(json.dumps(operator_payload, indent=2), encoding="utf-8")
    solution_path.write_text(json.dumps(solution_payload, indent=2), encoding="utf-8")
    residual_path.write_text(json.dumps(residual_payload, indent=2), encoding="utf-8")

    artifacts = TAPSResultArtifacts(
        factor_matrices=[ArtifactRef(uri=str(operator_path), kind="taps_nonlinear_operator", format="json")],
        reconstruction_metadata=ArtifactRef(uri=str(solution_path), kind="taps_solution_field", format="json"),
        residual_history=ArtifactRef(uri=str(residual_path), kind="taps_iteration_history", format="json"),
    )
    return (
        artifacts,
        TAPSResidualReport(
            residuals={
                "normalized_nonlinear_residual": final_residual,
                "relative_update": final_update,
                "nonlinear_iterations": float(len(history)),
            },
            rank=len(source_modes),
            converged=converged,
            recommended_action="accept" if converged else "refine_axes",
        ),
    )


def solve_scalar_elliptic_2d(taps_problem: TAPSProblem) -> tuple[TAPSResultArtifacts, TAPSResidualReport]:
    """Execute a first 2D tensorized TAPS weak-form kernel.

    The kernel solves the rectangular-domain model operator

        -k (d2u/dx2 + d2u/dy2) + c u = f

    with homogeneous Dirichlet boundaries using a truncated separable sine
    expansion. `TAPSBasisConfig.tensor_rank` controls the retained mode count.
    It is a deliberately small baseline for testing TAPS tensor axes,
    separable field factors, and residual verification before a full
    multidimensional Galerkin assembler is connected.
    """
    family = _family(taps_problem)
    output_dir = project_root() / "scratch" / _safe(taps_problem.problem_id) / "taps_generic"
    output_dir.mkdir(parents=True, exist_ok=True)

    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"Generic TAPS scalar assembler does not support family={family!r}.")
    axes = _space_axes(taps_problem)
    if len(axes) != 2:
        raise ValueError(f"2D generic TAPS scalar assembler requires exactly two space axes, got {axes}.")

    x = _axis_values(taps_problem, axes[0], default_points=32)
    y = _axis_values(taps_problem, axes[1], default_points=32)
    active_mask = _load_occupancy_mask(taps_problem, len(x), len(y))
    if len(x) < 3 or len(y) < 3:
        raise ValueError("2D generic TAPS scalar assembler requires at least three points per axis.")
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("2D generic TAPS scalar assembler requires increasing space axes.")

    diffusion = _material_number(taps_problem, {"k", "d", "diffusivity", "thermal_diffusivity"}, 1.0)
    reaction = _reaction_value(family)
    max_x_mode = max(1, min(len(x) - 2, 8))
    max_y_mode = max(1, min(len(y) - 2, 8))
    modes = _mode_pairs(max(1, taps_problem.basis.tensor_rank), max_x_mode, max_y_mode)
    if not modes:
        raise ValueError("2D generic TAPS scalar assembler could not build separable modes.")

    x_modes = {mx: [math.sin(mx * math.pi * value) for value in x] for mx, _, _ in modes}
    y_modes = {my: [math.sin(my * math.pi * value) for value in y] for _, my, _ in modes}
    solution_coefficients: list[dict[str, float | int]] = []
    for mx, my, source_coefficient in modes:
        lambda_x = 2.0 * (1.0 - math.cos(mx * math.pi * dx)) / (dx * dx)
        lambda_y = 2.0 * (1.0 - math.cos(my * math.pi * dy)) / (dy * dy)
        denominator = diffusion * (lambda_x + lambda_y) + reaction
        if abs(denominator) < 1e-14:
            raise ZeroDivisionError("2D generic TAPS scalar assembler encountered a resonant manufactured mode.")
        solution_coefficients.append(
            {
                "x_mode": mx,
                "y_mode": my,
                "source_coefficient": source_coefficient,
                "solution_coefficient": source_coefficient / denominator,
                "discrete_eigenvalue": denominator,
            }
        )

    rhs = [[0.0 for _ in y] for _ in x]
    solution = [[0.0 for _ in y] for _ in x]
    for mode in solution_coefficients:
        mx = int(mode["x_mode"])
        my = int(mode["y_mode"])
        source_coefficient = float(mode["source_coefficient"])
        solution_coefficient = float(mode["solution_coefficient"])
        for i in range(len(x)):
            x_value = x_modes[mx][i]
            for j in range(len(y)):
                basis_value = x_value * y_modes[my][j]
                rhs[i][j] += source_coefficient * basis_value
                solution[i][j] += solution_coefficient * basis_value
    if active_mask is not None:
        for i in range(len(x)):
            for j in range(len(y)):
                if not active_mask[i][j] or i == 0 or j == 0 or i == len(x) - 1 or j == len(y) - 1:
                    rhs[i][j] = 0.0
                    solution[i][j] = 0.0
        if any(0 in row for row in active_mask):
            solution, mask_history = _masked_relaxation_2d(
                x,
                y,
                rhs,
                diffusion,
                reaction,
                active_mask,
                max_iterations=10000,
                tolerance=1e-10,
            )
        else:
            mask_history = []
    else:
        mask_history = []
    normalized_residual = _residual_norm_2d(x, y, solution, rhs, diffusion, reaction, active_mask=active_mask)
    convergence_tolerance = 1e-8 if mask_history else 1e-10

    operator_payload = {
        "type": "tensorized_scalar_elliptic_2d",
        "family": family,
        "operator": "-k (d2u/dx2 + d2u/dy2) + c u = f",
        "axes": {
            axes[0]: x,
            axes[1]: y,
        },
        "coefficients": {"diffusion": diffusion, "reaction": reaction},
        "geometry_encoding": {
            "occupancy_mask": active_mask is not None,
            "active_cells": sum(sum(row) for row in active_mask) if active_mask is not None else len(x) * len(y),
            "total_cells": len(x) * len(y),
            "masked_relaxation_iterations": int(mask_history[-1]["iteration"]) if mask_history else 0,
        },
        "basis": {
            "rank": len(solution_coefficients),
            "modes": solution_coefficients,
            "x_modes": {str(mode): values for mode, values in x_modes.items()},
            "y_modes": {str(mode): values for mode, values in y_modes.items()},
        },
    }
    solution_payload = {
        "field": taps_problem.weak_form.trial_fields[0] if taps_problem.weak_form and taps_problem.weak_form.trial_fields else "u",
        "axes": axes,
        "x": x,
        "y": y,
        "values": solution,
    }
    residual_payload = {
        "family": family,
        "normalized_linear_residual": normalized_residual,
        "separable_rank": len(solution_coefficients),
        "active_cell_fraction": (sum(sum(row) for row in active_mask) / (len(x) * len(y))) if active_mask is not None else 1.0,
        "masked_relaxation_iterations": float(mask_history[-1]["iteration"]) if mask_history else 0.0,
        "converged": normalized_residual < convergence_tolerance,
    }

    operator_path = output_dir / "assembled_operator_2d.json"
    solution_path = output_dir / "solution_field_2d.json"
    residual_path = output_dir / "residual_history_2d.json"
    operator_path.write_text(json.dumps(operator_payload, indent=2), encoding="utf-8")
    solution_path.write_text(json.dumps(solution_payload, indent=2), encoding="utf-8")
    residual_path.write_text(json.dumps(residual_payload, indent=2), encoding="utf-8")

    artifacts = TAPSResultArtifacts(
        factor_matrices=[ArtifactRef(uri=str(operator_path), kind="taps_assembled_operator", format="json")],
        reconstruction_metadata=ArtifactRef(uri=str(solution_path), kind="taps_solution_field", format="json"),
        residual_history=ArtifactRef(uri=str(residual_path), kind="taps_residual_history", format="json"),
    )
    report = TAPSResidualReport(
        residuals={
            "normalized_linear_residual": normalized_residual,
            "active_cell_fraction": (sum(sum(row) for row in active_mask) / (len(x) * len(y))) if active_mask is not None else 1.0,
            "masked_relaxation_iterations": float(mask_history[-1]["iteration"]) if mask_history else 0.0,
        },
        rank=len(solution_coefficients),
        converged=normalized_residual < convergence_tolerance,
        recommended_action="accept" if normalized_residual < convergence_tolerance else "refine_axes",
    )
    return artifacts, report


def solve_reaction_diffusion_nonlinear_2d(taps_problem: TAPSProblem) -> tuple[TAPSResultArtifacts, TAPSResidualReport]:
    """Solve a scalar 2D nonlinear reaction-diffusion model with damped fixed-point sweeps."""
    output_dir = project_root() / "scratch" / _safe(taps_problem.problem_id) / "taps_generic"
    output_dir.mkdir(parents=True, exist_ok=True)
    axes = _space_axes(taps_problem)
    if len(axes) != 2:
        raise ValueError(f"2D nonlinear reaction-diffusion requires exactly two space axes, got {axes}.")

    x = _axis_values(taps_problem, axes[0], default_points=32)
    y = _axis_values(taps_problem, axes[1], default_points=32)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    diffusion = _material_number(taps_problem, {"d", "diffusivity", "thermal_diffusivity"}, 1.0)
    beta = taps_problem.nonlinear.linear_reaction
    gamma = taps_problem.nonlinear.cubic_reaction
    cx = diffusion / (dx * dx)
    cy = diffusion / (dy * dy)
    max_x_mode = max(1, min(len(x) - 2, 8))
    max_y_mode = max(1, min(len(y) - 2, 8))
    modes = _mode_pairs(max(1, taps_problem.basis.tensor_rank), max_x_mode, max_y_mode)
    x_modes = {mx: [math.sin(mx * math.pi * value) for value in x] for mx, _, _ in modes}
    y_modes = {my: [math.sin(my * math.pi * value) for value in y] for _, my, _ in modes}
    rhs = [[0.0 for _ in y] for _ in x]
    for mx, my, coefficient in modes:
        for i in range(len(x)):
            x_value = x_modes[mx][i]
            for j in range(len(y)):
                rhs[i][j] += 0.2 * coefficient * x_value * y_modes[my][j]

    solution = [[0.0 for _ in y] for _ in x]
    for mx, my, coefficient in modes:
        lambda_x = 2.0 * (1.0 - math.cos(mx * math.pi * dx)) / (dx * dx)
        lambda_y = 2.0 * (1.0 - math.cos(my * math.pi * dy)) / (dy * dy)
        denominator = diffusion * (lambda_x + lambda_y) + beta
        if abs(denominator) < 1e-14:
            continue
        solution_coefficient = 0.2 * coefficient / denominator
        for i in range(len(x)):
            x_value = x_modes[mx][i]
            for j in range(len(y)):
                solution[i][j] += solution_coefficient * x_value * y_modes[my][j]
    history: list[dict[str, float | int]] = []
    for iteration in range(1, taps_problem.nonlinear.max_iterations + 1):
        update_sq = 0.0
        norm_sq = 0.0
        previous = [row[:] for row in solution]
        for i in range(1, len(x) - 1):
            for j in range(1, len(y) - 1):
                denom = 2.0 * cx + 2.0 * cy + beta + gamma * previous[i][j] * previous[i][j]
                candidate = (
                    rhs[i][j]
                    + cx * (solution[i - 1][j] + previous[i + 1][j])
                    + cy * (solution[i][j - 1] + previous[i][j + 1])
                ) / denom
                damped = taps_problem.nonlinear.damping * candidate + (1.0 - taps_problem.nonlinear.damping) * previous[i][j]
                update_sq += (damped - previous[i][j]) ** 2
                norm_sq += damped * damped
                solution[i][j] = damped
        residual = _nonlinear_residual_norm_2d(x, y, solution, rhs, diffusion, beta, gamma)
        update_norm = math.sqrt(update_sq) / (math.sqrt(norm_sq) + 1e-12)
        history.append({"iteration": iteration, "normalized_nonlinear_residual": residual, "relative_update": update_norm})
        if residual < taps_problem.nonlinear.tolerance or update_norm < taps_problem.nonlinear.tolerance:
            break

    final_residual = _nonlinear_residual_norm_2d(x, y, solution, rhs, diffusion, beta, gamma)
    final_update = float(history[-1]["relative_update"]) if history else float("inf")
    converged = final_residual < 1e-6 or final_update < taps_problem.nonlinear.tolerance
    operator_payload = {
        "type": "nonlinear_reaction_diffusion_2d",
        "operator": "-D (d2u/dx2 + d2u/dy2) + beta u + gamma u^3 = f",
        "axes": {axes[0]: x, axes[1]: y},
        "coefficients": {"diffusion": diffusion, "linear_reaction": beta, "cubic_reaction": gamma},
        "source_modes": [{"x_mode": mx, "y_mode": my, "coefficient": coefficient} for mx, my, coefficient in modes],
        "method": taps_problem.nonlinear.method,
    }
    solution_payload = {
        "field": taps_problem.weak_form.trial_fields[0] if taps_problem.weak_form and taps_problem.weak_form.trial_fields else "u",
        "axes": axes,
        "x": x,
        "y": y,
        "values": solution,
    }
    residual_payload = {
        "family": "reaction_diffusion",
        "normalized_nonlinear_residual": final_residual,
        "relative_update": final_update,
        "iterations": len(history),
        "iteration_history": history,
        "separable_rank": len(modes),
        "converged": converged,
    }
    operator_path = output_dir / "nonlinear_operator_2d.json"
    solution_path = output_dir / "nonlinear_solution_field_2d.json"
    residual_path = output_dir / "nonlinear_iteration_history_2d.json"
    operator_path.write_text(json.dumps(operator_payload, indent=2), encoding="utf-8")
    solution_path.write_text(json.dumps(solution_payload, indent=2), encoding="utf-8")
    residual_path.write_text(json.dumps(residual_payload, indent=2), encoding="utf-8")

    artifacts = TAPSResultArtifacts(
        factor_matrices=[ArtifactRef(uri=str(operator_path), kind="taps_nonlinear_operator", format="json")],
        reconstruction_metadata=ArtifactRef(uri=str(solution_path), kind="taps_solution_field", format="json"),
        residual_history=ArtifactRef(uri=str(residual_path), kind="taps_iteration_history", format="json"),
    )
    return (
        artifacts,
        TAPSResidualReport(
            residuals={
                "normalized_nonlinear_residual": final_residual,
                "relative_update": final_update,
                "nonlinear_iterations": float(len(history)),
            },
            rank=len(modes),
            converged=converged,
            recommended_action="accept" if converged else "refine_axes",
        ),
    )


def solve_coupled_reaction_diffusion_2d(taps_problem: TAPSProblem) -> tuple[TAPSResultArtifacts, TAPSResidualReport]:
    """Solve a two-field 2D coupled nonlinear reaction-diffusion model.

    Model:
        -D Δu + beta u + gamma u^3 + kappa (u - v) = f_u
        -D Δv + beta v + gamma v^3 + kappa (v - u) = f_v
    """
    output_dir = project_root() / "scratch" / _safe(taps_problem.problem_id) / "taps_generic"
    output_dir.mkdir(parents=True, exist_ok=True)
    axes = _space_axes(taps_problem)
    if len(axes) != 2:
        raise ValueError(f"2D coupled reaction-diffusion requires exactly two space axes, got {axes}.")

    x = _axis_values(taps_problem, axes[0], default_points=32)
    y = _axis_values(taps_problem, axes[1], default_points=32)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    diffusion = _material_number(taps_problem, {"d", "diffusivity", "thermal_diffusivity"}, 1.0)
    beta = taps_problem.nonlinear.linear_reaction
    gamma = taps_problem.nonlinear.cubic_reaction
    kappa = taps_problem.nonlinear.coupling_strength
    cx = diffusion / (dx * dx)
    cy = diffusion / (dy * dy)

    max_x_mode = max(1, min(len(x) - 2, 8))
    max_y_mode = max(1, min(len(y) - 2, 8))
    modes = _mode_pairs(max(1, taps_problem.basis.tensor_rank), max_x_mode, max_y_mode)
    x_modes = {mx: [math.sin(mx * math.pi * value) for value in x] for mx, _, _ in modes}
    y_modes = {my: [math.sin(my * math.pi * value) for value in y] for _, my, _ in modes}
    rhs_u = [[0.0 for _ in y] for _ in x]
    rhs_v = [[0.0 for _ in y] for _ in x]
    u = [[0.0 for _ in y] for _ in x]
    v = [[0.0 for _ in y] for _ in x]

    # Build two related but non-identical forcing fields and a linear warm start.
    for index, (mx, my, coefficient) in enumerate(modes):
        sign = -1.0 if index % 2 else 1.0
        lambda_x = 2.0 * (1.0 - math.cos(mx * math.pi * dx)) / (dx * dx)
        lambda_y = 2.0 * (1.0 - math.cos(my * math.pi * dy)) / (dy * dy)
        block_diag = diffusion * (lambda_x + lambda_y) + beta + kappa
        determinant = block_diag * block_diag - kappa * kappa
        if abs(determinant) < 1e-14:
            continue
        forcing_u = 0.18 * coefficient
        forcing_v = 0.12 * sign * coefficient
        solution_u = (block_diag * forcing_u + kappa * forcing_v) / determinant
        solution_v = (kappa * forcing_u + block_diag * forcing_v) / determinant
        for i in range(len(x)):
            x_value = x_modes[mx][i]
            for j in range(len(y)):
                basis = x_value * y_modes[my][j]
                rhs_u[i][j] += forcing_u * basis
                rhs_v[i][j] += forcing_v * basis
                u[i][j] += solution_u * basis
                v[i][j] += solution_v * basis

    history: list[dict[str, float | int]] = []
    for iteration in range(1, taps_problem.nonlinear.max_iterations + 1):
        update_sq = 0.0
        norm_sq = 0.0
        prev_u = [row[:] for row in u]
        prev_v = [row[:] for row in v]
        for i in range(1, len(x) - 1):
            for j in range(1, len(y) - 1):
                denom_u = 2.0 * cx + 2.0 * cy + beta + kappa + gamma * prev_u[i][j] * prev_u[i][j]
                candidate_u = (
                    rhs_u[i][j]
                    + cx * (u[i - 1][j] + prev_u[i + 1][j])
                    + cy * (u[i][j - 1] + prev_u[i][j + 1])
                    + kappa * prev_v[i][j]
                ) / denom_u
                new_u = taps_problem.nonlinear.damping * candidate_u + (1.0 - taps_problem.nonlinear.damping) * prev_u[i][j]
                u[i][j] = new_u

                denom_v = 2.0 * cx + 2.0 * cy + beta + kappa + gamma * prev_v[i][j] * prev_v[i][j]
                candidate_v = (
                    rhs_v[i][j]
                    + cx * (v[i - 1][j] + prev_v[i + 1][j])
                    + cy * (v[i][j - 1] + prev_v[i][j + 1])
                    + kappa * new_u
                ) / denom_v
                new_v = taps_problem.nonlinear.damping * candidate_v + (1.0 - taps_problem.nonlinear.damping) * prev_v[i][j]
                v[i][j] = new_v

                update_sq += (new_u - prev_u[i][j]) ** 2 + (new_v - prev_v[i][j]) ** 2
                norm_sq += new_u * new_u + new_v * new_v
        residual = _coupled_residual_norm_2d(x, y, u, v, rhs_u, rhs_v, diffusion, beta, gamma, kappa)
        update_norm = math.sqrt(update_sq) / (math.sqrt(norm_sq) + 1e-12)
        history.append({"iteration": iteration, "normalized_coupled_residual": residual, "relative_update": update_norm})
        if residual < taps_problem.nonlinear.tolerance or update_norm < taps_problem.nonlinear.tolerance:
            break

    final_residual = _coupled_residual_norm_2d(x, y, u, v, rhs_u, rhs_v, diffusion, beta, gamma, kappa)
    final_update = float(history[-1]["relative_update"]) if history else float("inf")
    converged = final_residual < 1e-6 or final_update < taps_problem.nonlinear.tolerance

    field_names = taps_problem.weak_form.trial_fields if taps_problem.weak_form and taps_problem.weak_form.trial_fields else ["u", "v"]
    operator_payload = {
        "type": "coupled_reaction_diffusion_2d",
        "operator": "-D Δu + beta u + gamma u^3 + kappa(u-v) = f_u; -D Δv + beta v + gamma v^3 + kappa(v-u) = f_v",
        "axes": {axes[0]: x, axes[1]: y},
        "fields": field_names[:2],
        "coefficients": {
            "diffusion": diffusion,
            "linear_reaction": beta,
            "cubic_reaction": gamma,
            "coupling_strength": kappa,
        },
        "source_modes": [{"x_mode": mx, "y_mode": my, "coefficient": coefficient} for mx, my, coefficient in modes],
        "method": taps_problem.nonlinear.method,
    }
    solution_payload = {
        "fields": {
            field_names[0]: u,
            field_names[1] if len(field_names) > 1 else "v": v,
        },
        "axes": axes,
        "x": x,
        "y": y,
    }
    residual_payload = {
        "family": "coupled_reaction_diffusion",
        "normalized_coupled_residual": final_residual,
        "relative_update": final_update,
        "iterations": len(history),
        "iteration_history": history,
        "separable_rank": len(modes),
        "converged": converged,
    }
    operator_path = output_dir / "coupled_operator_2d.json"
    solution_path = output_dir / "coupled_solution_fields_2d.json"
    residual_path = output_dir / "coupled_iteration_history_2d.json"
    operator_path.write_text(json.dumps(operator_payload, indent=2), encoding="utf-8")
    solution_path.write_text(json.dumps(solution_payload, indent=2), encoding="utf-8")
    residual_path.write_text(json.dumps(residual_payload, indent=2), encoding="utf-8")

    artifacts = TAPSResultArtifacts(
        factor_matrices=[ArtifactRef(uri=str(operator_path), kind="taps_coupled_operator", format="json")],
        reconstruction_metadata=ArtifactRef(uri=str(solution_path), kind="taps_coupled_solution_fields", format="json"),
        residual_history=ArtifactRef(uri=str(residual_path), kind="taps_iteration_history", format="json"),
    )
    return (
        artifacts,
        TAPSResidualReport(
            residuals={
                "normalized_coupled_residual": final_residual,
                "relative_update": final_update,
                "nonlinear_iterations": float(len(history)),
            },
            rank=len(modes),
            converged=converged,
            recommended_action="accept" if converged else "refine_axes",
        ),
    )
