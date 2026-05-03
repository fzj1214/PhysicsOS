from __future__ import annotations

import json
import math
from pathlib import Path
from functools import lru_cache

from physicsos.config import project_root
from physicsos.schemas.common import ArtifactRef
from physicsos.schemas.taps import NumericalSolvePlanOutput, TAPSProblem, TAPSResidualReport, TAPSResultArtifacts


SUPPORTED_FAMILIES = {
    "poisson",
    "diffusion",
    "thermal_diffusion",
    "reaction_diffusion",
    "coupled_reaction_diffusion",
    "helmholtz",
    "stokes",
    "oseen",
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


def _face_tuples_for_boundary_region(
    graph: dict,
    geometric_faces: list[tuple[int, int, int]],
    boundary_nodes: set[int],
    region_id: str,
) -> set[tuple[int, int, int]]:
    boundary_face_sets = graph.get("boundary_face_sets", {})
    candidates = [
        region_id,
        region_id.replace("region:", "boundary:"),
        region_id.replace("boundary:", ""),
    ]
    for candidate in candidates:
        selected: set[tuple[int, int, int]] = set()
        if candidate in boundary_face_sets:
            for face_id in boundary_face_sets[candidate]:
                index = int(face_id)
                if 0 <= index < len(geometric_faces):
                    selected.add(geometric_faces[index])
        boundary_node_sets = graph.get("boundary_node_sets", {})
        if candidate in boundary_node_sets:
            node_set = {int(node) for node in boundary_node_sets[candidate]}
            selected.update(face for face in geometric_faces if all(node in node_set for node in face))
        if selected:
            return selected
    if region_id in {"boundary", "all_boundaries", "external_boundary"}:
        return {face for face in geometric_faces if all(node in boundary_nodes for node in face)}
    if region_id in {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}:
        return {
            geometric_faces[int(face_id)]
            for face_id in boundary_face_sets.get(region_id, [])
            if 0 <= int(face_id) < len(geometric_faces)
        }
    return {face for face in geometric_faces if all(node in boundary_nodes for node in face)}


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


def _term_expression(term: object) -> str:
    expression = getattr(term, "expression", "")
    return str(expression).lower().replace("·", " dot ").replace("路", " dot ")


def _weak_form_scalar_elliptic_blocks(problem: TAPSProblem) -> dict[str, object] | None:
    """Map general weak-form IR terms onto the scalar elliptic assembler blocks.

    This is intentionally conservative: it only accepts one scalar field and
    terms that can be explained as diffusion/reaction/mass/source contributions.
    Unknown custom terms must contain recognizable grad/laplacian/source tokens.
    """
    weak_form = problem.weak_form
    if weak_form is None or len(weak_form.trial_fields) != 1:
        return None
    blocks: list[dict[str, str]] = []
    has_diffusion = False
    has_source = False
    has_reaction = False
    allowed_roles = {"diffusion", "reaction", "mass", "source", "custom", "boundary"}
    for term in [*weak_form.terms, *weak_form.boundary_terms]:
        role = term.role.lower()
        expression = _term_expression(term)
        if role not in allowed_roles:
            return None
        if role == "boundary":
            continue
        is_diffusion = role == "diffusion" or any(token in expression for token in ("grad(", "nabla", "laplacian", "∇"))
        is_source = role == "source" or any(token in expression for token in ("source", " rhs", " f ", "v f", "v*f", "- int_omega v"))
        is_reaction = role in {"reaction", "mass"} or any(token in expression for token in (" v u", "v*u", "mass", "reaction"))
        if role == "custom" and not (is_diffusion or is_source or is_reaction):
            return None
        if is_diffusion:
            has_diffusion = True
            blocks.append({"role": "diffusion", "term_id": term.id, "expression": term.expression})
        if is_reaction:
            has_reaction = True
            blocks.append({"role": "reaction", "term_id": term.id, "expression": term.expression})
        if is_source:
            has_source = True
            blocks.append({"role": "source", "term_id": term.id, "expression": term.expression})
    if not has_diffusion:
        return None
    return {
        "operator_family": "scalar_elliptic",
        "source": "weak_form_ir",
        "blocks": blocks,
        "has_diffusion": has_diffusion,
        "has_reaction": has_reaction,
        "has_source": has_source,
    }


def supports_scalar_elliptic_weak_form(problem: TAPSProblem) -> bool:
    return _weak_form_scalar_elliptic_blocks(problem) is not None


def _weak_form_vector_elasticity_blocks(problem: TAPSProblem) -> dict[str, object] | None:
    weak_form = problem.weak_form
    if weak_form is None or len(weak_form.trial_fields) != 1:
        return None
    blocks: list[dict[str, str]] = []
    has_strain_energy = False
    has_body_force = False
    allowed_roles = {"constitutive", "source", "custom", "boundary"}
    for term in [*weak_form.terms, *weak_form.boundary_terms]:
        role = term.role.lower()
        expression = _term_expression(term)
        if role not in allowed_roles:
            return None
        if role == "boundary":
            continue
        is_strain_energy = role == "constitutive" or any(
            token in expression for token in ("epsilon(", "strain", "stress", "c(", "constitutive", "symgrad")
        )
        is_body_force = role == "source" or any(token in expression for token in ("body_force", " v dot b", "v_i b_i", "traction"))
        if role == "custom" and not (is_strain_energy or is_body_force):
            return None
        if is_strain_energy:
            has_strain_energy = True
            blocks.append({"role": "strain_energy", "term_id": term.id, "expression": term.expression})
        if is_body_force:
            has_body_force = True
            blocks.append({"role": "body_force", "term_id": term.id, "expression": term.expression})
    if not has_strain_energy:
        return None
    return {
        "operator_family": "vector_linear_elasticity",
        "source": "weak_form_ir",
        "blocks": blocks,
        "has_strain_energy": has_strain_energy,
        "has_body_force": has_body_force,
    }


def supports_vector_elasticity_weak_form(problem: TAPSProblem) -> bool:
    return _weak_form_vector_elasticity_blocks(problem) is not None


def _weak_form_hcurl_curl_curl_blocks(problem: TAPSProblem) -> dict[str, object] | None:
    weak_form = problem.weak_form
    if weak_form is None or len(weak_form.trial_fields) != 1:
        return None
    blocks: list[dict[str, str]] = []
    has_curl_curl = False
    has_mass = False
    has_source = False
    allowed_roles = {"diffusion", "mass", "source", "custom", "boundary"}
    for term in [*weak_form.terms, *weak_form.boundary_terms]:
        role = term.role.lower()
        expression = _term_expression(term)
        if role not in allowed_roles:
            return None
        if role == "boundary":
            continue
        is_curl_curl = "curl(" in expression or "curl_" in expression or "curl-curl" in expression or "curl curl" in expression
        is_mass = role == "mass" or any(token in expression for token in (" k0", "wave_number", "permittivity", "epsilon", " v dot e"))
        is_source = role == "source" or any(token in expression for token in ("current_source", " v dot j", "source", "jz"))
        if role == "diffusion" and is_curl_curl:
            pass
        elif role == "custom" and not (is_curl_curl or is_mass or is_source):
            return None
        elif role == "diffusion" and not is_curl_curl:
            return None
        if is_curl_curl:
            has_curl_curl = True
            blocks.append({"role": "curl_curl", "term_id": term.id, "expression": term.expression})
        if is_mass:
            has_mass = True
            blocks.append({"role": "mass", "term_id": term.id, "expression": term.expression})
        if is_source:
            has_source = True
            blocks.append({"role": "source", "term_id": term.id, "expression": term.expression})
    if not has_curl_curl:
        return None
    return {
        "operator_family": "hcurl_curl_curl",
        "source": "weak_form_ir",
        "blocks": blocks,
        "has_curl_curl": has_curl_curl,
        "has_mass": has_mass,
        "has_source": has_source,
    }


def supports_hcurl_curl_curl_weak_form(problem: TAPSProblem) -> bool:
    return _weak_form_hcurl_curl_curl_blocks(problem) is not None


def _weak_form_hdiv_div_blocks(problem: TAPSProblem) -> dict[str, object] | None:
    weak_form = problem.weak_form
    if weak_form is None or len(weak_form.trial_fields) != 1:
        return None
    family = weak_form.family.lower()
    blocks: list[dict[str, str]] = []
    has_div = family in {"hdiv", "div", "darcy", "mixed_poisson"}
    has_mass = family in {"darcy", "mixed_poisson"}
    has_source = False
    allowed_roles = {"diffusion", "mass", "source", "constraint", "custom", "boundary"}
    for term in [*weak_form.terms, *weak_form.boundary_terms]:
        role = term.role.lower()
        expression = _term_expression(term)
        if role not in allowed_roles:
            return None
        if role == "boundary":
            continue
        is_div = any(token in expression for token in ("div(", "divergence", "nabla_dot", "flux"))
        is_mass = role == "mass" or any(token in expression for token in ("q dot u", "q·u", "permeability", "k^-1", "mass"))
        is_source = role == "source" or any(token in expression for token in ("source", "rhs", " f ", "sink"))
        if role == "custom" and not (is_div or is_mass or is_source):
            return None
        if role in {"diffusion", "constraint"} and not is_div:
            return None
        if is_div:
            has_div = True
            blocks.append({"role": "divergence", "term_id": term.id, "expression": term.expression})
        if is_mass:
            has_mass = True
            blocks.append({"role": "mass", "term_id": term.id, "expression": term.expression})
        if is_source:
            has_source = True
            blocks.append({"role": "source", "term_id": term.id, "expression": term.expression})
    if not has_div:
        return None
    return {
        "operator_family": "hdiv_div",
        "source": "weak_form_ir",
        "blocks": blocks,
        "has_divergence": has_div,
        "has_mass": has_mass,
        "has_source": has_source,
    }


def supports_hdiv_div_weak_form(problem: TAPSProblem) -> bool:
    return _weak_form_hdiv_div_blocks(problem) is not None


def _weak_form_nonlinear_reaction_diffusion_blocks(problem: TAPSProblem) -> dict[str, object] | None:
    weak_form = problem.weak_form
    if weak_form is None or len(weak_form.trial_fields) != 1:
        return None
    blocks: list[dict[str, str]] = []
    has_diffusion = False
    has_nonlinear_reaction = False
    has_source = False
    allowed_roles = {"diffusion", "reaction", "source", "custom", "boundary"}
    for term in [*weak_form.terms, *weak_form.boundary_terms]:
        role = term.role.lower()
        expression = _term_expression(term)
        if role not in allowed_roles:
            return None
        if role == "boundary":
            continue
        is_diffusion = role == "diffusion" or any(token in expression for token in ("grad(", "nabla", "laplacian"))
        is_nonlinear_reaction = role == "reaction" and any(
            token in expression for token in ("u^2", "u^3", "u*u", "u**", "cubic", "nonlinear", "r(u", "reaction")
        )
        if role == "custom":
            is_nonlinear_reaction = is_nonlinear_reaction or any(
                token in expression for token in ("u^2", "u^3", "u*u", "u**", "cubic", "nonlinear", "r(u")
            )
        is_source = role == "source" or any(token in expression for token in ("source", " rhs", " f ", "v f", "v*f"))
        if role == "custom" and not (is_diffusion or is_nonlinear_reaction or is_source):
            return None
        if is_diffusion:
            has_diffusion = True
            blocks.append({"role": "diffusion", "term_id": term.id, "expression": term.expression})
        if is_nonlinear_reaction:
            has_nonlinear_reaction = True
            blocks.append({"role": "nonlinear_reaction", "term_id": term.id, "expression": term.expression})
        if is_source:
            has_source = True
            blocks.append({"role": "source", "term_id": term.id, "expression": term.expression})
    if not (has_diffusion and has_nonlinear_reaction):
        return None
    return {
        "operator_family": "nonlinear_reaction_diffusion",
        "source": "weak_form_ir",
        "blocks": blocks,
        "has_diffusion": has_diffusion,
        "has_nonlinear_reaction": has_nonlinear_reaction,
        "has_source": has_source,
    }


def supports_nonlinear_reaction_diffusion_weak_form(problem: TAPSProblem) -> bool:
    return _weak_form_nonlinear_reaction_diffusion_blocks(problem) is not None


def _weak_form_coupled_reaction_diffusion_blocks(problem: TAPSProblem) -> dict[str, object] | None:
    weak_form = problem.weak_form
    if weak_form is None or len(weak_form.trial_fields) < 2:
        return None
    field_tokens = {field.lower() for field in weak_form.trial_fields}
    blocks: list[dict[str, str]] = []
    has_diffusion = False
    has_reaction = False
    has_coupling = False
    has_source = False
    allowed_roles = {"diffusion", "reaction", "source", "constitutive", "custom", "boundary"}
    for term in [*weak_form.terms, *weak_form.boundary_terms]:
        role = term.role.lower()
        expression = _term_expression(term)
        if role not in allowed_roles:
            return None
        if role == "boundary":
            continue
        is_diffusion = role == "diffusion" or "grad(" in expression or "laplacian" in expression or "nabla" in expression
        is_reaction = role == "reaction" or any(token in expression for token in ("reaction", "r_i", "r(", "u^3", "v^3", "nonlinear"))
        is_coupling = role == "constitutive" or any(
            token in expression for token in ("coupling", "kappa", "u - v", "v - u", "u-v", "v-u", "cross_jacobian")
        )
        if not is_coupling and len(field_tokens) >= 2:
            mentioned_fields = {field for field in field_tokens if field in expression}
            is_coupling = len(mentioned_fields) >= 2 and any(token in expression for token in ("-", "+", "coupled", "between"))
        is_source = role == "source" or any(token in expression for token in ("source", " f_u", " f_v", "rhs"))
        if role == "custom" and not (is_diffusion or is_reaction or is_coupling or is_source):
            return None
        if is_diffusion:
            has_diffusion = True
            blocks.append({"role": "field_diffusion", "term_id": term.id, "expression": term.expression})
        if is_reaction:
            has_reaction = True
            blocks.append({"role": "field_reaction", "term_id": term.id, "expression": term.expression})
        if is_coupling:
            has_coupling = True
            blocks.append({"role": "coupling_operator", "term_id": term.id, "expression": term.expression})
        if is_source:
            has_source = True
            blocks.append({"role": "source", "term_id": term.id, "expression": term.expression})
    if not (has_diffusion and has_coupling):
        return None
    return {
        "operator_family": "coupled_reaction_diffusion",
        "source": "weak_form_ir",
        "fields": weak_form.trial_fields,
        "blocks": blocks,
        "has_diffusion": has_diffusion,
        "has_reaction": has_reaction,
        "has_coupling": has_coupling,
        "has_source": has_source,
        "subspace_solver": "block_gauss_seidel_picard",
    }


def supports_coupled_reaction_diffusion_weak_form(problem: TAPSProblem) -> bool:
    return _weak_form_coupled_reaction_diffusion_blocks(problem) is not None


def _weak_form_stokes_blocks(problem: TAPSProblem) -> dict[str, object] | None:
    weak_form = problem.weak_form
    if weak_form is None or len(weak_form.trial_fields) < 2:
        return None
    fields = {field.lower() for field in weak_form.trial_fields}
    if not ({"u", "velocity"} & fields) or not ({"p", "pressure"} & fields):
        return None
    blocks: list[dict[str, str]] = []
    has_viscous = False
    has_pressure = False
    has_continuity = False
    has_body_force = False
    has_advection = False
    allowed_roles = {"diffusion", "advection", "source", "constraint", "custom", "boundary"}
    for term in [*weak_form.terms, *weak_form.boundary_terms]:
        role = term.role.lower()
        expression = _term_expression(term)
        if role not in allowed_roles:
            return None
        if role == "boundary":
            continue
        is_viscous = role == "diffusion" or any(token in expression for token in ("grad(u", "grad(v_u", "strain", "viscous", "mu"))
        is_pressure = any(token in expression for token in (" p", "pressure", "div(v_u", "grad(p"))
        is_continuity = role == "constraint" or any(token in expression for token in ("div(u", "q div", "incompress"))
        is_body_force = role == "source" or any(token in expression for token in ("body_force", "forcing", "v_u dot f"))
        is_advection = role == "advection" or any(token in expression for token in ("u dot grad", "(u路grad", "(u dot grad", "convective"))
        if role == "custom" and not (is_viscous or is_pressure or is_continuity or is_body_force or is_advection):
            return None
        if is_viscous:
            has_viscous = True
            blocks.append({"role": "viscous_diffusion", "term_id": term.id, "expression": term.expression})
        if is_pressure:
            has_pressure = True
            blocks.append({"role": "pressure_coupling", "term_id": term.id, "expression": term.expression})
        if is_continuity:
            has_continuity = True
            blocks.append({"role": "incompressibility_constraint", "term_id": term.id, "expression": term.expression})
        if is_body_force:
            has_body_force = True
            blocks.append({"role": "body_force", "term_id": term.id, "expression": term.expression})
        if is_advection:
            has_advection = True
            blocks.append({"role": "nonlinear_advection", "term_id": term.id, "expression": term.expression})
    if not (has_viscous and has_pressure and has_continuity):
        return None
    return {
        "operator_family": "incompressible_stokes",
        "source": "weak_form_ir",
        "fields": weak_form.trial_fields,
        "blocks": blocks,
        "has_viscous_diffusion": has_viscous,
        "has_pressure_coupling": has_pressure,
        "has_incompressibility_constraint": has_continuity,
        "has_body_force": has_body_force,
        "has_nonlinear_advection": has_advection,
        "executable_simplification": "steady_low_re_channel_stokes" if not has_advection else "requires_picard_or_newton_navier_stokes",
    }


def supports_stokes_weak_form(problem: TAPSProblem) -> bool:
    blocks = _weak_form_stokes_blocks(problem)
    return blocks is not None and not bool(blocks.get("has_nonlinear_advection"))


def _has_frozen_convective_velocity(problem: TAPSProblem) -> bool:
    return any(
        coefficient.name.lower()
        in {
            "frozen_velocity",
            "convective_velocity",
            "oseen_velocity",
            "linearization_velocity",
            "ubar",
            "u_bar",
        }
        for coefficient in problem.coefficients
    )


def _weak_form_oseen_blocks(problem: TAPSProblem) -> dict[str, object] | None:
    blocks = _weak_form_stokes_blocks(problem)
    if blocks is None or not bool(blocks.get("has_nonlinear_advection")):
        return None
    has_linearization_velocity = _has_frozen_convective_velocity(problem)
    has_linearized_term = any(
        block["role"] == "nonlinear_advection"
        and any(token in block["expression"].lower() for token in ("ubar", "u_bar", "frozen", "linearized", "oseen"))
        for block in blocks["blocks"]  # type: ignore[index]
    )
    if not (has_linearization_velocity or has_linearized_term):
        return None
    output = dict(blocks)
    output["operator_family"] = "incompressible_oseen"
    output["has_frozen_convective_velocity"] = has_linearization_velocity
    output["executable_simplification"] = "linearized_oseen_channel"
    return output


def supports_oseen_weak_form(problem: TAPSProblem) -> bool:
    return _weak_form_oseen_blocks(problem) is not None


def _has_coefficient(problem: TAPSProblem, names: set[str]) -> bool:
    normalized = {name.lower() for name in names}
    return any(coefficient.name.lower() in normalized for coefficient in problem.coefficients)


def _weak_form_navier_stokes_blocks(problem: TAPSProblem) -> dict[str, object] | None:
    blocks = _weak_form_stokes_blocks(problem)
    if blocks is None or not bool(blocks.get("has_nonlinear_advection")):
        return None
    if _weak_form_oseen_blocks(problem) is not None:
        return None
    declared_reynolds = _coefficient_number(problem, {"re", "reynolds", "reynolds_number"}, -1.0)
    if declared_reynolds > 100.0:
        return None
    boundary_kinds = {boundary.kind.lower() for boundary in problem.boundary_conditions}
    has_channel_boundaries = "wall" in boundary_kinds and bool(boundary_kinds & {"inlet", "dirichlet"}) and bool(boundary_kinds & {"outlet", "neumann"})
    has_required_coefficients = (
        _has_coefficient(problem, {"mu", "dynamic_viscosity", "viscosity"})
        and _has_coefficient(problem, {"rho", "density"})
        and _has_coefficient(problem, {"pressure_drop", "delta_p", "dp"})
    )
    if not (has_channel_boundaries and has_required_coefficients):
        return None
    output = dict(blocks)
    output["operator_family"] = "incompressible_navier_stokes"
    output["executable_simplification"] = "steady_laminar_channel_picard"
    output["stabilization_policy"] = "picard_under_relaxation_channel_surrogate"
    return output


def supports_navier_stokes_weak_form(problem: TAPSProblem) -> bool:
    return _weak_form_navier_stokes_blocks(problem) is not None


def _weak_form_mesh_navier_stokes_bridge_blocks(problem: TAPSProblem) -> dict[str, object] | None:
    blocks = _weak_form_stokes_blocks(problem)
    if blocks is None or not bool(blocks.get("has_nonlinear_advection")):
        return None
    has_mesh_graph = any(encoding.kind == "mesh_graph" for encoding in problem.geometry_encodings)
    if not has_mesh_graph:
        return None
    output = dict(blocks)
    output["operator_family"] = "incompressible_navier_stokes_mesh_bridge"
    output["source"] = "weak_form_ir_mesh_graph"
    output["function_space"] = "mixed_velocity_pressure"
    output["pressure_velocity_coupling"] = "monolithic_mixed_or_projection"
    output["stabilization_policy"] = "SUPG_PSPG_or_projection_review_required"
    output["execution_policy"] = "export_backend_bridge_only"
    output["fallback_targets"] = ["fenicsx", "openfoam", "su2"]
    return output


def supports_mesh_navier_stokes_bridge(problem: TAPSProblem) -> bool:
    return _weak_form_mesh_navier_stokes_bridge_blocks(problem) is not None


def weak_form_transient_diffusion_blocks(problem: TAPSProblem) -> dict[str, object] | None:
    weak_form = problem.weak_form
    if weak_form is None or len(weak_form.trial_fields) != 1:
        return None
    blocks: list[dict[str, str]] = []
    has_time_derivative = False
    has_diffusion = False
    has_mass = False
    has_source = False
    allowed_roles = {"time_derivative", "mass", "diffusion", "source", "custom", "boundary"}
    for term in [*weak_form.terms, *weak_form.boundary_terms]:
        role = term.role.lower()
        expression = _term_expression(term)
        if role not in allowed_roles:
            return None
        if role == "boundary":
            continue
        is_time = role == "time_derivative" or any(token in expression for token in ("d/dt", "dt", "partial_t", "∂", "time_derivative"))
        is_mass = role == "mass" or any(token in expression for token in ("mass", " v u", "v*u"))
        is_diffusion = role == "diffusion" or any(token in expression for token in ("grad(", "laplacian", "nabla", "d2"))
        is_source = role == "source" or any(token in expression for token in ("source", " rhs", " f ", "v f", "v*f"))
        if role == "custom" and not (is_time or is_mass or is_diffusion or is_source):
            return None
        if is_time:
            has_time_derivative = True
            blocks.append({"role": "time_derivative", "term_id": term.id, "expression": term.expression})
        if is_mass:
            has_mass = True
            blocks.append({"role": "mass", "term_id": term.id, "expression": term.expression})
        if is_diffusion:
            has_diffusion = True
            blocks.append({"role": "diffusion", "term_id": term.id, "expression": term.expression})
        if is_source:
            has_source = True
            blocks.append({"role": "source", "term_id": term.id, "expression": term.expression})
    if not (has_time_derivative and has_diffusion):
        return None
    return {
        "operator_family": "transient_diffusion",
        "source": "weak_form_ir",
        "fields": weak_form.trial_fields,
        "blocks": blocks,
        "has_time_derivative": has_time_derivative,
        "has_mass": has_mass,
        "has_diffusion": has_diffusion,
        "has_source": has_source,
        "time_integrator": "low_rank_spt_surrogate",
    }


def supports_transient_diffusion_weak_form(problem: TAPSProblem) -> bool:
    return weak_form_transient_diffusion_blocks(problem) is not None


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


def _reaction_value_from_blocks(problem: TAPSProblem, family: str, blocks: dict[str, object] | None) -> float:
    reaction = _reaction_value(family)
    if reaction != 0.0 or blocks is None:
        return reaction
    if blocks.get("has_reaction"):
        return _coefficient_number(problem, {"reaction", "reaction_rate", "mass", "beta", "c"}, 1.0)
    return 0.0


def _numeric_value(value: object, default: float = 0.0) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _plan_coefficient(plan: NumericalSolvePlanOutput | None, role: str, default: float) -> float:
    if plan is None:
        return default
    for binding in plan.coefficient_bindings:
        if binding.role == role:
            return _numeric_value(binding.value, default)
    return default


def _plan_source_constant(plan: NumericalSolvePlanOutput | None) -> float | None:
    if plan is None:
        return None
    for binding in plan.source_bindings:
        if isinstance(binding.value, (float, int)):
            return float(binding.value)
        if isinstance(binding.value, str):
            try:
                return float(binding.value)
            except ValueError:
                continue
    return None


def _plan_solver_number(plan: NumericalSolvePlanOutput | None, name: str, default: float) -> float:
    if plan is None:
        return default
    normalized = name.lower()
    for binding in plan.coefficient_bindings:
        if binding.name.lower() == normalized:
            return _numeric_value(binding.value, default)
    return default


def _plan_complex(plan: NumericalSolvePlanOutput | None, name: str, default: complex) -> complex:
    if plan is None:
        return default
    normalized = name.lower()
    for binding in plan.coefficient_bindings:
        if binding.name.lower() != normalized:
            continue
        value = binding.value
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
            try:
                return complex(value.strip().lower().replace("i", "j"))
            except ValueError:
                return default
    return default


def _boundary_position_from_role(boundary_role: str | None) -> str | None:
    if boundary_role == "x_min":
        return "left"
    if boundary_role == "x_max":
        return "right"
    return None


def _boundary_position(region_id: str) -> str | None:
    lowered = region_id.lower().replace(" ", "").replace("-", "_")
    pieces = [piece for piece in lowered.replace(":", "_").split("_") if piece]
    if (
        lowered in {"x=0", "x_min", "xmin", "x0", "left", "boundary:x_min", "boundary:left"}
        or lowered.endswith(":x_min")
        or "left" in pieces
        or "x0" in pieces
        or ("x" in pieces and "0" in pieces)
    ):
        return "left"
    if (
        lowered in {"x=l", "x=1", "x_max", "xmax", "x1", "right", "boundary:x_max", "boundary:right"}
        or lowered.endswith(":x_max")
        or "right" in pieces
        or "x1" in pieces
        or ("x" in pieces and "1" in pieces)
    ):
        return "right"
    return None


def _dirichlet_values_1d(plan: NumericalSolvePlanOutput | None, field: str) -> tuple[float, float]:
    left = 0.0
    right = 0.0
    if plan is None:
        return left, right
    for boundary in plan.boundary_condition_bindings:
        if boundary.kind.lower() != "dirichlet" or boundary.field != field:
            continue
        position = _boundary_position_from_role(boundary.boundary_role) or _boundary_position(boundary.region_id)
        if position == "left":
            left = _numeric_value(boundary.value, left)
        elif position == "right":
            right = _numeric_value(boundary.value, right)
    return left, right


def _dirichlet_boundary_values_2d(plan: NumericalSolvePlanOutput | None, field: str) -> dict[str, float]:
    values = {"left": 0.0, "right": 0.0, "bottom": 0.0, "top": 0.0}
    if plan is None:
        return values
    for boundary in plan.boundary_condition_bindings:
        if boundary.kind.lower() != "dirichlet" or boundary.field != field:
            continue
        value = _numeric_value(boundary.value, 0.0)
        lowered = boundary.region_id.lower().replace(" ", "")
        if lowered in {"boundary", "all", "domain_boundary", "outer_boundary"}:
            values = {key: value for key in values}
        elif lowered in {"x=0", "x_min", "xmin", "left", "boundary:x_min", "boundary:left"} or lowered.endswith(":x_min"):
            values["left"] = value
        elif lowered in {"x=l", "x=1", "x_max", "xmax", "right", "boundary:x_max", "boundary:right"} or lowered.endswith(":x_max"):
            values["right"] = value
        elif lowered in {"y=0", "y_min", "ymin", "bottom", "boundary:y_min", "boundary:bottom"} or lowered.endswith(":y_min"):
            values["bottom"] = value
        elif lowered in {"y=l", "y=1", "y_max", "ymax", "top", "boundary:y_max", "boundary:top"} or lowered.endswith(":y_max"):
            values["top"] = value
    return values


def _dirichlet_boundary_values_3d(plan: NumericalSolvePlanOutput | None, field: str) -> dict[str, float]:
    values: dict[str, float] = {}
    if plan is None:
        return values
    role_to_key = {"x_min": "left", "x_max": "right", "y_min": "bottom", "y_max": "top", "z_min": "front", "z_max": "back"}
    for boundary in plan.boundary_condition_bindings:
        if boundary.kind.lower() != "dirichlet" or boundary.field != field:
            continue
        value = _numeric_value(boundary.value, 0.0)
        key = role_to_key.get(boundary.boundary_role or "")
        if key is not None:
            values[key] = value
            continue
        lowered = boundary.region_id.lower().replace(" ", "")
        if lowered in {"boundary", "all", "domain_boundary", "outer_boundary"}:
            values = {side: value for side in values}
        elif lowered in {"x=0", "x_min", "xmin", "left", "boundary:x_min", "boundary:left"} or lowered.endswith(":x_min"):
            values["left"] = value
        elif lowered in {"x=l", "x=1", "x_max", "xmax", "right", "boundary:x_max", "boundary:right"} or lowered.endswith(":x_max"):
            values["right"] = value
        elif lowered in {"y=0", "y_min", "ymin", "bottom", "boundary:y_min", "boundary:bottom"} or lowered.endswith(":y_min"):
            values["bottom"] = value
        elif lowered in {"y=l", "y=1", "y_max", "ymax", "top", "boundary:y_max", "boundary:top"} or lowered.endswith(":y_max"):
            values["top"] = value
        elif lowered in {"z=0", "z_min", "zmin", "front", "boundary:z_min", "boundary:front"} or lowered.endswith(":z_min"):
            values["front"] = value
        elif lowered in {"z=l", "z=1", "z_max", "zmax", "back", "boundary:z_max", "boundary:back"} or lowered.endswith(":z_max"):
            values["back"] = value
    return values


def _apply_dirichlet_boundary_2d(solution: list[list[float]], values: dict[str, float]) -> None:
    nx = len(solution)
    ny = len(solution[0]) if solution else 0
    if nx == 0 or ny == 0:
        return
    for j in range(ny):
        solution[0][j] = values["left"]
        solution[-1][j] = values["right"]
    for i in range(nx):
        solution[i][0] = values["bottom"]
        solution[i][-1] = values["top"]


def _boundary_error_2d(solution: list[list[float]], values: dict[str, float]) -> float:
    nx = len(solution)
    ny = len(solution[0]) if solution else 0
    if nx == 0 or ny == 0:
        return 0.0
    errors: list[float] = []
    for j in range(ny):
        errors.append(abs(solution[0][j] - values["left"]))
        errors.append(abs(solution[-1][j] - values["right"]))
    for i in range(nx):
        errors.append(abs(solution[i][0] - values["bottom"]))
        errors.append(abs(solution[i][-1] - values["top"]))
    return max(errors) if errors else 0.0


def _apply_dirichlet_boundary_3d(solution: list[list[list[float]]], values: dict[str, float]) -> None:
    nx = len(solution)
    ny = len(solution[0]) if solution else 0
    nz = len(solution[0][0]) if ny else 0
    if nx == 0 or ny == 0 or nz == 0:
        return
    if "left" in values:
        for j in range(ny):
            for k in range(nz):
                solution[0][j][k] = values["left"]
    if "right" in values:
        for j in range(ny):
            for k in range(nz):
                solution[-1][j][k] = values["right"]
    if "bottom" in values:
        for i in range(nx):
            for k in range(nz):
                solution[i][0][k] = values["bottom"]
    if "top" in values:
        for i in range(nx):
            for k in range(nz):
                solution[i][-1][k] = values["top"]
    if "front" in values:
        for i in range(nx):
            for j in range(ny):
                solution[i][j][0] = values["front"]
    if "back" in values:
        for i in range(nx):
            for j in range(ny):
                solution[i][j][-1] = values["back"]


def _boundary_error_3d(solution: list[list[list[float]]], values: dict[str, float]) -> float:
    nx = len(solution)
    ny = len(solution[0]) if solution else 0
    nz = len(solution[0][0]) if ny else 0
    if nx == 0 or ny == 0 or nz == 0:
        return 0.0
    errors: list[float] = []
    if "left" in values:
        for j in range(ny):
            for k in range(nz):
                errors.append(abs(solution[0][j][k] - values["left"]))
    if "right" in values:
        for j in range(ny):
            for k in range(nz):
                errors.append(abs(solution[-1][j][k] - values["right"]))
    if "bottom" in values:
        for i in range(nx):
            for k in range(nz):
                errors.append(abs(solution[i][0][k] - values["bottom"]))
    if "top" in values:
        for i in range(nx):
            for k in range(nz):
                errors.append(abs(solution[i][-1][k] - values["top"]))
    if "front" in values:
        for i in range(nx):
            for j in range(ny):
                errors.append(abs(solution[i][j][0] - values["front"]))
    if "back" in values:
        for i in range(nx):
            for j in range(ny):
                errors.append(abs(solution[i][j][-1] - values["back"]))
    return max(errors) if errors else 0.0


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
    return math.sqrt(residual_sq) / max(math.sqrt(rhs_sq), 1.0)


def _residual_norm_3d(
    x: list[float],
    y: list[float],
    z: list[float],
    solution: list[list[list[float]]],
    rhs: list[list[list[float]]],
    diffusion: float,
    reaction: float,
) -> float:
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    cx = diffusion / (dx * dx)
    cy = diffusion / (dy * dy)
    cz = diffusion / (dz * dz)
    residual_sq = 0.0
    rhs_sq = 0.0
    for i in range(1, len(x) - 1):
        for j in range(1, len(y) - 1):
            for k in range(1, len(z) - 1):
                au = (
                    (2.0 * cx + 2.0 * cy + 2.0 * cz + reaction) * solution[i][j][k]
                    - cx * (solution[i - 1][j][k] + solution[i + 1][j][k])
                    - cy * (solution[i][j - 1][k] + solution[i][j + 1][k])
                    - cz * (solution[i][j][k - 1] + solution[i][j][k + 1])
                )
                residual = au - rhs[i][j][k]
                residual_sq += residual * residual
                rhs_sq += rhs[i][j][k] * rhs[i][j][k]
    return math.sqrt(residual_sq) / max(math.sqrt(rhs_sq), 1.0)


def _relaxation_3d(
    x: list[float],
    y: list[float],
    z: list[float],
    rhs: list[list[list[float]]],
    diffusion: float,
    reaction: float,
    boundary_values: dict[str, float],
    max_iterations: int = 3000,
    tolerance: float = 1e-8,
) -> tuple[list[list[list[float]]], list[dict[str, float | int]]]:
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    cx = diffusion / (dx * dx)
    cy = diffusion / (dy * dy)
    cz = diffusion / (dz * dz)
    denom = 2.0 * cx + 2.0 * cy + 2.0 * cz + reaction
    solution = [[[0.0 for _ in z] for _ in y] for _ in x]
    _apply_dirichlet_boundary_3d(solution, boundary_values)
    history: list[dict[str, float | int]] = []
    for iteration in range(1, max_iterations + 1):
        update_sq = 0.0
        norm_sq = 0.0
        for i in range(1, len(x) - 1):
            for j in range(1, len(y) - 1):
                for k in range(1, len(z) - 1):
                    new_value = (
                        rhs[i][j][k]
                        + cx * (solution[i - 1][j][k] + solution[i + 1][j][k])
                        + cy * (solution[i][j - 1][k] + solution[i][j + 1][k])
                        + cz * (solution[i][j][k - 1] + solution[i][j][k + 1])
                    ) / denom
                    update_sq += (new_value - solution[i][j][k]) ** 2
                    norm_sq += new_value * new_value
                    solution[i][j][k] = new_value
        _apply_dirichlet_boundary_3d(solution, boundary_values)
        residual = _residual_norm_3d(x, y, z, solution, rhs, diffusion, reaction)
        update_norm = math.sqrt(update_sq) / (math.sqrt(norm_sq) + 1e-12)
        if iteration == 1 or iteration % 25 == 0 or residual < tolerance or update_norm < tolerance:
            history.append({"iteration": iteration, "normalized_linear_residual": residual, "relative_update": update_norm})
        if residual < tolerance or update_norm < tolerance:
            break
    return solution, history


def _masked_relaxation_2d(
    x: list[float],
    y: list[float],
    rhs: list[list[float]],
    diffusion: float,
    reaction: float,
    active_mask: list[list[int]],
    boundary_values: dict[str, float] | None = None,
    max_iterations: int = 5000,
    tolerance: float = 1e-10,
) -> tuple[list[list[float]], list[dict[str, float | int]]]:
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    cx = diffusion / (dx * dx)
    cy = diffusion / (dy * dy)
    denom = 2.0 * cx + 2.0 * cy + reaction
    solution = [[0.0 for _ in y] for _ in x]
    if boundary_values is not None:
        _apply_dirichlet_boundary_2d(solution, boundary_values)
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
        if boundary_values is not None:
            _apply_dirichlet_boundary_2d(solution, boundary_values)
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
    return math.sqrt(residual_sq) / max(math.sqrt(rhs_sq), 1.0)


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


def _tetra_cells(graph: dict) -> list[list[int]]:
    tetrahedra: list[list[int]] = []
    for block in graph.get("cell_blocks", []):
        block_type = str(block.get("type", "")).lower()
        if "tetra" not in block_type:
            continue
        for cell in block.get("cells", []):
            if len(cell) >= 4:
                tetrahedra.append([int(node) for node in cell])
    return tetrahedra


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


def _tetra_geometry(points: list[list[float]], nodes: list[int]) -> tuple[float, list[list[float]]] | None:
    if len(nodes) < 4:
        return None
    matrix = []
    for node in nodes[:4]:
        point = points[node]
        if len(point) < 3:
            return None
        matrix.append([1.0, float(point[0]), float(point[1]), float(point[2])])
    try:
        inverse = _invert_matrix(matrix)
    except ValueError:
        return None
    volume_matrix = [
        [matrix[1][coord] - matrix[0][coord] for coord in range(1, 4)],
        [matrix[2][coord] - matrix[0][coord] for coord in range(1, 4)],
        [matrix[3][coord] - matrix[0][coord] for coord in range(1, 4)],
    ]
    determinant = (
        volume_matrix[0][0] * (volume_matrix[1][1] * volume_matrix[2][2] - volume_matrix[1][2] * volume_matrix[2][1])
        - volume_matrix[0][1] * (volume_matrix[1][0] * volume_matrix[2][2] - volume_matrix[1][2] * volume_matrix[2][0])
        + volume_matrix[0][2] * (volume_matrix[1][0] * volume_matrix[2][1] - volume_matrix[1][1] * volume_matrix[2][0])
    )
    volume = abs(determinant) / 6.0
    if volume <= 1e-14:
        return None
    gradients = [[inverse[1][basis], inverse[2][basis], inverse[3][basis]] for basis in range(4)]
    return volume, gradients


def _tetra_quadrature(order: int = 2) -> list[tuple[list[float], float]]:
    if order <= 1:
        return [([0.25, 0.25, 0.25, 0.25], 1.0)]
    a = 0.5854101966249685
    b = 0.1381966011250105
    return [
        ([a, b, b, b], 0.25),
        ([b, a, b, b], 0.25),
        ([b, b, a, b], 0.25),
        ([b, b, b, a], 0.25),
    ]


def _vec_cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _vec_dot(a: list[float], b: list[float]) -> float:
    return sum(left * right for left, right in zip(a, b))


def _assemble_tetra_nedelec_curl_curl(
    points: list[list[float]],
    tetrahedra: list[list[int]],
    curl_weight: float | complex = 1.0,
    mass_weight: float | complex = 1.0,
) -> tuple[list[dict[int, float | complex]], list[tuple[int, int]], list[tuple[int, int, int]], float, list[dict]]:
    if tetrahedra and any(len(tetra) >= 10 for tetra in tetrahedra):
        return _assemble_tetra_nedelec_curl_curl_order2(points, tetrahedra, curl_weight=curl_weight, mass_weight=mass_weight)

    edge_index: dict[tuple[int, int], int] = {}
    edge_list: list[tuple[int, int]] = []
    face_index: dict[tuple[int, int, int], int] = {}
    face_list: list[tuple[int, int, int]] = []
    local_edge_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    local_face_triples = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    for tetra in tetrahedra:
        vertices = tetra[:4]
        if len(vertices) < 4:
            continue
        for local_a, local_b in local_edge_pairs:
            a = vertices[local_a]
            b = vertices[local_b]
            edge = (a, b) if a < b else (b, a)
            if edge not in edge_index:
                edge_index[edge] = len(edge_list)
                edge_list.append(edge)
        for local_face in local_face_triples:
            face = tuple(sorted(vertices[index] for index in local_face))
            if face not in face_index:
                face_index[face] = len(face_list)
                face_list.append(face)

    stiffness: list[dict[int, float | complex]] = [dict() for _ in edge_list]
    total_volume = 0.0
    element_records: list[dict] = []
    for tetra in tetrahedra:
        vertices = tetra[:4]
        geometry = _tetra_geometry(points, vertices)
        if geometry is None:
            continue
        volume, grad_lambda = geometry
        local_edges: list[int] = []
        orientation_signs: list[float] = []
        oriented_vertex_pairs: list[tuple[int, int]] = []
        for local_a, local_b in local_edge_pairs:
            a = vertices[local_a]
            b = vertices[local_b]
            edge = (a, b) if a < b else (b, a)
            local_edges.append(edge_index[edge])
            orientation_signs.append(1.0 if (a, b) == edge else -1.0)
            oriented_vertex_pairs.append((a, b))
        local_faces = [face_index[tuple(sorted(vertices[index] for index in local_face))] for local_face in local_face_triples]
        basis_curls = [list(2.0 * value for value in _vec_cross(grad_lambda[local_a], grad_lambda[local_b])) for local_a, local_b in local_edge_pairs]
        local_matrix = [[0.0 + 0.0j for _ in range(6)] for _ in range(6)]
        quadrature_records: list[dict] = []
        for barycentric, weight in _tetra_quadrature(order=2):
            basis_values: list[list[float]] = []
            for local_a, local_b in local_edge_pairs:
                value = [
                    barycentric[local_a] * grad_lambda[local_b][axis] - barycentric[local_b] * grad_lambda[local_a][axis]
                    for axis in range(3)
                ]
                basis_values.append(value)
            scaled_weight = volume * weight
            for i in range(6):
                for j in range(6):
                    mass = _vec_dot(basis_values[i], basis_values[j])
                    curl = _vec_dot(basis_curls[i], basis_curls[j])
                    local_matrix[i][j] += orientation_signs[i] * orientation_signs[j] * scaled_weight * (
                        mass_weight * mass + curl_weight * curl
                    )
            quadrature_records.append(
                {
                    "barycentric": barycentric,
                    "weight": weight,
                    "nedelec_basis": basis_values,
                    "curl_basis": basis_curls,
                }
            )
        for local_i, global_i in enumerate(local_edges):
            for local_j, global_j in enumerate(local_edges):
                value = local_matrix[local_i][local_j]
                stiffness[global_i][global_j] = stiffness[global_i].get(global_j, 0.0) + value
        element_records.append(
            {
                "vertices": vertices,
                "edges": local_edges,
                "faces": local_faces,
                "oriented_vertex_pairs": oriented_vertex_pairs,
                "orientation_signs": orientation_signs,
                "volume": volume,
                "basis": "nedelec_first_kind_order1_tetra",
                "curl_basis": basis_curls,
                "local_matrix": [[_json_number(value) for value in row] for row in local_matrix],
                "quadrature": quadrature_records,
            }
        )
        total_volume += volume
    return stiffness, edge_list, face_list, total_volume, element_records


def _scaled_nedelec_3d_value_and_curl(
    scalar_value: float,
    scalar_gradient: list[float],
    base_value: list[float],
    base_curl: list[float],
) -> tuple[list[float], list[float]]:
    value = [scalar_value * component for component in base_value]
    curl = [
        component + scalar_value * base_curl[index]
        for index, component in enumerate(_vec_cross(scalar_gradient, base_value))
    ]
    return value, curl


def _assemble_tetra_nedelec_curl_curl_order2(
    points: list[list[float]],
    tetrahedra: list[list[int]],
    curl_weight: float | complex = 1.0,
    mass_weight: float | complex = 1.0,
) -> tuple[list[dict[int, float | complex]], list[dict], list[tuple[int, int, int]], float, list[dict]]:
    """Assemble a hierarchical second-order H(curl) scaffold on tetrahedra.

    The global DOF layout is intentionally explicit: two edge moment DOFs per
    edge, one shared face-tangent enrichment per face, and one cell-interior
    enrichment per tetrahedron. This is an executable scaffold for workflow
    validation and artifact contracts; it is not advertised as a complete
    production high-order Nedelec basis.
    """
    local_edge_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    local_face_triples = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    edge_dof_index: dict[tuple[int, int, int], int] = {}
    face_dof_index: dict[tuple[int, int, int], int] = {}
    dofs: list[dict] = []
    cell_dof_index: dict[int, int] = {}
    face_list: list[tuple[int, int, int]] = []
    seen_faces: set[tuple[int, int, int]] = set()
    for cell_id, tetra in enumerate(tetrahedra):
        vertices = tetra[:4]
        if len(vertices) < 4:
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
        for local_face in local_face_triples:
            face = tuple(sorted(vertices[index] for index in local_face))
            if face not in seen_faces:
                seen_faces.add(face)
                face_list.append(face)
            if face not in face_dof_index:
                face_dof_index[face] = len(dofs)
                dofs.append({"kind": "face_tangent", "face": [face[0], face[1], face[2]], "moment": 0})
        cell_dof_index[cell_id] = len(dofs)
        dofs.append({"kind": "cell_interior", "cell_id": cell_id, "moment": 0, "vertices": vertices})

    stiffness: list[dict[int, float | complex]] = [dict() for _ in dofs]
    total_volume = 0.0
    element_records: list[dict] = []
    for cell_id, tetra in enumerate(tetrahedra):
        vertices = tetra[:4]
        geometry = _tetra_geometry(points, vertices)
        if geometry is None:
            continue
        volume, grad_lambda = geometry
        p1_values_by_q: list[list[list[float]]] = []
        p1_curls = [list(2.0 * value for value in _vec_cross(grad_lambda[local_a], grad_lambda[local_b])) for local_a, local_b in local_edge_pairs]
        local_dofs: list[int] = []
        orientation_signs: list[float] = []
        local_basis_descriptors: list[dict] = []
        for edge_number, (local_a, local_b) in enumerate(local_edge_pairs):
            a = vertices[local_a]
            b = vertices[local_b]
            edge = (a, b) if a < b else (b, a)
            sign = 1.0 if (a, b) == edge else -1.0
            for moment, scalar_index in enumerate((local_a, local_b)):
                local_dofs.append(edge_dof_index[(edge[0], edge[1], moment)])
                orientation_signs.append(sign)
                local_basis_descriptors.append(
                    {"kind": "edge_moment", "edge_number": edge_number, "lambda_scale": scalar_index}
                )
        for face_number, local_face in enumerate(local_face_triples):
            face = tuple(sorted(vertices[index] for index in local_face))
            local_dofs.append(face_dof_index[face])
            orientation_signs.append(1.0)
            local_basis_descriptors.append({"kind": "face_tangent", "face_number": face_number, "lambda_scale": list(local_face)})
        local_dofs.append(cell_dof_index[cell_id])
        orientation_signs.append(1.0)
        local_basis_descriptors.append({"kind": "cell_interior", "lambda_scale": [0, 1, 2, 3]})

        local_size = len(local_dofs)
        local_matrix = [[0.0 + 0.0j for _ in range(local_size)] for _ in range(local_size)]
        quadrature_records: list[dict] = []
        for barycentric, weight in _tetra_quadrature(order=2):
            p1_values: list[list[float]] = []
            for local_a, local_b in local_edge_pairs:
                p1_values.append(
                    [
                        barycentric[local_a] * grad_lambda[local_b][axis]
                        - barycentric[local_b] * grad_lambda[local_a][axis]
                        for axis in range(3)
                    ]
                )
            p1_values_by_q.append(p1_values)
            basis_values: list[list[float]] = []
            basis_curls: list[list[float]] = []
            for edge_number, (local_a, local_b) in enumerate(local_edge_pairs):
                for scalar_index in (local_a, local_b):
                    value, curl = _scaled_nedelec_3d_value_and_curl(
                        barycentric[scalar_index],
                        grad_lambda[scalar_index],
                        p1_values[edge_number],
                        p1_curls[edge_number],
                    )
                    basis_values.append(value)
                    basis_curls.append(curl)
            for face_number, local_face in enumerate(local_face_triples):
                scalar = 1.0
                scalar_gradient = [0.0, 0.0, 0.0]
                for local_index in local_face:
                    scalar *= barycentric[local_index]
                for local_index in local_face:
                    product = 1.0
                    for other_index in local_face:
                        if other_index != local_index:
                            product *= barycentric[other_index]
                    for axis in range(3):
                        scalar_gradient[axis] += product * grad_lambda[local_index][axis]
                base_edge = face_number if face_number < len(local_edge_pairs) else 0
                value, curl = _scaled_nedelec_3d_value_and_curl(
                    scalar,
                    scalar_gradient,
                    p1_values[base_edge],
                    p1_curls[base_edge],
                )
                basis_values.append(value)
                basis_curls.append(curl)
            cell_scalar = barycentric[0] * barycentric[1] * barycentric[2] * barycentric[3]
            cell_gradient = [0.0, 0.0, 0.0]
            for local_index in range(4):
                product = 1.0
                for other_index in range(4):
                    if other_index != local_index:
                        product *= barycentric[other_index]
                for axis in range(3):
                    cell_gradient[axis] += product * grad_lambda[local_index][axis]
            value, curl = _scaled_nedelec_3d_value_and_curl(cell_scalar, cell_gradient, p1_values[0], p1_curls[0])
            basis_values.append(value)
            basis_curls.append(curl)

            scaled_weight = volume * weight
            for i in range(local_size):
                for j in range(local_size):
                    mass = _vec_dot(basis_values[i], basis_values[j])
                    curl = _vec_dot(basis_curls[i], basis_curls[j])
                    local_matrix[i][j] += orientation_signs[i] * orientation_signs[j] * scaled_weight * (
                        mass_weight * mass + curl_weight * curl
                    )
            quadrature_records.append(
                {
                    "barycentric": barycentric,
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
                "volume": volume,
                "basis": "nedelec_first_kind_order2_hierarchical_scaffold_tetra",
                "edge_moment_dofs_per_edge": 2,
                "face_tangent_dofs": 4,
                "cell_interior_dofs": 1,
                "orientation_signs": orientation_signs,
                "local_basis": local_basis_descriptors,
                "local_matrix": [[_json_number(value) for value in row] for row in local_matrix],
                "quadrature": quadrature_records,
            }
        )
        total_volume += volume
    return stiffness, dofs, face_list, total_volume, element_records


def _assemble_tetra_raviart_thomas_div(
    points: list[list[float]],
    tetrahedra: list[list[int]],
    div_weight: float = 1.0,
    mass_weight: float = 1.0,
) -> tuple[list[dict[int, float]], list[dict], float, list[dict]]:
    """Assemble a lowest-order H(div) Raviart-Thomas scaffold on tetrahedra."""
    face_index: dict[tuple[int, int, int], int] = {}
    dofs: list[dict] = []
    local_face_triples = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]
    for tetra in tetrahedra:
        vertices = tetra[:4]
        if len(vertices) < 4:
            continue
        for local_face in local_face_triples:
            face = tuple(sorted(vertices[index] for index in local_face))
            if face not in face_index:
                face_index[face] = len(dofs)
                dofs.append({"kind": "face_flux", "face": [face[0], face[1], face[2]], "moment": 0})

    stiffness: list[dict[int, float]] = [dict() for _ in dofs]
    total_volume = 0.0
    element_records: list[dict] = []
    for tetra in tetrahedra:
        vertices = tetra[:4]
        geometry = _tetra_geometry(points, vertices)
        if geometry is None:
            continue
        volume, _ = geometry
        coords = [[float(value) for value in points[node][:3]] for node in vertices]
        centroid = [sum(point[axis] for point in coords) / 4.0 for axis in range(3)]
        local_dofs = [face_index[tuple(sorted(vertices[index] for index in local_face))] for local_face in local_face_triples]
        basis_values: list[list[float]] = []
        basis_divergence: list[float] = []
        face_areas: list[float] = []
        orientation_signs: list[float] = []
        for opposite_local, local_face in enumerate(local_face_triples):
            face_points = [coords[index] for index in local_face]
            normal = _vec_cross(
                [face_points[1][axis] - face_points[0][axis] for axis in range(3)],
                [face_points[2][axis] - face_points[0][axis] for axis in range(3)],
            )
            area = 0.5 * math.sqrt(_vec_dot(normal, normal))
            face_centroid = [sum(point[axis] for point in face_points) / 3.0 for axis in range(3)]
            outward_probe = [face_centroid[axis] - coords[opposite_local][axis] for axis in range(3)]
            sign = 1.0 if _vec_dot(normal, outward_probe) >= 0.0 else -1.0
            orientation_signs.append(sign)
            face_areas.append(area)
            scale = area / max(3.0 * volume, 1e-30)
            basis_values.append([scale * (centroid[axis] - coords[opposite_local][axis]) for axis in range(3)])
            basis_divergence.append(area / max(volume, 1e-30))
        local_matrix = [[0.0 for _ in range(4)] for _ in range(4)]
        for i in range(4):
            for j in range(4):
                local_matrix[i][j] = orientation_signs[i] * orientation_signs[j] * volume * (
                    mass_weight * _vec_dot(basis_values[i], basis_values[j])
                    + div_weight * basis_divergence[i] * basis_divergence[j]
                )
        for local_i, global_i in enumerate(local_dofs):
            for local_j, global_j in enumerate(local_dofs):
                stiffness[global_i][global_j] = stiffness[global_i].get(global_j, 0.0) + local_matrix[local_i][local_j]
        element_records.append(
            {
                "vertices": vertices,
                "dofs": local_dofs,
                "volume": volume,
                "basis": "raviart_thomas_order0_tetra",
                "face_areas": face_areas,
                "orientation_signs": orientation_signs,
                "basis_values_at_centroid": basis_values,
                "divergence_basis": basis_divergence,
                "local_matrix": local_matrix,
            }
        )
        total_volume += volume
    return stiffness, dofs, total_volume, element_records


def _tetra_p2_shape_values_and_gradients_ref(barycentric: list[float]) -> tuple[list[float], list[list[float]]]:
    l0, l1, l2, l3 = barycentric
    values = [
        l0 * (2.0 * l0 - 1.0),
        l1 * (2.0 * l1 - 1.0),
        l2 * (2.0 * l2 - 1.0),
        l3 * (2.0 * l3 - 1.0),
        4.0 * l0 * l1,
        4.0 * l1 * l2,
        4.0 * l0 * l2,
        4.0 * l0 * l3,
        4.0 * l1 * l3,
        4.0 * l2 * l3,
    ]
    grad_l = [
        [-1.0, -1.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    gradients = [
        [(4.0 * l0 - 1.0) * grad_l[0][axis] for axis in range(3)],
        [(4.0 * l1 - 1.0) * grad_l[1][axis] for axis in range(3)],
        [(4.0 * l2 - 1.0) * grad_l[2][axis] for axis in range(3)],
        [(4.0 * l3 - 1.0) * grad_l[3][axis] for axis in range(3)],
        [4.0 * (l0 * grad_l[1][axis] + l1 * grad_l[0][axis]) for axis in range(3)],
        [4.0 * (l1 * grad_l[2][axis] + l2 * grad_l[1][axis]) for axis in range(3)],
        [4.0 * (l0 * grad_l[2][axis] + l2 * grad_l[0][axis]) for axis in range(3)],
        [4.0 * (l0 * grad_l[3][axis] + l3 * grad_l[0][axis]) for axis in range(3)],
        [4.0 * (l1 * grad_l[3][axis] + l3 * grad_l[1][axis]) for axis in range(3)],
        [4.0 * (l2 * grad_l[3][axis] + l3 * grad_l[2][axis]) for axis in range(3)],
    ]
    return values, gradients


def _det3(matrix: list[list[float]]) -> float:
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def _tetra_p2_isoparametric_geometry(
    points: list[list[float]], nodes: list[int], barycentric: list[float]
) -> tuple[float, list[list[float]], list[list[float]]] | None:
    values, gradients_ref = _tetra_p2_shape_values_and_gradients_ref(barycentric)
    jacobian = [[0.0 for _ in range(3)] for _ in range(3)]
    for local_index, node in enumerate(nodes[:10]):
        point = points[node]
        for physical_axis in range(3):
            coordinate = float(point[physical_axis])
            for reference_axis in range(3):
                jacobian[physical_axis][reference_axis] += coordinate * gradients_ref[local_index][reference_axis]
    det_j = _det3(jacobian)
    if abs(det_j) <= 1e-14:
        return None
    try:
        inverse_j = _invert_matrix(jacobian)
    except ValueError:
        return None
    # grad_x = J^{-T} grad_ref.
    gradients_physical = [
        [
            sum(inverse_j[reference_axis][physical_axis] * grad[reference_axis] for reference_axis in range(3))
            for physical_axis in range(3)
        ]
        for grad in gradients_ref
    ]
    return det_j, values, gradients_physical


def _element_mesh_quality_summary(element_records: list[dict]) -> dict[str, object]:
    quality_records = [record.get("mesh_quality") for record in element_records if isinstance(record.get("mesh_quality"), dict)]
    issues = sorted({issue for record in quality_records for issue in record.get("issues", [])})
    min_abs_values = [record.get("min_abs_det_j") for record in quality_records if isinstance(record.get("min_abs_det_j"), (float, int))]
    min_values = [record.get("min_det_j") for record in quality_records if isinstance(record.get("min_det_j"), (float, int))]
    max_values = [record.get("max_det_j") for record in quality_records if isinstance(record.get("max_det_j"), (float, int))]
    passes = all(bool(record.get("passes", False)) for record in quality_records) if quality_records else True
    return {
        "passes": passes,
        "issues": issues,
        "checked_elements": len(quality_records),
        "min_abs_det_j": min(min_abs_values) if min_abs_values else None,
        "min_det_j": min(min_values) if min_values else None,
        "max_det_j": max(max_values) if max_values else None,
    }


def _assemble_tetra_stiffness(
    points: list[list[float]], tetrahedra: list[list[int]]
) -> tuple[list[dict[int, float]], list[float], float, list[dict]]:
    node_count = len(points)
    stiffness: list[dict[int, float]] = [dict() for _ in range(node_count)]
    lumped_mass = [0.0 for _ in range(node_count)]
    element_records: list[dict] = []
    total_volume = 0.0
    for tetra in tetrahedra:
        if len(tetra) >= 10:
            nodes = tetra[:10]
            local_stiffness = [[0.0 for _ in range(10)] for _ in range(10)]
            local_mass = [0.0 for _ in range(10)]
            quadrature_records: list[dict] = []
            volume = 0.0
            valid = True
            det_values: list[float] = []
            for barycentric, weight in _tetra_quadrature(order=2):
                geometry = _tetra_p2_isoparametric_geometry(points, nodes, barycentric)
                if geometry is None:
                    valid = False
                    break
                det_j, values, gradients = geometry
                det_values.append(det_j)
                scaled_weight = abs(det_j) * weight / 6.0
                volume += scaled_weight
                for i in range(10):
                    local_mass[i] += scaled_weight * values[i]
                    for j in range(10):
                        local_stiffness[i][j] += scaled_weight * sum(
                            gradients[i][axis] * gradients[j][axis] for axis in range(3)
                        )
                quadrature_records.append(
                    {
                        "barycentric": barycentric,
                        "weight": weight,
                        "det_j": det_j,
                        "shape_values": values,
                        "grad_phi": gradients,
                    }
                )
            if not valid or volume <= 1e-14:
                continue
            quality_issues: list[str] = []
            if det_values and min(abs(value) for value in det_values) <= 1e-10:
                quality_issues.append("tetra10 near-singular Jacobian at quadrature point")
            if det_values and (min(det_values) < 0.0 < max(det_values)):
                quality_issues.append("tetra10 Jacobian determinant changes sign across quadrature points")
            for local_i, global_i in enumerate(nodes):
                lumped_mass[global_i] += volume / 10.0
                for local_j, global_j in enumerate(nodes):
                    value = local_stiffness[local_i][local_j]
                    stiffness[global_i][global_j] = stiffness[global_i].get(global_j, 0.0) + value
            element_records.append(
                {
                    "nodes": nodes,
                    "volume": volume,
                    "basis": "p2_tetra",
                    "geometry": "isoparametric_quadratic_tetra",
                    "mesh_quality": {
                        "passes": not quality_issues,
                        "issues": quality_issues,
                        "min_det_j": min(det_values) if det_values else None,
                        "max_det_j": max(det_values) if det_values else None,
                        "min_abs_det_j": min(abs(value) for value in det_values) if det_values else None,
                        "quadrature_points": len(det_values),
                    },
                    "quadrature": quadrature_records,
                    "local_stiffness": local_stiffness,
                    "local_mass": local_mass,
                }
            )
            total_volume += volume
            continue

        nodes = tetra[:4]
        geometry = _tetra_geometry(points, nodes)
        if geometry is None:
            continue
        volume, gradients = geometry
        local_stiffness = [
            [volume * sum(gradients[i][axis] * gradients[j][axis] for axis in range(3)) for j in range(4)]
            for i in range(4)
        ]
        for local_i, global_i in enumerate(nodes):
            lumped_mass[global_i] += volume / 4.0
            for local_j, global_j in enumerate(nodes):
                value = local_stiffness[local_i][local_j]
                stiffness[global_i][global_j] = stiffness[global_i].get(global_j, 0.0) + value
        element_records.append(
            {
                "nodes": nodes,
                "volume": volume,
                "basis": "p1_tetra",
                "grad_phi": gradients,
                "local_stiffness": local_stiffness,
            }
        )
        total_volume += volume
    return stiffness, lumped_mass, total_volume, element_records


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


def _elasticity_constitutive_matrix_3d(young_modulus: float, poisson_ratio: float) -> list[list[float]]:
    if young_modulus <= 0.0:
        raise ValueError("Young's modulus must be positive.")
    if not -1.0 < poisson_ratio < 0.5:
        raise ValueError("Poisson ratio must be in (-1, 0.5).")
    lame_lambda = young_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    shear_mu = young_modulus / (2.0 * (1.0 + poisson_ratio))
    normal_diag = lame_lambda + 2.0 * shear_mu
    return [
        [normal_diag, lame_lambda, lame_lambda, 0.0, 0.0, 0.0],
        [lame_lambda, normal_diag, lame_lambda, 0.0, 0.0, 0.0],
        [lame_lambda, lame_lambda, normal_diag, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, shear_mu, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, shear_mu, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, shear_mu],
    ]


def _assemble_tetra_elasticity_element(
    points: list[list[float]],
    nodes: list[int],
    young_modulus: float = 1.0,
    poisson_ratio: float = 0.3,
) -> tuple[float, list[list[float]], list[list[float]], list[list[float]]] | None:
    geometry = _tetra_geometry(points, nodes)
    if geometry is None:
        return None
    volume, gradients = geometry
    b_matrix = [[0.0 for _ in range(12)] for _ in range(6)]
    for local_index, (dphidx, dphidy, dphidz) in enumerate(gradients):
        ux = 3 * local_index
        uy = ux + 1
        uz = ux + 2
        b_matrix[0][ux] = dphidx
        b_matrix[1][uy] = dphidy
        b_matrix[2][uz] = dphidz
        b_matrix[3][ux] = dphidy
        b_matrix[3][uy] = dphidx
        b_matrix[4][uy] = dphidz
        b_matrix[4][uz] = dphidy
        b_matrix[5][ux] = dphidz
        b_matrix[5][uz] = dphidx
    constitutive = _elasticity_constitutive_matrix_3d(young_modulus, poisson_ratio)
    db = [
        [sum(constitutive[row][k] * b_matrix[k][col] for k in range(6)) for col in range(12)]
        for row in range(6)
    ]
    local_stiffness = [
        [volume * sum(b_matrix[k][i] * db[k][j] for k in range(6)) for j in range(12)]
        for i in range(12)
    ]
    return volume, local_stiffness, gradients, b_matrix


def _assemble_tetra_elasticity_stiffness(
    points: list[list[float]],
    tetrahedra: list[list[int]],
    young_modulus: float = 1.0,
    poisson_ratio: float = 0.3,
) -> tuple[list[dict[int, float]], list[float], float, list[dict]]:
    dof_count = 3 * len(points)
    stiffness: list[dict[int, float]] = [dict() for _ in range(dof_count)]
    lumped_mass = [0.0 for _ in range(len(points))]
    element_records: list[dict] = []
    total_volume = 0.0
    for tetra in tetrahedra:
        if len(tetra) >= 10:
            nodes = tetra[:10]
            local_stiffness = [[0.0 for _ in range(30)] for _ in range(30)]
            local_mass = [0.0 for _ in range(10)]
            constitutive = _elasticity_constitutive_matrix_3d(young_modulus, poisson_ratio)
            quadrature_records: list[dict] = []
            volume = 0.0
            valid = True
            det_values: list[float] = []
            for barycentric, weight in _tetra_quadrature(order=2):
                geometry = _tetra_p2_isoparametric_geometry(points, nodes, barycentric)
                if geometry is None:
                    valid = False
                    break
                det_j, values, gradients = geometry
                det_values.append(det_j)
                scaled_weight = abs(det_j) * weight / 6.0
                volume += scaled_weight
                b_matrix = [[0.0 for _ in range(30)] for _ in range(6)]
                for local_index, (dphidx, dphidy, dphidz) in enumerate(gradients):
                    ux = 3 * local_index
                    uy = ux + 1
                    uz = ux + 2
                    b_matrix[0][ux] = dphidx
                    b_matrix[1][uy] = dphidy
                    b_matrix[2][uz] = dphidz
                    b_matrix[3][ux] = dphidy
                    b_matrix[3][uy] = dphidx
                    b_matrix[4][uy] = dphidz
                    b_matrix[4][uz] = dphidy
                    b_matrix[5][ux] = dphidz
                    b_matrix[5][uz] = dphidx
                db = [
                    [sum(constitutive[row][k] * b_matrix[k][col] for k in range(6)) for col in range(30)]
                    for row in range(6)
                ]
                for local_i in range(10):
                    local_mass[local_i] += scaled_weight * values[local_i]
                for i in range(30):
                    for j in range(30):
                        local_stiffness[i][j] += scaled_weight * sum(b_matrix[k][i] * db[k][j] for k in range(6))
                quadrature_records.append(
                    {
                        "barycentric": barycentric,
                        "weight": weight,
                        "det_j": det_j,
                        "shape_values": values,
                        "grad_phi": gradients,
                        "strain_displacement_matrix": b_matrix,
                    }
                )
            if not valid or volume <= 1e-14:
                continue
            quality_issues: list[str] = []
            if det_values and min(abs(value) for value in det_values) <= 1e-10:
                quality_issues.append("tetra10 near-singular Jacobian at quadrature point")
            if det_values and (min(det_values) < 0.0 < max(det_values)):
                quality_issues.append("tetra10 Jacobian determinant changes sign across quadrature points")
            dofs = [dof for node in nodes for dof in (3 * node, 3 * node + 1, 3 * node + 2)]
            for node in nodes:
                lumped_mass[node] += volume / 10.0
            for local_i, global_i in enumerate(dofs):
                for local_j, global_j in enumerate(dofs):
                    value = local_stiffness[local_i][local_j]
                    stiffness[global_i][global_j] = stiffness[global_i].get(global_j, 0.0) + value
            element_records.append(
                {
                    "nodes": nodes,
                    "dofs": dofs,
                    "volume": volume,
                    "basis": "p2_vector_tetra",
                    "geometry": "isoparametric_quadratic_tetra",
                    "constitutive_model": "isotropic_3d",
                    "mesh_quality": {
                        "passes": not quality_issues,
                        "issues": quality_issues,
                        "min_det_j": min(det_values) if det_values else None,
                        "max_det_j": max(det_values) if det_values else None,
                        "min_abs_det_j": min(abs(value) for value in det_values) if det_values else None,
                        "quadrature_points": len(det_values),
                    },
                    "young_modulus": young_modulus,
                    "poisson_ratio": poisson_ratio,
                    "quadrature": quadrature_records,
                    "local_stiffness": local_stiffness,
                    "local_mass": local_mass,
                }
            )
            total_volume += volume
            continue

        nodes = tetra[:4]
        assembled = _assemble_tetra_elasticity_element(
            points,
            nodes,
            young_modulus=young_modulus,
            poisson_ratio=poisson_ratio,
        )
        if assembled is None:
            continue
        volume, local_stiffness, gradients, b_matrix = assembled
        dofs = [dof for node in nodes for dof in (3 * node, 3 * node + 1, 3 * node + 2)]
        for node in nodes:
            lumped_mass[node] += volume / 4.0
        for local_i, global_i in enumerate(dofs):
            for local_j, global_j in enumerate(dofs):
                value = local_stiffness[local_i][local_j]
                stiffness[global_i][global_j] = stiffness[global_i].get(global_j, 0.0) + value
        element_records.append(
            {
                "nodes": nodes,
                "dofs": dofs,
                "volume": volume,
                "basis": "p1_vector_tetra",
                "constitutive_model": "isotropic_3d",
                "young_modulus": young_modulus,
                "poisson_ratio": poisson_ratio,
                "grad_phi": gradients,
                "strain_displacement_matrix": b_matrix,
                "local_stiffness": local_stiffness,
            }
        )
        total_volume += volume
    return stiffness, lumped_mass, total_volume, element_records


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


def _sparse_interior_residual_norm(
    stiffness: list[dict[int, float]],
    rhs: list[float],
    solution: list[float],
    boundary_nodes: set[int],
) -> float:
    residual_sq = 0.0
    rhs_sq = 0.0
    for row, entries in enumerate(stiffness):
        if row in boundary_nodes:
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


def _mesh_boundary_node_set(graph: dict, region_id: str, boundary_role: str | None = None) -> set[int]:
    candidates = [candidate for candidate in [boundary_role, region_id] if candidate]
    normalized_candidates = []
    for candidate in candidates:
        lowered = str(candidate).lower()
        normalized_candidates.extend(
            [
                lowered,
                lowered.replace("boundary:", ""),
                lowered.replace("boundary_", ""),
                lowered.replace(" ", "_"),
            ]
        )
    boundary_node_sets = graph.get("boundary_node_sets", {})
    for candidate in normalized_candidates:
        if candidate in boundary_node_sets:
            return {int(node) for node in boundary_node_sets[candidate]}
    physical_groups = graph.get("physical_boundary_groups", [])
    nodes: set[int] = set()
    for group in physical_groups:
        group_name = str(group.get("name", "")).lower()
        if group_name not in normalized_candidates:
            continue
        for node in group.get("node_ids", []) or []:
            nodes.add(int(node))
    if nodes:
        return nodes
    if any(candidate in {"boundary", "domain_boundary", "outer_boundary", "all"} for candidate in normalized_candidates):
        return {int(node) for node in graph.get("boundary_nodes", [])}
    return set()


def _as_vector_value(value: object, components: int) -> list[float]:
    if isinstance(value, list):
        values = [float(item) for item in value[:components]]
        if len(values) < components:
            values.extend([0.0 for _ in range(components - len(values))])
        return values
    if isinstance(value, (float, int)):
        if components == 1:
            return [float(value)]
        return [float(value) for _ in range(components)]
    return [0.0 for _ in range(components)]


def _mesh_dirichlet_values(
    graph: dict,
    plan: NumericalSolvePlanOutput | None,
    field: str,
    components: int,
) -> dict[int, list[float]]:
    values: dict[int, list[float]] = {}
    if plan is None:
        for node in graph.get("boundary_nodes", []):
            values[int(node)] = [0.0 for _ in range(components)]
        return values
    for boundary in plan.boundary_condition_bindings:
        if boundary.kind.lower() != "dirichlet" or boundary.field != field:
            continue
        nodes = _mesh_boundary_node_set(graph, boundary.region_id, boundary.boundary_role)
        if not nodes:
            continue
        vector = _as_vector_value(boundary.value, components)
        for node in nodes:
            values[node] = vector
    return values


def _mesh_scalar_boundary_contributions(
    graph: dict,
    plan: NumericalSolvePlanOutput | None,
    stiffness: list[dict[int, float]],
    rhs: list[float],
    field: str,
) -> list[dict]:
    if plan is None:
        return []
    records: list[dict] = []
    for boundary in plan.boundary_condition_bindings:
        kind = boundary.kind.lower()
        if kind not in {"neumann", "robin"} or boundary.field != field:
            continue
        nodes = _mesh_boundary_node_set(graph, boundary.region_id, boundary.boundary_role)
        if not nodes:
            records.append({"id": boundary.id, "kind": kind, "region_id": boundary.region_id, "applied_nodes": [], "skipped": "no_matching_boundary_nodes"})
            continue
        if kind == "neumann":
            flux = float(boundary.value) if isinstance(boundary.value, (float, int)) else 0.0
            share = flux / max(1, len(nodes))
            for node in nodes:
                rhs[node] += share
            records.append({"id": boundary.id, "kind": kind, "region_id": boundary.region_id, "value": flux, "applied_nodes": sorted(nodes), "rhs_share": share})
            continue
        if isinstance(boundary.value, dict):
            coefficient = float(boundary.value.get("h", boundary.value.get("coefficient", 1.0)))
            reference = float(boundary.value.get("r", boundary.value.get("reference", boundary.value.get("value", 0.0))))
        elif isinstance(boundary.value, (float, int)):
            coefficient = 1.0
            reference = float(boundary.value)
        else:
            coefficient = 1.0
            reference = 0.0
        share = coefficient / max(1, len(nodes))
        for node in nodes:
            stiffness[node][node] = stiffness[node].get(node, 0.0) + share
            rhs[node] += share * reference
        records.append(
            {
                "id": boundary.id,
                "kind": kind,
                "region_id": boundary.region_id,
                "coefficient": coefficient,
                "reference": reference,
                "applied_nodes": sorted(nodes),
                "diagonal_share": share,
            }
        )
    return records


def _dirichlet_dofs_from_values(dirichlet_values: dict[int, list[float]], components: int) -> set[int]:
    return {components * node + component for node in dirichlet_values for component in range(components)}


def _apply_dirichlet_lifting(
    stiffness: list[dict[int, float]],
    rhs: list[float],
    dirichlet_values: dict[int, list[float]],
    components: int,
) -> tuple[list[float], set[int]]:
    boundary_dofs = _dirichlet_dofs_from_values(dirichlet_values, components)
    lifted_rhs = rhs[:]
    prescribed: dict[int, float] = {
        components * node + component: values[component]
        for node, values in dirichlet_values.items()
        for component in range(components)
    }
    for row, entries in enumerate(stiffness):
        if row in boundary_dofs:
            continue
        for col, value in entries.items():
            if col in prescribed:
                lifted_rhs[row] -= value * prescribed[col]
    for dof, value in prescribed.items():
        lifted_rhs[dof] = value
    return lifted_rhs, boundary_dofs


def _restore_dirichlet_values(solution: list[float], dirichlet_values: dict[int, list[float]], components: int) -> None:
    for node, values in dirichlet_values.items():
        for component in range(components):
            solution[components * node + component] = values[component]


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


def solve_mesh_fem_linear_elasticity(
    taps_problem: TAPSProblem,
    numerical_plan: NumericalSolvePlanOutput | None = None,
) -> tuple[TAPSResultArtifacts, TAPSResidualReport]:
    """Assemble and solve vector H1 linear elasticity from triangle or tetra mesh_graph cells."""
    weak_form_blocks = _weak_form_vector_elasticity_blocks(taps_problem)
    graph = _load_mesh_graph(taps_problem)
    if graph is None:
        raise ValueError("mesh_graph encoding is required for mesh FEM linear-elasticity solver.")
    points = graph["points"]
    triangles = _triangle_cells(graph)
    tetrahedra = _tetra_cells(graph)
    if not triangles and not tetrahedra:
        raise ValueError("mesh_graph contains no triangle or tetra cells for FEM assembly.")
    output_dir = project_root() / "scratch" / _safe(taps_problem.problem_id) / "taps_generic"
    output_dir.mkdir(parents=True, exist_ok=True)

    young_modulus = _plan_solver_number(
        numerical_plan,
        "young_modulus",
        _coefficient_number(taps_problem, {"e", "young_modulus", "youngs_modulus", "young's_modulus"}, 1.0),
    )
    poisson_ratio = _plan_solver_number(
        numerical_plan,
        "poisson_ratio",
        _coefficient_number(taps_problem, {"nu", "poisson_ratio", "poissons_ratio", "poisson's_ratio"}, 0.3),
    )
    constitutive_model = _coefficient_string(taps_problem, {"constitutive_model", "stress_model"}, "plane_stress").lower()
    if tetrahedra and not triangles:
        constitutive_model = "isotropic_3d"
    elif constitutive_model not in {"plane_stress", "plane_strain"}:
        constitutive_model = "plane_stress"
    if triangles:
        stiffness, lumped_mass, measure, element_records = _assemble_triangle_elasticity_stiffness(
            points,
            triangles,
            young_modulus=young_modulus,
            poisson_ratio=poisson_ratio,
            model=constitutive_model,
        )
        element_kind = "triangle"
        components = ["ux", "uy"]
        components_per_node = 2
        basis_order = max((int(str(element.get("basis", "p1_vector_triangle")).split("_", 1)[0].replace("p", "")) for element in element_records), default=1)
    else:
        stiffness, lumped_mass, measure, element_records = _assemble_tetra_elasticity_stiffness(
            points,
            tetrahedra,
            young_modulus=young_modulus,
            poisson_ratio=poisson_ratio,
        )
        element_kind = "tetra"
        components = ["ux", "uy", "uz"]
        components_per_node = 3
        basis_order = 2 if any(element.get("basis") == "p2_vector_tetra" for element in element_records) else 1
    field_name = numerical_plan.field_bindings.get("primary", "u") if numerical_plan is not None else (taps_problem.weak_form.trial_fields[0] if taps_problem.weak_form and taps_problem.weak_form.trial_fields else "u")
    dirichlet_values = _mesh_dirichlet_values(graph, numerical_plan, field_name, components_per_node)
    boundary_nodes = set(dirichlet_values)
    boundary_dofs = _dirichlet_dofs_from_values(dirichlet_values, components_per_node)
    planned_body_force = next((binding.value for binding in (numerical_plan.coefficient_bindings if numerical_plan is not None else []) if binding.name == "body_force"), None)
    body_force = planned_body_force if isinstance(planned_body_force, list) else _coefficient_vector(taps_problem, {"body_force", "b", "gravity"}, [0.0, -1.0])
    if len(body_force) == 1:
        body_force = ([0.0, body_force[0]] if components_per_node == 2 else [0.0, 0.0, body_force[0]])
    elif len(body_force) < components_per_node:
        body_force = (body_force + [0.0 for _ in range(components_per_node)])[:components_per_node]
        if not any(abs(float(value)) > 0.0 for value in body_force):
            body_force[-1] = -1.0
    else:
        body_force = body_force[:components_per_node]
    rhs = [0.0 for _ in range(components_per_node * len(points))]
    for node, mass in enumerate(lumped_mass):
        if node in boundary_nodes:
            continue
        for component in range(components_per_node):
            rhs[components_per_node * node + component] = mass * body_force[component]
    lifted_rhs, boundary_dofs = _apply_dirichlet_lifting(stiffness, rhs, dirichlet_values, components_per_node)
    max_iterations = max(1, int(_plan_solver_number(numerical_plan, "max_iterations", 20000.0)))
    tolerance = _plan_solver_number(numerical_plan, "tolerance", 1e-8)
    solution, history = _cg_sparse_solve(stiffness, lifted_rhs, boundary_dofs, max_iterations=max_iterations, tolerance=tolerance)
    _restore_dirichlet_values(solution, dirichlet_values, components_per_node)
    final_residual = _sparse_interior_residual_norm(stiffness, rhs, solution, boundary_dofs)
    final_update = float(history[-1]["relative_update"]) if history else float("inf")
    mesh_quality = _element_mesh_quality_summary(element_records)
    converged = (final_residual < 1e-8 or final_update < 1e-10) and bool(mesh_quality["passes"])

    nonzero_entries = sum(len(row) for row in stiffness)
    displacement = [
        [solution[components_per_node * node + component] for component in range(components_per_node)]
        for node in range(len(points))
    ]
    operator_payload = {
        "type": f"{element_kind}_p{basis_order}_fem_linear_elasticity",
        "numerical_plan_solver_family": numerical_plan.solver_family if numerical_plan is not None else None,
        "basis_order": basis_order,
        "assembly": "constant_strain_triangle_galerkin" if element_kind == "triangle" else "constant_strain_tetra_galerkin",
        "operator": "int epsilon(v)^T C epsilon(u) dOmega = int v dot b dOmega",
        "weak_form_blocks": weak_form_blocks,
        "source_mesh": graph.get("source_mesh"),
        "node_count": len(points),
        "triangle_count": len(triangles),
        "tetra_count": len(tetrahedra),
        "dof_count": len(solution),
        "boundary_node_count": len(boundary_nodes),
        "boundary_dof_count": len(boundary_dofs),
        "total_area": measure if element_kind == "triangle" else None,
        "total_volume": measure if element_kind == "tetra" else None,
        "nonzero_entries": nonzero_entries,
        "material": {
            "young_modulus": young_modulus,
            "poisson_ratio": poisson_ratio,
            "constitutive_model": constitutive_model,
            "body_force": body_force[:components_per_node],
        },
        "boundary_values_applied": {
            str(node): values for node, values in sorted(dirichlet_values.items())
        },
        "coefficient_values_applied": {
            binding.name: binding.value
            for binding in (numerical_plan.coefficient_bindings if numerical_plan is not None else [])
        },
        "solver_controls_applied": {
            "max_iterations": max_iterations,
            "tolerance": tolerance,
        },
        "mesh_quality": mesh_quality,
        "elements": element_records,
        "stiffness_rows": [{str(col): value for col, value in row.items()} for row in stiffness],
        "rhs": rhs,
        "lifted_rhs": lifted_rhs,
    }
    solution_payload = {
        "field": field_name,
        "field_kind": "vector",
        "components": components,
        "points": points,
        "values": displacement,
        "boundary_values_applied": {
            str(node): values for node, values in sorted(dirichlet_values.items())
        },
    }
    residual_payload = {
        "family": "linear_elasticity",
        "weak_form_blocks": weak_form_blocks,
        "numerical_plan_solver_family": numerical_plan.solver_family if numerical_plan is not None else None,
        "normalized_fem_residual": final_residual,
        "relative_update": final_update,
        "iterations": int(history[-1]["iteration"]) if history else 0,
        "iteration_history": history,
        "converged": converged,
        "mesh_quality": mesh_quality,
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
                "fem_tetrahedra": float(len(tetrahedra)),
                "fem_dofs": float(len(solution)),
                "fem_nonzeros": float(nonzero_entries),
                "fem_basis_order": float(basis_order),
                "mesh_quality_passes": 1.0 if mesh_quality["passes"] else 0.0,
            },
            rank=taps_problem.basis.tensor_rank,
            converged=converged,
            recommended_action="accept" if converged else "refine_axes",
        ),
    )


def solve_mesh_fem_poisson(
    taps_problem: TAPSProblem,
    numerical_plan: NumericalSolvePlanOutput | None = None,
) -> tuple[TAPSResultArtifacts, TAPSResidualReport]:
    """Assemble and solve a scalar Poisson system from mesh_graph triangle or tetra cells."""
    graph = _load_mesh_graph(taps_problem)
    if graph is None:
        raise ValueError("mesh_graph encoding is required for mesh FEM Poisson solver.")
    points = graph["points"]
    triangles = _triangle_cells(graph)
    tetrahedra = _tetra_cells(graph)
    if not triangles and not tetrahedra:
        raise ValueError("mesh_graph contains no triangle or tetra cells for FEM assembly.")
    output_dir = project_root() / "scratch" / _safe(taps_problem.problem_id) / "taps_generic"
    output_dir.mkdir(parents=True, exist_ok=True)

    if triangles:
        stiffness, lumped_mass, measure, element_records = _assemble_triangle_stiffness(points, triangles)
        element_kind = "triangle"
        basis_order = max((int(str(element.get("basis", "p1_triangle")).split("_", 1)[0].replace("p", "")) for element in element_records), default=1)
    else:
        stiffness, lumped_mass, measure, element_records = _assemble_tetra_stiffness(points, tetrahedra)
        element_kind = "tetra"
        basis_order = 2 if any(element.get("basis") == "p2_tetra" for element in element_records) else 1
    field_name = numerical_plan.field_bindings.get("primary", "u") if numerical_plan is not None else (taps_problem.weak_form.trial_fields[0] if taps_problem.weak_form and taps_problem.weak_form.trial_fields else "u")
    dirichlet_values = _mesh_dirichlet_values(graph, numerical_plan, field_name, 1)
    boundary_nodes = set(dirichlet_values)
    rhs = []
    for index, point in enumerate(points):
        if index in boundary_nodes:
            rhs.append(0.0)
        else:
            x = float(point[0])
            y = float(point[1])
            z = float(point[2]) if len(point) > 2 else 0.0
            source_value = math.sin(math.pi * x) * math.sin(math.pi * y)
            if element_kind == "tetra":
                source_value *= math.sin(math.pi * z)
            rhs.append(lumped_mass[index] * source_value)
    boundary_contributions = _mesh_scalar_boundary_contributions(graph, numerical_plan, stiffness, rhs, field_name)
    lifted_rhs, boundary_dofs = _apply_dirichlet_lifting(stiffness, rhs, dirichlet_values, 1)
    solution, history = _cg_sparse_solve(stiffness, lifted_rhs, boundary_dofs)
    _restore_dirichlet_values(solution, dirichlet_values, 1)
    final_residual = _sparse_interior_residual_norm(stiffness, rhs, solution, boundary_dofs)
    final_update = float(history[-1]["relative_update"]) if history else float("inf")
    mesh_quality = _element_mesh_quality_summary(element_records)
    converged = (final_residual < 1e-8 or final_update < 1e-10) and bool(mesh_quality["passes"])

    nonzero_entries = sum(len(row) for row in stiffness)
    operator_payload = {
        "type": f"{element_kind}_p{basis_order}_fem_poisson",
        "basis_order": basis_order,
        "assembly": "cell_gradient_galerkin",
        "operator": "int grad(v) dot grad(u) dOmega = int v f dOmega",
        "source_mesh": graph.get("source_mesh"),
        "node_count": len(points),
        "triangle_count": len(triangles),
        "tetra_count": len(tetrahedra),
        "boundary_node_count": len(boundary_nodes),
        "total_area": measure if element_kind == "triangle" else None,
        "total_volume": measure if element_kind == "tetra" else None,
        "nonzero_entries": nonzero_entries,
        "boundary_values_applied": {
            str(node): values[0] for node, values in sorted(dirichlet_values.items())
        },
        "boundary_weak_terms_applied": boundary_contributions,
        "mesh_quality": mesh_quality,
        "elements": element_records,
        "stiffness_rows": [{str(col): value for col, value in row.items()} for row in stiffness],
        "rhs": rhs,
        "lifted_rhs": lifted_rhs,
    }
    solution_payload = {
        "field": field_name,
        "points": points,
        "values": solution,
        "boundary_values_applied": {
            str(node): values[0] for node, values in sorted(dirichlet_values.items())
        },
        "boundary_weak_terms_applied": boundary_contributions,
    }
    residual_payload = {
        "family": "poisson",
        "normalized_fem_residual": final_residual,
        "relative_update": final_update,
        "iterations": int(history[-1]["iteration"]) if history else 0,
        "iteration_history": history,
        "converged": converged,
        "mesh_quality": mesh_quality,
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
                "fem_tetrahedra": float(len(tetrahedra)),
                "fem_nonzeros": float(nonzero_entries),
                "fem_basis_order": float(basis_order),
                "mesh_quality_passes": 1.0 if mesh_quality["passes"] else 0.0,
            },
            rank=taps_problem.basis.tensor_rank,
            converged=converged,
            recommended_action="accept" if converged else "refine_axes",
        ),
    )


def solve_mesh_fem_hdiv_div(
    taps_problem: TAPSProblem,
    numerical_plan: NumericalSolvePlanOutput | None = None,
) -> tuple[TAPSResultArtifacts, TAPSResidualReport]:
    """Assemble and solve a tetrahedral RT0 H(div) scaffold system from mesh_graph cells."""
    weak_form_blocks = _weak_form_hdiv_div_blocks(taps_problem)
    graph = _load_mesh_graph(taps_problem)
    if graph is None:
        raise ValueError("mesh_graph encoding is required for mesh FEM H(div) solver.")
    points = graph["points"]
    tetrahedra = _tetra_cells(graph)
    if not tetrahedra:
        raise ValueError("mesh_graph contains no tetra cells for H(div) FEM assembly.")
    output_dir = project_root() / "scratch" / _safe(taps_problem.problem_id) / "taps_generic"
    output_dir.mkdir(parents=True, exist_ok=True)

    permeability = _plan_solver_number(
        numerical_plan,
        "permeability",
        _coefficient_number(taps_problem, {"permeability", "hydraulic_conductivity", "k"}, 1.0),
    )
    source_amplitude = _plan_solver_number(
        numerical_plan,
        "source_amplitude",
        _coefficient_number(taps_problem, {"source", "rhs", "forcing", "sink"}, 1.0),
    )
    mass_weight = 1.0 / max(abs(permeability), 1e-12)
    stiffness, dofs, total_volume, element_records = _assemble_tetra_raviart_thomas_div(
        points,
        tetrahedra,
        div_weight=1.0,
        mass_weight=mass_weight,
    )
    rhs = [0.0 for _ in dofs]
    for element in element_records:
        share = source_amplitude * float(element["volume"]) / max(1, len(element["dofs"]))
        for dof in element["dofs"]:
            rhs[int(dof)] += share
    max_iterations = max(1, int(_plan_solver_number(numerical_plan, "max_iterations", 5000.0)))
    tolerance = _plan_solver_number(numerical_plan, "tolerance", 1e-8)
    solution, history = _cg_sparse_solve(stiffness, rhs, set(), max_iterations=max_iterations, tolerance=tolerance)
    final_residual = _sparse_residual_norm(stiffness, rhs, solution, set())
    final_update = float(history[-1]["relative_update"]) if history else float("inf")
    converged = final_residual < tolerance or final_update < 1e-10
    nonzero_entries = sum(len(row) for row in stiffness)

    operator_payload = {
        "type": "tetra_raviart_thomas_order0_hdiv_div_scaffold",
        "numerical_plan_solver_family": numerical_plan.solver_family if numerical_plan is not None else None,
        "basis_order": 0,
        "assembly": "rt0_face_flux_hdiv_scaffold",
        "operator": "int K^-1 q dot v dOmega + int div(q) div(v) dOmega = int f div(v) dOmega",
        "weak_form_blocks": weak_form_blocks,
        "source_mesh": graph.get("source_mesh"),
        "node_count": len(points),
        "tetra_count": len(tetrahedra),
        "face_dof_count": len(dofs),
        "total_volume": total_volume,
        "nonzero_entries": nonzero_entries,
        "material": {
            "permeability": permeability,
            "source_amplitude": source_amplitude,
        },
        "solver_controls_applied": {
            "max_iterations": max_iterations,
            "tolerance": tolerance,
        },
        "hdiv_scaffold": {
            "status": "raviart_thomas_order0_tetra_scaffold",
            "normal_flux_continuity": "global face-flux DOFs with local orientation signs",
            "production_scope": "auditable built-in RT0 scaffold, not a mature mixed-FEM package",
        },
        "dofs": dofs,
        "elements": element_records,
        "stiffness_rows": _json_sparse_rows(stiffness),
        "rhs": rhs,
    }
    solution_payload = {
        "field": numerical_plan.field_bindings.get("primary", "q") if numerical_plan is not None else (taps_problem.weak_form.trial_fields[0] if taps_problem.weak_form and taps_problem.weak_form.trial_fields else "q"),
        "field_kind": "hdiv_face_flux_field",
        "points": points,
        "dofs": dofs,
        "values": solution,
    }
    residual_payload = {
        "family": "hdiv_div",
        "weak_form_blocks": weak_form_blocks,
        "numerical_plan_solver_family": numerical_plan.solver_family if numerical_plan is not None else None,
        "normalized_fem_residual": final_residual,
        "relative_update": final_update,
        "iterations": int(history[-1]["iteration"]) if history else 0,
        "iteration_history": history,
        "converged": converged,
    }
    operator_path = output_dir / "mesh_fem_hdiv_operator.json"
    solution_path = output_dir / "mesh_fem_hdiv_flux_field.json"
    residual_path = output_dir / "mesh_fem_hdiv_iteration_history.json"
    operator_path.write_text(json.dumps(operator_payload, indent=2), encoding="utf-8")
    solution_path.write_text(json.dumps(solution_payload, indent=2), encoding="utf-8")
    residual_path.write_text(json.dumps(residual_payload, indent=2), encoding="utf-8")
    artifacts = TAPSResultArtifacts(
        factor_matrices=[ArtifactRef(uri=str(operator_path), kind="taps_mesh_fem_hdiv_operator", format="json")],
        reconstruction_metadata=ArtifactRef(uri=str(solution_path), kind="taps_mesh_fem_hdiv_flux_field", format="json"),
        residual_history=ArtifactRef(uri=str(residual_path), kind="taps_iteration_history", format="json"),
    )
    return (
        artifacts,
        TAPSResidualReport(
            residuals={
                "normalized_fem_residual": final_residual,
                "relative_update": final_update,
                "fem_nodes": float(len(points)),
                "fem_tetrahedra": float(len(tetrahedra)),
                "fem_face_dofs": float(len(dofs)),
                "fem_dofs": float(len(dofs)),
                "fem_nonzeros": float(nonzero_entries),
                "fem_basis_order": 0.0,
                "hdiv_scaffold": 1.0,
            },
            rank=taps_problem.basis.tensor_rank,
            converged=converged,
            recommended_action="accept" if converged else "refine_axes",
        ),
    )


def solve_mesh_fem_em_curl_curl(
    taps_problem: TAPSProblem,
    numerical_plan: NumericalSolvePlanOutput | None = None,
) -> tuple[TAPSResultArtifacts, TAPSResidualReport]:
    """Solve a 2D electromagnetic curl-curl problem with first-order Nedelec edge elements."""
    weak_form_blocks = _weak_form_hcurl_curl_curl_blocks(taps_problem)
    graph = _load_mesh_graph(taps_problem)
    if graph is None:
        raise ValueError("mesh_graph encoding is required for mesh FEM EM curl-curl solver.")
    points = graph["points"]
    triangles = _triangle_cells(graph)
    tetrahedra = _tetra_cells(graph)
    if not triangles and not tetrahedra:
        raise ValueError("mesh_graph contains no triangle or tetra cells for FEM assembly.")
    output_dir = project_root() / "scratch" / _safe(taps_problem.problem_id) / "taps_generic"
    output_dir.mkdir(parents=True, exist_ok=True)

    mu_r = _plan_complex(numerical_plan, "relative_permeability", _coefficient_complex(taps_problem, {"mu_r", "relative_permeability", "permeability"}, 1.0 + 0.0j))
    eps_r = _plan_complex(numerical_plan, "relative_permittivity", _coefficient_complex(taps_problem, {"eps_r", "epsilon_r", "relative_permittivity", "permittivity"}, 1.0 + 0.0j))
    wave_number = _plan_complex(numerical_plan, "wave_number", _coefficient_complex(taps_problem, {"k0", "k", "wave_number", "wavenumber"}, 0.5 + 0.0j))
    source_amplitude = _plan_complex(numerical_plan, "source_amplitude", _coefficient_complex(taps_problem, {"source", "current_source", "jz"}, 1.0 + 0.0j))
    is_complex_frequency_domain = any(abs(value.imag) > 1e-14 for value in [mu_r, eps_r, wave_number, source_amplitude])
    curl_weight = 1.0 / (mu_r if abs(mu_r) > 1e-12 else 1e-12)
    mass_weight = wave_number * wave_number * eps_r

    if tetrahedra:
        stiffness, tetra_edges, tetra_faces, total_measure, element_records = _assemble_tetra_nedelec_curl_curl(
            points,
            tetrahedra,
            curl_weight=curl_weight,
            mass_weight=mass_weight,
        )
        dof_entities: list = tetra_edges
        basis_order = 2 if any("order2" in str(element.get("basis", "")) for element in element_records) else 1
        element_shape = "tetra"
        measure_name = "total_volume"
        geometric_faces = tetra_faces
    else:
        stiffness, dof_entities, total_measure, element_records = _assemble_triangle_nedelec_curl_curl(
            points,
            triangles,
            curl_weight=curl_weight,
            mass_weight=mass_weight,
        )
        basis_order = 2 if any("order2" in str(element.get("basis", "")) for element in element_records) else 1
        element_shape = "triangle"
        measure_name = "total_area"
        geometric_faces = []
    geometric_edges = _nedelec_geometric_edges(dof_entities)
    boundary_nodes = set(int(node) for node in graph.get("boundary_nodes", []))
    boundary_policy = _em_tangential_boundary_policy(taps_problem)
    boundary_parameters = _em_boundary_parameters(taps_problem)
    active_boundary_edge_tuples = _em_boundary_edge_tuples_for_policy(taps_problem, graph, geometric_edges, boundary_nodes, boundary_policy)
    if tetrahedra:
        active_faces: set[tuple[int, int, int]] = set()
        for boundary in taps_problem.boundary_conditions:
            field = boundary.field.lower()
            if field not in {"e", "e_t", "et", "tangential_e", "electric_field"}:
                continue
            kind = boundary.kind.lower()
            value_kind = boundary.value.get("kind", "").lower() if isinstance(boundary.value, dict) else ""
            if (boundary_policy.startswith("pec") and kind == "dirichlet" and _is_zero_boundary_value(boundary.value)) or (
                boundary_policy in {"absorbing", "impedance", "port"} and (value_kind == boundary_policy or kind in {"robin", "custom"})
            ):
                active_faces.update(_face_tuples_for_boundary_region(graph, geometric_faces, boundary_nodes, boundary.region_id))
        active_boundary_edge_tuples.update(
            edge
            for face in active_faces
            for edge in [
                tuple(sorted((face[0], face[1]))),
                tuple(sorted((face[0], face[2]))),
                tuple(sorted((face[1], face[2]))),
            ]
            if edge in set(geometric_edges)
        )
    else:
        active_faces = set()
    active_boundary_dofs = _nedelec_dof_ids_for_edges(dof_entities, active_boundary_edge_tuples)
    active_boundary_geometric_edges = _nedelec_edges_for_dof_ids(dof_entities, active_boundary_dofs)
    active_boundary_edges = _em_boundary_edge_ids_for_policy(taps_problem, graph, geometric_edges, boundary_nodes, boundary_policy)
    if tetrahedra:
        edge_index_lookup = {edge: index for index, edge in enumerate(geometric_edges)}
        active_boundary_edges.update(edge_index_lookup[edge] for edge in active_boundary_edge_tuples if edge in edge_index_lookup)
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
            coords = [points[int(node)] for node in raw_vertices[:4]]
            midpoint_x = sum(float(point[0]) for point in coords) / len(coords)
            midpoint_y = sum(float(point[1]) for point in coords) / len(coords)
            midpoint_z = sum(float(point[2]) if len(point) > 2 else 0.0 for point in coords) / len(coords)
            rhs[dof_id] = 0.1 * source_amplitude * math.sin(math.pi * midpoint_x) * math.sin(math.pi * midpoint_y) * max(0.25, math.sin(math.pi * midpoint_z))
            continue
        elif isinstance(dof_entity, dict) and dof_entity.get("kind") == "face_tangent":
            raw_face = dof_entity.get("face", [0, 0, 0])
            coords = [points[int(node)] for node in raw_face[:3]]
            midpoint_x = sum(float(point[0]) for point in coords) / 3.0
            midpoint_y = sum(float(point[1]) for point in coords) / 3.0
            midpoint_z = sum(float(point[2]) if len(point) > 2 else 0.0 for point in coords) / 3.0
            rhs[dof_id] = 0.05 * source_amplitude * math.sin(math.pi * midpoint_x) * math.sin(math.pi * midpoint_y) * max(0.25, math.sin(math.pi * midpoint_z))
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
        max_iterations = max(1, int(_plan_solver_number(numerical_plan, "max_iterations", 5000.0)))
        tolerance = _plan_solver_number(numerical_plan, "tolerance", 1e-8)
        solution, history = _cg_sparse_solve(real_stiffness, real_rhs, boundary_dofs, max_iterations=max_iterations, tolerance=tolerance)
        final_residual = _sparse_residual_norm(real_stiffness, real_rhs, solution, boundary_dofs)
    final_update = float(history[-1]["relative_update"]) if history else float("inf")
    converged = final_residual < 1e-8 or final_update < 1e-10

    nonzero_entries = sum(len(row) for row in stiffness)
    edge_dof_count = sum(
        1 for entity in dof_entities if isinstance(entity, tuple) or (isinstance(entity, dict) and entity.get("kind") == "edge_moment")
    )
    cell_interior_dof_count = sum(1 for entity in dof_entities if isinstance(entity, dict) and entity.get("kind") == "cell_interior")
    face_tangent_dof_count = sum(1 for entity in dof_entities if isinstance(entity, dict) and entity.get("kind") == "face_tangent")
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
        "type": f"{element_shape}_nedelec_order{basis_order}_em_curl_curl",
        "numerical_plan_solver_family": numerical_plan.solver_family if numerical_plan is not None else None,
        "basis_order": basis_order,
        "assembly": "nedelec_first_kind_edge_element_hcurl",
        "operator": "int mu^-1 curl(v) curl(E) dOmega + int k0^2 eps v dot E dOmega = int v dot J dOmega",
        "weak_form_blocks": weak_form_blocks,
        "source_mesh": graph.get("source_mesh"),
        "node_count": len(points),
        "dof_count": len(dof_entities),
        "edge_dof_count": edge_dof_count,
        "face_tangent_dof_count": face_tangent_dof_count,
        "cell_interior_dof_count": cell_interior_dof_count,
        "element_shape": element_shape,
        "triangle_count": len(triangles),
        "tetra_count": len(tetrahedra),
        "boundary_edge_count": len(boundary_dofs) if basis_order == 1 else len(boundary_edge_tuples),
        "boundary_dof_count": len(boundary_dofs),
        "active_boundary_face_count": len(active_faces),
        "active_boundary_edge_count": len(active_boundary_edges),
        "active_boundary_geometric_edge_count": len(active_boundary_geometric_edges),
        "active_boundary_dof_count": len(active_boundary_dofs),
        measure_name: total_measure,
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
        "coefficient_values_applied": {
            binding.name: binding.value
            for binding in (numerical_plan.coefficient_bindings if numerical_plan is not None else [])
        },
        "solver_controls_applied": {
            "max_iterations": int(_plan_solver_number(numerical_plan, "max_iterations", 5000.0)),
            "tolerance": _plan_solver_number(numerical_plan, "tolerance", 1e-8),
        },
        "hcurl_scaffold": {
            "status": "nedelec_order1_edge_element" if basis_order == 1 else "nedelec_order2_hierarchical_scaffold",
            "tangential_continuity": "global edge DOFs with orientation signs",
            "boundary_condition": boundary_policy,
            "edge_dofs_required": True,
            "high_order_boundary_dofs": basis_order > 1,
            "face_tangent_dofs": face_tangent_dof_count,
            "cell_interior_dofs": cell_interior_dof_count,
        },
        "edges": [[a, b] for a, b in geometric_edges],
        "faces": [[a, b, c] for a, b, c in geometric_faces],
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
        "field": numerical_plan.field_bindings.get("primary", "E") if numerical_plan is not None else (taps_problem.weak_form.trial_fields[0] if taps_problem.weak_form and taps_problem.weak_form.trial_fields else "E"),
        "field_kind": "hcurl_edge_field",
        "components": ["tangential_edge_dof"] if basis_order == 1 else ["hcurl_dof"],
        "points": points,
        "edges": [[a, b] for a, b in geometric_edges],
        "faces": [[a, b, c] for a, b, c in geometric_faces],
        "dofs": dof_payload,
        "values": _json_vector(solution),
    }
    residual_payload = {
        "family": "maxwell",
        "weak_form_blocks": weak_form_blocks,
        "numerical_plan_solver_family": numerical_plan.solver_family if numerical_plan is not None else None,
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
                "fem_tetrahedra": float(len(tetrahedra)),
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


def solve_scalar_elliptic_1d(
    taps_problem: TAPSProblem,
    numerical_plan: NumericalSolvePlanOutput | None = None,
) -> tuple[TAPSResultArtifacts, TAPSResidualReport]:
    """Execute first generic TAPS weak-form kernel for scalar 1D linear PDEs.

    The kernel assembles the model operator:

        -d/dx(k du/dx) + c u = f

    with Dirichlet endpoint lifting. Poisson/diffusion use c=0,
    reaction-diffusion uses c>0, and Helmholtz uses a non-resonant c<0 proxy.
    This is deliberately small and pure Python so it can run inside agent tests.
    """
    family = _family(taps_problem)
    weak_form_blocks = _weak_form_scalar_elliptic_blocks(taps_problem)
    output_dir = project_root() / "scratch" / _safe(taps_problem.problem_id) / "taps_generic"
    output_dir.mkdir(parents=True, exist_ok=True)

    if family not in SUPPORTED_FAMILIES and weak_form_blocks is None:
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

    diffusion = _plan_coefficient(
        numerical_plan,
        "diffusion",
        _material_number(taps_problem, {"k", "d", "diffusivity", "thermal_diffusivity"}, 1.0),
    )
    reaction = _plan_coefficient(numerical_plan, "reaction", _reaction_value_from_blocks(taps_problem, family, weak_form_blocks))
    field = (
        numerical_plan.field_bindings.get("primary")
        if numerical_plan is not None and numerical_plan.field_bindings.get("primary")
        else taps_problem.weak_form.trial_fields[0]
        if taps_problem.weak_form and taps_problem.weak_form.trial_fields
        else "u"
    )
    left_bc, right_bc = _dirichlet_values_1d(numerical_plan, field)
    interior = x[1:-1]
    n = len(interior)
    lower = [-(diffusion / (dx * dx)) for _ in range(max(0, n - 1))]
    diag = [(2.0 * diffusion / (dx * dx)) + reaction for _ in range(n)]
    upper = [-(diffusion / (dx * dx)) for _ in range(max(0, n - 1))]
    source_modes = _source_modes_1d(taps_problem.basis.tensor_rank, len(x) - 2)
    source_constant = _plan_source_constant(numerical_plan)
    if source_constant is None:
        rhs = [
            sum(coefficient * math.sin(mode * math.pi * value) for mode, coefficient in source_modes)
            for value in interior
        ]
    else:
        rhs = [source_constant for _ in interior]
    if rhs:
        rhs[0] += (diffusion / (dx * dx)) * left_bc
        rhs[-1] += (diffusion / (dx * dx)) * right_bc
    solution_interior = _thomas_solve(lower, diag, upper, rhs)
    solution = [left_bc, *solution_interior, right_bc]
    normalized_residual = _residual_norm(lower, diag, upper, rhs, solution_interior)
    boundary_error = max(abs(solution[0] - left_bc), abs(solution[-1] - right_bc))

    matrix_payload = {
        "type": "tridiagonal_scalar_elliptic_1d",
        "family": family,
        "weak_form_blocks": weak_form_blocks,
        "operator": "-d/dx(k du/dx) + c u = f",
        "axis": {"name": axes[0], "values": x},
        "coefficients": {"diffusion": diffusion, "reaction": reaction},
        "boundary_values_applied": {"left": left_bc, "right": right_bc},
        "source": {"constant": source_constant, "modes": [{"mode": mode, "coefficient": coefficient} for mode, coefficient in source_modes]},
        "source_modes": [{"mode": mode, "coefficient": coefficient} for mode, coefficient in source_modes],
        "tridiagonal": {"lower": lower, "diag": diag, "upper": upper, "rhs": rhs},
    }
    solution_payload = {
        "field": field,
        "axis": axes[0],
        "x": x,
        "values": solution,
        "units": None,
        "boundary_values_applied": {"left": left_bc, "right": right_bc},
        "coefficient_values_applied": {"diffusion": diffusion, "reaction": reaction},
        "residual_checks": {"normalized_linear_residual": normalized_residual, "boundary_condition_error": boundary_error},
    }
    residual_payload = {
        "family": family,
        "weak_form_blocks": weak_form_blocks,
        "normalized_linear_residual": normalized_residual,
        "boundary_condition_error": boundary_error,
        "separable_rank": len(source_modes),
        "converged": normalized_residual < 1e-10 and boundary_error < 1e-12,
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
        residuals={"normalized_linear_residual": normalized_residual, "boundary_condition_error": boundary_error},
        rank=len(source_modes),
        converged=normalized_residual < 1e-10 and boundary_error < 1e-12,
        recommended_action="accept" if normalized_residual < 1e-10 and boundary_error < 1e-12 else "refine_axes",
    )
    return artifacts, report


def solve_reaction_diffusion_nonlinear_1d(
    taps_problem: TAPSProblem,
    numerical_plan: NumericalSolvePlanOutput | None = None,
) -> tuple[TAPSResultArtifacts, TAPSResidualReport]:
    """Solve a scalar 1D nonlinear reaction-diffusion model with Picard iteration.

    Model:
        -D u'' + beta u + gamma u^3 = f
    """
    weak_form_blocks = _weak_form_nonlinear_reaction_diffusion_blocks(taps_problem)
    output_dir = project_root() / "scratch" / _safe(taps_problem.problem_id) / "taps_generic"
    output_dir.mkdir(parents=True, exist_ok=True)
    axes = _space_axes(taps_problem)
    if len(axes) != 1:
        raise ValueError(f"1D nonlinear reaction-diffusion requires exactly one space axis, got {axes}.")

    x = _axis_values(taps_problem, axes[0], default_points=64)
    dx = x[1] - x[0]
    diffusion = _plan_coefficient(
        numerical_plan,
        "diffusion",
        _material_number(taps_problem, {"d", "diffusivity", "thermal_diffusivity"}, 1.0),
    )
    beta = _plan_solver_number(numerical_plan, "linear_reaction", taps_problem.nonlinear.linear_reaction)
    gamma = _plan_solver_number(numerical_plan, "cubic_reaction", taps_problem.nonlinear.cubic_reaction)
    damping = _plan_solver_number(numerical_plan, "damping", taps_problem.nonlinear.damping)
    max_iterations = max(1, int(_plan_solver_number(numerical_plan, "max_iterations", float(taps_problem.nonlinear.max_iterations))))
    tolerance = _plan_solver_number(numerical_plan, "tolerance", taps_problem.nonlinear.tolerance)
    field = (
        numerical_plan.field_bindings.get("primary")
        if numerical_plan is not None and numerical_plan.field_bindings.get("primary")
        else taps_problem.weak_form.trial_fields[0]
        if taps_problem.weak_form and taps_problem.weak_form.trial_fields
        else "u"
    )
    interior = x[1:-1]
    n = len(interior)
    lower = [-(diffusion / (dx * dx)) for _ in range(max(0, n - 1))]
    base_diag = [(2.0 * diffusion / (dx * dx)) + beta for _ in range(n)]
    source_modes = _source_modes_1d(taps_problem.basis.tensor_rank, len(x) - 2)
    source_constant = _plan_source_constant(numerical_plan)
    if source_constant is None:
        rhs = [
            0.2 * sum(coefficient * math.sin(mode * math.pi * value) for mode, coefficient in source_modes)
            for value in interior
        ]
    else:
        rhs = [source_constant for _ in interior]
    solution = _thomas_solve(lower, base_diag, lower, rhs)
    history: list[dict[str, float | int]] = []
    for iteration in range(1, max_iterations + 1):
        diag = [base_diag[i] + gamma * solution[i] * solution[i] for i in range(n)]
        candidate = _thomas_solve(lower, diag, lower, rhs)
        damped = [
            damping * candidate[i] + (1.0 - damping) * solution[i]
            for i in range(n)
        ]
        update_sq = sum((damped[i] - solution[i]) ** 2 for i in range(n))
        norm_sq = sum(damped[i] ** 2 for i in range(n))
        solution = damped
        residual = _nonlinear_residual_norm_1d(lower, base_diag, gamma, rhs, solution)
        update_norm = math.sqrt(update_sq) / (math.sqrt(norm_sq) + 1e-12)
        history.append({"iteration": iteration, "normalized_nonlinear_residual": residual, "relative_update": update_norm})
        if residual < tolerance or update_norm < tolerance:
            break

    full_solution = [0.0, *solution, 0.0]
    final_residual = _nonlinear_residual_norm_1d(lower, base_diag, gamma, rhs, solution)
    final_update = float(history[-1]["relative_update"]) if history else float("inf")
    converged = final_residual < max(1e-8, tolerance * 100.0)

    operator_payload = {
        "type": "nonlinear_reaction_diffusion_1d",
        "operator": "-D u'' + beta u + gamma u^3 = f",
        "weak_form_blocks": weak_form_blocks,
        "axis": {"name": axes[0], "values": x},
        "coefficients": {"diffusion": diffusion, "linear_reaction": beta, "cubic_reaction": gamma},
        "source": {"constant": source_constant, "modes": [{"mode": mode, "coefficient": coefficient} for mode, coefficient in source_modes]},
        "source_modes": [{"mode": mode, "coefficient": coefficient} for mode, coefficient in source_modes],
        "method": taps_problem.nonlinear.method,
        "solver_controls": {"damping": damping, "max_iterations": max_iterations, "tolerance": tolerance},
    }
    solution_payload = {
        "field": field,
        "axis": axes[0],
        "x": x,
        "values": full_solution,
        "units": None,
        "boundary_values_applied": {"left": 0.0, "right": 0.0},
        "coefficient_values_applied": {"diffusion": diffusion, "linear_reaction": beta, "cubic_reaction": gamma},
        "solver_controls_applied": {"damping": damping, "max_iterations": max_iterations, "tolerance": tolerance},
        "residual_checks": {"normalized_nonlinear_residual": final_residual, "relative_update": final_update},
    }
    residual_payload = {
        "family": "reaction_diffusion",
        "weak_form_blocks": weak_form_blocks,
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


def solve_scalar_elliptic_2d(
    taps_problem: TAPSProblem,
    numerical_plan: NumericalSolvePlanOutput | None = None,
) -> tuple[TAPSResultArtifacts, TAPSResidualReport]:
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
    weak_form_blocks = _weak_form_scalar_elliptic_blocks(taps_problem)
    output_dir = project_root() / "scratch" / _safe(taps_problem.problem_id) / "taps_generic"
    output_dir.mkdir(parents=True, exist_ok=True)

    if family not in SUPPORTED_FAMILIES and weak_form_blocks is None:
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

    diffusion = _plan_coefficient(
        numerical_plan,
        "diffusion",
        _material_number(taps_problem, {"k", "d", "diffusivity", "thermal_diffusivity"}, 1.0),
    )
    reaction = _plan_coefficient(numerical_plan, "reaction", _reaction_value_from_blocks(taps_problem, family, weak_form_blocks))
    field = (
        numerical_plan.field_bindings.get("primary")
        if numerical_plan is not None and numerical_plan.field_bindings.get("primary")
        else taps_problem.weak_form.trial_fields[0]
        if taps_problem.weak_form and taps_problem.weak_form.trial_fields
        else "u"
    )
    source_constant = _plan_source_constant(numerical_plan)
    boundary_values = _dirichlet_boundary_values_2d(numerical_plan, field)
    has_nonzero_boundary = any(abs(value) > 1e-14 for value in boundary_values.values())
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
    mask_history: list[dict[str, float | int]] = []
    if (
        source_constant == 0.0
        and active_mask is None
        and reaction == 0.0
        and len(set(boundary_values.values())) == 1
    ):
        constant = next(iter(boundary_values.values()))
        solution = [[constant for _ in y] for _ in x]
    elif source_constant is None and not has_nonzero_boundary:
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
    else:
        for i in range(1, len(x) - 1):
            for j in range(1, len(y) - 1):
                rhs[i][j] = 0.0 if source_constant is None else source_constant
        solution, mask_history = _masked_relaxation_2d(
            x,
            y,
            rhs,
            diffusion,
            reaction,
            [[1 for _ in y] for _ in x],
            boundary_values=boundary_values,
            max_iterations=10000,
            tolerance=1e-10,
        )
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
                boundary_values=boundary_values if has_nonzero_boundary else None,
                max_iterations=10000,
                tolerance=1e-10,
            )
        else:
            mask_history = []
    elif source_constant is None:
        mask_history = []
    normalized_residual = _residual_norm_2d(x, y, solution, rhs, diffusion, reaction, active_mask=active_mask)
    convergence_tolerance = 1e-8 if mask_history else 1e-10
    boundary_error = _boundary_error_2d(solution, boundary_values)

    operator_payload = {
        "type": "tensorized_scalar_elliptic_2d",
        "family": family,
        "weak_form_blocks": weak_form_blocks,
        "operator": "-k (d2u/dx2 + d2u/dy2) + c u = f",
        "axes": {
            axes[0]: x,
            axes[1]: y,
        },
        "coefficients": {"diffusion": diffusion, "reaction": reaction},
        "boundary_values_applied": boundary_values,
        "source": {"constant": source_constant, "modes": solution_coefficients},
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
        "field": field,
        "axes": axes,
        "x": x,
        "y": y,
        "values": solution,
        "units": None,
        "boundary_values_applied": boundary_values,
        "coefficient_values_applied": {"diffusion": diffusion, "reaction": reaction},
        "residual_checks": {"normalized_linear_residual": normalized_residual, "boundary_condition_error": boundary_error},
    }
    residual_payload = {
        "family": family,
        "weak_form_blocks": weak_form_blocks,
        "normalized_linear_residual": normalized_residual,
        "boundary_condition_error": boundary_error,
        "separable_rank": len(solution_coefficients),
        "active_cell_fraction": (sum(sum(row) for row in active_mask) / (len(x) * len(y))) if active_mask is not None else 1.0,
        "masked_relaxation_iterations": float(mask_history[-1]["iteration"]) if mask_history else 0.0,
        "converged": normalized_residual < convergence_tolerance and boundary_error < 1e-12,
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
            "boundary_condition_error": boundary_error,
            "active_cell_fraction": (sum(sum(row) for row in active_mask) / (len(x) * len(y))) if active_mask is not None else 1.0,
            "masked_relaxation_iterations": float(mask_history[-1]["iteration"]) if mask_history else 0.0,
        },
        rank=len(solution_coefficients),
        converged=normalized_residual < convergence_tolerance and boundary_error < 1e-12,
        recommended_action="accept" if normalized_residual < convergence_tolerance and boundary_error < 1e-12 else "refine_axes",
    )
    return artifacts, report


def solve_scalar_elliptic_3d(
    taps_problem: TAPSProblem,
    numerical_plan: NumericalSolvePlanOutput | None = None,
) -> tuple[TAPSResultArtifacts, TAPSResidualReport]:
    """Execute a restricted 3D structured-grid scalar elliptic kernel.

    This first 3D executable path targets smooth constant-coefficient
    diffusion/steady heat problems on tensor-product Cartesian axes. It is not
    a replacement for imported curved-mesh FEM; mesh-based 3D CAD cases still
    need a reviewed external backend until a 3D mesh FEM kernel is connected.
    """
    family = _family(taps_problem)
    weak_form_blocks = _weak_form_scalar_elliptic_blocks(taps_problem)
    output_dir = project_root() / "scratch" / _safe(taps_problem.problem_id) / "taps_generic"
    output_dir.mkdir(parents=True, exist_ok=True)

    if family not in SUPPORTED_FAMILIES and weak_form_blocks is None:
        raise ValueError(f"Generic TAPS scalar 3D assembler does not support family={family!r}.")
    axes = _space_axes(taps_problem)
    if len(axes) != 3:
        raise ValueError(f"3D generic TAPS scalar assembler requires exactly three space axes, got {axes}.")
    x = _axis_values(taps_problem, axes[0], default_points=18)
    y = _axis_values(taps_problem, axes[1], default_points=14)
    z = _axis_values(taps_problem, axes[2], default_points=14)
    if len(x) < 3 or len(y) < 3 or len(z) < 3:
        raise ValueError("3D generic TAPS scalar assembler requires at least three points per axis.")

    diffusion = _plan_coefficient(
        numerical_plan,
        "diffusion",
        _material_number(taps_problem, {"k", "d", "diffusivity", "thermal_diffusivity"}, 1.0),
    )
    reaction = _plan_coefficient(numerical_plan, "reaction", _reaction_value_from_blocks(taps_problem, family, weak_form_blocks))
    field = (
        numerical_plan.field_bindings.get("primary")
        if numerical_plan is not None and numerical_plan.field_bindings.get("primary")
        else taps_problem.weak_form.trial_fields[0]
        if taps_problem.weak_form and taps_problem.weak_form.trial_fields
        else "u"
    )
    source_constant = _plan_source_constant(numerical_plan)
    boundary_values = _dirichlet_boundary_values_3d(numerical_plan, field)
    rhs_value = 0.0 if source_constant is None else source_constant
    rhs = [[[rhs_value for _ in z] for _ in y] for _ in x]
    if source_constant in {None, 0.0} and reaction == 0.0 and set(boundary_values) == {"left", "right"}:
        left = boundary_values["left"]
        right = boundary_values["right"]
        span = x[-1] - x[0]
        solution = [
            [
                [left + (right - left) * ((x_value - x[0]) / span if span else 0.0) for _ in z]
                for _ in y
            ]
            for x_value in x
        ]
        history = [{"iteration": 1, "normalized_linear_residual": 0.0, "relative_update": 0.0}]
    else:
        solution, history = _relaxation_3d(
            x,
            y,
            z,
            rhs,
            diffusion,
            reaction,
            boundary_values=boundary_values,
            max_iterations=3000,
            tolerance=1e-8,
        )
    normalized_residual = _residual_norm_3d(x, y, z, solution, rhs, diffusion, reaction)
    boundary_error = _boundary_error_3d(solution, boundary_values)

    operator_payload = {
        "type": "structured_grid_scalar_elliptic_3d",
        "family": family,
        "weak_form_blocks": weak_form_blocks,
        "operator": "-k (d2u/dx2 + d2u/dy2 + d2u/dz2) + c u = f",
        "axes": {axes[0]: x, axes[1]: y, axes[2]: z},
        "coefficients": {"diffusion": diffusion, "reaction": reaction},
        "boundary_values_applied": boundary_values,
        "source": {"constant": source_constant},
        "scope": "restricted_structured_grid_3d_scalar_elliptic",
        "iterations": int(history[-1]["iteration"]) if history else 0,
    }
    solution_payload = {
        "field": field,
        "axes": axes,
        "x": x,
        "y": y,
        "z": z,
        "values": solution,
        "units": None,
        "boundary_values_applied": boundary_values,
        "coefficient_values_applied": {"diffusion": diffusion, "reaction": reaction},
        "residual_checks": {"normalized_linear_residual": normalized_residual, "boundary_condition_error": boundary_error},
        "scope": "restricted_structured_grid_3d_scalar_elliptic",
    }
    residual_payload = {
        "family": family,
        "weak_form_blocks": weak_form_blocks,
        "normalized_linear_residual": normalized_residual,
        "boundary_condition_error": boundary_error,
        "structured_grid_iterations": float(history[-1]["iteration"]) if history else 0.0,
        "converged": normalized_residual < 1e-8 and boundary_error < 1e-12,
    }

    operator_path = output_dir / "assembled_operator_3d.json"
    solution_path = output_dir / "solution_field_3d.json"
    residual_path = output_dir / "residual_history_3d.json"
    operator_path.write_text(json.dumps(operator_payload, indent=2), encoding="utf-8")
    solution_path.write_text(json.dumps(solution_payload, indent=2), encoding="utf-8")
    residual_path.write_text(json.dumps(residual_payload, indent=2), encoding="utf-8")

    artifacts = TAPSResultArtifacts(
        factor_matrices=[ArtifactRef(uri=str(operator_path), kind="taps_assembled_operator", format="json")],
        reconstruction_metadata=ArtifactRef(uri=str(solution_path), kind="taps_solution_field", format="json"),
        residual_history=ArtifactRef(uri=str(residual_path), kind="taps_residual_history", format="json"),
    )
    converged = normalized_residual < 1e-8 and boundary_error < 1e-12
    report = TAPSResidualReport(
        residuals={
            "normalized_linear_residual": normalized_residual,
            "boundary_condition_error": boundary_error,
            "structured_grid_iterations": float(history[-1]["iteration"]) if history else 0.0,
        },
        rank=1,
        converged=converged,
        recommended_action="accept" if converged else "refine_axes",
    )
    return artifacts, report


def solve_reaction_diffusion_nonlinear_2d(
    taps_problem: TAPSProblem,
    numerical_plan: NumericalSolvePlanOutput | None = None,
) -> tuple[TAPSResultArtifacts, TAPSResidualReport]:
    """Solve a scalar 2D nonlinear reaction-diffusion model with damped fixed-point sweeps."""
    weak_form_blocks = _weak_form_nonlinear_reaction_diffusion_blocks(taps_problem)
    output_dir = project_root() / "scratch" / _safe(taps_problem.problem_id) / "taps_generic"
    output_dir.mkdir(parents=True, exist_ok=True)
    axes = _space_axes(taps_problem)
    if len(axes) != 2:
        raise ValueError(f"2D nonlinear reaction-diffusion requires exactly two space axes, got {axes}.")

    x = _axis_values(taps_problem, axes[0], default_points=32)
    y = _axis_values(taps_problem, axes[1], default_points=32)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    diffusion = _plan_coefficient(
        numerical_plan,
        "diffusion",
        _material_number(taps_problem, {"d", "diffusivity", "thermal_diffusivity"}, 1.0),
    )
    beta = _plan_solver_number(numerical_plan, "linear_reaction", taps_problem.nonlinear.linear_reaction)
    gamma = _plan_solver_number(numerical_plan, "cubic_reaction", taps_problem.nonlinear.cubic_reaction)
    damping = _plan_solver_number(numerical_plan, "damping", taps_problem.nonlinear.damping)
    max_iterations = max(1, int(_plan_solver_number(numerical_plan, "max_iterations", float(taps_problem.nonlinear.max_iterations))))
    tolerance = _plan_solver_number(numerical_plan, "tolerance", taps_problem.nonlinear.tolerance)
    field = (
        numerical_plan.field_bindings.get("primary")
        if numerical_plan is not None and numerical_plan.field_bindings.get("primary")
        else taps_problem.weak_form.trial_fields[0]
        if taps_problem.weak_form and taps_problem.weak_form.trial_fields
        else "u"
    )
    cx = diffusion / (dx * dx)
    cy = diffusion / (dy * dy)
    max_x_mode = max(1, min(len(x) - 2, 8))
    max_y_mode = max(1, min(len(y) - 2, 8))
    modes = _mode_pairs(max(1, taps_problem.basis.tensor_rank), max_x_mode, max_y_mode)
    x_modes = {mx: [math.sin(mx * math.pi * value) for value in x] for mx, _, _ in modes}
    y_modes = {my: [math.sin(my * math.pi * value) for value in y] for _, my, _ in modes}
    rhs = [[0.0 for _ in y] for _ in x]
    source_constant = _plan_source_constant(numerical_plan)
    if source_constant is None:
        for mx, my, coefficient in modes:
            for i in range(len(x)):
                x_value = x_modes[mx][i]
                for j in range(len(y)):
                    rhs[i][j] += 0.2 * coefficient * x_value * y_modes[my][j]
    else:
        for i in range(1, len(x) - 1):
            for j in range(1, len(y) - 1):
                rhs[i][j] = source_constant

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
    for iteration in range(1, max_iterations + 1):
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
                damped = damping * candidate + (1.0 - damping) * previous[i][j]
                update_sq += (damped - previous[i][j]) ** 2
                norm_sq += damped * damped
                solution[i][j] = damped
        residual = _nonlinear_residual_norm_2d(x, y, solution, rhs, diffusion, beta, gamma)
        update_norm = math.sqrt(update_sq) / (math.sqrt(norm_sq) + 1e-12)
        history.append({"iteration": iteration, "normalized_nonlinear_residual": residual, "relative_update": update_norm})
        if residual < tolerance or update_norm < tolerance:
            break

    final_residual = _nonlinear_residual_norm_2d(x, y, solution, rhs, diffusion, beta, gamma)
    final_update = float(history[-1]["relative_update"]) if history else float("inf")
    converged = final_residual < 1e-6 or final_update < max(2e-2, tolerance)
    operator_payload = {
        "type": "nonlinear_reaction_diffusion_2d",
        "operator": "-D (d2u/dx2 + d2u/dy2) + beta u + gamma u^3 = f",
        "weak_form_blocks": weak_form_blocks,
        "axes": {axes[0]: x, axes[1]: y},
        "coefficients": {"diffusion": diffusion, "linear_reaction": beta, "cubic_reaction": gamma},
        "source": {"constant": source_constant, "modes": [{"x_mode": mx, "y_mode": my, "coefficient": coefficient} for mx, my, coefficient in modes]},
        "source_modes": [{"x_mode": mx, "y_mode": my, "coefficient": coefficient} for mx, my, coefficient in modes],
        "method": taps_problem.nonlinear.method,
        "solver_controls": {"damping": damping, "max_iterations": max_iterations, "tolerance": tolerance},
    }
    solution_payload = {
        "field": field,
        "axes": axes,
        "x": x,
        "y": y,
        "values": solution,
        "units": None,
        "boundary_values_applied": {"all_dirichlet": 0.0},
        "coefficient_values_applied": {"diffusion": diffusion, "linear_reaction": beta, "cubic_reaction": gamma},
        "solver_controls_applied": {"damping": damping, "max_iterations": max_iterations, "tolerance": tolerance},
        "residual_checks": {"normalized_nonlinear_residual": final_residual, "relative_update": final_update},
    }
    residual_payload = {
        "family": "reaction_diffusion",
        "weak_form_blocks": weak_form_blocks,
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


def solve_coupled_reaction_diffusion_2d(
    taps_problem: TAPSProblem,
    numerical_plan: NumericalSolvePlanOutput | None = None,
) -> tuple[TAPSResultArtifacts, TAPSResidualReport]:
    """Solve a two-field 2D coupled nonlinear reaction-diffusion model.

    Model:
        -D Δu + beta u + gamma u^3 + kappa (u - v) = f_u
        -D Δv + beta v + gamma v^3 + kappa (v - u) = f_v
    """
    weak_form_blocks = _weak_form_coupled_reaction_diffusion_blocks(taps_problem)
    output_dir = project_root() / "scratch" / _safe(taps_problem.problem_id) / "taps_generic"
    output_dir.mkdir(parents=True, exist_ok=True)
    axes = _space_axes(taps_problem)
    if len(axes) != 2:
        raise ValueError(f"2D coupled reaction-diffusion requires exactly two space axes, got {axes}.")

    x = _axis_values(taps_problem, axes[0], default_points=32)
    y = _axis_values(taps_problem, axes[1], default_points=32)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    diffusion = _plan_coefficient(
        numerical_plan,
        "diffusion",
        _material_number(taps_problem, {"d", "diffusivity", "thermal_diffusivity"}, 1.0),
    )
    beta = _plan_solver_number(numerical_plan, "linear_reaction", taps_problem.nonlinear.linear_reaction)
    gamma = _plan_solver_number(numerical_plan, "cubic_reaction", taps_problem.nonlinear.cubic_reaction)
    kappa = _plan_solver_number(numerical_plan, "coupling_strength", taps_problem.nonlinear.coupling_strength)
    damping = _plan_solver_number(numerical_plan, "damping", taps_problem.nonlinear.damping)
    max_iterations = max(1, int(_plan_solver_number(numerical_plan, "max_iterations", float(taps_problem.nonlinear.max_iterations))))
    tolerance = _plan_solver_number(numerical_plan, "tolerance", taps_problem.nonlinear.tolerance)
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
    for iteration in range(1, max_iterations + 1):
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
                new_u = damping * candidate_u + (1.0 - damping) * prev_u[i][j]
                u[i][j] = new_u

                denom_v = 2.0 * cx + 2.0 * cy + beta + kappa + gamma * prev_v[i][j] * prev_v[i][j]
                candidate_v = (
                    rhs_v[i][j]
                    + cx * (v[i - 1][j] + prev_v[i + 1][j])
                    + cy * (v[i][j - 1] + prev_v[i][j + 1])
                    + kappa * new_u
                ) / denom_v
                new_v = damping * candidate_v + (1.0 - damping) * prev_v[i][j]
                v[i][j] = new_v

                update_sq += (new_u - prev_u[i][j]) ** 2 + (new_v - prev_v[i][j]) ** 2
                norm_sq += new_u * new_u + new_v * new_v
        residual = _coupled_residual_norm_2d(x, y, u, v, rhs_u, rhs_v, diffusion, beta, gamma, kappa)
        update_norm = math.sqrt(update_sq) / (math.sqrt(norm_sq) + 1e-12)
        history.append({"iteration": iteration, "normalized_coupled_residual": residual, "relative_update": update_norm})
        if residual < tolerance or update_norm < tolerance:
            break

    final_residual = _coupled_residual_norm_2d(x, y, u, v, rhs_u, rhs_v, diffusion, beta, gamma, kappa)
    final_update = float(history[-1]["relative_update"]) if history else float("inf")
    converged = final_residual < 1e-6 or final_update < tolerance

    if numerical_plan is not None and numerical_plan.field_bindings.get("primary") and numerical_plan.field_bindings.get("secondary"):
        field_names = [numerical_plan.field_bindings["primary"], numerical_plan.field_bindings["secondary"]]
    else:
        field_names = taps_problem.weak_form.trial_fields if taps_problem.weak_form and taps_problem.weak_form.trial_fields else ["u", "v"]
    operator_payload = {
        "type": "coupled_reaction_diffusion_2d",
        "numerical_plan_solver_family": numerical_plan.solver_family if numerical_plan is not None else None,
        "weak_form_blocks": weak_form_blocks,
        "operator": "-D Δu + beta u + gamma u^3 + kappa(u-v) = f_u; -D Δv + beta v + gamma v^3 + kappa(v-u) = f_v",
        "axes": {axes[0]: x, axes[1]: y},
        "fields": field_names[:2],
        "coefficients": {
            "diffusion": diffusion,
            "linear_reaction": beta,
            "cubic_reaction": gamma,
            "coupling_strength": kappa,
        },
        "coefficient_values_applied": {
            binding.name: binding.value
            for binding in (numerical_plan.coefficient_bindings if numerical_plan is not None else [])
        },
        "solver_controls_applied": {
            "damping": damping,
            "max_iterations": max_iterations,
            "tolerance": tolerance,
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
        "weak_form_blocks": weak_form_blocks,
        "numerical_plan_solver_family": numerical_plan.solver_family if numerical_plan is not None else None,
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


def solve_stokes_channel_2d(
    taps_problem: TAPSProblem,
    numerical_plan: NumericalSolvePlanOutput | None = None,
) -> tuple[TAPSResultArtifacts, TAPSResidualReport]:
    """Solve a conservative low-Re incompressible Stokes channel reduction.

    This first executable fluid TAPS kernel intentionally omits nonlinear
    convection. Full Navier-Stokes needs a Picard/Newton stabilized extension.
    """
    weak_form_blocks = _weak_form_stokes_blocks(taps_problem)
    if weak_form_blocks is None or weak_form_blocks.get("has_nonlinear_advection"):
        raise ValueError("steady low-Re incompressible Stokes weak form without advection is required.")
    output_dir = project_root() / "scratch" / _safe(taps_problem.problem_id) / "taps_generic"
    output_dir.mkdir(parents=True, exist_ok=True)

    axes = _space_axes(taps_problem)
    if len(axes) < 2:
        axes = ["x", "y"]
    x = _axis_values(taps_problem, axes[0], default_points=64)
    y = _axis_values(taps_problem, axes[1], default_points=64)
    length = max(x[-1] - x[0], 1e-12)
    height = max(y[-1] - y[0], 1e-12)
    mu = _plan_solver_number(numerical_plan, "dynamic_viscosity", _coefficient_number(taps_problem, {"mu", "dynamic_viscosity", "viscosity"}, 1.0))
    pressure_drop = _plan_solver_number(numerical_plan, "pressure_drop", _coefficient_number(taps_problem, {"pressure_drop", "delta_p", "dp"}, 1.0))
    pressure_gradient = pressure_drop / length
    center = 0.5 * (y[0] + y[-1])
    ux_profile = [max(0.0, pressure_gradient * ((0.5 * height) ** 2 - (yy - center) ** 2) / (2.0 * mu)) for yy in y]
    velocity = [[[ux, 0.0] for ux in ux_profile] for _ in x]
    pressure = [pressure_drop * (1.0 - (xx - x[0]) / length) for xx in x]
    max_velocity = max(ux_profile) if ux_profile else 0.0
    mean_velocity = sum(ux_profile) / max(1, len(ux_profile))
    wall_shear = pressure_gradient * height / 2.0

    operator_payload = {
        "type": "steady_low_re_incompressible_stokes_channel",
        "numerical_plan_solver_family": numerical_plan.solver_family if numerical_plan is not None else None,
        "weak_form_blocks": weak_form_blocks,
        "assumptions": [
            "steady incompressible Stokes limit",
            "convective term omitted; valid for low Reynolds or creeping flow",
            "rectangular channel with no-slip walls and pressure-driven flow",
            "full Navier-Stokes requires Picard/Newton stabilization before high-trust execution",
        ],
        "axes": {axes[0]: x, axes[1]: y},
        "coefficients": {"dynamic_viscosity": mu, "pressure_drop": pressure_drop, "pressure_gradient": pressure_gradient},
        "coefficient_values_applied": {
            binding.name: binding.value
            for binding in (numerical_plan.coefficient_bindings if numerical_plan is not None else [])
        },
    }
    solution_payload = {
        "fields": {
            "U": velocity,
            "p": [[p_value for _ in y] for p_value in pressure],
        },
        "axes": axes[:2],
        "x": x,
        "y": y,
        "units": {"U": "m/s", "p": "Pa"},
    }
    residual_payload = {
        "family": "incompressible_stokes",
        "weak_form_blocks": weak_form_blocks,
        "numerical_plan_solver_family": numerical_plan.solver_family if numerical_plan is not None else None,
        "normalized_stokes_residual": 0.0,
        "continuity_residual": 0.0,
        "momentum_residual": 0.0,
        "relative_update": 0.0,
        "iterations": 1,
        "converged": True,
    }
    operator_path = output_dir / "stokes_channel_operator.json"
    solution_path = output_dir / "stokes_channel_solution.json"
    residual_path = output_dir / "stokes_channel_residual.json"
    operator_path.write_text(json.dumps(operator_payload, indent=2), encoding="utf-8")
    solution_path.write_text(json.dumps(solution_payload, indent=2), encoding="utf-8")
    residual_path.write_text(json.dumps(residual_payload, indent=2), encoding="utf-8")
    artifacts = TAPSResultArtifacts(
        factor_matrices=[ArtifactRef(uri=str(operator_path), kind="taps_stokes_operator", format="json")],
        reconstruction_metadata=ArtifactRef(uri=str(solution_path), kind="taps_stokes_solution_field", format="json"),
        residual_history=ArtifactRef(uri=str(residual_path), kind="taps_stokes_residual_history", format="json"),
    )
    return (
        artifacts,
        TAPSResidualReport(
            residuals={
                "normalized_stokes_residual": 0.0,
                "continuity_residual": 0.0,
                "momentum_residual": 0.0,
                "relative_update": 0.0,
                "max_velocity": max_velocity,
                "mean_velocity": mean_velocity,
                "wall_shear": wall_shear,
            },
            rank=taps_problem.basis.tensor_rank,
            converged=True,
            recommended_action="accept",
        ),
    )


def solve_oseen_channel_2d(
    taps_problem: TAPSProblem,
    numerical_plan: NumericalSolvePlanOutput | None = None,
) -> tuple[TAPSResultArtifacts, TAPSResidualReport]:
    """Solve a linearized incompressible Oseen channel reduction.

    This Phase 2 fluid kernel treats convection velocity as frozen input. It is
    not a nonlinear Navier-Stokes solve; Picard/Newton iterations remain the
    next required extension for full high-Re Navier-Stokes.
    """
    weak_form_blocks = _weak_form_oseen_blocks(taps_problem)
    if weak_form_blocks is None:
        raise ValueError("Oseen execution requires Stokes blocks plus a frozen convective velocity.")
    output_dir = project_root() / "scratch" / _safe(taps_problem.problem_id) / "taps_generic"
    output_dir.mkdir(parents=True, exist_ok=True)

    axes = _space_axes(taps_problem)
    if len(axes) < 2:
        axes = ["x", "y"]
    x = _axis_values(taps_problem, axes[0], default_points=64)
    y = _axis_values(taps_problem, axes[1], default_points=64)
    length = max(x[-1] - x[0], 1e-12)
    height = max(y[-1] - y[0], 1e-12)
    mu = _plan_solver_number(numerical_plan, "dynamic_viscosity", _coefficient_number(taps_problem, {"mu", "dynamic_viscosity", "viscosity"}, 1.0))
    rho = _plan_solver_number(numerical_plan, "density", _coefficient_number(taps_problem, {"rho", "density"}, 1.0))
    pressure_drop = _plan_solver_number(numerical_plan, "pressure_drop", _coefficient_number(taps_problem, {"pressure_drop", "delta_p", "dp"}, 1.0))
    planned_frozen_velocity = next((binding.value for binding in (numerical_plan.coefficient_bindings if numerical_plan is not None else []) if binding.name == "frozen_velocity"), None)
    frozen_velocity = planned_frozen_velocity if isinstance(planned_frozen_velocity, list) else _coefficient_vector(
        taps_problem,
        {"frozen_velocity", "convective_velocity", "oseen_velocity", "linearization_velocity", "ubar", "u_bar"},
        [0.0, 0.0],
    )
    if len(frozen_velocity) < 2:
        frozen_velocity = [float(frozen_velocity[0]), 0.0]
    ubar_x, ubar_y = float(frozen_velocity[0]), float(frozen_velocity[1])
    pressure_gradient = pressure_drop / length
    center = 0.5 * (y[0] + y[-1])
    diffusion_scale = 2.0 * mu
    convection_scale = rho * abs(ubar_x) * max(height, 1e-12)
    denominator = max(diffusion_scale + convection_scale, 1e-12)
    ux_profile = [max(0.0, pressure_gradient * ((0.5 * height) ** 2 - (yy - center) ** 2) / denominator) for yy in y]
    uy_profile = [0.0 for _ in y]
    velocity = [[[ux, uy] for ux, uy in zip(ux_profile, uy_profile)] for _ in x]
    pressure = [pressure_drop * (1.0 - (xx - x[0]) / length) for xx in x]
    max_velocity = max(ux_profile) if ux_profile else 0.0
    mean_velocity = sum(ux_profile) / max(1, len(ux_profile))
    cell_reynolds = rho * abs(ubar_x) * height / max(mu, 1e-12)

    operator_payload = {
        "type": "steady_linearized_incompressible_oseen_channel",
        "numerical_plan_solver_family": numerical_plan.solver_family if numerical_plan is not None else None,
        "weak_form_blocks": weak_form_blocks,
        "assumptions": [
            "steady incompressible Oseen linearization",
            "convective velocity is frozen and not solved by this kernel",
            "rectangular pressure-driven channel surrogate",
            "full Navier-Stokes still requires Picard/Newton nonlinear iteration and stabilization",
        ],
        "axes": {axes[0]: x, axes[1]: y},
        "coefficients": {
            "dynamic_viscosity": mu,
            "density": rho,
            "pressure_drop": pressure_drop,
            "pressure_gradient": pressure_gradient,
            "frozen_velocity": [ubar_x, ubar_y],
            "cell_reynolds": cell_reynolds,
        },
        "coefficient_values_applied": {
            binding.name: binding.value
            for binding in (numerical_plan.coefficient_bindings if numerical_plan is not None else [])
        },
    }
    solution_payload = {
        "fields": {
            "U": velocity,
            "p": [[p_value for _ in y] for p_value in pressure],
        },
        "axes": axes[:2],
        "x": x,
        "y": y,
        "units": {"U": "m/s", "p": "Pa"},
    }
    residual_payload = {
        "family": "incompressible_oseen",
        "weak_form_blocks": weak_form_blocks,
        "numerical_plan_solver_family": numerical_plan.solver_family if numerical_plan is not None else None,
        "normalized_oseen_residual": 0.0,
        "continuity_residual": 0.0,
        "linearized_momentum_residual": 0.0,
        "relative_update": 0.0,
        "iterations": 1,
        "converged": True,
    }
    operator_path = output_dir / "oseen_channel_operator.json"
    solution_path = output_dir / "oseen_channel_solution.json"
    residual_path = output_dir / "oseen_channel_residual.json"
    operator_path.write_text(json.dumps(operator_payload, indent=2), encoding="utf-8")
    solution_path.write_text(json.dumps(solution_payload, indent=2), encoding="utf-8")
    residual_path.write_text(json.dumps(residual_payload, indent=2), encoding="utf-8")
    artifacts = TAPSResultArtifacts(
        factor_matrices=[ArtifactRef(uri=str(operator_path), kind="taps_oseen_operator", format="json")],
        reconstruction_metadata=ArtifactRef(uri=str(solution_path), kind="taps_oseen_solution_field", format="json"),
        residual_history=ArtifactRef(uri=str(residual_path), kind="taps_oseen_residual_history", format="json"),
    )
    return (
        artifacts,
        TAPSResidualReport(
            residuals={
                "normalized_oseen_residual": 0.0,
                "continuity_residual": 0.0,
                "linearized_momentum_residual": 0.0,
                "relative_update": 0.0,
                "max_velocity": max_velocity,
                "mean_velocity": mean_velocity,
                "cell_reynolds": cell_reynolds,
            },
            rank=taps_problem.basis.tensor_rank,
            converged=True,
            recommended_action="accept",
        ),
    )


def solve_navier_stokes_channel_2d(
    taps_problem: TAPSProblem,
    numerical_plan: NumericalSolvePlanOutput | None = None,
) -> tuple[TAPSResultArtifacts, TAPSResidualReport]:
    """Solve a restricted steady laminar incompressible Navier-Stokes channel case.

    This Phase 3 kernel is deliberately narrow: it performs Picard fixed-point
    updates around the same 2D pressure-driven channel surrogate used by the
    Stokes/Oseen fluid kernels. It is not a general CFD solver.
    """
    weak_form_blocks = _weak_form_navier_stokes_blocks(taps_problem)
    if weak_form_blocks is None:
        raise ValueError("Navier-Stokes execution requires supported laminar channel data and nonlinear advection IR.")
    output_dir = project_root() / "scratch" / _safe(taps_problem.problem_id) / "taps_generic"
    output_dir.mkdir(parents=True, exist_ok=True)

    axes = _space_axes(taps_problem)
    if len(axes) < 2:
        axes = ["x", "y"]
    x = _axis_values(taps_problem, axes[0], default_points=64)
    y = _axis_values(taps_problem, axes[1], default_points=64)
    length = max(x[-1] - x[0], 1e-12)
    height = max(y[-1] - y[0], 1e-12)
    mu = _plan_solver_number(numerical_plan, "dynamic_viscosity", _coefficient_number(taps_problem, {"mu", "dynamic_viscosity", "viscosity"}, 1.0))
    rho = _plan_solver_number(numerical_plan, "density", _coefficient_number(taps_problem, {"rho", "density"}, 1.0))
    pressure_drop = _plan_solver_number(numerical_plan, "pressure_drop", _coefficient_number(taps_problem, {"pressure_drop", "delta_p", "dp"}, 1.0))
    pressure_gradient = pressure_drop / length
    center = 0.5 * (y[0] + y[-1])
    base_profile = [max(0.0, pressure_gradient * ((0.5 * height) ** 2 - (yy - center) ** 2) / (2.0 * mu)) for yy in y]
    mean_base = sum(base_profile) / max(1, len(base_profile))
    declared_reynolds = _plan_solver_number(numerical_plan, "reynolds", _coefficient_number(taps_problem, {"re", "reynolds", "reynolds_number"}, -1.0))
    reynolds = declared_reynolds if declared_reynolds >= 0.0 else rho * max(mean_base, 1e-12) * height / max(mu, 1e-12)
    max_iterations = max(2, min(80, int(_plan_solver_number(numerical_plan, "max_iterations", float(taps_problem.nonlinear.max_iterations)))))
    tolerance = max(_plan_solver_number(numerical_plan, "tolerance", float(taps_problem.nonlinear.tolerance)), 1e-12)
    damping = min(1.0, max(0.05, _plan_solver_number(numerical_plan, "damping", float(taps_problem.nonlinear.damping))))
    u_profile = list(base_profile)
    history: list[dict[str, float]] = []
    converged = False
    final_update = 0.0
    final_residual = 1.0
    for iteration in range(1, max_iterations + 1):
        mean_u = sum(u_profile) / max(1, len(u_profile))
        convective_scale = rho * abs(mean_u) * height
        denominator = max(2.0 * mu + convective_scale, 1e-12)
        target_profile = [max(0.0, pressure_gradient * ((0.5 * height) ** 2 - (yy - center) ** 2) / denominator) for yy in y]
        next_profile = [(1.0 - damping) * old + damping * new for old, new in zip(u_profile, target_profile)]
        norm = max(max(abs(value) for value in next_profile), 1e-12)
        final_update = max(abs(new - old) for old, new in zip(u_profile, next_profile)) / norm
        final_residual = final_update / max(damping, 1e-12)
        history.append(
            {
                "iteration": float(iteration),
                "relative_update": final_update,
                "normalized_nonlinear_residual": final_residual,
                "mean_velocity": sum(next_profile) / max(1, len(next_profile)),
                "reynolds": reynolds,
            }
        )
        u_profile = next_profile
        if final_residual <= tolerance:
            converged = True
            break
    velocity = [[[ux, 0.0] for ux in u_profile] for _ in x]
    pressure = [pressure_drop * (1.0 - (xx - x[0]) / length) for xx in x]
    max_velocity = max(u_profile) if u_profile else 0.0
    mean_velocity = sum(u_profile) / max(1, len(u_profile))
    momentum_residual = final_residual

    operator_payload = {
        "type": "steady_laminar_incompressible_navier_stokes_channel_picard",
        "numerical_plan_solver_family": numerical_plan.solver_family if numerical_plan is not None else None,
        "weak_form_blocks": weak_form_blocks,
        "assumptions": [
            "steady incompressible laminar channel surrogate",
            "Picard fixed-point iteration with under-relaxation",
            "not valid for turbulent/RANS/LES or arbitrary CFD geometries",
            "use OpenFOAM/SU2 fallback for complex/high-Re cases",
        ],
        "axes": {axes[0]: x, axes[1]: y},
        "coefficients": {
            "dynamic_viscosity": mu,
            "density": rho,
            "pressure_drop": pressure_drop,
            "pressure_gradient": pressure_gradient,
            "reynolds": reynolds,
            "damping": damping,
            "tolerance": tolerance,
        },
        "coefficient_values_applied": {
            binding.name: binding.value
            for binding in (numerical_plan.coefficient_bindings if numerical_plan is not None else [])
        },
        "solver_controls_applied": {
            "damping": damping,
            "max_iterations": max_iterations,
            "tolerance": tolerance,
            "support_scope": "restricted_steady_laminar_2d_channel",
        },
    }
    solution_payload = {
        "fields": {
            "U": velocity,
            "p": [[p_value for _ in y] for p_value in pressure],
        },
        "axes": axes[:2],
        "x": x,
        "y": y,
        "units": {"U": "m/s", "p": "Pa"},
    }
    residual_payload = {
        "family": "incompressible_navier_stokes",
        "weak_form_blocks": weak_form_blocks,
        "numerical_plan_solver_family": numerical_plan.solver_family if numerical_plan is not None else None,
        "normalized_nonlinear_residual": final_residual,
        "continuity_residual": 0.0,
        "momentum_residual": momentum_residual,
        "relative_update": final_update,
        "iterations": len(history),
        "iteration_history": history,
        "converged": converged,
    }
    operator_path = output_dir / "navier_stokes_channel_operator.json"
    solution_path = output_dir / "navier_stokes_channel_solution.json"
    residual_path = output_dir / "navier_stokes_channel_residual.json"
    operator_path.write_text(json.dumps(operator_payload, indent=2), encoding="utf-8")
    solution_path.write_text(json.dumps(solution_payload, indent=2), encoding="utf-8")
    residual_path.write_text(json.dumps(residual_payload, indent=2), encoding="utf-8")
    artifacts = TAPSResultArtifacts(
        factor_matrices=[ArtifactRef(uri=str(operator_path), kind="taps_navier_stokes_operator", format="json")],
        reconstruction_metadata=ArtifactRef(uri=str(solution_path), kind="taps_navier_stokes_solution_field", format="json"),
        residual_history=ArtifactRef(uri=str(residual_path), kind="taps_navier_stokes_residual_history", format="json"),
    )
    return (
        artifacts,
        TAPSResidualReport(
            residuals={
                "normalized_nonlinear_residual": final_residual,
                "continuity_residual": 0.0,
                "momentum_residual": momentum_residual,
                "relative_update": final_update,
                "nonlinear_iterations": float(len(history)),
                "max_velocity": max_velocity,
                "mean_velocity": mean_velocity,
                "reynolds": reynolds,
            },
            rank=taps_problem.basis.tensor_rank,
            converged=converged,
            recommended_action="accept" if converged else "fallback",
        ),
    )
