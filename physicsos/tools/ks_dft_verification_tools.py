from __future__ import annotations

import json
import math
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from physicsos.config import runtime_paths
from physicsos.paths import resolve_workspace_path, to_agent_path
from physicsos.schemas.common import ArtifactRef, StrictBaseModel
from physicsos.tools.case_tools import _append_event, _case_dir


def _workspace() -> Path:
    return runtime_paths().workspace


def _verification_dir(case_id: str) -> Path:
    path = _case_dir(case_id) / "verification" / "ks_dft"
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def _read_json(path_or_uri: str | Path) -> dict[str, Any]:
    path = resolve_workspace_path(path_or_uri, workspace=_workspace(), must_be_within_workspace=False)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path_or_uri}")
    return payload


def _read_array(path_or_uri: str | Path, *, keys: tuple[str, ...] = ()) -> np.ndarray:
    path = resolve_workspace_path(path_or_uri, workspace=_workspace(), must_be_within_workspace=False)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path))
    if suffix == ".npz":
        archive = np.load(path)
        key = next((item for item in keys if item in archive), None) or next(iter(archive.files))
        return np.asarray(archive[key])
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in keys:
            if key in payload:
                return np.asarray(payload[key])
        for key in ("values", "array", "data"):
            if key in payload:
                return np.asarray(payload[key])
        raise ValueError(f"Could not find any array key {keys!r} in {path_or_uri}")
    return np.asarray(payload)


def _array_from_inline_or_ref(value: Any, ref: str | None, *, keys: tuple[str, ...]) -> np.ndarray:
    if ref:
        return _read_array(ref, keys=keys)
    if value is None:
        raise ValueError("Provide either inline values or an artifact ref.")
    return np.asarray(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


class KSDftVerificationToolOutput(StrictBaseModel):
    artifact: ArtifactRef | None = None
    artifacts: dict[str, ArtifactRef] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)
    passes: bool = False
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class CheckKSChargeConservationInput(StrictBaseModel):
    case_id: str
    expected_electrons: float
    density: Any | None = None
    density_ref: str | None = None
    quadrature_weights: Any | None = None
    quadrature_weights_ref: str | None = None
    volume_factor: float = 1.0
    tolerance_abs: float = 1e-6


def check_ks_charge_conservation(input: CheckKSChargeConservationInput) -> KSDftVerificationToolOutput:
    """Check integral n(r) dr against the expected electron count."""
    try:
        density = _array_from_inline_or_ref(input.density, input.density_ref, keys=("density", "n", "values")).astype(float).reshape(-1)
        if input.quadrature_weights_ref or input.quadrature_weights is not None:
            weights = _array_from_inline_or_ref(
                input.quadrature_weights,
                input.quadrature_weights_ref,
                keys=("weights", "quadrature_weights", "values"),
            ).astype(float).reshape(-1)
            if weights.size != density.size:
                raise ValueError("quadrature_weights must have the same flattened size as density.")
        else:
            weights = np.ones_like(density)
        integrated_charge = float(np.sum(density * weights) * input.volume_factor)
        absolute_error = abs(integrated_charge - float(input.expected_electrons))
        payload = {
            "schema_version": "physicsos.ks_dft.charge_conservation.v1",
            "expected_electrons": float(input.expected_electrons),
            "integrated_charge": integrated_charge,
            "absolute_error": absolute_error,
            "tolerance_abs": float(input.tolerance_abs),
            "passes": absolute_error <= input.tolerance_abs,
            "density_shape": list(np.shape(density)),
        }
        path = _verification_dir(input.case_id) / "charge_conservation.json"
        _write_json(path, payload)
        _append_event(_case_dir(input.case_id), "check_ks_charge_conservation", {"passes": payload["passes"]})
        return KSDftVerificationToolOutput(
            artifact=_artifact(path, "ks_dft_charge_conservation"),
            data=payload,
            passes=bool(payload["passes"]),
        )
    except Exception as exc:
        return KSDftVerificationToolOutput(errors=[str(exc)])


class CheckKSOrthonormalityInput(StrictBaseModel):
    case_id: str
    coefficients: Any | None = None
    coefficients_ref: str | None = None
    overlap: Any | None = None
    overlap_ref: str | None = None
    tolerance_frobenius: float = 1e-8
    tolerance_max_abs: float | None = None


def check_ks_orthonormality(input: CheckKSOrthonormalityInput) -> KSDftVerificationToolOutput:
    """Check C^T S C = I, or C^T C = I when no overlap matrix is supplied."""
    try:
        coefficients = _array_from_inline_or_ref(input.coefficients, input.coefficients_ref, keys=("coefficients", "C", "values"))
        c_matrix = np.asarray(coefficients, dtype=float)
        if c_matrix.ndim != 2:
            raise ValueError("coefficients must be a 2D matrix with shape [basis, states].")
        if input.overlap_ref or input.overlap is not None:
            overlap = _array_from_inline_or_ref(input.overlap, input.overlap_ref, keys=("overlap", "S", "values")).astype(float)
            if overlap.shape != (c_matrix.shape[0], c_matrix.shape[0]):
                raise ValueError("overlap must have shape [basis, basis] matching coefficients.")
            gram = c_matrix.T @ overlap @ c_matrix
        else:
            gram = c_matrix.T @ c_matrix
        identity = np.eye(gram.shape[0])
        diff = gram - identity
        frobenius_error = float(np.linalg.norm(diff, ord="fro"))
        max_abs_error = float(np.max(np.abs(diff))) if diff.size else 0.0
        max_tol = input.tolerance_max_abs if input.tolerance_max_abs is not None else input.tolerance_frobenius
        passes = frobenius_error <= input.tolerance_frobenius and max_abs_error <= max_tol
        payload = {
            "schema_version": "physicsos.ks_dft.orthonormality.v1",
            "frobenius_error": frobenius_error,
            "max_abs_error": max_abs_error,
            "tolerance_frobenius": float(input.tolerance_frobenius),
            "tolerance_max_abs": float(max_tol),
            "passes": passes,
            "gram": _json_safe(gram),
        }
        path = _verification_dir(input.case_id) / "orthonormality.json"
        _write_json(path, payload)
        _append_event(_case_dir(input.case_id), "check_ks_orthonormality", {"passes": passes})
        return KSDftVerificationToolOutput(
            artifact=_artifact(path, "ks_dft_orthonormality"),
            data=payload,
            passes=passes,
        )
    except Exception as exc:
        return KSDftVerificationToolOutput(errors=[str(exc)])


class CheckKSSCFResidualInput(StrictBaseModel):
    case_id: str
    residual_history: list[float] | None = None
    residual_ref: str | None = None
    current_density: Any | None = None
    current_density_ref: str | None = None
    previous_density: Any | None = None
    previous_density_ref: str | None = None
    tolerance: float = 1e-6
    require_finite: bool = True


def _residual_history_from_input(input: CheckKSSCFResidualInput) -> list[float]:
    if input.residual_history is not None:
        return [float(value) for value in input.residual_history]
    if input.residual_ref:
        payload = _read_json(input.residual_ref)
        for key in ("residual_history", "scf_residual_history", "scf_residuals", "residuals"):
            values = payload.get(key)
            if isinstance(values, list):
                return [float(value) for value in values]
        for key in ("scf_residual", "final_residual", "residual"):
            if key in payload:
                return [float(payload[key])]
        raise ValueError(f"Could not find SCF residual values in {input.residual_ref}")
    current = _array_from_inline_or_ref(input.current_density, input.current_density_ref, keys=("density", "current_density", "values")).astype(float)
    previous = _array_from_inline_or_ref(input.previous_density, input.previous_density_ref, keys=("density", "previous_density", "values")).astype(float)
    if current.shape != previous.shape:
        raise ValueError("current_density and previous_density must have the same shape.")
    denom = max(float(np.linalg.norm(current.reshape(-1))), 1e-300)
    return [float(np.linalg.norm((current - previous).reshape(-1)) / denom)]


def check_ks_scf_residual(input: CheckKSSCFResidualInput) -> KSDftVerificationToolOutput:
    """Check the final KS fixed-point residual or density-mixing residual."""
    try:
        history = _residual_history_from_input(input)
        if not history:
            raise ValueError("SCF residual history is empty.")
        finite = all(math.isfinite(value) for value in history)
        final_residual = float(history[-1])
        passes = final_residual <= input.tolerance and (finite or not input.require_finite)
        payload = {
            "schema_version": "physicsos.ks_dft.scf_residual.v1",
            "residual_history": history,
            "iterations": len(history),
            "final_residual": final_residual,
            "tolerance": float(input.tolerance),
            "finite": finite,
            "passes": passes,
        }
        path = _verification_dir(input.case_id) / "scf_residual.json"
        _write_json(path, payload)
        _append_event(_case_dir(input.case_id), "check_ks_scf_residual", {"passes": passes})
        return KSDftVerificationToolOutput(
            artifact=_artifact(path, "ks_dft_scf_residual"),
            data=payload,
            passes=passes,
        )
    except Exception as exc:
        return KSDftVerificationToolOutput(errors=[str(exc)])


class CheckKSPoissonResidualInput(StrictBaseModel):
    case_id: str
    residual_values: Any | None = None
    residual_ref: str | None = None
    hartree_potential: Any | None = None
    hartree_potential_ref: str | None = None
    density: Any | None = None
    density_ref: str | None = None
    quadrature_weights: Any | None = None
    quadrature_weights_ref: str | None = None
    cell_volume: float | None = None
    grid_shape: list[int] | None = None
    laplacian_potential: Any | None = None
    laplacian_potential_ref: str | None = None
    source_term: Any | None = None
    source_term_ref: str | None = None
    tolerance_l2: float = 1e-6
    tolerance_max_abs: float | None = None


def _periodic_laplacian_grid(values: np.ndarray, volume: float) -> np.ndarray:
    grid = np.asarray(values, dtype=float)
    if grid.ndim != 3:
        raise ValueError("periodic Laplacian recomputation requires a 3D grid.")
    spacing = volume ** (1.0 / 3.0) / float(grid.shape[0])
    lap = np.zeros_like(grid)
    for axis in range(3):
        lap += (np.roll(grid, 1, axis=axis) - 2.0 * grid + np.roll(grid, -1, axis=axis)) / max(spacing * spacing, 1e-300)
    return lap


def _cell_volume_from_weights(weights: np.ndarray | None, density_size: int) -> float | None:
    if weights is None:
        return None
    flat = np.asarray(weights, dtype=float).reshape(-1)
    if flat.size != density_size:
        raise ValueError("quadrature weights must have the same flattened size as density.")
    return float(np.sum(flat))


def _hamiltonian_residuals_from_artifacts(
    coefficients_ref: str,
    effective_potential_ref: str,
    weights_ref: str | None,
    grid_shape: list[int] | None,
    cell_volume: float | None,
) -> dict[str, Any]:
    coefficients_payload = _read_json(coefficients_ref)
    coefficients = np.asarray(coefficients_payload.get("coefficients"), dtype=float)
    eigenvalues = np.asarray(coefficients_payload.get("eigenvalues"), dtype=float)
    if coefficients.ndim != 2:
        raise ValueError("coefficients artifact must contain a 2D coefficients array.")
    if eigenvalues.ndim != 1 or eigenvalues.size < coefficients.shape[1]:
        raise ValueError("coefficients artifact must contain one eigenvalue per state.")
    potential_payload = _read_json(effective_potential_ref)
    potential = np.asarray(potential_payload.get("effective_potential"), dtype=float)
    shape = tuple(int(value) for value in (grid_shape or potential_payload.get("grid_shape") or []))
    if len(shape) != 3:
        raise ValueError("Provide grid_shape or effective_potential.grid_shape for Hamiltonian residual recomputation.")
    potential_grid = potential.reshape(shape)
    weights = None
    if weights_ref:
        weights = _array_from_inline_or_ref(None, weights_ref, keys=("weights", "quadrature_weights", "values")).astype(float)
    volume = cell_volume if cell_volume is not None else _cell_volume_from_weights(weights, potential_grid.size)
    if volume is None:
        raise ValueError("Provide cell_volume or quadrature_weights_ref for Hamiltonian residual recomputation.")
    weight = float(volume) / float(potential_grid.size)
    residuals: list[float] = []
    rayleigh_values: list[float] = []
    for state_index in range(coefficients.shape[1]):
        psi = coefficients[:, state_index].reshape(shape)
        hpsi = -0.5 * _periodic_laplacian_grid(psi, float(volume)) + potential_grid * psi
        eps = float(eigenvalues[state_index])
        residual = hpsi.reshape(-1) - eps * coefficients[:, state_index]
        residuals.append(float(math.sqrt(float(np.sum(residual * residual) * weight))))
        rayleigh_values.append(float(np.sum(psi * hpsi) * weight))
    return {
        "state_residuals_l2": residuals,
        "max_residual_l2": max(residuals) if residuals else math.inf,
        "rayleigh_values": rayleigh_values,
        "cell_volume": float(volume),
        "grid_shape": list(shape),
    }


def _energy_consistency_from_artifacts(
    coefficients_ref: str,
    effective_potential_ref: str,
    density_ref: str,
    hartree_potential_ref: str,
    weights_ref: str,
    energy_terms: dict[str, Any],
    grid_shape: list[int] | None,
    cell_volume: float | None,
) -> dict[str, Any]:
    coefficients_payload = _read_json(coefficients_ref)
    eigenvalues = np.asarray(coefficients_payload.get("eigenvalues"), dtype=float)
    occupations = np.asarray(coefficients_payload.get("occupations"), dtype=float)
    if occupations.ndim != 1 or eigenvalues.ndim != 1 or occupations.size > eigenvalues.size:
        raise ValueError("coefficients artifact must contain compatible occupations and eigenvalues.")
    density = _array_from_inline_or_ref(None, density_ref, keys=("density", "n", "values")).astype(float)
    effective_payload = _read_json(effective_potential_ref)
    effective_potential = np.asarray(effective_payload.get("effective_potential"), dtype=float)
    hartree_potential = _array_from_inline_or_ref(None, hartree_potential_ref, keys=("hartree_potential", "potential", "values")).astype(float)
    weights = _array_from_inline_or_ref(None, weights_ref, keys=("weights", "quadrature_weights", "values")).astype(float).reshape(-1)
    shape = tuple(int(value) for value in (grid_shape or effective_payload.get("grid_shape") or []))
    if len(shape) != 3:
        raise ValueError("Provide grid_shape or effective_potential.grid_shape for energy consistency verification.")
    density_flat = density.reshape(-1)
    effective_flat = effective_potential.reshape(-1)
    hartree_flat = hartree_potential.reshape(-1)
    if weights.size != density_flat.size:
        raise ValueError("weights must match density size.")
    volume = cell_volume if cell_volume is not None else float(np.sum(weights))
    band_energy = float(np.sum(occupations * eigenvalues[: occupations.size]))
    total_reported = float(energy_terms["total"])
    local_pseudopotential_energy = float(energy_terms["local_pseudopotential"])
    hartree_energy = float(energy_terms["hartree_fft_neutral_background"])
    xc_energy = float(
        energy_terms.get(
            "xc_total",
            sum(float(value) for key, value in energy_terms.items() if key.startswith("lda_") and key not in {"local_pseudopotential"}),
        )
    )
    v_eff_integral = float(np.sum(density_flat * effective_flat * weights))
    v_h_integral = float(np.sum((density_flat - float(np.mean(density_flat))) * hartree_flat * weights))
    reconstructed_total = band_energy - v_eff_integral + local_pseudopotential_energy + hartree_energy + xc_energy
    return {
        "band_energy": band_energy,
        "v_eff_integral": v_eff_integral,
        "hartree_potential_integral": v_h_integral,
        "local_pseudopotential_energy": local_pseudopotential_energy,
        "hartree_energy": hartree_energy,
        "xc_energy": xc_energy,
        "reconstructed_total": reconstructed_total,
        "reported_total": total_reported,
        "absolute_error": abs(reconstructed_total - total_reported),
        "cell_volume": float(volume),
    }


def check_ks_poisson_residual(input: CheckKSPoissonResidualInput) -> KSDftVerificationToolOutput:
    """Check the discrete Poisson residual for the Hartree potential."""
    try:
        source = "provided_residual"
        cell_volume = input.cell_volume
        if input.residual_ref or input.residual_values is not None:
            residual = _array_from_inline_or_ref(input.residual_values, input.residual_ref, keys=("poisson_residual", "residual", "values")).astype(float)
        elif input.hartree_potential_ref or input.hartree_potential is not None:
            potential = _array_from_inline_or_ref(
                input.hartree_potential,
                input.hartree_potential_ref,
                keys=("hartree_potential", "potential", "values"),
            ).astype(float)
            density = _array_from_inline_or_ref(input.density, input.density_ref, keys=("density", "n", "values")).astype(float)
            weights = None
            if input.quadrature_weights_ref or input.quadrature_weights is not None:
                weights = _array_from_inline_or_ref(
                    input.quadrature_weights,
                    input.quadrature_weights_ref,
                    keys=("weights", "quadrature_weights", "values"),
                ).astype(float)
            if input.grid_shape:
                shape = tuple(int(value) for value in input.grid_shape)
                potential_grid = potential.reshape(shape)
                density_grid = density.reshape(shape)
            elif potential.ndim == 3:
                potential_grid = potential
                density_grid = density.reshape(potential.shape)
            else:
                raise ValueError("Provide grid_shape when hartree_potential is stored as a flat array.")
            inferred_volume = _cell_volume_from_weights(weights, density_grid.size)
            cell_volume = cell_volume if cell_volume is not None else inferred_volume
            if cell_volume is None:
                raise ValueError("Provide cell_volume or quadrature_weights for Poisson residual recomputation.")
            rho = density_grid - float(np.mean(density_grid))
            residual = _periodic_laplacian_grid(potential_grid, float(cell_volume)) + 4.0 * math.pi * rho
            source = "recomputed_from_hartree_potential_and_density"
        else:
            lhs = _array_from_inline_or_ref(
                input.laplacian_potential,
                input.laplacian_potential_ref,
                keys=("laplacian_potential", "lhs", "values"),
            ).astype(float)
            rhs = _array_from_inline_or_ref(input.source_term, input.source_term_ref, keys=("source_term", "rhs", "values")).astype(float)
            if lhs.shape != rhs.shape:
                raise ValueError("laplacian_potential and source_term must have the same shape.")
            residual = lhs - rhs
        flat = residual.reshape(-1)
        l2_error = float(math.sqrt(float(np.mean(flat * flat)))) if flat.size else 0.0
        max_abs_error = float(np.max(np.abs(flat))) if flat.size else 0.0
        max_tol = input.tolerance_max_abs if input.tolerance_max_abs is not None else input.tolerance_l2
        passes = l2_error <= input.tolerance_l2 and max_abs_error <= max_tol
        payload = {
            "schema_version": "physicsos.ks_dft.poisson_residual.v1",
            "l2_error": l2_error,
            "max_abs_error": max_abs_error,
            "tolerance_l2": float(input.tolerance_l2),
            "tolerance_max_abs": float(max_tol),
            "passes": passes,
            "residual_shape": list(residual.shape),
            "source": source,
            "cell_volume": cell_volume,
        }
        path = _verification_dir(input.case_id) / "poisson_residual.json"
        _write_json(path, payload)
        _append_event(_case_dir(input.case_id), "check_ks_poisson_residual", {"passes": passes})
        return KSDftVerificationToolOutput(
            artifact=_artifact(path, "ks_dft_poisson_residual"),
            data=payload,
            passes=passes,
        )
    except Exception as exc:
        return KSDftVerificationToolOutput(errors=[str(exc)])


class CheckKSRankGridKpointConvergenceInput(StrictBaseModel):
    case_id: str
    rank_history: list[dict[str, Any]] = Field(default_factory=list)
    grid_history: list[dict[str, Any]] = Field(default_factory=list)
    kpoint_history: list[dict[str, Any]] = Field(default_factory=list)
    convergence_ref: str | None = None
    metric_key: str = "energy_total"
    tolerance_abs: float = 1e-5
    tolerance_rel: float = 1e-6
    require_all_axes_present: bool = False


def _history_delta(history: list[dict[str, Any]], metric_key: str, tolerance_abs: float, tolerance_rel: float) -> dict[str, Any]:
    values = [float(row[metric_key]) for row in history if isinstance(row, dict) and row.get(metric_key) is not None]
    if len(values) < 2:
        return {
            "status": "missing",
            "passes": False,
            "reason": f"Need at least two {metric_key} values.",
            "num_points": len(values),
        }
    delta_abs = abs(values[-1] - values[-2])
    scale = max(abs(values[-1]), 1.0)
    limit = max(tolerance_abs, tolerance_rel * scale)
    return {
        "status": "checked",
        "passes": delta_abs <= limit,
        "num_points": len(values),
        "last_value": values[-1],
        "previous_value": values[-2],
        "delta_abs": delta_abs,
        "limit": limit,
        "tolerance_abs": tolerance_abs,
        "tolerance_rel": tolerance_rel,
    }


def check_ks_rank_grid_kpoint_convergence(input: CheckKSRankGridKpointConvergenceInput) -> KSDftVerificationToolOutput:
    """Check final-step rank, grid, and k-point convergence deltas."""
    try:
        rank_history = list(input.rank_history)
        grid_history = list(input.grid_history)
        kpoint_history = list(input.kpoint_history)
        if input.convergence_ref:
            payload = _read_json(input.convergence_ref)
            rank_history = rank_history or list(payload.get("rank_history", []))
            grid_history = grid_history or list(payload.get("grid_history", []))
            kpoint_history = kpoint_history or list(payload.get("kpoint_history", []))
        checks = {
            "rank": _history_delta(rank_history, input.metric_key, input.tolerance_abs, input.tolerance_rel),
            "grid": _history_delta(grid_history, input.metric_key, input.tolerance_abs, input.tolerance_rel),
            "kpoint": _history_delta(kpoint_history, input.metric_key, input.tolerance_abs, input.tolerance_rel),
        }
        if input.require_all_axes_present:
            passes = all(bool(item["passes"]) for item in checks.values())
        else:
            present = [item for item in checks.values() if item["status"] == "checked"]
            passes = bool(present) and all(bool(item["passes"]) for item in present)
        report = {
            "schema_version": "physicsos.ks_dft.rank_grid_kpoint_convergence.v1",
            "metric_key": input.metric_key,
            "checks": checks,
            "require_all_axes_present": input.require_all_axes_present,
            "passes": passes,
        }
        path = _verification_dir(input.case_id) / "rank_grid_kpoint_convergence.json"
        _write_json(path, report)
        _append_event(_case_dir(input.case_id), "check_ks_rank_grid_kpoint_convergence", {"passes": passes})
        return KSDftVerificationToolOutput(
            artifact=_artifact(path, "ks_dft_rank_grid_kpoint_convergence"),
            data=report,
            passes=passes,
        )
    except Exception as exc:
        return KSDftVerificationToolOutput(errors=[str(exc)])


class CheckKSMaterialArtifactUsageInput(StrictBaseModel):
    case_id: str
    runtime_metadata_ref: str | None = None
    material_context_ref: str | None = None
    required_material_artifacts: list[str] = Field(
        default_factory=lambda: [
            "standardized_structure_ref",
            "symmetry_ref",
            "reciprocal_lattice_ref",
            "kmesh_ref",
            "irreducible_kpoints_ref",
        ]
    )
    require_artifacts_exist: bool = True


def _ref_variants(ref: str) -> set[str]:
    variants = {ref, Path(ref).name}
    try:
        resolved = resolve_workspace_path(ref, workspace=_workspace(), must_be_within_workspace=False)
        variants.add(str(resolved))
        variants.add(to_agent_path(resolved, workspace=_workspace()))
        variants.add(resolved.name)
    except Exception:
        pass
    return variants


def check_ks_material_artifact_usage(input: CheckKSMaterialArtifactUsageInput) -> KSDftVerificationToolOutput:
    """Check that a case-local KS-DFT-TAPS kernel used the expected materials artifacts."""
    try:
        metadata_ref = input.runtime_metadata_ref or f"/workspace/cases/{input.case_id}/taps/ks_dft_runtime_metadata.json"
        context_ref = input.material_context_ref or f"/workspace/cases/{input.case_id}/materials/ks_dft_material_context.json"
        metadata = _read_json(metadata_ref)
        context = _read_json(context_ref)
        refs = context.get("refs", {})
        if not isinstance(refs, dict):
            raise ValueError("material context must contain a refs object.")
        used_raw = metadata.get("materials_artifacts_used", [])
        if not isinstance(used_raw, list):
            raise ValueError("runtime metadata materials_artifacts_used must be a list.")
        used_variants: set[str] = set()
        for item in used_raw:
            if isinstance(item, str):
                used_variants.update(_ref_variants(item))
        missing_keys = [key for key in input.required_material_artifacts if not refs.get(key)]
        missing_usage: list[str] = []
        missing_files: list[str] = []
        matched: dict[str, str] = {}
        for key in input.required_material_artifacts:
            ref = refs.get(key)
            if not isinstance(ref, str) or not ref:
                continue
            variants = _ref_variants(ref)
            if variants.isdisjoint(used_variants):
                missing_usage.append(key)
            else:
                matched[key] = ref
            if input.require_artifacts_exist:
                path = resolve_workspace_path(ref, workspace=_workspace(), must_be_within_workspace=False)
                if not path.exists():
                    missing_files.append(key)
        passes = not missing_keys and not missing_usage and not missing_files
        report = {
            "schema_version": "physicsos.ks_dft.material_artifact_usage.v1",
            "runtime_metadata_ref": metadata_ref,
            "material_context_ref": context_ref,
            "required_material_artifacts": input.required_material_artifacts,
            "materials_artifacts_used": used_raw,
            "matched": matched,
            "missing_context_keys": missing_keys,
            "missing_usage": missing_usage,
            "missing_files": missing_files,
            "passes": passes,
        }
        path = _verification_dir(input.case_id) / "material_artifact_usage.json"
        _write_json(path, report)
        _append_event(_case_dir(input.case_id), "check_ks_material_artifact_usage", {"passes": passes})
        return KSDftVerificationToolOutput(
            artifact=_artifact(path, "ks_dft_material_artifact_usage"),
            data=report,
            passes=passes,
        )
    except Exception as exc:
        return KSDftVerificationToolOutput(errors=[str(exc)])


class CheckKSHamiltonianEvidenceInput(StrictBaseModel):
    case_id: str
    hamiltonian_report_ref: str | None = None
    runtime_metadata_ref: str | None = None
    coefficients_ref: str | None = None
    effective_potential_ref: str | None = None
    quadrature_weights_ref: str | None = None
    grid_shape: list[int] | None = None
    cell_volume: float | None = None
    tolerance_eigen_residual_l2: float = 1e-5
    tolerance_energy_consistency: float = 1e-6
    require_matrix_free: bool = True
    require_energy_terms: bool = True
    require_xc_provenance: bool = True
    require_pseudopotential_provenance: bool = True
    recompute_eigen_residual: bool = False
    check_energy_variational_consistency: bool = False


def check_ks_hamiltonian_evidence(input: CheckKSHamiltonianEvidenceInput) -> KSDftVerificationToolOutput:
    """Check Hamiltonian/eigensolver evidence and XC/pseudopotential provenance."""
    try:
        hamiltonian_ref = input.hamiltonian_report_ref or f"/workspace/cases/{input.case_id}/taps/ks_dft_hamiltonian_report.json"
        metadata_ref = input.runtime_metadata_ref or f"/workspace/cases/{input.case_id}/taps/ks_dft_runtime_metadata.json"
        report = _read_json(hamiltonian_ref)
        metadata = _read_json(metadata_ref)

        missing: list[str] = []
        failed: list[str] = []
        warnings: list[str] = []

        eigen_residual = report.get("eigen_residual_l2")
        if eigen_residual is None:
            missing.append("hamiltonian_report.eigen_residual_l2")
            eigen_residual_value = math.inf
        else:
            eigen_residual_value = float(eigen_residual)
            if not math.isfinite(eigen_residual_value) or eigen_residual_value > input.tolerance_eigen_residual_l2:
                failed.append("eigen_residual_l2")

        hamiltonian_action = report.get("hamiltonian_action")
        if not isinstance(hamiltonian_action, str) or "Hpsi" not in hamiltonian_action:
            missing.append("hamiltonian_report.hamiltonian_action")

        if input.require_matrix_free:
            if report.get("operator_form") != "matrix_free_hamiltonian_action" or report.get("matrix_shape") is not None:
                failed.append("matrix_free_operator_form")

        solver = report.get("solver")
        chefsi = report.get("chefsi")
        history = report.get("eigensolver_history")
        if not isinstance(solver, str) or not solver:
            missing.append("hamiltonian_report.solver")
        if not isinstance(chefsi, dict) or not chefsi:
            missing.append("hamiltonian_report.chefsi")
        if not isinstance(history, list) or not history:
            missing.append("hamiltonian_report.eigensolver_history")

        if input.require_energy_terms:
            terms = report.get("energy_terms")
            required_terms = ["kinetic", "local_pseudopotential", "hartree_fft_neutral_background", "lda_exchange", "total"]
            if not isinstance(terms, dict):
                missing.append("hamiltonian_report.energy_terms")
            else:
                missing.extend([f"hamiltonian_report.energy_terms.{key}" for key in required_terms if key not in terms])
                try:
                    xc_total = float(terms.get("xc_total", sum(float(value) for key, value in terms.items() if key.startswith("lda_"))))
                    subtotal = (
                        float(terms["kinetic"])
                        + float(terms["local_pseudopotential"])
                        + float(terms["hartree_fft_neutral_background"])
                        + xc_total
                    )
                    total = float(terms["total"])
                    if abs(subtotal - total) > max(1e-8, 1e-8 * max(abs(total), 1.0)):
                        failed.append("energy_terms_total_consistency")
                except Exception:
                    failed.append("energy_terms_numeric_consistency")

        if input.require_xc_provenance:
            xc_policy = metadata.get("xc_policy")
            if not isinstance(xc_policy, str) or not xc_policy:
                missing.append("runtime_metadata.xc_policy")
            elif xc_policy not in {"lda_exchange", "lda_x_pz81_correlation"}:
                warnings.append(f"Unexpected XC policy for current kernel: {xc_policy}")

        if input.require_pseudopotential_provenance:
            pseudo_policy = report.get("pseudopotential_policy") or metadata.get("pseudopotential_policy")
            if not isinstance(pseudo_policy, str) or not pseudo_policy:
                missing.append("pseudopotential_policy")
            context_present = bool(report.get("pseudopotential_context_present") or metadata.get("pseudopotential_context_present"))
            context = report.get("pseudopotential_context")
            if context_present and not isinstance(context, dict):
                missing.append("hamiltonian_report.pseudopotential_context")
            if context_present and isinstance(context, dict):
                if context.get("total_valence_electrons") is None:
                    missing.append("hamiltonian_report.pseudopotential_context.total_valence_electrons")
                selected = context.get("selected")
                if not isinstance(selected, dict):
                    missing.append("hamiltonian_report.pseudopotential_context.selected")

        recomputed_residual: dict[str, Any] | None = None
        energy_consistency: dict[str, Any] | None = None
        if input.recompute_eigen_residual:
            coefficients_ref = input.coefficients_ref or f"/workspace/cases/{input.case_id}/taps/ks_dft_coefficients.json"
            effective_potential_ref = input.effective_potential_ref or f"/workspace/cases/{input.case_id}/taps/ks_dft_effective_potential.json"
            weights_ref = input.quadrature_weights_ref or f"/workspace/cases/{input.case_id}/taps/ks_dft_weights.json"
            try:
                recomputed_residual = _hamiltonian_residuals_from_artifacts(
                    coefficients_ref,
                    effective_potential_ref,
                    weights_ref,
                    input.grid_shape,
                    input.cell_volume,
                )
                if recomputed_residual["max_residual_l2"] > input.tolerance_eigen_residual_l2:
                    failed.append("recomputed_eigen_residual_l2")
            except Exception as exc:
                missing.append("recomputed_hamiltonian_residual_inputs")
                warnings.append(f"Could not recompute Hamiltonian residual: {exc}")

        if input.check_energy_variational_consistency:
            coefficients_ref = input.coefficients_ref or f"/workspace/cases/{input.case_id}/taps/ks_dft_coefficients.json"
            effective_potential_ref = input.effective_potential_ref or f"/workspace/cases/{input.case_id}/taps/ks_dft_effective_potential.json"
            weights_ref = input.quadrature_weights_ref or f"/workspace/cases/{input.case_id}/taps/ks_dft_weights.json"
            density_ref = f"/workspace/cases/{input.case_id}/taps/ks_dft_density.json"
            hartree_ref = f"/workspace/cases/{input.case_id}/taps/ks_dft_hartree_potential.json"
            terms = report.get("energy_terms")
            try:
                if not isinstance(terms, dict):
                    raise ValueError("hamiltonian report has no energy_terms object.")
                energy_consistency = _energy_consistency_from_artifacts(
                    coefficients_ref,
                    effective_potential_ref,
                    density_ref,
                    hartree_ref,
                    weights_ref,
                    terms,
                    input.grid_shape,
                    input.cell_volume,
                )
                if energy_consistency["absolute_error"] > input.tolerance_energy_consistency:
                    failed.append("energy_variational_consistency")
            except Exception as exc:
                missing.append("energy_variational_consistency_inputs")
                warnings.append(f"Could not verify energy variational consistency: {exc}")

        passes = not missing and not failed
        payload = {
            "schema_version": "physicsos.ks_dft.hamiltonian_evidence.v1",
            "hamiltonian_report_ref": hamiltonian_ref,
            "runtime_metadata_ref": metadata_ref,
            "eigen_residual_l2": eigen_residual_value,
            "recomputed_eigen_residual": recomputed_residual,
            "tolerance_eigen_residual_l2": float(input.tolerance_eigen_residual_l2),
            "energy_variational_consistency": energy_consistency,
            "tolerance_energy_consistency": float(input.tolerance_energy_consistency),
            "operator_form": report.get("operator_form"),
            "solver": report.get("solver"),
            "xc_policy": metadata.get("xc_policy"),
            "pseudopotential_policy": report.get("pseudopotential_policy") or metadata.get("pseudopotential_policy"),
            "pseudopotential_context_present": bool(report.get("pseudopotential_context_present") or metadata.get("pseudopotential_context_present")),
            "missing": missing,
            "failed": failed,
            "warnings": warnings,
            "passes": passes,
        }
        path = _verification_dir(input.case_id) / "hamiltonian_evidence.json"
        _write_json(path, payload)
        _append_event(_case_dir(input.case_id), "check_ks_hamiltonian_evidence", {"passes": passes})
        return KSDftVerificationToolOutput(
            artifact=_artifact(path, "ks_dft_hamiltonian_evidence"),
            data=payload,
            passes=passes,
            warnings=warnings,
        )
    except Exception as exc:
        return KSDftVerificationToolOutput(errors=[str(exc)])


class CheckKSBandDosProvenanceInput(StrictBaseModel):
    case_id: str
    preflight_ref: str | None = None
    band_plan_ref: str | None = None
    dos_plan_ref: str | None = None
    kmesh_hamiltonian_report_ref: str | None = None
    material_context_ref: str | None = None
    require_band_plan: bool = True
    require_dos_plan: bool = True
    require_lineage_files_exist: bool = True
    require_validated_multik_hamiltonian: bool = False


def _require_ref_file(ref: Any, label: str, missing: list[str]) -> None:
    if not isinstance(ref, str) or not ref:
        missing.append(label)
        return
    path = resolve_workspace_path(ref, workspace=_workspace(), must_be_within_workspace=False)
    if not path.exists():
        missing.append(f"{label}:file_missing")


def check_ks_band_dos_provenance(input: CheckKSBandDosProvenanceInput) -> KSDftVerificationToolOutput:
    """Check that band/DOS plans preserve SCF verification and k-point provenance."""
    try:
        preflight_ref = input.preflight_ref or f"/workspace/cases/{input.case_id}/postprocess/ks_dft_band_dos_preflight.json"
        band_plan_ref = input.band_plan_ref or f"/workspace/cases/{input.case_id}/postprocess/ks_dft_band_plan.json"
        dos_plan_ref = input.dos_plan_ref or f"/workspace/cases/{input.case_id}/postprocess/ks_dft_dos_plan.json"
        kmesh_hamiltonian_report_ref = input.kmesh_hamiltonian_report_ref or f"/workspace/cases/{input.case_id}/postprocess/ks_dft_kmesh_hamiltonian_report.json"
        material_context_ref = input.material_context_ref or f"/workspace/cases/{input.case_id}/materials/ks_dft_material_context.json"
        missing: list[str] = []
        failed: list[str] = []
        preflight = _read_json(preflight_ref)
        context = _read_json(material_context_ref)
        refs = context.get("refs", {}) if isinstance(context.get("refs"), dict) else {}
        required_checks = preflight.get("required_checks", [])
        check_artifacts = preflight.get("check_artifacts", {})
        if preflight.get("accepted") is not True:
            failed.append("preflight_not_accepted")
        if not isinstance(required_checks, list) or not required_checks:
            missing.append("preflight.required_checks")
        if not isinstance(check_artifacts, dict):
            missing.append("preflight.check_artifacts")
        else:
            for check in required_checks:
                _require_ref_file(check_artifacts.get(check), f"preflight.check_artifacts.{check}", missing)

        band_plan: dict[str, Any] | None = None
        dos_plan: dict[str, Any] | None = None
        if input.require_band_plan:
            band_plan = _read_json(band_plan_ref)
            provenance = band_plan.get("provenance")
            if not isinstance(provenance, dict):
                missing.append("band_plan.provenance")
            else:
                if provenance.get("preflight_ref") != preflight_ref:
                    failed.append("band_plan.preflight_ref_mismatch")
                if provenance.get("material_context_ref") != material_context_ref:
                    failed.append("band_plan.material_context_ref_mismatch")
                source_refs = provenance.get("source_refs", {})
                if not isinstance(source_refs, dict) or not (source_refs.get("line_kpoints_ref") or source_refs.get("kpath_ref")):
                    missing.append("band_plan.source_refs.line_kpoints_or_kpath")
                if input.require_lineage_files_exist:
                    for label in ("line_kpoints_ref", "kpath_ref"):
                        ref = source_refs.get(label) if isinstance(source_refs, dict) else None
                        if ref:
                            _require_ref_file(ref, f"band_plan.source_refs.{label}", missing)
        if input.require_dos_plan:
            dos_plan = _read_json(dos_plan_ref)
            provenance = dos_plan.get("provenance")
            if not isinstance(provenance, dict):
                missing.append("dos_plan.provenance")
            else:
                if provenance.get("preflight_ref") != preflight_ref:
                    failed.append("dos_plan.preflight_ref_mismatch")
                if provenance.get("material_context_ref") != material_context_ref:
                    failed.append("dos_plan.material_context_ref_mismatch")
                source_refs = provenance.get("source_refs", {})
                if not isinstance(source_refs, dict):
                    missing.append("dos_plan.source_refs")
                else:
                    for label in ("kmesh_ref", "irreducible_kpoints_ref"):
                        ref = source_refs.get(label)
                        if not ref:
                            missing.append(f"dos_plan.source_refs.{label}")
                        elif input.require_lineage_files_exist:
                            _require_ref_file(ref, f"dos_plan.source_refs.{label}", missing)
                        expected = refs.get(label)
                        if expected and ref != expected:
                            failed.append(f"dos_plan.{label}_mismatch")

        kmesh_report: dict[str, Any] | None = None
        if input.require_validated_multik_hamiltonian:
            kmesh_report = _read_json(kmesh_hamiltonian_report_ref)
            if kmesh_report.get("status") != "validated_multik_hamiltonian":
                failed.append("validated_multik_hamiltonian_missing")
            for label, plan in (("band_plan", band_plan), ("dos_plan", dos_plan)):
                if isinstance(plan, dict) and plan.get("band_dos_mode") != "validated_multik_hamiltonian":
                    failed.append(f"{label}.validated_multik_mode_missing")
            if not isinstance(kmesh_report.get("provenance"), dict):
                missing.append("kmesh_hamiltonian_report.provenance")
            if not isinstance(kmesh_report.get("k_weights_normalized"), list):
                missing.append("kmesh_hamiltonian_report.k_weights_normalized")

        passes = not missing and not failed
        payload = {
            "schema_version": "physicsos.ks_dft.band_dos_provenance.v1",
            "preflight_ref": preflight_ref,
            "band_plan_ref": band_plan_ref if input.require_band_plan else None,
            "dos_plan_ref": dos_plan_ref if input.require_dos_plan else None,
            "kmesh_hamiltonian_report_ref": kmesh_hamiltonian_report_ref if input.require_validated_multik_hamiltonian else None,
            "material_context_ref": material_context_ref,
            "required_checks": required_checks,
            "require_validated_multik_hamiltonian": input.require_validated_multik_hamiltonian,
            "missing": missing,
            "failed": failed,
            "passes": passes,
        }
        path = _verification_dir(input.case_id) / "band_dos_provenance.json"
        _write_json(path, payload)
        _append_event(_case_dir(input.case_id), "check_ks_band_dos_provenance", {"passes": passes})
        return KSDftVerificationToolOutput(
            artifact=_artifact(path, "ks_dft_band_dos_provenance"),
            data=payload,
            passes=passes,
        )
    except Exception as exc:
        return KSDftVerificationToolOutput(errors=[str(exc)])


class CheckKSMolecularContextEvidenceInput(StrictBaseModel):
    case_id: str
    molecular_context_ref: str | None = None
    runtime_metadata_ref: str | None = None
    scaling_policy_ref: str | None = None
    boundary_evidence_ref: str | None = None
    fragment_evidence_ref: str | None = None
    locality_evidence_ref: str | None = None
    scaling_evidence_ref: str | None = None
    require_boundary_evidence: bool = True
    require_fragment_evidence_when_selected: bool = True
    require_locality_evidence_when_selected: bool = True
    require_scaling_evidence_for_large_targets: bool = True
    require_charge_spin_metadata: bool = True
    forbid_crystal_kmesh_without_vacuum_box: bool = True
    boundary_residual_tolerance: float = 1e-5
    vacuum_correction_tolerance: float = 1e-6
    direct_coulomb_residual_tolerance: float = 1e-5
    cutoff_coulomb_residual_tolerance: float = 1e-5
    grid_poisson_residual_tolerance: float = 1e-5
    grid_boundary_residual_tolerance: float = 1e-5
    finite_size_correction_tolerance: float = 1e-6
    require_formula_manifest_when_present: bool = True
    fragment_charge_tolerance: float = 1e-5
    locality_delta_tolerance: float = 1e-5
    scaling_efficiency_min: float = 0.5


def _metadata_value(metadata: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metadata:
            return metadata[key]
    return None


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_optional_json(ref: str | None, missing: list[str], label: str) -> dict[str, Any] | None:
    if not ref:
        missing.append(label)
        return None
    try:
        return _read_json(ref)
    except Exception:
        missing.append(f"{label}:file_missing_or_invalid")
        return None


def _first_numeric(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _validate_boundary_numerics(
    evidence: dict[str, Any],
    metadata: dict[str, Any],
    input: CheckKSMolecularContextEvidenceInput,
    failed: list[str],
    missing: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {"checked": True}
    residual = _first_numeric(evidence, ("residual_l2", "poisson_residual_l2", "max_residual", "max_abs_residual"))
    if residual is None:
        missing.append("boundary_evidence.residual")
    else:
        summary["residual"] = residual
        if residual > input.boundary_residual_tolerance:
            failed.append("boundary_residual_above_tolerance")
    method = str(evidence.get("method") or evidence.get("poisson_boundary") or evidence.get("boundary_type") or "")
    if "vacuum" in method.lower() or "vacuum_box" in str(evidence.get("selected", "")).lower():
        correction = evidence.get("vacuum_box_correction")
        if not isinstance(correction, dict):
            missing.append("boundary_evidence.vacuum_box_correction")
        else:
            reported = _first_numeric(correction, ("correction_energy", "energy_correction", "delta_energy"))
            uncorrected = _first_numeric(correction, ("uncorrected_energy", "raw_energy"))
            corrected = _first_numeric(correction, ("corrected_energy", "final_energy"))
            if reported is not None and uncorrected is not None and corrected is not None:
                absolute_error = abs((uncorrected + reported) - corrected)
                summary["vacuum_correction_absolute_error"] = absolute_error
                if absolute_error > input.vacuum_correction_tolerance:
                    failed.append("vacuum_correction_inconsistent")
            else:
                missing.append("boundary_evidence.vacuum_box_correction.energy_terms")
    if "multipole" in method.lower() and not evidence.get("multipole_order"):
        warnings.append("Multipole boundary evidence has no multipole_order field.")
    if "cutoff" in method.lower() and not evidence.get("cutoff_radius_angstrom"):
        warnings.append("Coulomb-cutoff boundary evidence has no cutoff_radius_angstrom field.")
    direct_coulomb = evidence.get("direct_coulomb_check")
    if isinstance(direct_coulomb, dict):
        direct_summary = _direct_coulomb_residual(direct_coulomb, input, failed, missing)
        summary["direct_coulomb_check"] = direct_summary
    cutoff_check = evidence.get("coulomb_cutoff_check")
    if isinstance(cutoff_check, dict):
        cutoff_summary = _coulomb_cutoff_residual(cutoff_check, input, failed, missing)
        summary["coulomb_cutoff_check"] = cutoff_summary
    multipole_check = evidence.get("multipole_check")
    if isinstance(multipole_check, dict):
        multipole_summary = _multipole_far_field_check(multipole_check, input, failed, missing)
        summary["multipole_check"] = multipole_summary
    grid_poisson_check = evidence.get("grid_poisson_check")
    if isinstance(grid_poisson_check, dict):
        grid_summary = _grid_poisson_check(grid_poisson_check, input, failed, missing)
        summary["grid_poisson_check"] = grid_summary
    finite_size = evidence.get("vacuum_finite_size_correction")
    if isinstance(finite_size, dict):
        finite_size_summary = _vacuum_finite_size_correction_check(finite_size, input, failed, missing)
        summary["vacuum_finite_size_correction"] = finite_size_summary
    formula_manifest = evidence.get("correction_formula_manifest")
    if isinstance(formula_manifest, dict):
        formula_summary = _correction_formula_manifest_check(formula_manifest, metadata, input, failed, missing)
        summary["correction_formula_manifest"] = formula_summary
    return summary


def _as_float_array(value: Any, label: str, missing: list[str]) -> np.ndarray | None:
    if value is None:
        missing.append(label)
        return None
    try:
        array = np.asarray(value, dtype=float)
    except Exception:
        missing.append(f"{label}:not_numeric")
        return None
    if not np.isfinite(array).all():
        missing.append(f"{label}:non_finite")
        return None
    return array


def _direct_coulomb_residual(
    check: dict[str, Any],
    input: CheckKSMolecularContextEvidenceInput,
    failed: list[str],
    missing: list[str],
) -> dict[str, Any]:
    source_points = _as_float_array(check.get("source_points_angstrom"), "boundary_evidence.direct_coulomb_check.source_points_angstrom", missing)
    probe_points = _as_float_array(check.get("probe_points_angstrom"), "boundary_evidence.direct_coulomb_check.probe_points_angstrom", missing)
    density = _as_float_array(check.get("density_values"), "boundary_evidence.direct_coulomb_check.density_values", missing)
    weights = _as_float_array(check.get("quadrature_weights"), "boundary_evidence.direct_coulomb_check.quadrature_weights", missing)
    potential = _as_float_array(check.get("hartree_potential"), "boundary_evidence.direct_coulomb_check.hartree_potential", missing)
    summary: dict[str, Any] = {"checked": True}
    if source_points is None or probe_points is None or density is None or weights is None or potential is None:
        return summary
    if source_points.ndim != 2 or source_points.shape[1] != 3:
        missing.append("boundary_evidence.direct_coulomb_check.source_points_shape")
        return summary
    if probe_points.ndim != 2 or probe_points.shape[1] != 3:
        missing.append("boundary_evidence.direct_coulomb_check.probe_points_shape")
        return summary
    if density.reshape(-1).size != source_points.shape[0] or weights.reshape(-1).size != source_points.shape[0]:
        missing.append("boundary_evidence.direct_coulomb_check.source_value_sizes")
        return summary
    if potential.reshape(-1).size != probe_points.shape[0]:
        missing.append("boundary_evidence.direct_coulomb_check.potential_size")
        return summary
    softening = float(check.get("softening_angstrom", 0.0) or 0.0)
    prefactor = float(check.get("coulomb_prefactor", 1.0) or 1.0)
    self_cutoff = float(check.get("self_cutoff_angstrom", 1e-12) or 1e-12)
    source = source_points.astype(float)
    probe = probe_points.astype(float)
    charge = density.reshape(-1).astype(float) * weights.reshape(-1).astype(float)
    recomputed: list[float] = []
    for point in probe:
        distances = np.linalg.norm(source - point[None, :], axis=1)
        mask = distances > self_cutoff
        denom = np.sqrt(distances[mask] ** 2 + softening**2)
        recomputed.append(float(prefactor * np.sum(charge[mask] / denom)))
    recomputed_array = np.asarray(recomputed, dtype=float)
    residual = recomputed_array - potential.reshape(-1).astype(float)
    max_abs = float(np.max(np.abs(residual))) if residual.size else 0.0
    l2 = float(np.linalg.norm(residual) / math.sqrt(max(residual.size, 1)))
    summary.update(
        {
            "num_source_points": int(source_points.shape[0]),
            "num_probe_points": int(probe_points.shape[0]),
            "max_abs_residual": max_abs,
            "l2_residual": l2,
            "softening_angstrom": softening,
            "coulomb_prefactor": prefactor,
        }
    )
    if max_abs > input.direct_coulomb_residual_tolerance and l2 > input.direct_coulomb_residual_tolerance:
        failed.append("direct_coulomb_residual_above_tolerance")
    return summary


def _coulomb_cutoff_residual(
    check: dict[str, Any],
    input: CheckKSMolecularContextEvidenceInput,
    failed: list[str],
    missing: list[str],
) -> dict[str, Any]:
    source_points = _as_float_array(check.get("source_points_angstrom"), "boundary_evidence.coulomb_cutoff_check.source_points_angstrom", missing)
    probe_points = _as_float_array(check.get("probe_points_angstrom"), "boundary_evidence.coulomb_cutoff_check.probe_points_angstrom", missing)
    density = _as_float_array(check.get("density_values"), "boundary_evidence.coulomb_cutoff_check.density_values", missing)
    weights = _as_float_array(check.get("quadrature_weights"), "boundary_evidence.coulomb_cutoff_check.quadrature_weights", missing)
    potential = _as_float_array(check.get("hartree_potential"), "boundary_evidence.coulomb_cutoff_check.hartree_potential", missing)
    cutoff = _first_numeric(check, ("cutoff_radius_angstrom", "cutoff_radius", "rcut_angstrom"))
    summary: dict[str, Any] = {"checked": True}
    if source_points is None or probe_points is None or density is None or weights is None or potential is None:
        return summary
    if cutoff is None or cutoff <= 0:
        missing.append("boundary_evidence.coulomb_cutoff_check.cutoff_radius_angstrom")
        return summary
    if source_points.ndim != 2 or source_points.shape[1] != 3 or probe_points.ndim != 2 or probe_points.shape[1] != 3:
        missing.append("boundary_evidence.coulomb_cutoff_check.point_shapes")
        return summary
    if density.reshape(-1).size != source_points.shape[0] or weights.reshape(-1).size != source_points.shape[0]:
        missing.append("boundary_evidence.coulomb_cutoff_check.source_value_sizes")
        return summary
    if potential.reshape(-1).size != probe_points.shape[0]:
        missing.append("boundary_evidence.coulomb_cutoff_check.potential_size")
        return summary
    prefactor = float(check.get("coulomb_prefactor", 1.0) or 1.0)
    self_cutoff = float(check.get("self_cutoff_angstrom", 1e-12) or 1e-12)
    shifted = bool(check.get("shift_to_zero_at_cutoff", False))
    charges = density.reshape(-1).astype(float) * weights.reshape(-1).astype(float)
    recomputed: list[float] = []
    for point in probe_points.astype(float):
        distances = np.linalg.norm(source_points.astype(float) - point[None, :], axis=1)
        mask = (distances > self_cutoff) & (distances <= cutoff)
        if shifted:
            values = (1.0 / distances[mask]) - (1.0 / cutoff)
        else:
            values = 1.0 / distances[mask]
        recomputed.append(float(prefactor * np.sum(charges[mask] * values)))
    residual = np.asarray(recomputed, dtype=float) - potential.reshape(-1).astype(float)
    max_abs = float(np.max(np.abs(residual))) if residual.size else 0.0
    l2 = float(np.linalg.norm(residual) / math.sqrt(max(residual.size, 1)))
    summary.update(
        {
            "cutoff_radius_angstrom": float(cutoff),
            "shift_to_zero_at_cutoff": shifted,
            "max_abs_residual": max_abs,
            "l2_residual": l2,
            "num_source_points": int(source_points.shape[0]),
            "num_probe_points": int(probe_points.shape[0]),
        }
    )
    if max_abs > input.cutoff_coulomb_residual_tolerance and l2 > input.cutoff_coulomb_residual_tolerance:
        failed.append("coulomb_cutoff_residual_above_tolerance")
    return summary


def _multipole_far_field_check(
    check: dict[str, Any],
    input: CheckKSMolecularContextEvidenceInput,
    failed: list[str],
    missing: list[str],
) -> dict[str, Any]:
    probe_radii = _as_float_array(check.get("probe_radii_angstrom"), "boundary_evidence.multipole_check.probe_radii_angstrom", missing)
    potential = _as_float_array(check.get("hartree_potential"), "boundary_evidence.multipole_check.hartree_potential", missing)
    summary: dict[str, Any] = {"checked": True}
    monopole = _first_numeric(check, ("monopole_charge", "total_charge", "q0"))
    prefactor = float(check.get("coulomb_prefactor", 1.0) or 1.0)
    if monopole is None:
        missing.append("boundary_evidence.multipole_check.monopole_charge")
        return summary
    if probe_radii is None or potential is None:
        return summary
    radii = probe_radii.reshape(-1).astype(float)
    values = potential.reshape(-1).astype(float)
    if radii.size != values.size:
        missing.append("boundary_evidence.multipole_check.size_mismatch")
        return summary
    if np.any(radii <= 0):
        missing.append("boundary_evidence.multipole_check.positive_radii")
        return summary
    expected = prefactor * float(monopole) / radii
    residual = expected - values
    max_abs = float(np.max(np.abs(residual))) if residual.size else 0.0
    summary.update({"max_abs_residual": max_abs, "num_probe_points": int(radii.size), "monopole_charge": float(monopole)})
    if max_abs > input.direct_coulomb_residual_tolerance:
        failed.append("multipole_far_field_residual_above_tolerance")
    return summary


def _vacuum_finite_size_correction_check(
    check: dict[str, Any],
    input: CheckKSMolecularContextEvidenceInput,
    failed: list[str],
    missing: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {"checked": True}
    raw = _first_numeric(check, ("raw_energy", "uncorrected_energy", "periodic_energy"))
    corrected = _first_numeric(check, ("corrected_energy", "final_energy"))
    correction = _first_numeric(check, ("correction_energy", "finite_size_correction", "energy_correction"))
    terms = check.get("correction_terms")
    term_sum: float | None = None
    if isinstance(terms, list):
        values = []
        for index, term in enumerate(terms):
            if not isinstance(term, dict):
                missing.append(f"boundary_evidence.vacuum_finite_size_correction.correction_terms.{index}")
                continue
            value = _first_numeric(term, ("value", "energy", "correction_energy"))
            if value is None:
                missing.append(f"boundary_evidence.vacuum_finite_size_correction.correction_terms.{index}.value")
                continue
            values.append(value)
        if values:
            term_sum = float(sum(values))
            summary["correction_terms_sum"] = term_sum
    if correction is None and term_sum is not None:
        correction = term_sum
    if raw is None or corrected is None or correction is None:
        missing.append("boundary_evidence.vacuum_finite_size_correction.energy_terms")
        return summary
    absolute_error = abs((raw + correction) - corrected)
    summary.update({"raw_energy": raw, "correction_energy": correction, "corrected_energy": corrected, "absolute_error": absolute_error})
    if absolute_error > input.finite_size_correction_tolerance:
        failed.append("vacuum_finite_size_correction_inconsistent")
    if term_sum is not None and abs(term_sum - correction) > input.finite_size_correction_tolerance:
        failed.append("vacuum_finite_size_terms_inconsistent")

    sweep = check.get("padding_sweep") or check.get("cell_size_sweep")
    if isinstance(sweep, list) and len(sweep) >= 2:
        sizes: list[float] = []
        magnitudes: list[float] = []
        for item in sweep:
            if not isinstance(item, dict):
                continue
            size = _first_numeric(item, ("padding_angstrom", "cell_length_angstrom", "cell_size_angstrom"))
            corr = _first_numeric(item, ("correction_energy", "finite_size_correction", "energy_correction"))
            if size is not None and corr is not None:
                sizes.append(size)
                magnitudes.append(abs(corr))
        if len(sizes) >= 2:
            summary["padding_sweep_points"] = len(sizes)
            if sizes[-1] > sizes[0] and magnitudes[-1] > magnitudes[0] + input.finite_size_correction_tolerance:
                failed.append("vacuum_finite_size_correction_not_decreasing_with_padding")
    return summary


def _safe_formula_eval(expression: str, variables: dict[str, float]) -> float:
    import ast
    import operator

    allowed_binary = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
    }
    allowed_unary = {ast.UAdd: operator.pos, ast.USub: operator.neg}
    constants = {"pi": math.pi}

    def eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in variables:
                return float(variables[node.id])
            if node.id in constants:
                return float(constants[node.id])
            raise ValueError(f"Unknown formula variable: {node.id}")
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_binary:
            return float(allowed_binary[type(node.op)](eval_node(node.left), eval_node(node.right)))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_unary:
            return float(allowed_unary[type(node.op)](eval_node(node.operand)))
        raise ValueError(f"Unsupported formula expression node: {type(node).__name__}")

    parsed = ast.parse(expression, mode="eval")
    return eval_node(parsed)


def _correction_formula_manifest_check(
    manifest: dict[str, Any],
    metadata: dict[str, Any],
    input: CheckKSMolecularContextEvidenceInput,
    failed: list[str],
    missing: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {"checked": True}
    formula_id = manifest.get("formula_id") or manifest.get("name")
    expression = manifest.get("expression")
    variables_raw = manifest.get("variables")
    expected = _first_numeric(manifest, ("reported_value", "correction_energy", "value"))
    if not isinstance(formula_id, str) or not formula_id:
        missing.append("boundary_evidence.correction_formula_manifest.formula_id")
    if not isinstance(expression, str) or not expression:
        missing.append("boundary_evidence.correction_formula_manifest.expression")
    if not isinstance(variables_raw, dict):
        missing.append("boundary_evidence.correction_formula_manifest.variables")
    if expected is None:
        missing.append("boundary_evidence.correction_formula_manifest.reported_value")
    applicability = manifest.get("applicability")
    if input.require_formula_manifest_when_present:
        if not isinstance(applicability, dict):
            missing.append("boundary_evidence.correction_formula_manifest.applicability")
        else:
            for key in ("boundary_policy", "assumptions", "units"):
                if not applicability.get(key):
                    missing.append(f"boundary_evidence.correction_formula_manifest.applicability.{key}")
    if not isinstance(expression, str) or not isinstance(variables_raw, dict) or expected is None:
        return summary
    variables: dict[str, float] = {}
    for key, value in variables_raw.items():
        if not isinstance(value, (int, float)):
            missing.append(f"boundary_evidence.correction_formula_manifest.variables.{key}")
            continue
        variables[str(key)] = float(value)
    if not variables:
        return summary
    try:
        computed = _safe_formula_eval(expression, variables)
    except Exception as exc:
        missing.append(f"boundary_evidence.correction_formula_manifest.expression_eval:{exc}")
        return summary
    absolute_error = abs(computed - float(expected))
    summary.update({"formula_id": formula_id, "computed_value": computed, "reported_value": float(expected), "absolute_error": absolute_error})
    if absolute_error > input.finite_size_correction_tolerance:
        failed.append("correction_formula_manifest_value_mismatch")
    provenance = manifest.get("provenance")
    if input.require_formula_manifest_when_present and not isinstance(provenance, dict):
        missing.append("boundary_evidence.correction_formula_manifest.provenance")
    manifest_hash = str(manifest.get("sha256") or _stable_hash({key: value for key, value in manifest.items() if key != "sha256"}))
    summary["manifest_hash"] = manifest_hash
    runtime_ref = _metadata_value(metadata, "correction_formula_manifest", "correction_formula_manifest_ref", "molecular_correction_formula")
    if input.require_formula_manifest_when_present:
        if not isinstance(runtime_ref, dict):
            missing.append("runtime_metadata.correction_formula_manifest")
        else:
            runtime_formula_id = runtime_ref.get("formula_id") or runtime_ref.get("id")
            runtime_hash = runtime_ref.get("sha256") or runtime_ref.get("hash")
            runtime_policy = runtime_ref.get("selected_policy") or runtime_ref.get("boundary_policy")
            if formula_id and runtime_formula_id != formula_id:
                failed.append("runtime_formula_id_mismatch")
            if runtime_hash and runtime_hash != manifest_hash:
                failed.append("runtime_formula_hash_mismatch")
            if not runtime_hash:
                missing.append("runtime_metadata.correction_formula_manifest.sha256")
            if isinstance(applicability, dict) and runtime_policy and runtime_policy != applicability.get("boundary_policy"):
                failed.append("runtime_formula_policy_mismatch")
    return summary


def _grid_spacing_from_check(check: dict[str, Any], shape: tuple[int, int, int], missing: list[str]) -> tuple[float, float, float] | None:
    spacing = check.get("grid_spacing_angstrom") or check.get("spacing_angstrom")
    if isinstance(spacing, (int, float)):
        value = float(spacing)
        return (value, value, value)
    if isinstance(spacing, list) and len(spacing) == 3:
        values = tuple(float(item) for item in spacing)
        if all(value > 0 for value in values):
            return values
    cell_lengths = check.get("cell_lengths_angstrom")
    if isinstance(cell_lengths, list) and len(cell_lengths) == 3:
        values = tuple(float(cell_lengths[i]) / float(shape[i]) for i in range(3))
        if all(value > 0 for value in values):
            return values
    missing.append("boundary_evidence.grid_poisson_check.grid_spacing_or_cell_lengths")
    return None


def _nonperiodic_laplacian_grid(values: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    grid = np.asarray(values, dtype=float)
    lap = np.zeros_like(grid)
    for axis, h in enumerate(spacing):
        shifted_plus = np.roll(grid, -1, axis=axis)
        shifted_minus = np.roll(grid, 1, axis=axis)
        first = [slice(None)] * 3
        first[axis] = 0
        last = [slice(None)] * 3
        last[axis] = -1
        second = [slice(None)] * 3
        second[axis] = 1
        penultimate = [slice(None)] * 3
        penultimate[axis] = -2
        shifted_minus[tuple(first)] = grid[tuple(second)]
        shifted_plus[tuple(last)] = grid[tuple(penultimate)]
        lap += (shifted_plus - 2.0 * grid + shifted_minus) / max(h * h, 1e-300)
    return lap


def _grid_poisson_check(
    check: dict[str, Any],
    input: CheckKSMolecularContextEvidenceInput,
    failed: list[str],
    missing: list[str],
) -> dict[str, Any]:
    density = _as_float_array(check.get("density"), "boundary_evidence.grid_poisson_check.density", missing)
    potential = _as_float_array(check.get("hartree_potential"), "boundary_evidence.grid_poisson_check.hartree_potential", missing)
    summary: dict[str, Any] = {"checked": True}
    if density is None or potential is None:
        return summary
    shape_raw = check.get("grid_shape")
    if isinstance(shape_raw, list) and len(shape_raw) == 3:
        shape = tuple(int(item) for item in shape_raw)
    elif density.ndim == 3:
        shape = tuple(int(item) for item in density.shape)
    else:
        missing.append("boundary_evidence.grid_poisson_check.grid_shape")
        return summary
    density_grid = density.reshape(shape)
    potential_grid = potential.reshape(shape)
    spacing = _grid_spacing_from_check(check, shape, missing)
    if spacing is None:
        return summary
    source_sign = float(check.get("source_sign", 1.0) or 1.0)
    density_reference = str(check.get("density_reference", "mean_neutralized"))
    rho = density_grid.astype(float)
    if density_reference == "mean_neutralized":
        rho = rho - float(np.mean(rho))
    elif density_reference == "provided":
        pass
    else:
        missing.append("boundary_evidence.grid_poisson_check.density_reference")
        return summary
    laplacian = _nonperiodic_laplacian_grid(potential_grid, spacing)
    residual = laplacian + source_sign * 4.0 * math.pi * rho
    interior = check.get("interior_slices")
    if isinstance(interior, list) and len(interior) == 6:
        sx = slice(int(interior[0]), int(interior[1]))
        sy = slice(int(interior[2]), int(interior[3]))
        sz = slice(int(interior[4]), int(interior[5]))
        residual_eval = residual[sx, sy, sz]
    else:
        residual_eval = residual
    max_abs = float(np.max(np.abs(residual_eval))) if residual_eval.size else 0.0
    l2 = float(np.linalg.norm(residual_eval.reshape(-1)) / math.sqrt(max(residual_eval.size, 1)))
    summary.update({"grid_shape": list(shape), "spacing_angstrom": list(spacing), "max_abs_residual": max_abs, "l2_residual": l2})
    if max_abs > input.grid_poisson_residual_tolerance and l2 > input.grid_poisson_residual_tolerance:
        failed.append("grid_poisson_residual_above_tolerance")

    boundary_samples = check.get("boundary_samples")
    if isinstance(boundary_samples, list) and boundary_samples:
        errors: list[float] = []
        for index, sample in enumerate(boundary_samples):
            if not isinstance(sample, dict):
                missing.append(f"boundary_evidence.grid_poisson_check.boundary_samples.{index}")
                continue
            grid_index = sample.get("grid_index")
            expected = _first_numeric(sample, ("expected_potential", "dirichlet_value", "target_potential"))
            if not isinstance(grid_index, list) or len(grid_index) != 3 or expected is None:
                missing.append(f"boundary_evidence.grid_poisson_check.boundary_samples.{index}.fields")
                continue
            ix, iy, iz = (int(item) for item in grid_index)
            if not (0 <= ix < shape[0] and 0 <= iy < shape[1] and 0 <= iz < shape[2]):
                missing.append(f"boundary_evidence.grid_poisson_check.boundary_samples.{index}.grid_index")
                continue
            errors.append(abs(float(potential_grid[ix, iy, iz]) - expected))
        max_boundary_error = max(errors) if errors else None
        summary["max_boundary_error"] = max_boundary_error
        if max_boundary_error is not None and max_boundary_error > input.grid_boundary_residual_tolerance:
            failed.append("grid_boundary_residual_above_tolerance")
    return summary


def _validate_fragment_numerics(
    evidence: dict[str, Any],
    input: CheckKSMolecularContextEvidenceInput,
    failed: list[str],
    missing: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {"checked": True}
    fragments = evidence.get("fragments")
    if isinstance(fragments, list) and fragments:
        errors: list[float] = []
        total_expected = 0.0
        total_integrated = 0.0
        for index, fragment in enumerate(fragments):
            if not isinstance(fragment, dict):
                missing.append(f"fragment_evidence.fragments.{index}")
                continue
            expected = _first_numeric(fragment, ("expected_charge", "expected_electrons", "target_charge"))
            integrated = _first_numeric(fragment, ("integrated_charge", "integrated_electrons", "charge"))
            if expected is None or integrated is None:
                missing.append(f"fragment_evidence.fragments.{index}.charge_terms")
                continue
            total_expected += expected
            total_integrated += integrated
            errors.append(abs(integrated - expected))
        max_error = max(errors) if errors else None
        total_error = abs(total_integrated - total_expected)
        summary.update(
            {
                "num_fragments": len(fragments),
                "max_fragment_charge_error": max_error,
                "total_fragment_charge_error": total_error,
            }
        )
        if max_error is not None and max_error > input.fragment_charge_tolerance:
            failed.append("fragment_charge_error_above_tolerance")
        if total_error > input.fragment_charge_tolerance:
            failed.append("fragment_total_charge_error_above_tolerance")
    else:
        max_error = _first_numeric(evidence, ("max_fragment_charge_error", "charge_error", "fragment_charge_error"))
        if max_error is None:
            missing.append("fragment_evidence.fragments_or_max_error")
        else:
            summary["max_fragment_charge_error"] = max_error
            if max_error > input.fragment_charge_tolerance:
                failed.append("fragment_charge_error_above_tolerance")
    return summary


def _validate_locality_numerics(
    evidence: dict[str, Any],
    input: CheckKSMolecularContextEvidenceInput,
    failed: list[str],
    missing: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {"checked": True}
    delta = _first_numeric(evidence, ("final_delta", "energy_delta", "density_delta", "truncation_delta", "locality_delta"))
    sweep = evidence.get("sweep") or evidence.get("truncation_sweep") or evidence.get("locality_sweep")
    if isinstance(sweep, list) and len(sweep) >= 2:
        values: list[float] = []
        for item in sweep:
            if isinstance(item, dict):
                value = _first_numeric(item, ("energy_total", "energy", "residual", "observable"))
                if value is not None:
                    values.append(value)
        if len(values) >= 2:
            delta = abs(values[-1] - values[-2])
            summary["sweep_delta"] = delta
            summary["sweep_points"] = len(values)
    if delta is None:
        missing.append("locality_evidence.delta_or_sweep")
    else:
        summary["final_delta"] = delta
        if delta > input.locality_delta_tolerance:
            failed.append("locality_delta_above_tolerance")
    return summary


def _validate_scaling_numerics(
    evidence: dict[str, Any],
    input: CheckKSMolecularContextEvidenceInput,
    failed: list[str],
    missing: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {"checked": True}
    efficiency = _first_numeric(evidence, ("parallel_efficiency", "scaling_efficiency", "weak_scaling_efficiency"))
    if efficiency is not None:
        summary["scaling_efficiency"] = efficiency
        if efficiency < input.scaling_efficiency_min:
            failed.append("scaling_efficiency_below_minimum")
    samples = evidence.get("samples") or evidence.get("scale_sweep")
    if isinstance(samples, list) and len(samples) >= 2:
        sizes: list[float] = []
        costs: list[float] = []
        for item in samples:
            if not isinstance(item, dict):
                continue
            size = _first_numeric(item, ("num_atoms", "basis_size", "degrees_of_freedom", "size"))
            cost = _first_numeric(item, ("wall_time_seconds", "runtime_seconds", "cost", "matvec_count"))
            if size is not None and cost is not None and size > 0 and cost > 0:
                sizes.append(size)
                costs.append(cost)
        if len(sizes) >= 2:
            slope = math.log(costs[-1] / costs[0]) / math.log(sizes[-1] / sizes[0])
            summary["empirical_scaling_exponent"] = slope
            if slope > 2.0:
                failed.append("scaling_exponent_too_high_for_large_route")
    if "scaling_efficiency" not in summary and "empirical_scaling_exponent" not in summary:
        missing.append("scaling_evidence.efficiency_or_samples")
    return summary


def check_ks_molecular_context_evidence(input: CheckKSMolecularContextEvidenceInput) -> KSDftVerificationToolOutput:
    """Check molecule/cluster KS-DFT provenance without imposing a fixed molecular solver."""
    try:
        context_ref = input.molecular_context_ref or f"/workspace/cases/{input.case_id}/materials/ks_dft_molecular_context.json"
        metadata_ref = input.runtime_metadata_ref or f"/workspace/cases/{input.case_id}/taps/ks_dft_runtime_metadata.json"
        scaling_ref = input.scaling_policy_ref or f"/workspace/cases/{input.case_id}/taps/molecular_taps_scaling_policy.json"
        boundary_ref = input.boundary_evidence_ref or f"/workspace/cases/{input.case_id}/verification/ks_dft/molecular_boundary_evidence.json"
        fragment_ref = input.fragment_evidence_ref or f"/workspace/cases/{input.case_id}/verification/ks_dft/fragment_charge_consistency.json"
        locality_ref = input.locality_evidence_ref or f"/workspace/cases/{input.case_id}/verification/ks_dft/molecular_locality_sensitivity.json"
        scaling_evidence_ref = input.scaling_evidence_ref or f"/workspace/cases/{input.case_id}/verification/ks_dft/molecular_scaling_evidence.json"
        context = _read_json(context_ref)
        metadata = _read_json(metadata_ref)
        scaling = _read_json(scaling_ref)
        missing: list[str] = []
        failed: list[str] = []
        warnings: list[str] = []
        numerical_checks: dict[str, Any] = {}

        if context.get("schema_version") != "physicsos.ks_dft_molecular_context.v1":
            failed.append("molecular_context_schema")
        if scaling.get("schema_version") != "physicsos.molecular_taps_scaling_policy.v1":
            failed.append("molecular_scaling_policy_schema")
        molecule_ref = context.get("molecule_ref")
        if not isinstance(molecule_ref, str) or not molecule_ref:
            missing.append("molecular_context.molecule_ref")
        elif not resolve_workspace_path(molecule_ref, workspace=_workspace(), must_be_within_workspace=False).exists():
            missing.append("molecular_context.molecule_ref:file_missing")

        context_charge = context.get("charge")
        context_multiplicity = context.get("multiplicity")
        metadata_charge = _metadata_value(metadata, "charge", "total_charge", "molecular_charge")
        metadata_multiplicity = _metadata_value(metadata, "multiplicity", "spin_multiplicity")
        if input.require_charge_spin_metadata:
            if metadata_charge is None:
                missing.append("runtime_metadata.charge")
            elif context_charge is not None and int(metadata_charge) != int(context_charge):
                failed.append("charge_mismatch")
            if metadata_multiplicity is None:
                missing.append("runtime_metadata.multiplicity")
            elif context_multiplicity is not None and int(metadata_multiplicity) != int(context_multiplicity):
                failed.append("multiplicity_mismatch")

        boundary_policy = _metadata_value(metadata, "poisson_boundary_policy", "boundary_policy", "molecular_boundary_policy")
        if not isinstance(boundary_policy, dict):
            missing.append("runtime_metadata.poisson_boundary_policy")
            boundary_selected = None
        else:
            boundary_selected = boundary_policy.get("selected") or boundary_policy.get("poisson_boundary") or boundary_policy.get("type")
            if not boundary_selected or boundary_selected == "llm_select":
                failed.append("boundary_policy_not_finalized")
        if input.require_boundary_evidence:
            boundary_evidence = _load_optional_json(boundary_ref, missing, "boundary_evidence")
            if boundary_evidence is not None:
                if boundary_evidence.get("passes") is not True:
                    failed.append("boundary_evidence_not_passing")
                numerical_checks["boundary"] = _validate_boundary_numerics(boundary_evidence, metadata, input, failed, missing, warnings)
                method = boundary_evidence.get("method") or boundary_evidence.get("poisson_boundary")
                if boundary_selected and method and str(boundary_selected) not in {str(method), str(boundary_evidence.get("selected"))}:
                    warnings.append("Boundary evidence method differs from runtime boundary policy; inspect provenance.")

        selected_strategy = _metadata_value(metadata, "molecular_scaling_policy", "selected_molecular_strategy", "strategy_family")
        selected_strategies: list[str]
        if isinstance(selected_strategy, dict):
            raw = selected_strategy.get("selected_strategies") or selected_strategy.get("strategies") or selected_strategy.get("selected")
            selected_strategies = [str(item) for item in raw] if isinstance(raw, list) else [str(raw)] if raw else []
        elif isinstance(selected_strategy, list):
            selected_strategies = [str(item) for item in selected_strategy]
        elif isinstance(selected_strategy, str):
            selected_strategies = [selected_strategy]
        else:
            selected_strategies = []
        if not selected_strategies:
            missing.append("runtime_metadata.molecular_scaling_policy")
        candidate_strategies = scaling.get("candidate_strategies", [])
        if isinstance(candidate_strategies, list) and selected_strategies:
            unknown = [strategy for strategy in selected_strategies if strategy not in {str(item) for item in candidate_strategies}]
            if unknown:
                failed.append("selected_strategy_not_in_policy")

        fragment_selected = "fragment_partition" in selected_strategies
        locality_selected = any(
            strategy in selected_strategies
            for strategy in ("localized_orbitals", "density_matrix_truncation", "fragment_partition", "near_field_far_field_coulomb")
        )
        fragment_evidence: dict[str, Any] | None = None
        if input.require_fragment_evidence_when_selected and fragment_selected:
            fragment_evidence = _load_optional_json(fragment_ref, missing, "fragment_evidence")
            if fragment_evidence is not None and fragment_evidence.get("passes") is not True:
                failed.append("fragment_evidence_not_passing")
            if fragment_evidence is not None:
                numerical_checks["fragment"] = _validate_fragment_numerics(fragment_evidence, input, failed, missing)
        locality_evidence: dict[str, Any] | None = None
        if input.require_locality_evidence_when_selected and locality_selected:
            locality_evidence = _load_optional_json(locality_ref, missing, "locality_evidence")
            if locality_evidence is not None and locality_evidence.get("passes") is not True:
                failed.append("locality_evidence_not_passing")
            if locality_evidence is not None:
                numerical_checks["locality"] = _validate_locality_numerics(locality_evidence, input, failed, missing)

        target_scale = str(scaling.get("target_scale") or "")
        large_target = target_scale in {"large", "very_large"} or any(
            strategy in selected_strategies
            for strategy in ("density_matrix_truncation", "fragment_partition", "hierarchical_taps_axes")
        )
        if input.require_scaling_evidence_for_large_targets and large_target:
            scaling_evidence = _load_optional_json(scaling_evidence_ref, missing, "scaling_evidence")
            if scaling_evidence is not None:
                if scaling_evidence.get("passes") is not True:
                    failed.append("scaling_evidence_not_passing")
                numerical_checks["scaling"] = _validate_scaling_numerics(scaling_evidence, input, failed, missing)

        if input.forbid_crystal_kmesh_without_vacuum_box:
            used = metadata.get("materials_artifacts_used", [])
            used_text = " ".join(str(item) for item in used) if isinstance(used, list) else str(used)
            has_crystal_kmesh = "kmesh" in used_text or "irreducible_kpoints" in used_text or "kpath" in used_text
            is_vacuum_box = False
            if isinstance(boundary_policy, dict):
                is_vacuum_box = "vacuum" in str(boundary_policy.get("selected") or boundary_policy.get("type") or "").lower()
            if has_crystal_kmesh and not is_vacuum_box:
                failed.append("crystal_kmesh_used_without_vacuum_box_policy")

        passes = not missing and not failed
        payload = {
            "schema_version": "physicsos.ks_dft.molecular_context_evidence.v1",
            "molecular_context_ref": context_ref,
            "runtime_metadata_ref": metadata_ref,
            "scaling_policy_ref": scaling_ref,
            "boundary_evidence_ref": boundary_ref if input.require_boundary_evidence else None,
            "fragment_evidence_ref": fragment_ref if fragment_selected else None,
            "locality_evidence_ref": locality_ref if locality_selected else None,
            "scaling_evidence_ref": scaling_evidence_ref if large_target else None,
            "charge": context_charge,
            "multiplicity": context_multiplicity,
            "boundary_policy": boundary_policy,
            "selected_strategies": selected_strategies,
            "fragment_selected": fragment_selected,
            "locality_selected": locality_selected,
            "large_target": large_target,
            "numerical_checks": numerical_checks,
            "missing": missing,
            "failed": failed,
            "warnings": warnings,
            "passes": passes,
        }
        path = _verification_dir(input.case_id) / "molecular_context_evidence.json"
        _write_json(path, payload)
        _append_event(_case_dir(input.case_id), "check_ks_molecular_context_evidence", {"passes": passes})
        return KSDftVerificationToolOutput(
            artifact=_artifact(path, "ks_dft_molecular_context_evidence"),
            data=payload,
            passes=passes,
            warnings=warnings,
        )
    except Exception as exc:
        return KSDftVerificationToolOutput(errors=[str(exc)])


KS_DFT_VERIFICATION_TOOL_SPECS = [
    (check_ks_charge_conservation, CheckKSChargeConservationInput, KSDftVerificationToolOutput),
    (check_ks_orthonormality, CheckKSOrthonormalityInput, KSDftVerificationToolOutput),
    (check_ks_scf_residual, CheckKSSCFResidualInput, KSDftVerificationToolOutput),
    (check_ks_poisson_residual, CheckKSPoissonResidualInput, KSDftVerificationToolOutput),
    (check_ks_rank_grid_kpoint_convergence, CheckKSRankGridKpointConvergenceInput, KSDftVerificationToolOutput),
    (check_ks_material_artifact_usage, CheckKSMaterialArtifactUsageInput, KSDftVerificationToolOutput),
    (check_ks_hamiltonian_evidence, CheckKSHamiltonianEvidenceInput, KSDftVerificationToolOutput),
    (check_ks_band_dos_provenance, CheckKSBandDosProvenanceInput, KSDftVerificationToolOutput),
    (check_ks_molecular_context_evidence, CheckKSMolecularContextEvidenceInput, KSDftVerificationToolOutput),
]

for _tool, _input, _output in KS_DFT_VERIFICATION_TOOL_SPECS:
    _tool.input_model = _input
    _tool.output_model = _output
    _tool.side_effects = "workspace artifacts only"
    _tool.requires_approval = False
