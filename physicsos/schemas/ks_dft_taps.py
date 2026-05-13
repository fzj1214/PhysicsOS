from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from physicsos.schemas.common import StrictBaseModel


class KSDftTapsProblemSpec(StrictBaseModel):
    route: Literal["ks_dft_taps"] = "ks_dft_taps"
    system_type: Literal["molecule", "periodic_crystal", "slab", "defect", "toy_model"] = "periodic_crystal"
    structure_ref: str | None = None
    standardized_structure_ref: str | None = None
    symmetry_ref: str | None = None
    reciprocal_lattice_ref: str | None = None
    kmesh_ref: str | None = None
    irreducible_kpoints_ref: str | None = None
    kpath_ref: str | None = None
    electron_count: float | None = None
    spin_mode: Literal["nonmagnetic", "collinear", "noncollinear", "soc", "unknown"] = "unknown"
    xc_functional: str | None = None
    pseudopotential_spec: dict[str, Any] = Field(default_factory=dict)
    smearing_spec: dict[str, Any] = Field(default_factory=dict)
    hamiltonian_terms: list[str] = Field(default_factory=list)
    taps_basis_policy: dict[str, Any] = Field(default_factory=dict)
    subspace_update_policy: dict[str, Any] = Field(default_factory=dict)
    scf_policy: dict[str, Any] = Field(default_factory=dict)
    verification_policy: dict[str, Any] = Field(default_factory=dict)
    missing_assumptions: list[str] = Field(default_factory=list)


class KSDftTapsAxisSpec(StrictBaseModel):
    name: str
    kind: Literal["space", "reciprocal", "band_subspace", "parameter", "spin", "scf"]
    source_artifact_ref: str | None = None
    domain: dict[str, Any] = Field(default_factory=dict)
    points: int | None = None
    weights: list[float] = Field(default_factory=list)
    rank: int | None = None
    units: str | None = None
    separability_assumption: str | None = None
    refinement_policy: dict[str, Any] = Field(default_factory=dict)


class KSDftTapsResultSpec(StrictBaseModel):
    energy_total: float | None = None
    energy_terms: dict[str, float] = Field(default_factory=dict)
    density_ref: str | None = None
    hartree_potential_ref: str | None = None
    occupied_subspace_ref: str | None = None
    density_matrix_ref: str | None = None
    fermi_level: float | None = None
    band_gap_optional: float | None = None
    charge_error: float | None = None
    orthonormality_error: float | None = None
    scf_residual: float | None = None
    poisson_residual: float | None = None
    rank_history: list[dict[str, Any]] = Field(default_factory=list)
    grid_history: list[dict[str, Any]] = Field(default_factory=list)
    kpoint_history: list[dict[str, Any]] = Field(default_factory=list)
    materials_artifacts_used: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class KSDftTapsVerificationSpec(StrictBaseModel):
    charge_conservation: bool = True
    orthonormality: bool = True
    scf_residual: bool = True
    poisson_residual: bool = True
    rank_convergence: bool = True
    grid_convergence: bool = True
    kpoint_convergence: bool = True
    required_material_artifacts: list[str] = Field(default_factory=list)
