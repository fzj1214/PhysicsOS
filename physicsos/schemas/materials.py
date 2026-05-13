from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from physicsos.schemas.common import ArtifactRef, StrictBaseModel


class MaterialProperty(StrictBaseModel):
    name: str
    value: float | int | str | list[float]
    units: str | None = None
    temperature_dependence: str | None = None


class MaterialSpec(StrictBaseModel):
    id: str
    name: str
    phase: Literal["solid", "liquid", "gas", "plasma", "crystal", "molecule", "mixture", "custom"]
    region_ids: list[str] = Field(default_factory=list)
    properties: list[MaterialProperty] = Field(default_factory=list)


class CrystalStructureRef(StrictBaseModel):
    structure: ArtifactRef
    formula: str | None = None
    num_sites: int | None = None
    lattice: list[list[float]] = Field(default_factory=list)
    species: list[str] = Field(default_factory=list)
    frac_coords: list[list[float]] = Field(default_factory=list)
    charge: float | None = None
    site_properties: dict[str, Any] = Field(default_factory=dict)


class SymmetryDatasetRef(StrictBaseModel):
    symmetry: ArtifactRef
    spacegroup_symbol: str | None = None
    spacegroup_number: int | None = None
    point_group: str | None = None
    hall: str | None = None
    wyckoffs: list[str] = Field(default_factory=list)
    equivalent_atoms: list[int] = Field(default_factory=list)
    symprec: float | None = None
    angle_tolerance: float | None = None


class KPointMeshSpec(StrictBaseModel):
    kmesh: ArtifactRef
    mesh: list[int] = Field(default_factory=list)
    shift: list[int] = Field(default_factory=list)
    num_kpoints_full: int | None = None
    generation_policy: dict[str, Any] = Field(default_factory=dict)


class KPathSpec(StrictBaseModel):
    kpath: ArtifactRef
    point_coords: dict[str, list[float]] = Field(default_factory=dict)
    path: list[list[str]] = Field(default_factory=list)
    convention: str | None = None
    warnings: list[str] = Field(default_factory=list)


class MaterialsPreprocessResultSpec(StrictBaseModel):
    source_structure_ref: str | None = None
    standardized_structure_ref: str | None = None
    primitive_structure_ref: str | None = None
    conventional_structure_ref: str | None = None
    symmetry_ref: str | None = None
    reciprocal_lattice_ref: str | None = None
    kmesh_ref: str | None = None
    irreducible_kpoints_ref: str | None = None
    kpath_seekpath_ref: str | None = None
    kpath_pymatgen_ref: str | None = None
    taps_kpoint_axis_ref: str | None = None
    symprec: float | None = None
    angle_tolerance: float | None = None
    species_order: list[str] = Field(default_factory=list)
    transformations: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
