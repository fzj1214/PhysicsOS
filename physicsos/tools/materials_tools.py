from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field

from physicsos.config import runtime_paths
from physicsos.paths import resolve_workspace_path, to_agent_path
from physicsos.schemas.common import ArtifactRef, StrictBaseModel


def _workspace() -> Path:
    return runtime_paths().workspace


def _case_dir(case_id: str) -> Path:
    return _workspace() / "cases" / case_id


def _materials_dir(case_id: str) -> Path:
    path = _case_dir(case_id) / "materials"
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


def _missing_dependency(*modules: str) -> list[str]:
    missing = []
    for module in modules:
        if importlib.util.find_spec(module) is None:
            missing.append(module)
    return missing


def _require_modules(*modules: str) -> list[str]:
    missing = _missing_dependency(*modules)
    if missing:
        raise ImportError(
            "Missing optional materials dependencies: "
            + ", ".join(missing)
            + ". Install the PhysicsOS materials stack with pymatgen, spglib, and seekpath."
        )
    return missing


def _structure_from_ref(structure_ref: str):
    _require_modules("pymatgen")
    from pymatgen.core import Structure

    payload = _read_json(structure_ref)
    if "pymatgen_structure" in payload and isinstance(payload["pymatgen_structure"], dict):
        return Structure.from_dict(payload["pymatgen_structure"])
    if "@module" in payload or "lattice" in payload and "sites" in payload:
        return Structure.from_dict(payload)
    raise ValueError(f"Could not read pymatgen structure from {structure_ref}")


def _structure_summary(structure: Any) -> dict[str, Any]:
    return {
        "formula": structure.composition.reduced_formula,
        "num_sites": len(structure),
        "lattice": [[float(value) for value in row] for row in structure.lattice.matrix],
        "species": [str(site.specie) for site in structure],
        "frac_coords": [[float(value) for value in site.frac_coords] for site in structure],
        "charge": getattr(structure, "charge", None),
        "site_properties": {key: list(value) for key, value in structure.site_properties.items()},
    }


def _write_structure(path: Path, structure: Any, *, source: str | None = None) -> dict[str, Any]:
    payload = {
        "schema_version": "physicsos.material_structure.v1",
        "source": source,
        "pymatgen_structure": structure.as_dict(),
        **_structure_summary(structure),
    }
    _write_json(path, payload)
    return payload


def _element_symbol(value: str) -> str:
    cleaned = "".join(ch for ch in str(value).strip() if ch.isalpha())
    if not cleaned:
        raise ValueError("Atom element symbol is empty.")
    symbol = cleaned[0].upper() + cleaned[1:].lower()
    if not symbol.isalpha() or len(symbol) > 3:
        raise ValueError(f"Invalid atom element symbol: {value!r}")
    return symbol


def _molecular_formula(atoms: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for atom in atoms:
        symbol = str(atom["element"])
        counts[symbol] = counts.get(symbol, 0) + 1
    if not counts:
        return ""
    ordered = []
    for preferred in ("C", "H"):
        if preferred in counts:
            ordered.append((preferred, counts.pop(preferred)))
    ordered.extend(sorted(counts.items()))
    return "".join(symbol if count == 1 else f"{symbol}{count}" for symbol, count in ordered)


def _molecule_bounds(atoms: list[dict[str, Any]]) -> dict[str, Any]:
    coords = np.array([atom["xyz_angstrom"] for atom in atoms], dtype=float)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = maxs - mins
    return {
        "min_xyz_angstrom": [float(value) for value in mins],
        "max_xyz_angstrom": [float(value) for value in maxs],
        "span_angstrom": [float(value) for value in span],
        "center_angstrom": [float(value) for value in (0.5 * (mins + maxs))],
    }


def _parse_xyz_molecule(text: str) -> tuple[list[dict[str, Any]], str | None]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("XYZ input is empty.")
    try:
        count = int(lines[0].split()[0])
    except Exception as exc:
        raise ValueError("XYZ input must start with an atom count.") from exc
    if len(lines) < count + 2:
        raise ValueError(f"XYZ input declares {count} atoms but only {max(0, len(lines) - 2)} coordinate lines were found.")
    comment = lines[1] if len(lines) > 1 else None
    atoms: list[dict[str, Any]] = []
    for index, line in enumerate(lines[2 : 2 + count]):
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"XYZ atom line {index + 1} must contain element and x/y/z coordinates.")
        xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
        if not np.isfinite(np.array(xyz, dtype=float)).all():
            raise ValueError(f"XYZ atom line {index + 1} contains non-finite coordinates.")
        atoms.append({"index": index, "element": _element_symbol(parts[0]), "xyz_angstrom": xyz})
    return atoms, comment


def _parse_json_molecule(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_atoms = payload.get("atoms")
    if not isinstance(raw_atoms, list) or not raw_atoms:
        raise ValueError("Molecular JSON must contain a non-empty `atoms` list.")
    atoms: list[dict[str, Any]] = []
    for index, item in enumerate(raw_atoms):
        if not isinstance(item, dict):
            raise ValueError(f"Atom {index} must be an object.")
        element = item.get("element") or item.get("species") or item.get("symbol")
        coords = item.get("xyz_angstrom") or item.get("xyz") or item.get("coords")
        if element is None or coords is None:
            raise ValueError(f"Atom {index} requires `element` and `xyz_angstrom` fields.")
        if not isinstance(coords, list | tuple) or len(coords) != 3:
            raise ValueError(f"Atom {index} coordinates must be a length-3 list.")
        xyz = [float(coords[0]), float(coords[1]), float(coords[2])]
        if not np.isfinite(np.array(xyz, dtype=float)).all():
            raise ValueError(f"Atom {index} contains non-finite coordinates.")
        atoms.append({"index": index, "element": _element_symbol(str(element)), "xyz_angstrom": xyz})
    return atoms


def _spglib_cell(structure: Any) -> tuple[list[list[float]], list[list[float]], list[int]]:
    numbers = [int(site.specie.Z) for site in structure]
    positions = [[float(value) for value in site.frac_coords] for site in structure]
    lattice = [[float(value) for value in row] for row in structure.lattice.matrix]
    return lattice, positions, numbers


class MaterialsToolOutput(StrictBaseModel):
    artifact: ArtifactRef | None = None
    artifacts: dict[str, ArtifactRef] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class ParseMaterialStructureInput(StrictBaseModel):
    case_id: str
    source_path: str | None = None
    source_text: str | None = None
    format: Literal["cif", "poscar", "json", "auto"] = "auto"
    primitive: bool = False


def parse_material_structure(input: ParseMaterialStructureInput) -> MaterialsToolOutput:
    """Parse CIF/POSCAR/JSON/text into a stable pymatgen structure artifact."""
    try:
        _require_modules("pymatgen")
        from pymatgen.core import Structure
        from pymatgen.io.vasp import Poscar

        if input.source_path:
            source = resolve_workspace_path(input.source_path, workspace=_workspace(), must_be_within_workspace=False)
            structure = Structure.from_file(source, primitive=input.primitive) if input.format in {"auto", "cif", "json"} else Poscar.from_file(source).structure
            source_label = str(source)
        elif input.source_text:
            fmt = "cif" if input.format == "auto" else input.format
            if fmt == "poscar":
                structure = Poscar.from_str(input.source_text).structure
            else:
                structure = Structure.from_str(input.source_text, fmt=fmt, primitive=input.primitive)
            source_label = "source_text"
        else:
            return MaterialsToolOutput(errors=["Either source_path or source_text is required."])
        path = _materials_dir(input.case_id) / "source_structure.json"
        payload = _write_structure(path, structure, source=source_label)
        return MaterialsToolOutput(artifact=_artifact(path, "material_structure"), data=payload)
    except Exception as exc:
        return MaterialsToolOutput(errors=[str(exc)])


class WriteMaterialStructureInput(StrictBaseModel):
    case_id: str
    structure_ref: str
    format: Literal["cif", "poscar", "json"]
    filename: str


def write_material_structure(input: WriteMaterialStructureInput) -> MaterialsToolOutput:
    """Write a structure artifact to CIF/POSCAR/JSON."""
    try:
        _require_modules("pymatgen")
        from pymatgen.io.vasp import Poscar

        structure = _structure_from_ref(input.structure_ref)
        path = _materials_dir(input.case_id) / input.filename
        if input.format == "json":
            _write_structure(path, structure, source=input.structure_ref)
        elif input.format == "poscar":
            path.write_text(Poscar(structure).get_string(), encoding="utf-8")
        else:
            path.write_text(structure.to(fmt="cif"), encoding="utf-8")
        return MaterialsToolOutput(artifact=_artifact(path, f"material_structure_{input.format}"))
    except Exception as exc:
        return MaterialsToolOutput(errors=[str(exc)])


class ValidateMaterialStructureInput(StrictBaseModel):
    case_id: str
    structure_ref: str
    min_distance_angstrom: float = 0.5


def validate_material_structure(input: ValidateMaterialStructureInput) -> MaterialsToolOutput:
    """Validate basic crystallographic sanity before KS-DFT-TAPS derivation."""
    try:
        structure = _structure_from_ref(input.structure_ref)
        errors: list[str] = []
        warnings: list[str] = []
        determinant = float(np.linalg.det(np.array(structure.lattice.matrix, dtype=float)))
        if determinant <= 0:
            errors.append("Lattice determinant must be positive.")
        if len(structure) == 0:
            errors.append("Structure has no sites.")
        finite = np.isfinite(np.array(structure.frac_coords, dtype=float)).all()
        if not finite:
            errors.append("Fractional coordinates contain non-finite values.")
        distances = structure.distance_matrix
        min_dist = None
        if len(structure) > 1:
            masked = distances + np.eye(len(structure)) * 1.0e9
            min_dist = float(masked.min())
            if min_dist < input.min_distance_angstrom:
                warnings.append(f"Minimum inter-site distance {min_dist:.4g} A is below threshold.")
        for key in ("magmom", "selective_dynamics", "label"):
            if key in structure.site_properties:
                warnings.append(f"Site property `{key}` is present and must be preserved across standardization.")
        payload = {
            "schema_version": "physicsos.material_structure_validation.v1",
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
            "lattice_determinant": determinant,
            "min_distance_angstrom": min_dist,
        }
        path = _materials_dir(input.case_id) / "structure_validation.json"
        _write_json(path, payload)
        return MaterialsToolOutput(artifact=_artifact(path, "material_structure_validation"), data=payload, warnings=warnings, errors=errors)
    except Exception as exc:
        return MaterialsToolOutput(errors=[str(exc)])


class AnalyzeSpacegroupInput(StrictBaseModel):
    case_id: str
    structure_ref: str
    symprec: float = 1e-5
    angle_tolerance: float = -1.0


def analyze_spacegroup(input: AnalyzeSpacegroupInput) -> MaterialsToolOutput:
    """Analyze space group, point group, Wyckoff labels, and equivalent atoms."""
    try:
        _require_modules("pymatgen")
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        structure = _structure_from_ref(input.structure_ref)
        analyzer = SpacegroupAnalyzer(structure, symprec=input.symprec, angle_tolerance=input.angle_tolerance)
        dataset = analyzer.get_symmetry_dataset()
        dataset_dict = dict(dataset) if not isinstance(dataset, dict) else dataset
        payload = {
            "schema_version": "physicsos.symmetry_dataset.v1",
            "spacegroup_symbol": analyzer.get_space_group_symbol(),
            "spacegroup_number": int(analyzer.get_space_group_number()),
            "point_group": analyzer.get_point_group_symbol(),
            "hall": str(dataset_dict.get("hall", "")),
            "wyckoffs": [str(item) for item in dataset_dict.get("wyckoffs", [])],
            "equivalent_atoms": [int(item) for item in dataset_dict.get("equivalent_atoms", [])],
            "symprec": input.symprec,
            "angle_tolerance": input.angle_tolerance,
            "dataset": _json_safe(dataset_dict),
        }
        path = _materials_dir(input.case_id) / "symmetry_dataset.json"
        _write_json(path, payload)
        return MaterialsToolOutput(artifact=_artifact(path, "symmetry_dataset"), data=payload)
    except Exception as exc:
        return MaterialsToolOutput(errors=[str(exc)])


class StandardizeCrystalStructureInput(StrictBaseModel):
    case_id: str
    structure_ref: str
    symprec: float = 1e-5
    angle_tolerance: float = -1.0
    keep_site_properties: bool = True


def standardize_crystal_structure(input: StandardizeCrystalStructureInput) -> MaterialsToolOutput:
    """Write standardized, primitive, and conventional structures with a transformation report."""
    try:
        _require_modules("pymatgen")
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        structure = _structure_from_ref(input.structure_ref)
        analyzer = SpacegroupAnalyzer(structure, symprec=input.symprec, angle_tolerance=input.angle_tolerance)
        refined = analyzer.get_refined_structure(keep_site_properties=input.keep_site_properties)
        primitive = analyzer.get_primitive_standard_structure()
        conventional = analyzer.get_conventional_standard_structure()
        mats_dir = _materials_dir(input.case_id)
        std_path = mats_dir / "structure_standardized.json"
        prim_path = mats_dir / "structure_primitive.json"
        conv_path = mats_dir / "structure_conventional.json"
        _write_structure(std_path, refined, source=input.structure_ref)
        _write_structure(prim_path, primitive, source=input.structure_ref)
        _write_structure(conv_path, conventional, source=input.structure_ref)
        report = {
            "schema_version": "physicsos.structure_standardization.v1",
            "source_structure_ref": input.structure_ref,
            "standardized_ref": to_agent_path(std_path, workspace=_workspace()),
            "primitive_ref": to_agent_path(prim_path, workspace=_workspace()),
            "conventional_ref": to_agent_path(conv_path, workspace=_workspace()),
            "symprec": input.symprec,
            "angle_tolerance": input.angle_tolerance,
            "species_order": [str(site.specie) for site in refined],
        }
        report_path = mats_dir / "structure_standardization_report.json"
        _write_json(report_path, report)
        return MaterialsToolOutput(
            artifacts={
                "standardized": _artifact(std_path, "material_structure"),
                "primitive": _artifact(prim_path, "material_structure"),
                "conventional": _artifact(conv_path, "material_structure"),
                "report": _artifact(report_path, "structure_standardization_report"),
            },
            data=report,
        )
    except Exception as exc:
        return MaterialsToolOutput(errors=[str(exc)])


class CompareCrystalStructuresInput(StrictBaseModel):
    case_id: str
    structure_ref_a: str
    structure_ref_b: str
    ltol: float = 0.2
    stol: float = 0.3
    angle_tol: float = 5.0


def compare_crystal_structures(input: CompareCrystalStructuresInput) -> MaterialsToolOutput:
    """Compare two structures with pymatgen StructureMatcher."""
    try:
        _require_modules("pymatgen")
        from pymatgen.analysis.structure_matcher import StructureMatcher

        a = _structure_from_ref(input.structure_ref_a)
        b = _structure_from_ref(input.structure_ref_b)
        matcher = StructureMatcher(ltol=input.ltol, stol=input.stol, angle_tol=input.angle_tol)
        match = bool(matcher.fit(a, b))
        rms = matcher.get_rms_dist(a, b) if match else None
        payload = {
            "schema_version": "physicsos.structure_comparison.v1",
            "match": match,
            "rms_dist": float(rms[0]) if rms else None,
            "max_dist": float(rms[1]) if rms else None,
        }
        path = _materials_dir(input.case_id) / "structure_comparison.json"
        _write_json(path, payload)
        return MaterialsToolOutput(artifact=_artifact(path, "structure_comparison"), data=payload)
    except Exception as exc:
        return MaterialsToolOutput(errors=[str(exc)])


class ReduceLatticeCellInput(StrictBaseModel):
    case_id: str
    structure_ref: str
    method: Literal["niggli", "delaunay"] = "niggli"
    eps: float = 1e-5


def reduce_lattice_cell(input: ReduceLatticeCellInput) -> MaterialsToolOutput:
    """Reduce a lattice with spglib and write the reduced matrix."""
    try:
        _require_modules("spglib")
        import spglib

        structure = _structure_from_ref(input.structure_ref)
        lattice = np.array(structure.lattice.matrix, dtype=float)
        reduced = spglib.niggli_reduce(lattice, eps=input.eps) if input.method == "niggli" else spglib.delaunay_reduce(lattice, eps=input.eps)
        payload = {
            "schema_version": "physicsos.reduced_lattice.v1",
            "method": input.method,
            "reduced_lattice": np.array(reduced, dtype=float).tolist(),
        }
        path = _materials_dir(input.case_id) / f"lattice_{input.method}_reduced.json"
        _write_json(path, payload)
        return MaterialsToolOutput(artifact=_artifact(path, "reduced_lattice"), data=payload)
    except Exception as exc:
        return MaterialsToolOutput(errors=[str(exc)])


class ComputeReciprocalLatticeInput(StrictBaseModel):
    case_id: str
    structure_ref: str
    convention: Literal["physics_2pi", "crystallographic"] = "physics_2pi"


def compute_reciprocal_lattice(input: ComputeReciprocalLatticeInput) -> MaterialsToolOutput:
    """Write reciprocal lattice vectors with an explicit convention."""
    try:
        structure = _structure_from_ref(input.structure_ref)
        lattice = structure.lattice.reciprocal_lattice if input.convention == "physics_2pi" else structure.lattice.reciprocal_lattice_crystallographic
        payload = {
            "schema_version": "physicsos.reciprocal_lattice.v1",
            "convention": input.convention,
            "units": "1/angstrom",
            "b_vectors": [[float(value) for value in row] for row in lattice.matrix],
            "volume": float(lattice.volume),
        }
        path = _materials_dir(input.case_id) / "reciprocal_lattice.json"
        _write_json(path, payload)
        return MaterialsToolOutput(artifact=_artifact(path, "reciprocal_lattice"), data=payload)
    except Exception as exc:
        return MaterialsToolOutput(errors=[str(exc)])


class GenerateUniformKmeshInput(StrictBaseModel):
    case_id: str
    structure_ref: str
    mode: Literal["automatic_density", "automatic_density_by_vol", "gamma", "monkhorst"] = "automatic_density"
    kppa: int = 1000
    grid_density_by_vol: float | None = None
    force_gamma: bool = False
    shift: list[int] = Field(default_factory=lambda: [0, 0, 0])


def generate_uniform_kmesh(input: GenerateUniformKmeshInput) -> MaterialsToolOutput:
    """Generate a uniform k-mesh policy artifact."""
    try:
        _require_modules("pymatgen")
        from pymatgen.io.vasp.inputs import Kpoints

        structure = _structure_from_ref(input.structure_ref)
        if input.mode == "automatic_density_by_vol":
            kpoints = Kpoints.automatic_density_by_vol(structure, input.grid_density_by_vol or float(input.kppa), force_gamma=input.force_gamma)
        elif input.mode == "gamma":
            kpoints = Kpoints.gamma_automatic(kpts=(max(1, round(input.kppa ** (1 / 3))),) * 3, shift=input.shift)
        elif input.mode == "monkhorst":
            kpoints = Kpoints.monkhorst_automatic(kpts=(max(1, round(input.kppa ** (1 / 3))),) * 3, shift=input.shift)
        else:
            kpoints = Kpoints.automatic_density(structure, input.kppa, force_gamma=input.force_gamma)
        mesh = [int(value) for value in kpoints.kpts[0]]
        shift = [int(value) for value in (kpoints.kpts_shift or input.shift)]
        payload = {
            "schema_version": "physicsos.kmesh.v1",
            "mesh": mesh,
            "shift": shift,
            "num_kpoints_full": int(math.prod(mesh)),
            "generation_policy": input.model_dump(),
        }
        path = _materials_dir(input.case_id) / "kmesh.json"
        _write_json(path, payload)
        return MaterialsToolOutput(artifact=_artifact(path, "kmesh"), data=payload)
    except Exception as exc:
        return MaterialsToolOutput(errors=[str(exc)])


class ReduceIrreducibleKpointsInput(StrictBaseModel):
    case_id: str
    structure_ref: str
    mesh: list[int]
    is_shift: list[int] = Field(default_factory=lambda: [0, 0, 0])
    symprec: float = 1e-5


def reduce_irreducible_kpoints(input: ReduceIrreducibleKpointsInput) -> MaterialsToolOutput:
    """Reduce a uniform reciprocal mesh to irreducible k-points with spglib."""
    try:
        _require_modules("spglib")
        import spglib

        structure = _structure_from_ref(input.structure_ref)
        mapping, grid = spglib.get_ir_reciprocal_mesh(input.mesh, _spglib_cell(structure), is_shift=input.is_shift, symprec=input.symprec)
        unique = sorted(set(int(item) for item in mapping))
        weights = [int(np.count_nonzero(mapping == item)) for item in unique]
        kpoints = [[float(coord) / float(size) for coord, size in zip(grid[item], input.mesh, strict=True)] for item in unique]
        payload = {
            "schema_version": "physicsos.irreducible_kpoints.v1",
            "mesh": input.mesh,
            "is_shift": input.is_shift,
            "ir_kpoints_frac": kpoints,
            "weights": weights,
            "mapping": [int(item) for item in mapping.tolist()],
            "num_ir_kpoints": len(unique),
        }
        path = _materials_dir(input.case_id) / "irreducible_kpoints.json"
        _write_json(path, payload)
        return MaterialsToolOutput(artifact=_artifact(path, "irreducible_kpoints"), data=payload)
    except Exception as exc:
        return MaterialsToolOutput(errors=[str(exc)])


class GenerateSeekpathKpathInput(StrictBaseModel):
    case_id: str
    structure_ref: str
    with_time_reversal: bool = True
    recipe: str = "hpkot"
    threshold: float = 1e-7
    symprec: float = 1e-5
    angle_tolerance: float = -1.0


def generate_seekpath_kpath(input: GenerateSeekpathKpathInput) -> MaterialsToolOutput:
    """Generate a HPKOT high-symmetry k-path with seekpath."""
    try:
        _require_modules("seekpath")
        import seekpath

        structure = _structure_from_ref(input.structure_ref)
        result = seekpath.get_path(
            _spglib_cell(structure),
            with_time_reversal=input.with_time_reversal,
            recipe=input.recipe,
            threshold=input.threshold,
            symprec=input.symprec,
            angle_tolerance=input.angle_tolerance,
        )
        payload = {"schema_version": "physicsos.kpath_seekpath.v1", **_json_safe(dict(result))}
        path = _materials_dir(input.case_id) / "kpath_seekpath.json"
        _write_json(path, payload)
        return MaterialsToolOutput(artifact=_artifact(path, "kpath_seekpath"), data=payload)
    except Exception as exc:
        return MaterialsToolOutput(errors=[str(exc)])


class GeneratePymatgenHighSymmKpathInput(StrictBaseModel):
    case_id: str
    structure_ref: str
    path_type: Literal["setyawan_curtarolo", "hinuma", "latimer_munro", "all"] = "setyawan_curtarolo"
    symprec: float = 1e-5
    angle_tolerance: float = -1.0


def generate_pymatgen_highsymm_kpath(input: GeneratePymatgenHighSymmKpathInput) -> MaterialsToolOutput:
    """Generate or cross-check a high-symmetry path with pymatgen."""
    try:
        _require_modules("pymatgen")
        from pymatgen.symmetry.bandstructure import HighSymmKpath

        structure = _structure_from_ref(input.structure_ref)
        path_type = "all" if input.path_type == "all" else input.path_type
        kpath = HighSymmKpath(structure, symprec=input.symprec, angle_tolerance=input.angle_tolerance, path_type=path_type)
        payload = {
            "schema_version": "physicsos.kpath_pymatgen.v1",
            "path_type": input.path_type,
            "kpath": _json_safe(kpath.kpath),
        }
        path = _materials_dir(input.case_id) / "kpath_pymatgen.json"
        _write_json(path, payload)
        return MaterialsToolOutput(artifact=_artifact(path, "kpath_pymatgen"), data=payload)
    except Exception as exc:
        return MaterialsToolOutput(errors=[str(exc)])


class SampleKpathSegmentsInput(StrictBaseModel):
    case_id: str
    kpath_ref: str
    points_per_segment: int = 40
    coordinate_mode: Literal["fractional_reciprocal"] = "fractional_reciprocal"


def sample_kpath_segments(input: SampleKpathSegmentsInput) -> MaterialsToolOutput:
    """Sample high-symmetry path segments into line-mode k-points."""
    try:
        payload = _read_json(input.kpath_ref)
        point_coords = payload.get("point_coords") or payload.get("kpath", {}).get("kpoints", {})
        path_segments = payload.get("path") or payload.get("kpath", {}).get("path", [])
        if not isinstance(point_coords, dict) or not isinstance(path_segments, list):
            raise ValueError("kpath_ref must contain point_coords/path or kpath.kpoints/kpath.path.")
        kpoints: list[list[float]] = []
        labels: list[str] = []
        segments: list[list[int]] = []
        cumulative: list[float] = []
        distance = 0.0
        for segment in path_segments:
            if len(segment) != 2:
                continue
            start_label, end_label = str(segment[0]), str(segment[1])
            start = np.array(point_coords[start_label], dtype=float)
            end = np.array(point_coords[end_label], dtype=float)
            seg_start_index = len(kpoints)
            for i in range(input.points_per_segment + 1):
                t = i / float(input.points_per_segment)
                point = (1 - t) * start + t * end
                if kpoints and np.allclose(point, np.array(kpoints[-1], dtype=float)):
                    continue
                if kpoints:
                    distance += float(np.linalg.norm(point - np.array(kpoints[-1], dtype=float)))
                kpoints.append([float(value) for value in point])
                labels.append(start_label if i == 0 else (end_label if i == input.points_per_segment else ""))
                cumulative.append(distance)
            segments.append([seg_start_index, len(kpoints) - 1])
        out = {
            "schema_version": "physicsos.line_kpoints.v1",
            "coordinate_mode": input.coordinate_mode,
            "kpoints": kpoints,
            "labels": labels,
            "segment_indices": segments,
            "cumulative_distances": cumulative,
        }
        path = _materials_dir(input.case_id) / "line_kpoints.json"
        _write_json(path, out)
        return MaterialsToolOutput(artifact=_artifact(path, "line_kpoints"), data=out)
    except Exception as exc:
        return MaterialsToolOutput(errors=[str(exc)])


class BuildTapsKpointAxisInput(StrictBaseModel):
    case_id: str
    kmesh_ref: str | None = None
    kpath_ref: str | None = None
    axis_type: Literal["uniform_integration", "line_band_path"] = "uniform_integration"
    rank_policy: dict[str, Any] = Field(default_factory=lambda: {"initial_rank": 4, "max_rank": 32})


def build_taps_kpoint_axis(input: BuildTapsKpointAxisInput) -> MaterialsToolOutput:
    """Convert kmesh or kpath artifacts into a KS-DFT-TAPS reciprocal axis descriptor."""
    try:
        if input.axis_type == "uniform_integration":
            if not input.kmesh_ref:
                raise ValueError("kmesh_ref is required for uniform_integration.")
            source = _read_json(input.kmesh_ref)
            points = int(source.get("num_ir_kpoints") or source.get("num_kpoints_full") or 0)
            weights = source.get("weights", [])
        else:
            if not input.kpath_ref:
                raise ValueError("kpath_ref is required for line_band_path.")
            source = _read_json(input.kpath_ref)
            points = len(source.get("kpoints", []))
            weights = []
        payload = {
            "schema_version": "physicsos.taps_kpoint_axis.v1",
            "axis_name": "kpoint",
            "axis_type": input.axis_type,
            "points": points,
            "weights": weights,
            "rank_policy": input.rank_policy,
            "source_ref": input.kmesh_ref or input.kpath_ref,
        }
        path = _materials_dir(input.case_id) / "taps_kpoint_axis.json"
        _write_json(path, payload)
        return MaterialsToolOutput(artifact=_artifact(path, "taps_kpoint_axis"), data=payload)
    except Exception as exc:
        return MaterialsToolOutput(errors=[str(exc)])


class MakeSupercellStructureInput(StrictBaseModel):
    case_id: str
    structure_ref: str
    scaling_matrix: list[list[int]]


def make_supercell_structure(input: MakeSupercellStructureInput) -> MaterialsToolOutput:
    """Generate a supercell structure artifact."""
    try:
        structure = _structure_from_ref(input.structure_ref)
        supercell = structure.copy()
        supercell.make_supercell(input.scaling_matrix)
        path = _materials_dir(input.case_id) / "structure_supercell.json"
        payload = _write_structure(path, supercell, source=input.structure_ref)
        payload["scaling_matrix"] = input.scaling_matrix
        _write_json(path, payload)
        return MaterialsToolOutput(artifact=_artifact(path, "material_structure"), data=payload)
    except Exception as exc:
        return MaterialsToolOutput(errors=[str(exc)])


class GenerateStructureParameterAxisInput(StrictBaseModel):
    case_id: str
    structure_ref: str
    parameter_type: Literal["volume_scale", "strain", "site_displacement", "lattice_parameter"]
    values: list[float]
    target_sites: list[int] = Field(default_factory=list)


def generate_structure_parameter_axis(input: GenerateStructureParameterAxisInput) -> MaterialsToolOutput:
    """Generate a family of structures for a TAPS structure parameter axis."""
    try:
        structure = _structure_from_ref(input.structure_ref)
        mats_dir = _materials_dir(input.case_id)
        structures = []
        for index, value in enumerate(input.values):
            item = structure.copy()
            if input.parameter_type == "volume_scale":
                item.scale_lattice(float(structure.volume) * float(value))
            elif input.parameter_type == "lattice_parameter":
                item.scale_lattice(float(value) ** 3)
            elif input.parameter_type == "strain":
                item.apply_strain(float(value))
            elif input.parameter_type == "site_displacement":
                for site_index in input.target_sites:
                    item.translate_sites([site_index], [float(value), 0.0, 0.0], frac_coords=False)
            path = mats_dir / f"structure_parameter_{index:03d}.json"
            _write_structure(path, item, source=input.structure_ref)
            structures.append({"value": value, "structure_ref": to_agent_path(path, workspace=_workspace())})
        payload = {
            "schema_version": "physicsos.structure_parameter_axis.v1",
            "parameter_type": input.parameter_type,
            "structures": structures,
        }
        axis_path = mats_dir / "structure_parameter_axis.json"
        _write_json(axis_path, payload)
        return MaterialsToolOutput(artifact=_artifact(axis_path, "structure_parameter_axis"), data=payload)
    except Exception as exc:
        return MaterialsToolOutput(errors=[str(exc)])


class MapSitePropertiesInput(StrictBaseModel):
    case_id: str
    source_structure_ref: str
    target_structure_ref: str
    properties: list[str] = Field(default_factory=lambda: ["magmom", "selective_dynamics", "label"])


def map_site_properties(input: MapSitePropertiesInput) -> MaterialsToolOutput:
    """Map simple site properties by site index when structures have matching site counts."""
    try:
        source = _structure_from_ref(input.source_structure_ref)
        target = _structure_from_ref(input.target_structure_ref).copy()
        if len(source) != len(target):
            raise ValueError("Site property mapping requires matching site counts in this MVP tool.")
        mapped = []
        for prop in input.properties:
            if prop in source.site_properties:
                target.add_site_property(prop, source.site_properties[prop])
                mapped.append(prop)
        path = _materials_dir(input.case_id) / "structure_mapped_site_properties.json"
        payload = _write_structure(path, target, source=input.target_structure_ref)
        payload["mapped_properties"] = mapped
        _write_json(path, payload)
        return MaterialsToolOutput(artifact=_artifact(path, "material_structure"), data=payload)
    except Exception as exc:
        return MaterialsToolOutput(errors=[str(exc)])


class ParseMolecularStructureInput(StrictBaseModel):
    case_id: str
    source_path: str | None = None
    source_text: str | None = None
    format: Literal["xyz", "sdf", "mol2", "pdb", "json", "auto"] = "auto"
    charge: int = 0
    multiplicity: int = 1


def parse_molecular_structure(input: ParseMolecularStructureInput) -> MaterialsToolOutput:
    """Parse molecule/cluster coordinates into a case-local molecular artifact."""
    try:
        if input.source_path:
            source = resolve_workspace_path(input.source_path, workspace=_workspace(), must_be_within_workspace=False)
            text = source.read_text(encoding="utf-8")
            source_label = str(source)
            fmt = input.format
            if fmt == "auto":
                suffix = source.suffix.lower().removeprefix(".")
                fmt = suffix if suffix in {"xyz", "sdf", "mol2", "pdb", "json"} else "xyz"
        elif input.source_text:
            text = input.source_text
            source_label = "source_text"
            fmt = "xyz" if input.format == "auto" else input.format
        else:
            return MaterialsToolOutput(errors=["Either source_path or source_text is required."])
        if fmt == "xyz":
            atoms, comment = _parse_xyz_molecule(text)
            parsed_payload: dict[str, Any] = {"xyz_comment": comment}
        elif fmt == "json":
            source_payload = json.loads(text)
            if not isinstance(source_payload, dict):
                raise ValueError("Molecular JSON input must be an object.")
            atoms = _parse_json_molecule(source_payload)
            parsed_payload = {"source_schema_version": source_payload.get("schema_version")}
            if "charge" in source_payload and input.charge == 0:
                parsed_payload["source_charge"] = source_payload["charge"]
            if "multiplicity" in source_payload and input.multiplicity == 1:
                parsed_payload["source_multiplicity"] = source_payload["multiplicity"]
        else:
            return MaterialsToolOutput(
                errors=[
                    f"Molecular format `{fmt}` requires an external chemistry parser not wired into this minimal tool path. "
                    "Convert to XYZ/JSON or add a case-local parser before continuing."
                ]
            )
        if not atoms:
            raise ValueError("No atoms were parsed.")
        charge = int(parsed_payload.get("source_charge", input.charge))
        multiplicity = int(parsed_payload.get("source_multiplicity", input.multiplicity))
        if multiplicity < 1:
            raise ValueError("Multiplicity must be >= 1.")
        payload = {
            "schema_version": "physicsos.molecular_structure.v1",
            "case_id": input.case_id,
            "source": source_label,
            "format": fmt,
            "charge": charge,
            "multiplicity": multiplicity,
            "num_atoms": len(atoms),
            "formula": _molecular_formula(atoms),
            "atoms": atoms,
            "bounding_box": _molecule_bounds(atoms),
            "provenance": parsed_payload,
            "llm_notes": [
                "This is a molecular/open-boundary structure artifact, not a periodic crystal structure.",
                "Charge and multiplicity are explicit inputs for downstream KS-DFT assumptions.",
            ],
        }
        path = _materials_dir(input.case_id) / "molecule.json"
        _write_json(path, payload)
        xyz_path = _materials_dir(input.case_id) / "molecule.xyz"
        xyz_lines = [str(len(atoms)), f"charge={charge} multiplicity={multiplicity} formula={payload['formula']}"]
        xyz_lines.extend(f"{atom['element']} {atom['xyz_angstrom'][0]:.12g} {atom['xyz_angstrom'][1]:.12g} {atom['xyz_angstrom'][2]:.12g}" for atom in atoms)
        xyz_path.write_text("\n".join(xyz_lines) + "\n", encoding="utf-8")
        return MaterialsToolOutput(
            artifact=_artifact(path, "molecular_structure"),
            artifacts={"json": _artifact(path, "molecular_structure"), "xyz": _artifact(xyz_path, "molecular_structure_xyz")},
            data=payload,
        )
    except Exception as exc:
        return MaterialsToolOutput(errors=[str(exc)])


class PrepareKSDftMolecularContextInput(StrictBaseModel):
    case_id: str
    molecule_ref: str
    boundary_policy: Literal["isolated", "vacuum_box", "multipole_open", "llm_select"] = "llm_select"
    vacuum_padding_angstrom: float | None = None
    poisson_boundary: Literal["open", "multipole", "coulomb_cutoff", "vacuum_periodic_box", "llm_select"] = "llm_select"
    charge_model: Literal["explicit_total_charge", "fragment_charges", "llm_select"] = "explicit_total_charge"


def prepare_ks_dft_molecular_context(input: PrepareKSDftMolecularContextInput) -> MaterialsToolOutput:
    """Package molecular inputs and open-boundary constraints for LLM-driven KS-DFT-TAPS."""
    try:
        molecule = _read_json(input.molecule_ref)
        atoms = molecule.get("atoms")
        if molecule.get("schema_version") != "physicsos.molecular_structure.v1" or not isinstance(atoms, list) or not atoms:
            raise ValueError("molecule_ref must point to a physicsos.molecular_structure.v1 artifact with atoms.")
        warnings: list[str] = []
        molecule_path = resolve_workspace_path(input.molecule_ref, workspace=_workspace(), must_be_within_workspace=False)
        if not molecule_path.exists():
            warnings.append(f"Referenced molecule artifact does not exist: {input.molecule_ref}")
        boundary_requires_llm = input.boundary_policy == "llm_select" or input.poisson_boundary == "llm_select"
        if input.boundary_policy == "vacuum_box" and input.vacuum_padding_angstrom is None:
            warnings.append("vacuum_box boundary was requested without vacuum_padding_angstrom; LLM must choose and record padding before execution.")
        payload = {
            "schema_version": "physicsos.ks_dft_molecular_context.v1",
            "case_id": input.case_id,
            "system_type": "molecule_or_cluster",
            "molecule_ref": input.molecule_ref,
            "formula": molecule.get("formula"),
            "num_atoms": molecule.get("num_atoms"),
            "charge": molecule.get("charge"),
            "multiplicity": molecule.get("multiplicity"),
            "bounding_box": molecule.get("bounding_box"),
            "boundary_policy": {
                "selected": input.boundary_policy,
                "poisson_boundary": input.poisson_boundary,
                "vacuum_padding_angstrom": input.vacuum_padding_angstrom,
                "charge_model": input.charge_model,
                "llm_must_finalize": boundary_requires_llm,
            },
            "fixed_inputs": [
                "atom elements and Cartesian coordinates from molecule.json",
                "explicit total charge",
                "explicit spin multiplicity",
                "molecular/open-boundary system type",
            ],
            "llm_selectable_items": [
                "open-boundary Poisson strategy",
                "vacuum-box dimensions when a periodic embedding is intentionally chosen",
                "basis/grid representation",
                "fragment partition and locality assumptions",
                "SCF/eigensolver/mixing policies",
            ],
            "fail_closed_rules": [
                "do not use periodic crystal kmesh or high-symmetry kpath artifacts for this molecular context",
                "do not use the Gamma-only periodic reference kernel unless the LLM explicitly creates and records a vacuum-box embedding policy",
                "do not infer charge, multiplicity, or boundary conditions silently",
                "do not claim isolated Poisson behavior from a periodic solver without a recorded correction/cutoff/multipole policy",
            ],
            "warnings": warnings,
        }
        mats_dir = _materials_dir(input.case_id)
        json_path = mats_dir / "ks_dft_molecular_context.json"
        md_path = mats_dir / "ks_dft_molecular_context.md"
        _write_json(json_path, payload)
        lines = [
            "# KS-DFT Molecular Context",
            "",
            "Use this context when the case is a molecule or cluster rather than a periodic crystal.",
            "",
            f"- `molecule_ref`: `{input.molecule_ref}`",
            f"- `formula`: `{payload['formula']}`",
            f"- `num_atoms`: `{payload['num_atoms']}`",
            f"- `charge`: `{payload['charge']}`",
            f"- `multiplicity`: `{payload['multiplicity']}`",
            "",
            "## Boundary Policy",
            "",
            f"- selected: `{input.boundary_policy}`",
            f"- Poisson boundary: `{input.poisson_boundary}`",
            f"- vacuum padding Angstrom: `{input.vacuum_padding_angstrom}`",
            "",
            "## LLM Responsibilities",
            "",
            "- Finalize the numerical boundary strategy before execution.",
            "- Choose representation, parameters, and case-local kernel code from the derivation and artifacts.",
            "- Record the final policy in runtime metadata.",
            "",
            "## Fail-Closed Rules",
            "",
        ]
        lines.extend(f"- {rule}" for rule in payload["fail_closed_rules"])
        if warnings:
            lines.extend(["", "## Warnings", "", *(f"- {warning}" for warning in warnings)])
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return MaterialsToolOutput(
            artifacts={"context": _artifact(md_path, "ks_dft_molecular_context"), "json": _artifact(json_path, "ks_dft_molecular_context_manifest")},
            data=payload,
            warnings=warnings,
        )
    except Exception as exc:
        return MaterialsToolOutput(errors=[str(exc)])


class PrepareMolecularTapsScalingPolicyInput(StrictBaseModel):
    case_id: str
    molecular_context_ref: str
    target_scale: Literal["small", "medium", "large", "very_large", "llm_select"] = "llm_select"
    candidate_strategies: list[
        Literal[
            "localized_orbitals",
            "density_matrix_truncation",
            "fragment_partition",
            "near_field_far_field_coulomb",
            "atom_centered_grid",
            "adaptive_grid_axes",
            "hierarchical_taps_axes",
        ]
    ] = Field(
        default_factory=lambda: [
            "localized_orbitals",
            "density_matrix_truncation",
            "fragment_partition",
            "near_field_far_field_coulomb",
            "atom_centered_grid",
            "adaptive_grid_axes",
            "hierarchical_taps_axes",
        ]
    )
    require_llm_selection: bool = True


def prepare_molecular_taps_scaling_policy(input: PrepareMolecularTapsScalingPolicyInput) -> MaterialsToolOutput:
    """Write a molecular TAPS scaling strategy contract for LLM selection."""
    try:
        context = _read_json(input.molecular_context_ref)
        if context.get("schema_version") != "physicsos.ks_dft_molecular_context.v1":
            raise ValueError("molecular_context_ref must point to a physicsos.ks_dft_molecular_context.v1 artifact.")
        strategies = list(dict.fromkeys(str(item) for item in input.candidate_strategies))
        payload = {
            "schema_version": "physicsos.molecular_taps_scaling_policy.v1",
            "case_id": input.case_id,
            "molecular_context_ref": input.molecular_context_ref,
            "target_scale": input.target_scale,
            "llm_driven": True,
            "require_llm_selection": input.require_llm_selection,
            "candidate_strategies": strategies,
            "strategy_contracts": {
                "localized_orbitals": {
                    "llm_must_choose": ["localization criterion", "localization radius/cutoff", "orthogonality maintenance"],
                    "record_runtime_keys": ["localized_basis_policy", "overlap_policy", "localization_error_estimate"],
                },
                "density_matrix_truncation": {
                    "llm_must_choose": ["truncation metric", "cutoff schedule", "idempotency/error checks"],
                    "record_runtime_keys": ["density_matrix_truncation_policy", "nearsightedness_assumptions"],
                },
                "fragment_partition": {
                    "llm_must_choose": ["fragment definition", "buffer region", "fragment coupling policy"],
                    "record_runtime_keys": ["fragment_partition", "fragment_charge_consistency"],
                },
                "near_field_far_field_coulomb": {
                    "llm_must_choose": ["near-field exact region", "far-field approximation", "multipole/cutoff order"],
                    "record_runtime_keys": ["coulomb_decomposition_policy", "poisson_boundary_evidence"],
                },
                "atom_centered_grid": {
                    "llm_must_choose": ["radial/angular quadrature", "partition of unity", "grid pruning policy"],
                    "record_runtime_keys": ["atom_centered_grid_policy", "quadrature_error_checks"],
                },
                "adaptive_grid_axes": {
                    "llm_must_choose": ["refinement indicator", "coarsening rule", "basis transfer/checkpointing"],
                    "record_runtime_keys": ["adaptive_grid_policy", "grid_refinement_history"],
                },
                "hierarchical_taps_axes": {
                    "llm_must_choose": ["axis decomposition", "rank adaptation", "cross-axis coupling terms"],
                    "record_runtime_keys": ["taps_axis_policy", "rank_adaptation_history"],
                },
            },
            "disallowed_hidden_defaults": [
                "fixed cutoff radii without runtime metadata",
                "fixed fragment size without derivation or user/artifact support",
                "periodic Gamma-only kernel reuse without explicit vacuum-box policy",
                "large-system linear-scaling claims without fragment/locality verification",
            ],
            "required_verification": [
                "charge and spin consistency",
                "orbital orthonormality under the chosen overlap",
                "SCF residual",
                "isolated Poisson or multipole/cutoff boundary evidence",
                "fragment charge consistency when fragment_partition is used",
                "rank/grid/locality sensitivity for the selected strategy",
            ],
        }
        taps_dir = _case_dir(input.case_id) / "taps"
        taps_dir.mkdir(parents=True, exist_ok=True)
        json_path = taps_dir / "molecular_taps_scaling_policy.json"
        md_path = taps_dir / "molecular_taps_scaling_policy.md"
        _write_json(json_path, payload)
        lines = [
            "# Molecular TAPS Scaling Policy",
            "",
            "This is a strategy contract for LLM-driven molecular KS-DFT-TAPS kernels. It does not select hidden numerical defaults.",
            "",
            f"- `molecular_context_ref`: `{input.molecular_context_ref}`",
            f"- `target_scale`: `{input.target_scale}`",
            "",
            "## Candidate Strategies",
            "",
        ]
        lines.extend(f"- `{strategy}`" for strategy in strategies)
        lines.extend(
            [
                "",
                "## Required Runtime Records",
                "",
                "- final selected strategy family",
                "- selected parameters and why they are valid for this molecule/cluster",
                "- boundary/Poisson policy",
                "- locality, fragment, rank, and grid verification evidence when used",
                "",
                "## Disallowed Hidden Defaults",
                "",
            ]
        )
        lines.extend(f"- {item}" for item in payload["disallowed_hidden_defaults"])
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return MaterialsToolOutput(
            artifacts={"policy": _artifact(json_path, "molecular_taps_scaling_policy"), "notes": _artifact(md_path, "molecular_taps_scaling_policy_notes")},
            data=payload,
        )
    except Exception as exc:
        return MaterialsToolOutput(errors=[str(exc)])


class PrepareKSDftTapsMaterialContextInput(StrictBaseModel):
    case_id: str
    standardized_structure_ref: str
    symmetry_ref: str
    reciprocal_lattice_ref: str
    kmesh_ref: str
    irreducible_kpoints_ref: str
    kpath_ref: str | None = None
    taps_kpoint_axis_ref: str | None = None


def prepare_ks_dft_taps_material_context(input: PrepareKSDftTapsMaterialContextInput) -> MaterialsToolOutput:
    """Package materials artifacts into a derivation-ready KS-DFT-TAPS context."""
    try:
        refs = {
            "standardized_structure_ref": input.standardized_structure_ref,
            "symmetry_ref": input.symmetry_ref,
            "reciprocal_lattice_ref": input.reciprocal_lattice_ref,
            "kmesh_ref": input.kmesh_ref,
            "irreducible_kpoints_ref": input.irreducible_kpoints_ref,
            "kpath_ref": input.kpath_ref,
            "taps_kpoint_axis_ref": input.taps_kpoint_axis_ref,
        }
        warnings = [f"Missing optional reference: {key}" for key, value in refs.items() if value is None and key in {"kpath_ref", "taps_kpoint_axis_ref"}]
        for key, value in refs.items():
            if value is None:
                continue
            path = resolve_workspace_path(value, workspace=_workspace(), must_be_within_workspace=False)
            if not path.exists():
                warnings.append(f"Referenced artifact does not exist: {key}={value}")
        payload = {
            "schema_version": "physicsos.ks_dft_material_context.v1",
            "case_id": input.case_id,
            "refs": refs,
            "fixed_inputs": [
                "standardized structure",
                "reciprocal lattice convention",
                "symmetry dataset",
                "irreducible k-points and weights",
                "high-symmetry labels and segments when kpath_ref is present",
            ],
            "do_not_recompute_in_derivation": [
                "primitive/conventional transform",
                "space group",
                "k-path labels",
                "k-point weights",
            ],
            "warnings": warnings,
        }
        mats_dir = _materials_dir(input.case_id)
        json_path = mats_dir / "ks_dft_material_context.json"
        md_path = mats_dir / "ks_dft_material_context.md"
        _write_json(json_path, payload)
        lines = [
            "# KS-DFT-TAPS Material Context",
            "",
            "Use these artifacts as fixed material inputs for the Kohn-Sham TAPS derivation.",
            "",
            "## Artifact Refs",
            "",
        ]
        lines.extend(f"- `{key}`: `{value}`" for key, value in refs.items() if value is not None)
        lines.extend(
            [
                "",
                "## Hard Rules",
                "",
                "- Use `structure_standardized.json` as the only structure source.",
                "- Use `kmesh.json` / `irreducible_kpoints.json` for Brillouin-zone integration.",
                "- Use `kpath_seekpath.json` only for line-mode band path after SCF verification.",
                "- Do not invent or recompute space group, k-point labels, k-point weights, or reciprocal lattice convention.",
                "- If required material artifacts are missing, stop and request `materials-preprocess-agent`.",
                "",
            ]
        )
        if warnings:
            lines.extend(["## Warnings", "", *(f"- {warning}" for warning in warnings), ""])
        md_path.write_text("\n".join(lines), encoding="utf-8")
        return MaterialsToolOutput(
            artifacts={
                "context": _artifact(md_path, "ks_dft_material_context"),
                "json": _artifact(json_path, "ks_dft_material_context_manifest"),
            },
            data=payload,
            warnings=warnings,
        )
    except Exception as exc:
        return MaterialsToolOutput(errors=[str(exc)])


class ReviewKSDftMaterialContextInput(StrictBaseModel):
    case_id: str
    context_ref: str


def review_ks_dft_material_context(input: ReviewKSDftMaterialContextInput) -> MaterialsToolOutput:
    """Check whether material context is complete enough for KS-DFT-TAPS derivation."""
    try:
        path = resolve_workspace_path(input.context_ref, workspace=_workspace(), must_be_within_workspace=False)
        if path.suffix == ".md":
            json_path = path.with_suffix(".json")
        else:
            json_path = path
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        refs = payload.get("refs", {}) if isinstance(payload, dict) else {}
        required = ["standardized_structure_ref", "symmetry_ref", "reciprocal_lattice_ref", "kmesh_ref", "irreducible_kpoints_ref"]
        missing = [key for key in required if not refs.get(key)]
        warnings = list(payload.get("warnings", [])) if isinstance(payload.get("warnings", []), list) else []
        for key in required:
            value = refs.get(key)
            if value:
                artifact_path = resolve_workspace_path(value, workspace=_workspace(), must_be_within_workspace=False)
                if not artifact_path.exists():
                    missing.append(key)
        review = {
            "schema_version": "physicsos.ks_dft_material_context_review.v1",
            "ready_for_derivation": not missing,
            "missing": sorted(set(missing)),
            "warnings": warnings,
            "required_user_questions": [] if not missing else ["Run materials-preprocess-agent to create missing material artifacts."],
        }
        out_path = _materials_dir(input.case_id) / "ks_dft_material_context_review.json"
        _write_json(out_path, review)
        return MaterialsToolOutput(artifact=_artifact(out_path, "ks_dft_material_context_review"), data=review, warnings=warnings)
    except Exception as exc:
        return MaterialsToolOutput(errors=[str(exc)])


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


MATERIALS_TOOL_SPECS = [
    (parse_material_structure, ParseMaterialStructureInput, MaterialsToolOutput),
    (write_material_structure, WriteMaterialStructureInput, MaterialsToolOutput),
    (validate_material_structure, ValidateMaterialStructureInput, MaterialsToolOutput),
    (analyze_spacegroup, AnalyzeSpacegroupInput, MaterialsToolOutput),
    (standardize_crystal_structure, StandardizeCrystalStructureInput, MaterialsToolOutput),
    (compare_crystal_structures, CompareCrystalStructuresInput, MaterialsToolOutput),
    (reduce_lattice_cell, ReduceLatticeCellInput, MaterialsToolOutput),
    (compute_reciprocal_lattice, ComputeReciprocalLatticeInput, MaterialsToolOutput),
    (generate_uniform_kmesh, GenerateUniformKmeshInput, MaterialsToolOutput),
    (reduce_irreducible_kpoints, ReduceIrreducibleKpointsInput, MaterialsToolOutput),
    (generate_seekpath_kpath, GenerateSeekpathKpathInput, MaterialsToolOutput),
    (generate_pymatgen_highsymm_kpath, GeneratePymatgenHighSymmKpathInput, MaterialsToolOutput),
    (sample_kpath_segments, SampleKpathSegmentsInput, MaterialsToolOutput),
    (build_taps_kpoint_axis, BuildTapsKpointAxisInput, MaterialsToolOutput),
    (make_supercell_structure, MakeSupercellStructureInput, MaterialsToolOutput),
    (generate_structure_parameter_axis, GenerateStructureParameterAxisInput, MaterialsToolOutput),
    (map_site_properties, MapSitePropertiesInput, MaterialsToolOutput),
    (parse_molecular_structure, ParseMolecularStructureInput, MaterialsToolOutput),
    (prepare_ks_dft_molecular_context, PrepareKSDftMolecularContextInput, MaterialsToolOutput),
    (prepare_molecular_taps_scaling_policy, PrepareMolecularTapsScalingPolicyInput, MaterialsToolOutput),
    (prepare_ks_dft_taps_material_context, PrepareKSDftTapsMaterialContextInput, MaterialsToolOutput),
    (review_ks_dft_material_context, ReviewKSDftMaterialContextInput, MaterialsToolOutput),
]

for _tool, _input, _output in MATERIALS_TOOL_SPECS:
    _tool.input_model = _input
    _tool.output_model = _output
    _tool.side_effects = "workspace artifacts only"
    _tool.requires_approval = False
