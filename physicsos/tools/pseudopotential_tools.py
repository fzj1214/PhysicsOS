from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from physicsos.config import load_config, runtime_paths
from physicsos.paths import resolve_workspace_path, to_agent_path
from physicsos.schemas.common import ArtifactRef, StrictBaseModel
from physicsos.tools.case_tools import _append_event, _case_dir


def _workspace() -> Path:
    return runtime_paths().workspace


def _pseudopotential_dir(case_id: str) -> Path:
    path = _case_dir(case_id) / "pseudopotentials"
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


def _path_text(path: Path) -> str:
    return str(path).replace("\\", "/")


def _configured_library_root(library_id: str | None = None) -> tuple[str | None, str | None, str | None]:
    import os

    env_root = os.environ.get("PHYSICSOS_PSEUDOPOTENTIAL_DIR")
    if env_root:
        return env_root, "env:PHYSICSOS_PSEUDOPOTENTIAL_DIR", library_id
    config = load_config(create=True)
    section = config.get("pseudopotentials", {})
    if not isinstance(section, dict):
        return None, None, library_id
    selected_id = library_id or section.get("default_library_id")
    libraries = section.get("libraries", {})
    if not isinstance(selected_id, str) or not isinstance(libraries, dict):
        return None, None, library_id
    entry = libraries.get(selected_id, {})
    if not isinstance(entry, dict):
        return None, None, selected_id
    root = entry.get("root")
    if isinstance(root, str) and root:
        return root, f"config:pseudopotentials.libraries.{selected_id}.root", selected_id
    return None, f"config:pseudopotentials.libraries.{selected_id}.root", selected_id


def _read_json(path_or_uri: str | Path) -> dict[str, Any]:
    path = resolve_workspace_path(path_or_uri, workspace=_workspace(), must_be_within_workspace=False)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path_or_uri}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_text_prefix(path: Path, limit_bytes: int = 65536) -> str:
    with path.open("rb") as handle:
        data = handle.read(limit_bytes)
    return data.decode("latin-1", errors="replace")


def _match_float(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def _match_str(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else None


def _parse_vasp_potcar(potcar: Path) -> dict[str, Any]:
    prefix = _read_text_prefix(potcar)
    title = _match_str(prefix, r"^\s*TITEL\s*=\s*(.+)$") or prefix.splitlines()[0].strip()
    vrhfin = _match_str(prefix, r"VRHFIN\s*=\s*([A-Za-z]+)\s*:")
    title_parts = title.split()
    variant = potcar.parent.name
    element = vrhfin or re.match(r"([A-Z][a-z]?)", variant).group(1)  # type: ignore[union-attr]
    return {
        "source_format": "vasp_paw_potcar",
        "functional": "PBE" if "PBE" in title.upper() else None,
        "dataset_family": title_parts[0] if title_parts else None,
        "title": title,
        "element": element,
        "variant": variant,
        "lexch": _match_str(prefix, r"LEXCH\s*=\s*([A-Za-z0-9_+-]+)"),
        "zval": _match_float(prefix, r"ZVAL\s*=\s*([-+0-9.Ee]+)"),
        "enmax_eV": _match_float(prefix, r"ENMAX\s*=\s*([-+0-9.Ee]+)"),
        "enmin_eV": _match_float(prefix, r"ENMIN\s*=\s*([-+0-9.Ee]+)"),
        "pomass": _match_float(prefix, r"POMASS\s*=\s*([-+0-9.Ee]+)"),
        "rcore_bohr": _match_float(prefix, r"RCORE\s*=\s*([-+0-9.Ee]+)"),
        "rpacor_bohr": _match_float(prefix, r"RPACOR\s*=\s*([-+0-9.Ee]+)"),
        "lpaW": (_match_str(prefix, r"LPAW\s*=\s*([TF])") or "").upper() == "T",
        "potcar_path": _path_text(potcar),
        "psctr_path": _path_text(potcar.parent / "PSCTR") if (potcar.parent / "PSCTR").exists() else None,
        "potcar_sha256": _sha256(potcar),
        "potcar_size_bytes": potcar.stat().st_size,
    }


def _species_from_structure_ref(structure_ref: str) -> list[str]:
    payload = _read_json(structure_ref)
    species = payload.get("species")
    if isinstance(species, list) and species:
        return [str(item) for item in species]
    pmg = payload.get("pymatgen_structure")
    if isinstance(pmg, dict):
        out = []
        for site in pmg.get("sites", []):
            if not isinstance(site, dict):
                continue
            label = site.get("label")
            if isinstance(label, str):
                out.append(label)
                continue
            species_list = site.get("species")
            if isinstance(species_list, list) and species_list and isinstance(species_list[0], dict):
                element = species_list[0].get("element")
                if isinstance(element, str):
                    out.append(element)
        if out:
            return out
    raise ValueError(f"Could not determine species from {structure_ref}")


def _variant_score(entry: dict[str, Any], preference: str, allow_gw: bool, allow_hard_soft: bool) -> tuple[int, int, str]:
    variant = str(entry.get("variant") or "")
    element = str(entry.get("element") or "")
    suffix = variant.removeprefix(element)
    if not allow_gw and "_GW" in variant:
        return (10_000, len(variant), variant)
    if not allow_hard_soft and (suffix.startswith("_h") or suffix.startswith("_s") or "_AE" in variant or "." in variant):
        return (10_000, len(variant), variant)
    if preference == "standard":
        primary = 0 if variant == element else 20
    elif preference == "pv":
        primary = 0 if suffix.startswith("_pv") else 10 if variant == element else 20
    elif preference == "sv":
        primary = 0 if suffix.startswith("_sv") else 10 if suffix.startswith("_pv") else 20 if variant == element else 30
    else:
        primary = 0
    return (primary, len(variant), variant)


class PseudopotentialToolOutput(StrictBaseModel):
    artifact: ArtifactRef | None = None
    artifacts: dict[str, ArtifactRef] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class IndexVaspPawPbeLibraryInput(StrictBaseModel):
    case_id: str
    library_root: str | None = None
    library_id: str | None = None
    max_entries: int | None = None


def index_vasp_paw_pbe_library(input: IndexVaspPawPbeLibraryInput) -> PseudopotentialToolOutput:
    """Index a local VASP PAW PBE POTCAR library without copying POTCAR contents."""
    try:
        library_root = input.library_root
        root_source = "input.library_root" if library_root else None
        library_id = input.library_id or "vasp-paw-pbe"
        if not library_root:
            library_root, root_source, configured_id = _configured_library_root(input.library_id)
            library_id = configured_id or library_id
        if not library_root:
            raise ValueError(
                "No pseudopotential library root configured. Provide library_root, set PHYSICSOS_PSEUDOPOTENTIAL_DIR, "
                "or set pseudopotentials.libraries.<id>.root in ~/.physicsos/config.json."
            )
        root = Path(library_root).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"Pseudopotential library root does not exist: {root}")
        entries = []
        warnings: list[str] = []
        for potcar in sorted(root.glob("*/POTCAR")):
            try:
                entries.append(_parse_vasp_potcar(potcar))
            except Exception as exc:
                warnings.append(f"Could not parse {potcar}: {exc}")
            if input.max_entries is not None and len(entries) >= input.max_entries:
                break
        by_element: dict[str, int] = {}
        for entry in entries:
            element = str(entry.get("element") or "")
            by_element[element] = by_element.get(element, 0) + 1
        payload = {
            "schema_version": "physicsos.pseudopotential_library_index.v1",
            "library_type": "vasp_paw_pbe",
            "library_id": library_id,
            "library_root": _path_text(root),
            "library_root_source": root_source,
            "source_archive": _path_text(root / "potpaw_PBE.54.tar.gz") if (root / "potpaw_PBE.54.tar.gz").exists() else None,
            "data_base": _path_text(root / "data_base") if (root / "data_base").exists() else None,
            "readme_updates": _path_text(root / "README.UPDATES") if (root / "README.UPDATES").exists() else None,
            "entry_count": len(entries),
            "elements": sorted(key for key in by_element if key),
            "variants_per_element": by_element,
            "entries": entries,
            "legal_note": "Index stores metadata and file hashes only; it does not copy or redistribute POTCAR contents.",
            "physicsos_usage": "Current Gamma-only local-pseudopotential kernel uses this library for valence electron counts, cutoff/provenance metadata, and selection policy. Full PAW/nonlocal projector Hamiltonian support is a separate implementation task.",
            "warnings": warnings,
        }
        path = _pseudopotential_dir(input.case_id) / "vasp_paw_pbe_index.json"
        _write_json(path, payload)
        _append_event(_case_dir(input.case_id), "index_vasp_paw_pbe_library", {"entries": len(entries)})
        return PseudopotentialToolOutput(
            artifact=_artifact(path, "pseudopotential_library_index", "Metadata-only index of local VASP PAW PBE POTCAR files."),
            data=payload,
            warnings=warnings,
        )
    except Exception as exc:
        return PseudopotentialToolOutput(errors=[str(exc)])


class SelectPseudopotentialsForStructureInput(StrictBaseModel):
    case_id: str
    structure_ref: str
    index_ref: str | None = None
    library_root: str | None = None
    library_id: str | None = None
    preference: Literal["standard", "pv", "sv", "any"] = "standard"
    variant_overrides: dict[str, str] = Field(default_factory=dict)
    allow_gw: bool = False
    allow_hard_soft: bool = False


def select_pseudopotentials_for_structure(input: SelectPseudopotentialsForStructureInput) -> PseudopotentialToolOutput:
    """Select VASP PAW PBE POTCAR entries for a standardized structure and write a KS-DFT pseudopotential context."""
    try:
        if input.index_ref:
            index = _read_json(input.index_ref)
        else:
            indexed = index_vasp_paw_pbe_library(
                IndexVaspPawPbeLibraryInput(case_id=input.case_id, library_root=input.library_root, library_id=input.library_id)
            )
            if indexed.errors:
                raise RuntimeError("; ".join(indexed.errors))
            index = indexed.data
        entries = [entry for entry in index.get("entries", []) if isinstance(entry, dict)]
        by_element: dict[str, list[dict[str, Any]]] = {}
        by_variant: dict[str, dict[str, Any]] = {}
        for entry in entries:
            element = str(entry.get("element") or "")
            variant = str(entry.get("variant") or "")
            by_element.setdefault(element, []).append(entry)
            by_variant[variant] = entry
        species = _species_from_structure_ref(input.structure_ref)
        counts = Counter(species)
        selected: dict[str, dict[str, Any]] = {}
        missing: list[str] = []
        warnings: list[str] = []
        for element in sorted(counts):
            override = input.variant_overrides.get(element)
            if override:
                entry = by_variant.get(override)
                if not entry:
                    missing.append(f"{element}:{override}")
                    continue
            else:
                candidates = by_element.get(element, [])
                if not candidates:
                    missing.append(element)
                    continue
                entry = sorted(candidates, key=lambda item: _variant_score(item, input.preference, input.allow_gw, input.allow_hard_soft))[0]
                if _variant_score(entry, input.preference, input.allow_gw, input.allow_hard_soft)[0] >= 10_000:
                    missing.append(element)
                    continue
            selected[element] = entry
        if missing:
            raise ValueError("Missing allowed pseudopotentials for: " + ", ".join(missing))
        total_valence = 0.0
        recommended_encut = 0.0
        for element, entry in selected.items():
            zval = float(entry["zval"])
            total_valence += float(counts[element]) * zval
            recommended_encut = max(recommended_encut, float(entry.get("enmax_eV") or 0.0))
        payload = {
            "schema_version": "physicsos.ks_dft_pseudopotential_context.v1",
            "library_type": index.get("library_type", "vasp_paw_pbe"),
            "library_id": index.get("library_id", input.library_id or "vasp-paw-pbe"),
            "library_root": index.get("library_root", input.library_root),
            "library_root_source": index.get("library_root_source"),
            "structure_ref": input.structure_ref,
            "species_counts": dict(counts),
            "selection_policy": {
                "preference": input.preference,
                "variant_overrides": input.variant_overrides,
                "allow_gw": input.allow_gw,
                "allow_hard_soft": input.allow_hard_soft,
            },
            "selected": selected,
            "total_valence_electrons": total_valence,
            "recommended_encut_eV": recommended_encut,
            "usable_in_current_kernel": "metadata_valence_and_provenance_only",
            "not_yet_used_for": ["PAW augmentation", "nonlocal projectors", "radial local potential interpolation"],
            "warnings": warnings,
        }
        json_path = _pseudopotential_dir(input.case_id) / "ks_dft_pseudopotential_context.json"
        md_path = _pseudopotential_dir(input.case_id) / "ks_dft_pseudopotential_context.md"
        _write_json(json_path, payload)
        lines = [
            "# KS-DFT Pseudopotential Context",
            "",
            f"- Library type: `{payload['library_type']}`",
            f"- Total valence electrons: `{total_valence}`",
            f"- Recommended ENMAX: `{recommended_encut}` eV",
            f"- Current kernel usage: `{payload['usable_in_current_kernel']}`",
            "",
            "## Selected Entries",
            "",
        ]
        for element, entry in selected.items():
            lines.append(
                f"- `{element}` -> `{entry['variant']}`; ZVAL `{entry['zval']}`; ENMAX `{entry['enmax_eV']}` eV; SHA256 `{entry['potcar_sha256']}`"
            )
        lines.extend(
            [
                "",
                "The POTCAR files are referenced by path and hash only. PhysicsOS does not copy POTCAR contents into case artifacts.",
            ]
        )
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        _append_event(_case_dir(input.case_id), "select_pseudopotentials_for_structure", {"elements": sorted(selected)})
        return PseudopotentialToolOutput(
            artifact=_artifact(json_path, "ks_dft_pseudopotential_context"),
            artifacts={"json": _artifact(json_path, "ks_dft_pseudopotential_context"), "markdown": _artifact(md_path, "ks_dft_pseudopotential_context_markdown")},
            data=payload,
            warnings=warnings,
        )
    except Exception as exc:
        return PseudopotentialToolOutput(errors=[str(exc)])


class ValidateLocalPseudopotentialArtifactInput(StrictBaseModel):
    case_id: str
    local_pseudopotential_ref: str | None = None
    structure_ref: str | None = None
    required_elements: list[str] = Field(default_factory=list)
    fail_closed: bool = True


def _default_standardized_structure_ref(case_id: str) -> str | None:
    context_path = _case_dir(case_id) / "materials" / "ks_dft_material_context.json"
    if not context_path.exists():
        return None
    context = _read_json(context_path)
    refs = context.get("refs", {})
    if isinstance(refs, dict) and isinstance(refs.get("standardized_structure_ref"), str):
        return str(refs["standardized_structure_ref"])
    return None


def _default_local_pseudopotential_path(case_id: str) -> Path:
    return _pseudopotential_dir(case_id) / "ks_dft_local_pseudopotential.json"


def _required_species(input: ValidateLocalPseudopotentialArtifactInput) -> list[str]:
    if input.required_elements:
        return sorted(set(input.required_elements))
    structure_ref = input.structure_ref or _default_standardized_structure_ref(input.case_id)
    if not structure_ref:
        raise ValueError("No structure_ref or materials/ks_dft_material_context.json standardized_structure_ref available.")
    return sorted(set(_species_from_structure_ref(structure_ref)))


def _number_list(value: Any) -> list[float] | None:
    if not isinstance(value, list) or not value:
        return None
    out = []
    for item in value:
        if not isinstance(item, (int, float)):
            return None
        number = float(item)
        if not math.isfinite(number):
            return None
        out.append(number)
    return out


def _validate_local_entry(element: str, entry: Any) -> tuple[bool, list[str], dict[str, Any]]:
    issues: list[str] = []
    if not isinstance(entry, dict):
        return False, [f"{element}: entry must be an object"], {}
    radial = entry.get("radial_grid")
    potential = entry.get("local_potential")
    if not isinstance(radial, dict):
        issues.append(f"{element}: radial_grid must be an object")
        radial_values = None
    else:
        radial_values = _number_list(radial.get("values"))
        if radial_values is None:
            issues.append(f"{element}: radial_grid.values must be a non-empty finite number list")
        if not isinstance(radial.get("unit"), str) or not radial.get("unit"):
            issues.append(f"{element}: radial_grid.unit is required")
    if radial_values is not None and any(b <= a for a, b in zip(radial_values, radial_values[1:])):
        issues.append(f"{element}: radial_grid.values must be strictly increasing")
    if not isinstance(potential, dict):
        issues.append(f"{element}: local_potential must be an object")
        potential_values = None
    else:
        potential_values = _number_list(potential.get("values"))
        if potential_values is None:
            issues.append(f"{element}: local_potential.values must be a non-empty finite number list")
        if not isinstance(potential.get("unit"), str) or not potential.get("unit"):
            issues.append(f"{element}: local_potential.unit is required")
    if radial_values is not None and potential_values is not None and len(radial_values) != len(potential_values):
        issues.append(f"{element}: radial_grid.values and local_potential.values lengths must match")
    interpolation = entry.get("interpolation")
    if not isinstance(interpolation, dict) or not isinstance(interpolation.get("method"), str):
        issues.append(f"{element}: interpolation.method is required")
    provenance = entry.get("provenance")
    if not isinstance(provenance, dict) or not provenance:
        issues.append(f"{element}: provenance object is required")
    version_hash = entry.get("version_hash") or entry.get("sha256") or entry.get("source_sha256")
    if not isinstance(version_hash, str) or not version_hash:
        issues.append(f"{element}: version_hash/sha256/source_sha256 is required")
    summary = {
        "grid_points": len(radial_values or []),
        "radial_unit": radial.get("unit") if isinstance(radial, dict) else None,
        "potential_unit": potential.get("unit") if isinstance(potential, dict) else None,
        "interpolation": interpolation if isinstance(interpolation, dict) else {},
        "version_hash": version_hash,
        "provenance": provenance if isinstance(provenance, dict) else {},
    }
    return not issues, issues, summary


def validate_local_pseudopotential_artifact(input: ValidateLocalPseudopotentialArtifactInput) -> PseudopotentialToolOutput:
    """Validate a case-local radial local-pseudopotential artifact without generating one."""
    try:
        required_elements = _required_species(input)
        source_ref = input.local_pseudopotential_ref
        source_path = resolve_workspace_path(source_ref, workspace=_workspace(), must_be_within_workspace=False) if source_ref else _default_local_pseudopotential_path(input.case_id)
        issues: list[str] = []
        element_summaries: dict[str, Any] = {}
        artifact_present = source_path.exists()
        if not artifact_present:
            issues.append(f"Missing local pseudopotential artifact: {to_agent_path(source_path, workspace=_workspace())}")
            source = {}
        else:
            source = _read_json(source_path)
            elements = source.get("elements")
            if not isinstance(elements, dict):
                issues.append("local pseudopotential artifact must contain an elements object")
                elements = {}
            for element in required_elements:
                if element not in elements:
                    issues.append(f"{element}: missing local pseudopotential entry")
                    continue
                ok, entry_issues, summary = _validate_local_entry(element, elements[element])
                if not ok:
                    issues.extend(entry_issues)
                element_summaries[element] = summary
        accepted = not issues
        payload = {
            "schema_version": "physicsos.ks_dft.local_pseudopotential_contract.v1",
            "case_id": input.case_id,
            "accepted": accepted,
            "fail_closed": input.fail_closed,
            "status": "accepted_validated_local_pseudopotential" if accepted else "blocked_missing_or_invalid_local_pseudopotential",
            "source_ref": to_agent_path(source_path, workspace=_workspace()) if source_path.is_relative_to(_workspace()) else _path_text(source_path),
            "required_elements": required_elements,
            "artifact_present": artifact_present,
            "representation_required": "radial local potential per element",
            "required_contract": {
                "radial_grid": "finite strictly increasing values plus unit",
                "local_potential": "finite values with same length as radial_grid plus unit",
                "interpolation": "explicit interpolation.method and boundary policy when applicable",
                "version_hash": "stable hash of the local-potential source or transformed artifact",
                "provenance": "non-empty source, generator, license, and transform metadata where available",
            },
            "element_summaries": element_summaries,
            "issues": issues,
            "kernel_instruction": (
                "Generated kernels may consume this artifact for a local-potential Hamiltonian only when accepted is true; "
                "otherwise they must fail clearly or record an explicit prototype assumption."
            ),
        }
        json_path = _pseudopotential_dir(input.case_id) / "ks_dft_local_pseudopotential_contract.json"
        md_path = _pseudopotential_dir(input.case_id) / "ks_dft_local_pseudopotential_contract.md"
        _write_json(json_path, payload)
        md_lines = [
            "# KS-DFT Local Pseudopotential Contract",
            "",
            f"- Accepted: `{accepted}`",
            f"- Status: `{payload['status']}`",
            f"- Source: `{payload['source_ref']}`",
            f"- Required elements: `{', '.join(required_elements)}`",
            "",
            "Generated kernels must not replace a failed contract with an implicit built-in potential.",
        ]
        if issues:
            md_lines.extend(["", "## Issues", ""])
            md_lines.extend(f"- {issue}" for issue in issues)
        md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        _append_event(_case_dir(input.case_id), "validate_local_pseudopotential_artifact", {"accepted": accepted, "required_elements": required_elements})
        return PseudopotentialToolOutput(
            artifact=_artifact(json_path, "ks_dft_local_pseudopotential_contract"),
            artifacts={"json": _artifact(json_path, "ks_dft_local_pseudopotential_contract"), "markdown": _artifact(md_path, "ks_dft_local_pseudopotential_contract_markdown")},
            data=payload,
            warnings=[] if accepted else ["Local pseudopotential contract failed closed; generated kernels must not consume it as validated Hamiltonian data."],
        )
    except Exception as exc:
        return PseudopotentialToolOutput(errors=[str(exc)])


class ValidateNonlocalProjectorArtifactInput(StrictBaseModel):
    case_id: str
    projector_ref: str | None = None
    structure_ref: str | None = None
    required_elements: list[str] = Field(default_factory=list)
    accepted_representations: list[str] = Field(default_factory=lambda: ["norm_conserving_kleinman_bylander", "paw_projector_augmentation"])
    fail_closed: bool = True


def _default_projector_path(case_id: str) -> Path:
    return _pseudopotential_dir(case_id) / "ks_dft_projector_artifact.json"


def _validate_projector_entry(element: str, entry: Any, accepted_representations: list[str]) -> tuple[bool, list[str], dict[str, Any]]:
    issues: list[str] = []
    if not isinstance(entry, dict):
        return False, [f"{element}: entry must be an object"], {}
    representation = entry.get("representation")
    if not isinstance(representation, str) or representation not in accepted_representations:
        issues.append(f"{element}: representation must be one of {accepted_representations}")
    projectors = entry.get("projectors")
    if not isinstance(projectors, list) or not projectors:
        issues.append(f"{element}: projectors must be a non-empty list")
        projectors = []
    projector_summaries = []
    for index, projector in enumerate(projectors):
        label = f"{element}.projectors[{index}]"
        if not isinstance(projector, dict):
            issues.append(f"{label}: projector must be an object")
            continue
        if not isinstance(projector.get("angular_momentum_l"), int) or int(projector.get("angular_momentum_l")) < 0:
            issues.append(f"{label}: angular_momentum_l must be a non-negative integer")
        radial = projector.get("radial_grid")
        values = projector.get("projector_values")
        radial_values = _number_list(radial.get("values")) if isinstance(radial, dict) else None
        projector_values = _number_list(values)
        if radial_values is None:
            issues.append(f"{label}: radial_grid.values must be a non-empty finite number list")
        if isinstance(radial, dict) and not isinstance(radial.get("unit"), str):
            issues.append(f"{label}: radial_grid.unit is required")
        if radial_values is not None and any(b <= a for a, b in zip(radial_values, radial_values[1:])):
            issues.append(f"{label}: radial_grid.values must be strictly increasing")
        if projector_values is None:
            issues.append(f"{label}: projector_values must be a non-empty finite number list")
        if radial_values is not None and projector_values is not None and len(radial_values) != len(projector_values):
            issues.append(f"{label}: radial_grid.values and projector_values lengths must match")
        coefficient = projector.get("coefficient")
        if not isinstance(coefficient, (int, float)) or not math.isfinite(float(coefficient)):
            issues.append(f"{label}: coefficient must be a finite number")
        projector_summaries.append(
            {
                "label": projector.get("label", f"projector_{index}"),
                "angular_momentum_l": projector.get("angular_momentum_l"),
                "grid_points": len(radial_values or []),
                "coefficient": coefficient,
            }
        )
    quadrature = entry.get("quadrature")
    if not isinstance(quadrature, dict) or not isinstance(quadrature.get("method"), str):
        issues.append(f"{element}: quadrature.method is required")
    provenance = entry.get("provenance")
    if not isinstance(provenance, dict) or not provenance:
        issues.append(f"{element}: provenance object is required")
    version_hash = entry.get("version_hash") or entry.get("sha256") or entry.get("source_sha256")
    if not isinstance(version_hash, str) or not version_hash:
        issues.append(f"{element}: version_hash/sha256/source_sha256 is required")
    paw = entry.get("paw_augmentation")
    paw_summary = {}
    if representation == "paw_projector_augmentation":
        if not isinstance(paw, dict):
            issues.append(f"{element}: paw_augmentation object is required for PAW representation")
        else:
            for key in ["augmentation_charge_moments", "partial_waves", "compensation_charge_policy"]:
                if key not in paw:
                    issues.append(f"{element}: paw_augmentation.{key} is required")
            paw_summary = {"keys": sorted(paw.keys()) if isinstance(paw, dict) else []}
    summary = {
        "representation": representation,
        "projector_count": len(projectors),
        "projectors": projector_summaries,
        "quadrature": quadrature if isinstance(quadrature, dict) else {},
        "version_hash": version_hash,
        "provenance": provenance if isinstance(provenance, dict) else {},
        "paw_augmentation": paw_summary,
    }
    return not issues, issues, summary


def validate_nonlocal_projector_artifact(input: ValidateNonlocalProjectorArtifactInput) -> PseudopotentialToolOutput:
    """Validate a case-local nonlocal projector/PAW artifact without parsing POTCAR in generated kernels."""
    try:
        required_elements = _required_species(
            ValidateLocalPseudopotentialArtifactInput(
                case_id=input.case_id,
                structure_ref=input.structure_ref,
                required_elements=input.required_elements,
            )
        )
        source_ref = input.projector_ref
        source_path = resolve_workspace_path(source_ref, workspace=_workspace(), must_be_within_workspace=False) if source_ref else _default_projector_path(input.case_id)
        issues: list[str] = []
        element_summaries: dict[str, Any] = {}
        artifact_present = source_path.exists()
        if not artifact_present:
            issues.append(f"Missing nonlocal projector artifact: {to_agent_path(source_path, workspace=_workspace())}")
            source = {}
        else:
            source = _read_json(source_path)
            elements = source.get("elements")
            if not isinstance(elements, dict):
                issues.append("projector artifact must contain an elements object")
                elements = {}
            for element in required_elements:
                if element not in elements:
                    issues.append(f"{element}: missing projector entry")
                    continue
                ok, entry_issues, summary = _validate_projector_entry(element, elements[element], input.accepted_representations)
                if not ok:
                    issues.extend(entry_issues)
                element_summaries[element] = summary
        accepted = not issues
        payload = {
            "schema_version": "physicsos.ks_dft.nonlocal_projector_contract.v1",
            "case_id": input.case_id,
            "accepted": accepted,
            "fail_closed": input.fail_closed,
            "status": "accepted_nonlocal_projector_or_paw" if accepted else "blocked_missing_or_invalid_nonlocal_projector_or_paw",
            "source_ref": to_agent_path(source_path, workspace=_workspace()) if source_path.is_relative_to(_workspace()) else _path_text(source_path),
            "required_elements": required_elements,
            "artifact_present": artifact_present,
            "accepted_representations": input.accepted_representations,
            "required_contract": {
                "norm_conserving_kleinman_bylander": "projector radial functions, l channel, coefficient, quadrature, provenance",
                "paw_projector_augmentation": "projectors plus augmentation charge moments, partial waves, compensation-charge policy, quadrature, provenance",
                "hamiltonian_action_hook": "generated kernel must record how V_nonlocal psi is applied and validated",
            },
            "element_summaries": element_summaries,
            "issues": issues,
            "kernel_instruction": (
                "Generated kernels may include nonlocal/projector Hamiltonian terms only when accepted is true; "
                "otherwise they must fail clearly or record a prototype assumption with projector terms disabled."
            ),
        }
        json_path = _pseudopotential_dir(input.case_id) / "ks_dft_projector_context.json"
        md_path = _pseudopotential_dir(input.case_id) / "ks_dft_projector_context.md"
        _write_json(json_path, payload)
        md_lines = [
            "# KS-DFT Nonlocal Projector / PAW Contract",
            "",
            f"- Accepted: `{accepted}`",
            f"- Status: `{payload['status']}`",
            f"- Source: `{payload['source_ref']}`",
            f"- Required elements: `{', '.join(required_elements)}`",
            "",
            "Generated kernels must not invent projector or PAW augmentation data.",
        ]
        if issues:
            md_lines.extend(["", "## Issues", ""])
            md_lines.extend(f"- {issue}" for issue in issues)
        md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        _append_event(_case_dir(input.case_id), "validate_nonlocal_projector_artifact", {"accepted": accepted, "required_elements": required_elements})
        return PseudopotentialToolOutput(
            artifact=_artifact(json_path, "ks_dft_projector_context"),
            artifacts={"json": _artifact(json_path, "ks_dft_projector_context"), "markdown": _artifact(md_path, "ks_dft_projector_context_markdown")},
            data=payload,
            warnings=[] if accepted else ["Nonlocal projector/PAW contract failed closed; generated kernels must not invent projector data."],
        )
    except Exception as exc:
        return PseudopotentialToolOutput(errors=[str(exc)])


PSEUDOPOTENTIAL_TOOL_SPECS = [
    (index_vasp_paw_pbe_library, IndexVaspPawPbeLibraryInput, PseudopotentialToolOutput),
    (select_pseudopotentials_for_structure, SelectPseudopotentialsForStructureInput, PseudopotentialToolOutput),
    (validate_local_pseudopotential_artifact, ValidateLocalPseudopotentialArtifactInput, PseudopotentialToolOutput),
    (validate_nonlocal_projector_artifact, ValidateNonlocalProjectorArtifactInput, PseudopotentialToolOutput),
]

for _tool, _input, _output in PSEUDOPOTENTIAL_TOOL_SPECS:
    _tool.input_model = _input
    _tool.output_model = _output
    _tool.side_effects = "workspace artifacts only"
    _tool.requires_approval = False
