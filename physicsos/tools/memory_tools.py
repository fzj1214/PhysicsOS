from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import Field

from physicsos.config import project_root
from physicsos.schemas.common import StrictBaseModel
from physicsos.schemas.postprocess import PostprocessResult
from physicsos.schemas.problem import PhysicsProblem
from physicsos.schemas.solver import SolverResult
from physicsos.schemas.verification import VerificationReport


class CaseMemoryHit(StrictBaseModel):
    case_id: str
    score: float
    reason: str
    backend: str | None = None
    verification_status: str | None = None
    indexed_features: list[str] = Field(default_factory=list)
    metadata: dict[str, str | float | int | bool] = Field(default_factory=dict)


class CaseMemoryRecord(StrictBaseModel):
    id: str
    problem: PhysicsProblem
    result: SolverResult
    verification: VerificationReport
    postprocess: PostprocessResult | None = None
    indexed_features: list[str] = Field(default_factory=list)
    tokens: list[str] = Field(default_factory=list)
    dataset_tags: list[str] = Field(default_factory=list)
    usage_rights: str = "project_internal"


class SearchCaseMemoryInput(StrictBaseModel):
    problem: PhysicsProblem
    top_k: int = 5
    filters: dict[str, str | float | int] = Field(default_factory=dict)
    memory_uri: str | None = None


class SearchCaseMemoryOutput(StrictBaseModel):
    cases: list[CaseMemoryHit] = Field(default_factory=list)
    searched_records: int = 0
    memory_uri: str


def search_case_memory(input: SearchCaseMemoryInput) -> SearchCaseMemoryOutput:
    """Retrieve similar historical cases by physics, geometry, BCs, materials, and solver metadata."""
    path = _memory_path(input.memory_uri)
    query_tokens = _case_tokens(input.problem)
    hits: list[CaseMemoryHit] = []
    searched = 0

    for record in _read_records(path):
        searched += 1
        if not _passes_filters(record, input.filters):
            continue
        record_tokens = set(record.tokens or _case_tokens(record.problem))
        score = _jaccard(query_tokens, record_tokens)
        if score <= 0.0:
            continue
        matched = sorted(query_tokens & record_tokens)
        hits.append(
            CaseMemoryHit(
                case_id=record.id,
                score=round(score, 6),
                reason=f"Matched {len(matched)} indexed features: {', '.join(matched[:8]) or 'none'}.",
                backend=record.result.backend,
                verification_status=record.verification.status,
                indexed_features=record.indexed_features,
                metadata={
                    "domain": record.problem.domain,
                    "operator_count": len(record.problem.operators),
                    "field_count": len(record.problem.fields),
                },
            )
        )

    hits.sort(key=lambda item: item.score, reverse=True)
    return SearchCaseMemoryOutput(cases=hits[: max(input.top_k, 0)], searched_records=searched, memory_uri=str(path))


class StoreCaseResultInput(StrictBaseModel):
    problem: PhysicsProblem
    result: SolverResult
    verification: VerificationReport
    postprocess: PostprocessResult | None = None
    dataset_tags: list[str] = Field(default_factory=list)
    usage_rights: str = "project_internal"
    memory_uri: str | None = None


class StoreCaseResultOutput(StrictBaseModel):
    case_id: str
    indexed_features: list[str] = Field(default_factory=list)
    memory_uri: str
    stored: bool = True


def store_case_result(input: StoreCaseResultInput) -> StoreCaseResultOutput:
    """Store validated simulation results for retrieval, warm start, and future training."""
    path = _memory_path(input.memory_uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens = sorted(_case_tokens(input.problem))
    indexed_features = _indexed_feature_names(input.problem, input.result)
    record = CaseMemoryRecord(
        id=input.problem.id,
        problem=input.problem,
        result=input.result,
        verification=input.verification,
        postprocess=input.postprocess,
        indexed_features=indexed_features,
        tokens=tokens,
        dataset_tags=input.dataset_tags,
        usage_rights=input.usage_rights,
    )

    existing = [item for item in _read_raw_records(path) if item.get("id") != record.id]
    existing.append(record.model_dump(mode="json"))
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text("\n".join(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in existing) + "\n", encoding="utf-8")
    tmp_path.replace(path)
    return StoreCaseResultOutput(case_id=input.problem.id, indexed_features=indexed_features, memory_uri=str(path))


def _memory_path(uri: str | None = None) -> Path:
    if uri:
        return Path(uri).expanduser()
    return project_root() / "data" / "case_memory.jsonl"


def _read_raw_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _read_records(path: Path) -> list[CaseMemoryRecord]:
    records: list[CaseMemoryRecord] = []
    for payload in _read_raw_records(path):
        try:
            records.append(CaseMemoryRecord.model_validate(payload))
        except ValueError:
            continue
    return records


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9_:+.-]+", "_", value.strip().lower()).strip("_")


def _add_token(tokens: set[str], prefix: str, value: object | None) -> None:
    if value is None:
        return
    normalized = _normalize_token(str(value))
    if normalized:
        tokens.add(f"{prefix}:{normalized}")


def _case_tokens(problem: PhysicsProblem) -> set[str]:
    tokens: set[str] = set()
    _add_token(tokens, "domain", problem.domain)
    _add_token(tokens, "geometry_dimension", problem.geometry.dimension)
    _add_token(tokens, "geometry_source", problem.geometry.source.kind)
    if problem.mesh is not None:
        _add_token(tokens, "mesh_kind", problem.mesh.kind)
        _add_token(tokens, "mesh_dimension", problem.mesh.dimension)
        for cell_type in problem.mesh.topology.cell_types:
            _add_token(tokens, "mesh_cell", cell_type)
    for encoding in problem.geometry.encodings:
        _add_token(tokens, "geometry_encoding", encoding.kind)
    for field in problem.fields:
        _add_token(tokens, "field", field.name)
        _add_token(tokens, "field_kind", field.kind)
    for operator in problem.operators:
        _add_token(tokens, "operator", operator.equation_class)
        _add_token(tokens, "operator_name", operator.name)
        _add_token(tokens, "operator_form", operator.form)
        for conserved in operator.conserved_quantities:
            _add_token(tokens, "conserved", conserved)
        for number in operator.nondimensional_numbers:
            _add_token(tokens, "nondim", number.name)
    for material in problem.materials:
        _add_token(tokens, "material", material.name)
        _add_token(tokens, "material_phase", material.phase)
    for bc in problem.boundary_conditions:
        _add_token(tokens, "bc_kind", bc.kind)
        _add_token(tokens, "bc_field", bc.field)
        _add_token(tokens, "bc_region", bc.region_id)
    for target in problem.targets:
        _add_token(tokens, "target", target.name)
        _add_token(tokens, "target_objective", target.objective)
    return tokens


def _indexed_feature_names(problem: PhysicsProblem, result: SolverResult) -> list[str]:
    features = [
        "domain",
        "operators",
        "geometry",
        "fields",
        "boundary_conditions",
        "targets",
        "solver_backend",
        "verification",
    ]
    if problem.mesh is not None:
        features.append("mesh")
    if problem.geometry.encodings:
        features.append("geometry_encodings")
    if result.residuals:
        features.append("residuals")
    if result.uncertainty:
        features.append("uncertainty")
    return features


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _passes_filters(record: CaseMemoryRecord, filters: dict[str, str | float | int]) -> bool:
    for key, value in filters.items():
        if key == "domain" and record.problem.domain != value:
            return False
        if key == "backend" and record.result.backend != value:
            return False
        if key == "verification_status" and record.verification.status != value:
            return False
        if key == "geometry_dimension" and record.problem.geometry.dimension != value:
            return False
    return True


for _tool, _input, _output in [
    (search_case_memory, SearchCaseMemoryInput, SearchCaseMemoryOutput),
    (store_case_result, StoreCaseResultInput, StoreCaseResultOutput),
]:
    _tool.input_model = _input
    _tool.output_model = _output
    _tool.side_effects = "case memory"
