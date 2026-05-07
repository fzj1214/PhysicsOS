from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from physicsos.schemas.common import StrictBaseModel
from physicsos.schemas.problem import PhysicsProblem
from physicsos.schemas.taps import TAPSProblem


class PhysicsProblemContract(StrictBaseModel):
    """Locked typed intent that downstream agents must preserve."""

    problem_id: str
    raw_request: str
    domain: str
    geometry_dimension: int
    geometry_id: str
    fields: list[dict[str, Any]] = Field(default_factory=list)
    operators: list[dict[str, Any]] = Field(default_factory=list)
    materials: list[dict[str, Any]] = Field(default_factory=list)
    boundary_conditions: list[dict[str, Any]] = Field(default_factory=list)
    initial_conditions: list[dict[str, Any]] = Field(default_factory=list)
    targets: list[dict[str, Any]] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    fingerprint: str


class ContractReviewReport(StrictBaseModel):
    problem_id: str
    status: Literal["accepted", "needs_retry", "failed"]
    fingerprint: str
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    reviewed_items: dict[str, bool] = Field(default_factory=dict)
    retry_instruction: str | None = None


def _stable_payload(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _stable_payload(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_stable_payload(item) for item in value]
    return value


def _fingerprint_payload(payload: dict[str, Any]) -> str:
    import hashlib
    import json

    encoded = json.dumps(_stable_payload(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_physics_problem_contract(problem: PhysicsProblem, assumptions: list[str] | None = None) -> PhysicsProblemContract:
    payload = {
        "problem_id": problem.id,
        "raw_request": problem.user_intent.raw_request,
        "domain": problem.domain,
        "geometry_dimension": problem.geometry.dimension,
        "geometry_id": problem.geometry.id,
        "fields": [
            {"name": field.name, "kind": field.kind, "units": field.units, "location": field.location}
            for field in problem.fields
        ],
        "operators": [
            {
                "id": operator.id,
                "equation_class": operator.equation_class,
                "form": operator.form,
                "fields_in": operator.fields_in,
                "fields_out": operator.fields_out,
                "source_terms": [term.model_dump(mode="json") for term in operator.source_terms],
                "nondimensional_numbers": [number.model_dump(mode="json") for number in operator.nondimensional_numbers],
            }
            for operator in problem.operators
        ],
        "materials": [
            {
                "id": material.id,
                "name": material.name,
                "phase": material.phase,
                "region_ids": material.region_ids,
                "properties": [prop.model_dump(mode="json") for prop in material.properties],
            }
            for material in problem.materials
        ],
        "boundary_conditions": [bc.model_dump(mode="json") for bc in problem.boundary_conditions],
        "initial_conditions": [ic.model_dump(mode="json") for ic in problem.initial_conditions],
        "targets": [target.model_dump(mode="json") for target in problem.targets],
        "assumptions": assumptions or [],
    }
    return PhysicsProblemContract(**payload, fingerprint=_fingerprint_payload(payload))


def _canonical_operator_family(value: str | None) -> str:
    lowered = (value or "").strip().lower()
    normalized = lowered.replace("_", " ").replace("-", " ").replace("/", " ")
    if any(token in normalized for token in ("navier stokes", "incompressible ns")):
        return "navier_stokes"
    if "oseen" in normalized:
        return "oseen"
    if "stokes" in normalized:
        return "stokes"
    if any(token in normalized for token in ("heat", "thermal", "diffusion", "conduction", "elliptic pde", "elliptic", "poisson")):
        return "scalar_elliptic"
    if any(token in normalized for token in ("reaction diffusion", "reaction_diffusion")):
        return "reaction_diffusion"
    if any(token in normalized for token in ("elasticity", "linear elastic", "solid mechanics")):
        return "linear_elasticity"
    if any(token in normalized for token in ("maxwell", "curl curl", "electromagnetic")):
        return "curl_curl"
    return "_".join(normalized.split())


def review_problem_to_taps_contract(
    contract: PhysicsProblemContract,
    taps_problem: TAPSProblem,
) -> ContractReviewReport:
    errors = ["Mechanical TAPS contract review is disabled for the paper-replication route; review derivation.md and verification/report.json."]
    return ContractReviewReport(
        problem_id=contract.problem_id,
        status="failed",
        fingerprint=contract.fingerprint,
        errors=errors,
        reviewed_items={},
        retry_instruction="Use taps-derivation-agent to revise derivation.md and rerun the Fig.7 verification chain.",
    )
