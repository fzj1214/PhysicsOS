from __future__ import annotations

from typing import Literal

from pydantic import Field

from physicsos.schemas.common import StrictBaseModel
from physicsos.schemas.mesh import MeshQualityReport


class VerificationPolicy(StrictBaseModel):
    residual_tolerance: float | None = None
    conservation_tolerance: float | None = None
    max_ood_score: float | None = None
    require_full_solver_above_ood: bool = True
    required_checks: list[str] = Field(default_factory=lambda: ["residual", "conservation", "uncertainty", "ood"])


class VerificationReport(StrictBaseModel):
    problem_id: str
    result_id: str
    status: Literal["accepted", "accepted_with_warnings", "rejected", "needs_full_solver", "needs_user_input"]
    residuals: dict[str, float] = Field(default_factory=dict)
    conservation_errors: dict[str, float] = Field(default_factory=dict)
    uncertainty: dict[str, float] = Field(default_factory=dict)
    ood_score: float = 0.0
    mesh_quality: MeshQualityReport | None = None
    nearest_reference_cases: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    recommended_next_action: Literal[
        "accept",
        "refine_mesh",
        "rerun_surrogate",
        "run_full_solver",
        "run_higher_fidelity_solver",
        "ask_user_for_missing_input",
    ]
    explanation: str
