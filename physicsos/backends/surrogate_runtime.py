from __future__ import annotations

from physicsos.backends.surrogate_inference import run_checkpoint_surrogate
from physicsos.backends.surrogate_registry import checkpoint_exists, configured_surrogate_models
from physicsos.schemas.common import Provenance
from physicsos.schemas.problem import PhysicsProblem
from physicsos.schemas.solver import SolverResult
from physicsos.schemas.surrogate import SurrogateDecision, SurrogateModelSpec, SurrogateSupportScore


def list_surrogate_models() -> tuple[SurrogateModelSpec, ...]:
    return configured_surrogate_models()


def score_surrogate_model(problem: PhysicsProblem, model: SurrogateModelSpec) -> SurrogateSupportScore:
    reasons: list[str] = []
    risks: list[str] = []
    score = 0.0

    if problem.domain in model.domains or "custom" in model.domains:
        score += 0.35
        reasons.append(f"domain {problem.domain} is supported")
    else:
        risks.append(f"domain {problem.domain} not in {model.domains}")

    operator_classes = {operator.equation_class for operator in problem.operators}
    if not model.operator_families or operator_classes.intersection(model.operator_families):
        score += 0.25
        reasons.append("operator family appears compatible")
    else:
        risks.append(f"operator families {sorted(operator_classes)} may be unsupported")

    geometry_encodings = {encoding.kind for encoding in problem.geometry.encodings}
    if geometry_encodings.intersection(model.geometry_encodings):
        score += 0.25
        reasons.append("required geometry encoding is available")
    else:
        risks.append("missing preferred geometry encoding")

    if problem.mesh is None or problem.mesh.kind in model.mesh_kinds:
        score += 0.15
        reasons.append("mesh kind appears compatible")
    else:
        risks.append(f"mesh kind {problem.mesh.kind} may be unsupported")

    supported = score >= 0.5
    return SurrogateSupportScore(
        model_id=model.id,
        score=round(score, 4),
        supported=supported,
        reasons=reasons,
        risks=risks,
    )


def route_surrogate(problem: PhysicsProblem, mode: str = "fast_solver") -> SurrogateDecision:
    models = list_surrogate_models()
    scores = [score_surrogate_model(problem, model) for model in models]
    supported = [score for score in scores if score.supported]
    if not supported:
        return SurrogateDecision(
            selected_model_id="none",
            candidate_models=scores,
            mode="reject",
            reason="No surrogate model has enough support for this problem.",
        )

    by_id = {model.id: model for model in models}
    selected = max(
        supported,
        key=lambda score: (
            score.score,
            checkpoint_exists(by_id[score.model_id]),
            by_id[score.model_id].runner != "scaffold",
        ),
    )
    model = next(item for item in models if item.id == selected.model_id)
    selected_mode = mode if mode in {"fast_solver", "warm_start", "corrector"} else "fast_solver"
    return SurrogateDecision(
        selected_model_id=model.id,
        candidate_models=scores,
        mode=selected_mode,  # type: ignore[arg-type]
        reason="Selected highest-scoring surrogate model.",
        required_geometry_encodings=model.geometry_encodings,
        expected_uncertainty=1.0 - selected.score,
    )


def run_surrogate_scaffold(problem: PhysicsProblem, decision: SurrogateDecision) -> SolverResult:
    model = next((item for item in list_surrogate_models() if item.id == decision.selected_model_id), None)
    if model is not None and model.runner != "scaffold":
        return run_checkpoint_surrogate(problem, model, decision)
    if model is not None and checkpoint_exists(model):
        return run_checkpoint_surrogate(problem, model, decision)

    status = "needs_review" if decision.mode != "reject" else "failed"
    return SolverResult(
        id=f"result:{decision.selected_model_id}:scaffold",
        problem_id=problem.id,
        backend=decision.selected_model_id,
        status=status,
        scalar_outputs={
            "surrogate_mode": decision.mode,
            "message": "Surrogate runtime scaffold executed; no tensor model inference performed yet.",
        },
        uncertainty={"surrogate_expected_uncertainty": decision.expected_uncertainty or 1.0},
        provenance=Provenance(created_by="surrogate_runtime", source="scaffold"),
    )
