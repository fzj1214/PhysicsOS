from __future__ import annotations

from typing import Literal

from pydantic import Field

from physicsos.backends.surrogate_adapters import list_adapters
from physicsos.backends.surrogate_registry import checkpoint_exists
from physicsos.backends.surrogate_runtime import list_surrogate_models, route_surrogate, run_surrogate_scaffold, score_surrogate_model
from physicsos.schemas.common import StrictBaseModel
from physicsos.schemas.problem import PhysicsProblem
from physicsos.schemas.solver import SolverResult
from physicsos.schemas.surrogate import SurrogateDecision, SurrogateModelSpec, SurrogateSupportScore


class ListSurrogateModelsInput(StrictBaseModel):
    family: str | None = None
    domain: str | None = None


class ListSurrogateModelsOutput(StrictBaseModel):
    models: list[SurrogateModelSpec]
    checkpoint_status: dict[str, bool] = Field(default_factory=dict)
    adapters: list[str] = Field(default_factory=list)


def list_available_surrogates(input: ListSurrogateModelsInput) -> ListSurrogateModelsOutput:
    """List registered neural operator and surrogate model specs."""
    models = list(list_surrogate_models())
    if input.family is not None:
        models = [model for model in models if model.family == input.family]
    if input.domain is not None:
        models = [model for model in models if input.domain in model.domains]
    return ListSurrogateModelsOutput(
        models=models,
        checkpoint_status={model.id: checkpoint_exists(model) for model in models},
        adapters=list_adapters(),
    )


class EstimateSurrogateSupportInput(StrictBaseModel):
    problem: PhysicsProblem
    candidate_model_ids: list[str] = Field(default_factory=list)


class EstimateSurrogateSupportOutput(StrictBaseModel):
    scores: list[SurrogateSupportScore]


def estimate_surrogate_support(input: EstimateSurrogateSupportInput) -> EstimateSurrogateSupportOutput:
    """Score registered surrogate models for a PhysicsProblem."""
    requested = set(input.candidate_model_ids)
    models = [model for model in list_surrogate_models() if not requested or model.id in requested]
    return EstimateSurrogateSupportOutput(scores=[score_surrogate_model(input.problem, model) for model in models])


class RouteSurrogateModelInput(StrictBaseModel):
    problem: PhysicsProblem
    mode: Literal["fast_solver", "warm_start", "corrector"] = "fast_solver"


class RouteSurrogateModelOutput(StrictBaseModel):
    decision: SurrogateDecision


def route_surrogate_model(input: RouteSurrogateModelInput) -> RouteSurrogateModelOutput:
    """Select the best registered surrogate/neural-operator model for a PhysicsProblem."""
    return RouteSurrogateModelOutput(decision=route_surrogate(input.problem, mode=input.mode))


class RunSurrogateInferenceInput(StrictBaseModel):
    problem: PhysicsProblem
    decision: SurrogateDecision | None = None
    mode: Literal["fast_solver", "warm_start", "corrector"] = "fast_solver"


class RunSurrogateInferenceOutput(StrictBaseModel):
    result: SolverResult
    decision: SurrogateDecision


def run_surrogate_inference(input: RunSurrogateInferenceInput) -> RunSurrogateInferenceOutput:
    """Run registered surrogate runtime. Scaffold performs routing and returns a typed SolverResult."""
    decision = input.decision or route_surrogate(input.problem, mode=input.mode)
    result = run_surrogate_scaffold(input.problem, decision)
    return RunSurrogateInferenceOutput(result=result, decision=decision)


for _tool, _input, _output in [
    (list_available_surrogates, ListSurrogateModelsInput, ListSurrogateModelsOutput),
    (estimate_surrogate_support, EstimateSurrogateSupportInput, EstimateSurrogateSupportOutput),
    (route_surrogate_model, RouteSurrogateModelInput, RouteSurrogateModelOutput),
    (run_surrogate_inference, RunSurrogateInferenceInput, RunSurrogateInferenceOutput),
]:
    _tool.input_model = _input
    _tool.output_model = _output
    _tool.side_effects = "workspace artifacts only"
    _tool.requires_approval = False
