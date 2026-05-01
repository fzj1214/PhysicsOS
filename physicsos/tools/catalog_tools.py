from __future__ import annotations

from pydantic import Field

from physicsos.backends.physics_registry import (
    OPERATOR_TEMPLATES,
    POSTPROCESS_TEMPLATES,
    SOLVER_BACKEND_REGISTRY,
    VERIFICATION_RULES,
    OperatorTemplate,
    PostprocessTemplate,
    SolverBackendEntry,
    VerificationRule,
)
from physicsos.schemas.common import StrictBaseModel
from physicsos.schemas.operators import PhysicsDomain
from physicsos.schemas.problem import PhysicsProblem


class ListOperatorTemplatesInput(StrictBaseModel):
    domain: PhysicsDomain | None = None
    equation_class: str | None = None
    taps_candidate: bool | None = None


class ListOperatorTemplatesOutput(StrictBaseModel):
    templates: list[OperatorTemplate] = Field(default_factory=list)


def list_operator_templates(input: ListOperatorTemplatesInput) -> ListOperatorTemplatesOutput:
    """List registered governing-equation templates for problem construction."""
    templates = list(OPERATOR_TEMPLATES)
    if input.domain is not None:
        templates = [item for item in templates if item.domain == input.domain or item.domain == "custom"]
    if input.equation_class is not None:
        requested = input.equation_class.lower()
        templates = [item for item in templates if item.equation_class.lower() == requested]
    if input.taps_candidate is not None:
        templates = [item for item in templates if item.taps_candidate == input.taps_candidate]
    return ListOperatorTemplatesOutput(templates=templates)


class ListSolverBackendsInput(StrictBaseModel):
    domain: PhysicsDomain | None = None
    family: str | None = None
    requires_remote_service: bool | None = None


class ListSolverBackendsOutput(StrictBaseModel):
    backends: list[SolverBackendEntry] = Field(default_factory=list)


def list_solver_backends(input: ListSolverBackendsInput) -> ListSolverBackendsOutput:
    """List explicit solver backend registry entries for routing and planning."""
    backends = list(SOLVER_BACKEND_REGISTRY)
    if input.domain is not None:
        backends = [item for item in backends if input.domain in item.domains or "custom" in item.domains]
    if input.family is not None:
        backends = [item for item in backends if item.family == input.family]
    if input.requires_remote_service is not None:
        backends = [item for item in backends if item.requires_remote_service == input.requires_remote_service]
    return ListSolverBackendsOutput(backends=backends)


class ListVerificationRulesInput(StrictBaseModel):
    domain: PhysicsDomain | None = None
    equation_class: str | None = None


class ListVerificationRulesOutput(StrictBaseModel):
    rules: list[VerificationRule] = Field(default_factory=list)


def list_verification_rules(input: ListVerificationRulesInput) -> ListVerificationRulesOutput:
    """List verification rules applicable to domains or operator classes."""
    rules = list(VERIFICATION_RULES)
    if input.domain is not None:
        rules = [item for item in rules if input.domain in item.domains or "custom" in item.domains]
    if input.equation_class is not None:
        requested = input.equation_class.lower()
        rules = [
            item for item in rules
            if not item.operator_classes or requested in {operator.lower() for operator in item.operator_classes}
        ]
    return ListVerificationRulesOutput(rules=rules)


class ListPostprocessTemplatesInput(StrictBaseModel):
    domain: PhysicsDomain | None = None


class ListPostprocessTemplatesOutput(StrictBaseModel):
    templates: list[PostprocessTemplate] = Field(default_factory=list)


def list_postprocess_templates(input: ListPostprocessTemplatesInput) -> ListPostprocessTemplatesOutput:
    """List report/KPI/visualization templates for postprocess-agent."""
    templates = list(POSTPROCESS_TEMPLATES)
    if input.domain is not None:
        templates = [item for item in templates if input.domain in item.domains or "custom" in item.domains]
    return ListPostprocessTemplatesOutput(templates=templates)


class RecommendRuntimeStackInput(StrictBaseModel):
    problem: PhysicsProblem


class RecommendRuntimeStackOutput(StrictBaseModel):
    operator_templates: list[OperatorTemplate] = Field(default_factory=list)
    solver_backends: list[SolverBackendEntry] = Field(default_factory=list)
    verification_rules: list[VerificationRule] = Field(default_factory=list)
    postprocess_templates: list[PostprocessTemplate] = Field(default_factory=list)
    recommended_order: list[str] = Field(default_factory=list)


def recommend_runtime_stack(input: RecommendRuntimeStackInput) -> RecommendRuntimeStackOutput:
    """Recommend explicit operator, solver, verification, and postprocess registry entries for a problem."""
    equation_classes = {operator.equation_class.lower() for operator in input.problem.operators}
    operator_templates = [
        template for template in OPERATOR_TEMPLATES
        if template.domain in {input.problem.domain, "custom"} or template.equation_class.lower() in equation_classes
    ]
    solver_backends = list_solver_backends(ListSolverBackendsInput(domain=input.problem.domain)).backends
    verification_rules = [
        rule for rule in VERIFICATION_RULES
        if input.problem.domain in rule.domains or "custom" in rule.domains or equation_classes & {item.lower() for item in rule.operator_classes}
    ]
    postprocess_templates = list_postprocess_templates(ListPostprocessTemplatesInput(domain=input.problem.domain)).templates
    recommended_order = ["taps"]
    if any(backend.id in {"grid_neural_operator", "mesh_graph_operator"} for backend in solver_backends):
        recommended_order.append("neural_surrogate")
    if any(backend.requires_remote_service for backend in solver_backends):
        recommended_order.append("remote_full_solver")
    return RecommendRuntimeStackOutput(
        operator_templates=operator_templates,
        solver_backends=solver_backends,
        verification_rules=verification_rules,
        postprocess_templates=postprocess_templates,
        recommended_order=recommended_order,
    )


for _tool, _input, _output in [
    (list_operator_templates, ListOperatorTemplatesInput, ListOperatorTemplatesOutput),
    (list_solver_backends, ListSolverBackendsInput, ListSolverBackendsOutput),
    (list_verification_rules, ListVerificationRulesInput, ListVerificationRulesOutput),
    (list_postprocess_templates, ListPostprocessTemplatesInput, ListPostprocessTemplatesOutput),
    (recommend_runtime_stack, RecommendRuntimeStackInput, RecommendRuntimeStackOutput),
]:
    _tool.input_model = _input
    _tool.output_model = _output
    _tool.side_effects = "none"
