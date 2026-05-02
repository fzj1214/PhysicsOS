from __future__ import annotations

from pydantic import Field

from physicsos.schemas.common import StrictBaseModel
from physicsos.tools.problem_tools import (
    BuildPhysicsProblemInput,
    BuildPhysicsProblemOutput,
    ValidatePhysicsProblemOutput,
    build_physics_problem,
    validate_physics_problem,
    ValidatePhysicsProblemInput,
)
from physicsos.workflows.universal import PhysicsOSWorkflowResult, run_physicsos_workflow


class RunTypedPhysicsOSWorkflowInput(StrictBaseModel):
    user_request: str
    use_knowledge: bool = True
    arxiv_max_results: int = 0
    use_deepsearch: bool = False
    taps_rank: int = 8
    max_validation_attempts: int = 2


class RunTypedPhysicsOSWorkflowOutput(StrictBaseModel):
    build: BuildPhysicsProblemOutput
    initial_validation: ValidatePhysicsProblemOutput | None = None
    workflow: PhysicsOSWorkflowResult | None = None
    missing_inputs: list[str] = Field(default_factory=list)


def run_typed_physicsos_workflow(input: RunTypedPhysicsOSWorkflowInput) -> RunTypedPhysicsOSWorkflowOutput:
    """Natural-language entry point for the typed PhysicsOS workflow."""
    built = build_physics_problem(BuildPhysicsProblemInput(user_request=input.user_request))
    if built.problem is None:
        return RunTypedPhysicsOSWorkflowOutput(build=built, missing_inputs=built.missing_inputs)

    initial_validation = validate_physics_problem(ValidatePhysicsProblemInput(problem=built.problem))
    workflow = run_physicsos_workflow(
        problem=built.problem,
        use_knowledge=input.use_knowledge,
        arxiv_max_results=input.arxiv_max_results,
        use_deepsearch=input.use_deepsearch,
        taps_rank=input.taps_rank,
        max_validation_attempts=input.max_validation_attempts,
    )
    return RunTypedPhysicsOSWorkflowOutput(
        build=built,
        initial_validation=initial_validation,
        workflow=workflow,
        missing_inputs=built.missing_inputs,
    )


run_typed_physicsos_workflow.input_model = RunTypedPhysicsOSWorkflowInput
run_typed_physicsos_workflow.output_model = RunTypedPhysicsOSWorkflowOutput
run_typed_physicsos_workflow.side_effects = "may create workflow artifacts"
run_typed_physicsos_workflow.requires_approval = False
