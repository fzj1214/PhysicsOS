from __future__ import annotations

from pydantic import Field
from uuid import uuid4

from physicsos.agents.structured import CoreAgentLLMConfig, StructuredLLMClient, call_structured_agent, load_core_agent_config, structured_agent_event_context
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
    taps_max_wall_time_seconds: float = 120.0
    core_agents_mode: str | None = None


class RunTypedPhysicsOSWorkflowOutput(StrictBaseModel):
    build: BuildPhysicsProblemOutput
    initial_validation: ValidatePhysicsProblemOutput | None = None
    workflow: PhysicsOSWorkflowResult | None = None
    missing_inputs: list[str] = Field(default_factory=list)


BUILD_PHYSICS_PROBLEM_SYSTEM_PROMPT = """Extract a PhysicsOS BuildPhysicsProblemOutput from the user request.
Return only a JSON object matching the output schema. Preserve explicit geometry, materials, boundary conditions,
initial conditions, source terms, targets, and uncertainty. Do not replace user-provided values with defaults."""


def build_physics_problem_structured(
    input: BuildPhysicsProblemInput,
    *,
    client: StructuredLLMClient,
    config: CoreAgentLLMConfig | None = None,
) -> BuildPhysicsProblemOutput:
    result = call_structured_agent(
        agent_name="build-physics-problem-agent",
        input_model=input,
        output_model=BuildPhysicsProblemOutput,
        system_prompt=BUILD_PHYSICS_PROBLEM_SYSTEM_PROMPT,
        client=client,
        config=config,
    )
    if result.output is None:
        return BuildPhysicsProblemOutput(
            problem=None,
            missing_inputs=["structured_llm_output"],
            assumptions=[result.error or "Structured build-physics-problem agent failed validation."],
        )
    return result.output


def _build_problem_with_llm_first_fallback(
    build_input: BuildPhysicsProblemInput,
    *,
    core_config: CoreAgentLLMConfig,
    structured_client: StructuredLLMClient | None,
) -> BuildPhysicsProblemOutput:
    if structured_client is not None:
        built = build_physics_problem_structured(build_input, client=structured_client, config=core_config)
        if built.problem is not None:
            return built
        fallback = build_physics_problem(build_input)
        return fallback.model_copy(
            update={
                "assumptions": [
                    *fallback.assumptions,
                    "Structured LLM problem extraction failed after validation retries; deterministic fallback was used.",
                    *built.assumptions,
                ]
            }
        )
    fallback = build_physics_problem(build_input)
    return fallback.model_copy(
        update={
            "assumptions": [
                *fallback.assumptions,
                "No structured LLM client was available; deterministic fallback was used.",
            ]
        }
    )


def run_typed_physicsos_workflow(
    input: RunTypedPhysicsOSWorkflowInput,
    *,
    structured_client: StructuredLLMClient | None = None,
) -> RunTypedPhysicsOSWorkflowOutput:
    """Natural-language entry point for the typed PhysicsOS workflow."""
    run_id = f"workflow:{uuid4().hex}"
    core_config = load_core_agent_config()
    if input.core_agents_mode is not None:
        core_config = core_config.model_copy(update={"mode": input.core_agents_mode})
    build_input = BuildPhysicsProblemInput(user_request=input.user_request)
    with structured_agent_event_context(run_id=run_id):
        if core_config.mode == "llm":
            built = _build_problem_with_llm_first_fallback(build_input, core_config=core_config, structured_client=structured_client)
        elif core_config.mode == "hybrid" and structured_client is not None:
            built = _build_problem_with_llm_first_fallback(build_input, core_config=core_config, structured_client=structured_client)
        else:
            built = build_physics_problem(build_input)
    if built.problem is None:
        return RunTypedPhysicsOSWorkflowOutput(build=built, missing_inputs=built.missing_inputs)

    initial_validation = validate_physics_problem(ValidatePhysicsProblemInput(problem=built.problem))
    workflow = run_physicsos_workflow(
        problem=built.problem,
        run_id=run_id,
        use_knowledge=input.use_knowledge,
        arxiv_max_results=input.arxiv_max_results,
        use_deepsearch=input.use_deepsearch,
        taps_rank=input.taps_rank,
        max_validation_attempts=input.max_validation_attempts,
        taps_max_wall_time_seconds=input.taps_max_wall_time_seconds,
        structured_client=structured_client,
        core_agent_config=core_config,
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
