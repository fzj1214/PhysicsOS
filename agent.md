# PhysicsOS Agent Architecture Review

## Naming Rules

This document uses the following names strictly.

- Main agent: only means the flexible DeepAgents CLI main agent.
- DeepAgents subagent: only means a DeepAgents/CLI-style delegated assistant, such as a future research/documentation subagent.
- PhysicsOS workflow: the LangGraph typed workflow for simulation. It is not called an agent.
- PhysicsOS workflow node: a LangGraph node inside the typed workflow. It is not called a subagent unless we are explicitly describing old or current confusing code.
- Tool: must be qualified as one of:
  - DeepAgents runtime tool,
  - DeepAgents-exposed PhysicsOS tool,
  - PhysicsOS workflow node capability,
  - registry/catalog entry.

The key correction is that "main agent" must not mean the LangGraph workflow entry node. The main agent is the DeepAgents CLI operator. The PhysicsOS/LangGraph side should be described as workflow/node/state, not as an agent hierarchy.

## Purpose

PhysicsOS currently mixes two different execution systems:

- DeepAgents CLI: an interactive, CLI-friendly operator layer.
- LangGraph/PhysicsOS workflow: a strict, stateful, typed simulation pipeline.

These systems use overlapping words, but the meanings are different. The current code and documentation blur those meanings, which causes inconsistent behavior, confusing tool ownership, and unclear responsibility between flexible CLI operation and strict simulation orchestration.

This document records the corrected interpretation and a refactor plan for review before implementation.

## Two Systems, Two Meanings

### DeepAgents CLI System

In the DeepAgents CLI system, the main agent is an interactive operator.

It is responsible for:

- Reading user natural language.
- Choosing exposed tools.
- Calling optional DeepAgents subagents.
- Using shell, filesystem, edit, and project-context capabilities.
- Explaining results.
- Inspecting artifacts.
- Editing files and running tests.
- Maintaining a conversational CLI/TUI experience.

In this system:

- The main agent is flexible and exploratory.
- DeepAgents runtime tools include shell/filesystem/edit/ask-user/memory/skills capabilities.
- DeepAgents-exposed PhysicsOS tools are Python functions registered as callable tools for the main agent.
- DeepAgents subagents are optional delegated assistants. They can be document/research/code-review style assistants in the future.
- Output is a streamed conversation, tool-call, and event experience.

This layer is useful for:

- Inspecting files.
- Running commands.
- Editing code.
- Reviewing logs.
- Explaining artifacts.
- Writing reports.
- Asking follow-up questions.
- Launching a strict PhysicsOS workflow when a simulation request is detected.

It is not the layer that should manually enforce every physics workflow step.

### LangGraph/PhysicsOS Workflow System

In the LangGraph/PhysicsOS system, the workflow is a typed state machine.

It is responsible for:

- Accepting structured input.
- Running nodes in a controlled order.
- Passing explicit typed state between nodes.
- Validating Pydantic input and output contracts.
- Retrying or stopping with structured error context.
- Producing structured result, events, and artifacts.

In this system:

- The PhysicsOS workflow is not the main agent.
- Workflow nodes are not DeepAgents subagents.
- Node capabilities are internal workflow implementation details unless explicitly exposed.
- State is explicit and typed.
- Output is a structured workflow result plus streamable events.

A core simulation path should look like:

```text
user request or typed PhysicsProblem
-> PhysicsOS workflow
-> build/accept PhysicsProblem
-> validate/canonicalize
-> knowledge context
-> geometry/mesh
-> TAPS/FEM/solver
-> verification
-> postprocess
-> case memory
-> structured result/events/artifacts
```

The priority here is physical correctness, reproducible state transfer, typed validation, and controlled branching.

## Current Problems

### 1. Current Tool Group Names Are Misleading

The current `MAIN_AGENT_TOOLS` name is confusing and should be treated as a legacy/transitional name.

Most tools in that group are not general tools for a flexible CLI main agent. They are closer to PhysicsOS workflow entry or PhysicsOS workflow-preparation tools:

- `build_physics_problem`
- `canonicalize_physics_problem`
- `validate_physics_problem`
- `run_typed_physicsos_workflow`
- `recommend_runtime_stack`
- catalog/listing helpers

This name makes it sound like these are all the tools the DeepAgents CLI main agent should naturally own. That is not necessarily true. Some are workflow entry tools, some are workflow-preparation tools, and some are catalog helpers.

The naming should be changed so the code expresses the actual boundary.

Proposed tool group names:

| Proposed name | Meaning | Should DeepAgents main agent see it? |
| --- | --- | --- |
| `DEEPAGENTS_MAIN_BRIDGE_TOOLS` | Small bridge tools exposed to the DeepAgents CLI main agent. These launch or inspect PhysicsOS workflows, but do not expose all workflow internals. | Yes |
| `PHYSICSOS_WORKFLOW_ENTRY_TOOLS` | High-level functions that start or resume a typed PhysicsOS workflow, especially `run_typed_physicsos_workflow`. | Usually yes, via bridge tools |
| `PHYSICSOS_PROBLEM_PREP_TOOLS` | Functions that build, canonicalize, validate, or inspect `PhysicsProblem` definitions. They are useful inside the PhysicsOS workflow or registry, but should not be exposed to the DeepAgents main agent by default. | No by default |
| `PHYSICSOS_CATALOG_TOOLS` | Read-only listing/recommendation helpers such as operator, solver, verification, and postprocess catalogs. | Yes |
| `PHYSICSOS_WORKFLOW_NODE_CAPABILITIES` | Internal node capabilities used by LangGraph workflow nodes, such as geometry planning, TAPS/FEM planning, numerical solve planning, verification, postprocess, and case memory. | No by default |
| `DEEPAGENTS_SUBAGENT_ASSISTANT_TOOLS` | Tools exposed to optional DeepAgents document/research/artifact-review subagents. | Only to those subagents |
| `PHYSICSOS_REGISTRY_TOOLS` | Full inventory/catalog of PhysicsOS Python tool functions. This is not a default LLM-visible tool set. | No by default |

Transitional mapping from current names:

| Current name | Problem | Target treatment |
| --- | --- | --- |
| `MAIN_AGENT_TOOLS` | Misleading: sounds like the full DeepAgents main-agent tool surface, but mostly contains PhysicsOS problem/workflow tools. | Replace with `DEEPAGENTS_MAIN_BRIDGE_TOOLS` or split into bridge + prep + catalog groups. |
| `PHYSICSOS_TOOLS` | Full registry currently used by `create_physicsos_agent()`, causing the Python-created DeepAgents agent to see 72 tools. | Rename conceptually to `PHYSICSOS_REGISTRY_TOOLS`; do not use as default DeepAgents main-agent tools. |
| `SUBAGENT_TOOL_GROUPS` | Blurs DeepAgents subagents with PhysicsOS workflow nodes and currently inherits confusing groups. | Rename/split into `DEEPAGENTS_SUBAGENT_TOOL_GROUPS`; keep separate from LangGraph workflow node capabilities. |
| `GEOMETRY_MESH_TOOLS`, `TAPS_TOOLS`, `SOLVER_TOOLS`, `VERIFICATION_TOOLS`, `POSTPROCESS_TOOLS` | These are mostly workflow-node capability groups, not necessarily DeepAgents subagent tool groups. | Treat as `PHYSICSOS_*_NODE_CAPABILITIES`; expose to DeepAgents subagents only by explicit design. |
| `KNOWLEDGE_TOOLS` | Can be used both by workflow nodes and document/research DeepAgents subagents. | Split into workflow knowledge capabilities and DeepAgents research/document assistant tools if needed. |

The target is not merely cosmetic renaming. The names should encode ownership:

```text
DeepAgents main agent sees bridge/catalog tools.
PhysicsOS workflow nodes use internal node capabilities.
DeepAgents subagents get assistant tools by role.
The full registry is only a registry.
```

### 2. There Are Two DeepAgents Entrypoints With Different Tool Sets

There are currently two ways to create a DeepAgents-powered PhysicsOS interface.

The CLI patch path modifies the generated DeepAgents CLI `server_graph.py` and registers:

```python
tools = [fetch_url, *wrap_tools_for_events(MAIN_AGENT_TOOLS)]
```

The Python API path in `physicsos/agents/main.py` defines:

```python
def create_physicsos_agent(...):
    ...
    return create_deep_agent(...)
```

Important clarification:

`create_physicsos_agent()` is not a LangGraph workflow function. It is a Python convenience function that creates a DeepAgents agent by calling `deepagents.create_deep_agent`.

It currently passes:

```python
tools = PHYSICSOS_TOOLS
subagents = SUBAGENTS
system_prompt = PHYSICSOS_SYSTEM_PROMPT
```

Therefore, `create_physicsos_agent()` creates a DeepAgents agent, not the PhysicsOS LangGraph workflow.

The confusion is that the CLI patch path gives the DeepAgents main agent a smaller scoped tool set, while `create_physicsos_agent()` gives the DeepAgents agent all 72 PhysicsOS registry tools.

Measured current schema footprint:

- CLI patch PhysicsOS tools: now based on `DEEPAGENTS_MAIN_BRIDGE_TOOLS`, 5 tools, about 22 KB of input schema.
- `create_physicsos_agent()` tools: currently `PHYSICSOS_TOOLS`, 72 tools, about 1.08 MB of input schema.
- `run_typed_physicsos_workflow` alone: about 20.8 KB of input schema.

This can produce different behavior, latency, and tool choices depending on which entrypoint is used.

### 3. Tool Concepts Are Mixed

There are several tool categories that must not be treated as one thing:

- DeepAgents runtime tools: shell, filesystem, edit, ask-user, memory, skills.
- DeepAgents-exposed PhysicsOS tools: Python functions made visible to the DeepAgents main agent.
- PhysicsOS workflow node capabilities: internal functions used by LangGraph workflow nodes.
- Registry/catalog entries: inventory of available tool functions, not necessarily public main-agent tools.

Filesystem and shell tools do not come from `physicsos.tools.registry`. They come from the DeepAgents CLI runtime/backend.

PhysicsOS registry tools are domain functions. They should not be assumed to define the complete CLI capability surface.

### 4. DeepAgents Subagents Are Not PhysicsOS Workflow Nodes

DeepAgents subagents are optional delegated assistants selected by the DeepAgents main agent.

PhysicsOS workflow nodes are strict LangGraph nodes inside the typed simulation workflow.

These should not be merged conceptually.

Current code blurs this boundary because `SUBAGENT_TOOL_GROUPS` gives each DeepAgents subagent `MAIN_AGENT_TOOLS` plus its scoped tools. Because `MAIN_AGENT_TOOLS` itself is not well named, this makes tool ownership hard to reason about.

Future DeepAgents subagents may include document-style assistants, for example:

- research subagent,
- documentation subagent,
- artifact-review subagent,
- report-writing subagent,
- code-inspection subagent.

Those subagents should be designed as flexible CLI assistants. They should not be confused with PhysicsOS workflow nodes.

### 5. Workflow Execution Is Displayed Like Main-Agent Thinking

Before a tool call, long thinking can come from:

- large tool schemas,
- long conversation context,
- uncertain tool choice,
- many exposed tools.

After `run_typed_physicsos_workflow` is called, long waiting comes from the PhysicsOS workflow execution:

- structured LLM calls,
- validation retries,
- geometry generation,
- solver execution,
- postprocess,
- artifact writing.

If CLI/TUI does not clearly show workflow events, users see a long "thinking..." state even when the DeepAgents main agent has already handed off to the PhysicsOS workflow.

The display should distinguish:

- main agent thinking,
- tool call started,
- PhysicsOS workflow running,
- workflow node started,
- workflow node output,
- retry/validation failure,
- artifact produced,
- workflow completed.

### 6. `problem: PhysicsProblem | None` Must Stay Accepted

`run_typed_physicsos_workflow` must continue accepting:

```python
problem: PhysicsProblem | None
```

This is an explicit product/architecture decision.

Reason:

- The DeepAgents main agent may already have clarified, repaired, or locked a typed `PhysicsProblem`.
- The workflow entrypoint must be able to accept that typed problem directly.
- The PhysicsOS workflow should not be forced to re-extract the problem from natural language after the typed problem is already known.

Important clarification:

`PhysicsProblem` is not a tool. It is part of the tool input schema for `run_typed_physicsos_workflow`.

That means the schema can be large, but removing `problem: PhysicsProblem | None` is not an acceptable solution. Any schema-size mitigation must preserve this accepted input.

Possible future mitigations, if needed, must be additive and reviewed, such as:

- improving descriptions,
- constraining other open-ended fields,
- adding clearer prompts for when to pass `problem`,
- using references/artifacts as an additional option,
- improving CLI/TUI display so workflow waits are not mistaken for thinking.

### 7. Invalid Open-Ended Tool Arguments

`core_agents_mode` is currently too open if typed as a plain string. The DeepAgents main agent can invent unsupported values such as:

```text
core_agents_mode="typed"
```

If the workflow only supports values such as `llm`, `hybrid`, and `deterministic`, this causes outputs like:

```text
missing_inputs=["unsupported_core_agents_mode"]
assumptions=["Unsupported core_agents_mode='typed'."]
```

The root issue is that the public tool contract does not constrain or normalize allowed values tightly enough.

This should be fixed without removing `problem: PhysicsProblem | None`.

## Desired Architecture

### Boundary

```text
DeepAgents CLI main agent
= flexible shell/chat/document/report/operator layer

PhysicsOS LangGraph workflow
= strict typed simulation pipeline

DeepAgents-exposed PhysicsOS tools
= small bridge from CLI to workflow plus selected query/catalog/artifact helpers

PhysicsOS workflow node capabilities
= internal implementation details used by typed workflow nodes

DeepAgents subagents
= optional exploratory/document/research assistants, not PhysicsOS workflow nodes
```

### DeepAgents Main Agent Role

The main agent should remain CLI-friendly and flexible.

It should be able to:

- Read and edit project files through DeepAgents runtime tools.
- Run shell commands through DeepAgents runtime tools.
- Inspect artifacts.
- Summarize workflow outputs.
- Ask clarifying questions.
- Help the user revise modeling requirements.
- Launch the PhysicsOS typed workflow when appropriate.

It should not manually orchestrate the core simulation workflow step by step.

### PhysicsOS Workflow Role

The PhysicsOS typed workflow should take over once the user asks for a concrete simulation/modeling task.

It should own:

- Natural-language problem extraction or typed `PhysicsProblem` intake.
- Canonicalization.
- Validation.
- Knowledge context.
- Geometry/mesh.
- TAPS/FEM/solver routing.
- Verification.
- Postprocess.
- Case memory.
- Structured event emission.

### DeepAgents Subagent Role

DeepAgents subagents should be optional helper agents.

They are suitable for:

- Research and documentation lookup.
- Inspecting files.
- Reviewing artifacts.
- Drafting explanations.
- Searching knowledge.
- Helping repair user-facing context.

They are not the same as PhysicsOS workflow nodes.

## Proposed Refactor Plan

No implementation should happen until this plan is reviewed.

### Phase 1: Rename And Clarify Tool Groups

Goal: make tool group names describe ownership and visibility before changing behavior.

Steps:

1. Audit every current tool group in `physicsos/tools/registry.py`.
2. Introduce explicit names:
   - `DEEPAGENTS_MAIN_BRIDGE_TOOLS`
   - `PHYSICSOS_WORKFLOW_ENTRY_TOOLS`
   - `PHYSICSOS_PROBLEM_PREP_TOOLS`
   - `PHYSICSOS_CATALOG_TOOLS`
   - `PHYSICSOS_WORKFLOW_NODE_CAPABILITIES`
   - `DEEPAGENTS_SUBAGENT_ASSISTANT_TOOLS`
   - `PHYSICSOS_REGISTRY_TOOLS`
3. Keep compatibility aliases initially:
   - `MAIN_AGENT_TOOLS = DEEPAGENTS_MAIN_BRIDGE_TOOLS`
   - `PHYSICSOS_TOOLS = PHYSICSOS_REGISTRY_TOOLS`
   - `SUBAGENT_TOOL_GROUPS = DEEPAGENTS_SUBAGENT_TOOL_GROUPS`
4. Add code comments saying these aliases are temporary compatibility names.
5. Do not change runtime behavior in this phase unless the rename mechanically requires imports to move.
6. Add tests that assert the aliases point to the intended new groups.
7. Add comments explaining that the DeepAgents main agent is the only "main agent" in this document.

Initial proposed grouping:

```python
PHYSICSOS_WORKFLOW_ENTRY_TOOLS = [
    run_typed_physicsos_workflow,
]

PHYSICSOS_PROBLEM_PREP_TOOLS = [
    build_physics_problem,
    canonicalize_physics_problem,
    validate_physics_problem,
    recommend_runtime_stack,
]

PHYSICSOS_CATALOG_TOOLS = [
    list_operator_templates,
    list_solver_backends,
    list_verification_rules,
    list_postprocess_templates,
]

DEEPAGENTS_MAIN_BRIDGE_TOOLS = unique(
    PHYSICSOS_WORKFLOW_ENTRY_TOOLS,
    PHYSICSOS_CATALOG_TOOLS,
)

PHYSICSOS_WORKFLOW_NODE_CAPABILITIES = unique(
    GEOMETRY_MESH_TOOLS,
    TAPS_TOOLS,
    SOLVER_TOOLS,
    VERIFICATION_TOOLS,
    POSTPROCESS_TOOLS,
    KNOWLEDGE_TOOLS,
)

PHYSICSOS_REGISTRY_TOOLS = unique(
    DEEPAGENTS_MAIN_BRIDGE_TOOLS,
    PHYSICSOS_PROBLEM_PREP_TOOLS,
    PHYSICSOS_WORKFLOW_NODE_CAPABILITIES,
)
```

The exact contents can be refined later. The important first step is that the name tells the reader whether the group is LLM-visible to the DeepAgents main agent or internal to PhysicsOS workflow execution.

Expected result:

```text
The code no longer uses MAIN_AGENT_TOOLS as if it were a clean concept.
Tool group names reveal DeepAgents visibility versus PhysicsOS workflow ownership.
```

### Phase 2: Make DeepAgents Entrypoints Consistent

Goal: remove divergence between CLI-created and Python-created DeepAgents agents.

Status: implemented in code after Phase 1.

Steps:

1. Decide the DeepAgents main agent's PhysicsOS tool policy.
2. Update `create_physicsos_agent()` so it creates the same kind of DeepAgents main agent as the CLI patch path.
3. Do not treat `create_physicsos_agent()` as a LangGraph workflow constructor.
4. Keep the actual PhysicsOS LangGraph workflow entry separate.
5. Add tests that compare the CLI patch tool policy and `create_physicsos_agent()` tool policy.

Expected result:

```text
Both DeepAgents entrypoints expose the same intended PhysicsOS tool surface.
```

Implemented policy:

```text
DeepAgents main agent PhysicsOS tools = DEEPAGENTS_MAIN_BRIDGE_TOOLS
PhysicsOS workflow entry tool = run_typed_physicsos_workflow
Problem-prep tools stay in PHYSICSOS_PROBLEM_PREP_TOOLS/PHYSICSOS_REGISTRY_TOOLS, not in the DeepAgents main bridge by default
Full registry = PHYSICSOS_REGISTRY_TOOLS, not exposed to the DeepAgents main agent by default
```

Current measured schema footprint after this phase:

- `DEEPAGENTS_MAIN_BRIDGE_TOOLS`: 5 tools, about 22 KB of input schema.
- `PHYSICSOS_REGISTRY_TOOLS`: 72 tools, about 1.08 MB of input schema.

`create_physicsos_agent()` still creates a DeepAgents agent. It does not create the LangGraph/PhysicsOS workflow. The bridge into that workflow is the DeepAgents-exposed tool `run_typed_physicsos_workflow`.

### Phase 3: Separate DeepAgents Subagents From PhysicsOS Workflow Nodes

Goal: prevent recursive or confusing orchestration.

Status: implemented for the current built-in workflow-node names.

Steps:

1. Define DeepAgents subagents as optional flexible assistants.
2. Define PhysicsOS workflow nodes separately in workflow code/docs.
3. Stop describing LangGraph workflow nodes as subagents in docs.
4. Review whether DeepAgents subagents should inherit any workflow entry tools.
5. Allow future document/research-style DeepAgents subagents without conflating them with typed workflow nodes.

Expected result:

```text
DeepAgents subagents assist the CLI experience.
PhysicsOS workflow nodes execute the typed simulation pipeline.
```

Implemented policy:

```text
geometry-mesh-agent, taps-agent, solver-agent, verification-agent, postprocess-agent, and knowledge-agent are not registered as DeepAgents subagents.
The DeepAgents CLI patch does not inject tools into those names.
The DeepAgents config generator does not write .deepagents/physicsos/agents/<workflow-node>/AGENTS.md files.
Workflow-internal node names are preserved temporarily for trace/event compatibility.
Future DeepAgents subagents must use distinct document/research/artifact assistant names.
```

### Phase 4: Preserve And Clarify `run_typed_physicsos_workflow(problem=...)`

Goal: keep `problem: PhysicsProblem | None` as the workflow state handoff while making the entrypoint contract unambiguous.

Status: implemented for the public input contract and structured build path.

Steps:

1. Keep `problem: PhysicsProblem | None` in `RunTypedPhysicsOSWorkflowInput`.
2. Remove `messages` from `RunTypedPhysicsOSWorkflowInput`; the current user request and optional current `problem` state are the contract.
3. Define the branch strictly by whether `problem` is `None`:
   - `problem is None`: build a new `PhysicsProblem` from `user_request`.
   - `problem is not None`: use `problem` as the current typed PhysicsProblem state and apply `user_request` as an update/repair/continuation through `build_physics_problem`.
4. Do not skip `build_physics_problem` just because `problem` is provided. The build step remains the state creation/update step.
5. Make prompt/tool descriptions clear that the DeepAgents main agent should call this single entrypoint and should not call problem-prep tools directly.
6. Constrain or document `core_agents_mode` so unsupported invented values are avoided.
7. Do not replace `problem` with JSON/ref as the main path.
8. Any optional JSON/ref handoff must be additive, not a replacement.

Expected result:

```text
The DeepAgents main agent calls run_typed_physicsos_workflow(user_request=..., problem=None) when no current problem exists.
The DeepAgents main agent calls run_typed_physicsos_workflow(user_request=..., problem=current_problem) when a typed state exists.
The PhysicsOS workflow always performs build_physics_problem as create/update state logic before canonicalization, validation, and solve.
```

### Phase 5: Improve CLI/TUI Workflow Event Display

Goal: distinguish main-agent thinking from PhysicsOS workflow execution.

Status: completed without further structural migration.

Decision:

```text
Do not rename the existing workflow event schema from agent.* to node.* in this phase.
The existing PhysicsOS custom stream events already carry workflow progress through stage/status/summary.
CLI/TUI display should continue to render those events compactly through PhysicsOSEventRenderer.
Any future node-oriented wording should be a display-layer mapping only, not a breaking event-schema migration.
```

Rationale:

```text
The current workflow already emits stable progress events:
workflow.started, case_memory.hit, agent.started, agent.output, validation.retry, workflow.completed.
Structured LLM attempts include attempt/max_attempts and raw-response artifacts.
Artifacts are surfaced through event.artifacts rather than requiring separate artifact.created events.
Keeping this shape avoids breaking existing tests, event logs, case-memory traces, and LangGraph custom-stream consumers.
```

Steps:

1. Keep emitting structured events from the PhysicsOS workflow and structured attempts.
2. Ensure CLI/TUI subscribes to custom stream events.
3. Display clear event labels:
   - `workflow.started`
   - `node.started`
   - `node.output`
   - `validation.retry`
   - `artifact.created`
   - `workflow.completed`
4. Show node name and attempt count.
5. Surface raw structured LLM attempts as artifacts, not inline walls of JSON.
6. Keep the visual style consistent with existing DeepAgents CLI/TUI output.

Expected result:

```text
thinking...
calling run_typed_physicsos_workflow
PhysicsOS workflow started
knowledge node complete
geometry/mesh node retry 1/5
TAPS/FEM/solver node complete
artifact created
workflow completed
```

Implemented interpretation:

```text
DeepAgents CLI/TUI subscribes to custom stream events.
PhysicsOSEventRenderer renders compact [stage] summary lines.
Workflow-node progress is represented by stage/status/summary on existing agent.* events.
Structured attempt JSON is written to artifacts and linked from the rendered event line.
No Phase 5 schema migration is planned.
```

### Phase 6: Document Runtime Tool Sources

Goal: eliminate confusion about where shell/filesystem/edit tools come from.

Steps:

1. Document that `ls`, `edit`, `read`, `write`, and `execute` are DeepAgents CLI runtime/backend capabilities.
2. Document that PhysicsOS registry tools are domain/workflow tools only.
3. Document that PhysicsOS workflow node capabilities are internal workflow implementation tools.
4. Add a developer note explaining how tool schemas reach the LLM.

Expected result:

```text
No one expects filesystem tools to appear in physicsos.tools.registry.
```

### Phase 7: Add Architecture Tests

Goal: prevent the same confusion from returning.

Tests to add:

1. DeepAgents main tool set is stable and intentionally scoped.
2. `create_physicsos_agent()` uses the same DeepAgents PhysicsOS tool policy as the CLI patch.
3. DeepAgents subagent tool groups do not accidentally include workflow entry tools unless explicitly allowed.
4. `run_typed_physicsos_workflow` continues accepting `problem: PhysicsProblem | None`.
5. `core_agents_mode` rejects or normalizes unsupported values.
6. PhysicsOS workflow emits streamable events for every major workflow node.

## Decisions Needed Before Implementation

1. Should the first rename use `DEEPAGENTS_MAIN_BRIDGE_TOOLS`, or do we want a different final name?
2. Should `create_physicsos_agent()` be updated to mirror the CLI patch tool policy?
3. Which PhysicsOS tools should the DeepAgents main agent see by default?
4. Should DeepAgents subagents inherit any workflow entry tools?
5. Should `core_agents_mode` remain public, become constrained, or be removed from the public tool input?
6. What DeepAgents document/research subagents should be planned separately from PhysicsOS workflow nodes?

## Recommended Direction

Use this boundary:

```text
DeepAgents CLI main agent = flexible user-facing operator
PhysicsOS LangGraph workflow = strict typed simulation workflow
DeepAgents-exposed PhysicsOS tools = small bridge between the two
PhysicsOS workflow node capabilities = internal workflow implementation
DeepAgents subagents = optional document/research/artifact assistants
```

The DeepAgents main agent should stay flexible and CLI-friendly. When a concrete simulation request is identified, it should call a small, reliable PhysicsOS workflow entrypoint and let the LangGraph workflow own the physical logic.
