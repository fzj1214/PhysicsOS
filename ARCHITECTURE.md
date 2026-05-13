# PhysicsOS Architecture

PhysicsOS should closely follow `2509.11447v1.pdf`. Except for the added STL/Gmsh geometry module, the system is a paper-style CAE agent for TAPS prompt engineering: the LLM uses analysis files, tools, online/local resources, and context-window examples to derive TAPS equations, implement case-local code, verify it, and revise it.

The system must not become a fixed numerical solver framework, a typed TAPS IR compiler, or a LangGraph workflow. The one-dimensional heat/TAPS derivation in the paper is a one-shot example for prompt engineering, not a solver to hard-code.

## 1. Paper Architecture

The paper's CAE agent has four modules. PhysicsOS maps them directly:

```text
analysis files
  -> user problem statement, uploaded notes/scripts, generated problem brief, STL/CAD geometry files

tools
  -> derivation prompt builder, case-local code scaffold, code execution, exact solution code generation,
     convergence code generation, plotting, geometry preprocessing tools

online resources
  -> local references first; optional web/arXiv lookup only when needed

context window
  -> system prompt, paper-style one-shot examples, matrix definitions, CoT outline,
     current problem statement, geometry embedding notes
```

DeepAgents CLI is only the agent harness for these modules. It provides subagents, filesystem access, tool calling, and todo maintenance. It is not the physics architecture.

## 2. Main Route

The main route follows the paper's loop:

```text
problem / analysis files
-> context window assembled from analysis files, tools, resources, and examples
-> paper-style TAPS derivation prompt
-> mathematical derivation with step-by-step subspace iteration
-> case-local code implementation
-> exact/manufactured solution code generation
-> convergence code generation
-> execution
-> plotting and inspection
-> revise derivation or code if verification fails
```

PhysicsOS adds one geometry preprocessing branch before derivation when STL/CAD is present:

```text
STL/CAD or natural-language geometry with dimensions
-> if natural language: generate a simple STL primitive first
-> Gmsh preprocessing
-> Cartesian background grid
-> SDF / voxel occupancy / boundary samples / normals / cut cells
-> geometry embedding notes
-> included as an extra input to the TAPS derivation prompt
```

The old `run_typed_physicsos_workflow` / `run_physicsos_workflow` path is not part of this architecture. Legacy typed TAPS IR schemas, separated-operator backends, and fixed LangGraph node workflows must stay removed from the default route.

## 3. Prompt Design

The derivation prompt should replicate Fig. 4 and Appendix D:

```text
Role-playing:
You are a computational mechanics expert tasked with making targeted corrections to a mathematical derivation.

Task:
Create the mathematical derivation for the given problem statement based on the template example.

Input:
- Template example
- Problem statement
- Matrix definition
- Geometry embedding notes          # PhysicsOS extension only when geometry exists

Required actions:
1. Start with the complete template derivation.
2. Replace only the parts required by the new PDE, coefficients, parameters, boundary conditions, and geometry.
3. Preserve the valid TAPS structure from the example.
4. Derive every subspace iteration step by step.

Important:
- Start with the weak form.
- Insert trial and test functions into the weak form.
- Define mass, stiffness, coefficient, source, boundary, and geometry matrices before use.
- Derive the final matrix for the current subspace iteration.
- Repeat for each spatial, parameter, temporal, and geometry axis.
- Do not jump directly to the final matrix form.

Output:
- `/cases/<case_id>/taps/derivation.md`
- `/cases/<case_id>/taps/implementation_notes.md`
```

The visible derivation artifact is allowed and expected to show mathematical reasoning. This is not hidden model chain-of-thought; it is the paper's inspectable derivation content.

## 4. Reference Files

Each case should copy local references into `/cases/<case_id>/references/`:

```text
taps_template_eq5.md              # one-shot derivation example from the paper style
taps_matrix_definitions.md        # allowed matrix vocabulary
taps_cot_outline.md               # step-by-step derivation outline like Fig. 5/6
taps_verification_workflow.md     # Fig. 7 tool chain
ibm_ife_geometry_embedding.md     # PhysicsOS geometry extension
```

The template is a worked example. The implementation agent must not treat it as a fixed solver. It should use the example the way the paper uses it: copy the structure, replace the problem-specific PDE and matrices, and keep all subspace derivations explicit.

## 5. Agent Roles

### physicsos-main

Role: orchestrate the paper loop.

Responsibilities:

- Create or reuse `/cases/<case_id>/`.
- Maintain a visible todo list.
- Use `update_case_stage_status` to keep `/cases/<case_id>/execution_plan.md` and `manifest.json` synchronized as each paper-loop stage completes. This is audit state, not hidden orchestration.
- Keep the current problem, references, derivation, generated code, verification scripts, and plots in files.
- Build or refresh `/cases/<case_id>/context/context_window.md` after analysis files, references, or geometry notes change.
- Delegate work to subagents.
- Retry derivation or implementation when verification fails.
- Never call the old typed LangGraph workflow.
- Expose only paper-route bridge tools to the main agent: case creation, stage status, reference loading, context-window building, and derivation-prompt building.

### analysis-file-agent

Role: prepare the paper's analysis-files module.

Responsibilities:

- Read user text, uploaded scripts, notes, and geometry file references.
- Extract PDE, fields, coefficients, parameters, boundary/initial conditions, units, outputs, and missing data.
- Write:

```text
/cases/<case_id>/problem/problem_statement.md
/cases/<case_id>/problem/problem.json
/cases/<case_id>/problem/open_questions.md
```

Domain tools: stage status plus optional local knowledge/case-memory lookup. Use DeepAgents filesystem tools for reading and writing problem artifacts.

### geometry-embedding-agent

Role: PhysicsOS extension for CAD/STL geometry.

Responsibilities:

- Prefer the one-step tool `prepare_geometry_analysis_files` for the standard route.
- Import STL/CAD files.
- For simple natural-language geometry with explicit dimensions, generate a simple box/sphere/cylinder STL primitive and then use the same STL path.
- For composite or nontrivial natural-language geometry, the geometry agent should use DeepAgents filesystem tools to author case-local geometry source files, notes, or STL/CSG artifacts. Do not hard-code natural-language geometry parsers in the tool layer.
- Prepare Gmsh geometry artifacts.
- Generate background grid, SDF or fallback SDF, occupancy, boundary samples, normals, and cut-cell metadata.
- Generate `sdf_quality.json` so fallback SDF quality, occupancy coverage, normal degeneracy, and cut-cell availability are visible before derivation/implementation.
- Write geometry notes for the derivation prompt.
- Write a geometry handoff that tells the derivation, implementation, and verification agents how to consume STL/CAD embedding artifacts inside the paper-style TAPS loop.
- Do not solve the PDE.
- Domain tools: only the STL/Gmsh/background-grid/SDF/voxel/embedding tools plus optional local knowledge lookup. Legacy mesh-backend/export/labeling workflow tools are not part of this subagent surface.

Outputs:

```text
/cases/<case_id>/geometry/input.stl
/cases/<case_id>/geometry/generated_geometry.json
/cases/<case_id>/geometry/gmsh_model.geo
/cases/<case_id>/geometry/gmsh_sampled_sdf.json
/cases/<case_id>/geometry/background_grid.json
/cases/<case_id>/geometry/sdf.npy
/cases/<case_id>/geometry/sdf_quality.json
/cases/<case_id>/geometry/occupancy.npy
/cases/<case_id>/geometry/boundary_samples.npy
/cases/<case_id>/geometry/normals.npy
/cases/<case_id>/geometry/cut_cells.npy
/cases/<case_id>/geometry/embedding.json
/cases/<case_id>/geometry/geometry_embedding.md
/cases/<case_id>/geometry/taps_geometry_context.md
/cases/<case_id>/geometry/taps_geometry_handoff.md
```

### taps-derivation-agent

Role: reproduce the paper's mathematical derivation behavior.

Responsibilities:

- Load the template derivation, matrix definitions, CoT outline, problem statement, and geometry notes.
- Produce a complete derivation, not only a final matrix.
- Explicitly derive the weak form, C-HiDeNN-TD approximation, and every subspace iteration.
- For STL geometry, show how `phi(x)`, `chi(x)`, boundary samples, normals, and cut cells enter coefficient or boundary matrices.
- Read `/geometry/taps_geometry_handoff.md` when present so geometry coupling is propagated into the derivation rather than left as generic notes.
- Domain tools: derivation-prompt builder, local knowledge context, geometry-readiness assessment, derivation summary, and paper-route TAPS problem manifest. Use DeepAgents filesystem tools for derivation files.

Outputs:

```text
/cases/<case_id>/taps/derivation_prompt.md
/cases/<case_id>/taps/derivation.md
/cases/<case_id>/taps/implementation_notes.md
```

### taps-implementation-agent

Role: reproduce the paper's code-implementation step.

Responsibilities:

- Use the one-shot derivation and implementation prompt as examples.
- Replace the scaffold in `/cases/<case_id>/taps/kernel.py` with generated case-local code.
- Preserve the derivation's matrix and subspace-iteration structure.
- Read `/geometry/taps_geometry_handoff.md` when present and load/validate the listed geometry artifacts case-locally.
- Write executable code only for the current case.
- If derivation or physics inputs are missing, fail clearly instead of inventing a result.
- Domain tools: prompt package/scaffold generation, static check, case-local spec review, and execution of generated `kernel.py`. Do not expose support estimators, backend bundle tools, or old runtime-extension helpers to this subagent.

Outputs:

```text
/cases/<case_id>/taps/implementation_prompt.md
/cases/<case_id>/taps/implementation_manifest.json
/cases/<case_id>/taps/kernel.py
/cases/<case_id>/taps/execution_plan.json
/cases/<case_id>/taps/static_review.md
/cases/<case_id>/taps/kernel_review_spec.json
/cases/<case_id>/taps/generated_kernel_review.json
/cases/<case_id>/taps/generated_kernel_review.md
```

### verification-agent

Role: reproduce Fig. 7.

Tool chain:

```text
generate_exact_sol_code
execute_exact_sol_code
generate_convergence_code
execute_convergence_code
plot_result
```

Responsibilities:

- Generate and run exact/manufactured solution code when possible.
- Generate and run convergence-study code.
- For geometry cases, report which geometry artifacts are present and whether boundary/SDF evidence exists; do not treat preprocessing as numerical verification.
- Plot results.
- Report whether code should be accepted or revised.
- Domain tools: only the Fig. 7 chain plus stage status. Additional residual/conservation/OOD helper tools are not exposed in the DeepAgents subagent surface unless they become part of a paper-derived verification prompt.

Outputs:

```text
/cases/<case_id>/verification/exact_solution.py
/cases/<case_id>/verification/exact_solution.json
/cases/<case_id>/verification/convergence_study.py
/cases/<case_id>/verification/convergence_report.json
/cases/<case_id>/verification/plots/
/cases/<case_id>/verification/report.md
/cases/<case_id>/verification/report.json
```

### postprocess-agent

Role: summarize verified results.

Responsibilities:

- Plot final fields and convergence/residual figures.
- Write assumptions, evidence, warnings, and recommendations.
- Domain tools: KPI extraction, visualization generation, and report writing. Planning should be handled in the agent prompt/files, not by a deterministic postprocess planner tool.

## 6. Geometry Extension

Geometry is the main PhysicsOS addition beyond the paper.

For STL/CAD, use:

```text
Omega_bg = Cartesian background domain
phi(x)   = signed distance / level set from STL or Gmsh
chi(x)   = H(-phi(x)) occupancy / characteristic function
```

The derivation prompt should add geometry terms to the weak form:

```text
Integral_Omega_bg chi k grad(v) . grad(u) dOmega
+ boundary_constraint_terms(phi, u, g, v)
= Integral_Omega_bg chi v f dOmega
+ Neumann_terms(phi, h, v)
```

Gmsh is a preprocessing tool for distance and geometry artifacts. It is not a PDE solver in this route. If Gmsh sampling fails or times out, the tool must write an explicit fallback manifest; it must not pretend a production distance field succeeded.

The geometry module should maximize its value through a strong handoff rather than an independent workflow:

```text
geometry artifacts
-> taps_geometry_context.md      # what enters the weak form
-> taps_geometry_handoff.md      # how derivation, implementation, and verification agents consume the artifacts
-> context_window.md             # exposes the handoff to the paper-style main loop
```

The handoff must explicitly cover:

- Derivation: `chi`-weighted volume terms, `phi`-derived boundary constraints, normals, cut-cell quadrature, and optional geometry parameter axes.
- Implementation: case-local loading of background grid, SDF, `sdf_quality.json`, occupancy, boundary samples, normals, and cut cells; shape consistency checks; clear failure on missing geometry.
- Verification: report geometry artifacts and SDF quality consumed or missing, state whether errors are measured on `Omega_bg` or the `chi`-weighted physical domain, and keep geometry preprocessing separate from numerical verification.

## 7. Case Filesystem

```text
/cases/<case_id>/
  problem/
    problem_statement.md
    problem.json
    open_questions.md
  references/
    taps_template_eq5.md
    taps_matrix_definitions.md
    taps_cot_outline.md
    taps_verification_workflow.md
    ibm_ife_geometry_embedding.md
  context/
    context_window.md
    context_window.json
  geometry/
    input.stl
    generated_geometry.json
    gmsh_model.geo
    gmsh_sampled_sdf.json
    background_grid.json
    sdf.npy
    sdf_quality.json
    occupancy.npy
    boundary_samples.npy
    normals.npy
    cut_cells.npy
    embedding.json
    geometry_embedding.md
    taps_geometry_context.md
    taps_geometry_handoff.md
  taps/
    derivation_prompt.md
    derivation.md
    implementation_notes.md
    implementation_prompt.md
    implementation_manifest.json
    kernel_review_spec.json
    execution_plan.json
    kernel.py
    static_review.md
  verification/
    exact_solution.py
    exact_solution.json
    convergence_study.py
    convergence_report.json
    plots/
    report.md
    report.json
  report/
    figures/
    report.md
    summary.json
  events.jsonl
  manifest.json
```

## 8. Implementation Plan

[done] Remove old LangGraph typed workflow from the default route.

[done] Remove legacy typed TAPS IR schemas/tools/backends from the default implementation.

[done] Reframe TAPS as paper-style prompt engineering instead of a fixed built-in numerical executor.

[done] Add case-local paper references and prompt builders.

[done] Add a case-local `context_window.md/json` artifact that packages the paper's four modules: analysis files, tools, online/local resources, and context examples. This is a prompt context artifact, not a workflow engine.

[done] Add geometry preprocessing tools for STL, Gmsh artifacts, background grids, SDF/occupancy, normals, and cut cells.

[done] Add `prepare_geometry_analysis_files` so STL files and natural-language primitive geometry become one derivation-ready geometry context for the paper loop.

[done] Add `implementation_prompt.md`, `implementation_manifest.json`, and a scaffold `kernel.py` that refuses to fabricate numerical output.

[done] Add Fig. 7 style verification tool chain.

[done] Tighten `taps-derivation-agent` and `build_taps_derivation_prompt` so their text mirrors Appendix D's five prompt parts: role-playing, few-shot prompt, constraints, chain-of-thought derivation requirements, and formatting guidelines.

[done] Tighten `taps-implementation-agent` so it behaves like the paper's implementation step from one-shot examples, not like a prebuilt solver wrapper.

[done] Add spec-driven review checks for generated case-local code. Review criteria live in `/taps/kernel_review_spec.json`; the tool only interprets the case-local spec and must not hard-code fixed review rules.

[done] Improve geometry prompts so STL/Gmsh artifacts are introduced only as the PhysicsOS extension to the paper route, supplying analysis files rather than solver or verification behavior.

[done] Add a strong geometry handoff so STL/CAD embedding artifacts are propagated into derivation, implementation, and verification responsibilities while staying inside the paper-style TAPS prompt-engineering loop.

[done] Add visible case-stage status maintenance through `update_case_stage_status`, keeping `[done]/[todo]` markers in `execution_plan.md` without reintroducing a workflow engine.

[done] Prune DeepAgents subagent PhysicsOS tool surfaces to the minimum paper-route responsibilities while retaining DeepAgents native filesystem/shell tools through middleware.
