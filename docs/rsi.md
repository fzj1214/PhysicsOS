# Runtime strategy improvement

`physicsos.rsi.RSIRuntime` improves reusable strategy revisions using the same
`CaseRuntime` that prepares geometry, executes case-local kernels and collects
independent numerical evidence. It contains no PDE-specific evaluation loop.

```text
scope + outcome diagnostics
  -> agent-authored strategy revision
  -> frozen development evaluation
  -> selected revision + fresh holdout comparison
  -> guarded promotion
  -> next production solve + pre-run capability estimate
  -> observed outcomes / regression rollback
```

## Strategy and problem boundaries

A scope declares `problem_family`, physics domains, regime, dimension,
representation and interior/exterior domain side. Use a stable family name
that identifies the governing problem and boundary assumptions. The runtime
checks scope equality; benchmark authors define the physical meaning of a
family. Scope membership is not inferred from arbitrary Python code.

A strategy revision freezes:

- Mesh policy and optional PaMO repair settings.
- Solver controls explicitly allowed by each problem's `tunable_controls`.
- Generation guidance and optional reusable kernel or kernel builder.
- Parent revision, implementation files and their checksums.

Geometry, physical parameters, boundary requirements, element quality limits,
independent references and acceptance thresholds belong to the benchmark.
A strategy cannot override undeclared controls or weaken repair fidelity,
surface quality or validation sample requirements. Benchmark authors list only
numerical controls as tunable; material coefficients and physical forcing
parameters remain fixed.

Native meshing limitations remain CaseRuntime limitations. Requests for missing
boundary-layer/local-refinement providers, unsupported sampling or unavailable
CUDA remain unresolved and cannot qualify a strategy for promotion.

## Register immutable inputs

```python
from physicsos.rsi import RSIRuntime
from physicsos.schemas.mesh import MeshPolicy
from physicsos.schemas.rsi import (
    StrategyScope, StrategySpec, BenchmarkCase, BenchmarkSuiteInput,
    RefinementCheck, EvaluateStrategiesInput, AssessCapabilityInput,
)

rsi = RSIRuntime()
scope = StrategyScope(
    problem_family="poisson-dirichlet",
    physics_domains=["thermal"], regime="steady-conduction",
    dimension=3, representation="mesh",
)
candidate = rsi.register_strategy(StrategySpec(
    name="refined-p1", scope=scope,
    description="Reassemble P1 FEM on a globally refined mesh.",
    guidance="Use the current mesh and explicit boundary groups; retain the forcing and material data.",
    mesh_policy=MeshPolicy(target_element_size=0.2),
    kernel_uri="strategies/p1/kernel.py",
))
```

`register_suite` accepts a `BenchmarkSuiteInput` containing `BenchmarkCase`
objects. Each case provides a `PrepareDomainInput`, physical controls,
independent reference implementation and field thresholds. By default, every
case also needs a `RefinementCheck`; the suite requires at least one development
and three distinct holdout problems. Authors can explicitly configure an
accuracy-only protocol with `require_convergence=False`.

```python
# These prepare requests describe different physical geometries/parameters.
benchmarks = [
    BenchmarkCase(
        id=f"geometry-{index}",
        split="development" if index == 0 else "holdout",
        prepare=prepare_request, controls=physical_controls,
        kernel_uri="strategies/p1/kernel.py",
        reference_uri="references/poisson_reference.py",
        relative_tolerance=0.05,
        convergence=RefinementCheck(refinements=[0.4, 0.25, 0.15]),
    )
    for index, (prepare_request, physical_controls) in enumerate(problem_variants)
]
suite = rsi.register_suite(BenchmarkSuiteInput(
    name="poisson-geometry-suite", scope=scope, benchmarks=benchmarks,
))
```

Registration snapshots source geometry, kernel/reference modules, sibling
Python helpers and case resources. Changing the authoring files afterwards
does not change a registered evaluation. Existing asset checksums are enforced.
Self-contained CAD/mesh assets are required, as in CaseRuntime.

Problem identity excludes case IDs, file paths, discretization and acceptance
tolerances. It includes source content, declared geometry/coordinates,
boundary intent, physics and non-tunable controls. Insignificant floating-point
representation differences are normalized. Renamed copies cannot cross the
development/holdout split or increase the independent problem count. Identity
is based on the declared data; geometric equivalence across different asset
encodings is not established automatically.

## Candidate selection and promotion

```python
evaluation = rsi.evaluate(EvaluateStrategiesInput(
    suite=suite.manifest,
    candidates=[candidate.manifest],
    max_kernel_runs=128,
    timeout_seconds=60,
))
```

Evaluation performs actual preparation, execution, independent field comparison
and requested refinement reruns. Each attempt has a fresh case workspace.
Baseline and candidates compare at one checksummed physical probe set for each
benchmark. Mesh probes use deterministic interior barycentric coordinates to
reduce coincidence with refinement nodes. The norm remains weighted probe RMS,
with the sampling limits described in [CaseRuntime](case_runtime.md).

The current default is the baseline. All candidates are ranked on development
data by verified coverage and normalized field error. Only the selected
candidate and baseline proceed to holdouts. Selection is saved before holdout
execution. Each holdout problem can be exposed by one evaluation campaign;
renamed suites and partially overlapping holdout sets also require fresh data.
`include_holdout=False` supports development experiments without consuming
holdouts or qualifying a promotion.

Exposure is tracked across campaigns and strategy revisions. A problem already
used for development cannot later be relabelled as a fresh holdout. Existing
runtime observation history is migrated into the development-exposure index.

Promotion requires:

1. Every candidate development and holdout comparison passes the frozen checks.
2. Required refinement evidence comes from actual rebuilds and solves.
3. Every previously verified baseline holdout remains verified, with no error
   regression beyond the fixed allowance.
4. Against an incumbent, holdout error improves by the declared amount, or the
   candidate verifies a problem on which the baseline demonstrably failed.
5. All strategy, suite, output and reference evidence is still intact.
6. The registry generation still matches the baseline used for evaluation.

An inconclusive baseline does not prove a capability improvement. Wall-clock
time is reported but is not a promotion signal. An initial strategy can bootstrap
an empty scope after passing the required independent holdouts.

`auto_promote=True` is the default. With `auto_promote=False`, an eligible
evaluation can later be passed to `promote_rsi_strategy` / `rsi.promote`.
Promotion rechecks evidence and uses an atomic SQLite transaction. An old
evaluation cannot replace a subsequently changed default.

The kernel-run budget is checked before execution, including the worst-case
development, baseline and refinement work. Solver/reference/builder subprocesses
have stage timeouts. Geometry preparation retains its own bounded attempts and
backend timeouts; this is not a hard whole-campaign wall-clock deadline.

Qualification links the original preparation request, actual controls/field,
run identity, reference function, sampling budget and refinement criteria to
the frozen benchmark. Copying valid output evidence to another problem does
not establish independent coverage.

Holdout separation is enforced by evaluation scheduling and recorded exposure.
Benchmark authors and implementation providers are trusted local code; the
workspace does not provide an adversarial secrecy boundary for holdout files.

## Case-local code generation

`builder_uri` provides a reusable `build_case_kernel(config)` function instead
of a fixed `kernel_uri`:

```python
def build_case_kernel(config):
    # config: original problem, strategy guidance, controls, ready domain,
    # and native domain artifact paths.
    # Derive or instantiate code for this problem/domain.
    return {"kernel_source": generated_python_source}
```

The builder runs in a separate interpreter after domain readiness. Its output
must contain `run_case(config)` and is executed through CaseRuntime. Generated
code and generation logs stay with the attempt. A guidance-only strategy uses
the benchmark/case implementation; agents read that guidance before authoring
the case-local kernel.

Candidate revisions are authored by the agent or a registered implementation
provider. The evaluator supplies measured diagnostics, failure examples and
suggested next actions. It does not train model weights or synthesize correct
physics from an unknown family automatically.

## Budgeted automatic revision campaigns

`RSIRuntime.improve` / `improve_rsi_strategy` coordinates automatic candidate
revisions. A frozen revision provider implements:

```python
def revise_strategy(config):
    # scope, parent strategy/implementation, development cases/outcomes,
    # independent error diagnostics and remaining campaign budget.
    return {
        "rationale": "Explain the change using the development evidence.",
        "patch": {"mesh_policy": {"target_element_size": 0.15}},
        # Optional: kernel_source exposing run_case(config), OR
        # builder_source exposing build_case_kernel(config).
    }
```

The provider can use an author-supplied algorithm or model client. The controller
contains no PDE-specific repair rule. It freezes the provider and helpers,
invokes it in a separate interpreter, checks request/source integrity and
registers each proposed strategy with an immutable parent revision.

```python
from physicsos.schemas.rsi import RevisionProviderSpec, ImproveStrategiesInput

provider = rsi.register_revision_provider(RevisionProviderSpec(
    name="development-reviser",
    description="Revise the reusable implementation from measured failures.",
    python_uri="strategies/revise.py",
))
campaign = rsi.improve(ImproveStrategiesInput(
    suite=suite.manifest, revision_provider=provider.manifest,
    initial_strategy=candidate.manifest,
    max_revisions=3, max_stagnant_revisions=2, max_kernel_runs=128,
))
```

The current default is the promotion baseline. An empty scope requires an
initial strategy. Development probes are shared across the campaign, so changing
mesh density does not change the comparison points. Actual CaseRuntime solves
and required refinement runs rank revisions; the best measured revision remains
the next proposal's parent while unsuccessful proposals supply failure feedback.

The controller reserves final evaluation work before spending revision budget.
It stops at the revision limit, stagnation limit, insufficient budget, an explicit
provider stop, or a verified development improvement. An optional normalized
`development_error_target` can request further revisions after an initial gain.
Provider calls and kernel-run reservations are recorded separately.

Only development cases/outcomes appear in revision requests. Reference file paths
and holdout definitions/results are excluded. The selected revision then passes
through the existing final baseline/holdout evaluator and promotion checks.
No revision provider is invoked after final holdout exposure. Failed final checks
retain the incumbent; a later campaign needs fresh holdouts. `include_holdout=False`
supports development-only campaigns without promotion.

Strategy patches cannot change scope, parent lineage, physical data, references
or acceptance thresholds. New implementations must expose the required callable;
mesh/repair and numerical-control changes retain the existing readiness gates.
The active registry generation is fixed for the campaign, and a concurrent
default change stops further work. This remains a trusted-local-code protocol,
as described above for holdout separation.

## Capability estimates and production feedback

```python
estimate = rsi.assess(AssessCapabilityInput(
    scope=scope, relative_tolerance=0.01,
))
```

Capability estimates use distinct holdout/production physical problems.
Repeated runs or verifications do not add independent coverage. Development
results supply failure/revision guidance but do not increase the capability
success count. Changed or missing evidence becomes unresolved.

The estimate reports verified/failed/uncertain counts, a 95% Wilson interval
and a Beta(1,1)-smoothed verification-pass probability. High confidence requires
at least twenty independent problems and an interval lower bound of 0.8;
three successful holdouts remain a small sample. Optional accuracy/refinement
criteria filter evidence. Looser checks cannot support a stricter target.
Promoted strategy criteria are the default assessment contract.

Use `solve_with_rsi_strategy` / `rsi.solve` with `SolveWithStrategyInput` for
production. It selects the exact scope's default, applies its policy, prepares
the current geometry and generates/executes the case-local kernel. Supply the
physical controls, tunable-control allowlist, independent reference and required
refinement checks. Every production attempt saves a capability estimate before
execution; the observed outcome supports subsequent Brier-score reporting.

A production failure with intact evidence counts toward the activation's rollback limit
(default: one distinct failed problem). At that limit, the controller restores
the preceding valid default using the expected generation. Inconclusive evidence
does not masquerade as a verified success or a demonstrated regression.
`rollback_rsi_strategy` also supports an explicit audited rollback.

Before restoring a predecessor, rollback rechecks its saved numerical
qualification as well as its implementation files. If that evidence is lost
or corrupted, the scope's default is cleared. A blocked rollback reports the
actual current registry state.

## Learned strategy context for agents

```python
from physicsos.schemas.rsi import BindCaseStrategyInput

context = rsi.bind_context(BindCaseStrategyInput(
    case_id="next-problem",
    assessment=AssessCapabilityInput(scope=scope, relative_tolerance=0.01),
))
```

`bind_rsi_case_context` / `physicsos rsi bind-context REQUEST.json` saves the
scope/accuracy request and a versioned guidance snapshot. The case receives
`context/rsi_strategy.md` and `context/rsi_strategy.json`, containing the current
revision, generation, mesh/repair/control policy, capability evidence and
observed revision actions.

`build_paper_context_window` and `build_taps_derivation_prompt` accept an optional
`rsi_assessment`. They also refresh an existing case binding, so a default change
or rollback is reflected before the next derivation/implementation. Invalid
bindings or a conflicting prepared-domain scope produce `needs_review` and
clear applicable policy guidance. Each previous snapshot remains available.

The API agent's filesystem backend now uses the same physical workspace as
CaseRuntime. `/cases/...` and `/workspace/cases/...` address the same files.
Agent-written kernels can be executed directly, and agents can read the
versioned domain/run/verification artifacts returned by runtime tools.

## CLI and artifacts

The CLI accepts JSON matching each input model and needs no model API key:

```bash
physicsos rsi register-strategy strategy.json
physicsos rsi register-suite suite.json
physicsos rsi register-revision-provider provider.json
physicsos rsi improve campaign-request.json
physicsos rsi evaluate evaluation-request.json
physicsos rsi assess assessment-request.json
physicsos rsi bind-context context-request.json
physicsos rsi solve production-request.json
physicsos rsi promote promotion-request.json
physicsos rsi rollback rollback-request.json
```

Exit 0 means a successful registration/assessment, eligible or completed
promotion, verified solve, or completed rollback. Unresolved/rejected operations
return exit 2.

```text
data/rsi/
  strategies/<revision>/manifest.json
  suites/<revision>/manifest.json
  revision_providers/<revision>/manifest.json
  campaigns/<id>/report.json
  campaigns/<id>/revisions/<round>/  request, provider response/log, generated code
  evaluations/<id>/selection.json
  evaluations/<id>/report.json
  evaluations/<id>/attempts/.../outcome.json
  assessments/<id>/report.json
  executions/<id>/report.json
  transitions/<id>/promotion.json or rollback.json
  state.sqlite3                    active generations, exposure, observations
cases/<attempt>/domains, runs, studies  shared CaseRuntime numerical evidence
cases/<case>/context/rsi/<id>/         versioned learned strategy context
```

The main, geometry, implementation, verification and knowledge agent tool
surfaces expose the relevant RSI operations. `SelfDiagnostic` can delegate to
`RSIRuntime` with an explicit scope; its legacy runtime-history adapter now
reads real records and avoids high confidence from one successful case.
