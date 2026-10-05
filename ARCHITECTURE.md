# PhysicsOS Architecture

Implementation status: Alpha. The shared CaseRuntime joins geometry preparation,
case-local kernel execution and independently evaluated field/refinement evidence.
Outcome history is retained and searchable. Scoped runtime strategies can now
be evaluated, calibrated, promoted and rolled back using frozen benchmarks.
Experimental validation and model-weight learning remain planned.

## Shared runtime

```text
raw CAD / surface / existing mesh / explicit generated geometry
    |
    v
prepare_simulation_domain
    source snapshot -> optional PaMO repair -> generate/reuse mesh
    -> recompute element quality -> bind boundary semantics
    -> solver-facing mesh arrays OR signed-distance/background-grid arrays
    |
    v
PreparedDomain (ready / needs_input / needs_review / unavailable / failed)
    |
    v
case-local kernel.py: run_case(config)
    current domain artifacts + controls -> fresh solution artifacts
    |
    v
verify_case_solution / run_case_convergence
    separate reference implementation + actual solution samples
    actual rebuilds and reruns of frozen code
    |
    v
VERIFIED / FAILED / UNCERTAIN -> revision, report, searchable outcome history
```

A case is a runtime instance, not a separately implemented workflow. Physics
and matrix assembly remain generated in that case. Discretization preparation,
execution, version checks and evidence collection are shared across cases.
TAPS is currently a paper-derived artifact workflow, not a universal executable
IR that replaces these case-local implementations.

The API agent and runtime share physical workspace files. The agent filesystem
supports both `/cases/...` and `/workspace/cases/...`; virtual memory is no longer
the backing store for simulation artifacts. Case contexts can bind an RSI scope
and refresh learned guidance before derivation or implementation.

## TUI workbench

`physicsos/workbench.py` adds a terminal workbench alongside the chat interface:
cases/runs, RSI strategies/campaigns and verification evidence. It reads the
same workspace artifacts and active registry, launches bounded campaigns in
background workers and delegates promotion/rollback checks to RSIRuntime.
The toolbar, F3, `/workbench` and `/rsi` open it in chat; `physicsos workbench`
also runs without a model server. See [workbench guide](docs/workbench.md).

## Geometry readiness

`physicsos/runtime/domain.py` coordinates existing geometry tools and backend
providers against `DomainRequirements`:

- Bare CAD is meshed; existing source meshes are read and checked before reuse.
- Closed triangle surfaces become shells/volumes before volume meshing. Separate
  components and cavities are retained; missing closure requires repair.
- Gmsh evaluates the actual requested element dimension, including tetrahedra,
  rather than accepting a volume based on its boundary triangles.
- Unsupported/missing quality metrics and unresolved boundary roles block
  readiness. Physical groups are preserved or explicitly rebound, never guessed.
- Element order is checked on actual elements. Unsupported mesh strategies,
  boundary layers and local refinement requests remain unresolved. Existing
  material/interface tags survive mesh reuse; surface-only conversion of material
  partitions requires an interface-preserving provider.
- Background grids, triangle-distance SDF, occupancy, samples, normals and cut
  cells are generated from one final surface revision. Exterior grid domains
  require explicit enclosing bounds and have separate outer boundary groups.
- Exterior mesh problems require an explicit fluid computational-domain asset.
  A solid object's interior is not silently used for exterior flow.

`physicsos/backends/geometry_repair.py` delegates to the standalone PaMO CUDA
worker through a separate Python environment or Docker. The repair validates
output topology, self-intersections, triangle quality and sampled surface
fidelity, preserves the original asset and invalidates old entity bindings and
encodings. See [PaMO runner](runners/pamo/README.md).

## Kernel execution contract

The kernel exposes `run_case(config)` and consumes `domain_artifacts`, `controls`
and `output_dir`. It reassembles against the current discretization and writes
`solution.npy`, `residual_history.json` and `runtime_metadata.json`.

`physicsos/runtime/execution.py` freezes inputs/code, invokes the entrypoint in
an isolated working case, checks hashes and field shape, and keeps every run.
Old solution arrays are not inputs. Solver success describes execution only.

The compatibility `execute_taps_kernel` entrypoint delegates to CaseRuntime when
provided a prepared domain. Specialized legacy calls remain unprepared and use
fresh standard output files; they cannot stand in for runtime readiness.

## Independent numerical evidence

`physicsos/runtime/verification.py` compares actual field arrays to a separately
authored `exact_solution(points, config)` reference. Mesh sampling uses supplied
connectivity, so it cannot fill holes with a new Delaunay mesh. Current field
samplers support linear simplexes and structured grids; unsupported or mixed
fields remain uncertain.

`physicsos/runtime/convergence.py` freezes implementations, replays the adopted
source geometry, rebuilds discretizations and reruns the same kernel. It measures
errors at common physical probes and fits order from actual spacing. At an
accuracy floor the order stays unknown. No synthetic O(h^p) errors are emitted.
The norm is weighted probe RMS, not a certified global L2 integral.

Legacy missing conservation metrics remain uncertain. Empty verifier pipelines
cannot return verified, contradictory report states are rejected, and final
solver residuals are not treated as mesh-convergence evidence.

## RSI strategy improvement

Every attempt links geometry, mesh, code, controls and verification artifacts.
`data/runtime_attempts.jsonl` retains verified, failed and uncertain observations.
`search_runtime_history` retrieves outcomes by physics regime/domain, dimension,
representation and case for subsequent agent revisions. Re-verifying one run
does not inflate its historical attempt count.

`physicsos/rsi/` adds a controller around the same CaseRuntime:

- Immutable strategy revisions carry repair/mesh policies, numerical controls,
  implementation/generation providers and agent guidance within a declared scope.
- Budgeted revision campaigns invoke a frozen `revise_strategy(config)` provider
  with development diagnostics, register reusable descendants and rerun them
  through CaseRuntime. Shared probes, reserved final work, stagnation/revision
  limits and registry generation checks bound the campaign. One final holdout
  evaluation follows selection; its outcomes do not feed another revision.
- Frozen suites preserve physical problems, independent references and acceptance
  thresholds. Development selection precedes fresh holdout evaluation; shared
  probe sets support baseline/candidate comparisons.
- Eligibility requires verified coverage, actual requested refinements and
  measured holdout improvement without unacceptable regression. Changes to
  inputs, outputs, criteria or the active baseline invalidate promotion.
- SQLite transactions track active generations and holdout exposure. Copied
  problems cannot inflate coverage, and overlapping exposed holdouts require
  replacement before another campaign.
- Development exposure is retained across campaigns. Original preparation
  requests and linked run/reference/refinement identities prevent numerical
  evidence from being assigned to a different physical problem.
- Capability estimates use independent holdout/production outcomes, uncertainty
  intervals and saved pre-run predictions. Development failures provide revision
  guidance. Production failures trigger an audited return to the preceding
  valid default.
- Rollback rechecks the predecessor's numerical qualification. Learned case
  contexts refresh registry generations and clear obsolete or invalid guidance.

Agents or implementation providers author candidate revisions; the controller
tests and adopts them. This establishes bounded empirical strategy improvement.
It does not establish open-ended self-improvement or model-weight learning.
Physics-family definitions and reference independence still require competent
benchmark authors. See [RSI usage and evidence limits](docs/rsi.md).

## Locations

| Component | Source |
| --- | --- |
| Runtime contracts | `physicsos/schemas/case_runtime.py` |
| Preparation/execution/verification/studies | `physicsos/runtime/` |
| Runtime agent tools | `physicsos/tools/case_runtime_tools.py` |
| RSI contracts/controller | `physicsos/schemas/rsi.py`, `physicsos/rsi/` |
| RSI agent tools/CLI | `physicsos/tools/rsi_tools.py`, `physicsos rsi` |
| Actual element quality | `physicsos/backends/mesh_quality.py` |
| PaMO host adapter and worker | `physicsos/backends/geometry_repair.py`, `pamo_worker.py` |
| Agents/prompts | `physicsos/agents/`, `physicsos/tools/registry.py` |
| Local CLI | `physicsos runtime prepare/execute/verify/convergence REQUEST.json` |

See [CaseRuntime usage and limits](docs/case_runtime.md) for inputs, artifact
layout, version rules and reproducible execution examples.
