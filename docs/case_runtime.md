# CaseRuntime: one path from assets to numerical evidence

`physicsos.runtime.CaseRuntime` joins domain preparation, `run_case(config)`,
independent field comparison and real discretization studies. Physics code
remains case-local; preparation and evidence management are shared.

## Prepare the computational domain

```python
from physicsos.runtime import CaseRuntime
from physicsos.schemas.case_runtime import (
    DomainRequirements, PrepareDomainInput, ExecuteCaseInput,
    VerifyCaseInput, ConvergenceStudyInput,
)
from physicsos.schemas.geometry import GeometrySource, GeometrySpec
from physicsos.schemas.mesh import MeshPolicy

runtime = CaseRuntime()
prepared = runtime.prepare(PrepareDomainInput(
    case_id="asset-demo",
    geometry=GeometrySpec(
        id="asset", source=GeometrySource(kind="cad_step", uri="assets/body.step"),
        dimension=3,
    ),
    requirements=DomainRequirements(
        representation="mesh", dimension=3,
        mesh_policy=MeshPolicy(target_element_size=0.2),
        whole_boundary_role="wall", required_boundary_roles=["wall"],
    ),
))
if prepared.domain.status != "ready":
    raise RuntimeError(prepared.domain.required_actions)
```

The whole-boundary role above is an explicit problem assumption, not an
inference that every imported surface is a wall. For several boundary types,
preserve named physical groups and use `boundary_roles` to bind those names.
Missing required roles or empty boundary node/sample groups block readiness.

Existing source meshes can be reused only after reading and checking their
actual elements and element order. `force_remesh=True` rebuilds them when the
provider can preserve their semantics. Bare CAD is meshed; closed
STL/mesh surfaces are turned into shells and volumes, including separate
components and nested cavities. Surface meshes are not accepted as volume
meshes just because they contain elements.

The native mesher supports `auto`/`unstructured` strategies, global element size
and element order. Boundary layers, local refinement regions and other strategy
requests remain `needs_review` until a provider implements them. Background-grid
resolution is controlled separately by `grid_resolution`.

Reused material meshes retain their physical region/interface tags in the
native `.msh`; declared region kinds and merged entity bindings are preserved.
Surface-only repair/remeshing and single-SDF embedding cannot preserve internal
material partitions, so such conversions remain `needs_review`. Missing declared
region bindings also block readiness. Internal surface elements are detected from
volume connectivity even with arbitrary physical names. If they exist, bind
exterior groups through `boundary_roles`; `whole_boundary_role` cannot overwrite
their interface semantics.

Quality is recomputed with Gmsh in the requested element dimension, including
Jacobian determinants, signed shape quality and edge aspect ratios. Required
unavailable metrics remain unresolved. A flag in an old `MeshSpec` is not
quality evidence. Known geometric dimensions are required for generated
primitives; the runtime does not substitute a unit box for missing geometry.

Meshing/quality attempts are bounded. The PaMO repair adapter is available for
surface defects, with its Python/Docker settings in `repair_options`. Missing
CUDA is reported as `backend_unavailable`; no repair is claimed.

For immersed-boundary TAPS, select `representation="background_grid"` and
three `grid_resolution` values. The runtime computes triangle distances and
signed occupancy from the final surface, plus boundary samples/normals and
cut-cell candidates. These files share a domain revision. Self-intersection
coverage is reported separately in `sdf_quality.json`.

An exterior background domain requires explicit enclosing bounds. Its mask
is the complement of the body, wall normals are reversed, and outer grid
faces have separate samples and labels such as `outer:x_min`. Bind those
labels explicitly to inlet/outlet roles. For an exterior **mesh** domain,
provide a preconstructed fluid-domain CAD/mesh asset; the runtime never treats
the solid object's interior as the fluid exterior.

## Case-local kernel contract

Write `cases/<case_id>/taps/kernel.py`:

```python
def run_case(config: dict) -> dict:
    paths = config["domain_artifacts"]
    controls = config["controls"]
    output_dir = config["output_dir"]
    # Read this run's mesh/geometry, reassemble, solve, then write:
    # solution.npy, residual_history.json, runtime_metadata.json.
    return {"status": "success"}
```

`domain_artifacts` contains native paths to the copied inputs:

| Representation | Main data |
| --- | --- |
| Mesh | `mesh_arrays.npz`: `points` and cell blocks such as `tetra`; `boundary_nodes.npz`: named/role node masks; source `.msh` |
| Background grid | `background_grid.json`: ordered x/y/z axes; `sdf.npy`, `occupancy.npy`, `boundary_samples.npy`, `normals.npy`, `boundary_groups.npz`, `cut_cells.npy` |

The solution is nodal for mesh domains, with leading dimension equal to the
point count. Grid solutions use leading dimensions equal to the grid shape.
Scalar and vector trailing dimensions must match the independent reference.
Entrypoints return JSON metadata, not large arrays.

```python
executed = runtime.execute(ExecuteCaseInput(
    case_id="asset-demo", prepared_domain=prepared.manifest,
    controls={"material_coefficient": 2.0},
))
```

Every run has a new working case and output directory. It invokes the callable
entrypoint, passes controls, hashes the kernel and inputs, and rejects input
mutation or a stale domain checksum. Old solution files are not copied.
Execution success is distinct from numerical verification.

## Independent verification and actual reruns

A separately authored reference file provides:

```python
def exact_solution(points, config):
    # Derive from the actual problem and its parameters.
    # Return an array with one field value per physical comparison point.
    ...
```

```python
verified = runtime.verify(VerifyCaseInput(
    run_manifest=executed.manifest,
    reference_uri="cases/asset-demo/verification/reference.py",
    relative_tolerance=0.01,
))
study = runtime.convergence(ConvergenceStudyInput(
    case_id="asset-demo", prepared_domain=prepared.manifest,
    reference_uri="cases/asset-demo/verification/reference.py",
    refinements=[0.4, 0.2, 0.1], axis="mesh", expected_order=2,
))
```

Studies freeze implementation files, replay the adopted source geometry,
rebuild each discretization, rerun the same kernel, and compare fields at
common physical points. Observed order uses actual mesh/grid spacing and
measured errors. At an accuracy floor it remains `null`; no order is invented.
Without a reference, reruns are retained but verification is `uncertain`.

Mesh sampling currently supports linear line/triangle/tetrahedron fields.
Other or mixed top-dimensional elements remain uncertain until a sampling
provider exists. It uses supplied connectivity, so holes are not filled by an
unconstrained Delaunay triangulation. Grid comparison points avoid dyadic node
coincidence and are checked against the physical body/interior or exterior.
Reported norms are weighted **probe RMS**, not certified global L2 integrals.
The result verifies the configured reference comparison, not every physical
law or agreement with experiments.

## Artifacts and feedback

```text
cases/<case_id>/
  geometry/prepared_domain.json       active preparation pointer
  geometry/taps_geometry_handoff.md  versioned input links
  domains/<revision>/manifest.json   source, requirements, checks, recipe
  domains/<revision>/request.json    original preparation request
  runs/<run_id>/manifest.json        code/input hashes, controls, outputs
  runs/<run_id>/verification/<id>/   independent evidence
  studies/<study_id>/report.json     actual rerun/error history
  runtime_events.jsonl
data/runtime_attempts.jsonl          append-only outcome observations
```

All outcome observations are retained. `search_runtime_history` filters by
physics regime/domain, dimension, representation or case and returns run and
verification references for subsequent revisions. Re-verifying a run does not
inflate its attempt count. [RSIRuntime](rsi.md) now consumes this same execution
path for frozen candidate/baseline evaluations, holdout checks, capability
estimates and guarded strategy promotion/rollback.

The main/geometry/implementation/verification agents expose the shared runtime
tools. CLI operations accept JSON matching each input model, without a model
API key:

```bash
physicsos runtime prepare prepare-request.json
physicsos runtime execute execute-request.json
physicsos runtime verify verify-request.json
physicsos runtime convergence study-request.json
```

Exit 0 means the requested stage passed; exit 2 means failure or unresolved
evidence. The old generated-convergence script has synthetic errors disabled;
it needs a prepared domain and explicit refinements to delegate actual work.
