# PhysicsOS foamvm/E2B runner

This directory is a trimmed copy of `D:\foamvm` for PhysicsOS external full-solver execution.

## Contract

PhysicsOS should call:

```text
POST /api/physicsos/jobs
Authorization: Bearer $PHYSICSOS_RUNNER_TOKEN
Content-Type: application/json
```

with a `physicsos.full_solver_job.v1` manifest:

```json
{
  "schema_version": "physicsos.full_solver_job.v1",
  "problem_id": "problem:openfoam-smoke",
  "backend": "openfoam",
  "backend_command": "icoFoam",
  "openfoam": {
    "solver": "icoFoam"
  }
}
```

If `openfoam.case_files` is absent, the runner executes the built-in OpenFOAM cavity smoke case:

```text
copy $FOAM_TUTORIALS/incompressible/icoFoam/cavity/cavity -> /workspace/case
blockMesh
icoFoam
foamToVTK
collect /workspace/output artifacts
```

For this smoke path, the runner forces `icoFoam` even if the catalog-level
`backend_command` is `simpleFoam`, because the built-in tutorial case is the
transient cavity case.

## Verified Smoke Run

Local endpoint:

```text
http://localhost:3100/api/physicsos/jobs
```

Result:

```text
status=completed
backend=openfoam
solver=icoFoam
artifacts=log.blockMesh, log.icoFoam, log.foamToVTK
OpenFOAM version=2412
```

## Environment

Do not commit `.env.local`.

Required:

```text
E2B_API_KEY
E2B_TEMPLATE_ID
```

Optional:

```text
PHYSICSOS_RUNNER_TOKEN
```
