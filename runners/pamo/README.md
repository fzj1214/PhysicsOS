# PaMO geometry repair worker

PhysicsOS runs [PaMO](https://github.com/SarahWeiii/pamo) in a separate CUDA
environment. The host application requires neither torch nor PaMO. The worker
uses all three stages: remeshing, simplification, and safe projection, followed
by independent topology, self-intersection, surface-quality, and sampled shape
deviation checks. See the [paper](https://arxiv.org/html/2509.05595v1).

## Docker

Build from the PhysicsOS repository root:

```bash
docker build --platform linux/amd64 -f runners/pamo/Dockerfile -t physicsos-pamo:latest .
```

The image extends upstream `sarahwei0210/pamo:0.0.2`, which already contains the
compiled CUDA stages. Run on an NVIDIA host with NVIDIA Container Toolkit:

```bash
physicsos geometry repair path/to/asset.stl \
  --executor docker --docker-image physicsos-pamo:latest \
  --case-id repaired-asset --output repaired-geometry.json
```

The same command accepts STEP/IGES, OBJ/STL/PLY and other triangle assets
supported by trimesh, or meshio volume meshes (`.msh`, `.vtu`, `.vtk`, etc.).
Volume meshes are reduced to their external surface for PaMO. CAD surfaces are
tessellated using Gmsh. Inputs with external file dependencies should first be
converted to a self-contained asset. Point clouds require a separate surface
reconstruction step. High-order mesh inputs are linearized to corner nodes.

Alternatively use an existing PaMO environment:

```bash
physicsos geometry repair asset.obj --executor python \
  --python /path/to/pamo/env/bin/python --output repaired-geometry.json
```

Environment defaults: `PHYSICSOS_PAMO_EXECUTOR=python|docker`,
`PHYSICSOS_PAMO_PYTHON`, and `PHYSICSOS_PAMO_DOCKER_IMAGE`.
Install the worker dependencies in `requirements.txt` in that CUDA environment.
The standalone worker and its sibling `surface_mesh.py` ship in the PhysicsOS
package and are compatible with Python 3.10.

## Policy and outputs

`conservative` (default), `balanced`, and `aggressive` request face ratios of
1.0, 0.5, and 0.1 relative to the input. Set `--ratio` or `--target-faces` to
override. These are PaMO stopping targets, not guarantees of the output count.
Remeshing can increase the number of faces; the worker explicitly passes
`min_verts=0` and records the actual DualMC resolution selected upstream.

The default acceptance limits are a sampled bidirectional distance of at most
1% of the input bounding-box diagonal, aspect-ratio p95 at most 25, and maximum
skewness at most 0.95. Use `--max-relative-distance` to set a task-specific
distance tolerance; the Python tool also exposes quality limits and sample
count. Distance sampling is deterministic and **not** an exact Hausdorff bound.
Deviation is measured against the extracted/tessellated input surface; CAD
tessellation error relative to the original BREP needs a separate check.
Small holes/gaps and thin surfaces need explicit physical-feature review:
PaMO remeshing may alter topology below its voxel resolution.

Every job preserves an input snapshot and writes `request.json`,
`execution_log.json`, and `repair_report.json` in a unique directory under
`cases/<case_id>/geometry/repairs/` or `scratch/<geometry_id>/geometry_repair/`.
Candidates also contain full-precision ASCII `repaired.stl` and indexed
`repaired.npz`. Worker status is one of:

- `repaired`: the surface passed checks and becomes the returned geometry source.
- `needs_review`: a candidate exists but failed or lacked a required check;
  the original geometry remains active.
- `backend_unavailable`: the configured Python/CUDA/Docker environment is unavailable.
- `failed`: invalid input, execution failure, timeout, or invalid output evidence.

The host checks job/input/output identity and recomputes topology/triangle
quality from the indexed output, checks its agreement with the actual STL,
and enforces a supplied source checksum before adopting a repaired surface. Missing
self-intersection evidence never counts as zero. PyMeshLab reports the count
of **self-intersecting faces**, stored in `quality.self_intersections`.

On successful repair, old numerical encodings and geometric entity bindings
are invalidated. Region/boundary names and roles remain as semantic intent,
but boundary confidence becomes zero until labels are rebound. The report
does not certify a solver-ready volume mesh or physical simulation result.
Build/check the required volume mesh or recompute TAPS SDF/occupancy/boundary
data from the **final repaired surface**, not PaMO's intermediate UDF.

The geometry agent can now call import, repair, meshing, quality and semantic
tools. For the embedding route, pass the returned source explicitly:

```python
from physicsos.tools.geometry_tools import RepairGeometryInput, repair_geometry
from physicsos.tools.geometry_embedding_tools import (
    PrepareGeometryAnalysisFilesInput, prepare_geometry_analysis_files,
)

repair = repair_geometry(RepairGeometryInput(
    geometry=geometry, executor="docker", case_id=case_id,
))
if repair.status != "repaired":
    raise RuntimeError(repair.geometry.quality.issues)
embedding = prepare_geometry_analysis_files(PrepareGeometryAnalysisFilesInput(
    case_id=case_id, source_uri=repair.geometry.source.uri,
    units=repair.geometry.coordinate_system.units,
))
```

CUDA execution is intentionally absent from the CPU regression tests. GPU
acceptance should run the Docker command on defective assets and inspect
actual before/after topology, surface deviation, and critical physical features.
