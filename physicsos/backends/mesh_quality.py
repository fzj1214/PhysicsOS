"""Read the actual mesh and check elements in its requested dimension."""
from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np

from physicsos.paths import resolve_workspace_path
from physicsos.config import project_root
from physicsos.schemas.mesh import MeshQualityReport, MeshSpec


def assess_mesh_backend(mesh: MeshSpec) -> MeshQualityReport:
    from physicsos.backends.geometry_mesh import _gmsh_requires_subprocess, _run_backend_subprocess
    if _gmsh_requires_subprocess():
        result = _run_backend_subprocess({"action": "mesh_quality", "mesh": mesh.model_dump(mode="json")})
        if result.get("ok"):
            return MeshQualityReport.model_validate(result["quality"])
        return MeshQualityReport(passes=False, issues=[str(result.get("error", "Mesh quality worker failed."))])
    return assess_mesh_main(mesh)


def assess_mesh_main(mesh: MeshSpec) -> MeshQualityReport:
    try:
        import gmsh
        import meshio
        artifact = next((item for item in mesh.files if item.format in {"msh", "vtu", "vtk", "mesh", "stl"}), None)
        if artifact is None:
            raise ValueError("No readable mesh artifact is attached; stored quality flags are not evidence.")
        path = resolve_workspace_path(artifact.uri, workspace=project_root())
        raw = meshio.read(path)
        types = sorted({block.type for block in raw.cells if block.dim == mesh.dimension})
        if not types:
            raise ValueError(f"Mesh has no dimension-{mesh.dimension} elements.")
        if mesh.dimension == 1 and types == ["line"]:
            cells = np.concatenate([block.data for block in raw.cells if block.type == "line"])
            lengths = np.linalg.norm(raw.points[cells[:, 1]] - raw.points[cells[:, 0]], axis=1)
            if not np.isfinite(lengths).all():
                raise ValueError("Line elements have nonfinite coordinates.")
            minimum = float(lengths.min() / 2)
            issues = ["Degenerate line element detected."] if minimum <= 0 else []
            # A linear line has one edge, zero shape skew and Jacobian norm h/2.
            # Gmsh's signed condition-number measure is undefined for lines.
            return MeshQualityReport(min_jacobian=minimum, max_skewness=0., aspect_ratio_p95=1.,
                                     passes=not issues, issues=issues, checked_dimension=1,
                                     checked_cell_types=types, checked_element_orders=[1],
                                     checked_elements=len(cells), method="linear_line_lengths_recomputed")
        with tempfile.TemporaryDirectory(prefix="physicsos-mesh-quality-") as temporary:
            if path.suffix.lower() != ".msh":
                path = Path(temporary) / "input.msh"
                meshio.write(path, raw, file_format="gmsh22", binary=False)
            gmsh.initialize()
            try:
                gmsh.option.setNumber("General.Terminal", 0)
                gmsh.open(str(path))
                element_types, blocks, _ = gmsh.model.mesh.getElements(mesh.dimension)
                orders = sorted({int(gmsh.model.mesh.getElementProperties(int(kind))[2]) for kind in element_types})
                tags = np.concatenate(blocks) if blocks else np.empty(0, dtype=np.uint64)
                if not len(tags):
                    raise ValueError("Gmsh found no elements in the requested dimension.")
                measures = {name: np.asarray(gmsh.model.mesh.getElementQualities(tags, name)) for name in ("minDetJac", "minSICN", "minEdge", "maxEdge")}
            finally:
                gmsh.finalize()
        if any(len(values) != len(tags) or not np.isfinite(values).all() for values in measures.values()):
            raise ValueError("Mesh quality measures are missing or nonfinite.")
        minimum = float(measures["minDetJac"].min())
        shape = float(measures["minSICN"].min())
        min_edges = measures["minEdge"]
        ratios = measures["maxEdge"] / np.maximum(min_edges, np.finfo(float).tiny)
        aspect = float(np.percentile(ratios, 95))
        skewness = float(1 - np.clip(shape, 0, 1))
        issues = []
        if minimum <= 0 or (min_edges <= 0).any():
            issues.append("Inverted or degenerate element detected.")
        if shape <= 0:
            issues.append("Nonpositive signed element shape quality.")
        if aspect > 25:
            issues.append(f"Element aspect ratio p95={aspect:.6g} exceeds 25.")
        if skewness > .95:
            issues.append(f"Element skewness proxy={skewness:.6g} exceeds 0.95.")
        return MeshQualityReport(
            min_jacobian=minimum, aspect_ratio_p95=aspect, max_skewness=skewness,
            passes=not issues, issues=issues, checked_dimension=mesh.dimension,
            checked_cell_types=types, checked_element_orders=orders,
            checked_elements=len(tags), method="gmsh_element_quality_recomputed",
        )
    except Exception as exc:
        return MeshQualityReport(passes=False, issues=[f"Mesh quality could not be established: {exc}"], checked_dimension=mesh.dimension)
