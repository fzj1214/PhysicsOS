"""Host adapter for PaMO; importing it does not import CUDA dependencies."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from time import perf_counter
from uuid import uuid4

import numpy as np

from physicsos.config import project_root
from physicsos.paths import resolve_workspace_path, to_agent_path
from physicsos.schemas.common import ArtifactRef
from physicsos.schemas.geometry import GeometryEntity, GeometryQualityReport, GeometrySource, GeometryTransform
from physicsos.schemas.geometry_repair import GeometryRepairReport, RepairGeometryInput, RepairGeometryOutput
from physicsos.backends.surface_mesh import read_mesh_surface, surface_metrics


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _safe(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip(".-")[:96] or "geometry"


def _artifact(path: Path, kind: str, workspace: Path) -> ArtifactRef:
    return ArtifactRef(uri=to_agent_path(path, workspace=workspace), kind=kind, format=path.suffix.lstrip("."))


def _output_path(directory: Path, value: str | None) -> Path:
    if not value:
        raise ValueError("Worker did not return the required output artifact.")
    path = (directory / value).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError("Worker output is missing or outside the repair directory.")
    return path


def _command(input: RepairGeometryInput, directory: Path, executor: str) -> list[str]:
    if executor == "docker":
        image = input.docker_image or os.environ.get("PHYSICSOS_PAMO_DOCKER_IMAGE") or "physicsos-pamo:latest"
        return [
            "docker", "run", "--rm", "--name", f"physicsos-pamo-{directory.name}",
            "--gpus", "all", "--network", "none", "--volume", f"{directory}:/job",
            image, "--request", "/job/request.json", "--response", "/job/response.json",
        ]
    python = input.python_executable or os.environ.get("PHYSICSOS_PAMO_PYTHON") or sys.executable
    return [python, str(Path(__file__).with_name("pamo_worker.py")), "--request", str(directory / "request.json"), "--response", str(directory / "response.json")]


def _validate_repaired_output(report: GeometryRepairReport, directory: Path, input: RepairGeometryInput) -> None:
    surface = _output_path(directory, report.output_surface)
    indexed_mesh = _output_path(directory, report.output_mesh)
    if _sha256(surface) != report.output_sha256:
        raise ValueError("Repaired surface checksum does not match the report.")
    with np.load(indexed_mesh, allow_pickle=False) as mesh:
        vertices, faces = mesh["vertices"], mesh["faces"]
        metrics = surface_metrics(vertices, faces)
    surface_vertices, surface_faces = read_mesh_surface(surface)
    if not np.array_equal(surface_vertices[surface_faces], vertices[faces]):
        raise ValueError("Repaired STL does not match the checked indexed surface.")
    quality = report.quality
    if not quality.passes or quality.watertight is not True or quality.manifold is not True or quality.self_intersections != 0:
        raise ValueError("Worker marked an unchecked or defective surface as repaired.")
    if not all(metrics[key] for key in ("watertight", "manifold", "winding_consistent")):
        raise ValueError("Host surface topology check failed.")
    if metrics["aspect_ratio_p95"] > input.max_aspect_ratio_p95 or metrics["max_skewness"] > input.max_skewness:
        raise ValueError("Host surface quality check failed.")
    deviation = report.deviation.get("relative_sampled_max_distance")
    if isinstance(deviation, bool) or not isinstance(deviation, (float, int)) or not np.isfinite(deviation) or deviation < 0 or deviation > input.max_relative_surface_distance:
        raise ValueError("Surface deviation is missing, nonfinite, or above the requested limit.")
    report.after = metrics


def run_pamo_repair(input: RepairGeometryInput, *, workspace: Path | None = None) -> RepairGeometryOutput:
    geometry = input.geometry
    workspace = Path(workspace or project_root()).resolve()
    job_id = uuid4().hex
    parent = (workspace / "cases" / _safe(input.case_id) / "geometry" / "repairs") if input.case_id else (workspace / "scratch" / _safe(geometry.id) / "geometry_repair")
    directory = parent / job_id
    directory.mkdir(parents=True, exist_ok=False)
    report_path = directory / "repair_report.json"
    log_path = directory / "execution_log.json"
    report = GeometryRepairReport(job_id=job_id, status="failed", input_sha256="", quality=GeometryQualityReport(passes=False))
    log: dict[str, object] = {
        "schema_version": "physicsos.geometry_repair_execution.v1", "job_id": job_id,
        "source_uri": geometry.source.uri, "units": geometry.coordinate_system.units,
    }
    started = perf_counter()
    command: list[str] = []
    executor = input.executor if input.executor != "auto" else os.environ.get("PHYSICSOS_PAMO_EXECUTOR", "python")
    try:
        if executor not in {"python", "docker"}:
            raise ValueError("PaMO executor must be python or docker.")
        if not geometry.source.uri:
            raise ValueError("Geometry repair requires an actual geometry source file.")
        source = resolve_workspace_path(geometry.source.uri, workspace=workspace)
        if not source.is_file():
            raise FileNotFoundError(f"Geometry source not found: {geometry.source.uri}")
        snapshot = directory / ("source" + source.suffix.lower())
        shutil.copyfile(source, snapshot)
        report.input_sha256 = _sha256(snapshot)
        if geometry.source.checksum and geometry.source.checksum.removeprefix("sha256:") != report.input_sha256:
            raise ValueError("Geometry source changed since its recorded checksum; prepare a new geometry revision.")
        request = {
            "schema_version": "physicsos.pamo_request.v1", "job_id": job_id,
            "input_path": snapshot.name, "input_sha256": report.input_sha256,
            "source_kind": geometry.source.kind, "units": geometry.coordinate_system.units,
            "options": input.model_dump(exclude={"geometry", "executor", "python_executable", "docker_image", "case_id", "timeout_seconds"}),
        }
        (directory / "request.json").write_text(json.dumps(request, indent=2, allow_nan=False), encoding="utf-8")
        command = _command(input, directory, executor)
        log["command"] = command
        completed = subprocess.run(command, cwd=directory, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=input.timeout_seconds, check=False)
        log.update(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr, timed_out=False)
        response = directory / "response.json"
        if not response.is_file():
            if executor == "docker" and completed.returncode == 125:
                report.status = "backend_unavailable"
            raise RuntimeError(f"PaMO worker exited {completed.returncode} without a response: {completed.stderr[-2000:]}")
        parsed = GeometryRepairReport.model_validate_json(response.read_text(encoding="utf-8"))
        if parsed.job_id != job_id or parsed.input_sha256 != report.input_sha256:
            raise ValueError("Worker response belongs to another job or input snapshot.")
        if completed.returncode != 0 and parsed.status in {"repaired", "needs_review"}:
            raise RuntimeError("Worker reported success despite a nonzero exit status.")
        if parsed.status == "repaired":
            _validate_repaired_output(parsed, directory, input)
        elif parsed.output_surface:
            _output_path(directory, parsed.output_surface)
            if parsed.output_mesh:
                _output_path(directory, parsed.output_mesh)
        report = parsed
    except subprocess.TimeoutExpired as exc:
        report.error = f"PaMO repair exceeded {input.timeout_seconds}s."
        log.update(returncode=124, timed_out=True, stdout=_text(exc.stdout), stderr=_text(exc.stderr))
        if executor == "docker":
            try:
                cleanup = subprocess.run(["docker", "stop", "--time", "1", f"physicsos-pamo-{job_id}"], capture_output=True, text=True, timeout=15, check=False)
                log["container_cleanup_returncode"] = cleanup.returncode
            except (OSError, subprocess.TimeoutExpired) as cleanup_error:
                log["container_cleanup_error"] = str(cleanup_error)
    except FileNotFoundError as exc:
        report.status = "backend_unavailable" if command and exc.filename == command[0] else "failed"
        report.error = str(exc)
    except Exception as exc:
        report.error = f"{type(exc).__name__}: {exc}"
    if report.error:
        report.quality.passes = False
        report.quality.issues = sorted(set([*report.quality.issues, report.error]))
    if report.status != "repaired":
        report.quality.passes = False
    log["wall_time_seconds"] = perf_counter() - started

    result_geometry = geometry.model_copy(deep=True)
    changes = []
    invalidated = []
    candidate = None
    if report.output_surface:
        candidate = _artifact(_output_path(directory, report.output_surface), "repaired_surface", workspace)
        candidate.checksum = report.output_sha256
    if report.status == "repaired":
        invalidated = [ArtifactRef(uri=encoding.uri, kind=f"geometry_encoding:{encoding.kind}") for encoding in geometry.encodings]
        result_geometry.source = GeometrySource(kind="stl", uri=candidate.uri, checksum=report.output_sha256)
        result_geometry.entities = [GeometryEntity(
            id=f"surface:{job_id}", kind="surface", label="PaMO repaired surface", artifact=candidate,
            metadata={"repair_job_id": job_id, "original_source": geometry.source.uri or "", "original_sha256": report.input_sha256},
        )]
        result_geometry.encodings = []
        # Historical placeholder attempts remain in the audit trail without
        # making a later real repair fail the old scaffold-marker gate.
        for transform in result_geometry.transforms:
            if transform.kind == "repair" and any(marker in transform.description.lower() for marker in ("no-op", "scaffold")):
                transform.kind = "custom"
                transform.description = "Historical unsuccessful repair: " + transform.description
        for region in result_geometry.regions:
            region.entity_ids = []
        for boundary in result_geometry.boundaries:
            boundary.entity_ids = []
            boundary.confidence = 0.0
        result_geometry.transforms.append(GeometryTransform(kind="repair", description=f"PaMO remeshing, simplification, and safe projection; report={to_agent_path(report_path, workspace=workspace)}"))
        changes = ["Adopted the validated PaMO surface.", "Invalidated geometry encodings and old entity bindings; rebuild mesh/SDF and rebind boundary labels."]
    result_geometry.quality = report.quality.model_copy(deep=True)
    if report.status == "repaired":
        result_geometry.quality.unresolved_regions = ["boundary_and_region_rebinding"]
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    report_artifact = _artifact(report_path, "geometry_repair_report", workspace)
    log_artifact = _artifact(log_path, "geometry_repair_execution_log", workspace)
    artifacts = [report_artifact, log_artifact]
    snapshot_paths = list(directory.glob("source.*"))
    if snapshot_paths:
        artifacts.append(_artifact(snapshot_paths[0], "geometry_source_snapshot", workspace))
    if candidate:
        artifacts.append(candidate)
        if report.output_mesh:
            artifacts.append(_artifact(_output_path(directory, report.output_mesh), "repaired_indexed_surface", workspace))
    return RepairGeometryOutput(
        geometry=result_geometry, status=report.status, repair_report=report_artifact,
        execution_log=log_artifact, repaired_surface=candidate, artifacts=artifacts,
        invalidated_encodings=invalidated, requires_boundary_relabel=report.status == "repaired",
        changes=changes, warnings=[*report.warnings, *([report.error] if report.error else [])],
    )


def _text(value: str | bytes | None) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""
