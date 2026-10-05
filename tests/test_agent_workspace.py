from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import runpy

from physicsos.agents.runtime import _build_filesystem_backend
from physicsos.runtime import CaseRuntime
from physicsos.runtime.artifacts import write_json
from physicsos.schemas.case_runtime import DomainRequirements, ExecuteCaseInput, PrepareDomainInput
from physicsos.schemas.geometry import GeometryEntity, GeometrySource, GeometrySpec
from physicsos.schemas.mesh import MeshPolicy


def test_agent_files_and_case_runtime_share_one_physical_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(tmp_path))
    backend = _build_filesystem_backend()
    fixture = runpy.run_path(str(Path(__file__).with_name("test_rsi.py")))
    written = backend.write("/cases/shared/taps/kernel.py", fixture["KERNEL"])
    assert written.error is None
    assert (tmp_path / "cases" / "shared" / "taps" / "kernel.py").is_file()
    geometry = GeometrySpec(id="line", source=GeometrySource(kind="generated"), dimension=1,
                            entities=[GeometryEntity(id="line", kind="curve", metadata={"length": 1.})])
    runtime = CaseRuntime(tmp_path)
    prepared = runtime.prepare(PrepareDomainInput(case_id="shared", geometry=geometry,
                                                requirements=DomainRequirements(dimension=1, whole_boundary_role="wall", mesh_policy=MeshPolicy(target_element_size=.2))))
    assert prepared.domain.status == "ready", prepared.domain.required_actions
    run = runtime.execute(ExecuteCaseInput(case_id="shared", prepared_domain=prepared.manifest))
    assert run.run.result.status == "success", run.run.errors
    seen = backend.read(run.manifest.uri)
    assert seen.error is None
    content = seen.file_data["content"]
    text = content if isinstance(content, str) else "\n".join(content)
    assert json.loads(text)["id"] == run.run.id
    edited = backend.edit("/workspace/cases/shared/taps/kernel.py", '"method":"P1 FEM"', '"method":"P1 FEM, agent edit"')
    assert edited.error is None
    assert "agent edit" in (tmp_path / "cases" / "shared" / "taps" / "kernel.py").read_text()


def test_concurrent_artifact_writes_leave_complete_json_and_no_temp_files(tmp_path):
    path = tmp_path / "active.json"
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda index: write_json(path, {"index": index, "items": list(range(100))}), range(32)))
    result = json.loads(path.read_text())
    assert result["index"] in range(32)
    assert result["items"] == list(range(100))
    assert not list(tmp_path.glob("*.tmp"))
