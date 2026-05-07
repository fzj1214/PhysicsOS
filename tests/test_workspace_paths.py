from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from physicsos.cli import _patch_deepagents_workspace_paths, _prepare_deepagents_env


def test_deepagents_workspace_alias_matches_shell_workspace_when_cwd_differs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("deepagents_cli")

    from deepagents.backends import CompositeBackend, LocalShellBackend
    import deepagents_cli.agent as cli_agent

    launch_cwd = tmp_path / "launch-cwd"
    physicsos_workspace = tmp_path / "physicsos-workspace"
    launch_cwd.mkdir()
    physicsos_workspace.mkdir()
    monkeypatch.chdir(launch_cwd)
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(physicsos_workspace))

    _prepare_deepagents_env()

    backend = LocalShellBackend(root_dir=Path.cwd(), inherit_env=True)
    composite = CompositeBackend(default=backend, routes={})

    class DummyAgent:
        pass

    monkeypatch.setattr(cli_agent, "create_cli_agent", lambda *args, **kwargs: (DummyAgent(), composite))
    _patch_deepagents_workspace_paths()

    _, patched_backend = cli_agent.create_cli_agent()

    write_result = patched_backend.write("/workspace/fs_marker.txt", "filesystem")
    assert write_result.error is None

    shell_result = patched_backend.execute(
        f"{sys.executable} -c \"import os, pathlib; "
        "print(pathlib.Path.cwd()); "
        "print(os.environ['PHYSICSOS_WORKSPACE']); "
        "pathlib.Path('shell_marker.txt').write_text('shell', encoding='utf-8')\""
    )
    assert shell_result.exit_code == 0, shell_result.output

    output_lines = [line.strip() for line in shell_result.output.splitlines() if line.strip()]
    shell_cwd = Path(output_lines[0])
    shell_workspace = Path(output_lines[1])

    assert shell_workspace == physicsos_workspace
    assert not (launch_cwd / "fs_marker.txt").exists()
    assert not (launch_cwd / "shell_marker.txt").exists()
    assert shell_cwd == shell_workspace
    assert (physicsos_workspace / "fs_marker.txt").read_text(encoding="utf-8") == "filesystem"
    assert (physicsos_workspace / "shell_marker.txt").read_text(encoding="utf-8") == "shell"
