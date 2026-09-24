"""Exercise the distribution outside the checkout, as a pip user would."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[1]
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]


def run(*args: str, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture(scope="session")
def installed_package(tmp_path_factory: pytest.TempPathFactory) -> Path:
    work = tmp_path_factory.mktemp("installed physicsos")
    # `build` builds the wheel FROM the sdist, checking both distribution formats.
    run(sys.executable, "-m", "build", "--outdir", str(work / "dist"), str(ROOT), cwd=work)
    wheel = next((work / "dist").glob("*.whl"))
    target = work / "site-packages"
    run(
        sys.executable, "-m", "pip", "install", "--no-deps", "--no-compile",
        "--target", str(target), str(wheel), cwd=work,
    )
    return target


@pytest.fixture
def installed_env(installed_package: Path, tmp_path: Path) -> dict[str, str]:
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("PHYSICSOS_", "DEEPAGENTS_", "OPENAI_"))
    }
    env["PHYSICSOS_HOME"] = str(tmp_path / "config")
    env["PYTHONPATH"] = str(installed_package)
    env["PYTHONUTF8"] = "1"
    return env


def console_script(installed_package: Path) -> str:
    relative = "Scripts/physicsos.exe" if os.name == "nt" else "bin/physicsos"
    return str(installed_package / relative)


def test_installed_console_version(
    installed_package: Path, installed_env: dict[str, str], tmp_path: Path,
) -> None:
    for flag in ("--version", "-v"):
        result = run(console_script(installed_package), flag, cwd=tmp_path, env=installed_env)
        assert result.stdout.strip() == f"physicsos {VERSION}"
    assert not (tmp_path / "config").exists()
    assert not (tmp_path / "scratch").exists()
    result = run(sys.executable, "-m", "physicsos", "--version", cwd=tmp_path, env=installed_env)
    assert result.stdout.strip() == f"physicsos {VERSION}"


def test_installed_help_uses_compatible_deepagents(
    installed_package: Path, installed_env: dict[str, str], tmp_path: Path,
) -> None:
    # This used to exit 2 with deepagents-cli 0.3.0's different command interface.
    result = run(console_script(installed_package), "--help", cwd=tmp_path, env=installed_env)
    assert "--model" in result.stdout
    assert "--resume" in result.stdout
    assert "physicsos config" in result.stdout
    assert "/settings" in result.stdout
    assert "Traceback" not in result.stderr
    # Exercise the embedded CLI too; the root help now belongs to PhysicsOS.
    run(console_script(installed_package), "--model=openai:gpt-5.4", "--help", cwd=tmp_path, env=installed_env)


def test_unconfigured_noninteractive_launch_explains_how_to_configure(
    installed_package: Path, installed_env: dict[str, str], tmp_path: Path,
) -> None:
    result = subprocess.run(
        [console_script(installed_package), "--non-interactive", "hello"],
        cwd=tmp_path, env=installed_env, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 2
    assert "physicsos config" in result.stderr
    assert "Traceback" not in result.stderr


def test_installed_config_status_does_not_expose_api_key(
    installed_package: Path, installed_env: dict[str, str], tmp_path: Path,
) -> None:
    key = "config-status-test-secret"
    result = run(
        console_script(installed_package), "config", "--show", cwd=tmp_path,
        env={**installed_env, "OPENAI_API_KEY": key},
    )
    assert json.loads(result.stdout)["api_key_configured"] is True
    assert key not in result.stdout + result.stderr


def test_installed_paths_use_callers_workspace(
    installed_package: Path, installed_env: dict[str, str], tmp_path: Path,
) -> None:
    result = run(console_script(installed_package), "paths", cwd=tmp_path, env=installed_env)
    paths = json.loads(result.stdout)
    assert Path(paths["workspace"]).resolve() == tmp_path.resolve()
    assert Path(paths["home"]) == tmp_path / "config"
    workspace = tmp_path / "自定义 workspace"
    workspace.mkdir()
    result = run(
        console_script(installed_package), "paths", cwd=tmp_path,
        env={**installed_env, "PHYSICSOS_WORKSPACE": str(workspace)},
    )
    assert Path(json.loads(result.stdout)["workspace"]) == workspace


def test_installed_reference_library_is_complete(
    installed_package: Path, installed_env: dict[str, str], tmp_path: Path,
) -> None:
    result = run(sys.executable, "-c", """
import json
from pathlib import Path
import physicsos
from physicsos.tools.case_tools import LoadTAPSCaseReferencesInput, load_taps_case_references
result = load_taps_case_references(LoadTAPSCaseReferencesInput(case_id='install-check', include_ks_dft=True))
print(json.dumps({'module': physicsos.__file__, 'warnings': result.warnings, 'count': len(result.references)}))
""", cwd=tmp_path, env=installed_env)
    payload = json.loads(result.stdout)
    assert Path(payload["module"]).is_relative_to(installed_package)
    assert payload["warnings"] == []
    assert payload["count"] == 10
    references = tmp_path / "cases" / "install-check" / "references"
    for source in (ROOT / "physicsos" / "references").glob("*.md"):
        assert (references / source.name).read_bytes() == source.read_bytes()


def test_embedded_cli_does_not_offer_an_incompatible_update(
    installed_env: dict[str, str], tmp_path: Path,
) -> None:
    result = run(sys.executable, "-c", """
from physicsos.cli import _prepare_deepagents_env
_prepare_deepagents_env()
from deepagents_cli.update_check import is_update_check_enabled
print(is_update_check_enabled())
""", cwd=tmp_path, env={**installed_env, "DEEPAGENTS_CLI_AUTO_UPDATE": "1"})
    assert result.stdout.strip() == "False"


@pytest.mark.parametrize("override", [False, True])
def test_dotenv_settings_reach_deepagents(
    installed_env: dict[str, str], tmp_path: Path, override: bool,
) -> None:
    (tmp_path / ".env").write_text(
        "PHYSICSOS_OPENAI_API_KEY=install-test-placeholder\n"
        "PHYSICSOS_OPENAI_BASE_URL=https://example.invalid/v1\n"
        "PHYSICSOS_OPENAI_MODEL=dotenv-model\n"
        "PHYSICSOS_OPENAI_USE_RESPONSES_API=true\n",
        encoding="utf-8",
    )
    if override:
        installed_env["PHYSICSOS_OPENAI_MODEL"] = "environment-model"
    result = run(sys.executable, "-c", """
import json
import os
from physicsos.cli import _prepare_deepagents_env, _deepagents_model_args, _deepagents_model_params_args
_prepare_deepagents_env()
print(json.dumps({
    'key': os.environ.get('DEEPAGENTS_CLI_OPENAI_API_KEY'),
    'base_url': os.environ.get('OPENAI_BASE_URL'),
    'model': _deepagents_model_args([]),
    'params': json.loads(_deepagents_model_params_args([])[1]),
}))
""", cwd=tmp_path, env=installed_env)
    payload = json.loads(result.stdout)
    assert payload["key"] == "install-test-placeholder"
    assert payload["base_url"] == "https://example.invalid/v1"
    model = "environment-model" if override else "dotenv-model"
    assert payload["model"] == ["--model", f"openai:{model}"]
    assert payload["params"]["use_responses_api"] is True
