from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import stat

import httpx
import pytest
from textual.widgets import Button, Input, Select, Static

from physicsos.config import config_path, default_config, load_config, save_config
from physicsos.model_config import ModelSettings, apply_model_environment, model_settings, save_model_settings
from physicsos.settings import ModelSettingsApp, ModelSettingsScreen


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith(("PHYSICSOS_", "DEEPAGENTS_", "OPENAI_", "LANGSMITH_", "LANGCHAIN_")):
            monkeypatch.delenv(key)
    monkeypatch.setenv("PHYSICSOS_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("DEEPAGENTS_CLI_NO_UPDATE_CHECK", "1")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.chdir(tmp_path)


def test_standard_openai_environment_and_config_precedence(monkeypatch):
    save_model_settings(ModelSettings("saved-model", "https://saved.invalid/v1", "saved-key"))
    monkeypatch.setenv("OPENAI_API_KEY", "standard-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://standard.invalid/v1")
    assert model_settings().api_key == "standard-key"
    assert model_settings().base_url == "https://standard.invalid/v1"
    monkeypatch.setenv("PHYSICSOS_OPENAI_API_KEY", "physics-key")
    monkeypatch.setenv("PHYSICSOS_OPENAI_MODEL", "custom-model")
    assert model_settings().api_key == "physics-key"
    assert model_settings().name == "custom-model"


def test_model_save_preserves_other_settings_and_keeps_key_private():
    config = default_config()
    assert config["model"]["base_url"] == "https://api.openai.com/v1"
    config["cloud"]["access_token"] = "existing-cloud-token"
    save_config(config)
    model = ModelSettings("my-model", "https://example.invalid/v1", "private-test-key", True)
    save_model_settings(model)
    assert load_config()["cloud"]["access_token"] == "existing-cloud-token"
    assert model_settings() == model
    assert "private-test-key" not in repr(model)
    assert "private-test-key" not in json.dumps(model.params)
    if os.name != "nt":
        assert stat.S_IMODE(config_path().stat().st_mode) == 0o600


def test_settings_screen_save_masks_key_and_works_in_small_terminal():
    async def scenario():
        app = ModelSettingsApp()
        async with app.run_test(size=(80, 24)) as pilot:
            screen = app.screen
            assert isinstance(screen, ModelSettingsScreen)
            assert screen.query_one("#api-key", Input).password
            screen.query_one("#base-url", Input).value = "https://example.invalid/v1"
            screen.query_one("#model-name", Input).value = "provider-model"
            screen.query_one("#api-key", Input).value = "ui-test-key"
            screen.query_one("#api-type", Select).value = True
            await pilot.click("#save-settings")
        assert app.return_value.name == "provider-model"
        assert model_settings().api_key == "ui-test-key"
        assert model_settings().use_responses_api
    asyncio.run(scenario())


def test_cancel_does_not_change_existing_configuration():
    original = ModelSettings("saved-model", "https://saved.invalid/v1", "saved-key")
    save_model_settings(original)
    before = config_path().read_bytes()
    async def scenario():
        app = ModelSettingsApp()
        async with app.run_test(size=(80, 24)) as pilot:
            app.screen.query_one("#api-key", Input).value = "not-saved"
            await pilot.press("escape")
        assert app.return_value is None
    asyncio.run(scenario())
    assert config_path().read_bytes() == before


def test_changing_endpoint_requires_a_key_for_the_new_provider():
    save_model_settings(ModelSettings("saved-model", "https://saved.invalid/v1", "saved-key"))
    async def scenario():
        app = ModelSettingsApp()
        async with app.run_test(size=(80, 24)) as pilot:
            screen = app.screen
            screen.query_one("#base-url", Input).value = "https://new.invalid/v1"
            await pilot.click("#save-settings")
            assert isinstance(app.screen, ModelSettingsScreen)
            assert "API Key" in str(screen.query_one("#settings-status", Static).content)
            assert model_settings().base_url == "https://saved.invalid/v1"
            await pilot.press("escape")
    asyncio.run(scenario())


def test_fetch_models_uses_entered_endpoint_without_saving(monkeypatch):
    requests = []
    real_client = httpx.AsyncClient
    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"data": [{"id": "provider-model-a"}, {"id": "provider-model-b"}]})
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs))
    async def scenario():
        app = ModelSettingsApp()
        async with app.run_test(size=(80, 24)) as pilot:
            screen = app.screen
            screen.query_one("#base-url", Input).value = "https://models.invalid/v1"
            screen.query_one("#model-name", Input).value = ""
            screen.query_one("#api-key", Input).value = "connection-test-key"
            await pilot.click("#test-connection")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert screen.query_one("#model-name", Input).value == "provider-model-a"
            assert "2" in str(screen.query_one("#settings-status", Static).content)
            assert not config_path().exists()
            await pilot.press("escape")
    asyncio.run(scenario())
    assert str(requests[0].url) == "https://models.invalid/v1/models"
    assert requests[0].headers["Authorization"] == "Bearer connection-test-key"


def test_new_settings_override_stale_runtime_environment():
    apply_model_environment(ModelSettings("old", "https://old.invalid/v1", "old-key"))
    new = ModelSettings("new", "https://new.invalid/v1", "new-key", True)
    save_model_settings(new)
    apply_model_environment(new)
    assert model_settings() == new
    assert os.environ["DEEPAGENTS_CLI_OPENAI_API_KEY"] == "new-key"
    assert os.environ["OPENAI_BASE_URL"] == new.base_url


def quiet_tui(monkeypatch, *, start_server=False):
    from physicsos.tui import PhysicsOSApp, install_settings_ui
    install_settings_ui()
    base = PhysicsOSApp.__bases__[0]
    async def initialize(app):
        if start_server:
            app.run_worker(app._start_server_background, group="server-startup")
    monkeypatch.setattr(base, "_post_paint_init", initialize)
    monkeypatch.setattr(base, "_prewarm_deferred_imports", lambda self: None)
    return PhysicsOSApp, base


@pytest.mark.parametrize("entry", ["button", "f2", "/settings", "/model"])
def test_settings_can_open_after_server_start_failure(monkeypatch, entry):
    PhysicsOSApp, _ = quiet_tui(monkeypatch)
    async def scenario():
        app = PhysicsOSApp()
        app._server_startup_error = "No credentials found for provider 'openai'"
        async with app.run_test(size=(80, 24)) as pilot:
            assert app.query_one("#physicsos-settings", Button).region.y == 0
            if entry == "button":
                await pilot.click("#physicsos-settings")
            elif entry == "f2":
                await pilot.press("f2")
            else:
                await app._handle_command(entry)
                await pilot.pause()
            assert isinstance(app.screen, ModelSettingsScreen)
            await pilot.click("#cancel-settings")
            assert not isinstance(app.screen, ModelSettingsScreen)
    asyncio.run(scenario())


@pytest.mark.parametrize("spec", ["openai:gpt-5.4", "gpt-4o"])
def test_first_run_configures_credentials_before_starting_server(monkeypatch, spec):
    PhysicsOSApp, base = quiet_tui(monkeypatch, start_server=True)
    starts = []
    async def start(app):
        starts.append((os.environ["OPENAI_API_KEY"], app._server_kwargs.copy()))
        app._connecting = False
    monkeypatch.setattr(base, "_start_server_background", start)
    async def scenario():
        app = PhysicsOSApp(server_kwargs={"model_name": spec}, model_kwargs={"model_spec": spec})
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            assert isinstance(app.screen, ModelSettingsScreen)
            assert app.screen.first_run
            assert app.screen.query_one("#model-name", Input).value == spec.removeprefix("openai:")
            assert starts == []
            app.screen.query_one("#api-key", Input).value = "first-run-key"
            app.screen.query_one("#model-name", Input).value = "custom-model"
            await pilot.click("#save-settings")
            await pilot.pause()
            assert len(starts) == 1
            assert starts[0][0] == "first-run-key"
            assert starts[0][1]["model_name"] == "openai:custom-model"
            assert model_settings().api_key == "first-run-key"
    asyncio.run(scenario())


def test_cancel_first_run_does_not_start_server(monkeypatch):
    PhysicsOSApp, base = quiet_tui(monkeypatch, start_server=True)
    starts = []
    async def start(app):
        starts.append(True)
    monkeypatch.setattr(base, "_start_server_background", start)
    async def scenario():
        app = PhysicsOSApp(server_kwargs={"model_name": "openai:gpt-5.4"})
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            assert isinstance(app.screen, ModelSettingsScreen)
            await pilot.click("#cancel-settings")
        assert starts == []
        assert not config_path().exists()
    asyncio.run(scenario())


def test_changed_key_restarts_backend_even_when_model_name_is_unchanged(monkeypatch):
    from types import SimpleNamespace
    import deepagents_cli.config as cli_config
    PhysicsOSApp, _ = quiet_tui(monkeypatch)
    monkeypatch.setattr(cli_config, "create_model", lambda **kwargs: SimpleNamespace(apply_to_settings=lambda: None))
    restarted = []
    class Server:
        url = "http://127.0.0.1:12345"
        def update_env(self, **kwargs):
            self.overrides = kwargs
        async def restart(self):
            restarted.append(os.environ["OPENAI_API_KEY"])
    async def scenario():
        app = PhysicsOSApp(thread_id="keep-this-thread")
        async with app.run_test(size=(80, 24)):
            app._server_proc = Server()
            app._server_kwargs = {"model_name": "openai:gpt-5.4"}
            app._server_startup_error = "old error"
            await app._apply_physicsos_settings(ModelSettings("gpt-5.4", "https://new.invalid/v1", "new-key"))
            assert restarted == ["new-key"]
            assert app._lc_thread_id == "keep-this-thread"
            assert app._agent is not None
            assert app._server_startup_error is None
            assert "new-key" not in json.dumps(app._server_proc.overrides)
            assert not app._connecting
    asyncio.run(scenario())
