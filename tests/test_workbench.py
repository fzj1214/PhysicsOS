from __future__ import annotations

import asyncio
import json
from pathlib import Path
import runpy

import pytest
from textual.widgets import DataTable, Input, Select, Static, TabbedContent, TextArea

from physicsos.rsi import RSIRuntime
from physicsos.schemas.rsi import RevisionProviderSpec
from physicsos.workbench import WorkbenchApp, WorkbenchScreen
from physicsos.workbench_data import records


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(tmp_path))
    fixture = runpy.run_path(str(Path(__file__).with_name("test_rsi.py")))
    setup = fixture["setup"].__wrapped__(tmp_path, monkeypatch)
    registered = fixture["suite"](setup)
    initial = fixture["strategy"](setup)
    source = tmp_path / "revision.py"
    source.write_text("def revise_strategy(config):\n    return {'stop':True,'rationale':'Development strategy is retained.'}\n")
    provider = setup[0].register_revision_provider(RevisionProviderSpec(name="test-provider", description="TUI provider", python_uri=str(source)))
    return tmp_path, setup[0], registered, initial, provider


def test_empty_workbench_opens_three_tabs_without_model_credentials(tmp_path):
    async def scenario():
        app = WorkbenchApp(tmp_path)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, WorkbenchScreen)
            assert screen.query_one("#wb-case-table", DataTable).row_count == 0
            assert screen.query_one("#wb-rsi-table", DataTable).row_count == 0
            screen.query_one("#wb-tabs", TabbedContent).active = "wb-evidence"
            assert screen.query_one("#wb-details", TextArea).read_only
            assert not (tmp_path / "data" / "rsi" / "state.sqlite3").exists()
            await pilot.press("escape")
    asyncio.run(scenario())


def test_registered_data_and_corrupted_records_are_visible_without_crashing(workspace):
    root, _, _, _, _ = workspace
    corrupted = root / "data" / "rsi" / "strategies" / "broken" / "manifest.json"
    corrupted.parent.mkdir()
    corrupted.write_text("not JSON")
    async def scenario():
        app = WorkbenchApp(root)
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert screen.query_one("#wb-rsi-table", DataTable).row_count == 3
            assert "broken" in str(screen.query_one("#wb-status", Static).content)
            screen.query_one("#wb-tabs", TabbedContent).active = "wb-rsi"
            await pilot.pause()
            screen.query_one("#wb-rsi-table", DataTable).focus()
            await pilot.press("enter")
            await screen.workers.wait_for_complete()
            await pilot.pause()
            assert screen.query_one("#wb-tabs", TabbedContent).active == "wb-evidence"
            assert "记录内容" in screen.query_one("#wb-details", TextArea).text
    asyncio.run(scenario())


def test_workbench_starts_real_bounded_campaign_and_displays_evidence(workspace):
    root, _, suite, initial, provider = workspace
    async def scenario():
        app = WorkbenchApp(root)
        async with app.run_test(size=(120, 45)) as pilot:
            await pilot.pause()
            screen = app.screen
            key_for = lambda ref: next(item.key for item in screen.rows.values() if item.reference == ref)
            screen.query_one("#wb-suite", Select).value = key_for(suite.manifest)
            screen.query_one("#wb-provider", Select).value = key_for(provider.manifest)
            screen.query_one("#wb-initial", Select).value = key_for(initial.manifest)
            screen.query_one("#wb-rounds", Input).value = "1"
            screen.query_one("#wb-runs", Input).value = "8"
            screen.start_campaign()
            assert screen.busy
            await screen.workers.wait_for_complete()
            await pilot.pause()
            assert not screen.busy
            assert "已晋升" in str(screen.query_one("#wb-status", Static).content)
            result = json.loads(screen.query_one("#wb-details", TextArea).text)
            assert result["status"] == "promoted"
            assert result["reserved_kernel_runs"] <= 8
            assert result["final_evaluation"] is not None
    asyncio.run(scenario())


def test_workbench_invalid_budget_or_active_agent_does_not_launch_task(workspace):
    root, _, _, _, _ = workspace
    async def scenario():
        app = WorkbenchApp(root)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            screen = app.screen
            screen.mutations_allowed = lambda: False
            screen.start_campaign()
            assert not screen.busy
            assert "agent 正在工作" in str(screen.query_one("#wb-status", Static).content)
            screen.mutations_allowed = lambda: True
            screen.start_campaign()
            assert not screen.busy
            assert "请选择" in str(screen.query_one("#wb-status", Static).content)
    asyncio.run(scenario())


@pytest.mark.parametrize("entry", ["button", "f3", "/workbench", "/rsi"])
def test_workbench_is_integrated_into_chat_tui(tmp_path, monkeypatch, entry):
    from physicsos.tui import PhysicsOSApp, install_settings_ui
    monkeypatch.setenv("PHYSICSOS_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("PHYSICSOS_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("DEEPAGENTS_CLI_NO_UPDATE_CHECK", "1")
    install_settings_ui()
    base = PhysicsOSApp.__bases__[0]
    async def initialize(app):
        pass
    monkeypatch.setattr(base, "_post_paint_init", initialize)
    monkeypatch.setattr(base, "_prewarm_deferred_imports", lambda self: None)
    async def scenario():
        app = PhysicsOSApp(cwd=tmp_path)
        async with app.run_test(size=(80, 24)) as pilot:
            if entry == "button":
                await pilot.click("#physicsos-workbench")
            elif entry == "f3":
                await pilot.press("f3")
            else:
                await app._handle_command(entry)
            await pilot.pause()
            assert isinstance(app.screen, WorkbenchScreen)
            assert app.screen.workspace == tmp_path
            from deepagents_cli.widgets.messages import AppMessage
            await app._mount_message(AppMessage("后台聊天消息继续保留"))
            if entry == "/rsi":
                assert app.screen.query_one("#wb-tabs", TabbedContent).active == "wb-rsi"
            await pilot.press("escape")
            assert not isinstance(app.screen, WorkbenchScreen)
            assert "后台聊天消息继续保留" in str(app.query_one("#messages").render()) or any("后台聊天消息继续保留" in str(getattr(widget, "content", "")) for widget in app.query_one("#messages").children)
    asyncio.run(scenario())
