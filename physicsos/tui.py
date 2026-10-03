"""PhysicsOS onboarding and settings integration for the pinned DeepAgents TUI."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from textual import on, work
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Button, Static

import deepagents_cli.app as cli_app
from deepagents_cli.widgets.messages import AppMessage
from deepagents_cli.widgets.welcome import WelcomeBanner

from physicsos.model_config import (
    ModelSettings,
    apply_deepsearch_environment,
    apply_model_environment,
    deepsearch_settings,
    model_settings,
    uses_openai_model,
)
from physicsos.settings import ModelSettingsScreen


class PhysicsOSApp(cli_app.DeepAgentsApp):
    TITLE = "PhysicsOS"
    CSS_PATH = str(Path(cli_app.__file__).with_name("app.tcss"))
    CSS = """
    #physicsos-toolbar { height: 3; padding: 0 1; background: $panel; }
    #physicsos-current-model { width: 1fr; height: 3; content-align: left middle; }
    #physicsos-settings { min-width: 20; height: 3; }
    """
    BINDINGS = [Binding("f2", "physicsos_settings", "模型设置", priority=True)]

    def compose(self):
        spec = (self._server_kwargs or {}).get("model_name") or model_settings().spec
        with Horizontal(id="physicsos-toolbar"):
            yield Static(f"PhysicsOS · {spec}", id="physicsos-current-model", markup=False)
            yield Button("模型设置 / F2", id="physicsos-settings")
        yield from super().compose()

    def _configure_runtime(self, model: ModelSettings) -> None:
        from deepagents_cli.config import settings
        from deepagents_cli.model_config import clear_caches

        apply_model_environment(model)
        apply_deepsearch_environment(deepsearch_settings())
        settings.reload_from_environment()
        clear_caches()
        if self._server_kwargs is not None:
            self._server_kwargs.update(model_name=model.spec, model_params=model.params)
        self._model_kwargs = {
            "model_spec": model.spec, "extra_kwargs": model.params,
            "profile_overrides": self._profile_override,
        }
        self._model_override = None
        self._model_params_override = None
        self._server_startup_error = None
        self.query_one("#physicsos-current-model", Static).update(f"PhysicsOS · {model.spec}")

    async def _start_server_background(self) -> None:
        spec = str((self._model_kwargs or {}).get("model_spec") or (self._server_kwargs or {}).get("model_name") or "openai:")
        current = model_settings()
        if uses_openai_model(spec) and not current.api_key.strip():
            current = ModelSettings(name=spec.removeprefix("openai:") or current.name, base_url=current.base_url, api_key="", use_responses_api=current.use_responses_api)
            configured = await self.push_screen_wait(ModelSettingsScreen(first_run=True, current=current))
            if configured is None:
                self.exit()
                return
            self._configure_runtime(configured)
        await super()._start_server_background()

    @on(Button.Pressed, "#physicsos-settings")
    def open_settings(self) -> None:
        self.action_physicsos_settings()

    @work(group="physicsos-settings")
    async def action_physicsos_settings(self) -> None:
        if isinstance(self.screen, ModelSettingsScreen) or getattr(self, "_physicsos_settings_open", False):
            return
        if self._agent_running or self._shell_running or self._connecting or self._thread_switching or self._startup_sequence_running:
            self.notify("请等待当前操作结束后修改模型设置。", timeout=3)
            return
        self._physicsos_settings_open = True
        try:
            configured = await self.push_screen_wait(ModelSettingsScreen())
            if configured is not None:
                await self._apply_physicsos_settings(configured)
        finally:
            self._physicsos_settings_open = False

    async def _apply_physicsos_settings(self, configured: ModelSettings) -> None:
        """Restart the local backend with new credentials, keeping the same thread."""
        from deepagents_cli._env_vars import SERVER_ENV_PREFIX
        from deepagents_cli.config import create_model, settings
        from deepagents_cli.remote_client import RemoteAgent

        self._configure_runtime(configured)
        self._connecting = True
        banner = self.query_one("#welcome-banner", WelcomeBanner)
        banner.set_connecting()
        try:
            await self._await_prewarm_imports()
            create_model(**self._model_kwargs).apply_to_settings()
            self._model_kwargs = None
            if self._server_proc is None:
                await super()._start_server_background()
            else:
                self._server_proc.update_env(**{
                    f"{SERVER_ENV_PREFIX}MODEL": configured.spec,
                    f"{SERVER_ENV_PREFIX}MODEL_PARAMS": json.dumps(configured.params),
                })
                await self._server_proc.restart()
                self._agent = RemoteAgent(url=self._server_proc.url, graph_name="agent")
                self._connecting = False
                banner.set_connected(self._mcp_tool_count, mcp_unauthenticated=self._mcp_unauthenticated, mcp_errored=self._mcp_errored)
            if self._status_bar:
                self._status_bar.set_model(provider=settings.model_provider or "openai", model=settings.model_name or configured.name)
            await self._mount_message(AppMessage("模型配置已保存并应用。可随时通过 /settings 或 F2 修改。"))
            if not self._connecting:
                await self._process_next_from_queue()
        except asyncio.CancelledError:
            raise
        except Exception:
            self._agent = None
            self.post_message(self.ServerStartFailed(error=RuntimeError("配置已保存，但连接失败。请打开「模型设置 / F2」检查 API 地址、Key 和模型后重试。")))

    async def _handle_command(self, command: str) -> None:
        cmd = command.strip().lower()
        if cmd in {"/settings", "/config", "/model"}:
            self.action_physicsos_settings()
            return
        if cmd == "/help":
            await self._mount_message(AppMessage("PhysicsOS 设置：点击顶部「模型设置」、按 F2 或输入 /settings。\n可配置服务商、API 地址、API Key、模型和 API 类型；退出后也可运行 physicsos config。"))
        await super()._handle_command(command)


def install_settings_ui() -> None:
    from deepagents_cli import command_registry
    from deepagents_cli.widgets import welcome

    if cli_app.DeepAgentsApp is PhysicsOSApp:
        return
    entry = command_registry.SlashCommand(
        name="/settings", description="模型、API 地址与 API Key 设置 / Model settings",
        bypass_tier=command_registry.BypassTier.IMMEDIATE_UI, aliases=("/config",),
    )
    command_registry.COMMANDS += (entry,)
    command_registry.SLASH_COMMANDS.append(entry.to_entry())
    command_registry.IMMEDIATE_UI |= {"/settings", "/config"}
    command_registry.ALL_CLASSIFIED |= {"/settings", "/config"}
    welcome._TIPS = [
        "模型与 API Key 设置：点击顶部按钮、按 F2 或输入 /settings",
        "退出后也可运行 physicsos config 修改模型配置",
        "输入物理问题开始仿真；使用 @ 引用几何、材料或其他输入文件",
    ]
    cli_app.DeepAgentsApp = PhysicsOSApp
