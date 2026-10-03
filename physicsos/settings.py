"""Model settings shared by first-run onboarding and the in-app settings screen."""

from __future__ import annotations

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static

from physicsos.config import config_path
from physicsos.model_config import (
    DeepSearchSettings,
    ModelSettings,
    deepsearch_settings,
    model_settings,
    save_deepsearch_settings,
    save_model_settings,
)


PROVIDERS = {
    "openai": ("OpenAI", "https://api.openai.com/v1", "gpt-5.4"),
    "deepseek": ("DeepSeek", "https://api.deepseek.com/v1", "deepseek-chat"),
    "custom": ("自定义 / OpenAI-compatible", "", ""),
}
MODELS = ("gpt-5.4", "gpt-4.1", "deepseek-chat", "deepseek-reasoner")
DEEPSEARCH_MODELS = (
    "gemini-2.5-pro-deepsearch",
    "gemini-3-pro-deepsearch",
    "gemini-2.5-pro-deepsearch-async",
    "grok-3-deepsearch",
    "o3-deep-research",
)


class ModelSettingsScreen(ModalScreen[ModelSettings | None]):
    BINDINGS = [Binding("escape", "cancel", "Cancel", priority=True)]
    DEFAULT_CSS = """
    ModelSettingsScreen { align: center middle; background: $background 80%; }
    #model-settings-panel { width: 78; max-width: 96%; height: 92%; border: round $primary; padding: 0 1; background: $surface; }
    #settings-title { height: auto; text-style: bold; color: $accent; padding: 1 0; }
    #settings-intro { height: auto; margin-bottom: 1; }
    #settings-fields { height: 1fr; }
    #settings-fields Label { margin-top: 1; height: auto; }
    #settings-fields Input, #settings-fields Select { width: 100%; }
    #settings-notice { height: auto; color: $text-muted; margin: 1 0; }
    #settings-status { height: auto; max-height: 4; color: $warning; }
    #settings-actions { height: 3; margin-top: 1; align-horizontal: right; }
    #settings-actions Button { min-width: 10; margin-left: 1; }
    """

    def __init__(self, *, first_run: bool = False, current: ModelSettings | None = None) -> None:
        super().__init__()
        self.current = current or model_settings()
        self.current_deepsearch = deepsearch_settings()
        self.first_run = first_run
        self._preset = next((key for key, (_, url, _) in PROVIDERS.items() if url == self.current.base_url), "custom")

    def compose(self) -> ComposeResult:
        with Vertical(id="model-settings-panel"):
            yield Static("欢迎使用 PhysicsOS · 首次配置" if self.first_run else "PhysicsOS · 模型设置 / Model settings", id="settings-title")
            yield Static("配置模型后即可开始物理仿真。以后可点击首页「模型设置」或输入 /settings 修改。", id="settings-intro")
            with VerticalScroll(id="settings-fields"):
                yield Label("服务商 / Provider")
                yield Select([(label, key) for key, (label, _, _) in PROVIDERS.items()], value=self._preset, allow_blank=False, id="provider")
                yield Label("API 地址 / Base URL")
                yield Input(self.current.base_url, placeholder="https://api.example.com/v1", id="base-url")
                yield Label("API Key（输入隐藏；留空保留当前服务的 Key）")
                yield Input(password=True, placeholder="已配置，留空保留" if self.current.api_key else "请输入服务商提供的 API Key", id="api-key")
                yield Label("快捷选择模型 / Model presets")
                names = list(dict.fromkeys([self.current.name, *MODELS]))
                yield Select([(name, name) for name in names if name] + [("自定义模型 / Custom", "")], value=self.current.name or "", allow_blank=False, id="model-choice")
                yield Label("模型名称 / Model ID（可直接填写）")
                yield Input(self.current.name, placeholder="填写服务商的 Model ID", id="model-name")
                yield Label("API 类型")
                yield Select([("Chat Completions", False), ("Responses", True)], value=self.current.use_responses_api, allow_blank=False, id="api-type")
                yield Label("检索模型 / DeepSearch（文献检索与综述）")
                yield Select(
                    [("与主模型相同 / Same as main model", True), ("使用独立服务 / Separate", False)],
                    value=self.current_deepsearch.reuse_main_model, allow_blank=False, id="deepsearch-reuse",
                )
                yield Label("DeepSearch 模型 / Model ID")
                yield Select(
                    [(name, name) for name in dict.fromkeys([self.current_deepsearch.name, *DEEPSEARCH_MODELS]) if name] + [("自定义模型 / Custom", "")],
                    value=self.current_deepsearch.name or "", allow_blank=False, id="deepsearch-choice",
                )
                yield Input(self.current_deepsearch.name, placeholder="例如 gemini-2.5-pro-deepsearch", id="deepsearch-name")
                yield Label("DeepSearch API 地址（留空则用主模型地址）")
                yield Input(self.current_deepsearch.base_url, placeholder="留空继承主模型地址", id="deepsearch-url")
                yield Label("DeepSearch API Key（留空则用主模型 Key）")
                yield Input(password=True, placeholder="已配置，留空保留" if self.current_deepsearch.api_key else "留空继承主模型 Key", id="deepsearch-key")
                yield Static(
                    f"配置保存在 {config_path()}。Key 仅保存在本机，不显示在聊天中。\n"
                    "环境变量在下次启动时优先于保存的配置。检测连接会获取服务商模型列表。\n"
                    "DeepSearch 用于 knowledge-agent 的文献检索与综述，可与主模型使用不同服务商。",
                    id="settings-notice", markup=False,
                )
            yield Static("", id="settings-status", markup=False)
            with Horizontal(id="settings-actions"):
                yield Button("检测连接", id="test-connection")
                yield Button("取消", id="cancel-settings")
                yield Button("保存并继续" if self.first_run else "保存并应用", variant="primary", id="save-settings")

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#cancel-settings")
    def cancel_settings(self) -> None:
        self.action_cancel()

    def on_mount(self) -> None:
        self._sync_deepsearch_fields(self.current_deepsearch.reuse_main_model)

    def _sync_deepsearch_fields(self, reuse_main_model: bool) -> None:
        for widget_id in ("#deepsearch-url", "#deepsearch-key"):
            self.query_one(widget_id, Input).disabled = reuse_main_model

    @on(Select.Changed, "#provider")
    def provider_changed(self, event: Select.Changed) -> None:
        if event.value == self._preset or event.value == Select.BLANK:
            return
        self._preset = str(event.value)
        _, url, name = PROVIDERS[self._preset]
        self.query_one("#base-url", Input).value = url
        self.query_one("#model-name", Input).value = name
        self.query_one("#model-choice", Select).value = name
        self.query_one("#api-key", Input).value = ""
        self.query_one("#api-type", Select).value = False
        # DeepSearch had its own key for the previous endpoint; do not carry it over.
        if self.query_one("#deepsearch-reuse", Select).value is False:
            self.query_one("#deepsearch-key", Input).value = ""

    @on(Select.Changed, "#model-choice")
    def model_changed(self, event: Select.Changed) -> None:
        if event.value != Select.BLANK:
            self.query_one("#model-name", Input).value = str(event.value)

    @on(Select.Changed, "#deepsearch-choice")
    def deepsearch_model_changed(self, event: Select.Changed) -> None:
        if event.value != Select.BLANK:
            self.query_one("#deepsearch-name", Input).value = str(event.value)

    @on(Select.Changed, "#deepsearch-reuse")
    def deepsearch_reuse_changed(self, event: Select.Changed) -> None:
        self._sync_deepsearch_fields(event.value is True)

    def deepsearch_values(self, main: ModelSettings | None = None) -> DeepSearchSettings:
        reuse = self.query_one("#deepsearch-reuse", Select).value is True
        if reuse:
            return DeepSearchSettings(
                enabled=True, name=self.current_deepsearch.name,
                base_url="", api_key="", reuse_main_model=True,
            )
        base_url = self.query_one("#deepsearch-url", Input).value.strip().rstrip("/")
        api_key = self.query_one("#deepsearch-key", Input).value.strip()
        # An untouched key field keeps the stored key for the same endpoint.
        if not api_key and base_url == self.current_deepsearch.base_url:
            api_key = self.current_deepsearch.api_key
        result = DeepSearchSettings(
            enabled=True,
            name=self.query_one("#deepsearch-name", Input).value.strip(),
            base_url=base_url, api_key=api_key, reuse_main_model=False,
        )
        result.validate(main=main)
        return result

    def values(self, *, require_model: bool = True) -> ModelSettings:
        base_url = self.query_one("#base-url", Input).value.strip().rstrip("/")
        api_key = self.query_one("#api-key", Input).value.strip()
        # A key for the old endpoint must not be silently sent to a new provider.
        if not api_key and base_url == self.current.base_url:
            api_key = self.current.api_key
        result = ModelSettings(
            name=self.query_one("#model-name", Input).value.strip().removeprefix("openai:"),
            base_url=base_url, api_key=api_key,
            use_responses_api=bool(self.query_one("#api-type", Select).value),
        )
        result.validate(require_model=require_model)
        return result

    @on(Button.Pressed, "#save-settings")
    def save_settings(self) -> None:
        try:
            result = self.values()
            deepsearch = self.deepsearch_values(main=result)
            save_model_settings(result)
            save_deepsearch_settings(deepsearch, main=result)
        except ValueError as exc:
            self.query_one("#settings-status", Static).update(str(exc))
            return
        except OSError:
            self.query_one("#settings-status", Static).update("配置保存失败，请检查配置目录是否可写。")
            return
        self.dismiss(result)

    @on(Button.Pressed, "#test-connection")
    @work(exclusive=True, group="model-connection")
    async def test_connection(self) -> None:
        import httpx

        status = self.query_one("#settings-status", Static)
        try:
            model = self.values(require_model=False)
        except ValueError as exc:
            status.update(str(exc))
            return
        status.update("正在连接并获取模型列表…")
        try:
            async with httpx.AsyncClient(timeout=10, follow_redirects=False) as client:
                response = await client.get(f"{model.base_url}/models", headers={"Authorization": f"Bearer {model.api_key}"})
            response.raise_for_status()
            payload = response.json()
            models = [item["id"] for item in payload.get("data", []) if isinstance(item, dict) and isinstance(item.get("id"), str)]
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            message = "API Key 无效或无访问权限。" if code in (401, 403) else f"模型列表返回 HTTP {code}；若服务商不支持 /models，可手动填写后保存。"
            status.update(message)
            return
        except (httpx.HTTPError, ValueError, AttributeError):
            status.update("未能获取模型列表，请检查 API 地址和网络；也可以手动填写模型后保存。")
            return
        choice = self.query_one("#model-choice", Select)
        names = list(dict.fromkeys(name for name in [model.name, *models] if name))
        choice.set_options([(name, name) for name in names] + [("自定义模型 / Custom", "")])
        choice.value = model.name or (names[0] if names else "")
        status.update(f"连接正常，获取到 {len(models)} 个模型。请选择模型后保存；具体模型可用性以请求结果为准。")


class ModelSettingsApp(App[ModelSettings | None]):
    TITLE = "PhysicsOS · Model settings"

    def on_mount(self) -> None:
        self.push_screen(ModelSettingsScreen(), self.exit)
