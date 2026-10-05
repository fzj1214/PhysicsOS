"""Simulation and RSI workbench mounted alongside the existing chat TUI."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, DataTable, Input, Label, Select, Static, TabbedContent, TabPane, TextArea

from physicsos.config import runtime_paths
from physicsos.rsi import RSIRuntime
from physicsos.schemas.rsi import AssessCapabilityInput, ImproveStrategiesInput, PromoteStrategyInput, RollbackStrategyInput
from physicsos.workbench_data import WorkbenchRecord, active_strategies, artifact_references, inspect_record, open_artifact, records


STATUS = {"ready": "可模拟", "needs_input": "需补充输入", "needs_review": "需检查", "backend_unavailable": "后端不可用",
          "success": "执行成功", "verified": "验证通过", "uncertain": "证据不足", "failed": "失败", "registered": "已登记",
          "promoted": "已晋升", "eligible": "可晋升", "rejected": "未通过", "blocked": "已阻止", "needs_holdout": "待保留集检查",
          "budget_exhausted": "预算耗尽", "no_improvement": "未发现改进", "rolled_back": "已回滚"}
CATEGORY = {"domain": "几何准备", "run": "求解", "verification": "独立验证", "convergence": "细化检查",
            "strategy": "策略", "suite": "基准集", "provider": "修订 provider", "campaign": "自动修订",
            "evaluation": "候选评估", "transition": "晋升 / 回滚"}


class WorkbenchScreen(ModalScreen):
    BINDINGS = [Binding("escape", "close_workbench", "返回聊天", priority=True), Binding("f5", "refresh", "刷新", priority=True)]
    DEFAULT_CSS = """
    WorkbenchScreen { background: $background; }
    #wb-header { height: 3; padding: 0 1; background: $panel; }
    #wb-title { width: 1fr; height: 3; content-align: left middle; text-style: bold; }
    #wb-tabs { height: 1fr; }
    #wb-summary, #wb-status, #wb-live { height: auto; padding: 0 1; }
    #wb-status { color: $warning; max-height: 4; }
    #wb-filter { margin: 0 1; }
    .wb-table { height: 1fr; min-height: 5; margin: 0 1; }
    #wb-rsi-table { height: 1fr; min-height: 5; }
    #wb-rsi-form { height: 1fr; min-height: 8; padding: 0 1; }
    #wb-rsi-form Label { height: auto; }
    #wb-rsi-form Select, #wb-rsi-form Input { height: 3; }
    #wb-rsi-actions { height: 3; }
    #wb-rsi-actions Button { width: 1fr; min-width: 12; }
    #wb-budget-row { height: 3; }
    #wb-budget-row Input { width: 1fr; }
    #wb-details { height: 1fr; }
    #wb-artifact { margin: 0 1; }
    #wb-artifact-path { height: auto; padding: 0 1; }
    """

    def __init__(self, workspace: str | Path | None = None, *, active_tab: str = "wb-cases", mutations_allowed=None):
        super().__init__()
        self.workspace = Path(workspace or runtime_paths().workspace).resolve()
        self.active_tab = active_tab
        self.mutations_allowed = mutations_allowed or (lambda: True)
        self.rows: dict[str, WorkbenchRecord] = {}
        self.active = {}
        self.current: WorkbenchRecord | None = None
        self.busy = False
        self.references = {}
        self._refreshing = False

    def compose(self) -> ComposeResult:
        with Horizontal(id="wb-header"):
            yield Static("PhysicsOS · 仿真工作台", id="wb-title")
            yield Button("刷新 / F5", id="wb-refresh")
            yield Button("返回 / Esc", id="wb-close")
        yield Static("正在读取工作区…", id="wb-summary", markup=False)
        with TabbedContent(initial=self.active_tab, id="wb-tabs"):
            with TabPane("案例与运行", id="wb-cases"):
                yield Input(placeholder="筛选案例名称、状态或记录类型", id="wb-filter")
                yield DataTable(id="wb-case-table", classes="wb-table", cursor_type="row", zebra_stripes=True)
            with TabPane("RSI 策略与修订", id="wb-rsi"):
                yield DataTable(id="wb-rsi-table", classes="wb-table", cursor_type="row", zebra_stripes=True)
                with VerticalScroll(id="wb-rsi-form"):
                    yield Label("自动修订：选择已登记的基准集与 provider")
                    yield Select([], prompt="选择基准集", id="wb-suite")
                    yield Select([], prompt="选择修订 provider", id="wb-provider")
                    yield Select([("使用该范围的默认策略", "default")], value="default", allow_blank=False, id="wb-initial")
                    yield Label("预算：修订轮数 / 求解次数 / 单步超时（秒）")
                    with Horizontal(id="wb-budget-row"):
                        yield Input("3", type="integer", id="wb-rounds", tooltip="最大修订轮数，1–8")
                        yield Input("128", type="integer", id="wb-runs", tooltip="求解预算，包含细化重跑与最终检查")
                        yield Input("60", type="integer", id="wb-timeout", tooltip="每个执行步骤的超时秒数")
                    yield Checkbox("仅运行开发集修订，保留晋升用的保留集", id="wb-development-only")
                    yield Checkbox("通过最终检查后自动晋升", value=True, id="wb-auto-promote")
                with Horizontal(id="wb-rsi-actions"):
                    yield Button("启动修订", variant="primary", id="wb-start")
                    yield Button("查看能力", id="wb-assess")
                    yield Button("晋升评估", id="wb-promote", disabled=True)
                    yield Button("回滚默认", id="wb-rollback", disabled=True)
        yield Static("选择记录可在证据页查看详情。", id="wb-status", markup=False)
        yield Static("", id="wb-live", markup=False)

    def on_mount(self) -> None:
        self.query_one("#wb-case-table", DataTable).add_columns("案例", "记录", "状态", "编号")
        self.query_one("#wb-rsi-table", DataTable).add_columns("名称", "类型", "状态", "默认版本")
        # Mount separately so the detail widgets remain available during refresh.
        self.call_after_refresh(self._mount_evidence_tab)
        self.action_refresh()
        self.set_interval(2, self._poll)

    async def _mount_evidence_tab(self):
        tabs = self.query_one("#wb-tabs", TabbedContent)
        pane = TabPane("验证证据", id="wb-evidence")
        await tabs.add_pane(pane)
        await pane.mount(Static("选择案例或策略记录后查看关联证据。", id="wb-artifact-path", markup=False),
                         Select([], prompt="选择关联产物", id="wb-artifact"),
                         TextArea(read_only=True, show_line_numbers=False, soft_wrap=True, id="wb-details"))

    def action_close_workbench(self) -> None:
        if self.busy:
            self.query_one("#wb-status", Static).update("任务仍在执行；请等待本次操作完成后返回聊天。")
            return
        self.workers.cancel_group(self, "workbench-refresh")
        self.workers.cancel_group(self, "workbench-details")
        self.workers.cancel_group(self, "workbench-artifact")
        if isinstance(self.app, WorkbenchApp):
            self.app.exit()
        else:
            self.dismiss(None)

    @on(Button.Pressed, "#wb-close")
    def close_clicked(self):
        self.action_close_workbench()

    @on(Button.Pressed, "#wb-refresh")
    def refresh_clicked(self):
        self.action_refresh()

    def _poll(self):
        if self.busy and not self._refreshing:
            self.action_refresh()

    @work(group="workbench-refresh", exclusive=True)
    async def action_refresh(self):
        self._refreshing = True
        try:
            found, warnings = await asyncio.to_thread(records, self.workspace)
            active, extra = await asyncio.to_thread(active_strategies, self.workspace)
            self.rows = {item.key: item for item in found}
            self.active = active
            self._fill_tables()
            for selector, category in (("#wb-suite", "suite"), ("#wb-provider", "provider"), ("#wb-initial", "strategy")):
                widget = self.query_one(selector, Select)
                prior = widget.value
                options = [(item.title, item.key) for item in found if item.category == category]
                if category == "strategy":
                    options = [("使用该范围的默认策略", "default"), *options]
                widget.set_options(options)
                if prior in {value for _, value in options}:
                    widget.value = prior
                elif category == "strategy":
                    widget.value = "default"
            self.query_one("#wb-summary", Static).update(f"工作区：{self.workspace}\n{sum(item.category == 'run' for item in found)} 次求解 · {sum(item.category == 'strategy' for item in found)} 个策略版本 · {sum(item.category == 'campaign' for item in found)} 次自动修订")
            if warnings or extra:
                self.query_one("#wb-status", Static).update("\n".join([*warnings, *extra][:3]))
            if self.busy:
                recent = next((item for item in found if item.category == "run"), None)
                self.query_one("#wb-live", Static).update("后台任务执行中。" + (f"最近求解：{recent.title} · {STATUS.get(recent.status, recent.status)}" if recent else ""))
        finally:
            self._refreshing = False

    def _fill_tables(self):
        query = self.query_one("#wb-filter", Input).value.lower().strip()
        for selector, categories in (("#wb-case-table", {"domain", "run", "verification", "convergence"}),
                                     ("#wb-rsi-table", {"strategy", "suite", "provider", "campaign", "evaluation", "transition"})):
            table = self.query_one(selector, DataTable)
            table.clear()
            for item in self.rows.values():
                if item.category not in categories:
                    continue
                if selector == "#wb-case-table" and query and query not in (item.title + item.status + CATEGORY[item.category] + STATUS.get(item.status, item.status)).lower():
                    continue
                status = STATUS.get(item.status, item.status)
                tail = item.path.parent.name[:12]
                if selector == "#wb-rsi-table":
                    state = self.active.get(item.reference.uri)
                    tail = f"第 {state['generation']} 版 · {state['integrity']}" if state else ""
                table.add_row(item.title, CATEGORY[item.category], status, tail, key=item.key)
        if self.current is not None:
            self.current = self.rows.get(self.current.key, self.current)
        self.query_one("#wb-promote", Button).disabled = self.busy or self.current is None or self.current.category != "evaluation" or self.current.status != "eligible"
        self.query_one("#wb-rollback", Button).disabled = self.busy or self.current is None or self.current.category != "strategy" or self.current.reference.uri not in self.active

    @on(Input.Changed, "#wb-filter")
    def filter_changed(self):
        if self.is_mounted:
            self._fill_tables()

    @on(DataTable.RowSelected)
    def selected_record(self, event: DataTable.RowSelected):
        item = self.rows.get(str(event.row_key.value))
        if item is None:
            return
        self.current = item
        self.query_one("#wb-promote", Button).disabled = self.busy or item.category != "evaluation" or item.status != "eligible"
        self.query_one("#wb-rollback", Button).disabled = self.busy or item.category != "strategy" or item.reference.uri not in self.active
        self._load_record(item)

    @work(group="workbench-details", exclusive=True)
    async def _load_record(self, item):
        try:
            text = await asyncio.to_thread(inspect_record, item, self.workspace)
            self.query_one("#wb-details", TextArea).load_text(text)
            self.query_one("#wb-artifact-path", Static).update(str(item.path))
            refs = artifact_references(item.payload)
            self.references = {str(index): ref for index, ref in enumerate(refs)}
            self.query_one("#wb-artifact", Select).set_options([(f"{ref.kind} · {ref.uri}", str(index)) for index, ref in enumerate(refs)])
            self.query_one("#wb-tabs", TabbedContent).active = "wb-evidence"
            self.query_one("#wb-details", TextArea).focus()
        except (ValueError, OSError) as exc:
            self.query_one("#wb-status", Static).update(str(exc))

    @on(Select.Changed, "#wb-artifact")
    @work(group="workbench-artifact", exclusive=True)
    async def selected_artifact(self, event: Select.Changed):
        reference = self.references.get(str(event.value))
        if reference:
            try:
                path, text = await asyncio.to_thread(open_artifact, reference, self.workspace)
                self.query_one("#wb-artifact-path", Static).update(path)
                self.query_one("#wb-details", TextArea).load_text(text)
            except (ValueError, OSError) as exc:
                self.query_one("#wb-status", Static).update(str(exc))

    def _choice(self, selector: str, category: str) -> WorkbenchRecord:
        choice = str(self.query_one(selector, Select).value)
        record = self.rows.get(choice)
        if record is None or record.category != category:
            raise ValueError("请选择" + CATEGORY[category] + "。")
        return record

    def _can_start(self) -> bool:
        if self.busy:
            return False
        if not self.mutations_allowed():
            self.query_one("#wb-status", Static).update("聊天中的 agent 正在工作；请等当前操作结束再启动或更改策略。")
            return False
        return True

    def _set_busy(self, busy: bool):
        self.busy = busy
        for selector in ("#wb-start", "#wb-assess", "#wb-suite", "#wb-provider", "#wb-initial", "#wb-rounds", "#wb-runs", "#wb-timeout", "#wb-development-only", "#wb-auto-promote"):
            self.query_one(selector).disabled = busy
        self.query_one("#wb-promote", Button).disabled = busy or self.current is None or self.current.category != "evaluation" or self.current.status != "eligible"
        self.query_one("#wb-rollback", Button).disabled = busy or self.current is None or self.current.category != "strategy" or self.current.reference.uri not in self.active

    def _launch_operation(self, label, operation):
        self._set_busy(True)
        self._run_operation(label, operation)

    @on(Button.Pressed, "#wb-start")
    def start_campaign(self):
        if not self._can_start():
            return
        try:
            suite = self._choice("#wb-suite", "suite")
            provider = self._choice("#wb-provider", "provider")
            initial = None if self.query_one("#wb-initial", Select).value == "default" else self._choice("#wb-initial", "strategy").reference
            request = ImproveStrategiesInput(suite=suite.reference, revision_provider=provider.reference, initial_strategy=initial,
                                             max_revisions=int(self.query_one("#wb-rounds", Input).value),
                                             max_kernel_runs=int(self.query_one("#wb-runs", Input).value),
                                             timeout_seconds=int(self.query_one("#wb-timeout", Input).value),
                                             include_holdout=not self.query_one("#wb-development-only", Checkbox).value,
                                             auto_promote=self.query_one("#wb-auto-promote", Checkbox).value)
        except (ValueError, TypeError) as exc:
            self.query_one("#wb-status", Static).update("无法启动：" + str(exc))
            return
        self._launch_operation("自动修订", lambda: RSIRuntime(self.workspace).improve(request))

    @on(Button.Pressed, "#wb-assess")
    def assess_selected(self):
        if not self._can_start():
            return
        try:
            if self.current and self.current.category == "strategy":
                revision = RSIRuntime(self.workspace)._strategy(self.current.reference)
                request = AssessCapabilityInput(scope=revision.spec.scope, strategy=self.current.reference)
            else:
                suite = RSIRuntime(self.workspace)._suite(self._choice("#wb-suite", "suite").reference)
                request = AssessCapabilityInput(scope=suite.spec.scope)
        except (ValueError, OSError) as exc:
            self.query_one("#wb-status", Static).update(str(exc))
            return
        self._launch_operation("能力评估", lambda: RSIRuntime(self.workspace).assess(request))

    @on(Button.Pressed, "#wb-promote")
    def promote_selected(self):
        if self._can_start() and self.current and self.current.category == "evaluation":
            request = PromoteStrategyInput(evaluation=self.current.reference)
            self._launch_operation("晋升", lambda: RSIRuntime(self.workspace).promote(request))

    @on(Button.Pressed, "#wb-rollback")
    def rollback_selected(self):
        if not self._can_start() or not self.current or self.current.category != "strategy":
            return
        try:
            runtime = RSIRuntime(self.workspace)
            revision = runtime._strategy(self.current.reference)
            state = runtime.store.active(revision.spec.scope)
            if state["strategy"] != self.current.reference:
                raise ValueError("该记录已不是默认策略，请刷新后选择当前版本。")
            request = RollbackStrategyInput(scope=revision.spec.scope, expected_generation=state["generation"], reason="用户在仿真工作台请求回滚当前默认策略。")
        except (ValueError, OSError) as exc:
            self.query_one("#wb-status", Static).update(str(exc))
            return
        self._launch_operation("回滚", lambda: runtime.rollback(request))

    @work(group="workbench-operation", exclusive=True)
    async def _run_operation(self, label, operation):
        self._set_busy(True)
        self.query_one("#wb-status", Static).update(label + "已启动，正在处理真实工作区记录…")
        try:
            result = await asyncio.to_thread(operation)
            payload = result.model_dump(mode="json")
            self.query_one("#wb-details", TextArea).load_text(json.dumps(payload, indent=2, ensure_ascii=False))
            self.query_one("#wb-tabs", TabbedContent).active = "wb-evidence"
            self.query_one("#wb-details", TextArea).focus()
            result_status = str(payload.get("status") or payload.get("confidence") or "complete")
            self.query_one("#wb-status", Static).update(label + "：" + STATUS.get(result_status, result_status))
            self.references = {str(index): ref for index, ref in enumerate(artifact_references(payload))}
            self.query_one("#wb-artifact", Select).set_options([(f"{ref.kind} · {ref.uri}", key) for key, ref in self.references.items()])
        except Exception as exc:
            self.query_one("#wb-status", Static).update(label + "失败：" + str(exc))
        finally:
            self._set_busy(False)
            self.query_one("#wb-live", Static).update("")
            self.action_refresh()


class WorkbenchApp(App):
    TITLE = "PhysicsOS · 仿真工作台"

    def __init__(self, workspace: str | Path | None = None, **kwargs):
        super().__init__(**kwargs)
        self.workspace = workspace

    def on_mount(self):
        self.push_screen(WorkbenchScreen(self.workspace))
