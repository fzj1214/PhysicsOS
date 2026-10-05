"""Read the shared runtime artifacts for the TUI without starting an agent."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from typing import Any

from physicsos.paths import resolve_workspace_path
from physicsos.runtime.artifacts import artifact, checked_path
from physicsos.schemas.common import ArtifactRef


@dataclass(frozen=True)
class WorkbenchRecord:
    key: str
    category: str
    title: str
    status: str
    path: Path
    payload: dict[str, Any]
    reference: ArtifactRef


KINDS = {"strategy": "rsi_strategy", "suite": "rsi_suite", "provider": "rsi_revision_provider",
         "campaign": "rsi_campaign", "evaluation": "rsi_evaluation", "transition": "rsi_transition",
         "domain": "prepared_domain", "run": "case_run", "verification": "runtime_verification",
         "convergence": "runtime_convergence"}


def read_json(path: Path) -> dict:
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("记录过大，请直接查看产物文件。")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("记录不是 JSON 对象。")
    return value


def records(workspace: Path, limit: int = 300) -> tuple[list[WorkbenchRecord], list[str]]:
    patterns = {
        "domain": "cases/*/domains/*/manifest.json", "run": "cases/*/runs/*/manifest.json",
        "verification": "cases/*/runs/*/verification/*/report.json", "convergence": "cases/*/studies/*/report.json",
        "strategy": "data/rsi/strategies/*/manifest.json", "suite": "data/rsi/suites/*/manifest.json",
        "provider": "data/rsi/revision_providers/*/manifest.json", "campaign": "data/rsi/campaigns/*/report.json",
        "evaluation": "data/rsi/evaluations/*/report.json", "transition": "data/rsi/transitions/*/*.json",
    }
    result, warnings = [], []
    for category, pattern in patterns.items():
        paths = sorted(workspace.glob(pattern), key=lambda path: path.stat().st_mtime_ns, reverse=True)[:limit]
        for path in paths:
            try:
                payload = read_json(path)
                spec = payload.get("spec", {})
                status = str(payload.get("status") or payload.get("overall_status") or payload.get("result", {}).get("status") or "registered")
                case_name = path.relative_to(workspace).parts[1] if category in {"domain", "run", "verification", "convergence"} else None
                title = str(payload.get("case_id") or payload.get("problem_id") or case_name or spec.get("name") or payload.get("name") or path.parent.name)
                if category in {"campaign", "evaluation", "transition"}:
                    title = path.parent.name[:12]
                reference = artifact(path, KINDS[category], workspace)
                result.append(WorkbenchRecord(key=path.relative_to(workspace).as_posix(), category=category,
                                              title=title, status=status, path=path, payload=payload, reference=reference))
            except (ValueError, OSError) as exc:
                warnings.append(f"{path.relative_to(workspace)}: {exc}")
    return result, warnings


def active_strategies(workspace: Path) -> tuple[dict[str, dict], list[str]]:
    path = workspace / "data" / "rsi" / "state.sqlite3"
    if not path.is_file():
        return {}, []
    result = {}
    try:
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=2) as connection:
            for scope, generation, strategy, activation in connection.execute("SELECT scope,generation,strategy,activation_id FROM active"):
                if strategy:
                    reference = ArtifactRef.model_validate_json(strategy)
                    try:
                        checked_path(reference, workspace)
                        integrity = "完整"
                    except (OSError, ValueError):
                        integrity = "证据已变化"
                    result[reference.uri] = {"generation": generation, "activation_id": activation,
                                             "scope_key": scope, "integrity": integrity}
        return result, []
    except (sqlite3.Error, ValueError) as exc:
        return {}, ["策略登记表读取失败：" + str(exc)]


def artifact_references(payload: Any) -> list[ArtifactRef]:
    found = {}
    def visit(value):
        if isinstance(value, dict):
            if isinstance(value.get("uri"), str) and isinstance(value.get("kind"), str):
                try:
                    ref = ArtifactRef.model_validate({key: item for key, item in value.items() if key in ArtifactRef.model_fields})
                    found[(ref.uri, ref.checksum)] = ref
                except ValueError:
                    pass
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(payload)
    return list(found.values())


def inspect_record(record: WorkbenchRecord, workspace: Path) -> str:
    payload = read_json(record.path)
    lines = [f"记录：{record.category} · {record.title}", f"状态：{record.status}",
             f"文件：{record.path}"]
    if record.category == "domain":
        quality = payload.get("mesh", {}).get("quality", {}) if payload.get("mesh") else {}
        lines.extend([f"离散表示：{payload.get('requirements', {}).get('representation')}",
                      f"已检查单元：{quality.get('checked_elements', '—')} · 最小 Jacobian：{quality.get('min_jacobian', '—')}",
                      f"质量 / 语义检查：{json.dumps(payload.get('checks', {}), ensure_ascii=False)}"])
    if record.category == "verification":
        lines.append(f"通过 / 失败 / 不确定：{payload.get('passed_checks', 0)} / {payload.get('failed_checks', 0)} / {payload.get('uncertain_checks', 0)}")
        for name, check in payload.get("individual_results", {}).items():
            lines.append(f"{name}: {check.get('message', '')} · {json.dumps(check.get('metrics', {}), ensure_ascii=False)}")
    if record.category == "convergence":
        lines.append(f"观测收敛阶：{payload.get('observed_order')} · 预期：{payload.get('expected_order')}")
        lines.extend(f"细化 {row.get('level')}: h={row.get('h')} · 误差={row.get('error', '未提供') }" for row in payload.get("rows", []))
    if record.category == "campaign":
        lines.extend([f"修订 provider 调用：{payload.get('provider_calls', 0)} · 求解预算占用：{payload.get('reserved_kernel_runs', 0)}", f"停止原因：{payload.get('stop_reason', '')}"])
    for key in ("required_actions", "errors", "reasons"):
        lines.extend(str(value) for value in payload.get(key, []))
    lines.extend(["", "关联证据"])
    for ref in artifact_references(payload)[:200]:
        try:
            path = checked_path(ref, workspace)
            if not path.is_relative_to(workspace):
                raise ValueError("产物位于当前工作区之外")
            state = "完整"
        except (ValueError, OSError) as exc:
            state = "待检查：" + str(exc)
        lines.append(f"{state} | {ref.kind} | {ref.uri}")
    lines.extend(["", "记录内容", json.dumps(payload, indent=2, ensure_ascii=False)])
    return "\n".join(lines)


def open_artifact(reference: ArtifactRef, workspace: Path) -> tuple[str, str]:
    path = checked_path(reference, workspace)
    if not path.is_relative_to(workspace):
        raise ValueError("请选择当前工作区内的产物。")
    if path.suffix.lower() not in {".json", ".jsonl", ".md", ".txt", ".py", ".log"}:
        return str(path), "此产物为网格或数组文件。可通过显示的路径使用相应查看器打开。"
    with path.open(encoding="utf-8", errors="replace") as stream:
        text = stream.read(256_000)
    if path.stat().st_size > 256_000:
        text += "\n\n…仅显示前 256 KB。"
    return str(path), text
