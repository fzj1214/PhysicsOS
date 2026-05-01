from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
import shlex
import sys
from pathlib import Path

from physicsos.agents.prompts import (
    GEOMETRY_MESH_AGENT_PROMPT,
    KNOWLEDGE_AGENT_PROMPT,
    PHYSICSOS_SYSTEM_PROMPT,
    POSTPROCESS_AGENT_PROMPT,
    SOLVER_AGENT_PROMPT,
    TAPS_AGENT_PROMPT,
    VERIFICATION_AGENT_PROMPT,
)
from physicsos.cloud.auth import start_device_login
from physicsos.cloud.foamvm_client import FoamVMClient
from physicsos.agents.main import create_physicsos_agent
from physicsos.agents.openai_compatible import create_openai_compatible_model
from physicsos.config import runtime_paths


BANNER = "PhysicsOS\nPhysicsOS"

LOCAL_COMMANDS = {"auth", "account", "paths", "runner", "legacy-repl"}

SUBAGENT_PROMPTS = {
    "geometry-mesh-agent": (
        "Build GeometrySpec and MeshSpec from geometry, CAD, mesh, text, and boundary labeling inputs.",
        GEOMETRY_MESH_AGENT_PROMPT,
    ),
    "taps-agent": (
        "Primary TAPS compiler and solver agent for equation-driven physics simulation.",
        TAPS_AGENT_PROMPT,
    ),
    "solver-agent": (
        "Route surrogate, full-solver, and hybrid fallback backends after TAPS-first planning.",
        SOLVER_AGENT_PROMPT,
    ),
    "verification-agent": (
        "Check residuals, conservation, uncertainty, mesh quality, OOD risk, and trustworthiness.",
        VERIFICATION_AGENT_PROMPT,
    ),
    "postprocess-agent": (
        "Extract KPIs, generate visualizations, and write simulation reports.",
        POSTPROCESS_AGENT_PROMPT,
    ),
    "knowledge-agent": (
        "Retrieve scientific computing, PDE, solver, materials, and TAPS knowledge.",
        KNOWLEDGE_AGENT_PROMPT,
    ),
}


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _physicsos_agent_prompt() -> str:
    return (
        "# PhysicsOS\n\n"
        + PHYSICSOS_SYSTEM_PROMPT
        + "\n\n"
        "You are running inside the official DeepAgents CLI/TUI as the PhysicsOS agent.\n"
        "Use the built-in DeepAgents todo, filesystem, shell, subagent, MCP, and skills capabilities.\n"
        "For local PhysicsOS package state, use `physicsos paths`.\n"
        "For PhysicsOS Cloud device login, use `physicsos auth login`.\n"
        "For cloud runner jobs, use `physicsos runner ...` commands.\n"
        "Prefer TAPS-first reasoning and delegate to the registered PhysicsOS subagents when useful.\n"
        "Do not claim a high-trust physics solve unless residual, conservation, and verification evidence is available.\n"
    )


def _ensure_deepagents_physicsos_config() -> None:
    agent_dir = Path.home() / ".deepagents" / "physicsos"
    agents_dir = agent_dir / "agents"
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "AGENTS.md").write_text(_physicsos_agent_prompt(), encoding="utf-8")
    for name, (description, prompt) in SUBAGENT_PROMPTS.items():
        subagent_dir = agents_dir / name
        subagent_dir.mkdir(parents=True, exist_ok=True)
        content = f"---\nname: {name}\ndescription: {description}\n---\n\n{prompt}\n"
        (subagent_dir / "AGENTS.md").write_text(content, encoding="utf-8")


def _deepagents_model_args(argv: list[str]) -> list[str]:
    if any(arg in {"-M", "--model", "--default-model", "--clear-default-model"} for arg in argv):
        return []
    model = os.getenv("PHYSICSOS_OPENAI_MODEL", "gpt-5.4")
    return ["--model", f"openai:{model}"]


def _deepagents_model_params_args(argv: list[str]) -> list[str]:
    if "--model-params" in argv:
        return []
    base_url = os.getenv("PHYSICSOS_OPENAI_BASE_URL")
    if not base_url:
        return []
    return ["--model-params", json.dumps({"base_url": base_url})]


def _prepare_deepagents_env() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except (OSError, ValueError):
                pass

    if os.getenv("PHYSICSOS_OPENAI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["PHYSICSOS_OPENAI_API_KEY"]


def _launch_deepagents_cli(argv: list[str]) -> int:
    if not any(arg in {"-h", "--help", "-v", "--version"} for arg in argv):
        _ensure_deepagents_physicsos_config()
    _prepare_deepagents_env()
    try:
        from deepagents_cli import cli_main
    except ImportError as exc:
        raise RuntimeError("deepagents-cli is required. Reinstall with `pip install -U physicsos`.") from exc

    forwarded = list(argv)
    if not any(arg in {"-a", "--agent"} for arg in forwarded):
        forwarded = ["--agent", "physicsos", *forwarded]
    forwarded = [*_deepagents_model_args(forwarded), *_deepagents_model_params_args(forwarded), *forwarded]
    previous_argv = sys.argv
    sys.argv = ["deepagents", *forwarded]
    try:
        cli_main()
    finally:
        sys.argv = previous_argv
    return 0


def _rich_console():
    try:
        from rich.console import Console
    except ImportError:
        return None
    return Console()


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8") if not path.exists() else None
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _append_session_event(path: Path, event: str, payload: dict[str, object] | None = None) -> None:
    item = {"event": event, "timestamp": datetime.now(UTC).isoformat()}
    if payload:
        item.update(payload)
    _append_jsonl(path, item)


def _new_session_path() -> Path:
    paths = runtime_paths()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return paths.sessions / f"session-{stamp}.jsonl"


def _paths_payload() -> dict[str, str]:
    paths = runtime_paths()
    return {
        "home": str(paths.home),
        "workspace": str(paths.workspace),
        "cloud_config": str(paths.cloud_config),
        "sessions": str(paths.sessions),
        "history": str(paths.history),
        "scratch": str(paths.scratch),
        "case_memory": str(paths.case_memory),
        "knowledge_base": str(paths.knowledge_base),
    }


def _print_welcome() -> None:
    paths = runtime_paths()
    console = _rich_console()
    if console is not None:
        from rich import box
        from rich.align import Align
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        title = Text()
        title.append("PhysicsOS\n", style="bold cyan")
        title.append("PhysicsOS", style="bold white")
        console.print(
            Panel(
                Align.center(title),
                subtitle="TAPS-first physics simulation agent",
                border_style="cyan",
                box=box.DOUBLE,
            )
        )
        console.print(f"[dim]Home:[/]      {paths.home}")
        console.print(f"[dim]Workspace:[/] {paths.workspace}")
        table = Table(title="Commands", box=box.SIMPLE_HEAVY, show_lines=False)
        table.add_column("Input", style="cyan", no_wrap=True)
        table.add_column("Action")
        table.add_row("<natural language>", "Chat with the PhysicsOS DeepAgents orchestrator")
        table.add_row("/help", "Show commands")
        table.add_row("/paths", "Show runtime storage paths")
        table.add_row("/login", "Device-code login to PhysicsOS Cloud")
        table.add_row("/account", "Show cloud account")
        table.add_row("/submit <manifest.json>", "Submit runner manifest")
        table.add_row("/status <job_id>", "Show cloud job status")
        table.add_row("/logs <job_id>", "Show cloud job logs")
        table.add_row("/artifacts <job_id>", "Show cloud job artifacts")
        table.add_row("/exit", "Quit")
        console.print(table)
        return

    print(BANNER)
    print("TAPS-first physics simulation agent")
    print(f"Home:      {paths.home}")
    print(f"Workspace: {paths.workspace}")
    print()
    print("Commands:")
    print("  <natural language>           Chat with the PhysicsOS DeepAgents orchestrator")
    print("  /help                        Show commands")
    print("  /paths                       Show runtime storage paths")
    print("  /login                       Device-code login to PhysicsOS Cloud")
    print("  /account                     Show cloud account")
    print("  /submit <manifest.json>      Submit runner manifest")
    print("  /status <job_id>             Show cloud job status")
    print("  /logs <job_id>               Show cloud job logs")
    print("  /artifacts <job_id>          Show cloud job artifacts")
    print("  /exit                        Quit")
    print()


def _extract_agent_text(result: object) -> str:
    if isinstance(result, dict):
        interrupts = result.get("__interrupt__")
        if interrupts:
            return f"[approval required] {interrupts}"
        messages = result.get("messages")
        if isinstance(messages, list) and messages:
            for message in reversed(messages):
                content = getattr(message, "content", None)
                if content is None and isinstance(message, dict):
                    content = message.get("content")
                if content:
                    return str(content)
        if "output" in result:
            return str(result["output"])
    return str(result)


def _create_agent() -> object:
    model = create_openai_compatible_model()
    return create_physicsos_agent(model=model)


def _run_local_command(command: str, parts: list[str]) -> bool:
    if command == "paths":
        _print_json(_paths_payload())
    elif command == "login":
        result = start_device_login(open_browser=True)
        _print_json({"runner_url": result.runner_url, "user_code": result.user_code, "status": "logged_in"})
    elif command == "account":
        _print_json(FoamVMClient.from_config().me())
    elif command == "submit" and len(parts) >= 2:
        manifest = json.loads(Path(parts[1]).read_text(encoding="utf-8"))
        _print_json(FoamVMClient.from_config().submit_job(manifest))
    elif command == "status" and len(parts) >= 2:
        _print_json(FoamVMClient.from_config().job_status(parts[1]))
    elif command == "logs" and len(parts) >= 2:
        _print_json(FoamVMClient.from_config().job_events(parts[1]))
    elif command == "artifacts" and len(parts) >= 2:
        _print_json(FoamVMClient.from_config().job_artifacts(parts[1]))
    else:
        return False
    return True


def _interactive(agent: object | None = None) -> int:
    paths = runtime_paths()
    session_path = _new_session_path()
    messages: list[dict[str, str]] = []
    _print_welcome()
    _append_session_event(session_path, "session_start", {"workspace": str(paths.workspace)})
    while True:
        try:
            raw = input("physicsos> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not raw:
            continue
        entry = {"event": "input", "timestamp": datetime.now(UTC).isoformat(), "content": raw}
        _append_jsonl(paths.history, entry)
        _append_jsonl(session_path, entry)
        try:
            command_raw = raw[1:] if raw.startswith("/") else raw
            parts = shlex.split(command_raw)
        except ValueError as exc:
            print(f"error: {exc}")
            continue
        command = parts[0].lower()
        if command in {"exit", "quit", "q"}:
            return 0
        if command in {"help", "?"}:
            _print_welcome()
            continue
        try:
            if raw.startswith("/") or command in {"paths", "login", "account", "submit", "status", "logs", "artifacts"}:
                if not _run_local_command(command, parts):
                    print("Unknown or incomplete command. Type `/help`.")
                continue

            if agent is None:
                agent = _create_agent()
            messages.append({"role": "user", "content": raw})
            result = agent.invoke({"messages": messages})
            text = _extract_agent_text(result)
            print(text)
            messages.append({"role": "assistant", "content": text})
            _append_session_event(session_path, "assistant", {"content": text})
        except Exception as exc:  # pragma: no cover - interactive guard
            print(f"error: {exc}")
            _append_session_event(session_path, "error", {"message": str(exc)})


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv or argv[0] not in LOCAL_COMMANDS:
        return _launch_deepagents_cli(argv)

    parser = argparse.ArgumentParser(prog="physicsos")
    sub = parser.add_subparsers(dest="command")

    auth = sub.add_parser("auth")
    auth_sub = auth.add_subparsers(dest="auth_command", required=True)
    login = auth_sub.add_parser("login")
    login.add_argument("--runner-url", default="https://foamvm.vercel.app")
    login.add_argument("--no-browser", action="store_true")

    sub.add_parser("account")
    sub.add_parser("paths")
    sub.add_parser("legacy-repl")

    runner = sub.add_parser("runner")
    runner_sub = runner.add_subparsers(dest="runner_command", required=True)
    submit = runner_sub.add_parser("submit")
    submit.add_argument("manifest")
    status = runner_sub.add_parser("status")
    status.add_argument("job_id")
    logs = runner_sub.add_parser("logs")
    logs.add_argument("job_id")
    logs.add_argument("--after", type=int)
    artifacts = runner_sub.add_parser("artifacts")
    artifacts.add_argument("job_id")

    args = parser.parse_args(argv)

    if args.command == "legacy-repl":
        return _interactive()

    if args.command == "auth" and args.auth_command == "login":
        result = start_device_login(args.runner_url, open_browser=not args.no_browser)
        _print_json({"runner_url": result.runner_url, "user_code": result.user_code, "status": "logged_in"})
        return 0

    if args.command == "account":
        _print_json(FoamVMClient.from_config().me())
        return 0

    if args.command == "paths":
        _print_json(_paths_payload())
        return 0

    if args.command == "runner":
        client = FoamVMClient.from_config()
        if args.runner_command == "submit":
            manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
            _print_json(client.submit_job(manifest))
            return 0
        if args.runner_command == "status":
            _print_json(client.job_status(args.job_id))
            return 0
        if args.runner_command == "logs":
            _print_json(client.job_events(args.job_id, after=args.after))
            return 0
        if args.runner_command == "artifacts":
            _print_json(client.job_artifacts(args.job_id))
            return 0

    parser.error("Unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
