from __future__ import annotations

import argparse
from datetime import UTC, datetime
import inspect
import json
import os
import shlex
import sys
import textwrap
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
from physicsos.config import load_config, runtime_paths
from physicsos.events import PhysicsOSEventRenderer, collect_physicsos_events, read_physicsos_events


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


def _physicsos_banner() -> str:
    return """
██████╗ ██╗  ██╗██╗   ██╗███████╗██╗ ██████╗███████╗
██╔══██╗██║  ██║╚██╗ ██╔╝██╔════╝██║██╔════╝██╔════╝
██████╔╝███████║ ╚████╔╝ ███████╗██║██║     ███████╗
██╔═══╝ ██╔══██║  ╚██╔╝  ╚════██║██║██║     ╚════██║
██║     ██║  ██║   ██║   ███████║██║╚██████╗███████║
╚═╝     ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝ ╚═════╝╚══════╝

 ██████╗ ███████╗
██╔═══██╗██╔════╝
██║   ██║███████╗
██║   ██║╚════██║
╚██████╔╝███████║
 ╚═════╝ ╚══════╝

                 PhysicsOS
"""


def _patch_deepagents_banner() -> None:
    def get_banner() -> str:
        return _physicsos_banner()

    try:
        import deepagents_cli.config as cli_config
        cli_config.get_banner = get_banner
        if "deepagents_cli.widgets.welcome" in sys.modules:
            import deepagents_cli.widgets.welcome as welcome
            welcome.get_banner = get_banner
    except ImportError:
        return


def _patch_deepagents_allow_blocking() -> None:
    """Force DeepAgents' local LangGraph server to allow local sync I/O."""
    try:
        import deepagents_cli.server as cli_server
    except ImportError:
        return

    original = cli_server._build_server_cmd
    if getattr(original, "_physicsos_allow_blocking", False):
        return

    def build_server_cmd(*args: object, **kwargs: object) -> list[str]:
        cmd = list(original(*args, **kwargs))
        if "--allow-blocking" not in cmd:
            cmd.append("--allow-blocking")
        return cmd

    build_server_cmd._physicsos_allow_blocking = True  # type: ignore[attr-defined]
    cli_server._build_server_cmd = build_server_cmd


def _patch_deepagents_physicsos_tools() -> None:
    """Inject scoped PhysicsOS tools into the DeepAgents CLI server graph."""
    try:
        import deepagents_cli.server_manager as server_manager
    except ImportError:
        return

    original = server_manager._scaffold_workspace
    if getattr(original, "_physicsos_tools", False):
        return

    def scaffold_workspace(work_dir: Path) -> None:
        original(work_dir)
        server_graph = work_dir / "server_graph.py"
        source = server_graph.read_text(encoding="utf-8")
        old = (
            "    from deepagents_cli.config import settings\n"
            "    from deepagents_cli.tools import fetch_url, web_search\n\n"
            "    tools: list[Any] = [fetch_url]\n"
        )
        new = (
            "    from deepagents_cli.config import settings\n"
            "    from deepagents_cli.tools import fetch_url, web_search\n\n"
            "    try:\n"
            "        from physicsos.tools.registry import MAIN_AGENT_TOOLS\n"
            "        from physicsos.events import wrap_tools_for_events\n"
            "    except Exception:\n"
            "        MAIN_AGENT_TOOLS = []\n\n"
            "        def wrap_tools_for_events(tools):\n"
            "            return tools\n\n"
            "    tools: list[Any] = [fetch_url, *wrap_tools_for_events(MAIN_AGENT_TOOLS)]\n"
        )
        if old in source and "MAIN_AGENT_TOOLS" not in source:
            source = source.replace(old, new)

        marker = "    async_subagents = load_async_subagents() or None\n\n"
        injection = (
            "    try:\n"
            "        from physicsos.tools.registry import SUBAGENT_TOOL_GROUPS\n"
            "        from physicsos.events import wrap_tools_for_events\n"
            "    except Exception:\n"
            "        SUBAGENT_TOOL_GROUPS = {}\n\n"
            "        def wrap_tools_for_events(tools):\n"
            "            return tools\n\n"
            "    import deepagents_cli.agent as cli_agent\n"
            "    original_load_subagents = cli_agent.list_subagents\n\n"
            "    def load_scoped_subagents(*args, **kwargs):\n"
            "        subagents = original_load_subagents(*args, **kwargs)\n"
            "        for subagent in subagents:\n"
            "            tools_for_agent = SUBAGENT_TOOL_GROUPS.get(subagent.get(\"name\"))\n"
            "            if tools_for_agent is not None:\n"
            "                subagent[\"tools\"] = wrap_tools_for_events(tools_for_agent)\n"
            "        return subagents\n\n"
            "    cli_agent.list_subagents = load_scoped_subagents\n\n"
        )
        if marker in source and "load_scoped_subagents" not in source:
            source = source.replace(marker, marker + injection)

        server_graph.write_text(source, encoding="utf-8")

    scaffold_workspace._physicsos_tools = True  # type: ignore[attr-defined]
    server_manager._scaffold_workspace = scaffold_workspace


def _patch_deepagents_physicsos_tui_events() -> None:
    """Render PhysicsOS custom stream events inside the DeepAgents Textual TUI."""
    try:
        import deepagents_cli.textual_adapter as textual_adapter
    except ImportError:
        return

    original = textual_adapter.execute_task_textual
    if getattr(original, "_physicsos_tui_events", False):
        return

    try:
        source = inspect.getsource(original)
    except (OSError, TypeError):
        return

    old_stream_mode = 'stream_mode=["messages", "updates"],'
    new_stream_mode = 'stream_mode=["messages", "updates", "custom"],'
    if old_stream_mode not in source:
        return

    custom_branch_marker = "                # Handle MESSAGES stream - for content and tool calls\n"
    custom_branch = (
        "                # Handle CUSTOM stream - PhysicsOS typed workflow events\n"
        "                elif current_stream_mode == \"custom\":\n"
        "                    try:\n"
        "                        from physicsos.events import PhysicsOSEventRenderer, collect_physicsos_events\n"
        "                        physicsos_events = collect_physicsos_events(data)\n"
        "                        if physicsos_events:\n"
        "                            renderer = PhysicsOSEventRenderer()\n"
        "                            for rendered_event in renderer.render_many(physicsos_events):\n"
        "                                await adapter._mount_message(AppMessage(rendered_event))\n"
        "                            if adapter._set_spinner and not adapter._current_tool_messages:\n"
        "                                await adapter._set_spinner(\"Thinking\")\n"
        "                    except Exception:\n"
        "                        logger.debug(\"Failed to render PhysicsOS custom event\", exc_info=True)\n"
        "                    continue\n\n"
    )
    if custom_branch_marker not in source:
        return

    patched_source = source.replace(old_stream_mode, new_stream_mode, 1)
    patched_source = patched_source.replace(custom_branch_marker, custom_branch + custom_branch_marker, 1)
    namespace = textual_adapter.__dict__
    exec(compile(textwrap.dedent(patched_source), "<physicsos_deepagents_tui_patch>", "exec"), namespace)
    textual_adapter.execute_task_textual._physicsos_tui_events = True  # type: ignore[attr-defined]
    textual_adapter.execute_task_textual._physicsos_stream_modes = ("messages", "updates", "custom")  # type: ignore[attr-defined]


def _patch_deepagents_physicsos_noninteractive_events() -> None:
    """Render PhysicsOS custom stream events in DeepAgents non-interactive mode."""
    try:
        import deepagents_cli.non_interactive as non_interactive
    except ImportError:
        return

    process_stream_chunk = non_interactive._process_stream_chunk
    if not getattr(process_stream_chunk, "_physicsos_noninteractive_events", False):
        original_process_stream_chunk = process_stream_chunk

        def patched_process_stream_chunk(chunk, state, console, file_op_tracker):  # type: ignore[no-untyped-def]
            if isinstance(chunk, tuple) and len(chunk) == 3:
                namespace, stream_mode, data = chunk
                if not namespace and stream_mode == "custom":
                    try:
                        from rich.text import Text
                        from physicsos.events import PhysicsOSEventRenderer, collect_physicsos_events

                        renderer = PhysicsOSEventRenderer()
                        for rendered_event in renderer.render_many(collect_physicsos_events(data)):
                            if state.spinner:
                                state.spinner.stop()
                            console.print(Text(rendered_event, style="dim"), highlight=False)
                        if state.spinner:
                            state.spinner.start()
                    except Exception:
                        non_interactive.logger.debug(
                            "Failed to render PhysicsOS non-interactive custom event",
                            exc_info=True,
                        )
                    return
            return original_process_stream_chunk(chunk, state, console, file_op_tracker)

        patched_process_stream_chunk._physicsos_noninteractive_events = True  # type: ignore[attr-defined]
        non_interactive._process_stream_chunk = patched_process_stream_chunk

    stream_agent = non_interactive._stream_agent
    if getattr(stream_agent, "_physicsos_noninteractive_stream_modes", False):
        return
    try:
        source = inspect.getsource(stream_agent)
    except (OSError, TypeError):
        return

    old_stream_mode = 'stream_mode=["messages", "updates"],'
    new_stream_mode = 'stream_mode=["messages", "updates", "custom"],'
    if old_stream_mode not in source:
        return
    patched_source = source.replace(old_stream_mode, new_stream_mode, 1)
    namespace = non_interactive.__dict__
    exec(compile(textwrap.dedent(patched_source), "<physicsos_deepagents_noninteractive_patch>", "exec"), namespace)
    non_interactive._stream_agent._physicsos_noninteractive_stream_modes = ("messages", "updates", "custom")  # type: ignore[attr-defined]


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
    config = load_config()
    model = os.getenv("PHYSICSOS_OPENAI_MODEL") or config.get("model", {}).get("name") or "gpt-5.4"
    return ["--model", f"openai:{model}"]


def _deepagents_model_params_args(argv: list[str]) -> list[str]:
    if "--model-params" in argv:
        return []
    config = load_config()
    base_url = os.getenv("PHYSICSOS_OPENAI_BASE_URL") or config.get("model", {}).get("base_url")
    if not base_url:
        return []
    return ["--model-params", json.dumps({"base_url": base_url})]


def _prepare_deepagents_env() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    # DeepAgents CLI starts a local langgraph dev server. Its local filesystem
    # and shell tools perform synchronous I/O, which LangGraph otherwise rejects
    # as BlockingError when the agent writes or edits files.
    os.environ.setdefault("LANGGRAPH_ALLOW_BLOCKING", "true")
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except (OSError, ValueError):
                pass

    config = load_config()
    api_key = os.getenv("PHYSICSOS_OPENAI_API_KEY") or config.get("model", {}).get("api_key")
    if api_key and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = str(api_key)


def _launch_deepagents_cli(argv: list[str]) -> int:
    if not any(arg in {"-h", "--help", "-v", "--version"} for arg in argv):
        _ensure_deepagents_physicsos_config()
    _prepare_deepagents_env()
    _patch_deepagents_banner()
    _patch_deepagents_allow_blocking()
    _patch_deepagents_physicsos_tools()
    _patch_deepagents_physicsos_tui_events()
    _patch_deepagents_physicsos_noninteractive_events()
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
        "config_json": str(paths.config_json),
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


def _render_physicsos_events(result: object, *, session_path: Path | None = None) -> str | None:
    events = collect_physicsos_events(result)
    if not events and session_path is not None:
        events = read_physicsos_events(session_path)
    if not events:
        return None
    renderer = PhysicsOSEventRenderer()
    return "\n".join(renderer.render_many(events[-12:]))


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
            event_text = _render_physicsos_events(result, session_path=session_path)
            if event_text:
                print(event_text)
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
