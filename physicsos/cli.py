from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import inspect
import json
import os
import shlex
import sys
import textwrap
from pathlib import Path

from physicsos.agents.prompts import PHYSICSOS_SYSTEM_PROMPT
from physicsos.agents.subagents import SUBAGENTS
from physicsos.cloud.auth import start_device_login
from physicsos.cloud.foamvm_client import FoamVMClient
from physicsos.agents.main import create_physicsos_agent
from physicsos.agents.openai_compatible import create_openai_compatible_model
from physicsos.config import config_path, load_config, runtime_paths, save_config
from physicsos.events import PhysicsOSEventRenderer, collect_physicsos_events, read_physicsos_events
from physicsos.schemas.common import ArtifactRef
from physicsos.schemas.geometry import GeometrySpec
from physicsos.tools.geometry_tools import ApplyBoundaryLabelingArtifactInput, apply_boundary_labeling_artifact
from physicsos.tools.pseudopotential_tools import (
    IndexVaspPawPbeLibraryInput,
    SelectPseudopotentialsForStructureInput,
    index_vasp_paw_pbe_library,
    select_pseudopotentials_for_structure,
)


BANNER = "PhysicsOS\nPhysicsOS"

LOCAL_COMMANDS = {"auth", "account", "paths", "runner", "geometry", "pseudopotentials", "pp", "legacy-repl"}

def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _quote_shell_path(path: Path) -> str:
    value = str(path)
    if os.name == "nt":
        return '"' + value.replace('"', '\\"') + '"'
    return shlex.quote(value)


def _translate_workspace_paths_for_shell(command: str, workspace: str | Path) -> str:
    """Translate agent-facing /workspace paths inside shell commands."""
    root = Path(workspace).resolve()
    marker = "/workspace"
    result: list[str] = []
    index = 0
    length = len(command)
    while index < length:
        found = command.find(marker, index)
        if found == -1:
            result.append(command[index:])
            break
        before = command[found - 1] if found > 0 else ""
        after_index = found + len(marker)
        after = command[after_index] if after_index < length else ""
        if before not in {"", " ", "\t", "\n", "\r", "\"", "'", "=", "(", "[", "{"} or after not in {"", "/", "\\", " ", "\t", "\n", "\r", "\"", "'", ")", "]", "}"}:
            result.append(command[index : after_index])
            index = after_index
            continue

        end = after_index
        while end < length and command[end] not in " \t\r\n\"'`|&;<>()[]{}":
            end += 1
        raw_path = command[found:end]
        suffix = raw_path.removeprefix(marker).lstrip("/\\")
        native_path = root / Path(*suffix.replace("\\", "/").split("/")) if suffix else root
        result.append(command[index:found])
        result.append(_quote_shell_path(native_path))
        index = end
    return "".join(result)


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


def _patch_deepagents_workspace_paths() -> None:
    """Use /workspace virtual paths for local DeepAgents filesystem tools."""
    try:
        import deepagents_cli.agent as cli_agent
    except ImportError:
        return

    try:
        from deepagents.backends import LocalShellBackend
    except ImportError:
        LocalShellBackend = None  # type: ignore[assignment]

    if LocalShellBackend is not None and not getattr(LocalShellBackend.execute, "_physicsos_utf8_shell", False):
        try:
            source = inspect.getsource(LocalShellBackend.execute)
        except (OSError, TypeError):
            source = ""
        old = (
            "                text=True,\n"
            "                timeout=effective_timeout,\n"
        )
        new = (
            "                text=True,\n"
            "                encoding=\"utf-8\",\n"
            "                errors=\"replace\",\n"
            "                timeout=effective_timeout,\n"
        )
        if old in source:
            namespace = LocalShellBackend.execute.__globals__
            exec(compile(textwrap.dedent(source.replace(old, new, 1)), "<physicsos_deepagents_shell_utf8_patch>", "exec"), namespace)
            namespace["execute"]._physicsos_utf8_shell = True  # type: ignore[attr-defined]
            LocalShellBackend.execute = namespace["execute"]

    if LocalShellBackend is not None and not getattr(LocalShellBackend.execute, "_physicsos_workspace_shell_paths", False):
        original_execute = LocalShellBackend.execute

        def execute_with_workspace_shell_paths(self, command: str, *, timeout: int | None = None):  # type: ignore[no-untyped-def]
            workspace = os.environ.get("PHYSICSOS_WORKSPACE") or str(getattr(self, "cwd", runtime_paths().workspace))
            translated = _translate_workspace_paths_for_shell(command, workspace)
            return original_execute(self, translated, timeout=timeout)

        execute_with_workspace_shell_paths._physicsos_workspace_shell_paths = True  # type: ignore[attr-defined]
        LocalShellBackend.execute = execute_with_workspace_shell_paths

    original_get_system_prompt = cli_agent.get_system_prompt
    if not getattr(original_get_system_prompt, "_physicsos_workspace_paths", False):

        def get_system_prompt_with_workspace_paths(*args: object, **kwargs: object) -> str:
            try:
                prompt = original_get_system_prompt(*args, **kwargs)
            except UnicodeDecodeError:
                prompt = "## Additional Guidelines\n\nFollow the active PhysicsOS system instructions."
            replacement = (
                "### Current Working Directory\n\n"
                "PhysicsOS uses one workspace path system. The real workspace root is available as `PHYSICSOS_WORKSPACE`; "
                "the agent-visible alias for the same directory is `/workspace`.\n\n"
                "### File System and Paths\n\n"
                "**IMPORTANT - Path Handling:**\n"
                "- Filesystem tools should use `/workspace/...`, for example `/workspace/scratch/result.png`\n"
                "- Shell and Python run from `PHYSICSOS_WORKSPACE`; use cwd-relative paths like `scratch/result.png` or build paths from `os.environ['PHYSICSOS_WORKSPACE']`\n"
                "- `/workspace/scratch/result.png`, `scratch/result.png`, and the native path under `PHYSICSOS_WORKSPACE` refer to the same file\n"
                "- Avoid Windows drive paths in prompts unless you are printing diagnostics; use `/workspace/...` when showing paths to the agent\n\n"
            )
            start = prompt.find("### Current Working Directory")
            end = prompt.find("## Additional Guidelines", start)
            if start != -1 and end != -1:
                return prompt[:start] + replacement + prompt[end:]
            return prompt + "\n\n" + replacement

        get_system_prompt_with_workspace_paths._physicsos_workspace_paths = True  # type: ignore[attr-defined]
        cli_agent.get_system_prompt = get_system_prompt_with_workspace_paths

    original_create_cli_agent = cli_agent.create_cli_agent
    if getattr(original_create_cli_agent, "_physicsos_workspace_paths", False):
        return

    def create_cli_agent_with_workspace_paths(*args: object, **kwargs: object):
        result = original_create_cli_agent(*args, **kwargs)
        agent, backend = result
        try:
            workspace = Path(os.environ.get("PHYSICSOS_WORKSPACE") or runtime_paths().workspace)
            os.environ["PHYSICSOS_WORKSPACE"] = str(workspace)
            os.environ["PHYSICSOS_CWD"] = str(workspace)
            default_backend = backend.default
            if hasattr(default_backend, "cwd"):
                default_backend.cwd = workspace.resolve()
            if hasattr(default_backend, "root_dir"):
                default_backend.root_dir = workspace.resolve()
            default_backend.virtual_mode = True
            if "/workspace/" not in backend.routes:
                backend.routes = {"/workspace/": default_backend, **backend.routes}
                backend.sorted_routes = sorted(backend.routes.items(), key=lambda item: len(item[0]), reverse=True)
        except Exception:
            return result
        return agent, backend

    create_cli_agent_with_workspace_paths._physicsos_workspace_paths = True  # type: ignore[attr-defined]
    cli_agent.create_cli_agent = create_cli_agent_with_workspace_paths


def _physicsos_cli_system_prompt(existing: str | None) -> str:
    prefix = _physicsos_agent_prompt()
    if existing and PHYSICSOS_SYSTEM_PROMPT not in existing:
        return prefix + "\n\n# DeepAgents CLI Base Instructions\n\n" + existing
    return existing or prefix


def _patch_deepagents_physicsos_agent_config() -> None:
    """Attach PhysicsOS prompt and runtime subagents to DeepAgents CLI agents."""
    try:
        import deepagents_cli.agent as cli_agent
    except ImportError:
        return

    original_create_deep_agent = cli_agent.create_deep_agent
    if not getattr(original_create_deep_agent, "_physicsos_subagents", False):

        def create_deep_agent_with_physicsos_subagents(*args: object, **kwargs: object):
            existing = list(kwargs.get("subagents") or [])
            names = {item.get("name") for item in existing if isinstance(item, dict)}
            physicsos_subagents = [item for item in SUBAGENTS if item.get("name") not in names]
            kwargs["subagents"] = [*existing, *physicsos_subagents] or None
            return original_create_deep_agent(*args, **kwargs)

        create_deep_agent_with_physicsos_subagents._physicsos_subagents = True  # type: ignore[attr-defined]
        cli_agent.create_deep_agent = create_deep_agent_with_physicsos_subagents

    original = cli_agent.create_cli_agent
    if getattr(original, "_physicsos_agent_config", False):
        return

    def create_cli_agent_with_physicsos_config(*args: object, **kwargs: object):
        kwargs["system_prompt"] = _physicsos_cli_system_prompt(kwargs.get("system_prompt"))  # type: ignore[arg-type]
        return original(*args, **kwargs)

    create_cli_agent_with_physicsos_config._physicsos_agent_config = True  # type: ignore[attr-defined]
    cli_agent.create_cli_agent = create_cli_agent_with_physicsos_config


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
        path_patch_marker = "from deepagents_cli.project_utils import ProjectContext, get_server_project_context\n"
        path_patch = (
            "from deepagents_cli.project_utils import ProjectContext, get_server_project_context\n"
            "\n"
            "try:\n"
            "    from physicsos.cli import _patch_deepagents_workspace_paths\n"
            "    _patch_deepagents_workspace_paths()\n"
            "    from physicsos.cli import _patch_deepagents_physicsos_agent_config\n"
            "    _patch_deepagents_physicsos_agent_config()\n"
            "except Exception:\n"
            "    pass\n"
        )
        if path_patch_marker in source and "_patch_deepagents_workspace_paths" not in source:
            source = source.replace(path_patch_marker, path_patch, 1)

        old = (
            "    from deepagents_cli.config import settings\n"
            "    from deepagents_cli.tools import fetch_url, web_search\n\n"
            "    tools: list[Any] = [fetch_url]\n"
        )
        new = (
            "    from deepagents_cli.config import settings\n"
            "    from deepagents_cli.tools import fetch_url, web_search\n\n"
            "    try:\n"
            "        from physicsos.tools.registry import DEEPAGENTS_MAIN_BRIDGE_TOOLS\n"
            "        from physicsos.events import wrap_tools_for_events\n"
            "    except Exception:\n"
            "        DEEPAGENTS_MAIN_BRIDGE_TOOLS = []\n\n"
            "        def wrap_tools_for_events(tools):\n"
            "            return tools\n\n"
            "    tools: list[Any] = [fetch_url, *wrap_tools_for_events(DEEPAGENTS_MAIN_BRIDGE_TOOLS)]\n"
        )
        if old in source and "DEEPAGENTS_MAIN_BRIDGE_TOOLS" not in source:
            source = source.replace(old, new)

        server_graph.write_text(source, encoding="utf-8")

    scaffold_workspace._physicsos_tools = True  # type: ignore[attr-defined]
    server_manager._scaffold_workspace = scaffold_workspace


def _is_retryable_agent_error(exc: BaseException) -> bool:
    name = exc.__class__.__name__
    module = exc.__class__.__module__
    text = f"{name}: {exc}".lower()
    if name in {"APIConnectionError", "APIStatusError", "InternalServerError", "RateLimitError"}:
        return True
    if "openai" in module and any(token in text for token in ("connection", "internal error", "rate limit", "timeout")):
        return True
    return any(
        token in text
        for token in (
            "apiconnectionerror",
            "api connection",
            "internal error occurred",
            "connection error",
            "connection reset",
            "read timed out",
            "temporarily unavailable",
            "rate limit",
            "too many requests",
            "503",
            "502",
            "500",
        )
    )


async def _retry_async_agent_call(call, *, max_attempts: int = 5, base_delay_seconds: float = 1.0):  # type: ignore[no-untyped-def]
    attempt = 1
    while True:
        try:
            return await call()
        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception as exc:
            if attempt >= max_attempts or not _is_retryable_agent_error(exc):
                raise
            await asyncio.sleep(base_delay_seconds * (2 ** (attempt - 1)))
            attempt += 1


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
        "                # Handle CUSTOM stream - PhysicsOS events\n"
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
    patched_execute_task_textual = textual_adapter.execute_task_textual

    async def execute_task_textual_with_retry(*args: object, **kwargs: object):
        return await _retry_async_agent_call(lambda: patched_execute_task_textual(*args, **kwargs))

    execute_task_textual_with_retry._physicsos_tui_events = True  # type: ignore[attr-defined]
    execute_task_textual_with_retry._physicsos_stream_modes = ("messages", "updates", "custom")  # type: ignore[attr-defined]
    execute_task_textual_with_retry._physicsos_retry_attempts = 5  # type: ignore[attr-defined]
    textual_adapter.execute_task_textual = execute_task_textual_with_retry


def _patch_deepagents_physicsos_noninteractive_events() -> None:
    """Render PhysicsOS custom stream events in DeepAgents non-interactive mode."""
    try:
        import deepagents_cli.non_interactive as non_interactive
    except ImportError:
        return

    write_text = non_interactive._write_text
    if not getattr(write_text, "_physicsos_safe_stdout", False):

        def write_text_safe(text: str) -> None:
            try:
                sys.stdout.write(text)
                sys.stdout.flush()
            except OSError:
                # Some Windows non-interactive shells expose a stdout handle
                # that Rich/DeepAgents can write to but fail to flush. Do not
                # abort a completed PhysicsOS response for a console flush bug.
                try:
                    non_interactive.logger.debug("Ignored non-interactive stdout flush failure", exc_info=True)
                except OSError:
                    pass

        write_text_safe._physicsos_safe_stdout = True  # type: ignore[attr-defined]
        non_interactive._write_text = write_text_safe

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
    if not getattr(stream_agent, "_physicsos_noninteractive_stream_modes", False):
        try:
            source = inspect.getsource(stream_agent)
        except (OSError, TypeError):
            source = ""

        old_stream_mode = 'stream_mode=["messages", "updates"],'
        new_stream_mode = 'stream_mode=["messages", "updates", "custom"],'
        if old_stream_mode in source:
            patched_source = source.replace(old_stream_mode, new_stream_mode, 1)
            namespace = non_interactive.__dict__
            exec(compile(textwrap.dedent(patched_source), "<physicsos_deepagents_noninteractive_patch>", "exec"), namespace)

    patched_stream_agent = non_interactive._stream_agent
    if getattr(patched_stream_agent, "_physicsos_retry_attempts", None) == 5:
        return

    async def stream_agent_with_retry(*args: object, **kwargs: object):
        return await _retry_async_agent_call(lambda: patched_stream_agent(*args, **kwargs))

    stream_agent_with_retry._physicsos_noninteractive_stream_modes = ("messages", "updates", "custom")  # type: ignore[attr-defined]
    stream_agent_with_retry._physicsos_retry_attempts = 5  # type: ignore[attr-defined]
    non_interactive._stream_agent = stream_agent_with_retry


def _physicsos_agent_prompt() -> str:
    return (
        "# PhysicsOS\n\n"
        + PHYSICSOS_SYSTEM_PROMPT
        + "\n\n"
        "You are running inside the official DeepAgents CLI/TUI as the PhysicsOS agent.\n"
        "Use the built-in DeepAgents todo, filesystem, shell, and skills capabilities. MCP is optional and should not define the local architecture.\n"
        "For end-to-end natural-language simulation requests, act as `physicsos-main`: create a case workspace and reproduce the paper loop through DeepAgents subagents.\n"
        "Default loop: analysis files -> TAPS derivation prompt -> derivation.md -> implementation_prompt.md -> case-local kernel.py -> Fig. 7 verification chain -> revise or report.\n"
        "When maintaining case status, call `update_case_stage_status` only with exact stage names from the case manifest. Workspace creation is handled by `create_case_workspace`; never use `workspace` as a stage.\n"
        "If material coefficients, constitutive laws, boundary values, geometry dimensions, TAPS settings, or verification goals are missing, ask the user or use configured search/knowledge tools before deriving the TAPS kernel.\n"
        "For STL/CAD cases, prefer the immersed-boundary / IFE route: Gmsh preprocessing, Cartesian background grid, SDF or voxel geometry embedding, and geometry-coupled Galerkin weak form.\n"
        "For local PhysicsOS package state, use `physicsos paths`.\n"
        "For PhysicsOS Cloud device login, use `physicsos auth login`.\n"
        "For cloud runner jobs, use `physicsos runner ...` commands.\n"
        "Do not use the old typed LangGraph workflow route.\n"
        "Do not claim a high-trust physics solve unless residual, conservation, and verification evidence is available.\n"
    )


def _ensure_deepagents_physicsos_config() -> None:
    agent_dir = Path.home() / ".deepagents" / "physicsos"
    agent_dir.mkdir(parents=True, exist_ok=True)
    agent_prompt = _physicsos_agent_prompt()
    agent_path = agent_dir / "AGENTS.md"
    try:
        if agent_path.exists() and agent_path.read_text(encoding="utf-8") == agent_prompt:
            return
        agent_path.write_text(agent_prompt, encoding="utf-8")
    except PermissionError:
        # A locked or read-only DeepAgents config file should not block running
        # the package-local PhysicsOS agent when the existing file is usable.
        return


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
    model_config = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    params: dict[str, object] = {}
    base_url = os.getenv("PHYSICSOS_OPENAI_BASE_URL") or model_config.get("base_url")
    if base_url:
        params["base_url"] = base_url
    use_responses_api = os.getenv("PHYSICSOS_OPENAI_USE_RESPONSES_API")
    if use_responses_api is None:
        use_responses_api = os.getenv("PHYSICSOS_STRUCTURED_USE_RESPONSES_API")
    if use_responses_api is None and "use_responses_api" in model_config:
        params["use_responses_api"] = bool(model_config.get("use_responses_api"))
    elif use_responses_api is not None:
        params["use_responses_api"] = use_responses_api.strip().lower() in {"1", "true", "yes", "on"}
    if not params:
        return []
    return ["--model-params", json.dumps(params)]


def _prepare_deepagents_env() -> None:
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    temp_dir = runtime_paths().scratch / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    current_temp = str(os.getenv("TEMP") or os.getenv("TMP") or "")
    if os.name == "nt" and "\\windows\\temp" in current_temp.lower():
        os.environ["TEMP"] = str(temp_dir)
        os.environ["TMP"] = str(temp_dir)
        os.environ["TMPDIR"] = str(temp_dir)
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
    model_config = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    search_config = config.get("search", {}) if isinstance(config.get("search"), dict) else {}
    api_key = os.getenv("PHYSICSOS_OPENAI_API_KEY") or model_config.get("api_key")
    base_url = os.getenv("PHYSICSOS_OPENAI_BASE_URL") or model_config.get("base_url")
    model = os.getenv("PHYSICSOS_OPENAI_MODEL") or model_config.get("name")
    search_provider = os.getenv("PHYSICSOS_SEARCH_PROVIDER") or search_config.get("provider")
    tavily_api_key = os.getenv("TAVILY_API_KEY") or os.getenv("PHYSICSOS_TAVILY_API_KEY") or search_config.get("tavily_api_key")
    search_enabled = os.getenv("PHYSICSOS_SEARCH_ENABLED")
    if search_enabled is None and "enabled" in search_config:
        search_enabled = "true" if bool(search_config.get("enabled")) else "false"
    search_max_results = os.getenv("PHYSICSOS_SEARCH_MAX_RESULTS") or search_config.get("max_results")
    if api_key:
        os.environ["PHYSICSOS_OPENAI_API_KEY"] = str(api_key)
        os.environ["DEEPAGENTS_CLI_OPENAI_API_KEY"] = str(api_key)
        os.environ["OPENAI_API_KEY"] = str(api_key)
    if base_url:
        os.environ["PHYSICSOS_OPENAI_BASE_URL"] = str(base_url)
        os.environ["OPENAI_BASE_URL"] = str(base_url)
    if model:
        os.environ["PHYSICSOS_OPENAI_MODEL"] = str(model)
    workspace = Path.cwd()
    existing_workspace = os.environ.get("PHYSICSOS_WORKSPACE")
    existing_auto_workspace = (
        existing_workspace is not None
        and os.environ.get("PHYSICSOS_WORKSPACE_SOURCE") == "physicsos_cli_auto"
        and os.environ.get("PHYSICSOS_WORKSPACE_AUTO_VALUE") == existing_workspace
    )
    if existing_workspace and not existing_auto_workspace:
        os.environ.setdefault("PHYSICSOS_WORKSPACE_SOURCE", "user")
    else:
        os.environ["PHYSICSOS_WORKSPACE"] = str(workspace)
        os.environ["PHYSICSOS_WORKSPACE_SOURCE"] = "physicsos_cli_auto"
        os.environ["PHYSICSOS_WORKSPACE_AUTO_VALUE"] = str(workspace)
    os.environ["PHYSICSOS_AGENT_WORKSPACE"] = "/workspace"
    os.environ["PHYSICSOS_CWD"] = os.environ["PHYSICSOS_WORKSPACE"]
    if search_provider:
        os.environ["PHYSICSOS_SEARCH_PROVIDER"] = str(search_provider)
    if tavily_api_key:
        os.environ["TAVILY_API_KEY"] = str(tavily_api_key)
        os.environ["PHYSICSOS_TAVILY_API_KEY"] = str(tavily_api_key)
    if search_enabled is not None:
        os.environ["PHYSICSOS_SEARCH_ENABLED"] = str(search_enabled)
    if search_max_results:
        os.environ["PHYSICSOS_SEARCH_MAX_RESULTS"] = str(search_max_results)


def _launch_deepagents_cli(argv: list[str]) -> int:
    if not any(arg in {"-h", "--help", "-v", "--version"} for arg in argv):
        _ensure_deepagents_physicsos_config()
    _prepare_deepagents_env()
    _patch_deepagents_banner()
    _patch_deepagents_allow_blocking()
    _patch_deepagents_workspace_paths()
    _patch_deepagents_physicsos_agent_config()
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


def _pseudopotential_config_payload() -> dict[str, object]:
    config = load_config(create=True)
    section = config.get("pseudopotentials", {})
    if not isinstance(section, dict):
        section = {}
    default_library_id = section.get("default_library_id")
    env_root = os.environ.get("PHYSICSOS_PSEUDOPOTENTIAL_DIR")
    return {
        "config_json": str(config_path()),
        "env_override": {
            "PHYSICSOS_PSEUDOPOTENTIAL_DIR": env_root,
            "active": bool(env_root),
        },
        "default_library_id": default_library_id,
        "libraries": section.get("libraries", {}),
        "resolution_order": [
            "tool input library_root",
            "PHYSICSOS_PSEUDOPOTENTIAL_DIR",
            "~/.physicsos/config.json pseudopotentials.libraries.<id>.root",
        ],
        "legal_note": "PhysicsOS stores POTCAR metadata, hashes, paths, and provenance only; it does not copy or redistribute POTCAR contents.",
    }


def _set_pseudopotential_root(*, library_id: str, library_type: str, root: str, description: str | None = None, make_default: bool = True) -> dict[str, object]:
    config = load_config(create=True)
    section = config.setdefault("pseudopotentials", {})
    if not isinstance(section, dict):
        section = {}
        config["pseudopotentials"] = section
    libraries = section.setdefault("libraries", {})
    if not isinstance(libraries, dict):
        libraries = {}
        section["libraries"] = libraries
    entry = libraries.setdefault(library_id, {})
    if not isinstance(entry, dict):
        entry = {}
        libraries[library_id] = entry
    entry["type"] = library_type
    entry["root"] = str(Path(root).expanduser())
    entry["description"] = description or entry.get("description") or "Local pseudopotential library. Stores metadata/provenance only in case artifacts."
    if make_default:
        section["default_library_id"] = library_id
    saved = save_config(config)
    return {
        "config_json": str(saved),
        "library_id": library_id,
        "type": library_type,
        "root": entry["root"],
        "default_library_id": section.get("default_library_id"),
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
                subtitle="Capability-aware physics simulation agent",
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
    print("Capability-aware physics simulation agent")
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

    pseudopotentials = sub.add_parser("pseudopotentials", aliases=["pp"])
    pp_sub = pseudopotentials.add_subparsers(dest="pseudopotentials_command", required=True)
    pp_sub.add_parser("config")
    pp_set_root = pp_sub.add_parser("set-root")
    pp_set_root.add_argument("root")
    pp_set_root.add_argument("--library-id", default="vasp-paw-pbe")
    pp_set_root.add_argument("--type", default="vasp_paw_pbe")
    pp_set_root.add_argument("--description")
    pp_set_root.add_argument("--no-default", action="store_true")
    pp_index = pp_sub.add_parser("index")
    pp_index.add_argument("--case-id", default="pseudopotential-index")
    pp_index.add_argument("--library-id")
    pp_index.add_argument("--root")
    pp_index.add_argument("--max-entries", type=int)
    pp_select = pp_sub.add_parser("select")
    pp_select.add_argument("--case-id", required=True)
    pp_select.add_argument("--structure-ref", required=True)
    pp_select.add_argument("--index-ref")
    pp_select.add_argument("--library-id")
    pp_select.add_argument("--root")
    pp_select.add_argument("--preference", choices=["standard", "pv", "sv", "any"], default="standard")
    pp_select.add_argument("--variant", action="append", default=[], help="Element=Variant override, for example Si=Si_GW.")
    pp_select.add_argument("--allow-gw", action="store_true")
    pp_select.add_argument("--allow-hard-soft", action="store_true")

    geometry = sub.add_parser("geometry")
    geometry_sub = geometry.add_subparsers(dest="geometry_command", required=True)
    apply_labels = geometry_sub.add_parser("apply-boundary-labels")
    apply_labels.add_argument("geometry_json")
    apply_labels.add_argument("labeling_artifact_json")
    apply_labels.add_argument("--output")
    apply_labels.add_argument("--replace-existing", action="store_true")

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
    download = runner_sub.add_parser("download")
    download.add_argument("job_id")
    download.add_argument("artifact_id")
    download.add_argument("--output-dir", default=".")
    download_all = runner_sub.add_parser("download-all")
    download_all.add_argument("job_id")
    download_all.add_argument("--output-dir", default=".")

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

    if args.command in {"pseudopotentials", "pp"}:
        if args.pseudopotentials_command == "config":
            _print_json(_pseudopotential_config_payload())
            return 0
        if args.pseudopotentials_command == "set-root":
            _print_json(
                _set_pseudopotential_root(
                    library_id=args.library_id,
                    library_type=args.type,
                    root=args.root,
                    description=args.description,
                    make_default=not args.no_default,
                )
            )
            return 0
        if args.pseudopotentials_command == "index":
            result = index_vasp_paw_pbe_library(
                IndexVaspPawPbeLibraryInput(
                    case_id=args.case_id,
                    library_root=args.root,
                    library_id=args.library_id,
                    max_entries=args.max_entries,
                )
            )
            if result.errors:
                _print_json({"errors": result.errors, "warnings": result.warnings})
                return 1
            _print_json(
                {
                    "artifact": result.artifact.model_dump() if result.artifact else None,
                    "entry_count": result.data.get("entry_count"),
                    "elements": result.data.get("elements"),
                    "library_id": result.data.get("library_id"),
                    "library_root": result.data.get("library_root"),
                    "library_root_source": result.data.get("library_root_source"),
                    "warnings": result.warnings,
                    "legal_note": result.data.get("legal_note"),
                }
            )
            return 0
        if args.pseudopotentials_command == "select":
            overrides: dict[str, str] = {}
            for raw in args.variant:
                if "=" not in raw:
                    parser.error("--variant must use Element=Variant form")
                element, variant = raw.split("=", 1)
                if not element or not variant:
                    parser.error("--variant must use Element=Variant form")
                overrides[element] = variant
            result = select_pseudopotentials_for_structure(
                SelectPseudopotentialsForStructureInput(
                    case_id=args.case_id,
                    structure_ref=args.structure_ref,
                    index_ref=args.index_ref,
                    library_root=args.root,
                    library_id=args.library_id,
                    preference=args.preference,
                    variant_overrides=overrides,
                    allow_gw=args.allow_gw,
                    allow_hard_soft=args.allow_hard_soft,
                )
            )
            if result.errors:
                _print_json({"errors": result.errors, "warnings": result.warnings})
                return 1
            _print_json(
                {
                    "artifact": result.artifact.model_dump() if result.artifact else None,
                    "artifacts": {key: artifact.model_dump() for key, artifact in result.artifacts.items()},
                    "total_valence_electrons": result.data.get("total_valence_electrons"),
                    "recommended_encut_eV": result.data.get("recommended_encut_eV"),
                    "selected": result.data.get("selected"),
                    "warnings": result.warnings,
                }
            )
            return 0

    if args.command == "geometry":
        if args.geometry_command == "apply-boundary-labels":
            geometry_path = Path(args.geometry_json)
            labeling_path = Path(args.labeling_artifact_json)
            geometry_payload = json.loads(geometry_path.read_text(encoding="utf-8"))
            geometry_spec = GeometrySpec.model_validate(geometry_payload)
            result = apply_boundary_labeling_artifact(
                ApplyBoundaryLabelingArtifactInput(
                    geometry=geometry_spec,
                    labeling_artifact=ArtifactRef(uri=str(labeling_path), kind="boundary_labeling_artifact", format="json"),
                    replace_existing=args.replace_existing,
                )
            )
            output_path = Path(args.output) if args.output else geometry_path.with_name(f"{geometry_path.stem}.confirmed{geometry_path.suffix}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(result.geometry.model_dump_json(indent=2), encoding="utf-8")
            _print_json(
                {
                    "geometry": str(output_path),
                    "applied": result.applied,
                    "warnings": result.warnings,
                    "boundary_count": len(result.geometry.boundaries),
                }
            )
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
        if args.runner_command == "download":
            _print_json(client.download_artifact(args.job_id, args.artifact_id, output_dir=args.output_dir))
            return 0
        if args.runner_command == "download-all":
            _print_json(client.download_all_artifacts(args.job_id, output_dir=args.output_dir))
            return 0

    parser.error("Unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
