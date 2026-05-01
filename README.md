# PhysicsOS

PhysicsOS is an AI-native physics simulation scaffold built around a TAPS-first workflow:

- compile physics problems into typed weak-form / solver IR;
- route executable IR blocks to local TAPS kernels;
- prepare audited fallback bundles for FEniCSx, MFEM, PETSc, or cloud runners;
- expose a CLI for PhysicsOS Cloud / foamvm device-login and job submission.

The project is currently alpha-stage research infrastructure. Local TAPS kernels are intentionally lightweight and deterministic; full-solver execution is prepared through manifests and approval gates rather than silently running external software.

## Install

From a local checkout:

```bash
pip install -e .
```

The default install includes the official DeepAgents CLI/TUI, LangGraph, and the OpenAI-compatible model adapter, so `physicsos` starts the natural-language PhysicsOS agent loop immediately after model credentials are configured.

With Gmsh / meshio geometry tooling:

```bash
pip install -e ".[geometry]"
```

For package build and test tooling:

```bash
pip install -e ".[dev]"
```

## CLI

```bash
physicsos
physicsos paths
physicsos auth login
physicsos account
physicsos runner submit path/to/manifest.json
physicsos runner status JOB_ID
physicsos runner logs JOB_ID
physicsos runner artifacts JOB_ID
```

Running `physicsos` with no arguments launches the official DeepAgents Textual TUI with the `physicsos` agent selected.
PhysicsOS automatically installs its DeepAgents agent prompt and subagent prompts under `~/.deepagents/physicsos/`.
DeepAgents manages interactive threads, TUI state, model selection, MCP tools, skills, and approval prompts.
PhysicsOS local control commands remain available as normal shell commands, for example `physicsos paths`, `physicsos auth login`, and `physicsos runner ...`.
Device-login tokens are stored under the user config directory, not in the repository.

To enable model calls, configure an OpenAI-compatible model:

```powershell
$env:PHYSICSOS_OPENAI_API_KEY="..."
$env:PHYSICSOS_OPENAI_BASE_URL="https://api.tu-zi.com/v1"
$env:PHYSICSOS_OPENAI_MODEL="gpt-5.4"
```

These variables are mapped to the official DeepAgents CLI at startup. You can still pass native DeepAgents flags through `physicsos`, for example:

```bash
physicsos --message "simulate a 1D steady heat conduction problem"
physicsos --resume
physicsos --model openai:gpt-5.4
```

## Local Data

PhysicsOS uses `PHYSICSOS_HOME` when set. Otherwise, pip-installed usage stores runtime state under:

```text
~/.physicsos/
```

In a source checkout, runtime artifacts stay under the repository so tests and development remain reproducible.

Default paths:

- Unified config: `~/.physicsos/config.json`
- Cloud auth config: `~/.physicsos/config.json` under the `cloud` object
- Interactive sessions: `~/.physicsos/sessions/session-*.jsonl` for pip installs, or `./sessions/session-*.jsonl` in a source checkout
- Command history: `~/.physicsos/history.jsonl` for pip installs, or `./history.jsonl` in a source checkout
- Solver/session artifacts: `~/.physicsos/scratch/...` for pip installs, or `./scratch/...` in a source checkout
- Case memory: `~/.physicsos/data/case_memory.jsonl` for pip installs, or `./data/case_memory.jsonl` in a source checkout
- Knowledge base: `~/.physicsos/data/knowledge/physicsos_knowledge.sqlite` for pip installs, or `./data/knowledge/physicsos_knowledge.sqlite` in a source checkout

Set `PHYSICSOS_HOME=/path/to/physicsos-home` to relocate these files.
Run `physicsos paths` to print the exact paths used by the current environment.

## Config

PhysicsOS creates `~/.physicsos/config.json` on first run. Edit it to set model, API, cloud, and storage preferences in one place:

```json
{
  "model": {
    "provider": "openai",
    "name": "gpt-5.4",
    "api_key": "",
    "base_url": "https://api.tu-zi.com/v1",
    "use_responses_api": false
  },
  "cloud": {
    "runner_url": "https://foamvm.vercel.app",
    "access_token": ""
  }
}
```

Environment variables such as `PHYSICSOS_OPENAI_API_KEY`, `PHYSICSOS_OPENAI_BASE_URL`, and `PHYSICSOS_OPENAI_MODEL` still override this file for one-off runs.

## Development

```bash
python -B -m pytest -q
python -m build
python -m twine check dist/*
```

## Notes

- `ARCHITECTURE.md` is the main design document.
- `taps.md` describes the TAPS-first solver strategy.
- `vm.md` describes PhysicsOS Cloud / foamvm integration.
- Heavy model weights, generated scratch artifacts, local secrets, and knowledge databases are excluded from package distribution.
