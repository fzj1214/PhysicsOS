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

With agent integrations:

```bash
pip install -e ".[agents]"
```

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

Running `physicsos` with no arguments starts the interactive CLI shell.
Plain text input is sent to the DeepAgents-powered PhysicsOS orchestrator as a natural-language chat turn.
Local control commands are prefixed with `/`, for example `/paths`, `/login`, and `/exit`.
The shell records user inputs, assistant outputs, and errors to a session file plus the global history file.
Device-login tokens are stored under the user config directory, not in the repository.

To enable the agent loop, install the agent extras and configure an OpenAI-compatible model:

```bash
pip install -e ".[agents]"
```

```powershell
$env:PHYSICSOS_OPENAI_API_KEY="..."
$env:PHYSICSOS_OPENAI_BASE_URL="https://api.tu-zi.com/v1"
$env:PHYSICSOS_OPENAI_MODEL="gpt-5.4"
```

## Local Data

PhysicsOS uses `PHYSICSOS_HOME` when set. Otherwise, pip-installed usage stores runtime state under:

```text
~/.physicsos/
```

In a source checkout, runtime artifacts stay under the repository so tests and development remain reproducible.

Default paths:

- Cloud auth config: `~/.physicsos/config.toml`
- Interactive sessions: `~/.physicsos/sessions/session-*.jsonl` for pip installs, or `./sessions/session-*.jsonl` in a source checkout
- Command history: `~/.physicsos/history.jsonl` for pip installs, or `./history.jsonl` in a source checkout
- Solver/session artifacts: `~/.physicsos/scratch/...` for pip installs, or `./scratch/...` in a source checkout
- Case memory: `~/.physicsos/data/case_memory.jsonl` for pip installs, or `./data/case_memory.jsonl` in a source checkout
- Knowledge base: `~/.physicsos/data/knowledge/physicsos_knowledge.sqlite` for pip installs, or `./data/knowledge/physicsos_knowledge.sqlite` in a source checkout

Set `PHYSICSOS_HOME=/path/to/physicsos-home` to relocate these files.
Run `physicsos paths` to print the exact paths used by the current environment.

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
