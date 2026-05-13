# PhysicsOS

PhysicsOS is an alpha-stage CAE agent workspace for paper-style TAPS workflows. It uses
the DeepAgents CLI as the interactive agent harness, then coordinates specialist agents
and local tools to turn a physics problem into auditable case files: problem statements,
TAPS derivations, case-local kernels, verification scripts, plots, reports, and optional
cloud-runner manifests.

The project is not a fixed numerical solver framework. The default route is an
agent-in-the-loop workflow that follows the architecture in `ARCHITECTURE.md`: assemble
analysis files and local references, derive the TAPS formulation, implement the current
case, verify it, and revise when checks fail.

## What It Does

- Launches a PhysicsOS-flavored DeepAgents TUI with project prompts, subagents, tools,
  runtime events, and workspace path handling.
- Creates reproducible `/cases/<case_id>/` workspaces for problem analysis, context
  assembly, derivations, generated kernels, verification, and reports.
- Supports paper-style TAPS derivation and implementation for PDE problems, including
  exact/manufactured-solution and convergence-check workflows.
- Adds geometry preprocessing for STL/CAD or simple generated primitives using Gmsh,
  background grids, SDF/voxel artifacts, boundary samples, normals, and cut-cell metadata.
- Provides materials preprocessing helpers based on `pymatgen`, `spglib`, and `seekpath`
  for crystal structures, k-point paths, supercells, reciprocal lattices, and related
  KS-DFT-TAPS context.
- Tracks local pseudopotential libraries by metadata, hashes, and provenance only. It
  does not copy or redistribute POTCAR contents.
- Provides PhysicsOS Cloud / foamvm device login, job submission, status, logs, and
  artifact download commands.

## Current Status

PhysicsOS is research infrastructure. Expect visible intermediate files, explicit
approval gates, and case-local generated code. Some tools are deterministic local
helpers; others prepare prompts, manifests, or artifacts for agent review. External
solver execution is not hidden behind the agent.

## Requirements

- Python 3.12 or newer
- An OpenAI-compatible chat model endpoint for the agent runtime
- Optional local tools/data depending on the workflow:
  - Gmsh/mesh tooling for geometry workflows
  - local VASP PAW/PBE pseudopotential directories for KS-DFT-TAPS provenance workflows
  - PhysicsOS Cloud / foamvm account for remote runner commands

## Install

From this checkout:

```bash
pip install -e .
```

For development and packaging tools:

```bash
pip install -e ".[dev]"
```

The core install already includes the Python packages declared by the project, including
DeepAgents CLI integration, LangGraph/OpenAI-compatible model support, geometry/materials
dependencies, and the `physicsos` console command.

## Configure A Model

Set model credentials with environment variables:

```powershell
$env:PHYSICSOS_OPENAI_API_KEY="..."
$env:PHYSICSOS_OPENAI_BASE_URL="https://api.example.com/v1"
$env:PHYSICSOS_OPENAI_MODEL="gpt-5.4"
```

If your provider uses the OpenAI Responses API:

```powershell
$env:PHYSICSOS_OPENAI_USE_RESPONSES_API="true"
```

PhysicsOS also creates a config file on first run. Environment variables override this
file for one-off runs.

```json
{
  "model": {
    "provider": "openai",
    "name": "gpt-5.4",
    "api_key": "",
    "base_url": "https://api.example.com/v1",
    "use_responses_api": false
  },
  "cloud": {
    "runner_url": "https://foamvm.vercel.app",
    "access_token": ""
  }
}
```

## Use The Agent

Launch the interactive PhysicsOS DeepAgents TUI:

```bash
physicsos
```

Run a single prompt:

```bash
physicsos --message "derive and verify a 1D steady heat conduction TAPS case"
```

Resume an existing DeepAgents session:

```bash
physicsos --resume
```

Use a specific model through the underlying DeepAgents CLI:

```bash
physicsos --model openai:gpt-5.4
```

PhysicsOS patches the DeepAgents runtime at startup so the agent sees the project as a
`/workspace` tree while shell commands still run from the real local workspace.

## Local Commands

These commands run locally without entering the agent route:

```bash
physicsos paths
physicsos auth login
physicsos account
physicsos runner submit path/to/manifest.json
physicsos runner status JOB_ID
physicsos runner logs JOB_ID
physicsos runner artifacts JOB_ID
physicsos runner download JOB_ID ARTIFACT_ID
physicsos runner download-all JOB_ID
```

Geometry helper:

```bash
physicsos geometry apply-boundary-labels geometry.json labeling_artifact.json --output confirmed.json
```

Pseudopotential helpers:

```bash
physicsos pseudopotentials config
physicsos pseudopotentials set-root D:\path\to\vasp_paw_pbe --library-id vasp-paw-pbe
physicsos pseudopotentials index --case-id pp-index
physicsos pseudopotentials select --case-id si-case --structure-ref cases/si/structure.json
```

`physicsos pp ...` is an alias for `physicsos pseudopotentials ...`.

## Case Workflow

A typical successful run writes files under:

```text
cases/<case_id>/
  problem/
    problem_statement.md
    problem.json
    open_questions.md
  context/
    context_window.md
  references/
  geometry/
  taps/
    derivation_prompt.md
    derivation.md
    implementation_notes.md
    kernel.py
  verification/
  postprocess/
  execution_plan.md
  manifest.json
```

The exact tree depends on the task. A geometry problem will include SDF, occupancy,
boundary, normal, and embedding artifacts. A materials or KS-DFT-TAPS task will include
structure, k-point, pseudopotential-provenance, and verification artifacts.

## Runtime Data

PhysicsOS uses `PHYSICSOS_HOME` when set. Otherwise:

- pip-installed usage stores runtime state under `~/.physicsos/`
- source-checkout usage keeps development artifacts in this repository

Common paths:

```text
config:        ~/.physicsos/config.json
sessions:      ~/.physicsos/sessions/session-*.jsonl
history:       ~/.physicsos/history.jsonl
scratch:       ~/.physicsos/scratch/
case memory:   ~/.physicsos/data/case_memory.jsonl
knowledge DB:  ~/.physicsos/data/knowledge/physicsos_knowledge.sqlite
```

Run this to see the exact paths for the current environment:

```bash
physicsos paths
```

## Knowledge And References

Local reference notes live under `docs/knowledge_seed/`. To build or refresh the local
knowledge database:

```powershell
python -B scripts\build_knowledge_base.py
```

To seed a broader computational-physics corpus:

```powershell
python -B scripts\seed_computational_physics_knowledge.py --max-results 5
```

See `QUICKSTART.md` for DeepSearch and surrogate-checkpoint notes.

## Development

Run tests:

```bash
python -B -m pytest -q
```

Build and check a package:

```bash
python -m build
python -m twine check dist/*
```

Useful docs:

- `ARCHITECTURE.md` - current project architecture and agent roles
- `QUICKSTART.md` - model setup, knowledge tools, and surrogate notes
- `DFT.md` - KS-DFT-TAPS notes
- `taps.md` - TAPS strategy and background
- `vm.md` - PhysicsOS Cloud / foamvm integration

## Packaging Notes

Package distribution includes the `physicsos` Python package and console command.
Generated cases, scratch data, local sessions, model weights, secrets, knowledge
databases, and large research PDFs are intentionally excluded from package data.
