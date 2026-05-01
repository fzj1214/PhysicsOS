# PhysicsOS Quickstart

## 1. OpenAI-compatible DeepAgents Runtime

Copy `.env.example` to `.env` or set these environment variables directly.

```powershell
$env:PHYSICSOS_OPENAI_API_KEY="your_api_key_here"
$env:PHYSICSOS_OPENAI_BASE_URL="https://api.tu-zi.com/v1"
$env:PHYSICSOS_OPENAI_MODEL="gpt-5.4"
$env:PHYSICSOS_DEEPSEARCH_MODEL="gemini-3-pro-deepsearch-async"
$env:PHYSICSOS_OPENAI_USE_RESPONSES_API="true"
```

Then create the agent:

```python
from physicsos.agents import create_openai_compatible_model, create_physicsos_agent

model = create_openai_compatible_model()
agent = create_physicsos_agent(model=model)
```

Notes:

```text
PHYSICSOS_OPENAI_API_KEY is where you put the API key.
PHYSICSOS_OPENAI_BASE_URL should normally be the /v1 API root.
For https://api.tu-zi.com/v1/responses, set base_url to https://api.tu-zi.com/v1 and PHYSICSOS_OPENAI_USE_RESPONSES_API=true.
```

## 2. Knowledge Base And DeepSearch

Build the local knowledge base from project docs and extracted paper notes:

```powershell
python -B scripts\build_knowledge_base.py
```

Build the richer computational physics seed corpus:

```powershell
python -B scripts\seed_computational_physics_knowledge.py --max-results 5
```

This seeds:

```text
project architecture and TAPS docs
core computational physics formula notes
arXiv abstracts across FEM/FVM/PETSc/MOR/neural operators/mesh/DFT/MD/UQ/TAPS
selected arXiv PDF full text
DeepSearch synthesis reports when the provider channel is available
```

Knowledge tools available to `knowledge-agent`:

```text
search_knowledge_base
build_knowledge_context
search_arxiv
run_deepsearch
ingest_knowledge_document
search_case_memory
store_case_result
```

DeepSearch uses the same OpenAI-compatible API key and base URL:

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_API_KEY",
    base_url="https://api.tu-zi.com/v1",
)

response = client.chat.completions.create(
    model="gemini-3-pro-deepsearch-async",
    messages=[{"role": "user", "content": "你好啊"}],
    temperature=0.7,
    stream=False,
)
```

If `gemini-3-pro-deepsearch-async` returns a provider/channel error, the tool will return a structured `DeepSearchReport.error` instead of crashing the agent. Fix the provider group/channel or set another model:

```powershell
$env:PHYSICSOS_DEEPSEARCH_MODEL="another-deepsearch-model"
```

Current MCP status:

```text
No arXiv MCP server is configured in this workspace yet.
PhysicsOS provides search_arxiv through the official arXiv Atom API now.
If an arXiv MCP server is added later, knowledge-agent should wrap it behind the same search_arxiv contract.
```

Smoke test:

```powershell
python -B examples\openai_compatible_agent.py
```

## 3. Surrogate / Neural Operator Checkpoints

Recommended small representative download:

```powershell
python -B scripts\download_surrogate_weights.py
```

This downloads four public representative checkpoints:

```text
polymathic-ai/FNO-rayleigh_benard
polymathic-ai/TFNO-helmholtz_staircase
polymathic-ai/UNetConvNext-gray_scott_reaction_diffusion
polymathic-ai/UNetClassic-acoustic_scattering_maze
```

Download a specific repo:

```powershell
python -B scripts\download_surrogate_weights.py --repo polymathic-ai/FNO-shear_flow
```

Download the full Polymathic FNO/TFNO/UNet set only when you really want the large local cache:

```powershell
python -B scripts\download_surrogate_weights.py --all-polymathic
```

Copy the example registry:

```powershell
Copy-Item configs\surrogates.example.json configs\surrogates.local.json
```

Set the registry path:

```powershell
$env:PHYSICSOS_SURROGATE_REGISTRY="configs/surrogates.local.json"
```

Put real trained checkpoints under `models/`, for example:

```text
models/
  grid_fno_local/
    model.ts
    metadata.json
  mesh_graph_operator_local/
    model.ts
    metadata.json
```

Then edit `configs/surrogates.local.json` so each model has:

```json
{
  "checkpoint": {
    "uri": "models/grid_fno_local/model.ts",
    "kind": "model_checkpoint",
    "format": "torchscript"
  },
  "runner": "torchscript",
  "input_adapter": "your_adapter_name",
  "output_adapter": "your_adapter_name"
}
```

Check registry state:

```powershell
python -B examples\surrogate_registry_smoke.py
```

Current built-in adapters:

```text
the_well_grid -> FNO / TFNO checkpoints from polymathic-ai The Well benchmark
the_well_unet -> UNetClassic / UNetConvNext checkpoints from polymathic-ai The Well benchmark
taps_apriori  -> equation-driven TAPS-style backend metadata
```

Important:

```text
The framework can now register checkpoints and build model-specific input/output bundles.
Actual numerical inference still requires a real input tensor that matches each model's dataset preprocessing.
The adapters intentionally do not fabricate physics tensors.
```
