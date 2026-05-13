# PhysicsOS

<p align="center">
  <strong>AI-native CAE workspace for paper-style TAPS simulation workflows.</strong><br />
  <span>From a physics problem to derivation, case-local code, verification evidence, and cloud-ready artifacts.</span>
</p>

<p align="center">
  <a href="#english">English</a>
  ·
  <a href="#中文">中文</a>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.12%2B-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img alt="Status" src="https://img.shields.io/badge/status-alpha-orange?style=flat-square" />
  <img alt="Workflow" src="https://img.shields.io/badge/workflow-TAPS--first-4B8BBE?style=flat-square" />
  <img alt="Runtime" src="https://img.shields.io/badge/runtime-DeepAgents-111827?style=flat-square" />
</p>

---

<a id="english"></a>

## English

PhysicsOS is not another black-box solver wrapper. It is a research-grade CAE agent
workspace that makes simulation work inspectable: the agent reads the problem, prepares
analysis files, builds a compact context window, derives a TAPS formulation, writes
case-local code, runs verification steps, and leaves the full evidence trail on disk.

The current system is built around three ideas:

- **Paper-style TAPS workflows**: derivations are generated from explicit templates,
  matrix definitions, and verification routines instead of silently jumping to a solver.
- **Case-local artifacts**: every run writes a structured `cases/<case_id>/` workspace
  with problem statements, derivations, generated kernels, metadata, plots, and reports.
- **Agent orchestration with deterministic tools**: DeepAgents handles interaction and
  delegation; PhysicsOS tools handle geometry, materials, pseudopotentials, context
  assembly, runner commands, and verification contracts.

### Why It Matters

CAE and scientific computing workflows usually fail in places that are hard to audit:
ambiguous assumptions, hidden solver defaults, missing geometry provenance, unverified
generated code, or material data invented by a model. PhysicsOS is designed to keep those
decisions visible. If a problem needs geometry, materials metadata, Kohn-Sham assumptions,
pseudopotential provenance, or convergence evidence, the system turns that requirement
into a file, a tool call, or an explicit open question.

### What PhysicsOS Can Do

| Area | Capability |
| --- | --- |
| PDE/TAPS | Builds paper-style TAPS derivation prompts, derivation files, case-local kernels, implementation notes, and verification reports. |
| Geometry | Converts STL/CAD or simple generated primitives into Gmsh/SDF/voxel/background-grid artifacts, boundary samples, normals, and cut-cell metadata. |
| Materials | Uses `pymatgen`, `spglib`, and `seekpath` for structure parsing, standardization, symmetry, reciprocal lattices, k-meshes, irreducible k-points, supercells, and high-symmetry paths. |
| KS-DFT-TAPS | Prepares Kohn-Sham TAPS problem context, tensor-basis notes, SCF assumptions, band/DOS provenance checks, and verification contracts. |
| Pseudopotentials | Indexes local VASP PAW/PBE libraries by metadata, hashes, paths, and provenance. PhysicsOS does **not** copy or redistribute POTCAR contents. |
| Cloud runner | Provides PhysicsOS Cloud / foamvm login, job submission, status, logs, artifact listing, and artifact download commands. |
| Agent UX | Launches a PhysicsOS-flavored DeepAgents TUI with subagents, local tool bridges, workspace path translation, runtime events, and model configuration. |

### Architecture In One Screen

```text
user request / files / geometry / materials
        |
        v
analysis files
  problem statement, structured inputs, open questions
        |
        v
context window
  local references, tool outputs, templates, geometry/materials notes
        |
        v
TAPS derivation
  weak form, C-HiDeNN-TD approximation, axis matrices, subspace iterations
        |
        v
case-local implementation
  generated kernel.py, execution plan, runtime metadata
        |
        v
verification
  exact/manufactured solution, convergence, physics checks, plots, reports
        |
        v
revise or package runner artifacts
```

DeepAgents is the interactive harness. PhysicsOS is the domain layer: prompts, tools,
schemas, case files, verification contracts, materials processing, and cloud runner
integration.

### Install

Install from PyPI:

```bash
pip install physicsos
```

Install from this checkout:

```bash
pip install -e .
```

For development utilities:

```bash
pip install -e ".[dev]"
```

Requirements:

- Python 3.12+
- An OpenAI-compatible chat model endpoint
- Optional local geometry/materials data depending on the workflow
- Optional PhysicsOS Cloud / foamvm account for remote runner commands

### Configure A Model

PowerShell:

```powershell
$env:PHYSICSOS_OPENAI_API_KEY="..."
$env:PHYSICSOS_OPENAI_BASE_URL="https://api.example.com/v1"
$env:PHYSICSOS_OPENAI_MODEL="gpt-5.4"
```

If your provider uses the OpenAI Responses API:

```powershell
$env:PHYSICSOS_OPENAI_USE_RESPONSES_API="true"
```

PhysicsOS also writes a local config file under the active PhysicsOS home directory.
Environment variables override config values for one-off runs.

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

### Start The Agent

```bash
physicsos
```

Run a single request:

```bash
physicsos --message "derive and verify a 1D steady heat conduction TAPS case"
```

Resume a previous interactive session:

```bash
physicsos --resume
```

Use a specific model through DeepAgents:

```bash
physicsos --model openai:gpt-5.4
```

### Local CLI

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
physicsos pseudopotentials set-root "D:\path\to\vasp_paw_pbe" --library-id vasp-paw-pbe
physicsos pseudopotentials index --case-id pp-index
physicsos pseudopotentials select --case-id si-case --structure-ref cases/si/structure.json
```

`physicsos pp ...` is the short alias for `physicsos pseudopotentials ...`.

### What A Case Produces

```text
cases/<case_id>/
  problem/
  context/
  references/
  geometry/
  materials/
  pseudopotentials/
  taps/
  verification/
  report/
  execution_plan.md
  manifest.json
```

The exact tree depends on the request. Geometry cases include SDF/voxel/boundary
artifacts. Materials cases include standardized structures, symmetry, reciprocal lattice,
k-point, and pseudopotential-provenance artifacts. TAPS cases include derivations,
implementation notes, generated kernels, and verification outputs.

### Runtime Data

Set `PHYSICSOS_HOME` to control where runtime state is stored. Without it, installed
usage stores data under `~/.physicsos/`; source-checkout usage keeps development state in
the repository workspace.

```text
config        ~/.physicsos/config.json
sessions      ~/.physicsos/sessions/
history       ~/.physicsos/history.jsonl
scratch       ~/.physicsos/scratch/
case memory   ~/.physicsos/data/case_memory.jsonl
knowledge DB  ~/.physicsos/data/knowledge/physicsos_knowledge.sqlite
```

Print the exact active paths:

```bash
physicsos paths
```

### Project Status

PhysicsOS is alpha-stage research infrastructure. It is intentionally transparent and
file-heavy. Expect inspectable intermediate artifacts, explicit assumptions, local
generated code, and verification evidence. It is not a certified solver, not a hidden
VASP/QE/CP2K wrapper, and not a promise that every generated case is correct without
review.

---

<a id="中文"></a>

## 中文

PhysicsOS 不是又一个黑盒求解器封装。它是一个面向 CAE 和科学计算的 AI 原生工作台：从用户给出的物理问题、几何、材料结构或脚本出发，自动组织分析文件，构建上下文窗口，推导 TAPS 公式，生成当前 case 专属代码，执行验证链，并把所有证据留在本地文件中。

它的定位很明确：**让仿真代理的每一步都可检查、可复现、可追责。**

### 核心特点

- **TAPS-first**：以论文式 TAPS / C-HiDeNN-TD 推导为主线，不把问题偷偷塞进固定求解器。
- **case-local**：每个任务都有独立的 `cases/<case_id>/` 工作区，包含问题、推导、代码、验证、图和报告。
- **多 Agent 协作**：DeepAgents 负责交互、子代理和工具调用；PhysicsOS 提供物理、几何、材料、验证和云 runner 工具。
- **几何可追溯**：STL/CAD 或简单几何会被转成 Gmsh、SDF、体素、边界采样、法向和 cut-cell 元数据。
- **材料确定性处理**：晶体结构、空间群、倒格矢、k 点、seekpath 高对称路径由 `pymatgen` / `spglib` / `seekpath` 工具生成，不靠模型硬猜。
- **KS-DFT-TAPS 扩展**：支持 Kohn-Sham TAPS 上下文、张量基、SCF 假设、能带/DOS provenance、赝势元数据和验证契约。
- **赝势不搬运**：PhysicsOS 只记录本地 POTCAR 的 metadata、hash、路径和 provenance，不复制、不分发 POTCAR 正文。

### 它解决什么问题

传统 CAE/DFT/AI 代码生成工作流经常在这些地方失控：假设写不清、默认参数藏起来、几何来源不明、材料结构被模型猜错、生成代码没有验证、结果图无法追溯。PhysicsOS 的做法是把这些关键点变成明确的文件、工具输出、验证报告或 open question。

### 一屏理解架构

```text
用户问题 / 文件 / 几何 / 材料
        |
        v
分析文件
  问题陈述、结构化输入、未决问题
        |
        v
上下文窗口
  本地参考、工具输出、模板、几何/材料说明
        |
        v
TAPS 推导
  弱形式、C-HiDeNN-TD、矩阵定义、子空间迭代
        |
        v
当前 case 专属实现
  kernel.py、执行计划、运行元数据
        |
        v
验证
  精确/制造解、收敛性、物理一致性、图和报告
        |
        v
修正或打包云端 runner 产物
```

### 安装

从 PyPI 安装：

```bash
pip install physicsos
```

从当前源码目录安装：

```bash
pip install -e .
```

开发工具：

```bash
pip install -e ".[dev]"
```

需要：

- Python 3.12+
- OpenAI-compatible 模型接口
- 视任务需要准备本地几何/材料/赝势数据
- 如需远端运行，准备 PhysicsOS Cloud / foamvm 账号

### 配置模型

PowerShell:

```powershell
$env:PHYSICSOS_OPENAI_API_KEY="..."
$env:PHYSICSOS_OPENAI_BASE_URL="https://api.example.com/v1"
$env:PHYSICSOS_OPENAI_MODEL="gpt-5.4"
```

如果你的模型服务使用 OpenAI Responses API：

```powershell
$env:PHYSICSOS_OPENAI_USE_RESPONSES_API="true"
```

### 启动

```bash
physicsos
```

单次请求：

```bash
physicsos --message "为一维稳态热传导问题推导并验证 TAPS case"
```

恢复会话：

```bash
physicsos --resume
```

指定模型：

```bash
physicsos --model openai:gpt-5.4
```

### 常用命令

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

几何辅助：

```bash
physicsos geometry apply-boundary-labels geometry.json labeling_artifact.json --output confirmed.json
```

赝势辅助：

```bash
physicsos pseudopotentials config
physicsos pseudopotentials set-root "D:\path\to\vasp_paw_pbe" --library-id vasp-paw-pbe
physicsos pseudopotentials index --case-id pp-index
physicsos pseudopotentials select --case-id si-case --structure-ref cases/si/structure.json
```

### 当前状态

PhysicsOS 仍处于 alpha 阶段。它适合研究、原型、方法验证和可审计的 AI-CAE 工作流实验。它不是认证工程软件，也不会绕过人工检查。这里的核心价值不是“自动给出一个神奇答案”，而是把推导、实现、验证和假设完整摊开，让用户能看见每一步。

