# PhysicsOS Architecture

PhysicsOS 的最终目标不是单个 CFD copilot，也不是单个 PDE surrogate，而是一个统一的 AI-native physics simulation OS。

系统采用三核架构：

```text
1. Geometry + Mesh Agent
2. Solver Model / Solver Runtime
3. Postprocess + Analysis Agent
```

工程骨架使用 `langchain-ai/deepagents`，但 PhysicsOS 的物理接口、类型、工具注册、求解器协议必须全部显式定义，不能只靠 prompt 约定。

DeepAgents 负责：

```text
planning
subagent orchestration
filesystem/context management
human-in-the-loop approval
tool calling loop
memory/checkpoint integration
```

PhysicsOS 自己负责：

```text
PhysicsProblem IR
GeometrySpec / MeshSpec
OperatorSpec
TAPSCompilationPlan / TAPSWeakFormSpec / TAPSProblem / equation-driven solver compiler
SolverBackend protocol
Verification protocol
Postprocess protocol
registered physics tools
case memory / training data flywheel
local knowledge base / arXiv / DeepSearch research layer
```

---

## 1. DeepAgents Integration Principle

`deepagents` 是 agent harness，不是 physics runtime。它的作用是把复杂任务拆解、分发给子 agent、调用工具、管理上下文和文件系统。

PhysicsOS 中的主 agent 应该是 orchestrator：

```text
PhysicsOS Main Agent
→ parse user intent
→ build/validate PhysicsProblem
→ delegate geometry/mesh work
→ try TAPS-first equation-driven solve when applicable
→ route neural/full-solver fallback only when needed
→ run verifier
→ delegate postprocess/report
```

三个核心子 agent 对应 PhysicsOS 的长期架构：

```text
geometry-mesh-agent
taps-agent
solver-agent
postprocess-agent
```

另外保留两个支撑 agent：

```text
verification-agent
knowledge-agent
```

DeepAgents 原生能力的映射：

```text
write_todos              -> simulation plan / workflow trace
read_file/write_file     -> case workspace, generated configs, reports
task                     -> geometry/solver/postprocess subagent dispatch
custom tools             -> PhysicsOS tool registry
backend                  -> case filesystem + persistent memory
interrupt_on             -> risky solver / HPC / destructive file operations approval
response_format          -> structured JSON/Pydantic outputs
```

---

## 2. Runtime Topology

推荐采用 Python DeepAgents SDK 作为第一版实现。

TypeScript/前端侧只消费 JSON Schema；权威类型定义放在 Python Pydantic models 中，再导出 JSON Schema。

```text
physicsos/
  agents/
    main.py
    prompts.py
    subagents.py

  schemas/
    problem.py
    taps.py
    geometry.py
    mesh.py
    operators.py
    materials.py
    boundary.py
    solver.py
    verification.py
    postprocess.py

  tools/
    registry.py
    geometry_tools.py
    mesh_tools.py
    solver_tools.py
    taps_tools.py
    verification_tools.py
    postprocess_tools.py
    memory_tools.py

  backends/
    solver_base.py
    surrogate_neural_operator.py
    openfoam.py
    su2.py
    fenics.py
    mfem.py
    quantum_espresso.py
    cp2k.py
    lammps.py
    cantera.py
    openmc.py
    taps.py

  memory/
    case_store.py
    embedding.py
    datasets.py

  workflows/
    cfd_thermal.py
    structural.py
    materials.py
```

---

## 3. DeepAgents Bootstrap

PhysicsOS main agent should be created with:

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from physicsos.agents.prompts import PHYSICSOS_SYSTEM_PROMPT
from physicsos.agents.subagents import SUBAGENTS
from physicsos.tools.registry import PHYSICSOS_TOOLS

checkpointer = MemorySaver()
store = InMemoryStore()

agent = create_deep_agent(
    model="openai:gpt-5.4",
    tools=PHYSICSOS_TOOLS,
    subagents=SUBAGENTS,
    system_prompt=PHYSICSOS_SYSTEM_PROMPT,
    backend=CompositeBackend(
        default=StateBackend(),
        routes={
            "/cases/": StoreBackend(),
            "/datasets/": StoreBackend(),
            "/reports/": StoreBackend(),
            "/scratch/": StateBackend(),
        },
    ),
    store=store,
    checkpointer=checkpointer,
    interrupt_on={
        "run_full_solver": {"allowed_decisions": ["approve", "edit", "reject"]},
        "submit_hpc_job": {"allowed_decisions": ["approve", "edit", "reject"]},
        "delete_case_artifacts": {"allowed_decisions": ["approve", "reject"]},
    },
    name="physicsos-main",
)
```

Implementation notes:

```text
HITL requires a checkpointer because interrupted runs must be resumed by thread id.
StoreBackend requires a LangGraph store in local development, or a provisioned store in deployment.
For production, full solver execution should run in a sandbox/HPC service, not directly through a host shell backend.
```

The main agent prompt must enforce:

```text
1. Always create or update PhysicsProblem before solving.
2. Never call a solver without GeometrySpec, OperatorSpec, BCs, materials, targets, and verification policy.
3. Use subagents for geometry/mesh, solver execution, verification, and postprocess.
4. Return structured outputs, not raw logs.
5. Treat TAPS as the primary solver for applicable equation-driven, parameterized PDEs.
6. For high-risk/OOD/non-TAPS cases, call neural surrogate or full solver fallback.
```

---

## 4. Subagent Definitions

DeepAgents subagents should be dictionary specs with strict names, descriptions, system prompts, minimal tool sets, and optional structured output.

### 4.1 Geometry + Mesh Agent

```python
geometry_mesh_agent = {
    "name": "geometry-mesh-agent",
    "description": (
        "Builds GeometrySpec and MeshSpec from CAD, STL, STEP, CIF, POSCAR, "
        "molecular graphs, images, or text descriptions. Use for geometry cleanup, "
        "region labeling, boundary labeling, mesh generation, mesh quality checks, "
        "and surrogate-ready geometry encodings."
    ),
    "system_prompt": GEOMETRY_MESH_AGENT_PROMPT,
    "tools": [
        import_geometry,
        repair_geometry,
        label_regions,
        generate_mesh,
        assess_mesh_quality,
        generate_geometry_encoding,
        export_backend_mesh,
    ],
    "response_format": GeometryMeshResult,
}
```

Output contract:

```python
class GeometryMeshResult(BaseModel):
    geometry: GeometrySpec
    mesh: MeshSpec | None
    encodings: list[GeometryEncoding]
    quality: MeshQualityReport | None
    warnings: list[str] = []
    recommended_next_action: str
```

### 4.2 TAPS Agent

```python
taps_agent = {
    "name": "taps-agent",
    "description": (
        "Compiles PhysicsProblem into TAPSProblem and runs the equation-driven "
        "TAPS backend. Use first for explicit PDEs with identifiable parameters, "
        "tensorizable geometry, and smooth solution structure."
    ),
    "system_prompt": TAPS_AGENT_PROMPT,
    "tools": [
        estimate_taps_support,
        build_taps_problem,
        run_taps_backend,
        estimate_taps_residual,
        refine_taps_problem,
    ],
    "response_format": TAPSRunResult,
}
```

Output contract:

```python
class TAPSRunResult(BaseModel):
    result: SolverResult
    taps_problem: TAPSProblem
    residual_report: VerificationReport
    fallback_recommended: bool
    recommended_next_action: str
```

### 4.3 Solver Agent

```python
solver_agent = {
    "name": "solver-agent",
    "description": (
        "Selects and runs non-TAPS solver backends for a validated PhysicsProblem. "
        "Use after taps-agent rejects a case or verification recommends fallback."
    ),
    "system_prompt": SOLVER_AGENT_PROMPT,
    "tools": [
        validate_physics_problem,
        estimate_solver_support,
        route_solver_backend,
        run_surrogate_solver,
        run_full_solver,
        run_hybrid_solver,
        collect_solver_artifacts,
    ],
    "response_format": SolverRunResult,
}
```

Output contract:

```python
class SolverRunResult(BaseModel):
    result: SolverResult
    backend_decision: SolverDecision
    artifacts: list[ArtifactRef]
    warnings: list[str] = []
    recommended_next_action: str
```

### 4.4 Verification Agent

```python
verification_agent = {
    "name": "verification-agent",
    "description": (
        "Checks whether a simulation result is physically and numerically trustworthy. "
        "Use after every surrogate, hybrid, or full solver run."
    ),
    "system_prompt": VERIFICATION_AGENT_PROMPT,
    "tools": [
        compute_physics_residuals,
        check_conservation_laws,
        estimate_uncertainty,
        detect_ood_case,
        compare_against_reference,
        recommend_verification_action,
    ],
    "response_format": VerificationReport,
}
```

### 4.5 Postprocess + Analysis Agent

```python
postprocess_agent = {
    "name": "postprocess-agent",
    "description": (
        "Turns raw fields and scalar outputs into engineering insight. Use for "
        "visualization, KPI extraction, sensitivity analysis, optimization suggestions, "
        "and report generation."
    ),
    "system_prompt": POSTPROCESS_AGENT_PROMPT,
    "tools": [
        load_solver_result,
        extract_kpis,
        generate_visualizations,
        run_sensitivity_analysis,
        propose_design_update,
        write_simulation_report,
    ],
    "response_format": PostprocessResult,
}
```

### 4.6 Knowledge Agent

```python
knowledge_agent = {
    "name": "knowledge-agent",
    "description": (
        "Retrieves prior cases, physical assumptions, material data, solver templates, "
        "paper notes, and validation guidance. Use before routing difficult or OOD cases."
    ),
    "system_prompt": KNOWLEDGE_AGENT_PROMPT,
    "tools": [
        search_case_memory,
        retrieve_material_properties,
        retrieve_operator_template,
        retrieve_validation_rule,
        retrieve_paper_notes,
    ],
    "response_format": KnowledgeRetrievalResult,
}
```

---

## 5. Core Physics Types

All tools and subagents must exchange these models. Free-form dicts are not allowed past prototyping.

### 5.1 PhysicsProblem

```python
class PhysicsProblem(BaseModel):
    id: str
    user_intent: UserIntent
    domain: PhysicsDomain
    geometry: GeometrySpec
    mesh: MeshSpec | None = None
    fields: list[FieldSpec]
    operators: list[OperatorSpec]
    materials: list[MaterialSpec]
    boundary_conditions: list[BoundaryConditionSpec]
    initial_conditions: list[InitialConditionSpec] = []
    parameters: list[ParameterSpec] = []
    targets: list[TargetSpec]
    constraints: list[ConstraintSpec] = []
    verification_policy: VerificationPolicy
    provenance: Provenance
```

```python
PhysicsDomain = Literal[
    "fluid",
    "thermal",
    "solid",
    "electromagnetic",
    "acoustic",
    "molecular",
    "quantum",
    "multiphysics",
    "custom",
]
```

### 5.2 GeometrySpec

```python
class GeometrySpec(BaseModel):
    id: str
    source: GeometrySource
    dimension: Literal[0, 1, 2, 3]
    coordinate_system: CoordinateSystem
    entities: list[GeometryEntity]
    regions: list[RegionSpec]
    boundaries: list[BoundaryRegionSpec]
    transforms: list[GeometryTransform] = []
    encodings: list[GeometryEncoding] = []
    quality: GeometryQualityReport | None = None
```

```python
class GeometrySource(BaseModel):
    kind: Literal[
        "text",
        "cad_step",
        "cad_iges",
        "stl",
        "mesh_file",
        "cif",
        "poscar",
        "molecular_graph",
        "image",
        "generated",
    ]
    uri: str | None = None
    checksum: str | None = None
```

```python
class GeometryEncoding(BaseModel):
    kind: Literal[
        "sdf",
        "occupancy_mask",
        "surface_point_cloud",
        "volume_point_cloud",
        "mesh_graph",
        "boundary_graph",
        "laplacian_eigenbasis",
        "multi_resolution_grid",
        "parametric_shape_vector",
    ]
    uri: str
    resolution: list[int] | None = None
    feature_names: list[str] = []
    target_backend: str | None = None
```

Design note from FlowBench:

```text
For complex-geometry flow datasets, store masks/SDFs, multi-resolution fields,
geometry class, Reynolds/Grashof ranges, and engineering summary statistics.
Do not store only raw meshes.
```

### 5.3 MeshSpec

```python
class MeshSpec(BaseModel):
    id: str
    kind: Literal[
        "structured",
        "unstructured",
        "hybrid",
        "surface",
        "volume",
        "particle",
        "k_space",
        "none",
    ]
    dimension: Literal[0, 1, 2, 3]
    topology: MeshTopology
    elements: ElementStats
    regions: list[RegionSpec]
    boundaries: list[BoundaryRegionSpec]
    quality: MeshQualityReport
    files: list[ArtifactRef] = []
    solver_compatibility: list[str] = []
```

```python
class MeshQualityReport(BaseModel):
    min_jacobian: float | None = None
    max_skewness: float | None = None
    max_nonorthogonality: float | None = None
    aspect_ratio_p95: float | None = None
    boundary_layer_quality: dict[str, float] = {}
    passes: bool
    issues: list[str] = []
```

### 5.4 OperatorSpec

```python
class OperatorSpec(BaseModel):
    id: str
    name: str
    domain: PhysicsDomain
    equation_class: str
    form: Literal["strong", "weak", "integral", "discrete", "learned"]
    fields_in: list[str]
    fields_out: list[str]
    conserved_quantities: list[str] = []
    differential_terms: list[DifferentialTerm] = []
    source_terms: list[SourceTerm] = []
    coupling: list[CouplingSpec] = []
    assumptions: list[str] = []
    nondimensional_numbers: list[NondimensionalNumber] = []
```

Examples:

```text
Navier-Stokes
Heat equation
Linear elasticity
Maxwell equations
Poisson equation
Schrodinger / Kohn-Sham approximation
Reaction-diffusion
Molecular force-field dynamics
```

### 5.5 BoundaryConditionSpec

```python
class BoundaryConditionSpec(BaseModel):
    id: str
    region_id: str
    field: str
    kind: Literal[
        "dirichlet",
        "neumann",
        "robin",
        "periodic",
        "symmetry",
        "wall",
        "inlet",
        "outlet",
        "interface",
        "farfield",
        "initial",
        "custom",
    ]
    value: ScalarVectorTensor | ExpressionRef
    units: str | None = None
    confidence: float
    source: Literal["user", "inferred", "template", "retrieved"]
```

### 5.6 SolverResult

```python
class SolverResult(BaseModel):
    id: str
    problem_id: str
    backend: str
    status: Literal["success", "failed", "partial", "needs_review"]
    fields: list[FieldDataRef]
    scalar_outputs: dict[str, float | int | str]
    residuals: dict[str, float] = {}
    uncertainty: dict[str, float] = {}
    runtime: RuntimeStats
    artifacts: list[ArtifactRef] = []
    provenance: Provenance
```

---

## 6. Solver Backend Protocol

Solver Runtime is TAPS-first. TAPS is the primary equation-driven surrogate backend. Neural operators and full solvers are fallback or supporting backends.

```python
class SolverBackend(Protocol):
    name: str
    family: Literal[
        "surrogate",
        "neural_operator",
        "fem",
        "fvm",
        "md",
        "dft",
        "rom",
        "legacy",
        "custom",
    ]

    def supports(self, problem: PhysicsProblem) -> SupportScore:
        ...

    def estimate_cost(self, problem: PhysicsProblem) -> CostEstimate:
        ...

    def prepare(self, problem: PhysicsProblem) -> PreparedSolverCase:
        ...

    def solve(self, prepared: PreparedSolverCase) -> SolverResult:
        ...

    def collect_artifacts(self, result: SolverResult) -> list[ArtifactRef]:
        ...
```

```python
class SupportScore(BaseModel):
    backend: str
    score: float
    supported: bool
    reasons: list[str]
    risks: list[str]
    required_missing_inputs: list[str] = []
```

```python
class SolverDecision(BaseModel):
    selected_backend: str
    candidate_backends: list[SupportScore]
    mode: Literal["surrogate_only", "full_solver", "hybrid", "warm_start", "corrector"]
    reason: str
    expected_error: float | None = None
    expected_runtime_seconds: float | None = None
```

### 6.1 Backend Families

First-class backend order:

```text
TAPSBackend                     -> primary equation-driven surrogate / ROM
NeuralOperatorBackend
GraphNeuralOperatorBackend
MeshGraphNetBackend
CustomPythonBackend
FullSolverBackend
```

TAPS design note:

```text
TAPS-style backends are not trained only from data. They construct reduced-order
surrogates from governing equations over space-parameter-time variables. This is
valuable for ultra large-scale simulations where full field storage is impossible.
```

### 6.2 TAPS-First Solver Policy

TAPS should be attempted first when:

```text
operator is explicit
parameter axes are identifiable
geometry can be tensorized, parameterized, represented by SDF, or decomposed
solution is expected to be reasonably smooth
verification budget allows residual checks and selected slice validation
```

Use neural/full-solver fallback when:

```text
geometry cannot be converted into a TAPS-compatible domain
PDE/operator is ambiguous
solution is discontinuity-dominated
case involves high-Re turbulence, shocks, violent multiphase flow, contact, or fracture
TAPS residual or slice validation fails
```

Routing order:

```text
1. taps-agent / TAPSBackend
2. neural operator surrogate
3. full solver fallback
```

### 6.3 Open-source Full Solver Baseline

PhysicsOS should not depend on paid or closed-source full solvers. VASP is explicitly excluded from the default backend set because it is commercial/licensed software. The default full solver stack should be open-source, CLI-capable, container-friendly, and strongly connected to Python.

The baseline is split into four tiers.

#### Tier 0: Numerical and Workflow Kernels

These are not user-facing full solvers, but most backends should use or interoperate with them.

| Component | Role | Why it belongs |
| --- | --- | --- |
| `PETSc` + `petsc4py` | scalable linear/nonlinear solvers, time stepping, MPI | common HPC numerical substrate for FEM/FVM/custom PDE backends |
| `SLEPc` + `slepc4py` | scalable eigenvalue problems | modal analysis, stability, quantum/material eigenproblems |
| `Gmsh` | CLI + Python API mesh generation | default geometry/mesh bridge for CAD/STEP/STL to solver meshes |
| `meshio` | mesh format conversion | glue between Gmsh, FEniCSx, VTK/XDMF, OpenFOAM-like workflows |
| `ParaView/pvpython` + `PyVista` | postprocessing and visualization | standard CLI/Python visualization path for fields and reports |

#### Tier 1: Default Full Solver Backends

These are the default full solvers PhysicsOS should maintain first.

| Backend | Domains covered | CLI/Python fit | Decision |
| --- | --- | --- | --- |
| `OpenFOAM` | CFD, heat transfer, multiphase, turbulence, combustion-style workflows | CLI-native; Python wrapper can generate dictionaries and parse results | default heavy-duty CFD backend |
| `SU2` | CFD, aerodynamics, conjugate heat transfer, adjoint/design optimization | CLI-native with Python interface/scripts | default lightweight CFD/design backend |
| `FEniCSx/DOLFINx` | general PDE FEM: solid, thermal, diffusion, electromagnetics prototypes, coupled weak forms | Python-first; PETSc/MPI underneath | default programmable PDE backend |
| `MFEM/PyMFEM` | high-order FEM, scalable FEM, GPU/HPC-oriented custom PDEs | C++ CLI/library with Python bindings | second programmable PDE backend for high-order/HPC cases |
| `Quantum ESPRESSO` | periodic plane-wave DFT, electronic structure, phonons, materials | CLI-native; controlled through ASE/Python wrappers | default VASP replacement for periodic materials |
| `CP2K` | DFT, molecular dynamics, force fields, QM/MM, large atomistic systems | CLI-native; ASE/Python ecosystem integration | default atomistic multiphysics backend |
| `LAMMPS` | classical molecular dynamics, coarse-grained MD, materials, fluids, granular systems | CLI-native plus Python module and ASE integration | default classical MD backend |
| `Cantera` | thermodynamics, chemical kinetics, transport, reactors, 1D flames | Python-first plus CLI tools | default chemistry/combustion mechanism backend |

This Tier 1 set covers the main HPC simulation classes:

```text
continuum CFD / heat transfer        -> OpenFOAM, SU2
general PDE / FEM / multiphysics     -> FEniCSx, MFEM
electronic structure / DFT           -> Quantum ESPRESSO, CP2K
classical atomistic simulation       -> LAMMPS, CP2K
chemical kinetics / reacting systems -> Cantera
```

#### Tier 2: Optional Domain Packs

These should be added only when a user workflow needs them.

| Backend | Use when | Reason not Tier 1 |
| --- | --- | --- |
| `OpenMC` | neutron/photon transport, reactor physics, Monte Carlo particle transport | important but specialized |
| `MOOSE` | large multiphysics apps, phase field, porous flow, nuclear engineering workflows | powerful but heavier framework commitment |
| `Elmer FEM` | packaged multiphysics FEM with CLI solvers | useful broad fallback, but Python integration is weaker than FEniCSx |
| `CalculiX` | Abaqus-like structural/thermal CLI workflows | lightweight structural fallback, but less general than FEniCSx/MFEM |
| `ABINIT` | alternative open-source DFT stack | useful cross-check, but overlaps Quantum ESPRESSO/CP2K |
| `GPAW` | Python-native DFT, grid/PAW workflows | excellent Python fit, but not the first periodic-materials workhorse |

#### Tier 3: Explicitly Not Default

```text
VASP      -> paid/licensed; may be user-provided plugin only
COMSOL    -> proprietary GUI/license; not default
Abaqus    -> proprietary/license; not default
ANSYS     -> proprietary/license; not default
STAR-CCM+ -> proprietary/license; not default
```

If an enterprise user owns these solvers, PhysicsOS can support them through private plugins. They must not be part of the open-source default distribution.

### 6.4 Backend Selection Rules

Use these defaults:

```text
parameterized thermal/diffusion/Poisson     -> TAPS first
reaction-diffusion / Helmholtz              -> TAPS first, neural fallback
linear elasticity / thermoelastic coupling  -> TAPS first where geometry permits
CFD with complex industrial physics       -> OpenFOAM
external aero / shape optimization        -> SU2
custom weak-form PDE / research PDE       -> FEniCSx
high-order FEM / GPU/HPC finite elements  -> MFEM
periodic DFT materials                    -> Quantum ESPRESSO
large atomistic / QM/MM / mixed DFT+MD    -> CP2K
classical MD                              -> LAMMPS
chemical kinetics / reactor mechanisms    -> Cantera
nuclear particle transport                -> OpenMC optional pack
```

Python integration policy:

```text
Prefer Python-native APIs when available.
For CLI solvers, PhysicsOS owns the input generator, runner, parser, and artifact collector.
All solver adapters expose the same SolverBackend protocol.
All external binaries must be invoked through sandboxed runner services, not arbitrary shell strings.
```

### 6.5 Hybrid Solve Policy

```text
try TAPS
→ compute S-P-T Galerkin residual
→ validate selected parameter/time slices
→ if trusted: accept TAPS result
→ if not trusted: run neural operator or full solver fallback
→ store TAPS factors, residual history, and validation slices
```

Fallback policy:

```text
run neural operator
→ compute physics residuals
→ estimate uncertainty
→ detect OOD geometry/regime
→ if trusted: accept
→ if medium risk: run local correction or low-fidelity full solver
→ if high risk: ask approval and run full solver
→ store result in case memory
```

---

## 7. Tool Registry

Every new tool must have:

```text
stable name
single responsibility
Pydantic input schema
Pydantic output schema
explicit side effects
approval requirement
artifact paths
failure modes
```

### 7.1 Geometry Tools

```python
class ImportGeometryInput(BaseModel):
    source: GeometrySource
    target_units: str = "SI"

class ImportGeometryOutput(BaseModel):
    geometry: GeometrySpec
    artifacts: list[ArtifactRef]

def import_geometry(input: ImportGeometryInput) -> ImportGeometryOutput:
    """Import CAD/mesh/material geometry into GeometrySpec."""
```

```python
class RepairGeometryInput(BaseModel):
    geometry: GeometrySpec
    repair_policy: Literal["conservative", "aggressive", "manual_review"]

class RepairGeometryOutput(BaseModel):
    geometry: GeometrySpec
    changes: list[str]
    warnings: list[str]

def repair_geometry(input: RepairGeometryInput) -> RepairGeometryOutput:
    """Repair invalid, non-manifold, open, or self-intersecting geometry."""
```

```python
class LabelRegionsInput(BaseModel):
    geometry: GeometrySpec
    physics_domain: PhysicsDomain
    hints: list[str] = []

class LabelRegionsOutput(BaseModel):
    geometry: GeometrySpec
    confidence_by_region: dict[str, float]
    unresolved_regions: list[str]

def label_regions(input: LabelRegionsInput) -> LabelRegionsOutput:
    """Infer physical regions and boundary labels."""
```

```python
class GenerateGeometryEncodingInput(BaseModel):
    geometry: GeometrySpec
    encodings: list[str]
    resolutions: list[list[int]] = []

class GenerateGeometryEncodingOutput(BaseModel):
    encodings: list[GeometryEncoding]
    artifacts: list[ArtifactRef]

def generate_geometry_encoding(input: GenerateGeometryEncodingInput) -> GenerateGeometryEncodingOutput:
    """Generate SDF, masks, graph, point cloud, Laplacian eigenbasis, or multiresolution grid encodings."""
```

### 7.2 Mesh Tools

```python
class GenerateMeshInput(BaseModel):
    geometry: GeometrySpec
    physics: PhysicsSpec
    mesh_policy: MeshPolicy
    target_backends: list[str]

class GenerateMeshOutput(BaseModel):
    mesh: MeshSpec
    artifacts: list[ArtifactRef]

def generate_mesh(input: GenerateMeshInput) -> GenerateMeshOutput:
    """Generate solver-compatible mesh from GeometrySpec."""
```

```python
class AssessMeshQualityInput(BaseModel):
    mesh: MeshSpec
    physics: PhysicsSpec
    backend: str | None = None

class AssessMeshQualityOutput(BaseModel):
    report: MeshQualityReport
    recommended_action: Literal["accept", "refine", "remesh", "manual_review"]

def assess_mesh_quality(input: AssessMeshQualityInput) -> AssessMeshQualityOutput:
    """Evaluate mesh quality for the selected physics and backend."""
```

```python
class ExportBackendMeshInput(BaseModel):
    mesh: MeshSpec
    backend: str
    output_dir: str

class ExportBackendMeshOutput(BaseModel):
    artifacts: list[ArtifactRef]

def export_backend_mesh(input: ExportBackendMeshInput) -> ExportBackendMeshOutput:
    """Export mesh into OpenFOAM/SU2/FEniCSx/MFEM/LAMMPS/Quantum ESPRESSO/CP2K-compatible formats."""
```

### 7.3 Problem and Operator Tools

```python
class BuildPhysicsProblemInput(BaseModel):
    user_request: str
    geometry: GeometrySpec | None = None
    uploaded_artifacts: list[ArtifactRef] = []
    assumptions: list[str] = []

class BuildPhysicsProblemOutput(BaseModel):
    problem: PhysicsProblem
    missing_inputs: list[str]
    assumptions: list[str]

def build_physics_problem(input: BuildPhysicsProblemInput) -> BuildPhysicsProblemOutput:
    """Convert user intent and artifacts into PhysicsProblem IR."""
```

```python
class ValidatePhysicsProblemInput(BaseModel):
    problem: PhysicsProblem

class ValidatePhysicsProblemOutput(BaseModel):
    valid: bool
    errors: list[str]
    warnings: list[str]

def validate_physics_problem(input: ValidatePhysicsProblemInput) -> ValidatePhysicsProblemOutput:
    """Validate that the problem has geometry, fields, operators, materials, BCs, targets, and verification policy."""
```

### 7.4 Solver Tools

```python
class EstimateSolverSupportInput(BaseModel):
    problem: PhysicsProblem
    candidate_backends: list[str] = []

class EstimateSolverSupportOutput(BaseModel):
    scores: list[SupportScore]

def estimate_solver_support(input: EstimateSolverSupportInput) -> EstimateSolverSupportOutput:
    """Score solver backends for a PhysicsProblem."""
```

```python
class RouteSolverBackendInput(BaseModel):
    problem: PhysicsProblem
    support_scores: list[SupportScore]
    policy: SolverPolicy

class RouteSolverBackendOutput(BaseModel):
    decision: SolverDecision

def route_solver_backend(input: RouteSolverBackendInput) -> RouteSolverBackendOutput:
    """Select surrogate, full solver, hybrid, warm-start, or corrector mode."""
```

```python
class RunSurrogateSolverInput(BaseModel):
    problem: PhysicsProblem
    backend: str
    checkpoint: str | None = None

class RunSurrogateSolverOutput(BaseModel):
    result: SolverResult

def run_surrogate_solver(input: RunSurrogateSolverInput) -> RunSurrogateSolverOutput:
    """Run neural operator, graph neural operator, MeshGraphNet, or ROM surrogate."""
```

```python
class RunFullSolverInput(BaseModel):
    problem: PhysicsProblem
    backend: str
    budget: ComputeBudget
    approval_token: str | None = None

class RunFullSolverOutput(BaseModel):
    result: SolverResult

def run_full_solver(input: RunFullSolverInput) -> RunFullSolverOutput:
    """Run trusted open-source full solver backend such as OpenFOAM, SU2, FEniCSx, MFEM, Quantum ESPRESSO, CP2K, LAMMPS, Cantera, or OpenMC."""
```

```python
class RunHybridSolverInput(BaseModel):
    problem: PhysicsProblem
    surrogate_backend: str
    full_backend: str
    hybrid_policy: HybridPolicy

class RunHybridSolverOutput(BaseModel):
    result: SolverResult
    stages: list[SolverResult]

def run_hybrid_solver(input: RunHybridSolverInput) -> RunHybridSolverOutput:
    """Use surrogate as fast solver, warm start, or corrector around full solver."""
```

### 7.5 Verification Tools

```python
class ComputePhysicsResidualsInput(BaseModel):
    problem: PhysicsProblem
    result: SolverResult

class ComputePhysicsResidualsOutput(BaseModel):
    residuals: dict[str, float]
    normalized_residuals: dict[str, float]
    passes: bool

def compute_physics_residuals(input: ComputePhysicsResidualsInput) -> ComputePhysicsResidualsOutput:
    """Compute PDE/operator residuals for solver output."""
```

```python
class EstimateUncertaintyInput(BaseModel):
    problem: PhysicsProblem
    result: SolverResult
    method: Literal["ensemble", "dropout", "conformal", "residual_proxy", "backend_reported"]

class EstimateUncertaintyOutput(BaseModel):
    uncertainty: dict[str, float]
    confidence: float

def estimate_uncertainty(input: EstimateUncertaintyInput) -> EstimateUncertaintyOutput:
    """Estimate predictive uncertainty for fields and KPIs."""
```

```python
class DetectOODCaseInput(BaseModel):
    problem: PhysicsProblem
    reference_scope: Literal["model_training_set", "case_memory", "both"]

class DetectOODCaseOutput(BaseModel):
    ood_score: float
    reasons: list[str]
    nearest_cases: list[str]

def detect_ood_case(input: DetectOODCaseInput) -> DetectOODCaseOutput:
    """Detect out-of-distribution geometry, regime, material, or boundary conditions."""
```

### 7.6 Postprocess Tools

```python
class ExtractKPIsInput(BaseModel):
    problem: PhysicsProblem
    result: SolverResult

class ExtractKPIsOutput(BaseModel):
    kpis: dict[str, float | str]
    units: dict[str, str]

def extract_kpis(input: ExtractKPIsInput) -> ExtractKPIsOutput:
    """Extract engineering metrics such as drag, lift, pressure drop, max temperature, stress, band gap, or adsorption energy."""
```

```python
class GenerateVisualizationsInput(BaseModel):
    problem: PhysicsProblem
    result: SolverResult
    visualization_plan: list[VisualizationSpec]

class GenerateVisualizationsOutput(BaseModel):
    artifacts: list[ArtifactRef]

def generate_visualizations(input: GenerateVisualizationsInput) -> GenerateVisualizationsOutput:
    """Generate plots, contours, streamlines, isosurfaces, DOS, band structures, or stress maps."""
```

```python
class WriteSimulationReportInput(BaseModel):
    problem: PhysicsProblem
    result: SolverResult
    verification: VerificationReport
    postprocess: PostprocessResult
    format: Literal["markdown", "pdf", "html", "pptx"]

class WriteSimulationReportOutput(BaseModel):
    report: ArtifactRef

def write_simulation_report(input: WriteSimulationReportInput) -> WriteSimulationReportOutput:
    """Generate final simulation report with assumptions, results, uncertainty, and recommended next actions."""
```

### 7.7 Memory Tools

```python
class SearchCaseMemoryInput(BaseModel):
    problem: PhysicsProblem
    top_k: int = 5
    filters: dict[str, str | float | int] = {}

class SearchCaseMemoryOutput(BaseModel):
    cases: list[CaseMemoryHit]

def search_case_memory(input: SearchCaseMemoryInput) -> SearchCaseMemoryOutput:
    """Retrieve similar historical cases by geometry, BCs, parameters, physics, and target KPIs."""
```

```python
class StoreCaseResultInput(BaseModel):
    problem: PhysicsProblem
    result: SolverResult
    verification: VerificationReport
    postprocess: PostprocessResult | None = None

class StoreCaseResultOutput(BaseModel):
    case_id: str
    indexed_features: list[str]

def store_case_result(input: StoreCaseResultInput) -> StoreCaseResultOutput:
    """Store validated simulation results for retrieval, warm start, and future training."""
```

---

## 8. Verification Layer

Verification is not optional. It is the trust boundary between demo and engineering use.

```python
class VerificationReport(BaseModel):
    problem_id: str
    result_id: str
    status: Literal["accepted", "accepted_with_warnings", "rejected", "needs_full_solver", "needs_user_input"]
    residuals: dict[str, float]
    conservation_errors: dict[str, float]
    uncertainty: dict[str, float]
    ood_score: float
    mesh_quality: MeshQualityReport | None
    nearest_reference_cases: list[str] = []
    recommended_next_action: Literal[
        "accept",
        "refine_mesh",
        "rerun_surrogate",
        "run_full_solver",
        "run_higher_fidelity_solver",
        "ask_user_for_missing_input",
    ]
    explanation: str
```

Every solver result must pass through:

```text
physics residual check
conservation check
mesh quality check
uncertainty estimate
OOD detection
reference case comparison when available
```

---

## 9. Case Memory and Data Flywheel

Every completed case becomes training and retrieval material.

```python
class CaseRecord(BaseModel):
    id: str
    problem: PhysicsProblem
    geometry_embeddings: list[EmbeddingRef]
    operator_embeddings: list[EmbeddingRef]
    mesh: MeshSpec | None
    result: SolverResult
    verification: VerificationReport
    postprocess: PostprocessResult | None
    dataset_tags: list[str] = []
    usage_rights: str
```

Index by:

```text
geometry similarity
boundary condition similarity
material similarity
operator similarity
dimensionless numbers
mesh topology
solver backend
error/uncertainty metadata
engineering KPI similarity
```

FlowBench design consequence:

```text
The memory layer must support complex geometries, 2D/3D samples, multiple
resolutions, velocity/pressure/temperature fields, Reynolds/Grashof regimes,
and engineering statistics. This makes it useful both as retrieval memory and
as neural-operator benchmark/training data.
```

TAPS design consequence:

```text
The memory layer should store separable parameter axes and reduced bases when
available, not just dense field snapshots. Otherwise ultra large-scale cases
become impossible to store or reuse efficiently.
```

TAPS-specific memory artifacts:

```text
TAPSProblem
operator weak form
parameter axes
C-HiDeNN basis config
CP rank / factor matrices
subspace iteration history
residual history
selected validation slices
reconstruction metadata
```

---

## 10. TAPS As Universal Equation-Driven Solver Compiler

The solver model should be TAPS-first. TAPS-agent is not a fixed list of PDE templates. It is the primary equation-driven solver compiler:

```text
PhysicsProblem
-> formulate governing equations
-> request missing science from knowledge-agent when needed
-> compile weak-form IR
-> choose space / parameter / time / geometry axes
-> choose C-HiDeNN basis and tensor rank
-> assemble or dispatch executable TAPS kernel
-> compute residuals and validation slices
-> accept, refine, or fallback
```

Unknown physics should not immediately route to solver-agent. Unknown physics should first become a knowledge problem:

```text
missing governing equation
missing constitutive relation
missing nondimensional regime
missing boundary condition
missing validation residual
missing material law
```

TAPS-agent must produce precise `required_knowledge_queries` for knowledge-agent, then recompile once the missing scientific context is retrieved.

Core TAPS IR:

```text
TAPSCompilationPlan
TAPSWeakFormSpec
TAPSEquationTerm
TAPSAxisSpec
TAPSBasisConfig
TAPSNonlinearConfig
TAPSGeometryEncodingSpec
TAPSCoefficientSpec
TAPSBoundaryConditionSpec
TAPSProblem
TAPSResidualReport
TAPSRuntimeExtensionSpec
```

When the missing capability is implementation rather than science, taps-agent may author a reviewed case-local `TAPSRuntimeExtensionSpec` artifact. This lets it flexibly prototype missing kernels without silently modifying PhysicsOS core runtime:

```text
unknown but well-specified operator
-> formulate weak-form IR
-> identify missing assembler/kernel
-> author case-local runtime extension draft
-> require review before execution/promotion
-> promote to core backend only after tests
```

Current executable kernel:

```text
1D transient heat equation low-rank S-P-T backend
1D multi-mode scalar elliptic weak-form assembler for Poisson / steady diffusion / reaction-diffusion / Helmholtz-style operators
2D multi-mode tensorized scalar elliptic weak-form assembler for rectangular Poisson / steady diffusion / reaction-diffusion / Helmholtz-style operators
1D/2D nonlinear reaction-diffusion Picard/fixed-point kernels with iteration history
2-field 2D coupled reaction-diffusion kernel with coupled residual and field artifacts
2D occupancy-mask geometry-encoded scalar elliptic execution path, including central-hole masks with masked relaxation
Gmsh mesh -> mesh_graph encoding -> triangle P1/P2 FEM-like Poisson execution path
triangle cell gradients -> sparse stiffness matrix -> Dirichlet Poisson solve
Gmsh mesh -> mesh_graph encoding -> triangle P1/P2/P3 vector FEM-like linear-elasticity execution path
constant-strain P1 / quadratic-cubic Lagrange P2-P3 triangle -> vector sparse stiffness matrix -> clamped-boundary displacement solve
MaterialSpec -> TAPSCoefficientSpec -> backend material/source coefficients for elasticity
Gmsh mesh -> mesh_graph encoding -> 2D first/second-order Nedelec EM curl-curl path
edge DOFs + orientation signs -> H(curl) tangential continuity -> residual-checked EM edge-field artifact
second-order Nedelec H(curl) executable scaffold -> two edge moment DOFs per edge + cell interior DOFs -> local curl-curl/mass assembly
PhysicsProblem boundary_conditions -> TAPSBoundaryConditionSpec -> PEC/natural EM edge-boundary policy
complex-valued EM coefficients -> `[real, imag]` JSON representation -> complex frequency-domain edge solve
custom/robin EM boundary value dict -> absorbing impedance / port excitation -> boundary-edge Robin contribution
mesh_graph boundary_edge_sets -> region_id-specific PEC / absorbing / port edge selection -> matched high-order Nedelec boundary DOFs
generated Gmsh physical groups -> meshio field_data/cell_data -> named boundary edge sets
imported Gmsh .geo physical curves -> GeometrySpec boundaries + mesh_graph named boundary edge sets -> EM region-specific ports/walls
mesh_graph physical_boundary_groups -> OpenFOAM patch / SU2 marker / FEniCSx facet-tag export hints
export_backend_mesh -> solver-facing manifest without local external conversion execution
3D Gmsh physical surfaces -> mesh_graph boundary_face_sets / physical_boundary_groups.face_ids -> solver patch/marker/facet export manifests
boundary_labeling_artifact -> viewer/CLI selectable face/edge groups + weak suggestions requiring explicit confirmation
create_geometry_labeler_viewer -> standalone HTML mesh/facet viewer -> confirm face/edge physical groups -> export confirmed_boundary_labels JSON
confirmed_boundary_labels -> apply_boundary_labeling_artifact -> GeometrySpec.boundaries for solver export
backend mesh export manifest -> mesh_conversion_runner_manifest with inline .msh + boundary mapping -> dry-run/http runner submit contract
```

Current compiler IR coverage:

```text
custom weak-form terms from OperatorSpec
custom scalar elliptic weak-form IR terms mapped to reusable diffusion / reaction / source assembler blocks
custom vector elasticity weak-form IR terms mapped to reusable strain-energy / body-force assembler blocks
custom H(curl) curl-curl weak-form IR terms mapped to reusable curl-curl / mass / source assembler blocks
custom H(div) divergence/mass/source weak-form IR terms mapped to tetrahedral RT0 face-flux scaffold blocks
custom nonlinear reaction-diffusion weak-form IR terms mapped to reusable diffusion / nonlinear-reaction / source Picard blocks
custom coupled-field weak-form IR terms mapped to reusable field-diffusion / coupling-operator / reaction / source subspace-solver blocks
custom transient diffusion weak-form IR terms mapped to reusable time-derivative / mass / diffusion / source time-integration blocks
symbolic TAPS IR validation for field, coefficient, initial-condition, geometry-encoding, and function-space compatibility
safe TAPS backend bridge manifest export for FEniCSx / MFEM / PETSc review without executing external solvers
reviewable FEniCSx / PyMFEM / petsc4py draft artifacts generated from TAPS IR bridge manifests
adaptive fallback decision artifacts for knowledge-agent, runtime-extension, backend-bridge, or full-solver preparation
auditable backend case bundles with dependency checks, mesh-export requirements, coefficient binding, boundary binding, and approval gates
strong-form diffusion/time/reaction/curl token patterns compiled into weak-form IR terms for reviewed execution
Neumann / Robin / interface / custom boundary conditions compiled into first-class boundary weak-term IR metadata
heat / diffusion
Poisson
reaction-diffusion
Helmholtz
knowledge-assisted unspecified physics
```

Next implementation requirement:

```text
harden reviewed backend drafts into auditable executable exports with dependency,
mesh, coefficient, boundary, and safety gates.

near-term executable IR extensions:
1. transient mass/time-derivative blocks:
   time_derivative + mass + stiffness/source -> M du/dt + K u = f
   connect to explicit/implicit Euler, Crank-Nicolson, BDF-style time integrators
   consume InitialConditionSpec and emit time-slice solution/residual histories
2. coupled-field blocks:
   field_block + self_operator + coupling_operator + cross_jacobian metadata
   connect to monolithic block solve, block Gauss-Seidel, and operator-splitting policies
3. boundary weak-term blocks:
   Neumann / Robin / impedance / port / interface terms as first-class weak-form blocks
   avoid hard-coded boundary semantics inside solver-specific kernels
4. strong-form compiler:
   normalize strong-form PDEs and paper equations into weak-form/discrete-residual IR before execution
   use knowledge-agent when integration by parts, constitutive laws, or function spaces are ambiguous
5. symbolic validation:
   check dimensions, field ranks, coefficient availability, boundary compatibility, and function-space requirements
6. adaptive fallback:
   if IR cannot be safely mapped, ask knowledge-agent or author a reviewed runtime extension instead of guessing
7. real backend bridge:
   export IR blocks to FEniCSx/UFL, MFEM/PyMFEM, PETSc, or other high-fidelity open backends
```

Neural/operator backends remain fallback, warm start, and correction engines.

Required neural backend classes:

```text
GridNeuralOperatorBackend       -> regular grids / FNO-like tasks
GeometryInformedNOBackend       -> complex 3D geometry with geometry encodings
MeshGraphOperatorBackend        -> arbitrary meshes / graph message passing
ManifoldOperatorBackend         -> surface/solid manifold domains
TAPSBackend                     -> equation-driven a priori surrogate / ROM
PINNCorrectorBackend            -> residual correction / data-sparse regimes
```

Routing features:

```text
geometry encoding type
mesh topology
operator family
Re / Gr / Mach / Peclet / Knudsen / other nondimensional numbers
steady vs transient
linear vs nonlinear
single physics vs coupled multiphysics
training distribution distance
target accuracy
runtime budget
```

TAPS routing features:

```text
operator explicitness
parameter-axis quality
geometry tensorizability
boundary-condition compatibility
expected smoothness
rank budget
residual tolerance
slice-validation budget
```

---

## 11. Universal Workflow

```text
User
↓
PhysicsOS Main Agent
↓
build_physics_problem
↓
geometry-mesh-agent
↓
validate_physics_problem
↓
knowledge-agent
↓
taps-agent
↓
if TAPS rejected or fails verification: solver-agent
↓
solver-agent
↓
verification-agent
↓
if rejected: refine mesh / run full solver / ask user
↓
postprocess-agent
↓
store_case_result
↓
final answer / report / optimization proposal
```

This workflow can cover:

```text
CFD
thermal simulation
solid mechanics
electromagnetics
acoustics
reactive transport
molecular dynamics
DFT / electronic structure
multiphysics coupling
```

Different physics domains differ only by:

```text
operator specs
material models
boundary conditions
geometry encodings
solver backends
verification rules
postprocess templates
```

Knowledge rule:

```text
All agents should obtain external scientific knowledge through knowledge-agent.
knowledge-agent owns local KB search, arXiv discovery, DeepSearch synthesis, prior cases, and paper notes.
Other agents should not invent citations, solver capabilities, or physical assumptions without knowledge-agent context.
```

Current knowledge integrations:

```text
local SQLite/FTS knowledge base -> search_knowledge_base
official arXiv Atom API         -> search_arxiv
OpenAI-compatible DeepSearch    -> run_deepsearch with gemini-3-pro-deepsearch-async
combined retrieval              -> build_knowledge_context
```

MCP status:

```text
No arXiv MCP server is configured in this workspace yet.
When an arXiv MCP server is added, it should be wrapped behind the same search_arxiv contract.
Agent-facing interfaces should not depend on whether arXiv is reached through MCP or the official Atom API.
```

---

## 12. First MVP

Do not start with all physics. Start with a path that validates the whole OS loop.

First MVP:

```text
TAPS-first thermal / diffusion over parameterized or tensorized geometries
```

Scope:

```text
CAD/STL import
geometry repair
region and boundary labeling
mesh generation
SDF/mask/mesh-graph encoding
TAPSProblem compilation
TAPS-first thermal/diffusion solve
neural operator fallback
FEniCSx/OpenFOAM selected-slice fallback
verification report
engineering report
case memory storage
```

Target tasks:

```text
PCB cooling
heat sink cooling
duct flow
pipe pressure drop
HVAC
forced/free convection around complex shapes
```

This MVP directly exercises:

```text
Geometry + Mesh Agent
Solver Model / Solver Runtime
Postprocess + Analysis Agent
Verification
Case Memory
DeepAgents orchestration
```

---

## 13. Roadmap

### Stage 1: DeepAgents Skeleton

Deliver:

```text
create_deep_agent bootstrap
subagent specs
Pydantic schemas
tool registry
structured outputs
case workspace backend
human approval for full solver/HPC
```

### Stage 2: TAPS Thermal MVP

Deliver:

```text
geometry import/repair/label
mesh generation and quality reports
SDF/mask/graph encodings
TAPSProblem builder
TAPS thermal/diffusion backend scaffold
rank/residual refinement policy
neural operator fallback
selected-slice full solver validation
verification report
postprocess report
```

### Stage 3: General Operator Runtime

Deliver:

```text
OperatorSpec registry
SolverBackend registry
VerificationRule registry
PostprocessTemplate registry
case memory retrieval
```

### Stage 4: TAPS + Physics-MoE Solver

Deliver:

```text
geometry-aware neural operator experts
mesh-aware graph operator experts
TAPS-style a priori surrogate backend
OOD detection
RL/runtime routing policy
full-solver fallback
```

### Stage 5: Cross-domain PhysicsOS

Deliver:

```text
CFD + thermal + solid + EM + materials
unified geometry/mesh/solver/postprocess workflow
legacy solvers as backend engines
agent-facing product interface
```

---

## 14. Product Positioning

PhysicsOS should be positioned as:

```text
Cursor for simulation engineers
```

Not:

```text
We replace OpenFOAM, Quantum ESPRESSO, CP2K, LAMMPS, COMSOL immediately.
```

Instead:

```text
simple and common cases -> AI-native surrogate solve
hard and high-risk cases -> orchestrate trusted full solver
all cases -> unified workflow + verification + report
```

The long-term endpoint is that traditional solvers become backend engines. Users interact with PhysicsOS, not with solver-specific input files.

---

## 15. References

- DeepAgents repository: https://github.com/langchain-ai/deepagents
- DeepAgents documentation: https://docs.langchain.com/oss/python/deepagents/overview
- FlowBench: A Large Scale Benchmark for Flow Simulation over Complex Geometries: https://arxiv.org/abs/2409.18032
- Tensor-decomposition-based A Priori Surrogate (TAPS) modeling for ultra large-scale simulations: https://arxiv.org/abs/2503.13933
- OpenFOAM Foundation: https://openfoam.org/
- SU2: https://su2code.github.io/
- DOLFINx/FEniCSx: https://docs.fenicsproject.org/dolfinx/main/python/
- MFEM/PyMFEM: https://mfem.org/
- PETSc/petsc4py: https://petsc.org/release/petsc4py/
- Gmsh: https://gmsh.info/
- meshio: https://pypi.org/project/meshio/
- Quantum ESPRESSO: https://www.quantum-espresso.org/
- CP2K: https://www.cp2k.org/
- LAMMPS: https://www.lammps.org/
- ASE: https://ase-lib.org/
- Cantera: https://cantera.org/
- OpenMC: https://openmc.org/

---

## 16. Implementation Status

Current implemented baseline:

```text
Stage 1 DeepAgents Skeleton
-> create_physicsos_agent bootstrap
-> subagent specs
-> Pydantic schemas
-> registered PhysicsOS tools with input/output contracts
-> OpenAI-compatible runtime configuration
-> local knowledge base, arXiv API wrapper, DeepSearch wrapper
```

```text
Stage 2 TAPS Thermal MVP
-> optional Gmsh/meshio geometry-mesh backend
-> generated primitive mesh smoke path
-> deterministic 1D transient heat equation PhysicsProblem
-> TAPSProblem builder
-> pure-Python equation-driven low-rank S-P-T thermal backend
-> pure-Python 1D multi-mode scalar weak-form assembler for first generic TAPS IR execution
-> pure-Python 2D multi-mode tensorized scalar weak-form assembler for rectangular domains
-> pure-Python 1D/2D nonlinear reaction-diffusion kernels with warm-start and iteration residuals
-> pure-Python 2-field 2D coupled reaction-diffusion kernel
-> geometry encoding generation for TAPS occupancy masks
-> mesh_graph encoding generation from Gmsh/meshio mesh artifacts
-> conservative generated-geometry boundary labeling for geometry-agent
-> meshio-backed triangle mesh quality metrics: Jacobian, aspect ratio, skewness proxy
-> TAPSProblem carries geometry encodings into executable kernels
-> masked finite-difference relaxation for irregular/hole occupancy domains
-> triangle P1/P2 FEM-like Poisson kernel for mesh_graph domains
-> local triangle gradient assembly into sparse stiffness matrices
-> second-order Gmsh triangle6 mesh support via MeshPolicy.element_order
-> generic triangle Lagrange Pk basis scaffold with P3 stiffness/mass assembly tests
-> vector-valued 2D linear-elasticity weak form, P1/P2/P3 triangle assembly, and mesh_graph displacement solve
-> MaterialSpec property compilation into TAPSCoefficientSpec for linearly elastic E/nu/stress model/body-force coefficients
-> electromagnetic Maxwell/curl-curl weak-form routing and first-order Nedelec edge-element mesh_graph execution
-> TAPSBoundaryConditionSpec compilation with EM PEC tangential-zero and natural farfield/symmetry boundary policies
-> complex-valued EM frequency-domain coefficients for lossy permittivity/permeability/source terms
-> EM absorbing/impedance/port boundary policies with edge Robin contribution and port RHS injection
-> region-specific EM boundary edge selection from mesh_graph bbox/physical boundary sets
-> generated 2D Gmsh physical-group propagation for domain/x_min/x_max/y_min/y_max
-> reviewed case-local runtime extension artifact for taps-agent generated prototype code
-> factor_matrices / reconstruction_metadata / residual_history artifacts
-> TAPS residual verification
-> backend residual extraction, residual-proxy uncertainty, and heuristic OOD detection
-> conservation-law metric checks and selected-slice validation artifacts
-> sandbox full-solver fallback manifest for external runner/service execution
-> foamvm/E2B OpenFOAM external runner smoke path
-> postprocess KPI extraction
-> residual-summary JSON and solution-preview SVG visualization artifacts
-> richer markdown report generation with executive summary, verification appendix, embedded visualization links, and machine-readable manifest
-> JSONL case memory store/retrieval with domain/operator/geometry/material/BC similarity features
-> Stage 3 runtime registries for operator templates, solver backends, verification rules, and postprocess templates
-> local universal workflow mirroring the DeepAgents tool sequence
```

The runnable local OS loop is:

```text
run_physicsos_workflow
-> problem ready
-> knowledge-agent context
-> validate_physics_problem
-> taps-agent support check
-> taps-agent solve
-> verification
-> postprocess-agent report
-> case-memory store
```

Implementation status and next work:

```text
done: geometry/mesh agent generated boundary labels, imported Gmsh physical-group labels, explicit boundary-label application, and local triangle mesh quality metrics.
done: TAPS Poisson/diffusion mesh FEM-like P1/P2 assembly, reaction-diffusion, coupled reaction-diffusion, Helmholtz-family routing.
done: backend residual extraction, conservation metric checks, selected-slice artifacts, residual-proxy uncertainty, and first-pass OOD heuristics.
done: sandboxed full-solver fallback manifest plus dry-run/http remote runner adapter.
done: external foamvm / PhysicsOS Cloud runner service with device-code CLI auth, `/api/physicsos/jobs`, and OpenFOAM/E2B execution.
done: postprocess residual-summary and solution-preview visualization artifacts.
done: richer report artifacts with embedded visualization references, artifact manifest, solver provenance, and validation appendix.
done: JSONL case memory store/retrieve tools with explainable similarity scoring and filter support.
done: OperatorSpec registry, SolverBackend registry, VerificationRule registry, and PostprocessTemplate registry exposed as agent tools.
done: higher-order scalar FEM foundation beyond P2 via Vandermonde triangle Lagrange basis and P3 stiffness/mass assembly.
done: vector-valued FEM foundation via 2D P1/P2/P3 linear elasticity weak-form routing and triangle stiffness assembly.
done: MaterialSpec-to-TAPS coefficient compilation for elasticity material/source parameters.
done: electromagnetic curl-curl first H(curl) executable path via 2D first-order Nedelec edge elements on mesh_graph.
done: second-order Nedelec H(curl) local assembler scaffold with edge moment DOFs, cell interior DOFs, orientation signs, and curl-curl/mass matrices.
done: EM boundary-condition semantics for Nedelec edge DOFs: PEC tangential-zero fixes boundary edges; natural/farfield leaves edge DOFs free.
done: complex-valued EM frequency-domain coefficient support with JSON `[real, imag]` artifacts and complex edge-field solve.
done: EM absorbing/impedance/port boundary policies via structured boundary value dictionaries.
done: region-specific EM boundary edge selection using mesh_graph `boundary_edge_sets` for bbox and Gmsh physical line groups.
done: second-order Nedelec scaffold is wired into the executable EM curl-curl solve path with edge-moment/cell-interior DOFs and high-order boundary DOF selection.
done: generated Gmsh rectangle physical groups propagate through meshio into named mesh_graph boundary edge sets.
done: imported Gmsh `.geo` physical curve labels propagate into `GeometrySpec.boundaries`, `mesh_graph.boundary_edge_sets`, and EM port/wall edge selection.
done: mesh_graph physical boundary metadata carries solver-native export hints for OpenFOAM patches, SU2 markers, and FEniCSx facet tags.
done: `export_backend_mesh` produces OpenFOAM/SU2/FEniCSx/MFEM/TAPS mesh export manifests with physical boundary mappings and no unapproved local external conversion.
done: Gmsh/meshio-first 3D facet layer propagates physical surfaces into `boundary_face_sets`, `physical_boundary_groups.face_ids`, and solver export manifests.
done: user-confirmed physical-group labeling artifacts separate weak suggestions from confirmed labels before GeometrySpec mutation or solver export.
done: standalone geometry labeler viewer tool can load boundary-labeling artifacts, rotate mesh/facet previews, select groups, and export confirmed labels without foamvm, database, or E2B.
done: mesh conversion runner manifests can be prepared from backend mesh export manifests, inline source `.msh`, and dry-run/http submitted without local external conversion.
done: foamvm/E2B runner dispatches `physicsos.mesh_conversion_job.v1` manifests, decodes inline `.msh`, writes boundary mapping artifacts, and runs approved converter commands inside E2B.
done: custom scalar elliptic weak-form IR can execute through reusable diffusion/reaction/source assembler blocks without relying on a fixed PDE family label.
done: custom vector elasticity weak-form IR can execute through reusable strain-energy/body-force assembler blocks on mesh_graph geometry.
done: custom H(curl) curl-curl weak-form IR can execute through reusable curl-curl/mass/source blocks on the Nedelec mesh_graph EM path.
done: custom H(div) / Darcy / mixed-Poisson weak-form IR can execute through a tetrahedral RT0 face-flux scaffold path with typed `mesh_fem_hdiv_div` planning, validation, backend dispatch, operator/flux/residual artifacts, and explicit scaffold metadata.
done: custom nonlinear reaction-diffusion weak-form IR can execute through reusable diffusion/nonlinear-reaction/source Picard blocks.
done: custom coupled-field weak-form IR can execute through reusable field-diffusion/coupling/reaction/source blocks on the 2-field subspace solver path.
done: custom transient diffusion weak-form IR can execute through reusable time-derivative/mass/diffusion/source blocks on the low-rank S-P-T time path.
done: fluid Navier-Stokes now compiles into velocity-pressure incompressible weak-form IR with viscous, pressure-coupling, continuity, and optional advection terms.
done: Phase 1 fluid TAPS execution supports steady low-Re incompressible Stokes as a conservative Navier-Stokes simplification. The executable kernel is a 2D pressure-driven channel / Poiseuille-like solve with velocity-pressure output, continuity/momentum residual metrics, and explicit `full_navier_stokes_supported=0` metadata.
done: Phase 2 fluid TAPS execution supports Oseen / linearized Navier-Stokes when the problem provides a frozen convective velocity from user input, a previous iterate, or a surrogate warm start. The executable kernel is a linear 2D channel reduction with velocity-pressure output, Oseen residual metadata, explicit frozen-convection assumptions, and `full_navier_stokes_supported=0`.
done: Phase 3 fluid TAPS execution supports a restricted steady laminar 2D channel Navier-Stokes kernel using Picard fixed-point iteration with under-relaxation, nonlinear residual history, continuity/momentum residual metadata, and explicit `support_scope=restricted_steady_laminar_2d_channel`. It only passes validation when channel boundary data, viscosity, density, pressure drop, and low/moderate Reynolds scope are available.
done: mesh-based incompressible Navier-Stokes IR with `mesh_graph` now exports reviewed backend bridge manifests and case bundles for FEniCSx/OpenFOAM/SU2 instead of pretending local TAPS can execute arbitrary CFD. The bridge records mixed velocity-pressure/FVM field requirements, pressure-velocity coupling options, SUPG/PSPG/projection or solver-native stabilization review, inlet/outlet/wall boundary tag requirements, and no-execute approval gates.
done: structured core-agent foundation added for LLM-backed modules: `call_structured_agent` records bounded structured attempts, validates Pydantic outputs, feeds validation errors back for retry, and returns typed retry exhaustion instead of prose. `PhysicsProblemContract` now locks validated problem intent, and TAPS execution is gated by `review_problem_to_taps_contract` before running local kernels.
done: Phase B started for LLM-backed problem extraction. `run_typed_physicsos_workflow` now supports `core_agents_mode` plus injectable structured clients; default mode is LLM-first, missing/failed structured clients fall back to deterministic parsing after bounded attempts, and explicit `deterministic` mode remains available for CI/offline use.
done: Phase B now includes LLM-first TAPS formulation. `FormulateTAPSEquationInput` carries the locked `PhysicsProblemContract`, `formulate_taps_equation_structured` calls the TAPS formulation agent through strict Pydantic validation, and `run_physicsos_workflow` passes `structured_client/core_agent_config` through to TAPS. If structured formulation fails validation, deterministic TAPS formulation is used as fallback; if the formulated smooth PDE has no executable local kernel, TAPS returns a compiled weak-form IR with `needs_review`/extension or backend-bridge guidance instead of pretending to solve it.
done: `validate_taps_ir` performs symbolic IR readiness checks and recommends knowledge/runtime-extension/full-solver fallback when mappings are unsafe.
done: `export_taps_backend_bridge` emits reviewed real-backend bridge manifests for FEniCSx/MFEM/PETSc without running external solvers.
done: backend bridge export now also emits reviewable FEniCSx/PyMFEM/petsc4py draft artifacts.
done: `plan_taps_adaptive_fallback` emits explicit safe fallback decision artifacts without executing external solvers.
done: `prepare_taps_backend_case_bundle` emits auditable backend case bundles with dependency checks, mesh-export requirements, coefficient/boundary binding, and no-execute approval gates.
done: strong-form diffusion/time/reaction/curl token patterns compile into reviewed weak-form IR terms.
done: Neumann/Robin/interface/custom boundary conditions compile into first-class boundary weak-term IR metadata.
next: turn mesh-based Navier-Stokes bridge manifests into executable reviewed backends: first FEniCSx mixed FEM/projection prototype with SUPG/PSPG or projection stabilization, then OpenFOAM/SU2 runner bundles that consume backend mesh export manifests and enforce approval/dependency checks.
next: never mark nonlinear Navier-Stokes as high-trust TAPS output until continuity, momentum, boundary-condition, and nonlinear residual checks pass. If the weak-form IR contains advection and no stabilized nonlinear TAPS kernel is selected, return `needs_review` or fallback instead of executing the low-Re Stokes simplification.
next: turn auditable backend case bundles into executable exports only after dependency checks, mesh export manifests, coefficient binding, boundary-tag binding, and explicit user approval are satisfied.
done: Phase 1 numerical solver planning replaces the most brittle hard-coded TAPS scalar/nonlinear/backend-preparation routing with typed planning layers while keeping execution deterministic and validated. Scalar elliptic 1D/2D, nonlinear reaction-diffusion 1D/2D, transient diffusion 1D, coupled reaction-diffusion 2D, mesh FEM linear elasticity, mesh FEM EM curl-curl, restricted fluid channel kernels, and backend bridge/case-bundle preparation now build and validate typed plans before execution or artifact emission.
  principle: the LLM may choose discretization strategy, basis/order, axis resolution, coefficient bindings, source-term projection, boundary-condition lifting, nondimensional simplifications, solver tolerances, and fallback strategy. The LLM may not directly write trusted numerical arrays, mark residuals accepted, bypass dependency checks, or execute shell/code. Deterministic code still assembles matrices, applies boundary conditions, runs kernels, writes artifacts, and computes residuals.
  input context: `NumericalSolvePlanInput` should include `PhysicsProblemContract`, `PhysicsProblem`, `KnowledgeContext`, `CaseMemoryContext`, `TAPSCompilationPlan`, `TAPSProblem`, geometry/mesh summary, available backend capabilities, compute budget, and prior validation/contract-review errors.
  output contract: `NumericalSolvePlanOutput` includes `solver_family`, `backend_target`, `discretization` (dimension, axis ranges, node counts, element order, quadrature/order), `field_bindings`, `coefficient_bindings`, `source_bindings`, `boundary_condition_bindings`, `initial_condition_bindings`, `linearization_policy`, `expected_artifacts`, `validation_checks`, `fallback_decision`, `assumptions`, `warnings`, and `unsupported_reasons`.
  validation: deterministic `validate_numerical_solve_plan(problem_contract, taps_problem, plan)` runs before execution. It rejects missing/nonpreserved boundary values, unbound coefficients, invented fields, unsupported dimensions, unsafe simplifications, invalid mesh/axis references, and impossible backend choices. Validation errors plus the original input are fed back through bounded structured retry when an LLM planner is active.
  TAPS execution refactor: `run_taps_backend` now plans, validates, and executes selected scalar elliptic, nonlinear reaction-diffusion, transient diffusion, coupled reaction-diffusion, mesh FEM elasticity, mesh FEM EM, and restricted fluid channel kernels through typed numerical plans. `export_taps_backend_bridge` and `prepare_taps_backend_case_bundle` now use `BackendPreparationPlanOutput` for target backend, field spaces, mesh export, coefficient maps, boundary tags, stabilization policy, dependency checks, and approval gates.
  correctness completed: 1D steady heat conduction supports nonzero Dirichlet boundary lifting. A request such as `T(0)=300 K` and `T(1)=350 K` produces a `taps_solution_field` whose endpoints preserve those values and whose zero-source, constant-conductivity interior is linear.
  2D scalar completed: scalar elliptic 2D supports constant and side-specific nonzero Dirichlet boundary values through deterministic lifting/relaxation, with boundary values and residual checks recorded in the solution artifact.
  3D scalar completed: restricted structured-grid scalar elliptic / steady heat conduction executes through `scalar_elliptic_3d` with canonical boundary roles, constant diffusion/reaction/source bindings, 3D solution-field artifacts, residual checks, and original-BC verification. This covers smooth 3D axial conduction cases such as a cylinder with `x_min/x_max` temperatures and insulated side wall; imported curved-mesh 3D FEM remains a separate backend/kernel target.
  3D FEM scalar path completed: mesh_graph Poisson / steady-heat execution now supports P1 and P2 tetrahedral cells in addition to the existing P1/P2/P3 triangle scalar FEM path. The deterministic executor reads `cell_blocks[type~=tetra|tetra10]`, assembles tetra stiffness and positive lumped mass, applies typed Dirichlet boundary lifting from `boundary_role`/`boundary_node_sets`, solves with sparse CG, and emits `tetra_p1_fem_poisson` or `tetra_p2_fem_poisson` operator/solution/residual artifacts. The P2 tetra path uses quadratic isoparametric geometry, per-quadrature-point Jacobians, reference-to-physical gradient mapping, and a 4-point tetra quadrature rule; artifacts record detJ, shape values, gradients, consistent local mass, and local stiffness.
  3D vector FEM path completed: mesh_graph linear elasticity now supports P1 and P2 tetrahedral vector H1 elements. The deterministic executor assembles the 3D isotropic small-strain constitutive matrix, 6x12 or quadrature-built 6x30 strain-displacement matrices, tetra stiffness, typed vector Dirichlet lifting from `boundary_role`/`boundary_node_sets`, 3-component body-force RHS, sparse CG solve, and 3-component displacement artifacts under `tetra_p1_fem_linear_elasticity` or `tetra_p2_fem_linear_elasticity`.
  3D FEM H(curl) completed for first-order tetrahedra: mesh_graph electromagnetic curl-curl now supports 3D P1 tetra Nedelec edge elements in addition to the existing 2D triangle Nedelec path. The deterministic executor assembles global edge DOFs with orientation signs, tetra curl-curl/mass/source blocks, boundary face/edge selection for tangential policies, sparse/complex solves, and H(curl) edge-field artifacts with tetra element records.
  3D FEM mesh-quality gate completed for curved tetra10 elements: P2 tetra scalar Poisson and vector elasticity element artifacts now record per-quadrature detJ quality metrics. Solver residuals include `mesh_quality_passes`, and inverted/sign-changing curved tetra10 elements force `needs_review` instead of accepted solver success.
  3D FEM H(curl) higher-order scaffold completed: tetra10 EM curl-curl now routes through a second-order hierarchical Nedelec scaffold with two edge-moment DOFs per edge, face-tangent DOFs, a cell-interior DOF, orientation signs, local curl-curl/mass quadrature records, high-order boundary DOF metadata, typed numerical planning, and executable residual artifacts. This is an auditable internal scaffold, not a replacement for MFEM/FEniCSx high-order Nedelec production elements.
  3D FEM H(div) RT0 scaffold completed: Darcy/H(div)/mixed-Poisson weak-form IR and `mesh_graph` tetra cells now route through typed `mesh_fem_hdiv_div` planning and validation, assemble global face-flux DOFs with Raviart-Thomas-order0 local matrices, solve the scaffold system, and emit `taps_mesh_fem_hdiv_operator`, `taps_mesh_fem_hdiv_flux_field`, and residual artifacts. This is a minimal built-in face-flux scaffold for auditable routing and tests; richer mixed pressure-flux coupling, boundary flux enforcement, and production RT/BDM bases still require future work or reviewed external backends.
  scalar FEM weak-boundary support completed for mesh Poisson: Neumann and Robin boundary conditions now compile to first-class boundary weak-term metadata and are applied as deterministic scalar RHS/diagonal contributions when matching boundary node sets are available. Vector traction, impedance-rich general Robin, and higher-order boundary integration remain future scope.
  3D FEM remaining scope: do not claim complete production-grade general 3D FEM yet. Next kernels should strengthen mixed H(div)-pressure coupling, production high-order H(curl)/H(div) basis fidelity, richer Neumann/Robin/vector traction boundary-condition kinds, imported-mesh quality summaries, and verified external backend handoff when mature FEM packages are needed. Keep FEniCSx/MFEM bridging paused unless explicitly re-prioritized.
  nonlinear completed: nonlinear reaction-diffusion 1D/2D kernels consume typed numerical plans for diffusion/reaction/source bindings and solver controls, then emit iteration/residual metadata and canonical solution-field artifacts.
  transient completed: transient diffusion 1D consumes typed numerical plans for field, initial condition, boundary values, low-rank solver rank, and deterministic S-P-T/Taylor approximation metadata. The solver still owns factor construction, residual computation, and artifact writing.
  coupled completed: coupled reaction-diffusion 2D consumes typed numerical plans for primary/secondary field binding, diffusion/coupling/reaction coefficients, damping, iteration budget, and tolerance, then emits applied coefficients, solver controls, residual history, and two-field solution artifacts.
  mesh FEM elasticity completed: linear elasticity consumes typed numerical plans for displacement field binding, Young's modulus, Poisson ratio, constitutive model/body force when present, mesh_graph requirement, and solver controls. Deterministic code owns 2D triangle P1/P2/P3 and 3D tetra P1 element assembly, Dirichlet DOF elimination, iterative solve, displacement artifact writing, and FEM residuals.
  mesh FEM EM completed: H(curl) curl-curl consumes typed numerical plans for electric field binding, permeability, permittivity, wave number, current/source amplitude, mesh_graph requirement, boundary policy metadata, and solver controls. Complex frequency-domain coefficient values are preserved as list/dict/string bindings and converted only inside deterministic execution.
  fluid completed: restricted channel Stokes, Oseen, and laminar Navier-Stokes kernels consume typed numerical plans for velocity/pressure field binding, viscosity, density, pressure drop, Reynolds number, frozen velocity when required, Picard damping/iteration/tolerance, and explicit `support_scope=restricted_steady_laminar_2d_channel` for nonlinear Navier-Stokes. Deterministic code still owns the channel reduction, Oseen linearization, Picard iteration, residual metrics, and high-Re/unsupported fallback behavior.
  numerical solver LLM adapter completed for mesh FEM kernels. Typed workflow TAPS-agent now calls `numerical-solve-planning-agent` through strict `NumericalSolvePlanOutput` Pydantic validation when a structured client is available, validates the result with `validate_numerical_solve_plan`, and passes the validated plan into `run_taps_backend`. This covers `mesh_fem_poisson`, `mesh_fem_linear_elasticity`, and existing EM/fluid/nonlinear kernels. Invalid executable plans stop before kernel execution; unsupported weak-form IR may still emit reviewed `needs_review` IR artifacts without pretending to solve.
  backend preparation completed: `BackendPreparationPlanOutput` validates external backend preparation without execution. It records target backend, backend family, field-space plan, mesh export manifest requirements, coefficient map, boundary tag map, stabilization review policy, solver controls, dependency checks, approval gates, expected artifacts, warnings, and validation status. Bridge manifests and case bundles now embed this plan so CLI/TUI can show exactly why execution remains blocked.
  backend preparation LLM retry completed: `plan_backend_preparation_structured` now uses the structured-agent call path for `BackendPreparationPlanOutput`, then runs deterministic semantic validation for no-execute gates, mesh export requirements, boundary tag maps, coefficient maps, and stabilization review policy. Invalid structured plans are retried with validation feedback; retry exhaustion falls back to deterministic backend preparation planning.
  artifact contract: covered numerical executors emit canonical field artifacts with `field`, `axes`, `values`, `units`, `boundary_values_applied`, `coefficient_values_applied`, and `residual_checks`. This is what the main agent receives and summarizes; main-agent does not need direct plotting/report tools if workflow returns these artifacts plus compact typed summaries.
  main-agent context policy: do not expose low-level plotting/report tools to main-agent by default. Instead ensure `run_typed_physicsos_workflow` returns enough structured context for flexible narrative: problem contract, solver plan, solver result, artifact manifests, sample field statistics, verification report, postprocess report, and event summaries. Main-agent can then explain, compare, or request follow-up without owning numerical execution.
  tests: deterministic executor coverage now includes nonzero Dirichlet 1D heat, 2D scalar elliptic nonzero boundary lifting, scalar assembler 2D Poisson, nonlinear reaction-diffusion 1D/2D numerical-plan execution, transient diffusion 1D plan metadata, coupled reaction-diffusion 2D plan metadata, mesh FEM EM curl-curl plan metadata including complex coefficients, mesh FEM linear-elasticity plan metadata, Stokes/Oseen/restricted Navier-Stokes plan metadata, and backend preparation plan metadata in bridge/case-bundle artifacts. Keep adding fake-model tests for valid plan, invalid plan retry, source-term projection, coefficient binding, unsupported simplification rejection, fallback to backend bridge, and artifact schema completeness.
next: extend backend preparation from manifest quality to runnable reviewed adapters behind approval gates. FEniCSx/MFEM/OpenFOAM/SU2 adapters should consume the typed case bundle, run dependency checks, require mesh export manifests and user approval, then execute only through a controlled runner/service.
  OpenFOAM first adapter steps:
  done: Add a PhysicsOS-side `prepare_openfoam_runner_manifest(taps_backend_case_bundle, solver=simpleFoam|icoFoam)` adapter that converts the reviewed `physicsos.taps_backend_case_bundle.v1` into foamvm's current executable `physicsos.full_solver_job.v1` schema.
  done: The adapter emits `openfoam.case_files` for the narrow first scope only: steady incompressible laminar blockMesh channel with `0/U`, `0/p`, `constant/transportProperties`, `system/controlDict`, `system/fvSchemes`, `system/fvSolution`, and `system/blockMeshDict`.
  done: Keep `submit_full_solver_job` as the only HTTP execution path. `prepare_openfoam_runner_manifest` only writes a manifest artifact; it must not call foamvm or any local OpenFOAM command.
  done: Require approval token for HTTP submission and preserve foamvm scopes (`runner:submit`, `runner:read`, `artifacts:read`). Dry-run mode validates the generated `physicsos.full_solver_job.v1` manifest without network execution.
  done: Add manifest validation that rejects unsafe paths, unsupported solvers, missing `case_files`, missing `inlet/outlet/wall`, and case bundles that require a provided mesh export manifest because first adapter only supports blockMesh channel cases.
  6. After the adapter works locally, update foamvm to accept `physicsos.mesh_conversion_job.v1` separately. Do not make foamvm directly execute `physicsos.taps_backend_case_bundle.v1`; it is an audit/preparation artifact, not a runner contract.
  7. For online smoke: run device-code login, check `/api/physicsos/me`, dry-run locally, then submit one tiny simpleFoam/blockMesh case to foamvm and poll `/jobs/{id}/events` plus `/artifacts`.
done: improved structured semantic validation observability. `call_structured_agent` now accepts semantic validator and feedback hooks, so Pydantic validation and domain validation share one attempt counter, one `validation.retry` event stream, one artifact stream, and consistent `attempt/max_attempts` display in CLI/TUI.
done: added contract-aware LLM retry prompts for `validate_numerical_solve_plan` failures. Numerical solve planning retry context now includes the locked `PhysicsProblemContract`, invalid `NumericalSolvePlanOutput`, validation errors, original `PhysicsProblem`, `TAPSProblem`, and compilation plan, with explicit instructions to repair only invalid bindings or return a real fallback decision.
done: tightened numerical planner retry-exhaustion behavior. If the LLM produced a Pydantic-valid but semantically invalid numerical plan, the workflow preserves that failed plan for deterministic validation and stops before kernel execution instead of silently replacing it with a deterministic plan that could solve a different problem. Deterministic fallback remains for no-client/offline or completely unparseable structured-planning failures.
next: redesign the core typed workflow so every core module can use LLM reasoning internally while preserving strict typed contracts. The current deterministic Python `_run_*_agent` functions are useful fallbacks/tests, but they are not sufficient for a general simulation assistant. The target architecture is `typed graph + LLM structured agents + deterministic validators`:
  principle: core agents are allowed, and usually expected, to call an LLM for interpretation, formulation, repair, review, and synthesis; however, no free-form LLM text may cross a workflow boundary. Every agent must return a Pydantic-validated input/output model, and every tool execution must consume those validated models.
  separation of roles: deterministic code owns validation, schema coercion, artifact IO, backend execution, residual checks, and safety gates. LLMs own ambiguous reasoning: natural-language problem extraction, geometry intent interpretation, boundary-condition mapping, PDE/weak-form derivation, constitutive-law selection, solver strategy explanation, fallback planning, and report synthesis.
  model interfaces: introduce `CoreAgentLLMConfig`, `StructuredAgentCall`, `StructuredAgentAttempt`, and `StructuredAgentResult[T]`. Each call records agent name, model, prompt version, input schema name/hash, output schema name/hash, raw LLM response artifact, parsed output, validation errors, retry count, and final status.
  invocation layer: introduce `call_structured_agent(agent_name, input_model, output_model, system_prompt, tools, policy)` as the only path for core LLM agent calls. It should support OpenAI/compatible structured outputs when available, JSON-mode fallback, and repair prompts when validation fails. The caller receives either a validated Pydantic object or a typed retry/failure context.
  retry policy: all core agents get bounded structured retries. On validation failure, feed the exact Pydantic errors, the invalid output, and the original typed input back to the same agent. After `max_structured_attempts`, return `ValidationRetryContext` rather than silently coercing or continuing.
  problem building: replace the mostly rule-based `build_physics_problem` path with `BuildPhysicsProblemAgentInput -> BuildPhysicsProblemAgentOutput`. The LLM must extract domain, geometry, fields, operators, materials, boundary conditions, initial conditions, source terms, assumptions, unknowns, and confidence. Deterministic validators then decide whether to proceed, ask the user, or run a repair agent.
  problem contract locking: after validation, create a `PhysicsProblemContract` fingerprint containing raw request, domain, equations, fields, geometry dimension/regions/boundaries, material coefficients, BC/IC values, source terms, targets, assumptions, and missing/uncertain items. All downstream agents receive this contract and must prove their outputs preserve or explicitly justify any change.
  TAPS LLM agent: `TAPSAgentInput -> TAPSAgentOutput` must become an LLM-backed structured call. The LLM formulates/repairs the weak form, chooses trial/test fields, maps coefficients and boundary terms, decides whether knowledge is required, and produces `TAPSCompilationPlan`. Deterministic tools then validate IR, build `TAPSProblem`, execute available kernels, or export backend bridges. TAPS must not re-interpret the user request from scratch; it must compile the locked `PhysicsProblemContract`.
  geometry-mesh LLM agent: `GeometryMeshAgentInput -> GeometryMeshAgentOutput` should use LLM reasoning to map user geometry language/CAD labels into region and boundary semantics, propose mesh policy, identify ambiguity, and select encodings. Deterministic geometry tools still perform import/mesh/quality/export. Low-confidence boundary labeling must produce a human-confirmation artifact before solver execution.
  knowledge LLM agent: knowledge remains the gateway for external scientific facts. Other core agents may request knowledge through typed `KnowledgeQuery` objects. Knowledge-agent returns source-grounded `KnowledgeContext` with citations, uncertainty, and extracted equations/material/validation facts. Core agents cannot invent citations or unsupported solver capabilities.
  solver LLM agent: solver-agent may use LLM reasoning to compare TAPS limitations, surrogate models, OpenFOAM/SU2/FEniCSx/MFEM requirements, runtime constraints, and user goals. It must return `SolverAgentOutput` with a typed decision, not prose. Full-solver execution remains approval gated.
  verification LLM agent: verification-agent may use LLM reasoning to interpret residuals, conservation failures, mesh quality, OOD warnings, and physical plausibility, but deterministic residual/conservation computations remain authoritative. The LLM can recommend next actions only through `VerificationReport`.
  postprocess LLM agent: postprocess-agent may use LLM reasoning for engineering interpretation, report writing, figure captions, and recommendation synthesis, but KPIs/artifact paths must come from typed solver/postprocess data.
  case-memory LLM use: case-memory can use LLM summarization/embedding only to index and retrieve cases. It must preserve append-only typed events and canonical structured case records. Prior cases can guide agents, but they cannot override the current `PhysicsProblemContract` without explicit typed justification.
  contract review gate: add `review_problem_to_taps_contract(problem_contract, taps_problem, compilation_plan) -> ContractReviewReport` before `run_taps_backend`. It verifies fields, BC/IC values, coefficients, geometry regions, targets, and assumptions survived the TAPS compilation. If mismatch exists, retry TAPS formulation with the report; if still mismatched, stop before execution.
  workflow shape: `User natural language -> LLM BuildPhysicsProblemAgent -> deterministic validate -> KnowledgeAgent if needed -> GeometryMeshAgent -> validate/contract lock -> LLM TAPSAgent -> deterministic TAPS IR validation -> contract review -> execution or backend bridge -> VerificationAgent -> PostprocessAgent -> CaseMemoryAgent`.
  CLI/TUI visibility: show LLM structured attempts as workflow events, not hidden thinking. Display compact lines such as `[problem] extracted 2D heat equation, 3 BCs`, `[taps] contract review failed: right boundary value changed`, `[taps] retry 2/3`, `[solver] OpenFOAM bridge prepared`. Full raw LLM responses should be saved as artifacts but hidden by default.
  testing strategy: keep deterministic tests for validators and kernels. Add fake-model tests for every LLM-backed core agent proving valid structured output, invalid-output retry, contract mismatch detection, no-prose boundary crossing, and fallback behavior when max attempts are exhausted.
  migration plan: phase A adds structured agent call infrastructure and fake-model tests; phase B converts `build_physics_problem` and TAPS formulation to LLM-backed structured calls behind a feature flag; phase C converts geometry/solver/verification/postprocess; phase D makes LLM-backed core workflow default while retaining deterministic fallback for CI/offline tests. Phase A is started: structured call infrastructure and TAPS contract review are in place; remaining Phase A work is adding raw-response artifacts/events and config-driven model clients. Phase B now covers `build_physics_problem` and TAPS formulation with fake structured clients; remaining Phase B work is wiring real model clients, saving raw structured attempts as artifacts/events, and adding contract-aware retry prompts after TAPS review failures.
  feature flags: support `PHYSICSOS_CORE_AGENTS_MODE=llm|hybrid|deterministic`. Default is `llm`: try structured LLM extraction first, retry bounded validation failures, then use deterministic fallback if no client is configured or attempts are exhausted. `deterministic` keeps the current code path for CI/offline use. `hybrid` remains an explicit compatibility mode for staged rollout.
  safety rule: LLM participation increases reasoning generality but never weakens execution gates. No LLM output can directly execute solvers, write trusted kernels, alter core code, or mark a result accepted without passing deterministic validation, contract review, and verification.
done: normalized agent-facing paths across Windows, macOS, and Linux. Internally tools keep `pathlib.Path` at filesystem boundaries, while agent-visible paths, prompts, event artifacts, and DeepAgents filesystem tool references use portable `/workspace/...` paths with forward slashes. Tests cover round-trip conversion and DeepAgents virtual workspace behavior.
done: separated local path display from URI semantics for current artifact flows. Local artifact paths emitted to agents/events are workspace-relative `/workspace/...` references where possible; real remote URLs are preserved; incoming agent paths are converted back to local paths at the boundary.
done: added shared path normalization helpers in `physicsos.paths` and wired them into events, structured agent artifacts, tool outputs, and DeepAgents CLI patches so Windows drive paths and mixed separators do not leak into agent-visible filesystem operations.
done: replaced the current DeepAgents CLI all-tools inheritance workaround with scoped tool injection. The main PhysicsOS agent exposes orchestration/catalog/problem-validation tools, while each subagent receives scoped tools plus shared knowledge/case-memory basics:
  geometry-mesh-agent -> geometry import/repair/label/mesh/encoding/export/quality tools
  taps-agent -> PhysicsProblem validation, knowledge lookup, TAPS formulation/build/validate/bridge/fallback/run/residual tools
  solver-agent -> surrogate routing/inference, full-solver preparation/submission/run, hybrid solver tools
  verification-agent -> residual, conservation, slice validation, uncertainty, OOD tools
  postprocess-agent -> KPI, visualization, report tools
  knowledge-agent -> local KB, arXiv, DeepSearch, ingest, case-memory tools
done: made scoped tool ownership explicit in code through `MAIN_AGENT_TOOLS`, domain tool groups, and `SUBAGENT_TOOL_GROUPS`. Tests verify every subagent tool exists in the registry, shared case-memory tools are available, and domain tools such as TAPS, solver, geometry, and postprocess do not leak into unrelated subagents.
done: reduced long `thinking` tails by narrowing tool choice, making the typed workflow canonical for core simulation paths, and tightening the main-agent prompt so it does not delegate when the typed workflow already owns geometry/TAPS/solver/verification/postprocess sequencing.
done: converted `geometry-mesh-agent` to the same LLM-first structured pattern used by problem-building and TAPS, while keeping all geometry execution deterministic. `GeometryMeshPlanInput -> GeometryMeshPlanOutput` now has strict Pydantic validation, bounded structured retries, deterministic fallback, workflow event/artifact recording, boundary-confirmation handoff for ambiguous imported geometry, and tool-registry coverage. The LLM may infer geometry intent, region semantics, boundary roles, mesh policy, boundary-layer/refinement needs, and required encodings from `PhysicsProblemContract`, user intent, CAD/mesh labels, knowledge context, and case memory. It must not directly generate trusted mesh files or silently relabel solver-critical boundaries. Deterministic tools still own CAD/mesh import, gmsh/meshio execution, mesh quality checks, region/boundary application, encoding generation, backend mesh export, and artifact writing.
  workflow: `GeometryMeshAgentInput -> LLM GeometryMeshPlan -> validate_geometry_mesh_plan -> deterministic label/mesh/encoding/export tools -> GeometryMeshAgentOutput`.
  boundary policy: if labels are missing or low confidence, emit a `boundary_labeling_artifact` and set `handoff.status=needs_user_input` before solver execution. Auto-label generated primitives only when confidence and physics-domain defaults are explicit; for CAD/mesh imports, prefer confirmation unless physical groups are clear.
  mesh policy: typed plan should include target element size, order, local refinement regions, boundary-layer requirements for fluid cases, backend compatibility targets, and quality thresholds. Deterministic validation rejects impossible policies and records retry context for LLM repair.
  encoding policy: typed plan should choose `mesh_graph`, `occupancy_mask`, `sdf`, `boundary_graph`, or `laplacian_eigenbasis` based on downstream solver/TAPS needs. Generated encodings must use workspace-relative portable artifact paths.
  observability: every geometry LLM attempt must use the structured attempt event/artifact path so CLI/TUI can show `[geometry] structured attempt 1/3`, boundary confidence, mesh status, quality warnings, and confirmation artifacts without exposing raw LLM output inline.
  tests: add fake-model tests for valid geometry plan, invalid-output retry, low-confidence boundary confirmation, mesh policy validation, encoding selection, and deterministic fallback when no structured client is configured.
done: upgraded `postprocess-agent` Phase 1 from the lightweight SVG preview into a typed LLM-planned plus deterministic matplotlib/reporting pipeline. `PostprocessPlanInput -> PostprocessPlanOutput` now lets the LLM select figures, report sections, captions, and follow-up recommendations from the problem contract, solver artifacts, verification report, knowledge context, case memory, and user objective. The workflow records structured postprocess attempts as events/artifacts and falls back deterministically after validation failure. Deterministic renderers now use matplotlib in an isolated subprocess for 1D line plots, 2D heatmaps, and fluid velocity quiver plots, with SVG/JSON fallback and a report manifest.
deferred: do not extend `postprocess-agent` beyond Phase 1 as a current priority. Main agent remains the flexible document-style summarizer when it receives enough typed workflow context and artifacts.
  renderer scope: implement deterministic matplotlib renderers for 1D scalar line plots, residual history curves, 2D scalar heatmaps/contours, mesh scalar fields, vector quiver/stream plots for fluid velocity, pressure contours, displacement/vector magnitude plots, and uncertainty/error summaries. The renderer reads `SolverResult.artifacts` such as `taps_solution_field`, `taps_reconstruction_metadata`, `taps_stokes_solution_field`, `taps_navier_stokes_solution_field`, and mesh FEM solution artifacts; it should not rely on the LLM to parse raw arrays.
  report ownership: `postprocess-agent` owns formal artifact generation: figures, captions, KPI tables, markdown/html/pdf-ready report, and machine-readable `simulation_report_manifest`. The main agent owns the conversational summary, user-facing explanation, and follow-up questions. The main agent may quote or summarize the postprocess report, but should not replace the typed report artifact as the canonical result.
  direct plotting tools: expose plotting as typed tools (`generate_solution_figures`, `generate_residual_plots`, `write_simulation_report`) that can call matplotlib directly inside the Python environment. If matplotlib is missing, return a typed dependency error and fallback to the existing SVG/JSON summary instead of failing the workflow.
  visualization plan validation: require every requested figure to bind to an existing field/artifact or declare a missing input. Reject hallucinated fields, invalid file paths, and plots whose dimensionality does not match the data. Feed validation errors back through structured retries.
  artifact policy: write all figures under `scratch/<case_id>/visualizations/` with portable `/workspace/...` paths in events/tool outputs; include native paths only in internal artifact payloads when needed. Emit `artifact.created` and `agent.output` events for every generated figure and report.
  testing: add tests for 1D TAPS line plot, 2D scalar contour/heatmap, fluid velocity quiver or magnitude plot, report manifest completeness, missing matplotlib fallback, invalid figure-plan retry, and CLI/TUI event rendering of generated figures.
done: upgraded case memory from a final archival step into a shared workflow memory layer. The final `case-memory-agent` remains the curator/indexer, while case memory is exposed to the main agent and all core subagents as a shared read/append store:
  read/search: main-agent, knowledge-agent, geometry-mesh-agent, taps-agent, solver-agent, verification-agent, postprocess-agent
  append_event: all core agents can append typed stage events, failed attempts, solver-routing reasons, geometry decisions, verification findings, user corrections, and fallback decisions
  commit_final_case/update_index: only case-memory-agent, or main-agent after explicit workflow completion, should write the canonical searchable case record
  architecture: introduce `CaseMemoryContext`, `CaseMemoryEvent`, and `CaseMemoryCommit` Pydantic models; store event log and canonical case index separately; keep event writes idempotent by `run_id`, `case_id`, `stage`, and `event_id`
  workflow placement: search similar cases immediately after `build_physics_problem`, pass `CaseMemoryContext` through typed workflow state, allow every subagent input to receive relevant prior cases, and commit the final case after verification/postprocess
  storage path: start with the existing JSONL implementation for compatibility, but define the interface so it can move to SQLite/FTS/vector indexes without changing agent contracts
done: made DeepAgents CLI display PhysicsOS typed workflow events and tool returns with a stream style comparable to the existing CLI/TUI. The assistant final message is no longer the only structured-result surface; PhysicsOS has an event bus with a stable event schema and renderer layer:
  event schema: `PhysicsOSEvent(run_id, case_id, event, stage, status, summary, payload, artifacts, timestamp, display)`
  event types: `workflow.started`, `agent.started`, `agent.output`, `tool.started`, `tool.output`, `validation.retry`, `artifact.created`, `case_memory.hit`, `case_memory.event`, `workflow.completed`, `workflow.failed`
  producers: typed workflow, PhysicsOS tools, case-memory tools, and DeepAgents subagent wrappers emit events to the session JSONL log and, when running interactively, to a live sink
  renderer: build one shared `PhysicsOSEventRenderer` used by legacy CLI, DeepAgents CLI patches, and future web UI; default view is compact stage lines, while `--verbose`, `/events`, or an expandable TUI panel shows full typed Pydantic JSON payloads
  DeepAgents CLI integration: patch or wrap LangGraph streaming rather than only calling `invoke`; consume graph stream updates/tool messages/custom events, map them into `PhysicsOSEvent`, and render them using the same visual hierarchy, colors, status labels, and collapsed/expanded behavior as the existing DeepAgents CLI
  UX target: users should see progress such as `[knowledge] retrieved 4 chunks`, `[geometry] mesh ready`, `[validate] retry 1/2`, `[taps] backend=taps:thermal_1d`, `[verification] accepted`, plus clickable/visible artifact paths and a final typed workflow summary
  persistence: every displayed event must also be written to the session log so `/last-result`, `/events`, `/artifacts`, and debugging after a crash can reconstruct the full run
next: productize the human-in-the-loop geometry boundary confirmation loop for imported 3D Gmsh/CAD/mesh workflows. The geometry-mesh agent must not stop at saying labels are ambiguous; it should create a complete confirmation handoff:
  trigger: imported/CAD/mesh geometries with missing, low-confidence, or solver-critical physical groups set `GeometryMeshPlanOutput.require_boundary_confirmation=true`.
  artifacts: workflow creates both `boundary_labeling_artifact.json` and `geometry_labeler_viewer.html`; both are attached to the geometry handoff and emitted as visible workflow artifacts/events.
  user experience: main agent/CLI should show the viewer path prominently, explain that the user opens the standalone HTML, confirms/selects face or edge groups, and saves the updated JSON with `confirmed_boundary_labels`.
  resume command/tool: done for the local apply step via `physicsos geometry apply-boundary-labels <geometry.json> <boundary_labeling_artifact.json> --output <geometry.confirmed.json>`, preserving confirmed canonical boundary roles. Done for local workflow resume via `physicsos workflow resume-confirmed-geometry <problem.json> <geometry.confirmed.json> --output <workflow_result.json>`, which replaces `PhysicsProblem.geometry` with the confirmed artifact and continues TAPS/solver/verification/postprocess/case-memory execution.
  safety: solver export and external execution must require confirmed labels for imported 3D solver-critical boundaries unless physical groups are already explicit and high confidence.
  tests: add workflow tests proving the geometry handoff contains both viewer and JSON artifacts, and that applying confirmed labels clears the `needs_user_input` boundary state before backend export.
next: harden the LLM-to-solver contract after the real API cylinder heat-conduction test exposed a false-success path. This is a blocker for trusted simulation, not a cosmetic cleanup:
  problem: structured LLM output currently passes Pydantic shape validation while leaving solver-critical semantics as free strings. Boundary ids such as `bnd_left`, `bnd_x0`, or `surface_hot_end` are all valid strings but not safe solver contracts. A deterministic fallback can also preserve JSON shape while silently losing user boundary values. Verification then checks backend self-consistency rather than comparing the solved artifact back to the locked `PhysicsProblem`.
  done: schema fix added first-class canonical boundary roles to `BoundaryRegionSpec`, `BoundaryConditionSpec`, `TAPSBoundaryConditionSpec`, and `NumericalBoundaryConditionBinding`. Roles use typed enums such as `x_min`, `x_max`, `y_min`, `y_max`, `z_min`, `z_max`, `side_wall`, `inlet`, `outlet`, `wall`, `symmetry`, `farfield`, `interface`, and `custom`. `id` remains an opaque stable identifier only; executable kernels must not infer physics from `id` strings.
  done: canonicalization step added `canonicalize_physics_problem(problem) -> CanonicalPhysicsProblemOutput` before TAPS/solver planning in the natural-language typed workflow. It maps generated primitive labels and confirmed physical groups into boundary roles, records assignments/warnings, and returns `needs_user_input` when imported/CAD/mesh solver-critical roles cannot be determined. LLM repair may propose roles, but deterministic validation owns acceptance.
  done: fallback policy tightened. Deterministic fallback outputs are drafts, not trusted execution inputs. If an LLM structured attempt produces a semantically invalid numerical plan, workflow preserves the invalid plan and stops through deterministic validation instead of silently executing a deterministic plan that may solve a different problem. Candidate fallbacks must pass canonical role, coefficient, field, unit, and contract-preservation validators before execution.
  done: solver plan contract hardened. `validate_numerical_solve_plan` rejects executable 1D scalar endpoint kernels when required Dirichlet roles are missing, rejects boundary value drift from the locked `PhysicsProblem`, rejects invented fields/boundaries, and requires explicit unsupported/fallback plans to carry `unsupported_reasons` and `fallback_decision`.
  done: verification contract now compares solver artifacts against the original locked problem, not only backend residuals. `check_boundary_condition_application` checks `boundary_values_applied` by boundary id, region id, canonical role, and legacy backend keys, supports scalar/vector values, and rejects wrong applied boundary values even when backend residuals self-report success.
  1D thermal acceptance: steady 1D heat conduction may be accepted only when endpoint roles `x_min/x_max`, endpoint values, diffusion coefficient, residual, and solution artifact all agree with the locked problem. The zero-source constant-conductivity sanity check should confirm an approximately linear profile.
  3D TAPS status: done for restricted structured-grid scalar elliptic/steady heat conduction with canonical boundary roles and verified solution artifacts. Continue to return `fallback_required -> backend/full_solver` for imported curved meshes, heterogeneous/material-interface cases, nonlinear 3D PDEs, and any 3D weak-form family without an executable reviewed kernel.
  tests: add fake-model and deterministic regression tests for arbitrary LLM boundary ids with canonical roles, missing role rejection, fallback-draft rejection, original-BC vs solver-artifact mismatch, 1D cylinder axial success with 300/350 K endpoints, and 3D cylinder thermal returning `fallback_required` instead of false success.
```

PhysicsOS Cloud / foamvm scope:

```text
runners/foamvm is only the OpenFOAM/E2B runner test harness and existing deployed cloud runner adapter.
It should not own geometry labeling UI, local mesh viewers, or database-independent interaction tools.
Standalone geometry labeling is generated by the Python tool `create_geometry_labeler_viewer` as an HTML artifact.
```
