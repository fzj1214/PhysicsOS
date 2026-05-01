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
Gmsh mesh -> mesh_graph encoding -> 2D first-order Nedelec EM curl-curl path
edge DOFs + orientation signs -> H(curl) tangential continuity -> residual-checked EM edge-field artifact
PhysicsProblem boundary_conditions -> TAPSBoundaryConditionSpec -> PEC/natural EM edge-boundary policy
complex-valued EM coefficients -> `[real, imag]` JSON representation -> complex frequency-domain edge solve
custom/robin EM boundary value dict -> absorbing impedance / port excitation -> boundary-edge Robin contribution
mesh_graph boundary_edge_sets -> region_id-specific PEC / absorbing / port edge selection
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
heat / diffusion
Poisson
reaction-diffusion
Helmholtz
knowledge-assisted unspecified physics
```

Next implementation requirement:

```text
connect weak-form IR to generic assemblers/subspace solvers so compiled
non-template physics can execute, not only compile.
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
done: EM boundary-condition semantics for Nedelec edge DOFs: PEC tangential-zero fixes boundary edges; natural/farfield leaves edge DOFs free.
done: complex-valued EM frequency-domain coefficient support with JSON `[real, imag]` artifacts and complex edge-field solve.
done: EM absorbing/impedance/port boundary policies via structured boundary value dictionaries.
done: region-specific EM boundary edge selection using mesh_graph `boundary_edge_sets` for bbox and Gmsh physical line groups.
done: generated Gmsh rectangle physical groups propagate through meshio into named mesh_graph boundary edge sets.
done: imported Gmsh `.geo` physical curve labels propagate into `GeometrySpec.boundaries`, `mesh_graph.boundary_edge_sets`, and EM port/wall edge selection.
done: mesh_graph physical boundary metadata carries solver-native export hints for OpenFOAM patches, SU2 markers, and FEniCSx facet tags.
done: `export_backend_mesh` produces OpenFOAM/SU2/FEniCSx/MFEM/TAPS mesh export manifests with physical boundary mappings and no unapproved local external conversion.
done: Gmsh/meshio-first 3D facet layer propagates physical surfaces into `boundary_face_sets`, `physical_boundary_groups.face_ids`, and solver export manifests.
done: user-confirmed physical-group labeling artifacts separate weak suggestions from confirmed labels before GeometrySpec mutation or solver export.
done: standalone geometry labeler viewer tool can load boundary-labeling artifacts, rotate mesh/facet previews, select groups, and export confirmed labels without foamvm, database, or E2B.
done: mesh conversion runner manifests can be prepared from backend mesh export manifests, inline source `.msh`, and dry-run/http submitted without local external conversion.
next: implement the foamvm/E2B runner execution branch for `physicsos.mesh_conversion_job.v1` manifests.
next: add higher-order Nedelec spaces after the Gmsh/meshio facet and boundary-label pipeline is stable.
```

PhysicsOS Cloud / foamvm scope:

```text
runners/foamvm is only the OpenFOAM/E2B runner test harness and existing deployed cloud runner adapter.
It should not own geometry labeling UI, local mesh viewers, or database-independent interaction tools.
Standalone geometry labeling is generated by the Python tool `create_geometry_labeler_viewer` as an HTML artifact.
```
