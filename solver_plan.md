# PhysicsOS Solver Refactor Plan

本文档记录 PhysicsOS 求解器可信度重构计划。目标不是继续堆模板，而是把系统改成：

- DeepAgents 文档型主 agent 负责交互、审阅、检索和 artifact 辅助。
- LangGraph typed state machine 负责严格仿真流程、契约校验、路由、执行和验证。
- 有限数量真实求解器、几何/mesh 自动化后端负责数值结果。
- TAPS 保留，但定位为 weak-form IR compiler 和 a priori surrogate builder，不再作为普通单案例仿真的默认演示求解器。

## 0. 结论

当前系统能跑通 CLI，但不能把所有现有 TAPS、fallback、geometry/mesh 产物都当成可信通用仿真结果。关键问题是：TAPS demo、solver fallback、verification、geometry/mesh 中混入了演示问题、受限核、默认参数和 backend 自报残差。

铝杆 case 的正确路线应是：

1. 从中文 prompt 解析出 1 m 长度、铝、初温 20 degC、左端 100 degC、右端 0 degC、模拟 10 s。
2. LLM structured extraction 显式填入铝热参数并保留 provenance；若 main agent 需要外部依据，应先调用材料数据库/搜索工具，再把结果写入 typed problem，workflow 主路径不自动补材料默认值。
3. 路由到匹配范围明确的 `fdm_heat_1d`，而不是旧 `taps:thermal_1d` demo。
4. 从 canonical solution artifact 独立复算离散残差、BC、IC、范围约束。
5. 若用户明确要求参数 sweep/surrogate，再构建真正的 TAPS S-P-T surrogate，并用 `fdm_heat_1d` 或 FEniCSx 抽查。

## 1. 当前问题根因

铝杆热传导 residual 过大不是调参问题，而是旧 TAPS thermal demo 不适配真实 case：

- 旧 demo 默认 unit slab、sinusoidal IC、固定或弱绑定参数轴。
- 几何长度、材料热扩散率、边界温度、终止时间不一定严格进入执行核。
- 验证层过去容易依赖 backend-reported residual，没有从实际 `T(x,t)` artifact 独立复算。
- CLI/agent prompt 过去有 TAPS-first 倾向，会把普通单案例往 demo 路径拉。

## 2. 已发现的硬编码或演示风险

| 区域 | 风险 | 处理原则 |
| --- | --- | --- |
| `taps_thermal.py` | 1D transient heat demo，适合 smoke test，不适合生产单案例 | 标记 `demo_only`，不能返回 `success` |
| `taps_tools._default_axes()` | 可能默认 `[0,1]` 和默认 time axis | P2 删除 transient heat 特判，axes 来自 typed problem |
| TAPS generic scalar kernels | 有可复用数值核，但 scope 和验证不一 | 分级为 production/experimental/demo/scaffold |
| H(div)/H(curl) kernels | 含明显 scaffold 字段 | 未完成前不允许 production acceptance |
| OpenFOAM runner | 固定流程过硬，未发挥 solver 能力 | 改成 backend-agnostic sandbox job runner |
| Geometry/mesh | physical group、facet tags、boundary confidence 不完整 | 增加 semantic gate 和 export manifest |

## 3. 论文理解

### 3.1 TAPS: `2503.13933v1(1).pdf`

TAPS 是 Tensor-decomposition-based A Priori Surrogate。它的重点不是固定模板跑一次，而是：

- 从 PDE/weak form 构造 space-time 或 space-parameter-time 轴，而不是从某个单案例模板开始。
- 用 C-HiDeNN/低秩张量结构形成可复用 surrogate。论文将解场写成 CP/TD 形式，独立变量可以包含空间、参数和时间。
- 将 generalized S-P-T Galerkin weak form 离散为 separated matrix contractions。由于 TD 乘积结构，即使原 PDE 是线性的，离散系统通常也是 nonlinear。
- execution 阶段采用 subspace iteration：每次只把一个维度的 solution matrix 当未知，固定其它维度因子；该子问题变成线性系统，解完后切换到下一个维度反复迭代。
- 长时间或 moving source 问题需要 time slab，把一个巨大 S-P-T 域拆成多个较小 slab，控制 mode 数和时间维求解成本。
- 需要 residual/refinement policy、rank/slab adaptation 和 baseline spot checks。

参考依据：

- arXiv 摘要说明 TAPS 直接从含空间、参数、时间等独立变量的 governing equations 获得 reduced-order model，并构造 generalized space-parameter-time Galerkin weak form: <https://arxiv.org/abs/2503.13933>。
- 论文 HTML 第 2.2-2.4 节说明 CP/TD 插值、S-P-T independent variables、Galerkin weak form 和 time slab: <https://ar5iv.org/html/2503.13933v1>。
- 第 2.5-2.6 节说明 TD weak form 离散后是 nonlinear system，并用 subspace iteration 逐维求解 TD linear system: <https://ar5iv.org/html/2503.13933v1>。

因此，对于“单根 1 m 铝杆 10 s transient heat”这种普通单案例，隐式有限差分或 FEM 更可靠。TAPS 应在用户要求 surrogate/sweep 或存在多查询需求时使用。

### 3.2 ALL-FEM: `2603.21011v2.pdf`

可借鉴 agentic FEM 代码生成思想，但 PhysicsOS 不应让 LLM 直接改 core backend。更合适的策略：

- LLM 只生成 case-local FEniCSx/FEM extension artifact。
- 必须通过 syntax/import/runtime/verification gate。
- 通过测试和人工 review 后才能提升为 core backend。

### 3.3 FeaGPT: `2510.21993v1.pdf`

可借鉴 geometry/mesh semantic automation：

- 自动识别几何实体、材料区、边界角色。
- Gmsh physical groups 作为 domain/material/boundary 的显式语义桥。
- 低置信边界标签必须 `needs_user_input`，不能假装已确认。

### 3.4 FEniCSx/Gmsh

FEniCSx 应作为真实 FEM 后端候选。Gmsh physical groups 和 mesh tags 是关键契约。FEniCSx bundle 需要包含 mesh、facet tags、cell tags、PDE script、solver controls、expected artifacts。

## 4. 新架构原则

### 4.1 DeepAgents 与 typed workflow 边界

DeepAgents main agent：

- 接收用户自然语言。
- 调用 `run_typed_physicsos_workflow`。
- 审阅报告、解释失败、触发文档检索或用户确认。
- 不直接调用内部 solver primitives。

DeepAgents subagents：

- 只做文档、研究、artifact 审阅、报告辅助。
- 不复用 `taps-agent`、`solver-agent`、`verification-agent` 等 workflow node 名称。

Typed workflow nodes：

- problem build/canonicalize
- case memory
- knowledge
- geometry/mesh
- validation
- capability-aware route
- TAPS formulation/capability/execution
- solver execution
- verification
- postprocess
- case memory commit

### 4.2 TAPS 新定位

TAPS 节点拆为三个角色：

- `taps_formulation_node`: 编译 faithful weak-form IR。
- `taps_capability_node`: 判断是否有 executable TAPS kernel 或 surrogate route。
- `taps_execution_node`: 只有 capability scope、bindings、verification policy 都满足时才执行。

TAPS 不应该：

- 对普通单案例默认抢占真实 solver。
- 用 unit geometry 或默认材料参数隐藏用户输入缺失。
- 将 demo residual 包装成 high-trust result。

TAPS 应该：

- 记录 axes/coefficient/boundary provenance。
- 对 surrogate/sweep 场景提供价值。
- baseline spot checks 是高置信/回归抽查证据，不是 TAPS execution 成功的硬性前置条件。

### 4.2.1 TAPS execution 阶段 kernel 要求

TAPS execution 不能再按 PDE family 堆窄函数。应按论文的 separated Galerkin/subspace iteration 拆成通用执行层：

1. `TAPSFormulationIR`
   - 来自 `PhysicsProblem` 和 weak-form terms。
   - 只描述 fields、axes、basis、coefficients、boundary/source terms、parameter/time/slab policy。
   - 不直接包含任何求解结果。
2. `SeparatedOperatorIR`
   - 将 weak-form term 编译为 separated axis operators。
   - 每个 term 保存 `field_in`、`field_out`、`test_field`、`coefficient_ref`、`axis_blocks`、`boundary_region`、`integration_measure`。
   - axis blocks 是 `mass`、`stiffness`、`gradient`、`time_derivative`、`identity`、`source_projection` 等通用块。
3. `CapabilityCheck`
   - 判断 term 是否可 separated。
   - 判断 coefficient 是否有 separated representation 或可近似分解。
   - 判断 BC/IC/slab 是否可绑定。
   - 不能执行时返回 `missing_bindings`，而不是 fallback 到模板。
4. `SubspaceIterationExecution`
   - 初始化每个 field 的 factor matrices。
   - 对每个 rank/field/axis 组装子空间线性系统。
   - 固定其它轴，通过 tensor contraction 得到当前轴 operator 权重。
   - 求解当前轴 factor，归一化，并循环到收敛。
   - 可做 rank enrichment 和 time-slab continuation。
5. `Acceptance`
   - `converged` 只表示 subspace iteration 收敛。
   - `accepted` 必须通过 typed IR、semantic validation、subspace/rank residual gate；baseline spot checks 可提升置信度，但缺失时不应自动否定 execution。

需要避免的错误实现：

- `if family == "heat": solve_heat()` 这种直接内核不能称为 TAPS execution，只能称为 backend adapter。
- 解析式/Taylor 展开、FDM、FEM 单次求解不是 TAPS execution，除非只是用于 baseline check。
- `SubspaceIterationKernel` 不能只支持某个固定 weak form。最小实现也要消费 generic separated term contractions。

### 4.2.2 Geometry/mesh 与 TAPS 的协作边界

TAPS weak form 和 execution 都受 geometry/mesh 深度影响，但影响方式不同于传统 FEM/FVM：

- Formulation 阶段需要 geometry semantics。
  - 必须知道 domain、material region、boundary region、boundary role、source/support region。
  - 这些信息决定 weak-form integration domain、boundary integral、BC/IC/source binding。
  - 如果 boundary label 低置信或缺失，TAPS formulation 只能输出 `needs_user_input` 或 `missing_bindings`。
- Execution 阶段需要 geometry numerical representation。
  - 对 tensor-product/规则几何，geometry 可直接映射为 separated axes、1D quadrature 和 per-axis mass/stiffness/derivative blocks。
  - 对参数化几何，需要把 Jacobian、metric tensor、domain indicator、boundary measure 写成 separated coefficients 或可近似分离的 low-rank fields。
  - 对任意 CAD/Gmsh mesh，不能直接声称 TAPS 可执行；必须先有 mesh-to-separated-geometry compression、immersed/occupancy representation，或退到 mesh FEM/FEniCSx。
- Mesh 不应直接污染 TAPS weak form。
  - weak form 应保留 PDE/region/boundary 语义，不应写死 node id、element id。
  - mesh 只在 `AxisBasisKernel`、`SeparatedOperatorIR`、baseline FEM bridge 中出现。
- Mesh semantics 必须能回传给 TAPS。
  - physical groups、facet tags、cell tags、boundary confidence、geometry provenance 都要进入 TAPS capability check。
  - TAPS result 必须记录使用了哪些 geometry representation：tensor axes、mesh_graph、occupancy mask、SDF、separated Jacobian/indicator 等。

因此，Geometry/Mesh node 与 TAPS node 的协作应分为四层：

1. `GeometrySemanticContract`
   - 用户尺寸、primitive/CAD source、material domains、boundary labels、physical groups、confidence。
   - 供 `taps_formulation_node` 绑定 weak-form regions。
2. `GeometryNumericalEncoding`
   - structured axes、mesh_graph、occupancy mask、SDF、boundary graph、facet/cell tags。
   - 供 `taps_capability_node` 判断是否可 separated execution。
3. `SeparatedGeometryOperator`
   - metric/Jacobian/domain indicator/boundary measure 的 separated blocks。
   - 供 `SubspaceLinearSystemKernel` 组装 per-axis linear systems。
4. `BaselineMeshBridge`
   - 当 geometry 不适合 TAPS 或 TAPS 需要 spot check 时，导出 FEniCSx/OpenFOAM/SU2 mesh bundle。
   - 不把 baseline solver 结果冒充为 TAPS execution。

实现原则：

- Geometry/Mesh 是 TAPS 的前置契约，不是可选装饰。
- TAPS capability 不能只看 PDE family，还必须看 geometry representation 是否可 separated。
- 任意复杂几何默认不走 TAPS execution；只有满足 separated geometry、immersed indicator 或已验证 mesh compression 时才可执行。
- 如果 TAPS 使用 mesh_graph 做本地 FEM，那属于 `mesh_fem_*` experimental/scaffold backend，不等同于论文式 TAPS separated execution。

### 4.2.3 Geometry/Mesh 与 TAPS 的交互协议

论文中的 TAPS 先把解写成 CP/TD separated representation，再把 generalized S-P-T Galerkin weak form 离散成按轴分离的矩阵收缩，最后用 subspace iteration 固定其它轴、逐轴求解线性系统。因此 geometry/mesh 对 TAPS 的影响分两层：

1. 对 weak form 的影响是语义层。
   - weak form 需要 `Omega`、material subdomain、interface、`Gamma_D`、`Gamma_N`、source/support、initial slice、parameter domain。
   - 这些对象来自 geometry/mesh 的 semantic labels 和 confidence，不来自 mesh node id、element id。
   - weak-form IR 应表达为 `integral(domain_ref, integrand, measure_ref)`，其中 `domain_ref` 指向语义 region。
   - 如果 geometry node 不能可靠绑定 “left end”、“hole boundary”、“inlet”、“wall” 等边界语义，TAPS formulation 必须停止在 `missing_bindings`，不能猜默认边界。

2. 对 execution 的影响是数值可分离层。
   - structured interval/rectangle/box 可以直接生成 tensor axes、1D basis、mass/stiffness/derivative/quadrature blocks。
   - 参数化几何可以作为 extra-coordinate，但 mapping 的 Jacobian、metric tensor、domain indicator、boundary measure 必须能表示成 separated coefficients 或可压缩 low-rank factors。
   - 任意 CAD/Gmsh mesh 默认只提供 FEM/FVM execution 的 mesh graph，不自动提供 TAPS separated execution。
   - 对复杂几何若要走 TAPS，需要至少一种受验证的转换：NURBS/geometry mapping 到规则参考域、domain decomposition 到若干可分离子域、immersed/fictitious-domain indicator 的 low-rank 分解、或 SIMP/occupancy separated field。

两者应按以下契约协作：

```text
problem_contract
  -> geometry_mesh_node
       outputs:
         GeometrySemanticContract
         GeometryNumericalEncoding
         MeshExportManifest
  -> taps_formulation_node
       consumes: problem_contract + GeometrySemanticContract
       outputs: TAPSFormulationIR + GeometryRequirements
  -> taps_capability_node
       consumes: TAPSFormulationIR + GeometryNumericalEncoding + GeometryRequirements
       outputs: executable_taps | needs_geometry_compression | route_to_fem
  -> taps_execution_node
       consumes: TAPSFormulationIR + SeparatedGeometryOperator
       outputs: TAPS factors + iteration history + residual evidence
  -> baseline_mesh_bridge
       consumes: MeshExportManifest
       outputs: FEniCSx/OpenFOAM/SU2 spot-check artifacts
```

关键边界：

- Geometry/Mesh node 负责“这个边界/区域是什么”和“它是否能数值执行”；TAPS formulation 负责“这个 PDE 的 weak form 是什么”。
- TAPS execution 不能直接消费裸 mesh graph 后宣称 TAPS；它必须消费 `SeparatedGeometryOperator`。
- `SeparatedGeometryOperator` 的最小字段包括 `axis_domains`、`basis_refs`、`volume_measure_factors`、`boundary_measure_factors`、`metric_factors`、`indicator_factors`、`coefficient_factors`、`compression_error`、`validity_region`。
- mesh FEM 可以作为 baseline 或 fallback；它的 residual/solution 不能反向证明 TAPS separated kernel 可执行，只能作为 spot check。
- TAPS capability gate 必须同时检查 PDE separability、coefficient separability、geometry separability、BC/source separability、rank/slab budget 和 baseline policy。
- 对 moving boundary/moving mesh 问题，geometry 变化应进入 time/parameter axis 的 mapping factors；如果只能通过 ALE mesh update 表示，则优先走 FEniCSx/OpenFOAM，不走 TAPS separated execution。

### 4.2.4 LLM-driven TAPS execution 设计

execution 层从一开始允许 LLM-driven proposal，但数值执行不能由 LLM 自由发挥。正确边界是：

```text
TAPSFormulationIR + SeparatedGeometryOperator
  -> LLM proposes SeparatedOperatorIR + TAPSExecutionPlan
  -> typed schema validation
  -> deterministic semantic validator
  -> deterministic numerical kernels
  -> independent residual/baseline verification
```

LLM 可以做：

- 提议 weak-form term 到 separated axis operator blocks 的映射。
- 提议 rank、axis sweep order、damping、slab policy、fallback strategy。
- 在 validator 失败后根据错误反馈修复 proposal。

LLM 不能做：

- 直接运行矩阵组装、线性求解、残差计算。
- invent coefficient、axis、geometry factor、boundary region。
- 把 raw `mesh_graph` 当成 separated geometry operator。
- 漏掉 weak-form term 后仍标 `ready`。

第一批 execution schema：

- `SeparatedOperatorIR`
  - 只描述 operator decomposition，不执行。
  - 每个 `SeparatedOperatorTermIR` 必须引用一个 formulation term。
  - 每个 axis block 必须引用合法 axis、basis、operator kind、coefficient/geometry factor。
- `TAPSExecutionPlan`
  - rank、axis order、subspace solver、damping、max iterations、tolerance、slab policy、fallback。
  - `ready` 只表示 execution plan 和 separated operator 通过 typed/semantic gate，不表示数值已验证。
- `propose_separated_operator_structured()`
  - LLM-backed proposal。
  - 使用 structured output 和 retry feedback。
  - validator 检查 term coverage、axis refs、coefficient refs、geometry factors、mesh_graph 禁止规则。

数值 kernel 分层：

- `AxisBasisKernel`: deterministic 1D basis/quadrature/operator block generator。
- `SeparatedWeakFormCompiler`: LLM proposal + deterministic validation，输出 `SeparatedOperatorIR`。
- `SubspaceLinearSystemKernel`: deterministic contraction 和线性系统组装。
- `SubspaceIterationKernel`: deterministic ALS/PGD/TAPS subspace iteration。
- `TAPSVerificationKernel`: independent residual、spot check、rank/slab convergence evidence。

接受规则：

- formulation ready 不等于 execution ready。
- separated operator ready 不等于 simulation accepted。
- execution accepted 需要 subspace convergence + independent residual verification；baseline verification 是可选高置信 gate，适合 benchmark、回归和高风险场景。
- LLM proposal 失败时返回 `needs_operator_review` 或 fallback，不走硬编码“猜一个能跑的算子”。

### 4.3 求解器有限但真实

优先做少量可验证求解器：

| Backend | Scope | 状态 |
| --- | --- | --- |
| `fdm_heat_1d` | 1D transient heat/diffusion, uniform IC, left/right Dirichlet, scalar alpha | P1 production |
| FEniCSx | Poisson/heat/linear elasticity with mesh tags | P3 planned |
| OpenFOAM | CFD/FVM through sandbox runner | P4/P8 planned |
| TAPS separated Galerkin engine | general separated weak-form execution with subspace iteration; first validated families are limited | P5 planned |

## 5. Typed State Machine 设计

建议逐步迁移为显式 LangGraph `StateGraph`，保留当前 public API：

```python
class PhysicsOSWorkflowState(StrictBaseModel):
    run_id: str
    problem: PhysicsProblem | None
    problem_contract: PhysicsProblemContract | None
    capability_scores: list[BackendCapabilityScore] = []
    route_decision: SolverDecision | None
    taps_ir: TAPSAgentOutput | None
    solver_result: SolverResult | None
    verification: VerificationReport | None
    artifacts: list[ArtifactRef] = []
    validation_attempts: list[ValidationRetryContext] = []
```

核心 gates：

- `problem_contract_gate`
- `mesh_semantics_gate`
- `capability_scope_gate`
- `execution_safety_gate`
- `solution_artifact_gate`
- `independent_verification_gate`

## 6. 标准数据契约

### 6.1 BackendCapability

需要集中 schema：

```python
class BackendCapability(StrictBaseModel):
    backend_id: str
    family: Literal["fdm", "fem", "fvm", "taps", "surrogate", "custom"]
    status: Literal["production", "experimental", "demo_only", "scaffold", "disabled"]
    supported_domains: list[str]
    supported_equations: list[str]
    support_scope: str
    required_inputs: list[str]
    verification_methods: list[str]
```

### 6.2 SolutionArtifact

Canonical solution artifact 最少包含：

- schema version
- backend id
- problem id
- fields and units
- mesh/grid coordinates
- time coordinates if transient
- values
- BC/IC/coefficient/solver controls applied
- provenance

### 6.3 VerificationEvidence

Verifier 必须输出：

- independent residual
- BC check
- IC check for transient
- conservation or skipped-with-warning
- range/invariant check
- artifact references

## 7. Solver Routing

路由输入：

- typed `PhysicsProblem`
- capability registry
- user goal: single solve, sweep, surrogate, high fidelity, explanation
- compute budget
- verification policy
- available local/sandbox backends

路由原则：

```text
if production backend exactly matches problem and verifier exists:
    run production backend
elif user asks sweep/surrogate and TAPS capability + baseline checks exist:
    run TAPS formulation/execution + baseline spot checks
elif FEM/FVM backend supports problem and sandbox exists:
    prepare/run full solver
else:
    return needs_user_input or unsupported with precise missing capability
```

这不是“默认 deterministic solver”。`deterministic solver` 只是传统数值后端类别，例如 FDM/FEM/FVM/direct nonlinear solver。agent 仍必须基于 capability 和 verification 选择。

## 8. Geometry/Mesh 自动化

### 8.1 Semantic Gate

generated/imported geometry 需要：

- dimension 和 primitive 与用户请求一致。
- user dimensions 进入 geometry parameters。
- material regions 可映射到 cell groups。
- boundary conditions 可映射到 boundary groups。
- 低置信标签返回 `needs_user_input`。

### 8.2 Gmsh Pipeline

```text
GeometrySpec
  -> semantic regions/boundaries
  -> Gmsh physical groups
  -> mesh
  -> export manifest with cell/facet tags
```

### 8.3 Backend Export

- FEniCSx: XDMF/HDF5 + facet/cell tag manifest。
- OpenFOAM/SU2: patch/marker manifest。
- TAPS: mesh_graph/SDF/occupancy encodings with provenance。

## 9. Verification 重构

每个 result 必须通过：

- `solution_artifact_present`
- `contract_preserved`
- `bc_application`
- `initial_condition_application`
- `pde_residual`
- `convergence_or_refinement`
- `range_or_invariant`

1D heat residual:

```text
R_i^n = (T_i^{n+1} - T_i^n) / dt
        - alpha * (T_{i-1}^{n+1} - 2T_i^{n+1} + T_{i+1}^{n+1}) / dx^2
```

## 10. P0-P6 改装计划

### P0: 隔离 demo/scaffold

目标：demo kernel 不能冒充 production solver。

- 标记 `taps_thermal.py` 当前核为 `demo_only`。
- `demo_only/scaffold` 不能返回 `SolverResult.status == "success"`。
- verification 不能把 demo/scaffold 升级为 `accepted`。
- report 显示 `capability_status` 和 `support_scope`。

### P1: 真实 1D transient heat solver

目标：铝杆 case 正确求解和验证。

- `physicsos/backends/heat_1d.py`
- implicit Euler 三对角求解。
- 支持 Dirichlet BC。
- 读取 L、final time、IC、BC、alpha。
- 由 `k/(rho*cp)` 派生 alpha。
- 输出 `solution.json` 和 `residual_check.json`。

### P2: TAPS 编译和执行解耦

- `build_taps_problem()` 不再硬编码 transient heat axes。
- `validate_taps_ir()` 输出 executable capability。
- TAPS result 区分 compiled IR、capability、execution result、baseline verification。

### P3: FEniCSx production backend

- 生成 deterministic FEniCSx scripts。
- 支持 mesh import + facet/cell tags。
- 先支持 Poisson/heat/linear elasticity。
- 沙盒执行并采集 XDMF/VTU/JSON/log/residual artifacts。

### P4: Geometry/mesh semantic automation

- physical groups 读写。
- boundary confidence gate。
- generated geometry 使用用户尺寸和 primitive。
- FEniCSx/OpenFOAM/SU2 export manifests。

### P5: 真正 TAPS separated Galerkin engine

P5 目标不是 `taps_spt_heat` 专用 kernel，而是实现论文式 general execution skeleton：separated weak-form contractions + subspace iteration + rank/slab policy。第一批验证可以选 1D heat/diffusion，但内核抽象不能写死 heat 方程。

核心 kernel 设计：

- `SeparatedFieldKernel`
  - 管理解场的 CP/TD 表示：`u(q1, q2, ..., qD) = sum_r prod_d U_d[:, r]`。
  - 支持多个 field，共享或独立 rank，记录 normalization/gauge fixing。
  - 输出 factor matrices、rank history、mode norms、field reconstruction sampler。
- `AxisBasisKernel`
  - 对每个 independent variable 生成 1D basis、quadrature、mass/stiffness/derivative matrices。
  - 轴类型包括 `space`、`parameter`、`time`、`geometry`。
  - 初期可用 P1 finite element 或 C-HiDeNN-like local basis；接口必须允许后续替换成 C-HiDeNN。
- `SeparatedWeakFormKernel`
  - 将 TAPS weak-form IR 编译成 separated term contractions。
  - 每个 term 形式为一组 per-axis operator blocks，例如 mass、gradient/stiffness、time derivative、reaction、source、boundary integral。
  - 支持 coefficient field 的 separated approximation；不能分离时返回 `missing_bindings` 或要求 fallback/FEniCSx spot check。
- `SubspaceLinearSystemKernel`
  - 对第 `d` 个维度固定其它维度 factors，组装该维度的线性系统 `A_d(U_except_d) U_d = b_d(U_except_d)`。
  - 不允许只识别 heat 特例；它必须消费 generic separated term contractions。
  - term contraction 通过除当前维外的 factor inner-products 缩并得到标量权重，再乘当前维 operator block。
- `SubspaceIterationKernel`
  - 外层 ALS/subspace loop：初始化 factors -> 遍历 dimensions/fields -> solve subspace linear system -> normalize -> residual/update check。
  - 支持 rank enrichment、damping/relaxation、line search 或 restart。
  - 输出 iteration history，包括 per-axis residual、relative update、condition estimate、rank changes。
- `SlabKernel`
  - 对 time axis 分 slab，slab 内执行 separated solve，slab 间传递初值或 discontinuous Galerkin penalty。
  - 长时程、moving source、强非线性默认走 slab，而不是单个巨大 time axis。
- `NonlinearCoefficientUpdateKernel`
  - 对温度相关材料、非线性 reaction/source、radiation/phase change 等，根据当前 factors 更新 coefficient contractions。
  - 明确 nonlinear outer loop 与 subspace inner loop 的收敛标准。
- `TAPSVerificationKernel`
  - 从 factor matrices 重构抽样点/切片。
  - 复算 PDE residual、BC/IC、守恒量或能量项。
  - 对 production 接受必须有 `fdm_heat_1d`、FEniCSx 或其它确定性 backend spot checks；TAPS 自身 residual 不能单独提升为 high-trust。

最小可验收切片：

- 不叫 `taps_spt_heat`，命名为 `taps_separated_galerkin` 或 `taps_subspace_iteration`。
- 实现至少一个 scalar linear parabolic/elliptic family 的 generic separated weak-form execution，但代码路径必须经 `SeparatedWeakFormKernel` 和 `SubspaceIterationKernel`。
- 测试同一 subspace kernel 至少消费两种 term 组合，例如 diffusion-only 和 diffusion+reaction，证明不是 heat 专用模板。
- artifacts 必须包含 `taps_factors.json`、`taps_axis_operators.json`、`taps_subspace_iteration_history.json`、`taps_reconstruction_samples.json`。
- 若只有 heat 特例、解析式、Taylor 展开或直接 FDM，则只能标为 demo/adapter，不能标为 TAPS execution。

### P6: LLM-generated solver extension sandbox

- LLM 生成 case-local code。
- 必须通过 syntax/import/runtime/verification。
- 不自动 promotion 到 core backend。
- promotion 需要测试、代码审阅和用户确认。

## 11. 测试计划

- `tests/test_heat_1d_solver.py`
  - aluminum rod prompt end-to-end
  - direct PhysicsProblem -> `fdm_heat_1d`
  - BC/IC/coefficient binding checks
- `tests/test_solver_capabilities.py`
  - demo/scaffold backend cannot return success
  - route picks production backend before TAPS demo
- `tests/test_verification_independent.py`
  - verifier rejects missing solution artifact
  - verifier recomputes heat residual
  - verifier catches backend self-reported zero residual with wrong field
- `tests/test_mesh_semantics.py`
  - physical groups map to boundary conditions
  - unresolved boundary labels produce `needs_user_input`
- `tests/test_fenicsx_case_generation.py`
  - deterministic script generation for Poisson/heat
  - runtime skip if FEniCSx unavailable, schema still validated

## 12. 已确认决策

1. 允许材料数据库/专业参数搜索 tool 存在，供 main agent 或 LLM 显式调用；typed workflow 主路径不自动从材料库补值。Tavily API 放在 config，可由 `load_config()` 启用。
2. 路由由 agent/node 结合 prompt 和 capability registry 选择，不做死板默认。
3. `deterministic solver` 指传统确定性数值后端类别，不是默认路线。
4. FEniCSx 可参考 OpenFOAM 的 E2B 沙盒策略，但要用新 template 和更灵活的 execution plan。
5. OpenFOAM runner 后续要从固定流程升级为 backend-agnostic sandbox job runner。
6. `taps_generic.py` kernel 分级必须保守：production、experimental、demo_only、scaffold 分开。
7. 本项目终端、环境变量、CLI 输入输出、报告和 artifact 中文统一使用 UTF-8。源码和测试可以直接写中文；禁止在业务层用中文乱码 marker、GBK/GB18030 猜测或 mojibake 修复逻辑。若 Windows 终端出乱码，应修终端/环境/IO 编码，不应修改物理解析语义。

## 13. 具体任务清单

任务状态用 `[done]` 或 `[todo]` 标记。后续实现时实时更新。

### A. 文档和决策

- [done] 记录铝杆热传导 case 的失败现象和直接根因。
- [done] 明确 TAPS 不应作为普通单案例默认替代求解器。
- [done] 明确 DeepAgents main agent、subagent、PhysicsOS typed workflow node 的边界。
- [done] 记录材料库、Tavily、路由策略、FEniCSx E2B、OpenFOAM runner、P0/P1 许可。
- [todo] 在 `README.md` 或 `ARCHITECTURE.md` 同步 solver refactor 高层摘要。

### B. Config、材料库和参数搜索

- [done] 在 `physicsos/config.py::default_config()` 增加 `search` 配置段。
- [done] 支持 `TAVILY_API_KEY`、`PHYSICSOS_SEARCH_PROVIDER`、`PHYSICSOS_SEARCH_ENABLED`。
- [done] CLI/DeepAgents 启动时从 config `search.tavily_api_key` 注入 `TAVILY_API_KEY` 和 `PHYSICSOS_TAVILY_API_KEY`，供 DeepAgents 框架自带 Tavily/web search 使用。
- [done] 增加材料库 schema，包含值、单位、温度范围、来源、置信度。
- [done] 撤销内置材料库节点：不再提供 `data/materials/core_materials.json` 作为 PhysicsOS 主路径资源，不再让 workflow 自动查询或补材料参数。
- [done] 撤销 `physicsos/tools/material_tools.py`：当前不注册材料 lookup/search tool；后续由用户接入专业材料数据库/搜索工具。
- [done] Tavily 仅作为 DeepAgents 框架自带 web search 的环境变量配置，不作为 PhysicsOS 材料搜索节点。
- [done] 材料参数由 LLM structured problem extraction 显式写入 typed problem；旧 `build_physics_problem()` 不再参与自然语言生产路径。
- [done] typed workflow 主路径不自动补材料库默认值，缺材料时返回 typed missing input；材料参数必须由 LLM 或后续接入的专业工具显式填入 typed problem。
- [done] build-physics-problem agent 对 schema 合法但 `missing_inputs` 非空的输出执行 semantic retry，并把缺失项作为 `validation_feedback` 传回 LLM；重试耗尽后阻断 workflow，不进入 TAPS/solver。
- [done] 添加铝热扩散率解析和 Tavily 未配置测试。

### C. Backend capability registry

- [todo] 新增 `physicsos/schemas/capability.py`。
- [todo] 新增 `physicsos/backends/capabilities.py` 集中登记所有 backend。
- [done] 在现有 registry 中暴露 `capability_status`、`support_scope`、`verification_methods`。
- [done] 给 `fdm_heat_1d` 注册 production capability metadata。
- [done] 给现有 TAPS thermal demo result 标记 `demo_only`。
- [done] 给 Stokes/Oseen/Navier-Stokes channel kernels 标记 `experimental`。
- [done] 给 H(div)/H(curl) scaffold kernels 标记 `scaffold`，禁止返回 `success`。
- [todo] 评估 scalar elliptic、mesh Poisson、elasticity kernels 的分级。
- [done] 添加测试：demo backend 不能成功通过；registry 每个 backend 都有 scope 和 verification methods。

### D. 1D 热传导生产求解器 P1

- [done] 新增 `physicsos/backends/heat_1d.py`。
- [done] 支持 Dirichlet 边界，预留 Neumann/Robin 扩展点。
- [done] 支持 implicit Euler 三对角求解。
- [done] 从 `PhysicsProblem` 读取长度、终止时间、初值、左右边界温度、材料热扩散率。
- [done] 当只有 `k/rho/cp` 时派生 `alpha = k/(rho*cp)`。
- [done] 输出 canonical `solution.json`。
- [done] 输出 `residual_check.json`，最终接受由 verifier 决定。
- [done] 在 `solver_tools.py` 注册 `fdm_heat_1d` 执行路径。
- [done] 添加中文 prompt 端到端回归测试。
- [done] 添加直接构造 `PhysicsProblem` 的单元测试。

### E. 求解器路由

- [done] 将 `_run_solver_agent()` 改为 capability-aware routing。
- [done] 路由规则解释 `fdm_heat_1d` 的能力和验证要求。
- [todo] 消除“普通单案例默认走某后端”的硬编码倾向，改成 problem/capability/budget/user goal/verification policy 综合选择。
- [done] 铝杆 case 选择 `fdm_heat_1d`，不走 `taps_thermal` demo。
- [todo] 用户明确要求 sweep/surrogate 时，路由可选择 TAPS formulation + baseline spot checks。

### F. TAPS 改装 P0/P2/P5

- [todo] 拆分 TAPS formulation、capability check、execution 三个 workflow node 或内部函数。
- [todo] `build_taps_problem()` 删除 transient heat 固定 axes 特判。
- [todo] TAPS axes 必须来自 geometry/material/parameters/time horizon。
- [done] 新增 `TAPSFormulationIR`，将 fields、axes、basis、weak-form terms、coefficients、BC/IC/source/slab policy 与 execution result 解耦。
- [done] `TAPSFormulationIR` 增加 coefficient/source bindings 和 binding diagnostics，ready 状态要求 term 引用的系数显式绑定。
- [done] `formulate_taps_ir()` 改为保守非 LLM 入口：不再按硬编码 PDE family 生成 weak-form terms，只报告需要 LLM formulation engine。
- [done] 新增 `formulate_taps_ir_structured()`，作为真正 LLM-backed formulation engine，带 schema 校验、semantic validator、错误反馈重试。
- [done] 新增测试：LLM candidate invent/unbound coefficient 且标 ready 时被拒绝，错误进入下一次 `validation_feedback`。
- [done] 新增测试：LLM candidate 的 advection/reaction/source/incompressibility constraint/boundary terms 能进入同一 formulation IR。
- [done] 新增 `SeparatedOperatorIR`，把 weak-form term 编译成 per-axis mass/stiffness/derivative/source/boundary blocks。
- [done] 新增 `SeparatedOperatorIR`、`SeparatedOperatorTermIR`、`SeparatedAxisOperatorBlock`、`TAPSExecutionPlan` schema。
- [done] 新增 `propose_separated_operator_structured()`，作为 LLM-driven separated operator / execution plan proposal 入口。
- [done] separated operator validator 检查 formulation term coverage、unknown axis、invent coefficient、unknown geometry factor、plan/operator status consistency。
- [done] 新增测试：LLM 漏 term、invent axis/coefficient 时 retry，错误进入 `validation_feedback`，修复后 accepted。
- [done] 新增 `SeparatedFieldKernel`，统一管理 CP factor matrices、rank、normalization 和 reconstruction sampler；当前为 execution 层通用场状态，不编码具体 PDE，TD/时间 slab 因子留给后续 `SlabKernel`。
- [done] 新增 `AxisBasisKernel`，先支持 P1 1D basis/quadrature/operator matrices，接口预留 C-HiDeNN basis；当前实现位于 `physicsos/backends/taps_execution.py`，覆盖 mass/stiffness/derivative/identity/source vector 和 Gauss quadrature。
- [done] 新增 `SubspaceLinearSystemKernel`，固定其它 axis factors 后组装当前 axis 线性系统；当前支持从 generic `SeparatedOperatorIR` 的 mass/stiffness/derivative/identity/source blocks 装配 axis-local lhs/rhs，并记录 term contribution。
- [done] 新增 `SubspaceIterationKernel` 最小 execution 闭环：消费 generic `SeparatedOperatorIR`，通过 `SubspaceLinearSystemKernel` 组装 single-axis Galerkin 系统，支持 essential constraints、direct solve、residual history；已用 diffusion-only 与 diffusion+reaction 两类 term 组合验证同一路径。
- [done] 扩展 `SubspaceIterationKernel` 为 rank-one 多轴 ALS/PGD prototype：固定其它 axis factors 轮换求当前 axis、支持 damping、axis_order、dense residual diagnostic、iteration history；已用 2D rank-one reaction/source case 验证多轴轮换和重构。
- [done] 扩展 `SubspaceIterationKernel`：新增 multi-rank enrichment/restart prototype、rank-wise diagnostics、multi-rank dense residual diagnostic。
- [done] 继续扩展 `SubspaceIterationKernel`：支持 rank-wise orthogonalization、非线性 Picard/Newton/fixed-point 外层回调接口、可选 independent baseline gate。
- [done] 新增 slab continuation 接口，支持 time slab list 和 slab 间 state 传递。
- [done] 将 nonlinear/slab/baseline policy 接入 typed `TAPSExecutionPlan` schema：新增 `nonlinear_policy`、`operator_update_policy`、`time_slabs`、`orthogonalize_axis`、`baseline_policy`、`baseline_samples`、`baseline_tolerance`；baseline 不作为默认必需项。
- [done] `run_taps_backend()` 的 `taps_separated_galerkin` 主路径消费 execution plan policy：linear/single-axis、rank enrichment、Picard/Newton/fixed-point frozen-IR 外层、time slab continuation、rank-wise orthogonalization、required/optional baseline gate 都走同一个 typed dispatcher。
- [done] `write_taps_execution_artifacts()` 支持 nonlinear 和 time-slab result，`taps_subspace_iteration_history.json` 记录 `execution_metadata`、`nonlinear_history`、`linear_solve_count` 或 slab-by-slab continuation 证据。
- [todo] baseline 输入后续扩展 provenance/source 字段，支持 FDM/FEM/FEniCSx/实验/解析 manufactured sample 等多来源；当前 typed baseline sample 只表达 node index/value。
- [done] 新增 damping policy 和 iteration history 输出。
- [done] 新增 rank enrichment / restart policy，并输出 rank-wise convergence diagnostics。
- [done] `run_taps_backend()` 对 transient heat demo 写入 `capability_status=demo_only` 并禁止 `success`。
- [done] TAPS transient heat result 增加 `capability_status` 和 `support_scope`。
- [done] 其它 TAPS kernels 初步补齐 `capability_status`、`support_scope`、`verification_methods`；H(div)/H(curl) scaffold 增加 `missing_bindings`。
- [todo] 当前 `solve_transient_heat_1d()` 改名为 `solve_demo_sine_heat_1d()` 或移入 examples；当前已从默认 `run_taps_backend()` execution path 隔离，只能通过显式 legacy flag 调用。
- [done] 实现 `taps_separated_galerkin` 的 execution-kernel prototype 底座：第一批覆盖 1D scalar diffusion/reaction，代码路径走 generic `SeparatedOperatorIR`、`AxisBasisKernel`、`SubspaceLinearSystemKernel` 和 `SubspaceIterationKernel`。
- [done] 将 `taps_separated_galerkin` 注册为默认 TAPS execution 主路径：必须提供 `NumericalSolvePlanOutput(solver_family="taps_separated_galerkin")`、`SeparatedOperatorIR` 和 `TAPSExecutionPlan`。
- [done] 默认 `run_taps_backend()` 不再自动执行旧 TAPS thermal/generic/mesh FEM demo/experimental 分支；旧分支保留为显式 `allow_legacy_kernels=True` 的迁移/对照路径。
- [done] `taps_separated_galerkin` promotion 不只改字符串：新增 execution contract gate，校验 IR/plan ready、formulation/operator id 一致、axis/node_counts 匹配、coefficient/source/geometry refs 已绑定、unsupported axis operator 拒绝、required artifacts 完整。
- [done] `estimate_taps_support()` 明确降级为本地 advisory fallback；新增 `estimate_taps_support_structured()` 作为 LLM-driven TAPS support estimator，带 typed schema 校验、semantic validator、错误反馈 retry。
- [done] TAPS support prompt 明确说明适用范围：weak-form PDE、scalar elliptic/diffusion/Poisson/Laplace/稳态导热、transient diffusion、reaction-diffusion、linear/low-Re problems、separated/low-rank/parametric/time-slab 场景；同时要求识别同义 operator family，不得把 estimate 写成 hard gate。
- [done] workflow 中 `taps.support` 只作为提醒/风险信号记录，trace/event/gate 标记为 advisory；`supported=false` 不再阻断 formulation、numerical planning 或 execution，只有真实 formulation/contract/planning/execution failure 才能停止或转向 fallback。
- [done] LLM support estimator 自身 retry 失败也不能阻断主流程；workflow 记录 typed advisory failure risk 后继续进入 formulation/execution，且不在 LLM 主路径用本地机械评分替代 LLM 判断。
- [done] `taps_separated_galerkin` result 标记 `capability_status=production`，但 scope 明确限制为 typed separated Galerkin、structured P1 axes、validated operator contract、residual/rank convergence；baseline gate 是可选 confidence evidence，不是默认成功前提。
- [done] 添加测试：同一 `SubspaceIterationKernel` 消费 diffusion-only 和 diffusion+reaction 两种 weak-form term 组合。
- [done] 添加测试：默认 TAPS backend 不再把 heat demo 自动标为 TAPS execution；demo 只能显式 legacy 调用，且仍为 `demo_only/needs_review`。
- [done] 添加测试：`taps_separated_galerkin` 拒绝 unbound coefficient、axis node count mismatch、execution plan/operator id mismatch。
- [done] artifacts payload 增加 `taps_factors`、`taps_axis_operators`、`taps_subspace_iteration_history`、`taps_reconstruction_samples` 生成函数。
- [done] 将 TAPS execution artifacts 写入实际文件：`taps_factors.json`、`taps_axis_operators.json`、`taps_subspace_iteration_history.json`、`taps_reconstruction_samples.json`，并在显式 `taps_separated_galerkin` backend path 挂到 `SolverResult.artifacts`。
- [done] 添加测试：`taps_separated_galerkin` 会按 `TAPSExecutionPlan` 执行 nonlinear Picard policy、time slab continuation policy，并在 required baseline sample 缺失或失败时拒绝/降级为 `needs_review`。
- [todo] TAPS surrogate 接受前必须通过 `fdm_heat_1d` 或 FEniCSx 抽查。

### G. 独立 verification

- [todo] 定义 canonical `SolutionArtifact` schema。
- [done] `verification_tools.py` 对 `fdm_heat_1d` 优先读取 canonical solution artifact。
- [done] 实现 1D heat residual 复算。
- [done] 实现 transient IC 检查或等价 residual/range gate。
- [done] 强化 BC 检查：从 `T(x,t)` 边界列验证。
- [done] 实现 heat maximum principle/range check。
- [done] conservation skipped 作为 advisory warning 记录；缺少守恒量 imbalance evidence 不再把已收敛 TAPS execution 硬降级为 `needs_full_solver`。
- [done] TAPS separated artifacts -> verifier 映射：`taps_factors`、`taps_axis_operators`、`taps_reconstruction_samples` 可重建 verifier solution payload，供 BC 检查、selected slices 和 verification report 使用。
- [done] `estimate_taps_residual()` 原生识别 `normalized_separated_residual`，避免 TAPS separated Galerkin 因缺少 legacy `relative_l2_reconstruction_error` 被误判为未收敛。
- [done] 添加测试：解/边界错误时必须 fail。
- [done] demo/scaffold capability status 不能被 verification 升级为 accepted。

### H. Geometry/mesh semantics

- [done] 增加 `mesh_semantics_gate`。
- [done] generated geometry 必须使用用户尺寸和 primitive；缺少显式 primitive/dimension metadata 时 gate 返回 `needs_user_input`。
- [done] no-op repair 不能直接 pass，应记录 `not_repaired` 或 `requires_confirmation`。
- [done] Gmsh physical groups 必须进入 mesh export manifest。
- [todo] FEniCSx export 需要 XDMF/HDF5 + facet/cell tag manifest；当前只在 manifest 中保留 facet tag contract，不启用默认执行。
- [done] OpenFOAM/SU2 export 需要 patch/marker manifest；当前只导出 manifest，不启用默认执行。
- [done] boundary label confidence 低于阈值时返回 `needs_user_input`。
- [done] 新增 `GeometrySemanticContract` schema：domain、subdomain、interface、boundary role、source/support、confidence、provenance。
- [done] 新增 `GeometryNumericalEncoding` schema：structured axes、mesh graph、SDF、occupancy、NURBS/mapping、facet/cell tags、quality metrics。
- [done] 新增测试：boundary/region 语义缺失时，TAPS formulation 返回 `missing_bindings`，不能生成默认边界。
- [done] 新增测试：Gmsh/FEniCSx mesh graph 只能触发 FEM bridge 或 `needs_geometry_compression`，不能被标记为 TAPS separated execution。
- [done] 将 boundary labeling artifact 接入 `build_geometry_mesh_contract()`，确认标签进入 contract provenance。
- [done] 新增测试：未确认 artifact 继续阻断 TAPS formulation，人工确认后 boundary semantics 放行 formulation。
- [done] 新增 typed boundary/region candidate 生成工具，候选只进入 viewer/artifact，不直接进入 contract。
- [done] viewer 支持候选一键填入、手动修改 label/kind/role/confidence、下载确认后的 JSON。

### H2. Geometry-to-TAPS separated execution

- [done] 新增 `GeometryRequirements`：TAPS weak form 对 domain measure、boundary measure、metric/Jacobian、indicator、coefficient separability 的要求。
- [done] 新增 `SeparatedGeometryOperator` schema，并记录 compression method、rank、error estimate、valid parameter/time range。
- [done] 新增 `geometry_separability_gate`：structured tensor domain 通过；raw arbitrary mesh graph 不通过；有 low-rank indicator/mapping 且误差达标才通过。
- [todo] 新增 `geometry_compression_plan`：NURBS/reference mapping、domain decomposition、immersed/fictitious-domain indicator、SIMP/occupancy separated field。
- [done] 新增测试：同一个 weak-form IR 在 structured interval/box 上可进入 TAPS capability，在 arbitrary mesh-only encoding 上必须 fallback。
- [todo] 新增测试：moving geometry 若没有 separated mapping factors，路由到 FEniCSx/OpenFOAM ALE bridge，而不是 TAPS execution。

### I. FEniCSx sandbox backend

- [todo] 在 `D:\foamvm` 或 `runners/foamvm` 设计 FEniCSx E2B template。
- [todo] 增加 runner manifest 字段：`backend`、`template_id`、`case_files`、`entrypoint`、`expected_artifacts`。
- [todo] FEniCSx template 预装 DOLFINx、mpi4py、petsc4py、meshio、h5py、pyvista/VTK。
- [todo] PhysicsOS 生成 deterministic FEniCSx case bundle，先支持 Poisson/heat/linear elasticity。
- [todo] 沙盒执行后收集 XDMF/VTU/JSON/log/residual artifacts。
- [todo] 本地无 FEniCSx 时只验证 case bundle schema，runtime 放 runner integration。

### J. OpenFOAM/E2B runner 改造

- [todo] 将 `D:\foamvm\lib\physicsos-runner.ts` 从 only-openfoam 固定流程升级为 backend-agnostic job runner。
- [todo] manifest 由 capability allowlist + case-specific plan 控制。
- [todo] 支持 `commands` 或 `execution_plan`，经过安全 allowlist、cwd 限制、timeout、artifact policy。
- [todo] OpenFOAM plan 支持 mesh conversion、`blockMesh`、`snappyHexMesh`、`gmshToFoam`、`decomposePar`、solver、`reconstructPar`、`postProcess`、`foamToVTK`、sample/functionObjects。
- [todo] runner 提取 residuals、Courant number、continuity errors、time directories、field ranges。
- [todo] 增加 FEniCSx template selection 和 backend dispatch。
- [todo] 保留严格安全边界。

### K. DeepAgents/typed workflow 适配

- [done] 保持 DeepAgents main bridge 小工具面，不暴露内部 solver primitives；main bridge 只暴露 `run_typed_physicsos_workflow` 等窄入口。
- [done] CLI `-n` 与 DeepAgents main agent 自然语言入口统一：两者都进入 `run_typed_physicsos_workflow()`，不再维护 CLI-only deterministic shortcut。
- [done] `run_typed_physicsos_workflow()` 不再允许自然语言请求 fallback 到旧 `build_physics_problem()` 模板解析；`core_agents.mode=deterministic` 返回 `deterministic_core_agents_mode_removed`，缺 LLM client 返回 `structured_llm_client_unavailable`。
- [done] `build_physics_problem` 从 DeepAgents/main tool registry 移除；仅保留为 disabled compatibility stub，直接返回 `legacy_build_physics_problem_disabled`，不能生产 `PhysicsProblem`，且不再携带可注册 tool metadata。
- [done] 移除 legacy problem builder 中的 300/350/300、generic k=1、规则式中英文 PDE 解析和任何可执行默认注入；自然语言缺 LLM/校验失败时只能返回 typed missing inputs，不能静默生成可执行物理问题。
- [done] 更新 `PHYSICSOS_SYSTEM_PROMPT` 和 DeepAgents CLI prompt，说明 capability-aware routing。
- [done] 强制 PhysicsOS CLI/DeepAgents 启动环境使用 `PYTHONUTF8=1` 和 `PYTHONIOENCODING=utf-8`。
- [done] 记录 UTF-8 中文约束：不在业务层做 mojibake/GBK 猜测修复。
- [done] 清理 CLI 自然语言识别中的乱码中文常量，真实中文关键词直接使用 UTF-8；源码/测试不保留 mojibake 中文样例。
- [done] 清理测试中的 mojibake 中文输入样例，统一改为真实 UTF-8 中文；项目不再用乱码 marker 代表中文 case。
- [todo] 将 `run_physicsos_workflow()` 逐步迁移到显式 LangGraph `StateGraph`，保留 public API；当前已先把 typed state gate 记录接入现有 workflow。
- [done] 新增 route/execution/verification gates 到 `PhysicsOSWorkflowState`。
- [done] DeepAgents subagents 只做文档/审阅/artifact 辅助，不复用 workflow node 名称。

### L. 回归和验收

- [done] 当前中文铝杆 CLI 命令不再走 deterministic fallback：有 LLM client 时进入统一 typed workflow；LLM/schema 失败时返回 missing_inputs，不得注入 300/350/300 默认值或静默 accepted。
- [done] 求解 artifact 不再记录或使用内置材料库来源；材料/物理系数来源只能是用户、LLM、knowledge/search/tool 显式写入的 typed problem 或由显式 typed 系数组合派生。
- [done] 撤销材料搜索 tool 验收项；Tavily/API 仅验证为 DeepAgents 环境变量注入。
- [done] TAPS transient heat demo 在 result/report 中显示能力状态。
- [done] 所有其它 demo/scaffold backend 在报告中必须显示能力状态。
- [done] `fdm_heat_1d` result/report 显式显示 `capability_status=production`。
- [done] 用户中文铝杆 CLI 命令验证：当前真实 config 下 structured LLM extraction 未通过 schema，CLI 返回 `structured_llm_output`/`structured_llm_problem_extraction_failed` 且退出码 1；未执行旧模板、未生成默认数值求解。
- [done] TAPS focused tests 通过：默认 demo 隔离、`taps_separated_galerkin` artifact 输出、contract negative cases、真实 UTF-8 中文 CLI detection。
- [done] 本轮改造后 `tests/test_scaffold.py` 全量通过：206 passed，2 warnings（deepagents deprecation warnings）。

### M. Workspace Path System

- [done] CLI/DeepAgents 启动环境导出统一 workspace：`PHYSICSOS_WORKSPACE` 为 shell/Python 的真实 cwd 根目录，`PHYSICSOS_AGENT_WORKSPACE=/workspace` 为 agent/filesystem tool 的虚拟路径别名，`PHYSICSOS_CWD` 与真实 workspace 保持一致。
- [done] `resolve_workspace_path()` 统一支持 `/workspace/...`、workspace-relative 路径和本机绝对路径；artifact 读取热点改用该 resolver，避免斜杠/反斜杠和 cwd 不一致导致找不到文件。
- [done] DeepAgents prompt 明确要求 filesystem tools 使用 `/workspace/...`，shell/Python 使用 cwd-relative 或 `os.environ["PHYSICSOS_WORKSPACE"]`，三种写法指向同一文件。
- [done] 区分用户显式 `PHYSICSOS_WORKSPACE` 与 PhysicsOS CLI 自动注入 workspace；自动注入值带 `PHYSICSOS_WORKSPACE_SOURCE=physicsos_cli_auto` 和 `PHYSICSOS_WORKSPACE_AUTO_VALUE`，不会污染后续只设置 `PHYSICSOS_HOME` 的同进程调用。
- [done] DeepAgents shell backend 在 PhysicsOS patch 中固定使用 UTF-8 解码 stdout/stderr，避免中文用户名路径在 agent/TUI 输出中变成 mojibake；项目继续坚持终端、CLI、Python、agent 中文统一 UTF-8，不做业务层 GBK 猜测修复。
- [done] 增加 DeepAgents/TUI 相关路径测试：虚拟 `/workspace` 文件工具写入、环境变量注入、Python 子进程解析 `/workspace`、LocalShellBackend cwd/env 一致性，以及自动 workspace 不覆盖后续 `PHYSICSOS_HOME`。

### N. 单元测试清理与新主路径验收

- [done] 删除重复的 tetra elasticity mesh_graph 旧测试，保留 tetra10 mesh_graph 用例覆盖“arbitrary mesh graph 不能被误标为 separated TAPS execution”。
- [done] 收窄 legacy scalar elliptic 2D 非零 Dirichlet 测试：只验证旧显式 legacy 分支的边界 lifting 行为，状态必须是 `needs_review`，能力等级为 `experimental`，不能再断言 `success`。
- [done] 更新 `fdm_heat_1d` 生产求解器测试：独立 residual/BC/IC/range/slice 校验通过时断言 `success` 和 `capability_status=production`。
- [done] 更新 `taps_separated_galerkin` 生产主路径测试：SeparatedOperatorIR + TAPSExecutionPlan + residual/artifacts/baseline policy 通过时断言 `success` 和 `capability_status=production`。
- [done] 保留 required baseline 缺失和 baseline mismatch 的降级/阻断测试，baseline 仍是可选 confidence evidence，不是默认成功前提。
- [done] 保留 LLM structured numerical plan 失败测试，断言 workflow 记录 typed validation retry 且不进入 solver execution。
- [done] `tests/test_scaffold.py` 全量通过：222 passed，2 warnings；warnings 来自 deepagents deprecation，不影响本轮 TAPS/typed workflow 语义。
- [done] 真实 config 前 4 个案例暴露旧 `TAPSProblem` 早期 contract review 过严：它用 legacy shell 的 basis/quadrature/tolerance 默认值阻断新主路径，且不检查真正的 execution IR。
- [done] `_run_taps_agent()` 不再调用早期 `taps-contract-review-agent`；旧 `TAPSProblem` review 固定写为 accepted/skipped，避免 `needs_retry` 阻断 workflow。
- [done] 当前 hard gate 只保留在 `taps-execution-contract-review-agent`：审查 `TAPSFormulationIR + SeparatedOperatorIR + TAPSExecutionPlan + NumericalSolvePlanOutput`，这是实际执行合同。
- [done] build problem prompt 明确 workflow 会自动生成 solution/residual/TAPS artifacts，不应要求用户提供 artifact URI/path/format。
- [done] 按当前产品目标调整 gate/reviewer/verifier 策略：除 typed schema 无法构造、运行时异常这类真正无法继续的错误外，geometry/TAPS/reviewer/verifier 一律作为 advisory/confidence/risk 记录，不阻断 workflow。
- [done] `assess_taps_geometry_separability()` 不再把 raw mesh_graph 或缺少 explicit structured_axes 作为硬阻断；只要 formulation axes 可用，就构造 provisional `SeparatedGeometryOperator` 并以 warnings 标记低置信度。
- [done] `taps-execution-contract-review-agent` prompt 改为 advisory review：优先 accepted，风险写入 warnings；workflow 中 reviewer 给出 needs_retry/failed 时也强制转为 accepted advisory 并继续。
- [done] TAPS agent 没能生成 ready separated execution payload 时返回 `fallback_required` 而不是 `failed`，workflow 继续进入 solver-agent；workflow 不再因 taps handoff failed 早停。
- [done] route/execution/verification gate 记录改为 advisory：报告风险但 `required_actions=[]`，不再作为硬性阻断。
- [done] Geometry planner prompt 明确：1D rod/interval、2D square/rectangle/plate、3D box/cuboid/block 等简单显式几何优先请求 `structured_axes`。
- [done] `GeometryEncoding`、`TAPSGeometryEncodingSpec` 支持 `structured_axes`，`generate_geometry_encoding()` 可生成 `structured_axes.json` artifact。
- [done] `build_geometry_mesh_contract()` 会读取 `structured_axes` artifact 的 `axis_names`、resolution、confidence 并写入 `GeometryNumericalEncoding`。
- [done] LLM geometry planner 即使漏掉 `structured_axes`，对简单显式几何也会自动补入 requested encoding；case1 类 1D 杆不再只能靠 formulation axes provisional 放行。
- [done] `assess_taps_geometry_separability()` 优先识别 contract 中的 explicit `structured_axes`，把 formulation missing bindings 作为 warning，而不是覆盖几何可分离结论。

### O. Universal TAPS Executor Upgrade Plan

目标：将当前 `taps_separated_galerkin` 从结构化 P1 separated Galerkin prototype 升级为 PhysicsOS 的“普适 TAPS 执行器”。这里的“普适”不是让一个内核吞掉所有 PDE/mesh，而是建立一个通用的 TAPS execution framework：任何可写成 separated weak-form contractions 的问题，都走同一套 IR、operator algebra、subspace iteration、residual verification 和 artifact contract；无法 separated 的部分必须被明确标记为需要 compression、domain decomposition、runtime extension 或 full solver baseline。

#### O.1 立即修复当前 2D diffusion-reaction case

- [todo] 修复 `SubspaceLinearSystemKernel._term_scale()` 对 source fixed-axis block 的处理：`operator="source"` 是 axis load vector/linear functional，不是 matrix；fixed-axis 缩并应使用 `dot(source_vector_for_constant(1), fixed_factor)`。
- [todo] 明确 source term 的 axis block 语义：target axis 产生 RHS vector，fixed axes 产生 scalar contraction；非 source/bilinear term 中出现 `operator="source"` 必须 hard fail。
- [todo] 为 2D source term `f * b_x tensor b_y` 增加 regression test，覆盖 target axis 为 x 和 y 时的 ALS 组装。
- [todo] 修复 `_separated_execution_constraints()`：从 `x_min/x_max/y_min/y_max/z_min/z_max` 自动映射 axis endpoint constraints，不能只处理 x 轴。
- [todo] 增加 boundary application audit artifact：记录 requested Dirichlet BC、canonical role、axis/node index、是否 applied、未 applied 原因。
- [todo] 对 homogeneous multi-axis Dirichlet 允许 axis endpoint essential constraints；对 nonzero multi-axis Dirichlet 默认要求 lifting 或 boundary penalty，不允许静默把非零值逐轴硬塞。
- [todo] 用真实 API 重跑中文 steady diffusion-reaction unit square case，验收标准是 backend=`taps:separated_galerkin`，artifacts 包含 factors、axis operators、iteration history、reconstruction samples，report 不再只给 `fenicsx:scaffold`。

#### O.2 统一 TAPS executable algebra

- [todo] 将 `SeparatedOperatorTermIR` 的 axis block 类型区分为 matrix block、vector block、scalar coefficient block、boundary functional block；当前只用 `operator` 字符串不足以表达 source/boundary/constraint 的线性泛函语义。
- [todo] 定义 executable operator vocabulary v2：`mass`, `stiffness`, `gradient`, `derivative`, `identity`, `load`, `boundary_mass`, `boundary_load`, `trace`, `normal_flux`, `time_mass`, `parameter_mass`。
- [todo] 将 `source` 从 matrix operator 名称迁移为 linear functional kind；保留旧 `operator="source"` 作为兼容输入并在 normalizer 中转为 `load`。
- [todo] 新增 `SeparatedOperatorNormalizer`：拆分 sum-of-products、规范 coefficient refs、规范 geometry refs、把 weak-form provenance terms 转成 executable product terms。
- [todo] 新增 `SeparatedOperatorSemanticValidator`：检查 field/test-field consistency、axis coverage、operator arity、matrix/vector block 位置、coefficient/source/geometry binding、BC/IC/source support。
- [todo] 把 normalizer 和 semantic validator 接入 LLM retry feedback，让 LLM 修 proposal；backend 仍保留最终 hard validation。
- [todo] 为每个 executable term 写入 `target="lhs"|"rhs"|"constraint"`，避免仅靠 `role=="source"` 推断 RHS。
- [todo] 支持 sign convention 显式字段：`residual_sign`, `rhs_sign`, 或 normalized residual form `A(u)-b=0`，防止 source/reaction 符号被 prompt 猜测。

#### O.3 真正的 weak-form assembly contract

- [todo] 将 `SubspaceLinearSystemKernel` 从“按 role 粗略组装”升级为“按 executable term target 和 block type 组装”。
- [todo] 对 bilinear term：target axis 使用 matrix block，fixed axes 使用 quadratic contractions。
- [todo] 对 linear/source term：target axis 使用 vector/load block，fixed axes 使用 linear contractions。
- [todo] 对 boundary term：根据 `domain_ref`/boundary role 生成 trace/boundary load 或 penalty block。
- [todo] 对 constraint term：生成 saddle-point、penalty、lifting 或 nullspace/gauge policy，不要混入普通 source。
- [todo] 支持多 field/mixed weak form 的 block matrix assembly：field_in、field_out/test_field、component index、coupling block。
- [todo] 支持 vector fields 的 component-wise separated factors，不能把 vector PDE 当 scalar PDE 近似执行。
- [todo] 支持 time axis 的 mass/derivative block 和 initial-slice functional，作为 TAPS S-P-T execution 的一等公民。
- [todo] 支持 parameter axes 的 mass/identity/collocation block，支持 parameter sweep/surrogate 作为 TAPS 的核心场景。

#### O.4 Basis 与 geometry 泛化

- [todo] 将 `AxisBasisKernel` 从只支持 uniform P1 扩展为 basis registry：P1、P2/Pk、spectral/Chebyshev、C-HiDeNN-compatible local basis、collocation basis。
- [todo] 让 `TAPSBasisConfig` 真实控制 basis family、order、quadrature order、node distribution、boundary basis treatment，而不是被 P1 hardcode 忽略。
- [todo] 支持 per-axis nonuniform nodes 和 physical coordinate mapping。
- [todo] 实现 separated geometry factor application：metric、Jacobian、volume measure、boundary measure、domain indicator 不能再全部当 1。
- [todo] 增加 geometry factor rank/compression error 对 residual 和 acceptance 的影响记录。
- [todo] 对 arbitrary mesh_graph 明确不等同 TAPS separated execution；需先经过 geometry compression 或 domain decomposition 生成 `SeparatedGeometryOperator`。
- [todo] 设计 domain decomposition：多个 separable patches/subdomains，每个 patch 一个 separated operator，patch/interface terms 通过 mortar/penalty/continuity constraint 连接。
- [todo] 设计 immersed/fictitious-domain path：occupancy/domain indicator 作为 separated coefficient factor，并记录 compression error。

#### O.5 Boundary/IC/constraint 普适化

- [todo] 建立 `BoundaryConditionCompiler`：Dirichlet、Neumann、Robin、periodic、symmetry、interface、wall/inlet/outlet 等边界语义转为 executable separated boundary terms。
- [todo] Homogeneous Dirichlet：支持 axis endpoint elimination/essential constraints。
- [todo] Nonzero Dirichlet：默认使用 lifting `u=g+w` 或 penalty/Nitsche；多维 rank factors 不直接施加非零 endpoint value。
- [todo] Neumann/source boundary：生成 boundary load vector，支持 boundary_measure factor。
- [todo] Robin：生成 boundary mass + boundary load。
- [todo] Periodic：生成 DOF identification 或 constraint system。
- [todo] Pure Neumann/pressure gauge：要求 gauge/mean constraint，缺失时 hard fail 并给 repair instruction。
- [todo] Initial condition：对 time-dependent TAPS 生成 initial slice constraint 或 projection vector，并记录 IC residual。

#### O.6 Coefficients/source/constitutive 普适化

- [todo] 将 scalar constant coefficient 扩展为 `SeparatedCoefficientField`：constant、axis-separable list、low-rank CP factors、piecewise region field、parameter-dependent function。
- [todo] LLM 只能提议 coefficient mapping；实际 numeric coefficient expansion/compression 由 deterministic code 生成或由 reviewed runtime extension 提供。
- [todo] 支持 source term 的 spatial dependence：constant、separable expression、tabulated nodal data、region-supported load、time/parameter dependence。
- [todo] 对 nonseparable coefficient/source 提供 compression plan：ALS/SVD/Tucker/CP decomposition，记录 tolerance、rank、error。
- [todo] 建立 coefficient alias/canonical registry，供 formulation、numerical plan、operator proposal、contract review 共用，避免 prompt 内散落 hardcoded names。
- [todo] 对 declared field/coefficient name 冲突做全链路校验：field `c` 不能当 reaction coefficient；field `u` 不能当 velocity coefficient，除非它是给定 frozen/advection field 且 provenance 明确。
- [todo] 支持 nonlinear constitutive callbacks：只能是 typed, reviewed, sandboxed runtime extension；核心 executor 调 deterministic callback，不让 LLM 在求解循环中生成数值。

#### O.7 Subspace iteration / rank enrichment 升级

- [todo] 将 rank enrichment 改成 residual-correction enrichment：新增 rank 解上一轮 residual，而不是每一 rank 重复解完整 RHS。
- [todo] 为 ALS/PGD/TAPS subspace iteration 实现 proper normalization、orthogonalization、line search/damping、stagnation detection、restart policy。
- [todo] 支持 multiple fields/components 的 block ALS sweep order。
- [todo] 支持 nonlinear Picard/Newton：operator update callback 产生新 separated operator，Jacobian/residual 的一致性由 deterministic callback 保证。
- [todo] 支持 time-slab continuation：每个 slab 的 IC 来自前一 slab projection，记录 slab residual 和 interface mismatch。
- [todo] 支持 adaptive rank：基于 residual reduction、compression error、baseline spot checks 自动增加 rank 或停止。
- [todo] 支持 adaptive basis/order refinement：基于 residual indicator 或 boundary error 建议 refine axis nodes/order。
- [todo] 对每次 axis solve 使用稀疏/带状 solver，而不是 dense Gaussian elimination；大规模 axes 不能继续 dense solve。
- [todo] 为 ill-conditioned/singular subproblem 增加 regularization、nullspace/gauge handling 和 clear diagnostic。

#### O.8 Residual/verification/artifacts 升级

- [todo] 将 dense diagnostic residual 升级为 assembled weak residual norm：记录 absolute、relative、per-field、per-term、per-axis residual。
- [todo] 单独计算 boundary residual/BC error、IC error、constraint violation、conservation/flux imbalance。
- [todo] artifacts 必须包含 reconstructable solution field：axis metadata、rank factors、weights、basis info、coordinate mapping、sample/reconstruction helper。
- [todo] 对 2D/3D scalar field 自动生成 dense preview grid 或 contour-ready artifact，不只输出 residual_summary。
- [todo] 对 vector/mixed fields 输出 component-wise field artifacts。
- [todo] 输出 operator audit：每个 weak-form term 到 executable term、coefficient values、geometry factors、boundary constraints 的映射。
- [todo] 输出 solver audit：rank history、axis sweep history、linear solve stats、condition estimates、stagnation/restart events。
- [todo] verification 不接受 backend 自报 success；必须能从 artifacts 重建 residual/BC/IC checks。

#### O.9 Agentic vs deterministic 边界

- [todo] 保持数值内核纯 deterministic：basis generation、operator assembly、linear/nonlinear solves、rank update、residual、BC/IC verification、artifact writing 不能 LLM-driven。
- [todo] LLM/agentic 只参与 compiler/proposal/review：natural language -> problem contract、weak-form proposal、separated operator proposal、execution plan proposal、repair instruction。
- [todo] 所有 LLM proposal 都必须经过 schema validation、semantic validation、normalization、final execution contract validation。
- [todo] 对 validator failure 使用 LLM retry；retry 仍失败时返回 `needs_operator_review` 或 fallback，不让 LLM 在 runtime 内临时改数值。
- [todo] 长远可做 deterministic compiler library：常见 weak-form algebra 直接从 symbolic/UFL-like IR 编译到 separated operator，LLM 只负责把自然语言变成 symbolic IR。
- [todo] 对 case-local runtime extension 使用 agentic authoring 但严格 sandbox/review/test；提升到 core executor 前必须有 deterministic tests 和 verification evidence。

#### O.10 分阶段验收

- [todo] P0：修复 2D source fixed-axis contraction、x/y/z endpoint constraints、boundary audit；中文 diffusion-reaction case 跑出 `taps:separated_galerkin` artifacts。
- [todo] P1：建立 executable algebra v2 和 normalizer/validator；覆盖 scalar elliptic/diffusion/reaction/source/Dirichlet/Neumann/Robin 的 structured axes cases。
- [todo] P2：实现 separated coefficient/source fields 和 geometry factor application；支持非单位矩形、非均匀轴、piecewise/separable coefficient。
- [todo] P3：实现 nonzero Dirichlet lifting、pure Neumann gauge、periodic/interface constraints。
- [todo] P4：实现 residual-correction rank enrichment、adaptive rank、sparse axis solvers。
- [todo] P5：支持 transient S-P-T execution：time axis、IC projection、time slab continuation。
- [todo] P6：支持 multi-field/vector/mixed weak forms 的 block separated execution。
- [todo] P7：支持 geometry compression/domain decomposition/immersed indicator，并建立 compression error gate。
- [todo] P8：建立 FEniCSx baseline spot-check bridge，用于 high-risk cases 和 regression validation。
- [todo] P9：把 capability registry 与 prompt/schema 合并为单一 source of truth，自动生成 LLM prompt snippets、validator allowlist 和 docs。
