# TAPS as the Primary PhysicsOS Solver

TAPS should be treated as PhysicsOS 的主力 solver backend，而不是普通的 neural surrogate。它的核心价值在于：不依赖训练数据，直接从 governing equations 构造可验证、可收敛、可压缩的 equation-driven surrogate。

参考论文：

- Tensor-decomposition-based A Priori Surrogate (TAPS) modeling for ultra large-scale simulations: https://arxiv.org/abs/2503.13933

---

## 1. Positioning

TAPS 不是 FNO、UNet、MeshGraphNet 那种下载权重后推理的模型。

TAPS 是：

```text
equation-driven surrogate solver
```

它直接从 PDE、边界条件、初始条件、参数空间、时间空间构造 reduced-order surrogate。

在 PhysicsOS 中，solver 优先级应该是：

```text
1. TAPS backend
2. Neural operator / downloaded surrogate
3. Full solver fallback
```

TAPS 适合承担首要 solver，因为它的误差来源是数值分析意义上的误差，而不是黑盒泛化误差。

---

## 2. Core Idea

传统 full solver 解的是固定参数下的场：

```text
u(xs, t | parameter = p0)
```

TAPS 直接解一个包含空间、参数、时间的函数：

```text
u(xs, xp, xt)
```

其中：

```text
xs = spatial coordinates
xp = design / material / boundary / geometry parameters
xt = time
```

这意味着 TAPS 天然就是 surrogate。它不是先跑很多 full solver 样本再训练，而是直接在 space-parameter-time domain 上求解 governing equation。

---

## 3. Technical Stack

TAPS = C-HiDeNN interpolation + tensor decomposition + generalized space-parameter-time Galerkin weak form + subspace iteration.

### 3.1 C-HiDeNN Basis

C-HiDeNN 是带有限元性质的可学习/可调 basis。

它保留有限元的重要性质：

```text
local support
Kronecker-delta property
partition of unity
Dirichlet BC can be enforced naturally
Gaussian quadrature integration
interpretable approximation space
```

相比 MLP/PINN：

```text
MLP/PINN -> boundary often handled by penalty
C-HiDeNN -> boundary can be built into interpolation

MLP/PINN -> stochastic optimization, convergence weak
C-HiDeNN -> closer to FEM/Galerkin numerical method
```

C-HiDeNN 的关键超参数：

```text
s = patch size
a = dilation parameter
p = reproducing polynomial order
```

这些决定 basis 的局部连接、尺度归一化和可复现多项式阶数。

### 3.2 Tensor Decomposition

如果直接离散 D 维空间：

```text
DoF = n^D
```

TAPS 使用 CP tensor decomposition，把解写成：

```text
u(x1, x2, ..., xD) ≈ Σ_m u1_m(x1) * u2_m(x2) * ... * uD_m(xD)
```

压缩后：

```text
DoF = M * D * n
```

其中：

```text
M = tensor rank / number of modes
D = total dimensions, including space + parameters + time
n = grid points per dimension
```

这是 TAPS 能处理 ultra large-scale / high-dimensional simulation 的根本原因。

### 3.3 Space-Parameter-Time Galerkin Form

TAPS 不是拟合数据，而是把 governing equation 投影到 space-parameter-time trial/test space。

它求的是：

```text
Find u_TD(xs, xp, xt)
such that PDE residual is weakly zero over the S-P-T domain.
```

这使得 TAPS 的 residual、conservation error、rank error、quadrature error 都可以被显式检查。

### 3.4 Subspace Iteration

CP tensor decomposition 让 Galerkin system 变成非线性耦合问题。

TAPS 用 subspace iteration：

```text
fix all factor directions except one
solve/update one directional factor
cycle through all dimensions
repeat until residual/convergence tolerance is satisfied
```

这相当于在低秩 tensor factor space 里做方向性 Galerkin solve。

---

## 4. Why TAPS Is Different From Neural Operators

Neural operator 路线：

```text
generate dataset
train model
infer new case
trust depends on training distribution
```

TAPS 路线：

```text
define PDE
define S-P-T domain
assemble weak form
solve low-rank surrogate directly
verify residual
```

因此 TAPS 的商业价值更适合工程仿真：

```text
no expensive offline DNS data generation
accuracy controlled by numerical hyperparameters
residual and convergence are inspectable
parameter sweep is built in
storage can be reduced dramatically
```

TAPS 不是替代所有 neural operator。它应该成为主力 solver，neural operator 作为：

```text
fast approximate fallback
geometry prior
warm start
corrector
OOD heuristic
```

---

## 5. Accuracy Sources

TAPS 的精度主要由这些误差组成：

```text
basis approximation error
tensor rank truncation error
Galerkin residual error
quadrature error
subspace iteration error
nonlinear material update error
domain/slab decomposition error
```

这些误差可控、可检查、可 refine。

主要精度旋钮：

```text
n       -> discretization points per dimension
M       -> CP rank / modes
p       -> reproducing polynomial order
s       -> C-HiDeNN patch size
a       -> dilation parameter
slabs   -> time / parameter / geometry decomposition
quad    -> quadrature order
tol     -> residual and nonlinear iteration tolerance
```

Adaptive strategy:

```text
run TAPS
→ compute S-P-T Galerkin residual
→ sample parameter slices
→ compare selected slices with full solver / local FEM
→ if rank error high: increase M
→ if spatial error high: refine n or increase p/s
→ if non-smooth: split slab/domain
→ if nonlinear residual high: run global-local update
```

---

## 6. Where TAPS Should Be Used First

TAPS-first domains:

```text
transient heat equation
diffusion
reaction-diffusion
Poisson
Helmholtz
linear elasticity
thermal-structural coupling
parameterized manufacturing process
LPBF / additive manufacturing thermal simulation
IC / semiconductor thermal simulation
```

TAPS should not be the first backend for:

```text
high-Re turbulence
shock-dominated flow
violent multiphase flow
fracture/contact
strongly discontinuous moving interfaces
unstructured CAD-heavy CFD without parameterized geometry
```

Those cases should use neural operator warm-start or full solver fallback.

---

## 7. Geometry Strategy

TAPS works best when the domain can be represented in tensor-product or parameterized coordinates.

For real PhysicsOS geometry, Geometry + Mesh Agent must convert CAD/mesh into one of:

```text
parametric geometry coordinates
SDF / occupancy mask
immersed boundary representation
domain decomposition patches
geometry parameters as xp
laplacian / spectral basis
```

This is why Geometry Agent is not just a mesher. It must become a TAPS-compatible geometry compiler.

For complex geometry:

```text
CAD/STL
→ repair and label
→ SDF or occupancy field
→ parameterized shape descriptor
→ TAPS tensor domain or decomposed patches
```

If this conversion fails or creates high residuals, route to neural operator or full solver.

---

## 8. TAPS Backend Design

PhysicsOS should implement:

```text
TAPSBackend
├─ TAPSProblemBuilder
├─ OperatorWeakFormCompiler
├─ ParameterSpaceBuilder
├─ GeometryToTensorDomainCompiler
├─ CHiDeNNBasisFactory
├─ TensorBasisFactory
├─ GalerkinAssembler
├─ CPFactorSolver
├─ SubspaceIterationSolver
├─ NonlinearGlobalLocalSolver
├─ Evaluator
├─ ResidualEstimator
└─ Verifier
```

Input:

```text
PhysicsProblem
GeometrySpec
OperatorSpec
BoundaryConditionSpec
MaterialSpec
TargetSpec
VerificationPolicy
```

Compiled TAPS problem:

```text
TAPSProblem = {
  spatial_axes,
  parameter_axes,
  time_axis,
  operator_weak_form,
  boundary_conditions,
  initial_conditions,
  material_laws,
  tensor_rank,
  basis_order,
  patch_size,
  dilation,
  slabs,
  quadrature_policy,
  residual_policy
}
```

Output:

```text
TAPSResult = {
  factor_matrices,
  reduced_coefficients,
  reconstruction_metadata,
  sampled_field_refs,
  residual_report,
  convergence_history,
  verification_report
}
```

---

## 9. TAPS Agent

PhysicsOS should include a dedicated `taps-agent`.

Responsibilities:

```text
decide whether TAPS is applicable
compile PhysicsProblem into TAPSProblem
choose parameter axes
choose tensor rank M
choose C-HiDeNN hyperparameters s/a/p
choose S-P-T slabs
assemble weak form
run TAPS backend
inspect residual/convergence
decide refine/rerun/fallback
produce TAPS-specific explanation
```

TAPS Agent is the primary universal equation-driven physics solver compiler in PhysicsOS.
It should not behave like a fixed PDE template selector. It should compile any sufficiently specified physical problem into a typed weak-form IR, then execute available TAPS kernels or report the exact missing assembler/runtime capability.

Required compiler contract:

```text
1. Read PhysicsProblem.
2. Formulate or retrieve governing equations.
3. Convert strong/integral/discrete descriptions into weak-form IR.
4. Define space, parameter, time, and geometry axes.
5. Choose C-HiDeNN basis settings, tensor rank, slabs, quadrature, and residual policy.
6. Compile TAPSProblem.
7. Run available executable TAPS kernels.
8. If no executable kernel exists, return the compiled IR and exact missing runtime capability.
9. Ask knowledge-agent for missing science before giving up.
10. Recommend neural/full-solver fallback only after TAPS compilation or verification fails.
```

Agent-facing TAPS IR:

```text
TAPSCompilationPlan
TAPSWeakFormSpec
TAPSEquationTerm
TAPSAxisSpec
TAPSBasisConfig
TAPSNonlinearConfig
TAPSGeometryEncodingSpec
TAPSProblem
TAPSResidualReport
TAPSRuntimeExtensionSpec
```

Unknown physics should produce `required_knowledge_queries`, not a silent rejection. Examples:

```text
derive governing weak form for a new coupled PDE from a paper
identify constitutive law and coefficients for a material model
retrieve nondimensional assumptions and validation residuals
find boundary/interface conditions for a multiphysics coupling
```

If the missing piece is implementation rather than scientific knowledge, taps-agent should write a reviewed case-local runtime extension draft instead of silently editing core code:

```text
compile weak-form IR
identify missing assembler/kernel
author TAPSRuntimeExtensionSpec artifact under scratch/case workspace
require review before execution or promotion
promote to core backend only with tests
```

Current code state:

```text
executable: 1D transient heat low-rank S-P-T kernel
executable: 1D multi-mode scalar elliptic weak-form assembler for Poisson / steady diffusion / reaction-diffusion / Helmholtz-style operators
executable: 2D multi-mode tensorized scalar elliptic weak-form assembler for rectangular Poisson / steady diffusion / reaction-diffusion / Helmholtz-style operators
executable: 1D/2D nonlinear reaction-diffusion Picard/fixed-point kernels with iteration history
executable: 2-field 2D coupled reaction-diffusion kernel with coupled residual and field artifacts
executable: 2D occupancy-mask geometry-encoded scalar elliptic path, including central-hole masks with masked relaxation
executable: Gmsh mesh -> mesh_graph encoding -> triangle P1/P2 FEM-like Poisson path
executable: triangle cell gradients / P2 quadrature gradients -> sparse stiffness matrix -> Dirichlet Poisson solve
compilable IR: custom OperatorSpec terms, heat/diffusion, Poisson, reaction-diffusion, Helmholtz
missing: Pk beyond quadratic, broader coupled multiphysics families, learned/optimized geometry-encoded TAPS basis, distributed subspace solver
```

DeepAgents role:

```text
main-agent
→ geometry-mesh-agent
→ taps-agent
→ verification-agent
→ postprocess-agent
```

If TAPS fails applicability or verification:

```text
taps-agent
→ solver-agent
→ neural operator / full solver fallback
```

---

## 10. TAPS-First Routing Policy

Default routing:

```text
if operator is explicit
and parameter axes are identifiable
and geometry can be tensorized or decomposed
and solution is expected to be reasonably smooth:
    use TAPS first
else:
    use neural operator or full solver fallback
```

Routing score:

```text
taps_score =
  operator_clarity
+ parameter_axis_quality
+ geometry_tensorizability
+ boundary_condition_compatibility
+ smoothness_expectation
+ verification_budget
- discontinuity_risk
- turbulence/shock/contact risk
```

TAPS should be the first attempted backend for:

```text
thermal
diffusion
Poisson
Helmholtz
reaction-diffusion
linear elasticity
parameterized manufacturing
```

---

## 11. MVP Implementation Plan

### Stage 1: TAPS Thermal

Implement:

```text
transient heat equation
parameterized material coefficients
parameterized heat source
convection/radiation boundary conditions
space-time and space-parameter-time domains
CP rank adaptation
residual estimator
selected FEniCSx slice validation
```

Target use case:

```text
laser powder bed fusion thermal process
PCB/IC thermal parameter sweep
heat sink parameter design
```

### Stage 2: TAPS Scalar PDE Pack

Add:

```text
Poisson
Helmholtz
reaction-diffusion
advection-diffusion
```

### Stage 3: TAPS Solid/Thermal-Structural

Add:

```text
linear elasticity
thermoelastic coupling
modal/eigen surrogate
```

### Stage 4: Geometry-Aware TAPS

Add:

```text
SDF geometry parameters
immersed boundary penalties
domain decomposition
patch-wise TAPS
complex CAD conversion
```

### Stage 5: Nonlinear and Global-Local TAPS

Add:

```text
temperature-dependent material
moving source
phase/process variables
global-local nonlinear update
slab adaptation
```

---

## 12. Engineering Rule

Do not treat TAPS like a checkpoint model.

TAPS artifacts should be:

```text
operator weak form
parameter axes
basis config
factor matrices
rank history
residual history
verification slices
reconstruction metadata
```

This makes TAPS results reusable, compressible, auditable, and suitable for engineering use.
