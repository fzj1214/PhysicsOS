# PhysicsOS 的 KS-DFT-TAPS 路线设计

## 结论

PhysicsOS 做 DFT 的唯一路线是 **KS-DFT-TAPS**。当前阶段不接入 QE、VASP、CP2K、ELSI 等外部 DFT 引擎，也不保留 OF-DFT 并行路线。

正确分工是：

```text
pymatgen + seekpath/spglib:
  负责晶体结构、对称性、标准化、倒空间、k 点、band path 等高稳定性材料工程操作

local pseudopotential library tools:
  负责索引本地 VASP PAW PBE POTCAR 库，提取 TITEL/ZVAL/ENMAX/LEXCH/hash/provenance，按结构元素选择赝势，并生成 case-local pseudopotential context

TAPS / C-HiDeNN-TD / KS-DFT prompt:
  负责 Kohn-Sham 方程、低秩 tensor basis、occupied subspace、SCF residual、CheFSI、LRDM 的推导和 case-local kernel

PhysicsOS DFT agent:
  把上述工具产物写成 analysis files，放进 context window，并要求 derivation/implementation agent 明确消费这些产物
```

主流程：

```text
用户材料问题 / CIF / POSCAR / 结构文本
-> materials-preprocess-agent 调用 pymatgen/seekpath/spglib 工具
-> 生成 standardized_structure、symmetry、reciprocal_lattice、kmesh、kpath 等稳定产物
-> pseudopotential tools 选择本地 PBE POTCAR 元数据，生成 valence/provenance/cutoff context
-> ks-dft-analysis-agent 生成 KS-DFT-TAPS problem statement
-> context window 汇总 TAPS 模板、KS-DFT notes、材料工具产物
-> ks-dft-taps-derivation-agent 推导 Kohn-Sham-TAPS 形式
-> ks-dft-taps-implementation-agent 写 case-local kernel
-> ks-dft-verification-agent 检查 charge、orthonormality、SCF residual、rank/grid/k-point 收敛
```

**关键原则**：不要让 prompt 手写晶体学标准化和 k 点路径。prompt 只负责声明必须调用哪些工具、如何读取工具产物、哪些量进入 TAPS 推导。

赝势库路径解析优先级：

```text
1. tool input: library_root
2. environment: PHYSICSOS_PSEUDOPOTENTIAL_DIR
3. config: ~/.physicsos/config.json -> pseudopotentials.libraries.<id>.root
```

不同设备不应依赖某个固定的本机赝势库路径。用户或管理员只需要在本机 config 中写入本机路径；case artifact 只保存 metadata、hash、library_id、selection policy 和 provenance，不保存 POTCAR 正文。

## 文献和接口依据

本设计基于：

- `taps.md` 与 TAPS 论文：TAPS 是 data-free、equation-driven surrogate，核心是 C-HiDeNN-TD、space-parameter-time Galerkin weak form 和 subspace iteration。
- `2509.11447v1.pdf` / `ARCHITECTURE.md`：PhysicsOS 应遵循 paper-style CAE agent，用 analysis files、tools、resources、context-window examples 生成推导、实现和验证。
- 本地 KS-DFT / tensor-basis 参考论文笔记：KS-DFT 可以用 Tucker tensor basis、additive separable Hamiltonian、localized 1D basis 和 Chebyshev-filtered subspace iteration 做 reduced-order scaling。
- 本地 CheFSI 参考论文笔记：CheFSI 通过更新 occupied eigenspace 避免每个 SCF 步完整对角化。
- 本地 LRDM 参考论文笔记：LRDM 通过低秩 dielectric response 预条件 Kohn-Sham SCF residual。
- 本地 KS-DFT 数值方法参考笔记：KS-DFT 的核心难点是 nonlinear SCF、本征/密度矩阵、Hartree Poisson、XC、离散化和可扩展性。
- pymatgen 官方文档：`Structure.from_file` / `Structure.from_str`、`Structure.to`、`SpacegroupAnalyzer`、band/DOS 和材料 IO 能力。
- SeekPath 官方文档：`seekpath.get_path` 返回 HPKOT 高对称路径，输入结构采用 spglib cell tuple。
- spglib 官方文档：`get_symmetry_dataset`、`standardize_cell`、`find_primitive`、`get_ir_reciprocal_mesh` 等用于对称性、标准化和不可约 k 网格。

## Prompt 里写，还是做成工具函数？

结论：**必须做成可暴露给 agent 的工具函数，并在 prompt 中强制使用这些工具函数。**

原因：

- k-path、primitive/conventional cell、空间群 setting、倒格矢 convention 很容易被 LLM 写错。
- pymatgen/seekpath/spglib 已经把这些规则工程化，适合作为 deterministic tool layer。
- prompt 的职责是描述工具契约，而不是复刻库逻辑。
- 工具函数输出应写入 case artifacts，让后续 derivation/implementation/verification 都引用同一份稳定数据。

推荐规则：

```text
Must call tools for:
  structure parsing, structure serialization, symmetry dataset, primitive/conventional cell,
  reciprocal lattice, k-mesh, irreducible k-points, high-symmetry k-path, site/species ordering.

Must not derive by prompt:
  space group, Wyckoff labels, primitive cell transform, high-symmetry labels,
  reciprocal lattice convention, k-path segment labels, irreducible k-point reduction.

May derive by TAPS prompt:
  KS matrix form, tensor basis, occupied subspace update, SCF residual,
  CheFSI update, LRDM preconditioner, rank/grid/k-point verification.
```

## Agent 接入设计

### 1. materials-preprocess-agent

新增专门 agent，位于 KS-DFT-TAPS 前处理阶段。

职责：

- 读取 CIF/POSCAR/JSON/结构文本。
- 调用 pymatgen/spglib/seekpath 工具。
- 生成标准化材料 artifacts。
- 写清楚所有 tolerance、cell transform、species ordering、k-point convention。
- 不推导 Kohn-Sham 方程，不写 TAPS kernel。

输出：

```text
/cases/<case_id>/materials/source_structure.json
/cases/<case_id>/materials/structure_standardized.json
/cases/<case_id>/materials/structure_primitive.json
/cases/<case_id>/materials/structure_conventional.json
/cases/<case_id>/materials/symmetry_dataset.json
/cases/<case_id>/materials/reciprocal_lattice.json
/cases/<case_id>/materials/kmesh.json
/cases/<case_id>/materials/irreducible_kpoints.json
/cases/<case_id>/materials/kpath_seekpath.json
/cases/<case_id>/materials/kpath_pymatgen.json
/cases/<case_id>/materials/materials_preprocess_report.md
```

### 2. ks-dft-analysis-agent

职责：

- 读取 materials-preprocess-agent 的 artifacts。
- 生成 `ks_dft_problem.json`。
- 明确哪些材料量来自工具，哪些物理假设来自用户或默认 policy。
- 把缺失项写入 `ks_dft_open_questions.md`。

输出：

```text
/cases/<case_id>/problem/problem_statement.md
/cases/<case_id>/problem/ks_dft_problem.json
/cases/<case_id>/problem/ks_dft_open_questions.md
```

### 3. ks-dft-taps-derivation-agent

职责：

- 必须读取 `materials_preprocess_report.md`、`symmetry_dataset.json`、`kmesh.json`、`kpath_seekpath.json`。
- 在推导中明确 `k` 轴、倒格矢、cell transform 来自工具产物。
- 只推导 Kohn-Sham-TAPS 数学对象。

输出：

```text
/cases/<case_id>/taps/ks_dft_derivation_prompt.md
/cases/<case_id>/taps/ks_dft_derivation.md
/cases/<case_id>/taps/ks_dft_implementation_notes.md
```

### 4. ks-dft-taps-implementation-agent

职责：

- 读取标准化结构和 k-point artifacts。
- 实现 case-local KS-DFT-TAPS kernel。
- 不重新判断空间群或 k-path。
- 若材料工具产物缺失，直接失败。

输出：

```text
/cases/<case_id>/taps/kernel.py
/cases/<case_id>/taps/ks_dft_execution_plan.json
/cases/<case_id>/taps/ks_dft_runtime_metadata.json
/cases/<case_id>/taps/ks_dft_solution_summary.json
```

### 5. ks-dft-verification-agent

职责：

- 验证 charge、orthonormality、SCF residual、Hermiticity、Poisson residual。
- 验证 rank/grid/k-point refinement。
- 检查 implementation 是否使用了 materials artifacts 中的 standardized structure 和 k-point data。
- 当前阶段不调用外部 DFT 引擎做 reference。

输出：

```text
/cases/<case_id>/verification/ks_dft_verification_plan.md
/cases/<case_id>/verification/convergence_study.py
/cases/<case_id>/verification/report.md
/cases/<case_id>/verification/report.json
```

## 应暴露给 agent 的工具函数

下面是建议的工具接口。它们应成为 PhysicsOS tool registry 中可被 `materials-preprocess-agent`、`ks-dft-analysis-agent`、`ks-dft-verification-agent` 调用的 structured tools。

### A. 结构读取和序列化

#### `parse_material_structure`

用途：读取 CIF/POSCAR/JSON/字符串结构，返回 pymatgen 结构 JSON。

底层建议：

- `pymatgen.core.Structure.from_file`
- `pymatgen.core.Structure.from_str`
- `pymatgen.io.cif.CifParser`
- `pymatgen.io.vasp.Poscar`

输入：

```json
{
  "case_id": "string",
  "source_path": "string|null",
  "source_text": "string|null",
  "format": "cif|poscar|json|auto",
  "primitive": false
}
```

输出：

```json
{
  "structure_ref": "/workspace/cases/<case_id>/materials/source_structure.json",
  "formula": "Si2",
  "num_sites": 2,
  "lattice": [[...], [...], [...]],
  "species": ["Si", "Si"],
  "frac_coords": [[...], [...]],
  "warnings": []
}
```

#### `write_material_structure`

用途：把标准结构写成 CIF/POSCAR/JSON，保证后续工具使用同一份结构。

底层建议：

- `Structure.to`
- `Poscar(structure).get_string`

输入：

```json
{
  "case_id": "string",
  "structure_ref": "string",
  "format": "cif|poscar|json",
  "filename": "string"
}
```

输出：

```json
{
  "artifact_ref": "string",
  "format": "string"
}
```

#### `validate_material_structure`

用途：检查结构是否可用于 KS-DFT-TAPS。

检查项：

- lattice determinant positive。
- fractional coordinates finite。
- species valid。
- site count > 0。
- minimum inter-site distance。
- partial occupancy warning。
- oxidation/magmom site properties warning。

底层建议：

- `Structure.is_valid`
- `Structure.distance_matrix`
- `Composition`

输入：

```json
{
  "case_id": "string",
  "structure_ref": "string",
  "min_distance_angstrom": 0.5
}
```

输出：

```json
{
  "valid": true,
  "errors": [],
  "warnings": [],
  "min_distance_angstrom": 2.35
}
```

### B. 结构标准化和对称性

#### `analyze_spacegroup`

用途：用 pymatgen/spglib 获取空间群、点群、Wyckoff、对称操作。

底层建议：

- `pymatgen.symmetry.analyzer.SpacegroupAnalyzer`
- `get_space_group_symbol`
- `get_space_group_number`
- `get_symmetry_dataset`
- `get_symmetry_operations`

输入：

```json
{
  "case_id": "string",
  "structure_ref": "string",
  "symprec": 1e-5,
  "angle_tolerance": -1.0
}
```

输出：

```json
{
  "symmetry_ref": "/workspace/cases/<case_id>/materials/symmetry_dataset.json",
  "spacegroup_symbol": "Fd-3m",
  "spacegroup_number": 227,
  "point_group": "m-3m",
  "hall": "...",
  "wyckoffs": [],
  "equivalent_atoms": [],
  "symprec": 1e-5,
  "warnings": []
}
```

#### `standardize_crystal_structure`

用途：生成 standardized、primitive、conventional 三类结构，并记录 transform。

底层建议：

- `SpacegroupAnalyzer.get_primitive_standard_structure`
- `SpacegroupAnalyzer.get_conventional_standard_structure`
- `SpacegroupAnalyzer.get_refined_structure`
- `spglib.standardize_cell`
- `spglib.find_primitive`

输入：

```json
{
  "case_id": "string",
  "structure_ref": "string",
  "symprec": 1e-5,
  "angle_tolerance": -1.0,
  "keep_site_properties": true
}
```

输出：

```json
{
  "standardized_ref": "string",
  "primitive_ref": "string",
  "conventional_ref": "string",
  "transformation_report_ref": "string",
  "species_order": ["Si"],
  "warnings": []
}
```

#### `compare_crystal_structures`

用途：比较两个结构是否等价，避免标准化前后误配。

底层建议：

- `pymatgen.analysis.structure_matcher.StructureMatcher`

输入：

```json
{
  "case_id": "string",
  "structure_ref_a": "string",
  "structure_ref_b": "string",
  "ltol": 0.2,
  "stol": 0.3,
  "angle_tol": 5.0
}
```

输出：

```json
{
  "match": true,
  "rms_dist": 0.0,
  "max_dist": 0.0,
  "warnings": []
}
```

#### `reduce_lattice_cell`

用途：生成 Niggli/Delaunay reduced cell，用于稳定化倒空间和 k 点。

底层建议：

- `spglib.niggli_reduce`
- `spglib.delaunay_reduce`
- pymatgen lattice reduction helpers if available

输入：

```json
{
  "case_id": "string",
  "structure_ref": "string",
  "method": "niggli|delaunay",
  "eps": 1e-5
}
```

输出：

```json
{
  "reduced_lattice": [[...], [...], [...]],
  "reduced_structure_ref": "string",
  "warnings": []
}
```

### C. 倒空间和 k 点

#### `compute_reciprocal_lattice`

用途：把标准化晶胞转成倒格矢、体积、单位 convention。

底层建议：

- `Structure.lattice.reciprocal_lattice`
- `Structure.lattice.reciprocal_lattice_crystallographic`

输入：

```json
{
  "case_id": "string",
  "structure_ref": "string",
  "convention": "physics_2pi|crystallographic"
}
```

输出：

```json
{
  "reciprocal_lattice_ref": "string",
  "b_vectors": [[...], [...], [...]],
  "convention": "physics_2pi",
  "units": "1/angstrom",
  "warnings": []
}
```

#### `generate_uniform_kmesh`

用途：生成 Monkhorst-Pack / Gamma-centered k mesh。

底层建议：

- `pymatgen.io.vasp.inputs.Kpoints.automatic_density`
- `Kpoints.automatic_density_by_vol`
- `Kpoints.gamma_automatic`
- `Kpoints.monkhorst_automatic`

输入：

```json
{
  "case_id": "string",
  "structure_ref": "string",
  "mode": "automatic_density|automatic_density_by_vol|gamma|monkhorst",
  "kppa": 1000,
  "grid_density_by_vol": null,
  "force_gamma": false,
  "shift": [0, 0, 0]
}
```

输出：

```json
{
  "kmesh_ref": "/workspace/cases/<case_id>/materials/kmesh.json",
  "mesh": [6, 6, 6],
  "shift": [0, 0, 0],
  "num_kpoints_full": 216,
  "generation_policy": {},
  "warnings": []
}
```

#### `reduce_irreducible_kpoints`

用途：用 spglib 把 uniform k mesh 约化成不可约 k 点及权重。

底层建议：

- `spglib.get_ir_reciprocal_mesh`

输入：

```json
{
  "case_id": "string",
  "structure_ref": "string",
  "mesh": [6, 6, 6],
  "is_shift": [0, 0, 0],
  "symprec": 1e-5
}
```

输出：

```json
{
  "irreducible_kpoints_ref": "string",
  "ir_kpoints_frac": [[...]],
  "weights": [1, 2, 4],
  "mapping": [0, 0, 1],
  "num_ir_kpoints": 28,
  "warnings": []
}
```

#### `generate_seekpath_kpath`

用途：用 SeekPath/HPKOT 生成高对称 band path。

底层建议：

- `seekpath.get_path`

输入：

```json
{
  "case_id": "string",
  "structure_ref": "string",
  "with_time_reversal": true,
  "recipe": "hpkot",
  "threshold": 1e-7,
  "symprec": 1e-5,
  "angle_tolerance": -1.0
}
```

输出：

```json
{
  "kpath_ref": "/workspace/cases/<case_id>/materials/kpath_seekpath.json",
  "point_coords": {"GAMMA": [0, 0, 0], "X": [0.5, 0, 0]},
  "path": [["GAMMA", "X"], ["X", "W"]],
  "has_inversion_symmetry": true,
  "augmented_path": false,
  "is_supercell": false,
  "warnings": []
}
```

#### `generate_pymatgen_highsymm_kpath`

用途：用 pymatgen 生成/交叉检查 high-symmetry path。

底层建议：

- `pymatgen.symmetry.bandstructure.HighSymmKpath`

输入：

```json
{
  "case_id": "string",
  "structure_ref": "string",
  "path_type": "setyawan_curtarolo|hinuma|latimer_munro|all",
  "symprec": 1e-5,
  "angle_tolerance": -1.0
}
```

输出：

```json
{
  "kpath_ref": "/workspace/cases/<case_id>/materials/kpath_pymatgen.json",
  "kpoints": {},
  "path": [],
  "equiv_labels": {},
  "warnings": []
}
```

#### `sample_kpath_segments`

用途：把 high-symmetry path 离散成 line-mode k 点，供 KS-DFT-TAPS 的 k-path axis 或 band postprocess 使用。

输入：

```json
{
  "case_id": "string",
  "kpath_ref": "string",
  "points_per_segment": 40,
  "coordinate_mode": "fractional_reciprocal"
}
```

输出：

```json
{
  "line_kpoints_ref": "string",
  "kpoints": [[0, 0, 0], [0.0125, 0, 0]],
  "labels": ["GAMMA", "", "X"],
  "segment_indices": [[0, 40]],
  "cumulative_distances": [0.0, 0.01],
  "warnings": []
}
```

#### `build_taps_kpoint_axis`

用途：把 uniform kmesh 或 line-mode kpath 转成 TAPS axis descriptor。

输入：

```json
{
  "case_id": "string",
  "kmesh_ref": "string|null",
  "kpath_ref": "string|null",
  "axis_type": "uniform_integration|line_band_path",
  "rank_policy": {"initial_rank": 4, "max_rank": 32}
}
```

输出：

```json
{
  "axis_ref": "/workspace/cases/<case_id>/materials/taps_kpoint_axis.json",
  "axis_name": "kpoint",
  "axis_type": "uniform_integration",
  "points": 28,
  "weights": [],
  "rank_policy": {},
  "warnings": []
}
```

### D. 超胞、缺陷和参数轴

#### `make_supercell_structure`

用途：生成超胞，后续 defect/surface 或参数 sweep 使用。

底层建议：

- `Structure.make_supercell`

输入：

```json
{
  "case_id": "string",
  "structure_ref": "string",
  "scaling_matrix": [[2,0,0],[0,2,0],[0,0,2]]
}
```

输出：

```json
{
  "supercell_ref": "string",
  "num_sites": 64,
  "scaling_matrix": [[2,0,0],[0,2,0],[0,0,2]]
}
```

#### `generate_structure_parameter_axis`

用途：为 TAPS 的结构参数轴生成结构族，例如应变、体积、原子位移。

底层建议：

- pymatgen `Structure.copy`
- lattice scaling / strain operations

输入：

```json
{
  "case_id": "string",
  "structure_ref": "string",
  "parameter_type": "volume_scale|strain|site_displacement|lattice_parameter",
  "values": [0.98, 1.0, 1.02],
  "target_sites": []
}
```

输出：

```json
{
  "parameter_axis_ref": "string",
  "structures": [{"value": 1.0, "structure_ref": "string"}],
  "warnings": []
}
```

#### `map_site_properties`

用途：保存 magmom、selective dynamics、labels 等 site properties，避免标准化后丢失。

输入：

```json
{
  "case_id": "string",
  "source_structure_ref": "string",
  "target_structure_ref": "string",
  "properties": ["magmom", "selective_dynamics", "label"]
}
```

输出：

```json
{
  "mapped_structure_ref": "string",
  "unmapped_sites": [],
  "warnings": []
}
```

### E. KS-DFT-TAPS prompt package

#### `prepare_ks_dft_taps_material_context`

用途：把所有材料工具产物打包成 derivation prompt 可读的 context。

输入：

```json
{
  "case_id": "string",
  "standardized_structure_ref": "string",
  "symmetry_ref": "string",
  "reciprocal_lattice_ref": "string",
  "kmesh_ref": "string",
  "irreducible_kpoints_ref": "string",
  "kpath_ref": "string|null",
  "taps_kpoint_axis_ref": "string|null"
}
```

输出：

```json
{
  "context_ref": "/workspace/cases/<case_id>/materials/ks_dft_material_context.md",
  "json_ref": "/workspace/cases/<case_id>/materials/ks_dft_material_context.json",
  "warnings": []
}
```

该 context 必须写明：

```text
Use these as fixed inputs:
- standardized structure
- reciprocal lattice convention
- symmetry dataset
- irreducible k-points and weights
- high-symmetry labels and segments

Do not recompute in derivation:
- primitive/conventional transform
- space group
- k-path labels
- k-point weights
```

#### `review_ks_dft_material_context`

用途：在进入 TAPS 推导前检查材料 context 是否完整。

输入：

```json
{
  "case_id": "string",
  "context_ref": "string"
}
```

输出：

```json
{
  "ready_for_derivation": true,
  "missing": [],
  "warnings": [],
  "required_user_questions": []
}
```

## KS-DFT-TAPS 数学对象

基础方程：

```text
[-1/2 ∇² + V_eff[n](r)] ψ_i(r) = ε_i ψ_i(r)
n(r) = Σ_i f_i |ψ_i(r)|²
V_eff[n] = V_ext + V_H[n] + V_xc[n]
R_scf[n] = F[V_eff[n]] - n = 0
```

周期体系：

```text
ψ_{i,k}(r) = exp(i k · r) u_{i,k}(r)
H_k[n] u_{i,k} = ε_{i,k} S_k u_{i,k}
n(r) = Σ_k w_k Σ_i f_{i,k} |ψ_{i,k}(r)|²
```

TAPS 表示：

```text
n_TD(r, k, p, s)
V_eff,TD(r, p, s)
Ψ_occ,TD(r, band, k, p)
Γ_TD = f(H_TD, μ)
H_TD[n]
R_scf,TD[n]
```

其中：

- `r` 来自 standardized structure 的 real-space domain。
- `k` 来自 `taps_kpoint_axis.json`。
- `p` 来自 structure parameter axis。
- `s` 是 SCF continuation / mixing history axis。

## 核心算法路线

### 1. Tensor-Structured KS Basis

构造 Hamiltonian-adapted tensor basis：

```text
H[n] ≈ H_x ⊕ H_y ⊕ H_z + low-rank correction
1D localized functions -> Tucker / CP / C-HiDeNN-TD tensor basis
```

材料工具层提供：

- standardized cell。
- reciprocal lattice。
- symmetry dataset。
- k-point axis。

TAPS 层推导：

- mass/overlap matrix。
- kinetic/stiffness matrix。
- potential matrices。
- Hamiltonian separability。
- rank refinement policy。

### 2. Chebyshev-Filtered Occupied Subspace Update

KS-DFT-TAPS 第一版内核不做完整对角化主流程，而做：

```text
given H[n_j], Ψ_occ,j
-> estimate spectral bounds
-> apply Chebyshev filter to Ψ_occ,j
-> S-orthonormalize
-> Rayleigh-Ritz subspace projection when needed
-> reconstruct n_{j+1}
```

### 3. Low-Rank SCF Preconditioning

SCF route：

```text
R[n] = F[n] - n
J ≈ Σ_l g_l ⊗ dR[g_l]
P_LRDM ≈ ε^{-1}_low_rank
n_{j+1} = n_j + P_LRDM * mixed_update(R[n_j])
```

先实现 Anderson/Pulay/Kerker baseline，再加 LRDM。所有 preconditioner 信息都要进入 runtime metadata。

## Context Window 规则

KS-DFT-TAPS 的 context window 必须包含：

```text
TAPS template derivation
TAPS matrix definitions
TAPS CoT outline
KS-DFT formula notes
CheFSI notes
Tucker tensor KS notes
LRDM SCF notes
ks_dft_material_context.md
ks_dft_problem.json
verification policy
```

derivation prompt 必须包含硬规则：

```text
Before deriving, read ks_dft_material_context.md.
Use standardized_structure.json as the only structure source.
Use kmesh.json / irreducible_kpoints.json for Brillouin-zone integration.
Use kpath_seekpath.json only for line-mode band path after SCF verification.
Do not invent or recompute space group, k-point labels, k-point weights, reciprocal lattice convention.
If required material artifacts are missing, stop and request materials-preprocess-agent.
```

## 目录结构

```text
/cases/<case_id>/
  problem/
    problem_statement.md
    ks_dft_problem.json
    ks_dft_open_questions.md
  materials/
    source_structure.json
    structure_standardized.json
    structure_primitive.json
    structure_conventional.json
    symmetry_dataset.json
    reciprocal_lattice.json
    kmesh.json
    irreducible_kpoints.json
    kpath_seekpath.json
    kpath_pymatgen.json
    taps_kpoint_axis.json
    ks_dft_material_context.md
    ks_dft_material_context.json
    materials_preprocess_report.md
  references/
    taps_template_eq5.md
    taps_matrix_definitions.md
    taps_cot_outline.md
    taps_verification_workflow.md
    ks_dft_formula_notes.md
    ks_tensor_basis_notes.md
    chefsi_notes.md
    lrdm_scf_notes.md
  context/
    context_window.md
    context_window.json
  taps/
    ks_dft_derivation_prompt.md
    ks_dft_derivation.md
    ks_dft_implementation_notes.md
    kernel.py
    ks_dft_execution_plan.json
    ks_dft_runtime_metadata.json
    ks_dft_solution_summary.json
  verification/
    ks_dft_verification_plan.md
    convergence_study.py
    report.md
    report.json
  report/
    report.md
    figures/
```

## Schema 建议

### KSDftTapsProblemSpec

```text
route: ks_dft_taps
system_type
structure_ref
standardized_structure_ref
symmetry_ref
reciprocal_lattice_ref
kmesh_ref
irreducible_kpoints_ref
kpath_ref
electron_count
spin_mode
xc_functional
pseudopotential_spec
smearing_spec
hamiltonian_terms
taps_basis_policy
subspace_update_policy
scf_policy
verification_policy
```

### KSDftTapsAxisSpec

```text
name
kind: space | reciprocal | band_subspace | parameter | spin | scf
source_artifact_ref
domain
points
weights
rank
units
separability_assumption
refinement_policy
```

### KSDftTapsResultSpec

```text
energy_total
energy_terms
density_ref
hartree_potential_ref
occupied_subspace_ref
density_matrix_ref
fermi_level
band_gap_optional
charge_error
orthonormality_error
scf_residual
poisson_residual
rank_history
grid_history
kpoint_history
materials_artifacts_used
warnings
```

### MaterialsPreprocessResultSpec

```text
source_structure_ref
standardized_structure_ref
primitive_structure_ref
conventional_structure_ref
symmetry_ref
reciprocal_lattice_ref
kmesh_ref
irreducible_kpoints_ref
kpath_seekpath_ref
kpath_pymatgen_ref
taps_kpoint_axis_ref
symprec
angle_tolerance
species_order
transformations
warnings
```

## 路线图

### Phase 0：材料工具层和 prompt contract

目标：先把稳定晶体学操作从 prompt 中拿出来。

- 实现并暴露上述 materials tools。
- 新增 `materials-preprocess-agent`。
- 新增 `ks_dft_material_context.md` 生成器。
- 更新 KS-DFT-TAPS prompt，明确工具产物必须使用。

验收：

- 输入 CIF/POSCAR 后生成 standardized structure、symmetry、kmesh、irreducible kpoints、kpath。
- derivation prompt 明确引用这些 artifact。
- prompt 不再手写空间群和 k-path。

### Phase 1：KS-DFT-TAPS 知识和推导资产

- 新增 `ks_dft_formula_notes.md`。
- 新增 `ks_tensor_basis_notes.md`。
- 新增 `chefsi_notes.md`。
- 新增 `lrdm_scf_notes.md`。
- 更新 context-window builder。

验收：

- 给 Si 或 toy periodic potential，agent 能写出 `H C = S C ε`、`n(r)`、`R_scf`、`C_occ`、kpoint axis、rank/grid/k-point verification。

### Phase 2：Toy KS-DFT-TAPS Kernel

- 1D/2D toy Kohn-Sham potential。
- real-space grid。
- Hamiltonian assembly。
- occupied subspace initialization。
- Chebyshev filter update。
- density reconstruction。
- simple Hartree/XC placeholder with explicit assumptions。
- Anderson/Pulay mixing。

验收：

- charge error 可控。
- orthonormality error 可控。
- SCF residual 下降。
- rank/grid refinement 有趋势。

### Phase 3：3D Gamma-Only Periodic Local-Pseudopotential Solver

- 读取 materials artifacts。
- 使用 standardized structure。
- Gamma-only。
- 内置 local Gaussian pseudopotential，作为可运行的本地赝势求解器入口。
- 3D tensor grid。
- matrix-free Hamiltonian action：`Hpsi = -0.5*periodic_laplacian(psi) + V_eff[n]*psi`。
- CheFSI-style occupied/near-Fermi subspace solve。
- neutral-background FFT Hartree Poisson solve。
- LDA exchange-only potential/energy。
- fractional occupation 处理 Gamma 点近简并壳层。
- adaptive damped density mixing。
- Hamiltonian、SCF、Poisson、energy terms、provenance report。

限制：

- 当前已能索引本地 VASP PAW PBE `POTCAR` metadata，用于价电子数、ENMAX 建议和 provenance。
- 还没有把 PAW augmentation / nonlocal projectors 真正接入 Hamiltonian。
- 还没有多 k 点 BZ integration。
- 还没有 LDA correlation / GGA / meta-GGA。

验收：

- 能在简单半导体/绝缘体 local-pseudopotential case 跑通真实 SCF。
- charge、orthonormality、SCF residual、Poisson residual 必须过验证工具。
- energy、density、Hamiltonian evidence、SCF residual history、materials artifacts used 都有报告。

### Phase 4：k-Point Axis 和 Band/DOS 前置能力

- 使用 `irreducible_kpoints.json` 做 BZ integration。
- 使用 `taps_kpoint_axis.json` 做 reciprocal TAPS axis。
- 使用 `kpath_seekpath.json` 做 line-mode band path。
- band/DOS 只在 SCF verified 后开放。

验收：

- k-point refinement 报告。
- Gamma-only 与 k-point 结果差异可解释。
- band/DOS 输出标注 kpath 来源和误差警告。

### Phase 5：LRDM SCF Acceleration

- 实现 Kerker / Anderson / Pulay baseline。
- 实现 LRDM direction functions。
- 计算 Gateaux derivative of SCF residual。
- 构造 low-rank dielectric preconditioner。
- 记录 preconditioner rank 和 residual reduction。

验收：

- 在金属/半导体/异质 toy cases 上比较迭代次数。
- LRDM 失败模式进入报告。

### Phase 6：工程化材料任务

- relaxation support with force verification。
- DOS/band workflow。
- defect/surface selected-slice TAPS parameterization。
- spin-polarized KS route。
- SOC/U/vdW 作为显式高级假设。

验收：

- 每个任务都有 materials artifact provenance、rank/grid/k-point verification。
- 不满足验证时标记为 research prototype，不给生产结论。

## 代码改造点

架构纠偏：DFT 优化部分必须和 PhysicsOS 主体保持一致，即 **LLM-driven case-local implementation**。确定性工具只产出 materials/pseudopotential/reference-kernel/verification artifacts；数值策略、参数选择和 `kernel.py` 由 `ks-dft-taps-implementation-agent` 基于 derivation/context 自己生成，并由 KS-DFT verification tools 验收。`prepare_toy_ks_dft_taps_kernel` 与 `prepare_gamma_only_ks_dft_taps_kernel` 不应是隐藏默认求解器；它们的原型代码应作为每个 DFT case 中 LLM 可查看、编辑、复制、替换的 reference source/知识 artifact 进入 `taps/reference_kernels/`，最终执行仍以 case-local `taps/kernel.py` 为准。

1. `physicsos/tools/materials_tools.py`
   - 放所有 pymatgen/seekpath/spglib wrapper。
   - 每个工具写 artifact，不只返回内存对象。

2. `physicsos/tools/registry.py`
   - 新增 `MATERIALS_TOOLS`。
   - 新增 `KS_DFT_TAPS_TOOLS`。
   - 新增 `KS_DFT_VERIFICATION_TOOLS`。
   - 对 `materials-preprocess-agent` 暴露完整 materials tools。
   - 对 derivation/implementation agent 只暴露 review/read context 类工具，避免它们重新标准化结构。

3. `physicsos/agents/prompts.py`
   - 新增 `materials-preprocess-agent`。
   - 新增 `ks-dft-analysis-agent`、`ks-dft-taps-derivation-agent`、`ks-dft-taps-implementation-agent`、`ks-dft-verification-agent`。
   - prompt 中明确必须使用 materials artifacts。

4. `physicsos/schemas/materials.py`
   - 新增 `CrystalStructureRef`、`SymmetryDatasetRef`、`KPointMeshSpec`、`KPathSpec`、`MaterialsPreprocessResultSpec`。

5. `physicsos/schemas/ks_dft_taps.py`
   - 新增 KS-DFT-TAPS problem、axis、result、verification schema。

6. `docs/knowledge_seed/references/`
   - 新增 KS-DFT-TAPS notes。
   - 新增 `materials_tool_contract.md`。

7. `physicsos/tools/ks_dft_verification_tools.py`
   - 新增 charge、orthonormality、SCF residual、Poisson residual、rank/grid/k-point convergence。
   - 新增 check：kernel 是否读取了 expected materials artifacts。

8. `physicsos/tools/ks_dft_taps_tools.py`
   - 新增 Phase 1 toy KS-DFT-TAPS kernel scaffold 生成工具。
   - 生成 `ks_dft_derivation.md`、`ks_dft_implementation_notes.md`、`ks_dft_execution_plan.json`、`kernel.py`。
   - kernel 写出 KS 专用 artifacts：density、weights、coefficients、overlap、Poisson residual、SCF history、rank/grid/k-point convergence、materials artifact provenance。
   - 新增 Phase 2 3D Gamma-only periodic local-pseudopotential solver 生成工具。
   - Gamma-only kernel 强制读取 `ks_dft_material_context.json` 和 `standardized_structure_ref`，写出 3D density、weights、Hamiltonian report、SCF report、Gamma-only runtime metadata。
   - 新增 Phase 3 band/DOS preflight gate。
   - preflight 必须读取 KS verification artifacts，确认 charge、orthonormality、SCF、Poisson、rank/grid/k-point convergence、material provenance 通过后，才生成 band/DOS 计划。
   - 新增 Phase 5 LRDM SCF acceleration plan/report 工具。
   - LRDM plan 读取 SCF residual history，输出 Kerker/Anderson/Pulay/LRDM 建议、低秩方向数、failure modes 和 runtime metadata provenance。

## 不做的事

- 当前阶段不接入 QE/VASP/CP2K/ELSI。
- 不保留 OF-DFT-TAPS 路线。
- 不让 prompt 手写空间群、标准晶胞、k-path、不可约 k 点。
- 不把 pymatgen/seekpath/spglib 的结果当作 DFT 解；它们只是稳定前处理。
- 不在没有 charge、orthonormality、SCF、rank/grid/k-point 验证时宣称 DFT 精度。
- 不默认选择 U、SOC、vdW、磁序、赝势或 XC functional。

## 最终路线

```text
Phase 0  materials tools + prompt contract
Phase 1  KS-DFT-TAPS knowledge and derivation assets
Phase 2  toy KS-DFT-TAPS kernel
Phase 3  3D Gamma-only periodic local-pseudopotential solver
Phase 4  k-point axis + verified band/DOS
Phase 5  LRDM SCF acceleration
Phase 6  relaxation, defects, surfaces, spin/SOC/U/vdW
```

最终形态是：

```text
pymatgen/seekpath/spglib 保证材料前处理正确性；
KS-DFT-TAPS 保证方程驱动的低秩求解；
PhysicsOS prompt 保证两者边界清晰、artifact 可追溯、验证闭环完整。
```

## 实施任务清单

任务语义约束：
- `[done]` 不表示把某个数值规则永久硬编码进默认 kernel；它表示当前 LLM-driven 架构已经有对应的工具产物、prompt contract、review spec 或 verification gate，能让 implementation agent 在 case-local 代码中实现并被验收。
- `[todo]` 不应被实现为新的默认硬编码规则；它应补齐 artifact contract、上下文注入、LLM 选择空间、case-local review 要求和 verification gate。
- 如果某个物理能力需要真实数据（validated local potential、projector、PBE/spin、SOC/U/vdW），架构默认应 fail closed 或要求 agent 明确写 prototype assumption，而不是偷偷替换成内置数值规则。

[done] 明确 DFT 唯一路线为 KS-DFT-TAPS，不接入 QE/VASP/CP2K/ELSI，不保留 OF-DFT 并行路线。

[done] 明确 pymatgen/seekpath/spglib 应作为 agent 可调用 structured tools，而不是只写在 prompt 里。

[done] 设计 `materials-preprocess-agent -> ks-dft-analysis-agent -> ks-dft-taps-derivation-agent -> ks-dft-taps-implementation-agent -> ks-dft-verification-agent` 的专门 DFT 接入路径。

[done] 在文档中列出第一批 materials tools 的接口、输入输出和 artifact 契约。

[done] 将 `pymatgen`、`spglib`、`seekpath` 加入基础 pip install 依赖，使用户安装 PhysicsOS 时直接内置材料工具栈。

[done] 检查当前仓库的 schema/tool/registry/prompt 结构，确定 materials tools 和 KS-DFT-TAPS schema 的最小落点。

[done] 新增 `physicsos/tools/materials_tools.py`，实现结构读取、写出、校验、空间群分析、结构标准化、结构比较、lattice reduction。

[done] 在 `materials_tools.py` 中实现倒空间和 k 点工具：reciprocal lattice、uniform kmesh、irreducible kpoints、seekpath kpath、pymatgen high-symmetry kpath、kpath sampling、TAPS kpoint axis。

[done] 在 `materials_tools.py` 中实现结构参数工具：supercell、structure parameter axis、site properties mapping。

[done] 在 `materials_tools.py` 中实现 `prepare_ks_dft_taps_material_context` 和 `review_ks_dft_material_context`。

[done] 新增或扩展 `physicsos/schemas/materials.py`，定义 `CrystalStructureRef`、`SymmetryDatasetRef`、`KPointMeshSpec`、`KPathSpec`、`MaterialsPreprocessResultSpec`。

[done] 新增 `physicsos/schemas/ks_dft_taps.py`，定义 KS-DFT-TAPS problem、axis、result、verification schema。

[done] 在 `physicsos/tools/registry.py` 注册 `MATERIALS_TOOLS`，并按 agent 角色控制工具暴露面。

[done] 更新 `physicsos/agents/prompts.py`，新增 materials/KS-DFT-TAPS 相关 agent prompt，并写入强制使用 materials artifacts 的规则。

[done] 新增 `docs/knowledge_seed/references/materials_tool_contract.md`。

[done] 新增 `docs/knowledge_seed/references/ks_dft_formula_notes.md`。

[done] 新增 `docs/knowledge_seed/references/ks_tensor_basis_notes.md`。

[done] 新增 `docs/knowledge_seed/references/chefsi_notes.md`。

[done] 新增 `docs/knowledge_seed/references/lrdm_scf_notes.md`。

[done] 更新 context-window 构建逻辑，让 KS-DFT case 自动纳入 materials context 和 KS-DFT-TAPS references。

[done] 新增 `physicsos/tools/ks_dft_verification_tools.py`，实现 charge conservation、orthonormality、SCF residual、Poisson residual、rank/grid/k-point convergence verification helpers。

[done] 新增 verification check：case-local kernel 是否读取 expected materials artifacts，并验证这些 artifacts 是否存在。

[done] 新增 `ks-dft-verification-agent` prompt、subagent 注册和工具暴露面，避免 KS-DFT 验证误走通用 Fig.7 PDE manufactured-solution 流程。

[done] 编写 materials tools 的单元测试，覆盖缺依赖错误、material context review、Si diamond CIF parsing、标准化、space group、kmesh、irreducible kpoints、seekpath path 和 KS-DFT context window 注入。

[done] 编写 KS-DFT-TAPS prompt/context 测试，确认 context window 引用 materials artifacts 和 KS-DFT-TAPS references。

[done] 编写 KS-DFT verification tools 测试，覆盖内联数组、artifact runtime metadata、工具注册表和 `ks-dft-verification-agent` 暴露面。

[done] 实现 Phase 1 toy KS-DFT-TAPS kernel 的最小推导和 case-local scaffold：新增 `prepare_toy_ks_dft_taps_kernel`，可生成、静态检查、执行，并输出 KS 验证工具可读取的 artifacts。

[done] 编写 Phase 1 toy KS-DFT-TAPS kernel 测试，覆盖 kernel 生成、`execute_taps_kernel` 执行、charge/orthonormality/SCF/Poisson/convergence/material provenance 全套验证。

[done] 实现 Phase 2 3D Gamma-only periodic local-pseudopotential solver：新增 `prepare_gamma_only_ks_dft_taps_kernel`，读取 materials context/standardized structure，构造 3D periodic tensor grid，使用内置 local Gaussian pseudopotential 跑真实 Gamma-only KS SCF，并显式记录其不是 validated element pseudopotential library。

[done] 编写 Phase 2 3D Gamma-only periodic prototype 测试，覆盖 material artifact 读取、kernel 执行、Hamiltonian report、Gamma-only metadata、charge/orthonormality/SCF/Poisson/convergence/material provenance 验证。

[done] 实现 Phase 3 k-point axis 和 verified band/DOS 前置能力：新增 `prepare_verified_ks_dft_band_dos_preflight`，读取 KS verification artifacts、materials context、kmesh、irreducible kpoints、kpath/line_kpoints provenance，只有 SCF 验证闭环通过后才写出 band/DOS plan。

[done] 编写 Phase 3 band/DOS preflight 测试，覆盖验证通过后生成 band/DOS plan，以及缺少 KS verification checks 时拒绝生成 plan。

[done] 实现 Phase 5 LRDM SCF acceleration 的策略/报告层：新增 `plan_lrdm_scf_acceleration`，从 `ks_dft_runtime_metadata.json` 或显式 residual history 生成 Kerker/Anderson/Pulay/LRDM 推荐、LRDM rank、direction functions 和 failure modes。

[done] 将 Gamma-only kernel 从 artifact scaffold 推进为 Hamiltonian evidence solver：新增周期 finite-difference Laplacian action、local Gaussian ionic potential、Rayleigh quotient/eigen residual、energy_terms，并写入 `ks_dft_hamiltonian_report.json`。

[done] 将 Gamma-only Hamiltonian evidence/report 推进为参与 occupied-subspace solve：生成的 kernel 使用 Hamiltonian action 求 occupied state 后重构 density。

[done] 将 dense Gamma Rayleigh-Ritz 替换为 matrix-free Gamma imaginary-time 迭代 eigensolver：不装配 dense Hamiltonian，只调用 Hamiltonian action，并记录 eigensolver history。

[done] 将 matrix-free imaginary-time 升级为 CheFSI 风格 eigensolver：spectral bounds、Chebyshev-like filter、S/weight orthonormalization、projected Rayleigh-Ritz solve，并记录 CheFSI metadata。

[done] 强化 CheFSI 第一轮：支持 occupied + near-Fermi 多态求解、SCF 步间 subspace reuse、自动补足 recycled subspace rank，并记录 projected Rayleigh-Ritz metadata。

[done] 强化 CheFSI 第二轮：Gamma-only eigensolver 已从 `alpha*I-H` 的 Chebyshev-like filter 改为 low-energy window-scaled Chebyshev recurrence，将 unwanted high-energy interval 映射到 `[-1, 1]` 并放大 cutoff 以下子空间；Hamiltonian report 记录 filter degree、filter window、locked states、restart count、stagnation count 和 convergence policy。

[done] 强化 CheFSI 第三轮：加入更可靠的谱界估计策略、用户可配置 filter degree/锁定阈值、失败 case 自动降阶/升阶诊断，以及真实 block residual locking 而不是只记录 locked-state metadata。

[done] 为 Gamma-only kernel 增加 Hartree Poisson solve：使用与周期有限差分 Laplacian 一致的离散 Fourier 符号求解 neutral-background Hartree potential，并输出 `ks_dft_hartree_potential.json` 与离散 Poisson residual。

[done] 为 Gamma-only kernel 增加明确 XC 策略：已实现 LDA exchange potential/energy，并把 XC policy、energy terms 和 provenance 写入 runtime metadata / Hamiltonian report。

[done] 将 Phase 5 的最基础 mixing 接入实际 kernel：Gamma-only kernel 已经使用 adaptive damped linear density mixing，并修复 density history 更新时序，SCF residual 可通过验证工具。

[done] 为 Gamma-only kernel 增加 fractional occupation：对 Gamma 点近简并 shell 做分数占据，避免简并子空间任意旋转导致 density/SCF residual 振荡。

[done] 新增 pseudopotential library 工具层：`index_vasp_paw_pbe_library` 扫描本地 VASP PAW PBE `POTCAR` 库，提取 `TITEL/ZVAL/ENMAX/LEXCH/RCORE/SHA256/path` metadata，不复制 POTCAR 内容。

[done] 新增 `select_pseudopotentials_for_structure`：按 standardized structure 的元素选择本地 PBE 赝势，写出 `/cases/<case_id>/pseudopotentials/ks_dft_pseudopotential_context.json/md`，记录 species count、总价电子数、推荐 ENMAX、variant policy 和 provenance。

[done] 将 pseudopotential context 接入 Gamma-only kernel：当 `ks_dft_pseudopotential_context.json` 存在且用户未显式覆盖 electron count 时，用 `total_valence_electrons` 作为电子数，并把 pseudopotential context 摘要写入 Hamiltonian report/runtime metadata。

[done] 将 pseudopotential library 路径接入 PhysicsOS config：`~/.physicsos/config.json` 新增 `pseudopotentials.default_library_id` 与 `pseudopotentials.libraries.<id>.root`，工具在未显式传 `library_root` 时可自动读取；`PHYSICSOS_PSEUDOPOTENTIAL_DIR` 可作为环境变量覆盖。

[done] 新增 pseudopotential CLI 入口：`physicsos pseudopotentials config` 查看配置和环境变量覆盖，`physicsos pseudopotentials set-root <path>` 写入本机伪势库路径，`physicsos pseudopotentials index` 生成 metadata-only POTCAR 索引，`physicsos pseudopotentials select` 为结构生成 KS-DFT pseudopotential context。

[done] 将 Anderson/Pulay/Kerker 从 plan/report 层接入实际 KS-DFT-TAPS kernel：Gamma-only kernel 已实现 reciprocal-space Kerker residual filter、Pulay/DIIS 候选、Anderson secant 候选、候选更新接受/拒绝 guard、adaptive linear fallback，并把 mixing history 写入 runtime metadata 与 Hamiltonian report。

[done] 强化 SCF mixing 生产诊断：Gamma-only kernel 已为 Pulay/Anderson/Kerker 候选增加 residual map 复评估、density-only energy proxy 接受准则、连续拒绝后的 history restart、基于 Gamma 占据/能隙的 Kerker q0 自动选择，并在 runtime metadata / Hamiltonian report 中写入 accepted_method_counts、rejection_counts、restart_events、failure_modes 和 candidate_acceptance_policy。

[done] 强化 SCF mixing 下一轮：把候选 residual 复评估从当前密度映射 proxy 升级为轻量 one-step KS map 复评估；加入真正的 total-energy line search / trust-region mixing；把 metal/insulator 分类改为来自 kmesh DOS 或用户材料上下文，而不是只用 Gamma gap。

[done] 将 LRDM 从 plan/report 层接入实际 KS-DFT-TAPS kernel：实现 density perturbation directions、SCF residual Gateaux derivative、low-rank dielectric preconditioner 和 rank truncation policy。

[done] 将 Phase 3 band/DOS 从 preflight gate 升级为真实计算：Gamma eigenvalue output、line-mode kpath band energies、irreducible-kmesh DOS/Fermi level/band gap，并保留 provenance 和误差警告。

[done] 将 k-point axis 接入实际求解：多 k 点 Hamiltonian、k weights 积分、Gamma-only vs kmesh convergence 报告。

[done] 架构纠偏：新增 `compile_ks_dft_taps_kernel`，让 KS-DFT implementation-agent 默认走 LLM-driven prompt package / scaffold / review spec / verification loop；`prepare_toy_ks_dft_taps_kernel` 和 `prepare_gamma_only_ks_dft_taps_kernel` 降级为 prototype/fixture/reference，不再作为 implementation-agent 默认暴露工具。

[done] 将原型 kernel 从“隐藏 fixture”改造成每个 DFT case 的可编辑 reference source：`compile_ks_dft_taps_kernel` 会写出 `taps/reference_kernels/gamma_only_reference_kernel.py`、`gamma_only_reference_numerical_policy.json`、`reference_kernel_manifest.json` 和 README。LLM implementation agent 必须把这些当作可阅读、可编辑、可复制、可替换的数值实现参考，最终仍需写入/修改 `taps/kernel.py`，并在 runtime metadata 中记录 `adapted_from_reference_kernel`、最终 `numerical_policy` 和 prototype assumptions。

[done] 将 Gamma-only prototype kernel 的数值选择外置为 LLM-editable case artifact：`prepare_gamma_only_ks_dft_taps_kernel` 写出 `taps/ks_dft_numerical_policy.json`，runtime 从该 policy 读取 grid/SCF/CheFSI/mixing/pseudopotential/XC 策略，并在 runtime metadata 与 Hamiltonian report 记录 `numerical_policy_ref`、`strategy_family` 和实际 policy；测试覆盖执行前编辑 policy 后 kernel 采用新参数，证明 prototype fixture 不再把这些选择藏成不可追踪常量。

[done] 增强 KS 验证第一轮：新增 Hamiltonian evidence verification，检查 `ks_dft_hamiltonian_report.json` 与 runtime metadata 中的 matrix-free Hamiltonian action、CheFSI/eigensolver history、eigen residual 阈值、energy term 总和一致性、XC policy、pseudopotential policy/context provenance，并将 `hamiltonian_evidence.json` 纳入 band/DOS preflight gate。

[done] 增强 KS 验证第二轮之一：`check_ks_poisson_residual` 支持从 `ks_dft_hartree_potential.json`、`ks_dft_density.json`、quadrature weights/cell volume 和 grid shape 重新施加周期离散 Laplacian，复算 neutral-background Poisson residual，而不是只信任 kernel 写出的 residual artifact。

[done] 增强 KS 验证第二轮之二：Gamma-only kernel 输出 `ks_dft_effective_potential.json`；`check_ks_hamiltonian_evidence` 支持读取 coefficients/eigenvalues、effective potential、weights/cell volume 和 grid shape，重新施加 `Hpsi = -0.5*Laplacian(psi) + V_eff psi`，复算每个态的 Hamiltonian eigen residual。

[done] 增强 KS 验证第二轮之三：band/DOS plan 写入 provenance；新增 `check_ks_band_dos_provenance`，检查 preflight 已接受、所有 required KS checks artifact 存在且通过 lineage、band plan 引用 line kpoints/kpath、DOS plan 引用 kmesh/irreducible kpoints，并验证这些来源文件存在。

[done] 增强 KS 验证第二轮之四：`check_ks_hamiltonian_evidence` 支持能量变分一致性检查，读取 eigenvalues/occupations、density、effective potential、Hartree potential、weights 和 Hamiltonian energy terms，验证 KS band-energy double-counting 关系 `E_total = sum f_i eps_i - int n V_eff + E_local + E_H + E_xc`。

[done] 建立 validated local pseudopotential artifact contract：新增 `validate_local_pseudopotential_artifact`，验证每个必需元素的径向 local potential、单位、严格递增径向网格、插值策略、version hash、provenance，并输出 `pseudopotentials/ks_dft_local_pseudopotential_contract.json/md`；缺元素或字段时 fail closed。materials/analysis/implementation prompts 与 agent tool scope 已接入该 contract，kernel 只有在 contract accepted 时才能把 local-potential artifact 当作 validated Hamiltonian 数据，否则必须失败或显式记录 prototype assumption。

[done] 建立 PAW/nonlocal projector artifact contract：新增 `validate_nonlocal_projector_artifact`，验证 norm-conserving Kleinman-Bylander projector 或 PAW projector/augmentation artifact 的元素覆盖、径向 projector、角动量通道、系数、quadrature、version hash、provenance；PAW representation 要求 augmentation charge moments、partial waves、compensation-charge policy。输出 `pseudopotentials/ks_dft_projector_context.json/md`，缺数据时 fail closed；prompts/tool scope 要求 LLM 只有在 contract accepted 时才能在 case-local Hamiltonian action 中加入 projector terms，否则必须禁用或显式记录 assumption。

[done] 增加真实 XC 层第一轮：Gamma-only kernel 已接入 Perdew-Zunger 1981 unpolarized LDA correlation，`xc_policy = lda_x_pz81_correlation`；Hamiltonian report 写出 `lda_exchange`、`lda_correlation_pz81`、`xc_total`，能量变分一致性检查继续通过。

[done] 建立真实 XC 层第二轮 contract：新增 `prepare_ks_dft_xc_policy`，为 PBE GGA、LSDA、spin-PBE 等生成 `taps/xc_policy.json/md`，明确 density inputs、GGA gradient/boundary policy、energy density/potential outputs、runtime metadata keys、Hamiltonian report keys 和 energy/potential consistency check 要求。Gamma reference kernel 只声明 nonmagnetic LDA 支持；PBE/spin 由 LLM 在 case-local kernel 中实现或 fail closed，不硬编码为默认数值规则。

[done] 建立 Phase 6 工程化材料任务 assumption manifest：新增 `prepare_ks_dft_task_assumptions`，写出 `problem/ks_dft_task_assumptions.json/md`，对 relaxation、DOS/band、defect/surface、spin、SOC、DFT+U、vdW 做显式 enabled/disabled/unspecified gate；高级物理或 defect/surface model 未指定时 blocked，implementation agent 不得静默推断默认磁序、SOC、U、vdW 或结构松弛策略。

[done] 建立 LLM-driven molecular-DFT 子路线的第一层工具/契约：新增 `parse_molecular_structure`（XYZ/JSON 内置支持，SDF/MOL2/PDB fail closed 并要求 case-local parser/转换）、`prepare_ks_dft_molecular_context`（charge、multiplicity、molecule/cluster system type、open-boundary/vacuum-box policy gate）、`prepare_molecular_taps_scaling_policy`（localized orbital / density-matrix truncation / fragment partition / near-field far-field Coulomb / atom-centered or adaptive grid axes / hierarchical TAPS axes 的 LLM-selectable strategy contracts）。这些工具已经接入 registry、materials/analysis/derivation/implementation/verification prompts、context window 和 DeepAgents subagent tool scope；`compile_ks_dft_taps_kernel` 现在会在每个 DFT case 的 `taps/reference_kernels/` 下生成可编辑 `molecular_reference_kernel.py` 与 `molecular_reference_policy.json`，作为 LLM 可查看、可编辑、可替换的分子路线 scaffold，而不是固定 solver。

[done] 推进 molecular-DFT 专用 verification 第一层：新增 `check_ks_molecular_context_evidence`，读取 `materials/ks_dft_molecular_context.json`、`taps/molecular_taps_scaling_policy.json`、`taps/ks_dft_runtime_metadata.json` 以及可选 boundary/fragment/locality evidence artifacts，检查 charge/multiplicity provenance、open-boundary 或 vacuum-box Poisson policy 是否最终化、LLM 选择的 molecular scaling strategy 是否来自 case policy、fragment/locality 策略是否有通过的 evidence，并拒绝 molecule/cluster case 在没有 vacuum-box policy 时偷偷使用 crystal kmesh/kpath artifacts。该工具已接入 KS-DFT verification registry、`ks-dft-verification-agent` 工具面、implementation manifest required checks 和 prompt。

[done] 增强 molecular-DFT verification 数值证据解析层：`check_ks_molecular_context_evidence` 现在不只检查 provenance 和 `passes: true`，还会解析并验证 boundary residual、vacuum-box correction energy consistency、fragment charge integration（逐 fragment 与总 charge error）、locality/truncation sweep delta，以及 large/very_large route 的 scaling evidence（parallel efficiency 或经验 scaling exponent）。测试覆盖证据齐全通过、缺 evidence fail closed、以及坏数值证据被拒绝。该层仍只验证 LLM 选择的 case-local kernel 输出 artifacts，不引入固定分子 DFT solver。

[done] 增强 molecular-DFT verification 的直接复算能力第一轮：`check_ks_molecular_context_evidence` 支持在 boundary evidence 中读取 `direct_coulomb_check`，从 source/probe points、density values、quadrature weights 和 Hartree potential 直接复算 Coulomb residual；同时支持 `multipole_check`，用 monopole far-field `q/r` 复算远场 potential residual。bad evidence 会触发 `direct_coulomb_residual_above_tolerance` 或 `multipole_far_field_residual_above_tolerance`。这仍是对 LLM kernel 输出 artifacts 的验证，不规定 kernel 必须采用 direct Coulomb 或 multipole solver。

[done] 增强 molecular-DFT verification 的 3D 网格直接复算能力第一轮：`check_ks_molecular_context_evidence` 支持 boundary evidence 中的 `grid_poisson_check`，从 density、Hartree potential、grid shape、grid spacing/cell lengths 复算非周期二阶差分 Poisson residual，并检查可选 boundary samples 的 Dirichlet potential error。bad evidence 会触发 `grid_poisson_residual_above_tolerance` 或 `grid_boundary_residual_above_tolerance`。该 gate 仍读取 LLM kernel 输出 artifacts，不指定 kernel 内部 Poisson solver。

[done] 增强 molecular-DFT verification 的 cutoff/vacuum correction gate：`check_ks_molecular_context_evidence` 支持 `coulomb_cutoff_check`，从 source/probe points、density values、quadrature weights、cutoff radius 和 Hartree potential 复算截断 Coulomb residual；支持 `vacuum_finite_size_correction`，复核 `raw_energy + correction_energy = corrected_energy`、correction terms 求和一致性，以及 padding/cell-size sweep 中 correction magnitude 是否随 padding 增大而下降。bad evidence 会触发 `coulomb_cutoff_residual_above_tolerance`、`vacuum_finite_size_correction_inconsistent`、`vacuum_finite_size_terms_inconsistent` 或 `vacuum_finite_size_correction_not_decreasing_with_padding`。该层仍只验证 case-local artifacts，不把某个 correction 公式硬编码为默认求解路径。

[done] 增强 molecular-DFT correction formula artifact：`check_ks_molecular_context_evidence` 支持 boundary evidence 中的 `correction_formula_manifest`，由 LLM 在 case-local artifact 里声明 `formula_id`、受限算术 `expression`、`variables`、`reported_value`、`applicability` 和 `provenance`；verification 工具用安全表达式求值复核 reported value、一致性和适用条件/provenance。该机制允许 Makov-Payne、Martyna-Tuckerman、truncated Coulomb 或自定义 multipole correction 作为 LLM 选择的 case-local 公式出现，但不把任何一个公式设为默认硬编码路线。

[done] 推进 molecular-DFT correction formula manifest 的 runtime provenance 闭环：`check_ks_molecular_context_evidence` 现在要求 `taps/ks_dft_runtime_metadata.json` 在使用 correction formula 时记录 `correction_formula_manifest`，包含 `formula_id`、`sha256/hash` 和 `selected_policy/boundary_policy`；verification 会检查 runtime metadata、boundary evidence 中的 formula manifest、applicability policy 三者是否一致，并在 mismatch 时触发 `runtime_formula_id_mismatch`、`runtime_formula_hash_mismatch` 或 `runtime_formula_policy_mismatch`。implementation prompt 也要求分子 correction formula 写入 runtime metadata 和 `verification/ks_dft/molecular_boundary_evidence.json`，让该 gate 反向约束 LLM 生成 kernel 的输出，而不只是事后检查。

[done] 回到 LLM-driven molecular implementation 侧：增强 `molecular_reference_kernel.py` 和 implementation prompt。reference kernel 现在提供 `write_molecular_output_templates`，会写出 `taps/ks_dft_runtime_metadata.json`、`verification/ks_dft/molecular_boundary_evidence.json`、`fragment_charge_consistency.json`、`molecular_locality_sensitivity.json`、`molecular_scaling_evidence.json` 的 case-local 模板，并全部标记 `template_only_not_solution` / `passes: false` 后 fail closed。implementation prompt 明确 LLM 可以复制这些输出模板，但必须用真实计算 evidence 替换所有模板标记后才能报告结果。该 reference 仍是可编辑 scaffold，不是固定分子 solver。

[done] 推进 molecular implementation handoff 的 review 层：`compile_ks_dft_taps_kernel` 现在会在检测到 `materials/ks_dft_molecular_context.json` 时，把 molecular-specific static checks 写入 `ks_dft_kernel_review_spec.json`：拒绝 final `taps/kernel.py` 保留 `template_only_not_solution`，要求写入 `verification/ks_dft/molecular_boundary_evidence.json`，并要求 `ks_dft_runtime_metadata.json` 中显式记录 `poisson_boundary_policy` 与 `molecular_scaling_policy`。通用 `review_generated_taps_kernel` 在没有普通 `kernel_review_spec.json` 时会自动拾取 DFT 专用 spec，因此 DeepAgents/TUI 默认 review 也能走 DFT handoff gate。该层只约束 LLM 产物的 artifact/provenance，不把分子 Poisson、scaling 或 correction 的数值规则硬编码为默认 solver。

[done] 推进 multi-k band/DOS 的 LLM-driven contract：新增 `prepare_ks_dft_multik_integration_policy`，为 band/DOS 写出 `taps/ks_dft_multik_integration_policy.json/md`，声明 allowed k-dependent Hamiltonian strategies、`required_mode`、是否要求 self-consistent multi-k density、runtime metadata keys、validated multi-k 输出要求和禁止把 Gamma-derived k-shift model 冒充 multi-k Hamiltonian 的规则。`prepare_verified_ks_dft_band_dos_preflight` 现在会纳入该 policy，并把现有 band/DOS 显式标成 `post_scf_model`；`check_ks_band_dos_provenance` 新增 `require_validated_multik_hamiltonian=True` gate，能在需要 validated multi-k 时拒绝 post-SCF Gamma 派生输出。这仍是让 LLM 实现 case-local multi-k kernel 的 artifact/review/verification 契约，而不是新增固定 multi-k solver。
