# Materials Tool Contract For KS-DFT-TAPS

Use deterministic materials tools before KS-DFT-TAPS derivation.

Required artifacts:

```text
materials/source_structure.json
materials/structure_standardized.json
materials/symmetry_dataset.json
materials/reciprocal_lattice.json
materials/kmesh.json
materials/irreducible_kpoints.json
materials/ks_dft_material_context.md
```

Hard rules:

- Use pymatgen, seekpath, and spglib wrappers for structure parsing, standardization, symmetry, reciprocal lattice, k-mesh, irreducible k-points, and high-symmetry paths.
- Do not infer space group, Wyckoff labels, primitive/conventional transforms, reciprocal conventions, or k-path labels in prompt text.
- Derivation and implementation agents must treat material artifacts as fixed inputs.
- If required material artifacts are missing, stop and request `materials-preprocess-agent`.
