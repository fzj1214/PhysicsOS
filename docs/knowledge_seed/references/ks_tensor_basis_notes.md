# KS Tensor Basis Notes

KS-DFT-TAPS should use tensor-structured representations for density, potential, Hamiltonian factors, and occupied subspace.

Representative form:

```text
n_TD(r, k, p, s)
V_eff,TD(r, p, s)
Ψ_occ,TD(r, band, k, p)
H_TD[n]
R_scf,TD[n]
```

The material tools provide the stable real and reciprocal domains. The TAPS derivation defines the basis functions, tensor rank, axis matrices, and refinement policy.

Minimum verification:

```text
rank refinement
grid refinement
k-point refinement
charge conservation
occupied-subspace orthonormality
SCF residual
```
