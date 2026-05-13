# LRDM SCF Notes For KS-DFT-TAPS

Kohn-Sham SCF can be accelerated with a low-rank approximation of the dielectric response.

SCF residual:

```text
R[n] = F[n] - n
```

Low-rank dielectric preconditioner sketch:

```text
J ≈ Σ_l g_l ⊗ dR[g_l]
P_LRDM ≈ ε^{-1}_low_rank
n_{j+1} = n_j + P_LRDM * mixed_update(R[n_j])
```

Record direction functions, preconditioner rank, residual history, and failure modes in runtime metadata.
