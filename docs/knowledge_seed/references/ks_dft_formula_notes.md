# KS-DFT Formula Notes

Kohn-Sham equation:

```text
[-1/2 ∇² + V_eff[n](r)] ψ_i(r) = ε_i ψ_i(r)
n(r) = Σ_i f_i |ψ_i(r)|²
V_eff[n] = V_ext + V_H[n] + V_xc[n]
R_scf[n] = F[V_eff[n]] - n
```

Periodic Bloch form:

```text
ψ_{i,k}(r) = exp(i k · r) u_{i,k}(r)
H_k[n] u_{i,k} = ε_{i,k} S_k u_{i,k}
n(r) = Σ_k w_k Σ_i f_{i,k} |ψ_{i,k}(r)|²
```

KS-DFT-TAPS derivations must define:

```text
M / S    mass or overlap matrix
K        kinetic or stiffness matrix
V_ext    pseudopotential / external potential matrix
V_H      Hartree matrix
V_xc     exchange-correlation matrix
H        Kohn-Sham Hamiltonian
C_occ    occupied subspace coefficients
Γ        density matrix
R_scf    SCF residual
```
