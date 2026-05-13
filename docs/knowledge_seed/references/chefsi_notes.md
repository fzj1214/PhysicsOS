# CheFSI Notes For KS-DFT-TAPS

Chebyshev-filtered subspace iteration updates the occupied eigenspace without full diagonalization at every SCF step.

Sketch:

```text
given H[n_j], Ψ_occ,j
estimate spectral bounds
apply Chebyshev filter to Ψ_occ,j
S-orthonormalize
Rayleigh-Ritz projection when needed
reconstruct n_{j+1}
```

In KS-DFT-TAPS, CheFSI is the occupied-subspace update inside the tensorized SCF loop.
