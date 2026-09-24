# TAPS Matrix Definition Reference

Allowed matrix concepts for TAPS derivations:

```text
M[d]      mass matrix on axis d
K[d]      stiffness matrix on axis d
C[d]      coefficient matrix on axis d
Q[d]      source/load vector on axis d
A[d]      assembled operator block for subspace iteration on axis d
B[d]      time derivative or auxiliary operator block on axis d
```

Allowed tensor/vector operations:

```text
vec(U[d])
Kronecker products between axis matrices
Hadamard products for separated coefficient factors
axis-wise quadrature over x, y, z, parameter axes, and time
```

Rules:

- Define every matrix before using it.
- State which axes each matrix depends on.
- Do not invent matrix symbols outside this reference unless the derivation adds a clear definition.
- For geometry embedding, define `Phi`, `Chi`, boundary sample, normal, and cut-cell matrices before use.
