# TAPS Template Derivation Reference

This is the case-local few-shot template source for PhysicsOS TAPS derivations.

Use this file the way the paper uses the complete derivation for Eq. 5:

1. Start from the complete template derivation.
2. Preserve the TAPS derivation structure.
3. Replace only the problem-specific PDE, coefficients, boundary terms, parameter axes, and subspace matrices.
4. Keep all subspace iterations explicit.

Required section outline:

```text
1. Problem Setup and Governing Equation
2. Weak Form Derivation
3. Tensor Decomposition Framework
3. C-HiDeNN-TD Approximation
4. Subspace Iteration Concept
5. X-Direction Subspace Iteration - Complete Derivation
6. T-Direction Subspace Iteration - Complete Derivation
7. K-Direction Subspace Iteration - Complete Derivation
8. Matrix Assembly and Kronecker Products Physical
9. Interpretation and Computational Aspects
```

Template PDE family:

```text
du/dt - d/dx(k du/dx) = f(x, k, t)
```

Template TAPS representation:

```text
u_TD(x, k, t) = sum_m u_x^m(x) u_k^m(k) u_t^m(t)
```

The actual generated derivation must include the full weak-form and matrix derivation for the target problem.
