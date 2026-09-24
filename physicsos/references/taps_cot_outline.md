# TAPS Chain-of-Thought Derivation Outline

The derivation artifact must show the mathematical reasoning steps. It must not jump directly to final matrices.

For every subspace iteration:

```text
1. Start with the weak form.
2. Insert the TAPS trial function.
3. Insert the corresponding test function variation.
4. Separate axis-dependent factors.
5. Define all stiffness, mass, coefficient, source, and geometry matrices.
6. Assemble the subspace linear or nonlinear system.
7. State how the current axis update changes the remaining axes.
```

For STL/immersed-boundary cases, add:

```text
8. Insert chi(x) or smoothed Heaviside weighting into volume integrals.
9. Insert phi(x), boundary samples, and normals into boundary constraint terms.
10. Explain cut-cell or near-boundary stabilization terms.
```
