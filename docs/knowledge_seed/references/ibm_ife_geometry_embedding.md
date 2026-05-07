# IBM/IFE Geometry Embedding Reference For STL TAPS

PhysicsOS uses STL/CAD geometry through immersed-boundary or immersed finite element coupling on a Cartesian TAPS background grid.

This is the only PhysicsOS extension to the paper-style route. It supplies analysis files for the TAPS derivation prompt; it is not a separate PDE solver, not a full-solver fallback, and not verification evidence.

Geometry definitions:

```text
Omega_bg  = Cartesian background domain
Omega_stl = physical domain embedded in Omega_bg
phi(x)    = signed distance field from STL/Gmsh
chi(x)    = H(-phi(x)), characteristic or smoothed occupancy function
n(x)      = boundary normal derived from grad(phi)
```

Representative immersed weak form for diffusion:

```text
Integral_Omega_bg chi k grad(v) . grad(u) dOmega
+ boundary_constraint_terms(phi, u, g, v)
= Integral_Omega_bg chi v f dOmega
+ Neumann_terms(phi, h, v)
```

Default first implementation:

- Use SDF/occupancy weighting for volume integrals.
- Use penalty or Nitsche-style Dirichlet enforcement on boundary samples.
- Use normals from SDF gradients or Gmsh boundary sampling.
- Record cut-cell and near-boundary cells for stabilization.

TAPS derivation requirement:

- Derive how `chi`, `phi`, boundary samples, normals, and geometry parameters enter the coefficient matrices and subspace iterations.
- Keep Gmsh as a geometry/SDF preprocessor only.
- Leave mathematical derivation, generated implementation code, and Fig. 7 verification in the main TAPS prompt-engineering workflow.
