# TAPS Verification Workflow Reference

The paper-style verification chain is:

```text
generate_exact_sol_code
execute_exact_sol_code
generate_convergence_code
execute_convergence_code
plot_result
```

PhysicsOS verification artifacts:

```text
verification/exact_solution.py
verification/convergence_study.py
verification/plots/
verification/report.md
verification/report.json
```

Verification checks:

- PDE residual or manufactured-solution residual.
- Boundary condition enforcement.
- Relative L2 error when exact/manufactured solution exists.
- Convergence rate across background-grid refinements or TAPS rank/order refinements.
- Geometry embedding sensitivity for STL/SDF cases.

If no exact solution is available, report the missing evidence explicitly and use residual, boundary, and convergence evidence.
