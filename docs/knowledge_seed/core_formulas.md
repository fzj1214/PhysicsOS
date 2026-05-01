# Core Computational Physics Formula Notes

This file is a compact formula seed for PhysicsOS knowledge retrieval. It is not a replacement for domain references; it gives agents a stable local context for common equations, weak forms, residuals, and verification checks.

## Conservation Law Template

Strong conservative form:

```text
∂u/∂t + ∇ · F(u, ∇u, x, t; μ) = S(u, x, t; μ)
```

Residual:

```text
R(u) = ∂u/∂t + ∇ · F(u, ∇u) - S(u)
```

Integral conservation over control volume Ω:

```text
d/dt ∫_Ω u dΩ + ∫_∂Ω F · n dΓ = ∫_Ω S dΩ
```

## Galerkin Weak Form Template

Given a strong form:

```text
L(u; μ) = f
```

Find `u_h ∈ V_h` such that for all test functions `v_h ∈ V_h`:

```text
∫_Ω v_h L(u_h; μ) dΩ = ∫_Ω v_h f dΩ
```

After integration by parts for second-order diffusion:

```text
∫_Ω ∇v_h · k ∇u_h dΩ = ∫_Ω v_h f dΩ + ∫_ΓN v_h g_N dΓ
```

## Heat Equation

Transient heat equation:

```text
ρ c_p ∂T/∂t - ∇ · (k ∇T) = Q
```

Weak form:

```text
∫_Ω v ρ c_p ∂T/∂t dΩ + ∫_Ω ∇v · k ∇T dΩ = ∫_Ω v Q dΩ + ∫_ΓN v q dΓ
```

Robin convection boundary:

```text
-k ∇T · n = h (T - T_∞)
```

Radiation boundary:

```text
-k ∇T · n = ε σ (T^4 - T_∞^4)
```

## Poisson / Diffusion

Poisson equation:

```text
-∇ · (κ ∇u) = f
```

Weak form:

```text
∫_Ω ∇v · κ ∇u dΩ = ∫_Ω v f dΩ + ∫_ΓN v g dΓ
```

## Linear Elasticity

Balance of linear momentum:

```text
∇ · σ + b = ρ ∂²u/∂t²
```

Small strain:

```text
ε(u) = 1/2 (∇u + ∇uᵀ)
```

Linear isotropic constitutive law:

```text
σ = λ tr(ε) I + 2 μ ε
```

Weak static form:

```text
∫_Ω ε(v) : C : ε(u) dΩ = ∫_Ω v · b dΩ + ∫_ΓN v · t dΓ
```

## Incompressible Navier-Stokes

Momentum:

```text
ρ(∂u/∂t + u · ∇u) = -∇p + μ ∇²u + f
```

Continuity:

```text
∇ · u = 0
```

Dimensionless Reynolds number:

```text
Re = ρ U L / μ
```

## Helmholtz

Helmholtz equation:

```text
∇²u + k²u = f
```

Weak form:

```text
∫_Ω ∇v · ∇u dΩ - ∫_Ω k² v u dΩ = -∫_Ω v f dΩ + boundary terms
```

## Reaction-Diffusion

General form:

```text
∂u/∂t = D ∇²u + R(u; μ)
```

Gray-Scott model:

```text
∂u/∂t = D_u ∇²u - u v² + F(1 - u)
∂v/∂t = D_v ∇²v + u v² - (F + k)v
```

## DFT / Kohn-Sham

Kohn-Sham equation:

```text
[-1/2 ∇² + V_eff[n](r)] ψ_i(r) = ε_i ψ_i(r)
```

Electron density:

```text
n(r) = Σ_i f_i |ψ_i(r)|²
```

Total energy:

```text
E[n] = T_s[n] + E_ext[n] + E_H[n] + E_xc[n] + E_ion-ion
```

## Molecular Dynamics

Newtonian dynamics:

```text
m_i d²r_i/dt² = F_i = -∇_{r_i} U(r_1, ..., r_N)
```

Velocity Verlet:

```text
r(t+Δt) = r(t) + v(t)Δt + 1/2 a(t)Δt²
v(t+Δt) = v(t) + 1/2 [a(t) + a(t+Δt)]Δt
```

## Neural Operator Template

Operator learning:

```text
G_θ: a(x) ↦ u(x)
```

Fourier Neural Operator layer:

```text
v_{l+1}(x) = σ(W v_l(x) + F^{-1}(R · F(v_l))(x))
```

DeepONet form:

```text
G(u)(y) ≈ Σ_k b_k(u(x_1), ..., u(x_m)) t_k(y)
```

## TAPS Formula Template

Space-parameter-time function:

```text
u = u(x_s, x_p, x_t)
```

CP tensor approximation:

```text
u(x_1, ..., x_D) ≈ Σ_{m=1}^M ∏_{d=1}^D u_m^{(d)}(x_d)
```

DoF compression:

```text
full tensor: n^D
CP/TAPS: M · D · n
```

TAPS weak residual target:

```text
Find u_TD such that ∫_{Ω_s × Ω_p × Ω_t} v_TD R(u_TD; x_s, x_p, x_t) dΩ = 0
```

Main accuracy controls:

```text
M = tensor rank
n = points per axis
p = reproducing polynomial order
s = C-HiDeNN patch size
a = dilation
quad = quadrature order
slabs = domain/time/parameter partitioning
```

## Verification Metrics

Normalized residual:

```text
η_R = ||R(u_h)|| / (||f|| + ε)
```

Conservation error:

```text
η_C = |inflow - outflow + source - accumulation| / reference_scale
```

Relative L2 error:

```text
e_L2 = ||u_h - u_ref||_2 / ||u_ref||_2
```

Energy norm error:

```text
e_E = sqrt((u_h - u_ref)^T K (u_h - u_ref))
```

