from __future__ import annotations

from typing import Literal

from pydantic import Field

from physicsos.backends.catalog import DEFAULT_BACKENDS
from physicsos.schemas.common import StrictBaseModel
from physicsos.schemas.operators import PhysicsDomain


class OperatorTemplate(StrictBaseModel):
    id: str
    name: str
    domain: PhysicsDomain
    equation_class: str
    default_form: Literal["strong", "weak", "integral", "discrete", "learned"]
    fields: list[str] = Field(default_factory=list)
    required_boundary_kinds: list[str] = Field(default_factory=list)
    conserved_quantities: list[str] = Field(default_factory=list)
    nondimensional_numbers: list[str] = Field(default_factory=list)
    taps_candidate: bool = False
    notes: list[str] = Field(default_factory=list)


class SolverBackendEntry(StrictBaseModel):
    id: str
    family: Literal["taps", "surrogate", "neural_operator", "fem", "fvm", "md", "dft", "kinetics", "monte_carlo", "custom"]
    domains: list[PhysicsDomain]
    command: str | None = None
    python_integration: str
    role: str
    default_remote_runner: str | None = None
    open_source: bool = True
    requires_remote_service: bool = False
    notes: list[str] = Field(default_factory=list)


class VerificationRule(StrictBaseModel):
    id: str
    domains: list[PhysicsDomain]
    operator_classes: list[str] = Field(default_factory=list)
    checks: list[str] = Field(default_factory=list)
    failure_action: str
    notes: list[str] = Field(default_factory=list)


class PostprocessTemplate(StrictBaseModel):
    id: str
    domains: list[PhysicsDomain]
    result_kinds: list[str] = Field(default_factory=list)
    kpis: list[str] = Field(default_factory=list)
    visualizations: list[str] = Field(default_factory=list)
    report_sections: list[str] = Field(default_factory=list)


OPERATOR_TEMPLATES: tuple[OperatorTemplate, ...] = (
    OperatorTemplate(
        id="operator_template:heat",
        name="Heat equation",
        domain="thermal",
        equation_class="heat",
        default_form="weak",
        fields=["T"],
        required_boundary_kinds=["dirichlet", "neumann", "robin"],
        conserved_quantities=["energy"],
        nondimensional_numbers=["Fourier", "Biot", "Peclet"],
        taps_candidate=True,
        notes=["Primary TAPS MVP operator for transient and steady diffusion-like problems."],
    ),
    OperatorTemplate(
        id="operator_template:poisson",
        name="Poisson equation",
        domain="custom",
        equation_class="poisson",
        default_form="weak",
        fields=["u"],
        required_boundary_kinds=["dirichlet", "neumann"],
        taps_candidate=True,
        notes=["Supported by tensorized 1D/2D and mesh FEM-like TAPS kernels."],
    ),
    OperatorTemplate(
        id="operator_template:reaction_diffusion",
        name="Reaction-diffusion",
        domain="custom",
        equation_class="reaction_diffusion",
        default_form="weak",
        fields=["u"],
        required_boundary_kinds=["dirichlet", "neumann"],
        taps_candidate=True,
        notes=["Supported by nonlinear fixed-point TAPS kernels for scalar fields."],
    ),
    OperatorTemplate(
        id="operator_template:navier_stokes",
        name="Navier-Stokes",
        domain="fluid",
        equation_class="navier_stokes",
        default_form="strong",
        fields=["U", "p"],
        required_boundary_kinds=["inlet", "outlet", "wall", "symmetry"],
        conserved_quantities=["mass", "momentum"],
        nondimensional_numbers=["Reynolds", "Mach", "Peclet"],
        taps_candidate=False,
        notes=["Route high-risk cases to OpenFOAM/SU2; TAPS requires reduced or simplified regimes first."],
    ),
    OperatorTemplate(
        id="operator_template:linear_elasticity",
        name="Linear elasticity",
        domain="solid",
        equation_class="linear_elasticity",
        default_form="weak",
        fields=["u", "sigma"],
        required_boundary_kinds=["dirichlet", "neumann"],
        conserved_quantities=["momentum"],
        taps_candidate=False,
        notes=["Planned vector-valued FEM/TAPS extension."],
    ),
    OperatorTemplate(
        id="operator_template:maxwell",
        name="Maxwell equations",
        domain="electromagnetic",
        equation_class="maxwell",
        default_form="weak",
        fields=["E", "H"],
        required_boundary_kinds=["dirichlet", "farfield", "interface"],
        conserved_quantities=["charge", "energy"],
        taps_candidate=False,
        notes=["Requires vector-valued curl/curl weak forms."],
    ),
)


SOLVER_BACKEND_REGISTRY: tuple[SolverBackendEntry, ...] = (
    SolverBackendEntry(
        id="taps",
        family="taps",
        domains=["thermal", "custom", "multiphysics"],
        python_integration="PhysicsOS native",
        role="primary equation-driven solver compiler",
        notes=["TAPS-first for explicit PDEs with verifiable residuals."],
    ),
    SolverBackendEntry(
        id="grid_neural_operator",
        family="neural_operator",
        domains=["fluid", "thermal", "custom"],
        python_integration="PyTorch/checkpoint adapter",
        role="regular-grid surrogate fallback",
        notes=["Use for structured fields and fast approximate inference."],
    ),
    SolverBackendEntry(
        id="mesh_graph_operator",
        family="neural_operator",
        domains=["fluid", "thermal", "solid", "custom"],
        python_integration="PyTorch Geometric-style adapter",
        role="mesh graph surrogate fallback",
        notes=["Use when geometry is naturally represented as graph/mesh connectivity."],
    ),
    *(
        SolverBackendEntry(
            id=backend.name,
            family=(
                "fvm" if backend.name in {"openfoam", "su2"} else
                "fem" if backend.name in {"fenicsx", "mfem"} else
                "dft" if backend.name in {"quantum_espresso", "cp2k"} else
                "md" if backend.name == "lammps" else
                "kinetics" if backend.name == "cantera" else
                "custom"
            ),
            domains=list(backend.domains),  # type: ignore[arg-type]
            command=backend.command,
            python_integration=backend.python_integration,
            role=backend.role,
            default_remote_runner="foamvm" if backend.name == "openfoam" else None,
            requires_remote_service=backend.name in {"openfoam", "su2", "quantum_espresso", "cp2k", "lammps"},
        )
        for backend in DEFAULT_BACKENDS
    ),
)


VERIFICATION_RULES: tuple[VerificationRule, ...] = (
    VerificationRule(
        id="verification:diffusion_residual",
        domains=["thermal", "custom"],
        operator_classes=["heat", "poisson", "diffusion", "reaction_diffusion", "helmholtz"],
        checks=["normalized_residual", "selected_slice_validation", "boundary_condition_consistency"],
        failure_action="refine_taps_or_run_full_solver",
    ),
    VerificationRule(
        id="verification:fluid_conservation",
        domains=["fluid"],
        operator_classes=["navier_stokes", "euler", "rans"],
        checks=["mass_conservation", "momentum_balance", "mesh_quality", "courant_or_residual_history"],
        failure_action="run_openfoam_or_su2_validation",
    ),
    VerificationRule(
        id="verification:surrogate_ood",
        domains=["fluid", "thermal", "solid", "custom", "multiphysics"],
        checks=["training_distribution_distance", "regime_bounds", "geometry_encoding_coverage", "residual_proxy"],
        failure_action="run_full_solver_or_request_user_review",
    ),
)


POSTPROCESS_TEMPLATES: tuple[PostprocessTemplate, ...] = (
    PostprocessTemplate(
        id="postprocess:thermal_report",
        domains=["thermal", "custom"],
        result_kinds=["temperature_field", "scalar_field"],
        kpis=["max_temperature", "mean_temperature", "thermal_resistance", "residual_norm"],
        visualizations=["temperature_contour", "residual_history", "slice_comparison"],
        report_sections=["assumptions", "geometry_and_mesh", "solver", "verification", "thermal_kpis", "next_actions"],
    ),
    PostprocessTemplate(
        id="postprocess:cfd_report",
        domains=["fluid"],
        result_kinds=["velocity_pressure_field"],
        kpis=["pressure_drop", "drag", "lift", "mass_flow", "residual_norm"],
        visualizations=["velocity_contour", "pressure_contour", "streamlines", "residual_history"],
        report_sections=["assumptions", "case_setup", "mesh_quality", "solver_logs", "cfd_kpis", "artifacts"],
    ),
    PostprocessTemplate(
        id="postprocess:surrogate_validation",
        domains=["fluid", "thermal", "solid", "custom", "multiphysics"],
        result_kinds=["surrogate_prediction"],
        kpis=["uncertainty", "ood_score", "residual_proxy", "reference_error"],
        visualizations=["prediction_preview", "uncertainty_map", "reference_slices"],
        report_sections=["model_scope", "input_encoding", "uncertainty", "verification", "fallback_recommendation"],
    ),
)
