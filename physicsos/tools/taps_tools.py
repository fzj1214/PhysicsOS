from __future__ import annotations

from pydantic import Field

from physicsos.config import project_root
from physicsos.backends.taps_generic import SUPPORTED_FAMILIES as GENERIC_TAPS_FAMILIES
from physicsos.backends.taps_generic import solve_coupled_reaction_diffusion_2d
from physicsos.backends.taps_generic import solve_graph_poisson
from physicsos.backends.taps_generic import solve_mesh_fem_em_curl_curl
from physicsos.backends.taps_generic import solve_mesh_fem_linear_elasticity
from physicsos.backends.taps_generic import solve_mesh_fem_poisson
from physicsos.backends.taps_generic import solve_reaction_diffusion_nonlinear_1d
from physicsos.backends.taps_generic import solve_reaction_diffusion_nonlinear_2d
from physicsos.backends.taps_generic import solve_scalar_elliptic_1d
from physicsos.backends.taps_generic import solve_scalar_elliptic_2d
from physicsos.backends.taps_thermal import solve_transient_heat_1d
from physicsos.schemas.common import ArtifactRef, Provenance, StrictBaseModel
from physicsos.schemas.knowledge import KnowledgeContext
from physicsos.schemas.problem import PhysicsProblem
from physicsos.schemas.solver import SolverResult
from physicsos.schemas.taps import (
    TAPSAxisSpec,
    TAPSBasisConfig,
    TAPSBoundaryConditionSpec,
    TAPSCompilationPlan,
    TAPSCoefficientSpec,
    TAPSEquationTerm,
    TAPSGeometryEncodingSpec,
    TAPSProblem,
    TAPSResidualReport,
    TAPSRuntimeExtensionSpec,
    TAPSSupportScore,
    TAPSWeakFormSpec,
)


class EstimateTAPSSupportInput(StrictBaseModel):
    problem: PhysicsProblem


class EstimateTAPSSupportOutput(StrictBaseModel):
    support: TAPSSupportScore


def estimate_taps_support(input: EstimateTAPSSupportInput) -> EstimateTAPSSupportOutput:
    """Estimate whether TAPS should be the first solver for this PhysicsProblem."""
    problem = input.problem
    reasons: list[str] = []
    risks: list[str] = []
    score = 0.0

    if problem.domain in {"thermal", "solid", "acoustic", "custom", "multiphysics"}:
        score += 0.25
        reasons.append(f"domain {problem.domain} is compatible with first TAPS targets")
    else:
        risks.append(f"domain {problem.domain} may be difficult for early TAPS")

    operator_classes = {operator.equation_class.lower() for operator in problem.operators}
    preferred = {
        "heat",
        "diffusion",
        "poisson",
        "helmholtz",
        "reaction_diffusion",
        "coupled_reaction_diffusion",
        "elasticity",
        "linear_elasticity",
        "maxwell",
        "curl_curl",
        "electromagnetic",
    }
    if operator_classes.intersection(preferred):
        score += 0.35
        reasons.append("operator family is a TAPS-first target")
    else:
        risks.append("operator family is not yet a TAPS-first target")

    if problem.parameters:
        score += 0.2
        reasons.append("parameter axes are available")
    else:
        risks.append("no explicit parameter axes; TAPS still possible but less valuable")

    if problem.geometry.encodings:
        score += 0.2
        reasons.append("geometry encodings can support tensorization")
    else:
        risks.append("geometry needs tensorization/SDF/parameter encoding")

    return EstimateTAPSSupportOutput(
        support=TAPSSupportScore(score=round(score, 4), supported=score >= 0.45, reasons=reasons, risks=risks)
    )


class FormulateTAPSEquationInput(StrictBaseModel):
    problem: PhysicsProblem
    knowledge_context: KnowledgeContext | None = None
    allow_knowledge_queries: bool = True


class FormulateTAPSEquationOutput(StrictBaseModel):
    plan: TAPSCompilationPlan


def _default_axes(problem: PhysicsProblem) -> list[TAPSAxisSpec]:
    dimension = max(1, problem.geometry.dimension)
    axis_names = ["x", "y", "z"][:dimension]
    axes = [TAPSAxisSpec(name=name, kind="space", min_value=0.0, max_value=1.0, points=64) for name in axis_names]
    axes.extend(TAPSAxisSpec(name=parameter.name, kind="parameter", points=16, units=parameter.units) for parameter in problem.parameters)
    if problem.initial_conditions:
        axes.append(TAPSAxisSpec(name="t", kind="time", min_value=0.0, max_value=1.0, points=32))
    return axes


def _weak_form_from_problem(problem: PhysicsProblem, knowledge_context: KnowledgeContext | None) -> TAPSWeakFormSpec | None:
    if not problem.operators or not problem.fields:
        return None
    family = problem.operators[0].equation_class.lower()
    fields = [field.name for field in problem.fields]
    terms: list[TAPSEquationTerm] = []
    source = "physics_problem"
    if knowledge_context is not None and (knowledge_context.chunks or knowledge_context.deepsearch):
        source = "hybrid"

    for operator in problem.operators:
        operator_family = operator.equation_class.lower()
        if operator_family in {"heat", "diffusion", "thermal_diffusion"}:
            if problem.initial_conditions:
                terms.append(
                    TAPSEquationTerm(
                        id=f"{operator.id}:time",
                        role="time_derivative",
                        expression="int_Omega v * dT/dt dOmega",
                        fields=operator.fields_out or fields,
                    )
                )
            terms.append(
                TAPSEquationTerm(
                    id=f"{operator.id}:diffusion",
                    role="diffusion",
                    expression="int_Omega grad(v) · alpha grad(T) dOmega",
                    fields=operator.fields_out or fields,
                    coefficients=["alpha", "thermal_diffusivity"],
                )
            )
        elif operator_family == "poisson":
            terms.append(
                TAPSEquationTerm(
                    id=f"{operator.id}:laplacian",
                    role="diffusion",
                    expression="int_Omega grad(v) · k grad(u) dOmega - int_Omega v f dOmega",
                    fields=operator.fields_out or fields,
                    coefficients=["k", "f"],
                )
            )
        elif operator_family == "helmholtz":
            terms.extend(
                [
                    TAPSEquationTerm(
                        id=f"{operator.id}:stiffness",
                        role="diffusion",
                        expression="int_Omega grad(v) · grad(u) dOmega",
                        fields=operator.fields_out or fields,
                    ),
                    TAPSEquationTerm(
                        id=f"{operator.id}:wavenumber",
                        role="reaction",
                        expression="- int_Omega k^2 v u dOmega",
                        fields=operator.fields_out or fields,
                        coefficients=["k"],
                    ),
                ]
            )
        elif operator_family == "reaction_diffusion":
            terms.extend(
                [
                    TAPSEquationTerm(
                        id=f"{operator.id}:diffusion",
                        role="diffusion",
                        expression="int_Omega grad(v) · D grad(u) dOmega",
                        fields=operator.fields_out or fields,
                        coefficients=["D"],
                    ),
                    TAPSEquationTerm(
                        id=f"{operator.id}:reaction",
                        role="reaction",
                        expression="- int_Omega v R(u, parameters) dOmega",
                        fields=operator.fields_out or fields,
                        coefficients=["reaction_parameters"],
                    ),
                ]
            )
        elif operator_family == "coupled_reaction_diffusion":
            terms.extend(
                [
                    TAPSEquationTerm(
                        id=f"{operator.id}:diffusion",
                        role="diffusion",
                        expression="sum_i int_Omega grad(v_i) · D_i grad(u_i) dOmega",
                        fields=operator.fields_out or fields,
                        coefficients=["D_u", "D_v"],
                    ),
                    TAPSEquationTerm(
                        id=f"{operator.id}:reaction",
                        role="reaction",
                        expression="- sum_i int_Omega v_i R_i(u, v, parameters) dOmega",
                        fields=operator.fields_out or fields,
                        coefficients=["reaction_parameters"],
                    ),
                    TAPSEquationTerm(
                        id=f"{operator.id}:coupling",
                        role="constitutive",
                        expression="int_Omega kappa (u - v) (v_u - v_v) dOmega",
                        fields=operator.fields_out or fields,
                        coefficients=["coupling_strength"],
                    ),
                ]
            )
        elif operator_family in {"elasticity", "linear_elasticity"}:
            terms.extend(
                [
                    TAPSEquationTerm(
                        id=f"{operator.id}:strain_energy",
                        role="constitutive",
                        expression="int_Omega epsilon(v)^T C(E, nu) epsilon(u) dOmega",
                        fields=operator.fields_out or fields,
                        coefficients=["E", "nu", "constitutive_model"],
                        assumptions=["Small-strain 2D linear elasticity; default executable kernel uses plane stress."],
                    ),
                    TAPSEquationTerm(
                        id=f"{operator.id}:body_force",
                        role="source",
                        expression="- int_Omega v dot b dOmega",
                        fields=operator.fields_out or fields,
                        coefficients=["body_force"],
                    ),
                ]
            )
        elif operator_family in {"maxwell", "curl_curl", "electromagnetic"}:
            terms.extend(
                [
                    TAPSEquationTerm(
                        id=f"{operator.id}:curl_curl",
                        role="diffusion",
                        expression="int_Omega mu^-1 curl(v) dot curl(E) dOmega",
                        fields=operator.fields_out or fields,
                        coefficients=["mu_r", "relative_permeability"],
                        assumptions=["Executable local kernel uses 2D first-order Nedelec edge elements for H(curl) tangential continuity."],
                    ),
                    TAPSEquationTerm(
                        id=f"{operator.id}:mass",
                        role="mass",
                        expression="- int_Omega k0^2 epsilon v dot E dOmega",
                        fields=operator.fields_out or fields,
                        coefficients=["eps_r", "relative_permittivity", "k0", "wave_number"],
                    ),
                    TAPSEquationTerm(
                        id=f"{operator.id}:current_source",
                        role="source",
                        expression="- int_Omega v dot J dOmega",
                        fields=operator.fields_out or fields,
                        coefficients=["current_source", "J"],
                    ),
                ]
            )
        else:
            for index, term in enumerate(operator.differential_terms):
                expression = term.expression if hasattr(term, "expression") else term.get("expression", "")
                term_fields = term.fields if hasattr(term, "fields") else term.get("fields", [])
                terms.append(
                    TAPSEquationTerm(
                        id=f"{operator.id}:term:{index}",
                        role="custom",
                        expression=expression,
                        fields=term_fields or operator.fields_out or fields,
                    )
                )
            for index, term in enumerate(operator.source_terms):
                expression = term.expression if hasattr(term, "expression") else term.get("expression", "")
                terms.append(
                    TAPSEquationTerm(
                        id=f"{operator.id}:source:{index}",
                        role="source",
                        expression=expression,
                        fields=operator.fields_out or fields,
                    )
                )

    if not terms:
        return None
    residual = " + ".join(term.expression for term in terms)
    return TAPSWeakFormSpec(
        family=family,
        strong_form="; ".join(f"{operator.name}:{operator.equation_class}" for operator in problem.operators),
        trial_fields=fields,
        test_functions=[f"v_{field}" for field in fields],
        terms=terms,
        constraints=[constraint.expression for constraint in problem.constraints],
        residual_expression=f"Find fields such that {residual} = 0 for all test functions.",
        source=source,  # type: ignore[arg-type]
    )


def formulate_taps_equation(input: FormulateTAPSEquationInput) -> FormulateTAPSEquationOutput:
    """Formulate a general equation-driven TAPS compilation plan.

    This tool is intentionally not limited to solved templates. It extracts the
    best weak-form IR available from PhysicsProblem and asks knowledge-agent for
    missing equations, constitutive laws, nondimensional regimes, or validation
    rules when the problem is under-specified.
    """
    problem = input.problem
    missing: list[str] = []
    queries: list[str] = []
    risks: list[str] = []
    assumptions: list[str] = ["TAPS is treated as a weak-form equation compiler, not a fixed PDE template table."]

    if not problem.operators:
        missing.append("governing_operator")
    if not problem.fields:
        missing.append("unknown_fields")
    if not problem.boundary_conditions:
        missing.append("boundary_conditions")
    if problem.geometry.dimension > 1 and not problem.mesh and not problem.geometry.encodings:
        risks.append("geometry has no mesh or tensorizable encoding yet")
    if problem.domain in {"molecular", "quantum"}:
        risks.append("domain may require eigenvalue/statistical formulations before TAPS compilation")

    equation_family = problem.operators[0].equation_class.lower() if problem.operators else "unknown"
    weak_form = _weak_form_from_problem(problem, input.knowledge_context)
    axes = _default_axes(problem)

    if weak_form is None:
        missing.append("weak_form")
    if input.allow_knowledge_queries and missing:
        queries.append(
            f"Derive governing equations, weak form, boundary conditions, and validation residuals for: {problem.user_intent.raw_request}"
        )
    if input.allow_knowledge_queries and equation_family in {"unspecified", "unknown", "custom"}:
        queries.append(
            f"Identify a computable PDE/operator formulation for domain={problem.domain}, fields={[field.name for field in problem.fields]}"
        )

    if missing and queries:
        status = "needs_knowledge"
        action = "ask_knowledge_agent"
    elif missing:
        status = "needs_user_input"
        action = "ask_user"
    else:
        status = "ready"
        action = "compile_taps_problem"

    plan = TAPSCompilationPlan(
        problem_id=problem.id,
        status=status,  # type: ignore[arg-type]
        equation_family=equation_family,
        unknown_fields=[field.name for field in problem.fields],
        axes=axes,
        weak_form=weak_form,
        required_knowledge_queries=queries,
        missing_inputs=sorted(set(missing)),
        assumptions=assumptions,
        risks=risks,
        recommended_next_action=action,  # type: ignore[arg-type]
    )
    return FormulateTAPSEquationOutput(plan=plan)


class BuildTAPSProblemInput(StrictBaseModel):
    problem: PhysicsProblem
    basis: TAPSBasisConfig = Field(default_factory=TAPSBasisConfig)
    compilation_plan: TAPSCompilationPlan | None = None


class BuildTAPSProblemOutput(StrictBaseModel):
    taps_problem: TAPSProblem


def _compile_taps_coefficients(problem: PhysicsProblem) -> list[TAPSCoefficientSpec]:
    coefficients: list[TAPSCoefficientSpec] = []
    seen: set[tuple[str, str]] = set()
    for material in problem.materials:
        for prop in material.properties:
            key = (prop.name.lower(), ",".join(material.region_ids))
            if key in seen:
                continue
            seen.add(key)
            coefficients.append(
                TAPSCoefficientSpec(
                    name=prop.name,
                    value=prop.value,
                    units=prop.units,
                    role="material",
                    region_ids=material.region_ids,
                    source="physics_problem",
                )
            )
    for operator in problem.operators:
        for index, source_term in enumerate(operator.source_terms):
            expression = source_term.expression.strip()
            if not expression:
                continue
            lower = expression.lower()
            value: float | int | str | list[float] = expression
            name = f"{operator.id}:source:{index}"
            if lower in {"gravity_y", "unit_gravity_y"}:
                name = "body_force"
                value = [0.0, -1.0]
            elif lower in {"gravity_x", "unit_gravity_x"}:
                name = "body_force"
                value = [1.0, 0.0]
            coefficients.append(
                TAPSCoefficientSpec(
                    name=name,
                    value=value,
                    units=source_term.units,
                    role="source",
                    source="physics_problem",
                )
            )
    return coefficients


def _compile_taps_boundary_conditions(problem: PhysicsProblem) -> list[TAPSBoundaryConditionSpec]:
    return [
        TAPSBoundaryConditionSpec(
            id=boundary.id,
            region_id=boundary.region_id,
            field=boundary.field,
            kind=boundary.kind,
            value=boundary.value,
            units=boundary.units,
            source="physics_problem",
        )
        for boundary in problem.boundary_conditions
    ]


def build_taps_problem(input: BuildTAPSProblemInput) -> BuildTAPSProblemOutput:
    """Compile PhysicsProblem into a TAPSProblem."""
    plan = input.compilation_plan or formulate_taps_equation(FormulateTAPSEquationInput(problem=input.problem)).plan
    operator_classes = {operator.equation_class.lower() for operator in input.problem.operators}
    is_heat = bool(operator_classes.intersection({"heat", "diffusion", "thermal_diffusion"}))
    is_transient_heat = is_heat and bool(input.problem.initial_conditions)
    if is_transient_heat:
        axes = [
            TAPSAxisSpec(name="x", kind="space", min_value=0.0, max_value=1.0, points=128, units="m"),
            TAPSAxisSpec(name="alpha", kind="parameter", min_value=0.01, max_value=0.1, points=48, units="m^2/s"),
            TAPSAxisSpec(name="t", kind="time", min_value=0.0, max_value=1.0, points=64, units="s"),
        ]
    else:
        axes = plan.axes

    geometry_encodings = [
        TAPSGeometryEncodingSpec(
            kind=encoding.kind,
            uri=encoding.uri,
            resolution=encoding.resolution,
            target_backend=encoding.target_backend,
        )
        for encoding in input.problem.geometry.encodings
        if encoding.target_backend in {None, "taps"} or encoding.kind in {"occupancy_mask", "sdf", "mesh_graph", "laplacian_eigenbasis"}
    ]
    weak_form = "; ".join(f"{operator.name}:{operator.equation_class}" for operator in input.problem.operators)
    taps_problem = TAPSProblem(
        id=f"taps:{input.problem.id}",
        problem_id=input.problem.id,
        axes=axes,
        operator_weak_form=weak_form or None,
        weak_form=plan.weak_form,
        compilation_status="compiled" if plan.status == "ready" else "scaffold",
        basis=input.basis,
        geometry_encodings=geometry_encodings,
        coefficients=_compile_taps_coefficients(input.problem),
        boundary_conditions=_compile_taps_boundary_conditions(input.problem),
        assumptions=[
            "TAPSProblem generated from PhysicsProblem.",
            "TAPS-agent must use knowledge-agent to fill missing weak forms before high-trust solve.",
            "Current executable kernels support 1D transient heat, 1D/2D scalar elliptic weak forms, 1D/2D nonlinear reaction-diffusion, 2-field 2D coupled reaction-diffusion, mesh P1/P2/P3 2D linear elasticity, and mesh first-order Nedelec EM curl-curl.",
            "Geometry encodings are carried into TAPSProblem and executable kernels consume occupancy masks when available.",
            *plan.assumptions,
        ],
    )
    return BuildTAPSProblemOutput(taps_problem=taps_problem)


class RunTAPSBackendInput(StrictBaseModel):
    problem: PhysicsProblem
    taps_problem: TAPSProblem


class RunTAPSBackendOutput(StrictBaseModel):
    result: SolverResult


class AuthorTAPSRuntimeExtensionInput(StrictBaseModel):
    problem: PhysicsProblem
    purpose: str
    code: str
    entrypoint: str = "solve"
    language: str = "python"


class AuthorTAPSRuntimeExtensionOutput(StrictBaseModel):
    extension: TAPSRuntimeExtensionSpec


def author_taps_runtime_extension(input: AuthorTAPSRuntimeExtensionInput) -> AuthorTAPSRuntimeExtensionOutput:
    """Write a case-local draft TAPS runtime extension artifact.

    This tool lets taps-agent prototype a missing assembler/kernel without
    modifying PhysicsOS core code. Execution/registration should require review.
    """
    output_dir = project_root() / "scratch" / input.problem.id.replace(":", "_") / "taps_extensions"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "py" if input.language == "python" else "txt"
    path = output_dir / f"{input.entrypoint}.{suffix}"
    path.write_text(input.code, encoding="utf-8")
    extension = TAPSRuntimeExtensionSpec(
        id=f"taps-extension:{input.problem.id}:{input.entrypoint}",
        problem_id=input.problem.id,
        purpose=input.purpose,
        language=input.language,  # type: ignore[arg-type]
        entrypoint=input.entrypoint,
        artifact=ArtifactRef(uri=str(path), kind="taps_runtime_extension", format=suffix),
        required_inputs=["TAPSProblem", "PhysicsProblem"],
        expected_outputs=["TAPSResultArtifacts", "TAPSResidualReport"],
        safety_status="requires_review",
        notes=["Draft extension written by taps-agent; review before execution or promotion to core backend."],
    )
    return AuthorTAPSRuntimeExtensionOutput(extension=extension)


def run_taps_backend(input: RunTAPSBackendInput) -> RunTAPSBackendOutput:
    """Run the TAPS backend."""
    operator_classes = {operator.equation_class.lower() for operator in input.problem.operators}
    is_heat = bool(operator_classes.intersection({"heat", "diffusion", "thermal_diffusion"}))
    if is_heat and input.problem.initial_conditions:
        artifacts, residual_report = solve_transient_heat_1d(input.taps_problem)
        artifact_refs = []
        artifact_refs.extend(artifacts.factor_matrices)
        if artifacts.reconstruction_metadata is not None:
            artifact_refs.append(artifacts.reconstruction_metadata)
        if artifacts.residual_history is not None:
            artifact_refs.append(artifacts.residual_history)
        result = SolverResult(
            id=f"result:{input.taps_problem.id}",
            problem_id=input.problem.id,
            backend="taps:thermal_1d",
            status="success" if residual_report.converged else "needs_review",
            scalar_outputs={
                "message": "TAPS thermal MVP backend executed.",
                "tensor_rank": input.taps_problem.basis.tensor_rank,
                **residual_report.residuals,
            },
            residuals=residual_report.residuals,
            artifacts=artifact_refs,
            provenance=Provenance(created_by="run_taps_backend", source="taps_thermal_1d"),
        )
        return RunTAPSBackendOutput(result=result)

    family = input.taps_problem.weak_form.family.lower() if input.taps_problem.weak_form is not None else ""
    space_axis_count = len([axis for axis in input.taps_problem.axes if axis.kind == "space"])
    field_count = len(input.taps_problem.weak_form.trial_fields) if input.taps_problem.weak_form is not None else 0
    has_mesh_graph = any(encoding.kind == "mesh_graph" for encoding in input.taps_problem.geometry_encodings)
    if family in {"maxwell", "curl_curl", "electromagnetic"} and has_mesh_graph:
        artifacts, residual_report = solve_mesh_fem_em_curl_curl(input.taps_problem)
        artifact_refs = []
        artifact_refs.extend(artifacts.factor_matrices)
        if artifacts.reconstruction_metadata is not None:
            artifact_refs.append(artifacts.reconstruction_metadata)
        if artifacts.residual_history is not None:
            artifact_refs.append(artifacts.residual_history)
        result = SolverResult(
            id=f"result:{input.taps_problem.id}",
            problem_id=input.problem.id,
            backend=f"taps:mesh_fem_em_curl_curl:{family}",
            status="success" if residual_report.converged else "needs_review",
            scalar_outputs={
                "message": "Triangle first-order Nedelec H(curl) EM curl-curl TAPS kernel executed on mesh_graph.",
                "equation_family": family,
                **residual_report.residuals,
            },
            residuals=residual_report.residuals,
            artifacts=artifact_refs,
            provenance=Provenance(created_by="run_taps_backend", source="taps_mesh_fem_em_curl_curl"),
        )
        return RunTAPSBackendOutput(result=result)

    if family in {"elasticity", "linear_elasticity"} and has_mesh_graph:
        artifacts, residual_report = solve_mesh_fem_linear_elasticity(input.taps_problem)
        artifact_refs = []
        artifact_refs.extend(artifacts.factor_matrices)
        if artifacts.reconstruction_metadata is not None:
            artifact_refs.append(artifacts.reconstruction_metadata)
        if artifacts.residual_history is not None:
            artifact_refs.append(artifacts.residual_history)
        result = SolverResult(
            id=f"result:{input.taps_problem.id}",
            problem_id=input.problem.id,
            backend=f"taps:mesh_fem_linear_elasticity:{family}",
            status="success" if residual_report.converged else "needs_review",
            scalar_outputs={
                "message": "Triangle P1 vector FEM-like TAPS linear-elasticity kernel executed on mesh_graph geometry encoding.",
                "equation_family": family,
                **residual_report.residuals,
            },
            residuals=residual_report.residuals,
            artifacts=artifact_refs,
            provenance=Provenance(created_by="run_taps_backend", source="taps_mesh_fem_linear_elasticity"),
        )
        return RunTAPSBackendOutput(result=result)

    if family in {"poisson", "diffusion", "thermal_diffusion"} and has_mesh_graph:
        source = "taps_mesh_fem_poisson"
        backend = f"taps:mesh_fem_poisson:{family}"
        message = "Triangle P1/P2 FEM-like TAPS Poisson kernel executed on mesh_graph geometry encoding."
        try:
            artifacts, residual_report = solve_mesh_fem_poisson(input.taps_problem)
        except ValueError:
            source = "taps_graph_poisson"
            backend = f"taps:graph_poisson:{family}"
            message = "Graph-Laplacian TAPS Poisson kernel executed on mesh_graph geometry encoding."
            artifacts, residual_report = solve_graph_poisson(input.taps_problem)
        artifact_refs = []
        artifact_refs.extend(artifacts.factor_matrices)
        if artifacts.reconstruction_metadata is not None:
            artifact_refs.append(artifacts.reconstruction_metadata)
        if artifacts.residual_history is not None:
            artifact_refs.append(artifacts.residual_history)
        result = SolverResult(
            id=f"result:{input.taps_problem.id}",
            problem_id=input.problem.id,
            backend=backend,
            status="success" if residual_report.converged else "needs_review",
            scalar_outputs={
                "message": message,
                "equation_family": family,
                **residual_report.residuals,
            },
            residuals=residual_report.residuals,
            artifacts=artifact_refs,
            provenance=Provenance(created_by="run_taps_backend", source=source),
        )
        return RunTAPSBackendOutput(result=result)

    if family == "coupled_reaction_diffusion" and space_axis_count == 2 and field_count >= 2:
        artifacts, residual_report = solve_coupled_reaction_diffusion_2d(input.taps_problem)
        artifact_refs = []
        artifact_refs.extend(artifacts.factor_matrices)
        if artifacts.reconstruction_metadata is not None:
            artifact_refs.append(artifacts.reconstruction_metadata)
        if artifacts.residual_history is not None:
            artifact_refs.append(artifacts.residual_history)
        result = SolverResult(
            id=f"result:{input.taps_problem.id}",
            problem_id=input.problem.id,
            backend="taps:coupled_reaction_diffusion_2d",
            status="success" if residual_report.converged else "needs_review",
            scalar_outputs={
                "message": "Coupled-field TAPS 2D reaction-diffusion fixed-point kernel executed.",
                "equation_family": family,
                "field_count": field_count,
                "tensor_rank": residual_report.rank,
                **residual_report.residuals,
            },
            residuals=residual_report.residuals,
            artifacts=artifact_refs,
            provenance=Provenance(created_by="run_taps_backend", source="taps_coupled_reaction_diffusion_2d"),
        )
        return RunTAPSBackendOutput(result=result)

    if family == "reaction_diffusion" and space_axis_count == 1:
        artifacts, residual_report = solve_reaction_diffusion_nonlinear_1d(input.taps_problem)
        artifact_refs = []
        artifact_refs.extend(artifacts.factor_matrices)
        if artifacts.reconstruction_metadata is not None:
            artifact_refs.append(artifacts.reconstruction_metadata)
        if artifacts.residual_history is not None:
            artifact_refs.append(artifacts.residual_history)
        result = SolverResult(
            id=f"result:{input.taps_problem.id}",
            problem_id=input.problem.id,
            backend="taps:nonlinear_reaction_diffusion_1d",
            status="success" if residual_report.converged else "needs_review",
            scalar_outputs={
                "message": "Nonlinear TAPS 1D reaction-diffusion Picard kernel executed.",
                "equation_family": family,
                "tensor_rank": residual_report.rank,
                **residual_report.residuals,
            },
            residuals=residual_report.residuals,
            artifacts=artifact_refs,
            provenance=Provenance(created_by="run_taps_backend", source="taps_nonlinear_reaction_diffusion_1d"),
        )
        return RunTAPSBackendOutput(result=result)

    if family == "reaction_diffusion" and space_axis_count == 2:
        artifacts, residual_report = solve_reaction_diffusion_nonlinear_2d(input.taps_problem)
        artifact_refs = []
        artifact_refs.extend(artifacts.factor_matrices)
        if artifacts.reconstruction_metadata is not None:
            artifact_refs.append(artifacts.reconstruction_metadata)
        if artifacts.residual_history is not None:
            artifact_refs.append(artifacts.residual_history)
        result = SolverResult(
            id=f"result:{input.taps_problem.id}",
            problem_id=input.problem.id,
            backend="taps:nonlinear_reaction_diffusion_2d",
            status="success" if residual_report.converged else "needs_review",
            scalar_outputs={
                "message": "Nonlinear TAPS 2D reaction-diffusion fixed-point kernel executed.",
                "equation_family": family,
                "tensor_rank": residual_report.rank,
                **residual_report.residuals,
            },
            residuals=residual_report.residuals,
            artifacts=artifact_refs,
            provenance=Provenance(created_by="run_taps_backend", source="taps_nonlinear_reaction_diffusion_2d"),
        )
        return RunTAPSBackendOutput(result=result)

    if family in GENERIC_TAPS_FAMILIES and space_axis_count == 1:
        artifacts, residual_report = solve_scalar_elliptic_1d(input.taps_problem)
        artifact_refs = []
        artifact_refs.extend(artifacts.factor_matrices)
        if artifacts.reconstruction_metadata is not None:
            artifact_refs.append(artifacts.reconstruction_metadata)
        if artifacts.residual_history is not None:
            artifact_refs.append(artifacts.residual_history)
        result = SolverResult(
            id=f"result:{input.taps_problem.id}",
            problem_id=input.problem.id,
            backend=f"taps:generic_scalar_elliptic_1d:{family}",
            status="success" if residual_report.converged else "needs_review",
            scalar_outputs={
                "message": "Generic TAPS scalar 1D weak-form assembler executed.",
                "equation_family": family,
                "tensor_rank": input.taps_problem.basis.tensor_rank,
                **residual_report.residuals,
            },
            residuals=residual_report.residuals,
            artifacts=artifact_refs,
            provenance=Provenance(created_by="run_taps_backend", source="taps_generic_scalar_elliptic_1d"),
        )
        return RunTAPSBackendOutput(result=result)

    if family in GENERIC_TAPS_FAMILIES and space_axis_count == 2:
        artifacts, residual_report = solve_scalar_elliptic_2d(input.taps_problem)
        artifact_refs = []
        artifact_refs.extend(artifacts.factor_matrices)
        if artifacts.reconstruction_metadata is not None:
            artifact_refs.append(artifacts.reconstruction_metadata)
        if artifacts.residual_history is not None:
            artifact_refs.append(artifacts.residual_history)
        result = SolverResult(
            id=f"result:{input.taps_problem.id}",
            problem_id=input.problem.id,
            backend=f"taps:generic_scalar_elliptic_2d:{family}",
            status="success" if residual_report.converged else "needs_review",
            scalar_outputs={
                "message": "Generic TAPS scalar 2D tensorized weak-form assembler executed.",
                "equation_family": family,
                "tensor_rank": residual_report.rank,
                **residual_report.residuals,
            },
            residuals=residual_report.residuals,
            artifacts=artifact_refs,
            provenance=Provenance(created_by="run_taps_backend", source="taps_generic_scalar_elliptic_2d"),
        )
        return RunTAPSBackendOutput(result=result)

    result = SolverResult(
        id=f"result:{input.taps_problem.id}",
        problem_id=input.problem.id,
        backend="taps",
        status="needs_review",
        scalar_outputs={
            "message": "TAPSProblem compiled, but no executable TAPS kernel is connected for this weak-form family and axis layout yet.",
            "equation_family": family or "unknown",
            "space_axis_count": space_axis_count,
            "tensor_rank": input.taps_problem.basis.tensor_rank,
        },
        artifacts=[
            ArtifactRef(uri=f"scratch/{input.problem.id.replace(':', '_')}/taps_problem.json", kind="taps_problem", format="json")
        ],
        provenance=Provenance(created_by="run_taps_backend", source="scaffold"),
    )
    return RunTAPSBackendOutput(result=result)


class EstimateTAPSResidualInput(StrictBaseModel):
    problem: PhysicsProblem
    taps_problem: TAPSProblem
    result: SolverResult


class EstimateTAPSResidualOutput(StrictBaseModel):
    report: TAPSResidualReport


def estimate_taps_residual(input: EstimateTAPSResidualInput) -> EstimateTAPSResidualOutput:
    """Estimate TAPS residual."""
    if input.result.residuals:
        if "normalized_coupled_residual" in input.result.residuals:
            normalized = input.result.residuals["normalized_coupled_residual"]
            update = input.result.residuals.get("relative_update", 1.0)
            converged = normalized < 1e-6 or update < input.taps_problem.nonlinear.tolerance
            recommended_action = "accept" if converged else "refine_axes"
        elif "normalized_fem_residual" in input.result.residuals:
            normalized = input.result.residuals["normalized_fem_residual"]
            update = input.result.residuals.get("relative_update", 1.0)
            converged = normalized < 1e-8 or update < 1e-10
            recommended_action = "accept" if converged else "refine_axes"
        elif "normalized_nonlinear_residual" in input.result.residuals:
            normalized = input.result.residuals["normalized_nonlinear_residual"]
            update = input.result.residuals.get("relative_update", 1.0)
            converged = normalized < 1e-6 or update < input.taps_problem.nonlinear.tolerance
            recommended_action = "accept" if converged else "refine_axes"
        elif "normalized_linear_residual" in input.result.residuals:
            normalized = input.result.residuals["normalized_linear_residual"]
            tolerance = 1e-8 if input.result.residuals.get("masked_relaxation_iterations", 0.0) else 1e-10
            converged = normalized < tolerance
            recommended_action = "accept" if converged else "refine_axes"
        elif "normalized_graph_residual" in input.result.residuals:
            normalized = input.result.residuals["normalized_graph_residual"]
            update = input.result.residuals.get("relative_update", 1.0)
            converged = normalized < 1e-8 or update < 1e-10
            recommended_action = "accept" if converged else "refine_axes"
        else:
            normalized = input.result.residuals.get("normalized_pde_residual", 1.0)
            reconstruction = input.result.residuals.get("relative_l2_reconstruction_error", 1.0)
            converged = normalized < 1e-2 and reconstruction < 1e-3
            recommended_action = "accept" if converged else "increase_rank"
        return EstimateTAPSResidualOutput(
            report=TAPSResidualReport(
                residuals=input.result.residuals,
                rank=input.taps_problem.basis.tensor_rank,
                converged=converged,
                recommended_action=recommended_action,  # type: ignore[arg-type]
            )
        )
    return EstimateTAPSResidualOutput(
        report=TAPSResidualReport(
            residuals={},
            rank=input.taps_problem.basis.tensor_rank,
            converged=False,
            recommended_action="fallback",
        )
    )


for _tool, _input, _output in [
    (estimate_taps_support, EstimateTAPSSupportInput, EstimateTAPSSupportOutput),
    (formulate_taps_equation, FormulateTAPSEquationInput, FormulateTAPSEquationOutput),
    (build_taps_problem, BuildTAPSProblemInput, BuildTAPSProblemOutput),
    (author_taps_runtime_extension, AuthorTAPSRuntimeExtensionInput, AuthorTAPSRuntimeExtensionOutput),
    (run_taps_backend, RunTAPSBackendInput, RunTAPSBackendOutput),
    (estimate_taps_residual, EstimateTAPSResidualInput, EstimateTAPSResidualOutput),
]:
    _tool.input_model = _input
    _tool.output_model = _output
    _tool.side_effects = "workspace artifacts only"
    _tool.requires_approval = False
