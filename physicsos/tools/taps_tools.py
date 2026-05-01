from __future__ import annotations

import json

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
from physicsos.backends.taps_generic import supports_coupled_reaction_diffusion_weak_form
from physicsos.backends.taps_generic import supports_hcurl_curl_curl_weak_form
from physicsos.backends.taps_generic import supports_nonlinear_reaction_diffusion_weak_form
from physicsos.backends.taps_generic import supports_scalar_elliptic_weak_form
from physicsos.backends.taps_generic import supports_transient_diffusion_weak_form
from physicsos.backends.taps_generic import supports_vector_elasticity_weak_form
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


def _compile_strong_form_term(expression: str, fields: list[str]) -> tuple[str, str, list[str]]:
    lowered = expression.lower()
    field = fields[0] if fields else "u"
    test = f"v_{field}"
    if any(token in lowered for token in ("d/dt", "dt", "partial_t", "time_derivative")):
        return "time_derivative", f"int_Omega {test} d({field})/dt dOmega", []
    if any(token in lowered for token in ("laplacian", "div(", "grad(", "d2", "diffusion")):
        return "diffusion", f"int_Omega grad({test}) dot k grad({field}) dOmega", ["k"]
    if any(token in lowered for token in ("u^3", "cubic", "nonlinear", "r(")):
        return "reaction", f"- int_Omega {test} R({field}) dOmega", ["reaction_parameters"]
    if "curl" in lowered:
        return "diffusion", f"int_Omega curl({test}) dot curl({field}) dOmega", []
    return "custom", expression, []


def _boundary_weak_terms(problem: PhysicsProblem) -> list[TAPSEquationTerm]:
    terms: list[TAPSEquationTerm] = []
    for boundary in problem.boundary_conditions:
        kind = boundary.kind.lower()
        field = boundary.field
        if kind == "dirichlet":
            continue
        role = "boundary"
        if kind == "neumann":
            expression = f"int_Gamma v_{field} g dGamma on {boundary.region_id}"
            coefficients = ["neumann_flux"]
        elif kind == "robin":
            expression = f"int_Gamma h v_{field} {field} dGamma - int_Gamma v_{field} r dGamma on {boundary.region_id}"
            coefficients = ["robin_h", "robin_r"]
        elif kind in {"interface", "periodic"}:
            expression = f"interface constraint for {field} on {boundary.region_id}"
            coefficients = []
        else:
            expression = f"boundary weak term for {kind} {field} on {boundary.region_id}"
            coefficients = []
        terms.append(
            TAPSEquationTerm(
                id=f"{boundary.id}:weak_boundary",
                role=role,
                expression=expression,
                fields=[field],
                coefficients=coefficients,
                integration_domain=boundary.region_id,
                assumptions=["Boundary term compiled as IR metadata; executable kernels may still use backend-specific boundary handling."],
            )
        )
    return terms


def _weak_form_from_problem(problem: PhysicsProblem, knowledge_context: KnowledgeContext | None) -> TAPSWeakFormSpec | None:
    if not problem.operators or not problem.fields:
        return None
    family = problem.operators[0].equation_class.lower()
    fields = [field.name for field in problem.fields]
    terms: list[TAPSEquationTerm] = []
    boundary_terms: list[TAPSEquationTerm] = _boundary_weak_terms(problem)
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
                role = "custom"
                coefficients: list[str] = []
                if operator.form == "strong":
                    role, expression, coefficients = _compile_strong_form_term(expression, term_fields or operator.fields_out or fields)
                terms.append(
                    TAPSEquationTerm(
                        id=f"{operator.id}:term:{index}",
                        role=role,  # type: ignore[arg-type]
                        expression=expression,
                        fields=term_fields or operator.fields_out or fields,
                        coefficients=coefficients,
                        assumptions=["Compiled from strong-form token pattern; knowledge-agent review is recommended for high-trust use."]
                        if operator.form == "strong"
                        else [],
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
    residual = " + ".join(term.expression for term in [*terms, *boundary_terms])
    return TAPSWeakFormSpec(
        family=family,
        strong_form="; ".join(f"{operator.name}:{operator.equation_class}" for operator in problem.operators),
        trial_fields=fields,
        test_functions=[f"v_{field}" for field in fields],
        terms=terms,
        boundary_terms=boundary_terms,
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


class ValidateTAPSIRInput(StrictBaseModel):
    problem: PhysicsProblem
    taps_problem: TAPSProblem


class ValidateTAPSIROutput(StrictBaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    checks: list[dict[str, object]] = Field(default_factory=list)
    fallback_recommended: bool = False
    recommended_action: str = "run_taps_backend"
    artifact: ArtifactRef


def validate_taps_ir(input: ValidateTAPSIRInput) -> ValidateTAPSIROutput:
    """Validate TAPS weak-form IR before execution or backend export."""
    errors: list[str] = []
    warnings: list[str] = []
    checks: list[dict[str, object]] = []
    weak_form = input.taps_problem.weak_form
    if weak_form is None:
        errors.append("missing weak_form IR")
        checks.append({"name": "weak_form_present", "passes": False})
    else:
        checks.append({"name": "weak_form_present", "passes": True, "family": weak_form.family})
        if not weak_form.trial_fields:
            errors.append("weak_form has no trial_fields")
        checks.append({"name": "trial_fields_declared", "passes": bool(weak_form.trial_fields), "trial_fields": weak_form.trial_fields})
        field_names = {field.name for field in input.problem.fields}
        missing_fields = [field for field in weak_form.trial_fields if field not in field_names]
        if missing_fields:
            errors.append(f"weak_form trial_fields not declared in PhysicsProblem.fields: {missing_fields}")
        checks.append({"name": "trial_fields_match_problem_fields", "passes": not missing_fields, "missing_fields": missing_fields})
        if not weak_form.terms:
            errors.append("weak_form has no equation terms")
        checks.append({"name": "equation_terms_present", "passes": bool(weak_form.terms), "term_count": len(weak_form.terms)})
        if any(term.role == "time_derivative" for term in weak_form.terms) and not input.problem.initial_conditions:
            errors.append("time_derivative term requires at least one InitialConditionSpec")
        checks.append(
            {
                "name": "time_derivative_has_initial_condition",
                "passes": not any(term.role == "time_derivative" for term in weak_form.terms) or bool(input.problem.initial_conditions),
            }
        )
    coefficient_names = {coefficient.name.lower() for coefficient in input.taps_problem.coefficients}
    required_coefficients = {
        coefficient.lower()
        for term in (weak_form.terms if weak_form is not None else [])
        for coefficient in term.coefficients
    }
    missing_coefficients = sorted(required_coefficients - coefficient_names)
    if missing_coefficients:
        warnings.append(f"coefficients referenced by weak_form but not provided explicitly: {missing_coefficients}")
    checks.append({"name": "referenced_coefficients_available", "passes": not missing_coefficients, "missing_coefficients": missing_coefficients})
    has_mesh_graph = any(encoding.kind == "mesh_graph" for encoding in input.taps_problem.geometry_encodings)
    if supports_vector_elasticity_weak_form(input.taps_problem) and not has_mesh_graph:
        errors.append("vector elasticity IR requires mesh_graph geometry encoding")
    if supports_hcurl_curl_curl_weak_form(input.taps_problem) and not has_mesh_graph:
        errors.append("H(curl) curl-curl IR requires mesh_graph geometry encoding")
    supported = any(
        [
            supports_transient_diffusion_weak_form(input.taps_problem),
            supports_coupled_reaction_diffusion_weak_form(input.taps_problem),
            supports_hcurl_curl_curl_weak_form(input.taps_problem),
            supports_nonlinear_reaction_diffusion_weak_form(input.taps_problem),
            supports_scalar_elliptic_weak_form(input.taps_problem),
            supports_vector_elasticity_weak_form(input.taps_problem),
            (weak_form.family.lower() if weak_form is not None else "") in GENERIC_TAPS_FAMILIES,
        ]
    )
    if not supported:
        warnings.append("no executable TAPS IR block mapping is currently connected for this weak_form")
    checks.append({"name": "executable_block_mapping_connected", "passes": supported})
    fallback_recommended = bool(errors) or not supported
    recommended_action = "run_taps_backend"
    if errors:
        recommended_action = "ask_user_or_knowledge_agent"
    elif not supported:
        recommended_action = "author_runtime_extension_or_export_full_solver"
    output_dir = project_root() / "scratch" / input.problem.id.replace(":", "_") / "taps_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "taps_ir_validation.json"
    path.write_text(
        json.dumps(
            {
                "valid": not errors,
                "errors": errors,
                "warnings": warnings,
                "checks": checks,
                "fallback_recommended": fallback_recommended,
                "recommended_action": recommended_action,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return ValidateTAPSIROutput(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        checks=checks,
        fallback_recommended=fallback_recommended,
        recommended_action=recommended_action,
        artifact=ArtifactRef(uri=str(path), kind="taps_ir_validation", format="json"),
    )


class ExportTAPSBackendBridgeInput(StrictBaseModel):
    problem: PhysicsProblem
    taps_problem: TAPSProblem
    backend: str = "fenicsx"


class ExportTAPSBackendBridgeOutput(StrictBaseModel):
    manifest: ArtifactRef
    draft_artifact: ArtifactRef | None = None
    warnings: list[str] = Field(default_factory=list)


def _backend_bridge_draft(backend: str, blocks: list[dict[str, object]], weak_form: TAPSWeakFormSpec | None) -> tuple[str, str]:
    families = {str(block.get("family")) for block in blocks}
    if backend == "fenicsx":
        lines = [
            "# Draft FEniCSx bridge generated from TAPS IR. Review before execution.",
            "from dolfinx import fem",
            "import ufl",
            "",
            "# TODO: load mesh/facet tags from PhysicsOS backend mesh export manifest.",
            "# TODO: bind coefficients and boundary terms from TAPSProblem.",
        ]
        if "hcurl_curl_curl" in families:
            lines.append("# Use Nedelec element: element = ufl.FiniteElement('N1curl', cell, degree)")
        elif "vector_elasticity" in families:
            lines.append("# Use vector H1 space and form: inner(eps(v), C*eps(u))*dx")
        elif "coupled_reaction_diffusion" in families:
            lines.append("# Use mixed H1 space and block/mixed nonlinear form.")
        elif "transient_diffusion" in families:
            lines.append("# Use H1 space with implicit Euler or Crank-Nicolson time stepping.")
            lines.append("# Example residual: (u-u_n)/dt*v*dx + k*dot(grad(u), grad(v))*dx")
        else:
            lines.append("# Use H1 space and scalar elliptic form: k*dot(grad(u), grad(v))*dx")
        return "py", "\n".join(lines) + "\n"
    if backend == "mfem":
        return "py", "\n".join(
            [
                "# Draft PyMFEM bridge generated from TAPS IR. Review before execution.",
                "# TODO: load MFEM mesh converted from PhysicsOS mesh export manifest.",
                "# TODO: choose H1, ND, or mixed finite element collection from blocks.",
                f"# Blocks: {[block.get('family') for block in blocks]}",
            ]
        ) + "\n"
    if backend == "petsc":
        return "py", "\n".join(
            [
                "# Draft petsc4py bridge generated from TAPS IR. Review before execution.",
                "# TODO: assemble Mat/Vec from exported TAPS block operators or backend FEM code.",
                f"# Trial fields: {weak_form.trial_fields if weak_form is not None else []}",
            ]
        ) + "\n"
    return "md", "# Generic TAPS backend bridge draft\n\nReview IR blocks and map them to a trusted backend manually.\n"


def export_taps_backend_bridge(input: ExportTAPSBackendBridgeInput) -> ExportTAPSBackendBridgeOutput:
    """Export a safe manifest describing how TAPS IR maps to a real PDE backend.

    This does not execute external solvers or generate trusted production code.
    It records weak-form blocks, required function-space hints, and fallback
    targets for a reviewed FEniCSx/MFEM/PETSc implementation.
    """
    validation = validate_taps_ir(ValidateTAPSIRInput(problem=input.problem, taps_problem=input.taps_problem))
    backend = input.backend.lower()
    weak_form = input.taps_problem.weak_form
    blocks: list[dict[str, object]] = []
    if supports_transient_diffusion_weak_form(input.taps_problem):
        blocks.append({"family": "transient_diffusion", "space": "H1", "time_integrator": "implicit_euler_or_crank_nicolson"})
    if supports_scalar_elliptic_weak_form(input.taps_problem):
        blocks.append({"family": "scalar_elliptic", "space": "H1"})
    if supports_vector_elasticity_weak_form(input.taps_problem):
        blocks.append({"family": "vector_elasticity", "space": "vector_H1"})
    if supports_hcurl_curl_curl_weak_form(input.taps_problem):
        blocks.append({"family": "hcurl_curl_curl", "space": "Nedelec_Hcurl"})
    if supports_nonlinear_reaction_diffusion_weak_form(input.taps_problem):
        blocks.append({"family": "nonlinear_reaction_diffusion", "space": "H1", "nonlinear_solver": "Picard_or_Newton"})
    if supports_coupled_reaction_diffusion_weak_form(input.taps_problem):
        blocks.append({"family": "coupled_reaction_diffusion", "space": "mixed_H1", "block_solver": "monolithic_or_block_gauss_seidel"})
    warnings = list(validation.warnings)
    if validation.errors:
        warnings.extend(validation.errors)
    if backend not in {"fenicsx", "mfem", "petsc", "generic"}:
        warnings.append(f"backend={input.backend!r} is not a known bridge target; manifest emitted as generic guidance")
    output_dir = project_root() / "scratch" / input.problem.id.replace(":", "_") / "taps_backend_bridge"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{backend}_bridge_manifest.json"
    draft_format, draft_text = _backend_bridge_draft(backend, blocks, weak_form)
    draft_path = output_dir / f"{backend}_bridge_draft.{draft_format}"
    payload = {
        "schema_version": "physicsos.taps_backend_bridge.v1",
        "problem_id": input.problem.id,
        "taps_problem_id": input.taps_problem.id,
        "target_backend": backend,
        "weak_form_family": weak_form.family if weak_form is not None else None,
        "trial_fields": weak_form.trial_fields if weak_form is not None else [],
        "blocks": blocks,
        "validation_artifact": validation.artifact.uri,
        "fallback_policy": {
            "execute_external_solver": False,
            "requires_review": True,
            "recommended_action": validation.recommended_action,
        },
        "warnings": warnings,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    draft_path.write_text(draft_text, encoding="utf-8")
    return ExportTAPSBackendBridgeOutput(
        manifest=ArtifactRef(uri=str(path), kind="taps_backend_bridge_manifest", format="json"),
        draft_artifact=ArtifactRef(uri=str(draft_path), kind="taps_backend_bridge_draft", format=draft_format),
        warnings=warnings,
    )


class PlanTAPSAdaptiveFallbackInput(StrictBaseModel):
    problem: PhysicsProblem
    taps_problem: TAPSProblem
    preferred_backend: str = "fenicsx"


class PlanTAPSAdaptiveFallbackOutput(StrictBaseModel):
    decision: dict[str, object]
    artifact: ArtifactRef


def _dependency_checks_for_backend(backend: str) -> list[dict[str, object]]:
    if backend == "fenicsx":
        return [
            {"name": "python_import_dolfinx", "command": "python -c \"import dolfinx, ufl\"", "required": True},
            {"name": "meshio_available", "command": "python -c \"import meshio\"", "required": True},
        ]
    if backend == "mfem":
        return [
            {"name": "python_import_mfem", "command": "python -c \"import mfem.ser\"", "required": True},
            {"name": "meshio_available", "command": "python -c \"import meshio\"", "required": True},
        ]
    if backend == "petsc":
        return [{"name": "python_import_petsc4py", "command": "python -c \"import petsc4py\"", "required": True}]
    return [{"name": "python_available", "command": "python --version", "required": True}]


def plan_taps_adaptive_fallback(input: PlanTAPSAdaptiveFallbackInput) -> PlanTAPSAdaptiveFallbackOutput:
    """Plan the next safe action when TAPS IR is incomplete, unsafe, or unsupported."""
    validation = validate_taps_ir(ValidateTAPSIRInput(problem=input.problem, taps_problem=input.taps_problem))
    if not validation.fallback_recommended:
        mode = "run_taps_backend"
        reason = "TAPS IR passed readiness checks and has an executable block mapping."
    elif validation.errors:
        mode = "ask_knowledge_agent"
        reason = "TAPS IR has blocking validation errors that should be resolved before execution."
    else:
        mode = "export_backend_bridge"
        reason = "TAPS IR compiled but lacks a connected local executable mapping; export reviewed backend bridge."
    decision: dict[str, object] = {
        "mode": mode,
        "reason": reason,
        "validation_artifact": validation.artifact.uri,
        "preferred_backend": input.preferred_backend,
        "allowed_actions": ["ask_knowledge_agent", "author_runtime_extension", "export_backend_bridge", "prepare_full_solver_case"],
        "execute_external_solver": False,
    }
    output_dir = project_root() / "scratch" / input.problem.id.replace(":", "_") / "taps_fallback"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "adaptive_fallback_decision.json"
    path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return PlanTAPSAdaptiveFallbackOutput(
        decision=decision,
        artifact=ArtifactRef(uri=str(path), kind="taps_adaptive_fallback_decision", format="json"),
    )


class PrepareTAPSBackendCaseBundleInput(StrictBaseModel):
    problem: PhysicsProblem
    taps_problem: TAPSProblem
    backend: str = "fenicsx"
    mesh_export_manifest: ArtifactRef | None = None


class PrepareTAPSBackendCaseBundleOutput(StrictBaseModel):
    bundle: ArtifactRef
    bridge_manifest: ArtifactRef
    draft_artifact: ArtifactRef | None = None
    warnings: list[str] = Field(default_factory=list)


def prepare_taps_backend_case_bundle(input: PrepareTAPSBackendCaseBundleInput) -> PrepareTAPSBackendCaseBundleOutput:
    """Prepare a reviewed backend execution bundle without running external solvers."""
    bridge = export_taps_backend_bridge(
        ExportTAPSBackendBridgeInput(problem=input.problem, taps_problem=input.taps_problem, backend=input.backend)
    )
    fallback = plan_taps_adaptive_fallback(
        PlanTAPSAdaptiveFallbackInput(problem=input.problem, taps_problem=input.taps_problem, preferred_backend=input.backend)
    )
    backend = input.backend.lower()
    warnings = list(bridge.warnings)
    mesh_requirement = {
        "required": backend in {"fenicsx", "mfem"},
        "provided": input.mesh_export_manifest is not None,
        "manifest_uri": input.mesh_export_manifest.uri if input.mesh_export_manifest is not None else None,
        "expected_kind": "backend_mesh_export_manifest",
    }
    if mesh_requirement["required"] and not mesh_requirement["provided"]:
        warnings.append("backend case bundle requires a backend mesh export manifest before execution")
    bundle_payload = {
        "schema_version": "physicsos.taps_backend_case_bundle.v1",
        "problem_id": input.problem.id,
        "taps_problem_id": input.taps_problem.id,
        "backend": backend,
        "bridge_manifest": bridge.manifest.uri,
        "draft_artifact": bridge.draft_artifact.uri if bridge.draft_artifact is not None else None,
        "fallback_decision": fallback.artifact.uri,
        "dependency_checks": _dependency_checks_for_backend(backend),
        "mesh_export": mesh_requirement,
        "coefficient_binding": [{"name": coefficient.name, "role": coefficient.role, "region_ids": coefficient.region_ids} for coefficient in input.taps_problem.coefficients],
        "boundary_binding": [
            {"id": boundary.id, "field": boundary.field, "kind": boundary.kind, "region_id": boundary.region_id}
            for boundary in input.taps_problem.boundary_conditions
        ],
        "approval_gate": {
            "execute_external_solver": False,
            "requires_user_approval": True,
            "requires_dependency_checks": True,
            "requires_mesh_export_manifest": bool(mesh_requirement["required"]),
        },
        "warnings": warnings,
    }
    output_dir = project_root() / "scratch" / input.problem.id.replace(":", "_") / "taps_backend_bridge"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{backend}_case_bundle.json"
    path.write_text(json.dumps(bundle_payload, indent=2), encoding="utf-8")
    return PrepareTAPSBackendCaseBundleOutput(
        bundle=ArtifactRef(uri=str(path), kind="taps_backend_case_bundle", format="json"),
        bridge_manifest=bridge.manifest,
        draft_artifact=bridge.draft_artifact,
        warnings=warnings,
    )


def run_taps_backend(input: RunTAPSBackendInput) -> RunTAPSBackendOutput:
    """Run the TAPS backend."""
    operator_classes = {operator.equation_class.lower() for operator in input.problem.operators}
    is_heat = bool(operator_classes.intersection({"heat", "diffusion", "thermal_diffusion"}))
    is_transient_diffusion_ir = supports_transient_diffusion_weak_form(input.taps_problem)
    if (is_heat or is_transient_diffusion_ir) and input.problem.initial_conditions:
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
            backend="taps:thermal_1d" if is_heat else "taps:weak_ir_transient_diffusion_1d:custom",
            status="success" if residual_report.converged else "needs_review",
            scalar_outputs={
                "message": "TAPS thermal MVP backend executed.",
                "weak_form_ir_blocks": is_transient_diffusion_ir,
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
    is_coupled_reaction_diffusion_ir = supports_coupled_reaction_diffusion_weak_form(input.taps_problem)
    is_hcurl_curl_curl_ir = supports_hcurl_curl_curl_weak_form(input.taps_problem)
    is_nonlinear_reaction_diffusion_ir = supports_nonlinear_reaction_diffusion_weak_form(input.taps_problem)
    is_scalar_elliptic_ir = supports_scalar_elliptic_weak_form(input.taps_problem)
    is_vector_elasticity_ir = supports_vector_elasticity_weak_form(input.taps_problem)
    if (family in {"maxwell", "curl_curl", "electromagnetic"} or is_hcurl_curl_curl_ir) and has_mesh_graph:
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
            backend=f"taps:{'weak_ir' if family not in {'maxwell', 'curl_curl', 'electromagnetic'} else 'mesh_fem'}_em_curl_curl:{family}",
            status="success" if residual_report.converged else "needs_review",
            scalar_outputs={
                "message": "Triangle Nedelec H(curl) EM curl-curl TAPS kernel executed from reusable curl/mass/source blocks.",
                "equation_family": family,
                "weak_form_ir_blocks": is_hcurl_curl_curl_ir,
                **residual_report.residuals,
            },
            residuals=residual_report.residuals,
            artifacts=artifact_refs,
            provenance=Provenance(created_by="run_taps_backend", source="taps_mesh_fem_em_curl_curl"),
        )
        return RunTAPSBackendOutput(result=result)

    if (family in {"elasticity", "linear_elasticity"} or is_vector_elasticity_ir) and has_mesh_graph:
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
            backend=f"taps:{'weak_ir' if family not in {'elasticity', 'linear_elasticity'} else 'mesh_fem'}_linear_elasticity:{family}",
            status="success" if residual_report.converged else "needs_review",
            scalar_outputs={
                "message": "Triangle vector FEM-like TAPS linear-elasticity kernel executed from reusable strain/body-force blocks.",
                "equation_family": family,
                "weak_form_ir_blocks": is_vector_elasticity_ir,
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

    if (family == "coupled_reaction_diffusion" or is_coupled_reaction_diffusion_ir) and space_axis_count == 2 and field_count >= 2:
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
            backend="taps:coupled_reaction_diffusion_2d" if family == "coupled_reaction_diffusion" else f"taps:weak_ir_coupled_reaction_diffusion_2d:{family}",
            status="success" if residual_report.converged else "needs_review",
            scalar_outputs={
                "message": "Coupled-field TAPS 2D reaction-diffusion fixed-point kernel executed.",
                "equation_family": family,
                "weak_form_ir_blocks": is_coupled_reaction_diffusion_ir,
                "field_count": field_count,
                "tensor_rank": residual_report.rank,
                **residual_report.residuals,
            },
            residuals=residual_report.residuals,
            artifacts=artifact_refs,
            provenance=Provenance(created_by="run_taps_backend", source="taps_coupled_reaction_diffusion_2d"),
        )
        return RunTAPSBackendOutput(result=result)

    if (family == "reaction_diffusion" or is_nonlinear_reaction_diffusion_ir) and space_axis_count == 1:
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
            backend="taps:nonlinear_reaction_diffusion_1d" if family == "reaction_diffusion" else f"taps:weak_ir_nonlinear_reaction_diffusion_1d:{family}",
            status="success" if residual_report.converged else "needs_review",
            scalar_outputs={
                "message": "Nonlinear TAPS 1D reaction-diffusion Picard kernel executed.",
                "equation_family": family,
                "weak_form_ir_blocks": is_nonlinear_reaction_diffusion_ir,
                "tensor_rank": residual_report.rank,
                **residual_report.residuals,
            },
            residuals=residual_report.residuals,
            artifacts=artifact_refs,
            provenance=Provenance(created_by="run_taps_backend", source="taps_nonlinear_reaction_diffusion_1d"),
        )
        return RunTAPSBackendOutput(result=result)

    if (family == "reaction_diffusion" or is_nonlinear_reaction_diffusion_ir) and space_axis_count == 2:
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
            backend="taps:nonlinear_reaction_diffusion_2d" if family == "reaction_diffusion" else f"taps:weak_ir_nonlinear_reaction_diffusion_2d:{family}",
            status="success" if residual_report.converged else "needs_review",
            scalar_outputs={
                "message": "Nonlinear TAPS 2D reaction-diffusion fixed-point kernel executed.",
                "equation_family": family,
                "weak_form_ir_blocks": is_nonlinear_reaction_diffusion_ir,
                "tensor_rank": residual_report.rank,
                **residual_report.residuals,
            },
            residuals=residual_report.residuals,
            artifacts=artifact_refs,
            provenance=Provenance(created_by="run_taps_backend", source="taps_nonlinear_reaction_diffusion_2d"),
        )
        return RunTAPSBackendOutput(result=result)

    if (family in GENERIC_TAPS_FAMILIES or is_scalar_elliptic_ir) and space_axis_count == 1:
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
            backend=f"taps:{'weak_ir' if family not in GENERIC_TAPS_FAMILIES else 'generic'}_scalar_elliptic_1d:{family}",
            status="success" if residual_report.converged else "needs_review",
            scalar_outputs={
                "message": "TAPS scalar 1D weak-form assembler executed from reusable diffusion/reaction/source blocks.",
                "equation_family": family,
                "weak_form_ir_blocks": is_scalar_elliptic_ir,
                "tensor_rank": input.taps_problem.basis.tensor_rank,
                **residual_report.residuals,
            },
            residuals=residual_report.residuals,
            artifacts=artifact_refs,
            provenance=Provenance(created_by="run_taps_backend", source="taps_generic_scalar_elliptic_1d"),
        )
        return RunTAPSBackendOutput(result=result)

    if (family in GENERIC_TAPS_FAMILIES or is_scalar_elliptic_ir) and space_axis_count == 2:
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
            backend=f"taps:{'weak_ir' if family not in GENERIC_TAPS_FAMILIES else 'generic'}_scalar_elliptic_2d:{family}",
            status="success" if residual_report.converged else "needs_review",
            scalar_outputs={
                "message": "TAPS scalar 2D tensorized weak-form assembler executed from reusable diffusion/reaction/source blocks.",
                "equation_family": family,
                "weak_form_ir_blocks": is_scalar_elliptic_ir,
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
    (validate_taps_ir, ValidateTAPSIRInput, ValidateTAPSIROutput),
    (export_taps_backend_bridge, ExportTAPSBackendBridgeInput, ExportTAPSBackendBridgeOutput),
    (plan_taps_adaptive_fallback, PlanTAPSAdaptiveFallbackInput, PlanTAPSAdaptiveFallbackOutput),
    (prepare_taps_backend_case_bundle, PrepareTAPSBackendCaseBundleInput, PrepareTAPSBackendCaseBundleOutput),
    (run_taps_backend, RunTAPSBackendInput, RunTAPSBackendOutput),
    (estimate_taps_residual, EstimateTAPSResidualInput, EstimateTAPSResidualOutput),
]:
    _tool.input_model = _input
    _tool.output_model = _output
    _tool.side_effects = "workspace artifacts only"
    _tool.requires_approval = False
