from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from pydantic import Field

from physicsos.agents.structured import CoreAgentLLMConfig, StructuredLLMClient, call_structured_agent
from physicsos.config import project_root
from physicsos.backends.taps_generic import SUPPORTED_FAMILIES as GENERIC_TAPS_FAMILIES
from physicsos.backends.taps_generic import solve_coupled_reaction_diffusion_2d
from physicsos.backends.taps_generic import solve_graph_poisson
from physicsos.backends.taps_generic import solve_mesh_fem_em_curl_curl
from physicsos.backends.taps_generic import solve_mesh_fem_hdiv_div
from physicsos.backends.taps_generic import solve_mesh_fem_linear_elasticity
from physicsos.backends.taps_generic import solve_mesh_fem_poisson
from physicsos.backends.taps_generic import solve_navier_stokes_channel_2d
from physicsos.backends.taps_generic import solve_reaction_diffusion_nonlinear_1d
from physicsos.backends.taps_generic import solve_reaction_diffusion_nonlinear_2d
from physicsos.backends.taps_generic import solve_scalar_elliptic_1d
from physicsos.backends.taps_generic import solve_scalar_elliptic_2d
from physicsos.backends.taps_generic import solve_scalar_elliptic_3d
from physicsos.backends.taps_generic import solve_oseen_channel_2d
from physicsos.backends.taps_generic import solve_stokes_channel_2d
from physicsos.backends.taps_generic import supports_coupled_reaction_diffusion_weak_form
from physicsos.backends.taps_generic import supports_hcurl_curl_curl_weak_form
from physicsos.backends.taps_generic import supports_hdiv_div_weak_form
from physicsos.backends.taps_generic import supports_mesh_navier_stokes_bridge
from physicsos.backends.taps_generic import supports_nonlinear_reaction_diffusion_weak_form
from physicsos.backends.taps_generic import supports_navier_stokes_weak_form
from physicsos.backends.taps_generic import supports_oseen_weak_form
from physicsos.backends.taps_generic import supports_scalar_elliptic_weak_form
from physicsos.backends.taps_generic import supports_stokes_weak_form
from physicsos.backends.taps_generic import supports_transient_diffusion_weak_form
from physicsos.backends.taps_generic import supports_vector_elasticity_weak_form
from physicsos.backends.taps_thermal import solve_transient_heat_1d
from physicsos.schemas.common import ArtifactRef, ComputeBudget, Provenance, StrictBaseModel
from physicsos.schemas.contracts import PhysicsProblemContract
from physicsos.schemas.knowledge import KnowledgeContext
from physicsos.schemas.problem import PhysicsProblem
from physicsos.schemas.solver import SolverResult
from physicsos.schemas.taps import (
    BackendPreparationPlanOutput,
    NumericalBoundaryConditionBinding,
    NumericalCoefficientBinding,
    NumericalDiscretizationSpec,
    NumericalSolvePlanOutput,
    NumericalSourceBinding,
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


def _has_laminar_channel_inputs(problem: PhysicsProblem) -> bool:
    boundary_kinds = {boundary.kind.lower() for boundary in problem.boundary_conditions}
    has_channel_boundaries = "wall" in boundary_kinds and bool(boundary_kinds & {"inlet", "dirichlet"}) and bool(boundary_kinds & {"outlet", "neumann"})
    material_properties = {
        prop.name.lower()
        for material in problem.materials
        for prop in material.properties
    }
    has_coefficients = bool(material_properties & {"mu", "dynamic_viscosity", "viscosity"}) and bool(material_properties & {"rho", "density"}) and bool(
        material_properties & {"pressure_drop", "delta_p", "dp"}
    )
    reynolds = [
        number.value
        for operator in problem.operators
        for number in operator.nondimensional_numbers
        if number.name.lower() in {"re", "reynolds", "reynolds_number"}
    ]
    return has_channel_boundaries and has_coefficients and (not reynolds or max(reynolds) <= 100.0)


def estimate_taps_support(input: EstimateTAPSSupportInput) -> EstimateTAPSSupportOutput:
    """Estimate whether TAPS should be the first solver for this PhysicsProblem."""
    problem = input.problem
    reasons: list[str] = []
    risks: list[str] = []
    score = 0.0

    if problem.domain in {"thermal", "solid", "acoustic", "custom", "multiphysics"}:
        score += 0.25
        reasons.append(f"domain {problem.domain} is compatible with first TAPS targets")
    elif problem.domain == "fluid":
        score += 0.20
        reasons.append("fluid domain is compatible with the low-Re Stokes TAPS target")
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
        "stokes",
        "oseen",
    }
    if operator_classes.intersection(preferred):
        score += 0.35
        reasons.append("operator family is a TAPS-first target")
    elif "navier_stokes" in operator_classes:
        reynolds = [
            number.value
            for operator in problem.operators
            for number in operator.nondimensional_numbers
            if number.name.lower() in {"re", "reynolds", "reynolds_number"}
        ]
        has_frozen_velocity = any(
            prop.name.lower()
            in {"frozen_velocity", "convective_velocity", "oseen_velocity", "linearization_velocity", "ubar", "u_bar"}
            for material in problem.materials
            for prop in material.properties
        )
        if reynolds and max(reynolds) <= 1.0:
            score += 0.35
            reasons.append("Navier-Stokes can use the low-Re incompressible Stokes TAPS simplification")
        elif has_frozen_velocity:
            score += 0.35
            reasons.append("Navier-Stokes can use the linearized Oseen TAPS target with a frozen convective velocity")
        elif _has_laminar_channel_inputs(problem):
            score += 0.30
            reasons.append("Navier-Stokes can use the restricted laminar channel Picard TAPS target")
        else:
            score += 0.15
            risks.append("full nonlinear Navier-Stokes TAPS execution is not connected yet; use low-Re Stokes, Oseen linearization, or fallback")
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
    problem_contract: PhysicsProblemContract | None = None
    knowledge_context: KnowledgeContext | None = None
    allow_knowledge_queries: bool = True


class FormulateTAPSEquationOutput(StrictBaseModel):
    plan: TAPSCompilationPlan


FORMULATE_TAPS_EQUATION_SYSTEM_PROMPT = """You are the PhysicsOS TAPS formulation agent.
Return only a JSON object matching FormulateTAPSEquationOutput.

Task:
- Compile the locked PhysicsProblem into a TAPSCompilationPlan.
- Preserve the PhysicsProblemContract exactly when it is provided: do not change fields, boundary values, operator family, material bindings, or problem_id.
- Use the knowledge_context only to improve the weak form, constitutive assumptions, validation risks, and required knowledge queries.
- For smooth stable continuum PDEs, produce a faithful weak-form IR even when no local TAPS kernel exists yet.
- Mark the plan as ready only when the governing equation, fields, boundary conditions, coefficients, axes, and weak form are explicit.
- If execution would require a full solver, backend bridge, or runtime extension, keep the weak form faithful and set risks/recommended_next_action accordingly instead of simplifying the physics away.
"""


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
        elif operator_family in {"navier_stokes", "stokes", "incompressible_navier_stokes"}:
            low_re = any(
                number.name.lower() in {"re", "reynolds", "reynolds_number"} and number.value <= 1.0
                for number in operator.nondimensional_numbers
            )
            has_frozen_velocity = any(
                prop.name.lower()
                in {"frozen_velocity", "convective_velocity", "oseen_velocity", "linearization_velocity", "ubar", "u_bar"}
                for material in problem.materials
                for prop in material.properties
            )
            terms.extend(
                [
                    TAPSEquationTerm(
                        id=f"{operator.id}:viscous",
                        role="diffusion",
                        expression="int_Omega 2 mu epsilon(v_U) : epsilon(U) dOmega",
                        fields=operator.fields_out or fields,
                        coefficients=["mu", "dynamic_viscosity"],
                        assumptions=["Velocity-pressure incompressible weak form; executable local kernel currently uses the low-Re Stokes limit."],
                    ),
                    TAPSEquationTerm(
                        id=f"{operator.id}:pressure",
                        role="custom",
                        expression="- int_Omega p div(v_U) dOmega",
                        fields=operator.fields_out or fields,
                        coefficients=["pressure"],
                    ),
                    TAPSEquationTerm(
                        id=f"{operator.id}:continuity",
                        role="constraint",
                        expression="int_Omega q div(U) dOmega",
                        fields=operator.fields_out or fields,
                    ),
                ]
            )
            if operator_family != "stokes" and not low_re:
                expression = (
                    "int_Omega v_U dot rho (U_bar dot grad) U dOmega"
                    if has_frozen_velocity
                    else "int_Omega v_U dot rho (U dot grad) U dOmega"
                )
                assumptions = (
                    ["Oseen linearization: convective velocity is frozen from user input, warm start, or previous iterate."]
                    if has_frozen_velocity
                    else ["Full Navier-Stokes requires nonlinear Picard/Newton iteration and stabilization before executable TAPS support."]
                )
                terms.append(
                    TAPSEquationTerm(
                        id=f"{operator.id}:advection",
                        role="advection",
                        expression=expression,
                        fields=operator.fields_out or fields,
                        coefficients=["rho", "density"],
                        assumptions=assumptions,
                    )
                )
            for source_term in operator.source_terms:
                expression = source_term.expression.strip()
                if expression:
                    terms.append(
                        TAPSEquationTerm(
                            id=f"{operator.id}:body_force",
                            role="source",
                            expression=f"int_Omega v_U dot {expression} dOmega",
                            fields=operator.fields_out or fields,
                        )
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
                        assumptions=["Small-strain linear elasticity; triangle meshes use plane stress/strain and tetrahedral meshes use 3D isotropic elasticity."],
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
        elif operator_family in {"hdiv", "div", "darcy", "mixed_poisson"}:
            flux_field = (operator.fields_out or fields or ["q"])[0]
            test_field = f"v_{flux_field}"
            terms.extend(
                [
                    TAPSEquationTerm(
                        id=f"{operator.id}:rt_mass",
                        role="mass",
                        expression=f"int_Omega K^-1 {test_field} dot {flux_field} dOmega",
                        fields=operator.fields_out or fields,
                        coefficients=["permeability", "K"],
                        assumptions=["Executable local kernel uses tetrahedral RT0 face-flux H(div) scaffold DOFs."],
                    ),
                    TAPSEquationTerm(
                        id=f"{operator.id}:divergence",
                        role="diffusion",
                        expression=f"int_Omega div({test_field}) div({flux_field}) dOmega",
                        fields=operator.fields_out or fields,
                        coefficients=[],
                    ),
                    TAPSEquationTerm(
                        id=f"{operator.id}:source",
                        role="source",
                        expression=f"- int_Omega source div({test_field}) dOmega",
                        fields=operator.fields_out or fields,
                        coefficients=["source"],
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


def formulate_taps_equation_structured(
    input: FormulateTAPSEquationInput,
    *,
    client: StructuredLLMClient,
    config: CoreAgentLLMConfig | None = None,
) -> FormulateTAPSEquationOutput:
    """LLM-backed TAPS formulation with strict Pydantic validation."""
    result = call_structured_agent(
        agent_name="taps-formulation-agent",
        input_model=input,
        output_model=FormulateTAPSEquationOutput,
        system_prompt=FORMULATE_TAPS_EQUATION_SYSTEM_PROMPT,
        client=client,
        config=config,
    )
    if result.output is not None:
        return result.output

    fallback = formulate_taps_equation(input)
    plan = fallback.plan.model_copy(
        update={
            "assumptions": [
                *fallback.plan.assumptions,
                "Structured LLM TAPS formulation failed validation after bounded retries; deterministic formulation fallback was used.",
                result.error or "Structured TAPS formulation returned no validated output.",
            ]
        }
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
        for number in operator.nondimensional_numbers:
            coefficients.append(
                TAPSCoefficientSpec(
                    name=number.name,
                    value=number.value,
                    role="operator",
                    source="physics_problem",
                )
            )
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
            boundary_role=boundary.boundary_role or _canonical_boundary_role(boundary.region_id),
            field=boundary.field,
            kind=boundary.kind,
            value=boundary.value,
            units=boundary.units,
            source="physics_problem",
        )
        for boundary in problem.boundary_conditions
    ]


def _canonical_boundary_role(region_id: str) -> str | None:
    lowered = region_id.lower().replace(" ", "").replace("-", "_")
    pieces = [piece for piece in lowered.replace(":", "_").split("_") if piece]
    if (
        lowered in {"x=0", "x_min", "xmin", "x0", "left", "boundary:x_min", "boundary:left"}
        or lowered.endswith(":x_min")
        or "left" in pieces
        or "x0" in pieces
        or ("x" in pieces and "0" in pieces)
    ):
        return "x_min"
    if (
        lowered in {"x=l", "x=1", "x_max", "xmax", "x1", "right", "boundary:x_max", "boundary:right"}
        or lowered.endswith(":x_max")
        or "right" in pieces
        or "x1" in pieces
        or ("x" in pieces and "1" in pieces)
    ):
        return "x_max"
    return None


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
            "Current executable kernels support 1D transient heat, 1D/2D/3D scalar elliptic weak forms, 1D/2D nonlinear reaction-diffusion, 2-field 2D coupled reaction-diffusion, mesh P1/P2 3D tetra scalar Poisson with Dirichlet lifting, mesh P1/P2/P3 2D triangle linear elasticity, mesh P1/P2 3D tetra linear elasticity with Dirichlet lifting, and mesh first-order Nedelec EM curl-curl.",
            "Geometry encodings are carried into TAPSProblem and executable kernels consume occupancy masks when available.",
            *plan.assumptions,
        ],
    )
    return BuildTAPSProblemOutput(taps_problem=taps_problem)


class RunTAPSBackendInput(StrictBaseModel):
    problem: PhysicsProblem
    taps_problem: TAPSProblem
    budget: ComputeBudget = Field(default_factory=lambda: ComputeBudget(max_wall_time_seconds=120.0))
    numerical_plan: NumericalSolvePlanOutput | None = None


class RunTAPSBackendOutput(StrictBaseModel):
    result: SolverResult


class NumericalSolvePlanInput(StrictBaseModel):
    problem: PhysicsProblem
    taps_problem: TAPSProblem
    problem_contract: PhysicsProblemContract | None = None
    compilation_plan: TAPSCompilationPlan | None = None
    knowledge_context: KnowledgeContext | None = None
    budget: ComputeBudget = Field(default_factory=lambda: ComputeBudget(max_wall_time_seconds=120.0))


class ValidateNumericalSolvePlanInput(StrictBaseModel):
    problem: PhysicsProblem
    taps_problem: TAPSProblem
    plan: NumericalSolvePlanOutput
    problem_contract: PhysicsProblemContract | None = None


class ValidateNumericalSolvePlanOutput(StrictBaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


NUMERICAL_SOLVE_PLAN_SYSTEM_PROMPT = """You are the PhysicsOS numerical solve planning agent.
Return only a JSON object matching NumericalSolvePlanOutput.

Task:
- Preserve the PhysicsProblemContract when provided.
- Bind fields, coefficients, sources, boundary conditions, discretization, expected artifacts, and validation checks for the selected deterministic solver kernel.
- Do not invent coefficients, boundary values, fields, mesh references, or solution arrays.
- If a requested solve is unsupported, return status=unsupported or fallback_required with explicit unsupported_reasons.
"""


def _as_float(value: object) -> float | None:
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _coefficient_value(problem: PhysicsProblem, names: set[str], default: float) -> tuple[float, str, str | None]:
    normalized = {name.lower() for name in names}
    for parameter in problem.parameters:
        if parameter.name.lower() in normalized:
            value = _as_float(parameter.value)
            if value is not None:
                return value, parameter.name, parameter.units
    for material in problem.materials:
        for prop in material.properties:
            if prop.name.lower() in normalized:
                value = _as_float(prop.value)
                if value is not None:
                    return value, prop.name, prop.units
    return default, "default", None


def _coefficient_raw_value(
    problem: PhysicsProblem,
    names: set[str],
    default: float | int | str | list[float] | dict,
) -> tuple[float | int | str | list[float] | dict, str, str | None]:
    normalized = {name.lower() for name in names}
    for parameter in problem.parameters:
        if parameter.name.lower() in normalized:
            return parameter.value, parameter.name, parameter.units
    for material in problem.materials:
        for prop in material.properties:
            if prop.name.lower() in normalized:
                return prop.value, prop.name, prop.units
    return default, "default", None


def _source_binding_from_problem(problem: PhysicsProblem) -> NumericalSourceBinding:
    for operator in problem.operators:
        for index, source in enumerate(operator.source_terms):
            value = _as_float(source.expression)
            if value is not None:
                return NumericalSourceBinding(name=f"{operator.id}:source:{index}", value=value, expression=source.expression, units=source.units)
    return NumericalSourceBinding(name="zero_source", value=0.0, expression="0")


def plan_numerical_solve(input: NumericalSolvePlanInput) -> NumericalSolvePlanOutput:
    """Deterministic numerical solve planning fallback."""
    problem = input.problem
    taps_problem = input.taps_problem
    family = taps_problem.weak_form.family.lower() if taps_problem.weak_form is not None else "unknown"
    space_axes = [axis for axis in taps_problem.axes if axis.kind == "space"]
    field = taps_problem.weak_form.trial_fields[0] if taps_problem.weak_form and taps_problem.weak_form.trial_fields else (problem.fields[0].name if problem.fields else "u")
    diffusion, diffusion_name, diffusion_units = _coefficient_value(
        problem,
        {"k", "thermal_conductivity", "d", "diffusivity", "thermal_diffusivity"},
        1.0,
    )
    reaction = 0.0
    operator_classes = {operator.equation_class.lower() for operator in problem.operators}
    is_transient_diffusion = bool(input.problem.initial_conditions) and (
        bool(operator_classes.intersection({"heat", "diffusion", "thermal_diffusion"}))
        or supports_transient_diffusion_weak_form(taps_problem)
    )
    is_coupled_reaction = "coupled_reaction_diffusion" in operator_classes or supports_coupled_reaction_diffusion_weak_form(taps_problem)
    is_vector_elasticity = bool(operator_classes.intersection({"elasticity", "linear_elasticity"})) or supports_vector_elasticity_weak_form(taps_problem)
    is_hcurl_curl_curl = bool(operator_classes.intersection({"maxwell", "curl_curl", "electromagnetic"})) or supports_hcurl_curl_curl_weak_form(taps_problem)
    is_hdiv_div = bool(operator_classes.intersection({"hdiv", "div", "darcy", "mixed_poisson"})) or supports_hdiv_div_weak_form(taps_problem)
    is_oseen = supports_oseen_weak_form(taps_problem)
    is_navier_stokes = supports_navier_stokes_weak_form(taps_problem)
    is_stokes = supports_stokes_weak_form(taps_problem)
    is_nonlinear_reaction = "reaction_diffusion" in operator_classes or (
        supports_nonlinear_reaction_diffusion_weak_form(taps_problem)
    )
    has_mesh_graph = any(encoding.kind == "mesh_graph" for encoding in taps_problem.geometry_encodings)
    boundary_bindings = [
        NumericalBoundaryConditionBinding(
            id=boundary.id,
            region_id=boundary.region_id,
            boundary_role=boundary.boundary_role or _canonical_boundary_role(boundary.region_id),
            field=boundary.field,
            kind=boundary.kind,
            value=boundary.value,
            units=boundary.units,
        )
        for boundary in problem.boundary_conditions
    ]
    if is_oseen:
        solver_family = "incompressible_oseen_channel_2d"
    elif is_navier_stokes:
        solver_family = "incompressible_navier_stokes_channel_2d"
    elif is_stokes:
        solver_family = "incompressible_stokes_channel_2d"
    elif is_hcurl_curl_curl:
        solver_family = "mesh_fem_em_curl_curl"
    elif is_hdiv_div and has_mesh_graph:
        solver_family = "mesh_fem_hdiv_div"
    elif is_vector_elasticity:
        solver_family = "mesh_fem_linear_elasticity"
    elif has_mesh_graph and family in {"poisson", "diffusion", "thermal_diffusion"}:
        solver_family = "mesh_fem_poisson"
    elif len(space_axes) == 1 and field and is_transient_diffusion:
        solver_family = "transient_diffusion_1d"
    elif len(space_axes) == 2 and len(problem.fields) >= 2 and is_coupled_reaction:
        solver_family = "coupled_reaction_diffusion_2d"
    elif len(space_axes) == 1 and field and is_nonlinear_reaction:
        solver_family = "nonlinear_reaction_diffusion_1d"
    elif len(space_axes) == 2 and field and is_nonlinear_reaction:
        solver_family = "nonlinear_reaction_diffusion_2d"
    elif len(space_axes) == 1 and field:
        solver_family = "scalar_elliptic_1d"
    elif len(space_axes) == 2 and field:
        solver_family = "scalar_elliptic_2d"
    elif len(space_axes) == 3 and field:
        solver_family = "scalar_elliptic_3d"
    else:
        solver_family = family
    ready_families = {
        "scalar_elliptic_1d",
        "scalar_elliptic_2d",
        "scalar_elliptic_3d",
        "nonlinear_reaction_diffusion_1d",
        "nonlinear_reaction_diffusion_2d",
        "coupled_reaction_diffusion_2d",
        "transient_diffusion_1d",
        "mesh_fem_poisson",
        "mesh_fem_linear_elasticity",
        "mesh_fem_em_curl_curl",
        "mesh_fem_hdiv_div",
        "incompressible_stokes_channel_2d",
        "incompressible_oseen_channel_2d",
        "incompressible_navier_stokes_channel_2d",
    }
    status = "ready" if solver_family in ready_families else "fallback_required"
    coefficient_bindings = [
        NumericalCoefficientBinding(name="diffusion", role="diffusion", value=diffusion, source_name=diffusion_name, units=diffusion_units),
        NumericalCoefficientBinding(name="reaction", role="reaction", value=reaction, source_name="weak_form", units=None),
    ]
    if solver_family.startswith("nonlinear_reaction_diffusion"):
        coefficient_bindings.extend(
            [
                NumericalCoefficientBinding(name="linear_reaction", role="reaction", value=input.taps_problem.nonlinear.linear_reaction, source_name="taps_nonlinear"),
                NumericalCoefficientBinding(name="cubic_reaction", role="reaction", value=input.taps_problem.nonlinear.cubic_reaction, source_name="taps_nonlinear"),
                NumericalCoefficientBinding(name="damping", role="solver", value=input.taps_problem.nonlinear.damping, source_name="taps_nonlinear"),
                NumericalCoefficientBinding(name="max_iterations", role="solver", value=input.taps_problem.nonlinear.max_iterations, source_name="taps_nonlinear"),
                NumericalCoefficientBinding(name="tolerance", role="solver", value=input.taps_problem.nonlinear.tolerance, source_name="taps_nonlinear"),
            ]
        )
    if solver_family == "coupled_reaction_diffusion_2d":
        coefficient_bindings.extend(
            [
                NumericalCoefficientBinding(name="linear_reaction", role="reaction", value=input.taps_problem.nonlinear.linear_reaction, source_name="taps_nonlinear"),
                NumericalCoefficientBinding(name="cubic_reaction", role="reaction", value=input.taps_problem.nonlinear.cubic_reaction, source_name="taps_nonlinear"),
                NumericalCoefficientBinding(name="coupling_strength", role="reaction", value=input.taps_problem.nonlinear.coupling_strength, source_name="taps_nonlinear"),
                NumericalCoefficientBinding(name="damping", role="solver", value=input.taps_problem.nonlinear.damping, source_name="taps_nonlinear"),
                NumericalCoefficientBinding(name="max_iterations", role="solver", value=input.taps_problem.nonlinear.max_iterations, source_name="taps_nonlinear"),
                NumericalCoefficientBinding(name="tolerance", role="solver", value=input.taps_problem.nonlinear.tolerance, source_name="taps_nonlinear"),
            ]
        )
    if solver_family == "transient_diffusion_1d":
        coefficient_bindings.extend(
            [
                NumericalCoefficientBinding(name="rank", role="solver", value=input.taps_problem.basis.tensor_rank, source_name="taps_basis"),
                NumericalCoefficientBinding(name="time_integrator", role="solver", value="low_rank_taylor_spt", source_name="deterministic_fallback"),
            ]
        )
    if solver_family == "mesh_fem_linear_elasticity":
        young_modulus, young_name, young_units = _coefficient_value(problem, {"e", "young_modulus", "youngs_modulus", "young's_modulus"}, 1.0)
        poisson_ratio, poisson_name, poisson_units = _coefficient_value(problem, {"nu", "poisson_ratio", "poissons_ratio", "poisson's_ratio"}, 0.3)
        coefficient_bindings.extend(
            [
                NumericalCoefficientBinding(name="young_modulus", role="operator", value=young_modulus, source_name=young_name, units=young_units),
                NumericalCoefficientBinding(name="poisson_ratio", role="operator", value=poisson_ratio, source_name=poisson_name, units=poisson_units),
                NumericalCoefficientBinding(name="max_iterations", role="solver", value=20000, source_name="deterministic_fallback"),
                NumericalCoefficientBinding(name="tolerance", role="solver", value=1e-8, source_name="deterministic_fallback"),
            ]
        )
        for material in problem.materials:
            for prop in material.properties:
                if prop.name.lower() in {"constitutive_model", "stress_model"}:
                    coefficient_bindings.append(
                        NumericalCoefficientBinding(name="constitutive_model", role="operator", value=prop.value, source_name=prop.name, units=prop.units)
                    )
                if prop.name.lower() in {"body_force", "b", "gravity"}:
                    coefficient_bindings.append(
                        NumericalCoefficientBinding(name="body_force", role="source", value=prop.value, source_name=prop.name, units=prop.units)
                    )
    if solver_family == "mesh_fem_em_curl_curl":
        for output_name, names, default in [
            ("relative_permeability", {"mu_r", "relative_permeability", "permeability"}, 1.0),
            ("relative_permittivity", {"eps_r", "epsilon_r", "relative_permittivity", "permittivity"}, 1.0),
            ("wave_number", {"k0", "k", "wave_number", "wavenumber"}, 0.5),
            ("source_amplitude", {"source", "current_source", "jz"}, 1.0),
        ]:
            value, source_name, units = _coefficient_raw_value(problem, names, default)
            role = "source" if output_name == "source_amplitude" else "operator"
            coefficient_bindings.append(NumericalCoefficientBinding(name=output_name, role=role, value=value, source_name=source_name, units=units))
        coefficient_bindings.extend(
            [
                NumericalCoefficientBinding(name="max_iterations", role="solver", value=5000, source_name="deterministic_fallback"),
                NumericalCoefficientBinding(name="tolerance", role="solver", value=1e-8, source_name="deterministic_fallback"),
            ]
        )
    if solver_family == "mesh_fem_hdiv_div":
        for output_name, names, default in [
            ("permeability", {"permeability", "hydraulic_conductivity", "k"}, 1.0),
            ("source_amplitude", {"source", "rhs", "forcing", "sink"}, 1.0),
        ]:
            value, source_name, units = _coefficient_raw_value(problem, names, default)
            role = "source" if output_name == "source_amplitude" else "operator"
            coefficient_bindings.append(NumericalCoefficientBinding(name=output_name, role=role, value=value, source_name=source_name, units=units))
        coefficient_bindings.extend(
            [
                NumericalCoefficientBinding(name="max_iterations", role="solver", value=5000, source_name="deterministic_fallback"),
                NumericalCoefficientBinding(name="tolerance", role="solver", value=1e-8, source_name="deterministic_fallback"),
            ]
        )
    if solver_family in {"incompressible_stokes_channel_2d", "incompressible_oseen_channel_2d", "incompressible_navier_stokes_channel_2d"}:
        for output_name, names, default in [
            ("dynamic_viscosity", {"mu", "dynamic_viscosity", "viscosity"}, 1.0),
            ("density", {"rho", "density"}, 1.0),
            ("pressure_drop", {"pressure_drop", "delta_p", "dp"}, 1.0),
            ("reynolds", {"re", "reynolds", "reynolds_number"}, -1.0),
        ]:
            value, source_name, units = _coefficient_raw_value(problem, names, default)
            coefficient_bindings.append(NumericalCoefficientBinding(name=output_name, role="operator", value=value, source_name=source_name, units=units))
        if solver_family == "incompressible_oseen_channel_2d":
            frozen_velocity, source_name, units = _coefficient_raw_value(
                problem,
                {"frozen_velocity", "convective_velocity", "oseen_velocity", "linearization_velocity", "ubar", "u_bar"},
                [0.0, 0.0],
            )
            coefficient_bindings.append(NumericalCoefficientBinding(name="frozen_velocity", role="operator", value=frozen_velocity, source_name=source_name, units=units))
        if solver_family == "incompressible_navier_stokes_channel_2d":
            coefficient_bindings.extend(
                [
                    NumericalCoefficientBinding(name="damping", role="solver", value=input.taps_problem.nonlinear.damping, source_name="taps_nonlinear"),
                    NumericalCoefficientBinding(name="max_iterations", role="solver", value=min(80, input.taps_problem.nonlinear.max_iterations), source_name="taps_nonlinear"),
                    NumericalCoefficientBinding(name="tolerance", role="solver", value=input.taps_problem.nonlinear.tolerance, source_name="taps_nonlinear"),
                    NumericalCoefficientBinding(name="support_scope", role="solver", value="restricted_steady_laminar_2d_channel", source_name="taps_phase3_policy"),
                ]
            )
    field_bindings = {"primary": field}
    if solver_family == "coupled_reaction_diffusion_2d" and len(problem.fields) >= 2:
        field_bindings["secondary"] = problem.fields[1].name
    if solver_family.startswith("incompressible_") and len(problem.fields) >= 2:
        field_bindings = {"velocity": problem.fields[0].name, "pressure": problem.fields[1].name}
    return NumericalSolvePlanOutput(
        problem_id=problem.id,
        status=status,  # type: ignore[arg-type]
        solver_family=solver_family,
        backend_target="taps",
        field_bindings=field_bindings,
        discretization=NumericalDiscretizationSpec(
            dimension=max(1, len(space_axes)),
            node_counts={axis.name: axis.points or 64 for axis in space_axes},
            element_order=1,
            quadrature_order=taps_problem.basis.quadrature_order,
        ),
        coefficient_bindings=coefficient_bindings,
        source_bindings=[_source_binding_from_problem(problem)],
        boundary_condition_bindings=boundary_bindings,
        initial_condition_bindings=[
            {
                "id": initial.id,
                "field": initial.field,
                "value": initial.value.model_dump(mode="json") if hasattr(initial.value, "model_dump") else initial.value,
                **({"units": initial.units} if initial.units is not None else {}),
            }
            for initial in problem.initial_conditions
        ],
        expected_artifacts=["taps_assembled_operator", "taps_solution_field", "taps_residual_history"],
        validation_checks=["contract_preserved", "coefficient_bound", "dirichlet_endpoint_match", "linear_residual"],
        assumptions=["Deterministic numerical solve fallback bound coefficients, sources, and boundary conditions from PhysicsProblem."],
        unsupported_reasons=[] if status == "ready" else [f"No deterministic numerical plan fallback for solver_family={solver_family}."],
    )


def validate_numerical_solve_plan(input: ValidateNumericalSolvePlanInput) -> ValidateNumericalSolvePlanOutput:
    """Validate a numerical solve plan before deterministic TAPS kernel execution."""
    errors: list[str] = []
    warnings: list[str] = []
    problem = input.problem
    taps_problem = input.taps_problem
    plan = input.plan
    if plan.problem_id != problem.id:
        errors.append(f"plan.problem_id={plan.problem_id!r} does not match problem.id={problem.id!r}.")
    if plan.backend_target != "taps":
        errors.append(f"Unsupported backend_target={plan.backend_target!r} for run_taps_backend.")
    if plan.status != "ready":
        if plan.status in {"fallback_required", "unsupported"}:
            if not plan.unsupported_reasons:
                errors.append(f"Numerical solve plan status={plan.status} must include unsupported_reasons.")
            if plan.status == "fallback_required" and not plan.fallback_decision:
                errors.append("fallback_required numerical solve plan must include fallback_decision.")
        else:
            errors.append(f"Numerical solve plan is not ready: status={plan.status}.")
        return ValidateNumericalSolvePlanOutput(valid=not errors, errors=errors, warnings=warnings)
    fields = {field.name for field in problem.fields}
    if plan.solver_family.startswith("incompressible_"):
        primary_field = plan.field_bindings.get("velocity")
        pressure_field = plan.field_bindings.get("pressure")
        if not primary_field or not pressure_field:
            errors.append("Incompressible fluid numerical solve plan must bind field_bindings.velocity and field_bindings.pressure.")
        else:
            for role, field_name in {"velocity": primary_field, "pressure": pressure_field}.items():
                if fields and field_name not in fields:
                    errors.append(f"Numerical {role} field {field_name!r} is not in PhysicsProblem fields {sorted(fields)}.")
    else:
        primary_field = plan.field_bindings.get("primary")
        if not primary_field:
            errors.append("Numerical solve plan must bind field_bindings.primary.")
        elif fields and primary_field not in fields:
            errors.append(f"Numerical primary field {primary_field!r} is not in PhysicsProblem fields {sorted(fields)}.")
    bc_by_id = {boundary.id: boundary for boundary in problem.boundary_conditions}
    geometry_boundary_roles = {boundary.id: boundary.role for boundary in problem.geometry.boundaries if boundary.role is not None}
    for binding in plan.boundary_condition_bindings:
        source = bc_by_id.get(binding.id)
        if source is None:
            errors.append(f"Numerical solve plan invented boundary condition {binding.id!r}.")
            continue
        source_role = source.boundary_role or geometry_boundary_roles.get(source.region_id) or _canonical_boundary_role(source.region_id)
        if binding.field != source.field or binding.kind != source.kind or binding.region_id != source.region_id or binding.value != source.value:
            errors.append(f"Boundary binding {binding.id!r} does not preserve PhysicsProblem boundary condition.")
        if binding.boundary_role != source_role:
            errors.append(
                f"Boundary binding {binding.id!r} role {binding.boundary_role!r} does not preserve canonical role {source_role!r}."
            )
    if any(boundary.kind.lower() == "dirichlet" for boundary in problem.boundary_conditions):
        dirichlet_bindings = [binding for binding in plan.boundary_condition_bindings if binding.kind.lower() == "dirichlet"]
        if len(dirichlet_bindings) < 2 and plan.solver_family == "scalar_elliptic_1d":
            warnings.append("1D scalar elliptic plan has fewer than two Dirichlet bindings; defaults may be underconstrained.")
        if plan.solver_family == "scalar_elliptic_1d":
            roles = {binding.boundary_role for binding in dirichlet_bindings}
            if not {"x_min", "x_max"}.issubset(roles):
                errors.append("1D scalar elliptic plan requires canonical Dirichlet boundary_role values x_min and x_max.")
            if len(dirichlet_bindings) != 2:
                errors.append("1D scalar elliptic executable kernel currently requires exactly two Dirichlet endpoint bindings.")
    if not any(binding.role == "diffusion" for binding in plan.coefficient_bindings):
        errors.append("Numerical solve plan must bind a diffusion coefficient for scalar elliptic execution.")
    if plan.solver_family.startswith("nonlinear_reaction_diffusion"):
        for required in {"linear_reaction", "cubic_reaction", "damping", "max_iterations", "tolerance"}:
            if not any(binding.name == required for binding in plan.coefficient_bindings):
                errors.append(f"Nonlinear numerical solve plan must bind {required}.")
    if plan.solver_family == "coupled_reaction_diffusion_2d":
        if "secondary" not in plan.field_bindings:
            errors.append("Coupled reaction-diffusion plan must bind field_bindings.secondary.")
        elif fields and plan.field_bindings["secondary"] not in fields:
            errors.append(f"Numerical secondary field {plan.field_bindings['secondary']!r} is not in PhysicsProblem fields {sorted(fields)}.")
        for required in {"linear_reaction", "cubic_reaction", "coupling_strength", "damping", "max_iterations", "tolerance"}:
            if not any(binding.name == required for binding in plan.coefficient_bindings):
                errors.append(f"Coupled reaction-diffusion numerical solve plan must bind {required}.")
    if plan.solver_family == "transient_diffusion_1d":
        if not problem.initial_conditions:
            errors.append("Transient diffusion numerical solve plan requires at least one PhysicsProblem initial condition.")
        if not plan.initial_condition_bindings:
            errors.append("Transient diffusion numerical solve plan must bind initial_condition_bindings.")
        initial_by_id = {initial.id: initial for initial in problem.initial_conditions}
        for binding in plan.initial_condition_bindings:
            binding_id = str(binding.get("id", ""))
            source = initial_by_id.get(binding_id)
            if source is None:
                errors.append(f"Numerical solve plan invented initial condition {binding_id!r}.")
                continue
            source_value = source.value.model_dump(mode="json") if hasattr(source.value, "model_dump") else source.value
            if binding.get("field") != source.field or binding.get("value") != source_value:
                errors.append(f"Initial condition binding {binding_id!r} does not preserve PhysicsProblem initial condition.")
        if not any(binding.name == "rank" and binding.role == "solver" for binding in plan.coefficient_bindings):
            warnings.append("Transient diffusion plan did not explicitly bind low-rank solver rank; TAPS basis rank will be used.")
    if plan.solver_family == "mesh_fem_linear_elasticity":
        if not any(encoding.kind == "mesh_graph" for encoding in taps_problem.geometry_encodings):
            errors.append("Mesh FEM linear-elasticity plan requires a mesh_graph geometry encoding.")
        for required in {"young_modulus", "poisson_ratio"}:
            if not any(binding.name == required for binding in plan.coefficient_bindings):
                errors.append(f"Mesh FEM linear-elasticity numerical solve plan must bind {required}.")
        if not any(binding.name == "body_force" for binding in plan.coefficient_bindings):
            warnings.append("Mesh FEM linear-elasticity plan did not bind body_force; deterministic default may be used.")
    if plan.solver_family == "mesh_fem_poisson":
        if not any(encoding.kind == "mesh_graph" for encoding in taps_problem.geometry_encodings):
            errors.append("Mesh FEM Poisson plan requires a mesh_graph geometry encoding.")
        if not any(binding.role == "diffusion" for binding in plan.coefficient_bindings):
            errors.append("Mesh FEM Poisson numerical solve plan must bind diffusion.")
    if plan.solver_family == "mesh_fem_em_curl_curl":
        if not any(encoding.kind == "mesh_graph" for encoding in taps_problem.geometry_encodings):
            errors.append("Mesh FEM EM curl-curl plan requires a mesh_graph geometry encoding.")
        for required in {"relative_permeability", "relative_permittivity", "wave_number", "source_amplitude"}:
            if not any(binding.name == required for binding in plan.coefficient_bindings):
                errors.append(f"Mesh FEM EM curl-curl numerical solve plan must bind {required}.")
    if plan.solver_family == "mesh_fem_hdiv_div":
        if not any(encoding.kind == "mesh_graph" for encoding in taps_problem.geometry_encodings):
            errors.append("Mesh FEM H(div) plan requires a mesh_graph geometry encoding.")
        for required in {"permeability", "source_amplitude"}:
            if not any(binding.name == required for binding in plan.coefficient_bindings):
                errors.append(f"Mesh FEM H(div) numerical solve plan must bind {required}.")
    if plan.solver_family in {"incompressible_stokes_channel_2d", "incompressible_oseen_channel_2d", "incompressible_navier_stokes_channel_2d"}:
        for required in {"dynamic_viscosity", "pressure_drop"}:
            if not any(binding.name == required for binding in plan.coefficient_bindings):
                errors.append(f"Fluid numerical solve plan must bind {required}.")
        if plan.discretization.dimension != 2:
            errors.append("Fluid channel numerical solve plan must be 2D.")
    if plan.solver_family == "incompressible_oseen_channel_2d":
        if not any(binding.name == "frozen_velocity" for binding in plan.coefficient_bindings):
            errors.append("Oseen numerical solve plan must bind frozen_velocity.")
    if plan.solver_family == "incompressible_navier_stokes_channel_2d":
        for required in {"density", "damping", "max_iterations", "tolerance", "support_scope"}:
            if not any(binding.name == required for binding in plan.coefficient_bindings):
                errors.append(f"Restricted Navier-Stokes numerical solve plan must bind {required}.")
        scope = next((binding.value for binding in plan.coefficient_bindings if binding.name == "support_scope"), None)
        if scope != "restricted_steady_laminar_2d_channel":
            errors.append("Restricted Navier-Stokes numerical solve plan must declare support_scope=restricted_steady_laminar_2d_channel.")
    if plan.discretization.dimension != len([axis for axis in taps_problem.axes if axis.kind == "space"]):
        errors.append("Numerical discretization dimension does not match TAPS space axes.")
    return ValidateNumericalSolvePlanOutput(valid=not errors, errors=errors, warnings=warnings)


def plan_numerical_solve_structured(
    input: NumericalSolvePlanInput,
    *,
    client: StructuredLLMClient,
    config: CoreAgentLLMConfig | None = None,
) -> NumericalSolvePlanOutput:
    """LLM-backed numerical solve planning with strict Pydantic validation."""
    def semantic_errors(plan: NumericalSolvePlanOutput) -> list[str]:
        validation = validate_numerical_solve_plan(
            ValidateNumericalSolvePlanInput(
                problem=input.problem,
                taps_problem=input.taps_problem,
                plan=plan,
                problem_contract=input.problem_contract,
            )
        )
        return validation.errors

    def retry_feedback(plan: NumericalSolvePlanOutput, errors: list[str]) -> list[str]:
        context = {
            "instruction": (
                "Repair only the invalid NumericalSolvePlanOutput. Preserve the locked PhysicsProblemContract, "
                "PhysicsProblem ids, fields, coefficients, boundary ids, boundary roles, values, units, and selected "
                "solver_family unless an explicit fallback_required decision is necessary."
            ),
            "errors": errors,
            "invalid_plan": plan.model_dump(mode="json"),
            "problem_contract": input.problem_contract.model_dump(mode="json") if input.problem_contract is not None else None,
            "physics_problem": input.problem.model_dump(mode="json"),
            "taps_problem": input.taps_problem.model_dump(mode="json"),
            "compilation_plan": input.compilation_plan.model_dump(mode="json") if input.compilation_plan is not None else None,
        }
        return [json.dumps(context, ensure_ascii=False)]

    result = call_structured_agent(
        agent_name="numerical-solve-planning-agent",
        input_model=input,
        output_model=NumericalSolvePlanOutput,
        system_prompt=NUMERICAL_SOLVE_PLAN_SYSTEM_PROMPT,
        client=client,
        config=config,
        semantic_validator=semantic_errors,
        semantic_feedback_builder=retry_feedback,
    )
    if result.output is not None:
        return result.output
    last_parsed_plan: NumericalSolvePlanOutput | None = None
    for attempt in reversed(result.attempts):
        if attempt.parsed is None:
            continue
        try:
            last_parsed_plan = NumericalSolvePlanOutput.model_validate(attempt.parsed)
            break
        except Exception:
            continue
    if last_parsed_plan is not None:
        validation_errors = result.attempts[-1].validation_errors if result.attempts else []
        return last_parsed_plan.model_copy(
            update={
                "warnings": [
                    *last_parsed_plan.warnings,
                    "Structured LLM numerical solve planning exhausted semantic validation retries; plan was preserved for workflow validation instead of deterministic fallback.",
                    *(validation_errors or [result.error or "Structured numerical solve planner returned no validated output."]),
                ]
            }
        )
    fallback = plan_numerical_solve(input)
    return fallback.model_copy(
        update={
            "assumptions": [
                *fallback.assumptions,
                "Structured LLM numerical solve planning failed validation; deterministic fallback was used.",
                result.error or "Structured numerical solve planner returned no validated output.",
            ]
        }
    )


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
    mesh_navier_stokes_bridge = supports_mesh_navier_stokes_bridge(input.taps_problem)
    supported = any(
        [
            supports_transient_diffusion_weak_form(input.taps_problem),
            supports_coupled_reaction_diffusion_weak_form(input.taps_problem),
            supports_hcurl_curl_curl_weak_form(input.taps_problem),
            supports_nonlinear_reaction_diffusion_weak_form(input.taps_problem),
            supports_navier_stokes_weak_form(input.taps_problem),
            supports_oseen_weak_form(input.taps_problem),
            supports_scalar_elliptic_weak_form(input.taps_problem),
            supports_stokes_weak_form(input.taps_problem),
            supports_vector_elasticity_weak_form(input.taps_problem),
            (weak_form.family.lower() if weak_form is not None else "") in GENERIC_TAPS_FAMILIES,
        ]
    )
    if not supported:
        warnings.append("no executable TAPS IR block mapping is currently connected for this weak_form")
    if mesh_navier_stokes_bridge and not supported:
        warnings.append("mesh-based incompressible Navier-Stokes bridge is available, but local TAPS execution is intentionally disabled; export to FEniCSx/OpenFOAM/SU2.")
    checks.append({"name": "executable_block_mapping_connected", "passes": supported})
    checks.append({"name": "mesh_navier_stokes_backend_bridge_available", "passes": mesh_navier_stokes_bridge})
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
    backend_preparation_plan: BackendPreparationPlanOutput | None = None
    mesh_export_manifest: ArtifactRef | None = None


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
        elif "incompressible_navier_stokes_mesh" in families:
            lines.extend(
                [
                    "# Use mixed velocity-pressure spaces such as VectorElement('Lagrange', cell, 2) * FiniteElement('Lagrange', cell, 1).",
                    "# Assemble incompressible NS residual: rho*dot(dot(u, nabla_grad(u)), v)*dx + 2*mu*inner(sym(grad(u)), sym(grad(v)))*dx - p*div(v)*dx + q*div(u)*dx.",
                    "# Add reviewed SUPG/PSPG or projection stabilization before execution.",
                    "# Bind facet tags from backend_mesh_export_manifest for inlet/outlet/wall conditions.",
                ]
            )
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
    if backend == "openfoam":
        return "md", "\n".join(
            [
                "# Draft OpenFOAM bridge generated from TAPS IR",
                "",
                "- Target solver: simpleFoam for steady incompressible laminar/RANS cases, or icoFoam/pimpleFoam when transient handling is required.",
                "- Required mesh export: `constant/polyMesh` with inlet/outlet/wall patches from `backend_mesh_export_manifest`.",
                "- Required dictionaries: `0/U`, `0/p`, `constant/transportProperties`, `system/fvSchemes`, `system/fvSolution`, `system/controlDict`.",
                "- Stabilization/discretization review: convection scheme, pressure-velocity coupling, non-orthogonal correction, turbulence model policy.",
                "- Execution remains disabled until an approved runner consumes the case bundle.",
            ]
        ) + "\n"
    if backend == "su2":
        return "md", "\n".join(
            [
                "# Draft SU2 bridge generated from TAPS IR",
                "",
                "- Target solver: SU2_CFD with incompressible or low-Mach configuration when applicable.",
                "- Required mesh export: `.su2` mesh with inlet/outlet/wall markers from `backend_mesh_export_manifest`.",
                "- Required config: fluid model, viscosity/density, markers, convergence criteria, CFL/linear solver policy.",
                "- Stabilization/discretization review: pressure coupling, convective scheme, limiter/turbulence policy.",
                "- Execution remains disabled until an approved runner consumes the case bundle.",
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
    preparation_plan = input.backend_preparation_plan or plan_backend_preparation(
        PlanBackendPreparationInput(
            problem=input.problem,
            taps_problem=input.taps_problem,
            backend=backend,
            mesh_export_manifest=input.mesh_export_manifest,
        )
    )
    preparation_validation = validate_backend_preparation_plan(
        ValidateBackendPreparationPlanInput(
            problem=input.problem,
            taps_problem=input.taps_problem,
            plan=preparation_plan,
            mesh_export_manifest=input.mesh_export_manifest,
        )
    )
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
    if supports_mesh_navier_stokes_bridge(input.taps_problem):
        blocks.append(
            {
                "family": "incompressible_navier_stokes_mesh",
                "space": "mixed_H1_velocity_pressure_or_FVM_cell_fields",
                "nonlinear_solver": "picard_or_newton_review_required",
                "pressure_velocity_coupling": "monolithic_projection_SIMPLE_or_PISO",
                "stabilization": "SUPG_PSPG_projection_or_solver_native_scheme_review_required",
                "execute_locally": False,
                "fallback_targets": ["fenicsx", "openfoam", "su2"],
            }
        )
    if supports_navier_stokes_weak_form(input.taps_problem):
        blocks.append({"family": "incompressible_navier_stokes", "space": "mixed_H1_velocity_pressure", "nonlinear_solver": "picard_channel"})
    if supports_oseen_weak_form(input.taps_problem):
        blocks.append({"family": "incompressible_oseen", "space": "mixed_H1_velocity_pressure", "solver": "linearized_oseen_channel"})
    if supports_stokes_weak_form(input.taps_problem):
        blocks.append({"family": "incompressible_stokes", "space": "mixed_H1_velocity_pressure", "solver": "low_re_channel_stokes"})
    warnings = list(validation.warnings)
    if validation.errors:
        warnings.extend(validation.errors)
    warnings.extend(preparation_plan.warnings)
    warnings.extend(preparation_validation.warnings)
    if preparation_validation.errors:
        warnings.extend(preparation_validation.errors)
    if backend not in {"fenicsx", "mfem", "petsc", "openfoam", "su2", "generic"}:
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
            "preferred_full_solver_targets": ["openfoam", "su2"] if any(block.get("family") == "incompressible_navier_stokes_mesh" for block in blocks) else [],
        },
        "mesh_requirements": {
            "requires_backend_mesh_export_manifest": any(block.get("family") == "incompressible_navier_stokes_mesh" for block in blocks),
            "requires_boundary_tags": any(block.get("family") == "incompressible_navier_stokes_mesh" for block in blocks),
            "required_boundary_roles": ["inlet", "outlet", "wall"] if any(block.get("family") == "incompressible_navier_stokes_mesh" for block in blocks) else [],
        },
        "backend_preparation_plan": preparation_plan.model_dump(mode="json"),
        "backend_preparation_validation": preparation_validation.model_dump(mode="json"),
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
    if backend == "openfoam":
        return [
            {"name": "openfoam_runner_available", "command": "foamVersion", "required": False},
            {"name": "mesh_conversion_manifest_available", "command": "backend_mesh_export_manifest", "required": True},
        ]
    if backend == "su2":
        return [
            {"name": "su2_runner_available", "command": "SU2_CFD --help", "required": False},
            {"name": "mesh_conversion_manifest_available", "command": "backend_mesh_export_manifest", "required": True},
        ]
    return [{"name": "python_available", "command": "python --version", "required": True}]


class PlanBackendPreparationInput(StrictBaseModel):
    problem: PhysicsProblem
    taps_problem: TAPSProblem
    backend: str = "fenicsx"
    mesh_export_manifest: ArtifactRef | None = None


class ValidateBackendPreparationPlanInput(StrictBaseModel):
    problem: PhysicsProblem
    taps_problem: TAPSProblem
    plan: BackendPreparationPlanOutput
    mesh_export_manifest: ArtifactRef | None = None


class ValidateBackendPreparationPlanOutput(StrictBaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def plan_backend_preparation(input: PlanBackendPreparationInput) -> BackendPreparationPlanOutput:
    """Plan a reviewed backend bridge/case bundle without executing external solvers."""
    backend = input.backend.lower()
    backend_family = backend if backend in {"fenicsx", "mfem", "petsc", "openfoam", "su2"} else "generic"
    weak_form = input.taps_problem.weak_form
    has_mesh_bridge = supports_mesh_navier_stokes_bridge(input.taps_problem)
    mesh_required = backend_family in {"fenicsx", "mfem", "openfoam", "su2"} or has_mesh_bridge
    field_space_plan: list[dict[str, object]] = []
    if weak_form is not None:
        family = weak_form.family.lower()
        if has_mesh_bridge:
            field_space_plan.append(
                {
                    "family": "incompressible_navier_stokes_mesh",
                    "fields": weak_form.trial_fields,
                    "space": "mixed_velocity_pressure",
                    "backend_space": "VectorElement(P2)*FiniteElement(P1)" if backend_family == "fenicsx" else "solver_native_velocity_pressure",
                }
            )
        elif supports_hcurl_curl_curl_weak_form(input.taps_problem):
            field_space_plan.append({"family": "hcurl_curl_curl", "fields": weak_form.trial_fields, "space": "Nedelec_Hcurl"})
        elif supports_vector_elasticity_weak_form(input.taps_problem):
            field_space_plan.append({"family": "vector_elasticity", "fields": weak_form.trial_fields, "space": "vector_H1"})
        elif "reaction_diffusion" in family and len(weak_form.trial_fields) > 1:
            field_space_plan.append({"family": "coupled_reaction_diffusion", "fields": weak_form.trial_fields, "space": "mixed_H1"})
        else:
            field_space_plan.append({"family": family, "fields": weak_form.trial_fields, "space": "H1"})
    boundary_roles = []
    for boundary in input.problem.boundary_conditions:
        role = boundary.kind
        if boundary.kind in {"inlet", "outlet", "wall"}:
            role = boundary.kind
        boundary_roles.append(
            {
                "id": boundary.id,
                "field": boundary.field,
                "kind": boundary.kind,
                "region_id": boundary.region_id,
                "backend_tag": boundary.region_id,
                "required": boundary.kind in {"inlet", "outlet", "wall", "dirichlet"},
            }
        )
    stabilization_policy: dict[str, object] = {"required_review": False}
    if has_mesh_bridge or (weak_form is not None and "navier_stokes" in weak_form.family.lower()):
        stabilization_policy = {
            "required_review": True,
            "options": ["SUPG/PSPG", "projection", "SIMPLE/PISO", "solver_native_scheme"],
            "selected": "review_required_before_execution",
        }
    approval_gate = {
        "execute_external_solver": False,
        "requires_user_approval": True,
        "requires_dependency_checks": True,
        "requires_mesh_export_manifest": mesh_required,
        "execute_local_taps_kernel": False if has_mesh_bridge else None,
    }
    status = "ready"
    warnings: list[str] = []
    if mesh_required and input.mesh_export_manifest is None:
        status = "needs_inputs"
        warnings.append("backend preparation requires a backend mesh export manifest before execution")
    return BackendPreparationPlanOutput(
        problem_id=input.problem.id,
        status=status,  # type: ignore[arg-type]
        target_backend=backend,
        backend_family=backend_family,  # type: ignore[arg-type]
        field_space_plan=field_space_plan,
        mesh_export={
            "required": mesh_required,
            "provided": input.mesh_export_manifest is not None,
            "manifest_uri": input.mesh_export_manifest.uri if input.mesh_export_manifest is not None else None,
            "expected_kind": "backend_mesh_export_manifest",
        },
        coefficient_map=[
            {"name": coefficient.name, "role": coefficient.role, "region_ids": coefficient.region_ids}
            for coefficient in input.taps_problem.coefficients
        ],
        boundary_tag_map=boundary_roles,
        stabilization_policy=stabilization_policy,
        solver_controls={
            "external_execution_enabled": False,
            "review_required": True,
            "target_backend": backend,
        },
        dependency_checks=_dependency_checks_for_backend(backend),
        approval_gate=approval_gate,
        expected_artifacts=["taps_backend_bridge_manifest", "taps_backend_bridge_draft", "taps_backend_case_bundle"],
        validation_checks=["problem_id_matches", "backend_supported", "approval_gate_blocks_execution", "mesh_manifest_declared"],
        assumptions=["Backend preparation plan is deterministic fallback; external solver execution remains disabled."],
        warnings=warnings,
    )


def validate_backend_preparation_plan(input: ValidateBackendPreparationPlanInput) -> ValidateBackendPreparationPlanOutput:
    """Validate a backend preparation plan before bridge or case bundle emission."""
    errors: list[str] = []
    warnings: list[str] = []
    plan = input.plan
    if plan.problem_id != input.problem.id:
        errors.append(f"plan.problem_id={plan.problem_id!r} does not match problem.id={input.problem.id!r}.")
    if plan.backend_family not in {"fenicsx", "mfem", "petsc", "openfoam", "su2", "generic"}:
        errors.append(f"Unsupported backend_family={plan.backend_family!r}.")
    if plan.approval_gate.get("execute_external_solver") is not False:
        errors.append("Backend preparation plan must keep execute_external_solver=False.")
    if plan.solver_controls.get("external_execution_enabled") is not False:
        errors.append("Backend preparation plan must keep external_execution_enabled=False.")
    mesh_export = plan.mesh_export
    if mesh_export.get("required") and input.mesh_export_manifest is not None and mesh_export.get("manifest_uri") != input.mesh_export_manifest.uri:
        errors.append("Backend preparation plan mesh_export.manifest_uri does not match provided mesh_export_manifest.")
    if mesh_export.get("required") and not mesh_export.get("provided"):
        warnings.append("Backend preparation plan requires a mesh export manifest before execution.")
    boundary_ids = {boundary.id for boundary in input.problem.boundary_conditions}
    for item in plan.boundary_tag_map:
        item_id = str(item.get("id", ""))
        if item_id and item_id not in boundary_ids:
            errors.append(f"Backend preparation plan invented boundary tag mapping for {item_id!r}.")
    coefficient_names = {coefficient.name for coefficient in input.taps_problem.coefficients}
    for item in plan.coefficient_map:
        name = str(item.get("name", ""))
        if name and name not in coefficient_names:
            errors.append(f"Backend preparation plan invented coefficient mapping for {name!r}.")
    if supports_mesh_navier_stokes_bridge(input.taps_problem):
        roles = {str(item.get("kind", item.get("backend_tag", ""))).lower() for item in plan.boundary_tag_map}
        if not {"inlet", "outlet", "wall"}.issubset(roles):
            errors.append("Mesh Navier-Stokes backend preparation requires inlet, outlet, and wall boundary tag mappings.")
        if not plan.stabilization_policy.get("required_review"):
            errors.append("Mesh Navier-Stokes backend preparation must require stabilization review.")
    return ValidateBackendPreparationPlanOutput(valid=not errors, errors=errors, warnings=warnings)


BACKEND_PREPARATION_PLAN_SYSTEM_PROMPT = """You are the PhysicsOS backend preparation planning agent.
Return only a JSON object matching BackendPreparationPlanOutput.

Task:
- Prepare, but do not execute, a reviewed external solver backend case.
- Bind target backend, field spaces, mesh export manifest requirements, coefficient maps, boundary tags/patches, stabilization policy, solver controls, dependency checks, and approval gates.
- Never set execute_external_solver=true or external_execution_enabled=true.
- Preserve PhysicsProblem boundary IDs and TAPS coefficient names. Do not invent boundary tags or coefficients.
- If required inputs such as a backend mesh export manifest are missing, return status=needs_inputs with explicit warnings.
"""


def plan_backend_preparation_structured(
    input: PlanBackendPreparationInput,
    *,
    client: StructuredLLMClient,
    config: CoreAgentLLMConfig | None = None,
) -> BackendPreparationPlanOutput:
    """LLM-backed backend preparation planning with semantic validation and fallback."""
    def semantic_errors(plan: BackendPreparationPlanOutput) -> list[str]:
        validation = validate_backend_preparation_plan(
            ValidateBackendPreparationPlanInput(
                problem=input.problem,
                taps_problem=input.taps_problem,
                plan=plan,
                mesh_export_manifest=input.mesh_export_manifest,
            )
        )
        return validation.errors

    def retry_feedback(plan: BackendPreparationPlanOutput, errors: list[str]) -> list[str]:
        context = {
            "instruction": (
                "Repair only the invalid BackendPreparationPlanOutput. Preserve no-execute approval gates, "
                "PhysicsProblem boundary ids, TAPS coefficient names, mesh export manifest requirements, and target backend. "
                "Do not enable external execution."
            ),
            "errors": errors,
            "invalid_plan": plan.model_dump(mode="json"),
            "physics_problem": input.problem.model_dump(mode="json"),
            "taps_problem": input.taps_problem.model_dump(mode="json"),
            "mesh_export_manifest": input.mesh_export_manifest,
        }
        return [json.dumps(context, ensure_ascii=False)]

    result = call_structured_agent(
        agent_name="backend-preparation-planning-agent",
        input_model=input,
        output_model=BackendPreparationPlanOutput,
        system_prompt=BACKEND_PREPARATION_PLAN_SYSTEM_PROMPT,
        client=client,
        config=config,
        semantic_validator=semantic_errors,
        semantic_feedback_builder=retry_feedback,
    )
    if result.output is not None:
        validation = validate_backend_preparation_plan(
            ValidateBackendPreparationPlanInput(
                problem=input.problem,
                taps_problem=input.taps_problem,
                plan=result.output,
                mesh_export_manifest=input.mesh_export_manifest,
            )
        )
        return result.output.model_copy(
            update={"warnings": [*result.output.warnings, *validation.warnings]}
        )
    fallback = plan_backend_preparation(input)
    return fallback.model_copy(
        update={
            "assumptions": [
                *fallback.assumptions,
                "Structured LLM backend preparation planning failed validation; deterministic fallback was used.",
                result.error or "Structured backend preparation planner returned no validated output.",
            ]
        }
    )


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
    backend_preparation_plan: BackendPreparationPlanOutput | None = None


class PrepareTAPSBackendCaseBundleOutput(StrictBaseModel):
    bundle: ArtifactRef
    bridge_manifest: ArtifactRef
    draft_artifact: ArtifactRef | None = None
    warnings: list[str] = Field(default_factory=list)


def prepare_taps_backend_case_bundle(input: PrepareTAPSBackendCaseBundleInput) -> PrepareTAPSBackendCaseBundleOutput:
    """Prepare a reviewed backend execution bundle without running external solvers."""
    preparation_plan = input.backend_preparation_plan or plan_backend_preparation(
        PlanBackendPreparationInput(
            problem=input.problem,
            taps_problem=input.taps_problem,
            backend=input.backend,
            mesh_export_manifest=input.mesh_export_manifest,
        )
    )
    preparation_validation = validate_backend_preparation_plan(
        ValidateBackendPreparationPlanInput(
            problem=input.problem,
            taps_problem=input.taps_problem,
            plan=preparation_plan,
            mesh_export_manifest=input.mesh_export_manifest,
        )
    )
    bridge = export_taps_backend_bridge(
        ExportTAPSBackendBridgeInput(
            problem=input.problem,
            taps_problem=input.taps_problem,
            backend=input.backend,
            mesh_export_manifest=input.mesh_export_manifest,
            backend_preparation_plan=preparation_plan,
        )
    )
    fallback = plan_taps_adaptive_fallback(
        PlanTAPSAdaptiveFallbackInput(problem=input.problem, taps_problem=input.taps_problem, preferred_backend=input.backend)
    )
    backend = input.backend.lower()
    warnings = list(bridge.warnings)
    warnings.extend(preparation_validation.warnings)
    if preparation_validation.errors:
        warnings.extend(preparation_validation.errors)
    mesh_requirement = preparation_plan.mesh_export
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
        "backend_preparation_plan": preparation_plan.model_dump(mode="json"),
        "backend_preparation_validation": preparation_validation.model_dump(mode="json"),
        "dependency_checks": preparation_plan.dependency_checks,
        "mesh_export": mesh_requirement,
        "coefficient_binding": preparation_plan.coefficient_map,
        "boundary_binding": preparation_plan.boundary_tag_map,
        "field_space_plan": preparation_plan.field_space_plan,
        "stabilization_policy": preparation_plan.stabilization_policy,
        "solver_controls": preparation_plan.solver_controls,
        "approval_gate": preparation_plan.approval_gate,
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
    if input.budget.max_wall_time_seconds is not None:
        return _run_taps_backend_subprocess(input)
    return _run_taps_backend_local(input)


def _run_taps_backend_subprocess(input: RunTAPSBackendInput) -> RunTAPSBackendOutput:
    timeout_seconds = max(1.0, float(input.budget.max_wall_time_seconds or 1.0))
    payload = input.model_dump(mode="json")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
        input_path = Path(handle.name)
        json.dump(payload, handle)
    output_path = input_path.with_suffix(".out.json")
    script = (
        "import json, traceback\n"
        "from pathlib import Path\n"
        "from physicsos.tools.taps_tools import RunTAPSBackendInput, _run_taps_backend_local\n"
        f"input_path = Path({str(input_path)!r})\n"
        f"output_path = Path({str(output_path)!r})\n"
        "payload = json.loads(input_path.read_text(encoding='utf-8'))\n"
        "try:\n"
        "    result = _run_taps_backend_local(RunTAPSBackendInput.model_validate(payload))\n"
        "    output = {'ok': True, 'result': result.model_dump(mode='json')}\n"
        "except Exception as exc:\n"
        "    output = {'ok': False, 'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()}\n"
        "output_path.write_text(json.dumps(output), encoding='utf-8')\n"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(project_root()),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        if completed.returncode != 0 and not output_path.exists():
            stderr = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
            raise RuntimeError(f"TAPS backend subprocess failed: {stderr}")
        if not output_path.exists():
            raise RuntimeError("TAPS backend subprocess did not write an output payload.")
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("TAPS backend subprocess returned a non-object payload.")
        if not payload.get("ok"):
            raise RuntimeError(str(payload.get("error") or "TAPS backend subprocess failed."))
        return RunTAPSBackendOutput.model_validate(payload["result"])
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"TAPS backend exceeded wall-time budget of {timeout_seconds:.0f}s.") from exc
    finally:
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


def _run_taps_backend_local(input: RunTAPSBackendInput) -> RunTAPSBackendOutput:
    """Run the TAPS backend in-process. Use run_taps_backend for timeout enforcement."""
    started_at = time.monotonic()

    def remaining_budget() -> float | None:
        if input.budget.max_wall_time_seconds is None:
            return None
        return input.budget.max_wall_time_seconds - (time.monotonic() - started_at)

    def ensure_budget(stage: str) -> None:
        remaining = remaining_budget()
        if remaining is not None and remaining <= 0:
            raise TimeoutError(f"TAPS backend exceeded wall-time budget before {stage}.")

    def call_with_budget(stage: str, runner):
        ensure_budget(stage)
        output = runner()
        ensure_budget(stage)
        return output

    operator_classes = {operator.equation_class.lower() for operator in input.problem.operators}
    is_heat = bool(operator_classes.intersection({"heat", "diffusion", "thermal_diffusion"}))
    is_transient_diffusion_ir = supports_transient_diffusion_weak_form(input.taps_problem)
    if (is_heat or is_transient_diffusion_ir) and input.problem.initial_conditions:
        numerical_plan = input.numerical_plan or plan_numerical_solve(
            NumericalSolvePlanInput(problem=input.problem, taps_problem=input.taps_problem, budget=input.budget)
        )
        validation = validate_numerical_solve_plan(
            ValidateNumericalSolvePlanInput(problem=input.problem, taps_problem=input.taps_problem, plan=numerical_plan)
        )
        if not validation.valid:
            raise ValueError("Invalid numerical solve plan for transient diffusion: " + "; ".join(validation.errors))
        artifacts, residual_report = call_with_budget("transient_heat_1d", lambda: solve_transient_heat_1d(input.taps_problem, numerical_plan=numerical_plan))
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
                "numerical_plan_solver_family": numerical_plan.solver_family,
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
    is_hdiv_div_ir = supports_hdiv_div_weak_form(input.taps_problem)
    is_nonlinear_reaction_diffusion_ir = supports_nonlinear_reaction_diffusion_weak_form(input.taps_problem)
    is_navier_stokes_ir = supports_navier_stokes_weak_form(input.taps_problem)
    is_oseen_ir = supports_oseen_weak_form(input.taps_problem)
    is_scalar_elliptic_ir = supports_scalar_elliptic_weak_form(input.taps_problem)
    is_stokes_ir = supports_stokes_weak_form(input.taps_problem)
    is_vector_elasticity_ir = supports_vector_elasticity_weak_form(input.taps_problem)
    if (family in {"oseen", "navier_stokes", "incompressible_navier_stokes"} and is_oseen_ir) and space_axis_count == 2:
        numerical_plan = input.numerical_plan or plan_numerical_solve(
            NumericalSolvePlanInput(problem=input.problem, taps_problem=input.taps_problem, budget=input.budget)
        )
        validation = validate_numerical_solve_plan(
            ValidateNumericalSolvePlanInput(problem=input.problem, taps_problem=input.taps_problem, plan=numerical_plan)
        )
        if not validation.valid:
            raise ValueError("Invalid numerical solve plan for Oseen channel flow: " + "; ".join(validation.errors))
        artifacts, residual_report = call_with_budget("oseen_channel_2d", lambda: solve_oseen_channel_2d(input.taps_problem, numerical_plan=numerical_plan))
        artifact_refs = []
        artifact_refs.extend(artifacts.factor_matrices)
        if artifacts.reconstruction_metadata is not None:
            artifact_refs.append(artifacts.reconstruction_metadata)
        if artifacts.residual_history is not None:
            artifact_refs.append(artifacts.residual_history)
        result = SolverResult(
            id=f"result:{input.taps_problem.id}",
            problem_id=input.problem.id,
            backend=f"taps:oseen_channel_2d:{family}",
            status="success" if residual_report.converged else "needs_review",
            scalar_outputs={
                "message": "Linearized incompressible Oseen TAPS channel-flow kernel executed with a frozen convective velocity.",
                "equation_family": family,
                "weak_form_ir_blocks": is_oseen_ir,
                "full_navier_stokes_supported": 0,
                "simplification": "linearized_oseen_frozen_convection",
                "numerical_plan_solver_family": numerical_plan.solver_family,
                **residual_report.residuals,
            },
            residuals=residual_report.residuals,
            artifacts=artifact_refs,
            provenance=Provenance(created_by="run_taps_backend", source="taps_oseen_channel_2d"),
        )
        return RunTAPSBackendOutput(result=result)

    if (family in {"navier_stokes", "incompressible_navier_stokes"} and is_navier_stokes_ir) and space_axis_count == 2:
        numerical_plan = input.numerical_plan or plan_numerical_solve(
            NumericalSolvePlanInput(problem=input.problem, taps_problem=input.taps_problem, budget=input.budget)
        )
        validation = validate_numerical_solve_plan(
            ValidateNumericalSolvePlanInput(problem=input.problem, taps_problem=input.taps_problem, plan=numerical_plan)
        )
        if not validation.valid:
            raise ValueError("Invalid numerical solve plan for restricted Navier-Stokes channel flow: " + "; ".join(validation.errors))
        artifacts, residual_report = call_with_budget("navier_stokes_channel_2d", lambda: solve_navier_stokes_channel_2d(input.taps_problem, numerical_plan=numerical_plan))
        artifact_refs = []
        artifact_refs.extend(artifacts.factor_matrices)
        if artifacts.reconstruction_metadata is not None:
            artifact_refs.append(artifacts.reconstruction_metadata)
        if artifacts.residual_history is not None:
            artifact_refs.append(artifacts.residual_history)
        result = SolverResult(
            id=f"result:{input.taps_problem.id}",
            problem_id=input.problem.id,
            backend=f"taps:navier_stokes_channel_2d:{family}",
            status="success" if residual_report.converged else "needs_review",
            scalar_outputs={
                "message": "Restricted steady laminar incompressible Navier-Stokes TAPS channel kernel executed with Picard iteration.",
                "equation_family": family,
                "weak_form_ir_blocks": is_navier_stokes_ir,
                "full_navier_stokes_supported": 1,
                "support_scope": "restricted_steady_laminar_2d_channel",
                "nonlinear_solver": "picard_under_relaxation",
                "numerical_plan_solver_family": numerical_plan.solver_family,
                **residual_report.residuals,
            },
            residuals=residual_report.residuals,
            artifacts=artifact_refs,
            provenance=Provenance(created_by="run_taps_backend", source="taps_navier_stokes_channel_2d"),
        )
        return RunTAPSBackendOutput(result=result)

    if (family == "stokes" or is_stokes_ir) and space_axis_count == 2:
        numerical_plan = input.numerical_plan or plan_numerical_solve(
            NumericalSolvePlanInput(problem=input.problem, taps_problem=input.taps_problem, budget=input.budget)
        )
        validation = validate_numerical_solve_plan(
            ValidateNumericalSolvePlanInput(problem=input.problem, taps_problem=input.taps_problem, plan=numerical_plan)
        )
        if not validation.valid:
            raise ValueError("Invalid numerical solve plan for Stokes channel flow: " + "; ".join(validation.errors))
        artifacts, residual_report = call_with_budget("stokes_channel_2d", lambda: solve_stokes_channel_2d(input.taps_problem, numerical_plan=numerical_plan))
        artifact_refs = []
        artifact_refs.extend(artifacts.factor_matrices)
        if artifacts.reconstruction_metadata is not None:
            artifact_refs.append(artifacts.reconstruction_metadata)
        if artifacts.residual_history is not None:
            artifact_refs.append(artifacts.residual_history)
        result = SolverResult(
            id=f"result:{input.taps_problem.id}",
            problem_id=input.problem.id,
            backend=f"taps:{'weak_ir' if family not in {'stokes', 'navier_stokes', 'incompressible_navier_stokes'} else 'stokes'}_channel_2d:{family}",
            status="success" if residual_report.converged else "needs_review",
            scalar_outputs={
                "message": "Low-Re incompressible Stokes TAPS channel-flow kernel executed as a conservative Navier-Stokes simplification.",
                "equation_family": family,
                "weak_form_ir_blocks": is_stokes_ir,
                "full_navier_stokes_supported": 0,
                "simplification": "steady_low_re_stokes_no_convection",
                "numerical_plan_solver_family": numerical_plan.solver_family,
                **residual_report.residuals,
            },
            residuals=residual_report.residuals,
            artifacts=artifact_refs,
            provenance=Provenance(created_by="run_taps_backend", source="taps_stokes_channel_2d"),
        )
        return RunTAPSBackendOutput(result=result)

    if (family in {"maxwell", "curl_curl", "electromagnetic"} or is_hcurl_curl_curl_ir) and has_mesh_graph:
        numerical_plan = input.numerical_plan or plan_numerical_solve(
            NumericalSolvePlanInput(problem=input.problem, taps_problem=input.taps_problem, budget=input.budget)
        )
        validation = validate_numerical_solve_plan(
            ValidateNumericalSolvePlanInput(problem=input.problem, taps_problem=input.taps_problem, plan=numerical_plan)
        )
        if not validation.valid:
            raise ValueError("Invalid numerical solve plan for mesh FEM EM curl-curl: " + "; ".join(validation.errors))
        artifacts, residual_report = call_with_budget("mesh_fem_em_curl_curl", lambda: solve_mesh_fem_em_curl_curl(input.taps_problem, numerical_plan=numerical_plan))
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
                "numerical_plan_solver_family": numerical_plan.solver_family,
                **residual_report.residuals,
            },
            residuals=residual_report.residuals,
            artifacts=artifact_refs,
            provenance=Provenance(created_by="run_taps_backend", source="taps_mesh_fem_em_curl_curl"),
        )
        return RunTAPSBackendOutput(result=result)

    if (family in {"hdiv", "div", "darcy", "mixed_poisson"} or is_hdiv_div_ir) and has_mesh_graph:
        numerical_plan = input.numerical_plan or plan_numerical_solve(
            NumericalSolvePlanInput(problem=input.problem, taps_problem=input.taps_problem, budget=input.budget)
        )
        validation = validate_numerical_solve_plan(
            ValidateNumericalSolvePlanInput(problem=input.problem, taps_problem=input.taps_problem, plan=numerical_plan)
        )
        if not validation.valid:
            raise ValueError("Invalid numerical solve plan for mesh FEM H(div): " + "; ".join(validation.errors))
        artifacts, residual_report = call_with_budget("mesh_fem_hdiv_div", lambda: solve_mesh_fem_hdiv_div(input.taps_problem, numerical_plan=numerical_plan))
        artifact_refs = []
        artifact_refs.extend(artifacts.factor_matrices)
        if artifacts.reconstruction_metadata is not None:
            artifact_refs.append(artifacts.reconstruction_metadata)
        if artifacts.residual_history is not None:
            artifact_refs.append(artifacts.residual_history)
        result = SolverResult(
            id=f"result:{input.taps_problem.id}",
            problem_id=input.problem.id,
            backend=f"taps:{'weak_ir' if family not in {'hdiv', 'div', 'darcy', 'mixed_poisson'} else 'mesh_fem'}_hdiv_div:{family}",
            status="success" if residual_report.converged else "needs_review",
            scalar_outputs={
                "message": "Tetrahedral RT0 H(div) face-flux scaffold kernel executed from reusable divergence/mass/source blocks.",
                "equation_family": family,
                "weak_form_ir_blocks": is_hdiv_div_ir,
                "numerical_plan_solver_family": numerical_plan.solver_family,
                **residual_report.residuals,
            },
            residuals=residual_report.residuals,
            artifacts=artifact_refs,
            provenance=Provenance(created_by="run_taps_backend", source="taps_mesh_fem_hdiv_div"),
        )
        return RunTAPSBackendOutput(result=result)

    if (family in {"elasticity", "linear_elasticity"} or is_vector_elasticity_ir) and has_mesh_graph:
        numerical_plan = input.numerical_plan or plan_numerical_solve(
            NumericalSolvePlanInput(problem=input.problem, taps_problem=input.taps_problem, budget=input.budget)
        )
        validation = validate_numerical_solve_plan(
            ValidateNumericalSolvePlanInput(problem=input.problem, taps_problem=input.taps_problem, plan=numerical_plan)
        )
        if not validation.valid:
            raise ValueError("Invalid numerical solve plan for mesh FEM linear elasticity: " + "; ".join(validation.errors))
        artifacts, residual_report = call_with_budget("mesh_fem_linear_elasticity", lambda: solve_mesh_fem_linear_elasticity(input.taps_problem, numerical_plan=numerical_plan))
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
                "numerical_plan_solver_family": numerical_plan.solver_family,
                **residual_report.residuals,
            },
            residuals=residual_report.residuals,
            artifacts=artifact_refs,
            provenance=Provenance(created_by="run_taps_backend", source="taps_mesh_fem_linear_elasticity"),
        )
        return RunTAPSBackendOutput(result=result)

    if family in {"poisson", "diffusion", "thermal_diffusion"} and has_mesh_graph:
        numerical_plan = input.numerical_plan or plan_numerical_solve(
            NumericalSolvePlanInput(problem=input.problem, taps_problem=input.taps_problem, budget=input.budget)
        )
        validation = validate_numerical_solve_plan(
            ValidateNumericalSolvePlanInput(problem=input.problem, taps_problem=input.taps_problem, plan=numerical_plan)
        )
        if not validation.valid:
            raise ValueError("Invalid numerical solve plan for mesh FEM Poisson: " + "; ".join(validation.errors))
        source = "taps_mesh_fem_poisson"
        backend = f"taps:mesh_fem_poisson:{family}"
        message = "Mesh FEM TAPS Poisson kernel executed on mesh_graph geometry encoding."
        try:
            artifacts, residual_report = call_with_budget("mesh_fem_poisson", lambda: solve_mesh_fem_poisson(input.taps_problem, numerical_plan=numerical_plan))
        except ValueError:
            source = "taps_graph_poisson"
            backend = f"taps:graph_poisson:{family}"
            message = "Graph-Laplacian TAPS Poisson kernel executed on mesh_graph geometry encoding."
            artifacts, residual_report = call_with_budget("graph_poisson", lambda: solve_graph_poisson(input.taps_problem))
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
                "numerical_plan_solver_family": numerical_plan.solver_family,
                **residual_report.residuals,
            },
            residuals=residual_report.residuals,
            artifacts=artifact_refs,
            provenance=Provenance(created_by="run_taps_backend", source=source),
        )
        return RunTAPSBackendOutput(result=result)

    if (family == "coupled_reaction_diffusion" or is_coupled_reaction_diffusion_ir) and space_axis_count == 2 and field_count >= 2:
        numerical_plan = input.numerical_plan or plan_numerical_solve(
            NumericalSolvePlanInput(problem=input.problem, taps_problem=input.taps_problem, budget=input.budget)
        )
        validation = validate_numerical_solve_plan(
            ValidateNumericalSolvePlanInput(problem=input.problem, taps_problem=input.taps_problem, plan=numerical_plan)
        )
        if not validation.valid:
            raise ValueError("Invalid numerical solve plan for coupled reaction-diffusion: " + "; ".join(validation.errors))
        artifacts, residual_report = call_with_budget(
            "coupled_reaction_diffusion_2d",
            lambda: solve_coupled_reaction_diffusion_2d(input.taps_problem, numerical_plan=numerical_plan),
        )
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
                "numerical_plan_solver_family": numerical_plan.solver_family,
                **residual_report.residuals,
            },
            residuals=residual_report.residuals,
            artifacts=artifact_refs,
            provenance=Provenance(created_by="run_taps_backend", source="taps_coupled_reaction_diffusion_2d"),
        )
        return RunTAPSBackendOutput(result=result)

    if (family == "reaction_diffusion" or is_nonlinear_reaction_diffusion_ir) and space_axis_count == 1:
        numerical_plan = input.numerical_plan or plan_numerical_solve(
            NumericalSolvePlanInput(problem=input.problem, taps_problem=input.taps_problem, budget=input.budget)
        )
        plan_validation = validate_numerical_solve_plan(
            ValidateNumericalSolvePlanInput(problem=input.problem, taps_problem=input.taps_problem, plan=numerical_plan)
        )
        if not plan_validation.valid:
            raise ValueError("Numerical solve plan validation failed: " + "; ".join(plan_validation.errors))
        artifacts, residual_report = call_with_budget(
            "reaction_diffusion_nonlinear_1d",
            lambda: solve_reaction_diffusion_nonlinear_1d(input.taps_problem, numerical_plan=numerical_plan),
        )
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
                "numerical_plan_solver_family": numerical_plan.solver_family,
                **residual_report.residuals,
            },
            residuals=residual_report.residuals,
            artifacts=artifact_refs,
            provenance=Provenance(created_by="run_taps_backend", source="taps_nonlinear_reaction_diffusion_1d"),
        )
        return RunTAPSBackendOutput(result=result)

    if (family == "reaction_diffusion" or is_nonlinear_reaction_diffusion_ir) and space_axis_count == 2:
        numerical_plan = input.numerical_plan or plan_numerical_solve(
            NumericalSolvePlanInput(problem=input.problem, taps_problem=input.taps_problem, budget=input.budget)
        )
        plan_validation = validate_numerical_solve_plan(
            ValidateNumericalSolvePlanInput(problem=input.problem, taps_problem=input.taps_problem, plan=numerical_plan)
        )
        if not plan_validation.valid:
            raise ValueError("Numerical solve plan validation failed: " + "; ".join(plan_validation.errors))
        artifacts, residual_report = call_with_budget(
            "reaction_diffusion_nonlinear_2d",
            lambda: solve_reaction_diffusion_nonlinear_2d(input.taps_problem, numerical_plan=numerical_plan),
        )
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
                "numerical_plan_solver_family": numerical_plan.solver_family,
                **residual_report.residuals,
            },
            residuals=residual_report.residuals,
            artifacts=artifact_refs,
            provenance=Provenance(created_by="run_taps_backend", source="taps_nonlinear_reaction_diffusion_2d"),
        )
        return RunTAPSBackendOutput(result=result)

    if (family in GENERIC_TAPS_FAMILIES or is_scalar_elliptic_ir) and space_axis_count == 1:
        numerical_plan = input.numerical_plan or plan_numerical_solve(
            NumericalSolvePlanInput(problem=input.problem, taps_problem=input.taps_problem, budget=input.budget)
        )
        plan_validation = validate_numerical_solve_plan(
            ValidateNumericalSolvePlanInput(problem=input.problem, taps_problem=input.taps_problem, plan=numerical_plan)
        )
        if not plan_validation.valid:
            raise ValueError("Numerical solve plan validation failed: " + "; ".join(plan_validation.errors))
        artifacts, residual_report = call_with_budget(
            "scalar_elliptic_1d",
            lambda: solve_scalar_elliptic_1d(input.taps_problem, numerical_plan=numerical_plan),
        )
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
                "numerical_plan_solver_family": numerical_plan.solver_family,
                **residual_report.residuals,
            },
            residuals=residual_report.residuals,
            artifacts=artifact_refs,
            provenance=Provenance(created_by="run_taps_backend", source="taps_generic_scalar_elliptic_1d"),
        )
        return RunTAPSBackendOutput(result=result)

    if (family in GENERIC_TAPS_FAMILIES or is_scalar_elliptic_ir) and space_axis_count == 2:
        numerical_plan = input.numerical_plan or plan_numerical_solve(
            NumericalSolvePlanInput(problem=input.problem, taps_problem=input.taps_problem, budget=input.budget)
        )
        plan_validation = validate_numerical_solve_plan(
            ValidateNumericalSolvePlanInput(problem=input.problem, taps_problem=input.taps_problem, plan=numerical_plan)
        )
        if not plan_validation.valid:
            raise ValueError("Numerical solve plan validation failed: " + "; ".join(plan_validation.errors))
        artifacts, residual_report = call_with_budget(
            "scalar_elliptic_2d",
            lambda: solve_scalar_elliptic_2d(input.taps_problem, numerical_plan=numerical_plan),
        )
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
                "numerical_plan_solver_family": numerical_plan.solver_family,
                **residual_report.residuals,
            },
            residuals=residual_report.residuals,
            artifacts=artifact_refs,
            provenance=Provenance(created_by="run_taps_backend", source="taps_generic_scalar_elliptic_2d"),
        )
        return RunTAPSBackendOutput(result=result)

    if (family in GENERIC_TAPS_FAMILIES or is_scalar_elliptic_ir) and space_axis_count == 3:
        numerical_plan = input.numerical_plan or plan_numerical_solve(
            NumericalSolvePlanInput(problem=input.problem, taps_problem=input.taps_problem, budget=input.budget)
        )
        plan_validation = validate_numerical_solve_plan(
            ValidateNumericalSolvePlanInput(problem=input.problem, taps_problem=input.taps_problem, plan=numerical_plan)
        )
        if not plan_validation.valid:
            raise ValueError("Numerical solve plan validation failed: " + "; ".join(plan_validation.errors))
        artifacts, residual_report = call_with_budget(
            "scalar_elliptic_3d",
            lambda: solve_scalar_elliptic_3d(input.taps_problem, numerical_plan=numerical_plan),
        )
        artifact_refs = []
        artifact_refs.extend(artifacts.factor_matrices)
        if artifacts.reconstruction_metadata is not None:
            artifact_refs.append(artifacts.reconstruction_metadata)
        if artifacts.residual_history is not None:
            artifact_refs.append(artifacts.residual_history)
        result = SolverResult(
            id=f"result:{input.taps_problem.id}",
            problem_id=input.problem.id,
            backend=f"taps:{'weak_ir' if family not in GENERIC_TAPS_FAMILIES else 'generic'}_scalar_elliptic_3d:{family}",
            status="success" if residual_report.converged else "needs_review",
            scalar_outputs={
                "message": "TAPS scalar 3D structured-grid weak-form kernel executed from reusable diffusion/reaction/source blocks.",
                "equation_family": family,
                "weak_form_ir_blocks": is_scalar_elliptic_ir,
                "tensor_rank": residual_report.rank,
                "numerical_plan_solver_family": numerical_plan.solver_family,
                **residual_report.residuals,
            },
            residuals=residual_report.residuals,
            artifacts=artifact_refs,
            provenance=Provenance(created_by="run_taps_backend", source="taps_generic_scalar_elliptic_3d"),
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
            converged = normalized < 1e-6 or update < max(2e-2, input.taps_problem.nonlinear.tolerance)
            recommended_action = "accept" if converged else "refine_axes"
        elif "normalized_linear_residual" in input.result.residuals:
            normalized = input.result.residuals["normalized_linear_residual"]
            tolerance = 1e-8 if (
                input.result.residuals.get("masked_relaxation_iterations", 0.0)
                or input.result.residuals.get("structured_grid_iterations", 0.0)
            ) else 1e-10
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
    (plan_numerical_solve, NumericalSolvePlanInput, NumericalSolvePlanOutput),
    (plan_numerical_solve_structured, NumericalSolvePlanInput, NumericalSolvePlanOutput),
    (validate_numerical_solve_plan, ValidateNumericalSolvePlanInput, ValidateNumericalSolvePlanOutput),
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
