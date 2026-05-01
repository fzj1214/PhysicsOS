from __future__ import annotations

from pydantic import Field

from physicsos.schemas.boundary import BoundaryConditionSpec, InitialConditionSpec
from physicsos.schemas.common import ParameterSpec, Provenance, StrictBaseModel, TargetSpec, UserIntent
from physicsos.schemas.geometry import GeometryEncoding, GeometrySource, GeometrySpec
from physicsos.schemas.materials import MaterialProperty, MaterialSpec
from physicsos.schemas.operators import FieldSpec, OperatorSpec
from physicsos.schemas.problem import PhysicsProblem
from physicsos.schemas.solver import SolverResult
from physicsos.schemas.taps import TAPSProblem, TAPSResidualReport, TAPSBasisConfig
from physicsos.schemas.verification import VerificationReport
from physicsos.tools.knowledge_tools import BuildKnowledgeContextInput, build_knowledge_context
from physicsos.tools.taps_tools import (
    BuildTAPSProblemInput,
    EstimateTAPSResidualInput,
    EstimateTAPSSupportInput,
    RunTAPSBackendInput,
    build_taps_problem,
    estimate_taps_residual,
    estimate_taps_support,
    run_taps_backend,
)


class TapsThermalWorkflowResult(StrictBaseModel):
    problem: PhysicsProblem
    taps_problem: TAPSProblem
    result: SolverResult
    residual: TAPSResidualReport
    verification: VerificationReport
    knowledge_titles: list[str] = Field(default_factory=list)


def build_default_thermal_problem() -> PhysicsProblem:
    geometry = GeometrySpec(
        id="geometry:unit-slab-1d",
        source=GeometrySource(kind="generated"),
        dimension=1,
        encodings=[GeometryEncoding(kind="parametric_shape_vector", uri="generated:unit_slab")],
    )
    return PhysicsProblem(
        id="problem:taps-thermal-1d",
        user_intent=UserIntent(raw_request="Solve a parameterized 1D transient heat equation with TAPS."),
        domain="thermal",
        geometry=geometry,
        fields=[FieldSpec(name="T", kind="scalar", units="K", location="node")],
        operators=[
            OperatorSpec(
                id="operator:heat-1d",
                name="Transient heat equation",
                domain="thermal",
                equation_class="heat",
                form="weak",
                fields_in=["T", "alpha"],
                fields_out=["T"],
                conserved_quantities=[],
                assumptions=[
                    "1D slab",
                    "zero Dirichlet boundaries",
                    "sinusoidal initial condition",
                    "thermal energy is not globally conserved with fixed-temperature Dirichlet boundaries",
                ],
            )
        ],
        materials=[
            MaterialSpec(
                id="material:generic-diffusive",
                name="Generic diffusive material",
                phase="solid",
                properties=[
                    MaterialProperty(name="thermal_diffusivity", value="alpha", units="m^2/s"),
                ],
            )
        ],
        boundary_conditions=[
            BoundaryConditionSpec(id="bc:left", region_id="x=0", field="T", kind="dirichlet", value=0.0, units="K"),
            BoundaryConditionSpec(id="bc:right", region_id="x=L", field="T", kind="dirichlet", value=0.0, units="K"),
        ],
        initial_conditions=[
            InitialConditionSpec(id="ic:sin", field="T", value={"expression": "sin(pi*x/L)", "language": "text"}, units="K")
        ],
        parameters=[
            ParameterSpec(name="alpha", value=0.05, units="m^2/s", description="thermal diffusivity parameter axis"),
        ],
        targets=[TargetSpec(name="temperature_field", field="T", objective="observe")],
        provenance=Provenance(created_by="taps_thermal_workflow"),
    )


def run_taps_thermal_workflow(rank: int = 8, use_knowledge: bool = True) -> TapsThermalWorkflowResult:
    problem = build_default_thermal_problem()
    knowledge_titles: list[str] = []
    if use_knowledge:
        context = build_knowledge_context(
            BuildKnowledgeContextInput(
                query="TAPS transient heat equation tensor decomposition Galerkin",
                local_top_k=4,
                arxiv_max_results=0,
                use_deepsearch=False,
            )
        ).context
        knowledge_titles = [chunk.source.title for chunk in context.chunks]

    support = estimate_taps_support(EstimateTAPSSupportInput(problem=problem)).support
    if not support.supported:
        raise RuntimeError(f"TAPS support rejected unexpectedly: {support.risks}")

    taps_problem = build_taps_problem(
        BuildTAPSProblemInput(problem=problem, basis=TAPSBasisConfig(tensor_rank=rank, reproducing_order=2))
    ).taps_problem
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result
    residual = estimate_taps_residual(EstimateTAPSResidualInput(problem=problem, taps_problem=taps_problem, result=result)).report
    verification = VerificationReport(
        problem_id=problem.id,
        result_id=result.id,
        status="accepted" if residual.converged else "accepted_with_warnings",
        residuals=residual.residuals,
        uncertainty={"rank_truncation_proxy": residual.residuals.get("relative_l2_reconstruction_error", 1.0)},
        ood_score=0.0,
        recommended_next_action="accept" if residual.converged else "run_higher_fidelity_solver",
        explanation="TAPS thermal MVP verified by low-rank reconstruction and sampled PDE residual.",
    )
    return TapsThermalWorkflowResult(
        problem=problem,
        taps_problem=taps_problem,
        result=result,
        residual=residual,
        verification=verification,
        knowledge_titles=knowledge_titles,
    )
