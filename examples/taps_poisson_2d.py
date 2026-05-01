from physicsos.schemas.common import Provenance
from physicsos.schemas.geometry import GeometrySource, GeometrySpec
from physicsos.schemas.operators import FieldSpec, OperatorSpec
from physicsos.schemas.problem import PhysicsProblem
from physicsos.tools.taps_tools import (
    BuildTAPSProblemInput,
    EstimateTAPSResidualInput,
    FormulateTAPSEquationInput,
    RunTAPSBackendInput,
    build_taps_problem,
    estimate_taps_residual,
    formulate_taps_equation,
    run_taps_backend,
)


def build_problem() -> PhysicsProblem:
    return PhysicsProblem(
        id="problem:poisson-2d-demo",
        user_intent={"raw_request": "solve a 2D Poisson equation on a square with homogeneous Dirichlet boundaries"},
        domain="custom",
        geometry=GeometrySpec(id="geometry:square", source=GeometrySource(kind="generated"), dimension=2),
        fields=[FieldSpec(name="u", kind="scalar")],
        operators=[
            OperatorSpec(
                id="operator:poisson",
                name="Poisson",
                domain="custom",
                equation_class="poisson",
                form="weak",
                fields_out=["u"],
            )
        ],
        materials=[],
        boundary_conditions=[{"id": "bc:u", "region_id": "boundary", "field": "u", "kind": "dirichlet", "value": 0.0}],
        targets=[{"name": "field", "field": "u", "objective": "observe"}],
        provenance=Provenance(created_by="taps_poisson_2d_example"),
    )


def main() -> None:
    problem = build_problem()
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result
    residual = estimate_taps_residual(EstimateTAPSResidualInput(problem=problem, taps_problem=taps_problem, result=result)).report
    print(f"backend={result.backend}")
    print(f"status={result.status}")
    print(f"converged={residual.converged}")
    print(f"separable_rank={residual.rank}")
    print(f"residuals={residual.residuals}")
    print("artifacts:")
    for artifact in result.artifacts:
        print(f"  {artifact.kind}: {artifact.uri}")


if __name__ == "__main__":
    main()
