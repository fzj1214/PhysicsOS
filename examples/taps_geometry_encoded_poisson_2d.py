from physicsos.schemas.common import Provenance
from physicsos.schemas.geometry import GeometrySource, GeometrySpec
from physicsos.schemas.operators import FieldSpec, OperatorSpec
from physicsos.schemas.problem import PhysicsProblem
from physicsos.tools.geometry_tools import GenerateGeometryEncodingInput, generate_geometry_encoding
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
    geometry = GeometrySpec(id="geometry:encoded-square", source=GeometrySource(kind="generated"), dimension=2)
    encodings = generate_geometry_encoding(
        GenerateGeometryEncodingInput(geometry=geometry, encodings=["occupancy_mask"], resolutions=[[32, 32]])
    ).encodings
    geometry.encodings.extend(encodings)
    return PhysicsProblem(
        id="problem:geometry-encoded-poisson-2d-demo",
        user_intent={"raw_request": "solve a 2D Poisson equation using a geometry occupancy mask"},
        domain="custom",
        geometry=geometry,
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
        provenance=Provenance(created_by="taps_geometry_encoded_poisson_2d_example"),
    )


def main() -> None:
    problem = build_problem()
    plan = formulate_taps_equation(FormulateTAPSEquationInput(problem=problem)).plan
    taps_problem = build_taps_problem(BuildTAPSProblemInput(problem=problem, compilation_plan=plan)).taps_problem
    result = run_taps_backend(RunTAPSBackendInput(problem=problem, taps_problem=taps_problem)).result
    residual = estimate_taps_residual(EstimateTAPSResidualInput(problem=problem, taps_problem=taps_problem, result=result)).report
    print(f"backend={result.backend}")
    print(f"status={result.status}")
    print(f"geometry_encodings={len(taps_problem.geometry_encodings)}")
    print(f"converged={residual.converged}")
    print(f"residuals={residual.residuals}")
    print("artifacts:")
    for artifact in result.artifacts:
        print(f"  {artifact.kind}: {artifact.uri}")


if __name__ == "__main__":
    main()
