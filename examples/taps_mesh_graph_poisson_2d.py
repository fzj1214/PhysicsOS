from physicsos.schemas.common import Provenance
from physicsos.schemas.geometry import GeometrySource, GeometrySpec
from physicsos.schemas.mesh import MeshPolicy
from physicsos.schemas.operators import FieldSpec, OperatorSpec, PhysicsSpec
from physicsos.schemas.problem import PhysicsProblem
from physicsos.tools.geometry_tools import (
    GenerateGeometryEncodingInput,
    GenerateMeshInput,
    generate_geometry_encoding,
    generate_mesh,
)
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
    geometry = GeometrySpec(id="geometry:mesh-graph-square-demo", source=GeometrySource(kind="generated"), dimension=2)
    mesh = generate_mesh(
        GenerateMeshInput(
            geometry=geometry,
            physics=PhysicsSpec(domains=["custom"]),
            mesh_policy=MeshPolicy(target_element_size=0.35),
            target_backends=["taps"],
        )
    ).mesh
    encodings = generate_geometry_encoding(
        GenerateGeometryEncodingInput(geometry=geometry, mesh=mesh, encodings=["mesh_graph"])
    ).encodings
    geometry.encodings.extend(encodings)
    return PhysicsProblem(
        id="problem:mesh-graph-poisson-2d-demo",
        user_intent={"raw_request": "solve a 2D Poisson equation on a Gmsh mesh graph"},
        domain="custom",
        geometry=geometry,
        mesh=mesh,
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
        provenance=Provenance(created_by="taps_mesh_graph_poisson_2d_example"),
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
    print(f"fem_nodes={result.residuals.get('fem_nodes')}")
    print(f"fem_triangles={result.residuals.get('fem_triangles')}")
    print(f"fem_nonzeros={result.residuals.get('fem_nonzeros')}")
    print(f"residuals={residual.residuals}")
    print("artifacts:")
    for artifact in result.artifacts:
        print(f"  {artifact.kind}: {artifact.uri}")


if __name__ == "__main__":
    main()
