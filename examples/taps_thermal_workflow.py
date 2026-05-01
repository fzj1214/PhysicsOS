from physicsos.workflows import run_taps_thermal_workflow


def main() -> None:
    result = run_taps_thermal_workflow(rank=8)
    print(f"backend={result.result.backend}")
    print(f"status={result.result.status}")
    print(f"converged={result.residual.converged}")
    print(f"residuals={result.residual.residuals}")
    print("artifacts:")
    for artifact in result.result.artifacts:
        print(f"  {artifact.kind}: {artifact.uri}")


if __name__ == "__main__":
    main()

