from physicsos.workflows import run_physicsos_workflow


def main() -> None:
    result = run_physicsos_workflow(taps_rank=8, arxiv_max_results=0)
    print(f"problem={result.problem.id}")
    print(f"backend={result.solver_result.backend}")
    print(f"verification={result.verification.status}")
    print(f"report={result.postprocess.report.uri if result.postprocess.report else 'none'}")
    print("trace=" + " > ".join(step.name for step in result.trace))


if __name__ == "__main__":
    main()
