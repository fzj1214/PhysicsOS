from physicsos.schemas.case_runtime import (
    ConvergenceStudyInput, ConvergenceStudyOutput, ExecuteCaseInput, ExecuteCaseOutput,
    PrepareDomainInput, PrepareDomainOutput, VerifyCaseInput, VerifyCaseOutput,
    SearchRuntimeHistoryInput, SearchRuntimeHistoryOutput,
)


def prepare_simulation_domain(input: PrepareDomainInput) -> PrepareDomainOutput:
    """Prepare imported assets/meshes against kernel requirements; inspect readiness before solving."""
    from physicsos.runtime import CaseRuntime
    return CaseRuntime().prepare(input)


def execute_case_kernel(input: ExecuteCaseInput) -> ExecuteCaseOutput:
    """Run run_case(config) with an immutable prepared domain and fresh output directory."""
    from physicsos.runtime import CaseRuntime
    return CaseRuntime().execute(input)


def verify_case_solution(input: VerifyCaseInput) -> VerifyCaseOutput:
    """Compare actual field artifacts to separately authored reference code; missing evidence is uncertain."""
    from physicsos.runtime import CaseRuntime
    return CaseRuntime().verify(input)


def run_case_convergence(input: ConvergenceStudyInput) -> ConvergenceStudyOutput:
    """Rebuild discretizations and rerun the frozen case-local kernel, measuring real field errors."""
    from physicsos.runtime import CaseRuntime
    return CaseRuntime().convergence(input)


def search_runtime_history(input: SearchRuntimeHistoryInput) -> SearchRuntimeHistoryOutput:
    """Retrieve real successful/failed/uncertain attempts by physics and discretization features."""
    from physicsos.runtime import CaseRuntime
    return CaseRuntime().history(input)


CASE_RUNTIME_TOOLS = [prepare_simulation_domain, execute_case_kernel, verify_case_solution, run_case_convergence, search_runtime_history]
for function, input_model, output_model in [
    (prepare_simulation_domain, PrepareDomainInput, PrepareDomainOutput),
    (execute_case_kernel, ExecuteCaseInput, ExecuteCaseOutput),
    (verify_case_solution, VerifyCaseInput, VerifyCaseOutput),
    (run_case_convergence, ConvergenceStudyInput, ConvergenceStudyOutput),
    (search_runtime_history, SearchRuntimeHistoryInput, SearchRuntimeHistoryOutput),
]:
    function.input_model = input_model
    function.output_model = output_model
    function.side_effects = "versioned case artifacts and runtime history"
