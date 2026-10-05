from physicsos.schemas.rsi import (
    AssessCapabilityInput, BenchmarkSuiteInput, BindCaseStrategyInput, BindCaseStrategyOutput, CapabilityEstimate,
    EvaluateStrategiesInput, EvaluateStrategiesOutput, PromoteStrategyInput,
    PromotionOutput, RegisterStrategyOutput, RegisterSuiteOutput, RollbackOutput,
    RollbackStrategyInput, SolveWithStrategyInput, SolveWithStrategyOutput, StrategySpec,
    RevisionProviderSpec, RegisterRevisionProviderOutput, ImproveStrategiesInput, ImproveStrategiesOutput,
)


def register_rsi_strategy(input: StrategySpec) -> RegisterStrategyOutput:
    """Freeze a reusable mesh/repair/kernel or generation strategy in an explicit physics scope."""
    from physicsos.rsi import RSIRuntime
    return RSIRuntime().register_strategy(input)


def register_rsi_benchmark_suite(input: BenchmarkSuiteInput) -> RegisterSuiteOutput:
    """Freeze distinct development/holdout problems, independent references and acceptance thresholds."""
    from physicsos.rsi import RSIRuntime
    return RSIRuntime().register_suite(input)


def register_rsi_revision_provider(input: RevisionProviderSpec) -> RegisterRevisionProviderOutput:
    """Freeze a revise_strategy(config) provider that proposes reusable changes from development feedback."""
    from physicsos.rsi import RSIRuntime
    return RSIRuntime().register_revision_provider(input)


def improve_rsi_strategy(input: ImproveStrategiesInput) -> ImproveStrategiesOutput:
    """Run a bounded revision campaign, reuse CaseRuntime evidence and check holdouts only after selection."""
    from physicsos.rsi import RSIRuntime
    return RSIRuntime().improve(input)


def evaluate_rsi_candidates(input: EvaluateStrategiesInput) -> EvaluateStrategiesOutput:
    """Use CaseRuntime for real candidate/baseline solves, then evaluate the selected candidate on fresh holdouts."""
    from physicsos.rsi import RSIRuntime
    return RSIRuntime().evaluate(input)


def assess_rsi_capability(input: AssessCapabilityInput) -> CapabilityEstimate:
    """Estimate scope-specific verification success from distinct, intact holdout/production evidence."""
    from physicsos.rsi import RSIRuntime
    return RSIRuntime().assess(input)


def bind_rsi_case_context(input: BindCaseStrategyInput) -> BindCaseStrategyOutput:
    """Bind a case to learned strategy guidance, current generation and capability evidence."""
    from physicsos.rsi import RSIRuntime
    return RSIRuntime().bind_context(input)


def solve_with_rsi_strategy(input: SolveWithStrategyInput) -> SolveWithStrategyOutput:
    """Apply the promoted strategy through CaseRuntime; independently verify and monitor regression rollback."""
    from physicsos.rsi import RSIRuntime
    return RSIRuntime().solve(input)


def promote_rsi_strategy(input: PromoteStrategyInput) -> PromotionOutput:
    """Recheck an eligible evaluation and atomically adopt its strategy if the baseline generation is current."""
    from physicsos.rsi import RSIRuntime
    return RSIRuntime().promote(input)


def rollback_rsi_strategy(input: RollbackStrategyInput) -> RollbackOutput:
    """Restore the preceding valid default strategy for an exact registry generation."""
    from physicsos.rsi import RSIRuntime
    return RSIRuntime().rollback(input)


RSI_TOOLS = [register_rsi_strategy, register_rsi_benchmark_suite, evaluate_rsi_candidates,
             register_rsi_revision_provider, improve_rsi_strategy, assess_rsi_capability,
             bind_rsi_case_context, solve_with_rsi_strategy, promote_rsi_strategy, rollback_rsi_strategy]
for function, input_model, output_model in [
    (register_rsi_strategy, StrategySpec, RegisterStrategyOutput),
    (register_rsi_benchmark_suite, BenchmarkSuiteInput, RegisterSuiteOutput),
    (register_rsi_revision_provider, RevisionProviderSpec, RegisterRevisionProviderOutput),
    (improve_rsi_strategy, ImproveStrategiesInput, ImproveStrategiesOutput),
    (evaluate_rsi_candidates, EvaluateStrategiesInput, EvaluateStrategiesOutput),
    (assess_rsi_capability, AssessCapabilityInput, CapabilityEstimate),
    (bind_rsi_case_context, BindCaseStrategyInput, BindCaseStrategyOutput),
    (solve_with_rsi_strategy, SolveWithStrategyInput, SolveWithStrategyOutput),
    (promote_rsi_strategy, PromoteStrategyInput, PromotionOutput),
    (rollback_rsi_strategy, RollbackStrategyInput, RollbackOutput),
]:
    function.input_model = input_model
    function.output_model = output_model
    function.side_effects = "versioned RSI evidence, strategy registry and isolated CaseRuntime attempts"
