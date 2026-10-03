# PhysicsOS Interface & Alignment Specification

**Version**: 0.1.30  
**Purpose**: Define interfaces, contracts, and alignment constraints for trustworthy autonomous physics simulation

---

## 1. Core Interface Contracts

### 1.1 TAPS IR Schema

**Location**: `physicsos/schemas/problem.py`

All physics problems MUST be representable as structured TAPS IR before code generation.

**Required Fields**:
```python
PhysicsProblem:
    id: str                                 # Unique identifier
    domain: str                             # Physics domain (fluid, solid, thermal, etc.)
    geometry: GeometrySpec
    mesh: MeshSpec | None
    fields: list[FieldSpec]                 # Solution variables
    operators: list[OperatorSpec]           # PDEs
    materials: list[MaterialSpec]
    boundary_conditions: list[BoundaryCondition]
    targets: list[Target]                   # Objectives
```

**Alignment Constraints**:
- Every field must declare its `kind` (scalar, vector, tensor)
- Every operator must list `conserved_quantities` (mass, momentum, energy, etc.)
- Boundary conditions must reference valid `field` names
- Materials must specify `phase` (solid, liquid, gas, plasma)

**Validation**:
```python
problem = PhysicsProblem.model_validate(data)  # Pydantic validation
assert all(bc.field in [f.name for f in problem.fields] for bc in problem.boundary_conditions)
```

---

### 1.2 Solver Result Schema

**Location**: `physicsos/schemas/solver.py`

**Required Fields**:
```python
SolverResult:
    id: str
    backend: str                            # Solver used (fenics, jax, numpy, etc.)
    status: str                             # "success" | "failed" | "timeout"
    script: str | None                      # Generated code
    output_path: str | None                 # Where artifacts saved
    residuals: dict[str, float]             # Final residuals
    uncertainty: dict[str, float] | None    # UQ metrics (if available)
```

**Alignment Constraints**:
- `status="success"` requires `script` and `output_path` to be set
- `residuals` must include all fields from the problem
- If `uncertainty` provided, must cover all target quantities

---

### 1.3 Verification Interface

**Location**: `physicsos/verification/base.py`

All verification methods MUST implement:

```python
class Verifier(ABC):
    @abstractmethod
    def verify(
        self, 
        problem: PhysicsProblem, 
        result: SolverResult
    ) -> VerificationResult:
        """
        Independently verify a solution.
        
        MUST NOT use the same model/method that generated the solution.
        MUST return quantitative metrics.
        MUST be deterministic.
        """
        pass
    
    @abstractmethod
    def required_data(self) -> list[str]:
        """Declare dependencies (e.g., ["field_values", "mesh"])."""
        pass
```

**VerificationResult Schema**:
```python
VerificationResult:
    verifier_name: str
    status: VerificationStatus             # VERIFIED | FAILED | UNCERTAIN
    metrics: dict[str, float]              # Quantitative results
    message: str                           # Human-readable summary
    details: dict[str, Any]                # Full diagnostic data
    timestamp: datetime
    compute_time: float
```

**Alignment Constraints**:

1. **Independence**: 
   - Verification MUST NOT reuse the code generator's outputs as truth
   - Use different discretization, different solver, or analytical comparison
   
2. **Quantitative**:
   - Every check MUST produce numerical metrics (not just pass/fail)
   - Example: `{"mass_conservation_error": 1.2e-10}`
   
3. **Determinism**:
   - Same `(problem, result)` → same `VerificationResult`
   - No randomness in verification logic
   
4. **Diagnostic Requirement**:
   - If `status=FAILED`, `message` and `details` MUST explain why
   - Details MUST be actionable (e.g., "mass flux at inlet: 1.5 kg/s, outlet: 1.2 kg/s")

5. **Status Semantics**:
   ```
   VERIFIED:   All checks passed within tolerance
   FAILED:     At least one check failed
   UNCERTAIN:  Cannot determine (missing data, inconclusive)
   UNVERIFIED: Not yet checked
   VALIDATED:  Also compared with experimental data
   ```

---

## 2. Verification Specifications

### 2.1 Conservation Checking

**Verifier**: `ConservationChecker`  
**Location**: `physicsos/verification/conservation.py`

**Purpose**: Verify fundamental conservation laws.

**Checks**:

1. **Mass Conservation**:
   ```
   Integral constraint: d/dt ∫∫∫ ρ dV + ∫∫ ρ(u·n) dS = ∫∫∫ source dV
   Tolerance: relative error < 1e-10 (default)
   ```

2. **Momentum Conservation**:
   ```
   Integral constraint: ∑F_ext = d/dt ∫∫∫ ρu dV + ∫∫ ρu(u·n) dS
   Tolerance: relative error < 1e-9 (default)
   ```

3. **Energy Conservation**:
   ```
   Integral constraint: Energy balance
   Tolerance: relative error < 1e-8 (default)
   ```

**Interface**:
```python
checker = ConservationChecker(
    mass_tolerance=1e-10,
    momentum_tolerance=1e-9,
    energy_tolerance=1e-8
)
result = checker.verify(problem, solver_result)
```

**Output Metrics**:
- `mass_relative_error`: |imbalance| / total_mass
- `momentum_relative_error`: |imbalance| / total_momentum
- `energy_relative_error`: |imbalance| / total_energy

**Alignment Constraints**:
- Tolerance must be physically motivated (not arbitrary)
- Must account for boundary fluxes correctly
- Must handle source/sink terms

---

### 2.2 Convergence Checking

**Verifier**: `ConvergenceChecker`  
**Location**: `physicsos/verification/convergence.py`

**Purpose**: Verify numerical convergence through systematic refinement.

**Checks**:

1. **Mesh Convergence**:
   ```
   Run solver on progressively refined meshes: h, h/2, h/4, ...
   Compute observed convergence rate: p_obs = log(e_coarse / e_fine) / log(r)
   Compare to expected rate: |p_obs - p_expected| < tolerance
   ```

2. **Richardson Extrapolation**:
   ```
   Estimate exact solution: f_exact ≈ (r^p * f_fine - f_coarse) / (r^p - 1)
   ```

**Interface**:
```python
checker = ConvergenceChecker(
    refinement_levels=3,
    refinement_factor=2.0,
    rate_tolerance=0.5  # Allow 0.5 order deviation
)
result = checker.verify(problem, solver_result)
```

**Output Metrics**:
- `convergence_rate`: Observed order of convergence
- `expected_rate`: Expected from method (e.g., 2.0 for P1 FEM)
- `rate_deviation`: |observed - expected|
- `extrapolated_error`: Richardson extrapolation estimate

**Alignment Constraints**:
- Refinement must be systematic (not random)
- Must test at least 3 refinement levels
- Must report extrapolated error for uncertainty quantification

---

### 2.3 Aggregate Verification

**Location**: `physicsos/verification/pipeline.py::VerificationPipeline`

**Purpose**: Orchestrate multiple verifiers and aggregate results.

**Interface**:
```python
pipeline = VerificationPipeline()
pipeline.register(ConservationChecker())
pipeline.register(ConvergenceChecker())

report = pipeline.verify(problem, solver_result, confidence=conf_score)
# → AggregateVerificationReport
```

**AggregateVerificationReport Schema**:
```python
AggregateVerificationReport:
    problem_id: str
    result_id: str
    overall_status: VerificationStatus
    individual_results: dict[str, VerificationResult]
    passed_checks: int
    failed_checks: int
    uncertain_checks: int
    confidence: ConfidenceScore           # From capability assessment
    failure_mode: str | None              # If failed, primary cause
    similar_cases: list[str]              # Knowledge base links
    timestamp: datetime
    total_compute_time: float
```

**Status Aggregation Rules**:
```
Overall VERIFIED:   All individual checks VERIFIED
Overall FAILED:     At least one FAILED
Overall UNCERTAIN:  At least one UNCERTAIN and no FAILED
```

**Alignment Constraint**:
- Overall `VERIFIED` status REQUIRES `failed_checks=0` and `uncertain_checks=0`
- Cannot claim `VERIFIED` with any uncertainty

---

## 3. Knowledge Base Interface

### 3.1 Case Memory

**Location**: `physicsos/tools/memory_tools.py`

**Storage Format**: JSONL (`~/.physicsos/case_memory.jsonl`)

**Schema**:
```python
CaseMemoryRecord:
    id: str
    problem: PhysicsProblem
    result: SolverResult
    verification: VerificationReport
    postprocess: PostprocessResult | None
    indexed_features: list[str]
    tokens: list[str]
    dataset_tags: list[str]
    usage_rights: str                     # "project_internal" | "public" | etc.
```

**Search Interface**:
```python
output = search_case_memory(
    SearchCaseMemoryInput(
        problem=current_problem,
        top_k=5,
        filters={"verification_status": "verified", "domain": "fluid"}
    )
)
# → SearchCaseMemoryOutput with list[CaseMemoryHit]
```

**CaseMemoryHit**:
```python
CaseMemoryHit:
    case_id: str
    score: float                          # Jaccard similarity [0, 1]
    reason: str                           # What features matched
    backend: str | None
    verification_status: str | None
    indexed_features: list[str]
    metadata: dict[str, Any]
```

**Alignment Constraints**:
- Only cases with `verification.status != "rejected"` should be indexed
- Similarity score must be Jaccard (intersection / union of tokens)
- Filters must be exact match (no fuzzy matching without explicit user request)

---

### 3.2 Failure Database

**Location**: `physicsos/verification/base.py::FailureRecord`

**Purpose**: Structured record of verification failures for learning.

**Schema**:
```python
FailureRecord:
    uuid: str
    timestamp: datetime
    case_uuid: str
    problem: PhysicsProblem
    generated_code: str
    failure_mode: FailureMode
    severity: str                         # "critical" | "high" | "medium" | "low"
    diagnostic: str                       # Root cause
    verification_logs: dict[str, Any]
    pattern_embedding: list[float] | None
    similar_failures: list[str]
    attempted_fixes: list[str]
    resolution: str | None
    resolved_at: datetime | None
```

**FailureMode Taxonomy**:
```python
class FailureMode(str, Enum):
    # Conservation
    MASS_NOT_CONSERVED = "mass_not_conserved"
    MOMENTUM_NOT_CONSERVED = "momentum_not_conserved"
    ENERGY_NOT_CONSERVED = "energy_not_conserved"
    
    # Convergence
    DIVERGED = "diverged"
    OSCILLATORY = "oscillatory"
    STALLED = "stalled"
    
    # Stability
    NUMERICAL_INSTABILITY = "numerical_instability"
    CFL_VIOLATION = "cfl_violation"
    
    # Accuracy
    LOW_CONVERGENCE_RATE = "low_convergence_rate"
    HIGH_ERROR = "high_error"
    NONPHYSICAL_SOLUTION = "nonphysical_solution"
    
    # Code
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    DEPENDENCY_ERROR = "dependency_error"
    
    UNKNOWN = "unknown"
```

**Alignment Constraint**:
- If `resolved_at` is set, `resolution` MUST be provided
- `diagnostic` MUST include actionable root cause (not just "failed")
- `severity` determines retry strategy:
  - `critical`: Stop and ask user
  - `high`: Try one alternative, then ask
  - `medium/low`: Try multiple alternatives

---

## 4. RSI (Recursive Self-Improvement) Interfaces

### 4.1 Capability Assessment

**Location**: `physicsos/verification/pipeline.py::SelfDiagnostic`

**Purpose**: Assess system's confidence in solving a problem BEFORE attempting.

**Interface**:
```python
diagnostic = SelfDiagnostic(knowledge_base=case_memory)
assessment = diagnostic.assess_capability(problem)
# → CapabilityAssessment
```

**CapabilityAssessment Schema**:
```python
CapabilityAssessment:
    confidence: ConfidenceScore           # UNKNOWN | LOW | MEDIUM | HIGH
    reasoning: str                        # Why this confidence
    similar_cases: list[str]              # Knowledge base references
    recommendation: str                   # What user should expect
    success_rate: float | None            # Of similar cases
    total_similar_cases: int
```

**ConfidenceScore Semantics**:
```
UNKNOWN:  No similar cases → "I haven't seen this before"
LOW:      Success rate < 50% → "High risk, generating multiple candidates"
MEDIUM:   Success rate 50-90% → "Moderate confidence, will verify carefully"
HIGH:     Success rate > 90% → "This is well within my capabilities"
```

**Alignment Constraints**:
- **Epistemic Humility**: System MUST say "I don't know" when confidence is UNKNOWN
- Cannot claim HIGH confidence without verified similar cases
- `reasoning` must cite quantitative success rate (not hand-wave)
- `recommendation` must be actionable for user

**Usage**:
```python
# Before code generation:
assessment = self_diagnostic.assess_capability(problem)

if assessment.confidence == ConfidenceScore.UNKNOWN:
    print(assessment.recommendation)
    # → "I haven't encountered this before. Proceeding with exploratory verification."
    
elif assessment.confidence == ConfidenceScore.LOW:
    # Generate multiple candidate solutions
    candidates = [generate_solver(problem) for _ in range(3)]
    # Verify each and pick best
```

---

### 4.2 Failure Analysis

**Location**: `physicsos/verification/pipeline.py::FailureAnalyzer`

**Purpose**: Diagnose root cause of verification failures.

**Interface**:
```python
analyzer = FailureAnalyzer()
failure = analyzer.diagnose_failure(problem, result, verification_report)
# → FailureRecord
```

**Diagnostic Logic**:
1. Extract failed checks from `verification_report.individual_results`
2. Map verifier names to `FailureMode` taxonomy
3. Extract quantitative diagnostics from check details
4. Generate root cause explanation
5. Link to similar failures in database

**Example Diagnostic**:
```python
FailureRecord(
    failure_mode=FailureMode.MASS_NOT_CONSERVED,
    diagnostic="Mass conservation violated: inlet flux 1.5 kg/s, outlet 1.2 kg/s, imbalance 0.3 kg/s (20% error). Root cause: boundary condition not applied correctly.",
    verification_logs={
        "ConservationChecker": {
            "mass_relative_error": 0.2,
            "inlet_flux": 1.5,
            "outlet_flux": 1.2
        }
    }
)
```

**Alignment Constraint**:
- `diagnostic` must be quantitative (include actual values)
- `diagnostic` must point to likely root cause (not just restate failure)
- If multiple checks failed, prioritize the most fundamental (conservation > convergence > accuracy)

---

## 5. Alignment Constraints Summary

### 5.1 Verification Independence

**Constraint**: Verification MUST be independent of code generation.

**Enforcement**:
- Verifiers cannot call the same LLM with the same prompt
- Cannot use generated code as ground truth
- Must use different discretization or analytical comparison

**Allowed**:
- Verifiers can *parse* generated code to extract solver settings
- Verifiers can *execute* generated code to check it runs
- Verifiers can compare with reference solutions from different sources

**Forbidden**:
- Verifier asking the generator "is this correct?"
- Using generator's confidence as verification metric
- Skipping verification because generator said "high quality"

---

### 5.2 Epistemic Humility

**Constraint**: System must recognize and communicate uncertainty.

**Required Behaviors**:

1. **Admit Ignorance**:
   ```python
   if no_similar_cases:
       return CapabilityAssessment(
           confidence=ConfidenceScore.UNKNOWN,
           recommendation="I haven't seen this before. Success not guaranteed."
       )
   ```

2. **Quantify Confidence**:
   - Must cite success rate from historical data
   - Cannot claim confidence without evidence
   
3. **Clear Status**:
   - Use `UNCERTAIN` status when checks are inconclusive
   - Never claim `VERIFIED` when uncertain

4. **Transparent Limitations**:
   - If verification incomplete (e.g., no experimental data), state it clearly
   - "VERIFIED but not yet VALIDATED" vs. "VALIDATED"

**Forbidden**:
- Claiming success without verification
- Inventing confidence scores
- Hiding failures in verbose output

---

### 5.3 Determinism

**Constraint**: Verification must be reproducible.

**Requirements**:
- Same `(problem, result)` → same `VerificationResult`
- No randomness in verification logic
- Tolerances must be fixed or derived from problem parameters

**Allowed Non-Determinism**:
- Timestamp fields
- Compute time (system-dependent)
- Order of dict keys (if semantically equivalent)

**Forbidden**:
- Random sampling in verification
- Using different tolerances on repeated runs
- LLM-based verification (inherently non-deterministic)

---

### 5.4 Quantitative Reporting

**Constraint**: All verification results must include numerical metrics.

**Required**:
```python
VerificationResult(
    metrics={
        "mass_conservation_error": 1.2e-10,
        "convergence_rate": 1.97,
        "max_residual": 3.4e-8
    }
)
```

**Forbidden**:
```python
VerificationResult(
    message="Looks good!",  # ❌ No metrics
    metrics={}
)
```

**Rationale**: Metrics enable:
- Quantitative tracking of system improvements
- Threshold-based decision making
- Regression detection

---

## 6. Extension Guidelines

### 6.1 Adding a New Verifier

1. **Inherit from `Verifier`**:
```python
class MyVerifier(Verifier):
    def verify(self, problem, result) -> VerificationResult:
        # Implementation
        pass
    
    def required_data(self) -> list[str]:
        return ["field_values", "mesh"]
```

2. **Document in this spec**:
   - Add section under "2. Verification Specifications"
   - Specify checks, tolerances, metrics
   - List alignment constraints

3. **Register in default pipeline**:
```python
# In physicsos/verification/pipeline.py
DEFAULT_PIPELINE = [
    ConservationChecker(),
    ConvergenceChecker(),
    MyVerifier(),  # Add here
]
```

4. **Add tests**:
```python
# In tests/verification/test_my_verifier.py
def test_my_verifier_pass():
    verifier = MyVerifier()
    result = verifier.verify(simple_problem, exact_solution)
    assert result.status == VerificationStatus.VERIFIED
```

---

### 6.2 Adding a New FailureMode

1. **Extend enum**:
```python
class FailureMode(str, Enum):
    # ... existing modes
    MY_NEW_MODE = "my_new_mode"
```

2. **Update `FailureAnalyzer.diagnose_failure()`**:
```python
def diagnose_failure(self, ...):
    # Add detection logic
    if self._detect_my_condition(verification):
        failure_mode = FailureMode.MY_NEW_MODE
```

3. **Document diagnostic criteria**:
   - Add to section "3.2 Failure Database"
   - Specify how to detect this mode
   - Provide example diagnostic

---

## 7. Testing Requirements

### 7.1 Verifier Tests

Every verifier must have:

1. **Pass test**: Solution that should verify
2. **Fail test**: Solution with known defect
3. **Uncertain test**: Inconclusive case (e.g., missing data)
4. **Determinism test**: Same inputs → same output
5. **Metrics test**: All declared metrics populated

### 7.2 Pipeline Tests

1. **Single verifier**: One check passes/fails
2. **Multiple verifiers**: Aggregate status correct
3. **Independence**: Verification doesn't call generator
4. **Failure analysis**: Correct FailureMode classification

### 7.3 RSI Tests

1. **Capability assessment**: Confidence matches actual success rate
2. **Epistemic humility**: UNKNOWN returned when no similar cases
3. **Learning**: Failure record stored and retrievable

---

## 8. Version History

### v0.1.30 (Current)
- Independent verification framework
- Conservation & convergence checkers
- Verification pipeline with RSI hooks
- Capability assessment (epistemic humility)
- Failure taxonomy and analysis

### v0.2.0 (Planned)
- Manufactured solution generation
- Automated convergence studies
- Active knowledge gap exploration

### v0.3.0 (Planned)
- Experimental validation framework
- Uncertainty quantification
- Validation vs. verification distinction

---

**Compliance Note**: Any component violating these specifications should be treated as a bug. Alignment constraints are not optional.
