# PhysicsOS Architecture

**Version**: 0.1.30 (RSI-capable verification framework)  
**Status**: Alpha - Independent verification and self-improving capabilities active

---

## Design Principles

PhysicsOS is built around three core principles:

1. **Trustworthy Autonomy**: Agentic workflow with independent verification, clear success/failure records, and epistemic humility
2. **Physics-First IR**: TAPS (Tensor Approximation of Partial Differential Equations) as the universal representation
3. **Recursive Self-Improvement**: The system learns from verification outcomes to expand its capability boundaries

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                            │
│  TUI (DeepAgents) / CLI / API / Cloud Runner                    │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│                     Agent Orchestration                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Plan Agent   │  │ Solver Agent │  │ Postprocess  │          │
│  │ (TAPS gen)   │  │ (code gen)   │  │ Agent        │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                  │                  │                   │
│         └──────────────────┼──────────────────┘                  │
│                            │                                      │
│                 ┌──────────▼──────────┐                          │
│                 │ Verification        │                          │
│                 │ Pipeline (NEW)      │                          │
│                 └──────────┬──────────┘                          │
│                            │                                      │
│                 ┌──────────▼──────────┐                          │
│                 │ Knowledge Base       │                          │
│                 │ (Case Memory + RSI)  │                          │
│                 └─────────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    Domain Layer (Tools)                          │
│  Geometry │ Materials │ Mesh │ Solvers │ Verification │ Cloud   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. TAPS IR (Intermediate Representation)

**Location**: `physicsos/schemas/taps.py`

The universal physics representation. All problems are translated into TAPS before code generation.

**Structure**:
```
PhysicsProblem:
  - geometry: GeometrySpec (dimension, source, encodings)
  - mesh: MeshSpec (kind, topology, quality)
  - fields: list[FieldSpec] (name, kind, components)
  - operators: list[OperatorSpec] (equation_class, form, conserved_quantities)
  - materials: list[MaterialSpec] (name, phase, properties)
  - boundary_conditions: list[BoundaryCondition]
  - targets: list[Target] (objective, constraints)
```

**Alignment Constraint**: Every solver-specific code must be traceable back to TAPS IR.

### 2. Agent Orchestration

**Location**: `physicsos/agents/`

Three-agent pipeline:
1. **Plan Agent**: Problem → TAPS IR
2. **Solver Agent**: TAPS IR → executable code
3. **Postprocess Agent**: Solution → plots/reports

**New (v0.1.30)**: Verification pipeline runs between Solver and Postprocess.

### 3. Independent Verification Framework ⭐ NEW

**Location**: `physicsos/verification/`

**Purpose**: Ensure trustworthy simulation results through independent checks, separate from the code generator.

**Components**:

#### 3.1 Verifier Interface
```python
class Verifier(ABC):
    @abstractmethod
    def verify(problem: PhysicsProblem, result: SolverResult) -> VerificationResult:
        """Independent check - must NOT use the same model that generated code."""
        pass
```

**Alignment Constraint**: Verifiers must be independent of code generation to avoid circular validation.

#### 3.2 Built-in Verifiers

| Verifier | Purpose | Checks |
|----------|---------|--------|
| `ConservationChecker` | Verify conservation laws | Mass, momentum, energy balance |
| `ConvergenceChecker` | Mesh/temporal convergence | Grid refinement, convergence rate |
| `StabilityChecker` | Numerical stability | CFL, oscillations, blow-up |
| `AnalyticalVerifier` | Compare with known solutions | Exact solutions, manufactured solutions |

#### 3.3 Verification Pipeline

```python
pipeline = VerificationPipeline()
pipeline.register(ConservationChecker())
pipeline.register(ConvergenceChecker())

report = pipeline.verify(problem, result, confidence=confidence_score)
# → AggregateVerificationReport with overall VERIFIED/FAILED/UNCERTAIN status
```

**Verification Status**:
- `UNVERIFIED`: Not yet checked
- `VERIFIED`: Passed all independent checks
- `VALIDATED`: Also compared with experimental data (future)
- `FAILED`: Failed verification checks
- `UNCERTAIN`: Verification inconclusive

### 4. Recursive Self-Improvement (RSI) Loop

**Purpose**: Learn from verification outcomes to expand system capabilities.

**Flow**:
```
Problem → Capability Assessment → Code Generation → Verification → Learning
   ↑                                                                    │
   └────────────────────────────────────────────────────────────────────┘
```

#### 4.1 Capability Assessment (Epistemic Humility)

**Location**: `physicsos/verification/pipeline.py::SelfDiagnostic`

Before generating code, assess confidence:
```python
assessment = self_diagnostic.assess_capability(problem)
# → CapabilityAssessment with confidence: UNKNOWN/LOW/MEDIUM/HIGH
```

**Confidence Scoring**:
- `UNKNOWN`: No similar cases in knowledge base → "I haven't seen this before"
- `LOW`: High failure rate in similar cases → Generate multiple candidates
- `MEDIUM`: Mixed track record → Proceed with caution
- `HIGH`: Consistently successful → Proceed normally

**Alignment Constraint**: System must explicitly state "I don't know" rather than hallucinating confidence.

#### 4.2 Failure Analysis

**Location**: `physicsos/verification/pipeline.py::FailureAnalyzer`

When verification fails:
```python
failure_record = analyzer.diagnose_failure(problem, result, verification)
# → FailureRecord with:
#   - failure_mode: taxonomy of what went wrong
#   - diagnostic: root cause analysis
#   - similar_failures: links to knowledge base
```

**Failure Taxonomy** (`FailureMode`):
- Conservation violations: `MASS_NOT_CONSERVED`, `MOMENTUM_NOT_CONSERVED`, etc.
- Convergence: `DIVERGED`, `OSCILLATORY`, `STALLED`
- Stability: `NUMERICAL_INSTABILITY`, `CFL_VIOLATION`
- Accuracy: `LOW_CONVERGENCE_RATE`, `HIGH_ERROR`, `NONPHYSICAL_SOLUTION`
- Code: `SYNTAX_ERROR`, `RUNTIME_ERROR`, `DEPENDENCY_ERROR`

#### 4.3 Knowledge Base Integration

**Location**: `physicsos/tools/memory_tools.py`

**Extended Schema** (v0.1.30):
```python
CaseMemoryRecord:
  problem: PhysicsProblem
  result: SolverResult
  verification: VerificationReport       # Now includes detailed verification data
  postprocess: PostprocessResult | None
  
  # RSI fields:
  indexed_features: list[str]
  tokens: list[str]                      # For similarity search
  dataset_tags: list[str]                # For training data curation
```

**Search & Retrieve**:
```python
similar_cases = search_case_memory(
    problem=current_problem,
    filters={"verification_status": "verified"}
)
# Used for capability assessment
```

#### 4.4 Learning from Outcomes

**Future (v0.2.0+)**:
- Pattern extraction from failures → Update generation strategies
- Active exploration → Identify knowledge gaps, generate test cases
- Model catalog updates → Improve TAPS IR → Solver mapping confidence

---

## Verification Alignment Specification

### Core Constraints

1. **Independence**: Verification must use different methods/models than code generation
2. **Quantitative**: All checks must produce numerical metrics (not just pass/fail)
3. **Deterministic**: Same inputs → same verification outcome
4. **Traceable**: Every verification result links back to TAPS IR and solver code
5. **Epistemic Humility**: System must recognize and communicate uncertainty

### Status Transition Rules

```
UNVERIFIED → VERIFIED:     All checks passed
UNVERIFIED → FAILED:       At least one check failed
UNVERIFIED → UNCERTAIN:    Checks inconclusive or missing data
VERIFIED   → VALIDATED:    Experimental data comparison added (future)

Forbidden transitions:
  FAILED → VERIFIED (without regenerating solution)
  UNCERTAIN → VERIFIED (without resolving uncertainty)
```

### Failure Handling Protocol

When verification FAILS:
1. **Record**: Create `FailureRecord` with diagnostic
2. **Analyze**: Classify `FailureMode`
3. **Store**: Add to knowledge base
4. **Report**: Return detailed diagnostic to user
5. **Learn**: Update generation strategies (future)

Do NOT:
- Silently retry without diagnosing
- Claim success when uncertain
- Invent verification metrics

---

## Data Flow

### 1. Problem Ingestion
```
User Input → Analysis Files → Structured Problem → TAPS IR
```

### 2. Code Generation
```
TAPS IR + Similar Cases → Solver Code + Confidence Score
```

### 3. Execution
```
Solver Code → Numerical Solution → SolverResult
```

### 4. Verification (NEW)
```
(Problem, Result) → Multiple Verifiers → AggregateVerificationReport
```

### 5. Knowledge Update
```
(Problem, Result, Verification) → Case Memory + Failure DB
```

---

## File System Layout

```
physicsos/
├── agents/               # Agent implementations
│   ├── main_agent.py
│   ├── plan_agent.py
│   ├── solver_agent.py
│   └── postprocess_agent.py
├── schemas/              # Pydantic data models
│   ├── problem.py        # PhysicsProblem (TAPS IR)
│   ├── solver.py         # SolverResult
│   ├── verification.py   # VerificationReport
│   └── ...
├── tools/                # Domain tools
│   ├── memory_tools.py   # Case memory search/store
│   ├── geometry_tools.py
│   ├── materials_tools.py
│   └── ...
├── verification/         # ⭐ NEW: Independent verification
│   ├── base.py           # Verifier interface
│   ├── conservation.py   # Conservation checker
│   ├── convergence.py    # Convergence checker
│   └── pipeline.py       # Orchestration + RSI
└── ...
```

**Runtime Data** (`~/.physicsos/`):
```
~/.physicsos/
├── config.json           # Model config
├── case_memory.jsonl     # Historical cases
├── case_memory_events.jsonl  # Event log
└── cases/
    └── <case_id>/
        ├── problem.json
        ├── result.json
        ├── verification.json  # ⭐ NEW
        └── ...
```

---

## Extension Points

### Adding a New Verifier

1. Inherit from `Verifier`:
```python
class MyVerifier(Verifier):
    def verify(self, problem, result) -> VerificationResult:
        # Your check logic
        pass
    
    def required_data(self) -> list[str]:
        return ["field_values", "mesh"]
```

2. Register in pipeline:
```python
pipeline.register(MyVerifier())
```

3. Document in `spec.md` → Verification Interface

### Adding Failure Recovery

Location: `physicsos/agents/solver_agent.py`

```python
if verification.overall_status == VerificationStatus.FAILED:
    failure = analyzer.diagnose_failure(problem, result, verification)
    # Strategy 1: Try alternative solver
    # Strategy 2: Refine discretization
    # Strategy 3: Ask user for constraints
```

---

## Performance Characteristics

| Stage | Typical Time | Bottleneck |
|-------|--------------|------------|
| Problem analysis | 1-3s | File I/O |
| TAPS generation | 5-15s | LLM inference |
| Code generation | 10-30s | LLM inference |
| Solver execution | 1s - 10min | Physics solver |
| Verification | 1-5s | Conservation checks |
| Case memory store | <100ms | Disk write |

**Token Budget** (per case):
- Problem analysis: ~2k tokens
- TAPS derivation: ~8k tokens
- Code generation: ~10k tokens
- Verification: ~1k tokens

---

## Security & Privacy

- No external network calls during verification (all local checks)
- Case memory stored locally (`~/.physicsos/case_memory.jsonl`)
- Cloud runner uses isolated workspaces
- No training data uploaded without explicit consent (`dataset_tags` + `usage_rights`)

---

## Future Roadmap

### v0.2.0: Active Learning
- Knowledge gap identification
- Manufactured solution generation
- Convergence study automation

### v0.3.0: Experimental Validation
- Experimental data ingestion
- Validation vs. verification distinction
- Uncertainty quantification

### v0.4.0: Self-Improving Generator
- Pattern-based generation strategy updates
- Automatic fix generation from failure diagnosis
- Multi-candidate generation for low-confidence cases

### v0.5.0: Domain-Specific Modules
- Composite fracture (reserved for collaboration)
- Fluid-structure interaction
- Multiphase flow

---

## References

- TAPS framework: [Coming soon]
- DeepAgents: Agentic orchestration harness
- Knowledge-Space Theory: ALEKS-style prerequisite mapping (future)

---

**Maintainer Note**: This architecture is designed for trustworthy autonomy in computational physics. Every design decision prioritizes verification over convenience, and epistemic humility over false confidence.
