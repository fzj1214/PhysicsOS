from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from physicsos.backends.surrogate_adapters import get_adapter, safe_path_component
from physicsos.backends.surrogate_registry import checkpoint_exists, resolve_path
from physicsos.schemas.common import ArtifactRef, Provenance
from physicsos.schemas.problem import PhysicsProblem
from physicsos.schemas.solver import FieldDataRef
from physicsos.schemas.solver import SolverResult
from physicsos.schemas.surrogate import SurrogateDecision, SurrogateModelSpec


def _write_metadata_result(problem: PhysicsProblem, model: SurrogateModelSpec, decision: SurrogateDecision) -> ArtifactRef:
    output_dir = resolve_path(f"scratch/{safe_path_component(problem.id)}/{safe_path_component(model.id)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "surrogate_result.json"
    payload: dict[str, Any] = {
        "problem_id": problem.id,
        "model_id": model.id,
        "family": model.family,
        "decision": decision.model_dump(mode="json"),
        "message": "Checkpoint found; adapter metadata generated. Real tensor inference requires a matching input tensor.",
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return ArtifactRef(uri=str(output_path), kind="surrogate_result", format="json")


def run_checkpoint_surrogate(problem: PhysicsProblem, model: SurrogateModelSpec, decision: SurrogateDecision) -> SolverResult:
    """Run a configured real checkpoint when an adapter exists.

    Current implementation validates that the checkpoint exists and records an
    artifact. Real tensor execution requires a model-specific input/output
    adapter because FNO, graph operators, PINNs, and TAPS use different tensors.
    """
    if model.checkpoint is None:
        raise FileNotFoundError(f"Surrogate model {model.id} has no checkpoint configured.")
    if not checkpoint_exists(model):
        raise FileNotFoundError(f"Checkpoint for surrogate model {model.id} does not exist: {model.checkpoint.uri}")

    adapter = get_adapter(model.input_adapter, model)
    input_bundle = adapter.build_input(problem, model)

    if model.runner == "torchscript":
        artifact = _run_torchscript_or_record(problem, model, decision)
    else:
        artifact = _write_metadata_result(problem, model, decision)

    raw_output_uri = str(resolve_path(f"scratch/{safe_path_component(problem.id)}/{safe_path_component(model.id)}/output_tensor.npy"))
    output_bundle = adapter.parse_output(problem, model, raw_output_uri)
    input_metadata = ArtifactRef(
        uri=_write_input_bundle(problem, model, input_bundle),
        kind="surrogate_input_bundle",
        format="json",
    )
    output_metadata = ArtifactRef(uri=output_bundle.metadata_uri, kind="surrogate_output_bundle", format="json")

    fields = [
        FieldDataRef(
            field=tensor.name,
            uri=tensor.uri,
            format=tensor.format,
            location="grid",
            units=None,
        )
        for tensor in output_bundle.tensors
    ]

    return SolverResult(
        id=f"result:{model.id}:checkpoint",
        problem_id=problem.id,
        backend=model.id,
        status="needs_review",
        scalar_outputs={
            "surrogate_mode": decision.mode,
            "runner": model.runner,
            "checkpoint": model.checkpoint.uri,
            "message": "Real checkpoint is configured. Model-specific tensor adapter still controls actual field inference.",
        },
        uncertainty={"surrogate_expected_uncertainty": decision.expected_uncertainty or 1.0},
        fields=fields,
        artifacts=[artifact, input_metadata, output_metadata],
        provenance=Provenance(created_by="surrogate_inference", source=model.checkpoint.uri),
    )


def _write_input_bundle(problem: PhysicsProblem, model: SurrogateModelSpec, input_bundle) -> str:
    path = resolve_path(f"scratch/{safe_path_component(problem.id)}/{safe_path_component(model.id)}/input_bundle.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(input_bundle.model_dump(mode="json"), indent=2), encoding="utf-8")
    return str(path)


def _run_torchscript_or_record(problem: PhysicsProblem, model: SurrogateModelSpec, decision: SurrogateDecision) -> ArtifactRef:
    """Load TorchScript to validate the checkpoint, then record metadata.

    We intentionally do not fabricate input tensors. Each trained model needs a
    declared input_adapter before safe inference can run.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("TorchScript surrogate requires torch to be installed.") from exc

    checkpoint_path = resolve_path(model.checkpoint.uri if model.checkpoint else "")
    torch.jit.load(str(checkpoint_path), map_location="cpu")
    return _write_metadata_result(problem, model, decision)
