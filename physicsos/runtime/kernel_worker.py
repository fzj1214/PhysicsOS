"""Invoke a case-local entrypoint in a fresh interpreter."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import traceback

import numpy as np


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError("Entrypoints must return JSON metadata and write arrays as artifacts.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--entrypoint", default="run_case")
    parser.add_argument("--points", type=Path)
    parser.add_argument("--values", type=Path)
    args = parser.parse_args()
    response = {}
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        sys.path.insert(0, str(args.kernel.parent))
        specification = importlib.util.spec_from_file_location("physicsos_case_module", args.kernel)
        module = importlib.util.module_from_spec(specification)
        sys.modules[specification.name] = module
        specification.loader.exec_module(module)
        function = getattr(module, args.entrypoint)
        if args.points:
            values = np.asarray(function(np.load(args.points, allow_pickle=False), config))
            if args.values is None or values.dtype.kind not in "fci" or not np.isfinite(values).all():
                raise ValueError("Reference output must be a finite numeric array.")
            np.save(args.values, values, allow_pickle=False)
            result = {"shape": list(values.shape)}
        else:
            result = function(config)
            if result is None:
                result = {}
            if not isinstance(result, dict):
                raise ValueError("run_case(config) must return a metadata dictionary.")
            if result.get("status") in {"failed", "not_implemented"}:
                raise ValueError(f"Kernel reported {result['status']}.")
        response = {"ok": True, "result": result}
        text = json.dumps(response, indent=2, default=_json_value, allow_nan=False)
    except Exception as exc:
        response = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        text = json.dumps(response, indent=2)
    args.response.write_text(text, encoding="utf-8")
    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
