# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic-only, aggregate characterization of private row verification."""

from __future__ import annotations

import json
import sys
import time
import tracemalloc
from pathlib import Path
from types import FrameType

import pandas as pd

from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import COL_REPLACED_TEXT, COL_TEXT
from anonymizer.engine.private_row_verification import _InvocationRowVerifier
from anonymizer.engine.replace.strategies import apply_local_replace_strategy


def _synthetic_entities(text: str) -> list[dict[str, object]]:
    value = "synthetic-secret"
    positions: list[int] = []
    start = 0
    while (found := text.find(value, start)) >= 0:
        positions.append(found)
        start = found + len(value)
    return [
        {
            "value": value,
            "label": "synthetic_identifier",
            "start_position": position,
            "end_position": position + len(value),
        }
        for position in positions
    ]


def _run_arm(name: str, rows: list[str], *, suitable: bool = True) -> dict[str, object]:
    if not suitable:
        return {
            "arm": name,
            "status": "blocked",
            "reason_code": "source_specific_manifest_not_owned",
            "input_bytes": 0,
            "output_bytes": 0,
            "rows": 0,
            "targets": 0,
            "provider_calls": 0,
            "elapsed_ms": 0,
            "peak_memory_bytes": 0,
            "raw_copy_count": 0,
            "artifact_delta_bytes": 0,
            "structural_validity": False,
            "privacy_check": False,
            "reconstruction_failures": 0,
        }
    frame = pd.DataFrame(
        {
            COL_TEXT: rows,
            "final_entities": [{"entities": _synthetic_entities(text)} for text in rows],
        }
    )
    verifier = _InvocationRowVerifier(frame)
    bound = verifier.bind(frame)
    verifier.freeze_accepted_detections(bound)
    # Count actual live references to the synthetic source objects through the
    # private working frames.  This is an aggregate copy-pressure proxy, not a
    # claim about allocator-level copies.
    source_ids = {id(text) for text in rows}
    provider_calls = 0

    def profile_provider_calls(frame: FrameType, event: str, _arg: object) -> None:
        nonlocal provider_calls
        module_name = getattr(frame, "f_globals", {}).get("__name__", "")
        if event == "call" and module_name.startswith(("data_designer", "openai")):
            provider_calls += 1

    previous_profiler = sys.getprofile()
    tracemalloc.start()
    started = time.perf_counter()
    sys.setprofile(profile_provider_calls)
    try:
        protected = apply_local_replace_strategy(bound, strategy=Redact())
    finally:
        sys.setprofile(previous_profiler)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    verified = verifier.finish(protected)
    outputs = verified[COL_REPLACED_TEXT].astype(str).tolist()
    targets = sum(text.count("synthetic-secret") for text in rows)
    input_bytes = sum(len(text.encode()) for text in rows)
    output_bytes = sum(len(text.encode()) for text in outputs)
    raw_copy_count = sum(
        id(value) in source_ids
        for dataframe in (frame, bound, protected, verified)
        for value in dataframe[COL_TEXT].tolist()
    )
    structural_validity = len(verified) == len(rows) and len(outputs) == len(rows)
    privacy_check = all("synthetic-secret" not in text for text in outputs)
    return {
        "arm": name,
        "status": "completed",
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "rows": len(rows),
        "targets": targets,
        "provider_calls": provider_calls,
        "elapsed_ms": elapsed_ms,
        "peak_memory_bytes": peak,
        "raw_copy_count": raw_copy_count,
        "artifact_delta_bytes": output_bytes - input_bytes,
        "structural_validity": structural_validity,
        "privacy_check": privacy_check,
        "reconstruction_failures": int(not structural_validity),
    }


def main() -> None:
    report = {
        "fixture_policy": str(Path("tests/fixtures/streaming/POLICY.md")),
        "synthetic_only": True,
        "arms": [
            _run_arm("field_per_row", ["synthetic-secret alpha", "synthetic-secret beta"]),
            _run_arm("whole_synthetic_blob", ["synthetic-secret alpha\nsynthetic-secret beta"]),
            _run_arm("generic_manifest", [], suitable=False),
        ],
    }
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
