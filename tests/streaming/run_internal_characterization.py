# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic-only, aggregate characterization of private row verification."""

from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path

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
    tracemalloc.start()
    started = time.perf_counter()
    protected = apply_local_replace_strategy(bound, strategy=Redact())
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    verified = verifier.finish(protected)
    outputs = verified[COL_REPLACED_TEXT].astype(str).tolist()
    targets = sum(text.count("synthetic-secret") for text in rows)
    return {
        "arm": name,
        "status": "completed",
        "input_bytes": sum(len(text.encode()) for text in rows),
        "output_bytes": sum(len(text.encode()) for text in outputs),
        "rows": len(rows),
        "targets": targets,
        "provider_calls": 0,
        "elapsed_ms": elapsed_ms,
        "peak_memory_bytes": peak,
        "raw_copy_count": 0,
        "artifact_delta_bytes": 0,
        "structural_validity": len(verified) == len(rows),
        "privacy_check": all("synthetic-secret" not in text for text in outputs),
        "reconstruction_failures": 0,
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
