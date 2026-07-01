# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MEASUREMENT_FIXTURES = REPO_ROOT / "tests/fixtures/measurement"


def _load_measurement_fixture(name: str) -> dict[str, Any]:
    return json.loads((MEASUREMENT_FIXTURES / name).read_text(encoding="utf-8"))


def _minimal_benchmark_spec(
    tool: ModuleType,
    *,
    suite_id: str = "suite",
    configs: list[Any] | None = None,
    case_retries: int = 0,
    case_retry_backoff_sec: float = 0.0,
    run_tags: dict[str, Any] | None = None,
) -> Any:
    return tool.BenchmarkSpec(
        suite_id=suite_id,
        case_retries=case_retries,
        case_retry_backoff_sec=case_retry_backoff_sec,
        run_tags=run_tags or {},
        workloads=[tool.WorkloadSpec(id="input", source="input.csv")],
        configs=configs or [tool.ConfigSpec(id="redact", replace="redact")],
    )


def _minimal_benchmark_case(
    tool: ModuleType,
    *,
    suite_id: str = "suite",
    workload_id: str = "input",
    config_id: str = "redact",
    repetition: int = 0,
) -> Any:
    return tool.BenchmarkCase(
        suite_id=suite_id,
        workload_id=workload_id,
        config_id=config_id,
        repetition=repetition,
        case_id=f"{workload_id}__{config_id}__r{repetition:03d}",
    )


def _write_text_input(tmp_path: Path, text: str = "Alice works at Acme") -> Path:
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"text": [text]}).to_csv(input_path, index=False)
    return input_path


def _write_yaml(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _simple_suite_payload(*, suite_id: str = "base-suite", include_model_paths: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "suite_id": suite_id,
        "workloads": [{"id": "input", "source": "input.csv"}],
        "configs": [{"id": "redact", "replace": "redact"}],
    }
    if include_model_paths:
        payload.update({"model_configs": "./models.yaml", "model_providers": "./providers.yaml"})
    return payload


def _strict_record_payload(*, run_id: str = "run-a", **updates: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "record_type": "record",
        "run_id": run_id,
        "run_tags": {},
        "timestamp_unix_sec": 1.0,
        "mode": "replace",
        "strategy": "Redact",
        "row_index": 0,
        "record_hash": "a" * 64,
        "text_length_chars": 5,
        "text_length_chars_bucket": "1-127",
        "text_length_tokens": 1,
        "text_length_tokens_bucket": "1-127",
        "final_entity_count": 1,
        "final_entity_label_counts": {"person": 1},
    }
    payload.update(updates)
    return payload
