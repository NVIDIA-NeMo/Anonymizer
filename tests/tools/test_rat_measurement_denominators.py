# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


@pytest.fixture
def rat_record() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "record_type": "rat_bench_reidentification",
        "run_id": "synthetic",
        "run_tags": {},
        "timestamp_unix_sec": 1.0,
        "rows_processed": 10,
        "rows_failed": 0,
        "rows_requested": 10,
        "rows_scored": 10,
        "rows_missing": 0,
        "rows_excluded": 0,
        "reidentified_rows": 2,
        "direct_reidentified_rows": 1,
        "correctmatch_reidentified_rows": 1,
        "reid_error_rows": 0,
        "reidentification_rate_pct": 20.0,
        "reidentification_rate_ci95_low_pct": 5.0,
        "reidentification_rate_ci95_high_pct": 50.0,
        "coverage_pct": 50.0,
        "correct_guess_count": 10,
        "incorrect_guess_count": 10,
        "total_guess_count": 20,
        "mean_reid_score": 0.1,
        "reid_threshold": 0.2,
    }


def test_denominators_and_interval_survive_sanitized_export(
    rat_record: dict[str, Any],
    tmp_path: Path,
    wandb_ingress_tool: ModuleType,
    wandb_logging_tool: ModuleType,
) -> None:
    path = tmp_path / "measurements.jsonl"
    path.write_text(json.dumps(rat_record) + "\n")
    snapshot = wandb_ingress_tool.read_measurement_snapshot(path)
    _, summary, tables = wandb_logging_tool.build_outbound_measurements(snapshot, cases=[], log_tables=True)
    for field in (
        "rows_requested",
        "rows_scored",
        "rows_missing",
        "rows_excluded",
        "reidentification_rate_ci95_low_pct",
        "reidentification_rate_ci95_high_pct",
    ):
        assert summary.metrics[f"measurement/rat_bench_reidentification/{field}"] == rat_record[field]
        assert getattr(tables[0].rows[0], field) == rat_record[field]
    assert "measurement/rat_bench_reidentification/reidentification_rate_ci95_low_pct_mean" not in summary.metrics


@pytest.mark.parametrize("requested,rate", [(None, 20.0), (20, 10.0)])
def test_rate_uses_requested_denominator_when_supplied(
    rat_record: dict[str, Any],
    requested: int | None,
    rate: float,
    wandb_ingress_tool: ModuleType,
) -> None:
    rat_record.update(rows_requested=requested, reidentification_rate_pct=rate)
    record = wandb_ingress_tool.RatBenchReidentificationMeasurement.model_validate(rat_record)
    assert record.reidentification_rate_pct == rate


@pytest.mark.parametrize(
    "changes",
    [
        {"rows_requested": 9},
        {"rows_scored": 11},
        {"rows_scored": 9},
        {"rows_missing": 11},
        {"rows_excluded": 11},
        {"rows_missing": 1},
        {"rows_excluded": 1},
        {"rows_failed": 1},
        {"reid_error_rows": 1, "rows_scored": None, "rows_missing": 2},
        {"rows_missing": -1},
        {"rows_requested": 20},
        {"reidentification_rate_ci95_low_pct": None},
        {"reidentification_rate_ci95_high_pct": None},
        {"reidentification_rate_ci95_low_pct": 25.0},
        {"reidentification_rate_ci95_high_pct": 10.0},
        {"reidentification_rate_ci95_high_pct": 101.0},
        {"reidentification_rate_ci95_low_pct": float("nan")},
    ],
)
def test_invalid_denominators_and_intervals_are_rejected(
    rat_record: dict[str, Any],
    changes: dict[str, Any],
    wandb_ingress_tool: ModuleType,
) -> None:
    with pytest.raises(ValueError):
        wandb_ingress_tool.RatBenchReidentificationMeasurement.model_validate(rat_record | changes)


@pytest.mark.parametrize("count,low,high", [(0, 1e-14, 30.0), (10, 70.0, 100.0 - 1e-14)])
def test_interval_containment_tolerates_boundary_rounding(
    rat_record: dict[str, Any],
    count: int,
    low: float,
    high: float,
    wandb_ingress_tool: ModuleType,
) -> None:
    rat_record.update(
        reidentified_rows=count,
        direct_reidentified_rows=count,
        correctmatch_reidentified_rows=0,
        reidentification_rate_pct=count * 10.0,
        reidentification_rate_ci95_low_pct=low,
        reidentification_rate_ci95_high_pct=high,
    )
    record = wandb_ingress_tool.RatBenchReidentificationMeasurement.model_validate(rat_record)
    assert record.reidentification_rate_pct == count * 10.0


@pytest.mark.parametrize("scored", [None, 8])
def test_missing_rows_do_not_double_count_processed_errors(
    rat_record: dict[str, Any],
    scored: int | None,
    wandb_ingress_tool: ModuleType,
) -> None:
    rat_record.update(reid_error_rows=2, rows_scored=scored, rows_missing=1, rows_failed=1)
    record = wandb_ingress_tool.RatBenchReidentificationMeasurement.model_validate(rat_record)
    assert record.rows_requested == 10


@pytest.mark.parametrize("second_interval", [False, True])
def test_multiple_records_keep_intervals_in_tables(
    rat_record: dict[str, Any],
    second_interval: bool,
    tmp_path: Path,
    wandb_ingress_tool: ModuleType,
    wandb_logging_tool: ModuleType,
) -> None:
    second = rat_record | {
        "reidentified_rows": 8,
        "direct_reidentified_rows": 4,
        "correctmatch_reidentified_rows": 4,
        "reidentification_rate_pct": 80.0,
        "reidentification_rate_ci95_low_pct": 60.0 if second_interval else None,
        "reidentification_rate_ci95_high_pct": 95.0 if second_interval else None,
    }
    path = tmp_path / "measurements.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in (rat_record, second)) + "\n")
    snapshot = wandb_ingress_tool.read_measurement_snapshot(path)
    _, summary, tables = wandb_logging_tool.build_outbound_measurements(snapshot, cases=[], log_tables=True)
    prefix = "measurement/rat_bench_reidentification/"
    assert summary.metrics[prefix + "reidentification_rate_pct_mean"] == 50.0
    assert prefix + "reidentification_rate_ci95_low_pct" not in summary.metrics
    assert prefix + "reidentification_rate_ci95_high_pct" not in summary.metrics
    assert tables[0].rows[0].reidentification_rate_ci95_low_pct == 5.0
    assert tables[0].rows[1].reidentification_rate_ci95_low_pct == second["reidentification_rate_ci95_low_pct"]


@pytest.mark.parametrize("field", ["rows_requested", "rows_scored", "rows_missing", "rows_excluded"])
def test_partial_denominators_are_not_reported_as_totals(
    rat_record: dict[str, Any],
    field: str,
    tmp_path: Path,
    wandb_ingress_tool: ModuleType,
    wandb_logging_tool: ModuleType,
) -> None:
    second = rat_record | {field: None}
    path = tmp_path / "measurements.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in (rat_record, second)) + "\n")
    snapshot = wandb_ingress_tool.read_measurement_snapshot(path)
    _, summary, tables = wandb_logging_tool.build_outbound_measurements(snapshot, cases=[], log_tables=True)
    prefix = "measurement/rat_bench_reidentification/"
    assert prefix + field not in summary.metrics
    assert summary.metrics[prefix + "rows_processed"] == 20
    assert getattr(tables[0].rows[0], field) == rat_record[field]
    assert getattr(tables[0].rows[1], field) is None
