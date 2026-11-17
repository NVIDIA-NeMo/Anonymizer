# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect value-free warning events from structured pipeline diagnostics."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from anonymizer.engine import constants as cols
from anonymizer.engine.evaluation.judge_findings import build_judge_report
from anonymizer.engine.rewrite.parsers import normalize_payload


def _record_id(row: pd.Series) -> str | None:
    value = row.get(cols.RECORD_ID_COLUMN)
    return None if value is None or value is pd.NA or (isinstance(value, float) and pd.isna(value)) else str(value)


def collect_run_warnings(detection: pd.DataFrame, output: pd.DataFrame) -> list[dict[str, Any]]:
    """Capture correction attempts separately from affected records and skipped spans."""
    events: list[dict[str, Any]] = []
    for _, row in detection.iterrows():
        payload = normalize_payload(row.get(cols.COL_AUGMENTATION_ATTEMPTS))
        attempts = payload.get("attempts", []) if isinstance(payload, dict) else []
        for attempt in attempts:
            if not isinstance(attempt, dict) or not attempt.get("failures"):
                continue
            categories = Counter(
                failure.get("category", "unknown") for failure in attempt["failures"] if isinstance(failure, dict)
            )
            events.append(
                {
                    "stage": "Detection",
                    "code": "augmentation_grounding",
                    "message": "Augmentation response had grounding failures",
                    "record_id": _record_id(row),
                    "count": 1,
                    "unit": "attempts",
                    "detail": f"Attempt {attempt.get('attempt_index', '?')}: "
                    + ", ".join(f"{key}: {count}" for key, count in categories.items()),
                }
            )
    for _, row in output.iterrows():
        warnings = normalize_payload(row.get(cols.COL_REPLACEMENT_WARNINGS))
        if isinstance(warnings, list):
            events.extend(dict(warning) for warning in warnings if isinstance(warning, dict))
        application = normalize_payload(row.get(cols.COL_REPLACEMENT_APPLICATION))
        if isinstance(application, dict) and application.get("skipped_span_count", 0):
            events.append(
                {
                    "stage": "Replacement",
                    "code": "skipped_spans",
                    "message": "Targeted spans were not replaced",
                    "record_id": _record_id(row),
                    "count": int(application["skipped_span_count"]),
                    "unit": "spans",
                    "detail": "Skipped spans by label: " + str(application.get("skipped_span_label_counts", {})),
                }
            )
    return events


def collect_evaluation_warnings(trace: pd.DataFrame) -> list[dict[str, Any]]:
    """Save unavailable inference/judge diagnostics without duplicating failed records."""
    events: list[dict[str, Any]] = []
    for _, row in trace.iterrows():
        error = row.get(cols.COL_PRIVACY_QA_ERROR)
        if isinstance(error, str) and error.strip():
            events.append(
                {
                    "stage": "Evaluation",
                    "code": "privacy_qa_unavailable",
                    "message": "Privacy inference evaluation unavailable",
                    "record_id": _record_id(row),
                    "count": 1,
                    "unit": "evaluations",
                    "detail": error,
                }
            )
    _, issues = build_judge_report(trace)
    seen: set[tuple[int, str]] = set()
    for _, issue in issues[issues["Result"] == "Unavailable"].iterrows():
        key = (int(issue["Row"]), str(issue["Check"]))
        if key in seen:
            continue
        seen.add(key)
        events.append(
            {
                "stage": "Evaluation",
                "code": "judge_unavailable:" + key[1],
                "message": key[1] + " judge unavailable",
                "record_id": issue["Record ID"],
                "count": 1,
                "unit": "evaluations",
                "detail": str(issue["Reason"]),
            }
        )
    return events
