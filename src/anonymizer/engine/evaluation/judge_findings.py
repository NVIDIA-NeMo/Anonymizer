# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize saved judge results without model calls or invented verdicts."""

from __future__ import annotations

from typing import Any

import pandas as pd

from anonymizer.engine import constants as cols
from anonymizer.engine.evaluation.entity_sources import entity_source
from anonymizer.engine.rewrite.parsers import normalize_payload


def _missing(value: Any) -> bool:
    return value is None or value is pd.NA or (isinstance(value, float) and pd.isna(value))


def _entries(value: Any) -> list[dict[str, Any]]:
    value = normalize_payload(value)
    return [entry for entry in value if isinstance(entry, dict)] if isinstance(value, list) else []


def build_judge_report(trace: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-judge coverage/results and problems with positional record indexes.

    Only judges with saved result columns are included. Null verdicts are
    unavailable, never passes. Rewrite rubric ratings remain categorical;
    medium and low ratings are listed as issues without calling them failures.
    Coverage is pooled from candidate counts, never averaged across records.
    """
    summaries: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    def issue(index: int, row: pd.Series, check: str, status: str, detail: Any) -> None:
        record_id = row.get(cols.RECORD_ID_COLUMN)
        entry = detail if isinstance(detail, dict) else {}
        target = entry.get("original", entry.get("value", entry.get("description", "")))
        if entry.get("entities"):
            target = f"{target}: " + "; ".join(str(entity) for entity in entry["entities"])
        source = (
            entity_source(row, entry.get("original", entry.get("value")), entry.get("label"))
            if entry.get("label")
            else "—"
        )
        if check == "Entity coverage" and source == "Unknown":
            source = "Not detected"
        issues.append(
            {
                "Row": index,
                "Record ID": None if _missing(record_id) else record_id,
                "Check": check,
                "Entity / relation": target,
                "Label": entry.get("label", ""),
                "Source": source,
                "Replacement": entry.get("synthetic", ""),
                "Result": status,
                "Reason": entry.get("reasoning", "No saved reasoning.") if entry else str(detail),
            }
        )

    checks = [
        ("Entity coverage", cols.COL_ENTITY_COVERAGE, cols.COL_MISSED_ENTITIES),
        ("Detection validity", cols.COL_DETECTION_VALID, cols.COL_DETECTION_INVALID_ENTITIES),
        ("Type fidelity", cols.COL_TYPE_FIDELITY_VALID, cols.COL_TYPE_FIDELITY_INVALID_REPLACEMENTS),
        ("Attribute fidelity", cols.COL_ATTRIBUTE_FIDELITY_VALID, cols.COL_ATTRIBUTE_FIDELITY_INVALID_ENTITIES),
        (
            "Relational consistency",
            cols.COL_RELATIONAL_CONSISTENCY_VALID,
            cols.COL_RELATIONAL_CONSISTENCY_INVALID_RELATIONS,
        ),
    ]
    for check, verdict_col, details_col in checks:
        if verdict_col not in trace:
            continue
        evaluated = passed = unavailable = 0
        candidate_total = missed_total = 0
        pooled_complete = True
        for index, (_, row) in enumerate(trace.iterrows()):
            verdict = normalize_payload(row.get(verdict_col))
            if _missing(verdict):
                unavailable += 1
                issue(index, row, check, "Unavailable", "Saved judge result unavailable.")
                continue
            # Postprocessed booleans and 0–1 fractions are both supported.
            if not isinstance(verdict, (bool, int, float)) or not 0 <= float(verdict) <= 1:
                unavailable += 1
                issue(index, row, check, "Unavailable", "Saved judge result malformed.")
                continue
            evaluated += 1
            details = _entries(row.get(details_col))
            is_pass = float(verdict) == 1.0
            passed += int(is_pass)
            if not is_pass:
                for detail in details or ["Judge reported an issue without saved details."]:
                    issue(index, row, check, "Issue", detail)
            if verdict_col == cols.COL_ENTITY_COVERAGE:
                candidates = normalize_payload(row.get(cols.COL_ENTITY_COVERAGE_N_CANDIDATES))
                if _missing(candidates) or not isinstance(candidates, (int, float)) or candidates < len(details):
                    pooled_complete = False
                else:
                    candidate_total += int(candidates)
                    missed_total += len(details)
        result = f"{passed}/{evaluated} records fully passed; {evaluated - passed} with issues"
        if verdict_col == cols.COL_ENTITY_COVERAGE and evaluated:
            if pooled_complete:
                pooled = (candidate_total - missed_total) / candidate_total if candidate_total else 1.0
                result += (
                    f"; pooled coverage {pooled:.1%} ({candidate_total - missed_total}/{candidate_total} entities)"
                )
            else:
                result += "; pooled coverage unavailable (missing candidate counts)"
        summaries.append(
            {"Check": check, "Evaluated": f"{evaluated}/{len(trace)}", "Result": result, "Unavailable": unavailable}
        )

    if cols.COL_JUDGE_EVALUATION in trace:
        for rubric, label in [
            ("privacy", "Rewrite privacy"),
            ("quality", "Rewrite quality"),
            ("style", "Rewrite style"),
        ]:
            counts = {"high": 0, "medium": 0, "low": 0}
            unavailable = 0
            for index, (_, row) in enumerate(trace.iterrows()):
                payload = normalize_payload(row.get(cols.COL_JUDGE_EVALUATION))
                rating = payload.get(rubric) if isinstance(payload, dict) else None
                score = rating.get("score") if isinstance(rating, dict) else None
                if not isinstance(rating, dict) or not isinstance(score, str) or score not in counts:
                    unavailable += 1
                    issue(index, row, label, "Unavailable", "Saved rubric rating unavailable or malformed.")
                    continue
                counts[score] += 1
                if score != "high":
                    issue(index, row, label, score, rating.get("reasoning", "No saved reasoning."))
            summaries.append(
                {
                    "Check": label,
                    "Evaluated": f"{sum(counts.values())}/{len(trace)}",
                    "Result": "; ".join(f"{key}: {value}" for key, value in counts.items()),
                    "Unavailable": unavailable,
                }
            )
    return (
        pd.DataFrame(summaries, columns=pd.Index(["Check", "Evaluated", "Result", "Unavailable"])),
        pd.DataFrame(
            issues,
            columns=pd.Index(
                ["Row", "Record ID", "Check", "Entity / relation", "Label", "Source", "Replacement", "Result", "Reason"]
            ),
        ),
    )
