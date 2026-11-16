# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from anonymizer.config.replace_strategies import Annotate, Hash, Redact, Substitute
from anonymizer.config.rewrite import EvaluationCriteria, PrivacyGoal
from anonymizer.engine import constants as cols
from anonymizer.engine.evaluation.judge_findings import build_judge_report
from anonymizer.interface.results import AnonymizerResult, PreviewResult


def _result(trace: pd.DataFrame, preview: bool = False, **metadata: Any) -> AnonymizerResult | PreviewResult:
    args: dict[str, Any] = dict(
        dataframe=trace.copy(), trace_dataframe=trace, resolved_text_column="text", failed_records=[], **metadata
    )
    return PreviewResult(**args, preview_num_records=len(trace)) if preview else AnonymizerResult(**args)


def test_judge_summary_pools_counts_and_separates_unavailable() -> None:
    trace = pd.DataFrame(
        [
            {
                cols.RECORD_ID_COLUMN: "a",
                cols.COL_ENTITY_COVERAGE: 0.5,
                cols.COL_ENTITY_COVERAGE_N_CANDIDATES: 2,
                cols.COL_MISSED_ENTITIES: [{"value": "Alice", "label": "name", "reasoning": "Missed"}],
                cols.COL_TYPE_FIDELITY_VALID: False,
                cols.COL_TYPE_FIDELITY_INVALID_REPLACEMENTS: [],
            },
            {
                cols.COL_ENTITY_COVERAGE: 1.0,
                cols.COL_ENTITY_COVERAGE_N_CANDIDATES: 8,
                cols.COL_MISSED_ENTITIES: [],
                cols.COL_TYPE_FIDELITY_VALID: True,
            },
            {cols.COL_ENTITY_COVERAGE: None, cols.COL_TYPE_FIDELITY_VALID: None},
        ]
    )
    summary, issues = build_judge_report(trace)
    assert summary.iloc[0]["Evaluated"] == "2/3"
    assert "90.0% (9/10 entities)" in summary.iloc[0]["Result"]
    assert summary.iloc[0]["Unavailable"] == 1
    assert summary.iloc[1]["Result"] == "1/2 records fully passed; 1 with issues"
    assert issues["Row"].tolist() == [0, 2, 0, 2]
    assert "Alice" in issues.iloc[0]["Entity / relation"]
    assert "without saved details" in issues.iloc[2]["Reason"]
    assert trace.iloc[0][cols.COL_MISSED_ENTITIES][0]["value"] == "Alice"


def test_rewrite_ratings_are_not_boolean_verdicts_and_partial_rubrics_are_unavailable() -> None:
    trace = pd.DataFrame(
        [
            {
                cols.COL_JUDGE_EVALUATION: json.dumps(
                    {
                        "privacy": {"score": "high", "reasoning": "Protected"},
                        "quality": {"score": "medium", "reasoning": "Some meaning lost"},
                        "style": {"score": "low", "reasoning": "Awkward"},
                    }
                )
            },
            {cols.COL_JUDGE_EVALUATION: {"privacy": {"score": "high"}}},
        ]
    )
    summary, issues = build_judge_report(trace)
    assert "high: 2; medium: 0; low: 0" == summary.iloc[0]["Result"]
    assert summary["Unavailable"].tolist() == [0, 1, 1]
    assert issues["Result"].tolist() == ["medium", "Unavailable", "low", "Unavailable"]
    assert "Some meaning lost" in issues.iloc[0]["Reason"]


@pytest.mark.parametrize("strategy", [Redact(), Substitute(), Annotate(), Hash()])
@pytest.mark.parametrize("preview", [False, True])
def test_combined_report_layout_strategy_escape_and_export(
    strategy: Any, preview: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = pd.DataFrame(
        [
            {
                cols.COL_TYPE_FIDELITY_VALID: False,
                cols.COL_TYPE_FIDELITY_INVALID_REPLACEMENTS: [{"reasoning": "<script>bad</script> É"}],
            }
        ]
    )
    result = _result(trace, preview, replace_method=strategy)
    rendered: list[str] = []
    monkeypatch.setattr("anonymizer.interface.results._display_html", rendered.append)
    result.display_evaluation_report()
    report = rendered[0]
    assert "NeMo Anonymizer Evaluation Report" in report
    assert f"Replace - {type(strategy).__name__}" in report
    assert report.index("Privacy &amp; Quality Judge Summary") < report.index("Privacy Inference Summary")
    assert report.index("Privacy &amp; Quality Judge Issues") < report.index("Privacy Inference Issues")
    assert "Not evaluated." in report
    assert "&lt;script&gt;bad&lt;/script&gt; É" in report
    assert "<script>bad</script>" not in report
    destination = tmp_path / "report.html"
    assert result.export_evaluation_report(str(destination)) == destination
    assert report in destination.read_text(encoding="utf-8")


def test_rewrite_strategy_and_settings_in_combined_report(tmp_path: Path) -> None:
    result = _result(
        pd.DataFrame(
            [
                {
                    cols.COL_LEAKAGE_MASS: 0,
                    cols.COL_REPAIR_ITERATIONS: 2,
                    cols.COL_ENTITIES_BY_VALUE: {"entities_by_value": []},
                }
            ]
        ),
        rewrite_config=PrivacyGoal(protect="Protect personal names", preserve="Preserve important meaning"),
        rewrite_evaluation=EvaluationCriteria(max_repair_iterations=4),
    )
    html = result.export_evaluation_report(tmp_path / "rewrite.html").read_text()
    assert "<strong>Strategy:</strong> Rewrite" in html
    assert "Max repair iterations" in html
    assert "Repair iterations taken" in html
    assert "Not evaluated." in html  # judges have not run


def test_numpy_verdicts_and_missing_candidate_counts_are_supported() -> None:
    trace = pd.DataFrame([{cols.COL_ENTITY_COVERAGE: np.float64(0.5), cols.COL_TYPE_FIDELITY_VALID: np.bool_(True)}])
    summary, _ = build_judge_report(trace)
    assert "pooled coverage unavailable" in summary.iloc[0]["Result"]
    assert "1/1 records fully passed" in summary.iloc[1]["Result"]


def test_unknown_strategy_does_not_guess_and_absent_judges_are_omitted(tmp_path: Path) -> None:
    result = _result(pd.DataFrame([{}]))
    path = result.export_evaluation_report(tmp_path / "unknown.html")
    html = path.read_text()
    assert "Unknown (strategy metadata unavailable)" in html
    assert "Type fidelity" not in html


def test_detection_fraction_and_relation_failures_keep_problem_details() -> None:
    trace = pd.DataFrame(
        [
            {
                cols.COL_DETECTION_VALID: 0.75,
                cols.COL_DETECTION_INVALID_ENTITIES: json.dumps(
                    [{"value": "123", "label": "name", "reasoning": "Not a name"}]
                ),
                cols.COL_ATTRIBUTE_FIDELITY_VALID: False,
                cols.COL_ATTRIBUTE_FIDELITY_INVALID_ENTITIES: [
                    {"original": "Alice", "synthetic": "Bob", "label": "name", "reasoning": "Changed attribute"}
                ],
                cols.COL_RELATIONAL_CONSISTENCY_VALID: False,
                cols.COL_RELATIONAL_CONSISTENCY_INVALID_RELATIONS: [
                    {
                        "description": "City/state",
                        "entities": ["city -> Dallas", "state -> CA"],
                        "reasoning": "Inconsistent geography",
                    }
                ],
            }
        ]
    )
    summary, issues = build_judge_report(trace)
    assert len(summary) == 3
    assert (summary["Evaluated"] == "1/1").all()
    assert (summary["Unavailable"] == 0).all()
    assert issues["Reason"].tolist() == ["Not a name", "Changed attribute", "Inconsistent geography"]
    assert issues.iloc[1]["Replacement"] == "Bob"
    assert "city -> Dallas" in issues.iloc[2]["Entity / relation"]
    assert "state -> CA" in issues.iloc[2]["Entity / relation"]


@pytest.mark.parametrize("preview", [False, True])
def test_report_counts_unique_failed_records_near_title(preview: bool, tmp_path: Path) -> None:
    from anonymizer.engine.ndd.adapter import FailedRecord

    result = _result(pd.DataFrame([{}]), preview, replace_method=Redact())
    result.failed_records = [
        FailedRecord(record_id="r1", step="detection", reason="Parse failed"),
        FailedRecord(record_id="r1", step="evaluation", reason="Timeout"),
        FailedRecord(record_id="r2", step="replacement", reason="Missing map"),
    ]
    document = result.export_evaluation_report(tmp_path / "report.html").read_text()
    assert "<strong>Failed records:</strong> 2" in document
    assert document.index("Failed records:") < document.index("Privacy &amp; Quality Judge Summary")


@pytest.mark.parametrize("rewrite_ratings", [False, True])
def test_rubric_explanation_only_appears_with_saved_rewrite_ratings(tmp_path: Path, rewrite_ratings: bool) -> None:
    row = {cols.COL_TYPE_FIDELITY_VALID: True}
    if rewrite_ratings:
        row[cols.COL_JUDGE_EVALUATION] = {
            rubric: {"score": "high", "reasoning": "Good"} for rubric in ("privacy", "quality", "style")
        }
    result = _result(pd.DataFrame([row]), replace_method=None if rewrite_ratings else Substitute())
    document = result.export_evaluation_report(tmp_path / "report.html").read_text()
    assert ("Rewrite rubric ratings are categorical" in document) is rewrite_ratings
