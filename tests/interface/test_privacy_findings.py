# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from anonymizer.engine import constants as cols
from anonymizer.engine.evaluation.privacy_findings import PRIVACY_FINDING_COLUMNS, build_privacy_findings
from anonymizer.engine.schemas import EntityCategory, PrivacyAnswer, SensitivityLevel
from anonymizer.interface.results import AnonymizerResult, PreviewResult


def _result(trace: pd.DataFrame, preview: bool = False) -> AnonymizerResult | PreviewResult:
    kwargs: dict[str, Any] = {
        "dataframe": pd.DataFrame(),
        "trace_dataframe": trace,
        "resolved_text_column": "biography",
        "failed_records": [],
    }
    return PreviewResult(**kwargs, preview_num_records=len(trace)) if preview else AnonymizerResult(**kwargs)


def _row() -> dict[str, Any]:
    targets = [
        {
            "entity_value": "Alice",
            "entity_label": "first_name",
            "category": "direct_identifier",
            "sensitivity": "high",
            "reason": "Name sensitivity reason",
        },
        {
            "entity_value": "Catholic",
            "entity_label": "religious_belief",
            "category": "quasi_identifier",
            "sensitivity": "medium",
            "reason": "Religion sensitivity reason",
        },
    ]
    questions = [
        {
            "id": index,
            "question": f"Can {target['entity_value']} be deduced?",
            **{key: value for key, value in target.items() if key != "reason"},
        }
        for index, target in enumerate(targets, 1)
    ]
    return {
        cols.RECORD_ID_COLUMN: "record-a",
        cols.COL_ENTITY_SENSITIVITY: {"entities": targets},
        cols.COL_PRIVACY_QA: {"items": list(reversed(questions))},
        cols.COL_PRIVACY_QA_REANSWER: {
            "answers": [
                {"id": 2, "answer": "yes", "confidence": 0.8, "reason": "Attends Mass.", "evidence": ["attends Mass"]},
                {"id": 1, "answer": "no", "confidence": 1.0, "reason": "No original name.", "evidence": []},
            ]
        },
        cols.COL_PRIVACY_QA_ERROR: None,
    }


@pytest.mark.parametrize("preview", [False, True])
@pytest.mark.parametrize("representation", ["native", "json", "numpy"])
def test_shared_accessor_preserves_assessments_and_joins_ids(preview: bool, representation: str) -> None:
    row = _row()
    for column, key in [
        (cols.COL_ENTITY_SENSITIVITY, "entities"),
        (cols.COL_PRIVACY_QA, "items"),
        (cols.COL_PRIVACY_QA_REANSWER, "answers"),
    ]:
        if representation == "json":
            row[column] = json.dumps(row[column])
        elif representation == "numpy":
            row[column][key] = np.array(row[column][key], dtype=object)
    table = _result(pd.DataFrame([row], index=pd.Index([99])), preview).trace_dataframe.pipe(build_privacy_findings)
    assert table.columns.tolist() == list(PRIVACY_FINDING_COLUMNS)
    assert table[cols.COL_FINDING_RECORD_INDEX].tolist() == [0, 0]
    assert table[cols.COL_FINDING_RECORD_ID].tolist() == ["record-a", "record-a"]
    assert table[cols.COL_FINDING_ENTITY_VALUE].tolist() == ["Alice", "Catholic"]
    assert table[cols.COL_FINDING_ANSWER].tolist() == ["no", "yes"]
    assert table[cols.COL_FINDING_CONTRIBUTION].tolist() == [0.0, 0.48]
    assert table[cols.COL_FINDING_SENSITIVITY_REASON].tolist() == [
        "Name sensitivity reason",
        "Religion sensitivity reason",
    ]
    assert table[cols.COL_FINDING_STATUS].tolist() == ["evaluated", "evaluated"]


def test_accessor_is_fresh_and_evidence_edits_do_not_mutate_trace() -> None:
    row = _row()
    result = _result(pd.DataFrame([row]))
    first = result.trace_dataframe.pipe(build_privacy_findings)
    first.at[1, cols.COL_FINDING_EVIDENCE].append("changed")
    assert result.trace_dataframe.pipe(build_privacy_findings).at[1, cols.COL_FINDING_EVIDENCE] == ["attends Mass"]
    row[cols.COL_PRIVACY_QA_REANSWER]["answers"][0]["reason"] = "Updated final reason"
    assert result.trace_dataframe.pipe(build_privacy_findings).at[1, cols.COL_FINDING_REASON] == "Updated final reason"


def test_numeric_looking_record_ids_remain_strings() -> None:
    row = _row()
    row[cols.RECORD_ID_COLUMN] = "12345"
    table = _result(pd.DataFrame([row])).trace_dataframe.pipe(build_privacy_findings)
    assert table[cols.COL_FINDING_RECORD_ID].tolist() == ["12345", "12345"]


def test_schema_enums_are_exposed_as_plain_strings() -> None:
    row = _row()
    row[cols.COL_ENTITY_SENSITIVITY]["entities"][0]["category"] = EntityCategory.direct_identifier
    row[cols.COL_ENTITY_SENSITIVITY]["entities"][0]["sensitivity"] = SensitivityLevel.high
    row[cols.COL_PRIVACY_QA_REANSWER]["answers"][0]["answer"] = PrivacyAnswer.yes
    table = _result(pd.DataFrame([row])).trace_dataframe.pipe(build_privacy_findings)
    for column in (cols.COL_FINDING_CATEGORY, cols.COL_FINDING_SENSITIVITY, cols.COL_FINDING_ANSWER):
        assert all(type(value) is str for value in table[column])


def test_rewrite_exemptions_and_latent_targets_use_final_answers() -> None:
    row = _row()
    del row[cols.COL_ENTITY_SENSITIVITY]
    row[cols.COL_SENSITIVITY_DISPOSITION] = {
        "sensitivity_disposition": [
            {
                "entity_value": "married",
                "entity_label": "marital_status",
                "category": "latent_identifier",
                "sensitivity": "medium",
                "protection_reason": "Spouse makes marriage inferable",
                "protection_method_suggestion": "suppress_inference",
            },
            {
                "entity_value": "USA",
                "entity_label": "country",
                "category": "quasi_identifier",
                "sensitivity": "low",
                "low_sensitivity_reason": "Broad geography",
                "protection_method_suggestion": "leave_as_is",
            },
        ]
    }
    row[cols.COL_PRIVACY_QA] = {
        "items": [
            {
                "id": 1,
                "question": "Can married be deduced?",
                "entity_value": "married",
                "entity_label": "marital_status",
                "category": "latent_identifier",
                "sensitivity": "medium",
            }
        ]
    }
    row[cols.COL_PRIVACY_QA_REANSWER] = {
        "answers": [
            {
                "id": 1,
                "answer": "no",
                "confidence": 0.9,
                "reason": "Final repaired text has no spouse reference.",
                "evidence": [],
            }
        ]
    }
    row[cols.COL_LEAKED_PRIVACY_ITEMS] = "Stale pre-repair finding: married leaked"
    table = _result(pd.DataFrame([row])).trace_dataframe.pipe(build_privacy_findings)
    assert table[cols.COL_FINDING_STATUS].tolist() == ["evaluated", "not_evaluated"]
    assert table[cols.COL_FINDING_CATEGORY].tolist() == ["latent_identifier", "quasi_identifier"]
    assert table[cols.COL_FINDING_CONTRIBUTION].tolist() == [0.0, None]
    assert table[cols.COL_FINDING_ANSWER].tolist() == ["no", None]
    assert table[cols.COL_FINDING_SENSITIVITY_REASON].tolist() == ["Spouse makes marriage inferable", "Broad geography"]


def test_saved_contribution_and_question_metadata_survive_without_assessments() -> None:
    row = _row()
    del row[cols.COL_ENTITY_SENSITIVITY]
    row[cols.COL_PRIVACY_LEAKAGE_CONTRIBUTIONS] = [
        {"id": 2, "entity_value": "Catholic", "entity_label": "religious_belief", "leakage_contribution": 0.123}
    ]
    table = _result(pd.DataFrame([row])).trace_dataframe.pipe(build_privacy_findings)
    assert table[cols.COL_FINDING_ENTITY_VALUE].tolist() == ["Catholic", "Alice"]
    assert table.at[0, cols.COL_FINDING_CONTRIBUTION] == 0.123
    assert table.at[0, cols.COL_FINDING_SENSITIVITY] == "medium"


@pytest.mark.parametrize("failure", ["error", "missing_answer", "null_confidence"])
def test_unavailable_answers_are_not_passes_or_zero_contributions(failure: str) -> None:
    row = _row()
    if failure == "error":
        row[cols.COL_PRIVACY_QA_ERROR] = "Model unavailable"
        row[cols.COL_PRIVACY_QA_REANSWER] = None
    elif failure == "missing_answer":
        row[cols.COL_PRIVACY_QA_REANSWER]["answers"] = []
    else:
        for answer in row[cols.COL_PRIVACY_QA_REANSWER]["answers"]:
            answer["confidence"] = None
    table = _result(pd.DataFrame([row])).trace_dataframe.pipe(build_privacy_findings)
    assert table[cols.COL_FINDING_STATUS].tolist() == ["unavailable", "unavailable"]
    assert table[cols.COL_FINDING_CONTRIBUTION].tolist() == [None, None]
    assert all(table[cols.COL_FINDING_ERROR])


def test_failed_upstream_record_with_no_targets_remains_visible() -> None:
    table = _result(
        pd.DataFrame([{cols.RECORD_ID_COLUMN: "failed", cols.COL_PRIVACY_QA_ERROR: "Domain classification failed"}])
    ).trace_dataframe.pipe(build_privacy_findings)
    assert len(table) == 1
    assert table.at[0, cols.COL_FINDING_RECORD_ID] == "failed"
    assert table.at[0, cols.COL_FINDING_STATUS] == "unavailable"
    assert pd.isna(table.at[0, cols.COL_FINDING_ENTITY_VALUE])


def test_unevaluated_detected_targets_remain_visible() -> None:
    row = {
        cols.RECORD_ID_COLUMN: "before-evaluation",
        cols.COL_ENTITIES_BY_VALUE: {"entities_by_value": [{"value": "Alice", "labels": ["first_name", "full_name"]}]},
    }
    table = _result(pd.DataFrame([row])).trace_dataframe.pipe(build_privacy_findings)
    assert table[cols.COL_FINDING_STATUS].tolist() == ["not_evaluated", "not_evaluated"]
    assert table[cols.COL_FINDING_ENTITY_LABEL].tolist() == ["first_name", "full_name"]
    assert table[cols.COL_FINDING_CONTRIBUTION].tolist() == [None, None]


@pytest.mark.parametrize(
    "trace",
    [
        pd.DataFrame(),
        pd.DataFrame([{cols.COL_ENTITY_SENSITIVITY: {"entities": []}}]),
        pd.DataFrame([{cols.COL_ENTITIES_BY_VALUE: {"entities_by_value": []}}]),
    ],
)
def test_empty_findings_have_stable_columns(trace: pd.DataFrame) -> None:
    table = _result(trace).trace_dataframe.pipe(build_privacy_findings)
    assert table.empty
    assert table.columns.tolist() == list(PRIVACY_FINDING_COLUMNS)


def test_duplicate_question_ids_are_unavailable_instead_of_misjoined() -> None:
    row = _row()
    row[cols.COL_PRIVACY_QA]["items"][0]["id"] = 1
    table = _result(pd.DataFrame([row])).trace_dataframe.pipe(build_privacy_findings)
    assert table[cols.COL_FINDING_STATUS].tolist() == ["unavailable"]
    assert table.at[0, cols.COL_FINDING_CONTRIBUTION] is None


@pytest.mark.parametrize("preview", [False, True])
def test_public_findings_view_has_single_inference_column_and_hides_details(preview: bool) -> None:
    row = _row()
    result = _result(pd.DataFrame([row]), preview)
    table = result.privacy_findings
    assert table.columns.tolist() == [
        cols.COL_FINDING_RECORD_INDEX,
        cols.COL_FINDING_RECORD_ID,
        cols.COL_FINDING_ENTITY_VALUE,
        cols.COL_FINDING_ENTITY_LABEL,
        cols.COL_FINDING_SOURCE,
        cols.COL_FINDING_CATEGORY,
        cols.COL_FINDING_SENSITIVITY,
        cols.COL_FINDING_INFERRED,
        cols.COL_FINDING_CONFIDENCE,
        cols.COL_FINDING_REASON,
        cols.COL_FINDING_EVIDENCE,
        cols.COL_FINDING_CONTRIBUTION,
    ]
    assert table[cols.COL_FINDING_INFERRED].tolist() == ["yes"]
    assert table.loc[table[cols.COL_FINDING_INFERRED] == "yes", cols.COL_FINDING_ENTITY_VALUE].tolist() == ["Catholic"]
    table.at[1, cols.COL_FINDING_EVIDENCE].append("changed")
    assert result.privacy_findings.at[1, cols.COL_FINDING_EVIDENCE] == ["attends Mass"]
    assert row[cols.COL_ENTITY_SENSITIVITY]["entities"][0]["category"] == "direct_identifier"


def test_public_view_distinguishes_unavailable_from_negative_answers() -> None:
    row = _row()
    row[cols.COL_PRIVACY_QA_ERROR] = "Evaluation unavailable"
    table = _result(pd.DataFrame([row])).privacy_findings
    assert table[cols.COL_FINDING_INFERRED].tolist() == ["Unavailable", "Unavailable"]
    assert table[cols.COL_FINDING_CONTRIBUTION].tolist() == [None, None]


def test_public_view_marks_unevaluated_targets() -> None:
    row = {cols.COL_ENTITIES_BY_VALUE: {"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]}}
    table = _result(pd.DataFrame([row])).privacy_findings
    assert table.empty


@pytest.mark.parametrize("preview", [False, True])
def test_summary_pools_weights_and_excludes_unavailable_records(preview: bool) -> None:
    first = _row()
    second = _row()
    second[cols.COL_PRIVACY_QA_REANSWER]["answers"][0]["answer"] = "no"
    second[cols.COL_PRIVACY_QA_REANSWER]["answers"][1]["answer"] = "yes"
    failed = _row()
    failed[cols.COL_PRIVACY_QA_ERROR] = "Failed"
    summary = _result(pd.DataFrame([first, second, failed]), preview).privacy_summary
    assert summary["records_evaluated"] == 2
    assert summary["records_unavailable"] == 1
    assert summary["entities_evaluated"] == 4
    assert summary["entities_leaked"] == 2
    assert summary["high_sensitivity_entities_leaked"] == 1
    assert summary["leakage_mass"] == pytest.approx(1.48)
    assert summary["weighted_leakage_rate"] == pytest.approx(1.48 / 3.2)
    assert summary["leakage_mass_by_sensitivity"] == {"high": 1.0, "medium": 0.48, "low": 0.0}


def test_summary_empty_evaluation_and_unevaluated_records() -> None:
    empty = {cols.COL_PRIVACY_QA: {"items": []}, cols.COL_PRIVACY_QA_REANSWER: {"answers": []}}
    summary = _result(pd.DataFrame([empty, {}])).privacy_summary
    assert summary["records_evaluated"] == 1
    assert summary["records_not_evaluated"] == 1
    assert summary["weighted_leakage_rate"] == 0.0
    assert _result(pd.DataFrame([{}])).privacy_summary["weighted_leakage_rate"] is None


@pytest.mark.parametrize("preview", [False, True])
def test_report_displays_summary_then_only_problems_with_full_escaped_text(
    preview: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = _row()
    reason = "<script>bad</script>" + "long reason " * 100
    row[cols.COL_PRIVACY_QA_REANSWER]["answers"][0]["reason"] = reason
    result = _result(pd.DataFrame([row]), preview)
    rendered: list[str] = []
    monkeypatch.setattr("anonymizer.interface.results._display_html", rendered.append)
    result.display_privacy_report()
    report = rendered[0]
    assert report.index("Privacy evaluation summary") < report.index("Reported problems")
    assert "<th>Type</th>" in report
    assert "<td>quasi</td>" in report
    assert "<td>med</td>" in report
    assert "Catholic" in report
    assert "Alice" not in report
    assert "&lt;script&gt;bad&lt;/script&gt;" in report
    assert "long reason " * 99 in report
    row[cols.COL_PRIVACY_QA_REANSWER]["answers"][0]["answer"] = "no"
    clean = _result(pd.DataFrame([row]), preview)
    clean.display_privacy_report()
    assert "No reported leaks or unavailable evaluations." in rendered[1]


def test_summary_uses_pooled_denominator_rather_than_mean_record_rates() -> None:
    first = _row()
    second = _row()
    second[cols.COL_ENTITY_SENSITIVITY]["entities"] = second[cols.COL_ENTITY_SENSITIVITY]["entities"][:1]
    second[cols.COL_PRIVACY_QA]["items"] = [
        question for question in second[cols.COL_PRIVACY_QA]["items"] if question["id"] == 1
    ]
    second[cols.COL_PRIVACY_QA_REANSWER]["answers"] = [
        answer for answer in second[cols.COL_PRIVACY_QA_REANSWER]["answers"] if answer["id"] == 1
    ]
    second[cols.COL_PRIVACY_QA_REANSWER]["answers"][0]["answer"] = "yes"
    summary = _result(pd.DataFrame([first, second])).privacy_summary
    assert summary["weighted_leakage_rate"] == pytest.approx(1.48 / 2.6)
    assert summary["weighted_leakage_rate"] != pytest.approx((0.48 / 1.6 + 1.0) / 2)


@pytest.mark.parametrize("preview", [False, True])
def test_rewrite_report_includes_saved_risk_settings(preview: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    from anonymizer.config.rewrite import EvaluationCriteria, RiskTolerance

    result = _result(pd.DataFrame([_row()]), preview)
    result.rewrite_evaluation = EvaluationCriteria(risk_tolerance=RiskTolerance.moderate)
    summary = result.privacy_summary
    assert summary["risk_tolerance"] == "moderate"
    assert summary["repair_threshold"] == result.rewrite_evaluation.repair_threshold
    rendered: list[str] = []
    monkeypatch.setattr("anonymizer.interface.results._display_html", rendered.append)
    result.display_privacy_report()
    assert "Risk tolerance" in rendered[0]
    assert "moderate" in rendered[0]
    assert "Repair threshold (leakage mass)" in rendered[0]
    result.rewrite_evaluation = None
    assert "risk_tolerance" not in result.privacy_summary


@pytest.mark.parametrize("preview", [False, True])
@pytest.mark.parametrize("as_string", [False, True])
def test_export_privacy_report_is_standalone_and_matches_display(
    tmp_path: Path, preview: bool, as_string: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = _row()
    row[cols.COL_PRIVACY_QA_REANSWER]["answers"][0]["reason"] = "Évidence <script>alert(1)</script>"
    result = _result(pd.DataFrame([row]), preview)
    rendered: list[str] = []
    monkeypatch.setattr("anonymizer.interface.results._display_html", rendered.append)
    result.display_privacy_report()
    path = tmp_path / "report.html"
    path.write_text("old contents", encoding="utf-8")
    exported = result.export_privacy_report(str(path) if as_string else path)
    assert exported == path
    document = path.read_text(encoding="utf-8")
    assert document.startswith("<!doctype html>")
    assert '<meta charset="utf-8">' in document
    assert rendered[0] in document
    assert "Évidence &lt;script&gt;alert(1)&lt;/script&gt;" in document
    assert "Alice" not in document
    assert document.endswith("</body></html>")
    assert row[cols.COL_PRIVACY_QA_REANSWER]["answers"][0]["reason"] == "Évidence <script>alert(1)</script>"


def test_export_privacy_report_propagates_file_errors(tmp_path: Path) -> None:
    result = _result(pd.DataFrame([_row()]))
    with pytest.raises(FileNotFoundError):
        result.export_privacy_report(tmp_path / "missing" / "report.html")


@pytest.mark.parametrize("preview", [False, True])
@pytest.mark.parametrize("counts,expected", [([0, 1, 2], 2), ([0, 0, 0], 0), ([None, None, None], None)])
def test_rewrite_summary_reports_repair_limit_and_maximum_taken(
    preview: bool, counts: list[int | None], expected: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from anonymizer.config.rewrite import EvaluationCriteria

    trace = pd.DataFrame([_row(), _row(), _row()])
    trace[cols.COL_REPAIR_ITERATIONS] = counts
    result = _result(trace, preview)
    result.rewrite_evaluation = EvaluationCriteria(max_repair_iterations=4)
    assert result.privacy_summary["max_repair_iterations"] == 4
    assert result.privacy_summary["repair_iterations_taken"] == expected
    rendered: list[str] = []
    monkeypatch.setattr("anonymizer.interface.results._display_html", rendered.append)
    result.display_privacy_report()
    assert "Max repair iterations" in rendered[0]
    assert "Repair iterations taken (maximum across records)" in rendered[0]


@pytest.mark.parametrize("preview", [False, True])
def test_privacy_findings_join_sources_by_value_and_label(preview: bool) -> None:
    row = _row()
    row[cols.COL_FINAL_ENTITIES] = {
        "entities": [
            {"value": "Catholic", "label": "religious_belief", "source": "detector"},
            {"value": "Catholic", "label": "religious_belief", "source": "augmenter"},
            {"value": "Catholic", "label": "other_label", "source": "other_source"},
        ]
    }
    result = _result(pd.DataFrame([row]), preview)
    assert result.privacy_findings.iloc[0][cols.COL_FINDING_SOURCE] == "augment, detector"
    row[cols.COL_ENTITY_SENSITIVITY]["entities"][1]["source"] = "latent"
    assert _result(pd.DataFrame([row]), preview).privacy_findings.iloc[0][cols.COL_FINDING_SOURCE] == "latent"
    assert _result(pd.DataFrame([_row()]), preview).privacy_findings.iloc[0][cols.COL_FINDING_SOURCE] == "Unknown"
