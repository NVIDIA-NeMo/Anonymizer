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
    metadata.setdefault("privacy_inference_enabled", True)
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
    assert issues.iloc[2]["Source"] == "N/A"


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


@pytest.mark.parametrize("evaluated", [False, True])
def test_inference_summary_defines_inference(tmp_path: Path, evaluated: bool) -> None:
    row = {}
    if evaluated:
        row = {cols.COL_ENTITIES_BY_VALUE: {"entities_by_value": []}, cols.COL_LEAKAGE_MASS: 0}
    result = _result(pd.DataFrame([row]), replace_method=Redact())
    document = result.export_evaluation_report(tmp_path / "report.html").read_text()
    definition = "Privacy inference checks whether each original entity value can still be identified or inferred"
    assert definition in document
    assert document.index("Privacy Inference Summary") < document.index(definition)
    assert "including direct mentions and contextual clues." in document


@pytest.mark.parametrize("preview", [False, True])
def test_failure_summary_groups_records_and_escapes_full_details(
    preview: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from anonymizer.engine.ndd.adapter import FailedRecord

    reason = "Missing label <script>unsafe</script> " + "full saved reason " * 50
    failures = [
        FailedRecord(record_id="r1", step="entity-detection", reason=reason),
        FailedRecord(record_id="r1", step="entity-detection", reason=reason),
        FailedRecord(record_id="r2", step="entity-detection", reason=reason),
        FailedRecord(record_id="r1", step="evaluation", reason="Timeout"),
    ]
    result = _result(pd.DataFrame([{}]), preview, replace_method=Redact())
    result.failed_records = failures
    rendered: list[str] = []
    monkeypatch.setattr("anonymizer.interface.results._display_html", rendered.append)
    result.display_evaluation_report()
    document = result.export_evaluation_report(tmp_path / "report.html").read_text()
    assert rendered[0] in document
    assert "<strong>Failed records:</strong> 2" in document
    assert document.index("Failed Record Summary") < document.index("Privacy &amp; Quality Judge Summary")
    assert "<details><summary>Failed record details</summary>" in document
    assert "<td>2</td>" in document
    assert "<td>1</td>" in document
    assert "<script>unsafe</script>" not in document
    assert "&lt;script&gt;unsafe&lt;/script&gt;" in document
    assert "full saved reason " * 49 in document
    assert document.count("<td>r1</td>") == 2  # two stages, duplicate detection entry removed
    assert document.count("<td>r2</td>") == 1
    assert "<td>evaluation</td><td>Timeout</td><td>1</td>" in document
    assert len(result.failed_records) == 4  # presentation never mutates saved failures


def test_no_failure_summary_when_there_are_no_saved_failures(tmp_path: Path) -> None:
    result = _result(pd.DataFrame([{}]), replace_method=Redact())
    document = result.export_evaluation_report(tmp_path / "report.html").read_text()
    assert "<strong>Failed records:</strong> 0" in document
    assert "Failed Record Summary" not in document
    assert "Failed record details" not in document


def test_judge_entity_source_does_not_guess_for_missed_entities() -> None:
    trace = pd.DataFrame(
        [
            {
                cols.COL_FINAL_ENTITIES: {
                    "entities": [{"value": "Alice", "label": "first_name", "source": "augmenter"}]
                },
                cols.COL_TYPE_FIDELITY_VALID: False,
                cols.COL_TYPE_FIDELITY_INVALID_REPLACEMENTS: [
                    {"original": "Alice", "label": "first_name", "synthetic": "123", "reasoning": "Not a name"}
                ],
                cols.COL_ENTITY_COVERAGE: 0.5,
                cols.COL_MISSED_ENTITIES: [{"value": "Denver", "label": "city", "reasoning": "Missed"}],
            }
        ]
    )
    _, issues = build_judge_report(trace)
    assert issues["Source"].tolist() == ["N/A", "augment"]


@pytest.mark.parametrize("preview", [False, True])
def test_models_section_uses_saved_metadata_and_escapes_values(preview: bool, tmp_path: Path) -> None:
    result = _result(pd.DataFrame([{}]), preview, replace_method=Substitute())
    result.model_metadata = [
        {
            "phase": "run",
            "group": "Anonymization",
            "task": "Entity detection",
            "alias": 'alias"bad',
            "model": "example/detector",
            "provider": "local",
        },
        {
            "phase": "evaluate",
            "group": "Evaluation",
            "task": "Entity coverage",
            "alias": "coverage",
            "model": "<script>judge</script>",
            "provider": "openrouter",
        },
    ]
    document = result.export_evaluation_report(tmp_path / "report.html").read_text()
    assert document.index("Models Used") < document.index("Privacy &amp; Quality Judge Summary")
    assert document.index("Anonymization tasks") < document.index("Evaluation tasks")
    assert "example/detector" in document and "openrouter" in document
    assert "&lt;script&gt;judge&lt;/script&gt;" in document
    assert "<script>judge</script>" not in document
    assert "Alias: alias&quot;bad" in document


def test_models_section_marks_missing_older_metadata(tmp_path: Path) -> None:
    result = _result(pd.DataFrame([{}]), replace_method=Redact())
    document = result.export_evaluation_report(tmp_path / "report.html").read_text()
    assert "Models Used" in document
    assert "Model information unavailable." in document


@pytest.mark.parametrize("preview", [False, True])
@pytest.mark.parametrize("labels", [None, ["first_name", "custom_case_id", "<script>unsafe</script>"]])
def test_labels_used_distinguishes_defaults_and_lists_all_custom_labels(
    preview: bool, labels: list[str] | None, tmp_path: Path
) -> None:
    result = _result(pd.DataFrame([{}]), preview, replace_method=Redact(), entity_labels=labels)
    document = result.export_evaluation_report(tmp_path / "report.html").read_text()
    assert (
        document.index("Models Used")
        < document.index("Labels Used")
        < document.index("Privacy &amp; Quality Judge Summary")
    )
    if labels is None:
        assert "Default label list." in document
        assert "Custom label list:" not in document
    else:
        assert "Custom label list:" in document
        assert "<code>first_name</code>" in document
        assert "<code>custom_case_id</code>" in document
        assert "<code>&lt;script&gt;unsafe&lt;/script&gt;</code>" in document
        assert "<script>unsafe</script>" not in document


def test_default_labels_with_exclusions_are_explicit_in_report(tmp_path: Path) -> None:
    result = _result(pd.DataFrame([{}]), replace_method=Redact(), excluded_entity_labels=["city", "country"])
    document = result.export_evaluation_report(tmp_path / "report.html").read_text()
    assert "Default label list." in document
    assert "Excluded labels: <code>city</code>, <code>country</code>" in document


@pytest.mark.parametrize("preview", [False, True])
@pytest.mark.parametrize(
    "batch_size,display", [(5, "5"), (2, "2"), (None, "None (all questions in one call per record)")]
)
def test_report_shows_saved_batching_settings(
    preview: bool, batch_size: int | None, display: str, tmp_path: Path
) -> None:
    result = _result(pd.DataFrame([{}]), preview, replace_method=Redact())
    result.execution_settings = {"privacy_qa_batch_size": batch_size, "validation_max_entities_per_call": 17}
    document = result.export_evaluation_report(tmp_path / "report.html").read_text()
    assert (
        document.index("Labels Used")
        < document.index("Batching Settings")
        < document.index("Privacy &amp; Quality Judge Summary")
    )
    assert f"<th>privacy_qa_batch_size</th><td>{display}</td>" in document
    assert "<th>validation_max_entities_per_call</th><td>17</td>" in document


def test_report_does_not_invent_missing_batch_settings(tmp_path: Path) -> None:
    result = _result(pd.DataFrame([{}]), replace_method=Redact())
    document = result.export_evaluation_report(tmp_path / "report.html").read_text()
    assert "<th>privacy_qa_batch_size</th><td>Not evaluated</td>" in document
    assert "<th>validation_max_entities_per_call</th><td>Unavailable</td>" in document


@pytest.mark.parametrize("is_rewrite", [False, True])
@pytest.mark.parametrize("choice", [None, False, True])
def test_report_mode_default_and_saved_inference_choice(is_rewrite: bool, choice: bool | None, tmp_path: Path) -> None:
    result = _result(
        pd.DataFrame([{}]),
        privacy_inference_enabled=choice,
        replace_method=None if is_rewrite else Substitute(),
        rewrite_evaluation=EvaluationCriteria() if is_rewrite else None,
    )
    expected = is_rewrite if choice is None else choice
    document = result.export_evaluation_report(tmp_path / "report.html").read_text()
    assert ("Privacy Inference Summary" in document) is expected
    assert ("Privacy Inference Issues" in document) is expected
    assert "Privacy &amp; Quality Judge Issues" in document


def test_disabled_inference_report_never_reads_saved_inference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _result(pd.DataFrame([{}]), replace_method=Substitute(), privacy_inference_enabled=False)

    def unexpected(_self: object) -> None:
        raise AssertionError("Inference should not be parsed for judge-only reports")

    monkeypatch.setattr(type(result), "privacy_summary", property(unexpected))
    monkeypatch.setattr(type(result), "privacy_findings", property(unexpected))
    result.export_evaluation_report(tmp_path / "report.html")


def test_rewrite_judge_issue_sections_have_only_applicable_columns(tmp_path: Path) -> None:
    trace = pd.DataFrame(
        [
            {
                cols.RECORD_ID_COLUMN: "record-a",
                cols.COL_ENTITY_COVERAGE: 0.5,
                cols.COL_ENTITY_COVERAGE_N_CANDIDATES: 2,
                cols.COL_MISSED_ENTITIES: [{"value": "BA in English", "label": "degree", "reasoning": "Degree missed"}],
                cols.COL_JUDGE_EVALUATION: {
                    "privacy": {"score": "high", "reasoning": "Protected"},
                    "quality": {"score": "medium", "reasoning": "Some meaning lost <script>bad</script>"},
                    "style": {"score": "low", "reasoning": "Awkward phrasing"},
                },
            },
            {
                cols.RECORD_ID_COLUMN: "record-b",
                cols.COL_ENTITY_COVERAGE: None,
                cols.COL_JUDGE_EVALUATION: {"privacy": {"score": "high"}, "quality": {"score": "high"}},
            },
        ]
    )
    result = _result(trace, rewrite_evaluation=EvaluationCriteria(), privacy_inference_enabled=False)
    document = result.export_evaluation_report(tmp_path / "report.html").read_text()

    def section(heading: str) -> str:
        return document.split(f"<h3>{heading}</h3>", 1)[1].split("</section>", 1)[0]

    coverage = section("Entity Coverage Misses")
    assert "BA in English" in coverage
    assert "Unavailable: Saved judge result unavailable." in coverage
    for column in ["Row", "Record ID", "Entity", "Label", "Reason"]:
        assert f"<th>{column}</th>" in coverage
    for column in ["Source", "Replacement", "Result", "Check", "Rating"]:
        assert f"<th>{column}</th>" not in coverage
    assert "<p>None.</p>" in section("Rewrite Privacy Issues")
    quality = section("Rewrite Quality Issues")
    assert "<th>Rating</th>" in quality
    assert "<td>medium</td>" in quality
    assert "&lt;script&gt;bad&lt;/script&gt;" in quality
    for column in ["Entity", "Label", "Source", "Replacement", "Check", "Result"]:
        assert f"<th>{column}</th>" not in quality
    style = section("Rewrite Style Issues")
    assert "<td>low</td>" in style
    assert "<td>Unavailable</td>" in style
    assert "Detection Validity Issues" not in document
    assert "Privacy Inference Issues" not in document


def test_rewrite_judges_not_run_show_not_evaluated_and_optional_detection_has_own_section(tmp_path: Path) -> None:
    result = _result(pd.DataFrame([{}]), rewrite_evaluation=EvaluationCriteria(), privacy_inference_enabled=False)
    document = result.export_evaluation_report(tmp_path / "not-run.html").read_text()
    for heading in [
        "Entity Coverage Misses",
        "Rewrite Privacy Issues",
        "Rewrite Quality Issues",
        "Rewrite Style Issues",
    ]:
        assert f"<h3>{heading}</h3><p>Not evaluated.</p>" in document
    result.trace_dataframe[cols.COL_DETECTION_VALID] = [True]
    document = result.export_evaluation_report(tmp_path / "detection.html").read_text()
    assert "<h3>Detection Validity Issues</h3><p>None.</p>" in document


def test_replace_keeps_combined_judge_issue_table(tmp_path: Path) -> None:
    result = _result(
        pd.DataFrame(
            [
                {
                    cols.COL_ENTITY_COVERAGE: 0.5,
                    cols.COL_MISSED_ENTITIES: [{"value": "Alice", "label": "first_name", "reasoning": "Missed"}],
                }
            ]
        ),
        replace_method=Substitute(),
        privacy_inference_enabled=False,
    )
    document = result.export_evaluation_report(tmp_path / "replace.html").read_text()
    issues = document.split("<h2>Privacy &amp; Quality Judge Issues</h2>", 1)[1]
    assert "Entity Coverage Misses" not in issues
    assert "Rewrite Privacy Issues" not in issues
    assert "<th>Check</th>" in issues and "<th>Replacement</th>" in issues


@pytest.mark.parametrize("preview", [False, True])
@pytest.mark.parametrize("strategy", [Substitute(), Redact(), Annotate(), Hash()])
def test_replace_issues_omit_result_and_mark_unavailable_in_reason(
    preview: bool, strategy: Any, tmp_path: Path
) -> None:
    trace = pd.DataFrame(
        [
            {
                cols.COL_ENTITY_COVERAGE: 0.5,
                cols.COL_MISSED_ENTITIES: [{"value": "Alice", "label": "name", "reasoning": "Name missed"}],
            },
            {cols.COL_ENTITY_COVERAGE: None},
        ]
    )
    result = _result(trace, preview, replace_method=strategy, privacy_inference_enabled=False)
    document = result.export_evaluation_report(tmp_path / "report.html").read_text()
    issues_html = document.split("<h2>Privacy &amp; Quality Judge Issues</h2>", 1)[1]
    assert "<th>Result</th>" not in issues_html
    assert "<td>Issue</td>" not in issues_html
    assert "Evaluation unavailable: Saved judge result unavailable." in issues_html
    assert "Name missed" in issues_html
    assert "<th>Reason</th>" in issues_html
    _, saved_issues = build_judge_report(trace)
    assert saved_issues["Result"].tolist() == ["Issue", "Unavailable"]


@pytest.mark.parametrize("strict", [True, False, None])
def test_rewrite_strict_protection_is_visible_even_without_inference(strict: bool | None, tmp_path: Path) -> None:
    result = _result(pd.DataFrame([{}]), rewrite_evaluation=EvaluationCriteria(), privacy_inference_enabled=False)
    if strict is not None:
        result.execution_settings["strict_entity_protection"] = strict
    document = result.export_evaluation_report(tmp_path / "report.html").read_text()
    expected = str(strict) if strict is not None else "Unavailable"
    assert "Rewrite Settings" in document
    assert f"<th>strict_entity_protection</th><td>{expected}</td>" in document


@pytest.mark.parametrize("preview", [False, True])
@pytest.mark.parametrize(
    "summary", [None, "", "  \n\t ", "Paragraph one.\n\nParagraph two <script>bad</script>\n" + "Long context. " * 200]
)
def test_data_summary_is_expandable_escaped_and_complete(
    preview: bool, summary: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import html

    result = _result(pd.DataFrame([{}]), preview, replace_method=Redact(), data_summary=summary)
    rendered: list[str] = []
    monkeypatch.setattr("anonymizer.interface.results._display_html", rendered.append)
    result.display_evaluation_report()
    document = result.export_evaluation_report(tmp_path / "report.html").read_text()
    assert rendered[0] in document
    assert document.index("Labels Used") < document.index("Data Summary") < document.index("Batching Settings")
    if summary is not None and summary.strip():
        assert "<details><summary>Show data summary</summary>" in document
        assert html.escape(summary) in document
        assert "<script>bad</script>" not in document
        assert "white-space:pre-wrap" in document
    else:
        assert "<h3>Data Summary</h3><p>None.</p>" in document
        assert "Show data summary" not in document
    assert result.data_summary == summary


@pytest.mark.parametrize("preview", [False, True])
def test_run_warnings_group_occurrences_and_unique_records_with_escaped_details(preview: bool, tmp_path: Path) -> None:
    result = _result(pd.DataFrame([{}]), preview, replace_method=Redact())
    base = {
        "stage": "Detection",
        "code": "grounding",
        "message": "Grounding correction",
        "unit": "attempts",
        "count": 1,
    }
    result.warning_metadata = {
        "run": [
            {**base, "record_id": "r1", "detail": "<script>unsafe</script>"},
            {**base, "record_id": "r1", "detail": "second attempt"},
            {**base, "record_id": "r2", "detail": "first attempt"},
        ]
    }
    document = result.export_evaluation_report(tmp_path / "warnings.html").read_text()
    assert document.index("Run Warnings") < document.index("Models Used")
    assert "<details><summary>Show warning summary</summary>" in document
    assert "<summary>Warning details</summary>" in document
    assert "<td>3 attempts</td>" in document
    assert "<td>2</td>" in document
    assert "&lt;script&gt;unsafe&lt;/script&gt;" in document
    assert "<script>unsafe</script>" not in document
    assert len(result.warning_metadata["run"]) == 3


@pytest.mark.parametrize(
    "metadata,expected",
    [
        ({}, "Warning information unavailable."),
        ({"run": []}, "None."),
        ({"evaluate": []}, "Warning information unavailable for the original run."),
    ],
)
def test_run_warnings_distinguish_clean_run_from_unknown_history(
    metadata: dict[str, Any], expected: str, tmp_path: Path
) -> None:
    result = _result(pd.DataFrame([{}]), replace_method=Redact())
    result.warning_metadata = metadata
    document = result.export_evaluation_report(tmp_path / "warnings.html").read_text()
    section = document.split("<h3>Run Warnings</h3>", 1)[1].split("</section>", 1)[0]
    assert expected in section
    if not metadata or "run" not in metadata:
        assert "<p>None.</p>" not in section


def test_legacy_result_without_warning_attribute_shows_unknown_history(tmp_path: Path) -> None:
    result = _result(pd.DataFrame([{}]), replace_method=Redact())
    del result.warning_metadata
    document = result.export_evaluation_report(tmp_path / "legacy.html").read_text()
    assert "Warning information unavailable." in document


@pytest.mark.parametrize("preview", [False, True])
def test_report_generation_date_appears_under_title(
    preview: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import datetime, timezone
    from types import SimpleNamespace

    fixed = datetime(2026, 11, 17, 12, tzinfo=timezone.utc)
    monkeypatch.setattr("anonymizer.interface.evaluation_report.datetime", SimpleNamespace(now=lambda: fixed))
    result = _result(pd.DataFrame([{}]), preview, replace_method=Redact())
    document = result.export_evaluation_report(tmp_path / "dated.html").read_text()
    date_line = f"<p><strong>Date generated:</strong> {fixed.astimezone().date().isoformat()}</p>"
    assert date_line in document
    assert (
        document.index("<h1>NeMo Anonymizer Evaluation Report</h1>")
        < document.index(date_line)
        < document.index("<strong>Strategy:</strong>")
    )


def test_execution_times_are_saved_only_and_missing_times_are_unavailable() -> None:
    from anonymizer.interface.evaluation_report import _render_execution_times

    report = _render_execution_times(
        {
            "run": {"Detector (GLiNER)": 1.25, "LLM validation": None, "Total elapsed": 12.5},
            "evaluate": {"Privacy <QA>": 0.0, "Total elapsed": 3.0},
        }
    )
    assert "Execution Time" in report
    assert "1.25 s" in report
    assert "12.50 s" in report
    assert "0.00 s" in report
    assert "Unavailable" in report
    assert "Privacy &lt;QA&gt;" in report
    assert "should not be added" in report
    assert _render_execution_times({}).count("<td>Unavailable</td>") == 2


@pytest.mark.parametrize("preview", [True, False])
def test_execution_times_appear_in_full_report_and_old_results(preview: bool, tmp_path: Path) -> None:
    result = _result(
        pd.DataFrame(),
        preview,
        execution_timings={
            "run": {"Total elapsed": 20.0},
            "evaluate": {"Total elapsed": 5.0},
        },
    )
    path = result.export_evaluation_report(tmp_path / "timed.html")
    report = path.read_text()
    assert "Total anonymization time</th><td>20.00 s" in report
    assert "Total evaluation time</th><td>5.00 s" in report
    del result.execution_timings
    assert "Execution Time" in result._evaluation_report_html()
    assert "Total anonymization time</th><td>Unavailable" in result._evaluation_report_html()


@pytest.mark.parametrize("preview", [False, True])
@pytest.mark.parametrize("include_inference", [False, True])
def test_rewrite_leave_as_is_count_and_trailing_details(preview: bool, include_inference: bool, tmp_path: Path) -> None:
    item = {
        "entity_value": "USA <value>",
        "entity_label": "country",
        "category": "quasi_identifier",
        "sensitivity": "low",
        "source": "tagged",
        "protection_method_suggestion": "leave_as_is",
        "low_sensitivity_reason": "Broad geography <reason>",
    }
    trace = pd.DataFrame(
        [
            {
                cols.RECORD_ID_COLUMN: "record-a",
                cols.COL_SENSITIVITY_DISPOSITION: {"sensitivity_disposition": [item, dict(item)]},
                cols.COL_PRIVACY_QA: {"items": []},
                cols.COL_PRIVACY_QA_REANSWER: {"answers": []},
            }
        ]
    )
    result = _result(
        trace, preview, rewrite_evaluation=EvaluationCriteria(), privacy_inference_enabled=include_inference
    )
    assert result.privacy_summary["entities_left_as_is"] == 1
    path = result.export_evaluation_report(tmp_path / "leave-as-is.html")
    report = path.read_text()
    assert "Entities assigned leave_as_is</th><td>1</td>" in report
    assert report.index("Entities Assigned leave_as_is") > report.index("Privacy &amp; Quality Judge Issues")
    if include_inference:
        assert report.index("Entities Assigned leave_as_is") > report.index("Privacy Inference Issues")
    assert "USA &lt;value&gt;" in report
    assert "Broad geography &lt;reason&gt;" in report
    assert "USA <value>" not in report
    privacy = result.export_privacy_report(tmp_path / "privacy.html").read_text()
    assert "Entities assigned leave_as_is</th><td>1</td>" in privacy
    assert "Entities Assigned leave_as_is" in privacy


@pytest.mark.parametrize("payload, expected", [(None, "Unavailable"), ({"sensitivity_disposition": []}, "0")])
def test_rewrite_leave_as_is_zero_and_missing_have_no_details(payload: object, expected: str) -> None:
    trace = pd.DataFrame([{}] if payload is None else [{cols.COL_SENSITIVITY_DISPOSITION: payload}])
    result = _result(trace, rewrite_evaluation=EvaluationCriteria())
    report = result._evaluation_report_html()
    assert f"Entities assigned leave_as_is</th><td>{expected}</td>" in report
    assert "Entities Assigned leave_as_is" not in report


def test_replace_report_has_no_leave_as_is_section() -> None:
    result = _result(pd.DataFrame(), replace_method=Substitute())
    assert "leave_as_is" not in result._evaluation_report_html()
