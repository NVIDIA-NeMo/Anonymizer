# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Presentation and standalone export for the combined evaluation report."""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.interface.privacy_report import (
    PRIVACY_INFERENCE_DESCRIPTION,
    privacy_report_styles,
    render_leave_as_is_section,
    render_privacy_report_sections,
)


def _render_failed_records(failures: list[FailedRecord]) -> tuple[int, str]:
    """Group saved failures, counting each record once per stage/reason pair."""
    if not failures:
        return 0, ""
    groups: dict[tuple[str, str], set[str]] = {}
    unique: dict[tuple[str, str, str], FailedRecord] = {}
    for failure in failures:
        groups.setdefault((failure.step, failure.reason), set()).add(failure.record_id)
        unique.setdefault((failure.record_id, failure.step, failure.reason), failure)
    summary_rows = "".join(
        "<tr><td>" + html.escape(stage) + "</td><td>" + html.escape(reason) + f"</td><td>{len(record_ids)}</td></tr>"
        for (stage, reason), record_ids in groups.items()
    )
    detail_rows = "".join(
        "<tr><td>"
        + html.escape(failure.record_id)
        + "</td><td>"
        + html.escape(failure.step)
        + "</td><td>"
        + html.escape(failure.reason)
        + "</td></tr>"
        for failure in unique.values()
    )
    return len({failure.record_id for failure in failures}), (
        '<section class="failed-records"><h3>Failed Record Summary</h3>'
        '<div class="privacy-details"><table><thead><tr>'
        "<th>Stage</th><th>Saved reason</th><th>Records</th></tr></thead><tbody>"
        + summary_rows
        + "</tbody></table></div>"
        "<p>Counts are unique records per stage and reason. A record may appear in more than one group; "
        "the headline counts each failed record once. Reasons are shown as saved; "
        "more specific diagnostics may only be available in the run log.</p>"
        "<details><summary>Failed record details</summary>"
        '<div class="privacy-details"><table><thead><tr>'
        "<th>Record ID</th><th>Stage</th><th>Saved reason</th></tr></thead><tbody>"
        + detail_rows
        + "</tbody></table></div></details></section>"
    )


def _render_run_warnings(metadata: dict[str, list[dict[str, Any]]]) -> str:
    """Summarize structured events without treating missing history as no warnings."""
    heading = "<h3>Run Warnings</h3>"
    if not metadata:
        return "<section>" + heading + "<p>Warning information unavailable.</p></section>"
    events = [event for phase in metadata.values() for event in phase]
    historical_note = "<p>Warning information unavailable for the original run.</p>" if "run" not in metadata else ""
    if not events:
        return "<section>" + heading + historical_note + ("<p>None.</p>" if not historical_note else "") + "</section>"
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for event in events:
        key = tuple(str(event.get(field, "Unknown")) for field in ("stage", "code", "message", "unit"))
        groups.setdefault(key, []).append(event)
    summary_rows = []
    for (stage, _code, message, unit), group in groups.items():
        count = sum(int(event.get("count", 1)) for event in group)
        records = {str(event["record_id"]) for event in group if event.get("record_id") is not None}
        summary_rows.append(
            {
                "Stage": stage,
                "Warning": message,
                "Occurrences": f"{count} {unit}",
                "Affected records": str(len(records)) if records else "Unknown",
            }
        )
    detail_rows = [
        {
            "Stage": event.get("stage"),
            "Record ID": event.get("record_id"),
            "Warning": event.get("message"),
            "Occurrences": f"{event.get('count', 1)} {event.get('unit', '')}",
            "Details": event.get("detail", ""),
        }
        for event in events
    ]
    with pd.option_context("display.max_colwidth", None):
        summary_table = pd.DataFrame(summary_rows).to_html(index=False, escape=True, border=0)
        detail_table = pd.DataFrame(detail_rows).to_html(index=False, escape=True, border=0)
    return (
        "<section>"
        + heading
        + "<details><summary>Show warning summary</summary>"
        + historical_note
        + '<div class="privacy-details">'
        + summary_table
        + "</div>"
        "<p>Occurrences count attempts, entries, spans, or evaluations as labeled; affected records are unique per warning. "
        "One record may appear in multiple groups. This covers saved Anonymizer pipeline diagnostics; "
        "provider, network, and library messages may require the run log.</p>"
        "<details><summary>Warning details</summary>"
        '<div class="privacy-details">' + detail_table + "</div></details></details></section>"
    )


def _render_models_used(metadata: list[dict[str, str]]) -> str:
    """Render compact grouped model assignments from the saved run snapshot."""
    if not metadata:
        return "<section><h3>Models Used</h3><p>Model information unavailable.</p></section>"
    sections = []
    for group in ("Anonymization", "Evaluation"):
        rows = []
        for entry in metadata:
            if entry.get("group") != group:
                continue
            model = html.escape(entry.get("model", "Unknown"))
            provider = html.escape(entry.get("provider", "Unknown"))
            alias = html.escape(entry.get("alias", "Unknown"), quote=True)
            task = html.escape(entry.get("task", "Unknown"))
            rows.append(f'<tr><th>{task}</th><td title="Alias: {alias}">{model} <span>({provider})</span></td></tr>')
        content = "<table>" + "".join(rows) + "</table>" if rows else "<p>Model information unavailable.</p>"
        sections.append("<section><h4>" + group + " tasks</h4>" + content + "</section>")
    return (
        '<section class="models-used"><h3>Models Used</h3><div class="evaluation-summaries">'
        + "".join(sections)
        + "</div></section>"
    )


def _render_labels_used(labels: list[str] | None, excluded: list[str] | None) -> str:
    """Distinguish the default taxonomy from an explicitly supplied label list."""

    def label_list(values: list[str]) -> str:
        return ", ".join("<code>" + html.escape(value) + "</code>" for value in values)

    description = (
        "<p>Default label list.</p>" if labels is None else "<p>Custom label list: " + label_list(labels) + "</p>"
    )
    if excluded:
        description += "<p>Excluded labels: " + label_list(excluded) + "</p>"
    return '<section class="labels-used"><h3>Labels Used</h3>' + description + "</section>"


def _render_data_summary(summary: str | None) -> str:
    """Show saved dataset context as escaped, expandable text, without truncation."""
    body = "<p>None.</p>"
    if summary is not None and summary.strip():
        body = (
            "<details><summary>Show data summary</summary>"
            '<div style="white-space:pre-wrap;overflow-wrap:anywhere;margin:12px 0;">'
            + html.escape(summary)
            + "</div></details>"
        )
    return "<section><h3>Data Summary</h3>" + body + "</section>"


def _render_batching_settings(settings: dict[str, int | None], summary: dict[str, Any]) -> str:
    """Show saved settings; distinguish unbatched QA from missing metadata."""
    validation = str(settings.get("validation_max_entities_per_call", "Unavailable"))
    if "privacy_qa_batch_size" in settings:
        batch = settings["privacy_qa_batch_size"]
        privacy = str(batch) if batch is not None else "None (all questions in one call per record)"
    else:
        privacy = (
            "Unavailable" if summary.get("records_evaluated") or summary.get("records_unavailable") else "Not evaluated"
        )
    return (
        "<section><h3>Batching Settings</h3><table>"
        "<tr><th>privacy_qa_batch_size</th><td>" + html.escape(privacy) + "</td></tr>"
        "<tr><th>validation_max_entities_per_call</th><td>" + html.escape(validation) + "</td></tr>"
        "</table></section>"
    )


def _render_rewrite_judge_issues(summary: pd.DataFrame, issues: pd.DataFrame) -> str:
    """Separate entity checks from record-level Rewrite ratings."""
    sections: list[str] = []
    specs = [
        ("Entity coverage", "Entity Coverage Misses", False),
        ("Rewrite privacy", "Rewrite Privacy Issues", True),
        ("Rewrite quality", "Rewrite Quality Issues", True),
        ("Rewrite style", "Rewrite Style Issues", True),
    ]
    if "Detection validity" in set(summary["Check"]):
        specs.append(("Detection validity", "Detection Validity Issues", False))
    for check, heading, rubric in specs:
        saved = summary[summary["Check"] == check]
        problems = issues[issues["Check"] == check].copy()
        if saved.empty:
            body = "<p>Not evaluated.</p>"
        elif problems.empty:
            evaluated = int(str(saved.iloc[0]["Evaluated"]).split("/")[0])
            body = "<p>None.</p>" if evaluated else "<p>Not evaluated.</p>"
        else:
            if rubric:
                table = problems[["Row", "Record ID", "Result", "Reason"]].rename(columns={"Result": "Rating"})
                table_class = "rewrite-rubric-issues"
            else:
                unavailable = problems["Result"] == "Unavailable"
                problems.loc[unavailable, "Reason"] = "Unavailable: " + problems.loc[unavailable, "Reason"].astype(str)
                columns = ["Row", "Record ID", "Entity / relation", "Label"]
                if check == "Detection validity":
                    columns.append("Source")
                columns.append("Reason")
                table = problems[columns].rename(columns={"Entity / relation": "Entity"})
                table_class = "rewrite-entity-issues"
            with pd.option_context("display.max_colwidth", None):
                body = (
                    '<div class="privacy-details">'
                    + table.to_html(index=False, escape=True, border=0, classes=["rewrite-judge-issues", table_class])
                    + "</div>"
                )
        sections.append("<section><h3>" + heading + "</h3>" + body + "</section>")
    return "".join(sections)


def render_evaluation_report_html(
    *,
    strategy: str,
    include_inference: bool,
    failed_records: list[FailedRecord],
    warning_metadata: dict[str, list[dict[str, Any]]],
    model_metadata: list[dict[str, str]],
    execution_settings: dict[str, int | None],
    entity_labels: list[str] | None,
    excluded_entity_labels: list[str] | None,
    data_summary: str | None,
    privacy_summary: dict[str, Any],
    privacy_issues: pd.DataFrame,
    judge_summary: pd.DataFrame,
    judge_issues: pd.DataFrame,
    execution_timings: dict[str, dict[str, float | None]] | None = None,
    left_as_is: pd.DataFrame | None = None,
) -> str:
    """Render summaries side by side and problem details below, with escaped data."""
    failed_count, failure_summary = _render_failed_records(failed_records)
    styles = privacy_report_styles()
    inference = inference_issues = ""
    if include_inference:
        styles, inference, inference_issues = render_privacy_report_sections(
            privacy_summary, privacy_issues, heading="Privacy Inference Summary"
        )
        if privacy_summary["records_evaluated"] == 0 and privacy_summary["records_unavailable"] == 0:
            inference = (
                "<h3>Privacy Inference Summary</h3><p>" + PRIVACY_INFERENCE_DESCRIPTION + "</p><p>Not evaluated.</p>"
            )
            if "entities_left_as_is" in privacy_summary:
                count = privacy_summary["entities_left_as_is"]
                value = str(count) if count is not None else "Unavailable"
                inference += "<table><tr><th>Entities assigned leave_as_is</th><td>" + value + "</td></tr></table>"
            inference_issues = "<p>Not evaluated.</p>"
    display_issues = judge_issues.copy()
    if strategy.startswith("Replace - "):
        unavailable = display_issues["Result"] == "Unavailable"
        display_issues.loc[unavailable, "Reason"] = "Evaluation unavailable: " + display_issues.loc[
            unavailable, "Reason"
        ].astype(str)
        display_issues = display_issues.drop(columns="Result")
    with pd.option_context("display.max_colwidth", None):
        judges = (
            judge_summary.to_html(index=False, escape=True, border=0)
            if not judge_summary.empty
            else "<p>Not evaluated.</p>"
        )
        issues = (
            display_issues.to_html(index=False, escape=True, border=0, classes="judge-issues")
            if not judge_issues.empty
            else ("<p>No reported judge issues.</p>" if not judge_summary.empty else "<p>Not evaluated.</p>")
        )
    if strategy == "Rewrite":
        issues = _render_rewrite_judge_issues(judge_summary, judge_issues)
    judge_note = "Pass counts refer to records; unavailable results are excluded from evaluated counts."
    if judge_summary["Check"].isin(["Rewrite privacy", "Rewrite quality", "Rewrite style"]).any():
        judge_note += " Rewrite rubric ratings are categorical; medium and low ratings appear below as issues."
    summary_layout = "evaluation-summaries" if include_inference else "evaluation-summaries judge-only"
    rewrite_settings = ""
    if strategy == "Rewrite":
        strict_protection = str(execution_settings.get("strict_entity_protection", "Unavailable"))
        rewrite_settings = (
            "<section><h3>Rewrite Settings</h3><table><tr><th>strict_entity_protection</th><td>"
            + html.escape(strict_protection)
            + "</td></tr>"
            + (
                "<tr><th>Entities assigned leave_as_is</th><td>"
                + (str(len(left_as_is)) if left_as_is is not None else "Unavailable")
                + "</td></tr>"
                if not include_inference
                else ""
            )
            + "</table></section>"
        )
    return (
        '<div class="anonymizer-privacy-report anonymizer-evaluation-report">'
        + styles
        + "<style>.anonymizer-evaluation-report .evaluation-summaries"
        "{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:32px;}"
        ".anonymizer-evaluation-report .evaluation-summaries section{min-width:0;overflow-x:auto;}"
        ".anonymizer-evaluation-report .evaluation-summaries.judge-only{grid-template-columns:1fr;}"
        ".anonymizer-evaluation-report .evaluation-summaries th{white-space:normal!important;}"
        ".anonymizer-evaluation-report .models-used h4{margin:8px 0;}"
        ".anonymizer-evaluation-report .models-used td span{color:#666;}"
        ".anonymizer-evaluation-report .judge-issues td:nth-child(2){min-width:110px;max-width:110px;}"
        ".anonymizer-evaluation-report .judge-issues td:nth-child(6)"
        "{min-width:110px;white-space:nowrap!important;overflow-wrap:normal!important;}"
        ".anonymizer-evaluation-report .judge-issues td:nth-child(3),"
        ".anonymizer-evaluation-report .judge-issues td:nth-child(5)"
        "{white-space:nowrap!important;overflow-wrap:normal!important;}"
        ".anonymizer-evaluation-report .judge-issues td:nth-child(4),"
        ".anonymizer-evaluation-report .judge-issues td:nth-child(7){min-width:160px;}"
        ".anonymizer-evaluation-report .judge-issues td:last-child{min-width:280px;max-width:600px;}"
        ".anonymizer-evaluation-report .rewrite-judge-issues td:nth-child(2)"
        "{min-width:110px;max-width:110px;}"
        ".anonymizer-evaluation-report .rewrite-judge-issues td:last-child{min-width:280px;max-width:650px;}"
        ".anonymizer-evaluation-report .rewrite-entity-issues td:nth-child(3){min-width:180px;}"
        ".anonymizer-evaluation-report .rewrite-entity-issues td:nth-child(4),"
        ".anonymizer-evaluation-report .rewrite-entity-issues td:nth-child(5):not(:last-child),"
        ".anonymizer-evaluation-report .rewrite-rubric-issues td:nth-child(3)"
        "{white-space:nowrap!important;overflow-wrap:normal!important;}"
        "@media(max-width:1000px){.anonymizer-evaluation-report .evaluation-summaries{grid-template-columns:1fr;}}"
        "@media print{.anonymizer-evaluation-report .evaluation-summaries{display:block;}"
        ".anonymizer-evaluation-report .privacy-details{overflow:visible;}"
        ".anonymizer-evaluation-report thead{display:table-header-group;}}"
        "</style><h1>NeMo Anonymizer Evaluation Report</h1>"
        + "<p><strong>Date generated:</strong> "
        + datetime.now().astimezone().date().isoformat()
        + "</p>"
        + "<p><strong>Strategy:</strong> "
        + html.escape(strategy)
        + f" &nbsp; | &nbsp; <strong>Failed records:</strong> {failed_count}</p>"
        + failure_summary
        + _render_run_warnings(warning_metadata)
        + _render_execution_times(execution_timings or {})
        + _render_models_used(model_metadata)
        + _render_labels_used(entity_labels, excluded_entity_labels)
        + _render_data_summary(data_summary)
        + _render_batching_settings(execution_settings, privacy_summary)
        + rewrite_settings
        + f'<div class="{summary_layout}"><section><h3>Privacy &amp; Quality Judge Summary</h3>'
        + judges
        + "<p>"
        + judge_note
        + "</p></section>"
        + ("<section>" + inference + "</section>" if include_inference else "")
        + "</div>"
        '<h2>Privacy &amp; Quality Judge Issues</h2><div class="privacy-details">'
        + issues
        + "</div>"
        + ("<h2>Privacy Inference Issues</h2>" + inference_issues if include_inference else "")
        + (render_leave_as_is_section(left_as_is) if strategy == "Rewrite" else "")
        + "</div>"
    )


def write_report_html(path: str | Path, report: str, *, title: str) -> Path:
    """Write self-contained UTF-8 HTML; propagate filesystem errors to the caller."""
    destination = Path(path)
    document = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>" + html.escape(title) + "</title>"
        "<style>body{font-family:Arial,sans-serif;margin:24px;}"
        "@media print{@page{size:landscape;}"
        ".anonymizer-privacy-report .privacy-details{overflow:visible;}"
        ".anonymizer-privacy-report thead{display:table-header-group;}}"
        "</style></head><body>" + report + "</body></html>"
    )
    destination.write_text(document, encoding="utf-8")
    return destination


def _render_execution_times(metadata: dict[str, dict[str, float | None]]) -> str:
    """Render saved timings, distinguishing absent measurements from zero."""
    sections = []
    for phase, title in (("run", "Anonymization"), ("evaluate", "Evaluation")):
        tasks = metadata.get(phase, {})
        rows = []
        for task, elapsed in tasks.items():
            if task == "Total elapsed" or (task == "Detection workflow" and "Total detection" in tasks):
                continue
            value = "Unavailable" if elapsed is None else f"{elapsed:,.2f} s"
            rows.append(f"<tr><th>{html.escape(task)}</th><td>{value}</td></tr>")
        total = tasks.get("Total elapsed")
        value = "Unavailable" if total is None else f"{total:,.2f} s"
        rows.append(f"<tr><th>Total {title.lower()} time</th><td>{value}</td></tr>")
        sections.append(f"<section><h4>{title}</h4><table>{''.join(rows)}</table></section>")
    return (
        '<section><h3>Execution Time</h3><div class="evaluation-summaries">'
        + "".join(sections)
        + "</div><p>Elapsed wall-clock time. Stages can overlap across records and should not be added together. "
        "Evaluation time covers evaluate(), including its model calls; displaying or exporting this report does not rerun evaluation. "
        "Unavailable means no timing was saved.</p></section>"
    )
