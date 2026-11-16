# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Presentation and standalone export for the combined evaluation report."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import pandas as pd

from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.interface.privacy_report import PRIVACY_INFERENCE_DESCRIPTION, render_privacy_report_sections


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
        + "</div><p>Saved task assignments; conditional tasks may not run for every record. "
        "Local replacement operations do not use a model. Names and providers reflect the saved run "
        "or evaluation configuration, not per-request usage counts.</p></section>"
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


def _render_batching_settings(settings: dict[str, int | None], summary: dict[str, Any]) -> str:
    """Show saved settings; distinguish unbatched QA from missing metadata."""
    validation = str(settings.get("validation_max_entities_per_call", "Unavailable"))
    if "privacy_qa_batch_size" in settings:
        batch = settings["privacy_qa_batch_size"]
        privacy = str(batch) if batch is not None else "None (all questions in one call per record)"
    else:
        privacy = "Unavailable" if summary["records_evaluated"] or summary["records_unavailable"] else "Not evaluated"
    return (
        "<section><h3>Batching Settings</h3><table>"
        "<tr><th>privacy_qa_batch_size</th><td>" + html.escape(privacy) + "</td></tr>"
        "<tr><th>validation_max_entities_per_call</th><td>" + html.escape(validation) + "</td></tr>"
        "</table></section>"
    )


def render_evaluation_report_html(
    *,
    strategy: str,
    failed_records: list[FailedRecord],
    model_metadata: list[dict[str, str]],
    execution_settings: dict[str, int | None],
    entity_labels: list[str] | None,
    excluded_entity_labels: list[str] | None,
    privacy_summary: dict[str, Any],
    privacy_issues: pd.DataFrame,
    judge_summary: pd.DataFrame,
    judge_issues: pd.DataFrame,
) -> str:
    """Render summaries side by side and problem details below, with escaped data."""
    failed_count, failure_summary = _render_failed_records(failed_records)
    styles, inference, inference_issues = render_privacy_report_sections(
        privacy_summary, privacy_issues, heading="Privacy Inference Summary"
    )
    if privacy_summary["records_evaluated"] == 0 and privacy_summary["records_unavailable"] == 0:
        inference = (
            "<h3>Privacy Inference Summary</h3><p>" + PRIVACY_INFERENCE_DESCRIPTION + "</p><p>Not evaluated.</p>"
        )
        inference_issues = "<p>Not evaluated.</p>"
    with pd.option_context("display.max_colwidth", None):
        judges = (
            judge_summary.to_html(index=False, escape=True, border=0)
            if not judge_summary.empty
            else "<p>Not evaluated.</p>"
        )
        issues = (
            judge_issues.to_html(index=False, escape=True, border=0, classes="judge-issues")
            if not judge_issues.empty
            else ("<p>No reported judge issues.</p>" if not judge_summary.empty else "<p>Not evaluated.</p>")
        )
    judge_note = "Pass counts refer to records; unavailable results are excluded from evaluated counts."
    if judge_summary["Check"].isin(["Rewrite privacy", "Rewrite quality", "Rewrite style"]).any():
        judge_note += " Rewrite rubric ratings are categorical; medium and low ratings appear below as issues."
    return (
        '<div class="anonymizer-privacy-report anonymizer-evaluation-report">'
        + styles
        + "<style>.anonymizer-evaluation-report .evaluation-summaries"
        "{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:32px;}"
        ".anonymizer-evaluation-report .evaluation-summaries section{min-width:0;overflow-x:auto;}"
        ".anonymizer-evaluation-report .evaluation-summaries th{white-space:normal!important;}"
        ".anonymizer-evaluation-report .models-used h4{margin:8px 0;}"
        ".anonymizer-evaluation-report .models-used td span{color:#666;}"
        ".anonymizer-evaluation-report .judge-issues td:nth-child(2){min-width:110px;max-width:110px;}"
        ".anonymizer-evaluation-report .judge-issues td:nth-child(3),"
        ".anonymizer-evaluation-report .judge-issues td:nth-child(5),"
        ".anonymizer-evaluation-report .judge-issues td:nth-child(8)"
        "{white-space:nowrap!important;overflow-wrap:normal!important;}"
        ".anonymizer-evaluation-report .judge-issues td:nth-child(4),"
        ".anonymizer-evaluation-report .judge-issues td:nth-child(7){min-width:160px;}"
        ".anonymizer-evaluation-report .judge-issues td:last-child{min-width:280px;max-width:600px;}"
        "@media(max-width:1000px){.anonymizer-evaluation-report .evaluation-summaries{grid-template-columns:1fr;}}"
        "@media print{.anonymizer-evaluation-report .evaluation-summaries{display:block;}"
        ".anonymizer-evaluation-report .privacy-details{overflow:visible;}"
        ".anonymizer-evaluation-report thead{display:table-header-group;}}"
        "</style><h1>NeMo Anonymizer Evaluation Report</h1>"
        "<p><strong>Strategy:</strong> "
        + html.escape(strategy)
        + f" &nbsp; | &nbsp; <strong>Failed records:</strong> {failed_count}</p>"
        + failure_summary
        + _render_models_used(model_metadata)
        + _render_labels_used(entity_labels, excluded_entity_labels)
        + _render_batching_settings(execution_settings, privacy_summary)
        + '<div class="evaluation-summaries"><section><h3>Privacy &amp; Quality Judge Summary</h3>'
        + judges
        + "<p>"
        + judge_note
        + "</p></section>"
        "<section>" + inference + "</section></div>"
        '<h2>Privacy &amp; Quality Judge Issues</h2><div class="privacy-details">' + issues + "</div>"
        "<h2>Privacy Inference Issues</h2>" + inference_issues + "</div>"
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
