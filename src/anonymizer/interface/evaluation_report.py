# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Presentation and standalone export for the combined evaluation report."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import pandas as pd

from anonymizer.interface.privacy_report import render_privacy_report_sections


def render_evaluation_report_html(
    *,
    strategy: str,
    failed_records: int,
    privacy_summary: dict[str, Any],
    privacy_issues: pd.DataFrame,
    judge_summary: pd.DataFrame,
    judge_issues: pd.DataFrame,
) -> str:
    """Render summaries side by side and problem details below, with escaped data."""
    styles, inference, inference_issues = render_privacy_report_sections(
        privacy_summary, privacy_issues, heading="Privacy Inference Summary"
    )
    if privacy_summary["records_evaluated"] == 0 and privacy_summary["records_unavailable"] == 0:
        inference = "<h3>Privacy Inference Summary</h3><p>Not evaluated.</p>"
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
        ".anonymizer-evaluation-report .judge-issues td:nth-child(2){min-width:110px;max-width:110px;}"
        ".anonymizer-evaluation-report .judge-issues td:nth-child(3),"
        ".anonymizer-evaluation-report .judge-issues td:nth-child(5),"
        ".anonymizer-evaluation-report .judge-issues td:nth-child(7)"
        "{white-space:nowrap!important;overflow-wrap:normal!important;}"
        ".anonymizer-evaluation-report .judge-issues td:nth-child(4),"
        ".anonymizer-evaluation-report .judge-issues td:nth-child(6){min-width:160px;}"
        ".anonymizer-evaluation-report .judge-issues td:last-child{min-width:280px;max-width:600px;}"
        "@media(max-width:1000px){.anonymizer-evaluation-report .evaluation-summaries{grid-template-columns:1fr;}}"
        "@media print{.anonymizer-evaluation-report .evaluation-summaries{display:block;}"
        ".anonymizer-evaluation-report .privacy-details{overflow:visible;}"
        ".anonymizer-evaluation-report thead{display:table-header-group;}}"
        "</style><h1>NeMo Anonymizer Evaluation Report</h1>"
        "<p><strong>Strategy:</strong> "
        + html.escape(strategy)
        + f" &nbsp; | &nbsp; <strong>Failed records:</strong> {failed_records}</p>"
        '<div class="evaluation-summaries"><section><h3>Privacy &amp; Quality Judge Summary</h3>'
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
