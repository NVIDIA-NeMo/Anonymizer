# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Notebook presentation of privacy coverage, recovery metrics, and problems."""

from __future__ import annotations

import html
from typing import Any

import pandas as pd

from anonymizer.engine.constants import (
    COL_FINDING_CATEGORY,
    COL_FINDING_CONFIDENCE,
    COL_FINDING_CONTRIBUTION,
    COL_FINDING_ENTITY_LABEL,
    COL_FINDING_ENTITY_VALUE,
    COL_FINDING_EVIDENCE,
    COL_FINDING_INFERRED,
    COL_FINDING_REASON,
    COL_FINDING_RECORD_ID,
    COL_FINDING_RECORD_INDEX,
    COL_FINDING_SENSITIVITY,
    COL_FINDING_SOURCE,
)

PRIVACY_INFERENCE_DESCRIPTION = (
    "Privacy inference checks whether each original entity value can still be identified or inferred "
    "from the anonymized text, including direct mentions and contextual clues."
)


def render_privacy_report_sections(
    summary: dict[str, Any], problems: pd.DataFrame, *, heading: str = "Privacy evaluation summary"
) -> tuple[str, str, str]:
    """Render a summary above untruncated, escaped problem details."""
    rows = [
        ("Records evaluated", f"{summary['records_evaluated']} / {summary['records_total']}"),
        ("Unavailable records", str(summary["records_unavailable"])),
        ("Records not evaluated", str(summary["records_not_evaluated"])),
        (
            "Records with reported leakage",
            _count_and_rate(
                summary["records_with_leakage"], summary["records_evaluated"], summary["record_leakage_rate"]
            ),
        ),
        (
            "Entities reported recoverable",
            _count_and_rate(summary["entities_leaked"], summary["entities_evaluated"], summary["entity_recovery_rate"]),
        ),
        (
            "Weighted entity recovery rate",
            f"{summary['weighted_leakage_rate']:.1%}"
            if summary["weighted_leakage_rate"] is not None
            else "Unavailable",
        ),
        ("High-sensitivity entities leaked", str(summary["high_sensitivity_entities_leaked"])),
        ("Leakage mass", _number(summary["leakage_mass"])),
    ]
    if "risk_tolerance" in summary:
        rows[0:0] = [
            ("Risk tolerance", summary["risk_tolerance"]),
            ("Repair threshold (leakage mass)", _number(summary["repair_threshold"])),
            ("Max repair iterations", str(summary["max_repair_iterations"])),
            (
                "Repair iterations taken (maximum across records)",
                str(summary["repair_iterations_taken"])
                if summary["repair_iterations_taken"] is not None
                else "Unavailable",
            ),
        ]
    for level, mass in summary["leakage_mass_by_sensitivity"].items():
        rows.append((f"Leakage mass — {level} sensitivity", _number(mass)))
    metrics = "".join(f"<tr><th>{html.escape(label)}</th><td>{html.escape(value)}</td></tr>" for label, value in rows)
    problems = problems.copy()
    problems[COL_FINDING_SENSITIVITY] = problems[COL_FINDING_SENSITIVITY].replace({"medium": "med"})
    problems[COL_FINDING_CATEGORY] = problems[COL_FINDING_CATEGORY].replace(
        {"direct_identifier": "direct", "quasi_identifier": "quasi", "latent_identifier": "latent"}
    )
    with pd.option_context("display.max_colwidth", None):
        details = (
            problems.rename(
                columns={
                    COL_FINDING_RECORD_INDEX: "Row",
                    COL_FINDING_RECORD_ID: "Record ID",
                    COL_FINDING_ENTITY_VALUE: "Entity",
                    COL_FINDING_ENTITY_LABEL: "Label",
                    COL_FINDING_SOURCE: "Source",
                    COL_FINDING_CATEGORY: "Type",
                    COL_FINDING_SENSITIVITY: "Sens.",
                    COL_FINDING_INFERRED: "Inferred?",
                    COL_FINDING_CONFIDENCE: "Conf.",
                    COL_FINDING_REASON: "Reason",
                    COL_FINDING_EVIDENCE: "Evidence",
                    COL_FINDING_CONTRIBUTION: "Leak mass",
                }
            ).to_html(index=False, escape=True, border=0, classes="privacy-problems")
            if not problems.empty
            else "<p>No reported leaks or unavailable evaluations.</p>"
        )
    styles = (
        "<style>.anonymizer-privacy-report th,.anonymizer-privacy-report td"
        "{text-align:left;vertical-align:top;white-space:normal;padding:6px;overflow-wrap:anywhere;}"
        ".anonymizer-privacy-report table{border-collapse:collapse;}"
        ".anonymizer-privacy-report th"
        "{white-space:nowrap!important;overflow-wrap:normal!important;word-break:normal!important;}"
        ".anonymizer-privacy-report .privacy-details{overflow-x:auto;}"
        ".anonymizer-privacy-report table.privacy-problems"
        "{table-layout:auto!important;width:auto!important;max-width:none!important;}"
        ".anonymizer-privacy-report .privacy-problems th,"
        ".anonymizer-privacy-report .privacy-problems td"
        "{text-align:left!important;width:1px!important;min-width:0!important;padding:6px!important;}"
        ".anonymizer-privacy-report .privacy-problems td"
        "{white-space:nowrap!important;overflow-wrap:normal!important;word-break:normal!important;}"
        ".anonymizer-privacy-report .privacy-problems td:nth-child(2)"
        "{width:110px!important;min-width:110px!important;max-width:110px!important;"
        "white-space:normal!important;overflow-wrap:anywhere!important;word-break:break-all!important;}"
        ".anonymizer-privacy-report .privacy-problems td:nth-child(3),"
        ".anonymizer-privacy-report .privacy-problems td:nth-child(4)"
        "{min-width:160px!important;white-space:nowrap;overflow-wrap:normal;}"
        ".anonymizer-privacy-report .privacy-problems td:nth-child(10),"
        ".anonymizer-privacy-report .privacy-problems td:nth-child(11)"
        "{width:200px!important;min-width:200px!important;max-width:200px!important;"
        "white-space:normal!important;overflow-wrap:anywhere!important;}"
        "</style>"
    )
    inference_summary = (
        "<h3>"
        + html.escape(heading)
        + "</h3><p>"
        + PRIVACY_INFERENCE_DESCRIPTION
        + "</p><table>"
        + metrics
        + "</table>"
        "<p><strong>Weighted entity recovery rate</strong> = leakage mass divided by the total "
        "sensitivity weight of evaluated entity targets. Each reported recoverable entity contributes "
        "its sensitivity weight × the judge's confidence; negative answers contribute zero. "
        "Weights: high = 1.0, medium = 0.6, low = 0.3. "
        "This is a judge-based recovery score, not a probability of identifying a person.</p>"
        "<p>Rates use complete evaluated records and evaluated entity targets; unavailable records "
        "and exempted entities are excluded. These are the privacy judge's reported findings.</p>"
    )
    inference_issues = (
        "<p>Source: detector (GLiNER in the standard pipeline), augment, or latent. "
        "Multiple sources may be listed for repeated occurrences. Unknown means provenance was not saved; "
        "tagged means an explicit entity whose detector/augmentation source is unavailable.</p>"
        '<div class="privacy-details">' + details + "</div>"
    )
    return styles, inference_summary, inference_issues


def render_privacy_report_html(summary: dict[str, Any], problems: pd.DataFrame) -> str:
    """Render the existing privacy-only report for compatibility."""
    styles, inference_summary, inference_issues = render_privacy_report_sections(summary, problems)
    return (
        '<div class="anonymizer-privacy-report">'
        + styles
        + inference_summary
        + "<h3>Reported problems</h3>"
        + inference_issues
        + "</div>"
    )


def _count_and_rate(count: int, total: int, rate: float | None) -> str:
    return f"{count} / {total} ({rate:.1%})" if rate is not None else f"{count} / {total} (not evaluated)"


def _number(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "Unavailable"
