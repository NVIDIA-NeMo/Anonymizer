# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from anonymizer.config.replace_strategies import ReplaceMethod
from anonymizer.config.rewrite import EvaluationCriteria, PrivacyGoal
from anonymizer.engine.constants import (
    COL_FINDING_ANSWER,
    COL_FINDING_ERROR,
    COL_FINDING_INFERRED,
    COL_FINDING_QUESTION,
    COL_FINDING_SENSITIVITY_REASON,
    COL_FINDING_STATUS,
)
from anonymizer.engine.evaluation.privacy_findings import build_privacy_findings, summarize_privacy_findings
from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.interface.display import render_record_html
from anonymizer.interface.privacy_report import render_privacy_report_html


class _DisplayMixin:
    """Shared record display and privacy findings for result types."""

    trace_dataframe: pd.DataFrame
    resolved_text_column: str
    _display_cycle_index: int
    rewrite_evaluation: EvaluationCriteria | None

    @property
    def privacy_findings(self) -> pd.DataFrame:
        """Reported leaks and unavailable assessments as a fresh dataframe.

        Includes record index/ID, original value/label, sensitivity, confidence,
        reason, evidence, contribution, and "Can entity still be inferred?".
        The inference column contains yes for leaks and Unavailable for failures.
        Negative answers and unevaluated exemptions are omitted. No model calls run.
        Missing answers have unavailable contributions, never implied zero leakage.
        Full diagnostic details remain in trace_dataframe.
        """
        findings = build_privacy_findings(self.trace_dataframe)
        findings = findings[
            ((findings[COL_FINDING_STATUS] == "evaluated") & (findings[COL_FINDING_ANSWER] == "yes"))
            | (findings[COL_FINDING_STATUS] == "unavailable")
        ].copy()
        findings[COL_FINDING_ANSWER] = findings[COL_FINDING_ANSWER].where(
            findings[COL_FINDING_STATUS] == "evaluated",
            findings[COL_FINDING_STATUS].map({"unavailable": "Unavailable", "not_evaluated": "Not evaluated"}),
        )
        return findings.drop(
            columns=[
                COL_FINDING_SENSITIVITY_REASON,
                COL_FINDING_QUESTION,
                COL_FINDING_STATUS,
                COL_FINDING_ERROR,
            ]
        ).rename(columns={COL_FINDING_ANSWER: COL_FINDING_INFERRED})

    @property
    def privacy_summary(self) -> dict[str, Any]:
        """Coverage, reported recovery, and weighted metrics from saved assessments.

        The overall weighted rate is summed leakage mass divided by summed
        possible weight, excluding entire unavailable records from both sums.
        """
        summary = summarize_privacy_findings(self.trace_dataframe)
        if self.rewrite_evaluation is not None:
            summary["risk_tolerance"] = self.rewrite_evaluation.risk_tolerance.value
            summary["repair_threshold"] = self.rewrite_evaluation.repair_threshold
        return summary

    def display_privacy_report(self) -> None:
        """Display a summary above reported leaks and unavailable assessments."""
        _display_html(render_privacy_report_html(self.privacy_summary, self.privacy_findings))

    def export_privacy_report(self, path: str | Path) -> Path:
        """Write a standalone UTF-8 HTML report and return its path.

        Includes the same summary and problems as display_privacy_report(),
        without model calls or an IPython dependency. Existing files are
        overwritten; the parent directory must exist. Open in a browser to
        view or print to PDF. Original entity values are included.
        """
        destination = Path(path)
        report = render_privacy_report_html(self.privacy_summary, self.privacy_findings)
        document = (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            "<title>Privacy evaluation report</title>"
            "<style>body{font-family:Arial,sans-serif;margin:24px;}"
            "@media print{@page{size:landscape;}"
            ".anonymizer-privacy-report .privacy-details{overflow:visible;}"
            ".anonymizer-privacy-report thead{display:table-header-group;}}"
            "</style></head><body>" + report + "</body></html>"
        )
        destination.write_text(document, encoding="utf-8")
        return destination

    def display_record(self, index: int | None = None) -> None:
        """Render a record with entity highlights and replacement map in a notebook.

        Args:
            index: Row index to display. If None, cycles through records on repeated calls.
        """
        i = index if index is not None else self._display_cycle_index
        if i < 0 or i >= len(self.trace_dataframe):
            raise IndexError(f"Record index {i} is out of bounds for {len(self.trace_dataframe)} records.")

        row = self.trace_dataframe.iloc[i]
        html_str = render_record_html(row, record_index=i, resolved_text_column=self.resolved_text_column)

        _display_html(html_str)

        if index is None:
            self._display_cycle_index = (self._display_cycle_index + 1) % len(self.trace_dataframe)


def _display_html(html_str: str) -> None:
    try:
        from IPython.display import HTML, display

        display(HTML(html_str))
    except ImportError:
        print(html_str)


@dataclass
class AnonymizerResult(_DisplayMixin):
    """Result returned by full anonymization runs.

    Attributes:
        dataframe: User-facing columns only (text, replaced/rewritten text, scores).
        trace_dataframe: Full pipeline trace including all internal columns.
        resolved_text_column: Name of the user-facing text column. Equals the
            user's requested ``text_column`` unless the reader had to rename it
            to avoid colliding with an Anonymizer output column, in which case
            it is the post-rename identifier (e.g. ``"final_entities__input"``).
        failed_records: Records that failed during pipeline processing.
        privacy_findings: Fresh dataframe of saved entity-level privacy assessments
            and evaluation statuses, built without model calls.
        replace_method: The replace strategy that produced this result. Set by
            ``run()`` / ``preview()``; consumed by ``evaluate()`` to dispatch the
            right judges. ``None`` on results that were constructed by hand or
            loaded from a pre-strategy-tracking format.
        rewrite_config: The privacy goal that produced this result when rewrite
            mode was used. Set by ``run()`` / ``preview()``; consumed by
            ``evaluate()`` to dispatch the rewrite judges. Mutually exclusive
            with ``replace_method``.
        rewrite_evaluation: Saved Rewrite evaluation settings used for the privacy
            report's risk tolerance and repair threshold. None when unavailable.
        entity_labels: Allowlist of entity labels that were in scope during
            detection. Preserved for ``evaluate()`` so the coverage judge scopes
            its evaluation to the same label set. ``None`` means detection was
            permissive: default labels were configured, and the augmenter may
            have emitted additional labels. Configured examples are intentionally
            not persisted for evaluation.
        data_summary: Optional dataset context supplied with the original input.
            Preserved for ``evaluate()`` so entity-coverage judging uses the
            same context as detection.
        excluded_entity_labels: Labels that were explicitly excluded from
            detection. Preserved for ``evaluate()`` so the coverage judge does
            not penalise the output for not anonymizing excluded labels.
        id_column: Record identifier column retained in ``dataframe`` for
            in-memory record input. File-backed input currently leaves this as
            ``None``.
    """

    dataframe: pd.DataFrame
    trace_dataframe: pd.DataFrame
    resolved_text_column: str
    failed_records: list[FailedRecord]
    replace_method: ReplaceMethod | None = None
    rewrite_config: PrivacyGoal | None = None
    entity_labels: list[str] | None = None
    data_summary: str | None = None
    excluded_entity_labels: list[str] | None = None
    id_column: str | None = None
    rewrite_evaluation: EvaluationCriteria | None = None
    _display_cycle_index: int = field(default=0, init=False, repr=False)

    def __repr__(self) -> str:
        return (
            "AnonymizerResult("
            f"rows={len(self.dataframe)}, "
            f"columns={len(self.dataframe.columns)}, "
            f"trace_columns={len(self.trace_dataframe.columns)}, "
            f"failed_records={len(self.failed_records)}"
            ")"
        )


@dataclass
class PreviewResult(_DisplayMixin):
    """Result returned by preview runs.

    Attributes:
        dataframe: User-facing columns only (text, replaced/rewritten text, scores).
        trace_dataframe: Full pipeline trace including all internal columns.
        resolved_text_column: Name of the user-facing text column. Equals the
            user's requested ``text_column`` unless the reader had to rename it
            to avoid colliding with an Anonymizer output column, in which case
            it is the post-rename identifier (e.g. ``"final_entities__input"``).
        failed_records: Records that failed during pipeline processing.
        preview_num_records: Number of records requested for the preview.
        privacy_findings: Fresh dataframe of saved entity-level privacy assessments
            and evaluation statuses, built without model calls.
        replace_method: The replace strategy that produced this preview. Set by
            ``preview()``; consumed by ``evaluate()`` to dispatch the right
            judges. ``None`` on results that were constructed by hand or loaded
            from a pre-strategy-tracking format.
        rewrite_config: The privacy goal that produced this preview when rewrite
            mode was used. Set by ``preview()``; consumed by ``evaluate()`` to
            dispatch the rewrite judges. Mutually exclusive with ``replace_method``.
        rewrite_evaluation: Saved Rewrite evaluation settings used for the privacy
            report's risk tolerance and repair threshold. None when unavailable.
        entity_labels: Allowlist of entity labels that were in scope during
            detection. Preserved for ``evaluate()`` so the coverage judge scopes
            its evaluation to the same label set. ``None`` means detection was
            permissive: default labels were configured, and the augmenter may
            have emitted additional labels. Configured examples are intentionally
            not persisted for evaluation.
        data_summary: Optional dataset context supplied with the original input.
            Preserved for ``evaluate()`` so entity-coverage judging uses the
            same context as detection.
        excluded_entity_labels: Labels that were explicitly excluded from
            detection. Preserved for ``evaluate()`` so the coverage judge does
            not penalise the output for not anonymizing excluded labels.
        id_column: Record identifier column retained in ``dataframe`` for
            in-memory record input. File-backed input currently leaves this as
            ``None``.
    """

    dataframe: pd.DataFrame
    trace_dataframe: pd.DataFrame
    resolved_text_column: str
    failed_records: list[FailedRecord]
    preview_num_records: int
    replace_method: ReplaceMethod | None = None
    rewrite_config: PrivacyGoal | None = None
    entity_labels: list[str] | None = None
    data_summary: str | None = None
    excluded_entity_labels: list[str] | None = None
    id_column: str | None = None
    rewrite_evaluation: EvaluationCriteria | None = None
    _display_cycle_index: int = field(default=0, init=False, repr=False)

    def __repr__(self) -> str:
        return (
            "PreviewResult("
            f"rows={len(self.dataframe)}, "
            f"columns={len(self.dataframe.columns)}, "
            f"trace_columns={len(self.trace_dataframe.columns)}, "
            f"failed_records={len(self.failed_records)}, "
            f"preview_num_records={self.preview_num_records}"
            ")"
        )
