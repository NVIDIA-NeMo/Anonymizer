# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from anonymizer.config.replace_strategies import ReplaceMethod
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.interface.display import render_record_html


class _DisplayMixin:
    """Shared ``display_record`` behavior for result types."""

    trace_dataframe: pd.DataFrame
    resolved_text_column: str
    _display_cycle_index: int

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

        try:
            from IPython.display import HTML, display

            display(HTML(html_str))
        except ImportError:
            print(html_str)

        if index is None:
            self._display_cycle_index = (self._display_cycle_index + 1) % len(self.trace_dataframe)


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
        replace_method: The replace strategy that produced this result. Set by
            ``run()`` / ``preview()``; consumed by ``evaluate()`` to dispatch the
            right judges. ``None`` on results that were constructed by hand or
            loaded from a pre-strategy-tracking format.
        rewrite_config: The privacy goal that produced this result when rewrite
            mode was used. Set by ``run()`` / ``preview()``; consumed by
            ``evaluate()`` to dispatch the rewrite judges. Mutually exclusive
            with ``replace_method``.
        strict_entity_protection: Whether the rewrite ran with strict entity
            protection. Set by ``run()`` / ``preview()``; consumed by
            ``evaluate()`` so the entity-coverage judge scores in strict mode
            (no benefit-of-the-doubt for missed quasi-identifiers).
    """

    dataframe: pd.DataFrame
    trace_dataframe: pd.DataFrame
    resolved_text_column: str
    failed_records: list[FailedRecord]
    replace_method: ReplaceMethod | None = None
    rewrite_config: PrivacyGoal | None = None
    entity_labels: list[str] | None = None
    strict_entity_protection: bool = False
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
        replace_method: The replace strategy that produced this preview. Set by
            ``preview()``; consumed by ``evaluate()`` to dispatch the right
            judges. ``None`` on results that were constructed by hand or loaded
            from a pre-strategy-tracking format.
        rewrite_config: The privacy goal that produced this preview when rewrite
            mode was used. Set by ``preview()``; consumed by ``evaluate()`` to
            dispatch the rewrite judges. Mutually exclusive with ``replace_method``.
        strict_entity_protection: Whether the rewrite ran with strict entity
            protection. Set by ``preview()``; consumed by ``evaluate()`` so the
            entity-coverage judge scores in strict mode (no benefit-of-the-doubt
            for missed quasi-identifiers).
    """

    dataframe: pd.DataFrame
    trace_dataframe: pd.DataFrame
    resolved_text_column: str
    failed_records: list[FailedRecord]
    preview_num_records: int
    replace_method: ReplaceMethod | None = None
    rewrite_config: PrivacyGoal | None = None
    entity_labels: list[str] | None = None
    strict_entity_protection: bool = False
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
