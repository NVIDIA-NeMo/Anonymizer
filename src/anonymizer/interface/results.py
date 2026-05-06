# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.interface.display import render_record_html


class _DisplayMixin:
    """Shared ``display_record`` behavior for result types."""

    trace_dataframe: pd.DataFrame
    original_text_column: str
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
        html_str = render_record_html(row, record_index=i, original_text_column=self.original_text_column)

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
        original_text_column: Name of the user-facing text column.
        failed_records: Records that failed during pipeline processing.
    """

    dataframe: pd.DataFrame
    trace_dataframe: pd.DataFrame
    original_text_column: str
    failed_records: list[FailedRecord]
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
        original_text_column: Name of the user-facing text column.
        failed_records: Records that failed during pipeline processing.
        preview_num_records: Number of records requested for the preview.
    """

    dataframe: pd.DataFrame
    trace_dataframe: pd.DataFrame
    original_text_column: str
    failed_records: list[FailedRecord]
    preview_num_records: int
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
