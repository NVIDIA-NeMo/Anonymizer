# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.interface.display import render_record_html


@dataclass(frozen=True)
class AnonymizerResult:
    """Result returned by full anonymization runs."""

    dataframe: pd.DataFrame
    trace_dataframe: pd.DataFrame
    failed_records: list[FailedRecord]


@dataclass
class PreviewResult:
    """Result returned by preview runs."""

    dataframe: pd.DataFrame
    trace_dataframe: pd.DataFrame
    failed_records: list[FailedRecord]
    preview_num_records: int
    _display_cycle_index: int = field(default=0, init=False, repr=False)

    def display_record(self, index: int | None = None) -> None:
        """Render a record with entity highlights and replacement map in a notebook.

        Args:
            index: Row index to display. If None, cycles through records on repeated calls.
        """
        i = index if index is not None else self._display_cycle_index
        if i < 0 or i >= len(self.trace_dataframe):
            raise IndexError(f"Record index {i} is out of bounds for {len(self.trace_dataframe)} records.")

        row = self.trace_dataframe.iloc[i]
        html_str = render_record_html(row, record_index=i)

        try:
            from IPython.display import HTML, display

            display(HTML(html_str))
        except ImportError:
            print(html_str)

        if index is None:
            self._display_cycle_index = (self._display_cycle_index + 1) % len(self.trace_dataframe)
