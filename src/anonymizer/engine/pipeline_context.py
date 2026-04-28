# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed container for pipeline-level metadata that travels with a DataFrame.

Replaces the use of ``DataFrame.attrs`` (an experimental pandas slot that is
silently dropped by ``merge``/``concat``/``groupby``) for threading metadata
through workflow stages. Add new pipeline-wide fields here rather than
reintroducing ``df.attrs``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd


@dataclass(frozen=True)
class PipelineContext:
    """A DataFrame plus the pipeline-level metadata that describes it."""

    dataframe: pd.DataFrame
    original_text_column: str

    def with_dataframe(self, dataframe: pd.DataFrame) -> PipelineContext:
        """Return a new context wrapping *dataframe* with unchanged metadata."""
        return replace(self, dataframe=dataframe)
