# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed result of input normalization.

Captures the user's input DataFrame after the reader has applied any
collision-driven renames, plus the user-requested and the post-resolution
text-column names. Carried by the orchestrator so downstream rendering can
restore user-facing column names without depending on ``DataFrame.attrs``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd


@dataclass(frozen=True)
class ResolvedInput:
    """The normalized input DataFrame plus the metadata that describes it.

    Attributes:
        dataframe: The input DataFrame after collision renames and the
            internal-text-column rename. Workflows operate on this frame.
        requested_text_column: The text column name the user originally
            requested via ``AnonymizerInput.text_column``.
        resolved_text_column: The text column name actually used downstream
            after collision resolution. Equals ``requested_text_column``
            when there was no collision; otherwise equals the renamed
            identifier (e.g. ``"final_entities__input"`` when the user
            requested ``"final_entities"``). Use this when restoring
            user-facing output column names.

    Note:
        ``==`` and ``hash()`` are unsupported: the ``dataframe`` field is
        not hashable and not bool-coercible. Compare fields explicitly if
        needed.
    """

    dataframe: pd.DataFrame
    requested_text_column: str
    resolved_text_column: str

    def with_dataframe(self, dataframe: pd.DataFrame) -> ResolvedInput:
        """Return a new instance wrapping *dataframe* with unchanged metadata."""
        return replace(self, dataframe=dataframe)
