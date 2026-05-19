# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from anonymizer.engine.resolved_input import ResolvedInput


def test_holds_dataframe_and_metadata() -> None:
    df = pd.DataFrame({"x": [1, 2]})
    resolved = ResolvedInput(dataframe=df, requested_text_column="bio", resolved_text_column="bio")
    assert resolved.dataframe is df
    assert resolved.requested_text_column == "bio"
    assert resolved.resolved_text_column == "bio"


def test_distinguishes_requested_and_resolved_columns() -> None:
    """Collision-rename case: the user asked for ``final_entities`` but the
    reader had to rename it to avoid clashing with a fixed output column.
    """
    df = pd.DataFrame({"final_entities__input": ["bio text"]})
    resolved = ResolvedInput(
        dataframe=df,
        requested_text_column="final_entities",
        resolved_text_column="final_entities__input",
    )
    assert resolved.requested_text_column == "final_entities"
    assert resolved.resolved_text_column == "final_entities__input"


def test_is_frozen() -> None:
    resolved = ResolvedInput(
        dataframe=pd.DataFrame(),
        requested_text_column="bio",
        resolved_text_column="bio",
    )
    with pytest.raises(FrozenInstanceError):
        resolved.resolved_text_column = "other"  # type: ignore[misc]


def test_with_dataframe_swaps_frame_keeps_metadata() -> None:
    resolved = ResolvedInput(
        dataframe=pd.DataFrame({"a": [1]}),
        requested_text_column="content",
        resolved_text_column="content",
    )
    new_df = pd.DataFrame({"a": [2, 3]})
    new_resolved = resolved.with_dataframe(new_df)

    assert new_resolved is not resolved
    assert new_resolved.dataframe is new_df
    assert new_resolved.requested_text_column == "content"
    assert new_resolved.resolved_text_column == "content"
    assert len(resolved.dataframe) == 1


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(
            lambda df: df.merge(pd.DataFrame({"id": [1], "y": [10]}), on="id"),
            id="merge",
        ),
        pytest.param(
            lambda df: pd.concat([df, df], ignore_index=True),
            id="concat",
        ),
        pytest.param(
            lambda df: df.groupby("id", as_index=False)["x"].sum(),
            id="groupby",
        ),
    ],
)
def test_metadata_survives_pandas_operations_that_drop_attrs(operation) -> None:
    """The motivating invariant: metadata is independent of pandas operations
    that silently drop ``DataFrame.attrs``.
    """
    df = pd.DataFrame({"id": [1, 2], "x": [10, 20]})
    resolved = ResolvedInput(
        dataframe=df,
        requested_text_column="bio",
        resolved_text_column="bio",
    )

    transformed = operation(resolved.dataframe)
    new_resolved = resolved.with_dataframe(transformed)

    assert new_resolved.requested_text_column == "bio"
    assert new_resolved.resolved_text_column == "bio"
