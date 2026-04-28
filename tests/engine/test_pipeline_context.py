# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from anonymizer.engine.pipeline_context import PipelineContext


def test_holds_dataframe_and_metadata() -> None:
    df = pd.DataFrame({"x": [1, 2]})
    ctx = PipelineContext(dataframe=df, original_text_column="bio")
    assert ctx.dataframe is df
    assert ctx.original_text_column == "bio"


def test_is_frozen() -> None:
    ctx = PipelineContext(dataframe=pd.DataFrame(), original_text_column="bio")
    with pytest.raises(FrozenInstanceError):
        ctx.original_text_column = "other"  # type: ignore[misc]


def test_with_dataframe_swaps_frame_keeps_metadata() -> None:
    ctx = PipelineContext(dataframe=pd.DataFrame({"a": [1]}), original_text_column="content")
    new_df = pd.DataFrame({"a": [2, 3]})
    new_ctx = ctx.with_dataframe(new_df)

    assert new_ctx is not ctx
    assert new_ctx.dataframe is new_df
    assert new_ctx.original_text_column == "content"
    assert len(ctx.dataframe) == 1


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
    ctx = PipelineContext(dataframe=df, original_text_column="bio")

    transformed = operation(ctx.dataframe)

    assert "original_text_column" not in transformed.attrs
    assert ctx.with_dataframe(transformed).original_text_column == "bio"
