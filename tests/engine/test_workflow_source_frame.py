# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Literal source correspondence independent of legacy record identifiers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from data_designer.engine.storage.artifact_storage import ArtifactStorage

from anonymizer.engine.ndd._source_frame import WorkflowSourceFrame
from anonymizer.interface.errors import AnonymizerWorkflowError


def test_reordered_duplicate_records_keep_distinct_source_positions() -> None:
    source = pd.DataFrame({"text": ["same", "same", "last"], "extra": [17, 23, 31]}, index=pd.Index([5, 5, 2]))
    frame = WorkflowSourceFrame.capture(source)
    seed = frame.seed()
    returned = seed.iloc[[2, 0]].copy()
    returned["generated"] = ["third", "first"]
    projected, positions = frame.project(returned)
    assert positions == (0, 2)
    assert projected.to_dict("list") == {"text": ["same", "last"], "extra": [17, 31], "generated": ["first", "third"]}
    assert source.columns.tolist() == ["text", "extra"]
    assert source.index.tolist() == [5, 5, 2]


@pytest.mark.parametrize("fault", ["missing", "foreign", "duplicate", "nonstring", "duplicate_column"])
def test_uncorrelated_output_is_rejected(fault: str) -> None:
    frame = WorkflowSourceFrame.capture(pd.DataFrame({"text": ["a", "b"]}))
    returned = frame.seed()
    if fault == "missing":
        returned = returned.drop(columns=frame.key_column)
    elif fault == "duplicate_column":
        returned = pd.concat([returned, returned[[frame.key_column]]], axis=1)
    else:
        returned.loc[0, frame.key_column] = {
            "foreign": "foreign",
            "duplicate": returned.iloc[1][frame.key_column],
            "nonstring": 3,
        }[fault]
    with pytest.raises(AnonymizerWorkflowError, match="source correspondence"):
        frame.project(returned)


def test_empty_output_needs_no_transport_column() -> None:
    frame = WorkflowSourceFrame.capture(pd.DataFrame({"text": ["a"]}))
    result, positions = frame.project(pd.DataFrame({"text": []}))
    assert result.empty and positions == ()


def test_preview_cannot_publish_an_unrequested_source() -> None:
    frame = WorkflowSourceFrame.capture(pd.DataFrame({"text": ["a", "b"]}))
    with pytest.raises(AnonymizerWorkflowError, match="source correspondence"):
        frame.project(frame.seed().iloc[[1]], requested=1)


def test_nonzero_rows_without_columns_reject_missing_correspondence() -> None:
    frame = WorkflowSourceFrame.capture(pd.DataFrame({"text": ["a", "b"]}))
    returned = pd.DataFrame(index=pd.Index([0]))
    assert len(returned) == 1 and returned.empty
    with pytest.raises(AnonymizerWorkflowError, match="source correspondence"):
        frame.project(returned)


def test_storage_metadata_omits_private_source_and_restores_writer(tmp_path: Path) -> None:
    frame = WorkflowSourceFrame.capture(pd.DataFrame({"text": ["synthetic"]}))
    storage = ArtifactStorage(artifact_path=tmp_path, dataset_name="source-metadata")
    original = storage.write_metadata
    public = {"column_name": "text", "num_records": 1}
    private = {"column_name": frame.key_column, "num_records": 1}
    metadata: dict[str, Any] = {"column_statistics": [public, private], "actual_num_records": 1}
    with pytest.raises(RuntimeError, match="after write"), frame._bind_storage(storage):
        storage.write_metadata(metadata)
        assert storage.read_metadata() == {"column_statistics": [public], "actual_num_records": 1}
        assert metadata["column_statistics"] == [public, private]
        storage.update_metadata({"num_completed_batches": 1})
        assert storage.read_metadata()["num_completed_batches"] == 1
        assert storage.read_metadata()["column_statistics"] == [public]
        raise RuntimeError("after write")
    assert storage.write_metadata == original
