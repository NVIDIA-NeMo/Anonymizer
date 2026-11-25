# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User and trace frames of every recorded case survive a Parquet write and read-back, cell by cell."""

from __future__ import annotations

import io
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import pytest

from anonymizer import Anonymizer
from anonymizer.config.anonymizer_config import TextRecord, TextRecordsInput
from tests.parity.harness import CASES, PREVIEW_ROWS, RECORDS, Case, _cell, scripted

# Two kinds of cell do not come back as written today. A missing cell changes its spelling (NaN in an object column
# reads back as None, None in a numeric column as NaN), and a mapping keyed by label gains the other rows' labels with
# None values, because Parquet stores it as one struct per column. Any other difference fails the test.
LABEL_KEYED_COLUMNS = {"_entity_classification"}


def _plain(value: Any) -> Any:
    """``_cell`` with Parquet's read-back containers (numpy arrays) taken as lists."""
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return _cell(value)


def _results(case: Case) -> dict[str, Any]:
    data = TextRecordsInput(records=[TextRecord(id=i, text=t) for i, t in RECORDS])
    with tempfile.TemporaryDirectory() as artifacts:
        anonymizer = Anonymizer(artifact_path=artifacts)
        with scripted(case):
            run = anonymizer.run(config=case.config(), data=data)
        with scripted(case):
            preview = anonymizer.preview(config=case.config(), data=data, num_records=PREVIEW_ROWS)
        with scripted(case):
            evaluated = anonymizer.evaluate(run, config=case.evaluate)
    return {"run": run, "preview": preview, "evaluate": evaluated}


_MISSING = (None, {"$": "NaN"})


def _known(column: str, written: Any, read: Any) -> bool:
    if written in _MISSING and read in _MISSING:
        return True
    if column in LABEL_KEYED_COLUMNS and isinstance(written, dict) and isinstance(read, dict):
        return {k: v for k, v in read.items() if v is not None} == written
    return False


def _unexpected(frame: pd.DataFrame) -> list[str]:
    """One line per cell that reads back differently, other than the two known kinds."""
    buffer = io.BytesIO()
    frame.to_parquet(buffer)
    back = pd.read_parquet(io.BytesIO(buffer.getvalue()))
    assert list(back.columns) == list(frame.columns)
    return [
        f"{column}[{row}]: wrote {w!r}, read {r!r}"
        for column in frame.columns
        for row, (w, r) in enumerate(zip(map(_plain, frame[column]), map(_plain, back[column]), strict=True))
        if w != r and not _known(str(column), w, r)
    ]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_frames_read_back_as_written(case: Case) -> None:
    for call, result in _results(case).items():
        for frame in ("dataframe", "trace_dataframe"):
            unexpected = _unexpected(getattr(result, frame))
            assert not unexpected, f"{call} {frame}:\n" + "\n".join(unexpected[:20])
