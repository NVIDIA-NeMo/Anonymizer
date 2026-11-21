# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test helper: build the ``DetectorResult`` a custom detector returns from frame-shaped fixtures.

Many fixtures describe a detection as rows of ``final_entities`` (and optionally ``_entities_by_value`` pairs and
``_latent_entities``). ``as_detector_result`` turns such a frame into a ``DetectorResult`` for the input ``frame`` it
answers, so a fixture keeps its data and returns the type the seam accepts. It is a test helper only: the product
accepts a frame from the built-in ``EntityDetectionWorkflow`` alone.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import cast
from unittest.mock import Mock

import pandas as pd

from anonymizer.engine.constants import COL_ENTITIES_BY_VALUE, COL_FINAL_ENTITIES, COL_LATENT_ENTITIES, COL_TEXT
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult, EntityDetectionWorkflow
from anonymizer.engine.detection.result import DetectedSpan, DetectorResult, DetectorRow
from anonymizer.engine.ndd.adapter import RECORD_ID_COLUMN, FailedRecord
from anonymizer.engine.schemas import EntitiesByValueSchema, LatentEntitiesSchema


def _parse(raw: object) -> object:
    return json.loads(raw) if isinstance(raw, str) else raw


def _located(e: dict, text: str) -> tuple[int, int]:
    """A fixture entity's offsets; one written without offsets is placed at its value's first occurrence."""
    if "start_position" in e:
        return int(e["start_position"]), int(e["end_position"])
    start = text.find(str(e.get("value", "")))
    return start, start + len(str(e.get("value", "")))


def _spans(raw: object, text: str = "") -> tuple[DetectedSpan, ...]:
    payload = _parse(raw)
    if payload is None or (isinstance(payload, float) and payload != payload):
        return ()
    entities = cast(list[dict], payload.get("entities", []) if isinstance(payload, dict) else payload)
    return tuple(
        DetectedSpan(
            start=_located(e, text)[0],
            end=_located(e, text)[1],
            label=str(e.get("label") or ""),
            value=str(e.get("value", "")),
            score=float(e.get("score", 1.0) or 1.0),
            source=str(e.get("source") or "detector"),
        )
        for e in entities
    )


def _pairs(raw: object, spans: tuple[DetectedSpan, ...]) -> tuple[tuple[str, str], ...]:
    if raw is None or (isinstance(raw, float) and raw != raw):
        return ()
    spanned = {(s.value, s.label) for s in spans}
    pairs = []
    for item in EntitiesByValueSchema.from_raw(raw).entities_by_value:
        for label in item.labels:
            if item.value and label and (item.value, label) not in spanned:
                pairs.append((item.value, label))
    return tuple(pairs)


def _latent(raw: object) -> tuple:
    if raw is None or (isinstance(raw, float) and raw != raw):
        return ()
    return tuple(LatentEntitiesSchema.model_validate(_parse(raw)).latent_entities)


def as_detector_result(
    frame: pd.DataFrame,
    returned: pd.DataFrame | EntityDetectionResult,
    positions: Sequence[int] | None = None,
    failed: Sequence[FailedRecord] = (),
) -> DetectorResult:
    """A ``DetectorResult`` answering ``frame``'s rows at ``positions`` (default: every row, in order) with the
    detection columns of ``returned``'s rows, one per position. A failure named by a row index is renamed to the
    input record id at that index."""
    if isinstance(returned, EntityDetectionResult):
        failed = [*failed, *returned.failed_records]
        positions = positions if positions is not None else (returned.source_positions or None)
        returned = returned.dataframe
    ids = [str(i) for i in frame[RECORD_ID_COLUMN]]
    positions = list(range(min(len(returned), len(ids)))) if positions is None else list(positions)
    rows = []
    for row, position in enumerate(positions):
        cells = returned.iloc[row]
        text = frame[COL_TEXT].iloc[position] if COL_TEXT in frame.columns else ""
        spans = _spans(cells.get(COL_FINAL_ENTITIES), text if isinstance(text, str) else "")
        rows.append(
            DetectorRow(
                record_id=ids[position],
                spans=spans,
                extra_values=_pairs(cells.get(COL_ENTITIES_BY_VALUE), spans),
                latent=_latent(cells.get(COL_LATENT_ENTITIES)),
            )
        )
    named = [
        FailedRecord(ids[int(f.record_id)], f.step, f.reason)
        if f.record_id not in ids and str(f.record_id).isdigit() and int(f.record_id) < len(ids)
        else f
        for f in failed
    ]
    return DetectorResult(rows=tuple(rows), failed=tuple(named))


def input_record_ids(data: object, nrows: int | None = None) -> list[str]:
    """The record id the seam gives each input row of ``data``: the ``TextRecord`` id, else the row's uuid5."""
    from anonymizer.engine.io.reader import read_input
    from anonymizer.engine.row_partitioning import with_record_ids

    frame = with_record_ids(read_input(data, nrows=nrows).dataframe)  # ty: ignore[invalid-argument-type]
    return [str(i) for i in frame[RECORD_ID_COLUMN]]


def returns_detector_result(detector: Mock) -> Mock:
    """``detector`` (a ``Mock(spec=EntityDetectionWorkflow)``) with ``run`` returning its configured
    ``run.return_value`` frame as a ``DetectorResult`` for the frame it is called with; a ``DetectorResult`` return
    value passes through. Call assertions on ``detector.run`` are unchanged."""

    mock_run = detector.run

    def run(frame: pd.DataFrame, *_: object, **__: object) -> DetectorResult:
        value = mock_run.return_value
        if isinstance(value, DetectorResult):
            return value
        # A fixture failure named by a made-up id ("r1") names the input record at its own position instead.
        ids = [str(i) for i in frame[RECORD_ID_COLUMN]]
        failed = [
            f if f.record_id in ids or str(f.record_id).isdigit() else FailedRecord(ids[k], f.step, f.reason)
            for k, f in enumerate(value.failed_records)
        ]
        failing = {f.record_id if f.record_id in ids else ids[int(f.record_id)] for f in failed}
        frame_rows = [p for p in range(min(len(value.dataframe), len(ids))) if ids[p] not in failing]
        return as_detector_result(frame, value.dataframe.iloc[frame_rows], frame_rows, failed)

    mock_run.side_effect = run
    return detector


def builtin_stub(run: object) -> EntityDetectionWorkflow:
    """The built-in ``EntityDetectionWorkflow`` with its ``run`` replaced by ``run`` (a callable or a ``Mock``): its
    type is exactly the built-in class, so the seam reads its frame as the built-in detector's."""
    detector = EntityDetectionWorkflow(adapter=Mock())
    detector.run = run if isinstance(run, Mock) else Mock(side_effect=run)
    return detector
