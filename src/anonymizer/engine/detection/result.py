# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a detector returns, and the one conversion from it to the frame later steps read.

A custom ``detection_workflow`` returns a ``DetectorResult``: one ``DetectorRow`` per answered input record, named by
the input's ``_anonymizer_record_id``, plus the records it failed. The built-in ``EntityDetectionWorkflow`` returns
its own frame, which ``place_detection_frame`` places by source position. Either way the seam hands the later steps an
``EntityDetectionResult`` whose rows are input rows, in input order, carrying their input record id.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from dataclasses import dataclass

import pandas as pd

from anonymizer.engine.constants import (
    COL_ENTITIES_BY_VALUE,
    COL_FINAL_ENTITIES,
    COL_LATENT_ENTITIES,
    COL_TAG_NOTATION,
    COL_TAGGED_TEXT,
    COL_TEXT,
    RECORD_ID_COLUMN,
)
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult
from anonymizer.engine.detection.postprocess import EntitySpan, build_tagged_text, get_tag_notation
from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.engine.protection_policy import assess_any
from anonymizer.engine.row_partitioning import cell_text, resolve_source_positions
from anonymizer.engine.schemas import EntitiesByValueSchema, LatentEntitySchema
from anonymizer.engine.schemas.detection import EntityByValueSchema
from anonymizer.interface.errors import InvalidDetectorResultError


@dataclass(frozen=True)
class DetectedSpan:
    """One detected span: ``text[start:end] == value``, half-open, non-empty and labelled."""

    start: int
    end: int
    label: str
    value: str
    score: float = 1.0
    source: str = "detector"


@dataclass(frozen=True)
class DetectorRow:
    """A detector's answer for one input record, named by that record's ``_anonymizer_record_id``.

    ``extra_values`` are ``(value, label)`` pairs reported without a span; ``latent`` feeds Rewrite's latent stage.
    """

    record_id: str
    spans: tuple[DetectedSpan, ...] = ()
    extra_values: tuple[tuple[str, str], ...] = ()
    latent: tuple[LatentEntitySchema, ...] = ()


@dataclass(frozen=True)
class DetectorResult:
    """A custom detector's return value: the rows it answered and the records it failed (step ``detect``)."""

    rows: tuple[DetectorRow, ...] = ()
    failed: tuple[FailedRecord, ...] = ()


def validate_detector_result(source: pd.DataFrame, result: object) -> DetectorResult:
    """``result`` when it can be placed against ``source``; else ``InvalidDetectorResultError``.

    Batch-level rules: the result is a ``DetectorResult``; every row and every failure names an input record id; no
    id is answered twice or both answered and failed; every field is well typed: a span's ``start`` and ``end`` are
    ints, ``value``, ``label`` and ``source`` are strings and ``score`` is a number; an extra value is a non-empty
    ``(value, label)`` pair of strings; a latent entry is a ``LatentEntity``; a failure's step and reason are strings.
    A well-typed span that breaks the span contract (range, slice, an empty label, overlap) is not a batch error: it
    fails its own record at ``detect`` with that contract's reason.
    """
    if not isinstance(result, DetectorResult):
        raise InvalidDetectorResultError(
            f"detection_workflow.run returned {type(result).__name__}; return a DetectorResult "
            '(see "Custom detection workflows")'
        )
    if not _sequence(result.rows) or not _sequence(result.failed):
        raise InvalidDetectorResultError("DetectorResult.rows and .failed must each be a tuple or a list")
    ids = set(_input_ids(source))
    seen: Counter[str] = Counter()
    for row in result.rows:
        if not isinstance(row, DetectorRow):
            raise InvalidDetectorResultError(f"DetectorResult.rows holds a {type(row).__name__}, not a DetectorRow")
        if not all(_sequence(part) for part in (row.spans, row.extra_values, row.latent)):
            raise InvalidDetectorResultError(
                f"record {row.record_id!r}: spans, extra_values and latent must each be a tuple or a list"
            )
        _check_id(row.record_id, ids, "row")
        seen[row.record_id] += 1
        for span in row.spans:
            if not _well_typed(span):
                raise InvalidDetectorResultError(f"record {row.record_id!r} has a malformed span: {span!r}")
        for entity in row.latent:
            if not isinstance(entity, LatentEntitySchema):
                raise InvalidDetectorResultError(
                    f"record {row.record_id!r} has a latent entry that is not a LatentEntity"
                )
        for pair in row.extra_values:
            if not (isinstance(pair, tuple) and len(pair) == 2 and all(isinstance(v, str) and v for v in pair)):
                raise InvalidDetectorResultError(
                    f"record {row.record_id!r} has an extra value that is not a non-empty (value, label) pair"
                )
    for failure in result.failed:
        if not isinstance(failure, FailedRecord) or not all(isinstance(v, str) for v in (failure.step, failure.reason)):
            raise InvalidDetectorResultError("DetectorResult.failed holds an entry that is not a FailedRecord")
        _check_id(failure.record_id, ids, "failure")
        seen[failure.record_id] += 1
    if repeated := next((i for i, n in seen.items() if n > 1), None):
        raise InvalidDetectorResultError(f"record id {repeated!r} is answered or failed more than once")
    return result


def render_detector_result(source: pd.DataFrame, result: DetectorResult) -> EntityDetectionResult:
    """The frame later steps read for a validated ``result``: one row per answered input record, in input order.

    Each row keeps its input columns and gains ``final_entities`` (its spans), ``_entities_by_value`` (its spans'
    values plus its extra values), ``_tag_notation``, ``tagged_text`` (rendered from its spans) and
    ``_latent_entities``. A record neither answered nor failed fails with ``dropped_output``.
    """
    ids = _input_ids(source)
    position = {record_id: p for p, record_id in enumerate(ids)}
    rows = sorted(result.rows, key=lambda row: position[row.record_id])
    positions = tuple(position[row.record_id] for row in rows)
    frame = source.iloc[list(positions)].reset_index(drop=True)
    texts = [cell_text(t) for t in frame[COL_TEXT]] if COL_TEXT in frame.columns else [""] * len(rows)
    notations = [get_tag_notation(text) for text in texts]
    frame[COL_FINAL_ENTITIES] = [{"entities": [_entity(s) for s in row.spans]} for row in rows]
    frame[COL_ENTITIES_BY_VALUE] = [_by_value(row) for row in rows]
    frame[COL_TAG_NOTATION] = notations
    frame[COL_TAGGED_TEXT] = [
        build_tagged_text(text, _tag_spans(text, row.spans), notation=notation)
        for text, row, notation in zip(texts, rows, notations)
    ]
    frame[COL_LATENT_ENTITIES] = [
        {"latent_entities": [entity.model_dump(mode="json") for entity in row.latent]} for row in rows
    ]
    named = {row.record_id for row in rows} | {failure.record_id for failure in result.failed}
    reason = assess_any("dropped_record", detector="custom").reason
    dropped = [FailedRecord(record_id, "detect", reason) for record_id in ids if record_id not in named]
    return EntityDetectionResult(frame, [*result.failed, *dropped], positions)


def place_detection_frame(source: pd.DataFrame, result: object) -> EntityDetectionResult:
    """The built-in detector's ``result``, with every returned row placed at one distinct input position.

    The built-in workflow returns the adapter's frame, which carries the input record ids; a frame that cannot be
    placed raises ``AnonymizerWorkflowError``, as for any other workflow result.
    """
    if not isinstance(result, EntityDetectionResult):
        raise InvalidDetectorResultError(f"EntityDetectionWorkflow.run returned {type(result).__name__}")
    positions = resolve_source_positions(source, result.dataframe, result.source_positions or None)
    return dataclasses.replace(result, source_positions=positions)


def _sequence(value: object) -> bool:
    """Only a tuple or a list is accepted: an iterator would be consumed by validation, and ``None`` or a number is
    not a collection."""
    return isinstance(value, (tuple, list))


def _well_typed(span: object) -> bool:
    def integer(v: object) -> bool:
        return isinstance(v, int) and not isinstance(v, bool)

    return (
        isinstance(span, DetectedSpan)
        and integer(span.start)
        and integer(span.end)
        and all(isinstance(v, str) for v in (span.value, span.label, span.source))
        and (integer(span.score) or isinstance(span.score, float))
    )


def _input_ids(source: pd.DataFrame) -> list[str]:
    return [str(i) for i in source[RECORD_ID_COLUMN]]


def _check_id(record_id: object, ids: set[str], what: str) -> None:
    if not isinstance(record_id, str) or record_id not in ids:
        raise InvalidDetectorResultError(
            f"a DetectorResult {what} names {record_id!r}, which is not an input record id"
        )


def _entity(span: DetectedSpan) -> dict[str, object]:
    return {
        "value": span.value,
        "label": span.label,
        "start_position": span.start,
        "end_position": span.end,
        "score": span.score,
        "source": span.source,
    }


def _tag_spans(text: str, spans: tuple[DetectedSpan, ...]) -> list[EntitySpan]:
    """The sound, labelled spans to tag, outer first, less any span that overlaps one already taken: a span that
    breaks the contract fails its record at ``detect``, so it only needs to leave the rendering well formed."""
    taken: list[DetectedSpan] = []
    for s in sorted(spans, key=lambda s: (s.start, -s.end)):
        sound = 0 <= s.start < s.end <= len(text) and text[s.start : s.end] == s.value and s.label
        if sound and all(s.start >= t.end or s.end <= t.start for t in taken):
            taken.append(s)
    return [EntitySpan(f"e{i}", s.value, s.label, s.start, s.end, s.score, s.source) for i, s in enumerate(taken)]


def _by_value(row: DetectorRow) -> dict[str, object]:
    labels: dict[str, list[str]] = {}
    for value, label in [*((s.value, s.label) for s in row.spans), *row.extra_values]:
        if value:
            listed = labels.setdefault(value, [])
            if label and label not in listed:
                listed.append(label)
    items = [EntityByValueSchema(value=v, labels=ls) for v, ls in labels.items()]
    return EntitiesByValueSchema(entities_by_value=items).model_dump(mode="json")
