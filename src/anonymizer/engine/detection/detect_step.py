# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``detect`` step shared by the Replace and Rewrite invocations.

One phase-scoped operation calls the caller's detector over the whole input frame and returns each surviving
row's accepted mentions. The caller reads what the detector returned from a ``DetectState``.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import repeat
from typing import Any, cast

import pandas as pd

from anonymizer.engine.constants import COL_ENTITIES_BY_VALUE, COL_FINAL_ENTITIES, COL_TEXT
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult, _build_entities_by_value
from anonymizer.engine.graph_core.authoring import DATUM, MENTIONS, Datum, Mention, Operation, Requests, mention_problem
from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.engine.ndd.backend import per_target
from anonymizer.engine.row_partitioning import try_resolve_source_positions
from anonymizer.engine.schemas import EntitiesByValueSchema, EntitiesSchema
from anonymizer.engine.schemas.shared import to_python


@dataclass
class DetectState:
    """What the ``detect`` step recorded: the result, the target that owns each returned row, and its timing."""

    detected: list[EntityDetectionResult] = field(default_factory=list)
    owners: list[Datum] = field(default_factory=list)
    unresolved: list[bool] = field(default_factory=list)
    elapsed: list[float] = field(default_factory=list)
    rows: dict[Datum, EntitiesSchema] = field(default_factory=dict)
    failures: list[FailedRecord] = field(default_factory=list)


def detect_step(
    dataframe: pd.DataFrame, detect: Callable[[pd.DataFrame], EntityDetectionResult], state: DetectState
) -> Operation:
    """The ``detect`` operation over ``dataframe``: one ``detect`` call, its outcome recorded in ``state``.

    A result that is not verifiably one input row per returned row (see ``resolve_positions``), or that returns a row
    whose text differs from its input row's, fails every target; the caller then works over the returned frame.
    """

    def run_detection(target: tuple[Datum, ...]) -> list[tuple[object, ...] | None]:
        started = time.perf_counter()
        state.detected.append(result := detect(dataframe))
        state.elapsed.append(time.perf_counter() - started)
        state.failures.extend(result.failed_records)
        positions = resolve_positions(dataframe, result)
        returned = result.dataframe.get(COL_TEXT)
        if positions is not None and returned is not None:
            # The detector's offsets are for the text it returned: a changed text cannot be protected as the input.
            # A cell missing on both sides is unchanged, however each side spells the missing value.
            given = dataframe[COL_TEXT]
            changed = any(
                not (missing(t) and missing(given.iloc[p])) and str(t) != target[p].text
                for t, p in zip(returned, positions)
            )
            positions = None if changed else positions
        if positions is None:
            state.unresolved.append(True)
            return [None] * len(target)
        surviving = [target[position] for position in positions]
        state.owners.extend(surviving)
        raw_entities = result.dataframe.get(COL_FINAL_ENTITIES, repeat({}, len(positions)))
        rebuilt: dict[int, set[int]] = {}  # row -> the entities it loses
        malformed: set[Datum] = set()
        by_value = result.dataframe.get(COL_ENTITIES_BY_VALUE, repeat(None, len(positions)))
        for row, (datum, raw, listed) in enumerate(zip(surviving, raw_entities, by_value)):
            try:  # the parse Replace applies, so a malformed payload fails its record in every mode
                entities = EntitiesSchema.from_raw(raw, strict=True)
            except ValueError:
                malformed.add(datum)
                state.rows[datum] = EntitiesSchema()
                continue
            if covered := covered_spans(datum, entities):
                rebuilt[row] = covered
                entities = EntitiesSchema(entities=[e for i, e in enumerate(entities.entities) if i not in covered])
            state.rows[datum] = entities
            # A by-value view that is missing, empty or lacks a kept pair is rebuilt from the kept entities.
            if not _absent(listed) and {(e.value, e.label) for e in entities.entities} <= _pairs(listed):
                continue
            if entities.entities:
                rebuilt.setdefault(row, set())
        # Later steps read the by-value column for every returned row, so it is added even when no row is rebuilt.
        if rebuilt or COL_ENTITIES_BY_VALUE not in result.dataframe.columns:
            state.detected[-1] = dataclasses.replace(result, dataframe=_with_entities(result.dataframe, rebuilt))
        # A non-mention entry fails the record with ``invalid_output``.
        mentions: list[tuple[object, ...]] = [
            ("malformed",) if d in malformed else tuple(accepted_mentions(d, state.rows[d])) for d in surviving
        ]
        return per_target(mentions, positions, len(target))

    return Operation.phase_scoped("detect", run_detection, {"target": DATUM}, {"mentions": MENTIONS}, Requests.opaque())


def missing(text: object) -> bool:
    return not isinstance(text, str) and bool(pd.isna(text))


def resolve_positions(dataframe: pd.DataFrame, result: EntityDetectionResult) -> tuple[int, ...] | None:
    """One distinct input position per returned row, or ``None``: the detector's reported positions, else unique
    record IDs; positions it did not report (the default ``()``) are not inferred from equal text or row counts."""
    return try_resolve_source_positions(dataframe, result.dataframe, result.source_positions or None)


def accepted_mentions(datum: Datum, entities: EntitiesSchema) -> list[Mention]:
    """Every entity as a mention in start order, less the spans ``covered_spans`` drops."""
    covered = covered_spans(datum, entities)
    mentions = [
        Mention(datum, e.start_position, e.end_position, e.label or "", e.value)
        for i, e in enumerate(entities.entities)
        if i not in covered
    ]
    return sorted(mentions, key=lambda m: m.start)


def covered_spans(datum: Datum, entities: EntitiesSchema) -> set[int]:
    """The positions of the entities a labelled sound span already covers.

    Only spans that are sound on their own (in range, non-empty, slice equal to the value) are normalised, and only a
    labelled span covers another, since a span without a label gets no replacement. Of identical spans the first
    labelled one in the detector's order is kept, with its label: that order is the only preference a detector
    states, and it is the label Replace applied before spans were checked. A span contained in another gives way to
    the outer one, which protects every character either covers. Everything else, a partial overlap, an unlabelled
    span that would cover another, or any unsound span, is left for acceptance to reject with its reason.
    """
    mentions = [Mention(datum, e.start_position, e.end_position, e.label or "", e.value) for e in entities.entities]
    sound = [i for i, m in enumerate(mentions) if mention_problem((m,)) in ("", "mention_unlabelled")]
    covered: set[int] = set()
    reach = -1  # the furthest end of a labelled sound span starting at or before the current one
    # Outer before inner at the same start; within identical spans, labelled first, then detector order.
    for i in sorted(sound, key=lambda i: (mentions[i].start, -mentions[i].end, not mentions[i].label, i)):
        if mentions[i].end <= reach:
            covered.add(i)
        elif mentions[i].label:
            reach = mentions[i].end
    return covered


def _with_entities(frame: pd.DataFrame, covered: dict[int, set[int]]) -> pd.DataFrame:
    """A copy of ``frame`` whose rows at the positions in ``covered`` lose those entities, with their by-value view
    rebuilt (the column is added, with a missing cell for every other row, when the detector returned none). Each rewritten cell keeps its own row's encoding
    (a JSON string stays a string) and its own keys; later stages encode every seed cell on its own, so a column may
    mix encodings."""
    frame = frame.copy()
    has_entities = COL_FINAL_ENTITIES in frame.columns
    given = list(frame[COL_FINAL_ENTITIES]) if has_entities else [None] * len(frame)
    final = [_without(raw, covered[row]) if covered.get(row) else raw for row, raw in enumerate(given)]
    if has_entities:
        frame[COL_FINAL_ENTITIES] = final
    by_value = frame[COL_ENTITIES_BY_VALUE] if COL_ENTITIES_BY_VALUE in frame.columns else repeat(None)
    frame[COL_ENTITIES_BY_VALUE] = [
        # A rebuilt cell takes the encoding of its own stale cell, or, when that is missing, of its entities cell.
        _encoded_like(given[row] if _absent(raw) else raw, _build_entities_by_value(final[row]))
        if row in covered
        else raw
        for row, raw in zip(range(len(given)), by_value)
    ]
    return frame


def _without(raw: object, covered: set[int]) -> object:
    """The entities payload ``raw`` less the entities at ``covered``, every other key as the detector wrote it."""
    payload = cast(dict[str, Any], to_python(json.loads(raw) if isinstance(raw, str) else raw))
    entities = payload["entities"]
    if not isinstance(entities, list):  # a wrapped model or another shape ``from_raw`` reads: use what it read
        entities = [e.model_dump(mode="json") for e in EntitiesSchema.from_raw(raw).entities]
    return _encoded_like(raw, {**payload, "entities": [e for i, e in enumerate(entities) if i not in covered]})


def _pairs(cell: object) -> set[tuple[str, str]]:
    return {
        (item.value, label) for item in EntitiesByValueSchema.from_raw(cell).entities_by_value for label in item.labels
    }


def _encoded_like(raw: object, payload: dict[str, Any]) -> object:
    return json.dumps(payload, ensure_ascii=False) if isinstance(raw, str) else payload


def _absent(cell: object) -> bool:
    return cell is None or cell is pd.NA or (isinstance(cell, float) and cell != cell)
