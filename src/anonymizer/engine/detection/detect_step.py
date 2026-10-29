# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``detect`` step shared by the Replace and Rewrite invocations.

One invocation-scoped operation calls the caller's detector over the whole input frame and returns each surviving
row's accepted mentions. The caller reads what the detector returned from a ``DetectState``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import repeat

import pandas as pd

from anonymizer.engine.constants import COL_FINAL_ENTITIES, COL_TEXT
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult
from anonymizer.engine.graph_core.authoring import DATUM, MENTIONS, Datum, Mention, Operation, Requests
from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.engine.ndd.backend import per_target
from anonymizer.engine.row_partitioning import try_resolve_source_positions
from anonymizer.engine.schemas import EntitiesSchema


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

    A result that is not verifiably one input row per returned row (see ``resolve_positions``) fails every target.
    """

    def run_detection(target: tuple[Datum, ...]) -> list[tuple[Mention, ...] | None]:
        started = time.perf_counter()
        state.detected.append(result := detect(dataframe))
        state.elapsed.append(time.perf_counter() - started)
        state.failures.extend(result.failed_records)
        positions = resolve_positions(dataframe, result)
        if positions is None:
            state.unresolved.append(True)
            return [None] * len(target)
        surviving = [target[position] for position in positions]
        state.owners.extend(surviving)
        raw_entities = result.dataframe.get(COL_FINAL_ENTITIES, repeat({}, len(positions)))
        for datum, raw in zip(surviving, raw_entities):
            state.rows[datum] = EntitiesSchema.from_raw(raw)
        return per_target([tuple(accepted_mentions(d, state.rows[d])) for d in surviving], positions, len(target))

    return Operation.per_invocation(
        "detect", run_detection, {"target": DATUM}, {"mentions": MENTIONS}, Requests.opaque()
    )


def missing(text: object) -> bool:
    return not isinstance(text, str) and bool(pd.isna(text))


def resolve_positions(dataframe: pd.DataFrame, result: EntityDetectionResult) -> tuple[int, ...] | None:
    """One distinct input position per returned row, or ``None``.

    Positions the detector did not report (the default ``()``) are accepted only if every returned row's text equals the
    text of the input row at its position, a missing value equalling a missing value.
    """
    reported = result.source_positions
    positions = try_resolve_source_positions(
        dataframe, result.dataframe, reported or (None if len(result.dataframe) else ())
    )
    if positions is None:
        return None
    if reported or COL_TEXT not in result.dataframe.columns:
        return positions
    given = [None if missing(text) else text for text in result.dataframe[COL_TEXT]]
    expected = [None if missing(text) else text for text in dataframe[COL_TEXT].iloc[list(positions)]]
    return positions if given == expected else None


def accepted_mentions(datum: Datum, entities: EntitiesSchema) -> list[Mention]:
    """Exact, non-overlapping spans in start order: the output regions the graph owns."""
    accepted: list[Mention] = []
    cursor = 0
    text = datum.text
    for e in sorted(entities.entities, key=lambda e: e.start_position):
        start, end = e.start_position, e.end_position
        if start >= cursor and start < end <= len(text) and text[start:end] == e.value:
            accepted.append(Mention(datum, start, end, e.label or "", e.value))
            cursor = end
    return accepted
