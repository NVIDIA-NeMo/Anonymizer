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
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from itertools import repeat
from typing import Any, cast

import pandas as pd

from anonymizer.engine.constants import COL_ENTITIES_BY_VALUE, COL_FINAL_ENTITIES, COL_TEXT
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult, _build_entities_by_value
from anonymizer.engine.graph_core.authoring import (
    DATUM,
    MENTIONS,
    Datum,
    Mention,
    Operation,
    Refusal,
    Requests,
    mention_problem,
)
from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.engine.ndd.backend import per_target
from anonymizer.engine.row_partitioning import cell_text, resolve_source_positions
from anonymizer.engine.schemas import EntitiesByValueSchema, EntitiesSchema
from anonymizer.engine.schemas.shared import to_python
from anonymizer.engine.value_matching import _normalized, count_values, entity_spans

# Whether Replace fails a built-in detection row whose detected value occurs outside every kept span. Custom detectors
# are always checked. Off until it is ruled: a built-in span model that misses a repeat would then fail the row.
UNSPANNED_CHECK_BUILTIN = False


@dataclass
class DetectState:
    """What the ``detect`` step recorded: the result, the target that owns each returned row, and its timing."""

    detected: list[EntityDetectionResult] = field(default_factory=list)
    owners: list[Datum] = field(default_factory=list)
    elapsed: list[float] = field(default_factory=list)
    rows: dict[Datum, EntitiesSchema] = field(default_factory=dict)
    failures: list[FailedRecord] = field(default_factory=list)


def detect_step(
    dataframe: pd.DataFrame,
    detect: Callable[[pd.DataFrame], EntityDetectionResult],
    state: DetectState,
    *,
    spans_only: bool = False,
) -> Operation:
    """The ``detect`` operation over ``dataframe``: one ``detect`` call, its outcome recorded in ``state``.

    The seam has already placed every returned row at one distinct input position (``resolve_positions``).

    With ``spans_only`` (Replace, which replaces spans and nothing else), a row whose by-value view lists a value that
    occurs in its text outside every kept span fails with ``unspanned_value``: no step would protect that value.
    """

    def run_detection(target: tuple[Datum, ...]) -> list[tuple[object, ...] | None]:
        started = time.perf_counter()
        state.detected.append(result := detect(dataframe))
        state.elapsed.append(time.perf_counter() - started)
        state.failures.extend(result.failed_records)
        positions = resolve_positions(dataframe, result)
        surviving = [target[position] for position in positions]
        state.owners.extend(surviving)
        raw_entities = result.dataframe.get(COL_FINAL_ENTITIES, repeat({}, len(positions)))
        rebuilt: dict[int, set[int]] = {}  # row -> the entities it loses
        malformed: set[Datum] = set()
        unspanned: set[Datum] = set()
        by_value = result.dataframe.get(COL_ENTITIES_BY_VALUE, repeat(None, len(positions)))
        for row, (datum, raw, listed) in enumerate(zip(surviving, raw_entities, by_value)):
            try:  # the parse Replace applies, so a malformed payload fails its record in every mode
                entities = EntitiesSchema.from_raw(raw, strict=True)
            except ValueError:
                malformed.add(datum)
                state.rows[datum] = EntitiesSchema()
                continue
            spanned = [e.value for e in entities.entities]  # before the collapse: a dropped span's value counts too
            if covered := covered_spans(datum, entities):
                rebuilt[row] = covered
                entities = EntitiesSchema(entities=[e for i, e in enumerate(entities.entities) if i not in covered])
            state.rows[datum] = entities
            # A span that breaks the contract fails the row with its own reason first.
            sound = not mention_problem(accepted_mentions(datum, entities))
            # Replace replaces spans only: every value the detector reported (its spans' values, collapsed spans
            # included, and its pairs) must lie inside a kept span wherever it occurs.
            values = spanned + [value for value, _ in _listed_pairs(listed)]
            if spans_only and sound and values_outside_spans(datum.text, entities, values):
                unspanned.add(datum)
            # The by-value rule: a row with spans gets the view of its kept spans plus every detector pair, and every
            # collapsed span's value, that occurs in its text outside them (see ``_with_entities``); a row without
            # spans keeps the detector's view.
            if entities.entities:
                rebuilt.setdefault(row, set())
        # Later steps read the by-value column for every returned row, so it is added even when no row is rebuilt.
        if rebuilt or COL_ENTITIES_BY_VALUE not in result.dataframe.columns:
            state.detected[-1] = dataclasses.replace(result, dataframe=_with_entities(result.dataframe, rebuilt))
        # A non-mention entry fails the record with ``invalid_output``; a ``Refusal`` with its own reason.
        mentions: list[tuple[object, ...]] = [
            ("malformed",)
            if d in malformed
            else (Refusal("unspanned_value"),)
            if d in unspanned
            else tuple(accepted_mentions(d, state.rows[d]))
            for d in surviving
        ]
        return per_target(mentions, positions, len(target))

    return Operation.phase_scoped("detect", run_detection, {"target": DATUM}, {"mentions": MENTIONS}, Requests.opaque())


def missing(text: object) -> bool:
    return not isinstance(text, str) and bool(pd.isna(text))


def resolve_positions(dataframe: pd.DataFrame, result: EntityDetectionResult) -> tuple[int, ...]:
    """One distinct input position per returned row: the reported positions, else unique record ids; a result that
    cannot be placed raises ``AnonymizerWorkflowError``."""
    return resolve_source_positions(dataframe, result.dataframe, result.source_positions or None)


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
    rebuilt (the column is added when the detector returned none; a row not in ``covered`` keeps its cell).

    The invariant: for each row, every value the detector reports, from a span or a by-value pair, that occurs in the
    row's text is either in the replacement view or fails the row. Nothing reported is dropped silently, and released
    text never holds such a value unflagged. So a rebuilt view is the view of the kept spans plus every pair of the
    detector's own view, and every value of a span the collapse dropped, that occurs in the row's text outside those
    spans; a value inside a kept span is protected by that span, and a value that does not occur protects nothing.

    A rewritten entities cell keeps its own encoding (a JSON string stays a string) and its own keys. A rebuilt
    by-value cell takes the encoding of the column's first present cell, else of its row's entities cell, so a
    homogeneous column stays homogeneous and the trace can be written to parquet. Seeds do not depend on it: later
    stages encode every seed cell on its own."""
    frame = frame.copy()
    has_entities = COL_FINAL_ENTITIES in frame.columns
    given = list(frame[COL_FINAL_ENTITIES]) if has_entities else [None] * len(frame)
    final = [_without(raw, covered[row]) if covered.get(row) else raw for row, raw in enumerate(given)]
    if has_entities:
        frame[COL_FINAL_ENTITIES] = final
    by_value = list(frame[COL_ENTITIES_BY_VALUE]) if COL_ENTITIES_BY_VALUE in frame.columns else [None] * len(given)
    texts = [cell_text(t) for t in frame[COL_TEXT]] if COL_TEXT in frame.columns else [""] * len(given)
    sample = next((cell for cell in by_value if not _absent(cell)), None)
    frame[COL_ENTITIES_BY_VALUE] = [
        _encoded_like(
            given[row] if sample is None else sample,
            _union(_build_entities_by_value(final[row]), raw, texts[row], final[row], given[row]),
        )
        if row in covered
        else raw
        for row, raw in enumerate(by_value)
    ]
    return frame


def _union(derived: dict[str, Any], listed: object, text: str, kept: object, returned: object) -> dict[str, Any]:
    """``derived`` plus every pair whose value occurs in ``text`` outside the spans of ``kept``: the pairs of the
    detector's view ``listed``, and those of the spans it ``returned`` before the collapse, so a collapsed span's value
    is added wherever no kept span covers it."""
    items = {item["value"]: list(item["labels"]) for item in derived["entities_by_value"]}
    spans = EntitiesSchema.from_raw(kept)
    candidates = unspanned_pairs(listed, text, spans) + unspanned_pairs(_build_entities_by_value(returned), text, spans)
    for value, labels in candidates:
        merged = items.setdefault(value, [])
        merged.extend(label for label in labels if label not in merged)
    return {**derived, "entities_by_value": [{"value": v, "labels": labels} for v, labels in items.items()]}


def unspanned_pairs(listed: object, text: str, entities: EntitiesSchema) -> list[tuple[str, list[str]]]:
    """The pairs of the by-value cell ``listed`` whose value occurs in ``text`` outside every kept span of
    ``entities``: values the detector reported that its spans do not protect."""
    outside = values_outside_spans(text, entities, [value for value, _ in _listed_pairs(listed)])
    return [(value, labels) for value, labels in _listed_pairs(listed) if value in outside]


def values_outside_spans(text: str, entities: EntitiesSchema, values: Iterable[str]) -> set[str]:
    """The ``values`` with a match in ``text`` that no kept span contains, under the leftover check's own matching
    rule (``count_values``: whole words, case and accents folded). A match that crosses two spans is outside both.

    A value that rule cannot judge (it builds no matcher for a one-character key in an unspaced script, such as 王)
    fails closed: it is outside when a plain search finds it anywhere no kept span contains."""
    distinct = list(dict.fromkeys(values))
    spans = entity_spans(text, entities.entities)
    judged = {_normalized(leftover.value): leftover for leftover in count_values(text, spans, "", distinct)}
    kept = [(span.start, span.end) for span, _ in spans if span is not None]
    outside = set()
    for value in distinct:
        leftover = judged.get(_normalized(value))
        if leftover is not None and leftover.undetected_spans:
            outside.add(value)
        elif leftover is None and value and _found_outside(value, text, kept):
            outside.add(value)
    return outside


def _found_outside(value: str, text: str, kept: list[tuple[int, int]]) -> bool:
    start = text.find(value)
    while start >= 0:
        if not any(s <= start and start + len(value) <= e for s, e in kept):
            return True
        start = text.find(value, start + 1)
    return False


def _listed_pairs(listed: object) -> list[tuple[str, list[str]]]:
    if _absent(listed):
        return []
    return [
        (item.value, list(item.labels))
        for item in EntitiesByValueSchema.from_raw(listed).entities_by_value
        if item.value
    ]


def _without(raw: object, covered: set[int]) -> object:
    """The entities payload ``raw`` less the entities at ``covered``, every other key as the detector wrote it."""
    payload = cast(dict[str, Any], to_python(json.loads(raw) if isinstance(raw, str) else raw))
    entities = payload["entities"]
    if not isinstance(entities, list):  # a wrapped model or another shape ``from_raw`` reads: use what it read
        entities = [e.model_dump(mode="json") for e in EntitiesSchema.from_raw(raw).entities]
    return _encoded_like(raw, {**payload, "entities": [e for i, e in enumerate(entities) if i not in covered]})


def _encoded_like(raw: object, payload: dict[str, Any]) -> object:
    return json.dumps(payload, ensure_ascii=False) if isinstance(raw, str) else payload


def _absent(cell: object) -> bool:
    return cell is None or cell is pd.NA or (isinstance(cell, float) and cell != cell)
