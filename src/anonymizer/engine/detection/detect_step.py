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
from typing import Any, Literal, NewType, cast

import pandas as pd

from anonymizer.engine.constants import COL_ENTITIES_BY_VALUE, COL_FINAL_ENTITIES, RECORD_ID_COLUMN
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult, _build_entities_by_value
from anonymizer.engine.errors import AnonymizerWorkflowError
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
from anonymizer.engine.protection_policy import UNSPANNED_VALUE
from anonymizer.engine.row_partitioning import resolve_source_positions
from anonymizer.engine.schemas import EntitiesByValueSchema, EntitiesSchema
from anonymizer.engine.schemas.detection import EntitySchema
from anonymizer.engine.schemas.shared import to_python
from anonymizer.engine.value_matching import count_values, entity_spans, normalized

RecordId = NewType("RecordId", str)
"""The facade's label for one input record, from its ``_anonymizer_record_id``; it lives within one call."""


@dataclass(frozen=True)
class RowDetections:
    """One input record's detections: its record id and the kept (collapsed) entities."""

    record_id: RecordId
    entities: EntitiesSchema
    added: tuple[EntitySchema, ...] = ()  # spans the engine added at a built-in detector's missed repeats
    # Reported values the kept spans do not protect, as ``(value, labels)``: the detector's own pairs and the values of
    # spans the collapse dropped, where they occur outside every kept span. Rewrite's view lists them.
    extras: tuple[tuple[str, tuple[str, ...]], ...] = ()


@dataclass
class DetectState:
    """What the ``detect`` step recorded: the result, the target that owns each returned row, and its timing."""

    detected: list[EntityDetectionResult] = field(default_factory=list)
    owners: list[Datum] = field(default_factory=list)
    elapsed: list[float] = field(default_factory=list)
    rows: dict[Datum, RowDetections] = field(default_factory=dict)
    failures: list[FailedRecord] = field(default_factory=list)


def detect_step(
    dataframe: pd.DataFrame,
    detect: Callable[[pd.DataFrame], EntityDetectionResult],
    state: DetectState,
    *,
    repeats: Literal["release", "fail", "protect"] = "release",
) -> Operation:
    """The ``detect`` operation over ``dataframe``: one ``detect`` call, its outcome recorded in ``state``.

    The seam has already placed every returned row at one distinct input position (``resolve_positions``).

    ``repeats`` says what happens to a reported value that occurs in the text outside every kept span (a missed
    repeat). Replace replaces spans and nothing else, so with ``"fail"`` (a custom detector) the row fails with
    ``unspanned_value``, and with ``"protect"`` (the built-in detector) a span is added at each such occurrence, with
    the label of the first returned span whose value has the same normalized key. ``"release"`` (Rewrite) leaves it
    to the by-value view the rewriter reads.
    """

    def run_detection(target: tuple[Datum, ...]) -> list[tuple[object, ...] | None]:
        started = time.perf_counter()
        state.detected.append(result := detect(dataframe))
        state.elapsed.append(time.perf_counter() - started)
        state.failures.extend(result.failed_records)
        positions = resolve_positions(dataframe, result)
        ids = _input_ids(dataframe)
        record_ids = [ids[position] for position in positions]
        _check_carried_ids(result.dataframe, record_ids)
        surviving = [target[position] for position in positions]
        state.owners.extend(surviving)
        raw_entities = result.dataframe.get(COL_FINAL_ENTITIES, repeat({}, len(positions)))
        rebuilt: dict[int, set[int]] = {}  # row -> the entities it loses
        appended: dict[int, list[EntitySchema]] = {}  # row -> its kept spans once repeat spans are added
        extras: dict[int, tuple[tuple[str, tuple[str, ...]], ...]] = {}  # row -> its view's extra pairs
        malformed: set[Datum] = set()
        unspanned: set[Datum] = set()
        by_value = result.dataframe.get(COL_ENTITIES_BY_VALUE, repeat(None, len(positions)))
        for row, (datum, raw, listed) in enumerate(zip(surviving, raw_entities, by_value)):
            try:  # the parse Replace applies, so a malformed payload fails its record in every mode
                entities = EntitiesSchema.from_raw(raw, strict=True)
            except ValueError:
                malformed.add(datum)
                state.rows[datum] = RowDetections(record_ids[row], EntitiesSchema())
                continue
            returned = list(entities.entities)  # before the collapse: a dropped span's value counts too
            spanned = [e.value for e in returned]
            if covered := covered_spans(datum, entities):
                rebuilt[row] = covered
                entities = EntitiesSchema(entities=[e for i, e in enumerate(entities.entities) if i not in covered])
            # A span that breaks the contract fails the row with its own reason first.
            sound = not mention_problem(accepted_mentions(datum, entities))
            # Replace replaces spans only: every value the detector reported (its spans' values, collapsed spans
            # included, and its pairs) must lie inside a kept span wherever it occurs.
            pairs = _listed_pairs(listed)
            values = spanned + [value for value, _ in pairs]
            added: list[EntitySchema] = []
            absorbed: list[EntitySchema] = []  # kept spans an added span contains
            detected = entities  # the kept spans before any repeat span is added
            if repeats == "protect" and sound:
                added = _repeat_spans(datum.text, entities, values, _first_labels(returned, pairs))
                if added:  # an added span that contains a kept span is the outer span: collapse again
                    merged = EntitiesSchema(entities=[*entities.entities, *added])
                    inner = covered_spans(datum, merged)
                    absorbed = [e for i, e in enumerate(merged.entities) if i in inner and e.source != "repeat"]
                    entities = EntitiesSchema(entities=[e for i, e in enumerate(merged.entities) if i not in inner])
                    appended[row] = list(entities.entities)
            # The view's extras are judged against the detected spans, before repeat spans were added, and a kept span
            # an added span absorbed keeps its value: the Substitute map is asked for every value the base asked for,
            # so a model that maps only an inner value (a card's last four digits) can still complete the map. Rewrite
            # adds no spans, so for it ``detected`` is the kept set.
            extras[row] = _extra_pairs(datum.text, detected, listed, raw) + tuple(
                (e.value, (e.label,)) for e in absorbed
            )
            state.rows[datum] = RowDetections(record_ids[row], entities, tuple(added), extras[row])
            if repeats != "release" and sound and values_outside_spans(datum.text, entities, values):
                unspanned.add(datum)
            # The by-value rule: a row with spans gets the view of its kept spans plus every detector pair, and every
            # collapsed span's value, that occurs in its text outside them (see ``_with_entities``); a row without
            # spans keeps the detector's view.
            if entities.entities:
                rebuilt.setdefault(row, set())
        # Later steps read the by-value column for every returned row, so it is added even when no row is rebuilt.
        frame = result.dataframe
        if rebuilt or COL_ENTITIES_BY_VALUE not in frame.columns:
            frame = _with_entities(frame, rebuilt, appended, extras)
        # Every returned row carries the input record id at its resolved position: later stages name it by that id.
        if RECORD_ID_COLUMN not in frame.columns:
            frame = frame.assign(**{RECORD_ID_COLUMN: pd.Series(record_ids, index=frame.index, dtype=object)})
        if frame is not result.dataframe:
            state.detected[-1] = dataclasses.replace(result, dataframe=frame)
        # A non-mention entry fails the record with ``invalid_output``; a ``Refusal`` with its own reason.
        mentions: list[tuple[object, ...]] = [
            ("malformed",)
            if d in malformed
            else (Refusal(UNSPANNED_VALUE),)
            if d in unspanned
            else tuple(accepted_mentions(d, state.rows[d].entities))
            for d in surviving
        ]
        return per_target(mentions, positions, len(target))

    return Operation.phase_scoped("detect", run_detection, {"target": DATUM}, {"mentions": MENTIONS}, Requests.opaque())


def _input_ids(dataframe: pd.DataFrame) -> list[RecordId]:
    """Each input row's record id; the seam attaches the column, so a frame without it is a wiring fault."""
    if RECORD_ID_COLUMN not in dataframe.columns:
        raise AnonymizerWorkflowError(f"The detect step's input has no {RECORD_ID_COLUMN} column")
    return [RecordId(str(i)) for i in dataframe[RECORD_ID_COLUMN]]


def _check_carried_ids(returned: pd.DataFrame, record_ids: list[RecordId]) -> None:
    """A returned frame that carries record ids must carry each placed row's input id."""
    if RECORD_ID_COLUMN in returned.columns and [str(i) for i in returned[RECORD_ID_COLUMN]] != record_ids:
        raise AnonymizerWorkflowError("The detection result's record ids disagree with the rows it is placed at")


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


def _with_entities(
    frame: pd.DataFrame,
    covered: dict[int, set[int]],
    appended: dict[int, list[EntitySchema]] | None = None,
    extras: dict[int, tuple[tuple[str, tuple[str, ...]], ...]] | None = None,
) -> pd.DataFrame:
    """A copy of ``frame`` whose rows at the positions in ``covered`` lose those entities, with their by-value view
    rebuilt (the column is added when the detector returned none; a row not in ``covered`` keeps its cell).

    The invariant: for each row, every value the detector reports, from a span or a by-value pair, that occurs in the
    row's text is either in the replacement view or fails the row. Nothing reported is dropped silently, and released
    text never holds such a value unflagged. So a rebuilt view is the view of the kept spans plus the row's ``extras``
    (``RowDetections.extras``, computed once by the detect step); this function only renders them.

    A rewritten entities cell keeps its own encoding (a JSON string stays a string) and its own keys. A rebuilt
    by-value cell takes the encoding of the column's first present cell, else of its row's entities cell, so a
    homogeneous column stays homogeneous and the trace can be written to parquet. Seeds do not depend on it: later
    stages encode every seed cell on its own."""
    frame = frame.copy()
    extras = extras or {}
    has_entities = COL_FINAL_ENTITIES in frame.columns
    given = list(frame[COL_FINAL_ENTITIES]) if has_entities else [None] * len(frame)
    appended = appended or {}
    final = [
        _with_added(raw, appended[row]) if row in appended else _without(raw, covered[row]) if covered.get(row) else raw
        for row, raw in enumerate(given)
    ]
    if has_entities:
        frame[COL_FINAL_ENTITIES] = final
    by_value = list(frame[COL_ENTITIES_BY_VALUE]) if COL_ENTITIES_BY_VALUE in frame.columns else [None] * len(given)
    sample = next((cell for cell in by_value if not _absent(cell)), None)
    frame[COL_ENTITIES_BY_VALUE] = [
        _encoded_like(
            given[row] if sample is None else sample,
            _union(
                _build_entities_by_value(final[row]),
                extras.get(row, ()),
            ),
        )
        if row in covered
        else raw
        for row, raw in enumerate(by_value)
    ]
    return frame


def _extra_pairs(
    text: str, kept: EntitiesSchema, listed: object, returned: object
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """The pairs whose value occurs in ``text`` outside every span of ``kept``: those of the detector's view
    ``listed``, then those of the spans it ``returned`` before the collapse, so a collapsed span's value is listed
    wherever no kept span covers it."""
    candidates = unspanned_pairs(listed, text, kept) + unspanned_pairs(_build_entities_by_value(returned), text, kept)
    return tuple((value, tuple(labels)) for value, labels in candidates)


def _union(derived: dict[str, Any], extras: tuple[tuple[str, tuple[str, ...]], ...]) -> dict[str, Any]:
    """``derived`` plus the row's ``extras`` (``RowDetections.extras``), each label once per value."""
    items = {item["value"]: list(item["labels"]) for item in derived["entities_by_value"]}
    for value, labels in extras:
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
    judged = {normalized(leftover.value): leftover for leftover in count_values(text, spans, "", distinct)}
    kept = [(span.start, span.end) for span, _ in spans if span is not None]
    outside = set()
    for value in distinct:
        leftover = judged.get(normalized(value))
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


def _with_added(raw: object, kept: list[EntitySchema]) -> object:
    """The entities payload ``raw`` with its entities replaced by ``kept`` (a row that gained repeat spans), in its
    own encoding and keeping its other keys; ``raw`` itself when the row gained none."""
    if not kept:
        return raw
    payload = cast(dict[str, Any], to_python(json.loads(raw) if isinstance(raw, str) else raw))
    return _encoded_like(raw, {**payload, "entities": [e.model_dump(mode="json") for e in kept]})


def _first_labels(returned: list[EntitySchema], pairs: list[tuple[str, list[str]]]) -> dict[str, str]:
    """Each reported value's label for an added span, by normalized key: the first returned span's (in detector
    order, before the collapse), else the value's first by-value label."""
    labels: dict[str, str] = {}
    for entity in returned:
        if entity.label:
            labels.setdefault(normalized(entity.value), entity.label)
    for value, listed in pairs:
        if listed:
            labels.setdefault(normalized(value), listed[0])
    return labels


def _repeat_spans(text: str, kept: EntitiesSchema, values: list[str], labels: dict[str, str]) -> list[EntitySchema]:
    """A labelled span (``source="repeat"``) at each occurrence of a reported value outside every kept span.

    Occurrences are ``count_values``' undetected spans, and for a value it cannot match (a one-character key in an
    unspaced script) each plain match outside the kept spans. An occurrence that partly overlaps a kept or an earlier
    added span is left out, so the row still fails ``unspanned_value``; one that contains a kept span is added, and the
    collapse keeps it as the outer span."""
    distinct = [v for v in dict.fromkeys(values) if v]
    spans = entity_spans(text, kept.entities)
    taken = [(span.start, span.end) for span, _ in spans if span is not None]
    found: list[tuple[int, int, str]] = []
    judged = set()
    for leftover in count_values(text, spans, "", distinct):
        judged.add(normalized(leftover.value))
        found.extend((u.start, u.end, leftover.value) for u in leftover.undetected_spans)
    for value in distinct:
        if normalized(value) not in judged:
            start = text.find(value)
            while start >= 0:
                end = start + len(value)
                if not any(s <= start and end <= e for s, e in taken):
                    found.append((start, end, value))
                start = text.find(value, start + 1)
    added: list[EntitySchema] = []
    for start, end, value in sorted(set(found)):
        label = labels.get(normalized(value))
        crosses = any(s < end and start < e and not (start <= s and e <= end) for s, e in taken)
        if label and not crosses:
            taken.append((start, end))
            added.append(
                EntitySchema(
                    value=text[start:end], label=label, start_position=start, end_position=end, source="repeat"
                )
            )
    return added


def _encoded_like(raw: object, payload: dict[str, Any]) -> object:
    return json.dumps(payload, ensure_ascii=False) if isinstance(raw, str) else payload


def _absent(cell: object) -> bool:
    return cell is None or cell is pd.NA or (isinstance(cell, float) and cell != cell)
