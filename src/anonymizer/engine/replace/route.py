# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Redact, Annotate, Hash and Substitute facade routes on the graph kernel.

Each row lowers to one target datum whose exact, non-overlapping entity spans are its mentions. A row is a transform
(``replace``: the 0.4.0 per-row replacement) followed by a verifier (``verify-patches``: one patch per accepted
mention, checked by source-plus-patches reconstruction). The legacy DataFrame projection derives from
the canonical result without rerunning work; Annotate output is an annotation result and is never Protected.

Substitute runs as one invocation. Its invocation-scoped ``substitute-map`` step makes one ``LlmReplaceWorkflow`` call
over the whole frame (DataDesigner batches it) and returns each row's replacement map; a row DataDesigner dropped gets
no map and fails at that step. ``replace`` then reads the map from its ``replacement_map`` port, and the patch verifier
runs per row.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from itertools import repeat
from typing import Any

import pandas as pd

from anonymizer.config.replace_strategies import Annotate, LocalReplaceMethod, Substitute
from anonymizer.engine.constants import (
    COL_FINAL_ENTITIES,
    COL_REPLACED_TEXT,
    COL_REPLACEMENT_APPLICATION,
    COL_REPLACEMENT_MAP,
    COL_REPLACEMENT_MAP_SOURCE,
    COL_TEXT,
)
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult
from anonymizer.engine.graph_core.authoring import (
    DATUM,
    MENTIONS,
    TEXT,
    Assessment,
    Datum,
    GraphScope,
    Limits,
    Mention,
    Operation,
    ProtectionGraph,
    Requests,
    Workflow,
)
from anonymizer.engine.graph_core.execution import ProtectionResult, prepare, protect
from anonymizer.engine.graph_core.kernel import register_kind
from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.engine.replace.llm_replace_workflow import (
    REPLACEMENT_MAP_SOURCE_LLM,
    LlmReplaceWorkflow,
)
from anonymizer.engine.replace.replace_runner import ReplacementResult
from anonymizer.engine.replace.strategies import (
    _build_local_replacement_map,
    _parse_replacements,
    apply_replacements_to_spans,
)
from anonymizer.engine.row_partitioning import resolve_source_positions, try_resolve_source_positions
from anonymizer.engine.schemas import EntitiesSchema
from anonymizer.logging import ProgressTracker

logger = logging.getLogger("anonymizer")

REPLACEMENT_MAP = register_kind("replacement-map", lambda v: isinstance(v, dict))
REPLACEMENT_APPLICATION = register_kind("replacement-application", lambda v: isinstance(v, dict))


def run_local_route(
    dataframe: pd.DataFrame,
    *,
    strategy: LocalReplaceMethod,
    text_column: str = COL_TEXT,
    entities_column: str = COL_FINAL_ENTITIES,
) -> pd.DataFrame:
    """Run one local strategy over every row as one graph invocation and project the 0.4.0 columns."""
    output_df = dataframe.copy()
    tracker = ProgressTracker(total=len(output_df), label="Replacement")

    def build_map(position: int, entities: EntitiesSchema) -> dict[str, Any]:
        tracker.record_success()
        return _build_local_replacement_map(entities=entities, strategy=strategy)

    lowered = _lower(output_df, text_column, entities_column)
    graph, result = _protect_rows(lowered, type(strategy).__name__.lower(), build_map, isinstance(strategy, Annotate))
    output_df[COL_REPLACEMENT_MAP] = [result.value(d, "replacement_map") for d in graph.targets]
    _log_stats(output_df[COL_REPLACEMENT_MAP])
    output_df[COL_REPLACED_TEXT] = [result.value(d) for d in graph.targets]
    output_df[COL_REPLACEMENT_APPLICATION] = [result.value(d, "application") for d in graph.targets]
    tracker.log_final()
    return output_df


def run_substitute_route(
    dataframe: pd.DataFrame,
    *,
    llm_workflow: LlmReplaceWorkflow,
    replace_method: Substitute,
    map_arguments: dict[str, Any],
) -> ReplacementResult:
    """Run Substitute as one invocation: the whole-frame LLM map step, then ``replace`` and the verifier per row."""
    failures: list[FailedRecord] = []
    map_step = _substitute_map_step(
        llm_workflow, replace_method, map_arguments, failures, lambda target: (dataframe, target)
    )
    graph, result = _protect_rows(_lower(dataframe, COL_TEXT, COL_FINAL_ENTITIES), "substitute", None, False, map_step)
    kept = [position for position, target in enumerate(graph.targets) if result.outcome(target).available]
    output_df = dataframe.iloc[kept].reset_index(drop=True)
    targets = [graph.targets[position] for position in kept]
    output_df[COL_REPLACEMENT_MAP] = [result.value(d, "replacement_map") for d in targets]
    output_df[COL_REPLACEMENT_MAP_SOURCE] = REPLACEMENT_MAP_SOURCE_LLM
    output_df[COL_REPLACED_TEXT] = [result.value(d) for d in targets]
    output_df[COL_REPLACEMENT_APPLICATION] = [result.value(d, "application") for d in targets]
    return ReplacementResult(dataframe=output_df, failed_records=failures)


def _substitute_map_step(
    llm_workflow: LlmReplaceWorkflow,
    replace_method: Substitute,
    map_arguments: dict[str, Any],
    failures: list[FailedRecord],
    frame: Callable[[tuple[Datum, ...]], tuple[pd.DataFrame, Sequence[Datum]]],
) -> Operation:
    """One ``generate_map_only`` call over ``frame(target)``'s frame and the target each of its rows belongs to."""

    def generate_maps(target: tuple[Datum, ...]) -> list[dict[str, Any] | None]:
        source, owners = frame(target)
        result = llm_workflow.generate_map_only(source, instructions=replace_method.instructions, **map_arguments)
        failures.extend(result.failed_records)
        positions = resolve_source_positions(source, result.dataframe, result.source_positions)
        maps = (
            result.dataframe[COL_REPLACEMENT_MAP]
            if COL_REPLACEMENT_MAP in result.dataframe.columns
            else repeat({}, len(positions))
        )
        by_owner = {
            owners[position]: {"replacements": [dict(vars(r)) for r in _parse_replacements(rmap)]}
            for position, rmap in zip(positions, maps)
        }
        return [by_owner.get(datum) for datum in target]

    return Operation.per_invocation(
        "substitute-map", generate_maps, {"target": DATUM}, {"replacement_map": REPLACEMENT_MAP}, Requests.opaque()
    )


def run_replace_pipeline(
    dataframe: pd.DataFrame,
    *,
    detect: Callable[[pd.DataFrame], EntityDetectionResult],
    replace_method: LocalReplaceMethod | Substitute,
    llm_workflow: LlmReplaceWorkflow | None,
    map_arguments: dict[str, Any],
    reroute: Callable[[pd.DataFrame], ReplacementResult],
) -> ReplacementResult:
    """Run ``detect``, the Substitute map step and the per-target replace and verify as one invocation.

    A row the detector drops fails at ``detect`` and blocks the later steps. Output rows are the detection frame's
    rows for the targets that reached the end. A result that is not verifiably one input row per returned row (see
    ``_resolve_positions``) fails every target at ``detect``, and ``reroute`` replaces over the detection frame as returned.
    """
    scope = GraphScope()
    graph = ProtectionGraph(scope, tuple(scope.datum(str(text)) for text in dataframe[COL_TEXT]))
    rows: dict[Datum, EntitiesSchema] = {}
    failures: list[FailedRecord] = []
    detected: list[EntityDetectionResult] = []
    owners: list[Datum] = []
    unresolved: list[bool] = []
    elapsed: list[float] = []

    def run_detection(target: tuple[Datum, ...]) -> list[tuple[Mention, ...] | None]:
        started = time.perf_counter()
        detected.append(result := detect(dataframe))
        elapsed.append(time.perf_counter() - started)
        failures.extend(result.failed_records)
        positions = _resolve_positions(dataframe, result)
        if positions is None:
            unresolved.append(True)
            return [None] * len(target)
        owners.extend(target[position] for position in positions)
        raw_entities = result.dataframe.get(COL_FINAL_ENTITIES, repeat({}, len(positions)))
        for datum, raw in zip(owners, raw_entities):
            rows[datum] = EntitiesSchema.from_raw(raw)
        return [tuple(_accepted_mentions(d, rows[d])) if d in rows else None for d in target]

    detect_step = Operation.per_invocation(
        "detect", run_detection, {"target": DATUM}, {"mentions": MENTIONS}, Requests.opaque()
    )
    tracker = ProgressTracker(total=len(graph.targets), label="Replacement")
    map_step = build_map = None
    if isinstance(replace_method, Substitute):
        if llm_workflow is None:
            raise ValueError("Substitute requires an llm_workflow, but none was provided.")
        map_step = _substitute_map_step(
            llm_workflow, replace_method, map_arguments, failures, lambda target: (detected[0].dataframe, owners)
        )
    else:

        def build_map(position: int, entities: EntitiesSchema) -> dict[str, Any]:
            tracker.record_success()
            return _build_local_replacement_map(entities=entities, strategy=replace_method)

    _, result = _protect_rows(
        (graph, rows),
        type(replace_method).__name__.lower(),
        build_map,
        isinstance(replace_method, Annotate),
        map_step,
        detect_step,
    )
    if not detected:  # no targets: the step never ran, but the detector still sees the (empty) input
        run_detection(())
    if unresolved:
        rerouted = reroute(detected[0].dataframe)
        return ReplacementResult(rerouted.dataframe, [*failures, *rerouted.failed_records], detected[0], elapsed[0])
    targets = [d for d in graph.targets if result.outcome(d).available]
    row_of = {datum: row for row, datum in enumerate(owners)}
    output_df = detected[0].dataframe.iloc[[row_of[d] for d in targets]].reset_index(drop=True)
    output_df[COL_REPLACEMENT_MAP] = [result.value(d, "replacement_map") for d in targets]
    if map_step:
        output_df[COL_REPLACEMENT_MAP_SOURCE] = REPLACEMENT_MAP_SOURCE_LLM
    else:
        _log_stats(output_df[COL_REPLACEMENT_MAP])
    shown = output_df[COL_TEXT] if COL_TEXT in output_df.columns else repeat("", len(targets))
    # A row whose detected text is missing keeps the detection frame's own stringification (``<NA>`` for ``pd.NA``).
    output_df[COL_REPLACED_TEXT] = [str(text) if _missing(text) else result.value(d) for d, text in zip(targets, shown)]
    output_df[COL_REPLACEMENT_APPLICATION] = [result.value(d, "application") for d in targets]
    tracker.log_final()
    return ReplacementResult(output_df, failures, detected[0], elapsed[0])


def _missing(text: object) -> bool:
    return not isinstance(text, str) and bool(pd.isna(text))


def _resolve_positions(dataframe: pd.DataFrame, result: EntityDetectionResult) -> tuple[int, ...] | None:
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
    given = [None if _missing(text) else text for text in result.dataframe[COL_TEXT]]
    expected = [None if _missing(text) else text for text in dataframe[COL_TEXT].iloc[list(positions)]]
    return positions if given == expected else None


def _lower(
    dataframe: pd.DataFrame, text_column: str, entities_column: str
) -> tuple[ProtectionGraph, dict[Datum, EntitiesSchema]]:
    """One target per row (``str(text)``, as 0.4.0) with its accepted entity spans as mentions."""
    n = len(dataframe)
    texts = dataframe[text_column] if text_column in dataframe.columns else repeat("", n)
    raw_entities = dataframe[entities_column] if entities_column in dataframe.columns else repeat({}, n)
    scope = GraphScope()
    rows: dict[Datum, EntitiesSchema] = {}
    mentions: list[Mention] = []
    for text, raw in zip(texts, raw_entities):
        datum = scope.datum(str(text))
        rows[datum] = entities = EntitiesSchema.from_raw(raw)
        mentions.extend(_accepted_mentions(datum, entities))
    return ProtectionGraph(scope, tuple(rows), tuple(mentions)), rows


def _protect_rows(
    lowered: tuple[ProtectionGraph, dict[Datum, EntitiesSchema]],
    name: str,
    build_map: Callable[[int, EntitiesSchema], dict[str, Any]] | None,
    annotate: bool,
    map_step: Operation | None = None,
    detect_step: Operation | None = None,
) -> tuple[ProtectionGraph, ProtectionResult]:
    """Run ``replace`` then the patch verifier over every lowered row.

    Each row's map comes from ``map_step``'s ``replacement_map`` port when there is one, else from ``build_map``;
    ``detect_step``, when given, runs first and supplies the mentions and row entities the graph does not declare.
    """
    graph, rows = lowered
    positions = {target: position for position, target in enumerate(graph.targets)}

    def replace(text: str, target: Datum, replacement_map: dict[str, Any] | None = None) -> dict[str, Any]:
        if map_step is None:
            assert build_map is not None, "a workflow without a map step needs build_map"
            rmap = build_map(positions[target], rows[target])
        else:
            rmap = replacement_map
        replaced, application = apply_replacements_to_spans(
            text=text, entities=rows[target], replacements=_parse_replacements(rmap), allow_value_fallback=True
        )
        return {"text": replaced, "replacement_map": rmap, "application": application.to_metrics()}

    replace_inputs = {"text": TEXT, "target": DATUM}
    if map_step is not None:
        replace_inputs["replacement_map"] = REPLACEMENT_MAP
    operations = (
        Operation(
            "replace",
            replace,
            replace_inputs,
            {"text": TEXT, "replacement_map": REPLACEMENT_MAP, "application": REPLACEMENT_APPLICATION},
        ),
        Operation.verifier(
            "verify-patches",
            _verify_patches,
            {
                "text": TEXT,
                "source": TEXT,
                "mentions": MENTIONS,
                "replacement_map": REPLACEMENT_MAP,
                "application": REPLACEMENT_APPLICATION,
                "target": DATUM,
            },
        ),
    )
    operations = (*(s for s in (detect_step, map_step) if s is not None), *operations)
    workflow = Workflow(name, operations, release="annotation" if annotate else "protected")
    # The facade admits the workload it was given, as 0.4.0 did: its limits are sized to the input.
    limits = Limits(
        max_operations=len(operations),
        max_targets=len(graph.targets),
        max_mentions=len(graph.mentions),
        max_context_items=0,
    )
    return graph, _run(graph, workflow, limits)


def _run(graph: ProtectionGraph, workflow: Workflow, limits: Limits) -> ProtectionResult:
    """Execute one invocation; the first error a non-verifier step raised is raised, as 0.4.0 raised it."""
    result = protect(prepare(graph, workflow, limits=limits))
    steps = [step for _, step in workflow.flatten()]
    for entry in result.state.entries:
        if entry.error is not None and not steps[entry.node].is_verifier:
            raise entry.error
    return result


def _accepted_mentions(datum: Datum, entities: EntitiesSchema) -> list[Mention]:
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


def _verify_patches(
    text: str,
    source: str,
    mentions: tuple[Mention, ...],
    replacement_map: dict,
    application: dict,
    target: Datum,
) -> Assessment:
    """Reconstruct source-plus-patches, one patch per accepted mention, and compare with the candidate."""
    patches = {(r.original, r.label): r.synthetic for r in _parse_replacements(replacement_map)}
    parts: list[str] = []
    cursor = 0
    for m in mentions:
        synthetic = patches.get((source[m.start : m.end], m.label))
        if synthetic is None:
            return Assessment(False, target, "unpatched_mention")
        parts += [source[cursor : m.start], synthetic]
        cursor = m.end
    parts.append(source[cursor:])
    satisfied = "".join(parts) == text and application.get("skipped_span_count") == 0
    return Assessment(satisfied, target, "" if satisfied else "reconstruction_mismatch")


def _log_stats(maps: pd.Series) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    total = 0
    for rmap in maps:
        n = len(rmap.get("replacements", []))
        total += n
        logger.debug("  record replacement count: %d", n)
    logger.debug("replacement stats: %d unique entities replaced", total)
