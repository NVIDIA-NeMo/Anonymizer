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
from collections.abc import Callable
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
from anonymizer.engine.replace.llm_replace_workflow import (
    REPLACEMENT_MAP_SOURCE_LLM,
    LlmReplaceResult,
    LlmReplaceWorkflow,
)
from anonymizer.engine.replace.replace_runner import ReplacementResult
from anonymizer.engine.replace.strategies import (
    _build_local_replacement_map,
    _parse_replacements,
    apply_replacements_to_spans,
)
from anonymizer.engine.row_partitioning import resolve_source_positions
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

    graph, result = _protect_rows(
        output_df,
        text_column,
        entities_column,
        type(strategy).__name__.lower(),
        build_map,
        isinstance(strategy, Annotate),
    )
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
    mapped: list[LlmReplaceResult] = []

    def generate_maps(target: tuple[Datum, ...]) -> list[dict[str, Any] | None]:
        result = llm_workflow.generate_map_only(dataframe, instructions=replace_method.instructions, **map_arguments)
        mapped.append(result)
        positions = resolve_source_positions(dataframe, result.dataframe, result.source_positions)
        maps = (
            result.dataframe[COL_REPLACEMENT_MAP]
            if COL_REPLACEMENT_MAP in result.dataframe.columns
            else repeat({}, len(positions))
        )
        by_position = {
            position: {"replacements": [dict(vars(r)) for r in _parse_replacements(rmap)]}
            for position, rmap in zip(positions, maps)
        }
        return [by_position.get(position) for position in range(len(target))]

    map_step = Operation.per_invocation(
        "substitute-map", generate_maps, {"target": DATUM}, {"replacement_map": REPLACEMENT_MAP}, Requests.opaque()
    )
    graph, result = _protect_rows(dataframe, COL_TEXT, COL_FINAL_ENTITIES, "substitute", None, False, map_step)
    kept = [position for position, target in enumerate(graph.targets) if result.outcome(target).available]
    output_df = dataframe.iloc[kept].reset_index(drop=True)
    targets = [graph.targets[position] for position in kept]
    output_df[COL_REPLACEMENT_MAP] = [result.value(d, "replacement_map") for d in targets]
    output_df[COL_REPLACEMENT_MAP_SOURCE] = REPLACEMENT_MAP_SOURCE_LLM
    output_df[COL_REPLACED_TEXT] = [result.value(d) for d in targets]
    output_df[COL_REPLACEMENT_APPLICATION] = [result.value(d, "application") for d in targets]
    return ReplacementResult(dataframe=output_df, failed_records=list(mapped[0].failed_records if mapped else []))


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
    frame: pd.DataFrame,
    text_column: str,
    entities_column: str,
    name: str,
    build_map: Callable[[int, EntitiesSchema], dict[str, Any]] | None,
    annotate: bool,
    map_step: Operation | None = None,
) -> tuple[ProtectionGraph, ProtectionResult]:
    """Run ``replace`` then the patch verifier over every row of ``frame``.

    Each row's map comes from ``map_step``'s ``replacement_map`` port when there is one, else from ``build_map``.
    """
    graph, rows = _lower(frame, text_column, entities_column)
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
    operations = (map_step, *operations) if map_step is not None else operations
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
