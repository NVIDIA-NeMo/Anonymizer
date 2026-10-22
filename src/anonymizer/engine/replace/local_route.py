# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Redact, Annotate and Hash facade routes on the graph kernel.

Each row lowers to one target datum whose exact, non-overlapping entity spans are its mentions. The workflow is a
transform (``replace``: the 0.4.0 per-row replacement) followed by a verifier (``verify-patches``: one patch per
accepted mention, checked by source-plus-patches reconstruction; D.REQ10.s5, RFC :3198-3201). The legacy DataFrame
projection derives from the canonical result without rerunning work (D.REQ8.s5); Annotate output is an annotation
result and is never Protected (D.REQ8.s3).
"""

from __future__ import annotations

import logging
from itertools import repeat
from typing import Any

import pandas as pd

from anonymizer.config.replace_strategies import Annotate, LocalReplaceMethod
from anonymizer.engine.constants import (
    COL_FINAL_ENTITIES,
    COL_REPLACED_TEXT,
    COL_REPLACEMENT_APPLICATION,
    COL_REPLACEMENT_MAP,
    COL_TEXT,
)
from anonymizer.engine.graph_core import (
    Assessment,
    Datum,
    GraphScope,
    Limits,
    Mention,
    Operation,
    ProtectionGraph,
    Workflow,
    prepare,
    protect,
    register_kind,
)
from anonymizer.engine.graph_core.authoring import DATUM, MENTIONS, TEXT
from anonymizer.engine.replace.strategies import (
    _build_local_replacement_map,
    _parse_replacements,
    apply_replacements_to_spans,
)
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
    n = len(output_df)
    tracker = ProgressTracker(total=n, label="Replacement")
    texts = output_df[text_column] if text_column in output_df.columns else repeat("", n)
    raw_entities = output_df[entities_column] if entities_column in output_df.columns else repeat({}, n)
    scope = GraphScope()
    rows: dict[Datum, EntitiesSchema] = {}
    mentions: list[Mention] = []
    for text, raw in zip(texts, raw_entities):
        datum = scope.datum(str(text))
        rows[datum] = entities = EntitiesSchema.from_raw(raw)
        mentions.extend(_accepted_mentions(datum, entities))
    graph = ProtectionGraph(scope, tuple(rows), tuple(mentions))

    def replace(text: str, target: Datum) -> dict[str, Any]:
        rmap = _build_local_replacement_map(entities=rows[target], strategy=strategy)
        replaced, application = apply_replacements_to_spans(
            text=text, entities=rows[target], replacements=_parse_replacements(rmap), allow_value_fallback=True
        )
        tracker.record_success()
        return {"text": replaced, "replacement_map": rmap, "application": application.to_metrics()}

    workflow = Workflow(
        type(strategy).__name__.lower(),
        (
            Operation(
                "replace",
                replace,
                {"text": TEXT, "target": DATUM},
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
        ),
        release="annotation" if isinstance(strategy, Annotate) else "protected",
    )
    # The facade admits the workload it was given, as 0.4.0 did: its limits are sized to the input.
    limits = Limits(max_operations=2, max_targets=n, max_mentions=len(mentions), max_context_items=0)
    result = protect(prepare(graph, workflow, limits=limits))
    for entry in result.state.entries:
        if entry.error is not None and entry.node == 0:
            raise entry.error
    targets = graph.targets
    output_df[COL_REPLACEMENT_MAP] = [result.value(d, "replacement_map") for d in targets]
    _log_stats(output_df[COL_REPLACEMENT_MAP])
    output_df[COL_REPLACED_TEXT] = [result.value(d) for d in targets]
    output_df[COL_REPLACEMENT_APPLICATION] = [result.value(d, "application") for d in targets]
    tracker.log_final()
    return output_df


def _accepted_mentions(datum: Datum, entities: EntitiesSchema) -> list[Mention]:
    """Exact, non-overlapping spans in start order: the output regions the graph owns (D.REQ10.s1, s3)."""
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
