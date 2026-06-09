# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from anonymizer.engine.constants import COL_FINAL_ENTITIES
from anonymizer.measurement._coerce import (
    _coerce_int,
    _count_items,
    _count_text_tokens,
    _safe_row_index,
    _size_bucket,
)
from anonymizer.measurement.metrics.entities import (
    _GROUND_TRUTH_ENTITY_COLUMNS,
    _entities_from_raw,
    _entity_ground_truth_metrics,
)
from anonymizer.measurement.metrics.llm_calls import _validation_chunk_count, estimate_llm_calls_by_stage
from anonymizer.measurement.metrics.replacements import (
    _replacement_collision_metrics,
    _replacement_coverage_metrics,
    _replacement_map_metrics,
)
from anonymizer.measurement.metrics.rewrite import _original_value_leak_record_fields, _rewrite_record_fields
from anonymizer.measurement.session import current_collector

if TYPE_CHECKING:
    import pandas as pd

    from anonymizer.measurement.collector import MeasurementCollector


def record_record_metrics(
    dataframe: pd.DataFrame,
    *,
    mode: str,
    strategy: str,
    text_column: str,
    validation_max_entities_per_call: int,
) -> None:
    """Record per-row count, length, and nominal-call metrics from a trace DataFrame."""
    collector = current_collector()
    if collector is None or not collector.record_level:
        return

    ground_truth_column = next((col for col in _GROUND_TRUTH_ENTITY_COLUMNS if col in dataframe.columns), None)
    columns = set(dataframe.columns)
    for row_index, row in dataframe.iterrows():
        final_entities = _entities_from_raw(row.get(COL_FINAL_ENTITIES))
        collector.record(
            "record",
            **_base_record_fields(
                collector=collector,
                row_index=row_index,
                row=row,
                text_column=text_column,
                mode=mode,
                strategy=strategy,
            ),
            **_entity_record_fields(row, final_entities=final_entities, ground_truth_column=ground_truth_column),
            **_replacement_record_fields(row, columns=columns, final_entities=final_entities),
            **_rewrite_record_fields(row, columns=columns),
            **_original_value_leak_record_fields(row, columns=columns, final_entities=final_entities),
            **_llm_record_fields(
                row,
                columns=columns,
                mode=mode,
                strategy=strategy,
                final_entity_count=len(final_entities),
                validation_max_entities_per_call=validation_max_entities_per_call,
            ),
        )


def _base_record_fields(
    *,
    collector: MeasurementCollector,
    row_index: object,
    row: Any,
    text_column: str,
    mode: str,
    strategy: str,
) -> dict[str, Any]:
    text = str(row.get(text_column, ""))
    text_length_tokens = _count_text_tokens(text)
    return {
        "mode": mode,
        "strategy": strategy,
        "row_index": _safe_row_index(row_index),
        "record_hash": collector.record_hash(row_index=row_index, text=text),
        "text_length_chars": len(text),
        "text_length_chars_bucket": _size_bucket(len(text)),
        "text_length_tokens": text_length_tokens,
        "text_length_tokens_bucket": _size_bucket(text_length_tokens),
    }


def _entity_record_fields(
    row: Any,
    *,
    final_entities: list[dict[str, Any]],
    ground_truth_column: str | None,
) -> dict[str, Any]:
    ground_truth_entities = (
        _entities_from_raw(row.get(ground_truth_column)) if ground_truth_column is not None else None
    )
    return {
        "final_entity_count": len(final_entities),
        "final_entity_label_counts": dict(
            sorted(Counter(e.get("label", "") for e in final_entities if e.get("label")).items())
        ),
        **_entity_ground_truth_metrics(final_entities, ground_truth_entities),
    }


def _replacement_record_fields(
    row: Any,
    *,
    columns: set[str],
    final_entities: list[dict[str, Any]],
) -> dict[str, Any]:
    from anonymizer.engine.constants import COL_REPLACEMENT_MAP

    if COL_REPLACEMENT_MAP not in columns:
        return {}
    raw_map = row.get(COL_REPLACEMENT_MAP)
    return {
        **_replacement_map_metrics(raw_map),
        **_replacement_coverage_metrics(raw_map, final_entities),
        **_replacement_collision_metrics(raw_map, final_entities),
    }


def _llm_record_fields(
    row: Any,
    *,
    columns: set[str],
    mode: str,
    strategy: str,
    final_entity_count: int,
    validation_max_entities_per_call: int,
) -> dict[str, Any]:
    from anonymizer.engine.constants import COL_REPAIR_ITERATIONS

    detected_candidate_count = _detected_candidate_count(row, columns=columns)
    validation_chunk_count = _validation_chunk_count(
        detected_candidate_count,
        validation_max_entities_per_call=validation_max_entities_per_call,
    )
    grouped_entity_count = _grouped_entity_count(row, columns=columns, final_entity_count=final_entity_count)
    repair_iterations = _coerce_int(row.get(COL_REPAIR_ITERATIONS, 0), default=0)
    replace_map_generation_uses_llm = _replace_map_generation_uses_llm(row, columns=columns)
    calls_by_stage = estimate_llm_calls_by_stage(
        mode=mode,
        strategy=strategy,
        has_grouped_entities=grouped_entity_count > 0,
        validation_chunk_count=validation_chunk_count,
        repair_iterations=repair_iterations,
        replace_map_generation_uses_llm=replace_map_generation_uses_llm,
    )
    total_estimated = (
        sum(calls_by_stage.values()) if all(value is not None for value in calls_by_stage.values()) else None
    )
    return {
        "detected_candidate_count": detected_candidate_count,
        "validation_chunk_count": validation_chunk_count,
        "repair_iterations": repair_iterations if mode == "rewrite" else 0,
        "llm_calls_estimated_by_stage": calls_by_stage,
        "llm_calls_estimated_total": total_estimated,
    }


def _replace_map_generation_uses_llm(row: Any, *, columns: set[str]) -> bool:
    del row, columns
    return True


def _detected_candidate_count(row: Any, *, columns: set[str]) -> int | None:
    from anonymizer.engine.constants import COL_SEED_VALIDATION_CANDIDATES

    if COL_SEED_VALIDATION_CANDIDATES not in columns:
        return None
    return _count_items(row.get(COL_SEED_VALIDATION_CANDIDATES), primary_key="candidates", fallback_keys=("entities",))


def _grouped_entity_count(row: Any, *, columns: set[str], final_entity_count: int) -> int:
    from anonymizer.engine.constants import COL_ENTITIES_BY_VALUE

    if COL_ENTITIES_BY_VALUE not in columns:
        return final_entity_count
    return _count_items(row.get(COL_ENTITIES_BY_VALUE), primary_key="entities_by_value", fallback_keys=("entities",))
