# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent symbolic oracle for SDK Phase 9 result compatibility."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations, product
from typing import cast

REFERENCE_VERSION = "phase9-result-compatibility-reference-v1"
GENERATOR_VERSION = "phase9-result-compatibility-pairwise-corpus-v1"

PAIRWISE_DIMENSIONS: dict[str, tuple[str, ...]] = {
    "mode": ("replace", "rewrite"),
    "value_dtype": ("string", "Float64", "boolean"),
    "index_shape": ("range-unnamed", "string-duplicate-named", "multi-duplicate-named"),
    "attrs_shape": ("none", "nested"),
    "collision_history": ("none", "fixed-point"),
    "metadata_token": ("strategy", "entity-labels"),
}

_TEXT = "__nemo_anonymizer_text_input__"
_REPLACED = "__nemo_anonymizer_text_output__"
_TAGGED = "tagged_text"
_REWRITTEN = "_rewritten_text"

_REPLACE_ALLOWED = frozenset(
    {
        "attribute_fidelity_invalid_entities",
        "attribute_fidelity_valid",
        "entity_coverage",
        "final_entities",
        "missed_entities",
        "relational_consistency_invalid_relations",
        "relational_consistency_valid",
        "type_fidelity_invalid_replacements",
        "type_fidelity_valid",
    }
)
_REWRITE_ALLOWED = frozenset(
    {
        "any_high_leaked",
        "entity_coverage",
        "judge_evaluation",
        "leakage_mass",
        "missed_entities",
        "needs_human_review",
        "utility_score",
        "weighted_leakage_rate",
    }
)
_DETECTION_ALLOWED = frozenset({"detection_invalid_entities", "detection_valid"})


def reduce_reference(case: Mapping[str, object]) -> dict[str, object]:
    """Reduce one frozen symbolic case without production or pandas imports."""
    operation = _string(case, "operation")
    if operation == "project":
        return _project(case)
    if operation == "unrename":
        return _unrename(case)
    if operation == "preview":
        return {
            "result_type": "PreviewResult",
            "preview_num_records": _integer(case, "requested"),
            "dataframe_binding": "run.dataframe",
            "trace_binding": "run.trace_dataframe",
            "failures_binding": "run.failed_records",
            "strategy_binding": "config",
            "data_summary_binding": "run.data_summary",
        }
    if operation == "evaluate_failures":
        rewrite = _boolean(case, "rewrite")
        primary = _strings(case, "primary_failures")
        coverage = _strings(case, "coverage_failures")
        failures = primary + coverage if rewrite else primary
        return {
            "result_type": "AnonymizerResult",
            "failures": list(failures),
            "failure_container": "new" if rewrite else "primary",
            "prior_failures_retained": False,
        }
    if operation == "pickle":
        result_type = _string(case, "result_type")
        if result_type == "AnonymizerResult":
            return {
                "module": "anonymizer.interface.results",
                "fields": [
                    "dataframe",
                    "trace_dataframe",
                    "resolved_text_column",
                    "failed_records",
                    "replace_method",
                    "rewrite_config",
                    "entity_labels",
                    "data_summary",
                    "_display_cycle_index",
                ],
            }
        if result_type == "PreviewResult":
            return {
                "module": "anonymizer.interface.results",
                "fields": [
                    "dataframe",
                    "trace_dataframe",
                    "resolved_text_column",
                    "failed_records",
                    "preview_num_records",
                    "replace_method",
                    "rewrite_config",
                    "entity_labels",
                    "data_summary",
                    "_display_cycle_index",
                ],
            }
        if result_type == "FailedRecord":
            return {
                "module": "anonymizer.engine.ndd.adapter",
                "fields": ["record_id", "step", "reason"],
            }
        raise ValueError(f"unknown pickle result type: {result_type!r}")
    if operation == "cli":
        command = _string(case, "command")
        if command == "run":
            return {
                "serialized_frame": "result.dataframe",
                "index": False,
                "degraded_success_exit": 0,
                "failure_details_printed": False,
            }
        if command == "preview":
            return {
                "render": "result.dataframe.to_string(max_colwidth=80)",
                "failure_details_printed": False,
            }
        raise ValueError(f"unknown CLI command: {command!r}")
    if operation == "telemetry":
        input_count = _integer(case, "input_count")
        failures = _strings(case, "failures")
        failure_count = len(failures)
        return {
            "status": "completed",
            "failure_count": failure_count,
            "success_count": max(input_count - failure_count, 0),
            "deduplicate_failures": False,
        }
    if operation == "exception":
        return {
            "type": "AnonymizerWorkflowError",
            "message": "Anonymization pipeline failed.",
            "cause": None,
            "partial_result": False,
        }
    if operation == "platform":
        return {
            "run_frames": ["result.dataframe", "result.trace_dataframe"],
            "run_format": "parquet-index-false",
            "metadata_key": "original_text_column",
            "failure_fields": ["record_id", "step", "reason"],
            "failure_order": "result.failed_records",
            "preview_format": "json-records",
            "claim_status": "pinned-source-only",
        }
    if operation == "graph":
        return {"admission": "rejected", "public_projection": None}
    if operation == "pandas_bridge":
        return _pandas_bridge(case)
    raise ValueError(f"unknown reference operation: {operation!r}")


def generate_pairwise_core() -> list[dict[str, object]]:
    """Generate a deterministic finite all-pairs core from literal dimensions."""
    names = tuple(PAIRWISE_DIMENSIONS)
    candidates = list(product(*(PAIRWISE_DIMENSIONS[name] for name in names)))
    uncovered = {
        (left, right, candidate[left], candidate[right])
        for candidate in candidates
        for left, right in combinations(range(len(names)), 2)
    }
    selected: list[tuple[str, ...]] = []
    while uncovered:
        candidate = max(
            candidates,
            key=lambda values: sum(
                (left, right, values[left], values[right]) in uncovered
                for left, right in combinations(range(len(names)), 2)
            ),
        )
        selected.append(candidate)
        for left, right in combinations(range(len(names)), 2):
            uncovered.discard((left, right, candidate[left], candidate[right]))
        candidates.remove(candidate)
    return [
        {
            "name": f"pairwise-{position:02d}",
            "operation": "pandas_bridge",
            **dict(zip(names, values, strict=True)),
        }
        for position, values in enumerate(selected, start=1)
    ]


def _project(case: Mapping[str, object]) -> dict[str, object]:
    columns = _strings(case, "columns")
    resolved = _string(case, "resolved_text_column")
    compute_detection_validity = _boolean(case, "compute_detection_validity")
    rename = {
        _TEXT: resolved,
        _REPLACED: f"{resolved}_replaced",
        _TAGGED: f"{resolved}_with_spans",
        _REWRITTEN: f"{resolved}_rewritten",
    }
    trace = tuple(rename.get(column, column) for column in columns)
    if f"{resolved}_rewritten" in trace:
        allowed = _REWRITE_ALLOWED | {resolved, f"{resolved}_rewritten"}
    elif f"{resolved}_replaced" in trace:
        allowed = _REPLACE_ALLOWED | {
            resolved,
            f"{resolved}_replaced",
            f"{resolved}_with_spans",
        }
    else:
        allowed = {resolved, f"{resolved}_with_spans", "final_entities"}
    if compute_detection_validity:
        allowed |= _DETECTION_ALLOWED
    selected_labels = tuple(column for column in trace if column in allowed)
    # pandas expands every duplicate label for every repeated selector label.
    public = tuple(column for selected in selected_labels for column in trace if column == selected)
    return {
        "trace_columns": list(trace),
        "public_columns": list(public),
        "forward_returns_same_object": not any(column in rename for column in columns),
    }


def _unrename(case: Mapping[str, object]) -> dict[str, object]:
    columns = _strings(case, "columns")
    resolved = _string(case, "resolved_text_column")
    if _TEXT in columns:
        return {"columns": list(columns), "returns_same_object": True}
    rename = {
        resolved: _TEXT,
        f"{resolved}_replaced": _REPLACED,
        f"{resolved}_with_spans": _TAGGED,
        f"{resolved}_rewritten": _REWRITTEN,
    }
    changed = any(column in rename for column in columns)
    return {
        "columns": [rename.get(column, column) for column in columns],
        "returns_same_object": not changed,
    }


def _pandas_bridge(case: Mapping[str, object]) -> dict[str, object]:
    mode = _string(case, "mode")
    collision_history = _string(case, "collision_history")
    resolved = "body" if collision_history == "none" else "final_entities__input_3"
    if mode == "replace":
        columns = [_TEXT, _REPLACED, "final_entities"]
        value_column = f"{resolved}_replaced"
        nested_column = "final_entities"
    elif mode == "rewrite":
        columns = [_TEXT, _REWRITTEN, "utility_score", "missed_entities"]
        value_column = f"{resolved}_rewritten"
        nested_column = "missed_entities"
    else:
        raise ValueError(f"unknown pandas bridge mode: {mode!r}")
    projected = _project(
        {
            "columns": columns,
            "resolved_text_column": resolved,
            "compute_detection_validity": False,
        }
    )
    index_shape = _string(case, "index_shape")
    if index_shape == "range-unnamed":
        index_values: list[object] = [0, 1, 2]
        index_name: object = None
    elif index_shape == "string-duplicate-named":
        index_values = ["b", "a", "b"]
        index_name = "source-row"
    elif index_shape == "multi-duplicate-named":
        index_values = [["a", 2], ["a", 1], ["a", 2]]
        index_name = ["group", "position"]
    else:
        raise ValueError(f"unknown pandas bridge index shape: {index_shape!r}")
    attrs_shape = _string(case, "attrs_shape")
    attrs = {} if attrs_shape == "none" else {"dataset": {"kind": "pairwise"}}
    return {
        "trace_columns": projected["trace_columns"],
        "public_columns": projected["public_columns"],
        "resolved_text_column": resolved,
        "value_column": value_column,
        "nested_column": nested_column,
        "value_dtype": _string(case, "value_dtype"),
        "index_shape": index_shape,
        "index_values": index_values,
        "index_name": index_name,
        "column_index_name": "pipeline-column",
        "column_index_dtype": "object",
        "attrs": attrs,
        "attrs_binding": "equal-not-identical",
        "nested_cell_binding": "shared",
        "metadata_binding": _string(case, "metadata_token"),
        "collision_history": collision_history,
    }


def _string(case: Mapping[str, object], key: str) -> str:
    value = case.get(key)
    if type(value) is not str:
        raise TypeError(f"{key} must be a string")
    return value


def _strings(case: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = case.get(key)
    if type(value) is not list or not all(type(item) is str for item in value):
        raise TypeError(f"{key} must be a list of strings")
    return tuple(cast(list[str], value))


def _integer(case: Mapping[str, object], key: str) -> int:
    value = case.get(key)
    if type(value) is not int:
        raise TypeError(f"{key} must be an integer")
    return value


def _boolean(case: Mapping[str, object], key: str) -> bool:
    value = case.get(key)
    if type(value) is not bool:
        raise TypeError(f"{key} must be a boolean")
    return value
