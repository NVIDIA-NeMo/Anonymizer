# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from anonymizer.measurement._coerce import _coerce_int
from anonymizer.measurement.records.model import _model_workflow_fields, _row_throughput_fields, _summarize_model_usage
from anonymizer.measurement.records.run import (
    _detect_config_metadata,
    _model_config_metadata,
    _replace_config_metadata,
    _rewrite_config_metadata,
    _runtime_metadata,
    _source_metadata,
)
from anonymizer.measurement.session import current_collector


@contextmanager
def stage_timer(stage: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Record wall time for a stage when collection is active."""
    collector = current_collector()
    if collector is None:
        yield fields
        return

    started = time.perf_counter()
    status = "completed"
    try:
        yield fields
    except BaseException:
        status = "error"
        raise
    finally:
        elapsed_sec = time.perf_counter() - started
        collector.record(
            "stage",
            stage=stage,
            status=status,
            elapsed_sec=elapsed_sec,
            **fields,
            **_row_throughput_fields(
                elapsed_sec=elapsed_sec,
                input_row_count=_coerce_int(fields.get("input_row_count"), default=-1),
                output_row_count=_coerce_int(fields.get("output_row_count"), default=-1),
            ),
        )


def record_stage(stage: str, *, elapsed_sec: float, status: str = "completed", **fields: Any) -> None:
    """Record a pre-timed stage measurement if collection is active."""
    collector = current_collector()
    if collector is None:
        return
    collector.record(
        "stage",
        stage=stage,
        status=status,
        elapsed_sec=elapsed_sec,
        **fields,
        **_row_throughput_fields(
            elapsed_sec=elapsed_sec,
            input_row_count=_coerce_int(fields.get("input_row_count"), default=-1),
            output_row_count=_coerce_int(fields.get("output_row_count"), default=-1),
        ),
    )


def record_ndd_workflow(
    *,
    workflow_name: str,
    model_aliases: list[str],
    input_row_count: int,
    output_row_count: int | None,
    failed_record_count: int | None,
    elapsed_sec: float,
    status: str = "completed",
    seed_row_count: int | None = None,
    preview_num_records: int | None = None,
    column_count: int | None = None,
    column_names: list[str] | None = None,
    model_usage: Mapping[str, Any] | None = None,
) -> None:
    """Record one DataDesigner workflow execution through the adapter boundary."""
    _record_model_workflow(
        workflow_name=workflow_name,
        model_aliases=model_aliases,
        input_row_count=input_row_count,
        output_row_count=output_row_count,
        failed_record_count=failed_record_count,
        elapsed_sec=elapsed_sec,
        status=status,
        seed_row_count=seed_row_count,
        preview_num_records=preview_num_records,
        column_count=column_count,
        column_names=column_names,
        model_usage=model_usage,
        record_type="ndd_workflow",
        extra_fields=None,
    )


def record_model_workflow(
    *,
    workflow_name: str,
    model_aliases: list[str],
    input_row_count: int,
    output_row_count: int | None,
    failed_record_count: int | None,
    elapsed_sec: float,
    status: str = "completed",
    seed_row_count: int | None = None,
    preview_num_records: int | None = None,
    column_count: int | None = None,
    column_names: list[str] | None = None,
    model_usage: Mapping[str, Any] | None = None,
    extra_fields: Mapping[str, Any] | None = None,
) -> None:
    """Record one sanitized model-backed workflow execution.

    Use this for non-DataDesigner model calls that still need benchmark
    accounting. Raw prompts, text, responses, and replacement values do not
    belong in ``model_usage``.
    """
    _record_model_workflow(
        workflow_name=workflow_name,
        model_aliases=model_aliases,
        input_row_count=input_row_count,
        output_row_count=output_row_count,
        failed_record_count=failed_record_count,
        elapsed_sec=elapsed_sec,
        status=status,
        seed_row_count=seed_row_count,
        preview_num_records=preview_num_records,
        column_count=column_count,
        column_names=column_names,
        model_usage=model_usage,
        record_type="model_workflow",
        extra_fields=extra_fields,
    )


def _record_model_workflow(
    *,
    workflow_name: str,
    model_aliases: list[str],
    input_row_count: int,
    output_row_count: int | None,
    failed_record_count: int | None,
    elapsed_sec: float,
    status: str,
    seed_row_count: int | None,
    preview_num_records: int | None,
    column_count: int | None,
    column_names: list[str] | None,
    model_usage: Mapping[str, Any] | None,
    record_type: str,
    extra_fields: Mapping[str, Any] | None,
) -> None:
    collector = current_collector()
    if collector is None:
        return
    observed_usage = _summarize_model_usage(model_usage)
    workflow_fields = {
        "workflow_name": workflow_name,
        "status": status,
        "model_aliases": sorted(set(model_aliases)),
        "input_row_count": input_row_count,
        "seed_row_count": seed_row_count,
        "output_row_count": output_row_count,
        "failed_record_count": failed_record_count,
        "elapsed_sec": elapsed_sec,
        "preview_num_records": preview_num_records,
        "column_count": column_count,
        "column_names": column_names or [],
        "model_usage": dict(model_usage or {}),
        **dict(extra_fields or {}),
    }
    collector.record(record_type, **_model_workflow_fields(workflow_fields, observed_usage))


def record_run_metadata(
    *,
    config: Any,
    data: Any,
    mode: str,
    strategy: str,
    input_row_count: int,
    preview_num_records: int | None,
    model_configs: list[Any],
) -> None:
    """Record sanitized run/config metadata once per anonymizer run."""
    collector = current_collector()
    if collector is None:
        return

    detect = getattr(config, "detect", None)
    source = str(getattr(data, "source", ""))
    collector.record(
        "run",
        mode=mode,
        strategy=strategy,
        input_row_count=input_row_count,
        preview_num_records=preview_num_records,
        source_hash=collector.record_hash(row_index="source", text=source),
        input_source=_source_metadata(source),
        input_text_column=str(getattr(data, "text_column", "")),
        input_has_id_column=bool(getattr(data, "id_column", None)),
        input_has_data_summary=bool(getattr(data, "data_summary", None)),
        detect=_detect_config_metadata(detect),
        replace=_replace_config_metadata(getattr(config, "replace", None)),
        rewrite=_rewrite_config_metadata(getattr(config, "rewrite", None)),
        models=[_model_config_metadata(model_config) for model_config in model_configs],
        runtime=_runtime_metadata(),
    )
