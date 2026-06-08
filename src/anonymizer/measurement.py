# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import platform
import secrets
import time
import uuid
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast
from urllib.parse import urlparse

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict, SettingsError

from anonymizer.engine.constants import COL_FINAL_ENTITIES

if TYPE_CHECKING:
    import pandas as pd


_ACTIVE_COLLECTOR: ContextVar[MeasurementCollector | None] = ContextVar(
    "anonymizer_measurement_collector",
    default=None,
)
_GROUND_TRUTH_ENTITY_COLUMNS = ("ground_truth_entities", "gt_entities", "expected_entities")
MEASUREMENT_SCHEMA_VERSION = 1
DEFAULT_MEASUREMENT_ENV_PREFIX = "ANONYMIZER_MEASUREMENT_"
DD_TRACE_MODES = {"none", "last_message", "all_messages"}
DDTraceMode = Literal["none", "last_message", "all_messages"]

logger = logging.getLogger("anonymizer.measurement")


class _MeasurementWriter(Protocol):
    def write(self, records: list[dict[str, Any]], path: str | Path) -> None: ...


class _MeasurementSink(Protocol):
    def write_record(self, record: dict[str, Any]) -> None: ...

    def close(self) -> None: ...


class _JsonlMeasurementWriter:
    def write(self, records: list[dict[str, Any]], path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


class _JsonMeasurementWriter:
    def write(self, records: list[dict[str, Any]], path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=True, indent=2, sort_keys=True)


def _writer_for_format(output_format: Literal["jsonl", "json"]) -> _MeasurementWriter:
    if output_format == "json":
        return _JsonMeasurementWriter()
    return _JsonlMeasurementWriter()


class _JsonlMeasurementSink:
    def __init__(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = output_path.open("w", encoding="utf-8", buffering=1)

    def write_record(self, record: dict[str, Any]) -> None:
        self._file.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")

    def close(self) -> None:
        self._file.close()


class _MeasurementEnvSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix=DEFAULT_MEASUREMENT_ENV_PREFIX,
        env_ignore_empty=True,
        extra="ignore",
    )

    output_path: str | None = None
    output_format: Literal["jsonl", "json"] = "jsonl"
    record_level: bool = True
    streaming: bool = False
    keep_records: bool = True
    dd_trace: DDTraceMode = "none"
    dd_trace_path: str | None = None
    fail_on_write_error: bool = False
    run_id: str | None = None
    run_tags: dict[str, Any] = Field(default_factory=dict)


class MeasurementCollector:
    """In-memory collector for local benchmark and throughput records.

    Records contain counts, labels, lengths, aliases, timings, and run-scoped
    HMACs. They must not contain raw text, entity values, prompts, generated
    outputs, replacement maps, provider secrets, or API keys.
    """

    def __init__(
        self,
        *,
        run_id: str | None = None,
        record_hash_key: bytes | str | None = None,
        record_level: bool = True,
        run_tags: Mapping[str, Any] | None = None,
        record_sink: _MeasurementSink | None = None,
        keep_records: bool = True,
        dd_trace_mode: DDTraceMode = "none",
        dd_trace_sink: _MeasurementSink | None = None,
        fail_on_write_error: bool = False,
    ) -> None:
        self.run_id = run_id or uuid.uuid4().hex
        self.record_level = record_level
        self.run_tags = cast(dict[str, Any], _json_safe(dict(run_tags or {})))
        self._record_sink = record_sink
        self._keep_records = keep_records
        self._dd_trace_mode = dd_trace_mode
        self._dd_trace_sink = dd_trace_sink
        self._fail_on_write_error = fail_on_write_error
        self._sink_failed = False
        self._dd_trace_failed = False
        if record_hash_key is None:
            self._record_hash_key = secrets.token_bytes(32)
        elif isinstance(record_hash_key, str):
            self._record_hash_key = record_hash_key.encode("utf-8")
        else:
            self._record_hash_key = bytes(record_hash_key)
        self._records: list[dict[str, Any]] = []

    @property
    def records(self) -> list[dict[str, Any]]:
        """Return a shallow copy of collected measurement records."""
        return list(self._records)

    def record(self, record_type: str, **fields: Any) -> None:
        """Append one machine-readable measurement record."""
        record = {
            **fields,
            "schema_version": MEASUREMENT_SCHEMA_VERSION,
            "record_type": record_type,
            "run_id": self.run_id,
            "run_tags": self.run_tags,
            "timestamp_unix_sec": time.time(),
        }
        safe_record = _json_safe(record)
        if self._keep_records:
            self._records.append(safe_record)
        if self._record_sink is not None:
            self._write_record_to_sink(safe_record)

    def close(self) -> None:
        """Close any streaming measurement sink attached to this collector."""
        if self._record_sink is not None:
            self._record_sink.close()
        if self._dd_trace_sink is not None:
            self._dd_trace_sink.close()

    @property
    def dd_trace_mode(self) -> DDTraceMode:
        return self._dd_trace_mode

    @property
    def dd_trace_enabled(self) -> bool:
        return self._dd_trace_mode != "none" and self._dd_trace_sink is not None

    def record_dd_message_trace(self, **fields: Any) -> None:
        """Write an explicitly opt-in DataDesigner message trace record.

        These records may contain raw prompts, input text, model outputs, and
        PII. They are intentionally written to a separate trace sink and are
        never appended to the safe measurement record list.
        """
        if not self.dd_trace_enabled or self._dd_trace_failed:
            return

        record = _json_safe(
            {
                **fields,
                "schema_version": MEASUREMENT_SCHEMA_VERSION,
                "record_type": "dd_message_trace",
                "run_id": self.run_id,
                "run_tags": self.run_tags,
                "timestamp_unix_sec": time.time(),
            }
        )
        try:
            cast(_MeasurementSink, self._dd_trace_sink).write_record(record)
        except Exception:
            self._dd_trace_failed = True
            logger.warning("Failed to write DataDesigner message trace records")
            if self._fail_on_write_error:
                raise

    def _write_record_to_sink(self, record: dict[str, Any]) -> None:
        if self._sink_failed:
            return
        try:
            cast(_MeasurementSink, self._record_sink).write_record(record)
        except Exception:
            self._sink_failed = True
            logger.warning("Failed to stream Anonymizer measurement records")
            if self._fail_on_write_error:
                raise

    def record_hash(self, *, row_index: object, text: str) -> str:
        """Return a run-scoped HMAC for joining records without storing text."""
        serialized = json.dumps(
            {"row_index": str(row_index), "text": text},
            default=str,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hmac.new(self._record_hash_key, serialized.encode("utf-8"), hashlib.sha256).hexdigest()

    def write_jsonl(self, path: str | Path) -> None:
        """Write records as newline-delimited JSON."""
        _JsonlMeasurementWriter().write(self._records, path)

    def write_json(self, path: str | Path) -> None:
        """Write records as a JSON array."""
        _JsonMeasurementWriter().write(self._records, path)

    def to_dataframe(self) -> pd.DataFrame:
        """Return records as a pandas DataFrame for benchmark tooling."""
        import pandas as pd

        return pd.DataFrame(self._records)


@dataclass(frozen=True)
class MeasurementConfig:
    """Configuration for writing structured measurement records around a run."""

    output_path: str | Path
    output_format: Literal["jsonl", "json"] = "jsonl"
    record_level: bool = True
    streaming: bool = False
    keep_records: bool = True
    dd_trace: DDTraceMode = "none"
    dd_trace_path: str | Path | None = None
    run_id: str | None = None
    record_hash_key: bytes | str | None = None
    run_tags: Mapping[str, Any] | None = None
    fail_on_write_error: bool = False

    def __post_init__(self) -> None:
        if self.output_format not in {"jsonl", "json"}:
            raise ValueError("output_format must be 'jsonl' or 'json'")
        if self.streaming and self.output_format != "jsonl":
            raise ValueError("streaming measurement output only supports jsonl")
        if self.dd_trace not in DD_TRACE_MODES:
            raise ValueError("dd_trace must be 'none', 'last_message', or 'all_messages'")
        if self.dd_trace != "none" and self.dd_trace_path is None:
            raise ValueError("dd_trace_path is required when dd_trace is enabled")

    @classmethod
    def from_env(cls, *, prefix: str = DEFAULT_MEASUREMENT_ENV_PREFIX) -> MeasurementConfig | None:
        """Build measurement config from environment variables, or None if output is unset.

        This is intentionally opt-in. Anonymizer API and CLI calls do not read
        measurement environment variables unless benchmark/tooling code calls this
        helper explicitly.
        """
        try:
            settings = _load_measurement_env_settings(prefix=prefix)
        except (SettingsError, ValidationError) as exc:
            raise ValueError(_measurement_env_error_message(exc, prefix=prefix)) from None

        if settings.output_path is None:
            return None
        return cls(
            output_path=settings.output_path,
            output_format=settings.output_format,
            record_level=settings.record_level,
            streaming=settings.streaming,
            keep_records=settings.keep_records,
            dd_trace=settings.dd_trace,
            dd_trace_path=settings.dd_trace_path,
            run_id=settings.run_id,
            run_tags=settings.run_tags,
            fail_on_write_error=settings.fail_on_write_error,
        )

    @classmethod
    def from_sources(
        cls,
        explicit: MeasurementConfig | None = None,
        *,
        env: bool = False,
        prefix: str = DEFAULT_MEASUREMENT_ENV_PREFIX,
    ) -> MeasurementConfig | None:
        """Resolve measurement config from explicit config first, then optional env."""
        if explicit is not None:
            return explicit
        if env:
            return cls.from_env(prefix=prefix)
        return None

    def write_collector(self, collector: MeasurementCollector) -> None:
        """Write a collector using this config's output format."""
        _writer_for_format(self.output_format).write(collector.records, self.output_path)


@contextmanager
def measurement_session(collector: MeasurementCollector | None = None) -> Iterator[MeasurementCollector]:
    """Activate a collector for code running in this context."""
    active = collector or MeasurementCollector()
    token = _ACTIVE_COLLECTOR.set(active)
    try:
        yield active
    finally:
        _ACTIVE_COLLECTOR.reset(token)


@contextmanager
def configured_measurement_session(config: MeasurementConfig | None) -> Iterator[MeasurementCollector | None]:
    """Activate and persist a collector when a measurement config is provided."""
    if config is None:
        yield None
        return

    sink = _JsonlMeasurementSink(config.output_path) if config.streaming else None
    dd_trace_sink = _JsonlMeasurementSink(config.dd_trace_path) if config.dd_trace != "none" else None
    collector = MeasurementCollector(
        run_id=config.run_id,
        record_hash_key=config.record_hash_key,
        record_level=config.record_level,
        run_tags=config.run_tags,
        record_sink=sink,
        keep_records=config.keep_records,
        dd_trace_mode=config.dd_trace,
        dd_trace_sink=dd_trace_sink,
        fail_on_write_error=config.fail_on_write_error,
    )
    with measurement_session(collector):
        body_error: BaseException | None = None
        try:
            yield collector
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            if config.streaming:
                _close_collector_safely(config=config, collector=collector, body_error=body_error)
            else:
                _write_collector_safely(config=config, collector=collector, body_error=body_error)
                _close_collector_safely(config=config, collector=collector, body_error=body_error)


def current_collector() -> MeasurementCollector | None:
    """Return the active collector, if measurement is enabled."""
    return _ACTIVE_COLLECTOR.get()


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


def _model_workflow_fields(fields: dict[str, Any], observed_usage: dict[str, int | None]) -> dict[str, Any]:
    return {
        **fields,
        **observed_usage,
        "observed_failed_request_rate": _safe_ratio(
            observed_usage["observed_failed_requests"],
            observed_usage["observed_total_requests"],
        ),
        **_throughput_fields(
            elapsed_sec=cast(float, fields["elapsed_sec"]),
            input_row_count=cast(int, fields["input_row_count"]),
            output_row_count=cast(int | None, fields["output_row_count"]),
            total_tokens=observed_usage["observed_total_tokens"],
            total_requests=observed_usage["observed_total_requests"],
            successful_requests=observed_usage["observed_successful_requests"],
        ),
    }


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


def _detect_config_metadata(detect: Any | None) -> dict[str, Any]:
    entity_labels = getattr(detect, "entity_labels", None)
    if entity_labels is None:
        from anonymizer.engine.constants import DEFAULT_ENTITY_LABELS

        entity_label_count = len(DEFAULT_ENTITY_LABELS)
    else:
        entity_label_count = len(entity_labels)
    return {
        "gliner_threshold": getattr(detect, "gliner_threshold", None),
        "entity_label_source": "custom" if entity_labels is not None else "default",
        "entity_label_count": entity_label_count,
        "entity_labels": list(entity_labels) if entity_labels is not None else None,
        "validation_max_entities_per_call": getattr(detect, "validation_max_entities_per_call", None),
        "validation_excerpt_window_chars": getattr(detect, "validation_excerpt_window_chars", None),
    }


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


def _rewrite_record_fields(row: Any, *, columns: set[str]) -> dict[str, Any]:
    from anonymizer.engine.constants import (
        COL_ANY_HIGH_LEAKED,
        COL_LEAKAGE_MASS,
        COL_NEEDS_HUMAN_REVIEW,
        COL_NEEDS_REPAIR,
        COL_UTILITY_SCORE,
        COL_WEIGHTED_LEAKAGE_RATE,
    )

    return {
        "utility_score": _coerce_float(row.get(COL_UTILITY_SCORE)) if COL_UTILITY_SCORE in columns else None,
        "leakage_mass": _coerce_float(row.get(COL_LEAKAGE_MASS)) if COL_LEAKAGE_MASS in columns else None,
        "weighted_leakage_rate": (
            _coerce_float(row.get(COL_WEIGHTED_LEAKAGE_RATE)) if COL_WEIGHTED_LEAKAGE_RATE in columns else None
        ),
        "any_high_leaked": _coerce_bool(row.get(COL_ANY_HIGH_LEAKED)) if COL_ANY_HIGH_LEAKED in columns else None,
        "needs_human_review": (
            _coerce_bool(row.get(COL_NEEDS_HUMAN_REVIEW)) if COL_NEEDS_HUMAN_REVIEW in columns else None
        ),
        "needs_repair": _coerce_bool(row.get(COL_NEEDS_REPAIR)) if COL_NEEDS_REPAIR in columns else None,
    }


def _original_value_leak_record_fields(
    row: Any,
    *,
    columns: set[str],
    final_entities: list[dict[str, Any]],
) -> dict[str, Any]:
    output_column = _output_text_column(columns)
    if output_column is None:
        return {"original_value_leak_count": None, "original_value_leak_label_counts": {}}
    output_text = str(row.get(output_column, ""))
    leaked = [
        entity
        for entity in final_entities
        if entity.get("value") and _output_contains_original_value(output_text, str(entity.get("value")))
    ]
    return {
        "original_value_leak_count": len(leaked),
        "original_value_leak_label_counts": dict(
            sorted(Counter(str(entity.get("label") or "") for entity in leaked if entity.get("label")).items())
        ),
    }


def _output_contains_original_value(output_text: str, value: str) -> bool:
    if _needs_boundary_sensitive_leak_match(value):
        return _contains_with_alnum_boundaries(output_text, value)
    return value in output_text


def _needs_boundary_sensitive_leak_match(value: str) -> bool:
    return len(value) <= 4 or value.isdigit()


def _contains_with_alnum_boundaries(output_text: str, value: str) -> bool:
    start = 0
    while True:
        match_start = output_text.find(value, start)
        if match_start < 0:
            return False
        match_end = match_start + len(value)
        if _has_alnum_boundaries(output_text, match_start, match_end):
            return True
        start = match_start + 1


def _has_alnum_boundaries(text: str, start: int, end: int) -> bool:
    before_is_alnum = start > 0 and text[start - 1].isalnum()
    after_is_alnum = end < len(text) and text[end].isalnum()
    return not before_is_alnum and not after_is_alnum


def _output_text_column(columns: set[str]) -> str | None:
    from anonymizer.engine.constants import COL_REPLACED_TEXT, COL_REWRITTEN_TEXT

    if COL_REPLACED_TEXT in columns:
        return COL_REPLACED_TEXT
    if COL_REWRITTEN_TEXT in columns:
        return COL_REWRITTEN_TEXT
    return None


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
    from anonymizer.engine.constants import COL_REPLACEMENT_MAP_SOURCE
    from anonymizer.engine.replace.structured_substitute import REPLACEMENT_MAP_SOURCE_LOCAL_STRUCTURED

    if COL_REPLACEMENT_MAP_SOURCE not in columns:
        return True
    return row.get(COL_REPLACEMENT_MAP_SOURCE) != REPLACEMENT_MAP_SOURCE_LOCAL_STRUCTURED


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


def _source_metadata(source: str) -> dict[str, Any]:
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        return {
            "kind": "remote_file",
            "scheme": parsed.scheme,
            "suffix": Path(parsed.path).suffix.lower() or None,
        }
    if parsed.scheme == "file":
        return {
            "kind": "local_file",
            "scheme": "file",
            "suffix": Path(parsed.path).suffix.lower() or None,
        }
    return {
        "kind": "local_file" if source else "unknown",
        "scheme": None,
        "suffix": Path(source).suffix.lower() or None,
    }


def _replace_config_metadata(replace_config: Any | None) -> dict[str, Any] | None:
    if replace_config is None:
        return None

    metadata: dict[str, Any] = {
        "strategy": type(replace_config).__name__,
        "has_instructions": bool(getattr(replace_config, "instructions", None)),
    }
    for attr in ("normalize_label", "algorithm", "digest_length"):
        if hasattr(replace_config, attr):
            metadata[attr] = getattr(replace_config, attr)
    if hasattr(replace_config, "format_template"):
        metadata["has_format_template"] = True
    return metadata


def _rewrite_config_metadata(rewrite_config: Any | None) -> dict[str, Any] | None:
    if rewrite_config is None:
        return None
    return {
        "risk_tolerance": _enum_value(getattr(rewrite_config, "risk_tolerance", None)),
        "max_repair_iterations": getattr(rewrite_config, "max_repair_iterations", None),
        "strict_entity_protection": getattr(rewrite_config, "strict_entity_protection", None),
        "has_privacy_goal": bool(getattr(rewrite_config, "privacy_goal", None)),
        "has_instructions": bool(getattr(rewrite_config, "instructions", None)),
    }


def _model_config_metadata(model_config: Any) -> dict[str, Any]:
    inference_parameters = getattr(model_config, "inference_parameters", None)
    return {
        "alias": getattr(model_config, "alias", None),
        "model": getattr(model_config, "model", None),
        "provider": _enum_value(getattr(model_config, "provider", None)),
        "base_url": bool(getattr(model_config, "base_url", None)),
        "max_parallel_requests": getattr(inference_parameters, "max_parallel_requests", None),
    }


def _runtime_metadata() -> dict[str, Any]:
    try:
        anonymizer_version = version("nemo-anonymizer")
    except PackageNotFoundError:
        anonymizer_version = None
    return {
        "anonymizer_version": anonymizer_version,
        "measurement_schema_version": MEASUREMENT_SCHEMA_VERSION,
        "platform_machine": platform.machine(),
        "platform_system": platform.system(),
        "python_version": platform.python_version(),
    }


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _throughput_fields(
    *,
    elapsed_sec: float,
    input_row_count: int | None,
    output_row_count: int | None,
    total_tokens: int | None,
    total_requests: int | None,
    successful_requests: int | None,
) -> dict[str, float | None]:
    return {
        "input_rows_per_sec": _safe_rate(input_row_count, elapsed_sec),
        "output_rows_per_sec": _safe_rate(output_row_count, elapsed_sec),
        "observed_tokens_per_sec": _safe_rate(total_tokens, elapsed_sec),
        "observed_requests_per_sec": _safe_rate(total_requests, elapsed_sec),
        "observed_tokens_per_successful_request": _safe_ratio(total_tokens, successful_requests),
    }


def _row_throughput_fields(
    *,
    elapsed_sec: float,
    input_row_count: int | None,
    output_row_count: int | None,
) -> dict[str, float | None]:
    if input_row_count is not None and input_row_count < 0:
        input_row_count = None
    if output_row_count is not None and output_row_count < 0:
        output_row_count = None
    return {
        "input_rows_per_sec": _safe_rate(input_row_count, elapsed_sec),
        "output_rows_per_sec": _safe_rate(output_row_count, elapsed_sec),
    }


def estimate_llm_calls_by_stage(
    *,
    mode: str,
    strategy: str,
    has_grouped_entities: bool,
    validation_chunk_count: int | None,
    repair_iterations: int = 0,
    replace_map_generation_uses_llm: bool = True,
) -> dict[str, int | None]:
    """Estimate nominal model calls for one record, split by workflow stage."""
    detection_calls = None if validation_chunk_count is None else 2 + validation_chunk_count
    replace_map_generation = 0
    if replace_map_generation_uses_llm and has_grouped_entities and (mode == "rewrite" or strategy == "Substitute"):
        replace_map_generation = 1

    if mode != "rewrite":
        return {
            "entity_detection": detection_calls,
            "replace_map_generation": replace_map_generation,
        }

    rewrite_body_calls = has_grouped_entities
    return {
        "entity_detection": detection_calls,
        "latent_entity_detection": 1 if rewrite_body_calls else 0,
        "replace_map_generation": replace_map_generation,
        "rewrite_pipeline": 5 if rewrite_body_calls else 0,
        "rewrite_evaluate": 3 * (1 + repair_iterations) if rewrite_body_calls else 0,
        "rewrite_repair": repair_iterations if rewrite_body_calls else 0,
        "rewrite_final_judge": 1 if rewrite_body_calls else 0,
    }


def _write_collector_safely(
    *,
    config: MeasurementConfig,
    collector: MeasurementCollector,
    body_error: BaseException | None,
) -> None:
    try:
        config.write_collector(collector)
    except Exception as exc:
        logger.warning("Failed to write Anonymizer measurement records (%s)", type(exc).__name__)
        if body_error is None and config.fail_on_write_error:
            raise


def _close_collector_safely(
    *,
    config: MeasurementConfig,
    collector: MeasurementCollector,
    body_error: BaseException | None,
) -> None:
    try:
        collector.close()
    except Exception as exc:
        logger.warning("Failed to close Anonymizer measurement stream (%s)", type(exc).__name__)
        if body_error is None and config.fail_on_write_error:
            raise


def _measurement_env_error_message(exc: SettingsError | ValidationError, *, prefix: str) -> str:
    fields: set[str] = set()
    if isinstance(exc, ValidationError):
        for error in exc.errors(include_input=False):
            loc = error.get("loc", ())
            if loc:
                fields.add(str(loc[0]).upper())
    else:
        error_text = str(exc).lower()
        for field_name in _MeasurementEnvSettings.model_fields:
            if field_name in error_text:
                fields.add(field_name.upper())

    if fields:
        env_fields = ", ".join(f"{prefix}{field}" for field in sorted(fields))
        return f"Invalid Anonymizer measurement environment configuration for: {env_fields}"
    return "Invalid Anonymizer measurement environment configuration"


def _load_measurement_env_settings(*, prefix: str) -> _MeasurementEnvSettings:
    settings_factory = cast(Any, _MeasurementEnvSettings)
    return cast(_MeasurementEnvSettings, settings_factory(_env_prefix=prefix))


def _validation_chunk_count(
    detected_candidate_count: int | None,
    *,
    validation_max_entities_per_call: int,
) -> int | None:
    if detected_candidate_count is None:
        return None
    if detected_candidate_count <= 0:
        return 0
    return int(math.ceil(detected_candidate_count / validation_max_entities_per_call))


def _safe_row_index(row_index: object) -> int | None:
    if isinstance(row_index, bool):
        return None
    if isinstance(row_index, Integral):
        return int(row_index)
    return None


def _entities_from_raw(raw: object) -> list[dict[str, Any]]:
    payload = _coerce_payload(raw)
    if isinstance(payload, Mapping):
        items = cast(Mapping[str, Any], payload).get("entities", [])
    elif isinstance(payload, list):
        items = payload
    else:
        items = []
    return [dict(cast(Mapping[str, Any], item)) for item in items if isinstance(item, Mapping)]


def _entity_ground_truth_metrics(
    final_entities: list[dict[str, Any]],
    ground_truth_entities: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    if ground_truth_entities is None:
        return {
            "ground_truth_entity_count": None,
            "ground_truth_entity_label_counts": None,
            "entity_true_positive_count": None,
            "entity_false_positive_count": None,
            "entity_false_negative_count": None,
            "entity_precision": None,
            "entity_recall": None,
            "entity_f1": None,
        }

    predicted = _entity_identity_set(final_entities)
    expected = _entity_identity_set(ground_truth_entities)
    true_positive = len(predicted & expected)
    false_positive = len(predicted - expected)
    false_negative = len(expected - predicted)
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    f1 = (
        None
        if precision is None or recall is None or precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    return {
        "ground_truth_entity_count": len(ground_truth_entities),
        "ground_truth_entity_label_counts": dict(
            sorted(Counter(e.get("label", "") for e in ground_truth_entities if e.get("label")).items())
        ),
        "entity_true_positive_count": true_positive,
        "entity_false_positive_count": false_positive,
        "entity_false_negative_count": false_negative,
        "entity_precision": precision,
        "entity_recall": recall,
        "entity_f1": f1,
    }


def _entity_identity_set(entities: list[dict[str, Any]]) -> set[tuple[str, str]]:
    identities: set[tuple[str, str]] = set()
    for entity in entities:
        label = entity.get("label")
        value = entity.get("value")
        if label is None or value is None:
            continue
        identities.add((str(value), str(label)))
    return identities


def _replacement_map_metrics(raw: object) -> dict[str, Any]:
    replacement_maps = _replacement_maps_from_raw(raw)
    synthetic_values = []
    for item in replacement_maps:
        synthetic = item.get("replacement", item.get("synthetic"))
        if synthetic is not None:
            synthetic_values.append(str(synthetic))
    return {
        "replacement_count": len(replacement_maps),
        "replacement_label_counts": dict(
            sorted(Counter(item.get("label", "") for item in replacement_maps if item.get("label")).items())
        ),
        "replacement_duplicate_value_count": max(0, len(synthetic_values) - len(set(synthetic_values))),
    }


def _replacement_coverage_metrics(raw: object, final_entities: list[dict[str, Any]]) -> dict[str, Any]:
    replacement_original_values = {
        str(original)
        for item in _replacement_maps_from_raw(raw)
        if (original := item.get("original")) is not None and str(original)
    }
    missing_entities = [
        entity
        for entity in final_entities
        if entity.get("value") and str(entity.get("value")) not in replacement_original_values
    ]
    missing_values = {str(entity.get("value")) for entity in missing_entities if entity.get("value")}
    return {
        "replacement_missing_final_entity_count": len(missing_entities),
        "replacement_missing_final_entity_label_counts": dict(
            sorted(
                Counter(str(entity.get("label") or "") for entity in missing_entities if entity.get("label")).items()
            )
        ),
        "replacement_missing_final_value_count": len(missing_values),
    }


def _replacement_collision_metrics(raw: object, final_entities: list[dict[str, Any]]) -> dict[str, Any]:
    synthetic_values = {
        str(synthetic)
        for item in _replacement_maps_from_raw(raw)
        if (synthetic := item.get("replacement", item.get("synthetic"))) is not None and str(synthetic)
    }
    collided_entities = [
        entity for entity in final_entities if entity.get("value") and str(entity.get("value")) in synthetic_values
    ]
    collided_values = {str(entity.get("value")) for entity in collided_entities if entity.get("value")}
    return {
        "replacement_synthetic_original_collision_count": len(collided_entities),
        "replacement_synthetic_original_collision_label_counts": dict(
            sorted(
                Counter(str(entity.get("label") or "") for entity in collided_entities if entity.get("label")).items()
            )
        ),
        "replacement_synthetic_original_collision_value_count": len(collided_values),
    }


def _replacement_maps_from_raw(raw: object) -> list[Mapping[str, Any]]:
    payload = _coerce_payload(raw)
    if isinstance(payload, Mapping):
        replacements_raw = cast(Mapping[str, Any], payload).get("replacements")
        tolist = getattr(replacements_raw, "tolist", None)
        if callable(tolist):
            replacements_raw = tolist()
        replacements = replacements_raw if isinstance(replacements_raw, list) else []
    elif isinstance(payload, list):
        replacements = payload
    else:
        replacements = []
    return [cast(Mapping[str, Any], item) for item in replacements if isinstance(item, Mapping)]


def _count_items(raw: object, *, primary_key: str, fallback_keys: tuple[str, ...] = ()) -> int:
    payload = _coerce_payload(raw)
    if isinstance(payload, Mapping):
        payload_map = cast(Mapping[str, Any], payload)
        for key in (primary_key, *fallback_keys):
            items = payload_map.get(key)
            if isinstance(items, list):
                return len(items)
        return 0
    if isinstance(payload, list):
        return len(payload)
    return 0


def _coerce_payload(raw: object) -> object:
    model_dump = getattr(raw, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="python")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if raw is None:
        return {}
    return raw


def _coerce_int(raw: object, *, default: int) -> int:
    try:
        return int(cast(Any, raw))
    except (TypeError, ValueError):
        return default


def _coerce_float(raw: object) -> float | None:
    try:
        value = float(cast(Any, raw))
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) else value


def _coerce_bool(raw: object) -> bool | None:
    if raw is None:
        return None
    if isinstance(raw, float) and math.isnan(raw):
        return None
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
        return None
    try:
        return bool(cast(Any, raw))
    except (TypeError, ValueError):
        return None


def _safe_rate(numerator: int | float | None, elapsed_sec: float) -> float | None:
    if numerator is None or elapsed_sec <= 0:
        return None
    return float(numerator) / elapsed_sec


def _safe_ratio(numerator: int | float | None, denominator: int | float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _size_bucket(value: int) -> str:
    if value == 0:
        return "0"
    for upper in (128, 512, 2048, 8192):
        if value < upper:
            return f"1-{upper - 1}" if upper == 128 else f"{upper // 4}-{upper - 1}"
    return "8192+"


def _count_text_tokens(text: str) -> int:
    try:
        import tiktoken

        tokenizer = tiktoken.get_encoding("cl100k_base")
        return len(tokenizer.encode(text, disallowed_special=()))
    except Exception:
        return len(text.split())


def _summarize_model_usage(model_usage: Mapping[str, Any] | None) -> dict[str, int | None]:
    totals = _empty_model_usage_totals()
    for usage in (model_usage or {}).values():
        if not isinstance(usage, Mapping):
            continue
        _add_model_usage_totals(totals, usage)

    if totals["total_tokens"] == 0:
        totals["total_tokens"] = totals["input_tokens"] + totals["output_tokens"]
    if totals["total_requests"] == 0:
        totals["total_requests"] = totals["successful_requests"] + totals["failed_requests"]

    return {
        "observed_input_tokens": totals["input_tokens"],
        "observed_output_tokens": totals["output_tokens"],
        "observed_total_tokens": totals["total_tokens"],
        "observed_reasoning_tokens": totals["reasoning_tokens"] if totals["has_reasoning_tokens"] else None,
        "observed_successful_requests": totals["successful_requests"],
        "observed_failed_requests": totals["failed_requests"],
        "observed_total_requests": totals["total_requests"],
    }


def _empty_model_usage_totals() -> dict[str, int | bool]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
        "has_reasoning_tokens": False,
        "successful_requests": 0,
        "failed_requests": 0,
        "total_requests": 0,
    }


def _add_model_usage_totals(totals: dict[str, int | bool], usage: Mapping[str, Any]) -> None:
    token_usage = usage.get("token_usage")
    if isinstance(token_usage, Mapping):
        totals["input_tokens"] += _coerce_int(token_usage.get("input_tokens"), default=0)
        totals["output_tokens"] += _coerce_int(token_usage.get("output_tokens"), default=0)
        totals["total_tokens"] += _coerce_int(token_usage.get("total_tokens"), default=0)
        if token_usage.get("reasoning_tokens") is not None:
            totals["has_reasoning_tokens"] = True
            totals["reasoning_tokens"] += _coerce_int(token_usage.get("reasoning_tokens"), default=0)

    request_usage = usage.get("request_usage")
    if isinstance(request_usage, Mapping):
        totals["successful_requests"] += _coerce_int(request_usage.get("successful_requests"), default=0)
        totals["failed_requests"] += _coerce_int(request_usage.get("failed_requests"), default=0)
        totals["total_requests"] += _coerce_int(request_usage.get("total_requests"), default=0)


def _json_safe(value: object) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return sorted((_json_safe(v) for v in value), key=str)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
