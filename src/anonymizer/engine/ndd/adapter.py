# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import tempfile
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, cast

from data_designer.config.column_configs import CustomColumnConfig, LLMStructuredColumnConfig, LLMTextColumnConfig
from data_designer.config.column_types import ColumnConfigT
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.models import ModelConfig
from data_designer.config.run_config import RunConfig
from data_designer.config.seed import SamplingStrategy
from data_designer.config.seed_source import LocalFileSeedSource
from data_designer.config.utils.constants import TRACE_COLUMN_POSTFIX
from data_designer.config.utils.trace_type import TraceType

from anonymizer.interface.errors import AnonymizerWorkflowError
from anonymizer.measurement import current_collector, record_ndd_workflow

if TYPE_CHECKING:
    import pandas as pd
    from data_designer.interface.data_designer import DataDesigner

logger = logging.getLogger("anonymizer.ndd")

RECORD_ID_COLUMN = "_anonymizer_record_id"
_TRACEABLE_LLM_COLUMN_TYPES = (LLMTextColumnConfig, LLMStructuredColumnConfig)


@dataclass(frozen=True)
class FailedRecord:
    """A record that did not appear in workflow output."""

    record_id: str
    step: str
    reason: str


@dataclass(frozen=True)
class WorkflowRunResult:
    """Result of a single NDD workflow execution."""

    dataframe: pd.DataFrame
    failed_records: list[FailedRecord]


@dataclass(frozen=True)
class _NativeTraceColumn:
    column_name: str
    trace_column_name: str
    model_alias: str | None
    model_name: str | None
    model_provider_name: str | None


class NddAdapter:
    """Adapter for running NDD workflows with uniform I/O and record tracking."""

    def __init__(self, data_designer: DataDesigner) -> None:
        self._data_designer = data_designer
        self._run_lock = RLock()
        logger.debug("NDD adapter: artifact_path=%s", getattr(data_designer, "_artifact_path", "unknown"))

    def run_workflow(
        self,
        df: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        columns: list[ColumnConfigT],
        workflow_name: str,
        preview_num_records: int | None = None,
    ) -> WorkflowRunResult:
        """Run one workflow and return output with missing-record tracking.

        Wraps a DataFrame slice plus NDD column configs into a DataDesigner
        run and returns `WorkflowRunResult(dataframe, failed_records)`.
        Records missing from the output surface as `FailedRecord` objects
        rather than silently disappearing.

        This is the engine boundary for *executing* DataDesigner workflows.
        Engine sub-workflows declare column configs and call this method;
        they do not call `DataDesigner.create()` or `.preview()` directly.

        Args:
            df: Input DataFrame.
            model_configs: NDD model aliases available to the workflow.
            columns: NDD column configs to add to the workflow.
            workflow_name: Identifier used in logs and on `FailedRecord` entries.
            preview_num_records: If set, run in preview mode against this many rows.

        Returns:
            `WorkflowRunResult` with the output DataFrame and any `FailedRecord` entries.
        """
        workflow_input_df = self._attach_record_ids(df=df)
        logger.debug("NDD workflow '%s' starting with %d records", workflow_name, len(workflow_input_df))
        col_names = [c.name for c in columns]
        logger.debug("NDD workflow '%s': %d columns %s", workflow_name, len(col_names), col_names)
        available_model_aliases = [m.alias for m in model_configs]
        model_aliases = _extract_workflow_model_aliases(columns) or available_model_aliases
        record_count = (
            min(preview_num_records, len(workflow_input_df))
            if preview_num_records is not None
            else len(workflow_input_df)
        )
        started = time.perf_counter()
        collector = current_collector()
        usage_probe = _DataDesignerUsageProbe(self._data_designer, enabled=collector is not None)
        columns, native_trace_columns, unsupported_trace_columns = _configure_native_dd_message_traces(
            columns=columns,
            model_configs=model_configs,
            collector=collector,
        )
        _record_dd_trace_coverage(
            workflow_name=workflow_name,
            collector=collector,
            native_trace_columns=native_trace_columns,
            unsupported_trace_columns=unsupported_trace_columns,
        )

        with tempfile.TemporaryDirectory(prefix=f"anonymizer_{workflow_name}_") as tmp_dir:
            seed_path = str(Path(tmp_dir) / "seed.parquet")
            seed_source = LocalFileSeedSource.from_dataframe(workflow_input_df, seed_path)

            config_builder = DataDesignerConfigBuilder(model_configs=model_configs)
            config_builder.with_seed_dataset(seed_source, sampling_strategy=SamplingStrategy.ORDERED)
            for column in columns:
                config_builder.add_column(column)

            try:
                task_traces: list[Any] = []
                with self._run_lock, usage_probe, _temporary_dd_task_trace(self._data_designer, collector=collector):
                    if preview_num_records is None:
                        run_results = self._data_designer.create(
                            config_builder,
                            num_records=len(workflow_input_df),
                            dataset_name=workflow_name,
                        )
                        task_traces = _task_traces_from_result(run_results)
                        output_df = run_results.load_dataset()
                    else:
                        preview_results = self._data_designer.preview(
                            config_builder,
                            num_records=record_count,
                        )
                        task_traces = _task_traces_from_result(preview_results)
                        if preview_results.dataset is None:
                            output_df = workflow_input_df.iloc[0:0].copy()
                        else:
                            output_df = preview_results.dataset
                    output_df = _record_and_strip_native_dd_message_traces(
                        output_df=output_df,
                        workflow_name=workflow_name,
                        collector=collector,
                        native_trace_columns=native_trace_columns,
                    )
                    _record_dd_task_traces(
                        workflow_name=workflow_name,
                        collector=collector,
                        task_traces=task_traces,
                    )
            except Exception as exc:
                logger.warning(
                    "Workflow failed for %d input record(s) on model(s) %s: %s",
                    record_count,
                    available_model_aliases,
                    exc,
                )
                logger.debug(
                    "Workflow '%s' failure context: columns=%s",
                    workflow_name,
                    col_names,
                )
                record_ndd_workflow(
                    workflow_name=workflow_name,
                    model_aliases=model_aliases,
                    input_row_count=record_count,
                    seed_row_count=len(workflow_input_df),
                    output_row_count=None,
                    failed_record_count=None,
                    elapsed_sec=time.perf_counter() - started,
                    status="error",
                    preview_num_records=preview_num_records,
                    column_count=len(col_names),
                    column_names=col_names,
                    model_usage=usage_probe.model_usage(),
                )
                raise AnonymizerWorkflowError(f"Workflow failed: {exc}") from exc

        logger.debug("NDD workflow '%s' returned %d records", workflow_name, len(output_df))
        failed_records = self._detect_missing_records(
            workflow_name=workflow_name,
            input_df=(
                workflow_input_df.iloc[:preview_num_records].copy()
                if preview_num_records is not None
                else workflow_input_df
            ),
            output_df=output_df,
        )
        record_ndd_workflow(
            workflow_name=workflow_name,
            model_aliases=model_aliases,
            input_row_count=record_count,
            seed_row_count=len(workflow_input_df),
            output_row_count=len(output_df),
            failed_record_count=len(failed_records),
            elapsed_sec=time.perf_counter() - started,
            preview_num_records=preview_num_records,
            column_count=len(col_names),
            column_names=col_names,
            model_usage=usage_probe.model_usage(),
        )
        return WorkflowRunResult(dataframe=output_df, failed_records=failed_records)

    def _attach_record_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        if RECORD_ID_COLUMN in df.columns:
            return df.copy()

        dataframe = df.copy()
        dataframe[RECORD_ID_COLUMN] = dataframe.apply(self._compute_record_id, axis=1)
        return dataframe

    @staticmethod
    def _compute_record_id(row: pd.Series) -> str:
        # uuid5 is deterministic so input/output IDs match for missing-record tracking.
        # Include row index so duplicate rows get distinct IDs.
        # TODO: consider whether id_column from AnonymizerInput should be used here instead.
        row_data = {**row.to_dict(), "__row_idx__": row.name}
        serialized = json.dumps(row_data, default=str, sort_keys=True, separators=(",", ":"))
        return uuid.uuid5(uuid.NAMESPACE_URL, serialized).hex

    def _detect_missing_records(
        self,
        *,
        workflow_name: str,
        input_df: pd.DataFrame,
        output_df: pd.DataFrame,
    ) -> list[FailedRecord]:
        if RECORD_ID_COLUMN not in input_df.columns:
            logger.warning(
                "Missing-record detection skipped: input DataFrame lacks the record-tracking "
                "column, so the adapter cannot verify whether any of %d input record(s) were "
                "dropped.",
                len(input_df),
            )
            logger.debug(
                "Workflow '%s' detection skipped: input missing '%s'; input_columns=%s",
                workflow_name,
                RECORD_ID_COLUMN,
                list(input_df.columns),
            )
            return []
        if RECORD_ID_COLUMN not in output_df.columns:
            input_cols = set(input_df.columns)
            output_cols = set(output_df.columns)
            other_dropped = sorted((input_cols - output_cols) - {RECORD_ID_COLUMN})
            added = sorted(output_cols - input_cols)
            logger.warning(
                "Missing-record detection disabled: workflow output does not contain "
                "the record-tracking column, so all %d input record(s) are being marked as "
                "failed. This typically means seed-column pass-through is disabled or a "
                "user-supplied column config overwrote it. Other input columns that were "
                "also dropped: %s. Columns added by the workflow: %s.",
                len(input_df),
                other_dropped,
                added,
            )
            logger.debug(
                "Workflow '%s' detection disabled: output missing '%s'; input_columns=%s output_columns=%s",
                workflow_name,
                RECORD_ID_COLUMN,
                list(input_df.columns),
                list(output_df.columns),
            )
            return [
                FailedRecord(
                    record_id=record_id,
                    step=workflow_name,
                    reason=f"Output is missing required tracking column '{RECORD_ID_COLUMN}'",
                )
                for record_id in input_df[RECORD_ID_COLUMN].astype(str).tolist()
            ]

        input_ids = set(input_df[RECORD_ID_COLUMN].astype(str).tolist())
        output_ids = set(output_df[RECORD_ID_COLUMN].astype(str).tolist())
        missing_ids = sorted(input_ids - output_ids)
        return [
            FailedRecord(
                record_id=record_id,
                step=workflow_name,
                reason="Record missing from workflow output",
            )
            for record_id in missing_ids
        ]


def _extract_workflow_model_aliases(columns: list[ColumnConfigT]) -> list[str]:
    aliases: list[str] = []
    for column in columns:
        aliases.extend(_as_alias_list(getattr(column, "model_alias", None)))
        generator = getattr(column, "generator_function", None)
        metadata = getattr(generator, "custom_column_metadata", None)
        if isinstance(metadata, dict):
            aliases.extend(_as_alias_list(metadata.get("model_aliases")))
    return list(dict.fromkeys(alias for alias in aliases if alias))


def _as_alias_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, (list, tuple, set)):
        return [str(item) for item in raw if item is not None and str(item)]
    return [str(raw)]


class _DataDesignerUsageProbe:
    """Capture DataDesigner model usage from the per-run private ResourceProvider."""

    def __init__(self, data_designer: DataDesigner, *, enabled: bool) -> None:
        self._data_designer = data_designer
        self._enabled = enabled
        self._original_create_resource_provider: Any | None = None
        self._resource_providers: list[Any] = []

    def __enter__(self) -> _DataDesignerUsageProbe:
        if not self._enabled:
            return self

        original = getattr(self._data_designer, "_create_resource_provider", None)
        if not callable(original):
            return self

        self._original_create_resource_provider = original

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            resource_provider = original(*args, **kwargs)
            self._resource_providers.append(resource_provider)
            return resource_provider

        setattr(self._data_designer, "_create_resource_provider", wrapper)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._original_create_resource_provider is not None:
            setattr(self._data_designer, "_create_resource_provider", self._original_create_resource_provider)

    def model_usage(self) -> dict[str, Any] | None:
        usage: dict[str, Any] = {}
        for resource_provider in self._resource_providers:
            model_registry = getattr(resource_provider, "model_registry", None)
            snapshot = _get_model_usage_snapshot(model_registry)
            if not snapshot:
                continue
            for model_name, stats in snapshot.items():
                usage[str(model_name)] = _model_usage_as_json(stats)
        return usage or None


def _get_model_usage_snapshot(model_registry: object) -> Mapping[str, object] | None:
    alias_snapshot = _get_model_usage_snapshot_by_alias(model_registry)
    if alias_snapshot:
        return alias_snapshot

    get_snapshot = getattr(model_registry, "get_model_usage_snapshot", None)
    if not callable(get_snapshot):
        return None
    snapshot = get_snapshot()
    if isinstance(snapshot, Mapping):
        return snapshot
    return None


def _get_model_usage_snapshot_by_alias(model_registry: object) -> Mapping[str, object] | None:
    models = getattr(model_registry, "_models", None)
    if not isinstance(models, Mapping):
        return None

    snapshot: dict[str, object] = {}
    for model_alias, model_facade in models.items():
        stats = getattr(model_facade, "usage_stats", None)
        if stats is None or not getattr(stats, "has_usage", False):
            continue
        payload = _model_usage_as_json(stats)
        if isinstance(payload, Mapping):
            payload = {
                **payload,
                "model_alias": getattr(model_facade, "model_alias", str(model_alias)),
                "model_name": getattr(model_facade, "model_name", None),
                "model_provider_name": getattr(model_facade, "model_provider_name", None),
            }
        snapshot[str(model_alias)] = payload
    return snapshot or None


def _model_usage_as_json(stats: object) -> Any:
    model_dump = getattr(stats, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return stats


@contextmanager
def _temporary_dd_task_trace(data_designer: DataDesigner, *, collector: Any | None) -> Iterator[None]:
    if collector is None or not collector.dd_task_trace_enabled:
        yield
        return

    original_run_config = getattr(data_designer, "run_config", None)
    set_run_config = getattr(data_designer, "set_run_config", None)
    if original_run_config is None or not callable(set_run_config):
        yield
        return

    traced_run_config = _run_config_with_async_trace(original_run_config)
    set_run_config(traced_run_config)
    try:
        yield
    finally:
        set_run_config(original_run_config)


def _run_config_with_async_trace(run_config: Any) -> Any:
    model_copy = getattr(run_config, "model_copy", None)
    if callable(model_copy):
        return model_copy(update={"async_trace": True})
    if isinstance(run_config, RunConfig):
        return run_config.model_copy(update={"async_trace": True})
    return run_config


def _task_traces_from_result(result: Any) -> list[Any]:
    raw_traces = getattr(result, "task_traces", None)
    if raw_traces is None:
        return []
    if isinstance(raw_traces, list):
        return raw_traces
    try:
        return list(raw_traces)
    except TypeError:
        return []


def _configure_native_dd_message_traces(
    *,
    columns: list[ColumnConfigT],
    model_configs: list[ModelConfig],
    collector: Any | None,
) -> tuple[list[ColumnConfigT], list[_NativeTraceColumn], list[ColumnConfigT]]:
    if collector is None or not collector.dd_trace_enabled:
        return columns, [], []

    model_configs_by_alias = {model_config.alias: model_config for model_config in model_configs}
    traced_columns: list[_NativeTraceColumn] = []
    unsupported_columns: list[ColumnConfigT] = []
    configured_columns: list[ColumnConfigT] = []
    trace_type = _native_dd_trace_type()

    for column in columns:
        if isinstance(column, _TRACEABLE_LLM_COLUMN_TYPES):
            configured_column = cast(ColumnConfigT, column.model_copy(update={"with_trace": trace_type}))
            configured_columns.append(configured_column)
            model_config = model_configs_by_alias.get(column.model_alias)
            traced_columns.append(
                _NativeTraceColumn(
                    column_name=column.name,
                    trace_column_name=f"{column.name}{TRACE_COLUMN_POSTFIX}",
                    model_alias=column.model_alias,
                    model_name=getattr(model_config, "model", None),
                    model_provider_name=getattr(model_config, "provider", None),
                )
            )
            continue

        configured_columns.append(column)
        if _column_has_untraced_model_calls(column):
            unsupported_columns.append(column)

    return configured_columns, traced_columns, unsupported_columns


def _native_dd_trace_type() -> TraceType:
    # Preserve Anonymizer's existing dd_trace=last_message semantics: the trace
    # sink records the final prompt message and response separately, while DD's
    # native LAST_MESSAGE side effect only keeps the final assistant message.
    return TraceType.ALL_MESSAGES


def _column_has_untraced_model_calls(column: ColumnConfigT) -> bool:
    return isinstance(column, CustomColumnConfig) and bool(_extract_workflow_model_aliases([column]))


def _record_dd_trace_coverage(
    *,
    workflow_name: str,
    collector: Any,
    native_trace_columns: list[_NativeTraceColumn],
    unsupported_trace_columns: list[ColumnConfigT],
) -> Any:
    if collector is None or not collector.dd_trace_enabled:
        return
    collector.record(
        "dd_trace_coverage",
        workflow_name=workflow_name,
        trace_backend="data_designer_column",
        trace_mode=collector.dd_trace_mode,
        native_trace_type=_native_dd_trace_type().value,
        traced_column_count=len(native_trace_columns),
        traced_column_names=[column.column_name for column in native_trace_columns],
        unsupported_column_count=len(unsupported_trace_columns),
        unsupported_column_names=[column.name for column in unsupported_trace_columns],
        unsupported_column_types=[_column_type_name(column) for column in unsupported_trace_columns],
    )


def _column_type_name(column: ColumnConfigT) -> str:
    column_type = getattr(column, "column_type", None)
    return str(column_type) if column_type is not None else type(column).__name__


def _record_and_strip_native_dd_message_traces(
    *,
    output_df: pd.DataFrame,
    workflow_name: str,
    collector: Any,
    native_trace_columns: list[_NativeTraceColumn],
) -> pd.DataFrame:
    if not native_trace_columns:
        return output_df

    trace_column_names = [column.trace_column_name for column in native_trace_columns]
    if collector is not None and collector.dd_trace_enabled:
        for _, row in output_df.iterrows():
            for trace_column in native_trace_columns:
                if trace_column.trace_column_name not in output_df.columns:
                    continue
                trace_messages = _native_trace_messages(row.get(trace_column.trace_column_name))
                if not trace_messages:
                    continue
                collector.record_dd_message_trace(
                    workflow_name=workflow_name,
                    trace_source="data_designer_column",
                    column_name=trace_column.column_name,
                    trace_column_name=trace_column.trace_column_name,
                    model_alias=trace_column.model_alias,
                    model_name=trace_column.model_name,
                    model_provider_name=trace_column.model_provider_name,
                    modality="chat",
                    is_async=None,
                    status="completed",
                    error_type=None,
                    elapsed_sec=None,
                    messages=_select_native_trace_messages(trace_messages, mode=collector.dd_trace_mode),
                    response=_native_trace_response(trace_messages),
                    usage=None,
                )

    existing_trace_columns = [column_name for column_name in trace_column_names if column_name in output_df.columns]
    if not existing_trace_columns:
        return output_df
    return output_df.drop(columns=existing_trace_columns)


def _native_trace_messages(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [_trace_message(message) for message in value]


def _select_native_trace_messages(messages: list[dict[str, Any]], *, mode: str) -> list[dict[str, Any]]:
    if mode == "all_messages":
        return messages
    last_prompt = next((message for message in reversed(messages) if message.get("role") != "assistant"), None)
    return [last_prompt] if last_prompt is not None else []


def _native_trace_response(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    assistant_message = next((message for message in reversed(messages) if message.get("role") == "assistant"), None)
    if assistant_message is None:
        return None
    return {
        "content": assistant_message.get("content"),
        "reasoning_content": assistant_message.get("reasoning_content"),
        "tool_calls": _trace_tool_calls(assistant_message.get("tool_calls", [])),
    }


def _trace_message(message: Any) -> dict[str, Any]:
    to_dict = getattr(message, "to_dict", None)
    if callable(to_dict):
        return cast(dict[str, Any], to_dict())
    if isinstance(message, Mapping):
        return dict(message)
    return {"role": getattr(message, "role", None), "content": getattr(message, "content", None)}


def _trace_tool_calls(tool_calls: Any) -> list[Any]:
    if isinstance(tool_calls, list):
        return [getattr(tool_call, "__dict__", tool_call) for tool_call in tool_calls]
    return []


def _record_dd_task_traces(*, workflow_name: str, collector: Any | None, task_traces: list[Any]) -> None:
    if collector is None or not collector.dd_task_trace_enabled:
        return
    for task_trace in task_traces:
        collector.record_dd_task_trace(
            workflow_name=workflow_name,
            trace_source="data_designer_scheduler",
            column=_trace_attr(task_trace, "column"),
            row_group=_trace_attr(task_trace, "row_group"),
            row_index=_trace_attr(task_trace, "row_index"),
            task_type=_trace_attr(task_trace, "task_type"),
            status=_trace_attr(task_trace, "status"),
            error_present=bool(_trace_attr(task_trace, "error")),
            queue_wait_sec=_trace_duration(
                _trace_attr(task_trace, "dispatched_at"),
                _trace_attr(task_trace, "slot_acquired_at"),
            ),
            execution_sec=_trace_duration(
                _trace_attr(task_trace, "slot_acquired_at"),
                _trace_attr(task_trace, "completed_at"),
            ),
            total_sec=_trace_duration(
                _trace_attr(task_trace, "dispatched_at"),
                _trace_attr(task_trace, "completed_at"),
            ),
        )


def _trace_attr(task_trace: Any, name: str) -> Any:
    if isinstance(task_trace, Mapping):
        return task_trace.get(name)
    return getattr(task_trace, name, None)


def _trace_duration(start: Any, end: Any) -> float | None:
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return None
    if start <= 0 or end <= 0 or end < start:
        return None
    return float(end - start)
