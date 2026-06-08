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

from data_designer.config.column_types import ColumnConfigT
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.models import ModelConfig
from data_designer.config.seed import SamplingStrategy
from data_designer.config.seed_source import LocalFileSeedSource

from anonymizer.interface.errors import AnonymizerWorkflowError
from anonymizer.measurement import current_collector, record_ndd_workflow

if TYPE_CHECKING:
    import pandas as pd
    from data_designer.interface.data_designer import DataDesigner

logger = logging.getLogger("anonymizer.ndd")

RECORD_ID_COLUMN = "_anonymizer_record_id"
_DD_MESSAGE_TRACE_PATCH_LOCK = RLock()


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


class NddAdapter:
    """Adapter for running NDD workflows with uniform I/O and record tracking."""

    def __init__(self, data_designer: DataDesigner) -> None:
        self._data_designer = data_designer
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
        usage_probe = _DataDesignerUsageProbe(self._data_designer, enabled=current_collector() is not None)

        with tempfile.TemporaryDirectory(prefix=f"anonymizer_{workflow_name}_") as tmp_dir:
            seed_path = str(Path(tmp_dir) / "seed.parquet")
            seed_source = LocalFileSeedSource.from_dataframe(workflow_input_df, seed_path)

            config_builder = DataDesignerConfigBuilder(model_configs=model_configs)
            config_builder.with_seed_dataset(seed_source, sampling_strategy=SamplingStrategy.ORDERED)
            for column in columns:
                config_builder.add_column(column)

            try:
                with usage_probe, _dd_message_trace(workflow_name=workflow_name):
                    if preview_num_records is None:
                        run_results = self._data_designer.create(
                            config_builder,
                            num_records=len(workflow_input_df),
                            dataset_name=workflow_name,
                        )
                        output_df = run_results.load_dataset()
                    else:
                        preview_results = self._data_designer.preview(
                            config_builder,
                            num_records=record_count,
                        )
                        if preview_results.dataset is None:
                            output_df = workflow_input_df.iloc[0:0].copy()
                        else:
                            output_df = preview_results.dataset
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
        return [str(item) for item in raw if str(item)]
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
def _dd_message_trace(*, workflow_name: str) -> Iterator[None]:
    collector = current_collector()
    if collector is None or not collector.dd_trace_enabled:
        yield
        return

    from data_designer.engine.models.facade import ModelFacade

    with _DD_MESSAGE_TRACE_PATCH_LOCK:
        original_completion = ModelFacade.completion
        original_acompletion = ModelFacade.acompletion
        ModelFacade.completion = _traced_completion(
            original_completion, collector=collector, workflow_name=workflow_name
        )
        ModelFacade.acompletion = _traced_acompletion(
            original_acompletion,
            collector=collector,
            workflow_name=workflow_name,
        )
        try:
            yield
        finally:
            ModelFacade.completion = original_completion
            ModelFacade.acompletion = original_acompletion


def _traced_completion(original_completion: Any, *, collector: Any, workflow_name: str) -> Any:
    def traced(model_facade: Any, messages: list[Any], *args: Any, **kwargs: Any) -> Any:
        return _run_traced_completion(
            original_completion,
            collector=collector,
            workflow_name=workflow_name,
            model_facade=model_facade,
            messages=messages,
            args=args,
            kwargs=kwargs,
        )

    return traced


def _run_traced_completion(
    completion: Any,
    *,
    collector: Any,
    workflow_name: str,
    model_facade: Any,
    messages: list[Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    started = time.perf_counter()
    response: Any | None = None
    status = "completed"
    error_type: str | None = None
    try:
        response = completion(model_facade, messages, *args, **kwargs)
        return response
    except BaseException as exc:
        status = "error"
        error_type = type(exc).__name__
        raise
    finally:
        _record_dd_message_trace(
            collector=collector,
            workflow_name=workflow_name,
            model_facade=model_facade,
            messages=messages,
            response=response,
            elapsed_sec=time.perf_counter() - started,
            status=status,
            error_type=error_type,
            is_async=False,
        )


def _traced_acompletion(original_acompletion: Any, *, collector: Any, workflow_name: str) -> Any:
    async def traced(model_facade: Any, messages: list[Any], *args: Any, **kwargs: Any) -> Any:
        return await _run_traced_acompletion(
            original_acompletion,
            collector=collector,
            workflow_name=workflow_name,
            model_facade=model_facade,
            messages=messages,
            args=args,
            kwargs=kwargs,
        )

    return traced


async def _run_traced_acompletion(
    acompletion: Any,
    *,
    collector: Any,
    workflow_name: str,
    model_facade: Any,
    messages: list[Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    started = time.perf_counter()
    response: Any | None = None
    status = "completed"
    error_type: str | None = None
    try:
        response = await acompletion(model_facade, messages, *args, **kwargs)
        return response
    except BaseException as exc:
        status = "error"
        error_type = type(exc).__name__
        raise
    finally:
        _record_dd_message_trace(
            collector=collector,
            workflow_name=workflow_name,
            model_facade=model_facade,
            messages=messages,
            response=response,
            elapsed_sec=time.perf_counter() - started,
            status=status,
            error_type=error_type,
            is_async=True,
        )


def _record_dd_message_trace(
    *,
    collector: Any,
    workflow_name: str,
    model_facade: Any,
    messages: list[Any],
    response: Any | None,
    elapsed_sec: float,
    status: str,
    error_type: str | None,
    is_async: bool,
) -> None:
    collector.record_dd_message_trace(
        workflow_name=workflow_name,
        model_alias=getattr(model_facade, "model_alias", None),
        model_name=getattr(model_facade, "model_name", None),
        model_provider_name=getattr(model_facade, "model_provider_name", None),
        modality="chat",
        is_async=is_async,
        status=status,
        error_type=error_type,
        elapsed_sec=elapsed_sec,
        messages=_trace_messages(messages, mode=collector.dd_trace_mode),
        response=_trace_response(response),
        usage=_trace_usage(getattr(response, "usage", None) if response is not None else None),
    )


def _trace_messages(messages: list[Any], *, mode: str) -> list[dict[str, Any]]:
    selected = messages if mode == "all_messages" else messages[-1:]
    return [_trace_message(message) for message in selected]


def _trace_message(message: Any) -> dict[str, Any]:
    to_dict = getattr(message, "to_dict", None)
    if callable(to_dict):
        return cast(dict[str, Any], to_dict())
    if isinstance(message, Mapping):
        return dict(message)
    return {"role": getattr(message, "role", None), "content": getattr(message, "content", None)}


def _trace_response(response: Any | None) -> dict[str, Any] | None:
    if response is None:
        return None
    message = getattr(response, "message", None)
    if message is None:
        return None
    return {
        "content": getattr(message, "content", None),
        "reasoning_content": getattr(message, "reasoning_content", None),
        "tool_calls": _trace_tool_calls(getattr(message, "tool_calls", [])),
    }


def _trace_tool_calls(tool_calls: Any) -> list[Any]:
    if isinstance(tool_calls, list):
        return [getattr(tool_call, "__dict__", tool_call) for tool_call in tool_calls]
    return []


def _trace_usage(usage: Any | None) -> dict[str, Any] | None:
    if usage is None:
        return None
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }
