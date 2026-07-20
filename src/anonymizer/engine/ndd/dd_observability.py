# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owns message/task/usage tracing of DataDesigner runs for the measurement
collector, including the temporary private model-facade monkeypatch shims.
"""

from __future__ import annotations

import importlib
import re
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict, TypeGuard, cast

from data_designer.config.column_configs import CustomColumnConfig, LLMStructuredColumnConfig, LLMTextColumnConfig
from data_designer.config.column_types import ColumnConfigT
from data_designer.config.models import ModelConfig
from data_designer.config.run_config import RunConfig
from data_designer.config.utils.constants import TRACE_COLUMN_POSTFIX
from data_designer.config.utils.trace_type import TraceType

if TYPE_CHECKING:
    import pandas as pd
    from data_designer.interface.data_designer import DataDesigner

__all__ = [
    "DDMessageTracePlan",
    "DataDesignerUsageProbe",
    "as_alias_list",
    "extract_workflow_model_aliases",
    "record_dd_task_traces",
    "task_traces_from_result",
    "temporary_dd_task_trace",
]


_TRACEABLE_LLM_COLUMN_TYPES = (LLMTextColumnConfig, LLMStructuredColumnConfig)


_MODEL_TRACE_COLUMN: ContextVar[str | None] = ContextVar("anonymizer_dd_model_trace_column", default=None)


_MODEL_TRACE_PURPOSE: ContextVar[str | None] = ContextVar("anonymizer_dd_model_trace_purpose", default=None)


@dataclass(frozen=True)
class _NativeTraceColumn:
    column_name: str
    trace_column_name: str
    model_alias: str | None
    model_name: str | None
    model_provider_name: str | None


@dataclass(frozen=True)
class _PrivateFacadeTraceColumn:
    column_name: str


@dataclass(frozen=True)
class DDMessageTracePlan:
    columns: list[ColumnConfigT]
    native_columns: list[_NativeTraceColumn]
    private_columns: list[_PrivateFacadeTraceColumn]
    unsupported_columns: list[ColumnConfigT]

    @classmethod
    def from_columns(
        cls,
        *,
        columns: list[ColumnConfigT],
        model_configs: list[ModelConfig],
        collector: Any | None,
    ) -> DDMessageTracePlan:
        if collector is None or not collector.dd_trace_enabled:
            return cls(columns=columns, native_columns=[], private_columns=[], unsupported_columns=[])

        model_configs_by_alias = {model_config.alias: model_config for model_config in model_configs}
        native_columns: list[_NativeTraceColumn] = []
        private_columns: list[_PrivateFacadeTraceColumn] = []
        unsupported_columns: list[ColumnConfigT] = []
        configured_columns: list[ColumnConfigT] = []

        for column in columns:
            if isinstance(column, _TRACEABLE_LLM_COLUMN_TYPES):
                configured_columns.append(
                    cast(ColumnConfigT, column.model_copy(update={"with_trace": cls.trace_type()}))
                )
                model_config = model_configs_by_alias.get(column.model_alias)
                native_columns.append(
                    _NativeTraceColumn(
                        column_name=column.name,
                        trace_column_name=f"{column.name}{TRACE_COLUMN_POSTFIX}",
                        model_alias=column.model_alias,
                        model_name=getattr(model_config, "model", None),
                        model_provider_name=getattr(model_config, "provider", None),
                    )
                )
                continue

            if _column_has_private_facade_model_calls(column):
                configured_columns.append(_custom_column_with_trace_context(column))
                private_columns.append(_PrivateFacadeTraceColumn(column_name=column.name))
                continue

            unsupported_columns.append(column)
            configured_columns.append(column)

        return cls(
            columns=configured_columns,
            native_columns=native_columns,
            private_columns=private_columns,
            unsupported_columns=unsupported_columns,
        )

    @staticmethod
    def trace_type() -> TraceType:
        # Preserve Anonymizer's existing dd_trace=last_message semantics: the trace
        # sink records the final prompt message and response separately, while DD's
        # native LAST_MESSAGE side effect only keeps the final assistant message.
        return TraceType.ALL_MESSAGES

    def record_coverage(self, *, workflow_name: str, collector: Any | None) -> None:
        if collector is None or not collector.dd_trace_enabled:
            return

        traced_column_names = [column.column_name for column in self.native_columns] + [
            column.column_name for column in self.private_columns
        ]
        collector.record(
            "dd_trace_coverage",
            workflow_name=workflow_name,
            trace_backend=self.backend,
            trace_mode=collector.dd_trace_mode,
            native_trace_type=self.trace_type().value,
            traced_column_count=len(traced_column_names),
            traced_column_names=traced_column_names,
            native_trace_column_count=len(self.native_columns),
            native_trace_column_names=[column.column_name for column in self.native_columns],
            private_trace_column_count=len(self.private_columns),
            private_trace_column_names=[column.column_name for column in self.private_columns],
            private_trace_backend="anonymizer_private_model_facade" if self.private_columns else None,
            private_trace_note=(
                "temporary private DataDesigner model registry/facade instrumentation" if self.private_columns else None
            ),
            unsupported_column_count=len(self.unsupported_columns),
            unsupported_column_names=[column.name for column in self.unsupported_columns],
            unsupported_column_types=[_column_type_name(column) for column in self.unsupported_columns],
        )

    @property
    def backend(self) -> str:
        if self.native_columns and self.private_columns:
            return "mixed"
        if self.private_columns:
            return "anonymizer_private_model_facade"
        return "data_designer_column"

    def record_and_strip_native_traces(
        self,
        *,
        output_df: pd.DataFrame,
        workflow_name: str,
        collector: Any | None,
    ) -> pd.DataFrame:
        if not self.native_columns:
            return output_df

        trace_column_names = [column.trace_column_name for column in self.native_columns]
        if collector is not None and collector.dd_trace_enabled:
            for _, row in output_df.iterrows():
                for trace_column in self.native_columns:
                    if trace_column.trace_column_name not in output_df.columns:
                        continue
                    self._record_native_trace(
                        trace_column=trace_column,
                        trace_value=row.get(trace_column.trace_column_name),
                        workflow_name=workflow_name,
                        collector=collector,
                    )

        existing_trace_columns = [column_name for column_name in trace_column_names if column_name in output_df.columns]
        if not existing_trace_columns:
            return output_df
        return output_df.drop(columns=existing_trace_columns)

    @staticmethod
    def _record_native_trace(
        *,
        trace_column: _NativeTraceColumn,
        trace_value: Any,
        workflow_name: str,
        collector: Any,
    ) -> None:
        trace_messages = _native_trace_messages(trace_value)
        if not trace_messages:
            return
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


class _TaskTraceLike(Protocol):
    column: Any
    row_group: Any
    row_index: Any
    task_type: Any
    status: Any
    error: Any
    dispatched_at: Any
    slot_acquired_at: Any
    completed_at: Any


_TaskTrace = Mapping[str, Any] | _TaskTraceLike


class _DDTaskTraceFields(TypedDict):
    workflow_name: str
    trace_source: Literal["data_designer_scheduler"]
    column: Any
    row_group: Any
    row_index: Any
    task_type: Any
    status: Any
    error_present: bool
    dispatched_offset_sec: float | None
    slot_acquired_offset_sec: float | None
    completed_offset_sec: float | None
    queue_wait_sec: float | None
    execution_sec: float | None
    total_sec: float | None


def extract_workflow_model_aliases(columns: list[ColumnConfigT]) -> list[str]:
    aliases: list[str] = []
    for column in columns:
        aliases.extend(as_alias_list(getattr(column, "model_alias", None)))
        generator = getattr(column, "generator_function", None)
        metadata = getattr(generator, "custom_column_metadata", None)
        if isinstance(metadata, dict):
            aliases.extend(as_alias_list(metadata.get("model_aliases")))
    return list(dict.fromkeys(alias for alias in aliases if alias))


def as_alias_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, (list, tuple, set)):
        return [str(item) for item in raw if item is not None and str(item)]
    return [str(raw)]


class DataDesignerUsageProbe:
    """Capture DataDesigner model usage from the per-run private ResourceProvider."""

    def __init__(
        self,
        data_designer: DataDesigner,
        *,
        enabled: bool,
        collector: Any | None = None,
        workflow_name: str | None = None,
        private_trace_columns: list[_PrivateFacadeTraceColumn] | None = None,
    ) -> None:
        self._data_designer = data_designer
        self._enabled = enabled
        self._collector = collector
        self._workflow_name = workflow_name
        self._private_trace_column_names = {column.column_name for column in private_trace_columns or []}
        self._original_create_resource_provider: Any | None = None
        self._resource_providers: list[Any] = []
        self._model_registry_patches: list[tuple[Any, Any]] = []
        self._facade_patches: dict[int, tuple[Any, dict[str, Any]]] = {}
        self._private_trace_records: list[dict[str, Any]] = []

    def __enter__(self) -> DataDesignerUsageProbe:
        if not self._enabled:
            return self

        original = getattr(self._data_designer, "_create_resource_provider", None)
        if not callable(original):
            return self

        self._original_create_resource_provider = original

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            resource_provider = original(*args, **kwargs)
            self._resource_providers.append(resource_provider)
            self._install_private_model_trace(resource_provider)
            return resource_provider

        setattr(self._data_designer, "_create_resource_provider", wrapper)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._restore_private_trace_patches()
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

    def flush_private_trace_records(self) -> None:
        collector = self._collector
        if collector is None:
            self._private_trace_records.clear()
            return
        while self._private_trace_records:
            collector.record_dd_message_trace(**self._private_trace_records.pop(0))

    def _private_trace_enabled(self) -> bool:
        return bool(
            self._collector is not None
            and self._collector.dd_trace_enabled
            and self._workflow_name
            and self._private_trace_column_names
        )

    def _install_private_model_trace(self, resource_provider: Any) -> None:
        if not self._private_trace_enabled():
            return
        model_registry = getattr(resource_provider, "model_registry", None)
        get_model = getattr(model_registry, "get_model", None)
        if not callable(get_model):
            return

        def wrapped_get_model(*args: Any, **kwargs: Any) -> Any:
            facade = get_model(*args, **kwargs)
            self._patch_model_facade(facade)
            return facade

        # Temporary private DataDesigner shim: CustomColumnConfig receives
        # ModelFacade objects directly and DD does not yet expose a public
        # model-call event sink for those calls.
        setattr(model_registry, "get_model", wrapped_get_model)
        self._model_registry_patches.append((model_registry, get_model))

    def _patch_model_facade(self, facade: Any) -> None:
        facade_id = id(facade)
        if facade_id in self._facade_patches:
            return

        originals: dict[str, Any] = {}
        for method_name in ("completion", "acompletion", "generate", "agenerate"):
            method = getattr(facade, method_name, None)
            if not callable(method):
                continue
            originals[method_name] = method
            setattr(facade, method_name, self._wrap_facade_method(facade, method_name, method))

        if originals:
            self._facade_patches[facade_id] = (facade, originals)

    def _wrap_facade_method(self, facade: Any, method_name: str, method: Any) -> Any:
        if method_name == "acompletion":
            return self._wrap_async_completion(facade, method)
        if method_name == "completion":
            return self._wrap_completion(facade, method)
        if method_name == "agenerate":
            return self._wrap_async_generate(method)
        return self._wrap_generate(method)

    def _wrap_generate(self, method: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            token = _MODEL_TRACE_PURPOSE.set(_purpose_from_kwargs(kwargs))
            try:
                return method(*args, **kwargs)
            finally:
                _MODEL_TRACE_PURPOSE.reset(token)

        return wrapper

    def _wrap_async_generate(self, method: Any) -> Any:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            token = _MODEL_TRACE_PURPOSE.set(_purpose_from_kwargs(kwargs))
            try:
                return await method(*args, **kwargs)
            finally:
                _MODEL_TRACE_PURPOSE.reset(token)

        return wrapper

    def _wrap_completion(self, facade: Any, method: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            error: Exception | None = None
            response: Any = None
            try:
                response = method(*args, **kwargs)
                return response
            except Exception as exc:
                error = exc
                raise
            finally:
                self._record_private_completion_trace(facade, args, kwargs, started, response, error, is_async=False)

        return wrapper

    def _wrap_async_completion(self, facade: Any, method: Any) -> Any:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            error: Exception | None = None
            response: Any = None
            try:
                response = await method(*args, **kwargs)
                return response
            except Exception as exc:
                error = exc
                raise
            finally:
                self._record_private_completion_trace(facade, args, kwargs, started, response, error, is_async=True)

        return wrapper

    def _record_private_completion_trace(
        self,
        facade: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        started: float,
        response: Any,
        error: Exception | None,
        *,
        is_async: bool,
    ) -> None:
        if not self._private_trace_enabled():
            return
        column_name = _private_trace_column_name(
            column_names=self._private_trace_column_names,
            purpose=_purpose_from_kwargs(kwargs) or _MODEL_TRACE_PURPOSE.get(),
        )
        if column_name is None:
            return
        collector = self._collector
        if collector is None:
            return
        self._private_trace_records.append(
            _private_completion_trace_fields(
                workflow_name=self._workflow_name,
                column_name=column_name,
                facade=facade,
                args=args,
                kwargs=kwargs,
                response=response,
                error=error,
                elapsed_sec=time.perf_counter() - started,
                is_async=is_async,
                trace_mode=collector.dd_trace_mode,
            )
        )

    def _restore_private_trace_patches(self) -> None:
        for facade, originals in reversed(list(self._facade_patches.values())):
            for method_name, original in originals.items():
                setattr(facade, method_name, original)
        self._facade_patches.clear()

        for model_registry, get_model in reversed(self._model_registry_patches):
            setattr(model_registry, "get_model", get_model)
        self._model_registry_patches.clear()


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


def _purpose_from_kwargs(kwargs: Mapping[str, Any]) -> str | None:
    purpose = kwargs.get("purpose")
    return purpose if isinstance(purpose, str) and purpose else None


def _private_trace_column_name(*, column_names: set[str], purpose: str | None) -> str | None:
    context_column = _MODEL_TRACE_COLUMN.get()
    if context_column in column_names:
        return context_column

    task_column = _runtime_correlation_task_column()
    if task_column in column_names:
        return task_column

    purpose_column = _column_name_from_purpose(purpose)
    if purpose_column in column_names:
        return purpose_column

    if len(column_names) == 1:
        return next(iter(column_names))
    return None


def _runtime_correlation_task_column() -> str | None:
    try:
        observability = importlib.import_module("data_designer.engine.observability")
    except Exception:
        return None

    runtime_correlation_provider = getattr(observability, "runtime_correlation_provider", None)
    current = getattr(runtime_correlation_provider, "current", None)
    if not callable(current):
        return None
    correlation = current()
    task_column = getattr(correlation, "task_column", None)
    return task_column if isinstance(task_column, str) and task_column else None


def _column_name_from_purpose(purpose: str | None) -> str | None:
    if not purpose:
        return None
    match = re.search(r"column '([^']+)'", purpose)
    if match:
        return match.group(1)
    return None


def _model_provider_endpoint(facade: Any) -> str | None:
    provider = getattr(facade, "model_provider", None)
    endpoint = getattr(provider, "endpoint", None)
    return endpoint if isinstance(endpoint, str) and endpoint else None


def _private_trace_messages(*, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> list[dict[str, Any]]:
    messages = args[0] if args else kwargs.get("messages")
    if isinstance(messages, list):
        return [_trace_message(message) for message in messages]
    return []


def _private_completion_trace_fields(
    *,
    workflow_name: str | None,
    column_name: str,
    facade: Any,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    response: Any,
    error: Exception | None,
    elapsed_sec: float,
    is_async: bool,
    trace_mode: str,
) -> dict[str, Any]:
    return {
        "workflow_name": workflow_name,
        "trace_source": "anonymizer_private_model_facade",
        "column_name": column_name,
        "trace_column_name": None,
        "model_alias": getattr(facade, "model_alias", None),
        "model_name": getattr(facade, "model_name", None),
        "model_provider_name": getattr(facade, "model_provider_name", None),
        "model_provider_endpoint": _model_provider_endpoint(facade),
        "modality": "chat",
        "is_async": is_async,
        "status": "error" if error is not None else "completed",
        "error_type": type(error).__name__ if error is not None else None,
        "elapsed_sec": elapsed_sec,
        "messages": _select_native_trace_messages(_private_trace_messages(args=args, kwargs=kwargs), mode=trace_mode),
        "response": _model_trace_response(response),
        "usage": _model_trace_usage(response),
    }


def _model_trace_response(response: Any) -> dict[str, Any] | None:
    message = getattr(response, "message", None)
    if message is None:
        return None
    return {
        "content": getattr(message, "content", None),
        "reasoning_content": getattr(message, "reasoning_content", None),
        "tool_calls": _trace_tool_calls(getattr(message, "tool_calls", [])),
    }


def _model_trace_usage(response: Any) -> Any:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    model_dump = getattr(usage, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if isinstance(usage, Mapping):
        return dict(usage)
    fields = ("input_tokens", "output_tokens", "total_tokens", "reasoning_tokens")
    payload = {field: getattr(usage, field) for field in fields if getattr(usage, field, None) is not None}
    return payload or None


@contextmanager
def temporary_dd_task_trace(data_designer: DataDesigner, *, collector: Any | None) -> Iterator[None]:
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


def task_traces_from_result(result: Any) -> list[_TaskTrace]:
    raw_traces = getattr(result, "task_traces", None)
    if raw_traces is None:
        return []
    if isinstance(raw_traces, list):
        return cast(list[_TaskTrace], raw_traces)
    try:
        return cast(list[_TaskTrace], list(raw_traces))
    except TypeError:
        return []


def _custom_column_with_trace_context(column: CustomColumnConfig) -> ColumnConfigT:
    generator = column.generator_function

    @wraps(generator)
    def traced_generator(*args: Any, **kwargs: Any) -> Any:
        token = _MODEL_TRACE_COLUMN.set(column.name)
        try:
            return generator(*args, **kwargs)
        finally:
            _MODEL_TRACE_COLUMN.reset(token)

    traced_generator.custom_column_metadata = getattr(generator, "custom_column_metadata", {})  # type: ignore[attr-defined]
    return cast(ColumnConfigT, column.model_copy(update={"generator_function": traced_generator}))


def _column_has_private_facade_model_calls(column: ColumnConfigT) -> TypeGuard[CustomColumnConfig]:
    return isinstance(column, CustomColumnConfig) and bool(extract_workflow_model_aliases([column]))


def _column_type_name(column: ColumnConfigT) -> str:
    column_type = getattr(column, "column_type", None)
    return str(column_type) if column_type is not None else type(column).__name__


def _native_trace_messages(value: Any) -> list[dict[str, Any]]:
    if value is None or isinstance(value, (str, bytes, Mapping)):
        return []
    try:
        messages = list(value)
    except TypeError:
        return []
    return [_trace_message(message) for message in messages]


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


def record_dd_task_traces(*, workflow_name: str, collector: Any | None, task_traces: list[_TaskTrace]) -> None:
    if collector is None or not collector.dd_task_trace_enabled:
        return
    trace_origin = _task_trace_origin(task_traces)
    for task_trace in task_traces:
        collector.record_dd_task_trace(**_dd_task_trace_fields(workflow_name, task_trace, trace_origin))


def _dd_task_trace_fields(
    workflow_name: str,
    task_trace: _TaskTrace,
    trace_origin: float | None,
) -> _DDTaskTraceFields:
    dispatched_at = _trace_attr(task_trace, "dispatched_at")
    slot_acquired_at = _trace_attr(task_trace, "slot_acquired_at")
    completed_at = _trace_attr(task_trace, "completed_at")
    return {
        "workflow_name": workflow_name,
        "trace_source": "data_designer_scheduler",
        "column": _trace_attr(task_trace, "column"),
        "row_group": _trace_attr(task_trace, "row_group"),
        "row_index": _trace_attr(task_trace, "row_index"),
        "task_type": _trace_attr(task_trace, "task_type"),
        "status": _trace_attr(task_trace, "status"),
        "error_present": bool(_trace_attr(task_trace, "error")),
        "dispatched_offset_sec": _trace_offset(trace_origin, dispatched_at),
        "slot_acquired_offset_sec": _trace_offset(trace_origin, slot_acquired_at),
        "completed_offset_sec": _trace_offset(trace_origin, completed_at),
        "queue_wait_sec": _trace_duration(dispatched_at, slot_acquired_at),
        "execution_sec": _trace_duration(slot_acquired_at, completed_at),
        "total_sec": _trace_duration(dispatched_at, completed_at),
    }


def _task_trace_origin(task_traces: list[_TaskTrace]) -> float | None:
    dispatch_times: list[float] = []
    for task_trace in task_traces:
        dispatched_at = _trace_attr(task_trace, "dispatched_at")
        if isinstance(dispatched_at, (int, float)) and dispatched_at > 0:
            dispatch_times.append(float(dispatched_at))
    return min(dispatch_times) if dispatch_times else None


def _trace_attr(task_trace: _TaskTrace, name: str) -> Any:
    if isinstance(task_trace, Mapping):
        return cast(Mapping[str, Any], task_trace).get(name)
    return getattr(task_trace, name, None)


def _trace_offset(origin: float | None, timestamp: Any) -> float | None:
    if origin is None or not isinstance(timestamp, (int, float)):
        return None
    if timestamp <= 0 or timestamp < origin:
        return None
    return float(timestamp - origin)


def _trace_duration(start: Any, end: Any) -> float | None:
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return None
    if start <= 0 or end <= 0 or end < start:
        return None
    return float(end - start)
