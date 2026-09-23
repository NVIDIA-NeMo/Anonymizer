# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter from copied Relay event text to NeMo Anonymizer's typed API."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, order=True)
class RedactionSpan:
    start: int
    end: int
    label: str


class BackendResultError(RuntimeError):
    """A safe, text-free adapter failure suitable for exported diagnostics."""

    def __init__(self, code: str) -> None:
        self.safe_code = code
        super().__init__(code)


@dataclass(frozen=True)
class _Placement:
    text_index: int
    start: int
    end: int


@dataclass(frozen=True)
class _PackedRecord:
    text: str
    placements: tuple[_Placement, ...]


_FIELD_SEPARATOR = "\n\n[RELAY OBSERVABILITY FIELD]\n\n"
_MAX_PACKED_RECORD_CHARS = 16_000


@dataclass(frozen=True)
class BackendConfig:
    detector_endpoint: str
    detector_model: str
    detector_api_key_env: str
    evaluator_endpoint: str
    evaluator_model: str
    evaluator_api_key_env: str
    threshold: float
    timeout_seconds: int
    data_summary: str | None


@dataclass(frozen=True)
class _BackendRuntime:
    api: Any
    anonymizer: Any
    temporary_directory: tempfile.TemporaryDirectory[str]
    artifact_path: Path


class AnonymizerBackend:
    """Run one reusable Anonymizer pipeline and return validated entity spans."""

    def __init__(self, config: BackendConfig) -> None:
        self._config = config
        self._runtime: _BackendRuntime | None = None
        self._lock = threading.Lock()
        self._closed = False

    def detect(self, texts: list[str]) -> list[list[RedactionSpan]]:
        if not texts:
            return []

        with self._lock:
            if self._closed:
                raise BackendResultError("anonymizer_backend_closed")
            runtime: _BackendRuntime | None = None
            packed = _pack_texts(texts)
            try:
                runtime = self._get_runtime()
                api = runtime.api
                records = [api.TextRecord(id=str(index), text=record.text) for index, record in enumerate(packed)]
                result = runtime.anonymizer.run(
                    config=api.AnonymizerConfig(
                        detect=api.Detect(gliner_threshold=self._config.threshold),
                        replace=api.Redact(),
                        emit_telemetry=False,
                    ),
                    data=api.TextRecordsInput(records=records, data_summary=self._config.data_summary),
                )
                if result.failed_records:
                    raise BackendResultError("anonymizer_record_failure")
                if result.id_column is None:
                    raise BackendResultError("missing_result_id_column")
                return _spans_from_result(
                    result.dataframe,
                    packed,
                    len(texts),
                    id_column=result.id_column,
                    text_column=result.resolved_text_column,
                )
            except BackendResultError:
                raise
            except Exception:
                # Provider and parser exceptions may contain source text. Only
                # a fixed code is allowed to cross the worker RPC boundary.
                raise BackendResultError("anonymizer_runtime_failure") from None
            finally:
                if runtime is not None:
                    self._clear_artifacts(runtime)

    def close(self) -> None:
        """Release the private artifact directory after active detection ends."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            runtime = self._runtime
            self._runtime = None
            if runtime is not None:
                try:
                    runtime.temporary_directory.cleanup()
                except Exception:
                    raise BackendResultError("anonymizer_artifact_cleanup_failure") from None

    def _get_runtime(self) -> _BackendRuntime:
        runtime = self._runtime
        if runtime is not None:
            return runtime

        # Importing Anonymizer is expensive. Keep it off plugin discovery and
        # config-validation paths and initialize it on the first copied event.
        import anonymizer as api

        config = self._config
        providers = [
            api.ModelProvider(
                name="relay-gliner",
                endpoint=config.detector_endpoint,
                provider_type="openai",
                api_key=config.detector_api_key_env,
            ),
            api.ModelProvider(
                name="relay-evaluator",
                endpoint=config.evaluator_endpoint,
                provider_type="openai",
                api_key=config.evaluator_api_key_env,
            ),
        ]
        directory = tempfile.TemporaryDirectory(prefix="nemo-relay-anonymizer-")
        artifact_path = Path(directory.name) / "artifacts"
        try:
            runtime = _BackendRuntime(
                api=api,
                anonymizer=api.Anonymizer(
                    model_configs=self._model_config_json(),
                    model_providers=providers,
                    artifact_path=artifact_path,
                ),
                temporary_directory=directory,
                artifact_path=artifact_path,
            )
        except Exception:
            directory.cleanup()
            raise
        self._runtime = runtime
        return runtime

    @staticmethod
    def _clear_artifacts(runtime: _BackendRuntime) -> None:
        try:
            if runtime.artifact_path.exists():
                shutil.rmtree(runtime.artifact_path)
        except Exception:
            # Artifacts can contain copied source text. Treat retention as a
            # failed sanitization rather than silently accumulating them.
            raise BackendResultError("anonymizer_artifact_cleanup_failure") from None

    def _model_config_json(self) -> str:
        config = self._config
        evaluator_parameters: dict[str, Any] = {
            "max_parallel_requests": 8,
            "max_tokens": 8192,
            "temperature": 0.0,
            "timeout": config.timeout_seconds,
        }
        if "nemotron" in config.evaluator_model.lower():
            evaluator_parameters["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        return json.dumps(
            {
                "model_configs": [
                    {
                        "alias": "relay-gliner",
                        "model": config.detector_model,
                        "provider": "relay-gliner",
                        "skip_health_check": True,
                        "inference_parameters": {
                            "max_parallel_requests": 8,
                            "timeout": config.timeout_seconds,
                        },
                    },
                    {
                        "alias": "relay-evaluator",
                        "model": config.evaluator_model,
                        "provider": "relay-evaluator",
                        # Avoid a paid generation probe every time an event
                        # constructs an Anonymizer pipeline. Runtime provider
                        # failures become fail-closed omission records.
                        "skip_health_check": True,
                        "inference_parameters": evaluator_parameters,
                    },
                ],
                "selected_models": {
                    "detection": {
                        "entity_detector": "relay-gliner",
                        "entity_validator": ["relay-evaluator"],
                        "entity_augmenter": "relay-evaluator",
                        "latent_detector": "relay-evaluator",
                    }
                },
            }
        )


def _spans_from_result(
    dataframe: Any,
    packed: list[_PackedRecord],
    text_count: int,
    *,
    id_column: str,
    text_column: str,
) -> list[list[RedactionSpan]]:
    required = {id_column, text_column, "final_entities"}
    if not required.issubset(dataframe.columns):
        missing = sorted(required - set(dataframe.columns))
        raise BackendResultError(f"missing_result_columns_{'_'.join(missing)}")

    by_id: dict[int, list[RedactionSpan]] = {}
    decisions: list[list[RedactionSpan]] = [[] for _ in range(text_count)]
    for _, row in dataframe.iterrows():
        try:
            batch_id = int(row[id_column])
        except (TypeError, ValueError) as error:
            raise BackendResultError("invalid_batch_id") from error
        if batch_id in by_id or not 0 <= batch_id < len(packed):
            raise BackendResultError("duplicate_or_out_of_range_batch_id")
        record = packed[batch_id]
        if row[text_column] != record.text:
            raise BackendResultError("source_text_changed_or_reordered")
        packed_spans = _parse_spans(row["final_entities"], record.text)
        by_id[batch_id] = packed_spans
        for span in packed_spans:
            mapped = False
            for placement in record.placements:
                clipped_start = max(span.start, placement.start)
                clipped_end = min(span.end, placement.end)
                if clipped_start >= clipped_end:
                    continue
                mapped = True
                decisions[placement.text_index].append(
                    RedactionSpan(
                        start=clipped_start - placement.start,
                        end=clipped_end - placement.start,
                        label=span.label,
                    )
                )
            if not mapped:
                # A detector may label the fixed separator itself. It contains
                # no caller data, so dropping that decision is safe. A span
                # that crosses a separator is clipped into both adjacent
                # caller fields, which may over-redact but cannot leak it.
                continue

    if set(by_id) != set(range(len(packed))):
        raise BackendResultError("missing_input_batch")
    return decisions


def _pack_texts(texts: list[str], max_chars: int = _MAX_PACKED_RECORD_CHARS) -> list[_PackedRecord]:
    records: list[_PackedRecord] = []
    parts: list[str] = []
    placements: list[_Placement] = []
    length = 0

    def flush() -> None:
        nonlocal parts, placements, length
        if parts:
            records.append(_PackedRecord("".join(parts), tuple(placements)))
        parts = []
        placements = []
        length = 0

    for text_index, text in enumerate(texts):
        addition = len(text) + (len(_FIELD_SEPARATOR) if parts else 0)
        if parts and length + addition > max_chars:
            flush()
        if parts:
            parts.append(_FIELD_SEPARATOR)
            length += len(_FIELD_SEPARATOR)
        start = length
        parts.append(text)
        length += len(text)
        placements.append(_Placement(text_index=text_index, start=start, end=length))
    flush()
    return records


def _parse_spans(raw: Any, text: str) -> list[RedactionSpan]:
    if isinstance(raw, str):
        raw = json.loads(raw)
    entities = raw.get("entities") if isinstance(raw, dict) else None
    if entities is None and isinstance(raw, list):
        entities = raw
    if not isinstance(entities, list):
        raise BackendResultError("malformed_final_entities")

    spans: list[RedactionSpan] = []
    for entity in entities:
        if not isinstance(entity, dict):
            raise BackendResultError("malformed_entity")
        start = entity.get("start_position", entity.get("start"))
        end = entity.get("end_position", entity.get("end"))
        label = entity.get("label")
        value = entity.get("value", entity.get("text"))
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not isinstance(label, str)
            or start < 0
            or end <= start
            or end > len(text)
        ):
            raise BackendResultError("invalid_entity_span")
        if isinstance(value, str) and text[start:end] != value:
            # Anonymizer's detector parser trims the display value while
            # retaining the original standoff offsets. Treat offsets as
            # authoritative only for this documented whitespace difference.
            if text[start:end].strip() != value.strip():
                raise BackendResultError("entity_value_mismatch")
        spans.append(RedactionSpan(start=start, end=end, label=label))

    spans.sort()
    merged: list[RedactionSpan] = []
    for span in spans:
        if merged and span.start < merged[-1].end:
            previous = merged[-1]
            merged[-1] = RedactionSpan(
                start=previous.start,
                end=max(previous.end, span.end),
                label=previous.label,
            )
        else:
            merged.append(span)
    return merged


def require_api_keys(config: BackendConfig) -> None:
    """Fail activation if named provider credentials are absent."""

    key_names = [config.detector_api_key_env, config.evaluator_api_key_env]
    missing = [name for name in key_names if name != "EMPTY" and not os.getenv(name)]
    if missing:
        raise ValueError(f"missing required API key environment variable(s): {', '.join(missing)}")
