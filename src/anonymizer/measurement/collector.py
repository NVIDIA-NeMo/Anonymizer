# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from anonymizer.measurement._coerce import _json_safe
from anonymizer.measurement.constants import MEASUREMENT_SCHEMA_VERSION, DDTraceMode
from anonymizer.measurement.sinks import _JsonlMeasurementWriter, _JsonMeasurementWriter, _MeasurementSink

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger("anonymizer.measurement")


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
        dd_task_trace_sink: _MeasurementSink | None = None,
        fail_on_write_error: bool = False,
    ) -> None:
        self.run_id = run_id or uuid.uuid4().hex
        self.record_level = record_level
        self.run_tags = cast(dict[str, Any], _json_safe(dict(run_tags or {})))
        self._record_sink = record_sink
        self._keep_records = keep_records
        self._dd_trace_mode = dd_trace_mode
        self._dd_trace_sink = dd_trace_sink
        self._dd_task_trace_sink = dd_task_trace_sink
        self._fail_on_write_error = fail_on_write_error
        self._sink_failed = False
        self._dd_trace_failed = False
        self._dd_task_trace_failed = False
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
        close_error: Exception | None = None
        for sink in (self._record_sink, self._dd_trace_sink, self._dd_task_trace_sink):
            if sink is None:
                continue
            try:
                sink.close()
            except Exception as exc:
                if close_error is None:
                    close_error = exc
        if close_error is not None:
            raise close_error

    @property
    def dd_trace_mode(self) -> DDTraceMode:
        return self._dd_trace_mode

    @property
    def dd_trace_enabled(self) -> bool:
        return self._dd_trace_mode != "none" and self._dd_trace_sink is not None

    @property
    def dd_task_trace_enabled(self) -> bool:
        return self._dd_task_trace_sink is not None

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

    def record_dd_task_trace(self, **fields: Any) -> None:
        """Write an opt-in sanitized DataDesigner scheduler task trace record."""
        if not self.dd_task_trace_enabled or self._dd_task_trace_failed:
            return

        record = _json_safe(
            {
                **fields,
                "schema_version": MEASUREMENT_SCHEMA_VERSION,
                "record_type": "dd_task_trace",
                "run_id": self.run_id,
                "run_tags": self.run_tags,
                "timestamp_unix_sec": time.time(),
            }
        )
        try:
            cast(_MeasurementSink, self._dd_task_trace_sink).write_record(record)
        except Exception:
            self._dd_task_trace_failed = True
            logger.warning("Failed to write DataDesigner task trace records")
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
