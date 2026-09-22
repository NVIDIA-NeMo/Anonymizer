# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded batching and protected ATOF export outside the Relay worker."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from nemo_anonymizer_relay.projection import (
    EXPORT_SANITIZE_FIELDS,
    omitted_event,
    prepare_event_for_export,
    requires_sanitization,
)

Json = Any


class EventSink(Protocol):
    def write(self, events: list[dict[str, Json]]) -> None: ...


class EventSanitizer(Protocol):
    async def sanitize_export(self, value: Json) -> Json: ...


@dataclass(frozen=True)
class QueuedEvent:
    sequence: int
    event: dict[str, Json]
    encoded_bytes: int
    accepted_at: float


@dataclass
class ExportStats:
    accepted: int = 0
    rejected_full: int = 0
    rejected_oversize: int = 0
    rejected_failed: int = 0
    batches: int = 0
    exported: int = 0
    omitted: int = 0
    payloads_omitted: int = 0
    backend_failures: int = 0
    sink_failures: int = 0
    metrics_failures: int = 0
    dropped_after_failure: int = 0
    input_bytes: int = 0
    output_bytes: int = 0
    backend_seconds: float = 0.0
    max_queue_depth: int = 0
    max_queued_bytes: int = 0
    max_event_lag_seconds: float = 0.0


class JsonlSink:
    """Append sanitized events only; create the destination as owner-readable."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        os.fchmod(descriptor, 0o600)
        self._file = os.fdopen(descriptor, "a", encoding="utf-8")

    def write(self, events: list[dict[str, Json]]) -> None:
        for event in events:
            self._file.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
            self._file.write("\n")
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self) -> None:
        self._file.close()


class ProtectedExporter:
    """Accept copied events quickly, then sanitize and export serial batches."""

    def __init__(
        self,
        sanitizer: EventSanitizer,
        sink: EventSink,
        *,
        queue_capacity: int = 2048,
        queue_max_bytes: int = 64 * 1024 * 1024,
        max_event_bytes: int = 2 * 1024 * 1024,
        batch_max_events: int = 8,
        batch_max_bytes: int = 2 * 1024 * 1024,
        batch_wait_seconds: float = 0.25,
        metrics_path: Path | None = None,
    ) -> None:
        if (
            min(
                queue_capacity,
                queue_max_bytes,
                max_event_bytes,
                batch_max_events,
                batch_max_bytes,
            )
            <= 0
        ):
            raise ValueError("exporter limits must be positive")
        if batch_wait_seconds < 0:
            raise ValueError("batch_wait_seconds must be non-negative")
        self._sanitizer = sanitizer
        self._sink = sink
        self._queue: asyncio.Queue[QueuedEvent | None] = asyncio.Queue(queue_capacity)
        self._queue_max_bytes = queue_max_bytes
        self._max_event_bytes = max_event_bytes
        self._batch_max_events = batch_max_events
        self._batch_max_bytes = batch_max_bytes
        self._batch_wait_seconds = batch_wait_seconds
        self._metrics_path = metrics_path
        self._stats = ExportStats()
        self._task: asyncio.Task[None] | None = None
        self._idle = asyncio.Event()
        self._idle.set()
        self._started_at = time.monotonic()
        self._next_sequence = 1
        self._accepting = True
        self._queued_bytes = 0
        self._fatal_error: str | None = None

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("exporter already started")
        self._task = asyncio.create_task(self._run(), name="nemo-anonymizer-exporter")

    def quiesce(self) -> None:
        """Stop new admission before draining already accepted events."""

        self._accepting = False

    def enqueue(self, event: dict[str, Json], encoded_bytes: int) -> dict[str, Json]:
        if self._fatal_error is not None:
            self._stats.rejected_failed += 1
            return {"status": "failed", "queue_depth": self.queue_depth}
        if not self._accepting:
            return {"status": "stopping", "queue_depth": self.queue_depth}
        if encoded_bytes > self._max_event_bytes:
            self._stats.rejected_oversize += 1
            self._write_metrics_safely()
            return {"status": "oversize", "queue_depth": self.queue_depth}
        if self._queued_bytes + encoded_bytes > self._queue_max_bytes:
            self._stats.rejected_full += 1
            self._write_metrics_safely()
            return {"status": "full", "queue_depth": self.queue_depth}
        item = QueuedEvent(
            sequence=self._next_sequence,
            event=event,
            encoded_bytes=encoded_bytes,
            accepted_at=time.monotonic(),
        )
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            self._stats.rejected_full += 1
            self._write_metrics_safely()
            return {"status": "full", "queue_depth": self.queue_depth}
        self._next_sequence += 1
        self._stats.accepted += 1
        self._stats.input_bytes += encoded_bytes
        self._queued_bytes += encoded_bytes
        self._stats.max_queue_depth = max(self._stats.max_queue_depth, self.queue_depth)
        self._stats.max_queued_bytes = max(self._stats.max_queued_bytes, self._queued_bytes)
        self._idle.clear()
        return {
            "status": "accepted",
            "sequence": item.sequence,
            "queue_depth": self.queue_depth,
        }

    async def flush(self) -> None:
        await self._queue.join()
        await self._idle.wait()
        if self._fatal_error is not None:
            raise RuntimeError(f"exporter failed: {self._fatal_error}")

    async def close(self) -> None:
        self.quiesce()
        flush_error: Exception | None = None
        try:
            await self.flush()
        except Exception as error:
            flush_error = error
        finally:
            await self._queue.put(None)
            if self._task is not None:
                await self._task
            close = getattr(self._sink, "close", None)
            if close is not None:
                close()
        if flush_error is not None:
            raise flush_error

    def snapshot(self) -> dict[str, Json]:
        result = asdict(self._stats)
        result.update(
            {
                "queue_depth": self.queue_depth,
                "queued_bytes": self._queued_bytes,
                "accepting": self._accepting,
                "busy": not self._idle.is_set(),
                "fatal_error": self._fatal_error,
                "uptime_seconds": round(time.monotonic() - self._started_at, 6),
            }
        )
        result["backend_seconds"] = round(self._stats.backend_seconds, 6)
        result["max_event_lag_seconds"] = round(self._stats.max_event_lag_seconds, 6)
        return result

    async def _run(self) -> None:
        pending: QueuedEvent | None = None
        while True:
            first = pending if pending is not None else await self._queue.get()
            pending = None
            if first is None:
                self._queue.task_done()
                return
            batch = [first]
            batch_bytes = first.encoded_bytes
            if self._batch_wait_seconds:
                await asyncio.sleep(self._batch_wait_seconds)
            while len(batch) < self._batch_max_events:
                try:
                    item = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is None:
                    self._queue.task_done()
                    self._accepting = False
                    break
                if batch and batch_bytes + item.encoded_bytes > self._batch_max_bytes:
                    pending = item
                    break
                batch.append(item)
                batch_bytes += item.encoded_bytes
            try:
                await self._process(batch)
            except Exception as error:
                self._fatal_error = type(error).__name__
                self._accepting = False
                self._stats.sink_failures += 1
                self._stats.dropped_after_failure += len(batch)
            finally:
                for item in batch:
                    self._queued_bytes -= item.encoded_bytes
                    self._queue.task_done()
            if self._fatal_error is not None:
                if pending is not None:
                    self._queued_bytes -= pending.encoded_bytes
                    self._stats.dropped_after_failure += 1
                    self._queue.task_done()
                    pending = None
                self._discard_queued_after_failure()
            if pending is None and self._queue.empty():
                self._idle.set()
            self._write_metrics_safely()

    async def _process(self, batch: list[QueuedEvent]) -> None:
        projected: list[dict[str, Json]] = []
        try:
            projected = [prepare_event_for_export(item.event) for item in batch]
            started = time.perf_counter()
            selected = [(index, event) for index, event in enumerate(projected) if requires_sanitization(event)]
            payloads = [{field: event.get(field) for field in EXPORT_SANITIZE_FIELDS} for _, event in selected]
            try:
                sanitized = await self._sanitizer.sanitize_export(payloads)
            finally:
                self._stats.backend_seconds += time.perf_counter() - started
            if not isinstance(sanitized, list) or len(sanitized) != len(selected):
                raise RuntimeError("sanitizer returned the wrong number of events")
            output = projected
            for (index, event), payload in zip(selected, sanitized, strict=True):
                if not isinstance(payload, dict):
                    raise RuntimeError("sanitizer returned a malformed event payload")
                for field in EXPORT_SANITIZE_FIELDS:
                    event[field] = payload.get(field)
                output[index] = event
            self._stats.payloads_omitted += len(projected) - len(selected)
        except Exception as error:  # fail closed; never serialize the source events
            self._stats.backend_failures += 1
            reason = getattr(error, "safe_code", type(error).__name__)
            output = [omitted_event(item.event, reason) for item in batch]
            self._stats.omitted += len(batch)

        now = time.monotonic()
        for item, event in zip(batch, output, strict=True):
            metadata = event.setdefault("metadata", {})
            if isinstance(metadata, dict):
                metadata["nemo_anonymizer.sequence"] = item.sequence
            self._stats.max_event_lag_seconds = max(self._stats.max_event_lag_seconds, now - item.accepted_at)
        # Destination latency must not block event admission, health, or drain
        # requests on the service loop.
        await asyncio.to_thread(self._sink.write, output)
        self._stats.batches += 1
        self._stats.exported += len(output)
        self._stats.output_bytes += sum(
            len(json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) for event in output
        )

    def _discard_queued_after_failure(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if item is None:
                self._queue.task_done()
                continue
            self._queued_bytes -= item.encoded_bytes
            self._stats.dropped_after_failure += 1
            self._queue.task_done()

    def _write_metrics_safely(self) -> None:
        try:
            self._write_metrics()
        except OSError:
            self._stats.metrics_failures += 1

    def _write_metrics(self) -> None:
        if self._metrics_path is None:
            return
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._metrics_path.with_suffix(f"{self._metrics_path.suffix}.tmp")
        temporary.write_text(json.dumps(self.snapshot(), indent=2) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o600)
        temporary.replace(self._metrics_path)


__all__ = ["ExportStats", "JsonlSink", "ProtectedExporter"]
