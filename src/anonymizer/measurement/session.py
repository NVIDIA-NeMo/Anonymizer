# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from anonymizer.measurement.collector import MeasurementCollector
from anonymizer.measurement.config import MeasurementConfig
from anonymizer.measurement.sinks import _JsonlMeasurementSink

logger = logging.getLogger("anonymizer.measurement")

_ACTIVE_COLLECTOR: ContextVar[MeasurementCollector | None] = ContextVar(
    "anonymizer_measurement_collector",
    default=None,
)


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
    dd_trace_sink = None
    if config.dd_trace != "none":
        if config.dd_trace_path is None:
            raise ValueError("dd_trace_path is required when dd_trace is enabled")
        dd_trace_sink = _JsonlMeasurementSink(config.dd_trace_path)
    dd_task_trace_sink = _JsonlMeasurementSink(config.dd_task_trace_path) if config.dd_task_trace_path else None
    collector = MeasurementCollector(
        run_id=config.run_id,
        record_hash_key=config.record_hash_key,
        record_level=config.record_level,
        run_tags=config.run_tags,
        record_sink=sink,
        keep_records=config.keep_records,
        dd_trace_mode=config.dd_trace,
        dd_trace_sink=dd_trace_sink,
        dd_task_trace_sink=dd_task_trace_sink,
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
                write_error: BaseException | None = None
                try:
                    _write_collector_safely(config=config, collector=collector, body_error=body_error)
                except BaseException as exc:
                    write_error = exc
                    raise
                finally:
                    _close_collector_safely(
                        config=config,
                        collector=collector,
                        body_error=body_error or write_error,
                    )


def current_collector() -> MeasurementCollector | None:
    """Return the active collector, if measurement is enabled."""
    return _ACTIVE_COLLECTOR.get()


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
