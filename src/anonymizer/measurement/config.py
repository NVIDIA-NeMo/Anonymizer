# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict, SettingsError

from anonymizer.measurement.collector import MeasurementCollector
from anonymizer.measurement.constants import DD_TRACE_MODES, DEFAULT_MEASUREMENT_ENV_PREFIX, DDTraceMode
from anonymizer.measurement.sinks import _writer_for_format


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
    dd_task_trace_path: str | None = None
    fail_on_write_error: bool = False
    run_id: str | None = None
    run_tags: dict[str, Any] = Field(default_factory=dict)


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
    dd_task_trace_path: str | Path | None = None
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
            dd_task_trace_path=settings.dd_task_trace_path,
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
