# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any, Literal, Protocol


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
        self._lock = Lock()

    def write_record(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._file.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")

    def close(self) -> None:
        with self._lock:
            self._file.close()
