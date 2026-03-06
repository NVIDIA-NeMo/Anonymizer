# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logging utilities for the Anonymizer package."""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass

LOG_INDENT = "  |-- "

_DEFAULT_NOISY_LOGGERS = ["httpx", "httpcore", "mcp"]

_anonymizer_handler: logging.Handler | None = None


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration with preset factories."""

    anonymizer_level: str = "INFO"
    data_designer_level: str = "WARNING"

    @classmethod
    def default(cls) -> LoggingConfig:
        return cls(anonymizer_level="INFO", data_designer_level="WARNING")

    @classmethod
    def verbose(cls) -> LoggingConfig:
        return cls(anonymizer_level="INFO", data_designer_level="INFO")

    @classmethod
    def debug(cls) -> LoggingConfig:
        return cls(anonymizer_level="DEBUG", data_designer_level="DEBUG")


def configure_logging(
    config: LoggingConfig | None = None,
    *,
    verbose: bool = False,
) -> None:
    """Set up logging for Anonymizer.

    Args:
        config: Logging preset. Defaults to ``LoggingConfig.default()``.
        verbose: Deprecated convenience flag. ``True`` maps to
            ``LoggingConfig.verbose()``. Ignored when *config* is provided.
    """
    if config is None:
        config = LoggingConfig.verbose() if verbose else LoggingConfig.default()

    global _anonymizer_handler

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Replace only our own handler to avoid removing handlers added by
    # test infrastructure (e.g. pytest caplog) or other libraries.
    if _anonymizer_handler is not None:
        root.removeHandler(_anonymizer_handler)

    _anonymizer_handler = logging.StreamHandler(sys.stderr)
    _anonymizer_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S"))
    root.addHandler(_anonymizer_handler)

    logging.getLogger("anonymizer").setLevel(config.anonymizer_level)
    logging.getLogger("data_designer").setLevel(config.data_designer_level)

    for name in _DEFAULT_NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


_PROGRESS_THRESHOLD = 50

_progress_logger = logging.getLogger("anonymizer")


class ProgressTracker:
    """Log-based progress tracker for sequential record processing."""

    def __init__(self, total: int, label: str, log_interval_percent: int = 10) -> None:
        self.total = total
        self.label = label
        self.completed = 0
        self.failed = 0

        interval_fraction = max(1, log_interval_percent) / 100.0
        self.log_interval = max(1, int(total * interval_fraction)) if total > 0 else 1
        self.next_log_at = self.log_interval

        self.start_time = time.perf_counter()
        self._enabled = total >= _PROGRESS_THRESHOLD

    def record_success(self) -> None:
        self._record(success=True)

    def record_failure(self) -> None:
        self._record(success=False)

    def log_final(self) -> None:
        if self.completed > 0:
            self._log_progress()

    def _record(self, *, success: bool) -> None:
        self.completed += 1
        if not success:
            self.failed += 1
        if self._enabled and self.completed >= self.next_log_at and self.completed < self.total:
            self._log_progress()
            while self.next_log_at <= self.completed:
                self.next_log_at += self.log_interval

    def _log_progress(self) -> None:
        elapsed = time.perf_counter() - self.start_time
        rate = self.completed / elapsed if elapsed > 0 else 0.0
        remaining = max(0, self.total - self.completed)
        eta = f"{(remaining / rate):.1f}s" if rate > 0 else "unknown"
        percent = (self.completed / self.total) * 100 if self.total else 100.0

        _progress_logger.info(
            "%s📊 %s progress: %d/%d (%.0f%%) — %d failed, %.1f rec/s, eta %s",
            LOG_INDENT,
            self.label,
            self.completed,
            self.total,
            percent,
            self.failed,
            rate,
            eta,
        )
