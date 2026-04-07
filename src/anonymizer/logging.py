# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logging utilities for the Anonymizer package."""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass

LOG_INDENT = "  |-- "

_DEFAULT_NOISY_LOGGERS = [
    "httpx",
    "httpcore",
    "mcp",
    "litellm",
    "LiteLLM",
    "openai",
    "asyncio",
]

_anonymizer_handler: logging.Handler | None = None
_configured = False


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration with preset factories."""

    anonymizer_level: str = "INFO"
    data_designer_level: str = "WARNING"

    @classmethod
    def default(cls) -> LoggingConfig:
        """Anonymizer at INFO, Data Designer at WARNING."""
        return cls(anonymizer_level="INFO", data_designer_level="WARNING")

    @classmethod
    def verbose(cls) -> LoggingConfig:
        """Both Anonymizer and Data Designer at INFO."""
        return cls(anonymizer_level="INFO", data_designer_level="INFO")

    @classmethod
    def debug(cls) -> LoggingConfig:
        """Anonymizer at DEBUG, Data Designer at INFO."""
        return cls(anonymizer_level="DEBUG", data_designer_level="INFO")


def configure_logging(
    config: LoggingConfig | None = None,
    *,
    verbose: bool = False,
    enabled: bool = True,
) -> None:
    """Set up logging for Anonymizer.

    Args:
        config: Logging preset. Defaults to ``LoggingConfig.default()``.
        verbose: Deprecated convenience flag. ``True`` maps to
            ``LoggingConfig.verbose()``. Ignored when *config* is provided.
        enabled: Set to ``False`` to prevent Anonymizer from adding any log
            handlers. Useful when the caller manages logging independently.
    """
    global _anonymizer_handler, _configured
    _configured = True

    if not enabled:
        return

    if config is None:
        config = LoggingConfig.verbose() if verbose else LoggingConfig.default()

    anon_logger = logging.getLogger("anonymizer")
    dd_logger = logging.getLogger("data_designer")

    # Attach our handler to the anonymizer logger (not root) so we don't
    # clobber application-level logging configured via stdlib APIs.
    if _anonymizer_handler is not None and _anonymizer_handler in anon_logger.handlers:
        anon_logger.removeHandler(_anonymizer_handler)
    anon_logger.handlers = [h for h in anon_logger.handlers if type(h) is not logging.StreamHandler]

    _anonymizer_handler = logging.StreamHandler(sys.stderr)
    _anonymizer_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S"))
    anon_logger.addHandler(_anonymizer_handler)
    anon_logger.propagate = False
    anon_logger.setLevel(config.anonymizer_level)

    # DD logger gets its own handler so its messages are formatted consistently.
    dd_logger.handlers = [h for h in dd_logger.handlers if type(h) is not logging.StreamHandler]
    dd_handler = logging.StreamHandler(sys.stderr)
    dd_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S"))
    dd_logger.addHandler(dd_handler)
    dd_logger.propagate = False
    dd_logger.setLevel(config.data_designer_level)

    for name in _DEFAULT_NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


_PROGRESS_THRESHOLD = 50

_progress_logger = logging.getLogger("anonymizer")


class ProgressTracker:
    """Log-based progress tracker for sequential record processing."""

    def __init__(self, total: int, label: str, log_interval_percent: int = 10) -> None:
        """Create a progress tracker.

        Args:
            total: Total number of records to process.
            label: Human-readable label for the progress messages.
            log_interval_percent: How often to log, as a percentage of total.
        """
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
        """Record a successfully processed record and log progress if due."""
        self._record(success=True)

    def record_failure(self) -> None:
        """Record a failed record and log progress if due."""
        self._record(success=False)

    def log_final(self) -> None:
        """Emit a final progress line summarizing the run."""
        if self._enabled and self.completed > 0:
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
