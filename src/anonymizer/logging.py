# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logging utilities for the Anonymizer package."""

from __future__ import annotations

import logging
import sys
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
