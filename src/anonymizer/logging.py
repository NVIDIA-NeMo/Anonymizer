# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logging utilities for the Anonymizer package."""

from __future__ import annotations

import logging
import sys

LOG_INDENT = "  |-- "

_DEFAULT_NOISY_LOGGERS = ["httpx", "httpcore", "mcp"]

def configure_logging(*, verbose: bool = False) -> None:
    """Set up logging for Anonymizer.

    Called automatically on first :class:`~anonymizer.interface.anonymizer.Anonymizer`
    instantiation. Call explicitly before creating an Anonymizer to customize behavior.

    Args:
        verbose: When False (default), only Anonymizer progress messages are
            shown and DataDesigner engine logs are suppressed.  When True,
            DataDesigner DEBUG logs are also emitted.
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicates on repeated calls
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S"))
    root.addHandler(handler)

    logging.getLogger("anonymizer").setLevel(logging.INFO)
    logging.getLogger("data_designer").setLevel(logging.DEBUG if verbose else logging.WARNING)

    for name in _DEFAULT_NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
