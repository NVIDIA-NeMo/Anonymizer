#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared CLI logging helpers for measurement tools."""

from __future__ import annotations

import json
import logging
import sys
from enum import StrEnum

from pydantic import ValidationError


class LogFormat(StrEnum):
    plain = "plain"
    json = "json"


_log_format = LogFormat.plain


def configure_logging(log_format: LogFormat) -> None:
    global _log_format

    _log_format = log_format
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def log_bad_input(logger: logging.Logger, error: str) -> None:
    if _log_format == LogFormat.json:
        payload = {"level": "error", "event": "bad_input", "error": error}
        sys.stderr.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
        return
    logger.error("bad_input error=%s", error)


def summarize_validation_error(error: ValidationError) -> str:
    """Describe rejected fields without echoing their potentially sensitive values."""
    details = sorted(
        {
            f"{'.'.join(str(part) for part in item['loc']) or 'input'}:{item['type']}"
            for item in error.errors(include_input=False, include_context=False, include_url=False)
        }
    )
    return "validation failed at " + ", ".join(details[:12])
