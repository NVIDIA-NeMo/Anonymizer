#!/usr/bin/env -S uv run --script
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "cyclopts>=3",
#   "httpx>=0.27",
#   "pydantic>=2.9,<3",
#   "structlog>=24.4",
# ]
# ///
"""Compile and manage local inference services from typed specifications."""

from __future__ import annotations

from inference_service_compiler.cli import app

if __name__ == "__main__":
    app()
