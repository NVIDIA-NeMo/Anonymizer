# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-free record of a dropped workflow row.

Kept apart from ``adapter`` so frame publication can import it without the adapter's
DataDesigner and rewrite-binding imports (which would close an import cycle with the SDK).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FailedRecord:
    """A record that did not appear in workflow output."""

    record_id: str
    step: str
    reason: str
