# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-free record of a dropped workflow row.

Kept apart from ``adapter`` so frame publication can import it without the adapter's
DataDesigner and rewrite-binding imports (which would close an import cycle with the SDK).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from anonymizer.engine.protection_policy import FailureReason, FailureStep


@dataclass(frozen=True)
class FailedRecord:
    """A record that did not appear in workflow output.

    ``step`` and ``reason`` are a ``FailureStep`` and a ``FailureReason`` wherever Anonymizer names them; a workflow's
    own name, a model's error text, or a custom detector's or runner's failure stays free text. A member is stored as
    its plain string, so a record reads, prints and compares as it always has.
    """

    record_id: str
    step: FailureStep | str
    reason: FailureReason | str

    def __post_init__(self) -> None:
        for name in ("step", "reason"):
            if isinstance(value := getattr(self, name), StrEnum):
                object.__setattr__(self, name, value.value)
