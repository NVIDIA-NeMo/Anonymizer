# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact content size retained with closed immutable artifact values."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ArtifactShape:
    """Content bytes and collection members, not a Python heap-size estimate."""

    content_bytes: int
    items: int = 0


def content_bytes(text: str) -> int:
    """Charge malformed Unicode too; the ingress validator rejects it separately."""
    return sum(
        len(text[offset : offset + 1024].encode("utf-8", "surrogatepass")) for offset in range(0, len(text), 1024)
    )
