# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small declaration laws; no execution transitions or product imports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def forest_code(parents: Mapping[str, str | None], *, max_depth: int) -> str | None:
    """Completed root paths charge depth; cyclic paths do not invent a root."""
    longest = 0
    cycle = False
    missing = False
    for start in parents:
        seen: set[str] = set()
        node: str | None = start
        while node is not None:
            if node in seen:
                cycle = True
                break
            if node not in parents:
                missing = True
                break
            seen.add(node)
            node = parents[node]
        if node is None:
            longest = max(longest, len(seen))
    if longest > max_depth:
        return "limit_exceeded"
    if missing:
        return "missing"
    return "cycle" if cycle else None


def unique_names(names: Sequence[str]) -> bool:
    return len(names) == len(set(names))
