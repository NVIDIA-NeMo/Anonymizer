# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic test-time thinning of large frozen corpora.

A behavior signature is computed only from frozen case data (declaration shape
and the authenticated expected answer), never from product output. Each
signature keeps its first and last case in corpus order, so both ends of an
enumerated sweep run. Cases matching ``sweep`` run in full.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from typing import TypeVar

Case = TypeVar("Case")


def representatives(
    cases: Sequence[Case],
    signature: Callable[[Case], Hashable],
    *,
    sweep: Callable[[Case], bool] = lambda _case: False,
) -> list[Case]:
    """Return the selected cases in corpus order."""
    first: dict[Hashable, int] = {}
    last: dict[Hashable, int] = {}
    for index, case in enumerate(cases):
        key = signature(case)
        first.setdefault(key, index)
        last[key] = index
    selected = set(first.values()) | set(last.values())
    selected.update(index for index, case in enumerate(cases) if sweep(case))
    return [cases[index] for index in sorted(selected)]
