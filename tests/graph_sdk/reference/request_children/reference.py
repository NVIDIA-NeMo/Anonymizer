# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Finite trace language, independent of request-reducer implementation."""

from __future__ import annotations

from itertools import product

ALPHABET = "IPCREV"
COMPLETE = ("IPCPEV", "IPRPEV")


def cases() -> tuple[str, ...]:
    bounded = {"".join(items) for size in range(5) for items in product(ALPHABET, repeat=size)}
    return tuple(sorted(bounded | {word[:size] for word in COMPLETE for size in (5, 6)}))


def expected(trace: str) -> tuple[bool, int]:
    accepted = any(word.startswith(trace) for word in COMPLETE)
    return accepted, sum(letter in "ICRV" for letter in trace) if accepted else 0
