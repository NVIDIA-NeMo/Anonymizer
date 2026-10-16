# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Finite trace language, independent of request-reducer implementation."""

from __future__ import annotations

ALPHABET = "IPCREV"
COMPLETE = ("IPCPEV", "IPRPEV")


def cases() -> tuple[str, ...]:
    """Keep valid prefixes and the first illegal transition within length four."""
    prefixes = {word[:size] for word in COMPLETE for size in range(len(word) + 1)}
    invalid = {
        prefix + letter
        for prefix in prefixes
        if len(prefix) < 4
        for letter in ALPHABET
        if prefix + letter not in prefixes
    }
    return tuple(sorted(prefixes | invalid))


def expected(trace: str) -> tuple[bool, int]:
    accepted = any(word.startswith(trace) for word in COMPLETE)
    return accepted, sum(letter in "ICRV" for letter in trace) if accepted else 0
