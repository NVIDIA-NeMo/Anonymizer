# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Boundary-aware text windowing shared across long-context workflows.

Splits a long document into sequential, non-overlapping windows of at most
``max_chars`` each. Rather than cutting at an arbitrary character offset, each
window is backed off to the last ``delimiter`` (default a newline) within the
window so a chunk boundary lands on a natural break instead of mid-line /
mid-token. If no delimiter occurs in the window, it falls back to a hard cut at
``max_chars`` so progress is always made.

Used by the chunked Substitute (map generation) and Rewrite long-context paths.
"""

from __future__ import annotations

DEFAULT_DELIMITER = "\n"


def next_window_end(text: str, start: int, max_chars: int, *, delimiter: str = DEFAULT_DELIMITER) -> int:
    """Return the end offset for a window starting at ``start``.

    The window is at most ``max_chars`` long; when it does not reach the end of
    ``text`` it is backed off to just after the last ``delimiter`` inside the
    window. If the window contains no delimiter (other than possibly at the very
    start), a hard cut at ``start + max_chars`` is returned.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    hard_end = min(len(text), start + max_chars)
    if hard_end >= len(text):
        return len(text)
    window = text[start:hard_end]
    idx = window.rfind(delimiter)
    # idx > 0 ensures we make progress (a delimiter at offset 0 would not advance).
    if delimiter and idx > 0:
        return start + idx + len(delimiter)
    return hard_end


def iter_boundary_windows(text: str, max_chars: int, *, delimiter: str = DEFAULT_DELIMITER) -> list[tuple[int, int]]:
    """Tile ``[0, len(text))`` into sequential boundary-aligned ``(start, end)`` windows."""
    n = len(text)
    if n == 0:
        return []
    bounds: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = next_window_end(text, start, max_chars, delimiter=delimiter)
        if end <= start:  # defensive: always advance
            end = min(n, start + max_chars)
        bounds.append((start, end))
        start = end
    return bounds
