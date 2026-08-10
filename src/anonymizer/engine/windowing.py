# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Boundary-aware text windowing shared across long-context workflows.

Splits a long document into sequential, non-overlapping windows of at most
``max_chars`` each. Rather than cutting at an arbitrary character offset, each
window is backed off to the last ``delimiter`` (default a newline) within the
window so a chunk boundary lands on a natural break instead of mid-line /
mid-token. A back-off is only accepted if it keeps the window at least
``MIN_WINDOW_FRACTION`` of ``max_chars`` — a delimiter that appears only near
the start of the window would produce a degenerate, tiny window (and thus many
extra LLM calls). When the primary delimiter has no acceptable occurrence, the
sentence boundary ``"."`` is tried next; if that also fails, the window is
hard-cut at ``max_chars`` (possibly mid-sentence) and a warning is logged so
progress is always made.

Used by the chunked Substitute (map generation) and Rewrite long-context paths.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("anonymizer.windowing")

DEFAULT_DELIMITER = "\n"

# Secondary boundary tried when the primary delimiter has no acceptable
# occurrence in the window: a sentence end.
FALLBACK_DELIMITER = "."

# A delimiter back-off must keep the window at least this fraction of
# ``max_chars``; earlier occurrences would yield degenerate, tiny windows.
MIN_WINDOW_FRACTION = 0.5


def next_window_end(text: str, start: int, max_chars: int, *, delimiter: str = DEFAULT_DELIMITER) -> int:
    """Return the end offset for a window starting at ``start``.

    The window is at most ``max_chars`` long; when it does not reach the end of
    ``text`` it is backed off to just after the last acceptable boundary inside
    the window, trying ``delimiter`` first and then ``FALLBACK_DELIMITER``. A
    boundary is acceptable only if it keeps the window at least
    ``MIN_WINDOW_FRACTION * max_chars`` long. If neither delimiter has an
    acceptable occurrence, a hard cut at ``start + max_chars`` is returned and a
    warning is logged (the cut may land mid-sentence).
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    hard_end = min(len(text), start + max_chars)
    if hard_end >= len(text):
        return len(text)
    window = text[start:hard_end]
    min_window_len = max(1, int(max_chars * MIN_WINDOW_FRACTION))

    tried: list[str] = []
    for delim in (delimiter, FALLBACK_DELIMITER):
        if not delim or delim in tried:
            continue
        tried.append(delim)
        idx = window.rfind(delim)
        # idx > 0 ensures we make progress (a delimiter at offset 0 would not advance).
        if idx > 0 and idx + len(delim) >= min_window_len:
            return start + idx + len(delim)

    logger.warning(
        "no %s boundary in the trailing %d%% of the %d-char window at offset %d; "
        "hard-cutting at max_chars (the cut may split a sentence or token)",
        " or ".join(repr(d) for d in tried),
        round((1 - MIN_WINDOW_FRACTION) * 100),
        max_chars,
        start,
    )
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
