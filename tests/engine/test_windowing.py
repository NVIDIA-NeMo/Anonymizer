# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for boundary-aware windowing."""

from __future__ import annotations

import pytest

from anonymizer.engine.windowing import iter_boundary_windows, next_window_end


class TestNextWindowEnd:
    def test_backs_off_to_last_delimiter(self) -> None:
        # "aaaa\nbbbb\ncccc" — cap at 12 lands inside "cccc"; back off to after second "\n" (offset 10).
        text = "aaaa\nbbbb\ncccc"
        assert next_window_end(text, 0, 12) == 10  # after the "\n" at index 9
        assert text[:10] == "aaaa\nbbbb\n"

    def test_hard_cut_when_no_delimiter(self) -> None:
        text = "abcdefghij"  # no newline
        assert next_window_end(text, 0, 4) == 4

    def test_returns_end_when_remainder_fits(self) -> None:
        text = "short\ntext"
        assert next_window_end(text, 0, 1000) == len(text)

    def test_custom_delimiter(self) -> None:
        text = "a;b;c;d;e"
        assert next_window_end(text, 0, 4, delimiter=";") == 4  # "a;b;" (last ';' within [0,4))


class TestIterBoundaryWindows:
    def test_tiles_on_newlines(self) -> None:
        text = "line1\nline2\nline3\nline4"
        windows = iter_boundary_windows(text, 12)
        # backs off to the LAST newline within the cap, e.g. first window "line1\nline2\n"
        assert windows[0] == (0, 12)
        assert all((e - s) <= 12 for s, e in windows)
        assert all(text[s:e].endswith("\n") for s, e in windows[:-1])  # non-last windows end on a newline
        assert windows[-1][1] == len(text)
        assert "".join(text[s:e] for s, e in windows) == text  # exact reconstruction

    def test_empty(self) -> None:
        assert iter_boundary_windows("", 10) == []

    def test_no_delimiter_hard_cuts(self) -> None:
        text = "x" * 25
        windows = iter_boundary_windows(text, 10)
        assert windows == [(0, 10), (10, 20), (20, 25)]
