# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The release decisions Replace and Rewrite share, named once.

``missed_repeat`` is the one situation whose answer depends on the mode and the detector kind. The others have a
fixed answer: a record fails with one of ``FAIL_REASONS``, a Rewrite row is flagged for review by one of
``REVIEW_FLAGS`` and stays met, and Rewrite withholds a final candidate that fails a privacy check or whose privacy
evaluation had no usable response. The fact checks that feed these decisions stay with their verifiers.
"""

from __future__ import annotations

from typing import Final, Literal

Mode = Literal["replace", "rewrite"]
Detector = Literal["builtin", "custom"]
RepeatHandling = Literal["fail", "protect", "release"]

UNSPANNED_VALUE: Final = "unspanned_value"
"""A Replace row with a custom detector's missed repeat, or with a repeat no added span can cover, fails with it."""
DROPPED_OUTPUT: Final = "dropped_output"
"""A record a custom detector neither answered nor failed fails at ``detect`` with it."""
UNPLACED_OUTPUT: Final = "unplaced_output"
"""An input record that no placed row of an override runner's output answers fails with it."""
FAIL_REASONS: Final = frozenset({UNSPANNED_VALUE, DROPPED_OUTPUT, UNPLACED_OUTPUT})

UNPLACED_CAUSE: Final = "unplaced"
"""The cause every verifier reports for a result row ``evaluate()`` cannot place; the row is unassessed."""

REVIEW_FLAGS: Final = ("undetected_value_kept", "placeholder_in_text")
"""The Rewrite metrics that flag a met row for review. Substitute output sets ``placeholder_in_text`` the same way."""


def missed_repeat(mode: Mode, detector: Detector) -> RepeatHandling:
    """What happens to a reported value that occurs in the text outside every kept span.

    Replace replaces spans only: a custom detector must span every occurrence, so the row fails with
    ``UNSPANNED_VALUE``; the built-in detector's repeat gets a span. Rewrite passes it to the rewriter in its
    replacement view, for either detector kind.
    """
    if mode == "rewrite":
        return "release"
    return "fail" if detector == "custom" else "protect"
