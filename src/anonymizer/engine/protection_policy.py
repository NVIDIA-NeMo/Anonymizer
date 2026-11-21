# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The release decisions Replace and Rewrite share, named once.

``missed_repeat`` is the one situation whose answer depends on the mode and the detector kind; it returns a
``RepeatHandling``. The others have a fixed answer: a record fails with one of the ``FailureReason`` members, a
Rewrite row is flagged for review by one of the ``ReviewFlag`` members and stays met, and Rewrite withholds a final
candidate that fails a privacy check or whose privacy evaluation had no usable response. The fact checks that feed
these decisions stay with their verifiers. The three enums are ``StrEnum``: a member equals, hashes and serialises as
its string, so reasons and flags stay plain codes wherever they are read.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

Mode = Literal["replace", "rewrite"]
Detector = Literal["builtin", "custom"]


class RepeatHandling(StrEnum):
    """What happens to a reported value that occurs in the text outside every kept span."""

    FAIL = "fail"
    PROTECT = "protect"
    RELEASE = "release"


class FailureReason(StrEnum):
    """The reasons the release policy fails a record with; the protection graph has its own besides."""

    UNSPANNED_VALUE = "unspanned_value"
    """A Replace row with a custom detector's missed repeat, or with a repeat no added span can cover, fails with it."""
    DROPPED_OUTPUT = "dropped_output"
    """A record a custom detector neither answered nor failed fails at ``detect`` with it."""
    UNPLACED_OUTPUT = "unplaced_output"
    """An input record that no placed row of an override runner's output answers fails with it."""


class ReviewFlag(StrEnum):
    """The Rewrite metrics that flag a met row for review. Substitute output sets ``placeholder_in_text`` the same way."""

    UNDETECTED_VALUE_KEPT = "undetected_value_kept"
    PLACEHOLDER_IN_TEXT = "placeholder_in_text"


def missed_repeat(mode: Mode, detector: Detector) -> RepeatHandling:
    """What happens to a reported value that occurs in the text outside every kept span.

    Replace replaces spans only: a custom detector must span every occurrence, so the row fails with
    ``FailureReason.UNSPANNED_VALUE``; the built-in detector's repeat gets a span. Rewrite passes it to the rewriter in its
    replacement view, for either detector kind.
    """
    if mode == "rewrite":
        return RepeatHandling.RELEASE
    return RepeatHandling.FAIL if detector == "custom" else RepeatHandling.PROTECT
