# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The release decisions Replace and Rewrite share, named once.

``missed_repeat`` is the one situation whose answer depends on the mode and the detector kind; it returns a
``RepeatHandling``. The others have a fixed answer: a record fails with one of the ``FailureReason`` members, a
Rewrite row is flagged for review by one of the ``ReviewFlag`` members and stays met, and Rewrite withholds a final
candidate that fails a privacy check, whose privacy evaluation had no usable response, or whose privacy evaluation
left a question unanswered. The fact checks that feed these decisions stay with their verifiers. The enums are
``StrEnum``: a member equals, hashes and serialises as its string, so reasons and flags stay plain codes wherever they
are read.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Literal

Mode = Literal["replace", "rewrite"]
Detector = Literal["builtin", "custom"]

_REWRITE_ROUND_RE = re.compile(r"rewrite-(evaluate|repair)-\d+")


class RepeatHandling(StrEnum):
    """What happens to a reported value that occurs in the text outside every kept span."""

    FAIL = "fail"
    PROTECT = "protect"
    RELEASE = "release"


class FailureReason(StrEnum):
    """Every reason a failed record can carry that Anonymizer itself sets: the protection graph's (an operation's
    failure or a rejected mention) and the release policy's. A model error, a custom detector's failure or a runner
    override may still give free text (``FailedRecord.reason``). The graph package imports nothing outside itself, so
    it writes its codes as strings; ``tests/engine/test_protection_policy.py`` keeps them in this set."""

    INVALID_OUTPUT = "invalid_output"
    """An operation returned a value that is not its declared outputs."""
    MENTION_NOT_ON_TARGET = "mention_not_on_target"
    MENTION_OUT_OF_RANGE = "mention_out_of_range"
    MENTION_SLICE_MISMATCH = "mention_slice_mismatch"
    MENTION_UNLABELLED = "mention_unlabelled"
    MISSING_OUTPUT = "missing_output"
    """An invocation operation returned ``None`` for the record."""
    OVER_LIMIT = "over_limit"
    OVERLAPPING_OUTPUT_OWNERSHIP = "overlapping_output_ownership"
    RAISED = "raised"
    """An operation raised for the record."""
    UNSPANNED_VALUE = "unspanned_value"
    """A Replace row with a custom detector's missed repeat, or with a repeat no added span can cover, fails with it."""
    DROPPED_OUTPUT = "dropped_output"
    """A record a custom detector neither answered nor failed fails at ``detect`` with it."""
    UNPLACED_OUTPUT = "unplaced_output"
    """An input record that no placed row of an override runner's output answers fails with it."""


class FailureStep(StrEnum):
    """The step names Anonymizer gives a failed record besides a workflow's own name (``FailedRecord.step``).

    Each member carries the run-summary failure-count field it counts under (``count_field``). A member still equals,
    formats and is stored as its plain step name.
    """

    count_field: str

    def __new__(cls, value: str, count_field: str) -> FailureStep:
        member = str.__new__(cls, value)
        member._value_ = value
        member.count_field = count_field
        return member

    DETECT = "detect", "entity_detection"
    ENTITY_DETECTION = "entity-detection", "entity_detection"
    LATENT_ENTITY_DETECTION = "latent-entity-detection", "latent_detection"
    # Replace, Rewrite and the verifiers' steps have no run-summary field of their own, so they count as unknown.
    REPLACE = "replace", "unknown"
    REPLACE_MAP_GENERATION = "replace-map-generation", "replace_map_generation"
    SUBSTITUTE_MAP = "substitute-map", "replace_map_generation"
    REWRITE = "rewrite", "unknown"
    REWRITE_PIPELINE = "rewrite-pipeline", "rewrite_pipeline"
    REWRITE_COMBINED = "rewrite-combined", "rewrite_pipeline"
    REWRITE_LABEL_CLASSIFICATION = "rewrite-label-classification", "rewrite_pipeline"
    REWRITE_FINAL_JUDGE = "rewrite-final-judge", "rewrite_final_judge"
    VERIFY_NO_ORIGINAL = "verify-no-original", "unknown"
    VERIFY_PATCHES = "verify-patches", "unknown"

    @classmethod
    def count_of(cls, step: str) -> str:
        """The run-summary field a failed record's step counts under: a member's ``count_field``, ``rewrite_evaluate`` or
        ``rewrite_repair`` for a numbered Rewrite round, and ``unknown`` for any other step."""
        try:
            return cls(step).count_field
        except ValueError:
            round_ = _REWRITE_ROUND_RE.fullmatch(step)
            return f"rewrite_{round_.group(1)}" if round_ else "unknown"


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
