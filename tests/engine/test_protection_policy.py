# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The release decisions Replace and Rewrite share: ``missed_repeat`` per mode and detector kind, and the fixed ones."""

from __future__ import annotations

from typing import get_args

from anonymizer.engine.protection_policy import (
    Detector,
    FailureReason,
    RepeatHandling,
    ReviewFlag,
    missed_repeat,
)
from anonymizer.interface.anonymizer import _KNOWN_FAILURE_REASONS, _UNPLACED_CAUSE

# The failure reasons the code emits, listed from the emit sites rather than from the policy: the protection graph's
# own (execution and mention rules), the detect step's refusal, the seam's dropped and unplaced records.
_EMITTED_REASONS = {
    "invalid_output",
    "mention_not_on_target",
    "mention_out_of_range",
    "mention_slice_mismatch",
    "mention_unlabelled",
    "missing_output",
    "over_limit",
    "overlapping_output_ownership",
    "raised",
    "unspanned_value",
    "dropped_output",
    "unplaced_output",
}


def test_the_tally_knows_every_emitted_reason() -> None:
    assert set(FailureReason) == {"unspanned_value", "dropped_output", "unplaced_output"}
    assert _KNOWN_FAILURE_REASONS == _EMITTED_REASONS


def test_rewrite_releases_a_missed_repeat_for_either_detector_kind() -> None:
    """Rewrite's runners take no detector kind, so its answer must not depend on it."""
    assert {missed_repeat("rewrite", d) for d in get_args(Detector)} == {"release"}


def test_the_ruled_decisions() -> None:
    assert missed_repeat("replace", "custom") == "fail"
    assert missed_repeat("replace", "builtin") == "protect"
    assert FailureReason.UNSPANNED_VALUE == "unspanned_value"
    assert FailureReason.DROPPED_OUTPUT == "dropped_output"
    assert FailureReason.UNPLACED_OUTPUT == "unplaced_output"
    assert _UNPLACED_CAUSE == "unplaced"
    assert list(ReviewFlag) == ["undetected_value_kept", "placeholder_in_text"]
    assert list(RepeatHandling) == ["fail", "protect", "release"]


def test_the_members_are_plain_strings_where_they_are_read() -> None:
    """A member prints, formats and keys a dict as its code, so no enum name reaches a log or a record."""
    for member in (*FailureReason, *ReviewFlag, *RepeatHandling):
        assert f"{member}" == str(member) == member.value
        assert {member.value: 1}[member] == 1
        assert type(member.value) is str
