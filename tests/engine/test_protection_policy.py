# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The release decisions Replace and Rewrite share: ``missed_repeat`` per mode and detector kind, and the fixed ones."""

from __future__ import annotations

from typing import get_args

from anonymizer.engine.protection_policy import (
    DROPPED_OUTPUT,
    FAIL_REASONS,
    REVIEW_FLAGS,
    UNPLACED_CAUSE,
    UNPLACED_OUTPUT,
    UNSPANNED_VALUE,
    Detector,
    missed_repeat,
)
from anonymizer.interface.anonymizer import _KNOWN_FAILURE_REASONS

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
    assert FAIL_REASONS == {UNSPANNED_VALUE, DROPPED_OUTPUT, UNPLACED_OUTPUT}
    assert _KNOWN_FAILURE_REASONS == _EMITTED_REASONS


def test_rewrite_releases_a_missed_repeat_for_either_detector_kind() -> None:
    """Rewrite's runners take no detector kind, so its answer must not depend on it."""
    assert {missed_repeat("rewrite", d) for d in get_args(Detector)} == {"release"}


def test_the_ruled_decisions() -> None:
    assert missed_repeat("replace", "custom") == "fail"
    assert UNSPANNED_VALUE == "unspanned_value"
    assert missed_repeat("replace", "builtin") == "protect"
    assert UNPLACED_CAUSE == "unplaced"
    assert DROPPED_OUTPUT == "dropped_output"
    assert UNPLACED_OUTPUT == "unplaced_output"
    assert REVIEW_FLAGS == ("undetected_value_kept", "placeholder_in_text")
