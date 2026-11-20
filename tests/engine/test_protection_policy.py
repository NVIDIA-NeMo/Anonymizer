# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The release-policy table: one row for every situation, mode and detector kind, and the decisions read from it."""

from __future__ import annotations

from itertools import product
from typing import get_args

from anonymizer.engine.protection_policy import POLICY, Detector, Mode, Situation, Verdict, assess, assess_any
from anonymizer.interface.anonymizer import _KNOWN_FAILURE_REASONS


def test_every_situation_has_a_row_for_every_mode_and_detector() -> None:
    """Totality: a situation nobody has ruled on for some mode or detector kind fails here."""
    expected = set(product(get_args(Situation), get_args(Mode), get_args(Detector)))
    assert set(POLICY) == expected


# The failure reasons the code emits, listed from the emit sites rather than from the table: the protection graph's
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


def test_every_failure_verdict_names_its_reason_and_the_tally_knows_every_emitted_reason() -> None:
    assert all(v.reason for v in POLICY.values() if v.action == "fail")
    assert _KNOWN_FAILURE_REASONS == _EMITTED_REASONS


def test_rewrite_verdicts_do_not_depend_on_the_detector_kind() -> None:
    """Rewrite's runners take no detector kind, so its live rows must agree across kinds."""
    for situation in get_args(Situation):
        live = {assess(situation, "rewrite", d) for d in get_args(Detector)} - {Verdict("n/a")}
        assert len(live) <= 1, situation


def test_the_ruled_rows() -> None:
    assert assess("missed_repeat", "replace", "custom").reason == "unspanned_value"
    assert assess("missed_repeat", "replace", "builtin").action == "protect"
    assert assess("missed_repeat", "rewrite", "builtin").action == "release"
    assert {assess("placeholder_in_text", m, "builtin").action for m in get_args(Mode)} == {"flag"}
    assert assess_any("unplaced_result_row") == Verdict("unassessed", "unplaced")
    assert assess_any("dropped_record", detector="custom") == Verdict("fail", "dropped_output")
