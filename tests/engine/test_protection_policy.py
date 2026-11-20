# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The release-policy table: one row for every situation, mode and detector kind, and the decisions read from it."""

from __future__ import annotations

from itertools import product
from typing import get_args

from anonymizer.engine.protection_policy import POLICY, Detector, Mode, Situation, assess, fail_reasons
from anonymizer.interface.anonymizer import _GRAPH_FAILURE_REASONS, _KNOWN_FAILURE_REASONS


def test_every_situation_has_a_row_for_every_mode_and_detector() -> None:
    """Totality: a situation nobody has ruled on for some mode or detector kind fails here."""
    expected = set(product(get_args(Situation), get_args(Mode), get_args(Detector)))
    assert set(POLICY) == expected


def test_every_failure_verdict_names_its_reason_and_the_reason_set_is_derived() -> None:
    assert all(v.reason for v in POLICY.values() if v.action == "fail")
    assert _KNOWN_FAILURE_REASONS == _GRAPH_FAILURE_REASONS | fail_reasons()
    assert {"unspanned_value", "dropped_output", "unplaced_output", "stopped_early"} <= _KNOWN_FAILURE_REASONS


def test_rewrite_verdicts_do_not_depend_on_the_detector_kind() -> None:
    """Rewrite's runners take no detector kind (J3), so its rows must agree across kinds."""
    for situation in get_args(Situation):
        assert assess(situation, "rewrite", "builtin") == assess(situation, "rewrite", "custom"), situation


def test_the_ruled_rows() -> None:
    assert assess("missed_repeat", "replace", "custom").reason == "unspanned_value"
    assert assess("missed_repeat", "replace", "builtin").action == "protect"
    assert assess("missed_repeat", "rewrite", "builtin").action == "release"
    assert {assess("placeholder_in_text", m, "builtin").action for m in get_args(Mode)} == {"flag"}
