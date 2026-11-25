# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The release decisions Replace and Rewrite share: ``missed_repeat`` per mode and detector kind, and the fixed ones."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import get_args

import anonymizer.engine.graph_core
from anonymizer.engine.ndd.failed_record import FailedRecord
from anonymizer.engine.protection_policy import (
    Detector,
    FailureReason,
    FailureStep,
    RepeatHandling,
    ReviewFlag,
    missed_repeat,
)
from anonymizer.interface.anonymizer import _KNOWN_FAILURE_REASONS, _UNPLACED_CAUSE

_GRAPH_CORE = Path(anonymizer.engine.graph_core.__file__).parent
_PROBLEM_FUNCTIONS = {"mention_problem", "span_problem"}


def _graph_reasons() -> set[str]:
    """The record failure reasons the graph package writes, read from its source: the reason of every ``Failed(...)``
    event, every code ``mention_problem`` and ``span_problem`` return, and a ``failed.get(unit, <code>)`` default."""
    found: set[str] = set()
    for path in _GRAPH_CORE.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in _PROBLEM_FUNCTIONS:
                found |= {
                    str(r.value.value)
                    for r in ast.walk(node)
                    if isinstance(r, ast.Return) and isinstance(r.value, ast.Constant) and r.value.value
                }
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            args = node.args
            if name == "Failed" and len(args) >= 3 and isinstance(args[2], ast.Constant):
                found.add(str(args[2].value))
            if name == "get" and len(args) == 2 and isinstance(args[1], ast.Constant):
                found.add(str(args[1].value))
    return found


# The failure reasons the code emits: the graph package's (scanned from its source), the detect step's refusal, and the
# seam's dropped and unplaced records.
_EMITTED_REASONS = _graph_reasons() | {"unspanned_value", "dropped_output", "unplaced_output"}


def test_the_tally_knows_every_emitted_reason() -> None:
    assert len(_graph_reasons()) == 9  # the scan finds every site
    assert set(FailureReason) == _EMITTED_REASONS
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


def test_a_failure_step_is_its_plain_string_despite_its_count() -> None:
    """A step's member value is the step name alone; the count field it carries stays off the string."""
    assert str(FailureStep.DETECT) == f"{FailureStep.DETECT}" == FailureStep.DETECT.value == "detect"
    assert FailureStep("detect") is FailureStep.DETECT
    assert FailureStep.DETECT.count("e") == 2  # the field does not shadow str.count
    assert all(type(step.value) is str and type(step.count_field) is str for step in FailureStep)
    record = FailedRecord("r1", FailureStep.DETECT, FailureReason.DROPPED_OUTPUT)
    assert (type(record.step), record.step) == (str, "detect")


def test_a_failure_step_counts_under_its_field() -> None:
    assert FailureStep.count_of("detect") == FailureStep.DETECT.count_field == "entity_detection"
    assert FailureStep.count_of("rewrite-evaluate-2") == "rewrite_evaluate"
    assert FailureStep.count_of("rewrite-repair-10") == "rewrite_repair"
    assert FailureStep.count_of("rewrite-repair-") == "unknown"
    assert FailureStep.count_of("some-future-step") == "unknown"
    # No run-summary field of their own yet: these count as unknown, as they did before the counts moved here.
    unknown = {"replace", "rewrite", "verify-no-original", "verify-patches"}
    assert {step for step in FailureStep if step.count_field == "unknown"} == unknown
