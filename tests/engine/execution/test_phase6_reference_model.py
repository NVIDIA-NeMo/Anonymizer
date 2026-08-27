# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path

import pytest

from tests.engine.execution.phase6_reference_model import (
    INDEPENDENCE_RELATION,
    ReferenceCandidate,
    ReferenceCase,
    ReferenceEvent,
    ReferenceEventKind,
    ReferenceEvidence,
    canonical_schedule,
    default_schedule,
    finite_reference_cases,
    lifecycle_reference_cases,
    ordered_race_schedules,
    reduce_reference,
    reference_manifest,
)


def test_phase6_reference_model_is_independent_and_manifest_is_frozen() -> None:
    source_path = Path(__file__).with_name("phase6_reference_model.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert imported_roots.isdisjoint({"anonymizer", "pandas", "pytest"})
    manifest = json.loads(Path(__file__).with_name("phase6_reference_manifest.json").read_text(encoding="utf-8"))
    assert reference_manifest() == manifest
    assert manifest["case_count"] == len(finite_reference_cases())
    assert manifest["actual_event_count"] > manifest["canonical_trace_count"]
    assert manifest["max_event_count"] > 0


def test_phase6_reference_cases_contain_executable_bounded_schedules() -> None:
    cases = finite_reference_cases()

    assert all(case.events for case in cases)
    assert all(reduce_reference(case).event_count <= reduce_reference(case).max_event_count for case in cases)
    schedule_class_counts = reference_manifest()["schedule_class_counts"]
    assert isinstance(schedule_class_counts, dict)
    assert schedule_class_counts == {
        "cancel-after-verification": 1,
        "cancel-dispatch": 2,
        "dispatch-terminal": 2,
        "cancel-terminal": 2,
        "cleanup-failure": 1,
        "contradictory-record-patch": 2,
        "duplicate-resolver-completion": 1,
        "finalize-release": 2,
        "late-candidate-after-cancel": 1,
        "late-candidate-after-loss": 1,
        "late-evidence-after-cancel": 1,
        "late-evidence-after-loss": 1,
        "local-failure-independent-success": 1,
        "success": 21,
        "teardown-failure-after-acceptance": 1,
        "teardown-acceptance": 2,
        "verify-release": 2,
    }

    assert {case.name for case in lifecycle_reference_cases()} == {
        "cancel-before-dispatch",
        "cancel-after-dispatch",
        "late-candidate-after-cancel",
        "late-evidence-after-cancel",
        "late-candidate-after-loss",
        "late-evidence-after-loss",
        "duplicate-resolver-completion",
        "patch-before-contradictory-record",
        "contradictory-record-before-patch",
        "local-failure-independent-success",
        "cancel-after-verification",
        "cleanup-failure",
        "teardown-failure-after-acceptance",
    }
    assert all(case.events for case in lifecycle_reference_cases())
    assert all(
        reduce_reference(case).event_count <= reduce_reference(case).max_event_count
        for case in lifecycle_reference_cases()
    )


def test_phase6_schedule_canonicalization_collapses_only_commuting_swaps() -> None:
    for left, right in INDEPENDENCE_RELATION:
        first = (
            ReferenceEvent(ReferenceEventKind(left), "z"),
            ReferenceEvent(ReferenceEventKind(right), "a"),
        )
        second = tuple(reversed(first))
        assert canonical_schedule(first) == canonical_schedule(second)

    for _name, first, second in ordered_race_schedules():
        assert canonical_schedule(first) != canonical_schedule(second)


@pytest.mark.parametrize(("name", "first", "second"), ordered_race_schedules())
def test_phase6_ordered_races_have_distinct_observable_outcomes(
    name: str,
    first: tuple[ReferenceEvent, ...],
    second: tuple[ReferenceEvent, ...],
) -> None:
    case = ReferenceCase("race", ("A",), (), events=first)
    first_result = reduce_reference(case)
    second_result = reduce_reference(replace(case, events=second))

    assert first_result.schedule != second_result.schedule, name
    if name == "dispatch-terminal":
        assert first_result.schedule.task_terminal != second_result.schedule.task_terminal
    elif name == "cancel-terminal":
        assert first_result.schedule.cancellation != second_result.schedule.cancellation
    else:
        assert first_result.schedule.release != second_result.schedule.release


def test_phase6_lifecycle_schedules_are_fail_closed_without_rewriting_accepted_results() -> None:
    results = {case.name: reduce_reference(case) for case in lifecycle_reference_cases()}

    assert results["late-candidate-after-cancel"].schedule.task_outcomes == (("target-0", "cancelled"),)
    assert results["late-evidence-after-cancel"].schedule.task_outcomes == (("target-0", "cancelled"),)
    assert results["late-candidate-after-loss"].schedule.task_outcomes == (("target-0", "lost"),)
    assert results["late-evidence-after-loss"].schedule.task_outcomes == (("target-0", "lost"),)
    assert results["duplicate-resolver-completion"].schedule.invocation == "inconsistent"
    assert results["patch-before-contradictory-record"].schedule.invocation == "inconsistent"
    assert results["local-failure-independent-success"].released_groups == (1,)
    assert results["cancel-after-verification"].released_groups == ()
    assert results["cleanup-failure"].schedule.invocation == "inconsistent"
    accepted = results["teardown-failure-after-acceptance"]
    assert accepted.schedule.immutable_result
    assert accepted.schedule.teardown == "failed"
    assert accepted.schedule.invocation == "completed"
    assert accepted.released_groups == (0,)


def test_phase6_reference_reducer_rejects_schedule_over_bound() -> None:
    case = ReferenceCase("over-bound", ("A",), (), events=default_schedule())
    result = reduce_reference(case)
    excessive = replace(
        case,
        events=case.events
        + tuple(ReferenceEvent(ReferenceEventKind.CANCEL) for _index in range(result.max_event_count + 1)),
    )

    with pytest.raises(AssertionError, match="bound"):
        reduce_reference(excessive)


def test_reference_oracle_keeps_repeated_occurrences_anchored_and_reconstructs_exactly() -> None:
    result = reduce_reference(
        ReferenceCase(
            "repeated",
            ("Alice and Alice",),
            (ReferenceCandidate(0, 0, 5, "Alice", "name"),),
        )
    )

    assert result.rejection is None
    assert result.outputs == ("[REDACTED] and Alice",)
    assert result.clusters == ((0,),)
    assert result.released_groups == (0,)


def test_reference_oracle_clusters_only_explicit_evidence_and_rejects_transitive_contradiction() -> None:
    candidates = (
        ReferenceCandidate(0, 0, 1, "A", "name"),
        ReferenceCandidate(0, 2, 3, "B", "name"),
        ReferenceCandidate(0, 4, 5, "C", "name"),
    )
    separate = reduce_reference(ReferenceCase("separate", ("A B C",), candidates))
    same = reduce_reference(
        ReferenceCase(
            "same",
            ("A B C",),
            candidates,
            (ReferenceEvidence("same_subject", 0, 1),),
        )
    )
    contradictory = reduce_reference(
        ReferenceCase(
            "contradictory",
            ("A B C",),
            candidates,
            (
                ReferenceEvidence("same_subject", 0, 1),
                ReferenceEvidence("same_subject", 1, 2),
                ReferenceEvidence("distinct_subject", 0, 2),
            ),
        )
    )

    assert separate.clusters == ((0,), (1,), (2,))
    assert same.clusters == ((0, 1), (2,))
    assert contradictory.rejection == "evidence_contradiction"
    assert contradictory.released_groups == ()


def test_reference_group_predicate_failure_is_monotone_through_dependencies() -> None:
    result = reduce_reference(
        ReferenceCase(
            "propagation",
            ("A", "B"),
            (),
            dependencies=((0, 1),),
            groups=((0,), (1,)),
            group_passes=(False, True),
        )
    )

    assert result.rejection is None
    assert result.released_groups == ()
