# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path

import pytest

from tests.engine.execution.phase5_reference_model import (
    ReferenceAdmission,
    ReferenceEvent,
    ReferenceEventKind,
    ReferenceInvocation,
    corpus_manifest,
    evaluate,
    reference_cases,
)

_MANIFEST = json.loads(Path(__file__).with_name("phase5_reference_manifest.json").read_text())


def test_phase5_reference_corpus_matches_frozen_manifest() -> None:
    assert corpus_manifest() == _MANIFEST


def test_reference_model_conserves_bindings_and_respects_event_bound() -> None:
    for case in reference_cases():
        result = evaluate(case)
        assert result.event_count <= result.event_max
        if result.admission is not ReferenceAdmission.ADMITTED:
            assert result.invocation is ReferenceInvocation.NOT_OPENED
            assert result.released == ()
        if result.cleanup != "verified":
            assert result.released == ()


def test_reference_model_distinguishes_local_and_global_binding_faults() -> None:
    admitted = next(
        case
        for case in reference_cases()
        if case.schedule_class == "exact:verified:success:accepted"
        and sum(len(scope.context) for scope in case.scopes)
        and len(case.targets) == 2
    )
    local = next(case for case in reference_cases() if case.case_id == f"{admitted.case_id}-binding-missing")
    global_fault = next(
        case for case in reference_cases() if case.case_id == f"{admitted.case_id}-binding-cross_target"
    )

    local_result = evaluate(local)
    global_result = evaluate(global_fault)

    assert len(local_result.private_inconsistent) == 1
    assert local_result.invocation is ReferenceInvocation.COMPLETED
    assert global_result.private_inconsistent == tuple(identifier for identifier, _text in global_fault.targets)
    assert global_result.invocation is ReferenceInvocation.INCONSISTENT
    assert global_result.released == ()


def test_reference_generator_freezes_required_ceiling_payload_and_event_domains() -> None:
    manifest = corpus_manifest()

    assert manifest["ceiling_domain"] == ["zero", "exact", "exact_plus_one"]
    assert manifest["payload_domain"] == ["empty", "one_byte", "multibyte", "exact_limit", "one_over_limit"]
    actual_events = manifest["actual_event_count"]
    trace_count = manifest["canonical_trace_count"]
    assert isinstance(actual_events, int)
    assert isinstance(trace_count, int)
    assert actual_events > trace_count
    assert all(case.events for case in reference_cases())


def test_reference_model_rejects_a_trace_over_its_computed_bound() -> None:
    case = next(reference_cases())
    excessive = replace(
        case,
        events=case.events
        + tuple(ReferenceEvent(ReferenceEventKind.CANCELLATION) for _index in range(case.limits.expanded_bytes + 100)),
    )

    try:
        evaluate(excessive)
    except AssertionError as error:
        assert "bound" in str(error)
    else:
        raise AssertionError("over-bound trace was not rejected")


def test_reference_model_imports_no_production_or_dataframe_modules() -> None:
    source = Path(__file__).with_name("phase5_reference_model.py").read_text()
    imported = {
        alias.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert not any(name.startswith(("anonymizer", "pandas", "data_designer")) for name in imported)


def test_reference_cancellation_linearization_matches_release_embargo() -> None:
    cases = tuple(reference_cases())
    late = next(case for case in cases if case.schedule_class == "exact:verified:terminal_then_cancel:accepted")
    before_dispatch = next(
        case for case in cases if case.schedule_class == "exact:verified:cancel_pre_dispatch:accepted"
    )

    late_result = evaluate(late)
    pre_dispatch_result = evaluate(before_dispatch)

    assert late_result.invocation is ReferenceInvocation.CANCELLED
    assert all(state == "succeeded" for _target, state, _reason in late_result.task_outcomes)
    assert late_result.released == ()
    assert pre_dispatch_result.invocation is ReferenceInvocation.CANCELLED
    assert all(state == "cancelled" for _target, state, _reason in pre_dispatch_result.task_outcomes)
    assert pre_dispatch_result.cleanup == "not_entered"


def test_reference_model_rejects_publication_before_cleanup() -> None:
    case = next(
        case
        for case in reference_cases()
        if case.schedule_class == "exact:verified:success:accepted" and len(case.targets) == 2
    )
    publication = next(event for event in case.events if event.kind is ReferenceEventKind.PUBLICATION)
    reordered = replace(case, events=(publication, *(event for event in case.events if event is not publication)))

    with pytest.raises(AssertionError, match="publication"):
        evaluate(reordered)


def test_reference_corpus_covers_terminal_evidence_corruptions() -> None:
    cases = tuple(reference_cases())
    for fault in ("missing", "duplicate", "foreign", "stale", "cross_target", "plan_mismatch", "contradictory"):
        case = next(
            case
            for case in cases
            if case.schedule_class == f"exact:verified:terminal_{fault}:accepted" and len(case.targets) == 2
        )
        result = evaluate(case)
        expected_release = () if fault != "missing" else ("t1",)
        assert result.released == expected_release
        if fault != "missing":
            assert result.invocation is ReferenceInvocation.INCONSISTENT
