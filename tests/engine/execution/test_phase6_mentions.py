# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import importlib.util
import pickle
from dataclasses import replace

import pytest

from anonymizer.engine.execution import accounting_plan as accounting_plan_module
from anonymizer.engine.execution.accounting_admission import _compile_accounting_plan
from anonymizer.engine.execution.accounting_ledger import _AccountingLedger
from anonymizer.engine.execution.accounting_plan import _AccountingLimits, _AccountingPlan
from anonymizer.engine.execution.graph import _DatumId, _TextDatum, _trivial_graph
from anonymizer.engine.execution.mention_admission import (
    _CandidateToken,
    _DetectedGraph,
    _finalize_mentions,
    _MentionLimits,
    _MentionProvenance,
    _MentionRejected,
    _MentionRejectionCode,
    _MentionTarget,
    _MentionTargetToken,
    _ProvisionalCandidate,
    _ValidationDecision,
    _ValidationDecisionKind,
)

_MENTION_LIMITS = _MentionLimits(
    max_candidates_per_target=8,
    max_mentions_per_target=8,
    max_label_bytes=64,
    max_source_slice_bytes=64,
)


def test_phase6_test_infrastructure() -> None:
    assert _AccountingPlan.__name__ == "_AccountingPlan"


def test_explicit_task_predecessor_delays_resolver_until_referenced_target_finalizes() -> None:
    predecessor_type = getattr(accounting_plan_module, "_TaskPredecessor", None)
    assert predecessor_type is not None, "Phase 6 requires typed cross-task predecessors"

    graph = _trivial_graph(
        (
            _TextDatum(_DatumId("target-a"), "Alice"),
            _TextDatum(_DatumId("target-b"), "A. Example"),
        )
    )
    plan = _compile_accounting_plan(
        graph,
        limits=_AccountingLimits(max_datums=2, max_datum_bytes=32, max_graph_bytes=64),
        stages=("finalize", "resolve"),
    )
    assert isinstance(plan, _AccountingPlan)
    finalize_a, finalize_b, resolve_a, resolve_b = plan.tasks
    extended = plan.with_task_predecessors((predecessor_type(finalize_b, resolve_a),))
    ledger: _AccountingLedger[str] = _AccountingLedger(extended)
    ledger.open()

    assert ledger.ready_tasks() == (finalize_a, finalize_b)
    ledger.accept_success(ledger.dispatch(finalize_a), "finalized-a")
    assert ledger.ready_tasks() == (finalize_b,)
    ledger.accept_success(ledger.dispatch(finalize_b), "finalized-b")
    assert ledger.ready_tasks() == (resolve_a, resolve_b)


def test_phase6_mention_module_exposes_strict_finalization_boundary() -> None:
    module_name = "anonymizer.engine.execution.mention_admission"
    assert importlib.util.find_spec(module_name) is not None, "Phase 6 mention admission module is missing"
    module = importlib.import_module(module_name)

    assert callable(getattr(module, "_finalize_mentions", None))


def test_finalization_anchors_unicode_and_keeps_repeated_equal_text_distinct() -> None:
    target_token = _MentionTargetToken()
    target = _MentionTarget(target_token, _DatumId("target"), "A😀A")
    first_token = _CandidateToken()
    emoji_token = _CandidateToken()
    last_token = _CandidateToken()
    candidates = (
        _ProvisionalCandidate(first_token, target_token, 0, 1, "A", "name", _MentionProvenance.SPAN_DETECTOR),
        _ProvisionalCandidate(emoji_token, target_token, 1, 2, "😀", "symbol", _MentionProvenance.EXACT_AUGMENTER),
        _ProvisionalCandidate(last_token, target_token, 2, 3, "A", "name", _MentionProvenance.SPAN_DETECTOR),
    )
    decisions = (
        _ValidationDecision(last_token, _ValidationDecisionKind.KEEP),
        _ValidationDecision(first_token, _ValidationDecisionKind.KEEP),
        _ValidationDecision(emoji_token, _ValidationDecisionKind.RECLASS, "emoji"),
    )

    result = _finalize_mentions((target,), candidates, decisions, limits=_MENTION_LIMITS)

    assert isinstance(result, _DetectedGraph)
    assert tuple(
        (mention.start, mention.end, mention.source_slice, mention.detector_label) for mention in result.mentions
    ) == ((0, 1, "A", "name"), (1, 2, "😀", "emoji"), (2, 3, "A", "name"))
    assert len({mention.id for mention in result.mentions}) == 3


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"start": True}, _MentionRejectionCode.INVALID_OFFSET),
        ({"start": -1}, _MentionRejectionCode.INVALID_OFFSET),
        ({"start": 1, "end": 1}, _MentionRejectionCode.INVALID_OFFSET),
        ({"end": 99}, _MentionRejectionCode.INVALID_OFFSET),
        ({"source_slice": "Mallory"}, _MentionRejectionCode.SOURCE_SLICE_MISMATCH),
        ({"detector_label": ""}, _MentionRejectionCode.CONTRADICTORY_CANDIDATE),
        ({"provenance": "legacy"}, _MentionRejectionCode.UNSUPPORTED_PROVENANCE),
    ],
)
def test_finalization_rejects_unverifiable_candidate_fields(
    changes: dict[str, object],
    code: _MentionRejectionCode,
) -> None:
    target_token = _MentionTargetToken()
    token = _CandidateToken()
    candidate = _ProvisionalCandidate(
        token,
        target_token,
        0,
        5,
        "Alice",
        "name",
        _MentionProvenance.SPAN_DETECTOR,
    )

    result = _finalize_mentions(
        (_MentionTarget(target_token, _DatumId("target"), "Alice"),),
        (replace(candidate, **changes),),
        (_ValidationDecision(token, _ValidationDecisionKind.KEEP),),
        limits=_MENTION_LIMITS,
    )

    assert result == _MentionRejected(code, target_token)


@pytest.mark.parametrize(
    ("decisions", "code"),
    [
        ((), _MentionRejectionCode.MISSING_DECISION),
        (
            (
                _ValidationDecisionKind.KEEP,
                _ValidationDecisionKind.KEEP,
            ),
            _MentionRejectionCode.DUPLICATE_DECISION,
        ),
    ],
)
def test_finalization_requires_exactly_one_terminal_decision(
    decisions: tuple[_ValidationDecisionKind, ...],
    code: _MentionRejectionCode,
) -> None:
    target_token = _MentionTargetToken()
    candidate_token = _CandidateToken()
    candidate = _ProvisionalCandidate(
        candidate_token,
        target_token,
        0,
        5,
        "Alice",
        "name",
        _MentionProvenance.SPAN_DETECTOR,
    )

    result = _finalize_mentions(
        (_MentionTarget(target_token, _DatumId("target"), "Alice"),),
        (candidate,),
        tuple(_ValidationDecision(candidate_token, kind) for kind in decisions),
        limits=_MENTION_LIMITS,
    )

    assert result == _MentionRejected(code, target_token)


def test_finalization_rejects_foreign_decision_and_overlapping_final_mentions() -> None:
    target_token = _MentionTargetToken()
    target = _MentionTarget(target_token, _DatumId("target"), "Alice")
    first_token = _CandidateToken()
    second_token = _CandidateToken()
    first = _ProvisionalCandidate(
        first_token,
        target_token,
        0,
        5,
        "Alice",
        "name",
        _MentionProvenance.SPAN_DETECTOR,
    )
    second = _ProvisionalCandidate(
        second_token,
        target_token,
        1,
        4,
        "lic",
        "alias",
        _MentionProvenance.EXACT_AUGMENTER,
    )
    foreign = _ValidationDecision(_CandidateToken(), _ValidationDecisionKind.KEEP)

    assert _finalize_mentions((target,), (first,), (foreign,), limits=_MENTION_LIMITS) == _MentionRejected(
        _MentionRejectionCode.FOREIGN_TOKEN
    )
    assert _finalize_mentions(
        (target,),
        (first, second),
        (
            _ValidationDecision(first_token, _ValidationDecisionKind.KEEP),
            _ValidationDecision(second_token, _ValidationDecisionKind.KEEP),
        ),
        limits=_MENTION_LIMITS,
    ) == _MentionRejected(_MentionRejectionCode.OVERLAP, target_token)


def test_finalization_collapses_exact_lineage_duplicates_and_honors_drop() -> None:
    target_token = _MentionTargetToken()
    first_token = _CandidateToken()
    dropped_token = _CandidateToken()
    first = _ProvisionalCandidate(
        first_token,
        target_token,
        0,
        5,
        "Alice",
        "name",
        _MentionProvenance.SPAN_DETECTOR,
    )
    dropped = _ProvisionalCandidate(
        dropped_token,
        target_token,
        6,
        9,
        "Bob",
        "name",
        _MentionProvenance.SPAN_DETECTOR,
    )

    result = _finalize_mentions(
        (_MentionTarget(target_token, _DatumId("target"), "Alice Bob"),),
        (first, first, dropped),
        (
            _ValidationDecision(first_token, _ValidationDecisionKind.KEEP),
            _ValidationDecision(dropped_token, _ValidationDecisionKind.DROP),
        ),
        limits=_MENTION_LIMITS,
    )

    assert isinstance(result, _DetectedGraph)
    assert tuple((mention.start, mention.end) for mention in result.mentions) == ((0, 5),)
    assert "Alice" not in repr(result)
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(result)
