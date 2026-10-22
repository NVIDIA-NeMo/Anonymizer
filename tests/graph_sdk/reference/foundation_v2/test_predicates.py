# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hand-derived discriminators, separate from generated expected answers."""

from __future__ import annotations

from tests.graph_sdk.reference.foundation_v2.predicates import (
    OccurrenceEvent,
    RequestEvent,
    final_loop,
    has_disjoint_outputs,
    has_total_order,
    is_legal_activation,
    is_legal_requests,
    is_partition,
    map_join,
    matches_window,
    may_close,
    may_reissue,
    terminal_projection,
)


def test_data() -> None:
    assert has_total_order((0, 1, 2), ((0, 1), (1, 2)))
    assert not has_total_order((0, 1), ((0, 1), (1, 0)))
    assert matches_window("aé", "é", 1, 2)
    assert not matches_window("aé", "a", 1, 2)
    assert matches_window("aé", "", 1, 1)
    assert has_disjoint_outputs(((0, 0, 1), (0, 1, 2)))
    assert not has_disjoint_outputs(((0, 0, 2), (0, 1, 2)))
    assert has_disjoint_outputs(((0, 0, 2), (1, 0, 2)))
    assert is_partition(frozenset({0, 1}), (frozenset({0, 1}),))
    assert not is_partition(frozenset({0, 1}), (frozenset({0, 1}), frozenset({1})))


def test_renaming_and_relation_independence() -> None:
    assert has_total_order((10, 20, 30), ((10, 20), (20, 30)))
    assert has_disjoint_outputs(((42, 0, 1), (42, 1, 2)))
    # Equal text never supplies identity, a readiness edge or an atomic relation.
    coherence = (frozenset({0, 1}),)
    atomic = (frozenset({0}), frozenset({1}))
    assert is_partition(frozenset({0, 1}), coherence)
    assert is_partition(frozenset({0, 1}), atomic)
    assert coherence != atomic
    assert has_total_order((0, 1), ())


def test_activation() -> None:
    start = OccurrenceEvent("A", "start")
    done = OccurrenceEvent("A", "terminal", "success")
    assert is_legal_activation(frozenset({"A"}), (), (start, done), True)
    assert not is_legal_activation(frozenset({"A"}), (), (done, start), True)
    assert not is_legal_activation(frozenset({"A"}), (), (start, done, done), True)
    assert not is_legal_activation(frozenset({"B"}), (), (start, done), True)
    assert not is_legal_activation(frozenset({"B"}), (), (start,), False)
    assert not is_legal_activation(frozenset({"A"}), (), (start,), True)
    assert not is_legal_activation(frozenset({"A", "B"}), (("A", "B"),), (OccurrenceEvent("B", "start"),), False)
    assert map_join("success", True, frozenset(), ()) == "success"
    assert map_join("failed", True, frozenset(), ()) == "blocked"
    assert map_join("success", False, frozenset(), ()) == "pending"
    assert map_join("success", True, frozenset({"A", "B"}), (("A", "success"),)) == "pending"
    assert map_join("success", True, frozenset({"A", "B"}), (("A", "success"), ("B", "failure"))) == "blocked"
    assert final_loop(("continue", "exit"), 2, True) == ("success", 1)
    assert final_loop(("continue", "continue"), 2, True) == ("inconsistent", None)
    assert final_loop((), 0, True) == ("inconsistent", None)
    assert final_loop((), 0, False) == ("success", None)


def test_requests() -> None:
    reserve = RequestEvent(0, "reserve", (0,))
    dispatch = RequestEvent(0, "dispatch")
    lost = RequestEvent(0, "terminal", value="lost")
    settled = RequestEvent(0, "settlement", usage=3, stopped=True)
    assert is_legal_requests((reserve, dispatch, lost, settled), 1)
    assert not is_legal_requests((dispatch, reserve), 1)
    assert not is_legal_requests((dispatch,), 1)
    assert not is_legal_requests((reserve,), 0)
    assert not is_legal_requests((reserve, RequestEvent(1, "reserve", (1,))), 1)
    cancel = RequestEvent(0, "cancel_reserved")
    assert is_legal_requests((reserve, cancel, RequestEvent(1, "reserve", (1,))), 1)
    assert terminal_projection((reserve, dispatch, lost), 0) == ("lost", None, False)
    late = RequestEvent(0, "result", (0,))
    assert terminal_projection((reserve, dispatch, lost, late, settled), 0) == ("lost", 3, True)
    assert terminal_projection((reserve, dispatch, late), 0) == ("success", None, False)
    assert terminal_projection((reserve, dispatch, RequestEvent(0, "result", (0, 0))), 0)[0] == "inconsistent"
    assert terminal_projection((reserve, dispatch, settled, RequestEvent(0, "settlement", usage=0)), 0)[1] == 3


def test_reissue_and_resources() -> None:
    assert may_reissue("executor", "before_acceptance", "rejected_before_acceptance", "retry", 1, 2, False)
    assert not may_reissue("executor", "before_acceptance", "transport_unknown", "retry", 1, 2, False)
    assert not may_reissue("executor", "idempotent", "retryable", "retry", 2, 2, False)
    assert not may_reissue("none", "idempotent", "retryable", "retry", 0, 2, False)
    assert may_reissue("executor", "idempotent", "permanent", "failover", 1, 2, True)
    assert not may_reissue("executor", "idempotent", "permanent", "failover", 1, 2, False)
    assert not may_close(False, False, False, True)
    assert not may_close(True, True, False, True)
    assert not may_close(True, False, True, False)
    assert may_close(True, False, True, True)
