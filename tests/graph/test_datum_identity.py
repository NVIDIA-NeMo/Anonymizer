# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Datum identity: data are unforgeable, scope-bound, order-independent and never correlated by text."""

from __future__ import annotations

import copy
import dataclasses

import pytest

from anonymizer.engine.graph_core.authoring import GraphScope, Mention, Operation, ProtectionGraph, Workflow
from anonymizer.engine.graph_core.execution import prepare, protect
from anonymizer.engine.graph_core.kernel import GraphRejection
from tests.graph.helpers import Recorder, run


def test_datum_is_frozen_and_copies_equal() -> None:
    x = GraphScope().datum("same text")
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(x, "_id", "forged")
    assert copy.copy(x) == x
    assert copy.deepcopy(x) == x


def test_foreign_datum_is_rejected_before_any_callback() -> None:
    x = GraphScope().datum("same text")
    rec = Recorder()
    with pytest.raises(GraphRejection) as info:
        run(ProtectionGraph(GraphScope(), (x,)), rec.transform())
    assert info.value.code == "foreign_datum"
    assert rec.count() == 0


def test_results_follow_identity_not_declaration_order() -> None:
    scope = GraphScope()
    x, y, z = (scope.datum("same text") for _ in range(3))
    rec = Recorder()
    r1 = run(ProtectionGraph(scope, (x, y, z)), rec.transform(fn=str.upper))
    r2 = run(ProtectionGraph(scope, (z, x, y)), rec.transform(fn=str.upper))
    for d in (x, y, z):
        assert (r1.value(d), r1.outcome(d)) == (r2.value(d), r2.outcome(d))


def test_repeated_runs_share_no_runtime_refs() -> None:
    scope = GraphScope()
    x = scope.datum("same text")
    p = prepare(ProtectionGraph(scope, (x,)), Workflow("w", (Operation("t", lambda text: text.upper()),)))
    a, b = protect(p), protect(p)
    refs_a, refs_b = {t.ref for t in a.attempts}, {t.ref for t in b.attempts}
    assert refs_a
    assert refs_b
    assert not refs_a & refs_b
    assert a.outcome(x).terminal == "success"


def test_equal_text_does_not_make_equal_identities() -> None:
    scope = GraphScope()
    d1, d2 = scope.datum("Alice"), scope.datum("Alice")
    assert d1 != d2
    assert Mention(d1, 0, 5, "first_name") != Mention(d2, 0, 5, "first_name")
