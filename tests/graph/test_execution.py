# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution-only and verified runs: one call per target, Protected release only when verification is met."""

from __future__ import annotations

import threading

from tests.graph.helpers import NAME, Recorder, graph, run


def test_execution_only_run_transforms_each_target_once_without_protection() -> None:
    text = f"{NAME} met Bob."
    g, (a, b) = graph(text, text)
    rec = Recorder()
    result = run(g, rec.transform())
    attempts = [x.target for x in result.attempts]
    assert rec.count() == 2
    assert attempts.count(a) == 1
    assert attempts.count(b) == 1

    content = result.content()
    assert len(content) == 2
    assert all(t == "[NAME] met Bob." for _, t in content)
    assert a != b

    quals = [result.outcome(d).qualification for d in (a, b)]
    assert quals == ["not_assessed"] * 2
    assert len(result.protected) == 0
    assert all(NAME not in t for _, t in content)

    before = rec.count()
    again = (result.content(), result.protected, result.summary().render(), result.ledger())
    assert rec.count() == before
    assert again[0] == content

    me = threading.get_ident()
    assert all(tid == me for _, _, tid in rec.calls)


def test_met_verification_releases_protected_text_in_order() -> None:
    g, (a,) = graph(f"{NAME} met Bob.")
    rec = Recorder()
    r = run(g, rec.transform(), rec.verifier())
    o = r.outcome(a)
    assert (rec.count("transform"), rec.count("verifier")) == (1, 1)
    assert o.qualification == "met"
    assert r.protected == {a: "[NAME] met Bob."}

    order = [op for op, _, _ in rec.calls]
    received = [t for op, t, _ in rec.calls if op == "verifier"]
    assert order == ["transform", "verifier"]
    assert received == ["[NAME] met Bob."]


def test_unmet_verification_withholds_protected_text() -> None:
    g, (a,) = graph(f"{NAME} met Bob.")
    rec = Recorder()
    r = run(g, rec.transform(fn=lambda t: t), rec.verifier())
    o = r.outcome(a)
    assert o.qualification != "met"
    assert len(r.protected) == 0
    assert o.terminal == "success"
