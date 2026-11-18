# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sequences passed to the graph types are copied at construction: mutating the caller's list after ``prepare`` does
not change what the prepared graph does."""

from __future__ import annotations

from typing import Any

import anonymizer.graph as graph

IDENTITY = graph.Operation("identity", lambda text: text)


def verifier(name: str, passes: Any, checks: Any) -> graph.Operation:
    return graph.Operation(name, lambda text, target: graph.Assessment(passes(target), target),
                           {"text": "text", "target": "datum"}, {"assessment": "assessment"},
                           checks=checks, subject=("text",))  # fmt: skip


def failing_for(*failing: graph.Datum) -> graph.Operation:
    return verifier("v", lambda t: t not in failing, ("privacy",))


def requiring_privacy(*steps: graph.Operation) -> graph.Workflow:
    return graph.Workflow("w", steps, requires=("privacy",))


def recorder(seen: list[Any], port: str) -> graph.Operation:
    return graph.Operation("op", lambda text, **kw: seen.append(kw[port]) or text, {"text": "text", port: port})


def test_checks_swapped_after_prepare_keep_failed_target_withheld() -> None:
    scope = graph.GraphScope()
    d = scope.datum("Alice lives in Paris")
    fail_checks, pass_checks = ["privacy"], ["note"]
    v_fail = verifier("v_fail", lambda t: False, fail_checks)
    v_pass = verifier("v_pass", lambda t: True, pass_checks)
    prepared = graph.prepare(graph.ProtectionGraph(scope, (d,)), requiring_privacy(IDENTITY, v_fail, v_pass))
    fail_checks[:], pass_checks[:] = ["note"], ["privacy"]
    assert graph.protect(prepared).outcome(d).protected is False


def test_steps_reordered_after_prepare_keep_failed_target_withheld() -> None:
    scope = graph.GraphScope()
    d = scope.datum("x")
    steps = [IDENTITY, verifier("v_fail", lambda t: False, ("privacy",)), verifier("v_pass", lambda t: True, ("note",))]
    workflow = graph.Workflow("w", steps, requires=("privacy",))  # ty: ignore[invalid-argument-type]
    prepared = graph.prepare(graph.ProtectionGraph(scope, (d,)), workflow)
    steps[1], steps[2] = steps[2], steps[1]
    assert graph.protect(prepared).outcome(d).protected is False


def test_dependencies_cleared_after_prepare_keep_dependent_withheld() -> None:
    scope = graph.GraphScope()
    a, b = scope.datum("a"), scope.datum("b")
    deps = [graph.Dependency(dependent=b, on=a)]
    g = graph.ProtectionGraph(scope, (a, b), dependencies=deps)  # ty: ignore[invalid-argument-type]
    prepared = graph.prepare(g, requiring_privacy(failing_for(a)))
    deps.clear()
    assert graph.protect(prepared).outcome(b).protected is False


def test_groups_cleared_after_prepare_keep_member_withheld() -> None:
    scope = graph.GraphScope()
    a, b = scope.datum("a"), scope.datum("b")
    groups = [graph.OutputGroup((a, b))]
    g = graph.ProtectionGraph(scope, (a, b), groups=groups)  # ty: ignore[invalid-argument-type]
    prepared = graph.prepare(g, requiring_privacy(failing_for(a)))
    groups.clear()
    assert graph.protect(prepared).outcome(b).protected is False


def test_group_members_shrunk_after_prepare_keep_member_withheld() -> None:
    scope = graph.GraphScope()
    a, b = scope.datum("a"), scope.datum("b")
    members = [a, b]
    g = graph.ProtectionGraph(scope, (a, b), groups=(graph.OutputGroup(members),))  # ty: ignore[invalid-argument-type]
    prepared = graph.prepare(g, requiring_privacy(failing_for(a)))
    members.remove(a)
    assert graph.protect(prepared).outcome(b).protected is False


def test_mention_appended_after_construction_is_not_delivered() -> None:
    scope = graph.GraphScope()
    a = scope.datum("Alice")
    seen: list[Any] = []
    mentions = [graph.Mention(a, 0, 5, "first_name")]
    g = graph.ProtectionGraph(scope, (a,), mentions=mentions)  # ty: ignore[invalid-argument-type]
    prepared = graph.prepare(g, graph.Workflow("w", (recorder(seen, "mentions"),)))
    mentions.append(graph.Mention(a, 2, 999, "bogus"))  # out of range and overlapping: rejected at construction
    graph.protect(prepared)
    assert [(m.start, m.end) for m in seen[-1]] == [(0, 5)]


def test_context_scope_appended_after_construction_is_not_delivered() -> None:
    scope = graph.GraphScope()
    a = scope.datum("a")
    foreign = graph.GraphScope().datum("SECRET from another graph")
    seen: list[Any] = []
    context: list[graph.ContextScope] = []
    g = graph.ProtectionGraph(scope, (a,), context=context)  # ty: ignore[invalid-argument-type]
    prepared = graph.prepare(g, graph.Workflow("w", (recorder(seen, "context"),)))
    context.append(graph.ContextScope(a, (foreign,)))  # a foreign datum: rejected at construction
    graph.protect(prepared)
    assert not any("SECRET" in str(item.text) for item in seen[-1])


def test_context_item_appended_after_construction_is_not_delivered() -> None:
    scope = graph.GraphScope()
    a = scope.datum("a")
    foreign = graph.GraphScope().datum("SECRET from another graph")
    seen: list[Any] = []
    items: list[Any] = []
    g = graph.ProtectionGraph(scope, (a,), context=(graph.ContextScope(a, items),))  # ty: ignore[invalid-argument-type]
    prepared = graph.prepare(g, graph.Workflow("w", (recorder(seen, "context"),)))
    items.append(foreign)
    graph.protect(prepared)
    assert not any("SECRET" in str(item.text) for item in seen[-1])


def test_target_appended_after_prepare_is_not_run() -> None:
    scope = graph.GraphScope()
    a, extra = scope.datum("a"), scope.datum("extra")
    targets = [a]
    g = graph.ProtectionGraph(scope, targets)  # ty: ignore[invalid-argument-type]
    prepared = graph.prepare(g, graph.Workflow("w", (IDENTITY,)))
    targets.append(extra)
    result = graph.protect(prepared)
    assert result.targets == (a,) and len(result.outcomes) == 1 and result.outcome(a).terminal == "success"
