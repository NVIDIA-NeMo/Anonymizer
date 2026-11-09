# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable

import anonymizer.graph as graph


def _run(fn: Callable[[str], str], cancel: graph.CancelToken | None = None) -> str:
    scope = graph.GraphScope()
    workflow = graph.Workflow("w", (graph.Operation("op", fn),))
    result = graph.protect(graph.prepare(graph.ProtectionGraph(scope, (scope.datum("x"),)), workflow), cancel=cancel)
    return result.outcomes[0].terminal


def test_raising_output_check_fails_only_its_target() -> None:
    kind = graph.register_kind("test-raising-check", lambda value: value["ok"])
    scope = graph.GraphScope()
    bad, good = scope.datum("bad"), scope.datum("good")
    op = graph.Operation("produce", lambda text: {} if text == "bad" else {"ok": True}, outputs={"out": kind})
    result = graph.protect(graph.prepare(graph.ProtectionGraph(scope, (bad, good)), graph.Workflow("w", (op,))))
    assert (result.outcome(bad).terminal, result.outcome(bad).reason) == ("failed", "invalid_output")
    assert result.outcome(good).terminal == "success"


def test_retrieved_context_counts_on_top_of_static_context() -> None:
    scope = graph.GraphScope()
    target = scope.datum("hello")
    source = graph.ContextSource("src", lambda: ["RRRRRR"], max_items=1)
    calls: list[str] = []
    op = graph.Operation("read", lambda text, context: calls.append(text) or text, inputs={"text": "text", "context": "context"})

    def protect(*items: graph.Datum | graph.ContextSource) -> graph.ProtectionResult:
        g = graph.ProtectionGraph(scope, (target,), context=(graph.ContextScope(target, items),))
        return graph.protect(graph.prepare(g, graph.Workflow("w", (op,)), limits=graph.Limits(max_context_bytes=10)))

    result = protect(scope.datum("SSSSSS"), source)
    assert calls == []
    assert result.retrievals[0].status == "over_byte_ceiling"
    assert (result.outcome(target).terminal, result.outcome(target).reason) == ("failed", "binding_failed")
    assert protect(source).outcome(target).terminal == "success"


def test_cancel_token_is_followed_but_never_rewired() -> None:
    token, outer = graph.CancelToken(), graph.CancelToken()
    inner = lambda text: _run(lambda t: t, token) and text  # noqa: E731
    assert _run(lambda text: _run(inner) and text, token) == "success"
    assert _run(inner, outer) == "success"
    outer.cancel()
    assert not token.cancelled
    assert _run(lambda text: text, token) == "success"

    seen: list[str] = []
    late = graph.CancelToken()

    def cancels_outer(text: str) -> str:
        late.cancel()
        seen.append(_run(lambda t: t))
        return text

    _run(cancels_outer, late)
    assert seen == ["cancelled"]


def test_verifier_only_release_maps_target_to_its_source_text() -> None:
    scope = graph.GraphScope()
    target = scope.datum("verified text")
    check = graph.Operation.verifier(
        "v", lambda text, target: graph.Assessment(True, target), {"text": "text", "target": "datum"}, checks=("c",)
    )
    result = graph.protect(graph.prepare(graph.ProtectionGraph(scope, (target,)), graph.Workflow("w", (check,), requires=("c",))))
    assert result.protected == {target: "verified text"}
