# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle
from collections.abc import Callable
from typing import Any

import pytest

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
    op = graph.Operation(
        "read", lambda text, context: calls.append(text) or text, inputs={"text": "text", "context": "context"}
    )

    def protect(*items: graph.Datum | graph.ContextSource) -> graph.ProtectionResult:
        g = graph.ProtectionGraph(scope, (target,), context=(graph.ContextScope(target, items),))
        return graph.protect(graph.prepare(g, graph.Workflow("w", (op,)), limits=graph.Limits(max_context_bytes=10)))

    result = protect(scope.datum("SSSSSS"), source)
    assert calls == []
    assert result.retrievals[0].status == "over_byte_ceiling"
    assert (result.outcome(target).terminal, result.outcome(target).reason) == ("failed", "binding_failed")
    assert protect(source).outcome(target).terminal == "success"


def _bind_with_copy(name: str, copy: Callable[[Any], Any]) -> tuple[graph.ProtectionResult, list[str]]:
    """Source A hands over a shared dict; source B, retrieved after A was checked and measured, mutates it."""
    kind = graph.register_kind(name, lambda v: isinstance(v, dict), lambda v: len(v["text"].encode()), copy=copy)
    shared = {"text": "safe"}

    def source_b() -> list[dict[str, str]]:
        shared["text"] = "x" * 1000
        return []

    seen: list[str] = []
    op = graph.Operation(
        "consume",
        lambda text, context: seen.extend(item.text["text"] for item in context) or text,
        inputs={"text": "text", "context": kind},
    )
    scope = graph.GraphScope()
    target = scope.datum("hello")
    sources = (
        graph.ContextSource("A", lambda: [shared], max_items=1, kind=kind),
        graph.ContextSource("B", source_b, max_items=1, kind=kind),
    )
    g = graph.ProtectionGraph(scope, (target,), context=(graph.ContextScope(target, sources),))
    return graph.protect(graph.prepare(g, graph.Workflow("w", (op,)), limits=graph.Limits(max_context_bytes=10))), seen


def test_context_items_are_the_kinds_copies_checked_and_measured() -> None:
    copies: list[object] = []
    result, seen = _bind_with_copy("test-context-copy", lambda v: copies.append(v) or dict(v))
    assert [(r.status, r.observed_bytes) for r in result.retrievals] == [("ok", 4), ("ok", 0)]
    assert result.outcomes[0].terminal == "success"
    assert len(copies) == 1
    assert seen == ["safe"]


def test_context_item_whose_copy_raises_is_invalid() -> None:
    def copy(value: dict[str, str]) -> dict[str, str]:
        raise RuntimeError("no copy")

    result, seen = _bind_with_copy("test-context-copy-raises", copy)
    assert result.retrievals[0].status == "invalid_item"
    assert (result.outcomes[0].terminal, result.outcomes[0].reason) == ("failed", "binding_failed")
    assert seen == []


def test_cancel_token_is_followed_but_never_rewired() -> None:
    token, outer = graph.CancelToken(), graph.CancelToken()
    nested: list[str] = []

    def inner(text: str) -> str:
        nested.append(_run(lambda text: text, token))
        return text

    def middle(text: str) -> str:
        nested.append(_run(inner))
        return text

    assert _run(middle, token) == "success"
    assert _run(inner, outer) == "success"
    assert nested == ["success"] * 3
    outer.cancel()
    assert not token.cancelled
    assert _run(lambda text: text, token) == "success"

    seen: list[str] = []
    late = graph.CancelToken()

    def cancels_outer(text: str) -> str:
        late.cancel()
        seen.append(_run(lambda text: text))
        return text

    _run(cancels_outer, late)
    assert seen == ["cancelled"]


def test_verifier_only_release_maps_target_to_its_source_text() -> None:
    scope = graph.GraphScope()
    target = scope.datum("verified text")
    check = graph.Operation.verifier(
        "v",
        lambda text, target: graph.Assessment(True, target),
        {"text": "text", "target": "datum"},
        checks=("c",),
        subject=("text",),
    )
    result = graph.protect(
        graph.prepare(graph.ProtectionGraph(scope, (target,)), graph.Workflow("w", (check,), requires=("c",)))
    )
    assert result.protected == {target: "verified text"}


def test_static_context_is_not_encoded_without_a_byte_ceiling() -> None:
    scope = graph.GraphScope()
    target = scope.datum("hello")
    op = graph.Operation("read", lambda text, context: text, inputs={"text": "text", "context": "context"})
    g = graph.ProtectionGraph(scope, (target,), context=(graph.ContextScope(target, (scope.datum("\ud800"),)),))
    assert graph.protect(graph.prepare(g, graph.Workflow("w", (op,)))).outcome(target).terminal == "success"


def test_final_text_of_another_kind_is_rejected() -> None:
    kind = graph.register_kind("test-dict-text", lambda value: isinstance(value, dict))
    scope = graph.GraphScope()
    g = graph.ProtectionGraph(scope, (scope.datum("Alice"),))
    check = graph.Operation.verifier(
        "v",
        lambda text, target: graph.Assessment(True, target),
        {"text": kind, "target": "datum"},
        checks=("c",),
        subject=("text",),
    )
    wrap = graph.Operation("wrap", lambda text: {"value": text}, outputs={"text": kind})
    with pytest.raises(graph.GraphRejection) as rejected:
        graph.prepare(g, graph.Workflow("w", (wrap, check), requires=("c",)))
    assert rejected.value.code == "final_text_not_text"


@pytest.mark.parametrize("release", ["Annotation", "annotated", "protected"])
def test_nested_release_other_than_protected_is_rejected(release: str) -> None:
    scope = graph.GraphScope()
    g = graph.ProtectionGraph(scope, (scope.datum("Alice"),))
    check = graph.Operation.verifier(
        "v",
        lambda text, target: graph.Assessment(True, target),
        {"text": "text", "target": "datum"},
        checks=("c",),
        subject=("text",),
    )
    inner = graph.Workflow("tag", (graph.Operation("wrap", lambda text: f"<{text}>"), check), release=release)  # ty: ignore[invalid-argument-type]
    outer = graph.Workflow("w", (inner,), requires=("c",))
    if release == "protected":
        assert graph.protect(graph.prepare(g, outer)).protected
        return
    with pytest.raises(graph.GraphRejection) as rejected:
        graph.prepare(g, outer)
    assert rejected.value.code == "annotation_in_protected"


@pytest.mark.parametrize(
    ("steps", "duplicate"),
    [
        (lambda op: (op("rewrite"), op("rewrite")), "rewrite"),
        (lambda op: (graph.Workflow("inner", (op("step"),)), op("inner/step")), "inner/step"),
    ],
    ids=["flat", "nested-path"],
)
def test_duplicate_flattened_operation_names_are_rejected(steps: Callable[[Any], tuple], duplicate: str) -> None:
    """A producer read selects an operation by its flattened name, so two operations may not share one."""
    scope = graph.GraphScope()
    g = graph.ProtectionGraph(scope, (scope.datum("Alice"),))
    with pytest.raises(graph.GraphRejection) as rejected:
        graph.prepare(g, graph.Workflow("w", steps(lambda name: graph.Operation(name, lambda text: text))))
    assert (rejected.value.code, rejected.value.detail) == ("duplicate_operation_name", duplicate)


def test_value_from_an_unknown_producer_is_none() -> None:
    scope = graph.GraphScope()
    target = scope.datum("Alice")
    workflow = graph.Workflow("w", (graph.Operation("rewrite", lambda text: "candidate"),))
    result = graph.protect(graph.prepare(graph.ProtectionGraph(scope, (target,)), workflow))
    assert result.value(target, producer="rewrite") == "candidate"
    assert result.value(target, producer="unknown") is None


def test_annotation_nested_in_protected_workflow_is_rejected() -> None:
    scope = graph.GraphScope()
    target = scope.datum("Alice")
    g = graph.ProtectionGraph(scope, (target,))
    check = graph.Operation.verifier(
        "v",
        lambda text, target: graph.Assessment(True, target),
        {"text": "text", "target": "datum"},
        checks=("c",),
        subject=("text",),
    )
    inner = graph.Workflow("tag", (graph.Operation("wrap", lambda text: f"<{text}>"), check), release="annotation")
    with pytest.raises(graph.GraphRejection) as rejected:
        graph.prepare(g, graph.Workflow("w", (inner,), requires=("c",)))
    assert rejected.value.code == "annotation_in_protected"
    outer = graph.Workflow("w", (inner,), release="annotation", requires=("c",))
    assert graph.protect(graph.prepare(g, outer)).protected == {}


def _prepare(cycle: bool = False, **kw: Any) -> graph.PreparedProtection:
    scope = graph.GraphScope()
    a, b = scope.datum("a"), scope.datum("b")
    deps = (graph.Dependency(a, b), graph.Dependency(b, a)) if cycle else ()
    g = graph.ProtectionGraph(scope, (a, b), dependencies=deps)
    return graph.prepare(g, graph.Workflow("w", (graph.Operation("op", lambda text: text),)), **kw)


@pytest.mark.parametrize(
    "build, code",
    [
        (lambda: graph.GraphScope().datum(b"Alice"), "datum_not_text"),  # ty: ignore[invalid-argument-type]
        (lambda: _prepare(capabilities="judge"), "names_not_tuple"),
    ],
)
def test_wrong_python_type_is_also_a_type_error(build: Callable[[], Any], code: str) -> None:
    with pytest.raises(TypeError) as info:
        build()
    rejection = info.value
    assert isinstance(rejection, graph.GraphRejection) and isinstance(rejection, ValueError) and rejection.code == code
    copy = pickle.loads(pickle.dumps(rejection))
    assert type(copy) is type(rejection) and (copy.code, copy.detail) == (code, rejection.detail)


@pytest.mark.parametrize(
    "build, code",
    [
        (lambda: _prepare(limits=graph.Limits(max_targets=1)), "over_limit"),
        (lambda: _prepare(cycle=True), "dependency_cycle"),
        (lambda: graph.Requests.exact(-1), "invalid_request_count"),
    ],
)
def test_other_rejections_are_value_errors_only(build: Callable[[], Any], code: str) -> None:
    try:
        build()
    except ValueError as caught:
        assert isinstance(caught, graph.GraphRejection) and caught.code == code
        assert not isinstance(caught, TypeError)
    else:
        pytest.fail("no rejection")


def test_subclass_with_extra_arguments_constructs_and_keeps_its_class() -> None:
    class Hinted(graph.GraphRejection):
        def __init__(self, code: str, detail: str = "", *, hint: str = "") -> None:
            super().__init__(code, detail)
            self.hint = hint

    rejection: Hinted = Hinted("names_not_tuple", hint="h")  # ty pin: the constructor is typed as the subclass
    assert type(rejection) is Hinted and rejection.hint == "h" and not isinstance(rejection, TypeError)
