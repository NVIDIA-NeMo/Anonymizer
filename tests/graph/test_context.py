# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context binding: caller-owned providers are retrieved within declared bounds and typed kinds before the operation runs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from anonymizer.engine.graph_core.authoring import (
    ContextScope,
    ContextSource,
    GraphScope,
    Limits,
    Operation,
    ProtectionGraph,
)
from anonymizer.engine.graph_core.kernel import GraphRejection, register_kind
from tests.graph.helpers import run

PERSONA = register_kind("sp28-persona", lambda v: isinstance(v, dict) and "role" in v, lambda v: len(repr(v)))


def _bound(provider: Callable[[], list[str]], max_items: int = 2, max_bytes: int | None = None) -> tuple:
    scope = GraphScope()
    t = scope.datum("Alice met Bob.")
    seen: list[tuple] = []

    def op(text: str, context: tuple) -> str:
        seen.append(context)
        return text

    src = ContextSource("history", provider, max_items=max_items, max_bytes=max_bytes)
    g = ProtectionGraph(scope, (t,), context=(ContextScope(t, (src,)),))
    r = run(g, Operation("transform", op, {"text": "text", "context": "context"}))
    return r, r.outcome(t), seen


def test_bounded_provider_delivers_context_once() -> None:
    r, o, seen = _bound(lambda: ["one", "two"])
    assert o.terminal == "success"
    assert len(seen) == 1
    assert len(seen[0]) == 2
    assert len(r.retrievals) == 1


def _broken() -> list[str]:
    raise ConnectionError("provider down")


@pytest.mark.parametrize(
    ("provider", "max_bytes", "status"),
    [
        (lambda: ["one", "two", "three"], None, "overflow"),
        (lambda: ["x" * 40], 10, "over_byte_ceiling"),
        (_broken, None, "failed"),
    ],
)
def test_failed_retrieval_fails_target_without_running_operation(
    provider: Callable[[], list[str]], max_bytes: int | None, status: str
) -> None:
    r, o, seen = _bound(provider, max_bytes=max_bytes)
    assert not seen
    assert o.terminal == "failed"
    assert r.retrievals[0].status == status


def _two(a_items: list[str], b_provider: Callable[[], list[str]], a_max: int = 1) -> tuple:
    scope = GraphScope()
    t = scope.datum("Alice met Bob.")
    seen: list[tuple] = []

    def op(text: str, context: tuple) -> str:
        seen.append(context)
        return text

    a = ContextSource("A", lambda: a_items, max_items=a_max)
    b = ContextSource("B", b_provider, max_items=2)
    g = ProtectionGraph(scope, (t,), context=(ContextScope(t, (a, b)),))
    r = run(g, Operation("op", op, {"text": "text", "context": "context"}))
    return r, r.outcome(t), seen


def test_two_sources_deliver_items_in_source_order() -> None:
    r, _, seen = _two(["a1"], lambda: ["b1", "b2"])
    assert seen
    assert [i.source for i in seen[0]] == ["A", "B", "B"]
    assert [x.source for x in r.retrievals] == ["A", "B"]
    assert len(r.attempts) == 1


def test_one_failed_source_blocks_the_operation() -> None:
    def down() -> list[str]:
        raise TimeoutError("B down")

    r, o, seen = _two(["a1"], down)
    status = {x.source: x.status for x in r.retrievals}
    assert status["B"] == "failed"
    assert not seen
    assert o.terminal != "success"
    assert o.qualification != "met"


def test_large_item_without_byte_cap_is_delivered_whole() -> None:
    big = "x" * (1024 * 1024 + 1)
    _, o, seen = _two([big], list)
    assert o.terminal == "success"
    assert seen
    assert seen[0][0].text == big


def test_retrieval_records_declared_and_observed_counts_and_utf8_bytes() -> None:
    r, _, _ = _two(["a1", "a2"], list, a_max=5)
    ra = r.retrievals[0]
    assert (ra.declared_max_items, ra.observed_items) == (5, 2)
    r, _, _ = _two(["Zoë Ågren"], list)
    assert r.retrievals[0].observed_bytes == 11


class Unprintable:
    def __repr__(self) -> str:
        raise ValueError("no measure")


def _bind(kind: str, items: list[Any], calls: list[str], seen: list[tuple], consumer: str = PERSONA) -> Any:
    def provider() -> list[Any]:
        calls.append("provider")
        return items

    def op(text: str, context: tuple) -> str:
        seen.append(context)
        return text

    scope = GraphScope()
    t = scope.datum("Alice met Bob.")
    source = ContextSource("persona", provider, max_items=2, kind=kind)
    g = ProtectionGraph(scope, (t,), context=(ContextScope(t, (source,)),))
    return run(g, Operation("op", op, {"text": "text", "context": consumer}), limits=Limits())


def test_typed_context_is_delivered_with_its_kind() -> None:
    persona = {"role": "nurse"}
    seen: list[tuple] = []
    _bind(PERSONA, [persona], [], seen)
    item = seen[0][0]
    assert item.text == persona
    assert item.kind == PERSONA


def test_kind_mismatch_is_rejected_before_retrieval() -> None:
    calls: list[str] = []
    with pytest.raises(GraphRejection) as info:
        _bind(PERSONA, [{"role": "nurse"}], calls, [], consumer="context")
    assert info.value.code == "port_kind_mismatch"
    assert not calls


@pytest.mark.parametrize("bad", [{"name": "x"}, {"role": Unprintable()}])
def test_invalid_typed_item_blocks_the_operation(bad: dict) -> None:
    seen: list[tuple] = []
    assert _bind(PERSONA, [bad], [], seen).retrievals[0].status == "invalid_item"
    assert not seen
