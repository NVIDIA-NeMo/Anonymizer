# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SP-28 typed context bindings: structured kinds, compatibility before retrieval, invalid items."""

from __future__ import annotations

from typing import Any

from anonymizer.engine.graph_core.authoring import (
    ContextScope,
    ContextSource,
    GraphScope,
    Limits,
    Operation,
    ProtectionGraph,
)
from anonymizer.engine.graph_core.kernel import register_kind
from tests.spec_programs._builder import Checks, rejected, run

PERSONA = register_kind("sp28-persona", lambda v: isinstance(v, dict) and "role" in v, lambda v: len(repr(v)))


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


def program() -> list[dict[str, str]]:
    c = Checks("SP-28")
    persona = {"role": "nurse"}
    calls: list[str] = []
    seen: list[tuple] = []
    _bind(PERSONA, [persona], calls, seen)
    item = seen[0][0] if seen else None
    c.check(
        "E28.1",
        item is not None and item.text == persona and item.kind == PERSONA,
        f"delivered kind={getattr(item, 'kind', None)}, value type={type(getattr(item, 'text', None)).__name__}",
    )
    calls = []
    code = rejected(lambda: _bind(PERSONA, [persona], calls, [], consumer="context"))
    c.check("E28.2", code == "port_kind_mismatch" and not calls, f"code={code}, provider calls={len(calls)}")
    seen = []
    statuses = [
        _bind(PERSONA, [bad], [], seen).retrievals[0].status for bad in ({"name": "x"}, {"role": Unprintable()})
    ]
    c.check(
        "E28.3",
        statuses == ["invalid_item"] * 2 and not seen,
        f"statuses={statuses}, operation ran={bool(seen)}",
    )
    return c.lines
