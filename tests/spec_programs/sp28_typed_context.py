# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SP-28 typed context bindings: structured kinds, compatibility before retrieval, invalid items (D.REQ9.s2)."""

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
UNMEASURED = register_kind("sp28-unmeasured", lambda v: True)


def _bind(kind: str, items: list[Any], calls: list[str], seen: list[tuple], **limits: Any) -> Any:
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
    return run(g, Operation("op", op, {"text": "text", "context": "context"}), limits=Limits(**limits))


def program() -> list[dict[str, str]]:
    c = Checks("SP-28")
    persona = {"role": "nurse"}
    calls: list[str] = []
    seen: list[tuple] = []
    _bind(PERSONA, [persona], calls, seen)
    item = seen[0][0] if seen else None
    c.check(
        "D.REQ9.1",
        item is not None and item.text == persona and item.kind == PERSONA,
        f"delivered kind={getattr(item, 'kind', None)}, value type={type(getattr(item, 'text', None)).__name__}",
    )
    calls = []
    code = rejected(lambda: _bind(UNMEASURED, [persona], calls, [], max_context_bytes=100))
    c.check("D.REQ9.2", code == "unsupported_context_kind" and not calls, f"code={code}, provider calls={len(calls)}")
    seen = []
    r = _bind(PERSONA, [{"name": "x"}], [], seen)
    status = r.retrievals[0].status
    c.check("D.REQ9.3", status == "invalid_item" and not seen, f"status={status}, operation ran={bool(seen)}")
    return c.lines
