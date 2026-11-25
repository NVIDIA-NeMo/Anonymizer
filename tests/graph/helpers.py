# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the graph kernel tests: a call recorder, one-text graphs, a one-workflow run and rejection codes."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import Any

from anonymizer.engine.graph_core.authoring import Assessment, GraphScope, Operation, ProtectionGraph, Workflow
from anonymizer.engine.graph_core.execution import prepare, protect
from anonymizer.engine.graph_core.kernel import GraphRejection

NAME = "Alice"


class Recorder:
    """Records callback invocations: (operation, input text, thread id) in call order."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any, int]] = []

    def count(self, op: str | None = None) -> int:
        return sum(1 for c in self.calls if op is None or c[0] == op)

    def transform(self, name: str = "transform", fn: Callable[[str], str] | None = None) -> Operation:
        def call(text: str) -> str:
            self.calls.append((name, text, threading.get_ident()))
            return (fn or (lambda t: t.replace(NAME, "[NAME]")))(text)

        return Operation(name, call)

    def invocation_transform(
        self, name: str = "transform", fn: Callable[[tuple[str, ...]], Sequence[Any]] | None = None
    ) -> Operation:
        """A phase-scoped transform; each call records the whole text tuple it received."""

        def call(text: tuple[str, ...]) -> Sequence[Any]:
            self.calls.append((name, text, threading.get_ident()))
            return (fn or (lambda ts: [t.replace(NAME, "[NAME]") for t in ts]))(text)

        return Operation.per_invocation(name, call)

    def verifier(self, name: str = "verifier", needle: str = NAME) -> Operation:
        def call(text: str, target: Any) -> Assessment:
            self.calls.append((name, text, threading.get_ident()))
            return Assessment(needle not in text, target)

        return Operation.verifier(name, call, {"text": "text", "target": "datum"}, checks=(name,), subject=("text",))


def graph(*texts: str) -> tuple[ProtectionGraph, tuple]:
    scope = GraphScope()
    targets = tuple(scope.datum(t) for t in texts)
    return ProtectionGraph(scope, targets), targets


def run(g: ProtectionGraph, *ops: Operation, **prepare_kwargs: Any):
    """Run ``ops`` as one workflow that requires every check its verifiers declare."""
    requires = tuple(dict.fromkeys(check for op in ops for check in op.checks))
    return protect(prepare(g, Workflow("w", ops, requires=requires), **prepare_kwargs))


def rejected(fn: Callable[[], Any]) -> str | None:
    """The rejection code if ``fn`` raises the public rejection type, else None."""
    try:
        fn()
    except GraphRejection as exc:
        return exc.code
    return None
