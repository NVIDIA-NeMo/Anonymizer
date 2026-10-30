# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct authoring values: datums, relations, operations, workflows and limits.

Every value here is immutable and constructing it has no effect: no retrieval, no callback.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from anonymizer.engine.graph_core.kernel import GraphRejection, register_kind

TEXT = register_kind("text", lambda v: isinstance(v, str), lambda v: len(v.encode("utf-8")))
MENTIONS = register_kind("mentions", lambda v: isinstance(v, tuple))
CONTEXT = register_kind("context", lambda v: isinstance(v, tuple))
DATUM = register_kind("datum", lambda v: isinstance(v, Datum))
ASSESSMENT = register_kind("assessment", lambda v: isinstance(v, Assessment))


def utf8_size(text: str) -> int:
    return len(text.encode("utf-8"))


# ---------------------------------------------------------------- data relationship graph


class GraphScope:
    """Identity namespace of one data graph; datums are declared from it."""

    def __init__(self) -> None:
        self._token = secrets.token_hex(8)

    def datum(self, text: str) -> Datum:
        if not isinstance(text, str):
            raise GraphRejection("datum_not_text", type(text).__name__)
        return Datum(text=text, _scope=self._token, _id=secrets.token_hex(8))


@dataclass(frozen=True, eq=False)
class Datum:
    """Immutable text value with opaque, graph-scoped identity. Equality and hashing use identity only, never text."""

    text: str = field(repr=False)
    _scope: str = field(repr=False)
    _id: str = field(repr=False)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Datum) and (self._scope, self._id) == (other._scope, other._id)

    def __hash__(self) -> int:
        return hash((self._scope, self._id))

    def __repr__(self) -> str:
        return "Datum(<opaque>)"


@dataclass(frozen=True)
class Mention:
    """An output-producing region: an exact half-open character interval of one target datum."""

    datum: Datum
    start: int
    end: int
    label: str
    text: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class ContextView:
    """A read-only view ``[start, end)`` over a datum, delivered as context to ``target``."""

    target: Datum
    over: Datum
    start: int
    end: int


@dataclass(frozen=True)
class ContextSource:
    """A caller-owned provider retrieved once per invocation at binding.

    ``provider()`` returns an iterable of texts. ``max_items`` is the declared cardinality bound; ``max_bytes`` is an
    optional caller-supplied UTF-8 ceiling over the returned items (absent means none).
    """

    name: str
    provider: Callable[[], Iterable[str]] = field(repr=False, compare=False)
    max_items: int
    max_bytes: int | None = None


@dataclass(frozen=True)
class ContextScope:
    """The ordered context a target may read: context datums, views and sources."""

    target: Datum
    items: tuple[Datum | ContextView | ContextSource, ...]


@dataclass(frozen=True)
class Dependency:
    """Release dependency: ``dependent`` is withheld when ``on`` is withheld."""

    dependent: Datum
    on: Datum


@dataclass(frozen=True)
class OutputGroup:
    """All-or-none release group; groups form a flat partition."""

    members: tuple[Datum, ...]


@dataclass(frozen=True)
class ProtectionGraph:
    """Immutable bound semantic input: target datums plus separately typed relations."""

    scope: GraphScope = field(repr=False)
    targets: tuple[Datum, ...]
    mentions: tuple[Mention, ...] = ()
    context: tuple[ContextScope, ...] = ()
    dependencies: tuple[Dependency, ...] = ()
    groups: tuple[OutputGroup, ...] = ()

    def __post_init__(self) -> None:
        targets = set(self.targets)
        if len(targets) != len(self.targets):
            raise GraphRejection("duplicate_target")
        declared = set(targets)
        for cs in self.context:
            self._own(cs.target, targets, "context_scope_target")
            for item in cs.items:
                if isinstance(item, Datum):
                    declared.add(self._own(item, None, "context_datum"))
                elif isinstance(item, ContextView):
                    self._own(item.target, {cs.target}, "view_target")
                    declared.add(self._own(item.over, None, "view_datum"))
                    if not 0 <= item.start < item.end <= len(item.over.text):
                        raise GraphRejection("view_out_of_range")
        for d in self.targets:
            self._own(d, None, "target")
        sources: dict[str, ContextSource] = {}
        for cs in self.context:
            for item in cs.items:
                if isinstance(item, ContextSource):
                    first = sources.setdefault(item.name, item)
                    if first is not item and (first != item or first.provider is not item.provider):
                        # One name is one source: bindings are keyed by name, so a second source under it would
                        # hand one target another target's context.
                        raise GraphRejection("duplicate_context_source", item.name)
        self._check_mentions(targets, declared)
        for dep in self.dependencies:
            self._own(dep.dependent, targets, "dependency")
            self._own(dep.on, targets, "dependency")
        self._check_acyclic()
        seen: set[Datum] = set()
        for group in self.groups:
            for m in group.members:
                if self._own(m, targets, "group_member") in seen:
                    raise GraphRejection("group_overlap")
                seen.add(m)

    def _own(self, d: Datum, allowed: set[Datum] | None, what: str) -> Datum:
        if not isinstance(d, Datum) or d._scope != self.scope._token:
            raise GraphRejection("foreign_datum", what)
        if allowed is not None and d not in allowed:
            raise GraphRejection("unknown_reference", what)
        return d

    def _check_mentions(self, targets: set[Datum], declared: set[Datum]) -> None:
        by_target: dict[Datum, list[Mention]] = {}
        for m in self.mentions:
            self._own(m.datum, None, "mention")
            if m.datum not in targets:
                # Context must not supply a mention endpoint or replacement span.
                raise GraphRejection("mention_not_on_target" if m.datum in declared else "unknown_reference")
            if not (0 <= m.start < m.end <= len(m.datum.text)):
                raise GraphRejection("mention_out_of_range")
            if m.text is not None and m.datum.text[m.start : m.end] != m.text:
                raise GraphRejection("mention_slice_mismatch")
            by_target.setdefault(m.datum, []).append(m)
        for ms in by_target.values():
            ms.sort(key=lambda m: m.start)
            for a, b in zip(ms, ms[1:]):
                if b.start < a.end:
                    raise GraphRejection("overlapping_output_ownership")

    def _check_acyclic(self) -> None:
        edges: dict[Datum, set[Datum]] = {}
        for dep in self.dependencies:
            edges.setdefault(dep.dependent, set()).add(dep.on)
        done: set[Datum] = set()

        def visit(d: Datum, path: set[Datum]) -> None:
            if d in path:
                raise GraphRejection("dependency_cycle")
            if d not in done:
                for nxt in edges.get(d, ()):
                    visit(nxt, path | {d})
                done.add(d)

        for d in list(edges):
            visit(d, set())


# ---------------------------------------------------------------- operations and workflows

RequestBound = Literal["none", "exact", "upper", "unknown", "opaque"]


@dataclass(frozen=True)
class Requests:
    """Declared external requests per activation: exact, known upper bound, unknown, or opaque.

    ``opaque`` declares requests the implementation can neither make visible nor control before dispatch.
    """

    bound: RequestBound = "none"
    count: int | None = 0

    @classmethod
    def none(cls) -> Requests:
        return cls("none", 0)

    @classmethod
    def exact(cls, n: int) -> Requests:
        return cls("exact", n) if n else cls.none()

    @classmethod
    def at_most(cls, n: int) -> Requests:
        return cls("upper", n) if n else cls.none()

    @classmethod
    def unknown(cls) -> Requests:
        return cls("unknown", None)

    @classmethod
    def opaque(cls, n: int | None = None) -> Requests:
        return cls("opaque", n)


@dataclass(frozen=True)
class Assessment:
    """Evidence returned by a verifier: ``satisfied`` True/False, or None when it could not assess.

    ``target`` attributes the finding; evidence attributed to another target destroys attribution and makes the
    whole invocation inconsistent.
    """

    satisfied: bool | None
    target: Datum | None = None
    reason: str = ""


@dataclass(frozen=True)
class Operation:
    """A caller-supplied local operation with typed ports and declared requests.

    Input ports bind by name to the latest earlier output, else to the per-target seeds ``text`` (current candidate),
    ``source`` (original text), ``mentions``, ``context`` and ``target``. An operation whose output kind is
    ``assessment`` is a verifier.
    """

    name: str
    fn: Callable[..., Any] = field(repr=False, compare=False)
    inputs: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({"text": TEXT}))
    outputs: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({"text": TEXT}))
    requests: Requests = Requests()
    scope: Literal["target", "invocation"] = "target"

    @classmethod
    def verifier(
        cls, name: str, fn: Callable[..., Any], inputs: Mapping[str, str] | None = None, requests: Requests = Requests()
    ) -> Operation:
        return cls(name, fn, MappingProxyType(dict(inputs or {"text": TEXT})), {"assessment": ASSESSMENT}, requests)

    @classmethod
    def per_invocation(
        cls,
        name: str,
        fn: Callable[..., Any],
        inputs: Mapping[str, str] | None = None,
        outputs: Mapping[str, str] | None = None,
        requests: Requests = Requests(),
    ) -> Operation:
        """An operation called once over every ready target; each input is a tuple of per-target values.

        ``fn`` returns one entry per target in that order: an output value, ``None`` (no output) or an
        ``Exception`` instance (that target failed). Its declared requests count once per invocation.
        """
        return cls(
            name,
            fn,
            MappingProxyType(dict(inputs or {"text": TEXT})),
            MappingProxyType(dict(outputs or {"text": TEXT})),
            requests,
            "invocation",
        )

    @property
    def is_verifier(self) -> bool:
        return ASSESSMENT in self.outputs.values()


@dataclass(frozen=True)
class Workflow:
    """Sequence composition; a nested ``Workflow`` is a declared, inspectable subgraph.

    ``release`` selects the result contract: ``protected`` output, or ``annotation`` output that retains source
    text and is never Protected.
    """

    name: str
    steps: tuple[Operation | Workflow, ...]
    release: Literal["protected", "annotation"] = "protected"

    def flatten(self, prefix: str = "") -> tuple[tuple[str, Operation], ...]:
        out: list[tuple[str, Operation]] = []
        for step in self.steps:
            if isinstance(step, Workflow):
                out.extend(step.flatten(f"{prefix}{step.name}/"))
            else:
                out.append((f"{prefix}{step.name}", step))
        return tuple(out)


@dataclass(frozen=True)
class Limits:
    """One limits record. Count limits have defaults (provisional values); byte ceilings default to none and are
    enforced only when supplied."""

    max_operations: int = 32
    max_targets: int = 1_000
    max_mentions: int = 10_000
    max_context_items: int = 256
    max_text_bytes: int | None = None
    max_context_bytes: int | None = None
