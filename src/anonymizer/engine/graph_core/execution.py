# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Admission, synchronous execution, qualification and inspection over the domain-free kernel.

Anonymizer owns identity, accounting and qualification here; the kernel owns topology and readiness.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import cached_property
from itertools import islice
from typing import Any

from anonymizer.engine.graph_core import kernel as k
from anonymizer.engine.graph_core.authoring import (
    CONTEXT,
    DATUM,
    MENTIONS,
    TEXT,
    Assessment,
    ContextSource,
    ContextView,
    Datum,
    Limits,
    Operation,
    ProtectionGraph,
    Workflow,
    utf8_size,
)

SEEDS = {"text": TEXT, "source": TEXT, "mentions": MENTIONS, "context": CONTEXT, "target": DATUM}


@dataclass(frozen=True)
class Count:
    """A count that is known exactly, known as an upper bound, or explicitly unknown (never confused with 0)."""

    value: int | None
    bound: str = "exact"  # exact | upper | unknown

    def __str__(self) -> str:
        return "unknown" if self.value is None else f"{self.bound} {self.value}"


@dataclass(frozen=True)
class PreparedProtection:
    """Immutable, process-local result of successful admission; the only value :func:`protect` accepts."""

    graph: ProtectionGraph = field(repr=False)
    workflow: Workflow = field(repr=False)
    limits: Limits
    plan: k.Plan = field(repr=False)
    names: tuple[str, ...]
    planned_requests: Count
    covering_verifiers: frozenset[int]

    def summary(self) -> InspectionSummary:
        return InspectionSummary(self.names, len(self.graph.targets), self.planned_requests)

    def __reduce__(self) -> Any:
        raise TypeError("PreparedProtection is process-local and not serializable")


def prepare(
    graph: ProtectionGraph,
    workflow: Workflow,
    *,
    limits: Limits | None = None,
    capabilities: tuple[str, ...] | None = None,
    request_budget: int | None = None,
) -> PreparedProtection:
    """Validate everything before effects and compile an immutable plan. No callback runs here.

    ``capabilities`` names the operation implementations admitted for this invocation (default: the workflow's);
    ``request_budget`` selects a hard external-request budget.
    """
    limits = limits or Limits()
    ops = workflow.flatten()
    if len(ops) > limits.max_operations:
        raise k.GraphRejection("over_limit", "max_operations")
    if len(graph.targets) > limits.max_targets:
        raise k.GraphRejection("over_limit", "max_targets")
    if len(graph.mentions) > limits.max_mentions:
        raise k.GraphRejection("over_limit", "max_mentions")
    if limits.max_text_bytes is not None and any(utf8_size(d.text) > limits.max_text_bytes for d in graph.targets):
        raise k.GraphRejection("over_byte_ceiling", "max_text_bytes")
    static = [_static_text(i) for cs in graph.context for i in cs.items if not isinstance(i, ContextSource)]
    sources = [i for cs in graph.context for i in cs.items if isinstance(i, ContextSource)]
    if len(static) + sum(s.max_items for s in sources) > limits.max_context_items:
        raise k.GraphRejection("over_limit", "max_context_items")
    if limits.max_context_bytes is not None and sum(map(utf8_size, static)) > limits.max_context_bytes:
        raise k.GraphRejection("over_byte_ceiling", "max_context_bytes")
    admitted = {name for name, _ in ops} if capabilities is None else set(capabilities)
    if not ops or any(name not in admitted for name, _ in ops):
        raise k.GraphRejection("capability_not_admitted")
    planned = _planned(ops, len(graph.targets))
    if request_budget is not None and (planned.value is None or planned.bound == "opaque"):
        raise k.GraphRejection("hard_budget_without_visibility")  # never downgraded to soft
    if request_budget is not None and planned.value is not None and planned.value > request_budget:
        raise k.GraphRejection("over_request_budget")
    nodes = tuple(k.Node(name, op.inputs, op.outputs, op.fn, op.scope) for name, op in ops)
    plan = k.compile_sequence(nodes, SEEDS)
    final_text = max((i for i, n in enumerate(nodes) if "text" in n.outputs), default=-1)
    covering = frozenset(
        i for i, (_, op) in enumerate(ops) if op.is_verifier and plan.sources[i].get("text", (None,))[0] == final_text
    )
    if planned.bound == "opaque":
        planned = Count(None, "unknown")
    return PreparedProtection(graph, workflow, limits, plan, tuple(n for n, _ in ops), planned, covering)


def _static_text(item: Datum | ContextView) -> str:
    return item.text if isinstance(item, Datum) else item.over.text[item.start : item.end]


def _planned(ops: tuple[tuple[str, Operation], ...], targets: int) -> Count:
    bounds = {op.requests.bound for _, op in ops}
    if "opaque" in bounds:
        return Count(None, "opaque")
    if "unknown" in bounds:
        return Count(None, "unknown")
    per_target = sum((op.requests.count or 0) for _, op in ops if op.scope != "invocation")
    total = per_target * targets + sum((op.requests.count or 0) for _, op in ops if op.scope == "invocation")
    return Count(total, "upper" if "upper" in bounds else "exact")


# ---------------------------------------------------------------- results


@dataclass(frozen=True)
class ContextItem:
    """One bound context artifact, attributable to its source."""

    source: str
    index: int
    text: str = field(repr=False)


@dataclass(frozen=True)
class Retrieval:
    """Binding accounting for one source, separate from execution attempts."""

    source: str
    status: str  # ok | failed | overflow | over_byte_ceiling | invalid_item | not_started
    declared_max_items: int
    observed_items: int | None
    observed_bytes: int | None


@dataclass(frozen=True)
class Attempt:
    """One accounting entry per actual dispatch."""

    target: Datum
    operation: str
    status: str
    reason: str = ""
    ref: str = ""


@dataclass(frozen=True)
class TargetOutcome:
    """Execution terminal, criteria satisfaction and output availability, kept separate."""

    terminal: str  # success | failed | cancelled | blocked
    qualification: str  # met | not_met | not_assessed
    available: bool
    protected: bool
    reason: str = ""


@dataclass(frozen=True)
class InspectionSummary:
    """Bounded, content-free summary: structure, counts and reason codes only."""

    operations: tuple[str, ...]
    targets: int
    planned_requests: Count
    terminals: tuple[tuple[str, int], ...] | None = None
    qualifications: tuple[tuple[str, int], ...] | None = None
    reasons: tuple[tuple[str, int], ...] = ()
    observed: tuple[tuple[str, Count], ...] = ()
    protected: int | None = None
    consistent: bool | None = None

    def render(self) -> str:
        ops = ", ".join(name[:64] for name in self.operations)
        lines = [
            f"workflow: {len(self.operations)} operations [{ops}]; targets: {self.targets}",
            f"planned external requests: {self.planned_requests}",
        ]
        if self.terminals is None:
            lines.append("terminals: none (not executed)")
        else:
            fmt = lambda pairs: ", ".join(f"{a}={b}" for a, b in pairs) or "-"  # noqa: E731
            lines += [
                f"terminals: {fmt(self.terminals)}; qualification: {fmt(self.qualifications or ())}",
                f"reasons: {fmt(self.reasons)}; protected: {self.protected}; consistent: {self.consistent}",
                f"observed: {fmt(self.observed)}",
            ]
        return "\n".join(lines)


@dataclass(frozen=True)
class ProtectionResult:
    """Reconciled result: the canonical record every projection derives from without rerunning work."""

    prepared: PreparedProtection = field(repr=False)
    state: k.State = field(repr=False)
    retrievals: tuple[Retrieval, ...]
    outcomes: tuple[TargetOutcome, ...]
    consistent: bool

    @property
    def targets(self) -> tuple[Datum, ...]:
        return self.prepared.graph.targets

    def outcome(self, target: Datum) -> TargetOutcome:
        return self.outcomes[self._index(target)]

    def value(self, target: Datum, port: str = "text") -> Any:
        """Explicit content access: the latest value of ``port`` for ``target``, or None if never produced."""
        values = self.state.row(self._index(target)).values
        found = [(i, v) for (i, p), v in values.items() if p == port and i >= 0]
        return max(found, key=lambda t: t[0])[1] if found else None

    def content(self) -> tuple[tuple[Datum, str | None], ...]:
        """Content projection: each target with its final candidate text (not a Protected claim)."""
        return tuple((d, self.value(d) if o.available else None) for d, o in zip(self.targets, self.outcomes))

    @property
    def protected(self) -> dict[Datum, str]:
        """Only qualified, releasable output."""
        return {d: self.value(d) for d, o in zip(self.targets, self.outcomes) if o.protected}

    @property
    def attempts(self) -> tuple[Attempt, ...]:
        names = self.prepared.names
        return tuple(
            Attempt(self.targets[e.unit], names[e.node], e.status, e.reason, e.ref) for e in self.state.entries
        )

    def ledger(self) -> tuple[tuple[Datum, str, str], ...]:
        """One terminal per target and operation, dispatched or not."""
        stopped = k.CANCELLED if self.state.cancelled else k.BLOCKED
        return tuple(
            (d, name, {k.PENDING: stopped}.get(status, status))
            for u, d in enumerate(self.targets)
            for name, status in zip(self.prepared.names, self.state.row(u).status)
        )

    def summary(self) -> InspectionSummary:
        p = self.prepared
        dispatched = {e.node for e in self.state.entries}
        silent = all(p.workflow.flatten()[i][1].requests.bound == "none" for i in dispatched)
        zero_or_unknown = Count(0 if silent else None, "exact" if silent else "unknown")
        binding = Count(sum(r.status != "not_started" for r in self.retrievals))
        total = binding if silent else Count(None, "unknown")
        return InspectionSummary(
            p.names,
            len(self.targets),
            p.planned_requests,
            tuple(sorted(Counter(o.terminal for o in self.outcomes).items())),
            tuple(sorted(Counter(o.qualification for o in self.outcomes).items())),
            tuple(sorted(Counter(o.reason for o in self.outcomes if o.reason).items())),
            (("binding", binding), ("correction", zero_or_unknown), ("retry", zero_or_unknown), ("total", total)),
            sum(o.protected for o in self.outcomes),
            self.consistent,
        )

    @cached_property
    def _positions(self) -> dict[Datum, int]:
        return {d: i for i, d in enumerate(self.targets)}

    def _index(self, target: Datum) -> int:
        if target not in self._positions:
            raise k.GraphRejection("unknown_reference", "target")
        return self._positions[target]


# ---------------------------------------------------------------- execution


def protect(prepared: PreparedProtection, *, cancel: k.CancelToken | None = None) -> ProtectionResult:
    """Execute synchronously on the calling thread in a copy of the calling context.

    Callbacks and providers run on this thread. A raising callback closes its target as failed; nothing is raised.
    ``cancel`` stops new dispatches and retrievals; a frame nested in another inherits its
    cancellation.
    """
    if not isinstance(prepared, PreparedProtection):
        raise k.GraphRejection("not_prepared")
    return k.run_in_frame(_protect, prepared, token=cancel)


def _protect(p: PreparedProtection) -> ProtectionResult:
    token = k.current_token()
    cancelled = (lambda: token.cancelled) if token is not None else (lambda: False)
    g = p.graph
    retrievals, bound = _bind(g, p.limits, cancelled)
    mentions: dict[Datum, list] = {}
    for m in sorted(g.mentions, key=lambda m: m.start):
        mentions.setdefault(m.datum, []).append(m)
    scopes: dict[Datum, list] = {}
    for cs in g.context:
        scopes.setdefault(cs.target, []).extend(cs.items)
    seeds, closed = [], []
    for u, d in enumerate(g.targets):
        items: list[ContextItem] = []
        for item in scopes.get(d, ()):
            if isinstance(item, ContextSource):
                if bound.get(item.name) is None and not cancelled():
                    closed.append(u)
                items.extend(bound.get(item.name) or ())
            else:
                items.append(
                    ContextItem("datum" if isinstance(item, Datum) else "view", len(items), _static_text(item))
                )
        m = tuple(mentions.get(d, ()))
        seeds.append({"text": d.text, "source": d.text, "mentions": m, "context": tuple(items), "target": d})
    state = k.initial_state(p.plan, seeds)
    for u in closed:
        state = k.reduce(p.plan, state, k.Closed(u, "binding_failed"))
    state = k.execute(p.plan, state, cancelled)
    return _reconcile(p, state, tuple(retrievals))


def _bind(g: ProtectionGraph, limits: Limits, cancelled: Any) -> tuple[list[Retrieval], dict[str, tuple | None]]:
    """Initial context binding: one retrieval per declared source; no truncation, no silent empty context."""
    retrievals: list[Retrieval] = []
    bound: dict[str, tuple | None] = {}
    total_bytes = 0
    for src in dict.fromkeys(i for cs in g.context for i in cs.items if isinstance(i, ContextSource)):
        if cancelled():
            retrievals.append(Retrieval(src.name, "not_started", src.max_items, None, None))
            bound[src.name] = None
            continue
        status, items = "ok", []
        try:
            items = list(islice(iter(src.provider()), src.max_items + 1))
        except Exception:
            status = "failed"
        if status == "ok" and not all(isinstance(t, str) for t in items):
            status = "invalid_item"
        size = sum(map(utf8_size, items)) if status != "failed" and status != "invalid_item" else None
        if status == "ok" and len(items) > src.max_items:
            status = "overflow"  # one over closes as overflow, never truncated
        ceiling = src.max_bytes
        if status == "ok" and ceiling is not None and size is not None and size > ceiling:
            status = "over_byte_ceiling"
        total_bytes += size or 0
        if status == "ok" and limits.max_context_bytes is not None and total_bytes > limits.max_context_bytes:
            status = "over_byte_ceiling"
        observed = len(items) if status != "failed" else None
        retrievals.append(Retrieval(src.name, status, src.max_items, observed, size))
        bound[src.name] = tuple(ContextItem(src.name, i, t) for i, t in enumerate(items)) if status == "ok" else None
    return retrievals, bound


def _reconcile(p: PreparedProtection, state: k.State, retrievals: tuple[Retrieval, ...]) -> ProtectionResult:
    """Derive outcomes from the final state: terminal, qualification, then fixed-point release."""
    g = p.graph
    consistent = True
    verdicts: list[tuple[str, str, str]] = []
    failures: dict[int, str] = {}
    for e in state.entries:
        if e.status == k.FAILED:
            failures.setdefault(e.unit, e.reason)
    for u, d in enumerate(g.targets):
        row = state.row(u)
        terminal = {k.SUCCEEDED: "success"}.get(t := k.unit_terminal(row, state.cancelled), t)
        reason = "" if terminal == "success" else row.closed or failures.get(u, terminal)
        quals = []
        for i in p.covering_verifiers:
            a: Assessment | None = row.values.get((i, "assessment"))
            if a is not None and a.target is not None and a.target != d:
                consistent = False  # foreign or swapped evidence destroys attribution
            quals.append(a.satisfied is True if a is not None else False)
        if terminal != "success":
            qual = "not_met"
        elif not p.covering_verifiers:
            qual = "not_assessed"
        else:
            qual = "met" if all(quals) else "not_met"
            reason = reason or ("" if qual == "met" else "unsatisfied")
        verdicts.append((terminal, qual, reason))
    if not consistent:  # the invocation is the whole call: nothing qualifies
        verdicts = [(t, "not_met", "inconsistent_invocation") for t, _, _ in verdicts]
    released = {d for d, (_, q, _) in zip(g.targets, verdicts) if q == "met"}
    if p.workflow.release != "protected":
        released = set()
    changed = True
    while changed:
        before = len(released)
        released -= {dep.dependent for dep in g.dependencies if dep.on not in released}
        for grp in g.groups:
            if not set(grp.members) <= released:
                released -= set(grp.members)
        changed = len(released) != before
    outcomes = []
    for u, (d, (terminal, qual, reason)) in enumerate(zip(g.targets, verdicts)):
        available = any(n == "text" for (i, n) in state.row(u).values if i >= 0)
        if qual == "met" and d not in released and not reason:
            reason = "withheld" if p.workflow.release == "protected" else "annotation"
        outcomes.append(TargetOutcome(terminal, qual, available, d in released, reason))
    return ProtectionResult(p, state, retrievals, tuple(outcomes), consistent)
