# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Admission, synchronous execution, qualification and inspection over the domain-free kernel.

Anonymizer owns identity, accounting and qualification here; the kernel owns topology and readiness.
"""

from __future__ import annotations

import hashlib
import secrets
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from functools import cached_property
from itertools import islice
from typing import Any, Literal, cast

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
    mention_problem,
    utf8_size,
)

Terminal = Literal["success", "failed", "cancelled", "blocked"]
Qualification = Literal["met", "not_met", "not_assessed"]
AttemptStatus = Literal["running", "succeeded", "failed", "blocked", "cancelled"]

SEEDS = {"text": TEXT, "source": TEXT, "mentions": MENTIONS, "context": CONTEXT, "target": DATUM}
_SEAL_KEY = secrets.token_bytes(32)


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
    capabilities: frozenset[str] = frozenset()
    final_text: int | None = None
    seal: str = field(default="", repr=False)

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

    def requirements(w: Workflow) -> tuple[str, ...]:
        return (*w.requires, *(c for s in w.steps if isinstance(s, Workflow) for c in requirements(s)))

    # A nested workflow's requirements are requirements of the invocation.
    workflow = replace(workflow, requires=tuple(dict.fromkeys(requirements(workflow))))
    ops = workflow.flatten()
    admitted = frozenset(name for name, _ in ops) if capabilities is None else frozenset(capabilities)
    _admit(graph, ops, limits, admitted)
    planned = _planned(ops, len(graph.targets))
    if request_budget is not None and (planned.value is None or planned.bound == "opaque"):
        raise k.GraphRejection("hard_budget_without_visibility")  # never downgraded to soft
    if request_budget is not None and planned.value is not None and planned.value > request_budget:
        raise k.GraphRejection("over_request_budget")
    nodes = tuple(k.Node(name, op.inputs, op.outputs, op.fn, op.scope) for name, op in ops)
    plan = k.compile_sequence(nodes, {**SEEDS, "context": _context_kind(graph)})
    final_text = max((i for i, n in enumerate(nodes) if "text" in n.outputs), default=None)
    covering = frozenset(
        i
        for i, (_, op) in enumerate(ops)
        if op.is_verifier and plan.sources[i].get("text") == k.Source(final_text, "text")
    )
    answered = {c for i in covering for c in ops[i][1].checks}
    if unanswered := [c for c in workflow.requires if c not in answered]:
        # Evidence on earlier text is a diagnostic; a requirement only it answers could never be met.
        raise k.GraphRejection("required_check_not_final", ", ".join(unanswered))
    if planned.bound == "opaque":
        planned = Count(None, "unknown")
    names = tuple(n for n, _ in ops)
    prepared = PreparedProtection(graph, workflow, limits, plan, names, planned, covering, admitted, final_text)
    return replace(prepared, seal=_seal(prepared))


def _seal(p: PreparedProtection) -> str:
    """A process-local digest of the identities of everything admission produced."""
    ids = repr(tuple(id(getattr(p, f.name)) for f in fields(p) if f.name != "seal"))
    return hashlib.blake2b(ids.encode(), key=_SEAL_KEY).hexdigest()


def _admit(
    graph: ProtectionGraph, ops: tuple[tuple[str, Operation], ...], limits: Limits, admitted: frozenset[str]
) -> None:
    """The limit and capability checks, run at preparation and again before execution."""
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
    for src in sources:  # an unknown kind, or one without a measure under a byte ceiling, is unsupported
        ceiling = src.max_bytes is not None or limits.max_context_bytes is not None
        if k.kind(src.kind).measure is None and ceiling:
            raise k.GraphRejection("unsupported_context_kind", src.name)
    _context_kind(graph)
    if not ops or any(name not in admitted for name, _ in ops):
        raise k.GraphRejection("capability_not_admitted")


def _context_kind(graph: ProtectionGraph) -> str:
    """The ``context`` seed kind: ``context`` for text items, else the one structured kind every item has.

    A consumer reading ``context`` must declare this kind on its port, so a mismatch is rejected at compilation.
    """
    kinds = {i.kind if isinstance(i, ContextSource) else TEXT for cs in graph.context for i in cs.items} - {TEXT}
    if not kinds:
        return CONTEXT
    if len(kinds) > 1 or any(
        not isinstance(i, ContextSource) or i.kind == TEXT for cs in graph.context for i in cs.items
    ):
        raise k.GraphRejection("unsupported_context_kind", "mixed")
    return kinds.pop()


def _static_text(item: Datum | ContextView) -> str:
    return item.text if isinstance(item, Datum) else item.over.text[item.start : item.end]


def _planned(ops: tuple[tuple[str, Operation], ...], targets: int) -> Count:
    bounds = {op.requests.bound for _, op in ops}
    if "opaque" in bounds:
        return Count(None, "opaque")
    if "unknown" in bounds:
        return Count(None, "unknown")
    per_target = sum((op.requests.count or 0) for _, op in ops if op.scope != "phase_scoped")
    total = per_target * targets + sum((op.requests.count or 0) for _, op in ops if op.scope == "phase_scoped")
    return Count(total, "upper" if "upper" in bounds else "exact")


# ---------------------------------------------------------------- results


@dataclass(frozen=True)
class ContextItem:
    """One bound context artifact, attributable to its source; ``text`` holds a value of kind ``kind``."""

    source: str
    index: int
    text: Any = field(repr=False)
    kind: str = TEXT


@dataclass(frozen=True)
class Retrieval:
    """Binding accounting for one source, separate from execution attempts."""

    source: str
    status: Literal["ok", "failed", "overflow", "over_byte_ceiling", "invalid_item", "not_started"]
    declared_max_items: int
    observed_items: int | None
    observed_bytes: int | None


@dataclass(frozen=True)
class Attempt:
    """One accounting entry per actual dispatch."""

    target: Datum
    operation: str
    status: AttemptStatus
    reason: str = ""
    ref: str = ""


@dataclass(frozen=True)
class Reason:
    """Why a target did not qualify or release: a code, the check and verifier it concerns, and the verifier's
    stated cause when it could not assess. Fields hold codes and declared names, never exception text."""

    code: str
    check: str = ""
    verifier: str = ""
    cause: str = ""


@dataclass(frozen=True)
class TargetOutcome:
    """Execution terminal, criteria satisfaction and output availability, kept separate.

    ``reason`` is the code of the last of ``reasons``. ``error`` is the exception that failed the target, for
    re-raising; it is never rendered. ``withheld_by`` names the targets whose withholding withheld this one.
    """

    terminal: Terminal
    qualification: Qualification
    available: bool
    protected: bool
    reasons: tuple[Reason, ...] = ()
    error: BaseException | None = field(default=None, repr=False, compare=False)
    withheld_by: tuple[Datum, ...] = ()

    @property
    def reason(self) -> str:
        return self.reasons[-1].code if self.reasons else ""


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
    """Reconciled result of one invocation: per-target outcomes and the run state the accessors below read."""

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

    def value(self, target: Datum, port: str = "text", producer: str | None = None) -> Any:
        """Explicit port read: the latest value of ``port`` for ``target`` from any operation, or from the
        operation named ``producer`` only; None if never produced."""
        outputs = self.state.rows[self._index(target)].outputs
        if producer is not None:
            return outputs.get(self.prepared.names.index(producer), {}).get(port)
        found = [out[port] for _, out in sorted(outputs.items()) if port in out]
        return found[-1] if found else None

    def content(self) -> tuple[tuple[Datum, str | None], ...]:
        """Content projection: each target with the last text operation's output (not a Protected claim)."""
        return tuple((d, self._final(d) if o.available else None) for d, o in zip(self.targets, self.outcomes))

    def compose(self, parts: Sequence[Datum]) -> str:
        """Caller-authored composition: the candidate texts of ``parts`` in the caller's order, any subset.

        A listed part that did not succeed with an available candidate is rejected as ``unavailable_input``, never
        dropped.
        """
        return "".join(self._available(d) for d in parts)

    def reconstruct(self, parts: Sequence[Datum], source: Mapping[Datum, int]) -> str:
        """Source reconstruction: the candidate texts of ``parts`` in the order of their ``source`` positions.

        Every part must be in ``source`` (``not_in_source``), at most one part may take a position
        (``conflicting_position``) and every position in ``source`` must be taken (``missing_source_part``).
        """
        taken: dict[int, Datum] = {}
        for d in parts:
            if d not in source:
                raise k.GraphRejection("not_in_source")
            if taken.setdefault(source[d], d) is not d:
                raise k.GraphRejection("conflicting_position", str(source[d]))
        if missing := set(source.values()) - set(taken):
            raise k.GraphRejection("missing_source_part", str(min(missing)))
        return self.compose([taken[i] for i in sorted(taken)])

    def _available(self, target: Datum) -> str:
        o = self.outcome(target)
        if o.terminal != "success" or not o.available:
            raise k.GraphRejection("unavailable_input", o.reason or o.terminal)
        return self._final(target)

    @property
    def protected(self) -> dict[Datum, str]:
        """Only qualified, releasable output."""
        # With no text-producing operation, the verified text is the source text.
        verified = (lambda d: d.text) if self.prepared.final_text is None else self._final
        return {d: verified(d) for d, o in zip(self.targets, self.outcomes) if o.protected}

    @property
    def attempts(self) -> tuple[Attempt, ...]:
        names = self.prepared.names
        return tuple(
            Attempt(self.targets[e.unit], names[e.node], cast(AttemptStatus, e.status.value), e.reason, e.ref)
            for e in self.state.entries
        )

    def ledger(self) -> tuple[tuple[Datum, str, str], ...]:
        """One terminal per target and operation, dispatched or not.

        A node that never ran is ``blocked`` when its target had already failed or closed, else ``cancelled`` if
        the invocation was cancelled; a later cancellation does not relabel an earlier failure's blocked nodes.
        """
        ledger = []
        for d, row in zip(self.targets, self.state.rows):
            failed = bool(row.closed) or k.Status.FAILED in row.status
            stopped = k.Status.CANCELLED if self.state.cancelled and not failed else k.Status.BLOCKED
            ledger += [
                (d, name, (stopped if s == k.Status.PENDING else s).value)
                for name, s in zip(self.prepared.names, row.status)
            ]
        return tuple(ledger)

    def summary(self) -> InspectionSummary:
        p = self.prepared
        dispatched = {e.node for e in self.state.entries}
        silent = all(p.workflow.flatten()[i][1].requests.bound == "none" for i in dispatched)
        zero_or_unknown = Count(0 if silent else None, "exact" if silent else "unknown")
        binding = Count(sum(r.status != "not_started" for r in self.retrievals))
        total = binding if silent else Count(None, "unknown")
        requests = self._observed_requests()
        return InspectionSummary(
            p.names,
            len(self.targets),
            p.planned_requests,
            tuple(sorted(Counter(o.terminal for o in self.outcomes).items())),
            tuple(sorted(Counter(o.qualification for o in self.outcomes).items())),
            tuple(sorted(Counter(o.reason for o in self.outcomes if o.reason).items())),
            (
                ("binding", binding),
                ("correction", zero_or_unknown),
                ("retry", zero_or_unknown),
                ("total", total),
                ("requests", requests),
            ),
            sum(o.protected for o in self.outcomes),
            self.consistent,
        )

    def _observed_requests(self) -> Count:
        """Exact total when every dispatched step declared ``none`` or reported (an invocation step once), else unknown."""
        ops = self.prepared.workflow.flatten()
        total, covered = 0, True
        for node in {e.node for e in self.state.entries}:
            entries = [e for e in self.state.entries if e.node == node]
            reports = [e.observed_requests for e in entries if e.observed_requests is not None]
            total += sum(reports)
            _, op = ops[node]
            needed = 1 if op.scope == "phase_scoped" else len(entries)
            covered = covered and (op.requests.bound == "none" or len(reports) >= needed)
        return Count(total) if covered else Count(None, "unknown")

    def _final(self, target: Datum) -> Any:
        """The final candidate: the last text operation's own output; earlier text is never substituted."""
        final = self.prepared.final_text
        return None if final is None else self.state.rows[self._index(target)].outputs.get(final, {}).get("text")

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
    if not isinstance(prepared, PreparedProtection) or prepared.seal != _seal(prepared):
        raise k.GraphRejection("not_prepared")
    _admit(prepared.graph, prepared.workflow.flatten(), prepared.limits, prepared.capabilities)
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
        k.apply(state, k.Closed(u, "binding_failed"))
    k.execute(p.plan, state, cancelled, _mention_review(p))
    return _reconcile(p, state, tuple(retrievals))


def _mention_review(p: PreparedProtection) -> k.Review:
    """Check produced mentions at acceptance: each publication's are valid for their target, and the publication
    as a whole fits what remains of ``max_mentions`` after the declared and earlier accepted mentions."""
    room = p.limits.max_mentions - len(p.graph.mentions)
    ports = [[port for port, name in node.outputs.items() if name == MENTIONS] for node in p.plan.nodes]

    def review(index: int, published: list[tuple[int, Mapping[str, Any]]]) -> dict[int, str]:
        nonlocal room
        failed: dict[int, str] = {}
        count = 0
        for unit, outputs in published:
            for port in ports[index]:
                if problem := mention_problem(outputs[port], p.graph.targets[unit]):
                    failed[unit] = problem
                    break
                count += len(outputs[port])
        if count > room:  # every target of an over-limit publication fails
            return {unit: failed.get(unit, "over_limit") for unit, _ in published}
        room -= count
        return failed

    return review


def _bind(g: ProtectionGraph, limits: Limits, cancelled: Any) -> tuple[list[Retrieval], dict[str, tuple | None]]:
    """Initial context binding: one retrieval per declared source; no truncation, no silent empty context."""
    retrievals: list[Retrieval] = []
    bound: dict[str, tuple | None] = {}
    static = (_static_text(i) for cs in g.context for i in cs.items if not isinstance(i, ContextSource))
    # retrieved bytes count on top of the static context, measured only under a ceiling as at admission
    total_bytes = sum(map(utf8_size, static)) if limits.max_context_bytes is not None else 0
    for src in dict.fromkeys(i for cs in g.context for i in cs.items if isinstance(i, ContextSource)):
        if cancelled():
            retrievals.append(Retrieval(src.name, "not_started", src.max_items, None, None))
            bound[src.name] = None
            continue
        status, items, found = "ok", [], k.kind(src.kind)
        try:
            items = list(islice(iter(src.provider()), src.max_items + 1))
        except Exception:
            status = "failed"
        sizes = None
        try:  # a check or measure that rejects or raises on an item makes it invalid
            if status == "ok" and not all(map(found.check, items)):
                status = "invalid_item"
            sizes = [found.measure(t) for t in items] if status == "ok" and found.measure else None
        except Exception:
            status = "invalid_item"
        if sizes is not None and None in sizes:
            status, sizes = "invalid_item", None
        size = sum(n or 0 for n in sizes) if sizes is not None else None
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
        bound[src.name] = (
            tuple(ContextItem(src.name, i, t, src.kind) for i, t in enumerate(items)) if status == "ok" else None
        )
    return retrievals, bound


def _reconcile(p: PreparedProtection, state: k.State, retrievals: tuple[Retrieval, ...]) -> ProtectionResult:
    """Derive outcomes from the final state: terminal, qualification, then fixed-point release."""
    g = p.graph
    requires = p.workflow.requires
    checks_of = [[c for c in op.checks if c in requires] for _, op in p.workflow.flatten()]
    consistent = True
    verdicts: list[tuple[Terminal, Qualification, list[Reason]]] = []
    failures: dict[int, k.Entry] = {}
    for e in state.entries:
        if e.status == k.Status.FAILED:
            failures.setdefault(e.unit, e)
    for u, d in enumerate(g.targets):
        row = state.rows[u]
        status = k.unit_terminal(row, state.cancelled)
        terminal: Terminal = "success" if status == k.Status.SUCCEEDED else cast(Terminal, status.value)
        reasons: list[Reason] = []
        if terminal != "success":
            failed = failures.get(u)
            reasons.append(Reason(row.closed or (failed.reason if failed else terminal)))
        # Unattributed, foreign or swapped evidence from any verifier destroys attribution.
        assessments = [v for out in row.outputs.values() for v in out.values() if isinstance(v, Assessment)]
        consistent = consistent and all(a.target == d for a in assessments)
        qual: Qualification
        if terminal != "success":
            qual = "not_met"
        elif not requires:
            qual = "not_assessed"
        else:
            covered: set[str] = set()
            for i in sorted(p.covering_verifiers):
                a: Assessment | None = row.outputs.get(i, {}).get("assessment")
                if a is None:
                    continue
                name, checks = p.names[i], checks_of[i]
                covered.update(checks)
                if a.satisfied is None:
                    reasons += [Reason("could_not_assess", c, name, a.reason) for c in checks]
                elif a.satisfied is not True:
                    reasons += [Reason("failed", c, name) for c in checks]
            reasons += [Reason("missing", c) for c in requires if c not in covered]
            qual = "not_met" if reasons else "met"
        verdicts.append((terminal, qual, reasons))
    if not consistent:  # the invocation is the whole call: nothing qualifies
        verdicts = [(t, "not_met", [*rs, Reason("inconsistent_invocation")]) for t, _, rs in verdicts]
    released = {d for d, (_, q, _) in zip(g.targets, verdicts) if q == "met"}
    if p.workflow.release != "protected":
        released = set()
    # Worklist withholding: each target and each group is withheld at most once.
    dependents: dict[Datum, list[Datum]] = {}
    for dep in g.dependencies:
        dependents.setdefault(dep.on, []).append(dep.dependent)
    group_of = {m: i for i, grp in enumerate(g.groups) for m in grp.members}
    withheld = [d for d in g.targets if d not in released]
    releasable = set(released)
    withheld_by: dict[Datum, list[Datum]] = {}
    spent: set[int] = set()
    for d in withheld:  # grows while iterating
        i = group_of.get(d, -1)
        members = () if i < 0 or i in spent else g.groups[i].members
        spent.add(i)
        for nxt in dependents.get(d, ()):
            if nxt in releasable:
                withheld_by.setdefault(nxt, []).append(d)
        for nxt in (*dependents.get(d, ()), *members):
            if nxt in released:
                released.discard(nxt)
                withheld.append(nxt)
    # A group member withheld only by its group names every member withheld for another cause.
    for i in spent - {-1}:
        members = g.groups[i].members
        causes = [m for m in members if m not in releasable or m in withheld_by]
        cause_set = set(causes)
        for m in members:
            if m not in cause_set:
                withheld_by.setdefault(m, []).extend(causes)
    outcomes = []
    for u, (d, (terminal, qual, reasons)) in enumerate(zip(g.targets, verdicts)):
        available = "text" in state.rows[u].outputs.get(p.final_text, {})
        if qual == "met" and d not in released and not reasons:
            reasons.append(Reason("withheld" if p.workflow.release == "protected" else "annotation"))
        failed = failures.get(u)
        error = failed.error if failed is not None and terminal != "success" else None
        outcomes.append(
            TargetOutcome(
                terminal, qual, available, d in released, tuple(reasons), error, tuple(withheld_by.get(d, ()))
            )
        )
    return ProtectionResult(p, state, retrievals, tuple(outcomes), consistent)
