# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Admission, synchronous execution, qualification and inspection over the domain-free kernel.

Anonymizer owns identity, accounting and qualification here; the kernel owns topology and readiness.
"""

from __future__ import annotations

import contextlib
import hashlib
import secrets
import sys
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
    IDENTITIES,
    MENTIONS,
    RECORD,
    TEXT,
    Assessment,
    ContextSource,
    ContextView,
    Datum,
    Limits,
    Operation,
    ProtectionGraph,
    Workflow,
    _reject_bare_string,
    mention_problem,
    utf8_size,
)
from anonymizer.engine.graph_core.session import DECISIONS, SESSION_VIEW, Decision, Session, SessionChanges, SessionMode

Terminal = Literal["success", "failed", "cancelled", "blocked"]
Qualification = Literal["met", "not_met", "not_assessed"]
AttemptStatus = Literal["running", "succeeded", "failed", "blocked", "cancelled"]
AdvisoryStatus = Literal["passed", "failed", "could_not_assess", "scored", "not_applicable"]

SEEDS = {
    "text": TEXT,
    "source": TEXT,
    "mentions": MENTIONS,
    "context": CONTEXT,
    "target": DATUM,
    "session": SESSION_VIEW,
    "identities": IDENTITIES,
    "record": RECORD,
}
_SEAL_KEY = secrets.token_bytes(32)


@dataclass(frozen=True)
class Count:
    """A count that is known exactly, as an upper or lower bound, or explicitly unknown (never confused with 0)."""

    value: int | None
    bound: str = "exact"  # exact | upper | lower | unknown

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
    session_mode: SessionMode = SessionMode()
    partial_release: bool = False

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
    session: Session | None = None,
    partial_release: bool = False,
) -> PreparedProtection:
    """Validate everything before effects and compile an immutable plan. No callback runs here.

    ``capabilities`` names the operation implementations admitted for this invocation (default: the workflow's);
    ``request_budget`` rejects the workflow here when its planned external-request count is unknown, opaque or
    above the budget; it is not enforced while the workflow runs. ``session`` is read for its mode only; the
    prepared value holds no live Session, and ``protect`` must get a Session of the same mode. A declared Record is
    released whole unless ``partial_release`` is true, which releases each member on its own checks.
    """
    if not isinstance(partial_release, bool):
        raise k.GraphRejection("invalid_partial_release")
    _reject_bare_string("capabilities", capabilities, "operation names")
    mode = _session_mode(session)
    if mode.identity_matching:
        raise k.GraphRejection("session_mode_unsupported")
    with _measure("graph_prepare", operation_count=None, target_count=len(graph.targets)) as m:
        prepared = _prepare(graph, workflow, limits, capabilities, request_budget, mode, partial_release)
        m["operation_count"] = len(prepared.names)
    return prepared


def _prepare(
    graph: ProtectionGraph,
    workflow: Workflow,
    limits: Limits | None,
    capabilities: tuple[str, ...] | None,
    request_budget: int | None,
    mode: SessionMode,
    partial_release: bool = False,
) -> PreparedProtection:
    limits = limits or Limits()

    def requirements(w: Workflow) -> tuple[str, ...]:
        return (*w.requires, *(c for s in w.steps if isinstance(s, Workflow) for c in requirements(s)))

    def annotates(w: Workflow) -> bool:
        return any(isinstance(s, Workflow) and (s.release != "protected" or annotates(s)) for s in w.steps)

    # A nested annotation keeps source text; releasing it as protected would drop that restriction. Any nested
    # release other than "protected" is refused, so a misspelled one cannot slip through.
    if workflow.release == "protected" and annotates(workflow):
        raise k.GraphRejection("annotation_in_protected", workflow.name)
    # A nested workflow's requirements are requirements of the invocation.
    workflow = replace(workflow, requires=tuple(dict.fromkeys(requirements(workflow))))
    ops = workflow.flatten()
    # A producer read selects an operation by its flattened name, so each name must be unique.
    if duplicates := [name for name, n in Counter(name for name, _ in ops).items() if n > 1]:
        raise k.GraphRejection("duplicate_operation_name", ", ".join(duplicates))
    admitted = frozenset(name for name, _ in ops) if capabilities is None else frozenset(capabilities)
    _admit(graph, ops, limits, admitted)
    if mode.stable_replacements and not any(DECISIONS in op.outputs.values() for _, op in ops):
        raise k.GraphRejection("session_without_resolution")
    planned = _planned(ops, len(graph.targets))
    if request_budget is not None and (planned.value is None or planned.bound == "opaque"):
        raise k.GraphRejection("hard_budget_without_visibility")  # never downgraded to soft
    if request_budget is not None and planned.value is not None and planned.value > request_budget:
        raise k.GraphRejection("over_request_budget")
    nodes = tuple(k.Node(name, op.inputs, op.outputs, op.fn, op.scope) for name, op in ops)
    plan = k.compile_sequence(nodes, {**SEEDS, "context": _context_kind(graph)})
    final_text = max((i for i, n in enumerate(nodes) if "text" in n.outputs), default=None)
    if final_text is not None and nodes[final_text].outputs["text"] != TEXT:
        raise k.GraphRejection("final_text_not_text", nodes[final_text].name)
    covering = _covering(ops, plan, -1 if final_text is None else final_text)
    answered = {c for i in covering for c in ops[i][1].checks}
    if unanswered := [c for c in workflow.requires if c not in answered]:
        # Evidence on earlier text is a diagnostic; a requirement only it answers could never be met.
        raise k.GraphRejection("required_check_not_final", ", ".join(unanswered))
    if planned.bound == "opaque":
        planned = Count(None, "unknown")
    names = tuple(n for n, _ in ops)
    prepared = PreparedProtection(
        graph,
        workflow,
        limits,
        plan,
        names,
        planned,
        covering,
        admitted,
        final_text,
        session_mode=mode,
        partial_release=partial_release,
    )
    return replace(prepared, seal=_seal(prepared))


def _release_units(g: ProtectionGraph, partial: bool) -> tuple[tuple[Datum, ...], ...]:
    """The all-or-none release units: the output groups, merged with every declared Record of two or more members.

    Under partial release the output groups alone. Otherwise groups and Records that share a target merge into one
    unit (a flat partition). A group that touches no multi-member Record keeps its members and its position; a
    merged unit lists its members in target order. The graph's own declarations are not changed.
    """
    groups = [grp.members for grp in g.groups]
    records = [r.members for r in g.records if len(r.members) > 1]
    if partial or not records:
        return tuple(groups)
    sets = [set(m) for m in groups + records]
    parent = list(range(len(sets)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner: dict[Datum, int] = {}
    for i, members in enumerate(sets):
        for m in members:
            if m in owner:
                parent[root(i)] = root(owner[m])
            else:
                owner[m] = i
    merged: dict[int, set[Datum]] = {}
    for i, members in enumerate(sets):
        merged.setdefault(root(i), set()).update(members)
    order = {d: n for n, d in enumerate(g.targets)}
    with_record = {root(j) for j in range(len(groups), len(sets))}
    units: list[tuple[Datum, ...]] = []
    for i, members in enumerate(sets):
        r = root(i)
        if r not in merged:
            continue
        unit = merged.pop(r)
        if i < len(groups) and unit == members and r not in with_record:
            units.append(groups[i])  # an untouched group keeps its declared member order
        else:
            units.append(tuple(sorted(unit, key=order.__getitem__)))
    return tuple(units)


def _covering(ops: tuple[tuple[str, Operation], ...], plan: k.Plan, final: int) -> frozenset[int]:
    """The verifiers whose every declared subject derives from the final candidate, from declared ports only.

    A candidate version is the index of the operation whose ``text`` output it is, or -1 for the source text. The
    seeds ``text``, ``source``, ``mentions`` and ``target`` derive from the source text; ``context``, ``session``,
    ``identities`` and ``record`` from none.
    Interim rule: every output of a text-producing operation derives from the version it emits, so evidence it
    returns beside its text counts as about that text. Any other operation's outputs derive from all its inputs.
    """
    outputs: list[dict[str, frozenset[int]]] = []
    covering = set()
    for i, (name, op) in enumerate(ops):
        read = {
            port: (frozenset() if port in ("context", "session", "identities", "record") else frozenset({-1}))
            if source.node is None
            else outputs[source.node][source.port]
            for port, source in plan.sources[i].items()
        }
        if op.subject and not op.is_verifier:
            raise k.GraphRejection("verifier_subject_invalid", f"{name}: not a verifier")
        if op.is_verifier and op.checks and not op.subject:
            raise k.GraphRejection("verifier_subject_invalid", f"{name}: undeclared")
        for port in op.subject:
            if port not in read:
                raise k.GraphRejection("verifier_subject_invalid", f"{name}.{port}: not an input")
            if not read[port]:
                raise k.GraphRejection("verifier_subject_invalid", f"{name}.{port}: about no candidate")
        if op.subject and all(final in read[port] for port in op.subject):
            covering.add(i)
        derived = frozenset({i}) if "text" in op.outputs else frozenset().union(*read.values())
        outputs.append(dict.fromkeys(op.outputs, derived))
    return frozenset(covering)


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
class Advisory:
    """A check the workflow does not require, as a final-text verifier assessed it: ``passed``, ``failed`` or
    ``could_not_assess``; ``cause`` is the verifier's stated cause for ``failed`` and ``could_not_assess``. It never
    changes qualification or release. Fields hold codes and declared names, never record text.

    The kernel emits only those three. The ``Anonymizer`` facade also reports ``evaluate()`` judges as advisories
    and adds ``scored`` (an ungraded judge returned scores) and ``not_applicable`` (the judge skips such rows by
    design)."""

    check: str
    verifier: str
    status: AdvisoryStatus
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
    advisories: tuple[Advisory, ...] = ()

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
    budget: str | None = None  # exceeded | within | unknown: observed requests against planned_requests

    def render(self) -> str:
        ops = ", ".join(name[:64] for name in self.operations)
        lines = [
            f"workflow: {len(self.operations)} operations [{ops}]; targets: {self.targets}",
            f"planned external requests: {self.planned_requests}",
        ]
        if self.terminals is None:
            lines.append("terminals: none (not executed)")
        else:
            fmt = lambda pairs: ", ".join(f"{a}={b}" for a, b in pairs) or "-"
            lines += [
                f"terminals: {fmt(self.terminals)}; qualification: {fmt(self.qualifications or ())}",
                f"reasons: {fmt(self.reasons)}; protected: {self.protected}; consistent: {self.consistent}",
                f"observed: {fmt(self.observed)}; budget: {self.budget}",
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
    session_changes: SessionChanges = SessionChanges()

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
            names = self.prepared.names
            return outputs.get(names.index(producer), {}).get(port) if producer in names else None
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

    def assemble_regions(self, source: str, regions: Sequence[tuple[int, int, Datum]]) -> str:
        """Region assembly: ``source`` with each ``(start, end, target)`` range replaced by that target's candidate.

        Ranges index the original ``source`` whatever the replacement lengths; text outside them is kept. A range
        outside ``source`` (``region_out_of_bounds``) or overlapping another (``overlapping_region``) is rejected,
        and a target without an available candidate is ``unavailable_input``, never filled with source text.
        """
        out, at = [], 0
        for start, end, d in sorted(regions, key=lambda r: (r[0], r[1])):
            if not 0 <= start <= end <= len(source):
                raise k.GraphRejection("region_out_of_bounds", f"{start}:{end}")
            if start < at:
                raise k.GraphRejection("overlapping_region", f"{start}:{end}")
            out += [source[at:start], d]
            at = end
        return "".join(x if isinstance(x, str) else self._available(x) for x in [*out, source[at:]])

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
        retrieved = sum(r.status != "not_started" for r in self.retrievals)
        binding = Count(retrieved)
        requests, seen = self._observed_requests()
        # Context retrievals plus the requests steps reported; only as exact as the reports are complete.
        total = requests if requests.value is None else Count(retrieved + seen, requests.bound)
        plan = p.planned_requests.value
        budget = (
            "unknown"
            if plan is None
            else "exceeded"
            if seen > plan
            else "within"
            if requests.bound == "exact"
            else "unknown"
        )
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
            budget,
        )

    def _observed_requests(self) -> tuple[Count, int]:
        """The reported sum: exact when every dispatched step declared ``none`` or reported (an invocation step
        once), a lower bound when only some reported, unknown when none did; and the sum itself."""
        ops = self.prepared.workflow.flatten()
        total, covered, reported = 0, True, False
        for node in {e.node for e in self.state.entries}:
            entries = [e for e in self.state.entries if e.node == node]
            reports = [e.observed_requests for e in entries if e.observed_requests is not None]
            total += sum(reports)
            reported = reported or bool(reports)
            _, op = ops[node]
            needed = 1 if op.scope == "phase_scoped" else len(entries)
            covered = covered and (op.requests.bound == "none" or len(reports) >= needed)
        if covered:
            return Count(total), total
        return (Count(total, "lower") if reported else Count(None, "unknown")), total

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


def protect(
    prepared: PreparedProtection, *, cancel: k.CancelToken | None = None, session: Session | None = None
) -> ProtectionResult:
    """Execute synchronously on the calling thread in a copy of the calling context.

    Callbacks and providers run on this thread. A raising callback closes its target as failed; nothing is raised.
    ``cancel`` stops new dispatches and retrievals; a frame nested in another inherits its
    cancellation.

    ``session`` (default: a fresh off/off Session private to the call) must have the mode ``prepare`` saw
    (``session_mismatch``) and no other execution may hold it (``session_busy``); both are checked before effects.
    """
    if not isinstance(prepared, PreparedProtection) or prepared.seal != _seal(prepared):
        raise k.GraphRejection("not_prepared")
    _admit(prepared.graph, prepared.workflow.flatten(), prepared.limits, prepared.capabilities)
    if _session_mode(session) != prepared.session_mode:
        raise k.GraphRejection("session_mismatch")
    # Without a Session, a fresh private off/off one: nothing else can hold it, and it accepts nothing.
    session = Session() if session is None else session
    with session._hold(), _measure("graph_protect", **_counts(prepared)) as m:
        result = k.run_in_frame(_protect, prepared, session, token=cancel)
        m["terminal_counts"] = dict(Counter(o.terminal for o in result.outcomes))
    return result


def _session_mode(session: Session | None) -> SessionMode:
    if session is not None and not isinstance(session, Session):
        raise k.GraphRejection("not_a_session")
    return SessionMode() if session is None else session.mode


def _measure(stage: str, **fields: Any) -> contextlib.AbstractContextManager[dict[str, Any]]:
    """A measurement stage when a measurement session can be active (its module is loaded), else a no-op."""
    if "anonymizer.measurement.session" not in sys.modules:
        return contextlib.nullcontext(fields)
    from anonymizer.measurement.recorders import stage_timer

    return stage_timer(stage, **fields)


def _counts(p: PreparedProtection) -> dict[str, int]:
    """The content-free sizes a boundary measurement carries."""
    return {"operation_count": len(p.names), "target_count": len(p.graph.targets)}


def _protect(p: PreparedProtection, session: Session) -> ProtectionResult:
    token = k.current_token()
    cancelled = (lambda: token.cancelled) if token is not None else (lambda: False)
    g = p.graph
    retrievals, bound = _bind(g, p.limits, cancelled)
    mentions: dict[Datum, list] = {}
    for m in sorted(g.mentions, key=lambda m: m.start):
        mentions.setdefault(m.datum, []).append(m)
    view = session._view()
    identities: dict[Datum, list] = {}
    for i in sorted(g.identities, key=lambda i: (i.start, i.end)):
        identities.setdefault(i.target, []).append(i)
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
        seeds.append(
            {
                "text": d.text,
                "source": d.text,
                "mentions": m,
                "context": tuple(items),
                "target": d,
                "session": view,
                "identities": tuple(identities.get(d, ())),
                "record": g.record_of(d),
            }
        )
    state = k.initial_state(p.plan, seeds)
    for u in closed:
        k.apply(state, k.Closed(u, "binding_failed"))
    k.execute(p.plan, state, cancelled, _mention_review(p))
    with _measure("graph_reconcile", **_counts(p)) as m:
        result = _reconcile(p, state, tuple(retrievals))
        m["terminal_counts"] = dict(Counter(o.terminal for o in result.outcomes))
    return replace(result, session_changes=_accept_decisions(p, state, result.consistent, session))


def _acceptance_order(graph: ProtectionGraph) -> range:
    """The target order in which decisions are accepted; the first fixes a new slot."""
    return range(len(graph.targets))


def _accept_decisions(p: PreparedProtection, state: k.State, consistent: bool, session: Session) -> SessionChanges:
    """Accept the ``decisions`` outputs of every succeeded operation, whatever the release; a cancelled or
    inconsistent invocation accepts nothing. Order: target, then operation, then the emitted tuple."""
    if state.cancelled or not consistent:
        return SessionChanges()
    ports = [[port for port, name in node.outputs.items() if name == DECISIONS] for node in p.plan.nodes]
    found: list[tuple[Decision, str]] = []
    for u in _acceptance_order(p.graph):
        outputs = state.rows[u].outputs
        for i in sorted(outputs):
            found += [(d, p.names[i]) for port in ports[i] for d in outputs[i][port]]
    return session._accept(found)


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
        try:  # a copy, check or measure that rejects or raises on an item makes it invalid
            if status == "ok" and found.copy:  # the kind's copy is what is checked, measured and bound
                items = [found.copy(t) for t in items]
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
    advisory_of = [[c for c in op.checks if c not in requires] for _, op in p.workflow.flatten()]
    advice: list[tuple[Advisory, ...]] = []
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
        found: list[Advisory] = []
        for i in sorted(p.covering_verifiers):
            a = row.outputs.get(i, {}).get("assessment")
            if a is None or not advisory_of[i]:
                continue
            status = "passed" if a.satisfied is True else "could_not_assess" if a.satisfied is None else "failed"
            cause = "" if status == "passed" else a.reason
            found += [Advisory(c, p.names[i], status, cause) for c in advisory_of[i]]
        advice.append(tuple(found))
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
                    reasons += [Reason("failed", c, name, a.reason) for c in checks]
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
    units = _release_units(g, p.partial_release)
    group_of = {m: i for i, members in enumerate(units) for m in members}
    withheld = [d for d in g.targets if d not in released]
    releasable = set(released)
    withheld_by: dict[Datum, list[Datum]] = {}
    spent: set[int] = set()
    for d in withheld:  # grows while iterating
        i = group_of.get(d, -1)
        members = () if i < 0 or i in spent else units[i]
        spent.add(i)
        for nxt in dependents.get(d, ()):
            if nxt in releasable:
                withheld_by.setdefault(nxt, []).append(d)
        for nxt in (*dependents.get(d, ()), *members):
            if nxt in released:
                released.discard(nxt)
                withheld.append(nxt)
    # A group member withheld only by its group names every member withheld for another cause. Those members
    # share one tuple; each has no entry yet, since a member with an entry is itself a cause.
    shared: dict[Datum, tuple[Datum, ...]] = {}
    for i in spent - {-1}:
        members = units[i]
        causes = tuple(m for m in members if m not in releasable or m in withheld_by)
        cause_set = set(causes)
        shared.update((m, causes) for m in members if m not in cause_set)
    outcomes = []
    for u, (d, (terminal, qual, reasons)) in enumerate(zip(g.targets, verdicts)):
        available = "text" in state.rows[u].outputs.get(p.final_text, {})
        if qual == "met" and d not in released and not reasons:
            reasons.append(Reason("withheld" if p.workflow.release == "protected" else "annotation"))
        failed = failures.get(u)
        error = failed.error if failed is not None and terminal != "success" else None
        by = shared.get(d) or tuple(withheld_by.get(d, ()))
        outcomes.append(TargetOutcome(terminal, qual, available, d in released, tuple(reasons), error, by, advice[u]))
    return ProtectionResult(p, state, retrievals, tuple(outcomes), consistent)
