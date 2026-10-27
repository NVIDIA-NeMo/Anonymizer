# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Domain-free graph kernel: artifact-kind registry, plan topology, sequential executor, execution frame.

The kernel knows no Anonymizer artifact kind; layers above register theirs with :func:`register_kind`. Execution runs
a plan's nodes in order over every unit, changing a :class:`State` only through :func:`apply`; the caller derives
outcomes from the final state.
"""

from __future__ import annotations

import contextvars
import secrets
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal, NamedTuple

from anonymizer.engine.graph_core.observe import activation


class GraphRejection(Exception):
    """Typed rejection raised before any callback runs. Not an ``AnonymizerError``."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


# ---------------------------------------------------------------- artifact kinds (routes register their own)


@dataclass(frozen=True)
class ArtifactKind:
    """A registered value kind for typed ports: a membership check and an optional UTF-8 byte measure."""

    name: str
    check: Callable[[Any], bool]
    measure: Callable[[Any], int | None] | None = None


_KINDS: dict[str, ArtifactKind] = {}
_KINDS_LOCK = threading.Lock()


def register_kind(name: str, check: Callable[[Any], bool], measure: Callable[[Any], int | None] | None = None) -> str:
    """Register an artifact kind once; re-registering the same name returns it unchanged (idempotent by name)."""
    with _KINDS_LOCK:
        _KINDS.setdefault(name, ArtifactKind(name, check, measure))
    return name


def kind(name: str) -> ArtifactKind:
    found = _KINDS.get(name)
    if found is None:
        raise GraphRejection("unknown_kind", repr(name))
    return found


# ---------------------------------------------------------------- plan topology


@dataclass(frozen=True)
class Node:
    """One operation node with typed input and output ports; ``impl`` is a caller-supplied callable."""

    name: str
    inputs: Mapping[str, str]
    outputs: Mapping[str, str]
    impl: Callable[..., Any] = field(repr=False, compare=False)
    scope: Literal["target", "invocation"] = "target"


class Source(NamedTuple):
    """Where an input port reads: a seed value (``node`` is None) or the output ``port`` of node ``node``."""

    node: int | None
    port: str


@dataclass(frozen=True)
class Plan:
    """Immutable compiled plan: nodes in declared order and each node's input sources."""

    nodes: tuple[Node, ...]
    sources: tuple[Mapping[str, Source], ...]
    seeds: Mapping[str, str]


def compile_sequence(nodes: tuple[Node, ...], seeds: Mapping[str, str]) -> Plan:
    """Bind each input port to the latest earlier output of the same name, else to a seed; check kinds."""
    kind_names = list(seeds.values())
    for node in nodes:
        kind_names.extend(node.inputs.values())
        kind_names.extend(node.outputs.values())
    for name in kind_names:
        if name not in _KINDS:
            raise GraphRejection("unknown_kind", repr(name))
    latest = {port: (Source(None, port), name) for port, name in seeds.items()}
    sources: list[Mapping[str, Source]] = []
    for i, node in enumerate(nodes):
        bound: dict[str, Source] = {}
        for port, want in node.inputs.items():
            if port not in latest:
                raise GraphRejection("unbound_port", f"{node.name}.{port}")
            source, have = latest[port]
            if have != want:
                raise GraphRejection("port_kind_mismatch", f"{node.name}.{port}: {have} != {want}")
            bound[port] = source
        sources.append(MappingProxyType(bound))
        latest.update({port: (Source(i, port), name) for port, name in node.outputs.items()})
    return Plan(tuple(nodes), tuple(sources), MappingProxyType(dict(seeds)))


# ---------------------------------------------------------------- sequential executor


class Status(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class Entry:
    """One accounting entry per actual dispatch.

    ``observed_requests`` is what the callable reported, ``None`` when it reported nothing. An invocation-scoped
    call is one call over all its units, so its count is recorded once, on the first unit's entry.
    """

    unit: int
    node: int
    status: Status
    reason: str = ""
    error: BaseException | None = field(default=None, repr=False, compare=False)
    ref: str = field(default_factory=lambda: secrets.token_hex(8))  # random, invocation-local
    observed_requests: int | None = None


@dataclass
class Row:
    """One unit's node statuses, seed values, node outputs by node index and pre-dispatch close."""

    status: list[Status]
    seeds: Mapping[str, Any]
    outputs: dict[int, Mapping[str, Any]] = field(default_factory=dict)
    closed: str = ""


@dataclass
class State:
    """Rows in unit order and entries in dispatch order, changed in place by :func:`apply`."""

    rows: list[Row] = field(repr=False)
    history: list[Entry] = field(default_factory=list, repr=False)
    cancelled: bool = False

    @property
    def entries(self) -> tuple[Entry, ...]:
        return tuple(self.history)


class Dispatched(NamedTuple):
    unit: int
    node: int


class Completed(NamedTuple):
    unit: int
    node: int
    outputs: Mapping[str, Any]
    observed: int | None = None


class Failed(NamedTuple):
    unit: int
    node: int
    reason: str
    error: BaseException | None = None
    observed: int | None = None


class Closed(NamedTuple):
    """A unit closed before dispatch (e.g. its binding failed); its nodes become blocked."""

    unit: int
    reason: str


class CancelRequested(NamedTuple): ...


Event = Dispatched | Completed | Failed | Closed | CancelRequested


def initial_state(plan: Plan, seeds: list[Mapping[str, Any]]) -> State:
    return State([Row([Status.PENDING] * len(plan.nodes), MappingProxyType(dict(s))) for s in seeds])


def apply(state: State, event: Event) -> None:
    """Apply one event in place. Accepted terminals are immutable: a later event cannot rewrite them."""
    match event:
        case CancelRequested():
            state.cancelled = True
        case Closed(unit, reason):
            row = state.rows[unit]
            if not row.closed:
                row.status = [Status.BLOCKED if s == Status.PENDING else s for s in row.status]
                row.closed = reason
        case Dispatched(unit, node):
            row = state.rows[unit]
            if state.cancelled or row.status[node] != Status.PENDING:
                raise RuntimeError(f"dispatch refused: {row.status[node].value!r}, cancelled={state.cancelled}")
            row.status[node] = Status.RUNNING
        case Completed(unit, node, outputs, observed):
            row = state.rows[unit]
            if row.status[node] == Status.RUNNING:
                row.status[node] = Status.SUCCEEDED
                row.outputs[node] = outputs
                state.history.append(Entry(unit, node, Status.SUCCEEDED, observed_requests=observed))
        case Failed(unit, node, reason, error, observed):
            row = state.rows[unit]
            if row.status[node] == Status.RUNNING:
                # Dependents stay pending and never run; the ledger reports them blocked.
                row.status[node] = Status.FAILED
                state.history.append(Entry(unit, node, Status.FAILED, reason, error, observed_requests=observed))


def unit_terminal(row: Row, cancelled: bool) -> Status:
    """Derived per-unit status: failed, cancelled (stopped before all nodes ran), succeeded or blocked."""
    if row.closed or Status.FAILED in row.status:
        return Status.FAILED
    if all(s == Status.SUCCEEDED for s in row.status):
        return Status.SUCCEEDED
    return Status.CANCELLED if cancelled and Status.PENDING in row.status else Status.BLOCKED


def _ready(row: Row, node: int) -> bool:
    return not row.closed and all(s == Status.SUCCEEDED for s in row.status[:node])


def _inputs(plan: Plan, row: Row, node: int) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for port, source in plan.sources[node].items():
        origin = row.seeds if source.node is None else row.outputs[source.node]
        values[port] = origin[port]
    return values


def _outputs(node: Node, raw: Any) -> Mapping[str, Any]:
    ports = list(node.outputs)
    if len(ports) == 1:
        raw = {ports[0]: raw}
    if not isinstance(raw, Mapping) or set(raw) != set(ports):
        raise TypeError("output ports mismatch")
    for port, value in raw.items():
        if not kind(node.outputs[port]).check(value):
            raise TypeError(f"output port {port!r} is not of kind {node.outputs[port]!r}")
    return MappingProxyType(dict(raw))


def _accept(state: State, node: Node, index: int, unit: int, raw: Any, observed: int | None = None) -> None:
    """Complete the node with a returned value's outputs, or fail it as ``invalid_output``."""
    try:
        event = Completed(unit, index, _outputs(node, raw), observed)
    except TypeError as exc:
        event = Failed(unit, index, "invalid_output", exc, observed)
    apply(state, event)


def _run_target(plan: Plan, state: State, index: int, unit: int) -> None:
    """One call for one unit; a raised ``Exception`` fails the unit as ``raised``."""
    apply(state, Dispatched(unit, index))
    node = plan.nodes[index]
    with activation() as observed:
        try:
            raw = node.impl(**_inputs(plan, state.rows[unit], index))
        except Exception as exc:
            apply(state, Failed(unit, index, "raised", exc, observed.requests))
            return
    _accept(state, node, index, unit, raw, observed.requests)


def _run_invocation(plan: Plan, state: State, index: int, units: list[int]) -> None:
    """One call over every unit; a returned ``None`` is ``missing_output``, an ``Exception`` instance ``raised``."""
    for unit in units:
        apply(state, Dispatched(unit, index))
    node = plan.nodes[index]
    calls = [_inputs(plan, state.rows[unit], index) for unit in units]
    raised: Exception | None = None
    raw: Any = None
    with activation() as observed:
        try:
            raw = node.impl(**{port: tuple(args[port] for args in calls) for port in node.inputs})
        except Exception as exc:
            raised = exc
    first = {units[0]: observed.requests}
    if raised is not None:
        for unit in units:
            apply(state, Failed(unit, index, "raised", raised, first.get(unit)))
        return
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != len(units):
        size = f" of length {len(raw)}" if hasattr(raw, "__len__") else ""
        error = TypeError(f"invocation operation returned {type(raw).__name__}{size}, expected {len(units)} entries")
        for unit in units:
            apply(state, Failed(unit, index, "invalid_output", error, first.get(unit)))
        return
    for unit, entry in zip(units, raw):
        count = first.get(unit)
        match entry:
            case None:
                apply(state, Failed(unit, index, "missing_output", None, count))
            case Exception():
                apply(state, Failed(unit, index, "raised", entry, count))
            case _:
                _accept(state, node, index, unit, entry, count)


def execute(plan: Plan, state: State, cancelled: Callable[[], bool]) -> None:
    """Run the nodes in order, each over every ready unit in order, on the calling thread.

    Cancellation is checked before each per-target call and the invocation call. An ``Exception`` fails its unit
    (an invocation: every unit); ``BaseException`` (``KeyboardInterrupt``) propagates unchanged.
    """
    for index, node in enumerate(plan.nodes):
        ready = [unit for unit, row in enumerate(state.rows) if _ready(row, index)]
        if node.scope == "invocation":
            if ready:
                if cancelled():
                    apply(state, CancelRequested())
                    return
                _run_invocation(plan, state, index, ready)
        else:
            for unit in ready:
                if cancelled():
                    apply(state, CancelRequested())
                    return
                _run_target(plan, state, index, unit)
    if cancelled():
        apply(state, CancelRequested())


# ---------------------------------------------------------------- execution frame and cancellation


class CancelToken:
    """Synchronous cancellation request, settable from any thread; an outer frame's cancel reaches nested runs."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._parent: CancelToken | None = None

    def link(self, parent: CancelToken | None) -> None:
        """Follow ``parent``'s cancellation unless a parent is already linked."""
        if self._parent is None and parent is not None and parent is not self:
            self._parent = parent

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set() or (self._parent is not None and self._parent.cancelled)


_FRAME_TOKEN: contextvars.ContextVar[CancelToken | None] = contextvars.ContextVar(
    "anonymizer_graph_frame", default=None
)


def current_token() -> CancelToken | None:
    """The cancellation token of the innermost running frame, if any."""
    return _FRAME_TOKEN.get()


def run_in_frame(fn: Callable[..., Any], *args: Any, token: CancelToken | None = None, **kwargs: Any) -> Any:
    """Run ``fn`` on the calling thread in a copy of the calling context (copy-in without copy-out)."""
    frame_token = token or CancelToken()
    frame_token.link(_FRAME_TOKEN.get())
    frame = contextvars.copy_context()
    frame.run(_FRAME_TOKEN.set, frame_token)
    return frame.run(fn, *args, **kwargs)
