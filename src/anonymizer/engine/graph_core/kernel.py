# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Domain-free graph kernel: artifact-kind registry, plan topology, sequential executor, execution frame.

The kernel knows no Anonymizer artifact kind; it owns graph structure, topology and readiness only. Layers above
register the kinds they need with :func:`register_kind`. Execution runs a plan's nodes in order over every unit and
advances only by applying events to a :class:`State` (:func:`apply`); outcomes are derived from the final state by
the caller.
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
    """Bind each input port to the latest earlier output of the same name, else to a seed; check kinds.

    Each node runs after its predecessor, so execution preserves the declared order.
    """
    names = (*seeds.values(), *(name for n in nodes for name in (*n.inputs.values(), *n.outputs.values())))
    unknown = next((name for name in names if name not in _KINDS), None)
    if unknown is not None:
        raise GraphRejection("unknown_kind", repr(unknown))
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
    """One accounting entry per actual dispatch."""

    unit: int
    node: int
    status: Status
    reason: str = ""
    error: BaseException | None = field(default=None, repr=False, compare=False)
    ref: str = field(default_factory=lambda: secrets.token_hex(8))  # random, invocation-local


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


@dataclass(frozen=True)
class Dispatched:
    unit: int
    node: int


@dataclass(frozen=True)
class Completed:
    unit: int
    node: int
    outputs: Mapping[str, Any]


@dataclass(frozen=True)
class Failed:
    unit: int
    node: int
    reason: str
    error: BaseException | None = None


@dataclass(frozen=True)
class Closed:
    """A unit closed before dispatch (e.g. its binding failed); its nodes become blocked."""

    unit: int
    reason: str


@dataclass(frozen=True)
class CancelRequested:
    pass


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
        case Completed(unit, node, outputs) if state.rows[unit].status[node] == Status.RUNNING:
            row = state.rows[unit]
            row.status[node] = Status.SUCCEEDED
            row.outputs[node] = outputs
            state.history.append(Entry(unit, node, Status.SUCCEEDED))
        case Failed(unit, node, reason, error) if state.rows[unit].status[node] == Status.RUNNING:
            # Dependents stay pending and never run; the ledger reports them blocked.
            state.rows[unit].status[node] = Status.FAILED
            state.history.append(Entry(unit, node, Status.FAILED, reason, error))


def unit_terminal(row: Row, cancelled: bool) -> Status:
    """Derived per-unit status: failed, cancelled (stopped before all nodes ran), succeeded or blocked."""
    if row.closed or Status.FAILED in row.status:
        return Status.FAILED
    if all(s == Status.SUCCEEDED for s in row.status):
        return Status.SUCCEEDED
    return Status.CANCELLED if cancelled and Status.PENDING in row.status else Status.BLOCKED


def _ready(row: Row, node: int) -> bool:
    """A node is ready when it is pending, the unit is open and every earlier node succeeded."""
    return (
        row.status[node] == Status.PENDING and not row.closed and all(s == Status.SUCCEEDED for s in row.status[:node])
    )


def _inputs(plan: Plan, row: Row, node: int) -> dict[str, Any]:
    return {
        port: row.seeds[port] if src.node is None else row.outputs[src.node][port]
        for port, src in plan.sources[node].items()
    }


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


def _accept(state: State, node: Node, index: int, unit: int, raw: Any) -> None:
    """Complete the node with a returned value's outputs, or fail it as ``invalid_output``."""
    try:
        event = Completed(unit, index, _outputs(node, raw))
    except TypeError as exc:
        event = Failed(unit, index, "invalid_output", exc)
    apply(state, event)


def _run_invocation(state: State, node: Node, index: int, units: list[int], calls: list[dict[str, Any]]) -> None:
    """One call over every unit: one tuple per input port, aligned with ``units``; one returned entry per unit.

    An entry of ``None`` fails its unit as ``missing_output``, an ``Exception`` instance as ``raised``; any other
    value is checked as a target operation's output. A raise or a malformed return fails every unit alike.
    """
    reason = "raised"
    try:
        raw = node.impl(**{port: tuple(args[port] for args in calls) for port in node.inputs})
    except Exception as exc:
        raw = [exc] * len(units)
    else:
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != len(units):
            size = f" of length {len(raw)}" if hasattr(raw, "__len__") else ""
            error = TypeError(
                f"invocation operation returned {type(raw).__name__}{size}, expected {len(units)} entries"
            )
            raw, reason = [error] * len(units), "invalid_output"
    for unit, entry in zip(units, raw):
        match entry:
            case None:
                apply(state, Failed(unit, index, "missing_output"))
            case Exception():
                apply(state, Failed(unit, index, reason, entry))
            case _:
                _accept(state, node, index, unit, entry)


def execute(plan: Plan, state: State, cancelled: Callable[[], bool]) -> None:
    """Run the nodes in order, each over every ready unit in order, on the calling thread.

    Cancellation is checked before each per-target call and before an invocation-scoped call. A callable that
    raises ``Exception`` fails its unit (an invocation: every unit); nothing is raised to the caller.
    ``BaseException`` (``KeyboardInterrupt``) propagates unchanged, as in 0.4.0.
    """
    for index, node in enumerate(plan.nodes):
        ready = [u for u, row in enumerate(state.rows) if _ready(row, index)]
        batches = [ready] if node.scope == "invocation" and ready else [[u] for u in ready]
        for units in batches:
            if cancelled():
                apply(state, CancelRequested())
            if state.cancelled:
                return
            for unit in units:
                apply(state, Dispatched(unit, index))
            calls = [_inputs(plan, state.rows[u], index) for u in units]
            if node.scope == "invocation":
                _run_invocation(state, node, index, units, calls)
                continue
            try:
                raw = node.impl(**calls[0])
            except Exception as exc:
                apply(state, Failed(units[0], index, "raised", exc))
            else:
                _accept(state, node, index, units[0], raw)
    if cancelled():
        apply(state, CancelRequested())


# ---------------------------------------------------------------- execution frame and cancellation


class CancelToken:
    """Synchronous cancellation request, settable from a callback or any caller thread.

    A frame's token is linked to the enclosing frame's token: cancelling an outer run also stops new dispatches
    of runs nested inside it.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._parent: CancelToken | None = None

    def link(self, parent: CancelToken) -> None:
        """Follow ``parent``'s cancellation unless a parent is already linked."""
        if self._parent is None and parent is not self:
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
    parent = _FRAME_TOKEN.get()
    if parent is not None:
        frame_token.link(parent)

    def body() -> Any:
        _FRAME_TOKEN.set(frame_token)
        return fn(*args, **kwargs)

    return contextvars.copy_context().run(body)
