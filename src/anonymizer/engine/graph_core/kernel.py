# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Domain-free graph kernel: artifact-kind registry, plan topology, reducer executor, execution frame.

The kernel knows no Anonymizer artifact kind; it owns graph structure, topology and readiness only. Layers above register the kinds they need with :func:`register_kind`.
Execution state advances only by applying events to an immutable :class:`State` (:func:`reduce`); outcomes are
derived from the final state by the caller.
"""

from __future__ import annotations

import contextvars
import secrets
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Literal


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


@dataclass(frozen=True)
class Plan:
    """Immutable compiled plan: nodes in declared order, per-node input sources and dependencies.

    ``sources[i][port]`` is ``(-1, port)`` for a seed value or ``(j, port)`` for node ``j``'s output.
    """

    nodes: tuple[Node, ...]
    sources: tuple[Mapping[str, tuple[int, str]], ...]
    deps: tuple[frozenset[int], ...]
    seeds: Mapping[str, str]


def compile_sequence(nodes: tuple[Node, ...], seeds: Mapping[str, str]) -> Plan:
    """Bind each input port to the latest earlier output of the same name, else to a seed; check kinds.

    Sequence order is a dependency edge (each node depends on its predecessor), so readiness preserves the
    declared order.
    """
    for k in (*seeds.values(), *(v for n in nodes for v in (*n.inputs.values(), *n.outputs.values()))):
        kind(k)
    sources: list[Mapping[str, tuple[int, str]]] = []
    deps: list[frozenset[int]] = []
    produced: dict[str, tuple[int, str]] = {}
    for i, node in enumerate(nodes):
        bound: dict[str, tuple[int, str]] = {}
        for port, want in node.inputs.items():
            if port in produced:
                j, k = produced[port]
                bound[port] = (j, port)
            elif port in seeds:
                k = seeds[port]
                bound[port] = (-1, port)
            else:
                raise GraphRejection("unbound_port", f"{node.name}.{port}")
            if k != want:
                raise GraphRejection("port_kind_mismatch", f"{node.name}.{port}: {k} != {want}")
        sources.append(MappingProxyType(bound))
        deps.append(frozenset({i - 1} if i else set()) | {j for j, _ in bound.values() if j >= 0})
        for port, k in node.outputs.items():
            produced[port] = (i, k)
    return Plan(tuple(nodes), tuple(sources), tuple(deps), MappingProxyType(dict(seeds)))


# ---------------------------------------------------------------- reducer executor

PENDING, RUNNING, SUCCEEDED, FAILED, BLOCKED, CANCELLED = (
    "pending",
    "running",
    "succeeded",
    "failed",
    "blocked",
    "cancelled",
)


@dataclass(frozen=True)
class Entry:
    """One accounting entry per actual dispatch."""

    unit: int
    node: int
    status: str
    reason: str = ""
    error: BaseException | None = field(default=None, repr=False, compare=False)
    ref: str = field(default_factory=lambda: secrets.token_hex(8))  # random, invocation-local


@dataclass(frozen=True)
class Row:
    """One unit's node statuses, produced values keyed ``(node, port)`` (node -1: seeds) and pre-dispatch close."""

    status: tuple[str, ...]
    values: Mapping[tuple[int, str], Any]
    closed: str = ""


class _Vec:
    """Persistent vector (32-way path-copying trie) so each event copies O(log n) slots, not the whole state."""

    __slots__ = ("_root", "_depth", "size")

    def __init__(self, root: tuple, depth: int, size: int) -> None:
        self._root, self._depth, self.size = root, depth, size

    @classmethod
    def of(cls, items: list[Any]) -> _Vec:
        nodes = [tuple(items[i : i + 32]) for i in range(0, len(items), 32)] or [()]
        depth = 1
        while len(nodes) > 1:
            nodes = [tuple(nodes[i : i + 32]) for i in range(0, len(nodes), 32)]
            depth += 1
        return cls(nodes[0], depth, len(items))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, i: int) -> Any:
        node: Any = self._root
        for level in range(self._depth - 1, -1, -1):
            node = node[(i >> (5 * level)) & 31]
        return node

    def set(self, i: int, value: Any) -> _Vec:
        def path(node: tuple, level: int) -> tuple:
            j = (i >> (5 * level)) & 31
            child = value if level == 0 else path(node[j], level - 1)
            return (*node[:j], child, *node[j + 1 :])

        return _Vec(path(self._root, self._depth - 1), self._depth, self.size)

    def __iter__(self) -> Any:
        return (self[i] for i in range(self.size))


@dataclass(frozen=True)
class State:
    rows: _Vec = field(repr=False)
    log: tuple | None = field(default=None, repr=False)  # entries as a persistent cons list, newest first
    cancelled: bool = False
    cursor: tuple[int, int] = (0, 0)  # breadth-first scan position (node, unit)

    def row(self, unit: int) -> Row:
        return self.rows[unit]

    @property
    def entries(self) -> tuple[Entry, ...]:
        out, cell = [], self.log
        while cell is not None:
            out.append(cell[0])
            cell = cell[1]
        return tuple(reversed(out))


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
    pending = (PENDING,) * len(plan.nodes)
    return State(_Vec.of([Row(pending, MappingProxyType({(-1, k): v for k, v in s.items()})) for s in seeds]))


def _set(items: tuple[Any, ...], i: int, value: Any) -> tuple[Any, ...]:
    return (*items[:i], value, *items[i + 1 :])


def reduce(plan: Plan, state: State, event: Event) -> State:
    """Apply one event. Accepted terminals are immutable: a later event cannot rewrite them."""
    if isinstance(event, CancelRequested):
        return state if state.cancelled else replace(state, cancelled=True)
    row = state.rows[event.unit]
    if isinstance(event, Closed):
        if row.closed:
            return state
        blocked = tuple(BLOCKED if s == PENDING else s for s in row.status)
        return replace(state, rows=state.rows.set(event.unit, Row(blocked, row.values, event.reason)))
    if isinstance(event, Dispatched):
        if state.cancelled or row.status[event.node] != PENDING:
            raise RuntimeError(f"dispatch refused: {row.status[event.node]!r}, cancelled={state.cancelled}")
        new = Row(_set(row.status, event.node, RUNNING), row.values, row.closed)
        return replace(state, rows=state.rows.set(event.unit, new))
    if row.status[event.node] != RUNNING:
        return state
    if isinstance(event, Completed):
        values = MappingProxyType({**row.values, **{(event.node, k): v for k, v in event.outputs.items()}})
        new = Row(_set(row.status, event.node, SUCCEEDED), values, row.closed)
        entry = Entry(event.unit, event.node, SUCCEEDED)
    else:  # Failed: dependents stay pending and never become ready; the ledger reports them blocked.
        new = Row(_set(row.status, event.node, FAILED), row.values, row.closed)
        entry = Entry(event.unit, event.node, FAILED, event.reason, event.error)
    return replace(state, rows=state.rows.set(event.unit, new), log=(entry, state.log))


def unit_terminal(row: Row, cancelled: bool) -> str:
    """Derived per-unit status: failed, cancelled (stopped before all nodes ran), succeeded or blocked."""
    if row.closed or FAILED in row.status:
        return FAILED
    if all(s == SUCCEEDED for s in row.status):
        return SUCCEEDED
    return CANCELLED if cancelled and PENDING in row.status else BLOCKED


def next_ready(plan: Plan, state: State) -> tuple[State, tuple[int, int] | None]:
    """Next pending activation in (node, unit) order whose dependencies succeeded.

    Breadth-first by stage: every dependency of node ``i`` is an earlier node, all closed before the scan reaches
    ``i``, so the cursor never moves back.
    """
    if state.cancelled:
        return state, None
    i, u = state.cursor
    n = len(state.rows)
    while i < len(plan.nodes) and n:
        row = state.rows[u]
        if row.status[i] == PENDING and all(row.status[d] == SUCCEEDED for d in plan.deps[i]):
            return replace(state, cursor=(i, u)), (u, i)
        i, u = (i, u + 1) if u + 1 < n else (i + 1, 0)
    return replace(state, cursor=(i, u)), None


def _inputs(plan: Plan, state: State, unit: int, node: int) -> dict[str, Any]:
    values = state.rows[unit].values
    return {port: values[src] for port, src in plan.sources[node].items()}


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


def _dispatch_invocation(plan: Plan, state: State, i: int) -> State:
    """Run invocation-scoped node ``i`` once over every unit that will ever be ready at it.

    The callable gets one tuple per input port, aligned with the dispatched units, and returns one entry per unit:
    ``None`` closes the unit as ``missing_output``, an ``Exception`` instance as ``raised``, any other value is
    checked as a target operation's output.
    """
    node = plan.nodes[i]
    units = [
        u
        for u, row in enumerate(state.rows)
        if row.status[i] == PENDING and all(row.status[d] == SUCCEEDED for d in plan.deps[i])
    ]
    for u in units:
        state = reduce(plan, state, Dispatched(u, i))
    per_unit = [_inputs(plan, state, u, i) for u in units]
    try:
        raw = node.impl(**{port: tuple(values[port] for values in per_unit) for port in node.inputs})
    except Exception as exc:
        for u in units:
            state = reduce(plan, state, Failed(u, i, "raised", exc))
        return state
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != len(units):
        size = f" of length {len(raw)}" if hasattr(raw, "__len__") else ""
        error = TypeError(f"invocation operation returned {type(raw).__name__}{size}, expected {len(units)} entries")
        for u in units:
            state = reduce(plan, state, Failed(u, i, "invalid_output", error))
        return state
    for u, value in zip(units, raw):
        if value is None:
            state = reduce(plan, state, Failed(u, i, "missing_output"))
        elif isinstance(value, Exception):
            state = reduce(plan, state, Failed(u, i, "raised", value))
        else:
            try:
                outputs = _outputs(node, value)
            except TypeError as exc:
                state = reduce(plan, state, Failed(u, i, "invalid_output", exc))
            else:
                state = reduce(plan, state, Completed(u, i, outputs))
    return state


def execute(plan: Plan, state: State, cancelled: Callable[[], bool]) -> State:
    """Drive the reducer to quiescence on the calling thread; callables run here, one at a time.

    A callable that raises ``Exception`` closes its activation as failed; nothing is raised to the caller.
    ``BaseException`` (``KeyboardInterrupt``) propagates unchanged, as in 0.4.0.
    """
    while True:
        if cancelled():
            state = reduce(plan, state, CancelRequested())
        state, ready = next_ready(plan, state)
        if ready is None:
            return state
        unit, i = ready
        node = plan.nodes[i]
        if node.scope == "invocation":
            state = _dispatch_invocation(plan, state, i)
            continue
        state = reduce(plan, state, Dispatched(unit, i))
        try:
            raw = node.impl(**_inputs(plan, state, unit, i))
        except Exception as exc:
            state = reduce(plan, state, Failed(unit, i, "raised", exc))
            continue
        try:
            outputs = _outputs(node, raw)
        except TypeError as exc:
            state = reduce(plan, state, Failed(unit, i, "invalid_output", exc))
        else:
            state = reduce(plan, state, Completed(unit, i, outputs))


# ---------------------------------------------------------------- execution frame and cancellation


class CancelToken:
    """Synchronous cancellation request, settable from a callback or any caller thread.

    A frame's token is linked to the enclosing frame's token: cancelling an outer run also stops new dispatches
    of runs nested inside it.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._parent: CancelToken | None = None

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
    if parent is not None and parent is not frame_token and frame_token._parent is None:
        frame_token._parent = parent

    def body() -> Any:
        _FRAME_TOKEN.set(frame_token)
        return fn(*args, **kwargs)

    return contextvars.copy_context().run(body)
