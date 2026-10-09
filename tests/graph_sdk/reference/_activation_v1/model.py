# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent activation_v1 reference: model."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

Json: TypeAlias = str | int | bool | None | list["Json"] | dict[str, "Json"]


Object: TypeAlias = dict[str, Json]


Category: TypeAlias = Literal["success", "failure", "cancelled", "lost", "blocked", "inconsistent"]


Status: TypeAlias = Literal[
    "unstarted", "ready", "running", "success", "failure", "cancelled", "lost", "blocked", "inconsistent"
]


Role: TypeAlias = Literal["ordinary", "subgraph", "map_expander", "map_member", "join", "loop_starter", "loop_member"]


Boundary: TypeAlias = Literal[
    "event_construction", "static_admission", "dynamic_admission", "initialization", "transition"
]


Code: TypeAlias = Literal[
    "invalid_type",
    "invalid_value",
    "limit_exceeded",
    "foreign_owner",
    "duplicate",
    "missing",
    "overlap",
    "cycle",
    "contradictory",
]


CONTRACT_SHA256 = "9f58d60ad4ecc25065cc6c74d784cd05ad6fb2a72f183a615122eb981c7b265b"


CONSUMPTION_ADDENDUM_SHA256 = "359ef3adf7986685463e6eb80c0d96f893a39dcefead16ba2f76919f148b94d9"


CORPUS_PATH = "tests/graph_sdk/reference/activation_v1_cases.json"


GENERATOR_VERSION = "workflow-activation-v1-generator-8"


SELF_TEST_VERSION = "workflow-activation-v1-self-test-8"


SUPPORT_PATH = "tests/graph_sdk/reference/activation_v1_support.md"


TEMPLATES = ("N0", "N1", "N2")


INVOCATIONS = ("I0", "I1")


ACTIVATIONS = tuple(f"A{i}" for i in range(12))


CATEGORIES: tuple[Category, ...] = ("success", "failure", "cancelled", "lost", "blocked", "inconsistent")


TERMINAL = frozenset(CATEGORIES)


ALPHABET = (
    "initialize",
    "select",
    "start",
    "terminal_success",
    "terminal_failure",
    "terminal_cancelled",
    "terminal_lost",
    "terminal_blocked",
    "terminal_inconsistent",
    "close_blocked",
    "close_inconsistent",
    "membership_open",
    "membership_close",
    "membership_overflow",
)


FAMILY_IDS = (
    "sequence_single",
    "sequence_linked_pair",
    "sequence_independent_siblings",
    "sequence_mutations",
    "choice",
    "subgraph",
    "map",
    "join",
    "loop",
    "nested_map_loop",
    "precedence",
    "terminal_coverage",
)


RULE_IDS = (
    "different_parent_activations",
    "same_parent_unrelated_siblings",
    "no_sequence_choice_membership_loop_join_dependency",
)


ERROR_ORDER: tuple[Code, ...] = (
    "invalid_type",
    "invalid_value",
    "limit_exceeded",
    "foreign_owner",
    "duplicate",
    "missing",
    "overlap",
    "cycle",
    "contradictory",
)


RENAME_TEMPLATE = {"N0": "N2", "N1": "N0", "N2": "N1"}


INVERSE_TEMPLATE = {value: key for key, value in RENAME_TEMPLATE.items()}


RENAME_ACTIVATION = {f"A{i}": f"A{11 - i}" for i in range(12)}


INVERSE_ACTIVATION = {value: key for key, value in RENAME_ACTIVATION.items()}


@dataclass(frozen=True, slots=True)
class Limits:
    max_events: int
    max_entries: int
    max_parent_depth: int


@dataclass(frozen=True, slots=True)
class Seed:
    key: str
    template: str
    invocation: str
    parent: str | None
    iteration: int | None
    role: Role
    scope: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Outcome:
    scope: tuple[str, ...]
    template: str
    name: str
    category: Category
    produced_ports: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class InputDependency:
    source: str
    source_port: str
    destination: str
    destination_port: str


@dataclass(frozen=True, slots=True)
class Choice:
    selector: str
    branches: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True, slots=True)
class Subgraph:
    parent: str
    roots: tuple[str, ...]
    sink: str


@dataclass(frozen=True, slots=True)
class MapAggregate:
    parent: str
    members: tuple[str, ...]
    join: str
    bound: int
    expansion_outcomes: tuple[str, ...]
    accepted_categories: tuple[Category, ...]


@dataclass(frozen=True, slots=True)
class LoopAggregate:
    starter: str
    members: tuple[str, ...]
    join: str
    bound: int
    enter_outcomes: tuple[str, ...]
    bypass_outcomes: tuple[str, ...]
    continue_outcomes: tuple[str, ...]
    exit_outcomes: tuple[str, ...]
    initial_binding: tuple[str, str, str] | None
    carried_binding: tuple[str, str, str] | None
    member_scope: tuple[str, ...]
    member_template: str


Aggregate: TypeAlias = MapAggregate | LoopAggregate


@dataclass(frozen=True, slots=True)
class Declaration:
    invocation: str
    seeds: tuple[Seed, ...]
    required: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    input_dependencies: tuple[InputDependency, ...]
    outcomes: tuple[Outcome, ...]
    choices: tuple[Choice, ...]
    subgraphs: tuple[Subgraph, ...]
    aggregates: tuple[Aggregate, ...]
    limits: Limits
    static_groups: tuple[tuple[str, ...], ...]
    static_edges: tuple[tuple[str, str], ...]
    compatible: bool


@dataclass(frozen=True, slots=True)
class Initialize: ...


@dataclass(frozen=True, slots=True)
class Select:
    keys: tuple[str, ...]
    invocation: str


@dataclass(frozen=True, slots=True)
class Start:
    key: str
    invocation: str


@dataclass(frozen=True, slots=True)
class TerminalEvent:
    key: str
    invocation: str
    category: Category
    outcome: str | None


@dataclass(frozen=True, slots=True)
class Close:
    key: str
    invocation: str
    category: Literal["blocked", "inconsistent"]


@dataclass(frozen=True, slots=True)
class Membership:
    parent: str
    invocation: str
    members: tuple[str, ...]
    closed: bool


@dataclass(frozen=True, slots=True)
class Overflow:
    parent: str
    invocation: str
    count: int


Event: TypeAlias = Initialize | Select | Start | TerminalEvent | Close | Membership | Overflow


@dataclass(frozen=True, slots=True)
class Entry:
    activation: str
    scope: tuple[str, ...]
    template: str
    status: Status
    outcome: str | None
    category: Category | None
    produced_ports: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Expansion:
    parent: str
    status: Literal["pending", "closed", "failed", "overflow"]
    members: frozenset[str]


@dataclass(frozen=True, slots=True)
class ReferenceState:
    entries: frozenset[Entry]
    expansions: frozenset[Expansion]
    outputs: frozenset[str]
    events_applied: int
    complete: bool


class Rejected(ValueError):
    def __init__(self, *codes: Code) -> None:
        pool = frozenset(codes)
        self.code = next(code for code in ERROR_ORDER if code in pool)
        super().__init__(self.code)


def _canonical(value: Json) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def canonical_bytes(cases: Sequence[Object]) -> bytes:
    return _canonical(cast(Json, list(cases))) + b"\n"


def initial_capacity(reservation_count: int, map_expander_count: int = 0) -> tuple[int, int]:
    if isinstance(reservation_count, bool) or isinstance(map_expander_count, bool):
        raise TypeError("invalid capacity type")
    if reservation_count < 0 or map_expander_count < 0 or map_expander_count > reservation_count:
        raise ValueError("invalid capacity value")
    return reservation_count, 3 * reservation_count + map_expander_count


def completion_reserve(
    *, absent: int = 0, unstarted_or_ready: int = 0, running_ordinary: int = 0, open_map_expanders: int = 0
) -> int:
    values = absent, unstarted_or_ready, running_ordinary, open_map_expanders
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("invalid reserve type")
    if any(value < 0 for value in values):
        raise ValueError("invalid reserve value")
    return 3 * absent + 2 * unstarted_or_ready + running_ordinary + open_map_expanders


def _rename(value: Json, *, inverse: bool = False) -> Json:
    templates, activations = (INVERSE_TEMPLATE, INVERSE_ACTIVATION) if inverse else (RENAME_TEMPLATE, RENAME_ACTIVATION)
    if isinstance(value, list):
        return [_rename(item, inverse=inverse) for item in value]
    if isinstance(value, dict):
        return {
            templates.get(key, activations.get(key, key)): _rename(item, inverse=inverse) for key, item in value.items()
        }
    if isinstance(value, str):
        return templates.get(value, activations.get(value, value))
    return value
