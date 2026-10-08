# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent activation_v1 reference: evaluation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import cast

from tests.graph_sdk.reference._activation_v1.model import (
    Code,
    Entry,
    Expansion,
    Initialize,
    Json,
    LoopAggregate,
    Object,
    ReferenceState,
    Rejected,
    _canonical,
    _rename,
)
from tests.graph_sdk.reference._activation_v1.parsing import (
    _list,
    _obj,
    _strs,
    parse_declaration,
    parse_event,
)
from tests.graph_sdk.reference._activation_v1.reducer import (
    _apply,
    _complete,
    _initialize,
    _reserve,
)


def _state_object(state: ReferenceState, *, digest: bool = True) -> Object:
    result: Object = {
        "complete": state.complete,
        "entries": [
            {
                "activation": item.activation,
                "category": item.category,
                "outcome": item.outcome,
                "produced_ports": list(item.produced_ports),
                "scope": list(item.scope),
                "status": item.status,
                "template": item.template,
            }
            for item in sorted(state.entries, key=lambda item: item.activation)
        ],
        "events_applied": state.events_applied,
        "expansions": [
            {"members": sorted(item.members), "parent": item.parent, "status": item.status}
            for item in sorted(state.expansions, key=lambda item: item.parent)
        ],
        "outputs": sorted(state.outputs),
    }
    if digest:
        result["semantic_hash"] = hashlib.sha256(_canonical(cast(Json, _state_object(state, digest=False)))).hexdigest()
    return result


def alpha_normalize_state(value: Mapping[str, Json], *, inverse: bool = False) -> Object:
    """Rename a state and recompute its deterministic whole-state digest."""
    renamed = cast(
        Object, _rename({key: item for key, item in value.items() if key != "semantic_hash"}, inverse=inverse)
    )
    renamed["entries"] = sorted(
        _list(renamed["entries"]),
        key=lambda item: cast(
            str,
            _obj(item, {"activation", "category", "outcome", "produced_ports", "scope", "status", "template"})[
                "activation"
            ],
        ),
    )
    renamed["outputs"] = sorted(_strs(renamed["outputs"]))
    expansions = _list(renamed["expansions"])
    for expansion_value in expansions:
        expansion = _obj(expansion_value, {"members", "parent", "status"})
        expansion["members"] = sorted(_strs(expansion["members"]))
    renamed["expansions"] = sorted(
        expansions, key=lambda item: cast(str, _obj(item, {"members", "parent", "status"})["parent"])
    )
    renamed["semantic_hash"] = hashlib.sha256(_canonical(renamed)).hexdigest()
    return renamed


def admit_dynamic_workflow(declaration: Mapping[str, Json]) -> Object:
    try:
        facts = parse_declaration(declaration)
        _initialize(facts)
        seeds = {seed.key: seed for seed in facts.seeds}
        for aggregate in facts.aggregates:
            if not isinstance(aggregate, LoopAggregate):
                continue
            starter = seeds.get(aggregate.starter)
            if starter is None or aggregate.initial_binding is None or aggregate.carried_binding is None:
                raise Rejected("missing")
            starter_outcomes = {
                outcome.name
                for outcome in facts.outcomes
                if (outcome.scope, outcome.template) == (starter.scope, starter.template)
            }
            member_outcomes = {
                outcome.name
                for outcome in facts.outcomes
                if (outcome.scope, outcome.template) == (aggregate.member_scope, aggregate.member_template)
            }
            if starter_outcomes != set((*aggregate.enter_outcomes, *aggregate.bypass_outcomes)):
                raise Rejected("missing")
            if member_outcomes != set((*aggregate.continue_outcomes, *aggregate.exit_outcomes)):
                raise Rejected("missing")
        return {"code": None, "state": None, "status": "accepted"}
    except Rejected as error:
        return {"code": error.code, "state": None, "status": "rejected"}


def admit_static_support(declaration: Mapping[str, Json]) -> Object:
    try:
        facts = parse_declaration(declaration)
        _initialize(facts)
        seeds = {seed.key: seed for seed in facts.seeds}
        for dependency in facts.input_dependencies:
            producer = seeds.get(dependency.source)
            if producer is None:
                raise Rejected("missing")
            named_outcomes = [
                outcome
                for outcome in facts.outcomes
                if (outcome.scope, outcome.template) == (producer.scope, producer.template)
            ]
            if not named_outcomes or any(
                dependency.source_port not in outcome.produced_ports for outcome in named_outcomes
            ):
                raise Rejected("missing")
        return {"code": None, "state": None, "status": "accepted"}
    except Rejected as error:
        return {"code": error.code, "state": None, "status": "rejected"}


def reduce_trace(declaration: Mapping[str, Json], events: Sequence[Mapping[str, Json]]) -> Object:
    try:
        facts = parse_declaration(declaration)
        admission = admit_dynamic_workflow(declaration)
        if admission["status"] == "rejected":
            raise Rejected(cast(Code, admission["code"]))
        if not events:
            raise Rejected("missing")
        first = parse_event(events[0])
        if not isinstance(first, Initialize):
            raise Rejected("missing")
        _initialize(facts)
        entries: dict[str, Entry] = {}
        expansions: dict[str, Expansion] = {}
        outputs: set[str] = set()
        applied = 0
        for raw_event in events[1:]:
            try:
                event = parse_event(raw_event)
            except Rejected as error:
                if error.code in ("invalid_type", "invalid_value"):
                    raise
                if applied + 1 > facts.limits.max_events:
                    raise Rejected("limit_exceeded") from None
                raise
            if applied + 1 > facts.limits.max_events:
                raise Rejected("limit_exceeded")
            next_entries, next_expansions, next_outputs = dict(entries), dict(expansions), set(outputs)
            _apply(event, next_entries, next_expansions, next_outputs, facts)
            next_applied = applied + 1
            if (
                len(next_entries) > facts.limits.max_entries
                or next_applied + _reserve(next_entries, next_expansions, facts) > facts.limits.max_events
            ):
                raise Rejected("limit_exceeded")
            entries, expansions, outputs, applied = next_entries, next_expansions, next_outputs, next_applied
        state = ReferenceState(
            frozenset(entries.values()),
            frozenset(expansions.values()),
            frozenset(outputs),
            applied,
            _complete(entries, expansions, facts),
        )
        return {"code": None, "state": _state_object(state), "status": "accepted"}
    except Rejected as error:
        return {"code": error.code, "state": None, "status": "rejected"}
