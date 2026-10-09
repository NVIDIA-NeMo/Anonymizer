# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent effects_v1 reference: request facts."""

from __future__ import annotations

from typing import cast

from tests.graph_sdk.reference._effects_v1.model import (
    Json,
    Object,
    _array,
    _object,
    _strings,
)


def _initial(declaration: Object) -> Object:
    state: Object = {
        "artifacts": [],
        "association_requests": {},
        "association_terminals": {},
        "attempts": {},
        "binding_sources": {},
        "binding_declarations": declaration.get("binding_declarations", {}),
        "binding_terminal": None,
        "bindings": {},
        "cancel_requested": [],
        "cleanup": {},
        "closed_unstarted": {},
        "decisions": {},
        "defects": [],
        "denials": {},
        "dispatched": [],
        "dispatched_count": 0,
        "local_in_flight": [],
        "policies": declaration.get("policies", {}),
        "remote_outstanding": [],
        "request_associations": {},
        "request_failures": {},
        "request_facts": {},
        "request_policies": {},
        "reservation_policies": {},
        "reservations": {},
        "resource_count": 0,
        "resources": {},
        "settlements": {},
        "tasks": {},
        "task_requests": {},
        "terminals": {},
    }
    raw_specs = declaration.get("materializations", [])
    if (
        declaration.get("binding_cleanup_projection") is True
        or isinstance(raw_specs, list)
        and any(isinstance(raw, dict) and raw.get("version_selection") == "latest" for raw in raw_specs)
    ):
        state.update(
            {
                "allocator_next": 0,
                "binding_artifacts": [],
                "binding_cleanup": {},
                "binding_cleanup_associations": {},
                "binding_finished": False,
                "lineage_allocations": {},
                "operation_occurrences": {},
                "publication_attempts": [],
                "publication_failures": [],
            }
        )
    return state


def _reject(code: str) -> Object:
    return {"code": code, "status": "rejected"}


def _unique(values: list[Json], value: Json) -> None:
    if value not in values:
        values.append(value)


def _terminal(state: Object, request: str, category: str) -> None:
    terminals = _object(state["terminals"])
    if request not in terminals:
        terminals[request] = category
    elif terminals[request] != category:
        _unique(_array(state["defects"]), "conflicting_terminal")


def _request_fact(state: Object, request: str, fact: Object) -> None:
    facts = _object(state["request_facts"])
    if request not in facts:
        facts[request] = fact


def _remove(state: Object, key: str, value: str) -> None:
    values = _strings(state[key])
    if value in values:
        values.remove(value)
        state[key] = values


def _valid_usage(value: Json) -> bool:
    if value == "unknown":
        return True
    if not isinstance(value, dict) or set(value) != {"input", "output"}:
        return False
    return all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in value.values())


def _valid_settlement(value: Json, *, optional: bool) -> bool:
    if value is None:
        return optional
    if not isinstance(value, dict) or set(value) != {"disposition", "remote_stopped", "usage"}:
        return False
    disposition = value.get("disposition")
    remote_stopped = value.get("remote_stopped")
    return (
        disposition in ("completed", "rejected", "stopped", "unknown")
        and (remote_stopped is None or isinstance(remote_stopped, bool))
        and (disposition == "unknown") == (remote_stopped is not True)
        and _valid_usage(cast(Json, value.get("usage")))
    )


def _embedded_settlement() -> Object:
    return {
        "disposition": "completed",
        "remote_stopped": True,
        "usage": {"input": 0, "output": 0},
    }


def _apply_embedded_settlement(state: Object, request: str, value: Json) -> None:
    if value is None:
        return
    settlement = _object(value)
    settlements = _object(state["settlements"])
    if request in settlements and settlements[request] != settlement:
        _unique(_array(state["defects"]), "conflicting_settlement")
    else:
        settlements[request] = settlement
        if settlement["remote_stopped"] is True:
            _remove(state, "remote_outstanding", request)


def _record_request_failure(state: Object, request: str, failure: str) -> bool:
    first_terminal = request not in _object(state["terminals"])
    if first_terminal:
        _object(state["request_failures"])[request] = failure
        _request_fact(state, request, {"condition": "failure", "failure": failure})
        for association in _strings(_object(state["request_associations"])[request]):
            if _object(state["association_requests"]).get(association) == request:
                _object(state["association_terminals"])[association] = {
                    "failure": failure,
                    "policy": _object(state["request_policies"])[request],
                    "request": request,
                }
    else:
        fact = _object(_object(state["request_facts"]).get(request, {}))
        if fact.get("condition") != "failure" or fact.get("failure") != failure:
            _unique(_array(state["defects"]), "conflicting_terminal")
    _terminal(state, request, "failure")
    _remove(state, "local_in_flight", request)
    if first_terminal:
        _remove(state, "remote_outstanding", request)
    return first_terminal


def _record_binding_success(state: Object, request: str, association: str, outcome: str = "retrieved") -> None:
    first_terminal = request not in _object(state["terminals"])
    if first_terminal:
        outcomes: Object = {association: outcome}
        _request_fact(state, request, {"condition": "result", "outcomes": outcomes})
        _object(state["association_terminals"])[association] = {
            "outcome": outcome,
            "policy": _object(state["request_policies"])[request],
            "request": request,
        }
    else:
        fact = _object(_object(state["request_facts"]).get(request, {}))
        if fact.get("condition") != "result" or fact.get("outcomes") != {association: outcome}:
            _unique(_array(state["defects"]), "conflicting_terminal")
    _terminal(state, request, "success")
    _remove(state, "local_in_flight", request)
    if first_terminal:
        _remove(state, "remote_outstanding", request)


def _record_late_terminal_conflict(state: Object, request: str, *, association: str, failure: str | None) -> None:
    results: list[Json] = []
    category = "failure" if failure is not None else "success"
    if failure is None:
        results.append(
            {
                "association": association,
                "consumed_context_ports": [],
                "outcome": "retrieved",
                "outputs": [],
            }
        )
    facts = state.setdefault("conflicting_terminal_facts", [])
    _array(facts).append(
        {
            "association": None,
            "code": "conflicting_terminal",
            "request": request,
            "settlement": None,
            "terminal": {
                "category": category,
                "failure": failure,
                "request": request,
                "results": results,
            },
        }
    )
