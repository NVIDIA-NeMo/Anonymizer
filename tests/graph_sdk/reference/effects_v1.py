# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Finite independent reference for request, binding, and lifecycle effects."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TypeAlias, cast

Json: TypeAlias = str | int | bool | None | list["Json"] | dict[str, "Json"]
Object: TypeAlias = dict[str, Json]
CONTRACT_SHA256 = "9b0ab07b8c0212ffd954dc37eb37540141da6899753fc778ad26238494aeaeca"
GENERATOR_VERSION = "effects-v1-generator-1"
SELF_TEST_VERSION = "effects-v1-self-test-1"
CORPUS_PATH = "tests/graph_sdk/reference/effects_v1_cases.json"
FAMILIES = (
    "budgets",
    "keyed",
    "retry",
    "races",
    "inflight",
    "binding",
    "resources",
    "bridges",
    "decisions",
    "admission",
)
FAILURE_CLASSES = (
    "rejected_before_acceptance",
    "retryable",
    "malformed_response",
    "permanent",
    "transport_unknown",
    "implementation_exception",
)
RUNTIME_CONDITIONS = (
    "result",
    "failure",
    "cancel_before_start",
    "cancel_after_start",
    "cancel_after_dispatch",
    "lost",
    "request_inconsistent",
    "budget_exhausted",
    "request_limit_exhausted",
    "artifact_limit_exhausted",
    "deadline_exhausted",
)


def _object(value: Json) -> Object:
    if not isinstance(value, dict):
        raise TypeError(f"Expected object, got {type(value)!r}")
    return value


def _array(value: Json) -> list[Json]:
    if not isinstance(value, list):
        raise TypeError(f"Expected array, got {type(value)!r}")
    return value


def _strings(value: Json) -> list[str]:
    return [cast(str, item) for item in _array(value)]


def canonical_bytes(cases: Iterable[Object]) -> bytes:
    return (json.dumps(tuple(cases), indent=2, sort_keys=True) + "\n").encode()


def load_cases(value: object) -> tuple[Object, ...]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise TypeError("effects corpus must be a list of objects")
    return tuple(cast(Object, item) for item in value)


def _initial(declaration: Object) -> Object:
    return {
        "artifacts": [],
        "attempts": {},
        "binding_sources": {},
        "binding_terminal": None,
        "bindings": {},
        "cancel_requested": [],
        "cleanup": {},
        "closed_unstarted": {},
        "decision_defects": [],
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
        "reservations": {},
        "resource_count": 0,
        "resources": {},
        "settlements": {},
        "tasks": {},
        "terminals": {},
    }


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


def _remove(state: Object, key: str, value: str) -> None:
    values = _strings(state[key])
    if value in values:
        values.remove(value)
        state[key] = values


def _advance(state: Object, declaration: Object, event: Object) -> Object | None:
    kind = cast(str, event.get("kind"))
    reservations = _object(state["reservations"])
    dispatched = _strings(state["dispatched"])
    bindings = _object(state["bindings"])
    attempts = _object(state["attempts"])
    if kind == "bind_policy":
        association = cast(str, event["association"])
        if association in bindings:
            return _reject("duplicate_binding")
        bindings[association] = event["policies"]
    elif kind == "reserve":
        request, policy = cast(str, event["request"]), cast(str, event["policy"])
        associations = _strings(event["associations"])
        if request in reservations or request in dispatched:
            return _reject("duplicate_request")
        if policy not in _object(state["policies"]):
            return _reject("unknown_policy")
        if any(policy not in _strings(bindings.get(item, [])) for item in associations):
            return _reject("cross_policy")
        maximum = cast(int, _object(_object(state["policies"])[policy])["max_attempts"])
        if event.get("purpose") == "retry":
            replay = _object(_object(state["policies"])[policy])["replay"]
            failures = _object(state["request_failures"])
            prior = next((cast(str, failures[key]) for key in reversed(list(failures))), None)
            permitted = prior == "rejected_before_acceptance" and replay in ("before_acceptance", "idempotent")
            permitted = permitted or prior in ("retryable", "transport_unknown") and replay == "idempotent"
            if not permitted:
                return _reject("replay_forbidden")
        eligible: list[str] = []
        for item in associations:
            if cast(int, attempts.get(item, 0)) >= maximum:
                _object(state["denials"])[item] = "request_limit_stopped"
            else:
                eligible.append(item)
        limit = declaration.get("hard_limit")
        if (
            eligible
            and limit is not None
            and cast(int, state["dispatched_count"]) + len(reservations) >= cast(int, limit)
        ):
            for item in eligible:
                _object(state["denials"])[item] = "budget_stopped"
            eligible = []
        if eligible:
            reservations[request] = eligible
    elif kind == "dispatch":
        request = cast(str, event["request"])
        if request not in reservations:
            return _reject("missing_reservation")
        associations = _strings(reservations.pop(request))
        _object(state["request_associations"])[request] = associations
        dispatched.append(request)
        state["dispatched"] = sorted(set(dispatched))
        state["dispatched_count"] = cast(int, state["dispatched_count"]) + 1
        for item in associations:
            attempts[item] = cast(int, attempts.get(item, 0)) + 1
        for key in ("local_in_flight", "remote_outstanding"):
            values = _strings(state[key])
            _unique(cast(list[Json], values), request)
            state[key] = values
    elif kind == "result":
        request = cast(str, event["request"])
        if request not in dispatched:
            return _reject("not_dispatched")
        expected = set(_strings(_object(state["request_associations"])[request]))
        returned = _strings(event["returned"])
        seen: set[str] = set()
        defects: list[str] = []
        for item in returned:
            if item in seen:
                defects.append("duplicate_keyed_result")
            elif item not in expected:
                defects.append("foreign_keyed_result" if item.startswith("X") else "extra_keyed_result")
            seen.add(item)
        if expected - seen:
            defects.append("missing_keyed_result")
        for defect in defects:
            _unique(_array(state["defects"]), defect)
        _terminal(state, request, "inconsistent" if defects else "success")
        _remove(state, "local_in_flight", request)
        _remove(state, "remote_outstanding", request)
    elif kind == "failure":
        request = cast(str, event["request"])
        if request not in dispatched:
            return _reject("not_dispatched")
        _object(state["request_failures"])[request] = event["failure"]
        _terminal(state, request, "failure")
        _remove(state, "local_in_flight", request)
        _remove(state, "remote_outstanding", request)
    elif kind == "cancel":
        request = cast(str, event["request"])
        if request in _object(state["terminals"]):
            pass
        elif request in reservations:
            reservations.pop(request)
            _terminal(state, request, "cancelled")
        elif request in dispatched:
            values = _strings(state["cancel_requested"])
            _unique(cast(list[Json], values), request)
            state["cancel_requested"] = values
        else:
            return _reject("unknown_request")
    elif kind == "scope_cancel":
        for request in tuple(reservations):
            reservations.pop(request)
            _terminal(state, request, "cancelled")
        state["cancel_requested"] = sorted(set(_strings(state["cancel_requested"])) | set(dispatched))
    elif kind == "stop":
        request = cast(str, event["request"])
        if request not in _strings(state["cancel_requested"]):
            return _reject("cancel_not_requested")
        _terminal(state, request, "cancelled")
        _remove(state, "local_in_flight", request)
        _remove(state, "remote_outstanding", request)
    elif kind == "lost":
        request = cast(str, event["request"])
        if request not in dispatched:
            return _reject("not_dispatched")
        _terminal(state, request, "lost")
        _remove(state, "local_in_flight", request)
    elif kind == "settlement":
        request = cast(str, event["request"])
        settlements = _object(state["settlements"])
        value: Object = {key: event[key] for key in ("disposition", "usage", "remote_stopped")}
        if request in settlements and settlements[request] != value:
            _unique(_array(state["defects"]), "conflicting_settlement")
        else:
            settlements[request] = value
        if event["remote_stopped"] is True:
            _remove(state, "remote_outstanding", request)
    elif kind == "source_result":
        identity, source = cast(str, event["identity"]), cast(str, event["source"])
        items = _array(event["items"])
        limits = _object(declaration.get("binding_limits", {}))
        byte_count = sum(len(cast(str, _object(item)["text"]).encode()) for item in items)
        if len(items) > cast(int, limits.get("max_items", len(items))) or byte_count > cast(
            int, limits.get("max_bytes", byte_count)
        ):
            _object(state["binding_sources"])[identity] = "oversize"
            state["binding_terminal"] = "failed"
            return None
        for raw in items:
            item = _object(raw)
            artifact: Object = {
                "identity": f"{identity}:{item['key']}:{item['version']}",
                "source": source,
                "text": item["text"],
            }
            if artifact in _array(state["artifacts"]):
                _unique(_array(state["defects"]), "duplicate_keyed_result")
            else:
                _array(state["artifacts"]).append(artifact)
        _object(state["binding_sources"])[identity] = "bound"
    elif kind == "source_failure":
        identity = cast(str, event["identity"])
        _object(state["binding_sources"])[identity] = event.get("terminal", "failed")
        state["binding_terminal"] = "failed" if event.get("required", True) else "partial"
    elif kind == "binding_finish":
        if state["binding_terminal"] is None:
            state["binding_terminal"] = "success"
    elif kind == "resource":
        resource = cast(str, event["resource"])
        resources = _object(state["resources"])
        if resource not in resources:
            state["resource_count"] = cast(int, state["resource_count"]) + 1
        resources[resource] = {"owner": event["owner"], "safe_detachment": event["safe_detachment"]}
    elif kind == "close_resource":
        resource = cast(str, event["resource"])
        value = _object(_object(state["resources"])[resource])
        cleanup = _object(state["cleanup"])
        if resource in cleanup:
            return _reject("duplicate_cleanup")
        if value["owner"] == "caller":
            cleanup[resource] = "left_open"
        elif _array(state["local_in_flight"]):
            pass
        elif _array(state["remote_outstanding"]) and value["safe_detachment"] != "independent_after_dispatch":
            pass
        else:
            cleanup[resource] = event.get("disposition", "closed")
    elif kind == "bridge_start":
        task = cast(str, event["task"])
        if task in _object(state["tasks"]):
            return _reject("duplicate_task")
        _object(state["tasks"])[task] = "running"
    elif kind == "bridge_close_unstarted":
        category = cast(str, event["category"])
        if category not in ("blocked", "inconsistent"):
            return _reject("invalid_category")
        _object(state["closed_unstarted"])[cast(str, event["activation"])] = category
    elif kind == "bridge_condition":
        task = cast(str, event["task"])
        if _object(state["tasks"]).get(task) != "running":
            return _reject("task_not_running")
        _object(state["tasks"])[task] = f"pending:{event['outcome']}:{event['category']}"
    elif kind == "bridge_emit":
        task = cast(str, event["task"])
        expected = f"pending:{event['outcome']}:{event['category']}"
        if _object(state["tasks"]).get(task) != expected:
            return _reject("wrong_bridge_emit")
        _object(state["tasks"])[task] = event["category"]
    elif kind == "decision_open":
        wait = cast(str, event["wait"])
        decisions = _object(state["decisions"])
        if wait in decisions:
            return _reject("duplicate_wait")
        active = sum(1 for value in decisions.values() if _object(value).get("closed") is not True)
        if active >= cast(int, declaration.get("max_pending", active + 1)):
            return _reject("pending_limit")
        decisions[wait] = {key: event[key] for key in ("task", "workflow", "artifact", "allowed")}
    elif kind == "decision_submit":
        wait = cast(str, event["wait"])
        decisions = _object(state["decisions"])
        if wait not in decisions:
            _unique(_array(state["decision_defects"]), "foreign_wait")
        else:
            current = _object(decisions[wait])
            if current.get("closed") is True:
                _unique(_array(state["decision_defects"]), "late_or_duplicate")
            elif event["workflow"] != current["workflow"]:
                _unique(_array(state["decision_defects"]), "foreign_workflow")
            elif event["artifact"] != current["artifact"]:
                _unique(_array(state["decision_defects"]), "stale_artifact")
            elif event["decision"] not in _strings(current["allowed"]):
                _unique(_array(state["decision_defects"]), "unknown_decision")
            else:
                current["closed"] = True
                _object(state["tasks"])[cast(str, current["task"])] = "success"
    elif kind == "decision_deadline":
        current = _object(_object(state["decisions"])[cast(str, event["wait"])])
        current["closed"] = True
        _object(state["tasks"])[cast(str, current["task"])] = "failure"
    else:
        return _reject("unknown_event")
    return None


def reduce_trace(declaration: Object, events: Sequence[Object]) -> Object:
    state = _initial(declaration)
    for event in events:
        rejected = _advance(state, declaration, event)
        if rejected is not None:
            return rejected
    return {"state": state, "status": "accepted"}


def admit(declaration: Object) -> Object:
    if declaration.get("admission_error") is not None:
        return _reject(cast(str, declaration["admission_error"]))
    for raw in _array(declaration.get("execution_policies", [])):
        policy = _object(raw)
        implementations = _array(policy["implementations"])
        if policy["kind"] in ("local", "decision") and len(implementations) != 1:
            return _reject("implementation_count")
        if policy.get("retry_owner") == "implementation":
            return _reject("retry_owner")
        if any(_object(item).get("physical_policy") != policy.get("physical_policy") for item in implementations):
            return _reject("changed_failover_policy")
        if cast(int, policy.get("max_capabilities", len(implementations))) < len(implementations):
            return _reject("capability_limit")
    return {"status": "accepted"}


def evaluate_case(case: Mapping[str, Json]) -> Object:
    declaration = _object(case["declaration"])
    return (
        admit(declaration)
        if case["boundary"] == "admission"
        else reduce_trace(declaration, [_object(value) for value in _array(case["events"])])
    )


def _policy(max_attempts: int = 2, replay: str = "idempotent") -> Object:
    return {"max_attempts": max_attempts, "replay": replay, "retry_owner": "executor"}


def _decl(limit: int | None = 2, attempts: int = 2) -> Object:
    return {"hard_limit": limit, "policies": {"P0": _policy(attempts), "P1": _policy(3)}}


def _bind(item: str, *policies: str) -> Object:
    return {"association": item, "kind": "bind_policy", "policies": list(policies)}


def _reserve(request: str, items: Sequence[str], policy: str = "P0", purpose: str = "initial") -> Object:
    return {"associations": list(items), "kind": "reserve", "policy": policy, "purpose": purpose, "request": request}


def _dispatch(request: str) -> Object:
    return {"kind": "dispatch", "request": request}


def _result(request: str, expected: Sequence[str], returned: Sequence[str]) -> Object:
    del expected
    return {"kind": "result", "request": request, "returned": list(returned)}


def _trace(items: Sequence[str] = ("T0",), request: str = "R0") -> list[Object]:
    return [*[_bind(item, "P0") for item in items], _reserve(request, items), _dispatch(request)]


def _case(
    family: str,
    name: str,
    declaration: Object,
    events: list[Object],
    boundary: str = "runtime",
    traces: list[list[Object]] | None = None,
) -> Object:
    case: Object = {
        "boundary": boundary,
        "case_id": f"{family}/{name}",
        "declaration": declaration,
        "events": events,
        "family": family,
        "expected": {},
        "traces": [],
    }
    case["expected"] = evaluate_case(case)
    case["traces"] = [
        {"events": trace, "expected": reduce_trace(declaration, trace), "name": f"alternate_{index}"}
        for index, trace in enumerate(traces or [])
    ]
    return case


def _generate_specs() -> tuple[Object, ...]:
    c: list[Object] = []
    for limit in (0, 1, 2):
        events = [_bind("T0", "P0"), _reserve("R0", ["T0"])] + ([_dispatch("R0")] if limit else [])
        c.append(_case("budgets", f"limit_{limit}", _decl(limit), events))
    c += [
        _case("budgets", "partial_two", _decl(1), _trace() + [_bind("T1", "P0"), _reserve("R1", ["T1"])]),
        _case(
            "budgets", "exact_two", _decl(2), _trace() + [_bind("T1", "P0"), _reserve("R1", ["T1"]), _dispatch("R1")]
        ),
        _case(
            "budgets",
            "cancel_reserved_zero_charge",
            _decl(),
            [_bind("T0", "P0"), _reserve("R0", ["T0"]), {"kind": "cancel", "request": "R0"}],
        ),
    ]
    shared = _trace(("T0", "T1"))
    c.append(_case("keyed", "valid_shared_reordered", _decl(), shared + [_result("R0", ("T0", "T1"), ("T1", "T0"))]))
    for name, returned in (
        ("missing", ("T0",)),
        ("duplicate", ("T0", "T0", "T1")),
        ("extra", ("T0", "T1", "T2")),
        ("foreign", ("T0", "T1", "X0")),
    ):
        c.append(_case("keyed", name, _decl(), shared + [_result("R0", ("T0", "T1"), returned)]))
    c += [
        _case("keyed", "single_t0", _decl(), _trace() + [_result("R0", ("T0",), ("T0",))]),
        _case("keyed", "single_t1", _decl(), _trace(("T1",)) + [_result("R0", ("T1",), ("T1",))]),
        _case("keyed", "parent_summary_no_charge", _decl(), shared + [_result("R0", ("T0", "T1"), ("T0", "T1"))]),
    ]
    c += [
        _case("retry", "cross_policy_semantic", _decl(), [_bind("T0", "P0"), _reserve("R0", ["T0"], "P1")]),
        _case("retry", "cross_policy_binding", _decl(), [_bind("D0", "P0"), _reserve("R0", ["D0"], "P1")]),
    ]
    variants = (
        ("retry_success", "retryable", "retry"),
        ("correction", "malformed_response", "correction"),
        ("failover", "permanent", "failover"),
    )
    for name, failure, purpose in variants:
        c.append(
            _case(
                "retry",
                name,
                _decl(),
                _trace()
                + [
                    {"failure": failure, "kind": "failure", "request": "R0"},
                    _reserve("R1", ["T0"], purpose=purpose),
                    _dispatch("R1"),
                ],
            )
        )
    c.append(
        _case(
            "retry",
            "repair_new_task",
            _decl(),
            _trace()
            + [
                {"failure": "permanent", "kind": "failure", "request": "R0"},
                _bind("T1", "P0"),
                _reserve("R1", ["T1"], purpose="repair"),
                _dispatch("R1"),
            ],
        )
    )
    c.append(
        _case(
            "retry",
            "mixed_exhausted_shared",
            _decl(attempts=1),
            _trace()
            + [
                {"failure": "retryable", "kind": "failure", "request": "R0"},
                _bind("T1", "P0"),
                _reserve("R1", ["T0", "T1"], purpose="retry"),
                _dispatch("R1"),
            ],
        )
    )
    for replay, failure in (
        ("never", "retryable"),
        ("before_acceptance", "rejected_before_acceptance"),
        ("idempotent", "transport_unknown"),
    ):
        declaration = _decl()
        _object(declaration["policies"])["P0"] = _policy(2, replay)
        c.append(
            _case(
                "retry",
                f"replay_{replay}_{failure}",
                declaration,
                _trace()
                + [
                    {"failure": failure, "kind": "failure", "request": "R0"},
                    _reserve("R1", ["T0"], purpose="retry"),
                    _dispatch("R1"),
                ],
            )
        )
    c += [
        _case(
            "retry",
            "budget_denies_retry",
            _decl(1),
            _trace()
            + [{"failure": "retryable", "kind": "failure", "request": "R0"}, _reserve("R1", ["T0"], purpose="retry")],
        ),
        _case("retry", "implementation_owner_rejected", {"admission_error": "retry_owner"}, [], "admission"),
    ]
    base = _trace()
    settlement: Object = {
        "disposition": "completed",
        "kind": "settlement",
        "remote_stopped": True,
        "request": "R0",
        "usage": {"input": 1, "output": 1},
    }
    c += [
        _case("races", "success", _decl(), base + [_result("R0", ("T0",), ("T0",))]),
        _case("races", "failure", _decl(), base + [{"failure": "permanent", "kind": "failure", "request": "R0"}]),
        _case(
            "races",
            "cancel_trusted_stop",
            _decl(),
            base + [{"kind": "cancel", "request": "R0"}, {"kind": "stop", "request": "R0"}],
        ),
        _case(
            "races",
            "cancel_unknown_lost",
            _decl(),
            base + [{"kind": "cancel", "request": "R0"}, {"kind": "lost", "request": "R0"}],
        ),
        _case(
            "races",
            "cancel_then_result",
            _decl(),
            base + [{"kind": "cancel", "request": "R0"}, _result("R0", ("T0",), ("T0",))],
        ),
        _case(
            "races",
            "result_then_cancel",
            _decl(),
            base + [_result("R0", ("T0",), ("T0",)), {"kind": "cancel", "request": "R0"}],
        ),
        _case(
            "races",
            "lost_late_result_settlement",
            _decl(),
            base + [{"kind": "lost", "request": "R0"}, _result("R0", ("T0",), ("T0",)), settlement],
        ),
        _case(
            "races",
            "settlement_before_result",
            _decl(),
            base + [settlement, _result("R0", ("T0",), ("T0",))],
            traces=[base + [_result("R0", ("T0",), ("T0",)), settlement]],
        ),
        _case(
            "races",
            "failure_then_settlement",
            _decl(),
            base + [{"failure": "permanent", "kind": "failure", "request": "R0"}, settlement],
        ),
        _case(
            "races",
            "settlement_then_failure",
            _decl(),
            base + [settlement, {"failure": "permanent", "kind": "failure", "request": "R0"}],
        ),
        _case(
            "races",
            "lost_unknown_settlement",
            _decl(),
            base + [{"kind": "lost", "request": "R0"}, dict(settlement, remote_stopped=None, usage="unknown")],
        ),
        _case(
            "races",
            "cancel_stop_late_result",
            _decl(),
            base
            + [{"kind": "cancel", "request": "R0"}, {"kind": "stop", "request": "R0"}, _result("R0", ("T0",), ("T0",))],
        ),
        _case(
            "races",
            "identical_terminal",
            _decl(),
            base + [_result("R0", ("T0",), ("T0",)), _result("R0", ("T0",), ("T0",))],
        ),
        _case("races", "identical_settlement", _decl(), base + [settlement, settlement]),
        _case(
            "races",
            "conflicting_terminal",
            _decl(),
            base + [_result("R0", ("T0",), ("T0",)), {"failure": "permanent", "kind": "failure", "request": "R0"}],
        ),
        _case(
            "races", "conflicting_settlement", _decl(), base + [settlement, dict(settlement, disposition="rejected")]
        ),
        _case("races", "scope_cancel", _decl(), [_bind("T0", "P0"), _reserve("R0", ["T0"]), {"kind": "scope_cancel"}]),
    ]
    c += [
        _case("inflight", "dispatch_sets_both", _decl(), base),
        _case("inflight", "lost_keeps_remote", _decl(), base + [{"kind": "lost", "request": "R0"}]),
        _case(
            "inflight",
            "trusted_settlement_clears_remote",
            _decl(),
            base + [{"kind": "lost", "request": "R0"}, settlement],
        ),
        _case(
            "inflight",
            "unknown_settlement_keeps_remote",
            _decl(),
            base + [{"kind": "lost", "request": "R0"}, dict(settlement, remote_stopped=None, usage="unknown")],
        ),
        _case("inflight", "result_clears_both", _decl(), base + [_result("R0", ("T0",), ("T0",))]),
    ]

    def binding(identity: str, source: str, request: str, text: str) -> list[Object]:
        return [
            _bind(identity, "P0"),
            _reserve(request, [identity], purpose="initial_binding"),
            _dispatch(request),
            {
                "identity": identity,
                "items": [{"key": 0, "text": text, "version": 1}],
                "kind": "source_result",
                "source": source,
            },
        ]

    c += [
        _case(
            "binding",
            "two_sources_same_key",
            _decl(),
            binding("D0", "S0", "R0", "a") + binding("D1", "S1", "R1", "b") + [{"kind": "binding_finish"}],
        ),
        _case(
            "binding",
            "one_source_two_declarations",
            _decl(),
            [
                {"kind": "resource", "owner": "sdk", "resource": "Q0", "safe_detachment": "forbidden"},
                *binding("D0", "S0", "R0", "a"),
                *binding("D1", "S0", "R1", "b"),
                {"kind": "binding_finish"},
            ],
        ),
        _case(
            "binding",
            "required_failure_preserves_prior",
            _decl(),
            binding("D0", "S0", "R0", "a")
            + [
                _bind("D1", "P0"),
                _reserve("R1", ["D1"], purpose="initial_binding"),
                _dispatch("R1"),
                {"identity": "D1", "kind": "source_failure", "required": True},
            ],
        ),
        _case(
            "binding",
            "optional_failure_partial",
            _decl(),
            binding("D0", "S0", "R0", "a") + [{"identity": "D1", "kind": "source_failure", "required": False}],
        ),
        _case(
            "binding",
            "omitted_optional",
            _decl(),
            [
                {"identity": "D0", "kind": "source_failure", "required": False, "terminal": "omitted_optional"},
                {"kind": "binding_finish"},
            ],
        ),
        _case(
            "binding",
            "response_reordered",
            _decl(),
            binding("D1", "S1", "R1", "b") + binding("D0", "S0", "R0", "a") + [{"kind": "binding_finish"}],
        ),
        _case(
            "binding",
            "oversize_no_truncation",
            _decl(),
            [{"identity": "D0", "kind": "source_failure", "required": True, "terminal": "oversize"}],
        ),
        _case(
            "binding",
            "cancel_before_dispatch",
            _decl(),
            [_bind("D0", "P0"), _reserve("R0", ["D0"], purpose="initial_binding"), {"kind": "cancel", "request": "R0"}],
        ),
        _case(
            "binding",
            "cancel_after_dispatch_lost",
            _decl(),
            binding("D0", "S0", "R0", "a")[:3]
            + [{"kind": "cancel", "request": "R0"}, {"kind": "lost", "request": "R0"}],
        ),
        _case("binding", "invalid_local_zero_effects", {"admission_error": "foreign_target"}, [], "admission"),
    ]
    exact_bounds = _decl()
    exact_bounds["binding_limits"] = {"max_bytes": 2, "max_items": 2}
    one_byte = _decl()
    one_byte["binding_limits"] = {"max_bytes": 1, "max_items": 2}
    one_item = _decl()
    one_item["binding_limits"] = {"max_bytes": 2, "max_items": 1}
    two_items: Object = {
        "identity": "D0",
        "items": [{"key": 0, "text": "a", "version": 1}, {"key": 1, "text": "b", "version": 1}],
        "kind": "source_result",
        "source": "S0",
    }
    c += [
        _case(
            "binding",
            "exact_item_byte_bounds",
            exact_bounds,
            [_bind("D0", "P0"), _reserve("R0", ["D0"]), _dispatch("R0"), two_items],
        ),
        _case(
            "binding",
            "one_over_byte_bound",
            one_byte,
            [_bind("D0", "P0"), _reserve("R0", ["D0"]), _dispatch("R0"), two_items],
        ),
        _case(
            "binding",
            "one_over_item_bound",
            one_item,
            [_bind("D0", "P0"), _reserve("R0", ["D0"]), _dispatch("R0"), two_items],
        ),
    ]
    c += [
        _case(
            "resources",
            "caller_left_open",
            {},
            [
                {"kind": "resource", "owner": "caller", "resource": "Q0", "safe_detachment": "forbidden"},
                {"kind": "close_resource", "resource": "Q0"},
            ],
        ),
        _case(
            "resources",
            "sdk_closed",
            {},
            [
                {"kind": "resource", "owner": "sdk", "resource": "Q0", "safe_detachment": "forbidden"},
                {"kind": "close_resource", "resource": "Q0"},
            ],
        ),
        _case(
            "resources",
            "sdk_close_failed",
            {},
            [
                {"kind": "resource", "owner": "sdk", "resource": "Q0", "safe_detachment": "forbidden"},
                {"disposition": "close_failed", "kind": "close_resource", "resource": "Q0"},
            ],
        ),
        _case(
            "resources",
            "sdk_close_unknown",
            {},
            [
                {"kind": "resource", "owner": "sdk", "resource": "Q0", "safe_detachment": "forbidden"},
                {"disposition": "close_unknown", "kind": "close_resource", "resource": "Q0"},
            ],
        ),
    ]
    for name, detach, terminal in (
        ("sdk_remote_waits", "forbidden", "lost"),
        ("sdk_safe_detach", "independent_after_dispatch", "lost"),
        ("local_inflight_waits", "independent_after_dispatch", None),
        ("trusted_stop_then_close", "forbidden", "stop"),
    ):
        events: list[Object] = [
            {"kind": "resource", "owner": "sdk", "resource": "Q0", "safe_detachment": detach},
            *base,
        ]
        if terminal == "lost":
            events.append({"kind": "lost", "request": "R0"})
        elif terminal == "stop":
            events += [{"kind": "cancel", "request": "R0"}, {"kind": "stop", "request": "R0"}]
        events.append({"kind": "close_resource", "resource": "Q0"})
        c.append(_case("resources", name, _decl(), events))
    c.append(
        _case(
            "bridges",
            "cancel_before_start",
            {},
            [{"activation": "A0", "category": "blocked", "kind": "bridge_close_unstarted"}],
        )
    )
    for condition, outcome, category in (
        ("result", "ok", "success"),
        ("cancel_after_start", None, "cancelled"),
        ("cancel_after_dispatch", None, "cancelled"),
        ("lost", None, "lost"),
        ("request_inconsistent", None, "inconsistent"),
        ("budget_exhausted", None, "blocked"),
        ("request_limit_exhausted", None, "blocked"),
        ("artifact_limit_exhausted", None, "blocked"),
        ("deadline_exhausted", None, "failure"),
    ):
        c.append(
            _case(
                "bridges",
                condition,
                {},
                [
                    {"kind": "bridge_start", "node": "N0", "task": "T0"},
                    {
                        "category": category,
                        "condition": condition,
                        "kind": "bridge_condition",
                        "outcome": outcome,
                        "task": "T0",
                    },
                    {"category": category, "kind": "bridge_emit", "outcome": outcome, "task": "T0"},
                ],
            )
        )
    for failure in FAILURE_CLASSES:
        c.append(
            _case(
                "bridges",
                f"failure_{failure}",
                {},
                [
                    {"kind": "bridge_start", "node": "N0", "task": "T0"},
                    {
                        "category": "failure",
                        "condition": "failure",
                        "failure": failure,
                        "kind": "bridge_condition",
                        "outcome": None,
                        "task": "T0",
                    },
                    {"category": "failure", "kind": "bridge_emit", "outcome": None, "task": "T0"},
                ],
            )
        )
    opened: list[Object] = [
        {"kind": "bridge_start", "node": "N0", "task": "T0"},
        {
            "allowed": ["approve", "reject"],
            "artifact": "V0",
            "kind": "decision_open",
            "task": "T0",
            "wait": "W0",
            "workflow": "F0",
        },
    ]
    c.append(
        _case(
            "decisions",
            "matching_resume_unrelated_advances",
            {},
            [
                *opened,
                {"kind": "bridge_start", "node": "N1", "task": "T1"},
                {
                    "category": "success",
                    "condition": "result",
                    "kind": "bridge_condition",
                    "outcome": "ok",
                    "task": "T1",
                },
                {"category": "success", "kind": "bridge_emit", "outcome": "ok", "task": "T1"},
                {"artifact": "V0", "decision": "approve", "kind": "decision_submit", "wait": "W0", "workflow": "F0"},
            ],
        )
    )
    for name, change in (
        ("stale_artifact", {"artifact": "V1"}),
        ("foreign_workflow", {"workflow": "F1"}),
        ("foreign_wait", {"wait": "W1"}),
        ("unknown_decision", {"decision": "maybe"}),
    ):
        submit: Object = {
            "artifact": "V0",
            "decision": "approve",
            "kind": "decision_submit",
            "wait": "W0",
            "workflow": "F0",
        }
        submit.update(change)
        c.append(_case("decisions", name, {}, [*opened, submit]))
    ok: Object = {"artifact": "V0", "decision": "approve", "kind": "decision_submit", "wait": "W0", "workflow": "F0"}
    c += [
        _case("decisions", "duplicate_response", {}, [*opened, ok, ok]),
        _case("decisions", "deadline", {}, [*opened, {"kind": "decision_deadline", "wait": "W0"}]),
        _case(
            "decisions",
            "cancel_condition",
            {},
            [
                {"kind": "bridge_start", "node": "N0", "task": "T0"},
                {
                    "category": "cancelled",
                    "condition": "cancel_after_start",
                    "kind": "bridge_condition",
                    "outcome": None,
                    "task": "T0",
                },
                {"category": "cancelled", "kind": "bridge_emit", "outcome": None, "task": "T0"},
            ],
        ),
        _case(
            "decisions",
            "implementation_failure",
            {},
            [
                {"kind": "bridge_start", "node": "N0", "task": "T0"},
                {
                    "category": "failure",
                    "condition": "failure",
                    "kind": "bridge_condition",
                    "outcome": None,
                    "task": "T0",
                },
                {"category": "failure", "kind": "bridge_emit", "outcome": None, "task": "T0"},
            ],
        ),
    ]
    open1: Object = {
        "allowed": ["approve"],
        "artifact": "V1",
        "kind": "decision_open",
        "task": "T1",
        "wait": "W1",
        "workflow": "F0",
    }
    c += [
        _case(
            "decisions",
            "pending_exact",
            {"max_pending": 2},
            [*opened, {"kind": "bridge_start", "node": "N1", "task": "T1"}, open1],
        ),
        _case(
            "decisions",
            "pending_one_over",
            {"max_pending": 1},
            [*opened, {"kind": "bridge_start", "node": "N1", "task": "T1"}, open1],
        ),
    ]
    for error in (
        "outer_type",
        "aggregate_limit",
        "member_type",
        "invalid_value",
        "foreign_owner",
        "duplicate",
        "missing",
        "unsupported",
        "contradictory",
        "runtime_mapping_missing",
        "wrong_category",
        "unknown_outcome",
    ):
        c.append(_case("admission", error, {"admission_error": error}, [], "admission"))
    c += [
        _case(
            "admission",
            "valid_external_failover",
            {
                "execution_policies": [
                    {
                        "implementations": [{"physical_policy": "P0"}, {"physical_policy": "P0"}],
                        "kind": "external",
                        "max_capabilities": 2,
                        "physical_policy": "P0",
                        "retry_owner": "executor",
                    }
                ]
            },
            [],
            "admission",
        ),
        _case(
            "admission",
            "changed_failover_policy",
            {
                "execution_policies": [
                    {
                        "implementations": [{"physical_policy": "P0"}, {"physical_policy": "P1"}],
                        "kind": "external",
                        "max_capabilities": 2,
                        "physical_policy": "P0",
                        "retry_owner": "executor",
                    }
                ]
            },
            [],
            "admission",
        ),
        _case(
            "admission",
            "local_multiple_implementations",
            {
                "execution_policies": [
                    {
                        "implementations": [{"physical_policy": None}, {"physical_policy": None}],
                        "kind": "local",
                        "max_capabilities": 2,
                        "physical_policy": None,
                        "retry_owner": "none",
                    }
                ]
            },
            [],
            "admission",
        ),
        _case(
            "admission",
            "capability_one_over",
            {
                "execution_policies": [
                    {
                        "implementations": [{"physical_policy": "P0"}, {"physical_policy": "P0"}],
                        "kind": "external",
                        "max_capabilities": 1,
                        "physical_policy": "P0",
                        "retry_owner": "executor",
                    }
                ]
            },
            [],
            "admission",
        ),
    ]
    return tuple(c)


_CASES = _generate_specs()
FAMILY_COUNTS = Counter(cast(str, case["family"]) for case in _CASES)


def generate_cases() -> tuple[Object, ...]:
    return _generate_specs()


def case_by_id(case_id: str) -> Object:
    return next(case for case in _CASES if case["case_id"] == case_id)


def trace_count(cases: Sequence[Object]) -> int:
    return sum(1 + len(_array(case["traces"])) for case in cases)


def event_count(cases: Sequence[Object]) -> int:
    return sum(
        len(_array(case["events"])) + sum(len(_array(_object(trace)["events"])) for trace in _array(case["traces"]))
        for case in cases
    )


def build_manifest(cases: Sequence[Object]) -> Object:
    corpus = canonical_bytes(cases)
    return {
        "case_count": len(cases),
        "contract_sha256": CONTRACT_SHA256,
        "corpus_path": CORPUS_PATH,
        "corpus_sha256": hashlib.sha256(corpus).hexdigest(),
        "event_count": event_count(cases),
        "family_counts": dict(sorted(FAMILY_COUNTS.items())),
        "generator_version": GENERATOR_VERSION,
        "self_test_version": SELF_TEST_VERSION,
        "trace_count": trace_count(cases),
    }


if __name__ == "__main__":
    here = Path(__file__).parent
    cases = generate_cases()
    (here / "effects_v1_cases.json").write_bytes(canonical_bytes(cases))
    (here / "effects_v1_manifest.json").write_text(json.dumps(build_manifest(cases), indent=2, sort_keys=True) + "\n")
