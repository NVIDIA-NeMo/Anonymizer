# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Finite independent reference for request, binding, and lifecycle effects."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import TypeAlias, cast

Json: TypeAlias = str | int | bool | None | list["Json"] | dict[str, "Json"]
Object: TypeAlias = dict[str, Json]
MappingKey: TypeAlias = tuple[str, str | None, str | None]
CONTRACT_SHA256 = "9b0ab07b8c0212ffd954dc37eb37540141da6899753fc778ad26238494aeaeca"
REQUEST_BOUNDARY_ADDENDUM_SHA256 = "6344f7f1bdbb14a4c9e08f546928d31c89f26d3ed26e0c50362d9537d891c3a5"

GENERATOR_VERSION = "effects-v1-generator-16-caller-cleanup-lifecycle"
SELF_TEST_VERSION = "effects-v1-self-test-16-caller-cleanup-lifecycle"
MATERIALIZATION_ADDENDUM_SHA256 = "b1a5651ee2649b01c89e80bd1442e3f03209292b48ce846d27714698f57eb07c"
BASE_CORPUS_SHA256 = "c62f2cc7e7237ea030451ac8d35a3c30b39f71766b4275348597949d560a7a6a"
PREDECESSOR_CASE_COUNT = 214
PREDECESSOR_CORPUS_SHA256 = "b509a2e1dc697ae5e16c8f2f9652aac74d862e4b449f64958fbba99abda87e93"
OPTIONAL_OMISSION_ADDENDUM_SHA256 = "543d46e66988e11e6386ecdc686a77207b74e42211b8b164bc1decbb96e5d93e"
MAP_EXECUTION_ADDENDUM_SHA256 = "9563fd44c040d546352853de03e57bcc94a6e34b7772ca2d9239f84074bef41d"
BINDING_SUCCESS_ADDENDUM_SHA256 = "c6689c78f712837072235ad8343de8bd8240ea2c7e8580d1594e055de2e017cd"
MATERIALIZED_VERSION_CONTRACT_SHA256 = "165c7c95bce31a7c5808860f28d012bbe1986bf0712cebdc86fb08ad07afcb21"
ACCEPTED_PREDECESSOR_CORPUS_SHA256 = "d56c9c64367ca9f4aa211aeb0e1bc8c5c213c977af07f906eced572764bc7726"
ACCEPTED_PREDECESSOR_CASE_COUNT = 296
CORPUS_PATH = "future-contracts/r2-version-selection-v7/effects_v1_cases.json"
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
    "materialization",
    "map",
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
POLICY_CONDITIONS = {
    "local": (
        "cancel_before_start",
        "cancel_after_start",
        "artifact_limit_exhausted",
        "deadline_exhausted",
    ),
    "external": (
        "cancel_before_start",
        "cancel_after_start",
        "cancel_after_dispatch",
        "lost",
        "request_inconsistent",
        "budget_exhausted",
        "request_limit_exhausted",
        "artifact_limit_exhausted",
        "deadline_exhausted",
    ),
    "decision": (
        "cancel_before_start",
        "cancel_after_start",
        "artifact_limit_exhausted",
        "deadline_exhausted",
    ),
}


def _mapping_key(mapping: Object) -> MappingKey:
    return (
        cast(str, mapping.get("condition")),
        cast(str | None, mapping.get("reported_outcome")),
        cast(str | None, mapping.get("failure")),
    )


def _expected_mapping_keys(kind: str, outcomes: Sequence[str]) -> set[MappingKey]:
    keys: set[MappingKey] = {("result", outcome, None) for outcome in outcomes}
    failures = ("permanent", "implementation_exception") if kind == "decision" else FAILURE_CLASSES
    keys.update(("failure", None, failure) for failure in failures)
    keys.update((condition, None, None) for condition in POLICY_CONDITIONS[kind])
    return keys


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
    if isinstance(raw_specs, list) and any(
        isinstance(raw, dict) and raw.get("version_selection") == "latest" for raw in raw_specs
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


def _natural(value: Json, *, positive: bool = False) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= (1 if positive else 0)


def _materialization_declaration(declaration: Object, event: Object) -> Object | None:
    declarations = declaration.get("materializations", [])
    if not isinstance(declarations, list):
        return None
    identity = tuple(event.get(key) for key in ("path", "target", "node", "port", "declaration"))
    matches = [
        _object(raw)
        for raw in declarations
        if isinstance(raw, dict)
        and tuple(raw.get(key) for key in ("path", "target", "node", "port", "declaration")) == identity
    ]
    return matches[0] if len(matches) == 1 else None


def _requires_materialization(declaration: Object, path: str, association: str) -> bool:
    raw_specs = declaration.get("materializations", [])
    return isinstance(raw_specs, list) and any(
        isinstance(raw, dict) and raw.get("path") == path and raw.get("association") == association for raw in raw_specs
    )


def _materialize(state: Object, declaration: Object, event: Object) -> Object | None:
    spec = _materialization_declaration(declaration, event)
    if spec is None:
        return _reject("missing_materialization")
    raw_items = event.get("items")
    if not isinstance(raw_items, list):
        return _reject("invalid_type")
    limits = _object(declaration.get("materialization_limits", {}))
    binding_limits = _object(declaration.get("binding_limits", {}))
    path = cast(str, spec["path"])
    # Count bounds are checked before member scans.  A rejected collection is atomic.
    initial_item_limit = (
        binding_limits.get("max_items", len(raw_items)) if spec["path"] == "initial" else len(raw_items)
    )
    collection_limit = (
        limits.get("max_collection_items", len(raw_items)) if spec["kind"] == "collection" else len(raw_items)
    )
    if len(raw_items) > min(cast(int, collection_limit), cast(int, initial_item_limit)):
        return _reject("collection_limit_exceeded")
    selection = cast(str, spec.get("version_selection", "exact_one"))
    if spec.get("kind") == "single" and selection == "exact_one" and len(raw_items) != 1:
        return _reject("single_cardinality")
    if spec.get("kind") == "collection" and (not raw_items or len(raw_items) > cast(int, spec["max_items"])):
        return _reject("collection_cardinality")
    typed: list[Object] = []
    pairs: list[tuple[int, int]] = []
    for raw in raw_items:
        if not isinstance(raw, dict) or set(raw) != {"key", "value", "version"}:
            return _reject("invalid_item")
        item = _object(raw)
        if not _natural(item["key"]) or not _natural(item["version"], positive=True):
            return _reject("invalid_item")
        if not isinstance(item["value"], str):
            return _reject("nested_collection" if isinstance(item["value"], (list, dict)) else "wrong_item_type")
        typed.append(item)
        pairs.append((cast(int, item["key"]), cast(int, item["version"])))
    if len(pairs) != len(set(pairs)):
        return _reject("duplicate_item")
    typed.sort(key=lambda item: (cast(int, item["key"]), cast(int, item["version"])))
    if selection == "latest" and (not typed or len({item["key"] for item in typed}) != 1):
        return _reject("latest_key_mismatch")
    byte_count = sum(len(cast(str, item["value"]).encode()) for item in typed)
    initial_byte_limit = binding_limits.get("max_bytes", byte_count) if path == "initial" else byte_count
    if byte_count > min(cast(int, spec["max_bytes"]), cast(int, initial_byte_limit)):
        return _reject("materialization_bytes_exceeded")
    logical_artifacts = (
        len(typed) + 1
        if path == "initial" and spec["kind"] == "collection"
        else len(typed)
        if path == "initial" and selection == "latest"
        else 1
    )
    logical_bytes = byte_count * 2 if path == "initial" and spec["kind"] == "collection" else byte_count
    parents = (
        [
            f"BoundInputKey:{spec['target']}:{spec['node']}:{spec['port']}:{spec['declaration']}:{item['key']}:{item['version']}"
            for item in typed
        ]
        if path == "initial" and spec["kind"] == "collection"
        else _strings(event.get("parents", []))
    )
    current = _object(state.get("materialization", {}))
    if path == "adaptive":
        expected_parents = _strings(spec.get("selector_inputs", []))
        if parents != expected_parents or any(
            parent not in _object(current.get("provenance", {})) for parent in parents
        ):
            return _reject("invalid_provenance")
    elif event.get("parents") != []:
        return _reject("invalid_provenance")
    if selection != "latest":
        if cast(int, current.get("artifact_count", 0)) + logical_artifacts > cast(int, limits["max_artifacts"]):
            return _reject("artifact_count_exceeded")
        if cast(int, current.get("artifact_bytes", 0)) + logical_bytes > cast(int, limits["max_artifact_bytes"]):
            return _reject("artifact_bytes_exceeded")
        if cast(int, current.get("provenance_edges", 0)) + len(parents) > cast(int, limits["max_provenance_edges"]):
            return _reject("provenance_limit_exceeded")
    materialization = current or {
        "artifact_bytes": 0,
        "artifact_count": 0,
        "ports": {},
        "provenance": {},
        "provenance_edges": 0,
    }
    key = (
        f"InitialCollectionKey:{spec['target']}:{spec['node']}:{spec['port']}:{spec['declaration']}"
        if path == "initial" and spec["kind"] == "collection"
        else f"OperationOutputKey:{event.get('activation')}:{spec['target']}:{spec['port']}"
        if path == "adaptive"
        else f"BoundInputKey:{spec['target']}:{spec['node']}:{spec['port']}:{spec['declaration']}:{typed[0]['key']}:{typed[0]['version']}"
    )
    port_key = f"{path}:{spec['target']}:{spec['node']}:{spec['port']}:{spec.get('declaration')}"
    if path == "initial" and selection == "latest":
        lineage = f"{spec['declaration']}:{typed[0]['key']}"
        allocations = _object(state["lineage_allocations"])
        if lineage not in allocations:
            allocations[lineage] = f"K{state['allocator_next']}"
            state["allocator_next"] = cast(int, state["allocator_next"]) + 1
        allocated_key = cast(str, allocations[lineage])
        selected = max(typed, key=lambda item: cast(int, item["version"]))
        selected_ref = f"ArtifactRef:I0:{allocated_key}:{selected['version']}"
        if cast(int, current.get("artifact_count", 0)) + logical_artifacts > cast(int, limits["max_artifacts"]):
            return _reject("artifact_count_exceeded")
        if cast(int, current.get("artifact_bytes", 0)) + logical_bytes > cast(int, limits["max_artifact_bytes"]):
            return _reject("artifact_bytes_exceeded")
        _object(materialization["ports"])[port_key] = {
            "artifact_type": spec["output_type"],
            "key": selected_ref,
            "value": selected["value"],
        }
        for item in typed:
            artifact_ref = f"ArtifactRef:I0:{allocated_key}:{item['version']}"
            producer = (
                f"BoundInputKey:{spec['target']}:{spec['node']}:{spec['port']}:"
                f"{spec['declaration']}:{item['key']}:{item['version']}"
            )
            _object(materialization["provenance"])[producer] = {"artifact": artifact_ref, "parents": []}
            _array(state["artifacts"]).append(
                {
                    "artifact_type": spec["output_type"],
                    "identity": artifact_ref,
                    "source": spec["source"],
                    "text": item["value"],
                }
            )
        materialization["artifact_count"] = cast(int, materialization["artifact_count"]) + logical_artifacts
        materialization["artifact_bytes"] = cast(int, materialization["artifact_bytes"]) + logical_bytes
        state["materialization"] = materialization
        return None
    value: Json = typed[0]["value"] if spec["kind"] == "single" else typed
    _object(materialization["ports"])[port_key] = {"artifact_type": spec["output_type"], "key": key, "value": value}
    _object(materialization["provenance"])[key] = parents
    materialization["artifact_count"] = cast(int, materialization["artifact_count"]) + logical_artifacts
    materialization["artifact_bytes"] = cast(int, materialization["artifact_bytes"]) + logical_bytes
    materialization["provenance_edges"] = cast(int, materialization["provenance_edges"]) + len(parents)
    state["materialization"] = materialization
    if path == "initial" and spec["kind"] == "collection":
        for item, parent in zip(typed, parents, strict=True):
            _object(materialization["provenance"])[parent] = []
            _array(state["artifacts"]).append(
                {
                    "artifact_type": spec["item_type"],
                    "identity": parent,
                    "source": spec["source"],
                    "text": item["value"],
                }
            )
    _array(state["artifacts"]).append({"artifact_type": spec["output_type"], "identity": key, "value": value})
    return None


def _source_followup_available(state: Object, declaration: Object, identity: str, request: str, failure: str) -> bool:
    policy_key = cast(str, _object(state["request_policies"])[request])
    policy = _object(_object(state["policies"])[policy_key])
    replay = policy["replay"]
    permitted = (failure == "rejected_before_acceptance" and replay in ("before_acceptance", "idempotent")) or (
        failure in ("retryable", "transport_unknown", "malformed_response") and replay == "idempotent"
    )
    bounds = _object(declaration.get("retrieval_bounds", declaration.get("binding_limits", {})))
    maximum = min(cast(int, policy["max_attempts"]), cast(int, bounds.get("max_requests", policy["max_attempts"])))
    limit = declaration.get("hard_limit")
    return (
        permitted
        and cast(int, _object(state["attempts"])[identity]) < maximum
        and (
            limit is None
            or cast(int, state["dispatched_count"]) + len(_object(state["reservations"])) < cast(int, limit)
        )
    )


def _publish_operation(state: Object, declaration: Object, event: Object) -> Object | None:
    spec = _materialization_declaration(declaration, event)
    if spec is None or spec.get("version_selection") != "latest":
        return _reject("missing_materialization")
    publication = _object(spec["publication"])
    if {key: event.get(key) for key in ("node", "outcome", "output_port")} != {
        "node": publication["node"],
        "outcome": publication["outcome"],
        "output_port": publication["output_port"],
    }:
        return _reject("foreign_owner")
    occurrence_key = f"{event.get('activation')}:{event.get('attempt')}"
    occurrence = _object(_object(state["operation_occurrences"]).get(occurrence_key, {}))
    if not occurrence or occurrence.get("terminal") is not None:
        return _reject("duplicate" if occurrence else "foreign_owner")
    if not isinstance(event.get("value"), str):
        return _reject("invalid_type")
    materialization = _object(state.get("materialization", {}))
    port_key = f"initial:{spec['target']}:{spec['node']}:{spec['port']}:{spec['declaration']}"
    selected = _object(_object(materialization.get("ports", {})).get(port_key, {}))
    if (
        not selected
        or _object(state["binding_sources"]).get(cast(str, spec["association"])) != "bound"
        or state["binding_terminal"] not in ("success", "partial")
    ):
        return _reject("missing")
    parents = [
        key
        for key, raw in _object(materialization["provenance"]).items()
        if _object(raw).get("artifact") == selected["key"]
    ]
    if len(parents) != 1 or publication["inputs"] != [spec["port"]]:
        return _reject("contradictory")
    allocation = f"K{state['allocator_next']}"
    state["allocator_next"] = cast(int, state["allocator_next"]) + 1
    output_ref = f"ArtifactRef:I0:{allocation}:1"
    producer = (
        f"OperationOutputKey:{event['activation']}:{spec['target']}:"
        f"{publication['node']}:{publication['outcome']}:{publication['output_port']}"
    )
    byte_count = len(cast(str, event["value"]).encode())
    limits = _object(declaration["materialization_limits"])
    if cast(int, materialization["artifact_count"]) + 1 > cast(int, limits["max_artifacts"]):
        return _reject("artifact_count_exceeded")
    if cast(int, materialization["artifact_bytes"]) + byte_count > cast(int, limits["max_artifact_bytes"]):
        return _reject("artifact_bytes_exceeded")
    if cast(int, materialization["provenance_edges"]) + len(parents) > cast(int, limits["max_provenance_edges"]):
        return _reject("provenance_limit_exceeded")
    _object(materialization["provenance"])[producer] = {"artifact": output_ref, "parents": parents}
    materialization["artifact_count"] = cast(int, materialization["artifact_count"]) + 1
    materialization["artifact_bytes"] = cast(int, materialization["artifact_bytes"]) + byte_count
    materialization["provenance_edges"] = cast(int, materialization["provenance_edges"]) + len(parents)
    _array(state["artifacts"]).append(
        {
            "artifact_type": publication["artifact_type"],
            "identity": output_ref,
            "source": f"operation:{publication['node']}",
            "text": event["value"],
        }
    )
    occurrence["published_ports"] = [event["output_port"]]
    occurrence["terminal"] = {"category": "success", "outcome": event["outcome"]}
    return None


def _advance(state: Object, declaration: Object, event: Object) -> Object | None:
    kind = cast(str, event.get("kind"))
    reservations = _object(state["reservations"])
    dispatched = _strings(state["dispatched"])
    bindings = _object(state["bindings"])
    attempts = _object(state["attempts"])
    if state.get("binding_finished") is True:
        initial_associations = {
            cast(str, spec["association"])
            for spec in (_object(raw) for raw in _array(declaration.get("materializations", [])))
            if spec.get("path") == "initial"
        }
        request = cast(str, event.get("request", ""))
        request_associations = set(_strings(_object(state["request_associations"]).get(request, [])))
        reserved_associations = set(_strings(reservations.get(request, [])))
        event_associations = set(_strings(event.get("associations", [])))
        event_association = event.get("association")
        mutates_binding = (
            kind == "bind_policy"
            and event_association in initial_associations
            or kind == "reserve"
            and bool(event_associations & initial_associations)
            or kind == "dispatch"
            and bool(request_associations & initial_associations or reserved_associations & initial_associations)
            or kind in {"result", "failure", "cancel", "stop", "lost", "settlement"}
            and bool(request_associations & initial_associations or reserved_associations & initial_associations)
            or kind in {"source_result", "source_failure", "materialize_result"}
            and bool(
                event_association in initial_associations
                or request_associations & initial_associations
                or reserved_associations & initial_associations
            )
        )
        if mutates_binding:
            return _reject("contradictory")
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
        if event.get("purpose") in ("retry", "correction", "failover"):
            replay = _object(_object(state["policies"])[policy])["replay"]
            for item in associations:
                request_id = cast(str, _object(state["association_requests"]).get(item))
                facts = _object(state["request_facts"])
                if request_id not in facts:
                    return _reject("missing_predecessor")
                predecessor = _object(facts[request_id])
                if _object(state["request_policies"])[request_id] != policy:
                    return _reject("predecessor_policy")
                failure = predecessor.get("failure")
                purpose = event.get("purpose")
                if purpose == "retry":
                    if failure not in ("rejected_before_acceptance", "retryable", "transport_unknown"):
                        return _reject("invalid_retry")
                    permitted = failure == "rejected_before_acceptance" and replay in (
                        "before_acceptance",
                        "idempotent",
                    )
                    permitted = permitted or failure in ("retryable", "transport_unknown") and replay == "idempotent"
                elif purpose == "correction":
                    if failure != "malformed_response":
                        return _reject("invalid_correction")
                    permitted = failure == "malformed_response" and replay == "idempotent"
                else:
                    policy_value = _object(_object(state["policies"])[policy])
                    if failure not in _strings(policy_value.get("failover_failures", [])):
                        return _reject("invalid_failover")
                    permitted = replay == "idempotent" or (
                        failure == "rejected_before_acceptance" and replay == "before_acceptance"
                    )
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
            _object(state["reservation_policies"])[request] = policy
    elif kind == "dispatch":
        request = cast(str, event["request"])
        if request not in reservations:
            return _reject("missing_reservation")
        associations = _strings(reservations.pop(request))
        _object(state["request_associations"])[request] = associations
        policy = cast(str, _object(state["reservation_policies"]).pop(request))
        _object(state["request_policies"])[request] = policy
        dispatched.append(request)
        state["dispatched"] = sorted(set(dispatched))
        state["dispatched_count"] = cast(int, state["dispatched_count"]) + 1
        for item in associations:
            attempts[item] = cast(int, attempts.get(item, 0)) + 1
            _object(state["association_requests"])[item] = request
            if _object(state["tasks"]).get(item) == "running":
                _object(state["task_requests"])[item] = request
        for key in ("local_in_flight", "remote_outstanding"):
            values = _strings(state[key])
            _unique(cast(list[Json], values), request)
            state[key] = values
    elif kind == "result":
        request = cast(str, event["request"])
        if request not in dispatched:
            return _reject("missing")
        expected = set(_strings(_object(state["request_associations"])[request]))
        if any(_requires_materialization(declaration, "adaptive", association) for association in expected):
            return _reject("materialization_required")
        returned = _strings(event["returned"])
        outcomes_value = event.get("outcomes")
        if not isinstance(outcomes_value, dict) or any(not isinstance(value, str) for value in outcomes_value.values()):
            return _reject("invalid_result")
        outcomes = cast(Object, outcomes_value)
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
        if not defects and set(outcomes) != expected:
            defects.append("missing_keyed_result" if expected - set(outcomes) else "extra_keyed_result")
        for defect in defects:
            _unique(_array(state["defects"]), defect)
        terminal = "inconsistent" if defects else "success"
        first_terminal = request not in _object(state["terminals"])
        _terminal(state, request, terminal)
        if first_terminal:
            if defects:
                _request_fact(state, request, {"condition": "request_inconsistent"})
            else:
                _request_fact(state, request, {"condition": "result", "outcomes": outcomes})
        _remove(state, "local_in_flight", request)
        if first_terminal:
            _remove(state, "remote_outstanding", request)
    elif kind == "failure":
        request = cast(str, event["request"])
        if request not in dispatched:
            return _reject("not_dispatched")
        _record_request_failure(state, request, cast(str, event["failure"]))
    elif kind == "cancel":
        request = cast(str, event["request"])
        if request in _object(state["terminals"]):
            pass
        elif request in reservations:
            reservations.pop(request)
            _object(state["reservation_policies"]).pop(request)
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
            _object(state["reservation_policies"]).pop(request)
            _terminal(state, request, "cancelled")
        state["cancel_requested"] = sorted(set(_strings(state["cancel_requested"])) | set(dispatched))
    elif kind == "stop":
        request = cast(str, event["request"])
        if "usage" not in event or not _valid_usage(event["usage"]):
            return _reject("invalid_usage")
        if request not in _strings(state["cancel_requested"]):
            return _reject("cancel_not_requested")
        _terminal(state, request, "cancelled")
        _request_fact(state, request, {"condition": "cancel_after_dispatch"})
        _remove(state, "local_in_flight", request)
        _remove(state, "remote_outstanding", request)
    elif kind == "lost":
        request = cast(str, event["request"])
        if request not in dispatched:
            return _reject("not_dispatched")
        _terminal(state, request, "lost")
        _request_fact(state, request, {"condition": "lost"})
        _remove(state, "local_in_flight", request)
    elif kind == "settlement":
        request = cast(str, event["request"])
        settlements = _object(state["settlements"])
        if request not in dispatched:
            return _reject("not_dispatched")
        disposition = event.get("disposition")
        remote_stopped = event.get("remote_stopped")
        if disposition not in ("completed", "rejected", "stopped", "unknown"):
            return _reject("invalid_settlement")
        if not _valid_usage(event.get("usage")):
            return _reject("invalid_usage")
        if remote_stopped is not None and not isinstance(remote_stopped, bool):
            return _reject("invalid_settlement")
        if (disposition == "unknown") != (remote_stopped is not True):
            return _reject("invalid_settlement")
        value: Object = {key: event[key] for key in ("disposition", "usage", "remote_stopped")}
        if request in settlements and settlements[request] != value:
            _unique(_array(state["defects"]), "conflicting_settlement")
        else:
            settlements[request] = value
            if event["remote_stopped"] is True:
                _remove(state, "remote_outstanding", request)
    elif kind == "source_item_constructor":
        if any(not isinstance(_object(item).get("value"), str) for item in _array(event["items"])):
            return _reject("invalid_type")
    elif kind == "source_failure_constructor":
        # Required Python keywords and typed value invariants precede response acceptance.
        if "failure" not in event or "settlement" not in event:
            return {"status": "rejected", "exception": "TypeError"}
        if event.get("disposition") == "omitted_optional" and event.get("failure") != "permanent":
            return _reject("contradictory")
    elif kind == "binding_result":
        # AssociationResult construction precedes any request reducer event.
        if (
            event.get("outcome") != "retrieved"
            or event.get("outputs") != []
            or event.get("consumed_context_ports") != []
        ):
            return _reject("contradictory")
    elif kind == "source_result":
        request = cast(str, event.get("request"))
        if request not in dispatched or request not in _object(state["request_associations"]):
            return _reject("unsolicited_source")
        expected = _strings(_object(state["request_associations"])[request])
        if len(expected) != 1:
            return _reject("binding_request_shape")
        identity, source = expected[0], cast(str, event["source"])
        if _requires_materialization(declaration, "initial", identity):
            return _reject("materialization_required")
        if not _valid_settlement(event.get("settlement"), optional=False):
            return _reject("invalid_settlement")
        was_terminal = request in _object(state["terminals"])
        if _object(state["binding_declarations"]).get(identity) != source:
            _record_request_failure(state, request, "malformed_response")
            _apply_embedded_settlement(state, request, event["settlement"])
            if not was_terminal and not _source_followup_available(
                state, declaration, identity, request, "malformed_response"
            ):
                _object(state["binding_sources"])[identity] = "failed"
                requirements = _object(declaration.get("binding_requirements", {}))
                state["binding_terminal"] = "partial" if requirements.get(identity) == "optional" else "failed"
            return None
        if (
            event.get("outcome") != "retrieved"
            or event.get("outputs") != []
            or event.get("consumed_context_ports") != []
            or not isinstance(event.get("items"), list)
        ):
            _record_request_failure(state, request, "malformed_response")
            _apply_embedded_settlement(state, request, event["settlement"])
            if not was_terminal and not _source_followup_available(
                state, declaration, identity, request, "malformed_response"
            ):
                defects = state.setdefault("binding_defects", [])
                _unique(_array(defects), "malformed_source_result")
                requirements = _object(declaration.get("binding_requirements", {}))
                state["binding_terminal"] = "partial" if requirements.get(identity) == "optional" else "failed"
            return None
        items = [_object(item) for item in _array(event["items"]) if isinstance(item, dict)]
        returned = [cast(str, item.get("association")) for item in items]
        defects: list[str] = []
        if len(items) != len(_array(event["items"])) or not returned:
            defects.append("missing_keyed_result")
        if any(item != identity for item in returned):
            defects.append("foreign_keyed_result")
        item_keys = [(cast(str, item.get("association")), item.get("key"), item.get("version")) for item in items]
        if len(item_keys) != len(set(item_keys)):
            defects.append("duplicate_keyed_result")
        if any(
            not _natural(item.get("key"))
            or not _natural(item.get("version"), positive=True)
            or not isinstance(item.get("text"), str)
            for item in items
        ):
            defects.append("malformed_source_item")
        if defects:
            _record_request_failure(state, request, "malformed_response")
            _apply_embedded_settlement(state, request, event["settlement"])
            if not was_terminal and not _source_followup_available(
                state, declaration, identity, request, "malformed_response"
            ):
                _object(state["binding_sources"])[identity] = "failed"
                requirements = _object(declaration.get("binding_requirements", {}))
                state["binding_terminal"] = "partial" if requirements.get(identity) == "optional" else "failed"
            return None
        if request in _object(state["terminals"]):
            _record_binding_success(state, request, identity)
            _apply_embedded_settlement(state, request, event["settlement"])
            return None
        limits = _object(declaration.get("binding_limits", {}))
        byte_count = sum(len(cast(str, item["text"]).encode()) for item in items)
        _record_binding_success(state, request, identity)
        _apply_embedded_settlement(state, request, event["settlement"])
        if len(items) > cast(int, limits.get("max_items", len(items))) or byte_count > cast(
            int, limits.get("max_bytes", byte_count)
        ):
            _object(state["binding_sources"])[identity] = "oversize"
            requirements = _object(declaration.get("binding_requirements", {}))
            state["binding_terminal"] = "partial" if requirements.get(identity, "required") == "optional" else "failed"
            return None
        for item in items:
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
        request = cast(str, event.get("request"))
        if request not in dispatched or request not in _object(state["request_associations"]):
            return _reject("missing")
        associations = _strings(_object(state["request_associations"])[request])
        if len(associations) != 1:
            return _reject("binding_request_shape")
        identity = associations[0]
        if event.get("association") != identity:
            return _reject("foreign_association")
        retrieval_sources = _object(declaration.get("retrieval_sources", {}))
        adaptive = identity in retrieval_sources
        source = retrieval_sources.get(identity) if adaptive else _object(state["binding_declarations"]).get(identity)
        if event.get("source") != source:
            return _reject("foreign_source")
        failure = event.get("failure")
        if failure not in FAILURE_CLASSES:
            return _reject("invalid_failure")
        if "settlement" not in event or not _valid_settlement(event.get("settlement"), optional=True):
            return _reject("invalid_settlement")
        disposition = event.get("disposition", "failed")
        requirements = _object(declaration.get("binding_requirements", {}))
        valid_omission = (
            not adaptive
            and disposition == "omitted_optional"
            and requirements.get(identity) == "optional"
            and failure == "permanent"
        )
        if disposition not in ("failed", "omitted_optional") or (
            disposition == "omitted_optional" and not valid_omission
        ):
            failure = "malformed_response"
            disposition = "failed"
        was_terminal = request in _object(state["terminals"])
        _record_request_failure(state, request, cast(str, failure))
        _apply_embedded_settlement(state, request, event.get("settlement"))
        if was_terminal or _source_followup_available(state, declaration, identity, request, cast(str, failure)):
            return None
        if adaptive:
            _object(state["tasks"])[identity] = _materialization_category(
                declaration, "failure", failure=cast(str, failure)
            )
        else:
            _object(state["binding_sources"])[identity] = disposition
            state["binding_terminal"] = "partial" if requirements.get(identity) == "optional" else "failed"
    elif kind == "root_input":
        key = f"RootInputKey:{event.get('target')}:{event.get('port')}"
        specs = [_object(raw) for raw in _array(declaration.get("materializations", []))]
        if not any(key in _strings(spec.get("selector_inputs", [])) for spec in specs):
            return _reject("invalid_provenance")
        if event.get("artifact_type") not in _strings(declaration.get("root_input_types", [])) or not isinstance(
            event.get("value"), str
        ):
            return _reject("invalid_type")
        materialization = _object(state.get("materialization", {})) or {
            "artifact_bytes": 0,
            "artifact_count": 0,
            "ports": {},
            "provenance": {},
            "provenance_edges": 0,
        }
        if key in _object(materialization["provenance"]):
            return _reject("duplicate")
        limits = _object(declaration["materialization_limits"])
        byte_count = len(cast(str, event["value"]).encode())
        if cast(int, materialization["artifact_count"]) + 1 > cast(int, limits["max_artifacts"]):
            return _reject("artifact_count_exceeded")
        if cast(int, materialization["artifact_bytes"]) + byte_count > cast(int, limits["max_artifact_bytes"]):
            return _reject("artifact_bytes_exceeded")
        _object(materialization["provenance"])[key] = []
        materialization["artifact_count"] = cast(int, materialization["artifact_count"]) + 1
        materialization["artifact_bytes"] = cast(int, materialization["artifact_bytes"]) + byte_count
        state["materialization"] = materialization
        _array(state["artifacts"]).append(
            {"identity": key, "artifact_type": event["artifact_type"], "value": event["value"]}
        )
    elif kind == "root_artifact":
        if set(event) != {"artifact_type", "kind", "port", "target", "value"} or not isinstance(
            event.get("value"), str
        ):
            return _reject("invalid_type")
        admitted_root = {
            "artifact_type": event.get("artifact_type"),
            "port": event.get("port"),
            "target": event.get("target"),
        }
        if admitted_root not in [_object(raw) for raw in _array(declaration.get("root_artifacts", []))]:
            return _reject("foreign_owner")
        allocation = f"K{state['allocator_next']}"
        state["allocator_next"] = cast(int, state["allocator_next"]) + 1
        identity = f"ArtifactRef:I0:{allocation}:1"
        materialization = _object(state.get("materialization", {})) or {
            "artifact_bytes": 0,
            "artifact_count": 0,
            "ports": {},
            "provenance": {},
            "provenance_edges": 0,
        }
        producer = f"RootInputKey:{event['target']}:{event['port']}"
        _object(materialization["provenance"])[producer] = {"artifact": identity, "parents": []}
        materialization["artifact_count"] = cast(int, materialization["artifact_count"]) + 1
        materialization["artifact_bytes"] = cast(int, materialization["artifact_bytes"]) + len(
            cast(str, event["value"]).encode()
        )
        state["materialization"] = materialization
        _array(state["artifacts"]).append(
            {"artifact_type": event["artifact_type"], "identity": identity, "source": producer, "text": event["value"]}
        )
    elif kind == "operation_start":
        if set(event) != {"activation", "attempt", "binding_declaration", "kind", "node", "target"}:
            return _reject("invalid_value")
        admitted = [_object(raw) for raw in _array(declaration.get("operation_occurrences", []))]
        owner = {key: event[key] for key in ("activation", "attempt", "binding_declaration", "node", "target")}
        if owner not in admitted:
            return _reject("foreign_owner")
        initial_specs = [
            spec
            for spec in (_object(raw) for raw in _array(declaration.get("materializations", [])))
            if spec.get("path") == "initial"
        ]
        latest_specs = [
            spec
            for spec in initial_specs
            if spec.get("version_selection") == "latest" and spec.get("declaration") == owner["binding_declaration"]
        ]
        if state.get("binding_finished") is not True or any(
            _object(state["binding_sources"]).get(cast(str, spec["association"])) != "bound" for spec in latest_specs
        ):
            return _reject("missing")
        cleanup_associations = _object(state["binding_cleanup_associations"])
        cleanups = _object(state["binding_cleanup"])
        expected_cleanup = {cast(str, spec["association"]): cast(str, spec["target"]) for spec in initial_specs}
        actual_cleanup = {
            cast(str, _object(raw).get("association")): resource for resource, raw in cleanup_associations.items()
        }
        if set(actual_cleanup) != set(expected_cleanup) or set(cleanups) != set(actual_cleanup.values()):
            return _reject("missing")
        for association, target in expected_cleanup.items():
            resource = actual_cleanup[association]
            cleanup_owner = _object(cleanup_associations[resource]).get("owner")
            if _object(cleanup_associations[resource]).get("target") != target or cleanup_owner not in (
                "sdk",
                "caller",
            ):
                return _reject("foreign_owner")
            disposition = cleanups[resource]
            if cleanup_owner == "sdk" and disposition not in ("closed", "close_failed", "close_unknown"):
                return _reject("contradictory")
            if cleanup_owner == "caller" and disposition != "left_open":
                return _reject("missing")
        key = f"{event['activation']}:{event['attempt']}"
        occurrences = _object(state["operation_occurrences"])
        if key in occurrences:
            return _reject("duplicate")
        occurrences[key] = {**owner, "published_ports": [], "terminal": None}
    elif kind == "operation_publish":
        candidate = deepcopy(state)
        rejected = _publish_operation(candidate, declaration, event)
        if rejected is None:
            state.clear()
            state.update(candidate)
        elif rejected.get("code") in (
            "artifact_count_exceeded",
            "artifact_bytes_exceeded",
            "provenance_limit_exceeded",
        ):
            occurrence = _object(_object(state["operation_occurrences"])[f"{event['activation']}:{event['attempt']}"])
            occurrence["terminal"] = {"category": "failure", "reason": "artifact_limit_exhausted"}
            _array(state["publication_failures"]).append(
                {"activation": event["activation"], "attempt": event["attempt"], "reason": "artifact_limit_exhausted"}
            )
        else:
            return rejected
    elif kind == "materialize_result":
        request = cast(str, event.get("request"))
        if request not in dispatched or request not in _object(state["request_associations"]):
            return _reject("missing")
        spec = _materialization_declaration(declaration, event)
        if spec is None:
            return _reject("missing_materialization")
        expected = _strings(_object(state["request_associations"])[request])
        association = expected[0]
        if not _valid_settlement(event.get("settlement"), optional=False):
            return _reject("invalid_settlement")
        adaptive = spec["path"] == "adaptive"
        outcome = cast(str, event["reported_outcome"]) if adaptive else "retrieved"
        was_terminal = request in _object(state["terminals"])
        raw_items = _array(event["items"])
        pairs = [(_object(item).get("key"), _object(item).get("version")) for item in raw_items]
        malformed = (
            not raw_items
            or len(pairs) != len(set(pairs))
            or expected != [event.get("association")]
            or (spec.get("version_selection") == "latest" and len({pair[0] for pair in pairs}) != 1)
        )
        if malformed:
            if was_terminal:
                _unique(_array(state["defects"]), "malformed_late_response")
            _record_request_failure(state, request, "malformed_response")
            _apply_embedded_settlement(state, request, event["settlement"])
            if not was_terminal and not _source_followup_available(
                state, declaration, association, request, "malformed_response"
            ):
                if adaptive:
                    _object(state["tasks"]).setdefault(
                        cast(str, event["activation"]),
                        _materialization_category(declaration, "failure", failure="malformed_response"),
                    )
                else:
                    _object(state["binding_sources"])[association] = "failed"
                    state["binding_terminal"] = "failed"
            elif was_terminal and spec.get("version_selection") == "latest":
                _object(state["binding_sources"])[association] = cast(str, _object(state["terminals"])[request])
                state["binding_terminal"] = "failed"
            return None
        if was_terminal:
            _record_binding_success(state, request, association, outcome)
            _apply_embedded_settlement(state, request, event["settlement"])
            if spec.get("version_selection") == "latest":
                original = cast(str, _object(state["terminals"])[request])
                if original == "success":
                    original = "failed"
                _object(state["binding_sources"])[association] = original
                state["binding_terminal"] = "failed"
            return None
        candidate = deepcopy(state)
        rejected = _materialize(candidate, declaration, event)
        if rejected is None:
            state.clear()
            state.update(candidate)
        if "binding_artifacts" in state and (
            rejected is None
            or rejected.get("code")
            in ("artifact_count_exceeded", "artifact_bytes_exceeded", "provenance_limit_exceeded")
        ):
            _array(state["binding_artifacts"]).extend(
                {
                    "association": association,
                    "key": item.get("key"),
                    "source": spec.get("source"),
                    "text": item.get("value"),
                    "version": item.get("version"),
                }
                for item in (_object(raw) for raw in raw_items)
            )
        oversize = rejected is not None and rejected.get("code") in (
            "collection_limit_exceeded",
            "single_cardinality",
            "collection_cardinality",
            "materialization_bytes_exceeded",
        )
        publication_failure = rejected is not None and rejected.get("code") in (
            "artifact_count_exceeded",
            "artifact_bytes_exceeded",
            "provenance_limit_exceeded",
        )
        if rejected is not None and not oversize and not publication_failure:
            return rejected
        _record_binding_success(state, request, association, outcome)
        _apply_embedded_settlement(state, request, event["settlement"])
        if publication_failure:
            _array(state["publication_failures"]).append(
                {"association": association, "code": rejected["code"], "request": request}
            )
            _object(state["binding_sources"])[association] = "bound"
            state["binding_terminal"] = "success"
            return None
        if adaptive:
            condition = "artifact_limit_exhausted" if oversize else "result"
            _object(state["tasks"]).setdefault(
                cast(str, event["activation"]),
                _materialization_category(declaration, condition, reported_outcome=None if oversize else outcome),
            )
        else:
            _object(state["binding_sources"])[association] = "oversize" if oversize else "bound"
            if oversize:
                state["binding_terminal"] = "failed"
            elif all(
                _object(state["binding_sources"]).get(identity) == "bound"
                for identity in _object(state["binding_declarations"])
            ):
                state["binding_terminal"] = "success"
    elif kind == "binding_finish":
        if state.get("binding_finished") is True:
            return _reject("duplicate")
        requirements = _object(declaration.get("binding_requirements", {}))
        if "binding_finished" not in state:
            required = [
                cast(str, _object(raw)["association"])
                for raw in _array(declaration.get("materializations", []))
                if isinstance(raw, dict)
                and raw.get("path") == "initial"
                and requirements.get(cast(str, raw.get("association")), "required") != "optional"
            ]
            if any(_object(state["binding_sources"]).get(association) != "bound" for association in required):
                return _reject("missing_materialization")
            if state["binding_terminal"] is None:
                state["binding_terminal"] = "success"
            return None
        initial = [
            cast(str, spec["association"])
            for spec in (_object(raw) for raw in _array(declaration.get("materializations", [])))
            if spec.get("path") == "initial"
        ]
        required = {association for association in initial if requirements.get(association, "required") != "optional"}
        optional = set(initial) - required
        sources = _object(state["binding_sources"])
        if any(sources.get(association) != "bound" for association in required) or any(
            sources.get(association) not in ("bound", "omitted_optional") for association in optional
        ):
            return _reject("missing_materialization")
        expected_terminal = (
            "partial" if any(sources.get(association) == "omitted_optional" for association in optional) else "success"
        )
        if state["binding_terminal"] not in (None, expected_terminal):
            return _reject("missing_materialization")
        state["binding_terminal"] = expected_terminal
        if "binding_finished" in state:
            state["binding_finished"] = True
    elif kind == "resource":
        resource = cast(str, event["resource"])
        resources = _object(state["resources"])
        if resource not in resources:
            state["resource_count"] = cast(int, state["resource_count"]) + 1
        resources[resource] = {"owner": event["owner"], "safe_detachment": event["safe_detachment"]}
    elif kind == "binding_cleanup_association":
        if set(event) != {"association", "kind", "owner", "resource", "target"}:
            return _reject("invalid_value")
        resource = cast(str, event["resource"])
        associations = _object(state["binding_cleanup_associations"])
        if resource in associations:
            return _reject("duplicate")
        associations[resource] = {
            "association": event["association"],
            "owner": event["owner"],
            "target": event["target"],
        }
    elif kind == "binding_cleanup":
        if set(event) != {"disposition", "kind", "resource"}:
            return _reject("invalid_value")
        resource = cast(str, event["resource"])
        association = _object(_object(state["binding_cleanup_associations"]).get(resource, {}))
        if not association:
            return _reject("missing")
        if _array(state["local_in_flight"]) or _array(state["remote_outstanding"]):
            return _reject("contradictory")
        if association.get("owner") == "sdk" and event.get("disposition") not in (
            "closed",
            "close_failed",
            "close_unknown",
        ):
            return _reject("contradictory")
        _object(state["binding_cleanup"])[resource] = event["disposition"]
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
        request = _object(state["association_requests"]).get(task)
        if request is not None:
            _object(state["task_requests"])[task] = request
    elif kind == "bridge_close_unstarted":
        category = cast(str, event["category"])
        if category not in ("blocked", "inconsistent"):
            return _reject("invalid_category")
        _object(state["closed_unstarted"])[cast(str, event["activation"])] = category
    elif kind == "bridge_condition":
        task = cast(str, event["task"])
        if _object(state["tasks"]).get(task) != "running":
            return _reject("task_not_running")
        if set(event) - {"condition", "failure", "kind", "reported_outcome", "task"}:
            return _reject("runtime_mapping")
        request = _object(state["task_requests"]).get(task)
        if request is None:
            request = _object(state["association_requests"]).get(task)
            if request is not None:
                _object(state["task_requests"])[task] = request
        if request is not None:
            fact = _object(_object(state["request_facts"]).get(cast(str, request), {}))
            if fact.get("condition") != event.get("condition"):
                return _reject("request_causality")
            if fact.get("condition") == "failure" and fact.get("failure") != event.get("failure"):
                return _reject("request_causality")
            if fact.get("condition") == "result":
                outcomes = _object(fact.get("outcomes", {}))
                if outcomes.get(task) != event.get("reported_outcome"):
                    return _reject("request_causality")
        mappings = [
            _object(value)
            for value in _array(declaration.get("runtime_mappings", []))
            if _object(value).get("condition") == event.get("condition")
            and _object(value).get("failure") == event.get("failure")
            and _object(value).get("reported_outcome") == event.get("reported_outcome")
        ]
        if len(mappings) != 1:
            return _reject("runtime_mapping")
        mapping = mappings[0]
        condition = mapping.get("condition")
        outcome = mapping.get("outcome")
        category = mapping.get("category")
        categories = _object(declaration.get("outcome_categories", {}))
        if condition not in RUNTIME_CONDITIONS:
            return _reject("runtime_mapping")
        if outcome is None and category == "success":
            return _reject("runtime_mapping")
        if outcome is not None and categories.get(cast(str, outcome)) != category:
            return _reject("runtime_mapping")
        _object(state["tasks"])[task] = f"pending:{mapping['outcome']}:{mapping['category']}"
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
        if event.get("invocation", "I0") != "I0":
            return _reject("foreign_owner")
        if wait not in decisions:
            return _reject("missing")
        current = _object(decisions[wait])
        if current.get("closed") is True:
            return _reject("duplicate")
        if event["workflow"] != current["workflow"] or event["artifact"] != current["artifact"]:
            return _reject("foreign_owner")
        if event["decision"] not in _strings(current["allowed"]):
            return _reject("unsupported")
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


def _admit_materializations(declaration: Object) -> Object | None:
    raw_declarations = declaration.get("materializations", [])
    if not isinstance(raw_declarations, list):
        return _reject("invalid_type")
    if not raw_declarations:
        return None
    raw_limits = declaration.get("materialization_limits")
    required_limits = {
        "max_artifact_bytes",
        "max_artifacts",
        "max_collection_items",
        "max_declarations",
        "max_provenance_edges",
    }
    if not isinstance(raw_limits, dict) or set(raw_limits) != required_limits:
        return _reject("invalid_type")
    limits = _object(raw_limits)
    if any(not _natural(limits[key]) for key in required_limits):
        return _reject("invalid_type")
    if len(raw_declarations) > cast(int, limits["max_declarations"]):
        return _reject("limit_exceeded")
    if any(not isinstance(raw, dict) for raw in raw_declarations):
        return _reject("invalid_type")
    specs = [_object(raw) for raw in raw_declarations]
    base_fields = {
        "association",
        "declaration",
        "item_type",
        "kind",
        "max_bytes",
        "max_items",
        "node",
        "output_type",
        "path",
        "port",
        "source",
        "target",
        "selector_inputs",
    }
    if any(
        set(spec)
        != (
            base_fields
            | ({"version_selection"} if "version_selection" in spec else set())
            | ({"publication"} if spec.get("version_selection") == "latest" else set())
        )
        for spec in specs
    ):
        return _reject("invalid_value")
    if any(
        spec["path"] not in ("initial", "adaptive") or spec["kind"] not in ("single", "collection") for spec in specs
    ):
        return _reject("invalid_value")
    if any(
        not all(
            isinstance(spec[key], str) and spec[key]
            for key in ("association", "item_type", "node", "output_type", "port", "target")
        )
        or not _natural(spec["max_items"])
        or not _natural(spec["max_bytes"])
        for spec in specs
    ):
        return _reject("invalid_type")
    if any(
        spec["path"] == "initial" and (not isinstance(spec["declaration"], str) or not isinstance(spec["source"], str))
        for spec in specs
    ):
        return _reject("missing")
    if any(
        spec["path"] == "adaptive" and (spec["declaration"] is not None or spec["source"] is not None) for spec in specs
    ):
        return _reject("contradictory")
    if any(
        spec["path"] == "initial"
        and (
            spec.get("version_selection", "exact_one") not in ("exact_one", "latest")
            or (spec.get("version_selection") == "latest" and spec.get("kind") != "single")
            or (spec.get("version_selection") == "latest" and cast(int, spec["max_items"]) <= 0)
            or (
                spec.get("version_selection") == "latest"
                and (
                    not isinstance(spec.get("publication"), dict)
                    or set(_object(spec["publication"]))
                    != {"activation", "artifact_type", "inputs", "node", "outcome", "output_port"}
                    or not all(
                        isinstance(_object(spec["publication"]).get(field), str)
                        and bool(_object(spec["publication"])[field])
                        for field in ("activation", "artifact_type", "node", "outcome", "output_port")
                    )
                    or _object(spec["publication"]).get("node") != spec["node"]
                    or _object(spec["publication"]).get("inputs") != [spec["port"]]
                )
            )
        )
        for spec in specs
    ):
        return _reject("contradictory")
    if any(
        spec.get("selector_inputs") != ([f"RootInputKey:{spec['target']}:input"] if spec["path"] == "adaptive" else [])
        for spec in specs
    ):
        return _reject("invalid_provenance")
    identities = [tuple(spec[key] for key in ("path", "target", "node", "port", "declaration")) for spec in specs]
    if len(identities) != len(set(identities)):
        return _reject("duplicate")
    roots = [_object(raw) for raw in _array(declaration.get("root_artifacts", []))]
    if any(
        set(root) != {"artifact_type", "port", "target"}
        or root["target"] not in {spec["target"] for spec in specs}
        or not all(isinstance(root[field], str) and root[field] for field in root)
        for root in roots
    ):
        return _reject("foreign_owner")
    if len([(root["target"], root["port"]) for root in roots]) != len(
        set((root["target"], root["port"]) for root in roots)
    ):
        return _reject("duplicate")
    if any(
        spec["kind"] == "single"
        and (
            spec["output_type"] != spec["item_type"]
            or (spec.get("version_selection", "exact_one") == "exact_one" and spec["max_items"] != 1)
        )
        for spec in specs
    ):
        return _reject("contradictory")
    if any(
        spec["kind"] == "collection" and (spec["output_type"] == spec["item_type"] or cast(int, spec["max_items"]) < 1)
        for spec in specs
    ):
        return _reject("contradictory")
    schemas: dict[str, tuple[Json, Json]] = {}
    for spec in specs:
        shape = (spec["kind"], spec["item_type"])
        output = cast(str, spec["output_type"])
        if output in schemas and schemas[output] != shape:
            return _reject("contradictory")
        schemas[output] = shape
    collection_types = {cast(str, spec["output_type"]) for spec in specs if spec["kind"] == "collection"}
    if any(spec["item_type"] in collection_types for spec in specs):
        return _reject("contradictory")
    roots = declaration.get("root_input_types", [])
    if not isinstance(roots, list) or any(not isinstance(value, str) for value in roots):
        return _reject("invalid_type")
    if collection_types & set(cast(list[str], roots)):
        return _reject("contradictory")
    return None


def admit(declaration: Object) -> Object:
    allowed = {
        "admission_limits",
        "binding_targets",
        "declared_outcomes",
        "execution_policies",
        "outcome_categories",
        "capability_catalog",
        "runtime_mappings",
        "targets",
        "materializations",
        "materialization_limits",
        "root_artifacts",
        "root_input_types",
    }
    if set(declaration) - allowed:
        return _reject("invalid_value")
    materialized = _admit_materializations(declaration)
    if materialized is not None:
        return materialized
    policies_value = declaration.get("execution_policies", [])
    if not isinstance(policies_value, list):
        return _reject("invalid_type")
    catalog = declaration.get("capability_catalog", [])
    if not isinstance(catalog, list):
        return _reject("invalid_type")
    limits = declaration.get("admission_limits", {"max_capabilities": len(catalog)})
    if not isinstance(limits, dict) or not _natural(limits.get("max_capabilities")):
        return _reject("invalid_type")
    if len(catalog) > cast(int, limits["max_capabilities"]):
        return _reject("limit_exceeded")
    if any(not isinstance(raw, dict) for raw in catalog):
        return _reject("invalid_type")
    catalog_ids = [_object(raw).get("implementation") for raw in catalog]
    if len(catalog_ids) != len(set(cast(list[str], catalog_ids))):
        return _reject("duplicate")
    if any(not isinstance(raw, dict) for raw in policies_value):
        return _reject("invalid_type")
    policies = [_object(raw) for raw in policies_value]
    if any(policy.get("kind") not in ("local", "external", "decision") for policy in policies):
        return _reject("invalid_value")
    if any(policy.get("owner", "I0") != "I0" for policy in policies):
        return _reject("foreign_owner")
    nodes = [policy.get("node") for policy in policies]
    if len(nodes) != len({json.dumps(node, sort_keys=True) for node in nodes}):
        return _reject("duplicate")
    if any(not isinstance(policy.get("implementations"), list) for policy in policies):
        return _reject("invalid_type")
    if any(not policy["implementations"] for policy in policies):
        return _reject("implementation_count")
    if any(policy.get("retry_owner") == "implementation" for policy in policies):
        return _reject("unsupported")
    declared_values = _strings(declaration.get("declared_outcomes", []))
    if len(declared_values) != len(set(declared_values)):
        return _reject("duplicate")
    declared_outcomes = set(declared_values)
    mappings = declaration.get("runtime_mappings", [])
    if not isinstance(mappings, list) or any(not isinstance(value, dict) for value in mappings):
        return _reject("invalid_type")
    typed_mappings = [_object(value) for value in mappings]
    mapping_fields = {"category", "condition", "failure", "outcome", "reported_outcome"}
    if any(set(value) != mapping_fields for value in typed_mappings):
        return _reject("invalid_value")
    if any(value.get("condition") not in RUNTIME_CONDITIONS for value in typed_mappings):
        return _reject("invalid_value")
    mapping_keys = [_mapping_key(value) for value in typed_mappings]
    if len(mapping_keys) != len(set(mapping_keys)):
        return _reject("duplicate")
    if any(value.get("outcome") not in declared_outcomes | {None} for value in typed_mappings):
        return _reject("unsupported")
    categories = _object(declaration.get("outcome_categories", {}))
    if any(
        value.get("outcome") is not None and categories.get(cast(str, value["outcome"])) != value.get("category")
        for value in typed_mappings
    ):
        return _reject("contradictory")
    if any(value.get("outcome") is None and value.get("category") == "success" for value in typed_mappings):
        return _reject("contradictory")
    if any(
        value.get("condition") == "request_inconsistent" and value.get("category") != "inconsistent"
        for value in typed_mappings
    ):
        return _reject("contradictory")
    if any(
        value.get("condition") == "cancel_before_start"
        and (value.get("outcome") is not None or value.get("category") != "blocked")
        for value in typed_mappings
    ):
        return _reject("contradictory")
    targets = set(_strings(declaration.get("targets", [])))
    binding_targets = _strings(declaration.get("binding_targets", []))
    if any(target not in targets for target in binding_targets):
        return _reject("foreign_owner")
    for policy in policies:
        implementations = _array(policy["implementations"])
        if policy["kind"] in ("local", "decision") and len(implementations) != 1:
            return _reject("implementation_count")
        if policy.get("retry_owner") == "implementation":
            return _reject("retry_owner")
        if any(_object(item).get("physical_policy") != policy.get("physical_policy") for item in implementations):
            return _reject("changed_failover_policy")
    if len(policies) == 1:
        policy = policies[0]
        outcomes_value = policy.get("result_outcomes", [])
        if not isinstance(outcomes_value, list) or any(not isinstance(value, str) for value in outcomes_value):
            return _reject("invalid_type")
        outcomes = cast(list[str], outcomes_value)
        if len(outcomes) != len(set(outcomes)):
            return _reject("duplicate")
        if policy["kind"] in ("local", "external") and not outcomes:
            return _reject("missing")
        if policy["kind"] == "decision" and outcomes:
            return _reject("contradictory")
        expected_keys = _expected_mapping_keys(cast(str, policy["kind"]), outcomes)
        actual_keys = set(mapping_keys)
        if expected_keys - actual_keys:
            return _reject("missing")
        if actual_keys - expected_keys:
            return _reject("extra")
    return {"status": "accepted"}


def recheck_capabilities(declaration: Object) -> Object:
    admitted = _object(declaration["admitted"])
    initial = admit(admitted)
    if initial["status"] != "accepted":
        return initial
    supplied = _array(declaration["capability_catalog"])
    current = {_object(raw)["implementation"]: _object(raw) for raw in supplied}
    for raw in _array(admitted["capability_catalog"]):
        expected = _object(raw)
        if current.get(expected["implementation"]) != expected:
            return _reject("changed_failover_policy")
    return {"status": "accepted"}


def _materialization_preflight(declaration: Object, events: Sequence[Object]) -> Object:
    specs = [_object(raw) for raw in _array(declaration["materializations"])]
    binding_declaration = _binding_decl(
        {cast(str, spec["association"]): cast(str, spec["source"]) for spec in specs},
        max_items=max(cast(int, spec["max_items"]) for spec in specs),
        max_bytes=max(cast(int, spec["max_bytes"]) for spec in specs),
    )
    binding_events: list[Object] = []
    artifact_count = artifact_bytes = provenance_edges = 0
    for spec in specs:
        publication = _object(spec.get("publication", {}))
        if spec.get("version_selection") == "latest" and publication:
            artifact_count += 1
            provenance_edges += len(_array(publication["inputs"]))
    for event in events:
        if event["kind"] == "operation_publish":
            spec = _materialization_declaration(declaration, event)
            if spec is None:
                return _reject("missing")
            artifact_bytes += len(cast(str, event.get("value", "")).encode())
            continue
        if event["kind"] != "materialize_result":
            binding_events.append(event)
            continue
        spec = _materialization_declaration(declaration, event)
        if spec is None:
            return _reject("missing")
        items = [_object(raw) for raw in _array(event["items"])]
        response = _source_result(
            cast(str, spec["association"]),
            cast(str, spec["source"]),
            [
                dict(association=spec["association"], key=item["key"], version=item["version"], text=item["value"])
                for item in items
            ],
            request=cast(str, event["request"]),
        )
        response["settlement"] = event["settlement"]
        binding_events.append(response)
        collection = spec["kind"] == "collection"
        latest = spec.get("version_selection") == "latest"
        artifact_count += len(items) + 1 if collection else len(items) if latest else 1
        item_bytes = sum(len(cast(str, item["value"]).encode()) for item in items)
        artifact_bytes += item_bytes * (2 if collection else 1)
        provenance_edges += len(items) if collection else 0
    binding = reduce_trace(binding_declaration, [*binding_events, {"kind": "binding_finish"}])
    if binding["status"] != "accepted":
        return binding
    limits = _object(declaration["materialization_limits"])
    exceeded = (
        any(
            spec["kind"] == "collection" and cast(int, spec["max_items"]) > cast(int, limits["max_collection_items"])
            for spec in specs
        )
        or artifact_count > cast(int, limits["max_artifacts"])
        or artifact_bytes > cast(int, limits["max_artifact_bytes"])
        or provenance_edges > cast(int, limits["max_provenance_edges"])
    )
    result: Object = {"status": "rejected", "code": "limit_exceeded"} if exceeded else {"status": "accepted"}
    result["binding"] = binding["state"]
    return result


def _map_execution_preflight(declaration: Object) -> Object:
    admitted = _admit_map(declaration)
    if admitted["status"] != "accepted":
        return admitted
    # External output dependencies are added to both the public capacity and
    # structural requirement; this local projection retains only item edges.
    required = sum(
        cast(int, item["max_children"])
        for item in (_object(raw) for raw in _array(declaration["maps"]))
        if item["item_input"] is not None
    )
    if required > cast(int, _object(declaration["limits"])["max_provenance_edges"]):
        return _reject("limit_exceeded")
    return {"status": "accepted"}


def _map_local_result(declaration: Object, events: Sequence[Object]) -> Object:
    admitted = _admit_map(declaration)
    if admitted["status"] != "accepted":
        return admitted
    if len(events) != 1 or events[0].get("kind") != "local_result":
        return _reject("invalid_value")
    event = events[0]
    supplied = event.get("supplied_association")
    returned = event.get("returned_association")
    if supplied != returned:
        state = _empty_map_state()
        state["terminal"] = "malformed_response"
        state["parent_phase"] = "failed"
        state["expansion"] = {"parent": supplied, "members": [], "status": "failed"}
        return {"status": "accepted", "state": state}
    result = dict(event)
    result["kind"] = "map_result"
    result["parent"] = supplied
    del result["supplied_association"]
    del result["returned_association"]
    return _reduce_map(declaration, [result])


def evaluate_case(case: Mapping[str, Json]) -> Object:
    declaration = _object(case["declaration"])
    if case["boundary"] == "local_callback":
        return _map_local_result(declaration, [_object(raw) for raw in _array(case["events"])])
    if case["boundary"] == "map_execution_preflight":
        return _map_execution_preflight(declaration)
    if case["boundary"] == "execution_preflight":
        return _materialization_preflight(declaration, [_object(raw) for raw in _array(case["events"])])
    if case["boundary"] == "pre_execution":
        return recheck_capabilities(declaration)
    if case["family"] == "map":
        return (
            _admit_map(declaration)
            if case["boundary"] == "admission"
            else _reduce_map(declaration, [_object(value) for value in _array(case["events"])])
        )
    return (
        admit(declaration)
        if case["boundary"] == "admission"
        else reduce_trace(declaration, [_object(value) for value in _array(case["events"])])
    )


def _policy(max_attempts: int = 2, replay: str = "idempotent") -> Object:
    return {
        "failover_failures": ["permanent", "implementation_exception"],
        "max_attempts": max_attempts,
        "replay": replay,
        "retry_owner": "executor",
    }


def _decl(limit: int | None = 2, attempts: int = 2) -> Object:
    return {"hard_limit": limit, "policies": {"P0": _policy(attempts), "P1": _policy(3)}}


def _runtime_decl(
    condition: str,
    outcome: str | None,
    category: str,
    *,
    failure: str | None = None,
    reported_outcome: str | None = None,
) -> Object:
    mapping: Object = {
        "category": category,
        "condition": condition,
        "failure": failure,
        "outcome": outcome,
        "reported_outcome": reported_outcome,
    }
    declaration: Object = {"runtime_mappings": [mapping]}
    if outcome is not None:
        declaration["declared_outcomes"] = [outcome]
        declaration["outcome_categories"] = {outcome: category}
    return declaration


def _policy_runtime_decl(kind: str, outcomes: Sequence[str] = ("ok",)) -> Object:
    result_outcomes = () if kind == "decision" else tuple(outcomes)
    mappings: list[Json] = [
        {
            "category": "success",
            "condition": "result",
            "failure": None,
            "outcome": outcome,
            "reported_outcome": outcome,
        }
        for outcome in result_outcomes
    ]
    failures = ("permanent", "implementation_exception") if kind == "decision" else FAILURE_CLASSES
    mappings.extend(
        {
            "category": "failure",
            "condition": "failure",
            "failure": failure,
            "outcome": None,
            "reported_outcome": None,
        }
        for failure in failures
    )
    categories = {
        "artifact_limit_exhausted": "blocked",
        "budget_exhausted": "blocked",
        "cancel_after_dispatch": "cancelled",
        "cancel_after_start": "cancelled",
        "cancel_before_start": "blocked",
        "deadline_exhausted": "failure",
        "lost": "lost",
        "request_inconsistent": "inconsistent",
        "request_limit_exhausted": "blocked",
    }
    mappings.extend(
        {
            "category": categories[condition],
            "condition": condition,
            "failure": None,
            "outcome": None,
            "reported_outcome": None,
        }
        for condition in POLICY_CONDITIONS[kind]
    )
    declaration: Object = {
        "declared_outcomes": list(result_outcomes),
        "outcome_categories": {outcome: "success" for outcome in result_outcomes},
        "runtime_mappings": mappings,
    }
    return declaration


def _binding_decl(
    sources: Mapping[str, str],
    *,
    max_bytes: int = 8,
    max_items: int = 2,
    optional: Sequence[str] = (),
    adaptive: Sequence[str] = (),
    max_requests: int = 2,
) -> Object:
    declaration = _decl(attempts=max_requests)
    declaration["binding_declarations"] = dict(sources)
    if optional:
        declaration["binding_requirements"] = {
            identity: "optional" if identity in optional else "required" for identity in sources
        }
    if adaptive:
        declaration["binding_paths"] = {
            identity: "adaptive" if identity in adaptive else "initial" for identity in sources
        }
    declaration["binding_limits"] = {"max_bytes": max_bytes, "max_items": max_items, "max_requests": max_requests}
    return declaration


def _source_result(identity: str, source: str, items: Sequence[Object], *, request: str) -> Object:
    return {
        "consumed_context_ports": [],
        "items": list(items),
        "kind": "source_result",
        "outcome": "retrieved",
        "outputs": [],
        "request": request,
        "settlement": _embedded_settlement(),
        "source": source,
    }


def _source_failure(
    identity: str,
    source: str,
    *,
    request: str,
    failure: str = "permanent",
    disposition: str | None = None,
    settlement: Object | None = None,
) -> Object:
    event: Object = {
        "association": identity,
        "failure": failure,
        "kind": "source_failure",
        "request": request,
        "settlement": _embedded_settlement() if settlement is None else settlement,
        "source": source,
    }
    if disposition is not None:
        event["disposition"] = disposition
    return event


def _bind(item: str, *policies: str) -> Object:
    return {"association": item, "kind": "bind_policy", "policies": list(policies)}


def _reserve(request: str, items: Sequence[str], policy: str = "P0", purpose: str = "initial") -> Object:
    return {"associations": list(items), "kind": "reserve", "policy": policy, "purpose": purpose, "request": request}


def _dispatch(request: str) -> Object:
    return {"kind": "dispatch", "request": request}


def _result(request: str, expected: Sequence[str], returned: Sequence[str]) -> Object:
    expected_set = set(expected)
    outcomes = {item: "ok" for item in returned if item in expected_set}
    return {"kind": "result", "outcomes": outcomes, "request": request, "returned": list(returned)}


def _trace(items: Sequence[str] = ("T0",), request: str = "R0") -> list[Object]:
    return [*[_bind(item, "P0") for item in items], _reserve(request, items), _dispatch(request)]


def _admit_map(declaration: Object) -> Object:
    if set(declaration) != {"expansions", "limits", "loops", "maps", "schemas"}:
        return _reject("invalid_value")
    limits = declaration.get("limits")
    maps_value, loops_value, expansions_value, schemas_value = (
        declaration.get("maps"),
        declaration.get("loops"),
        declaration.get("expansions"),
        declaration.get("schemas"),
    )
    if not isinstance(limits, dict) or not all(
        isinstance(value, list) for value in (maps_value, loops_value, expansions_value, schemas_value)
    ):
        return _reject("invalid_type")
    typed_limits = _object(limits)
    if set(typed_limits) != {
        "max_artifact_bytes",
        "max_artifacts",
        "max_declarations",
        "max_collection_items",
        "max_provenance_edges",
        "max_schemas",
    } or any(not _natural(value) for value in typed_limits.values()):
        return _reject("invalid_type")
    maps = [_object(value) for value in cast(list[Json], maps_value)]
    loops = [_object(value) for value in cast(list[Json], loops_value)]
    expansions = [_object(value) for value in cast(list[Json], expansions_value)]
    schemas = [_object(value) for value in cast(list[Json], schemas_value)]
    if len(maps) + len(loops) + len(expansions) > cast(int, typed_limits["max_declarations"]) or len(schemas) > cast(
        int, typed_limits["max_schemas"]
    ):
        return _reject("limit_exceeded")
    if any(item.get("owner") != "W0" for item in (*maps, *loops, *expansions)):
        return _reject("foreign_owner")
    aggregate_sources = [item.get("expander") for item in maps] + [item.get("starter") for item in loops]
    if len(aggregate_sources) != len(set(aggregate_sources)):
        return _reject("duplicate")
    expansion_keys = [(item.get("scope"), item.get("expander"), item.get("outcome")) for item in expansions]
    if len(expansion_keys) != len(set(expansion_keys)):
        return _reject("duplicate")
    required = {
        (item.get("scope"), item.get("expander"), outcome)
        for item in maps
        for outcome in _strings(item.get("expansion_outcomes", []))
    }
    if set(expansion_keys) != required:
        return _reject("missing")
    schema_by_type: dict[str, tuple[Json, Json]] = {}
    for schema in schemas:
        if set(schema) != {"item_type", "kind", "minimum", "type"}:
            return _reject("invalid_value")
        artifact_type = schema.get("type")
        value = (schema.get("kind"), schema.get("item_type"))
        if not isinstance(artifact_type, str) or schema.get("kind") not in ("scalar", "collection"):
            return _reject("invalid_type")
        if artifact_type in schema_by_type and schema_by_type[artifact_type] != value:
            return _reject("contradictory")
        schema_by_type[artifact_type] = value
    collection_types = {name for name, value in schema_by_type.items() if value[0] == "collection"}
    if any(value[1] in collection_types for value in schema_by_type.values()):
        return _reject("nested_collection")
    for item in maps:
        if item.get("member_kind") not in ("operation", "subgraph") or not _natural(item.get("max_children")):
            return _reject("invalid_value")
        if item.get("item_input") is not None and item.get("item_input") != "item":
            return _reject("missing")
        if item.get("item_input") is not None and item.get("context_override") is True:
            return _reject("contradictory")
        if cast(int, item["max_children"]) > 1 and item.get("outward_scalar") in (
            "join",
            "ordinary",
            "workflow_output",
        ):
            return _reject("unsupported")
        retained_dependencies = _strings(item.get("retained_dependencies", []))
        expected_dependencies = sorted(set(_strings(item.get("membership_dependencies", []))))
        if retained_dependencies != expected_dependencies or item.get("retained_identity") is not False:
            return _reject("contradictory")
    for expansion in expansions:
        schema = schema_by_type.get(cast(str, expansion.get("collection_type")))
        if (
            schema != ("collection", expansion.get("item_type"))
            or expansion.get("collection_type") == expansion.get("item_type")
            or expansion.get("membership_port") != "members"
        ):
            return _reject("contradictory")
        minimum = next(
            schema_item.get("minimum")
            for schema_item in schemas
            if schema_item.get("type") == expansion.get("collection_type")
        )
        if minimum != 0:
            return _reject("contradictory")
        owner_map = next(item for item in maps if item.get("expander") == expansion.get("expander"))
        if owner_map.get("item_input") is not None and owner_map.get("item_type") != expansion.get("item_type"):
            return _reject("contradictory")
    return {"status": "accepted"}


def _empty_map_state() -> Object:
    return {
        "publication": {
            "artifacts": [],
            "assessments": [],
            "inputs": [],
            "membership": None,
            "ports": [],
            "provenance": [],
            "request_success": False,
        },
        "resolution": None,
        "parent_phase": "running",
        "terminal": None,
        "transition": None,
    }


def _reduce_map(declaration: Object, events: Sequence[Object]) -> Object:
    state = _empty_map_state()
    admitted = _admit_map(declaration)
    if admitted.get("status") != "accepted":
        return admitted
    map_decl = _object(_array(declaration["maps"])[0]) if _array(declaration["maps"]) else {}
    expansion = _object(_array(declaration["expansions"])[0]) if _array(declaration["expansions"]) else {}
    for event in events:
        if event.get("kind") == "collection_constructor":
            items = [_object(raw) for raw in _array(_object(_array(event["outputs"])[0])["items"])]
            pairs = [(cast(int, item["key"]), cast(int, item["version"])) for item in items]
            if len(pairs) != len(set(pairs)):
                return _reject("duplicate")
            if pairs != sorted(pairs):
                return _reject("invalid_value")
            continue
        if event.get("kind") == "close_parent":
            if event.get("parent") != "E0" or event.get("category") != "cancelled":
                return _reject("invalid_value")
            state["parent_phase"] = "cancelled"
            continue
        if "p3_accept" in event:
            return _reject("invalid_value")
        if event.get("kind") == "resolve_scalar":
            source_kind = event.get("source_kind")
            if source_kind == "map":
                count = event.get("member_count")
                maximum = event.get("maximum")
                destination = event.get("destination")
                if not _natural(count) or not _natural(maximum):
                    return _reject("invalid_type")
                if cast(int, maximum) > 1:
                    return _reject("unsupported")
                state["resolution"] = (
                    "member:0" if count == 1 else f"blocked:{destination}" if count == 0 else "overflow"
                )
            elif source_kind == "loop":
                outcomes = _strings(event.get("outcomes", []))
                if not outcomes:
                    state["resolution"] = "blocked:bypass"
                elif outcomes[-1] == "exit":
                    state["resolution"] = f"member:{len(outcomes) - 1}:exit"
                else:
                    state["resolution"] = "blocked:no_exit"
            else:
                return _reject("invalid_value")
            continue
        if event.get("kind") != "map_result":
            return _reject("unknown_event")
        if event.get("parent") != "E0":
            state["terminal"] = "wrong_parent"
            continue
        outputs_value = event.get("outputs")
        if not isinstance(outputs_value, list) or any(not isinstance(value, dict) for value in outputs_value):
            state["terminal"] = "malformed_response"
            continue
        outputs = [_object(value) for value in outputs_value]
        selected = [value for value in outputs if value.get("port") == expansion.get("membership_port")]
        if len(selected) != 1 or selected[0].get("artifact_type") != expansion.get("collection_type"):
            state["terminal"] = "malformed_response"
            continue
        items_value = selected[0].get("items")
        if not isinstance(items_value, list):
            state["terminal"] = "malformed_response"
            continue
        items: list[Object] = []
        pairs: list[tuple[int, int]] = []
        malformed = False
        for raw in items_value:
            if not isinstance(raw, dict) or set(raw) != {"key", "value", "version"}:
                malformed = True
                break
            item = _object(raw)
            if (
                not _natural(item["key"])
                or not _natural(item["version"], positive=True)
                or not isinstance(item["value"], str)
            ):
                malformed = True
                break
            items.append(item)
            pairs.append((cast(int, item["key"]), cast(int, item["version"])))
        if malformed or len(pairs) != len(set(pairs)) or pairs != sorted(pairs):
            state["terminal"] = "malformed_response"
            continue
        count = len(items)
        overflow = count > cast(int, map_decl["max_children"])
        byte_count = sum(len(cast(str, item["value"]).encode()) for item in items)
        other_outputs = [value for value in outputs if value is not selected[0]]
        if any(value.get("artifact_type") != "other_t" or value.get("items") != [] for value in other_outputs):
            state["terminal"] = "malformed_response"
            continue
        binds_items = map_decl.get("item_input") is not None and not overflow
        artifact_count = 1 + len(other_outputs) + (count if binds_items else 0)
        logical_bytes = byte_count * (2 if binds_items else 1)
        provenance_edges = count if binds_items else 0
        limits = _object(declaration["limits"])
        if (
            count > cast(int, limits["max_collection_items"])
            or artifact_count > cast(int, limits.get("max_artifacts", artifact_count))
            or logical_bytes > cast(int, limits.get("max_artifact_bytes", logical_bytes))
            or provenance_edges > cast(int, limits.get("max_provenance_edges", provenance_edges))
        ):
            state["terminal"] = "artifact_limit"
            continue
        # Prospective transition: a terminal parent cannot publish a late manifest.
        if state["parent_phase"] != "running":
            state["terminal"] = "transition_rejected"
            continue
        members = [] if overflow else [f"M{index}" for index in range(count)]
        collection_key = "OperationOutputKey:E0:T0:members"
        artifacts: list[Json] = [{"artifact_type": expansion["collection_type"], "identity": "C0", "items": items}]
        ports: list[Json] = [{"activation": "E0", "artifact": "C0", "port": "members"}]
        for index, output in enumerate(other_outputs):
            artifacts.append({"artifact_type": output["artifact_type"], "identity": f"O{index}", "items": []})
            ports.append({"activation": "E0", "artifact": f"O{index}", "port": output["port"]})
        inputs: list[Json] = []
        provenance: list[Json] = [
            {
                "artifact": port["artifact"],
                "decision": False,
                "key": f"OperationOutputKey:E0:T0:{port['port']}",
                "parents": [],
            }
            for port in cast(list[Object], ports)
        ]
        for member, item in zip(members, items[: len(members)], strict=True):
            if not binds_items:
                continue
            artifact = f"I{member[1:]}"
            key = f"MapItemKey:E0:{member}:T0:item:{item['key']}:{item['version']}"
            artifacts.append({"artifact_type": expansion["item_type"], "identity": artifact, "value": item["value"]})
            ports.append({"activation": member, "artifact": artifact, "port": "item"})
            inputs.append({"activation": member, "artifact": artifact, "key": key, "port": "item"})
            provenance.append({"artifact": artifact, "decision": False, "key": key, "parents": [collection_key]})
        state["publication"] = {
            "artifacts": artifacts,
            "assessments": list(_array(event.get("assessments", []))),
            "inputs": inputs,
            "membership": None if overflow else {"closed": True, "members": members, "parent": "E0"},
            "ports": ports,
            "provenance": provenance,
            "request_success": True,
        }
        state["parent_phase"] = "complete"
        state["terminal"] = "overflow" if overflow else "published"
        state["transition"] = (
            {"count": count, "kind": "ObserveOverflow", "parent": "E0"}
            if overflow
            else {"kind": "ObserveMembership", "members": members, "parent": "E0"}
        )
    return {"state": state, "status": "accepted"}


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


def _materialization_limits(**changes: int) -> Object:
    limits: Object = {
        "max_artifact_bytes": 64,
        "max_artifacts": 8,
        "max_collection_items": 3,
        "max_declarations": 4,
        "max_provenance_edges": 8,
    }
    limits.update(changes)
    return limits


def _materialization_spec(
    path: str,
    kind: str,
    *,
    association: str | None = None,
    declaration: str | None = None,
    item_type: str = "text",
    node: str = "N0",
    output_type: str | None = None,
    port: str = "context",
    target: str = "T0",
    version_selection: str | None = None,
) -> Object:
    identity = association or (declaration if path == "initial" else "A0") or "A0"
    result: Object = {
        "association": identity,
        "declaration": declaration if path == "initial" else None,
        "item_type": item_type,
        "kind": kind,
        "max_bytes": 12,
        "max_items": 3 if version_selection == "latest" else 1 if kind == "single" else 3,
        "node": node,
        "output_type": output_type or (item_type if kind == "single" else f"{item_type}_collection"),
        "path": path,
        "port": port,
        "source": "S0" if path == "initial" else None,
        "target": target,
        "selector_inputs": [f"RootInputKey:{target}:input"] if path == "adaptive" else [],
    }
    if path == "initial" and version_selection is not None:
        result["version_selection"] = version_selection
        if version_selection == "latest":
            result["publication"] = {
                "activation": f"OP:{identity}",
                "artifact_type": result["output_type"],
                "inputs": [port],
                "node": node,
                "outcome": "ok",
                "output_port": "result",
            }
    return result


def _materialization_category(
    declaration: Object, condition: str, *, failure: str | None = None, reported_outcome: str | None = None
) -> str:
    mappings = [
        _object(raw)
        for raw in _array(declaration["runtime_mappings"])
        if _object(raw).get("condition") == condition
        and _object(raw).get("failure") == failure
        and _object(raw).get("reported_outcome") == reported_outcome
    ]
    if len(mappings) != 1:
        raise ValueError("materialization requires one admitted runtime mapping")
    return cast(str, mappings[0]["category"])


def _materialization_decl(
    *specs: Object,
    admission: bool = False,
    max_requests: int = 2,
    limit_changes: Mapping[str, int] | None = None,
) -> Object:
    sources = {cast(str, spec["association"]): cast(str, spec["source"]) for spec in specs if spec["path"] == "initial"}
    max_items = max((cast(int, spec["max_items"]) for spec in specs if spec["path"] == "initial"), default=0)
    max_bytes = max((cast(int, spec["max_bytes"]) for spec in specs if spec["path"] == "initial"), default=0)
    declaration: Object = (
        {} if admission else _binding_decl(sources, max_bytes=max_bytes, max_items=max_items, max_requests=max_requests)
    )
    declaration["materializations"] = list(specs)
    declaration["materialization_limits"] = _materialization_limits(**dict(limit_changes or {}))
    declaration["root_input_types"] = ["text"]
    if any(spec.get("version_selection") == "latest" for spec in specs):
        targets = sorted({cast(str, spec["target"]) for spec in specs})
        declaration["root_artifacts"] = [
            {"artifact_type": "text", "port": port, "target": target}
            for target in targets
            for port in ("left", "right")
        ]
        if not admission:
            declaration["operation_occurrences"] = [
                {
                    "activation": _object(spec["publication"])["activation"],
                    "attempt": f"TASK:{_object(spec['publication'])['activation']}",
                    "binding_declaration": spec["declaration"],
                    "node": _object(spec["publication"])["node"],
                    "target": spec["target"],
                }
                for spec in specs
                if spec.get("version_selection") == "latest"
            ]
    if not admission and any(spec["path"] == "adaptive" for spec in specs):
        declaration.update(_policy_runtime_decl("external"))
        declaration["retrieval_bounds"] = {"max_requests": max_requests}
    return declaration


def _materialization_event(spec: Object, items: Sequence[Object], *, request: str = "R0") -> Object:
    return {
        "activation": "A0" if spec["path"] == "adaptive" else None,
        "association": spec["association"],
        "declaration": spec["declaration"],
        "items": list(items),
        "kind": "materialize_result",
        "settlement": _embedded_settlement(),
        "node": spec["node"],
        "parents": spec["selector_inputs"],
        "reported_outcome": "ok" if spec["path"] == "adaptive" else None,
        "path": spec["path"],
        "port": spec["port"],
        "request": request,
        "target": spec["target"],
    }


def _operation_start(spec: Object, *, activation: str | None = None) -> Object:
    publication = _object(spec["publication"])
    owner = cast(str, activation or publication["activation"])
    return {
        "activation": owner,
        "attempt": f"TASK:{owner}",
        "binding_declaration": spec["declaration"],
        "kind": "operation_start",
        "node": publication["node"],
        "target": spec["target"],
    }


def _publication_event(spec: Object, *, activation: str | None = None, value: str = "result") -> Object:
    publication = _object(spec["publication"])
    owner = cast(str, activation or publication["activation"])
    return {
        "activation": owner,
        "attempt": f"TASK:{owner}",
        "declaration": spec["declaration"],
        "kind": "operation_publish",
        "node": publication["node"],
        "outcome": publication["outcome"],
        "output_port": publication["output_port"],
        "path": spec["path"],
        "port": spec["port"],
        "target": spec["target"],
        "value": value,
    }


def _items(*values: str) -> list[Object]:
    return [{"key": index, "value": value, "version": 1} for index, value in enumerate(values)]


def _materialization_trace(spec: Object, items: Sequence[Object], *, request: str = "R0") -> list[Object]:
    association = cast(str, spec["association"])
    roots: list[Object] = (
        [
            {
                "kind": "root_input",
                "target": spec["target"],
                "port": "input",
                "artifact_type": "text",
                "value": "selector",
            }
        ]
        if spec["path"] == "adaptive"
        else []
    )
    purpose = "adaptive_retrieval" if spec["path"] == "adaptive" else "initial_binding"
    return [
        *roots,
        _bind(association, "P0"),
        _reserve(request, (association,), purpose=purpose),
        _dispatch(request),
        _materialization_event(spec, items, request=request),
    ]


def _materialization_specs() -> list[Object]:
    cases: list[Object] = []
    initial_single = _materialization_spec("initial", "single", declaration="D0")
    initial_collection = _materialization_spec("initial", "collection", declaration="D0")
    adaptive_single = _materialization_spec("adaptive", "single")
    adaptive_collection = _materialization_spec("adaptive", "collection")
    for name, spec in (
        ("admit_initial_single", initial_single),
        ("admit_initial_collection", initial_collection),
        ("admit_adaptive_single", adaptive_single),
        ("admit_adaptive_collection", adaptive_collection),
    ):
        cases.append(_case("materialization", name, _materialization_decl(spec, admission=True), [], "admission"))
    admission_mutants: tuple[tuple[str, Object, Object], ...] = (
        ("single_output_mismatch", initial_single, {"output_type": "text_collection"}),
        ("single_max_items", initial_single, {"max_items": 2}),
        ("collection_scalar_output", initial_collection, {"output_type": "text"}),
        ("collection_zero_max", initial_collection, {"max_items": 0}),
        ("adaptive_binding_identity", adaptive_collection, {"declaration": "D0"}),
    )
    for name, base, changes in admission_mutants:
        mutant = dict(base)
        mutant.update(changes)
        cases.append(_case("materialization", name, _materialization_decl(mutant, admission=True), [], "admission"))
    collection_ceiling = dict(initial_collection, max_items=4)
    cases.append(
        _case(
            "materialization",
            "collection_ceiling",
            _materialization_decl(collection_ceiling),
            _materialization_trace(collection_ceiling, _items("one")),
            "execution_preflight",
        )
    )
    text_collection = _materialization_spec("initial", "collection", declaration="D0")
    nested = _materialization_spec(
        "adaptive", "collection", association="A1", item_type="text_collection", output_type="nested_collection"
    )
    cases.extend(
        [
            _case(
                "materialization",
                "conflicting_schema",
                _materialization_decl(
                    text_collection,
                    dict(text_collection, association="D1", declaration="D1", item_type="bytes"),
                    admission=True,
                ),
                [],
                "admission",
            ),
            _case(
                "materialization",
                "nested_collection_schema",
                _materialization_decl(text_collection, nested, admission=True),
                [],
                "admission",
            ),
            _case(
                "materialization",
                "collection_root_input",
                dict(_materialization_decl(text_collection, admission=True), root_input_types=["text_collection"]),
                [],
                "admission",
            ),
        ]
    )
    for label, spec in (("initial", initial_single), ("adaptive", adaptive_single)):
        cases.append(
            _case(
                "materialization",
                f"{label}_single_exact",
                _materialization_decl(spec),
                _materialization_trace(spec, _items("one")),
            )
        )
        cases.append(
            _case(
                "materialization",
                f"{label}_single_multiple",
                _materialization_decl(spec),
                _materialization_trace(spec, _items("one", "two")),
            )
        )
    for label, spec in (("initial", initial_collection), ("adaptive", adaptive_collection)):
        for size in (1, 2, 3, 4, 0):
            values = tuple(f"v{index}" for index in range(size))
            cases.append(
                _case(
                    "materialization",
                    f"{label}_collection_{'one_over' if size == 4 else size}",
                    _materialization_decl(spec, max_requests=1 if size == 0 else 2),
                    _materialization_trace(spec, _items(*values)),
                )
            )
        reordered = list(reversed(_items("zero", "one")))
        cases.append(
            _case(
                "materialization",
                f"{label}_canonical_reorder",
                _materialization_decl(spec),
                _materialization_trace(spec, reordered),
            )
        )
        malformed: tuple[tuple[str, list[Object]], ...] = (
            ("duplicate", [{"key": 0, "value": "a", "version": 1}, {"key": 0, "value": "b", "version": 1}]),
            ("wrong_item", [{"key": 0, "value": 7, "version": 1}]),
            ("nested_value", [{"key": 0, "value": {"items": []}, "version": 1}]),
        )
        for suffix, values in malformed:
            trace = _materialization_trace(spec, values)
            if suffix in ("wrong_item", "nested_value"):
                trace[-1]["kind"] = "source_item_constructor"
            cases.append(
                _case(
                    "materialization",
                    f"{label}_{suffix}",
                    _materialization_decl(spec, max_requests=1 if suffix == "duplicate" else 2),
                    trace,
                )
            )
        one_over_malformed = _items("a", "b", "c", "d")
        cases.append(
            _case(
                "materialization",
                f"{label}_outer_count_precedence",
                _materialization_decl(spec),
                _materialization_trace(spec, one_over_malformed),
            )
        )
    for suffix, limits in (
        ("artifact_count_one_over", {"max_artifacts": 2}),
        ("logical_bytes_one_over", {"max_artifact_bytes": 7}),
        ("provenance_one_over", {"max_provenance_edges": 1}),
    ):
        cases.append(
            _case(
                "materialization",
                f"initial_{suffix}",
                _materialization_decl(initial_collection, limit_changes=limits),
                _materialization_trace(initial_collection, _items("aa", "bb")),
                "execution_preflight",
            )
        )
    for suffix, maximum in (("declared_bytes_exact", 4), ("declared_bytes_one_over", 3)):
        byte_spec = dict(initial_collection, max_bytes=maximum)
        cases.append(
            _case(
                "materialization",
                f"initial_{suffix}",
                _materialization_decl(byte_spec),
                _materialization_trace(byte_spec, _items("aa", "bb")),
            )
        )
    cases.append(
        _case(
            "materialization",
            "adaptive_foreign_association",
            _materialization_decl(adaptive_collection, max_requests=1),
            [
                *_materialization_trace(adaptive_collection, _items("a"))[:-1],
                dict(_materialization_event(adaptive_collection, _items("a")), association="A1"),
            ],
        )
    )
    cases.extend(
        [
            _case(
                "materialization",
                "initial_unmaterialized_source_result",
                _materialization_decl(initial_collection),
                _materialization_trace(initial_collection, _items("a")),
            ),
            _case(
                "materialization",
                "adaptive_unmaterialized_result",
                _materialization_decl(adaptive_collection),
                _materialization_trace(adaptive_collection, _items("a")),
            ),
            _case(
                "materialization",
                "binding_finish_before_materialization",
                _materialization_decl(initial_collection),
                _materialization_trace(initial_collection, _items("a")),
            ),
        ]
    )
    first = _materialization_spec("initial", "collection", declaration="D0", node="N0")
    second = _materialization_spec("initial", "collection", declaration="D1", node="N1")
    cases.append(
        _case(
            "materialization",
            "same_port_distinct_nodes",
            _materialization_decl(first, second),
            [
                *_materialization_trace(first, _items("a")),
                *_materialization_trace(second, _items("b"), request="R1"),
            ],
        )
    )
    for path, spec in (("initial", initial_single), ("adaptive", adaptive_single)):
        cases.append(
            _case(
                "materialization",
                f"{path}_single_zero_collection_limit",
                _materialization_decl(spec, limit_changes={"max_collection_items": 0}),
                _materialization_trace(spec, _items("one")),
            )
        )
        for terminal in ("lost", "cancelled"):
            prefix = _materialization_trace(spec, _items("one"))[:-1]
            terminals: list[Object] = (
                [{"kind": "lost", "request": "R0"}]
                if terminal == "lost"
                else [{"kind": "cancel", "request": "R0"}, {"kind": "stop", "request": "R0", "usage": "unknown"}]
            )
            if terminal == "lost":
                terminals.insert(0, {"kind": "cancel", "request": "R0"})
            cases.append(
                _case(
                    "materialization",
                    f"{path}_late_{terminal}",
                    _materialization_decl(spec),
                    [*prefix, *terminals, _materialization_event(spec, _items("late"))],
                )
            )
    # Historical inaccessible provenance mutations are explicit positive invariant aliases.
    for suffix in ("missing_parent", "foreign_parent", "invented_parent", "missing_source_fact"):
        cases.append(
            _case(
                "materialization",
                f"adaptive_{suffix}",
                _materialization_decl(adaptive_collection),
                _materialization_trace(adaptive_collection, _items("one")),
            )
        )
    bridge_decl = _materialization_decl(adaptive_collection)
    bridge_decl["runtime_mappings"] = [
        {"condition": "result", "failure": None, "reported_outcome": "ok", "outcome": "ok", "category": "success"}
    ]
    bridge_decl["outcome_categories"] = {"ok": "success"}
    cases.append(
        _case(
            "materialization",
            "adaptive_result_bridge",
            bridge_decl,
            [
                {"kind": "bridge_start", "task": "A0"},
                *_materialization_trace(adaptive_collection, _items("one")),
                {"kind": "bridge_condition", "task": "A0", "condition": "result", "reported_outcome": "ok"},
                {"kind": "bridge_emit", "task": "A0", "outcome": "ok", "category": "success"},
            ],
        )
    )
    return cases


def _map_decl(
    *,
    max_children: int = 2,
    item_input: str | None = "item",
    member_kind: str = "operation",
    outward_scalar: str | None = None,
    **changes: Json,
) -> Object:
    map_value: Object = {
        "context_override": False,
        "default_dependencies": ["default_root"],
        "expander": "E",
        "expansion_outcomes": ["expand"],
        "item_input": item_input,
        "item_type": "text",
        "join": "J",
        "max_children": max_children,
        "member": "M",
        "member_kind": member_kind,
        "membership_dependencies": ["actual_membership_root"],
        "outward_scalar": outward_scalar,
        "owner": "W0",
        "retained_dependencies": ["actual_membership_root"],
        "retained_identity": False,
        "scope": "S0",
    }
    map_value.update(changes)
    return {
        "expansions": [
            {
                "collection_type": "members_t",
                "expander": "E",
                "item_type": "text",
                "membership_port": "members",
                "outcome": "expand",
                "owner": "W0",
                "scope": "S0",
            }
        ],
        "limits": {
            "max_artifact_bytes": 32,
            "max_artifacts": 8,
            "max_declarations": 8,
            "max_collection_items": 4,
            "max_provenance_edges": 8,
            "max_schemas": 8,
        },
        "loops": [],
        "maps": [map_value],
        "schemas": [
            {"item_type": None, "kind": "scalar", "minimum": 0, "type": "text"},
            {"item_type": "text", "kind": "collection", "minimum": 0, "type": "members_t"},
            {"item_type": "text", "kind": "collection", "minimum": 0, "type": "other_t"},
        ],
    }


def _map_event(
    values: Sequence[str],
    *,
    parent: str = "E0",
    port: str = "members",
    artifact_type: str = "members_t",
    extra_outputs: int = 0,
) -> Object:
    outputs: list[Json] = [
        {
            "artifact_type": artifact_type,
            "items": [{"key": index, "value": value, "version": 1} for index, value in enumerate(values)],
            "port": port,
        }
    ]
    outputs.extend({"artifact_type": "other_t", "items": [], "port": f"other{index}"} for index in range(extra_outputs))
    return {
        "assessments": ["assessment0"],
        "kind": "map_result",
        "outputs": outputs,
        "parent": parent,
    }


def _map_specs() -> list[Object]:
    cases: list[Object] = []
    for name, declaration in (
        ("admit_operation", _map_decl()),
        ("admit_subgraph", _map_decl(member_kind="subgraph")),
        ("admit_control_only", _map_decl(item_input=None)),
        ("default_override_dependency", _map_decl(default_dependencies=["unavailable_default"])),
        ("max_zero_scalar_join", _map_decl(max_children=0, outward_scalar="join")),
        ("max_one_scalar_ordinary", _map_decl(max_children=1, outward_scalar="ordinary")),
        ("max_one_scalar_workflow", _map_decl(max_children=1, outward_scalar="workflow_output")),
    ):
        cases.append(_case("map", name, declaration, [], "admission"))
    for destination in ("join", "ordinary", "workflow_output"):
        cases.append(
            _case(
                "map",
                f"max_two_scalar_{destination}",
                _map_decl(max_children=2, outward_scalar=destination),
                [],
                "admission",
            )
        )
    admission_mutants: list[tuple[str, Object]] = []
    missing_expansion = _map_decl()
    missing_expansion["expansions"] = []
    admission_mutants.append(("missing_expansion", missing_expansion))
    duplicate_map = _map_decl()
    _array(duplicate_map["maps"]).append(dict(_object(_array(duplicate_map["maps"])[0]), member="M1"))
    admission_mutants.append(("duplicate_map_source", duplicate_map))
    map_loop = _map_decl()
    map_loop["loops"] = [{"join": "LJ", "member": "LM", "owner": "W0", "scope": "S0", "starter": "E"}]
    admission_mutants.append(("map_loop_duplicate_source", map_loop))
    duplicate_loop = _map_decl()
    duplicate_loop["loops"] = [
        {"join": "LJ0", "member": "LM0", "owner": "W0", "scope": "S0", "starter": "L"},
        {"join": "LJ1", "member": "LM1", "owner": "W0", "scope": "S0", "starter": "L"},
    ]
    admission_mutants.append(("duplicate_loop_source", duplicate_loop))
    minimum = _map_decl()
    _object(_array(minimum["schemas"])[1])["minimum"] = 1
    admission_mutants.append(("context_minimum_conflict", minimum))
    schema_conflict = _map_decl()
    _array(schema_conflict["schemas"]).append(
        {"item_type": "bytes", "kind": "collection", "minimum": 0, "type": "members_t"}
    )
    admission_mutants.append(("collection_item_schema_conflict", schema_conflict))
    item_mismatch = _map_decl()
    _object(_array(item_mismatch["expansions"])[0])["item_type"] = "bytes"
    admission_mutants.append(("item_type_mismatch", item_mismatch))
    admission_mutants.append(("context_override_conflict", _map_decl(context_override=True)))
    admission_mutants.append(("dependency_summary_mismatch", _map_decl(retained_dependencies=["default_root"])))
    admission_mutants.append(("false_identity_summary", _map_decl(retained_identity=True)))
    foreign = _map_decl()
    _object(_array(foreign["maps"])[0])["owner"] = "W1"
    _array(foreign["maps"]).append(dict(_object(_array(foreign["maps"])[0]), member="M1"))
    admission_mutants.append(("foreign_before_duplicate", foreign))
    for name, declaration in admission_mutants:
        cases.append(_case("map", name, declaration, [], "admission"))
    for extra in (0, 1, 2):
        cases.append(
            _case(
                "map",
                f"membership_with_{extra}_other_outputs",
                _map_decl(),
                [_map_event(("a", "b"), extra_outputs=extra)],
            )
        )
    for size in (0, 1, 2, 3):
        cases.append(
            _case(
                "map",
                f"membership_{'one_over' if size == 3 else size}",
                _map_decl(),
                [_map_event(tuple(chr(97 + index) for index in range(size)))],
            )
        )
    malformed_events: list[tuple[str, Object]] = [
        ("missing_membership_port", _map_event(("a",), port="other")),
        ("wrong_membership_type", _map_event(("a",), artifact_type="text")),
    ]
    wrong_association = _map_event(("a",))
    wrong_association.pop("parent")
    wrong_association.update({"kind": "local_result", "supplied_association": "E0", "returned_association": "E1"})
    cases.append(_case("map", "wrong_parent", _map_decl(), [wrong_association], "local_callback"))
    duplicate_port = _map_event(("a",))
    _array(duplicate_port["outputs"]).append(dict(_object(_array(duplicate_port["outputs"])[0])))
    malformed_events.append(("duplicate_membership_port", duplicate_port))
    duplicate_item = _map_event(("a", "b"))
    _object(_array(_object(_array(duplicate_item["outputs"])[0])["items"])[1])["key"] = 0
    duplicate_item["kind"] = "collection_constructor"
    malformed_events.append(("duplicate_item", duplicate_item))
    noncanonical = _map_event(("a", "b"))
    _array(_object(_array(noncanonical["outputs"])[0])["items"]).reverse()
    noncanonical["kind"] = "collection_constructor"
    malformed_events.append(("noncanonical_items", noncanonical))
    for name, event in malformed_events:
        cases.append(_case("map", name, _map_decl(), [event]))
    cases.extend(
        [
            _case("map", "subgraph_item_binding", _map_decl(member_kind="subgraph"), [_map_event(("left", "right"))]),
            _case("map", "control_only_members", _map_decl(item_input=None), [_map_event(("a", "b"))]),
            _case(
                "map",
                "default_override_no_fallback",
                _map_decl(default_dependencies=["unavailable_default"]),
                [_map_event(("actual",))],
            ),
            _case(
                "map",
                "prospective_transition_rejected",
                _map_decl(),
                [{"kind": "close_parent", "parent": "E0", "category": "cancelled"}, _map_event(("a",))],
            ),
        ]
    )
    for name, limit, value in (
        ("artifact_count_one_over", "max_artifacts", 2),
        ("artifact_bytes_one_over", "max_artifact_bytes", 7),
        ("provenance_one_over", "max_provenance_edges", 1),
    ):
        declaration = _map_decl()
        _object(declaration["limits"])[limit] = value
        if name == "provenance_one_over":
            cases.append(_case("map", name, declaration, [], "map_execution_preflight"))
        else:
            cases.append(_case("map", name, declaration, [_map_event(("aa", "bb"))]))
    exact = _map_decl()
    _object(exact["limits"]).update({"max_artifacts": 3, "max_artifact_bytes": 8, "max_provenance_edges": 2})
    cases.append(_case("map", "bounds_exact", exact, [_map_event(("aa", "bb"))]))
    for maximum in (0, 1, 2):
        for destination in ("join", "ordinary", "workflow_output"):
            declaration = _map_decl(max_children=maximum, outward_scalar=destination)
            count = 0 if maximum == 0 else 1
            cases.append(
                _case(
                    "map",
                    f"resolve_{destination}_max_{maximum}",
                    declaration,
                    [
                        {
                            "destination": destination,
                            "kind": "resolve_scalar",
                            "maximum": maximum,
                            "member_count": count,
                            "source_kind": "map",
                        }
                    ],
                )
            )
    for destination in ("join", "ordinary", "workflow_output"):
        cases.append(
            _case(
                "map",
                f"resolve_{destination}_max_1_empty",
                _map_decl(max_children=1, outward_scalar=destination),
                [
                    {
                        "destination": destination,
                        "kind": "resolve_scalar",
                        "maximum": 1,
                        "member_count": 0,
                        "source_kind": "map",
                    }
                ],
            )
        )
    for name, outcomes in (
        ("loop_exit", ["again", "exit"]),
        ("loop_bypass", []),
        ("loop_prior_continue", ["again"]),
        ("loop_failure", ["again", "failure"]),
        ("loop_overflow", ["again", "again"]),
    ):
        cases.append(
            _case(
                "map",
                name,
                _map_decl(),
                [{"kind": "resolve_scalar", "outcomes": outcomes, "source_kind": "loop"}],
            )
        )
    for name, maximum, values in (
        ("collection_items_exact", 2, ("a", "b")),
        ("collection_items_one_over", 1, ("a", "b")),
        ("overflow_collection_storage_one_over", 2, ("a", "b", "c")),
        ("overflow_collection_storage_exact", 3, ("a", "b", "c")),
    ):
        declaration = _map_decl()
        _object(declaration["limits"])["max_collection_items"] = maximum
        cases.append(_case("map", name, declaration, [_map_event(values)]))
    invalid_limit = _map_decl()
    _object(invalid_limit["limits"])["max_collection_items"] = True
    cases.append(_case("map", "collection_items_invalid_limit", invalid_limit, [], "admission"))
    # Historical parser-only ID is an explicit redundant alias of the real rejection.
    prospective = next(case for case in cases if case["case_id"] == "map/prospective_transition_rejected")
    cases.append(
        _case(
            "map",
            "caller_transition_verdict_rejected",
            _object(prospective["declaration"]),
            [_object(event) for event in _array(prospective["events"])],
        )
    )
    return cases


def _binding_correction_specs() -> list[Object]:
    cases: list[Object] = []
    prior = [
        _bind("D0", "P0"),
        _reserve("R0", ["D0"], purpose="initial_binding"),
        _dispatch("R0"),
        _source_result(
            "D0",
            "S0",
            [{"association": "D0", "key": 0, "text": "kept", "version": 1}],
            request="R0",
        ),
    ]
    cases.append(
        _case(
            "binding",
            "explicit_omission_preserves_prior",
            _binding_decl({"D0": "S0", "D1": "S1"}, optional=("D1",)),
            [
                *prior,
                _bind("D1", "P0"),
                _reserve("R1", ["D1"], purpose="initial_binding"),
                _dispatch("R1"),
                _source_failure("D1", "S1", request="R1", disposition="omitted_optional"),
                {"kind": "binding_finish"},
            ],
        )
    )
    misuse: tuple[tuple[str, Object, Object], ...] = (
        (
            "required_omission_misuse",
            _binding_decl({"D0": "S0"}, max_requests=1),
            _source_failure("D0", "S0", request="R0", disposition="omitted_optional"),
        ),
        (
            "omission_failure_mismatch",
            _binding_decl({"D0": "S0"}, optional=("D0",)),
            dict(
                _source_failure("D0", "S0", request="R0", failure="retryable", disposition="omitted_optional"),
                kind="source_failure_constructor",
            ),
        ),
    )
    for name, declaration, event in misuse:
        cases.append(
            _case(
                "binding",
                name,
                declaration,
                [*_trace(("D0",)), event],
            )
        )
    adaptive_spec = _materialization_spec("adaptive", "collection")
    adaptive_declaration = _materialization_decl(adaptive_spec, max_requests=1)
    adaptive_declaration["retrieval_sources"] = {"A0": "S0"}
    adaptive_prefix = _materialization_trace(adaptive_spec, _items("selector"))[:-1]
    for event in adaptive_prefix:
        if event["kind"] == "reserve":
            event["purpose"] = "adaptive_retrieval"
    cases.append(
        _case(
            "binding",
            "adaptive_omission_misuse",
            adaptive_declaration,
            [*adaptive_prefix, _source_failure("A0", "S0", request="R0", disposition="omitted_optional")],
        )
    )
    for name, failure, purpose in (
        ("source_failure_retry_authority", "retryable", "retry"),
        ("source_failure_correction_authority", "malformed_response", "correction"),
    ):
        cases.append(
            _case(
                "binding",
                name,
                _binding_decl({"D0": "S0"}),
                [
                    *_trace(("D0",)),
                    _source_failure("D0", "S0", request="R0", failure=failure),
                    _reserve("R1", ["D0"], purpose=purpose),
                ],
            )
        )
    missing_failure = _source_failure("D0", "S0", request="R0")
    missing_failure.pop("failure")
    missing_failure["kind"] = "source_failure_constructor"
    missing_settlement = _source_failure("D0", "S0", request="R0")
    missing_settlement.pop("settlement")
    missing_settlement["kind"] = "source_failure_constructor"
    cases.extend(
        [
            _case(
                "binding",
                "source_failure_missing_failure",
                _binding_decl({"D0": "S0"}),
                [*_trace(("D0",)), missing_failure],
            ),
            _case(
                "binding",
                "source_failure_missing_settlement",
                _binding_decl({"D0": "S0"}),
                [*_trace(("D0",)), missing_settlement],
            ),
        ]
    )
    no_settlement = _source_failure("D0", "S0", request="R0")
    no_settlement["settlement"] = None
    cases.append(
        _case(
            "binding",
            "source_failure_explicit_no_settlement",
            _binding_decl({"D0": "S0"}),
            [*_trace(("D0",)), no_settlement],
        )
    )
    base_result = _source_result(
        "D0",
        "S0",
        [{"association": "D0", "key": 0, "text": "a", "version": 1}],
        request="R0",
    )
    for name, changes in (
        ("source_result_wrong_outcome", {"outcome": "ok"}),
        ("source_result_outputs_present", {"outputs": [{"port": "x"}]}),
        ("source_result_consumed_present", {"consumed_context_ports": ["x"]}),
    ):
        cases.append(
            _case(
                "binding",
                name,
                _binding_decl({"D0": "S0"}),
                [
                    *_trace(("D0",)),
                    {
                        "kind": "binding_result",
                        "association": "D0",
                        "outcome": changes.get("outcome", "retrieved"),
                        "outputs": changes.get("outputs", []),
                        "consumed_context_ports": changes.get("consumed_context_ports", []),
                    },
                ],
            )
        )
    cases.append(
        _case(
            "binding",
            "empty_optional_response_malformed",
            _binding_decl({"D0": "S0"}, optional=("D0",), max_requests=1),
            [*_trace(("D0",)), _source_result("D0", "S0", [], request="R0")],
        )
    )
    oversize = _source_result(
        "D0",
        "S0",
        [{"association": "D0", "key": 0, "text": "too-big", "version": 1}],
        request="R0",
    )
    _object(oversize["settlement"])["usage"] = {"input": 3, "output": 5}
    cases.append(
        _case(
            "binding",
            "oversize_retrieved_known_usage",
            _binding_decl({"D0": "S0"}, max_bytes=1),
            [*_trace(("D0",)), oversize],
        )
    )
    cases.append(
        _case(
            "binding",
            "success_after_failure_preserves_authority",
            _binding_decl({"D0": "S0"}),
            [
                *_trace(("D0",)),
                _source_failure("D0", "S0", request="R0", failure="retryable"),
                base_result,
            ],
        )
    )
    cases.append(
        _case(
            "binding",
            "adaptive_semantic_outcome_independent",
            _decl(),
            [
                *_trace(("A0",)),
                {"kind": "result", "outcomes": {"A0": "adaptive_ok"}, "request": "R0", "returned": ["A0"]},
            ],
        )
    )
    return cases


def _latest_selection_specs() -> list[Object]:
    """Exercise latest selection through the complete binding/request owner chain."""
    cases: list[Object] = []
    latest = _materialization_spec("initial", "single", declaration="D0", version_selection="latest")

    def declaration(
        *specs: Object,
        limits: Mapping[str, int] | None = None,
        max_requests: int = 3,
        optional: bool = False,
    ) -> Object:
        value = _materialization_decl(*specs, max_requests=max_requests, limit_changes=limits)
        if optional:
            value["binding_requirements"] = {cast(str, spec["association"]): "optional" for spec in specs}
        return value

    def cleanup(owner: str = "sdk", disposition: str = "closed", *, association: str = "D0") -> list[Object]:
        return [
            {
                "association": association,
                "kind": "binding_cleanup_association",
                "owner": owner,
                "resource": f"Q:{association}",
                "target": "T0",
            },
            {"disposition": disposition, "kind": "binding_cleanup", "resource": f"Q:{association}"},
        ]

    def trace(items: Sequence[Object], *, spec: Object = latest, request: str = "R0") -> list[Object]:
        association = cast(str, spec["association"])
        return [
            _bind(association, "P0"),
            _reserve(request, [association], purpose="initial_binding"),
            _dispatch(request),
            _materialization_event(spec, items, request=request),
            {"kind": "binding_finish"},
            *cleanup(association=association),
            _operation_start(spec),
            _publication_event(spec),
        ]

    one: list[Object] = [{"key": 0, "value": "one", "version": 1}]
    two: list[Object] = [
        {"key": 0, "value": "old", "version": 1},
        {"key": 0, "value": "new", "version": 2},
    ]
    gap: list[Object] = [
        {"key": 0, "value": "old", "version": 1},
        {"key": 0, "value": "new", "version": 3},
    ]
    for name, items in (
        ("one_version", one),
        ("two_versions", two),
        ("reordered_versions", tuple(reversed(two))),
        ("version_gap", gap),
    ):
        cases.append(_case("materialization", f"latest_{name}", declaration(latest), trace(items)))

    duplicate: list[Object] = [two[0], dict(two[0])]
    multikey: list[Object] = [two[0], {"key": 1, "value": "other", "version": 2}]
    for name, items in (("duplicate_pair", duplicate), ("multiple_keys", multikey)):
        cases.append(
            _case(
                "materialization",
                f"latest_{name}",
                declaration(latest, max_requests=1),
                [*trace(items)[:4], *cleanup()],
            )
        )
    items_one_over: list[Object] = [{"key": 0, "value": str(index), "version": index + 1} for index in range(4)]
    items_exact: list[Object] = items_one_over[:3]
    cases.append(
        _case(
            "materialization",
            "latest_items_exact",
            declaration(latest),
            trace(items_exact),
        )
    )
    cases.append(
        _case(
            "materialization",
            "latest_items_one_over",
            declaration(latest),
            [*trace(items_one_over)[:4], *cleanup()],
        )
    )
    bytes_one_over: list[Object] = [{"key": 0, "value": "x" * 13, "version": 1}]
    bytes_exact: list[Object] = [{"key": 0, "value": "x" * 12, "version": 1}]
    cases.append(
        _case(
            "materialization",
            "latest_bytes_exact",
            declaration(latest),
            trace(bytes_exact),
        )
    )
    cases.append(
        _case(
            "materialization",
            "latest_bytes_one_over",
            declaration(latest),
            [*trace(bytes_one_over)[:4], *cleanup()],
        )
    )

    for terminal in ("cancelled", "lost"):
        prefix = trace(two)[:3]
        terminal_events: list[Object] = (
            [
                {"kind": "cancel", "request": "R0"},
                {"kind": "stop", "request": "R0", "usage": {"input": 0, "output": 0}},
            ]
            if terminal == "cancelled"
            else [{"kind": "cancel", "request": "R0"}, {"kind": "lost", "request": "R0"}]
        )
        for shape, items in (("valid", two), ("multikey", multikey)):
            cases.append(
                _case(
                    "materialization",
                    f"latest_{terminal}_late_{shape}",
                    declaration(latest),
                    [*prefix, *terminal_events, _materialization_event(latest, items), *cleanup()],
                )
            )

    retry_prefix: list[Object] = [
        _bind("D0", "P0"),
        _reserve("R0", ["D0"], purpose="initial_binding"),
        _dispatch("R0"),
        _source_failure("D0", "S0", request="R0", failure="retryable"),
        _reserve("R1", ["D0"], purpose="retry"),
        _dispatch("R1"),
        _materialization_event(latest, two, request="R1"),
        {"kind": "binding_finish"},
        *cleanup(),
        _operation_start(latest),
        _publication_event(latest),
    ]
    cases.append(_case("materialization", "latest_retry", declaration(latest), retry_prefix))
    correction_prefix: list[Object] = [
        _bind("D0", "P0"),
        _reserve("R0", ["D0"], purpose="initial_binding"),
        _dispatch("R0"),
        _materialization_event(latest, multikey),
        _reserve("R1", ["D0"], purpose="correction"),
        _dispatch("R1"),
        _materialization_event(latest, two, request="R1"),
        {"kind": "binding_finish"},
        *cleanup(),
        _operation_start(latest),
        _publication_event(latest),
    ]
    cases.append(_case("materialization", "latest_correction", declaration(latest), correction_prefix))
    cases.append(
        _case(
            "materialization",
            "latest_optional_omission",
            declaration(latest, optional=True),
            [
                _bind("D0", "P0"),
                _reserve("R0", ["D0"], purpose="initial_binding"),
                _dispatch("R0"),
                _source_failure("D0", "S0", request="R0", disposition="omitted_optional"),
                *cleanup(),
            ],
        )
    )
    caller_cleanup = trace(one)
    next(event for event in caller_cleanup if event["kind"] == "binding_cleanup_association")["owner"] = "caller"
    next(event for event in caller_cleanup if event["kind"] == "binding_cleanup")["disposition"] = "left_open"
    cases.append(_case("materialization", "latest_caller_cleanup", declaration(latest), caller_cleanup))
    for disposition in ("close_failed", "close_unknown"):
        events = trace(one)
        next(event for event in events if event["kind"] == "binding_cleanup")["disposition"] = disposition
        cases.append(
            _case(
                "materialization",
                f"latest_sdk_cleanup_{disposition}",
                declaration(latest),
                events,
            )
        )
    optional = _materialization_spec(
        "initial",
        "single",
        declaration="D1",
        port="optional_context",
        version_selection="exact_one",
    )
    mixed_declaration = declaration(latest, optional)
    mixed_declaration["binding_requirements"] = {"D0": "required", "D1": "optional"}
    mixed_events: list[Object] = [
        _bind("D0", "P0"),
        _bind("D1", "P0"),
        _reserve("R0", ["D0"], purpose="initial_binding"),
        _dispatch("R0"),
        _materialization_event(latest, one),
        _reserve("R1", ["D1"], purpose="initial_binding"),
        _dispatch("R1"),
        _source_failure("D1", "S0", request="R1", disposition="omitted_optional"),
        {"kind": "binding_finish"},
        *cleanup(association="D0"),
        *cleanup(association="D1"),
        _operation_start(latest),
        _publication_event(latest),
    ]
    cases.append(
        _case(
            "materialization",
            "latest_bound_with_optional_omission",
            mixed_declaration,
            mixed_events,
        )
    )
    unresolved = [
        deepcopy(event)
        for event in mixed_events
        if not (event.get("request") == "R1" or event.get("kind") == "reserve" and event.get("associations") == ["D1"])
    ]
    cases.append(
        _case(
            "materialization",
            "latest_unresolved_optional_at_finish",
            mixed_declaration,
            unresolved,
        )
    )
    late_failure = deepcopy(mixed_events)
    finish_index = next(i for i, event in enumerate(late_failure) if event["kind"] == "binding_finish")
    late_failure.insert(
        finish_index + 1,
        deepcopy(next(event for event in late_failure if event["kind"] == "source_failure")),
    )
    cases.append(
        _case(
            "materialization",
            "latest_post_finish_source_failure",
            mixed_declaration,
            late_failure,
        )
    )
    late_result = deepcopy(trace(one))
    finish_index = next(i for i, event in enumerate(late_result) if event["kind"] == "binding_finish")
    late_result.insert(
        finish_index + 1,
        deepcopy(next(event for event in late_result if event["kind"] == "materialize_result")),
    )
    cases.append(
        _case(
            "materialization",
            "latest_post_finish_materialization",
            declaration(latest),
            late_result,
        )
    )
    for name, maximum in (("exact", 1), ("one_over", 0)):
        cases.append(
            _case(
                "materialization",
                f"latest_provenance_edges_{name}",
                declaration(latest, limits={"max_provenance_edges": maximum}),
                trace(one)[:4],
                "execution_preflight",
            )
        )

    rollback_events: list[Object] = [
        {"artifact_type": "text", "kind": "root_artifact", "port": "left", "target": "T0", "value": "L"},
        {"artifact_type": "text", "kind": "root_artifact", "port": "right", "target": "T0", "value": "R"},
        _bind("D0", "P0"),
        _reserve("R0", ["D0"], purpose="initial_binding"),
        _dispatch("R0"),
        _materialization_event(latest, two),
        {"kind": "binding_finish"},
        *cleanup(association="D0"),
        _operation_start(latest),
        _publication_event(latest, value="x" * 9),
        _operation_start(latest, activation="OP:D1"),
        _publication_event(latest, activation="OP:D1", value="ok"),
    ]
    rollback_decl = declaration(latest, limits={"max_artifact_bytes": 16, "max_artifacts": 5})
    _array(rollback_decl["operation_occurrences"]).append(
        {
            "activation": "OP:D1",
            "attempt": "TASK:OP:D1",
            "binding_declaration": "D0",
            "node": "N0",
            "target": "T0",
        }
    )
    cases.append(
        _case(
            "materialization",
            "latest_publication_rollback_then_reuse",
            rollback_decl,
            rollback_events,
        )
    )
    return cases


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
    both_mappings = _decl()
    result_mapping = _object(
        _array(_runtime_decl("result", "ok", "success", reported_outcome="ok")["runtime_mappings"])[0]
    )
    inconsistent_mapping = _object(
        _array(_runtime_decl("request_inconsistent", None, "inconsistent")["runtime_mappings"])[0]
    )
    both_mappings.update(
        {
            "declared_outcomes": ["ok"],
            "outcome_categories": {"ok": "success"},
            "runtime_mappings": [result_mapping, inconsistent_mapping],
        }
    )
    c += [
        _case(
            "bridges",
            "inconsistent_request_cannot_emit_success",
            both_mappings,
            _trace(("T0", "T1"))
            + [
                {"kind": "bridge_start", "node": "N0", "task": "T0"},
                _result("R0", ("T0", "T1"), ("T0",)),
                {"condition": "result", "kind": "bridge_condition", "reported_outcome": "ok", "task": "T0"},
            ],
        ),
        _case(
            "bridges",
            "valid_request_cannot_emit_inconsistent",
            both_mappings,
            _trace(("T0", "T1"))
            + [
                {"kind": "bridge_start", "node": "N0", "task": "T0"},
                _result("R0", ("T0", "T1"), ("T0", "T1")),
                {"condition": "request_inconsistent", "kind": "bridge_condition", "task": "T0"},
            ],
        ),
    ]
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
                _reserve("R1", ["T0", "T1"]),
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
        _case(
            "retry",
            "cross_association_predecessor",
            _decl(),
            _trace()
            + [
                {"failure": "retryable", "kind": "failure", "request": "R0"},
                _bind("T1", "P0"),
                _reserve("R1", ["T1"], purpose="retry"),
            ],
        ),
        _case(
            "retry",
            "implementation_owner_rejected",
            {
                "execution_policies": [
                    {
                        "implementations": [{"physical_policy": "P0"}],
                        "kind": "external",
                        "node": "N0",
                        "physical_policy": "P0",
                        "retry_owner": "implementation",
                    }
                ]
            },
            [],
            "admission",
        ),
        _case(
            "retry",
            "retry_after_permanent",
            _decl(),
            _trace()
            + [
                {"failure": "permanent", "kind": "failure", "request": "R0"},
                _reserve("R1", ["T0"], purpose="retry"),
            ],
        ),
        _case(
            "retry",
            "correction_after_retryable",
            _decl(),
            _trace()
            + [
                {"failure": "retryable", "kind": "failure", "request": "R0"},
                _reserve("R1", ["T0"], purpose="correction"),
            ],
        ),
        _case(
            "retry",
            "failover_after_malformed",
            _decl(),
            _trace()
            + [
                {"failure": "malformed_response", "kind": "failure", "request": "R0"},
                _reserve("R1", ["T0"], purpose="failover"),
            ],
        ),
        _case(
            "retry",
            "late_failure_does_not_change_authority",
            _decl(),
            _trace()
            + [
                {"failure": "permanent", "kind": "failure", "request": "R0"},
                {"failure": "retryable", "kind": "failure", "request": "R0"},
                _reserve("R1", ["T0"], purpose="retry"),
            ],
        ),
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
            base + [{"kind": "cancel", "request": "R0"}, {"kind": "stop", "request": "R0", "usage": "unknown"}],
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
            "lost_late_result_without_settlement",
            _decl(),
            base + [{"kind": "lost", "request": "R0"}, _result("R0", ("T0",), ("T0",))],
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
            base
            + [
                {"kind": "lost", "request": "R0"},
                dict(settlement, disposition="unknown", remote_stopped=None, usage="unknown"),
            ],
        ),
        _case(
            "races",
            "cancel_stop_late_result",
            _decl(),
            base
            + [
                {"kind": "cancel", "request": "R0"},
                {"kind": "stop", "request": "R0", "usage": "unknown"},
                _result("R0", ("T0",), ("T0",)),
            ],
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
        _case(
            "races",
            "stop_missing_usage",
            _decl(),
            base + [{"kind": "cancel", "request": "R0"}, {"kind": "stop", "request": "R0"}],
        ),
        _case(
            "races", "settlement_completed_without_remote_stop", _decl(), base + [dict(settlement, remote_stopped=None)]
        ),
        _case(
            "races", "settlement_unknown_with_remote_stop", _decl(), base + [dict(settlement, disposition="unknown")]
        ),
        _case(
            "races",
            "settlement_invalid_remote_stopped_type",
            _decl(),
            base + [dict(settlement, disposition="unknown", remote_stopped="invalid", usage="unknown")],
        ),
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
            base
            + [
                {"kind": "lost", "request": "R0"},
                dict(settlement, disposition="unknown", remote_stopped=None, usage="unknown"),
            ],
        ),
        _case("inflight", "result_clears_both", _decl(), base + [_result("R0", ("T0",), ("T0",))]),
    ]

    def binding(identity: str, source: str, request: str, text: str) -> list[Object]:
        return [
            _bind(identity, "P0"),
            _reserve(request, [identity], purpose="initial_binding"),
            _dispatch(request),
            _source_result(
                identity,
                source,
                [{"association": identity, "key": 0, "text": text, "version": 1}],
                request=request,
            ),
        ]

    c += [
        _case(
            "binding",
            "two_sources_same_key",
            _binding_decl({"D0": "S0", "D1": "S1"}),
            binding("D0", "S0", "R0", "a") + binding("D1", "S1", "R1", "b") + [{"kind": "binding_finish"}],
        ),
        _case(
            "binding",
            "one_source_two_declarations",
            _binding_decl({"D0": "S0", "D1": "S0"}),
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
            _binding_decl({"D0": "S0", "D1": "S1"}),
            binding("D0", "S0", "R0", "a")
            + [
                _bind("D1", "P0"),
                _reserve("R1", ["D1"], purpose="initial_binding"),
                _dispatch("R1"),
                _source_failure("D1", "S1", request="R1"),
            ],
        ),
        _case(
            "binding",
            "optional_failure_partial",
            _binding_decl({"D0": "S0", "D1": "S1"}, optional=("D1",)),
            binding("D0", "S0", "R0", "a")
            + [
                _bind("D1", "P0"),
                _reserve("R1", ["D1"], purpose="initial_binding"),
                _dispatch("R1"),
                _source_failure("D1", "S1", request="R1"),
            ],
        ),
        _case(
            "binding",
            "omitted_optional",
            _binding_decl({"D0": "S0"}, optional=("D0",)),
            [
                _bind("D0", "P0"),
                _reserve("R0", ["D0"], purpose="initial_binding"),
                _dispatch("R0"),
                _source_failure("D0", "S0", request="R0", disposition="omitted_optional"),
                {"kind": "binding_finish"},
            ],
        ),
        _case(
            "binding",
            "response_reordered",
            _binding_decl({"D0": "S0", "D1": "S1"}),
            binding("D1", "S1", "R1", "b") + binding("D0", "S0", "R0", "a") + [{"kind": "binding_finish"}],
        ),
        _case(
            "binding",
            "oversize_no_truncation",
            _binding_decl({"D0": "S0"}, max_bytes=1),
            binding("D0", "S0", "R0", "aa"),
        ),
        _case(
            "binding",
            "cancel_before_dispatch",
            _binding_decl({"D0": "S0"}),
            [_bind("D0", "P0"), _reserve("R0", ["D0"], purpose="initial_binding"), {"kind": "cancel", "request": "R0"}],
        ),
        _case(
            "binding",
            "cancel_after_dispatch_lost",
            _binding_decl({"D0": "S0"}),
            binding("D0", "S0", "R0", "a")[:3]
            + [{"kind": "cancel", "request": "R0"}, {"kind": "lost", "request": "R0"}],
        ),
        _case(
            "binding",
            "invalid_local_zero_effects",
            {"binding_targets": ["B0"], "targets": ["A0"]},
            [],
            "admission",
        ),
    ]
    exact_bounds = _binding_decl({"D0": "S0"}, max_bytes=2, max_items=2)
    one_byte = _binding_decl({"D0": "S0"}, max_bytes=1, max_items=2)
    one_item = _binding_decl({"D0": "S0"}, max_bytes=2, max_items=1)
    two_items = _source_result(
        "D0",
        "S0",
        [
            {"association": "D0", "key": 0, "text": "a", "version": 1},
            {"association": "D0", "key": 1, "text": "b", "version": 1},
        ],
        request="R0",
    )
    c += [
        _case(
            "binding",
            "exact_item_byte_bounds",
            exact_bounds,
            [_bind("D0", "P0"), _reserve("R0", ["D0"]), _dispatch("R0"), two_items, {"kind": "binding_finish"}],
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
        _case(
            "binding",
            "unsolicited_source_result",
            _binding_decl({"D0": "S0"}),
            [dict(_result("R0", ("D0",), ("D0",)), outcomes={"D0": "retrieved"})],
        ),
        _case(
            "binding",
            "wrong_source",
            _binding_decl({"D0": "S0"}, max_requests=1),
            [
                _bind("D0", "P0"),
                _reserve("R0", ["D0"]),
                _dispatch("R0"),
                _source_result("D0", "S1", [{"association": "D0", "key": 0, "version": 1, "text": "a"}], request="R0"),
                {"kind": "binding_finish"},
            ],
        ),
        _case(
            "binding",
            "missing_result",
            _binding_decl({"D0": "S0"}, max_requests=1),
            [
                _bind("D0", "P0"),
                _reserve("R0", ["D0"]),
                _dispatch("R0"),
                _source_result("D0", "S0", [], request="R0"),
            ],
        ),
        _case(
            "binding",
            "duplicate_result",
            _binding_decl({"D0": "S0"}, max_requests=1),
            [
                _bind("D0", "P0"),
                _reserve("R0", ["D0"]),
                _dispatch("R0"),
                _source_result(
                    "D0",
                    "S0",
                    [
                        {"association": "D0", "key": 0, "text": "a", "version": 1},
                        {"association": "D0", "key": 0, "text": "a", "version": 1},
                    ],
                    request="R0",
                ),
            ],
        ),
        _case(
            "binding",
            "foreign_result_association",
            _binding_decl({"D0": "S0"}, max_requests=1),
            [
                _bind("D0", "P0"),
                _reserve("R0", ["D0"]),
                _dispatch("R0"),
                _source_result(
                    "D0",
                    "S0",
                    [{"association": "D1", "key": 0, "text": "a", "version": 1}],
                    request="R0",
                ),
            ],
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
            events += [
                {"kind": "cancel", "request": "R0"},
                {"kind": "stop", "request": "R0", "usage": "unknown"},
            ]
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
                _runtime_decl(condition, outcome, category, reported_outcome="ok" if condition == "result" else None),
                [
                    {"kind": "bridge_start", "node": "N0", "task": "T0"},
                    {
                        "condition": condition,
                        "kind": "bridge_condition",
                        **({"reported_outcome": "ok"} if condition == "result" else {}),
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
                _runtime_decl("failure", None, "failure", failure=failure),
                [
                    {"kind": "bridge_start", "node": "N0", "task": "T0"},
                    {
                        "condition": "failure",
                        "failure": failure,
                        "kind": "bridge_condition",
                        "task": "T0",
                    },
                    {"category": "failure", "kind": "bridge_emit", "outcome": None, "task": "T0"},
                ],
            )
        )
    shared_bridge = _decl()
    shared_bridge.update(_runtime_decl("result", "ok", "success", reported_outcome="ok"))
    c.append(
        _case(
            "bridges",
            "shared_request_two_targets",
            shared_bridge,
            _trace(("T0", "T1"))
            + [
                {"kind": "bridge_start", "node": "N0", "task": "T0"},
                {"kind": "bridge_start", "node": "N1", "task": "T1"},
                _result("R0", ("T0", "T1"), ("T1", "T0")),
                {"condition": "result", "kind": "bridge_condition", "reported_outcome": "ok", "task": "T0"},
                {"condition": "result", "kind": "bridge_condition", "reported_outcome": "ok", "task": "T1"},
                {"category": "success", "kind": "bridge_emit", "outcome": "ok", "task": "T0"},
                {"category": "success", "kind": "bridge_emit", "outcome": "ok", "task": "T1"},
            ],
        )
    )
    c.append(
        _case(
            "bridges",
            "shared_request_start_before_dispatch",
            shared_bridge,
            [
                {"kind": "bridge_start", "node": "N0", "task": "T0"},
                {"kind": "bridge_start", "node": "N1", "task": "T1"},
                *_trace(("T0", "T1")),
                _result("R0", ("T0", "T1"), ("T0", "T1")),
                {"condition": "result", "kind": "bridge_condition", "reported_outcome": "ok", "task": "T0"},
                {"condition": "result", "kind": "bridge_condition", "reported_outcome": "ok", "task": "T1"},
                {"category": "success", "kind": "bridge_emit", "outcome": "ok", "task": "T0"},
                {"category": "success", "kind": "bridge_emit", "outcome": "ok", "task": "T1"},
            ],
        )
    )
    retry_bridge = _decl()
    retry_bridge.update(_runtime_decl("result", "ok", "success", reported_outcome="ok"))
    c.append(
        _case(
            "bridges",
            "retry_uses_latest_physical_request",
            retry_bridge,
            [
                {"kind": "bridge_start", "node": "N0", "task": "T0"},
                *_trace(),
                {"failure": "retryable", "kind": "failure", "request": "R0"},
                _reserve("R1", ["T0"], purpose="retry"),
                _dispatch("R1"),
                _result("R1", ("T0",), ("T0",)),
                {"condition": "result", "kind": "bridge_condition", "reported_outcome": "ok", "task": "T0"},
                {"category": "success", "kind": "bridge_emit", "outcome": "ok", "task": "T0"},
            ],
        )
    )
    for name, returned in (
        ("missing", ("T0",)),
        ("duplicate", ("T0", "T0", "T1")),
        ("extra", ("T0", "T1", "T2")),
        ("foreign", ("T0", "T1", "X0")),
    ):
        inconsistent = _decl()
        inconsistent.update(_runtime_decl("request_inconsistent", None, "inconsistent"))
        c.append(
            _case(
                "bridges",
                f"shared_request_{name}",
                inconsistent,
                _trace(("T0", "T1"))
                + [
                    {"kind": "bridge_start", "node": "N0", "task": "T0"},
                    {"kind": "bridge_start", "node": "N1", "task": "T1"},
                    _result("R0", ("T0", "T1"), returned),
                    {"condition": "request_inconsistent", "kind": "bridge_condition", "task": "T0"},
                    {"condition": "request_inconsistent", "kind": "bridge_condition", "task": "T1"},
                    {"category": "inconsistent", "kind": "bridge_emit", "outcome": None, "task": "T0"},
                    {"category": "inconsistent", "kind": "bridge_emit", "outcome": None, "task": "T1"},
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
            _runtime_decl("result", "ok", "success", reported_outcome="ok"),
            [
                *opened,
                {"kind": "bridge_start", "node": "N1", "task": "T1"},
                {
                    "condition": "result",
                    "kind": "bridge_condition",
                    "reported_outcome": "ok",
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
        ("foreign_wait", {"wait": "W1", "invocation": "I1"}),
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
            _runtime_decl("cancel_after_start", None, "cancelled"),
            [
                {"kind": "bridge_start", "node": "N0", "task": "T0"},
                {
                    "condition": "cancel_after_start",
                    "kind": "bridge_condition",
                    "task": "T0",
                },
                {"category": "cancelled", "kind": "bridge_emit", "outcome": None, "task": "T0"},
            ],
        ),
        _case(
            "decisions",
            "implementation_failure",
            _runtime_decl("failure", None, "failure", failure="implementation_exception"),
            [
                {"kind": "bridge_start", "node": "N0", "task": "T0"},
                {
                    "condition": "failure",
                    "failure": "implementation_exception",
                    "kind": "bridge_condition",
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
    valid_policy: Object = {
        "implementations": [{"physical_policy": "P0"}],
        "kind": "external",
        "node": "N0",
        "physical_policy": "P0",
        "result_outcomes": ["ok"],
        "retry_owner": "executor",
    }
    result_mapping = _object(
        _array(_runtime_decl("result", "ok", "success", reported_outcome="ok")["runtime_mappings"])[0]
    )
    missing_runtime = _policy_runtime_decl("external")
    missing_runtime["execution_policies"] = [valid_policy]
    missing_runtime["runtime_mappings"] = [
        raw for raw in _array(missing_runtime["runtime_mappings"]) if _object(raw)["condition"] != "result"
    ]
    duplicate_condition = _policy_runtime_decl("external")
    duplicate_condition["execution_policies"] = [valid_policy]
    blocked_row = next(
        _object(raw)
        for raw in _array(duplicate_condition["runtime_mappings"])
        if _object(raw)["condition"] == "cancel_before_start"
    )
    _array(duplicate_condition["runtime_mappings"]).append(dict(blocked_row, category="cancelled"))
    primary_capability: Object = {"implementation": "C0", "physical_policy": "P0"}
    alternate_capability: Object = {"implementation": "C1", "physical_policy": "P0"}
    admission_negatives: tuple[tuple[str, Object], ...] = (
        ("outer_type", {"execution_policies": "invalid"}),
        (
            "aggregate_limit",
            {
                "admission_limits": {"max_capabilities": 1},
                "capability_catalog": [primary_capability, alternate_capability],
            },
        ),
        ("member_type", {"execution_policies": [0]}),
        ("invalid_value", {"execution_policies": [dict(valid_policy, kind="unknown")]}),
        ("foreign_owner", {"execution_policies": [dict(valid_policy, owner="I1")]}),
        ("duplicate", {"execution_policies": [valid_policy, valid_policy]}),
        ("missing", {"execution_policies": [dict(valid_policy, implementations=[])]}),
        ("unsupported", {"execution_policies": [dict(valid_policy, retry_owner="implementation")]}),
        (
            "contradictory",
            {
                "declared_outcomes": ["ok"],
                "outcome_categories": {"ok": "success"},
                "runtime_mappings": [dict(result_mapping, category="failure")],
            },
        ),
        (
            "runtime_mapping_missing",
            missing_runtime,
        ),
        (
            "wrong_category",
            {
                "declared_outcomes": ["ok"],
                "outcome_categories": {"ok": "success"},
                "runtime_mappings": [dict(result_mapping, category="blocked")],
            },
        ),
        (
            "unknown_outcome",
            {
                "declared_outcomes": ["ok"],
                "runtime_mappings": [dict(result_mapping, outcome="other", reported_outcome="other")],
            },
        ),
        (
            "duplicate_runtime_mapping",
            {
                "declared_outcomes": ["ok"],
                "outcome_categories": {"ok": "success"},
                "runtime_mappings": [result_mapping, result_mapping],
            },
        ),
        (
            "duplicate_required_condition",
            duplicate_condition,
        ),
    )
    for name, declaration in admission_negatives:
        c.append(_case("admission", name, declaration, [], "admission"))
    c += [
        _case(
            "admission",
            "valid_external_failover",
            {
                **_policy_runtime_decl("external"),
                "execution_policies": [
                    {
                        "implementations": [{"physical_policy": "P0"}, {"physical_policy": "P0"}],
                        "kind": "external",
                        "node": "N0",
                        "physical_policy": "P0",
                        "result_outcomes": ["ok"],
                        "retry_owner": "executor",
                    }
                ],
            },
            [],
            "admission",
        ),
        _case(
            "admission",
            "changed_failover_policy",
            {
                "admitted": {
                    **_policy_runtime_decl("external"),
                    "execution_policies": [
                        dict(valid_policy, implementations=[{"physical_policy": "P0"}, {"physical_policy": "P0"}])
                    ],
                    "capability_catalog": [primary_capability, alternate_capability],
                },
                "capability_catalog": [primary_capability, dict(alternate_capability, physical_policy="P1")],
            },
            [],
            "pre_execution",
        ),
        _case(
            "admission",
            "local_multiple_implementations",
            {
                "execution_policies": [
                    {
                        "implementations": [{"physical_policy": None}, {"physical_policy": None}],
                        "kind": "local",
                        "node": "N0",
                        "physical_policy": None,
                        "result_outcomes": ["ok"],
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
                "admission_limits": {"max_capabilities": 1},
                "capability_catalog": [primary_capability, primary_capability],
            },
            [],
            "admission",
        ),
        _case(
            "admission",
            "valid_local_runtime_product",
            {
                **_policy_runtime_decl("local"),
                "execution_policies": [
                    {
                        "implementations": [{"physical_policy": None}],
                        "kind": "local",
                        "node": "N0",
                        "physical_policy": None,
                        "result_outcomes": ["ok"],
                        "retry_owner": "none",
                    }
                ],
            },
            [],
            "admission",
        ),
        _case(
            "admission",
            "valid_decision_runtime_product",
            {
                **_policy_runtime_decl("decision"),
                "execution_policies": [
                    {
                        "implementations": [{"physical_policy": None}],
                        "kind": "decision",
                        "node": "N0",
                        "physical_policy": None,
                        "result_outcomes": [],
                        "retry_owner": "none",
                    }
                ],
            },
            [],
            "admission",
        ),
    ]
    external_policy: Object = {
        "implementations": [{"physical_policy": "P0"}],
        "kind": "external",
        "node": "N0",
        "physical_policy": "P0",
        "result_outcomes": ["ok"],
        "retry_owner": "executor",
    }
    missing_product = _policy_runtime_decl("external")
    missing_product["execution_policies"] = [external_policy]
    missing_product["runtime_mappings"] = [
        value for value in _array(missing_product["runtime_mappings"]) if _object(value).get("condition") != "result"
    ]
    extra_product = _policy_runtime_decl("local")
    extra_product["execution_policies"] = [
        {
            "implementations": [{"physical_policy": None}],
            "kind": "local",
            "node": "N0",
            "physical_policy": None,
            "result_outcomes": ["ok"],
            "retry_owner": "none",
        }
    ]
    extra_product["runtime_mappings"] = [
        *_array(extra_product["runtime_mappings"]),
        _object(_array(_runtime_decl("lost", None, "lost")["runtime_mappings"])[0]),
    ]
    duplicate_outcome = _policy_runtime_decl("external")
    duplicate_outcome["execution_policies"] = [external_policy]
    _array(duplicate_outcome["runtime_mappings"]).append(dict(result_mapping, category="failure"))
    c += [
        _case("admission", "runtime_product_missing_result", missing_product, [], "admission"),
        _case("admission", "runtime_product_extra_local_condition", extra_product, [], "admission"),
        _case("admission", "duplicate_result_outcome", duplicate_outcome, [], "admission"),
    ]
    for replay in ("never", "before_acceptance", "idempotent"):
        failover_decl = _decl()
        _object(failover_decl["policies"])["P0"] = _policy(replay=replay)
        c.append(
            _case(
                "retry",
                f"failover_permanent_replay_{replay}",
                failover_decl,
                [
                    *_trace(),
                    {"kind": "failure", "request": "R0", "failure": "permanent"},
                    _reserve("R1", ["T0"], purpose="failover"),
                ],
            )
        )
    c.append(
        _case(
            "races",
            "lost_late_failure_without_settlement",
            _decl(),
            [
                *_trace(),
                {"kind": "lost", "request": "R0"},
                {"kind": "failure", "request": "R0", "failure": "retryable"},
            ],
        )
    )
    c.append(
        _case(
            "races",
            "lost_conflicting_settlement_preserves_remote",
            _decl(),
            [
                *_trace(),
                {"kind": "lost", "request": "R0"},
                dict(settlement, disposition="unknown", remote_stopped=None, usage="unknown"),
                settlement,
            ],
        )
    )
    late_binding_responses: tuple[Object, ...] = (
        _source_result(
            "D0",
            "S0",
            [{"association": "D0", "key": 0, "version": 1, "text": "late"}],
            request="R0",
        ),
        _source_failure("D0", "S0", request="R0"),
    )
    for terminal in ("lost", "cancelled"):
        closure: list[Object] = (
            [{"kind": "lost", "request": "R0"}]
            if terminal == "lost"
            else [
                {"kind": "cancel", "request": "R0"},
                {"kind": "stop", "request": "R0", "usage": "unknown"},
            ]
        )
        if terminal == "lost":
            closure.insert(0, {"kind": "cancel", "request": "R0"})
        for response in late_binding_responses:
            late_events = [*_trace(("D0",)), *closure, response]
            c.append(
                _case(
                    "binding",
                    f"{terminal}_late_{response['kind']}",
                    _binding_decl({"D0": "S0"}),
                    late_events,
                    traces=[[*late_events, settlement]],
                )
            )
    retry_prefix: list[Object] = [
        *_trace(),
        {"kind": "failure", "request": "R0", "failure": "retryable"},
        _reserve("R1", ["T0"], purpose="retry"),
        _dispatch("R1"),
    ]
    latest_cases: tuple[tuple[str, str, list[Object]], ...] = (
        ("pending", "retry", []),
        ("success", "retry", [_result("R1", ("T0",), ("T0",))]),
        ("permanent", "retry", [{"kind": "failure", "request": "R1", "failure": "permanent"}]),
        ("malformed", "correction", [{"kind": "failure", "request": "R1", "failure": "malformed_response"}]),
    )
    for latest, continuation, terminal_events in latest_cases:
        c.append(
            _case(
                "retry",
                f"latest_request_{latest}",
                _decl(limit=3, attempts=3),
                [*retry_prefix, *terminal_events, _reserve("R2", ["T0"], purpose=continuation)],
            )
        )
    c.extend(_materialization_specs())
    c.extend(_map_specs())
    c.extend(_binding_correction_specs())
    oversize = next(case for case in c if case["case_id"] == "binding/oversize_retrieved_known_usage")
    optional_decl = dict(_object(oversize["declaration"]))
    optional_decl["binding_requirements"] = {"D0": "optional"}
    c.append(
        _case(
            "binding",
            "optional_oversize_partial",
            optional_decl,
            [_object(event) for event in _array(oversize["events"])],
        )
    )
    c.extend(_latest_selection_specs())
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
        "historical_base_case_count": 156,
        "historical_base_corpus_sha256": BASE_CORPUS_SHA256,
        "predecessor_case_count": PREDECESSOR_CASE_COUNT,
        "predecessor_corpus_sha256": PREDECESSOR_CORPUS_SHA256,
        "accepted_predecessor_case_count": ACCEPTED_PREDECESSOR_CASE_COUNT,
        "accepted_predecessor_corpus_sha256": ACCEPTED_PREDECESSOR_CORPUS_SHA256,
        "accepted_predecessor_prefix_sha256": hashlib.sha256(
            canonical_bytes(cases[:ACCEPTED_PREDECESSOR_CASE_COUNT])
        ).hexdigest(),
        "corrected_predecessor_prefix_sha256": hashlib.sha256(
            canonical_bytes(cases[:PREDECESSOR_CASE_COUNT])
        ).hexdigest(),
        "contract_sha256": CONTRACT_SHA256,
        "materialization_addendum_sha256": MATERIALIZATION_ADDENDUM_SHA256,
        "materialized_version_contract_sha256": MATERIALIZED_VERSION_CONTRACT_SHA256,
        "optional_omission_addendum_sha256": OPTIONAL_OMISSION_ADDENDUM_SHA256,
        "map_execution_addendum_sha256": MAP_EXECUTION_ADDENDUM_SHA256,
        "binding_success_addendum_sha256": BINDING_SUCCESS_ADDENDUM_SHA256,
        "request_boundary_addendum_sha256": REQUEST_BOUNDARY_ADDENDUM_SHA256,
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
