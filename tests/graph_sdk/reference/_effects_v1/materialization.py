# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent effects_v1 reference: materialization."""

from __future__ import annotations

from typing import cast

from tests.graph_sdk.reference._effects_v1.model import (
    Json,
    Object,
    _array,
    _object,
    _strings,
)
from tests.graph_sdk.reference._effects_v1.request_facts import (
    _reject,
)


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
