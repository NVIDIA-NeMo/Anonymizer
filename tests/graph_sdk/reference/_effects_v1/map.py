# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent effects_v1 reference: map."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from tests.graph_sdk.reference._effects_v1.materialization import (
    _natural,
)
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
