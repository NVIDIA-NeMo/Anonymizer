# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent effects_v1 reference: evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import cast

from tests.graph_sdk.reference._effects_v1.admission import (
    admit,
    recheck_capabilities,
)
from tests.graph_sdk.reference._effects_v1.builders import (
    _binding_decl,
    _source_result,
)
from tests.graph_sdk.reference._effects_v1.map import (
    _admit_map,
    _empty_map_state,
    _reduce_map,
)
from tests.graph_sdk.reference._effects_v1.materialization import (
    _materialization_declaration,
)
from tests.graph_sdk.reference._effects_v1.model import (
    Json,
    Object,
    _array,
    _object,
)
from tests.graph_sdk.reference._effects_v1.reducer import (
    reduce_trace,
)
from tests.graph_sdk.reference._effects_v1.request_facts import (
    _reject,
)


def _materialization_preflight(declaration: Object, events: Sequence[Object]) -> Object:
    specs = [_object(raw) for raw in _array(declaration["materializations"])]
    completed_binding = declaration.get("preflight_uses_completed_binding") is True
    # Corrected latest cases preserve the completed P6 receipt. Older frozen
    # families retain their independent bounded preflight projection.
    binding_declaration = (
        deepcopy(declaration)
        if completed_binding
        else _binding_decl(
            {cast(str, spec["association"]): cast(str, spec["source"]) for spec in specs},
            max_items=max(cast(int, spec["max_items"]) for spec in specs),
            max_bytes=max(cast(int, spec["max_bytes"]) for spec in specs),
        )
    )
    if completed_binding:
        for key in (
            "materializations",
            "materialization_limits",
            "operation_occurrences",
            "preflight_uses_completed_binding",
            "root_artifacts",
            "root_input_types",
        ):
            binding_declaration.pop(key, None)
        binding_declaration["binding_cleanup_projection"] = True
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
        if completed_binding and event["kind"] in {"operation_start", "operation_blocked", "root_artifact"}:
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
    binding = reduce_trace(
        binding_declaration,
        binding_events if completed_binding else [*binding_events, {"kind": "binding_finish"}],
    )
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
