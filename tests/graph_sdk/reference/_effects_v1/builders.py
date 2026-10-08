# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent effects_v1 reference: builders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from tests.graph_sdk.reference._effects_v1.model import (
    FAILURE_CLASSES,
    POLICY_CONDITIONS,
    Json,
    Object,
    _array,
    _object,
)
from tests.graph_sdk.reference._effects_v1.request_facts import (
    _embedded_settlement,
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
