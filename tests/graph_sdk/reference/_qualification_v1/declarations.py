# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent qualification_v1 reference: declarations."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import cast

from tests.graph_sdk.reference._qualification_v1.model import (
    LIMIT_KEYS,
    TERMINAL_CATEGORIES,
    Obj,
    arr,
    limits,
    obj,
    reject,
)


def declaration(
    *,
    targets: Sequence[str] = ("A",),
    purpose: str = "protection",
    deps: Sequence[tuple[str, str]] = (),
    atomic: Sequence[Sequence[str]] | None = None,
    limit_changes: Mapping[str, int] | None = None,
) -> Obj:
    groups = [list(group) for group in (atomic or ())]
    grouped = {target for group in groups for target in group}
    groups.extend([target] for target in targets if target not in grouped)
    return {
        "atomic": groups,
        "binding_inputs": [],
        "dependencies": [{"dependent": b, "prerequisite": a} for a, b in deps],
        "execution": {"factory": "EXEC:I0", "graph": "G0", "invocation": "I0", "plan": "P0"},
        "limits": limits(**dict(limit_changes or {})),
        "initial_collections": [],
        "map_inputs": [
            {
                "expander": "EXP",
                "item_input": "item",
                "membership_port": "members",
                "outcome": "ok",
            }
        ],
        "node_kinds": {
            "EXP": "container",
            "MEM": "operation",
            "N": "operation",
            "N0": "operation",
            "N1": "operation",
        },
        "output_dependencies": [
            {
                "identity_input": "subject",
                "inputs": ["subject", "context"],
                "node": "N",
                "outcome": "ok",
                "port": "result",
            },
            {
                "identity_input": None,
                "inputs": ["context"],
                "node": "N",
                "outcome": "ok",
                "port": "evidence",
            },
            {"identity_input": None, "inputs": ["context"], "node": "EXP", "outcome": "ok", "port": "members"},
        ],
        "productions": [
            {
                "absence_queries": ["Q0"],
                "configuration": "c0",
                "consumed_ports": ["context"],
                "coverage": ["K0", "K1"],
                "evidence_port": "evidence",
                "findings": ["satisfied", "unsatisfied", "unknown"],
                "meaning": "privacy",
                "node": "N",
                "outcome": "ok",
                "promise": "P",
                "read_state": ["read"],
                "subject_port": "subject",
                "subject_source": "candidate",
            }
        ],
        "purpose": purpose,
        "root_outputs": [
            {"input_port": None, "port": "result", "source_node": "N", "source_port": "result", "target": t}
            for t in targets
        ],
        "subgraphs": [],
        "requirements": [
            {
                "consumed_ports": ["context"],
                "coverage": ["K0"],
                "meaning": "privacy",
                "outcome": "ok",
                "subject_port": "subject",
                "target": t,
            }
            for t in targets
        ],
        "targets": list(targets),
    }


def set_output_dependency(
    d: Obj, node: str, port_name: str, inputs: Sequence[str], *, identity_input: str | None = None
) -> None:
    dependencies = arr(d["output_dependencies"])
    dependencies[:] = [
        raw
        for raw in dependencies
        if not (obj(raw).get("node") == node and obj(raw).get("outcome") == "ok" and obj(raw).get("port") == port_name)
    ]
    dependencies.append(
        {"identity_input": identity_input, "inputs": list(inputs), "node": node, "outcome": "ok", "port": port_name}
    )


def set_production_consumed(d: Obj, ports: Sequence[str], *, promise: str = "P") -> None:
    production = next(p for p in map(obj, arr(d["productions"])) if p["promise"] == promise)
    production["consumed_ports"] = list(ports)


def admit(d: Obj) -> Obj:
    required = {
        "atomic",
        "binding_inputs",
        "dependencies",
        "execution",
        "initial_collections",
        "limits",
        "map_inputs",
        "node_kinds",
        "output_dependencies",
        "productions",
        "purpose",
        "root_outputs",
        "subgraphs",
        "requirements",
        "targets",
    }
    optional = {"keyed_joins", "map_item_requirements", "map_routes"}
    if not required.issubset(d) or set(d) - required - optional:
        return reject("invalid_value")
    if (
        not all(
            isinstance(d[k], list)
            for k in (
                "atomic",
                "binding_inputs",
                "dependencies",
                "initial_collections",
                "map_inputs",
                "productions",
                "output_dependencies",
                "requirements",
                "root_outputs",
                "subgraphs",
                "targets",
            )
        )
        or not isinstance(d["limits"], dict)
        or not isinstance(d["node_kinds"], dict)
        or ("keyed_joins" in d and not isinstance(d["keyed_joins"], list))
        or ("map_item_requirements" in d and not isinstance(d["map_item_requirements"], list))
        or ("map_routes" in d and not isinstance(d["map_routes"], list))
    ):
        return reject("invalid_type")
    if not obj(d["node_kinds"]) or any(
        not isinstance(node, str) or kind not in ("operation", "container")
        for node, kind in obj(d["node_kinds"]).items()
    ):
        return reject("invalid_value")
    lim = obj(d["limits"])
    if set(lim) != LIMIT_KEYS or any(type(v) is not int for v in lim.values()):
        return reject("invalid_type")
    if any(cast(int, value) < 0 for value in lim.values()) or cast(int, lim["max_fixed_point_steps"]) == 0:
        return reject("invalid_value")
    targets = [cast(str, x) for x in arr(d["targets"])]
    if len(targets) != len(set(targets)):
        return reject("duplicate")
    if cast(int, lim["max_fixed_point_steps"]) < len(targets) or len(arr(d["productions"])) > cast(
        int, lim["max_productions"]
    ):
        return reject("limit_exceeded")
    ps = [obj(x) for x in arr(d["productions"])]
    pfields = {
        "absence_queries",
        "configuration",
        "consumed_ports",
        "coverage",
        "evidence_port",
        "findings",
        "meaning",
        "node",
        "outcome",
        "promise",
        "read_state",
        "subject_port",
        "subject_source",
    }
    if any(set(p) != pfields for p in ps):
        return reject("invalid_value")
    if any(p["subject_source"] not in ("candidate", "input") for p in ps):
        return reject("invalid_value")
    if any(obj(d["node_kinds"]).get(cast(str, p["node"])) != "operation" for p in ps):
        return reject("contradictory")
    ids = [(p["node"], p["outcome"], p["promise"]) for p in ps]
    if len(ids) != len(set(ids)):
        return reject("duplicate")
    if any(not arr(p["findings"]) or len(arr(p["coverage"])) > cast(int, lim["max_coverage_atoms"]) for p in ps):
        return reject("limit_exceeded")
    requirement_fields = {"consumed_ports", "coverage", "meaning", "outcome", "subject_port", "target"}
    for r in map(obj, arr(d["requirements"])):
        if set(r) != requirement_fields:
            return reject("invalid_value")
        matches = [p for p in ps if p["meaning"] == r.get("meaning") and p["outcome"] == r.get("outcome")]
        if r.get("target") not in targets:
            return reject("foreign_owner")
        if not matches:
            return reject(
                "protection_ineligible"
                if any(production.get("outcome") == r.get("outcome") for production in ps)
                else "missing"
            )
        matching_nodes = {(cast(str, production["node"]), cast(str, production["outcome"])) for production in matches}
        declared_inputs = {
            cast(str, port)
            for dependency in map(obj, arr(d["output_dependencies"]))
            if (cast(str, dependency.get("node")), cast(str, dependency.get("outcome"))) in matching_nodes
            for port in arr(dependency["inputs"])
        }
        declared_ports = declared_inputs | {
            cast(str, dependency["port"])
            for dependency in map(obj, arr(d["output_dependencies"]))
            if (cast(str, dependency.get("node")), cast(str, dependency.get("outcome"))) in matching_nodes
        }
        if r.get("subject_port") not in declared_ports or not set(
            cast(str, port) for port in arr(r["consumed_ports"])
        ).issubset(declared_inputs):
            return reject("missing")
        eligible = [
            p
            for p in matches
            if r.get("subject_port") == p["subject_port"]
            and set(cast(str, x) for x in arr(r["consumed_ports"])).issubset(
                set(cast(str, x) for x in arr(p["consumed_ports"]))
            )
            and set(cast(str, x) for x in arr(r["coverage"])).issubset(set(cast(str, x) for x in arr(p["coverage"])))
        ]
        if not eligible:
            return reject("contradictory")
    if any(
        obj(x).get("prerequisite") not in targets or obj(x).get("dependent") not in targets
        for x in arr(d["dependencies"])
    ):
        return reject("foreign_owner")
    if any(not arr(group) for group in arr(d["atomic"])):
        return reject("invalid_value")
    groups = [[cast(str, x) for x in arr(group)] for group in arr(d["atomic"])]
    flat = [x for group in groups for x in group]
    if any(x not in targets for x in flat):
        return reject("foreign_owner")
    if any(len(group) != len(set(group)) for group in groups):
        return reject("duplicate")
    unique_groups = {frozenset(group) for group in groups}
    if any(left & right for left in unique_groups for right in unique_groups if left != right):
        return reject("overlap")
    mentioned = set(flat)
    normalized_atomic = sorted(
        [sorted(group) for group in unique_groups] + [[target] for target in targets if target not in mentioned],
        key=lambda group: tuple(group),
    )
    binding_fields = {
        "declaration",
        "materialization",
        "node",
        "port",
        "source",
        "target",
        "version_selection",
    }
    bindings = [obj(x) for x in arr(d["binding_inputs"])]
    if any(set(binding) != binding_fields for binding in bindings):
        return reject("invalid_value")
    binding_ids = [binding["declaration"] for binding in bindings]
    if len(binding_ids) != len(set(binding_ids)):
        return reject("duplicate")
    if any(
        binding["target"] not in targets
        or obj(d["node_kinds"]).get(cast(str, binding["node"])) != "operation"
        or binding["materialization"] not in ("single", "collection")
        or binding["version_selection"] not in ("exact_one", "latest")
        or (binding["version_selection"] == "latest" and binding["materialization"] != "single")
        or (binding["materialization"] == "collection" and binding["version_selection"] != "exact_one")
        or not all(isinstance(binding[field], str) and binding[field] for field in binding_fields)
        for binding in bindings
    ):
        return reject("foreign_owner")
    collection_fields = {"declaration", "node", "port", "target"}
    collections = [obj(x) for x in arr(d["initial_collections"])]
    if any(set(collection) != collection_fields for collection in collections):
        return reject("invalid_value")
    binding_destinations = [{field: binding[field] for field in collection_fields} for binding in bindings]
    if any(collection not in binding_destinations for collection in collections):
        return reject("missing")
    expected_collections = [
        {field: binding[field] for field in collection_fields}
        for binding in bindings
        if binding["materialization"] == "collection"
    ]
    if collections != expected_collections:
        return reject("contradictory")
    map_fields = {"expander", "item_input", "membership_port", "outcome"}
    maps = [obj(x) for x in arr(d["map_inputs"])]
    if any(set(map_input) != map_fields for map_input in maps):
        return reject("invalid_value")
    map_ids = [(x["expander"], x["outcome"]) for x in maps]
    if len(map_ids) != len(set(map_ids)):
        return reject("duplicate")
    if any(
        obj(d["node_kinds"]).get(cast(str, map_input["expander"])) not in ("operation", "container")
        or not all(isinstance(map_input[field], str) and map_input[field] for field in map_fields)
        for map_input in maps
    ):
        return reject("foreign_owner")
    route_fields = {
        "expander",
        "item_input",
        "member",
        "membership_port",
        "outcome",
        "path",
    }
    routes = [obj(x) for x in arr(d.get("map_routes"))]
    if any(set(route) != route_fields for route in routes):
        return reject("invalid_value")
    if any(
        not isinstance(route["path"], list)
        or any(
            not isinstance(node, str) or obj(d["node_kinds"]).get(node) != "container" for node in arr(route["path"])
        )
        or len(arr(route["path"])) != len(set(cast(str, node) for node in arr(route["path"])))
        or obj(d["node_kinds"]).get(cast(str, route["expander"])) not in ("operation", "container")
        or obj(d["node_kinds"]).get(cast(str, route["member"])) != "operation"
        or route["expander"] in arr(route["path"])
        or route["member"] in arr(route["path"])
        or not all(isinstance(route[field], str) and route[field] for field in route_fields - {"path"})
        for route in routes
    ):
        return reject("foreign_owner")
    route_ids = [json.dumps(route, sort_keys=True) for route in routes]
    if len(route_ids) != len(set(route_ids)):
        return reject("duplicate")
    if any(
        not any(
            map_input["expander"] == route["expander"]
            and map_input["item_input"] == route["item_input"]
            and map_input["membership_port"] == route["membership_port"]
            and map_input["outcome"] == route["outcome"]
            for map_input in maps
        )
        for route in routes
    ):
        return reject("missing")
    join_fields = {"accepted_categories", "join", "reduction", "source"}
    joins = [obj(x) for x in arr(d.get("keyed_joins"))]
    if any(set(join) != join_fields for join in joins):
        return reject("invalid_value")
    if any(
        not isinstance(join["accepted_categories"], list)
        or not arr(join["accepted_categories"])
        or any(category not in TERMINAL_CATEGORIES for category in arr(join["accepted_categories"]))
        or len(arr(join["accepted_categories"]))
        != len(set(cast(str, category) for category in arr(join["accepted_categories"])))
        or join["reduction"] != "all_by_key"
        or obj(d["node_kinds"]).get(cast(str, join["source"])) != "operation"
        or obj(d["node_kinds"]).get(cast(str, join["join"])) != "operation"
        for join in joins
    ):
        return reject("contradictory")
    if len({cast(str, join["source"]) for join in joins}) != len(joins) or len(
        {cast(str, join["join"]) for join in joins}
    ) != len(joins):
        return reject("duplicate")
    if any(len([route for route in routes if route["expander"] == join["source"]]) != 1 for join in joins):
        return reject("missing")
    if any(len([join for join in joins if join["source"] == route["expander"]]) != 1 for route in routes):
        return reject("missing")
    dependency_fields = {"identity_input", "inputs", "node", "outcome", "port"}
    dependencies = [obj(x) for x in arr(d["output_dependencies"])]
    if any(
        set(x) != dependency_fields
        or obj(d["node_kinds"]).get(cast(str, x["node"])) is None
        or not isinstance(x["port"], str)
        or not isinstance(x["outcome"], str)
        or not isinstance(x["inputs"], list)
        or any(not isinstance(value, str) or not value for value in arr(x["inputs"]))
        or len(arr(x["inputs"])) != len(set(cast(str, value) for value in arr(x["inputs"])))
        or (x["identity_input"] is not None and x["identity_input"] not in arr(x["inputs"]))
        for x in dependencies
    ):
        return reject("invalid_value")
    dependency_ids = [(x["node"], x["outcome"], x["port"]) for x in dependencies]
    if len(dependency_ids) != len(set(dependency_ids)):
        return reject("duplicate")
    dependency_by_output = {(x["node"], x["outcome"], x["port"]): x for x in dependencies}
    for production in ps:
        dependency = dependency_by_output.get((production["node"], production["outcome"], production["evidence_port"]))
        if dependency is None:
            return reject("missing")
        if set(arr(dependency["inputs"])) != set(arr(production["consumed_ports"])):
            return reject("contradictory")
    root_fields = {"input_port", "port", "source_node", "source_port", "target"}
    roots = [obj(x) for x in arr(d["root_outputs"])]
    if any(set(x) != root_fields or x["target"] not in targets for x in roots):
        return reject("invalid_value")
    if {x["target"] for x in roots} != set(targets):
        return reject("missing")
    if len(roots) != len(targets):
        return reject("duplicate")
    endpoint_fields = {
        "expander",
        "expansion_outcome",
        "item_input",
        "member",
        "membership_port",
        "path",
    }
    typed_requirement_fields = {
        "candidate_port",
        "consumed_endpoints",
        "coverage",
        "meaning",
        "promise",
        "subject_endpoint",
        "target",
    }
    typed_requirement_ids: list[str] = []
    if bool(arr(d.get("map_item_requirements"))) != bool(routes) or bool(routes) != bool(joins):
        return reject("missing")
    for requirement in map(obj, arr(d.get("map_item_requirements"))):
        if set(requirement) != typed_requirement_fields:
            return reject("invalid_value")
        endpoints = [
            obj(endpoint)
            for endpoint in [requirement.get("subject_endpoint"), *arr(requirement.get("consumed_endpoints"))]
            if endpoint is not None
        ]
        if not endpoints or any(set(endpoint) != endpoint_fields for endpoint in endpoints):
            return reject("invalid_value")
        if any(
            not isinstance(endpoint["path"], list)
            or any(not isinstance(node, str) or not node for node in arr(endpoint["path"]))
            or any(not isinstance(endpoint[field], str) or not endpoint[field] for field in endpoint_fields - {"path"})
            for endpoint in endpoints
        ):
            return reject("invalid_value")
        domains = {
            (
                tuple(arr(endpoint["path"])),
                endpoint["expander"],
                endpoint["member"],
                endpoint["item_input"],
                endpoint["expansion_outcome"],
                endpoint["membership_port"],
            )
            for endpoint in endpoints
        }
        if len(domains) != 1:
            return reject("contradictory")
        domain = next(iter(domains))
        if any(obj(d["node_kinds"]).get(cast(str, node)) != "container" for node in domain[0]):
            return reject("invalid_value")
        route_matches = [
            route
            for route in routes
            if tuple(arr(route["path"])) == domain[0]
            and route["expander"] == domain[1]
            and route["member"] == domain[2]
            and route["item_input"] == domain[3]
            and route["outcome"] == domain[4]
            and route["membership_port"] == domain[5]
        ]
        if len(route_matches) != 1:
            same_map_routes = [
                route
                for route in routes
                if route["expander"] == domain[1]
                and route["member"] == domain[2]
                and route["item_input"] == domain[3]
                and route["outcome"] == domain[4]
                and route["membership_port"] == domain[5]
            ]
            return reject("duplicate" if route_matches else "foreign_owner" if same_map_routes else "missing")
        if not any(
            map_input["expander"] == domain[1]
            and map_input["item_input"] == domain[3]
            and map_input["outcome"] == domain[4]
            and map_input["membership_port"] == domain[5]
            for map_input in maps
        ):
            return reject("missing")
        if obj(d["node_kinds"]).get(cast(str, domain[2])) != "operation":
            return reject("foreign_owner")
        productions = [
            production
            for production in ps
            if production["node"] == domain[2]
            and production["promise"] == requirement["promise"]
            and production["meaning"] == requirement["meaning"]
        ]
        if len(productions) != 1:
            return reject("missing" if not productions else "duplicate")
        production = productions[0]
        subject_endpoint = obj(requirement.get("subject_endpoint"))
        if subject_endpoint and (production["subject_source"] != "input" or production["subject_port"] != domain[3]):
            return reject("contradictory")
        if arr(requirement["consumed_endpoints"]) and domain[3] not in arr(production["consumed_ports"]):
            return reject("contradictory")
        if requirement["target"] not in targets:
            return reject("foreign_owner")
        if not any(
            root["target"] == requirement["target"] and root["port"] == requirement["candidate_port"] for root in roots
        ):
            return reject("missing")
        if not set(cast(str, value) for value in arr(requirement["coverage"])).issubset(
            set(cast(str, value) for value in arr(production["coverage"]))
        ):
            return reject("contradictory")
        typed_requirement_ids.append(json.dumps([requirement["target"], requirement["promise"], *domain]))
    if len(typed_requirement_ids) != len(set(typed_requirement_ids)):
        return reject("duplicate")
    subgraph_fields = {
        "body_input",
        "body_node",
        "body_outcome",
        "body_port",
        "body_source",
        "input_port",
        "node",
        "outcome",
        "port",
    }
    if any(set(obj(x)) != subgraph_fields for x in arr(d["subgraphs"])):
        return reject("invalid_value")
    if any(
        obj(d["node_kinds"]).get(cast(str, obj(x)["node"])) != "container"
        or obj(x).get("body_source") not in ("node_output", "workflow_input")
        or (
            obj(x).get("body_source") == "node_output"
            and obj(d["node_kinds"]).get(cast(str, obj(x)["body_node"])) != "operation"
        )
        or (
            obj(x).get("body_source") == "workflow_input"
            and (obj(x).get("body_node") is not None or obj(x).get("body_port") is not None)
        )
        for x in arr(d["subgraphs"])
    ):
        return reject("foreign_owner")
    accepted: Obj = {"status": "accepted"}
    if mentioned != set(targets):
        accepted["atomic"] = normalized_atomic
    return accepted
