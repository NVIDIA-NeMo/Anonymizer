# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent fact-driven D08 qualification reference."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import TypeAlias, cast

Json: TypeAlias = str | int | bool | None | list["Json"] | dict[str, "Json"]
Obj: TypeAlias = dict[str, Json]
CONTRACT_SHA256 = "239bdaf97eda6b90caeb13d29826abead08e2beff6297460c26409b3e1f5d87c"
STRUCTURAL_CONTRACT_SHA256 = "88c0ef075b225847f1b2668d2a749307c220eceb50215718db722119d828dc6b"
MATERIALIZED_VERSION_CONTRACT_SHA256 = "165c7c95bce31a7c5808860f28d012bbe1986bf0712cebdc86fb08ad07afcb21"
MAP_ITEM_EVIDENCE_CONTRACT_SHA256 = "d5e270fe413f4f5632b522e3ce57e0c913a143060997d63b7673268ae9acfbd8"
GENERATOR_VERSION = "qualification-v1-generator-25-admission-precedence-v6"
SELF_TEST_VERSION = "qualification-v1-self-test-25-admission-precedence-v6"
CORPUS_PATH = "future-contracts/r3-map-item-v6/qualification_v1_cases.json"
V10_IDS_SHA256 = "043c433056b1ecb21d17ef48efa8bcca6a678fd6b722e7a91ea2e1b3a05c9544"

TERMINAL_CATEGORIES = {"blocked", "cancelled", "failure", "inconsistent", "lost", "success"}
REASON_CODES = {
    "cancel_requested",
    "contradictory",
    "duplicate",
    "execution_failed",
    "foreign",
    "missing",
    "prerequisite",
    "stale",
    "transport_lost",
}


def arr(x: Json) -> list[Json]:
    return cast(list[Json], x) if isinstance(x, list) else []


def obj(x: Json) -> Obj:
    return cast(Obj, x) if isinstance(x, dict) else {}


def reject(code: str) -> Obj:
    return {"code": code, "status": "rejected"}


def canonical_bytes(xs: Iterable[Obj]) -> bytes:
    return (json.dumps(tuple(xs), indent=2, sort_keys=True) + "\n").encode()


LIMIT_KEYS = {
    "max_absence_revisions",
    "max_consumed_per_assessment",
    "max_coverage_atoms",
    "max_fixed_point_steps",
    "max_port_facts",
    "max_productions",
    "max_provenance_edges",
    "max_required_decisions",
    "max_revision_entries",
    "max_submissions",
    "max_verified_evidence",
    "max_artifacts",
    "max_artifact_bytes",
}


def limits(**kw: int) -> Obj:
    x: Obj = {
        "max_artifact_bytes": 64,
        "max_artifacts": 32,
        "max_absence_revisions": 2,
        "max_consumed_per_assessment": 3,
        "max_coverage_atoms": 2,
        "max_fixed_point_steps": 3,
        "max_port_facts": 16,
        "max_productions": 2,
        "max_provenance_edges": 16,
        "max_required_decisions": 2,
        "max_revision_entries": 16,
        "max_submissions": 3,
        "max_verified_evidence": 3,
    }
    x.update(kw)
    return x


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
    if set(lim) != LIMIT_KEYS or any(type(v) is not int or v < 0 for v in lim.values()):
        return reject("invalid_type")
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
        if r.get("target") not in targets or not matches:
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
    flat = [cast(str, x) for g in arr(d["atomic"]) for x in arr(g)]
    if any(x not in targets for x in flat):
        return reject("foreign_owner")
    if len(flat) != len(set(flat)):
        return reject("contradictory")
    if set(flat) != set(targets):
        return reject("contradictory")
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
    return {"status": "accepted"}


def initial(d: Obj) -> Obj:
    e = obj(d["execution"])
    return {
        "artifacts": {},
        "assessment_facts": {},
        "assessment_submissions": [],
        "binding_receipt": None,
        "binding_requests": {},
        "binding_cleanup_associations": {},
        "binding_cleanups": {},
        "cleanup_associations": {},
        "cleanups": {},
        "collection_values": {},
        "entries": {},
        "finals": {},
        "memberships": {},
        "input_producers": {},
        "ports": {},
        "provenance": {},
        "requests": {},
        "reservations": {},
        "revision": {"absences": {}, "artifacts": {}, "configurations": {}, "state": {}},
        "revision_sealed": False,
        "terminals": {},
        "execution": {"factory": e["factory"], "graph": e["graph"], "invocation": e["invocation"], "plan": e["plan"]},
    }


def add_unique(store: Obj, key: str, value: Obj) -> Obj | None:
    if key in store:
        return reject("duplicate")
    store[key] = value
    return None


def advance(s: Obj, e: Obj) -> Obj | None:
    k = e.get("kind")
    if k == "binding_receipt":
        if set(e) != {"artifacts", "binding", "kind", "sources", "terminal"}:
            return reject("invalid_value")
        if s["binding_receipt"] is not None:
            return reject("duplicate")
        if e.get("terminal") not in ("success", "partial"):
            return reject("contradictory")
        sources = [obj(source) for source in arr(e.get("sources"))]
        artifacts = [obj(artifact_value) for artifact_value in arr(e.get("artifacts"))]
        source_fields = {"declaration", "node", "port", "source", "target", "terminal"}
        artifact_fields = {"declaration", "key", "node", "port", "source", "target", "version"}
        if any(set(source) != source_fields or source.get("terminal") != "bound" for source in sources):
            return reject("invalid_value")
        if any(set(artifact_value) != artifact_fields for artifact_value in artifacts):
            return reject("invalid_value")
        references = [
            (artifact_value.get("declaration"), artifact_value.get("key"), artifact_value.get("version"))
            for artifact_value in artifacts
        ]
        if len(references) != len(set(references)):
            return reject("duplicate")
        if any(
            type(artifact_value.get("key")) is not int
            or cast(int, artifact_value["key"]) < 0
            or type(artifact_value.get("version")) is not int
            or cast(int, artifact_value["version"]) <= 0
            for artifact_value in artifacts
        ):
            return reject("invalid_value")
        s["binding_receipt"] = {
            "artifacts": artifacts,
            "binding": e["binding"],
            "sources": sources,
            "terminal": e["terminal"],
        }
        return None
    if k == "artifact":
        return add_unique(
            obj(s["artifacts"]),
            cast(str, e["ref"]),
            {x: e[x] for x in ("bytes", "invocation", "key", "target", "version")},
        )
    if k == "binding_reserve":
        if set(e) != {"association", "kind", "policy", "purpose", "request", "resource"}:
            return reject("invalid_value")
        if e.get("purpose") != "initial_binding" or e.get("policy") != "P0":
            return reject("contradictory")
        return add_unique(
            obj(s["binding_requests"]),
            cast(str, e["request"]),
            {
                "association": e["association"],
                "phase": "reserved",
                "policy": e["policy"],
                "purpose": e["purpose"],
                "resource": e["resource"],
                "settlement": None,
                "terminal": None,
            },
        )
    if k == "binding_dispatch":
        request = obj(obj(s["binding_requests"]).get(cast(str, e.get("request")), {}))
        if set(e) != {"kind", "request"} or request.get("phase") != "reserved":
            return reject("missing")
        request["phase"] = "dispatched"
        return None
    if k == "binding_result":
        request = obj(obj(s["binding_requests"]).get(cast(str, e.get("request")), {}))
        if set(e) != {"kind", "outcome", "request"} or request.get("phase") != "dispatched":
            return reject("missing")
        if e.get("outcome") != "retrieved" or request.get("terminal") is not None:
            return reject("contradictory")
        request["terminal"] = {"condition": "result", "outcome": e["outcome"]}
        request["phase"] = "terminal"
        return None
    if k == "binding_settlement":
        request = obj(obj(s["binding_requests"]).get(cast(str, e.get("request")), {}))
        if set(e) != {"kind", "remote_stopped", "request", "usage"} or request.get("phase") != "terminal":
            return reject("missing")
        if e.get("remote_stopped") is not True or e.get("usage") != "known":
            return reject("contradictory")
        request["settlement"] = {"remote_stopped": True, "usage": "known"}
        request["phase"] = "settled"
        return None
    if k == "port":
        return add_unique(
            obj(s["ports"]),
            f"{e['activation']}|{e['port']}",
            {x: e[x] for x in ("activation", "artifact", "node", "port", "role", "target")},
        )
    if k == "input_producer":
        if set(e) != {"activation", "kind", "node", "port", "producer", "target"}:
            return reject("invalid_value")
        return add_unique(
            obj(s["input_producers"]),
            f"{e['activation']}|{e['port']}",
            {x: e[x] for x in ("activation", "node", "port", "producer", "target")},
        )
    if k == "entry":
        if type(e.get("occurrence")) is not int or cast(int, e["occurrence"]) < 0:
            return reject("invalid_value")
        return add_unique(
            obj(s["entries"]),
            cast(str, e["activation"]),
            {
                x: e[x]
                for x in (
                    "closed_unstarted",
                    "node",
                    "node_kind",
                    "occurrence",
                    "parent",
                    "state_category",
                    "state_outcome",
                    "target",
                )
            },
        )
    if k == "terminal":
        if type(e.get("structural")) is not bool:
            return reject("invalid_type")
        category = e.get("category")
        attempt = e.get("attempt")
        reasons = arr(e.get("reasons"))
        if category not in TERMINAL_CATEGORIES or any(reason not in REASON_CODES for reason in reasons):
            return reject("invalid_value")
        if e["structural"] is True and attempt is not None:
            return reject("contradictory")
        if e["structural"] is False and category in ("success", "failure", "cancelled", "lost") and attempt is None:
            return reject("contradictory")
        if (category == "success" and reasons) or (category != "success" and not reasons):
            return reject("contradictory")
        return add_unique(
            obj(s["terminals"]),
            cast(str, e["activation"]),
            {x: e[x] for x in ("attempt", "category", "outcome", "reasons", "structural", "target")},
        )
    if k == "reservation":
        return add_unique(
            obj(s["reservations"]), cast(str, e["activation"]), {x: e[x] for x in ("parent", "selected", "target")}
        )
    if k == "membership":
        key = "__ROOT__" if e["parent"] is None else cast(str, e["parent"])
        return add_unique(
            obj(s["memberships"]),
            key,
            {x: e[x] for x in ("closed", "expansion_outcome", "members", "parent", "status", "target")},
        )
    if k == "result_owner":
        s["execution"] = {x: e[x] for x in ("factory", "graph", "invocation", "plan")}
        return None
    if k == "assessment":
        fields = (
            "activation",
            "authenticated_factory",
            "consumed",
            "coverage",
            "environment",
            "evidence_artifact",
            "evidence_port",
            "finding",
            "node",
            "outcome",
            "promise",
            "subject_artifact",
            "subject_port",
            "target",
        )
        return add_unique(obj(s["assessment_facts"]), cast(str, e["fact"]), {x: e[x] for x in fields})
    if k == "assessment_submission":
        if set(e) != {"fact", "kind"} or not isinstance(e.get("fact"), str):
            return reject("invalid_type")
        arr(s["assessment_submissions"]).append(cast(str, e["fact"]))
        return None
    if k == "final":
        return add_unique(
            obj(s["finals"]),
            cast(str, e["target"]),
            {x: e[x] for x in ("activation", "candidate", "node", "outcome", "port", "producer")},
        )
    if k == "provenance":
        provenance_fields = {
            "activation",
            "artifact",
            "binding_artifact",
            "declaration",
            "decision",
            "expander",
            "item_key",
            "item_version",
            "key",
            "kind",
            "member",
            "node",
            "parents",
            "port",
            "root_target",
            "source",
            "target",
        }
        if set(e) != provenance_fields:
            return reject("invalid_value")
        parent_keys = [cast(str, parent) for parent in arr(e["parents"])]
        if len(parent_keys) != len(set(parent_keys)):
            return reject("duplicate")
        prov = obj(s["provenance"])
        if any(parent not in prov for parent in arr(e["parents"])):
            return reject("missing")
        return add_unique(
            prov,
            cast(str, e["key"]),
            {
                x: e[x]
                for x in (
                    "activation",
                    "artifact",
                    "binding_artifact",
                    "declaration",
                    "decision",
                    "expander",
                    "item_key",
                    "item_version",
                    "node",
                    "member",
                    "parents",
                    "port",
                    "root_target",
                    "source",
                    "target",
                )
            },
        )
    if k == "collection_value":
        if set(e) != {"artifact", "items", "kind", "producer", "target"}:
            return reject("invalid_value")
        items = arr(e.get("items"))
        if any(
            set(obj(item)) != {"key", "version"}
            or type(obj(item).get("key")) is not int
            or cast(int, obj(item)["key"]) < 0
            or type(obj(item).get("version")) is not int
            or cast(int, obj(item)["version"]) <= 0
            for item in items
        ):
            return reject("invalid_value")
        identities = [(obj(item)["key"], obj(item)["version"]) for item in items]
        if identities != sorted(identities) or len(identities) != len(set(identities)):
            return reject("contradictory")
        return add_unique(
            obj(s["collection_values"]),
            cast(str, e["producer"]),
            {"artifact": e["artifact"], "items": items, "target": e["target"]},
        )
    if k == "revision":
        if s["revision_sealed"] is True:
            return reject("contradictory")
        group = cast(str, e["collection"])
        store = obj(obj(s["revision"])[group])
        return add_unique(store, cast(str, e["key"]), {"value": e["value"]})
    if k == "seal_revision":
        s["revision_sealed"] = True
        return None
    if k == "request_attempt":
        associations = arr(e["associations"])
        if len(associations) != len(set(cast(str, item) for item in associations)):
            return reject("duplicate")
        return add_unique(
            obj(s["requests"]),
            cast(str, e["request"]),
            {
                "associations": sorted(cast(str, item) for item in associations),
                "policy": e["policy"],
                "predecessor": e["predecessor"],
                "purpose": e["purpose"],
                "settlement": None,
                "terminal": None,
            },
        )
    if k == "request_terminal":
        r = obj(obj(s["requests"]).get(cast(str, e["request"]), {}))
        if not r:
            return reject("missing")
        if r["terminal"] is not None:
            return reject("duplicate")
        r["terminal"] = {"condition": e["condition"], "failure": e["failure"]}
        return None
    if k == "settlement":
        r = obj(obj(s["requests"]).get(cast(str, e["request"]), {}))
        if not r:
            return reject("missing")
        if r["settlement"] is not None:
            return reject("duplicate")
        r["settlement"] = {"remote_stopped": e["remote_stopped"], "usage": e["usage"]}
        return None
    if k == "cleanup_association":
        return add_unique(
            obj(s["cleanup_associations"]), cast(str, e["resource"]), {x: e[x] for x in ("owner", "purpose", "targets")}
        )
    if k == "cleanup":
        return add_unique(obj(s["cleanups"]), cast(str, e["resource"]), {"disposition": e["disposition"]})
    if k == "binding_cleanup_association":
        return add_unique(
            obj(s["binding_cleanup_associations"]),
            cast(str, e["resource"]),
            {x: e[x] for x in ("owner", "purpose", "targets")},
        )
    if k == "binding_cleanup":
        return add_unique(obj(s["binding_cleanups"]), cast(str, e["resource"]), {"disposition": e["disposition"]})
    return reject("unknown_event")


def typed_endpoint_owner(d: Obj, s: Obj, activation: str, target: str, endpoint: Obj) -> tuple[str | None, str | None]:
    entries = obj(s["entries"])
    ports = obj(s["ports"])
    producers = obj(s["input_producers"])
    provenance_facts = obj(s["provenance"])
    memberships = obj(s["memberships"])
    member_entry = obj(entries.get(activation, {}))
    if not member_entry:
        return None, "missing"
    if member_entry.get("node") != endpoint.get("member") or member_entry.get("target") != target:
        return None, "foreign_owner"
    expander_activation = cast(str, member_entry.get("parent"))
    expander_entry = obj(entries.get(expander_activation, {}))
    if not expander_entry:
        return None, "missing"
    if (
        expander_entry.get("node") != endpoint.get("expander")
        or expander_entry.get("state_outcome") != endpoint.get("expansion_outcome")
        or expander_entry.get("target") != target
    ):
        return None, "foreign_owner"
    parent = expander_entry.get("parent")
    for expected_node in reversed([cast(str, value) for value in arr(endpoint.get("path"))]):
        wrapper = obj(entries.get(cast(str, parent), {}))
        if wrapper.get("node") != expected_node or wrapper.get("target") != target:
            return None, "foreign_owner"
        parent = wrapper.get("parent")
    if parent is not None:
        return None, "foreign_owner"
    item_port = obj(ports.get(f"{activation}|{endpoint['item_input']}", {}))
    input_fact = obj(producers.get(f"{activation}|{endpoint['item_input']}", {}))
    if not item_port or not input_fact:
        return None, "missing"
    producer_key = cast(str, input_fact.get("producer"))
    item_owner = obj(provenance_facts.get(producer_key, {}))
    membership_key = f"OP:{expander_activation}:{target}:{endpoint['membership_port']}"
    membership_owner = obj(provenance_facts.get(membership_key, {}))
    membership = obj(memberships.get(expander_activation, {}))
    artifact_value = obj(obj(s["artifacts"]).get(cast(str, item_port.get("artifact")), {}))
    if not item_owner or not membership_owner or not membership:
        return None, "missing"
    if artifact_value.get("target") != target or artifact_value.get("invocation") != obj(d["execution"])["invocation"]:
        return None, "foreign_owner"
    if (
        item_port.get("node") != endpoint.get("member")
        or item_port.get("role") != "artifact"
        or input_fact.get("node") != endpoint.get("member")
        or input_fact.get("target") != target
        or item_owner.get("source") != "map_item"
        or item_owner.get("artifact") != item_port.get("artifact")
        or item_owner.get("member") != activation
        or item_owner.get("expander") != expander_activation
        or item_owner.get("port") != endpoint.get("item_input")
        or item_owner.get("target") != target
        or arr(item_owner.get("parents")) != [membership_key]
        or membership_owner.get("activation") != expander_activation
        or membership_owner.get("node") != endpoint.get("expander")
        or membership_owner.get("port") != endpoint.get("membership_port")
        or membership_owner.get("target") != target
        or activation not in arr(membership.get("members"))
        or membership.get("target") != target
        or artifact_value.get("key") != cast(str, item_owner.get("artifact", "")).rsplit("v", 1)[0]
        or artifact_value.get("version") != item_owner.get("item_version")
    ):
        return None, "contradictory"
    return producer_key, None


def authenticate(d: Obj, s: Obj) -> tuple[list[Obj], Obj | None]:
    lim = obj(d["limits"])
    raw = [cast(str, value) for value in arr(s["assessment_submissions"])]
    if len(raw) > cast(int, lim["max_submissions"]):
        return [], reject("limit_exceeded")
    ps = {(p["node"], p["outcome"], p["promise"]): p for p in map(obj, arr(d["productions"]))}
    seen_references: set[str] = set()
    seen_identities: set[tuple[Json, Json]] = set()
    verified = []
    facts = obj(s["assessment_facts"])
    for reference in raw:
        if reference in seen_references:
            return [], reject("duplicate")
        seen_references.add(reference)
        a = obj(facts.get(reference, {}))
        if not a:
            return [], reject("foreign_owner")
        identity = (a["activation"], a["promise"])
        if identity in seen_identities:
            return [], reject("duplicate")
        seen_identities.add(identity)
        p = ps.get((a["node"], a["outcome"], a["promise"]))
        if p is None:
            return [], reject("unsupported")
        if a["authenticated_factory"] != obj(d["execution"])["factory"]:
            return [], reject("foreign_owner")
        if a["evidence_port"] != p["evidence_port"] or a["subject_port"] != p["subject_port"]:
            return [], reject("contradictory")
        assessment_coverage = [cast(str, x) for x in arr(a["coverage"])]
        promise_coverage = [cast(str, x) for x in arr(p["coverage"])]
        if len(assessment_coverage) != len(set(assessment_coverage)):
            return [], reject("duplicate")
        if a["finding"] not in arr(p["findings"]) or set(assessment_coverage) - set(promise_coverage):
            return [], reject("unsupported")
        if set(assessment_coverage) != set(promise_coverage):
            return [], reject("contradictory")
        consumed = obj(a["consumed"])
        if len(consumed) > cast(int, lim["max_consumed_per_assessment"]):
            return [], reject("limit_exceeded")
        declared_consumed = set(cast(str, x) for x in arr(p["consumed_ports"]))
        if set(consumed) - declared_consumed:
            return [], reject("unsupported")
        if set(consumed) != declared_consumed:
            return [], reject("contradictory")
        target = cast(str, a["target"])
        activation = cast(str, a["activation"])
        ports = obj(s["ports"])
        arts = obj(s["artifacts"])
        entry_value = obj(obj(s["entries"]).get(activation, {}))
        terminal_value = obj(obj(s["terminals"]).get(activation, {}))
        typed_requirements = [
            requirement
            for requirement in map(obj, arr(d.get("map_item_requirements")))
            if requirement.get("target") == target
            and requirement.get("promise") == a["promise"]
            and requirement.get("meaning") == p["meaning"]
            and any(
                obj(endpoint).get("member") == p["node"]
                for endpoint in [requirement.get("subject_endpoint"), *arr(requirement.get("consumed_endpoints"))]
                if endpoint is not None
            )
        ]
        if entry_value and entry_value.get("node") not in obj(d["node_kinds"]):
            return [], reject("foreign_owner")
        if (
            not entry_value
            or entry_value.get("closed_unstarted") is True
            or entry_value.get("node_kind") != "operation"
            or entry_value.get("node") != p["node"]
            or entry_value.get("target") != target
        ):
            return [], reject("contradictory")
        if not terminal_value:
            return [], reject("missing")
        if (
            terminal_value.get("target") != target
            or terminal_value.get("structural") is not False
            or (
                terminal_value.get("category") not in ("blocked", "inconsistent")
                and terminal_value.get("attempt") is None
            )
        ):
            return [], reject("contradictory")
        if entry_value.get("state_category") != "success" or terminal_value.get("category") != "success":
            continue
        if entry_value.get("state_outcome") != p["outcome"] or terminal_value.get("outcome") != p["outcome"]:
            return [], reject("contradictory")
        ep = obj(ports.get(f"{activation}|{a['evidence_port']}", {}))
        sp = obj(ports.get(f"{activation}|{a['subject_port']}", {}))
        if not ep or not sp:
            return [], reject("missing")
        root_bound = any(
            obj(root).get("target") == target
            and obj(root).get("source_node") == p["node"]
            and obj(root).get("source_port") == a["evidence_port"]
            for root in arr(d["root_outputs"])
        )
        evidence_role = "candidate" if root_bound or a["evidence_port"] == a["subject_port"] else "evidence"
        if ep.get("target") not in (None, target) or sp.get("target") not in (None, target):
            return [], reject("foreign_owner")
        typed_subject = next(
            (
                obj(requirement["subject_endpoint"])
                for requirement in typed_requirements
                if requirement.get("subject_endpoint") is not None
            ),
            {},
        )
        subject_producer: str | None = None
        if typed_subject:
            subject_producer, endpoint_error = typed_endpoint_owner(d, s, activation, target, typed_subject)
            if endpoint_error:
                return [], reject(endpoint_error)
        if (
            ep.get("artifact") != a["evidence_artifact"]
            or ep.get("role") != evidence_role
            or ep.get("node") != p["node"]
            or ep.get("target") != target
            or sp.get("artifact") != a["subject_artifact"]
            or sp.get("role") != ("artifact" if typed_subject else "candidate")
            or sp.get("node") != p["node"]
            or sp.get("target") != target
        ):
            return [], reject("contradictory")
        consumed_roles: Obj = {}
        for consumed_port, ref in consumed.items():
            fact = obj(ports.get(f"{activation}|{consumed_port}", {}))
            if not fact:
                return [], reject("missing")
            if fact.get("artifact") != ref or fact.get("node") != p["node"] or fact.get("target") != target:
                return [], reject("contradictory")
            if fact.get("role") not in ("artifact", "candidate", "decision", "evidence"):
                return [], reject("unsupported")
            typed_consumed = next(
                (
                    obj(endpoint)
                    for requirement in typed_requirements
                    for endpoint in arr(requirement.get("consumed_endpoints"))
                    if obj(endpoint).get("item_input") == consumed_port
                ),
                {},
            )
            if typed_consumed:
                consumed_producer, endpoint_error = typed_endpoint_owner(d, s, activation, target, typed_consumed)
                if endpoint_error:
                    return [], reject(endpoint_error)
                if consumed_producer != obj(obj(s["input_producers"])[f"{activation}|{consumed_port}"])["producer"]:
                    return [], reject("contradictory")
            consumed_roles[consumed_port] = fact["role"]
        environment = obj(a["environment"])
        if set(cast(str, x) for x in arr(p["absence_queries"])) - set(obj(environment.get("absences", {}))):
            return [], reject("missing")
        if (
            set(obj(environment.get("absences", {}))) != set(cast(str, x) for x in arr(p["absence_queries"]))
            or obj(environment.get("configurations", {})) != {cast(str, p["node"]): p["configuration"]}
            or set(obj(environment.get("state", {}))) != set(cast(str, x) for x in arr(p["read_state"]))
        ):
            return [], reject("contradictory")
        for ref in [a["evidence_artifact"], a["subject_artifact"], *consumed.values()]:
            if ref not in arts:
                return [], reject("missing")
            if obj(arts[cast(str, ref)]).get("invocation") != obj(d["execution"])["invocation"]:
                return [], reject("foreign_owner")
            if obj(arts[cast(str, ref)]).get("target") != target:
                return [], reject("foreign_owner")
        verified_fact = dict(a)
        verified_fact["consumed_roles"] = consumed_roles
        verified_fact["meaning"] = p["meaning"]
        if typed_subject:
            verified_fact["subject"] = {"artifact": a["subject_artifact"], "producer": subject_producer}
        verified.append(verified_fact)
    if len(verified) > cast(int, lim["max_verified_evidence"]):
        return [], reject("limit_exceeded")
    return verified, None


def reconcile(d: Obj, s: Obj) -> tuple[dict[str, set[str]], Obj | None]:
    targets = [cast(str, x) for x in arr(d["targets"])]
    codes = {t: set() for t in targets}
    entries = obj(s["entries"])
    terms = obj(s["terminals"])
    reserv = obj(s["reservations"])
    members = obj(s["memberships"])
    if "__ROOT__" not in members or obj(members["__ROOT__"]).get("parent") is not None:
        return codes, reject("missing")
    owners: dict[str, str | None] = {}
    for membership_key, mraw in members.items():
        m = obj(mraw)
        parent = m.get("parent")
        t = m.get("target")
        root = membership_key == "__ROOT__"
        if root:
            if parent is not None or t is not None:
                return codes, reject("contradictory")
        elif parent != membership_key or t not in codes:
            return codes, reject("foreign_owner")
        if m.get("status") not in ("closed", "failed", "overflow", "open"):
            return codes, reject("invalid_value")
        if m.get("closed") is not (m.get("status") != "open"):
            return codes, reject("contradictory")
        if root:
            if m["closed"] is not True:
                for target in targets:
                    codes[target].add("incomplete_membership")
        elif parent not in entries or parent not in terms:
            codes[cast(str, t)].add("incomplete_membership")
        else:
            parent_entry = obj(entries[cast(str, parent)])
            parent_terminal = obj(terms[cast(str, parent)])
            if parent_entry.get("target") != t:
                return codes, reject("foreign_owner")
            is_map_expander = any(obj(x).get("expander") == parent_entry.get("node") for x in arr(d["map_inputs"]))
            is_subgraph = any(
                obj(x).get("node") == parent_entry.get("node")
                and obj(x).get("outcome") == parent_entry.get("state_outcome")
                for x in arr(d["subgraphs"])
            )
            expected_structural = parent_entry.get("node_kind") == "container"
            if (not is_map_expander and not is_subgraph) or parent_terminal.get(
                "structural"
            ) is not expected_structural:
                return codes, reject("contradictory")
            if (
                parent_terminal.get("category") != parent_entry.get("state_category")
                or parent_terminal.get("outcome") != parent_entry.get("state_outcome")
                or parent_terminal.get("target") != parent_entry.get("target")
            ):
                return codes, reject("contradictory")
            if m.get("status") == "closed" and parent_terminal.get("outcome") != m.get("expansion_outcome"):
                codes[cast(str, t)].add("incomplete_membership")
            if m.get("status") in ("failed", "overflow"):
                codes[cast(str, t)].add("terminal_failure")
        if not root and m["closed"] is not True:
            codes[cast(str, t)].add("incomplete_membership")
        for child in arr(m["members"]):
            c = cast(str, child)
            if c in owners:
                return codes, reject("duplicate")
            owners[c] = None if parent is None else cast(str, parent)
            if c not in entries:
                if root:
                    return codes, reject("missing")
                codes[cast(str, t)].add("incomplete_membership")
                continue
            child_entry = obj(entries[c])
            if child_entry.get("parent") != parent:
                return codes, reject("contradictory")
            if not root and child_entry.get("target") != t:
                return codes, reject("foreign_owner")
    for activation, rraw in reserv.items():
        r = obj(rraw)
        if r.get("target") not in codes:
            return codes, reject("foreign_owner")
        selected = r["selected"] is True
        parent_key = "__ROOT__" if r.get("parent") is None else cast(str, r.get("parent"))
        parent_membership = obj(members.get(parent_key, {}))
        terminal_expansion = parent_membership.get("status") in ("failed", "overflow")
        if selected and activation not in entries and not terminal_expansion:
            codes[cast(str, r["target"])].add("incomplete_membership")
        if not selected and activation in entries:
            return codes, reject("contradictory")
        if selected and activation in entries and r["parent"] is not None and owners.get(activation) != r["parent"]:
            codes[cast(str, r["target"])].add("incomplete_membership")
    for activation, eraw in entries.items():
        e = obj(eraw)
        t = cast(str, e["target"])
        if t not in codes:
            return codes, reject("foreign_owner")
        if obj(d["node_kinds"]).get(cast(str, e.get("node"))) != e.get("node_kind"):
            return codes, reject("contradictory")
        if activation not in owners:
            return codes, reject("missing")
        if owners[activation] != e.get("parent"):
            return codes, reject("contradictory")
        if activation not in terms:
            codes[t].add("incomplete_membership")
        if activation in terms:
            fact = obj(terms[activation])
            expected_structural = e.get("node_kind") == "container"
            if fact.get("structural") is not expected_structural:
                return codes, reject("contradictory")
            if (
                fact.get("category") != e.get("state_category")
                or fact.get("outcome") != e.get("state_outcome")
                or fact.get("target") != e.get("target")
            ):
                return codes, reject("contradictory")
            if fact["category"] != "success":
                codes[t].add("terminal_failure")
    if any(activation not in entries for activation in terms):
        return codes, reject("missing")
    for join in map(obj, arr(d.get("keyed_joins"))):
        sources = [source for source in entries.values() if obj(source).get("node") == join["source"]]
        for source in map(obj, sources):
            matching = [
                candidate
                for candidate in entries.values()
                if obj(candidate).get("node") == join["join"]
                and obj(candidate).get("target") == source.get("target")
                and obj(candidate).get("parent") == source.get("parent")
            ]
            if len(matching) != 1:
                return codes, reject("missing" if not matching else "duplicate")
    return codes, None


def validity(a: Obj, s: Obj) -> str:
    rev = obj(s["revision"])
    changed = False
    missing = False
    deps = [a["evidence_artifact"], a["subject_artifact"], *obj(a["consumed"]).values()]
    for ref in deps:
        art = obj(obj(s["artifacts"])[cast(str, ref)])
        key = cast(str, art["key"])
        current = obj(rev["artifacts"])
        if key not in current:
            missing = True
        elif obj(current[key])["value"] != art["version"]:
            changed = True
    env = obj(a["environment"])
    for group in ("absences", "configurations", "state"):
        for key, value in obj(env[group]).items():
            current = obj(rev[group])
            if key not in current:
                missing = True
            elif obj(current[key])["value"] != value:
                changed = True
    return "stale" if changed else "unknown" if missing else "current"


def request_codes(d: Obj, s: Obj) -> dict[str, set[str]]:
    targets = [cast(str, x) for x in arr(d["targets"])]
    out = {t: set() for t in targets}
    requests = obj(s["requests"])
    latest: dict[str, str] = {}
    for rid, rraw in requests.items():
        r = obj(rraw)
        for t in arr(r["associations"]):
            target = cast(str, t)
            if target not in out:
                for owned in targets:
                    out[owned].add("inconsistent_attribution")
            else:
                latest[target] = rid
    allowed = {"retryable": "retry", "malformed": "correction", "permanent": "failover"}
    for rid, rraw in requests.items():
        r = obj(rraw)
        terminal = obj(r.get("terminal", {}))
        sett = obj(r.get("settlement", {}))
        assoc = [cast(str, x) for x in arr(r["associations"])]
        unresolved = (
            not terminal
            or not sett
            or sett.get("remote_stopped") is not True
            or sett.get("usage") == "unknown"
            or terminal.get("failure") in ("lost", "inconsistent")
        )
        if unresolved:
            for t in assoc:
                out[t].add("request_uncertain")
        elif terminal.get("condition") != "result" and any(latest.get(t) == rid for t in assoc):
            for t in assoc:
                out[t].add("request_accounting")
        elif terminal.get("condition") != "result":
            successor = [obj(x) for x in requests.values() if obj(x).get("predecessor") == rid]
            expected = allowed.get(cast(str, terminal.get("failure")))
            valid_successor = (
                len(successor) == 1
                and expected is not None
                and successor[0].get("purpose") == expected
                and successor[0].get("policy") == r.get("policy")
                and successor[0].get("associations") == r.get("associations")
            )
            if not valid_successor:
                for t in assoc:
                    out[t].add("request_accounting")
    return out


def cleanup_codes(d: Obj, s: Obj) -> dict[str, set[str]]:
    targets = [cast(str, x) for x in arr(d["targets"])]
    out = {t: set() for t in targets}
    ledgers = (
        (obj(s["cleanup_associations"]), obj(s["cleanups"]), False),
        (obj(s["binding_cleanup_associations"]), obj(s["binding_cleanups"]), True),
    )
    for assocs, cleanups, binding in ledgers:
        if set(assocs) != set(cleanups):
            for t in targets:
                out[t].add("inconsistent_attribution")
        for resource, craw in cleanups.items():
            c = obj(craw)
            a = obj(assocs.get(resource, {}))
            association_targets = [cast(str, target) for target in arr(a.get("targets"))]
            if (
                not a
                or (not association_targets and a.get("purpose") != "transport_only")
                or len(association_targets) != len(set(association_targets))
                or (binding and (a.get("owner") != "sdk" or a.get("purpose") != "accounting"))
            ):
                for t in targets:
                    out[t].add("inconsistent_attribution")
            elif any(cast(str, t) not in out for t in arr(a["targets"])):
                for t in targets:
                    out[t].add("inconsistent_attribution")
            elif (
                c["disposition"] not in ("closed",)
                and not (c["disposition"] == "left_open" and a["owner"] == "caller")
                and a["purpose"] != "transport_only"
            ):
                code = "cleanup_verification" if a["purpose"] == "verification" else "cleanup_accounting"
                for t in arr(a["targets"]):
                    out[cast(str, t)].add(code)
    return out


def qualify(d: Obj, s: Obj) -> Obj:
    targets = [cast(str, x) for x in arr(d["targets"])]
    lim = obj(d["limits"])
    if s["execution"] != d["execution"]:
        return reject("foreign_owner")
    if s["revision_sealed"] is not True:
        return reject("missing")
    if len(obj(s["ports"])) > cast(int, lim["max_port_facts"]):
        return reject("limit_exceeded")
    if len(obj(s["artifacts"])) > cast(int, lim["max_artifacts"]) or sum(
        cast(int, obj(value)["bytes"]) for value in obj(s["artifacts"]).values()
    ) > cast(int, lim["max_artifact_bytes"]):
        return reject("limit_exceeded")
    if sum(len(arr(obj(x)["parents"])) for x in obj(s["provenance"]).values()) > cast(int, lim["max_provenance_edges"]):
        return reject("limit_exceeded")
    if sum(len(obj(obj(s["revision"])[k])) for k in ("artifacts", "configurations", "state")) > cast(
        int, lim["max_revision_entries"]
    ) or len(obj(obj(s["revision"])["absences"])) > cast(int, lim["max_absence_revisions"]):
        return reject("limit_exceeded")
    revision = obj(s["revision"])
    retained_pairs = {(cast(str, obj(value)["key"]), obj(value)["version"]) for value in obj(s["artifacts"]).values()}
    if any((key, obj(value).get("value")) not in retained_pairs for key, value in obj(revision["artifacts"]).items()):
        return reject("missing")
    admitted_absences = {
        cast(str, query)
        for production in map(obj, arr(d["productions"]))
        for query in arr(production["absence_queries"])
    }
    admitted_state = {
        cast(str, effect) for production in map(obj, arr(d["productions"])) for effect in arr(production["read_state"])
    }
    if set(obj(revision["absences"])) - admitted_absences:
        return reject("unsupported")
    if set(obj(revision["configurations"])) - set(obj(d["node_kinds"])):
        return reject("missing")
    if set(obj(revision["state"])) - admitted_state:
        return reject("unsupported")
    execution_only = d["purpose"] == "execution_only"
    if set(obj(s["input_producers"])) - set(obj(s["ports"])):
        return reject("contradictory")
    verified: list[Obj] = []
    if not execution_only:
        verified, bad = authenticate(d, s)
        if bad:
            return bad
    codes, bad = reconcile(d, s)
    if bad:
        return bad
    for source in (request_codes(d, s), cleanup_codes(d, s)):
        for t in targets:
            codes[t].update(source[t])
    finals = obj(s["finals"])
    if set(finals) - set(targets):
        return reject("foreign_owner")
    if any(obj(value).get("outcome") != "ok" for value in finals.values()):
        return reject("contradictory")
    arts = obj(s["artifacts"])
    ports = obj(s["ports"])
    prov = obj(s["provenance"])
    materialized_by_artifact: dict[str, list[Obj]] = {}
    for raw_value in prov.values():
        value = obj(raw_value)
        if value.get("source") in ("bound_input", "map_item"):
            materialized_by_artifact.setdefault(cast(str, value["artifact"]), []).append(value)
    artifact_lineages: dict[tuple[str, str], list[tuple[str, Obj]]] = {}
    for ref, raw_artifact in arts.items():
        artifact_value = obj(raw_artifact)
        artifact_lineages.setdefault(
            (cast(str, artifact_value["invocation"]), cast(str, artifact_value["key"])), []
        ).append((ref, artifact_value))
    for (invocation, _), versions in artifact_lineages.items():
        if len(versions) < 2:
            continue
        if len({obj(value)["version"] for _, value in versions}) != len(versions):
            return reject("duplicate")
        identities = []
        for ref, artifact_value in versions:
            owners = materialized_by_artifact.get(ref, [])
            if len(owners) != 1:
                return reject("missing")
            owner = owners[0]
            if owner["source"] == "bound_input":
                binding = obj(owner["binding_artifact"])
                if binding.get("version") != artifact_value["version"]:
                    return reject("contradictory")
                identities.append(
                    (
                        "initial",
                        invocation,
                        binding.get("declaration"),
                        binding.get("key"),
                        owner.get("target"),
                    )
                )
            else:
                if owner.get("item_version") != artifact_value["version"]:
                    return reject("contradictory")
                expander_entry = obj(obj(s["entries"]).get(cast(str, owner.get("expander")), {}))
                member_entry = obj(obj(s["entries"]).get(cast(str, owner.get("member")), {}))
                if not expander_entry:
                    return reject("missing")
                if (
                    owner.get("target") not in targets
                    or expander_entry.get("target") != owner.get("target")
                    or member_entry.get("target") != owner.get("target")
                ):
                    return reject("foreign_owner")
                identities.append(
                    (
                        "map",
                        invocation,
                        owner.get("expander"),
                        owner.get("target"),
                        owner.get("item_key"),
                    )
                )
        if len(set(identities)) != 1:
            return reject("contradictory")
    allowed_sources = {
        "bound_input",
        "initial_collection",
        "map_item",
        "operation_output",
        "root_input",
    }
    bindings = [obj(x) for x in arr(d["binding_inputs"])]
    collections = [obj(x) for x in arr(d["initial_collections"])]
    map_inputs = [obj(x) for x in arr(d["map_inputs"])]
    receipt = obj(s["binding_receipt"])
    receipt_sources = [obj(source) for source in arr(receipt.get("sources"))]
    receipt_artifacts = [obj(item) for item in arr(receipt.get("artifacts"))]
    if receipt:
        source_sites = [
            {field: source[field] for field in ("declaration", "node", "port", "source", "target")}
            for source in receipt_sources
        ]
        if len({cast(str, source["declaration"]) for source in receipt_sources}) != len(receipt_sources):
            return reject("duplicate")
        expected_source_sites = [
            {field: binding[field] for field in ("declaration", "node", "port", "source", "target")}
            for binding in bindings
        ]
        if source_sites != expected_source_sites:
            return reject("contradictory")
        artifacts_by_declaration: dict[str, list[Obj]] = {}
        for item in receipt_artifacts:
            artifacts_by_declaration.setdefault(cast(str, item["declaration"]), []).append(item)
        for binding in bindings:
            retained = artifacts_by_declaration.get(cast(str, binding["declaration"]), [])
            if binding["materialization"] == "single" and binding["version_selection"] == "exact_one":
                if len(retained) != 1:
                    return reject("contradictory")
            elif binding["materialization"] == "single" and binding["version_selection"] == "latest":
                keys = {item["key"] for item in retained}
                if not retained or len(keys) != 1:
                    return reject("contradictory")
            if binding["version_selection"] != "latest":
                continue
            requests = [
                request
                for request in map(obj, obj(s["binding_requests"]).values())
                if obj(request.get("association")).get("declaration") == binding["declaration"]
            ]
            if len(requests) != 1:
                return reject("foreign_owner" if obj(s["binding_requests"]) else "missing")
            request = requests[0]
            association = obj(request["association"])
            if association != {field: binding[field] for field in ("declaration", "node", "port", "source", "target")}:
                return reject("foreign_owner")
            if (
                request.get("phase") != "settled"
                or obj(request.get("terminal")).get("outcome") != "retrieved"
                or obj(request.get("settlement")) != {"remote_stopped": True, "usage": "known"}
            ):
                return reject("missing")
            resource = cast(str, request["resource"])
            cleanup_association = obj(obj(s["binding_cleanup_associations"]).get(resource, {}))
            cleanup = obj(obj(s["binding_cleanups"]).get(resource, {}))
            if not cleanup_association or not cleanup:
                return reject("missing")
            if cleanup_association != {"owner": "sdk", "purpose": "accounting", "targets": [binding["target"]]}:
                return reject("foreign_owner")
            if cleanup.get("disposition") != "closed":
                return reject("contradictory")
        for item in receipt_artifacts:
            site = {field: item[field] for field in ("declaration", "node", "port", "source", "target")}
            if site not in source_sites:
                return reject("foreign_owner")
    entries = obj(s["entries"])
    memberships = obj(s["memberships"])
    admitted_nodes = obj(d["node_kinds"])
    output_dependencies = {
        (item["node"], item["outcome"], item["port"]): item for item in map(obj, arr(d["output_dependencies"]))
    }
    input_producers = obj(s["input_producers"])
    admitted_input_ports: dict[str, set[str]] = {}
    for dependency in output_dependencies.values():
        admitted_input_ports.setdefault(cast(str, dependency["node"]), set()).update(
            cast(str, value) for value in arr(dependency["inputs"])
        )
    for production in map(obj, arr(d["productions"])):
        admitted_input_ports.setdefault(cast(str, production["node"]), set()).update(
            cast(str, value) for value in arr(production["consumed_ports"])
        )
        if production["subject_source"] == "input":
            admitted_input_ports.setdefault(cast(str, production["node"]), set()).add(
                cast(str, production["subject_port"])
            )
    for raw_subgraph in arr(d["subgraphs"]):
        subgraph = obj(raw_subgraph)
        admitted_input_ports.setdefault(cast(str, subgraph["node"]), set()).add(cast(str, subgraph["input_port"]))
        if subgraph["body_source"] == "node_output":
            admitted_input_ports.setdefault(cast(str, subgraph["body_node"]), set()).add(
                cast(str, subgraph["body_input"])
            )
    expected_input_keys = {
        key
        for key, raw_port in ports.items()
        if obj(raw_port).get("port") in admitted_input_ports.get(cast(str, obj(raw_port).get("node")), set())
        and obj(raw_port).get("activation") in entries
    }
    dynamic_item_ports = {cast(str, map_input["item_input"]) for map_input in map_inputs}
    expected_input_keys.update(
        key
        for key, raw_port in ports.items()
        if obj(raw_port).get("port") in dynamic_item_ports
        and obj(entries.get(cast(str, obj(raw_port).get("activation")), {})).get("parent") is not None
    )
    if expected_input_keys - set(input_producers):
        return reject("missing")
    if set(input_producers) - expected_input_keys:
        return reject("contradictory")
    for key, raw_input in input_producers.items():
        input_fact = obj(raw_input)
        port_fact = obj(ports.get(key, {}))
        producer_fact = obj(prov.get(cast(str, input_fact.get("producer")), {}))
        if producer_fact and producer_fact.get("artifact") not in arts:
            return reject("missing")
        if producer_fact and producer_fact.get("target") not in targets:
            return reject("foreign_owner")
        if (
            input_fact.get("activation") != port_fact.get("activation")
            or input_fact.get("node") != port_fact.get("node")
            or input_fact.get("port") != port_fact.get("port")
            or input_fact.get("target") != port_fact.get("target")
            or producer_fact.get("artifact") != port_fact.get("artifact")
            or producer_fact.get("target") != port_fact.get("target")
        ):
            return reject("contradictory")
    for producer, raw_collection in obj(s["collection_values"]).items():
        collection = obj(raw_collection)
        producer_fact = obj(prov.get(producer, {}))
        if not producer_fact:
            return reject("missing")
        activation = cast(str, producer_fact.get("activation"))
        source_entry = obj(entries.get(activation, {}))
        owners = [
            map_input
            for map_input in map_inputs
            if map_input.get("expander") == source_entry.get("node")
            and map_input.get("outcome") == source_entry.get("state_outcome")
            and map_input.get("membership_port") == producer_fact.get("port")
            and producer == f"OP:{activation}:{collection.get('target')}:{map_input.get('membership_port')}"
        ]
        if len(owners) != 1:
            return reject("contradictory")
        if (
            producer_fact.get("source") != "operation_output"
            or producer_fact.get("artifact") != collection.get("artifact")
            or producer_fact.get("target") != collection.get("target")
        ):
            return reject("contradictory")
    for provenance_key, raw_value in prov.items():
        value = obj(raw_value)
        source = value.get("source")
        if source not in allowed_sources:
            return reject("unsupported")
        if value.get("target") not in targets:
            return reject("foreign_owner")
        artifact_ref = cast(str, value.get("artifact"))
        artifact_value = obj(arts.get(artifact_ref, {}))
        if not artifact_value:
            return reject("missing")
        if artifact_value.get("invocation") != obj(d["execution"])["invocation"]:
            return reject("foreign_owner")
        if artifact_value.get("target") not in (value.get("target"), "shared"):
            return reject("foreign_owner")
        if type(value.get("decision")) is not bool:
            return reject("invalid_type")
        if source in ("operation_output", "root_input"):
            if any(
                value.get(field) is not None
                for field in (
                    "binding_artifact",
                    "declaration",
                    "expander",
                    "item_key",
                    "item_version",
                    "member",
                    "root_target",
                )
            ):
                return reject("contradictory")
            source_matches = [
                fact
                for fact in map(obj, ports.values())
                if fact.get("activation") == value.get("activation")
                and fact.get("artifact") == artifact_ref
                and fact.get("node") == value.get("node")
                and fact.get("port") == value.get("port")
                and fact.get("target") == value.get("target")
            ]
            if len(source_matches) != 1:
                return reject("contradictory")
            source_entry = obj(entries.get(cast(str, value.get("activation")), {}))
            map_output = any(
                map_input.get("expander") == value.get("node")
                and map_input.get("outcome") == source_entry.get("state_outcome")
                and map_input.get("membership_port") == value.get("port")
                for map_input in map_inputs
            )
            structural_output = next(
                (
                    obj(item)
                    for item in arr(d["subgraphs"])
                    if obj(item).get("node") == value.get("node")
                    and obj(item).get("outcome") == source_entry.get("state_outcome")
                    and obj(item).get("port") == value.get("port")
                ),
                {},
            )
            if (
                admitted_nodes.get(cast(str, value.get("node"))) != "operation"
                and not map_output
                and not structural_output
            ) or (source_entry.get("node") != value.get("node") or source_entry.get("target") != value.get("target")):
                return reject("contradictory")
            if source == "root_input" and arr(value["parents"]):
                return reject("contradictory")
            if source == "operation_output":
                if structural_output:
                    wrapper_input = obj(
                        input_producers.get(f"{value.get('activation')}|{structural_output.get('input_port')}", {})
                    )
                    if structural_output.get("body_source") == "workflow_input":
                        actual_parents = [cast(str, parent) for parent in arr(value["parents"])]
                        if (
                            len(actual_parents) != 1
                            or actual_parents[0] != wrapper_input.get("producer")
                            or obj(prov.get(actual_parents[0], {})).get("artifact") != artifact_ref
                            or wrapper_input.get("target") != value.get("target")
                        ):
                            return reject("contradictory")
                        continue
                    body_parents = [
                        key
                        for key, candidate_raw in prov.items()
                        if obj(candidate_raw).get("source") == "operation_output"
                        and obj(candidate_raw).get("node") == structural_output.get("body_node")
                        and obj(candidate_raw).get("port") == structural_output.get("body_port")
                        and obj(candidate_raw).get("target") == value.get("target")
                        and obj(entries.get(cast(str, obj(candidate_raw).get("activation")), {})).get("parent")
                        == value.get("activation")
                    ]
                    if len(arr(value["parents"])) != len(set(cast(str, x) for x in arr(value["parents"]))) or set(
                        cast(str, x) for x in arr(value["parents"])
                    ) != set(body_parents):
                        return reject("contradictory")
                    body_parent = obj(prov.get(body_parents[0], {})) if len(body_parents) == 1 else {}
                    body_entry = obj(entries.get(cast(str, body_parent.get("activation")), {}))
                    body_input = obj(
                        input_producers.get(
                            f"{body_parent.get('activation')}|{structural_output.get('body_input')}", {}
                        )
                    )
                    if (
                        body_parent.get("artifact") != artifact_ref
                        or body_entry.get("node") != structural_output.get("body_node")
                        or wrapper_input.get("producer") != body_input.get("producer")
                        or wrapper_input.get("target") != value.get("target")
                        or body_input.get("target") != value.get("target")
                    ):
                        return reject("contradictory")
                    continue
                dependency = output_dependencies.get(
                    (value.get("node"), source_entry.get("state_outcome"), value.get("port"))
                ) or output_dependencies.get((value.get("node"), "ok", value.get("port")))
                if dependency is None:
                    return reject("missing")
                producer_facts = [
                    obj(input_producers.get(f"{value.get('activation')}|{input_port}", {}))
                    for input_port in arr(dependency["inputs"])
                ]
                if any(
                    not fact or fact.get("node") != value.get("node") or fact.get("target") != value.get("target")
                    for fact in producer_facts
                ):
                    return reject("missing")
                expected_parents = {cast(str, fact["producer"]) for fact in producer_facts}
                actual_parents = [cast(str, parent) for parent in arr(value["parents"])]
                if len(actual_parents) != len(set(actual_parents)) or set(actual_parents) != expected_parents:
                    return reject("contradictory")
                identity_input = dependency.get("identity_input")
                if identity_input is not None:
                    identity_fact = obj(input_producers.get(f"{value.get('activation')}|{identity_input}", {}))
                    identity_parent = obj(prov.get(cast(str, identity_fact.get("producer")), {}))
                    if identity_parent.get("artifact") != artifact_ref:
                        return reject("contradictory")
                elif any(obj(prov.get(parent, {})).get("artifact") == artifact_ref for parent in expected_parents):
                    return reject("contradictory")
        if source == "bound_input":
            binding = obj(value.get("binding_artifact"))
            if set(binding) != {"declaration", "key", "version"}:
                return reject("invalid_type")
            if (
                type(binding.get("key")) is not int
                or cast(int, binding["key"]) < 0
                or type(binding.get("version")) is not int
                or cast(int, binding["version"]) <= 0
            ):
                return reject("invalid_value")
            identity = {
                "declaration": binding.get("declaration"),
                "node": value.get("node"),
                "port": value.get("port"),
                "target": value.get("target"),
            }
            binding_matches = [
                admitted
                for admitted in bindings
                if all(admitted.get(field) == field_value for field, field_value in identity.items())
            ]
            if len(binding_matches) != 1:
                return reject("contradictory")
            receipt_match = [
                item
                for item in receipt_artifacts
                if item.get("declaration") == binding.get("declaration")
                and item.get("key") == binding.get("key")
                and item.get("version") == binding.get("version")
                and item.get("node") == value.get("node")
                and item.get("port") == value.get("port")
                and item.get("target") == value.get("target")
            ]
            if len(receipt_match) != 1:
                return reject("missing")
            if any(
                value.get(field) is not None
                for field in (
                    "activation",
                    "declaration",
                    "expander",
                    "item_key",
                    "item_version",
                    "member",
                    "root_target",
                )
            ):
                return reject("contradictory")
            if value.get("decision") is not False or arr(value["parents"]):
                return reject("contradictory")
        if source == "initial_collection":
            identity = {
                "declaration": value.get("declaration"),
                "node": value.get("node"),
                "port": value.get("port"),
                "target": value.get("target"),
            }
            if identity not in collections:
                return reject("contradictory")
            if any(
                value.get(field) is not None
                for field in (
                    "activation",
                    "binding_artifact",
                    "expander",
                    "item_key",
                    "item_version",
                    "member",
                    "root_target",
                )
            ):
                return reject("contradictory")
            expected_parents = {
                key
                for key, parent_raw in prov.items()
                if obj(parent_raw).get("source") == "bound_input"
                and obj(obj(parent_raw).get("binding_artifact")).get("declaration") == value.get("declaration")
                and obj(parent_raw).get("node") == value.get("node")
                and obj(parent_raw).get("port") == value.get("port")
                and obj(parent_raw).get("target") == value.get("target")
            }
            expected_inventory = {
                (
                    item.get("declaration"),
                    item.get("key"),
                    item.get("version"),
                    item.get("node"),
                    item.get("port"),
                    item.get("target"),
                )
                for item in receipt_artifacts
                if item.get("declaration") == value.get("declaration")
                and item.get("node") == value.get("node")
                and item.get("port") == value.get("port")
                and item.get("target") == value.get("target")
            }
            actual_inventory = {
                (
                    obj(obj(prov[parent]).get("binding_artifact")).get("declaration"),
                    obj(obj(prov[parent]).get("binding_artifact")).get("key"),
                    obj(obj(prov[parent]).get("binding_artifact")).get("version"),
                    obj(prov[parent]).get("node"),
                    obj(prov[parent]).get("port"),
                    obj(prov[parent]).get("target"),
                )
                for parent in expected_parents
            }
            if (
                set(cast(str, parent) for parent in arr(value["parents"])) != expected_parents
                or actual_inventory != expected_inventory
            ):
                return reject("contradictory")
            if value.get("decision") is not False:
                return reject("contradictory")
        if source == "map_item":
            member = cast(str, value.get("member"))
            expander = cast(str, value.get("expander"))
            member_entry = obj(entries.get(member, {}))
            expander_entry = obj(entries.get(expander, {}))
            if not expander_entry:
                return reject("missing")
            matches = [
                map_input
                for map_input in map_inputs
                if map_input.get("expander") == expander_entry.get("node")
                and map_input.get("outcome") == expander_entry.get("state_outcome")
            ]
            membership = obj(memberships.get(expander, {}))
            item_key = value.get("item_key")
            item_version = value.get("item_version")
            parent_keys = [cast(str, parent) for parent in arr(value["parents"])]
            if len(matches) != 1:
                return reject("contradictory")
            map_input = matches[0]
            if (
                value.get("target") not in targets
                or expander_entry.get("target") != value.get("target")
                or member_entry.get("target") != value.get("target")
                or membership.get("target") != value.get("target")
            ):
                return reject("foreign_owner")
            expected_parent = f"OP:{expander}:{value.get('target')}:{map_input['membership_port']}"
            if (
                value.get("activation") is not None
                or value.get("binding_artifact") is not None
                or value.get("declaration") is not None
                or value.get("node") is not None
                or value.get("root_target") is not None
                or value.get("port") != map_input["item_input"]
                or value.get("decision") is not False
                or type(item_key) is not int
                or item_key < 0
                or type(item_version) is not int
                or item_version <= 0
                or expander_entry.get("node_kind") not in ("operation", "container")
                or member_entry.get("parent") != expander
                or member not in arr(membership.get("members"))
                or len(parent_keys) != 1
                or parent_keys[0] != expected_parent
            ):
                return reject("contradictory")
            parent_fact = obj(prov.get(parent_keys[0], {}))
            collection = obj(obj(s["collection_values"]).get(parent_keys[0], {}))
            member_port = obj(ports.get(f"{member}|{map_input['item_input']}", {}))
            membership_members = [cast(str, item) for item in arr(membership.get("members"))]
            ordered_members = sorted(membership_members, key=lambda item: cast(int, obj(entries[item])["occurrence"]))
            occurrences = [obj(entries[item]).get("occurrence") for item in ordered_members]
            if len(occurrences) != len(set(occurrences)):
                return reject("contradictory")
            member_index = ordered_members.index(member) if member in ordered_members else -1
            items = [obj(item) for item in arr(collection.get("items"))]
            if (
                parent_fact.get("source") != "operation_output"
                or parent_fact.get("activation") != expander
                or parent_fact.get("node") != map_input["expander"]
                or parent_fact.get("port") != map_input["membership_port"]
                or parent_fact.get("target") != value.get("target")
                or collection.get("artifact") != parent_fact.get("artifact")
                or collection.get("target") != value.get("target")
                or len(items) != len(ordered_members)
                or member_index < 0
                or items[member_index].get("key") != item_key
                or items[member_index].get("version") != item_version
                or member_port.get("artifact") != artifact_ref
                or member_port.get("target") != value.get("target")
                or member_port.get("node") != member_entry.get("node")
            ):
                return reject("contradictory")
        if value.get("decision") is True and not any(
            fact.get("artifact") == artifact_ref
            and fact.get("role") == "decision"
            and fact.get("target") == value.get("target")
            for fact in map(obj, ports.values())
        ):
            return reject("contradictory")
    for binding in bindings:
        if binding["materialization"] != "single" or binding["version_selection"] != "latest":
            continue
        inventory = [item for item in receipt_artifacts if item.get("declaration") == binding["declaration"]]
        maximum = max(cast(int, item["version"]) for item in inventory)
        selected_owners = [
            (key, value)
            for key, value in map(lambda pair: (pair[0], obj(pair[1])), prov.items())
            if value.get("source") == "bound_input"
            and obj(value.get("binding_artifact")).get("declaration") == binding["declaration"]
            and obj(value.get("binding_artifact")).get("version") == maximum
        ]
        if len(selected_owners) != 1:
            return reject("missing")
        selected_key, selected_owner = selected_owners[0]
        destination_ports = [
            value
            for value in map(obj, ports.values())
            if value.get("node") == binding["node"]
            and value.get("port") == binding["port"]
            and value.get("target") == binding["target"]
        ]
        if len(destination_ports) != 1:
            return reject("missing")
        destination = destination_ports[0]
        producer = obj(input_producers.get(f"{destination.get('activation')}|{binding['port']}", {}))
        if destination.get("artifact") != selected_owner.get("artifact") or producer.get("producer") != selected_key:
            return reject("contradictory")
    valid = []
    for a in verified:
        a["validity"] = validity(a, s)
        valid.append(a)
    supporting: dict[str, list[Obj]] = {target: [] for target in targets}
    candidate_uncertain: set[str] = set()
    for t in targets:
        f = obj(finals.get(t, {}))
        if not f:
            codes[t].add("missing_candidate")
            continue
        candidate = cast(str, f["candidate"])
        p = obj(prov.get(cast(str, f["producer"]), {}))
        root_output = next((obj(x) for x in arr(d["root_outputs"]) if obj(x).get("target") == t), {})
        port_matches = [
            obj(x)
            for x in ports.values()
            if obj(x).get("activation") == f["activation"]
            and obj(x).get("artifact") == candidate
            and obj(x).get("node") == f["node"]
            and obj(x).get("role") == "candidate"
            and obj(x).get("target") == t
            and obj(x).get("port") == f["port"]
        ]
        if (
            candidate not in arts
            or obj(arts.get(candidate, {})).get("target") != t
            or obj(arts.get(candidate, {})).get("invocation") != obj(d["execution"])["invocation"]
        ):
            codes[t].add("missing_candidate")
        candidate_artifact = obj(arts.get(candidate, {}))
        candidate_key = candidate_artifact.get("key")
        candidate_version = candidate_artifact.get("version")
        candidate_revision = obj(obj(s["revision"])["artifacts"]).get(cast(str, candidate_key))
        candidate_current = (
            bool(candidate_artifact)
            and isinstance(candidate_revision, dict)
            and obj(candidate_revision).get("value") == candidate_version
        )
        if not candidate_current:
            candidate_uncertain.add(t)
            codes[t].add("missing_candidate")
        if not p:
            return reject("missing")
        direct_root = root_output.get("input_port") is not None
        if direct_root:
            expected_key = f"ROOT:{t}:{root_output['input_port']}"
            if f.get("producer") != expected_key or p.get("source") != "root_input":
                return reject("contradictory")
        elif (
            f.get("node") != root_output.get("source_node")
            or f.get("port") != root_output.get("port")
            or p.get("node") != root_output.get("source_node")
            or p.get("port") != root_output.get("source_port")
        ):
            return reject("contradictory")
        if p.get("artifact") != candidate or p.get("target") != t or (not direct_root and len(port_matches) != 1):
            return reject("contradictory")
        reqs = [obj(x) for x in arr(d["requirements"]) if obj(x)["target"] == t]
        for req in () if execution_only else reqs:
            eligible_productions = [
                p
                for p in map(obj, arr(d["productions"]))
                if p["meaning"] == req["meaning"]
                and p["outcome"] == req["outcome"]
                and p["subject_port"] == req["subject_port"]
                and set(cast(str, x) for x in arr(req["consumed_ports"])).issubset(
                    set(cast(str, x) for x in arr(p["consumed_ports"]))
                )
            ]
            eligible_promises = {cast(str, p["promise"]) for p in eligible_productions}
            input_subject_promises = {
                cast(str, p["promise"]) for p in eligible_productions if p["subject_source"] == "input"
            }

            def same_candidate_lineage(assessment_value: Obj) -> bool:
                subject = obj(arts.get(cast(str, assessment_value["subject_artifact"]), {}))
                final_artifact = obj(arts.get(candidate, {}))
                return subject.get("invocation") == final_artifact.get("invocation") and subject.get(
                    "key"
                ) == final_artifact.get("key")

            matches = [
                a
                for a in valid
                if a["target"] == t
                and (
                    a["promise"] in input_subject_promises
                    or a["subject_artifact"] == candidate
                    or same_candidate_lineage(a)
                )
                and a["promise"] in eligible_promises
                and a["meaning"] == req["meaning"]
                and a["outcome"] == req["outcome"]
                and a["subject_port"] == req["subject_port"]
                and set(cast(str, x) for x in arr(req["consumed_ports"])).issubset(set(obj(a["consumed"])))
            ]
            if not matches:
                codes[t].add("missing_assessment")
                continue
            complete = [
                a
                for a in matches
                if a["validity"] == "current"
                and a["finding"] == "satisfied"
                and set(cast(str, x) for x in arr(req["coverage"])).issubset(
                    set(cast(str, x) for x in arr(a["coverage"]))
                )
            ]
            if complete:
                for a in complete:
                    if a not in supporting[t]:
                        supporting[t].append(a)
                continue
            if any(a["validity"] == "stale" for a in matches):
                codes[t].add("stale_evidence")
            if any(a["validity"] == "unknown" or a["finding"] == "unknown" for a in matches):
                codes[t].add("assessment_unknown")
            if any(a["finding"] == "unsatisfied" for a in matches):
                codes[t].add("assessment_unsatisfied")
            if any(
                a["validity"] == "current"
                and a["finding"] == "satisfied"
                and not set(cast(str, x) for x in arr(req["coverage"])).issubset(
                    set(cast(str, x) for x in arr(a["coverage"]))
                )
                for a in matches
            ):
                codes[t].add("incomplete_coverage")

        def provenance_ancestry(key: str) -> set[str]:
            pending = [key]
            seen: set[str] = set()
            while pending:
                current = pending.pop()
                if current in seen:
                    continue
                seen.add(current)
                pending.extend(cast(str, parent) for parent in arr(obj(prov.get(current, {})).get("parents")))
            return seen

        final_ancestry = provenance_ancestry(cast(str, f.get("producer")))
        for typed_requirement in (
            requirement
            for requirement in map(obj, arr(d.get("map_item_requirements")))
            if requirement.get("target") == t
        ):
            endpoints = [
                obj(endpoint)
                for endpoint in [
                    typed_requirement.get("subject_endpoint"),
                    *arr(typed_requirement.get("consumed_endpoints")),
                ]
                if endpoint is not None
            ]
            endpoint = endpoints[0]
            if f.get("port") != typed_requirement.get("candidate_port"):
                codes[t].add("missing_candidate")
                continue
            expander_activations = [
                activation
                for activation, raw_entry in entries.items()
                if obj(raw_entry).get("node") == endpoint["expander"]
                and obj(raw_entry).get("state_outcome") == endpoint["expansion_outcome"]
                and obj(raw_entry).get("target") == t
            ]
            scoped_members: list[str] = []
            ancestry_complete = True
            for expander_activation in expander_activations:
                membership_producer = f"OP:{expander_activation}:{t}:{endpoint['membership_port']}"
                if membership_producer not in final_ancestry:
                    ancestry_complete = False
                membership = obj(memberships.get(expander_activation, {}))
                for member in (cast(str, value) for value in arr(membership.get("members"))):
                    member_entry = obj(entries.get(member, {}))
                    member_terminal = obj(obj(s["terminals"]).get(member, {}))
                    if (
                        member_entry.get("node") == endpoint["member"]
                        and member_entry.get("parent") == expander_activation
                        and member_entry.get("closed_unstarted") is False
                        and member_entry.get("state_category") == "success"
                        and member_terminal.get("category") == "success"
                        and member_terminal.get("outcome") == member_entry.get("state_outcome")
                        and member_terminal.get("target") == t
                    ):
                        scoped_members.append(member)
            if not ancestry_complete:
                codes[t].add("missing_assessment")
                continue
            for member in scoped_members:
                matches = [
                    assessment_value
                    for assessment_value in valid
                    if assessment_value["activation"] == member
                    and assessment_value["target"] == t
                    and assessment_value["promise"] == typed_requirement["promise"]
                    and assessment_value["meaning"] == typed_requirement["meaning"]
                ]
                if not matches:
                    codes[t].add("missing_assessment")
                    continue
                complete = [
                    assessment_value
                    for assessment_value in matches
                    if assessment_value["validity"] == "current"
                    and assessment_value["finding"] == "satisfied"
                    and set(cast(str, value) for value in arr(typed_requirement["coverage"])).issubset(
                        set(cast(str, value) for value in arr(assessment_value["coverage"]))
                    )
                ]
                if complete:
                    for assessment_value in complete:
                        if assessment_value not in supporting[t]:
                            supporting[t].append(assessment_value)
                    continue
                if any(assessment_value["validity"] == "stale" for assessment_value in matches):
                    codes[t].add("stale_evidence")
                if any(
                    assessment_value["validity"] == "unknown" or assessment_value["finding"] == "unknown"
                    for assessment_value in matches
                ):
                    codes[t].add("assessment_unknown")
                if any(assessment_value["finding"] == "unsatisfied" for assessment_value in matches):
                    codes[t].add("assessment_unsatisfied")
                if any(
                    assessment_value["validity"] == "current"
                    and assessment_value["finding"] == "satisfied"
                    and not set(cast(str, value) for value in arr(typed_requirement["coverage"])).issubset(
                        set(cast(str, value) for value in arr(assessment_value["coverage"]))
                    )
                    for assessment_value in matches
                ):
                    codes[t].add("incomplete_coverage")
    for _ in range(cast(int, lim["max_fixed_point_steps"])):
        before = sum(map(len, codes.values()))
        for raw in arr(d["dependencies"]):
            dep = obj(raw)
            a = cast(str, dep["prerequisite"])
            b = cast(str, dep["dependent"])
            if codes[a]:
                codes[b].add("dependency")
        for raw in arr(d["atomic"]):
            group = [cast(str, x) for x in arr(raw)]
            if len(group) > 1 and any(codes[x] for x in group):
                for x in group:
                    codes[x].add("atomic_group")
        if sum(map(len, codes.values())) == before:
            break

    def decisions(t: str) -> list[str]:
        stack = [cast(str, obj(finals[t])["producer"])]
        seen = set()
        out = set()
        for a in supporting[t]:
            for consumed_port, ref in obj(a["consumed"]).items():
                if obj(a["consumed_roles"]).get(consumed_port) == "decision":
                    out.add(cast(str, ref))
        while stack:
            key = stack.pop()
            if key in seen:
                continue
            seen.add(key)
            p = obj(prov.get(key, {}))
            if not p:
                return ["!missing"]
            if p["decision"] is True:
                ref = cast(str, p["artifact"])
                artifact_value = obj(arts.get(ref, {}))
                if (
                    artifact_value.get("target") not in (t, "shared")
                    or artifact_value.get("invocation") != obj(d["execution"])["invocation"]
                    or p.get("target") != t
                ):
                    return ["!invalid"]
                out.add(ref)
            stack.extend(cast(str, x) for x in arr(p["parents"]))
        return sorted(out)

    if not execution_only:
        possible_assessment_owners = {
            (activation, entry["target"], production["node"], production["outcome"], production["promise"])
            for activation, entry in map(lambda item: (item[0], obj(item[1])), obj(s["entries"]).items())
            for production in map(obj, arr(d["productions"]))
            if entry.get("closed_unstarted") is False
            and entry.get("node_kind") == "operation"
            and entry.get("state_category") == "success"
            and entry.get("node") == production["node"]
            and entry.get("state_outcome") == production["outcome"]
        }
        required_assessment_owners = {
            owner
            for owner in possible_assessment_owners
            for terminal_value in [obj(obj(s["terminals"]).get(cast(str, owner[0]), {}))]
            if terminal_value.get("structural") is False
            and terminal_value.get("category") == "success"
            and terminal_value.get("outcome") == owner[3]
            and terminal_value.get("target") == owner[1]
        }
        retained_assessment_owners = [
            (fact["activation"], fact["target"], fact["node"], fact["outcome"], fact["promise"])
            for fact in map(obj, obj(s["assessment_facts"]).values())
        ]
        actual_owner_set = set(retained_assessment_owners)
        if len(retained_assessment_owners) != len(actual_owner_set):
            return reject("duplicate")
        if required_assessment_owners - actual_owner_set:
            return reject("missing")
        if actual_owner_set - possible_assessment_owners:
            return reject("unsupported")
    rows = []
    released = []
    union = set()
    for t in targets:
        f = obj(finals.get(t, {}))
        decs = decisions(t) if f and not execution_only else []
        if "!missing" in decs:
            return reject("missing")
        if "!invalid" in decs:
            return reject("foreign_owner")
        ok = not codes[t] and bool(f) and d["purpose"] == "protection"
        if ok:
            union.update(decs)
            released.append(
                {
                    "candidate": f["candidate"],
                    "evidence": sorted(cast(str, a["evidence_artifact"]) for a in supporting[t]),
                    "required_decisions": decs,
                    "target": t,
                }
            )
        q = (
            "not_assessed"
            if execution_only
            else "met"
            if ok
            else "unknown"
            if t in candidate_uncertain
            or codes[t] & {"assessment_unknown", "request_uncertain", "inconsistent_attribution"}
            else "unmet"
        )
        rows.append(
            {
                "artifact_available": bool(f)
                and cast(str, f.get("candidate")) in arts
                and (execution_only or t not in candidate_uncertain),
                "candidate": None if execution_only else f.get("candidate"),
                "completion": "pending" if "incomplete_membership" in codes[t] else "closed",
                "protection_available": ok,
                "qualification": q,
                "required_decisions": decs if ok else [],
                "target": t,
                "verified": sorted(cast(str, a["evidence_artifact"]) for a in supporting[t]),
                "withholding": sorted(
                    ({"request_accounting"} if "request_uncertain" in codes[t] else set())
                    | (codes[t] - {"request_uncertain"})
                    | ({"execution_only"} if execution_only else set())
                ),
            }
        )
    if len(union) > cast(int, lim["max_required_decisions"]):
        return reject("limit_exceeded")
    record = {
        "artifacts": sorted(arts),
        "evidence": sorted(cast(str, a["evidence_artifact"]) for a in valid),
        "execution": s["execution"],
        "memberships": s["memberships"],
        "targets": rows,
        "terminals": s["terminals"],
    }
    return {
        "qualified": released,
        "record": record,
        "required_decisions": sorted(union),
        "status": "accepted",
        "targets": rows,
        "verified": valid,
    }


def reduce(d: Obj, events: Sequence[Obj]) -> Obj:
    a = admit(d)
    if a["status"] != "accepted":
        return a
    s = initial(d)
    for e in events:
        bad = advance(s, e)
        if bad:
            return bad
    return qualify(d, s)


# fact constructors
def artifact(ref: str, target: str, role: str) -> Obj:
    return {
        "bytes": 1,
        "invocation": "I0",
        "key": ref.rsplit("v", 1)[0],
        "kind": "artifact",
        "ref": ref,
        "role": role,
        "target": target,
        "version": int(ref.rsplit("v", 1)[1]),
    }


def entry(
    a: str,
    t: str,
    node: str = "N",
    closed: bool = False,
    *,
    node_kind: str = "operation",
    occurrence: int | None = None,
    parent: str | None = None,
    state_category: str = "success",
    state_outcome: str | None = "ok",
) -> Obj:
    suffix = a.rstrip("0123456789")
    derived_occurrence = int(a[len(suffix) :]) if len(suffix) < len(a) else 0
    return {
        "activation": a,
        "closed_unstarted": closed,
        "kind": "entry",
        "node": node,
        "node_kind": node_kind,
        "occurrence": derived_occurrence if occurrence is None else occurrence,
        "parent": parent,
        "state_category": state_category,
        "state_outcome": state_outcome,
        "target": t,
    }


def terminal(
    a: str,
    t: str,
    category: str = "success",
    outcome: str | None = "ok",
    *,
    structural: bool = False,
    attempt: str | None = None,
    reasons: Sequence[str] = (),
) -> Obj:
    actual_attempt = None if structural or category in ("blocked", "inconsistent") else attempt or f"TASK:{a}"
    default_reason = {
        "blocked": "prerequisite",
        "cancelled": "cancel_requested",
        "failure": "execution_failed",
        "inconsistent": "contradictory",
        "lost": "transport_lost",
    }
    actual_reasons = list(reasons) if reasons else ([] if category == "success" else [default_reason[category]])
    return {
        "activation": a,
        "attempt": actual_attempt,
        "category": category,
        "kind": "terminal",
        "outcome": outcome,
        "reasons": actual_reasons,
        "structural": structural,
        "target": t,
    }


def port(a: str, t: str, name: str, ref: str, role: str, node: str = "N") -> Obj:
    return {"activation": a, "artifact": ref, "kind": "port", "node": node, "port": name, "role": role, "target": t}


def input_producer(a: str, t: str, node: str, port_name: str, producer: str) -> Obj:
    return {
        "activation": a,
        "kind": "input_producer",
        "node": node,
        "port": port_name,
        "producer": producer,
        "target": t,
    }


def binding_ref(declaration: str, key: int = 0, version: int = 1) -> Obj:
    return {"declaration": declaration, "key": key, "version": version}


def binding_declaration(
    *sites: tuple[str, str, str, str],
    collections: Sequence[str] = (),
    targets: Sequence[str] = ("A",),
) -> Obj:
    d = declaration(targets=targets)
    d["binding_inputs"] = [
        {
            "declaration": identity,
            "materialization": "collection" if identity in collections else "single",
            "node": node,
            "port": port_name,
            "source": f"SRC:{identity}@1",
            "target": target,
            "version_selection": "exact_one",
        }
        for identity, node, port_name, target in sites
    ]
    d["initial_collections"] = [
        {field: obj(site)[field] for field in ("declaration", "node", "port", "target")}
        for site in arr(d["binding_inputs"])
        if obj(site).get("declaration") in collections
    ]
    if sites:
        set_output_dependency(
            d,
            "N",
            "result",
            (*tuple(f"bound_{i}" for i, _ in enumerate(sites)), "subject"),
            identity_input="subject",
        )
    return d


def binding_receipt(
    sites: Sequence[tuple[str, str, str, str]],
    artifacts: Sequence[tuple[str, int, int]],
) -> Obj:
    by_identity = {identity: (node, port_name, target) for identity, node, port_name, target in sites}
    return {
        "artifacts": [
            {
                "declaration": identity,
                "key": key,
                "node": by_identity[identity][0],
                "port": by_identity[identity][1],
                "source": f"SRC:{identity}@1",
                "target": by_identity[identity][2],
                "version": version,
            }
            for identity, key, version in artifacts
        ],
        "binding": "BIND0",
        "kind": "binding_receipt",
        "sources": [
            {
                "declaration": identity,
                "node": node,
                "port": port_name,
                "source": f"SRC:{identity}@1",
                "target": target,
                "terminal": "bound",
            }
            for identity, node, port_name, target in sites
        ],
        "terminal": "success",
    }


def provenance(
    key: str,
    ref: str,
    t: str,
    parents: Sequence[str] = (),
    decision: bool = False,
    *,
    source: str = "operation_output",
    node: str | None = "N",
    port_name: str | None = "result",
    root_target: str | None = None,
    activation: str | None = None,
    binding_artifact: Obj | str | None = None,
    declaration: str | None = None,
    expander: str | None = None,
    item_key: int | None = None,
    item_version: int | None = None,
    member: str | None = None,
) -> Obj:
    return {
        "artifact": ref,
        "activation": activation,
        "binding_artifact": binding_artifact,
        "declaration": declaration,
        "decision": decision,
        "expander": expander,
        "item_key": item_key,
        "item_version": item_version,
        "key": key,
        "kind": "provenance",
        "node": node,
        "member": member,
        "parents": list(parents),
        "port": port_name,
        "root_target": root_target,
        "source": source,
        "target": t,
    }


def assessment(t: str, **changes: Json) -> Obj:
    x: Obj = {
        "activation": f"ROOT:{t}",
        "authenticated_factory": "EXEC:I0",
        "consumed": {"context": f"X{t}v0"},
        "coverage": ["K0", "K1"],
        "environment": {"absences": {"Q0": 1}, "configurations": {"N": "c0"}, "state": {"read": 1}},
        "evidence_artifact": f"E{t}v0",
        "evidence_port": "evidence",
        "finding": "satisfied",
        "kind": "assessment",
        "node": "N",
        "outcome": "ok",
        "promise": "P",
        "subject_artifact": f"{t}v0",
        "subject_port": "subject",
        "target": t,
    }
    x.update(changes)
    x["fact"] = x.get("fact", f"F:{t}:{x['promise']}")
    return x


def assessment_submission(t: str, promise: str = "P") -> Obj:
    return {"fact": f"F:{t}:{promise}", "kind": "assessment_submission"}


def final(t: str, **changes: Json) -> Obj:
    x: Obj = {
        "activation": f"ROOT:{t}",
        "candidate": f"{t}v0",
        "kind": "final",
        "node": "N",
        "outcome": "ok",
        "port": "result",
        "producer": f"OUT:{t}",
        "target": t,
    }
    x.update(changes)
    return x


def revisions(targets: Sequence[str], **changes: int) -> list[Obj]:
    out: list[Obj] = []
    for t in targets:
        for key, value in ((t, 0), (f"E{t}", 0), (f"X{t}", 0)):
            out.append({"collection": "artifacts", "key": key, "kind": "revision", "value": changes.get(key, value)})
    out += [
        {"collection": "absences", "key": "Q0", "kind": "revision", "value": changes.get("Q0", 1)},
        {"collection": "configurations", "key": "N", "kind": "revision", "value": changes.get("N", "c0")},
        {"collection": "state", "key": "read", "kind": "revision", "value": changes.get("read", 1)},
        {"kind": "seal_revision"},
    ]
    return out


def base_events(targets: Sequence[str], *, assess: bool = True) -> list[Obj]:
    e = []
    for t in targets:
        root = f"ROOT:{t}"
        e.extend(
            [
                artifact(f"{t}v0", t, "candidate"),
                artifact(f"E{t}v0", t, "evidence"),
                artifact(f"X{t}v0", t, "artifact"),
                entry(root, t),
                terminal(root, t),
                port(root, t, "subject", f"{t}v0", "candidate"),
                port(root, t, "evidence", f"E{t}v0", "evidence"),
                port(root, t, "context", f"X{t}v0", "artifact"),
                port(root, t, "result", f"{t}v0", "candidate"),
                provenance(
                    f"ROOT:{t}:subject",
                    f"{t}v0",
                    t,
                    source="root_input",
                    port_name="subject",
                    activation=root,
                ),
                provenance(
                    f"ROOT:{t}:context",
                    f"X{t}v0",
                    t,
                    source="root_input",
                    port_name="context",
                    activation=root,
                ),
                input_producer(root, t, "N", "subject", f"ROOT:{t}:subject"),
                input_producer(root, t, "N", "context", f"ROOT:{t}:context"),
                provenance(
                    f"OUT:{t}",
                    f"{t}v0",
                    t,
                    (f"ROOT:{t}:subject", f"ROOT:{t}:context"),
                    activation=root,
                ),
                provenance(
                    f"EVID:{t}",
                    f"E{t}v0",
                    t,
                    (f"ROOT:{t}:context",),
                    port_name="evidence",
                    activation=root,
                ),
                final(t),
            ]
        )
        if assess:
            e.append(assessment(t))
            e.append(assessment_submission(t))
    root_membership: Obj = {
        "closed": True,
        "expansion_outcome": None,
        "kind": "membership",
        "members": [f"ROOT:{t}" for t in targets],
        "parent": None,
        "status": "closed",
        "target": None,
    }
    e.append(root_membership)
    e.extend(revisions(targets))
    return e


VERSIONED_CASES = {
    "validity/candidate_stale",
    "validity/candidate_stale_no_assessment",
    "validity/evidence_stale",
    "validity/consumed_stale",
    "validity/stale_precedes_unknown",
    "validity/evidence_output_replaced",
    "selective/a_only",
    "selective/b_only",
    "selective/a_b",
    "selective/a_only_stale",
    "selective/b_only_stale",
    "selective/a_b_stale",
    "selective/decision_stale",
    "provenance/version_edge",
    "lineage/initial_two_versions",
    "lineage/two_selected_current_versions",
    "lineage/invented_version",
    "lineage/declaration_crossover",
    "lineage/target_crossover",
    "lineage/invocation_crossover",
    "lineage/artifacts_exact",
    "lineage/artifacts_one_over",
    "lineage/bytes_exact",
    "lineage/bytes_one_over",
    "lineage/provenance_exact",
    "lineage/provenance_one_over",
    "lineage/rejected_publication_rollback",
    "lineage/latest_older_selected",
    "lineage/latest_missing_binding_request",
    "lineage/latest_foreign_binding_association",
    "lineage/latest_missing_binding_settlement",
    "lineage/latest_missing_binding_cleanup",
    "lineage/latest_foreign_cleanup_target",
    "validity/final_candidate_sibling_substitution",
}


def _replace_refs(value: Json, replacements: Mapping[str, str]) -> Json:
    if isinstance(value, str):
        return replacements.get(value, value)
    if isinstance(value, list):
        return [_replace_refs(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace_refs(item, replacements) for key, item in value.items()}
    return value


def _materialize_initial_versions(case_id: str, d: Obj, e: list[Obj]) -> None:
    artifacts = [item for item in e if item.get("kind") == "artifact"]
    by_key: dict[str, list[Obj]] = {}
    for value in artifacts:
        by_key.setdefault(cast(str, value["key"]), []).append(value)
    revisions_by_key = {
        cast(str, value["key"]): value
        for value in e
        if value.get("kind") == "revision" and value.get("collection") == "artifacts"
    }
    if case_id == "provenance/version_edge":
        old = next(value for value in artifacts if value["ref"] == "XAv0")
        e.insert(e.index(old) + 1, artifact("XAv1", "A", "artifact"))
        revisions_by_key["XA"]["value"] = 1
        by_key["XA"] = [old, next(value for value in e if value.get("ref") == "XAv1")]
    for key, revision in revisions_by_key.items():
        values = by_key.get(key, [])
        if values and revision["value"] not in {value["version"] for value in values} and case_id in VERSIONED_CASES:
            template = values[0]
            ref = f"{key}v{revision['value']}"
            created = artifact(ref, cast(str, template["target"]), "artifact")
            e.insert(e.index(template) + 1, created)
            values.append(created)
            by_key[key] = values
    replacements: dict[str, str] = {}
    shifts: dict[str, int] = {}
    for key, values in by_key.items():
        if len(values) > 1 and min(cast(int, value["version"]) for value in values) == 0:
            shifts[key] = 1
            for value in values:
                replacements[cast(str, value["ref"])] = f"{key}v{cast(int, value['version']) + 1}"
    if replacements:
        updated = cast(list[Obj], _replace_refs(e, replacements))
        e[:] = updated
        for value in e:
            if value.get("kind") == "artifact" and value.get("key") in shifts:
                value["version"] = cast(int, value["version"]) + shifts[cast(str, value["key"])]
            if (
                value.get("kind") == "revision"
                and value.get("collection") == "artifacts"
                and value.get("key") in shifts
            ):
                value["value"] = cast(int, value["value"]) + shifts[cast(str, value["key"])]
    artifacts = [item for item in e if item.get("kind") == "artifact"]
    by_key = {}
    for value in artifacts:
        by_key.setdefault(cast(str, value["key"]), []).append(value)
    multi = {key: values for key, values in by_key.items() if len(values) > 1}
    if not multi:
        return
    existing_materialized = {
        cast(str, value["artifact"])
        for value in e
        if value.get("kind") == "provenance" and value.get("source") in ("bound_input", "map_item")
    }
    pending = {
        key: values
        for key, values in multi.items()
        if any(cast(str, value["ref"]) not in existing_materialized for value in values)
    }
    if not pending:
        return
    sites: list[tuple[str, str, str, str]] = []
    receipt_artifacts: list[tuple[str, int, int]] = []
    lineage_facts: list[Obj] = []
    lifecycle: list[Obj] = []
    selections: dict[str, tuple[str, str, str]] = {}
    for index, (key, values) in enumerate(sorted(pending.items())):
        declaration_id = f"DVER{index}"
        target = cast(str, values[0]["target"])
        refs = {cast(str, value["ref"]) for value in values}
        root_source = next(
            (
                value
                for value in e
                if value.get("kind") == "provenance"
                and value.get("source") == "root_input"
                and value.get("artifact") in refs
            ),
            None,
        )
        port_name = "evidence_version" if root_source is None and key == "EA" else cast(str, obj(root_source)["port"])
        node = "N"
        site = (declaration_id, node, port_name, target)
        sites.append(site)
        association = {
            "declaration": declaration_id,
            "node": node,
            "port": port_name,
            "source": f"SRC:{declaration_id}@1",
            "target": target,
        }
        request = f"BINDREQ:{declaration_id}"
        resource = f"BINDRES:{declaration_id}"
        lifecycle.extend(
            [
                {
                    "association": association,
                    "kind": "binding_reserve",
                    "policy": "P0",
                    "purpose": "initial_binding",
                    "request": request,
                    "resource": resource,
                },
                {"kind": "binding_dispatch", "request": request},
                {"kind": "binding_result", "outcome": "retrieved", "request": request},
                {
                    "kind": "binding_settlement",
                    "remote_stopped": True,
                    "request": request,
                    "usage": "known",
                },
            ]
        )
        arr(d["binding_inputs"]).append(
            {
                "declaration": declaration_id,
                "materialization": "single",
                "node": node,
                "port": port_name,
                "source": f"SRC:{declaration_id}@1",
                "target": target,
                "version_selection": "latest",
            }
        )
        for value in sorted(values, key=lambda item: cast(int, item["version"])):
            version = cast(int, value["version"])
            receipt_artifacts.append((declaration_id, 0, version))
            producer = f"BOUND:{key}:{version}"
            lineage_facts.append(
                provenance(
                    producer,
                    cast(str, value["ref"]),
                    target,
                    source="bound_input",
                    node=node,
                    port_name=port_name,
                    binding_artifact=binding_ref(declaration_id, 0, version),
                )
            )
        selected = max(values, key=lambda item: cast(int, item["version"]))
        selections[key] = (cast(str, selected["ref"]), f"BOUND:{key}:{selected['version']}", port_name)
    ownership: list[Obj] = []
    for declaration_id, _, _, target in sites:
        resource = f"BINDRES:{declaration_id}"
        ownership.extend(
            [
                {
                    "kind": "binding_cleanup_association",
                    "owner": "sdk",
                    "purpose": "accounting",
                    "resource": resource,
                    "targets": [target],
                },
                {"disposition": "closed", "kind": "binding_cleanup", "resource": resource},
            ]
        )
    e[0:0] = [*lifecycle, binding_receipt(sites, receipt_artifacts), *ownership]
    first_provenance = next(i for i, value in enumerate(e) if value.get("kind") == "provenance")
    e[first_provenance:first_provenance] = lineage_facts

    # Initial ``latest`` retains every version but supplies only the numeric
    # maximum to the scalar destination.  The input producer is its exact
    # BoundInputKey; no collection or synthetic scalar extraction exists.
    for key, values in sorted(multi.items()):
        selected_ref, bound_key, port_name = selections[key]
        refs = {cast(str, value["ref"]) for value in values}
        root_sources = [
            value
            for value in e
            if value.get("kind") == "provenance"
            and value.get("source") == "root_input"
            and value.get("artifact") in refs
        ]
        for source in root_sources:
            source_key = cast(str, source["key"])
            for value in e:
                if value.get("kind") == "input_producer" and value.get("producer") == source_key:
                    value["producer"] = bound_key
                if value.get("kind") == "provenance":
                    value["parents"] = [
                        bound_key if parent == source_key else parent for parent in arr(value["parents"])
                    ]
            e.remove(source)
        for value in e:
            if value.get("kind") == "port" and value.get("artifact") in refs:
                value["artifact"] = selected_ref
            if value.get("kind") == "assessment":
                value["consumed"] = {
                    consumed_port: selected_ref if consumed_ref in refs else consumed_ref
                    for consumed_port, consumed_ref in obj(value["consumed"]).items()
                }
                if value.get("evidence_artifact") in refs:
                    value["evidence_artifact"] = selected_ref
                if value.get("subject_artifact") in refs:
                    value["subject_artifact"] = selected_ref
            if value.get("kind") == "final" and value.get("candidate") in refs:
                value["candidate"] = selected_ref
            if (
                value.get("kind") == "provenance"
                and value.get("artifact") in refs
                and value.get("source") == "operation_output"
            ):
                value["artifact"] = selected_ref
            if value.get("kind") == "provenance" and value.get("key") == "EVID:A" and value.get("artifact") in refs:
                value["artifact"] = selected_ref

    if case_id in {"validity/evidence_stale", "validity/evidence_output_replaced"}:
        evidence_values = sorted(multi["EA"], key=lambda item: cast(int, item["version"]))
        selected = evidence_values[-1]
        selected_ref = cast(str, selected["ref"])
        bound_key = f"BOUND:EA:{selected['version']}"
        assessment_fact = next(value for value in e if value.get("kind") == "assessment")
        assessment_fact["consumed"] = {"evidence_version": selected_ref}
        production = obj(arr(d["productions"])[0])
        production["consumed_ports"] = ["evidence_version"]
        obj(arr(d["requirements"])[0])["consumed_ports"] = ["evidence_version"]
        evidence_output = next(
            value for value in e if value.get("kind") == "provenance" and value.get("key") == "EVID:A"
        )
        next(value for value in e if value.get("kind") == "port" and value.get("port") == "evidence")["artifact"] = (
            selected_ref
        )
        evidence_output["artifact"] = selected_ref
        insert_at = e.index(evidence_output)
        e[insert_at:insert_at] = [
            port("ROOT:A", "A", "evidence_version", selected_ref, "evidence"),
            input_producer("ROOT:A", "A", "N", "evidence_version", bound_key),
        ]
        evidence_output["parents"] = [bound_key]
        set_output_dependency(d, "N", "evidence", ("evidence_version",), identity_input="evidence_version")

    if case_id == "provenance/version_edge":
        selected_ref, _, _ = selections["XA"]
        next(value for value in e if value.get("kind") == "port" and value.get("port") == "version")["artifact"] = (
            selected_ref
        )
        next(value for value in e if value.get("kind") == "provenance" and value.get("key") == "VERSION")[
            "artifact"
        ] = selected_ref
        next(
            value
            for value in e
            if value.get("kind") == "revision" and value.get("collection") == "artifacts" and value.get("key") == "XA"
        )["value"] = max(cast(int, value["version"]) for value in multi["XA"])

    stale_keys = {
        "validity/candidate_stale": "A",
        "validity/candidate_stale_no_assessment": "A",
        "validity/evidence_stale": "EA",
        "validity/evidence_output_replaced": "EA",
        "validity/consumed_stale": "XA",
        "validity/stale_precedes_unknown": "XA",
        "selective/b_only": "YA",
        "selective/a_b": "YA",
        "selective/a_only_stale": "XA",
        "selective/b_only_stale": "YA",
        "selective/a_b_stale": "YA",
        "selective/decision_stale": "YA",
    }
    if case_id in stale_keys:
        stale_key = stale_keys[case_id]
        next(
            value
            for value in e
            if value.get("kind") == "revision"
            and value.get("collection") == "artifacts"
            and value.get("key") == stale_key
        )["value"] = min(cast(int, value["version"]) for value in multi[stale_key])

    if case_id == "lineage/two_selected_current_versions":
        seal = next(i for i, value in enumerate(e) if value.get("kind") == "seal_revision")
        current = next(value for value in e if value.get("kind") == "revision" and value.get("key") == "XA")
        e.insert(seal, {**current, "value": 2})
    elif case_id == "lineage/invented_version":
        next(value for value in e if value.get("kind") == "revision" and value.get("key") == "XA")["value"] = 3
    elif case_id == "lineage/declaration_crossover":
        second = next(value for value in e if value.get("kind") == "provenance" and value.get("key") == "BOUND:XA:2")
        obj(second["binding_artifact"])["declaration"] = "OTHER"
    elif case_id == "lineage/target_crossover":
        next(value for value in e if value.get("kind") == "artifact" and value.get("ref") == "XAv2")["target"] = "B"
    elif case_id == "lineage/invocation_crossover":
        next(value for value in e if value.get("kind") == "artifact" and value.get("ref") == "XAv2")["invocation"] = (
            "I1"
        )
    if case_id == "lineage/latest_older_selected":
        values = sorted(multi["XA"], key=lambda item: cast(int, item["version"]))
        older = cast(str, values[0]["ref"])
        older_key = f"BOUND:XA:{values[0]['version']}"
        selected = cast(str, values[-1]["ref"])
        for value in e:
            if value.get("kind") == "port" and value.get("port") == "context" and value.get("artifact") == selected:
                value["artifact"] = older
            if value.get("kind") == "input_producer" and value.get("port") == "context":
                value["producer"] = older_key
            if value.get("kind") == "provenance":
                value["parents"] = [
                    older_key if parent == f"BOUND:XA:{values[-1]['version']}" else parent
                    for parent in arr(value["parents"])
                ]
            if value.get("kind") == "assessment":
                value["consumed"] = {
                    port_name: older if artifact_ref == selected else artifact_ref
                    for port_name, artifact_ref in obj(value["consumed"]).items()
                }
    elif case_id == "lineage/latest_missing_binding_request":
        e[:] = [value for value in e if value.get("kind") != "binding_reserve"]
    elif case_id == "lineage/latest_foreign_binding_association":
        reserve = next(value for value in e if value.get("kind") == "binding_reserve")
        obj(reserve["association"])["declaration"] = "OTHER"
    elif case_id == "lineage/latest_missing_binding_settlement":
        e[:] = [value for value in e if value.get("kind") != "binding_settlement"]
    elif case_id == "lineage/latest_missing_binding_cleanup":
        e[:] = [value for value in e if value.get("kind") != "binding_cleanup"]
    elif case_id == "lineage/latest_foreign_cleanup_target":
        association = next(value for value in e if value.get("kind") == "binding_cleanup_association")
        association["targets"] = ["B"]
    elif case_id == "validity/final_candidate_sibling_substitution":
        values = sorted(multi["A"], key=lambda item: cast(int, item["version"]))
        next(value for value in e if value.get("kind") == "final")["candidate"] = values[0]["ref"]


def case(
    family: str, name: str, d: Obj, e: list[Obj], boundary: str = "qualification", alternates: Sequence[list[Obj]] = ()
) -> Obj:
    case_id = f"{family}/{name}"
    if case_id in VERSIONED_CASES:
        _materialize_initial_versions(case_id, d, e)
    c: Obj = {
        "boundary": boundary,
        "case_id": case_id,
        "declaration": d,
        "events": e,
        "expected": admit(d) if boundary == "admission" else reduce(d, e),
        "family": family,
        "traces": [],
    }
    neutral_only = {
        "assessment/consumed_port",
        "assessment/duplicate_coverage",
        "assessment/extra_coverage",
        "assessment/foreign_consumed",
        "assessment/foreign_subject",
        "assessment/foreign_target",
        "assessment/incomplete_coverage",
        "assessment/wrong_kind_coverage",
        "joins/evidence_port_swap",
        "joins/subject_port_swap",
        "map_item_evidence/wrong_subject_artifact",
    }
    c["comparison_scope"] = "neutral_only" if case_id in neutral_only else "production_boundary"
    c["traces"] = [{"events": x, "expected": reduce(d, x), "name": f"alternate_{i}"} for i, x in enumerate(alternates)]
    return c


def mutate(events: list[Obj], kind: str, field: str, value: Json, index: int = 0) -> list[Obj]:
    x = deepcopy(events)
    matches = [e for e in x if e.get("kind") == kind]
    matches[index][field] = value
    return x


def generate_cases() -> tuple[Obj, ...]:
    c = []
    base = base_events(("A",))

    def execution_only_shape() -> tuple[Obj, list[Obj]]:
        declaration_value = declaration(purpose="execution_only")
        declaration_value["productions"] = []
        declaration_value["requirements"] = []
        events = [
            value
            for value in base_events(("A",), assess=False)
            if not (value.get("kind") == "artifact" and value.get("role") == "evidence")
            and not (value.get("kind") == "port" and value.get("role") == "evidence")
            and not (value.get("kind") == "provenance" and value.get("key") == "EVID:A")
            and not (value.get("kind") == "revision" and value.get("key") == "EA")
            and not (value.get("kind") == "revision" and value.get("collection") != "artifacts")
        ]
        return declaration_value, events

    def wire_result(d: Obj, e: list[Obj], target: str, producer: str, input_name: str) -> None:
        output = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == f"OUT:{target}")
        producer_fact = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == producer)
        e.insert(
            e.index(output),
            port(f"ROOT:{target}", target, input_name, cast(str, producer_fact["artifact"]), "artifact"),
        )
        e.insert(e.index(output), input_producer(f"ROOT:{target}", target, "N", input_name, producer))
        output["parents"] = [producer]
        set_output_dependency(d, "N", "result", (input_name,))

    def retain_root_input(e: list[Obj], target: str, port_name: str, ref: str, *, node: str = "N") -> None:
        output = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == f"OUT:{target}")
        key = f"ROOT:{target}:{port_name}"
        e.insert(
            e.index(output),
            provenance(
                key, ref, target, source="root_input", node=node, port_name=port_name, activation=f"ROOT:{target}"
            ),
        )
        e.insert(e.index(output), input_producer(f"ROOT:{target}", target, node, port_name, key))

    def wire_evidence_consumed(d: Obj, e: list[Obj], ports: Sequence[str]) -> None:
        set_production_consumed(d, ports)
        set_output_dependency(d, "N", "evidence", ports)
        evidence = next(value for value in e if value.get("kind") == "provenance" and value.get("key") == "EVID:A")
        producers = {
            cast(str, value["port"]): cast(str, value["producer"])
            for value in e
            if value.get("kind") == "input_producer"
            and value.get("activation") == "ROOT:A"
            and value.get("node") == "N"
        }
        evidence["parents"] = [producers[port_name] for port_name in ports]

    c.append(case("release", "protection_success", declaration(), base))
    execution_declaration, execution_events = execution_only_shape()
    c.append(case("release", "execution_only", execution_declaration, execution_events))
    # exact joins and reviewer probes
    for name, kind, field, value in (
        ("missing_candidate_artifact", "artifact", "ref", "DROP"),
        ("missing_evidence_artifact", "artifact", "ref", "DROP"),
        ("producer_artifact_mismatch", "provenance", "artifact", "XAv0"),
        ("producer_target_mismatch", "provenance", "target", "B"),
        ("evidence_port_swap", "assessment", "evidence_port", "wrong"),
        ("subject_port_swap", "assessment", "subject_port", "wrong"),
        ("invocation_mismatch", "artifact", "invocation", "I1"),
        ("node_mismatch", "assessment", "node", "OTHER"),
        ("outcome_mismatch", "assessment", "outcome", "other"),
        ("promise_mismatch", "assessment", "promise", "other"),
    ):
        e = deepcopy(base)
        if name == "missing_candidate_artifact":
            e = [x for x in e if not (x.get("kind") == "artifact" and x.get("ref") == "Av0")]
        elif name == "missing_evidence_artifact":
            e = [x for x in e if not (x.get("kind") == "artifact" and x.get("ref") == "EAv0")]
        else:
            target = next(x for x in e if x.get("kind") == kind and (kind != "provenance" or x.get("key") == "OUT:A"))
            target[field] = value
        c.append(case("joins", name, declaration(), e))
    # validity dependency products
    for label, field, key in (
        ("candidate", "artifacts", "A"),
        ("evidence", "artifacts", "EA"),
        ("consumed", "artifacts", "XA"),
        ("absence", "absences", "Q0"),
        ("configuration", "configurations", "N"),
        ("state", "state", "read"),
    ):
        for mode in ("current", "stale", "unknown"):
            e = deepcopy(base)
            rev = [
                x for x in e if x.get("kind") == "revision" and x.get("collection") == field and x.get("key") == key
            ][0]
            if mode == "stale":
                rev["value"] = 2 if field != "configurations" else "c1"
                if field == "artifacts":
                    ref = {"A": "Av2", "EA": "EAv2", "XA": "XAv2"}[key]
                    e.insert(0, artifact(ref, "A", "artifact"))
            elif mode == "unknown":
                e.remove(rev)
            c.append(case("validity", f"{label}_{mode}", declaration(), e))
    e = deepcopy(base)
    next(x for x in e if x.get("kind") == "revision" and x.get("key") == "A")["value"] = 1
    e.insert(0, artifact("Av1", "A", "artifact"))
    c.append(case("validity", "final_candidate_sibling_substitution", declaration(), e))
    e = deepcopy(base)
    e = [value for value in e if value.get("kind") != "assessment_submission"]
    next(value for value in e if value.get("kind") == "revision" and value.get("key") == "A")["value"] = 2
    e.insert(0, artifact("Av2", "A", "artifact"))
    c.append(case("validity", "candidate_stale_no_assessment", declaration(), e))
    e = deepcopy(base)
    e = [x for x in e if not (x.get("kind") == "revision" and x.get("key") == "Q0")]
    next(x for x in e if x.get("kind") == "revision" and x.get("key") == "XA")["value"] = 1
    e.insert(0, artifact("XAv1", "A", "artifact"))
    c.append(case("validity", "stale_precedes_unknown", declaration(), e))
    # assessments
    for name, field, value in (
        ("unsatisfied", "finding", "unsatisfied"),
        ("unknown", "finding", "unknown"),
        ("incomplete_coverage", "coverage", []),
        ("extra_coverage", "coverage", ["K0", "K1", "K2"]),
        ("duplicate_coverage", "coverage", ["K0", "K0"]),
        ("caller_copy", "authenticated_factory", "CALLER"),
        ("consumed_port", "consumed", {"wrong": "XAv0"}),
        ("foreign_consumed", "consumed", {"context": "XBv0"}),
    ):
        c.append(case("assessment", name, declaration(), mutate(base, "assessment", field, value)))
    c.append(
        case(
            "assessment",
            "missing",
            declaration(),
            [x for x in base if x.get("kind") != "assessment_submission"],
        )
    )
    e = deepcopy(base)
    e.insert(
        e.index(next(x for x in e if x.get("kind") == "revision")),
        deepcopy(next(x for x in e if x.get("kind") == "assessment_submission")),
    )
    c.append(case("assessment", "duplicate", declaration(), e))
    c.append(case("assessment", "unsupported_finding", declaration(), mutate(base, "assessment", "finding", "other")))
    c.append(
        case(
            "assessment",
            "absence_query",
            declaration(),
            mutate(
                base,
                "assessment",
                "environment",
                {"absences": {"Q1": 1}, "configurations": {"N": "c0"}, "state": {"read": 1}},
            ),
        )
    )

    def multi_promise_declaration() -> Obj:
        d = declaration()
        partial = obj(arr(d["productions"])[0])
        partial["promise"] = "P_PART"
        partial["coverage"] = ["K0"]
        full = deepcopy(partial)
        full.update({"coverage": ["K0", "K1"], "evidence_port": "evidence2", "promise": "P_FULL"})
        arr(d["productions"]).append(full)
        obj(arr(d["requirements"])[0])["coverage"] = ["K0", "K1"]
        set_output_dependency(d, "N", "evidence2", ("context",))
        return d

    def multi_promise_events(*, partial: bool, complete: bool) -> list[Obj]:
        e = deepcopy(base)
        partial_assessment = next(x for x in e if x.get("kind") == "assessment")
        partial_assessment.update({"coverage": ["K0"], "promise": "P_PART"})
        partial_assessment["fact"] = "F:A:P_PART"
        partial_submission = next(x for x in e if x.get("kind") == "assessment_submission")
        partial_submission["fact"] = "F:A:P_PART"
        if not partial:
            e.remove(partial_submission)
        out = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "OUT:A")
        idx = e.index(out)
        complete_submission = assessment_submission("A", "P_FULL")
        e[idx:idx] = [
            artifact("E2v0", "A", "evidence"),
            port("ROOT:A", "A", "evidence2", "E2v0", "evidence"),
            provenance(
                "EVID2:A",
                "E2v0",
                "A",
                ("ROOT:A:context",),
                port_name="evidence2",
                activation="ROOT:A",
            ),
            assessment(
                "A",
                coverage=["K0", "K1"],
                evidence_artifact="E2v0",
                evidence_port="evidence2",
                promise="P_FULL",
            ),
            complete_submission,
        ]
        if not complete:
            e.remove(complete_submission)
        seal = next(i for i, x in enumerate(e) if x.get("kind") == "seal_revision")
        e.insert(seal, {"collection": "artifacts", "key": "E2", "kind": "revision", "value": 0})
        return e

    for name, partial, complete in (
        ("partial_promise_only", True, False),
        ("complete_promise_only", False, True),
        ("partial_and_complete_promises", True, True),
    ):
        c.append(
            case(
                "assessment",
                name,
                multi_promise_declaration(),
                multi_promise_events(partial=partial, complete=complete),
            )
        )
    c.append(
        case(
            "assessment",
            "caller_replacement",
            declaration(),
            mutate(base, "assessment", "authenticated_factory", "CALLER"),
        )
    )

    # membership 0/1/2, nested, choice and defects
    def map_events(
        n: int, *, parent_terminal: bool = True, nested: bool = False, operation_expander: bool = False
    ) -> list[Obj]:
        e = deepcopy(base)
        cut = next(i for i, x in enumerate(e) if x.get("kind") == "revision")
        root_membership = next(x for x in e if x.get("kind") == "membership" and x.get("parent") is None)
        arr(root_membership["members"]).append("MAP")
        expander_kind = "operation" if operation_expander else "container"
        facts: list[Obj] = [entry("MAP", "A", "EXP", node_kind=expander_kind)]
        if parent_terminal:
            facts.append(terminal("MAP", "A", structural=not operation_expander))
        members = []
        for i in range(n):
            m = f"M{i}"
            members.append(m)
            facts += [
                {"activation": m, "kind": "reservation", "parent": "MAP", "selected": True, "target": "A"},
                entry(m, "A", "MEM", parent="MAP"),
                terminal(m, "A"),
            ]
        facts.append(
            {
                "closed": True,
                "expansion_outcome": "ok",
                "kind": "membership",
                "members": members,
                "parent": "MAP",
                "status": "closed",
                "target": "A",
            }
        )
        if nested:
            members.append("INNER")
            facts += [
                {"activation": "INNER", "kind": "reservation", "parent": "MAP", "selected": True, "target": "A"},
                entry("INNER", "A", "EXP", node_kind="container", parent="MAP"),
                terminal("INNER", "A", structural=True),
                {"activation": "IM0", "kind": "reservation", "parent": "INNER", "selected": True, "target": "A"},
                entry("IM0", "A", "MEM", parent="INNER"),
                terminal("IM0", "A"),
                {
                    "closed": True,
                    "expansion_outcome": "ok",
                    "kind": "membership",
                    "members": ["IM0"],
                    "parent": "INNER",
                    "status": "closed",
                    "target": "A",
                },
            ]
        e[cut:cut] = facts
        return e

    for n in (0, 1, 2):
        c.append(case("membership", f"closed_{n}", declaration(), map_events(n)))
    c.append(case("membership", "nested_closed", declaration(), map_events(1, nested=True)))
    c.append(case("membership", "missing_expander_terminal", declaration(), map_events(0, parent_terminal=False)))
    e = map_events(1)
    e = [x for x in e if not (x.get("kind") == "terminal" and x.get("activation") == "M0")]
    c.append(case("membership", "missing_member_terminal", declaration(), e))
    e = map_events(1)
    idx = next(i for i, x in enumerate(e) if x.get("kind") == "provenance" and x.get("key") == "OUT:A")
    e.insert(idx, dict(terminal("M0", "A"), category="success"))
    c.append(case("membership", "duplicate_terminal", declaration(), e))
    e = deepcopy(base)
    idx = next(i for i, x in enumerate(e) if x.get("kind") == "revision")
    arr(next(x for x in e if x.get("kind") == "membership" and x.get("parent") is None)["members"]).append("CHOICE")
    e[idx:idx] = [
        {"activation": "CHOICE", "kind": "reservation", "parent": None, "selected": True, "target": "A"},
        entry("CHOICE", "A", closed=True, state_category="blocked", state_outcome=None),
        terminal("CHOICE", "A", "blocked", None),
    ]
    c.append(case("membership", "selected_closed_unstarted", declaration(), e))
    e = deepcopy(base)
    idx = next(i for i, x in enumerate(e) if x.get("kind") == "revision")
    e.insert(idx, {"activation": "OTHER", "kind": "reservation", "parent": None, "selected": False, "target": "A"})
    c.append(case("membership", "unselected_choice", declaration(), e))
    e = map_events(1)
    m = next(x for x in e if x.get("kind") == "membership" and x.get("parent") == "MAP")
    arr(m["members"]).append("M0")
    c.append(case("membership", "duplicate_member", declaration(), e))

    # request histories
    def request_history(failure: str | None, purpose: str | None, unknown: bool = False) -> list[Obj]:
        e = deepcopy(base)
        idx = next(i for i, x in enumerate(e) if x.get("kind") == "revision")
        facts: list[Obj] = [
            {
                "associations": ["A"],
                "kind": "request_attempt",
                "policy": "P",
                "predecessor": None,
                "purpose": "initial",
                "request": "R0",
            },
            {
                "condition": "failure" if failure else "result",
                "failure": failure,
                "kind": "request_terminal",
                "request": "R0",
            },
            {
                "kind": "settlement",
                "remote_stopped": not unknown,
                "request": "R0",
                "usage": "unknown" if unknown else 1,
            },
        ]
        if purpose:
            facts += [
                {
                    "associations": ["A"],
                    "kind": "request_attempt",
                    "policy": "P",
                    "predecessor": "R0",
                    "purpose": purpose,
                    "request": "R1",
                },
                {"condition": "result", "failure": None, "kind": "request_terminal", "request": "R1"},
                {"kind": "settlement", "remote_stopped": True, "request": "R1", "usage": 1},
            ]
        e[idx:idx] = facts
        return e

    for purpose, failure in (("retry", "retryable"), ("correction", "malformed"), ("failover", "permanent")):
        c.append(case("request", f"{purpose}_recovered", declaration(), request_history(failure, purpose)))
    c.append(case("request", "lost_unknown", declaration(), request_history("lost", None, True)))
    c.append(case("request", "final_failure", declaration(), request_history("permanent", None)))
    for purpose in ("initial_binding", "adaptive_retrieval", "repair"):
        e = deepcopy(base)
        idx = next(i for i, x in enumerate(e) if x.get("kind") == "revision")
        e[idx:idx] = [
            {
                "associations": ["A"],
                "kind": "request_attempt",
                "policy": "P",
                "predecessor": None,
                "purpose": purpose,
                "request": f"R:{purpose}",
            },
            {"condition": "result", "failure": None, "kind": "request_terminal", "request": f"R:{purpose}"},
            {"kind": "settlement", "remote_stopped": True, "request": f"R:{purpose}", "usage": 1},
        ]
        c.append(case("request", f"fresh_{purpose}", declaration(), e))
    # cleanup derived association
    for name, purpose, disp, associated in (
        ("verification_failed", "verification", "close_failed", True),
        ("accounting_unknown", "accounting", "close_unknown", True),
        ("transport_failed", "transport_only", "close_failed", True),
        ("caller_left_open", "verification", "left_open", True),
        ("missing_association", "verification", "close_failed", False),
    ):
        e = deepcopy(base)
        idx = next(i for i, x in enumerate(e) if x.get("kind") == "revision")
        facts: list[Obj] = []
        if associated:
            facts.append(
                {
                    "kind": "cleanup_association",
                    "owner": "caller" if name == "caller_left_open" else "sdk",
                    "purpose": purpose,
                    "resource": "Q0",
                    "targets": ["A"],
                }
            )
        facts.append({"disposition": disp, "kind": "cleanup", "resource": "Q0"})
        e[idx:idx] = facts
        c.append(case("cleanup", name, declaration(), e))
    for name, purpose in (
        ("empty_transport_only", "transport_only"),
        ("empty_verification", "verification"),
        ("empty_accounting", "accounting"),
    ):
        e = deepcopy(base)
        idx = next(i for i, x in enumerate(e) if x.get("kind") == "revision")
        e[idx:idx] = [
            {"kind": "cleanup_association", "owner": "sdk", "purpose": purpose, "resource": "EMPTY", "targets": []},
            {"disposition": "close_failed", "kind": "cleanup", "resource": "EMPTY"},
        ]
        c.append(case("cleanup", name, declaration(), e))

    def binding_cleanup_events(
        *, disposition: str | None, association_targets: Sequence[str] | None, duplicate: bool = False
    ) -> list[Obj]:
        e = base_events(("A", "C"))
        e.insert(0, binding_receipt((), ()))
        idx = next(i for i, x in enumerate(e) if x.get("kind") == "revision")
        facts: list[Obj] = []
        if association_targets is not None:
            association: Obj = {
                "kind": "binding_cleanup_association",
                "owner": "sdk",
                "purpose": "accounting",
                "resource": "BIND:R0",
                "targets": list(association_targets),
            }
            facts.append(association)
            if duplicate:
                facts.append(deepcopy(association))
        if disposition is not None:
            facts.append({"disposition": disposition, "kind": "binding_cleanup", "resource": "BIND:R0"})
        e[idx:idx] = facts
        return e

    binding_cleanup_declaration = declaration(targets=("A", "C"))
    for name, disposition, association_targets, duplicate in (
        ("binding_closed", "closed", ("A",), False),
        ("binding_failed_local_a", "close_failed", ("A",), False),
        ("binding_missing_association", "close_failed", None, False),
        ("binding_foreign_association", "close_failed", ("Z",), False),
        ("binding_extra_association", None, ("A",), False),
        ("binding_duplicate_association", "closed", ("A",), True),
    ):
        c.append(
            case(
                "cleanup",
                name,
                deepcopy(binding_cleanup_declaration),
                binding_cleanup_events(
                    disposition=disposition, association_targets=association_targets, duplicate=duplicate
                ),
            )
        )
    # propagation direction
    for name, d, failed in (
        ("independent", declaration(targets=("A", "B", "C")), "A"),
        ("a_to_b", declaration(targets=("A", "B", "C"), deps=(("A", "B"),)), "A"),
        ("b_to_a", declaration(targets=("A", "B", "C"), deps=(("B", "A"),)), "A"),
        ("atomic_ab", declaration(targets=("A", "B", "C"), atomic=(("A", "B"),)), "A"),
        ("atomic_bc", declaration(targets=("A", "B", "C"), atomic=(("B", "C"),)), "B"),
    ):
        e = base_events(("A", "B", "C"))
        next(x for x in e if x.get("kind") == "assessment" and x.get("target") == failed)["finding"] = "unsatisfied"
        c.append(case("propagation", name, d, e))
    # decision/provenance variants
    for name, chain in (("direct", ("DEC",)), ("transitive", ("MID", "DEC")), ("unrelated", ())):
        d = declaration()
        e = deepcopy(base)
        out = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "OUT:A")
        parents = []
        if chain:
            e.insert(1, artifact("DAv0", "A", "decision"))
            out_index = e.index(out)
            e.insert(out_index, port("ROOT:A", "A", "decision_source", "DAv0", "decision"))
            e.insert(
                out_index + 1,
                provenance(
                    "DEC",
                    "DAv0",
                    "A",
                    decision=True,
                    source="operation_output",
                    port_name="decision_source",
                    activation="ROOT:A",
                ),
            )
            e.insert(out_index + 2, input_producer("ROOT:A", "A", "N", "decision_source", "DEC"))
            set_output_dependency(d, "N", "decision_source", ())
            parents = ["DEC"]
            if name == "transitive":
                e.insert(out_index + 2, port("ROOT:A", "A", "mid", "Av0", "candidate"))
                e.insert(
                    out_index + 3,
                    provenance(
                        "MID",
                        "Av0",
                        "A",
                        ("DEC",),
                        source="operation_output",
                        port_name="mid",
                        activation="ROOT:A",
                    ),
                )
                e.insert(out_index + 5, input_producer("ROOT:A", "A", "N", "mid", "MID"))
                set_output_dependency(d, "N", "mid", ("decision_source", "subject"), identity_input="subject")
                set_output_dependency(d, "N", "result", ("mid", "subject"), identity_input="subject")
                next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "MID")["parents"] = [
                    "DEC",
                    "ROOT:A:subject",
                ]
                parents = ["MID", "ROOT:A:subject"]
            else:
                set_output_dependency(d, "N", "result", ("decision_source", "subject"), identity_input="subject")
                parents.append("ROOT:A:subject")
            out["parents"] = parents
            seal = next(i for i, x in enumerate(e) if x.get("kind") == "seal_revision")
            e.insert(seal, {"collection": "artifacts", "key": "DA", "kind": "revision", "value": 0})
        c.append(case("decisions", name, d, e))
    e = mutate(base, "assessment", "consumed", {"context": "XAv0", "decision": "DAv0"})
    e.insert(1, artifact("DAv0", "A", "decision"))
    e.insert(
        next(i for i, x in enumerate(e) if x.get("kind") == "assessment"),
        port("ROOT:A", "A", "decision", "DAv0", "decision"),
    )
    retain_root_input(e, "A", "decision", "DAv0")
    seal = next(i for i, x in enumerate(e) if x.get("kind") == "seal_revision")
    e.insert(seal, {"collection": "artifacts", "key": "DA", "kind": "revision", "value": 0})
    d = declaration()
    wire_evidence_consumed(d, e, ("context", "decision"))
    c.append(case("decisions", "consumed", d, e))
    # Direct WorkflowInputRef root binding: the RootInputKey is itself the final producer.
    d = declaration()
    obj(arr(d["productions"])[0])["subject_source"] = "input"
    d["root_outputs"] = [
        {"input_port": "subject", "port": "result", "source_node": None, "source_port": None, "target": "A"}
    ]
    e = [x for x in deepcopy(base) if not (x.get("kind") == "provenance" and x.get("key") == "OUT:A")]
    f = next(x for x in e if x.get("kind") == "final")
    f["producer"] = "ROOT:A:subject"
    c.append(case("provenance", "root", d, e))
    wrong_root = deepcopy(e)
    next(x for x in wrong_root if x.get("kind") == "final")["producer"] = "ROOT:A:context"
    c.append(case("provenance", "direct_root_wrong_input", deepcopy(d), wrong_root))

    # A distinct identity-preserving output occurrence, rather than a renamed root key.
    d = declaration()
    set_output_dependency(d, "N", "alias", ("context",), identity_input="context")
    set_output_dependency(d, "N", "result", ("alias", "subject"), identity_input="subject")
    e = deepcopy(base)
    out = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "OUT:A")
    idx = e.index(out)
    e[idx:idx] = [
        port("ROOT:A", "A", "alias", "XAv0", "artifact"),
        provenance("ALIAS", "XAv0", "A", ("ROOT:A:context",), port_name="alias", activation="ROOT:A"),
        input_producer("ROOT:A", "A", "N", "alias", "ALIAS"),
    ]
    out["parents"] = ["ALIAS", "ROOT:A:subject"]
    c.append(case("provenance", "identity_alias", d, e))
    missing = [x for x in deepcopy(e) if not (x.get("kind") == "input_producer" and x.get("port") == "alias")]
    c.append(case("provenance", "missing_executed_input_producer", deepcopy(d), missing))
    extra = deepcopy(e)
    next(x for x in extra if x.get("kind") == "provenance" and x.get("key") == "ALIAS")["parents"] = [
        "ROOT:A:context",
        "ROOT:A:subject",
    ]
    c.append(case("provenance", "extra_dependency_parent", deepcopy(d), extra))
    swapped = deepcopy(base)
    subject_input = next(x for x in swapped if x.get("kind") == "input_producer" and x.get("port") == "subject")
    context_input = next(x for x in swapped if x.get("kind") == "input_producer" and x.get("port") == "context")
    subject_input["producer"], context_input["producer"] = context_input["producer"], subject_input["producer"]
    c.append(case("provenance", "swapped_dependency_parents", declaration(), swapped))
    commuted_parents = deepcopy(base)
    next(x for x in commuted_parents if x.get("kind") == "provenance" and x.get("key") == "OUT:A")["parents"] = [
        "ROOT:A:context",
        "ROOT:A:subject",
    ]
    c.append(case("provenance", "parent_order_invariant", declaration(), commuted_parents))
    extra_input = deepcopy(base)
    extra_input.insert(
        next(i for i, x in enumerate(extra_input) if x.get("kind") == "provenance" and x.get("key") == "OUT:A"),
        input_producer("ROOT:A", "A", "N", "not_an_executed_input", "ROOT:A:subject"),
    )
    c.append(case("provenance", "extra_input_producer", declaration(), extra_input))

    # Real structural wrapper/body projection with body WorkflowInputRef passthrough.
    d = declaration()
    obj(d["node_kinds"]).update({"SG": "container", "BN": "operation"})
    d["subgraphs"] = [
        {
            "body_input": "input",
            "body_node": "BN",
            "body_outcome": "ok",
            "body_port": "result",
            "body_source": "node_output",
            "input_port": "input",
            "node": "SG",
            "outcome": "ok",
            "port": "result",
        }
    ]
    set_output_dependency(d, "BN", "result", ("input",), identity_input="input")
    set_output_dependency(d, "N", "result", ("subgraph_result", "subject"), identity_input="subject")
    e = deepcopy(base)
    root_members = next(x for x in e if x.get("kind") == "membership" and x.get("parent") is None)
    arr(root_members["members"]).append("SUB")
    idx = next(i for i, x in enumerate(e) if x.get("kind") == "provenance" and x.get("key") == "OUT:A")
    e[idx:idx] = [
        entry("SUB", "A", "SG", node_kind="container", occurrence=1),
        terminal("SUB", "A", structural=True),
        port("SUB", "A", "input", "XAv0", "artifact", "SG"),
        input_producer("SUB", "A", "SG", "input", "ROOT:A:context"),
        {
            "closed": True,
            "expansion_outcome": "ok",
            "kind": "membership",
            "members": ["BODY"],
            "parent": "SUB",
            "status": "closed",
            "target": "A",
        },
        entry("BODY", "A", "BN", occurrence=0, parent="SUB"),
        terminal("BODY", "A"),
        port("BODY", "A", "input", "XAv0", "artifact", "BN"),
        input_producer("BODY", "A", "BN", "input", "ROOT:A:context"),
        port("BODY", "A", "result", "XAv0", "artifact", "BN"),
        provenance("BODYOUT", "XAv0", "A", ("ROOT:A:context",), node="BN", activation="BODY"),
        port("SUB", "A", "result", "XAv0", "artifact", "SG"),
        provenance("SUBGRAPH", "XAv0", "A", ("BODYOUT",), node="SG", activation="SUB"),
        port("ROOT:A", "A", "subgraph_result", "XAv0", "artifact"),
        input_producer("ROOT:A", "A", "N", "subgraph_result", "SUBGRAPH"),
    ]
    out = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "OUT:A")
    out["parents"] = ["SUBGRAPH", "ROOT:A:subject"]
    c.append(case("provenance", "subgraph", d, e))
    wrong_projection = deepcopy(e)
    next(x for x in wrong_projection if x.get("kind") == "provenance" and x.get("key") == "SUBGRAPH")["parents"] = [
        "ROOT:A:context"
    ]
    c.append(case("provenance", "subgraph_wrong_body_projection", deepcopy(d), wrong_projection))
    wrong_passthrough = deepcopy(e)
    next(
        x
        for x in wrong_passthrough
        if x.get("kind") == "input_producer" and x.get("activation") == "BODY" and x.get("port") == "input"
    )["producer"] = "ROOT:A:subject"
    c.append(case("provenance", "subgraph_wrong_input_passthrough", deepcopy(d), wrong_passthrough))

    # Structural body WorkflowInputRef -> WorkflowOutputRef passthrough has no body operation occurrence.
    d = declaration()
    obj(d["node_kinds"])["SG"] = "container"
    d["subgraphs"] = [
        {
            "body_input": "input",
            "body_node": None,
            "body_outcome": "ok",
            "body_port": None,
            "body_source": "workflow_input",
            "input_port": "input",
            "node": "SG",
            "outcome": "ok",
            "port": "result",
        }
    ]
    set_output_dependency(d, "N", "result", ("passthrough", "subject"), identity_input="subject")
    e = deepcopy(base)
    arr(next(x for x in e if x.get("kind") == "membership" and x.get("parent") is None)["members"]).append("SUB")
    out = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "OUT:A")
    idx = e.index(out)
    e[idx:idx] = [
        entry("SUB", "A", "SG", node_kind="container", occurrence=1),
        terminal("SUB", "A", structural=True),
        port("SUB", "A", "input", "XAv0", "artifact", "SG"),
        input_producer("SUB", "A", "SG", "input", "ROOT:A:context"),
        {
            "closed": True,
            "expansion_outcome": "ok",
            "kind": "membership",
            "members": [],
            "parent": "SUB",
            "status": "closed",
            "target": "A",
        },
        port("SUB", "A", "result", "XAv0", "artifact", "SG"),
        provenance("PASSTHROUGH", "XAv0", "A", ("ROOT:A:context",), node="SG", activation="SUB"),
        port("ROOT:A", "A", "passthrough", "XAv0", "artifact"),
        input_producer("ROOT:A", "A", "N", "passthrough", "PASSTHROUGH"),
    ]
    out["parents"] = ["PASSTHROUGH", "ROOT:A:subject"]
    c.append(case("provenance", "subgraph_nested_workflow_input_passthrough", d, e))
    missing_capture = [
        x
        for x in deepcopy(e)
        if not (x.get("kind") == "input_producer" and x.get("activation") == "SUB" and x.get("port") == "input")
    ]
    c.append(case("provenance", "subgraph_passthrough_missing_capture", deepcopy(d), missing_capture))
    wrong_capture = deepcopy(e)
    next(
        x
        for x in wrong_capture
        if x.get("kind") == "input_producer" and x.get("activation") == "SUB" and x.get("port") == "input"
    )["producer"] = "ROOT:A:subject"
    c.append(case("provenance", "subgraph_passthrough_wrong_capture", deepcopy(d), wrong_capture))

    d = declaration()
    set_output_dependency(d, "N", "version", ("context",), identity_input="context")
    set_output_dependency(d, "N", "result", ("version", "subject"), identity_input="subject")
    e = deepcopy(base)
    out = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "OUT:A")
    idx = e.index(out)
    e[idx:idx] = [
        port("ROOT:A", "A", "version", "XAv0", "artifact"),
        provenance("VERSION", "XAv0", "A", ("ROOT:A:context",), port_name="version", activation="ROOT:A"),
        input_producer("ROOT:A", "A", "N", "version", "VERSION"),
    ]
    out["parents"] = ["VERSION", "ROOT:A:subject"]
    c.append(case("provenance", "version_edge", d, e))

    def bound_events(node: str, declaration_id: str, key: str) -> list[Obj]:
        e = deepcopy(base)
        site = (declaration_id, node, "p", "A")
        e.insert(0, binding_receipt((site,), ((declaration_id, 0, 1),)))
        out = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "OUT:A")
        e.insert(
            e.index(out),
            provenance(
                key,
                "XAv0",
                "A",
                source="bound_input",
                node=node,
                port_name="p",
                binding_artifact=binding_ref(declaration_id),
            ),
        )
        e.insert(e.index(out), port("ROOT:A", "A", "bound_0", "XAv0", "artifact"))
        e.insert(e.index(out), input_producer("ROOT:A", "A", "N", "bound_0", key))
        out["parents"] = [key, "ROOT:A:subject"]
        return e

    c.append(
        case(
            "provenance",
            "bound_n0",
            binding_declaration(("D0", "N0", "p", "A")),
            bound_events("N0", "D0", "BOUND:N0:p"),
        )
    )
    c.append(
        case(
            "provenance",
            "bound_n1",
            binding_declaration(("D2", "N1", "p", "A")),
            bound_events("N1", "D2", "BOUND:N1:p"),
        )
    )

    def initial_collection_events() -> list[Obj]:
        e = deepcopy(base)
        site = ("D1", "N0", "items", "A")
        e.insert(0, binding_receipt((site,), (("D1", 0, 1), ("D1", 1, 1))))
        out = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "OUT:A")
        idx = e.index(out)
        facts = [artifact("B0v0", "A", "artifact"), artifact("B1v0", "A", "artifact")]
        for item_key, ref in enumerate(("B0v0", "B1v0")):
            facts.append(
                provenance(
                    f"BOUND:D1:{item_key}",
                    ref,
                    "A",
                    source="bound_input",
                    node="N0",
                    port_name="items",
                    binding_artifact=binding_ref("D1", item_key),
                )
            )
        facts.append(
            provenance(
                "INITIAL",
                "XAv0",
                "A",
                ("BOUND:D1:0", "BOUND:D1:1"),
                source="initial_collection",
                node="N0",
                port_name="items",
                declaration="D1",
            )
        )
        facts.append(input_producer("ROOT:A", "A", "N", "bound_0", "INITIAL"))
        facts.append(port("ROOT:A", "A", "bound_0", "XAv0", "artifact"))
        e[idx:idx] = facts
        out["parents"] = ["INITIAL", "ROOT:A:subject"]
        return e

    initial_declaration = binding_declaration(("D1", "N0", "items", "A"), collections=("D1",))
    c.append(case("provenance", "initial_collection", initial_declaration, initial_collection_events()))

    def map_item_events(count: int = 1, *, operation_expander: bool = False) -> list[Obj]:
        e = map_events(count, operation_expander=operation_expander)
        out = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "OUT:A")
        idx = e.index(out)
        producer = "OP:MAP:A:members"
        facts: list[Obj] = [
            artifact("CAv0", "A", "artifact"),
            {
                "artifact": "CAv0",
                "items": [{"key": item_key, "version": 1} for item_key in range(count)],
                "kind": "collection_value",
                "producer": producer,
                "target": "A",
            },
            port("MAP", "A", "members", "CAv0", "artifact", "EXP"),
            port("MAP", "A", "context", "XAv0", "artifact", "EXP"),
            input_producer("MAP", "A", "EXP", "context", "ROOT:A:context"),
            provenance(
                producer,
                "CAv0",
                "A",
                ("ROOT:A:context",),
                source="operation_output",
                node="EXP",
                port_name="members",
                activation="MAP",
            ),
        ]
        item_parents = []
        for item_key in range(count):
            member = f"M{item_key}"
            ref = f"MI{item_key}v1"
            key = f"MAPITEM:{item_key}"
            facts.extend(
                [
                    artifact(ref, "A", "artifact"),
                    port(member, "A", "item", ref, "artifact", "MEM"),
                    provenance(
                        key,
                        ref,
                        "A",
                        (producer,),
                        source="map_item",
                        node=None,
                        port_name="item",
                        expander="MAP",
                        member=member,
                        item_key=item_key,
                        item_version=1,
                    ),
                    input_producer(member, "A", "MEM", "item", key),
                    input_producer("ROOT:A", "A", "N", f"mapped_{item_key}", key),
                    port("ROOT:A", "A", f"mapped_{item_key}", ref, "artifact"),
                ]
            )
            item_parents.append(key)
        e[idx:idx] = facts
        out["parents"] = [*item_parents, "ROOT:A:subject"]
        return e

    def map_declaration(count: int, *, operation_expander: bool = False) -> Obj:
        d = declaration()
        if operation_expander:
            obj(d["node_kinds"])["EXP"] = "operation"
        set_output_dependency(
            d,
            "N",
            "result",
            (*tuple(f"mapped_{item_key}" for item_key in range(count)), "subject"),
            identity_input="subject",
        )
        return d

    c.append(case("provenance", "map_item", map_declaration(1), map_item_events()))
    c.append(case("provenance", "map_item_two_members", map_declaration(2), map_item_events(2)))
    d = map_declaration(2, operation_expander=True)
    c.append(
        case("provenance", "map_item_two_members_operation_expander", d, map_item_events(2, operation_expander=True))
    )

    def assessed_map_declaration(count: int) -> Obj:
        d = map_declaration(count)
        obj(d["node_kinds"])["MN"] = "operation"
        member_production = deepcopy(obj(arr(d["productions"])[0]))
        member_production.update(
            {
                "absence_queries": [],
                "consumed_ports": ["subject"],
                "evidence_port": "evidence",
                "meaning": "member_privacy",
                "node": "MN",
                "promise": "P_MEMBER",
                "read_state": [],
                "subject_port": "subject",
                "subject_source": "input",
            }
        )
        arr(d["productions"]).append(member_production)
        arr(d["requirements"]).append(
            {
                "consumed_ports": ["subject"],
                "coverage": ["K0"],
                "meaning": "member_privacy",
                "outcome": "ok",
                "subject_port": "subject",
                "target": "A",
            }
        )
        set_output_dependency(d, "MN", "evidence", ("subject",))
        return d

    def assessed_map_events(count: int) -> list[Obj]:
        e = map_item_events(count)
        insert_at = next(
            i for i, event in enumerate(e) if event.get("kind") == "provenance" and event.get("key") == "OUT:A"
        )
        member_facts: list[Obj] = []
        for item_key in range(count):
            member = f"M{item_key}"
            evidence_ref = f"ME{item_key}v0"
            entry_fact = next(
                event for event in e if event.get("kind") == "entry" and event.get("activation") == member
            )
            entry_fact["node"] = "MN"
            item_port = next(
                event
                for event in e
                if event.get("kind") == "port" and event.get("activation") == member and event.get("port") == "item"
            )
            item_port.update({"node": "MN", "role": "artifact"})
            item_input = next(
                event
                for event in e
                if event.get("kind") == "input_producer"
                and event.get("activation") == member
                and event.get("port") == "item"
            )
            item_input["node"] = "MN"
            member_facts.extend(
                [
                    artifact(evidence_ref, "A", "evidence"),
                    port(member, "A", "subject", "Av0", "candidate", "MN"),
                    input_producer(member, "A", "MN", "subject", "ROOT:A:subject"),
                    port(member, "A", "evidence", evidence_ref, "evidence", "MN"),
                    provenance(
                        f"EVID:{member}",
                        evidence_ref,
                        "A",
                        ("ROOT:A:subject",),
                        port_name="evidence",
                        activation=member,
                        node="MN",
                    ),
                    assessment(
                        "A",
                        activation=member,
                        consumed={"subject": "Av0"},
                        evidence_artifact=evidence_ref,
                        environment={
                            "absences": {},
                            "configurations": {"MN": "c0"},
                            "state": {},
                        },
                        fact=f"F:{member}:P_MEMBER",
                        node="MN",
                        promise="P_MEMBER",
                        subject_artifact="Av0",
                        subject_port="subject",
                    ),
                ]
            )
        e[insert_at:insert_at] = member_facts
        seal = next(i for i, event in enumerate(e) if event.get("kind") == "seal_revision")
        for item_key in range(count):
            e.insert(
                seal,
                {"collection": "artifacts", "key": f"ME{item_key}", "kind": "revision", "value": 0},
            )
            seal += 1
        e.insert(seal, {"collection": "configurations", "key": "MN", "kind": "revision", "value": "c0"})
        return e

    for count in range(3):
        c.append(
            case(
                "assessment",
                f"dynamic_occurrences_{count}",
                assessed_map_declaration(count),
                assessed_map_events(count),
            )
        )
    e = assessed_map_events(1)
    e = [event for event in e if not (event.get("kind") == "assessment" and event.get("activation") == "M0")]
    c.append(case("assessment", "dynamic_occurrence_missing", assessed_map_declaration(1), e))
    e = assessed_map_events(1)
    duplicate = deepcopy(
        next(event for event in e if event.get("kind") == "assessment" and event.get("activation") == "M0")
    )
    duplicate["fact"] = "F:M0:P_MEMBER:DUPLICATE"
    e.insert(next(i for i, event in enumerate(e) if event.get("kind") == "revision"), duplicate)
    c.append(case("assessment", "dynamic_occurrence_duplicate", assessed_map_declaration(1), e))

    def submit_dynamic_occurrences(count: int) -> list[Obj]:
        events = assessed_map_events(count)
        insertion = next(i for i, event in enumerate(events) if event.get("kind") == "revision")
        events[insertion:insertion] = [assessment_submission(f"M{index}", "P_MEMBER") for index in range(count)]
        return events

    for count in (1, 2):
        c.append(
            case(
                "assessment",
                f"dynamic_submissions_{count}",
                assessed_map_declaration(count),
                submit_dynamic_occurrences(count),
            )
        )
    e = submit_dynamic_occurrences(1)
    insertion = next(i for i, event in enumerate(e) if event.get("kind") == "revision")
    e.insert(insertion, assessment_submission("M0", "P_MEMBER"))
    c.append(case("assessment", "dynamic_submission_repeated", assessed_map_declaration(1), e))
    e = submit_dynamic_occurrences(1)
    insertion = next(i for i, event in enumerate(e) if event.get("kind") == "revision")
    e.insert(insertion, {"fact": "F:FOREIGN:P_MEMBER", "kind": "assessment_submission"})
    c.append(case("assessment", "dynamic_submission_foreign", assessed_map_declaration(1), e))

    def inject_member_assessment(events: list[Obj]) -> list[Obj]:
        injected = deepcopy(events)
        insertion = next(i for i, event in enumerate(injected) if event.get("kind") == "revision")
        injected.insert(
            insertion,
            assessment(
                "A",
                activation="M0",
                consumed={"subject": "Av0"},
                evidence_artifact="ME0v0",
                environment={"absences": {}, "configurations": {"MN": "c0"}, "state": {}},
                fact="F:M0:P_MEMBER",
                node="MN",
                promise="P_MEMBER",
                subject_artifact="Av0",
                subject_port="subject",
            ),
        )
        return injected

    # A terminal expansion can retain a selected reservation without starting
    # the child. It contributes no assessment owner.
    e = map_events(0)
    next(event for event in e if event.get("kind") == "provenance" and event.get("key") == "OUT:A")["parents"] = [
        "ROOT:A:subject"
    ]
    map_entry = next(event for event in e if event.get("kind") == "entry" and event.get("activation") == "MAP")
    map_entry.update({"state_category": "failure", "state_outcome": None})
    map_terminal = next(event for event in e if event.get("kind") == "terminal" and event.get("activation") == "MAP")
    map_terminal.update({"category": "failure", "outcome": None, "reasons": ["execution_failed"]})
    membership = next(event for event in e if event.get("kind") == "membership" and event.get("parent") == "MAP")
    membership.update({"expansion_outcome": None, "status": "failed"})
    insertion = e.index(membership)
    e.insert(
        insertion,
        {"activation": "M0", "kind": "reservation", "parent": "MAP", "selected": True, "target": "A"},
    )
    c.append(case("assessment", "dynamic_unreached_failed_expansion", assessed_map_declaration(0), e))
    c.append(
        case(
            "assessment",
            "dynamic_unreached_failed_expansion_injected",
            assessed_map_declaration(0),
            inject_member_assessment(e),
        )
    )

    def non_success_member_events(category: str, *, closed_unstarted: bool) -> list[Obj]:
        events = map_item_events(1)
        member_entry = next(
            event for event in events if event.get("kind") == "entry" and event.get("activation") == "M0"
        )
        member_entry.update(
            {
                "closed_unstarted": closed_unstarted,
                "node": "MN",
                "state_category": category,
                "state_outcome": None,
            }
        )
        member_terminal = next(
            event for event in events if event.get("kind") == "terminal" and event.get("activation") == "M0"
        )
        replacement = terminal("M0", "A", category, None)
        member_terminal.clear()
        member_terminal.update(replacement)
        next(
            event
            for event in events
            if event.get("kind") == "port" and event.get("activation") == "M0" and event.get("port") == "item"
        )["node"] = "MN"
        next(
            event
            for event in events
            if event.get("kind") == "input_producer" and event.get("activation") == "M0" and event.get("port") == "item"
        )["node"] = "MN"
        return events

    for name, category, closed_unstarted in (
        ("blocked_unreached", "blocked", True),
        ("started_failure", "failure", False),
    ):
        e = non_success_member_events(category, closed_unstarted=closed_unstarted)
        c.append(case("assessment", f"dynamic_{name}", assessed_map_declaration(1), e))
        c.append(
            case(
                "assessment",
                f"dynamic_{name}_injected",
                assessed_map_declaration(1),
                inject_member_assessment(e),
            )
        )

    root_missing_terminal = [
        deepcopy(event)
        for event in base
        if not (event.get("kind") == "terminal" and event.get("activation") == "ROOT:A")
    ]
    root_unsubmitted = [event for event in root_missing_terminal if event.get("kind") != "assessment_submission"]
    c.append(case("assessment", "root_missing_terminal_unsubmitted", declaration(), root_unsubmitted))
    c.append(case("assessment", "root_missing_terminal_submitted", declaration(), root_missing_terminal))

    dynamic_missing_terminal = [
        event
        for event in assessed_map_events(1)
        if not (event.get("kind") == "terminal" and event.get("activation") == "M0")
    ]
    c.append(
        case(
            "assessment",
            "dynamic_missing_terminal_unsubmitted",
            assessed_map_declaration(1),
            dynamic_missing_terminal,
        )
    )
    dynamic_submitted = deepcopy(dynamic_missing_terminal)
    insertion = next(i for i, event in enumerate(dynamic_submitted) if event.get("kind") == "revision")
    dynamic_submitted.insert(insertion, assessment_submission("M0", "P_MEMBER"))
    c.append(
        case(
            "assessment",
            "dynamic_missing_terminal_submitted",
            assessed_map_declaration(1),
            dynamic_submitted,
        )
    )

    def map_item_endpoint(
        *,
        expander: str = "EXP",
        expansion_outcome: str = "ok",
        item_input: str = "item",
        member: str = "MN",
        membership_port: str = "members",
        path: Sequence[str] = (),
    ) -> Obj:
        return {
            "expander": expander,
            "expansion_outcome": expansion_outcome,
            "item_input": item_input,
            "member": member,
            "membership_port": membership_port,
            "path": list(path),
        }

    def direct_item_declaration(
        count: int,
        *,
        consumed_only: bool = False,
        endpoint: Obj | None = None,
        candidate_port: str = "result",
    ) -> Obj:
        d = map_declaration(count, operation_expander=True)
        actual_endpoint = deepcopy(endpoint or map_item_endpoint())
        obj(d["node_kinds"]).update({cast(str, actual_endpoint["member"]): "operation", "J": "operation"})
        member_production = deepcopy(obj(arr(d["productions"])[0]))
        member_production.update(
            {
                "absence_queries": [],
                "consumed_ports": [actual_endpoint["item_input"]],
                "evidence_port": "evidence",
                "meaning": "item_privacy",
                "node": actual_endpoint["member"],
                "promise": "P_ITEM",
                "read_state": [],
                "subject_port": "subject" if consumed_only else actual_endpoint["item_input"],
                "subject_source": "input",
            }
        )
        arr(d["productions"]).append(member_production)
        d["map_item_requirements"] = [
            {
                "candidate_port": candidate_port,
                "consumed_endpoints": [deepcopy(actual_endpoint)],
                "coverage": ["K0"],
                "meaning": "item_privacy",
                "promise": "P_ITEM",
                "subject_endpoint": None if consumed_only else deepcopy(actual_endpoint),
                "target": "A",
            }
        ]
        d["map_routes"] = [
            {
                "expander": actual_endpoint["expander"],
                "item_input": actual_endpoint["item_input"],
                "member": actual_endpoint["member"],
                "membership_port": actual_endpoint["membership_port"],
                "outcome": actual_endpoint["expansion_outcome"],
                "path": deepcopy(actual_endpoint["path"]),
            }
        ]
        d["keyed_joins"] = [
            {
                "accepted_categories": ["success"],
                "join": "J",
                "reduction": "all_by_key",
                "source": actual_endpoint["expander"],
            }
        ]
        set_output_dependency(d, "MN", "evidence", (cast(str, actual_endpoint["item_input"]),))
        set_output_dependency(d, "N", "result", ("membership", "subject"), identity_input="subject")
        return d

    def direct_item_events(count: int, *, consumed_only: bool = False, submit: bool = True) -> list[Obj]:
        events = map_item_events(count, operation_expander=True)
        root_membership = next(
            event for event in events if event.get("kind") == "membership" and event.get("parent") is None
        )
        arr(root_membership["members"]).append("JOIN")
        cut = next(i for i, event in enumerate(events) if event.get("kind") == "revision")
        events[cut:cut] = [entry("JOIN", "A", "J"), terminal("JOIN", "A")]
        events[:] = [
            event
            for event in events
            if not (
                event.get("activation") == "ROOT:A"
                and (
                    (event.get("kind") == "port" and cast(str, event.get("port")).startswith("mapped_"))
                    or (event.get("kind") == "input_producer" and cast(str, event.get("port")).startswith("mapped_"))
                )
            )
        ]
        out = next(event for event in events if event.get("kind") == "provenance" and event.get("key") == "OUT:A")
        out["parents"] = ["OP:MAP:A:members", "ROOT:A:subject"]
        out_index = events.index(out)
        events[out_index:out_index] = [
            port("ROOT:A", "A", "membership", "CAv0", "artifact"),
            input_producer("ROOT:A", "A", "N", "membership", "OP:MAP:A:members"),
        ]
        member_facts: list[Obj] = []
        for item_key in range(count):
            member = f"M{item_key}"
            item_ref = f"MI{item_key}v1"
            evidence_ref = f"ME{item_key}v0"
            next(event for event in events if event.get("kind") == "entry" and event.get("activation") == member)[
                "node"
            ] = "MN"
            next(
                event
                for event in events
                if event.get("kind") == "port" and event.get("activation") == member and event.get("port") == "item"
            )["node"] = "MN"
            next(
                event
                for event in events
                if event.get("kind") == "input_producer"
                and event.get("activation") == member
                and event.get("port") == "item"
            )["node"] = "MN"
            facts = [
                artifact(evidence_ref, "A", "evidence"),
                port(member, "A", "evidence", evidence_ref, "evidence", "MN"),
                provenance(
                    f"EVID:{member}",
                    evidence_ref,
                    "A",
                    (f"MAPITEM:{item_key}",),
                    port_name="evidence",
                    activation=member,
                    node="MN",
                ),
            ]
            subject_ref = item_ref
            subject_port = "item"
            if consumed_only:
                facts.extend(
                    [
                        port(member, "A", "subject", "Av0", "candidate", "MN"),
                        input_producer(member, "A", "MN", "subject", "ROOT:A:subject"),
                    ]
                )
                subject_ref = "Av0"
                subject_port = "subject"
            facts.append(
                assessment(
                    "A",
                    activation=member,
                    consumed={"item": item_ref},
                    evidence_artifact=evidence_ref,
                    environment={"absences": {}, "configurations": {"MN": "c0"}, "state": {}},
                    fact=f"F:{member}:P_ITEM",
                    node="MN",
                    promise="P_ITEM",
                    subject_artifact=subject_ref,
                    subject_port=subject_port,
                )
            )
            if submit:
                facts.append(assessment_submission(member, "P_ITEM"))
            member_facts.extend(facts)
        events[out_index:out_index] = member_facts
        seal = next(i for i, event in enumerate(events) if event.get("kind") == "seal_revision")
        for item_key in range(count):
            events.insert(
                seal,
                {"collection": "artifacts", "key": f"ME{item_key}", "kind": "revision", "value": 0},
            )
            seal += 1
            events.insert(
                seal,
                {"collection": "artifacts", "key": f"MI{item_key}", "kind": "revision", "value": 1},
            )
            seal += 1
        events.insert(seal, {"collection": "configurations", "key": "MN", "kind": "revision", "value": "c0"})
        return events

    def block_keyed_join(events: list[Obj], activation: str = "JOIN") -> None:
        join_entry = next(
            event for event in events if event.get("kind") == "entry" and event.get("activation") == activation
        )
        join_entry.update({"closed_unstarted": True, "state_category": "blocked", "state_outcome": None})
        join_terminal = next(
            event for event in events if event.get("kind") == "terminal" and event.get("activation") == activation
        )
        join_terminal.clear()
        join_terminal.update(terminal(activation, "A", "blocked", None))

    for count in range(3):
        c.append(
            case("map_item_evidence", f"direct_{count}", direct_item_declaration(count), direct_item_events(count))
        )
    d = direct_item_declaration(2)
    obj(d["limits"])["max_submissions"] = 2
    c.append(case("map_item_bounds", "submissions_one_over", d, direct_item_events(2)))
    d = direct_item_declaration(2)
    obj(d["limits"])["max_verified_evidence"] = 2
    c.append(case("map_item_bounds", "verified_one_over", d, direct_item_events(2)))
    c.append(
        case(
            "map_item_evidence",
            "typed_consumed_endpoint",
            direct_item_declaration(1, consumed_only=True),
            direct_item_events(1, consumed_only=True),
        )
    )

    e = direct_item_events(1)
    e = [
        event
        for event in e
        if not (event.get("kind") == "assessment_submission" and event.get("fact") == "F:M0:P_ITEM")
    ]
    c.append(case("map_item_evidence", "missing_submission", direct_item_declaration(1), e))
    e = direct_item_events(1)
    next(event for event in e if event.get("kind") == "assessment_submission" and event.get("fact") == "F:M0:P_ITEM")[
        "fact"
    ] = "F:FOREIGN:P_ITEM"
    c.append(case("map_item_evidence", "foreign_submission", direct_item_declaration(1), e))
    e = direct_item_events(1)
    insertion = next(i for i, event in enumerate(e) if event.get("kind") == "revision")
    e.insert(insertion, assessment_submission("M0", "P_ITEM"))
    c.append(case("map_item_evidence", "repeated_submission", direct_item_declaration(1), e))
    e = [
        event
        for event in direct_item_events(1)
        if not (
            event.get("kind") == "assessment"
            and event.get("activation") == "M0"
            or event.get("kind") == "assessment_submission"
            and event.get("fact") == "F:M0:P_ITEM"
        )
    ]
    c.append(case("map_item_evidence", "missing_fact", direct_item_declaration(1), e))
    e = direct_item_events(2)
    next(event for event in e if event.get("kind") == "assessment_submission" and event.get("fact") == "F:M0:P_ITEM")[
        "fact"
    ] = "F:M1:P_ITEM"
    c.append(case("map_item_evidence", "copied_member_submission", direct_item_declaration(2), e))

    for field, value in (
        ("expander", "OTHER"),
        ("member", "OTHER"),
        ("item_input", "other"),
        ("membership_port", "other"),
        ("expansion_outcome", "other"),
    ):
        d = direct_item_declaration(1)
        obj(obj(arr(d["map_item_requirements"])[0])["subject_endpoint"])[field] = value
        c.append(case("map_item_admission", f"invalid_{field}", d, [], "admission"))
    d = direct_item_declaration(1, candidate_port="other")
    c.append(case("map_item_admission", "invalid_candidate_port", d, [], "admission"))
    d = direct_item_declaration(1)
    obj(obj(arr(d["map_item_requirements"])[0])["subject_endpoint"])["path"] = ["MN"]
    c.append(case("map_item_admission", "invalid_path_owner", d, [], "admission"))
    d = direct_item_declaration(1)
    typed_requirement = obj(arr(d["map_item_requirements"])[0])
    obj(typed_requirement["subject_endpoint"])["expander"] = "OTHER"
    obj(arr(typed_requirement["consumed_endpoints"])[0])["expander"] = "OTHER"
    c.append(case("map_item_admission", "unresolved_projection", d, [], "admission"))
    d = direct_item_declaration(1)
    obj(d["node_kinds"])["UNRELATED"] = "container"
    typed_requirement = obj(arr(d["map_item_requirements"])[0])
    obj(typed_requirement["subject_endpoint"])["path"] = ["UNRELATED"]
    obj(arr(typed_requirement["consumed_endpoints"])[0])["path"] = ["UNRELATED"]
    c.append(case("map_item_admission", "valid_container_wrong_route", d, [], "admission"))
    d = direct_item_declaration(1)
    arr(d["map_item_requirements"]).append(deepcopy(arr(d["map_item_requirements"])[0]))
    c.append(case("map_item_admission", "duplicate_typed_requirement", d, [], "admission"))

    e = direct_item_events(1, consumed_only=True)
    next(event for event in e if event.get("kind") == "provenance" and event.get("key") == "MAPITEM:0")["member"] = (
        "OTHER"
    )
    c.append(
        case(
            "map_item_evidence",
            "typed_consumed_wrong_owner",
            direct_item_declaration(1, consumed_only=True),
            e,
        )
    )

    e = direct_item_events(1)
    item_assessment = next(
        event for event in e if event.get("kind") == "assessment" and event.get("activation") == "M0"
    )
    item_assessment["subject_artifact"] = "Av0"
    c.append(case("map_item_evidence", "wrong_subject_artifact", direct_item_declaration(1), e))
    e = direct_item_events(1)
    next(event for event in e if event.get("kind") == "provenance" and event.get("key") == "MAPITEM:0")[
        "item_version"
    ] = 2
    c.append(case("map_item_evidence", "wrong_item_version", direct_item_declaration(1), e))
    e = direct_item_events(1)
    next(event for event in e if event.get("kind") == "provenance" and event.get("key") == "MAPITEM:0")["item_key"] = 1
    c.append(case("map_item_evidence", "wrong_item_key", direct_item_declaration(1), e))
    e = direct_item_events(1)
    next(
        event
        for event in e
        if event.get("kind") == "input_producer" and event.get("activation") == "M0" and event.get("port") == "item"
    )["producer"] = "ROOT:A:subject"
    c.append(case("map_item_evidence", "wrong_item_owner", direct_item_declaration(1), e))
    e = direct_item_events(1)
    e = [
        event
        for event in e
        if not (
            event.get("kind") == "revision" and event.get("collection") == "artifacts" and event.get("key") == "MI0"
        )
    ]
    c.append(case("map_item_evidence", "item_unknown", direct_item_declaration(1), e))

    e = direct_item_events(1)
    e = [
        event
        for event in e
        if not (
            event.get("kind") == "assessment"
            and event.get("activation") == "M0"
            or event.get("kind") == "assessment_submission"
            and event.get("fact") == "F:M0:P_ITEM"
            or event.get("kind") == "artifact"
            and event.get("ref") == "ME0v0"
            or event.get("kind") == "port"
            and event.get("activation") == "M0"
            and event.get("port") == "evidence"
            or event.get("kind") == "provenance"
            and event.get("key") == "EVID:M0"
            or event.get("kind") == "revision"
            and event.get("collection") == "artifacts"
            and event.get("key") == "ME0"
        )
    ]
    member_entry = next(event for event in e if event.get("kind") == "entry" and event.get("activation") == "M0")
    member_entry.update({"state_category": "failure", "state_outcome": None})
    member_terminal = next(event for event in e if event.get("kind") == "terminal" and event.get("activation") == "M0")
    member_terminal.clear()
    member_terminal.update(terminal("M0", "A", "failure", None))
    block_keyed_join(e)
    c.append(case("map_item_evidence", "member_non_success", direct_item_declaration(1), e))
    e = direct_item_events(1)
    e = [
        event
        for event in e
        if not (
            event.get("kind") == "assessment"
            and event.get("activation") == "M0"
            or event.get("kind") == "assessment_submission"
            and event.get("fact") == "F:M0:P_ITEM"
            or event.get("kind") == "artifact"
            and event.get("ref") == "ME0v0"
            or event.get("kind") == "port"
            and event.get("activation") == "M0"
            and event.get("port") == "evidence"
            or event.get("kind") == "provenance"
            and event.get("key") == "EVID:M0"
            or event.get("kind") == "revision"
            and event.get("collection") == "artifacts"
            and event.get("key") == "ME0"
        )
    ]
    member_entry = next(event for event in e if event.get("kind") == "entry" and event.get("activation") == "M0")
    member_entry.update({"closed_unstarted": True, "state_category": "blocked", "state_outcome": None})
    member_terminal = next(event for event in e if event.get("kind") == "terminal" and event.get("activation") == "M0")
    member_terminal.clear()
    member_terminal.update(terminal("M0", "A", "blocked", None))
    block_keyed_join(e)
    c.append(case("map_item_evidence", "member_blocked_unreached", direct_item_declaration(1), e))

    for status in ("failed", "overflow"):
        e = direct_item_events(0)
        membership = next(event for event in e if event.get("kind") == "membership" and event.get("parent") == "MAP")
        membership.update({"expansion_outcome": None, "status": status})
        block_keyed_join(e)
        if status == "failed":
            e.insert(
                e.index(membership),
                {"activation": "M0", "kind": "reservation", "parent": "MAP", "selected": True, "target": "A"},
            )
        c.append(case("map_item_evidence", f"expansion_{status}", direct_item_declaration(0), e))
    e = direct_item_events(0)
    membership = next(event for event in e if event.get("kind") == "membership" and event.get("parent") == "MAP")
    membership.update({"closed": False, "status": "open"})
    e = [event for event in e if not (event.get("kind") == "terminal" and event.get("activation") == "JOIN")]
    c.append(case("map_item_evidence", "expansion_open", direct_item_declaration(0), e))

    d = direct_item_declaration(1)
    set_output_dependency(d, "N", "result", ("subject",), identity_input="subject")
    e = direct_item_events(1)
    e = [
        event
        for event in e
        if not (
            event.get("activation") == "ROOT:A"
            and event.get("port") == "membership"
            and event.get("kind") in {"port", "input_producer"}
        )
    ]
    next(event for event in e if event.get("kind") == "provenance" and event.get("key") == "OUT:A")["parents"] = [
        "ROOT:A:subject"
    ]
    c.append(case("map_item_evidence", "different_final_ancestry", d, e))

    d = direct_item_declaration(1)
    d["root_outputs"] = [
        {"input_port": "subject", "port": "result", "source_node": None, "source_port": None, "target": "A"}
    ]
    e = direct_item_events(1)
    next(event for event in e if event.get("kind") == "final")["producer"] = "ROOT:A:subject"
    next(
        event
        for event in e
        if event.get("kind") == "port"
        and event.get("activation") == "ROOT:A"
        and event.get("node") == "N"
        and event.get("port") == "result"
    )["role"] = "artifact"
    c.append(case("map_item_evidence", "candidate_passthrough_unrelated", d, e))

    nested_endpoint = map_item_endpoint(path=("SG",))
    d = direct_item_declaration(1, endpoint=nested_endpoint)
    obj(d["node_kinds"])["SG"] = "container"
    arr(d["subgraphs"]).append(
        {
            "body_input": "context",
            "body_node": "EXP",
            "body_outcome": "ok",
            "body_port": "members",
            "body_source": "node_output",
            "input_port": "context",
            "node": "SG",
            "outcome": "ok",
            "port": "nested_members",
        }
    )
    e = direct_item_events(1)
    root_membership = next(event for event in e if event.get("kind") == "membership" and event.get("parent") is None)
    root_members = arr(root_membership["members"])
    root_members[root_members.index("MAP")] = "WRAP"
    root_members.remove("JOIN")
    next(event for event in e if event.get("kind") == "entry" and event.get("activation") == "MAP")["parent"] = "WRAP"
    next(event for event in e if event.get("kind") == "entry" and event.get("activation") == "JOIN")["parent"] = "WRAP"
    next(
        event
        for event in e
        if event.get("kind") == "input_producer"
        and event.get("activation") == "ROOT:A"
        and event.get("port") == "membership"
    )["producer"] = "SGOUT:WRAP:A:nested_members"
    next(event for event in e if event.get("kind") == "provenance" and event.get("key") == "OUT:A")["parents"] = [
        "SGOUT:WRAP:A:nested_members",
        "ROOT:A:subject",
    ]
    insertion = next(
        i for i, event in enumerate(e) if event.get("kind") == "provenance" and event.get("key") == "OUT:A"
    )
    e[insertion:insertion] = [
        {"activation": "WRAP", "kind": "reservation", "parent": None, "selected": True, "target": "A"},
        entry("WRAP", "A", "SG", node_kind="container"),
        terminal("WRAP", "A", structural=True),
        port("WRAP", "A", "context", "XAv0", "artifact", "SG"),
        input_producer("WRAP", "A", "SG", "context", "ROOT:A:context"),
        port("WRAP", "A", "nested_members", "CAv0", "artifact", "SG"),
        provenance(
            "SGOUT:WRAP:A:nested_members",
            "CAv0",
            "A",
            ("OP:MAP:A:members",),
            node="SG",
            port_name="nested_members",
            activation="WRAP",
        ),
        {"activation": "MAP", "kind": "reservation", "parent": "WRAP", "selected": True, "target": "A"},
        {"activation": "JOIN", "kind": "reservation", "parent": "WRAP", "selected": True, "target": "A"},
        {
            "closed": True,
            "expansion_outcome": "ok",
            "kind": "membership",
            "members": ["MAP", "JOIN"],
            "parent": "WRAP",
            "status": "closed",
            "target": "A",
        },
    ]
    c.append(case("map_item_evidence", "nested_path", d, e))

    alternate_endpoint = map_item_endpoint(expansion_outcome="alternate", membership_port="alternate_members")
    d = direct_item_declaration(1, endpoint=alternate_endpoint)
    map_input = obj(arr(d["map_inputs"])[0])
    map_input.update({"membership_port": "alternate_members", "outcome": "alternate"})
    expander_dependency = next(
        dependency
        for dependency in map(obj, arr(d["output_dependencies"]))
        if dependency.get("node") == "EXP" and dependency.get("port") == "members"
    )
    expander_dependency.update({"outcome": "alternate", "port": "alternate_members"})
    e = cast(
        list[Obj],
        _replace_refs(
            direct_item_events(1),
            {"OP:MAP:A:members": "OP:MAP:A:alternate_members"},
        ),
    )
    next(event for event in e if event.get("kind") == "entry" and event.get("activation") == "MAP")["state_outcome"] = (
        "alternate"
    )
    next(event for event in e if event.get("kind") == "terminal" and event.get("activation") == "MAP")["outcome"] = (
        "alternate"
    )
    next(event for event in e if event.get("kind") == "membership" and event.get("parent") == "MAP")[
        "expansion_outcome"
    ] = "alternate"
    next(
        event
        for event in e
        if event.get("kind") == "port" and event.get("activation") == "MAP" and event.get("port") == "members"
    )["port"] = "alternate_members"
    next(
        event for event in e if event.get("kind") == "provenance" and event.get("key") == "OP:MAP:A:alternate_members"
    )["port"] = "alternate_members"
    c.append(case("map_item_evidence", "distinct_outcome_port", d, e))

    e = cast(list[Obj], _replace_refs(direct_item_events(2), {"MI1v1": "MI0v2"}))
    second_artifact = next(event for event in e if event.get("kind") == "artifact" and event.get("ref") == "MI0v2")
    second_artifact.update({"key": "MI0", "version": 2})
    second_item = next(event for event in e if event.get("kind") == "provenance" and event.get("key") == "MAPITEM:1")
    second_item.update({"item_key": 0, "item_version": 2})
    collection = next(event for event in e if event.get("kind") == "collection_value")
    obj(arr(collection["items"])[1]).update({"key": 0, "version": 2})
    e = [
        event
        for event in e
        if not (
            event.get("kind") == "revision" and event.get("collection") == "artifacts" and event.get("key") == "MI1"
        )
    ]
    next(
        event
        for event in e
        if event.get("kind") == "revision" and event.get("collection") == "artifacts" and event.get("key") == "MI0"
    )["value"] = 2
    c.append(case("map_item_evidence", "item_stale", direct_item_declaration(2), e))

    for name, field, value in (
        ("wrong_expander", "expander", "OTHER"),
        ("wrong_member", "member", "OTHER"),
        ("wrong_target", "target", "B"),
    ):
        e = direct_item_events(1)
        next(event for event in e if event.get("kind") == "provenance" and event.get("key") == "MAPITEM:0")[field] = (
            value
        )
        c.append(case("map_item_evidence", name, direct_item_declaration(1), e))
    e = direct_item_events(1)
    next(event for event in e if event.get("kind") == "artifact" and event.get("ref") == "MI0v1")["invocation"] = "I1"
    c.append(case("map_item_evidence", "wrong_invocation", direct_item_declaration(1), e))

    d = direct_item_declaration(1)
    obj(d["limits"])["max_productions"] = 3
    obj(d["node_kinds"]).update({"EXP2": "operation", "J2": "operation", "MN2": "operation"})
    arr(d["map_inputs"]).append(
        {"expander": "EXP2", "item_input": "item2", "membership_port": "members2", "outcome": "ok"}
    )
    second_production = deepcopy(
        next(production for production in map(obj, arr(d["productions"])) if production.get("promise") == "P_ITEM")
    )
    second_production.update(
        {
            "consumed_ports": ["item2"],
            "meaning": "item_privacy_2",
            "node": "MN2",
            "promise": "P_ITEM_2",
            "subject_port": "item2",
        }
    )
    arr(d["productions"]).append(second_production)
    arr(d["output_dependencies"]).extend(
        [
            {"identity_input": None, "inputs": ["context"], "node": "EXP2", "outcome": "ok", "port": "members2"},
            {"identity_input": None, "inputs": ["item2"], "node": "MN2", "outcome": "ok", "port": "evidence"},
        ]
    )
    second_endpoint = map_item_endpoint(expander="EXP2", item_input="item2", member="MN2", membership_port="members2")
    arr(d["map_item_requirements"]).append(
        {
            "candidate_port": "result",
            "consumed_endpoints": [deepcopy(second_endpoint)],
            "coverage": ["K0"],
            "meaning": "item_privacy_2",
            "promise": "P_ITEM_2",
            "subject_endpoint": deepcopy(second_endpoint),
            "target": "A",
        }
    )
    arr(d["map_routes"]).append(
        {
            "expander": "EXP2",
            "item_input": "item2",
            "member": "MN2",
            "membership_port": "members2",
            "outcome": "ok",
            "path": [],
        }
    )
    arr(d["keyed_joins"]).append(
        {
            "accepted_categories": ["success"],
            "join": "J2",
            "reduction": "all_by_key",
            "source": "EXP2",
        }
    )
    set_output_dependency(d, "N", "result", ("membership", "membership2", "subject"), identity_input="subject")
    e = direct_item_events(1)
    root_membership = next(event for event in e if event.get("kind") == "membership" and event.get("parent") is None)
    arr(root_membership["members"]).extend(("MAP2", "JOIN2"))
    out = next(event for event in e if event.get("kind") == "provenance" and event.get("key") == "OUT:A")
    arr(out["parents"]).insert(1, "OP:MAP2:A:members2")
    insertion = e.index(out)
    e[insertion:insertion] = [
        artifact("CBv0", "A", "artifact"),
        artifact("MJ0v1", "A", "artifact"),
        artifact("MF0v0", "A", "evidence"),
        {
            "artifact": "CBv0",
            "items": [{"key": 0, "version": 1}],
            "kind": "collection_value",
            "producer": "OP:MAP2:A:members2",
            "target": "A",
        },
        port("ROOT:A", "A", "membership2", "CBv0", "artifact"),
        input_producer("ROOT:A", "A", "N", "membership2", "OP:MAP2:A:members2"),
        port("MAP2", "A", "members2", "CBv0", "artifact", "EXP2"),
        port("MAP2", "A", "context", "XAv0", "artifact", "EXP2"),
        input_producer("MAP2", "A", "EXP2", "context", "ROOT:A:context"),
        provenance(
            "OP:MAP2:A:members2",
            "CBv0",
            "A",
            ("ROOT:A:context",),
            node="EXP2",
            port_name="members2",
            activation="MAP2",
        ),
        port("Z0", "A", "item2", "MJ0v1", "artifact", "MN2"),
        provenance(
            "MAPITEM2:0",
            "MJ0v1",
            "A",
            ("OP:MAP2:A:members2",),
            source="map_item",
            node=None,
            port_name="item2",
            expander="MAP2",
            member="Z0",
            item_key=0,
            item_version=1,
        ),
        input_producer("Z0", "A", "MN2", "item2", "MAPITEM2:0"),
        port("Z0", "A", "evidence", "MF0v0", "evidence", "MN2"),
        provenance(
            "EVID:Z0",
            "MF0v0",
            "A",
            ("MAPITEM2:0",),
            port_name="evidence",
            activation="Z0",
            node="MN2",
        ),
        assessment(
            "A",
            activation="Z0",
            consumed={"item2": "MJ0v1"},
            evidence_artifact="MF0v0",
            environment={"absences": {}, "configurations": {"MN2": "c0"}, "state": {}},
            fact="F:Z0:P_ITEM_2",
            node="MN2",
            promise="P_ITEM_2",
            subject_artifact="MJ0v1",
            subject_port="item2",
        ),
        assessment_submission("Z0", "P_ITEM_2"),
        entry("MAP2", "A", "EXP2"),
        terminal("MAP2", "A"),
        entry("JOIN2", "A", "J2"),
        terminal("JOIN2", "A"),
        {"activation": "Z0", "kind": "reservation", "parent": "MAP2", "selected": True, "target": "A"},
        entry("Z0", "A", "MN2", parent="MAP2"),
        terminal("Z0", "A"),
        {
            "closed": True,
            "expansion_outcome": "ok",
            "kind": "membership",
            "members": ["Z0"],
            "parent": "MAP2",
            "status": "closed",
            "target": "A",
        },
    ]
    seal = next(i for i, event in enumerate(e) if event.get("kind") == "seal_revision")
    e[seal:seal] = [
        {"collection": "artifacts", "key": "MF0", "kind": "revision", "value": 0},
        {"collection": "artifacts", "key": "MJ0", "kind": "revision", "value": 1},
        {"collection": "configurations", "key": "MN2", "kind": "revision", "value": "c0"},
    ]
    c.append(case("map_item_evidence", "two_independent_maps", d, e))
    missing_second = [
        deepcopy(event)
        for event in e
        if not (event.get("kind") == "assessment_submission" and event.get("fact") == "F:Z0:P_ITEM_2")
    ]
    c.append(case("map_item_evidence", "two_maps_no_crossproduct", deepcopy(d), missing_second))
    cross_owner = deepcopy(e)
    next(event for event in cross_owner if event.get("kind") == "assessment" and event.get("activation") == "Z0")[
        "subject_artifact"
    ] = "MI0v1"
    c.append(case("map_item_evidence", "two_maps_cross_owner", deepcopy(d), cross_owner))

    def map_version_events() -> list[Obj]:
        events = map_item_events(2)
        events = cast(list[Obj], _replace_refs(events, {"MI1v1": "MI0v2"}))
        second_artifact = next(
            value for value in events if value.get("kind") == "artifact" and value.get("ref") == "MI0v2"
        )
        second_artifact.update({"key": "MI0", "version": 2})
        collection = next(value for value in events if value.get("kind") == "collection_value")
        obj(arr(collection["items"])[1]).update({"key": 0, "version": 2})
        second_item = next(
            value for value in events if value.get("kind") == "provenance" and value.get("key") == "MAPITEM:1"
        )
        second_item.update({"item_key": 0, "item_version": 2})
        return events

    c.append(case("lineage", "map_two_versions", map_declaration(2), map_version_events()))
    e = map_version_events()
    next(value for value in e if value.get("kind") == "provenance" and value.get("key") == "MAPITEM:1")["expander"] = (
        "OTHER"
    )
    c.append(case("lineage", "expander_crossover", map_declaration(2), e))
    e = map_version_events()
    next(value for value in e if value.get("kind") == "provenance" and value.get("key") == "MAPITEM:1")["target"] = "B"
    c.append(case("lineage", "map_target_crossover", map_declaration(2), e))
    e = map_version_events()
    next(value for value in e if value.get("kind") == "artifact" and value.get("ref") == "MI0v2")["invocation"] = "I1"
    c.append(case("lineage", "map_invocation_crossover", map_declaration(2), e))

    e = bound_events("N0", "D0", "BOUND:N0:p")
    next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "BOUND:N0:p")["binding_artifact"] = "XAv0"
    c.append(
        case(
            "provenance",
            "bound_invocation_ref_substitution",
            binding_declaration(("D0", "N0", "p", "A")),
            e,
        )
    )
    e = bound_events("N0", "D0", "BOUND:N0:p")
    obj(next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "BOUND:N0:p")["binding_artifact"])[
        "version"
    ] = 0
    c.append(case("provenance", "bound_invalid_version", binding_declaration(("D0", "N0", "p", "A")), e))
    e = bound_events("N0", "D0", "BOUND:N0:p")
    obj(next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "BOUND:N0:p")["binding_artifact"])[
        "key"
    ] = 999
    c.append(
        case(
            "provenance",
            "bound_unretained_reference",
            binding_declaration(("D0", "N0", "p", "A")),
            e,
        )
    )
    e = bound_events("N0", "D0", "BOUND:N0:p")
    obj(arr(next(x for x in e if x.get("kind") == "binding_receipt")["artifacts"])[0])["source"] = "OTHER@1"
    c.append(
        case(
            "provenance",
            "bound_foreign_receipt_source",
            binding_declaration(("D0", "N0", "p", "A")),
            e,
        )
    )

    e = initial_collection_events()
    next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "INITIAL")["parents"] = ["BOUND:D1:0"]
    c.append(case("provenance", "initial_missing_bound_parent", initial_declaration, e))
    e = initial_collection_events()
    next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "INITIAL")["declaration"] = "D0"
    c.append(case("provenance", "initial_wrong_declaration", initial_declaration, e))
    e = initial_collection_events()
    obj(next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "BOUND:D1:1")["binding_artifact"])[
        "key"
    ] = 999
    c.append(case("provenance", "initial_unretained_bound_parent", initial_declaration, e))

    for name, field, value in (
        ("map_wrong_member", "member", "MAP"),
        ("map_wrong_port", "port", "not_item_input"),
        ("map_negative_item_key", "item_key", -1),
        ("map_zero_item_version", "item_version", 0),
        ("map_wrong_expander", "expander", "M0"),
    ):
        e = map_item_events()
        next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "MAPITEM:0")[field] = value
        c.append(case("provenance", name, declaration(), e))
    e = map_item_events()
    next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "MAPITEM:0")["parents"] = []
    c.append(case("provenance", "map_missing_collection_parent", declaration(), e))
    e = map_item_events()
    next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "MAPITEM:0")["parents"] = ["ROOT:A:context"]
    c.append(case("provenance", "map_wrong_collection_parent", declaration(), e))
    e = map_item_events()
    next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "MAPITEM:0")["parents"] = [
        "OP:MAP:A:members",
        "ROOT:A:context",
    ]
    c.append(case("provenance", "map_multiple_collection_parents", declaration(), e))
    e = map_item_events()
    obj(arr(next(x for x in e if x.get("kind") == "collection_value")["items"])[0])["key"] = 1
    c.append(case("provenance", "map_collection_identity_mismatch", declaration(), e))
    e = map_item_events()
    collection_value = next(x for x in e if x.get("kind") == "collection_value")
    collection_value["producer"] = "EVID:A"
    collection_value["artifact"] = "EAv0"
    next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "MAPITEM:0")["parents"] = ["EVID:A"]
    evidence_producer = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "EVID:A")
    e.remove(evidence_producer)
    e.insert(e.index(collection_value), evidence_producer)
    c.append(case("provenance", "map_unrelated_collection_producer", declaration(), e))
    for name, field, value in (
        ("map_wrong_membership_port", "membership_port", "other"),
        ("map_wrong_expansion_outcome", "outcome", "other"),
    ):
        d = declaration()
        obj(arr(d["map_inputs"])[0])[field] = value
        c.append(case("provenance", name, d, map_item_events()))
    e = map_item_events()
    producer_fact = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "OP:MAP:A:members")
    producer_fact["activation"] = "ROOT:A"
    c.append(case("provenance", "map_wrong_producer_activation", declaration(), e))
    e = map_item_events(2)
    first_item = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "MAPITEM:0")
    second_item = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "MAPITEM:1")
    first_item["item_key"], second_item["item_key"] = second_item["item_key"], first_item["item_key"]
    c.append(case("provenance", "map_swapped_member_items", declaration(), e))
    e = map_item_events()
    e.insert(1, artifact("MAv0", "B", "artifact"))
    item = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "MAPITEM:0")
    item["artifact"] = "MAv0"
    item["target"] = "B"
    item_port = next(
        x for x in e if x.get("kind") == "port" and x.get("activation") == "M0" and x.get("port") == "item"
    )
    item_port["artifact"] = "MAv0"
    item_port["target"] = "B"
    c.append(case("provenance", "map_wrong_target", declaration(targets=("A", "B")), e))

    e = deepcopy(base)
    out = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "OUT:A")
    idx = e.index(out)
    e[idx:idx] = [
        port("ROOT:A", "A", "decision_alias", "Av0", "decision"),
        provenance(
            "DECISION:ALIAS",
            "Av0",
            "A",
            ("ROOT:A:subject",),
            decision=True,
            source="operation_output",
            port_name="decision_alias",
            activation="ROOT:A",
        ),
        input_producer("ROOT:A", "A", "N", "decision_alias", "DECISION:ALIAS"),
    ]
    out["parents"] = ["DECISION:ALIAS", "ROOT:A:subject"]
    d = declaration()
    set_output_dependency(d, "N", "decision_alias", ("subject",), identity_input="subject")
    set_output_dependency(d, "N", "result", ("decision_alias", "subject"), identity_input="subject")
    c.append(case("roles", "candidate_decision_distinct_occurrence_alias", d, e))
    # record mutations and execution-only reconciliation
    c.append(
        case(
            "record", "open_execution_only", declaration(purpose="execution_only"), map_events(0, parent_terminal=False)
        )
    )
    for name, field, value in (("plan", "plan", "P1"), ("invocation", "invocation", "I1"), ("graph", "graph", "G1")):
        e = base_events(("A",))
        owner: Obj = {
            "factory": "EXEC:I0",
            "graph": "G0",
            "invocation": "I0",
            "kind": "result_owner",
            "plan": "P0",
        }
        owner[field] = value
        e.insert(0, owner)
        c.append(case("record", name, declaration(), e))
    # Review289 exact authentication and environment joins.
    e = [x for x in deepcopy(base) if not (x.get("kind") == "port" and x.get("port") == "context")]
    c.append(case("authentication", "missing_consumed_port_fact", declaration(), e))
    for name, kind, selector, field, value in (
        ("evidence_port_node", "port", "evidence", "node", "OTHER"),
        ("evidence_port_target", "port", "evidence", "target", "B"),
        ("entry_node", "entry", "ROOT:A", "node", "OTHER"),
        ("entry_target", "entry", "ROOT:A", "target", "B"),
        ("terminal_outcome", "terminal", "ROOT:A", "outcome", "other"),
        ("terminal_target", "terminal", "ROOT:A", "target", "B"),
    ):
        e = deepcopy(base)
        fact = next(
            x for x in e if x.get("kind") == kind and (x.get("port") == selector or x.get("activation") == selector)
        )
        fact[field] = value
        c.append(case("authentication", name, declaration(), e))
    for name, environment in (
        ("missing_absence_environment", {"absences": {}, "configurations": {"N": "c0"}, "state": {"read": 1}}),
        ("missing_configuration_environment", {"absences": {"Q0": 1}, "configurations": {}, "state": {"read": 1}}),
        ("missing_state_environment", {"absences": {"Q0": 1}, "configurations": {"N": "c0"}, "state": {}}),
    ):
        c.append(case("authentication", name, declaration(), mutate(base, "assessment", "environment", environment)))
    for name, field, value in (
        ("subject_port", "subject_port", "other"),
        ("consumed_port", "consumed_ports", ["other"]),
    ):
        d = declaration()
        obj(arr(d["requirements"])[0])[field] = value
        c.append(case("admission", f"requirement_{name}", d, [], "admission"))
    # Invalid recovery purpose, policy, association and latest authority.
    for name, failure, purpose in (
        ("permanent_retry", "permanent", "retry"),
        ("retryable_failover", "retryable", "failover"),
        ("malformed_retry", "malformed", "retry"),
    ):
        c.append(case("request", name, declaration(), request_history(failure, purpose)))
    e = request_history("retryable", "retry")
    next(x for x in e if x.get("kind") == "request_attempt" and x.get("request") == "R1")["policy"] = "OTHER"
    c.append(case("request", "changed_policy", declaration(), e))
    e = request_history("retryable", "retry")
    next(x for x in e if x.get("kind") == "request_attempt" and x.get("request") == "R1")["associations"] = ["B"]
    c.append(case("request", "changed_association", declaration(), e))
    e = request_history("retryable", "retry")
    idx = next(i for i, x in enumerate(e) if x.get("kind") == "revision")
    e[idx:idx] = [
        {
            "associations": ["A"],
            "kind": "request_attempt",
            "policy": "P",
            "predecessor": "R1",
            "purpose": "retry",
            "request": "R2",
        },
        {"condition": "failure", "failure": "retryable", "kind": "request_terminal", "request": "R2"},
        {"kind": "settlement", "remote_stopped": True, "request": "R2", "usage": 1},
    ]
    c.append(case("request", "latest_failure_authority", declaration(), e))
    e = request_history("lost", None, False)
    c.append(case("request", "lost_known_usage", declaration(), e))
    e = request_history(None, None, True)
    c.append(case("request", "success_unknown_usage", declaration(), e))
    e = request_history(None, None)
    next(x for x in e if x.get("kind") == "request_attempt")["associations"] = ["B"]
    c.append(case("request", "foreign_association", declaration(), e))
    # Cleanup owner and localization.
    e = deepcopy(base)
    idx = next(i for i, x in enumerate(e) if x.get("kind") == "revision")
    e[idx:idx] = [
        {"kind": "cleanup_association", "owner": "sdk", "purpose": "verification", "resource": "Q0", "targets": ["A"]},
        {"disposition": "left_open", "kind": "cleanup", "resource": "Q0"},
    ]
    c.append(case("cleanup", "sdk_left_open", declaration(), e))
    e = base_events(("A", "C"))
    idx = next(i for i, x in enumerate(e) if x.get("kind") == "revision")
    e[idx:idx] = [
        {"kind": "cleanup_association", "owner": "sdk", "purpose": "verification", "resource": "QC", "targets": ["C"]},
        {"disposition": "close_failed", "kind": "cleanup", "resource": "QC"},
    ]
    c.append(case("cleanup", "unrelated_c", declaration(targets=("A", "C")), e))
    e = deepcopy(base)
    idx = next(i for i, x in enumerate(e) if x.get("kind") == "revision")
    e[idx:idx] = [
        {"kind": "cleanup_association", "owner": "sdk", "purpose": "verification", "resource": "QX", "targets": ["B"]},
        {"disposition": "close_failed", "kind": "cleanup", "resource": "QX"},
    ]
    c.append(case("cleanup", "foreign_target", declaration(), e))
    # Expansion outcomes and restored open/terminal categories.
    e = map_events(0)
    next(x for x in e if x.get("kind") == "membership" and x.get("parent") == "MAP")["expansion_outcome"] = "other"
    c.append(case("membership", "wrong_expansion_outcome", declaration(), e))
    for name, closed, nested in (("open", False, False), ("nested_open", False, True)):
        e = map_events(1, nested=nested)
        memberships = [x for x in e if x.get("kind") == "membership"]
        memberships[-1]["closed"] = closed
        memberships[-1]["status"] = "closed" if closed else "open"
        if not closed:
            root_membership = next(x for x in memberships if x.get("parent") is None)
            root_membership["closed"] = False
            root_membership["status"] = "open"
        c.append(case("membership", name, declaration(), e))
    for category in ("blocked", "cancelled", "lost", "inconsistent"):
        e = map_events(0)
        map_entry = next(x for x in e if x.get("kind") == "entry" and x.get("activation") == "MAP")
        map_entry["state_category"] = category
        map_entry["state_outcome"] = None
        map_terminal = next(x for x in e if x.get("kind") == "terminal" and x.get("activation") == "MAP")
        map_terminal["category"] = category
        map_terminal["outcome"] = None
        map_terminal["reasons"] = [
            {
                "blocked": "prerequisite",
                "cancelled": "cancel_requested",
                "inconsistent": "contradictory",
                "lost": "transport_lost",
            }[category]
        ]
        membership = next(x for x in e if x.get("kind") == "membership" and x.get("parent") == "MAP")
        membership["status"] = "failed"
        membership["expansion_outcome"] = None
        c.append(case("membership", f"terminal_{category}", declaration(), e))
    e = map_events(0)
    next(x for x in e if x.get("kind") == "membership" and x.get("parent") == "MAP")["target"] = "B"
    c.append(case("membership", "foreign_target", declaration(), e))
    # Adopted structural-accounting terminal kind, attempt and category rules.
    c.append(case("structural", "valid_container", declaration(), map_events(0)))
    e = deepcopy(base)
    next(x for x in e if x.get("kind") == "terminal")["structural"] = 1
    c.append(case("structural", "non_bool_flag", declaration(), e))
    e = deepcopy(base)
    next(x for x in e if x.get("kind") == "terminal")["reasons"] = ["unexpected"]
    c.append(case("structural", "success_with_reason", declaration(), e))
    e = map_events(0)
    next(x for x in e if x.get("kind") == "terminal" and x.get("activation") == "MAP")["attempt"] = "TASK:MAP"
    c.append(case("structural", "container_attempt_forbidden", declaration(), e))
    e = deepcopy(base)
    root_terminal = next(x for x in e if x.get("kind") == "terminal" and x.get("activation") == "ROOT:A")
    root_terminal["structural"] = True
    root_terminal["attempt"] = None
    c.append(case("structural", "structural_on_operation", declaration(), e))
    e = map_events(0)
    map_terminal = next(x for x in e if x.get("kind") == "terminal" and x.get("activation") == "MAP")
    map_terminal["structural"] = False
    map_terminal["attempt"] = "TASK:MAP"
    c.append(case("structural", "operation_on_container", declaration(), e))
    e = deepcopy(base)
    next(x for x in e if x.get("kind") == "terminal" and x.get("activation") == "ROOT:A")["attempt"] = None
    c.append(case("structural", "operation_success_missing_attempt", declaration(), e))
    for name, category in (("failed", "failure"), ("cancelled", "cancelled"), ("lost", "lost")):
        e = deepcopy(base)
        root_entry = next(x for x in e if x.get("kind") == "entry" and x.get("activation") == "ROOT:A")
        root_entry["state_category"] = category
        root_entry["state_outcome"] = None
        root_terminal = next(x for x in e if x.get("kind") == "terminal" and x.get("activation") == "ROOT:A")
        root_terminal["attempt"] = None
        root_terminal["category"] = category
        root_terminal["outcome"] = None
        root_terminal["reasons"] = [
            {"failure": "execution_failed", "cancelled": "cancel_requested", "lost": "transport_lost"}[category]
        ]
        c.append(case("structural", f"operation_{name}_missing_attempt", declaration(), e))
    e = map_events(0)
    map_terminal = next(x for x in e if x.get("kind") == "terminal" and x.get("activation") == "MAP")
    map_terminal["category"] = "failure"
    map_terminal["outcome"] = None
    map_terminal["reasons"] = ["execution_failed"]
    c.append(case("structural", "category_mismatch", declaration(), e))
    e = map_events(0)
    map_entry = next(x for x in e if x.get("kind") == "entry" and x.get("activation") == "MAP")
    map_entry["state_category"] = "failure"
    map_entry["state_outcome"] = None
    map_terminal = next(x for x in e if x.get("kind") == "terminal" and x.get("activation") == "MAP")
    map_terminal["category"] = "failure"
    map_terminal["outcome"] = None
    map_terminal["reasons"] = []
    c.append(case("structural", "failure_missing_reason", declaration(), e))
    e = map_events(0)
    next(x for x in e if x.get("kind") == "entry" and x.get("activation") == "MAP")["node_kind"] = "operation"
    c.append(case("structural", "node_kind_mismatch", declaration(), e))
    for status, category in (("failed", "failure"), ("overflow", "inconsistent")):
        e = map_events(1)
        map_entry = next(x for x in e if x.get("kind") == "entry" and x.get("activation") == "MAP")
        map_entry["state_category"] = category
        map_entry["state_outcome"] = None
        map_terminal = next(x for x in e if x.get("kind") == "terminal" and x.get("activation") == "MAP")
        map_terminal["category"] = category
        map_terminal["outcome"] = None
        map_terminal["reasons"] = ["execution_failed" if status == "failed" else "contradictory"]
        membership = next(x for x in e if x.get("kind") == "membership" and x.get("parent") == "MAP")
        membership["status"] = status
        membership["expansion_outcome"] = None
        insertion = next(i for i, x in enumerate(e) if x.get("kind") == "revision")
        e.insert(
            insertion,
            {"activation": "M1", "kind": "reservation", "parent": "MAP", "selected": True, "target": "A"},
        )
        c.append(case("structural", f"{status}_actual_members", declaration(), e))
    # Decision ownership and producer defects.
    for name, field, value in (("missing_artifact", "artifact", "MISSINGv0"), ("foreign_target", "target", "B")):
        direct = next(x for x in c if x["case_id"] == "decisions/direct")
        e = cast(list[Obj], deepcopy(direct["events"]))
        decision_fact = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "DEC")
        decision_fact[field] = value
        c.append(case("decisions", name, cast(Obj, deepcopy(direct["declaration"])), e))
    e = [x for x in deepcopy(base) if not (x.get("kind") == "provenance" and x.get("key") == "OUT:A")]
    c.append(case("provenance", "missing_producer", declaration(), e))
    e = deepcopy(base)
    out = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "OUT:A")
    e.insert(e.index(out) + 1, deepcopy(out))
    c.append(case("provenance", "duplicate_producer", declaration(), e))
    # True execution-only empty path.
    d, e = execution_only_shape()
    c.append(case("release", "execution_only_empty", d, e))
    # Restored selective, duplicate-key, attribution, final-output and shared-decision families.
    for name, deps_value in (
        ("a_only", {"context": "XAv0"}),
        ("b_only", {"decision": "YAv0"}),
        ("a_b", {"context": "XAv0", "decision": "YAv0"}),
    ):
        e = base_events(("A",))
        a = next(x for x in e if x.get("kind") == "assessment")
        if "decision" in deps_value:
            e.insert(1, artifact("YAv0", "A", "artifact"))
            e.insert(e.index(a), port("ROOT:A", "A", "decision", "YAv0", "artifact"))
            retain_root_input(e, "A", "decision", "YAv0")
        a["consumed"] = deps_value
        if "decision" in deps_value:
            seal = next(i for i, x in enumerate(e) if x.get("kind") == "seal_revision")
            e.insert(seal, {"collection": "artifacts", "key": "YA", "kind": "revision", "value": 1})
            e.insert(1, artifact("YAv1", "A", "artifact"))
        d = declaration()
        wire_evidence_consumed(d, e, tuple(deps_value))
        obj(arr(d["requirements"])[0])["consumed_ports"] = list(deps_value)
        c.append(case("selective", name, d, e))
    # Complete dependency products: XA models A, YA models B/decision.
    for dependency, consumed in (
        ("a_only", {"context": "XAv0"}),
        ("b_only", {"decision": "YAv0"}),
        ("a_b", {"context": "XAv0", "decision": "YAv0"}),
        ("decision", {"decision": "YAv0"}),
    ):
        for mode in ("current", "stale", "unknown"):
            e = base_events(("A",))
            assessment_fact = next(x for x in e if x.get("kind") == "assessment")
            e.insert(1, artifact("YAv0", "A", "decision"))
            e.insert(e.index(assessment_fact), port("ROOT:A", "A", "decision", "YAv0", "decision"))
            if "decision" in consumed:
                retain_root_input(e, "A", "decision", "YAv0")
            assessment_fact["consumed"] = consumed
            seal = next(i for i, x in enumerate(e) if x.get("kind") == "seal_revision")
            e.insert(seal, {"collection": "artifacts", "key": "YA", "kind": "revision", "value": 0})
            changed_key = "XA" if dependency == "a_only" else "YA"
            revision = next(
                x
                for x in e
                if x.get("kind") == "revision" and x.get("collection") == "artifacts" and x.get("key") == changed_key
            )
            if mode == "stale":
                revision["value"] = 1
                e.insert(1, artifact("XAv1" if changed_key == "XA" else "YAv1", "A", "artifact"))
            elif mode == "unknown":
                e.remove(revision)
            d = declaration()
            wire_evidence_consumed(d, e, tuple(consumed))
            obj(arr(d["requirements"])[0])["consumed_ports"] = list(consumed)
            c.append(case("selective", f"{dependency}_{mode}", d, e))
    e = deepcopy(base)
    e.insert(1, artifact("A1v0", "A", "candidate"))
    result_port = next(x for x in e if x.get("kind") == "port" and x.get("port") == "result")
    result_port["artifact"] = "A1v0"
    f = next(x for x in e if x.get("kind") == "final")
    f["candidate"] = "A1v0"
    p = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "OUT:A")
    p["artifact"] = "A1v0"
    seal = next(i for i, x in enumerate(e) if x.get("kind") == "seal_revision")
    e.insert(seal, {"collection": "artifacts", "key": "A1", "kind": "revision", "value": 0})
    d = declaration()
    set_output_dependency(d, "N", "result", ("subject", "context"))
    c.append(case("selective", "intermediate_a0_final_a1", d, e))
    for collection, key in (("artifacts", "A"), ("absences", "Q0"), ("configurations", "N"), ("state", "read")):
        e = deepcopy(base)
        source = next(
            x for x in e if x.get("kind") == "revision" and x.get("collection") == collection and x.get("key") == key
        )
        duplicate = deepcopy(source)
        if collection == "state":
            duplicate["value"] = cast(int, source["value"]) + 1
        e.insert(e.index(source) + 1, duplicate)
        c.append(case("bounds", f"duplicate_{collection}_key", declaration(), e))
    for name, field, value in (
        ("foreign_target", "target", "B"),
        ("foreign_evidence", "evidence_artifact", "EBv0"),
        ("foreign_subject", "subject_artifact", "Bv0"),
    ):
        c.append(case("assessment", name, declaration(), mutate(base, "assessment", field, value)))
    e = deepcopy(base)
    next(x for x in e if x.get("kind") == "final")["target"] = "B"
    c.append(case("final_output", "wrong_target", declaration(), e))
    e = deepcopy(base)
    next(x for x in e if x.get("kind") == "final")["outcome"] = "other"
    c.append(case("final_output", "wrong_outcome", declaration(), e))
    e = base_events(("A", "B"))
    e.insert(1, artifact("Dv0", "shared", "decision"))
    first_out = next(i for i, x in enumerate(e) if x.get("kind") == "provenance" and x.get("key") == "OUT:A")
    e[first_out:first_out] = [
        port("ROOT:A", "A", "shared_decision", "Dv0", "decision"),
        port("ROOT:B", "B", "shared_decision", "Dv0", "decision"),
        provenance("SHARED:DEC:A", "Dv0", "A", decision=True, port_name="shared_decision", activation="ROOT:A"),
        provenance("SHARED:DEC:B", "Dv0", "B", decision=True, port_name="shared_decision", activation="ROOT:B"),
        input_producer("ROOT:A", "A", "N", "shared_decision", "SHARED:DEC:A"),
        input_producer("ROOT:B", "B", "N", "shared_decision", "SHARED:DEC:B"),
    ]
    for target in ("A", "B"):
        next(x for x in e if x.get("kind") == "provenance" and x.get("key") == f"OUT:{target}")["parents"] = [
            f"SHARED:DEC:{target}",
            f"ROOT:{target}:subject",
        ]
    seal = next(i for i, x in enumerate(e) if x.get("kind") == "seal_revision")
    e.insert(seal, {"collection": "artifacts", "key": "D", "kind": "revision", "value": 0})
    d = declaration(targets=("A", "B"))
    set_output_dependency(d, "N", "shared_decision", ())
    set_output_dependency(d, "N", "result", ("shared_decision", "subject"), identity_input="subject")
    c.append(case("decisions", "shared_deduplicated", d, e))
    e = deepcopy(base)
    sites = (("D0", "N0", "p", "A"), ("D2", "N1", "p", "A"))
    e.insert(0, binding_receipt(sites, (("D0", 0, 1), ("D2", 0, 1))))
    out = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "OUT:A")
    idx = e.index(out)
    e[idx:idx] = [
        provenance(
            "BOUND:0",
            "XAv0",
            "A",
            source="bound_input",
            node="N0",
            port_name="p",
            binding_artifact=binding_ref("D0"),
        ),
        provenance(
            "BOUND:1",
            "XAv0",
            "A",
            source="bound_input",
            node="N1",
            port_name="p",
            binding_artifact=binding_ref("D2"),
        ),
        input_producer("ROOT:A", "A", "N", "bound_0", "BOUND:0"),
        input_producer("ROOT:A", "A", "N", "bound_1", "BOUND:1"),
        port("ROOT:A", "A", "bound_0", "XAv0", "artifact"),
        port("ROOT:A", "A", "bound_1", "XAv0", "artifact"),
    ]
    out["parents"] = ["BOUND:0", "BOUND:1", "ROOT:A:subject"]
    c.append(case("provenance", "multiple_bound_artifacts", binding_declaration(*sites), e))
    # Exact final producer occurrence and wrong-kind coverage.
    for name, kind, selector, field, value in (
        ("producer_node", "port", "result", "node", "OTHER"),
        ("producer_activation", "port", "result", "activation", "OTHER:A"),
        ("final_node", "final", None, "node", "OTHER"),
    ):
        e = deepcopy(base)
        fact = next(x for x in e if x.get("kind") == kind and (selector is None or x.get("port") == selector))
        fact[field] = value
        c.append(case("final_output", name, declaration(), e))
    c.append(
        case(
            "assessment",
            "wrong_kind_coverage",
            declaration(),
            mutate(base, "assessment", "coverage", ["OTHER"]),
        )
    )
    # Evidence-output revision changes are independent of the consumed set.
    for name, mode in (("evidence_output_removed", "removed"), ("evidence_output_replaced", "replaced")):
        e = deepcopy(base)
        rev = next(x for x in e if x.get("kind") == "revision" and x.get("key") == "EA")
        if mode == "removed":
            e.remove(rev)
        else:
            rev["value"] = 1
            e.insert(1, artifact("EAv1", "A", "artifact"))
        c.append(case("validity", name, declaration(), e))
    e = deepcopy(base)
    next(
        x for x in e if x.get("kind") == "revision" and x.get("collection") == "configurations" and x.get("key") == "N"
    )["value"] = 0
    c.append(case("validity", "configuration_type_changed", declaration(), e))
    # Structured provenance identities and bound-root ownership.
    e = base_events(("A", "B"))
    e.insert(1, artifact("Sv0", "shared", "artifact"))
    out_a = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "OUT:A")
    out_b = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "OUT:B")
    first = min(e.index(out_a), e.index(out_b))
    e[first:first] = [
        port("ROOT:A", "A", "shared_context", "Sv0", "artifact"),
        port("ROOT:B", "B", "shared_context", "Sv0", "artifact"),
        provenance(
            "SHARED:A",
            "Sv0",
            "A",
            source="root_input",
            port_name="shared_context",
            activation="ROOT:A",
        ),
        provenance(
            "SHARED:B",
            "Sv0",
            "B",
            source="root_input",
            port_name="shared_context",
            activation="ROOT:B",
        ),
        input_producer("ROOT:A", "A", "N", "shared_context", "SHARED:A"),
        input_producer("ROOT:B", "B", "N", "shared_context", "SHARED:B"),
    ]
    out_a["parents"] = ["SHARED:A", "ROOT:A:subject"]
    out_b["parents"] = ["SHARED:B", "ROOT:B:subject"]
    seal = next(i for i, x in enumerate(e) if x.get("kind") == "seal_revision")
    e.insert(seal, {"collection": "artifacts", "key": "S", "kind": "revision", "value": 0})
    d = declaration(targets=("A", "B"))
    set_output_dependency(d, "N", "result", ("shared_context", "subject"), identity_input="subject")
    c.append(case("provenance", "shared_context_identity", d, e))
    for name, declaration_id, target, d in (
        (
            "mismatched_bound_root",
            "D2",
            "A",
            binding_declaration(("D0", "N0", "p", "A")),
        ),
        (
            "foreign_bound_target",
            "D0",
            "B",
            binding_declaration(("D0", "N0", "p", "A"), targets=("A", "B")),
        ),
    ):
        e = deepcopy(base)
        e.insert(0, binding_receipt((("D0", "N0", "p", "A"),), (("D0", 0, 1),)))
        out = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "OUT:A")
        idx = e.index(out)
        e.insert(
            idx,
            provenance(
                "BOUND:BAD",
                "XAv0",
                target,
                source="bound_input",
                node="N0",
                port_name="p",
                binding_artifact=binding_ref(declaration_id),
            ),
        )
        out["parents"] = ["BOUND:BAD"]
        c.append(case("provenance", name, d, e))
    # A wholly withheld target contributes no decision unless a released target
    # has that exact decision occurrence in its own producer ancestry.
    for name, linked in (("withheld_unrelated", False), ("withheld_linked", True)):
        e = base_events(("A", "B"))
        next(x for x in e if x.get("kind") == "assessment" and x.get("target") == "A")["finding"] = "unsatisfied"
        e.insert(1, artifact("Dv0", "shared", "decision"))
        out_a = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "OUT:A")
        out_b = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "OUT:B")
        insertion = min(e.index(out_a), e.index(out_b))
        e[insertion:insertion] = [
            port("ROOT:A", "A", "withheld_decision", "Dv0", "decision"),
            port("ROOT:B", "B", "withheld_decision", "Dv0", "decision"),
            provenance(
                "WITHHELD:DEC:A",
                "Dv0",
                "A",
                decision=True,
                port_name="withheld_decision",
                activation="ROOT:A",
            ),
            provenance(
                "WITHHELD:DEC:B",
                "Dv0",
                "B",
                decision=True,
                port_name="withheld_decision",
                activation="ROOT:B",
            ),
        ]
        d = declaration(targets=("A", "B"))
        set_output_dependency(d, "N", "withheld_decision", ())
        if linked:
            e.insert(e.index(out_a), input_producer("ROOT:A", "A", "N", "withheld_decision", "WITHHELD:DEC:A"))
            e.insert(e.index(out_b), input_producer("ROOT:B", "B", "N", "withheld_decision", "WITHHELD:DEC:B"))
            out_a["parents"] = ["WITHHELD:DEC:A", "ROOT:A:subject"]
            out_b["parents"] = ["WITHHELD:DEC:B", "ROOT:B:subject"]
            set_output_dependency(d, "N", "result", ("withheld_decision", "subject"), identity_input="subject")
        seal = next(i for i, x in enumerate(e) if x.get("kind") == "seal_revision")
        e.insert(seal, {"collection": "artifacts", "key": "D", "kind": "revision", "value": 0})
        c.append(case("decisions", name, d, e))
    # Review293 root/parent partition and revision ownership negatives.
    e = [x for x in deepcopy(base) if not (x.get("kind") == "membership" and x.get("parent") is None)]
    c.append(case("membership", "missing_root", declaration(), e))
    e = deepcopy(base)
    idx = next(i for i, x in enumerate(e) if x.get("kind") == "revision")
    e[idx:idx] = [entry("ORPHAN", "A"), terminal("ORPHAN", "A")]
    c.append(case("membership", "orphan_entry", declaration(), e))
    e = deepcopy(base)
    idx = next(i for i, x in enumerate(e) if x.get("kind") == "revision")
    e.insert(idx, terminal("STRAY", "A"))
    c.append(case("membership", "stray_terminal", declaration(), e))
    e = map_events(1)
    next(x for x in e if x.get("kind") == "entry" and x.get("activation") == "M0")["target"] = "B"
    next(x for x in e if x.get("kind") == "terminal" and x.get("activation") == "M0")["target"] = "B"
    c.append(case("membership", "cross_target_member", declaration(targets=("A", "B")), e))
    revision_mutants: tuple[tuple[str, Obj], ...] = (
        ("invented_artifact", {"collection": "artifacts", "key": "INVENTED", "kind": "revision", "value": 0}),
        ("invented_absence", {"collection": "absences", "key": "Q1", "kind": "revision", "value": 1}),
        ("invented_configuration", {"collection": "configurations", "key": "OTHER", "kind": "revision", "value": "c0"}),
        ("invented_state", {"collection": "state", "key": "write", "kind": "revision", "value": 1}),
    )
    for name, event in revision_mutants:
        e = deepcopy(base)
        e.insert(next(i for i, x in enumerate(e) if x.get("kind") == "seal_revision"), event)
        c.append(case("revision", name, declaration(), e))
    # Occurrence roles, same-artifact aliasing and provenance ownership.
    e = deepcopy(base)
    next(x for x in e if x.get("kind") == "artifact" and x.get("ref") == "EAv0")["role"] = "anything"
    c.append(case("roles", "global_role_ignored", declaration(), e))
    e = deepcopy(base)
    next(x for x in e if x.get("kind") == "port" and x.get("port") == "evidence")["role"] = "candidate"
    c.append(case("roles", "distinct_evidence_wrong_role", declaration(), e))
    d = declaration()
    obj(arr(d["productions"])[0])["subject_source"] = "input"
    d["root_outputs"] = [
        {"input_port": None, "port": "evidence", "source_node": "N", "source_port": "evidence", "target": "A"}
    ]
    e = deepcopy(base)
    next(x for x in e if x.get("kind") == "port" and x.get("port") == "evidence")["role"] = "candidate"
    f = next(x for x in e if x.get("kind") == "final")
    f.update({"candidate": "EAv0", "port": "evidence", "producer": "EVID:A"})
    c.append(case("roles", "root_bound_distinct_evidence_candidate", d, e))
    e = deepcopy(base)
    next(x for x in e if x.get("kind") == "port" and x.get("port") == "subject")["role"] = "decision"
    c.append(case("roles", "subject_decision_collision", declaration(), e))
    e = deepcopy(base)
    next(x for x in e if x.get("kind") == "port" and x.get("port") == "context")["role"] = "other"
    c.append(case("roles", "consumed_unsupported_role", declaration(), e))
    d = declaration()
    set_production_consumed(d, ("subject", "context"))
    obj(arr(d["requirements"])[0])["consumed_ports"] = ["subject", "context"]
    set_output_dependency(d, "N", "evidence", ("subject", "context"), identity_input="subject")
    e = deepcopy(base)
    next(x for x in e if x.get("kind") == "port" and x.get("port") == "evidence")["artifact"] = "Av0"
    evidence_provenance = next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "EVID:A")
    evidence_provenance.update({"artifact": "Av0", "parents": ["ROOT:A:subject", "ROOT:A:context"]})
    e = [x for x in e if not (x.get("kind") == "artifact" and x.get("ref") == "EAv0")]
    e = [
        x
        for x in e
        if not (x.get("kind") == "revision" and x.get("collection") == "artifacts" and x.get("key") == "EA")
    ]
    a = next(x for x in e if x.get("kind") == "assessment")
    a["evidence_artifact"] = "Av0"
    a["consumed"] = {"context": "XAv0", "subject": "Av0"}
    c.append(case("roles", "candidate_evidence_alias", d, e))
    for name, ref in (("missing_parent_artifact", "MISSINGv0"), ("mismatched_source_artifact", "Av0")):
        e = deepcopy(base)
        next(x for x in e if x.get("kind") == "provenance" and x.get("key") == "ROOT:A:context")["artifact"] = ref
        c.append(case("provenance", name, declaration(), e))
    # Unrequired verified evidence remains canonical but cannot affect release.
    d = declaration()
    p2 = deepcopy(obj(arr(d["productions"])[0]))
    p2.update(
        {"consumed_ports": ["decision"], "evidence_port": "evidence2", "promise": "P2", "subject_port": "subject2"}
    )
    arr(d["productions"]).append(p2)
    set_output_dependency(d, "N", "evidence2", ("decision",))
    e = deepcopy(base)
    insertion = next(i for i, x in enumerate(e) if x.get("kind") == "revision")
    e[insertion:insertion] = [
        artifact("E2v0", "A", "evidence"),
        artifact("Dv0", "A", "decision"),
        port("ROOT:A", "A", "evidence2", "E2v0", "evidence"),
        port("ROOT:A", "A", "subject2", "Av0", "candidate"),
        port("ROOT:A", "A", "decision", "Dv0", "decision"),
        assessment(
            "A",
            promise="P2",
            evidence_port="evidence2",
            subject_port="subject2",
            evidence_artifact="E2v0",
            consumed={"decision": "Dv0"},
        ),
        assessment_submission("A", "P2"),
        provenance(
            "EVID2:A",
            "E2v0",
            "A",
            ("ROOT:A:decision",),
            port_name="evidence2",
            activation="ROOT:A",
        ),
    ]
    retain_root_input(e, "A", "decision", "Dv0")
    seal = next(i for i, x in enumerate(e) if x.get("kind") == "seal_revision")
    e[seal:seal] = [
        {"collection": "artifacts", "key": "E2", "kind": "revision", "value": 0},
        {"collection": "artifacts", "key": "D", "kind": "revision", "value": 0},
    ]
    c.append(case("selection", "unrelated_assessment", d, e))
    # Request association identity is a set, independent of serialized order.
    d = declaration(targets=("A", "B"))
    e = base_events(("A", "B"))
    idx = next(i for i, x in enumerate(e) if x.get("kind") == "revision")
    e[idx:idx] = [
        {
            "associations": ["A", "B"],
            "kind": "request_attempt",
            "policy": "P",
            "predecessor": None,
            "purpose": "initial",
            "request": "R0",
        },
        {"condition": "failure", "failure": "retryable", "kind": "request_terminal", "request": "R0"},
        {"kind": "settlement", "remote_stopped": True, "request": "R0", "usage": 1},
        {
            "associations": ["B", "A"],
            "kind": "request_attempt",
            "policy": "P",
            "predecessor": "R0",
            "purpose": "retry",
            "request": "R1",
        },
        {"condition": "result", "failure": None, "kind": "request_terminal", "request": "R1"},
        {"kind": "settlement", "remote_stopped": True, "request": "R1", "usage": 1},
    ]
    c.append(case("request", "association_order_invariant", d, e))
    # bounds exact/one-over
    c.append(case("bounds", "exact", declaration(), base))
    for name, exact, over, e in (
        ("ports", {"max_port_facts": 4}, {"max_port_facts": 3}, base),
        ("consumed", {"max_consumed_per_assessment": 1}, {"max_consumed_per_assessment": 0}, base),
        ("edges", {"max_provenance_edges": 3}, {"max_provenance_edges": 2}, base),
        ("submissions", {"max_submissions": 1}, {"max_submissions": 0}, base),
        ("revisions", {"max_revision_entries": 5}, {"max_revision_entries": 4}, base),
        ("absences", {"max_absence_revisions": 1}, {"max_absence_revisions": 0}, base),
    ):
        c.append(case("bounds", f"{name}_exact", declaration(limit_changes=exact), deepcopy(e)))
        c.append(case("bounds", f"{name}_one_over", declaration(limit_changes=over), deepcopy(e)))
    for name, exact, over in (
        ("productions", {"max_productions": 1}, {"max_productions": 0}),
        ("coverage", {"max_coverage_atoms": 2}, {"max_coverage_atoms": 1}),
        ("verified", {"max_verified_evidence": 1}, {"max_verified_evidence": 0}),
    ):
        c.append(case("bounds", f"{name}_exact", declaration(limit_changes=exact), deepcopy(base)))
        boundary = "admission" if name in ("productions", "coverage") else "qualification"
        c.append(
            case(
                "bounds",
                f"{name}_one_over",
                declaration(limit_changes=over),
                [] if boundary == "admission" else deepcopy(base),
                boundary,
            )
        )
    direct_case = next(x for x in c if x["case_id"] == "decisions/direct")
    decision_events = cast(list[Obj], deepcopy(direct_case["events"]))
    decision_declaration = cast(Obj, deepcopy(direct_case["declaration"]))
    obj(decision_declaration["limits"])["max_required_decisions"] = 1
    c.append(
        case(
            "bounds",
            "required_decisions_exact",
            decision_declaration,
            decision_events,
        )
    )
    c.append(
        case(
            "bounds",
            "required_decisions_one_over",
            dict(decision_declaration, limits=dict(obj(decision_declaration["limits"]), max_required_decisions=0)),
            deepcopy(decision_events),
        )
    )
    fixed_events = base_events(("A", "B", "C"))
    c.append(
        case(
            "bounds",
            "fixed_point_exact",
            declaration(targets=("A", "B", "C"), limit_changes={"max_fixed_point_steps": 3}),
            fixed_events,
        )
    )
    c.append(
        case(
            "bounds",
            "fixed_point_one_over",
            declaration(targets=("A", "B", "C"), limit_changes={"max_fixed_point_steps": 2}),
            [],
            "admission",
        )
    )

    # Adopted same-lineage materialized-version products.
    def initial_version_seed(*, limit_changes: Mapping[str, int] | None = None) -> tuple[Obj, list[Obj]]:
        d = declaration(limit_changes=limit_changes)
        events = base_events(("A",))
        old = next(value for value in events if value.get("kind") == "artifact" and value.get("ref") == "XAv0")
        events.insert(events.index(old) + 1, artifact("XAv1", "A", "artifact"))
        next(
            value
            for value in events
            if value.get("kind") == "revision" and value.get("collection") == "artifacts" and value.get("key") == "XA"
        )["value"] = 1
        return d, events

    for name in (
        "initial_two_versions",
        "two_selected_current_versions",
        "invented_version",
        "declaration_crossover",
        "target_crossover",
        "invocation_crossover",
        "rejected_publication_rollback",
        "latest_older_selected",
        "latest_missing_binding_request",
        "latest_foreign_binding_association",
        "latest_missing_binding_settlement",
        "latest_missing_binding_cleanup",
        "latest_foreign_cleanup_target",
    ):
        d, e = initial_version_seed()
        c.append(case("lineage", name, d, e))
    for name, limit_changes in (
        ("artifacts_exact", {"max_artifacts": 4}),
        ("artifacts_one_over", {"max_artifacts": 3}),
        ("bytes_exact", {"max_artifact_bytes": 4}),
        ("bytes_one_over", {"max_artifact_bytes": 3}),
        ("provenance_exact", {"max_provenance_edges": 3}),
        ("provenance_one_over", {"max_provenance_edges": 2}),
    ):
        d, e = initial_version_seed(limit_changes=limit_changes)
        c.append(case("lineage", name, d, e))
    d = binding_declaration(("D0", "N0", "p", "A"))
    e = base_events(("A",))
    receipt = binding_receipt((("D0", "N0", "p", "A"),), (("D0", 0, 1), ("D0", 0, 1)))
    e.insert(0, receipt)
    c.append(case("lineage", "source_duplicate_pair", d, e))
    # admission mutants
    for name, d in (
        ("duplicate_production", declaration()),
        ("fixed_point", declaration(limit_changes={"max_fixed_point_steps": 0})),
        ("foreign_dependency", declaration(deps=(("A", "B"),))),
        ("production_one_over", declaration(limit_changes={"max_productions": 0})),
        ("overlapping_atomic", declaration(targets=("A", "B", "C"), atomic=(("A", "B"), ("B", "C")))),
        ("unsupported_outcome", declaration()),
        ("missing_promise", declaration()),
    ):
        if name == "duplicate_production":
            arr(d["productions"]).append(deepcopy(arr(d["productions"])[0]))
        elif name == "unsupported_outcome":
            obj(arr(d["requirements"])[0])["outcome"] = "other"
        elif name == "missing_promise":
            obj(arr(d["requirements"])[0])["meaning"] = "other"
        c.append(case("admission", name, d, [], "admission"))
    d = declaration(targets=("A", "B", "C"), atomic=(("A", "B"),))
    d["atomic"] = [["A", "B"]]
    c.append(case("admission", "incomplete_atomic_partition", d, [], "admission"))
    d = binding_declaration(("D0", "N0", "p", "A"))
    obj(arr(d["binding_inputs"])[0])["node"] = "OTHER"
    c.append(case("admission", "binding_foreign_node", d, [], "admission"))
    d = declaration()
    arr(d["initial_collections"]).append({"declaration": "OTHER", "node": "N0", "port": "items", "target": "A"})
    c.append(case("admission", "initial_collection_without_binding", d, [], "admission"))
    d = declaration()
    arr(d["map_inputs"]).append(deepcopy(arr(d["map_inputs"])[0]))
    c.append(case("admission", "duplicate_map_input", d, [], "admission"))
    d = declaration()
    set_output_dependency(d, "N", "evidence", ("subject", "context"))
    c.append(case("admission", "evidence_dependency_consumed_mismatch", d, [], "admission"))
    d = binding_declaration(("D0", "N0", "p", "A"), ("D1", "N0", "items", "A"))
    obj(arr(d["binding_inputs"])[1])["declaration"] = "D0"
    c.append(case("admission", "duplicate_binding_declaration", d, [], "admission"))
    c.append(case("admission", "empty_atomic_group", declaration(atomic=((),)), [], "admission"))
    # true commutation alternate: retained artifact A and admitted absence Q0.
    d = declaration()
    primary = deepcopy(base)
    alternate = deepcopy(primary)
    a = next(i for i, x in enumerate(alternate) if x.get("kind") == "revision" and x.get("key") == "A")
    q = next(i for i, x in enumerate(alternate) if x.get("kind") == "revision" and x.get("key") == "Q0")
    alternate[a], alternate[q] = alternate[q], alternate[a]
    c.append(case("commutation", "independent_revisions", d, primary, alternates=(alternate,)))
    return tuple(c)


CASES = generate_cases()
FAMILY_COUNTS = Counter(cast(str, x["family"]) for x in CASES)


def trace_count(xs: Sequence[Obj]) -> int:
    return sum(1 + len(arr(x["traces"])) for x in xs)


def event_count(xs: Sequence[Obj]) -> int:
    return sum(len(arr(x["events"])) + sum(len(arr(obj(t)["events"])) for t in arr(x["traces"])) for x in xs)


def manifest(xs: Sequence[Obj]) -> Obj:
    data = canonical_bytes(xs)
    return {
        "case_count": len(xs),
        "contract_sha256": CONTRACT_SHA256,
        "corpus_path": CORPUS_PATH,
        "corpus_sha256": hashlib.sha256(data).hexdigest(),
        "event_count": event_count(xs),
        "family_counts": dict(sorted(FAMILY_COUNTS.items())),
        "generator_version": GENERATOR_VERSION,
        "map_item_evidence_contract_sha256": MAP_ITEM_EVIDENCE_CONTRACT_SHA256,
        "materialized_version_contract_sha256": MATERIALIZED_VERSION_CONTRACT_SHA256,
        "self_test_version": SELF_TEST_VERSION,
        "structural_contract_sha256": STRUCTURAL_CONTRACT_SHA256,
        "trace_count": trace_count(xs),
        "v10_ids_sha256": V10_IDS_SHA256,
    }


if __name__ == "__main__":
    here = Path(__file__).parent
    (here / "qualification_v1_cases.json").write_bytes(canonical_bytes(CASES))
    (here / "qualification_v1_manifest.json").write_text(json.dumps(manifest(CASES), indent=2, sort_keys=True) + "\n")
