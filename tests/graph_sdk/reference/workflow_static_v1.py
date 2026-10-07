# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent finite reference for static workflow composition semantics.

This module owns the neutral expected behavior used by later conformance tests.
Its exhaustive claims are bounded to the eight declared families: it does not
cover arbitrary labels or artifact types, graphs above three expanded nodes,
all semantic cross-products, dynamic activation, maps, joins, loops, provider
behavior, or runtime evidence.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import platform
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Literal, TypeAlias, cast

Json: TypeAlias = str | int | bool | None | list["Json"] | dict[str, "Json"]
Object: TypeAlias = dict[str, Json]
ValidationCode: TypeAlias = Literal[
    "invalid_type",
    "invalid_value",
    "limit_exceeded",
    "foreign_owner",
    "duplicate",
    "missing",
    "invalid_range",
    "overlap",
    "cycle",
    "contradictory",
]

CONTRACT_SHA256 = "5b66dbedad875b95d93d79372dae2ee25e045403b794d40001fdb3bbc6471fa6"
CORPUS_PATH = "tests/graph_sdk/reference/workflow_static_v1_cases.json"
GENERATOR_VERSION = "workflow-static-v1-generator-1"
SELF_TEST_VERSION = "workflow-static-v1-self-test-1"
ALPHABET = (
    "new_workflow",
    "declare_interface",
    "declare_node",
    "declare_subgraph",
    "bind_input",
    "bind_output",
    "add_sequence",
    "add_choice",
    "declare_protection",
    "admit",
    "substitute",
)
FAMILY_IDS = (
    "sequence_topology",
    "typed_ports_bindings",
    "outcome_choice",
    "declared_subgraph",
    "substitution",
    "protection_eligibility",
    "lineage_projection",
    "limits_ownership",
)
RULE_IDS = (
    "distinct_node_declarations",
    "bindings_distinct_destinations",
    "distinct_sequence_edges",
    "choices_disjoint_selectors_and_members",
    "protection_distinct_outcome_meaning_subject",
)
FAMILIES = (
    "topology",
    "ports",
    "choice",
    "subgraph",
    "substitution",
    "protection",
    "lineage",
    "limits_ownership",
)
RESOURCE_FIELDS = (
    "max_activations",
    "max_model_requests",
    "max_input_bytes",
    "max_output_bytes",
)
LIMIT_FIELDS = (
    "max_nodes",
    "max_bindings",
    "max_sequence_edges",
    "max_choices",
    "max_branch_members",
    "max_subgraph_depth",
    "max_choice_states",
)
SEMANTIC_ORDERS = {
    "input_port": ("i0", "i1"),
    "output_port": ("o0", "o1"),
    "context": ("context", "context_alt"),
    "evidence": ("assessment", "assessment_alt"),
    "state": ("state", "state_alt"),
    "model": ("model", "model_alt"),
    "coverage": ("field", "field_alt", "source", "absence"),
}


def _canonical(value: Json) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sorted(values: Iterable[Json]) -> list[Json]:
    return sorted(values, key=_canonical)


def _ref(label: str, owner: str = "W0") -> Object:
    return {"label": label, "owner": owner}


def _ceiling(a: int, m: int, i: int, o: int) -> Object:
    return {
        "max_activations": a,
        "max_input_bytes": i,
        "max_model_requests": m,
        "max_output_bytes": o,
    }


def _outcome(
    name: str,
    category: str,
    produced: Iterable[str],
    ceiling: Object,
    *,
    context: Iterable[Object] = (),
    evidence: Iterable[Object] = (),
    state: Iterable[Object] = (),
    models: Iterable[Object] = (),
) -> Object:
    return {
        "category": category,
        "ceiling": deepcopy(ceiling),
        "context": _sorted(deepcopy(list(context))),
        "evidence": _sorted(deepcopy(list(evidence))),
        "model_requirements": _sorted(deepcopy(list(models))),
        "name": name,
        "produced_ports": sorted(produced),
        "state_effects": _sorted(deepcopy(list(state))),
    }


def _dependency(output: str, inputs: Iterable[str], identity: str | None) -> Object:
    return {"identity_input": identity, "inputs": sorted(inputs), "output": output}


def _operation(
    name: str,
    inputs: Sequence[tuple[str, str]],
    outputs: Sequence[tuple[str, str]],
    dependencies: Sequence[Object],
    outcomes: Sequence[Object],
) -> Object:
    return {
        "inputs": [{"artifact_type": artifact, "name": port} for port, artifact in inputs],
        "name": name,
        "outcomes": deepcopy(list(outcomes)),
        "output_dependencies": deepcopy(list(dependencies)),
        "outputs": [{"artifact_type": artifact, "name": port} for port, artifact in outputs],
    }


def _context(port: str, meaning: str) -> Object:
    return {"capture": "whole_artifact", "meaning": meaning, "port": port}


def _evidence(name: str, meaning: str, subject: str, consumed: Iterable[str], coverage: Iterable[str]) -> Object:
    return {
        "consumed_ports": sorted(consumed),
        "coverage": _sorted({"kind": "source_view" if item == "source" else item, "name": item} for item in coverage),
        "meaning": meaning,
        "name": name,
        "subject_port": subject,
    }


def _z(n: int = 1) -> Object:
    ceiling = _ceiling(n, 0, 0, 0)
    return _operation(
        "Z" if n == 1 else f"Z[{n}]",
        (),
        (),
        (),
        (_outcome("ok", "success", (), ceiling), _outcome("fail", "failure", (), ceiling)),
    )


def _p(*, two_inputs: bool = False) -> Object:
    contexts = [_context("i0", "context")]
    inputs = [("i0", "A0@1")]
    if two_inputs:
        inputs.append(("i1", "A0@1"))
        contexts.append(_context("i1", "context_alt"))
    state: list[Object] = [{"kind": "read", "name": "state"}]
    models: list[Object] = [{"capability": "model", "revision": 1}]
    return _operation(
        "P2I" if two_inputs else "P",
        inputs,
        (("o0", "A0@1"),),
        (_dependency("o0", ("i0",), None),),
        (
            _outcome(
                "ok",
                "success",
                ("o0",),
                _ceiling(1, 1, 8, 8),
                context=contexts,
                evidence=(_evidence("assessment", "assessment", "o0", ("i0",), ("field", "source")),),
                state=state,
                models=models,
            ),
            _outcome(
                "fail",
                "failure",
                (),
                _ceiling(1, 1, 8, 0),
                context=contexts,
                state=state,
                models=models,
            ),
        ),
    )


def _q(*, factor: int = 1) -> Object:
    ceiling = _ceiling(factor, 0, 8 * factor, 8 * factor)
    return _operation(
        "Q" if factor == 1 else "Q2",
        (("i0", "A0@1"),),
        (("o0", "A0@1"),),
        (_dependency("o0", ("i0",), "i0"),),
        (_outcome("ok", "success", ("o0",), ceiling), _outcome("fail", "failure", ("o0",), ceiling)),
    )


def _l(index: int) -> Object:
    meaning = "context" if index == 0 else "context_alt"
    assessment = "assessment" if index == 0 else "assessment_alt"
    result = _q()
    result["name"] = f"L{index}"
    for outcome in cast(list[Object], result["outcomes"]):
        outcome["context"] = [_context("i0", meaning)]
        outcome["evidence"] = [_evidence(assessment, assessment, "o0", ("i0",), ("field", "source"))]
    return result


def _l01() -> Object:
    result = _q(factor=2)
    result["name"] = "L01"
    for outcome in cast(list[Object], result["outcomes"]):
        outcome["context"] = _sorted((_context("i0", "context"), _context("i0", "context_alt")))
        outcome["evidence"] = _sorted(
            (
                _evidence("assessment", "assessment", "o0", ("i0",), ("field", "source")),
                _evidence("assessment_alt", "assessment_alt", "o0", ("i0",), ("field", "source")),
            )
        )
    return result


def _limits(
    *,
    nodes: int = 3,
    bindings: int = 16,
    edges: int = 8,
    choices: int = 2,
    members: int = 4,
    depth: int = 2,
    states: int = 4,
) -> Object:
    return {
        "max_bindings": bindings,
        "max_branch_members": members,
        "max_choice_states": states,
        "max_choices": choices,
        "max_nodes": nodes,
        "max_sequence_edges": edges,
        "max_subgraph_depth": depth,
    }


def _node(label: str, operation: Object, *, owner: str = "W0", body: Object | None = None) -> Object:
    value: Object = {
        "id": _ref(label, owner),
        "kind": "subgraph" if body is not None else "operation",
        "operation": deepcopy(operation),
    }
    if body is not None:
        value["body"] = deepcopy(body)
    return value


def _workflow(interface: Object, nodes: Sequence[Object], *, limits: Object | None = None) -> Object:
    return {
        "choices": [],
        "input_bindings": [],
        "interface": deepcopy(interface),
        "limits": deepcopy(limits or _limits()),
        "nodes": deepcopy(list(nodes)),
        "outcome_bindings": [],
        "output_bindings": [],
        "protection": [],
        "sequence": [],
        "workflow": "W0",
    }


def _input(source: str, node: str, port: str, *, owner: str = "W0") -> Object:
    source_value: Object = {"kind": "workflow_input", "port": source}
    if source.startswith("N"):
        source_value = {"kind": "node_output", "node": _ref(source), "port": "o0"}
    return {"destination": {"node": _ref(node, owner), "port": port}, "source": source_value}


def _output(node: str, source_port: str, destination: str, *, owner: str = "W0") -> Object:
    return {
        "destination": {"port": destination},
        "source": {"kind": "node_output", "node": _ref(node, owner), "port": source_port},
    }


def _outcome_binding(node: str, outcome: str, destination: str, *, owner: str = "W0") -> Object:
    return {"destination": {"outcome": destination}, "source": {"node": _ref(node, owner), "outcome": outcome}}


def _one(operation: Object) -> Object:
    declaration = _workflow(operation, (_node("N0", operation),))
    inputs = cast(list[Object], operation["inputs"])
    outputs = cast(list[Object], operation["outputs"])
    declaration["input_bindings"] = [_input(cast(str, port["name"]), "N0", cast(str, port["name"])) for port in inputs]
    declaration["output_bindings"] = [
        _output("N0", cast(str, port["name"]), cast(str, port["name"])) for port in outputs
    ]
    declaration["outcome_bindings"] = [_outcome_binding("N0", "ok", "ok"), _outcome_binding("N0", "fail", "fail")]
    return declaration


def _pipe(*, lineage: bool = False) -> Object:
    first, second = (_l(0), _l(1)) if lineage else (_q(), _q())
    interface = _l01() if lineage else _q(factor=2)
    declaration = _workflow(interface, (_node("N0", first), _node("N1", second)))
    declaration["input_bindings"] = [_input("i0", "N0", "i0"), _input("N0", "N1", "i0")]
    declaration["output_bindings"] = [_output("N1", "o0", "o0")]
    declaration["outcome_bindings"] = [_outcome_binding("N1", "ok", "ok"), _outcome_binding("N1", "fail", "fail")]
    declaration["sequence"] = [{"after": _ref("N1"), "before": _ref("N0")}]
    return declaration


def _choice() -> Object:
    declaration = _workflow(_z(2), tuple(_node(f"N{i}", _z()) for i in range(3)))
    declaration["sequence"] = [
        {"after": _ref("N1"), "before": _ref("N0")},
        {"after": _ref("N2"), "before": _ref("N0")},
    ]
    declaration["choices"] = [
        {
            "branches": [
                {"members": [_ref("N1")], "outcomes": ["ok"]},
                {"members": [_ref("N2")], "outcomes": ["fail"]},
            ],
            "selector": _ref("N0"),
        }
    ]
    declaration["outcome_bindings"] = [
        _outcome_binding("N1", "ok", "ok"),
        _outcome_binding("N1", "fail", "fail"),
        _outcome_binding("N2", "ok", "ok"),
        _outcome_binding("N2", "fail", "fail"),
    ]
    return declaration


def _wrap(body: Object) -> Object:
    interface = deepcopy(_obj(body["interface"]))
    declaration = _one(interface)
    cast(list[Object], declaration["nodes"])[0] = _node("N0", interface, body=_reowner_workflow(body, "W1"))
    return declaration


def _reowner_workflow(declaration: Object, owner: str) -> Object:
    result = deepcopy(declaration)
    old_owner = cast(str, result["workflow"])
    result["workflow"] = owner

    def replace_owner(value: Json) -> None:
        if isinstance(value, list):
            for item in value:
                replace_owner(item)
        elif isinstance(value, dict):
            if value.get("owner") == old_owner and "label" in value:
                value["owner"] = owner
            for item in value.values():
                replace_owner(item)

    replace_owner(result)
    return result


def _obj(value: Json) -> Object:
    if not isinstance(value, dict):
        raise ValueError("invalid neutral object")
    return cast(Object, value)


def _list(value: Json) -> list[Json]:
    if not isinstance(value, list):
        raise ValueError("invalid neutral array")
    return value


def _nodes(declaration: Object) -> list[Object]:
    return [_obj(value) for value in _list(declaration["nodes"])]


def _id_label(value: Json) -> str:
    return cast(str, _obj(value)["label"])


def _operation_ports(operation: Object, key: str) -> dict[str, str]:
    return {cast(str, _obj(item)["name"]): cast(str, _obj(item)["artifact_type"]) for item in _list(operation[key])}


def _operation_outcomes(operation: Object) -> dict[str, Object]:
    return {cast(str, _obj(item)["name"]): _obj(item) for item in _list(operation["outcomes"])}


def _metrics(declaration: Object) -> dict[str, int]:
    nodes = _nodes(declaration)
    expanded = len(nodes)
    depth = 1
    for node in nodes:
        if node["kind"] == "subgraph":
            child = _metrics(_obj(node["body"]))
            expanded += child["nodes"]
            depth = max(depth, child["subgraph_depth"] + 1)
    choices = [_obj(value) for value in _list(declaration["choices"])]
    branch_members = sum(
        len(_list(branch["members"])) for choice in choices for branch in map(_obj, _list(choice["branches"]))
    )
    choice_states = 1
    node_map = {_id_label(node["id"]): node for node in nodes}
    for choice in choices:
        selector = node_map.get(_id_label(choice["selector"]))
        if selector is not None:
            choice_states *= len(_list(_obj(selector["operation"])["outcomes"]))
    bindings = sum(len(_list(declaration[key])) for key in ("input_bindings", "output_bindings", "outcome_bindings"))
    return {
        "bindings": bindings,
        "branch_members": branch_members,
        "choice_states": choice_states,
        "choices": len(choices),
        "nodes": expanded,
        "sequence_edges": len(_list(declaration["sequence"])),
        "subgraph_depth": depth,
    }


def _cycle_and_sinks(declaration: Object) -> tuple[bool, int | None]:
    labels = {_id_label(node["id"]) for node in _nodes(declaration)}
    edges = [
        (_id_label(_obj(edge)["before"]), _id_label(_obj(edge)["after"])) for edge in _list(declaration["sequence"])
    ]
    incoming = {label: 0 for label in labels}
    outgoing = {label: 0 for label in labels}
    for before, after in edges:
        if before in labels and after in labels:
            incoming[after] += 1
            outgoing[before] += 1
    queue = [label for label, count in incoming.items() if count == 0]
    seen = 0
    while queue:
        current = queue.pop()
        seen += 1
        for before, after in edges:
            if before == current and after in incoming:
                incoming[after] -= 1
                if incoming[after] == 0:
                    queue.append(after)
    if seen != len(labels):
        return True, None
    return False, sum(count == 0 for count in outgoing.values())


def _foreign(declaration: Object) -> bool:
    refs: list[Object] = []
    for node in _nodes(declaration):
        refs.append(_obj(node["id"]))
    for key in ("input_bindings", "output_bindings", "outcome_bindings"):
        for binding in map(_obj, _list(declaration[key])):
            for side in ("source", "destination"):
                endpoint = _obj(binding[side])
                if "node" in endpoint:
                    refs.append(_obj(endpoint["node"]))
    for edge in map(_obj, _list(declaration["sequence"])):
        refs.extend((_obj(edge["before"]), _obj(edge["after"])))
    for choice in map(_obj, _list(declaration["choices"])):
        refs.append(_obj(choice["selector"]))
        for branch in map(_obj, _list(choice["branches"])):
            refs.extend(map(_obj, _list(branch["members"])))
    return any(ref["owner"] != declaration["workflow"] for ref in refs)


def _duplicates(declaration: Object) -> bool:
    nodes = [_canonical(node["id"]) for node in _nodes(declaration)]
    if len(nodes) != len(set(nodes)):
        return True
    for operation in [_obj(declaration["interface"]), *[_obj(node["operation"]) for node in _nodes(declaration)]]:
        for key in ("inputs", "outputs", "outcomes"):
            names = [cast(str, _obj(value)["name"]) for value in _list(operation[key])]
            if len(names) != len(set(names)):
                return True
        dependencies = [cast(str, _obj(value)["output"]) for value in _list(operation["output_dependencies"])]
        if len(dependencies) != len(set(dependencies)):
            return True
    for key in ("input_bindings", "output_bindings"):
        destinations = [_canonical(_obj(value)["destination"]) for value in _list(declaration[key])]
        if len(destinations) != len(set(destinations)):
            return True
    sources = [_canonical(_obj(value)["source"]) for value in _list(declaration["outcome_bindings"])]
    edges = [_canonical(value) for value in _list(declaration["sequence"])]
    selectors = [_canonical(_obj(value)["selector"]) for value in _list(declaration["choices"])]
    return len(sources) != len(set(sources)) or len(edges) != len(set(edges)) or len(selectors) != len(set(selectors))


def _missing(declaration: Object) -> bool:
    nodes = {_id_label(node["id"]): node for node in _nodes(declaration)}
    interface = _obj(declaration["interface"])
    workflow_inputs = _operation_ports(interface, "inputs")
    workflow_outputs = _operation_ports(interface, "outputs")
    for operation in [interface, *[_obj(node["operation"]) for node in nodes.values()]]:
        inputs = _operation_ports(operation, "inputs")
        outputs = _operation_ports(operation, "outputs")
        dependencies = [_obj(value) for value in _list(operation["output_dependencies"])]
        if {cast(str, value["output"]) for value in dependencies} != set(outputs):
            return True
        for dependency in dependencies:
            if dependency["output"] not in outputs or not set(cast(list[str], dependency["inputs"])) <= set(inputs):
                return True
        produced = {
            port
            for outcome in _operation_outcomes(operation).values()
            for port in cast(list[str], outcome["produced_ports"])
        }
        if not set(outputs) <= produced:
            return True
    input_destinations: set[tuple[str, str]] = set()
    for binding in map(_obj, _list(declaration["input_bindings"])):
        destination = _obj(binding["destination"])
        node = nodes.get(_id_label(destination["node"]))
        if node is None or destination["port"] not in _operation_ports(_obj(node["operation"]), "inputs"):
            return True
        input_destinations.add((_id_label(destination["node"]), cast(str, destination["port"])))
        source = _obj(binding["source"])
        if source["kind"] == "workflow_input":
            if source["port"] not in workflow_inputs:
                return True
        else:
            source_node = nodes.get(_id_label(source["node"]))
            if source_node is None or source["port"] not in _operation_ports(_obj(source_node["operation"]), "outputs"):
                return True
    required_inputs = {
        (label, port) for label, node in nodes.items() for port in _operation_ports(_obj(node["operation"]), "inputs")
    }
    if input_destinations != required_inputs:
        return True
    output_destinations: set[str] = set()
    for binding in map(_obj, _list(declaration["output_bindings"])):
        destination = cast(str, _obj(binding["destination"])["port"])
        source = _obj(binding["source"])
        node = nodes.get(_id_label(source["node"]))
        if (
            destination not in workflow_outputs
            or node is None
            or source["port"] not in _operation_ports(_obj(node["operation"]), "outputs")
        ):
            return True
        output_destinations.add(destination)
    if output_destinations != set(workflow_outputs):
        return True
    for binding in map(_obj, _list(declaration["outcome_bindings"])):
        source = _obj(binding["source"])
        destination = cast(str, _obj(binding["destination"])["outcome"])
        node = nodes.get(_id_label(source["node"]))
        if (
            node is None
            or source["outcome"] not in _operation_outcomes(_obj(node["operation"]))
            or destination not in _operation_outcomes(interface)
        ):
            return True
    for choice in map(_obj, _list(declaration["choices"])):
        selector = nodes.get(_id_label(choice["selector"]))
        if selector is None:
            return True
        outcomes = set(_operation_outcomes(_obj(selector["operation"])))
        for branch in map(_obj, _list(choice["branches"])):
            if not set(cast(list[str], branch["outcomes"])) <= outcomes:
                return True
            if any(_id_label(member) not in nodes for member in _list(branch["members"])):
                return True
    cyclic, sinks = _cycle_and_sinks(declaration)
    choices = _list(declaration["choices"])
    if not cyclic and not choices and sinks != 1:
        return True
    if not cyclic:
        sink_labels = {
            label
            for label in nodes
            if not any(_id_label(_obj(edge)["before"]) == label for edge in _list(declaration["sequence"]))
        }
        choice_members = {
            _id_label(member)
            for choice in map(_obj, _list(declaration["choices"]))
            for branch in map(_obj, _list(choice["branches"]))
            for member in _list(branch["members"])
        }
        candidates = sink_labels | choice_members
        bound = {
            (_id_label(_obj(_obj(value)["source"])["node"]), cast(str, _obj(_obj(value)["source"])["outcome"]))
            for value in _list(declaration["outcome_bindings"])
        }
        for label in candidates:
            if label in sink_labels or label in choice_members:
                for outcome in _operation_outcomes(_obj(nodes[label]["operation"])):
                    if (label, outcome) not in bound and label in sink_labels:
                        return True
    return False


def _overlap(declaration: Object) -> bool:
    all_members: set[str] = set()
    for choice in map(_obj, _list(declaration["choices"])):
        outcomes: set[str] = set()
        members: set[str] = set()
        for branch in map(_obj, _list(choice["branches"])):
            branch_outcomes = set(cast(list[str], branch["outcomes"]))
            branch_members = {_id_label(value) for value in _list(branch["members"])}
            if outcomes & branch_outcomes or members & branch_members:
                return True
            outcomes |= branch_outcomes
            members |= branch_members
        if all_members & members:
            return True
        all_members |= members
    return False


def _invalid_choice(declaration: Object) -> bool:
    for choice in map(_obj, _list(declaration["choices"])):
        selector = _id_label(choice["selector"])
        for branch in map(_obj, _list(choice["branches"])):
            if (
                not _list(branch["outcomes"])
                or not _list(branch["members"])
                or selector in {_id_label(item) for item in _list(branch["members"])}
            ):
                return True
    return False


def _incompatible_operation(expected: Object, actual: Object, *, allow_narrow: bool) -> bool:
    for key in ("inputs", "outputs", "output_dependencies"):
        if _canonical(expected[key]) != _canonical(actual[key]):
            return True
    expected_outcomes = _operation_outcomes(expected)
    actual_outcomes = _operation_outcomes(actual)
    if set(expected_outcomes) != set(actual_outcomes):
        return True
    for name, left in expected_outcomes.items():
        right = actual_outcomes[name]
        for key in ("category", "produced_ports", "context", "evidence", "state_effects", "model_requirements"):
            if _canonical(left[key]) != _canonical(right[key]):
                return True
        left_ceiling, right_ceiling = _obj(left["ceiling"]), _obj(right["ceiling"])
        for resource in RESOURCE_FIELDS:
            if (allow_narrow and cast(int, right_ceiling[resource]) > cast(int, left_ceiling[resource])) or (
                not allow_narrow and right_ceiling[resource] != left_ceiling[resource]
            ):
                return True
    return False


def _identity_endpoints(declaration: Object, node_label: str, port: str, *, forward: bool) -> set[str]:
    edges: dict[tuple[str, str], set[tuple[str, str]]] = {}
    nodes = {_id_label(node["id"]): node for node in _nodes(declaration)}
    for binding in map(_obj, _list(declaration["input_bindings"])):
        destination = _obj(binding["destination"])
        dst = (_id_label(destination["node"]), cast(str, destination["port"]))
        source = _obj(binding["source"])
        src = (
            ("W_IN", cast(str, source["port"]))
            if source["kind"] == "workflow_input"
            else (_id_label(source["node"]), cast(str, source["port"]))
        )
        edges.setdefault(src, set()).add(dst)
    for binding in map(_obj, _list(declaration["output_bindings"])):
        source = _obj(binding["source"])
        src = (_id_label(source["node"]), cast(str, source["port"]))
        dst = ("W_OUT", cast(str, _obj(binding["destination"])["port"]))
        edges.setdefault(src, set()).add(dst)
    for label, node in nodes.items():
        operation = _obj(node["operation"])
        for dependency in map(_obj, _list(operation["output_dependencies"])):
            identity = dependency["identity_input"]
            if identity is not None:
                edges.setdefault((label, cast(str, identity)), set()).add((label, cast(str, dependency["output"])))
    if not forward:
        reversed_edges: dict[tuple[str, str], set[tuple[str, str]]] = {}
        for source, destinations in edges.items():
            for destination in destinations:
                reversed_edges.setdefault(destination, set()).add(source)
        edges = reversed_edges
    start = (node_label, port)
    stack = [start]
    seen = {start}
    endpoints: set[str] = set()
    endpoint_kind = "W_OUT" if forward else "W_IN"
    while stack:
        current = stack.pop()
        if current[0] == endpoint_kind:
            endpoints.add(current[1])
        for adjacent in edges.get(current, set()):
            if adjacent not in seen:
                seen.add(adjacent)
                stack.append(adjacent)
    return endpoints


def _semantic_endpoint_code(declaration: Object) -> ValidationCode | None:
    for node in _nodes(declaration):
        label = _id_label(node["id"])
        operation = _obj(node["operation"])
        inputs = set(_operation_ports(operation, "inputs"))
        for outcome in _operation_outcomes(operation).values():
            for context in map(_obj, _list(outcome["context"])):
                endpoints = _identity_endpoints(declaration, label, cast(str, context["port"]), forward=False)
                if not endpoints:
                    return "missing"
                if len(endpoints) > 1:
                    return "contradictory"
            for evidence in map(_obj, _list(outcome["evidence"])):
                for consumed in cast(list[str], evidence["consumed_ports"]):
                    endpoints = _identity_endpoints(declaration, label, consumed, forward=False)
                    if not endpoints:
                        return "missing"
                    if len(endpoints) > 1:
                        return "contradictory"
                subject = cast(str, evidence["subject_port"])
                endpoints = _identity_endpoints(declaration, label, subject, forward=subject not in inputs)
                if not endpoints:
                    return "missing"
                if len(endpoints) > 1:
                    return "contradictory"
    return None


def _contradictory(declaration: Object) -> bool:
    nodes = {_id_label(node["id"]): node for node in _nodes(declaration)}
    interface = _obj(declaration["interface"])
    for binding in map(_obj, _list(declaration["input_bindings"])):
        destination = _obj(binding["destination"])
        destination_type = _operation_ports(_obj(nodes[_id_label(destination["node"])]["operation"]), "inputs")[
            cast(str, destination["port"])
        ]
        source = _obj(binding["source"])
        source_type = (
            _operation_ports(interface, "inputs")[cast(str, source["port"])]
            if source["kind"] == "workflow_input"
            else _operation_ports(_obj(nodes[_id_label(source["node"])]["operation"]), "outputs")[
                cast(str, source["port"])
            ]
        )
        if source_type != destination_type:
            return True
    for binding in map(_obj, _list(declaration["output_bindings"])):
        source = _obj(binding["source"])
        destination = _obj(binding["destination"])
        if (
            _operation_ports(_obj(nodes[_id_label(source["node"])]["operation"]), "outputs")[cast(str, source["port"])]
            != _operation_ports(interface, "outputs")[cast(str, destination["port"])]
        ):
            return True
    for operation in [interface, *[_obj(node["operation"]) for node in nodes.values()]]:
        inputs, outputs = _operation_ports(operation, "inputs"), _operation_ports(operation, "outputs")
        for dependency in map(_obj, _list(operation["output_dependencies"])):
            identity = dependency["identity_input"]
            if identity is not None and (
                identity not in cast(list[str], dependency["inputs"])
                or inputs[cast(str, identity)] != outputs[cast(str, dependency["output"])]
            ):
                return True
    closure = {
        (_id_label(_obj(edge)["before"]), _id_label(_obj(edge)["after"])) for edge in _list(declaration["sequence"])
    }
    changed = True
    while changed:
        changed = False
        for a, b in tuple(closure):
            for c, d in tuple(closure):
                if b == c and (a, d) not in closure:
                    closure.add((a, d))
                    changed = True
    for choice in map(_obj, _list(declaration["choices"])):
        selector = _id_label(choice["selector"])
        if any(
            (selector, _id_label(member)) not in closure
            for branch in map(_obj, _list(choice["branches"]))
            for member in _list(branch["members"])
        ):
            return True
    for node in nodes.values():
        if node["kind"] == "subgraph" and _incompatible_operation(
            _obj(node["operation"]), _obj(_obj(node["body"])["interface"]), allow_narrow=False
        ):
            return True
    return False


def _normalized(declaration: Object) -> Object:
    metrics = _metrics(declaration)
    return {
        "choices": _sorted(deepcopy(_list(declaration["choices"]))),
        "expanded_node_count": metrics["nodes"],
        "input_bindings": _sorted(deepcopy(_list(declaration["input_bindings"]))),
        "interface": deepcopy(declaration["interface"]),
        "nodes": _sorted(deepcopy(_list(declaration["nodes"]))),
        "outcome_bindings": _sorted(deepcopy(_list(declaration["outcome_bindings"]))),
        "output_bindings": _sorted(deepcopy(_list(declaration["output_bindings"]))),
        "protection_requirements": _sorted(deepcopy(_list(declaration["protection"]))),
        "sequence": _sorted(deepcopy(_list(declaration["sequence"]))),
    }


def _protection(declaration: Object) -> tuple[list[str], list[Json]]:
    outcomes = _operation_outcomes(_obj(declaration["interface"]))
    unmet: list[Json] = []
    eligible: set[str] = set()
    requirements = list(map(_obj, _list(declaration["protection"])))
    for requirement in requirements:
        matched = False
        outcome = outcomes.get(cast(str, requirement["outcome"]))
        if outcome is not None:
            for promise in map(_obj, _list(outcome["evidence"])):
                if (
                    promise["meaning"] == requirement["meaning"]
                    and promise["subject_port"] == requirement["subject_port"]
                    and set(cast(list[str], requirement["consumed_ports"]))
                    <= set(cast(list[str], promise["consumed_ports"]))
                    and {_canonical(value) for value in _list(requirement["coverage"])}
                    <= {_canonical(value) for value in _list(promise["coverage"])}
                ):
                    matched = True
        if matched:
            eligible.add(cast(str, requirement["outcome"]))
        else:
            unmet.append(deepcopy(requirement))
    return sorted(eligible), _sorted(unmet)


def _admission_code(declaration: Object) -> ValidationCode | None:
    metrics = _metrics(declaration)
    limits = _obj(declaration["limits"])
    metric_map = {
        "max_nodes": "nodes",
        "max_bindings": "bindings",
        "max_sequence_edges": "sequence_edges",
        "max_choices": "choices",
        "max_branch_members": "branch_members",
        "max_subgraph_depth": "subgraph_depth",
        "max_choice_states": "choice_states",
    }
    if _invalid_choice(declaration):
        return "invalid_value"
    if any(metrics[metric] > cast(int, limits[limit]) for limit, metric in metric_map.items()):
        return "limit_exceeded"
    if _foreign(declaration):
        return "foreign_owner"
    if _duplicates(declaration):
        return "duplicate"
    if _missing(declaration):
        return "missing"
    if _overlap(declaration):
        return "overlap"
    cyclic, _ = _cycle_and_sinks(declaration)
    if cyclic:
        return "cycle"
    if _contradictory(declaration):
        return "contradictory"
    return _semantic_endpoint_code(declaration)


def _substituted(declaration: Object, target_node: Object, replacement: Object) -> Object:
    result = deepcopy(declaration)
    replacement_interface = _obj(replacement["interface"])
    replacement_node = _node(
        _id_label(target_node["id"]),
        replacement_interface,
        owner=cast(str, _obj(target_node["id"])["owner"]),
        body=replacement,
    )
    result["nodes"] = [
        replacement_node if _obj(node)["id"] == target_node["id"] else node for node in _list(result["nodes"])
    ]
    return result


def judge(declaration: Object, replacement: Object | None = None) -> Object:
    """Evaluate a neutral declaration without importing product implementation."""
    admitted = declaration
    code = _admission_code(declaration)
    if code is None and replacement is not None:
        target = _obj(declaration["substitution_target"])
        if target["owner"] != declaration["workflow"]:
            code = "foreign_owner"
        else:
            target_node = next((node for node in _nodes(declaration) if node["id"] == target), None)
            if target_node is None:
                code = "missing"
            elif _incompatible_operation(
                _obj(target_node["operation"]), _obj(replacement["interface"]), allow_narrow=True
            ):
                code = "contradictory"
            elif _admission_code(replacement) is not None:
                code = "contradictory"
            else:
                admitted = _substituted(declaration, target_node, replacement)
                code = _admission_code(admitted)
    topology: Json = None
    if declaration.get("family_marker") == "topology":
        cyclic, sinks = _cycle_and_sinks(declaration)
        topology = {"acyclic": not cyclic, "sink_count": sinks}
    if code is not None:
        return {
            "code": code,
            "normalized": None,
            "protection_eligible_outcomes": [],
            "status": "rejected",
            "topology": topology,
            "unmet_protection": [],
        }
    eligible, unmet = _protection(admitted)
    return {
        "code": None,
        "normalized": _normalized(admitted),
        "protection_eligible_outcomes": eligible,
        "status": "accepted",
        "topology": topology,
        "unmet_protection": unmet,
    }


def _events(declaration: Object, *, substitution: bool) -> list[Json]:
    events: list[Json] = [{"op": "new_workflow", "workflow": declaration["workflow"]}, {"op": "declare_interface"}]
    for node in _nodes(declaration):
        events.append({"node": node["id"], "op": "declare_subgraph" if node["kind"] == "subgraph" else "declare_node"})
    for binding in _list(declaration["input_bindings"]):
        events.append({"destination": _obj(binding)["destination"], "op": "bind_input"})
    for binding in _list(declaration["output_bindings"]):
        events.append({"destination": _obj(binding)["destination"], "op": "bind_output"})
    for binding in _list(declaration["outcome_bindings"]):
        events.append({"destination": _obj(binding)["destination"], "op": "bind_output"})
    for edge in _list(declaration["sequence"]):
        events.append({"edge": edge, "op": "add_sequence"})
    for choice in _list(declaration["choices"]):
        events.append({"choice": choice, "op": "add_choice"})
    for requirement in _list(declaration["protection"]):
        events.append({"op": "declare_protection", "requirement": requirement})
    events.append({"op": "admit"})
    if substitution:
        events.append({"op": "substitute", "target": declaration["substitution_target"]})
    return events


def independent(left: Object, right: Object) -> bool:
    """Return the exact conditional symmetric trace-independence relation."""
    left_op, right_op = left.get("op"), right.get("op")
    if left_op in {"declare_node", "declare_subgraph"} and right_op in {"declare_node", "declare_subgraph"}:
        return left.get("node") != right.get("node")
    if left_op in {"bind_input", "bind_output"} and right_op in {"bind_input", "bind_output"}:
        return left.get("destination") != right.get("destination")
    if left_op == right_op == "add_sequence":
        return left.get("edge") != right.get("edge")
    if left_op == right_op == "add_choice":
        left_choice, right_choice = _obj(cast(Json, left["choice"])), _obj(cast(Json, right["choice"]))
        left_members = {
            _canonical(member)
            for branch in map(_obj, _list(left_choice["branches"]))
            for member in _list(branch["members"])
        }
        right_members = {
            _canonical(member)
            for branch in map(_obj, _list(right_choice["branches"]))
            for member in _list(branch["members"])
        }
        left_nodes = left_members | {_canonical(left_choice["selector"])}
        right_nodes = right_members | {_canonical(right_choice["selector"])}
        return not left_nodes & right_nodes
    if left_op == right_op == "declare_protection":
        keys = ("outcome", "meaning", "subject_port")
        left_requirement, right_requirement = (
            _obj(cast(Json, left["requirement"])),
            _obj(cast(Json, right["requirement"])),
        )
        return tuple(left_requirement[key] for key in keys) != tuple(right_requirement[key] for key in keys)
    return False


def _rename_maps(declaration: Object, replacement: Object | None) -> dict[str, dict[str, str]]:
    labels = [_id_label(node["id"]) for node in _nodes(declaration)]
    if len(labels) == 1:
        nodes = {labels[0]: labels[0]}
    elif len(labels) == 2:
        nodes = {labels[0]: labels[1], labels[1]: labels[0]}
    else:
        ordered = sorted(labels)
        nodes = {label: ordered[(index + 1) % len(ordered)] for index, label in enumerate(ordered)}
    text = json.dumps((declaration, replacement), sort_keys=True)
    mappings = {"node": nodes}
    for role, order in SEMANTIC_ORDERS.items():
        present = [item for item in order if f'"{item}"' in text]
        mappings[role] = (
            {item: present[(index + 1) % len(present)] for index, item in enumerate(present)}
            if len(present) > 1
            else {item: item for item in present}
        )
    return mappings


def _rewrite_role(value: str, role: str, mappings: Mapping[str, Mapping[str, str]]) -> str:
    return mappings.get(role, {}).get(value, value)


def _rewrite(value: Json, mappings: Mapping[str, Mapping[str, str]], *, parent_key: str = "") -> Json:
    if isinstance(value, str):
        if parent_key == "label":
            return _rewrite_role(value, "node", mappings)
        if parent_key in {
            "port",
            "subject_port",
            "identity_input",
            "output",
            "inputs",
            "produced_ports",
            "consumed_ports",
        }:
            rewritten = _rewrite_role(value, "input_port", mappings)
            return _rewrite_role(rewritten, "output_port", mappings)
        if parent_key == "meaning":
            rewritten = _rewrite_role(value, "context", mappings)
            return _rewrite_role(rewritten, "evidence", mappings)
        if parent_key == "capability":
            return _rewrite_role(value, "model", mappings)
        return value
    if isinstance(value, list):
        return [_rewrite(item, mappings, parent_key=parent_key) for item in value]
    if isinstance(value, dict):
        rewritten = {key: _rewrite(item, mappings, parent_key=key) for key, item in value.items()}
        name = value.get("name")
        if isinstance(name, str):
            if "artifact_type" in value:
                port = _rewrite_role(name, "input_port", mappings)
                rewritten["name"] = _rewrite_role(port, "output_port", mappings)
            elif value.get("kind") in {"field", "source_view", "evaluation", "absence"}:
                rewritten["name"] = _rewrite_role(name, "coverage", mappings)
            elif value.get("kind") in {"read", "write"}:
                rewritten["name"] = _rewrite_role(name, "state", mappings)
            elif "consumed_ports" in value and "subject_port" in value:
                rewritten["name"] = _rewrite_role(name, "evidence", mappings)
        return rewritten
    return value


def _traces(declaration: Object, replacement: Object | None, expected: Object) -> list[Json]:
    mappings = _rename_maps(declaration, replacement)
    renamed = cast(Object, _rewrite(declaration, mappings))
    renamed_replacement = cast(Object | None, _rewrite(replacement, mappings)) if replacement is not None else None
    traces: list[Json] = [
        {
            "declaration": renamed,
            "events": _events(renamed, substitution=replacement is not None),
            "expected": judge(renamed, renamed_replacement),
            "replacement": renamed_replacement,
            "transformation": "rename",
        }
    ]
    if len(_nodes(declaration)) >= 2:
        reversed_declaration = deepcopy(declaration)
        reversed_declaration["nodes"] = list(reversed(_list(reversed_declaration["nodes"])))
        traces.append(
            {
                "declaration": reversed_declaration,
                "events": _events(reversed_declaration, substitution=replacement is not None),
                "expected": judge(reversed_declaration, replacement),
                "replacement": deepcopy(replacement),
                "transformation": "reverse_declaration_tuple",
            }
        )
    invariant_keys = ("status", "code", "topology")
    for trace_value in traces:
        trace = _obj(trace_value)
        if any(_obj(trace["expected"])[key] != expected[key] for key in invariant_keys):
            raise ValueError("metamorphic invariant failed")
    return traces


def _case(
    family: str, coordinates: Sequence[int], mutation: str, declaration: Object, replacement: Object | None = None
) -> Object:
    mode = "substitution" if replacement is not None else "admission"
    expected = judge(declaration, replacement)
    coordinate_text = "/".join(f"{value:03d}" for value in coordinates)
    case_id = f"{family}/{coordinate_text}/{mutation}"
    return {
        "case_id": case_id,
        "declaration": declaration,
        "expected": expected,
        "family": family,
        "mode": mode,
        "replacement": replacement,
        "traces": _traces(declaration, replacement, expected),
    }


def _topology_cases() -> Iterator[Object]:
    for n_ordinal, n in enumerate((1, 2, 3)):
        labels = tuple(f"N{index}" for index in range(n))
        domain = tuple((left, right) for left in labels for right in labels if left != right)
        for subset_ordinal, mask in enumerate(range(1 << len(domain))):
            edges = tuple(edge for index, edge in enumerate(domain) if mask & (1 << index))
            for permutation_ordinal, permutation in enumerate(itertools.permutations(labels)):
                declaration = _workflow(_z(n), tuple(_node(label, _z()) for label in permutation))
                declaration["family_marker"] = "topology"
                declaration["sequence"] = [{"after": _ref(after), "before": _ref(before)} for before, after in edges]
                cyclic, sinks = _cycle_and_sinks(declaration)
                if cyclic:
                    sink = "N0"
                elif sinks == 1:
                    outgoing = {before for before, _ in edges}
                    sink = next(label for label in labels if label not in outgoing)
                else:
                    sink = ""
                if sink:
                    declaration["outcome_bindings"] = [
                        _outcome_binding(sink, "ok", "ok"),
                        _outcome_binding(sink, "fail", "fail"),
                    ]
                yield _case("topology", (n_ordinal, subset_ordinal, permutation_ordinal), "edges", declaration)


def _ports_cases() -> Iterator[Object]:
    mutations = (
        "base",
        "input_type_A1",
        "output_type_A1",
        "remove_input_binding",
        "remove_output_binding",
        "duplicate_input_destination",
        "duplicate_output_destination",
        "remove_output_dependency",
        "duplicate_output_dependency",
        "missing_dependency_output",
        "missing_dependency_input",
        "missing_source_node",
        "missing_source_port",
        "missing_destination_node",
        "missing_destination_port",
        "internal_type_A1",
    )
    for base_ordinal, factory in enumerate((_one_q, _pipe_q)):
        for mutation_ordinal, mutation in enumerate(mutations):
            if mutation == "internal_type_A1" and base_ordinal == 0:
                continue
            declaration = factory()
            if mutation == "input_type_A1":
                _obj(_list(_obj(declaration["interface"])["inputs"])[0])["artifact_type"] = "A1@1"
            elif mutation == "output_type_A1":
                _obj(_list(_obj(declaration["interface"])["outputs"])[0])["artifact_type"] = "A1@1"
            elif mutation == "remove_input_binding":
                _list(declaration["input_bindings"]).pop(0)
            elif mutation == "remove_output_binding":
                _list(declaration["output_bindings"]).pop()
            elif mutation == "duplicate_input_destination":
                _list(declaration["input_bindings"]).append(deepcopy(_list(declaration["input_bindings"])[0]))
            elif mutation == "duplicate_output_destination":
                _list(declaration["output_bindings"]).append(deepcopy(_list(declaration["output_bindings"])[0]))
            elif mutation == "remove_output_dependency":
                _list(_obj(_nodes(declaration)[0]["operation"])["output_dependencies"]).clear()
            elif mutation == "duplicate_output_dependency":
                dependencies = _list(_obj(_nodes(declaration)[0]["operation"])["output_dependencies"])
                dependencies.append(deepcopy(dependencies[0]))
            elif mutation == "missing_dependency_output":
                _obj(_list(_obj(_nodes(declaration)[0]["operation"])["output_dependencies"])[0])["output"] = "o1"
            elif mutation == "missing_dependency_input":
                _obj(_list(_obj(_nodes(declaration)[0]["operation"])["output_dependencies"])[0])["inputs"] = ["i1"]
            elif mutation == "missing_source_node":
                _obj(_obj(_list(declaration["output_bindings"])[0])["source"])["node"] = _ref("N2")
            elif mutation == "missing_source_port":
                _obj(_obj(_list(declaration["output_bindings"])[0])["source"])["port"] = "o1"
            elif mutation == "missing_destination_node":
                _obj(_obj(_list(declaration["input_bindings"])[0])["destination"])["node"] = _ref("N2")
            elif mutation == "missing_destination_port":
                _obj(_obj(_list(declaration["input_bindings"])[0])["destination"])["port"] = "i1"
            elif mutation == "internal_type_A1":
                _obj(_list(_obj(_nodes(declaration)[1]["operation"])["inputs"])[0])["artifact_type"] = "A1@1"
            yield _case("ports", (base_ordinal, mutation_ordinal), mutation, declaration)


def _one_q() -> Object:
    return _one(_q())


def _pipe_q() -> Object:
    return _pipe()


def _choice_cases() -> Iterator[Object]:
    mutations = (
        "CHOICE_Z",
        "unmap_fail",
        "overlap_outcome",
        "overlap_member",
        "unknown_outcome",
        "selector_member",
        "remove_selector_edge",
        "second_choice_membership",
    )
    for ordinal, mutation in enumerate(mutations):
        declaration = _choice()
        branch_values = _list(_obj(_list(declaration["choices"])[0])["branches"])
        branches = [_obj(value) for value in branch_values]
        if mutation == "unmap_fail":
            branch_values.pop()
            declaration["nodes"] = _list(declaration["nodes"])[:2]
            declaration["sequence"] = _list(declaration["sequence"])[:1]
            declaration["outcome_bindings"] = _list(declaration["outcome_bindings"])[:2] + [
                _outcome_binding("N0", "fail", "fail")
            ]
        elif mutation == "overlap_outcome":
            _list(branches[1]["outcomes"]).append("ok")
        elif mutation == "overlap_member":
            _list(branches[1]["members"]).append(_ref("N1"))
        elif mutation == "unknown_outcome":
            branches[1]["outcomes"] = ["other"]
        elif mutation == "selector_member":
            _list(branches[0]["members"]).append(_ref("N0"))
        elif mutation == "remove_selector_edge":
            _list(declaration["sequence"]).pop(0)
        elif mutation == "second_choice_membership":
            _list(declaration["choices"]).append(
                {"branches": [{"members": [_ref("N1")], "outcomes": ["ok"]}], "selector": _ref("N2")}
            )
        yield _case("choice", (ordinal,), mutation, declaration)


def _change_semantic(operation: Object, mutation: str) -> None:
    if mutation == "input_type_A1":
        _obj(_list(operation["inputs"])[0])["artifact_type"] = "A1@1"
    elif mutation == "output_type_A1":
        _obj(_list(operation["outputs"])[0])["artifact_type"] = "A1@1"
    elif mutation == "dependency_empty":
        dependency = _obj(_list(operation["output_dependencies"])[0])
        dependency["inputs"] = []
        dependency["identity_input"] = None
    elif mutation == "remove_fail_outcome":
        operation["outcomes"] = [value for value in _list(operation["outcomes"]) if _obj(value)["name"] != "fail"]
    elif mutation == "context_alt":
        for outcome in map(_obj, _list(operation["outcomes"])):
            if _list(outcome["context"]):
                _obj(_list(outcome["context"])[0])["meaning"] = "context_alt"
    elif mutation == "evidence_meaning_alt":
        evidence = _list(_obj(_list(operation["outcomes"])[0])["evidence"])
        _obj(evidence[0])["meaning"] = "assessment_alt"
    elif mutation == "remove_field_coverage":
        evidence = _obj(_list(_obj(_list(operation["outcomes"])[0])["evidence"])[0])
        evidence["coverage"] = [value for value in _list(evidence["coverage"]) if _obj(value)["name"] != "field"]
    elif mutation == "state_alt":
        for outcome in map(_obj, _list(operation["outcomes"])):
            _obj(_list(outcome["state_effects"])[0])["name"] = "state_alt"
    elif mutation == "model_revision_2":
        for outcome in map(_obj, _list(operation["outcomes"])):
            _obj(_list(outcome["model_requirements"])[0])["revision"] = 2


def _consistent_body_mutation(body: Object, mutation: str) -> None:
    interface = _obj(body["interface"])
    affected_nodes = _nodes(body)
    if mutation in {"input_type_A1", "output_type_A1", "dependency_empty"}:
        _change_semantic(interface, mutation)
        target = affected_nodes[0] if mutation != "output_type_A1" or len(affected_nodes) == 1 else affected_nodes[-1]
        target_operation = _obj(target["operation"])
        _change_semantic(target_operation, mutation)
        if len(affected_nodes) > 1 and mutation in {"input_type_A1", "output_type_A1"}:
            _obj(_list(interface["output_dependencies"])[0])["identity_input"] = None
            _obj(_list(target_operation["output_dependencies"])[0])["identity_input"] = None
    else:
        _change_semantic(interface, mutation)
        target = affected_nodes[-1] if mutation == "remove_fail_outcome" else affected_nodes[0]
        _change_semantic(_obj(target["operation"]), mutation)
        if mutation == "remove_fail_outcome":
            body["outcome_bindings"] = [
                value for value in _list(body["outcome_bindings"]) if _obj(_obj(value)["source"])["outcome"] != "fail"
            ]


def _subgraph_cases() -> Iterator[Object]:
    bodies = (
        (_one(_z()), ()),
        (
            _one(_p()),
            (
                "input_type_A1",
                "output_type_A1",
                "dependency_empty",
                "context_alt",
                "evidence_meaning_alt",
                "remove_field_coverage",
                "state_alt",
                "model_revision_2",
            ),
        ),
        (_pipe(), ("input_type_A1", "output_type_A1", "dependency_empty")),
    )
    resources = tuple(f"widen_{field}_by_1" for field in RESOURCE_FIELDS)
    for body_ordinal, (base_body, extras) in enumerate(bodies):
        mutations = ("equal", "remove_fail_outcome", *resources, *extras)
        for mutation_ordinal, mutation in enumerate(mutations):
            body = deepcopy(base_body)
            declaration = _wrap(base_body)
            if mutation.startswith("widen_"):
                field = mutation.removeprefix("widen_").removesuffix("_by_1")
                for operation in (_obj(body["interface"]), _obj(_nodes(body)[0]["operation"])):
                    for outcome in map(_obj, _list(operation["outcomes"])):
                        ceiling = _obj(outcome["ceiling"])
                        ceiling[field] = cast(int, ceiling[field]) + 1
            elif mutation != "equal":
                _consistent_body_mutation(body, mutation)
            cast(list[Object], declaration["nodes"])[0]["body"] = _reowner_workflow(body, "W1")
            yield _case("subgraph", (body_ordinal, mutation_ordinal), mutation, declaration)


def _replacement_mutation(replacement: Object, mutation: str) -> None:
    interface = _obj(replacement["interface"])
    operation = _obj(_nodes(replacement)[0]["operation"])
    if mutation.startswith(("narrow_", "widen_")):
        direction = -1 if mutation.startswith("narrow_") else 1
        rest = mutation.split("_", 1)[1].removesuffix("_by_1")
        outcome_name, field = rest.split("_", 1)
        for current in (interface, operation):
            outcome = _operation_outcomes(current)[outcome_name]
            ceiling = _obj(outcome["ceiling"])
            ceiling[field] = cast(int, ceiling[field]) + direction
    else:
        _change_semantic(interface, mutation)
        _change_semantic(operation, mutation)
        if mutation == "remove_fail_outcome":
            replacement["outcome_bindings"] = [
                value
                for value in _list(replacement["outcome_bindings"])
                if _obj(_obj(value)["source"])["outcome"] != "fail"
            ]


def _substitution_cases() -> Iterator[Object]:
    base = _one(_p())
    base["substitution_target"] = _ref("N0")
    ceiling_mutations: list[str] = []
    for outcome, ceiling in (("ok", _ceiling(1, 1, 8, 8)), ("fail", _ceiling(1, 1, 8, 0))):
        for field in RESOURCE_FIELDS:
            if cast(int, ceiling[field]) > 0:
                ceiling_mutations.append(f"narrow_{outcome}_{field}_by_1")
            ceiling_mutations.append(f"widen_{outcome}_{field}_by_1")
    mutations = (
        "equal",
        *ceiling_mutations,
        "input_type_A1",
        "output_type_A1",
        "dependency_empty",
        "remove_fail_outcome",
        "context_alt",
        "evidence_meaning_alt",
        "remove_field_coverage",
        "state_alt",
        "model_revision_2",
    )
    for ordinal, mutation in enumerate(mutations):
        replacement = _reowner_workflow(_one(_p()), "W1")
        if mutation != "equal":
            _replacement_mutation(replacement, mutation)
        yield _case("substitution", (ordinal,), mutation, deepcopy(base), replacement)


def _protection_cases() -> Iterator[Object]:
    mutations = ("exact", "meaning_alt", "subject_i0", "add_consumed_i1", "add_field_alt", "add_absence")
    for ordinal, mutation in enumerate(mutations):
        declaration = _one(_p(two_inputs=True))
        requirement: Object = {
            "consumed_ports": ["i0"],
            "coverage": _sorted(({"kind": "field", "name": "field"}, {"kind": "source_view", "name": "source"})),
            "meaning": "assessment",
            "outcome": "ok",
            "subject_port": "o0",
        }
        if mutation == "meaning_alt":
            requirement["meaning"] = "assessment_alt"
        elif mutation == "subject_i0":
            requirement["subject_port"] = "i0"
        elif mutation == "add_consumed_i1":
            _list(requirement["consumed_ports"]).append("i1")
        elif mutation == "add_field_alt":
            _list(requirement["coverage"]).append({"kind": "field", "name": "field_alt"})
            requirement["coverage"] = _sorted(_list(requirement["coverage"]))
        elif mutation == "add_absence":
            _list(requirement["coverage"]).append({"kind": "absence", "name": "absence"})
            requirement["coverage"] = _sorted(_list(requirement["coverage"]))
        declaration["protection"] = [requirement]
        yield _case("protection", (ordinal,), mutation, declaration)


def _lineage_cases() -> Iterator[Object]:
    mutations = ("PIPE_L", "dependency_fanin", "zero_upstream", "zero_downstream", "many_downstream")
    for ordinal, mutation in enumerate(mutations):
        declaration = _pipe(lineage=True)
        if mutation == "dependency_fanin":
            _list(_obj(declaration["interface"])["inputs"]).append({"artifact_type": "A0@1", "name": "i1"})
            _list(_obj(_nodes(declaration)[0]["operation"])["inputs"]).append({"artifact_type": "A0@1", "name": "i1"})
            _list(declaration["input_bindings"]).append(_input("i1", "N0", "i1"))
            for operation in (_obj(declaration["interface"]), _obj(_nodes(declaration)[0]["operation"])):
                _obj(_list(operation["output_dependencies"])[0])["inputs"] = ["i0", "i1"]
        elif mutation == "zero_upstream":
            _obj(_list(_obj(_nodes(declaration)[0]["operation"])["output_dependencies"])[0])["identity_input"] = None
        elif mutation == "zero_downstream":
            _obj(_list(_obj(_nodes(declaration)[1]["operation"])["output_dependencies"])[0])["identity_input"] = None
        elif mutation == "many_downstream":
            interface = _obj(declaration["interface"])
            _list(interface["outputs"]).append({"artifact_type": "A0@1", "name": "o1"})
            _list(interface["output_dependencies"]).append(_dependency("o1", ("i0",), "i0"))
            for outcome in map(_obj, _list(interface["outcomes"])):
                _list(outcome["produced_ports"]).append("o1")
            _list(declaration["output_bindings"]).append(_output("N1", "o0", "o1"))
        yield _case("lineage", (ordinal,), mutation, declaration)


def _exact_limits(declaration: Object) -> None:
    metrics = _metrics(declaration)
    declaration["limits"] = {
        "max_bindings": metrics["bindings"],
        "max_branch_members": metrics["branch_members"],
        "max_choice_states": metrics["choice_states"],
        "max_choices": metrics["choices"],
        "max_nodes": metrics["nodes"],
        "max_sequence_edges": metrics["sequence_edges"],
        "max_subgraph_depth": metrics["subgraph_depth"],
    }


def _limits_cases() -> Iterator[Object]:
    bases = (_one(_p()), _pipe(), _choice())
    metric_for_limit = dict(
        zip(
            LIMIT_FIELDS,
            ("nodes", "bindings", "sequence_edges", "choices", "branch_members", "subgraph_depth", "choice_states"),
            strict=True,
        )
    )
    for base_ordinal, base in enumerate(bases):
        mutations = ["exact_all"]
        metrics = _metrics(base)
        for limit, metric in metric_for_limit.items():
            if metrics[metric] > (1 if limit in {"max_nodes", "max_subgraph_depth", "max_choice_states"} else 0):
                mutations.append(f"one_under_{metric}")
        for mutation_ordinal, mutation in enumerate(mutations):
            declaration = deepcopy(base)
            _exact_limits(declaration)
            if mutation != "exact_all":
                metric = mutation.removeprefix("one_under_")
                limit = next(key for key, value in metric_for_limit.items() if value == metric)
                _obj(declaration["limits"])[limit] = cast(int, _obj(declaration["limits"])[limit]) - 1
            yield _case("limits_ownership", (0, base_ordinal, mutation_ordinal), mutation, declaration)
    for wrapper_ordinal, body in enumerate((_one(_z()), _pipe())):
        base = _wrap(body)
        for mutation_ordinal, mutation in enumerate(("exact_all", "one_under_nodes", "one_under_subgraph_depth")):
            declaration = deepcopy(base)
            _exact_limits(declaration)
            if mutation == "one_under_nodes":
                _obj(declaration["limits"])["max_nodes"] = cast(int, _obj(declaration["limits"])["max_nodes"]) - 1
            elif mutation == "one_under_subgraph_depth":
                _obj(declaration["limits"])["max_subgraph_depth"] = 1
            yield _case("limits_ownership", (1, wrapper_ordinal, mutation_ordinal), mutation, declaration)
    foreign_specs = (
        ("operation_node_id", _one(_p())),
        ("subgraph_node_id", _wrap(_one(_z()))),
        ("output_ref", _one(_p())),
        ("input_ref", _one(_p())),
        ("outcome_ref", _one(_p())),
        ("sequence_before", _pipe()),
        ("sequence_after", _pipe()),
        ("choice_selector", _choice()),
        ("choice_member", _choice()),
        ("substitution_target", _one(_p())),
    )
    for ordinal, (location, declaration) in enumerate(foreign_specs):
        replacement: Object | None = None
        if location in {"operation_node_id", "subgraph_node_id"}:
            foreign_label = "N2" if location == "subgraph_node_id" else "N0"
            _nodes(declaration)[0]["id"] = _ref(foreign_label, "W1")
        elif location == "output_ref":
            _obj(_obj(_list(declaration["output_bindings"])[0])["source"])["node"] = _ref("N0", "W1")
        elif location == "input_ref":
            _obj(_obj(_list(declaration["input_bindings"])[0])["destination"])["node"] = _ref("N0", "W1")
        elif location == "outcome_ref":
            _obj(_obj(_list(declaration["outcome_bindings"])[0])["source"])["node"] = _ref("N0", "W1")
        elif location.startswith("sequence_"):
            _obj(_list(declaration["sequence"])[0])[location.split("_")[1]] = _ref("N0", "W1")
        elif location == "choice_selector":
            _obj(_list(declaration["choices"])[0])["selector"] = _ref("N0", "W1")
        elif location == "choice_member":
            _list(_obj(_list(_obj(_list(declaration["choices"])[0])["branches"])[0])["members"])[0] = _ref("N1", "W1")
        else:
            declaration["substitution_target"] = _ref("N2", "W1")
            replacement = _reowner_workflow(_one(_p()), "W1")
        yield _case("limits_ownership", (2, ordinal), f"foreign_owner_{location}", declaration, replacement)


def generate_cases() -> tuple[Object, ...]:
    """Enumerate the eight exact finite families and reject collisions."""
    generators = (
        _topology_cases,
        _ports_cases,
        _choice_cases,
        _subgraph_cases,
        _substitution_cases,
        _protection_cases,
        _lineage_cases,
        _limits_cases,
    )
    cases = [case for generator in generators for case in generator()]
    ids: set[str] = set()
    payloads: dict[bytes, str] = {}
    for case in cases:
        case_id = cast(str, case["case_id"])
        if case_id in ids:
            raise ValueError("duplicate case id")
        ids.add(case_id)
        payload = _canonical({key: case[key] for key in ("family", "mode", "declaration", "replacement", "expected")})
        if payload in payloads:
            raise ValueError("duplicate canonical payload")
        payloads[payload] = case_id
    return tuple(sorted(cases, key=lambda case: cast(str, case["case_id"]).encode()))


def canonical_bytes(cases: Sequence[Object]) -> bytes:
    return _canonical({"cases": list(cases), "schema_version": 1}) + b"\n"


def counts(cases: Sequence[Object]) -> Object:
    metrics = [_metrics(_obj(case["declaration"])) for case in cases]
    return {
        "case_count": len(cases),
        "event_count": sum(len(_list(trace["events"])) for case in cases for trace in map(_obj, _list(case["traces"]))),
        "max_choice_states": max(item["choice_states"] for item in metrics),
        "max_nodes": max(item["nodes"] for item in metrics),
        "max_subgraph_depth": max(item["subgraph_depth"] for item in metrics),
        "trace_count": sum(len(_list(case["traces"])) for case in cases),
    }


def build_manifest(
    cases: Sequence[Object],
    corpus: bytes,
    *,
    generator_sha256: str,
    self_test_sha256: str,
) -> Object:
    """Build the exact manifest for already generated canonical corpus bytes."""
    return {
        "alphabet": list(ALPHABET),
        "capability": "workflow_static_v1",
        "contract_sha256": CONTRACT_SHA256,
        "corpus_path": CORPUS_PATH,
        "corpus_sha256": hashlib.sha256(corpus).hexdigest(),
        "counts": counts(cases),
        "family_bounds": {
            "choice_state_max": 4,
            "expanded_node_count_max": 3,
            "family_ids": list(FAMILY_IDS),
            "subgraph_depth_max": 2,
            "topology_node_counts": [1, 2, 3],
        },
        "generation_provenance": {
            "byte_identical": True,
            "generations": 2,
            "tools": {
                "generator": GENERATOR_VERSION,
                "python": producer_python(),
                "self_test": SELF_TEST_VERSION,
            },
        },
        "generator_sha256": generator_sha256,
        "independence": {"kind": "conditional-symmetric-v1", "rule_ids": list(RULE_IDS)},
        "manifest_version": "workflow-static-reference-v1",
        "packet_id": "R1a",
        "schema_version": 1,
        "self_test_sha256": self_test_sha256,
    }


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def producer_python() -> str:
    return f"{platform.python_implementation()} {platform.python_version()}"


def _has_closed_vocabularies(declaration: Object) -> bool:
    operations = [_obj(declaration["interface"])]
    for node in _nodes(declaration):
        if node.get("kind") not in {"operation", "subgraph"}:
            return False
        operations.append(_obj(node["operation"]))
        if node["kind"] == "subgraph" and not _has_closed_vocabularies(_obj(node["body"])):
            return False
    for operation in operations:
        for outcome in map(_obj, _list(operation["outcomes"])):
            if outcome.get("category") not in {"success", "failure", "cancelled", "lost", "blocked", "inconsistent"}:
                return False
            if any(_obj(context).get("capture") != "whole_artifact" for context in _list(outcome["context"])):
                return False
            if any(
                _obj(coverage).get("kind") not in {"field", "source_view", "evaluation", "absence"}
                for evidence in map(_obj, _list(outcome["evidence"]))
                for coverage in _list(evidence["coverage"])
            ):
                return False
            if any(_obj(effect).get("kind") not in {"read", "write"} for effect in _list(outcome["state_effects"])):
                return False
    if any(
        _obj(_obj(binding)["source"]).get("kind") not in {"workflow_input", "node_output"}
        for binding in _list(declaration["input_bindings"])
    ):
        return False
    if any(
        _obj(_obj(binding)["source"]).get("kind") != "node_output" for binding in _list(declaration["output_bindings"])
    ):
        return False
    return all(
        _obj(coverage).get("kind") in {"field", "source_view", "evaluation", "absence"}
        for requirement in map(_obj, _list(declaration["protection"]))
        for coverage in _list(requirement["coverage"])
    )


def load_cases(value: Json) -> tuple[Object, ...]:
    root = _obj(value)
    if set(root) != {"cases", "schema_version"} or root["schema_version"] != 1:
        raise ValueError("invalid workflow fixture structure")
    cases = tuple(_obj(item) for item in _list(root["cases"]))
    required = {"case_id", "family", "mode", "declaration", "replacement", "expected", "traces"}
    for case in cases:
        if set(case) != required or case["family"] not in FAMILIES or case["mode"] not in {"admission", "substitution"}:
            raise ValueError("invalid workflow fixture structure")
        replacement = _obj(case["replacement"]) if case["replacement"] is not None else None
        if not _has_closed_vocabularies(_obj(case["declaration"])) or (
            replacement is not None and not _has_closed_vocabularies(replacement)
        ):
            raise ValueError("invalid workflow fixture structure")
        expected = _obj(case["expected"])
        if set(expected) != {
            "status",
            "code",
            "topology",
            "normalized",
            "protection_eligible_outcomes",
            "unmet_protection",
        }:
            raise ValueError("invalid workflow fixture structure")
        for trace in map(_obj, _list(case["traces"])):
            if set(trace) != {"transformation", "events", "declaration", "replacement", "expected"}:
                raise ValueError("invalid workflow fixture structure")
            trace_replacement = _obj(trace["replacement"]) if trace["replacement"] is not None else None
            if not _has_closed_vocabularies(_obj(trace["declaration"])) or (
                trace_replacement is not None and not _has_closed_vocabularies(trace_replacement)
            ):
                raise ValueError("invalid workflow fixture structure")
            if any(_obj(event).get("op") not in ALPHABET for event in _list(trace["events"])):
                raise ValueError("invalid workflow fixture structure")
    return cases


def module_is_independent() -> bool:
    return not any(name == "anonymizer" or name.startswith("anonymizer.") for name in sys.modules)
