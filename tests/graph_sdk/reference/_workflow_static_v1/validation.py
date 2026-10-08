# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent workflow_static_v1 reference: validation."""

from __future__ import annotations

from copy import deepcopy
from typing import cast

from tests.graph_sdk.reference._workflow_static_v1.builders import (
    _node,
)
from tests.graph_sdk.reference._workflow_static_v1.model import (
    RESOURCE_FIELDS,
    Json,
    Object,
    ValidationCode,
    _canonical,
    _obj,
    _sorted,
)


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
            _obj(node["operation"]),
            _obj(_obj(node["body"])["interface"]),
            allow_narrow=True,
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
    replacement_node = _node(
        _id_label(target_node["id"]),
        _obj(target_node["operation"]),
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
