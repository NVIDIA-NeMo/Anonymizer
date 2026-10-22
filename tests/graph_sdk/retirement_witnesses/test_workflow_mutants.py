# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retained independent named workflow mutant discriminators; no old interpreter."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, cast

import pytest

from tests.graph_sdk.retirement_witnesses.witness_sources import workflow_mutant_cases

Json = Any
Object = dict[str, Any]
FROZEN_CASES = workflow_mutant_cases()


def _object(value: Json) -> Object:
    if not isinstance(value, dict):
        raise ValueError("expected object")
    return cast(Object, value)


def _array(value: Json) -> list[Json]:
    if not isinstance(value, list):
        raise ValueError("expected array")
    return value


def _canonical(value: Json) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sorted(values: list[Json]) -> list[Json]:
    return sorted(values, key=_canonical)


def _label(identity: Json) -> str:
    return cast(str, _object(identity)["label"])


def _independent_topology(declaration: Object) -> tuple[bool, int | None]:
    labels = {_label(_object(node)["id"]) for node in _array(declaration["nodes"])}
    edges = {
        (_label(_object(edge)["before"]), _label(_object(edge)["after"]))
        for edge in map(_object, _array(declaration["sequence"]))
    }
    remaining = set(labels)
    while remaining:
        roots = {
            node for node in remaining if not any(after == node and before in remaining for before, after in edges)
        }
        if not roots:
            return False, None
        remaining -= roots
    sinks = sum(not any(before == node for before, _ in edges) for node in labels)
    return True, sinks


def _independent_normalized(declaration: Object) -> Object:
    def expanded(nodes: list[Json]) -> int:
        total = len(nodes)
        for node_value in nodes:
            node = _object(node_value)
            if node["kind"] == "subgraph":
                total += expanded(_array(_object(node["body"])["nodes"]))
        return total

    return {
        "choices": _sorted(deepcopy(_array(declaration["choices"]))),
        "expanded_node_count": expanded(_array(declaration["nodes"])),
        "input_bindings": _sorted(deepcopy(_array(declaration["input_bindings"]))),
        "interface": deepcopy(declaration["interface"]),
        "nodes": _sorted(deepcopy(_array(declaration["nodes"]))),
        "outcome_bindings": _sorted(deepcopy(_array(declaration["outcome_bindings"]))),
        "output_bindings": _sorted(deepcopy(_array(declaration["output_bindings"]))),
        "protection_requirements": _sorted(deepcopy(_array(declaration["protection"]))),
        "sequence": _sorted(deepcopy(_array(declaration["sequence"]))),
    }


def _independent_code(family: str, mutation: str, declaration: Object) -> str | None:
    if family == "topology":
        acyclic, sinks = _independent_topology(declaration)
        return None if acyclic and sinks == 1 else "missing" if acyclic else "cycle"
    if family == "ports":
        if mutation == "base":
            return None
        if mutation in {"input_type_A1", "output_type_A1", "internal_type_A1"}:
            return "contradictory"
        if mutation in {
            "duplicate_input_destination",
            "duplicate_output_destination",
            "duplicate_output_dependency",
        }:
            return "duplicate"
        return "missing"
    if family == "choice":
        return {
            "CHOICE_Z": None,
            "unmap_fail": None,
            "overlap_outcome": "overlap",
            "overlap_member": "overlap",
            "unknown_outcome": "missing",
            "selector_member": "invalid_value",
            "remove_selector_edge": "contradictory",
            "second_choice_membership": "overlap",
        }[mutation]
    if family == "subgraph":
        return None if mutation == "equal" else "contradictory"
    if family == "substitution":
        return None if mutation == "equal" or mutation.startswith("narrow_") else "contradictory"
    if family == "protection":
        return None
    if family == "lineage":
        return {
            "PIPE_L": None,
            "dependency_fanin": None,
            "zero_upstream": "missing",
            "zero_downstream": "missing",
            "many_downstream": "contradictory",
        }[mutation]
    if family == "limits_ownership":
        if mutation == "exact_all":
            return None
        return "foreign_owner" if mutation.startswith("foreign_owner_") else "limit_exceeded"
    raise AssertionError(f"unknown family {family}")


def _mutation(case: Object) -> str:
    return cast(str, case["case_id"]).rsplit("/", 1)[1]


def _find(family: str, mutation: str) -> Object:
    return next(case for case in FROZEN_CASES if case["family"] == family and _mutation(case) == mutation)


def _subgraph_operations(declaration: Object) -> tuple[Object, Object]:
    wrapper = _object(_array(declaration["nodes"])[0])
    return _object(wrapper["operation"]), _object(_object(wrapper["body"])["interface"])


def _without_semantic(operation: Object, name: str) -> Object:
    result = deepcopy(operation)
    for outcome in map(_object, _array(result["outcomes"])):
        if name == "erase_context":
            outcome.pop("context")
        elif name == "erase_evidence_meaning":
            for evidence in map(_object, _array(outcome["evidence"])):
                evidence.pop("meaning")
        elif name == "erase_coverage":
            for evidence in map(_object, _array(outcome["evidence"])):
                evidence.pop("coverage")
        elif name == "widen_state":
            outcome.pop("state_effects")
        elif name == "change_model":
            outcome.pop("model_requirements")
        elif name == "widen_resource":
            outcome.pop("ceiling")
    return result


def _mutant_accepts(name: str, declaration: Object) -> bool:
    if name in {
        "erase_context",
        "erase_evidence_meaning",
        "erase_coverage",
        "widen_state",
        "change_model",
        "widen_resource",
    }:
        wrapper, body = _subgraph_operations(declaration)
        return _without_semantic(wrapper, name) == _without_semantic(body, name)
    if name in {"erase_port_type", "trust_python_type"}:
        binding = _object(_array(declaration["input_bindings"])[-1 if name == "trust_python_type" else 0])
        source = _object(binding["source"])
        destination = _object(binding["destination"])
        interface = _object(declaration["interface"])
        nodes = {_label(_object(node)["id"]): _object(node) for node in _array(declaration["nodes"])}
        source_type = (
            next(
                _object(port)["artifact_type"]
                for port in _array(interface["inputs"])
                if _object(port)["name"] == source["port"]
            )
            if source["kind"] == "workflow_input"
            else next(
                _object(port)["artifact_type"]
                for port in _array(_object(nodes[_label(source["node"])]["operation"])["outputs"])
                if _object(port)["name"] == source["port"]
            )
        )
        destination_type = next(
            _object(port)["artifact_type"]
            for port in _array(_object(nodes[_label(destination["node"])]["operation"])["inputs"])
            if _object(port)["name"] == destination["port"]
        )
        return isinstance(source_type, str) and isinstance(destination_type, str)
    if name == "erase_output_dependency":
        operation = _object(_array(declaration["nodes"])[0])["operation"]
        return bool(_array(_object(operation)["outputs"]))
    if name == "infer_identity_from_dependency":
        operation = _object(_object(_array(declaration["nodes"])[0])["operation"])
        dependency = _object(_array(operation["output_dependencies"])[0])
        return dependency["identity_input"] is not None or bool(_array(dependency["inputs"]))
    if name == "erase_outcome":
        choice = _object(_array(declaration["choices"])[0])
        return all(_array(_object(branch)["outcomes"]) for branch in _array(choice["branches"]))
    if name == "overlap_choice":
        choice = _object(_array(declaration["choices"])[0])
        seen: set[str] = set()
        for branch in map(_object, _array(choice["branches"])):
            outcomes = set(cast(list[str], branch["outcomes"]))
            if seen & outcomes:
                return False
            seen |= outcomes
        return True
    raise AssertionError(f"unknown semantic mutant: {name}")


def _reverse_sequence_mutant(declaration: Object) -> Object:
    normalized = _independent_normalized(declaration)
    for edge in map(_object, _array(normalized["sequence"])):
        edge["before"], edge["after"] = edge["after"], edge["before"]
    return normalized


def _assert_named_semantics(name: str, case: Object, mutant_result: Json) -> None:
    declaration = _object(case["declaration"])
    if name == "reverse_sequence":
        correct: Json = _independent_normalized(declaration)
    else:
        correct = _independent_code(cast(str, case["family"]), _mutation(case), declaration) is None
    assert mutant_result == correct, f"named semantic assertion killed {name}"


def _mutant_case(name: str) -> Object:
    selectors = {
        "erase_port_type": ("ports", "input_type_A1"),
        "erase_output_dependency": ("ports", "remove_output_dependency"),
        "infer_identity_from_dependency": ("lineage", "zero_upstream"),
        "erase_outcome": ("choice", "unknown_outcome"),
        "erase_context": ("subgraph", "context_alt"),
        "erase_evidence_meaning": ("subgraph", "evidence_meaning_alt"),
        "erase_coverage": ("subgraph", "remove_field_coverage"),
        "widen_state": ("subgraph", "state_alt"),
        "change_model": ("subgraph", "model_revision_2"),
        "widen_resource": ("subgraph", "widen_max_activations_by_1"),
        "overlap_choice": ("choice", "overlap_member"),
        "reverse_sequence": ("lineage", "PIPE_L"),
        "trust_python_type": ("ports", "internal_type_A1"),
    }
    family, mutation = selectors[name]
    return _find(family, mutation)


@pytest.mark.parametrize(
    "name",
    [
        "erase_port_type",
        "erase_output_dependency",
        "infer_identity_from_dependency",
        "erase_outcome",
        "erase_context",
        "erase_evidence_meaning",
        "erase_coverage",
        "widen_state",
        "change_model",
        "widen_resource",
        "overlap_choice",
        "reverse_sequence",
        "trust_python_type",
    ],
)
def test_named_semantic_mutant_is_killed(name: str) -> None:
    case = _mutant_case(name)
    declaration = _object(case["declaration"])
    mutant_result: Json = (
        _reverse_sequence_mutant(declaration) if name == "reverse_sequence" else _mutant_accepts(name, declaration)
    )
    with pytest.raises(AssertionError, match=f"named semantic assertion killed {name}"):
        _assert_named_semantics(name, case, mutant_result)
