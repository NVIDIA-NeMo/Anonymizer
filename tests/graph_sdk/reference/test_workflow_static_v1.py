# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-tests for the frozen finite static-workflow reference."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import TypeAlias, cast

import pytest

from tests.graph_sdk.reference import workflow_static_v1 as reference

Json: TypeAlias = reference.Json
Object: TypeAlias = reference.Object

REFERENCE_DIR = Path(__file__).parent
GENERATOR_PATH = REFERENCE_DIR / "workflow_static_v1.py"
CORPUS_PATH = REFERENCE_DIR / "workflow_static_v1_cases.json"
MANIFEST_PATH = REFERENCE_DIR / "workflow_static_v1_manifest.json"
FROZEN_BYTES = CORPUS_PATH.read_bytes()
FROZEN_CASES = reference.load_cases(json.loads(FROZEN_BYTES))
MANIFEST = cast(Object, json.loads(MANIFEST_PATH.read_bytes()))


@pytest.fixture(scope="module")
def generations() -> tuple[tuple[Object, ...], tuple[Object, ...]]:
    """Generate exactly twice and share the results across freeze checks."""
    return reference.generate_cases(), reference.generate_cases()


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


def _independent_protection(declaration: Object) -> tuple[list[Json], list[Json]]:
    interface = _object(declaration["interface"])
    outcomes = {cast(str, _object(value)["name"]): _object(value) for value in _array(interface["outcomes"])}
    eligible: list[Json] = []
    unmet: list[Json] = []
    for requirement_value in _array(declaration["protection"]):
        requirement = _object(requirement_value)
        outcome = outcomes.get(cast(str, requirement["outcome"]))
        matched = False
        if outcome is not None:
            required_coverage = {_canonical(value) for value in _array(requirement["coverage"])}
            required_consumed = set(cast(list[str], requirement["consumed_ports"]))
            for promise in map(_object, _array(outcome["evidence"])):
                matched = (
                    promise["meaning"] == requirement["meaning"]
                    and promise["subject_port"] == requirement["subject_port"]
                    and required_consumed <= set(cast(list[str], promise["consumed_ports"]))
                    and required_coverage <= {_canonical(value) for value in _array(promise["coverage"])}
                )
                if matched:
                    break
        if matched:
            eligible.append(requirement["outcome"])
        else:
            unmet.append(deepcopy(requirement))
    return sorted(set(eligible), key=_canonical), _sorted(unmet)


def _independent_code(family: str, mutation: str, declaration: Object) -> str | None:
    if family == "topology":
        acyclic, sinks = _independent_topology(declaration)
        return None if acyclic and sinks == 1 else "missing" if acyclic else "cycle"
    if family == "ports":
        if mutation == "base":
            return None
        if mutation in {"input_type_A1", "output_type_A1", "internal_type_A1"}:
            return "contradictory"
        if mutation in {"duplicate_input_destination", "duplicate_output_destination", "duplicate_output_dependency"}:
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


def _independent_substitution(declaration: Object, replacement: Object) -> Object:
    result = deepcopy(declaration)
    target = _object(result["substitution_target"])
    nodes = _array(result["nodes"])
    for index, node_value in enumerate(nodes):
        node = _object(node_value)
        if node["id"] == target:
            nodes[index] = {
                "body": deepcopy(replacement),
                "id": deepcopy(node["id"]),
                "kind": "subgraph",
                "operation": deepcopy(_object(node["operation"])),
            }
            break
    else:
        raise AssertionError("accepted substitution target is absent")
    return result


def _independent_expected(family: str, mutation: str, declaration: Object, replacement: Object | None = None) -> Object:
    code = _independent_code(family, mutation, declaration)
    topology: Json = None
    if family == "topology":
        acyclic, sinks = _independent_topology(declaration)
        topology = {"acyclic": acyclic, "sink_count": sinks}
    if code is not None:
        return {
            "code": code,
            "normalized": None,
            "protection_eligible_outcomes": [],
            "status": "rejected",
            "topology": topology,
            "unmet_protection": [],
        }
    admitted = (
        _independent_substitution(declaration, replacement)
        if family == "substitution" and replacement is not None
        else declaration
    )
    eligible, unmet = _independent_protection(admitted)
    return {
        "code": None,
        "normalized": _independent_normalized(admitted),
        "protection_eligible_outcomes": eligible,
        "status": "accepted",
        "topology": topology,
        "unmet_protection": unmet,
    }


def _mutation(case: Object) -> str:
    return cast(str, case["case_id"]).rsplit("/", 1)[1]


def test_frozen_generation_is_byte_identical(
    generations: tuple[tuple[Object, ...], tuple[Object, ...]],
) -> None:
    first, second = generations
    first_bytes = reference.canonical_bytes(first)
    second_bytes = reference.canonical_bytes(second)
    assert first_bytes == second_bytes == FROZEN_BYTES
    assert hashlib.sha256(FROZEN_BYTES).hexdigest() == MANIFEST["corpus_sha256"]


def test_every_expected_result_and_trace_has_an_independent_witness() -> None:
    for case in FROZEN_CASES:
        family = cast(str, case["family"])
        mutation = _mutation(case)
        replacement = _object(case["replacement"]) if case["replacement"] is not None else None
        expected = _independent_expected(family, mutation, _object(case["declaration"]), replacement)
        assert case["expected"] == expected, case["case_id"]
        for trace in map(_object, _array(case["traces"])):
            trace_replacement = _object(trace["replacement"]) if trace["replacement"] is not None else None
            trace_expected = _independent_expected(family, mutation, _object(trace["declaration"]), trace_replacement)
            assert trace["expected"] == trace_expected, (case["case_id"], trace["transformation"])


def test_exact_family_domains_and_bounds() -> None:
    counts = Counter(cast(str, case["family"]) for case in FROZEN_CASES)
    assert counts == {
        "topology": 393,
        "ports": 31,
        "choice": 8,
        "subgraph": 29,
        "substitution": 25,
        "protection": 6,
        "lineage": 5,
        "limits_ownership": 29,
    }
    topology_counts = Counter(
        len(_array(_object(case["declaration"])["nodes"])) for case in FROZEN_CASES if case["family"] == "topology"
    )
    assert topology_counts == {1: 1, 2: 8, 3: 384}
    assert set(counts) == set(reference.FAMILIES)
    assert len({cast(str, case["case_id"]) for case in FROZEN_CASES}) == len(FROZEN_CASES)
    measured = reference.counts(FROZEN_CASES)
    assert measured["max_nodes"] == 3
    assert measured["max_subgraph_depth"] == 2
    assert measured["max_choice_states"] == 4
    observed_events = {
        cast(str, _object(event)["op"])
        for case in FROZEN_CASES
        for trace in map(_object, _array(case["traces"]))
        for event in _array(trace["events"])
    }
    assert observed_events == set(reference.ALPHABET)


def _rename_mappings(case: Object) -> dict[str, dict[str, str]]:
    declaration = _object(case["declaration"])
    labels = sorted(_label(_object(node)["id"]) for node in _array(declaration["nodes"]))
    if len(labels) == 1:
        nodes = {labels[0]: labels[0]}
    else:
        nodes = {label: labels[(index + 1) % len(labels)] for index, label in enumerate(labels)}
    text = json.dumps((declaration, case["replacement"]), sort_keys=True)
    mappings = {"node": nodes}
    for role, order in reference.SEMANTIC_ORDERS.items():
        present = [label for label in order if f'"{label}"' in text]
        mappings[role] = (
            {label: present[(index + 1) % len(present)] for index, label in enumerate(present)}
            if len(present) > 1
            else {label: label for label in present}
        )
    return mappings


def _mapped(value: str, role: str, mappings: dict[str, dict[str, str]]) -> str:
    return mappings.get(role, {}).get(value, value)


def _renamed(value: Json, mappings: dict[str, dict[str, str]], *, parent_key: str = "") -> Json:
    if isinstance(value, str):
        if parent_key == "label":
            return _mapped(value, "node", mappings)
        if parent_key in {
            "port",
            "subject_port",
            "identity_input",
            "output",
            "inputs",
            "produced_ports",
            "consumed_ports",
        }:
            return _mapped(_mapped(value, "input_port", mappings), "output_port", mappings)
        if parent_key == "meaning":
            return _mapped(_mapped(value, "context", mappings), "evidence", mappings)
        if parent_key == "capability":
            return _mapped(value, "model", mappings)
        return value
    if isinstance(value, list):
        return [_renamed(item, mappings, parent_key=parent_key) for item in value]
    if isinstance(value, dict):
        result = {key: _renamed(item, mappings, parent_key=key) for key, item in value.items()}
        name = value.get("name")
        if isinstance(name, str):
            if "artifact_type" in value:
                result["name"] = _mapped(_mapped(name, "input_port", mappings), "output_port", mappings)
            elif value.get("kind") in {"field", "source_view", "evaluation", "absence"}:
                result["name"] = _mapped(name, "coverage", mappings)
            elif value.get("kind") in {"read", "write"}:
                result["name"] = _mapped(name, "state", mappings)
            elif "consumed_ports" in value and "subject_port" in value:
                result["name"] = _mapped(name, "evidence", mappings)
        return result
    return value


def test_bijective_renaming_and_declaration_permutations_preserve_edges() -> None:
    for case in FROZEN_CASES:
        traces = {_object(trace)["transformation"]: _object(trace) for trace in _array(case["traces"])}
        rename = traces["rename"]
        mappings = _rename_mappings(case)
        assert all(len(mapping) == len(set(mapping.values())) for mapping in mappings.values())
        assert rename["declaration"] == _renamed(case["declaration"], mappings), case["case_id"]
        assert rename["replacement"] == _renamed(case["replacement"], mappings), case["case_id"]
        nodes = _array(_object(case["declaration"])["nodes"])
        if len(nodes) >= 2:
            reverse = traces["reverse_declaration_tuple"]
            reversed_declaration = deepcopy(_object(case["declaration"]))
            reversed_declaration["nodes"] = list(reversed(_array(reversed_declaration["nodes"])))
            assert reverse["declaration"] == reversed_declaration, case["case_id"]
            assert _object(reverse["declaration"])["sequence"] == _object(case["declaration"])["sequence"]


def _assert_closed_vocabularies(declaration: Object) -> None:
    operations = [_object(declaration["interface"])]
    for node in map(_object, _array(declaration["nodes"])):
        assert node["kind"] in {"operation", "subgraph"}
        operations.append(_object(node["operation"]))
        if node["kind"] == "subgraph":
            _assert_closed_vocabularies(_object(node["body"]))
    for operation in operations:
        for outcome in map(_object, _array(operation["outcomes"])):
            assert outcome["category"] in {"success", "failure", "cancelled", "lost", "blocked", "inconsistent"}
            for context in map(_object, _array(outcome["context"])):
                assert context["capture"] == "whole_artifact"
            for evidence in map(_object, _array(outcome["evidence"])):
                for coverage in map(_object, _array(evidence["coverage"])):
                    assert coverage["kind"] in {"field", "source_view", "evaluation", "absence"}
            for effect in map(_object, _array(outcome["state_effects"])):
                assert effect["kind"] in {"read", "write"}
    for binding in map(_object, _array(declaration["input_bindings"])):
        assert _object(binding["source"])["kind"] in {"workflow_input", "node_output"}
    for binding in map(_object, _array(declaration["output_bindings"])):
        assert _object(binding["source"])["kind"] == "node_output"
    for requirement in map(_object, _array(declaration["protection"])):
        for coverage in map(_object, _array(requirement["coverage"])):
            assert coverage["kind"] in {"field", "source_view", "evaluation", "absence"}


def test_base_trace_and_replacement_closed_vocabularies_are_independently_valid() -> None:
    valid_codes = {
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
    }
    for case in FROZEN_CASES:
        assert case["mode"] in {"admission", "substitution"}
        _assert_closed_vocabularies(_object(case["declaration"]))
        if case["replacement"] is not None:
            _assert_closed_vocabularies(_object(case["replacement"]))
        expected = _object(case["expected"])
        assert expected["status"] in {"accepted", "rejected"}
        assert expected["code"] is None or expected["code"] in valid_codes
        for trace in map(_object, _array(case["traces"])):
            _assert_closed_vocabularies(_object(trace["declaration"]))
            if trace["replacement"] is not None:
                _assert_closed_vocabularies(_object(trace["replacement"]))
            assert all(_object(event)["op"] in set(reference.ALPHABET) for event in _array(trace["events"]))


def test_subgraphs_and_substitutions_keep_separate_owners() -> None:
    for case in FROZEN_CASES:
        declaration = _object(case["declaration"])
        for node in map(_object, _array(declaration["nodes"])):
            if node["kind"] == "subgraph":
                body = _object(node["body"])
                assert body["workflow"] != declaration["workflow"]
                assert all(_object(child)["id"] != node["id"] for child in _array(body["nodes"]))
                assert _object(reference.judge(body))["status"] == "accepted"
        if case["mode"] == "substitution":
            replacement = _object(case["replacement"])
            assert replacement["workflow"] != declaration["workflow"]
            assert _object(reference.judge(replacement))["status"] == "accepted"
            if _object(case["expected"])["status"] == "accepted":
                normalized = _object(_object(case["expected"])["normalized"])
                substituted = _object(_array(normalized["nodes"])[0])
                assert substituted["kind"] == "subgraph"
                assert substituted["body"] == replacement
                assert normalized["expanded_node_count"] == 2


def test_narrow_substitutions_and_rename_traces_retain_target_operation() -> None:
    narrow_cases = [
        case for case in FROZEN_CASES if case["family"] == "substitution" and _mutation(case).startswith("narrow_")
    ]
    assert len(narrow_cases) == 7
    for case in narrow_cases:
        declaration = _object(case["declaration"])
        target = _object(declaration["substitution_target"])
        target_node = next(_object(node) for node in _array(declaration["nodes"]) if _object(node)["id"] == target)
        replacement = _object(case["replacement"])
        substituted = _object(_array(_object(_object(case["expected"])["normalized"])["nodes"])[0])
        assert substituted["operation"] == target_node["operation"]
        assert substituted["operation"] != replacement["interface"]
        assert substituted["body"] == replacement

        rename = next(
            _object(trace) for trace in _array(case["traces"]) if _object(trace)["transformation"] == "rename"
        )
        renamed_declaration = _object(rename["declaration"])
        renamed_target = _object(renamed_declaration["substitution_target"])
        renamed_target_node = next(
            _object(node) for node in _array(renamed_declaration["nodes"]) if _object(node)["id"] == renamed_target
        )
        renamed_replacement = _object(rename["replacement"])
        renamed_substituted = _object(_array(_object(_object(rename["expected"])["normalized"])["nodes"])[0])
        assert renamed_substituted["operation"] == renamed_target_node["operation"]
        assert renamed_substituted["operation"] != renamed_replacement["interface"]
        assert renamed_substituted["body"] == renamed_replacement


def _assert_symmetric(left: Object, right: Object, expected: bool) -> None:
    assert reference.independent(left, right) is expected
    assert reference.independent(right, left) is expected


def _event(operation: str) -> Object:
    return {"op": operation}


def test_conditional_symmetric_independence_positive_and_negative_pairs() -> None:
    node_a: Object = {"node": {"label": "N0", "owner": "W0"}, "op": "declare_node"}
    node_b: Object = {"node": {"label": "N1", "owner": "W0"}, "op": "declare_subgraph"}
    _assert_symmetric(node_a, node_b, True)
    _assert_symmetric(node_a, deepcopy(node_a), False)

    binding_a: Object = {"destination": {"node": "N0", "port": "i0"}, "op": "bind_input"}
    binding_b: Object = {"destination": {"node": "N1", "port": "i0"}, "op": "bind_input"}
    _assert_symmetric(binding_a, binding_b, True)
    _assert_symmetric(binding_a, deepcopy(binding_a), False)
    output_binding: Object = {"destination": {"port": "o0"}, "op": "bind_output"}
    _assert_symmetric(binding_a, output_binding, True)
    same_destination_output: Object = {"destination": binding_a["destination"], "op": "bind_output"}
    _assert_symmetric(binding_a, same_destination_output, False)

    edge_a: Object = {"edge": {"before": "N0", "after": "N1"}, "op": "add_sequence"}
    edge_b: Object = {"edge": {"before": "N0", "after": "N2"}, "op": "add_sequence"}
    _assert_symmetric(edge_a, edge_b, True)
    _assert_symmetric(edge_a, deepcopy(edge_a), False)

    choice_a: Object = {
        "choice": {"selector": "N0", "branches": [{"outcomes": ["ok"], "members": ["N1"]}]},
        "op": "add_choice",
    }
    choice_b: Object = {
        "choice": {"selector": "N2", "branches": [{"outcomes": ["ok"], "members": ["N3"]}]},
        "op": "add_choice",
    }
    _assert_symmetric(choice_a, choice_b, True)
    same_selector = deepcopy(choice_b)
    _object(same_selector["choice"])["selector"] = "N0"
    _assert_symmetric(choice_a, same_selector, False)
    shared_member = deepcopy(choice_b)
    _object(_array(_object(shared_member["choice"])["branches"])[0])["members"] = ["N1"]
    _assert_symmetric(choice_a, shared_member, False)
    selector_is_member = deepcopy(choice_b)
    _object(_array(_object(selector_is_member["choice"])["branches"])[0])["members"] = ["N0"]
    _assert_symmetric(choice_a, selector_is_member, False)

    requirement_a: Object = {
        "requirement": {"outcome": "ok", "meaning": "m", "subject_port": "o0"},
        "op": "declare_protection",
    }
    requirement_b: Object = {
        "requirement": {"outcome": "fail", "meaning": "m", "subject_port": "o0"},
        "op": "declare_protection",
    }
    _assert_symmetric(requirement_a, requirement_b, True)
    _assert_symmetric(requirement_a, deepcopy(requirement_a), False)
    for operation in ("new_workflow", "declare_interface", "admit", "substitute"):
        for other in reference.ALPHABET:
            _assert_symmetric(_event(operation), _event(other), False)


def _find(family: str, mutation: str) -> Object:
    return next(case for case in FROZEN_CASES if case["family"] == family and _mutation(case) == mutation)


def test_lineage_zero_singleton_multiple_and_dependency_witnesses() -> None:
    pipe = _find("lineage", "PIPE_L")
    fanin = _find("lineage", "dependency_fanin")
    assert _object(pipe["expected"])["status"] == "accepted"
    normalized = _object(_object(pipe["expected"])["normalized"])
    interface = _object(normalized["interface"])
    ok = next(_object(value) for value in _array(interface["outcomes"]) if _object(value)["name"] == "ok")
    assert {cast(str, _object(value)["meaning"]) for value in _array(ok["context"])} == {"context", "context_alt"}
    assert {cast(str, _object(value)["meaning"]) for value in _array(ok["evidence"])} == {
        "assessment",
        "assessment_alt",
    }
    assert _object(fanin["expected"])["status"] == "accepted"
    assert _object(_find("lineage", "zero_upstream")["expected"])["code"] == "missing"
    assert _object(_find("lineage", "zero_downstream")["expected"])["code"] == "missing"
    assert _object(_find("lineage", "many_downstream")["expected"])["code"] == "contradictory"


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


def test_manifest_has_exact_schema_hashes_counts_and_provenance() -> None:
    assert set(MANIFEST) == {
        "schema_version",
        "manifest_version",
        "packet_id",
        "capability",
        "contract_sha256",
        "corpus_path",
        "corpus_sha256",
        "counts",
        "family_bounds",
        "alphabet",
        "independence",
        "generator_sha256",
        "self_test_sha256",
        "generation_provenance",
    }
    assert MANIFEST["schema_version"] == 1
    assert MANIFEST["manifest_version"] == "workflow-static-reference-v1"
    assert MANIFEST["packet_id"] == "R1a"
    assert MANIFEST["capability"] == "workflow_static_v1"
    assert MANIFEST["contract_sha256"] == reference.CONTRACT_SHA256
    assert MANIFEST["corpus_path"] == reference.CORPUS_PATH
    assert MANIFEST["corpus_sha256"] == hashlib.sha256(FROZEN_BYTES).hexdigest()
    assert MANIFEST["generator_sha256"] == hashlib.sha256(GENERATOR_PATH.read_bytes()).hexdigest()
    assert MANIFEST["self_test_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert MANIFEST["counts"] == reference.counts(FROZEN_CASES)
    assert MANIFEST["alphabet"] == list(reference.ALPHABET)
    assert MANIFEST["independence"] == {"kind": "conditional-symmetric-v1", "rule_ids": list(reference.RULE_IDS)}
    assert MANIFEST["family_bounds"] == {
        "family_ids": list(reference.FAMILY_IDS),
        "topology_node_counts": [1, 2, 3],
        "expanded_node_count_max": 3,
        "subgraph_depth_max": 2,
        "choice_state_max": 4,
    }
    provenance = _object(MANIFEST["generation_provenance"])
    assert set(provenance) == {"tools", "generations", "byte_identical"}
    assert provenance["generations"] == 2
    assert provenance["byte_identical"] is True
    tools = _object(provenance["tools"])
    assert set(tools) == {"python", "generator", "self_test"}
    assert all(isinstance(value, str) and value for value in tools.values())


def test_clean_subprocess_denies_product_imports_and_generates() -> None:
    tree = ast.parse(GENERATOR_PATH.read_text())
    imported = {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)} | {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    }
    assert not any(name == "anonymizer" or name.startswith("anonymizer.") for name in imported)
    script = """
import builtins
import importlib.util
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
original_import = builtins.__import__
def deny(name, *args, **kwargs):
    if name == "anonymizer" or name.startswith("anonymizer."):
        raise AssertionError("product import denied")
    return original_import(name, *args, **kwargs)
builtins.__import__ = deny
spec = importlib.util.spec_from_file_location("workflow_static_reference_probe", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
cases = module.generate_cases()
assert len(cases) > 0
assert module.judge(cases[0]["declaration"]) == cases[0]["expected"]
assert module.module_is_independent()
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(GENERATOR_PATH)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
