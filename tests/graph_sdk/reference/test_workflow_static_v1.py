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


def _independent_expected(family: str, mutation: str, declaration: Object) -> Object:
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
    eligible, unmet = _independent_protection(declaration)
    return {
        "code": None,
        "normalized": _independent_normalized(declaration),
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
        expected = _independent_expected(family, mutation, _object(case["declaration"]))
        assert case["expected"] == expected, case["case_id"]
        for trace in map(_object, _array(case["traces"])):
            trace_expected = _independent_expected(family, mutation, _object(trace["declaration"]))
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


def _rename_mapping(case: Object) -> dict[str, str]:
    declaration = _object(case["declaration"])
    labels = sorted(_label(_object(node)["id"]) for node in _array(declaration["nodes"]))
    if len(labels) == 1:
        mapping = {labels[0]: labels[0]}
    else:
        mapping = {label: labels[(index + 1) % len(labels)] for index, label in enumerate(labels)}
    text = json.dumps((declaration, case["replacement"]), sort_keys=True)
    for order in reference.SEMANTIC_ORDERS.values():
        present = [label for label in order if f'"{label}"' in text]
        if len(present) > 1:
            mapping.update({label: present[(index + 1) % len(present)] for index, label in enumerate(present)})
    return mapping


def _renamed(value: Json, mapping: dict[str, str]) -> Json:
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [_renamed(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: _renamed(item, mapping) for key, item in value.items()}
    return value


def test_bijective_renaming_and_declaration_permutations_preserve_edges() -> None:
    for case in FROZEN_CASES:
        traces = {_object(trace)["transformation"]: _object(trace) for trace in _array(case["traces"])}
        rename = traces["rename"]
        mapping = _rename_mapping(case)
        assert len(mapping) == len(set(mapping.values()))
        assert rename["declaration"] == _renamed(case["declaration"], mapping), case["case_id"]
        assert rename["replacement"] == _renamed(case["replacement"], mapping), case["case_id"]
        nodes = _array(_object(case["declaration"])["nodes"])
        if len(nodes) >= 2:
            reverse = traces["reverse_declaration_tuple"]
            reversed_declaration = deepcopy(_object(case["declaration"]))
            reversed_declaration["nodes"] = list(reversed(_array(reversed_declaration["nodes"])))
            assert reverse["declaration"] == reversed_declaration, case["case_id"]
            assert _object(reverse["declaration"])["sequence"] == _object(case["declaration"])["sequence"]


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


def _mutant_result(name: str) -> tuple[Object, Object]:
    selectors = {
        "erase_port_type": ("ports", "input_type_A1"),
        "erase_output_dependency": ("ports", "remove_output_dependency"),
        "infer_identity_from_dependency": ("lineage", "zero_upstream"),
        "erase_outcome": ("choice", "unknown_outcome"),
        "erase_context": ("lineage", "PIPE_L"),
        "erase_evidence_meaning": ("lineage", "PIPE_L"),
        "erase_coverage": ("lineage", "PIPE_L"),
        "widen_state": ("subgraph", "equal"),
        "change_model": ("subgraph", "equal"),
        "widen_resource": ("subgraph", "widen_max_activations_by_1"),
        "overlap_choice": ("choice", "overlap_member"),
        "reverse_sequence": ("lineage", "PIPE_L"),
        "trust_python_type": ("ports", "internal_type_A1"),
    }
    family, mutation = selectors[name]
    case = _find(family, mutation)
    expected = _independent_expected(family, mutation, _object(case["declaration"]))
    mutant = deepcopy(expected)
    if _object(mutant)["status"] == "rejected":
        mutant = {
            "code": None,
            "normalized": _independent_normalized(_object(case["declaration"])),
            "protection_eligible_outcomes": [],
            "status": "accepted",
            "topology": None,
            "unmet_protection": [],
        }
    else:
        normalized = _object(_object(mutant)["normalized"])
        if name == "reverse_sequence":
            edge = _object(_array(normalized["sequence"])[0])
            edge["before"], edge["after"] = edge["after"], edge["before"]
        else:
            interface = _object(normalized["interface"])
            outcome = _object(_array(interface["outcomes"])[0])
            if name == "erase_context":
                outcome["context"] = []
            elif name == "erase_evidence_meaning":
                _object(_array(outcome["evidence"])[0])["meaning"] = ""
            elif name == "erase_coverage":
                _object(_array(outcome["evidence"])[0])["coverage"] = []
            elif name == "widen_state":
                outcome["state_effects"] = [{"kind": "write", "name": "state"}]
            elif name == "change_model":
                outcome["model_requirements"] = [{"capability": "model", "revision": 2}]
            else:
                raise AssertionError(f"unsupported accepted mutant: {name}")
    return expected, mutant


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
    expected, mutant = _mutant_result(name)
    assert mutant != expected, f"semantic mutant survived: {name}"


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
