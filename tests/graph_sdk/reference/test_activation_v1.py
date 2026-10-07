# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-tests for the frozen finite workflow-activation reference."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import TypeAlias, cast

import pytest

from tests.graph_sdk.reference import activation_v1 as reference

Json: TypeAlias = reference.Json
Object: TypeAlias = reference.Object

REFERENCE_DIR = Path(__file__).parent
GENERATOR_PATH = REFERENCE_DIR / "activation_v1.py"
CORPUS_PATH = REFERENCE_DIR / "activation_v1_cases.json"
MANIFEST_PATH = REFERENCE_DIR / "activation_v1_manifest.json"
FROZEN_BYTES = CORPUS_PATH.read_bytes()
FROZEN_CASES = reference.load_cases(json.loads(FROZEN_BYTES))
MANIFEST = cast(Object, json.loads(MANIFEST_PATH.read_bytes()))


@pytest.fixture(scope="module")
def generations() -> tuple[tuple[Object, ...], tuple[Object, ...]]:
    """Generate exactly twice and share both results."""
    return reference.generate_cases(), reference.generate_cases()


def _object(value: Json) -> Object:
    assert isinstance(value, dict)
    return cast(Object, value)


def _array(value: Json) -> list[Json]:
    assert isinstance(value, list)
    return value


def _case(case_id: str) -> Object:
    return next(case for case in FROZEN_CASES if case["case_id"] == case_id)


def _cases(family: str) -> tuple[Object, ...]:
    return tuple(case for case in FROZEN_CASES if case["family"] == family)


def _canonical(value: Json) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def test_frozen_generation_is_byte_identical(
    generations: tuple[tuple[Object, ...], tuple[Object, ...]],
) -> None:
    first, second = generations
    assert reference.canonical_bytes(first) == reference.canonical_bytes(second) == FROZEN_BYTES
    assert hashlib.sha256(FROZEN_BYTES).hexdigest() == MANIFEST["corpus_sha256"]


def test_every_frozen_result_and_metamorphic_trace_is_reduced_independently() -> None:
    for case in FROZEN_CASES:
        declaration = _object(case["declaration"])
        events = list(map(_object, _array(case["events"])))
        assert reference.reduce_trace(declaration, events) == case["expected"], case["case_id"]
        for trace_value in _array(case["traces"]):
            trace = _object(trace_value)
            trace_events = list(map(_object, _array(trace["events"])))
            if trace["name"] == "rename":
                renamed = cast(Object, reference._rename(declaration))
                actual = reference.reduce_trace(renamed, trace_events)
            else:
                actual = reference.reduce_trace(declaration, trace_events)
            assert actual == trace["expected"], case["case_id"]
            state = _object(actual)["state"]
            base_state = _object(case["expected"])["state"]
            if trace["name"] == "commute_independent_siblings":
                assert trace["expected"] == case["expected"]
                assert state == base_state
                assert hash(_canonical(cast(Json, state))) == hash(_canonical(cast(Json, base_state)))
            else:
                renamed_back = reference._rename(cast(Json, state))
                assert _object(cast(Json, renamed_back))["complete"] == _object(cast(Json, base_state))["complete"]
                assert len(_array(_object(cast(Json, renamed_back))["entries"])) == len(
                    _array(_object(cast(Json, base_state))["entries"])
                )


def test_family_order_coordinates_and_payloads_are_exact() -> None:
    seen = tuple(dict.fromkeys(cast(str, case["family"]) for case in FROZEN_CASES))
    assert seen == reference.FAMILY_IDS
    identifiers = [cast(str, case["case_id"]) for case in FROZEN_CASES]
    payloads = [_canonical({key: value for key, value in case.items() if key != "case_id"}) for case in FROZEN_CASES]
    assert len(identifiers) == len(set(identifiers))
    assert len(payloads) == len(set(payloads))
    for family in reference.FAMILY_IDS:
        coordinates = [cast(str, case["case_id"]).split("/")[1] for case in _cases(family)]
        assert len(coordinates) == len(set(coordinates))


def test_all_twelve_normative_family_loops_have_the_required_cardinality() -> None:
    family_counts = Counter(cast(str, case["family"]) for case in FROZEN_CASES)
    assert family_counts == {
        "sequence_single": 6,
        "sequence_linked_pair": 36,
        "sequence_independent_siblings": 36,
        "sequence_mutations": 10,
        "choice": 8,
        "subgraph": 10,
        "map": 25,
        "join": 26,
        "loop": 16,
        "nested_map_loop": 11,
        "precedence": 7,
        "terminal_coverage": 15,
    }
    sibling_traces = _cases("sequence_independent_siblings")
    assert all(
        [_object(value)["name"] for value in _array(case["traces"])] == ["rename", "commute_independent_siblings"]
        for case in sibling_traces
    )
    assert all(
        [_object(value)["name"] for value in _array(case["traces"])] == ["rename"]
        for case in _cases("sequence_single") + _cases("sequence_linked_pair")
    )


def test_nested_coordinate_table_reaches_twelve_distinct_keys() -> None:
    table: dict[tuple[int, int], int] = {}
    for case in _cases("nested_map_loop")[:9]:
        declaration = _object(case["declaration"])
        state = _object(_object(case["expected"])["state"])
        entries = _array(state["entries"])
        coordinate = (cast(int, declaration["map_children"]), cast(int, declaration["loop_iterations"]))
        table[coordinate] = len(entries)
        assert len({cast(str, _object(entry)["activation"]) for entry in entries}) == len(entries)
    assert table == {(m, i): 2 + m * (3 + i) for m in (0, 1, 2) for i in (0, 1, 2)}
    assert table[(2, 2)] == 12


def test_bounds_zero_one_two_and_one_over_three_are_observed() -> None:
    map_bounds = {cast(int, _object(case["declaration"]).get("bound", -1)) for case in _cases("map")}
    loop_bounds = {cast(int, _object(case["declaration"]).get("bound", -1)) for case in _cases("loop")}
    assert {0, 1, 2} <= map_bounds
    assert {0, 1, 2} <= loop_bounds
    map_one_over = next(case for case in _cases("map") if cast(str, case["case_id"]).endswith("/one_over_3"))
    assert _object(map_one_over["declaration"])["observed_count"] == 3
    assert any("one_over_3" in cast(str, case["case_id"]) for case in _cases("loop"))


def test_zero_bound_enter_reserves_no_member_and_closes_inconsistent() -> None:
    case = next(case for case in _cases("loop") if cast(str, case["case_id"]).endswith("/enter_bound_zero"))
    expected = _object(case["expected"])
    state = _object(expected["state"])
    expansion = _object(_array(state["expansions"])[0])
    assert expansion == {"members": [], "parent": "A0", "status": "overflow"}
    assert expected["join_status"] == "inconsistent"


def test_subgraph_closure_is_derived_and_premature_parent_rejects() -> None:
    ordinary = _case("subgraph/000/body_1_success")
    assert _object(ordinary["expected"])["derived_parent"] is True
    assert _object(_case("subgraph/004/premature_parent_terminal")["expected"])["code"] == "contradictory"
    abnormal = next(case for case in _cases("subgraph") if cast(str, case["case_id"]).endswith("/abnormal_sink_loss"))
    entries = _array(_object(_object(abnormal["expected"])["state"])["entries"])
    assert all(_object(entry)["outcome"] is None for entry in entries)


def test_abnormal_none_outcomes_cover_all_five_non_success_categories() -> None:
    abnormal = [case for case in _cases("sequence_mutations") if "/abnormal_" in cast(str, case["case_id"])]
    assert {cast(str, case["case_id"]).rsplit("_", 1)[1] for case in abnormal} == set(reference.CATEGORIES[1:])
    for case in abnormal:
        expected = _object(case["expected"])
        state = _object(expected["state"])
        assert state["outputs"] == []
        assert _object(_array(state["entries"])[0])["outcome"] is None
    assert _object(_case("sequence_mutations/009/success_without_outcome")["expected"])["code"] == "invalid_value"


def test_exact_seven_precedence_boundaries() -> None:
    cases = _cases("precedence")
    assert [cast(str, case["case_id"]).rsplit("/", 1)[1] for case in cases] == [
        "overflow_type_before_value",
        "event_limit_before_foreign",
        "foreign_before_duplicate",
        "duplicate_before_missing",
        "missing_before_contradictory",
        "overlap_before_cycle",
        "cycle_before_contradictory",
    ]
    assert [_object(case["expected"])["code"] for case in cases] == [
        "invalid_type",
        "limit_exceeded",
        "foreign_owner",
        "duplicate",
        "missing",
        "overlap",
        "cycle",
    ]
    assert [_object(case["declaration"])["boundary"] for case in cases][-2:] == ["static_admission", "static_admission"]


def test_all_by_key_is_conjunction_over_closed_exact_membership() -> None:
    accepted_empty = _case("join/000/children_0_empty")
    assert _object(accepted_empty["expected"])["join_status"] == "ready"
    mixed = next(case for case in _cases("join") if "children_2_success-failure" in cast(str, case["case_id"]))
    assert _object(mixed["expected"])["join_status"] == "blocked"
    open_survivors = next(
        case for case in _cases("join") if cast(str, case["case_id"]).endswith("/open_complete_survivors")
    )
    assert _object(open_survivors["expected"])["join_status"] == "unstarted"


def test_failed_map_closure_preserves_empty_and_partial_members() -> None:
    failed_empty = next(case for case in _cases("map") if cast(str, case["case_id"]).endswith("/failed_empty"))
    failed_partial = next(case for case in _cases("map") if cast(str, case["case_id"]).endswith("/failed_partial"))
    for case in (failed_empty, failed_partial):
        expected = _object(case["expected"])
        assert expected["aggregate"] == "failed"
        assert expected["join_status"] == "blocked"
    empty_expansion = _object(_array(_object(_object(failed_empty["expected"])["state"])["expansions"])[0])
    partial_expansion = _object(_array(_object(_object(failed_partial["expected"])["state"])["expansions"])[0])
    assert empty_expansion["members"] == []
    assert partial_expansion["members"] == ["A1"]


def test_whole_state_completion_requires_terminal_children_and_join() -> None:
    missing_terminal = next(
        case for case in _cases("map") if cast(str, case["case_id"]).endswith("/closed_missing_terminal")
    )
    assert _object(_object(missing_terminal["expected"])["state"])["complete"] is False
    ready_join = _case("join/000/children_0_empty")
    assert _object(_object(ready_join["expected"])["state"])["complete"] is False
    failed_partial = next(case for case in _cases("map") if cast(str, case["case_id"]).endswith("/failed_partial"))
    assert _object(_object(failed_partial["expected"])["state"])["complete"] is True


def test_settlement_capacity_reserves_real_completion_and_map_closure() -> None:
    assert reference.initial_capacity(1) == (1, 3)
    assert reference.initial_capacity(3, 1) == (3, 10)
    for undersized in (0, 1, 2):
        assert undersized < reference.initial_capacity(1)[1]
    assert reference.completion_reserve(absent=1) == 3
    assert reference.completion_reserve(unstarted_or_ready=1) == 2
    assert reference.completion_reserve(running_ordinary=1) == 1
    assert reference.completion_reserve(open_map_expanders=1) == 1
    assert reference.completion_reserve(running_ordinary=1) + 2 == 3
    with pytest.raises(TypeError):
        reference.initial_capacity(True)


def test_alphabet_universe_independence_and_manifest_are_frozen() -> None:
    observed = {cast(str, _object(event)["kind"]) for case in FROZEN_CASES for event in _array(case["events"])}
    assert observed == set(reference.ALPHABET)
    expected_keys = {
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
    assert set(MANIFEST) == expected_keys
    assert MANIFEST["schema_version"] == 1
    assert MANIFEST["manifest_version"] == "workflow-activation-reference-v1"
    assert MANIFEST["packet_id"] == "R1b"
    assert MANIFEST["capability"] == "workflow_activation_v1"
    assert MANIFEST["contract_sha256"] == reference.CONTRACT_SHA256
    assert MANIFEST["alphabet"] == list(reference.ALPHABET)
    assert MANIFEST["independence"] == {"kind": "conditional-symmetric-v1", "rule_ids": list(reference.RULE_IDS)}
    assert MANIFEST["counts"] == reference.counts(FROZEN_CASES)
    counts = _object(MANIFEST["counts"])
    assert counts["max_templates"] == 3
    assert counts["max_activations"] == 12
    assert counts["max_map_children"] == counts["max_loop_iterations"] == 3
    assert counts["max_dynamic_depth"] == 2


def test_manifest_hashes_exact_sources_and_corpus() -> None:
    assert MANIFEST["corpus_sha256"] == hashlib.sha256(FROZEN_BYTES).hexdigest()
    assert MANIFEST["generator_sha256"] == hashlib.sha256(GENERATOR_PATH.read_bytes()).hexdigest()
    assert MANIFEST["self_test_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    provenance = _object(MANIFEST["generation_provenance"])
    assert set(provenance) == {"tools", "generations", "byte_identical"}
    assert provenance["generations"] == 2
    assert provenance["byte_identical"] is True
    assert set(_object(provenance["tools"])) == {"python", "generator", "self_test"}


def test_reference_source_uses_only_standard_library_and_has_reviewable_signatures() -> None:
    tree = ast.parse(GENERATOR_PATH.read_text())
    imports = {
        alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    }
    imports.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert imports <= set(sys.stdlib_module_names) | {"__future__"}
    functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for name in (
        "reduce_trace",
        "generate_cases",
        "canonical_bytes",
        "counts",
        "manifest",
        "initial_capacity",
        "completion_reserve",
    ):
        function = functions[name]
        assert function.returns is not None
        assert all(argument.annotation is not None for argument in function.args.args + function.args.kwonlyargs)
    source = GENERATOR_PATH.read_text()
    assert "anonymizer" not in source
    assert "donor" not in source.lower()


def test_clean_subprocess_denies_non_reference_imports() -> None:
    script = f"""
import importlib.abc
import importlib.util
import pathlib
import sys

class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'anonymizer' or fullname.startswith('anonymizer.') or 'donor' in fullname.lower():
            raise AssertionError(fullname)
        return None

sys.meta_path.insert(0, Deny())
path = pathlib.Path({str(GENERATOR_PATH)!r})
spec = importlib.util.spec_from_file_location('activation_reference_probe', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
first = module.canonical_bytes(module.generate_cases())
second = module.canonical_bytes(module.generate_cases())
assert first == second
assert module.reduce_trace({{'scenario': 'sequence', 'templates': {{'A0': 'N0'}}, 'dependencies': []}}, [{{'kind': 'initialize'}}, {{'kind': 'select', 'activation': 'A0'}}, {{'kind': 'start', 'activation': 'A0'}}, {{'kind': 'terminal_success', 'activation': 'A0', 'outcome': 'ok'}}])['status'] == 'accepted'
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_adapter_exception_is_not_a_reference_verdict() -> None:
    spec = importlib.util.spec_from_file_location("missing-reference-adapter", REFERENCE_DIR / "absent.py")
    assert spec is not None
    with pytest.raises(FileNotFoundError):
        assert spec.loader is not None
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
