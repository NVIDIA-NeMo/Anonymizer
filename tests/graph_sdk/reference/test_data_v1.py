# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-tests for the frozen finite data and record reference."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Callable, TypeAlias

import pytest

from tests.graph_sdk.reference import data_v1

Mutant: TypeAlias = Callable[[data_v1.FixtureCase], data_v1.ValidationResult]

REFERENCE_DIR = Path(__file__).parent
CORPUS_PATH = REFERENCE_DIR / "data_v1_cases.json"
MANIFEST_PATH = REFERENCE_DIR / "data_v1_manifest.json"
MANIFEST = data_v1._as_object(json.loads(MANIFEST_PATH.read_text()))
MANIFEST_COUNTS = data_v1._as_object(MANIFEST["counts"])
MANIFEST_BASE_COUNTS = data_v1._as_object(MANIFEST["base_case_counts"])
MANIFEST_MAXIMA = data_v1._as_object(MANIFEST["maxima"])
MANIFEST_ALPHABET = data_v1._as_object(MANIFEST["alphabet"])
MANIFEST_INDEPENDENCE = data_v1._as_object(MANIFEST["independence"])
MANIFEST_NAMED_WITNESSES = tuple(data_v1._string(value) for value in data_v1._as_list(MANIFEST["named_witnesses"]))
FROZEN_BYTES = CORPUS_PATH.read_bytes()
FROZEN_CASES = data_v1._parse_cases(json.loads(FROZEN_BYTES))


@pytest.fixture(scope="module")
def generations() -> tuple[tuple[data_v1.FixtureCase, ...], tuple[data_v1.FixtureCase, ...]]:
    """Generate exactly twice and share both results across freeze checks."""
    return data_v1.generate_cases(), data_v1.generate_cases()


def test_frozen_generation_is_reproducible(
    generations: tuple[tuple[data_v1.FixtureCase, ...], tuple[data_v1.FixtureCase, ...]],
) -> None:
    first, second = generations
    first_bytes = data_v1.canonical_bytes(first)
    second_bytes = data_v1.canonical_bytes(second)
    assert first_bytes == second_bytes == FROZEN_BYTES
    assert hashlib.sha256(first_bytes).hexdigest() == MANIFEST["corpus_sha256"]


def test_manifest_counts_and_bounds(
    generations: tuple[tuple[data_v1.FixtureCase, ...], tuple[data_v1.FixtureCase, ...]],
) -> None:
    cases, duplicate_generation = generations
    del duplicate_generation
    family_counts = Counter(case["family"] for case in cases)
    event_counts = Counter(event["op"] for case in cases for event in case["trace"])
    assert len(cases) == MANIFEST_COUNTS["deduplicated_cases"]
    assert data_v1._raw_variant_count() == MANIFEST_COUNTS["raw_variants"]
    assert dict(sorted(family_counts.items())) == MANIFEST_COUNTS["families"]
    assert sum(len(case["trace"]) for case in cases) == MANIFEST_COUNTS["events"]
    assert len(cases) == MANIFEST_COUNTS["traces"]
    assert dict(sorted(event_counts.items())) == MANIFEST_COUNTS["event_kinds"]
    assert _maximums(cases) == MANIFEST_MAXIMA
    base_counts = Counter(family for family, _, _ in data_v1._base_cases())
    assert base_counts["precedence"] == MANIFEST_BASE_COUNTS["precedence"] == 11
    assert base_counts["record-precedence"] == MANIFEST_BASE_COUNTS["record_precedence"] == 7
    assert sys.implementation.name == "cpython"
    assert MANIFEST["python"] == {
        "implementation": "CPython",
        "version": ".".join(str(part) for part in sys.version_info[:3]),
    }
    for case in cases:
        declaration = case["declaration"]
        expected = case["expected"]
        if declaration["kind"] == "data" and expected["verdict"] == "accept":
            counts = data_v1._raw_counts(data_v1._as_object(declaration))
            assert declaration["limits"] == counts, case["case_id"]


def test_every_frozen_expectation_matches_the_reference() -> None:
    for case in FROZEN_CASES:
        actual = data_v1.validate_case(case)
        assert actual == case["expected"], f"reference mismatch for {case['case_id']}"


def test_decoded_fixture_structure_is_runtime_validated() -> None:
    malformed = json.loads(json.dumps(FROZEN_CASES[0]))
    malformed["trace"][0]["value"] = 1.5
    with pytest.raises(ValueError, match="invalid JSON fixture structure"):
        data_v1._parse_cases([malformed])


def test_loader_rejects_unknown_operation() -> None:
    malformed = _mutable_case_with_event("declare_datum(id,text)")
    _mutable_trace(malformed)[0]["op"] = "bogus_operation"
    _assert_loader_rejects(malformed)


def test_loader_rejects_datum_event_missing_value() -> None:
    malformed = _mutable_case_with_event("declare_datum(id,text)")
    del _mutable_trace(malformed)[0]["value"]
    _assert_loader_rejects(malformed)


def test_loader_rejects_datum_event_with_unrelated_kind() -> None:
    malformed = _mutable_case_with_event("declare_datum(id,text)")
    _mutable_trace(malformed)[0]["kind"] = "contexts"
    _assert_loader_rejects(malformed)


def test_loader_rejects_unknown_relation_kind() -> None:
    malformed = _mutable_case_with_event("add_relation(kind,value)")
    event = next(event for event in _mutable_trace(malformed) if event["op"] == "add_relation(kind,value)")
    event["kind"] = "unknown_relation"
    _assert_loader_rejects(malformed)


@pytest.mark.parametrize("operation", ["close_declaration", "validate"])
def test_loader_rejects_payload_on_control_event(operation: str) -> None:
    malformed = _mutable_case_with_event("declare_datum(id,text)")
    event = next(event for event in _mutable_trace(malformed) if event["op"] == operation)
    event["value"] = None
    _assert_loader_rejects(malformed)


def test_loader_rejects_mixed_data_and_record_alphabets() -> None:
    data_case = _mutable_case_with_event("declare_datum(id,text)")
    data_trace = data_v1._as_list(data_case["trace"])
    data_trace[0] = {
        "op": "declare_record_fact(kind,value)",
        "kind": "invocation",
        "value": "I",
    }
    _assert_loader_rejects(data_case)

    record_case = _mutable_case_with_event("declare_record_fact(kind,value)")
    record_trace = data_v1._as_list(record_case["trace"])
    record_trace[0] = {
        "op": "declare_datum(id,text)",
        "value": {"id": ["local", 0], "text": "x"},
    }
    _assert_loader_rejects(record_case)


@pytest.mark.parametrize(
    "order",
    [
        ("validate", "close_declaration"),
        ("close_declaration", "close_declaration"),
    ],
)
def test_loader_rejects_invalid_close_validate_order(order: tuple[str, str]) -> None:
    malformed = _mutable_case_with_event("declare_datum(id,text)")
    trace = data_v1._as_list(malformed["trace"])
    trace[-2:] = [{"op": operation} for operation in order]
    _assert_loader_rejects(malformed)


def test_loader_rejects_malformed_operation_specific_envelope() -> None:
    malformed = _mutable_case_with_event("add_relation(kind,value)")
    event = next(event for event in _mutable_trace(malformed) if event["op"] == "add_relation(kind,value)")
    event["value"] = {"target": ["local", 0]}
    _assert_loader_rejects(malformed)


def test_loader_rejects_malformed_nested_datum_declaration() -> None:
    malformed = _mutable_case_with_event("declare_datum(id,text)")
    declaration = _mutable_declaration(malformed)
    datum = data_v1._as_object(data_v1._as_list(declaration["datums"])[0])
    del datum["text"]
    _assert_loader_rejects(malformed)


def test_loader_rejects_malformed_nested_relation_declaration() -> None:
    malformed = _mutable_case_with_event("add_relation(kind,value)")
    declaration = _mutable_declaration(malformed)
    relation_kind = next(key for key in data_v1.RELATION_KEYS if data_v1._as_list(declaration[key]))
    relation = data_v1._as_object(data_v1._as_list(declaration[relation_kind])[0])
    del relation[next(iter(data_v1.RELATION_FIELDS[relation_kind]))]
    _assert_loader_rejects(malformed)


def test_loader_rejects_unknown_result_code() -> None:
    malformed = json.loads(json.dumps(next(case for case in FROZEN_CASES if case["expected"]["verdict"] == "reject")))
    data_v1._as_object(malformed["expected"])["code"] = "unknown_code"
    _assert_loader_rejects(malformed)


def test_loader_rejects_unknown_family_and_record_dispatch_tag() -> None:
    unknown_family = _mutable_case_with_event("declare_datum(id,text)")
    unknown_family["family"] = "unknown_family"
    _assert_loader_rejects(unknown_family)

    unknown_boundary = _mutable_case_with_event("declare_record_fact(kind,value)")
    _mutable_declaration(unknown_boundary)["boundary"] = "unknown_boundary"
    _assert_loader_rejects(unknown_boundary)


def test_loader_rejects_record_fact_outside_constructor_boundary() -> None:
    malformed = _mutable_case_with_event("declare_record_fact(kind,value)")
    _mutable_trace(malformed)[0]["kind"] = "statuses"
    _assert_loader_rejects(malformed)


def test_loader_accepts_every_event_variant_and_semantic_invalid_values() -> None:
    operations = {event["op"] for case in FROZEN_CASES for event in case["trace"]}
    assert operations == {
        "declare_datum(id,text)",
        "select_target(id)",
        "add_relation(kind,value)",
        "declare_record_fact(kind,value)",
        "close_declaration",
        "validate",
    }
    relation_kinds = {
        event["kind"] for case in FROZEN_CASES for event in case["trace"] if event["op"] == "add_relation(kind,value)"
    }
    assert relation_kinds == set(data_v1.RELATION_KEYS)
    record_boundaries = {
        case["declaration"]["boundary"] for case in FROZEN_CASES if case["declaration"]["kind"] == "record"
    }
    assert record_boundaries == set(data_v1.RECORD_FACT_KEYS)
    semantic_invalid_labels = (
        "unknown-terminal-category",
        "unknown-reason",
        "unknown-completion",
        "unknown-qualification",
        "activation-bool-occurrence",
        "explicit-zero",
        "foreign_invocation",
    )
    for label in semantic_invalid_labels:
        case = _find(label, verdict="reject")
        assert data_v1._parse_cases(json.loads(json.dumps([case]))) == (case,)


def test_alphabet_is_exact_and_traces_close_before_validation() -> None:
    static = set(data_v1.STATIC_ALPHABET)
    record = set(data_v1.RECORD_ALPHABET)
    assert list(data_v1.STATIC_ALPHABET) == MANIFEST_ALPHABET["static"]
    assert list(data_v1.RECORD_ALPHABET) == MANIFEST_ALPHABET["record"]
    for case in FROZEN_CASES:
        trace = case["trace"]
        allowed = record if case["declaration"]["kind"] == "record" else static
        assert all(event["op"] in allowed for event in trace), case["case_id"]
        assert [event["op"] for event in trace[-2:]] == ["close_declaration", "validate"]


def test_independent_adjacent_additions_commute() -> None:
    pairs = list(data_v1._independence_witnesses(FROZEN_CASES))
    assertions = 0
    for case, index in pairs:
        declaration = case["declaration"]
        assert declaration["kind"] == "data"
        trace = deepcopy(case["trace"])
        left = trace[index]
        right = trace[index + 1]
        assert data_v1._independent(left, right, declaration)
        assert data_v1._independent(right, left, declaration)
        trace[index], trace[index + 1] = trace[index + 1], trace[index]
        swapped: data_v1.CaseInput = {"declaration": data_v1._parse_declaration(data_v1._replay_trace(case, trace))}
        assert data_v1.validate_case(swapped) == case["expected"], case["case_id"]
        assertions += 3
    assert any(case["expected"]["verdict"] == "reject" for case, _ in pairs)
    assert len(pairs) == MANIFEST_INDEPENDENCE["pair_count"]
    assert len(pairs) == MANIFEST_INDEPENDENCE["swapped_trace_count"]
    assert assertions == MANIFEST_INDEPENDENCE["assertion_count"]


def test_independence_classification_is_symmetric_and_contract_derived() -> None:
    raw_declaration = data_v1._data([(("local", 0), "x"), (("local", 1), "y")], [("local", 0), ("local", 1)])
    parsed_declaration = data_v1._parse_declaration(raw_declaration)
    assert parsed_declaration["kind"] == "data"
    declaration = parsed_declaration
    declaration["coherence"] = [data_v1._group_json([("local", 0)])]
    declaration["atomic"] = [data_v1._group_json([("local", 1)])]
    limits = data_v1._as_object(declaration["limits"])
    limits.update(data_v1._raw_counts(data_v1._as_object(declaration)))
    limits["max_declarations"] = 10
    limits["max_group_members"] = 10
    datum_0, datum_1, target_0, target_1, coherence, atomic = data_v1._make_trace(declaration)[:6]
    context_0: data_v1.RelationAdditionEvent = {
        "op": "add_relation(kind,value)",
        "kind": "contexts",
        "value": data_v1._context_json((("local", 0), ("local", 1), 0, 1)),
    }
    context_1: data_v1.RelationAdditionEvent = {
        "op": "add_relation(kind,value)",
        "kind": "contexts",
        "value": data_v1._context_json((("local", 1), ("local", 0), 0, 1)),
    }
    source_0: data_v1.RelationAdditionEvent = {
        "op": "add_relation(kind,value)",
        "kind": "source_relations",
        "value": data_v1._source_json((("local", 0), ("local", 1), 0, 1)),
    }
    source_1 = deepcopy(source_0)
    source_1["value"] = data_v1._source_json((("local", 0), ("local", 0), 0, 1))
    region_0: data_v1.RelationAdditionEvent = {
        "op": "add_relation(kind,value)",
        "kind": "output_regions",
        "value": data_v1._region_json((("local", 0), ("local", 0), 0, 1)),
    }
    region_1: data_v1.RelationAdditionEvent = {
        "op": "add_relation(kind,value)",
        "kind": "output_regions",
        "value": data_v1._region_json((("local", 1), ("local", 0), 0, 1)),
    }
    classifications = (
        (datum_0, datum_1, True, "distinct datum IDs"),
        (datum_0, target_1, True, "unreferenced datum and target"),
        (target_0, target_1, True, "distinct target memberships"),
        (context_0, context_1, True, "distinct read relations"),
        (coherence, atomic, True, "disjoint cross-kind groups"),
        (datum_0, deepcopy(datum_0), False, "duplicate datum ID"),
        (coherence, deepcopy(coherence), False, "overlapping group members"),
        (target_0, coherence, False, "target referenced by relation"),
        (source_0, source_1, False, "conflicting source declarations"),
        (region_0, region_1, False, "owned regions on one root source"),
    )
    for left, right, expected, label in classifications:
        assert data_v1._independent(left, right, declaration) is expected, label
        assert data_v1._independent(right, left, declaration) is expected, label

    rejected = deepcopy(declaration)
    rejected["dependencies"] = [data_v1._dependency_json((("local", 0), ("local", 99)))]
    data_v1._as_object(rejected["limits"]).update(data_v1._raw_counts(data_v1._as_object(rejected)))
    rejected_case: data_v1.CaseInput = {"declaration": rejected}
    assert data_v1.validate_case(rejected_case) == {"verdict": "reject", "code": "missing"}
    rejected_trace = data_v1._make_trace(rejected)
    rejected_coherence = next(event for event in rejected_trace if event.get("kind") == "coherence")
    rejected_atomic = next(event for event in rejected_trace if event.get("kind") == "atomic")
    assert data_v1._independent(rejected_coherence, rejected_atomic, rejected)
    assert data_v1._independent(rejected_atomic, rejected_coherence, rejected)


def test_permutation_and_renaming_variants_preserve_family_verdicts() -> None:
    for family, declaration, label in data_v1._base_cases():
        if declaration["kind"] != "data":
            continue
        for limited, suffix in data_v1._limit_cases(data_v1._as_object(declaration)):
            verdicts = {
                (
                    result["verdict"],
                    result.get("code"),
                )
                for variant in data_v1._data_variants(limited)
                for result in (data_v1.validate_case(data_v1._case_input(variant)),)
            }
            assert len(verdicts) == 1, f"variant mismatch for {family}:{label}{suffix}"


def test_required_data_semantic_witnesses() -> None:
    equal_text = _find("equal-text-distinct", verdict="accept")
    normalized = _accepted_result(equal_text)["normalized"]
    assert len(data_v1._as_list(normalized["targets"])) == 2
    assert _find("context", verdict="accept", predicate=_has_context_cycle)
    assert _find("dependency-cycle", verdict="reject")["expected"] == {
        "verdict": "reject",
        "code": "cycle",
    }
    assert _find("reads", verdict="accept", predicate=_has_overlapping_views)
    assert _find("outputs", verdict="reject", predicate=_has_overlapping_views)["expected"] == {
        "verdict": "reject",
        "code": "overlap",
    }
    empty = _find("empty-whole", verdict="accept")
    ownership = data_v1._as_list(_accepted_result(empty)["normalized"]["effective_ownership"])
    assert ownership == [{"target": ["local", 0], "source": ["local", 0], "start": 0, "end": 0}]
    assert _find("explicit-zero", verdict="reject")["expected"] == {
        "verdict": "reject",
        "code": "invalid_range",
    }


def test_self_source_range_error_precedes_cycle_with_exact_limits() -> None:
    case = _find("self-source-invalid-range-before-cycle", verdict="reject")
    declaration = _data_declaration(case)
    assert declaration["limits"] == data_v1._raw_counts(data_v1._as_object(declaration))
    assert case["expected"] == {"verdict": "reject", "code": "invalid_range"}


def test_effective_ownership_overlap_precedes_cycle_and_contradiction() -> None:
    declaration = data_v1._data(
        [(("local", 0), "abcd"), (("local", 1), "ab"), (("local", 2), "bc")],
        [("local", 1), ("local", 2)],
    )
    declaration["source_relations"] = [
        data_v1._source_json((("local", 1), ("local", 0), 0, 2)),
        data_v1._source_json((("local", 2), ("local", 0), 1, 3)),
    ]
    declaration["dependencies"] = [
        data_v1._dependency_json((("local", 1), ("local", 2))),
        data_v1._dependency_json((("local", 2), ("local", 1))),
    ]
    data_v1._as_object(declaration["limits"]).update(data_v1._raw_counts(declaration))
    assert data_v1.validate_case(data_v1._case_input(declaration)) == {
        "verdict": "reject",
        "code": "overlap",
    }

    contradictory = deepcopy(declaration)
    contradictory["dependencies"] = []
    first = data_v1._as_object(data_v1._as_list(contradictory["datums"])[1])
    first["text"] = "zz"
    data_v1._as_object(contradictory["limits"]).update(data_v1._raw_counts(contradictory))
    assert data_v1.validate_case(data_v1._case_input(contradictory)) == {
        "verdict": "reject",
        "code": "overlap",
    }


def test_required_record_semantic_witnesses() -> None:
    assert _find("closed-unassessed", verdict="accept")
    assert _find("closed-unmet", verdict="accept")
    assert _find("pending-unknown", verdict="accept")
    assert _find("closed-met-withheld", verdict="accept")
    assert _find("closed-met-available", verdict="accept")
    assert _rejected_result(_find("pending-protected", verdict="reject"))["code"] == "contradictory"
    assert _rejected_result(_find("unmet-protected", verdict="reject"))["code"] == "contradictory"
    assert _rejected_result(_find("duplicate_terminal", verdict="reject"))["code"] == "duplicate"
    assert _rejected_result(_find("foreign_invocation", verdict="reject"))["code"] == "foreign_owner"
    assert _rejected_result(_find("missing_evidence_artifact", verdict="reject"))["code"] == "missing"


def test_exhaustive_record_error_table_has_named_witnesses() -> None:
    expected_codes = {
        "activation-negative-occurrence": "invalid_value",
        "activation-bool-occurrence": "invalid_type",
        "activation-negative-iteration": "invalid_value",
        "activation-bool-iteration": "invalid_type",
        "activation-foreign-parent": "foreign_owner",
        "artifact-negative-key": "invalid_value",
        "artifact-bool-key": "invalid_type",
        "absence-negative-query": "invalid_value",
        "absence-bool-query": "invalid_type",
        "unknown-completion": "invalid_value",
        "unknown-qualification": "invalid_value",
        "nonbool-status": "invalid_type",
        "unknown-terminal-category": "invalid_value",
        "unknown-reason": "invalid_value",
        "terminal-vocabulary-wrong-type": "invalid_type",
        "identity-wrong-type": "invalid_type",
        "reason-element-wrong-type": "invalid_type",
        "reasons-wrong-collection": "invalid_type",
        "reference-wrong-type": "invalid_type",
        "attempt-mismatch": "foreign_owner",
        "membership-member-equals-parent": "contradictory",
        "membership-parent-mismatch": "contradictory",
        "membership-closed-wrong-type": "invalid_type",
        "evidence-foreign-dependency": "foreign_owner",
        "duplicate_terminal": "duplicate",
        "duplicate_membership_parent": "duplicate",
        "duplicate_evidence": "duplicate",
        "duplicate-status": "duplicate",
        "missing_parent": "missing",
        "undeclared_terminal": "missing",
        "missing_consumed_artifact": "missing",
        "missing_evidence_artifact": "missing",
        "missing-status": "missing",
        "foreign_invocation": "foreign_owner",
        "foreign_plan": "foreign_owner",
        "foreign_candidate_target": "foreign_owner",
        "foreign-status-target": "foreign_owner",
        "foreign-selected-target": "foreign_owner",
        "candidate-outside-targets": "invalid_value",
        "status-outside-targets": "invalid_value",
        "success-reason": "contradictory",
        "failure-no-reason": "contradictory",
        "failure-no-attempt": "contradictory",
    }
    for label, code in expected_codes.items():
        assert _find(label, verdict="reject")["expected"] == {"verdict": "reject", "code": code}

    assert _find("artifact-version-1", verdict="accept")
    assert _find("artifact-version-0", verdict="reject")["expected"] == {
        "verdict": "reject",
        "code": "invalid_value",
    }
    assert _find("artifact-version-True", verdict="reject")["expected"] == {
        "verdict": "reject",
        "code": "invalid_type",
    }
    assert _find("absence-revision-1", verdict="accept")
    assert _find("absence-revision-0", verdict="reject")["expected"] == {
        "verdict": "reject",
        "code": "invalid_value",
    }
    assert _find("absence-revision-True", verdict="reject")["expected"] == {
        "verdict": "reject",
        "code": "invalid_type",
    }
    assert _find("consume-v1", verdict="accept")
    assert _find("missing-v2", verdict="reject")["expected"] == {"verdict": "reject", "code": "missing"}
    assert _find("consume-v2", verdict="accept")

    for label in MANIFEST_NAMED_WITNESSES:
        assert any(label in case["case_id"] for case in FROZEN_CASES), label


def test_frozenset_duplicate_is_witnessed_at_manifest_boundary() -> None:
    # A repeated JSON member inside one list disappears when adapted to the
    # contracted frozenset. Repetition is observable only across manifests;
    # there it necessarily also repeats the parent and has the same code.
    case = _find("member-repeated-across-manifests-and-duplicate-parent", verdict="reject")
    assert case["expected"] == {"verdict": "reject", "code": "duplicate"}


def test_precedence_fixtures_have_frozen_codes() -> None:
    data_codes = [
        "invalid_type",
        "invalid_value",
        "limit_exceeded",
        "foreign_owner",
        "duplicate",
        "missing",
        "invalid_range",
        "overlap",
        "cycle",
        "missing",
        "invalid_range",
    ]
    record_codes = [
        "invalid_type",
        "invalid_type",
        "foreign_owner",
        "foreign_owner",
        "duplicate",
        "invalid_value",
        "missing",
    ]
    assert _family_codes("precedence") == data_codes
    assert _family_codes("record-precedence") == record_codes


def test_reference_imports_are_independent() -> None:
    source = (REFERENCE_DIR / "data_v1.py").read_text()
    tree = ast.parse(source)
    imports = {
        node.names[0].name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) and node.names
    }
    imports.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert imports <= {"__future__", "collections", "copy", "itertools", "json", "typing"}
    probe = """
import builtins
import importlib.util
from pathlib import Path
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith(('anonymizer', 'tests', 'donor')):
        raise RuntimeError('forbidden dependency')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
path = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('independent_data_v1', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
cases = module.generate_cases()
assert cases
assert module.validate_case(cases[0]) == cases[0]['expected']
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", "import sys;" + probe, str(REFERENCE_DIR / "data_v1.py")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_reference_file_hashes_match_manifest() -> None:
    assert _sha256(REFERENCE_DIR / "data_v1.py") == MANIFEST["generator_sha256"]
    assert _sha256(Path(__file__)) == MANIFEST["self_test_sha256"]
    assert MANIFEST["contract_sha256"] == "4335c135e534b3eece68021312b97f95eda63c8b60462e87da28455bb94f5341"


def test_mutant_collapse_equal_text_identities_is_killed() -> None:
    case = _find("equal-text-distinct", verdict="accept")
    _assert_named_mutant_killed("collapse_equal_text_identities", case, _collapse_equal_text)


def test_mutant_infer_atomic_groups_from_context_is_killed() -> None:
    case = _find("relation-separation", verdict="accept", predicate=_different_group_kinds)
    _assert_named_mutant_killed("infer_atomic_groups_from_context", case, _infer_atomic)


def test_mutant_admit_overlapping_output_ownership_is_killed() -> None:
    case = _find("outputs", verdict="reject", predicate=_has_overlapping_views)
    _assert_named_mutant_killed("admit_overlapping_output_ownership", case, _admit_overlap)


def test_mutant_reject_cyclic_context_is_killed() -> None:
    case = _find("context", verdict="accept", predicate=_has_context_cycle)
    _assert_named_mutant_killed("reject_cyclic_context", case, _reject_context_cycle)


def test_mutant_accept_foreign_identity_is_killed() -> None:
    case = _find("foreign-target", verdict="reject")
    _assert_named_mutant_killed("accept_foreign_identity", case, _accept_rejection)


def test_mutant_publish_successful_unassessed_target_is_killed() -> None:
    case = _find("closed-unassessed", verdict="accept")
    _assert_named_mutant_killed("publish_successful_unassessed_target", case, _publish_unassessed)


def _maximums(cases: tuple[data_v1.FixtureCase, ...]) -> dict[str, int]:
    maximums = {
        "datums": 0,
        "targets": 0,
        "declarations": 0,
        "group_members": 0,
        "text_bytes": 0,
        "trace_length": 0,
    }
    for case in cases:
        declaration = case["declaration"]
        maximums["trace_length"] = max(maximums["trace_length"], len(case["trace"]))
        if declaration["kind"] != "data":
            continue
        try:
            counts = data_v1._raw_counts(data_v1._as_object(declaration))
        except data_v1._Reject:
            continue
        maximums["datums"] = max(maximums["datums"], counts["max_datums"])
        maximums["targets"] = max(maximums["targets"], counts["max_targets"])
        maximums["declarations"] = max(maximums["declarations"], counts["max_declarations"])
        maximums["group_members"] = max(maximums["group_members"], counts["max_group_members"])
        maximums["text_bytes"] = max(maximums["text_bytes"], counts["max_text_bytes"])
    return maximums


def _find(
    label: str,
    *,
    verdict: str,
    predicate: Callable[[data_v1.FixtureCase], bool] | None = None,
) -> data_v1.FixtureCase:
    for case in FROZEN_CASES:
        if label not in case["case_id"]:
            continue
        expected = case["expected"]
        if expected["verdict"] == verdict and (predicate is None or predicate(case)):
            return case
    raise AssertionError(f"missing semantic fixture: {label}")


def _family_codes(family: str) -> list[str]:
    codes: list[str] = []
    seen: set[str] = set()
    for case in FROZEN_CASES:
        if case["family"] != family:
            continue
        case_id = case["case_id"]
        label = case_id.rsplit("-", 1)[-1]
        if label in seen:
            continue
        seen.add(label)
        codes.append(_rejected_result(case)["code"])
    return codes


def _data_declaration(case: data_v1.FixtureCase) -> data_v1.DataDeclaration:
    declaration = case["declaration"]
    if declaration["kind"] != "data":
        raise AssertionError(f"expected data fixture: {case['case_id']}")
    return declaration


def _accepted_result(case: data_v1.FixtureCase) -> data_v1.AcceptResult:
    expected = case["expected"]
    if expected["verdict"] != "accept":
        raise AssertionError(f"expected accepted fixture: {case['case_id']}")
    return expected


def _rejected_result(case: data_v1.FixtureCase) -> data_v1.RejectResult:
    expected = case["expected"]
    if expected["verdict"] != "reject":
        raise AssertionError(f"expected rejected fixture: {case['case_id']}")
    return expected


def _mutable_case_with_event(operation: str) -> data_v1.JsonObject:
    case = next(item for item in FROZEN_CASES if any(event["op"] == operation for event in item["trace"]))
    return data_v1._as_object(json.loads(json.dumps(case)))


def _mutable_trace(case: data_v1.JsonObject) -> list[data_v1.JsonObject]:
    return [data_v1._as_object(event) for event in data_v1._as_list(case["trace"])]


def _mutable_declaration(case: data_v1.JsonObject) -> data_v1.JsonObject:
    return data_v1._as_object(case["declaration"])


def _assert_loader_rejects(case: data_v1.JsonObject) -> None:
    with pytest.raises(ValueError, match="invalid JSON fixture structure"):
        data_v1._parse_cases([case])


def _has_context_cycle(case: data_v1.FixtureCase) -> bool:
    contexts = [data_v1._as_object(value) for value in _data_declaration(case)["contexts"]]
    edges = {
        (
            data_v1._ref(view["target"]),
            data_v1._ref(view["source"]),
        )
        for view in contexts
    }
    return any((right, left) in edges for left, right in edges if left != right)


def _has_overlapping_views(case: data_v1.FixtureCase) -> bool:
    declaration = _data_declaration(case)
    relations = [data_v1._as_object(value) for value in declaration["contexts"] or declaration["source_relations"]]
    if len(relations) != 2:
        return False
    return max(data_v1._offset(relations[0]["start"]), data_v1._offset(relations[1]["start"])) < min(
        data_v1._offset(relations[0]["end"]), data_v1._offset(relations[1]["end"])
    )


def _different_group_kinds(case: data_v1.FixtureCase) -> bool:
    normalized = _accepted_result(case)["normalized"]
    return bool(normalized["contexts"]) and normalized["coherence"] != normalized["atomic"]


def _assert_named_mutant_killed(name: str, case: data_v1.FixtureCase, mutant: Mutant) -> None:
    mutated = mutant(case)
    with pytest.raises(AssertionError, match=name):
        assert mutated == case["expected"], name


def _collapse_equal_text(case: data_v1.FixtureCase) -> data_v1.ValidationResult:
    result = deepcopy(data_v1.validate_case(case))
    if result["verdict"] != "accept":
        raise AssertionError("collapse_equal_text_identities requires an accepted fixture")
    normalized = result["normalized"]
    for key in ("datums", "targets", "coherence", "atomic", "effective_ownership"):
        normalized[key] = data_v1._as_list(normalized[key])[:1]
    return result


def _infer_atomic(case: data_v1.FixtureCase) -> data_v1.ValidationResult:
    result = deepcopy(data_v1.validate_case(case))
    if result["verdict"] != "accept":
        raise AssertionError("infer_atomic_groups_from_context requires an accepted fixture")
    normalized = result["normalized"]
    normalized["atomic"] = deepcopy(normalized["coherence"])
    return result


def _admit_overlap(case: data_v1.FixtureCase) -> data_v1.ValidationResult:
    del case
    return {"verdict": "accept", "normalized": {}}


def _reject_context_cycle(case: data_v1.FixtureCase) -> data_v1.ValidationResult:
    del case
    return {"verdict": "reject", "code": "cycle"}


def _accept_rejection(case: data_v1.FixtureCase) -> data_v1.ValidationResult:
    del case
    return {"verdict": "accept", "normalized": {}}


def _publish_unassessed(case: data_v1.FixtureCase) -> data_v1.ValidationResult:
    result = deepcopy(data_v1.validate_case(case))
    if result["verdict"] != "accept":
        raise AssertionError("publish_successful_unassessed_target requires an accepted fixture")
    normalized = result["normalized"]
    normalized["protection_available"] = True
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
