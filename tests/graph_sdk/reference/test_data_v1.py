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
from typing import Callable, TypeAlias, cast

import pytest

from tests.graph_sdk.reference import data_v1

JsonObject: TypeAlias = dict[str, object]
Mutant: TypeAlias = Callable[[dict], dict]

REFERENCE_DIR = Path(__file__).parent
CORPUS_PATH = REFERENCE_DIR / "data_v1_cases.json"
MANIFEST_PATH = REFERENCE_DIR / "data_v1_manifest.json"
MANIFEST = json.loads(MANIFEST_PATH.read_text())
FROZEN_BYTES = CORPUS_PATH.read_bytes()
FROZEN_CASES = cast(tuple[dict, ...], tuple(json.loads(FROZEN_BYTES)))


@pytest.fixture(scope="module")
def generations() -> tuple[tuple[dict, ...], tuple[dict, ...]]:
    """Generate exactly twice and share both results across freeze checks."""
    return data_v1.generate_cases(), data_v1.generate_cases()


def test_frozen_generation_is_reproducible(
    generations: tuple[tuple[dict, ...], tuple[dict, ...]],
) -> None:
    first, second = generations
    first_bytes = data_v1.canonical_bytes(first)
    second_bytes = data_v1.canonical_bytes(second)
    assert first_bytes == second_bytes == FROZEN_BYTES
    assert hashlib.sha256(first_bytes).hexdigest() == MANIFEST["corpus_sha256"]


def test_manifest_counts_and_bounds(
    generations: tuple[tuple[dict, ...], tuple[dict, ...]],
) -> None:
    cases, duplicate_generation = generations
    del duplicate_generation
    family_counts = Counter(case["family"] for case in cases)
    event_counts = Counter(event["op"] for case in cases for event in cast(list[dict[str, object]], case["trace"]))
    assert len(cases) == MANIFEST["counts"]["deduplicated_cases"]
    assert data_v1._raw_variant_count() == MANIFEST["counts"]["raw_variants"]
    assert dict(sorted(family_counts.items())) == MANIFEST["counts"]["families"]
    assert sum(len(case["trace"]) for case in cases) == MANIFEST["counts"]["events"]
    assert len(cases) == MANIFEST["counts"]["traces"]
    assert dict(sorted(event_counts.items())) == MANIFEST["counts"]["event_kinds"]
    assert _maximums(cases) == MANIFEST["maxima"]
    base_counts = Counter(family for family, _, _ in data_v1._base_cases())
    assert base_counts["precedence"] == MANIFEST["base_case_counts"]["precedence"] == 11
    assert base_counts["record-precedence"] == MANIFEST["base_case_counts"]["record_precedence"] == 7
    assert sys.implementation.name == "cpython"
    assert MANIFEST["python"] == {
        "implementation": "CPython",
        "version": ".".join(str(part) for part in sys.version_info[:3]),
    }
    for case in cases:
        declaration = cast(JsonObject, case["declaration"])
        expected = cast(JsonObject, case["expected"])
        if declaration["kind"] == "data" and expected["verdict"] == "accept":
            counts = data_v1._raw_counts(cast(data_v1.JsonObject, declaration))
            assert declaration["limits"] == counts, case["case_id"]


def test_every_frozen_expectation_matches_the_reference() -> None:
    for case in FROZEN_CASES:
        actual = data_v1.validate_case(case)
        assert actual == case["expected"], f"reference mismatch for {case['case_id']}"


def test_alphabet_is_exact_and_traces_close_before_validation() -> None:
    static = set(data_v1.STATIC_ALPHABET)
    record = set(data_v1.RECORD_ALPHABET)
    assert list(data_v1.STATIC_ALPHABET) == MANIFEST["alphabet"]["static"]
    assert list(data_v1.RECORD_ALPHABET) == MANIFEST["alphabet"]["record"]
    for case in FROZEN_CASES:
        trace = cast(list[dict[str, object]], case["trace"])
        allowed = record if cast(JsonObject, case["declaration"])["kind"] == "record" else static
        assert all(event["op"] in allowed for event in trace), case["case_id"]
        assert [event["op"] for event in trace[-2:]] == ["close_declaration", "validate"]


def test_independent_adjacent_additions_commute() -> None:
    pairs = list(data_v1._independence_witnesses(FROZEN_CASES))
    assertions = 0
    for case, index in pairs:
        trace = deepcopy(cast(list[data_v1.JsonValue], case["trace"]))
        left = data_v1._as_object(trace[index])
        right = data_v1._as_object(trace[index + 1])
        assert data_v1._independent(left, right)
        assert data_v1._independent(right, left)
        trace[index], trace[index + 1] = trace[index + 1], trace[index]
        swapped = {"declaration": data_v1._replay_trace(case, trace)}
        assert data_v1.validate_case(swapped) == case["expected"], case["case_id"]
        assertions += 3
    assert len(pairs) == MANIFEST["independence"]["pair_count"]
    assert len(pairs) == MANIFEST["independence"]["swapped_trace_count"]
    assert assertions == MANIFEST["independence"]["assertion_count"]


def test_permutation_and_renaming_variants_preserve_family_verdicts() -> None:
    for family, declaration, label in data_v1._base_cases():
        if declaration.get("kind") != "data":
            continue
        for limited, suffix in data_v1._limit_cases(declaration):
            verdicts = {
                (
                    cast(str, result["verdict"]),
                    cast(str | None, result.get("code")),
                )
                for variant in data_v1._data_variants(limited)
                for result in (data_v1.validate_case({"declaration": variant}),)
            }
            assert len(verdicts) == 1, f"variant mismatch for {family}:{label}{suffix}"


def test_required_data_semantic_witnesses() -> None:
    equal_text = _find("equal-text-distinct", verdict="accept")
    normalized = cast(JsonObject, cast(JsonObject, equal_text["expected"])["normalized"])
    assert len(cast(list[object], normalized["targets"])) == 2
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
    ownership = cast(
        list[JsonObject],
        cast(JsonObject, cast(JsonObject, empty["expected"])["normalized"])["effective_ownership"],
    )
    assert ownership == [{"target": ["local", 0], "source": ["local", 0], "start": 0, "end": 0}]
    assert _find("explicit-zero", verdict="reject")["expected"] == {
        "verdict": "reject",
        "code": "invalid_range",
    }


def test_self_source_range_error_precedes_cycle_with_exact_limits() -> None:
    case = _find("self-source-invalid-range-before-cycle", verdict="reject")
    declaration = cast(data_v1.JsonObject, case["declaration"])
    assert declaration["limits"] == data_v1._raw_counts(declaration)
    assert case["expected"] == {"verdict": "reject", "code": "invalid_range"}


def test_required_record_semantic_witnesses() -> None:
    assert _find("closed-unassessed", verdict="accept")
    assert _find("closed-unmet", verdict="accept")
    assert _find("pending-unknown", verdict="accept")
    assert _find("closed-met-withheld", verdict="accept")
    assert _find("closed-met-available", verdict="accept")
    assert cast(JsonObject, _find("pending-protected", verdict="reject")["expected"])["code"] == "contradictory"
    assert cast(JsonObject, _find("unmet-protected", verdict="reject")["expected"])["code"] == "contradictory"
    assert cast(JsonObject, _find("duplicate_terminal", verdict="reject")["expected"])["code"] == "duplicate"
    assert cast(JsonObject, _find("foreign_invocation", verdict="reject")["expected"])["code"] == "foreign_owner"
    assert cast(JsonObject, _find("missing_evidence_artifact", verdict="reject")["expected"])["code"] == "missing"


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

    for label in MANIFEST["named_witnesses"]:
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


def _maximums(cases: tuple[dict, ...]) -> dict[str, int]:
    maximums = {
        "datums": 0,
        "targets": 0,
        "declarations": 0,
        "group_members": 0,
        "text_bytes": 0,
        "trace_length": 0,
    }
    for case in cases:
        declaration = cast(JsonObject, case["declaration"])
        maximums["trace_length"] = max(maximums["trace_length"], len(cast(list[object], case["trace"])))
        if declaration["kind"] != "data":
            continue
        try:
            counts = data_v1._raw_counts(cast(data_v1.JsonObject, declaration))
        except data_v1._Reject:
            continue
        maximums["datums"] = max(maximums["datums"], counts["max_datums"])
        maximums["targets"] = max(maximums["targets"], counts["max_targets"])
        maximums["declarations"] = max(maximums["declarations"], counts["max_declarations"])
        maximums["group_members"] = max(maximums["group_members"], counts["max_group_members"])
        maximums["text_bytes"] = max(maximums["text_bytes"], counts["max_text_bytes"])
    return maximums


def _find(label: str, *, verdict: str, predicate: Callable[[dict], bool] | None = None) -> dict:
    for case in FROZEN_CASES:
        if label not in case["case_id"]:
            continue
        expected = cast(JsonObject, case["expected"])
        if expected["verdict"] == verdict and (predicate is None or predicate(case)):
            return case
    raise AssertionError(f"missing semantic fixture: {label}")


def _family_codes(family: str) -> list[str]:
    codes: list[str] = []
    seen: set[str] = set()
    for case in FROZEN_CASES:
        if case["family"] != family:
            continue
        case_id = cast(str, case["case_id"])
        label = case_id.rsplit("-", 1)[-1]
        if label in seen:
            continue
        seen.add(label)
        codes.append(cast(str, cast(JsonObject, case["expected"])["code"]))
    return codes


def _has_context_cycle(case: dict) -> bool:
    contexts = cast(list[JsonObject], cast(JsonObject, case["declaration"])["contexts"])
    edges = {
        (
            tuple(cast(list[object], view["target"])),
            tuple(cast(list[object], view["source"])),
        )
        for view in contexts
    }
    return any((right, left) in edges for left, right in edges if left != right)


def _has_overlapping_views(case: dict) -> bool:
    declaration = cast(JsonObject, case["declaration"])
    relations = cast(list[JsonObject], declaration["contexts"] or declaration["source_relations"])
    if len(relations) != 2:
        return False
    return max(cast(int, relations[0]["start"]), cast(int, relations[1]["start"])) < min(
        cast(int, relations[0]["end"]), cast(int, relations[1]["end"])
    )


def _different_group_kinds(case: dict) -> bool:
    normalized = cast(JsonObject, cast(JsonObject, case["expected"])["normalized"])
    return bool(normalized["contexts"]) and normalized["coherence"] != normalized["atomic"]


def _assert_named_mutant_killed(name: str, case: dict, mutant: Mutant) -> None:
    mutated = mutant(case)
    with pytest.raises(AssertionError, match=name):
        assert mutated == case["expected"], name


def _collapse_equal_text(case: dict) -> dict:
    result = deepcopy(data_v1.validate_case(case))
    normalized = cast(JsonObject, result["normalized"])
    for key in ("datums", "targets", "coherence", "atomic", "effective_ownership"):
        normalized[key] = cast(list[object], normalized[key])[:1]
    return result


def _infer_atomic(case: dict) -> dict:
    result = deepcopy(data_v1.validate_case(case))
    normalized = cast(JsonObject, result["normalized"])
    normalized["atomic"] = deepcopy(normalized["coherence"])
    return result


def _admit_overlap(case: dict) -> dict:
    del case
    return {"verdict": "accept", "normalized": {}}


def _reject_context_cycle(case: dict) -> dict:
    del case
    return {"verdict": "reject", "code": "cycle"}


def _accept_rejection(case: dict) -> dict:
    del case
    return {"verdict": "accept", "normalized": {}}


def _publish_unassessed(case: dict) -> dict:
    result = deepcopy(data_v1.validate_case(case))
    normalized = cast(JsonObject, result["normalized"])
    normalized["protection_available"] = True
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
