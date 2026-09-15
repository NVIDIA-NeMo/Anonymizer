# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import hashlib
import importlib
import json
from dataclasses import fields, replace
from pathlib import Path

import pytest

from tests.engine.execution.phase10_reference_model import (
    CAPTURE_BOUNDARIES,
    LIFECYCLE_STATES,
    LIMITS,
    MUTATION_CLASSES,
    OPERATIONS,
    REASONS,
    SUBJECTS,
    ReferenceCase,
    ReferencePayloadMeasurement,
    ReferenceResult,
    byte_bucket,
    canonical_corpus_bytes,
    case_by_name,
    count_bucket,
    finite_reference_cases,
    measure_reference_payload,
    reduce_reference,
    reference_manifest,
    structural_payload_witness,
)


def test_phase10_reference_model_has_no_production_or_runtime_imports() -> None:
    source = Path(__file__).with_name("phase10_reference_model.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    modules = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}

    assert not any(
        module == forbidden or module.startswith(f"{forbidden}.")
        for module in modules
        for forbidden in (
            "anonymizer",
            "pandas",
            "data_designer",
            "datadesigner",
            "tests.engine.execution.test_phase10_bounded_inspection",
        )
    )
    assert tuple(field.name for field in fields(ReferenceCase)) == (
        "name",
        "operation",
        "grant_state",
        "subject_kind",
        "capture_boundary",
        "lifecycle_state",
        "stages",
        "stage_lifecycle_states",
        "stage_task_buckets",
        "terminals",
        "diagnostics",
        "reconciliation_state",
        "cleanup_state",
        "release_state",
        "semantic_profile_version",
        "route",
        "capabilities",
        "aggregates",
        "rejection_category",
        "builder_working_bytes",
        "reasons_per_diagnostic",
        "bucket_value",
        "byte_value",
        "payload_witness",
        "canonical_limit_delta",
        "unsafe_value",
    )
    assert tuple(field.name for field in fields(ReferenceResult)) == (
        "decision",
        "rejection_code",
        "subject_accessed",
        "view_kind",
        "count_bucket",
        "byte_bucket",
        "canonical_json",
        "payload_measurement",
        "canonical_json_utf8_bytes",
        "protection_unchanged",
    )


def test_phase10_reference_manifest_freezes_counts_maxima_serialization_and_digest() -> None:
    frozen = json.loads(Path(__file__).with_name("phase10_reference_manifest.json").read_text(encoding="utf-8"))
    generated = reference_manifest()

    assert generated == frozen
    assert generated["reference_model_version"] == "phase10-bounded-inspection-reference-model/v1"
    assert generated["generator_version"] == "phase10-bounded-inspection-finite-envelope/v1"
    assert generated["case_count"] == generated["canonical_trace_count"] == len(finite_reference_cases())
    assert generated["operation_subject_case_count"] == len(OPERATIONS) * len(SUBJECTS)
    assert generated["capture_matrix_case_count"] == 8 * len(LIFECYCLE_STATES)
    assert generated["subject_boundary_case_count"] == 3 * 8
    assert generated["limit_case_count"] == 2 * (len(LIMITS) - 2) + 4
    assert generated["mutation_class_count"] == len(MUTATION_CLASSES) == 20
    assert generated["corpus_sha256"] == hashlib.sha256(canonical_corpus_bytes()).hexdigest()
    assert not canonical_corpus_bytes().endswith(b"\n")


def test_corpus_serialization_is_invariant_to_case_order(monkeypatch: pytest.MonkeyPatch) -> None:
    model = importlib.import_module("tests.engine.execution.phase10_reference_model")
    cases = finite_reference_cases()
    baseline = canonical_corpus_bytes()

    monkeypatch.setattr(model, "finite_reference_cases", lambda: tuple(reversed(cases)))

    assert canonical_corpus_bytes() == baseline


def test_operation_subject_matrix_is_exhaustive_and_closed() -> None:
    allowed = {
        "explain": {"admitted_plan", "admission_rejection_receipt"},
        "inspect": {"invocation_snapshot", "terminal_receipt", "cleanup_receipt"},
        "diagnose": {
            "admission_rejection_receipt",
            "invocation_snapshot",
            "terminal_receipt",
            "cleanup_receipt",
        },
    }
    observed = {
        (operation, subject): reduce_reference(case_by_name(f"operation-subject-{operation}-{subject}")).decision
        for operation in OPERATIONS
        for subject in SUBJECTS
    }

    assert {pair for pair, decision in observed.items() if decision == "view"} == {
        (operation, subject) for operation, subjects in allowed.items() for subject in subjects
    }


@pytest.mark.parametrize(
    ("domain", "changes"),
    (
        ("semantic-profile", {"semantic_profile_version": "unknown"}),
        ("route", {"operation": "explain", "subject_kind": "admitted_plan", "route": "unknown"}),
        (
            "capability",
            {"operation": "explain", "subject_kind": "admitted_plan", "capabilities": ("unknown",)},
        ),
        (
            "aggregate-dimension",
            {"operation": "explain", "subject_kind": "admitted_plan", "aggregates": (("unknown", "1"),)},
        ),
        (
            "aggregate-bucket",
            {"operation": "explain", "subject_kind": "admitted_plan", "aggregates": (("datums", "unknown"),)},
        ),
        ("reconciliation", {"reconciliation_state": "unknown"}),
        ("cleanup", {"cleanup_state": "unknown"}),
        ("release", {"release_state": "unknown"}),
    ),
)
def test_reference_value_domains_reject_unknown_members(domain: str, changes: dict[str, object]) -> None:
    result = reduce_reference(replace(ReferenceCase(f"unknown-{domain}"), **changes))

    assert result.decision == "rejected"
    assert result.rejection_code == "inspection_redaction_failed"
    assert result.canonical_json is None


def test_payload_measurement_failure_precedes_unsafe_schema_rejection() -> None:
    result = reduce_reference(
        ReferenceCase(
            "measurement-before-unsafe-schema",
            payload_witness="root-over",
            unsafe_value=True,
        )
    )

    assert result.rejection_code == "inspection_limit_exceeded"


def test_directed_corpus_covers_closed_domains_and_failure_precedence() -> None:
    expected = {
        "unknown-semantic-profile",
        "unknown-route",
        "unknown-capability",
        "unknown-aggregate-dimension",
        "unknown-aggregate-bucket",
        "unknown-reconciliation",
        "unknown-cleanup",
        "unknown-release",
        "measurement-before-unsafe-schema",
    }
    cases = {case.name: case for case in finite_reference_cases()}

    assert expected <= cases.keys()
    assert all(
        reduce_reference(cases[name]).rejection_code == "inspection_redaction_failed"
        for name in expected - {"measurement-before-unsafe-schema"}
    )
    assert reduce_reference(cases["measurement-before-unsafe-schema"]).rejection_code == "inspection_limit_exceeded"


def test_capture_boundary_lifecycle_matrix_is_exhaustive_and_closed() -> None:
    valid = {
        "invocation_opened": "opened",
        "pre_dispatch": "pre_dispatch",
        "post_dispatch": "post_dispatch",
        "terminal_evidence_accepted": "terminal",
        "pre_reduction_cleanup_terminal": "cleanup_terminal",
        "post_reduction_cleanup_terminal": "cleanup_terminal",
        "release_terminal": "release_terminal",
        "invocation_closed": "closed",
    }
    observed = {
        (boundary, lifecycle): reduce_reference(case_by_name(f"capture-{boundary}-{lifecycle}")).decision
        for boundary in valid
        for lifecycle in LIFECYCLE_STATES
    }

    assert {pair for pair, decision in observed.items() if decision == "view"} == set(valid.items())
    assert set(valid) == set(CAPTURE_BOUNDARIES) - {"admission_terminal"}


def test_every_subject_boundary_pair_is_accepted_only_when_owned() -> None:
    allowed = {
        "invocation_snapshot": set(CAPTURE_BOUNDARIES) - {"admission_terminal"},
        "terminal_receipt": {"terminal_evidence_accepted", "release_terminal", "invocation_closed"},
        "cleanup_receipt": {"pre_reduction_cleanup_terminal", "post_reduction_cleanup_terminal"},
    }
    for subject, boundaries in allowed.items():
        for boundary in set(CAPTURE_BOUNDARIES) - {"admission_terminal"}:
            result = reduce_reference(case_by_name(f"subject-boundary-{subject}-{boundary}"))
            assert (result.decision == "view") is (boundary in boundaries)


@pytest.mark.parametrize("grant", ["forged", "reused", "expired", "wrong_operation", "wrong_subject"])
def test_authorization_denies_before_subject_access(grant: str) -> None:
    result = reduce_reference(case_by_name(f"authorization-{grant}"))

    assert result == ReferenceResult(
        "rejected",
        "inspection_denied",
        False,
        None,
        None,
        None,
        None,
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-1, None),
        (0, "0"),
        (1, "1"),
        (2, "2-4"),
        (4, "2-4"),
        (5, "5-16"),
        (16, "5-16"),
        (17, "17-64"),
        (64, "17-64"),
        (65, "65+"),
    ],
)
def test_count_bucket_boundaries_are_exhaustive(value: int, expected: str | None) -> None:
    assert count_bucket(value) == expected
    assert reduce_reference(case_by_name(f"count-bucket-{value}")).count_bucket == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-1, None),
        (0, "0"),
        (1, "1-256"),
        (256, "1-256"),
        (257, "257-4096"),
        (4_096, "257-4096"),
        (4_097, "4097-65536"),
        (65_536, "4097-65536"),
        (65_537, "65537+"),
    ],
)
def test_byte_bucket_boundaries_are_exhaustive(value: int, expected: str | None) -> None:
    assert byte_bucket(value) == expected
    assert reduce_reference(case_by_name(f"byte-bucket-{value}")).byte_bucket == expected


@pytest.mark.parametrize("invalid", [True, 1.0, "1", None, object()])
def test_bucket_inputs_reject_non_exact_nonnegative_integers(invalid: object) -> None:
    assert count_bucket(invalid) is None
    assert byte_bucket(invalid) is None


@pytest.mark.parametrize(
    "limit_name",
    [name for name, _maximum in LIMITS if name not in {"subjects_per_call", "views_per_call"}],
)
def test_every_declared_limit_has_exact_and_one_over_witnesses(limit_name: str) -> None:
    exact = reduce_reference(case_by_name(f"limit-{limit_name}-exact"))
    over = reduce_reference(case_by_name(f"limit-{limit_name}-over"))

    if limit_name in {
        "max_provenance_fields",
        "max_top_level_fields",
        "max_json_nesting_depth",
        "max_allowlisted_string_utf8_bytes",
    }:
        assert exact.rejection_code == "inspection_redaction_failed"
        assert exact.payload_measurement is not None
    else:
        assert exact.decision == "view"
    assert over.decision == "rejected"
    assert over.rejection_code == "inspection_limit_exceeded"
    assert over.canonical_json is None
    assert over.protection_unchanged


def test_fixed_arity_limits_use_signature_and_scalar_evidence_not_request_counters() -> None:
    assert "subjects_per_call" not in {field.name for field in fields(ReferenceCase)}
    assert "views_per_call" not in {field.name for field in fields(ReferenceCase)}
    assert reference_manifest()["fixed_arity_evidence"] == [
        "private_operation_signatures_accept_owner_subject_grant",
        "private_operations_return_one_scalar_view_or_rejection",
    ]


@pytest.mark.parametrize(
    ("exact_witness", "over_witness", "expected"),
    (
        ("root-exact", "root-over", ReferencePayloadMeasurement(12, 0, 1, 3)),
        ("provenance-exact", "provenance-over", ReferencePayloadMeasurement(1, 8, 2, 10)),
        ("depth-exact", "depth-over", ReferencePayloadMeasurement(1, 0, 5, 1)),
        ("list-depth-exact", "list-depth-over", ReferencePayloadMeasurement(1, 0, 5, 1)),
        ("string-exact", "string-over", ReferencePayloadMeasurement(1, 0, 1, 96)),
        ("key-string-exact", "key-string-over", ReferencePayloadMeasurement(1, 0, 1, 96)),
    ),
)
def test_payload_measurements_are_derived_from_constructed_witnesses(
    exact_witness: str,
    over_witness: str,
    expected: ReferencePayloadMeasurement,
) -> None:
    assert measure_reference_payload(structural_payload_witness(exact_witness)) == expected
    assert measure_reference_payload(structural_payload_witness(over_witness)) == "inspection_limit_exceeded"


def test_reference_results_derive_measurements_and_canonical_size_from_payload() -> None:
    result = reduce_reference(case_by_name("synchronous-return-publication"))

    assert result.decision == "view"
    assert result.canonical_json is not None
    payload = json.loads(result.canonical_json)
    assert result.payload_measurement == measure_reference_payload(payload)
    assert result.canonical_json_utf8_bytes == len(result.canonical_json.encode("utf-8"))


def test_reason_and_terminal_domains_are_complete() -> None:
    assert {
        case.name.removeprefix("reason-") for case in finite_reference_cases() if case.name.startswith("reason-")
    } == set(REASONS)
    assert all(reduce_reference(case_by_name(f"reason-{reason}")).decision == "view" for reason in REASONS)
    assert all(
        reduce_reference(case_by_name(f"terminal-{terminal}")).decision == "view"
        for terminal in ("inconsistent", "lost", "cancelled", "failed", "blocked", "withheld", "rejected", "succeeded")
    )


def test_redaction_synchronous_publication_cleanup_and_publication_diagnostics_remain_bounded() -> None:
    assert reduce_reference(case_by_name("redaction-unknown-value")).rejection_code == "inspection_redaction_failed"
    assert reduce_reference(case_by_name("synchronous-return-publication")).decision == "view"
    assert reduce_reference(case_by_name("cleanup-failed")).decision == "view"
    assert reduce_reference(case_by_name("publication-failed")).decision == "view"
    assert reduce_reference(case_by_name("diagnose-no-reasons")).rejection_code == "inspection_state_unavailable"


def test_canonical_view_json_uses_exact_supported_scalar_and_container_types() -> None:
    for case in finite_reference_cases():
        result = reduce_reference(case)
        assert result.protection_unchanged
        if result.canonical_json is None:
            continue
        assert not result.canonical_json.endswith("\n")
        payload = json.loads(result.canonical_json)
        assert payload["view_kind"] in OPERATIONS
        _assert_json_shape(payload)


def _assert_json_shape(value: object) -> None:
    assert value is None or type(value) in {bool, int, str, list, dict}
    if type(value) is list:
        for item in value:
            _assert_json_shape(item)
    elif type(value) is dict:
        assert all(type(key) is str for key in value)
        for item in value.values():
            _assert_json_shape(item)


@pytest.mark.parametrize(
    ("operation", "field"),
    (
        ("explain", "capture_boundary"),
        ("explain", "lifecycle_state"),
        ("explain", "rejection_category"),
        ("diagnose", "rejection_category"),
    ),
)
def test_admission_observations_reject_unknown_provenance_and_rejection_domains(operation: str, field: str) -> None:
    case = case_by_name(f"operation-subject-{operation}-admission_rejection_receipt")
    result = reduce_reference(replace(case, **{field: "unknown"}))
    assert result.decision == "rejected"
    assert result.rejection_code == "inspection_redaction_failed"
    assert result.canonical_json is None
