# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-tests for the independent effects-v1 reference corpus."""

from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import cast

from tests.graph_sdk.reference import effects_v1 as reference

HERE = Path(__file__).parent
GENERATOR = HERE / "effects_v1.py"
CORPUS = HERE / "effects_v1_cases.json"
MANIFEST = HERE / "effects_v1_manifest.json"


def test_generated_cases_cover_every_closed_family() -> None:
    cases = reference.generate_cases()
    assert Counter(cast(str, case["family"]) for case in cases) == reference.FAMILY_COUNTS


def test_every_frozen_expectation_is_reduced_from_its_inputs() -> None:
    cases = reference.load_cases(json.loads(CORPUS.read_bytes()))
    for case in cases:
        assert reference.evaluate_case(case) == case["expected"], case["case_id"]
        declaration = cast(reference.Object, case["declaration"])
        for raw_trace in cast(list[object], case["traces"]):
            trace = cast(reference.Object, raw_trace)
            events = cast(list[reference.Object], trace["events"])
            assert reference.reduce_trace(declaration, events) == trace["expected"]


def test_generation_is_deterministic_and_manifest_is_actual() -> None:
    first = reference.canonical_bytes(reference.generate_cases())
    second = reference.canonical_bytes(reference.generate_cases())
    assert first == second == CORPUS.read_bytes()
    manifest = cast(dict[str, object], json.loads(MANIFEST.read_bytes()))
    assert manifest["case_count"] == len(reference.generate_cases())
    assert manifest["trace_count"] == reference.trace_count(reference.generate_cases())
    assert manifest["event_count"] == reference.event_count(reference.generate_cases())
    assert manifest["corpus_sha256"] == hashlib.sha256(first).hexdigest()


def test_request_mutants_change_semantics_at_the_real_boundary() -> None:
    case = reference.case_by_id("keyed/valid_shared_reordered")
    events = cast(list[reference.Object], case["events"])
    assert reference.reduce_trace(cast(reference.Object, case["declaration"]), events)["status"] == "accepted"
    assert reference.reduce_trace(
        cast(reference.Object, case["declaration"]), events[:2] + events[3:4] + events[2:3] + events[4:]
    ) == {
        "code": "missing_reservation",
        "status": "rejected",
    }
    cross = reference.case_by_id("retry/cross_policy_semantic")
    assert cast(reference.Object, cross["expected"])["code"] == "cross_policy"


def test_keyed_defects_and_first_terminal_are_immutable() -> None:
    for suffix, code in (
        ("keyed/missing", "missing_keyed_result"),
        ("keyed/duplicate", "duplicate_keyed_result"),
        ("keyed/extra", "extra_keyed_result"),
        ("keyed/foreign", "foreign_keyed_result"),
    ):
        result = cast(reference.Object, reference.case_by_id(suffix)["expected"])
        state = cast(reference.Object, result["state"])
        assert code in cast(list[str], state["defects"])
        assert cast(reference.Object, state["terminals"])["R0"] == "inconsistent"
    late = cast(reference.Object, reference.case_by_id("races/lost_late_result_settlement")["expected"])
    state = cast(reference.Object, late["state"])
    assert cast(reference.Object, state["terminals"])["R0"] == "lost"
    assert "R0" not in cast(list[str], state["remote_outstanding"])


def test_budget_attempt_and_shared_accounting_are_independent() -> None:
    mixed = cast(reference.Object, reference.case_by_id("retry/mixed_exhausted_shared")["expected"])
    state = cast(reference.Object, mixed["state"])
    assert cast(reference.Object, state["denials"])["T0"] == "request_limit_stopped"
    assert state["dispatched_count"] == 2
    assert cast(reference.Object, state["attempts"])["T1"] == 1
    shared = cast(reference.Object, reference.case_by_id("keyed/valid_shared_reordered")["expected"])
    assert cast(reference.Object, shared["state"])["dispatched_count"] == 1


def test_replay_and_runtime_mapping_products_are_closed() -> None:
    denied = cast(reference.Object, reference.case_by_id("retry/replay_never_retryable")["expected"])
    assert denied == {"code": "replay_forbidden", "status": "rejected"}
    accepted = cast(
        reference.Object,
        reference.case_by_id("retry/replay_before_acceptance_rejected_before_acceptance")["expected"],
    )
    assert accepted["status"] == "accepted"
    bridge_ids = {
        cast(str, case["case_id"]).split("/", 1)[1]
        for case in reference.generate_cases()
        if case["family"] == "bridges"
    }
    assert set(reference.RUNTIME_CONDITIONS) - {"cancel_before_start", "failure"} <= bridge_ids
    assert {f"failure_{failure}" for failure in reference.FAILURE_CLASSES} <= bridge_ids


def test_retries_require_a_terminal_for_the_same_association() -> None:
    cross = cast(reference.Object, reference.case_by_id("retry/cross_association_predecessor")["expected"])
    assert cross == {"code": "missing_predecessor", "status": "rejected"}
    for suffix, code in (
        ("retry_after_permanent", "invalid_retry"),
        ("correction_after_retryable", "invalid_correction"),
        ("failover_after_malformed", "invalid_failover"),
        ("late_failure_does_not_change_authority", "invalid_retry"),
    ):
        result = cast(reference.Object, reference.case_by_id(f"retry/{suffix}")["expected"])
        assert result == {"code": code, "status": "rejected"}


def test_failover_requires_replay_authority_after_failure_class_check() -> None:
    for replay in ("never", "before_acceptance", "idempotent"):
        case = reference.case_by_id(f"retry/failover_permanent_replay_{replay}")
        result = reference.evaluate_case(case)
        if replay == "idempotent":
            state = cast(reference.Object, result["state"])
            assert state["reservations"] == {"R1": ["T0"]}
        else:
            assert result == {"code": "replay_forbidden", "status": "rejected"}


def test_followup_requests_use_the_latest_request_terminal() -> None:
    for latest, code in (
        ("pending", "missing_predecessor"),
        ("success", "invalid_retry"),
        ("permanent", "invalid_retry"),
    ):
        result = reference.evaluate_case(reference.case_by_id(f"retry/latest_request_{latest}"))
        assert result == {"status": "rejected", "code": code}
    result = reference.evaluate_case(reference.case_by_id("retry/latest_request_malformed"))
    assert result["status"] == "accepted"
    state = cast(reference.Object, result["state"])
    assert state["reservations"] == {"R2": ["T0"]}
    assert state["dispatched_count"] == 2
    assert cast(reference.Object, state["request_failures"]) == {"R0": "retryable", "R1": "malformed_response"}


def test_late_binding_responses_cannot_create_outputs_or_change_terminal() -> None:
    for terminal in ("lost", "cancelled"):
        for response in ("source_result", "source_failure"):
            case = reference.case_by_id(f"binding/{terminal}_late_{response}")
            result = reference.evaluate_case(case)
            state = cast(reference.Object, result["state"])
            assert state["terminals"] == {"R0": terminal}
            assert state["artifacts"] == []
            assert state["binding_sources"] == {}
            assert state["binding_terminal"] is None
            assert state["remote_outstanding"] == []
            if terminal == "lost":
                events = cast(list[reference.Object], case["events"])
                settled = reference.reduce_trace(
                    cast(reference.Object, case["declaration"]),
                    [
                        *events,
                        {
                            "kind": "settlement",
                            "request": "R0",
                            "disposition": "completed",
                            "usage": "unknown",
                            "remote_stopped": True,
                        },
                    ],
                )
                assert cast(reference.Object, settled["state"])["remote_outstanding"] == []


def test_late_generic_failure_preserves_lost_remote_uncertainty() -> None:
    case = reference.case_by_id("races/lost_late_failure_without_settlement")
    state = cast(reference.Object, reference.evaluate_case(case)["state"])
    assert state["terminals"] == {"R0": "lost"}
    assert state["remote_outstanding"] == ["R0"]
    assert state["request_failures"] == {}
    assert state["association_terminals"] == {}


def test_conflicting_settlement_cannot_release_remote_capacity() -> None:
    case = reference.case_by_id("races/lost_conflicting_settlement_preserves_remote")
    state = cast(reference.Object, reference.evaluate_case(case)["state"])
    assert state["remote_outstanding"] == ["R0"]
    assert state["terminals"] == {"R0": "lost"}
    assert state["defects"] == ["conflicting_settlement"]
    assert cast(reference.Object, cast(reference.Object, state["settlements"])["R0"])["disposition"] == "unknown"


def test_settlement_and_usage_grammar_rejects_ambiguous_completion() -> None:
    completed = cast(
        reference.Object,
        reference.case_by_id("races/settlement_completed_without_remote_stop")["expected"],
    )
    assert completed == {"code": "invalid_settlement", "status": "rejected"}
    missing_usage = cast(reference.Object, reference.case_by_id("races/stop_missing_usage")["expected"])
    assert missing_usage == {"code": "invalid_usage", "status": "rejected"}
    invalid_type = cast(
        reference.Object,
        reference.case_by_id("races/settlement_invalid_remote_stopped_type")["expected"],
    )
    assert invalid_type == {"code": "invalid_settlement", "status": "rejected"}
    late = cast(reference.Object, reference.case_by_id("races/lost_late_result_without_settlement")["expected"])
    late_state = cast(reference.Object, late["state"])
    assert late_state["terminals"] == {"R0": "lost"}
    assert late_state["remote_outstanding"] == ["R0"]


def test_bridges_are_admitted_and_integrated_with_shared_requests() -> None:
    fabricated = reference.reduce_trace(
        {},
        [
            {"kind": "bridge_start", "node": "N0", "task": "T0"},
            {"condition": "result", "kind": "bridge_condition", "reported_outcome": "ok", "task": "T0"},
        ],
    )
    assert fabricated == {"code": "runtime_mapping", "status": "rejected"}
    admitted = cast(reference.Object, reference.case_by_id("bridges/result")["declaration"])
    supplied_output = reference.reduce_trace(
        admitted,
        [
            {"kind": "bridge_start", "node": "N0", "task": "T0"},
            {
                "category": "failure",
                "condition": "result",
                "kind": "bridge_condition",
                "outcome": None,
                "reported_outcome": "ok",
                "task": "T0",
            },
        ],
    )
    assert supplied_output == {"code": "runtime_mapping", "status": "rejected"}
    for suffix in ("two_targets", "start_before_dispatch", "missing", "duplicate", "extra", "foreign"):
        case = reference.case_by_id(f"bridges/shared_request_{suffix}")
        kinds = [event["kind"] for event in cast(list[reference.Object], case["events"])]
        assert "dispatch" in kinds and "result" in kinds and "bridge_emit" in kinds
        assert cast(reference.Object, case["expected"])["status"] == "accepted"
    for suffix in ("inconsistent_request_cannot_emit_success", "valid_request_cannot_emit_inconsistent"):
        result = cast(reference.Object, reference.case_by_id(f"bridges/{suffix}")["expected"])
        assert result == {"code": "request_causality", "status": "rejected"}
    retry = cast(reference.Object, reference.case_by_id("bridges/retry_uses_latest_physical_request")["expected"])
    assert cast(reference.Object, retry["state"])["tasks"] == {"T0": "success"}


def test_binding_identity_resource_and_partial_receipt_witnesses() -> None:
    collision = cast(reference.Object, reference.case_by_id("binding/two_sources_same_key")["expected"])
    state = cast(reference.Object, collision["state"])
    assert len(cast(list[object], state["artifacts"])) == 2
    assert len({cast(reference.Object, item)["identity"] for item in cast(list[object], state["artifacts"])}) == 2
    repeated = cast(reference.Object, reference.case_by_id("binding/one_source_two_declarations")["expected"])
    repeated_state = cast(reference.Object, repeated["state"])
    assert repeated_state["dispatched_count"] == 2
    assert repeated_state["resource_count"] == 1
    partial = cast(reference.Object, reference.case_by_id("binding/required_failure_preserves_prior")["expected"])
    assert cast(reference.Object, partial["state"])["binding_terminal"] == "failed"
    assert len(cast(list[object], cast(reference.Object, partial["state"])["artifacts"])) == 1


def test_binding_results_require_the_dispatched_declaration_association() -> None:
    unsolicited = cast(reference.Object, reference.case_by_id("binding/unsolicited_source_result")["expected"])
    assert unsolicited == {"code": "unsolicited_source", "status": "rejected"}
    wrong = cast(reference.Object, reference.case_by_id("binding/wrong_source")["expected"])
    assert wrong == {"code": "foreign_source", "status": "rejected"}
    for suffix, defect in (
        ("missing_result", "missing_keyed_result"),
        ("duplicate_result", "duplicate_keyed_result"),
        ("foreign_result_association", "foreign_keyed_result"),
    ):
        result = cast(reference.Object, reference.case_by_id(f"binding/{suffix}")["expected"])
        assert defect in cast(list[str], cast(reference.Object, result["state"])["defects"])
    oversize = cast(reference.Object, reference.case_by_id("binding/oversize_no_truncation")["expected"])
    oversize_state = cast(reference.Object, oversize["state"])
    assert oversize_state["terminals"] == {"R0": "success"}
    assert oversize_state["local_in_flight"] == []
    assert oversize_state["remote_outstanding"] == []


def test_admission_errors_are_derived_from_malformed_declarations() -> None:
    assert reference.admit({"admission_error": "invented"}) == {
        "code": "invalid_value",
        "status": "rejected",
    }
    cases = reference.generate_cases()
    assert all("admission_error" not in cast(reference.Object, case["declaration"]) for case in cases)
    assert (
        cast(reference.Object, reference.case_by_id("admission/aggregate_limit")["expected"])["code"]
        == "limit_exceeded"
    )
    for suffix in ("valid_external_failover", "valid_local_runtime_product", "valid_decision_runtime_product"):
        assert reference.case_by_id(f"admission/{suffix}")["expected"] == {"status": "accepted"}
    for suffix, code in (
        ("duplicate_runtime_mapping", "duplicate"),
        ("duplicate_required_condition", "duplicate"),
        ("duplicate_result_outcome", "duplicate"),
        ("runtime_product_missing_result", "missing"),
        ("runtime_product_extra_local_condition", "extra"),
    ):
        result = cast(reference.Object, reference.case_by_id(f"admission/{suffix}")["expected"])
        assert result == {"code": code, "status": "rejected"}
    for suffix in ("valid_external_failover", "valid_local_runtime_product", "valid_decision_runtime_product"):
        declaration = cast(reference.Object, reference.case_by_id(f"admission/{suffix}")["declaration"])
        mappings = cast(list[reference.Json], declaration["runtime_mappings"])
        for index in range(len(mappings)):
            mutant = json.loads(json.dumps(declaration))
            cast(list[reference.Json], mutant["runtime_mappings"]).pop(index)
            assert reference.admit(mutant) == {"code": "missing", "status": "rejected"}


def test_resource_cleanup_respects_remote_uncertainty_and_ownership() -> None:
    caller = cast(reference.Object, reference.case_by_id("resources/caller_left_open")["expected"])
    assert cast(reference.Object, caller["state"])["cleanup"] == {"Q0": "left_open"}
    waiting = cast(reference.Object, reference.case_by_id("resources/sdk_remote_waits")["expected"])
    assert cast(reference.Object, waiting["state"])["cleanup"] == {}
    detached = cast(reference.Object, reference.case_by_id("resources/sdk_safe_detach")["expected"])
    assert cast(reference.Object, detached["state"])["cleanup"] == {"Q0": "closed"}


def test_bridge_and_decision_wait_identity_are_exact() -> None:
    close = cast(reference.Object, reference.case_by_id("bridges/cancel_before_start")["expected"])
    state = cast(reference.Object, close["state"])
    assert state["closed_unstarted"] == {"A0": "blocked"}
    assert state["tasks"] == {}
    decision = cast(reference.Object, reference.case_by_id("decisions/matching_resume_unrelated_advances")["expected"])
    decision_state = cast(reference.Object, decision["state"])
    assert decision_state["tasks"] == {"T0": "success", "T1": "success"}
    stale = cast(reference.Object, reference.case_by_id("decisions/stale_artifact")["expected"])
    assert cast(reference.Object, stale["state"])["decision_defects"] == ["stale_artifact"]


def test_generator_has_no_production_imports_or_expected_count_spoofing() -> None:
    tree = ast.parse(GENERATOR.read_text())
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert all("anonymizer" not in ast.unparse(node) for node in imports)
    source = GENERATOR.read_text()
    assert "EXPECTED_CASE_COUNT" not in source
    assert "EXPECTED_EVENT_COUNT" not in source


def test_materialization_records_corrected_predecessor_and_covers_both_paths() -> None:
    cases = reference.generate_cases()
    manifest = cast(dict[str, object], json.loads(MANIFEST.read_bytes()))
    prefix = reference.canonical_bytes(cases[:214])
    assert manifest["predecessor_corpus_sha256"] == "b509a2e1dc697ae5e16c8f2f9652aac74d862e4b449f64958fbba99abda87e93"
    assert manifest["corrected_predecessor_prefix_sha256"] == hashlib.sha256(prefix).hexdigest()
    for path in ("initial", "adaptive"):
        single = cast(reference.Object, reference.case_by_id(f"materialization/{path}_single_exact")["expected"])
        collection = cast(reference.Object, reference.case_by_id(f"materialization/{path}_collection_3")["expected"])
        assert single["status"] == collection["status"] == "accepted"
        assert "materialization" in cast(reference.Object, collection["state"])


def test_materialization_is_atomic_and_canonical() -> None:
    reordered = cast(reference.Object, reference.case_by_id("materialization/initial_canonical_reorder")["expected"])
    state = cast(reference.Object, reordered["state"])
    materialization = cast(reference.Object, state["materialization"])
    port = next(iter(cast(reference.Object, materialization["ports"]).values()))
    assert [item["key"] for item in cast(list[reference.Object], cast(reference.Object, port)["value"])] == [0, 1]
    for suffix, code in (
        ("initial_collection_one_over", "collection_limit_exceeded"),
        ("initial_duplicate", "duplicate_item"),
        ("initial_nested_value", "nested_collection"),
        ("initial_logical_bytes_one_over", "artifact_bytes_exceeded"),
    ):
        assert reference.case_by_id(f"materialization/{suffix}")["expected"] == {
            "code": code,
            "status": "rejected",
        }


def test_initial_and_adaptive_provenance_are_distinct() -> None:
    initial = cast(reference.Object, reference.case_by_id("materialization/initial_collection_2")["expected"])
    initial_state = cast(reference.Object, initial["state"])
    initial_materialization = cast(reference.Object, initial_state["materialization"])
    initial_key = next(iter(cast(reference.Object, initial_materialization["provenance"])))
    assert initial_key.startswith("InitialCollectionKey:")
    assert initial_materialization["artifact_count"] == 3
    assert initial_materialization["artifact_bytes"] == 8
    adaptive = cast(reference.Object, reference.case_by_id("materialization/adaptive_collection_2")["expected"])
    adaptive_state = cast(reference.Object, adaptive["state"])
    adaptive_materialization = cast(reference.Object, adaptive_state["materialization"])
    adaptive_key = next(
        key
        for key in cast(reference.Object, adaptive_materialization["provenance"])
        if key.startswith("OperationOutputKey:")
    )
    assert adaptive_key.startswith("OperationOutputKey:")
    assert "Binding" not in adaptive_key
    assert adaptive_key == "OperationOutputKey:A0:T0:context"
    assert adaptive_materialization["artifact_count"] == 2  # selector root + one output
    assert adaptive_materialization["artifact_bytes"] == 12  # selector bytes + collection bytes
    assert cast(reference.Object, adaptive_materialization["provenance"])[adaptive_key] == ["RootInputKey:T0:input"]


def test_materialization_admission_mutants_are_real_declarations() -> None:
    expected = {
        "adaptive_binding_identity": "contradictory",
        "collection_ceiling": "contradictory",
        "collection_root_input": "collection_root_input",
        "conflicting_schema": "contradictory",
        "nested_collection_schema": "nested_collection",
        "single_max_items": "contradictory",
    }
    for suffix, code in expected.items():
        case = reference.case_by_id(f"materialization/{suffix}")
        assert case["boundary"] == "admission"
        assert case["expected"] == {"code": code, "status": "rejected"}
        assert "admission_error" not in cast(reference.Object, case["declaration"])


def test_same_port_name_keeps_scoped_initial_collection_keys() -> None:
    result = cast(reference.Object, reference.case_by_id("materialization/same_port_distinct_nodes")["expected"])
    state = cast(reference.Object, result["state"])
    provenance = cast(reference.Object, cast(reference.Object, state["materialization"])["provenance"])
    assert sorted(provenance) == [
        "BoundInputKey:T0:N0:context:D0:0:1",
        "BoundInputKey:T0:N1:context:D1:0:1",
        "InitialCollectionKey:T0:N0:context:D0",
        "InitialCollectionKey:T0:N1:context:D1",
    ]


def test_declared_materialization_cannot_use_legacy_result_paths() -> None:
    for suffix, code in (
        ("initial_unmaterialized_source_result", "materialization_required"),
        ("adaptive_unmaterialized_result", "materialization_required"),
        ("binding_finish_before_materialization", "missing_materialization"),
    ):
        assert reference.case_by_id(f"materialization/{suffix}")["expected"] == {
            "code": code,
            "status": "rejected",
        }


def test_materialization_item_roots_keep_exact_scope_and_scalar_types() -> None:
    result = cast(reference.Object, reference.case_by_id("materialization/initial_collection_2")["expected"])
    state = cast(reference.Object, result["state"])
    provenance = cast(reference.Object, cast(reference.Object, state["materialization"])["provenance"])
    parents = ["BoundInputKey:T0:N0:context:D0:0:1", "BoundInputKey:T0:N0:context:D0:1:1"]
    assert provenance["InitialCollectionKey:T0:N0:context:D0"] == parents
    artifacts = cast(list[reference.Object], state["artifacts"])
    for parent in parents:
        assert provenance[parent] == []
        assert next(item for item in artifacts if item["identity"] == parent)["artifact_type"] == "text"


def test_late_materialization_cannot_replace_first_terminal_or_create_output() -> None:
    for path in ("initial", "adaptive"):
        for terminal in ("lost", "cancelled"):
            case = reference.case_by_id(f"materialization/{path}_late_{terminal}")
            declaration = cast(reference.Object, case["declaration"])
            events = cast(list[reference.Object], case["events"])
            before = cast(reference.Object, reference.reduce_trace(declaration, events[:-1])["state"])
            after = cast(reference.Object, cast(reference.Object, case["expected"])["state"])
            for field in (
                "artifacts",
                "binding_sources",
                "binding_terminal",
                "remote_outstanding",
                "request_facts",
                "terminals",
            ):
                assert after[field] == before[field], (path, terminal, field)
            assert after.get("materialization") == before.get("materialization")
            assert cast(reference.Object, after["terminals"])["R0"] == terminal


def test_scalar_does_not_consume_collection_capacity() -> None:
    for path in ("initial", "adaptive"):
        result = cast(
            reference.Object, reference.case_by_id(f"materialization/{path}_single_zero_collection_limit")["expected"]
        )
        assert result["status"] == "accepted"


def test_adaptive_provenance_requires_declared_existing_producer() -> None:
    for suffix in ("missing_parent", "foreign_parent", "invented_parent", "missing_source_fact"):
        assert reference.case_by_id(f"materialization/adaptive_{suffix}")["expected"] == {
            "status": "rejected",
            "code": "invalid_provenance",
        }


def test_adaptive_result_drives_actual_task_outcome() -> None:
    case = reference.case_by_id("materialization/adaptive_result_bridge")
    result = cast(reference.Object, case["expected"])
    assert result["status"] == "accepted"
    state = cast(reference.Object, result["state"])
    assert state["tasks"] == {"A0": "success"}
    assert state["request_facts"] == {"R0": {"condition": "result", "outcomes": {"A0": "ok"}}}
    events = cast(list[reference.Object], json.loads(json.dumps(case["events"])))
    events[-2]["reported_outcome"] = "other"
    assert reference.reduce_trace(cast(reference.Object, case["declaration"]), events) == {
        "status": "rejected",
        "code": "request_causality",
    }


def test_binding_success_records_physical_and_association_authority() -> None:
    result = cast(reference.Object, reference.case_by_id("binding/two_sources_same_key")["expected"])
    state = cast(reference.Object, result["state"])
    assert state["request_facts"] == {
        "R0": {"condition": "result", "outcomes": {"D0": "retrieved"}},
        "R1": {"condition": "result", "outcomes": {"D1": "retrieved"}},
    }
    assert state["association_terminals"] == {
        "D0": {"outcome": "retrieved", "policy": "P0", "request": "R0"},
        "D1": {"outcome": "retrieved", "policy": "P0", "request": "R1"},
    }
    assert set(cast(reference.Object, state["settlements"])) == {"R0", "R1"}


def test_binding_success_shape_and_oversize_are_closed() -> None:
    for suffix in ("empty_optional_response_malformed",):
        state = cast(
            reference.Object,
            cast(reference.Object, reference.case_by_id(f"binding/{suffix}")["expected"])["state"],
        )
        assert cast(reference.Object, state["request_facts"])["R0"] == {
            "condition": "failure",
            "failure": "malformed_response",
        }
        assert state["artifacts"] == []
    oversize = cast(reference.Object, reference.case_by_id("binding/oversize_retrieved_known_usage")["expected"])
    state = cast(reference.Object, oversize["state"])
    assert cast(reference.Object, state["request_facts"])["R0"] == {
        "condition": "result",
        "outcomes": {"D0": "retrieved"},
    }
    assert cast(reference.Object, state["binding_sources"])["D0"] == "oversize"
    assert state["artifacts"] == []


def test_optional_omission_requires_explicit_permanent_initial_disposition() -> None:
    default = cast(reference.Object, reference.case_by_id("binding/optional_failure_partial")["expected"])
    omitted = cast(reference.Object, reference.case_by_id("binding/omitted_optional")["expected"])
    assert cast(reference.Object, default["state"])["binding_terminal"] == "partial"
    assert cast(reference.Object, omitted["state"])["binding_terminal"] == "partial"
    for suffix in ("required_omission_misuse", "adaptive_omission_misuse", "omission_failure_mismatch"):
        state = cast(
            reference.Object,
            cast(reference.Object, reference.case_by_id(f"binding/{suffix}")["expected"])["state"],
        )
        assert state["binding_terminal"] == "inconsistent"
        assert state["binding_sources"] == {}
        request_facts = cast(reference.Object, state["request_facts"])
        assert cast(reference.Object, request_facts["R0"])["condition"] == "failure"


def test_binding_failure_authority_precedes_retry_or_correction() -> None:
    for suffix, purpose, failure in (
        ("source_failure_retry_authority", "retry", "retryable"),
        ("source_failure_correction_authority", "correction", "malformed_response"),
    ):
        case = reference.case_by_id(f"binding/{suffix}")
        state = cast(reference.Object, cast(reference.Object, case["expected"])["state"])
        assert cast(reference.Object, state["request_facts"])["R0"] == {
            "condition": "failure",
            "failure": failure,
        }
        assert cast(reference.Object, state["reservations"])["R1"] == ["D0"]
        events = cast(list[reference.Object], case["events"])
        assert cast(reference.Object, events[-1])["purpose"] == purpose


def test_adaptive_result_retains_its_admitted_semantic_outcome() -> None:
    state = cast(
        reference.Object,
        cast(reference.Object, reference.case_by_id("binding/adaptive_semantic_outcome_independent")["expected"])[
            "state"
        ],
    )
    assert state["request_facts"] == {"R0": {"condition": "result", "outcomes": {"A0": "adaptive_ok"}}}


def test_map_membership_materializes_exact_child_values_and_provenance() -> None:
    result = cast(reference.Object, reference.case_by_id("map/membership_2")["expected"])
    publication = cast(reference.Object, cast(reference.Object, result["state"])["publication"])
    assert cast(reference.Object, publication["membership"])["members"] == ["M0", "M1"]
    artifacts = cast(list[reference.Object], publication["artifacts"])
    assert [(item["identity"], item.get("value")) for item in artifacts[1:]] == [("I0", "a"), ("I1", "b")]
    assert [item["key"] for item in cast(list[reference.Object], publication["provenance"])] == [
        "OperationOutputKey:E0:T0:members",
        "MapItemKey:E0:M0:T0:item:0:1",
        "MapItemKey:E0:M1:T0:item:1:1",
    ]
    provenance = cast(list[reference.Object], publication["provenance"])
    assert all(item["decision"] is False for item in provenance)
    assert provenance[0]["parents"] == []
    assert all(item["parents"] == ["OperationOutputKey:E0:T0:members"] for item in provenance[1:])
    assert [item["artifact"] for item in cast(list[reference.Object], publication["inputs"])] == ["I0", "I1"]


def test_map_control_only_and_other_outputs_do_not_select_hidden_membership() -> None:
    control = cast(reference.Object, reference.case_by_id("map/control_only_members")["expected"])
    publication = cast(reference.Object, cast(reference.Object, control["state"])["publication"])
    assert len(cast(list[object], publication["artifacts"])) == 1
    assert publication["inputs"] == []
    assert len(cast(list[object], publication["provenance"])) == 1
    others = cast(reference.Object, reference.case_by_id("map/membership_with_2_other_outputs")["expected"])
    other_publication = cast(reference.Object, cast(reference.Object, others["state"])["publication"])
    assert cast(reference.Object, other_publication["membership"])["members"] == ["M0", "M1"]
    assert [item["port"] for item in cast(list[reference.Object], other_publication["ports"])[:3]] == [
        "members",
        "other0",
        "other1",
    ]


def test_map_publication_is_atomic_after_transition_and_bounds() -> None:
    for suffix in (
        "prospective_transition_rejected",
        "artifact_count_one_over",
        "artifact_bytes_one_over",
        "provenance_one_over",
    ):
        state = cast(
            reference.Object,
            cast(reference.Object, reference.case_by_id(f"map/{suffix}")["expected"])["state"],
        )
        publication = cast(reference.Object, state["publication"])
        assert publication == {
            "artifacts": [],
            "assessments": [],
            "inputs": [],
            "membership": None,
            "ports": [],
            "provenance": [],
            "request_success": False,
        }


def test_map_static_and_runtime_products_are_explicit() -> None:
    for suffix in (
        "missing_expansion",
        "duplicate_map_source",
        "map_loop_duplicate_source",
        "duplicate_loop_source",
        "context_minimum_conflict",
        "collection_item_schema_conflict",
        "item_type_mismatch",
        "context_override_conflict",
        "dependency_summary_mismatch",
        "false_identity_summary",
    ):
        assert cast(reference.Object, reference.case_by_id(f"map/{suffix}")["expected"])["status"] == "rejected"
    for maximum in (0, 1):
        for destination in ("join", "ordinary", "workflow_output"):
            expected = cast(
                reference.Object,
                reference.case_by_id(f"map/resolve_{destination}_max_{maximum}")["expected"],
            )
            assert expected["status"] == "accepted"
    assert (
        cast(reference.Object, cast(reference.Object, reference.case_by_id("map/loop_exit")["expected"])["state"])[
            "resolution"
        ]
        == "member:1:exit"
    )


def test_map_overflow_retains_success_but_storage_failure_publishes_nothing() -> None:
    for suffix in ("membership_one_over", "overflow_collection_storage_exact"):
        state = cast(
            reference.Object, cast(reference.Object, reference.case_by_id(f"map/{suffix}")["expected"])["state"]
        )
        publication = cast(reference.Object, state["publication"])
        assert state["terminal"] == "overflow"
        assert publication["request_success"] is True
        assert publication["assessments"] == ["assessment0"]
        assert len(cast(list[object], publication["artifacts"])) == 1
        assert len(cast(list[object], publication["provenance"])) == 1
        assert publication["inputs"] == []
        assert publication["membership"] is None
    for suffix in ("collection_items_one_over", "overflow_collection_storage_one_over"):
        state = cast(
            reference.Object, cast(reference.Object, reference.case_by_id(f"map/{suffix}")["expected"])["state"]
        )
        assert state["terminal"] == "artifact_limit"
        publication = cast(reference.Object, state["publication"])
        assert publication["request_success"] is False
        assert publication["artifacts"] == []
    exact = cast(reference.Object, reference.case_by_id("map/collection_items_exact")["expected"])
    assert cast(reference.Object, exact["state"])["terminal"] == "published"
    assert (
        cast(reference.Object, reference.case_by_id("map/collection_items_invalid_limit")["expected"])["status"]
        == "rejected"
    )


def test_map_transition_rejection_is_derived_from_terminal_parent() -> None:
    case = reference.case_by_id("map/prospective_transition_rejected")
    events = cast(list[reference.Object], case["events"])
    assert events[0] == {"kind": "close_parent", "parent": "E0", "category": "cancelled"}
    assert all("p3_accept" not in event for event in events)
    state = cast(reference.Object, cast(reference.Object, case["expected"])["state"])
    assert state["parent_phase"] == "cancelled"
    assert state["terminal"] == "transition_rejected"
    assert (
        cast(reference.Object, reference.case_by_id("map/caller_transition_verdict_rejected")["expected"])["status"]
        == "rejected"
    )


def test_optional_oversize_preserves_physical_success_and_partial_binding() -> None:
    case = reference.case_by_id("binding/optional_oversize_partial")
    state = cast(reference.Object, cast(reference.Object, case["expected"])["state"])
    assert state["binding_terminal"] == "partial"
    assert state["binding_sources"] == {"D0": "oversize"}
    assert state["request_facts"] == {"R0": {"condition": "result", "outcomes": {"D0": "retrieved"}}}
    assert state["artifacts"] == []


def test_binding_result_constructor_mutants_do_not_claim_provider_terminals() -> None:
    for suffix in ("source_result_wrong_outcome", "source_result_outputs_present", "source_result_consumed_present"):
        case = reference.case_by_id(f"binding/{suffix}")
        result = cast(reference.Object, case["expected"])
        assert result == {"status": "rejected", "code": "contradictory"}
        event = cast(list[reference.Object], case["events"])[-1]
        assert event["kind"] == "binding_result"
        assert "items" not in event and "source" not in event


def test_manifest_binds_request_boundary_addendum() -> None:
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["request_boundary_addendum_sha256"] == reference.REQUEST_BOUNDARY_ADDENDUM_SHA256
    assert (
        reference.REQUEST_BOUNDARY_ADDENDUM_SHA256 == "6344f7f1bdbb14a4c9e08f546928d31c89f26d3ed26e0c50362d9537d891c3a5"
    )
