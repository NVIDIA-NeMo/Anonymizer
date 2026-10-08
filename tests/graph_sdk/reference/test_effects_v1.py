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
            assert state["remote_outstanding"] == (["R0"] if terminal == "lost" else [])
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
