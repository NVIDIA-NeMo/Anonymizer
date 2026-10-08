# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-tests for the independent effects-v1 reference corpus."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import cast

from tests.graph_sdk.reference import effects_v1 as reference
from tests.graph_sdk.reference.corpora import corpus_bytes
from tests.graph_sdk.reference.source_layout import assert_reference_imports, implementation_paths

HERE = Path(__file__).parent
GENERATOR = HERE / "effects_v1.py"
MANIFEST = HERE / "effects_v1_manifest.json"


def test_generated_cases_cover_every_closed_family() -> None:
    cases = reference.generate_cases()
    assert Counter(cast(str, case["family"]) for case in cases) == reference.FAMILY_COUNTS


def test_every_frozen_expectation_is_reduced_from_its_inputs() -> None:
    cases = reference.load_cases(json.loads(corpus_bytes("effects")))
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
    assert first == second == corpus_bytes("effects")
    manifest = cast(dict[str, object], json.loads(MANIFEST.read_bytes()))
    assert manifest["case_count"] == len(reference.generate_cases())
    assert manifest["trace_count"] == reference.trace_count(reference.generate_cases())
    assert manifest["event_count"] == reference.event_count(reference.generate_cases())
    assert manifest["corpus_sha256"] == hashlib.sha256(first).hexdigest()


def test_accepted_predecessor_is_an_exact_prefix() -> None:
    predecessor = reference.canonical_bytes(reference.generate_cases()[:296])
    assert hashlib.sha256(predecessor).hexdigest() == (
        "d56c9c64367ca9f4aa211aeb0e1bc8c5c213c977af07f906eced572764bc7726"
    )


def test_latest_retains_physical_history_and_selects_maximum() -> None:
    state = cast(reference.Object, reference.case_by_id("materialization/latest_version_gap")["expected"])["state"]
    state = cast(reference.Object, state)
    assert state["terminals"] == {"R0": "success"}
    assert cast(reference.Object, state["settlements"])["R0"] == {
        "disposition": "completed",
        "remote_stopped": True,
        "usage": {"input": 0, "output": 0},
    }
    port = next(iter(cast(reference.Object, cast(reference.Object, state["materialization"])["ports"]).values()))
    assert cast(reference.Object, port)["key"] == "ArtifactRef:I0:K0:3"
    assert state["binding_cleanup"] == {"Q:D0": "closed"}


def test_latest_malformed_oversize_and_late_paths_preserve_boundaries() -> None:
    duplicate = cast(reference.Object, reference.case_by_id("materialization/latest_duplicate_pair")["expected"])
    duplicate_state = cast(reference.Object, duplicate["state"])
    assert duplicate_state["request_failures"] == {"R0": "malformed_response"}
    assert duplicate_state["binding_artifacts"] == []
    for suffix in ("items_one_over", "bytes_one_over"):
        result = cast(reference.Object, reference.case_by_id(f"materialization/latest_{suffix}")["expected"])
        state = cast(reference.Object, result["state"])
        assert state["terminals"] == {"R0": "success"}
        assert state["binding_sources"] == {"D0": "oversize"}
        assert state["binding_artifacts"] == []
        assert state["artifacts"] == []
    for terminal in ("cancelled", "lost"):
        valid = cast(
            reference.Object, reference.case_by_id(f"materialization/latest_{terminal}_late_valid")["expected"]
        )
        malformed = cast(
            reference.Object,
            reference.case_by_id(f"materialization/latest_{terminal}_late_multikey")["expected"],
        )
        valid_state = cast(reference.Object, valid["state"])
        malformed_state = cast(reference.Object, malformed["state"])
        assert valid_state["terminals"] == {"R0": terminal}
        assert malformed_state["terminals"] == {"R0": terminal}
        assert valid_state["defects"] == malformed_state["defects"] == ["conflicting_terminal"]
        assert valid_state["conflicting_terminal_facts"] == [
            {
                "association": None,
                "code": "conflicting_terminal",
                "request": "R0",
                "settlement": None,
                "terminal": {
                    "category": "success",
                    "failure": None,
                    "request": "R0",
                    "results": [
                        {
                            "association": "D0",
                            "consumed_context_ports": [],
                            "outcome": "retrieved",
                            "outputs": [],
                        }
                    ],
                },
            }
        ]
        assert malformed_state["conflicting_terminal_facts"] == [
            {
                "association": None,
                "code": "conflicting_terminal",
                "request": "R0",
                "settlement": None,
                "terminal": {
                    "category": "failure",
                    "failure": "malformed_response",
                    "request": "R0",
                    "results": [],
                },
            }
        ]
        assert valid_state["binding_sources"] == {"D0": terminal}
        assert malformed_state["binding_sources"] == {"D0": terminal}
        aggregate = "lost" if terminal == "lost" else "failed"
        assert valid_state["binding_terminal"] == malformed_state["binding_terminal"] == aggregate
        assert valid_state["binding_cleanup"] == malformed_state["binding_cleanup"] == {"Q:D0": "closed"}


def test_latest_item_and_byte_limits_have_exact_and_one_over_pairs() -> None:
    for dimension in ("items", "bytes"):
        exact = cast(reference.Object, reference.case_by_id(f"materialization/latest_{dimension}_exact")["expected"])
        exact_state = cast(reference.Object, exact["state"])
        assert exact_state["binding_terminal"] == "success"
        assert exact_state["binding_cleanup"] == {"Q:D0": "closed"}
        one_over = cast(
            reference.Object, reference.case_by_id(f"materialization/latest_{dimension}_one_over")["expected"]
        )
        one_over_state = cast(reference.Object, one_over["state"])
        assert one_over_state["binding_sources"] == {"D0": "oversize"}
        assert one_over_state["binding_terminal"] == "failed"
        assert one_over_state["artifacts"] == []


def test_latest_followups_cleanup_capacity_and_transactional_rollback() -> None:
    for suffix, prior in (("retry", "retryable"), ("correction", "malformed_response")):
        state = cast(
            reference.Object,
            cast(reference.Object, reference.case_by_id(f"materialization/latest_{suffix}")["expected"])["state"],
        )
        assert cast(reference.Object, state["request_failures"])["R0"] == prior
        assert state["binding_terminal"] == "success"
        assert state["binding_cleanup"] == {"Q:D0": "closed"}
    exact = cast(reference.Object, reference.case_by_id("materialization/latest_provenance_edges_exact")["expected"])
    assert exact["status"] == "accepted"
    one_over = cast(
        reference.Object, reference.case_by_id("materialization/latest_provenance_edges_one_over")["expected"]
    )
    assert one_over["code"] == "limit_exceeded" and one_over["status"] == "rejected"
    assert "binding" in one_over
    rollback = cast(
        reference.Object,
        cast(
            reference.Object, reference.case_by_id("materialization/latest_publication_rollback_then_reuse")["expected"]
        )["state"],
    )
    assert rollback["lineage_allocations"] == {"D0:0": "K2"}
    assert rollback["allocator_next"] == 4
    assert rollback["publication_failures"] == [
        {"activation": "OP:D0", "attempt": "TASK:OP:D0", "reason": "artifact_limit_exhausted"}
    ]
    occurrences = cast(reference.Object, rollback["operation_occurrences"])
    assert cast(reference.Object, occurrences["OP:D0:TASK:OP:D0"])["terminal"] == {
        "category": "blocked",
        "reason": "artifact_limit_exhausted",
    }
    assert cast(reference.Object, occurrences["OP:N1:TASK:OP:N1"])["terminal"] == {
        "category": "success",
        "outcome": "ok",
    }
    provenance = cast(reference.Object, cast(reference.Object, rollback["materialization"])["provenance"])
    assert "OperationOutputKey:OP:D0:T0:N0:ok:result" not in provenance
    assert provenance["OperationOutputKey:OP:N1:T0:N1:ok:result"] == {
        "artifact": "ArtifactRef:I0:K3:1",
        "parents": ["RootInputKey:T0:right"],
    }
    assert rollback["final_outputs"] == ["ArtifactRef:I0:K3:1"]


def test_latest_publication_edges_come_from_exact_owned_parents() -> None:
    case = reference.case_by_id("materialization/latest_one_version")
    declaration = cast(reference.Object, case["declaration"])
    spec = cast(reference.Object, cast(list[reference.Object], declaration["materializations"])[0])
    assert reference.admit(reference._materialization_decl(spec, admission=True)) == {"status": "accepted"}
    assert "output_dependency_edges" not in spec and "output_artifacts" not in spec
    publication = cast(reference.Object, spec["publication"])
    assert publication["inputs"] == [spec["port"]]
    events = [dict(event) for event in cast(list[reference.Object], case["events"])]
    publish = next(event for event in events if event["kind"] == "operation_publish")
    finish = next(i for i, event in enumerate(events) if event["kind"] == "binding_finish")
    cleanup = next(i for i, event in enumerate(events) if event["kind"] == "binding_cleanup")
    start = next(i for i, event in enumerate(events) if event["kind"] == "operation_start")
    assert finish < cleanup < start < events.index(publish)
    publish["node"] = "OTHER"
    assert reference.reduce_trace(declaration, events) == {"code": "missing_materialization", "status": "rejected"}


def test_latest_preflight_is_structural_and_publication_is_at_most_once() -> None:
    item = reference.case_by_id("materialization/latest_provenance_edges_one_over")
    declaration = cast(reference.Object, item["declaration"])
    events = cast(list[reference.Object], item["events"])
    assert any(event["kind"] == "operation_publish" for event in events)
    assert any(event["kind"] == "binding_finish" for event in events)
    assert any(event["kind"] == "binding_cleanup" for event in events)
    assert reference._materialization_preflight(declaration, events) == item["expected"]
    binding = cast(reference.Object, item["expected"])["binding"]
    policies = cast(reference.Object, cast(reference.Object, binding)["policies"])
    assert cast(reference.Object, policies["P0"])["max_attempts"] == 3

    positive = reference.case_by_id("materialization/latest_one_version")
    positive_declaration = cast(reference.Object, positive["declaration"])
    duplicated = [dict(event) for event in cast(list[reference.Object], positive["events"])]
    no_finish = [event for event in duplicated if event["kind"] != "binding_finish"]
    assert reference.reduce_trace(positive_declaration, no_finish) == {"code": "missing", "status": "rejected"}
    early = [dict(event) for event in duplicated]
    start_index = next(i for i, event in enumerate(early) if event["kind"] == "operation_start")
    start = early.pop(start_index)
    early.insert(next(i for i, event in enumerate(early) if event["kind"] == "binding_finish"), start)
    assert reference.reduce_trace(positive_declaration, early) == {"code": "missing", "status": "rejected"}
    duplicated.append(dict(next(event for event in duplicated if event["kind"] == "operation_publish")))
    assert reference.reduce_trace(positive_declaration, duplicated) == {"code": "duplicate", "status": "rejected"}


def test_latest_sealed_context_survives_cleanup_defects_and_optional_omission() -> None:
    caller = reference.case_by_id("materialization/latest_caller_cleanup")
    caller_events = cast(list[reference.Object], caller["events"])
    caller_state = cast(reference.Object, cast(reference.Object, caller["expected"])["state"])
    assert cast(reference.Object, caller["expected"])["status"] == "accepted"
    assert caller_state["binding_cleanup_associations"] == {
        "Q:D0": {"association": "D0", "owner": "caller", "target": "T0"}
    }
    assert caller_state["binding_cleanup"] == {"Q:D0": "left_open"}
    assert caller_state["binding_finished"] is True
    assert caller_state["binding_terminal"] == "success"
    assert cast(reference.Object, caller_state["operation_occurrences"])
    assert any(event["kind"] == "operation_start" for event in caller_events)
    assert any(event["kind"] == "operation_publish" for event in caller_events)

    for disposition in ("close_failed", "close_unknown"):
        item = reference.case_by_id(f"materialization/latest_sdk_cleanup_{disposition}")
        state = cast(reference.Object, cast(reference.Object, item["expected"])["state"])
        assert state["binding_finished"] is True
        assert state["binding_terminal"] == "success"
        assert state["binding_cleanup"] == {"Q:D0": disposition}
        assert cast(reference.Object, state["operation_occurrences"])

    mixed = reference.case_by_id("materialization/latest_bound_with_optional_omission")
    state = cast(reference.Object, cast(reference.Object, mixed["expected"])["state"])
    assert state["binding_finished"] is True
    assert state["binding_terminal"] == "partial"
    assert state["binding_sources"] == {"D0": "bound", "D1": "omitted_optional"}
    assert state["binding_cleanup"] == {"Q:S0": "closed"}
    assert state["binding_cleanup_associations"] == {
        "Q:S0": {
            "associations": ["D0", "D1"],
            "owner": "sdk",
            "purpose": "binding",
            "targets": ["T0"],
        }
    }
    assert cast(reference.Object, cast(reference.Object, state["materialization"])["ports"])["initial:T0:N0:context:D0"]
    occurrences = cast(reference.Object, state["operation_occurrences"])
    assert cast(reference.Object, occurrences["OP:D1:TASK:OP:D1"])["terminal"] == {
        "category": "blocked",
        "reason": "omitted_optional",
    }

    missing_cleanup = [
        event for event in cast(list[reference.Object], mixed["events"]) if event["kind"] != "binding_cleanup"
    ]
    assert reference.reduce_trace(cast(reference.Object, mixed["declaration"]), missing_cleanup) == {
        "code": "missing",
        "status": "rejected",
    }
    foreign_cleanup = [dict(event) for event in cast(list[reference.Object], mixed["events"])]
    association = next(event for event in foreign_cleanup if event["kind"] == "binding_cleanup_association")
    association["targets"] = ["OTHER"]
    assert reference.reduce_trace(cast(reference.Object, mixed["declaration"]), foreign_cleanup) == {
        "code": "foreign_owner",
        "status": "rejected",
    }


def test_binding_finish_requires_every_terminal_and_freezes_binding_state() -> None:
    expected = {
        "latest_unresolved_optional_at_finish": "missing_materialization",
        "latest_post_finish_source_failure": "contradictory",
        "latest_post_finish_materialization": "contradictory",
    }
    for suffix, code in expected.items():
        item = reference.case_by_id(f"materialization/{suffix}")
        assert item["expected"] == {"code": code, "status": "rejected"}
        assert item["comparison_scope"] == "neutral_only"
        assert item["witness_obligation"]

    optional = reference.case_by_id("materialization/latest_optional_omission")
    optional_state = cast(reference.Object, cast(reference.Object, optional["expected"])["state"])
    assert optional_state["binding_finished"] is True
    assert optional_state["binding_terminal"] == "partial"
    assert optional_state["artifacts"] == []
    assert optional_state["materialization"] == {
        "artifact_bytes": 0,
        "artifact_count": 0,
        "ports": {},
        "provenance": {},
        "provenance_edges": 0,
    }

    mixed = reference.case_by_id("materialization/latest_bound_with_optional_omission")
    events = [dict(event) for event in cast(list[reference.Object], mixed["events"])]
    finish = next(i for i, event in enumerate(events) if event["kind"] == "binding_finish")
    reserve = reference._reserve("R2", ["D1"], purpose="initial_binding")
    events.insert(finish + 1, reserve)
    assert reference.reduce_trace(cast(reference.Object, mixed["declaration"]), events) == {
        "code": "contradictory",
        "status": "rejected",
    }

    positive = reference.case_by_id("materialization/latest_one_version")
    declaration = cast(reference.Object, positive["declaration"])
    baseline = [dict(event) for event in cast(list[reference.Object], positive["events"])]
    finish = next(i for i, event in enumerate(baseline) if event["kind"] == "binding_finish")
    original = {event["kind"]: dict(event) for event in baseline[:finish]}
    post_finish_mutations: list[reference.Object] = [
        original["bind_policy"],
        reference._reserve("R1", ["D0"], purpose="initial_binding"),
        original["dispatch"],
        {"kind": "result", "outcomes": {"D0": "retrieved"}, "request": "R0", "returned": ["D0"]},
        {"failure": "permanent", "kind": "failure", "request": "R0"},
        {"kind": "cancel", "request": "R0"},
        {"kind": "stop", "request": "R0", "usage": {"input": 0, "output": 0}},
        {"kind": "lost", "request": "R0"},
        {
            "disposition": "completed",
            "kind": "settlement",
            "remote_stopped": True,
            "request": "R0",
            "usage": {"input": 0, "output": 0},
        },
        {"association": "D0", "kind": "source_result", "request": "R0"},
        {"association": "D0", "kind": "source_failure", "request": "R0"},
        original["materialize_result"],
    ]
    for mutation in post_finish_mutations:
        events = [dict(event) for event in baseline]
        events.insert(finish + 1, mutation)
        assert reference.reduce_trace(declaration, events) == {
            "code": "contradictory",
            "status": "rejected",
        }, mutation["kind"]


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
    assert unsolicited == {"code": "missing", "status": "rejected"}
    wrong = cast(reference.Object, reference.case_by_id("binding/wrong_source")["expected"])
    wrong_state = cast(reference.Object, wrong["state"])
    assert wrong_state["binding_terminal"] == "failed"
    assert wrong_state["request_failures"] == {"R0": "malformed_response"}
    for suffix in ("missing_result", "duplicate_result", "foreign_result_association"):
        result = cast(reference.Object, reference.case_by_id(f"binding/{suffix}")["expected"])
        state = cast(reference.Object, result["state"])
        assert state["request_failures"] == {"R0": "malformed_response"}
        assert state["defects"] == []
        assert state["binding_sources"] == {"D0": "failed"}
        assert state["binding_terminal"] == "failed"
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
    assert stale == {"status": "rejected", "code": "foreign_owner"}


def test_generator_has_no_production_imports_or_expected_count_spoofing() -> None:
    assert_reference_imports("effects_v1")
    source = "\n".join(path.read_text() for path in implementation_paths("effects_v1"))
    assert "EXPECTED_CASE_COUNT" not in source
    assert "EXPECTED_EVENT_COUNT" not in source
