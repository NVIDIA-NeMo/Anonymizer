# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-tests for the independent effects-v1 reference corpus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

from tests.graph_sdk.reference import effects_v1 as reference

HERE = Path(__file__).parent
GENERATOR = HERE / "effects_v1.py"
MANIFEST = HERE / "effects_v1_manifest.json"


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
    nested = reference.case_by_id("materialization/initial_nested_value")
    assert nested["expected"] == {"code": "invalid_type", "status": "rejected"}
    for suffix in ("initial_collection_one_over", "initial_duplicate"):
        result = cast(reference.Object, reference.case_by_id(f"materialization/{suffix}")["expected"])
        assert result["status"] == "accepted"
        assert cast(reference.Object, result["state"])["artifacts"] == []


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
        "collection_root_input": "contradictory",
        "conflicting_schema": "contradictory",
        "nested_collection_schema": "contradictory",
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


def test_declared_materialization_is_owned_by_public_controller() -> None:
    for suffix in (
        "initial_unmaterialized_source_result",
        "adaptive_unmaterialized_result",
        "binding_finish_before_materialization",
    ):
        case = reference.case_by_id(f"materialization/{suffix}")
        result = cast(reference.Object, case["expected"])
        assert result["status"] == "accepted"
        state = cast(reference.Object, result["state"])
        assert cast(reference.Object, state["materialization"])["ports"]
        if suffix != "adaptive_unmaterialized_result":
            assert state["binding_terminal"] == "success"


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
                "request_facts",
                "terminals",
            ):
                assert after[field] == before[field], (path, terminal, field)
            assert after["remote_outstanding"] == []
            assert cast(reference.Object, after["settlements"])["R0"] == events[-1]["settlement"]
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
        result = cast(reference.Object, reference.case_by_id(f"materialization/adaptive_{suffix}")["expected"])
        assert result["status"] == "accepted"
        state = cast(reference.Object, result["state"])
        provenance = cast(reference.Object, cast(reference.Object, state["materialization"])["provenance"])
        assert provenance["OperationOutputKey:A0:T0:context"] == ["RootInputKey:T0:input"]
        assert "RootInputKey:T0:input" in provenance


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
    for suffix in ("required_omission_misuse", "adaptive_omission_misuse"):
        state = cast(
            reference.Object, cast(reference.Object, reference.case_by_id(f"binding/{suffix}")["expected"])["state"]
        )
        assert state["request_failures"] == {"R0": "malformed_response"}
        assert "binding_defects" not in state
        assert state["defects"] == []
        if suffix == "required_omission_misuse":
            assert state["binding_sources"] == {"D0": "failed"}
            assert state["binding_terminal"] == "failed"
        else:
            assert state["binding_sources"] == {}
            assert state["binding_terminal"] is None
            assert state["tasks"] == {"A0": "failure"}


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
    alias = reference.case_by_id("map/caller_transition_verdict_rejected")
    assert alias["events"] == case["events"]
    assert alias["expected"] == case["expected"]


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


def test_cancel_before_start_policy_matches_unstarted_bridge() -> None:
    bridge = cast(reference.Object, reference.case_by_id("bridges/cancel_before_start")["expected"])
    state = cast(reference.Object, bridge["state"])
    category = cast(reference.Object, state["closed_unstarted"])["A0"]
    assert category == "blocked"
    assert state["tasks"] == {}
    checked = 0
    for case in reference.generate_cases():
        if case["boundary"] != "admission" or case["expected"] != {"status": "accepted"}:
            continue
        declaration = cast(reference.Object, case["declaration"])
        for index, raw in enumerate(cast(list[reference.Json], declaration.get("runtime_mappings", []))):
            row = cast(reference.Object, raw)
            if row["condition"] != "cancel_before_start":
                continue
            checked += 1
            assert (row["outcome"], row["category"]) == (None, category)
            for replacement in ({"category": "cancelled"}, {"outcome": "ok", "category": "success"}):
                mutant = json.loads(json.dumps(declaration))
                mutant["runtime_mappings"][index].update(replacement)
                if "outcome" in replacement:
                    mutant["declared_outcomes"] = ["ok"]
                    mutant["outcome_categories"] = {"ok": "success"}
                assert reference.admit(mutant) == {"status": "rejected", "code": "contradictory"}
    assert checked >= 2


def test_typed_admission_catalog_limits_precede_duplicates() -> None:
    distinct = reference.case_by_id("admission/aggregate_limit")
    duplicate = reference.case_by_id("admission/capability_one_over")
    for case in (distinct, duplicate):
        assert case["expected"] == {"status": "rejected", "code": "limit_exceeded"}
    for case, expected in (
        (distinct, {"status": "accepted"}),
        (duplicate, {"status": "rejected", "code": "duplicate"}),
    ):
        declaration = json.loads(json.dumps(case["declaration"]))
        declaration["admission_limits"]["max_capabilities"] = 2
        assert reference.admit(declaration) == expected


def test_runtime_duplicate_witnesses_preserve_key_kind_and_precedence() -> None:
    for suffix, condition in (
        ("duplicate_required_condition", "cancel_before_start"),
        ("duplicate_result_outcome", "result"),
    ):
        case = reference.case_by_id(f"admission/{suffix}")
        declaration = json.loads(json.dumps(case["declaration"]))
        assert "required_runtime_conditions" not in declaration
        rows = declaration["runtime_mappings"]
        assert rows[-1]["condition"] == condition
        assert case["expected"] == {"status": "rejected", "code": "duplicate"}
        rows.pop()
        assert reference.admit(declaration) == {"status": "accepted"}


def test_failover_drift_is_rechecked_after_valid_admission() -> None:
    case = reference.case_by_id("admission/changed_failover_policy")
    assert case["boundary"] == "pre_execution"
    declaration = json.loads(json.dumps(case["declaration"]))
    assert reference.admit(declaration["admitted"]) == {"status": "accepted"}
    assert reference.recheck_capabilities(declaration) == {"status": "rejected", "code": "changed_failover_policy"}
    declaration["capability_catalog"] = declaration["admitted"]["capability_catalog"]
    assert reference.recheck_capabilities(declaration) == {"status": "accepted"}
    declaration["capability_catalog"] = declaration["capability_catalog"][:1]
    assert reference.recheck_capabilities(declaration) == {"status": "rejected", "code": "changed_failover_policy"}


def test_exact_binding_bounds_return_terminal_public_result() -> None:
    case = reference.case_by_id("binding/exact_item_byte_bounds")
    events = cast(list[reference.Object], case["events"])
    assert events[-1] == {"kind": "binding_finish"}
    declaration = cast(reference.Object, case["declaration"])
    before = cast(reference.Object, reference.reduce_trace(declaration, events[:-1])["state"])
    after = cast(reference.Object, reference.reduce_trace(declaration, events)["state"])
    assert before["binding_terminal"] is None
    assert after["binding_terminal"] == "success"
    assert {key: value for key, value in before.items() if key != "binding_terminal"} == {
        key: value for key, value in after.items() if key != "binding_terminal"
    }


def test_source_failure_constructor_rejections_leave_state_unchanged() -> None:
    for suffix, expected in (
        ("source_failure_missing_failure", {"status": "rejected", "exception": "TypeError"}),
        ("source_failure_missing_settlement", {"status": "rejected", "exception": "TypeError"}),
        ("omission_failure_mismatch", {"status": "rejected", "code": "contradictory"}),
    ):
        case = reference.case_by_id(f"binding/{suffix}")
        declaration = cast(reference.Object, case["declaration"])
        event = cast(list[reference.Object], case["events"])[-1]
        assert event["kind"] == "source_failure_constructor"
        assert case["expected"] == expected
        state = reference._initial(declaration)
        before = json.loads(json.dumps(state))
        assert reference._advance(state, declaration, event) == expected
        assert state == before
        valid = dict(event, failure="permanent", settlement=None)
        assert reference._advance(state, declaration, valid) is None
        assert state == before


def test_wrong_source_is_the_only_response_defect() -> None:
    case = reference.case_by_id("binding/wrong_source")
    declaration = cast(reference.Object, case["declaration"])
    events = json.loads(json.dumps(case["events"]))
    bad_state = cast(reference.Object, cast(reference.Object, case["expected"])["state"])
    assert bad_state["artifacts"] == []
    assert bad_state["terminals"] == {"R0": "failure"}
    assert bad_state["request_failures"] == {"R0": "malformed_response"}
    assert bad_state["dispatched_count"] == 1
    assert bad_state["local_in_flight"] == bad_state["remote_outstanding"] == []
    assert cast(reference.Object, bad_state["settlements"])["R0"] == events[-2]["settlement"]
    events[-2]["source"] = "S0"
    valid = cast(reference.Object, reference.reduce_trace(declaration, events)["state"])
    assert valid["binding_terminal"] == "success"
    assert valid["terminals"] == {"R0": "success"}
    assert len(cast(list[reference.Json], valid["artifacts"])) == 1


def test_unsolicited_result_has_a_valid_dispatched_counterpart() -> None:
    case = reference.case_by_id("binding/unsolicited_source_result")
    declaration = cast(reference.Object, case["declaration"])
    events = cast(list[reference.Object], case["events"])
    state = reference._initial(declaration)
    before = json.loads(json.dumps(state))
    assert reference._advance(state, declaration, events[0]) == {"status": "rejected", "code": "missing"}
    assert state == before
    accepted = reference.reduce_trace(declaration, [*reference._trace(("D0",)), *events])
    assert accepted["status"] == "accepted"
    assert cast(reference.Object, accepted["state"])["terminals"] == {"R0": "success"}


def test_source_item_constructor_invalid_values_have_valid_counterparts() -> None:
    for path in ("initial", "adaptive"):
        for suffix in ("wrong_item", "nested_value"):
            case = reference.case_by_id(f"materialization/{path}_{suffix}")
            assert case["expected"] == {"status": "rejected", "code": "invalid_type"}
            declaration = cast(reference.Object, case["declaration"])
            event = json.loads(json.dumps(cast(list[reference.Object], case["events"])[-1]))
            assert event["kind"] == "source_item_constructor"
            state = reference._initial(declaration)
            before = json.loads(json.dumps(state))
            assert reference._advance(state, declaration, event) == case["expected"]
            assert state == before
            event["items"][0]["value"] = "valid text"
            assert reference._advance(state, declaration, event) is None
            assert state == before


def test_collection_constructor_negatives_have_canonical_counterparts() -> None:
    for suffix, code in (("duplicate_item", "duplicate"), ("noncanonical_items", "invalid_value")):
        case = reference.case_by_id(f"map/{suffix}")
        assert case["expected"] == {"status": "rejected", "code": code}
        event = json.loads(json.dumps(cast(list[reference.Object], case["events"])[0]))
        assert event["kind"] == "collection_constructor"
        event["outputs"][0]["items"] = [{"key": 0, "version": 1, "value": "a"}, {"key": 1, "version": 1, "value": "b"}]
        result = reference._reduce_map(cast(reference.Object, case["declaration"]), [event])
        assert result == {"status": "accepted", "state": reference._empty_map_state()}


def test_decision_submission_rejects_without_state_mutation() -> None:
    for suffix, code in (
        ("stale_artifact", "foreign_owner"),
        ("foreign_workflow", "foreign_owner"),
        ("foreign_wait", "foreign_owner"),
        ("unknown_decision", "unsupported"),
        ("duplicate_response", "duplicate"),
    ):
        case = reference.case_by_id(f"decisions/{suffix}")
        declaration = cast(reference.Object, case["declaration"])
        events = cast(list[reference.Object], case["events"])
        state = cast(reference.Object, reference.reduce_trace(declaration, events[:-1])["state"])
        before = json.loads(json.dumps(state))
        assert case["expected"] == {"status": "rejected", "code": code}
        assert reference._advance(state, declaration, events[-1]) == case["expected"]
        assert state == before
        assert "decision_defects" not in state
        if suffix == "duplicate_response":
            assert state["tasks"] == {"T0": "success"}
        else:
            valid = dict(events[-1], wait="W0", invocation="I0", workflow="F0", artifact="V0", decision="approve")
            assert reference._advance(state, declaration, valid) is None
            assert state["tasks"] == {"T0": "success"}


def test_context_schema_contradictions_have_scalar_counterparts() -> None:
    nested = json.loads(json.dumps(reference.case_by_id("materialization/nested_collection_schema")["declaration"]))
    nested["materializations"][1]["item_type"] = "text"
    nested["materializations"][1]["output_type"] = "text_collection"
    assert reference.admit(nested) == {"status": "accepted"}
    root = json.loads(json.dumps(reference.case_by_id("materialization/collection_root_input")["declaration"]))
    root["root_input_types"] = ["text"]
    assert reference.admit(root) == {"status": "accepted"}


def test_materialization_malformed_responses_retain_physical_failure_and_settlement() -> None:
    for path in ("initial", "adaptive"):
        for suffix in ("collection_0", "duplicate"):
            case = reference.case_by_id(f"materialization/{path}_{suffix}")
            result = cast(reference.Object, case["expected"])
            assert result["status"] == "accepted"
            state = cast(reference.Object, result["state"])
            association = "D0" if path == "initial" else "A0"
            assert state["dispatched_count"] == 1
            assert state["terminals"] == {"R0": "failure"}
            assert state["request_failures"] == {"R0": "malformed_response"}
            assert state["association_terminals"] == {
                association: {"failure": "malformed_response", "policy": "P0", "request": "R0"}
            }
            assert (
                cast(reference.Object, state["settlements"])["R0"]
                == cast(list[reference.Object], case["events"])[-1]["settlement"]
            )
            assert state["local_in_flight"] == state["remote_outstanding"] == []
            if path == "initial":
                assert state["binding_sources"] == {"D0": "failed"}
                assert state["binding_terminal"] == "failed"
                assert state["artifacts"] == []
            else:
                assert state["tasks"] == {"A0": "failure"}
                assert len(cast(list[reference.Json], state["artifacts"])) == 1  # retained selector only


def test_materialization_oversize_keeps_success_without_output() -> None:
    for path in ("initial", "adaptive"):
        for suffix in ("single_multiple", "collection_one_over", "outer_count_precedence"):
            case = reference.case_by_id(f"materialization/{path}_{suffix}")
            result = cast(reference.Object, case["expected"])
            assert result["status"] == "accepted"
            state = cast(reference.Object, result["state"])
            association = "D0" if path == "initial" else "A0"
            outcome = "retrieved" if path == "initial" else "ok"
            assert state["request_facts"] == {"R0": {"condition": "result", "outcomes": {association: outcome}}}
            assert state["terminals"] == {"R0": "success"}
            assert state["association_terminals"] == {
                association: {"outcome": outcome, "policy": "P0", "request": "R0"}
            }
            assert (
                cast(reference.Object, state["settlements"])["R0"]
                == cast(list[reference.Object], case["events"])[-1]["settlement"]
            )
            if path == "initial":
                assert state["binding_sources"] == {"D0": "oversize"}
                assert state["binding_terminal"] == "failed"
                assert state["artifacts"] == []
            else:
                assert state["tasks"] == {"A0": "blocked"}
                assert len(cast(list[reference.Json], state["artifacts"])) == 1


def test_execution_preflight_retains_successful_binding_and_rejects_before_execution() -> None:
    for suffix in (
        "collection_ceiling",
        "initial_artifact_count_one_over",
        "initial_logical_bytes_one_over",
        "initial_provenance_one_over",
    ):
        case = reference.case_by_id(f"materialization/{suffix}")
        assert case["boundary"] == "execution_preflight"
        result = cast(reference.Object, case["expected"])
        assert result["status"] == "rejected" and result["code"] == "limit_exceeded"
        binding = cast(reference.Object, result["binding"])
        assert binding["binding_terminal"] == "success"
        assert binding["terminals"] == {"R0": "success"}
        assert binding["settlements"]
        assert "materialization" not in binding
        declaration = json.loads(json.dumps(case["declaration"]))
        declaration["materialization_limits"].update(
            max_artifacts=100, max_artifact_bytes=1000, max_provenance_edges=100, max_collection_items=100
        )
        valid = reference._materialization_preflight(declaration, cast(list[reference.Object], case["events"]))
        assert valid == {"status": "accepted", "binding": binding}


def test_final_malformed_receipts_exhaust_both_declared_request_limits() -> None:
    checked = 0
    for case in reference.generate_cases():
        result = cast(reference.Object, case["expected"])
        if result.get("status") != "accepted" or "state" not in result:
            continue
        state = cast(reference.Object, result["state"])
        failures = cast(reference.Object, state.get("request_failures", {}))
        if "malformed_response" not in failures.values():
            continue
        final = state.get("binding_terminal") in ("failed", "partial") or state.get("tasks") == {"A0": "failure"}
        if not final:
            continue
        checked += 1
        declaration = cast(reference.Object, case["declaration"])
        policies = cast(reference.Object, declaration["policies"])
        assert cast(reference.Object, policies["P0"])["max_attempts"] == 1, case["case_id"]
        bounds = cast(reference.Object, declaration.get("retrieval_bounds", declaration["binding_limits"]))
        assert bounds["max_requests"] == 1, case["case_id"]
    assert checked == 14


def test_retry_and_correction_authority_are_intermediate_binding_states() -> None:
    for suffix in ("source_failure_retry_authority", "source_failure_correction_authority"):
        case = reference.case_by_id(f"binding/{suffix}")
        state = cast(reference.Object, cast(reference.Object, case["expected"])["state"])
        assert state["reservations"] == {"R1": ["D0"]}
        assert state["terminals"] == {"R0": "failure"}
        assert state["binding_terminal"] is None
        assert state["binding_sources"] == {}
        declaration = cast(reference.Object, case["declaration"])
        assert cast(reference.Object, cast(reference.Object, declaration["policies"])["P0"])["max_attempts"] == 2


def test_adaptive_negative_requests_have_real_selector_readiness() -> None:
    for name in ("materialization/adaptive_foreign_association", "binding/adaptive_omission_misuse"):
        case = reference.case_by_id(name)
        events = cast(list[reference.Object], case["events"])
        assert events[0]["kind"] == "root_input"
        reserve = next(event for event in events if event["kind"] == "reserve")
        assert reserve["associations"] == ["A0"]
        assert reserve["purpose"] == "adaptive_retrieval"
        state = cast(reference.Object, cast(reference.Object, case["expected"])["state"])
        materialization = cast(reference.Object, state["materialization"])
        assert cast(reference.Object, materialization["provenance"])["RootInputKey:T0:input"] == []
        assert len(cast(list[reference.Json], state["artifacts"])) == 1
        assert state["association_terminals"] == {
            "A0": {"failure": "malformed_response", "policy": "P0", "request": "R0"}
        }
        assert state["tasks"] == {"A0": "failure"}
        assert state["binding_sources"] == {}
        assert state["settlements"]


def test_adaptive_late_loss_retains_causal_cancel_and_terminal_conflict() -> None:
    cases = {case["case_id"]: case for case in reference.generate_cases()}
    case = cases["materialization/adaptive_late_lost"]
    events = cast(list[reference.Object], case["events"])
    kinds = [event["kind"] for event in events]
    assert kinds.index("cancel") < kinds.index("lost") < kinds.index("materialize_result")
    expected = cast(reference.Object, case["expected"])
    state = cast(reference.Object, expected["state"])
    assert state["cancel_requested"] == ["R0"]
    assert state["terminals"] == {"R0": "lost"}
    assert state["defects"] == ["conflicting_terminal"]
    assert cast(reference.Object, state["materialization"])["ports"] == {}


def test_initial_late_loss_records_cancel_for_both_provider_result_kinds() -> None:
    cases = {case["case_id"]: case for case in reference.generate_cases()}
    for name in (
        "materialization/initial_late_lost",
        "binding/lost_late_source_result",
        "binding/lost_late_source_failure",
    ):
        case = cases[name]
        events = cast(list[reference.Object], case["events"])
        kinds = [event["kind"] for event in events]
        assert kinds.index("cancel") < kinds.index("lost")
        expected = cast(reference.Object, case["expected"])
        state = cast(reference.Object, expected["state"])
        assert state["cancel_requested"] == ["R0"]
        assert state["terminals"] == {"R0": "lost"}
        assert "conflicting_terminal" in cast(list[str], state["defects"])
        assert state["artifacts"] == []


def test_map_provenance_capacity_rejects_before_execution() -> None:
    case = reference.case_by_id("map/provenance_one_over")
    assert case["boundary"] == "map_execution_preflight"
    assert case["events"] == []
    assert case["expected"] == {"status": "rejected", "code": "limit_exceeded"}
    exact = reference.case_by_id("map/bounds_exact")
    assert reference._map_execution_preflight(cast(reference.Object, exact["declaration"])) == {"status": "accepted"}
    assert cast(reference.Object, exact["expected"])["status"] == "accepted"


def test_wrong_parent_uses_public_callback_association_boundary() -> None:
    case = reference.case_by_id("map/wrong_parent")
    assert case["boundary"] == "local_callback"
    event = cast(list[reference.Object], case["events"])[0]
    assert event["supplied_association"] == "E0"
    assert event["returned_association"] == "E1"
    state = cast(reference.Object, cast(reference.Object, case["expected"])["state"])
    assert state["terminal"] == "malformed_response"
    assert state["parent_phase"] == "failed"
    assert state["expansion"] == {"parent": "E0", "members": [], "status": "failed"}
    assert state["publication"] == reference._empty_map_state()["publication"]
    assert state["transition"] is None
    valid = dict(event, returned_association="E0")
    accepted = reference._map_local_result(cast(reference.Object, case["declaration"]), [valid])
    assert cast(reference.Object, accepted["state"])["terminal"] == "published"
