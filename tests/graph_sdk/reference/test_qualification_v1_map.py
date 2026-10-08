# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-tests for the independent fact-driven qualification reference."""

from __future__ import annotations

from pathlib import Path
from typing import cast

HERE = Path(__file__).parent
MANIFEST = HERE / "qualification_v1_manifest.json"
GENERATOR = HERE / "qualification_v1.py"
from tests.graph_sdk.reference.qualification_v1_selftest_helpers import case, reference, result, row


def test_dynamic_occurrence_submissions_are_exact_and_ordered() -> None:
    for count in (1, 2):
        case_id = f"assessment/dynamic_submissions_{count}"
        events = cast(list[reference.Obj], case(case_id)["events"])
        assert [event["fact"] for event in events if event.get("kind") == "assessment_submission"] == [
            "F:A:P",
            *(f"F:M{index}:P_MEMBER" for index in range(count)),
        ]
        verified = cast(list[reference.Obj], result(case_id)["verified"])
        assert [fact["activation"] for fact in verified] == [
            *(f"M{index}" for index in range(count)),
            "ROOT:A",
        ]
        assert row(case_id)["qualification"] == "met"
    assert result("assessment/dynamic_submission_repeated") == {"code": "duplicate", "status": "rejected"}
    assert result("assessment/dynamic_submission_foreign") == {"code": "foreign_owner", "status": "rejected"}


def test_dynamic_declaration_and_topology_are_count_independent() -> None:
    declarations = [case(f"assessment/dynamic_occurrences_{count}")["declaration"] for count in range(3)]
    assert declarations[0] == declarations[1] == declarations[2]
    for count in range(3):
        events = cast(list[reference.Obj], case(f"assessment/dynamic_occurrences_{count}")["events"])
        entries = {cast(str, event["activation"]): event for event in events if event.get("kind") == "entry"}
        assert entries["MAP"]["node"] == "EXP" and entries["MAP"]["node_kind"] == "operation"
        assert entries["JOIN"]["node"] == "J"
        assert all(
            event["attempt"] is not None
            for event in events
            if event.get("kind") == "terminal" and event.get("category") == "success"
        )
        root_ports = {
            cast(str, event["port"])
            for event in events
            if event.get("kind") == "port" and event.get("activation") == "ROOT:A"
        }
        assert "membership" in root_ports
        assert not any(port.startswith("mapped_") for port in root_ports)
        item_inputs = [
            event
            for event in events
            if event.get("kind") == "input_producer" and str(event.get("producer", "")).startswith("MAPITEM:")
        ]
        assert {(event["activation"], event["port"]) for event in item_inputs} == {
            (f"M{index}", "item") for index in range(count)
        }
        final_provenance = next(
            event for event in events if event.get("kind") == "provenance" and event.get("key") == "OUT:A"
        )
        assert final_provenance["parents"] == ["OP:MAP:A:members", "ROOT:A:subject"]
        assert not any(str(parent).startswith("MAPITEM:") for parent in cast(list[str], final_provenance["parents"]))


def test_verified_tuple_uses_declared_finite_ordinals_not_submission_order() -> None:
    for item in reference.CASES:
        expected = item["expected"]
        verified_raw = expected.get("verified")
        if not isinstance(verified_raw, list) or len(verified_raw) < 2:
            continue
        verified = cast(list[reference.Obj], verified_raw)
        state = reference.initial(item["declaration"])
        for event in cast(list[reference.Obj], item["events"]):
            assert reference.advance(state, event) is None
        declaration_value = item["declaration"]
        targets = {target: index for index, target in enumerate(cast(list[str], declaration_value["targets"]))}
        node_kinds = cast(dict[str, reference.Json], declaration_value["node_kinds"])
        nodes = {node: index for index, node in enumerate(sorted(node_kinds))}
        ports = {
            (dependency["node"], dependency["port"]): index
            for index, dependency in enumerate(cast(list[reference.Obj], declaration_value["output_dependencies"]))
        }
        artifact_values = cast(dict[str, reference.Json], state["artifacts"])
        artifacts = {artifact: index for index, artifact in enumerate(sorted(artifact_values))}

        def key(fact: reference.Obj) -> tuple[int, int, int, int, int]:
            entry_values = cast(dict[str, reference.Json], state["entries"])
            entry_value = entry_values[cast(str, fact["activation"])]
            return (
                targets[cast(str, fact["target"])],
                cast(int, entry_value["occurrence"]),
                nodes[cast(str, fact["node"])],
                ports[(cast(str, fact["node"]), cast(str, fact["evidence_port"]))],
                artifacts[cast(str, fact["evidence_artifact"])],
            )

        assert verified == sorted(verified, key=key), item["case_id"]

    two_map_verified = cast(list[reference.Obj], result("map_item_evidence/two_independent_maps")["verified"])
    assert [fact["activation"] for fact in two_map_verified] == ["M0", "ROOT:A", "Z0"]


def test_unreached_and_unsuccessful_assessed_members_require_no_fact() -> None:
    suppressed = case("assessment/dynamic_unreached_failed_expansion")
    suppressed_events = cast(list[reference.Obj], suppressed["events"])
    reservation = next(event for event in suppressed_events if event.get("kind") == "reservation")
    assert reservation == {
        "activation": "M0",
        "kind": "reservation",
        "parent": "MAP",
        "selected": True,
        "target": "A",
    }
    assert not any(event.get("kind") == "entry" and event.get("activation") == "M0" for event in suppressed_events)

    for name in ("unreached_failed_expansion", "blocked_unreached", "started_failure"):
        case_id = f"assessment/dynamic_{name}"
        events = cast(list[reference.Obj], case(case_id)["events"])
        assert not any(event.get("kind") == "assessment" and event.get("activation") == "M0" for event in events)
        assert result(case_id)["status"] == "accepted"
        assert "terminal_failure" in cast(list[str], row(case_id)["withholding"])
        injected_code = "missing" if name == "unreached_failed_expansion" else "unsupported"
        assert result(f"{case_id}_injected") == {"code": injected_code, "status": "rejected"}
        injected = next(
            event
            for event in cast(list[reference.Obj], case(f"{case_id}_injected")["events"])
            if event.get("kind") == "assessment" and event.get("activation") == "M0"
        )
        assert injected["subject_port"] == "item"
        assert injected["subject_artifact"] == "MI0v1"
        assert injected["consumed"] == {"item": "MI0v1"}

    blocked_events = cast(list[reference.Obj], case("assessment/dynamic_blocked_unreached")["events"])
    assert not any(
        event.get("activation") == "M0"
        and event.get("port") == "item"
        and event.get("kind") in {"port", "input_producer"}
        for event in blocked_events
    )
    assert any(event.get("kind") == "artifact" and event.get("ref") == "MI0v1" for event in blocked_events)
    assert any(event.get("kind") == "provenance" and event.get("key") == "MAPITEM:0" for event in blocked_events)

    assert result("assessment/dynamic_unreached_failed_expansion")["verified"] == []
    assert row("assessment/dynamic_unreached_failed_expansion")["withholding"] == [
        "missing_candidate",
        "terminal_failure",
    ]
    assert {event["ref"]: event["role"] for event in suppressed_events if event.get("kind") == "artifact"} == {
        "Av0": "artifact",
        "XAv0": "artifact",
    }
    assert {
        (event["activation"], event["port"], event["role"])
        for event in suppressed_events
        if event.get("kind") == "port"
    } == {("MAP", "context", "artifact")}
    assert {
        (event["activation"], event["port"], event["producer"])
        for event in suppressed_events
        if event.get("kind") == "input_producer"
    } == {("MAP", "context", "ROOT:A:context")}
    assert {event["key"] for event in suppressed_events if event.get("kind") == "provenance"} == {
        "ROOT:A:subject",
        "ROOT:A:context",
    }

    blocked_entry = next(
        event
        for event in cast(list[reference.Obj], case("assessment/dynamic_blocked_unreached")["events"])
        if event.get("kind") == "entry" and event.get("activation") == "M0"
    )
    assert blocked_entry["closed_unstarted"] is True
    blocked_events = cast(list[reference.Obj], case("assessment/dynamic_blocked_unreached")["events"])
    failed_source = next(
        event for event in blocked_events if event.get("kind") == "entry" and event.get("activation") == "FAILED"
    )
    assert failed_source["state_category"] == "failure"
    assert (
        next(event for event in blocked_events if event.get("kind") == "terminal" and event.get("activation") == "M0")[
            "attempt"
        ]
        is None
    )
    failure_entry = next(
        event
        for event in cast(list[reference.Obj], case("assessment/dynamic_started_failure")["events"])
        if event.get("kind") == "entry" and event.get("activation") == "M0"
    )
    assert failure_entry["closed_unstarted"] is False


def test_missing_terminal_retains_unsubmitted_possible_owner_history() -> None:
    expected_fact = {
        "assessment/root_missing_terminal_unsubmitted": "F:A:P",
        "assessment/dynamic_missing_terminal_unsubmitted": "F:M0:P_MEMBER",
    }
    for case_id, fact in expected_fact.items():
        item = case(case_id)
        state = reference.initial(item["declaration"])
        for event in cast(list[reference.Obj], item["events"]):
            assert reference.advance(state, event) is None
        assert fact in state["assessment_facts"]
        assert fact not in cast(list[str], state["assessment_submissions"])
        assert result(case_id)["status"] == "accepted"
        assert row(case_id)["completion"] == "pending"
        expected = (
            ["incomplete_membership"]
            if case_id == "assessment/dynamic_missing_terminal_unsubmitted"
            else ["incomplete_membership", "missing_assessment"]
        )
        assert row(case_id)["withholding"] == expected

    assert result("assessment/root_missing_terminal_submitted") == {"code": "missing", "status": "rejected"}
    assert result("assessment/dynamic_missing_terminal_submitted") == {"code": "missing", "status": "rejected"}


def test_direct_map_item_subjects_are_occurrence_scoped_artifacts() -> None:
    for count in range(3):
        case_id = f"map_item_evidence/direct_{count}"
        assert row(case_id)["qualification"] == "met"
        verified = cast(list[reference.Obj], result(case_id)["verified"])
        item_facts = [fact for fact in verified if fact["promise"] == "P_ITEM"]
        assert [fact["activation"] for fact in item_facts] == [f"M{index}" for index in range(count)]
        for index, fact in enumerate(item_facts):
            assert fact["subject"] == {"artifact": f"MI{index}v1", "producer": f"MAPITEM:{index}"}
            assert fact["subject_port"] == "item"
            assert fact["subject_artifact"] == f"MI{index}v1"
            assert fact["consumed_roles"] == {"item": "artifact"}
            assert fact["validity"] == "current"

    consumed = next(
        fact
        for fact in cast(list[reference.Obj], result("map_item_evidence/typed_consumed_endpoint")["verified"])
        if fact["promise"] == "P_ITEM"
    )
    assert consumed["subject_artifact"] == "Av0"
    assert "subject" not in consumed
    assert consumed["consumed"] == {"item": "MI0v1"}
    assert consumed["consumed_roles"] == {"item": "artifact"}


def test_map_item_endpoint_admission_and_exact_runtime_owner() -> None:
    for name in (
        "invalid_expander",
        "invalid_member",
        "invalid_item_input",
        "invalid_membership_port",
        "invalid_expansion_outcome",
        "invalid_candidate_port",
        "invalid_path_owner",
        "unresolved_projection",
        "valid_container_wrong_route",
        "duplicate_typed_requirement",
    ):
        assert result(f"map_item_admission/{name}")["status"] == "rejected"
    for name in (
        "wrong_subject_artifact",
        "wrong_item_version",
        "wrong_item_key",
        "wrong_item_owner",
        "wrong_expander",
        "wrong_member",
        "wrong_target",
        "wrong_invocation",
        "typed_consumed_wrong_owner",
        "two_maps_cross_owner",
    ):
        assert result(f"map_item_evidence/{name}")["status"] == "rejected"
    assert case("map_item_evidence/wrong_subject_artifact")["comparison_scope"] == "neutral_only"
    assert case("map_item_evidence/two_maps_cross_owner")["comparison_scope"] == "neutral_only"
    assert case("map_item_evidence/wrong_item_owner")["comparison_scope"] == "production_boundary"
    assert (
        result("map_item_evidence/wrong_subject_artifact")
        == result("map_item_evidence/wrong_item_owner")
        == {
            "code": "contradictory",
            "status": "rejected",
        }
    )


def test_map_item_routes_retain_ordered_containment_and_map_owner_transition() -> None:
    nested = case("map_item_evidence/nested_path")
    assert nested["declaration"]["map_routes"] == [
        {
            "expander": "EXP",
            "item_input": "item",
            "member": "MN",
            "membership_port": "members",
            "outcome": "ok",
            "path": ["SG"],
        }
    ]
    assert result("map_item_admission/valid_container_wrong_route") == {
        "code": "foreign_owner",
        "status": "rejected",
    }
    assert result("map_item_admission/invalid_path_owner") == {
        "code": "contradictory",
        "status": "rejected",
    }
    wrong_route = case("map_item_admission/valid_container_wrong_route")["declaration"]
    assert wrong_route["node_kinds"]["UNRELATED"] == "container"
    assert wrong_route["map_routes"][0]["path"] == []
    assert wrong_route["map_item_requirements"][0]["subject_endpoint"]["path"] == ["UNRELATED"]
    assert result("map_item_admission/duplicate_typed_requirement") == {
        "code": "duplicate",
        "status": "rejected",
    }
    assert result("map_item_evidence/typed_consumed_wrong_owner") == {
        "code": "missing",
        "status": "rejected",
    }


def test_map_item_corruption_uses_retained_owner_precedence() -> None:
    expected = {
        "wrong_item_version": "missing",
        "wrong_item_key": "missing",
        "wrong_item_owner": "contradictory",
        "wrong_expander": "missing",
        "wrong_member": "missing",
        "wrong_target": "foreign_owner",
        "wrong_invocation": "foreign_owner",
        "typed_consumed_wrong_owner": "missing",
    }
    for name, code in expected.items():
        assert result(f"map_item_evidence/{name}") == {"code": code, "status": "rejected"}


def test_map_item_records_retain_real_expanders_joins_and_structural_projection() -> None:
    direct = case("map_item_evidence/direct_1")
    assert direct["declaration"]["node_kinds"]["EXP"] == "operation"
    assert direct["declaration"]["keyed_joins"] == [
        {
            "accepted_categories": ["success"],
            "join": "J",
            "reduction": "all_by_key",
            "source": "EXP",
        }
    ]
    direct_events = cast(list[reference.Obj], direct["events"])
    assert {
        cast(str, event["activation"]): event["occurrence"] for event in direct_events if event.get("kind") == "entry"
    } == {"MAP": 0, "M0": 1, "ROOT:A": 3, "JOIN": 4}
    map_terminal = next(
        event for event in direct_events if event.get("kind") == "terminal" and event.get("activation") == "MAP"
    )
    join_terminal = next(
        event for event in direct_events if event.get("kind") == "terminal" and event.get("activation") == "JOIN"
    )
    assert map_terminal["structural"] is False and map_terminal["attempt"] == "TASK:MAP"
    assert join_terminal["structural"] is False and join_terminal["attempt"] == "TASK:JOIN"

    nested = case("map_item_evidence/nested_path")
    nested_events = cast(list[reference.Obj], nested["events"])
    wrapper_membership = next(
        event for event in nested_events if event.get("kind") == "membership" and event.get("parent") == "WRAP"
    )
    assert wrapper_membership["members"] == ["MAP", "JOIN"]
    assert next(
        event
        for event in nested_events
        if event.get("kind") == "provenance" and event.get("key") == "SGOUT:WRAP:A:nested_members"
    )["parents"] == ["OP:MAP:A:members"]
    assert (
        next(
            event
            for event in nested_events
            if event.get("kind") == "input_producer"
            and event.get("activation") == "ROOT:A"
            and event.get("port") == "membership"
        )["producer"]
        == "SGOUT:WRAP:A:nested_members"
    )

    two_maps = case("map_item_evidence/two_independent_maps")
    assert two_maps["declaration"]["keyed_joins"] == [
        {"accepted_categories": ["success"], "join": "J", "reduction": "all_by_key", "source": "EXP"},
        {"accepted_categories": ["success"], "join": "J2", "reduction": "all_by_key", "source": "EXP2"},
    ]
    root_members = next(
        event
        for event in cast(list[reference.Obj], two_maps["events"])
        if event.get("kind") == "membership" and event.get("parent") is None
    )["members"]
    assert root_members == ["ROOT:A", "MAP", "JOIN", "MAP2", "JOIN2"]
    two_map_occurrences = {
        cast(str, event["activation"]): event["occurrence"]
        for event in cast(list[reference.Obj], two_maps["events"])
        if event.get("kind") == "entry"
    }
    assert two_map_occurrences == {
        "MAP": 0,
        "M0": 1,
        "ROOT:A": 3,
        "JOIN": 4,
        "MAP2": 5,
        "Z0": 6,
        "JOIN2": 8,
    }
    assert len(set(two_map_occurrences.values())) == len(two_map_occurrences)

    for case_id in (
        "map_item_evidence/member_non_success",
        "map_item_evidence/member_blocked_unreached",
        "map_item_evidence/expansion_failed",
        "map_item_evidence/expansion_overflow",
    ):
        events = cast(list[reference.Obj], case(case_id)["events"])
        blocked_entry = next(
            event for event in events if event.get("kind") == "entry" and event.get("activation") == "JOIN"
        )
        blocked_terminal = next(
            event for event in events if event.get("kind") == "terminal" and event.get("activation") == "JOIN"
        )
        assert blocked_entry == {
            "activation": "JOIN",
            "closed_unstarted": True,
            "kind": "entry",
            "node": "J",
            "node_kind": "operation",
            "occurrence": 4,
            "parent": None,
            "state_category": "blocked",
            "state_outcome": None,
            "target": "A",
        }
        assert blocked_terminal == {
            "activation": "JOIN",
            "attempt": None,
            "category": "blocked",
            "kind": "terminal",
            "outcome": None,
            "reasons": ["prerequisite"],
            "structural": False,
            "target": "A",
        }
    assert all(
        not (event.get("kind") == "terminal" and event.get("activation") == "JOIN")
        for event in cast(list[reference.Obj], case("map_item_evidence/expansion_open")["events"])
    )
    nested_events = cast(list[reference.Obj], case("map_item_evidence/nested_path")["events"])
    assert not any(
        event.get("activation") == "WRAP"
        and event.get("kind") in {"port", "input_producer"}
        and event.get("port") == "context"
        for event in nested_events
    )
    assert any(
        event.get("activation") == "WRAP" and event.get("kind") == "port" and event.get("port") == "nested_members"
        for event in nested_events
    )
    wrapper_membership = next(
        event for event in nested_events if event.get("kind") == "membership" and event.get("parent") == "WRAP"
    )
    assert wrapper_membership["expansion_outcome"] is None


def test_map_item_currentness_and_incomplete_execution_withhold() -> None:
    assert row("map_item_evidence/item_stale")["withholding"] == ["stale_evidence"]
    assert row("map_item_evidence/item_unknown")["withholding"] == ["assessment_unknown"]
    for name in ("member_non_success", "member_blocked_unreached"):
        assert row(f"map_item_evidence/{name}")["withholding"] == ["terminal_failure"]
    for name in ("expansion_failed", "expansion_overflow"):
        assert row(f"map_item_evidence/{name}")["withholding"] == ["terminal_failure"]
    assert row("map_item_evidence/expansion_open")["withholding"] == ["incomplete_membership"]
    for name in ("expansion_failed", "expansion_overflow", "expansion_open"):
        assert case(f"map_item_evidence/{name}")["comparison_scope"] == "neutral_only"
    assert case("map_item_evidence/member_blocked_unreached")["comparison_scope"] == "neutral_only"


def test_map_item_paths_domains_and_candidate_ancestry_are_exact() -> None:
    for name in ("nested_path", "distinct_outcome_port", "two_independent_maps"):
        assert row(f"map_item_evidence/{name}")["qualification"] == "met"
    two_map_verified = cast(list[reference.Obj], result("map_item_evidence/two_independent_maps")["verified"])
    assert {
        (fact["activation"], fact["promise"], fact["subject"]["producer"])
        for fact in two_map_verified
        if "subject" in fact
    } == {("M0", "P_ITEM", "MAPITEM:0"), ("Z0", "P_ITEM_2", "MAPITEM2:0")}
    for name in (
        "different_final_ancestry",
        "candidate_passthrough_unrelated",
        "two_maps_no_crossproduct",
        "missing_submission",
    ):
        assert row(f"map_item_evidence/{name}")["withholding"] == ["missing_assessment"]
    assert result("map_item_evidence/foreign_submission") == {"code": "foreign_owner", "status": "rejected"}
    assert result("map_item_evidence/repeated_submission") == {"code": "duplicate", "status": "rejected"}
    assert result("map_item_evidence/copied_member_submission") == {"code": "duplicate", "status": "rejected"}
    assert result("map_item_evidence/missing_fact") == {"code": "missing", "status": "rejected"}
    assert result("map_item_bounds/submissions_one_over") == {"code": "limit_exceeded", "status": "rejected"}
    assert result("map_item_bounds/verified_one_over") == {"code": "limit_exceeded", "status": "rejected"}
