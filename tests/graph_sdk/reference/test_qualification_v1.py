# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-tests for the independent fact-driven qualification reference."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import cast

HERE = Path(__file__).parent
CORPUS = HERE / "qualification_v1_cases.json"
MANIFEST = HERE / "qualification_v1_manifest.json"
GENERATOR = HERE / "qualification_v1.py"
SPEC = importlib.util.spec_from_file_location("qualification_v1_reference", GENERATOR)
assert SPEC is not None and SPEC.loader is not None
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)


def case(cid: str) -> reference.Obj:
    return next(x for x in reference.CASES if x["case_id"] == cid)


def result(cid: str) -> reference.Obj:
    return case(cid)["expected"]


def row(cid: str, target: str = "A") -> reference.Obj:
    return next(x for x in cast(list[reference.Json], result(cid)["targets"]) if x["target"] == target)


def test_generation_manifest_and_independence() -> None:
    data = json.loads(CORPUS.read_bytes())
    assert reference.canonical_bytes(reference.generate_cases()) == CORPUS.read_bytes()
    for item in data:
        actual = (
            reference.admit(item["declaration"])
            if item["boundary"] == "admission"
            else reference.reduce(item["declaration"], item["events"])
        )
        assert actual == item["expected"], item["case_id"]
        for trace in item["traces"]:
            assert reference.reduce(item["declaration"], trace["events"]) == trace["expected"]
    manifest = json.loads(MANIFEST.read_bytes())
    assert manifest["case_count"] == len(data)
    assert manifest["corpus_sha256"] == hashlib.sha256(CORPUS.read_bytes()).hexdigest()
    assert manifest["structural_contract_sha256"] == reference.STRUCTURAL_CONTRACT_SHA256
    assert manifest["map_item_evidence_contract_sha256"] == reference.MAP_ITEM_EVIDENCE_CONTRACT_SHA256
    assert Counter(x["family"] for x in data) == reference.FAMILY_COUNTS
    imports = [
        ast.unparse(node)
        for node in ast.walk(ast.parse(GENERATOR.read_text()))
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert all("anonymizer" not in value for value in imports)
    assert "expected_error" not in GENERATOR.read_text()


def test_exact_execution_joins_and_environment() -> None:
    expected = {
        "missing_consumed_port_fact": "contradictory",
        "evidence_port_node": "contradictory",
        "evidence_port_target": "foreign_owner",
        "entry_node": "foreign_owner",
        "entry_target": "contradictory",
        "terminal_outcome": "contradictory",
        "terminal_target": "contradictory",
        "missing_absence_environment": "missing",
        "missing_configuration_environment": "contradictory",
        "missing_state_environment": "contradictory",
    }
    for name, code in expected.items():
        assert result(f"authentication/{name}") == {"code": code, "status": "rejected"}
    for name in ("producer_node", "producer_activation", "final_node"):
        assert result(f"final_output/{name}")["status"] == "rejected"


def test_requirement_and_assessment_authority() -> None:
    for name in ("requirement_subject_port", "requirement_consumed_port"):
        assert result(f"admission/{name}") == {"code": "contradictory", "status": "rejected"}
    assert result("assessment/wrong_kind_coverage")["code"] == "unsupported"
    for name in ("foreign_target", "foreign_evidence", "foreign_subject"):
        assert result(f"assessment/{name}")["status"] == "rejected"
    for name in ("node_mismatch", "outcome_mismatch", "promise_mismatch"):
        assert result(f"joins/{name}") == {"code": "unsupported", "status": "rejected"}
    assert result("assessment/absence_query") == {"code": "missing", "status": "rejected"}
    assert result("final_output/wrong_outcome") == {"code": "contradictory", "status": "rejected"}
    assert result("assessment/foreign_evidence") == {"code": "contradictory", "status": "rejected"}
    assert case("assessment/foreign_evidence")["comparison_scope"] == "production_boundary"


def test_admission_uses_public_constructor_and_preparation_boundaries() -> None:
    assert result("admission/fixed_point") == {"code": "invalid_value", "status": "rejected"}
    assert result("admission/missing_promise") == {
        "code": "protection_ineligible",
        "status": "rejected",
    }


def test_retained_assessment_inventory_is_independent_of_ordered_submissions() -> None:
    expected_submissions = {
        "assessment/missing": [],
        "assessment/duplicate": ["F:A:P", "F:A:P"],
        "bounds/submissions_exact": ["F:A:P"],
        "bounds/submissions_one_over": ["F:A:P"],
    }
    for case_id, submissions in expected_submissions.items():
        item = case(case_id)
        state = reference.initial(item["declaration"])
        for event in cast(list[reference.Obj], item["events"]):
            assert reference.advance(state, event) is None
        assert list(state["assessment_facts"]) == ["F:A:P"]
        assert state["assessment_submissions"] == submissions
    assert result("assessment/missing")["status"] == "accepted"
    assert row("assessment/missing")["withholding"] == ["missing_assessment"]
    assert result("assessment/duplicate") == {"code": "duplicate", "status": "rejected"}
    assert result("bounds/submissions_one_over") == {"code": "limit_exceeded", "status": "rejected"}
    baseline = case("release/protection_success")
    foreign = [dict(event) for event in cast(list[reference.Obj], baseline["events"])]
    next(event for event in foreign if event["kind"] == "assessment_submission")["fact"] = "FOREIGN"
    assert reference.reduce(baseline["declaration"], foreign) == {
        "code": "foreign_owner",
        "status": "rejected",
    }
    unretained = [
        dict(event) for event in cast(list[reference.Obj], baseline["events"]) if event["kind"] != "assessment"
    ]
    assert reference.reduce(baseline["declaration"], unretained) == {
        "code": "foreign_owner",
        "status": "rejected",
    }
    absent = [
        dict(event)
        for event in cast(list[reference.Obj], baseline["events"])
        if event["kind"] not in {"assessment", "assessment_submission"}
    ]
    assert reference.reduce(baseline["declaration"], absent) == {
        "code": "missing",
        "status": "rejected",
    }

    multi = case("propagation/independent")
    duplicate_a_missing_b = [
        dict(event)
        for event in cast(list[reference.Obj], multi["events"])
        if not (
            event["kind"] == "assessment"
            and event.get("target") == "B"
            or event["kind"] == "assessment_submission"
            and event.get("fact") == "F:B:P"
        )
    ]
    duplicate_a = dict(
        next(
            event
            for event in cast(list[reference.Obj], multi["events"])
            if event["kind"] == "assessment" and event.get("target") == "A"
        )
    )
    duplicate_a["fact"] = "F:A:P:DUPLICATE"
    duplicate_a_missing_b.insert(
        next(i for i, event in enumerate(duplicate_a_missing_b) if event["kind"] == "seal_revision"),
        duplicate_a,
    )
    assert reference.reduce(multi["declaration"], duplicate_a_missing_b) == {
        "code": "duplicate",
        "status": "rejected",
    }


def test_duplicate_state_key_uses_distinct_retained_revisions() -> None:
    events = cast(list[reference.Obj], case("bounds/duplicate_state_key")["events"])
    revisions = [event for event in events if event.get("kind") == "revision" and event.get("key") == "read"]
    assert [event["value"] for event in revisions] == [1, 2]
    assert result("bounds/duplicate_state_key") == {"code": "duplicate", "status": "rejected"}


def test_assessment_inventory_follows_successful_producing_occurrences() -> None:
    for case_id, members in (
        ("membership/closed_1", ("M0",)),
        ("provenance/map_item_two_members", ("M0", "M1")),
    ):
        item = case(case_id)
        entries = {
            event["activation"]: event
            for event in cast(list[reference.Obj], item["events"])
            if event.get("kind") == "entry"
        }
        assert all(entries[member]["node"] == "MEM" for member in members)
        assert {
            event["activation"]
            for event in cast(list[reference.Obj], item["events"])
            if event.get("kind") == "assessment"
        } == {"ROOT:A"}
        assert result(case_id)["status"] == "accepted"

    for count in range(3):
        case_id = f"assessment/dynamic_occurrences_{count}"
        item = case(case_id)
        events = cast(list[reference.Obj], item["events"])
        retained = {event["activation"] for event in events if event.get("kind") == "assessment"}
        assert retained == {"ROOT:A", *(f"M{index}" for index in range(count))}
        for index in range(count):
            member = f"M{index}"
            item_port = next(
                event
                for event in events
                if event.get("kind") == "port" and event.get("activation") == member and event.get("port") == "item"
            )
            subject_port = next(
                event
                for event in events
                if event.get("kind") == "port" and event.get("activation") == member and event.get("port") == "subject"
            )
            member_assessment = next(
                event for event in events if event.get("kind") == "assessment" and event.get("activation") == member
            )
            evidence = next(
                event for event in events if event.get("kind") == "provenance" and event.get("key") == f"EVID:{member}"
            )
            assert item_port["role"] == "artifact" and item_port["artifact"] == f"MI{index}v1"
            assert subject_port["role"] == "candidate" and subject_port["artifact"] == "Av0"
            assert member_assessment["consumed"] == {"subject": "Av0"}
            assert member_assessment["environment"] == {
                "absences": {},
                "configurations": {"MN": "c0"},
                "state": {},
            }
            assert evidence["parents"] == ["ROOT:A:subject"]
        assert result(case_id)["status"] == "accepted"

    assert result("assessment/dynamic_occurrence_missing") == {"code": "missing", "status": "rejected"}
    assert result("assessment/dynamic_occurrence_duplicate") == {"code": "duplicate", "status": "rejected"}


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
        assert result(f"{case_id}_injected") == {"code": "unsupported", "status": "rejected"}

    blocked_entry = next(
        event
        for event in cast(list[reference.Obj], case("assessment/dynamic_blocked_unreached")["events"])
        if event.get("kind") == "entry" and event.get("activation") == "M0"
    )
    assert blocked_entry["closed_unstarted"] is True
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
        assert row(case_id)["withholding"] == ["incomplete_membership", "missing_assessment"]

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


def test_root_input_passthrough_does_not_relabel_unused_operation_output_candidate() -> None:
    item = case("map_item_evidence/candidate_passthrough_unrelated")
    events = cast(list[reference.Obj], item["events"])
    ports = {
        (event["activation"], event["node"], event["port"]): event["role"]
        for event in events
        if event.get("kind") == "port"
    }
    assert ports[("ROOT:A", "N", "subject")] == "candidate"
    assert ports[("ROOT:A", "N", "result")] == "artifact"
    assert next(event for event in events if event.get("kind") == "final")["producer"] == "ROOT:A:subject"
    assert row("map_item_evidence/candidate_passthrough_unrelated")["withholding"] == ["missing_assessment"]


def test_validity_selectivity_and_output_independence() -> None:
    for dependency in ("candidate", "evidence", "consumed", "absence", "configuration", "state"):
        for expected_validity in ("current", "stale", "unknown"):
            verified = cast(list[reference.Obj], result(f"validity/{dependency}_{expected_validity}")["verified"])
            assert verified[0]["validity"] == expected_validity
    assert row("selective/a_only")["qualification"] == "met"
    assert "stale_evidence" in cast(list[str], row("selective/b_only")["withholding"])
    assert "stale_evidence" in cast(list[str], row("selective/a_b")["withholding"])
    assert row("validity/evidence_output_removed")["qualification"] == "unknown"
    assert row("validity/evidence_output_replaced")["qualification"] == "unmet"
    assert row("validity/configuration_type_changed")["qualification"] == "unmet"
    assert row("selective/intermediate_a0_final_a1")["withholding"] == ["missing_assessment"]


def test_membership_exact_outcome_and_terminal_families() -> None:
    for count in (0, 1, 2):
        assert row(f"membership/closed_{count}")["qualification"] == "met"
    assert row("membership/nested_closed")["qualification"] == "met"
    for name in (
        "wrong_expansion_outcome",
        "open",
        "nested_open",
        "missing_expander_terminal",
        "missing_member_terminal",
    ):
        assert "incomplete_membership" in cast(list[str], row(f"membership/{name}")["withholding"])
    for category in ("blocked", "cancelled", "lost", "inconsistent"):
        assert "terminal_failure" in cast(list[str], row(f"membership/terminal_{category}")["withholding"])
    assert result("membership/foreign_target")["code"] == "foreign_owner"
    assert result("membership/duplicate_terminal")["code"] == "duplicate"
    assert result("membership/duplicate_member")["code"] == "duplicate"


def test_structural_terminal_and_actual_membership_accounting() -> None:
    root_terminal = next(x for x in case("release/protection_success")["events"] if x.get("kind") == "terminal")
    assert root_terminal["structural"] is False
    assert root_terminal["attempt"] == "TASK:ROOT:A"
    valid_events = case("structural/valid_container")["events"]
    container_terminal = next(x for x in valid_events if x.get("kind") == "terminal" and x.get("activation") == "MAP")
    empty_membership = next(x for x in valid_events if x.get("kind") == "membership" and x.get("parent") == "MAP")
    assert container_terminal["structural"] is True
    assert container_terminal["attempt"] is None
    assert empty_membership["members"] == [] and empty_membership["status"] == "closed"
    expected = {
        "non_bool_flag": "invalid_type",
        "success_with_reason": "invalid_value",
        "container_attempt_forbidden": "contradictory",
        "structural_on_operation": "contradictory",
        "operation_on_container": "contradictory",
        "operation_success_missing_attempt": "contradictory",
        "operation_failed_missing_attempt": "contradictory",
        "operation_cancelled_missing_attempt": "contradictory",
        "operation_lost_missing_attempt": "contradictory",
        "category_mismatch": "contradictory",
        "failure_missing_reason": "contradictory",
        "node_kind_mismatch": "contradictory",
    }
    for name, code in expected.items():
        assert result(f"structural/{name}") == {"code": code, "status": "rejected"}
    for status in ("failed", "overflow"):
        item = case(f"structural/{status}_actual_members")
        membership = next(x for x in item["events"] if x.get("kind") == "membership" and x.get("parent") == "MAP")
        assert membership["status"] == status
        assert membership["members"] == ["M0"]
        assert any(x.get("kind") == "reservation" and x.get("activation") == "M1" for x in item["events"])
        assert all(not (x.get("kind") == "entry" and x.get("activation") == "M1") for x in item["events"])
        assert "incomplete_membership" not in row(f"structural/{status}_actual_members")["withholding"]
        assert "terminal_failure" in row(f"structural/{status}_actual_members")["withholding"]
        terminal = next(x for x in item["events"] if x.get("kind") == "terminal" and x.get("activation") == "MAP")
        assert (terminal["category"], terminal["reasons"]) == (
            ("failure", ["execution_failed"]) if status == "failed" else ("inconsistent", ["contradictory"])
        )


def test_request_recovery_is_exact_and_latest() -> None:
    for purpose in ("retry", "correction", "failover"):
        assert row(f"request/{purpose}_recovered")["qualification"] == "met"
    for name in (
        "permanent_retry",
        "retryable_failover",
        "malformed_retry",
        "changed_policy",
        "changed_association",
        "latest_failure_authority",
        "lost_known_usage",
        "success_unknown_usage",
    ):
        assert "request_accounting" in cast(list[str], row(f"request/{name}")["withholding"])
    assert "inconsistent_attribution" in cast(list[str], row("request/foreign_association")["withholding"])


def test_cleanup_owner_and_localization() -> None:
    assert "cleanup_verification" in cast(list[str], row("cleanup/sdk_left_open")["withholding"])
    assert row("cleanup/caller_left_open")["qualification"] == "met"
    assert row("cleanup/transport_failed")["qualification"] == "met"
    assert "cleanup_verification" in cast(list[str], row("cleanup/unrelated_c", "C")["withholding"])
    assert row("cleanup/unrelated_c", "A")["qualification"] == "met"
    assert "inconsistent_attribution" in cast(list[str], row("cleanup/foreign_target")["withholding"])


def test_decision_artifact_ownership_and_withheld_target_rules() -> None:
    assert result("decisions/missing_artifact")["code"] == "missing"
    assert result("decisions/foreign_target")["code"] == "foreign_owner"
    assert result("decisions/shared_deduplicated")["required_decisions"] == ["Dv0"]
    assert result("decisions/withheld_unrelated")["required_decisions"] == []
    assert result("decisions/withheld_linked")["required_decisions"] == ["Dv0"]
    assert row("decisions/withheld_linked", "A")["required_decisions"] == []
    assert row("decisions/withheld_linked", "B")["required_decisions"] == ["Dv0"]


def test_provenance_types_identity_and_producers() -> None:
    for name in (
        "root",
        "initial_collection",
        "map_item",
        "map_item_two_members",
        "subgraph",
        "identity_alias",
        "version_edge",
        "bound_n0",
        "bound_n1",
        "multiple_bound_artifacts",
        "shared_context_identity",
    ):
        assert row(f"provenance/{name}")["qualification"] == "met"
    assert result("provenance/missing_producer")["code"] == "missing"
    assert result("provenance/duplicate_producer")["code"] == "duplicate"
    assert result("provenance/mismatched_bound_root")["code"] == "contradictory"
    assert result("provenance/foreign_bound_target")["code"] == "foreign_owner"
    bound = next(
        x
        for x in cast(list[reference.Json], case("provenance/bound_n0")["events"])
        if x.get("kind") == "provenance" and x.get("source") == "bound_input"
    )
    assert bound["artifact"] == "XAv0"
    assert bound["binding_artifact"] == {"declaration": "D0", "key": 0, "version": 1}
    bound_receipt = next(
        x
        for x in cast(list[reference.Obj], case("provenance/bound_n0")["events"])
        if x.get("kind") == "binding_receipt"
    )
    assert cast(list[reference.Obj], bound_receipt["artifacts"])[0]["source"] == "SRC:D0@1"
    initial = next(
        x
        for x in cast(list[reference.Json], case("provenance/initial_collection")["events"])
        if x.get("kind") == "provenance" and x.get("source") == "initial_collection"
    )
    assert initial["declaration"] == "D1"
    assert initial["parents"] == ["BOUND:D1:0", "BOUND:D1:1"]
    mapped = next(
        x
        for x in cast(list[reference.Json], case("provenance/map_item")["events"])
        if x.get("kind") == "provenance" and x.get("source") == "map_item"
    )
    assert {key: mapped[key] for key in ("expander", "member", "target", "port", "item_key", "item_version")} == {
        "expander": "MAP",
        "member": "M0",
        "target": "A",
        "port": "item",
        "item_key": 0,
        "item_version": 1,
    }
    assert mapped["parents"] == ["OP:MAP:A:members"]
    map_case = case("provenance/map_item")
    assert cast(list[reference.Obj], map_case["declaration"]["map_inputs"])[0] == {
        "expander": "EXP",
        "item_input": "item",
        "membership_port": "members",
        "outcome": "ok",
    }
    map_producer = next(
        x
        for x in cast(list[reference.Obj], map_case["events"])
        if x.get("kind") == "provenance" and x.get("key") == "OP:MAP:A:members"
    )
    assert (map_producer["activation"], map_producer["port"]) == ("MAP", "members")
    two = case("provenance/map_item_two_members")
    mapped_two = [
        x
        for x in cast(list[reference.Obj], two["events"])
        if x.get("kind") == "provenance" and x.get("source") == "map_item"
    ]
    assert [(x["member"], x["item_key"], x["item_version"], x["parents"]) for x in mapped_two] == [
        ("M0", 0, 1, ["OP:MAP:A:members"]),
        ("M1", 1, 1, ["OP:MAP:A:members"]),
    ]
    assert result("provenance/bound_invocation_ref_substitution")["code"] == "invalid_type"
    assert result("provenance/bound_invalid_version")["code"] == "invalid_value"
    assert result("provenance/bound_unretained_reference")["code"] == "missing"
    assert result("provenance/bound_foreign_receipt_source")["code"] == "foreign_owner"
    assert result("provenance/initial_unretained_bound_parent")["code"] == "missing"
    for name in (
        "initial_missing_bound_parent",
        "initial_wrong_declaration",
        "map_wrong_member",
        "map_wrong_port",
        "map_negative_item_key",
        "map_zero_item_version",
        "map_wrong_expander",
        "map_missing_collection_parent",
        "map_wrong_collection_parent",
        "map_multiple_collection_parents",
        "map_collection_identity_mismatch",
        "map_unrelated_collection_producer",
        "map_wrong_membership_port",
        "map_wrong_expansion_outcome",
        "map_wrong_producer_activation",
        "map_swapped_member_items",
        "map_wrong_target",
    ):
        assert result(f"provenance/{name}")["code"] == "contradictory"


def test_result_owner_and_execution_only_empty_path() -> None:
    for field in ("plan", "invocation", "graph"):
        assert result(f"record/{field}") == {"code": "foreign_owner", "status": "rejected"}
    for cid in ("release/execution_only", "release/execution_only_empty"):
        assert result(cid)["verified"] == []
        assert result(cid)["qualified"] == []
        assert row(cid)["qualification"] == "not_assessed"
        assert row(cid)["candidate"] is None
        assert row(cid)["artifact_available"] is True
        assert row(cid)["withholding"] == ["execution_only"]
        assert "local_eligible" not in row(cid)
        assert case(cid)["declaration"]["productions"] == []
        events = cast(list[reference.Obj], case(cid)["events"])
        assert all(item.get("kind") != "assessment" for item in events)
        assert all(item.get("kind") != "revision" or item.get("collection") == "artifacts" for item in events)


def test_all_bounds_and_duplicate_revision_inputs() -> None:
    assert result("bounds/exact")["status"] == "accepted"
    for name in ("productions", "coverage", "verified", "required_decisions", "fixed_point"):
        assert result(f"bounds/{name}_exact")["status"] == "accepted"
        assert result(f"bounds/{name}_one_over") == {"code": "limit_exceeded", "status": "rejected"}
    for name in ("ports", "consumed", "edges", "submissions", "revisions", "absences"):
        assert result(f"bounds/{name}_exact")["status"] == "accepted"
        assert result(f"bounds/{name}_one_over") == {"code": "limit_exceeded", "status": "rejected"}
    for collection in ("artifacts", "absences", "configurations", "state"):
        assert result(f"bounds/duplicate_{collection}_key") == {"code": "duplicate", "status": "rejected"}


def test_propagation_and_true_commutation() -> None:
    independent = case("propagation/independent")
    assert independent["declaration"]["atomic"] == [["A"], ["B"], ["C"]]
    assert row("propagation/independent", "B")["qualification"] == "met"
    assert "atomic_group" not in cast(list[str], row("propagation/independent", "A")["withholding"])
    assert "dependency" in cast(list[str], row("propagation/a_to_b", "B")["withholding"])
    assert row("propagation/b_to_a", "B")["qualification"] == "met"
    assert "atomic_group" in cast(list[str], row("propagation/atomic_ab", "B")["withholding"])
    assert case("propagation/atomic_ab")["declaration"]["atomic"] == [["A", "B"], ["C"]]
    assert case("propagation/atomic_bc")["declaration"]["atomic"] == [["B", "C"], ["A"]]
    assert result("admission/incomplete_atomic_partition") == {"code": "contradictory", "status": "rejected"}
    assert result("admission/binding_foreign_node") == {"code": "foreign_owner", "status": "rejected"}
    assert result("admission/initial_collection_without_binding") == {"code": "missing", "status": "rejected"}
    assert result("admission/duplicate_map_input") == {"code": "duplicate", "status": "rejected"}
    assert result("admission/duplicate_binding_declaration") == {"code": "duplicate", "status": "rejected"}
    assert result("admission/empty_atomic_group") == {"code": "invalid_value", "status": "rejected"}
    commuted = case("commutation/independent_revisions")
    alternate = cast(list[reference.Json], commuted["traces"])[0]
    assert alternate["events"] != commuted["events"]
    assert alternate["expected"] == commuted["expected"]
    artifact_keys = {x["key"] for x in commuted["events"] if x.get("kind") == "artifact"}
    assert all(
        x["key"] in artifact_keys
        for x in commuted["events"]
        if x.get("kind") == "revision" and x.get("collection") == "artifacts"
    )


def test_root_partition_revision_and_stray_terminal_negatives() -> None:
    root = result("release/protection_success")["record"]["memberships"]["__ROOT__"]
    assert root == {
        "closed": True,
        "expansion_outcome": None,
        "members": ["ROOT:A"],
        "parent": None,
        "status": "closed",
        "target": None,
    }
    assert result("membership/missing_root")["code"] == "missing"
    assert result("membership/orphan_entry")["code"] == "missing"
    assert result("membership/stray_terminal")["code"] == "missing"
    assert result("membership/cross_target_member")["code"] == "foreign_owner"
    assert result("revision/invented_artifact")["code"] == "missing"
    assert result("revision/invented_absence")["code"] == "unsupported"
    assert result("revision/invented_configuration")["code"] == "missing"
    assert result("revision/invented_state")["code"] == "unsupported"


def test_occurrence_roles_provenance_and_support_selection() -> None:
    assert row("roles/global_role_ignored")["qualification"] == "met"
    assert result("roles/distinct_evidence_wrong_role")["code"] == "contradictory"
    assert result("roles/subject_decision_collision")["code"] == "contradictory"
    assert result("roles/consumed_unsupported_role")["code"] == "unsupported"
    assert result("roles/candidate_evidence_alias")["qualified"][0]["evidence"] == ["Av0"]


def test_evidence_output_dependencies_exactly_match_promise_consumption() -> None:
    baseline = case("release/protection_success")
    production = cast(list[reference.Obj], baseline["declaration"]["productions"])[0]
    evidence = next(
        dependency
        for dependency in cast(list[reference.Obj], baseline["declaration"]["output_dependencies"])
        if dependency["port"] == production["evidence_port"]
    )
    assert set(cast(list[str], evidence["inputs"])) == set(cast(list[str], production["consumed_ports"]))
    assert result("admission/evidence_dependency_consumed_mismatch") == {
        "code": "contradictory",
        "status": "rejected",
    }
    distinct_alias = result("roles/candidate_decision_distinct_occurrence_alias")
    assert distinct_alias["qualified"][0]["candidate"] == "Av0"
    assert distinct_alias["required_decisions"] == ["Av0"]
    assert result("provenance/missing_parent_artifact")["code"] == "missing"
    assert result("provenance/mismatched_source_artifact")["code"] == "contradictory"
    unrelated = result("selection/unrelated_assessment")
    assert [x["evidence_artifact"] for x in unrelated["verified"]] == ["EAv0", "E2v0"]
    assert unrelated["qualified"][0]["evidence"] == ["EAv0"]
    assert unrelated["required_decisions"] == []
    intermediate = result("selective/intermediate_a0_final_a1")
    assert [x["evidence_artifact"] for x in intermediate["verified"]] == ["EAv0"]
    assert intermediate["qualified"] == []


def test_request_association_set_order_is_irrelevant() -> None:
    for target in ("A", "B"):
        assert row("request/association_order_invariant", target)["qualification"] == "met"
        assert row("request/association_order_invariant", target)["withholding"] == []


def test_v8_exact_output_and_structural_projection() -> None:
    assert row("provenance/root")["qualification"] == "met"
    direct = case("provenance/root")
    assert (
        next(x for x in cast(list[reference.Obj], direct["events"]) if x.get("kind") == "final")["producer"]
        == "ROOT:A:subject"
    )
    assert row("provenance/identity_alias")["qualification"] == "met"
    assert row("provenance/subgraph")["qualification"] == "met"
    subgraph = case("provenance/subgraph")
    assert {x["activation"] for x in cast(list[reference.Obj], subgraph["events"]) if x.get("kind") == "entry"} >= {
        "SUB",
        "BODY",
    }
    assert (
        next(
            x
            for x in cast(list[reference.Obj], subgraph["events"])
            if x.get("kind") == "input_producer" and x["activation"] == "BODY"
        )["producer"]
        == "ROOT:A:context"
    )
    expected = {
        "missing_executed_input_producer": "missing",
        "extra_dependency_parent": "contradictory",
        "swapped_dependency_parents": "contradictory",
        "subgraph_wrong_body_projection": "contradictory",
        "subgraph_wrong_input_passthrough": "contradictory",
        "direct_root_wrong_input": "contradictory",
    }
    for name, code in expected.items():
        assert result(f"provenance/{name}") == {"code": code, "status": "rejected"}


def test_v8_closed_failure_request_cleanup_roles_and_expanders() -> None:
    for category in ("blocked", "cancelled", "lost", "inconsistent"):
        item = row(f"membership/terminal_{category}")
        assert item["completion"] == "closed" and item["qualification"] == "unmet"
    for status in ("failed", "overflow"):
        assert row(f"structural/{status}_actual_members")["completion"] == "closed"
    assert row("request/final_failure")["qualification"] == "unmet"
    assert row("request/lost_unknown")["qualification"] == "unknown"
    assert row("cleanup/empty_transport_only")["qualification"] == "met"
    for purpose in ("verification", "accounting"):
        item = row(f"cleanup/empty_{purpose}")
        assert item["qualification"] == "unknown"
        assert "inconsistent_attribution" in cast(list[str], item["withholding"])
    assert row("roles/root_bound_distinct_evidence_candidate")["qualification"] == "met"
    operation_map = case("provenance/map_item_two_members_operation_expander")
    terminal_value = next(
        x
        for x in cast(list[reference.Obj], operation_map["events"])
        if x.get("kind") == "terminal" and x.get("activation") == "MAP"
    )
    assert terminal_value["structural"] is False and terminal_value["attempt"] == "TASK:MAP"
    assert row("provenance/map_item_two_members_operation_expander")["qualification"] == "met"


def test_v9_set_provenance_inventory_and_structural_passthrough() -> None:
    assert row("provenance/parent_order_invariant")["qualification"] == "met"
    assert result("provenance/swapped_dependency_parents") == {"code": "contradictory", "status": "rejected"}
    assert result("provenance/extra_input_producer") == {"code": "contradictory", "status": "rejected"}
    assert row("provenance/subgraph_nested_workflow_input_passthrough")["qualification"] == "met"
    assert result("provenance/subgraph_passthrough_missing_capture") == {"code": "missing", "status": "rejected"}
    assert result("provenance/subgraph_passthrough_wrong_capture") == {
        "code": "contradictory",
        "status": "rejected",
    }
    passthrough = case("provenance/subgraph_nested_workflow_input_passthrough")
    assert all(x.get("activation") != "BODY" for x in cast(list[reference.Obj], passthrough["events"]))
    assert next(
        x
        for x in cast(list[reference.Obj], passthrough["events"])
        if x.get("kind") == "provenance" and x.get("key") == "PASSTHROUGH"
    )["parents"] == ["ROOT:A:context"]


def test_v9_request_uncertainty_and_multi_promise_support() -> None:
    assert row("request/lost_known_usage")["qualification"] == "unknown"
    assert row("request/final_failure")["qualification"] == "unmet"
    for purpose in ("initial_binding", "adaptive_retrieval", "repair"):
        assert row(f"request/fresh_{purpose}")["qualification"] == "met"
    partial = result("assessment/partial_promise_only")
    assert row("assessment/partial_promise_only")["qualification"] == "unmet"
    assert row("assessment/partial_promise_only")["verified"] == []
    assert [item["evidence_artifact"] for item in partial["verified"]] == ["EAv0"]
    assert row("assessment/complete_promise_only")["verified"] == ["E2v0"]
    both = result("assessment/partial_and_complete_promises")
    assert {item["evidence_artifact"] for item in both["verified"]} == {"EAv0", "E2v0"}
    assert row("assessment/partial_and_complete_promises")["verified"] == ["E2v0"]


def test_v9_initial_binding_cleanup_accounting() -> None:
    for target in ("A", "C"):
        assert row("cleanup/binding_closed", target)["qualification"] == "met"
    assert "cleanup_accounting" in cast(list[str], row("cleanup/binding_failed_local_a", "A")["withholding"])
    assert row("cleanup/binding_failed_local_a", "C")["qualification"] == "met"
    for name in ("binding_missing_association", "binding_foreign_association", "binding_extra_association"):
        for target in ("A", "C"):
            item = row(f"cleanup/{name}", target)
            assert item["qualification"] == "unknown"
            assert "inconsistent_attribution" in cast(list[str], item["withholding"])
    assert result("cleanup/binding_duplicate_association") == {"code": "duplicate", "status": "rejected"}


def test_v11_baseline_is_realizable_by_declared_identity_consumption_and_coverage() -> None:
    baseline = case("release/protection_success")
    result_dependency = next(
        item
        for item in cast(list[reference.Obj], baseline["declaration"]["output_dependencies"])
        if item["node"] == "N" and item["port"] == "result"
    )
    assert result_dependency["identity_input"] == "subject"
    assert set(cast(list[str], result_dependency["inputs"])) == {"subject", "context"}
    production = cast(list[reference.Obj], baseline["declaration"]["productions"])[0]
    assessment_fact = next(
        item for item in cast(list[reference.Obj], baseline["events"]) if item.get("kind") == "assessment"
    )
    assert production["consumed_ports"] == ["context"]
    assert set(assessment_fact["consumed"]) == {"context"}
    assert assessment_fact["coverage"] == production["coverage"] == ["K0", "K1"]
    assert result("assessment/incomplete_coverage") == {"code": "contradictory", "status": "rejected"}
    for case_id in ("decisions/direct", "provenance/map_item_two_members", "provenance/bound_n0"):
        item = case(case_id)
        dependency = next(
            value
            for value in cast(list[reference.Obj], item["declaration"]["output_dependencies"])
            if value["node"] == "N" and value["port"] == "result"
        )
        assert dependency["identity_input"] == "subject"


def test_v11_materialized_version_lineages_and_boundaries() -> None:
    predecessor_path = HERE / "qualification_v1_v10_ids.json"
    predecessor_bytes = predecessor_path.read_bytes()
    predecessor = json.loads(predecessor_bytes)
    assert hashlib.sha256(predecessor_bytes).hexdigest() == reference.V10_IDS_SHA256
    v10_ids = set(predecessor)
    assert v10_ids <= {item["case_id"] for item in reference.CASES}

    candidate = case("validity/candidate_stale")
    candidate_result = candidate["expected"]
    assert candidate_result["qualified"] == []
    candidate_row = cast(list[reference.Obj], candidate_result["targets"])[0]
    assert candidate_row["qualification"] == "unknown"
    assert candidate_row["artifact_available"] is False
    assert candidate_row["withholding"] == ["missing_candidate", "stale_evidence"]
    omitted = row("validity/candidate_stale_no_assessment")
    assert omitted["qualification"] == "unknown"
    assert omitted["artifact_available"] is False
    assert omitted["withholding"] == ["missing_assessment", "missing_candidate"]
    omitted_case = case("validity/candidate_stale_no_assessment")
    omitted_fact = next(
        item for item in cast(list[reference.Obj], omitted_case["events"]) if item.get("kind") == "assessment"
    )
    assert omitted_fact["fact"] == "F:A:P"
    assert all(
        item.get("kind") != "assessment_submission" for item in cast(list[reference.Obj], omitted_case["events"])
    )
    state = reference.initial(omitted_case["declaration"])
    for event in cast(list[reference.Obj], omitted_case["events"]):
        assert reference.advance(state, event) is None
    assert len(state["assessment_facts"]) == 1
    assert state["assessment_submissions"] == []
    assessment_fact = next(
        item for item in cast(list[reference.Obj], candidate["events"]) if item.get("kind") == "assessment"
    )
    final_fact = next(item for item in cast(list[reference.Obj], candidate["events"]) if item.get("kind") == "final")
    assert assessment_fact["subject_artifact"] == final_fact["candidate"]
    selected = next(
        item
        for item in cast(list[reference.Obj], candidate["events"])
        if item.get("kind") == "revision" and item.get("key") == "A"
    )
    final_artifact = next(
        item
        for item in cast(list[reference.Obj], candidate["events"])
        if item.get("kind") == "artifact" and item.get("ref") == final_fact["candidate"]
    )
    assert selected["value"] != final_artifact["version"]
    assert final_fact["producer"] == "OUT:A"

    for cid in ("validity/evidence_stale", "validity/consumed_stale", "provenance/version_edge"):
        item = case(cid)
        events = cast(list[reference.Obj], item["events"])
        declaration_value = reference.obj(item["declaration"])
        bindings = cast(list[reference.Obj], declaration_value["binding_inputs"])
        assert bindings
        assert all(
            binding["materialization"] == "single" and binding["version_selection"] == "latest" for binding in bindings
        )
        assert declaration_value["initial_collections"] == []
        versioned = [event for event in events if event.get("kind") == "artifact" and event.get("key") in {"EA", "XA"}]
        keys = Counter(cast(str, event["key"]) for event in versioned)
        repeated = {key for key, count in keys.items() if count == 2}
        assert repeated
        for key in repeated:
            refs = {event["ref"] for event in versioned if event["key"] == key}
            owners = {
                event["artifact"]
                for event in events
                if event.get("kind") == "provenance" and event.get("source") == "bound_input"
            }
            assert refs <= owners
        selected_ports = [
            event
            for event in events
            if event.get("kind") == "port" and event.get("artifact") in {value["ref"] for value in versioned}
        ]
        assert selected_ports
        assert all(
            next(value for value in versioned if value["ref"] == selected_port["artifact"])["version"]
            == max(
                value["version"]
                for value in versioned
                if value["key"]
                == next(candidate["key"] for candidate in versioned if candidate["ref"] == selected_port["artifact"])
            )
            for selected_port in selected_ports
        )

    map_case = case("lineage/map_two_versions")
    map_events = cast(list[reference.Obj], map_case["events"])
    map_items = [
        event for event in map_events if event.get("kind") == "provenance" and event.get("source") == "map_item"
    ]
    assert {(event["item_key"], event["item_version"]) for event in map_items} == {(0, 1), (0, 2)}
    assert {event["artifact"] for event in map_items} == {"MI0v1", "MI0v2"}

    expected_codes = {
        "lineage/two_selected_current_versions": "duplicate",
        "lineage/invented_version": "missing",
        "lineage/declaration_crossover": "contradictory",
        "lineage/target_crossover": "foreign_owner",
        "lineage/invocation_crossover": "foreign_owner",
        "lineage/expander_crossover": "missing",
        "lineage/map_target_crossover": "foreign_owner",
        "lineage/map_invocation_crossover": "foreign_owner",
        "lineage/artifacts_one_over": "limit_exceeded",
        "lineage/bytes_one_over": "limit_exceeded",
        "lineage/provenance_one_over": "limit_exceeded",
        "lineage/source_duplicate_pair": "duplicate",
    }
    for cid, code in expected_codes.items():
        assert result(cid) == {"code": code, "status": "rejected"}
    for cid in ("lineage/artifacts_exact", "lineage/bytes_exact", "lineage/provenance_exact"):
        assert result(cid)["status"] == "accepted"
    assert result("lineage/rejected_publication_rollback") == result("lineage/initial_two_versions")


def test_v12_latest_selection_and_owner_chain() -> None:
    assert result("lineage/latest_older_selected") == {"code": "contradictory", "status": "rejected"}
    assert result("validity/final_candidate_sibling_substitution") == {
        "code": "contradictory",
        "status": "rejected",
    }
    expected = {
        "lineage/latest_missing_binding_request": "missing",
        "lineage/latest_foreign_binding_association": "foreign_owner",
        "lineage/latest_missing_binding_settlement": "missing",
        "lineage/latest_missing_binding_cleanup": "missing",
        "lineage/latest_foreign_cleanup_target": "foreign_owner",
    }
    for identity, code in expected.items():
        assert result(identity) == {"code": code, "status": "rejected"}

    positive = case("lineage/initial_two_versions")
    events = cast(list[reference.Obj], positive["events"])
    assert [value["kind"] for value in events if cast(str, value["kind"]).startswith("binding_")][:7] == [
        "binding_reserve",
        "binding_dispatch",
        "binding_result",
        "binding_settlement",
        "binding_receipt",
        "binding_cleanup_association",
        "binding_cleanup",
    ]
    assert not any(value["kind"] == "publication_rejected" for value in events)
