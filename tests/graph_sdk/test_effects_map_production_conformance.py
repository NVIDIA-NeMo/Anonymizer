# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate frozen map effects cases through the real graph executor."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import cast

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected
from anonymizer.engine.graph_sdk.executor import (
    ExecutionLimits,
    MapExpansionDecl,
)
from anonymizer.graph._values import ContractViolation, ValidationCode
from anonymizer.graph.workflow import (
    ArtifactType,
    MapDecl,
    NodeId,
    SubgraphNode,
    WorkflowId,
    admit_activation_workflow,
)
from tests.graph_sdk.effects_map_assertions import (
    _assert_bound_context_map_item_conflict,
    _assert_cancelled_map_result,
    _assert_control_only_map,
    _assert_cross_returned_map_association,
    _assert_dynamic_map_items_override_default,
    _assert_empty_max_one_map_scalar_resolution,
    _assert_malformed_map_result,
    _assert_map_membership,
    _assert_map_overflow,
    _assert_map_result_publication,
    _assert_map_scalar_source_resolution,
    _assert_map_storage_limit,
    _assert_overflow_collection_storage_exact,
    _assert_terminal_map_without_scalar,
    _assert_versioned_map_lineages,
    _assert_versioned_map_rollback_reuse,
)
from tests.graph_sdk.effects_map_execution import _admit_fixture as _admit_fixture
from tests.graph_sdk.effects_map_execution import (
    _assert_loop_scalar_source_resolution,
    _assert_map_collection_item_schema_conflict,
    _assert_map_collection_schema_minimum_conflict,
    _assert_map_subgraph_item_binding,
)
from tests.graph_sdk.effects_map_execution import _execute_membership as _execute_membership
from tests.graph_sdk.effects_map_fixtures import MAP_CASES
from tests.graph_sdk.effects_map_fixtures import _map_fixture as _map_fixture
from tests.graph_sdk.effects_map_fixtures import _MapFixture as _MapFixture
from tests.graph_sdk.test_activation import _loop_workflow


@pytest.mark.parametrize(
    ("case_id", "mode", "expected_resolution"),
    (
        ("map/loop_exit", "exit", "member:1:exit"),
        ("map/loop_bypass", "bypass", "blocked:bypass"),
        ("map/loop_prior_continue", "prior_continue", "blocked:no_exit"),
        ("map/loop_failure", "failure", "blocked:no_exit"),
        ("map/loop_overflow", "overflow", "blocked:no_exit"),
    ),
)
def test_loop_scalar_source_resolution_matches_frozen_case(
    case_id: str,
    mode: str,
    expected_resolution: str,
) -> None:
    asyncio.run(_assert_loop_scalar_source_resolution(case_id, mode, expected_resolution))


def test_map_operation_declaration_admits_through_production() -> None:
    case = MAP_CASES["map/admit_operation"]
    fixture = _map_fixture()
    assert case["expected"] == {"status": "accepted"}
    assert fixture.workflow.scopes[0].maps[0].member == fixture.member


def test_map_subgraph_declaration_admits_through_production() -> None:
    case = MAP_CASES["map/admit_subgraph"]
    fixture = _map_fixture(member_subgraph=True)
    member = next(node for node in fixture.workflow.workflow.nodes if node.id == fixture.member)
    assert isinstance(member, SubgraphNode)
    assert case["expected"] == {"status": "accepted"}


def test_map_items_are_projected_into_real_subgraph_members() -> None:
    asyncio.run(_assert_map_subgraph_item_binding())


def test_control_only_map_declaration_admits_through_production() -> None:
    case = MAP_CASES["map/admit_control_only"]
    fixture = _map_fixture(control_only=True)
    assert case["expected"] == {"status": "accepted"}
    assert fixture.workflow.scopes[0].maps[0].item_input is None
    _admit_fixture(fixture)


def test_map_dynamic_item_dependency_overrides_static_default_summary() -> None:
    case = MAP_CASES["map/default_override_dependency"]
    fixture = _map_fixture(default_override=True)
    _admit_fixture(fixture)
    assert case["expected"] == {"status": "accepted"}


def test_boolean_collection_limit_rejects_at_typed_constructor() -> None:
    case = MAP_CASES["map/collection_items_invalid_limit"]
    with pytest.raises(EffectRejected) as error:
        ExecutionLimits(
            max_local_in_flight=1,
            max_remote_outstanding=0,
            max_runtime_artifacts=1,
            max_runtime_artifact_bytes=1,
            max_collection_items=True,
        )
    assert error.value.code == EffectCode.INVALID_TYPE
    assert error.value.code.value == case["expected"]["code"]


def test_map_collection_schema_minimum_conflict_rejects_execution_admission() -> None:
    asyncio.run(_assert_map_collection_schema_minimum_conflict())


def test_map_collection_item_schema_conflict_rejects_execution_admission() -> None:
    asyncio.run(_assert_map_collection_item_schema_conflict())


def test_false_map_item_identity_summary_rejects_execution_admission() -> None:
    case = MAP_CASES["map/false_identity_summary"]
    false_identity = _map_fixture(max_children=1, outward_scalar="workflow_output")
    with pytest.raises(EffectRejected) as error:
        _admit_fixture(false_identity)
    assert error.value.code == EffectCode.CONTRADICTORY
    assert error.value.code.value == case["expected"]["code"]

    nonidentity = _map_fixture(
        max_children=1,
        outward_scalar="workflow_output",
        outward_identity=False,
    )
    _admit_fixture(nonidentity)


def test_map_item_dependency_summary_mismatch_rejects_execution_admission() -> None:
    case = MAP_CASES["map/dependency_summary_mismatch"]
    fixture = _map_fixture(
        max_children=1,
        default_override=True,
        outward_scalar="workflow_output",
        outward_identity=False,
        outward_value_depends_on_default=False,
    )
    with pytest.raises(EffectRejected) as error:
        _admit_fixture(fixture)
    assert error.value.code == EffectCode.CONTRADICTORY
    assert error.value.code.value == case["expected"]["code"]


def test_bound_context_cannot_override_a_dynamic_map_item() -> None:
    asyncio.run(_assert_bound_context_map_item_conflict())


@pytest.mark.parametrize(
    ("case_id", "max_children", "destination", "accepted"),
    (
        ("map/max_zero_scalar_join", 0, "join", True),
        ("map/max_one_scalar_ordinary", 1, "ordinary", True),
        ("map/max_one_scalar_workflow", 1, "workflow_output", True),
        ("map/max_two_scalar_join", 2, "join", False),
        ("map/max_two_scalar_ordinary", 2, "ordinary", False),
        ("map/max_two_scalar_workflow_output", 2, "workflow_output", False),
    ),
)
def test_scalar_map_cardinality_admission_matches_frozen_case(
    case_id: str,
    max_children: int,
    destination: str,
    accepted: bool,
) -> None:
    case = MAP_CASES[case_id]
    if accepted:
        _map_fixture(max_children=max_children, outward_scalar=destination)
        assert case["expected"] == {"status": "accepted"}
        return

    with pytest.raises(ContractViolation) as error:
        _map_fixture(max_children=max_children, outward_scalar=destination)
    assert error.value.code is ValidationCode.UNSUPPORTED
    assert error.value.code.value == case["expected"]["code"]


@pytest.mark.parametrize(
    ("case_id", "declarations", "code"),
    (
        ("map/missing_expansion", (), EffectCode.MISSING),
        ("map/item_type_mismatch", "wrong_item", EffectCode.CONTRADICTORY),
    ),
)
def test_map_expansion_admission_rejections(
    case_id: str,
    declarations: tuple[MapExpansionDecl, ...] | str,
    code: EffectCode,
) -> None:
    case = MAP_CASES[case_id]
    fixture = _map_fixture()
    if declarations == "wrong_item":
        declarations = (
            MapExpansionDecl(
                expander=fixture.expander,
                outcome="expand",
                membership_port="members",
                item_type=ArtifactType(name="bytes", revision=1),
            ),
        )
    with pytest.raises(EffectRejected) as error:
        _admit_fixture(fixture, map_expansions=cast(tuple[MapExpansionDecl, ...], declarations))
    assert error.value.code == code
    assert case["expected"] == {"status": "rejected", "code": code.value}


@pytest.mark.parametrize(
    ("case_id", "foreign", "code"),
    (
        ("map/duplicate_map_source", False, ValidationCode.DUPLICATE),
        ("map/foreign_before_duplicate", True, ValidationCode.FOREIGN_OWNER),
    ),
)
def test_map_source_admission_precedence_matches_frozen_case(
    case_id: str,
    foreign: bool,
    code: ValidationCode,
) -> None:
    case = MAP_CASES[case_id]
    fixture = _map_fixture()
    scope = fixture.workflow.scopes[0]
    declaration = scope.maps[0]
    duplicate = replace(
        declaration,
        expander=NodeId.new(workflow=WorkflowId.new()) if foreign else declaration.expander,
    )

    with pytest.raises(ContractViolation) as error:
        admit_activation_workflow(
            workflow=fixture.workflow.workflow,
            scopes=(replace(scope, maps=(declaration, duplicate)),),
            limits=replace(fixture.workflow.limits, max_maps=2),
        )

    assert error.value.code is code
    assert case["expected"]["status"] == "rejected"
    assert case["expected"]["code"] == code.value


@pytest.mark.parametrize("case_id", ("map/map_loop_duplicate_source", "map/duplicate_loop_source"))
def test_duplicate_dynamic_source_roles_reject_before_shape_checks(case_id: str) -> None:
    case = MAP_CASES[case_id]
    workflow, starter, member, _ = _loop_workflow(2)
    scope = workflow.scopes[0]
    maps = scope.maps
    loops = scope.loops
    if case_id == "map/map_loop_duplicate_source":
        maps = (
            MapDecl(
                expander=starter,
                member=member,
                expansion_outcomes=frozenset({"again"}),
                max_children=1,
            ),
        )
    else:
        loops = (scope.loops[0], scope.loops[0])

    with pytest.raises(ContractViolation) as error:
        admit_activation_workflow(
            workflow=workflow.workflow,
            scopes=(replace(scope, maps=maps, loops=loops),),
            limits=replace(
                workflow.limits,
                max_maps=len(maps),
                max_loops=len(loops),
                max_children_per_map=1,
                max_activation_occurrences=8,
            ),
        )
    assert error.value.code == ValidationCode.DUPLICATE
    assert error.value.code.value == case["expected"]["code"]


def test_map_provenance_capacity_rejects_before_execution_at_one_over() -> None:
    rejected = MAP_CASES["map/provenance_one_over"]
    accepted = MAP_CASES["map/bounds_exact"]
    fixture = _map_fixture()

    with pytest.raises(EffectRejected) as error:
        _admit_fixture(fixture, baseline_provenance_edges=1, provenance_edge_headroom=1)

    assert error.value.code == EffectCode.LIMIT_EXCEEDED
    assert rejected["boundary"] == "map_execution_preflight"
    assert rejected["events"] == []
    assert rejected["expected"] == {"status": "rejected", "code": error.value.code.value}

    plan = _admit_fixture(fixture, baseline_provenance_edges=1, provenance_edge_headroom=2)
    assert plan.assessment_limits.max_provenance_edges == 3
    assert accepted["expected"]["status"] == "accepted"


@pytest.mark.parametrize("item_count", (0, 1, 2))
def test_map_harness_publishes_real_occurrences(item_count: int) -> None:
    asyncio.run(_assert_map_membership(item_count))


@pytest.mark.parametrize(
    ("case_id", "response_mode", "other_count"),
    (
        ("map/missing_membership_port", "missing", 1),
        ("map/wrong_membership_type", "wrong_type", 0),
        ("map/duplicate_membership_port", "duplicate", 0),
    ),
)
def test_malformed_map_results_publish_no_partial_facts(case_id: str, response_mode: str, other_count: int) -> None:
    asyncio.run(_assert_malformed_map_result(case_id, response_mode, other_count))


def test_cross_returned_map_association_fails_before_publication() -> None:
    asyncio.run(_assert_cross_returned_map_association())


@pytest.mark.parametrize("destination", ("join", "ordinary", "workflow_output"))
@pytest.mark.parametrize("maximum", (0, 1))
def test_map_scalar_source_resolution_uses_the_unique_dynamic_member(destination: str, maximum: int) -> None:
    asyncio.run(_assert_map_scalar_source_resolution(destination, maximum))


@pytest.mark.parametrize("destination", ("join", "ordinary", "workflow_output"))
def test_map_scalar_source_resolution_rejects_ambiguous_fanout(destination: str) -> None:
    case = MAP_CASES[f"map/resolve_{destination}_max_2"]
    with pytest.raises(ContractViolation) as error:
        _map_fixture(max_children=2, outward_scalar=destination)
    assert error.value.code == ValidationCode.UNSUPPORTED
    assert error.value.code.value == case["expected"]["code"]


@pytest.mark.parametrize("destination", ("join", "ordinary", "workflow_output"))
def test_map_scalar_source_resolution_blocks_an_empty_max_one_expansion(destination: str) -> None:
    asyncio.run(_assert_empty_max_one_map_scalar_resolution(destination))


@pytest.mark.parametrize("destination", ("join", "ordinary", "workflow_output"))
@pytest.mark.parametrize(("item_count", "response_mode"), ((2, "valid"), (1, "missing")))
def test_map_scalar_source_blocks_terminal_expansion_without_unique_member(
    destination: str,
    item_count: int,
    response_mode: str,
) -> None:
    asyncio.run(_assert_terminal_map_without_scalar(destination, item_count, response_mode))


def test_map_overflow_publishes_collection_without_item_facts() -> None:
    asyncio.run(_assert_map_overflow())


def test_overflow_collection_storage_exact_matches_frozen_publication() -> None:
    asyncio.run(_assert_overflow_collection_storage_exact())


def test_control_only_map_activates_members_without_item_facts() -> None:
    asyncio.run(_assert_control_only_map())


def test_dynamic_map_items_override_an_unavailable_static_default() -> None:
    asyncio.run(_assert_dynamic_map_items_override_default())


@pytest.mark.parametrize(
    ("case_id", "values", "artifact_headroom", "artifact_byte_headroom", "other_count"),
    (
        ("map/membership_0", (), 8, 32, 0),
        ("map/membership_1", ("a",), 8, 32, 0),
        ("map/membership_2", ("a", "b"), 8, 32, 0),
        ("map/membership_with_0_other_outputs", ("a", "b"), 8, 32, 0),
        ("map/membership_with_1_other_outputs", ("a", "b"), 8, 32, 1),
        ("map/membership_with_2_other_outputs", ("a", "b"), 8, 32, 2),
        ("map/collection_items_exact", ("a", "b"), 8, 32, 0),
        ("map/bounds_exact", ("aa", "bb"), 3, 8, 0),
    ),
)
def test_map_result_publication_matches_frozen_case(
    case_id: str,
    values: tuple[str, ...],
    artifact_headroom: int,
    artifact_byte_headroom: int,
    other_count: int,
) -> None:
    asyncio.run(
        _assert_map_result_publication(
            case_id,
            values,
            artifact_headroom,
            artifact_byte_headroom,
            other_count,
        )
    )


@pytest.mark.parametrize(
    ("case_id", "artifact_headroom", "max_collection_items"),
    (
        ("map/artifact_count_one_over", 2, 4),
        ("map/collection_items_one_over", 8, 1),
        ("map/overflow_collection_storage_one_over", 8, 2),
        ("map/artifact_bytes_one_over", 8, 4),
    ),
)
@pytest.mark.parametrize("same_key_versions", [False, True])
def test_map_storage_limits_rollback_publication(
    case_id: str,
    artifact_headroom: int,
    max_collection_items: int,
    same_key_versions: bool,
) -> None:
    asyncio.run(_assert_map_storage_limit(case_id, artifact_headroom, max_collection_items, same_key_versions))


@pytest.mark.parametrize(
    "case_id",
    ("map/prospective_transition_rejected", "map/caller_transition_verdict_rejected"),
)
def test_map_result_after_parent_cancellation_is_not_published(case_id: str) -> None:
    asyncio.run(_assert_cancelled_map_result(case_id))


@pytest.mark.parametrize(
    ("count", "mode", "status"), [(0, "valid", "closed"), (2, "valid", "overflow"), (1, "missing", "failed")]
)
def test_canonical_membership_retains_expansions_without_instantiated_children(
    count: int, mode: str, status: str
) -> None:
    _, result, _ = asyncio.run(_execute_membership(count, response_mode=mode, max_children=1, outward_scalar="join"))
    expansion = next(iter(result.states[0].expansions))
    assert expansion.status == status
    assert not expansion.members
    memberships = [item for item in result.record.memberships if item.parent == expansion.parent]
    assert len(memberships) == 1
    assert memberships[0].closed
    assert not memberships[0].members
    assert not any(entry.activation.parent == expansion.parent for entry in result.states[0].entries)
    assert {item.activation for item in result.record.terminals} == {
        member for membership in result.record.memberships for member in membership.members
    }


def test_versioned_map_lineages_separate_targets_and_invocations_at_exact_capacity() -> None:
    asyncio.run(_assert_versioned_map_lineages())


def test_versioned_map_rollback_restores_allocator_for_next_target() -> None:
    asyncio.run(_assert_versioned_map_rollback_reuse())


def test_versioned_map_provenance_one_over_rejects_at_execution_admission() -> None:
    with pytest.raises(EffectRejected) as rejected:
        asyncio.run(_execute_membership(2, same_key_versions=True, target_count=2, provenance_edge_headroom=1))
    assert rejected.value.code.value == "limit_exceeded"
