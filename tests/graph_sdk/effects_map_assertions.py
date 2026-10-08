# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Map publication, source resolution, and rollback assertions."""

from __future__ import annotations

from typing import Any, cast

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected
from anonymizer.engine.graph_sdk.binding import start_initial_binding
from anonymizer.engine.graph_sdk.context import (
    BindingLimits,
    ContextMaterialization,
    ContextResource,
    ContextSelector,
    ContextSourceCapability,
    InitialContextDecl,
    RetrievalBounds,
)
from anonymizer.engine.graph_sdk.executor import (
    BoundInputKey,
    InitialCollectionKey,
    MapItemKey,
    OperationOutputKey,
    RootInputKey,
)
from anonymizer.engine.graph_sdk.requests import (
    PhysicalRequestPolicy,
    SemanticAssociation,
    TextArtifactValue,
    TextCollectionValue,
)
from anonymizer.engine.graph_sdk.resources import ResourceLease
from anonymizer.graph.workflow import (
    NodeId,
)
from tests.graph_sdk.context_source_fixtures import SOURCE, _ContextProvider
from tests.graph_sdk.effects_map_execution import _admit_fixture, _execute_membership
from tests.graph_sdk.effects_map_fixtures import MAP_CASES, _map_fixture, _MapCallback, _MapFixture
from tests.graph_sdk.test_preparation import _data


async def _assert_bound_context_map_item_conflict() -> None:
    case = MAP_CASES["map/context_override_conflict"]
    fixture = _map_fixture(context_item_override=True)
    data = _data(1)
    target = next(iter(data.targets))
    policy = PhysicalRequestPolicy(
        visibility="dispatch_and_settlement",
        pre_dispatch_control="executor",
        retry_owner="executor",
        replay="idempotent",
        max_attempts=1,
    )
    capability = ContextSourceCapability(
        source=SOURCE,
        artifact_type=fixture.text_type,
        uses=frozenset({"initial_binding"}),
        execution="async",
        resource_owner="caller",
        cancellation="cooperative_ack",
        settlement="explicit_ack",
        usage="exact",
        request=policy,
        safe_detachment="forbidden",
    )
    provider = _ContextProvider(items=("context",))
    binding = await (
        await start_initial_binding(
            data=data,
            workflow=fixture.workflow,
            declarations=(
                InitialContextDecl(
                    target=target,
                    node=fixture.member,
                    port="item",
                    artifact_type=fixture.text_type,
                    source=SOURCE,
                    selector=ContextSelector(fields=()),
                    requirement="required",
                    bounds=RetrievalBounds(max_items=1, max_bytes=7, max_requests=1),
                    materialization=ContextMaterialization(kind="single", item_type=fixture.text_type),
                ),
            ),
            capabilities=(capability,),
            resources=(
                ContextResource(
                    source=SOURCE,
                    capability=capability,
                    lease=ResourceLease.create(owner="caller", safe_detachment="forbidden", handle=provider),
                    factory=None,
                ),
            ),
            limits=BindingLimits(
                max_declarations=1,
                max_sources=1,
                max_capabilities=1,
                max_selector_fields=0,
                max_selector_bytes=0,
                max_items=1,
                max_bytes=7,
                max_requests=1,
                max_resources=1,
            ),
        )
    ).wait()
    assert binding.context is not None
    with pytest.raises(EffectRejected) as error:
        _admit_fixture(fixture, data=data, bound_context=binding.context)
    assert error.value.code == EffectCode.CONTRADICTORY
    assert error.value.code.value == case["expected"]["code"]


async def _assert_map_membership(item_count: int) -> None:
    fixture, result, callbacks = await _execute_membership(item_count)
    state = result.states[0]
    expansion = next(iter(state.expansions))
    assert expansion.status == "closed"
    assert len(expansion.members) == item_count
    map_items = [fact for fact in result.provenance if isinstance(fact.key, MapItemKey)]
    assert len(map_items) == item_count
    assert {fact.key.item_key for fact in map_items} == set(range(item_count))
    parent = next(
        fact.key
        for fact in result.provenance
        if isinstance(fact.key, OperationOutputKey)
        and fact.key.activation == expansion.parent
        and fact.key.port == "members"
    )
    assert all(fact.parents == frozenset({parent}) for fact in map_items)
    membership = next(fact for fact in result.provenance if fact.key == parent)
    assert len(membership.parents) == 1
    assert isinstance(next(iter(membership.parents)), RootInputKey)
    member_inputs = [call[0] for call in callbacks[fixture.member].calls]
    assert [item.inputs[0].value for item in member_inputs] == [
        TextArtifactValue(text=f"item-{index}") for index in range(item_count)
    ]
    item_artifacts = {fact.key.member: fact.artifact for fact in map_items}
    assert all(
        isinstance(item.association, SemanticAssociation)
        and item.inputs[0].artifact == item_artifacts[item.association.task.activation]
        for item in member_inputs
    )
    assert {entry.template for entry in state.entries if entry.activation in expansion.members} == (
        {fixture.member} if item_count else set()
    )


async def _assert_malformed_map_result(case_id: str, response_mode: str, other_count: int) -> None:
    case = MAP_CASES[case_id]
    fixture, result, callbacks = await _execute_membership(1, response_mode=response_mode, other_count=other_count)
    expander = next(entry for entry in result.states[0].entries if entry.template == fixture.expander)
    assert expander.status == "failure"
    assert expander.outcome is None
    assert case["expected"]["state"]["terminal"] == "malformed_response"
    _assert_only_setup_baseline(fixture, result, callbacks)


async def _assert_cross_returned_map_association() -> None:
    case = MAP_CASES["map/wrong_parent"]
    fixture, result, callbacks = await _execute_membership(
        1,
        response_mode="cross_association",
        target_count=2,
    )
    callback = callbacks[fixture.expander]
    assert len(callback.calls) == len(callback.cross_associations) == 2
    assert callback.calls[0][0].association != callback.calls[1][0].association
    assert case["boundary"] == "local_callback"
    for state in result.states:
        expander = next(entry for entry in state.entries if entry.template == fixture.expander)
        assert expander.status == "failure"
        assert expander.outcome is None
        expansion = next(iter(state.expansions))
        assert expansion.parent == expander.activation
        assert expansion.status == "failed"
        assert not expansion.members
    assert not result.assessments
    assert not any(isinstance(fact.key, (OperationOutputKey, MapItemKey)) for fact in result.provenance)
    assert not callbacks[fixture.member].calls
    assert case["expected"]["state"]["terminal"] == "malformed_response"


async def _assert_map_scalar_source_resolution(destination: str, maximum: int) -> None:
    case = MAP_CASES[f"map/resolve_{destination}_max_{maximum}"]
    fixture, result, callbacks = await _execute_membership(
        maximum,
        item_values=("a",) if maximum else (),
        max_children=maximum,
        outward_scalar=destination,
        outward_identity=destination != "workflow_output",
    )
    expected_resolution = case["expected"]["state"]["resolution"]
    if destination == "workflow_output":
        values = [fact for fact in result.final_outputs if fact.port == "value"]
        assert len(values) == maximum
        if maximum:
            expansion = next(iter(result.states[0].expansions))
            assert len(expansion.members) == 1
            member_activation = next(iter(expansion.members))
            member = next(
                entry
                for entry in result.states[0].entries
                if entry.template == fixture.member and entry.activation == member_activation
            )
            output = next(
                fact
                for fact in result.provenance
                if isinstance(fact.key, OperationOutputKey)
                and fact.key.activation == member.activation
                and fact.key.port == "value"
            )
            assert values[0].producer == output.key
            assert values[0].candidate.artifact == output.artifact
            assert values[0].outcome == "ok"
        assert expected_resolution == ("member:0" if maximum else "blocked:workflow_output")
        return

    destination_node = fixture.join
    if destination == "ordinary":
        destination_node = next(
            node
            for node in fixture.implementation_nodes
            if node not in {fixture.expander, fixture.member_implementation, fixture.join}
        )
    calls = callbacks[destination_node].calls
    assert len(calls) == maximum
    if maximum:
        assert calls[0][0].inputs[0].value == TextArtifactValue(text="a")
        assert expected_resolution == "member:0"
    else:
        destination_entry = next(entry for entry in result.states[0].entries if entry.template == destination_node)
        assert destination_entry.status == "blocked"
        assert expected_resolution == f"blocked:{destination}"


async def _assert_empty_max_one_map_scalar_resolution(destination: str) -> None:
    case = MAP_CASES[f"map/resolve_{destination}_max_1_empty"]
    fixture, result, callbacks = await _execute_membership(
        0,
        item_values=(),
        max_children=1,
        outward_scalar=destination,
        outward_identity=destination != "workflow_output",
    )
    if destination == "workflow_output":
        assert not any(fact.port == "value" for fact in result.final_outputs)
    else:
        destination_node = fixture.join
        if destination == "ordinary":
            destination_node = next(
                node
                for node in fixture.implementation_nodes
                if node not in {fixture.expander, fixture.member_implementation, fixture.join}
            )
        assert not callbacks[destination_node].calls
        entry = next(item for item in result.states[0].entries if item.template == destination_node)
        assert entry.status == "blocked"
    assert case["expected"]["state"]["resolution"] == f"blocked:{destination}"


async def _assert_terminal_map_without_scalar(destination: str, item_count: int, response_mode: str) -> None:
    fixture, result, callbacks = await _execute_membership(
        item_count,
        response_mode=response_mode,
        max_children=1,
        outward_scalar=destination,
        outward_identity=destination != "workflow_output",
    )
    expansion = next(iter(result.states[0].expansions))
    assert expansion.status == ("overflow" if response_mode == "valid" else "failed")
    assert not expansion.members
    if destination == "workflow_output":
        assert not any(fact.port == "value" for fact in result.final_outputs)
        return
    destination_node = fixture.join
    if destination == "ordinary":
        destination_node = next(
            node
            for node in fixture.implementation_nodes
            if node not in {fixture.expander, fixture.member_implementation, fixture.join}
        )
    assert not callbacks[destination_node].calls
    entry = next(item for item in result.states[0].entries if item.template == destination_node)
    expected_status = "inconsistent" if destination == "join" and response_mode == "valid" else "blocked"
    assert entry.status == expected_status


async def _assert_map_overflow() -> None:
    case = MAP_CASES["map/membership_one_over"]
    fixture, result, _ = await _execute_membership(3)
    expansion = next(iter(result.states[0].expansions))
    assert expansion.status == "overflow"
    assert not expansion.members
    assert case["expected"]["state"]["terminal"] == "overflow"
    outputs = [fact for fact in result.provenance if isinstance(fact.key, OperationOutputKey)]
    assert len(outputs) == 1
    assert outputs[0].key.port == "members"
    assert not any(isinstance(fact.key, MapItemKey) for fact in result.provenance)
    assert len(result.artifacts) == 2  # one captured root plus the accepted collection
    assert len(result.assessments) == 1


async def _assert_overflow_collection_storage_exact() -> None:
    case = MAP_CASES["map/overflow_collection_storage_exact"]
    fixture, result, callbacks = await _execute_membership(
        3,
        item_values=("a", "b", "c"),
        max_collection_items=3,
    )
    expected = case["expected"]["state"]["publication"]
    expansion = next(iter(result.states[0].expansions))
    assert expansion.status == "overflow"
    assert not expansion.members
    root = next(fact for fact in result.provenance if isinstance(fact.key, RootInputKey))
    staged = [(artifact, value) for artifact, value in result.artifacts if artifact != root.artifact]
    assert len(staged) == len(expected["artifacts"]) == 1
    collection = cast(TextCollectionValue, staged[0][1])
    assert [item.value.text for item in collection.items] == ["a", "b", "c"]
    assert len(result.assessments) == len(expected["assessments"]) == 1
    output = next(fact for fact in result.provenance if isinstance(fact.key, OperationOutputKey))
    assert output.parents == frozenset({root.key})
    assert not any(isinstance(fact.key, MapItemKey) for fact in result.provenance)
    expander = next(entry for entry in result.states[0].entries if entry.template == fixture.expander)
    assert expander.status == "success"


async def _assert_dynamic_map_items_override_default() -> None:
    await _assert_map_result_publication(
        "map/default_override_no_fallback",
        ("actual",),
        artifact_headroom=8,
        artifact_byte_headroom=32,
        other_count=0,
        default_override=True,
    )


async def _assert_control_only_map() -> None:
    case = MAP_CASES["map/control_only_members"]
    fixture, result, callbacks = await _execute_membership(2, control_only=True)
    expansion = next(iter(result.states[0].expansions))
    assert expansion.status == "closed"
    assert len(expansion.members) == 2
    assert len(callbacks[fixture.member].calls) == 2
    assert all(not call[0].inputs for call in callbacks[fixture.member].calls)
    assert not any(isinstance(fact.key, MapItemKey) for fact in result.provenance)
    expected = case["expected"]["state"]["publication"]
    assert len(result.artifacts) - 1 == len(expected["artifacts"]) == 1


async def _assert_map_result_publication(
    case_id: str,
    values: tuple[str, ...],
    artifact_headroom: int,
    artifact_byte_headroom: int,
    other_count: int,
    default_override: bool = False,
) -> None:
    case = MAP_CASES[case_id]
    fixture, result, callbacks = await _execute_membership(
        len(values),
        item_values=values,
        artifact_headroom=artifact_headroom,
        artifact_byte_headroom=artifact_byte_headroom,
        other_count=other_count,
        default_override=default_override,
        provenance_edge_headroom=cast(dict[str, int], case["declaration"]["limits"])["max_provenance_edges"],
    )
    expected = case["expected"]["state"]["publication"]
    root = next(fact for fact in result.provenance if isinstance(fact.key, RootInputKey))
    baseline_provenance = [
        fact for fact in result.provenance if isinstance(fact.key, (RootInputKey, BoundInputKey, InitialCollectionKey))
    ]
    baseline_artifacts = {fact.artifact for fact in baseline_provenance}
    staged_provenance = [fact for fact in result.provenance if fact not in baseline_provenance]
    staged_artifacts = [(artifact, value) for artifact, value in result.artifacts if artifact not in baseline_artifacts]
    collection = next(value for _, value in staged_artifacts if isinstance(value, TextCollectionValue))
    items = [value for _, value in staged_artifacts if isinstance(value, TextArtifactValue)]
    assert [item.value.text for item in collection.items] == list(values)
    assert sorted(item.text for item in items) == sorted(values)
    assert len(staged_artifacts) == len(expected["artifacts"])
    assert len(result.assessments) == len(expected["assessments"]) == 1
    assert result.assessments[0].finding.code == "assessment0"
    staged_ports = [fact for fact in result.ports if fact.port not in {"default", "schema"}]
    assert len(staged_ports) == len(expected["ports"])
    assert len(staged_provenance) == len(expected["provenance"])
    outputs = {fact.key.port: fact for fact in staged_provenance if isinstance(fact.key, OperationOutputKey)}
    assert set(outputs) == {"members", *(f"other{index}" for index in range(other_count))}
    output = outputs["members"]
    assert output.parents == frozenset({root.key})
    values_by_artifact = dict(staged_artifacts)
    for index in range(other_count):
        other = outputs[f"other{index}"]
        assert other.parents == frozenset({root.key})
        assert values_by_artifact[other.artifact] == TextCollectionValue(items=())
    map_items = [fact for fact in staged_provenance if isinstance(fact.key, MapItemKey)]
    assert all(fact.parents == frozenset({output.key}) for fact in map_items)
    item_artifacts = {fact.key.member: fact.artifact for fact in map_items}
    member_inputs = [call[0] for call in callbacks[fixture.member].calls]
    assert [item.inputs[0].value for item in member_inputs] == [TextArtifactValue(text=value) for value in values]
    assert all(
        isinstance(item.association, SemanticAssociation)
        and item.inputs[0].artifact == item_artifacts[item.association.task.activation]
        for item in member_inputs
    )
    if default_override:
        assert fixture.default_source is not None
        assert len(callbacks[fixture.default_source].calls) == 1
    expander_ports = {fact.port: fact for fact in staged_ports if fact.node == fixture.expander}
    assert set(expander_ports) == set(outputs)
    assert all(expander_ports[port].artifact == fact.artifact for port, fact in outputs.items())
    item_ports = [fact for fact in staged_ports if fact.node == fixture.member]
    assert {fact.activation: fact.artifact for fact in item_ports} == item_artifacts
    assert all(fact.port == "item" and fact.artifact_type == fixture.text_type for fact in item_ports)
    expansion = next(iter(result.states[0].expansions))
    assert expansion.status == "closed"
    assert len(expansion.members) == len(values)
    expander = next(entry for entry in result.states[0].entries if entry.template == fixture.expander)
    assert expander.status == "success"


async def _assert_map_storage_limit(
    case_id: str, artifact_headroom: int, max_collection_items: int, same_key_versions: bool
) -> None:
    case = MAP_CASES[case_id]
    item_count = 3 if case_id == "map/overflow_collection_storage_one_over" else 2
    item_values = ("aa", "bb") if case_id == "map/artifact_bytes_one_over" else None
    fixture, result, callbacks = await _execute_membership(
        item_count,
        item_values=item_values,
        same_key_versions=same_key_versions,
        artifact_headroom=artifact_headroom,
        artifact_byte_headroom=7 if case_id == "map/artifact_bytes_one_over" else 32,
        max_collection_items=max_collection_items,
    )
    expander = next(entry for entry in result.states[0].entries if entry.template == fixture.expander)
    assert expander.status == "blocked"
    assert expander.outcome is None
    assert case["expected"]["state"]["terminal"] == "artifact_limit"
    _assert_only_setup_baseline(fixture, result, callbacks)


async def _assert_cancelled_map_result(case_id: str) -> None:
    case = MAP_CASES[case_id]
    fixture, result, callbacks = await _execute_membership(1, cancel_before_result=True)
    expander = next(entry for entry in result.states[0].entries if entry.template == fixture.expander)
    assert expander.status == "cancelled"
    assert case["expected"]["state"]["terminal"] == "transition_rejected"
    _assert_only_setup_baseline(fixture, result, callbacks)


def _assert_only_setup_baseline(
    fixture: _MapFixture,
    result: Any,
    callbacks: dict[NodeId, _MapCallback],
) -> None:
    assert not result.assessments
    assert all(isinstance(fact.key, (RootInputKey, BoundInputKey, InitialCollectionKey)) for fact in result.provenance)
    assert not any(isinstance(fact.key, (OperationOutputKey, MapItemKey)) for fact in result.provenance)
    assert {artifact for artifact, _ in result.artifacts} == {fact.artifact for fact in result.provenance}
    assert all(port.node == fixture.expander and port.port in {"default", "schema"} for port in result.ports)
    assert {port.artifact for port in result.ports} <= {fact.artifact for fact in result.provenance}
    requests = result.requests
    assert requests.dispatched_count == 0
    assert not requests.dispatches
    assert not requests.denials
    assert not requests.terminals
    assert not requests.settlements
    assert not requests.defects
    assert not requests.local_in_flight
    assert not requests.remote_outstanding
    assert not requests.cancel_requested
    expansions = result.states[0].expansions
    assert len(expansions) == 1
    expansion = next(iter(expansions))
    assert expansion.status == "failed"
    assert not expansion.members
    parent = next(entry.activation for entry in result.states[0].entries if entry.template == fixture.expander)
    assert expansion.parent == parent
    assert not callbacks[fixture.member].calls


async def _assert_versioned_map_lineages() -> None:
    invocations = []
    for _ in range(2):
        fixture, result, callbacks = await _execute_membership(
            2,
            same_key_versions=True,
            target_count=2,
            artifact_headroom=6,
            artifact_byte_headroom=48,
            provenance_edge_headroom=2,
        )
        items = [fact for fact in result.provenance if isinstance(fact.key, MapItemKey)]
        assert len(items) == 4
        owners = {}
        for fact in items:
            assert isinstance(fact.key, MapItemKey)
            owners.setdefault((fact.key.target, fact.key.expander), []).append(fact.artifact)
        assert len(owners) == 2
        assert len({refs[0].key for refs in owners.values()}) == 2
        for refs in owners.values():
            assert len({ref.key for ref in refs}) == 1
            assert {ref.version for ref in refs} == {1, 2}
        assert len(result.artifacts) == 8
        assert len(callbacks[fixture.member].calls) == 4
        assert all(state.complete for state in result.states)
        invocations.append(result.record.invocation)
    assert invocations[0] != invocations[1]


async def _assert_versioned_map_rollback_reuse() -> None:
    fixture, result, callbacks = await _execute_membership(
        2,
        same_key_versions=True,
        item_counts=(2, 1),
        target_count=2,
        artifact_headroom=2,
        artifact_byte_headroom=48,
    )
    expanders = [entry for state in result.states for entry in state.entries if entry.template == fixture.expander]
    assert sorted(entry.status for entry in expanders) == ["blocked", "success"]
    assert len(callbacks[fixture.expander].calls) == 2
    assert len(callbacks[fixture.member].calls) == 1
    assert sorted(ref.key for ref, _ in result.artifacts) == [0, 1, 2, 3]
    items = [fact for fact in result.provenance if isinstance(fact.key, MapItemKey)]
    assert len(items) == 1
    assert isinstance(items[0].key, MapItemKey)
    assert items[0].key.expander == next(entry.activation for entry in expanders if entry.status == "success")
