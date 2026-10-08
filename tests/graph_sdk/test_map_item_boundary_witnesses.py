# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual SDK witnesses for neutral traces without literal retained-record shapes."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected
from anonymizer.engine.graph_sdk.evidence import (
    AdmittedQualification,
    AssessmentSubmission,
    EvidenceRevisionView,
    admit_qualification,
    evidence_revision_view,
)
from anonymizer.engine.graph_sdk.executor import ExecutionResult, MapItemKey, OperationOutputKey
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.engine.graph_sdk.requests import TextCollectionValue
from anonymizer.graph.workflow import (
    InputBinding,
    InputPort,
    NodeId,
    NodeInputRef,
    NodeOutputRef,
    OperationNode,
    OperationSpec,
    OutputDependency,
    OutputPort,
    SequenceEdge,
    admit_activation_workflow,
    admit_static_workflow,
)
from tests.graph_sdk.test_dynamic_executor import _capability, _outcome
from tests.graph_sdk.test_effects_map_production_conformance import _execute_membership
from tests.graph_sdk.test_evidence import _qualification_limits
from tests.graph_sdk.test_map_item_evidence import _direct_item_fixture
from tests.graph_sdk.test_qualification_map_conformance import _execute_reference_map


def _qualification_inputs(
    result: ExecutionResult,
) -> tuple[AdmittedQualification, EvidenceRevisionView, tuple[AssessmentSubmission, ...]]:
    execution = result._execution
    productions = tuple(item for item in execution.assessment_productions if item.evidence_port == "evidence")
    admitted = admit_qualification(
        execution=execution, productions=productions, limits=_qualification_limits(max_revision_entries=32)
    )
    current = evidence_revision_view(
        admitted=admitted,
        result=result,
        artifacts=tuple(ref for ref, _ in result.artifacts),
        absences=tuple({ref for fact in result.assessments for ref in fact.environment.absences}),
        configurations=tuple(
            (item.node, item.capability.configuration) for item in execution.context.prepared.implementations
        ),
        state=execution.context.prepared.state,
    )
    owners = {(item.node, item.promise) for item in productions}
    submissions = tuple(
        AssessmentSubmission(fact=fact) for fact in result.assessments if (fact.node, fact.promise) in owners
    )
    return admitted, current, submissions


@pytest.mark.parametrize(("count", "mode", "status"), [(3, "valid", "overflow"), (1, "missing", "failed")])
def test_failed_map_publication_cannot_vacuously_qualify_item_protection(count: int, mode: str, status: str) -> None:
    fixture = _direct_item_fixture()
    _, result, callbacks = asyncio.run(
        _execute_membership(
            count,
            fixture=fixture,
            response_mode=mode,
            member_assessment=True,
            artifact_byte_headroom=64,
            port_fact_headroom=16,
            provenance_edge_headroom=16,
        )
    )
    (expansion,) = result.states[0].expansions
    assert expansion.status == status and not expansion.members
    assert not callbacks[fixture.member].calls
    assert not any(isinstance(fact.key, MapItemKey) for fact in result.provenance)
    collections = [
        dict(result.artifacts)[fact.artifact]
        for fact in result.provenance
        if isinstance(fact.key, OperationOutputKey) and fact.key.activation == expansion.parent
    ]
    if status == "overflow":
        assert len(collections) == 1
        assert isinstance(collections[0], TextCollectionValue)
        assert len(collections[0].items) == count
    else:
        assert not collections
    admitted, current, submissions = _qualification_inputs(result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert not output.qualified and not output.verified and not result.final_outputs
    assert output.targets[0].withholding == frozenset({"terminal_failure", "missing_candidate"})
    assert output.record.statuses[0].completion == "closed"


def test_pending_retained_map_membership_withholds_previously_qualified_candidate() -> None:
    fixture = _direct_item_fixture()
    _, result, _ = asyncio.run(
        _execute_membership(
            0,
            fixture=fixture,
            member_assessment=True,
            artifact_byte_headroom=64,
            port_fact_headroom=16,
            provenance_edge_headroom=16,
        )
    )
    admitted, current, submissions = _qualification_inputs(result)
    before = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(before.qualified) == 1
    (expansion,) = result.states[0].expansions
    assert expansion.status == "closed" and not expansion.members
    # A successful atomic publication cannot leave this pending. This is a
    # retained-state integrity witness, not a fabricated normal execution.
    object.__setattr__(expansion, "status", "pending")
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert not output.qualified
    assert "incomplete_membership" in output.targets[0].withholding
    assert output.record.statuses[0].completion == "pending"
    assert output.record.memberships == result.record.memberships
    assert any(item.parent == expansion.parent and item.closed for item in output.record.memberships)


@pytest.mark.parametrize("cross_map", [False, True])
def test_captured_item_subject_cannot_be_replaced_by_candidate_or_other_map(cross_map: bool) -> None:
    _, result, nodes = asyncio.run(_execute_reference_map(1, two_maps=cross_map))
    admitted, current, submissions = _qualification_inputs(result)
    assert len(qualify(admitted=admitted, result=result, current=current, submissions=submissions).qualified) == 1
    source = next(
        port
        for port in result.ports
        if port.node == nodes["MN" if cross_map else "N"] and port.port == ("item" if cross_map else "subject")
    )
    victim = next(
        port
        for port in result.ports
        if port.node == nodes["MN2" if cross_map else "MN"] and port.port == ("item2" if cross_map else "item")
    )
    assert source.artifact != victim.artifact
    # Assessment facts have no subject_artifact field. The actual owner of
    # subject identity is the captured input port plus its producer lineage.
    object.__setattr__(victim, "artifact", source.artifact)
    with pytest.raises(EffectRejected) as error:
        qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert error.value.code is EffectCode.CONTRADICTORY


def test_failed_ordinary_prerequisite_blocks_member_after_item_publication() -> None:
    fixture = _direct_item_fixture()
    body = fixture.workflow.workflow
    failed_source = NodeId.new(workflow=body.workflow)
    failure_operation = OperationSpec(
        name="failed-prerequisite",
        inputs=(),
        outputs=(OutputPort(name="default", artifact_type=fixture.text_type),),
        output_dependencies=(OutputDependency(output="default", inputs=frozenset(), identity_input=None),),
        outcomes=(_outcome("available", produced=frozenset({"default"})),),
    )
    member = next(node for node in body.nodes if node.id == fixture.member)
    member = replace(
        member,
        operation=replace(
            member.operation,
            inputs=member.operation.inputs + (InputPort(name="required", artifact_type=fixture.text_type),),
        ),
    )
    interface = replace(
        body.interface,
        outcomes=tuple(
            replace(outcome, ceiling=replace(outcome.ceiling, max_activations=outcome.ceiling.max_activations + 1))
            for outcome in body.interface.outcomes
        ),
    )
    static = admit_static_workflow(
        workflow=body.workflow,
        interface=interface,
        nodes=tuple(member if node.id == member.id else node for node in body.nodes)
        + (OperationNode(id=failed_source, operation=failure_operation),),
        input_bindings=tuple(body.input_bindings)
        + (
            InputBinding(
                source=NodeOutputRef(node=failed_source, port="default"),
                destination=NodeInputRef(node=member.id, port="required"),
            ),
        ),
        output_bindings=tuple(body.output_bindings),
        outcome_bindings=tuple(body.outcome_bindings),
        sequence=tuple(body.sequence) + (SequenceEdge(before=failed_source, after=member.id),),
        choices=tuple(body.choices),
        protection=tuple(body.protection_requirements),
        limits=replace(
            body.limits,
            max_nodes=body.limits.max_nodes + 1,
            max_bindings=body.limits.max_bindings + 1,
            max_sequence_edges=body.limits.max_sequence_edges + 1,
        ),
    )
    dynamic = admit_activation_workflow(
        workflow=static,
        scopes=(replace(fixture.workflow.scopes[0], workflow=static),),
        limits=replace(
            fixture.workflow.limits, max_activation_occurrences=fixture.workflow.limits.max_activation_occurrences + 1
        ),
    )
    implementation_nodes = fixture.implementation_nodes + (failed_source,)
    operations = {node.id: node.operation for node in static.nodes}
    fixture = replace(
        fixture,
        workflow=dynamic,
        default_source=failed_source,
        implementation_nodes=implementation_nodes,
        capabilities=tuple(_capability(operations[node], index) for index, node in enumerate(implementation_nodes)),
    )
    _, result, callbacks = asyncio.run(
        _execute_membership(
            1,
            fixture=fixture,
            member_assessment=True,
            artifact_byte_headroom=64,
            port_fact_headroom=16,
            provenance_edge_headroom=16,
        )
    )
    assert len(callbacks[failed_source].calls) == 1
    assert not callbacks[fixture.member].calls
    (expansion,) = result.states[0].expansions
    assert expansion.status == "closed" and len(expansion.members) == 1
    member_entry = next(entry for entry in result.states[0].entries if entry.template == fixture.member)
    assert member_entry.status == "blocked" and member_entry.outcome is None
    terminal = next(item for item in result.record.terminals if item.activation == member_entry.activation)
    assert (
        terminal.attempt is None and terminal.category == "blocked" and terminal.reasons == frozenset({"prerequisite"})
    )
    assert not any(fact.node == fixture.member for fact in result.assessments)
    assert any(
        isinstance(fact.key, MapItemKey) and fact.key.member == member_entry.activation for fact in result.provenance
    )
    assert any(entry.template == failed_source for entry in result.states[0].entries)
    admitted, current, submissions = _qualification_inputs(result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert not output.qualified and "terminal_failure" in output.targets[0].withholding
