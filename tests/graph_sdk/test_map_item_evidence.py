# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed map-item evidence must bind to an actual admitted map."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected
from anonymizer.engine.graph_sdk.evidence import (
    AssessmentSubmission,
    MapItemSubjectRef,
    admit_qualification,
    evidence_revision_view,
    evidence_validity,
    verify_evidence,
)
from anonymizer.engine.graph_sdk.executor import (
    AssessmentFinding,
    EvidenceProductionDecl,
    MapExpansionDecl,
    MapItemKey,
    RootInputKey,
)
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.engine.graph_sdk.records import CandidateRef
from anonymizer.graph._values import ContractViolation, ValidationCode
from anonymizer.graph.workflow import (
    DynamicScope,
    InputBinding,
    InputPort,
    KeyedJoinDecl,
    MapDecl,
    MapItemPort,
    NodeId,
    NodeInputRef,
    NodeOutcomeRef,
    NodeOutputRef,
    OperationNode,
    OutcomeBinding,
    OutputBinding,
    OutputDependency,
    OutputPort,
    ProtectionRequirement,
    SequenceEdge,
    SubgraphNode,
    WorkflowId,
    WorkflowInputRef,
    WorkflowLimits,
    WorkflowOutcomeRef,
    WorkflowOutputRef,
    admit_activation_workflow,
    admit_static_workflow,
)
from tests.graph_sdk.test_dynamic_executor import _capability
from tests.graph_sdk.test_effects_map_production_conformance import (
    _admit_fixture,
    _execute_membership,
    _map_fixture,
    _MapFixture,
)
from tests.graph_sdk.test_evidence import _qualification_limits


def _direct_item_fixture(*, candidate_uses_membership: bool = True) -> _MapFixture:
    fixture = _map_fixture(member_assessment=True)
    body = fixture.workflow.workflow
    endpoint = MapItemPort(
        path=(),
        expander=fixture.expander,
        member=fixture.member,
        item_input="item",
        membership_port="members",
        expansion_outcome="expand",
    )
    member = next(node for node in body.nodes if node.id == fixture.member)
    promise = next(iter(member.operation.outcomes[0].evidence))
    operation = replace(
        member.operation,
        outcomes=(
            replace(
                member.operation.outcomes[0],
                evidence=frozenset(
                    {
                        replace(
                            promise,
                            subject_port="item",
                            consumed_ports=frozenset({"item"}),
                        )
                    }
                ),
            ),
        ),
        output_dependencies=tuple(
            replace(item, inputs=frozenset({"item"})) for item in member.operation.output_dependencies
        ),
    )
    interface = replace(
        body.interface,
        outputs=(*body.interface.outputs, OutputPort(name="value", artifact_type=fixture.text_type)),
        output_dependencies=(
            *body.interface.output_dependencies,
            OutputDependency(
                output="value",
                inputs=frozenset({"default"}),
                identity_input=None if candidate_uses_membership else "default",
            ),
        ),
        outcomes=tuple(
            replace(
                outcome,
                produced_ports=outcome.produced_ports | {"value"},
                evidence=frozenset(
                    replace(item, subject_port=endpoint, consumed_ports=frozenset({endpoint}))
                    if item.name == promise.name
                    else item
                    for item in outcome.evidence
                ),
            )
            for outcome in body.interface.outcomes
        ),
    )
    join = next(node for node in body.nodes if node.id == fixture.join)
    join_operation = replace(
        join.operation,
        inputs=(InputPort(name="members", artifact_type=fixture.collection_type),),
        outputs=(OutputPort(name="value", artifact_type=fixture.text_type),),
        output_dependencies=(OutputDependency(output="value", inputs=frozenset({"members"}), identity_input=None),),
        outcomes=tuple(
            replace(outcome, produced_ports=frozenset({"value"}), ceiling=replace(outcome.ceiling, max_output_bytes=32))
            for outcome in join.operation.outcomes
        ),
    )
    static = admit_static_workflow(
        workflow=body.workflow,
        interface=interface,
        nodes=tuple(
            OperationNode(id=node.id, operation=operation)
            if node.id == member.id
            else OperationNode(id=node.id, operation=join_operation)
            if node.id == join.id
            else node
            for node in body.nodes
        ),
        input_bindings=(
            *body.input_bindings,
            InputBinding(
                source=NodeOutputRef(node=fixture.expander, port="members"),
                destination=NodeInputRef(node=fixture.join, port="members"),
            ),
        ),
        output_bindings=(
            *body.output_bindings,
            OutputBinding(
                source=NodeOutputRef(node=fixture.join, port="value")
                if candidate_uses_membership
                else WorkflowInputRef(port="default"),
                destination=WorkflowOutputRef(port="value"),
            ),
        ),
        outcome_bindings=tuple(body.outcome_bindings),
        sequence=tuple(body.sequence),
        choices=tuple(body.choices),
        protection=(
            ProtectionRequirement(
                outcome="ok",
                meaning=promise.meaning,
                subject_port=endpoint,
                consumed_ports=frozenset({endpoint}),
                coverage=frozenset(),
                candidate_port="value",
            ),
        ),
        limits=replace(body.limits, max_bindings=body.limits.max_bindings + 3),
    )
    dynamic = admit_activation_workflow(
        workflow=static,
        scopes=(replace(fixture.workflow.scopes[0], workflow=static),),
        limits=fixture.workflow.limits,
    )
    operations = {node.id: node.operation for node in static.nodes}
    return replace(
        fixture,
        workflow=dynamic,
        capabilities=tuple(
            _capability(operations[node], index) for index, node in enumerate(fixture.implementation_nodes)
        ),
    )


def test_direct_item_endpoint_survives_static_dynamic_and_execution_admission() -> None:
    fixture = _direct_item_fixture()
    endpoint = next(iter(fixture.workflow.workflow.map_item_ports))
    execution = _admit_fixture(fixture, port_fact_headroom=12)
    assert execution.context.prepared.workflow is fixture.workflow
    assert endpoint.member == fixture.member
    assert endpoint.expander == fixture.expander
    assert endpoint.membership_port == execution.map_expansions[0].membership_port


def _nested_item_fixture(fixture: _MapFixture) -> _MapFixture:
    body = fixture.workflow.workflow
    owner = WorkflowId.new()
    container = NodeId.new(workflow=owner)
    interface = replace(
        body.interface,
        outcomes=tuple(
            replace(
                outcome,
                evidence=frozenset(
                    replace(
                        promise,
                        subject_port=promise.subject_port.lifted(container)
                        if isinstance(promise.subject_port, MapItemPort)
                        else promise.subject_port,
                        consumed_ports=frozenset(
                            port.lifted(container) if isinstance(port, MapItemPort) else port
                            for port in promise.consumed_ports
                        ),
                    )
                    for promise in outcome.evidence
                ),
            )
            for outcome in body.interface.outcomes
        ),
    )
    static = admit_static_workflow(
        workflow=owner,
        interface=interface,
        nodes=(SubgraphNode(id=container, operation=body.interface, body=body),),
        input_bindings=(
            InputBinding(
                source=WorkflowInputRef(port="default"), destination=NodeInputRef(node=container, port="default")
            ),
        ),
        output_bindings=tuple(
            OutputBinding(
                source=NodeOutputRef(node=container, port=port.name), destination=WorkflowOutputRef(port=port.name)
            )
            for port in body.interface.outputs
        ),
        outcome_bindings=(
            OutcomeBinding(
                source=NodeOutcomeRef(node=container, outcome="ok"), destination=WorkflowOutcomeRef(outcome="ok")
            ),
        ),
        sequence=(),
        choices=(),
        protection=tuple(
            replace(
                requirement,
                subject_port=requirement.subject_port.lifted(container)
                if isinstance(requirement.subject_port, MapItemPort)
                else requirement.subject_port,
                consumed_ports=frozenset(
                    port.lifted(container) if isinstance(port, MapItemPort) else port
                    for port in requirement.consumed_ports
                ),
            )
            for requirement in body.protection_requirements
        ),
        limits=WorkflowLimits(
            max_nodes=body.expanded_node_count + 1,
            max_bindings=5,
            max_sequence_edges=0,
            max_choices=0,
            max_branch_members=0,
            max_subgraph_depth=body.limits.max_subgraph_depth + 1,
            max_choice_states=1,
        ),
    )
    dynamic = admit_activation_workflow(
        workflow=static,
        scopes=(DynamicScope(workflow=static, maps=(), joins=(), loops=()), *fixture.workflow.scopes),
        limits=replace(
            fixture.workflow.limits,
            max_dynamic_depth=fixture.workflow.limits.max_dynamic_depth + 1,
            max_activation_occurrences=fixture.workflow.limits.max_activation_occurrences + 1,
        ),
    )
    return replace(fixture, workflow=dynamic)


def _consumed_item_fixture() -> _MapFixture:
    fixture = _direct_item_fixture()
    body = fixture.workflow.workflow
    producer = NodeId.new(workflow=body.workflow)
    nodes = {node.id: node for node in body.nodes}
    operation = replace(nodes[fixture.join].operation, name="candidate")
    member = nodes[fixture.member]
    member_operation = replace(
        member.operation,
        outcomes=tuple(
            replace(
                outcome, evidence=frozenset(replace(promise, subject_port="subject") for promise in outcome.evidence)
            )
            for outcome in member.operation.outcomes
        ),
    )
    interface = replace(
        body.interface,
        outcomes=tuple(
            replace(
                outcome,
                evidence=frozenset(
                    replace(promise, subject_port="value") if promise.name == "member_checked" else promise
                    for promise in outcome.evidence
                ),
                ceiling=replace(
                    outcome.ceiling,
                    max_activations=outcome.ceiling.max_activations + 1,
                    max_input_bytes=outcome.ceiling.max_input_bytes + operation.outcomes[0].ceiling.max_input_bytes,
                    max_output_bytes=outcome.ceiling.max_output_bytes + operation.outcomes[0].ceiling.max_output_bytes,
                ),
            )
            for outcome in body.interface.outcomes
        ),
    )
    static = admit_static_workflow(
        workflow=body.workflow,
        interface=interface,
        nodes=(
            *(
                OperationNode(id=node.id, operation=member_operation) if node.id == fixture.member else node
                for node in body.nodes
            ),
            OperationNode(id=producer, operation=operation),
        ),
        input_bindings=(
            *(
                replace(binding, source=NodeOutputRef(node=producer, port="value"))
                if binding.destination == NodeInputRef(node=fixture.member, port="subject")
                else binding
                for binding in body.input_bindings
            ),
            InputBinding(
                source=NodeOutputRef(node=fixture.expander, port="members"),
                destination=NodeInputRef(node=producer, port="members"),
            ),
        ),
        output_bindings=tuple(
            replace(binding, source=NodeOutputRef(node=producer, port="value"))
            if binding.destination.port == "value"
            else binding
            for binding in body.output_bindings
        ),
        outcome_bindings=tuple(body.outcome_bindings),
        sequence=(
            *body.sequence,
            SequenceEdge(before=fixture.expander, after=producer),
            SequenceEdge(before=producer, after=fixture.member),
        ),
        choices=(),
        protection=tuple(replace(requirement, subject_port="value") for requirement in body.protection_requirements),
        limits=replace(
            body.limits,
            max_nodes=body.limits.max_nodes + 1,
            max_bindings=body.limits.max_bindings + 1,
            max_sequence_edges=body.limits.max_sequence_edges + 2,
        ),
    )
    dynamic = admit_activation_workflow(
        workflow=static,
        scopes=(replace(fixture.workflow.scopes[0], workflow=static),),
        limits=replace(
            fixture.workflow.limits, max_activation_occurrences=fixture.workflow.limits.max_activation_occurrences + 1
        ),
    )
    implementations = (*fixture.implementation_nodes, producer)
    operations = {node.id: node.operation for node in static.nodes}
    return replace(
        fixture,
        workflow=dynamic,
        implementation_nodes=implementations,
        capabilities=tuple(_capability(operations[node], index) for index, node in enumerate(implementations)),
    )


def test_direct_item_endpoint_cannot_be_admitted_without_its_dynamic_map() -> None:
    fixture = _direct_item_fixture()
    scope = fixture.workflow.scopes[0]
    with pytest.raises(ContractViolation) as error:
        admit_activation_workflow(
            workflow=scope.workflow,
            scopes=(replace(scope, maps=(), joins=()),),
            limits=fixture.workflow.limits,
        )
    assert error.value.code is ValidationCode.MISSING


def test_direct_item_endpoint_rejects_a_different_membership_output() -> None:
    fixture = _direct_item_fixture()
    with pytest.raises(EffectRejected) as error:
        _admit_fixture(
            fixture,
            map_expansions=(
                MapExpansionDecl(
                    expander=fixture.expander,
                    outcome="expand",
                    membership_port="wrong",
                    item_type=fixture.text_type,
                ),
            ),
        )
    assert error.value.code is EffectCode.CONTRADICTORY


def test_direct_item_subject_requires_a_separate_final_candidate_port() -> None:
    fixture = _direct_item_fixture()
    requirement = next(iter(fixture.workflow.workflow.protection_requirements))
    with pytest.raises(ContractViolation) as error:
        replace(requirement, candidate_port=None)
    assert error.value.code is ValidationCode.MISSING


def test_leaf_operation_cannot_declare_a_composite_map_endpoint() -> None:
    fixture = _direct_item_fixture()
    with pytest.raises(ContractViolation) as error:
        OperationNode(id=fixture.member, operation=fixture.workflow.workflow.interface)
    assert error.value.code is ValidationCode.UNSUPPORTED


@pytest.mark.parametrize("count", [0, 1, 2])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "mode", ["complete", "missing_submission", "missing_item_revision", "stale_item_revision", "unrelated_candidate"]
)
def test_actual_member_evidence_authenticates_its_item_subject(count: int, mode: str, nested: bool) -> None:
    fixture = _direct_item_fixture(candidate_uses_membership=mode != "unrelated_candidate")
    if nested:
        fixture = _nested_item_fixture(fixture)
    _, result, callbacks = asyncio.run(
        _execute_membership(
            count,
            fixture=fixture,
            member_assessment=True,
            artifact_byte_headroom=64,
            port_fact_headroom=16,
            provenance_edge_headroom=16,
            same_key_versions=mode == "stale_item_revision",
        )
    )
    execution = result._execution
    admitted = admit_qualification(
        execution=execution,
        productions=tuple(item for item in execution.assessment_productions if item.node == fixture.member),
        limits=_qualification_limits(),
    )
    facts = tuple(item for item in result.assessments if item.node == fixture.member)
    assert len(facts) == count
    assert len(callbacks[fixture.member].calls) == count
    verified = verify_evidence(
        admitted=admitted,
        result=result,
        submissions=tuple(AssessmentSubmission(fact=item) for item in facts),
    )
    assert len(verified) == count
    for item in verified:
        assert isinstance(item.subject, MapItemSubjectRef)
        assert item.subject.producer.member == item.activation
        assert item.subject.producer.target in result.record.targets
        assert item.consumed_by_port == (("item", item.subject.artifact),)
    current_artifacts = tuple(ref for ref, _ in result.artifacts)
    if mode == "stale_item_revision":
        current_artifacts = tuple(
            {ref.key: ref for ref in sorted(current_artifacts, key=lambda ref: ref.version)}.values()
        )
    current = evidence_revision_view(
        admitted=admitted,
        result=result,
        artifacts=tuple(
            ref
            for ref in current_artifacts
            if mode != "missing_item_revision" or not verified or ref != verified[0].subject.artifact
        ),
        absences=(),
        configurations=tuple({(fact.node, fact.environment.configuration) for fact in result.assessments}),
        state=execution.context.prepared.state,
    )
    for index, item in enumerate(verified):
        expected = (
            "unknown"
            if mode == "missing_item_revision" and index == 0
            else ("stale" if mode == "stale_item_revision" and count == 2 and index == 0 else "current")
        )
        assert evidence_validity(evidence=item, current=current) == expected
    output = qualify(
        admitted=admitted,
        result=result,
        current=current,
        submissions=tuple(
            AssessmentSubmission(fact=item) for item in (facts[1:] if mode == "missing_submission" else facts)
        ),
    )
    if (
        mode == "unrelated_candidate"
        or count
        and mode in {"missing_submission", "missing_item_revision"}
        or count == 2
        and mode == "stale_item_revision"
    ):
        assert not output.qualified
        code = {
            "unrelated_candidate": "missing_assessment",
            "missing_submission": "missing_assessment",
            "missing_item_revision": "assessment_unknown",
            "stale_item_revision": "stale_evidence",
        }[mode]
        assert code in output.targets[0].withholding
        return
    assert len(output.qualified) == 1
    assert output.qualified[0].candidate == next(
        item.candidate for item in result.final_outputs if item.port == "value"
    )
    assert len(output.qualified[0].evidence) == count


@pytest.mark.parametrize("count", [0, 1, 2])
@pytest.mark.parametrize("nested", [False, True])
def test_map_item_can_be_consumed_while_subject_is_the_final_scalar_candidate(count: int, nested: bool) -> None:
    fixture = _consumed_item_fixture()
    if nested:
        fixture = _nested_item_fixture(fixture)
    _, result, _ = asyncio.run(
        _execute_membership(
            count,
            fixture=fixture,
            member_assessment=True,
            artifact_byte_headroom=128,
            artifact_headroom=12,
            port_fact_headroom=20,
            provenance_edge_headroom=20,
        )
    )
    execution = result._execution
    admitted = admit_qualification(
        execution=execution,
        productions=tuple(item for item in execution.assessment_productions if item.node == fixture.member),
        limits=_qualification_limits(),
    )
    facts = tuple(item for item in result.assessments if item.node == fixture.member)
    assert len(facts) == count
    current = evidence_revision_view(
        admitted=admitted,
        result=result,
        artifacts=tuple(ref for ref, _ in result.artifacts),
        absences=(),
        configurations=tuple({(fact.node, fact.environment.configuration) for fact in result.assessments}),
        state=execution.context.prepared.state,
    )
    output = qualify(
        admitted=admitted,
        result=result,
        current=current,
        submissions=tuple(AssessmentSubmission(fact=item) for item in facts),
    )
    assert len(output.qualified) == 1
    candidate = output.qualified[0].candidate
    for item in output.verified:
        assert isinstance(item.subject, CandidateRef)
        assert item.subject == candidate
        assert dict(item.consumed_by_port)["item"] != candidate.artifact
        assert evidence_validity(evidence=item, current=current) == "current"
    assert len(output.qualified[0].evidence) == count


@pytest.mark.parametrize("consumed_only", [False, True])
@pytest.mark.parametrize(
    "mutation",
    ["sibling", "default", "item_version", "coordinated_item_key", "equal_value_sibling", "decision_provenance"],
)
def test_item_authentication_rejects_substituted_captured_input_owner(consumed_only: bool, mutation: str) -> None:
    fixture = _consumed_item_fixture() if consumed_only else _direct_item_fixture()
    _, result, _ = asyncio.run(
        _execute_membership(
            2,
            fixture=fixture,
            member_assessment=True,
            item_values=("same", "same") if mutation == "equal_value_sibling" else None,
            artifact_byte_headroom=128,
            artifact_headroom=12,
            port_fact_headroom=20,
            provenance_edge_headroom=20,
        )
    )
    execution = result._execution
    admitted = admit_qualification(
        execution=execution,
        productions=tuple(item for item in execution.assessment_productions if item.node == fixture.member),
        limits=_qualification_limits(),
    )
    facts = tuple(item for item in result.assessments if item.node == fixture.member)
    assert len(facts) == 2
    first, second = facts
    parents = result._input_parents
    original = next(key for _, activation, port, key in parents if activation == first.activation and port == "item")
    assert isinstance(original, MapItemKey)
    if mutation == "sibling":
        replacement = next(
            key for _, activation, port, key in parents if activation == second.activation and port == "item"
        )
    elif mutation == "default":
        replacement = next(item.key for item in result.provenance if isinstance(item.key, RootInputKey))
    elif mutation == "item_version":
        replacement = replace(original, item_version=original.item_version + 1)
    elif mutation == "decision_provenance":
        replacement = original
        provenance = next(item for item in result.provenance if item.key == original)
        object.__setattr__(provenance, "decision", True)
    else:
        if mutation == "equal_value_sibling":
            sibling = next(
                key for _, activation, port, key in parents if activation == second.activation and port == "item"
            )
            assert isinstance(sibling, MapItemKey)
            replacement = replace(original, item_key=sibling.item_key, item_version=sibling.item_version)
        else:
            replacement = replace(original, item_key=original.item_key + 99)
        provenance = next(item for item in result.provenance if item.key == original)
        object.__setattr__(provenance, "key", replacement)
    object.__setattr__(
        result,
        "_input_parents",
        tuple(
            (target, activation, port, replacement if activation == first.activation and port == "item" else key)
            for target, activation, port, key in parents
        ),
    )
    with pytest.raises(EffectRejected) as error:
        verify_evidence(admitted=admitted, result=result, submissions=(AssessmentSubmission(fact=first),))
    assert error.value.code is EffectCode.CONTRADICTORY


@pytest.mark.parametrize("bound", ["max_submissions", "max_verified_evidence"])
def test_dynamic_evidence_selection_obeys_exact_and_one_over_limits(bound: str) -> None:
    fixture = _direct_item_fixture()
    _, result, _ = asyncio.run(
        _execute_membership(
            2,
            fixture=fixture,
            member_assessment=True,
            artifact_byte_headroom=128,
            artifact_headroom=12,
            port_fact_headroom=20,
            provenance_edge_headroom=20,
        )
    )
    execution = result._execution
    facts = tuple(item for item in result.assessments if item.node == fixture.member)
    submissions = tuple(AssessmentSubmission(fact=item) for item in facts)
    for limit in (2, 1):
        admitted = admit_qualification(
            execution=execution,
            productions=tuple(item for item in execution.assessment_productions if item.node == fixture.member),
            limits=_qualification_limits(**{bound: limit}),
        )
        if limit == 2:
            assert len(verify_evidence(admitted=admitted, result=result, submissions=submissions)) == 2
        else:
            with pytest.raises(EffectRejected) as error:
                verify_evidence(admitted=admitted, result=result, submissions=submissions)
            assert error.value.code is EffectCode.LIMIT_EXCEEDED


def _two_map_fixture():
    fixture = _direct_item_fixture()
    body = fixture.workflow.workflow
    by_node = {node.id: node for node in body.nodes}
    right_expander, right_member, right_join, final = (NodeId.new(workflow=body.workflow) for _ in range(4))
    endpoint = MapItemPort(
        path=(),
        expander=right_expander,
        member=right_member,
        item_input="item",
        membership_port="members",
        expansion_outcome="expand",
    )

    def renamed(operation, name):
        return replace(
            operation,
            outcomes=tuple(
                replace(outcome, evidence=frozenset(replace(promise, name=name) for promise in outcome.evidence))
                for outcome in operation.outcomes
            ),
        )

    right_expander_operation = renamed(by_node[fixture.expander].operation, "right_membership")
    right_member_operation = renamed(by_node[fixture.member].operation, "right_checked")
    barrier = replace(
        by_node[fixture.join].operation,
        inputs=(),
        outputs=(),
        output_dependencies=(),
        outcomes=tuple(
            replace(outcome, produced_ports=frozenset()) for outcome in by_node[fixture.join].operation.outcomes
        ),
    )
    combine = replace(
        by_node[fixture.join].operation,
        inputs=(
            InputPort(name="members", artifact_type=fixture.collection_type),
            InputPort(name="right", artifact_type=fixture.collection_type),
        ),
        output_dependencies=(
            OutputDependency(output="value", inputs=frozenset({"members", "right"}), identity_input=None),
        ),
    )
    left_requirement = next(iter(body.protection_requirements))
    outcome = body.interface.outcomes[0]
    membership_promise = next(promise for promise in outcome.evidence if promise.name == "assessment0")
    item_promise = next(promise for promise in outcome.evidence if promise.name == "member_checked")
    interface = replace(
        body.interface,
        outputs=(*body.interface.outputs, OutputPort(name="right_members", artifact_type=fixture.collection_type)),
        output_dependencies=(
            *body.interface.output_dependencies,
            OutputDependency(output="right_members", inputs=frozenset({"default"}), identity_input=None),
        ),
        outcomes=(
            replace(
                outcome,
                produced_ports=outcome.produced_ports | {"right_members"},
                evidence=outcome.evidence
                | {
                    replace(membership_promise, name="right_membership", subject_port="right_members"),
                    replace(
                        item_promise, name="right_checked", subject_port=endpoint, consumed_ports=frozenset({endpoint})
                    ),
                },
                ceiling=replace(
                    outcome.ceiling,
                    max_activations=20,
                    max_input_bytes=outcome.ceiling.max_input_bytes * 3,
                    max_output_bytes=outcome.ceiling.max_output_bytes * 3,
                ),
            ),
        ),
    )
    static = admit_static_workflow(
        workflow=body.workflow,
        interface=interface,
        nodes=(
            *(OperationNode(id=node.id, operation=barrier) if node.id == fixture.join else node for node in body.nodes),
            OperationNode(id=right_expander, operation=right_expander_operation),
            OperationNode(id=right_member, operation=right_member_operation),
            OperationNode(id=right_join, operation=barrier),
            OperationNode(id=final, operation=combine),
        ),
        input_bindings=(
            *(
                replace(binding, destination=NodeInputRef(node=final, port="members"))
                if binding.destination.node == fixture.join
                else binding
                for binding in body.input_bindings
            ),
            InputBinding(
                source=WorkflowInputRef(port="default"), destination=NodeInputRef(node=right_expander, port="default")
            ),
            InputBinding(
                source=WorkflowInputRef(port="default"), destination=NodeInputRef(node=right_member, port="item")
            ),
            InputBinding(
                source=WorkflowInputRef(port="default"), destination=NodeInputRef(node=right_member, port="subject")
            ),
            InputBinding(
                source=NodeOutputRef(node=right_expander, port="members"),
                destination=NodeInputRef(node=final, port="right"),
            ),
        ),
        output_bindings=(
            *(
                replace(binding, source=NodeOutputRef(node=final, port="value"))
                if binding.destination.port == "value"
                else binding
                for binding in body.output_bindings
            ),
            OutputBinding(
                source=NodeOutputRef(node=right_expander, port="members"),
                destination=WorkflowOutputRef(port="right_members"),
            ),
        ),
        outcome_bindings=(
            OutcomeBinding(
                source=NodeOutcomeRef(node=final, outcome="ok"), destination=WorkflowOutcomeRef(outcome="ok")
            ),
        ),
        sequence=(
            *body.sequence,
            SequenceEdge(before=fixture.join, after=final),
            SequenceEdge(before=right_expander, after=right_member),
            SequenceEdge(before=right_member, after=right_join),
            SequenceEdge(before=right_join, after=final),
        ),
        choices=(),
        protection=(
            left_requirement,
            replace(left_requirement, subject_port=endpoint, consumed_ports=frozenset({endpoint})),
        ),
        limits=replace(body.limits, max_nodes=7, max_bindings=body.limits.max_bindings + 6, max_sequence_edges=6),
    )
    scope = fixture.workflow.scopes[0]
    dynamic = admit_activation_workflow(
        workflow=static,
        scopes=(
            replace(
                scope,
                workflow=static,
                maps=(
                    *scope.maps,
                    MapDecl(
                        expander=right_expander,
                        member=right_member,
                        expansion_outcomes=frozenset({"expand"}),
                        max_children=2,
                        item_input="item",
                    ),
                ),
                joins=(
                    *scope.joins,
                    KeyedJoinDecl(
                        source=right_expander,
                        join=right_join,
                        accepted_categories=frozenset({"success"}),
                        reduction="all_by_key",
                    ),
                ),
            ),
        ),
        limits=replace(fixture.workflow.limits, max_maps=2, max_joins=2, max_activation_occurrences=9),
    )
    implementations = (*fixture.implementation_nodes, right_expander, right_member, right_join, final)
    operations = {node.id: node.operation for node in static.nodes}
    fixture = replace(
        fixture,
        workflow=dynamic,
        implementation_nodes=implementations,
        capabilities=tuple(_capability(operations[node], index) for index, node in enumerate(implementations)),
    )
    productions = tuple(
        EvidenceProductionDecl(
            node=node,
            outcome=outcome,
            promise=promise,
            evidence_port=port,
            absence_queries=frozenset(),
            supported_findings=frozenset({AssessmentFinding(status="satisfied", code=promise)}),
        )
        for node, outcome, promise, port in (
            (fixture.expander, "expand", "assessment0", "members"),
            (fixture.member, "ok", "member_checked", "evidence"),
            (right_expander, "expand", "right_membership", "members"),
            (right_member, "ok", "right_checked", "evidence"),
        )
    )
    expansions = tuple(
        MapExpansionDecl(expander=node, outcome="expand", membership_port="members", item_type=fixture.text_type)
        for node in (fixture.expander, right_expander)
    )
    return fixture, right_member, productions, expansions


@pytest.mark.parametrize("omit_right", [False, True])
def test_two_maps_require_their_own_member_evidence(omit_right: bool) -> None:
    fixture, right_member, productions, expansions = _two_map_fixture()
    _, result, callbacks = asyncio.run(
        _execute_membership(
            2,
            fixture=fixture,
            member_assessment=True,
            assessment_productions=productions,
            map_expansions=expansions,
            artifact_headroom=20,
            artifact_byte_headroom=256,
            port_fact_headroom=32,
            provenance_edge_headroom=32,
        )
    )
    assert len(callbacks[fixture.member].calls) == len(callbacks[right_member].calls) == 2
    execution = result._execution
    admitted = admit_qualification(
        execution=execution,
        productions=tuple(item for item in productions if item.evidence_port == "evidence"),
        limits=_qualification_limits(max_port_facts=32, max_provenance_edges=32, max_revision_entries=32),
    )
    facts = tuple(item for item in result.assessments if item.node in {fixture.member, right_member})
    assert len(facts) == 4
    current = evidence_revision_view(
        admitted=admitted,
        result=result,
        artifacts=tuple(ref for ref, _ in result.artifacts),
        absences=(),
        configurations=tuple({(fact.node, fact.environment.configuration) for fact in result.assessments}),
        state=execution.context.prepared.state,
    )
    output = qualify(
        admitted=admitted,
        result=result,
        current=current,
        submissions=tuple(
            AssessmentSubmission(fact=fact) for fact in facts if not omit_right or fact.node != right_member
        ),
    )
    if omit_right:
        assert not output.qualified
        assert "missing_assessment" in output.targets[0].withholding
    else:
        assert len(output.qualified) == 1
        assert len(output.qualified[0].evidence) == 4
