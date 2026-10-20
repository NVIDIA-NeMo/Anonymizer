# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Protected map items retain a real container-expander producer."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from anonymizer.engine.graph_sdk.executor import MapItemKey, OperationOutputKey, admit_execution_plan
from anonymizer.engine.graph_sdk.preparation import prepare
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.graph import (
    InputBinding,
    NodeId,
    NodeInputRef,
    NodeOutcomeRef,
    NodeOutputRef,
    OperationNode,
    OutcomeBinding,
    OutputBinding,
    SubgraphNode,
    WorkflowId,
    WorkflowInputRef,
    WorkflowOutcomeRef,
    WorkflowOutputRef,
)
from tests.graph_sdk import qualification_map_execution as fixture
from tests.graph_sdk.test_map_item_boundary_witnesses import _qualification_inputs
from tests.graph_sdk.workflow_construction import DynamicScope, admit_activation_workflow, admit_static_workflow


def _container_expander(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Move only EXP implementation into a one-operation body, retaining its outer map identity."""
    context: dict[str, Any] = {}

    def static(**kwargs: Any) -> Any:
        (expander,) = tuple(node for node in kwargs["nodes"] if node.operation.name == "EXP")
        owner = WorkflowId.new()
        child = NodeId.new(workflow=owner)
        interface = replace(
            expander.operation,
            outcomes=tuple(
                replace(outcome, ceiling=replace(outcome.ceiling, max_activations=outcome.ceiling.max_activations + 1))
                for outcome in expander.operation.outcomes
            ),
        )
        body = admit_static_workflow(
            workflow=owner,
            interface=interface,
            nodes=(OperationNode(id=child, operation=expander.operation),),
            input_bindings=tuple(
                InputBinding(
                    source=WorkflowInputRef(port=port.name), destination=NodeInputRef(node=child, port=port.name)
                )
                for port in interface.inputs
            ),
            output_bindings=tuple(
                OutputBinding(
                    source=NodeOutputRef(node=child, port=port.name), destination=WorkflowOutputRef(port=port.name)
                )
                for port in interface.outputs
            ),
            outcome_bindings=tuple(
                OutcomeBinding(
                    source=NodeOutcomeRef(node=child, outcome=outcome.name),
                    destination=WorkflowOutcomeRef(outcome=outcome.name),
                )
                for outcome in interface.outcomes
            ),
            sequence=(),
            choices=(),
            protection=(),
            limits=replace(kwargs["limits"], max_subgraph_depth=2),
        )
        context.update(expander=expander.id, child=child, body=body)
        kwargs["nodes"] = tuple(
            SubgraphNode(id=expander.id, operation=interface, body=body) if node is expander else node
            for node in kwargs["nodes"]
        )
        kwargs["interface"] = replace(
            kwargs["interface"],
            outcomes=tuple(
                replace(outcome, ceiling=replace(outcome.ceiling, max_activations=outcome.ceiling.max_activations + 1))
                for outcome in kwargs["interface"].outcomes
            ),
        )
        kwargs["limits"] = replace(kwargs["limits"], max_subgraph_depth=2, max_nodes=kwargs["limits"].max_nodes + 1)
        return admit_static_workflow(**kwargs)

    def activation(**kwargs: Any) -> Any:
        kwargs["scopes"] = (*kwargs["scopes"], DynamicScope(workflow=context["body"], maps=(), joins=(), loops=()))
        kwargs["limits"] = replace(
            kwargs["limits"],
            max_activation_occurrences=kwargs["limits"].max_activation_occurrences + 1,
            max_dynamic_depth=2,
        )
        return admit_activation_workflow(**kwargs)

    def preparation(**kwargs: Any) -> Any:
        kwargs["selections"] = tuple(
            replace(row, node=context["child"]) if row.node == context["expander"] else row
            for row in kwargs["selections"]
        )
        kwargs["activation_limits"] = replace(
            kwargs["activation_limits"], max_entries=kwargs["activation_limits"].max_entries + 1, max_parent_depth=3
        )
        kwargs["limits"] = replace(
            kwargs["limits"], max_total_activation_slots=kwargs["limits"].max_total_activation_slots + 1
        )
        return prepare(**kwargs)

    def execution(**kwargs: Any) -> Any:
        kwargs["policies"] = tuple(
            replace(row, node=context["child"]) if row.node == context["expander"] else row
            for row in kwargs["policies"]
        )
        kwargs["assessment_limits"] = replace(
            kwargs["assessment_limits"],
            max_port_facts=kwargs["assessment_limits"].max_port_facts + 2,
            max_provenance_edges=kwargs["assessment_limits"].max_provenance_edges + 1,
        )
        return admit_execution_plan(**kwargs)

    monkeypatch.setattr(fixture, "admit_static_workflow", static)
    monkeypatch.setattr(fixture, "admit_activation_workflow", activation)
    monkeypatch.setattr(fixture, "prepare", preparation)
    monkeypatch.setattr(fixture, "admit_execution_plan", execution)
    return context


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("container", [False, True])
def test_protected_map_qualifies_with_exact_expander_kind(
    count: int, container: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _container_expander(monkeypatch) if container else {}
    execution, result, nodes = asyncio.run(fixture._execute_reference_map(count))
    admitted, current, submissions = _qualification_inputs(result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(output.qualified) == 1 and output.record.statuses[0].qualification == "met"
    assert len(output.verified) == count + 1
    (expansion,) = result.states[0].expansions
    (terminal,) = tuple(row for row in result.record.terminals if row.activation == expansion.parent)
    assert terminal.structural is container
    assert (terminal.attempt is None) is container
    items = tuple(row for row in result.provenance if isinstance(row.key, MapItemKey))
    assert len(items) == count and len(expansion.members) == count
    for item in items:
        assert isinstance(item.key, MapItemKey)
        assert item.key.expander == expansion.parent and item.key.member in expansion.members
        assert item.key.item_version == 1
        (producer,) = item.parents
        assert isinstance(producer, OperationOutputKey) and producer.activation == expansion.parent
        fact = next(row for row in result.provenance if row.key == producer)
        if container:
            (body_parent,) = fact.parents
            assert isinstance(body_parent, OperationOutputKey)
            assert body_parent.activation.parent == expansion.parent
            assert any(
                entry.activation == body_parent.activation and entry.template == context["child"]
                for entry in result.states[0].entries
            )
    assert not output.targets[0].withholding


def test_single_item_map_qualifies_with_one_map_item_producer() -> None:
    _, result, _ = asyncio.run(fixture._execute_reference_map(1))
    admitted, current, submissions = _qualification_inputs(result)
    baseline = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(baseline.qualified) == 1
    (fact,) = tuple(row for row in result.provenance if isinstance(row.key, MapItemKey))
    original = fact.key
    assert isinstance(original, MapItemKey)


def test_execution_only_map_item_assessment_verifies_without_release(monkeypatch: pytest.MonkeyPatch) -> None:
    from anonymizer.engine.graph_sdk.evidence import (
        AssessmentSubmission,
        MapItemSubjectRef,
        admit_qualification,
        evidence_revision_view,
        verify_evidence,
    )
    from tests.graph_sdk.evidence_fixtures import _qualification_limits

    configuration = fixture.PreparationConfiguration

    def execution_only(**kwargs: Any) -> Any:
        return replace(configuration(**kwargs), purpose="execution_only", required_protection_outcomes=frozenset())

    monkeypatch.setattr(fixture, "PreparationConfiguration", execution_only)
    execution, result, _ = asyncio.run(fixture._execute_reference_map(1))
    admitted = admit_qualification(
        execution=execution,
        productions=execution.assessment_productions,
        limits=_qualification_limits(max_revision_entries=32),
    )
    current = evidence_revision_view(
        admitted=admitted,
        result=result,
        artifacts=tuple(ref for ref, _ in result.artifacts),
        configurations=tuple(
            (item.node, item.capability.configuration) for item in execution.context.prepared.implementations
        ),
    )
    submissions = tuple(AssessmentSubmission(fact=fact) for fact in result.assessments)
    verified = verify_evidence(admitted=admitted, result=result, submissions=submissions)
    assert sum(isinstance(item.subject, MapItemSubjectRef) for item in verified) == 1
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert not output.qualified
    assert output.targets[0].withholding == frozenset({"execution_only"})
