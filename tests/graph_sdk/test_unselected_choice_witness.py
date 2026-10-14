# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An admitted but unselected choice node creates no terminal obligation."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.graph.workflow import (
    ChoiceBranch,
    ChoiceDecl,
    NodeOutcomeRef,
    NodeOutputRef,
    OutcomeBinding,
    SequenceEdge,
    WorkflowOutcomeRef,
)
from tests.graph_sdk import evidence_fixtures as fixture
from tests.graph_sdk.qualification_fixtures import _inputs
from tests.graph_sdk.workflow_construction import AdmittedWorkflow, admit_static_workflow


def test_unselected_choice_retains_reservation_without_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    def admit(**kwargs: Any) -> AdmittedWorkflow:
        selector, spare, assessor = kwargs["nodes"]
        operation = selector.operation
        object.__setattr__(operation, "outcomes", (*operation.outcomes, replace(operation.outcomes[0], name="unused")))
        kwargs["input_bindings"] = tuple(
            replace(binding, source=replace(binding.source, node=selector.id))
            if isinstance(binding.source, NodeOutputRef) and binding.source.node == spare.id
            else binding
            for binding in kwargs["input_bindings"]
        )
        kwargs["sequence"] = (
            SequenceEdge(before=selector.id, after=spare.id),
            SequenceEdge(before=selector.id, after=assessor.id),
        )
        kwargs["choices"] = (
            ChoiceDecl(
                selector=selector.id,
                branches=(
                    ChoiceBranch(outcomes=frozenset({"ok"}), members=frozenset({assessor.id})),
                    ChoiceBranch(outcomes=frozenset({"unused"}), members=frozenset({spare.id})),
                ),
            ),
        )
        interface = kwargs["interface"]
        kwargs["interface"] = replace(
            interface,
            outcomes=(
                *interface.outcomes,
                replace(
                    interface.outcomes[0],
                    name="unused",
                    produced_ports=frozenset(),
                    evidence=frozenset(),
                    state_effects=frozenset(),
                ),
            ),
        )
        kwargs["outcome_bindings"] = (
            *kwargs["outcome_bindings"],
            *(
                OutcomeBinding(
                    source=NodeOutcomeRef(node=spare.id, outcome=outcome.name),
                    destination=WorkflowOutcomeRef(outcome="unused"),
                )
                for outcome in spare.operation.outcomes
            ),
        )
        kwargs["limits"] = replace(
            kwargs["limits"],
            max_choices=1,
            max_branch_members=2,
            max_choice_states=2,
            max_sequence_edges=3,
            max_bindings=12,
        )
        return admit_static_workflow(**kwargs)

    monkeypatch.setattr(fixture, "admit_static_workflow", admit)
    execution, result = asyncio.run(fixture._execute_assessment(decision_input=True, decision_hop=True))
    state = result.states[0]
    admitted, current, submissions = _inputs(execution, result)
    assert len(state.entries) == 2
    assert len(state.reservations) == 3
    active = {entry.activation for entry in state.entries}
    assert {terminal.activation for terminal in result.record.terminals} == active
    assert frozenset(key for membership in result.record.memberships for key in membership.members) == active
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(output.qualified) == 1
    assert output.record.statuses[0].completion == "closed"
    assert output.record.statuses[0].qualification == "met"
