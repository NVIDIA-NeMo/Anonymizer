# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import replace

import pytest

from anonymizer.graph import (
    ContractViolation,
    NodeId,
    NodeOutcomeRef,
    OperationNode,
    OperationSpec,
    OutcomeBinding,
    ValidationCode,
    WorkflowId,
    WorkflowLimits,
    WorkflowOutcomeRef,
)
from tests.graph_sdk.test_activation import _interface_outcome, _outcome
from tests.graph_sdk.workflow_construction import admit_static_workflow


def test_unjoined_body_roots_reject_before_activation_admission() -> None:
    child_owner = WorkflowId.new()
    child_node = NodeId.new(workflow=child_owner)
    other_node = NodeId.new(workflow=child_owner)
    outcomes = (_outcome("ok", "success"), _outcome("fail", "failure"))
    child_operation = OperationSpec(name="child", inputs=(), outputs=(), output_dependencies=(), outcomes=outcomes)
    child_interface = OperationSpec(
        name="child-interface",
        inputs=(),
        outputs=(),
        output_dependencies=(),
        outcomes=tuple((_interface_outcome(outcome, 2) for outcome in outcomes)),
    )
    with pytest.raises(ContractViolation) as error:
        admit_static_workflow(
            workflow=child_owner,
            interface=child_interface,
            nodes=(
                OperationNode(id=child_node, operation=child_operation),
                OperationNode(
                    id=other_node,
                    operation=replace(child_operation, outcomes=(outcomes[0],)),
                ),
            ),
            input_bindings=(),
            output_bindings=(),
            outcome_bindings=tuple(
                (
                    OutcomeBinding(
                        source=NodeOutcomeRef(node=child_node, outcome=outcome.name),
                        destination=WorkflowOutcomeRef(outcome=outcome.name),
                    )
                    for outcome in outcomes
                )
            ),
            sequence=(),
            choices=(),
            protection=(),
            limits=WorkflowLimits(
                max_nodes=2,
                max_bindings=2,
                max_sequence_edges=0,
                max_choices=0,
                max_branch_members=0,
                max_subgraph_depth=1,
                max_choice_states=1,
            ),
        )
    assert error.value.code is ValidationCode.MISSING
