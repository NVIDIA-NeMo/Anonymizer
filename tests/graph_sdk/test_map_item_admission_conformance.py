# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate frozen map requirements through public typed admission."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from anonymizer.graph._values import ContractViolation
from anonymizer.graph.workflow import (
    CoverageAtom,
    MapItemPort,
    NodeId,
    NodeOutcomeRef,
    OperationNode,
    OutcomeBinding,
    ProtectionRequirement,
    SubgraphNode,
    WorkflowId,
    WorkflowLimits,
    WorkflowOutcomeRef,
    admit_static_workflow,
)
from tests.graph_sdk import test_qualification_map_conformance as fixture
from tests.graph_sdk.test_dynamic_executor import _operation, _outcome


@pytest.mark.parametrize(
    "case",
    [case for case in fixture.CORPUS if case["case_id"].startswith("map_item_admission/")],
    ids=lambda case: case["case_id"],
)
def test_map_requirement_admission_matches_reference(case: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    async def forbidden_callback(self: object, request: object) -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("An admission-negative workflow dispatched a callback")

    def admit(**kwargs: Any):
        nodes = {node.operation.name: node.id for node in kwargs["nodes"]}
        nodes["OTHER"] = NodeId.new(workflow=kwargs["workflow"])
        if "UNRELATED" in case["declaration"]["node_kinds"]:
            unrelated = NodeId.new(workflow=kwargs["workflow"])
            owner = WorkflowId.new()
            child = NodeId.new(workflow=owner)
            operation = _operation("unrelated", (_outcome("ok"),))
            body = admit_static_workflow(
                workflow=owner,
                interface=operation,
                nodes=(OperationNode(id=child, operation=operation),),
                input_bindings=(),
                output_bindings=(),
                outcome_bindings=(
                    OutcomeBinding(
                        source=NodeOutcomeRef(node=child, outcome="ok"), destination=WorkflowOutcomeRef(outcome="ok")
                    ),
                ),
                sequence=(),
                choices=(),
                protection=(),
                limits=WorkflowLimits(
                    max_nodes=1,
                    max_bindings=1,
                    max_sequence_edges=0,
                    max_choices=0,
                    max_branch_members=0,
                    max_subgraph_depth=1,
                    max_choice_states=1,
                ),
            )
            kwargs["nodes"] += (SubgraphNode(id=unrelated, operation=operation, body=body),)
            kwargs["limits"] = replace(
                kwargs["limits"],
                max_nodes=kwargs["limits"].max_nodes + 2,
                max_bindings=kwargs["limits"].max_bindings + 1,
                max_subgraph_depth=2,
            )
            nodes["UNRELATED"] = unrelated

        def endpoint(raw: dict[str, Any]) -> MapItemPort:
            return MapItemPort(
                path=tuple(nodes[name] for name in raw["path"]),
                expander=nodes[raw["expander"]],
                member=nodes[raw["member"]],
                item_input=raw["item_input"],
                membership_port=raw["membership_port"],
                expansion_outcome=raw["expansion_outcome"],
            )

        requirements = tuple(
            ProtectionRequirement(
                outcome="ok",
                meaning=raw["meaning"],
                subject_port=endpoint(raw["subject_endpoint"]),
                consumed_ports=frozenset(endpoint(value) for value in raw["consumed_endpoints"]),
                coverage=frozenset(CoverageAtom(kind="field", name=name) for name in raw["coverage"]),
                candidate_port=raw["candidate_port"],
            )
            for raw in case["declaration"]["map_item_requirements"]
        )
        kwargs["protection"] = (kwargs["protection"][0], *requirements)
        endpoints = {
            port
            for requirement in kwargs["protection"]
            for port in (requirement.subject_port, *requirement.consumed_ports)
            if isinstance(port, MapItemPort)
        } | {
            port
            for outcome in kwargs["interface"].outcomes
            for promise in outcome.evidence
            for port in (promise.subject_port, *promise.consumed_ports)
            if isinstance(port, MapItemPort)
        }
        binding_count = sum(len(kwargs[name]) for name in ("input_bindings", "output_bindings", "outcome_bindings"))
        kwargs["limits"] = replace(
            kwargs["limits"], max_bindings=binding_count + len(endpoints) + int("UNRELATED" in nodes)
        )
        return admit_static_workflow(**kwargs)

    monkeypatch.setattr(fixture, "admit_static_workflow", admit)
    monkeypatch.setattr(fixture._ReferenceMapCallback, "run", forbidden_callback)
    with pytest.raises(ContractViolation) as rejected:
        asyncio.run(fixture._execute_reference_map(1))
    assert calls == 0
    assert {"status": "rejected", "code": rejected.value.code.value} == case["expected"]
