# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded static history comparison; dynamic/subgraph atoms are not translated.

Expected histories are independently frozen. Product quotient/choice helpers
supply actual histories; public admission mutants live in test_workflow_analysis.
This adapter does not claim every neutral history is a valid public interface.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import pytest

from anonymizer.engine.graph_sdk._workflow_composition import _outcome_options, _selected_nodes
from anonymizer.graph import (
    ArtifactType,
    ChoiceBranch,
    ChoiceDecl,
    ContextUse,
    EvidencePromise,
    InputPort,
    ModelRequirement,
    NodeId,
    NodeOutcomeRef,
    OperationNode,
    OperationSpec,
    OutcomeBinding,
    OutcomeSpec,
    OutputDependency,
    OutputPort,
    ResourceCeiling,
    WorkflowId,
    WorkflowOutcomeRef,
)

_REFERENCE = Path(__file__).parent / "reference" / "compiler_v1"
_RAW = (_REFERENCE / "cases.json").read_bytes()
_ROWS: list[dict[str, Any]] = json.loads(_RAW)
# The SDK does not model state effects, so the reference's state-only neighbors are out of scope.
_APPLICABLE = [
    row
    for row in _ROWS
    if not row["case"]["name"].startswith(("neighbor-dynamic-", "neighbor-subgraph-", "neighbor-state-"))
]
_TEXT = ArtifactType(name="text", revision=1)


def _outcome(value: dict[str, Any]) -> OutcomeSpec:
    promises = frozenset(
        EvidencePromise(
            name=f"{kind}:{name}",
            meaning=kind,
            subject_port="in",
            consumed_ports=frozenset({"in"}),
            coverage=frozenset(),
        )
        for kind in ("labels", "evidence")
        for name in value[kind]
    )
    return OutcomeSpec(
        name=value["name"],
        category="success",
        produced_ports=frozenset(value["ports"]),
        context=frozenset(ContextUse(port="in", meaning=name, capture="whole_artifact") for name in value["context"]),
        evidence=promises,
        model_requirements=frozenset[ModelRequirement](),
        ceiling=ResourceCeiling(
            max_activations=value["cost"], max_model_requests=0, max_input_bytes=0, max_output_bytes=0
        ),
    )


def _operation(name: str, alternatives: list[dict[str, Any]]) -> OperationSpec:
    ports = sorted({port for outcome in alternatives for port in outcome["ports"]})
    return OperationSpec(
        name=name,
        inputs=(InputPort(name="in", artifact_type=_TEXT),),
        outputs=tuple(OutputPort(name=port, artifact_type=_TEXT) for port in ports),
        output_dependencies=tuple(
            OutputDependency(output=port, inputs=frozenset({"in"}), identity_input="in") for port in ports
        ),
        outcomes=tuple(_outcome(outcome) for outcome in alternatives),
    )


def _actual(case: dict[str, Any]) -> set[str]:
    owner = WorkflowId.new()
    nodes = tuple(
        OperationNode(id=NodeId.new(workflow=owner), operation=_operation(str(index), alternatives))
        for index, alternatives in enumerate(case["nodes"])
    )
    selected_names = frozenset(outcome["name"] for outcome in case["nodes"][0] if 1 in outcome["selected"])
    choices = (
        ChoiceDecl(
            selector=nodes[0].id,
            branches=(ChoiceBranch(outcomes=selected_names, members=frozenset({nodes[1].id})),),
        ),
    )
    mappings = tuple(
        OutcomeBinding(
            source=NodeOutcomeRef(node=nodes[2].id, outcome=outcome["name"]),
            destination=WorkflowOutcomeRef(outcome=outcome["interface"]),
        )
        for outcome in case["nodes"][2]
    )
    # Only its outcomes' dynamic endpoint declarations are consulted by quotient.
    interface = nodes[2].operation
    options = _outcome_options(nodes, mappings, choices, interface)
    observations: set[str] = set()
    for outcomes in itertools.product(*options):
        assignment = {node.id: outcome for node, outcome in zip(nodes, outcomes, strict=True)}
        chosen = _selected_nodes(frozenset(assignment), choices, assignment)
        selected = [index for index, node in enumerate(nodes) if node.id in chosen]
        row: dict[str, Any] = {
            "selected": selected,
            "cost": sum(outcomes[index].ceiling.max_activations for index in selected),
        }
        for field in ("ports", "labels", "context", "evidence", "dynamic", "subgraph"):
            values = []
            for index in selected:
                outcome = outcomes[index]
                if field == "ports":
                    actual = sorted(outcome.produced_ports)
                elif field == "context":
                    actual = sorted(item.meaning for item in outcome.context)
                elif field in {"labels", "evidence"}:
                    actual = sorted(item.name.split(":", 1)[1] for item in outcome.evidence if item.meaning == field)
                else:
                    actual = []
                values.append([index, actual])
            row[field] = values
        row["interface"] = next(
            item.destination.outcome for item in mappings if item.source.outcome == outcomes[2].name
        )
        observations.add(json.dumps(row, sort_keys=True))
    return observations


def _without_state(rows: list[dict[str, Any]]) -> set[str]:
    return {json.dumps({key: value for key, value in item.items() if key != "state"}, sort_keys=True) for item in rows}


@pytest.mark.parametrize("row", _APPLICABLE, ids=lambda row: row["case"]["name"])
def test_static_product_histories_match_independent_finite_cases(row: dict[str, Any]) -> None:
    actual = _actual(row["case"])
    assert actual == _without_state(row["expected"])
    assert actual != _without_state(row["rejected_summary"])


def test_reference_bytes_and_explicit_translation_scope() -> None:
    assert hashlib.sha256(_RAW).hexdigest() == "95869c546886e886c37c3bc80fa7375a46c4ea0e96eb10d52f25d8b8d167a585"
    assert hashlib.sha256((_REFERENCE / "compiler_reference.py").read_bytes()).hexdigest() == (
        "30b145d46180395fa786eaba5e3d2d5c05f9f116255b8121f7e77a8557528e70"
    )
    assert len(_ROWS) == 36 and len(_APPLICABLE) == 27
    assert sum(row["raw_assignment_count"] for row in _APPLICABLE) == 179
    assert all(
        not value["dynamic"] and not value["subgraph"]
        for row in _APPLICABLE
        for node in row["case"]["nodes"]
        for value in node
    )
