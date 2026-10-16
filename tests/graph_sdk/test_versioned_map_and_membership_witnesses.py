# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual versioned map ancestry without requiring every item assessment."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from anonymizer.engine.graph_sdk.evidence import (
    AdmittedQualification,
    AssessmentSubmission,
    EvidenceRevisionView,
    admit_qualification,
    evidence_revision_view,
)
from anonymizer.engine.graph_sdk.executor import AdmittedExecutionPlan, ExecutionResult, MapItemKey, start_execution
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.graph import EffectRejected
from anonymizer.graph._values import ActivationKey, DatumId, InvocationId
from anonymizer.graph.workflow import NodeId
from tests.graph_sdk import qualification_map_execution as fixture
from tests.graph_sdk.evidence_fixtures import _qualification_limits
from tests.graph_sdk.workflow_construction import AdmittedWorkflow, admit_static_workflow


class _ExecutedGraph(Exception):
    def __init__(self, execution: AdmittedExecutionPlan, result: ExecutionResult) -> None:
        self.execution = execution
        self.result = result


def _root_inputs(
    execution: AdmittedExecutionPlan, result: ExecutionResult
) -> tuple[AdmittedQualification, EvidenceRevisionView, tuple[AssessmentSubmission, ...]]:
    productions = tuple(row for row in execution.assessment_productions if row.promise == "P")
    admitted = admit_qualification(execution=execution, productions=productions, limits=_qualification_limits())
    refs = {ref.key: ref for ref, _ in sorted(result.artifacts, key=lambda row: row[0].version)}
    current = evidence_revision_view(
        admitted=admitted,
        result=result,
        artifacts=tuple(refs.values()),
        configurations=tuple({(fact.node, fact.environment.configuration) for fact in result.assessments}),
    )
    submissions = tuple(AssessmentSubmission(fact=fact) for fact in result.assessments if fact.promise == "P")
    return admitted, current, submissions


def _execute_root_protected_map(
    monkeypatch: pytest.MonkeyPatch,
    *,
    count: int,
    failed: bool = False,
    versioned: bool = False,
    blocked_member: bool = False,
) -> tuple[AdmittedExecutionPlan, ExecutionResult, dict[str, NodeId]]:
    """Reuse construction, but validate this declared graph's own execution shape."""

    def admit(**kwargs: Any) -> AdmittedWorkflow:
        kwargs["protection"] = tuple(row for row in kwargs["protection"] if row.meaning == "privacy")
        if failed or blocked_member:
            root = next(node.id for node in kwargs["nodes"] if node.operation.name == "N")
            kwargs["sequence"] = tuple(edge for edge in kwargs["sequence"] if edge.after != root)
        return admit_static_workflow(**kwargs)

    async def execute(**kwargs: Any) -> Any:
        running = await start_execution(**kwargs)
        result = await running.wait()
        # Return at the real executor boundary: the reused builder's trailing
        # fixed callback-count assertion describes its original sequence graph.
        raise _ExecutedGraph(kwargs["admitted"], result)

    monkeypatch.setattr(fixture, "admit_static_workflow", admit)
    monkeypatch.setattr(fixture, "start_execution", execute)
    try:
        asyncio.run(
            fixture._execute_reference_map(
                count,
                expander_failure=failed,
                versioned_items=versioned,
                candidate_uses_membership=not (failed or blocked_member),
                blocked_member=blocked_member,
            )
        )
    except _ExecutedGraph as captured:
        execution, result = captured.execution, captured.result
    else:
        raise AssertionError("Actual executor boundary was not reached")
    nodes = {node.operation.name: node.id for node in execution.context.prepared.workflow.operations}
    assert sum(entry.template == nodes["N"] and entry.status == "success" for entry in result.states[0].entries) == 1
    assert len(result.record.terminals) == count + 3 + int(blocked_member)
    return execution, result, nodes


@pytest.mark.parametrize("corruption", [None, "target", "expander"])
def test_same_item_key_versions_retain_exact_owner_ancestry(
    corruption: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    execution, result, _ = _execute_root_protected_map(monkeypatch, count=2, versioned=True)
    admitted, current, submissions = _root_inputs(execution, result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(output.qualified) == 1 and output.record.statuses[0].qualification == "met"
    items = [fact for fact in result.provenance if isinstance(fact.key, MapItemKey)]
    assert len(items) == 2
    assert {fact.key.item_key for fact in items if isinstance(fact.key, MapItemKey)} == {0}
    assert {fact.key.item_version for fact in items if isinstance(fact.key, MapItemKey)} == {1, 2}
    assert len({fact.artifact.key for fact in items}) == 1
    assert {fact.artifact.version for fact in items} == {1, 2}
    assert all(fact.parents for fact in items)
    if corruption is None:
        assert len(result.assessments) == 3 and len(output.verified) == 1
        assert len(output.qualified[0].evidence) == 1
        return
    selected = next(fact for fact in items if fact.artifact.version == 2)
    old = selected.key
    assert isinstance(old, MapItemKey)
    if corruption == "invocation":
        object.__setattr__(
            result,
            "artifacts",
            tuple(
                (
                    replace(ref, invocation=InvocationId.new(plan=ref.invocation.plan))
                    if ref == selected.artifact
                    else ref,
                    value,
                )
                for ref, value in result.artifacts
            ),
        )
    else:
        changed = replace(old)
        if corruption == "target":
            object.__setattr__(changed, "target", DatumId.new(graph=old.target.graph))
        else:
            other = ActivationKey(invocation=old.expander.invocation, occurrence=999, parent=None, iteration=None)
            object.__setattr__(changed, "expander", other)
        for fact in result.provenance:
            if fact.key == old:
                object.__setattr__(fact, "key", changed)
            object.__setattr__(fact, "parents", frozenset(changed if key == old else key for key in fact.parents))
        object.__setattr__(
            result,
            "_input_parents",
            tuple((t, a, p, changed if key == old else key) for t, a, p, key in result._input_parents),
        )
    with pytest.raises(EffectRejected) as error:
        qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert error.value.code.value == ("missing" if corruption == "expander" else "foreign_owner")


@pytest.mark.parametrize("category", ["failure", "overflow"])
def test_closed_failed_expansion_terminal_categories_withhold_root_success(
    category: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    execution, result, nodes = _execute_root_protected_map(monkeypatch, count=0, failed=True)
    admitted, current, submissions = _root_inputs(execution, result)
    state = result.states[0]
    expander = next(entry for entry in state.entries if entry.template == nodes["EXP"])
    expansion = next(item for item in state.expansions if item.parent == expander.activation)
    terminal = next(item for item in result.record.terminals if item.activation == expander.activation)
    assert expansion.status == "failed" and not expansion.members
    assert expander.status == terminal.category == "failure"
    assert result.final_outputs and result.assessments
    status = "failure" if category == "overflow" else category
    if category == "overflow":
        object.__setattr__(expansion, "status", "overflow")
    object.__setattr__(
        state,
        "entries",
        frozenset(
            replace(entry, status=status, outcome=None) if entry is expander else entry for entry in state.entries
        ),
    )
    object.__setattr__(terminal, "category", status)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert not output.qualified
    assert output.record.statuses[0].completion == "closed"
    assert output.record.statuses[0].qualification == "unmet"
    assert output.targets[0].withholding == frozenset({"terminal_failure"})


@pytest.mark.parametrize("corruption", ["open", "wrong-outcome"])
def test_empty_expansion_requires_closed_matching_outcome(corruption: str, monkeypatch: pytest.MonkeyPatch) -> None:
    execution, result, _ = _execute_root_protected_map(monkeypatch, count=0)
    admitted, current, submissions = _root_inputs(execution, result)
    assert len(qualify(admitted=admitted, result=result, current=current, submissions=submissions).qualified) == 1
    (expansion,) = result.states[0].expansions
    assert expansion.status == "closed" and not expansion.members
    if corruption == "open":
        object.__setattr__(expansion, "status", "pending")
        membership = next(row for row in result.record.memberships if row.parent == expansion.parent)
        object.__setattr__(membership, "closed", False)
    else:
        mapping = next(mapping for scope in execution.context.prepared.workflow.bodies() for mapping in scope.maps)
        object.__setattr__(mapping, "expansion_outcomes", frozenset({"other"}))
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert not output.qualified
    assert output.record.statuses[0].completion == "pending"
    assert output.record.statuses[0].qualification == "unmet"
    assert output.targets[0].withholding == frozenset({"incomplete_membership"})
