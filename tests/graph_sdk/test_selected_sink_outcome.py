# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Workflow and subgraph outcomes come from the selected-path sink, independent of hash order.

Admission defines a scope's outcome as the binding of its unique selected sink's
outcome. Publication, the activation reducer and qualification provenance must
agree with that definition, and a failing sink must never be released as success.
Bound outcome sets are frozensets, so each scenario runs under several hash seeds.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectRejected
from anonymizer.engine.graph_sdk.context import admit_context_plan
from anonymizer.engine.graph_sdk.evidence import admit_qualification, evidence_revision_view
from anonymizer.engine.graph_sdk.executor import (
    AssessmentLimits,
    ExecutionLimits,
    ExecutionResult,
    ExecutionServices,
    ImplementationHandle,
    LocalCompleted,
    LocalFailure,
    OperationExecutionPolicy,
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.preparation import PreparationConfiguration, prepare
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.engine.graph_sdk.requests import (
    AssociationInput,
    AssociationResult,
    PortArtifact,
    TextArtifactValue,
)
from anonymizer.graph import (
    ActivationLimits,
    ArtifactType,
    Graph,
    NodeId,
    NodeOutcomeRef,
    NodeOutputRef,
    OperationNode,
    OperationSpec,
    OutcomeBinding,
    OutcomeSpec,
    OutputBinding,
    OutputDependency,
    OutputPort,
    ResourceCeiling,
    SequenceEdge,
    SubgraphNode,
    WorkflowId,
    WorkflowLimits,
    WorkflowOutcomeRef,
    WorkflowOutputRef,
    compile_graph,
)
from tests.graph_sdk.implementation_fixtures import _execution_implementation, _selection
from tests.graph_sdk.test_dynamic_executor import _capability, _rows
from tests.graph_sdk.test_evidence import _qualification_limits
from tests.graph_sdk.test_local_executor import _Clock
from tests.graph_sdk.test_preparation import _data, _limits

_TEXT = ArtifactType(name="text", revision=1)
_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)
_ROOT = Path(__file__).resolve().parents[2]


def _outcome(name: str, activations: int = 1, produced: frozenset[str] = frozenset()) -> OutcomeSpec:
    return OutcomeSpec(
        name=name,
        category="success",
        produced_ports=produced,
        context=frozenset(),
        evidence=frozenset(),
        model_requirements=frozenset(),
        ceiling=ResourceCeiling(
            max_activations=activations,
            max_model_requests=0,
            max_input_bytes=0,
            max_output_bytes=100 * activations,
        ),
    )


def _operation(name: str, outcomes: tuple[OutcomeSpec, ...], outputs: tuple[str, ...] = ()) -> OperationSpec:
    return OperationSpec(
        name=name,
        inputs=(),
        outputs=tuple(OutputPort(name=port, artifact_type=_TEXT) for port in outputs),
        output_dependencies=tuple(
            OutputDependency(output=port, inputs=frozenset(), identity_input=None) for port in outputs
        ),
        outcomes=outcomes,
    )


def _limits_for(depth: int) -> WorkflowLimits:
    return WorkflowLimits(
        max_nodes=4,
        max_bindings=5,
        max_sequence_edges=2,
        max_choices=0,
        max_branch_members=0,
        max_subgraph_depth=depth,
        max_choice_states=1,
    )


def _bound(node: NodeId, outcome: str, destination: str) -> OutcomeBinding:
    return OutcomeBinding(
        source=NodeOutcomeRef(node=node, outcome=outcome), destination=WorkflowOutcomeRef(outcome=destination)
    )


_Scenario = tuple[Graph, dict[NodeId, OperationSpec], dict[NodeId, "_Complete | _Fail"]]


def _sink(succeeds: bool, outcome: str, ports: tuple[str, ...] = ()) -> _Complete | _Fail:
    return _Complete(outcome, ports) if succeeds else _Fail()


def _root_scenario(succeeds: bool) -> _Scenario:
    """A -> B where A's x binds "first" and B's ok2 binds "second"; B is the sink."""
    workflow = WorkflowId.new()
    a, b = NodeId.new(workflow=workflow), NodeId.new(workflow=workflow)
    operations = {
        a: _operation("a", (_outcome("x", produced=frozenset({"u"})),), ("u",)),
        b: _operation("b", (_outcome("ok1"), _outcome("ok2"))),
    }
    interface = _operation(
        "interface", (_outcome("first", 2, frozenset({"out"})), _outcome("second", 2, frozenset({"out"}))), ("out",)
    )
    graph = compile_graph(
        workflow=workflow,
        interface=interface,
        nodes=tuple(OperationNode(id=node, operation=operation) for node, operation in operations.items()),
        input_bindings=(),
        output_bindings=(
            OutputBinding(source=NodeOutputRef(node=a, port="u"), destination=WorkflowOutputRef(port="out")),
        ),
        outcome_bindings=(_bound(a, "x", "first"), _bound(b, "ok1", "first"), _bound(b, "ok2", "second")),
        sequence=(SequenceEdge(before=a, after=b),),
        protection=(),
        limits=_limits_for(1),
    )
    return graph, operations, {a: _Complete("x", ("u",)), b: _sink(succeeds, "ok2")}


def _chain_scenario(succeeds: bool) -> _Scenario:
    """A -> B -> C where A binds "first", B binds "second" and the sink C's z binds "first"."""
    workflow = WorkflowId.new()
    a, b, c = (NodeId.new(workflow=workflow) for _ in range(3))
    operations = {
        a: _operation("a", (_outcome("x", produced=frozenset({"u"})),), ("u",)),
        b: _operation("b", (_outcome("y"),)),
        c: _operation("c", (_outcome("z"), _outcome("w"))),
    }
    interface = _operation(
        "interface", (_outcome("first", 3, frozenset({"out"})), _outcome("second", 3, frozenset({"out"}))), ("out",)
    )
    graph = compile_graph(
        workflow=workflow,
        interface=interface,
        nodes=tuple(OperationNode(id=node, operation=operation) for node, operation in operations.items()),
        input_bindings=(),
        output_bindings=(
            OutputBinding(source=NodeOutputRef(node=a, port="u"), destination=WorkflowOutputRef(port="out")),
        ),
        outcome_bindings=(
            _bound(a, "x", "first"),
            _bound(b, "y", "second"),
            _bound(c, "z", "first"),
            _bound(c, "w", "second"),
        ),
        sequence=(SequenceEdge(before=a, after=b), SequenceEdge(before=b, after=c)),
        protection=(),
        limits=_limits_for(1),
    )
    return graph, operations, {a: _Complete("x", ("u",)), b: _Complete("y", ()), c: _sink(succeeds, "z")}


def _subgraph_scenario(succeeds: bool) -> _Scenario:
    """A root subgraph whose body A -> B binds A to "done" and B's ok2 to "other"; B is the body sink."""
    body_workflow, root_workflow = WorkflowId.new(), WorkflowId.new()
    a, b = NodeId.new(workflow=body_workflow), NodeId.new(workflow=body_workflow)
    container = NodeId.new(workflow=root_workflow)
    operations = {
        a: _operation("a", (_outcome("x"),)),
        b: _operation("b", (_outcome("ok1"), _outcome("ok2", produced=frozenset({"v"}))), ("v",)),
    }
    interface = _operation("interface", (_outcome("done", 2), _outcome("other", 2, frozenset({"out"}))), ("out",))
    body = compile_graph(
        workflow=body_workflow,
        interface=interface,
        nodes=tuple(OperationNode(id=node, operation=operation) for node, operation in operations.items()),
        input_bindings=(),
        output_bindings=(
            OutputBinding(source=NodeOutputRef(node=b, port="v"), destination=WorkflowOutputRef(port="out")),
        ),
        outcome_bindings=(_bound(a, "x", "done"), _bound(b, "ok1", "done"), _bound(b, "ok2", "other")),
        sequence=(SequenceEdge(before=a, after=b),),
        protection=(),
        limits=_limits_for(1),
    )
    graph = compile_graph(
        workflow=root_workflow,
        interface=_operation("root", (_outcome("done", 3), _outcome("other", 3, frozenset({"out"}))), ("out",)),
        nodes=(SubgraphNode(id=container, operation=interface, body=body),),
        input_bindings=(),
        output_bindings=(
            OutputBinding(source=NodeOutputRef(node=container, port="out"), destination=WorkflowOutputRef(port="out")),
        ),
        outcome_bindings=(_bound(container, "done", "done"), _bound(container, "other", "other")),
        sequence=(),
        protection=(),
        limits=_limits_for(2),
    )
    return graph, operations, {a: _Complete("x", ()), b: _sink(succeeds, "ok2", ("v",))}


_SCENARIOS = {"root": _root_scenario, "chain": _chain_scenario, "subgraph": _subgraph_scenario}


class _Complete:
    def __init__(self, outcome: str, ports: tuple[str, ...]) -> None:
        self.outcome, self.ports = outcome, ports

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted:
        association = request[0].association
        outputs = tuple(
            PortArtifact(port=port, artifact_type=_TEXT, artifact=None, value=TextArtifactValue(text="v"))
            for port in self.ports
        )
        return LocalCompleted(
            results=(
                AssociationResult(
                    association=association, outcome=self.outcome, outputs=outputs, consumed_context_ports=frozenset()
                ),
            )
        )


class _Fail:
    async def run(self, request: tuple[AssociationInput, ...]) -> LocalFailure:
        del request
        return LocalFailure(failure="implementation_exception")


def _report(scenario: str) -> dict[str, object]:
    """Execute one scenario and summarize its terminals, publication and qualification."""
    shape, sink = scenario.split("_")
    graph, operations, callbacks = _SCENARIOS[shape](sink == "success")
    capabilities = tuple(_capability(operation, index) for index, operation in enumerate(operations.values()))
    by_node = dict(zip(operations, capabilities, strict=True))
    prepared = prepare(
        data=_data(1),
        workflow=graph,
        activation_limits=ActivationLimits(max_events=44, max_entries=13, max_parent_depth=4),
        bound_inputs=(),
        configuration=PreparationConfiguration(
            purpose="execution_only", required_protection_outcomes=frozenset(), hard_request_limit=None
        ),
        selections=tuple(_selection(node, by_node[node]) for node in operations),
        capabilities=capabilities,
        limits=_limits(capabilities=5, slots=13),
    )
    admitted = admit_execution_plan(
        context=admit_context_plan(prepared=prepared, bound_context=None),
        capabilities=capabilities,
        policies=tuple(
            OperationExecutionPolicy(
                node=node,
                kind="local",
                request=None,
                safe_detachment="forbidden",
                implementations=(_execution_implementation(by_node[node], request=None),),
                result_outcomes=frozenset(item.name for item in operation.outcomes),
                runtime_outcomes=_rows(frozenset(item.name for item in operation.outcomes)),
            )
            for node, operation in operations.items()
        ),
        assessment_productions=(),
        assessment_limits=AssessmentLimits(
            max_productions=0,
            max_findings_per_production=0,
            max_finding_code_bytes=0,
            max_assessment_facts=0,
            max_port_facts=20,
            max_provenance_edges=10,
        ),
        map_expansions=(),
    )
    handles = tuple(
        ImplementationHandle(
            implementation=by_node[node].implementation,
            operation=operation,
            configuration=by_node[node].configuration,
            local=callbacks[node],
            transport=None,
            resource=None,
        )
        for node, operation in operations.items()
    )
    services = ExecutionServices(
        handles=handles,
        limits=ExecutionLimits(
            max_local_in_flight=4,
            max_remote_outstanding=0,
            max_runtime_artifacts=8,
            max_runtime_artifact_bytes=300,
            max_collection_items=2,
        ),
        clock=_Clock(),
    )

    async def run() -> ExecutionResult:
        return await (await start_execution(admitted=admitted, capabilities=capabilities, services=services)).wait()

    result = asyncio.run(run())
    roots = sorted(
        (entry.status, entry.outcome) for entry in result.states[0].entries if entry.activation.parent is None
    )
    report: dict[str, object] = {
        "roots": [list(item) for item in roots],
        "final": sorted([item.outcome, item.port] for item in result.final_outputs),
    }
    qualification = admit_qualification(
        execution=admitted, productions=(), limits=_qualification_limits(max_port_facts=32, max_provenance_edges=32)
    )
    current = evidence_revision_view(
        admitted=qualification,
        result=result,
        artifacts=tuple({record.key: record for record, _ in result.artifacts}.values()),
        configurations=(),
    )
    try:
        qualified = qualify(admitted=qualification, result=result, current=current, submissions=())
    except EffectRejected as error:
        report["qualify"] = f"rejected:{error.code.value}"
    else:
        report["qualify"] = sorted(sorted(target.withholding) for target in qualified.targets)
    return report


def _run_with_seed(scenario: str, seed: int) -> dict[str, object]:
    code = f"import json; from {__name__} import _report; print(json.dumps(_report({scenario!r})))"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_ROOT,
        env={**os.environ, "PYTHONHASHSEED": str(seed)},
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        # DF-12: the sink B's ok2 binds "second"; A's binding is not the workflow outcome.
        (
            "root_success",
            {
                "roots": [["success", "ok2"], ["success", "x"]],
                "final": [["second", "out"]],
                "qualify": [["execution_only"]],
            },
        ),
        # DF-12: the sink failed, so the workflow has no success outcome. The
        # unanimous candidate label stays inspectable but is withheld.
        (
            "root_failure",
            {
                "roots": [["failure", None], ["success", "x"]],
                "final": [["first", "out"]],
                "qualify": [["execution_only", "terminal_failure"]],
            },
        ),
        # DF-12: the sink C binds "first"; B's "second" is not the workflow outcome.
        (
            "chain_success",
            {
                "roots": [["success", "x"], ["success", "y"], ["success", "z"]],
                "final": [["first", "out"]],
                "qualify": [["execution_only"]],
            },
        ),
        # DF-12: with the sink failed, A's "first" and B's "second" disagree, so
        # no outcome is guessed from bind order and nothing is published.
        (
            "chain_failure",
            {
                "roots": [["failure", None], ["success", "x"], ["success", "y"]],
                "final": [],
                "qualify": [["execution_only", "terminal_failure"]],
            },
        ),
        # DF-13: the container takes the body sink B's binding "other", not A's "done".
        (
            "subgraph_success",
            {
                "roots": [["success", "other"]],
                "final": [["other", "out"]],
                "qualify": [["execution_only"]],
            },
        ),
        # DF-13: a failing body sink fails the container; no "done" success is published.
        (
            "subgraph_failure",
            {
                "roots": [["failure", None]],
                "final": [],
                "qualify": [["execution_only", "terminal_failure"]],
            },
        ),
    ],
)
def test_outcome_follows_selected_sink_for_every_hash_seed(scenario: str, expected: dict[str, object]) -> None:
    reports = {seed: _run_with_seed(scenario, seed) for seed in _SEEDS}
    assert reports == {seed: expected for seed in _SEEDS}
