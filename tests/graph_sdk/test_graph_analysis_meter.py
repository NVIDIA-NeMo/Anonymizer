# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public compiler accounting for unified graph declaration projections."""

from __future__ import annotations

import pytest

from anonymizer import graph as w
from anonymizer.engine.graph_sdk._workflow_analysis import _ANALYSIS, _AnalysisMeter
from anonymizer.graph import ValidationCode


def _operation(bound: int = 1) -> w.OperationSpec:
    return w.OperationSpec(
        name="local",
        inputs=(),
        outputs=(),
        output_dependencies=(),
        outcomes=(
            w.OutcomeSpec(
                name="ok",
                category="success",
                produced_ports=frozenset(),
                context=frozenset(),
                evidence=frozenset(),
                model_requirements=frozenset(),
                ceiling=w.ResourceCeiling(
                    max_activations=bound,
                    max_model_requests=0,
                    max_input_bytes=None,
                    max_output_bytes=None,
                ),
            ),
        ),
    )


def _compile_chain(limit: int = 1_000_000) -> w.Graph:
    owner = w.WorkflowId.new()
    operation = _operation()
    nodes = tuple(w.OperationNode(id=w.NodeId.new(workflow=owner), operation=operation) for _ in range(8))
    return w.compile_graph(
        workflow=owner,
        interface=_operation(8),
        nodes=nodes,
        input_bindings=(),
        output_bindings=(),
        outcome_bindings=(
            w.OutcomeBinding(
                source=w.NodeOutcomeRef(node=nodes[-1].id, outcome="ok"),
                destination=w.WorkflowOutcomeRef(outcome="ok"),
            ),
        ),
        sequence=tuple(w.SequenceEdge(before=a.id, after=b.id) for a, b in zip(nodes, nodes[1:])),
        protection=(),
        limits=w.WorkflowLimits(
            max_nodes=100,
            max_bindings=100,
            max_sequence_edges=100,
            max_choices=10,
            max_branch_members=100,
            max_subgraph_depth=20,
            max_choice_states=1000,
            max_analysis_work=limit,
        ),
    )


def test_public_compilation_charges_graph_scans_and_restores_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[_AnalysisMeter] = []
    tick = _AnalysisMeter.tick

    def record(meter: _AnalysisMeter) -> None:
        if not observed or observed[-1] is not meter:
            observed.append(meter)
        tick(meter)

    monkeypatch.setattr(_AnalysisMeter, "tick", record)
    graph = _compile_chain()
    required = observed[-1].work
    # Measured 1,262 visits at 109bf02f, of which 232 are selector entries and 16 are
    # constructor operation visits. The floor sits 12 below that, so uncharging either
    # group (-232 or -16) fails while small refactors of the admission walk still pass.
    assert required >= 1_250
    assert graph.activation_upper_bound == 8
    assert graph.dynamic_depth == 1
    for limit in (1_250 - 1, required - 1):
        with pytest.raises(w.ContractViolation) as failure:
            _compile_chain(limit)
        assert failure.value.code is ValidationCode.ANALYSIS_LIMIT_EXCEEDED
        assert observed[-1].work == limit
        assert _ANALYSIS.get() is None
    assert _compile_chain(required).activation_upper_bound == 8
    assert observed[-1].work == required
    assert _ANALYSIS.get() is None


@pytest.mark.parametrize("selector", ("operations", "choices", "maps", "loops", "joins"))
def test_graph_selectors_charge_repeated_visits_and_reject_exhaustion(selector: str) -> None:
    graph = _compile_chain()
    expected = getattr(graph, selector)
    meter = _AnalysisMeter(2 * len(graph.nodes))
    token = _ANALYSIS.set(meter)
    try:
        assert getattr(graph, selector) == expected
        assert meter.work == len(graph.nodes)
        assert getattr(graph, selector) == expected
        assert meter.work == 2 * len(graph.nodes)
        with pytest.raises(w.ContractViolation) as failure:
            getattr(graph, selector)
        assert failure.value.code is ValidationCode.ANALYSIS_LIMIT_EXCEEDED
        assert meter.work == meter.limit
    finally:
        _ANALYSIS.reset(token)
    assert getattr(graph, selector) == expected
