# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure planner comparisons; these normalized rows do not claim P7 admission."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected
from anonymizer.engine.graph_sdk.continuity import _survivors
from anonymizer.engine.graph_sdk.data import AtomicGroup, DataGraph, DatumDependency
from anonymizer.engine.graph_sdk.records import DecisionRef
from tests.graph_sdk.continuity_graph_fixture import execute_graph
from tests.graph_sdk.reference.continuity.cases import Case, generated, known
from tests.graph_sdk.reference.continuity.reference import greatest


@pytest.fixture(scope="module")
def graph():
    return asyncio.run(execute_graph("alternating"))


@pytest.mark.parametrize("case", (*generated(), *(case for case, _ in known())), ids=lambda c: c.name)
def test_planner_matches_independent_feasible_subsets(case: Case, graph) -> None:
    targets = dict(graph.targets)
    nodes = dict(graph.decision_nodes)
    decisions = {
        key: DecisionRef(artifact=next(port.artifact for port in graph.result.ports if port.node == node))
        for key, node in nodes.items()
    }
    originals = {output.target: output for output in graph.qualified.qualified}
    bundle = case.bundle
    qualified = tuple(
        replace(originals[targets[t.name]], required_decisions=frozenset(decisions[key] for key in t.required))
        for t in bundle.targets
    )
    existing = graph.execution.context.prepared.data
    data = DataGraph(graph=existing.graph, datums=tuple(existing.datums)).validate(
        targets=tuple(targets[t.name] for t in bundle.targets),
        source_relations=(),
        contexts=(),
        dependencies=tuple(
            DatumDependency(prerequisite=targets[p], dependent=targets[t.name])
            for t in bundle.targets
            for p in t.prerequisites
        ),
        coherence=(),
        atomic=tuple(
            AtomicGroup(members=tuple(targets[name] for name in sorted(t.atomic | {t.name})))
            for t in bundle.targets
            if t.atomic
        ),
        output_regions=(),
        limits=replace(existing.limits, max_declarations=30),
    )
    supports = {
        decisions[d.key]: tuple(
            frozenset(
                targets[t.name]
                for t in bundle.targets
                if d.key in t.required
                and any(
                    e.reference in t.evidence
                    and e.observed_revision == e.current_revision
                    and (e.node, e.outcome, e.promise) == (s.node, s.outcome, s.promise)
                    and s.coverage <= e.coverage
                    and d.key in e.consumed
                    and e.consumed <= t.required
                    for e in bundle.verified
                )
            )
            for s in d.supports
        )
        for d in bundle.decisions
    }
    conflicts = frozenset(decisions[key] for key in case.conflicts)
    result, steps = _survivors(qualified=qualified, data=data, supports=supports, conflicts=conflicts, maximum=100)
    assert result == frozenset(targets[name] for name in greatest(bundle, case.conflicts))
    assert _survivors(qualified=qualified, data=data, supports=supports, conflicts=conflicts, maximum=steps) == (
        result,
        steps,
    )
    if steps:
        with pytest.raises(EffectRejected) as failure:
            _survivors(qualified=qualified, data=data, supports=supports, conflicts=conflicts, maximum=steps - 1)
        assert failure.value.code is EffectCode.LIMIT_EXCEEDED
