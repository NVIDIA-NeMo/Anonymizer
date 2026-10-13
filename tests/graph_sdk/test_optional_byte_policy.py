# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Counts remain hard while explicit size policies survive independent crossings."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from anonymizer.engine.graph_sdk._artifact_shape import ArtifactShape
from anonymizer.engine.graph_sdk._effect_values import EffectRejected
from anonymizer.engine.graph_sdk._ingress import inspect_artifact, inspect_associations
from anonymizer.engine.graph_sdk.executor import IngressLimits, start_execution
from anonymizer.engine.graph_sdk.requests import TextArtifactValue
from anonymizer.graph._values import ContractViolation
from anonymizer.graph.workflow import admit_static_workflow, substitute
from tests.graph_sdk.qualification_fixtures import _inputs
from tests.graph_sdk.test_ingress_bounds import _rows
from tests.graph_sdk.test_workflow_static import _one_node_factory, _zero_operation
from tests.graph_sdk.typed_artifact_fixtures import span_graph


def test_optional_ingress_bytes_preserve_counts_utf8_and_exact_shape() -> None:
    value = TextArtifactValue(text="x" * (16 * 1024 * 1024 + 1))
    assert inspect_artifact(value, IngressLimits())
    assert not inspect_artifact(value, IngressLimits(max_text_bytes=16 * 1024 * 1024))
    object.__setattr__(value, "shape", ArtifactShape(1))
    assert not inspect_artifact(value, IngressLimits())
    assert not inspect_artifact(TextArtifactValue(text="\ud800"), IngressLimits())
    rows = _rows(collection=True)
    expected = frozenset(row.association for row in rows)
    assert inspect_associations(rows, expected, IngressLimits(max_collection_items=1)) == expected
    assert inspect_associations(rows, expected, IngressLimits(max_associations=1)) is None


@pytest.mark.parametrize("cap,success", [(None, True), (98, True), (97, False)])
def test_retained_byte_choice_preserves_exact_finite_boundary(cap: int | None, success: bool) -> None:
    async def run():
        execution, capabilities, services, _ = span_graph()
        services = replace(services, limits=replace(services.limits, max_runtime_artifact_bytes=cap))
        result = await (await start_execution(admitted=execution, capabilities=capabilities, services=services)).wait()
        _inputs(execution, result)
        return result

    result = asyncio.run(run())
    assert bool(result.final_outputs) is success
    assert sum(value.shape.content_bytes for _, value in result.artifacts) <= (98 if cap is None else cap)


@pytest.mark.parametrize("fault", ["mutation", "replacement", "missing_snapshot"])
def test_issued_ingress_policy_cannot_be_widened_before_execution(fault: str) -> None:
    async def run():
        execution, capabilities, services, _ = span_graph(ingress=IngressLimits(max_text_bytes=10))
        if fault == "mutation":
            object.__setattr__(execution.ingress_limits, "max_text_bytes", None)
        elif fault == "replacement":
            object.__setattr__(execution, "ingress_limits", replace(execution.ingress_limits, max_text_bytes=None))
        else:
            object.__delattr__(execution, "_ingress_policy")
        with pytest.raises(EffectRejected):
            await start_execution(admitted=execution, capabilities=capabilities, services=services)

    asyncio.run(run())


@pytest.mark.parametrize("policy", ["ingress", "retention"])
def test_callback_cannot_widen_issued_ingress_or_selected_retention(monkeypatch, policy) -> None:
    async def run():
        ingress = IngressLimits(max_text_bytes=10 if policy == "ingress" else None)
        execution, capabilities, services, _ = span_graph(ingress=ingress)
        original = services.handles[0].local
        assert original is not None
        run_original = original.run
        object.__setattr__(services.limits, "max_runtime_artifact_bytes", 14 if policy == "retention" else None)

        async def corrupt(request):
            object.__setattr__(ingress, "max_text_bytes", None)
            object.__setattr__(services.limits, "max_runtime_artifact_bytes", None)
            return await run_original(request)

        monkeypatch.setattr(original, "run", corrupt)
        result = await (await start_execution(admitted=execution, capabilities=capabilities, services=services)).wait()
        assert not result.final_outputs
        assert sum(value.shape.content_bytes for _, value in result.artifacts) == 14
        if policy == "ingress":
            with pytest.raises(EffectRejected):
                _inputs(execution, result)
        else:
            _inputs(execution, result)

    asyncio.run(run())


def test_equivalent_ingress_copy_and_narrower_explicit_retention_remain_valid() -> None:
    async def run():
        execution, capabilities, services, _ = span_graph(ingress=IngressLimits(max_text_bytes=100))
        object.__setattr__(execution, "ingress_limits", replace(execution.ingress_limits))
        services = replace(services, limits=replace(services.limits, max_runtime_artifact_bytes=98))
        result = await (await start_execution(admitted=execution, capabilities=capabilities, services=services)).wait()
        assert result.final_outputs
        _inputs(execution, result)

    asyncio.run(run())


@pytest.mark.parametrize(
    "parent,child,accepted", [(None, None, True), (None, 2, True), (2, None, False), (2, 3, False)]
)
def test_substitution_cannot_turn_finite_byte_promise_into_no_ceiling(parent, child, accepted) -> None:
    def operation(bound):
        original = _zero_operation()
        outcome = original.outcomes[0]
        return replace(original, outcomes=(replace(outcome, ceiling=replace(outcome.ceiling, max_output_bytes=bound)),))

    target = _one_node_factory(lambda: operation(parent))
    target = admit_static_workflow(
        workflow=target.workflow,
        interface=target.interface,
        nodes=tuple(target.nodes),
        input_bindings=(),
        output_bindings=(),
        outcome_bindings=tuple(target.outcome_bindings),
        sequence=(),
        choices=(),
        protection=(),
        limits=replace(target.limits, max_nodes=2, max_subgraph_depth=2, max_bindings=2),
    )
    replacement = _one_node_factory(lambda: operation(child))
    if accepted:
        substitute(workflow=target, target=next(iter(target.nodes)).id, replacement=replacement)
    else:
        with pytest.raises(ContractViolation):
            substitute(workflow=target, target=next(iter(target.nodes)).id, replacement=replacement)


@pytest.mark.parametrize(
    "field", ["max_local_in_flight", "max_remote_outstanding", "max_runtime_artifacts", "max_collection_items"]
)
def test_absent_count_limit_still_rejects_before_callbacks(field: str) -> None:
    async def run():
        execution, capabilities, services, _ = span_graph()
        object.__setattr__(services.limits, field, None)
        with pytest.raises(EffectRejected):
            await start_execution(admitted=execution, capabilities=capabilities, services=services)

    asyncio.run(run())
