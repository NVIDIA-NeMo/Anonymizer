# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual owner-boundary replacements for extracted combined-model transitions."""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from anonymizer.engine.graph_sdk.binding import RunningBinding
from anonymizer.engine.graph_sdk.context import ContextSourceRef, SourceResponse, admit_context_plan
from anonymizer.engine.graph_sdk.preparation import PreparedPlan
from anonymizer.engine.graph_sdk.requests import (
    AssociationResult,
    ExactUsage,
    ExternalSettlement,
    InvocationRequestScope,
    PhysicalRequestId,
)
from anonymizer.graph import EffectRejected
from anonymizer.graph._values import InvocationId, PlanId
from tests.graph_sdk import test_binding_lifecycle_owners as lifecycle
from tests.graph_sdk.effects_admission_fixtures import _assert_decision_submission
from tests.graph_sdk.reference.corpora import load_cases
from tests.graph_sdk.test_cleanup_attribution_amendment import (
    test_cleanup_attribution_corruption_withholds_successful_peer as check_cleanup,
)
from tests.graph_sdk.test_effects_production_conformance import (
    test_admission_corpus_case_through_production as check_admission,
)
from tests.graph_sdk.test_effects_production_conformance import (
    test_binding_corpus_case_through_real_provider as check_binding,
)
from tests.graph_sdk.test_effects_production_conformance import (
    test_late_adaptive_provider_result_is_suppressed_by_real_executor as check_late_adaptive,
)
from tests.graph_sdk.test_latest_binding_obligations import test_latest_binding_obligation as check_latest
from tests.graph_sdk.test_preparation import _capability, _prepare
from tests.graph_sdk.test_qualification import (
    test_initial_binding_cleanup_rejects_inconsistent_associations as check_missing_cleanup,
)
from tests.graph_sdk.test_requests import _association


def test_sealed_binding_has_no_second_dispatch_publication_or_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[RunningBinding] = []
    prepared_plans: list[PreparedPlan] = []
    start = lifecycle.start_initial_binding

    async def capture(**kwargs: Any) -> RunningBinding:
        running = await start(**kwargs)
        captured.append(running)
        prepared = _prepare(
            data=kwargs["data"],
            workflow=kwargs["workflow"],
            capability=_capability(kwargs["workflow"]),
            bound_inputs=(),
        )
        prepared_plans.append(prepared)
        # Both no-finish and genuinely pending required context reject at the
        # real admission boundary, before any execution can be constructed.
        for _ in range(2):
            with pytest.raises(EffectRejected) as rejected:
                admit_context_plan(
                    prepared=prepared, bound_context=None, adaptive_retrievals=(), context_capabilities=()
                )
            assert rejected.value.code.value == "missing"
            await asyncio.sleep(0)
        return running

    async def verify() -> None:
        await lifecycle._assert_binding_lifecycle()
        (running,) = captured
        first = await running.wait()
        (prepared,) = prepared_plans
        admitted = admit_context_plan(
            prepared=prepared, bound_context=first.context, adaptive_retrievals=(), context_capabilities=()
        )
        assert admitted.bound_context is first.context
        before = (first.receipt.sources, first.receipt.requests, first.receipt.artifacts, first.receipt.cleanup)
        for _ in range(3):
            running.request_cancel()
            assert await running.wait() is first
            assert (
                first.receipt.sources,
                first.receipt.requests,
                first.receipt.artifacts,
                first.receipt.cleanup,
            ) == before
        # Publication is available only through the completed owner; receipt fields
        # cannot become an event ingestion path after completion.
        with pytest.raises((AttributeError, FrozenInstanceError)):
            setattr(first.receipt, "terminal", "failure")
        assert first.receipt.terminal == "partial"

    monkeypatch.setattr(lifecycle, "start_initial_binding", capture)
    asyncio.run(verify())


@pytest.mark.parametrize("mutation", ["missing", "foreign-target"])
def test_exact_cleanup_owner_defect(mutation: str) -> None:
    if mutation == "missing":
        check_missing_cleanup("missing")
    else:
        check_cleanup(True, "foreign-target")


@pytest.mark.parametrize("case_id", ["materialization/adaptive_late_lost", "materialization/adaptive_late_cancelled"])
def test_adaptive_late_publication_uses_actual_owner(case_id: str) -> None:
    check_late_adaptive(next(row for row in load_cases("effects") if row["case_id"] == case_id))


@pytest.mark.parametrize("case_id", ["binding/exact_item_byte_bounds", "binding/wrong_source"])
def test_binding_finish_and_wrong_source(case_id: str) -> None:
    check_binding(next(row for row in load_cases("effects") if row["case_id"] == case_id))


@pytest.mark.parametrize(
    "case_id,code",
    [
        ("decisions/stale_artifact", "foreign_owner"),
        ("decisions/foreign_workflow", "foreign_owner"),
        ("decisions/foreign_wait", "foreign_owner"),
        ("decisions/unknown_decision", "unsupported"),
        ("decisions/duplicate_response", "duplicate"),
    ],
)
def test_rejected_decision_preserves_exact_wait(case_id: str, code: str) -> None:
    asyncio.run(_assert_decision_submission(case_id, code))


def test_callback_cannot_supply_runtime_category_or_producer_node() -> None:
    for field, value in (("category", "failure"), ("node", "OTHER")):
        with pytest.raises(TypeError, match="unexpected keyword"):
            AssociationResult(
                association=_association(InvocationId.new(plan=PlanId.new()), 0),
                outcome="ok",
                outputs=(),
                consumed_context_ports=frozenset(),
                **{field: value},
            )


def test_missing_runtime_mapping_rejects_real_admission() -> None:
    row = next(row for row in load_cases("effects") if row["case_id"] == "admission/runtime_mapping_missing")
    check_admission(row)


@pytest.mark.parametrize(
    "case_id",
    ["materialization/latest_bound_with_optional_omission", "materialization/latest_publication_rollback_then_reuse"],
)
def test_actual_publication_keeps_exact_parent_and_publishes_once(case_id: str) -> None:
    check_latest(next(row for row in load_cases("effects") if row["case_id"] == case_id))


def test_adaptive_provider_cannot_supply_semantic_outcome() -> None:
    request = PhysicalRequestId.new(scope=InvocationRequestScope(invocation=InvocationId.new(plan=PlanId.new())))
    settlement = ExternalSettlement(
        request=request, disposition="completed", usage=ExactUsage(input_units=0, output_units=0), remote_stopped=True
    )
    source = ContextSourceRef(name="source", revision=1)
    assert SourceResponse(source=source, items=(), settlement=settlement).settlement is settlement
    for field in ("outcome", "category"):
        with pytest.raises(TypeError, match="unexpected keyword"):
            SourceResponse(source=source, items=(), settlement=settlement, **{field: "forged"})
