# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct tests for physical request accounting."""

from __future__ import annotations

from typing import Literal

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected
from anonymizer.engine.graph_sdk.requests import (
    AcceptFailure,
    AcceptResult,
    AssociationResult,
    BindingAssociation,
    BindingDeclarationId,
    BindingId,
    Dispatch,
    FailureClass,
    InvocationRequestScope,
    PhysicalRequestId,
    PhysicalRequestPolicy,
    PortArtifact,
    RequestCancel,
    RequestPolicyBinding,
    Reserve,
    SemanticAssociation,
    StopAcknowledged,
    TextArtifactValue,
    UnknownUsage,
    advance_requests,
    bind_request_policies,
    can_reserve_followup,
    initialize_requests,
    request_receipt,
)
from anonymizer.graph._values import ActivationKey, InvocationId, PlanId, TaskAttemptId
from anonymizer.graph.workflow import ArtifactType


def _association(invocation: InvocationId, occurrence: int) -> SemanticAssociation:
    return SemanticAssociation(
        task=TaskAttemptId.new(
            activation=ActivationKey(invocation=invocation, occurrence=occurrence, parent=None, iteration=None)
        )
    )


def _policy(
    *,
    attempts: int = 2,
    replay: Literal["never", "before_acceptance", "idempotent"] = "idempotent",
) -> PhysicalRequestPolicy:
    return PhysicalRequestPolicy(
        visibility="dispatch_and_settlement",
        pre_dispatch_control="executor",
        retry_owner="executor",
        replay=replay,
        max_attempts=attempts,
    )


def test_shared_request_retains_dispatch_attribution_and_reordered_results() -> None:
    invocation = InvocationId.new(plan=PlanId.new())
    scope = InvocationRequestScope(invocation=invocation)
    policy = _policy()
    first, second = (_association(invocation, index) for index in range(2))
    state = initialize_requests(scope=scope, hard_limit=1, policies=frozenset({policy}))
    for association in (first, second):
        state = bind_request_policies(
            state=state,
            binding=RequestPolicyBinding.create(association=association, policies=frozenset({policy})),
        )
    request = PhysicalRequestId.new(scope=scope)
    state = advance_requests(
        state=state,
        event=Reserve(
            request=request,
            purpose="initial",
            associations=frozenset({first, second}),
            policy=policy,
        ),
    )
    state = advance_requests(state=state, event=Dispatch(request=request))
    results = tuple(
        AssociationResult(
            association=association,
            outcome="ok",
            outputs=(),
            consumed_context_ports=frozenset(),
        )
        for association in (second, first)
    )
    state = advance_requests(state=state, event=AcceptResult(request=request, results=results))
    receipt = request_receipt(state)
    assert receipt.dispatched_count == 1
    assert receipt.dispatches[0].associations == frozenset({first, second})
    assert receipt.terminals[0].category == "success"


def test_keyed_missing_is_inconsistent_and_first_terminal_is_immutable() -> None:
    invocation = InvocationId.new(plan=PlanId.new())
    scope = InvocationRequestScope(invocation=invocation)
    policy = _policy()
    association = _association(invocation, 0)
    state = initialize_requests(scope=scope, hard_limit=None, policies=frozenset({policy}))
    state = bind_request_policies(
        state=state,
        binding=RequestPolicyBinding.create(association=association, policies=frozenset({policy})),
    )
    request = PhysicalRequestId.new(scope=scope)
    state = advance_requests(
        state=state,
        event=Reserve(request=request, purpose="initial", associations=frozenset({association}), policy=policy),
    )
    state = advance_requests(state=state, event=Dispatch(request=request))
    state = advance_requests(state=state, event=AcceptResult(request=request, results=()))
    state = advance_requests(state=state, event=AcceptFailure(request=request, failure="permanent"))
    assert state.terminals[0].category == "inconsistent"
    assert {item.code for item in state.defects} == {"missing_keyed_result", "conflicting_terminal"}


def test_zero_budget_denies_without_dispatch_or_charge() -> None:
    invocation = InvocationId.new(plan=PlanId.new())
    scope = InvocationRequestScope(invocation=invocation)
    policy = _policy()
    association = _association(invocation, 0)
    state = initialize_requests(scope=scope, hard_limit=0, policies=frozenset({policy}))
    state = bind_request_policies(
        state=state,
        binding=RequestPolicyBinding.create(association=association, policies=frozenset({policy})),
    )
    state = advance_requests(
        state=state,
        event=Reserve(
            request=PhysicalRequestId.new(scope=scope),
            purpose="initial",
            associations=frozenset({association}),
            policy=policy,
        ),
    )
    assert not state.reserved and not state.dispatches
    assert state.denials[0].category == "budget_stopped"


@pytest.mark.parametrize(
    ("failure", "purpose", "replay", "expected"),
    (
        ("malformed_response", "correction", "never", False),
        ("malformed_response", "correction", "before_acceptance", False),
        ("malformed_response", "correction", "idempotent", True),
        ("permanent", "failover", "never", False),
        ("permanent", "failover", "before_acceptance", False),
        ("permanent", "failover", "idempotent", True),
    ),
)
def test_followup_eligibility_uses_the_request_authority_replay_rule(
    failure: FailureClass,
    purpose: Literal["correction", "failover"],
    replay: Literal["never", "before_acceptance", "idempotent"],
    expected: bool,
) -> None:
    invocation = InvocationId.new(plan=PlanId.new())
    scope = InvocationRequestScope(invocation=invocation)
    policy = _policy(replay=replay)
    association = _association(invocation, 0)
    state = initialize_requests(scope=scope, hard_limit=None, policies=frozenset({policy}))
    state = bind_request_policies(
        state=state,
        binding=RequestPolicyBinding.create(association=association, policies=frozenset({policy})),
    )
    request = PhysicalRequestId.new(scope=scope)
    state = advance_requests(
        state=state,
        event=Reserve(request=request, purpose="initial", associations=frozenset({association}), policy=policy),
    )
    state = advance_requests(state=state, event=Dispatch(request=request))
    state = advance_requests(state=state, event=AcceptFailure(request=request, failure=failure))
    assert (
        can_reserve_followup(
            state=state,
            purpose=purpose,
            associations=frozenset({association}),
            policy=policy,
        )
        is expected
    )


def test_dispatched_cancel_intent_is_retained_through_stop_and_receipt() -> None:
    invocation = InvocationId.new(plan=PlanId.new())
    scope = InvocationRequestScope(invocation=invocation)
    policy = _policy()
    association = _association(invocation, 0)
    state = initialize_requests(scope=scope, hard_limit=None, policies=frozenset({policy}))
    state = bind_request_policies(
        state=state,
        binding=RequestPolicyBinding.create(association=association, policies=frozenset({policy})),
    )
    request = PhysicalRequestId.new(scope=scope)
    state = advance_requests(
        state=state,
        event=Reserve(request=request, purpose="initial", associations=frozenset({association}), policy=policy),
    )
    state = advance_requests(state=state, event=Dispatch(request=request))
    state = advance_requests(state=state, event=RequestCancel(request=request))
    state = advance_requests(state=state, event=RequestCancel(request=request))
    state = advance_requests(state=state, event=StopAcknowledged(request=request, usage=UnknownUsage()))
    assert state.cancel_requested == frozenset({request})
    assert request_receipt(state).cancel_requested == frozenset({request})


@pytest.mark.parametrize(
    ("outcome", "outputs", "consumed"),
    (
        ("ok", (), frozenset()),
        (
            "retrieved",
            (
                PortArtifact(
                    port="unexpected",
                    artifact_type=ArtifactType(name="text", revision=1),
                    artifact=None,
                    value=TextArtifactValue(text="unexpected"),
                ),
            ),
            frozenset(),
        ),
        ("retrieved", (), frozenset({"context"})),
    ),
)
def test_binding_results_require_the_retrieved_empty_physical_shape(
    outcome: str,
    outputs: tuple[PortArtifact, ...],
    consumed: frozenset[str],
) -> None:
    binding = BindingId.new()
    association = BindingAssociation(declaration=BindingDeclarationId.new(binding=binding, ordinal=0))
    with pytest.raises(EffectRejected) as caught:
        AssociationResult(
            association=association,
            outcome=outcome,
            outputs=outputs,
            consumed_context_ports=consumed,
        )
    assert caught.value.code == EffectCode.CONTRADICTORY
