# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local and remote request dispatch."""

from __future__ import annotations

import asyncio
from contextlib import suppress

from anonymizer.engine.graph_sdk._effect_values import (
    EffectCode,
    reject,
)
from anonymizer.engine.graph_sdk._execution_publication import _validate_output_shape
from anonymizer.engine.graph_sdk._execution_state import _ExecutionControl
from anonymizer.engine.graph_sdk._execution_values import (
    AdmittedExecutionPlan,
    ExecutionLimits,
    ImplementationHandle,
    LocalAssessmentResult,
    LocalCompleted,
    LocalFailure,
    OperationExecutionPolicy,
    RuntimeOutcome,
    SubmissionFailure,
)
from anonymizer.engine.graph_sdk._ingress import (
    _trusted_association,
    inspect_associations,
    inspect_failure,
)
from anonymizer.engine.graph_sdk._ingress_values import IngressLimits
from anonymizer.engine.graph_sdk._request_driver import (
    _Cancellation,
    _RequestAuthority,
    await_completion_or_cancel,
    drive_request,
)
from anonymizer.engine.graph_sdk._workflow_values import OperationSpec
from anonymizer.engine.graph_sdk.blocking import BlockingTransport
from anonymizer.engine.graph_sdk.capabilities import (
    FrozenConfig,
    ImplementationRef,
)
from anonymizer.engine.graph_sdk.observations import _CURRENT
from anonymizer.engine.graph_sdk.requests import (
    AcceptFailure,
    AcceptResult,
    AssociationInput,
    AssociationResult,
    DispatchEnvelope,
    FailureClass,
    MarkLost,
    PhysicalRequestId,
    RequestAssociationSummary,
    RequestPolicyBinding,
    RequestPurpose,
    Reserve,
    SemanticAssociation,
    TransportFailure,
    TransportSuccess,
    can_reserve_followup,
)
from anonymizer.engine.graph_sdk.resources import (
    ResourceId,
)


async def _run_local(
    policy: OperationExecutionPolicy,
    handle: ImplementationHandle,
    association: SemanticAssociation,
    inputs: tuple[AssociationInput, ...],
    control: _Cancellation,
    ingress: IngressLimits,
    *,
    invocation: bool = False,
    cancel_waiters: set[asyncio.Future[None]] | None = None,
) -> (
    tuple[RuntimeOutcome, tuple[AssociationResult, ...], tuple[LocalAssessmentResult, ...]]
    | LocalCompleted
    | SubmissionFailure
):
    if handle.local is None:
        reject(EffectCode.MISSING)
    try:
        operation = asyncio.create_task(handle.local.run(inputs))
        await await_completion_or_cancel(operation, control, cancel_waiters)
        if not operation.done():
            operation.cancel()
            with suppress(Exception, asyncio.CancelledError):
                await operation
            return _mapping(policy, "cancel_after_start", None, None), (), ()
        result = operation.result()
    except asyncio.CancelledError:
        if not invocation:
            raise
        return _mapping(policy, "cancel_after_start", None, None), (), ()
    except Exception as error:
        if invocation:
            return SubmissionFailure(associations=frozenset({association}), cause=error)
        return _mapping(policy, "failure", None, "implementation_exception"), (), ()
    if invocation and type(result) is LocalCompleted:
        return result
    if type(result) is LocalFailure:
        return _mapping(policy, "failure", None, inspect_failure(result.failure)), (), ()
    if (
        type(result) is not LocalCompleted
        or getattr(result, "frame", False) is not None
        or getattr(result, "failed_sources", None) != frozenset()
    ):
        return _mapping(policy, "failure", None, "implementation_exception"), (), ()
    invalid = inspect_associations(result.results, frozenset({association}), ingress, assessments=result.assessments)
    if invalid is None or invalid:
        return _mapping(policy, "failure", None, "malformed_response"), (), ()
    reported = result.results[0].outcome
    if reported not in policy.result_outcomes:
        return _mapping(policy, "failure", None, "malformed_response"), (), ()
    return _mapping(policy, "result", reported, None), result.results, result.assessments


async def _immediate_execution_result(
    mapping: RuntimeOutcome,
) -> tuple[RuntimeOutcome, tuple[AssociationResult, ...], tuple[LocalAssessmentResult, ...]]:
    return mapping, (), ()


async def _run_external_batch(
    admitted: AdmittedExecutionPlan,
    policy: OperationExecutionPolicy,
    handles: dict[tuple[ImplementationRef, OperationSpec, FrozenConfig], ImplementationHandle],
    inputs: tuple[AssociationInput, ...],
    authority: _RequestAuthority,
    control: _ExecutionControl,
    limits: ExecutionLimits,
    request_resources: dict[PhysicalRequestId, ResourceId],
) -> dict[SemanticAssociation, tuple[RuntimeOutcome, tuple[AssociationResult, ...], tuple[LocalAssessmentResult, ...]]]:
    if policy.request is None:
        reject(EffectCode.MISSING)
    request_policy = policy.request
    semantic: set[SemanticAssociation] = set()
    for value in inputs:
        if not isinstance(value.association, SemanticAssociation):
            reject(EffectCode.CONTRADICTORY)
        semantic.add(value.association)
    associations = frozenset(semantic)
    for association in associations:
        authority.bind(RequestPolicyBinding.create(association=association, policies=frozenset({request_policy})))

    completed: dict[
        SemanticAssociation, tuple[RuntimeOutcome, tuple[AssociationResult, ...], tuple[LocalAssessmentResult, ...]]
    ] = {}

    def failed(
        mapping: RuntimeOutcome,
    ) -> dict[
        SemanticAssociation, tuple[RuntimeOutcome, tuple[AssociationResult, ...], tuple[LocalAssessmentResult, ...]]
    ]:
        return {**completed, **{association: (mapping, (), ()) for association in associations}}

    implementation_index = 0
    purpose: RequestPurpose = policy.initial_request_purpose
    while True:
        if (
            limits.max_remote_outstanding is not None
            and len(authority.state.remote_outstanding) >= limits.max_remote_outstanding
        ):
            return failed(_mapping(policy, "request_limit_exhausted", None, None))
        implementation = policy.implementations[implementation_index]
        handle = handles[
            (implementation.implementation, implementation.capability.operation, implementation.configuration)
        ]
        transport = handle.transport
        if transport is None:
            reject(EffectCode.MISSING)
        request = PhysicalRequestId.new(scope=authority.state.scope)
        authority.apply(
            Reserve(
                request=request,
                purpose=purpose,
                associations=associations,
                policy=request_policy,
            ),
        )
        if not any(item.request == request for item in authority.state.reserved):
            denial = next(item for item in reversed(authority.state.denials) if item.request == request)
            condition = "budget_exhausted" if denial.category == "budget_stopped" else "request_limit_exhausted"
            return failed(_mapping(policy, condition, None, None))
        if handle.resource is not None:
            request_resources[request] = handle.resource.resource
        envelope = DispatchEnvelope(request=request, purpose=purpose, operation=handle.operation, associations=inputs)
        completion = await drive_request(
            authority,
            request,
            control,
            lambda: transport.dispatch(envelope),
            transport.cancel,
            blocking=transport._invocation(envelope) if isinstance(transport, BlockingTransport) else None,
            cancel_waiters=control.cancel_waiters,
        )
        if _CURRENT.get() is not None and request in authority.state.dispatched:
            control.measured_handles.update((association, handle) for association in associations)
        if completion.cancelled:
            condition = (
                "cancel_before_start"
                if request not in authority.state.dispatched
                else "cancel_after_dispatch"
                if completion.stop_confirmed
                else "lost"
            )
            return failed(_mapping(policy, condition, None, None))
        correction_only = False
        result = (
            TransportFailure(failure="malformed_response", settlement=None)
            if completion.malformed
            else completion.value
        )
        if type(result) in {TransportSuccess, TransportFailure} and not completion.settlement_valid:
            failure = "malformed_response"
            authority.apply(AcceptFailure(request=request, failure=failure))
            correction_only = True
        elif type(result) is TransportSuccess:
            invalid = inspect_associations(result.results, associations, admitted.ingress_limits)
            if invalid is None:
                safe_rows = (
                    type(result.results) is tuple
                    and len(result.results) <= admitted.ingress_limits.max_associations
                    and all(
                        type(row) is AssociationResult and _trusted_association(row.association)
                        for row in result.results
                    )
                )
                authority.apply(AcceptResult(request=request, results=result.results if safe_rows else ()))
                return failed(_mapping(policy, "request_inconsistent", None, None))
            summaries: list[RequestAssociationSummary] = []
            malformed: set[SemanticAssociation] = set()
            for row in result.results:
                assert isinstance(row.association, SemanticAssociation)
                row_inputs = tuple(value for value in inputs if value.association == row.association)
                mapping = (
                    _mapping(policy, "result", row.outcome, None)
                    if row.association not in invalid and row.outcome in policy.result_outcomes
                    else None
                )
                if mapping is not None and _validate_output_shape(
                    admitted, implementation.capability.operation, mapping, row.association, row_inputs, (row,)
                ):
                    completed[row.association] = (mapping, (row,), ())
                    summaries.append(
                        RequestAssociationSummary(
                            association=row.association,
                            outcome=row.outcome,
                            category="success",
                            failure=None,
                            condition="result",
                            consumed_context_ports=row.consumed_context_ports,
                        )
                    )
                else:
                    malformed.add(row.association)
                    summaries.append(
                        RequestAssociationSummary(
                            association=row.association,
                            outcome=None,
                            category="failure",
                            failure="malformed_response",
                            condition="failure",
                            consumed_context_ports=frozenset(),
                        )
                    )
            authority.apply(AcceptResult(request=request, results=tuple(summaries)))
            if not malformed:
                return completed
            associations = frozenset(malformed)
            inputs = tuple(value for value in inputs if value.association in associations)
            failure = "malformed_response"
        elif type(result) is TransportFailure:
            failure = inspect_failure(result.failure)
            authority.apply(AcceptFailure(request=request, failure=failure))
        else:
            authority.apply(MarkLost(request=request))
            return failed(_mapping(policy, "lost", None, None))

        if failure == "malformed_response" and can_reserve_followup(
            state=authority.state,
            purpose="correction",
            associations=associations,
            policy=request_policy,
        ):
            purpose = "correction"
        elif (
            not correction_only
            and failure in {"permanent", "implementation_exception"}
            and implementation_index + 1 < len(policy.implementations)
            and can_reserve_followup(
                state=authority.state,
                purpose="failover",
                associations=associations,
                policy=request_policy,
            )
        ):
            implementation_index += 1
            purpose = "failover"
        elif (
            not correction_only
            and request_policy.retry_owner == "executor"
            and can_reserve_followup(
                state=authority.state,
                purpose="retry",
                associations=associations,
                policy=request_policy,
            )
        ):
            purpose = "retry"
        else:
            return failed(_mapping(policy, "failure", None, failure))


def _mapping(
    policy: OperationExecutionPolicy,
    condition: str,
    reported_outcome: str | None,
    failure: FailureClass | None,
) -> RuntimeOutcome:
    matches = [
        item
        for item in policy.runtime_outcomes
        if (item.condition, item.reported_outcome, item.failure) == (condition, reported_outcome, failure)
    ]
    if len(matches) != 1:
        reject(EffectCode.MISSING)
    return matches[0]
