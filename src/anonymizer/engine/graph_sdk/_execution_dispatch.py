# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local, remote and adaptive request dispatch."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from typing import cast

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
    inspect_source,
    inspect_source_identity,
)
from anonymizer.engine.graph_sdk._ingress_values import IngressLimits
from anonymizer.engine.graph_sdk._request_driver import (
    _Cancellation,
    _PhysicalCompletion,
    _RequestAuthority,
    drive_request,
    late_source_event,
)
from anonymizer.engine.graph_sdk.blocking import BlockingContextProvider, BlockingTransport
from anonymizer.engine.graph_sdk.capabilities import (
    FrozenConfig,
    ImplementationRef,
)
from anonymizer.engine.graph_sdk.context import (
    AdaptiveRetrievalDecl,
    ContextProvider,
    ContextSelector,
    ContextSourceCapability,
    SelectorField,
    SourceFailure,
    SourceLost,
    SourceResponse,
)
from anonymizer.engine.graph_sdk.observations import _CURRENT, _Span
from anonymizer.engine.graph_sdk.requests import (
    AcceptFailure,
    AcceptResult,
    ArtifactValue,
    AssociationInput,
    AssociationResult,
    DispatchEnvelope,
    FailureClass,
    MarkLost,
    PhysicalRequestId,
    PortArtifact,
    RejectSettlement,
    RequestAssociationSummary,
    RequestPolicyBinding,
    RequestPurpose,
    Reserve,
    SemanticAssociation,
    TextArtifactValue,
    TextCollectionItem,
    TextCollectionValue,
    TransportFailure,
    TransportLost,
    TransportResult,
    TransportSuccess,
    can_reserve_followup,
)
from anonymizer.engine.graph_sdk.resources import (
    ResourceId,
    ResourceLease,
)
from anonymizer.graph.workflow import (
    OperationSpec,
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
) -> (
    tuple[RuntimeOutcome, tuple[AssociationResult, ...], tuple[LocalAssessmentResult, ...]]
    | LocalCompleted
    | SubmissionFailure
):
    if handle.local is None:
        reject(EffectCode.MISSING)
    try:
        operation = asyncio.create_task(handle.local.run(inputs))
        while not operation.done():
            if control.cancelled:
                operation.cancel()
                with suppress(Exception, asyncio.CancelledError):
                    await operation
                return _mapping(policy, "cancel_after_start", None, None), (), ()
            await asyncio.sleep(0)
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


@dataclass(frozen=True, slots=True)
class _Retrieval:
    declaration: AdaptiveRetrievalDecl
    capability: ContextSourceCapability
    lease: ResourceLease
    selector: ContextSelector
    association: SemanticAssociation

    @property
    def provider(self) -> ContextProvider:
        return cast(ContextProvider, self.lease.handle)

    def response(
        self,
        completion: _PhysicalCompletion,
        policy: OperationExecutionPolicy,
        ingress: IngressLimits,
        authority: _RequestAuthority,
        request: PhysicalRequestId,
    ) -> tuple[TransportResult, bool, bool]:
        """Translate source ingress without assigning physical success or retry policy."""
        declaration, association = self.declaration, self.association
        value = (
            SourceFailure(source=declaration.source, failure="malformed_response", settlement=None)
            if completion.malformed
            else completion.value
        )
        failure: FailureClass = "malformed_response"
        if type(value) is SourceResponse:
            if (
                inspect_source(value, association, ingress)
                and value.source == declaration.source
                and completion.settlement_valid
                and bool(value.items)
                and all(item.association == association for item in value.items)
                and len({(item.key, item.version) for item in value.items}) == len(value.items)
            ):
                oversize = (
                    len(value.items) > declaration.bounds.max_items
                    or sum(len(item.text.encode()) for item in value.items) > declaration.bounds.max_bytes
                    or declaration.materialization.kind == "single"
                    and len(value.items) > 1
                )
                outputs = ()
                if not oversize:
                    artifact: ArtifactValue = (
                        TextArtifactValue(text=value.items[0].text)
                        if declaration.materialization.kind == "single"
                        else TextCollectionValue(
                            items=tuple(
                                TextCollectionItem(
                                    key=item.key, version=item.version, value=TextArtifactValue(text=item.text)
                                )
                                for item in sorted(value.items, key=lambda item: (item.key, item.version))
                            )
                        )
                    )
                    operation = policy.implementations[0].capability.operation
                    outputs = (
                        PortArtifact(
                            port=declaration.output_port,
                            artifact_type=next(
                                item.artifact_type for item in operation.outputs if item.name == declaration.output_port
                            ),
                            artifact=None,
                            value=artifact,
                        ),
                    )
                return (
                    TransportSuccess(
                        results=(
                            AssociationResult(
                                association=association,
                                outcome=_adaptive_success_outcome(policy),
                                outputs=outputs,
                                consumed_context_ports=frozenset(),
                            ),
                        ),
                        settlement=None,
                    ),
                    oversize,
                    False,
                )
        elif type(value) is SourceFailure:
            if (
                inspect_source_identity(value.source, declaration.source)
                and type(value.disposition) is str
                and value.disposition == "failed"
                and completion.settlement_valid
            ):
                failure = inspect_failure(value.failure)
        elif type(value) is SourceLost and not inspect_source_identity(value.source, declaration.source):
            if value.settlement is None:
                authority.apply(RejectSettlement(request=request))
            return TransportFailure(failure=failure, settlement=None), False, True
        else:
            return TransportLost(settlement=None), False, False
        return TransportFailure(failure=failure, settlement=None), False, False


async def _run_external_batch(
    admitted: AdmittedExecutionPlan,
    policy: OperationExecutionPolicy,
    handles: dict[tuple[ImplementationRef, OperationSpec, FrozenConfig], ImplementationHandle],
    inputs: tuple[AssociationInput, ...],
    authority: _RequestAuthority,
    control: _ExecutionControl,
    limits: ExecutionLimits,
    request_resources: dict[PhysicalRequestId, ResourceId],
    source: _Retrieval | None = None,
) -> dict[SemanticAssociation, tuple[RuntimeOutcome, tuple[AssociationResult, ...], tuple[LocalAssessmentResult, ...]]]:
    if policy.request is None:
        reject(EffectCode.MISSING)
    request_policy = policy.request if source is None else source.capability.request
    semantic: set[SemanticAssociation] = set()
    for value in inputs:
        if not isinstance(value.association, SemanticAssociation):
            reject(EffectCode.CONTRADICTORY)
        semantic.add(value.association)
    associations = frozenset(semantic)
    if source is None:
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
    purpose: RequestPurpose = policy.initial_request_purpose if source is None else "adaptive_retrieval"
    while True:
        if (
            limits.max_remote_outstanding is not None
            and len(authority.state.remote_outstanding) >= limits.max_remote_outstanding
        ):
            return failed(_mapping(policy, "request_limit_exhausted", None, None))
        implementation = policy.implementations[implementation_index]
        handle = (
            handles[(implementation.implementation, implementation.capability.operation, implementation.configuration)]
            if source is None
            else None
        )
        if source is None and (handle is None or handle.transport is None):
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
        if source is None:
            assert handle is not None and handle.transport is not None
            transport = handle.transport
            if handle.resource is not None:
                request_resources[request] = handle.resource.resource
            envelope = DispatchEnvelope(
                request=request, purpose=purpose, operation=handle.operation, associations=inputs
            )
            completion = await drive_request(
                authority,
                request,
                control,
                lambda: transport.dispatch(envelope),
                transport.cancel,
                blocking=transport._invocation(envelope) if isinstance(transport, BlockingTransport) else None,
            )
            if _CURRENT.get() is not None and request in authority.state.dispatched:
                control.measured_handles.update((association, handle) for association in associations)
        else:
            declaration, association, provider = source.declaration, source.association, source.provider
            request_resources[request] = source.lease.resource
            completion = await drive_request(
                authority,
                request,
                control,
                lambda: provider.retrieve(
                    request=request, association=association, selector=source.selector, bounds=declaration.bounds
                ),
                provider.cancel,
                lambda value: late_source_event(
                    value,
                    request,
                    association,
                    _adaptive_success_outcome(policy),
                    declaration.source,
                    lambda response: (
                        inspect_source(response, association, admitted._response_limits())
                        and response.source == declaration.source
                        and bool(response.items)
                        and len({(item.key, item.version) for item in response.items}) == len(response.items)
                    ),
                ),
                blocking=provider._invocation(
                    request=request, association=association, selector=source.selector, bounds=declaration.bounds
                )
                if isinstance(provider, BlockingContextProvider)
                else None,
            )
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
        settlement_valid = completion.settlement_valid
        if source is not None:
            result, oversize, correction_only = source.response(
                completion, policy, admitted._response_limits(), authority, request
            )
            settlement_valid = True  # Source ingress already classified the settlement with its payload.
            if type(result) is TransportSuccess:
                authority.apply(AcceptResult(request=request, results=result.results))
                return {
                    source.association: (
                        _mapping(policy, "artifact_limit_exhausted", None, None)
                        if oversize
                        else _mapping(policy, "result", result.results[0].outcome, None),
                        () if oversize else result.results,
                        (),
                    )
                }
        if type(result) in {TransportSuccess, TransportFailure} and not settlement_valid:
            failure = "malformed_response"
            authority.apply(AcceptFailure(request=request, failure=failure))
            correction_only = True
        elif type(result) is TransportSuccess:
            invalid = inspect_associations(result.results, associations, admitted._response_limits())
            if invalid is None:
                safe_rows = (
                    type(result.results) is tuple
                    and len(result.results) <= admitted._response_limits().max_associations
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
            and source is None
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


async def _run_adaptive(
    admitted: AdmittedExecutionPlan,
    policy: OperationExecutionPolicy,
    declaration: AdaptiveRetrievalDecl,
    association: SemanticAssociation,
    inputs: tuple[AssociationInput, ...],
    authority: _RequestAuthority,
    leases: dict[ContextSourceCapability, ResourceLease],
    control: _ExecutionControl,
    limits: ExecutionLimits,
    request_resources: dict[PhysicalRequestId, ResourceId],
) -> tuple[RuntimeOutcome, tuple[AssociationResult, ...], tuple[LocalAssessmentResult, ...]]:
    with _Span("adaptive_retrieval") as span:
        result = await _run_adaptive_body(
            admitted, policy, declaration, association, inputs, authority, leases, control, limits, request_resources
        )
        span.outcome = result[0].category
        if result[0].failure is not None:
            span.reasons = frozenset(
                {"malformed_response" if result[0].failure == "malformed_response" else "provider_failure"}
            )
        return result


async def _run_adaptive_body(
    admitted: AdmittedExecutionPlan,
    policy: OperationExecutionPolicy,
    declaration: AdaptiveRetrievalDecl,
    association: SemanticAssociation,
    inputs: tuple[AssociationInput, ...],
    authority: _RequestAuthority,
    leases: dict[ContextSourceCapability, ResourceLease],
    control: _ExecutionControl,
    limits: ExecutionLimits,
    request_resources: dict[PhysicalRequestId, ResourceId],
) -> tuple[RuntimeOutcome, tuple[AssociationResult, ...], tuple[LocalAssessmentResult, ...]]:
    capability = next(
        item
        for item in admitted.context.context_capabilities
        if item.source == declaration.source
        and item.artifact_type == declaration.materialization.item_type
        and "adaptive_retrieval" in item.uses
    )
    lease = leases.get(capability)
    if lease is None or not isinstance(lease.handle, ContextProvider) or policy.request is None:
        reject(EffectCode.MISSING)
    authority.bind(RequestPolicyBinding.create(association=association, policies=frozenset({capability.request})))
    source_inputs = {item.port: item.value for item in inputs[0].inputs}
    fields: list[SelectorField] = []
    for port in declaration.selector_ports:
        value = source_inputs.get(port)
        if not isinstance(value, TextArtifactValue):
            return _mapping(policy, "failure", None, "malformed_response"), (), ()
        fields.append(SelectorField(name=port, value=value.text))
    selector = ContextSelector(fields=tuple(fields))
    result = await _run_external_batch(
        admitted,
        policy,
        {},
        inputs,
        authority,
        control,
        limits,
        request_resources,
        _Retrieval(declaration, capability, lease, selector, association),
    )
    return result[association]


def _adaptive_success_outcome(policy: OperationExecutionPolicy) -> str:
    matches = [
        item
        for item in policy.implementations[0].capability.operation.outcomes
        if item.name in policy.result_outcomes and item.category == "success"
    ]
    if len(matches) != 1:
        reject(EffectCode.CONTRADICTORY)
    return matches[0].name


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
