# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local, remote and adaptive request dispatch."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Literal

from anonymizer.engine.graph_sdk._effect_values import (
    EffectCode,
    reject,
)
from anonymizer.engine.graph_sdk._execution_publication import _identity_input, _validate_output_shape
from anonymizer.engine.graph_sdk._execution_state import _ExecutionControl
from anonymizer.engine.graph_sdk._execution_values import (
    AdmittedExecutionPlan,
    DecisionWait,
    DecisionWaitId,
    ExecutionLimits,
    ExecutionServices,
    ImplementationHandle,
    LocalAssessmentResult,
    LocalCompleted,
    LocalDecisionWait,
    LocalFailure,
    OperationExecutionPolicy,
    RuntimeOutcome,
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
    Reserve,
    SemanticAssociation,
    TextArtifactValue,
    TextCollectionItem,
    TextCollectionValue,
    TransportFailure,
    TransportSuccess,
    can_reserve_followup,
)
from anonymizer.engine.graph_sdk.resources import (
    ResourceId,
    ResourceLease,
)
from anonymizer.graph._values import (
    ActivationKey,
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
) -> tuple[RuntimeOutcome, tuple[AssociationResult, ...], tuple[LocalAssessmentResult, ...]]:
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
    except Exception:
        return _mapping(policy, "failure", None, "implementation_exception"), (), ()
    if type(result) is LocalFailure:
        return _mapping(policy, "failure", None, inspect_failure(result.failure)), (), ()
    if type(result) is LocalDecisionWait:
        return _mapping(policy, "failure", None, "implementation_exception"), (), ()
    if type(result) is not LocalCompleted:
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


async def _run_decision(
    admitted: AdmittedExecutionPlan,
    policy: OperationExecutionPolicy,
    handle: ImplementationHandle,
    association: SemanticAssociation,
    inputs: tuple[AssociationInput, ...],
    activation: ActivationKey,
    services: ExecutionServices,
    control: _ExecutionControl,
) -> tuple[RuntimeOutcome, tuple[AssociationResult, ...], tuple[LocalAssessmentResult, ...]]:
    if handle.local is None:
        reject(EffectCode.MISSING)
    declaration = next(item for item in admitted.decisions if item.node == policy.node)
    artifact = next((item.artifact for item in inputs[0].inputs if item.port == declaration.artifact_port), None)
    if artifact is None:
        reject(EffectCode.MISSING)
    operation = asyncio.create_task(handle.local.run(inputs))
    try:
        while not operation.done():
            if control.cancelled:
                operation.cancel()
                with suppress(asyncio.CancelledError):
                    await operation
                return _mapping(policy, "cancel_after_start", None, None), (), ()
            await asyncio.sleep(0)
        result = operation.result()
    except Exception:
        return _mapping(policy, "failure", None, "implementation_exception"), (), ()
    if type(result) is LocalFailure:
        failure: FailureClass = (
            result.failure
            if result.failure in {"permanent", "implementation_exception"}
            else ("implementation_exception")
        )
        return _mapping(policy, "failure", None, failure), (), ()
    if type(result) is not LocalDecisionWait:
        return _mapping(policy, "failure", None, "implementation_exception"), (), ()
    if result.association != association or result.artifact != artifact:
        return _mapping(policy, "failure", None, "implementation_exception"), (), ()
    wait = DecisionWait(
        wait=DecisionWaitId.new(invocation=activation.invocation),
        activation=activation,
        workflow=admitted.context.prepared.workflow.workflow.workflow,
        artifact=artifact,
        allowed_decisions=frozenset(item.decision for item in declaration.outcomes),
        deadline_ns=services.clock.now_ns() + declaration.max_lifetime_ns,
    )
    pending = control.pending
    responses = control.responses
    closed = control.closed
    with _Span("decision_wait") as span:
        pending[wait.wait] = wait
        control.scheduler_changed.set()
        while True:
            if control.cancelled:
                mapping = _mapping(policy, "cancel_after_start", None, None)
                break
            response = responses.pop(wait.wait, None)
            if response is not None:
                selected = next(item for item in declaration.outcomes if item.decision == response.decision)
                outcome = next(item for item in handle.operation.outcomes if item.name == selected.outcome)
                category = outcome.category
                mapping = RuntimeOutcome(
                    condition="result",
                    reported_outcome=selected.outcome,
                    failure=None,
                    outcome=selected.outcome,
                    category=category,
                )
                available = {item.port: item for item in inputs[0].inputs}
                dependencies = {item.output: item for item in handle.operation.output_dependencies}
                output_types = {item.name: item.artifact_type for item in handle.operation.outputs}
                outputs = tuple(
                    PortArtifact(
                        port=port,
                        artifact_type=output_types[port],
                        artifact=None,
                        value=available[_identity_input(dependencies[port])].value,
                    )
                    for port in sorted(outcome.produced_ports)
                )
                break
            if services.clock.now_ns() >= wait.deadline_ns:
                mapping = _mapping(policy, "deadline_exhausted", None, None)
                break
            await asyncio.sleep(0)
        del pending[wait.wait]
        closed.add(wait.wait)
        span.outcome = mapping.category
        if mapping.condition == "deadline_exhausted":
            span.reasons = frozenset({"deadline"})
    if mapping.condition != "result":
        return mapping, (), ()
    return (
        mapping,
        (
            AssociationResult(
                association=association,
                outcome=mapping.outcome or "",
                outputs=outputs,
                consumed_context_ports=frozenset(use.port for use in outcome.context if use.port in available),
            ),
        ),
        (),
    )


async def _external_association_result(
    physical: asyncio.Task[
        dict[
            SemanticAssociation, tuple[RuntimeOutcome, tuple[AssociationResult, ...], tuple[LocalAssessmentResult, ...]]
        ]
    ],
    association: SemanticAssociation,
) -> tuple[RuntimeOutcome, tuple[AssociationResult, ...], tuple[LocalAssessmentResult, ...]]:
    return (await physical)[association]


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
    semantic: set[SemanticAssociation] = set()
    for value in inputs:
        if not isinstance(value.association, SemanticAssociation):
            reject(EffectCode.CONTRADICTORY)
        semantic.add(value.association)
    associations = frozenset(semantic)
    for association in associations:
        authority.bind(RequestPolicyBinding.create(association=association, policies=frozenset({policy.request})))

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
    purpose: Literal["initial", "repair", "retry", "correction", "failover"] = policy.initial_request_purpose
    while True:
        if len(authority.state.remote_outstanding) >= limits.max_remote_outstanding:
            return failed(_mapping(policy, "request_limit_exhausted", None, None))
        implementation = policy.implementations[implementation_index]
        handle = handles[
            (implementation.implementation, implementation.capability.operation, implementation.configuration)
        ]
        if handle.transport is None:
            reject(EffectCode.MISSING)
        transport = handle.transport
        request = PhysicalRequestId.new(scope=authority.state.scope)
        authority.apply(
            Reserve(
                request=request,
                purpose=purpose,
                associations=associations,
                policy=policy.request,
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
        )
        if _CURRENT.get() is not None and request in authority.state.dispatched:
            for association in associations:
                control.measured_handles[association] = handle
        if completion.cancelled:
            condition = (
                "cancel_before_start"
                if request not in authority.state.dispatched
                else "cancel_after_dispatch"
                if completion.stop_confirmed
                else "lost"
            )
            return failed(_mapping(policy, condition, None, None))
        result = (
            TransportFailure(failure="malformed_response", settlement=None)
            if completion.malformed
            else completion.value
        )
        settlement_valid = completion.settlement_valid
        if type(result) is TransportSuccess:
            if not settlement_valid:
                failure = "malformed_response"
                authority.apply(AcceptFailure(request=request, failure=failure))
                if can_reserve_followup(
                    state=authority.state,
                    purpose="correction",
                    associations=associations,
                    policy=policy.request,
                ):
                    purpose = "correction"
                    continue
                return failed(_mapping(policy, "failure", None, failure))
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
                    admitted, handle.operation, mapping, row.association, row_inputs, (row,)
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
            failure: FailureClass = "malformed_response"
        elif type(result) is TransportFailure:
            if not settlement_valid:
                failure = "malformed_response"
                authority.apply(AcceptFailure(request=request, failure=failure))
                if can_reserve_followup(
                    state=authority.state,
                    purpose="correction",
                    associations=associations,
                    policy=policy.request,
                ):
                    purpose = "correction"
                    continue
                return failed(_mapping(policy, "failure", None, failure))
            failure = inspect_failure(result.failure)
            authority.apply(AcceptFailure(request=request, failure=failure))
        else:
            authority.apply(MarkLost(request=request))
            return failed(_mapping(policy, "lost", None, None))

        if failure == "malformed_response" and can_reserve_followup(
            state=authority.state,
            purpose="correction",
            associations=associations,
            policy=policy.request,
        ):
            purpose = "correction"
        elif (
            failure in {"permanent", "implementation_exception"}
            and implementation_index + 1 < len(policy.implementations)
            and can_reserve_followup(
                state=authority.state,
                purpose="failover",
                associations=associations,
                policy=policy.request,
            )
        ):
            implementation_index += 1
            purpose = "failover"
        elif policy.request.retry_owner == "executor" and can_reserve_followup(
            state=authority.state,
            purpose="retry",
            associations=associations,
            policy=policy.request,
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
    provider = lease.handle
    authority.bind(RequestPolicyBinding.create(association=association, policies=frozenset({capability.request})))
    source_inputs = {item.port: item.value for item in inputs[0].inputs}
    fields: list[SelectorField] = []
    for port in declaration.selector_ports:
        value = source_inputs.get(port)
        if not isinstance(value, TextArtifactValue):
            return _mapping(policy, "failure", None, "malformed_response"), (), ()
        fields.append(SelectorField(name=port, value=value.text))
    selector = ContextSelector(fields=tuple(fields))
    purpose: Literal["adaptive_retrieval", "retry", "correction"] = "adaptive_retrieval"
    while True:
        if len(authority.state.remote_outstanding) >= limits.max_remote_outstanding:
            return _mapping(policy, "request_limit_exhausted", None, None), (), ()
        request = PhysicalRequestId.new(scope=authority.state.scope)
        authority.apply(
            Reserve(
                request=request,
                purpose=purpose,
                associations=frozenset({association}),
                policy=capability.request,
            ),
        )
        if not any(item.request == request for item in authority.state.reserved):
            denial = next(item for item in reversed(authority.state.denials) if item.request == request)
            condition = "budget_exhausted" if denial.category == "budget_stopped" else "request_limit_exhausted"
            return _mapping(policy, condition, None, None), (), ()
        request_resources[request] = lease.resource
        completion = await drive_request(
            authority,
            request,
            control,
            lambda: provider.retrieve(
                request=request, association=association, selector=selector, bounds=declaration.bounds
            ),
            provider.cancel,
            lambda value: late_source_event(
                value,
                request,
                association,
                _adaptive_success_outcome(policy),
                declaration.source,
                lambda response: (
                    inspect_source(response, association, admitted.ingress_limits)
                    and response.source == declaration.source
                    and bool(response.items)
                    and len({(item.key, item.version) for item in response.items}) == len(response.items)
                ),
            ),
            blocking=(
                provider._invocation(
                    request=request, association=association, selector=selector, bounds=declaration.bounds
                )
                if isinstance(provider, BlockingContextProvider)
                else None
            ),
        )
        if completion.cancelled:
            condition = (
                "cancel_before_start"
                if request not in authority.state.dispatched
                else "cancel_after_dispatch"
                if completion.stop_confirmed
                else "lost"
            )
            return _mapping(policy, condition, None, None), (), ()
        result = (
            SourceFailure(source=declaration.source, failure="malformed_response", settlement=None)
            if completion.malformed
            else completion.value
        )
        if type(result) is SourceResponse:
            valid = (
                inspect_source(result, association, admitted.ingress_limits)
                and result.source == declaration.source
                and completion.settlement_valid
                and bool(result.items)
                and all(item.association == association for item in result.items)
                and len({(item.key, item.version) for item in result.items}) == len(result.items)
            )
            byte_count = sum(len(item.text.encode()) for item in result.items) if valid else 0
            oversize = valid and (
                len(result.items) > declaration.bounds.max_items
                or byte_count > declaration.bounds.max_bytes
                or (declaration.materialization.kind == "single" and len(result.items) > 1)
            )
            if valid and oversize:
                authority.apply(
                    AcceptResult(
                        request=request,
                        results=(
                            AssociationResult(
                                association=association,
                                outcome=_adaptive_success_outcome(policy),
                                outputs=(),
                                consumed_context_ports=frozenset(),
                            ),
                        ),
                    )
                )
                return _mapping(policy, "artifact_limit_exhausted", None, None), (), ()
            if valid:
                value: ArtifactValue
                if declaration.materialization.kind == "single":
                    value = TextArtifactValue(text=result.items[0].text)
                else:
                    value = TextCollectionValue(
                        items=tuple(
                            TextCollectionItem(
                                key=item.key,
                                version=item.version,
                                value=TextArtifactValue(text=item.text),
                            )
                            for item in sorted(result.items, key=lambda item: (item.key, item.version))
                        )
                    )
                outcome = _adaptive_success_outcome(policy)
                operation = policy.implementations[0].capability.operation
                artifact_type = next(
                    item.artifact_type for item in operation.outputs if item.name == declaration.output_port
                )
                accepted = AssociationResult(
                    association=association,
                    outcome=outcome,
                    outputs=(
                        PortArtifact(
                            port=declaration.output_port,
                            artifact_type=artifact_type,
                            artifact=None,
                            value=value,
                        ),
                    ),
                    consumed_context_ports=frozenset(),
                )
                authority.apply(AcceptResult(request=request, results=(accepted,)))
                return _mapping(policy, "result", outcome, None), (accepted,), ()
            failure: FailureClass = "malformed_response"
            authority.apply(AcceptFailure(request=request, failure=failure))
        elif type(result) is SourceFailure:
            failure = (
                inspect_failure(result.failure)
                if inspect_source_identity(result.source, declaration.source)
                and type(result.disposition) is str
                and result.disposition == "failed"
                and (completion.settlement_valid)
                else "malformed_response"
            )
            authority.apply(AcceptFailure(request=request, failure=failure))
        else:
            if type(result) is SourceLost and not inspect_source_identity(result.source, declaration.source):
                if result.settlement is None:
                    authority.apply(RejectSettlement(request=request))
                authority.apply(AcceptFailure(request=request, failure="malformed_response"))
                if can_reserve_followup(
                    state=authority.state,
                    purpose="correction",
                    associations=frozenset({association}),
                    policy=capability.request,
                ):
                    purpose = "correction"
                    continue
                return _mapping(policy, "failure", None, "malformed_response"), (), ()
            authority.apply(MarkLost(request=request))
            return _mapping(policy, "lost", None, None), (), ()
        if failure == "malformed_response" and can_reserve_followup(
            state=authority.state,
            purpose="correction",
            associations=frozenset({association}),
            policy=capability.request,
        ):
            purpose = "correction"
        elif capability.request.retry_owner == "executor" and can_reserve_followup(
            state=authority.state,
            purpose="retry",
            associations=frozenset({association}),
            policy=capability.request,
        ):
            purpose = "retry"
        else:
            return _mapping(policy, "failure", None, failure), (), ()


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
