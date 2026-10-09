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
from anonymizer.engine.graph_sdk._execution_state import _DeferredAcceptance, _ExecutionControl, _RequestAuthority
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
from anonymizer.engine.graph_sdk.capabilities import (
    FrozenConfig,
    ImplementationRef,
)
from anonymizer.engine.graph_sdk.context import (
    AdaptiveRetrievalDecl,
    ContextProvider,
    ContextSelector,
    SelectorField,
    SourceFailure,
    SourceLost,
    SourceResponse,
)
from anonymizer.engine.graph_sdk.requests import (
    AcceptFailure,
    AcceptResult,
    ArtifactValue,
    AssociationInput,
    AssociationResult,
    Dispatch,
    DispatchEnvelope,
    FailureClass,
    MarkLost,
    ObserveSettlement,
    PhysicalRequestId,
    PortArtifact,
    RequestCancel,
    RequestPolicyBinding,
    Reserve,
    SemanticAssociation,
    StopAcknowledged,
    StopConfirmed,
    TextArtifactValue,
    TextCollectionItem,
    TextCollectionValue,
    TransportFailure,
    TransportLost,
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
    control: _ExecutionControl,
) -> tuple[RuntimeOutcome, tuple[AssociationResult, ...], tuple[LocalAssessmentResult, ...]]:
    if handle.local is None:
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
    if isinstance(result, LocalFailure):
        return _mapping(policy, "failure", None, result.failure), (), ()
    if isinstance(result, LocalDecisionWait):
        return _mapping(policy, "failure", None, "implementation_exception"), (), ()
    if not isinstance(result, LocalCompleted):
        return _mapping(policy, "failure", None, "implementation_exception"), (), ()
    if len(result.results) != 1 or result.results[0].association != association:
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
    if isinstance(result, LocalFailure):
        failure: FailureClass = (
            result.failure
            if result.failure in {"permanent", "implementation_exception"}
            else ("implementation_exception")
        )
        return _mapping(policy, "failure", None, failure), (), ()
    if not isinstance(result, LocalDecisionWait):
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
    deferred_acceptances: dict[SemanticAssociation, _DeferredAcceptance],
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

    def failed(
        mapping: RuntimeOutcome,
    ) -> dict[
        SemanticAssociation, tuple[RuntimeOutcome, tuple[AssociationResult, ...], tuple[LocalAssessmentResult, ...]]
    ]:
        return {association: (mapping, (), ()) for association in associations}

    implementation_index = 0
    purpose: Literal["initial", "retry", "correction", "failover"] = "initial"
    while True:
        if len(authority.state.remote_outstanding) >= limits.max_remote_outstanding:
            return failed(_mapping(policy, "request_limit_exhausted", None, None))
        implementation = policy.implementations[implementation_index]
        handle = handles[
            (implementation.implementation, implementation.capability.operation, implementation.configuration)
        ]
        if handle.transport is None:
            reject(EffectCode.MISSING)
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
        authority.apply(Dispatch(request=request))
        if handle.resource is not None:
            request_resources[request] = handle.resource.resource
        envelope = DispatchEnvelope(
            request=request,
            purpose=purpose,
            operation=handle.operation,
            associations=inputs,
        )
        dispatch = asyncio.create_task(handle.transport.dispatch(envelope))
        while not dispatch.done() and not control.cancelled:
            await asyncio.sleep(0)
        if control.cancelled and not dispatch.done():
            authority.apply(RequestCancel(request=request))
            try:
                stopped = await handle.transport.cancel(request)
            except Exception:
                stopped = None
            dispatch.cancel()
            with suppress(asyncio.CancelledError):
                await dispatch
            if isinstance(stopped, StopConfirmed):
                authority.apply(StopAcknowledged(request=request, usage=stopped.usage))
                return failed(_mapping(policy, "cancel_after_dispatch", None, None))
            authority.apply(MarkLost(request=request))
            return failed(_mapping(policy, "lost", None, None))
        try:
            result = dispatch.result()
        except Exception:
            result = TransportLost(settlement=None)
        if isinstance(result, TransportSuccess):
            if result.settlement is not None and result.settlement.request != request:
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
            returned = [value.association for value in result.results]
            keyed = len(returned) == len(associations) and set(returned) == associations
            mappings: dict[SemanticAssociation, RuntimeOutcome] = {}
            if keyed:
                for row in result.results:
                    if (
                        not isinstance(row.association, SemanticAssociation)
                        or row.outcome not in policy.result_outcomes
                    ):
                        break
                    mapping = _mapping(policy, "result", row.outcome, None)
                    row_inputs = tuple(value for value in inputs if value.association == row.association)
                    if not _validate_output_shape(
                        admitted, handle.operation, mapping, row.association, row_inputs, (row,)
                    ):
                        break
                    mappings[row.association] = mapping
            if keyed and len(mappings) == len(associations):
                deferred = _DeferredAcceptance(
                    request=request,
                    results=result.results,
                    settlement=result.settlement,
                )
                for association in associations:
                    deferred_acceptances[association] = deferred
                return {
                    association: (
                        mappings[association],
                        tuple(row for row in result.results if row.association == association),
                        (),
                    )
                    for association in associations
                }
            if not keyed:
                authority.apply(AcceptResult(request=request, results=result.results))
                if result.settlement is not None:
                    authority.apply(ObserveSettlement(settlement=result.settlement))
                return failed(_mapping(policy, "request_inconsistent", None, None))
            failure: FailureClass = "malformed_response"
            authority.apply(AcceptFailure(request=request, failure=failure))
            if result.settlement is not None and result.settlement.request == request:
                authority.apply(ObserveSettlement(settlement=result.settlement))
        elif isinstance(result, TransportFailure):
            if result.settlement is not None and result.settlement.request != request:
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
            failure = result.failure
            authority.apply(AcceptFailure(request=request, failure=failure))
            if result.settlement is not None:
                authority.apply(ObserveSettlement(settlement=result.settlement))
        else:
            authority.apply(MarkLost(request=request))
            if (
                isinstance(result, TransportLost)
                and result.settlement is not None
                and result.settlement.request == request
            ):
                authority.apply(ObserveSettlement(settlement=result.settlement))
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
    leases: dict[object, ResourceLease],
    control: _ExecutionControl,
    limits: ExecutionLimits,
    deferred_acceptances: dict[SemanticAssociation, _DeferredAcceptance],
    request_resources: dict[PhysicalRequestId, ResourceId],
) -> tuple[RuntimeOutcome, tuple[AssociationResult, ...], tuple[LocalAssessmentResult, ...]]:
    capability = next(
        item
        for item in admitted.context.context_capabilities
        if item.source == declaration.source and "adaptive_retrieval" in item.uses
    )
    lease = leases.get(declaration.source)
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
        authority.apply(Dispatch(request=request))
        request_resources[request] = lease.resource
        retrieval = asyncio.create_task(
            provider.retrieve(
                request=request,
                association=association,
                selector=selector,
                bounds=declaration.bounds,
            )
        )
        while not retrieval.done() and not control.cancelled:
            await asyncio.sleep(0)
        if control.cancelled and not retrieval.done():
            authority.apply(RequestCancel(request=request))
            try:
                stopped = await provider.cancel(request)
            except Exception:
                stopped = None
            retrieval.cancel()
            late_result: SourceResponse | SourceFailure | SourceLost | None = None
            try:
                late_result = await retrieval
            except asyncio.CancelledError:
                pass
            if isinstance(stopped, StopConfirmed):
                authority.apply(StopAcknowledged(request=request, usage=stopped.usage))
                mapping = _mapping(policy, "cancel_after_dispatch", None, None)
            else:
                authority.apply(MarkLost(request=request))
                mapping = _mapping(policy, "lost", None, None)
            if isinstance(late_result, SourceResponse):
                valid_late_result = (
                    late_result.source == declaration.source
                    and bool(late_result.items)
                    and all(item.association == association for item in late_result.items)
                    and len({(item.key, item.version) for item in late_result.items}) == len(late_result.items)
                )
                if valid_late_result:
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
                else:
                    authority.apply(AcceptFailure(request=request, failure="malformed_response"))
            late_settlement = late_result.settlement if late_result is not None else None
            if late_settlement is not None and late_settlement.request == request:
                authority.apply(ObserveSettlement(settlement=late_settlement))
            return mapping, (), ()
        try:
            result = retrieval.result()
        except Exception:
            result = SourceLost(source=declaration.source, settlement=None)
        if isinstance(result, SourceResponse):
            valid = (
                result.source == declaration.source
                and result.settlement.request == request
                and bool(result.items)
                and all(item.association == association for item in result.items)
                and len({(item.key, item.version) for item in result.items}) == len(result.items)
            )
            byte_count = sum(len(item.text.encode()) for item in result.items)
            oversize = (
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
                authority.apply(ObserveSettlement(settlement=result.settlement))
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
                deferred_acceptances[association] = _DeferredAcceptance(
                    request=request,
                    results=(accepted,),
                    settlement=result.settlement,
                )
                return _mapping(policy, "result", outcome, None), (accepted,), ()
            failure: FailureClass = "malformed_response"
            authority.apply(AcceptFailure(request=request, failure=failure))
            if result.settlement.request == request:
                authority.apply(ObserveSettlement(settlement=result.settlement))
        elif isinstance(result, SourceFailure):
            failure = (
                result.failure
                if result.source == declaration.source
                and result.disposition == "failed"
                and (result.settlement is None or result.settlement.request == request)
                else "malformed_response"
            )
            authority.apply(AcceptFailure(request=request, failure=failure))
            if result.settlement is not None and result.settlement.request == request:
                authority.apply(ObserveSettlement(settlement=result.settlement))
        else:
            if isinstance(result, SourceLost) and result.source != declaration.source:
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
            if (
                isinstance(result, SourceLost)
                and result.settlement is not None
                and result.settlement.request == request
            ):
                authority.apply(ObserveSettlement(settlement=result.settlement))
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
