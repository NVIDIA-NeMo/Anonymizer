# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Restricted same-loop borrowing of registered physical operation slots."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

from anonymizer.engine.graph_sdk._effect_values import EffectCode, PrivateValue, reject, require_count
from anonymizer.engine.graph_sdk._execution_dispatch import _mapping
from anonymizer.engine.graph_sdk._execution_state import _ExecutionJob
from anonymizer.engine.graph_sdk._execution_values import (
    ImplementationHandle,
    LocalAssessmentResult,
    LocalCompleted,
    LocalFailure,
    LocalResult,
    RuntimeOutcome,
)
from anonymizer.engine.graph_sdk._ingress import inspect_associations, inspect_failure
from anonymizer.engine.graph_sdk._request_driver import drive_request
from anonymizer.engine.graph_sdk.requests import (
    AcceptChildEvidence,
    AcceptFailure,
    AcceptResult,
    AssociationInput,
    AssociationResult,
    ChildEvidence,
    ChildSourceKind,
    DispatchEnvelope,
    MarkLost,
    PhysicalRequestId,
    RequestChain,
    RequestChainId,
    RequestChild,
    RequestPolicyBinding,
    Reserve,
    SemanticAssociation,
    TransportFailure,
    TransportSuccess,
)
from anonymizer.graph._values import TaskAttemptId
from anonymizer.graph.activation import CloseUnstarted, Start, advance_activation

if TYPE_CHECKING:
    from anonymizer.engine.graph_sdk._execution_runtime import _InvocationRuntime


class _DelegatedOwner(PrivateValue):
    """Borrow one invocation owner; never own a separate ledger or scheduler."""

    def __init__(self, runtime: _InvocationRuntime, job: _ExecutionJob) -> None:
        self.runtime = runtime
        self.job = job
        self.loop = asyncio.get_running_loop()
        self.closed = False
        self.active = False
        self.submission: asyncio.Task[LocalResult] | None = None
        contract = job.policy.delegated
        policy = job.policy.request
        assert contract is not None and policy is not None
        self.contract = contract
        self.chain = RequestChain(
            chain=RequestChainId.new(scope=runtime.request_authority.state.scope),
            associations=frozenset({job.association}),
            policy=policy,
            role="generation",
            routes=contract.routes,
            followups=contract.followups,
        )
        runtime.request_authority.bind(
            RequestPolicyBinding.create(
                association=job.association,
                policies=frozenset({policy}),
            )
        )
        runtime.request_authority.bind_chain(self.chain)

    def _check(self) -> None:
        if self.closed or asyncio.get_running_loop() is not self.loop:
            reject(EffectCode.UNSUPPORTED)

    def parser_rejected(self, request: PhysicalRequestId) -> ChildEvidence:
        self._check()
        if asyncio.current_task() is not self.submission:
            reject(EffectCode.UNSUPPORTED)
        evidence = ChildEvidence(chain=self.chain.chain, request=request, kind="parser_rejected")
        self.runtime.request_authority.apply(AcceptChildEvidence(evidence=evidence))
        return evidence

    async def request(
        self,
        slot: int,
        *,
        source_kind: ChildSourceKind = "initial",
        predecessor: PhysicalRequestId | None = None,
        evidence: ChildEvidence | None = None,
    ) -> PhysicalRequestId:
        self._check()
        if asyncio.current_task() is not self.submission:
            reject(EffectCode.UNSUPPORTED)
        require_count(slot)
        if slot not in self.contract.child_slots:
            reject(EffectCode.FOREIGN_OWNER)
        if self.active:
            reject(EffectCode.UNSUPPORTED)
        runtime = self.runtime
        key = runtime.target_keys[self.job.target][slot]
        entry = next(item for item in runtime.states[self.job.state_index].entries if item.activation == key)
        if key in runtime.attempts:
            reject(EffectCode.DUPLICATE)
        if entry.status != "ready":
            reject(EffectCode.REQUEST_CAUSALITY)
        if runtime.control.cancelled:
            reject(EffectCode.REQUEST_CAUSALITY)
        if len(runtime.request_authority.state.remote_outstanding) >= runtime.services.limits.max_remote_outstanding:
            reject(EffectCode.LIMIT_EXCEEDED)
        attempt = TaskAttemptId.new(activation=key)
        policy = runtime.policies[entry.template]
        implementation = policy.implementations[0]
        leaf = _ExecutionJob(
            state_index=self.job.state_index,
            target=self.job.target,
            activation=key,
            node=entry.template,
            policy=policy,
            implementation=implementation,
            inputs=(AssociationInput(association=SemanticAssociation(task=attempt), inputs=()),),
            input_parents={},
            association=SemanticAssociation(task=attempt),
        )
        handle = runtime.handles[
            implementation.implementation, implementation.capability.operation, implementation.configuration
        ]
        transport = handle.transport
        assert transport is not None
        authority = runtime.request_authority
        request = PhysicalRequestId.new(scope=authority.state.scope)
        authority.apply(
            Reserve(
                request=request,
                purpose="initial" if source_kind == "initial" else "correction",
                associations=self.chain.associations,
                policy=self.chain.policy,
                child=RequestChild(
                    chain=self.chain.chain,
                    role="generation",
                    route=self.contract.routes[0].name,
                    source_kind=source_kind,
                    predecessor=predecessor,
                    evidence=evidence,
                    leaf=attempt,
                ),
            )
        )
        if not any(item.request == request for item in authority.state.reserved):
            reject(EffectCode.LIMIT_EXCEEDED)
        envelope = DispatchEnvelope(
            request=request,
            purpose="initial" if source_kind == "initial" else "correction",
            operation=handle.operation,
            associations=(AssociationInput(association=self.job.association, inputs=()),),
        )

        async def invoke() -> object:
            runtime._start_observed_activation(leaf.activation, operation=True)
            return await transport.dispatch(envelope)

        self.active = True
        try:
            completion = await drive_request(
                authority,
                request,
                runtime.control,
                invoke,
                transport.cancel,
                prepare_dispatch=lambda: self._prepare(leaf, attempt),
            )
            if request not in authority.state.dispatched:
                reject(EffectCode.REQUEST_CAUSALITY)
            value = completion.value
            results: tuple[AssociationResult, ...] = ()
            if completion.cancelled:
                mapping = _mapping(policy, "cancel_after_dispatch" if completion.stop_confirmed else "lost", None, None)
            elif type(value) is TransportSuccess:
                invalid = inspect_associations(value.results, self.chain.associations, runtime.admitted.ingress_limits)
                if invalid is None:
                    authority.apply(AcceptResult(request=request, results=()))
                    mapping = _mapping(policy, "request_inconsistent", None, None)
                elif (
                    not completion.settlement_valid
                    or invalid
                    or len(value.results) != 1
                    or value.results[0].outcome not in policy.result_outcomes
                    or value.results[0].outputs
                    or value.results[0].consumed_context_ports
                ):
                    authority.apply(AcceptFailure(request=request, failure="malformed_response"))
                    mapping = _mapping(policy, "failure", None, "malformed_response")
                else:
                    authority.apply(AcceptResult(request=request, results=value.results))
                    results = (replace(value.results[0], association=leaf.association),)
                    mapping = _mapping(policy, "result", value.results[0].outcome, None)
            elif type(value) is TransportFailure:
                failure = inspect_failure(value.failure) if completion.settlement_valid else "malformed_response"
                authority.apply(AcceptFailure(request=request, failure=failure))
                mapping = _mapping(policy, "failure", None, failure)
            else:
                authority.apply(MarkLost(request=request))
                mapping = _mapping(policy, "lost", None, None)
            mapping, results, checkpoint, staged = runtime._stage_outputs(leaf, mapping, results, ())
            runtime._advance_completed(leaf, mapping, results, checkpoint, staged)
            runtime.control.scheduler_changed.set()
            span = runtime._operation_spans[leaf.activation]
            span.outcome = mapping.category
            span.finish()
            runtime._close_observed_activations()
            return request
        finally:
            self.active = False

    def _prepare(self, leaf: _ExecutionJob, attempt: TaskAttemptId) -> Callable[[], None]:
        self._check()
        runtime = self.runtime
        # Compute all rejecting activation work before the driver's authoritative Dispatch.
        state = advance_activation(state=runtime.states[leaf.state_index], event=Start(activation=leaf.activation))

        def commit() -> None:
            runtime.states[leaf.state_index] = state
            runtime.attempts[leaf.activation] = attempt

        return commit

    def close(self) -> bool:
        self.closed = True
        complete = True
        for slot in self.contract.child_slots:
            key = self.runtime.target_keys[self.job.target][slot]
            state = self.runtime.states[self.job.state_index]
            entry = next(item for item in state.entries if item.activation == key)
            if entry.status in {"ready", "unstarted"}:
                complete = False
                self.runtime.states[self.job.state_index] = advance_activation(
                    state=state,
                    event=CloseUnstarted(activation=key, category="blocked"),
                )
        return complete

    async def run(
        self, handle: ImplementationHandle
    ) -> tuple[RuntimeOutcome, tuple[AssociationResult, ...], tuple[LocalAssessmentResult, ...]]:
        assert handle.delegated is not None
        runtime = self.runtime
        task = asyncio.create_task(handle.delegated.submit(self.job.inputs, self))
        self.submission = task
        try:
            while not task.done():
                if runtime.control.cancelled and not self.active:
                    task.cancel()
                await asyncio.sleep(0)
            result = task.result()
        except asyncio.CancelledError:
            return _mapping(self.job.policy, "cancel_after_start", None, None), (), ()
        except Exception:
            return _mapping(self.job.policy, "failure", None, "implementation_exception"), (), ()
        finally:
            complete = self.close()
        if not complete:
            return _mapping(self.job.policy, "failure", None, "implementation_exception"), (), ()
        if type(result) is LocalFailure:
            return _mapping(self.job.policy, "failure", None, inspect_failure(result.failure)), (), ()
        if type(result) is not LocalCompleted:
            return _mapping(self.job.policy, "failure", None, "malformed_response"), (), ()
        invalid = inspect_associations(
            result.results, self.chain.associations, runtime.admitted.ingress_limits, assessments=result.assessments
        )
        if invalid is None or invalid or result.results[0].outcome not in self.job.policy.result_outcomes:
            return _mapping(self.job.policy, "failure", None, "malformed_response"), (), ()
        return _mapping(self.job.policy, "result", result.results[0].outcome, None), result.results, result.assessments
