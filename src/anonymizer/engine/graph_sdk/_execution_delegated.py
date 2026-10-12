# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Restricted invocation-owned borrowing of registered physical operation slots."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from contextvars import ContextVar, copy_context
from dataclasses import replace
from typing import TYPE_CHECKING, ClassVar, Literal, TypeVar

from anonymizer.engine.graph_sdk._effect_values import EffectCode, PrivateValue, reject, require_count
from anonymizer.engine.graph_sdk._execution_dispatch import _mapping, _run_local
from anonymizer.engine.graph_sdk._execution_state import _ExecutionJob
from anonymizer.engine.graph_sdk._execution_values import (
    DelegatedRequests,
    ImplementationHandle,
    LocalAssessmentResult,
    LocalCompleted,
    LocalFailure,
    LocalResult,
    RuntimeOutcome,
)
from anonymizer.engine.graph_sdk._ingress import inspect_associations, inspect_failure
from anonymizer.engine.graph_sdk._request_driver import drive_request
from anonymizer.engine.graph_sdk.blocking import BlockingAffinity, BlockingPool, _BlockingAdapter, _BlockingInvocation
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


_T = TypeVar("_T")
_WORKER_BORROW: ContextVar[_WorkerBorrow | None] = ContextVar("graph_worker_borrow", default=None)


class WorkerDelegatedSubmitter(_BlockingAdapter):
    """One caller-owned pool and borrowed loop; all physical calls stay on the host loop.

    The pool and loop are caller-owned execution resources, not stateless handles.
    Other captured values must be stateless and cross-thread safe. The caller keeps
    these resources live until this submission drains and owns their later close.
    """

    execution: ClassVar[Literal["worker"]] = "worker"

    def __init__(
        self,
        *,
        callback: Callable[[tuple[AssociationInput, ...], DelegatedRequests], LocalResult],
        pool: BlockingPool,
        affinity: BlockingAffinity,
        borrowed_loop: asyncio.AbstractEventLoop,
    ) -> None:
        super().__init__(callback=callback, pool=pool, affinity=affinity)
        if not isinstance(borrowed_loop, asyncio.AbstractEventLoop):
            reject(EffectCode.INVALID_TYPE)
        self._borrowed_loop = borrowed_loop

    @property
    def borrowed_loop(self) -> asyncio.AbstractEventLoop:
        return self._borrowed_loop

    async def submit(self, inputs: tuple[AssociationInput, ...], requests: DelegatedRequests) -> LocalResult:
        # Only the invocation owner may issue the authenticated worker borrow.
        reject(EffectCode.UNSUPPORTED)


class _WorkerBorrow(PrivateValue):
    def __init__(self, owner: _DelegatedOwner, adapter: WorkerDelegatedSubmitter) -> None:
        self.owner = owner
        self.adapter = adapter
        self.loop = adapter.borrowed_loop
        self.expired = False

    def _check_origin(self) -> None:
        if _WORKER_BORROW.get() is not self or self.expired:
            reject(EffectCode.FOREIGN_OWNER)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            reject(EffectCode.UNSUPPORTED)
        if loop is not self.loop:
            reject(EffectCode.UNSUPPORTED)

    async def _call(self, operation: Callable[[], Coroutine[object, object, _T]]) -> _T:
        self._check_origin()
        future = asyncio.run_coroutine_threadsafe(self.owner._worker_call(self, operation), self.owner.loop)
        try:
            completion = asyncio.wrap_future(future)
            # Observation may leave before a rejected/late callback finishes.
            completion.add_done_callback(lambda done: None if done.cancelled() else done.exception())
            return await asyncio.shield(completion)
        except asyncio.CancelledError:
            if future.done():
                return future.result()
            self.owner.loop.call_soon_threadsafe(self.owner._cancel_worker, self)
            raise

    async def request(
        self,
        slot: int,
        *,
        source_kind: ChildSourceKind = "initial",
        predecessor: PhysicalRequestId | None = None,
        evidence: ChildEvidence | None = None,
    ) -> PhysicalRequestId:
        return await self._call(
            lambda: self.owner.request(slot, source_kind=source_kind, predecessor=predecessor, evidence=evidence)
        )

    async def readiness(self) -> PhysicalRequestId:
        return await self._call(self.owner.readiness)

    async def probe(self) -> None:
        await self._call(self.owner.probe)

    def parser_rejected(self, request: PhysicalRequestId) -> ChildEvidence:
        self._check_origin()

        async def accept() -> ChildEvidence:
            return self.owner.parser_rejected(request)

        # D may wait for I here: I never waits synchronously for D or a worker.
        return asyncio.run_coroutine_threadsafe(self.owner._worker_call(self, accept), self.owner.loop).result()


class _DelegatedOwner(PrivateValue):
    """Borrow one invocation owner; never own a separate ledger or scheduler."""

    def __init__(self, runtime: _InvocationRuntime, job: _ExecutionJob) -> None:
        self.runtime = runtime
        self.job = job
        self.loop = asyncio.get_running_loop()
        self.closed = False
        self.child: asyncio.Task[PhysicalRequestId | None] | None = None
        self.cancel_requested = False
        self.submission: asyncio.Task[LocalResult] | None = None
        self.worker_borrow: _WorkerBorrow | None = None
        self.worker_call: asyncio.Task[object] | None = None
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
        task = asyncio.current_task()
        if task is not self.submission and (self.worker_call is None or task is not self.worker_call):
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
        if slot not in self.contract.child_slots:
            reject(EffectCode.FOREIGN_OWNER)
        readiness = self.contract.readiness
        if readiness is not None and (
            not self._satisfied(readiness.owner_slot, frozenset({readiness.ready_outcome}))
            or not self._satisfied(readiness.probe_slot, readiness.probe_outcomes)
        ):
            reject(EffectCode.REQUEST_CAUSALITY)
        self._check_submission()
        result = await self._borrow(self._request(slot, source_kind, predecessor, evidence, self.chain))
        assert result is not None
        return result

    def _check_submission(self) -> None:
        self._check()
        task = asyncio.current_task()
        if (task is not self.submission and (self.worker_call is None or task is not self.worker_call)) or self.active:
            reject(EffectCode.UNSUPPORTED)
        if self.cancelled:
            reject(EffectCode.REQUEST_CAUSALITY)

    async def _borrow(self, operation: Coroutine[object, object, PhysicalRequestId | None]) -> PhysicalRequestId | None:
        self.child = asyncio.create_task(operation)
        try:
            return await asyncio.shield(self.child)
        except asyncio.CancelledError:
            if not self.child.done():
                self.cancel_requested = True
            raise

    def _cancel_worker(self, borrow: _WorkerBorrow) -> None:
        if self.worker_borrow is borrow:
            self.cancel_requested = True

    async def _worker_call(self, borrow: _WorkerBorrow, operation: Callable[[], Coroutine[object, object, _T]]) -> _T:
        self._check()
        if self.worker_borrow is not borrow or _WORKER_BORROW.get() is not borrow or borrow.expired:
            reject(EffectCode.FOREIGN_OWNER)
        if self.worker_call is not None or self.active or self.cancelled:
            reject(EffectCode.REQUEST_CAUSALITY)
        self.worker_call = asyncio.current_task()
        try:
            return await operation()
        finally:
            self.worker_call = None

    async def _submit_worker(self, adapter: WorkerDelegatedSubmitter) -> LocalResult:
        borrow = _WorkerBorrow(self, adapter)
        self.worker_borrow = borrow
        context = copy_context()
        context.run(_WORKER_BORROW.set, borrow)
        invocation = _BlockingInvocation(adapter, lambda: context.run(adapter._callback, self.job.inputs, borrow))
        try:
            value = await invocation.run(lambda: not self.cancelled, self)
            return (
                value
                if isinstance(value, (LocalCompleted, LocalFailure))
                else LocalFailure(failure="malformed_response")
            )
        finally:
            borrow.expired = True

    def _satisfied(self, slot: int, outcomes: frozenset[str]) -> bool:
        key = self.runtime.target_keys[self.job.target][slot]
        return any(
            entry.activation == key and entry.status == "success" and entry.outcome in outcomes
            for entry in self.runtime.states[self.job.state_index].entries
        )

    async def readiness(self) -> PhysicalRequestId:
        self._check_submission()
        if self.contract.readiness is None:
            reject(EffectCode.UNSUPPORTED)
        result = await self._borrow(self._readiness())
        assert result is not None
        return result

    async def _readiness(self) -> PhysicalRequestId:
        descriptor = self.contract.readiness
        assert descriptor is not None
        semantic = self._job(descriptor.owner_slot)
        self._prepare(semantic, semantic.association.task)()
        self.runtime._start_observed_activation(semantic.activation, operation=True)
        policy = semantic.policy.request
        assert policy is not None
        chain = RequestChain(
            chain=RequestChainId.new(scope=self.runtime.request_authority.state.scope),
            associations=frozenset({semantic.association}),
            policy=policy,
            role="readiness",
            routes=(descriptor.route,),
            followups=frozenset(),
        )
        authority = self.runtime.request_authority
        authority.bind(RequestPolicyBinding.create(association=semantic.association, policies=frozenset({policy})))
        authority.bind_chain(chain)
        mapping = _mapping(semantic.policy, "failure", None, "implementation_exception")
        results: tuple[AssociationResult, ...] = ()
        try:
            request = await self._request(descriptor.request_slot, "initial", None, None, chain)
            key = self.runtime.target_keys[self.job.target][descriptor.request_slot]
            entry = next(item for item in self.runtime.states[self.job.state_index].entries if item.activation == key)
            if entry.status == "success":
                assert entry.outcome is not None
                mapping = _mapping(semantic.policy, "result", entry.outcome, None)
                results = (
                    AssociationResult(
                        association=semantic.association,
                        outcome=entry.outcome,
                        outputs=(),
                        consumed_context_ports=frozenset(),
                    ),
                )
            return request
        finally:
            self._finish(semantic, mapping, results)

    async def probe(self) -> None:
        self._check_submission()
        descriptor = self.contract.readiness
        if descriptor is None or not self._satisfied(descriptor.owner_slot, frozenset({descriptor.ready_outcome})):
            reject(EffectCode.REQUEST_CAUSALITY)
        await self._borrow(self._probe())

    async def _probe(self) -> None:
        descriptor = self.contract.readiness
        assert descriptor is not None
        job = self._job(descriptor.probe_slot)
        self._prepare(job, job.association.task)()
        self.runtime._start_observed_activation(job.activation, operation=True)
        implementation = job.implementation
        handle = self.runtime.handles[
            implementation.implementation, implementation.capability.operation, implementation.configuration
        ]
        try:
            mapping, results, assessments = await _run_local(
                job.policy, handle, job.association, job.inputs, self, self.runtime.admitted.ingress_limits
            )
        except asyncio.CancelledError:
            mapping, results, assessments = _mapping(job.policy, "cancel_after_start", None, None), (), ()
        self._finish(job, mapping, results, assessments)

    @property
    def active(self) -> bool:
        return self.child is not None and not self.child.done()

    @property
    def cancelled(self) -> bool:
        return self.cancel_requested or self.runtime.control.cancelled

    def _job(self, slot: int) -> _ExecutionJob:
        require_count(slot)
        if slot not in self.contract.owned_slots:
            reject(EffectCode.FOREIGN_OWNER)
        runtime = self.runtime
        key = runtime.target_keys[self.job.target][slot]
        entry = next(item for item in runtime.states[self.job.state_index].entries if item.activation == key)
        if key in runtime.attempts:
            reject(EffectCode.DUPLICATE)
        if entry.status != "ready":
            reject(EffectCode.REQUEST_CAUSALITY)
        if self.cancelled:
            reject(EffectCode.REQUEST_CAUSALITY)
        attempt = TaskAttemptId.new(activation=key)
        policy = runtime.policies[entry.template]
        implementation = policy.implementations[0]
        return _ExecutionJob(
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

    async def _request(
        self,
        slot: int,
        source_kind: ChildSourceKind,
        predecessor: PhysicalRequestId | None,
        evidence: ChildEvidence | None,
        chain: RequestChain,
    ) -> PhysicalRequestId:
        runtime = self.runtime
        if len(runtime.request_authority.state.remote_outstanding) >= runtime.services.limits.max_remote_outstanding:
            reject(EffectCode.LIMIT_EXCEEDED)
        leaf = self._job(slot)
        attempt = leaf.association.task
        policy = leaf.policy
        implementation = leaf.implementation
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
                associations=chain.associations,
                policy=chain.policy,
                child=RequestChild(
                    chain=chain.chain,
                    role=chain.role,
                    route=chain.routes[0].name,
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
            associations=(AssociationInput(association=next(iter(chain.associations)), inputs=()),),
        )

        async def invoke() -> object:
            runtime._start_observed_activation(leaf.activation, operation=True)
            return await transport.dispatch(envelope)

        completion = await drive_request(
            authority,
            request,
            self,
            invoke,
            transport.cancel,
            prepare_dispatch=lambda: self._prepare(leaf, attempt),
            drain_on_cancel=True,
        )
        if request not in authority.state.dispatched:
            reject(EffectCode.REQUEST_CAUSALITY)
        value = completion.value
        results: tuple[AssociationResult, ...] = ()
        if completion.cancelled:
            mapping = _mapping(policy, "cancel_after_dispatch" if completion.stop_confirmed else "lost", None, None)
        elif type(value) is TransportSuccess:
            invalid = inspect_associations(value.results, chain.associations, runtime.admitted.ingress_limits)
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
        self._finish(leaf, mapping, results)
        return request

    def _finish(
        self,
        job: _ExecutionJob,
        mapping: RuntimeOutcome,
        results: tuple[AssociationResult, ...],
        assessments: tuple[LocalAssessmentResult, ...] = (),
    ) -> None:
        runtime = self.runtime
        mapping, results, checkpoint, staged = runtime._stage_outputs(job, mapping, results, assessments)
        runtime._advance_completed(job, mapping, results, checkpoint, staged)
        runtime.control.scheduler_changed.set()
        span = runtime._operation_spans[job.activation]
        span.outcome = mapping.category
        span.finish()
        runtime._close_observed_activations()

    async def _drain(self, task: asyncio.Task[object]) -> None:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                self.cancel_requested = True
            except Exception:
                break
        if not task.cancelled():
            task.exception()  # Retrieve failures even when the submitting observer left.

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
        for slot in self.contract.owned_slots:
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
        task = asyncio.create_task(
            self._submit_worker(handle.delegated)
            if type(handle.delegated) is WorkerDelegatedSubmitter
            else handle.delegated.submit(self.job.inputs, self)
        )
        self.submission = task
        try:
            while not task.done():
                if runtime.control.cancelled and not self.active:
                    task.cancel()
                await asyncio.sleep(0)
            result = task.result()
        except asyncio.CancelledError:
            self.cancel_requested = True
            if not task.done():
                task.cancel()
            return _mapping(self.job.policy, "cancel_after_start", None, None), (), ()
        except Exception:
            return _mapping(self.job.policy, "failure", None, "implementation_exception"), (), ()
        finally:
            if self.child is not None:
                await self._drain(self.child)
            await self._drain(task)
            if self.worker_call is not None:
                await self._drain(self.worker_call)
            if self.child is not None:
                await self._drain(self.child)
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
