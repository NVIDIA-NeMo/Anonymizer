# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Restricted invocation-owned borrowing of registered physical operation slots."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from contextvars import ContextVar, copy_context
from dataclasses import replace
from typing import TYPE_CHECKING, ClassVar, Literal, TypeVar, cast

from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected, PrivateValue, reject, require_count
from anonymizer.engine.graph_sdk._execution_dispatch import _mapping, _run_local
from anonymizer.engine.graph_sdk._execution_publication import _accept_invocation_frame
from anonymizer.engine.graph_sdk._execution_state import (
    _OperationJob,
    _SubmissionResults,
)
from anonymizer.engine.graph_sdk._execution_values import (
    DelegatedRequests,
    ImplementationHandle,
    LocalAssessmentResult,
    LocalCompleted,
    LocalFailure,
    LocalResult,
    OperationOutputKey,
    RuntimeOutcome,
    SubmissionFailure,
)
from anonymizer.engine.graph_sdk._foreign_transport import ForeignLoopTransport
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
    FailureClass,
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
from anonymizer.graph._values import ActivationKey, DatumId, InvocationId, TaskAttemptId
from anonymizer.graph.activation import CloseUnstarted, advance_activation

if TYPE_CHECKING:
    import pandas as pd

    from anonymizer.engine.graph_sdk._execution_runtime import _InvocationRuntime
    from anonymizer.engine.ndd._source_frame import WorkflowSourceFrame


_T = TypeVar("_T")
_WORKER_BORROW: ContextVar[_WorkerBorrow | None] = ContextVar("graph_worker_borrow", default=None)


def source_worker_borrow() -> _WorkerBorrow:
    """Recover the issued worker; source membership is checked on its owner loop."""
    borrow = _WORKER_BORROW.get()
    if type(borrow) is not _WorkerBorrow or borrow.expired:
        reject(EffectCode.FOREIGN_OWNER)
    return borrow


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

    @property
    def cancelled(self) -> bool:
        return self.expired or self.owner.cancelled

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

    async def authorize_source(self, association: SemanticAssociation) -> None:
        if _WORKER_BORROW.get() is not self or self.expired:
            reject(EffectCode.FOREIGN_OWNER)
        future = asyncio.run_coroutine_threadsafe(
            self.owner._worker_call(self, lambda: self.owner._authorize_source(association)), self.owner.loop
        )
        completion = asyncio.wrap_future(future)
        cancellation: asyncio.CancelledError | None = None
        while not completion.done():
            try:
                await asyncio.shield(completion)
            except asyncio.CancelledError as error:
                if cancellation is None:
                    cancellation = error
            except Exception:
                break
        if cancellation is not None:
            if not completion.cancelled():
                completion.exception()
            raise cancellation
        completion.result()

    def authorize_source_sync(self, association: SemanticAssociation) -> None:
        if _WORKER_BORROW.get() is not self or self.expired:
            reject(EffectCode.FOREIGN_OWNER)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            if loop is self.owner.loop:
                reject(EffectCode.UNSUPPORTED)
        asyncio.run_coroutine_threadsafe(
            self.owner._worker_call(self, lambda: self.owner._authorize_source(association)), self.owner.loop
        ).result()

    def require_transport(self, transport: object) -> None:
        if _WORKER_BORROW.get() is not self or self.expired:
            reject(EffectCode.FOREIGN_OWNER)
        self.owner._require_transport(transport)

    def bind_frame(
        self, dataframe: pd.DataFrame, *, targets: tuple[DatumId, ...] | None = None, source_limit: int | None = None
    ) -> WorkflowSourceFrame:
        if _WORKER_BORROW.get() is not self or self.expired:
            reject(EffectCode.FOREIGN_OWNER)
        return self.owner.runtime.frame.bind_source_frame(
            tuple(self.owner.jobs.values()), dataframe, targets=targets, source_limit=source_limit
        )

    async def request(
        self,
        slot: int,
        *,
        source_kind: ChildSourceKind = "initial",
        predecessor: PhysicalRequestId | None = None,
        evidence: ChildEvidence | None = None,
        association: SemanticAssociation | None = None,
    ) -> PhysicalRequestId:
        return await self._call(
            lambda: self.owner.request(
                slot, source_kind=source_kind, predecessor=predecessor, evidence=evidence, association=association
            )
        )

    async def enter_operation(self, slot: int, association: SemanticAssociation) -> AssociationInput | None:
        return await self._call(lambda: self.owner.enter_operation(slot, association))

    async def operation_result(self, slot: int, association: SemanticAssociation) -> AssociationResult | None:
        return await self._call(lambda: self.owner.operation_result(slot, association))

    async def ready_operations(self, association: SemanticAssociation) -> tuple[int, ...]:
        return await self._call(lambda: self.owner.ready_operations(association))

    async def complete_operation(self, association: SemanticAssociation, result: LocalResult) -> RuntimeOutcome:
        return await self._call(lambda: self.owner.complete_operation(association, result))

    async def readiness(self) -> PhysicalRequestId:
        return await self._call(self.owner.readiness)

    async def probe(self) -> None:
        await self._call(self.owner.probe)

    async def aparser_rejected(self, request: PhysicalRequestId) -> ChildEvidence:
        async def accept() -> ChildEvidence:
            return self.owner.parser_rejected(request)

        return await self._call(accept)

    def parser_rejected(self, request: PhysicalRequestId) -> ChildEvidence:
        self._check_origin()

        async def accept() -> ChildEvidence:
            return self.owner.parser_rejected(request)

        # D may wait for I here: I never waits synchronously for D or a worker.
        return asyncio.run_coroutine_threadsafe(self.owner._worker_call(self, accept), self.owner.loop).result()


class _DelegatedOwner(PrivateValue):
    """Borrow one invocation owner; never own a separate ledger or scheduler."""

    def __init__(
        self,
        runtime: _InvocationRuntime,
        jobs: tuple[_OperationJob[DatumId], ...],
        *,
        invocation_job: _OperationJob[InvocationId] | None = None,
    ) -> None:
        self.runtime = runtime
        self.jobs = {job.association: job for job in jobs}
        self.invocation_job = invocation_job
        self.policy = invocation_job.policy if invocation_job is not None else jobs[0].policy
        self.inputs = tuple(value for job in jobs for value in job.inputs)
        self.loop = asyncio.get_running_loop()
        self.closed = False
        self.child: asyncio.Task[PhysicalRequestId | None] | None = None
        self.cancel_requested = False
        self.request_rejected = False
        self.submission: asyncio.Task[LocalResult] | None = None
        self.worker_borrow: _WorkerBorrow | None = None
        self.worker_call: asyncio.Task[object] | None = None
        self.worker_calls: set[asyncio.Task[object]] = set()
        self.worker_gate = asyncio.Lock()
        contract = self.policy.delegated
        policy = self.policy.request
        assert contract is not None
        self.contract = contract
        self.chains: dict[tuple[SemanticAssociation, int], RequestChain] = {}
        for job in jobs if policy is not None else ():
            assert policy is not None
            runtime.request_authority.bind(
                RequestPolicyBinding.create(association=job.association, policies=frozenset({policy}))
            )
            for group in contract.generations:
                chain = RequestChain(
                    chain=RequestChainId.new(scope=runtime.request_authority.state.scope),
                    associations=frozenset({job.association}),
                    policy=policy,
                    role="generation",
                    routes=contract.routes,
                    followups=contract.followups,
                )
                runtime.request_authority.bind_chain(chain)
                for slot in group:
                    self.chains[job.association, slot] = chain

    def require_transport(self, transport: object) -> None:
        self._check_submission()
        self._require_transport(transport)

    def _require_transport(self, transport: object) -> None:
        if self.closed:
            reject(EffectCode.FOREIGN_OWNER)
        slots = self.contract.child_slots + (
            () if self.contract.readiness is None else (self.contract.readiness.request_slot,)
        )
        for slot in slots:
            node = self.runtime.prepared.reservation_recipe[slot].template
            implementation = self.runtime.policies[node].implementations[0]
            handle = self.runtime.handles[
                implementation.implementation, implementation.capability.operation, implementation.configuration
            ]
            if type(handle.transport) is not ForeignLoopTransport or not handle.transport.owns(transport):
                reject(EffectCode.FOREIGN_OWNER)

    def bind_frame(
        self, dataframe: pd.DataFrame, *, targets: tuple[DatumId, ...] | None = None, source_limit: int | None = None
    ) -> WorkflowSourceFrame:
        self._check_submission()
        return self.runtime.frame.bind_source_frame(
            tuple(self.jobs.values()), dataframe, targets=targets, source_limit=source_limit
        )

    async def _authorize_source(self, association: SemanticAssociation) -> None:
        self._check_submission()
        if type(association) is not SemanticAssociation:
            reject(EffectCode.FOREIGN_OWNER)
        try:
            self._source_job(association)
        except (AttributeError, TypeError):
            reject(EffectCode.FOREIGN_OWNER)

    def _source_job(self, association: SemanticAssociation | None = None) -> _OperationJob[DatumId]:
        if association is None:
            if len(self.jobs) != 1:
                reject(EffectCode.MISSING)
            return next(iter(self.jobs.values()))
        if association not in self.jobs:
            reject(EffectCode.FOREIGN_OWNER)
        return self.jobs[association]

    def _check(self) -> None:
        if self.closed or asyncio.get_running_loop() is not self.loop:
            reject(EffectCode.UNSUPPORTED)

    async def aparser_rejected(self, request: PhysicalRequestId) -> ChildEvidence:
        return self.parser_rejected(request)

    def parser_rejected(self, request: PhysicalRequestId) -> ChildEvidence:
        self._check()
        task = asyncio.current_task()
        if task is not self.submission and (self.worker_call is None or task is not self.worker_call):
            reject(EffectCode.UNSUPPORTED)
        reservation = next(
            (item for item in self.runtime.request_authority.state.dispatches if item.request == request), None
        )
        if (
            reservation is None
            or reservation.child is None
            or reservation.child.chain not in {chain.chain for chain in self.chains.values()}
        ):
            reject(EffectCode.FOREIGN_OWNER)
        evidence = ChildEvidence(chain=reservation.child.chain, request=request, kind="parser_rejected")
        self.runtime.request_authority.apply(AcceptChildEvidence(evidence=evidence))
        return evidence

    async def enter_operation(self, slot: int, association: SemanticAssociation) -> AssociationInput | None:
        """Enter an admitted semantic child without inventing a physical request."""
        self._check_submission()
        require_count(slot)
        source_job = self._source_job(association)
        source = self.runtime.frame.sources[source_job.scope]
        key = source.keys.get(slot)
        entry = next((row for row in source.state.entries if row.activation == key), None)
        if entry is None:
            reservation = self.runtime.prepared.reservation_recipe[slot] if key is not None else None
            if reservation is not None and key is not None:
                scope = next(
                    scope
                    for scope in source.state.workflow.bodies()
                    if any(node.id == reservation.template for node in scope.operations)
                )
                for choice in scope.choices:
                    branch = next(
                        (branch for branch in choice.branches if reservation.template in branch.members), None
                    )
                    selector = next(
                        (
                            row
                            for row in source.state.entries
                            if row.template == choice.selector and row.activation.parent == key.parent
                        ),
                        None,
                    )
                    if (
                        branch is not None
                        and selector is not None
                        and selector.outcome is not None
                        and selector.outcome not in branch.outcomes
                    ):
                        return None
                parent = key
                while parent.parent is not None:
                    ancestor = parent.parent
                    expansion = next((item for item in source.state.expansions if item.parent == ancestor), None)
                    if expansion is not None and expansion.status == "closed" and parent not in expansion.members:
                        return None
                    parent = parent.parent
            reject(EffectCode.FOREIGN_OWNER)
        if entry.activation == source_job.activation:
            if entry.status == "success":
                return None
            if entry.status != "running":
                reject(EffectCode.REQUEST_CAUSALITY)
            return source_job.inputs[0]
        if self.runtime.policies[entry.template].submission_owner != self.policy.node:
            reject(EffectCode.FOREIGN_OWNER)
        active = self.runtime.operations.get(entry.activation)
        if active is not None and entry.status == "running":
            return active.inputs[0]
        if entry.status != "ready" or key in self.runtime.operations:
            reject(EffectCode.REQUEST_CAUSALITY)
        job = self.runtime._bind_operation(source_job.scope, entry, self.runtime.policies[entry.template])
        self.runtime._prepare_job(job)()
        self.runtime._start_observed_activation(job.activation, operation=True)
        return job.inputs[0]

    async def operation_result(self, slot: int, association: SemanticAssociation) -> AssociationResult | None:
        """Read an owned completed occurrence from canonical artifacts, without issuing another attempt."""
        self._check_submission()
        require_count(slot)
        source_job = self._source_job(association)
        source = self.runtime.frame.sources[source_job.scope]
        key = source.keys.get(slot)
        if key is None:
            reject(EffectCode.FOREIGN_OWNER)
        node = self.runtime.prepared.reservation_recipe[slot].template
        policy = self.runtime.policies.get(node)
        if policy is None or policy.submission_owner != self.policy.node:
            reject(EffectCode.FOREIGN_OWNER)
        return self.runtime._operation_result(source_job.scope, key)

    async def ready_operations(self, association: SemanticAssociation) -> tuple[int, ...]:
        """Expose only ready child occurrences belonging to this live submission."""
        self._check_submission()
        source = self.runtime.frame.sources[self._source_job(association).scope]
        ready = {
            entry.activation
            for entry in source.state.entries
            if entry.status == "ready"
            and (policy := self.runtime.policies.get(entry.template)) is not None
            and policy.submission_owner == self.policy.node
        }
        return tuple(slot for slot, key in source.keys.items() if key in ready)

    async def complete_operation(self, association: SemanticAssociation, result: LocalResult) -> RuntimeOutcome:
        self._check_submission()
        if type(association) is not SemanticAssociation:
            reject(EffectCode.FOREIGN_OWNER)
        try:
            if not isinstance(association.task, TaskAttemptId) or not isinstance(
                association.task.activation, ActivationKey
            ):
                reject(EffectCode.FOREIGN_OWNER)
            bound = self.runtime.operations.get(association.task.activation)
        except (AttributeError, TypeError):
            reject(EffectCode.FOREIGN_OWNER)
        if bound is None or bound.association != association or not isinstance(bound.scope, DatumId):
            reject(EffectCode.FOREIGN_OWNER)
        job = cast(_OperationJob[DatumId], bound)
        if association not in self.jobs and (
            job.policy.submission_owner != self.policy.node
            or job.scope not in {row.scope for row in self.jobs.values()}
        ):
            reject(EffectCode.FOREIGN_OWNER)
        entry = next(
            row for row in self.runtime.frame.sources[job.scope].state.entries if row.activation == job.activation
        )
        if entry.status != "running":
            reject(EffectCode.REQUEST_CAUSALITY)
        results: tuple[AssociationResult, ...] = ()
        assessments: tuple[LocalAssessmentResult, ...] = ()
        if type(result) is LocalFailure:
            mapping = _mapping(job.policy, "failure", None, inspect_failure(result.failure))
        elif type(result) is LocalCompleted:
            invalid = inspect_associations(
                result.results,
                frozenset({association}),
                self.runtime.admitted._response_limits(),
                assessments=result.assessments,
            )
            if invalid is None or invalid or result.frame is not None or result.failed_sources:
                mapping = _mapping(job.policy, "failure", None, "malformed_response")
            elif len(result.results) != 1 or result.results[0].outcome not in job.policy.result_outcomes:
                mapping = _mapping(job.policy, "failure", None, "malformed_response")
            else:
                results, assessments = result.results, result.assessments
                mapping = _mapping(job.policy, "result", results[0].outcome, None)
        else:
            mapping = _mapping(job.policy, "failure", None, "malformed_response")
        final = self.runtime._complete_job(job, mapping, results, assessments)
        self.runtime.control.scheduler_changed.set()
        return final

    async def request(
        self,
        slot: int,
        *,
        source_kind: ChildSourceKind = "initial",
        predecessor: PhysicalRequestId | None = None,
        evidence: ChildEvidence | None = None,
        association: SemanticAssociation | None = None,
    ) -> PhysicalRequestId:
        try:
            if slot not in self.contract.child_slots:
                reject(EffectCode.FOREIGN_OWNER)
            if self.contract.optional_tail:
                initial, correction = self.contract.child_slots
                if (slot == initial and source_kind != "initial") or (
                    slot == correction and source_kind != "correction"
                ):
                    reject(EffectCode.REQUEST_CAUSALITY)
            readiness = self.contract.readiness
            if readiness is not None and (
                not self._satisfied(readiness.owner_slot, frozenset({readiness.ready_outcome}))
                or not self._satisfied(readiness.probe_slot, readiness.probe_outcomes)
            ):
                reject(EffectCode.REQUEST_CAUSALITY)
            self._check_submission()
            job = self._source_job(association)
            result = await self._borrow(
                self._request(
                    slot, source_kind, predecessor, evidence, self.chains[job.association, slot], job.association
                )
            )
            assert result is not None
            return result
        except EffectRejected:
            self.request_rejected = True
            raise

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
        if (
            len(self.jobs) == 1
            and not self.contract.generation_groups
            and (self.worker_call is not None or self.active)
        ):
            reject(EffectCode.REQUEST_CAUSALITY)
        task = asyncio.current_task()
        assert task is not None
        self.worker_calls.add(task)
        try:
            async with self.worker_gate:
                self._check()
                if borrow.expired or self.active or self.cancelled:
                    reject(EffectCode.REQUEST_CAUSALITY)
                self.worker_call = task
                try:
                    return await operation()
                finally:
                    self.worker_call = None
        finally:
            self.worker_calls.discard(task)

    async def _submit_worker(self, adapter: WorkerDelegatedSubmitter) -> LocalResult:
        borrow = _WorkerBorrow(self, adapter)
        self.worker_borrow = borrow
        context = copy_context()
        context.run(_WORKER_BORROW.set, borrow)
        invocation = _BlockingInvocation(adapter, lambda: context.run(adapter._callback, self.inputs, borrow))
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
        source = self.runtime.frame.invocation_source or self.runtime.frame.sources[self._source_job().scope]
        key = source.keys[slot]
        return any(
            entry.activation == key and entry.status == "success" and entry.outcome in outcomes
            for entry in source.state.entries
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
        self.runtime._prepare_job(semantic)()
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
            source = self.runtime.frame.invocation_source or self.runtime.frame.sources[self._source_job().scope]
            key = source.keys[descriptor.request_slot]
            entry = next(item for item in source.state.entries if item.activation == key)
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
        self.runtime._prepare_job(job)()
        self.runtime._start_observed_activation(job.activation, operation=True)
        implementation = job.implementation
        handle = self.runtime.handles[
            implementation.implementation, implementation.capability.operation, implementation.configuration
        ]
        try:
            returned = await _run_local(
                job.policy, handle, job.association, job.inputs, self, self.runtime.admitted._response_limits()
            )
            assert isinstance(returned, tuple)
            mapping, results, assessments = returned
        except asyncio.CancelledError:
            mapping, results, assessments = _mapping(job.policy, "cancel_after_start", None, None), (), ()
        self._finish(job, mapping, results, assessments)

    @property
    def active(self) -> bool:
        return self.child is not None and not self.child.done()

    @property
    def cancelled(self) -> bool:
        return self.cancel_requested or self.runtime.control.cancelled

    def _job(self, slot: int, association: SemanticAssociation | None = None) -> _OperationJob[DatumId | InvocationId]:
        require_count(slot)
        if slot not in self.contract.owned_slots:
            reject(EffectCode.FOREIGN_OWNER)
        runtime = self.runtime
        shared = runtime.frame.invocation_source
        source = (
            shared
            if shared is not None and slot in shared.keys
            else runtime.frame.sources[self._source_job(association).scope]
        )
        key = source.keys[slot]
        entry = next(item for item in source.state.entries if item.activation == key)
        if key in runtime.operations:
            reject(EffectCode.DUPLICATE)
        if entry.status != "ready":
            reject(EffectCode.REQUEST_CAUSALITY)
        if self.cancelled:
            reject(EffectCode.REQUEST_CAUSALITY)
        return runtime._bind_operation(
            runtime.invocation if source is shared else self._source_job(association).scope,
            entry,
            runtime.policies[entry.template],
        )

    async def _request(
        self,
        slot: int,
        source_kind: ChildSourceKind,
        predecessor: PhysicalRequestId | None,
        evidence: ChildEvidence | None,
        chain: RequestChain,
        association: SemanticAssociation | None = None,
    ) -> PhysicalRequestId:
        runtime = self.runtime
        if (
            runtime.services.limits.max_remote_outstanding is not None
            and len(runtime.request_authority.state.remote_outstanding)
            >= runtime.services.limits.max_remote_outstanding
        ):
            reject(EffectCode.LIMIT_EXCEEDED)
        leaf = self._job(slot, association)
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

        foreign = transport._invocation(envelope) if type(transport) is ForeignLoopTransport else None
        if handle.resource is not None:
            runtime.request_resources[request] = handle.resource.resource

        def prepare() -> Callable[[], None]:
            commit = runtime._prepare_job(leaf)

            def start() -> None:
                commit()
                if foreign is not None:
                    runtime._start_observed_activation(leaf.activation, operation=True)

            return start

        completion = await drive_request(
            authority,
            request,
            self,
            invoke,
            transport.cancel,
            prepare_dispatch=prepare,
            foreign=foreign,
            drain_on_cancel=True,
        )
        if request not in authority.state.dispatched:
            reject(EffectCode.REQUEST_CAUSALITY)
        value = completion.value
        results: tuple[AssociationResult, ...] = ()
        if completion.cancelled:
            mapping = _mapping(policy, "cancel_after_dispatch" if completion.stop_confirmed else "lost", None, None)
        elif type(value) is TransportSuccess:
            invalid = inspect_associations(value.results, chain.associations, runtime.admitted._response_limits())
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
        job: _OperationJob[DatumId | InvocationId],
        mapping: RuntimeOutcome,
        results: tuple[AssociationResult, ...],
        assessments: tuple[LocalAssessmentResult, ...] = (),
    ) -> None:
        self.runtime._complete_job(job, mapping, results, assessments)
        self.runtime.control.scheduler_changed.set()

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

    def close(self) -> frozenset[SemanticAssociation]:
        self.closed = True
        incomplete: set[SemanticAssociation] = set()
        targets = {job.scope for job in self.jobs.values()}
        for operation in self.runtime.operations.values():
            if operation.policy.submission_owner != self.policy.node or operation.scope not in targets:
                continue
            source = self.runtime.frame.occurrence(operation.scope)
            if not any(
                row.activation == operation.activation and row.status == "running" for row in source.state.entries
            ):
                continue
            self.runtime._complete_job(
                operation,
                _mapping(
                    operation.policy,
                    "cancel_after_start" if self.cancelled else "failure",
                    None,
                    None if self.cancelled else "implementation_exception",
                ),
                (),
            )
            incomplete.update(job.association for job in self.jobs.values() if job.scope == operation.scope)
        owned = {
            policy.node for policy in self.runtime.policies.values() if policy.submission_owner == self.policy.node
        }
        for job in self.jobs.values():
            source = self.runtime.frame.sources[job.scope]
            for previous in tuple(source.state.entries):
                entry = next(row for row in source.state.entries if row.activation == previous.activation)
                if entry.template in owned and entry.status != "success":
                    incomplete.add(job.association)
                    if entry.status in {"ready", "unstarted"}:
                        source.state = advance_activation(
                            state=source.state, event=CloseUnstarted(activation=entry.activation, category="blocked")
                        )
            unused = {} if self.cancelled or self.request_rejected else self.runtime._unused_tails(job)
            for slot in (
                self.contract.child_slots
                if self.runtime.frame.invocation_source is not None
                else self.contract.owned_slots
            ):
                key = self.runtime.frame.sources[job.scope].keys[slot]
                state = self.runtime.frame.sources[job.scope].state
                entry = next(item for item in state.entries if item.activation == key)
                if entry.status in {"ready", "unstarted"}:
                    if key not in unused:
                        incomplete.add(job.association)
                    self.runtime.frame.sources[job.scope].state = advance_activation(
                        state=state, event=CloseUnstarted(activation=key, category="blocked")
                    )
        source = self.runtime.frame.invocation_source
        if source is not None:
            for entry in tuple(source.state.entries):
                if entry.status in {"ready", "unstarted"}:
                    incomplete.update(self.jobs)
                    source.state = advance_activation(
                        state=source.state, event=CloseUnstarted(activation=entry.activation, category="blocked")
                    )
        return frozenset(incomplete)

    def _failure(self, condition: str, failure: FailureClass | None = None) -> _SubmissionResults:
        associations = self.jobs if self.invocation_job is None else (self.invocation_job.association,)
        return {association: (_mapping(self.policy, condition, None, failure), (), ()) for association in associations}

    async def run(self, handle: ImplementationHandle) -> _SubmissionResults:
        assert handle.delegated is not None
        runtime = self.runtime
        task = asyncio.create_task(
            self._submit_worker(handle.delegated)
            if type(handle.delegated) is WorkerDelegatedSubmitter
            else handle.delegated.submit(self.inputs, self)
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
            return self._failure("cancel_after_start")
        except Exception as error:
            runtime.submission_failures.append(SubmissionFailure(associations=frozenset(self.jobs), cause=error))
            return self._failure("failure", "implementation_exception")
        finally:
            if self.child is not None:
                await self._drain(self.child)
            await self._drain(task)
            for worker_call in tuple(self.worker_calls):
                await self._drain(worker_call)
            if self.child is not None:
                await self._drain(self.child)
            incomplete = self.close()
        if self.contract.mode == "opaque" and self.cancelled:
            return self._failure("cancel_after_start")
        if type(result) is LocalFailure:
            return self._failure("failure", inspect_failure(result.failure))
        from anonymizer.engine.detection._frame_publication import InvocationFrame

        if (
            self.contract.mode == "opaque"
            and self.invocation_job is not None
            and type(getattr(result, "frame", None)) is InvocationFrame
        ):
            accepted = _accept_invocation_frame(result)
            if accepted is None:
                return self._failure("failure", "malformed_response")
            runtime.frame.publications.append(accepted)
            return {self.invocation_job.association: (_mapping(self.policy, "result", "ok", None), (), ())}
        if self.invocation_job is not None and self.contract.mode != "opaque":
            return self._failure("failure", "malformed_response")
        if type(result) is not LocalCompleted:
            return self._failure("failure", "malformed_response")
        failed_sources = getattr(result, "failed_sources", None)
        if (
            type(failed_sources) is not frozenset
            or any(type(source) is not SemanticAssociation for source in failed_sources)
            or not failed_sources <= self.jobs.keys()
        ):
            return self._failure("failure", "malformed_response")
        declared_drops = failed_sources
        invalid = inspect_associations(
            result.results,
            frozenset(self.jobs) - failed_sources,
            runtime.admitted._response_limits(),
            assessments=result.assessments,
        )
        if (
            invalid is None
            or invalid
            or any(item.outcome not in self.policy.result_outcomes for item in result.results)
        ):
            return self._failure("failure", "malformed_response")
        if any(policy.submission_owner == self.policy.node for policy in runtime.policies.values()):
            for item in result.results:
                assert isinstance(item.association, SemanticAssociation)
                job = self.jobs[item.association]
                entry = next(
                    row for row in runtime.frame.sources[job.scope].state.entries if row.activation == job.activation
                )
                if entry.status == "success" and (
                    item.outputs or any(value.association == item.association for value in result.assessments)
                ):
                    return self._failure("failure", "malformed_response")
        if any(policy.submission_owner == self.policy.node for policy in runtime.policies.values()):
            failed_sources = failed_sources | incomplete
        successful = tuple(
            item
            for item in result.results
            if item.association not in failed_sources
            and _mapping(self.policy, "result", item.outcome, None).category == "success"
        )
        for item in successful:
            assert isinstance(item.association, SemanticAssociation)
            selection = self.contract.selection
            if selection is None:
                if item.association in incomplete:
                    return self._failure("failure", "implementation_exception")
                continue
            outputs = [output for output in item.outputs if output.port == selection.port]
            try:
                if len(outputs) != 1:
                    return self._failure("failure", "malformed_response")
                omitted = selection.omitted(outputs[0].value)
            except EffectRejected:
                return self._failure("failure", "malformed_response")
            job = self.jobs[item.association]
            source = runtime.frame.sources[job.scope]
            missing = {slot for slot in self.contract.child_slots if source.keys[slot] not in runtime.operations}
            retry_unused = {key.occurrence for key in runtime._unused_tails(job)}
            missing -= {slot for slot in missing if source.keys[slot].occurrence in retry_unused}
            if missing != set(omitted):
                return self._failure("failure", "implementation_exception")
            readiness = self.contract.readiness
            if readiness is not None:
                ready_source = runtime.frame.invocation_source or source
                required = {ready_source.keys[slot] for slot in readiness.slots}
                if any(
                    entry.status != "success" for entry in ready_source.state.entries if entry.activation in required
                ):
                    return self._failure("failure", "implementation_exception")
        frame = getattr(result, "frame", False)
        if frame is not None:
            from anonymizer.engine.detection._frame_publication import FramePublication

            frame_results = successful
            if any(policy.submission_owner == self.policy.node for policy in runtime.policies.values()):
                from anonymizer.engine.graph_sdk._execution_publication import _final_outputs

                targets = {self.jobs[cast(SemanticAssociation, item.association)].scope for item in successful}
                finals = tuple(
                    item
                    for item in _final_outputs(runtime.prepared, runtime.frame.states, runtime.facts)
                    if item.target in targets
                )
                if len(finals) != len(targets) or {item.target for item in finals} != targets:
                    return self._failure("failure", "malformed_response")
                semantic_results = []
                for item in finals:
                    producer = item.producer
                    if not isinstance(producer, OperationOutputKey) or producer.activation not in runtime.operations:
                        return self._failure("failure", "malformed_response")
                    entry = next(
                        row
                        for row in runtime.frame.sources[item.target].state.entries
                        if row.activation == producer.activation
                    )
                    if runtime.policies[entry.template].submission_owner != self.policy.node:
                        return self._failure("failure", "malformed_response")
                    accepted = runtime._operation_result(item.target, producer.activation, port=producer.port)
                    if accepted is None:
                        return self._failure("failure", "malformed_response")
                    semantic_results.append(accepted)
                frame_results = tuple(semantic_results)

            try:
                accepted_frame = (
                    frame.accept(
                        frame_results,
                        opaque=tuple(
                            value
                            for job in self.jobs.values()
                            for value in runtime.frame.sources[job.scope].opaque_cells
                        ),
                    )
                    if type(frame) is FramePublication
                    else None
                )
            except (AttributeError, TypeError, ValueError):
                accepted_frame = None
            if accepted_frame is None:
                return self._failure("failure", "malformed_response")
            runtime.frame.publications.append(accepted_frame)
        if self.invocation_job is not None:
            return {self.invocation_job.association: (_mapping(self.policy, "result", "ok", None), (), ())}
        failed_results = {}
        for association in failed_sources:
            job = self.jobs[association]
            entry = next(
                row for row in runtime.frame.sources[job.scope].state.entries if row.activation == job.activation
            )
            if association in declared_drops and entry.status == "success":
                # A later row drop does not undo an already accepted prefix.
                # Descendants were closed above; no final output may survive it.
                from anonymizer.engine.graph_sdk._execution_publication import _final_outputs

                if any(
                    item.target == job.scope
                    for item in _final_outputs(runtime.prepared, runtime.frame.states, runtime.facts)
                ):
                    return self._failure("failure", "malformed_response")
                failed_results[association] = (_mapping(self.policy, "result", entry.outcome, None), (), ())
            else:
                failed_results[association] = (
                    _mapping(self.policy, "failure", None, "implementation_exception"),
                    (),
                    (),
                )
        return {
            **failed_results,
            **{
                item.association: (
                    _mapping(self.policy, "result", item.outcome, None),
                    (item,),
                    tuple(value for value in result.assessments if value.association == item.association),
                )
                for item in result.results
                if isinstance(item.association, SemanticAssociation) and item.association not in failed_sources
            },
        }
