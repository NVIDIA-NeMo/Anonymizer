# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Restricted invocation-owned borrowing of registered physical operation slots."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Coroutine
from concurrent.futures import Future
from contextlib import suppress
from contextvars import ContextVar, copy_context
from dataclasses import replace
from threading import get_ident
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeVar, cast

from anonymizer.engine.graph_sdk._activation_reducer import advance_activation
from anonymizer.engine.graph_sdk._activation_values import CloseUnstarted
from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected, PrivateValue, reject, require_count
from anonymizer.engine.graph_sdk._execution_dispatch import _mapping, _run_local
from anonymizer.engine.graph_sdk._execution_publication import (
    _accept_invocation_frame,
    _aggregate_frame,
    _final_outputs,
    _is_invocation_frame,
    _is_publication_frame,
)
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
from anonymizer.engine.graph_sdk._graph_values import DatumId, InvocationId
from anonymizer.engine.graph_sdk._ingress import inspect_associations, inspect_failure
from anonymizer.engine.graph_sdk._request_driver import await_completion_or_cancel, drive_request
from anonymizer.engine.graph_sdk._workflow_values import SubgraphNode
from anonymizer.engine.graph_sdk.blocking import (
    _NOT_STARTED,
    BlockingAffinity,
    BlockingPool,
    _BlockingAdapter,
    _BlockingInvocation,
)
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
    RequestPurpose,
    Reserve,
    SemanticAssociation,
    TransportFailure,
    TransportSuccess,
)

if TYPE_CHECKING:
    import pandas as pd

    from anonymizer.engine.graph_sdk._execution_runtime import _InvocationRuntime
    from anonymizer.engine.ndd._source_frame import WorkflowSourceFrame


_T = TypeVar("_T")
_WORKER_BORROW: ContextVar[_WorkerBorrow | None] = ContextVar("graph_worker_borrow", default=None)


class _GateHold:
    """One worker call's hold on the owner gate; released once, by whichever side finishes bookkeeping first."""

    __slots__ = ("concurrent", "gate", "held")

    def __init__(self, gate: asyncio.Lock, *, concurrent: bool) -> None:
        self.gate = gate
        self.concurrent = concurrent
        self.held = True

    def release(self) -> None:
        if self.held:
            self.held = False
            self.gate.release()


class _Failed(Exception):
    """A stage rejected the response; ``args`` are ``_failure``'s (condition, failure) for every association."""


_GATE_HOLD: ContextVar[_GateHold | None] = ContextVar("graph_worker_gate_hold", default=None)
# A conversation restart is a correction on its chain; retries and failovers keep their own purpose.
_CHILD_PURPOSE: dict[ChildSourceKind, RequestPurpose] = {
    "initial": "initial",
    "retry": "retry",
    "correction": "correction",
    "conversation_restart": "correction",
    "failover": "failover",
}


def drive(operation: Coroutine[object, object, _T]) -> _T:
    """Run delegated bookkeeping to completion on the calling thread.

    Owner bookkeeping never awaits anything that can suspend; this enforces it at runtime instead of
    hopping to the owner loop and waiting for it.
    """
    try:
        operation.send(None)
    except StopIteration as done:
        return cast(_T, done.value)
    operation.close()
    raise RuntimeError("delegated graph bookkeeping suspended")


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

    @property
    def released(self) -> bool:
        """Whether this submitter or its pool has stopped admitting work."""
        return self.closed or self.pool.closed

    async def submit(self, inputs: tuple[AssociationInput, ...], requests: DelegatedRequests) -> LocalResult:
        # Only the invocation owner may issue the authenticated worker borrow.
        reject(EffectCode.UNSUPPORTED)


class CallerChannel:
    """The synchronous caller's queue, seen from the host loop: ``channel(call)`` runs ``call`` on that thread."""

    __slots__ = ("_put",)

    def __init__(self, put: Callable[[Callable[[], Any]], Future[Any]]) -> None:
        self._put = put

    def __call__(self, call: Callable[[], Any]) -> Future[Any]:
        return self._put(call)


SubmitterPool = BlockingPool | CallerChannel


class CallerThreadSubmitter(PrivateValue):
    """Run the synchronous body on the thread that serves ``channel``, with the borrowed loop as host loop.

    ``channel(call)`` queues ``call`` for that thread and returns a future for its result. The submission
    keeps ownership until the queued call has actually returned, also when the observer is cancelled.
    """

    execution: ClassVar[Literal["worker"]] = "worker"

    def __init__(
        self,
        *,
        callback: Callable[[tuple[AssociationInput, ...], DelegatedRequests], LocalResult],
        channel: CallerChannel,
        borrowed_loop: asyncio.AbstractEventLoop,
    ) -> None:
        if not callable(callback) or inspect.iscoroutinefunction(callback) or not isinstance(channel, CallerChannel):
            reject(EffectCode.INVALID_TYPE)
        if not isinstance(borrowed_loop, asyncio.AbstractEventLoop):
            reject(EffectCode.INVALID_TYPE)
        self._callback = callback
        self._channel = channel
        self._borrowed_loop = borrowed_loop
        self.closed = False

    @property
    def borrowed_loop(self) -> asyncio.AbstractEventLoop:
        return self._borrowed_loop

    @property
    def released(self) -> bool:
        return self.closed

    async def submit(self, inputs: tuple[AssociationInput, ...], requests: DelegatedRequests) -> LocalResult:
        # Only the invocation owner may issue the authenticated worker borrow.
        reject(EffectCode.UNSUPPORTED)

    async def _run(self, callback: Callable[[], object], before: Callable[[], bool]) -> object:
        completion = asyncio.wrap_future(self._channel(lambda: callback() if before() else _NOT_STARTED))
        try:
            return await asyncio.shield(completion)
        except asyncio.CancelledError:
            while not completion.done():
                try:
                    await asyncio.shield(completion)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            return completion.result()


DELEGATED_SUBMITTERS = (WorkerDelegatedSubmitter, CallerThreadSubmitter)


def delegated_submitter_of(value: object) -> WorkerDelegatedSubmitter | CallerThreadSubmitter | None:
    """``value`` when it is exactly one of the owner-borrowing submitters, else None."""
    return (
        cast(WorkerDelegatedSubmitter | CallerThreadSubmitter, value) if type(value) in DELEGATED_SUBMITTERS else None
    )


class _WorkerBorrow(PrivateValue):
    def __init__(self, owner: _DelegatedOwner, adapter: WorkerDelegatedSubmitter | CallerThreadSubmitter) -> None:
        self.owner = owner
        self.adapter = adapter
        self.loop = adapter.borrowed_loop
        self.transport_loop: asyncio.AbstractEventLoop | None = None
        self.expired = False

    @property
    def cancelled(self) -> bool:
        return self.expired or self.owner.cancelled

    def _require_borrow(self) -> None:
        if _WORKER_BORROW.get() is not self or self.expired:
            reject(EffectCode.FOREIGN_OWNER)

    def _check_origin(self) -> None:
        """Async callbacks run on the borrowed loop or the transport's loop."""
        self._require_borrow()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            reject(EffectCode.UNSUPPORTED)
        if loop is not self.loop and loop is not self.transport_loop:
            reject(EffectCode.UNSUPPORTED)

    def _drive(self, operation: Callable[[], Coroutine[object, object, _T]]) -> _T:
        """Bookkeeping from synchronous code: driven here, or on the owner loop while physical children may run."""
        self._require_borrow()
        if self.owner.direct:
            return self.owner._drive_call(self, operation)
        if asyncio._get_running_loop() is self.owner.loop:
            reject(EffectCode.UNSUPPORTED)  # Waiting here would block the owner loop it waits for.
        return asyncio.run_coroutine_threadsafe(self.owner._worker_call(self, operation), self.owner.loop).result()

    async def _call(self, operation: Callable[[], Coroutine[object, object, _T]], *, concurrent: bool = False) -> _T:
        if not concurrent and self.owner.direct:
            return self._drive(operation)
        self._check_origin()
        future = asyncio.run_coroutine_threadsafe(
            self.owner._worker_call(self, operation, concurrent=concurrent), self.owner.loop
        )
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
        if self.owner.direct:
            return self.authorize_source_sync(association)
        self._require_borrow()
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
        self._drive(lambda: self.owner._authorize_source(association))

    def require_transport(self, transport: object) -> None:
        self._require_borrow()
        self.owner._require_transport(transport)
        self.transport_loop = cast(asyncio.AbstractEventLoop, getattr(transport, "loop"))

    def bind_frame(
        self, dataframe: pd.DataFrame, *, targets: tuple[DatumId, ...] | None = None, source_limit: int | None = None
    ) -> WorkflowSourceFrame:
        self._require_borrow()
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
            ),
            concurrent=True,
        )

    async def enter_operation(self, slot: int, association: SemanticAssociation | None) -> AssociationInput | None:
        return await self._call(lambda: self.owner.enter_operation(slot, association))

    async def operation_result(self, slot: int, association: SemanticAssociation) -> AssociationResult | None:
        return await self._call(lambda: self.owner.operation_result(slot, association))

    async def ready_operations(self, association: SemanticAssociation | None) -> tuple[int, ...]:
        return await self._call(lambda: self.owner.ready_operations(association))

    async def complete_operation(self, association: SemanticAssociation, result: LocalResult) -> RuntimeOutcome:
        return await self._call(lambda: self.owner.complete_operation(association, result))

    async def ready_operations_many(self, associations: tuple[SemanticAssociation, ...]) -> tuple[tuple[int, ...], ...]:
        return await self._call(lambda: self.owner.ready_operations_many(associations))

    async def enter_operations(
        self, slot: int, associations: tuple[SemanticAssociation, ...]
    ) -> tuple[AssociationInput | None, ...]:
        return await self._call(lambda: self.owner.enter_operations(slot, associations))

    async def operation_results(
        self, slot: int, associations: tuple[SemanticAssociation, ...]
    ) -> tuple[AssociationResult | None, ...]:
        return await self._call(lambda: self.owner.operation_results(slot, associations))

    async def complete_operations(
        self, results: tuple[tuple[SemanticAssociation, LocalResult], ...]
    ) -> tuple[RuntimeOutcome, ...]:
        return await self._call(lambda: self.owner.complete_operations(results))

    async def close_observed_requests(
        self, associations: tuple[SemanticAssociation, ...], generated: frozenset[SemanticAssociation]
    ) -> None:
        return await self._call(lambda: self.owner.close_observed_requests(associations, generated))

    def authorize_sources_sync(self, associations: tuple[SemanticAssociation, ...]) -> None:
        async def authorize() -> None:
            for association in associations:
                await self.owner._authorize_source(association)

        self._drive(authorize)

    async def readiness(self) -> PhysicalRequestId:
        return await self._call(self.owner.readiness)

    async def probe(self) -> None:
        await self._call(self.owner.probe)

    async def aparser_rejected(self, request: PhysicalRequestId) -> ChildEvidence:
        async def accept() -> ChildEvidence:
            return self.owner.parser_rejected(request)

        return await self._call(accept, concurrent=True)

    def parser_rejected(self, request: PhysicalRequestId) -> ChildEvidence:
        self._check_origin()

        async def accept() -> ChildEvidence:
            return self.owner.parser_rejected(request)

        # D may wait for I here: I never waits synchronously for D or a worker. Bookkeeping only, so it does
        # not wait for in-flight physical children.
        return asyncio.run_coroutine_threadsafe(
            self.owner._worker_call(self, accept, concurrent=True), self.owner.loop
        ).result()


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
        # The thread currently driving bookkeeping synchronously under the runtime lock, if any, and whether
        # that call changed the ledger in a way the scheduler must see.
        self.driving: int | None = None
        self.wake_pending = False
        # Borrowed children in flight, each keyed by its request chain (None: readiness or probe).
        self.children: dict[asyncio.Task[PhysicalRequestId | None], RequestChainId | None] = {}
        self.children_idle = asyncio.Event()
        self.children_idle.set()
        self.children_settled = asyncio.Event()
        self.cancel_requested = False
        self.request_rejected = False
        self.submission: asyncio.Task[LocalResult] | None = None
        self.worker_borrow: _WorkerBorrow | None = None
        # Worker calls that passed the gate checks and have not returned.
        self.entered_calls: set[asyncio.Task[object]] = set()
        self.worker_calls: set[asyncio.Task[object]] = set()
        # Serializes callback bookkeeping. A physical request releases it once reserved, so requests on
        # different chains are in flight together; other callbacks wait for in-flight children first.
        self.worker_gate = asyncio.Lock()
        contract = self.policy.delegated
        assert contract is not None
        self.contract = contract
        self.cleanup_contract = contract.snapshot()
        policy = contract.observed_request or self.policy.request
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
        self.own_chains = frozenset(chain.chain for chain in self.chains.values())
        # Without physical children nothing can be in flight beside the body, so its bookkeeping needs no
        # owner-loop gate: it is driven synchronously on the calling thread (every facade invocation).
        self.direct = not contract.child_slots and contract.readiness is None
        self.owns_children = any(item.submission_owner == self.policy.node for item in runtime.policies.values())

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
        # An exact SemanticAssociation holds an exact TaskAttemptId: lookup hashes only its engine token.
        self._source_job(association)

    def _source_job(self, association: SemanticAssociation | None = None) -> _OperationJob[DatumId | InvocationId]:
        if association is None:
            if self.invocation_job is not None:
                return self.invocation_job
            if len(self.jobs) != 1:
                reject(EffectCode.MISSING)
            return next(iter(self.jobs.values()))
        if association not in self.jobs:
            reject(EffectCode.FOREIGN_OWNER)
        return self.jobs[association]

    def _check(self) -> None:
        if self.closed or (self.driving != get_ident() and asyncio._get_running_loop() is not self.loop):
            reject(EffectCode.UNSUPPORTED)

    def _wake(self) -> None:
        """Wake the scheduler: after a synchronous drive returns, or now on the owner loop, or thread-safely."""
        if self.driving == get_ident():
            self.wake_pending = True
        elif asyncio._get_running_loop() is self.loop:
            self.runtime.control.scheduler_changed.set()
        else:
            with suppress(RuntimeError):  # A closed owner loop has no scheduler left to wake.
                self.loop.call_soon_threadsafe(self.runtime.control.scheduler_changed.set)

    async def aparser_rejected(self, request: PhysicalRequestId) -> ChildEvidence:
        return self.parser_rejected(request)

    def parser_rejected(self, request: PhysicalRequestId) -> ChildEvidence:
        self._check()
        task = asyncio.current_task()
        if task is not self.submission and task not in self.entered_calls:
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

    async def enter_operation(self, slot: int, association: SemanticAssociation | None) -> AssociationInput | None:
        """Enter an admitted semantic child without inventing a physical request."""
        self._check_submission()
        require_count(slot)
        source_job = self._source_job(association)
        source = self.runtime.frame.occurrence(source_job.scope)
        key = source.keys.get(slot)
        entry = source.find(key)
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
            return (
                source_job.inputs[0]
                if source_job.inputs
                else AssociationInput(association=source_job.association, inputs=())
            )
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
        source = self.runtime.frame.occurrence(source_job.scope)
        key = source.keys.get(slot)
        if key is None:
            reject(EffectCode.FOREIGN_OWNER)
        node = self.runtime.prepared.reservation_recipe[slot].template
        policy = self.runtime.policies.get(node)
        if policy is None or policy.submission_owner != self.policy.node:
            reject(EffectCode.FOREIGN_OWNER)
        if not isinstance(source_job.scope, DatumId):
            reject(EffectCode.UNSUPPORTED)
        return self.runtime._operation_result(source_job.scope, key)

    async def ready_operations(self, association: SemanticAssociation | None) -> tuple[int, ...]:
        """Expose only ready child occurrences belonging to this live submission."""
        self._check_submission()
        source = self.runtime.frame.occurrence(self._source_job(association).scope)
        ready = {
            entry.activation
            for entry in source.state.entries
            if (
                entry.status == "ready"
                and (policy := self.runtime.policies.get(entry.template)) is not None
                and policy.submission_owner == self.policy.node
            )
            or (
                association is None
                and self.invocation_job is not None
                and entry.activation == self.invocation_job.activation
                and entry.status == "running"
            )
        }
        return tuple(slot for slot, key in source.keys.items() if key in ready)

    async def complete_operation(self, association: SemanticAssociation, result: LocalResult) -> RuntimeOutcome:
        self._check_submission()
        if type(association) is not SemanticAssociation:
            reject(EffectCode.FOREIGN_OWNER)
        # Exact types all the way down (SemanticAssociation -> TaskAttemptId -> ActivationKey): the lookup hashes
        # only engine tokens and ints, and a foreign attempt is rejected below by ``bound.association``.
        bound = self.runtime.operations.get(association.task.activation)
        if (
            bound is None
            or bound.association != association
            or (
                not isinstance(bound.scope, DatumId)
                and (self.invocation_job is None or bound.scope != self.invocation_job.scope)
            )
        ):
            reject(EffectCode.FOREIGN_OWNER)
        job = bound
        owner_association = self.invocation_job is not None and association == self.invocation_job.association
        owned_scopes: set[DatumId | InvocationId] = {row.scope for row in self.jobs.values()}
        if self.invocation_job is not None:
            owned_scopes.add(self.invocation_job.scope)
        if (
            association not in self.jobs
            and not owner_association
            and (job.policy.submission_owner != self.policy.node or job.scope not in owned_scopes)
        ):
            reject(EffectCode.FOREIGN_OWNER)
        entry = self.runtime.frame.occurrence(job.scope).entry(job.activation)
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
                self.runtime.admitted.ingress_limits,
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
        self._wake()
        return final

    async def ready_operations_many(self, associations: tuple[SemanticAssociation, ...]) -> tuple[tuple[int, ...], ...]:
        """Coalesce loop handoffs while checking each source against the live owner."""
        self._check_submission()
        if type(associations) is not tuple:
            reject(EffectCode.INVALID_TYPE)
        return tuple([await self.ready_operations(association) for association in associations])

    async def enter_operations(
        self, slot: int, associations: tuple[SemanticAssociation, ...]
    ) -> tuple[AssociationInput | None, ...]:
        self._check_submission()
        if type(associations) is not tuple:
            reject(EffectCode.INVALID_TYPE)
        entered = []
        for index, association in enumerate(associations):
            entered.append(await self.enter_operation(slot, association))
        return tuple(entered)

    async def operation_results(
        self, slot: int, associations: tuple[SemanticAssociation, ...]
    ) -> tuple[AssociationResult | None, ...]:
        self._check_submission()
        if type(associations) is not tuple:
            reject(EffectCode.INVALID_TYPE)
        return tuple([await self.operation_result(slot, association) for association in associations])

    async def complete_operations(
        self, results: tuple[tuple[SemanticAssociation, LocalResult], ...]
    ) -> tuple[RuntimeOutcome, ...]:
        self._check_submission()
        if type(results) is not tuple:
            reject(EffectCode.INVALID_TYPE)
        outcomes = []
        for index, item in enumerate(results):
            if type(item) is not tuple or len(item) != 2:
                reject(EffectCode.INVALID_TYPE)
            association, result = item
            outcome = await self.complete_operation(association, result)
            outcomes.append(outcome)
            if type(result) is LocalCompleted and (outcome.condition != "result" or outcome.category != "success"):
                break
        return tuple(outcomes)

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
            self._check_submission(concurrent=True)
            job = self._source_job(association)
            chain = self.chains[job.association, slot]
            if any(key == chain.chain and not child.done() for child, key in self.children.items()):
                # One physical child per chain: a correction or restart follows its settled predecessor.
                reject(EffectCode.REQUEST_CAUSALITY)
            await self._await_remote_capacity()
            result = await self._borrow(
                self._request(slot, source_kind, predecessor, evidence, chain, job.association), chain.chain
            )
            assert result is not None
            return result
        except EffectRejected:
            self.request_rejected = True
            raise

    def _check_submission(self, *, concurrent: bool = False) -> None:
        self._check()
        driven = self.driving == get_ident()
        task = None if driven else asyncio.current_task()
        worker = driven or task in self.entered_calls
        # Only worker physical requests overlap borrowed children; same-loop submission keeps one at a time.
        if (task is not self.submission and not worker) or (self.active and not (concurrent and worker)):
            reject(EffectCode.UNSUPPORTED)
        if self.cancelled:
            reject(EffectCode.REQUEST_CAUSALITY)

    async def _borrow(
        self, operation: Coroutine[object, object, PhysicalRequestId | None], chain: RequestChainId | None = None
    ) -> PhysicalRequestId | None:
        child = asyncio.create_task(operation)
        self.children[child] = chain
        self.children_idle.clear()
        child.add_done_callback(self._child_done)
        try:
            return await asyncio.shield(child)
        except asyncio.CancelledError:
            if not child.done():
                self.cancel_requested = True
            raise

    def _child_done(self, child: asyncio.Task[PhysicalRequestId | None]) -> None:
        self.children.pop(child, None)
        if not child.cancelled():
            child.exception()  # Retrieved even when the borrowing observer left.
        self.children_settled.set()
        if not self.active:
            self.children_idle.set()

    def _remote_limit_reached(self) -> bool:
        # Reserved requests count: a concurrent child reserves before it dispatches.
        limit = self.runtime.services.limits.max_remote_outstanding
        if limit is None:
            return False
        state = self.runtime.request_authority.state
        chains = self.own_chains
        pending = sum(item.child is not None and item.child.chain in chains for item in state.reserved)
        return len(state.remote_outstanding) + pending >= limit

    async def _await_remote_capacity(self) -> None:
        """Queue behind in-flight children for a declared remote limit; reservation still enforces it."""
        while self._remote_limit_reached() and self.active and not self.cancelled:
            self.children_settled.clear()
            await self.children_settled.wait()

    async def _await_children_idle(self) -> None:
        while self.active:
            self.children_idle.clear()
            await self.children_idle.wait()

    def _release_reserved_gate(self) -> None:
        """After reservation, let other chains reserve while this physical request is in flight."""
        hold = _GATE_HOLD.get()
        if hold is not None and hold.concurrent:
            hold.release()

    def _cancel_worker(self, borrow: _WorkerBorrow) -> None:
        if self.worker_borrow is borrow:
            self.cancel_requested = True

    def _drive_call(self, borrow: _WorkerBorrow, operation: Callable[[], Coroutine[object, object, _T]]) -> _T:
        """Run one worker bookkeeping call synchronously on the calling thread, under the runtime lock."""
        with self.runtime.lock:
            if self.closed:
                reject(EffectCode.UNSUPPORTED)
            if self.worker_borrow is not borrow or _WORKER_BORROW.get() is not borrow or borrow.expired:
                reject(EffectCode.FOREIGN_OWNER)
            if self.driving is not None or self.active or self.cancelled:
                reject(EffectCode.REQUEST_CAUSALITY)
            if len(self.jobs) == 1 and not self.contract.generation_groups and self.entered_calls:
                reject(EffectCode.REQUEST_CAUSALITY)
            self.driving = get_ident()
            try:
                return drive(operation())
            finally:
                self.driving = None
                if self.wake_pending:
                    self.wake_pending = False
                    # The frontier the owner loop would apply before the body's next call; launches stay there.
                    self.runtime.advance()
                    self._wake()

    async def _worker_call(
        self,
        borrow: _WorkerBorrow,
        operation: Callable[[], Coroutine[object, object, _T]],
        *,
        concurrent: bool = False,
    ) -> _T:
        self._check()
        if self.worker_borrow is not borrow or _WORKER_BORROW.get() is not borrow or borrow.expired:
            reject(EffectCode.FOREIGN_OWNER)
        if len(self.jobs) == 1 and not self.contract.generation_groups and (self.entered_calls or self.active):
            reject(EffectCode.REQUEST_CAUSALITY)
        task = asyncio.current_task()
        assert task is not None
        self.worker_calls.add(task)
        try:
            await self.worker_gate.acquire()
            hold = _GateHold(self.worker_gate, concurrent=concurrent)
            try:
                if not concurrent:
                    # A queued callback runs after the in-flight physical children settle, as before.
                    await self._await_children_idle()
                self._check()
                if borrow.expired or (self.active and not concurrent) or self.cancelled:
                    reject(EffectCode.REQUEST_CAUSALITY)
                self.entered_calls.add(task)
                token = _GATE_HOLD.set(hold)
                try:
                    return await operation()
                finally:
                    _GATE_HOLD.reset(token)
                    self.entered_calls.discard(task)
            finally:
                hold.release()
        finally:
            self.worker_calls.discard(task)

    async def _submit_worker(self, adapter: WorkerDelegatedSubmitter | CallerThreadSubmitter) -> LocalResult:
        borrow = _WorkerBorrow(self, adapter)
        self.worker_borrow = borrow
        context = copy_context()
        context.run(_WORKER_BORROW.set, borrow)

        def call() -> object:
            return context.run(adapter._callback, self.inputs, borrow)

        try:
            if isinstance(adapter, CallerThreadSubmitter):
                value = await adapter._run(call, lambda: not self.cancelled)
            else:
                value = await _BlockingInvocation(adapter, call).run(lambda: not self.cancelled, self)
            return (
                value
                if isinstance(value, (LocalCompleted, LocalFailure))
                else LocalFailure(failure="malformed_response")
            )
        finally:
            borrow.expired = True

    def _satisfied(self, slot: int, outcomes: frozenset[str]) -> bool:
        source = self.runtime.frame.invocation_source or self.runtime.frame.occurrence(self._source_job().scope)
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

    async def close_observed_requests(
        self, associations: tuple[SemanticAssociation, ...], generated: frozenset[SemanticAssociation]
    ) -> None:
        self._check_submission()
        if self.contract.observed_request is None or type(associations) is not tuple:
            reject(EffectCode.UNSUPPORTED)
        for association in associations:
            job = self._source_job(association)
            source = self.runtime.frame.occurrence(job.scope)
            for group in self.contract.generations if association in generated else ():
                first = source.entry(source.keys[group[0]])
                if first.status in {"ready", "unstarted", "running"}:
                    reject(EffectCode.REQUEST_CAUSALITY)
            self.runtime.blocked_reasons.update(self.runtime._unused_tails(cast(_OperationJob[DatumId], job)))
        for association in associations:
            job = self._source_job(association)
            source = self.runtime.frame.occurrence(job.scope)
            slots = self.contract.child_slots + (
                ()
                if self.runtime.frame.invocation_source is not None or self.contract.readiness is None
                else self.contract.readiness.slots
            )
            for slot in slots:
                key = source.keys[slot]
                if source.entry(key).status in {"ready", "unstarted"}:
                    source.state = advance_activation(
                        state=source.state, event=CloseUnstarted(activation=key, category="blocked")
                    )
        if self.runtime.frame.invocation_source is not None and all(
            (owned := self.runtime.frame.occurrence(job.scope)).entry(owned.keys[self.contract.child_slots[0]]).status
            not in {"ready", "unstarted", "running"}
            for job in self.jobs.values()
        ):
            source = self.runtime.frame.invocation_source
            readiness = self.contract.readiness
            # Readiness that never started gated no generation (0.4.0 skips the map call and its health check
            # when no row has entities), so closing it omits shared work instead of failing its successors.
            owner = None if readiness is None else source.find(source.keys[readiness.owner_slot])
            unneeded = owner is not None and owner.status in {"ready", "unstarted"}
            for slot in readiness.slots if readiness is not None else ():
                key = source.keys[slot]
                if source.entry(key).status in {"ready", "unstarted"}:
                    if unneeded:
                        self.runtime.blocked_reasons[key] = "omitted_optional"
                    source.state = advance_activation(
                        state=source.state, event=CloseUnstarted(activation=key, category="blocked")
                    )
        self._wake()

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
            source = self.runtime.frame.invocation_source or self.runtime.frame.occurrence(self._source_job().scope)
            entry = source.entry(source.keys[descriptor.request_slot])
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
                job.policy,
                handle,
                job.association,
                job.inputs,
                self,
                self.runtime.admitted.ingress_limits,
                cancel_waiters=self.runtime.control.cancel_waiters,
            )
            assert isinstance(returned, tuple)
            mapping, results, assessments = returned
        except asyncio.CancelledError:
            mapping, results, assessments = _mapping(job.policy, "cancel_after_start", None, None), (), ()
        self._finish(job, mapping, results, assessments)

    @property
    def active(self) -> bool:
        return any(not child.done() for child in self.children)

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
            else runtime.frame.occurrence(self._source_job(association).scope)
        )
        key = source.keys[slot]
        entry = source.entry(key)
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
        if self._remote_limit_reached():
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
                purpose=_CHILD_PURPOSE[source_kind],
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
        self._release_reserved_gate()
        envelope = DispatchEnvelope(
            request=request,
            purpose=_CHILD_PURPOSE[source_kind],
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
            cancel_waiters=runtime.control.cancel_waiters,
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
        job: _OperationJob[DatumId | InvocationId],
        mapping: RuntimeOutcome,
        results: tuple[AssociationResult, ...],
        assessments: tuple[LocalAssessmentResult, ...] = (),
    ) -> None:
        self.runtime._complete_job(job, mapping, results, assessments)
        self._wake()

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
        contract = self.cleanup_contract
        incomplete: set[SemanticAssociation] = set()
        targets: set[DatumId | InvocationId] = {job.scope for job in self.jobs.values()}
        if self.invocation_job is not None:
            targets.add(self.invocation_job.scope)
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
                entry = source.entry(previous.activation)
                if entry.template in owned and entry.status != "success":
                    incomplete.add(job.association)
                    if entry.status in {"ready", "unstarted"}:
                        source.state = advance_activation(
                            state=source.state, event=CloseUnstarted(activation=entry.activation, category="blocked")
                        )
            unused = {} if self.cancelled or self.request_rejected else self.runtime._unused_tails(job, contract)
            for slot in (
                contract.child_slots if self.runtime.frame.invocation_source is not None else contract.owned_slots
            ):
                key = source.keys[slot]
                if source.entry(key).status in {"ready", "unstarted"}:
                    if key not in unused:
                        incomplete.add(job.association)
                    source.state = advance_activation(
                        state=source.state, event=CloseUnstarted(activation=key, category="blocked")
                    )
        source = self.runtime.frame.invocation_source
        if source is not None:
            owned_slots = set(contract.owned_slots)
            for entry in tuple(source.state.entries):
                if entry.status in {"ready", "unstarted"} and (
                    self.runtime.policies[entry.template].submission_owner == self.policy.node
                    or any(source.keys.get(slot) == entry.activation for slot in owned_slots)
                ):
                    incomplete.update(self.jobs)
                    source.state = advance_activation(
                        state=source.state, event=CloseUnstarted(activation=entry.activation, category="blocked")
                    )
        return frozenset(incomplete)

    def _failure(self, condition: str, failure: FailureClass | None = None) -> _SubmissionResults:
        associations = self.jobs if self.invocation_job is None else (self.invocation_job.association,)
        return {association: (_mapping(self.policy, condition, None, failure), (), ()) for association in associations}

    def _invocation_completed(self) -> _SubmissionResults:
        assert self.invocation_job is not None
        return {self.invocation_job.association: (_mapping(self.policy, "result", "ok", None), (), ())}

    async def run(self, handle: ImplementationHandle) -> _SubmissionResults:
        """Settle the submission, then accept its response in stages; any stage fails the whole response."""
        try:
            result, incomplete = await self._settle(handle)
            if self.contract.mode == "opaque" and self.cancelled:
                raise _Failed("cancel_after_start")
            if type(result) is LocalFailure:
                raise _Failed("failure", inspect_failure(result.failure))
            if self._publish_invocation_frame(result):
                return self._invocation_completed()
            completed = self._completed(result)
            failed = completed.failed_sources
            if self.owns_children:
                failed = failed | self._owned_incomplete(completed, incomplete)
            successful = tuple(
                item
                for item in completed.results
                if item.association not in failed
                and _mapping(self.policy, "result", item.outcome, None).category == "success"
            )
            self._check_selections(successful, incomplete)
            self._publish_frame(completed, successful)
            if self.invocation_job is not None:
                return self._invocation_completed()
            return self._settled_results(completed, failed)
        except _Failed as failed:
            return self._failure(*failed.args)

    async def _settle(self, handle: ImplementationHandle) -> tuple[LocalResult, frozenset[SemanticAssociation]]:
        assert handle.delegated is not None
        borrowing = delegated_submitter_of(handle.delegated)
        task = asyncio.create_task(
            self._submit_worker(borrowing) if borrowing is not None else handle.delegated.submit(self.inputs, self)
        )
        self.submission = task
        try:
            while not task.done():
                if self.runtime.control.cancelled and not self.active:
                    task.cancel()
                    await asyncio.wait((task,))
                else:
                    await await_completion_or_cancel(task, self.runtime.control, self.runtime.control.cancel_waiters)
                    if not task.done():
                        # Cancelled: in-flight children observe it and settle; then the submission is cancelled.
                        await self._await_children_idle()
            result = task.result()
        except asyncio.CancelledError:
            self.cancel_requested = True
            if not task.done():
                task.cancel()
            raise _Failed("cancel_after_start") from None
        except Exception as error:
            self.runtime.submission_failures.append(SubmissionFailure(associations=frozenset(self.jobs), cause=error))
            raise _Failed("failure", "implementation_exception") from None
        finally:
            for child in tuple(self.children):
                await self._drain(child)
            await self._drain(task)
            for worker_call in tuple(self.worker_calls):
                await self._drain(worker_call)
            while self.active:
                for child in tuple(self.children):
                    await self._drain(child)
            incomplete = self.close()
        return result, incomplete

    def _publish_invocation_frame(self, result: LocalResult) -> bool:
        """An opaque invocation owner may answer with one source-free frame once its owned work succeeded."""
        if (
            self.contract.mode != "opaque"
            or self.invocation_job is None
            or not _is_invocation_frame(getattr(result, "frame", None))
        ):
            return False
        source = self.runtime.frame.invocation_source
        if source is None or any(
            (policy := self.runtime.policies.get(entry.template)) is not None
            and policy.submission_owner == self.policy.node
            and entry.status != "success"
            for entry in source.state.entries
        ):
            raise _Failed("failure", "implementation_exception")
        accepted = _accept_invocation_frame(result)
        if accepted is None:
            raise _Failed("failure", "malformed_response")
        self.runtime.frame.publications.append(accepted)
        return True

    def _completed(self, result: LocalResult) -> LocalCompleted:
        """Shape: one completed response naming only this submission's associations and admitted outcomes."""
        if self.invocation_job is not None and self.contract.mode != "opaque":
            raise _Failed("failure", "malformed_response")
        if type(result) is not LocalCompleted:
            raise _Failed("failure", "malformed_response")
        failed_sources = getattr(result, "failed_sources", None)
        if (
            type(failed_sources) is not frozenset
            or any(type(source) is not SemanticAssociation for source in failed_sources)
            or not failed_sources <= self.jobs.keys()
        ):
            raise _Failed("failure", "malformed_response")
        invalid = inspect_associations(
            result.results,
            frozenset(self.jobs) - failed_sources,
            self.runtime.admitted.ingress_limits,
            assessments=result.assessments,
        )
        if (
            invalid is None
            or invalid
            or any(item.outcome not in self.policy.result_outcomes for item in result.results)
        ):
            raise _Failed("failure", "malformed_response")
        return result

    def _owned_incomplete(
        self, result: LocalCompleted, incomplete: frozenset[SemanticAssociation]
    ) -> frozenset[SemanticAssociation]:
        """Owned children: no rewrite of a published row, and every unfinished child fails its row."""
        for item in result.results:
            assert isinstance(item.association, SemanticAssociation)
            job = self.jobs[item.association]
            entry = self.runtime.frame.sources[job.scope].entry(job.activation)
            if entry.status == "success" and (
                item.outputs or any(value.association == item.association for value in result.assessments)
            ):
                raise _Failed("failure", "malformed_response")
        containers = frozenset(
            node.id
            for body in self.runtime.prepared.workflow.bodies()
            for node in body.operations
            if isinstance(node, SubgraphNode)
        )
        if any(
            entry.template not in self.runtime.policies and entry.template not in containers
            for job in self.jobs.values()
            for entry in self.runtime.frame.sources[job.scope].state.entries
        ):
            raise _Failed("failure", "malformed_response")
        if self.contract.mode == "opaque" and self.contract.observed_request is None:
            return incomplete
        return frozenset(
            job.association
            for job in self.jobs.values()
            if any(
                self.runtime.policies[entry.template].submission_owner == self.policy.node and entry.status != "success"
                for entry in self.runtime.frame.sources[job.scope].state.entries
                if entry.template not in containers
            )
        )

    def _check_selections(
        self, successful: tuple[AssociationResult, ...], incomplete: frozenset[SemanticAssociation]
    ) -> None:
        """A successful row names exactly the optional children it left undispatched, after its readiness."""
        selection = self.contract.selection
        for item in successful:
            assert isinstance(item.association, SemanticAssociation)
            if selection is None:
                if item.association in incomplete:
                    raise _Failed("failure", "implementation_exception")
                continue
            outputs = [output for output in item.outputs if output.port == selection.port]
            try:
                if len(outputs) != 1:
                    raise _Failed("failure", "malformed_response")
                omitted = selection.omitted(outputs[0].value)
            except EffectRejected:
                raise _Failed("failure", "malformed_response") from None
            job = self.jobs[item.association]
            source = self.runtime.frame.sources[job.scope]
            missing = {slot for slot in self.contract.child_slots if source.keys[slot] not in self.runtime.operations}
            retry_unused = {key.occurrence for key in self.runtime._unused_tails(job)}
            missing -= {slot for slot in missing if source.keys[slot].occurrence in retry_unused}
            if missing != set(omitted):
                raise _Failed("failure", "implementation_exception")
            readiness = self.contract.readiness
            if readiness is not None:
                ready_source = self.runtime.frame.invocation_source or source
                required = {ready_source.keys[slot] for slot in readiness.slots}
                if any(
                    entry.status != "success" for entry in ready_source.state.entries if entry.activation in required
                ):
                    raise _Failed("failure", "implementation_exception")

    def _publish_frame(self, result: LocalCompleted, successful: tuple[AssociationResult, ...]) -> None:
        """Publish a returned frame: aggregate for a source-free frame, otherwise bound to its semantic rows."""
        frame = result.frame
        if _is_invocation_frame(frame):
            self._publish_aggregate(result, successful, frame)
            return
        if frame is None:
            return
        frame_results = successful
        if self.contract.mode == "opaque" and self.owns_children and not any(item.outputs for item in successful):
            targets = {self.jobs[cast(SemanticAssociation, item.association)].scope for item in successful}
            frame_results = (
                self._publication_source_results(frame, targets)
                if self.contract.publication_sources
                else self._final_output_results(targets)
            )
        opaque = tuple(
            value for job in self.jobs.values() for value in self.runtime.frame.sources[job.scope].opaque_cells
        )
        try:
            accepted = frame.accept(frame_results, opaque=opaque) if _is_publication_frame(frame) else None
        except (AttributeError, TypeError, ValueError):
            accepted = None
        if accepted is None:
            raise _Failed("failure", "malformed_response")
        self.runtime.frame.publications.append(accepted)

    def _publish_aggregate(
        self, result: LocalCompleted, successful: tuple[AssociationResult, ...], frame: object
    ) -> None:
        if (
            self.contract.mode != "opaque"
            or self.runtime.prepared.configuration.purpose != "execution_only"
            or self.policy.implementations[0].capability.operation.outputs
            or any(item.outputs for item in result.results)
            or {item.association for item in result.results} != {item.association for item in successful}
            or result.assessments
        ):
            raise _Failed("failure", "malformed_response")
        # Source closures were validated above; the frame remains aggregate and nonqualifying.
        scopes = {job.scope for job in self.jobs.values()}
        publication = _aggregate_frame(
            frame,
            frozenset(
                operation.activation
                for operation in self.runtime.operations.values()
                if (operation.policy.submission_owner == self.policy.node or operation.policy.node == self.policy.node)
                and operation.scope in scopes
            ),
        )
        if publication is None:
            raise _Failed("failure", "malformed_response")
        self.runtime.frame.publications.append(publication)

    def _publication_source_results(self, frame: object, targets: set[DatumId]) -> tuple[AssociationResult, ...]:
        """Bind a frame to the declared owned publication sources, one per successful target."""
        allowed = dict(self.contract.publication_sources)
        if (
            self.runtime.prepared.configuration.purpose != "execution_only"
            or not _is_publication_frame(frame)
            or len(frame.associations) != len(targets)
        ):
            raise _Failed("failure", "malformed_response")
        results = []
        seen_targets = set()
        for association in frame.associations:
            job = self.runtime.operations.get(association.task.activation)
            if (
                job is None
                or job.association != association
                or not isinstance(job.scope, DatumId)
                or job.scope not in targets
                or job.scope in seen_targets
                or job.policy.submission_owner != self.policy.node
                or job.node not in allowed
                or not {port for _, port, _ in frame._text} <= allowed[job.node]
            ):
                raise _Failed("failure", "malformed_response")
            seen_targets.add(job.scope)
            accepted = self.runtime._operation_result(job.scope, job.activation)
            if accepted is None:
                raise _Failed("failure", "malformed_response")
            results.append(accepted)
        return tuple(results)

    def _final_output_results(self, targets: set[DatumId]) -> tuple[AssociationResult, ...]:
        """Bind a frame to each successful target's final output, produced by an owned child."""
        finals = tuple(
            item
            for item in _final_outputs(self.runtime.prepared, self.runtime.frame.states, self.runtime.facts)
            if item.target in targets
        )
        if len(finals) != len(targets) or {item.target for item in finals} != targets:
            raise _Failed("failure", "malformed_response")
        results = []
        for item in finals:
            producer = item.producer
            if not isinstance(producer, OperationOutputKey) or producer.activation not in self.runtime.operations:
                raise _Failed("failure", "malformed_response")
            entry = self.runtime.frame.sources[item.target].entry(producer.activation)
            if self.runtime.policies[entry.template].submission_owner != self.policy.node:
                raise _Failed("failure", "malformed_response")
            accepted = self.runtime._operation_result(item.target, producer.activation)
            if accepted is None:
                raise _Failed("failure", "malformed_response")
            results.append(accepted)
        return tuple(results)

    def _settled_results(self, result: LocalCompleted, failed: frozenset[SemanticAssociation]) -> _SubmissionResults:
        failed_results: _SubmissionResults = {}
        for association in failed:
            job = self.jobs[association]
            entry = self.runtime.frame.sources[job.scope].entry(job.activation)
            if association in result.failed_sources and entry.status == "success":
                # A later row drop does not undo an already accepted prefix.
                # Descendants were closed above; no final output may survive it.
                if any(
                    item.target == job.scope
                    for item in _final_outputs(self.runtime.prepared, self.runtime.frame.states, self.runtime.facts)
                ):
                    raise _Failed("failure", "malformed_response")
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
                if isinstance(item.association, SemanticAssociation) and item.association not in failed
            },
        }
