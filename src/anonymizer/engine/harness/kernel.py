# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Route-neutral execution kernel: plan preparation, admit/start, synchronous hosting and projection.

Every adapter route (detection, replace, rewrite, evaluation, detection transforms) prepares a
fixed-operation plan, admits and starts it on the single Graph SDK executor, hosts it on a private
thread while serving caller-thread callbacks, and projects the canonical frame back to a DataFrame.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from concurrent.futures import Future
from contextvars import copy_context
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Thread, get_ident
from time import monotonic_ns
from typing import TYPE_CHECKING, Any, cast

from anonymizer.engine.constants import COL_REPLACEMENT_MAP
from anonymizer.engine.graph_sdk._activation_values import ActivationLimits
from anonymizer.engine.graph_sdk._execution_delegated import (
    CallerChannel,
    CallerThreadSubmitter,
    SubmitterPool,
    WorkerDelegatedSubmitter,
)
from anonymizer.engine.graph_sdk._execution_values import DelegatedRequests
from anonymizer.engine.graph_sdk._workflow_values import Graph, NodeId, OperationNode, OperationSpec
from anonymizer.engine.graph_sdk.blocking import BlockingAffinity
from anonymizer.engine.graph_sdk.capabilities import (
    FrozenConfig,
    ImplementationCapability,
    ImplementationRef,
    ImplementationSelection,
    config_size,
)
from anonymizer.engine.graph_sdk.context import admit_context_plan
from anonymizer.engine.graph_sdk.data import ValidatedDataGraph
from anonymizer.engine.graph_sdk.evidence import QualificationLimits, admit_qualification, evidence_revision_view
from anonymizer.engine.graph_sdk.executor import (
    AdmittedExecutionPlan,
    AssessmentLimits,
    DelegatedRequestContract,
    EvidenceProductionDecl,
    ExecutionImplementation,
    ExecutionLimits,
    ExecutionResult,
    ExecutionServices,
    ImplementationHandle,
    IngressLimits,
    LocalCompleted,
    LocalFailure,
    OperationExecutionPolicy,
    RunningExecution,
    RuntimeOutcome,
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.preparation import (
    BoundInput,
    PreparationConfiguration,
    PreparationLimits,
    PreparedPlan,
    prepare,
)
from anonymizer.engine.graph_sdk.qualification import QualificationResult, qualify_retained
from anonymizer.engine.graph_sdk.requests import AssociationInput, AssociationResult
from anonymizer.engine.graph_sdk.resources import ResourceLease
from anonymizer.engine.harness.frames import InvocationFrame
from anonymizer.engine.replace.replace_runner import ReplacementResult
from anonymizer.interface.errors import AnonymizerWorkflowError

if TYPE_CHECKING:
    from anonymizer.engine.graph_sdk._foreign_transport import ForeignLoopTransport
    from anonymizer.engine.ndd._graph_requests import NddPhysicalTransport


class _Clock:
    def now_ns(self) -> int:
        return monotonic_ns()


@dataclass(frozen=True)
class _ProfileLocal:
    outcome: str

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted:
        if self.outcome == "async_required" and not ensure_async_engine_loop().is_running():
            raise RuntimeError("DD async engine is not running")
        return LocalCompleted(
            results=tuple(
                AssociationResult(
                    association=item.association,
                    outcome=self.outcome,
                    outputs=(),
                    consumed_context_ports=frozenset(),
                )
                for item in request
            )
        )


def _runtime_rows(names: tuple[str, ...], external: bool) -> tuple[RuntimeOutcome, ...]:
    rows = [
        RuntimeOutcome(
            condition="result",
            reported_outcome=name,
            failure=None,
            outcome=name,
            category="success",
        )
        for name in names
    ]
    rows += [
        RuntimeOutcome(condition="failure", reported_outcome=None, failure=failure, outcome=None, category="failure")
        for failure in (
            "rejected_before_acceptance",
            "retryable",
            "malformed_response",
            "permanent",
            "transport_unknown",
            "implementation_exception",
        )
    ]
    conditions = {
        "cancel_before_start": "blocked",
        "cancel_after_start": "cancelled",
        "artifact_limit_exhausted": "blocked",
        "deadline_exhausted": "failure",
    }
    if external:
        conditions.update(
            cancel_after_dispatch="cancelled",
            lost="lost",
            request_inconsistent="inconsistent",
            budget_exhausted="blocked",
            request_limit_exhausted="blocked",
        )
    rows += [
        RuntimeOutcome(
            condition=cast(Any, condition),
            reported_outcome=None,
            failure=None,
            outcome=None,
            category=cast(Any, category),
        )
        for condition, category in conditions.items()
    ]
    return tuple(rows)


def delegated_submitter(
    callback: Callable[..., Any], pool: SubmitterPool, borrowed_loop: asyncio.AbstractEventLoop
) -> WorkerDelegatedSubmitter | CallerThreadSubmitter:
    """Run the delegated body on the synchronous caller (kernel-hosted) or on ``pool`` (standalone use)."""
    if isinstance(pool, CallerChannel):
        return CallerThreadSubmitter(callback=callback, channel=pool, borrowed_loop=borrowed_loop)
    return WorkerDelegatedSubmitter(
        callback=callback,
        pool=pool,
        affinity=BlockingAffinity(cross_thread=True, concurrency_safe=False, close_on_event_loop=False),
        borrowed_loop=borrowed_loop,
    )


def _run_sync_profile(
    operation: Callable[
        [CallerChannel, Callable[[Callable[[], LocalCompleted | LocalFailure]], LocalCompleted | LocalFailure]],
        Awaitable[ReplacementResult],
    ],
) -> ReplacementResult:
    """Host one graph invocation on a private loop thread; every synchronous body runs on the caller's thread."""
    started = Event()
    cancellation: list[tuple[asyncio.AbstractEventLoop, asyncio.Task[Any]]] = []
    # (call, future) pairs for the caller's thread; (None, None) marks the host's completion.
    calls: Queue[tuple[Callable[[], Any], Future[Any]] | tuple[None, None]] = Queue()
    serving = get_ident()

    # Every queued body's future until it settles, so a Ctrl-C can settle what it interrupted.
    outstanding: set[Future[Any]] = set()

    def put(call: Callable[[], Any]) -> Future[Any]:
        returned: Future[Any] = Future()
        outstanding.add(returned)
        returned.add_done_callback(outstanding.discard)
        calls.put((call, returned))
        return returned

    def handoff(callback: Callable[[], LocalCompleted | LocalFailure]) -> LocalCompleted | LocalFailure:
        if get_ident() == serving:
            # Already on the caller's thread (inside a body it is running): run inline.
            return callback()
        return put(callback).result()

    def cancel_host() -> None:
        # The host loop may already be closed; a stop request then has nothing left to stop.
        if cancellation:
            loop, task = cancellation[0]
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                if not loop.is_closed():
                    raise

    def serve_callback() -> None:
        try:
            # The timeout only keeps Ctrl-C responsive where a blocking get is not interruptible.
            callback, returned = calls.get(timeout=0.1)
            if callback is None or returned is None:
                return
            if interrupted is not None:
                returned.set_result(LocalFailure(failure="implementation_exception"))
                return
            try:
                returned.set_result(callback())
            except Exception as error:
                returned.set_exception(error)
        except Empty:
            return
        except BaseException:
            cancel_host()
            # Only this thread runs bodies, so nothing is running now: settle whatever the
            # interrupt left waiting, including a body taken from the queue but not yet settled.
            for pending in list(outstanding):
                if not pending.done():
                    pending.set_result(LocalFailure(failure="implementation_exception"))
            raise

    async def run() -> ReplacementResult:
        task = asyncio.current_task()
        assert task is not None
        cancellation.append((asyncio.get_running_loop(), task))
        started.set()
        return await operation(CallerChannel(put), handoff)

    completion: Future[ReplacementResult] = Future()
    completion.add_done_callback(lambda _: calls.put((None, None)))

    def host() -> None:
        try:
            completion.set_result(asyncio.run(run()))
        except BaseException as error:
            completion.set_exception(error)
        finally:
            started.set()

    context = copy_context()
    thread = Thread(target=lambda: context.run(host), name="anonymizer-graph-host")
    thread.start()
    interrupted: BaseException | None = None
    try:
        while not completion.done():
            serve_callback()
        return completion.result()
    except BaseException as error:
        interrupted = error
        raise
    finally:
        # Cancellation only requests a stop. Keep serving until the host has drained
        # every queued body and the client work it started.
        while thread.is_alive():
            try:
                if interrupted is not None:
                    started.wait()
                    cancel_host()
                serve_callback()
                thread.join(timeout=0.01)
            except KeyboardInterrupt as error:
                interrupted = error
        if interrupted is not None:
            failure = completion.exception()
            owner = retained_execution(failure) if failure is not None else completion.result()._execution
            if owner is not None:
                _retained_execution = owner
            raise interrupted


def retained_execution(error: BaseException) -> ExecutionResult | None:
    """Return the execution the harness kept alive for ``error``, or None.

    Raise sites bind ``_retained_execution`` in their own frame instead of decorating the
    exception, so nothing is added to the caller's exception (release parity) while the
    traceback keeps that frame, and with it the execution and its leases, alive for exactly
    as long as the caller holds the error. The outermost binding wins, as the last raise
    site to see the error is the one whose execution owns it.
    """
    traceback = error.__traceback__
    while traceback is not None:
        local = traceback.tb_frame.f_locals
        if "_retained_execution" in local:
            return local["_retained_execution"]
        traceback = traceback.tb_next
    return None


async def _await_profile_execution(running: RunningExecution) -> ExecutionResult:
    """Retain the actual owner until cancellation and callback settlement finish."""
    completion = asyncio.create_task(running.wait())
    try:
        return await asyncio.shield(completion)
    except asyncio.CancelledError:
        running.request_cancel()
        while not completion.done():
            try:
                await asyncio.shield(completion)
            except asyncio.CancelledError:
                continue
        _retained_execution = completion.result()
        raise


def _prepare_frame_plan(
    data: ValidatedDataGraph,
    raw: Graph,
    capabilities: tuple[ImplementationCapability, ...],
    bound_inputs: tuple[BoundInput, ...],
    *,
    invocation_owned: bool = False,
) -> PreparedPlan:
    """Shared fixed-operation preparation for existing whole-frame consumers."""
    bodies = raw.bodies()
    nodes = tuple(node for body in bodies for node in body.operations)
    count = raw.activation_upper_bound
    workflow = raw
    sizes = [config_size(cap.configuration) for cap in capabilities]
    return prepare(
        data=data,
        workflow=workflow,
        activation_limits=ActivationLimits(max_events=4 * count, max_entries=count, max_parent_depth=len(bodies) + 1),
        bound_inputs=bound_inputs,
        configuration=PreparationConfiguration(
            purpose="execution_only",
            required_protection_outcomes=frozenset(),
            hard_request_limit=None,
            invocation_owned=invocation_owned,
        ),
        selections=tuple(
            ImplementationSelection(
                node=next(
                    node.id for node in nodes if isinstance(node, OperationNode) and node.operation == cap.operation
                ),
                implementation=cap.implementation,
                configuration=cap.configuration,
            )
            for cap in capabilities
        ),
        capabilities=capabilities,
        limits=PreparationLimits(
            max_bound_inputs=len(bound_inputs),
            max_selections=len(capabilities),
            max_config_depth=max(size[0] for size in sizes),
            max_config_atoms=sum(size[1] for size in sizes) * 2,
            max_total_activation_slots=count * max(1, len(data.targets)),
            max_capabilities=len(capabilities),
        ),
    )


# Native transport posture (recorded for DF-6/S8): every invocation that issues DataDesigner native requests
# acquires exactly one exclusive client, created on DataDesigner's engine loop and leased to every DD stage of
# that invocation; the per-request gate waits at most 1 s. ``ForeignLoopTransport.create`` refuses non-exclusive
# clients, so concurrency between stages comes from the runtime's scheduling, not from extra clients.
NATIVE_TRANSPORT_GATE_TIMEOUT_NS = 1_000_000_000


async def acquire_native_transport() -> tuple[NddPhysicalTransport, ForeignLoopTransport]:
    """Create the invocation's exclusive native client on DataDesigner's engine loop.

    Cancelling this observer never cancels or abandons the foreign factory: the first interruption is kept
    until the factory settles. A lease that was created anyway is closed before that interruption
    propagates; a factory failure after an interruption is chained under it; a factory that was itself
    cancelled propagates at once (no retry loop on a cancelled future).
    """
    from anonymizer.engine.graph_sdk._foreign_transport import ForeignLoopTransport
    from anonymizer.engine.ndd._graph_requests import NddPhysicalTransport

    async def create_resource() -> tuple[NddPhysicalTransport, ForeignLoopTransport]:
        physical = NddPhysicalTransport()
        try:
            return physical, ForeignLoopTransport.create(
                factory=lambda: physical,
                owner="sdk",
                gate_timeout_ns=NATIVE_TRANSPORT_GATE_TIMEOUT_NS,
                exclusive=True,
            )
        except BaseException:
            await physical.close()
            raise

    acquisition = asyncio.wrap_future(asyncio.run_coroutine_threadsafe(create_resource(), ensure_async_engine_loop()))
    interrupted: asyncio.CancelledError | None = None
    while True:
        try:
            physical, transport = await asyncio.shield(acquisition)
            break
        except asyncio.CancelledError as error:
            if acquisition.cancelled():
                raise
            if interrupted is None:
                interrupted = error
        except BaseException as error:
            if interrupted is not None:
                raise interrupted from error
            raise
    if interrupted is not None:
        await transport.close()
        raise interrupted
    return physical, transport


async def _execute_frame_plan(
    *,
    prepared: PreparedPlan,
    capabilities: tuple[ImplementationCapability, ...],
    policies: tuple[OperationExecutionPolicy, ...],
    callback: Callable[[tuple[AssociationInput, ...], DelegatedRequests], LocalCompleted | LocalFailure],
    pool: SubmitterPool,
    borrowed_loop: asyncio.AbstractEventLoop,
    limits: ExecutionLimits,
    ingress_limits: IngressLimits,
    assessment_limits: AssessmentLimits,
    assessment_productions: tuple[EvidenceProductionDecl, ...] = (),
    native_request_nodes: dict[str, NodeId] | None = None,
) -> ExecutionResult:
    """Run a single grouped submission with its admitted observed operations."""

    physical = None
    transport = None
    if native_request_nodes:
        from anonymizer.engine.replace._native_map_requests import bind_native_map_policy

        policies = bind_native_map_policy(prepared, policies, native_request_nodes)
        physical, transport = await acquire_native_transport()
    try:
        return await _execute_frame_plan_owned(
            prepared=prepared,
            capabilities=capabilities,
            policies=policies,
            callback=callback,
            pool=pool,
            borrowed_loop=borrowed_loop,
            limits=limits,
            ingress_limits=ingress_limits,
            assessment_limits=assessment_limits,
            assessment_productions=assessment_productions,
            native_request_nodes=native_request_nodes,
            physical=physical,
            transport=transport,
        )
    finally:
        if transport is not None and not transport._claimed:
            await transport.close()


async def _execute_frame_plan_owned(
    *,
    prepared: PreparedPlan,
    capabilities: tuple[ImplementationCapability, ...],
    policies: tuple[OperationExecutionPolicy, ...],
    callback: Callable[..., Any],
    pool: SubmitterPool,
    borrowed_loop: asyncio.AbstractEventLoop,
    limits: ExecutionLimits,
    ingress_limits: IngressLimits,
    assessment_limits: AssessmentLimits,
    assessment_productions: tuple[EvidenceProductionDecl, ...],
    native_request_nodes: dict[str, NodeId] | None,
    physical: Any,
    transport: Any,
) -> ExecutionResult:
    execution = admit_execution_plan(
        context=admit_context_plan(prepared=prepared, bound_context=None),
        capabilities=capabilities,
        policies=policies,
        assessment_productions=assessment_productions,
        ingress_limits=ingress_limits,
        assessment_limits=assessment_limits,
    )

    def dispatch(inputs: tuple[AssociationInput, ...], requests: DelegatedRequests):
        if not native_request_nodes:
            return callback(inputs, requests)
        from anonymizer.engine.ndd._graph_requests import NddRequestBridge
        from anonymizer.engine.replace._native_map_requests import _BRIDGE

        leaves = tuple(
            slot.index
            for slot in prepared.reservation_recipe
            if slot.template in {node for name, node in native_request_nodes.items() if name.startswith("leaf-")}
        )
        token = _BRIDGE.set(NddRequestBridge(requests, {(COL_REPLACEMENT_MAP, 0): leaves}, physical, retries=True))
        try:
            return callback(inputs, requests)
        finally:
            _BRIDGE.reset(token)

    worker = delegated_submitter(dispatch, pool, borrowed_loop)
    handles = tuple(
        ImplementationHandle(
            implementation=cap.implementation,
            operation=cap.operation,
            configuration=cap.configuration,
            local=_ProfileLocal("async_required")
            if native_request_nodes and policy.node == native_request_nodes["probe"]
            else None,
            transport=transport
            if native_request_nodes
            and policy.node
            in {node for name, node in native_request_nodes.items() if name not in {"readiness", "probe"}}
            else None,
            delegated=worker if policy.delegated is not None else None,
            resource=ResourceLease.create(owner="caller", safe_detachment="forbidden", handle=worker)
            if policy.delegated is not None
            else transport.lease
            if native_request_nodes
            and policy.node
            in {node for name, node in native_request_nodes.items() if name not in {"readiness", "probe"}}
            else None,
        )
        for policy, cap in zip(policies, capabilities, strict=True)
        if not native_request_nodes or policy.node != native_request_nodes["readiness"]
    )
    return await run_admitted(execution, capabilities, handles, limits)


async def run_admitted(
    admitted: AdmittedExecutionPlan,
    capabilities: tuple[ImplementationCapability, ...],
    handles: tuple[ImplementationHandle, ...],
    limits: ExecutionLimits,
    *,
    started: Callable[[], None] | None = None,
) -> ExecutionResult:
    """Start an admitted plan; ``started`` runs once the runtime owns the handles."""
    running = await start_execution(
        admitted=admitted,
        capabilities=capabilities,
        services=ExecutionServices(handles=handles, limits=limits, clock=_Clock()),
    )
    if started is not None:
        started()
    return await _await_profile_execution(running)


def _frame_implementation(
    node: NodeId,
    operation: OperationSpec,
    configuration: FrozenConfig,
    *,
    submission_owner: NodeId | None = None,
    local: bool = False,
    aggregate: bool = False,
) -> tuple[ImplementationCapability, OperationExecutionPolicy]:
    capability = ImplementationCapability(
        implementation=ImplementationRef(name=operation.name, revision=1),
        operation=operation,
        configuration=configuration,
        effect="local" if local else "external",
        attribution="aggregate_only" if aggregate else "per_task",
        request_visibility="none",
        pre_dispatch_control="none",
        retry_owner="none" if local else "implementation",
        error_reporting="typed_terminal",
        cancellation="cooperative_ack" if local else "unobservable",
        settlement="synchronous" if local else "uncertain_possible",
        usage="exact" if local else "unknown",
        resource_lifetime="caller_owned" if submission_owner is None else "stateless",
        max_physical_requests_per_activation=0 if local else None,
    )
    policy = OperationExecutionPolicy(
        node=node,
        kind="local" if local else "delegated",
        request=None,
        safe_detachment="forbidden",
        implementations=(
            ExecutionImplementation(
                implementation=capability.implementation,
                configuration=configuration,
                capability=capability,
                request=None,
            ),
        ),
        result_outcomes=frozenset(outcome.name for outcome in operation.outcomes),
        runtime_outcomes=_runtime_rows(tuple(outcome.name for outcome in operation.outcomes), not local),
        delegated=DelegatedRequestContract(
            mode="opaque", child_slots=(), routes=(), followups=frozenset(), usage_requirement="unknown_allowed"
        )
        if submission_owner is None
        else None,
        submission_owner=submission_owner,
    )
    return capability, policy


def qualify_detection_redact(result: ExecutionResult) -> QualificationResult:
    """Reconcile this profile's source checks and physical receipts before release."""
    execution = result._execution
    prepared = execution.context.prepared
    limits = execution.assessment_limits
    admitted = admit_qualification(
        execution=execution,
        productions=execution.assessment_productions,
        limits=QualificationLimits(
            max_productions=limits.max_productions,
            max_submissions=limits.max_assessment_facts,
            max_port_facts=limits.max_port_facts,
            max_consumed_per_assessment=3,
            max_coverage_atoms=0,
            max_verified_evidence=limits.max_assessment_facts,
            max_fixed_point_steps=len(prepared.target_occurrences) + 1,
            max_revision_entries=limits.max_port_facts + prepared.limits.max_capabilities,
            max_provenance_edges=limits.max_provenance_edges,
        ),
    )
    current = evidence_revision_view(
        admitted=admitted,
        result=result,
        artifacts=tuple(reference for reference, _ in result.artifacts),
        configurations=tuple(
            (policy.node, policy.implementations[0].configuration)
            for policy in execution.policies
            if any(production.node == policy.node for production in execution.assessment_productions)
        ),
    )
    return qualify_retained(admitted=admitted, result=result, current=current)


def project_detection_redact(result: ExecutionResult) -> ReplacementResult:
    """Project canonical content under its selected purpose, preserving original errors."""
    for failure in result.submission_failures:
        # Failure projection does not upgrade eligibility or alter physical history.
        # The traceback keeps this frame, and so the canonical execution and any
        # forbidden-detachment lease, alive while the caller handles the failure.
        _retained_execution = result
        raise failure.cause
    if (
        result._execution.context.prepared.configuration.purpose == "execution_only"
        and len(result.frames) == 1
        and type(result.frames[0]) is InvocationFrame
    ):
        frame = result.frames[0]
        if (
            result.invocation_state is not None
            and any(entry.status != "success" for entry in result.invocation_state.entries)
        ) or (
            result.invocation_state is None
            and (not result.states or any(not state.complete for state in result.states) or result.final_outputs)
        ):
            raise AnonymizerWorkflowError("Opaque invocation did not complete")
        return ReplacementResult(
            dataframe=frame.project(result), failed_records=list(frame.failed_records), _execution=result
        )
    if result._execution.context.prepared.configuration.purpose == "execution_only" and len(result.frames) == 1:
        frame = result.frames[0]
        return ReplacementResult(
            dataframe=frame.project(result), failed_records=list(frame.failed_records), _execution=result
        )
    qualified = qualify_detection_redact(result)
    if len(result.frames) != 1 or len(qualified.qualified) != len(result.final_outputs):
        _retained_execution = result
        raise AnonymizerWorkflowError("Detection/Redact execution did not qualify all returned sources")
    frame = result.frames[0]
    return ReplacementResult(
        dataframe=frame.project(result), failed_records=list(frame.failed_records), _execution=result
    )


def ensure_async_engine_loop() -> asyncio.AbstractEventLoop:
    """DataDesigner's shared async engine loop, imported on first use so importing this module needs no engine."""
    from data_designer.engine.dataset_builders.utils.async_concurrency import ensure_async_engine_loop as engine_loop

    return engine_loop()
