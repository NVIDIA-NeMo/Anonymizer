# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Route-neutral execution kernel: plan preparation, admit/start, synchronous hosting and projection.

Every adapter route (detection, replace, rewrite, evaluation, detection transforms) prepares a
fixed-operation plan, admits and starts it on the single Graph SDK executor, runs it on a private
event loop on the caller's thread with every synchronous body inline, and projects the canonical
frame back to a DataFrame.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from time import monotonic_ns
from typing import Any, cast

from anonymizer.engine.graph_sdk._activation_values import ActivationLimits
from anonymizer.engine.graph_sdk._execution_delegated import (
    InlineChannel,
    InlineSubmitter,
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
from anonymizer.engine.graph_sdk.requests import AssociationInput
from anonymizer.engine.graph_sdk.resources import ResourceLease
from anonymizer.engine.harness.frames import InvocationFrame
from anonymizer.engine.replace.replace_runner import ReplacementResult
from anonymizer.interface.errors import AnonymizerWorkflowError


class _Clock:
    def now_ns(self) -> int:
        return monotonic_ns()


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
) -> WorkerDelegatedSubmitter | InlineSubmitter:
    """Run the delegated body inline on the synchronous caller (kernel-hosted) or on ``pool`` (standalone use)."""
    if isinstance(pool, InlineChannel):
        return InlineSubmitter(callback=callback, channel=pool, borrowed_loop=borrowed_loop)
    return WorkerDelegatedSubmitter(
        callback=callback,
        pool=pool,
        affinity=BlockingAffinity(cross_thread=True, concurrency_safe=False, close_on_event_loop=False),
        borrowed_loop=borrowed_loop,
    )


def _run_sync_profile(
    operation: Callable[[InlineChannel], Coroutine[Any, Any, ReplacementResult]],
) -> ReplacementResult:
    """Run one graph invocation on a private event loop on the caller's thread; bodies run inline.

    No asyncio Runner, so no SIGINT handler: Ctrl-C stays a KeyboardInterrupt and lands where the body is
    waiting (inside DataDesigner, as in 0.4.0). A caller that already runs a loop (Jupyter, ``asyncio.run``)
    is blocked by this synchronous call anyway, so its running-loop marker is set aside for the private
    loop and restored around each body and on return.
    """
    channel = InlineChannel()
    loop = asyncio.new_event_loop()
    failure: BaseException | None = None
    channel.enter()
    try:
        root = loop.create_task(operation(channel))
        channel.root = root
        try:
            result = loop.run_until_complete(root)
        except BaseException as error:
            if channel.interrupted is None:
                raise
            failure = error
    finally:
        try:
            _close_private_loop(loop)
        finally:
            channel.exit()
    if channel.interrupted is not None:
        owner = retained_execution(failure) if failure is not None else result._execution
        if owner is not None:
            _retained_execution = owner
        raise channel.interrupted
    return result


def _close_private_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Cancel and settle whatever the invocation left on its loop, then close it (best effort, like a Runner)."""
    try:
        pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.run_until_complete(loop.shutdown_default_executor())
    finally:
        loop.close()


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
) -> ExecutionResult:
    """Run a single grouped submission with its admitted observed operations."""
    execution = admit_execution_plan(
        context=admit_context_plan(prepared=prepared, bound_context=None),
        capabilities=capabilities,
        policies=policies,
        assessment_productions=assessment_productions,
        ingress_limits=ingress_limits,
        assessment_limits=assessment_limits,
    )
    worker = delegated_submitter(callback, pool, borrowed_loop)
    handles = tuple(
        ImplementationHandle(
            implementation=cap.implementation,
            operation=cap.operation,
            configuration=cap.configuration,
            local=None,
            transport=None,
            delegated=worker if policy.delegated is not None else None,
            resource=ResourceLease.create(owner="caller", safe_detachment="forbidden", handle=worker)
            if policy.delegated is not None
            else None,
        )
        for policy, cap in zip(policies, capabilities, strict=True)
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
    if result._execution.context.prepared.configuration.purpose != "execution_only" or len(result.frames) != 1:
        _retained_execution = result
        raise AnonymizerWorkflowError("Detection/Redact execution did not qualify all returned sources")
    frame = result.frames[0]
    return ReplacementResult(
        dataframe=frame.project(result), failed_records=list(frame.failed_records), _execution=result
    )
