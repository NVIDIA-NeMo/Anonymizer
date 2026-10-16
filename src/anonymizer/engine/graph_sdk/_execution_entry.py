# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public execution entry points and running handle."""

from __future__ import annotations

import asyncio
from dataclasses import replace

from anonymizer.engine.graph_sdk._effect_values import (
    EffectCode,
    reject,
    require_instance,
)
from anonymizer.engine.graph_sdk._execution_admission import _validate_services
from anonymizer.engine.graph_sdk._execution_runtime import _InvocationRuntime
from anonymizer.engine.graph_sdk._execution_state import _ExecutionControl
from anonymizer.engine.graph_sdk._execution_values import (
    AdmittedExecutionPlan,
    ExecutionResult,
    ExecutionServices,
    NestedEventLoopError,
)
from anonymizer.engine.graph_sdk._foreign_transport import ForeignLoopTransport
from anonymizer.engine.graph_sdk.blocking import _reject_worker_entry
from anonymizer.engine.graph_sdk.capabilities import (
    ImplementationCapability,
)
from anonymizer.engine.graph_sdk.observations import (
    _CURRENT,
    _PRODUCER,
    ObservationSession,
    _exception_terminal,
    _requirements,
    _Span,
    execution_observation_requirements,
    reserve_observation_scope,
)
from anonymizer.engine.graph_sdk.preparation import recheck_capabilities
from anonymizer.graph._values import (
    InvocationId,
)


class RunningExecution:
    """Live owner of one graph invocation."""

    __slots__ = ("_control", "_task", "invocation")

    def __init__(
        self, invocation: InvocationId, control: _ExecutionControl, task: asyncio.Task[ExecutionResult]
    ) -> None:
        self.invocation = invocation
        self._control = control
        self._task = task

    def request_cancel(self) -> None:
        self._control.cancelled = True

    async def wait(self) -> ExecutionResult:
        return await asyncio.shield(self._task)


async def start_execution(
    *,
    admitted: AdmittedExecutionPlan,
    capabilities: tuple[ImplementationCapability, ...],
    services: ExecutionServices,
    observations: ObservationSession | None = None,
) -> RunningExecution:
    """Recheck a plan before allocating one graph invocation and starting work."""
    if observations is not None:
        try:
            requirements = (
                execution_observation_requirements(execution=admitted, limits=observations.limits)
                if isinstance(admitted, AdmittedExecutionPlan)
                else _requirements(1, 1, observations.limits)
            )
            reserve_observation_scope(session=observations, requirements=requirements)
            if isinstance(admitted, AdmittedExecutionPlan):
                observations._bind(
                    _PRODUCER,
                    data=admitted.context.prepared.data,
                    workflow=admitted.context.prepared.workflow,
                    prepared=admitted.context.prepared,
                    context=admitted.context,
                    execution_plan=admitted,
                )
                if "execution" in observations._owner:
                    observations._disable("producer_contract")
        except BaseException:
            observations._disable("producer_contract")
    span = _Span("execution", session=observations)
    token = _CURRENT.set(
        (span.session, span.ordinal) if span.session is not None and span.ordinal is not None else None
    )
    try:
        _reject_worker_entry()
        require_instance(admitted, AdmittedExecutionPlan)
        require_instance(services, ExecutionServices)
        prepared = admitted.context.prepared
        recheck_capabilities(prepared=prepared, capabilities=capabilities)
        if any(
            item not in capabilities
            for policy in admitted.policies
            for item in (x.capability for x in policy.implementations)
        ):
            reject(EffectCode.CHANGED_FAILOVER_POLICY)
        _validate_services(admitted, services)
        # Callback-held service objects cannot widen the policy selected here.
        services = replace(services, limits=replace(services.limits))
        ForeignLoopTransport.claim_many(
            tuple(handle.transport for handle in services.handles if type(handle.transport) is ForeignLoopTransport)
        )
        invocation = InvocationId.new(plan=prepared.plan)
        control = _ExecutionControl()
        task = asyncio.create_task(_run_execution(admitted, services, invocation, control, span))
        return RunningExecution(invocation, control, task)
    except BaseException as error:
        span.outcome, span.reasons = _exception_terminal(error)
        span.finish()
        raise
    finally:
        _CURRENT.reset(token)


def execute_sync(
    *,
    admitted: AdmittedExecutionPlan,
    capabilities: tuple[ImplementationCapability, ...],
    services: ExecutionServices,
    observations: ObservationSession | None = None,
) -> ExecutionResult:
    """Execute outside an active event loop using the authoritative async core."""
    _reject_worker_entry()
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise NestedEventLoopError("execute_sync cannot run inside an active event loop")

    async def run() -> ExecutionResult:
        running = await start_execution(
            admitted=admitted, capabilities=capabilities, services=services, observations=observations
        )
        return await running.wait()

    return asyncio.run(run())


async def _run_execution(
    admitted: AdmittedExecutionPlan,
    services: ExecutionServices,
    invocation: InvocationId,
    control: _ExecutionControl,
    span: _Span,
) -> ExecutionResult:
    with span:
        result = await _InvocationRuntime(admitted, services, invocation, control).run()
        if span.session is not None:
            span.session._bind(_PRODUCER, result=True, execution=result)
        categories = {fact.category for fact in result.record.terminals}
        span.outcome = next(
            (
                category
                for category in ("inconsistent", "lost", "cancelled", "failure", "blocked")
                if category in categories
            ),
            "success",
        )
        return result
