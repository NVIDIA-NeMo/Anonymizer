# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public execution entry points and running handle."""

from __future__ import annotations

import asyncio

from anonymizer.engine.graph_sdk._effect_values import (
    EffectCode,
    reject,
    require_instance,
)
from anonymizer.engine.graph_sdk._execution_admission import _validate_absences, _validate_services
from anonymizer.engine.graph_sdk._execution_runtime import _InvocationRuntime
from anonymizer.engine.graph_sdk._execution_state import _ExecutionControl
from anonymizer.engine.graph_sdk._execution_values import (
    AdmittedExecutionPlan,
    DecisionResponse,
    DecisionWait,
    ExecutionResult,
    ExecutionServices,
    NestedEventLoopError,
)
from anonymizer.engine.graph_sdk.capabilities import (
    ImplementationCapability,
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

    def pending_decisions(self) -> tuple[DecisionWait, ...]:
        return tuple(self._control.pending.values())

    def submit_decision(self, decision: DecisionResponse) -> None:
        require_instance(decision, DecisionResponse)
        if decision.wait.invocation != self.invocation:
            reject(EffectCode.FOREIGN_OWNER)
        wait = self._control.pending.get(decision.wait)
        if wait is None:
            reject(EffectCode.DUPLICATE if decision.wait in self._control.closed else EffectCode.MISSING)
        if decision.workflow != wait.workflow or decision.artifact != wait.artifact:
            reject(EffectCode.FOREIGN_OWNER)
        if decision.decision not in wait.allowed_decisions:
            reject(EffectCode.UNSUPPORTED)
        if decision.wait in self._control.responses:
            reject(EffectCode.DUPLICATE)
        self._control.responses[decision.wait] = decision

    async def wait(self) -> ExecutionResult:
        return await asyncio.shield(self._task)


async def start_execution(
    *,
    admitted: AdmittedExecutionPlan,
    capabilities: tuple[ImplementationCapability, ...],
    services: ExecutionServices,
) -> RunningExecution:
    """Recheck a plan before allocating one graph invocation and starting work."""
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
    _validate_absences(admitted, services)
    invocation = InvocationId.new(plan=prepared.plan)
    control = _ExecutionControl()
    task = asyncio.create_task(_run_execution(admitted, services, invocation, control))
    return RunningExecution(invocation, control, task)


def execute_sync(
    *,
    admitted: AdmittedExecutionPlan,
    capabilities: tuple[ImplementationCapability, ...],
    services: ExecutionServices,
) -> ExecutionResult:
    """Execute outside an active event loop using the authoritative async core."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise NestedEventLoopError("execute_sync cannot run inside an active event loop")

    async def run() -> ExecutionResult:
        running = await start_execution(admitted=admitted, capabilities=capabilities, services=services)
        return await running.wait()

    return asyncio.run(run())


async def _run_execution(
    admitted: AdmittedExecutionPlan, services: ExecutionServices, invocation: InvocationId, control: _ExecutionControl
) -> ExecutionResult:
    return await _InvocationRuntime(admitted, services, invocation, control).run()
