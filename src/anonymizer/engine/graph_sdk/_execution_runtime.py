# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Invocation scheduling and terminal publication."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import cast

from anonymizer.engine.graph_sdk._effect_values import (
    EffectCode,
    EffectRejected,
    reject,
)
from anonymizer.engine.graph_sdk._execution_cleanup import _acquire_context_resources, _cleanup_execution
from anonymizer.engine.graph_sdk._execution_delegated import _DelegatedOwner
from anonymizer.engine.graph_sdk._execution_dispatch import (
    _external_association_result,
    _immediate_execution_result,
    _mapping,
    _run_adaptive,
    _run_decision,
    _run_external_batch,
    _run_local,
)
from anonymizer.engine.graph_sdk._execution_materialization import (
    _bridge_structural_map_membership,
    _dynamic_membership_value,
    _materialize_bound_context,
    _materialize_subgraph_inputs,
    _materialize_subgraph_outputs,
    _observe_dynamic_membership,
    _operation_inputs,
    _valid_dynamic_membership_result,
)
from anonymizer.engine.graph_sdk._execution_publication import (
    _accept_outputs,
    _canonical_record,
    _capture_assessments,
    _final_outputs,
    _mark_assessment_subjects,
    _validate_assessment_returns,
)
from anonymizer.engine.graph_sdk._execution_state import (
    _ExecutionControl,
    _ExecutionFacts,
    _ExecutionJob,
    _ExecutionTask,
    _FactCheckpoint,
    _group_external_jobs,
)
from anonymizer.engine.graph_sdk._execution_topology import (
    _inherited_artifact_role,
    _operation_has_omitted_context,
)
from anonymizer.engine.graph_sdk._execution_values import (
    _FACT_KEY,
    _RESULT_KEY,
    AdmittedExecutionPlan,
    ArtifactProvenanceFact,
    ExecutionPortFact,
    ExecutionResult,
    ExecutionServices,
    LocalAssessmentResult,
    OperationExecutionPolicy,
    ProvenanceKey,
    PublicationFailureFact,
    RootInputKey,
    RuntimeOutcome,
)
from anonymizer.engine.graph_sdk._request_driver import _RequestAuthority
from anonymizer.engine.graph_sdk.observations import (
    _CURRENT,
    ObservationReason,
    WorkMeasurement,
    _Span,
    _valid_measurement,
)
from anonymizer.engine.graph_sdk.records import UnusedRetryCapacity
from anonymizer.engine.graph_sdk.requests import (
    AssociationResult,
    ExactUsage,
    InvocationRequestScope,
    PhysicalRequestId,
    ScopeCancel,
    SemanticAssociation,
    TextArtifactValue,
    initialize_requests,
    request_receipt,
)
from anonymizer.engine.graph_sdk.resources import (
    ResourceId,
)
from anonymizer.graph._values import (
    ActivationKey,
    ArtifactRef,
    ContractViolation,
    DatumId,
    InvocationId,
    TaskAttemptId,
)
from anonymizer.graph.activation import (
    ActivationEntry,
    ActivationSeed,
    ActivationState,
    CloseUnstarted,
    ObserveTerminal,
    Select,
    Start,
    advance_activation,
    initialize_activation,
)
from anonymizer.graph.workflow import (
    AdmittedWorkflow,
    ArtifactType,
    SubgraphNode,
)


class _InvocationRuntime:
    """Own mutable scheduling, publication, and cleanup for one invocation."""

    def __init__(
        self,
        admitted: AdmittedExecutionPlan,
        services: ExecutionServices,
        invocation: InvocationId,
        control: _ExecutionControl,
    ) -> None:
        self._activation_spans: dict[ActivationKey, _Span] = {}
        self._operation_spans: dict[ActivationKey, _Span] = {}
        self.admitted = admitted
        self.services = services
        self.invocation = invocation
        self.control = control
        self.prepared = self.admitted.context.prepared
        self.policies = {item.node: item for item in self.admitted.policies}
        self.delegated_slots = {
            self.prepared.reservation_recipe[index].template: policy.node
            for policy in self.admitted.policies
            if policy.delegated is not None
            for index in policy.delegated.owned_slots
        }
        self.handles = {
            (item.implementation, item.operation, item.configuration): item for item in self.services.handles
        }
        self.context_leases, self.failed_context_sources = _acquire_context_resources(self.services.context_resources)
        self.states: list[ActivationState] = []
        self.target_keys: dict[DatumId, dict[int, ActivationKey]] = {}
        self._initialize_activations()
        scope = InvocationRequestScope(invocation=self.invocation)
        self.request_authority = _RequestAuthority(
            state=initialize_requests(
                scope=scope,
                hard_limit=self.prepared.configuration.hard_request_limit,
                policies=frozenset((item.request for item in self.admitted.policies if item.request is not None)),
            )
        )
        self.request_resources: dict[PhysicalRequestId, ResourceId] = {}
        self.facts = _ExecutionFacts()
        self.root_inputs: dict[tuple[DatumId, str], ArtifactRef] = {}
        self.subgraph_inputs: dict[tuple[DatumId, ActivationKey, str], ArtifactRef] = {}
        self.subgraph_input_parents: dict[tuple[DatumId, ActivationKey, str], ProvenanceKey] = {}
        self._initialize_root_inputs()
        self.facts.next_artifact = _materialize_bound_context(
            self.admitted,
            self.invocation,
            self.facts.values,
            self.facts.produced,
            self.facts.provenance,
            self.facts.next_artifact,
            self.services.limits,
        )
        self.attempts: dict[ActivationKey, TaskAttemptId] = {}
        self.cancelled_unstarted: set[ActivationKey] = set()
        self.blocked_reasons: dict[ActivationKey, str | UnusedRetryCapacity] = {}
        self.publication_failures: list[PublicationFailureFact] = []
        self.jobs: dict[_ExecutionTask, _ExecutionJob] = {}
        self.physical_jobs: dict[asyncio.Task[object], asyncio.Task[object]] = {}

    def _initialize_activations(self) -> None:
        for target_map in self.prepared.target_occurrences:
            keys: dict[int, ActivationKey] = {}
            for slot in self.prepared.reservation_recipe:
                keys[slot.index] = ActivationKey(
                    invocation=self.invocation,
                    occurrence=target_map.occurrence_offset + slot.index,
                    parent=keys.get(slot.parent_index),
                    iteration=slot.iteration,
                )
            self.target_keys[target_map.target] = keys
            reservations = frozenset(
                (
                    ActivationSeed(template=slot.template, activation=keys[slot.index])
                    for slot in self.prepared.reservation_recipe
                )
            )
            initialized = initialize_activation(
                workflow=self.prepared.workflow,
                invocation=self.invocation,
                reservations=reservations,
                limits=self.prepared.activation_limits,
            )
            self.states.append(
                advance_activation(
                    state=initialized, event=Select(seeds=_initial_seeds(self.prepared.workflow.workflow, reservations))
                )
            )

    def _initialize_root_inputs(self) -> None:
        datum_text = {item.id: item.text for item in self.prepared.data.datums}
        references: dict[tuple[DatumId, ArtifactType], ArtifactRef] = {}
        for bound in self.prepared.bound_inputs:
            lineage = (bound.source, bound.artifact_type)
            reference = references.get(lineage)
            if reference is None:
                reference = ArtifactRef(invocation=self.invocation, key=self.facts.next_artifact, version=1)
                self.facts.next_artifact += 1
                self.facts.values[reference] = TextArtifactValue(text=datum_text[bound.source])
                references[lineage] = reference
            self.root_inputs[bound.target, bound.port] = reference
            self.facts.provenance.append(
                ArtifactProvenanceFact(
                    _key=_FACT_KEY,
                    key=RootInputKey(target=bound.target, port=bound.port),
                    artifact=reference,
                    parents=frozenset(),
                    decision=False,
                )
            )

    async def run(self) -> ExecutionResult:
        try:
            return await self._run()
        finally:
            self._close_observed_activations()
            for span in (*self._operation_spans.values(), *self._activation_spans.values()):
                if not span._done:
                    span.outcome = "unknown"
                    span.reasons = frozenset({"internal_failure"})
                    span.finish()

    def _start_observed_activation(self, activation: ActivationKey, *, operation: bool) -> None:
        parent = self._activation_spans.get(activation.parent)
        span = _Span("activation", parent=None if parent is None else parent.ordinal)
        self._activation_spans[activation] = span
        if operation:
            self._operation_spans[activation] = _Span("operation", session=span.session, parent=span.ordinal)

    def _close_observed_activations(self) -> None:
        for state in self.states:
            for entry in state.entries:
                span = self._activation_spans.get(entry.activation)
                if (
                    span is not None
                    and not span._done
                    and entry.status in {"success", "failure", "cancelled", "lost", "blocked", "inconsistent"}
                ):
                    span.outcome = entry.status
                    span.finish()

    async def _run(self) -> ExecutionResult:
        while True:
            scheduled = self._schedule_ready()
            self._close_observed_activations()
            if all((state.complete for state in self.states)) and (not self.jobs):
                break
            if not self.jobs:
                if not scheduled:
                    break
                continue
            await self._process_completed()
        return await self._finalize()

    def _schedule_ready(self) -> bool:
        scheduled = False
        external_jobs: list[_ExecutionJob] = []
        for state_index, (target_map, state) in enumerate(
            zip(self.prepared.target_occurrences, self.states, strict=True)
        ):
            ready = sorted(
                (item for item in state.entries if item.status == "ready"), key=lambda item: item.activation.occurrence
            )
            for entry in ready:
                scheduled = self._schedule_entry(state_index, target_map.target, entry, external_jobs) or scheduled
        self._dispatch_external(external_jobs)
        return scheduled

    def _schedule_entry(
        self, state_index: int, target: DatumId, entry: ActivationEntry, external_jobs: list[_ExecutionJob]
    ) -> bool:
        if entry.template in self.delegated_slots:
            owner = next(
                item
                for item in self.states[state_index].entries
                if item.template == self.delegated_slots[entry.template]
                and item.activation.parent == entry.activation.parent
            )
            if self.control.cancelled or owner.status not in {"ready", "unstarted", "running"}:
                if self.control.cancelled:
                    self.cancelled_unstarted.add(entry.activation)
                self.states[state_index] = advance_activation(
                    state=self.states[state_index],
                    event=CloseUnstarted(activation=entry.activation, category="blocked"),
                )
                return True
            return False
        policy = self.policies.get(entry.template)
        if policy is None:
            return self._start_subgraph(state_index, target, entry)
        decision_jobs = sum((item.policy.kind == "decision" for item in self.jobs.values()))
        if policy.kind == "decision" and decision_jobs >= self.services.decision_limits.max_pending:
            if self.services.decision_limits.max_pending == 0:
                reject(EffectCode.PENDING_LIMIT)
            return False
        if _operation_has_omitted_context(self.admitted, target, entry.template):
            self.blocked_reasons[entry.activation] = "omitted_optional"
            self.states[state_index] = advance_activation(
                state=self.states[state_index],
                event=CloseUnstarted(activation=entry.activation, category="blocked"),
            )
            return True
        waiting_activations = {wait.activation for wait in self.control.pending.values()}
        local_jobs = sum(
            (
                item.policy.kind != "external" and item.activation not in waiting_activations
                for item in self.jobs.values()
            )
        )
        remote_jobs = len(
            {self.physical_jobs.get(task, task) for task, item in self.jobs.items() if item.policy.kind == "external"}
        ) + len(_group_external_jobs(external_jobs))
        joins_pending_batch = any(
            (item.node == entry.template and item.policy == policy for item in external_jobs)
        ) and all((item.capability.attribution == "keyed_shared_request" for item in policy.implementations))
        if policy.kind != "external":
            if self.services.limits.max_local_in_flight == 0:
                reject(EffectCode.LIMIT_EXCEEDED)
            if local_jobs >= self.services.limits.max_local_in_flight:
                return False
        remote_capacity_stalled = (
            policy.kind == "external"
            and (not joins_pending_batch)
            and (
                max(len(self.request_authority.state.remote_outstanding), remote_jobs)
                >= self.services.limits.max_remote_outstanding
            )
        )
        if remote_capacity_stalled and remote_jobs:
            return False
        self._start_operation(state_index, target, entry, policy, external_jobs, remote_capacity_stalled)
        return True

    def _start_subgraph(self, state_index: int, target: DatumId, entry: ActivationEntry) -> bool:
        if isinstance(self.admitted.nodes[entry.template], SubgraphNode):
            try:
                _materialize_subgraph_inputs(
                    self.admitted,
                    target,
                    self.states[state_index],
                    entry.activation,
                    entry.template,
                    self.root_inputs,
                    self.subgraph_inputs,
                    self.subgraph_input_parents,
                    self.facts.produced,
                    self.facts.provenance,
                )
            except EffectRejected as exc:
                if exc.code != EffectCode.MISSING:
                    raise
                self.states[state_index] = advance_activation(
                    state=self.states[state_index],
                    event=CloseUnstarted(activation=entry.activation, category="blocked"),
                )
                return True
            self._start_observed_activation(entry.activation, operation=False)
            self.states[state_index] = advance_activation(
                state=self.states[state_index], event=Start(activation=entry.activation)
            )
            return True
        return False

    def _start_operation(
        self,
        state_index: int,
        target: DatumId,
        entry: ActivationEntry,
        policy: OperationExecutionPolicy,
        external_jobs: list[_ExecutionJob],
        remote_capacity_stalled: bool,
    ) -> None:
        if self.control.cancelled:
            self.cancelled_unstarted.add(entry.activation)
            self.states[state_index] = advance_activation(
                state=self.states[state_index], event=CloseUnstarted(activation=entry.activation, category="blocked")
            )
            return
        attempt = TaskAttemptId.new(activation=entry.activation)
        association = SemanticAssociation(task=attempt)
        try:
            inputs, input_parents = _operation_inputs(
                self.admitted,
                target,
                entry.template,
                entry.activation,
                self.states[state_index],
                association,
                self.root_inputs,
                self.subgraph_inputs,
                self.subgraph_input_parents,
                self.facts.produced,
                self.facts.values,
                self.facts.provenance,
            )
        except EffectRejected as exc:
            if exc.code != EffectCode.MISSING:
                raise
            self.states[state_index] = advance_activation(
                state=self.states[state_index], event=CloseUnstarted(activation=entry.activation, category="blocked")
            )
            return
        self._start_observed_activation(entry.activation, operation=True)
        self.states[state_index] = advance_activation(
            state=self.states[state_index], event=Start(activation=entry.activation)
        )
        self.attempts[entry.activation] = attempt
        implementation = policy.implementations[0]
        job = _ExecutionJob(
            state_index=state_index,
            target=target,
            activation=entry.activation,
            node=entry.template,
            policy=policy,
            implementation=implementation,
            inputs=inputs,
            input_parents=input_parents,
            association=association,
        )
        self._record_input_ports(job)
        span = self._operation_spans[entry.activation]
        token = _CURRENT.set(
            (span.session, span.ordinal) if span.session is not None and span.ordinal is not None else None
        )
        try:
            self._launch_job(job, external_jobs, remote_capacity_stalled)
        finally:
            _CURRENT.reset(token)

    def _record_input_ports(self, job: _ExecutionJob) -> None:
        self.facts.input_parents.extend(
            (job.target, job.activation, port, parent) for port, parent in job.input_parents.items()
        )
        for input_artifact in job.inputs[0].inputs:
            if input_artifact.artifact is None:
                reject(EffectCode.CONTRADICTORY)
            self.facts.ports.append(
                ExecutionPortFact(
                    _key=_FACT_KEY,
                    activation=job.activation,
                    node=job.node,
                    target=job.target,
                    port=input_artifact.port,
                    artifact=input_artifact.artifact,
                    artifact_type=input_artifact.artifact_type,
                    role="decision"
                    if job.policy.kind == "decision"
                    and input_artifact.port
                    == next((item.artifact_port for item in self.admitted.decisions if item.node == job.node))
                    else _inherited_artifact_role(
                        job.input_parents.get(input_artifact.port), self.facts.provenance, self.facts.ports
                    ),
                )
            )

    def _launch_job(
        self, job: _ExecutionJob, external_jobs: list[_ExecutionJob], remote_capacity_stalled: bool
    ) -> None:
        policy = job.policy
        implementation = job.implementation
        handle = self.handles[
            implementation.implementation, implementation.capability.operation, implementation.configuration
        ]
        if policy.kind == "external":
            if remote_capacity_stalled:
                coroutine = _immediate_execution_result(_mapping(policy, "request_limit_exhausted", None, None))
            else:
                adaptive = next(
                    (item for item in self.admitted.context.adaptive_retrievals if item.node == job.node), None
                )
                if adaptive is None:
                    external_jobs.append(job)
                    return
                if adaptive.source in self.failed_context_sources:
                    coroutine = _immediate_execution_result(
                        _mapping(policy, "failure", None, "implementation_exception")
                    )
                else:
                    coroutine = _run_adaptive(
                        self.admitted,
                        policy,
                        adaptive,
                        job.association,
                        job.inputs,
                        self.request_authority,
                        self.context_leases,
                        self.control,
                        self.services.limits,
                        self.request_resources,
                    )
        elif policy.kind == "delegated":
            coroutine = _DelegatedOwner(self, job).run(handle)
        elif policy.kind == "decision":
            coroutine = _run_decision(
                self.admitted, policy, handle, job.association, job.inputs, job.activation, self.services, self.control
            )
        else:
            coroutine = _run_local(
                policy, handle, job.association, job.inputs, self.control, self.admitted.ingress_limits
            )
        self.jobs[asyncio.create_task(coroutine)] = job

    def _dispatch_external(self, external_jobs: list[_ExecutionJob]) -> None:
        groups = _group_external_jobs(external_jobs)
        for group in groups:
            physical = asyncio.create_task(
                _run_external_batch(
                    self.admitted,
                    group[0].policy,
                    self.handles,
                    tuple((value for job in group for value in job.inputs)),
                    self.request_authority,
                    self.control,
                    self.services.limits,
                    self.request_resources,
                )
            )
            for job in group:
                task = asyncio.create_task(_external_association_result(physical, job.association))
                self.jobs[task] = job
                self.physical_jobs[task] = physical

    async def _process_completed(self) -> None:
        scheduler_wakeup = asyncio.create_task(self.control.scheduler_changed.wait())
        try:
            completed, _ = await asyncio.wait((*self.jobs, scheduler_wakeup), return_when=asyncio.FIRST_COMPLETED)
        finally:
            scheduler_wakeup.cancel()
            with suppress(asyncio.CancelledError):
                await scheduler_wakeup
        if scheduler_wakeup in completed:
            self.control.scheduler_changed.clear()
        pending_completed = {task for task in self.jobs if task in completed}
        while pending_completed:
            peers = await self._complete_group(next(iter(pending_completed)))
            pending_completed.difference_update(peers)

    async def _complete_group(self, first: _ExecutionTask) -> list[_ExecutionTask]:
        physical = self.physical_jobs.get(first)
        peers = (
            [task for task in self.jobs if self.physical_jobs.get(task) is physical]
            if physical is not None
            else [first]
        )
        await asyncio.gather(*peers)
        for task in peers:
            job = self.jobs.pop(task)
            self.physical_jobs.pop(task, None)
            mapping, results, assessments = task.result()
            accepted = mapping.condition == "result"
            mapping, results, checkpoint, staged_result = self._stage_outputs(job, mapping, results, assessments)
            mapping = self._advance_completed(job, mapping, results, checkpoint, staged_result)
            if (
                job.policy.delegated is not None
                and mapping.condition == "result"
                and mapping.category == "success"
                and any(key[0] == job.target and key[1] == job.activation for key in self.facts.produced)
            ):
                unused = self._unused_tail(job)
                if unused is not None:
                    self.blocked_reasons[unused[0]] = unused[1]
            span = self._operation_spans[job.activation]
            span.outcome = mapping.category
            if mapping.failure is not None:
                span.reasons = frozenset(
                    {"malformed_response" if mapping.failure == "malformed_response" else "provider_failure"}
                )
            elif mapping.condition in {"deadline_exhausted", "budget_exhausted", "request_limit_exhausted"}:
                span.reasons = frozenset(
                    {
                        cast(
                            ObservationReason,
                            {
                                "deadline_exhausted": "deadline",
                                "budget_exhausted": "budget_stopped",
                                "request_limit_exhausted": "request_limit_stopped",
                            }[mapping.condition],
                        )
                    }
                )
            handle = self.handles[
                job.implementation.implementation,
                job.implementation.capability.operation,
                job.implementation.configuration,
            ]
            handle = self.control.measured_handles.pop(job.association, handle)
            if span.session is not None and span.session._failure is None and handle.measurement is not None:
                try:
                    measurement = handle.measurement.terminal_measurement()
                    route = (
                        "context"
                        if any(item.node == job.node for item in self.admitted.context.adaptive_retrievals)
                        else job.policy.kind
                    )
                    if type(measurement) is WorkMeasurement and any(
                        type(getattr(measurement, name)) is int
                        and getattr(measurement, name) > span.session.limits.max_count
                        for name in ("input_count", "output_count", "artifact_count", "artifact_bytes", "request_count")
                    ):
                        span.session._disable("capacity_contract")
                    elif (
                        not _valid_measurement(measurement, span.session.limits.max_count) or measurement.route != route
                    ):
                        span.session._disable("producer_contract")
                    else:
                        span.measurement = measurement
                except BaseException:
                    span.reasons = span.reasons | frozenset[ObservationReason]({"internal_failure"})
            span.finish()
            self._close_observed_activations()
            if mapping.category == "blocked" and mapping.condition == "artifact_limit_exhausted":
                self.blocked_reasons[job.activation] = mapping.condition
            if mapping.condition == "artifact_limit_exhausted" or accepted and mapping.condition != "result":
                self.publication_failures.append(
                    PublicationFailureFact(
                        association=job.association,
                        target=job.target,
                        condition=mapping.condition,
                        failure=mapping.failure,
                    )
                )
        return peers

    def _unused_tail(self, job: _ExecutionJob) -> tuple[ActivationKey, UnusedRetryCapacity] | None:
        """Check provisional capacity against this invocation's live facts, without stamping."""
        contract = job.policy.delegated
        state = self.request_authority.state
        if not contract or not contract.optional_tail or self.control.cancelled or state.cancelled:
            return None
        if (
            state.reserved
            or state.denials
            or state.defects
            or state.cancel_requested
            or state.local_in_flight
            or state.remote_outstanding
        ):
            return None
        g, q = (self.target_keys[job.target][index] for index in contract.child_slots)
        entries = {item.activation: item for item in self.states[job.state_index].entries}
        if (
            q in self.attempts
            or entries[q].status not in {"ready", "unstarted", "blocked"}
            or entries[g].status != "success"
        ):
            return None
        chains = [item for item in state.chains if item.associations == {job.association}]
        if len(chains) != 1:
            return None
        history = [row for row in state.dispatches if row.child is not None and row.child.chain == chains[0].chain]
        expected_count = 1 + (contract.readiness is not None)
        if len(history) != 1 or len(state.dispatches) != expected_count:
            return None
        request = history[0]
        child = request.child
        if child is None or child.leaf != self.attempts.get(g) or child.source_kind != "initial":
            return None
        terminal = [row for row in state.terminals if row.request == request.request]
        settlement = [row for row in state.settlements if row.request == request.request]
        if (
            len(terminal) != 1
            or terminal[0].category != "success"
            or len(terminal[0].results) != 1
            or terminal[0].results[0].category != "success"
            or len(settlement) != 1
            or settlement[0].disposition != "completed"
            or settlement[0].remote_stopped is not True
            or (contract.usage_requirement == "exact" and not isinstance(settlement[0].usage, ExactUsage))
            or len(state.terminals) != expected_count
            or len(state.settlements) != expected_count
        ):
            return None
        if contract.readiness is not None and any(
            entries[self.target_keys[job.target][index]].status != "success" for index in contract.readiness.slots
        ):
            return None
        return q, UnusedRetryCapacity(owner=job.association.task, after=self.attempts[g])

    def _stage_outputs(
        self,
        job: _ExecutionJob,
        mapping: RuntimeOutcome,
        results: tuple[AssociationResult, ...],
        assessments: tuple[LocalAssessmentResult, ...],
    ) -> tuple[RuntimeOutcome, tuple[AssociationResult, ...], _FactCheckpoint | None, bool]:
        staged_result = False
        checkpoint: _FactCheckpoint | None = None
        if mapping.condition == "result":
            if job.policy.kind == "decision" and (not results):
                results = (
                    AssociationResult(
                        association=job.association,
                        outcome=mapping.outcome or "",
                        outputs=(),
                        consumed_context_ports=frozenset(),
                    ),
                )
            if not _validate_assessment_returns(self.admitted, job.association, job.node, mapping, assessments):
                mapping = _mapping(job.policy, "failure", None, "malformed_response")
            elif not _valid_dynamic_membership_result(self.admitted, job.node, mapping, results):
                mapping = _mapping(job.policy, "failure", None, "malformed_response")
            else:
                checkpoint = self.facts.checkpoint()
                output_status, created = _accept_outputs(
                    self.admitted,
                    self.invocation,
                    job.target,
                    job.activation,
                    job.node,
                    job.implementation.capability.operation,
                    mapping,
                    job.association,
                    job.inputs,
                    job.input_parents,
                    results,
                    self.facts,
                    self.services.limits,
                )
                if output_status != "valid":
                    self.facts.restore(checkpoint)
                    mapping = _mapping(
                        job.policy,
                        "artifact_limit_exhausted" if output_status == "limit" else "failure",
                        None,
                        None if output_status == "limit" else "malformed_response",
                    )
                elif not _capture_assessments(
                    self.admitted,
                    job.implementation,
                    job.association,
                    job.activation,
                    job.node,
                    mapping,
                    assessments,
                    job.target,
                    self.services,
                    self.facts,
                ):
                    self.facts.restore(checkpoint)
                    mapping = _mapping(job.policy, "failure", None, "malformed_response")
                else:
                    _mark_assessment_subjects(self.admitted, job.node, mapping, job.target, job.activation, self.facts)
                    staged_result = True
                del created
        return (mapping, results, checkpoint, staged_result)

    def _advance_completed(
        self,
        job: _ExecutionJob,
        mapping: RuntimeOutcome,
        results: tuple[AssociationResult, ...],
        checkpoint: _FactCheckpoint | None,
        staged_result: bool,
    ) -> RuntimeOutcome:
        prior_state = self.states[job.state_index]
        candidate_state = advance_activation(
            state=prior_state,
            event=ObserveTerminal(activation=job.activation, outcome=mapping.outcome, category=mapping.category),
        )
        if staged_result:
            try:
                membership = _dynamic_membership_value(self.admitted, job.node, mapping.outcome, results)
                if membership is not None:
                    candidate_state = _observe_dynamic_membership(
                        self.admitted,
                        candidate_state,
                        job.target,
                        job.activation,
                        job.node,
                        mapping.outcome,
                        membership,
                        self.facts,
                        self.services.limits,
                    )
            except (ContractViolation, EffectRejected) as exc:
                assert checkpoint is not None
                self.facts.restore(checkpoint)
                condition = (
                    "artifact_limit_exhausted"
                    if isinstance(exc, EffectRejected) and exc.code == EffectCode.LIMIT_EXCEEDED
                    else "failure"
                )
                mapping = _mapping(
                    job.policy,
                    condition,
                    None,
                    None if condition == "artifact_limit_exhausted" else "malformed_response",
                )
                candidate_state = advance_activation(
                    state=prior_state,
                    event=ObserveTerminal(
                        activation=job.activation, outcome=mapping.outcome, category=mapping.category
                    ),
                )
        try:
            while True:
                _materialize_subgraph_outputs(
                    self.admitted,
                    job.target,
                    candidate_state,
                    self.facts,
                    self.subgraph_input_parents,
                )
                candidate_state, bridged = _bridge_structural_map_membership(
                    self.admitted, candidate_state, job.target, self.facts, self.services.limits
                )
                if not bridged:
                    break
        except (ContractViolation, EffectRejected) as exc:
            if not staged_result or checkpoint is None:
                raise
            self.facts.restore(checkpoint)
            condition = (
                "artifact_limit_exhausted"
                if isinstance(exc, EffectRejected) and exc.code == EffectCode.LIMIT_EXCEEDED
                else "failure"
            )
            mapping = _mapping(
                job.policy, condition, None, None if condition == "artifact_limit_exhausted" else "malformed_response"
            )
            candidate_state = advance_activation(
                state=prior_state,
                event=ObserveTerminal(activation=job.activation, outcome=mapping.outcome, category=mapping.category),
            )
        if checkpoint is not None:
            self.facts.commit(checkpoint)
        self.states[job.state_index] = candidate_state
        return mapping

    async def _finalize(self) -> ExecutionResult:
        if self.control.cancelled:
            self.request_authority.apply(ScopeCancel())
        cleanup, cleanup_associations = await _cleanup_execution(
            self.admitted,
            self.services.handles,
            self.context_leases,
            self.request_authority.state,
            self.request_resources,
        )
        record = _canonical_record(
            self.admitted,
            self.invocation,
            tuple(self.states),
            self.target_keys,
            self.attempts,
            frozenset(self.facts.values),
            frozenset(self.cancelled_unstarted),
            self.blocked_reasons,
        )
        final_outputs = _final_outputs(self.prepared, tuple(self.states), self.facts.produced, self.facts.provenance)
        return ExecutionResult(
            _key=_RESULT_KEY,
            _execution=self.admitted,
            _input_parents=tuple(self.facts.input_parents),
            _passthrough_parents=tuple(
                (target, activation, port, parent)
                for (target, activation, port), parent in self.facts.passthrough_parents.items()
            ),
            record=record,
            states=tuple(self.states),
            requests=request_receipt(self.request_authority.state),
            cleanup=cleanup,
            pending_decisions=(),
            artifacts=tuple(self.facts.values.items()),
            assessments=tuple(self.facts.assessments),
            ports=tuple(self.facts.ports),
            final_outputs=final_outputs,
            provenance=tuple(self.facts.provenance),
            cleanup_associations=cleanup_associations,
            publication_failures=tuple(self.publication_failures),
        )


def _initial_seeds(workflow: AdmittedWorkflow, reservations: frozenset[ActivationSeed]) -> frozenset[ActivationSeed]:
    blocked = {member for choice in workflow.choices for branch in choice.branches for member in branch.members}
    return frozenset(seed for seed in reservations if seed.activation.parent is None and seed.template not in blocked)
