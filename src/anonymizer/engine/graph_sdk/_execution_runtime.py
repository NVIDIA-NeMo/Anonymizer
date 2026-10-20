# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Invocation scheduling and terminal publication."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from typing import TypeVar, cast

from anonymizer.engine.graph_sdk._activation_reducer import advance_activation
from anonymizer.engine.graph_sdk._activation_values import (
    ActivationEntry,
    ActivationState,
    CloseUnstarted,
    ObserveTerminal,
    Start,
)
from anonymizer.engine.graph_sdk._effect_values import (
    EffectCode,
    EffectRejected,
    reject,
)
from anonymizer.engine.graph_sdk._execution_cleanup import _cleanup_execution
from anonymizer.engine.graph_sdk._execution_delegated import _DelegatedOwner
from anonymizer.engine.graph_sdk._execution_dispatch import (
    _immediate_execution_result,
    _mapping,
    _run_external_batch,
    _run_local,
)
from anonymizer.engine.graph_sdk._execution_frame import _InvocationFrame
from anonymizer.engine.graph_sdk._execution_materialization import (
    _materialize_bound_context,
    _materialize_structure,
    _materialize_subgraph_inputs,
    _membership_value,
    _observe_dynamic_membership,
    _operation_inputs,
)
from anonymizer.engine.graph_sdk._execution_publication import (
    _accept_invocation_frame,
    _accept_outputs,
    _assessment_results,
    _canonical_record,
    _capture_assessments,
    _final_outputs,
)
from anonymizer.engine.graph_sdk._execution_state import (
    _ExecutionControl,
    _ExecutionFacts,
    _ExecutionTask,
    _OperationJob,
)
from anonymizer.engine.graph_sdk._execution_topology import (
    _inherited_artifact_role,
    _operation_has_omitted_context,
)
from anonymizer.engine.graph_sdk._execution_values import (
    _FACT_KEY,
    _RESULT_KEY,
    AdmittedExecutionPlan,
    DelegatedRequestContract,
    ExecutionPortFact,
    ExecutionResult,
    ExecutionServices,
    ImplementationResourceRegistration,
    LocalAssessmentResult,
    LocalCompleted,
    OperationExecutionPolicy,
    OperationOutputKey,
    ProvenanceKey,
    PublicationFailureFact,
    RuntimeOutcome,
    SubmissionFailure,
)
from anonymizer.engine.graph_sdk._graph_values import (
    ActivationKey,
    ContractViolation,
    DatumId,
    InvocationId,
    TaskAttemptId,
)
from anonymizer.engine.graph_sdk._request_driver import _RequestAuthority
from anonymizer.engine.graph_sdk._workflow_values import NodeId, SubgraphNode
from anonymizer.engine.graph_sdk.observations import (
    _CURRENT,
    ObservationReason,
    WorkMeasurement,
    _Span,
    _valid_measurement,
)
from anonymizer.engine.graph_sdk.records import UnusedRetryCapacity
from anonymizer.engine.graph_sdk.requests import (
    AssociationInput,
    AssociationResult,
    ExactUsage,
    InvocationRequestScope,
    PhysicalRequestId,
    PortArtifact,
    ScopeCancel,
    SemanticAssociation,
    initialize_requests,
    request_receipt,
)
from anonymizer.engine.graph_sdk.resources import (
    ResourceId,
)

_Scope = TypeVar("_Scope", bound=DatumId | InvocationId)


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
        self.delegated_slots.update(
            (policy.node, policy.submission_owner)
            for policy in self.admitted.policies
            if policy.submission_owner is not None
        )
        self.handles = {
            (item.implementation, item.operation, item.configuration): item for item in self.services.handles
        }
        # Capture every validated resource-bearing handle before acquiring providers
        # or executing operations. Stateless handles have no cleanup obligation.
        self.implementation_resources = tuple(
            ImplementationResourceRegistration(
                invocation=invocation,
                implementation=handle.implementation,
                operation=handle.operation,
                configuration=handle.configuration,
                resource=handle.resource.resource,
                owner=handle.resource.owner,
                safe_detachment=handle.resource.safe_detachment,
                _lease=handle.resource,
            )
            for handle in services.handles
            if handle.resource is not None
        )
        self.facts = _ExecutionFacts()
        self.frame = _InvocationFrame(
            self.prepared,
            invocation,
            self.facts,
            target_workflow=admitted.target_workflow,
            invocation_workflow=admitted.invocation_workflow,
        )
        scope = InvocationRequestScope(invocation=self.invocation)
        self.request_authority = _RequestAuthority(
            state=initialize_requests(
                scope=scope,
                hard_limit=self.prepared.configuration.hard_request_limit,
                policies=frozenset(
                    request
                    for item in self.admitted.policies
                    for request in (item.request, None if item.delegated is None else item.delegated.observed_request)
                    if request is not None
                ),
            )
        )
        self.request_resources: dict[PhysicalRequestId, ResourceId] = {}
        self.subgraph_input_parents: dict[tuple[DatumId, ActivationKey, str], ProvenanceKey] = {}
        _materialize_bound_context(
            self.admitted,
            self.invocation,
            self.facts,
            self.services.limits,
        )
        self.operations: dict[ActivationKey, _OperationJob[DatumId | InvocationId]] = {}
        self.cancelled_unstarted: set[ActivationKey] = set()
        self.blocked_reasons: dict[ActivationKey, str | UnusedRetryCapacity] = {}
        self.publication_failures: list[PublicationFailureFact] = []
        self.submission_failures: list[SubmissionFailure] = []
        self.submissions: dict[_ExecutionTask, tuple[_OperationJob[DatumId | InvocationId], ...]] = {}

    async def run(self) -> ExecutionResult:
        try:
            while True:
                scheduled = self._schedule_ready()
                self._close_observed_activations()
                if all((state.complete for state in self.frame.all_states)) and (not self.submissions):
                    break
                if not self.submissions:
                    if not scheduled:
                        break
                    continue
                await self._process_completed()
            return await self._finalize()
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

    def _close_observed_activations(self, changed: ActivationState | None = None) -> None:
        for state in self.frame.all_states if changed is None else (changed,):
            for entry in state.entries:
                span = self._activation_spans.get(entry.activation)
                if (
                    span is not None
                    and not span._done
                    and entry.status in {"success", "failure", "cancelled", "lost", "blocked", "inconsistent"}
                ):
                    span.outcome = entry.status
                    span.finish()

    def _schedule_ready(self) -> bool:
        if self.control.cancelled:
            changed = False
            for source in (*self.frame.sources.values(), self.frame.invocation_source):
                if source is None:
                    continue
                for previous in tuple(source.state.entries):
                    entry = source.entry(previous.activation)
                    if entry.status in {"ready", "unstarted"}:
                        self.cancelled_unstarted.add(entry.activation)
                        source.state = advance_activation(
                            state=source.state, event=CloseUnstarted(activation=entry.activation, category="blocked")
                        )
                        changed = True
            return changed

        # A frontier consumes capacity once per physical batch. No work runs until
        # this synchronous traversal has bound inputs and committed its entries.
        batches: dict[NodeId | ActivationKey, list[_OperationJob[DatumId]]] = {}
        pending_activations = {job.activation for jobs in self.submissions.values() for job in jobs}
        local = sum(any(job.policy.kind != "external" for job in jobs) for jobs in self.submissions.values())
        remote = sum(jobs[0].policy.kind == "external" for jobs in self.submissions.values())
        scopes = tuple(self.frame.sources) or ((self.invocation,) if self.frame.invocation_source is not None else ())
        successors: dict[NodeId, list[NodeId]] = {}
        for edge in self.prepared.workflow.sequence:
            successors.setdefault(edge.before, []).append(edge.after)
        shared_predecessors: dict[NodeId, ActivationEntry] = {}
        shared = self.frame.invocation_source
        if shared is not None:
            # This synchronous frontier cannot change the shared state before
            # returning. Preserve its first unfinished predecessor, including
            # the order between waiting and failed predecessors.
            for row in shared.state.entries:
                if row.status != "success" and self.blocked_reasons.get(row.activation) != "omitted_optional":
                    for destination in successors.get(row.template, ()):
                        shared_predecessors.setdefault(destination, row)
        scheduled = False
        for scope in scopes:
            source = self.frame.occurrence(scope)
            indexed_state: ActivationState | None = None
            entries_by_template: dict[NodeId, list[ActivationEntry]] = {}
            ready = sorted(
                (entry for entry in source.state.entries if entry.status == "ready"),
                key=lambda entry: entry.activation.occurrence,
            )
            for entry in ready:
                if self.control.cancelled:
                    self.cancelled_unstarted.add(entry.activation)
                    source.state = advance_activation(
                        state=source.state, event=CloseUnstarted(activation=entry.activation, category="blocked")
                    )
                    scheduled = True
                    continue
                blocked = False
                if isinstance(scope, DatumId):
                    predecessor = shared_predecessors.get(entry.template)
                    if predecessor is not None:
                        if predecessor.status in {"ready", "unstarted", "running"}:
                            continue
                        blocked = True
                    elif entry.template in self.delegated_slots:
                        # Refresh from the authoritative state after any local
                        # transition; repeated template occurrences keep their
                        # original order and lexical-parent qualification.
                        if indexed_state is not source.state:
                            indexed_state = source.state
                            entries_by_template = {}
                            for row in indexed_state.entries:
                                entries_by_template.setdefault(row.template, []).append(row)
                        owner = next(
                            row
                            for row in entries_by_template.get(self.delegated_slots[entry.template], ())
                            if (
                                self.policies[entry.template].submission_owner is not None
                                or row.activation.parent == entry.activation.parent
                            )
                        )
                        if owner.status in {"ready", "unstarted", "running"} or owner.activation in pending_activations:
                            continue
                        blocked = True
                policy = self.policies.get(entry.template)
                if not blocked and isinstance(scope, InvocationId):
                    if (
                        policy is None
                        or policy.kind not in {"delegated", "local"}
                        or entry.template in self.delegated_slots
                    ):
                        continue
                if not blocked and policy is not None:
                    if isinstance(scope, DatumId) and _operation_has_omitted_context(
                        self.admitted, scope, entry.template
                    ):
                        self.blocked_reasons[entry.activation] = "omitted_optional"
                        blocked = True
                if blocked:
                    source.state = advance_activation(
                        state=source.state, event=CloseUnstarted(activation=entry.activation, category="blocked")
                    )
                    scheduled = True
                    continue
                if policy is None:
                    if not isinstance(scope, DatumId) or not isinstance(
                        self.admitted.nodes[entry.template], SubgraphNode
                    ):
                        continue
                    try:
                        _materialize_subgraph_inputs(
                            self.admitted,
                            scope,
                            source.state,
                            entry.activation,
                            entry.template,
                            self.subgraph_input_parents,
                            self.facts,
                        )
                    except EffectRejected as exc:
                        if exc.code != EffectCode.MISSING:
                            raise
                        source.state = advance_activation(
                            state=source.state, event=CloseUnstarted(activation=entry.activation, category="blocked")
                        )
                    else:
                        self._start_observed_activation(entry.activation, operation=False)
                        source.state = advance_activation(state=source.state, event=Start(activation=entry.activation))
                    scheduled = True
                    continue

                batchable = isinstance(scope, DatumId) and policy.kind in {"delegated", "external"}
                shared_batch = policy.kind == "delegated" or all(
                    item.capability.attribution == "keyed_shared_request" for item in policy.implementations
                )
                batch_key = entry.template if shared_batch else entry.activation
                group = batches.get(batch_key)
                if policy.kind != "external":
                    if self.services.limits.max_local_in_flight == 0:
                        reject(EffectCode.LIMIT_EXCEEDED)
                    if group is None and local >= self.services.limits.max_local_in_flight:
                        continue
                if policy.kind == "delegated" and isinstance(scope, DatumId) and len(self.frame.sources) > 1:
                    pending = {job.scope for job in group or ()}
                    if any(
                        next((row.status for row in peer.state.entries if row.template == entry.template), None)
                        != "ready"
                        for target, peer in self.frame.sources.items()
                        if target not in pending
                    ):
                        continue
                remote_limit = self.services.limits.max_remote_outstanding
                stalled = (
                    policy.kind == "external"
                    and group is None
                    and remote_limit is not None
                    and (max(len(self.request_authority.state.remote_outstanding), remote) >= remote_limit)
                )
                if stalled and remote:
                    continue
                try:
                    job = self._bind_operation(scope, entry, policy, inputless=isinstance(scope, InvocationId))
                except EffectRejected as exc:
                    if exc.code != EffectCode.MISSING:
                        raise
                    source.state = advance_activation(
                        state=source.state, event=CloseUnstarted(activation=entry.activation, category="blocked")
                    )
                    scheduled = True
                    continue
                self._prepare_job(job)()
                self._start_observed_activation(entry.activation, operation=True)
                if batchable and not stalled:
                    batches.setdefault(batch_key, []).append(cast(_OperationJob[DatumId], job))
                else:
                    span = self._operation_spans[entry.activation]
                    token = (
                        _CURRENT.set(
                            (span.session, span.ordinal)
                            if span.session is not None and span.ordinal is not None
                            else None
                        )
                        if isinstance(scope, DatumId)
                        else None
                    )
                    try:
                        self._launch_batch((job,), stalled=stalled)
                        pending_activations.add(job.activation)
                    finally:
                        if token is not None:
                            _CURRENT.reset(token)
                if group is None:
                    local += policy.kind != "external"
                    remote += policy.kind == "external"
                scheduled = True
                if isinstance(scope, InvocationId):
                    return True
        for group in batches.values():
            self._launch_batch(tuple(group))
        return scheduled

    def _bind_operation(
        self, scope: _Scope, entry: ActivationEntry, policy: OperationExecutionPolicy, *, inputless: bool = False
    ) -> _OperationJob[_Scope]:
        association = SemanticAssociation(task=TaskAttemptId.new(activation=entry.activation))
        inputs, parents = (
            _operation_inputs(
                self.admitted,
                scope,
                entry.template,
                entry.activation,
                self.frame.sources[scope].state,
                association,
                self.subgraph_input_parents,
                self.facts,
            )
            if isinstance(scope, DatumId)
            else (() if inputless else (AssociationInput(association=association, inputs=()),), {})
        )
        return _OperationJob(scope=scope, policy=policy, inputs=inputs, input_parents=parents, association=association)

    def _prepare_job(self, job: _OperationJob[DatumId | InvocationId]) -> Callable[[], None]:
        """Stage activation validation before dispatch; commit only when its owner enters."""
        source = self.frame.occurrence(job.scope)
        state = advance_activation(state=source.state, event=Start(activation=job.activation))

        def commit() -> None:
            source.state = state
            self.operations[job.activation] = job
            if isinstance(job.scope, DatumId):
                self._record_input_ports(cast(_OperationJob[DatumId], job))

        return commit

    def _record_input_ports(self, job: _OperationJob[DatumId]) -> None:
        for input_artifact in job.inputs[0].inputs:
            if input_artifact.artifact is None:
                reject(EffectCode.CONTRADICTORY)
            self.facts.ports[(job.activation, input_artifact.port)] = ExecutionPortFact(
                _key=_FACT_KEY,
                activation=job.activation,
                node=job.node,
                target=job.scope,
                port=input_artifact.port,
                artifact=input_artifact.artifact,
                artifact_type=input_artifact.artifact_type,
                role=_inherited_artifact_role(
                    job.input_parents.get(input_artifact.port),
                    self.facts.provenance,
                    self.facts.ports,
                ),
            )

    def _launch_batch(self, jobs: tuple[_OperationJob[DatumId | InvocationId], ...], *, stalled: bool = False) -> None:
        job = jobs[0]
        policy, implementation = job.policy, job.implementation
        handle = self.handles[
            implementation.implementation, implementation.capability.operation, implementation.configuration
        ]
        if stalled:
            coroutine = _immediate_execution_result(_mapping(policy, "request_limit_exhausted", None, None))
        elif policy.kind == "delegated":
            coroutine = (
                _DelegatedOwner(self, (), invocation_job=cast(_OperationJob[InvocationId], job))
                if isinstance(job.scope, InvocationId)
                else _DelegatedOwner(self, cast(tuple[_OperationJob[DatumId], ...], jobs))
            ).run(handle)
        elif policy.kind == "external":
            coroutine = _run_external_batch(
                self.admitted,
                policy,
                self.handles,
                tuple(value for row in jobs for value in row.inputs),
                self.request_authority,
                self.control,
                self.services.limits,
                self.request_resources,
            )
        else:
            coroutine = _run_local(
                policy,
                handle,
                job.association,
                job.inputs,
                self.control,
                self.admitted.ingress_limits,
                invocation=isinstance(job.scope, InvocationId),
                cancel_waiters=self.control.cancel_waiters,
            )
        self.submissions[asyncio.create_task(coroutine)] = jobs

    async def _process_completed(self) -> None:
        scheduler_wakeup = asyncio.create_task(self.control.scheduler_changed.wait())
        try:
            completed, _ = await asyncio.wait(
                (*self.submissions, scheduler_wakeup), return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            scheduler_wakeup.cancel()
            with suppress(asyncio.CancelledError):
                await scheduler_wakeup
        if scheduler_wakeup in completed:
            self.control.scheduler_changed.clear()
        for task in tuple(self.submissions):
            if task in completed:
                self._complete_submission(task)

    def _complete_submission(self, task: _ExecutionTask) -> None:
        jobs = self.submissions.pop(task)
        returned = task.result()
        if isinstance(returned, (LocalCompleted, SubmissionFailure)):
            (job,) = jobs
            assert isinstance(job.scope, InvocationId)
            frame = None
            if isinstance(returned, SubmissionFailure):
                self.submission_failures.append(returned)
                mapping = _mapping(job.policy, "failure", None, "implementation_exception")
            else:
                frame = _accept_invocation_frame(returned)
                mapping = (
                    _mapping(job.policy, "result", "ok", None)
                    if frame is not None
                    else _mapping(job.policy, "failure", None, "malformed_response")
                )
            self._complete_job(job, mapping, (), ())
            if frame is not None and mapping.category == "success":
                self.frame.publications.append(frame)
            return
        if not isinstance(returned, dict):
            (job,) = jobs
            returned = {job.association: returned}
        for job in jobs:
            mapping, results, assessments = returned[job.association]
            if any(policy.submission_owner == job.node for policy in self.policies.values()):
                entry = self.frame.occurrence(job.scope).entry(job.activation)
                if entry.status not in {"ready", "unstarted", "running"}:
                    if mapping.category != "success":
                        self.submission_failures.append(
                            SubmissionFailure(
                                associations=frozenset({job.association}),
                                cause=RuntimeError("Delegated submission failed after semantic publication"),
                            )
                        )
                    continue
            self._complete_job(job, mapping, results, assessments)

    def _complete_job(
        self,
        job: _OperationJob[DatumId | InvocationId],
        mapping: RuntimeOutcome,
        results: tuple[AssociationResult, ...],
        assessments: tuple[LocalAssessmentResult, ...] = (),
    ) -> RuntimeOutcome:
        """Commit an operation atomically and retain its final publication outcome."""
        if isinstance(job.scope, InvocationId):
            source = self.frame.invocation_source
            assert source is not None
            source.state = advance_activation(
                state=source.state,
                event=ObserveTerminal(activation=job.activation, outcome=mapping.outcome, category=mapping.category),
            )
            span = self._operation_spans[job.activation]
            span.outcome = mapping.category
            span.finish()
            self._close_observed_activations(source.state)
            return mapping
        job = cast(_OperationJob[DatumId], job)
        accepted = mapping.condition == "result"
        mapping = self._publish_operation(job, mapping, results, assessments)
        if (
            job.policy.delegated is not None
            and mapping.condition == "result"
            and mapping.category == "success"
            and any(
                isinstance(key, OperationOutputKey) and key.target == job.scope and key.activation == job.activation
                for key in self.facts.provenance
            )
        ):
            self.blocked_reasons.update(self._unused_tails(job))
            selection = job.policy.delegated.selection
            if selection is not None:
                value = next(
                    output.value for item in results for output in item.outputs if output.port == selection.port
                )
                for slot in selection.omitted(value):
                    self.blocked_reasons[self.frame.sources[job.scope].keys[slot]] = "omitted_optional"
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
        handle = self.handles.get(
            (
                job.implementation.implementation,
                job.implementation.capability.operation,
                job.implementation.configuration,
            )
        )
        handle = self.control.measured_handles.pop(job.association, handle)
        if (
            handle is not None
            and span.session is not None
            and span.session._failure is None
            and handle.measurement is not None
        ):
            try:
                measurement = handle.measurement.terminal_measurement()
                if type(measurement) is WorkMeasurement and any(
                    type(getattr(measurement, name)) is int
                    and getattr(measurement, name) > span.session.limits.max_count
                    for name in ("input_count", "output_count", "artifact_count", "artifact_bytes", "request_count")
                ):
                    span.session._disable("capacity_contract")
                elif (
                    not _valid_measurement(measurement, span.session.limits.max_count)
                    or measurement.route != job.policy.kind
                ):
                    span.session._disable("producer_contract")
                else:
                    span.measurement = measurement
            except BaseException:
                span.reasons = span.reasons | frozenset[ObservationReason]({"internal_failure"})
        span.finish()
        self._close_observed_activations(self.frame.occurrence(job.scope).state)
        if mapping.category == "blocked" and mapping.condition == "artifact_limit_exhausted":
            self.blocked_reasons[job.activation] = mapping.condition
        if mapping.condition == "artifact_limit_exhausted" or accepted and mapping.condition != "result":
            self.publication_failures.append(
                PublicationFailureFact(
                    association=job.association,
                    target=job.scope,
                    condition=mapping.condition,
                    failure=mapping.failure,
                )
            )
        return mapping

    def _operation_result(
        self, scope: DatumId, activation: ActivationKey, *, port: str | None = None
    ) -> AssociationResult | None:
        """Project completed artifacts from the sole bound occurrence record."""
        entry = self.frame.sources[scope].find(activation)
        if entry is None or entry.status in {"ready", "running", "unstarted"}:
            return None
        if entry.status != "success" or entry.outcome is None or activation not in self.operations:
            reject(EffectCode.REQUEST_CAUSALITY)
        job = self.operations[activation]
        operation = job.implementation.capability.operation
        outcome = next(row for row in operation.outcomes if row.name == entry.outcome)
        outputs = []
        for declaration in operation.outputs:
            if port is not None and declaration.name != port:
                continue
            reference = self.facts.artifact(scope, activation, declaration.name)
            if declaration.name in outcome.produced_ports and reference is None:
                reject(EffectCode.MISSING)
            if reference is not None:
                outputs.append(
                    PortArtifact(
                        port=declaration.name,
                        artifact_type=declaration.artifact_type,
                        artifact=reference if port is None else None,
                        value=self.facts.values[reference],
                    )
                )
        return AssociationResult(
            association=job.association,
            outcome=entry.outcome,
            outputs=tuple(outputs),
            consumed_context_ports=frozenset(),
        )

    def _unused_tails(
        self, job: _OperationJob[DatumId], contract: DelegatedRequestContract | None = None
    ) -> dict[ActivationKey, UnusedRetryCapacity]:
        """Close only unentered suffixes of successfully completed generation chains."""
        contract = job.policy.delegated if contract is None else contract
        state = self.request_authority.state
        if not contract or not (contract.optional_tail or contract.generation_groups):
            return {}
        if (
            self.control.cancelled
            or state.cancelled
            or state.reserved
            or state.cancel_requested
            or state.local_in_flight
            or (contract.optional_tail and (state.denials or state.defects or state.remote_outstanding))
        ):
            return {}
        source = self.frame.sources[job.scope]
        entries = {item.activation: item for item in source.state.entries}
        if contract.readiness is not None:
            readiness = self.frame.invocation_source or source
            statuses = {item.activation: item.status for item in readiness.state.entries}
            if any(statuses[readiness.keys[index]] != "success" for index in contract.readiness.slots):
                return {}
        unused = {}
        for group in contract.generations:
            keys = tuple(source.keys[index] for index in group)
            entered = tuple(key for key in keys if key in self.operations)
            if not entered or entered != keys[: len(entered)] or len(entered) == len(keys):
                continue
            last = entered[-1]
            if entries[last].status != "success":
                continue
            history = [
                row
                for row in state.dispatches
                if row.child is not None and row.child.leaf == self.operations[last].association.task
            ]
            if len(history) != 1:
                continue
            request = history[0]
            terminals = [row for row in state.terminals if row.request == request.request]
            settlements = [row for row in state.settlements if row.request == request.request]
            if (
                len(terminals) != 1
                or terminals[0].category != "success"
                or len(settlements) != 1
                or settlements[0].disposition != "completed"
                or settlements[0].remote_stopped is not True
                or (
                    contract.optional_tail
                    and contract.usage_requirement == "exact"
                    and not isinstance(settlements[0].usage, ExactUsage)
                )
            ):
                continue
            cause = UnusedRetryCapacity(owner=job.association.task, after=self.operations[last].association.task)
            unused.update((key, cause) for key in keys[len(entered) :])
        return unused

    def _publish_operation(
        self,
        job: _OperationJob[DatumId],
        mapping: RuntimeOutcome,
        results: tuple[AssociationResult, ...],
        assessments: tuple[LocalAssessmentResult, ...],
    ) -> RuntimeOutcome:
        """Stage values, terminal state and structural expansion before one commit."""
        delta = self.facts.delta()
        staged = False
        membership = None
        if mapping.condition == "result":
            assessed = _assessment_results(self.admitted, job.node, mapping, job.association, assessments)
            try:
                membership = _membership_value(self.admitted, job.node, mapping.outcome, results)
            except EffectRejected:
                assessed = None
            status = (
                "malformed"
                if assessed is None
                else _accept_outputs(self.admitted, job, mapping, results, delta, self.services.limits)
            )
            if status == "valid" and not _capture_assessments(self.admitted, job, mapping, assessed or (), delta):
                status = "malformed"
            staged = status == "valid"
            if not staged:
                mapping = self._publication_failure(job, limit=status == "limit")
                delta = self.facts.delta()
        source = self.frame.sources[job.scope]
        state = advance_activation(
            state=source.state,
            event=ObserveTerminal(activation=job.activation, outcome=mapping.outcome, category=mapping.category),
        )
        if staged and membership is not None:
            try:
                state = _observe_dynamic_membership(
                    self.admitted,
                    state,
                    job.scope,
                    job.activation,
                    job.node,
                    mapping.outcome,
                    membership,
                    delta,
                    self.services.limits,
                )
            except (ContractViolation, EffectRejected) as exc:
                mapping = self._publication_failure(
                    job, limit=isinstance(exc, EffectRejected) and exc.code == EffectCode.LIMIT_EXCEEDED
                )
                state = advance_activation(
                    state=source.state,
                    event=ObserveTerminal(
                        activation=job.activation, outcome=mapping.outcome, category=mapping.category
                    ),
                )
                delta = self.facts.delta()
        try:
            state = _materialize_structure(
                self.admitted, job.scope, state, delta, self.subgraph_input_parents, self.services.limits
            )
        except (ContractViolation, EffectRejected) as exc:
            if not staged:
                raise
            mapping = self._publication_failure(
                job, limit=isinstance(exc, EffectRejected) and exc.code == EffectCode.LIMIT_EXCEEDED
            )
            state = advance_activation(
                state=source.state,
                event=ObserveTerminal(activation=job.activation, outcome=mapping.outcome, category=mapping.category),
            )
        else:
            self.facts.publish(delta)
        source.state = state
        return mapping

    @staticmethod
    def _publication_failure(job: _OperationJob[DatumId], *, limit: bool) -> RuntimeOutcome:
        return _mapping(
            job.policy,
            "artifact_limit_exhausted" if limit else "failure",
            None,
            None if limit else "malformed_response",
        )

    async def _finalize(self) -> ExecutionResult:
        if self.control.cancelled:
            self.request_authority.apply(ScopeCancel())
        cleanup, cleanup_associations = await _cleanup_execution(
            self.admitted,
            self.services.handles,
            self.request_authority.state,
            self.request_resources,
        )
        record = _canonical_record(
            self.admitted,
            self.invocation,
            self.frame.states,
            self.frame.target_keys,
            {key: job.association.task for key, job in self.operations.items()},
            frozenset(self.facts.values),
            frozenset(self.cancelled_unstarted),
            self.blocked_reasons,
            invocation_state=None if self.frame.invocation_source is None else self.frame.invocation_source.state,
        )
        final_outputs = _final_outputs(self.prepared, self.frame.states, self.facts)
        failed_attempts = {
            association.task for failure in self.submission_failures for association in failure.associations
        }
        failed_targets = {
            target
            for target, source in self.frame.sources.items()
            if any(
                entry.activation in self.operations
                and self.operations[entry.activation].association.task in failed_attempts
                for entry in source.state.entries
            )
        }
        final_outputs = tuple(output for output in final_outputs if output.target not in failed_targets)
        return ExecutionResult(
            _key=_RESULT_KEY,
            _execution=self.admitted,
            implementation_resources=self.implementation_resources,
            _retained_resources=tuple(
                {
                    lease.resource: lease
                    for lease in (handle.resource for handle in self.services.handles if handle.resource is not None)
                    if lease.owner == "sdk"
                    and any(fact.resource == lease.resource and fact.disposition == "left_open" for fact in cleanup)
                }.values()
            ),
            _input_parents=tuple(
                (job.scope, job.activation, port, parent)
                for job in self.operations.values()
                if isinstance(job.scope, DatumId)
                for port, parent in job.input_parents.items()
            ),
            _passthrough_parents=tuple(
                (target, activation, port, parent)
                for (target, activation, port), parent in self.facts.passthrough_parents.items()
            ),
            record=record,
            states=self.frame.states,
            invocation_state=None if self.frame.invocation_source is None else self.frame.invocation_source.state,
            requests=request_receipt(self.request_authority.state),
            cleanup=cleanup,
            artifacts=tuple(self.facts.values.items()),
            assessments=tuple(self.facts.assessments.values()),
            ports=tuple(self.facts.ports.values()),
            final_outputs=final_outputs,
            provenance=tuple(self.facts.provenance.values()),
            cleanup_associations=cleanup_associations,
            publication_failures=tuple(self.publication_failures),
            submission_failures=tuple(self.submission_failures),
            frames=tuple(
                bound
                for publication in self.frame.publications
                if (bound := publication.bind(tuple(self.facts.ports.values()))) is not None
            ),
        )
