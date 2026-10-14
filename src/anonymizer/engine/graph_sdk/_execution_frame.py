# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical source occurrences and root artifacts for one invocation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

from anonymizer.engine.constants import COL_TEXT
from anonymizer.engine.graph_sdk._effect_values import EffectCode, reject
from anonymizer.engine.graph_sdk._execution_state import _ExecutionFacts, _ExecutionJob
from anonymizer.engine.graph_sdk._execution_values import (
    _FACT_KEY,
    ArtifactProvenanceFact,
    OperationExecutionPolicy,
    RootInputKey,
)
from anonymizer.engine.graph_sdk.preparation import PreparedPlan
from anonymizer.engine.graph_sdk.requests import TextArtifactValue
from anonymizer.engine.ndd._source_frame import WorkflowSourceFrame
from anonymizer.graph._values import ActivationKey, ArtifactRef, DatumId, InvocationId
from anonymizer.graph.activation import (
    ActivationSeed,
    ActivationState,
    Select,
    advance_activation,
    initialize_activation,
)
from anonymizer.graph.workflow import (
    AdmittedActivationWorkflow,
    AdmittedWorkflow,
    ArtifactType,
    DynamicScope,
    NodeId,
    NodeOutcomeRef,
    OperationNode,
    OutcomeBinding,
    ResourceCeiling,
    SequenceEdge,
    WorkflowOutcomeRef,
    admit_activation_workflow,
    admit_static_workflow,
)

if TYPE_CHECKING:
    import pandas as pd

    from anonymizer.engine.detection._frame_publication import FramePublication, InvocationFrame


@dataclass(slots=True)
class _SourceOccurrence:
    state: ActivationState
    keys: dict[int, ActivationKey]
    opaque_cells: tuple[object, ...] = ()


class _InvocationFrame:
    """Own source-addressed execution state without a parallel ordinal index."""

    def __init__(
        self,
        prepared: PreparedPlan,
        invocation: InvocationId,
        facts: _ExecutionFacts,
        *,
        target_workflow: AdmittedActivationWorkflow | None = None,
        invocation_workflow: AdmittedActivationWorkflow | None = None,
    ) -> None:
        self.prepared = prepared
        self.invocation = invocation
        self.facts = facts
        self.target_workflow = target_workflow or prepared.workflow
        self.invocation_workflow = invocation_workflow
        self.invocation_source: _SourceOccurrence | None = None
        self.sources: dict[DatumId, _SourceOccurrence] = {}
        self.publications: list[FramePublication | InvocationFrame] = []
        self._initialize_activations()
        self._initialize_root_inputs()

    @property
    def states(self) -> tuple[ActivationState, ...]:
        return tuple(source.state for source in self.sources.values())

    @property
    def target_keys(self) -> dict[DatumId, dict[int, ActivationKey]]:
        return {target: source.keys for target, source in self.sources.items()}

    def occurrence(self, target: DatumId | None) -> _SourceOccurrence:
        if target is not None:
            return self.sources[target]
        if self.invocation_source is None:
            reject(EffectCode.MISSING)
        return self.invocation_source

    @property
    def all_states(self) -> tuple[ActivationState, ...]:
        return self.states + (() if self.invocation_source is None else (self.invocation_source.state,))

    def _initialize_activations(self) -> None:
        for target_map in self.prepared.target_occurrences:
            self.sources[target_map.target] = self._initialize(self.target_workflow, target_map.occurrence_offset)
        if self.invocation_workflow is not None:
            self.invocation_source = self._initialize(self.invocation_workflow, self.prepared.invocation_offset)

    def _initialize(self, workflow: AdmittedActivationWorkflow, offset: int) -> _SourceOccurrence:
        templates = {node.id for scope in workflow.scopes for node in scope.workflow.nodes}
        recipe = tuple(slot for slot in self.prepared.reservation_recipe if slot.template in templates)
        keys: dict[int, ActivationKey] = {}
        for slot in recipe:
            keys[slot.index] = ActivationKey(
                invocation=self.invocation,
                occurrence=offset + slot.index,
                parent=keys.get(slot.parent_index),
                iteration=slot.iteration,
            )
        reservations = frozenset(ActivationSeed(template=slot.template, activation=keys[slot.index]) for slot in recipe)
        initialized = initialize_activation(
            workflow=workflow,
            invocation=self.invocation,
            reservations=reservations,
            limits=self.prepared.activation_limits,
        )
        return _SourceOccurrence(
            state=advance_activation(
                state=initialized, event=Select(seeds=_initial_seeds(workflow.workflow, reservations))
            ),
            keys=keys,
        )

    def bind_source_frame(
        self,
        jobs: tuple[_ExecutionJob, ...],
        dataframe: pd.DataFrame,
        *,
        targets: tuple[DatumId, ...] | None = None,
        source_limit: int | None = None,
    ) -> WorkflowSourceFrame:
        """Authenticate the whole submitted frame against this invocation's source jobs."""
        if targets is not None:
            by_target = {job.target: job for job in jobs}
            if (
                type(targets) is not tuple
                or any(type(target) is not DatumId for target in targets)
                or len(targets) != len(jobs)
                or set(targets) != set(by_target)
            ):
                reject(EffectCode.FOREIGN_OWNER)
            jobs = tuple(by_target[target] for target in targets)
        datum_text = {item.id: item.text for item in self.prepared.data.datums}
        selected = len(dataframe) if source_limit is None else source_limit
        if (
            type(selected) is not int
            or not 0 <= selected <= len(dataframe)
            or len(jobs) != selected
            or list(dataframe.columns).count(COL_TEXT) != 1
        ):
            reject(EffectCode.MISSING)
        submitted = dataframe.iloc[:selected]
        if len({job.target for job in jobs}) != len(jobs) or any(
            job.target not in self.sources
            or job.activation not in self.sources[job.target].keys.values()
            or job.association.task.activation != job.activation
            for job in jobs
        ):
            reject(EffectCode.FOREIGN_OWNER)
        if any(
            not isinstance(value, str) or value != datum_text[job.target]
            for value, job in zip(submitted[COL_TEXT].tolist(), jobs, strict=True)
        ):
            reject(EffectCode.CONTRADICTORY)
        for job, cells in zip(jobs, submitted.itertuples(index=False, name=None), strict=True):
            source = self.sources[job.target]
            if not source.opaque_cells:
                source.opaque_cells = cells
        return WorkflowSourceFrame.capture(
            dataframe, associations=tuple(job.association for job in jobs), invocation_bound=True, source_limit=selected
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
            self.facts.add_provenance(
                ArtifactProvenanceFact(
                    _key=_FACT_KEY,
                    key=RootInputKey(target=bound.target, port=bound.port),
                    artifact=reference,
                    parents=frozenset(),
                    decision=False,
                )
            )


def _initial_seeds(workflow: AdmittedWorkflow, reservations: frozenset[ActivationSeed]) -> frozenset[ActivationSeed]:
    blocked = {member for choice in workflow.choices for branch in choice.branches for member in branch.members}
    return frozenset(seed for seed in reservations if seed.activation.parent is None and seed.template not in blocked)


def _invocation_workflows(
    prepared: PreparedPlan, policies: tuple[OperationExecutionPolicy, ...]
) -> tuple[AdmittedActivationWorkflow, AdmittedActivationWorkflow | None]:
    """Place shared readiness in its own admitted activation scope, without a target."""
    shared = {
        prepared.reservation_recipe[index].template
        for policy in policies
        if policy.delegated is not None and policy.delegated.readiness is not None
        for index in policy.delegated.readiness.slots
    }
    if prepared.configuration.invocation_owned:
        owners = [policy for policy in policies if policy.delegated is not None]
        raw = prepared.workflow.workflow
        local = len(policies) == 1 and policies[0].kind == "local" and policies[0].submission_owner is None
        if (
            (not local and (len(owners) != 1 or owners[0].delegated is None or owners[0].delegated.child_slots))
            or (
                local
                and (
                    prepared.configuration.purpose != "execution_only"
                    or len(raw.nodes) != 1
                    or policies[0].result_outcomes != frozenset({"ok"})
                )
            )
            or raw.interface.inputs
            or raw.interface.outputs
            or any(not isinstance(node, OperationNode) for node in raw.nodes)
            or any(node.operation.inputs or node.operation.outputs for node in raw.nodes)
            or any(
                policy.kind != "local"
                for policy in policies
                if policy.node not in shared | {owner.node for owner in owners}
            )
            or len(prepared.workflow.scopes) != 1
            or any(scope.maps or scope.loops or scope.joins for scope in prepared.workflow.scopes)
        ):
            reject(EffectCode.UNSUPPORTED)
        return prepared.workflow, prepared.workflow
    if len(prepared.target_occurrences) <= 1 or not shared:
        return prepared.workflow, None
    raw = prepared.workflow.workflow
    if (
        len(prepared.workflow.scopes) != 1
        or raw.choices
        or any(not isinstance(node, OperationNode) for node in raw.nodes)
    ):
        reject(EffectCode.UNSUPPORTED)
    scope = prepared.workflow.scopes[0]
    if scope.maps or scope.loops or scope.joins:
        reject(EffectCode.UNSUPPORTED)
    if any(node.operation.inputs or node.operation.outputs for node in raw.nodes if node.id in shared):
        reject(EffectCode.UNSUPPORTED)
    if any(edge.before not in shared and edge.after in shared for edge in raw.sequence):
        reject(EffectCode.UNSUPPORTED)

    def project(selected: frozenset[NodeId], invocation: bool) -> AdmittedActivationWorkflow:
        interface = raw.interface
        edges = tuple(edge for edge in raw.sequence if edge.before in selected and edge.after in selected)
        outcomes = tuple(raw.outcome_bindings)
        if invocation:
            descriptors = [
                policy.delegated.readiness
                for policy in policies
                if policy.delegated is not None and policy.delegated.readiness is not None
            ]
            if len(descriptors) != 1:
                reject(EffectCode.UNSUPPORTED)
            descriptor = descriptors[0]
            probe = prepared.reservation_recipe[descriptor.probe_slot].template
            physical = prepared.reservation_recipe[descriptor.request_slot].template
            operation = next(node.operation for node in raw.nodes if node.id == probe)
            interface = replace(
                operation,
                outcomes=tuple(
                    replace(
                        outcome,
                        ceiling=ResourceCeiling(
                            max_activations=len(selected),
                            max_model_requests=sum(
                                max(cast(int, row.ceiling.max_model_requests) for row in node.operation.outcomes)
                                for node in raw.nodes
                                if node.id in selected
                            ),
                            max_input_bytes=None
                            if any(
                                row.ceiling.max_input_bytes is None
                                for node in raw.nodes
                                if node.id in selected
                                for row in node.operation.outcomes
                            )
                            else sum(
                                max(cast(int, row.ceiling.max_input_bytes) for row in node.operation.outcomes)
                                for node in raw.nodes
                                if node.id in selected
                            ),
                            max_output_bytes=None
                            if any(
                                row.ceiling.max_output_bytes is None
                                for node in raw.nodes
                                if node.id in selected
                                for row in node.operation.outcomes
                            )
                            else sum(
                                max(cast(int, row.ceiling.max_output_bytes) for row in node.operation.outcomes)
                                for node in raw.nodes
                                if node.id in selected
                            ),
                        ),
                    )
                    for outcome in operation.outcomes
                ),
            )
            outcomes = tuple(
                OutcomeBinding(
                    source=NodeOutcomeRef(node=probe, outcome=outcome.name),
                    destination=WorkflowOutcomeRef(outcome=outcome.name),
                )
                for outcome in operation.outcomes
            )
            edges = (*edges, SequenceEdge(before=physical, after=probe))
        body = admit_static_workflow(
            workflow=raw.workflow,
            interface=interface,
            nodes=tuple(node for node in raw.nodes if node.id in selected),
            input_bindings=() if invocation else tuple(raw.input_bindings),
            output_bindings=() if invocation else tuple(raw.output_bindings),
            outcome_bindings=outcomes,
            sequence=edges,
            choices=(),
            protection=() if invocation else tuple(raw.protection_requirements),
            limits=raw.limits,
        )
        return admit_activation_workflow(
            workflow=body,
            scopes=(DynamicScope(workflow=body, maps=(), loops=(), joins=()),),
            limits=prepared.workflow.limits,
        )

    return project(frozenset(node.id for node in raw.nodes if node.id not in shared), False), project(
        frozenset(shared), True
    )
