# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical source occurrences and root artifacts for one invocation."""

from __future__ import annotations

from dataclasses import dataclass

from anonymizer.engine.graph_sdk._execution_state import _ExecutionFacts
from anonymizer.engine.graph_sdk._execution_values import _FACT_KEY, ArtifactProvenanceFact, RootInputKey
from anonymizer.engine.graph_sdk.preparation import PreparedPlan
from anonymizer.engine.graph_sdk.requests import TextArtifactValue
from anonymizer.graph._values import ActivationKey, ArtifactRef, DatumId, InvocationId
from anonymizer.graph.activation import (
    ActivationSeed,
    ActivationState,
    Select,
    advance_activation,
    initialize_activation,
)
from anonymizer.graph.workflow import AdmittedWorkflow, ArtifactType


@dataclass(slots=True)
class _SourceOccurrence:
    state: ActivationState
    keys: dict[int, ActivationKey]


class _InvocationFrame:
    """Own source-addressed execution state without a parallel ordinal index."""

    def __init__(self, prepared: PreparedPlan, invocation: InvocationId, facts: _ExecutionFacts) -> None:
        self.prepared = prepared
        self.invocation = invocation
        self.facts = facts
        self.sources: dict[DatumId, _SourceOccurrence] = {}
        self.root_inputs: dict[tuple[DatumId, str], ArtifactRef] = {}
        self._initialize_activations()
        self._initialize_root_inputs()

    @property
    def states(self) -> tuple[ActivationState, ...]:
        return tuple(source.state for source in self.sources.values())

    @property
    def target_keys(self) -> dict[DatumId, dict[int, ActivationKey]]:
        return {target: source.keys for target, source in self.sources.items()}

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
            self.sources[target_map.target] = _SourceOccurrence(
                state=advance_activation(
                    state=initialized, event=Select(seeds=_initial_seeds(self.prepared.workflow.workflow, reservations))
                ),
                keys=keys,
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


def _initial_seeds(workflow: AdmittedWorkflow, reservations: frozenset[ActivationSeed]) -> frozenset[ActivationSeed]:
    blocked = {member for choice in workflow.choices for branch in choice.branches for member in branch.members}
    return frozenset(seed for seed in reservations if seed.activation.parent is None and seed.template not in blocked)
