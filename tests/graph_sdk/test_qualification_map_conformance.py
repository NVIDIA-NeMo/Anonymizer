# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute the independent map-evidence topology through public SDK owners."""

from __future__ import annotations

import asyncio
import json
from copy import copy
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectRejected
from anonymizer.engine.graph_sdk.capabilities import ImplementationSelection
from anonymizer.engine.graph_sdk.context import admit_context_plan
from anonymizer.engine.graph_sdk.data import DataGraph, DataLimits
from anonymizer.engine.graph_sdk.evidence import (
    AssessmentSubmission,
    MapItemSubjectRef,
    QualificationLimits,
    admit_qualification,
    evidence_revision_view,
    evidence_validity,
)
from anonymizer.engine.graph_sdk.executor import (
    AdmittedExecutionPlan,
    AssessmentFinding,
    AssessmentLimits,
    DecisionLimits,
    EvidenceProductionDecl,
    ExecutionImplementation,
    ExecutionLimits,
    ExecutionResult,
    ExecutionServices,
    ImplementationHandle,
    LocalAssessmentResult,
    LocalCompleted,
    LocalFailure,
    MapExpansionDecl,
    MapItemKey,
    OperationExecutionPolicy,
    OperationOutputKey,
    RootInputKey,
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.preparation import (
    BoundInput,
    PreparationConfiguration,
    StateRevision,
    StateRevisionView,
    prepare,
)
from anonymizer.engine.graph_sdk.qualification import QualificationResult, qualify
from anonymizer.engine.graph_sdk.records import AbsenceRef, CandidateRef, DecisionRef
from anonymizer.engine.graph_sdk.requests import (
    AssociationInput,
    AssociationResult,
    PortArtifact,
    SemanticAssociation,
    TextArtifactValue,
    TextCollectionItem,
    TextCollectionValue,
)
from anonymizer.graph._values import ActivationKey, ArtifactRef
from anonymizer.graph.activation import ActivationLimits
from anonymizer.graph.workflow import (
    ArtifactType,
    CoverageAtom,
    DynamicLimits,
    DynamicScope,
    EvidencePromise,
    InputBinding,
    InputPort,
    KeyedJoinDecl,
    MapDecl,
    MapItemPort,
    NodeId,
    NodeInputRef,
    NodeOutcomeRef,
    NodeOutputRef,
    OperationNode,
    OperationSpec,
    OutcomeBinding,
    OutputBinding,
    OutputDependency,
    OutputPort,
    ProtectionRequirement,
    SequenceEdge,
    StateEffect,
    SubgraphNode,
    WorkflowId,
    WorkflowInputRef,
    WorkflowLimits,
    WorkflowOutcomeRef,
    WorkflowOutputRef,
    admit_activation_workflow,
    admit_static_workflow,
)
from tests.graph_sdk.test_dynamic_executor import _capability, _outcome, _rows
from tests.graph_sdk.test_local_executor import _Clock
from tests.graph_sdk.test_preparation import _limits


@dataclass
class _ReferenceMapCallback:
    role: str
    count: int
    text_type: ArtifactType
    collection_type: ArtifactType
    calls: int = 0
    member_failure: bool = False
    expander_failure: bool = False
    versioned_items: bool = False
    membership_port: str = "members"
    expansion_outcome: str = "ok"
    item_promise: str = "P_ITEM"

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted | LocalFailure:
        self.calls += 1
        (item,) = request
        assert isinstance(item.association, SemanticAssociation)
        if (
            self.role == "FAILED_SOURCE"
            or (self.role == "MN" and self.member_failure)
            or (self.role == "EXP" and self.expander_failure)
        ):
            return LocalFailure(failure="permanent")
        outputs: tuple[PortArtifact, ...] = ()
        assessments: tuple[LocalAssessmentResult, ...] = ()
        if self.role == "EXP":
            outputs = (
                PortArtifact(
                    port=self.membership_port,
                    artifact_type=self.collection_type,
                    artifact=None,
                    value=TextCollectionValue(
                        items=tuple(
                            TextCollectionItem(
                                key=0 if self.versioned_items else index,
                                version=index + 1 if self.versioned_items else 1,
                                value=TextArtifactValue(text=str(index)),
                            )
                            for index in range(self.count)
                        )
                    ),
                ),
            )
        elif self.role in {"MN", "N"}:
            outputs = (
                PortArtifact(
                    port="evidence", artifact_type=self.text_type, artifact=None, value=TextArtifactValue(text="E")
                ),
            )
            if self.role == "N":
                subject = next(port for port in item.inputs if port.port == "subject")
                outputs += (
                    PortArtifact(port="result", artifact_type=self.text_type, artifact=None, value=subject.value),
                )
            assessments = (
                LocalAssessmentResult(
                    association=item.association,
                    promise="P" if self.role == "N" else self.item_promise,
                    evidence_port="evidence",
                    finding=AssessmentFinding(status="satisfied", code="observed"),
                ),
            )
        return LocalCompleted(
            results=(
                AssociationResult(
                    association=item.association,
                    outcome=self.expansion_outcome if self.role == "EXP" else "ok",
                    outputs=outputs,
                    consumed_context_ports=frozenset(),
                ),
            ),
            assessments=assessments,
        )


async def _execute_reference_map(
    count: int,
    *,
    consumed_only: bool = False,
    member_failure: bool = False,
    expander_failure: bool = False,
    versioned_items: bool = False,
    candidate_uses_membership: bool = True,
    candidate_passthrough: bool = False,
    alternate_expansion: bool = False,
    nested: bool = False,
    two_maps: bool = False,
    member_promise: str = "P_ITEM",
    blocked_member: bool = False,
) -> tuple[AdmittedExecutionPlan, ExecutionResult, dict[str, NodeId]]:
    membership_port = "alternate_members" if alternate_expansion else "members"
    expansion_outcome = "alternate" if alternate_expansion else "ok"
    owner = WorkflowId.new()
    nodes = {name: NodeId.new(workflow=owner) for name in ("N", "EXP", "MN", "J")}
    if nested:
        inner_owner = WorkflowId.new()
        nodes.update({name: NodeId.new(workflow=inner_owner) for name in ("EXP", "MN", "J")})
        nodes["SG"] = NodeId.new(workflow=owner)
    if two_maps:
        nodes.update({name: NodeId.new(workflow=owner) for name in ("EXP2", "MN2", "J2")})
    if blocked_member:
        nodes["FAILED_SOURCE"] = NodeId.new(workflow=owner)
    text_type = ArtifactType(name="text", revision=1)
    collection_type = ArtifactType(name="members", revision=1)
    coverage = frozenset(CoverageAtom(kind="field", name=name) for name in ("K0", "K1"))
    required = frozenset({CoverageAtom(kind="field", name="K0")})
    read = StateEffect(kind="read", name="read")
    root_promise = EvidencePromise(
        name="P", meaning="privacy", subject_port="subject", consumed_ports=frozenset({"context"}), coverage=coverage
    )
    item_promise = EvidencePromise(
        name=member_promise,
        meaning="item_privacy",
        subject_port="subject" if consumed_only else "item",
        consumed_ports=frozenset({"item"}),
        coverage=coverage,
    )
    endpoint = MapItemPort(
        path=(nodes["SG"],) if nested else (),
        expander=nodes["EXP"],
        member=nodes["MN"],
        item_input="item",
        membership_port=membership_port,
        expansion_outcome=expansion_outcome,
    )
    operations = {
        "N": OperationSpec(
            name="N",
            inputs=(
                InputPort(name="subject", artifact_type=text_type),
                InputPort(name="context", artifact_type=text_type),
                *((InputPort(name="membership", artifact_type=collection_type),) if candidate_uses_membership else ()),
            ),
            outputs=tuple(OutputPort(name=name, artifact_type=text_type) for name in ("result", "evidence")),
            output_dependencies=(
                OutputDependency(
                    output="result",
                    inputs=frozenset({"subject", "membership"})
                    if candidate_uses_membership
                    else frozenset({"subject"}),
                    identity_input="subject",
                ),
                OutputDependency(output="evidence", inputs=frozenset({"context"}), identity_input=None),
            ),
            outcomes=(
                replace(
                    _outcome("ok", produced=frozenset({"result", "evidence"}), max_output_bytes=32),
                    evidence=frozenset({root_promise}),
                    state_effects=frozenset({read}),
                ),
            ),
        ),
        "EXP": OperationSpec(
            name="EXP",
            inputs=(InputPort(name="context", artifact_type=text_type),),
            outputs=(OutputPort(name=membership_port, artifact_type=collection_type),),
            output_dependencies=(
                OutputDependency(output=membership_port, inputs=frozenset({"context"}), identity_input=None),
            ),
            outcomes=(
                _outcome(
                    expansion_outcome, produced=frozenset({membership_port}), max_activations=4, max_output_bytes=32
                ),
            ),
        ),
        "MN": OperationSpec(
            name="MN",
            inputs=(
                InputPort(name="item", artifact_type=text_type),
                *((InputPort(name="subject", artifact_type=text_type),) if consumed_only else ()),
            ),
            outputs=(OutputPort(name="evidence", artifact_type=text_type),),
            output_dependencies=(OutputDependency(output="evidence", inputs=frozenset({"item"}), identity_input=None),),
            outcomes=(
                replace(
                    _outcome("ok", produced=frozenset({"evidence"}), max_output_bytes=32),
                    evidence=frozenset({item_promise}),
                ),
            ),
        ),
        "J": OperationSpec(name="J", inputs=(), outputs=(), output_dependencies=(), outcomes=(_outcome("ok"),)),
    }
    if blocked_member:
        operations["FAILED_SOURCE"] = OperationSpec(
            name="FAILED_SOURCE",
            inputs=(),
            outputs=(OutputPort(name="ready", artifact_type=text_type),),
            output_dependencies=(OutputDependency(output="ready", inputs=frozenset(), identity_input=None),),
            outcomes=(_outcome("ok", produced=frozenset({"ready"})),),
        )
        operations["MN"] = replace(
            operations["MN"],
            inputs=operations["MN"].inputs + (InputPort(name="prerequisite", artifact_type=text_type),),
        )
    if two_maps:
        second_promise = replace(
            item_promise,
            name="P_ITEM_2",
            meaning="item_privacy_2",
            subject_port="item2",
            consumed_ports=frozenset({"item2"}),
        )
        second_endpoint = MapItemPort(
            path=(),
            expander=nodes["EXP2"],
            member=nodes["MN2"],
            item_input="item2",
            membership_port="members2",
            expansion_outcome="ok",
        )
        operations["N"] = replace(
            operations["N"],
            inputs=operations["N"].inputs + (InputPort(name="membership2", artifact_type=collection_type),),
            output_dependencies=tuple(
                replace(dep, inputs=dep.inputs | {"membership2"}) if dep.output == "result" else dep
                for dep in operations["N"].output_dependencies
            ),
        )
        operations["EXP2"] = replace(
            operations["EXP"],
            name="EXP2",
            outputs=(OutputPort(name="members2", artifact_type=collection_type),),
            output_dependencies=(
                OutputDependency(output="members2", inputs=frozenset({"context"}), identity_input=None),
            ),
            outcomes=tuple(
                replace(outcome, produced_ports=frozenset({"members2"})) for outcome in operations["EXP"].outcomes
            ),
        )
        operations["MN2"] = replace(
            operations["MN"],
            name="MN2",
            inputs=(InputPort(name="item2", artifact_type=text_type),),
            output_dependencies=(
                OutputDependency(output="evidence", inputs=frozenset({"item2"}), identity_input=None),
            ),
            outcomes=tuple(
                replace(outcome, evidence=frozenset({second_promise})) for outcome in operations["MN"].outcomes
            ),
        )
        operations["J2"] = replace(operations["J"], name="J2")
    interface = OperationSpec(
        name="root",
        inputs=tuple(InputPort(name=name, artifact_type=text_type) for name in ("subject", "context")),
        outputs=(OutputPort(name="result", artifact_type=text_type),),
        output_dependencies=(
            OutputDependency(
                output="result",
                inputs=frozenset({"subject", "context"})
                if candidate_uses_membership and not candidate_passthrough
                else frozenset({"subject"}),
                identity_input="subject",
            ),
        ),
        outcomes=(
            replace(
                _outcome(
                    "ok",
                    produced=frozenset({"result"}),
                    max_activations=13 if two_maps else 8 if blocked_member else 7,
                    max_output_bytes=256 if two_maps else 128,
                ),
                evidence=frozenset(
                    {
                        root_promise,
                        replace(
                            item_promise,
                            subject_port="subject" if consumed_only else endpoint,
                            consumed_ports=frozenset({endpoint}),
                        ),
                    }
                ),
                state_effects=frozenset({read}),
            ),
        ),
    )
    if two_maps:
        interface = replace(
            interface,
            outcomes=tuple(
                replace(
                    outcome,
                    evidence=outcome.evidence
                    | {
                        replace(
                            second_promise, subject_port=second_endpoint, consumed_ports=frozenset({second_endpoint})
                        )
                    },
                )
                for outcome in interface.outcomes
            ),
        )
    static_options: dict[str, Any] = dict(
        workflow=owner,
        interface=interface,
        nodes=tuple(OperationNode(id=nodes[name], operation=operation) for name, operation in operations.items()),
        input_bindings=(
            *(
                InputBinding(source=WorkflowInputRef(port=name), destination=NodeInputRef(node=nodes["N"], port=name))
                for name in ("subject", "context")
            ),
            InputBinding(
                source=WorkflowInputRef(port="context"), destination=NodeInputRef(node=nodes["EXP"], port="context")
            ),
            InputBinding(
                source=WorkflowInputRef(port="subject"), destination=NodeInputRef(node=nodes["MN"], port="item")
            ),
            *(
                (
                    InputBinding(
                        source=NodeOutputRef(node=nodes["EXP"], port=membership_port),
                        destination=NodeInputRef(node=nodes["N"], port="membership"),
                    ),
                )
                if candidate_uses_membership
                else ()
            ),
            *(
                (
                    InputBinding(
                        source=WorkflowInputRef(port="subject"),
                        destination=NodeInputRef(node=nodes["MN"], port="subject"),
                    ),
                )
                if consumed_only
                else ()
            ),
        ),
        output_bindings=(
            OutputBinding(
                source=WorkflowInputRef(port="subject")
                if candidate_passthrough
                else NodeOutputRef(node=nodes["N"], port="result"),
                destination=WorkflowOutputRef(port="result"),
            ),
        ),
        outcome_bindings=tuple(
            OutcomeBinding(
                source=NodeOutcomeRef(node=nodes[name], outcome="ok"), destination=WorkflowOutcomeRef(outcome="ok")
            )
            for name in ("J", "N")
        ),
        sequence=(
            SequenceEdge(before=nodes["EXP"], after=nodes["MN"]),
            SequenceEdge(before=nodes["MN"], after=nodes["J"]),
            SequenceEdge(before=nodes["EXP"], after=nodes["N"]),
            SequenceEdge(before=nodes["N"], after=nodes["J"]),
        ),
        choices=(),
        protection=(
            ProtectionRequirement(
                outcome="ok",
                meaning="privacy",
                subject_port="subject",
                consumed_ports=frozenset({"context"}),
                coverage=required,
            ),
            ProtectionRequirement(
                outcome="ok",
                meaning="item_privacy",
                subject_port="subject" if consumed_only else endpoint,
                consumed_ports=frozenset({endpoint}),
                coverage=required,
                candidate_port="result",
            ),
        ),
        limits=WorkflowLimits(
            max_nodes=4,
            max_bindings=9 + int(consumed_only),
            max_sequence_edges=4,
            max_choices=0,
            max_branch_members=0,
            max_subgraph_depth=1,
            max_choice_states=1,
        ),
    )
    if blocked_member:
        prerequisite = InputBinding(
            source=NodeOutputRef(node=nodes["FAILED_SOURCE"], port="ready"),
            destination=NodeInputRef(node=nodes["MN"], port="prerequisite"),
        )
        edge = SequenceEdge(before=nodes["FAILED_SOURCE"], after=nodes["MN"])
        static_options["input_bindings"] += (prerequisite,)
        static_options["sequence"] += (edge,)
        static_options["limits"] = replace(static_options["limits"], max_nodes=5, max_bindings=10, max_sequence_edges=5)
    if two_maps:
        static_options["input_bindings"] += (
            InputBinding(
                source=WorkflowInputRef(port="context"), destination=NodeInputRef(node=nodes["EXP2"], port="context")
            ),
            InputBinding(
                source=WorkflowInputRef(port="subject"), destination=NodeInputRef(node=nodes["MN2"], port="item2")
            ),
            InputBinding(
                source=NodeOutputRef(node=nodes["EXP2"], port="members2"),
                destination=NodeInputRef(node=nodes["N"], port="membership2"),
            ),
        )
        static_options["sequence"] += (
            SequenceEdge(before=nodes["EXP2"], after=nodes["MN2"]),
            SequenceEdge(before=nodes["MN2"], after=nodes["J2"]),
            SequenceEdge(before=nodes["EXP2"], after=nodes["N"]),
            SequenceEdge(before=nodes["J2"], after=nodes["J"]),
        )
        static_options["protection"] += (
            ProtectionRequirement(
                outcome="ok",
                meaning="item_privacy_2",
                subject_port=second_endpoint,
                consumed_ports=frozenset({second_endpoint}),
                coverage=required,
                candidate_port="result",
            ),
        )
        static_options["limits"] = replace(static_options["limits"], max_nodes=7, max_bindings=13, max_sequence_edges=8)
    if nested:
        inner_endpoint = replace(endpoint, path=())
        inner_interface = OperationSpec(
            name="SG",
            inputs=(InputPort(name="context", artifact_type=text_type),),
            outputs=(OutputPort(name="nested_members", artifact_type=collection_type),),
            output_dependencies=(
                OutputDependency(output="nested_members", inputs=frozenset({"context"}), identity_input=None),
            ),
            outcomes=(
                replace(
                    _outcome("ok", produced=frozenset({"nested_members"}), max_activations=6, max_output_bytes=96),
                    evidence=frozenset(
                        {replace(item_promise, subject_port=inner_endpoint, consumed_ports=frozenset({inner_endpoint}))}
                    ),
                ),
            ),
        )
        inner = admit_static_workflow(
            workflow=inner_owner,
            interface=inner_interface,
            nodes=tuple(OperationNode(id=nodes[name], operation=operations[name]) for name in ("EXP", "MN", "J")),
            input_bindings=(
                InputBinding(
                    source=WorkflowInputRef(port="context"), destination=NodeInputRef(node=nodes["EXP"], port="context")
                ),
                InputBinding(
                    source=WorkflowInputRef(port="context"), destination=NodeInputRef(node=nodes["MN"], port="item")
                ),
            ),
            output_bindings=(
                OutputBinding(
                    source=NodeOutputRef(node=nodes["EXP"], port=membership_port),
                    destination=WorkflowOutputRef(port="nested_members"),
                ),
            ),
            outcome_bindings=(
                OutcomeBinding(
                    source=NodeOutcomeRef(node=nodes["J"], outcome="ok"), destination=WorkflowOutcomeRef(outcome="ok")
                ),
            ),
            sequence=(
                SequenceEdge(before=nodes["EXP"], after=nodes["MN"]),
                SequenceEdge(before=nodes["MN"], after=nodes["J"]),
            ),
            choices=(),
            protection=(),
            limits=WorkflowLimits(
                max_nodes=3,
                max_bindings=5,
                max_sequence_edges=2,
                max_choices=0,
                max_branch_members=0,
                max_subgraph_depth=1,
                max_choice_states=1,
            ),
        )
        static_options.update(
            nodes=(
                OperationNode(id=nodes["N"], operation=operations["N"]),
                SubgraphNode(id=nodes["SG"], operation=inner_interface, body=inner),
            ),
            input_bindings=(
                *(
                    InputBinding(
                        source=WorkflowInputRef(port=name), destination=NodeInputRef(node=nodes["N"], port=name)
                    )
                    for name in ("subject", "context")
                ),
                InputBinding(
                    source=WorkflowInputRef(port="context"), destination=NodeInputRef(node=nodes["SG"], port="context")
                ),
                InputBinding(
                    source=NodeOutputRef(node=nodes["SG"], port="nested_members"),
                    destination=NodeInputRef(node=nodes["N"], port="membership"),
                ),
            ),
            outcome_bindings=(
                OutcomeBinding(
                    source=NodeOutcomeRef(node=nodes["N"], outcome="ok"), destination=WorkflowOutcomeRef(outcome="ok")
                ),
            ),
            sequence=(SequenceEdge(before=nodes["SG"], after=nodes["N"]),),
            limits=WorkflowLimits(
                max_nodes=5,
                max_bindings=12,
                max_sequence_edges=3,
                max_choices=0,
                max_branch_members=0,
                max_subgraph_depth=2,
                max_choice_states=1,
            ),
        )
    static = admit_static_workflow(**static_options)
    if blocked_member:
        assert prerequisite in static.input_bindings
        assert edge in static.sequence
    workflow = admit_activation_workflow(
        workflow=static,
        scopes=(
            *((DynamicScope(workflow=static, maps=(), joins=(), loops=()),) if nested else ()),
            DynamicScope(
                workflow=inner if nested else static,
                maps=(
                    MapDecl(
                        expander=nodes["EXP"],
                        member=nodes["MN"],
                        expansion_outcomes=frozenset({expansion_outcome}),
                        max_children=2,
                        item_input="item",
                    ),
                    *(
                        (
                            MapDecl(
                                expander=nodes["EXP2"],
                                member=nodes["MN2"],
                                expansion_outcomes=frozenset({"ok"}),
                                max_children=2,
                                item_input="item2",
                            ),
                        )
                        if two_maps
                        else ()
                    ),
                ),
                joins=(
                    KeyedJoinDecl(
                        source=nodes["EXP"],
                        join=nodes["J"],
                        accepted_categories=frozenset({"success"}),
                        reduction="all_by_key",
                    ),
                    *(
                        (
                            KeyedJoinDecl(
                                source=nodes["EXP2"],
                                join=nodes["J2"],
                                accepted_categories=frozenset({"success"}),
                                reduction="all_by_key",
                            ),
                        )
                        if two_maps
                        else ()
                    ),
                ),
                loops=(),
            ),
        ),
        limits=DynamicLimits(
            max_maps=2 if two_maps else 1,
            max_joins=2 if two_maps else 1,
            max_loops=0,
            max_children_per_map=2,
            max_iterations_per_loop=0,
            max_dynamic_depth=2 if nested else 1,
            max_activation_occurrences=9 if two_maps else 6 if nested or blocked_member else 5,
        ),
    )
    graph = DataGraph.new()
    graph, target = graph.add_text("A")
    graph, context = graph.add_text("X")
    data = graph.validate(
        targets=(target,),
        source_relations=(),
        contexts=(),
        dependencies=(),
        coherence=(),
        atomic=(),
        output_regions=(),
        limits=DataLimits(max_datums=2, max_targets=1, max_text_bytes=2, max_declarations=0, max_group_members=1),
    )
    capabilities = tuple(_capability(operation, index) for index, operation in enumerate(operations.values()))
    prepared = prepare(
        data=data,
        workflow=workflow,
        activation_limits=ActivationLimits(
            max_events=64,
            max_entries=9 if two_maps else 6 if nested or blocked_member else 5,
            max_parent_depth=3 if nested else 2,
        ),
        bound_inputs=tuple(
            BoundInput(target=target, source=source, port=port, artifact_type=text_type)
            for port, source in (("subject", target), ("context", context))
        ),
        configuration=PreparationConfiguration(
            purpose="protection", required_protection_outcomes=frozenset({"ok"}), hard_request_limit=None
        ),
        state=StateRevisionView(revisions=frozenset({StateRevision(effect=read, revision=1)})),
        selections=tuple(
            ImplementationSelection(
                node=node, implementation=capability.implementation, configuration=capability.configuration
            )
            for node, capability in zip((nodes[name] for name in operations), capabilities, strict=True)
        ),
        capabilities=capabilities,
        limits=_limits(capabilities=len(operations), slots=9 if two_maps else 6 if nested or blocked_member else 5),
    )
    admitted = admit_execution_plan(
        context=admit_context_plan(
            prepared=prepared, bound_context=None, adaptive_retrievals=(), context_capabilities=()
        ),
        capabilities=capabilities,
        policies=tuple(
            OperationExecutionPolicy(
                node=node,
                kind="local",
                request=None,
                safe_detachment="forbidden",
                implementations=(
                    ExecutionImplementation(
                        implementation=capability.implementation,
                        configuration=capability.configuration,
                        capability=capability,
                        request=None,
                    ),
                ),
                result_outcomes=frozenset(outcome.name for outcome in capability.operation.outcomes),
                runtime_outcomes=_rows(frozenset(outcome.name for outcome in capability.operation.outcomes)),
            )
            for node, capability in zip((nodes[name] for name in operations), capabilities, strict=True)
        ),
        decisions=(),
        assessment_productions=tuple(
            EvidenceProductionDecl(
                node=nodes[name],
                outcome="ok",
                promise=promise,
                evidence_port="evidence",
                absence_queries=frozenset({0}) if name == "N" else frozenset(),
                supported_findings=frozenset({AssessmentFinding(status="satisfied", code="observed")}),
            )
            for name, promise in (("N", "P"), ("MN", member_promise), *((("MN2", "P_ITEM_2"),) if two_maps else ()))
        ),
        assessment_limits=AssessmentLimits(
            max_productions=3 if two_maps else 2,
            max_findings_per_production=1,
            max_finding_code_bytes=16,
            max_absence_queries=1,
            max_assessment_facts=5 if two_maps else 3,
            max_port_facts=24 if two_maps else 16,
            max_provenance_edges=24 if two_maps else 16,
        ),
        map_expansions=(
            MapExpansionDecl(
                expander=nodes["EXP"], outcome=expansion_outcome, membership_port=membership_port, item_type=text_type
            ),
            *(
                (
                    MapExpansionDecl(
                        expander=nodes["EXP2"], outcome="ok", membership_port="members2", item_type=text_type
                    ),
                )
                if two_maps
                else ()
            ),
        ),
    )
    callbacks = tuple(
        _ReferenceMapCallback(
            name.removesuffix("2"),
            count,
            text_type,
            collection_type,
            member_failure=member_failure,
            expander_failure=expander_failure,
            versioned_items=versioned_items,
            membership_port="members2" if name == "EXP2" else membership_port,
            expansion_outcome=expansion_outcome,
            item_promise="P_ITEM_2" if name == "MN2" else member_promise,
        )
        for name in operations
    )
    running = await start_execution(
        admitted=admitted,
        capabilities=capabilities,
        services=ExecutionServices(
            handles=tuple(
                ImplementationHandle(
                    implementation=capability.implementation,
                    operation=capability.operation,
                    configuration=capability.configuration,
                    local=callback,
                    transport=None,
                    resource=None,
                )
                for capability, callback in zip(capabilities, callbacks, strict=True)
            ),
            context_resources=(),
            limits=ExecutionLimits(
                max_local_in_flight=1,
                max_remote_outstanding=0,
                max_runtime_artifacts=32,
                max_runtime_artifact_bytes=64,
                max_collection_items=2,
            ),
            decision_limits=DecisionLimits(max_pending=0, max_lifetime_ns=0),
            clock=_Clock(),
            absence_revisions=((0, 1),),
        ),
    )
    result = await running.wait()
    assert [callback.calls for callback in callbacks] == (
        [1, 1, 0, 0, 1]
        if blocked_member
        else [0, 1, 0, 0]
        if expander_failure
        else [1, 1, count, int(not member_failure)] + ([1, count, 1] if two_maps else [])
    )
    return admitted, result, nodes


CORPUS = json.loads((Path(__file__).parent / "reference/qualification_v1_cases.json").read_text())
FLAT_CASE_IDS = {
    *(case["case_id"] for case in CORPUS if case["case_id"].startswith("assessment/dynamic_")),
    *(f"map_item_evidence/direct_{count}" for count in (0, 1, 2)),
    *(
        f"map_item_evidence/{suffix}"
        for suffix in (
            "typed_consumed_endpoint",
            "nested_path",
            "two_independent_maps",
            "two_maps_no_crossproduct",
            "member_non_success",
            "distinct_outcome_port",
            "item_stale",
            "different_final_ancestry",
            "candidate_passthrough_unrelated",
            "missing_submission",
            "foreign_submission",
            "repeated_submission",
            "missing_fact",
            "copied_member_submission",
            "item_unknown",
        )
    ),
    "map_item_bounds/submissions_one_over",
    "map_item_bounds/verified_one_over",
}


@pytest.mark.parametrize(
    "case", [case for case in CORPUS if case["case_id"] in FLAT_CASE_IDS], ids=lambda case: case["case_id"]
)
def test_reference_map_case_through_real_execution(case: dict[str, Any]) -> None:
    try:
        actual = asyncio.run(_run_reference_map_case(case))
    except EffectRejected as exc:
        actual = {"status": "rejected", "code": exc.code.value}
    expected = json.loads(json.dumps(case["expected"]))
    if "verified" in expected:
        expected["verified"].sort(key=lambda row: (row["target"], row["activation"], row["evidence_artifact"]))
    if "record" in expected:
        for membership in expected["record"]["memberships"].values():
            membership["members"].sort()
    assert actual == expected


async def _run_reference_map_case(case: dict[str, Any]) -> dict[str, Any]:
    events = case["events"]
    injected = case["case_id"] in {
        "assessment/dynamic_unreached_failed_expansion_injected",
        "assessment/dynamic_blocked_unreached_injected",
        "assessment/dynamic_started_failure_injected",
    }
    injected_fact = (
        next((event for event in events if event["kind"] == "assessment" and event["node"] == "MN"), None)
        if injected
        else None
    )
    if injected:
        # Authenticate the real failure record before corrupting its retained inventory.
        events = [event for event in events if event is not injected_fact]
        case = {**case, "events": events, "case_id": case["case_id"].removesuffix("_injected")}

    dynamic = case["case_id"].startswith("assessment/dynamic_")
    member_promise = "P_MEMBER" if dynamic else "P_ITEM"
    count = sum(event["kind"] == "entry" and event["node"] == "MN" for event in events)
    consumed_only = case["case_id"] == "map_item_evidence/typed_consumed_endpoint"
    member_failure = case["case_id"] in {"map_item_evidence/member_non_success", "assessment/dynamic_started_failure"}
    expander_failure = case["case_id"] == "assessment/dynamic_unreached_failed_expansion"
    blocked_member = case["case_id"] == "assessment/dynamic_blocked_unreached"
    versioned_items = case["case_id"] == "map_item_evidence/item_stale"
    candidate_uses_membership = case["case_id"] != "map_item_evidence/different_final_ancestry"
    candidate_passthrough = case["case_id"] == "map_item_evidence/candidate_passthrough_unrelated"
    alternate_expansion = case["case_id"] == "map_item_evidence/distinct_outcome_port"
    nested = case["case_id"] == "map_item_evidence/nested_path"
    two_maps = case["case_id"] in {
        "map_item_evidence/two_independent_maps",
        "map_item_evidence/two_maps_no_crossproduct",
    }
    baseline: dict[str, Any] = (
        case
        if consumed_only
        or member_failure
        or expander_failure
        or blocked_member
        or versioned_items
        or not candidate_uses_membership
        or candidate_passthrough
        or alternate_expansion
        or nested
        else next(item for item in CORPUS if item["case_id"] == f"map_item_evidence/direct_{count}")
    )
    if two_maps:
        baseline = next(item for item in CORPUS if item["case_id"] == "map_item_evidence/two_independent_maps")
    if dynamic:
        # The dynamic family names the same member promise differently.
        baseline = json.loads(json.dumps(baseline).replace("P_ITEM", member_promise))
    missing_terminal = case["case_id"] in {
        "assessment/dynamic_missing_terminal_unsubmitted",
        "assessment/dynamic_missing_terminal_submitted",
    }
    if missing_terminal:
        baseline = {
            **baseline,
            "events": [
                event
                for event in baseline["events"]
                if not (event["kind"] == "terminal" and event["activation"] == "M0")
            ],
        }
    mutable_events = {"assessment_submission", "assessment", "revision"}
    actual_immutable = [event for event in events if event["kind"] not in mutable_events]
    baseline_immutable = [event for event in baseline["events"] if event["kind"] not in mutable_events]
    if dynamic:
        # These retained owner facts are unordered; occurrence identities remain exact.
        actual_immutable.sort(key=lambda event: json.dumps(event, sort_keys=True))
        baseline_immutable.sort(key=lambda event: json.dumps(event, sort_keys=True))
    assert actual_immutable == baseline_immutable
    declaration_baseline = baseline["declaration"]
    if dynamic:
        # Failure events differ, but their declaration still uses the independent
        # ordinary map contract, plus the explicit failed prerequisite owner.
        declaration_baseline = json.loads(
            json.dumps(
                next(item["declaration"] for item in CORPUS if item["case_id"] == "map_item_evidence/direct_0")
            ).replace("P_ITEM", member_promise)
        )
        if blocked_member:
            declaration_baseline["node_kinds"]["FAILED_SOURCE"] = "operation"
    assert {key: value for key, value in case["declaration"].items() if key != "limits"} == {
        key: value for key, value in declaration_baseline.items() if key != "limits"
    }
    execution, result, nodes = await _execute_reference_map(
        count,
        consumed_only=consumed_only,
        member_failure=member_failure,
        expander_failure=expander_failure,
        versioned_items=versioned_items,
        candidate_uses_membership=candidate_uses_membership,
        candidate_passthrough=candidate_passthrough,
        alternate_expansion=alternate_expansion,
        nested=nested,
        two_maps=two_maps,
        member_promise=member_promise,
        blocked_member=blocked_member,
    )
    assert (
        len(result.states[0].entries)
        == len(result.record.terminals)
        == count + 3 + int(nested) + int(blocked_member) + (count + 2 if two_maps else 0)
    )
    if missing_terminal:
        (member_activation,) = (entry.activation for entry in result.states[0].entries if entry.template == nodes["MN"])
        member_terminals = [
            terminal for terminal in result.record.terminals if terminal.activation == member_activation
        ]
        assert len(member_terminals) == 1
        object.__setattr__(
            result.record,
            "terminals",
            tuple(terminal for terminal in result.record.terminals if terminal != member_terminals[0]),
        )
    names = _MapRecordNames(execution, result, nodes)
    names.assert_retained_facts(events)
    names.assert_assessments(baseline["events"])
    facts = {
        ("F:A:P" if fact.node == nodes["N"] else f"F:{names.activation(fact.activation)}:{fact.promise}"): fact
        for fact in result.assessments
    }
    if case["case_id"] == "assessment/dynamic_occurrence_duplicate":
        # Fact labels are reference aliases; the retained owner tuple is duplicated.
        events = [
            {**event, "fact": event["fact"].removesuffix(":DUPLICATE")} if event["kind"] == "assessment" else event
            for event in events
        ]
    retained = {event["fact"] for event in events if event["kind"] == "assessment"}
    assert retained <= set(facts)
    if retained != set(facts):
        assert case["case_id"] in {"map_item_evidence/missing_fact", "assessment/dynamic_occurrence_missing"}
        object.__setattr__(result, "assessments", tuple(fact for name, fact in facts.items() if name in retained))
    if case["case_id"] == "assessment/dynamic_occurrence_duplicate":
        retained_order = [event["fact"] for event in events if event["kind"] == "assessment"]
        assert len(retained_order) == len(facts) + 1
        object.__setattr__(result, "assessments", tuple(facts[name] for name in retained_order))
    names.assert_assessments(events)
    if any(event["kind"] == "assessment_submission" and event["fact"] not in facts for event in events):
        assert case["case_id"] in {"map_item_evidence/foreign_submission", "assessment/dynamic_submission_foreign"}
        _, foreign, foreign_nodes = await _execute_reference_map(1, member_promise=member_promise)
        facts[f"F:FOREIGN:{member_promise}"] = next(
            fact for fact in foreign.assessments if fact.node == foreign_nodes["MN"]
        )
    admitted = admit_qualification(
        execution=execution,
        productions=execution.assessment_productions,
        limits=QualificationLimits(
            **{field.name: case["declaration"]["limits"][field.name] for field in fields(QualificationLimits)}
        ),
    )
    revisions = [event for event in events if event["kind"] == "revision"]
    artifacts = {name: ref for ref, name in names.artifacts.items()}
    configurations = {item.node: item.capability.configuration for item in execution.context.prepared.implementations}
    current = evidence_revision_view(
        admitted=admitted,
        result=result,
        artifacts=tuple(
            artifacts[f"{event['key']}v{event['value']}"] for event in revisions if event["collection"] == "artifacts"
        ),
        absences=tuple(
            AbsenceRef(invocation=result.record.invocation, query=int(event["key"][1:]), scope_revision=event["value"])
            for event in revisions
            if event["collection"] == "absences"
        ),
        configurations=tuple(
            (nodes[event["key"]], configurations[nodes[event["key"]]])
            for event in revisions
            if event["collection"] == "configurations"
        ),
        state=StateRevisionView.from_revisions(
            revisions=tuple(
                StateRevision(effect=StateEffect(kind="read", name=event["key"]), revision=event["value"])
                for event in revisions
                if event["collection"] == "state"
            )
        ),
    )
    submissions = tuple(
        AssessmentSubmission(fact=facts[event["fact"]]) for event in events if event["kind"] == "assessment_submission"
    )
    if injected_fact is not None:
        _, successful, successful_nodes = await _execute_reference_map(1, member_promise=member_promise)
        original = next(fact for fact in successful.assessments if fact.node == successful_nodes["MN"])
        corrupted = copy(original)
        member = min(
            (
                reservation.activation
                for state in result.states
                for reservation in state.reservations
                if reservation.template == nodes["MN"]
            ),
            key=lambda activation: activation.occurrence,
        )
        object.__setattr__(corrupted, "activation", member)
        object.__setattr__(corrupted, "node", nodes["MN"])
        assert (
            injected_fact["activation"],
            injected_fact["node"],
            injected_fact["outcome"],
            injected_fact["promise"],
        ) == ("M0", "MN", corrupted.outcome, corrupted.promise)
        assert corrupted.finding.status == injected_fact["finding"]
        assert corrupted.activation.invocation == result.record.invocation
        assert not any(fact.activation == member for fact in result.assessments)
        object.__setattr__(result, "assessments", (*result.assessments, corrupted))
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    return names.normalize(output)


class _MapRecordNames:
    """Name every retained occurrence without dropping joins or producer facts."""

    def __init__(self, execution: AdmittedExecutionPlan, result: ExecutionResult, nodes: dict[str, NodeId]) -> None:
        self.execution = execution
        self.result = result
        self.nodes = {node: name for name, node in nodes.items()}
        self.entries = {entry.activation: entry for state in result.states for entry in state.entries}
        members = sorted(
            (entry for entry in self.entries.values() if entry.template == nodes["MN"]),
            key=lambda entry: entry.activation.occurrence,
        )
        self.activations = {entry.activation: f"M{index}" for index, entry in enumerate(members)}
        for entry in self.entries.values():
            if entry.template not in {nodes["MN"], nodes.get("MN2")}:
                self.activations[entry.activation] = {
                    "N": "ROOT:A",
                    "EXP": "MAP",
                    "J": "JOIN",
                    "SG": "WRAP",
                    "EXP2": "MAP2",
                    "J2": "JOIN2",
                    "FAILED_SOURCE": "FAILED",
                }[self.nodes[entry.template]]
        if "MN2" in nodes:
            second_members = sorted(
                (entry for entry in self.entries.values() if entry.template == nodes["MN2"]),
                key=lambda entry: entry.activation.occurrence,
            )
            self.activations.update({entry.activation: f"Z{index}" for index, entry in enumerate(second_members)})
        self.artifacts: dict[ArtifactRef, str] = {
            fact.artifact: {"subject": "Av0", "context": "XAv0"}[fact.key.port]
            for fact in result.provenance
            if isinstance(fact.key, RootInputKey)
        }
        for fact in result.provenance:
            if isinstance(fact.key, MapItemKey):
                prefix = "MJ" if self.entries[fact.key.member].template == nodes.get("MN2") else "MI"
                self.artifacts[fact.artifact] = f"{prefix}{fact.key.item_key}v{fact.key.item_version}"
        for port in result.ports:
            node = self.nodes[port.node]
            if node == "N":
                name = {
                    "subject": "Av0",
                    "result": "Av0",
                    "context": "XAv0",
                    "evidence": "EAv0",
                    "membership": "CAv0",
                    "membership2": "CBv0",
                }[port.port]
            elif node == "SG":
                name = "XAv0" if port.port == "context" else "CAv0"
            elif node in {"EXP", "EXP2"}:
                name = "XAv0" if port.port == "context" else "CBv0" if node == "EXP2" else "CAv0"
            else:
                assert node in {"MN", "MN2"}
                index = self.activations[port.activation][1:]
                if port.port in {"item", "item2"}:
                    key = next(
                        fact.key
                        for fact in result.provenance
                        if isinstance(fact.key, MapItemKey) and fact.key.member == port.activation
                    )
                    name = f"{'MJ' if node == 'MN2' else 'MI'}{key.item_key}v{key.item_version}"
                else:
                    name = "Av0" if port.port == "subject" else f"{'MF' if node == 'MN2' else 'ME'}{index}v0"
            if port.artifact in self.artifacts:
                assert self.artifacts[port.artifact] == name
            self.artifacts[port.artifact] = name
        assert len(set(self.artifacts.values())) == len(self.artifacts)
        assert set(self.artifacts) == {ref for ref, _ in result.artifacts} == set(result.record.artifacts)
        assert all(
            ref.version == (int(name.rsplit("v", 1)[1]) if name.startswith(("MI", "MJ")) else 1)
            for ref, name in self.artifacts.items()
        )
        self.producers = {
            fact.key: f"{'MAPITEM2' if self.activations[fact.key.member].startswith('Z') else 'MAPITEM'}:{self.activations[fact.key.member][1:]}"
            for fact in result.provenance
            if isinstance(fact.key, MapItemKey)
        }
        assert result.record.graph == execution.context.prepared.data.graph
        assert result.record.plan == execution.context.prepared.plan
        assert result.record.invocation.plan == result.record.plan
        assert result._execution is execution

    def assert_retained_facts(self, events: list[dict[str, Any]]) -> None:
        expected_ports = {
            (event["activation"], event["node"], event["port"], event["artifact"], event["role"])
            for event in events
            if event["kind"] == "port"
        }
        actual_ports = {
            (
                self.activation(port.activation),
                self.nodes[port.node],
                port.port,
                self.artifact(port.artifact),
                port.role,
            )
            for port in self.result.ports
        }
        assert len(actual_ports) == len(self.result.ports)
        assert actual_ports == expected_ports
        producers: dict[object, str] = {key: value for key, value in self.producers.items()}
        for fact in self.result.provenance:
            key = fact.key
            if isinstance(key, RootInputKey):
                producers[key] = f"ROOT:A:{key.port}"
            elif isinstance(key, OperationOutputKey):
                activation = self.activation(key.activation)
                if activation in {"MAP", "MAP2"}:
                    producers[key] = f"OP:{activation}:A:{key.port}"
                elif activation == "WRAP":
                    producers[key] = f"SGOUT:WRAP:A:{key.port}"
                elif activation == "ROOT:A":
                    producers[key] = "OUT:A" if key.port == "result" else "EVID:A"
                else:
                    assert key.port == "evidence"
                    producers[key] = f"EVID:{activation}"
            else:
                assert isinstance(key, MapItemKey)
        assert len(producers) == len(self.result.provenance)
        actual_inputs = {
            (self.activation(activation), port, producers[producer])
            for _, activation, port, producer in self.result._input_parents
        }
        expected_inputs = {
            (event["activation"], event["port"], event["producer"])
            for event in events
            if event["kind"] == "input_producer"
        }
        assert len(actual_inputs) == len(self.result._input_parents)
        assert actual_inputs == expected_inputs
        actual_provenance = {
            producers[fact.key]: {
                "artifact": self.artifact(fact.artifact),
                "parents": sorted(producers[parent] for parent in fact.parents),
                "decision": fact.decision,
            }
            for fact in self.result.provenance
        }
        expected_provenance = {
            event["key"]: {
                "artifact": event["artifact"],
                "parents": sorted(event["parents"]),
                "decision": event["decision"],
            }
            for event in events
            if event["kind"] == "provenance"
        }
        assert actual_provenance == expected_provenance
        for fact in self.result.provenance:
            if isinstance(fact.key, MapItemKey):
                event = next(
                    event for event in events if event["kind"] == "provenance" and event["key"] == producers[fact.key]
                )
                assert (
                    fact.key.item_key,
                    fact.key.item_version,
                    self.activation(fact.key.expander),
                    self.activation(fact.key.member),
                    fact.key.port,
                ) == (event["item_key"], event["item_version"], event["expander"], event["member"], event["port"])

    def assert_assessments(self, events: list[dict[str, Any]]) -> None:
        expected = [event for event in events if event["kind"] == "assessment"]
        actual: list[dict[str, Any]] = []
        target = self.execution.context.prepared.target_occurrences[0].target
        for fact in self.result.assessments:
            node = self.nodes[fact.node]
            activation = self.activation(fact.activation)
            selected = next(item for item in self.execution.context.prepared.implementations if item.node == fact.node)
            outcome = next(
                outcome for outcome in selected.capability.operation.outcomes if outcome.name == fact.outcome
            )
            promise = next(promise for promise in outcome.evidence if promise.name == fact.promise)
            production = next(
                production
                for production in self.execution.assessment_productions
                if production.node == fact.node and production.promise == fact.promise
            )
            assert fact.environment.configuration == selected.capability.configuration
            assert fact.finding in production.supported_findings
            ports = {port.port: port for port in self.result.ports if port.activation == fact.activation}
            assert all(port.target == target for port in ports.values())
            assert isinstance(promise.subject_port, str)
            assert all(isinstance(port, str) for port in promise.consumed_ports)
            assert all(atom.kind == "field" for atom in promise.coverage)
            actual.append(
                {
                    "kind": "assessment",
                    "fact": "F:A:P" if node == "N" else f"F:{activation}:{fact.promise}",
                    "activation": activation,
                    "authenticated_factory": "EXEC:I0",
                    "node": node,
                    "target": "A",
                    "outcome": fact.outcome,
                    "promise": fact.promise,
                    "subject_port": promise.subject_port,
                    "subject_artifact": self.artifact(ports[promise.subject_port].artifact),
                    "evidence_port": production.evidence_port,
                    "evidence_artifact": self.artifact(fact.evidence_artifact),
                    "consumed": {
                        port: self.artifact(ports[port].artifact)
                        for port in promise.consumed_ports
                        if isinstance(port, str)
                    },
                    "coverage": sorted(atom.name for atom in promise.coverage),
                    "finding": fact.finding.status,
                    "environment": {
                        "absences": {f"Q{ref.query}": ref.scope_revision for ref in fact.environment.absences},
                        "configurations": {node: "c0"},
                        "state": {
                            revision.effect.name: revision.revision for revision in fact.environment.state.revisions
                        },
                    },
                }
            )
        assert sorted(actual, key=lambda item: item["fact"]) == sorted(expected, key=lambda item: item["fact"])

    def artifact(self, ref: ArtifactRef) -> str:
        assert ref.invocation == self.result.record.invocation
        return self.artifacts[ref]

    def activation(self, key: ActivationKey) -> str:
        return self.activations[key]

    def normalize(self, output: QualificationResult) -> dict[str, Any]:
        assert output._result is self.result
        # Actual reservation ordinals are local to each admitted plan. Check
        # its public tuple before renaming identities for the neutral model.
        order = [(item.activation.occurrence, item.reference.artifact.key) for item in output.verified]
        assert order == sorted(order)
        (target,) = output.targets
        (status,) = output.record.statuses
        assert target.target == status.target == self.execution.context.prepared.target_occurrences[0].target
        target_row = {
            "target": "A",
            "candidate": None if target.candidate is None else self.artifact(target.candidate.artifact),
            "verified": sorted(self.artifact(ref.artifact) for ref in target.verified),
            "required_decisions": sorted(self.artifact(ref.artifact) for ref in target.required_decisions),
            "withholding": sorted(target.withholding),
            "completion": status.completion,
            "qualification": status.qualification,
            "artifact_available": status.artifact_available,
            "protection_available": status.protection_available,
        }
        verified = []
        for item in output.verified:
            assert item._result is self.result
            node = self.nodes[item.node]
            fact = next(fact for fact in self.result.assessments if fact.activation == item.activation)
            assert item.environment.configuration == fact.environment.configuration
            assert item.promise.name == ("P" if node == "N" else "P_ITEM_2" if node == "MN2" else fact.promise)
            assert item.promise.meaning == (
                "privacy" if node == "N" else "item_privacy_2" if node == "MN2" else "item_privacy"
            )
            consumed = {}
            roles = {}
            for port, ref in item.consumed_by_port:
                assert not isinstance(ref, AbsenceRef)
                consumed[port] = self.artifact(ref.artifact if isinstance(ref, (CandidateRef, DecisionRef)) else ref)
                roles[port] = (
                    "candidate"
                    if isinstance(ref, CandidateRef)
                    else "decision"
                    if isinstance(ref, DecisionRef)
                    else "artifact"
                )
            row = {
                "activation": self.activation(item.activation),
                "authenticated_factory": "EXEC:I0",
                "node": node,
                "target": "A",
                "outcome": item.outcome,
                "promise": item.promise.name,
                "meaning": item.promise.meaning,
                "subject_port": item.promise.subject_port,
                "subject_artifact": self.artifact(item.subject.artifact),
                "evidence_port": next(
                    production.evidence_port
                    for production in self.execution.assessment_productions
                    if production.node == item.node and production.promise == item.promise.name
                ),
                "evidence_artifact": self.artifact(item.reference.artifact),
                "consumed": consumed,
                "consumed_roles": roles,
                "coverage": sorted(atom.name for atom in item.coverage),
                "finding": item.finding.status,
                "environment": {
                    "absences": {f"Q{ref.query}": ref.scope_revision for ref in item.environment.absences},
                    "configurations": {node: "c0"},
                    "state": {revision.effect.name: revision.revision for revision in item.environment.state.revisions},
                },
                "validity": evidence_validity(evidence=item, current=output._current),
            }
            if isinstance(item.subject, MapItemSubjectRef):
                row["subject"] = {
                    "artifact": self.artifact(item.subject.artifact),
                    "producer": self.producers[item.subject.producer],
                }
                assert item.subject.producer.target == target.target
            else:
                assert item.subject.target == target.target
            verified.append(row)
        memberships = {}
        for membership in output.record.memberships:
            name = "__ROOT__" if membership.parent is None else self.activation(membership.parent)
            expansion = next(
                (
                    expansion
                    for state in self.result.states
                    for expansion in state.expansions
                    if expansion.parent == membership.parent
                ),
                None,
            )
            memberships[name] = {
                "parent": None if membership.parent is None else name,
                "target": None if membership.parent is None else "A",
                "members": sorted(self.activation(key) for key in membership.members),
                "closed": membership.closed,
                "status": expansion.status if expansion is not None else "closed" if membership.closed else "open",
                "expansion_outcome": None if expansion is None else self.entries[expansion.parent].outcome,
            }
        terminals = {}
        for terminal in output.record.terminals:
            name = self.activation(terminal.activation)
            if terminal.attempt is not None:
                assert terminal.attempt.activation == terminal.activation
            terminals[name] = {
                "attempt": None if terminal.attempt is None else f"TASK:{name}",
                "category": terminal.category,
                "outcome": self.entries[terminal.activation].outcome,
                "reasons": sorted(terminal.reasons),
                "structural": terminal.structural,
                "target": "A",
            }
        return {
            "status": "accepted",
            "targets": [target_row],
            "qualified": [
                {
                    "target": "A",
                    "candidate": self.artifact(item.candidate.artifact),
                    "evidence": sorted(self.artifact(ref.artifact) for ref in item.evidence),
                    "required_decisions": sorted(self.artifact(ref.artifact) for ref in item.required_decisions),
                }
                for item in output.qualified
            ],
            "verified": sorted(verified, key=lambda row: (row["target"], row["activation"], row["evidence_artifact"])),
            "required_decisions": sorted(self.artifact(ref.artifact) for ref in output.required_decisions),
            "record": {
                "execution": {"factory": "EXEC:I0", "graph": "G0", "invocation": "I0", "plan": "P0"},
                "artifacts": sorted(self.artifact(ref) for ref in output.record.artifacts),
                "evidence": sorted(self.artifact(ref.artifact) for ref in output.record.evidence),
                "memberships": memberships,
                "terminals": terminals,
                "targets": [target_row],
            },
        }
