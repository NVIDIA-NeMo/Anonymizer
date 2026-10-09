# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed graph construction and execution for map qualification scenarios."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from anonymizer.engine.graph_sdk.capabilities import ImplementationSelection
from anonymizer.engine.graph_sdk.context import admit_context_plan
from anonymizer.engine.graph_sdk.data import DataGraph, DataLimits
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
    MapExpansionDecl,
    OperationExecutionPolicy,
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
from tests.graph_sdk.qualification_map_callbacks import _ReferenceMapCallback
from tests.graph_sdk.reference.corpora import load_cases
from tests.graph_sdk.test_dynamic_executor import _capability, _outcome, _rows
from tests.graph_sdk.test_local_executor import _Clock
from tests.graph_sdk.test_preparation import _limits


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


CORPUS = load_cases("qualification")

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
