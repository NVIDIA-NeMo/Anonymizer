# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private source-bound prototype of the shared Rewrite evaluation and repair tail.

The public workflow remains on its accepted controller until bounded repair,
failed-source closure and final evidence selection replace that controller.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pandas as pd

from anonymizer.engine.constants import COL_REWRITTEN_TEXT, COL_REWRITTEN_TEXT_INITIAL, COL_TEXT
from anonymizer.engine.detection._frame_publication import FramePublication
from anonymizer.engine.detection._opaque_redact import _execute_frame_plan, _frame_implementation, _prepare_frame_plan
from anonymizer.engine.graph_sdk._effect_values import EffectCode, reject
from anonymizer.engine.graph_sdk._execution_values import DelegatedRequests
from anonymizer.engine.graph_sdk.blocking import BlockingPool
from anonymizer.engine.graph_sdk.capabilities import (
    FrozenConfig,
)
from anonymizer.engine.graph_sdk.data import DataGraph, DataLimits
from anonymizer.engine.graph_sdk.executor import (
    AssessmentLimits,
    ExecutionLimits,
    ExecutionResult,
    IngressLimits,
    LocalCompleted,
)
from anonymizer.engine.graph_sdk.preparation import (
    BoundInput,
)
from anonymizer.engine.graph_sdk.requests import AssociationInput, AssociationResult
from anonymizer.engine.ndd.adapter import NddAdapter
from anonymizer.engine.rewrite._semantic_evaluation import RewriteEvaluationBinding
from anonymizer.engine.rewrite.artifacts import (
    REWRITE_TYPES,
    EvaluationMetricsValue,
    PrivacyAnswersValue,
    QualityAnswersValue,
    QualityComparisonValue,
    RewriteQAValue,
)
from anonymizer.engine.rewrite.combined_rewrite_workflow import CombinedRewriteWorkflow, _evaluation_state
from anonymizer.engine.rewrite.rewrite_workflow import RewriteWorkflow
from anonymizer.engine.row_partitioning import resolve_source_positions
from anonymizer.graph.workflow import (
    ArtifactType,
    ChoiceBranch,
    ChoiceDecl,
    DynamicScope,
    InputBinding,
    InputPort,
    KeyedJoinDecl,
    LoopCarriedBinding,
    LoopDecl,
    LoopInitialBinding,
    NodeId,
    NodeInputRef,
    NodeOutcomeRef,
    NodeOutputRef,
    OperationNode,
    OperationSpec,
    OutcomeBinding,
    OutcomeSpec,
    OutputBinding,
    OutputDependency,
    OutputPort,
    ResourceCeiling,
    SequenceEdge,
    SubgraphNode,
    WorkflowId,
    WorkflowInputRef,
    WorkflowLimits,
    WorkflowOutcomeRef,
    WorkflowOutputRef,
    admit_static_workflow,
)


async def execute_initial_evaluation(
    source: pd.DataFrame,
    *,
    adapter: NddAdapter,
    combined: bool,
    arguments: dict[str, Any],
    configuration: FrozenConfig,
    pool: BlockingPool,
    limits: ExecutionLimits,
) -> ExecutionResult:
    """Execute actual DD prefix/Evaluate under one admitted source-bound owner.

    This incomplete private prototype requires caller-supplied limits/configuration.
    It publishes explicit loop exits but does not yet produce assessment evidence
    or replace the two schedule controllers. It is not a public strategy selector.
    """
    if source.empty or arguments.get("preview_num_records") is not None:
        reject(EffectCode.UNSUPPORTED)
    graph = DataGraph.new()
    targets = []
    for text in source[COL_TEXT]:
        graph, target = graph.add_text(text)
        targets.append(target)
    size = len(targets)
    allowance = arguments["evaluation"].max_repair_iterations
    data = graph.validate(
        targets=tuple(targets),
        source_relations=(),
        contexts=(),
        dependencies=(),
        coherence=(),
        atomic=(),
        output_regions=(),
        limits=DataLimits(
            max_datums=size,
            max_targets=size,
            max_text_bytes=sum(len(t.encode("utf-8")) for t in source[COL_TEXT]),
            max_declarations=0,
            max_group_members=0,
        ),
    )
    raw, scopes, capabilities, policies, ids, body_ids = _tail(configuration, allowance)
    text = ArtifactType(name="anonymizer.text", revision=1)
    prepared = _prepare_frame_plan(
        data,
        raw,
        capabilities,
        tuple(BoundInput(target=t, source=t, port="source", artifact_type=text) for t in targets),
        scopes=scopes,
        max_occurrences=8 + 6 * allowance,
    )
    slots = {
        name: next(slot.index for slot in prepared.reservation_recipe if slot.template == node)
        for name, node in ids.items()
        if name != "member"
    }

    def occurrence(slot):
        while slot.iteration is None and slot.parent_index is not None:
            slot = prepared.reservation_recipe[slot.parent_index]
        return slot.iteration

    iteration_slots = {
        iteration: {
            name: next(
                slot.index
                for slot in prepared.reservation_recipe
                if slot.template == node and occurrence(slot) == iteration
            )
            for name, node in body_ids.items()
        }
        for iteration in range(allowance)
    }
    loop = asyncio.get_running_loop()

    def callback(inputs: tuple[AssociationInput, ...], requests: DelegatedRequests) -> LocalCompleted:
        binding = RewriteEvaluationBinding(
            requests,
            requests.bind_frame(source, targets=tuple(targets)),
            slots,
            loop,
            _evaluation_state(0 if combined else None, COL_REWRITTEN_TEXT_INITIAL if combined else COL_REWRITTEN_TEXT),
            iteration_slots=iteration_slots,
            combined=combined,
        )
        asyncio.run_coroutine_threadsafe(binding.passthrough(source), loop).result()
        runner = (CombinedRewriteWorkflow if combined else RewriteWorkflow)(adapter, semantic_binding=binding)
        result = runner.run(source, **arguments)
        selected = asyncio.run_coroutine_threadsafe(binding.finish(), loop).result()
        if result.source_positions is None:
            reject(EffectCode.MISSING)
        positions = resolve_source_positions(source, result.dataframe, result.source_positions)
        successful = tuple(binding.source.associations[index] for index in positions)
        if set(successful) != set(selected):
            reject(EffectCode.CONTRADICTORY)
        publication = FramePublication.capture(
            result.dataframe,
            associations=tuple(selected[association] for association in successful),
            text_ports=((COL_REWRITTEN_TEXT, "candidate"),),
            failed_records=tuple(result.failed_records),
            opaque_source=source,
        )
        return LocalCompleted(
            results=tuple(
                AssociationResult(
                    association=item.association, outcome="ok", outputs=(), consumed_context_ports=frozenset()
                )
                for item in inputs
            ),
            frame=publication,
        )

    return await _execute_frame_plan(
        prepared=prepared,
        capabilities=tuple(capabilities),
        policies=tuple(policies),
        callback=callback,
        pool=pool,
        borrowed_loop=loop,
        limits=limits,
        ingress_limits=IngressLimits(
            max_associations=size,
            max_outputs_per_association=6,
            max_collection_items=limits.max_collection_items * size,
            max_assessments=1,
            max_metadata_items=size * (20 + 10 * limits.max_collection_items),
            max_text_bytes=limits.max_runtime_artifact_bytes,
            max_metadata_bytes=limits.max_runtime_artifact_bytes,
            max_depth=8,
        ),
        assessment_limits=AssessmentLimits(
            max_productions=0,
            max_findings_per_production=0,
            max_finding_code_bytes=0,
            max_absence_queries=0,
            max_assessment_facts=0,
            max_port_facts=(8 + 6 * allowance) * 14 * size,
            max_provenance_edges=(8 + 6 * allowance) * 42 * size,
        ),
    )


def _tail(configuration: FrozenConfig, allowance: int):
    """Declare one initial evaluator and one carried repair/evaluation body."""
    text = ArtifactType(name="anonymizer.text", revision=1)
    qa, answers, privacy, comparison, metrics = (
        REWRITE_TYPES[t]
        for t in (
            RewriteQAValue,
            QualityAnswersValue,
            PrivacyAnswersValue,
            QualityComparisonValue,
            EvaluationMetricsValue,
        )
    )
    carried = {"source": text, "baseline": text, "qa": qa, "candidate": text, "privacy": privacy, "metrics": metrics}
    root, body = WorkflowId.new(), WorkflowId.new()
    ids = {
        name: NodeId.new(workflow=root)
        for name in ("prefix", "quality", "privacy", "comparison", "metrics", "member", "join", "initial", "final")
    }
    body_ids = {name: NodeId.new(workflow=body) for name in ("repair", "quality", "privacy", "comparison", "metrics")}
    capabilities, policies = [], []

    def operation(name, inputs, outputs, *, identity=(), control=False, local=False, ceiling=1):
        return OperationSpec(
            name=name,
            inputs=tuple(InputPort(name="in_" + k if k in outputs else k, artifact_type=v) for k, v in inputs.items()),
            outputs=tuple(OutputPort(name=k, artifact_type=v) for k, v in outputs.items()),
            output_dependencies=tuple(
                OutputDependency(
                    output=k,
                    inputs=frozenset({"in_" + k})
                    if k in identity
                    else frozenset("in_" + i if i in outputs else i for i in inputs),
                    identity_input="in_" + k if k in identity else None,
                )
                for k in outputs
            ),
            outcomes=tuple(
                OutcomeSpec(
                    name=n,
                    category="success",
                    produced_ports=frozenset(outputs),
                    context=frozenset(),
                    evidence=frozenset(),
                    state_effects=frozenset(),
                    model_requirements=frozenset(),
                    ceiling=ResourceCeiling(
                        max_activations=ceiling,
                        max_model_requests=0 if local else None,
                        max_input_bytes=None,
                        max_output_bytes=None,
                    ),
                )
                for n in (
                    ("again", "stop", "exhausted")
                    if control
                    else ("ok", "skipped")
                    if name.endswith(("prefix", "quality", "privacy", "comparison"))
                    else ("ok",)
                )
            ),
        )

    def node(owner_ids, name, inputs, outputs, **kwargs):
        op = operation("rewrite-" + ("iteration-" if owner_ids is body_ids else "") + name, inputs, outputs, **kwargs)
        cap, policy = _frame_implementation(
            owner_ids[name],
            op,
            configuration,
            submission_owner=None if name == "prefix" else ids["prefix"],
            local=kwargs.get("local", False),
        )
        capabilities.append(cap)
        policies.append(policy)
        return OperationNode(id=owner_ids[name], operation=op)

    def evaluation(owner_ids, first, first_inputs, first_outputs):
        nodes = [
            node(owner_ids, first, first_inputs, first_outputs),
            node(owner_ids, "quality", {"candidate": text, "qa": qa}, {"answers": answers}),
            node(owner_ids, "privacy", {"candidate": text, "qa": qa}, {"privacy": privacy}),
            node(owner_ids, "comparison", {"qa": qa, "answers": answers}, {"comparison": comparison}),
            node(
                owner_ids,
                "metrics",
                {**{k: v for k, v in carried.items() if k != "metrics"}, "comparison": comparison},
                carried,
                identity=tuple(k for k in carried if k != "metrics"),
                control=True,
                local=True,
            ),
        ]
        bindings, edges = [], set()
        for current in nodes:
            name = next(k for k, v in owner_ids.items() if v == current.id)
            for port in current.operation.inputs:
                key = port.name.removeprefix("in_")
                if name == first:
                    origin = WorkflowInputRef(port="in_" + key if first == "repair" else key)
                elif key in {"source", "metrics"}:
                    origin = WorkflowInputRef(port="in_" + key if first == "repair" else key)
                elif key in {"qa", "baseline"} and first == "repair":
                    origin = WorkflowInputRef(port="in_" + key if first == "repair" else key)
                else:
                    predecessor = (
                        "quality"
                        if key == "answers"
                        else "privacy"
                        if key == "privacy"
                        else "comparison"
                        if key == "comparison"
                        else first
                    )
                    origin = NodeOutputRef(node=owner_ids[predecessor], port=key)
                    edges.add((owner_ids[predecessor], current.id))
                bindings.append(InputBinding(source=origin, destination=NodeInputRef(node=current.id, port=port.name)))
        return nodes, bindings, edges

    # Initial metrics have no prior metrics input; each repair body consumes all carried values.
    root_nodes, root_bindings, root_edges = evaluation(
        ids, "prefix", {"source": text}, {"candidate": text, "baseline": text, "qa": qa}
    )
    body_nodes, body_bindings, body_edges = evaluation(body_ids, "repair", carried, {"candidate": text})
    body_interface = operation(
        "rewrite-repair-evaluate", carried, carried, identity=("source", "baseline", "qa"), control=True, ceiling=5
    )

    def admit(owner, interface, nodes, bindings, edges, outputs, outcomes, choices=()):
        return admit_static_workflow(
            workflow=owner,
            interface=interface,
            nodes=tuple(nodes),
            input_bindings=tuple(bindings),
            output_bindings=tuple(outputs),
            outcome_bindings=tuple(outcomes),
            sequence=tuple(SequenceEdge(before=a, after=b) for a, b in sorted(edges, key=lambda e: repr(e))),
            choices=choices,
            protection=(),
            limits=WorkflowLimits(
                max_nodes=20,
                max_bindings=100,
                max_sequence_edges=30,
                max_choices=1,
                max_branch_members=4,
                max_subgraph_depth=2,
                max_choice_states=3,
            ),
        )

    body_raw = admit(
        body,
        body_interface,
        body_nodes,
        body_bindings,
        body_edges,
        [
            OutputBinding(source=NodeOutputRef(node=body_ids["metrics"], port=k), destination=WorkflowOutputRef(port=k))
            for k in carried
        ],
        [
            OutcomeBinding(
                source=NodeOutcomeRef(node=body_ids["metrics"], outcome=n), destination=WorkflowOutcomeRef(outcome=n)
            )
            for n in ("again", "stop", "exhausted")
        ],
    )
    root_nodes.append(SubgraphNode(id=ids["member"], operation=body_interface, body=body_raw))
    root_nodes.extend(
        [
            node(ids, "join", {"metrics": metrics}, {}, control=True, local=True),
            node(ids, "initial", {"candidate": text}, {"candidate": text}, identity=("candidate",), local=True),
            node(ids, "final", {"candidate": text}, {"candidate": text}, identity=("candidate",), local=True),
        ]
    )
    root_bindings.extend(
        InputBinding(
            source=NodeOutputRef(node=ids["metrics"], port=k),
            destination=NodeInputRef(node=ids["member"], port="in_" + k),
        )
        for k in carried
    )
    root_bindings.extend(
        InputBinding(
            source=NodeOutputRef(node=ids[src], port="candidate"),
            destination=NodeInputRef(node=ids[dst], port="in_candidate"),
        )
        for src, dst in (("metrics", "initial"), ("member", "final"))
    )
    root_bindings.append(
        InputBinding(
            source=NodeOutputRef(node=ids["metrics"], port="metrics"),
            destination=NodeInputRef(node=ids["join"], port="metrics"),
        )
    )
    root_edges.update(
        (ids[a], ids[b]) for a, b in (("metrics", "member"), ("member", "join"), ("join", "final"), ("join", "initial"))
    )
    interface = operation(
        "rewrite-tail",
        {"source": text},
        {"initial_candidate": text, "repaired_candidate": text},
        ceiling=max(14, 8 + 6 * allowance),
    )
    interface = replace(
        interface,
        outcomes=tuple(
            replace(interface.outcomes[0], name=n, produced_ports=frozenset({p}))
            for n, p in (("initial", "initial_candidate"), ("final", "repaired_candidate"))
        ),
    )
    raw = admit(
        root,
        interface,
        root_nodes,
        root_bindings,
        root_edges,
        [
            OutputBinding(
                source=NodeOutputRef(node=ids[n], port="candidate"),
                destination=WorkflowOutputRef(port="initial_candidate" if n == "initial" else "repaired_candidate"),
            )
            for n in ("initial", "final")
        ],
        [
            OutcomeBinding(source=NodeOutcomeRef(node=ids[n], outcome="ok"), destination=WorkflowOutcomeRef(outcome=n))
            for n in ("initial", "final")
        ],
        (
            ChoiceDecl(
                selector=ids["join"],
                branches=(
                    ChoiceBranch(outcomes=frozenset({"stop", "exhausted"}), members=frozenset({ids["initial"]})),
                    ChoiceBranch(outcomes=frozenset({"again"}), members=frozenset({ids["final"]})),
                ),
            ),
        ),
    )
    loop = LoopDecl(
        starter=ids["metrics"],
        member=ids["member"],
        join=ids["join"],
        enter_outcomes=frozenset({"again"}),
        bypass_outcomes=frozenset({"stop", "exhausted"}),
        continue_outcomes=frozenset({"again"}),
        exit_outcomes=frozenset({"stop", "exhausted"}),
        initial=tuple(
            LoopInitialBinding(
                source=NodeOutputRef(node=ids["metrics"], port=k),
                destination=NodeInputRef(node=ids["member"], port="in_" + k),
            )
            for k in carried
        ),
        carried=tuple(
            LoopCarriedBinding(
                source=NodeOutputRef(node=ids["member"], port=k),
                destination=NodeInputRef(node=ids["member"], port="in_" + k),
            )
            for k in carried
        ),
        max_iterations=allowance,
    )
    scopes = (
        DynamicScope(
            workflow=raw,
            maps=(),
            joins=(
                KeyedJoinDecl(
                    source=ids["metrics"],
                    join=ids["join"],
                    accepted_categories=frozenset({"success"}),
                    reduction="all_by_key",
                ),
            ),
            loops=(loop,),
        ),
        DynamicScope(workflow=body_raw, maps=(), joins=(), loops=()),
    )
    return raw, scopes, tuple(capabilities), tuple(policies), ids, body_ids
