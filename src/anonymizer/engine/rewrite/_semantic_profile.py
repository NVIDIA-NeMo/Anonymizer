# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private source-bound initial-evaluation checkpoint of the shared Rewrite tail.

The public workflow remains on its accepted controller until bounded repair,
failed-source closure and final evidence selection replace that controller.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pandas as pd

from anonymizer.engine.constants import COL_REWRITTEN_TEXT, COL_REWRITTEN_TEXT_INITIAL, COL_TEXT
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
from anonymizer.graph.workflow import (
    ArtifactType,
    InputBinding,
    InputPort,
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

    This incomplete private checkpoint deliberately accepts only zero repairs and
    caller-supplied limits/configuration. It is not a public strategy selector.
    """
    if (
        arguments["evaluation"].max_repair_iterations != 0
        or source.empty
        or arguments.get("preview_num_records") is not None
    ):
        reject(EffectCode.UNSUPPORTED)
    graph = DataGraph.new()
    targets = []
    for text in source[COL_TEXT]:
        graph, target = graph.add_text(text)
        targets.append(target)
    size = len(targets)
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
    workflow_id = WorkflowId.new()
    names = ("prefix", "quality", "privacy", "comparison", "metrics")
    ids = {name: NodeId.new(workflow=workflow_id) for name in names}
    text = ArtifactType(name="anonymizer.text", revision=1)
    qa = REWRITE_TYPES[RewriteQAValue]
    declarations = (
        ("prefix", {"source": text}, {"candidate": text, "qa": qa}),
        ("quality", {"candidate": text, "qa": qa}, {"answers": REWRITE_TYPES[QualityAnswersValue]}),
        ("privacy", {"candidate": text, "qa": qa}, {"answers": REWRITE_TYPES[PrivacyAnswersValue]}),
        (
            "comparison",
            {"qa": qa, "answers": REWRITE_TYPES[QualityAnswersValue]},
            {"comparison": REWRITE_TYPES[QualityComparisonValue]},
        ),
        (
            "metrics",
            {
                "candidate": text,
                "qa": qa,
                "privacy": REWRITE_TYPES[PrivacyAnswersValue],
                "comparison": REWRITE_TYPES[QualityComparisonValue],
            },
            {"metrics": REWRITE_TYPES[EvaluationMetricsValue]},
        ),
    )
    nodes = []
    capabilities = []
    policies = []
    for name, inputs, outputs in declarations:
        local = name == "metrics"
        operation = OperationSpec(
            name=f"rewrite-{name}",
            inputs=tuple(InputPort(name=k, artifact_type=v) for k, v in inputs.items()),
            outputs=tuple(OutputPort(name=k, artifact_type=v) for k, v in outputs.items()),
            output_dependencies=tuple(
                OutputDependency(output=k, inputs=frozenset(inputs), identity_input=None) for k in outputs
            ),
            outcomes=(
                OutcomeSpec(
                    name="ok",
                    category="success",
                    produced_ports=frozenset(outputs),
                    context=frozenset(),
                    evidence=frozenset(),
                    state_effects=frozenset(),
                    model_requirements=frozenset(),
                    ceiling=ResourceCeiling(
                        max_activations=1,
                        max_model_requests=0 if local else None,
                        max_input_bytes=None,
                        max_output_bytes=None,
                    ),
                ),
            ),
        )
        nodes.append(OperationNode(id=ids[name], operation=operation))
        capability, policy = _frame_implementation(
            ids[name],
            operation,
            configuration,
            submission_owner=None if name == "prefix" else ids["prefix"],
            local=local,
        )
        capabilities.append(capability)
        policies.append(policy)
    bindings = []
    edges = set()
    for name, inputs, _ in declarations:
        for port in inputs:
            predecessor = (
                None
                if name == "prefix"
                else "quality"
                if name == "comparison" and port == "answers"
                else "privacy"
                if port == "privacy"
                else "comparison"
                if port == "comparison"
                else "prefix"
            )
            origin = (
                WorkflowInputRef(port="source")
                if predecessor is None
                else NodeOutputRef(node=ids[predecessor], port="answers" if port == "privacy" else port)
            )
            bindings.append(InputBinding(source=origin, destination=NodeInputRef(node=ids[name], port=port)))
            if predecessor is not None:
                edges.add((predecessor, name))
    interface = replace(
        nodes[0].operation,
        name="rewrite-initial-evaluation",
        outputs=nodes[0].operation.outputs[:1],
        output_dependencies=nodes[0].operation.output_dependencies[:1],
        outcomes=(
            replace(
                nodes[0].operation.outcomes[0],
                produced_ports=frozenset({"candidate"}),
                ceiling=ResourceCeiling(
                    max_activations=5, max_model_requests=None, max_input_bytes=None, max_output_bytes=None
                ),
            ),
        ),
    )
    raw = admit_static_workflow(
        workflow=workflow_id,
        interface=interface,
        nodes=tuple(nodes),
        input_bindings=tuple(bindings),
        output_bindings=(
            OutputBinding(
                source=NodeOutputRef(node=ids["prefix"], port="candidate"),
                destination=WorkflowOutputRef(port="candidate"),
            ),
        ),
        outcome_bindings=(
            OutcomeBinding(
                source=NodeOutcomeRef(node=ids["metrics"], outcome="ok"), destination=WorkflowOutcomeRef(outcome="ok")
            ),
        ),
        sequence=tuple(SequenceEdge(before=ids[a], after=ids[b]) for a, b in sorted(edges)),
        choices=(),
        protection=(),
        limits=WorkflowLimits(
            max_nodes=5,
            max_bindings=20,
            max_sequence_edges=10,
            max_choices=0,
            max_branch_members=0,
            max_subgraph_depth=1,
            max_choice_states=1,
        ),
    )
    prepared = _prepare_frame_plan(
        data,
        raw,
        tuple(capabilities),
        tuple(BoundInput(target=t, source=t, port="source", artifact_type=text) for t in targets),
    )
    slots = {
        name: next(slot.index for slot in prepared.reservation_recipe if slot.template == ids[name]) for name in names
    }

    loop = asyncio.get_running_loop()

    def callback(inputs: tuple[AssociationInput, ...], requests: DelegatedRequests) -> LocalCompleted:
        binding = RewriteEvaluationBinding(
            requests,
            requests.bind_frame(source, targets=tuple(targets)),
            slots,
            loop,
            _evaluation_state(0 if combined else None, COL_REWRITTEN_TEXT_INITIAL if combined else COL_REWRITTEN_TEXT),
        )
        runner = (CombinedRewriteWorkflow if combined else RewriteWorkflow)(adapter, semantic_binding=binding)
        runner.run(source, **arguments)
        return LocalCompleted(
            results=tuple(
                AssociationResult(
                    association=item.association, outcome="ok", outputs=(), consumed_context_ports=frozenset()
                )
                for item in inputs
            )
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
            max_outputs_per_association=2,
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
            max_port_facts=30 * size,
            max_provenance_edges=30 * size,
        ),
    )
