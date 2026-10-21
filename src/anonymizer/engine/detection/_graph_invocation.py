# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Whole-frame detection/Redact consumer of an admitted graph invocation.

Source frames always bind authenticated positions and typed span/text outputs.
DataDesigner keeps its own transport, retries and scheduling.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextvars import copy_context
from dataclasses import dataclass, replace
from typing import Any, cast

import pandas as pd
from data_designer.config import ModelConfig

from anonymizer.config.models import DetectionModelSelection
from anonymizer.engine.constants import COL_FINAL_ENTITIES, COL_REPLACED_TEXT
from anonymizer.engine.detection.artifacts import ENTITY_SPANS_TYPE, EntitySpansValue
from anonymizer.engine.detection.detection_workflow import (
    EntityDetectionWorkflow,
)
from anonymizer.engine.detection.postprocess import EntitySpan
from anonymizer.engine.detection.run_arguments import detection_run_arguments
from anonymizer.engine.graph_sdk._execution_delegated import drive
from anonymizer.engine.graph_sdk._execution_values import DelegatedRequests, LocalCompleted, LocalFailure
from anonymizer.engine.graph_sdk._graph_values import ArtifactRef, DatumId
from anonymizer.engine.graph_sdk.capabilities import FrozenConfig
from anonymizer.engine.graph_sdk.requests import (
    AssociationInput,
    AssociationResult,
    PortArtifact,
    SemanticAssociation,
    TextArtifactValue,
)
from anonymizer.engine.harness.frames import FramePublication, InvocationFrame
from anonymizer.engine.replace._graph_redact import local_strategy
from anonymizer.engine.replace.strategies import apply_local_replace_strategy
from anonymizer.engine.schemas import EntitiesSchema
from anonymizer.interface.errors import AnonymizerWorkflowError


@dataclass(frozen=True, slots=True)
class DetectionRedactInvocation:
    """Execute actual detection once and return content to the graph publication owner."""

    source: pd.DataFrame
    targets: tuple[DatumId, ...]
    detection: EntityDetectionWorkflow
    model_configs: list[ModelConfig]
    selected_models: DetectionModelSelection
    redact_configuration: FrozenConfig
    detection_options: Mapping[str, Any]
    output_port: str
    outcome: str
    semantic_slots: Mapping[str, int] | None = None
    semantic_transform_slot: int | None = None
    semantic_entry_slot: int | None = None
    handoff: Callable[[Callable[[], LocalCompleted | LocalFailure]], LocalCompleted | LocalFailure] | None = None

    def run(self, inputs: tuple[AssociationInput, ...], requests: DelegatedRequests) -> LocalCompleted | LocalFailure:
        context = copy_context()
        return (
            self._run(inputs, requests)
            if self.handoff is None
            else self.handoff(lambda: context.run(self._run, inputs, requests))
        )

    def _run(self, inputs: tuple[AssociationInput, ...], requests: DelegatedRequests) -> LocalCompleted:
        if any(
            type(item.association) is not SemanticAssociation
            or len(item.inputs) != 1
            or item.inputs[0].artifact is None
            for item in inputs
        ):
            raise AnonymizerWorkflowError("DD graph detection requires one source-text input per association")
        frame = requests.bind_frame(self.source, targets=self.targets, source_limit=len(self.targets))
        arguments: dict[str, Any] = {
            "model_configs": self.model_configs,
            "selected_models": self.selected_models,
            "source_frame": frame,
            **self.detection_options,
        }
        from anonymizer.engine.detection._semantic_detection import (
            complete_detection_entry,
            complete_empty_stage,
            lower_detection,
        )

        # The submitting profile classified the runner: only a builtin one has detection children.
        if self.semantic_slots is not None:
            drive(complete_detection_entry(requests, inputs))
            detected = lower_detection(
                self.detection,
                self.source,
                arguments,
                requests,
                frame,
                dict(self.semantic_slots),
                self.semantic_entry_slot,
            )
        else:
            detected = self.detection.run(self.source, **detection_run_arguments(self.detection.run, arguments))
        positions = detected.source_positions
        if len(positions) != len(detected.dataframe):
            raise AnonymizerWorkflowError("DD graph detection has unclosed source results")
        successful = tuple(frame.associations[position] for position in positions)
        rows = {association: index for index, association in enumerate(successful)}
        protected = apply_local_replace_strategy(
            detected.dataframe,
            strategy=local_strategy(self.redact_configuration),
        )
        text = dict(zip(successful, protected[COL_REPLACED_TEXT], strict=True))
        spans = {
            item.association: EntitySpansValue(
                source=cast(ArtifactRef, item.inputs[0].artifact),
                entities=tuple(
                    EntitySpan(
                        entity.id,
                        entity.value,
                        entity.label,
                        entity.start_position,
                        entity.end_position,
                        entity.score,
                        entity.source,
                    )
                    for entity in EntitiesSchema.from_raw(
                        detected.dataframe[COL_FINAL_ENTITIES].iloc[rows[item.association]]
                    ).entities
                ),
            )
            for item in inputs
            if item.association in rows
        }
        results = tuple(
            AssociationResult(
                association=item.association,
                outcome=self.outcome,
                consumed_context_ports=frozenset(),
                outputs=(
                    PortArtifact(
                        port="spans", artifact_type=ENTITY_SPANS_TYPE, artifact=None, value=spans[item.association]
                    ),
                    PortArtifact(
                        port=self.output_port,
                        artifact_type=item.inputs[0].artifact_type,
                        artifact=None,
                        value=TextArtifactValue(text=text[item.association]),
                    ),
                ),
            )
            for item in inputs
            if item.association in rows
        )
        if self.semantic_transform_slot is not None:
            transform_slot = self.semantic_transform_slot
            completed_sources = {}

            async def finish() -> None:
                nonlocal results
                entered = await requests.enter_operations(
                    transform_slot, tuple(cast(SemanticAssociation, result.association) for result in results)
                )
                if any(item is None for item in entered):
                    raise AnonymizerWorkflowError("Detection transform is not ready")
                accepted = tuple(
                    replace(result, association=cast(AssociationInput, item).association)
                    for result, item in zip(results, entered, strict=True)
                )
                outcomes = await requests.complete_operations(
                    tuple(
                        (cast(SemanticAssociation, result.association), LocalCompleted(results=(result,)))
                        for result in accepted
                    )
                )
                for outcome in outcomes:
                    if outcome.condition != "result" or outcome.category != "success":
                        raise AnonymizerWorkflowError(f"Local transform semantic output was rejected: {outcome!r}")
                if len(outcomes) != len(accepted):
                    raise AnonymizerWorkflowError("Local transform semantic output was rejected")
                completed_sources.update(
                    (result.association, projected.association)
                    for result, projected in zip(results, accepted, strict=True)
                )
                if not frame.associations:
                    item = await requests.enter_operation(transform_slot, None)
                    if item is None:
                        raise AnonymizerWorkflowError("Empty detection transform is not ready")
                    await complete_empty_stage(requests, [item])
                results = accepted

            drive(finish())
            successful = tuple(completed_sources[association] for association in successful)
        results_by_source = {item.association: item for item in results}
        results = tuple(results_by_source[association] for association in successful)
        return LocalCompleted(
            results=tuple(
                AssociationResult(
                    association=item.association, outcome="ok", outputs=(), consumed_context_ports=frozenset()
                )
                for item in inputs
                if item.association in rows
            )
            if self.semantic_transform_slot is not None
            else results,
            failed_sources=frozenset(set(frame.associations) - set(rows)),
            frame=InvocationFrame(protected, detected.dataframe, tuple(detected.failed_records), (), 0.0, 0.0, None)
            if not frame.associations
            else FramePublication.capture(
                protected,
                associations=successful,
                text_ports=((COL_REPLACED_TEXT, self.output_port),),
                failed_records=tuple(detected.failed_records),
                opaque_source=self.source,
                span_ports=(
                    (
                        COL_FINAL_ENTITIES,
                        "spans",
                        tuple(
                            cast(EntitySpansValue, next(port.value for port in result.outputs if port.port == "spans"))
                            for result in results
                        ),
                    ),
                ),
            ),
        )
