# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Whole-frame detection/Redact consumer of an admitted graph invocation.

This is the development consumer, not the public route selector. The caller must
admit independent detector, validator-chunk and augmenter chains and register
its exact DD physical resource. Unsupported DD followups reject at that boundary.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd
from data_designer.config import ModelConfig

from anonymizer.config.models import DetectionModelSelection, ReplaceModelSelection
from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import COL_FINAL_ENTITIES, COL_REPLACED_TEXT
from anonymizer.engine.detection._frame_publication import FramePublication
from anonymizer.engine.detection.artifacts import ENTITY_SPANS_TYPE, EntitySpansValue
from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
from anonymizer.engine.detection.postprocess import EntitySpan
from anonymizer.engine.graph_sdk._execution_values import DelegatedRequests, LocalCompleted
from anonymizer.engine.graph_sdk.requests import (
    AssociationInput,
    AssociationResult,
    PortArtifact,
    SemanticAssociation,
    TextArtifactValue,
)
from anonymizer.engine.ndd._graph_requests import NddPhysicalTransport, NddRequestBridge
from anonymizer.engine.replace.replace_runner import ReplacementWorkflow
from anonymizer.engine.schemas import EntitiesSchema
from anonymizer.graph._values import ArtifactRef, DatumId
from anonymizer.interface.errors import AnonymizerWorkflowError


@dataclass(frozen=True, slots=True)
class DetectionRedactInvocation:
    """Execute actual detection once and return content to the graph publication owner."""

    source: pd.DataFrame
    targets: tuple[DatumId, ...]
    detection: EntityDetectionWorkflow
    model_configs: list[ModelConfig]
    selected_models: DetectionModelSelection
    physical: NddPhysicalTransport
    slots: Mapping[tuple[str, int], int]
    detection_options: Mapping[str, Any]
    output_port: str
    outcome: str
    execution: str = "same_loop"

    async def submit(self, inputs: tuple[AssociationInput, ...], requests: DelegatedRequests) -> LocalCompleted:
        return await asyncio.to_thread(self._run, inputs, requests)

    def _run(self, inputs: tuple[AssociationInput, ...], requests: DelegatedRequests) -> LocalCompleted:
        if any(
            type(item.association) is not SemanticAssociation
            or len(item.inputs) != 1
            or item.inputs[0].artifact is None
            for item in inputs
        ):
            raise AnonymizerWorkflowError("DD graph detection requires one source-text input per association")
        frame = requests.bind_frame(self.source, targets=self.targets)
        bridge = NddRequestBridge(requests, self.slots, self.physical)
        detected = self.detection.run(
            self.source,
            model_configs=self.model_configs,
            selected_models=self.selected_models,
            source_frame=frame,
            request_bridge=bridge,
            **self.detection_options,
        )
        # Partial-row completion needs an admitted source-failure outcome before
        # this development profile can publish it. Do not infer source identity
        # from public legacy IDs, which can legitimately repeat.
        if len(detected.dataframe) != len(frame.associations):
            raise AnonymizerWorkflowError("DD graph detection has unclosed source results")
        protected = ReplacementWorkflow().run(
            detected.dataframe,
            replace_method=Redact(),
            model_configs=self.model_configs,
            selected_models=ReplaceModelSelection(replacement_generator=self.selected_models.entity_augmenter),
        )
        text = dict(zip(frame.associations, protected.dataframe[COL_REPLACED_TEXT], strict=True))
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
                        detected.dataframe[COL_FINAL_ENTITIES].iloc[frame.associations.index(item.association)]
                    ).entities
                ),
            )
            for item in inputs
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
                        value=TextArtifactValue(text=text[cast(SemanticAssociation, item.association)]),
                    ),
                ),
            )
            for item in inputs
        )
        return LocalCompleted(
            results=results,
            frame=FramePublication.capture(
                protected.dataframe,
                associations=frame.associations,
                text_ports=((COL_REPLACED_TEXT, self.output_port),),
                failed_records=tuple(detected.failed_records + protected.failed_records),
                opaque_source=self.source,
                span_ports=(
                    (COL_FINAL_ENTITIES, "spans", tuple(spans[association] for association in frame.associations)),
                ),
            ),
        )
