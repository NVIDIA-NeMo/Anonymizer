# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source-bound candidate detections, distinct from accepted subject evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from anonymizer.engine.detection.postprocess import EntitySpan
from anonymizer.engine.graph_sdk._artifact_shape import ArtifactShape, content_bytes
from anonymizer.engine.graph_sdk._effect_values import EffectCode, PrivateValue, reject
from anonymizer.graph._values import ArtifactRef, InvocationId
from anonymizer.graph.workflow import ArtifactType

ENTITY_SPANS_TYPE = ArtifactType(name="anonymizer.candidate-spans", revision=1)


def valid_span_source(source: object) -> bool:
    """Validate a source reference before comparing it with a consumed artifact."""
    return (
        type(source) is ArtifactRef
        and type(source.invocation) is InvocationId
        and type(source.key) is int
        and source.key >= 0
        and type(source.version) is int
        and source.version > 0
    )


def valid_entity_span(entity: object) -> bool:
    """Validate closed field shapes without running foreign value hooks."""
    return (
        type(entity) is EntitySpan
        and all(type(value) is str for value in (entity.entity_id, entity.value, entity.label, entity.source))
        and type(entity.start_position) is int
        and type(entity.end_position) is int
        and 0 <= entity.start_position < entity.end_position
        and type(entity.score) is float
        and math.isfinite(entity.score)
    )


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class EntitySpansValue(PrivateValue):
    """Immutable candidate spans anchored to one source artifact version.

    Construction validates the closed shape; publication additionally checks the
    reference and exact substrings against the declared authoritative text input.
    """

    source: ArtifactRef
    entities: tuple[EntitySpan, ...]
    shape: ArtifactShape = field(init=False, compare=False)

    def __post_init__(self) -> None:
        if not valid_span_source(self.source) or type(self.entities) is not tuple:
            reject(EffectCode.INVALID_TYPE)
        if any(not valid_entity_span(entity) for entity in self.entities):
            reject(EffectCode.INVALID_VALUE)
        # Copy the domain values at this boundary, retaining no caller-owned container.
        entities = tuple(
            EntitySpan(e.entity_id, e.value, e.label, e.start_position, e.end_position, e.score, e.source)
            for e in self.entities
        )
        object.__setattr__(self, "entities", entities)
        object.__setattr__(
            self,
            "shape",
            ArtifactShape(
                sum(content_bytes(s) for e in entities for s in (e.entity_id, e.value, e.label, e.source)),
                len(entities),
            ),
        )

    def matches_source(self, reference: ArtifactRef, text: str) -> bool:
        """Check exact source ownership and spans after closed-shape validation."""
        return self.source == reference and all(
            e.end_position <= len(text) and text[e.start_position : e.end_position] == e.value for e in self.entities
        )
