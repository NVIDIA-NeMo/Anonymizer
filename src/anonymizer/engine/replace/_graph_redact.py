# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check Redact against the exact source and its accepted candidate spans."""

from __future__ import annotations

from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.detection.artifacts import EntitySpansValue
from anonymizer.engine.graph_sdk.executor import AssessmentFinding, LocalAssessmentResult, LocalCompleted
from anonymizer.engine.graph_sdk.requests import (
    AssociationInput,
    AssociationResult,
    PortArtifact,
    SemanticAssociation,
    TextArtifactValue,
)
from anonymizer.engine.replace.strategies import ReplacementEntry, apply_entity_spans
from anonymizer.graph.workflow import ArtifactType

REDACT_MEANING = "All admitted entity spans are redacted in the exact source text"
REDACT_EVIDENCE_TYPE = ArtifactType(name="anonymizer.redact-check", revision=1)
REDACT_MATCH = AssessmentFinding(status="satisfied", code="span_redaction_matches")
REDACT_MISMATCH = AssessmentFinding(status="unsatisfied", code="span_redaction_mismatch")


class RedactAssessment:
    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted:
        (item,) = request
        ports = {port.port: port for port in item.inputs}
        source, spans, candidate = ports["source"], ports["spans"].value, ports["candidate"].value
        if (
            not isinstance(item.association, SemanticAssociation)
            or source.artifact is None
            or type(source.value) is not TextArtifactValue
            or type(spans) is not EntitySpansValue
            or type(candidate) is not TextArtifactValue
        ):
            raise ValueError("Redact assessment requires source, spans and candidate artifacts")
        expected, application = apply_entity_spans(
            source.value.text,
            spans.entities,
            [
                ReplacementEntry(entity.value, entity.label, Redact().replace(text=entity.value, label=entity.label))
                for entity in spans.entities
            ],
            allow_value_fallback=False,
        )
        matched = (
            spans.matches_source(source.artifact, source.value.text)
            and application.skipped_span_count == 0
            and expected == candidate.text
        )
        finding = REDACT_MATCH if matched else REDACT_MISMATCH
        return LocalCompleted(
            results=(
                AssociationResult(
                    association=item.association,
                    outcome="ok",
                    consumed_context_ports=frozenset(),
                    outputs=(
                        PortArtifact(
                            port="assessment",
                            artifact_type=REDACT_EVIDENCE_TYPE,
                            artifact=None,
                            value=TextArtifactValue(text=finding.code),
                        ),
                    ),
                ),
            ),
            assessments=(
                LocalAssessmentResult(
                    association=item.association,
                    promise="redaction",
                    evidence_port="assessment",
                    finding=finding,
                ),
            ),
        )
