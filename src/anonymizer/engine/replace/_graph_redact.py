# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check Redact against the exact source and its accepted candidate spans."""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import TypeAdapter

from anonymizer.config.replace_strategies import LocalReplaceMethod, Redact
from anonymizer.engine.detection.artifacts import EntitySpansValue
from anonymizer.engine.graph_sdk._workflow_values import ArtifactType
from anonymizer.engine.graph_sdk.capabilities import (
    ConfigBoolean,
    ConfigField,
    ConfigSequence,
    ConfigText,
    FrozenConfig,
)
from anonymizer.engine.graph_sdk.executor import AssessmentFinding, LocalAssessmentResult, LocalCompleted
from anonymizer.engine.graph_sdk.requests import (
    AssociationInput,
    AssociationResult,
    PortArtifact,
    SemanticAssociation,
    TextArtifactValue,
)
from anonymizer.engine.replace.strategies import ReplacementEntry, apply_entity_spans

REDACT_MEANING = "All admitted entity spans are redacted in the exact source text"
REDACT_EVIDENCE_TYPE = ArtifactType(name="anonymizer.redact-check", revision=1)
REDACT_MATCH = AssessmentFinding(status="satisfied", code="span_redaction_matches")
REDACT_MISMATCH = AssessmentFinding(status="unsatisfied", code="span_redaction_mismatch")


def redact_configuration(strategy: Redact) -> FrozenConfig:
    """Snapshot existing Redact fields; an empty sequence preserves an empty template."""
    return FrozenConfig(
        fields=(
            ConfigField(
                name="format_template",
                value=ConfigSequence(
                    items=(ConfigText(value=strategy.format_template),) if strategy.format_template else ()
                ),
            ),
            ConfigField(name="normalize_label", value=ConfigBoolean(value=strategy.normalize_label)),
        )
    )


def redact_strategy(configuration: FrozenConfig) -> Redact:
    """Reconstruct the strategy from its admitted, immutable configuration."""
    fields = {field.name: field.value for field in configuration.fields}
    template, normalize = fields.get("format_template"), fields.get("normalize_label")
    if (
        set(fields) != {"format_template", "normalize_label"}
        or type(template) is not ConfigSequence
        or len(template.items) > 1
        or any(type(item) is not ConfigText for item in template.items)
        or type(normalize) is not ConfigBoolean
    ):
        raise ValueError("Invalid admitted Redact configuration")
    return Redact(
        format_template="".join(item.value for item in template.items if isinstance(item, ConfigText)),
        normalize_label=normalize.value,
    )


def local_configuration(strategy: LocalReplaceMethod) -> FrozenConfig:
    """Freeze local options with the public discriminator; retain Redact's established shape."""
    if isinstance(strategy, Redact):
        return redact_configuration(strategy)
    return FrozenConfig(
        fields=(ConfigField(name=type(strategy).__name__.lower(), value=ConfigText(value=strategy.model_dump_json())),)
    )


def local_strategy(configuration: FrozenConfig) -> LocalReplaceMethod:
    """Reconstruct the existing public model from the single admitted configuration."""
    if len(configuration.fields) == 1:
        (field,) = configuration.fields
        if field.name in {"hash", "annotate"} and type(field.value) is ConfigText:
            return TypeAdapter(LocalReplaceMethod).validate_python(
                {**json.loads(field.value.value), "kind": field.name}
            )
    return redact_strategy(configuration)


@dataclass(frozen=True, slots=True)
class RedactAssessment:
    configuration: FrozenConfig

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
        strategy = redact_strategy(self.configuration)
        expected, application = apply_entity_spans(
            source.value.text,
            spans.entities,
            [
                ReplacementEntry(entity.value, entity.label, strategy.replace(text=entity.value, label=entity.label))
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
