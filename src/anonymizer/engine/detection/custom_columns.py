# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom column generators for the entity detection workflow.

Each function is a single step in the NDD pipeline defined by
``EntityDetectionWorkflow.detect_and_validate_entities``.  They use the
``@custom_column_generator`` decorator so DataDesigner can execute them
as row-level transforms between LLM calls.
"""

from __future__ import annotations

import json
from typing import Any

from data_designer.config import custom_column_generator

from anonymizer.engine.constants import (
    COL_AUGMENTED_ENTITIES,
    COL_DETECTED_ENTITIES,
    COL_INITIAL_TAGGED_TEXT,
    COL_MERGED_ENTITIES,
    COL_MERGED_TAGGED_TEXT,
    COL_RAW_DETECTED,
    COL_REGEX_ACCEPTED_ENTITIES,
    COL_REGEX_ENTITIES,
    COL_SEED_ENTITIES,
    COL_SEED_ENTITIES_JSON,
    COL_SEED_TAGGED_TEXT,
    COL_SEED_VALIDATION_CANDIDATES,
    COL_TAG_NOTATION,
    COL_TAGGED_TEXT,
    COL_TEXT,
    COL_VALIDATED_ENTITIES,
    COL_VALIDATED_SEED_ENTITIES,
    COL_VALIDATION_CANDIDATES,
    COL_VALIDATION_DECISIONS,
)
from anonymizer.engine.detection.postprocess import (
    EntitySpan,
    apply_augmented_entities,
    apply_validation_decisions,
    build_tagged_text,
    build_validation_candidates,
    build_validation_overlap_groups,
    build_validation_tagged_text,
    coalesce_exact_entity_candidates,
    enforce_regex_constrained_evidence,
    entity_has_source_prefix,
    expand_entity_occurrences,
    filter_excluded_entity_spans,
    get_tag_notation,
    merge_entity_sources,
    normalize_label,
    normalize_labels,
    parse_raw_entities,
)
from anonymizer.engine.schemas import (
    EntitiesSchema,
    RawValidationDecisionsSchema,
    ValidatedDecisionSchema,
    ValidatedDecisionsSchema,
    ValidationCandidatesSchema,
)


@custom_column_generator(
    required_columns=[COL_TEXT, COL_RAW_DETECTED, COL_REGEX_ENTITIES],
    side_effect_columns=[COL_TAG_NOTATION],
)
def parse_detected_entities(
    row: dict[str, Any],
    *,
    excluded_entity_labels: list[str] | None = None,
) -> dict[str, Any]:
    """Parse detector payload and produce seed entities."""
    text = str(row.get(COL_TEXT, ""))
    detector_entities = parse_raw_entities(
        raw_response=str(row.get(COL_RAW_DETECTED, "")),
        text=text,
    )
    detector_entities = filter_excluded_entity_spans(detector_entities, excluded_entity_labels)
    regex_entities = _parse_entity_spans(row.get(COL_REGEX_ENTITIES, {}))
    entities = coalesce_exact_entity_candidates(regex_entities, detector_entities)
    seed_entities = [entity.as_dict() for entity in entities]
    row[COL_SEED_ENTITIES] = EntitiesSchema(entities=seed_entities).model_dump(mode="json")
    row[COL_TAG_NOTATION] = get_tag_notation(text=text)
    return row


@custom_column_generator(
    required_columns=[COL_TEXT, COL_VALIDATED_SEED_ENTITIES, COL_AUGMENTED_ENTITIES],
    side_effect_columns=[COL_MERGED_TAGGED_TEXT, COL_VALIDATION_CANDIDATES],
)
def merge_and_build_candidates(
    row: dict[str, Any],
    *,
    excluded_entity_labels: list[str] | None = None,
    regex_constrained_entity_labels: list[str] | None = None,
) -> dict[str, Any]:
    """Merge validated seed + augmented entities, then build tagged text and validation candidates.

    Contract:
    - ``COL_VALIDATED_SEED_ENTITIES`` and ``COL_MERGED_ENTITIES`` store ``EntitiesSchema`` payloads.
    - ``COL_VALIDATION_CANDIDATES`` stores ``ValidationCandidatesSchema`` payloads.
    """
    text = str(row.get(COL_TEXT, ""))
    seed_spans = _parse_entity_spans(row.get(COL_VALIDATED_SEED_ENTITIES, {}))
    merged = apply_augmented_entities(
        text=text,
        entities=seed_spans,
        augmented_output=row.get(COL_AUGMENTED_ENTITIES, {}),
        excluded_entity_labels=set(excluded_entity_labels or []),
        regex_constrained_entity_labels=set(regex_constrained_entity_labels or []),
    )
    merged_entities = [entity.as_dict() for entity in merged]
    row[COL_MERGED_ENTITIES] = EntitiesSchema(entities=merged_entities).model_dump(mode="json")
    row[COL_MERGED_TAGGED_TEXT] = build_tagged_text(text=text, entities=merged)
    row[COL_VALIDATION_CANDIDATES] = ValidationCandidatesSchema(
        candidates=build_validation_candidates(text=text, entities=merged)
    ).model_dump(mode="json")
    return row


@custom_column_generator(
    required_columns=[
        COL_TEXT,
        COL_SEED_ENTITIES,
        COL_VALIDATED_ENTITIES,
        COL_REGEX_ENTITIES,
        COL_REGEX_ACCEPTED_ENTITIES,
    ],
    side_effect_columns=[COL_INITIAL_TAGGED_TEXT, COL_SEED_ENTITIES_JSON, COL_VALIDATED_SEED_ENTITIES],
)
def apply_validation_to_seed_entities(
    row: dict[str, Any],
    *,
    excluded_entity_labels: list[str] | None = None,
    regex_constrained_entity_labels: list[str] | None = None,
) -> dict[str, Any]:
    """Apply validation decisions to detector entities before augmentation."""
    text = str(row.get(COL_TEXT, ""))
    seed_spans = _parse_entity_spans(row.get(COL_SEED_ENTITIES, {}))
    llm_validated_seed = apply_validation_decisions(
        entities=seed_spans,
        validation_output=row.get(COL_VALIDATED_ENTITIES, {}),
    )
    accepted_regex = _parse_entity_spans(row.get(COL_REGEX_ACCEPTED_ENTITIES, {}))
    validated_seed = _merge_detection_routes(accepted_regex, llm_validated_seed)
    regex_evidence = _validated_regex_evidence(row, accepted_regex=accepted_regex)
    validated_seed = enforce_regex_constrained_evidence(
        validated_seed,
        regex_constrained_entity_labels=regex_constrained_entity_labels,
        regex_evidence=regex_evidence,
    )
    validated_seed = filter_excluded_entity_spans(validated_seed, excluded_entity_labels)
    seed_entities = [entity.as_dict() for entity in validated_seed]
    row[COL_VALIDATED_SEED_ENTITIES] = EntitiesSchema(entities=seed_entities).model_dump(mode="json")
    row[COL_SEED_ENTITIES_JSON] = json.dumps(seed_entities)
    row[COL_INITIAL_TAGGED_TEXT] = build_tagged_text(text=text, entities=validated_seed)
    return row


@custom_column_generator(
    required_columns=[COL_TEXT, COL_SEED_ENTITIES],
    side_effect_columns=[COL_SEED_TAGGED_TEXT],
)
def prepare_validation_inputs(row: dict[str, Any]) -> dict[str, Any]:
    """Build prompt inputs for every detector or validation-route regex candidate.

    Locally accepted regex entities travel in ``COL_REGEX_ACCEPTED_ENTITIES``
    and are not part of the seed set. An exact accepted regex match therefore
    must not suppress validation of an independent detector candidate at the
    same label and span; only that validation can grant occurrence propagation.
    """
    text = str(row.get(COL_TEXT, ""))
    seed_spans = _parse_entity_spans(row.get(COL_SEED_ENTITIES, {}))
    validation_spans = seed_spans
    overlap_groups = build_validation_overlap_groups(
        seed_spans,
        {entity.entity_id for entity in validation_spans},
    )
    row[COL_SEED_TAGGED_TEXT] = build_validation_tagged_text(
        text=text,
        entities=seed_spans,
        overlap_groups=overlap_groups,
    )
    row[COL_SEED_VALIDATION_CANDIDATES] = ValidationCandidatesSchema(
        candidates=build_validation_candidates(text=text, entities=validation_spans)
    ).model_dump(mode="json")
    return row


@custom_column_generator(
    required_columns=[COL_VALIDATION_DECISIONS, COL_SEED_VALIDATION_CANDIDATES],
)
def enrich_validation_decisions(row: dict[str, Any]) -> dict[str, Any]:
    """Enrich validation decisions with entity value and filter to known candidate IDs only."""
    raw_decisions = RawValidationDecisionsSchema.from_raw(row.get(COL_VALIDATION_DECISIONS, {}))
    candidates = ValidationCandidatesSchema.from_raw(row.get(COL_SEED_VALIDATION_CANDIDATES, {}))

    candidate_lookup = {c.id: c for c in candidates.candidates}
    valid_ids = set(candidate_lookup)

    enriched = [
        ValidatedDecisionSchema(
            id=d.id,
            decision=d.decision,
            proposed_label=d.proposed_label,
            reason=d.reason,
            value=candidate_lookup[d.id].value,
            label=candidate_lookup[d.id].label,
        )
        for d in raw_decisions.decisions
        if d.id in valid_ids
    ]

    row[COL_VALIDATED_ENTITIES] = ValidatedDecisionsSchema(decisions=enriched).model_dump(mode="json")
    return row


@custom_column_generator(
    required_columns=[
        COL_TEXT,
        COL_MERGED_ENTITIES,
        COL_VALIDATED_ENTITIES,
        COL_REGEX_ENTITIES,
        COL_REGEX_ACCEPTED_ENTITIES,
    ],
    side_effect_columns=[COL_TAGGED_TEXT],
)
def apply_validation_and_finalize(
    row: dict[str, Any],
    *,
    excluded_entity_labels: list[str] | None = None,
    allowed_entity_labels: list[str] | None = None,
    regex_constrained_entity_labels: list[str] | None = None,
) -> dict[str, Any]:
    """Apply keep/reclass/drop decisions, expand to all occurrences, and produce final outputs."""
    text = str(row.get(COL_TEXT, ""))
    merged = _parse_entity_spans(row.get(COL_MERGED_ENTITIES, {}))
    validated = apply_validation_decisions(
        entities=merged,
        validation_output=row.get(COL_VALIDATED_ENTITIES, {}),
    )
    accepted_regex = _parse_entity_spans(row.get(COL_REGEX_ACCEPTED_ENTITIES, {}))
    protected = _merge_detection_routes(accepted_regex, validated)
    regex_evidence = _validated_regex_evidence(row, accepted_regex=accepted_regex)
    protected = enforce_regex_constrained_evidence(
        protected,
        regex_constrained_entity_labels=regex_constrained_entity_labels,
        regex_evidence=regex_evidence,
    )
    if allowed_entity_labels is not None:
        allowed = normalize_labels(allowed_entity_labels)
        protected = [entity for entity in protected if normalize_label(entity.label) in allowed]
    protected = filter_excluded_entity_spans(protected, excluded_entity_labels)
    expanded = expand_entity_occurrences(text=text, entities=protected)
    row[COL_DETECTED_ENTITIES] = EntitiesSchema(entities=[entity.as_dict() for entity in expanded]).model_dump(
        mode="json"
    )
    row[COL_TAGGED_TEXT] = build_tagged_text(text=text, entities=expanded)
    return row


def _parse_entity_spans(raw_payload: object) -> list[EntitySpan]:
    parsed = EntitiesSchema.from_raw(raw_payload)
    return [
        EntitySpan(
            entity_id=e.id,
            value=e.value,
            label=e.label,
            start_position=e.start_position,
            end_position=e.end_position,
            score=e.score,
            source=e.source,
            propagate_occurrences=e.propagate_occurrences,
        )
        for e in parsed.entities
    ]


def _validated_regex_evidence(
    row: dict[str, Any],
    *,
    accepted_regex: list[EntitySpan],
) -> list[EntitySpan]:
    """Return regex evidence that survived its configured validation route.

    Directly accepted matches are evidence immediately. Matches routed through
    contextual validation count only if they survive and retain the normalized
    label and exact span produced by their regex rule. A dropped match cannot
    authorize another candidate at the same span, and a reclassified regex
    candidate cannot authorize its newly proposed label.
    """
    regex_candidates = _parse_entity_spans(row.get(COL_REGEX_ENTITIES, {}))
    validated_candidates = apply_validation_decisions(
        entities=regex_candidates,
        validation_output=row.get(COL_VALIDATED_ENTITIES, {}),
    )
    original_by_id = {entity.entity_id: entity for entity in regex_candidates}
    surviving_evidence = [
        entity
        for entity in validated_candidates
        if (original := original_by_id.get(entity.entity_id)) is not None
        and normalize_label(entity.label) == normalize_label(original.label)
        and entity.start_position == original.start_position
        and entity.end_position == original.end_position
    ]
    return [*accepted_regex, *surviving_evidence]


def _merge_detection_routes(*routes: list[EntitySpan]) -> list[EntitySpan]:
    """Merge validation routes without allowing route order to change source precedence."""
    entities = [entity for route in routes for entity in route]
    source_groups = _partition_by_source_precedence(entities)
    coalesced = coalesce_exact_entity_candidates(*source_groups)
    return merge_entity_sources(*_partition_by_source_precedence(coalesced))


def _partition_by_source_precedence(
    entities: list[EntitySpan],
) -> tuple[list[EntitySpan], list[EntitySpan], list[EntitySpan]]:
    """Partition entities into user-regex, built-in-regex, and other precedence groups."""
    user_regex = [entity for entity in entities if entity_has_source_prefix(entity, "regex_user:")]
    builtin_regex = [
        entity
        for entity in entities
        if not entity_has_source_prefix(entity, "regex_user:") and entity_has_source_prefix(entity, "regex_builtin:")
    ]
    other_sources = [
        entity
        for entity in entities
        if not entity_has_source_prefix(entity, "regex_user:")
        and not entity_has_source_prefix(entity, "regex_builtin:")
    ]
    return user_regex, builtin_regex, other_sources
