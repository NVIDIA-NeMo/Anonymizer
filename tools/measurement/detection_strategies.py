#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Experimental detection strategies for benchmark-only performance probes."""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import pandas as pd
from data_designer.config import custom_column_generator
from data_designer.config.column_configs import CustomColumnConfig, LLMStructuredColumnConfig, LLMTextColumnConfig
from data_designer.config.models import ModelConfig
from dd_parser_compat import _load_embedded_json
from direct_detection_probe import DirectDetectionRequest, DirectGenerationRequest, PromptMode, build_direct_prompt
from staged_detection_probe import (
    CaseStatus as StagedCaseStatus,
)
from staged_detection_probe import (
    DirectCompletion,
    DirectDetectionClient,
    GlinerSeedClient,
    HttpxDirectDetectionClient,
    SeedSource,
    StagedDetectionCase,
    StagedDetectionRequest,
    StagedExecutionConfig,
    ValidationPromptMode,
    _run_augmentation_phase,
    _run_validation_phase,
    execute_staged_detection_case,
    normalize_birth_context_final_entities,
)

from anonymizer.config.models import DetectionModelSelection
from anonymizer.engine.constants import (
    COL_AUGMENTED_ENTITIES,
    COL_DETECTED_ENTITIES,
    COL_INITIAL_TAGGED_TEXT,
    COL_MERGED_ENTITIES,
    COL_RAW_DETECTED,
    COL_SEED_ENTITIES,
    COL_SEED_ENTITIES_JSON,
    COL_SEED_VALIDATION_CANDIDATES,
    COL_TAG_NOTATION,
    COL_TAGGED_TEXT,
    COL_TEXT,
    COL_VALIDATED_ENTITIES,
    COL_VALIDATED_SEED_ENTITIES,
    COL_VALIDATION_DECISIONS,
    _jinja,
)
from anonymizer.engine.detection import detection_workflow as dw
from anonymizer.engine.detection.chunked_validation import ChunkedValidationParams, make_chunked_validation_generator
from anonymizer.engine.detection.custom_columns import (
    apply_validation_and_finalize,
    apply_validation_to_seed_entities,
    enrich_validation_decisions,
    merge_and_build_candidates,
    parse_detected_entities,
    prepare_validation_inputs,
)
from anonymizer.engine.detection.postprocess import (
    EntitySpan,
    build_tagged_text,
    expand_entity_occurrences,
    get_tag_notation,
    parse_raw_entities,
    resolve_overlaps,
)
from anonymizer.engine.detection.rules import detect_high_confidence_entities
from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.engine.ndd.model_loader import resolve_model_alias, resolve_model_aliases
from anonymizer.engine.row_partitioning import merge_and_reorder, split_rows
from anonymizer.engine.schemas import AugmentedEntitiesSchema, EntitiesSchema, ValidationCandidatesSchema
from anonymizer.measurement import record_model_workflow

_NATIVE_DIRECT_MODEL_ALIAS = "native-direct"
_NATIVE_DIRECT_MODEL_NAME = "nvidia/nemotron-3-super"
_NATIVE_DIRECT_MODEL_PROVIDER = "local-vllm"
_NATIVE_DIRECT_ENDPOINT = "http://gpu-dev-pod-serve-svc:8000/v1"
_NATIVE_DIRECT_MAX_TOKENS = 4096
_NATIVE_DIRECT_TIMEOUT_SEC = 180.0
_GLINER_DIRECT_MODEL_ALIAS = "gliner-direct"
_GLINER_DIRECT_MODEL_NAME = "nvidia/gliner-pii"
_GLINER_DIRECT_MODEL_PROVIDER = "nvidia"
_NATIVE_STAGED_MAX_WORKERS = 4
_STRUCTURED_ASSIGNMENT_RE = re.compile(
    r"(?<![A-Za-z0-9_-])['\"]?"
    r"(?P<key>"
    r"api[_-]?key|aws[_-]?access[_-]?key[_-]?id|access[_-]?key[_-]?id|hf[_-]?token|"
    r"token|auth[_-]?token|session[_-]?id|authorization|"
    r"password|pass|secret|aws[_-]?secret[_-]?access[_-]?key|django[_-]?secret|database[_-]?url|"
    r"pin|user(?:_?name)?|username|login|account|cookie|"
    r"trace[-_]?id|request[-_]?id|req[-_]?id|order[-_]?id|tenant[-_]?id|unique[-_]?id|"
    r"url|uri|endpoint|callback|email"
    r")['\"]?\s*[:=]\s*"
    r"(?:['\"](?P<quoted>[^'\"\r\n]+)['\"]|(?P<bare>[^\s'\",;]+))",
    flags=re.IGNORECASE,
)


class ExperimentalDetectionStrategy(StrEnum):
    default = "default"
    prose_augment_focus = "prose_augment_focus"
    compact_validation = "compact_validation"
    rules_guardrail_compact_validation = "rules_guardrail_compact_validation"
    rules_guardrail = "rules_guardrail"
    rules_filter_guardrail = "rules_filter_guardrail"
    no_augment = "no_augment"
    rules_seed_no_augment = "rules_seed_no_augment"
    rules_guardrail_no_augment = "rules_guardrail_no_augment"
    rules_filter_guardrail_no_augment = "rules_filter_guardrail_no_augment"
    rules_guardrail_detector_only = "rules_guardrail_detector_only"
    detector_only = "detector_only"
    rules_only = "rules_only"
    rules_covered_or_default = "rules_covered_or_default"
    native_rules_router = "native_rules_router"
    native_candidate_validate_no_augment = "native_candidate_validate_no_augment"
    detector_native_validate_no_augment = "detector_native_validate_no_augment"
    detector_native_validate_native_augment = "detector_native_validate_native_augment"
    gliner_native_validate_no_augment = "gliner_native_validate_no_augment"
    gliner_native_validate_native_augment = "gliner_native_validate_native_augment"
    native_single_pass = "native_single_pass"
    native_single_pass_recall = "native_single_pass_recall"
    native_single_pass_values = "native_single_pass_values"
    native_single_pass_values_recall = "native_single_pass_values_recall"


_DetectAndValidate = Callable[..., dw.EntityDetectionResult]
_AugmentPrompt = Callable[..., str]
_MaterializeFinalEntities = Callable[..., dict]
PROSE_AUGMENT_FOCUS_TEXT = """\
Contextual prose recall focus:
- Re-scan untagged narrative prose for organization and institution names, named facilities, labs, research centers, street or place names, self-described beliefs, occupations, titles, family member names, and other quasi-identifiers that combine with already tagged entities.
- Prefer allowed labels when present, especially organization_name, company_name, place_name, religious_belief, political_view, occupation, first_name, last_name, city, state, university, degree, field_of_study, language, race_ethnicity, and age.
- Do not add generic common nouns, section headings, or labels outside the allowed list when strict labels are required.
"""


@dataclass(frozen=True)
class _NoAugmentOptions:
    include_rules: bool
    final_rule_guardrail: bool = False
    filter_rule_overlaps: bool = False
    rule_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class _NativeStagedTask:
    ordinal: int
    index: Any
    row: pd.Series


@dataclass(frozen=True)
class _NativeStagedRunParams:
    labels: list[str]
    client: DirectDetectionClient
    gliner_seed_client: GlinerSeedClient | None
    data_summary: str | None
    validation_prompt_mode: ValidationPromptMode
    validation_max_entities_per_call: int
    validation_excerpt_window_chars: int
    seed_source: SeedSource
    workflow_name: str
    skip_augmentation: bool


@dataclass(frozen=True)
class _NativeStagedRowResult:
    ordinal: int
    index: Any
    output_row: dict[str, Any] | None
    failed_record: FailedRecord | None
    case: StagedDetectionCase | None


@dataclass(frozen=True)
class _DetectorNativeValidationParams:
    labels: list[str]
    client: DirectDetectionClient
    data_summary: str | None
    validation_prompt_mode: ValidationPromptMode
    validation_max_entities_per_call: int
    validation_excerpt_window_chars: int
    workflow_name: str
    skip_augmentation: bool


@dataclass(frozen=True)
class _DetectorNativeValidationRowResult:
    ordinal: int
    index: Any
    workflow_name: str
    output_row: dict[str, Any] | None
    failed_record: FailedRecord | None
    completion: DirectCompletion | None
    elapsed_sec: float
    request_count: int


@contextmanager
def experimental_detection_strategy_context(
    strategy: ExperimentalDetectionStrategy,
    *,
    rule_labels: list[str] | None = None,
    native_client: DirectDetectionClient | None = None,
    gliner_seed_client: GlinerSeedClient | None = None,
) -> Iterator[None]:
    """Temporarily apply a benchmark-only detection strategy."""
    if strategy == ExperimentalDetectionStrategy.default:
        yield
        return

    original_method = dw.EntityDetectionWorkflow.detect_and_validate_entities
    original_augment_prompt = dw._get_augment_prompt
    original_materialize_final_entities = dw._materialize_final_entities
    if rule_labels:
        dw._materialize_final_entities = _make_rule_label_materializer(  # type: ignore[assignment]
            original_materialize_final_entities,
            rule_labels=rule_labels,
        )
    if strategy == ExperimentalDetectionStrategy.prose_augment_focus:
        dw._get_augment_prompt = _make_prose_augment_prompt(original_augment_prompt)  # type: ignore[assignment]
    else:
        dw.EntityDetectionWorkflow.detect_and_validate_entities = _method_for_strategy(  # type: ignore[method-assign]
            strategy,
            original=original_method,
            rule_labels=rule_labels,
            native_client=native_client,
            gliner_seed_client=gliner_seed_client,
        )
    try:
        yield
    finally:
        dw.EntityDetectionWorkflow.detect_and_validate_entities = original_method  # type: ignore[method-assign]
        dw._get_augment_prompt = original_augment_prompt  # type: ignore[assignment]
        dw._materialize_final_entities = original_materialize_final_entities  # type: ignore[assignment]


def _make_rule_label_materializer(
    original: _MaterializeFinalEntities,
    *,
    rule_labels: list[str],
) -> _MaterializeFinalEntities:
    def materialize_final_entities(raw: object, *, allowed_labels: set[str] | None) -> dict:
        if allowed_labels is None:
            return original(raw, allowed_labels=allowed_labels)
        return original(raw, allowed_labels={*allowed_labels, *rule_labels})

    return materialize_final_entities


def _make_prose_augment_prompt(original: _AugmentPrompt) -> _AugmentPrompt:
    def get_augment_prompt(*, data_summary: str | None, labels: list[str], strict_labels: bool = False) -> str:
        prompt = original(data_summary=data_summary, labels=labels, strict_labels=strict_labels)
        return prompt.replace("Rules:\n", f"{PROSE_AUGMENT_FOCUS_TEXT}\nRules:\n", 1)

    return get_augment_prompt


def _method_for_strategy(
    strategy: ExperimentalDetectionStrategy,
    *,
    original: _DetectAndValidate | None = None,
    rule_labels: list[str] | None = None,
    native_client: DirectDetectionClient | None = None,
    gliner_seed_client: GlinerSeedClient | None = None,
) -> _DetectAndValidate:
    if strategy == ExperimentalDetectionStrategy.compact_validation:
        if original is None:
            raise ValueError("compact_validation requires the original detection method")
        return _make_default_compact_validation_method(original)
    if strategy == ExperimentalDetectionStrategy.rules_guardrail:
        if original is None:
            raise ValueError("rules_guardrail requires the original detection method")
        return _make_default_with_rule_guardrail_method(original, rule_labels=rule_labels)
    if strategy == ExperimentalDetectionStrategy.rules_filter_guardrail:
        return _make_validated_augmented_rule_filter_guardrail_method(rule_labels=rule_labels)
    if strategy == ExperimentalDetectionStrategy.rules_guardrail_compact_validation:
        if original is None:
            raise ValueError("rules_guardrail_compact_validation requires the original detection method")
        return _make_default_with_rule_guardrail_method(
            original,
            rule_labels=rule_labels,
            compact_validation=True,
        )
    if strategy == ExperimentalDetectionStrategy.no_augment:
        return _make_validated_no_augment_method(include_rules=False)
    if strategy == ExperimentalDetectionStrategy.rules_seed_no_augment:
        return _make_validated_no_augment_method(include_rules=True, rule_labels=rule_labels)
    if strategy == ExperimentalDetectionStrategy.rules_guardrail_no_augment:
        return _make_validated_no_augment_method(
            include_rules=False,
            final_rule_guardrail=True,
            rule_labels=rule_labels,
        )
    if strategy == ExperimentalDetectionStrategy.rules_filter_guardrail_no_augment:
        return _make_validated_no_augment_method(
            include_rules=False,
            final_rule_guardrail=True,
            filter_rule_overlaps=True,
            rule_labels=rule_labels,
        )
    if strategy == ExperimentalDetectionStrategy.rules_guardrail_detector_only:
        return _make_detector_only_with_rule_guardrail_method(rule_labels=rule_labels)
    if strategy == ExperimentalDetectionStrategy.detector_only:
        return _detect_with_detector_only
    if strategy == ExperimentalDetectionStrategy.rules_covered_or_default:
        if original is None:
            raise ValueError("rules_covered_or_default requires the original detection method")
        return _make_rules_covered_or_default_method(original)
    if strategy == ExperimentalDetectionStrategy.rules_only:
        return _detect_with_rules_only
    if strategy == ExperimentalDetectionStrategy.native_rules_router:
        return _make_native_rules_router_method(native_client=native_client)
    if strategy == ExperimentalDetectionStrategy.native_candidate_validate_no_augment:
        return _make_native_candidate_validate_no_augment_method(native_client=native_client)
    if strategy == ExperimentalDetectionStrategy.detector_native_validate_no_augment:
        return _make_detector_native_validate_no_augment_method(native_client=native_client)
    if strategy == ExperimentalDetectionStrategy.detector_native_validate_native_augment:
        return _make_detector_native_validate_native_augment_method(native_client=native_client)
    if strategy == ExperimentalDetectionStrategy.gliner_native_validate_no_augment:
        return _make_gliner_native_validate_no_augment_method(
            native_client=native_client,
            gliner_seed_client=gliner_seed_client,
        )
    if strategy == ExperimentalDetectionStrategy.gliner_native_validate_native_augment:
        return _make_gliner_native_validate_native_augment_method(
            native_client=native_client,
            gliner_seed_client=gliner_seed_client,
        )
    if strategy == ExperimentalDetectionStrategy.native_single_pass:
        return _make_native_single_pass_method(native_client=native_client)
    if strategy == ExperimentalDetectionStrategy.native_single_pass_recall:
        return _make_native_single_pass_method(native_client=native_client, recall_prompt=True)
    if strategy == ExperimentalDetectionStrategy.native_single_pass_values:
        return _make_native_single_pass_method(native_client=native_client, value_only_prompt=True)
    if strategy == ExperimentalDetectionStrategy.native_single_pass_values_recall:
        return _make_native_single_pass_method(
            native_client=native_client,
            recall_prompt=True,
            value_only_prompt=True,
        )
    raise ValueError(f"unsupported experimental detection strategy: {strategy}")


def _make_default_compact_validation_method(original: _DetectAndValidate) -> _DetectAndValidate:
    def detect_and_validate_entities(
        self: dw.EntityDetectionWorkflow,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: DetectionModelSelection,
        gliner_detection_threshold: float,
        validation_max_entities_per_call: int = dw._DEFAULT_VALIDATION_MAX_ENTITIES_PER_CALL,
        validation_excerpt_window_chars: int = dw._DEFAULT_VALIDATION_EXCERPT_WINDOW_CHARS,
        entity_labels: list[str] | None = None,
        data_summary: str | None = None,
        preview_num_records: int | None = None,
    ) -> dw.EntityDetectionResult:
        return original(
            self,
            dataframe,
            model_configs=model_configs,
            selected_models=selected_models,
            gliner_detection_threshold=gliner_detection_threshold,
            validation_max_entities_per_call=validation_max_entities_per_call,
            validation_excerpt_window_chars=validation_excerpt_window_chars,
            validation_single_chunk_full_text=False,
            entity_labels=entity_labels,
            data_summary=data_summary,
            preview_num_records=preview_num_records,
        )

    return detect_and_validate_entities


def _make_default_with_rule_guardrail_method(
    original: _DetectAndValidate,
    *,
    rule_labels: list[str] | None = None,
    compact_validation: bool = False,
) -> _DetectAndValidate:
    def detect_and_validate_entities(
        self: dw.EntityDetectionWorkflow,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: DetectionModelSelection,
        gliner_detection_threshold: float,
        validation_max_entities_per_call: int = dw._DEFAULT_VALIDATION_MAX_ENTITIES_PER_CALL,
        validation_excerpt_window_chars: int = dw._DEFAULT_VALIDATION_EXCERPT_WINDOW_CHARS,
        entity_labels: list[str] | None = None,
        data_summary: str | None = None,
        preview_num_records: int | None = None,
    ) -> dw.EntityDetectionResult:
        result = original(
            self,
            dataframe,
            model_configs=model_configs,
            selected_models=selected_models,
            gliner_detection_threshold=gliner_detection_threshold,
            validation_max_entities_per_call=validation_max_entities_per_call,
            validation_excerpt_window_chars=validation_excerpt_window_chars,
            validation_single_chunk_full_text=not compact_validation,
            entity_labels=entity_labels,
            data_summary=data_summary,
            preview_num_records=preview_num_records,
        )
        output = _apply_rule_guardrail(
            result.dataframe.copy(),
            labels=_rule_labels_for_detection(entity_labels, extra_rule_labels=rule_labels),
        )
        return dw.EntityDetectionResult(dataframe=output, failed_records=result.failed_records)

    return detect_and_validate_entities


def _make_validated_augmented_rule_filter_guardrail_method(
    *,
    rule_labels: list[str] | None = None,
) -> _DetectAndValidate:
    def detect_and_validate_entities(
        self: dw.EntityDetectionWorkflow,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: DetectionModelSelection,
        gliner_detection_threshold: float,
        validation_max_entities_per_call: int = dw._DEFAULT_VALIDATION_MAX_ENTITIES_PER_CALL,
        validation_excerpt_window_chars: int = dw._DEFAULT_VALIDATION_EXCERPT_WINDOW_CHARS,
        entity_labels: list[str] | None = None,
        data_summary: str | None = None,
        preview_num_records: int | None = None,
    ) -> dw.EntityDetectionResult:
        return _run_validated_augmented_rule_filter_guardrail_detection(
            self,
            dataframe,
            model_configs=model_configs,
            selected_models=selected_models,
            gliner_detection_threshold=gliner_detection_threshold,
            preview_num_records=preview_num_records,
            validation_max_entities_per_call=validation_max_entities_per_call,
            validation_excerpt_window_chars=validation_excerpt_window_chars,
            entity_labels=entity_labels,
            data_summary=data_summary,
            rule_labels=rule_labels,
        )

    return detect_and_validate_entities


def _run_validated_augmented_rule_filter_guardrail_detection(
    workflow: dw.EntityDetectionWorkflow,
    dataframe: pd.DataFrame,
    *,
    model_configs: list[ModelConfig],
    selected_models: DetectionModelSelection,
    gliner_detection_threshold: float,
    preview_num_records: int | None,
    validation_max_entities_per_call: int,
    validation_excerpt_window_chars: int,
    entity_labels: list[str] | None,
    data_summary: str | None,
    rule_labels: list[str] | None,
) -> dw.EntityDetectionResult:
    labels = dw._resolve_detection_labels(entity_labels)
    workflow_model_configs = workflow._inject_detector_params(
        model_configs=model_configs,
        selected_models=selected_models,
        labels=labels,
        gliner_detection_threshold=gliner_detection_threshold,
    )
    detection_result = workflow._adapter.run_workflow(
        dataframe,
        model_configs=workflow_model_configs,
        columns=_validated_augmented_rule_filter_guardrail_columns(
            selected_models=selected_models,
            labels=labels,
            data_summary=data_summary,
            validation_max_entities_per_call=validation_max_entities_per_call,
            validation_excerpt_window_chars=validation_excerpt_window_chars,
            strict_labels=entity_labels is not None,
            rule_labels=rule_labels,
        ),
        workflow_name="entity-detection-rules-filter-guardrail",
        preview_num_records=preview_num_records,
    )
    return dw.EntityDetectionResult(
        dataframe=detection_result.dataframe.copy(),
        failed_records=detection_result.failed_records,
    )


def _validated_augmented_rule_filter_guardrail_columns(
    *,
    selected_models: DetectionModelSelection,
    labels: list[str],
    data_summary: str | None,
    validation_max_entities_per_call: int,
    validation_excerpt_window_chars: int,
    strict_labels: bool,
    rule_labels: list[str] | None,
) -> list[LLMTextColumnConfig | LLMStructuredColumnConfig | CustomColumnConfig]:
    validator_params = _validator_params(
        selected_models=selected_models,
        labels=labels,
        data_summary=data_summary,
        validation_max_entities_per_call=validation_max_entities_per_call,
        validation_excerpt_window_chars=validation_excerpt_window_chars,
    )
    rule_detection_labels = _rule_labels_for_detection(labels, extra_rule_labels=rule_labels)
    return [
        LLMTextColumnConfig(
            name=COL_RAW_DETECTED,
            prompt=_jinja(COL_TEXT),
            model_alias=_detector_alias(selected_models),
        ),
        CustomColumnConfig(
            name=COL_SEED_ENTITIES,
            generator_function=_make_parse_detected_entities_filtering_rules(rule_detection_labels),
        ),
        CustomColumnConfig(name=COL_SEED_VALIDATION_CANDIDATES, generator_function=prepare_validation_inputs),
        _validation_decisions_column(selected_models, validator_params),
        CustomColumnConfig(name=COL_VALIDATED_ENTITIES, generator_function=enrich_validation_decisions),
        CustomColumnConfig(
            name=COL_SEED_ENTITIES_JSON,
            generator_function=_make_apply_validation_to_seed_entities_with_additive_rule_guardrail(
                rule_detection_labels
            ),
        ),
        LLMStructuredColumnConfig(
            name=COL_AUGMENTED_ENTITIES,
            prompt=dw._get_augment_prompt(data_summary=data_summary, labels=labels, strict_labels=strict_labels),
            model_alias=resolve_model_alias("entity_augmenter", selected_models),
            output_format=AugmentedEntitiesSchema,
        ),
        CustomColumnConfig(name=COL_MERGED_ENTITIES, generator_function=merge_and_build_candidates),
        CustomColumnConfig(
            name=COL_DETECTED_ENTITIES,
            generator_function=_make_apply_validation_and_finalize_with_additive_rule_guardrail(rule_detection_labels),
        ),
    ]


def _rule_labels_for_detection(
    entity_labels: list[str] | None,
    *,
    extra_rule_labels: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    labels = set(dw._resolve_detection_labels(entity_labels))
    labels.update(extra_rule_labels or [])
    return sorted(labels)


def _apply_rule_guardrail(dataframe: pd.DataFrame, *, labels: list[str]) -> pd.DataFrame:
    if COL_TEXT not in dataframe.columns or COL_DETECTED_ENTITIES not in dataframe.columns:
        return dataframe
    dataframe[COL_DETECTED_ENTITIES] = dataframe[COL_DETECTED_ENTITIES].astype("object")
    if COL_TAGGED_TEXT in dataframe.columns:
        dataframe[COL_TAGGED_TEXT] = dataframe[COL_TAGGED_TEXT].astype("object")
    for index, row in dataframe.iterrows():
        guarded = _guarded_entities(
            text=str(row.get(COL_TEXT, "")), raw_entities=row.get(COL_DETECTED_ENTITIES), labels=labels
        )
        dataframe.at[index, COL_DETECTED_ENTITIES] = EntitiesSchema(
            entities=[entity.as_dict() for entity in guarded]
        ).model_dump(mode="json")
        if COL_TAGGED_TEXT in dataframe.columns:
            dataframe.at[index, COL_TAGGED_TEXT] = build_tagged_text(text=str(row.get(COL_TEXT, "")), entities=guarded)
    return dataframe


def _guarded_entities(*, text: str, raw_entities: object, labels: list[str]) -> list[EntitySpan]:
    final_spans = _entity_spans_from_payload(raw_entities)
    rule_spans = detect_high_confidence_entities(text, labels=labels)
    return _merge_rule_guardrail_spans(final_spans, rule_spans)


def _merge_rule_guardrail_spans(final_spans: list[EntitySpan], rule_spans: list[EntitySpan]) -> list[EntitySpan]:
    filtered_final = [
        entity
        for entity in final_spans
        if not any(
            rule.start_position == entity.start_position
            and rule.end_position == entity.end_position
            and rule.label != entity.label
            for rule in rule_spans
        )
    ]
    return resolve_overlaps([*filtered_final, *rule_spans])


def _make_validated_no_augment_method(
    *,
    include_rules: bool,
    final_rule_guardrail: bool = False,
    filter_rule_overlaps: bool = False,
    rule_labels: list[str] | None = None,
) -> _DetectAndValidate:
    options = _NoAugmentOptions(
        include_rules=include_rules,
        final_rule_guardrail=final_rule_guardrail,
        filter_rule_overlaps=filter_rule_overlaps,
        rule_labels=tuple(rule_labels or ()),
    )

    def detect_and_validate_entities(
        self: dw.EntityDetectionWorkflow,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: DetectionModelSelection,
        gliner_detection_threshold: float,
        validation_max_entities_per_call: int = dw._DEFAULT_VALIDATION_MAX_ENTITIES_PER_CALL,
        validation_excerpt_window_chars: int = dw._DEFAULT_VALIDATION_EXCERPT_WINDOW_CHARS,
        entity_labels: list[str] | None = None,
        data_summary: str | None = None,
        preview_num_records: int | None = None,
    ) -> dw.EntityDetectionResult:
        return _run_validated_no_augment_detection(
            self,
            dataframe,
            model_configs=model_configs,
            selected_models=selected_models,
            gliner_detection_threshold=gliner_detection_threshold,
            preview_num_records=preview_num_records,
            validation_max_entities_per_call=validation_max_entities_per_call,
            validation_excerpt_window_chars=validation_excerpt_window_chars,
            entity_labels=entity_labels,
            data_summary=data_summary,
            options=options,
        )

    return detect_and_validate_entities


def _run_validated_no_augment_detection(
    workflow: dw.EntityDetectionWorkflow,
    dataframe: pd.DataFrame,
    *,
    model_configs: list[ModelConfig],
    selected_models: DetectionModelSelection,
    gliner_detection_threshold: float,
    preview_num_records: int | None,
    validation_max_entities_per_call: int,
    validation_excerpt_window_chars: int,
    entity_labels: list[str] | None,
    data_summary: str | None,
    options: _NoAugmentOptions,
) -> dw.EntityDetectionResult:
    labels = dw._resolve_detection_labels(entity_labels)
    workflow_model_configs = workflow._inject_detector_params(
        model_configs=model_configs,
        selected_models=selected_models,
        labels=labels,
        gliner_detection_threshold=gliner_detection_threshold,
    )
    detection_result = workflow._adapter.run_workflow(
        dataframe,
        model_configs=workflow_model_configs,
        columns=_validated_no_augment_columns(
            selected_models=selected_models,
            labels=labels,
            data_summary=data_summary,
            validation_max_entities_per_call=validation_max_entities_per_call,
            validation_excerpt_window_chars=validation_excerpt_window_chars,
            options=options,
        ),
        workflow_name=_workflow_name_for_no_augment(options),
        preview_num_records=preview_num_records,
    )
    return dw.EntityDetectionResult(
        dataframe=detection_result.dataframe.copy(),
        failed_records=detection_result.failed_records,
    )


def _validated_no_augment_columns(
    *,
    selected_models: DetectionModelSelection,
    labels: list[str],
    data_summary: str | None,
    validation_max_entities_per_call: int,
    validation_excerpt_window_chars: int,
    options: _NoAugmentOptions,
) -> list[LLMTextColumnConfig | CustomColumnConfig]:
    validator_params = _validator_params(
        selected_models=selected_models,
        labels=labels,
        data_summary=data_summary,
        validation_max_entities_per_call=validation_max_entities_per_call,
        validation_excerpt_window_chars=validation_excerpt_window_chars,
    )
    parse_generator = _parse_generator(
        labels=_rule_labels_for_detection(labels, extra_rule_labels=options.rule_labels),
        include_rules=options.include_rules,
        filter_rule_overlaps=options.filter_rule_overlaps,
    )
    return [
        LLMTextColumnConfig(
            name=COL_RAW_DETECTED, prompt=_jinja(COL_TEXT), model_alias=_detector_alias(selected_models)
        ),
        CustomColumnConfig(name=COL_SEED_ENTITIES, generator_function=parse_generator),
        CustomColumnConfig(name=COL_SEED_VALIDATION_CANDIDATES, generator_function=prepare_validation_inputs),
        _validation_decisions_column(selected_models, validator_params),
        CustomColumnConfig(name=COL_VALIDATED_ENTITIES, generator_function=enrich_validation_decisions),
        CustomColumnConfig(name=COL_SEED_ENTITIES_JSON, generator_function=apply_validation_to_seed_entities),
        CustomColumnConfig(name=COL_AUGMENTED_ENTITIES, generator_function=_empty_augmentation),
        CustomColumnConfig(name=COL_MERGED_ENTITIES, generator_function=merge_and_build_candidates),
        CustomColumnConfig(
            name=COL_DETECTED_ENTITIES,
            generator_function=_finalizer(
                _rule_labels_for_detection(labels, extra_rule_labels=options.rule_labels), options
            ),
        ),
    ]


def _validation_decisions_column(
    selected_models: DetectionModelSelection,
    validator_params: ChunkedValidationParams,
) -> CustomColumnConfig:
    return CustomColumnConfig(
        name=COL_VALIDATION_DECISIONS,
        generator_function=make_chunked_validation_generator(
            resolve_model_aliases("entity_validator", selected_models)
        ),
        generator_params=validator_params,
        drop=True,
    )


def _finalizer(labels: list[str], options: _NoAugmentOptions) -> Callable[[dict[str, Any]], dict[str, Any]]:
    if options.filter_rule_overlaps:
        return _make_apply_validation_and_finalize_with_additive_rule_guardrail(labels)
    if options.final_rule_guardrail:
        return _make_apply_validation_and_finalize_with_rule_guardrail(labels)
    return apply_validation_and_finalize


def _workflow_name_for_no_augment(options: _NoAugmentOptions) -> str:
    if options.filter_rule_overlaps:
        return "entity-detection-rules-filter-guardrail-no-augment"
    if options.final_rule_guardrail:
        return "entity-detection-rules-guardrail-no-augment"
    if options.include_rules:
        return "entity-detection-rules-no-augment"
    return "entity-detection-no-augment"


def _validator_params(
    *,
    selected_models: DetectionModelSelection,
    labels: list[str],
    data_summary: str | None,
    validation_max_entities_per_call: int,
    validation_excerpt_window_chars: int,
) -> ChunkedValidationParams:
    validator_aliases = resolve_model_aliases("entity_validator", selected_models)
    return ChunkedValidationParams(
        pool=list(validator_aliases),
        max_entities_per_call=validation_max_entities_per_call,
        excerpt_window_chars=validation_excerpt_window_chars,
        prompt_template=dw._get_validation_prompt(data_summary=data_summary, labels=labels),
    )


def _detector_alias(selected_models: DetectionModelSelection) -> str:
    return resolve_model_alias("entity_detector", selected_models)


def _detect_with_detector_only(
    self: dw.EntityDetectionWorkflow,
    dataframe: pd.DataFrame,
    *,
    model_configs: list[ModelConfig],
    selected_models: DetectionModelSelection,
    gliner_detection_threshold: float,
    validation_max_entities_per_call: int = dw._DEFAULT_VALIDATION_MAX_ENTITIES_PER_CALL,
    validation_excerpt_window_chars: int = dw._DEFAULT_VALIDATION_EXCERPT_WINDOW_CHARS,
    validation_single_chunk_full_text: bool = True,
    entity_labels: list[str] | None = None,
    data_summary: str | None = None,
    preview_num_records: int | None = None,
) -> dw.EntityDetectionResult:
    return _run_detector_only_detection(
        self,
        dataframe,
        model_configs=model_configs,
        selected_models=selected_models,
        gliner_detection_threshold=gliner_detection_threshold,
        entity_labels=entity_labels,
        preview_num_records=preview_num_records,
        rule_labels=None,
        workflow_name="entity-detection-detector-only",
    )


def _make_detector_only_with_rule_guardrail_method(rule_labels: list[str] | None) -> _DetectAndValidate:
    def detect_and_validate_entities(
        self: dw.EntityDetectionWorkflow,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: DetectionModelSelection,
        gliner_detection_threshold: float,
        validation_max_entities_per_call: int = dw._DEFAULT_VALIDATION_MAX_ENTITIES_PER_CALL,
        validation_excerpt_window_chars: int = dw._DEFAULT_VALIDATION_EXCERPT_WINDOW_CHARS,
        validation_single_chunk_full_text: bool = True,
        entity_labels: list[str] | None = None,
        data_summary: str | None = None,
        preview_num_records: int | None = None,
    ) -> dw.EntityDetectionResult:
        return _run_detector_only_detection(
            self,
            dataframe,
            model_configs=model_configs,
            selected_models=selected_models,
            gliner_detection_threshold=gliner_detection_threshold,
            entity_labels=entity_labels,
            preview_num_records=preview_num_records,
            rule_labels=_rule_labels_for_detection(entity_labels, extra_rule_labels=rule_labels),
            workflow_name="entity-detection-rules-guardrail-detector-only",
        )

    return detect_and_validate_entities


def _run_detector_only_detection(
    workflow: dw.EntityDetectionWorkflow,
    dataframe: pd.DataFrame,
    *,
    model_configs: list[ModelConfig],
    selected_models: DetectionModelSelection,
    gliner_detection_threshold: float,
    entity_labels: list[str] | None,
    preview_num_records: int | None,
    rule_labels: list[str] | None,
    workflow_name: str,
) -> dw.EntityDetectionResult:
    labels = dw._resolve_detection_labels(entity_labels)
    workflow_model_configs = workflow._inject_detector_params(
        model_configs=model_configs,
        selected_models=selected_models,
        labels=labels,
        gliner_detection_threshold=gliner_detection_threshold,
    )
    detection_result = workflow._adapter.run_workflow(
        dataframe,
        model_configs=workflow_model_configs,
        columns=_detector_only_columns(selected_models, rule_labels=rule_labels),
        workflow_name=workflow_name,
        preview_num_records=preview_num_records,
    )
    return dw.EntityDetectionResult(
        dataframe=detection_result.dataframe.copy(),
        failed_records=detection_result.failed_records,
    )


def _detector_only_columns(
    selected_models: DetectionModelSelection,
    *,
    rule_labels: list[str] | None,
) -> list[LLMTextColumnConfig | CustomColumnConfig]:
    return [
        LLMTextColumnConfig(
            name=COL_RAW_DETECTED,
            prompt=_jinja(COL_TEXT),
            model_alias=_detector_alias(selected_models),
        ),
        CustomColumnConfig(name=COL_SEED_ENTITIES, generator_function=parse_detected_entities),
        CustomColumnConfig(name=COL_SEED_ENTITIES_JSON, generator_function=_copy_seed_entities_json),
        CustomColumnConfig(name=COL_DETECTED_ENTITIES, generator_function=_detector_only_finalizer(rule_labels)),
    ]


def _detector_only_finalizer(rule_labels: list[str] | None) -> Callable[[dict[str, Any]], dict[str, Any]]:
    if rule_labels is None:
        return _finalize_detector_only
    return _make_finalize_detector_only_with_rule_guardrail(rule_labels)


@custom_column_generator(required_columns=[COL_SEED_ENTITIES])
def _copy_seed_entities_json(row: dict[str, Any]) -> dict[str, Any]:
    row[COL_SEED_ENTITIES_JSON] = json.dumps(
        [span.as_dict() for span in _entity_spans_from_payload(row.get(COL_SEED_ENTITIES, {}))]
    )
    return row


@custom_column_generator(
    required_columns=[COL_TEXT, COL_SEED_ENTITIES],
    side_effect_columns=[COL_TAGGED_TEXT],
)
def _finalize_detector_only(row: dict[str, Any]) -> dict[str, Any]:
    text = str(row.get(COL_TEXT, ""))
    spans = expand_entity_occurrences(text=text, entities=_entity_spans_from_payload(row.get(COL_SEED_ENTITIES, {})))
    row[COL_DETECTED_ENTITIES] = EntitiesSchema(entities=[span.as_dict() for span in spans]).model_dump(mode="json")
    row[COL_TAGGED_TEXT] = build_tagged_text(text=text, entities=spans)
    return row


def _make_finalize_detector_only_with_rule_guardrail(labels: list[str]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    @custom_column_generator(
        required_columns=[COL_TEXT, COL_SEED_ENTITIES],
        side_effect_columns=[COL_TAGGED_TEXT],
    )
    def finalize_detector_only_with_rule_guardrail(row: dict[str, Any]) -> dict[str, Any]:
        row = _finalize_detector_only(row)
        text = str(row.get(COL_TEXT, ""))
        final_spans = _entity_spans_from_payload(row.get(COL_DETECTED_ENTITIES, {}))
        rule_spans = detect_high_confidence_entities(text, labels=labels)
        guarded = _merge_rule_guardrail_spans(final_spans, rule_spans)
        row[COL_DETECTED_ENTITIES] = EntitiesSchema(entities=[span.as_dict() for span in guarded]).model_dump(
            mode="json"
        )
        row[COL_TAGGED_TEXT] = build_tagged_text(text=text, entities=guarded)
        return row

    return finalize_detector_only_with_rule_guardrail


@custom_column_generator(required_columns=[COL_TEXT])
def _empty_augmentation(row: dict[str, Any]) -> dict[str, Any]:
    row[COL_AUGMENTED_ENTITIES] = AugmentedEntitiesSchema().model_dump(mode="json")
    return row


def _parse_generator(
    *,
    labels: list[str],
    include_rules: bool,
    filter_rule_overlaps: bool,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    if filter_rule_overlaps:
        return _make_parse_detected_entities_filtering_rules(labels)
    if include_rules:
        return _make_parse_detected_entities_with_rules(labels)
    return parse_detected_entities


def _make_parse_detected_entities_with_rules(labels: list[str]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    @custom_column_generator(
        required_columns=[COL_TEXT, COL_RAW_DETECTED],
        side_effect_columns=[COL_TAG_NOTATION],
    )
    def parse_detected_entities_with_rules(row: dict[str, Any]) -> dict[str, Any]:
        text = str(row.get(COL_TEXT, ""))
        detected = parse_raw_entities(raw_response=str(row.get(COL_RAW_DETECTED, "")), text=text)
        rule_spans = detect_high_confidence_entities(text, labels=labels)
        row[COL_SEED_ENTITIES] = EntitiesSchema(
            entities=[entity.as_dict() for entity in resolve_overlaps([*detected, *rule_spans])]
        ).model_dump(mode="json")
        row[COL_TAG_NOTATION] = get_tag_notation(text=text)
        return row

    return parse_detected_entities_with_rules


def _make_parse_detected_entities_filtering_rules(labels: list[str]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    @custom_column_generator(
        required_columns=[COL_TEXT, COL_RAW_DETECTED],
        side_effect_columns=[COL_TAG_NOTATION],
    )
    def parse_detected_entities_filtering_rules(row: dict[str, Any]) -> dict[str, Any]:
        text = str(row.get(COL_TEXT, ""))
        detected = parse_raw_entities(raw_response=str(row.get(COL_RAW_DETECTED, "")), text=text)
        rule_spans = detect_high_confidence_entities(text, labels=labels)
        filtered = [entity for entity in detected if not _is_rule_covered_detector_span(entity, rule_spans)]
        row[COL_SEED_ENTITIES] = EntitiesSchema(
            entities=[entity.as_dict() for entity in resolve_overlaps(filtered)]
        ).model_dump(mode="json")
        row[COL_TAG_NOTATION] = get_tag_notation(text=text)
        return row

    return parse_detected_entities_filtering_rules


def _make_apply_validation_and_finalize_with_rule_guardrail(
    labels: list[str],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    @custom_column_generator(
        required_columns=[COL_TEXT, COL_MERGED_ENTITIES, COL_VALIDATED_ENTITIES],
        side_effect_columns=[COL_TAGGED_TEXT],
    )
    def apply_validation_and_finalize_with_rule_guardrail(row: dict[str, Any]) -> dict[str, Any]:
        row = apply_validation_and_finalize(row)
        text = str(row.get(COL_TEXT, ""))
        final_spans = _entity_spans_from_payload(row.get(COL_DETECTED_ENTITIES, {}))
        rule_spans = detect_high_confidence_entities(text, labels=labels)
        guarded = _merge_rule_guardrail_spans(final_spans, rule_spans)
        row[COL_DETECTED_ENTITIES] = EntitiesSchema(entities=[entity.as_dict() for entity in guarded]).model_dump(
            mode="json"
        )
        row[COL_TAGGED_TEXT] = build_tagged_text(text=text, entities=guarded)
        return row

    return apply_validation_and_finalize_with_rule_guardrail


def _make_apply_validation_and_finalize_with_additive_rule_guardrail(
    labels: list[str],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    @custom_column_generator(
        required_columns=[COL_TEXT, COL_MERGED_ENTITIES, COL_VALIDATED_ENTITIES],
        side_effect_columns=[COL_TAGGED_TEXT],
    )
    def apply_validation_and_finalize_with_additive_rule_guardrail(row: dict[str, Any]) -> dict[str, Any]:
        row = apply_validation_and_finalize(row)
        text = str(row.get(COL_TEXT, ""))
        final_spans = _entity_spans_from_payload(row.get(COL_DETECTED_ENTITIES, {}))
        rule_spans = detect_high_confidence_entities(text, labels=labels)
        guarded = _add_non_overlapping_rule_spans(final_spans, rule_spans)
        row[COL_DETECTED_ENTITIES] = EntitiesSchema(entities=[entity.as_dict() for entity in guarded]).model_dump(
            mode="json"
        )
        row[COL_TAGGED_TEXT] = build_tagged_text(text=text, entities=guarded)
        return row

    return apply_validation_and_finalize_with_additive_rule_guardrail


def _make_apply_validation_to_seed_entities_with_rule_guardrail(
    labels: list[str],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    @custom_column_generator(
        required_columns=[COL_TEXT, COL_SEED_ENTITIES, COL_VALIDATED_ENTITIES],
        side_effect_columns=[COL_INITIAL_TAGGED_TEXT, COL_SEED_ENTITIES_JSON, COL_VALIDATED_SEED_ENTITIES],
    )
    def apply_validation_to_seed_entities_with_rule_guardrail(row: dict[str, Any]) -> dict[str, Any]:
        row = apply_validation_to_seed_entities(row)
        text = str(row.get(COL_TEXT, ""))
        validated_seed = _entity_spans_from_payload(row.get(COL_VALIDATED_SEED_ENTITIES, {}))
        rule_spans = detect_high_confidence_entities(text, labels=labels)
        guarded = _merge_rule_guardrail_spans(validated_seed, rule_spans)
        seed_entities = [entity.as_dict() for entity in guarded]
        row[COL_VALIDATED_SEED_ENTITIES] = EntitiesSchema(entities=seed_entities).model_dump(mode="json")
        row[COL_SEED_ENTITIES_JSON] = json.dumps(seed_entities)
        row[COL_INITIAL_TAGGED_TEXT] = build_tagged_text(text=text, entities=guarded)
        return row

    return apply_validation_to_seed_entities_with_rule_guardrail


def _make_apply_validation_to_seed_entities_with_additive_rule_guardrail(
    labels: list[str],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    @custom_column_generator(
        required_columns=[COL_TEXT, COL_SEED_ENTITIES, COL_VALIDATED_ENTITIES],
        side_effect_columns=[COL_INITIAL_TAGGED_TEXT, COL_SEED_ENTITIES_JSON, COL_VALIDATED_SEED_ENTITIES],
    )
    def apply_validation_to_seed_entities_with_additive_rule_guardrail(row: dict[str, Any]) -> dict[str, Any]:
        row = apply_validation_to_seed_entities(row)
        text = str(row.get(COL_TEXT, ""))
        validated_seed = _entity_spans_from_payload(row.get(COL_VALIDATED_SEED_ENTITIES, {}))
        rule_spans = detect_high_confidence_entities(text, labels=labels)
        guarded = _add_non_overlapping_rule_spans(validated_seed, rule_spans)
        seed_entities = [entity.as_dict() for entity in guarded]
        row[COL_VALIDATED_SEED_ENTITIES] = EntitiesSchema(entities=seed_entities).model_dump(mode="json")
        row[COL_SEED_ENTITIES_JSON] = json.dumps(seed_entities)
        row[COL_INITIAL_TAGGED_TEXT] = build_tagged_text(text=text, entities=guarded)
        return row

    return apply_validation_to_seed_entities_with_additive_rule_guardrail


def _is_rule_covered_detector_span(entity: EntitySpan, spans: list[EntitySpan]) -> bool:
    return any(
        entity.label == span.label
        and span.start_position <= entity.start_position
        and span.end_position >= entity.end_position
        for span in spans
    )


def _add_non_overlapping_rule_spans(
    existing_spans: list[EntitySpan],
    rule_spans: list[EntitySpan],
) -> list[EntitySpan]:
    additions = [rule for rule in rule_spans if not any(_spans_overlap(rule, existing) for existing in existing_spans)]
    return resolve_overlaps([*existing_spans, *additions])


def _spans_overlap(left: EntitySpan, right: EntitySpan) -> bool:
    return max(left.start_position, right.start_position) < min(left.end_position, right.end_position)


def _entity_spans_from_payload(raw_payload: object) -> list[EntitySpan]:
    return [
        EntitySpan(
            entity_id=entity.id,
            value=entity.value,
            label=entity.label,
            start_position=entity.start_position,
            end_position=entity.end_position,
            score=entity.score,
            source=entity.source,
        )
        for entity in EntitiesSchema.from_raw(raw_payload).entities
    ]


def _make_native_single_pass_method(
    native_client: DirectDetectionClient | None,
    *,
    recall_prompt: bool = False,
    value_only_prompt: bool = False,
) -> _DetectAndValidate:
    def detect_and_validate_entities(
        self: dw.EntityDetectionWorkflow,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: DetectionModelSelection,
        gliner_detection_threshold: float,
        validation_max_entities_per_call: int = dw._DEFAULT_VALIDATION_MAX_ENTITIES_PER_CALL,
        validation_excerpt_window_chars: int = dw._DEFAULT_VALIDATION_EXCERPT_WINDOW_CHARS,
        validation_single_chunk_full_text: bool = True,
        entity_labels: list[str] | None = None,
        data_summary: str | None = None,
        preview_num_records: int | None = None,
    ) -> dw.EntityDetectionResult:
        labels = dw._resolve_detection_labels(entity_labels)
        client = native_client or HttpxDirectDetectionClient()
        return _run_native_single_pass_detection(
            dataframe,
            labels=labels,
            client=client,
            data_summary=data_summary,
            preview_num_records=preview_num_records,
            recall_prompt=recall_prompt,
            value_only_prompt=value_only_prompt,
        )

    return detect_and_validate_entities


def _run_native_single_pass_detection(
    dataframe: pd.DataFrame,
    *,
    labels: list[str],
    client: DirectDetectionClient,
    data_summary: str | None,
    preview_num_records: int | None,
    recall_prompt: bool,
    value_only_prompt: bool,
) -> dw.EntityDetectionResult:
    source_df = dataframe.iloc[:preview_num_records].copy() if preview_num_records is not None else dataframe.copy()
    output_rows: list[dict[str, Any]] = []
    output_indices: list[Any] = []
    failed_records: list[FailedRecord] = []

    for index, row in source_df.iterrows():
        output_row, failed_record = _execute_native_single_pass_row(
            row,
            index=index,
            labels=labels,
            client=client,
            data_summary=data_summary,
            recall_prompt=recall_prompt,
            value_only_prompt=value_only_prompt,
        )
        if failed_record is not None:
            failed_records.append(failed_record)
            continue
        if output_row is None:
            failed_records.append(_native_single_pass_failed_record(index, error="native single-pass produced no row"))
            continue
        output_rows.append(output_row)
        output_indices.append(index)

    return dw.EntityDetectionResult(
        dataframe=_native_output_dataframe(source_df, output_rows=output_rows, output_indices=output_indices),
        failed_records=failed_records,
    )


def _execute_native_single_pass_row(
    row: pd.Series,
    *,
    index: object,
    labels: list[str],
    client: DirectDetectionClient,
    data_summary: str | None,
    recall_prompt: bool,
    value_only_prompt: bool,
) -> tuple[dict[str, Any] | None, FailedRecord | None]:
    text = str(row.get(COL_TEXT, ""))
    started = time.perf_counter()
    try:
        completion = _complete_native_single_pass(
            text=text,
            labels=labels,
            client=client,
            data_summary=data_summary,
            recall_prompt=recall_prompt,
            value_only_prompt=value_only_prompt,
        )
    except Exception as exc:  # noqa: BLE001 - benchmark experiment records case-local failures
        _record_native_single_pass_request_error(elapsed_sec=time.perf_counter() - started)
        return None, _native_single_pass_failed_record(index, error=f"{type(exc).__name__}: {exc}")
    try:
        spans = _native_single_pass_spans(text, completion.content, labels=labels)
    except Exception as exc:  # noqa: BLE001 - parser fragility is a benchmark failure mode
        _record_native_single_pass_completion(completion, status="error", output_row_count=0, failed_record_count=1)
        return None, _native_single_pass_failed_record(index, error=f"{type(exc).__name__}: {exc}")
    _record_native_single_pass_completion(completion, status="completed", output_row_count=1, failed_record_count=0)
    return _native_single_pass_result_row(row, spans=spans, labels=labels), None


def _complete_native_single_pass(
    *,
    text: str,
    labels: list[str],
    client: DirectDetectionClient,
    data_summary: str | None,
    recall_prompt: bool,
    value_only_prompt: bool,
) -> Any:
    return client.complete(
        DirectGenerationRequest(
            endpoint=_NATIVE_DIRECT_ENDPOINT,
            model=_NATIVE_DIRECT_MODEL_NAME,
            prompt=_native_single_pass_prompt(
                text=text,
                labels=labels,
                data_summary=data_summary,
                recall_prompt=recall_prompt,
                value_only_prompt=value_only_prompt,
            ),
            max_tokens=_NATIVE_DIRECT_MAX_TOKENS,
            timeout_sec=_NATIVE_DIRECT_TIMEOUT_SEC,
        )
    )


def _native_single_pass_prompt(
    *,
    text: str,
    labels: list[str],
    data_summary: str | None,
    recall_prompt: bool = False,
    value_only_prompt: bool = False,
) -> str:
    if value_only_prompt:
        return build_direct_prompt(
            DirectDetectionRequest(
                case_id="native-single-pass-values",
                text=text,
                labels=labels,
                prompt_mode=PromptMode.recall if recall_prompt else PromptMode.compact,
                data_summary=data_summary,
            )
        )

    label_text = dw._format_label_examples(labels) if recall_prompt else ", ".join(labels)
    recall_block = _native_single_pass_recall_block() if recall_prompt else ""
    summary = f"\nData context: {data_summary}\n" if data_summary else ""
    return f"""Extract privacy-sensitive entities from the input text in one pass.
{summary}
Use only these labels:
{label_text}

Rules:
- Return exact substrings from the input text.
- Do not invent values.
- `start` and `end` must be zero-based Python character offsets, where text[start:end] equals `value`.
- Missing a sensitive value is worse than returning one extra plausible value.
- Skip generic nouns, syntax, and non-sensitive filler.
{recall_block}- Return only a JSON object with this shape:
  {{"entities": [{{"value": "exact substring", "label": "one_allowed_label", "start": 0, "end": 0, "reason": "short reason"}}]}}

Input text:
---
{text}
---"""


def _native_single_pass_recall_block() -> str:
    return """- Bias toward high recall. Missing a sensitive value is worse than returning one extra plausible value.
- Include family members, colleagues, employers, schools, institutions, locations, dates, demographics, beliefs, and identifiers when allowed.
"""


def _native_single_pass_spans(text: str, content: str, *, labels: list[str]) -> list[EntitySpan]:
    payload = _load_embedded_json(content)
    if not isinstance(payload, dict):
        raise ValueError("native single-pass response must be a JSON object")
    entities = payload.get("entities")
    if not isinstance(entities, list):
        raise ValueError("native single-pass response must contain an entities list")
    spans: list[EntitySpan] = []
    allowed = set(labels)
    for item in entities:
        if not isinstance(item, dict):
            continue
        value = str(item.get("value", "")).strip()
        label = str(item.get("label", "")).strip()
        if not value or label not in allowed:
            continue
        offset_span = _native_single_pass_offset_span(text, item, value=value, label=label)
        if offset_span is not None:
            spans.append(offset_span)
            continue
        spans.extend(_native_single_pass_value_spans(text, value=value, label=label))
    return resolve_overlaps(spans)


def _native_single_pass_offset_span(
    text: str,
    item: dict[str, Any],
    *,
    value: str,
    label: str,
) -> EntitySpan | None:
    start = _coerce_native_offset(item.get("start"))
    end = _coerce_native_offset(item.get("end"))
    if start is None or end is None or start < 0 or end <= start or end > len(text):
        return None
    if text[start:end] != value:
        return None
    return _native_single_pass_span(value=value, label=label, start=start, end=end)


def _coerce_native_offset(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _native_single_pass_value_spans(text: str, *, value: str, label: str) -> list[EntitySpan]:
    spans: list[EntitySpan] = []
    start = 0
    while True:
        match_start = text.find(value, start)
        if match_start < 0:
            return spans
        match_end = match_start + len(value)
        spans.append(_native_single_pass_span(value=value, label=label, start=match_start, end=match_end))
        start = match_end


def _native_single_pass_span(*, value: str, label: str, start: int, end: int) -> EntitySpan:
    return EntitySpan(
        entity_id=f"{label}_{start}_{end}",
        value=value,
        label=label,
        start_position=start,
        end_position=end,
        score=1.0,
        source="direct_single_pass",
    )


def _native_single_pass_result_row(row: pd.Series, *, spans: list[EntitySpan], labels: list[str]) -> dict[str, Any]:
    text = str(row.get(COL_TEXT, ""))
    rule_spans = detect_high_confidence_entities(text, labels=labels)
    guarded = _add_non_overlapping_rule_spans(spans, rule_spans)
    output_row = row.to_dict()
    output_row[COL_DETECTED_ENTITIES] = EntitiesSchema(entities=[span.as_dict() for span in guarded]).model_dump(
        mode="json"
    )
    output_row[COL_TAGGED_TEXT] = build_tagged_text(text=text, entities=guarded)
    return output_row


def _native_single_pass_failed_record(index: object, *, error: str | None) -> FailedRecord:
    return FailedRecord(
        record_id=str(index),
        step="entity-detection-native-single-pass",
        reason=error or "native single-pass detection failed",
    )


def _record_native_single_pass_completion(
    completion: Any,
    *,
    status: str,
    output_row_count: int,
    failed_record_count: int,
) -> None:
    record_model_workflow(
        workflow_name="entity-detection-native-single-pass",
        model_aliases=[_NATIVE_DIRECT_MODEL_ALIAS],
        input_row_count=1,
        output_row_count=output_row_count,
        failed_record_count=failed_record_count,
        elapsed_sec=float(getattr(completion, "elapsed_sec", 0.0) or 0.0),
        status=status,
        model_usage=_native_single_pass_model_usage(
            successful_requests=1,
            failed_requests=0,
            usage=dict(getattr(completion, "usage", {}) or {}),
        ),
    )


def _record_native_single_pass_request_error(*, elapsed_sec: float) -> None:
    record_model_workflow(
        workflow_name="entity-detection-native-single-pass",
        model_aliases=[_NATIVE_DIRECT_MODEL_ALIAS],
        input_row_count=1,
        output_row_count=0,
        failed_record_count=1,
        elapsed_sec=elapsed_sec,
        status="error",
        model_usage=_native_single_pass_model_usage(successful_requests=0, failed_requests=1, usage={}),
    )


def _native_single_pass_model_usage(
    *,
    successful_requests: int,
    failed_requests: int,
    usage: dict[str, int],
) -> dict[str, dict[str, Any]]:
    total_requests = successful_requests + failed_requests
    return {
        _NATIVE_DIRECT_MODEL_ALIAS: {
            "model_alias": _NATIVE_DIRECT_MODEL_ALIAS,
            "model_name": _NATIVE_DIRECT_MODEL_NAME,
            "model_provider_name": _NATIVE_DIRECT_MODEL_PROVIDER,
            "request_usage": {
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "total_requests": total_requests,
            },
            "token_usage": _native_token_usage(usage),
        }
    }


def _make_native_rules_router_method(native_client: DirectDetectionClient | None) -> _DetectAndValidate:
    return _make_native_staged_method(
        native_client=native_client,
        gliner_seed_client=None,
        seed_source=SeedSource.rules_router,
        workflow_name="entity-detection-native-rules-router",
        skip_augmentation=False,
    )


def _make_native_candidate_validate_no_augment_method(
    native_client: DirectDetectionClient | None,
) -> _DetectAndValidate:
    return _make_native_staged_method(
        native_client=native_client,
        gliner_seed_client=None,
        seed_source=SeedSource.rules_plus_direct_llm,
        workflow_name="entity-detection-native-candidate-validate-no-augment",
        skip_augmentation=True,
    )


def _make_gliner_native_validate_no_augment_method(
    *,
    native_client: DirectDetectionClient | None,
    gliner_seed_client: GlinerSeedClient | None,
) -> _DetectAndValidate:
    return _make_native_staged_method(
        native_client=native_client,
        gliner_seed_client=gliner_seed_client,
        seed_source=SeedSource.gliner,
        workflow_name="entity-detection-gliner-native-validate-no-augment",
        skip_augmentation=True,
    )


def _make_gliner_native_validate_native_augment_method(
    *,
    native_client: DirectDetectionClient | None,
    gliner_seed_client: GlinerSeedClient | None,
) -> _DetectAndValidate:
    return _make_native_staged_method(
        native_client=native_client,
        gliner_seed_client=gliner_seed_client,
        seed_source=SeedSource.gliner,
        workflow_name="entity-detection-gliner-native-validate-native-augment",
        skip_augmentation=False,
    )


def _make_detector_native_validate_no_augment_method(
    native_client: DirectDetectionClient | None,
) -> _DetectAndValidate:
    return _make_detector_native_validate_method(
        native_client,
        workflow_name="entity-detection-detector-native-validate-no-augment",
        seed_workflow_name="entity-detection-detector-native-validate-no-augment-seed",
        skip_augmentation=True,
    )


def _make_detector_native_validate_native_augment_method(
    native_client: DirectDetectionClient | None,
) -> _DetectAndValidate:
    return _make_detector_native_validate_method(
        native_client,
        workflow_name="entity-detection-detector-native-validate-native-augment",
        seed_workflow_name="entity-detection-detector-native-validate-native-augment-seed",
        skip_augmentation=False,
    )


def _make_detector_native_validate_method(
    native_client: DirectDetectionClient | None,
    *,
    workflow_name: str,
    seed_workflow_name: str,
    skip_augmentation: bool,
) -> _DetectAndValidate:
    def detect_and_validate_entities(
        self: dw.EntityDetectionWorkflow,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: DetectionModelSelection,
        gliner_detection_threshold: float,
        validation_max_entities_per_call: int = dw._DEFAULT_VALIDATION_MAX_ENTITIES_PER_CALL,
        validation_excerpt_window_chars: int = dw._DEFAULT_VALIDATION_EXCERPT_WINDOW_CHARS,
        validation_single_chunk_full_text: bool = True,
        entity_labels: list[str] | None = None,
        data_summary: str | None = None,
        preview_num_records: int | None = None,
    ) -> dw.EntityDetectionResult:
        labels = dw._resolve_detection_labels(entity_labels)
        client = native_client or HttpxDirectDetectionClient()
        return _run_detector_native_validate_detection(
            self,
            dataframe,
            model_configs=model_configs,
            selected_models=selected_models,
            labels=labels,
            gliner_detection_threshold=gliner_detection_threshold,
            preview_num_records=preview_num_records,
            validation_max_entities_per_call=validation_max_entities_per_call,
            validation_excerpt_window_chars=validation_excerpt_window_chars,
            validation_single_chunk_full_text=validation_single_chunk_full_text,
            client=client,
            data_summary=data_summary,
            workflow_name=workflow_name,
            seed_workflow_name=seed_workflow_name,
            skip_augmentation=skip_augmentation,
        )

    return detect_and_validate_entities


def _run_detector_native_validate_detection(
    workflow: dw.EntityDetectionWorkflow,
    dataframe: pd.DataFrame,
    *,
    model_configs: list[ModelConfig],
    selected_models: DetectionModelSelection,
    labels: list[str],
    gliner_detection_threshold: float,
    preview_num_records: int | None,
    validation_max_entities_per_call: int,
    validation_excerpt_window_chars: int,
    validation_single_chunk_full_text: bool,
    client: DirectDetectionClient,
    data_summary: str | None,
    workflow_name: str,
    seed_workflow_name: str,
    skip_augmentation: bool,
) -> dw.EntityDetectionResult:
    workflow_model_configs = workflow._inject_detector_params(
        model_configs=model_configs,
        selected_models=selected_models,
        labels=labels,
        gliner_detection_threshold=gliner_detection_threshold,
    )
    seed_result = workflow._adapter.run_workflow(
        dataframe,
        model_configs=workflow_model_configs,
        columns=_detector_native_validate_seed_columns(selected_models),
        workflow_name=seed_workflow_name,
        preview_num_records=preview_num_records,
    )
    output_rows: list[dict[str, Any]] = []
    output_indices: list[Any] = []
    failed_records = list(seed_result.failed_records)
    validation_prompt_mode = _native_validation_prompt_mode(validation_single_chunk_full_text)
    tasks = [
        _NativeStagedTask(ordinal=ordinal, index=index, row=row.copy(deep=True))
        for ordinal, (index, row) in enumerate(seed_result.dataframe.iterrows())
    ]
    params = _DetectorNativeValidationParams(
        labels=labels,
        client=client,
        data_summary=data_summary,
        validation_prompt_mode=validation_prompt_mode,
        validation_max_entities_per_call=validation_max_entities_per_call,
        validation_excerpt_window_chars=validation_excerpt_window_chars,
        workflow_name=workflow_name,
        skip_augmentation=skip_augmentation,
    )

    for result in _execute_detector_native_validate_tasks(tasks, params=params):
        _record_detector_native_validation_result(result)
        if result.failed_record is not None:
            failed_records.append(result.failed_record)
            continue
        if result.output_row is None:
            failed_records.append(
                _native_failed_record(
                    result.index,
                    workflow_name=workflow_name,
                    error="native detector-seed validation produced no row",
                )
            )
            continue
        output_rows.append(result.output_row)
        output_indices.append(result.index)

    return dw.EntityDetectionResult(
        dataframe=_native_output_dataframe(
            seed_result.dataframe, output_rows=output_rows, output_indices=output_indices
        ),
        failed_records=failed_records,
    )


def _execute_detector_native_validate_tasks(
    tasks: list[_NativeStagedTask],
    *,
    params: _DetectorNativeValidationParams,
) -> list[_DetectorNativeValidationRowResult]:
    if not tasks:
        return []
    worker_count = _native_staged_worker_count(len(tasks))
    if worker_count == 1:
        return [_execute_detector_native_validate_task(task, params=params) for task in tasks]
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(
            executor.map(
                lambda task: _execute_detector_native_validate_task(task, params=params),
                tasks,
            )
        )


def _execute_detector_native_validate_task(
    task: _NativeStagedTask,
    *,
    params: _DetectorNativeValidationParams,
) -> _DetectorNativeValidationRowResult:
    return _execute_detector_native_validate_row(
        task.row,
        index=task.index,
        ordinal=task.ordinal,
        labels=params.labels,
        client=params.client,
        data_summary=params.data_summary,
        validation_prompt_mode=params.validation_prompt_mode,
        validation_max_entities_per_call=params.validation_max_entities_per_call,
        validation_excerpt_window_chars=params.validation_excerpt_window_chars,
        workflow_name=params.workflow_name,
        skip_augmentation=params.skip_augmentation,
    )


def _detector_native_validate_seed_columns(
    selected_models: DetectionModelSelection,
) -> list[LLMTextColumnConfig | CustomColumnConfig]:
    return [
        LLMTextColumnConfig(
            name=COL_RAW_DETECTED,
            prompt=_jinja(COL_TEXT),
            model_alias=_detector_alias(selected_models),
        ),
        CustomColumnConfig(name=COL_SEED_ENTITIES, generator_function=parse_detected_entities),
        CustomColumnConfig(name=COL_SEED_VALIDATION_CANDIDATES, generator_function=prepare_validation_inputs),
    ]


def _execute_detector_native_validate_row(
    row: pd.Series,
    *,
    index: object,
    ordinal: int,
    labels: list[str],
    client: DirectDetectionClient,
    data_summary: str | None,
    validation_prompt_mode: ValidationPromptMode,
    validation_max_entities_per_call: int,
    validation_excerpt_window_chars: int,
    workflow_name: str,
    skip_augmentation: bool,
) -> _DetectorNativeValidationRowResult:
    output_row = row.to_dict()
    request = _native_staged_request(row, index=index, ordinal=ordinal, labels=labels, data_summary=data_summary)
    config = StagedExecutionConfig(
        validation_prompt_mode=validation_prompt_mode,
        validation_max_entities_per_call=validation_max_entities_per_call,
        validation_excerpt_window_chars=validation_excerpt_window_chars,
        skip_augmentation=skip_augmentation,
    )
    request_count = _native_validation_request_count(
        output_row,
        validation_prompt_mode=validation_prompt_mode,
        validation_max_entities_per_call=validation_max_entities_per_call,
    )
    if not skip_augmentation:
        request_count += 1
    started = time.perf_counter()
    try:
        validation_completion = _run_validation_phase(output_row, request, client, config)
        augmentation_completion = _run_augmentation_phase(output_row, request, client, config)
        completion = _combine_detector_native_completions([validation_completion, augmentation_completion])
        merge_and_build_candidates(output_row)
        apply_validation_and_finalize(output_row)
        normalize_birth_context_final_entities(output_row, allowed_labels=labels)
    except Exception as exc:  # noqa: BLE001 - benchmark experiment records per-row failures
        return _DetectorNativeValidationRowResult(
            ordinal=ordinal,
            index=index,
            workflow_name=workflow_name,
            output_row=None,
            failed_record=_native_failed_record(
                index,
                workflow_name=workflow_name,
                error=f"{type(exc).__name__}: {exc}",
            ),
            completion=None,
            elapsed_sec=time.perf_counter() - started,
            request_count=request_count,
        )
    return _DetectorNativeValidationRowResult(
        ordinal=ordinal,
        index=index,
        workflow_name=workflow_name,
        output_row=output_row,
        failed_record=None,
        completion=completion,
        elapsed_sec=time.perf_counter() - started,
        request_count=request_count,
    )


def _native_validation_request_count(
    row: dict[str, Any],
    *,
    validation_prompt_mode: ValidationPromptMode,
    validation_max_entities_per_call: int,
) -> int:
    candidate_count = len(ValidationCandidatesSchema.from_raw(row.get(COL_SEED_VALIDATION_CANDIDATES, {})).candidates)
    if candidate_count == 0:
        return 0
    if validation_prompt_mode == ValidationPromptMode.full_text:
        return 1
    return (candidate_count + validation_max_entities_per_call - 1) // validation_max_entities_per_call


def _combine_detector_native_completions(completions: list[DirectCompletion]) -> DirectCompletion:
    return DirectCompletion(
        content=completions[-1].content if completions else "",
        elapsed_sec=sum(float(completion.elapsed_sec or 0.0) for completion in completions),
        usage=_sum_usage_dicts([dict(completion.usage or {}) for completion in completions]),
    )


def _record_detector_native_validation_completion(
    completion: DirectCompletion,
    *,
    request_count: int,
    workflow_name: str,
) -> None:
    record_model_workflow(
        workflow_name=workflow_name,
        model_aliases=[_NATIVE_DIRECT_MODEL_ALIAS],
        input_row_count=1,
        output_row_count=1,
        failed_record_count=0,
        elapsed_sec=float(completion.elapsed_sec or 0.0),
        model_usage=_native_single_pass_model_usage(
            successful_requests=request_count,
            failed_requests=0,
            usage=dict(completion.usage or {}),
        ),
    )


def _record_detector_native_validation_result(result: _DetectorNativeValidationRowResult) -> None:
    if result.completion is not None:
        _record_detector_native_validation_completion(
            result.completion,
            request_count=result.request_count,
            workflow_name=result.workflow_name,
        )
        return
    if result.failed_record is not None:
        _record_detector_native_validation_error(
            elapsed_sec=result.elapsed_sec,
            request_count=result.request_count,
            workflow_name=result.workflow_name,
        )


def _record_detector_native_validation_error(*, elapsed_sec: float, request_count: int, workflow_name: str) -> None:
    record_model_workflow(
        workflow_name=workflow_name,
        model_aliases=[_NATIVE_DIRECT_MODEL_ALIAS],
        input_row_count=1,
        output_row_count=0,
        failed_record_count=1,
        elapsed_sec=elapsed_sec,
        status="error",
        model_usage=_native_single_pass_model_usage(
            successful_requests=0,
            failed_requests=max(request_count, 1),
            usage={},
        ),
    )


def _make_native_staged_method(
    *,
    native_client: DirectDetectionClient | None,
    gliner_seed_client: GlinerSeedClient | None,
    seed_source: SeedSource,
    workflow_name: str,
    skip_augmentation: bool,
) -> _DetectAndValidate:
    def detect_and_validate_entities(
        self: dw.EntityDetectionWorkflow,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: DetectionModelSelection,
        gliner_detection_threshold: float,
        validation_max_entities_per_call: int = dw._DEFAULT_VALIDATION_MAX_ENTITIES_PER_CALL,
        validation_excerpt_window_chars: int = dw._DEFAULT_VALIDATION_EXCERPT_WINDOW_CHARS,
        validation_single_chunk_full_text: bool = True,
        entity_labels: list[str] | None = None,
        data_summary: str | None = None,
        preview_num_records: int | None = None,
    ) -> dw.EntityDetectionResult:
        _ = self, model_configs, selected_models, gliner_detection_threshold
        labels = dw._resolve_detection_labels(entity_labels)
        client = native_client or HttpxDirectDetectionClient()
        return _run_native_staged_detection(
            dataframe,
            labels=labels,
            client=client,
            gliner_seed_client=gliner_seed_client,
            data_summary=data_summary,
            preview_num_records=preview_num_records,
            validation_max_entities_per_call=validation_max_entities_per_call,
            validation_excerpt_window_chars=validation_excerpt_window_chars,
            validation_single_chunk_full_text=validation_single_chunk_full_text,
            seed_source=seed_source,
            workflow_name=workflow_name,
            skip_augmentation=skip_augmentation,
        )

    return detect_and_validate_entities


def _run_native_staged_detection(
    dataframe: pd.DataFrame,
    *,
    labels: list[str],
    client: DirectDetectionClient,
    gliner_seed_client: GlinerSeedClient | None,
    data_summary: str | None,
    preview_num_records: int | None,
    validation_max_entities_per_call: int,
    validation_excerpt_window_chars: int,
    validation_single_chunk_full_text: bool,
    seed_source: SeedSource,
    workflow_name: str,
    skip_augmentation: bool,
) -> dw.EntityDetectionResult:
    source_df = dataframe.iloc[:preview_num_records].copy() if preview_num_records is not None else dataframe.copy()
    output_rows: list[dict[str, Any]] = []
    output_indices: list[Any] = []
    failed_records: list[FailedRecord] = []
    validation_prompt_mode = _native_validation_prompt_mode(validation_single_chunk_full_text)
    params = _NativeStagedRunParams(
        labels=labels,
        client=client,
        gliner_seed_client=gliner_seed_client,
        data_summary=data_summary,
        validation_prompt_mode=validation_prompt_mode,
        validation_max_entities_per_call=validation_max_entities_per_call,
        validation_excerpt_window_chars=validation_excerpt_window_chars,
        seed_source=seed_source,
        workflow_name=workflow_name,
        skip_augmentation=skip_augmentation,
    )
    tasks = [
        _NativeStagedTask(ordinal=ordinal, index=index, row=row.copy(deep=True))
        for ordinal, (index, row) in enumerate(source_df.iterrows())
    ]

    for result in _execute_native_staged_tasks(tasks, params=params):
        if result.failed_record is not None:
            failed_records.append(result.failed_record)
            continue
        if result.output_row is None:
            failed_records.append(
                _native_failed_record(
                    result.index,
                    workflow_name=workflow_name,
                    error="native staged detection produced no row",
                )
            )
            continue
        if result.case is not None:
            _record_native_direct_usage(result.case, workflow_name=workflow_name)
        output_rows.append(result.output_row)
        output_indices.append(result.index)

    return dw.EntityDetectionResult(
        dataframe=_native_output_dataframe(source_df, output_rows=output_rows, output_indices=output_indices),
        failed_records=failed_records,
    )


def _execute_native_staged_tasks(
    tasks: list[_NativeStagedTask],
    *,
    params: _NativeStagedRunParams,
) -> list[_NativeStagedRowResult]:
    if not tasks:
        return []
    worker_count = _native_staged_worker_count(len(tasks))
    if worker_count == 1:
        return [_execute_native_staged_task(task, params=params) for task in tasks]
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(lambda task: _execute_native_staged_task(task, params=params), tasks))


def _native_staged_worker_count(task_count: int) -> int:
    return max(1, min(task_count, _NATIVE_STAGED_MAX_WORKERS))


def _execute_native_staged_task(
    task: _NativeStagedTask,
    *,
    params: _NativeStagedRunParams,
) -> _NativeStagedRowResult:
    try:
        request = _native_staged_request(
            task.row,
            index=task.index,
            ordinal=task.ordinal,
            labels=params.labels,
            data_summary=params.data_summary,
        )
        execution = execute_staged_detection_case(
            request,
            client=params.client,
            seed_client=params.gliner_seed_client,
            seed_source=params.seed_source,
            skip_augmentation=params.skip_augmentation,
            validation_prompt_mode=params.validation_prompt_mode,
            validation_max_entities_per_call=params.validation_max_entities_per_call,
            validation_excerpt_window_chars=params.validation_excerpt_window_chars,
        )
        if execution.case.status != StagedCaseStatus.completed:
            return _NativeStagedRowResult(
                ordinal=task.ordinal,
                index=task.index,
                output_row=None,
                failed_record=_native_failed_record(
                    task.index,
                    workflow_name=params.workflow_name,
                    error=execution.case.error,
                ),
                case=execution.case,
            )
        return _NativeStagedRowResult(
            ordinal=task.ordinal,
            index=task.index,
            output_row=_native_detection_result_row(task.row, execution_row=execution.row),
            failed_record=None,
            case=execution.case,
        )
    except Exception as exc:  # noqa: BLE001 - benchmark experiment records per-row failures
        return _NativeStagedRowResult(
            ordinal=task.ordinal,
            index=task.index,
            output_row=None,
            failed_record=_native_failed_record(
                task.index,
                workflow_name=params.workflow_name,
                error=f"{type(exc).__name__}: {exc}",
            ),
            case=None,
        )


def _native_validation_prompt_mode(validation_single_chunk_full_text: bool) -> ValidationPromptMode:
    return ValidationPromptMode.full_text if validation_single_chunk_full_text else ValidationPromptMode.chunked_excerpt


def _native_failed_record(index: object, *, workflow_name: str, error: str | None) -> FailedRecord:
    return FailedRecord(
        record_id=str(index),
        step=workflow_name,
        reason=error or "native staged detection failed",
    )


def _record_native_direct_usage(case: StagedDetectionCase, *, workflow_name: str) -> None:
    model_usage = _native_staged_model_usage(case)
    if not model_usage:
        return
    record_model_workflow(
        workflow_name=workflow_name,
        model_aliases=sorted(model_usage),
        input_row_count=1,
        output_row_count=1 if case.status == StagedCaseStatus.completed else 0,
        failed_record_count=0 if case.status == StagedCaseStatus.completed else 1,
        elapsed_sec=case.model_elapsed_sec or case.elapsed_sec or 0.0,
        model_usage=model_usage,
    )


def _native_staged_model_usage(case: StagedDetectionCase) -> dict[str, dict[str, Any]]:
    usage: dict[str, dict[str, Any]] = {}
    native_requests = 0
    native_usage: list[dict[str, int]] = []

    if case.phase_model_work.seed:
        if case.seed_source == SeedSource.gliner:
            usage[_GLINER_DIRECT_MODEL_ALIAS] = _direct_model_usage_entry(
                alias=_GLINER_DIRECT_MODEL_ALIAS,
                model_name=_GLINER_DIRECT_MODEL_NAME,
                provider_name=_GLINER_DIRECT_MODEL_PROVIDER,
                successful_requests=case.phase_model_requests.seed,
                usage=case.phase_usage.seed,
            )
        else:
            native_requests += case.phase_model_requests.seed
            native_usage.append(case.phase_usage.seed)
    if case.phase_model_work.validation:
        native_requests += case.phase_model_requests.validation
        native_usage.append(case.phase_usage.validation)
    if case.phase_model_work.augmentation:
        native_requests += case.phase_model_requests.augmentation
        native_usage.append(case.phase_usage.augmentation)
    if native_requests:
        usage[_NATIVE_DIRECT_MODEL_ALIAS] = _direct_model_usage_entry(
            alias=_NATIVE_DIRECT_MODEL_ALIAS,
            model_name=_NATIVE_DIRECT_MODEL_NAME,
            provider_name=_NATIVE_DIRECT_MODEL_PROVIDER,
            successful_requests=native_requests,
            usage=_sum_usage_dicts(native_usage),
        )
    return usage


def _direct_model_usage_entry(
    *,
    alias: str,
    model_name: str,
    provider_name: str,
    successful_requests: int,
    usage: dict[str, int],
) -> dict[str, Any]:
    return {
        "model_alias": alias,
        "model_name": model_name,
        "model_provider_name": provider_name,
        "request_usage": {
            "successful_requests": successful_requests,
            "failed_requests": 0,
            "total_requests": successful_requests,
        },
        "token_usage": _native_token_usage(usage),
    }


def _sum_usage_dicts(usages: list[dict[str, int]]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for usage in usages:
        for key, value in usage.items():
            if isinstance(value, int):
                totals[key] += value
    return dict(sorted(totals.items()))


def _native_token_usage(usage: dict[str, int]) -> dict[str, int]:
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
    total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _native_staged_request(
    row: pd.Series,
    *,
    index: object,
    ordinal: int,
    labels: list[str],
    data_summary: str | None,
) -> StagedDetectionRequest:
    return StagedDetectionRequest(
        case_id=f"native-rules-router-{ordinal}",
        text=str(row.get(COL_TEXT, "")),
        labels=labels,
        row_index=_safe_row_index(index, fallback=ordinal),
        data_summary=data_summary,
    )


def _native_detection_result_row(row: pd.Series, *, execution_row: dict[str, Any]) -> dict[str, Any]:
    output_row = row.to_dict()
    output_row[COL_DETECTED_ENTITIES] = execution_row.get(
        COL_DETECTED_ENTITIES,
        EntitiesSchema().model_dump(mode="json"),
    )
    output_row[COL_TAGGED_TEXT] = execution_row.get(COL_TAGGED_TEXT, str(row.get(COL_TEXT, "")))
    return output_row


def _safe_row_index(index: object, *, fallback: int) -> int:
    try:
        return int(index)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback


def _native_output_dataframe(
    source_df: pd.DataFrame,
    *,
    output_rows: list[dict[str, Any]],
    output_indices: list[Any],
) -> pd.DataFrame:
    if output_rows:
        return pd.DataFrame(output_rows, index=output_indices)
    output = source_df.iloc[0:0].copy()
    output[COL_DETECTED_ENTITIES] = pd.Series(dtype="object")
    output[COL_TAGGED_TEXT] = pd.Series(dtype="object")
    return output


def _detect_with_rules_only(
    self: dw.EntityDetectionWorkflow,
    dataframe: pd.DataFrame,
    *,
    model_configs: list[ModelConfig],
    selected_models: DetectionModelSelection,
    gliner_detection_threshold: float,
    validation_max_entities_per_call: int = dw._DEFAULT_VALIDATION_MAX_ENTITIES_PER_CALL,
    validation_excerpt_window_chars: int = dw._DEFAULT_VALIDATION_EXCERPT_WINDOW_CHARS,
    validation_single_chunk_full_text: bool = True,
    entity_labels: list[str] | None = None,
    data_summary: str | None = None,
    preview_num_records: int | None = None,
) -> dw.EntityDetectionResult:
    return self.detect_with_high_confidence_rules(dataframe, entity_labels=entity_labels)


def _make_rules_covered_or_default_method(original: _DetectAndValidate) -> _DetectAndValidate:
    def detect_and_validate_entities(
        self: dw.EntityDetectionWorkflow,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: DetectionModelSelection,
        gliner_detection_threshold: float,
        validation_max_entities_per_call: int = dw._DEFAULT_VALIDATION_MAX_ENTITIES_PER_CALL,
        validation_excerpt_window_chars: int = dw._DEFAULT_VALIDATION_EXCERPT_WINDOW_CHARS,
        validation_single_chunk_full_text: bool = True,
        entity_labels: list[str] | None = None,
        data_summary: str | None = None,
        preview_num_records: int | None = None,
    ) -> dw.EntityDetectionResult:
        labels = dw._resolve_detection_labels(entity_labels)
        if _labels_are_rules_only(labels):
            return _detect_rules_covered_rows_or_default(
                original,
                self,
                dataframe,
                model_configs=model_configs,
                selected_models=selected_models,
                gliner_detection_threshold=gliner_detection_threshold,
                validation_max_entities_per_call=validation_max_entities_per_call,
                validation_excerpt_window_chars=validation_excerpt_window_chars,
                validation_single_chunk_full_text=validation_single_chunk_full_text,
                entity_labels=entity_labels,
                data_summary=data_summary,
                preview_num_records=preview_num_records,
                labels=labels,
            )
        return original(
            self,
            dataframe,
            model_configs=model_configs,
            selected_models=selected_models,
            gliner_detection_threshold=gliner_detection_threshold,
            validation_max_entities_per_call=validation_max_entities_per_call,
            validation_excerpt_window_chars=validation_excerpt_window_chars,
            validation_single_chunk_full_text=validation_single_chunk_full_text,
            entity_labels=entity_labels,
            data_summary=data_summary,
            preview_num_records=preview_num_records,
        )

    return detect_and_validate_entities


def _detect_rules_covered_rows_or_default(
    original: _DetectAndValidate,
    self: dw.EntityDetectionWorkflow,
    dataframe: pd.DataFrame,
    *,
    model_configs: list[ModelConfig],
    selected_models: DetectionModelSelection,
    gliner_detection_threshold: float,
    validation_max_entities_per_call: int,
    validation_excerpt_window_chars: int,
    validation_single_chunk_full_text: bool,
    entity_labels: list[str] | None,
    data_summary: str | None,
    preview_num_records: int | None,
    labels: list[str],
) -> dw.EntityDetectionResult:
    started = time.perf_counter()
    if dataframe.empty:
        result = _detect_with_rules_only(
            self,
            dataframe,
            model_configs=model_configs,
            selected_models=selected_models,
            gliner_detection_threshold=gliner_detection_threshold,
            validation_max_entities_per_call=validation_max_entities_per_call,
            validation_excerpt_window_chars=validation_excerpt_window_chars,
            validation_single_chunk_full_text=validation_single_chunk_full_text,
            entity_labels=entity_labels,
            data_summary=data_summary,
            preview_num_records=preview_num_records,
        )
        _record_rules_covered_route(
            started=started,
            total_row_count=0,
            rule_row_count=0,
            fallback_row_count=0,
            result=result,
        )
        return result

    coverage_mask = dataframe[COL_TEXT].apply(
        lambda text: _structured_assignments_are_rule_covered(str(text), labels=labels)
    )
    if bool(coverage_mask.all()):
        result = _detect_with_rules_only(
            self,
            dataframe,
            model_configs=model_configs,
            selected_models=selected_models,
            gliner_detection_threshold=gliner_detection_threshold,
            validation_max_entities_per_call=validation_max_entities_per_call,
            validation_excerpt_window_chars=validation_excerpt_window_chars,
            validation_single_chunk_full_text=validation_single_chunk_full_text,
            entity_labels=entity_labels,
            data_summary=data_summary,
            preview_num_records=preview_num_records,
        )
        _record_rules_covered_route(
            started=started,
            total_row_count=len(dataframe),
            rule_row_count=len(dataframe),
            fallback_row_count=0,
            result=result,
        )
        return result
    if not bool(coverage_mask.any()):
        result = original(
            self,
            dataframe,
            model_configs=model_configs,
            selected_models=selected_models,
            gliner_detection_threshold=gliner_detection_threshold,
            validation_max_entities_per_call=validation_max_entities_per_call,
            validation_excerpt_window_chars=validation_excerpt_window_chars,
            validation_single_chunk_full_text=validation_single_chunk_full_text,
            entity_labels=entity_labels,
            data_summary=data_summary,
            preview_num_records=preview_num_records,
        )
        _record_rules_covered_route(
            started=started,
            total_row_count=len(dataframe),
            rule_row_count=0,
            fallback_row_count=len(dataframe),
            result=result,
        )
        return result

    rule_rows, default_rows = split_rows(
        dataframe,
        column=COL_TEXT,
        predicate=lambda text: _structured_assignments_are_rule_covered(str(text), labels=labels),
    )

    rule_result = _detect_with_rules_only(
        self,
        rule_rows,
        model_configs=model_configs,
        selected_models=selected_models,
        gliner_detection_threshold=gliner_detection_threshold,
        validation_max_entities_per_call=validation_max_entities_per_call,
        validation_excerpt_window_chars=validation_excerpt_window_chars,
        validation_single_chunk_full_text=validation_single_chunk_full_text,
        entity_labels=entity_labels,
        data_summary=data_summary,
        preview_num_records=preview_num_records,
    )
    default_result = original(
        self,
        default_rows,
        model_configs=model_configs,
        selected_models=selected_models,
        gliner_detection_threshold=gliner_detection_threshold,
        validation_max_entities_per_call=validation_max_entities_per_call,
        validation_excerpt_window_chars=validation_excerpt_window_chars,
        validation_single_chunk_full_text=validation_single_chunk_full_text,
        entity_labels=entity_labels,
        data_summary=data_summary,
        preview_num_records=preview_num_records,
    )
    result = dw.EntityDetectionResult(
        dataframe=merge_and_reorder(rule_result.dataframe, default_result.dataframe),
        failed_records=[*rule_result.failed_records, *default_result.failed_records],
    )
    _record_rules_covered_route(
        started=started,
        total_row_count=len(dataframe),
        rule_row_count=len(rule_rows),
        fallback_row_count=len(default_rows),
        result=result,
    )
    return result


def _record_rules_covered_route(
    *,
    started: float,
    total_row_count: int,
    rule_row_count: int,
    fallback_row_count: int,
    result: dw.EntityDetectionResult,
) -> None:
    record_model_workflow(
        workflow_name="entity-detection-rules-covered-router",
        model_aliases=[],
        input_row_count=total_row_count,
        output_row_count=len(result.dataframe),
        failed_record_count=len(result.failed_records),
        elapsed_sec=time.perf_counter() - started,
        status="completed" if not result.failed_records else "partial",
        extra_fields={
            "route_total_row_count": total_row_count,
            "route_rule_row_count": rule_row_count,
            "route_fallback_row_count": fallback_row_count,
        },
    )


def _structured_assignments_are_rule_covered(text: str, *, labels: list[str]) -> bool:
    allowed_labels = set(labels)
    rule_spans = detect_high_confidence_entities(text, labels=labels)
    covered_ranges = [(span.start_position, span.end_position) for span in rule_spans]
    for match in _STRUCTURED_ASSIGNMENT_RE.finditer(text):
        label = _structured_assignment_label(match.group("key"))
        if label not in allowed_labels:
            continue
        start, end = _structured_assignment_value_span(match)
        if not _range_overlaps_any(start, end, covered_ranges):
            return False
    return True


def _structured_assignment_value_span(match: re.Match[str]) -> tuple[int, int]:
    if match.group("quoted") is not None:
        return match.span("quoted")
    return match.span("bare")


def _range_overlaps_any(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start < range_end and end > range_start for range_start, range_end in ranges)


def _structured_assignment_label(key: str) -> str:
    normalized = key.lower().replace("-", "_")
    if normalized in {"api_key", "aws_access_key_id", "access_key_id", "hf_token", "token", "auth_token", "session_id"}:
        return "api_key"
    if normalized == "authorization":
        return "api_key"
    if normalized in {"password", "pass", "secret", "aws_secret_access_key", "django_secret"}:
        return "password"
    if normalized == "database_url":
        return "url"
    if normalized == "pin":
        return "pin"
    if normalized in {"user", "username", "user_name", "login", "account"}:
        return "user_name"
    if normalized == "cookie":
        return "http_cookie"
    if normalized in {"trace_id", "request_id", "req_id", "order_id", "tenant_id", "unique_id"}:
        return "unique_id"
    if normalized in {"url", "uri", "endpoint", "callback"}:
        return "url"
    if normalized == "email":
        return "email"
    return ""


def _labels_are_rules_only(labels: list[str]) -> bool:
    return dw.labels_are_supported_by_structured_rule_fast_lane(labels)
