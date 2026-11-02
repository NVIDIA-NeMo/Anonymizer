# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, cast

import pandas as pd
from data_designer.config.base import SkipConfig
from data_designer.config.column_configs import LLMStructuredColumnConfig, LLMTextColumnConfig
from data_designer.config.column_types import ColumnConfigT
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.models import ModelConfig
from pydantic import Field

from anonymizer.config.anonymizer_config import Detect as AnonymizerDetectConfig
from anonymizer.config.anonymizer_config import resolve_effective_detection_labels
from anonymizer.config.models import DetectionModelSelection
from anonymizer.config.regex import BuiltinRegex, RegexRule
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import (
    COL_ADMISSION_RESULTS,
    COL_AUGMENTED_ENTITIES,
    COL_DETECTED_ENTITIES,
    COL_DETECTOR_SOURCE,
    COL_ENTITIES_BY_VALUE,
    COL_FINAL_ENTITIES,
    COL_LATENT_ENTITIES,
    COL_MERGED_ENTITIES,
    COL_RAW_DETECTED,
    COL_REGEX_ENTITIES,
    COL_SEED_ENTITIES,
    COL_SEED_ENTITIES_JSON,
    COL_SEED_VALIDATION_CANDIDATES,
    COL_TAG_NOTATION,
    COL_TAGGED_TEXT,
    COL_TEXT,
    COL_VALIDATION_DECISIONS,
    COL_VALIDATION_SKELETON,
    ENTITY_LABEL_EXAMPLES,
    SEED_DETECTION_RESULT_ID,
    _jinja,
)
from anonymizer.engine.detection.chunked_validation import VALIDATION_EXCERPT_VAR
from anonymizer.engine.detection.entity_label_examples import ResolvedEntityLabelConfig, resolve_entity_label_config
from anonymizer.engine.detection.postprocess import (
    EntitySpan,
    group_entities_by_value,
    normalize_label,
    normalize_labels,
)
from anonymizer.engine.detection.regex_detection import (
    DEFAULT_MAX_MATCHES_PER_RULE,
    DEFAULT_REGEX_TIMEOUT_SECONDS,
    resolve_regex_constrained_labels,
    resolve_regex_rules,
    validate_exportable_regex_rules,
)
from anonymizer.engine.ndd.adapter import FailedRecord, NddAdapter
from anonymizer.engine.ndd.model_loader import resolve_model_alias, resolve_model_aliases
from anonymizer.engine.prompt_utils import substitute_placeholders
from anonymizer.engine.schemas import (
    EntitiesByValueSchema,
    EntitiesSchema,
    LatentEntitiesSchema,
)
from anonymizer.engine.workflow_columns.detection.config import (
    AugmentationConfig,
    ChunkedValidationConfig,
    DetectionTransformConfig,
    DetectionTransformOperation,
    RegexDetectionConfig,
)
from anonymizer.measurement import stage_timer

logger = logging.getLogger("anonymizer.detection")


class _PrivatePromptLLMStructuredColumnConfig(LLMStructuredColumnConfig):
    """Structured column config whose prompt is omitted from repr-based setup logs."""

    _normalize_for_data_designer: ClassVar[bool] = True
    prompt: str = Field(repr=False)


# Defaults for the two chunked-validation knobs. Sourced from the Detect config
# so there is a single source of truth; the workflow method defaults exist so
# internal tests and ad-hoc callers do not have to wire plumbing by hand.
_DEFAULT_VALIDATION_MAX_ENTITIES_PER_CALL: int = AnonymizerDetectConfig.model_fields[
    "validation_max_entities_per_call"
].default
_DEFAULT_VALIDATION_EXCERPT_WINDOW_CHARS: int = AnonymizerDetectConfig.model_fields[
    "validation_excerpt_window_chars"
].default


@dataclass(frozen=True)
class EntityDetectionResult:
    dataframe: pd.DataFrame
    failed_records: list[FailedRecord]


class EntityDetectionWorkflow:
    """Detection workflow using NDD LLM and Anonymizer workflow columns."""

    def __init__(self, adapter: NddAdapter) -> None:
        self._adapter = adapter

    def detect_and_validate_entities(
        self,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: DetectionModelSelection,
        gliner_detection_threshold: float,
        validation_max_entities_per_call: int = _DEFAULT_VALIDATION_MAX_ENTITIES_PER_CALL,
        validation_excerpt_window_chars: int = _DEFAULT_VALIDATION_EXCERPT_WINDOW_CHARS,
        validation_single_chunk_full_text: bool = True,
        entity_labels: list[str] | None = None,
        excluded_entity_labels: list[str] | None = None,
        entity_label_examples: dict[str, list[str]] | None = None,
        builtin_regexes: bool = True,
        regex_rules: list[BuiltinRegex | RegexRule] | None = None,
        data_summary: str | None = None,
        preview_num_records: int | None = None,
        _resolved_label_config: ResolvedEntityLabelConfig | None = None,
    ) -> EntityDetectionResult:
        """Run the core detection pipeline: GLiNER NER, LLM validation, LLM augmentation, and finalization.

        This is the primary detection workflow. It detects entities via GLiNER,
        validates/reclassifies them with an LLM (chunked across a pool of
        validator aliases), augments with additional entities the detector may
        have missed, and produces final standoff entity spans with overlap
        resolution.
        """
        workflow_model_configs, columns = self._build_detection_spec(
            model_configs=model_configs,
            selected_models=selected_models,
            gliner_detection_threshold=gliner_detection_threshold,
            validation_max_entities_per_call=validation_max_entities_per_call,
            validation_excerpt_window_chars=validation_excerpt_window_chars,
            validation_single_chunk_full_text=validation_single_chunk_full_text,
            entity_labels=entity_labels,
            excluded_entity_labels=excluded_entity_labels,
            entity_label_examples=entity_label_examples,
            builtin_regexes=builtin_regexes,
            regex_rules=regex_rules,
            data_summary=data_summary,
            _resolved_label_config=_resolved_label_config,
        )
        detection_result = self._adapter.run_workflow(
            dataframe,
            model_configs=workflow_model_configs,
            columns=columns,
            workflow_name="entity-detection",
            preview_num_records=preview_num_records,
        )
        detected_df = detection_result.dataframe.copy()
        return EntityDetectionResult(dataframe=detected_df, failed_records=detection_result.failed_records)

    def _build_detection_spec(
        self,
        *,
        model_configs: list[ModelConfig],
        selected_models: DetectionModelSelection,
        gliner_detection_threshold: float,
        validation_max_entities_per_call: int = _DEFAULT_VALIDATION_MAX_ENTITIES_PER_CALL,
        validation_excerpt_window_chars: int = _DEFAULT_VALIDATION_EXCERPT_WINDOW_CHARS,
        validation_single_chunk_full_text: bool = True,
        entity_labels: list[str] | None = None,
        excluded_entity_labels: list[str] | None = None,
        entity_label_examples: dict[str, list[str]] | None = None,
        builtin_regexes: bool = True,
        regex_rules: list[BuiltinRegex | RegexRule] | None = None,
        data_summary: str | None = None,
        _resolved_label_config: ResolvedEntityLabelConfig | None = None,
    ) -> tuple[list[ModelConfig], list[ColumnConfigT]]:
        """Build the (model_configs, columns) for the core detection workflow.

        Shared by :meth:`detect_and_validate_entities` (which executes it in-process)
        and :meth:`build_detection_config` (which exports it for an external runtime),
        so both paths run exactly the same workflow.
        """
        custom_rules = regex_rules or []
        labels = _resolve_detection_labels(
            entity_labels,
            regex_rules=custom_rules,
            excluded_entity_labels=set(excluded_entity_labels or []),
        )
        label_config = _resolved_label_config or resolve_entity_label_config(
            entity_labels=labels,
            excluded_entity_labels=excluded_entity_labels,
            entity_label_examples=entity_label_examples,
        )
        label_config.strict_labels = entity_labels is not None
        label_config.labels = labels
        for label in labels:
            label_config.validator_examples.setdefault(label, [])
        regex_constrained_labels = resolve_regex_constrained_labels(
            labels=labels,
            builtin_regexes=builtin_regexes,
            rules=custom_rules,
        )
        normalized_regex_constrained_labels = normalize_labels(regex_constrained_labels)
        model_labels = [label for label in labels if normalize_label(label) not in normalized_regex_constrained_labels]
        resolved_regex_rules = resolve_regex_rules(
            labels=labels,
            builtin_regexes=builtin_regexes,
            rules=custom_rules,
        )
        workflow_model_configs = self._inject_detector_params(
            model_configs=model_configs,
            selected_models=selected_models,
            labels=model_labels,
            gliner_detection_threshold=gliner_detection_threshold,
        )

        detection_alias = resolve_model_alias("entity_detector", selected_models)
        validator_aliases = resolve_model_aliases("entity_validator", selected_models)
        augmenter_alias = resolve_model_alias("entity_augmenter", selected_models)
        logger.debug(
            "detection aliases: detector=%s, validator_pool=%s, augmenter=%s",
            detection_alias,
            validator_aliases,
            augmenter_alias,
        )
        # ModelConfig.max_parallel_requests caps concurrency *per alias*. When
        # the pool has multiple validators each gets its own cap, so total
        # in-flight validator calls can reach sum(per-alias caps). Operators
        # provisioning rate budgets for downstream providers should size each
        # alias's cap accordingly.
        if len(validator_aliases) > 1:
            logger.warning(
                "entity validation runs across a pool of %d aliases (%s). "
                "max_parallel_requests is enforced per alias, so the pool "
                "multiplies total in-flight validator calls; budget provider "
                "TPM/RPM accordingly.",
                len(validator_aliases),
                validator_aliases,
            )

        columns = cast(
            list[ColumnConfigT],
            [
                RegexDetectionConfig(
                    name=COL_REGEX_ENTITIES,
                    rules=resolved_regex_rules,
                    timeout_seconds=DEFAULT_REGEX_TIMEOUT_SECONDS,
                    max_matches_per_rule=DEFAULT_MAX_MATCHES_PER_RULE,
                ),
                DetectionTransformConfig(
                    name=COL_DETECTOR_SOURCE,
                    operation=DetectionTransformOperation.PREPARE_DETECTOR_SOURCE,
                    drop=True,
                ),
                LLMTextColumnConfig(
                    name=COL_RAW_DETECTED,
                    prompt=_jinja(COL_DETECTOR_SOURCE, key="text"),
                    model_alias=detection_alias,
                    skip=SkipConfig(when=f"{{{{ {COL_DETECTOR_SOURCE} == {COL_DETECTOR_SOURCE} }}}}", value="")
                    if not model_labels
                    else None,
                ),
                DetectionTransformConfig(
                    name=COL_SEED_ENTITIES,
                    operation=DetectionTransformOperation.PARSE_DETECTED_ENTITIES,
                    detection_result_id=SEED_DETECTION_RESULT_ID,
                    source_instance=detection_alias,
                    regex_constrained_entity_labels=sorted(regex_constrained_labels),
                    propagate_skip=False,
                ),
                DetectionTransformConfig(
                    name=COL_SEED_VALIDATION_CANDIDATES,
                    operation=DetectionTransformOperation.PREPARE_VALIDATION_INPUTS,
                ),
                ChunkedValidationConfig(
                    name=COL_VALIDATION_DECISIONS,
                    pool=list(validator_aliases),
                    max_entities_per_call=validation_max_entities_per_call,
                    excerpt_window_chars=validation_excerpt_window_chars,
                    single_chunk_full_text=validation_single_chunk_full_text,
                    allowed_entity_labels=labels if label_config.strict_labels else None,
                    excluded_entity_labels=list(excluded_entity_labels or []),
                    prompt_template=_get_validation_prompt(
                        data_summary=data_summary,
                        labels=labels,
                        examples_by_label=label_config.validator_examples,
                        configured_examples=label_config.augmenter_examples,
                        excluded_entity_labels=list(excluded_entity_labels or []),
                    ),
                    drop=True,
                ),
                DetectionTransformConfig(
                    name=COL_ADMISSION_RESULTS,
                    operation=DetectionTransformOperation.ADMIT_VALIDATION_DECISIONS,
                    allowed_entity_labels=labels if label_config.strict_labels else None,
                    excluded_entity_labels=list(excluded_entity_labels or []),
                    regex_constrained_entity_labels=sorted(regex_constrained_labels),
                ),
                DetectionTransformConfig(
                    name=COL_SEED_ENTITIES_JSON,
                    operation=DetectionTransformOperation.APPLY_VALIDATION_TO_SEED_ENTITIES,
                ),
                AugmentationConfig(
                    name=COL_AUGMENTED_ENTITIES,
                    prompt_template=_get_augment_prompt(
                        data_summary=data_summary,
                        labels=model_labels,
                        strict_labels=label_config.strict_labels,
                        configured_examples=label_config.augmenter_examples,
                        excluded_entity_labels=list(excluded_entity_labels or []),
                    ),
                    model_alias=augmenter_alias,
                    allowed_entity_labels=labels if label_config.strict_labels else None,
                    excluded_entity_labels=list(excluded_entity_labels or []),
                    enabled=bool(model_labels),
                    regex_constrained_entity_labels=sorted(regex_constrained_labels),
                ),
                DetectionTransformConfig(
                    name=COL_MERGED_ENTITIES,
                    operation=DetectionTransformOperation.MERGE_AND_BUILD_CANDIDATES,
                ),
                DetectionTransformConfig(
                    name=COL_DETECTED_ENTITIES,
                    operation=DetectionTransformOperation.FINALIZE_ENTITIES,
                ),
            ],
        )
        return workflow_model_configs, columns

    def build_detection_config(
        self,
        dataframe: pd.DataFrame,
        *,
        seed_path: str | Path,
        model_configs: list[ModelConfig],
        selected_models: DetectionModelSelection,
        gliner_detection_threshold: float,
        validation_max_entities_per_call: int = _DEFAULT_VALIDATION_MAX_ENTITIES_PER_CALL,
        validation_excerpt_window_chars: int = _DEFAULT_VALIDATION_EXCERPT_WINDOW_CHARS,
        validation_single_chunk_full_text: bool = True,
        entity_labels: list[str] | None = None,
        excluded_entity_labels: list[str] | None = None,
        entity_label_examples: dict[str, list[str]] | None = None,
        builtin_regexes: bool = True,
        regex_rules: list[BuiltinRegex | RegexRule] | None = None,
        data_summary: str | None = None,
    ) -> DataDesignerConfigBuilder:
        """Build (without executing) the core detection workflow as a DataDesigner
        config, for an external at-scale executor to run. Produces the same columns
        as :meth:`detect_and_validate_entities` (culminating in final entities); the
        external runtime supplies the model providers and the seed dataset.
        """
        validate_exportable_regex_rules(
            regex_rules,
            active_labels=resolve_effective_detection_labels(
                entity_labels, regex_rules=regex_rules, excluded_entity_labels=excluded_entity_labels
            ),
        )
        workflow_model_configs, columns = self._build_detection_spec(
            model_configs=model_configs,
            selected_models=selected_models,
            gliner_detection_threshold=gliner_detection_threshold,
            validation_max_entities_per_call=validation_max_entities_per_call,
            validation_excerpt_window_chars=validation_excerpt_window_chars,
            validation_single_chunk_full_text=validation_single_chunk_full_text,
            entity_labels=entity_labels,
            excluded_entity_labels=excluded_entity_labels,
            entity_label_examples=entity_label_examples,
            builtin_regexes=builtin_regexes,
            regex_rules=regex_rules,
            data_summary=data_summary,
        )
        return self._adapter.build_config(
            dataframe,
            model_configs=workflow_model_configs,
            columns=columns,
            seed_path=seed_path,
        )

    def build_detection_builder_for_seed(
        self,
        *,
        seed_path: str | Path,
        model_configs: list[ModelConfig],
        selected_models: DetectionModelSelection,
        gliner_detection_threshold: float,
        validation_max_entities_per_call: int = _DEFAULT_VALIDATION_MAX_ENTITIES_PER_CALL,
        validation_excerpt_window_chars: int = _DEFAULT_VALIDATION_EXCERPT_WINDOW_CHARS,
        validation_single_chunk_full_text: bool = True,
        entity_labels: list[str] | None = None,
        excluded_entity_labels: list[str] | None = None,
        entity_label_examples: dict[str, list[str]] | None = None,
        builtin_regexes: bool = True,
        regex_rules: list[BuiltinRegex | RegexRule] | None = None,
        data_summary: str | None = None,
        job_index: int = 0,
        num_jobs: int = 1,
    ) -> DataDesignerConfigBuilder:
        """Build the detection workflow reading an EXISTING seed parquet (no write).

        Same columns as :meth:`build_detection_config`, but the seed dataset points at an
        already-written ``seed_path`` and optionally selects this worker's ordered
        partition (``job_index`` of ``num_jobs``). For a distributed executor (e.g. a SLURM
        orchestrator), the plugin column configs remain serializable and the model aliases
        are resolved by the runtime's providers.
        """
        validate_exportable_regex_rules(
            regex_rules,
            active_labels=resolve_effective_detection_labels(
                entity_labels, regex_rules=regex_rules, excluded_entity_labels=excluded_entity_labels
            ),
        )
        workflow_model_configs, columns = self._build_detection_spec(
            model_configs=model_configs,
            selected_models=selected_models,
            gliner_detection_threshold=gliner_detection_threshold,
            validation_max_entities_per_call=validation_max_entities_per_call,
            validation_excerpt_window_chars=validation_excerpt_window_chars,
            validation_single_chunk_full_text=validation_single_chunk_full_text,
            entity_labels=entity_labels,
            excluded_entity_labels=excluded_entity_labels,
            entity_label_examples=entity_label_examples,
            builtin_regexes=builtin_regexes,
            regex_rules=regex_rules,
            data_summary=data_summary,
        )
        return self._adapter.build_config_for_seed(
            model_configs=workflow_model_configs,
            columns=columns,
            seed_path=seed_path,
            job_index=job_index,
            num_jobs=num_jobs,
        )

    def identify_latent_entities(
        self,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: DetectionModelSelection,
        gliner_detection_threshold: float,
        entity_labels: list[str] | None = None,
        excluded_entity_labels: list[str] | None = None,
        entity_label_examples: dict[str, list[str]] | None = None,
        privacy_goal: PrivacyGoal | None,
        data_summary: str | None = None,
        preview_num_records: int | None = None,
        _resolved_label_config: ResolvedEntityLabelConfig | None = None,
    ) -> EntityDetectionResult:
        """Detect latent/inferred entities that could enable re-identification.

        Runs after ``detect_and_validate_entities`` when rewrite mode is
        enabled. Uses an LLM to identify entities inferable from context.
        """
        label_config = _resolved_label_config or resolve_entity_label_config(
            entity_labels=entity_labels,
            excluded_entity_labels=excluded_entity_labels,
            entity_label_examples=entity_label_examples,
        )
        labels = label_config.labels
        workflow_model_configs = self._inject_detector_params(
            model_configs=model_configs,
            selected_models=selected_models,
            labels=labels,
            gliner_detection_threshold=gliner_detection_threshold,
        )
        latent_alias = resolve_model_alias("latent_detector", selected_models)
        latent_result = self._adapter.run_workflow(
            dataframe,
            model_configs=workflow_model_configs,
            columns=[
                LLMStructuredColumnConfig(
                    name=COL_LATENT_ENTITIES,
                    prompt=_get_latent_prompt(
                        data_summary=data_summary,
                        privacy_goal=privacy_goal,
                        excluded_entity_labels=excluded_entity_labels,
                    ),
                    model_alias=latent_alias,
                    output_format=LatentEntitiesSchema,
                )
            ],
            workflow_name="latent-entity-detection",
            preview_num_records=preview_num_records,
        )
        latent_df = latent_result.dataframe.copy()
        if COL_LATENT_ENTITIES in latent_df.columns:
            latent_df[COL_LATENT_ENTITIES] = latent_df[COL_LATENT_ENTITIES].apply(
                lambda raw: _filter_excluded_latent_entities(raw, excluded_entity_labels)
            )
        return EntityDetectionResult(dataframe=latent_df, failed_records=latent_result.failed_records)

    def run(
        self,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: DetectionModelSelection,
        gliner_detection_threshold: float,
        validation_max_entities_per_call: int = _DEFAULT_VALIDATION_MAX_ENTITIES_PER_CALL,
        validation_excerpt_window_chars: int = _DEFAULT_VALIDATION_EXCERPT_WINDOW_CHARS,
        validation_single_chunk_full_text: bool = True,
        entity_labels: list[str] | None = None,
        excluded_entity_labels: list[str] | None = None,
        entity_label_examples: dict[str, list[str]] | None = None,
        builtin_regexes: bool = True,
        regex_rules: list[BuiltinRegex | RegexRule] | None = None,
        privacy_goal: PrivacyGoal | None = None,
        data_summary: str | None = None,
        tag_latent_entities: bool = True,
        compute_grouped_entities: bool | None = None,
        preview_num_records: int | None = None,
    ) -> EntityDetectionResult:
        """Run the full detection pipeline, optionally including latent entity detection.

        Calls ``detect_and_validate_entities`` first, then optionally
        ``identify_latent_entities`` if ``tag_latent_entities`` is True
        (rewrite mode). Merges failures from both stages.
        """
        with stage_timer(
            "EntityDetectionWorkflow.run",
            input_row_count=len(dataframe),
            tag_latent_entities=tag_latent_entities,
        ) as measurement:
            if tag_latent_entities and privacy_goal is None:
                raise ValueError("privacy_goal is required when tag_latent_entities=True (rewrite mode)")

            compute_grouped = True if compute_grouped_entities is None else compute_grouped_entities
            effective_labels = _resolve_detection_labels(
                entity_labels,
                regex_rules=regex_rules,
                excluded_entity_labels=excluded_entity_labels,
            )
            label_config = resolve_entity_label_config(
                entity_labels=effective_labels,
                excluded_entity_labels=excluded_entity_labels,
                entity_label_examples=entity_label_examples,
            )
            label_config.strict_labels = entity_labels is not None
            detected_result = self.detect_and_validate_entities(
                dataframe,
                model_configs=model_configs,
                selected_models=selected_models,
                gliner_detection_threshold=gliner_detection_threshold,
                validation_max_entities_per_call=validation_max_entities_per_call,
                validation_excerpt_window_chars=validation_excerpt_window_chars,
                validation_single_chunk_full_text=validation_single_chunk_full_text,
                entity_labels=entity_labels,
                excluded_entity_labels=excluded_entity_labels,
                entity_label_examples=entity_label_examples,
                builtin_regexes=builtin_regexes,
                regex_rules=regex_rules,
                data_summary=data_summary,
                preview_num_records=preview_num_records,
                _resolved_label_config=label_config,
            )

            if tag_latent_entities:
                latent_result = self.identify_latent_entities(
                    detected_result.dataframe,
                    model_configs=model_configs,
                    selected_models=selected_models,
                    gliner_detection_threshold=gliner_detection_threshold,
                    entity_labels=entity_labels,
                    excluded_entity_labels=excluded_entity_labels,
                    entity_label_examples=entity_label_examples,
                    privacy_goal=privacy_goal,
                    data_summary=data_summary,
                    preview_num_records=preview_num_records,
                    _resolved_label_config=label_config,
                )
                final_df = latent_result.dataframe.copy()
                final_failures = [*detected_result.failed_records, *latent_result.failed_records]
            else:
                final_df = detected_result.dataframe.copy()
                final_failures = detected_result.failed_records

            # When entity_labels is explicitly provided (even if it matches DEFAULT_ENTITY_LABELS),
            # the augmenter is strict and out-of-scope labels are filtered.
            # entity_labels=None is the only way to get permissive augmentation.
            if COL_DETECTED_ENTITIES in final_df.columns:
                allowed = set(label_config.labels) if label_config.strict_labels else None
                excluded_entity_labels_set = set(excluded_entity_labels) if excluded_entity_labels else None
                final_df[COL_FINAL_ENTITIES] = final_df[COL_DETECTED_ENTITIES].apply(
                    lambda raw: _materialize_final_entities(
                        raw,
                        allowed_labels=allowed,
                        excluded_entity_labels=excluded_entity_labels_set,
                    )
                )
                if compute_grouped:
                    final_df[COL_ENTITIES_BY_VALUE] = final_df[COL_FINAL_ENTITIES].apply(_build_entities_by_value)
            result = EntityDetectionResult(
                dataframe=final_df,
                failed_records=final_failures,
            )
            measurement.update(
                output_row_count=len(result.dataframe),
                failed_record_count=len(result.failed_records),
            )
            return result

    def _inject_detector_params(
        self,
        *,
        model_configs: list[ModelConfig],
        selected_models: DetectionModelSelection,
        labels: list[str],
        gliner_detection_threshold: float,
    ) -> list[ModelConfig]:
        resolved = deepcopy(model_configs)
        for config in resolved:
            if config.alias != selected_models.entity_detector:
                continue
            if config.inference_parameters.extra_body is None:
                config.inference_parameters.extra_body = {}
            config.inference_parameters.extra_body["labels"] = labels
            config.inference_parameters.extra_body["threshold"] = gliner_detection_threshold
            config.inference_parameters.extra_body["flat_ner"] = False
            break
        return resolved


def _resolve_detection_labels(
    entity_labels: list[str] | None,
    *,
    regex_rules: list[BuiltinRegex | RegexRule] | None = None,
    excluded_entity_labels: list[str] | set[str] | None = None,
) -> list[str]:
    labels = resolve_effective_detection_labels(
        entity_labels,
        regex_rules=regex_rules,
        excluded_entity_labels=excluded_entity_labels,
    )
    if not labels:
        logger.warning(
            "excluded_entity_labels removed all labels from the effective detection set. No entities will be detected."
        )
    return labels


def _materialize_final_entities(
    raw: object,
    *,
    allowed_labels: set[str] | None,
    excluded_entity_labels: set[str] | None = None,
) -> dict:
    """Build COL_FINAL_ENTITIES, applying the configured label scope."""
    parsed = EntitiesSchema.from_raw(raw)
    allowed = normalize_labels(allowed_labels) if allowed_labels is not None else None
    excluded = normalize_labels(excluded_entity_labels)
    kept = [
        e
        for e in parsed.entities
        if (allowed is None or normalize_label(e.label) in allowed) and normalize_label(e.label) not in excluded
    ]
    return EntitiesSchema(entities=kept).model_dump()


def _filter_excluded_latent_entities(raw: object, excluded_entity_labels: list[str] | None) -> object:
    """Remove excluded latent labels while preserving the structured payload shape."""
    excluded = normalize_labels(excluded_entity_labels)
    if not excluded:
        return raw

    if isinstance(raw, LatentEntitiesSchema):
        kept = [entity for entity in raw.latent_entities if normalize_label(entity.label) not in excluded]
        return LatentEntitiesSchema(latent_entities=kept).model_dump(mode="json")

    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return raw
        return _filter_excluded_latent_entities(parsed, excluded_entity_labels)

    if isinstance(raw, dict):
        entities = raw.get("latent_entities")
        if not isinstance(entities, list):
            return raw
        return {
            **raw,
            "latent_entities": [
                entity
                for entity in entities
                if not isinstance(entity, dict) or normalize_label(str(entity.get("label", ""))) not in excluded
            ],
        }

    # Retain support for legacy list-shaped traces.
    if isinstance(raw, list):
        return [
            entity
            for entity in raw
            if not isinstance(entity, dict) or normalize_label(str(entity.get("label", ""))) not in excluded
        ]

    return raw


def _build_entities_by_value(final_entities_raw: object) -> dict:
    """Derive COL_ENTITIES_BY_VALUE from COL_FINAL_ENTITIES."""
    parsed = EntitiesSchema.from_raw(final_entities_raw)
    spans = [
        EntitySpan(
            entity_id=e.id,
            value=e.value,
            label=e.label,
            start_position=e.start_position,
            end_position=e.end_position,
            score=e.score,
            source=e.source,
        )
        for e in parsed.entities
    ]
    return EntitiesByValueSchema(entities_by_value=group_entities_by_value(entities=spans)).model_dump(mode="json")


def _prompt_safe_configured_value(value: str) -> str:
    """Render configured prompt data without exposing template delimiters.

    Uses JSON string escaping for control characters, quotes, and backslashes
    (so a configured value can't break out of the single-line label/example
    format), then strips the surrounding quotes JSON adds so configured values
    render the same as the built-in examples they sit alongside.
    """
    escaped = json.dumps(value, ensure_ascii=False)[1:-1]
    return escaped.replace("{", r"\u007b").replace("}", r"\u007d")


def _format_label_examples(
    labels: list[str],
    examples_by_label: dict[str, list[str]] | None = None,
    configured_examples: dict[str, list[str]] | None = None,
) -> str:
    """Build a formatted list of entity classes with examples.

    ``examples_by_label`` supplies the resolved built-in and configured examples.
    Labels without resolved examples still appear so the LLM knows they are valid.
    """
    examples_by_label = ENTITY_LABEL_EXAMPLES if examples_by_label is None else examples_by_label
    configured_examples = configured_examples or {}
    lines: list[str] = []
    for label in labels:
        examples = examples_by_label.get(label)
        rendered_label = _prompt_safe_configured_value(label) if label in configured_examples else label
        if examples:
            configured_for_label = set(configured_examples.get(label, []))
            rendered = [
                _prompt_safe_configured_value(example) if example in configured_for_label else example
                for example in examples
            ]
            lines.append(f"- {rendered_label}: {', '.join(rendered)}")
        else:
            lines.append(f"- {rendered_label}")
    return "\n".join(lines)


def _get_validation_prompt(
    *,
    data_summary: str | None,
    labels: list[str],
    examples_by_label: dict[str, list[str]] | None = None,
    configured_examples: dict[str, list[str]] | None = None,
    excluded_entity_labels: list[str] | None = None,
) -> str:
    prompt = """Validate entity tags for privacy-sensitive information. For each entity in the template below, fill in the "decision" and "reason" fields. Fill in "proposed_label" only when decision is "reclass".

Use surrounding text and field names to determine what the candidate actually represents. Keep or reclassify it only when the context supports both its entity label and its sensitivity under the guidance in this prompt. If it is not sensitive, drop it rather than reclassifying it.
<<DATA_SUMMARY>>

Here are all the valid entity classes with examples (only classes from this list can be proposed):
<<LABEL_EXAMPLES>>

What to KEEP:
- Direct identifiers: names, emails, IDs, phone numbers, addresses, SSNs
- Quasi-identifiers: age, locations (cities, states, regions), occupations, dates, company names, education
- Technical data: credentials, URLs, device IDs, account numbers
- Demographics: gender, race, religion, language, etc.

What to RECLASS (reclassify):
- Entity IS privacy-relevant but has the WRONG label
- Set decision="reclass" and proposed_label to the correct label from the valid labels list
- Example: "San Diego" labeled as "country" → reclass to "city"

What to DROP:
- Universally famous entities ONLY (e.g., "Barack Obama")
  Note: Most cities/companies ARE quasi-identifiers and should be KEPT (Portland, London, Brasilia, etc.)
- Placeholders or label names (e.g., "name", "email", "date")
- Syntax/metadata in structured formats (field names, column headers, keywords)
- Vague time references (e.g., "today", "now", "soon")
- Kinship and family-role terms used as relationship descriptors are not quasi-identifiers.
- Filenames that are system executables or binaries (e.g., ending in .exe, .dll, .sys).
<<EXCLUDED_DROP>>
PERSONAL NAMES:
- Keep a supplied first_name or last_name label when context supports that personal-name use. Names can be short or written in lowercase, uppercase, or mixed case; capitalization alone neither proves nor rules out a name.
- Personal initials, with or without a period, can identify people in signatures, greetings, name sequences, or running text where they stand in for a person (e.g., "the lease was signed by K.M. on the tenant's behalf"). Do not drop an initial solely because no surname follows.
- Use context to distinguish given names from surnames; do not infer that role solely from word order. Never expand an initial into an inferred name.
- In structured data (JSON, forms, tables), a value in a name field is a name even when it is short (e.g., "surname": "Ng"); the field name itself is not.
- Drop name tags on variables, grades, units, list markers, ordinary words, or chat abbreviations when they are not being used as personal names.
- Examples: "First initial: Q." and "hi rose, can you review?" support personal-name uses; "a rose in the garden", "Grade: A", "4 m", and "thx for the help" do not.

PARTIAL-TOKEN RULE (HARD DROP):
- If the tagged value is only a substring of a larger contiguous token (letters, digits, underscore with no whitespace boundary), DROP it.
- A contiguous token means adjacent letters, digits, or underscore with no whitespace boundary.
- If characters immediately before or after the tagged span are alphanumeric, or underscore, the tag is a partial token and must be dropped.
- Example:
    {%- if <<TAG_NOTATION>> == "xml" -%}
    "internal_<unique_id>procID</unique_id>_id" → drop, because "procID" is inside "internal_procID_id"
    {%- elif <<TAG_NOTATION>> == "bracket" -%}
    "internal_[[procID|unique_id]]_id" → drop, because "procID" is inside "internal_procID_id"
    {%- elif <<TAG_NOTATION>> == "paren" -%}
    "internal_((SENSITIVE:unique_id|procID))_id" → drop, because "procID" is inside "internal_procID_id"
    {%- else -%}
    "internal_<<SENSITIVE:unique_id>>procID<</SENSITIVE:unique_id>>_id" → drop, because "procID" is inside "internal_procID_id"
    {%- endif -%}
- Example:
    {%- if <<TAG_NOTATION>> == "xml" -%}
    "<political_view>dem</political_view>eanor" → drop, because "dem" is inside "demeanor"
    {%- elif <<TAG_NOTATION>> == "bracket" -%}
    "[[dem|political_view]]eanor" → drop, because "dem" is inside "demeanor"
    {%- elif <<TAG_NOTATION>> == "paren" -%}
    "((SENSITIVE:political_view|dem))eanor" → drop, because "dem" is inside "demeanor"
    {%- else -%}
    "<<SENSITIVE:political_view>>dem<</SENSITIVE:political_view>>eanor" → drop, because "dem" is inside "demeanor"
    {%- endif -%}
- Apply this even if the substring itself looks privacy-relevant.

AGE RULE:
- Keep "age" only when the number clearly refers to a person's age (e.g., "35 years old", "35-year-old", "aged 35", "John, 35, ...")
- Patterns like "X years", "X months", or "X days" describing how long something happened indicate duration, not age.
- If the number refers to time, duration, counts, prices, measurements, codes, or dates — it is NOT age.
- If unclear, drop the tag.

DATE RULE:
- Keep complete and partial calendar dates under "date"; a day, month, and year are NOT all required.
- Keep named months, calendar years, and month/year combinations when context establishes calendar meaning: "March" in "The appointment is in March", "2024" in "She joined in 2024", and "March 2024" in "We met in March 2024".
- Keep a day of the month only with clear date context: "the 15th" in "The appointment is on the 15th". Standalone numbers without date context are not dates.
- Do not treat "March" in "March forward", a quantity such as "2024 items", or vague time words such as "today", "now", and "soon" as dates.


Additional rules:
- Check context matches label (not just format)
- Prefer ssn over national_id or account_number if ambiguous
- Prefer phone_number over fax_number unless "fax" is explicit
- VIN length 17 is vehicle_identifier, shorter is license_plate (check context for regional variations)
- Numbers preceded by '$' are monetary values, not entities like age or account number
- If classified as date_of_birth, verify context is appropriate; otherwise use date
- The word "straight" rarely has the label "sexuality"; if the context doesn't absolutely imply "sexuality", drop this tag
- The entity label "occupation" refers only to a specific paid job title or profession (e.g., "registered nurse", "software engineer", "retail salesperson", "teacher", "bartender"). Do NOT label generic roles, activities, or vague work-related nouns as occupations (e.g., "volunteer", "mentor", "guide", "partner", "supplier", "helper")
- You MUST fill in a decision for EVERY entry in the template — do not skip any
- Return ONLY the entries from the template — do not add new entries for entities not already in the template
- Copy ids exactly as given; never modify entries
- A tag named NAGROUP_gN marks one region with overlapping alternatives. Use the template's conflict_group_id, value, label, and offsets to decide each alternative independently.
- Template start_position and end_position are zero-based character offsets within the source excerpt, ignoring inserted tag markup. The start is inclusive and the end is exclusive. Return decisions using the supplied ids; do not generate or modify offsets.

Example:
{%- if <<TAG_NOTATION>> == "xml" -%}
Input text: My <first_name>name</first_name> is <first_name>Amy</first_name>, <age>35</age> in <NAGROUP_g1>San Diego</NAGROUP_g1>, <state>California</state>.
{%- elif <<TAG_NOTATION>> == "bracket" -%}
Input text: My [[name|first_name]] is [[Amy|first_name]], [[35|age]] in [[San Diego|NAGROUP_g1]], [[California|state]].
{%- elif <<TAG_NOTATION>> == "paren" -%}
Input text: My ((SENSITIVE:first_name|name)) is ((SENSITIVE:first_name|Amy)), ((SENSITIVE:age|35)) in ((SENSITIVE:NAGROUP_g1|San Diego)), ((SENSITIVE:state|California)).
{%- else -%}
Input text: My <<SENSITIVE:first_name>>name<</SENSITIVE:first_name>> is <<SENSITIVE:first_name>>Amy<</SENSITIVE:first_name>>, <<SENSITIVE:age>>35<</SENSITIVE:age>> in <<SENSITIVE:NAGROUP_g1>>San Diego<</SENSITIVE:NAGROUP_g1>>, <<SENSITIVE:state>>California<</SENSITIVE:state>>.
{%- endif -%}
Template: {"decisions": [{"id": "c1", "conflict_group_id": null, "value": "name", "label": "first_name", "start_position": 3, "end_position": 7, "decision": null, "proposed_label": null, "reason": null}, {"id": "c2", "conflict_group_id": null, "value": "Amy", "label": "first_name", "start_position": 11, "end_position": 14, "decision": null, "proposed_label": null, "reason": null}, {"id": "c3", "conflict_group_id": null, "value": "35", "label": "age", "start_position": 16, "end_position": 18, "decision": null, "proposed_label": null, "reason": null}, {"id": "c4", "conflict_group_id": "g1", "value": "San Diego", "label": "country", "start_position": 22, "end_position": 31, "decision": null, "proposed_label": null, "reason": null}, {"id": "c5", "conflict_group_id": "g1", "value": "Diego", "label": "first_name", "start_position": 26, "end_position": 31, "decision": null, "proposed_label": null, "reason": null}, {"id": "c6", "conflict_group_id": null, "value": "California", "label": "state", "start_position": 33, "end_position": 43, "decision": null, "proposed_label": null, "reason": null}]}
Output: <<VALIDATION_EXAMPLE_OUTPUT>>

---
Input text: <<TAGGED_TEXT>>
Template: <<VALIDATION_SKELETON>>
"""
    context_section = (
        "\nThe data summary may describe the dataset and provide sensitivity rules. "
        "Follow its explicit sensitivity rules when they conflict with the generic guidance or examples below. "
        "For categories or situations it does not address, use the generic guidance below. "
        "Do not infer that a value is non-sensitive merely because the data summary does not mention it. "
        "An allowed entity label defines what can be detected; "
        "it does not mean every instance of that category is sensitive.\n\n"
        "Keep or reclassify a candidate only when its value is sensitive in the context shown, "
        "applying explicit sensitivity rules from the data summary where provided "
        "and otherwise using the generic guidance below. "
        "Drop candidates that the data summary explicitly identifies as not sensitive, "
        "even when their entity label is correct.\n\n"
        f"Data summary: {data_summary}\n"
        if data_summary
        else ""
    )
    excluded = sorted(normalize_labels(excluded_entity_labels))
    excluded_drop = (
        "- Excluded entity types: drop these and close synonyms of them; do not return keep or reclass for: "
        + ", ".join(_prompt_safe_configured_value(label) for label in excluded)
        + "\n"
        if excluded
        else ""
    )
    return substitute_placeholders(
        prompt,
        {
            "<<EXCLUDED_DROP>>": excluded_drop,
            "<<VALIDATION_EXAMPLE_OUTPUT>>": _validation_example_output(labels=labels, excluded=set(excluded)),
            "<<TAG_NOTATION>>": COL_TAG_NOTATION,
            "<<TAGGED_TEXT>>": _jinja(VALIDATION_EXCERPT_VAR),
            "<<VALIDATION_SKELETON>>": _jinja(COL_VALIDATION_SKELETON),
            "<<LABEL_EXAMPLES>>": _format_label_examples(
                labels,
                examples_by_label=examples_by_label,
                configured_examples=configured_examples,
            ),
            "<<DATA_SUMMARY>>": context_section,
        },
    )


# Validator worked example: (id, label the decision depends on, decision, proposed_label, reason).
_VALIDATION_EXAMPLE_DECISIONS: tuple[tuple[str, str | None, str, str, str], ...] = (
    ("c1", None, "drop", "", "placeholder not actual name"),
    ("c2", "first_name", "keep", "", "real first name"),
    ("c3", "age", "keep", "", "age quasi-identifier"),
    ("c4", "city", "reclass", "city", "city not country"),
    ("c5", None, "drop", "", "part of the city name, not a person"),
    ("c6", "state", "keep", "", "state quasi-identifier"),
)


def _validation_example_output(*, labels: list[str], excluded: set[str]) -> str:
    """Example decisions that never keep or reclass to a label the prompt does not permit."""
    permitted = set(labels) - excluded
    decisions = []
    for decision_id, needs, decision, proposed, reason in _VALIDATION_EXAMPLE_DECISIONS:
        if needs is not None and needs not in permitted:
            decision, proposed, reason = "drop", "", f"{needs} is not an allowed label here"
        decisions.append({"id": decision_id, "decision": decision, "proposed_label": proposed, "reason": reason})
    output = json.dumps({"decisions": decisions}, ensure_ascii=False)
    if all(item["decision"] == "drop" for item in decisions):
        return f"{output}\n(None of these labels is allowed here, so every candidate is dropped.)"
    return output


_AUGMENT_EXAMPLE_TEXT = "Jane Doe lives in Santa Clara. She works full-time."
_AUGMENT_EXAMPLE_ENTITIES: tuple[dict[str, str], ...] = (
    {
        "value": "Jane",
        "context_before": "",
        "context_after": " Doe lives in ",
        "label": "first_name",
        "reason": "first name",
    },
    {
        "value": "Doe",
        "context_before": "Jane ",
        "context_after": " lives in Santa Clara. ",
        "label": "last_name",
        "reason": "last name",
    },
    {
        "value": "Santa Clara",
        "context_before": "Doe lives in ",
        "context_after": ". She works full-time.",
        "label": "city",
        "reason": "city",
    },
    {
        "value": "full-time",
        "context_before": "in Santa Clara. She works ",
        "context_after": ".",
        "label": "employment_status",
        "reason": "employment status",
    },
)

# Label-specific guidance, rendered only when the label is active.
_AUGMENT_LABEL_RULES: dict[str, tuple[str, ...]] = {
    "date": (
        '- date includes complete and partial calendar dates; a day, month, and year are NOT all required. Include named months, calendar years, month/year combinations, and days of the month when context establishes calendar meaning (e.g., "in March", "in 2024", "in March 2024", "on the 15th"). Standalone numbers without date context, "March forward", "2024 items", and vague time words such as "today", "now", and "soon" are not dates.',
    ),
    "unique_id": (
        '- unique_id rules: unique_id\'s are never only one character long (ex: "6" is not a unique_id). '
        'They are never event names or real words all strung together (ex: "ProcessManagementEvent" is NOT a unique_id).',
    ),
    "ipv4": (
        "- ipv4 label: This includes specific IP addresses as well as internal IP subnet ranges "
        "(ex: subnet=[224.0.0.0/4, 10.0.0.0/8]).",
    ),
    "filename": (
        '- Filename Exclusions: The "filename" label should be reserved for user-created documents or data exports '
        "(e.g., .pdf, .xlsx, .csv, .txt).",
        "- Executable Distinction: Do not tag executable binaries (ending in .exe, .dll, or .sys) as filename. "
        "Treat these extensions as non-sensitive system identifiers.",
    ),
}


def _augment_example_block(*, labels: list[str], strict_labels: bool, excluded: set[str]) -> str:
    """Build the worked example from labels the prompt actually allows.

    Strict mode shows only entities whose label is in the allowed list, so the
    example never demonstrates a label the model is told not to use. Permissive
    mode may demonstrate labels outside the list, but never an excluded one.
    """
    if strict_labels:
        active = set(labels)
        entities = [entity for entity in _AUGMENT_EXAMPLE_ENTITIES if entity["label"] in active]
    else:
        entities = [entity for entity in _AUGMENT_EXAMPLE_ENTITIES if entity["label"] not in excluded]
    if strict_labels:
        header = "Example (return only entities that fit an allowed label; skip everything else):"
        if not entities:
            header = "Example (none of these entities fits the allowed labels, so nothing is returned):"
    else:
        header = "Example:"
    output = json.dumps({"entities": entities}, ensure_ascii=False)
    return f"{header}\nInput text: {_AUGMENT_EXAMPLE_TEXT}\nOutput: {output}"


def _get_augment_prompt(
    *,
    data_summary: str | None,
    labels: list[str],
    strict_labels: bool = False,
    configured_examples: dict[str, list[str]] | None = None,
    excluded_entity_labels: list[str] | None = None,
) -> str:
    if strict_labels:
        label_block = (
            "Here are the allowed entity classes. Use ONLY labels from this list:\n"
            "<<VALID_CLASSES>>\n\n"
            "Do not create new labels. If an entity does not fit any label in the list, skip it."
        )
    else:
        label_block = (
            "Here are the known entity classes. Strongly prefer labels from this list when they fit:\n"
            "<<VALID_CLASSES>>\n\n"
            "If no known label fits, create a concise snake_case label (e.g., clinic_name, server_name, transaction_id)."
        )
    excluded = sorted(normalize_labels(excluded_entity_labels))
    if excluded:
        label_block += (
            "\n\nDo not return entities of these excluded types or close synonyms of them: "
            + ", ".join(_prompt_safe_configured_value(label) for label in excluded)
            + "."
        )
    example_block = _augment_example_block(labels=labels, strict_labels=strict_labels, excluded=set(excluded))
    rule_lines = [line for label in labels for line in _AUGMENT_LABEL_RULES.get(label, ())]
    rules_section = "Other information:\n" + "\n".join(rule_lines) + "\n\n" if rule_lines else ""

    disguised_hints_block = (
        "- Identifiers may be disguised, fragmented, hyphenated, misspelled, obfuscated,\n"
        "  spaced out, mixed with punctuation, or written in words instead of digits.\n"
        "  Detect the identifier and extract the exact text as written.\n"
        "  Examples of disguised identifiers to detect:\n"
        '  - Any identifier spoken as digit words, including "o" or "oh" used in place of zero:\n'
        '    "nine o two, five five five, one two three four"\n'
        "  - Values spelled out letter by letter with hyphens or commas:\n"
        '    "J-O-H-N", "M, A, R, Y"\n'
    )

    prompt = f"""Task: Find sensitive entities in text.

Use surrounding text and field names to determine what a value actually represents. Add it only when the context supports both its entity label and its sensitivity under the guidance in this prompt.<<SENSITIVITY_GUIDANCE>>
- Direct identifiers: Uniquely identify entities (names, emails, IDs), records (transaction IDs, case numbers), resources (file paths, URLs), or instances (server names, hostnames)
- Quasi-identifiers: Attributes that combine to narrow specificity (age, location, job title, timestamps, technical specs)
- Technical secrets: Credentials (passwords, API keys, tokens), access (internal URLs, endpoints), proprietary terms

<<DATA_SUMMARY>><<LABEL_BLOCK>>

Rules:
- Tag actual values, not placeholders
- In structured formats, distinguish between:
  - Syntax/metadata: field names, column headers, function names, keywords
  - Data: assigned values, cell contents, literals, user input
  Only tag data, never syntax

PERSONAL NAMES:
- Find personal names even when they are short or written in lowercase, uppercase, or mixed case; capitalization alone neither proves nor rules out a name.
- Personal initials, with or without a period, can identify people in signatures, greetings, name sequences, or running text where they stand in for a person (e.g., "the lease was signed by K.M. on the tenant's behalf"). Do not skip an initial solely because no surname follows.
- Use context and the permitted labels to distinguish given names from surnames; do not infer that role solely from word order. Never expand an initial into an inferred name.
- In structured data (JSON, forms, tables), a value in a name field is a name even when it is short (e.g., "surname": "Ng"); never extract the field name itself.
- Do not extract variables, grades, units, list markers, ordinary words, or chat abbreviations as names when they are not being used as personal names.
- Examples: "First initial: Q." and "hi rose, can you review?" support personal-name uses; "a rose in the garden", "Grade: A", "4 m", and "thx for the help" do not.

<<RULES_SECTION>>Additional extraction requirements:
- The "value" field must be the EXACT verbatim span from the input text.
  Copy the text character-for-character exactly as it appears.
  Do NOT normalize, correct spelling, expand abbreviations, decode encodings,
  infer hidden values, translate text, reformat numbers, or otherwise modify
  the extracted span.
- Extract only text that is explicitly present in the input.
  Never reconstruct, guess, or generate a value that does not appear verbatim.
- Return "context_before" and "context_after" as exact, immediately adjacent
  substrings from the underlying source text. Aim for a few words (two to five)
  on each side, or a comparably informative phrase for text without whitespace
  boundaries. Near a source boundary, use whatever context is available.
- Context does not have to identify a unique occurrence. Do not add extra words
  just to make it unique; the engine accepts every exact contextual match.
- At least one context field must be non-empty unless "value" is the entire
  source text. Do not return offsets; the engine derives them from the exact
  value and context.
{disguised_hints_block}
<<EXAMPLE_BLOCK>>

---
Input text: <<INPUT_TEXT>>
"""
    # Pre-substitute nested placeholders inside the block strings before
    # passing them into the single-pass substitution of the main prompt.
    configured_examples = configured_examples or {}
    rendered_labels = [
        _prompt_safe_configured_value(label) if label in configured_examples else label for label in labels
    ]
    label_block = label_block.replace("<<VALID_CLASSES>>", ", ".join(rendered_labels))
    active_configured_examples = {
        label: list(examples) for label, examples in configured_examples.items() if label in labels and examples
    }
    if active_configured_examples:
        configured_lines = [
            f"- {_prompt_safe_configured_value(label)}: "
            + ", ".join(_prompt_safe_configured_value(example) for example in examples)
            for label, examples in active_configured_examples.items()
        ]
        label_block += (
            "\n\nUser-configured positive examples for finding missed entities:\n"
            + "\n".join(configured_lines)
            + "\nThese examples are guidance, not an exhaustive format allowlist."
        )
    sensitivity_guidance = (
        "\n\nThe data summary may describe the dataset and provide sensitivity rules. "
        "Follow its explicit sensitivity rules when they conflict with the generic guidance or examples below. "
        "For categories or situations it does not address, use the generic guidance below. "
        "Do not infer that a value is non-sensitive merely because the data summary does not mention it. "
        "An allowed entity label defines what can be detected; "
        "it does not mean every instance of that category is sensitive.\n\n"
        "Add only values that are sensitive in the context shown, "
        "applying explicit sensitivity rules from the data summary where provided "
        "and otherwise using the generic guidance below. "
        "Do not add a value merely because it matches an allowed label, "
        "uniquely identifies a technical resource, or is a date, timestamp, location, URL, or organization name.\n\n"
        "Use the following categories to guide your search, "
        "subject to explicit sensitivity rules in the data summary:"
        if data_summary
        else "\n\nFocus on:"
    )
    context_section = f"Data summary: {data_summary}\n\n" if data_summary else ""
    return substitute_placeholders(
        prompt,
        {
            "<<SENSITIVITY_GUIDANCE>>": sensitivity_guidance,
            "<<LABEL_BLOCK>>": label_block,
            "<<RULES_SECTION>>": rules_section,
            "<<EXAMPLE_BLOCK>>": example_block,
            "<<INPUT_TEXT>>": _jinja(COL_TEXT),
            "<<DATA_SUMMARY>>": context_section,
        },
    )


def _get_latent_prompt(
    *,
    data_summary: str | None,
    privacy_goal: PrivacyGoal | None,
    excluded_entity_labels: list[str] | None = None,
) -> str:
    summary_line = data_summary.strip() if data_summary else "Not provided"
    privacy_goal_text = _format_privacy_goal(privacy_goal)
    excluded_labels = sorted(normalize_labels(excluded_entity_labels))
    exclusion_block = (
        "\n<excluded_entity_labels>\n"
        f"Do NOT return latent entities with these labels: {', '.join(excluded_labels)}.\n"
        "</excluded_entity_labels>\n"
        if excluded_labels
        else ""
    )
    prompt = """You are performing: LATENT ENTITY & INFERENCE ANALYSIS for privacy protection.

The text will be rewritten according to this privacy goal: <<PRIVACY_GOAL>>
<<EXCLUSION_BLOCK>>

Goal: Identify sensitive information that is NOT explicitly stated in the text, \
but is reasonably inferable from context and could materially increase re-identification \
risk or reveal sensitive attributes about a real person. Treat inference as a first-class \
privacy surface; do not assume removing explicit identifiers is sufficient.

Return only inferences that would help an adversary narrow down or recognize the subject in the real world, or would reveal sensitive attributes about them.

Examples of *eligible* latent content when strongly supported by contextual evidence:
- Identity linkage: specific employer, school, institution, unique role, affiliation group, recognizable relationship
- Where/when linkage: home area, prior location, travel pattern, event timeframe, distinctive routine
- Personal characteristics: gender, marital status (e.g. 'married', 'single', 'divorced', 'seperated'), date of birth,
  race/ethnicity, age, education level, employment status, occupation
- Sensitive attributes: health condition/procedure/medication, legal situation, immigration status, assault/violence, substance use, etc.
- Medical procedures, screenings, or anatomy-specific tests may imply biological sex or gender. These are valid latent inferences
  when strongly supported by clinical context (e.g., Pap smear → female).
- Derived attributes inferred from contextual signals.

Non-examples (exclude):
- Generic domain nouns that do not narrow identity (e.g., "a company", "a school", "a clinic", "a court", "a neighborhood")
- Broad facts that are not person-linking or sensitive (e.g., "consistent blood levels")
- Anything explicitly stated verbatim in the text

Threat model:
Assume an adversary who can read the text in full, has general domain knowledge and access to public information, and may have partial prior familiarity with the subject.

Data type summary:
<<DATA_SUMMARY>>

Input text (identifiers already tagged inline):
---
<<TAGGED_TEXT>>
---

Rules (strict)
1) True latent only:
{%- if <<TAG_NOTATION>> == "xml" -%}
   - Do NOT repeat any already-tagged entities, e.g., <first_name>Alex</first_name>.
{%- elif <<TAG_NOTATION>> == "bracket" -%}
   - Do NOT repeat any already-tagged entities, e.g., [[Alex|first_name]].
{%- elif <<TAG_NOTATION>> == "paren" -%}
   - Do NOT repeat any already-tagged entities, e.g., ((SENSITIVE:first_name|Alex)).
{%- else -%}
   - Do NOT repeat any already-tagged entities, e.g., <<SENSITIVE:first_name>>Alex<</SENSITIVE:first_name>>.
{%- endif -%}
   - Do NOT output verbatim spans from the text.
   - You MAY output structured attributes that are logically implied by the text,
     even if the exact phrase does not appear.
   Derived attributes are allowed when they summarize or normalize multiple
   contextual cues into a structured attribute useful for identification.
   Examples:
   - “my husband passed last year” → marital_status = widowed
   - references to "undergrad", "college", or "university" → bachelor's degree
   - finance-market cues (Bloomberg screens, earnings season, roadshow, closing bell)
     → works in finance / financial analyst role
   These inferred attributes must not simply restate the SAME value already
   tagged in the text. However, normalizing or summarizing tagged information
   into a higher-level structured attribute (e.g., "college/undergrad" →
   bachelor's degree) is allowed.
2) Evidence-bounded inference:
   - Every latent entity MUST include 1-2 short quotes from the text as evidence.
   - Do NOT "jump" to a specific named entity unless the evidence pins it down strongly.
   - If multiple plausible specifics exist, DO NOT guess: either generalize (e.g., "a major Boston cancer center") or omit.
3) High signal only:
   - Prefer returning an EMPTY list over generic filler.
   Attributes such as occupation, education level, immigration status,
   ethnicity, language ability, and institutional affiliation may be
   included when strongly supported, even if individually common,
   because they become identifying when intersected.
   Education inferences are eligible when strongly supported, including
   normalized education attributes such as undergraduate degree or bachelor's-level education.
4) Non-redundant:
   - Do not output multiple variants of the same inference.
Quality checks before finalizing
- Remove any item whose value is too generic to help an adversary (e.g., "a company", "a school", "a clinic", "a court", "a neighborhood") unless additional context would allow a real person to be narrowed down.
- Remove any item that restates explicit text.
- Remove any item without clear evidence quotes.

Now produce the JSON for the input.
"""
    return substitute_placeholders(
        prompt,
        {
            "<<TAG_NOTATION>>": COL_TAG_NOTATION,
            "<<PRIVACY_GOAL>>": privacy_goal_text,
            "<<DATA_SUMMARY>>": summary_line,
            "<<TAGGED_TEXT>>": _jinja(COL_TAGGED_TEXT),
            "<<EXCLUSION_BLOCK>>": exclusion_block,
        },
    )


def _format_privacy_goal(privacy_goal: PrivacyGoal | None) -> str:
    if privacy_goal is None:
        return "Not provided"
    return privacy_goal.to_prompt_string()
