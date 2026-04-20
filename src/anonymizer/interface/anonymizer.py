# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from data_designer.config.models import ModelProvider
from data_designer.config.utils.io_helpers import load_config_file
from data_designer.interface.data_designer import DataDesigner

from anonymizer.config.anonymizer_config import (
    AnonymizerConfig,
    AnonymizerInput,
)
from anonymizer.config.replace_strategies import Substitute
from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_DETECTED_ENTITIES,
    COL_FINAL_ENTITIES,
    COL_LEAKAGE_MASS,
    COL_NEEDS_HUMAN_REVIEW,
    COL_REPLACED_TEXT,
    COL_REWRITTEN_TEXT,
    COL_TAGGED_TEXT,
    COL_TEXT,
    COL_UTILITY_SCORE,
    COL_WEIGHTED_LEAKAGE_RATE,
    DEFAULT_ENTITY_LABELS,
)
from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
from anonymizer.engine.io.reader import read_input
from anonymizer.engine.ndd.adapter import NddAdapter
from anonymizer.engine.ndd.model_loader import parse_model_configs, validate_model_alias_references
from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceWorkflow
from anonymizer.engine.replace.replace_runner import ReplacementWorkflow
from anonymizer.engine.rewrite.rewrite_workflow import RewriteWorkflow
from anonymizer.interface.errors import InvalidConfigError
from anonymizer.interface.results import AnonymizerResult, PreviewResult
from anonymizer.logging import LOG_INDENT, configure_logging, reapply_log_levels

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger("anonymizer")


def _initialize_logging() -> None:
    """Run one-time logging setup if the user hasn't already called configure_logging()."""
    from anonymizer import logging as _logging_mod

    if _logging_mod._configured:
        return
    configure_logging()


class Anonymizer:
    """Public facade for anonymization workflows."""

    def __init__(
        self,
        *,
        model_configs: str | Path | None = None,
        model_providers: list[ModelProvider] | str | Path | None = None,
        artifact_path: str | Path | None = None,
        data_designer: DataDesigner | None = None,
        detection_workflow: EntityDetectionWorkflow | None = None,
        replace_runner: ReplacementWorkflow | None = None,
        rewrite_runner: RewriteWorkflow | None = None,
    ) -> None:
        """Create an Anonymizer instance.

        Args:
            model_configs: Unified YAML (string or file path) defining the model
                pool and optional ``selected_models`` overrides. ``None`` uses
                bundled defaults. See ``default_model_configs/README.md``.
            model_providers: Provider definitions (list, YAML string, or file path).
                Each provider maps a name to an endpoint and API key.
            artifact_path: Directory for intermediate artifacts. Defaults to
                ``.anonymizer-artifacts``.
            data_designer: Pre-configured DataDesigner instance (advanced usage).
            detection_workflow: Custom detection workflow (advanced/testing).
            replace_runner: Custom replacement workflow (advanced/testing).
            rewrite_runner: Custom rewrite workflow (advanced/testing).
        """
        _initialize_logging()
        resolved_artifact_path = Path(artifact_path or ".anonymizer-artifacts")
        parsed = parse_model_configs(model_configs)
        self._model_configs = parsed.model_configs
        self._selected_models = parsed.selected_models
        logger.info("🔧 Anonymizer initialized with %d model configs", len(self._model_configs))
        det = self._selected_models.detection
        logger.info(LOG_INDENT + "🔎 detector:  %s", det.entity_detector)
        logger.info(LOG_INDENT + "✅ validator: %s", det.entity_validator)
        logger.info(LOG_INDENT + "🧩 augmenter: %s", det.entity_augmenter)

        if data_designer is not None:
            self._data_designer = data_designer
        else:
            self._data_designer = DataDesigner(
                artifact_path=resolved_artifact_path,
                model_providers=_resolve_model_providers(model_providers),
            )
            reapply_log_levels()
        self._adapter = NddAdapter(data_designer=self._data_designer)
        self._detection_workflow = detection_workflow or EntityDetectionWorkflow(adapter=self._adapter)
        self._replace_runner = replace_runner or ReplacementWorkflow(
            llm_workflow=LlmReplaceWorkflow(adapter=self._adapter)
        )
        self._rewrite_runner = rewrite_runner or RewriteWorkflow(adapter=self._adapter)

    def run(
        self,
        *,
        config: AnonymizerConfig,
        data: AnonymizerInput,
    ) -> AnonymizerResult:
        """Run the full anonymization pipeline (detection + replacement).

        Args:
            config: Workflow behavior — replace strategy, entity labels, thresholds.
            data: Input source with file path, text column, and optional data summary.
        """
        self._validate_preflight_config(config)
        input_df = read_input(data)
        return self._run_internal(config=config, data=data, input_df=input_df, preview_num_records=None)

    def preview(
        self,
        *,
        config: AnonymizerConfig,
        data: AnonymizerInput,
        num_records: int = 10,
    ) -> PreviewResult:
        """Run the pipeline on a subset of records for quick inspection.

        Args:
            config: Workflow behavior — replace strategy, entity labels, thresholds.
            data: Input source with file path, text column, and optional data summary.
            num_records: Maximum records to process (default 10).
        """
        self._validate_preflight_config(config)
        input_df = read_input(data, nrows=num_records)
        result = self._run_internal(config=config, data=data, input_df=input_df, preview_num_records=num_records)
        return PreviewResult(
            dataframe=result.dataframe,
            trace_dataframe=result.trace_dataframe,
            failed_records=result.failed_records,
            preview_num_records=num_records,
        )

    def validate_config(self, config: AnonymizerConfig) -> None:
        """Validate that the active workflow config is compatible with model selections."""
        self._validate_preflight_config(config)

    def _run_internal(
        self,
        *,
        config: AnonymizerConfig,
        data: AnonymizerInput,
        input_df: pd.DataFrame,
        preview_num_records: int | None,
    ) -> AnonymizerResult:
        num_records = len(input_df)
        if preview_num_records is not None and preview_num_records != num_records:
            effective_records = min(preview_num_records, num_records)
            if effective_records < preview_num_records:
                logger.info(
                    LOG_INDENT + "🔍 Running entity detection on capped %d records (requested %d, available %d)",
                    effective_records,
                    preview_num_records,
                    num_records,
                )
            else:
                logger.info(
                    LOG_INDENT + "🔍 Running entity detection on %d of %d records", effective_records, num_records
                )
            preview_num_records = effective_records
        else:
            logger.info("🔍 Running entity detection on %d records", num_records)
        if logger.isEnabledFor(logging.DEBUG):
            text_lengths = input_df[COL_TEXT].astype(str).str.len()
            logger.debug(
                "input text lengths: min=%d, max=%d, mean=%.0f chars (%d records)",
                text_lengths.min(),
                text_lengths.max(),
                text_lengths.mean(),
                num_records,
            )
            logger.debug(
                "detection config: threshold=%.2f, labels=%s",
                config.detect.gliner_threshold,
                config.detect.entity_labels
                or f"(default: {len(DEFAULT_ENTITY_LABELS)} labels; see anonymizer.DEFAULT_ENTITY_LABELS for list)",
            )
        else:
            logger.info(
                "detection labels in scope: %s",
                config.detect.entity_labels
                or f"(default: {len(DEFAULT_ENTITY_LABELS)} labels; see anonymizer.DEFAULT_ENTITY_LABELS for list)",
            )

        t0 = time.perf_counter()
        detection_result = self._detection_workflow.run(
            input_df,
            model_configs=self._model_configs,
            selected_models=self._selected_models.detection,
            gliner_detection_threshold=config.detect.gliner_threshold,
            entity_labels=config.detect.entity_labels,
            privacy_goal=config.rewrite.privacy_goal if config.rewrite else None,
            data_summary=data.data_summary,
            tag_latent_entities=config.rewrite is not None,
            compute_grouped_entities=config.replace is not None or config.rewrite is not None,
            preview_num_records=preview_num_records,
        )
        detection_elapsed = time.perf_counter() - t0
        entity_count = _count_entities(detection_result.dataframe)
        detection_failed = len(detection_result.failed_records)
        logger.info(
            LOG_INDENT + "📋 Detection complete — %d entities found across %d records (%d failed) [%.1fs]",
            entity_count,
            len(detection_result.dataframe),
            detection_failed,
            detection_elapsed,
        )
        if COL_DETECTED_ENTITIES in detection_result.dataframe.columns:
            label_counts = _count_labels(detection_result.dataframe[COL_DETECTED_ENTITIES])
            if label_counts:
                summary = ", ".join(f"{label}={count}" for label, count in label_counts.most_common())
                logger.info(LOG_INDENT + "labels: %s", summary)
        if config.replace is not None:
            strategy_name = type(config.replace).__name__
            logger.info("🔄 Running %s replacement", strategy_name)
            t0 = time.perf_counter()
            replace_result = self._replace_runner.run(
                detection_result.dataframe,
                replace_method=config.replace,
                model_configs=self._model_configs,
                selected_models=self._selected_models.replace,
                preview_num_records=preview_num_records,
            )
            replace_elapsed = time.perf_counter() - t0
            final_df = replace_result.dataframe
            post_detection_failures = replace_result.failed_records
            logger.info(
                LOG_INDENT + "📋 Replacement complete (%d failed) [%.1fs]",
                len(post_detection_failures),
                replace_elapsed,
            )
        elif config.rewrite is not None:
            logger.info("✏️ Running rewrite pipeline")
            t0 = time.perf_counter()
            rewrite_result = self._rewrite_runner.run(
                detection_result.dataframe,
                model_configs=self._model_configs,
                selected_models=self._selected_models.rewrite,
                replace_model_selection=self._selected_models.replace,
                privacy_goal=config.rewrite.privacy_goal,
                evaluation=config.rewrite.evaluation,
                data_summary=data.data_summary,
                preview_num_records=preview_num_records,
            )
            rewrite_elapsed = time.perf_counter() - t0
            final_df = rewrite_result.dataframe
            post_detection_failures = rewrite_result.failed_records
            logger.info(
                LOG_INDENT + "📋 Rewrite complete (%d failed) [%.1fs]",
                len(post_detection_failures),
                rewrite_elapsed,
            )
        else:
            final_df = detection_result.dataframe
            post_detection_failures = []

        all_failures = [*detection_result.failed_records, *post_detection_failures]
        if all_failures:
            logger.warning("%d record(s) failed during pipeline processing.", len(all_failures))
            for f in all_failures:
                logger.debug("  %s (%s: %s)", f.record_id, f.step, f.reason)
        renamed_trace = _rename_output_columns(final_df)
        logger.info("🎉 Pipeline complete — %d records processed, %d total failures", num_records, len(all_failures))
        return AnonymizerResult(
            dataframe=_build_user_dataframe(renamed_trace),
            trace_dataframe=renamed_trace,
            failed_records=all_failures,
        )

    def _validate_preflight_config(self, config: AnonymizerConfig) -> None:
        """Run semantic preflight checks shared by validate, run, and preview."""
        try:
            validate_model_alias_references(
                self._model_configs,
                self._selected_models,
                check_substitute=isinstance(config.replace, Substitute) or config.rewrite is not None,
                check_rewrite=config.rewrite is not None,
            )
        except ValueError as exc:
            raise InvalidConfigError(str(exc)) from exc


def _unwrap_entities(raw: object) -> list:
    if isinstance(raw, dict):
        return raw.get("entities", [])
    if isinstance(raw, list):
        return raw
    return getattr(raw, "entities", [])


def _entity_label(entity: object) -> str | None:
    if isinstance(entity, dict):
        return entity.get("label")
    return getattr(entity, "label", None)


def _count_labels_for_row(raw: object) -> Counter[str]:
    counts: Counter[str] = Counter()
    for entity in _unwrap_entities(raw):
        label = _entity_label(entity)
        if label:
            counts[label] += 1
    return counts


def _count_labels(entities_column: pd.Series) -> Counter[str]:
    counts: Counter[str] = Counter()
    for raw in entities_column:
        counts += _count_labels_for_row(raw)
    return counts


def _count_entities(df: pd.DataFrame) -> int:
    if COL_DETECTED_ENTITIES not in df.columns:
        return 0
    return int(df[COL_DETECTED_ENTITIES].apply(lambda raw: len(_unwrap_entities(raw))).sum())


def _resolve_model_providers(
    model_providers: list[ModelProvider] | str | Path | None,
) -> list[ModelProvider] | None:
    if model_providers is None:
        return None
    if isinstance(model_providers, list):
        return model_providers
    if isinstance(model_providers, str) and "\n" not in model_providers:
        candidate = Path(model_providers.strip()).expanduser()
        if candidate.suffix in (".yaml", ".yml"):
            if not candidate.is_file():
                raise FileNotFoundError(f"Providers config file not found: {candidate}")
            model_providers = candidate
    config_dict = load_config_file(model_providers)
    raw_providers = config_dict.get("providers")
    if not isinstance(raw_providers, list):
        raise ValueError("model_providers YAML must contain a top-level 'providers' list.")
    return [ModelProvider.model_validate(provider) for provider in raw_providers]


def _rename_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename internal column names to user-facing names based on the original text column."""
    original_text_column = str(df.attrs["original_text_column"])
    rename_map: dict[str, str] = {}
    if COL_TEXT in df.columns:
        rename_map[COL_TEXT] = original_text_column
    if COL_REPLACED_TEXT in df.columns:
        rename_map[COL_REPLACED_TEXT] = f"{original_text_column}_replaced"
    if COL_TAGGED_TEXT in df.columns:
        rename_map[COL_TAGGED_TEXT] = f"{original_text_column}_with_spans"
    if COL_REWRITTEN_TEXT in df.columns:
        rename_map[COL_REWRITTEN_TEXT] = f"{original_text_column}_rewritten"
    if not rename_map:
        return df
    renamed = df.rename(columns=rename_map)
    renamed.attrs = dict(df.attrs)
    return renamed


def _build_user_dataframe(trace_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Filter trace dataframe to the public column set for the active mode.

    Replace:     {text_col}, {text_col}_replaced, {text_col}_with_spans, final_entities
    Rewrite:     {text_col}, {text_col}_rewritten, utility_score, leakage_mass, weighted_leakage_rate,
                 any_high_leaked, needs_human_review
    Detect-only: {text_col}, {text_col}_with_spans, final_entities
    """
    t = trace_dataframe
    text_col = str(t.attrs["original_text_column"])

    if f"{text_col}_rewritten" in t.columns:
        allowed = {
            text_col,
            f"{text_col}_rewritten",
            COL_UTILITY_SCORE,
            COL_LEAKAGE_MASS,
            COL_WEIGHTED_LEAKAGE_RATE,
            COL_ANY_HIGH_LEAKED,
            COL_NEEDS_HUMAN_REVIEW,
        }
    elif f"{text_col}_replaced" in t.columns:
        allowed = {
            text_col,
            f"{text_col}_replaced",
            f"{text_col}_with_spans",
            COL_FINAL_ENTITIES,
        }
    else:
        allowed = {
            text_col,
            f"{text_col}_with_spans",
            COL_FINAL_ENTITIES,
        }

    user_columns = [col for col in t.columns if col in allowed]
    user_dataframe = t[user_columns].copy()
    user_dataframe.attrs = dict(t.attrs)
    return user_dataframe
