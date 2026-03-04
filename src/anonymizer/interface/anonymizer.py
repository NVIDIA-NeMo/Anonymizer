# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from data_designer.config.models import ModelProvider
from data_designer.config.utils.io_helpers import load_config_file
from data_designer.interface.data_designer import DataDesigner

from anonymizer.config.anonymizer_config import (
    AnonymizerConfig,
    AnonymizerInput,
)
from anonymizer.engine.constants import COL_DETECTED_ENTITIES, COL_REPLACED_TEXT, COL_TAGGED_TEXT, COL_TEXT
from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
from anonymizer.engine.io.reader import read_input
from anonymizer.engine.ndd.adapter import NddAdapter
from anonymizer.engine.ndd.model_loader import parse_model_configs
from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceWorkflow
from anonymizer.engine.replace.replace_runner import ReplacementWorkflow
from anonymizer.interface.results import AnonymizerResult, PreviewResult
from anonymizer.logging import LOG_INDENT, configure_logging

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger("anonymizer")

_logging_initialized = False


def _initialize_logging() -> None:
    """Run one-time logging setup."""
    global _logging_initialized
    if _logging_initialized:
        return
    configure_logging()
    _logging_initialized = True


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
        """
        _initialize_logging()
        resolved_artifact_path = Path(artifact_path or ".anonymizer-artifacts")
        parsed = parse_model_configs(model_configs)
        self._model_configs = parsed.model_configs
        self._selected_models = parsed.selected_models
        logger.info("🔧 Anonymizer initialized with %d model configs", len(self._model_configs))

        self._data_designer = data_designer or DataDesigner(
            artifact_path=resolved_artifact_path,
            model_providers=_resolve_model_providers(model_providers),
        )
        self._adapter = NddAdapter(data_designer=self._data_designer)
        self._detection_workflow = detection_workflow or EntityDetectionWorkflow(adapter=self._adapter)
        self._replace_runner = replace_runner or ReplacementWorkflow(
            llm_workflow=LlmReplaceWorkflow(adapter=self._adapter)
        )

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
        return self._run_internal(config=config, data=data, preview_num_records=None)

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
        result = self._run_internal(config=config, data=data, preview_num_records=num_records)
        return PreviewResult(
            dataframe=result.dataframe,
            trace_dataframe=result.trace_dataframe,
            failed_records=result.failed_records,
            preview_num_records=num_records,
        )

    def _run_internal(
        self,
        *,
        config: AnonymizerConfig,
        data: AnonymizerInput,
        preview_num_records: int | None,
    ) -> AnonymizerResult:
        input_df = read_input(data)
        num_records = len(input_df)
        logger.info("📂 Loaded %d records from %s (column: '%s')", num_records, data.source, data.text_column)

        if preview_num_records is not None:
            logger.info(LOG_INDENT + "👀 Preview mode: processing %d of %d records", preview_num_records, num_records)

        logger.info("🔍 Running entity detection on %d records", num_records)
        detection_result = self._detection_workflow.run(
            input_df,
            model_configs=self._model_configs,
            selected_models=self._selected_models.detection,
            gliner_detection_threshold=config.detect.gliner_threshold,
            entity_labels=config.detect.entity_labels,
            privacy_goal=config.rewrite.privacy_goal if config.rewrite else None,
            data_summary=data.data_summary,
            tag_latent_entities=config.rewrite is not None,
            compute_grouped_entities=True if config.replace is not None else None,
            preview_num_records=preview_num_records,
        )
        entity_count = _count_entities(detection_result.dataframe)
        detection_failed = len(detection_result.failed_records)
        logger.info(
            LOG_INDENT + "✅ Detection complete — %d entities found across %d records (%d failed)",
            entity_count,
            len(detection_result.dataframe),
            detection_failed,
        )

        if config.replace is not None:
            strategy_name = type(config.replace).__name__
            logger.info("🔄 Running %s replacement", strategy_name)
            replace_result = self._replace_runner.run(
                detection_result.dataframe,
                replace_method=config.replace,
                model_configs=self._model_configs,
                selected_models=self._selected_models.replace,
                preview_num_records=preview_num_records,
            )
            final_df = replace_result.dataframe
            replace_failures = replace_result.failed_records
            logger.info(LOG_INDENT + "✅ Replacement complete (%d failed)", len(replace_failures))
        else:
            final_df = detection_result.dataframe
            replace_failures = []

        all_failures = [*detection_result.failed_records, *replace_failures]
        renamed_trace = _rename_output_columns(final_df)
        logger.info("🎉 Pipeline complete — %d records processed, %d total failures", num_records, len(all_failures))
        return AnonymizerResult(
            dataframe=_build_user_dataframe(renamed_trace),
            trace_dataframe=renamed_trace,
            failed_records=all_failures,
        )


def _count_entities(df: pd.DataFrame) -> int:
    if COL_DETECTED_ENTITIES not in df.columns:
        return 0
    return int(df[COL_DETECTED_ENTITIES].apply(len).sum())


def _resolve_model_providers(
    model_providers: list[ModelProvider] | str | Path | None,
) -> list[ModelProvider] | None:
    if model_providers is None:
        return None
    if isinstance(model_providers, list):
        return model_providers
    config_dict = load_config_file(model_providers)
    raw_providers = config_dict.get("providers")
    if not isinstance(raw_providers, list):
        raise ValueError("model_providers YAML must contain a top-level 'providers' list.")
    return [ModelProvider.model_validate(provider) for provider in raw_providers]


def _rename_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename internal column names to user-facing names based on the original text column."""
    original_text_column = str(df.attrs.get("original_text_column", "text"))
    rename_map: dict[str, str] = {}
    if COL_TEXT in df.columns:
        rename_map[COL_TEXT] = original_text_column
    if COL_REPLACED_TEXT in df.columns:
        rename_map[COL_REPLACED_TEXT] = f"{original_text_column}_replaced"
    if COL_TAGGED_TEXT in df.columns:
        rename_map[COL_TAGGED_TEXT] = f"{original_text_column}_with_spans"
    if not rename_map:
        return df
    renamed = df.rename(columns=rename_map)
    renamed.attrs = dict(df.attrs)
    return renamed


def _build_user_dataframe(trace_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Filter trace dataframe to only user-facing columns (already renamed)."""
    original_text_column = str(trace_dataframe.attrs.get("original_text_column", "text"))
    user_visible = {
        original_text_column,
        f"{original_text_column}_replaced",
        f"{original_text_column}_with_spans",
    }
    user_columns = [
        column for column in trace_dataframe.columns if not column.startswith("_") or column in user_visible
    ]
    user_dataframe = trace_dataframe[user_columns].copy()
    user_dataframe.attrs = dict(trace_dataframe.attrs)
    return user_dataframe
