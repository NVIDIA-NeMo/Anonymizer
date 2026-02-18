# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from data_designer.config.models import ModelProvider
from data_designer.config.utils.io_helpers import load_config_file
from data_designer.interface.data_designer import DataDesigner

from anonymizer.config.anonymizer_config import (
    AnonymizerConfig,
    AnonymizerInput,
)
from anonymizer.config.replace_strategies import LLMReplace
from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
from anonymizer.engine.io.reader import read_input
from anonymizer.engine.ndd.adapter import NddAdapter
from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceWorkflow
from anonymizer.engine.replace.replace_runner import ReplaceRunner
from anonymizer.interface.results import AnonymizerResult, PreviewResult

if TYPE_CHECKING:
    import pandas as pd
    from data_designer.config.models import ModelConfig


DEFAULT_MODEL_CONFIGS_PATH = Path(__file__).resolve().parents[1] / "config" / "model_configs" / "models.yaml"


class Anonymizer:
    """Public facade for anonymization workflows."""

    def __init__(
        self,
        *,
        data_designer: DataDesigner | None = None,
        artifact_path: str | Path = ".anonymizer-artifacts",
        model_providers: list["ModelProvider"] | str | Path | None = None,
        detection_workflow: EntityDetectionWorkflow | None = None,
        replace_runner: ReplaceRunner | None = None,
    ) -> None:
        self._data_designer = data_designer or DataDesigner(
            artifact_path=Path(artifact_path),
            model_providers=_resolve_model_providers(model_providers),
        )
        self._adapter = NddAdapter(data_designer=self._data_designer)
        self._detection_workflow = detection_workflow or EntityDetectionWorkflow(adapter=self._adapter)
        self._replace_runner = replace_runner or ReplaceRunner(llm_workflow=LlmReplaceWorkflow(adapter=self._adapter))

    def run(
        self,
        *,
        config: AnonymizerConfig,
        data: pd.DataFrame | AnonymizerInput,
        model_configs: list[ModelConfig] | str | Path | None = None,
        model_providers: list[ModelProvider] | str | Path | None = None,
    ) -> AnonymizerResult:
        """Run full anonymization workflow."""
        return self._run_internal(
            config=config,
            data=data,
            model_configs=model_configs,
            model_providers=model_providers,
            preview_num_records=None,
        )

    def preview(
        self,
        *,
        config: AnonymizerConfig,
        data: pd.DataFrame | AnonymizerInput,
        num_records: int = 10,
        model_configs: list[ModelConfig] | str | Path | None = None,
        model_providers: list[ModelProvider] | str | Path | None = None,
    ) -> PreviewResult:
        """Run preview mode on a limited number of records."""
        result = self._run_internal(
            config=config,
            data=data,
            model_configs=model_configs,
            model_providers=model_providers,
            preview_num_records=num_records,
        )
        return PreviewResult(
            dataframe=result.dataframe,
            failed_records=result.failed_records,
            preview_num_records=num_records,
        )

    def _run_internal(
        self,
        *,
        config: AnonymizerConfig,
        data: pd.DataFrame | AnonymizerInput,
        model_configs: list[ModelConfig] | str | Path | None,
        model_providers: list[ModelProvider] | str | Path | None,
        preview_num_records: int | None,
    ) -> AnonymizerResult:
        input_df = read_input(data)
        resolved_model_configs: list[ModelConfig] | str | Path = model_configs or DEFAULT_MODEL_CONFIGS_PATH

        detection_result = self._detection_workflow.run(
            input_df,
            model_configs=resolved_model_configs,
            model_providers=model_providers,
            selected_models=config.selected_models.detection,
            gliner_detection_threshold=config.gliner_detection_threshold,
            entity_labels=config.entity_labels,
            privacy_goal=config.privacy_goal,
            data_summary=config.data_summary,
            tag_latent_entities=config.rewrite is not None,
            preview_num_records=preview_num_records,
        )

        replaced_df, replace_failures = self._replace_runner.run(
            detection_result.dataframe,
            replace_strategy=config.replace,
            model_configs=resolved_model_configs,
            model_providers=model_providers,
            selected_models=config.selected_models.replace,
            preview_num_records=preview_num_records,
        )

        if isinstance(config.replace, LLMReplace):
            # LLM replacement path emits replacement maps used to produce replaced_text.
            pass

        return AnonymizerResult(
            dataframe=replaced_df,
            failed_records=[*detection_result.failed_records, *replace_failures],
        )


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
