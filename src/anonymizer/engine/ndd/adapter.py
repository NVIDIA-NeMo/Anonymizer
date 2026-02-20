# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from data_designer.config.column_types import ColumnConfigT
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.models import ModelConfig, ModelProvider, load_model_configs
from data_designer.config.seed import SamplingStrategy
from data_designer.config.seed_source import LocalFileSeedSource
from data_designer.config.utils.io_helpers import load_config_file
from data_designer.interface.data_designer import DataDesigner

if TYPE_CHECKING:
    import pandas as pd


RECORD_ID_COLUMN = "_anonymizer_record_id"


@dataclass(frozen=True)
class FailedRecord:
    """A record that did not appear in workflow output."""

    record_id: str
    step: str
    reason: str


@dataclass(frozen=True)
class WorkflowRunResult:
    """Result of a single NDD workflow execution."""

    dataframe: pd.DataFrame
    failed_records: list[FailedRecord]


class NddAdapter:
    """Adapter for running NDD workflows with uniform I/O and record tracking."""

    def __init__(self, data_designer: DataDesigner) -> None:
        self._data_designer = data_designer

    def run_workflow(
        self,
        df: pd.DataFrame,
        *,
        model_configs: list[ModelConfig] | str | Path,
        model_providers: list[ModelProvider] | str | Path | None,
        columns: list[ColumnConfigT],
        workflow_name: str,
        preview_num_records: int | None = None,
    ) -> WorkflowRunResult:
        """Run one workflow and return output with missing-record tracking."""
        workflow_input_df = self._attach_record_ids(df=df)
        workflow_data_designer = self._resolve_data_designer(model_providers=model_providers)
        resolved_model_configs = load_model_configs(model_configs)

        with tempfile.TemporaryDirectory(prefix=f"anonymizer_{workflow_name}_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            seed_path = tmp_path / "seed.parquet"
            workflow_input_df.to_parquet(seed_path, index=False)

            config_builder = DataDesignerConfigBuilder(model_configs=resolved_model_configs)
            config_builder.with_seed_dataset(
                LocalFileSeedSource(path=str(seed_path)),
                sampling_strategy=SamplingStrategy.ORDERED,
            )
            for column in columns:
                config_builder.add_column(column)

            if preview_num_records is None:
                run_results = workflow_data_designer.create(
                    config_builder,
                    num_records=len(workflow_input_df),
                    dataset_name=workflow_name,
                )
                output_df = run_results.load_dataset()
            else:
                preview_results = workflow_data_designer.preview(
                    config_builder,
                    num_records=preview_num_records,
                )
                if preview_results.dataset is None:
                    output_df = workflow_input_df.iloc[0:0].copy()
                else:
                    output_df = preview_results.dataset

        failed_records = self._detect_missing_records(
            workflow_name=workflow_name,
            input_df=(
                workflow_input_df.iloc[:preview_num_records].copy()
                if preview_num_records is not None
                else workflow_input_df
            ),
            output_df=output_df,
        )
        return WorkflowRunResult(dataframe=output_df, failed_records=failed_records)

    def _resolve_data_designer(self, model_providers: list[ModelProvider] | str | Path | None) -> DataDesigner:
        if model_providers is None:
            return self._data_designer

        resolved_providers = self._load_model_providers(model_providers)
        return DataDesigner(
            artifact_path=self._data_designer._artifact_path,
            model_providers=resolved_providers,
            secret_resolver=self._data_designer.secret_resolver,
        )

    def _load_model_providers(
        self,
        model_providers: list[ModelProvider] | str | Path,
    ) -> list[ModelProvider]:
        if isinstance(model_providers, list):
            if all(isinstance(provider, ModelProvider) for provider in model_providers):
                return model_providers
            raise ValueError("model_providers list must contain ModelProvider objects only.")

        config_dict = load_config_file(model_providers)
        raw_providers = config_dict.get("providers")
        if not isinstance(raw_providers, list):
            raise ValueError("model_providers YAML must contain a top-level 'providers' list.")
        return [ModelProvider.model_validate(provider) for provider in raw_providers]

    def _attach_record_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        if RECORD_ID_COLUMN in df.columns:
            return df.copy()

        dataframe = df.copy()
        dataframe[RECORD_ID_COLUMN] = dataframe.apply(self._compute_record_id, axis=1)
        return dataframe

    @staticmethod
    def _compute_record_id(row: pd.Series) -> str:
        serialized = json.dumps(row.to_dict(), default=str, sort_keys=True, separators=(",", ":"))
        return uuid.uuid5(uuid.NAMESPACE_URL, serialized).hex

    def _detect_missing_records(
        self,
        *,
        workflow_name: str,
        input_df: pd.DataFrame,
        output_df: pd.DataFrame,
    ) -> list[FailedRecord]:
        if RECORD_ID_COLUMN not in input_df.columns:
            return []
        if RECORD_ID_COLUMN not in output_df.columns:
            return [
                FailedRecord(
                    record_id=record_id,
                    step=workflow_name,
                    reason=f"Output is missing required tracking column '{RECORD_ID_COLUMN}'",
                )
                for record_id in input_df[RECORD_ID_COLUMN].astype(str).tolist()
            ]

        input_ids = set(input_df[RECORD_ID_COLUMN].astype(str).tolist())
        output_ids = set(output_df[RECORD_ID_COLUMN].astype(str).tolist())
        missing_ids = sorted(input_ids - output_ids)
        return [
            FailedRecord(
                record_id=record_id,
                step=workflow_name,
                reason="Record missing from workflow output",
            )
            for record_id in missing_ids
        ]
