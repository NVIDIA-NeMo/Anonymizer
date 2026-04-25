# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from data_designer.config.column_types import ColumnConfigT
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.models import ModelConfig
from data_designer.config.seed import SamplingStrategy
from data_designer.config.seed_source import LocalFileSeedSource

from anonymizer.interface.errors import AnonymizerWorkflowError

if TYPE_CHECKING:
    import pandas as pd
    from data_designer.interface.data_designer import DataDesigner

logger = logging.getLogger("anonymizer.ndd")

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
        logger.debug("NDD adapter: artifact_path=%s", getattr(data_designer, "_artifact_path", "unknown"))

    def run_workflow(
        self,
        df: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        columns: list[ColumnConfigT],
        workflow_name: str,
        preview_num_records: int | None = None,
    ) -> WorkflowRunResult:
        """Run one workflow and return output with missing-record tracking."""
        workflow_input_df = self._attach_record_ids(df=df)
        logger.debug("NDD workflow '%s' starting with %d records", workflow_name, len(workflow_input_df))
        col_names = [c.name for c in columns]
        logger.debug("NDD workflow '%s': %d columns %s", workflow_name, len(col_names), col_names)
        model_aliases = [m.alias for m in model_configs]

        with tempfile.TemporaryDirectory(prefix=f"anonymizer_{workflow_name}_") as tmp_dir:
            seed_path = str(Path(tmp_dir) / "seed.parquet")
            seed_source = LocalFileSeedSource.from_dataframe(workflow_input_df, seed_path)

            config_builder = DataDesignerConfigBuilder(model_configs=model_configs)
            config_builder.with_seed_dataset(seed_source, sampling_strategy=SamplingStrategy.ORDERED)
            for column in columns:
                config_builder.add_column(column)

            record_count = (
                min(preview_num_records, len(workflow_input_df))
                if preview_num_records is not None
                else len(workflow_input_df)
            )
            try:
                if preview_num_records is None:
                    run_results = self._data_designer.create(
                        config_builder,
                        num_records=len(workflow_input_df),
                        dataset_name=workflow_name,
                    )
                    output_df = run_results.load_dataset()
                else:
                    preview_results = self._data_designer.preview(
                        config_builder,
                        num_records=record_count,
                    )
                    if preview_results.dataset is None:
                        output_df = workflow_input_df.iloc[0:0].copy()
                    else:
                        output_df = preview_results.dataset
            except Exception as exc:
                logger.warning(
                    "Workflow failed for %d input record(s) on model(s) %s: %s",
                    record_count,
                    model_aliases,
                    exc,
                )
                logger.debug(
                    "Workflow '%s' failure context: columns=%s",
                    workflow_name,
                    col_names,
                )
                raise AnonymizerWorkflowError(f"Workflow failed: {exc}") from exc

        logger.debug("NDD workflow '%s' returned %d records", workflow_name, len(output_df))
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

    def _attach_record_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        if RECORD_ID_COLUMN in df.columns:
            return df.copy()

        dataframe = df.copy()
        dataframe[RECORD_ID_COLUMN] = dataframe.apply(self._compute_record_id, axis=1)
        return dataframe

    @staticmethod
    def _compute_record_id(row: pd.Series) -> str:
        # uuid5 is deterministic so input/output IDs match for missing-record tracking.
        # Include row index so duplicate rows get distinct IDs.
        # TODO: consider whether id_column from AnonymizerInput should be used here instead.
        row_data = {**row.to_dict(), "__row_idx__": row.name}
        serialized = json.dumps(row_data, default=str, sort_keys=True, separators=(",", ":"))
        return uuid.uuid5(uuid.NAMESPACE_URL, serialized).hex

    def _detect_missing_records(
        self,
        *,
        workflow_name: str,
        input_df: pd.DataFrame,
        output_df: pd.DataFrame,
    ) -> list[FailedRecord]:
        if RECORD_ID_COLUMN not in input_df.columns:
            logger.warning(
                "Missing-record detection skipped: input DataFrame lacks the record-tracking "
                "column, so the adapter cannot verify whether any of %d input record(s) were "
                "dropped. This indicates an internal invariant violation - `_attach_record_ids` "
                "was not called on this DataFrame before detection.",
                len(input_df),
            )
            logger.debug(
                "Workflow '%s' detection skipped: input missing '%s'; input_columns=%s",
                workflow_name,
                RECORD_ID_COLUMN,
                list(input_df.columns),
            )
            return []
        if RECORD_ID_COLUMN not in output_df.columns:
            input_cols = set(input_df.columns)
            output_cols = set(output_df.columns)
            other_dropped = sorted((input_cols - output_cols) - {RECORD_ID_COLUMN})
            added = sorted(output_cols - input_cols)
            logger.warning(
                "Missing-record detection disabled: workflow output does not contain "
                "the record-tracking column, so all %d input record(s) are being marked as "
                "failed. This typically means seed-column pass-through is disabled or a "
                "user-supplied column config overwrote it. Other input columns that were "
                "also dropped: %s. Columns added by the workflow: %s.",
                len(input_df),
                other_dropped,
                added,
            )
            logger.debug(
                "Workflow '%s' detection disabled: output missing '%s'; input_columns=%s output_columns=%s",
                workflow_name,
                RECORD_ID_COLUMN,
                list(input_df.columns),
                list(output_df.columns),
            )
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
