# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owns execution of DataDesigner workflows with uniform I/O, record-id tagging,
and missing-record (`FailedRecord`) tracking.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from data_designer.config.column_types import ColumnConfigT
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.models import ModelConfig
from data_designer.config.seed import SamplingStrategy
from data_designer.config.seed_source import LocalFileSeedSource

from anonymizer.engine.ndd.dd_observability import (
    DataDesignerUsageProbe,
    DDMessageTracePlan,
    _TaskTrace,
    extract_workflow_model_aliases,
    record_dd_task_traces,
    task_traces_from_result,
    temporary_dd_task_trace,
)
from anonymizer.interface.errors import AnonymizerWorkflowError
from anonymizer.measurement import current_collector, record_ndd_workflow

if TYPE_CHECKING:
    import pandas as pd
    from data_designer.interface.data_designer import DataDesigner

__all__ = ["FailedRecord", "NddAdapter", "RECORD_ID_COLUMN", "WorkflowRunResult"]


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
        self._run_lock = RLock()
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
        """Run one workflow and return output with missing-record tracking.

        Wraps a DataFrame slice plus NDD column configs into a DataDesigner
        run and returns `WorkflowRunResult(dataframe, failed_records)`.
        Records missing from the output surface as `FailedRecord` objects
        rather than silently disappearing.

        This is the engine boundary for *executing* DataDesigner workflows.
        Engine sub-workflows declare column configs and call this method;
        they do not call `DataDesigner.create()` or `.preview()` directly.

        Args:
            df: Input DataFrame.
            model_configs: NDD model aliases available to the workflow.
            columns: NDD column configs to add to the workflow.
            workflow_name: Identifier used in logs and on `FailedRecord` entries.
            preview_num_records: If set, run in preview mode against this many rows.

        Returns:
            `WorkflowRunResult` with the output DataFrame and any `FailedRecord` entries.
        """
        workflow_input_df = self._attach_record_ids(df=df)
        logger.debug("NDD workflow '%s' starting with %d records", workflow_name, len(workflow_input_df))
        col_names = [c.name for c in columns]
        logger.debug("NDD workflow '%s': %d columns %s", workflow_name, len(col_names), col_names)
        available_model_aliases = [m.alias for m in model_configs]
        model_aliases = extract_workflow_model_aliases(columns) or available_model_aliases
        record_count = (
            min(preview_num_records, len(workflow_input_df))
            if preview_num_records is not None
            else len(workflow_input_df)
        )
        started = time.perf_counter()
        collector = current_collector()
        trace_plan = DDMessageTracePlan.from_columns(
            columns=columns,
            model_configs=model_configs,
            collector=collector,
        )
        columns = trace_plan.columns
        usage_probe = DataDesignerUsageProbe(
            self._data_designer,
            enabled=collector is not None,
            collector=collector,
            workflow_name=workflow_name,
            private_trace_columns=trace_plan.private_columns,
        )
        trace_plan.record_coverage(workflow_name=workflow_name, collector=collector)

        with tempfile.TemporaryDirectory(prefix=f"anonymizer_{workflow_name}_") as tmp_dir:
            seed_path = str(Path(tmp_dir) / "seed.parquet")
            seed_source = LocalFileSeedSource.from_dataframe(workflow_input_df, seed_path)

            config_builder = DataDesignerConfigBuilder(model_configs=model_configs)
            config_builder.with_seed_dataset(seed_source, sampling_strategy=SamplingStrategy.ORDERED)
            for column in columns:
                config_builder.add_column(column)

            task_traces: list[_TaskTrace] = []
            try:
                with self._run_lock, usage_probe, temporary_dd_task_trace(self._data_designer, collector=collector):
                    if preview_num_records is None:
                        run_results = self._data_designer.create(
                            config_builder,
                            num_records=len(workflow_input_df),
                            dataset_name=workflow_name,
                        )
                        task_traces = task_traces_from_result(run_results)
                        output_df = run_results.load_dataset()
                    else:
                        preview_results = self._data_designer.preview(
                            config_builder,
                            num_records=record_count,
                        )
                        task_traces = task_traces_from_result(preview_results)
                        if preview_results.dataset is None:
                            output_df = workflow_input_df.iloc[0:0].copy()
                        else:
                            output_df = preview_results.dataset
            except Exception as exc:
                logger.warning(
                    "Workflow failed for %d input record(s) on model(s) %s: %s",
                    record_count,
                    available_model_aliases,
                    exc,
                )
                logger.debug(
                    "Workflow '%s' failure context: columns=%s",
                    workflow_name,
                    col_names,
                )
                try:
                    usage_probe.flush_private_trace_records()
                except Exception:
                    logger.warning("Failed to write DataDesigner private message trace records after workflow failure")
                record_ndd_workflow(
                    workflow_name=workflow_name,
                    model_aliases=model_aliases,
                    input_row_count=record_count,
                    seed_row_count=len(workflow_input_df),
                    output_row_count=None,
                    failed_record_count=None,
                    elapsed_sec=time.perf_counter() - started,
                    status="error",
                    preview_num_records=preview_num_records,
                    column_count=len(col_names),
                    column_names=col_names,
                    model_usage=usage_probe.model_usage(),
                )
                raise AnonymizerWorkflowError(f"Workflow failed: {exc}") from exc

            output_df = trace_plan.record_and_strip_native_traces(
                output_df=output_df,
                workflow_name=workflow_name,
                collector=collector,
            )
            record_dd_task_traces(
                workflow_name=workflow_name,
                collector=collector,
                task_traces=task_traces,
            )
            usage_probe.flush_private_trace_records()

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
        record_ndd_workflow(
            workflow_name=workflow_name,
            model_aliases=model_aliases,
            input_row_count=record_count,
            seed_row_count=len(workflow_input_df),
            output_row_count=len(output_df),
            failed_record_count=len(failed_records),
            elapsed_sec=time.perf_counter() - started,
            preview_num_records=preview_num_records,
            column_count=len(col_names),
            column_names=col_names,
            model_usage=usage_probe.model_usage(),
        )
        return WorkflowRunResult(dataframe=output_df, failed_records=failed_records)

    def build_config(
        self,
        df: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        columns: list[ColumnConfigT],
        seed_path: str | Path,
    ) -> DataDesignerConfigBuilder:
        """Assemble (but do NOT execute) the DataDesigner config for a workflow.

        Writes the record-id-tagged input to ``seed_path`` as the seed dataset and
        returns the assembled ``DataDesignerConfigBuilder`` for an *external* executor
        (e.g. an at-scale SLURM orchestrator) to run. This mirrors the config assembly
        in :meth:`run_workflow` without the ``DataDesigner.create()/.preview()`` call,
        so callers can hand the same workflow to a different DataDesigner runtime.

        Args:
            df: Input DataFrame.
            model_configs: NDD model aliases available to the workflow.
            columns: NDD column configs to add to the workflow.
            seed_path: Destination parquet path for the seed dataset (persisted; the
                caller owns its lifetime, unlike ``run_workflow``'s tempdir).

        Returns:
            The assembled ``DataDesignerConfigBuilder`` (seed dataset + columns added).
        """
        workflow_input_df = self._attach_record_ids(df=df)
        seed_source = LocalFileSeedSource.from_dataframe(workflow_input_df, str(seed_path))
        config_builder = DataDesignerConfigBuilder(model_configs=model_configs)
        config_builder.with_seed_dataset(seed_source, sampling_strategy=SamplingStrategy.ORDERED)
        for column in columns:
            config_builder.add_column(column)
        return config_builder

    def build_config_for_seed(
        self,
        *,
        model_configs: list[ModelConfig],
        columns: list[ColumnConfigT],
        seed_path: str | Path,
        job_index: int = 0,
        num_jobs: int = 1,
    ) -> DataDesignerConfigBuilder:
        """Assemble the workflow config reading an EXISTING seed parquet (no write).

        Like :meth:`build_config` but the seed dataset points at an already-written
        ``seed_path`` (record ids assumed already attached) instead of materializing a
        DataFrame. Use this on a distributed worker that received the seed from an
        orchestrator and must NOT rewrite the shared file. ``num_jobs > 1`` selects this
        worker's ordered partition (``job_index`` of ``num_jobs``), matching how the
        orchestrator shards the seed.
        """
        from data_designer.config.seed import PartitionBlock  # noqa: PLC0415

        if num_jobs < 1:
            raise ValueError(f"num_jobs must be >= 1, got {num_jobs}")
        if not (0 <= job_index < num_jobs):
            raise ValueError(f"job_index must be in [0, num_jobs), got job_index={job_index}, num_jobs={num_jobs}")

        config_builder = DataDesignerConfigBuilder(model_configs=model_configs)
        seed_source = LocalFileSeedSource(path=str(seed_path))
        selection = PartitionBlock(index=job_index, num_partitions=num_jobs) if num_jobs > 1 else None
        config_builder.with_seed_dataset(
            seed_source,
            sampling_strategy=SamplingStrategy.ORDERED,
            selection_strategy=selection,
        )
        for column in columns:
            config_builder.add_column(column)
        return config_builder

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
                "dropped.",
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
