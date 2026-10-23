# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import pandas as pd
from data_designer.config.models import ModelConfig

from anonymizer.config.models import EvaluateModelSelection, ReplaceModelSelection
from anonymizer.config.replace_strategies import (
    Annotate,
    Hash,
    Redact,
    ReplaceMethod,
    Substitute,
)
from anonymizer.engine.constants import (
    COL_ENTITIES_BY_VALUE,
    COL_REPLACEMENT_APPLICATION,
    COL_REPLACEMENT_MAP,
)
from anonymizer.engine.evaluation.detection_judge import DetectionJudgeWorkflow
from anonymizer.engine.evaluation.entity_coverage_judge import EntityCoverageWorkflow
from anonymizer.engine.evaluation.judge_base import _BaseJudgeWorkflow
from anonymizer.engine.evaluation.replace.attribute_fidelity_judge import AttributeFidelityJudgeWorkflow
from anonymizer.engine.evaluation.replace.relational_consistency_judge import RelationalConsistencyJudgeWorkflow
from anonymizer.engine.evaluation.replace.type_fidelity_judge import TypeFidelityJudgeWorkflow
from anonymizer.engine.ndd.adapter import FailedRecord, NddAdapter
from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceWorkflow
from anonymizer.engine.replace.strategies import apply_local_replace_strategy
from anonymizer.engine.row_partitioning import restore_diagnostic_columns
from anonymizer.measurement import stage_timer

logger = logging.getLogger("anonymizer.replace")


@dataclass(frozen=True)
class ReplacementResult:
    """Result of a replacement workflow execution."""

    dataframe: pd.DataFrame
    failed_records: list[FailedRecord]


class ReplacementWorkflow:
    """Dispatch replace execution between local and LLM-backed strategies."""

    def __init__(
        self,
        llm_workflow: LlmReplaceWorkflow | None = None,
        detection_judge: DetectionJudgeWorkflow | None = None,
        type_fidelity_judge: TypeFidelityJudgeWorkflow | None = None,
        relational_consistency_judge: RelationalConsistencyJudgeWorkflow | None = None,
        attribute_fidelity_judge: AttributeFidelityJudgeWorkflow | None = None,
        adapter: NddAdapter | None = None,
    ) -> None:
        self._llm_workflow = llm_workflow
        self._detection_judge = detection_judge
        self._type_fidelity_judge = type_fidelity_judge
        self._relational_consistency_judge = relational_consistency_judge
        self._attribute_fidelity_judge = attribute_fidelity_judge
        self._adapter = adapter

    def run(
        self,
        dataframe: pd.DataFrame,
        *,
        replace_method: ReplaceMethod,
        model_configs: list[ModelConfig],
        selected_models: ReplaceModelSelection,
        preview_num_records: int | None = None,
    ) -> ReplacementResult:
        """Apply the replacement strategy (no LLM judges).

        Evaluation is a separate concern — call ``evaluate()`` on the resulting
        dataframe when you want the Judge Agreement scores.
        """
        strategy = type(replace_method).__name__
        with stage_timer("ReplacementWorkflow.run", strategy=strategy, input_row_count=len(dataframe)) as measurement:
            logger.debug("replacement starting on %d records", len(dataframe))
            if isinstance(replace_method, (Annotate, Redact, Hash)):
                local_df = apply_local_replace_strategy(dataframe, strategy=replace_method)
                failed_records: list[FailedRecord] = []
            elif isinstance(replace_method, Substitute):
                if self._llm_workflow is None:
                    raise ValueError("Substitute requires an llm_workflow, but none was provided.")
                from anonymizer.engine.replace.route import run_substitute_route

                substituted = run_substitute_route(
                    dataframe,
                    llm_workflow=self._llm_workflow,
                    replace_method=replace_method,
                    map_arguments=dict(
                        model_configs=model_configs,
                        selected_models=selected_models,
                        preview_num_records=preview_num_records,
                    ),
                )
                local_df, failed_records = substituted.dataframe, substituted.failed_records
            else:
                raise ValueError(f"Unsupported replace method: {type(replace_method).__name__}")

            result = ReplacementResult(dataframe=local_df, failed_records=failed_records)
            measurement.update(
                output_row_count=len(result.dataframe),
                failed_record_count=len(result.failed_records),
            )
            return result

    def evaluate(
        self,
        dataframe: pd.DataFrame,
        *,
        replace_method: ReplaceMethod,
        model_configs: list[ModelConfig],
        selected_models: EvaluateModelSelection,
        preview_num_records: int | None = None,
        entity_labels: list[str] | None = None,
        excluded_entity_labels: list[str] | None = None,
        compute_detection_validity: bool = False,
        data_summary: str | None = None,
    ) -> ReplacementResult:
        """Run the LLM evaluation judges on an already-replaced dataframe.

        Entity coverage always runs for every replace strategy. Substitute also
        runs type fidelity, relational consistency, and attribute fidelity.
        Detection validity runs only when ``compute_detection_validity=True``.
        All active judges are submitted as columns of one DataDesigner workflow.

        ``entity_labels`` and ``excluded_entity_labels`` together define the label
        scope passed to the coverage judge — only entities whose labels were in
        scope during detection are evaluated.

        Raises ``ValueError`` if the workflow has no adapter wired up or if the
        dataframe is missing the columns the judges read.
        """
        if self._adapter is None:
            raise ValueError(
                "ReplacementWorkflow.evaluate() requires an adapter; construct "
                "ReplacementWorkflow(..., adapter=NddAdapter(...))."
            )
        is_substitute = isinstance(replace_method, Substitute)
        required = {COL_ENTITIES_BY_VALUE}
        if is_substitute:
            required.add(COL_REPLACEMENT_MAP)
        missing = required - set(dataframe.columns)
        if missing:
            raise ValueError(
                f"evaluate() requires the dataframe to contain {sorted(required)}; "
                f"missing: {sorted(missing)}. Pass the trace_dataframe from a "
                f"previous preview()/run() call."
            )
        entity_coverage_judge = EntityCoverageWorkflow(
            adapter=self._adapter,  # type: ignore[arg-type]
            entity_labels=entity_labels,
            excluded_entity_labels=excluded_entity_labels,
            data_summary=data_summary,
        )
        failed_records: list[FailedRecord] = []
        judged_df = self._run_merged_judges(
            dataframe,
            is_substitute=is_substitute,
            entity_coverage_judge=entity_coverage_judge,
            compute_detection_validity=compute_detection_validity,
            model_configs=model_configs,
            selected_models=selected_models,
            preview_num_records=preview_num_records,
            failed_records=failed_records,
        )
        return ReplacementResult(dataframe=judged_df, failed_records=failed_records)

    # ---------------------------------------------------------------------------
    # Merged-judge dispatch (DataDesigner parallelizes the columns internally)
    # ---------------------------------------------------------------------------

    def _run_merged_judges(
        self,
        dataframe: pd.DataFrame,
        *,
        is_substitute: bool,
        entity_coverage_judge: EntityCoverageWorkflow,
        compute_detection_validity: bool,
        model_configs: list[ModelConfig],
        selected_models: EvaluateModelSelection,
        preview_num_records: int | None,
        failed_records: list[FailedRecord],
    ) -> pd.DataFrame:
        """Run all active replace judges as columns of a single DD workflow.

        Each judge owns its own ``prepare()`` (adds intermediate columns),
        ``column_config()`` (the DD column spec), and ``postprocess()`` (flatten
        + apply passthrough defaults). The adapter sees one workflow with N
        columns and lets DD schedule them in parallel.
        """
        active: list[_BaseJudgeWorkflow] = [entity_coverage_judge]
        if compute_detection_validity and self._detection_judge is not None:
            active.append(self._detection_judge)
        if is_substitute:
            active.extend(
                j
                for j in [
                    self._type_fidelity_judge,
                    self._relational_consistency_judge,
                    self._attribute_fidelity_judge,
                ]
                if j is not None
            )
        if not active:
            return dataframe

        prepared = dataframe
        for judge in active:
            prepared = judge.prepare(prepared)
        # Retain the adapter's record-ID column in the public trace. Diagnostic
        # correspondence uses source positions, including when IDs repeat.
        adapter = cast(NddAdapter, self._adapter)
        prepared = adapter._attach_record_ids(prepared)

        try:
            # Replacement application telemetry can contain an all-empty nested
            # ``skipped_span_label_counts`` mapping. PyArrow cannot represent that
            # as a Parquet struct with no child fields, and the judges do not read
            # this diagnostic column. Keep it on ``prepared`` for the returned
            # trace, but omit it from DataDesigner's temporary seed dataframe.
            workflow_input = prepared.drop(columns=[COL_REPLACEMENT_APPLICATION], errors="ignore")
            run_result = adapter.run_workflow(
                workflow_input,
                model_configs=model_configs,
                columns=[judge.column_config(selected_models) for judge in active],
                workflow_name="replace-judges",
                preview_num_records=preview_num_records,
            )
            failed_records.extend(run_result.failed_records)
            new_cols = [c for c in run_result.dataframe.columns if c not in prepared.columns]
            judged_df = restore_diagnostic_columns(
                prepared,
                run_result.dataframe,
                dict.fromkeys(new_cols),
                source_positions=run_result.source_positions,
            ).reset_index(drop=True)
        except Exception:
            logger.debug("Replace judges workflow failed; evaluation scores may be unavailable.")
            judged_df = prepared

        for judge in active:
            judged_df = judge.postprocess(judged_df)
        return judged_df
