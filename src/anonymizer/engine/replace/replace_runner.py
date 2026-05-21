# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from data_designer.config.models import ModelConfig

from anonymizer.config.models import ReplaceModelSelection
from anonymizer.config.replace_strategies import (
    Annotate,
    Hash,
    Redact,
    ReplaceMethod,
    Substitute,
)
from anonymizer.engine.constants import (
    COL_ATTRIBUTE_FIDELITY_INVALID_ENTITIES,
    COL_ATTRIBUTE_FIDELITY_JUDGE,
    COL_ATTRIBUTE_FIDELITY_VALID,
    COL_DETECTION_INVALID_ENTITIES,
    COL_DETECTION_JUDGE,
    COL_DETECTION_VALID,
    COL_RELATIONAL_CONSISTENCY_INVALID_RELATIONS,
    COL_RELATIONAL_CONSISTENCY_JUDGE,
    COL_RELATIONAL_CONSISTENCY_VALID,
    COL_TYPE_FIDELITY_INVALID_REPLACEMENTS,
    COL_TYPE_FIDELITY_JUDGE,
    COL_TYPE_FIDELITY_VALID,
)
from anonymizer.engine.ndd.adapter import FailedRecord, NddAdapter
from anonymizer.engine.replace.attribute_fidelity_judge import AttributeFidelityJudgeWorkflow
from anonymizer.engine.replace.detection_judge import DetectionJudgeWorkflow
from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceWorkflow
from anonymizer.engine.replace.relational_consistency_judge import RelationalConsistencyJudgeWorkflow
from anonymizer.engine.replace.strategies import apply_local_replace_strategy, apply_replacement_map
from anonymizer.engine.replace.type_fidelity_judge import TypeFidelityJudgeWorkflow

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
        run_replace_evaluation: bool = True,
    ) -> ReplacementResult:
        logger.debug("replacement strategy: %s on %d records", type(replace_method).__name__, len(dataframe))
        is_substitute = isinstance(replace_method, Substitute)

        if isinstance(replace_method, (Annotate, Redact, Hash)):
            local_df = apply_local_replace_strategy(dataframe, strategy=replace_method)
            failed_records: list[FailedRecord] = []
        elif is_substitute:
            if self._llm_workflow is None:
                raise ValueError("Substitute requires an llm_workflow, but none was provided.")
            map_result = self._llm_workflow.generate_map_only(
                dataframe,
                model_configs=model_configs,
                selected_models=selected_models,
                instructions=replace_method.instructions,
                preview_num_records=preview_num_records,
            )
            local_df = apply_replacement_map(map_result.dataframe)
            failed_records = list(map_result.failed_records)
        else:
            raise ValueError(f"Unsupported replace method: {type(replace_method).__name__}")

        if not run_replace_evaluation:
            logger.info("⏭️  Replace evaluation disabled (run_replace_evaluation=False) — skipping LLM judges")
            return ReplacementResult(dataframe=local_df, failed_records=failed_records)

        # Production path: single DD workflow with all active judge columns. DD
        # parallelizes the columns internally, so the 4 LLM calls run concurrently
        # without us reaching for threads. Falls back to per-judge serial dispatch
        # when no adapter is wired up (kept for the mock-based unit tests).
        if self._adapter is not None:
            judged_df = self._run_merged_judges(
                local_df,
                is_substitute=is_substitute,
                model_configs=model_configs,
                selected_models=selected_models,
                preview_num_records=preview_num_records,
                failed_records=failed_records,
            )
            return ReplacementResult(dataframe=judged_df, failed_records=failed_records)

        judged_df = self._run_detection_judge(
            local_df,
            model_configs=model_configs,
            selected_models=selected_models,
            preview_num_records=preview_num_records,
            failed_records=failed_records,
        )
        if is_substitute:
            judged_df = self._run_type_fidelity_judge(
                judged_df,
                model_configs=model_configs,
                selected_models=selected_models,
                preview_num_records=preview_num_records,
                failed_records=failed_records,
            )
            judged_df = self._run_relational_consistency_judge(
                judged_df,
                model_configs=model_configs,
                selected_models=selected_models,
                preview_num_records=preview_num_records,
                failed_records=failed_records,
            )
            judged_df = self._run_attribute_fidelity_judge(
                judged_df,
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
        model_configs: list[ModelConfig],
        selected_models: ReplaceModelSelection,
        preview_num_records: int | None,
        failed_records: list[FailedRecord],
    ) -> pd.DataFrame:
        """Run all active replace judges as columns of a single DD workflow.

        Each judge owns its own ``prepare()`` (adds intermediate columns),
        ``column_config()`` (the DD column spec), and ``postprocess()`` (flatten
        + apply passthrough defaults). The adapter sees one workflow with N
        columns and lets DD schedule them in parallel.
        """
        active = [j for j in [self._detection_judge] if j is not None]
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

        try:
            run_result = self._adapter.run_workflow(  # type: ignore[union-attr]
                prepared,
                model_configs=model_configs,
                columns=[judge.column_config(selected_models) for judge in active],
                workflow_name="replace-judges",
                preview_num_records=preview_num_records,
            )
            failed_records.extend(run_result.failed_records)
            judged_df = run_result.dataframe
        except Exception:
            logger.warning(
                "Replace judges workflow failed; populating defaults for all judges", exc_info=True
            )
            judged_df = prepared

        for judge in active:
            judged_df = judge.postprocess(judged_df)
        # Preserve the caller's dataframe metadata (notably ``original_text_column``),
        # which downstream column-renaming relies on. DataDesigner's output loses it.
        judged_df.attrs = {**judged_df.attrs, **dataframe.attrs}
        return judged_df

    # ---------------------------------------------------------------------------
    # Detection judge (non-critical)
    # ---------------------------------------------------------------------------

    def _run_detection_judge(
        self,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: ReplaceModelSelection,
        preview_num_records: int | None,
        failed_records: list[FailedRecord],
    ) -> pd.DataFrame:
        if self._detection_judge is None:
            return dataframe
        try:
            judge_result = self._detection_judge.evaluate(
                dataframe,
                model_configs=model_configs,
                selected_models=selected_models,
                preview_num_records=preview_num_records,
            )
            failed_records.extend(judge_result.failed_records)
            return judge_result.dataframe
        except Exception:
            logger.warning("Detection judge step failed; populating defaults", exc_info=True)
            dataframe[COL_DETECTION_JUDGE] = None
            dataframe[COL_DETECTION_VALID] = None
            dataframe[COL_DETECTION_INVALID_ENTITIES] = [[] for _ in range(len(dataframe))]
            return dataframe

    # ---------------------------------------------------------------------------
    # Type-fidelity judge (Substitute only, non-critical)
    # ---------------------------------------------------------------------------

    def _run_type_fidelity_judge(
        self,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: ReplaceModelSelection,
        preview_num_records: int | None,
        failed_records: list[FailedRecord],
    ) -> pd.DataFrame:
        if self._type_fidelity_judge is None:
            return dataframe
        try:
            judge_result = self._type_fidelity_judge.evaluate(
                dataframe,
                model_configs=model_configs,
                selected_models=selected_models,
                preview_num_records=preview_num_records,
            )
            failed_records.extend(judge_result.failed_records)
            return judge_result.dataframe
        except Exception:
            logger.warning("Type-fidelity judge step failed; populating defaults", exc_info=True)
            dataframe[COL_TYPE_FIDELITY_JUDGE] = None
            dataframe[COL_TYPE_FIDELITY_VALID] = None
            dataframe[COL_TYPE_FIDELITY_INVALID_REPLACEMENTS] = [[] for _ in range(len(dataframe))]
            return dataframe

    # ---------------------------------------------------------------------------
    # Relational-consistency judge (Substitute only, non-critical)
    # ---------------------------------------------------------------------------

    def _run_relational_consistency_judge(
        self,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: ReplaceModelSelection,
        preview_num_records: int | None,
        failed_records: list[FailedRecord],
    ) -> pd.DataFrame:
        if self._relational_consistency_judge is None:
            return dataframe
        try:
            judge_result = self._relational_consistency_judge.evaluate(
                dataframe,
                model_configs=model_configs,
                selected_models=selected_models,
                preview_num_records=preview_num_records,
            )
            failed_records.extend(judge_result.failed_records)
            return judge_result.dataframe
        except Exception:
            logger.warning("Relational-consistency judge step failed; populating defaults", exc_info=True)
            dataframe[COL_RELATIONAL_CONSISTENCY_JUDGE] = None
            dataframe[COL_RELATIONAL_CONSISTENCY_VALID] = None
            dataframe[COL_RELATIONAL_CONSISTENCY_INVALID_RELATIONS] = [[] for _ in range(len(dataframe))]
            return dataframe

    # ---------------------------------------------------------------------------
    # Attribute-fidelity judge (Substitute only, non-critical)
    # ---------------------------------------------------------------------------

    def _run_attribute_fidelity_judge(
        self,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: ReplaceModelSelection,
        preview_num_records: int | None,
        failed_records: list[FailedRecord],
    ) -> pd.DataFrame:
        if self._attribute_fidelity_judge is None:
            return dataframe
        try:
            judge_result = self._attribute_fidelity_judge.evaluate(
                dataframe,
                model_configs=model_configs,
                selected_models=selected_models,
                preview_num_records=preview_num_records,
            )
            failed_records.extend(judge_result.failed_records)
            return judge_result.dataframe
        except Exception:
            logger.warning("Attribute-fidelity judge step failed; populating defaults", exc_info=True)
            dataframe[COL_ATTRIBUTE_FIDELITY_JUDGE] = None
            dataframe[COL_ATTRIBUTE_FIDELITY_VALID] = None
            dataframe[COL_ATTRIBUTE_FIDELITY_INVALID_ENTITIES] = [[] for _ in range(len(dataframe))]
            return dataframe
