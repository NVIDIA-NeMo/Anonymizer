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
    COL_ENTITIES_BY_VALUE,
    COL_REPLACEMENT_MAP,
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
    ) -> ReplacementResult:
        """Apply the replacement strategy (no LLM judges).

        Evaluation is a separate concern — call ``evaluate()`` on the resulting
        dataframe when you want the LLM alignment scores.
        """
        logger.debug("replacement strategy: %s on %d records", type(replace_method).__name__, len(dataframe))

        if isinstance(replace_method, (Annotate, Redact, Hash)):
            local_df = apply_local_replace_strategy(dataframe, strategy=replace_method)
            failed_records: list[FailedRecord] = []
        elif isinstance(replace_method, Substitute):
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

        return ReplacementResult(dataframe=local_df, failed_records=failed_records)

    def evaluate(
        self,
        dataframe: pd.DataFrame,
        *,
        replace_method: ReplaceMethod,
        model_configs: list[ModelConfig],
        selected_models: ReplaceModelSelection,
        preview_num_records: int | None = None,
    ) -> ReplacementResult:
        """Run the LLM evaluation judges on an already-replaced dataframe.

        For Substitute, runs all 4 judges (detection + type fidelity + relational
        consistency + attribute fidelity) as columns of a single DataDesigner
        workflow. For other strategies, runs the detection judge only.

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
        failed_records: list[FailedRecord] = []
        judged_df = self._run_merged_judges(
            dataframe,
            is_substitute=is_substitute,
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
            logger.warning("Replace judges workflow failed; populating defaults for all judges", exc_info=True)
            judged_df = prepared

        for judge in active:
            judged_df = judge.postprocess(judged_df)
        return judged_df
