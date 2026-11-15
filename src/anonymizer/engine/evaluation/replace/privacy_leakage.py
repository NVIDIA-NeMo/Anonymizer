# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input preparation for privacy leakage evaluation of replacement output."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from data_designer.config.models import ModelConfig

from anonymizer.config.models import EvaluateModelSelection
from anonymizer.engine.constants import (
    COL_DOMAIN,
    COL_DOMAIN_SUPPLEMENT,
    COL_DOMAIN_SUPPLEMENT_PRIVACY,
    COL_ENTITIES_BY_VALUE,
    COL_ENTITY_CLASSIFICATION,
    COL_ENTITY_SENSITIVITY,
    COL_LEAKAGE_MASS,
    COL_PRIVACY_LEAKAGE_CONTRIBUTIONS,
    COL_PRIVACY_QA,
    COL_PRIVACY_QA_ERROR,
    COL_PRIVACY_QA_REANSWER,
    COL_REPLACED_TEXT,
    COL_TEXT,
    COL_WEIGHTED_LEAKAGE_RATE,
    RECORD_ID_COLUMN,
)
from anonymizer.engine.evaluation.replace.entity_sensitivity import ReplaceEntitySensitivityWorkflow
from anonymizer.engine.evaluation.replace.privacy_answers import ReplacePrivacyAnswerWorkflow
from anonymizer.engine.evaluation.replace.privacy_qa import ReplacePrivacyQAGenerationWorkflow
from anonymizer.engine.evaluation.replace.privacy_scoring import ReplacePrivacyScoringWorkflow
from anonymizer.engine.ndd.adapter import FailedRecord, NddAdapter
from anonymizer.engine.rewrite.entity_classification import EntityClassificationWorkflow
from anonymizer.engine.rewrite.parsers import normalize_payload
from anonymizer.engine.rewrite.workflow_utils import derive_seed_columns, select_seed_cols
from anonymizer.engine.schemas import EntitiesByValueSchema

logger = logging.getLogger("anonymizer.evaluation.replace.privacy_leakage")


@dataclass(frozen=True)
class PrivacyLeakageEvaluationResult:
    dataframe: pd.DataFrame
    failed_records: list[FailedRecord]


class ReplacePrivacyLeakageWorkflow:
    """Run optional privacy evaluation while preserving all Replace records."""

    def __init__(self, adapter: NddAdapter) -> None:
        self._adapter = adapter

    def run(
        self,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: EvaluateModelSelection,
        data_summary: str | None = None,
        preview_num_records: int | None = None,
        privacy_qa_batch_size: int | None = 5,
    ) -> PrivacyLeakageEvaluationResult:
        prepared = self._adapter._attach_record_ids(dataframe).reset_index(drop=True)
        prepared = prepared.drop(
            columns=[COL_DOMAIN, COL_DOMAIN_SUPPLEMENT, COL_DOMAIN_SUPPLEMENT_PRIVACY], errors="ignore"
        )
        # Validate columns even for empty input, before partitioning record errors.
        prepare_privacy_evaluation_inputs(prepared.iloc[:0])
        if prepared[RECORD_ID_COLUMN].duplicated().any():
            raise ValueError("Privacy evaluation requires unique record IDs.")
        columns = [
            *EntityClassificationWorkflow().columns(selected_models=selected_models, label_categories={}),
            *ReplaceEntitySensitivityWorkflow().columns(selected_models=selected_models, data_summary=data_summary),
            *ReplacePrivacyQAGenerationWorkflow().columns(),
            *ReplacePrivacyAnswerWorkflow().columns(
                evaluator_alias=selected_models.privacy_qa_judge, batch_size=privacy_qa_batch_size
            ),
            *ReplacePrivacyScoringWorkflow().columns(),
        ]
        produced = list(
            dict.fromkeys(name for column in columns for name in [column.name, *column.side_effect_columns])
        )
        for name in produced:
            prepared[name] = pd.Series([None] * len(prepared), dtype=object)

        active_positions: list[int] = []
        failed: list[FailedRecord] = []
        for idx, raw in prepared[COL_ENTITIES_BY_VALUE].items():
            try:
                entities = EntitiesByValueSchema.from_raw(raw, strict=True)
            except ValueError:
                prepared.at[idx, COL_PRIVACY_QA_ERROR] = "Malformed detected entities; privacy evaluation unavailable."
                failed.append(
                    FailedRecord(
                        record_id=str(prepared.at[idx, RECORD_ID_COLUMN]),
                        step="replace-privacy",
                        reason="Malformed detected entities",
                    )
                )
                continue
            if entities.entities_by_value:
                active_positions.append(int(idx))
            else:
                prepared.at[idx, COL_ENTITY_SENSITIVITY] = {"entities": []}
                prepared.at[idx, COL_ENTITY_CLASSIFICATION] = {}
                prepared.at[idx, COL_PRIVACY_QA] = {"items": []}
                prepared.at[idx, COL_PRIVACY_QA_REANSWER] = {"answers": []}
                prepared.at[idx, COL_LEAKAGE_MASS] = 0.0
                prepared.at[idx, COL_WEIGHTED_LEAKAGE_RATE] = 0.0
                prepared.at[idx, COL_PRIVACY_LEAKAGE_CONTRIBUTIONS] = []

        if active_positions:
            active = prepare_privacy_evaluation_inputs(prepared.loc[active_positions])
            prepared.loc[active_positions, COL_PRIVACY_QA_ERROR] = "Privacy leakage evaluation unavailable."
            try:
                classifier = EntityClassificationWorkflow()
                categories = classifier.classify_dataset(
                    active,
                    adapter=self._adapter,
                    model_configs=model_configs,
                    selected_models=selected_models,
                    workflow_name="replace-label-classification",
                )
                columns[0] = classifier.columns(selected_models=selected_models, label_categories=categories)[0]
                seed_cols = derive_seed_columns(columns, active)
                seed = select_seed_cols(active, seed_cols)
                result = self._adapter.run_workflow(
                    seed,
                    model_configs=model_configs,
                    columns=columns,
                    workflow_name="replace-privacy",
                    preview_num_records=preview_num_records,
                )
            except Exception:
                # Optional evaluation must not discard already-anonymized records.
                logger.warning("Replace privacy workflow failed; privacy evaluation unavailable.")
                failed.extend(
                    FailedRecord(
                        record_id=str(record_id), step="replace-privacy", reason="Privacy evaluation unavailable"
                    )
                    for record_id in active[RECORD_ID_COLUMN]
                )
            else:
                failed.extend(result.failed_records)
                target_positions = {str(prepared.at[idx, RECORD_ID_COLUMN]): idx for idx in active_positions}
                for _, row in result.dataframe.iterrows():
                    idx = target_positions[str(row[RECORD_ID_COLUMN])]
                    for name in produced:
                        value = row.get(name)
                        missing = value is pd.NA or (isinstance(value, float) and pd.isna(value))
                        prepared.at[idx, name] = None if missing else normalize_payload(value)
                    if pd.isna(prepared.at[idx, COL_LEAKAGE_MASS]) and not prepared.at[idx, COL_PRIVACY_QA_ERROR]:
                        prepared.at[idx, COL_PRIVACY_QA_ERROR] = "Privacy leakage score unavailable."

        prepared.index = dataframe.index
        return PrivacyLeakageEvaluationResult(dataframe=prepared, failed_records=failed)


def prepare_privacy_evaluation_inputs(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Copy Replace evaluation inputs and normalize the original entity payloads.

    Expects internal column names, as restored by ``Anonymizer.evaluate()``.
    Original entity values remain the evaluation targets; replaced text is the
    evidence to evaluate. No detection, inference discovery, or model calls run.
    Dataset summaries remain workflow metadata rather than per-record columns.

    All rows, their IDs, index, ordering, and additional columns are preserved,
    including rows with no entities. The input dataframe is not modified.

    Raises:
        ValueError: If a required input column is missing or entities are malformed.
    """
    required = {RECORD_ID_COLUMN, COL_TEXT, COL_REPLACED_TEXT, COL_ENTITIES_BY_VALUE}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(
            f"Privacy leakage evaluation requires columns {sorted(required)}; missing: {sorted(missing)}. "
            "Pass internally named inputs from a previous Replace run()/preview() result."
        )

    prepared = dataframe.copy()
    prepared[COL_ENTITIES_BY_VALUE] = dataframe[COL_ENTITIES_BY_VALUE].map(
        lambda raw: EntitiesByValueSchema.from_raw(raw, strict=True).model_dump()
    )
    return prepared
