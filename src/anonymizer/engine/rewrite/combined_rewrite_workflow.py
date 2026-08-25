# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd
from data_designer.config import SkipConfig, custom_column_generator
from data_designer.config.column_configs import CustomColumnConfig, LLMStructuredColumnConfig
from data_designer.config.column_types import ColumnConfigT
from data_designer.config.models import ModelConfig
from pydantic import BaseModel

from anonymizer.config.models import ReplaceModelSelection, RewriteModelSelection
from anonymizer.config.rewrite import EvaluationCriteria, PrivacyGoal
from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_ENTITIES_BY_VALUE,
    COL_ENTITIES_FOR_REPLACE,
    COL_ENTITIES_FOR_REPLACE_JSON,
    COL_ENTITY_EXAMPLES,
    COL_LEAKAGE_MASS,
    COL_LEAKED_PRIVACY_ITEMS,
    COL_NEEDS_HUMAN_REVIEW,
    COL_NEEDS_REPAIR,
    COL_PRIVACY_QA_REANSWER,
    COL_QUALITY_QA_COMPARE,
    COL_QUALITY_QA_REANSWER,
    COL_REPAIR_ITERATIONS,
    COL_REPLACEMENT_MAP,
    COL_REPLACEMENT_MAP_RAW,
    COL_REPLACEMENT_MAP_SOURCE,
    COL_REWRITTEN_TEXT,
    COL_REWRITTEN_TEXT_INITIAL,
    COL_REWRITTEN_TEXT_NEXT,
    COL_UTILITY_SCORE,
    COL_WEIGHTED_LEAKAGE_RATE,
)
from anonymizer.engine.ndd.adapter import RECORD_ID_COLUMN, NddAdapter
from anonymizer.engine.ndd.model_loader import resolve_model_alias
from anonymizer.engine.replace.llm_replace_workflow import (
    REPLACEMENT_MAP_SOURCE_LLM,
    _create_entity_examples,
    _enrich_entities_for_template,
    _filter_replacement_map_to_input_entities,
    _get_replacement_mapping_prompt,
)
from anonymizer.engine.rewrite.domain_classification import DomainClassificationWorkflow
from anonymizer.engine.rewrite.evaluate import EvaluateWorkflow
from anonymizer.engine.rewrite.parsers import normalize_payload
from anonymizer.engine.rewrite.qa_generation import QAGenerationWorkflow
from anonymizer.engine.rewrite.repair import RepairWorkflow
from anonymizer.engine.rewrite.rewrite_generation import RewriteGenerationWorkflow
from anonymizer.engine.rewrite.rewrite_workflow import (
    RewriteResult,
    RewriteWorkflow,
    _apply_passthrough_defaults,
    _has_entities,
    _join_new_columns,
)
from anonymizer.engine.rewrite.sensitivity_disposition import SensitivityDispositionWorkflow
from anonymizer.engine.rewrite.workflow_utils import derive_seed_columns, select_seed_cols
from anonymizer.engine.row_partitioning import merge_and_reorder, split_rows
from anonymizer.engine.schemas import EntitiesByValueSchema, EntityReplacementMapSchema
from anonymizer.measurement import stage_timer


@dataclass(frozen=True)
class EvaluationState:
    iteration: int
    rewritten_text: str
    quality_reanswer: str
    privacy_reanswer: str
    quality_compare: str
    utility_score: str
    leakage_mass: str
    weighted_leakage_rate: str
    any_high_leaked: str
    needs_repair: str


@dataclass(frozen=True)
class RepairState:
    iteration: int
    leaked_items: str
    rewritten_text: str


@dataclass(frozen=True)
class CombinedRewriteGraph:
    columns: list[ColumnConfigT]
    evaluation_states: list[EvaluationState]
    repair_states: list[RepairState]
    internal_columns: list[str]


class _FinalizationParams(BaseModel):
    flag_utility_below: float | None
    flag_leakage_above: float | None


def _iteration_column(column: str, iteration: int) -> str:
    return f"{column}__iteration_{iteration}"


def _evaluation_state(iteration: int, rewritten_text: str) -> EvaluationState:
    return EvaluationState(
        iteration=iteration,
        rewritten_text=rewritten_text,
        quality_reanswer=_iteration_column(COL_QUALITY_QA_REANSWER, iteration),
        privacy_reanswer=_iteration_column(COL_PRIVACY_QA_REANSWER, iteration),
        quality_compare=_iteration_column(COL_QUALITY_QA_COMPARE, iteration),
        utility_score=_iteration_column(COL_UTILITY_SCORE, iteration),
        leakage_mass=_iteration_column(COL_LEAKAGE_MASS, iteration),
        weighted_leakage_rate=_iteration_column(COL_WEIGHTED_LEAKAGE_RATE, iteration),
        any_high_leaked=_iteration_column(COL_ANY_HIGH_LEAKED, iteration),
        needs_repair=_iteration_column(COL_NEEDS_REPAIR, iteration),
    )


def _repair_state(iteration: int) -> RepairState:
    return RepairState(
        iteration=iteration,
        leaked_items=_iteration_column(COL_LEAKED_PRIVACY_ITEMS, iteration),
        rewritten_text=_iteration_column(COL_REWRITTEN_TEXT_NEXT, iteration),
    )


@custom_column_generator(
    required_columns=[COL_ENTITIES_BY_VALUE],
    side_effect_columns=[COL_ENTITIES_FOR_REPLACE, COL_ENTITIES_FOR_REPLACE_JSON],
)
def _prepare_replacement_inputs(row: dict[str, Any]) -> dict[str, Any]:
    parsed = EntitiesByValueSchema.from_raw(row.get(COL_ENTITIES_BY_VALUE))
    row[COL_ENTITY_EXAMPLES] = _create_entity_examples(parsed)
    row[COL_ENTITIES_FOR_REPLACE] = _enrich_entities_for_template(parsed)
    row[COL_ENTITIES_FOR_REPLACE_JSON] = json.dumps(row[COL_ENTITIES_FOR_REPLACE])
    return row


@custom_column_generator(
    required_columns=[COL_REPLACEMENT_MAP_RAW, COL_ENTITIES_BY_VALUE],
    side_effect_columns=[COL_REPLACEMENT_MAP_SOURCE],
)
def _filter_replacement_map(row: dict[str, Any]) -> dict[str, Any]:
    row[COL_REPLACEMENT_MAP] = _filter_replacement_map_to_input_entities(
        raw_map=row.get(COL_REPLACEMENT_MAP_RAW, {"replacements": []}),
        parsed_entities=EntitiesByValueSchema.from_raw(row.get(COL_ENTITIES_BY_VALUE)),
        record_id=str(row.get(RECORD_ID_COLUMN, "")),
    )
    row[COL_REPLACEMENT_MAP_SOURCE] = REPLACEMENT_MAP_SOURCE_LLM
    return row


def _replacement_columns(selected_models: ReplaceModelSelection) -> list[ColumnConfigT]:
    replace_alias = resolve_model_alias("replacement_generator", selected_models)
    return [
        CustomColumnConfig(
            name=COL_ENTITY_EXAMPLES,
            generator_function=_prepare_replacement_inputs,
        ),
        LLMStructuredColumnConfig(
            name=COL_REPLACEMENT_MAP_RAW,
            prompt=_get_replacement_mapping_prompt(entities_column=COL_ENTITIES_FOR_REPLACE),
            model_alias=replace_alias,
            output_format=EntityReplacementMapSchema,
        ),
        CustomColumnConfig(
            name=COL_REPLACEMENT_MAP,
            generator_function=_filter_replacement_map,
        ),
    ]


def _canonical_row(row: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    canonical = row.copy()
    for original, remapped in mapping.items():
        if remapped in row:
            canonical[original] = row[remapped]
    return canonical


def _copy_remapped_outputs(
    row: dict[str, Any],
    generated: dict[str, Any],
    outputs: list[str],
    mapping: dict[str, str],
) -> dict[str, Any]:
    for output in outputs:
        row[mapping.get(output, output)] = generated.get(output)
    return row


def _remap_custom_column(
    column: ColumnConfigT,
    mapping: dict[str, str],
    *,
    skip: SkipConfig | None = None,
) -> CustomColumnConfig:
    if not isinstance(column, CustomColumnConfig):
        raise TypeError(f"Expected CustomColumnConfig, got {type(column).__name__}")

    generator = column.generator_function
    metadata = generator.custom_column_metadata
    required_columns = [mapping.get(name, name) for name in metadata["required_columns"]]
    side_effect_columns = [mapping.get(name, name) for name in metadata["side_effect_columns"]]
    model_aliases = list(metadata["model_aliases"])
    outputs = [column.name, *metadata["side_effect_columns"]]

    if model_aliases:

        @custom_column_generator(
            required_columns=required_columns,
            side_effect_columns=side_effect_columns,
            model_aliases=model_aliases,
        )
        def remapped_generator(row: dict[str, Any], generator_params: Any, models: dict) -> dict[str, Any]:
            generated = generator(_canonical_row(row, mapping), generator_params, models)
            return _copy_remapped_outputs(row, generated, outputs, mapping)

    elif column.generator_params is not None:

        @custom_column_generator(
            required_columns=required_columns,
            side_effect_columns=side_effect_columns,
        )
        def remapped_generator(row: dict[str, Any], generator_params: Any) -> dict[str, Any]:
            generated = generator(_canonical_row(row, mapping), generator_params)
            return _copy_remapped_outputs(row, generated, outputs, mapping)

    else:

        @custom_column_generator(
            required_columns=required_columns,
            side_effect_columns=side_effect_columns,
        )
        def remapped_generator(row: dict[str, Any]) -> dict[str, Any]:
            generated = generator(_canonical_row(row, mapping))
            return _copy_remapped_outputs(row, generated, outputs, mapping)

    updates: dict[str, Any] = {
        "name": mapping.get(column.name, column.name),
        "generator_function": remapped_generator,
    }
    if skip is not None:
        updates["skip"] = skip
    return column.model_copy(update=updates)


def _evaluation_columns(
    adapter: NddAdapter,
    *,
    selected_models: RewriteModelSelection,
    evaluation: EvaluationCriteria,
    state: EvaluationState,
) -> list[ColumnConfigT]:
    mapping = {
        COL_REWRITTEN_TEXT: state.rewritten_text,
        COL_QUALITY_QA_REANSWER: state.quality_reanswer,
        COL_PRIVACY_QA_REANSWER: state.privacy_reanswer,
        COL_QUALITY_QA_COMPARE: state.quality_compare,
        COL_UTILITY_SCORE: state.utility_score,
        COL_LEAKAGE_MASS: state.leakage_mass,
        COL_WEIGHTED_LEAKAGE_RATE: state.weighted_leakage_rate,
        COL_ANY_HIGH_LEAKED: state.any_high_leaked,
        COL_NEEDS_REPAIR: state.needs_repair,
    }
    return [
        _remap_custom_column(column, mapping)
        for column in EvaluateWorkflow(adapter).columns(
            selected_models=selected_models,
            evaluation=evaluation,
        )
    ]


def _repair_columns(
    adapter: NddAdapter,
    *,
    selected_models: RewriteModelSelection,
    privacy_goal: PrivacyGoal,
    evaluation: EvaluationCriteria,
    previous: EvaluationState,
    state: RepairState,
) -> list[ColumnConfigT]:
    mapping = {
        COL_PRIVACY_QA_REANSWER: previous.privacy_reanswer,
        COL_REWRITTEN_TEXT: previous.rewritten_text,
        COL_LEAKAGE_MASS: previous.leakage_mass,
        COL_WEIGHTED_LEAKAGE_RATE: previous.weighted_leakage_rate,
        COL_ANY_HIGH_LEAKED: previous.any_high_leaked,
        COL_UTILITY_SCORE: previous.utility_score,
        COL_LEAKED_PRIVACY_ITEMS: state.leaked_items,
        COL_REWRITTEN_TEXT_NEXT: state.rewritten_text,
    }
    columns = RepairWorkflow(adapter).columns(
        selected_models=selected_models,
        privacy_goal=privacy_goal,
        effective_threshold=evaluation.repair_threshold,
    )
    condition = SkipConfig(when=f"{{{{ not {previous.needs_repair} }}}}")
    return [
        _remap_custom_column(column, mapping, skip=condition if index == 0 else None)
        for index, column in enumerate(columns)
    ]


def _finalization_column(
    states: list[EvaluationState],
    evaluation: EvaluationCriteria,
) -> CustomColumnConfig:
    required_columns = list(
        dict.fromkeys(
            column
            for state in states
            for column in (
                state.rewritten_text,
                state.quality_reanswer,
                state.privacy_reanswer,
                state.quality_compare,
                state.utility_score,
                state.leakage_mass,
                state.weighted_leakage_rate,
                state.any_high_leaked,
                state.needs_repair,
            )
        )
    )
    side_effect_columns = [
        COL_QUALITY_QA_REANSWER,
        COL_PRIVACY_QA_REANSWER,
        COL_QUALITY_QA_COMPARE,
        COL_UTILITY_SCORE,
        COL_LEAKAGE_MASS,
        COL_WEIGHTED_LEAKAGE_RATE,
        COL_ANY_HIGH_LEAKED,
        COL_NEEDS_REPAIR,
        COL_REPAIR_ITERATIONS,
        COL_NEEDS_HUMAN_REVIEW,
    ]

    @custom_column_generator(
        required_columns=required_columns,
        side_effect_columns=side_effect_columns,
    )
    def finalize(row: dict[str, Any], generator_params: _FinalizationParams) -> dict[str, Any]:
        state = next(
            (candidate for candidate in reversed(states) if row.get(candidate.needs_repair) is not None),
            states[0],
        )
        row[COL_REWRITTEN_TEXT] = row.get(state.rewritten_text)
        row[COL_QUALITY_QA_REANSWER] = normalize_payload(row.get(state.quality_reanswer))
        row[COL_PRIVACY_QA_REANSWER] = normalize_payload(row.get(state.privacy_reanswer))
        row[COL_QUALITY_QA_COMPARE] = normalize_payload(row.get(state.quality_compare))
        row[COL_UTILITY_SCORE] = row.get(state.utility_score)
        row[COL_LEAKAGE_MASS] = row.get(state.leakage_mass)
        row[COL_WEIGHTED_LEAKAGE_RATE] = row.get(state.weighted_leakage_rate)
        row[COL_ANY_HIGH_LEAKED] = row.get(state.any_high_leaked)
        row[COL_NEEDS_REPAIR] = row.get(state.needs_repair)
        row[COL_REPAIR_ITERATIONS] = state.iteration

        needs_review = row[COL_REWRITTEN_TEXT] is None or bool(row[COL_ANY_HIGH_LEAKED])
        if generator_params.flag_utility_below is not None:
            needs_review = needs_review or float(row[COL_UTILITY_SCORE]) < generator_params.flag_utility_below
        if generator_params.flag_leakage_above is not None:
            needs_review = needs_review or float(row[COL_LEAKAGE_MASS]) > generator_params.flag_leakage_above
        row[COL_NEEDS_HUMAN_REVIEW] = needs_review
        return row

    return CustomColumnConfig(
        name=COL_REWRITTEN_TEXT,
        generator_function=finalize,
        generator_params=_FinalizationParams(
            flag_utility_below=evaluation.flag_utility_below,
            flag_leakage_above=evaluation.flag_leakage_above,
        ),
        propagate_skip=False,
    )


class CombinedRewriteWorkflow(RewriteWorkflow):
    """Proof-of-concept rewrite workflow executed as one DataDesigner graph."""

    def __init__(self, adapter: NddAdapter) -> None:
        super().__init__(adapter)

    def build_graph(
        self,
        *,
        selected_models: RewriteModelSelection,
        replace_model_selection: ReplaceModelSelection,
        privacy_goal: PrivacyGoal,
        evaluation: EvaluationCriteria,
        data_summary: str | None = None,
        strict_entity_protection: bool = False,
    ) -> CombinedRewriteGraph:
        columns = _replacement_columns(replace_model_selection)
        columns.extend(
            DomainClassificationWorkflow().columns(selected_models=selected_models, data_summary=data_summary)
        )
        columns.extend(
            SensitivityDispositionWorkflow().columns(
                selected_models=selected_models,
                privacy_goal=privacy_goal,
                data_summary=data_summary,
                strict_entity_protection=strict_entity_protection,
            )
        )
        columns.extend(QAGenerationWorkflow().columns(selected_models=selected_models))

        rewrite_columns = RewriteGenerationWorkflow().columns(
            selected_models=selected_models,
            privacy_goal=privacy_goal,
            data_summary=data_summary,
        )
        columns.extend(
            _remap_custom_column(column, {COL_REWRITTEN_TEXT: COL_REWRITTEN_TEXT_INITIAL})
            if column.name == COL_REWRITTEN_TEXT
            else column
            for column in rewrite_columns
        )

        evaluation_states = [_evaluation_state(0, COL_REWRITTEN_TEXT_INITIAL)]
        repair_states: list[RepairState] = []
        columns.extend(
            _evaluation_columns(
                self._adapter,
                selected_models=selected_models,
                evaluation=evaluation,
                state=evaluation_states[0],
            )
        )

        for iteration in range(evaluation.max_repair_iterations):
            repair_state = _repair_state(iteration)
            repair_states.append(repair_state)
            columns.extend(
                _repair_columns(
                    self._adapter,
                    selected_models=selected_models,
                    privacy_goal=privacy_goal,
                    evaluation=evaluation,
                    previous=evaluation_states[-1],
                    state=repair_state,
                )
            )
            evaluation_state = _evaluation_state(iteration + 1, repair_state.rewritten_text)
            evaluation_states.append(evaluation_state)
            columns.extend(
                _evaluation_columns(
                    self._adapter,
                    selected_models=selected_models,
                    evaluation=evaluation,
                    state=evaluation_state,
                )
            )

        columns.append(_finalization_column(evaluation_states, evaluation))
        internal_columns = [
            COL_ENTITY_EXAMPLES,
            COL_ENTITIES_FOR_REPLACE,
            COL_ENTITIES_FOR_REPLACE_JSON,
            COL_REPLACEMENT_MAP_RAW,
            COL_REWRITTEN_TEXT_INITIAL,
            *(column for state in evaluation_states for column in state.__dict__.values() if isinstance(column, str)),
            *(column for state in repair_states for column in state.__dict__.values() if isinstance(column, str)),
        ]
        return CombinedRewriteGraph(
            columns=columns,
            evaluation_states=evaluation_states,
            repair_states=repair_states,
            internal_columns=list(dict.fromkeys(internal_columns)),
        )

    def run(
        self,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: RewriteModelSelection,
        replace_model_selection: ReplaceModelSelection,
        privacy_goal: PrivacyGoal,
        evaluation: EvaluationCriteria,
        data_summary: str | None = None,
        preview_num_records: int | None = None,
        strict_entity_protection: bool = False,
    ) -> RewriteResult:
        with stage_timer("CombinedRewriteWorkflow.run", input_row_count=len(dataframe)) as measurement:
            entity_rows, passthrough_rows = split_rows(
                dataframe,
                column=COL_ENTITIES_BY_VALUE,
                predicate=_has_entities,
            )
            measurement.update(
                entity_row_count=len(entity_rows),
                passthrough_row_count=len(passthrough_rows),
            )
            if entity_rows.empty:
                _apply_passthrough_defaults(passthrough_rows)
                result = RewriteResult(dataframe=merge_and_reorder(passthrough_rows), failed_records=[])
                measurement.update(output_row_count=len(result.dataframe), failed_record_count=0)
                return result

            graph = self.build_graph(
                selected_models=selected_models,
                replace_model_selection=replace_model_selection,
                privacy_goal=privacy_goal,
                evaluation=evaluation,
                data_summary=data_summary,
                strict_entity_protection=strict_entity_protection,
            )
            seed = select_seed_cols(entity_rows, derive_seed_columns(graph.columns, entity_rows))
            run_result = self._adapter.run_workflow(
                seed,
                model_configs=model_configs,
                columns=graph.columns,
                workflow_name="rewrite-combined",
                preview_num_records=preview_num_records,
            )
            entity_rows = _join_new_columns(entity_rows, run_result.dataframe)
            entity_rows = entity_rows.drop(columns=graph.internal_columns, errors="ignore")
            _apply_passthrough_defaults(passthrough_rows)
            result = RewriteResult(
                dataframe=merge_and_reorder(entity_rows, passthrough_rows),
                failed_records=run_result.failed_records,
                failed_row_evidence=run_result.failed_row_evidence,
            )
            measurement.update(
                output_row_count=len(result.dataframe),
                failed_record_count=len(result.failed_records),
            )
            return result
