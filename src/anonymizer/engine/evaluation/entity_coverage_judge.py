# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from typing import ClassVar

import pandas as pd
from data_designer.config.column_configs import LLMTextColumnConfig
from data_designer.config.models import ModelConfig
from pydantic import BaseModel, Field

from anonymizer.config.models import EvaluateModelSelection
from anonymizer.engine.constants import (
    COL_ENTITIES_BY_VALUE,
    COL_ENTITY_COVERAGE,
    COL_ENTITY_COVERAGE_JUDGE,
    COL_LEAKED_ENTITIES,
    COL_TEXT,
    DEFAULT_ENTITY_LABELS,
    _jinja,
)
from anonymizer.engine.evaluation.judge_base import JudgeResult, _BaseJudgeWorkflow
from anonymizer.engine.ndd.adapter import FailedRecord, NddAdapter
from anonymizer.engine.ndd.model_loader import resolve_model_alias
from anonymizer.engine.prompt_utils import substitute_placeholders
from anonymizer.engine.row_partitioning import ROW_ORDER_COL, merge_and_reorder
from anonymizer.engine.schemas import EntitiesByValueSchema

logger = logging.getLogger("anonymizer.evaluation.entity_coverage_judge")

_ENTITIES_FOR_COVERAGE_COL = "_entities_for_coverage_judge"
_N_ENTITIES_DETECTED_COL = "_n_entities_detected_for_coverage"


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class LeakedEntity(BaseModel):
    value: str = Field(description="Exact text span from the original text that was missed by the anonymizer.")
    label: str = Field(description="Entity type label (e.g. first_name, email, phone_number).")
    reasoning: str = Field(description="One sentence explaining why this is PII that the anonymizer missed.")


class EntityCoverageSchema(BaseModel):
    leaked_entities: list[LeakedEntity] = Field(
        default_factory=list,
        description="All PII entities present in the original text that the anonymizer failed to detect. "
        "Empty when the anonymizer caught everything.",
    )


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def _entity_type_scope_block(entity_labels: list[str] | None) -> str:
    if entity_labels is None:
        return "<entity_type_scope>\nEvaluate for all PII and sensitive entity types.\n</entity_type_scope>"
    labels_str = ", ".join(entity_labels)
    return (
        "<entity_type_scope>\n"
        f"Detection was configured to target ONLY these entity types: {labels_str}.\n"
        "Only report missed entities that belong to one of these types. "
        "Do NOT flag PII of other types as leaked — those were intentionally excluded from detection.\n"
        "</entity_type_scope>"
    )


def _strict_protection_block(strict_entity_protection: bool) -> str:
    if not strict_entity_protection:
        return ""
    return (
        "<strict_entity_protection>\n"
        "STRICT PROTECTION MODE IS ENABLED.\n\n"
        "Flag ALL entities as leaked if they were not caught — including quasi-identifiers\n"
        "and low-risk entities that would normally be given benefit of the doubt.\n"
        "Do NOT apply MINIMUM NECESSARY CHANGE reasoning to excuse a missed entity.\n"
        "Do NOT excuse a missed entity because its combined re-identification risk is low.\n"
        "Any PII span not caught by the anonymizer is a miss in strict mode.\n"
        "</strict_entity_protection>"
    )


def _coverage_prompt(
    *,
    entity_labels: list[str] | None,
    strict_entity_protection: bool,
) -> str:
    entity_scope_block = _entity_type_scope_block(entity_labels)
    strict_block = _strict_protection_block(strict_entity_protection)

    entity_scope_guidance = (
        "- Respect the entity_type_scope: do not flag PII types outside the configured scope as leaked."
        if entity_labels is not None
        else ""
    )
    strict_guidance = (
        "- In strict protection mode, flag every missed entity regardless of its perceived risk level."
        if strict_entity_protection
        else ""
    )

    active_labels = entity_labels if entity_labels is not None else DEFAULT_ENTITY_LABELS
    labels_str = ", ".join(active_labels)

    prompt = f"""You are a privacy auditor. Given ORIGINAL text and a list of PII ENTITIES that were \
detected by an anonymization pipeline, identify any direct or quasi-identifiers in the \
original text that are NOT covered by the detected entities.

<original_text>
<<COL_TEXT>>
</original_text>

<detected_entities>
{{%- if <<ENTITIES_COLUMN>> %}}
{{%- for entity in <<ENTITIES_COLUMN>> %}}
- value="{{{{ entity.value }}}}" | label={{{{ entity.label }}}}
{{%- endfor %}}
{{%- else %}}
(none detected)
{{%- endif %}}
</detected_entities>

{entity_scope_block}

{strict_block}

<coverage_definition>
An identifier is "covered" (already protected) if:
- Its exact value appears in the detected entities list, OR
- Its value is a substring of any entity value in the list (e.g. a bare username is \
covered if a full file path containing that username is listed as an entity).
</coverage_definition>

<identifier_taxonomy>
The following entity types are considered sensitive and must be flagged if missed: {labels_str}.
Quasi-identifiers: combinations of values that together re-identify someone \
(e.g. job title + employer + city appearing together). Time values (specific timestamps, \
times of day, schedules) can act as quasi-identifiers when combined with other attributes \
in the same text — flag them if they appear alongside other identifying information.
</identifier_taxonomy>

<task>
Identify every direct or quasi-identifier present in the original text that is NOT covered \
by the detected entities. These are the "leaked" entities — identifiers that would survive \
anonymization because the detector missed them.

Only report findings you are HIGH CONFIDENCE about.

Return structured JSON:
- `leaked_entities`: list every missed identifier with its `value`, `label`, and a short `reasoning`.
- Return an empty list if the anonymizer covered all identifiers.
</task>

<guidance>
Do NOT flag:
- Items that are substrings of, or closely matched by, a listed entity value.
- Generic technical terms, common words, or place names that cannot re-identify a specific individual on their own.
- Information that is inferable but not literally present in the text.

Do flag:
- The `value` MUST be a literal substring found in the original text.
- `reasoning` MUST be one sentence explaining why this value is not covered by the detected entities.
- If `detected_entities` is empty, scan the full text for any direct or quasi-identifiers.
{entity_scope_guidance}
{strict_guidance}
</guidance>

<output_format>
Return ONLY the JSON object that matches the required schema. Do NOT include any commentary, \
reasoning, preamble, or text outside the JSON object.
</output_format>
"""
    return substitute_placeholders(
        prompt,
        {
            "<<COL_TEXT>>": _jinja(COL_TEXT),
            "<<ENTITIES_COLUMN>>": _ENTITIES_FOR_COVERAGE_COL,
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entities_for_coverage(parsed: EntitiesByValueSchema) -> list[dict[str, str]]:
    """Flatten EntitiesByValueSchema into one (value, label) row per pair for prompt context."""
    return [{"value": e.value, "label": label} for e in parsed.entities_by_value for label in e.labels]


def _parse_leaked_entities(raw: object) -> list[dict[str, object]] | None:
    """Parse raw LLM output into the leaked entities list.

    Returns the list (possibly empty) on success, or None when the payload is
    malformed or missing so downstream display renders "judge unavailable".
    """
    if raw is None:
        return None
    if isinstance(raw, BaseModel):
        raw = raw.model_dump(mode="python")
    if isinstance(raw, str):
        # Strip optional ```json ... ``` fence before parsing so models that
        # return fenced or unfenced JSON are both handled.
        stripped = raw.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            raw = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            return None
    if not isinstance(raw, dict):
        return None
    try:
        parsed = EntityCoverageSchema.model_validate(raw)
    except Exception:
        return None
    return [e.model_dump() for e in parsed.leaked_entities]


def _compute_coverage(n_detected: int, n_leaked: int) -> float:
    total = n_detected + n_leaked
    return 1.0 if total == 0 else n_detected / total


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class EntityCoverageWorkflow(_BaseJudgeWorkflow):
    """LLM-as-judge that measures recall: what fraction of PII did the anonymizer catch?

    Runs on the **original text** and compares all PII present against what the
    anonymizer already detected.  The delta gives the leaked (missed) entities.

    Output columns:
      ``COL_ENTITY_COVERAGE`` (float|None) — n_detected / (n_detected + n_leaked)
      ``COL_LEAKED_ENTITIES`` (list)        — missed entities with value, label, reasoning
    """

    RAW_COL: ClassVar[str] = COL_ENTITY_COVERAGE_JUDGE
    VALID_COL: ClassVar[str] = COL_ENTITY_COVERAGE
    INVALID_COL: ClassVar[str] = COL_LEAKED_ENTITIES
    SCHEMA: ClassVar[type[BaseModel]] = EntityCoverageSchema
    VERDICT_FIELD: ClassVar[str] = "leaked_entities"
    DEFAULT_PAYLOAD: ClassVar[dict] = {"leaked_entities": []}
    MODEL_ROLE: ClassVar[str] = "entity_coverage_judge"
    WORKFLOW_NAME: ClassVar[str] = "entity-coverage-judge"

    def __init__(
        self,
        adapter: NddAdapter,
        *,
        entity_labels: list[str] | None = None,
        strict_entity_protection: bool = False,
    ) -> None:
        super().__init__(adapter)
        self._entity_labels = entity_labels
        self._strict_entity_protection = strict_entity_protection

    # ------------------------------------------------------------------ hooks

    def prepare(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        working_df = dataframe.copy()
        parsed = working_df[COL_ENTITIES_BY_VALUE].apply(EntitiesByValueSchema.from_raw)
        working_df[_ENTITIES_FOR_COVERAGE_COL] = parsed.apply(_entities_for_coverage)
        working_df[_N_ENTITIES_DETECTED_COL] = working_df[_ENTITIES_FOR_COVERAGE_COL].apply(len)
        return working_df

    def _passthrough_mask(self, dataframe: pd.DataFrame) -> pd.Series:
        return dataframe[_ENTITIES_FOR_COVERAGE_COL].apply(lambda items: items is None or len(items) == 0)

    @classmethod
    def _build_prompt(cls) -> str:
        return _coverage_prompt(entity_labels=None, strict_entity_protection=False)

    @classmethod
    def _extract_invalid(cls, parsed: BaseModel) -> list[dict[str, object]]:
        return [e.model_dump() for e in parsed.leaked_entities]

    # ----------------------------------------------------------------- overrides

    def column_config(self, selected_models: EvaluateModelSelection) -> LLMTextColumnConfig:
        """Override to inject instance-specific entity_labels and strict_entity_protection."""
        return LLMTextColumnConfig(
            name=self.RAW_COL,
            prompt=_coverage_prompt(
                entity_labels=self._entity_labels,
                strict_entity_protection=self._strict_entity_protection,
            ),
            model_alias=resolve_model_alias(self.MODEL_ROLE, selected_models),
        )

    def postprocess(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Override to write float coverage instead of bool verdict."""
        out = dataframe.copy()
        passthrough_mask = self._passthrough_mask(out)

        coverage_vals: list[float | None] = []
        leaked_lists: list[list[dict]] = []

        for idx in out.index:
            if passthrough_mask.loc[idx]:
                coverage_vals.append(1.0)
                leaked_lists.append([])
            else:
                raw = out[self.RAW_COL].loc[idx] if self.RAW_COL in out.columns else None
                leaked = _parse_leaked_entities(raw)
                n_detected = (
                    int(out[_N_ENTITIES_DETECTED_COL].loc[idx]) if _N_ENTITIES_DETECTED_COL in out.columns else 0
                )
                if leaked is None:
                    coverage_vals.append(None)
                    leaked_lists.append([])
                else:
                    coverage_vals.append(_compute_coverage(n_detected, len(leaked)))
                    leaked_lists.append(leaked)

        out[self.VALID_COL] = coverage_vals
        out[self.INVALID_COL] = leaked_lists
        if self.RAW_COL in out.columns:
            out.loc[passthrough_mask, self.RAW_COL] = [self.DEFAULT_PAYLOAD] * int(passthrough_mask.sum())
        return out

    def run_non_critical(
        self,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: EvaluateModelSelection,
        preview_num_records: int | None = None,
    ) -> tuple[pd.DataFrame, list[FailedRecord]]:
        """Run coverage and annotate ``dataframe`` in-place; never raise.

        Rows the LLM drops get ``entity_coverage=None`` / ``leaked_entities=[]``
        rather than disappearing. On total workflow failure, all rows are defaulted.
        Returns ``(annotated_df, failed_records)``.
        """
        try:
            result = self.evaluate(
                dataframe,
                model_configs=model_configs,
                selected_models=selected_models,
                preview_num_records=preview_num_records,
            )
            out = dataframe.copy()
            for col in (self.VALID_COL, self.INVALID_COL):
                if col in result.dataframe.columns:
                    out[col] = result.dataframe[col].values
            return out, result.failed_records
        except Exception:
            logger.warning("Entity coverage step failed; populating defaults", exc_info=True)
            out = dataframe.copy()
            out[self.VALID_COL] = None
            out[self.INVALID_COL] = [[] for _ in range(len(out))]
            return out, []

    def evaluate(
        self,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: EvaluateModelSelection,
        preview_num_records: int | None = None,
    ) -> JudgeResult:
        """Standalone entry point. Overrides base to write float coverage instead of bool."""
        working_df = self.prepare(dataframe)
        working_df[ROW_ORDER_COL] = range(len(working_df))
        passthrough_mask = self._passthrough_mask(working_df)
        passthrough_rows = working_df[passthrough_mask].copy()
        with_content = working_df[~passthrough_mask].copy()

        passthrough_rows[self.RAW_COL] = [self.DEFAULT_PAYLOAD for _ in range(len(passthrough_rows))]
        passthrough_rows[self.VALID_COL] = 1.0
        passthrough_rows[self.INVALID_COL] = [[] for _ in range(len(passthrough_rows))]

        if with_content.empty:
            combined = merge_and_reorder(passthrough_rows)
            return JudgeResult(dataframe=combined, failed_records=[])

        effective_preview = min(preview_num_records, len(with_content)) if preview_num_records is not None else None
        run_result = self._adapter.run_workflow(
            with_content,
            model_configs=model_configs,
            columns=[self.column_config(selected_models)],
            workflow_name=self.WORKFLOW_NAME,
            preview_num_records=effective_preview,
        )

        judged_df = run_result.dataframe.copy()
        coverage_vals: list[float | None] = []
        leaked_lists: list[list[dict]] = []
        for idx in judged_df.index:
            raw = judged_df[self.RAW_COL].loc[idx] if self.RAW_COL in judged_df.columns else None
            leaked = _parse_leaked_entities(raw)
            n_detected = (
                int(judged_df[_N_ENTITIES_DETECTED_COL].loc[idx])
                if _N_ENTITIES_DETECTED_COL in judged_df.columns
                else 0
            )
            if leaked is None:
                coverage_vals.append(None)
                leaked_lists.append([])
            else:
                coverage_vals.append(_compute_coverage(n_detected, len(leaked)))
                leaked_lists.append(leaked)
        judged_df[self.VALID_COL] = coverage_vals
        judged_df[self.INVALID_COL] = leaked_lists

        combined = merge_and_reorder(judged_df, passthrough_rows)
        return JudgeResult(dataframe=combined, failed_records=run_result.failed_records)
