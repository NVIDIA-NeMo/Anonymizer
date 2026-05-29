# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import pandas as pd
from data_designer.config.column_configs import LLMStructuredColumnConfig
from data_designer.config.models import ModelConfig
from pydantic import BaseModel, Field

from anonymizer.config.models import EvaluateModelSelection
from anonymizer.engine.constants import (
    COL_DETECTION_INVALID_ENTITIES,
    COL_DETECTION_JUDGE,
    COL_DETECTION_VALID,
    COL_ENTITIES_BY_VALUE,
    COL_TEXT,
    ENTITY_LABEL_EXAMPLES,
    _jinja,
)
from anonymizer.engine.ndd.adapter import FailedRecord, NddAdapter
from anonymizer.engine.ndd.model_loader import resolve_model_alias
from anonymizer.engine.prompt_utils import substitute_placeholders
from anonymizer.engine.row_partitioning import merge_and_reorder, split_rows
from anonymizer.engine.schemas import EntitiesByValueSchema

logger = logging.getLogger("anonymizer.evaluation.detection_judge")

_ENTITIES_FOR_JUDGE_COL = "_entities_for_detection_judge"
_ENTITY_EXAMPLES_FOR_JUDGE_COL = "_entity_examples_for_detection_judge"


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class InvalidDetectedEntity(BaseModel):
    value: str = Field(description="Original detected span verbatim.")
    label: str = Field(description="The label the detector assigned to this span.")
    reasoning: str = Field(
        description="One short sentence explaining why this (value, label) is not a valid detection."
    )


class DetectionJudgmentSchema(BaseModel):
    all_valid: bool = Field(
        description="True only if every detected entity is a correct (value, label) detection in context."
    )
    invalid_entities: list[InvalidDetectedEntity] = Field(
        default_factory=list,
        description="Every detected entity that is not a valid detection. Empty when all_valid is True.",
    )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectionJudgeResult:
    dataframe: pd.DataFrame
    failed_records: list[FailedRecord]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def _judge_prompt() -> str:
    prompt = """You are an expert judge evaluating the accuracy of automated PII / sensitive-entity detection.

<original_text>
<<COL_TEXT>>
</original_text>

<detected_entities>
{%- for entity in <<ENTITIES_COLUMN>> %}
- value="{{ entity.value }}" | label={{ entity.label }}
{%- endfor %}
</detected_entities>

<reference_label_examples>
{{ <<EXAMPLES_COLUMN>> }}
</reference_label_examples>

<task>
For each detected entity above, decide whether the (value, label) pair is a correct PII / sensitive-entity \
detection given the original text.

Return structured JSON:
- Set `all_valid` to true ONLY if every detected entity is a correct detection.
- Otherwise set `all_valid` to false and list every incorrect detection in `invalid_entities`, with a short \
`reasoning` per entry.
</task>

<invalid_criteria>
A detection is INVALID if any of the following hold:
- false_positive: the span is not actually identifying or sensitive in this context (e.g. a common word, \
generic phrase, or boilerplate flagged as PII).
- wrong_label: the span IS sensitive, but the chosen label does not fit. Treat labels as \
BUCKETS, not precise taxonomy nodes. A label is NOT wrong when the chosen label is a SIBLING \
within the same broad domain — a more specific, more general, or peer member of the same \
parent concept (e.g. higher-education institutions, organizational entities, geographic \
places, communication identifiers). Only flag `wrong_label` when the chosen label sits in a \
clearly DIFFERENT domain (e.g. a company name labeled `first_name`; an email labeled `url`; \
a job title labeled `degree`).
- not_in_text: the literal `value` does not appear in the original text.
- wrong_boundary: the span is a clear partial or over-extended capture of the real entity. \
Flag this ONLY when the span itself is broken — i.e. it omits part of the actual value, or \
it absorbs surrounding tokens (titles, prepositions, conjunctions, function words) that are \
not part of the value. \
Treat the span as CORRECT when it captures the bare value of the entity, even if that value \
appears inside a longer descriptive phrase or compound expression. Surrounding descriptive \
words in natural prose are NOT part of the entity, and trimming them is the right behavior, \
not a boundary error. Apply the "form-field" test: if you were filling out a structured form \
for this entity type, the bare value would be the answer.
- contextual_mismatch: in this context the span refers to something other than the labeled entity type \
(e.g. "Apple" used as the fruit and labeled `company_name`; "May" used as a verb/month and labeled `first_name`).
</invalid_criteria>

<valid_criteria>
A detection is VALID when ALL of the following hold:
- The `value` appears in the original text.
- The chosen label is a reasonable fit; if multiple labels could plausibly apply, the chosen one is \
acceptable.
- The span is a complete and reasonable boundary for the entity in context.
- Removing or replacing this span meaningfully contributes to anonymizing the record.
</valid_criteria>

<guidance>
- Use `reference_label_examples` as a guide for what each label is supposed to capture; do not invent labels \
that are not in that mapping.
- Be charitable when multiple labels could plausibly apply: if the chosen label is reasonable, mark valid.
- Be strict about clear false positives, mislabels, and obvious boundary errors.
- `reasoning` MUST be one short sentence per invalid entity, naming the failure mode.
- If `detected_entities` is empty, return `all_valid: true` and an empty `invalid_entities` list.
- Do NOT include entities you consider valid in `invalid_entities`.
- Do NOT introduce entities that were not in the detected list.
</guidance>

<output_format>
Return ONLY the JSON object that matches the required schema. Do NOT wrap your output in \
``` or ```json markdown fences. Do NOT include any commentary, reasoning, preamble, or text \
outside the JSON object. Your entire response must be a single valid JSON object.
</output_format>
"""
    return substitute_placeholders(
        prompt,
        {
            "<<COL_TEXT>>": _jinja(COL_TEXT),
            "<<ENTITIES_COLUMN>>": _ENTITIES_FOR_JUDGE_COL,
            "<<EXAMPLES_COLUMN>>": _ENTITY_EXAMPLES_FOR_JUDGE_COL,
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_EXAMPLE_LOOKUP: dict[str, str] = {
    label: f"(e.g. {', '.join(examples)})" for label, examples in ENTITY_LABEL_EXAMPLES.items()
}


def _entities_for_judge(parsed: EntitiesByValueSchema) -> list[dict[str, str]]:
    """Flatten EntitiesByValueSchema into one (value, label) row per pair.

    The judge schema and the display denominator both work at (value, label)
    granularity, so the prompt input mirrors that shape instead of grouping
    labels under a single value.
    """
    return [{"value": e.value, "label": label} for e in parsed.entities_by_value for label in e.labels]


def _label_examples_for_judge(parsed: EntitiesByValueSchema) -> str:
    """Build a JSON map of {label: example_hint} for labels present in this row."""
    labels: set[str] = set()
    for entity in parsed.entities_by_value:
        labels.update(label for label in entity.labels if label)
    if not labels:
        return "{}"
    examples = {label: _EXAMPLE_LOOKUP.get(label, "(no canonical example available)") for label in sorted(labels)}
    return json.dumps(examples, ensure_ascii=True)


def _flatten_judgment(raw: object) -> tuple[bool | None, list[dict[str, str]]]:
    """Normalize an LLM judge output into (all_valid, invalid_entities).

    Returns ``(None, [])`` for any malformed or missing payload so downstream
    display can render "judge unavailable" rather than fabricate a verdict.
    """
    if raw is None:
        return None, []
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump(mode="python")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None, []
    if not isinstance(raw, dict):
        return None, []
    try:
        parsed = DetectionJudgmentSchema.model_validate(raw)
    except Exception:
        return None, []
    return parsed.all_valid, [entry.model_dump() for entry in parsed.invalid_entities]


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class DetectionJudgeWorkflow:
    """LLM-as-judge evaluator that flags invalid PII detections per record.

    Runs after replacement and validates the detection step that fed the
    replacement. Output columns: ``COL_DETECTION_VALID`` (bool|None) and
    ``COL_DETECTION_INVALID_ENTITIES`` (list of {value, label, reasoning}).
    """

    def __init__(self, adapter: NddAdapter) -> None:
        self._adapter = adapter

    # ------------------------------------------------------------------------
    # Decomposed pieces — the orchestrator in ReplacementWorkflow uses these
    # to merge all 4 judges into a single adapter.run_workflow() call.
    # ------------------------------------------------------------------------

    def prepare(
        self,
        dataframe: pd.DataFrame,
        *,
        entities_column: str = COL_ENTITIES_BY_VALUE,
    ) -> pd.DataFrame:
        """Add the intermediate columns this judge's prompt template references.

        Returns a copy of ``dataframe`` with ``_entities_for_detection_judge`` and
        ``_entity_examples_for_detection_judge`` populated.
        """
        working_df = dataframe.copy()
        parsed = working_df[entities_column].apply(EntitiesByValueSchema.from_raw)
        working_df[_ENTITIES_FOR_JUDGE_COL] = parsed.apply(_entities_for_judge)
        working_df[_ENTITY_EXAMPLES_FOR_JUDGE_COL] = parsed.apply(_label_examples_for_judge)
        return working_df

    def column_config(self, selected_models: EvaluateModelSelection) -> LLMStructuredColumnConfig:
        """The DD column config — name, prompt, model alias, structured-output schema."""
        return LLMStructuredColumnConfig(
            name=COL_DETECTION_JUDGE,
            prompt=_judge_prompt(),
            model_alias=resolve_model_alias("detection_validity_judge", selected_models),
            output_format=DetectionJudgmentSchema,
        )

    def postprocess(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Flatten the raw judge output into VALID / INVALID columns and apply
        the passthrough default (rows with no detected entities trivially pass).
        """
        out = dataframe.copy()
        flattened = out[COL_DETECTION_JUDGE].apply(_flatten_judgment) if COL_DETECTION_JUDGE in out.columns else None
        # `items` may be a numpy array after a parquet round-trip via DD, so use
        # `len()` rather than `bool()` (which is ambiguous on multi-element arrays).
        passthrough_mask = out[_ENTITIES_FOR_JUDGE_COL].apply(lambda items: items is None or len(items) == 0)

        valid: list[bool | None] = []
        invalid: list[list[dict[str, str]]] = []
        for idx in out.index:
            if passthrough_mask.loc[idx]:
                valid.append(True)
                invalid.append([])
            elif flattened is not None:
                v, inv = flattened.loc[idx]
                valid.append(v)
                invalid.append(inv)
            else:
                valid.append(None)
                invalid.append([])
        out[COL_DETECTION_VALID] = valid
        out[COL_DETECTION_INVALID_ENTITIES] = invalid
        # Stamp passthrough rows with the default raw judge payload so display logic stays consistent.
        if COL_DETECTION_JUDGE in out.columns:
            out.loc[passthrough_mask, COL_DETECTION_JUDGE] = [{"all_valid": True, "invalid_entities": []}] * int(
                passthrough_mask.sum()
            )
        return out

    # ------------------------------------------------------------------------
    # Legacy single-judge entry point. Kept so existing callers/tests still work.
    # ------------------------------------------------------------------------

    def evaluate(
        self,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: EvaluateModelSelection,
        entities_column: str = COL_ENTITIES_BY_VALUE,
        preview_num_records: int | None = None,
    ) -> DetectionJudgeResult:
        working_df = self.prepare(dataframe, entities_column=entities_column)

        entity_rows, passthrough_rows = split_rows(working_df, column=_ENTITIES_FOR_JUDGE_COL, predicate=bool)
        passthrough_rows[COL_DETECTION_JUDGE] = [
            {"all_valid": True, "invalid_entities": []} for _ in range(len(passthrough_rows))
        ]
        passthrough_rows[COL_DETECTION_VALID] = True
        passthrough_rows[COL_DETECTION_INVALID_ENTITIES] = [[] for _ in range(len(passthrough_rows))]

        if entity_rows.empty:
            combined = merge_and_reorder(passthrough_rows)
            return DetectionJudgeResult(dataframe=combined, failed_records=[])

        effective_preview_num_records = (
            min(preview_num_records, len(entity_rows)) if preview_num_records is not None else None
        )
        run_result = self._adapter.run_workflow(
            entity_rows,
            model_configs=model_configs,
            columns=[self.column_config(selected_models)],
            workflow_name="replace-detection-judge",
            preview_num_records=effective_preview_num_records,
        )

        judged_df = run_result.dataframe.copy()
        flattened = judged_df[COL_DETECTION_JUDGE].apply(_flatten_judgment)
        judged_df[COL_DETECTION_VALID] = flattened.apply(lambda pair: pair[0])
        judged_df[COL_DETECTION_INVALID_ENTITIES] = flattened.apply(lambda pair: pair[1])

        combined = merge_and_reorder(judged_df, passthrough_rows)
        return DetectionJudgeResult(dataframe=combined, failed_records=run_result.failed_records)
