# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from typing import ClassVar, cast

import pandas as pd
from pydantic import BaseModel, Field

from anonymizer.engine.constants import (
    COL_DETECTION_INVALID_ENTITIES,
    COL_DETECTION_JUDGE,
    COL_DETECTION_VALID,
    COL_ENTITIES_BY_VALUE,
    COL_TEXT,
    ENTITY_LABEL_EXAMPLES,
    _jinja,
)
from anonymizer.engine.evaluation.judge_base import _BaseJudgeWorkflow
from anonymizer.engine.prompt_utils import substitute_placeholders
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


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class DetectionJudgeWorkflow(_BaseJudgeWorkflow):
    """LLM-as-judge evaluator that flags invalid PII detections per record.

    Runs after replacement and validates the detection step that fed the
    replacement. Output columns: ``COL_DETECTION_VALID`` (bool|None) and
    ``COL_DETECTION_INVALID_ENTITIES`` (list of {value, label, reasoning}).
    """

    RAW_COL: ClassVar[str] = COL_DETECTION_JUDGE
    VALID_COL: ClassVar[str] = COL_DETECTION_VALID
    INVALID_COL: ClassVar[str] = COL_DETECTION_INVALID_ENTITIES
    SCHEMA: ClassVar[type[BaseModel]] = DetectionJudgmentSchema
    VERDICT_FIELD: ClassVar[str] = "all_valid"
    DEFAULT_PAYLOAD: ClassVar[dict] = {"all_valid": True, "invalid_entities": []}
    MODEL_ROLE: ClassVar[str] = "detection_validity_judge"
    WORKFLOW_NAME: ClassVar[str] = "replace-detection-judge"

    def prepare(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        working_df = dataframe.copy()

        def _safe_parse(raw: object) -> EntitiesByValueSchema:
            try:
                return EntitiesByValueSchema.from_raw(raw)
            except Exception:
                logger.warning(
                    "Could not parse entities_by_value for a row; treating as no entities.",
                    exc_info=True,
                )
                return EntitiesByValueSchema(entities_by_value=[])

        parsed = working_df[COL_ENTITIES_BY_VALUE].apply(_safe_parse)
        working_df[_ENTITIES_FOR_JUDGE_COL] = parsed.apply(_entities_for_judge)
        working_df[_ENTITY_EXAMPLES_FOR_JUDGE_COL] = parsed.apply(_label_examples_for_judge)
        return working_df

    def _passthrough_mask(self, dataframe: pd.DataFrame) -> pd.Series:
        # `items` may be a numpy array after a parquet round-trip via DD, so use
        # `len()` rather than `bool()` (which is ambiguous on multi-element arrays).
        return dataframe[_ENTITIES_FOR_JUDGE_COL].apply(lambda items: items is None or len(items) == 0)

    @classmethod
    def _build_prompt(cls) -> str:
        return _judge_prompt()

    @classmethod
    def _extract_invalid(cls, parsed: BaseModel) -> list[dict[str, object]]:
        parsed = cast(DetectionJudgmentSchema, parsed)
        return [entry.model_dump() for entry in parsed.invalid_entities]
