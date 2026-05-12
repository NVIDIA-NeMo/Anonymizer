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

from anonymizer.config.models import ReplaceModelSelection
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

logger = logging.getLogger("anonymizer.replace.detection_judge")

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
- "{{ entity.value }}" -> {{ entity.labels_str }}
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
- wrong_label: the span IS sensitive, but the chosen label does not fit (e.g. a company name labeled \
`first_name`; an email labeled `url`; a job title labeled `degree`).
- not_in_text: the literal `value` does not appear in the original text.
- wrong_boundary: the span is a clear partial or over-extended capture of the real entity \
(e.g. "John" labeled as full `last_name`; "Dr. John Smith MD" labeled as `first_name`).
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


def _entities_for_judge(parsed: EntitiesByValueSchema) -> list[dict[str, str | list[str]]]:
    """Flatten EntitiesByValueSchema into Jinja-friendly dicts."""
    return [
        {"value": e.value, "labels": e.labels, "labels_str": ", ".join(e.labels)}
        for e in parsed.entities_by_value
    ]


def _label_examples_for_judge(parsed: EntitiesByValueSchema) -> str:
    """Build a JSON map of {label: example_hint} for labels present in this row."""
    labels: set[str] = set()
    for entity in parsed.entities_by_value:
        labels.update(label for label in entity.labels if label)
    if not labels:
        return "{}"
    examples = {
        label: _EXAMPLE_LOOKUP.get(label, "(no canonical example available)") for label in sorted(labels)
    }
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

    def evaluate(
        self,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: ReplaceModelSelection,
        entities_column: str = COL_ENTITIES_BY_VALUE,
        preview_num_records: int | None = None,
    ) -> DetectionJudgeResult:
        judge_alias = resolve_model_alias("detection_judge", selected_models)

        working_df = dataframe.copy()
        parsed_entities = working_df[entities_column].apply(EntitiesByValueSchema.from_raw)
        working_df[_ENTITIES_FOR_JUDGE_COL] = parsed_entities.apply(_entities_for_judge)
        working_df[_ENTITY_EXAMPLES_FOR_JUDGE_COL] = parsed_entities.apply(_label_examples_for_judge)

        # Rows with no detected entities trivially pass — skip the LLM call.
        entity_rows, passthrough_rows = split_rows(
            working_df, column=_ENTITIES_FOR_JUDGE_COL, predicate=bool
        )
        passthrough_rows[COL_DETECTION_JUDGE] = [
            {"all_valid": True, "invalid_entities": []} for _ in range(len(passthrough_rows))
        ]
        passthrough_rows[COL_DETECTION_VALID] = True
        passthrough_rows[COL_DETECTION_INVALID_ENTITIES] = [[] for _ in range(len(passthrough_rows))]

        if entity_rows.empty:
            combined = merge_and_reorder(passthrough_rows, attrs=dataframe.attrs)
            return DetectionJudgeResult(dataframe=combined, failed_records=[])

        effective_preview_num_records = (
            min(preview_num_records, len(entity_rows)) if preview_num_records is not None else None
        )
        run_result = self._adapter.run_workflow(
            entity_rows,
            model_configs=model_configs,
            columns=[
                LLMStructuredColumnConfig(
                    name=COL_DETECTION_JUDGE,
                    prompt=_judge_prompt(),
                    model_alias=judge_alias,
                    output_format=DetectionJudgmentSchema,
                )
            ],
            workflow_name="replace-detection-judge",
            preview_num_records=effective_preview_num_records,
        )

        judged_df = run_result.dataframe.copy()
        flattened = judged_df[COL_DETECTION_JUDGE].apply(_flatten_judgment)
        judged_df[COL_DETECTION_VALID] = flattened.apply(lambda pair: pair[0])
        judged_df[COL_DETECTION_INVALID_ENTITIES] = flattened.apply(lambda pair: pair[1])

        combined = merge_and_reorder(
            judged_df, passthrough_rows, attrs={**run_result.dataframe.attrs, **dataframe.attrs}
        )
        return DetectionJudgeResult(dataframe=combined, failed_records=run_result.failed_records)
