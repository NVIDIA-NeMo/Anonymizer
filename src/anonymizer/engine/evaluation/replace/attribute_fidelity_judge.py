# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from data_designer.config.column_configs import LLMStructuredColumnConfig
from data_designer.config.models import ModelConfig
from pydantic import BaseModel, Field

from anonymizer.config.models import EvaluateModelSelection
from anonymizer.engine.constants import (
    COL_ATTRIBUTE_FIDELITY_INVALID_ENTITIES,
    COL_ATTRIBUTE_FIDELITY_JUDGE,
    COL_ATTRIBUTE_FIDELITY_VALID,
    COL_REPLACEMENT_MAP,
)
from anonymizer.engine.ndd.adapter import FailedRecord, NddAdapter
from anonymizer.engine.ndd.model_loader import resolve_model_alias
from anonymizer.engine.prompt_utils import substitute_placeholders
from anonymizer.engine.row_partitioning import merge_and_reorder, split_rows
from anonymizer.engine.schemas import EntityReplacementMapSchema

logger = logging.getLogger("anonymizer.evaluation.replace.attribute_fidelity_judge")

_REPLACEMENTS_FOR_JUDGE_COL = "_replacements_for_attribute_fidelity_judge"


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class AttributeCheck(BaseModel):
    original: str = Field(description="Original value taken verbatim from the replacement map.")
    label: str = Field(description="Entity label assigned to the original value.")
    synthetic: str = Field(description="Synthetic value that replaced the original.")
    attributes_checked: list[str] = Field(
        default_factory=list,
        description=(
            "Salient within-entity attributes inspected for this triple (e.g. ['gender', 'age_bucket'])."
        ),
    )
    passes: bool = Field(
        description=(
            "True when EVERY attribute in `attributes_checked` is preserved by the synthetic; "
            "false when any clearly changes."
        )
    )
    reasoning: str = Field(
        description=("One short sentence naming the attribute(s) and whether they were preserved or changed.")
    )


class AttributeFidelityJudgmentSchema(BaseModel):
    all_valid: bool = Field(
        description=("True only if every entry in `entities` has passes=true. False if even one fails.")
    )
    entities: list[AttributeCheck] = Field(
        default_factory=list,
        description=(
            "One entry per replacement triple that has AT LEAST ONE salient attribute. "
            "Triples with no salient attributes (opaque identifiers) are omitted."
        ),
    )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttributeFidelityJudgeResult:
    dataframe: pd.DataFrame
    failed_records: list[FailedRecord]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def _judge_prompt() -> str:
    prompt = """You are an expert judge evaluating ATTRIBUTE FIDELITY of synthetic PII replacements.

<scope>
ATTRIBUTE FIDELITY answers ONE simple question per (original, label, synthetic) triple:
  "Is the synthetic CLOSE ENOUGH to the original on its salient within-entity attributes?"

A GOOD replacement keeps the obvious semantic properties of the original — the things a \
reader would intuit just from looking at the entity. A BAD replacement clearly flips one of \
those properties.

CANONICAL EXAMPLES (these set the bar — calibrate against these):
  - first_name "Valentina" -> "Natalia"  ->  PASS (gender preserved: both feminine).
  - first_name "Valentina" -> "Mike"     ->  FAIL (gender clearly flipped).
  - age        "40"        -> "42"       ->  PASS (same adult bucket).
  - age        "40"        -> "12"       ->  FAIL (adult -> child).

You are NOT judging:
  - whether the synthetic is the right TYPE / format / class (that is a DIFFERENT metric).
  - whether the entities AGREE with each other across the record, e.g. city <-> state, \
    DOB <-> age, name <-> email (DIFFERENT metric).
  - whether the original detection was correct (assume it was).
  - whether the replacement is "diverse" or "different enough" from the original.

This metric is NORMATIVE: pass = a good attribute-preserving replacement; fail = a clearly \
attribute-violating replacement. Be charitable on borderline cases — only flag CLEAR \
attribute flips.
</scope>

<replacements>
{%- for entry in <<REPLACEMENTS_COLUMN>> %}
- original="{{ entry.original }}" | label={{ entry.label }} | synthetic="{{ entry.synthetic }}"
{%- endfor %}
</replacements>

<task>
For each (original, label, synthetic) triple:
1. Decide which salient attributes apply (see <salient_attributes_by_label>).
2. For each applicable attribute the ORIGINAL clearly carries, decide whether the synthetic \
preserves it.
3. Emit one entry in `entities` per triple where AT LEAST ONE salient attribute applies AND \
the original clearly carries it. The entry must include the verbatim entity, the actually-\
inspected `attributes_checked`, a `passes` boolean, and a one-sentence `reasoning`.
4. SKIP triples that have no salient attributes (opaque identifiers, hashes, codes). DO NOT \
emit them.
5. SKIP triples where the original is too ambiguous to anchor any attribute (e.g. gender-\
neutral name with no other signal). DO NOT emit them.
6. Set `all_valid=true` ONLY if every emitted entry has `passes=true`. Otherwise set false. \
If no triple yields a checkable attribute, return `all_valid: true` and `entities: []`.
</task>

<salient_attributes_by_label>
This metric checks ONLY TWO attributes. Do not check anything else.

1. GENDER OF NAME — applies to labels: first_name, last_name, user_name.
   - Check only when the original name CLEARLY implies a gender (e.g. "Valentina", \
     "Michael"). If the original is gender-neutral or ambiguous (e.g. "Alex", "Taylor", \
     "J.", a surname-only token whose gender you can't reliably tell) -> SKIP this triple.
   - The check: does the synthetic name carry the SAME implied gender as the original?

2. AGE BUCKET — applies to labels: age, date_of_birth.
   - Buckets: child (0-12), teen (13-19), young adult (20-29), adult (30-44), \
     middle-aged (45-64), senior (65+). It's fine if the years are +/- 1 year of the original.
   - For `age`, the bucket comes from the numeric value directly. For `date_of_birth`, \
     compute age = <<CURRENT_YEAR>> - dob_year, then map to a bucket.
   - The check: does the synthetic land in the SAME bucket as the original (or an \
     ADJACENT bucket — adjacent counts as preserved; only flag clear bucket flips like \
     adult -> child)?

ALL OTHER LABELS — SKIP. Do not emit entries for any label not listed above. This includes \
cities, countries, occupations, education, organizations, phone numbers, dates that are not \
date_of_birth, protected categorical entities, and every opaque identifier. Their attributes \
are either checked by other metrics or are too unreliable to judge here.
</salient_attributes_by_label>

<rules>
  - Use the LITERAL `label` field to decide which attributes apply. Do not infer attributes \
    from the value's surface form alone.
  - Only check an attribute when the ORIGINAL clearly carries it. If the original is \
    ambiguous on an attribute (gender-neutral name, undated entity, generic city), do NOT \
    check that attribute.
  - For each entity you emit, list every checked attribute in `attributes_checked`. If \
    nothing was checked, DO NOT emit the entry.
  - `passes` = true when EVERY checked attribute is preserved. `passes` = false when any \
    checked attribute clearly changes. Be CHARITABLE on borderline cases — prefer passes=true.
  - `reasoning` must be ONE short sentence per entry, naming the attribute(s) and stating \
    preserved-or-changed.
  - DO NOT introduce entities that were not in the replacement list.
  - `entities` may be empty when the record has no checkable attributes; that is a valid \
    `all_valid=true` outcome.
</rules>

<output_format>
Return ONLY the JSON object that matches the required schema. Do NOT wrap your output in \
``` or ```json markdown fences. Do NOT include any commentary, reasoning, preamble, or text \
outside the JSON object. Your entire response must be a single valid JSON object.
</output_format>
"""
    return substitute_placeholders(
        prompt,
        {
            "<<REPLACEMENTS_COLUMN>>": _REPLACEMENTS_FOR_JUDGE_COL,
            "<<CURRENT_YEAR>>": str(datetime.now().year),
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _replacements_for_judge(raw_map: object) -> list[dict[str, str]]:
    """Flatten COL_REPLACEMENT_MAP into Jinja-friendly dicts."""
    if raw_map is None:
        return []
    if hasattr(raw_map, "model_dump"):
        raw_map = raw_map.model_dump(mode="python")
    if isinstance(raw_map, str):
        try:
            raw_map = json.loads(raw_map)
        except (json.JSONDecodeError, ValueError):
            return []
    if not isinstance(raw_map, dict):
        return []
    try:
        parsed = EntityReplacementMapSchema.model_validate(raw_map)
    except Exception:
        return []
    return [{"original": r.original, "label": r.label, "synthetic": r.synthetic} for r in parsed.replacements]


def _flatten_judgment(raw: object) -> tuple[bool | None, list[dict[str, object]]]:
    """Normalize an LLM judge output into (all_valid, invalid_entities).

    Returns ``(None, [])`` for any malformed or missing payload so downstream
    display renders "judge unavailable" rather than fabricating a verdict.
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
        parsed = AttributeFidelityJudgmentSchema.model_validate(raw)
    except Exception:
        return None, []
    invalid = [e.model_dump() for e in parsed.entities if not e.passes]
    return parsed.all_valid, invalid


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class AttributeFidelityJudgeWorkflow:
    """LLM-as-judge evaluator that checks per-entity attribute preservation.

    Runs after Substitute generates the replacement map. Output columns:
    ``COL_ATTRIBUTE_FIDELITY_VALID`` (bool|None),
    ``COL_ATTRIBUTE_FIDELITY_INVALID_ENTITIES`` (list of failing per-entity checks),
    and the raw judge output ``COL_ATTRIBUTE_FIDELITY_JUDGE`` (kept so display
    can derive the success-rate denominator from the full entities list).
    """

    def __init__(self, adapter: NddAdapter) -> None:
        self._adapter = adapter

    def prepare(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        working_df = dataframe.copy()
        working_df[_REPLACEMENTS_FOR_JUDGE_COL] = working_df[COL_REPLACEMENT_MAP].apply(_replacements_for_judge)
        return working_df

    def column_config(self, selected_models: EvaluateModelSelection) -> LLMStructuredColumnConfig:
        return LLMStructuredColumnConfig(
            name=COL_ATTRIBUTE_FIDELITY_JUDGE,
            prompt=_judge_prompt(),
            model_alias=resolve_model_alias("replace_attribute_fidelity_judge", selected_models),
            output_format=AttributeFidelityJudgmentSchema,
        )

    def postprocess(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        out = dataframe.copy()
        flattened = (
            out[COL_ATTRIBUTE_FIDELITY_JUDGE].apply(_flatten_judgment)
            if COL_ATTRIBUTE_FIDELITY_JUDGE in out.columns
            else None
        )
        passthrough_mask = out[_REPLACEMENTS_FOR_JUDGE_COL].apply(lambda items: items is None or len(items) == 0)

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
        out[COL_ATTRIBUTE_FIDELITY_VALID] = valid
        out[COL_ATTRIBUTE_FIDELITY_INVALID_ENTITIES] = invalid
        if COL_ATTRIBUTE_FIDELITY_JUDGE in out.columns:
            out.loc[passthrough_mask, COL_ATTRIBUTE_FIDELITY_JUDGE] = [{"all_valid": True, "entities": []}] * int(
                passthrough_mask.sum()
            )
        return out

    def evaluate(
        self,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: EvaluateModelSelection,
        preview_num_records: int | None = None,
    ) -> AttributeFidelityJudgeResult:
        working_df = self.prepare(dataframe)

        with_replacements, passthrough_rows = split_rows(working_df, column=_REPLACEMENTS_FOR_JUDGE_COL, predicate=bool)
        passthrough_rows[COL_ATTRIBUTE_FIDELITY_JUDGE] = [
            {"all_valid": True, "entities": []} for _ in range(len(passthrough_rows))
        ]
        passthrough_rows[COL_ATTRIBUTE_FIDELITY_VALID] = True
        passthrough_rows[COL_ATTRIBUTE_FIDELITY_INVALID_ENTITIES] = [[] for _ in range(len(passthrough_rows))]

        if with_replacements.empty:
            combined = merge_and_reorder(passthrough_rows)
            return AttributeFidelityJudgeResult(dataframe=combined, failed_records=[])

        effective_preview_num_records = (
            min(preview_num_records, len(with_replacements)) if preview_num_records is not None else None
        )
        run_result = self._adapter.run_workflow(
            with_replacements,
            model_configs=model_configs,
            columns=[self.column_config(selected_models)],
            workflow_name="replace-attribute-fidelity-judge",
            preview_num_records=effective_preview_num_records,
        )

        judged_df = run_result.dataframe.copy()
        flattened = judged_df[COL_ATTRIBUTE_FIDELITY_JUDGE].apply(_flatten_judgment)
        judged_df[COL_ATTRIBUTE_FIDELITY_VALID] = flattened.apply(lambda pair: pair[0])
        judged_df[COL_ATTRIBUTE_FIDELITY_INVALID_ENTITIES] = flattened.apply(lambda pair: pair[1])

        combined = merge_and_reorder(judged_df, passthrough_rows)
        return AttributeFidelityJudgeResult(dataframe=combined, failed_records=run_result.failed_records)
