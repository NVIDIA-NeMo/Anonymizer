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
    COL_REPLACEMENT_MAP,
    COL_TYPE_FIDELITY_INVALID_REPLACEMENTS,
    COL_TYPE_FIDELITY_JUDGE,
    COL_TYPE_FIDELITY_VALID,
    ENTITY_LABEL_EXAMPLES,
)
from anonymizer.engine.ndd.adapter import FailedRecord, NddAdapter
from anonymizer.engine.ndd.model_loader import resolve_model_alias
from anonymizer.engine.prompt_utils import substitute_placeholders
from anonymizer.engine.row_partitioning import merge_and_reorder, split_rows
from anonymizer.engine.schemas import EntityReplacementMapSchema

logger = logging.getLogger("anonymizer.replace.type_fidelity_judge")

_REPLACEMENTS_FOR_JUDGE_COL = "_replacements_for_type_fidelity_judge"
_EXAMPLES_FOR_JUDGE_COL = "_label_examples_for_type_fidelity_judge"


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class InvalidReplacement(BaseModel):
    original: str = Field(description="Original value taken verbatim from the replacement map.")
    label: str = Field(description="Entity label assigned to the original value.")
    synthetic: str = Field(description="The synthetic value that fails type fidelity.")
    reasoning: str = Field(description="One short sentence naming the specific class or format violation.")


class TypeFidelityJudgmentSchema(BaseModel):
    all_valid: bool = Field(
        description=(
            "True only if every synthetic replacement preserves the entity class AND the "
            "format/type expected for that class. False if even one replacement violates either."
        )
    )
    invalid_replacements: list[InvalidReplacement] = Field(
        default_factory=list,
        description="Every replacement that fails type fidelity. Empty when all_valid is True.",
    )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TypeFidelityJudgeResult:
    dataframe: pd.DataFrame
    failed_records: list[FailedRecord]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def _judge_prompt() -> str:
    prompt = """You are an expert judge evaluating TYPE FIDELITY of synthetic PII replacements.

<scope>
TYPE FIDELITY answers a single, narrow question for each (original, label, synthetic) triple:
  "Does the synthetic value still belong to the same entity class AND match the format/type \
expected for that class?"

You are NOT judging:
  - whether the replacement preserves semantic attributes (gender of a name, age bucket, \
locale, ethnicity). That is a DIFFERENT metric.
  - whether the replacement is internally consistent with other entities (city/state, \
name/email, DOB/age). That is a DIFFERENT metric.
  - whether the replacement is "creative", "diverse", or "different enough" from the original.
  - whether the original was correctly detected. Assume the detection was correct.

Only judge the class membership AND the format/type of the synthetic value, in isolation.
</scope>

<original_anchoring>
PRIMARY REFERENCE: the ORIGINAL value sets the expected shape and granularity for the synthetic.
Do NOT impose stricter standards than the original itself satisfies.

The synthetic only needs to be SHAPE-COMPATIBLE with the original — same granularity \
(or finer) and same character class — not match an external "canonical" template.

If a stricter rule in <format_type_rules> seems to conflict with what the original itself \
already looks like, defer to the original's shape.
</original_anchoring>

<replacements>
{%- for entry in <<REPLACEMENTS_COLUMN>> %}
- original="{{ entry.original }}" | label={{ entry.label }} | synthetic="{{ entry.synthetic }}"
{%- endfor %}
</replacements>

<reference_label_examples>
{{ <<EXAMPLES_COLUMN>> }}
</reference_label_examples>

<task>
For each (original, label, synthetic) triple above, decide whether the synthetic value:
  (a) belongs to the SAME entity class as the label, AND
  (b) matches the format/type expected for that class.

Return structured JSON:
  - Set `all_valid` to true ONLY if every triple satisfies both (a) and (b).
  - Otherwise set `all_valid` to false and list every failing triple in `invalid_replacements`, \
each with the verbatim `original`, `label`, `synthetic`, and a short `reasoning`.
</task>

<class_membership_rules>
A synthetic value FAILS class membership when:
  - The synthetic is not a plausible instance of the label, regardless of format. \
Examples:
      * label=first_name, synthetic="The blue sky" -> not a first name.
      * label=company_name, synthetic="Tuesday" -> not a company.
      * label=occupation, synthetic="42" -> not an occupation.
  - The synthetic resolves to a different entity type that happens to share \
surface form. Examples:
      * label=last_name, synthetic="Boston" -> a city, not a last name.
      * label=city, synthetic="Smith" -> a surname, not a city.

A synthetic value PASSES class membership when:
  - A reasonable reader would recognize the synthetic as an instance of the labeled class.
  - The reference_label_examples block illustrates the expected class for each label.
  - The synthetic belongs to a SIBLING category within the same broad domain as the label \
    (a more specific, more general, or peer member of the same parent concept). Treat the \
    label as a bucket, not a precise taxonomy node — any member of the same bucket passes.
</class_membership_rules>

<format_type_rules>
A synthetic value FAILS format/type when its surface form does not match what the label \
demands. Apply the following structural checks where applicable:

  - label=email -> must contain exactly one '@' with non-empty local part and a domain that \
    contains at least one '.'. "alice example com" or "alice@@x" FAIL.
  - label=phone_number, fax_number -> must be predominantly digits with optional '+', spaces, \
    dashes, parentheses, or dots. "call-me-maybe" FAILS.
  - label=ssn -> must look like a US SSN (e.g. ddd-dd-dddd or 9 contiguous digits). \
    "abc-de-fghi" FAILS.
  - label=date_of_birth, date -> must be a recognizable date (numeric or month-name). \
    "yesterday" or "soonish" FAIL.
  - label=date_time, time -> must look like a date+time or time-of-day expression. \
    "after lunch" FAILS.
  - label=age -> must be a non-negative integer (digits) or simple numeric string. \
    "forty" or "old" FAIL.
  - label=ipv4 -> four dot-separated 0-255 integers. "192.cat.1.1" FAILS.
  - label=ipv6 -> colon-separated hex groups. "not-an-ipv6" FAILS.
  - label=mac_address -> six hex pairs separated by ':' or '-'. "12-34" FAILS (too short).
  - label=credit_debit_card -> 13-19 digits, with optional separators. "1234" FAILS.
  - label=cvv -> 3-4 contiguous digits. "ABCD" FAILS.
  - label=pin -> short numeric string (typically 4-8 digits). "letmein" FAILS.
  - label=url -> must start with a scheme (http://, https://, ftp://) or be a clear hostname \
    with a TLD. "click here" FAILS.
  - label=postcode -> short alphanumeric postal code consistent with some country format. \
    "the post office" FAILS.
  - label=coordinate -> latitude,longitude pair of numeric values. "north of here" FAILS.
  - label=monetary_amount -> numeric amount with optional currency symbol/code. \
    "expensive" FAILS.
  - label=ssn, tax_id, employee_id, customer_id, account_number, national_id, \
    bank_routing_number, license_plate, vehicle_identifier, certificate_license_number, \
    medical_record_number, health_plan_beneficiary_number, swift_bic, api_key, \
    device_identifier, biometric_identifier, unique_id, http_cookie -> must look like a \
    structured identifier (alphanumeric, optional separators), NOT a free-text phrase.
  - label=first_name, last_name, user_name -> single token without spaces (user_name may \
    include digits/underscores/dots). "Mr. John Smith Jr." FAILS as a first_name.
  - label=street_address -> contains a number plus a street word OR a recognizable \
    address-like sequence. "somewhere downtown" FAILS.
  - label=country, state, county, city, place_name, landmark, university, court_name, \
    prison_detention_facility, organization_name, company_name, nationality, language, \
    religious_belief, political_view, race_ethnicity, gender, sexuality, blood_type, \
    employment_status, education_level, degree, field_of_study, occupation, color -> must \
    be a proper noun or short canonical phrase from that domain, not a sentence or generic \
    placeholder. "[REDACTED]" or "some place" FAIL.

When a label is not listed above, apply the underlying principle: the synthetic must read \
as a valid instance of that label's type to a reasonable reader.
</format_type_rules>

<edge_cases>
  - Case differences alone are fine ("john" vs "John" -> valid).
  - Reasonable transliteration is fine ("Yaroslav" for a name label -> valid).
  - Placeholders like "[REDACTED]", "***", "N/A", "TODO", empty string -> ALWAYS invalid \
    (no class membership, no real format).
  - synthetic == original verbatim -> still judge type fidelity on its merits (other metrics \
    handle anonymization strength). Do NOT mark invalid solely because synthetic equals \
    original.
  - If the label is unknown to you, fall back to the reference_label_examples and apply \
    the underlying principle. Do not invent new failure modes.
</edge_cases>

<output_rules>
  - Reasoning MUST be ONE short sentence per invalid replacement, naming the specific \
    failure mode ("class membership: not a person name", "format: not a valid email shape", \
    etc.).
  - Use the EXACT verbatim `original`, `label`, and `synthetic` strings from the input list \
    when populating `invalid_replacements`.
  - Do NOT include valid replacements in `invalid_replacements`.
  - Do NOT introduce replacements that were not in the input list.
  - If `replacements` is empty, return `all_valid: true` and an empty list.
</output_rules>

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
            "<<EXAMPLES_COLUMN>>": _EXAMPLES_FOR_JUDGE_COL,
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_EXAMPLE_LOOKUP: dict[str, str] = {
    label: f"(e.g. {', '.join(examples)})" for label, examples in ENTITY_LABEL_EXAMPLES.items()
}


def _replacements_for_judge(raw_map: object) -> list[dict[str, str]]:
    """Flatten COL_REPLACEMENT_MAP into Jinja-friendly dicts.

    Accepts the dict-wrapper form ``{"replacements": [...]}``, an
    ``EntityReplacementMapSchema`` instance, or a JSON string. Returns an
    empty list for anything malformed.
    """
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


def _label_examples_for_judge(replacements: list[dict[str, str]]) -> str:
    """Build a JSON map of {label: example_hint} for labels present in this row."""
    labels = {entry["label"] for entry in replacements if entry.get("label")}
    if not labels:
        return "{}"
    examples = {label: _EXAMPLE_LOOKUP.get(label, "(no canonical example available)") for label in sorted(labels)}
    return json.dumps(examples, ensure_ascii=True)


def _flatten_judgment(raw: object) -> tuple[bool | None, list[dict[str, str]]]:
    """Normalize an LLM judge output into (all_valid, invalid_replacements).

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
        parsed = TypeFidelityJudgmentSchema.model_validate(raw)
    except Exception:
        return None, []
    return parsed.all_valid, [entry.model_dump() for entry in parsed.invalid_replacements]


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class TypeFidelityJudgeWorkflow:
    """LLM-as-judge evaluator that flags replacements failing type fidelity.

    Runs after Substitute generates the replacement map. Output columns:
    ``COL_TYPE_FIDELITY_VALID`` (bool|None) and
    ``COL_TYPE_FIDELITY_INVALID_REPLACEMENTS`` (list of
    {original, label, synthetic, reasoning}).
    """

    def __init__(self, adapter: NddAdapter) -> None:
        self._adapter = adapter

    def evaluate(
        self,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: ReplaceModelSelection,
        preview_num_records: int | None = None,
    ) -> TypeFidelityJudgeResult:
        judge_alias = resolve_model_alias("type_fidelity_judge", selected_models)

        working_df = dataframe.copy()
        replacements_per_row = working_df[COL_REPLACEMENT_MAP].apply(_replacements_for_judge)
        working_df[_REPLACEMENTS_FOR_JUDGE_COL] = replacements_per_row
        working_df[_EXAMPLES_FOR_JUDGE_COL] = replacements_per_row.apply(_label_examples_for_judge)

        # Rows with no replacements trivially pass — skip the LLM call.
        with_replacements, passthrough_rows = split_rows(working_df, column=_REPLACEMENTS_FOR_JUDGE_COL, predicate=bool)
        passthrough_rows[COL_TYPE_FIDELITY_JUDGE] = [
            {"all_valid": True, "invalid_replacements": []} for _ in range(len(passthrough_rows))
        ]
        passthrough_rows[COL_TYPE_FIDELITY_VALID] = True
        passthrough_rows[COL_TYPE_FIDELITY_INVALID_REPLACEMENTS] = [[] for _ in range(len(passthrough_rows))]

        if with_replacements.empty:
            combined = merge_and_reorder(passthrough_rows, attrs=dataframe.attrs)
            return TypeFidelityJudgeResult(dataframe=combined, failed_records=[])

        effective_preview_num_records = (
            min(preview_num_records, len(with_replacements)) if preview_num_records is not None else None
        )
        run_result = self._adapter.run_workflow(
            with_replacements,
            model_configs=model_configs,
            columns=[
                LLMStructuredColumnConfig(
                    name=COL_TYPE_FIDELITY_JUDGE,
                    prompt=_judge_prompt(),
                    model_alias=judge_alias,
                    output_format=TypeFidelityJudgmentSchema,
                )
            ],
            workflow_name="replace-type-fidelity-judge",
            preview_num_records=effective_preview_num_records,
        )

        judged_df = run_result.dataframe.copy()
        flattened = judged_df[COL_TYPE_FIDELITY_JUDGE].apply(_flatten_judgment)
        judged_df[COL_TYPE_FIDELITY_VALID] = flattened.apply(lambda pair: pair[0])
        judged_df[COL_TYPE_FIDELITY_INVALID_REPLACEMENTS] = flattened.apply(lambda pair: pair[1])

        combined = merge_and_reorder(
            judged_df, passthrough_rows, attrs={**run_result.dataframe.attrs, **dataframe.attrs}
        )
        return TypeFidelityJudgeResult(dataframe=combined, failed_records=run_result.failed_records)
