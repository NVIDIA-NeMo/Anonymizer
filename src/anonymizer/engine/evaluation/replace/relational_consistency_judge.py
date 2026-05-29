# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from typing import ClassVar

import pandas as pd
from pydantic import BaseModel, Field

from anonymizer.engine.constants import (
    COL_RELATIONAL_CONSISTENCY_INVALID_RELATIONS,
    COL_RELATIONAL_CONSISTENCY_JUDGE,
    COL_RELATIONAL_CONSISTENCY_VALID,
    COL_REPLACED_TEXT,
    COL_REPLACEMENT_MAP,
    _jinja,
)
from anonymizer.engine.evaluation.judge_base import _BaseJudgeWorkflow
from anonymizer.engine.prompt_utils import substitute_placeholders
from anonymizer.engine.schemas import EntityReplacementMapSchema

logger = logging.getLogger("anonymizer.evaluation.replace.relational_consistency_judge")

_REPLACEMENTS_FOR_JUDGE_COL = "_replacements_for_relational_consistency_judge"


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class RelationCheck(BaseModel):
    description: str = Field(
        description=("Short label of the relation being verified, e.g. 'city <-> state' or 'date_of_birth <-> age'.")
    )
    entities: list[str] = Field(
        default_factory=list,
        description=(
            "The entities involved in this relation, each rendered as a single string of the "
            "form '<original> (<label>) -> <synthetic>'. Two or more entries per relation."
        ),
    )
    passes: bool = Field(
        description=("True if the synthetic replacements preserve the relation; false if they violate it.")
    )
    reasoning: str = Field(
        description=(
            "One short sentence. For passes=true, briefly state why the relation holds. "
            "For passes=false, name the specific inconsistency."
        )
    )


class RelationalConsistencyJudgmentSchema(BaseModel):
    all_consistent: bool = Field(
        description=("True only if every relation in `relations` has passes=true. False if even one fails.")
    )
    relations: list[RelationCheck] = Field(
        default_factory=list,
        description=("Every relation actually checked in this record. Empty when no checkable relations exist."),
    )


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def _judge_prompt() -> str:
    prompt = """You are an expert judge evaluating RELATIONAL CONSISTENCY of synthetic PII replacements.

<scope>
RELATIONAL CONSISTENCY answers a single question for a record:
  "Do the synthetic entities preserve the same relational coherence with each other that the \
original entities had?"

You are NOT judging:
  - whether each synthetic value matches the type/format of its label (that is a DIFFERENT metric).
  - whether the original detection was correct (assume it was).
  - whether the replacement is "diverse" or "different enough" from the original.
  - whether semantic attributes like gender, age bucket, locale are preserved (DIFFERENT metric).

Only judge cross-entity coherence within this single record.
</scope>

<replaced_text>
<<COL_REPLACED_TEXT>>
</replaced_text>

<replacements>
{%- for entry in <<REPLACEMENTS_COLUMN>> %}
- original="{{ entry.original }}" | label={{ entry.label }} | synthetic="{{ entry.synthetic }}"
{%- endfor %}
</replacements>

<task>
1. Identify each pair or small group of entities in this record that has an EXPECTED \
relationship (see <relations_to_inspect>).
2. For each relationship you can verify, decide whether the synthetic replacements preserve \
that relationship.
3. Emit one entry in `relations` per relationship you actually checked. Each entry has: \
a short `description`; `entities` as a list of STRINGS (each formatted exactly as \
`"<original> (<label>) -> <synthetic>"` — verbatim values, single string per participant); \
`passes` (true/false); and a one-sentence `reasoning`.
4. Set `all_consistent=true` ONLY if every entry in `relations` has `passes=true`. Otherwise \
set it to false.
5. If the record has NO checkable relations (e.g. one entity, or entities with no expected \
relation), return `all_consistent: true` and `relations: []`.
</task>

<relations_to_inspect>
GEOGRAPHIC:
  - city <-> state: the city should be located in the state.
  - city <-> postcode: the postcode should be valid for the city.
  - state <-> postcode: the postcode prefix should match the state.
  - country <-> any of: state, city, postcode, phone_number country code, nationality.
  - coordinate <-> any geographic entity: roughly consistent.

  IMPORTANT — same-referent precondition for GEOGRAPHIC pairs: before flagging a \
city/state/postcode/country pair as inconsistent, check the `replaced_text` to confirm \
both entities describe the SAME person, place, or location. When the text makes clear \
that one geographic entity describes one referent (e.g. the data subject's residence) \
and another describes a different referent (e.g. an adult child who lives elsewhere, \
an employer in another country, a relative abroad), they are INDEPENDENT narrative \
threads — SKIP the relation rather than flag it. Apply this precondition only when the \
two referents are clearly distinguished in the prose (separate sentences, separate \
clauses, separate subjects). When the text leaves the referent ambiguous, fall back to \
the standard pair check.

TEMPORAL:
  - date_of_birth <-> age: age should equal current_year - dob_year within +/-1. \
    REQUIRES an entity whose label is literally `date_of_birth`. A generic `date` is NOT a \
    substitute for `date_of_birth`, even when its value looks like a year.
  - date_of_birth <-> date references to the person's life events (graduation, hire): \
    should be plausibly ordered (DOB before the event). REQUIRES an entity whose label is \
    literally `date_of_birth`.

IDENTITY:
  - first_name and/or last_name <-> email local part: the local part should contain a \
    recognizable form of the synthetic name (initials, fuller form, or a clear derivative).
  - first_name and/or last_name <-> user_name: same idea — user_name should reference the name.
  - name (first/last) <-> pronouns in the <replaced_text>: pronouns (he/she/they/his/her/their) \
    should match the implied gender of the synthetic name. Be lenient with gender-neutral or \
    ambiguous names — only flag clear contradictions.

ORGANIZATIONAL:
  - company_name <-> url / email domain / organization_name: the domain or organization name \
    should reference the company.

ROLE / EMPLOYMENT COHERENCE:
  - occupation <-> company_name / organization_name: the role should be PLAUSIBLE at that \
    organization (e.g. a "registered nurse" at a hospital is plausible; the same role at an \
    oil-rig company is not). Be lenient — only flag CLEAR mismatches, not stylistic ones.
  - occupation <-> education_level / degree / field_of_study: the role and credentials \
    should be COMPATIBLE (e.g. a "surgeon" with a high-school education is a clear \
    contradiction; a "surgeon" with a medical degree is consistent). Only flag clear \
    contradictions.

DEMOGRAPHIC COHERENCE:
  - age <-> occupation: only flag CLEAR impossibilities (e.g. an 8-year-old neurosurgeon, a \
    5-year-old CEO). Do NOT flag stylistic or unusual but possible combinations.
  - age <-> education_level / degree: only flag CLEAR contradictions (e.g. a 10-year-old \
    with a Ph.D., a 6-year-old with a bachelor's). Be lenient with mild gaps (e.g. someone \
    who finished a degree a bit younger or older than typical is fine).
  - date_of_birth <-> any life-event date (graduation, hire, marriage, etc.) present in the \
    replacements: the life event should occur AFTER the DOB, with a plausible gap \
    (no graduations a year after birth). REQUIRES an entity whose label is literally \
    `date_of_birth`.

COMMUNICATION:
  - phone_number country code <-> country: e.g. "+1" pairs with USA/Canada, "+44" with UK.
  - fax_number country code <-> country: same idea.

If a category is not present in the replacements, SKIP it. DO NOT fabricate relations that \
the data does not contain.
</relations_to_inspect>

<rules>
  - Only check a relation when BOTH (or all) participating entities appear in the \
    `replacements` list, OR when one entity in the list has a clear textual reference in \
    `replaced_text` (e.g. a pronoun pointing back at a name).
  - Match relations by the LITERAL `label` field of each entity. Do NOT infer a label from \
    the value's surface form. A 4-digit year is not automatically a `date_of_birth`; a \
    capitalized noun is not automatically a `city`; etc. If the listed label does not appear \
    in a relation's signature, that relation does not apply to that entity.
  - For each relation you check, ALWAYS include it in `relations` — even when it passes. \
    The total length of `relations` is used as the denominator for a success rate, so omitting \
    passing relations would distort the score.
  - The `entities` field for each relation must be a LIST OF STRINGS, one per participant, \
    each formatted exactly as `"<original> (<label>) -> <synthetic>"` using the verbatim \
    values from the replacement map. For pronoun checks, include only the name entity. \
    Do NOT emit nested objects in `entities` — strings only.
  - DO NOT include a relation where you cannot confidently judge pass/fail — skip it instead \
    of guessing.
  - DO NOT list the same relation twice. If the same pair of labels appears multiple times \
    in the data (e.g. two cities), treat each instance as its own relation.
  - `reasoning` must be ONE short sentence per relation. For failures, name the specific \
    inconsistency.
</rules>

<edge_cases>
  - Gender-neutral or ambiguous synthetic names (e.g. "Alex", "Jordan", "Taylor") + ANY \
    pronoun -> mark the pronoun relation as passes=true. Only flag a clear mismatch.
  - Compound name in one entity ("Sarah Chen" labeled first_name) -> treat its parts \
    individually when matching against email/user_name.
  - If the email or user_name uses initials only ("sc@x.com" for "Sarah Chen") -> passes=true.
  - A generic `date` entity (graduation year, hire year, "returned home in 2012", etc.) is \
    NOT a `date_of_birth`. Do NOT pair such a `date` with `age` and compute current_year - \
    date. The `date_of_birth <-> age` check only fires when an entity labeled exactly \
    `date_of_birth` is present. If only a generic `date` and an `age` are present and the \
    replaced_text does not unambiguously identify that date as the person's birth year, \
    SKIP the temporal relation entirely.
  - If a relation requires external knowledge you are unsure about (e.g. obscure city in a \
    small country) -> SKIP rather than guess.
</edge_cases>

<output_format>
Return ONLY the JSON object that matches the required schema. Do NOT wrap your output in \
``` or ```json markdown fences. Do NOT include any commentary, reasoning, preamble, or text \
outside the JSON object. Your entire response must be a single valid JSON object.
</output_format>
"""
    return substitute_placeholders(
        prompt,
        {
            "<<COL_REPLACED_TEXT>>": _jinja(COL_REPLACED_TEXT),
            "<<REPLACEMENTS_COLUMN>>": _REPLACEMENTS_FOR_JUDGE_COL,
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


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class RelationalConsistencyJudgeWorkflow(_BaseJudgeWorkflow):
    """LLM-as-judge evaluator that checks cross-entity coherence within a record.

    Runs after Substitute generates the replacement map. Output columns:
    ``COL_RELATIONAL_CONSISTENCY_VALID`` (bool|None),
    ``COL_RELATIONAL_CONSISTENCY_INVALID_RELATIONS`` (list of failing relations),
    and the raw judge output ``COL_RELATIONAL_CONSISTENCY_JUDGE`` (kept so
    display can derive the success-rate denominator from the full relations list).
    """

    RAW_COL: ClassVar[str] = COL_RELATIONAL_CONSISTENCY_JUDGE
    VALID_COL: ClassVar[str] = COL_RELATIONAL_CONSISTENCY_VALID
    INVALID_COL: ClassVar[str] = COL_RELATIONAL_CONSISTENCY_INVALID_RELATIONS
    SCHEMA: ClassVar[type[BaseModel]] = RelationalConsistencyJudgmentSchema
    VERDICT_FIELD: ClassVar[str] = "all_consistent"
    DEFAULT_PAYLOAD: ClassVar[dict] = {"all_consistent": True, "relations": []}
    MODEL_ROLE: ClassVar[str] = "replace_relational_consistency_judge"
    WORKFLOW_NAME: ClassVar[str] = "replace-relational-consistency-judge"

    def prepare(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        working_df = dataframe.copy()
        working_df[_REPLACEMENTS_FOR_JUDGE_COL] = working_df[COL_REPLACEMENT_MAP].apply(_replacements_for_judge)
        return working_df

    def _passthrough_mask(self, dataframe: pd.DataFrame) -> pd.Series:
        # Fewer than 2 replacements => no checkable relations.
        return dataframe[_REPLACEMENTS_FOR_JUDGE_COL].apply(lambda items: items is None or len(items) < 2)

    @classmethod
    def _build_prompt(cls) -> str:
        return _judge_prompt()

    @classmethod
    def _extract_invalid(cls, parsed: BaseModel) -> list[dict[str, object]]:
        return [r.model_dump() for r in parsed.relations if not r.passes]
