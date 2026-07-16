# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import re
from typing import ClassVar

import pandas as pd
from data_designer.config.column_configs import LLMTextColumnConfig
from data_designer.config.models import ModelConfig
from pydantic import BaseModel, Field

from anonymizer.config.models import EvaluateModelSelection
from anonymizer.engine.constants import (
    COL_ENTITIES_BY_VALUE,
    COL_ENTITY_COVERAGE,
    COL_ENTITY_COVERAGE_CANDIDATE_TOTAL,
    COL_ENTITY_COVERAGE_JUDGE,
    COL_ENTITY_COVERAGE_TOTAL,
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

_FINAL_ENTITIES_FOR_COVERAGE_COL = "_final_entities_for_coverage_judge"
_WORD_RE = re.compile(r"\w+", re.UNICODE)
# Grammatical stopwords only — function words that carry no PII and just absorb
# article/preposition noise in a value (e.g. "the Nawabganj" matches "Nawabganj").
# Deliberately does NOT include generic content descriptors (festival, summit, club,
# conference, …): those turn a named event/org into a quasi-identifier, so ignoring
# them would suppress real leaks (e.g. "Davos Summit" collapsing into "Davos").
_COVERAGE_IGNORE_TOKENS = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "by",
        "for",
        "from",
        "in",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
)


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

    prompt = f"""You are a privacy auditor. Given ORIGINAL text and a list of ANONYMIZER FINAL ENTITIES, \
identify any direct or quasi-identifiers in the original text that are NOT covered by the \
anonymizer final entities.

<task>
Identify every direct or quasi-identifier present in the original text that is NOT covered \
by the anonymizer final entities. These are the "leaked" entities — identifiers that would \
survive anonymization because they are absent from the final entity set.

Report every literal candidate whose in-scope semantic type is supported by the original-text context.
Do not require a value to identify someone on its own; direct and quasi-identifiers are both reportable.

Return structured JSON:
- `leaked_entities`: list every missed identifier with its `value`, `label`, and a short `reasoning`.
- Return an empty list if the anonymizer covered all identifiers.
</task>

<identifier_taxonomy>
These entity types are sensitive and should be flagged when a value of that type is present but \
not covered by the anonymizer final entities: {labels_str}.
Check each type and report missed values, including ones that are easy to overlook — but report \
only a concrete identifying value (an actual name, code, date, or attribute value), not a pronoun \
or a generic reference that merely implies a type.
Quasi-identifiers: combinations of values that together re-identify someone \
(e.g. job title + employer + city appearing together). Time values (specific timestamps, \
times of day, schedules) can act as quasi-identifiers when combined with other attributes \
in the same text — flag them if they appear alongside other identifying information.
</identifier_taxonomy>

{entity_scope_block}

<label_interpretation>
Treat each configured label as a semantic entity category. Labels may use compact, compound, \
or abbreviated names; interpret their intended meaning from the label and the original-text \
context. Return labels exactly as they appear in the entity_type_scope.
</label_interpretation>

<coverage_definition>
An identifier is "covered" (already protected) if:
- Its exact value appears in the anonymizer final entities list, OR
- Its complete tokens are contained within one final entity value, OR
- Its value is composed entirely of complete final entity values.
Partial character matches, similar meanings, or shared context alone do not establish coverage.
</coverage_definition>

<reportability_check>
Before reporting an entity, verify ALL of the following:
1. Its value is a literal, non-empty span in the original text.
2. It is an actual data value, not syntax or metadata such as a field name, heading, \
form instruction, blank placeholder, document title, or category label.
3. Its semantic type matches one of the entity types in scope.
4. It is not already covered by an anonymizer final entity.
5. Report the complete contiguous span that represents one sensitive value. Preserve all \
tokens belonging to that value, including multi-token values, while excluding surrounding \
labels, punctuation, instructions, or boilerplate.
6. Evaluate the value using its original-text context rather than its form or how identifying \
it appears in isolation.

In structured or semi-structured text, distinguish:
- Syntax and metadata: field names, headings, instructions, placeholders, category labels, \
formatting, and other structural text.
- Data: assigned values, cell contents, literals, and user-provided content.
Report only the smallest sensitive data value. Do not report surrounding syntax or metadata \
unless it independently contains a literal sensitive value in scope.
</reportability_check>

<guidance>
Do NOT flag:
- Items that satisfy the coverage_definition.
- Text whose in-scope semantic type is not supported by its original-text context.
- Information that is inferable but not literally present in the text.

Do flag:
- The `value` MUST be a literal substring found in the original text.
- `reasoning` MUST be one sentence explaining why this value is not covered by the anonymizer final entities.
- If `anonymizer_final_entities` is empty, scan the full text for any direct or quasi-identifiers.
- A value that fills the role of a listed sensitive type in context, even when it is
  short, a single token, an unfamiliar or foreign-looking word, or resembles an ordinary
  word or number. Decide by the value's role in the surrounding text, not by its length,
  rarity, or familiarity. (This still excludes pronouns and generic references that only
  imply a type — those are not concrete values.)

{entity_scope_guidance}
{strict_guidance}
</guidance>

{strict_block}

<inputs>
<original_text>
<<COL_TEXT>>
</original_text>

<anonymizer_final_entities>
{{%- if <<ENTITIES_COLUMN>> %}}
{{%- for entity in <<ENTITIES_COLUMN>> %}}
- value="{{{{ entity.value }}}}" | label={{{{ entity.label }}}}
{{%- endfor %}}
{{%- else %}}
(none)
{{%- endif %}}
</anonymizer_final_entities>
</inputs>

<output_format>
Return ONLY the JSON object that matches the required schema. Do NOT wrap your output in \
``` or ```json markdown fences. Do NOT include any commentary, reasoning, preamble, or text \
outside the JSON object. Your entire response must be a single valid JSON object.
</output_format>
"""

    # prompt = f"""<task>
    # You are a privacy auditor. Given ORIGINAL text and a list of ANONYMIZER FINAL ENTITIES, \
    # identify every direct or quasi-identifier in the original text that is NOT covered by the \
    # anonymizer final entities. These are the "leaked" entities.

    # Return structured JSON:
    # - `leaked_entities`: list every missed identifier with its `value`, `label`, and a short `reasoning`.
    # - Return an empty list if the anonymizer covered all identifiers.
    # </task>

    # <identifier_taxonomy>
    # These entity types are sensitive and should be flagged when a value of that type is present but \
    # not covered by the anonymizer final entities: {labels_str}.
    # Quasi-identifiers: combinations of values that together re-identify someone \
    # (e.g. job title + employer + city appearing together). Time values (specific timestamps, \
    # times of day, schedules) can act as quasi-identifiers when combined with other attributes \
    # in the same text — flag them if they appear alongside other identifying information.
    # </identifier_taxonomy>

    # {entity_scope_block}

    # <label_interpretation>
    # Treat each configured label as a semantic entity category. Labels may use compact, compound, \
    # or abbreviated names; interpret their intended meaning from the label and the original-text \
    # context. Return labels exactly as they appear in the entity_type_scope.
    # </label_interpretation>

    # <coverage_definition>
    # An identifier is "covered" (already protected) if:
    # - Its exact value appears in the anonymizer final entities list, OR
    # - Its value is covered by a listed entity value or by a combination of listed entity values.
    # </coverage_definition>

    # {strict_block}

    # <output_format>
    # Return ONLY the JSON object that matches the required schema. Do NOT wrap your output in \
    # ``` or ```json markdown fences. Do NOT include any commentary, reasoning, preamble, or text \
    # outside the JSON object. Your entire response must be a single valid JSON object.
    # </output_format>

    # ---
    # <original_text>
    # <<COL_TEXT>>
    # </original_text>

    # <anonymizer_final_entities>
    # {{%- if <<ENTITIES_COLUMN>> %}}
    # {{%- for entity in <<ENTITIES_COLUMN>> %}}
    # - value="{{{{ entity.value }}}}" | label={{{{ entity.label }}}}
    # {{%- endfor %}}
    # {{%- else %}}
    # (none)
    # {{%- endif %}}
    # </anonymizer_final_entities>
    # """
    return substitute_placeholders(
        prompt,
        {
            "<<COL_TEXT>>": _jinja(COL_TEXT),
            "<<ENTITIES_COLUMN>>": _FINAL_ENTITIES_FOR_COVERAGE_COL,
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _final_entities_for_coverage(parsed: EntitiesByValueSchema) -> list[dict[str, str]]:
    """Flatten EntitiesByValueSchema into one (value, label) row per pair for prompt context."""
    return [{"value": e.value, "label": label} for e in parsed.entities_by_value for label in e.labels]


def _parse_leaked_entities(raw: object) -> list[dict[str, object]] | None:
    """Parse raw LLM output into the leaked entity list.

    Returns the list (possibly empty) on success, or None when the payload is
    malformed or missing so downstream display renders "judge unavailable".
    """
    if raw is None:
        return None
    if isinstance(raw, BaseModel):
        raw = raw.model_dump(mode="python")
    if isinstance(raw, str):
        raw = _parse_json_object(raw)
        if raw is None:
            return None
    if not isinstance(raw, dict):
        return None
    try:
        parsed = EntityCoverageSchema.model_validate(raw)
    except Exception:
        return None
    return [e.model_dump() for e in parsed.leaked_entities]


def _parse_judge_entities(raw: object) -> list[dict[str, object]] | None:
    """Compatibility alias for experiment notebooks parsing raw judge output."""
    return _parse_leaked_entities(raw)


def _parse_json_object(raw: str) -> dict[str, object] | None:
    """Parse a JSON object, tolerating fences or brief surrounding model text."""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", raw):
            try:
                parsed, _ = decoder.raw_decode(raw[match.start() :])
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(parsed, dict):
                return parsed
        return None
    return parsed if isinstance(parsed, dict) else None


def _coverage_token_list(value: object) -> list[str]:
    """Unicode-aware, case-insensitive word tokens (order preserved).

    Uses ``casefold()`` + ``\\w`` so accented and non-Latin scripts tokenize
    correctly (e.g. ``José`` -> ``["josé"]``) instead of being dropped or mangled.
    """
    return _WORD_RE.findall(str(value).casefold())


def _is_concatenation_of_whole_values(leaked_tokens: list[str], final_token_lists: list[list[str]]) -> bool:
    """True when ``leaked_tokens`` segment exactly into a sequence of WHOLE final values.

    This is the composite case: a leak that is the concatenation of adjacent detected
    entities (e.g. ``"Nawabganj - 382210"`` == ``"Nawabganj"`` + ``"382210"``). Each
    segment must equal a full final-entity value, so a leak whose pieces are only
    *partial* tokens of unrelated entities is NOT matched here.
    """

    def consume(start: int) -> bool:
        if start == len(leaked_tokens):
            return True
        for final_tokens in final_token_lists:
            end = start + len(final_tokens)
            if final_tokens and leaked_tokens[start:end] == final_tokens and consume(end):
                return True
        return False

    return consume(0)


def _is_leaked_value_covered(leaked_value: object, final_values: list[str]) -> bool:
    """Return True when a judge-reported leak is already covered by final entities.

    Coverage is decided **per final entity** — never against a pooled bag of tokens
    from *all* final entities — so a leak whose pieces come from unrelated entities is
    not wrongly suppressed (e.g. ``"John Smith"`` is NOT covered by ``"John Doe"`` +
    ``"Jane Smith"``). A leak is covered when either:

    - **subspan** — its (core) tokens are a subset of a *single* final entity's tokens
      (``"Mstr"`` ⊂ ``"Mstr Marzella"``, ``"44"`` ⊂ ``"44 Dunsfold Drive"``), or
    - **composite** — its tokens are a concatenation of *whole* final-entity values
      (``"Nawabganj - 382210"`` == ``"Nawabganj"`` + ``"382210"``).

    Matching is on whole tokens, so a leak only matches a final token it equals
    (``"m"`` is not covered by ``"Mstr Marzella"``: ``"m"`` != ``"mstr"``). Grammatical
    stopwords (see ``_COVERAGE_IGNORE_TOKENS``) are dropped from the leak's core so
    article/preposition noise does not block a subspan match.
    """
    leaked_tokens = _coverage_token_list(leaked_value)
    if not leaked_tokens:
        return False

    final_token_lists = [tokens for tokens in (_coverage_token_list(value) for value in final_values) if tokens]
    if not final_token_lists:
        return False

    # Exact match against a single final value.
    if any(leaked_tokens == final_tokens for final_tokens in final_token_lists):
        return True

    # Subspan of a single final entity.
    leaked_core = set(leaked_tokens) - _COVERAGE_IGNORE_TOKENS
    if leaked_core and any(leaked_core <= set(final_tokens) for final_tokens in final_token_lists):
        return True

    # Composite: concatenation of whole final-entity values.
    return _is_concatenation_of_whole_values(leaked_tokens, final_token_lists)


def _filter_covered_leaked_entities(
    leaked_entities: list[dict[str, object]],
    final_entities: object,
) -> list[dict[str, object]]:
    """Drop judge-reported leaks that are already covered by final entity values."""
    if not isinstance(final_entities, list):
        return leaked_entities

    final_values = [str(entity.get("value", "")) for entity in final_entities if isinstance(entity, dict)]
    if not final_values:
        return leaked_entities

    return [entity for entity in leaked_entities if not _is_leaked_value_covered(entity.get("value", ""), final_values)]


def _normalize_literal_text(value: object) -> str:
    """Normalize case and whitespace while preserving the literal token sequence."""
    return " ".join(str(value).casefold().split())


def _filter_nonliteral_entities(
    entities: list[dict[str, object]],
    original_text: object,
) -> list[dict[str, object]]:
    """Drop judge-reported values that are not literal spans in the original text."""
    normalized_original = _normalize_literal_text(original_text)
    return [
        entity
        for entity in entities
        if (value := _normalize_literal_text(entity.get("value", ""))) and value in normalized_original
    ]


def _deduplicate_judge_entities(entities: list[dict[str, object]]) -> list[dict[str, object]]:
    """Keep one judge entity per normalized (value, label) pair."""
    deduplicated: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for entity in entities:
        key = (
            _normalize_literal_text(entity.get("value", "")),
            _normalize_literal_text(entity.get("label", "")),
        )
        if not key[0] or key in seen:
            continue
        seen.add(key)
        deduplicated.append(entity)
    return deduplicated


def _compare_judge_to_final(
    judge_entities: list[dict[str, object]],
    final_entities: object,
) -> tuple[float, list[dict[str, object]], int]:
    """Return independent-judge recall, missed entities, and judge total."""
    deduplicated = _deduplicate_judge_entities(judge_entities)
    leaked = _filter_covered_leaked_entities(deduplicated, final_entities)
    total = len(deduplicated)
    coverage = 1.0 if total == 0 else (total - len(leaked)) / total
    return coverage, leaked, total


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class EntityCoverageWorkflow(_BaseJudgeWorkflow):
    """LLM judge that reports entities not covered by Anonymizer final entities.

    The judge sees the original text, entity-type scope, and final entities.
    Deterministic postprocessing removes nonliteral and already-covered findings.

    Output columns:
      ``COL_ENTITY_COVERAGE`` (float|None) — n_final / (n_final + n_leaked)
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
        working_df[_FINAL_ENTITIES_FOR_COVERAGE_COL] = parsed.apply(_final_entities_for_coverage)
        return working_df

    def _passthrough_mask(self, dataframe: pd.DataFrame) -> pd.Series:
        # Independent extraction must run even when Anonymizer found no entities.
        return pd.Series(False, index=dataframe.index, dtype=bool)

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
        """Validate judge-reported leaks and calculate coverage."""
        out = dataframe.copy()

        coverage_vals: list[float | None] = []
        leaked_lists: list[list[dict]] = []
        total_vals: list[int | None] = []
        candidate_total_vals: list[int | None] = []

        for idx in out.index:
            raw = out[self.RAW_COL].loc[idx] if self.RAW_COL in out.columns else None
            leaked = _parse_leaked_entities(raw)
            if leaked is None:
                coverage_vals.append(None)
                leaked_lists.append([])
                total_vals.append(None)
                candidate_total_vals.append(None)
            else:
                leaked = _filter_nonliteral_entities(leaked, out[COL_TEXT].loc[idx])
                leaked = _deduplicate_judge_entities(leaked)
                candidate_total_vals.append(len(leaked))
                final_entities = out[_FINAL_ENTITIES_FOR_COVERAGE_COL].loc[idx]
                leaked = _filter_covered_leaked_entities(leaked, final_entities)
                n_final = len(final_entities) if isinstance(final_entities, list) else 0
                total = n_final + len(leaked)
                coverage = 1.0 if total == 0 else n_final / total
                coverage_vals.append(coverage)
                leaked_lists.append(leaked)
                total_vals.append(total)

        out[self.VALID_COL] = coverage_vals
        out[self.INVALID_COL] = leaked_lists
        out[COL_ENTITY_COVERAGE_TOTAL] = total_vals
        out[COL_ENTITY_COVERAGE_CANDIDATE_TOTAL] = candidate_total_vals
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
            for col in (
                self.RAW_COL,
                self.VALID_COL,
                self.INVALID_COL,
                COL_ENTITY_COVERAGE_TOTAL,
                COL_ENTITY_COVERAGE_CANDIDATE_TOTAL,
            ):
                if col in result.dataframe.columns:
                    out[col] = result.dataframe[col].values
            return out, result.failed_records
        except Exception:
            logger.warning("Entity coverage step failed; populating defaults", exc_info=True)
            out = dataframe.copy()
            out[self.VALID_COL] = None
            out[self.INVALID_COL] = [[] for _ in range(len(out))]
            out[COL_ENTITY_COVERAGE_TOTAL] = None
            out[COL_ENTITY_COVERAGE_CANDIDATE_TOTAL] = None
            return out, []

    def evaluate(
        self,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: EvaluateModelSelection,
        preview_num_records: int | None = None,
    ) -> JudgeResult:
        """Run leak detection against the supplied final entities."""
        working_df = self.prepare(dataframe)
        working_df[ROW_ORDER_COL] = range(len(working_df))
        effective_preview = min(preview_num_records, len(working_df)) if preview_num_records is not None else None
        run_result = self._adapter.run_workflow(
            working_df,
            model_configs=model_configs,
            columns=[self.column_config(selected_models)],
            workflow_name=self.WORKFLOW_NAME,
            preview_num_records=effective_preview,
        )

        judged_df = self.postprocess(run_result.dataframe)
        combined = merge_and_reorder(judged_df)
        return JudgeResult(dataframe=combined, failed_records=run_result.failed_records)
