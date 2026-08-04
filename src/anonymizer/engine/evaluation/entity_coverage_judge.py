# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import re
from typing import ClassVar

import pandas as pd
from data_designer.config.column_configs import LLMStructuredColumnConfig
from data_designer.config.models import ModelConfig
from pydantic import BaseModel, Field

from anonymizer.config.models import EvaluateModelSelection
from anonymizer.engine.constants import (
    COL_ENTITIES_BY_VALUE,
    COL_ENTITY_COVERAGE,
    COL_ENTITY_COVERAGE_JUDGE,
    COL_MISSED_ENTITIES,
    COL_TEXT,
    DEFAULT_ENTITY_LABELS,
    _jinja,
)
from anonymizer.engine.evaluation.judge_base import JudgeResult, _BaseJudgeWorkflow
from anonymizer.engine.ndd.adapter import RECORD_ID_COLUMN, FailedRecord, NddAdapter
from anonymizer.engine.ndd.model_loader import resolve_model_alias
from anonymizer.engine.prompt_utils import substitute_placeholders
from anonymizer.engine.row_partitioning import ROW_ORDER_COL, merge_and_reorder
from anonymizer.engine.schemas import EntitiesByValueSchema

logger = logging.getLogger("anonymizer.evaluation.entity_coverage_judge")

_FINAL_ENTITIES_FOR_COVERAGE_COL = "_final_entities_for_coverage_judge"
_WORD_RE = re.compile(r"\w+", re.UNICODE)
# Leading articles stripped during core-token normalization so that a judge value
# like "the Nawabganj" still matches a detected entity "Nawabganj". Restricted to
# leading position only — prepositions such as "of" and "at" are intentionally
# excluded because they can be load-bearing in entity names (e.g. "Bank of America",
# "AT&T") and stripping them from arbitrary positions would suppress real leaks.
_LEADING_ARTICLES = frozenset({"a", "an", "the"})


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class CandidateEntity(BaseModel):
    value: str = Field(description="Exact text span from the original text that is a PII or sensitive entity.")
    label: str = Field(description="Entity type label (e.g. first_name, email, phone_number).")
    reasoning: str = Field(description="One sentence explaining why this value is a PII or sensitive entity.")


class EntityCoverageSchema(BaseModel):
    candidate_entities: list[CandidateEntity] = Field(
        description="All in-scope PII and sensitive entities found in the original text. "
        "Empty list when the original text contains no in-scope entity values.",
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
        "Strict mode lowers the threshold for borderline literal spans — it does NOT change\n"
        "the requirement that every flagged value must be literally present in the original text.\n"
        "</strict_entity_protection>"
    )


def _data_summary_block(data_summary: str | None) -> str:
    """Return optional dataset context without changing prompts when absent."""
    if not data_summary or not data_summary.strip():
        return ""
    return (
        "<data_summary_context>\n"
        f"{data_summary.strip()}\n"
        "Use this context only to interpret literal values and their semantic types. "
        "Do not infer or invent entities that are absent from the original text.\n"
        "</data_summary_context>"
    )


def _coverage_prompt(
    *, entity_labels: list[str] | None, strict_entity_protection: bool, data_summary: str | None = None
) -> str:
    entity_scope_block = _entity_type_scope_block(entity_labels)
    strict_block = _strict_protection_block(strict_entity_protection)
    data_context_section = f"\n\n{_data_summary_block(data_summary)}" if data_summary and data_summary.strip() else ""

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

    prompt = f"""You are a privacy-entity span extractor. Your task is defined below.

<task>
Work independently from the anonymizer: identify all in-scope direct and quasi-identifiers \
without deciding whether another system already found or protected them. A deterministic \
postprocessing step will compare your candidates with the anonymizer final entities.

Return structured JSON:
- `candidate_entities`: every in-scope candidate with its `value`, `label`, and a short `reasoning`.
- Return an empty list only when the original text contains no in-scope entity values.
</task>

<identifier_taxonomy>
These entity types are in scope: {labels_str}.
Quasi-identifiers: combinations of values that together re-identify someone \
(e.g. job title + employer + city appearing together). Time values (specific timestamps, \
times of day, schedules) can act as quasi-identifiers when combined with other attributes \
in the same text — flag them if they appear alongside other identifying information.
</identifier_taxonomy>{data_context_section}

{entity_scope_block}

<label_interpretation>
Treat each configured label as a semantic entity category. Labels may use compact, compound, \
or abbreviated names; interpret their intended meaning from the label and the original-text \
context. Return labels exactly as they appear in the entity_type_scope.
</label_interpretation>

<systematic_scan>
Before returning, scan the entire text label by label and inspect:
- Prose, salutations, signatures, addresses, and contact blocks.
- Tables, bullets, forms, and other key/value or semi-structured content.
- Short or single-token values whose nearby wording or syntax establishes their type.
- Honorifics attached to person names, compact date formats used as dates, and categorical \
or coded values whose context establishes an in-scope type.
</systematic_scan>

<reportability_check>
For each candidate, verify ALL of the following:
1. Its value is a literal, non-empty span in the original text.
2. It is an actual assigned or stated value, not a generic field name, heading, instruction, \
blank placeholder, or category name.
3. Report the complete contiguous span that represents one sensitive value. Preserve all \
tokens belonging to that value, including multi-token values, while excluding surrounding \
labels, punctuation, instructions, or boilerplate.
4. Evaluate the value using its original-text context rather than its form or how identifying \
it appears in isolation.

In structured text, distinguish a generic category name from a literal category value. A short \
literal that itself instantiates an in-scope type remains a candidate when it appears in a form \
row, list item, or signature. Do not merge tokens from unrelated people or fields into one value.
</reportability_check>

<guidance>
Do NOT flag:
- Text whose in-scope semantic type is not supported by its original-text context.
- Information that is inferable but not literally present in the text.

Do flag:
- `reasoning` MUST be one sentence explaining which in-scope semantic type the value represents.
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
</inputs>
"""
    return substitute_placeholders(
        prompt,
        {
            "<<COL_TEXT>>": _jinja(COL_TEXT),
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _final_entities_for_coverage(parsed: EntitiesByValueSchema) -> list[dict[str, str]]:
    """Flatten EntitiesByValueSchema into one entry per (value, label) pair.

    The coverage denominator counts (value, label) pairs rather than unique
    values or source occurrences. This keeps the numerator and denominator in
    the same unit — the judge also returns missed entities as (value, label)
    pairs — so the score remains consistent. An entity detected under multiple
    labels (e.g. "Alice" as both first_name and user_name) contributes one
    entry per label, which is intentional: each detection deserves credit.
    """
    return [{"value": e.value, "label": label} for e in parsed.entities_by_value for label in e.labels]


def _parse_missed_entities(raw: object) -> list[dict[str, object]] | None:
    """Parse structured judge output into the leaked entity list.

    Returns the list (possibly empty) on success, or None when the payload is
    malformed or missing so downstream display renders "judge unavailable".
    """
    if raw is None:
        return None
    if isinstance(raw, BaseModel):
        raw = raw.model_dump(mode="python")
    if not isinstance(raw, dict):
        return None
    try:
        parsed = EntityCoverageSchema.model_validate(raw)
    except Exception:
        return None
    return [e.model_dump() for e in parsed.candidate_entities]


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


def _core_token_sequence(tokens: list[str]) -> list[str]:
    """Tokens with a leading article (a/an/the) stripped, preserving all other positions.

    Only the leading article is removed so that prepositions load-bearing in entity
    names (e.g. "of" in "Bank of America") are never silently dropped.
    """
    if tokens and tokens[0] in _LEADING_ARTICLES:
        return tokens[1:]
    return tokens


def _is_contiguous_sublist(needle: list[str], haystack: list[str]) -> bool:
    """True when ``needle`` appears as a contiguous, in-order run within ``haystack``."""
    if not needle or len(needle) > len(haystack):
        return False
    return any(haystack[i : i + len(needle)] == needle for i in range(len(haystack) - len(needle) + 1))


def _is_leaked_value_covered(leaked_value: object, final_values: list[str]) -> bool:
    """Return True when a judge-reported leak is already covered by final entities.

    Coverage is decided **per final entity** — never against a pooled bag of tokens
    from *all* final entities — so a leak whose pieces come from unrelated entities is
    not wrongly suppressed (e.g. ``"John Smith"`` is NOT covered by ``"John Doe"`` +
    ``"Jane Smith"``). A leak is covered when either:

    - **subspan** — its (stopword-stripped) tokens appear as a *contiguous, in-order run*
      within a single final entity's tokens (``"Mstr"`` in ``"Mstr Marzella"``,
      ``"White House"`` in ``"White House Road"``). Contiguity and order are required —
      shared tokens alone do not qualify — so ``"Ann Lee"`` is NOT covered by
      ``"Lee Ann Boulevard"``.
    - **composite** — its tokens are a concatenation of *whole* final-entity values
      (``"Nawabganj - 382210"`` == ``"Nawabganj"`` + ``"382210"``).

    Matching is whole-token (``"m"`` != ``"mstr"``) and **value-only** — labels are not
    compared. Known limitation: a single-token leak is covered whenever that token equals
    a whole token in ANY final entity, regardless of type — e.g. a surname ``"Green"`` is
    treated as covered by a detected street ``"Bowling Green Road"``. This mirrors the
    intended "bare username covered by a file path that contains it" behavior; fixing it
    would require label-aware matching, which is deliberately avoided (judge labels are
    free-form), so it is accepted as a tradeoff.
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

    # Subspan: the leak's core tokens appear as a contiguous, in-order run within a single
    # final entity (adjacency + order required, not merely a shared set of tokens).
    leaked_core = _core_token_sequence(leaked_tokens)
    if leaked_core and any(
        _is_contiguous_sublist(leaked_core, _core_token_sequence(final_tokens)) for final_tokens in final_token_lists
    ):
        return True

    # Composite: concatenation of whole final-entity values.
    return _is_concatenation_of_whole_values(leaked_tokens, final_token_lists)


def _filter_covered_missed_entities(
    missed_entities: list[dict[str, object]],
    final_entities: object,
) -> list[dict[str, object]]:
    """Drop judge-reported leaks that are already covered by final entity values."""
    if not isinstance(final_entities, list):
        return missed_entities

    final_values = [str(entity.get("value", "")) for entity in final_entities if isinstance(entity, dict)]
    if not final_values:
        return missed_entities

    return [entity for entity in missed_entities if not _is_leaked_value_covered(entity.get("value", ""), final_values)]


def _normalize_literal_text(value: object) -> str:
    """Normalize case and whitespace while preserving the literal token sequence."""
    return " ".join(str(value).casefold().split())


def _filter_out_of_scope_entities(
    entities: list[dict[str, object]],
    entity_labels: list[str] | None,
) -> list[dict[str, object]]:
    """Drop entities with empty labels or labels outside the configured scope.

    When ``entity_labels`` is None all labels are in scope; only empty labels
    are dropped. This mirrors the prompt's scope instruction deterministically
    so a model that returns out-of-scope labels does not lower the coverage score.

    Label drift (e.g. the model returning ``"given_name"`` instead of
    ``"first_name"``) is unlikely in practice — the prompt explicitly instructs
    the model to return labels exactly as they appear in the entity_type_scope,
    and ``LLMStructuredColumnConfig`` reinforces this via the schema field
    descriptions. The filter therefore drops genuine hallucinated labels without
    meaningfully risking false negatives on well-formed responses.
    """
    allowed = {label.casefold() for label in entity_labels} if entity_labels is not None else None
    result = []
    for entity in entities:
        label = str(entity.get("label", "")).strip()
        if not label:
            continue
        if allowed is not None and label.casefold() not in allowed:
            continue
        result.append(entity)
    return result


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


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class EntityCoverageWorkflow(_BaseJudgeWorkflow):
    """LLM judge that reports entities not covered by Anonymizer final entities.

    The judge independently extracts candidates from the original text and entity-type
    scope. Deterministic postprocessing removes nonliteral and already-covered findings.

    Output columns:
      ``COL_ENTITY_COVERAGE`` (float|None) — n_final / (n_final + n_leaked)
      ``COL_MISSED_ENTITIES`` (list)        — missed entities with value, label, reasoning
    """

    RAW_COL: ClassVar[str] = COL_ENTITY_COVERAGE_JUDGE
    VALID_COL: ClassVar[str] = COL_ENTITY_COVERAGE
    INVALID_COL: ClassVar[str] = COL_MISSED_ENTITIES
    SCHEMA: ClassVar[type[BaseModel]] = EntityCoverageSchema
    VERDICT_FIELD: ClassVar[str] = "candidate_entities"
    DEFAULT_PAYLOAD: ClassVar[dict] = {"candidate_entities": []}
    MODEL_ROLE: ClassVar[str] = "entity_coverage_judge"
    WORKFLOW_NAME: ClassVar[str] = "entity-coverage-judge"

    def __init__(
        self,
        adapter: NddAdapter,
        *,
        entity_labels: list[str] | None = None,
        strict_entity_protection: bool = False,
        data_summary: str | None = None,
    ) -> None:
        super().__init__(adapter)
        self._entity_labels = entity_labels
        self._strict_entity_protection = strict_entity_protection
        self._data_summary = data_summary

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
        """Unused abstract-base hook; instance configuration is required here.

        ``column_config()`` below builds the prompt from ``entity_labels``,
        ``strict_entity_protection``, and ``data_summary``. Fail loudly if a
        future refactor accidentally routes through the base implementation
        instead of silently evaluating with incorrect default scope.
        """
        raise NotImplementedError("EntityCoverageWorkflow builds its prompt in column_config().")

    @classmethod
    def _extract_invalid(cls, parsed: BaseModel) -> list[dict[str, object]]:
        return [e.model_dump() for e in parsed.candidate_entities]

    # ----------------------------------------------------------------- overrides

    def column_config(self, selected_models: EvaluateModelSelection) -> LLMStructuredColumnConfig:
        """Override to inject instance-specific entity_labels and strict_entity_protection."""
        return LLMStructuredColumnConfig(
            name=self.RAW_COL,
            prompt=_coverage_prompt(
                entity_labels=self._entity_labels,
                strict_entity_protection=self._strict_entity_protection,
                data_summary=self._data_summary,
            ),
            model_alias=resolve_model_alias(self.MODEL_ROLE, selected_models),
            output_format=EntityCoverageSchema,
        )

    def postprocess(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Validate judge-reported leaks and calculate coverage."""
        out = dataframe.copy()

        coverage_vals: list[float | None] = []
        leaked_lists: list[list[dict]] = []

        for idx in out.index:
            raw = out[self.RAW_COL].loc[idx] if self.RAW_COL in out.columns else None
            leaked = _parse_missed_entities(raw)
            if leaked is None:
                coverage_vals.append(None)
                leaked_lists.append([])
            else:
                leaked = _filter_out_of_scope_entities(leaked, self._entity_labels)
                leaked = _filter_nonliteral_entities(leaked, out[COL_TEXT].loc[idx])
                leaked = _deduplicate_judge_entities(leaked)
                final_entities = out[_FINAL_ENTITIES_FOR_COVERAGE_COL].loc[idx]
                leaked = _filter_covered_missed_entities(leaked, final_entities)
                n_final = len(final_entities) if isinstance(final_entities, list) else 0
                total = n_final + len(leaked)
                coverage = 1.0 if total == 0 else n_final / total
                coverage_vals.append(coverage)
                leaked_lists.append(leaked)

        out[self.VALID_COL] = coverage_vals
        out[self.INVALID_COL] = leaked_lists
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

        Rows the LLM drops get ``entity_coverage=None`` / ``missed_entities=[]``
        rather than disappearing. On total workflow failure, all rows are defaulted.
        Returns ``(annotated_df, failed_records)``.
        """
        try:
            had_record_ids = RECORD_ID_COLUMN in dataframe.columns
            prepared = self._adapter._attach_record_ids(dataframe)
            result = self.evaluate(
                prepared,
                model_configs=model_configs,
                selected_models=selected_models,
                preview_num_records=preview_num_records,
            )
            if RECORD_ID_COLUMN not in result.dataframe.columns:
                raise ValueError("Entity coverage output is missing record IDs required for row alignment.")

            score_cols = [
                col for col in (self.RAW_COL, self.VALID_COL, self.INVALID_COL) if col in result.dataframe.columns
            ]
            out = prepared.drop(columns=score_cols, errors="ignore").merge(
                result.dataframe[[RECORD_ID_COLUMN, *score_cols]],
                on=RECORD_ID_COLUMN,
                how="left",
                sort=False,
                validate="one_to_one",
            )

            if self.VALID_COL not in out.columns:
                out[self.VALID_COL] = None
            else:
                out[self.VALID_COL] = out[self.VALID_COL].astype(object)
                out.loc[out[self.VALID_COL].isna(), self.VALID_COL] = None
            if self.INVALID_COL not in out.columns:
                out[self.INVALID_COL] = [[] for _ in range(len(out))]
            else:
                out[self.INVALID_COL] = out[self.INVALID_COL].apply(
                    lambda value: value if isinstance(value, list) else []
                )
            if not had_record_ids:
                out = out.drop(columns=[RECORD_ID_COLUMN])
            return out, result.failed_records
        except Exception as exc:
            logger.warning("Entity coverage workflow failed; scores may be unavailable. Reason: %s", exc)
            logger.debug("Entity coverage workflow failed.", exc_info=True)
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
