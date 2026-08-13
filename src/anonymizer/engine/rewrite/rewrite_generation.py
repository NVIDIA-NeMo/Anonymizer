# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import re
from typing import Any

from data_designer.config import custom_column_generator
from data_designer.config.column_configs import CustomColumnConfig, LLMStructuredColumnConfig
from data_designer.config.column_types import ColumnConfigT

from anonymizer.config.models import RewriteModelSelection
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import (
    COL_FINAL_ENTITIES,
    COL_FULL_REWRITE,
    COL_PREREPLACE_TAGGED_TEXT,
    COL_PREREPLACE_TEXT,
    COL_REPLACEMENT_MAP,
    COL_REWRITE_DISPOSITION_BLOCK,
    COL_REWRITTEN_TEXT,
    COL_SENSITIVITY_DISPOSITION,
    COL_TAG_NOTATION,
    COL_TAGGED_TEXT,
    COL_TEXT,
    _jinja,
)
from anonymizer.engine.ndd.model_loader import resolve_model_alias
from anonymizer.engine.prompt_utils import substitute_placeholders
from anonymizer.engine.replace.strategies import ReplacementEntry, _apply_replacement_map_to_text
from anonymizer.engine.rewrite.parsers import normalize_payload, parse_sensitivity_disposition
from anonymizer.engine.schemas import (
    EntitiesSchema,
    EntityReplacementMapSchema,
    RewriteOutputSchema,
)

logger = logging.getLogger("anonymizer.rewrite.generation")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def _get_rewrite_prompt(privacy_goal: PrivacyGoal, data_summary: str | None = None) -> str:
    """Build the full rewrite prompt with XML section headers."""
    data_context_section = ""
    if data_summary and data_summary.strip():
        data_context_section = "\n<data_context>\nDataset description: " + data_summary.strip() + "\n</data_context>\n"

    prompt = """You are an expert writer. You excel at rewriting, paraphrasing, rewording, and following instructions.

<instructions>
Your task is to rewrite the text below so that it protects the privacy of the entities described,
following the entity protection rules provided. The rewrite must read naturally as
plain, fluent text — no tags, brackets, or annotation artifacts.

Apply each protection decision consistently across ALL occurrences of the same entity value.
Do not add justification text or commentary in the output. Only output the rewritten text.
</instructions>

<privacy_goal>
<<PRIVACY_GOAL>>
</privacy_goal>
<<DATA_CONTEXT>>
<input>
The text below contains inline entity tags marking identified entities.
{% if <<TAG_NOTATION>> == 'bracket' %}Tags use the format [[entity_value|entity_label]]. Remove all [[...]] tags.
{% elif <<TAG_NOTATION>> == 'xml' %}Tags use the format <entity_label>entity_value</entity_label>. Remove all XML entity tags.
{% elif <<TAG_NOTATION>> == 'paren' %}Tags use the format ((SENSITIVE:entity_label|entity_value)). Remove all ((SENSITIVE:...)) tags.
{% elif <<TAG_NOTATION>> == 'sentinel' %}Tags use the format <<SENSITIVE:entity_label>>entity_value<</SENSITIVE:entity_label>>. Remove all <<SENSITIVE:...>> tags.
{% endif %}
The rewritten text must read like normal prose with no tags remaining.

Tagged text:
<<TAGGED_TEXT>>
</input>

<sensitivity_disposition>
Protection decisions for each entity that needs protection:
{% for entity in <<REWRITE_DISPOSITION_BLOCK>> %}
- {{ entity.entity_label }}: "{{ entity.entity_value }}"
  Sensitivity: {{ entity.sensitivity }}
  Protection method: {{ entity.protection_method_suggestion }}
  Reason: {{ entity.protection_reason }}
{% endfor %}

Entities NOT listed above may be kept as-is.
</sensitivity_disposition>

<output_requirements>
Apply each protection method as follows:
- "generalize": Replace with a broader category or range
  (e.g., a specific city → "a city in the Pacific Northwest", exact age → "in their late 30s").
- "remove": Omit the detail entirely. Rewrite the surrounding sentence so it reads naturally without it.
- "suppress_inference": Modify the text so the attribute cannot be reliably inferred by a motivated reader.

Rules:
1. ALL entity tags (as described above) must be removed. Output must be plain text.
2. Apply changes consistently — the same entity value must be treated the same way everywhere it appears.
3. Entities with protection_method_suggestion="leave_as_is" should be retained verbatim (tags removed only).
4. The rewritten text must flow naturally and preserve the meaning and narrative structure of the original.
5. Do not introduce new identifying details not present in the original.
</output_requirements>"""
    return substitute_placeholders(
        prompt,
        {
            "<<PRIVACY_GOAL>>": privacy_goal.to_prompt_string(),
            "<<DATA_CONTEXT>>": data_context_section,
            "<<TAG_NOTATION>>": COL_TAG_NOTATION,
            "<<TAGGED_TEXT>>": _jinja(COL_PREREPLACE_TAGGED_TEXT),
            "<<REWRITE_DISPOSITION_BLOCK>>": COL_REWRITE_DISPOSITION_BLOCK,
        },
    )


# ---------------------------------------------------------------------------
# Custom column generators (pure Python, no LLM)
# ---------------------------------------------------------------------------


@custom_column_generator(required_columns=[COL_SENSITIVITY_DISPOSITION])
def _format_rewrite_disposition_block(row: dict[str, Any]) -> dict[str, Any]:
    """Pre-filter and serialize protected entities for the rewrite prompt.

    Excludes leave_as_is entities and replace entities (the latter are handled
    programmatically by _apply_direct_replacements before the LLM sees the text).
    """
    disposition = parse_sensitivity_disposition(row[COL_SENSITIVITY_DISPOSITION])
    block = []
    for e in disposition.sensitivity_disposition:
        if not e.needs_protection:
            continue
        if e.protection_method_suggestion == "replace":
            continue
        d = e.model_dump(mode="json")
        block.append(
            {
                "entity_label": d["entity_label"],
                "entity_value": d["entity_value"],
                "sensitivity": d["sensitivity"],
                "protection_method_suggestion": d["protection_method_suggestion"],
                "protection_reason": d["protection_reason"],
            }
        )
    row[COL_REWRITE_DISPOSITION_BLOCK] = block
    return row


def _normalize_ws(s: str) -> str:
    """Collapse all Unicode whitespace variants to a single ASCII space."""
    return " ".join(s.split())


def _get_replace_pairs(row: dict[str, Any]) -> tuple[list[tuple[str, str, str]], set[str]]:
    """Return (pairs, replace_values) for entities with protection_method='replace'.

    ``pairs`` contains (original, synthetic, label) tuples ready for substitution.
    ``replace_values`` is the full set of entity values that required replacement,
    returned so the caller can detect and raise on any unmatched entries.

    Falls back to whitespace-normalized matching when the map's ``original`` field
    differs only in Unicode whitespace from the disposition entity value (e.g. the
    LLM normalised U+202F → U+0020). In that case the disposition value is used as
    the substitution key because it reflects what is actually present in the text.
    """
    disposition = parse_sensitivity_disposition(row[COL_SENSITIVITY_DISPOSITION])
    replace_values = {
        e.entity_value for e in disposition.sensitivity_disposition if e.protection_method_suggestion == "replace"
    }
    if not replace_values:
        return [], set()
    raw_map = row.get(COL_REPLACEMENT_MAP)
    if not raw_map:
        return [], replace_values
    raw_map = normalize_payload(raw_map)
    if hasattr(raw_map, "model_dump"):
        raw_map = raw_map.model_dump(mode="python")
    parsed_map = EntityReplacementMapSchema.model_validate(raw_map)

    # normalized form → original disposition value (for fuzzy fallback)
    normalized_to_disposition: dict[str, str] = {_normalize_ws(v): v for v in replace_values}

    pairs: list[tuple[str, str, str]] = []  # (original, synthetic, label)
    matched: set[str] = set()
    for r in parsed_map.replacements:
        if r.original in replace_values:
            pairs.append((r.original, r.synthetic, r.label))
            matched.add(r.original)
        else:
            disposition_value = normalized_to_disposition.get(_normalize_ws(r.original))
            if disposition_value is not None and disposition_value not in matched:
                pairs.append((disposition_value, r.synthetic, r.label))
                matched.add(disposition_value)

    return pairs, replace_values


def _apply_tagged_text_replacements(tagged_text: str, pairs: list[tuple[str, str, str]], tag_notation: str) -> str:
    """Replace entity values in tagged text using a single-pass, tag-boundary-aware substitution.

    Builds a lookup from full tagged span → replacement tagged span, then applies all
    substitutions in one regex pass so a synthetic value that matches another entity's
    original is never re-replaced (same cascade-prevention guarantee as the plain-text path).
    """
    if not pairs:
        return tagged_text
    lookup: dict[str, str] = {}
    for original, synthetic, label in sorted(pairs, key=lambda p: len(p[0]), reverse=True):
        if tag_notation == "xml":
            tagged_original = f"<{label}>{original}</{label}>"
            tagged_synthetic = f"<{label}>{synthetic}</{label}>"
        elif tag_notation == "bracket":
            tagged_original = f"[[{original}|{label}]]"
            tagged_synthetic = f"[[{synthetic}|{label}]]"
        elif tag_notation == "paren":
            tagged_original = f"((SENSITIVE:{label}|{original}))"
            tagged_synthetic = f"((SENSITIVE:{label}|{synthetic}))"
        else:  # sentinel
            tagged_original = f"<<SENSITIVE:{label}>>{original}<</SENSITIVE:{label}>>"
            tagged_synthetic = f"<<SENSITIVE:{label}>>{synthetic}<</SENSITIVE:{label}>>"
        lookup[tagged_original] = tagged_synthetic
    pattern = re.compile("|".join(re.escape(k) for k in lookup))
    return pattern.sub(lambda m: lookup[m.group(0)], tagged_text)


@custom_column_generator(
    required_columns=[
        COL_SENSITIVITY_DISPOSITION,
        COL_REPLACEMENT_MAP,
        COL_TEXT,
        COL_TAGGED_TEXT,
        COL_FINAL_ENTITIES,
        COL_TAG_NOTATION,
    ],
    side_effect_columns=[COL_PREREPLACE_TAGGED_TEXT],
)
def _apply_direct_replacements(row: dict[str, Any]) -> dict[str, Any]:
    """Programmatically replace direct identifier entities before the rewrite LLM call.

    Uses span-aware replacement for plain text (character offsets from COL_FINAL_ENTITIES)
    and tag-boundary-aware replacement for tagged text, preventing substring corruption
    (e.g. replacing 'Ann' must not modify 'Anna').

    Raises on failure rather than falling back to unmodified text: replace entities are
    excluded from COL_REWRITE_DISPOSITION_BLOCK, so a silent passthrough would send
    PII-containing text to the LLM with no instructions to protect those entities.
    """
    plain_text = str(row.get(COL_TEXT, ""))
    tagged_text = str(row.get(COL_TAGGED_TEXT, ""))
    pairs, replace_values = _get_replace_pairs(row)
    matched = {original for original, _, _ in pairs}
    unmatched = replace_values - matched
    if unmatched:
        disposition = parse_sensitivity_disposition(row[COL_SENSITIVITY_DISPOSITION])
        value_to_label: dict[str, str] = {e.entity_value: e.entity_label for e in disposition.sensitivity_disposition}
        unmatched_labels = sorted({value_to_label.get(v, "unknown") for v in unmatched})
        raise RuntimeError(
            f"Replace entities have no entry in the replacement map; refusing to pass PII-containing text "
            f"to the rewrite LLM without protection instructions. "
            f"{len(unmatched)} entities unmatched (labels: {unmatched_labels})"
        )
    if pairs:
        tag_notation = str(row.get(COL_TAG_NOTATION, "xml"))
        # Plain text: span-aware using character offsets from COL_FINAL_ENTITIES.
        # Falls back to sorted-pairs regex when entities are absent (e.g. in unit tests
        # that call this function directly without a full detection pipeline).
        all_entities = EntitiesSchema.from_raw(row.get(COL_FINAL_ENTITIES, {}))
        replace_value_set = {original for original, _, _ in pairs}
        replace_entities = EntitiesSchema(
            entities=[
                e for e in all_entities.entities if e.value in replace_value_set and e.end_position > e.start_position
            ]
        )
        replacement_entries = [
            ReplacementEntry(original=original, label=label, synthetic=synthetic)
            for original, synthetic, label in pairs
        ]
        if replace_entities.entities:
            plain_text = _apply_replacement_map_to_text(plain_text, replace_entities, replacement_entries)
        else:
            sorted_pairs = sorted(pairs, key=lambda p: len(p[0]), reverse=True)
            pattern = re.compile("|".join(re.escape(original) for original, _, _ in sorted_pairs))
            lookup = {original: synthetic for original, synthetic, _ in sorted_pairs}
            plain_text = pattern.sub(lambda m: lookup[m.group(0)], plain_text)
        tagged_text = _apply_tagged_text_replacements(tagged_text, pairs, tag_notation)
    row[COL_PREREPLACE_TEXT] = plain_text
    row[COL_PREREPLACE_TAGGED_TEXT] = tagged_text
    return row


@custom_column_generator(required_columns=[COL_FULL_REWRITE])
def _extract_rewritten_text(row: dict[str, Any]) -> dict[str, Any]:
    """Extract rewritten_text from the LLM structured output.

    Sets ``COL_REWRITTEN_TEXT`` to ``None`` on failure or blank output so
    downstream steps (repair, judge, human-review flagging) can distinguish
    a failed rewrite from a valid one.
    """
    try:
        full_rewrite = row[COL_FULL_REWRITE]
        if hasattr(full_rewrite, "model_dump"):
            full_rewrite = full_rewrite.model_dump(mode="python")
        text = str(full_rewrite["rewritten_text"])
        if not text.strip():
            logger.warning("LLM returned blank rewritten_text; marking as unavailable.")
            row[COL_REWRITTEN_TEXT] = None
        else:
            row[COL_REWRITTEN_TEXT] = text
    except Exception:
        logger.warning("Failed to extract rewritten_text from COL_FULL_REWRITE; marking as unavailable.")
        row[COL_REWRITTEN_TEXT] = None
    return row


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class RewriteGenerationWorkflow:
    """Column factory for the rewrite generation step.

    Returns column configs for disposition-block formatting,
    replacement-map filtering, LLM rewrite, and text extraction.
    The orchestrator (``RewriteWorkflow``) collects these alongside
    domain/disposition/QA columns for a single adapter call.
    """

    def columns(
        self,
        *,
        selected_models: RewriteModelSelection,
        privacy_goal: PrivacyGoal,
        data_summary: str | None = None,
    ) -> list[ColumnConfigT]:
        rewriter_alias = resolve_model_alias("rewriter", selected_models)
        return [
            CustomColumnConfig(
                name=COL_REWRITE_DISPOSITION_BLOCK,
                generator_function=_format_rewrite_disposition_block,
            ),
            CustomColumnConfig(
                name=COL_PREREPLACE_TEXT,
                generator_function=_apply_direct_replacements,
            ),
            LLMStructuredColumnConfig(
                name=COL_FULL_REWRITE,
                prompt=_get_rewrite_prompt(privacy_goal, data_summary),
                model_alias=rewriter_alias,
                output_format=RewriteOutputSchema,
            ),
            CustomColumnConfig(
                name=COL_REWRITTEN_TEXT,
                generator_function=_extract_rewritten_text,
            ),
        ]
