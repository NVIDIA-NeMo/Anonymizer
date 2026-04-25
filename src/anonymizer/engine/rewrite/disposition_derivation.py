# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Server-side reconstruction of the strict EntityDispositionSchema from the
loose wire-contract SimpleDispositionResult + the per-entity context columns.

The disposition_analyzer LLM now emits a minimal `SimpleDispositionResult`
(8 optional/loose fields per item). This module rebuilds the strict form
deterministically: pair each simple item with its entity context (by id,
with entity_label/value echoes as belt-and-braces), derive needs_protection,
and template protection_reason when the model did not provide one.

No LLM calls; no I/O. Pure python for the reconstruction column.
"""

from __future__ import annotations

import logging

from anonymizer.engine.schemas.rewrite import (
    _ENTITY_LABEL_TO_CATEGORY,
    EntityDispositionSchema,
    SensitivityDispositionSchema,
    SimpleDispositionItem,
    SimpleDispositionResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Derivation helpers
# ---------------------------------------------------------------------------


_VALID_CATEGORIES: frozenset[str] = frozenset(
    {"direct_identifier", "quasi_identifier", "sensitive_attribute", "latent_identifier"}
)


def _normalize_category(raw: object, *, entity_label: str = "") -> str:
    """Resolve a free-form category string emitted by the disposition LLM
    into a valid EntityCategory value.

    Handles four small-model drift modes observed on gemma4-e2b / Nemotron:

    - **Display variants** — ``"Direct-Identifier"``, ``"DIRECT IDENTIFIERS"``
      → lowercased, separator-normalized, plural-stripped to the enum value.
    - **Merged enums** — ``"latent_sensitive_attribute"`` (Nemotron splices
      two enums) → matched by substring with strongest-protection priority,
      so harm dimension wins over inference dimension.
    - **Entity-label confusion** — ``"last_name"``, ``"date_of_birth"``
      written in the category slot → looked up in
      ``_ENTITY_LABEL_TO_CATEGORY`` and mapped back to the most-likely
      category. Source label provenance is preserved by the ``source``
      field on the strict schema.
    - **Empty / unknown** — falls back to ``"quasi_identifier"`` (the
      conservative default; pessimistic protection rather than dropping
      the row).
    """
    if not isinstance(raw, str) or not raw.strip():
        return "quasi_identifier"
    normalized = raw.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in _VALID_CATEGORIES:
        return normalized
    if normalized.endswith("s") and normalized[:-1] in _VALID_CATEGORIES:
        return normalized[:-1]
    if normalized.endswith("_identifiers"):
        return normalized[:-1]
    # Merged-enum hallucination: order = strongest protection wins so that
    # "latent_sensitive_attribute" maps to sensitive_attribute (harm) rather
    # than latent_identifier (inference).
    for sub, target in (
        ("direct", "direct_identifier"),
        ("sensitive", "sensitive_attribute"),
        ("latent", "latent_identifier"),
        ("quasi", "quasi_identifier"),
    ):
        if sub in normalized:
            return target
    # Entity-label confusion: model wrote an entity_label value in the slot.
    mapped = _ENTITY_LABEL_TO_CATEGORY.get(normalized)
    if mapped is not None:
        return mapped
    # Last resort: if the LLM echoed the entity_label verbatim into category,
    # fall back to quasi_identifier (matches the entity-label confusion path).
    if entity_label and normalized == entity_label.strip().lower():
        return "quasi_identifier"
    return "quasi_identifier"


def derive_needs_protection(method: str) -> bool:
    """Tautological with EntityDispositionSchema._validate_protection_consistency.

    If the model picks any method other than leave_as_is, the entity needs
    protection; otherwise it does not. Deriving this instead of asking the
    LLM for it eliminates the consistency-rule drift (class K).
    """
    return (method or "").strip() != "leave_as_is"


# (category, method) -> template text (without leading sensitivity prefix).
# Sensitivity fills a prefix ("high-risk ...", "moderate-risk ...", "").
_REASON_TEMPLATES: dict[tuple[str, str], str] = {
    ("direct_identifier", "replace"): "direct identifier — replaced with a contextual surrogate",
    ("direct_identifier", "remove"): "direct identifier — removed to prevent re-identification",
    ("direct_identifier", "generalize"): "direct identifier — generalized to reduce re-identification",
    ("direct_identifier", "suppress_inference"): "direct identifier — suppressed to prevent inference",
    ("quasi_identifier", "generalize"): "quasi-identifier — generalized to reduce re-identification risk",
    ("quasi_identifier", "replace"): "quasi-identifier — replaced with a plausible surrogate",
    ("quasi_identifier", "remove"): "quasi-identifier — removed due to re-identification risk",
    ("quasi_identifier", "suppress_inference"): "quasi-identifier — suppressed to prevent inference",
    ("sensitive_attribute", "remove"): "sensitive attribute — removed to prevent disclosure harm",
    ("sensitive_attribute", "generalize"): "sensitive attribute — generalized to reduce harm",
    ("sensitive_attribute", "suppress_inference"): "sensitive attribute — suppressed to prevent disclosure",
    ("sensitive_attribute", "replace"): "sensitive attribute — replaced with a less harmful value",
    ("latent_identifier", "suppress_inference"): "latent inference — suppressed to prevent deduction",
    ("latent_identifier", "remove"): "latent identifier — removed to prevent inference",
    ("latent_identifier", "generalize"): "latent identifier — generalized to reduce inference",
    ("latent_identifier", "replace"): "latent identifier — replaced with a less specific surrogate",
}

_SENSITIVITY_PREFIX = {"low": "", "medium": "moderate-risk ", "high": "high-risk "}


def template_protection_reason(category: str, method: str, sensitivity: str) -> str:
    """Build a reason string guaranteed ≥10 chars (EntityDispositionSchema min_length).

    Used when the LLM omits or emits a too-short protection_reason. Strong
    models that provide their own document-specific reason have theirs
    kept verbatim by the reconstructor.
    """
    method = (method or "").strip()
    category = (category or "").strip()
    sensitivity = (sensitivity or "").strip().lower()

    if method == "leave_as_is":
        cat_label = category.replace("_", " ") if category else "entity"
        return f"Low-risk {cat_label}; retained as-is for utility."

    base = _REASON_TEMPLATES.get((category, method))
    if base is None:
        cat_label = category.replace("_", " ") if category else "entity"
        method_label = method or "an appropriate method"
        base = f"{cat_label} — protected via {method_label}"

    prefix = _SENSITIVITY_PREFIX.get(sensitivity, "")
    reason = (prefix + base).strip()
    # Capitalize first letter; template shapes already make this ≥10 chars.
    return reason[:1].upper() + reason[1:] if reason else "Protection applied per policy."


# ---------------------------------------------------------------------------
# Entity-context flattening
# ---------------------------------------------------------------------------


def _coerce_entity_list(raw: object) -> list[dict]:
    """DataDesigner hands context columns to custom generators in several
    shapes: a pydantic-dump dict with a keyed list, a raw list, a JSON-
    encoded string, or None. Normalize to a plain list of dicts.
    """
    import json
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    if isinstance(raw, dict):
        # pydantic dump of a wrapper schema like EntitiesByValueSchema or
        # LatentEntitiesSchema — the inner list lives under one of these keys.
        for key in ("entities_by_value", "latent_entities", "entities", "items"):
            if key in raw and isinstance(raw[key], list):
                raw = raw[key]
                break
        else:
            return []
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(item)
        elif isinstance(item, str):
            # JSON-string-per-item (rare but seen).
            try:
                parsed = json.loads(item)
                if isinstance(parsed, dict):
                    out.append(parsed)
            except Exception:
                continue
    return out


def _flatten_context(
    entities_by_value: object,
    latent_entities: object,
) -> list[dict]:
    """Produce a flat, ordered list of {source, entity_label, entity_value}.

    Order matches how the disposition prompt enumerates entities:
    tagged entries from entities_by_value (one per (value, label) pair)
    followed by latent entries. The returned list index+1 is the expected id.
    """
    flat: list[dict] = []
    for ev in _coerce_entity_list(entities_by_value):
        value = ev.get("value", "")
        labels = ev.get("labels") or []
        if not labels:
            flat.append({"source": "tagged", "entity_label": "", "entity_value": value})
            continue
        for label in labels:
            flat.append({"source": "tagged", "entity_label": label, "entity_value": value})
    for le in _coerce_entity_list(latent_entities):
        flat.append({
            "source": "latent",
            "entity_label": le.get("label", ""),
            "entity_value": le.get("value", ""),
        })
    return flat


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


def reconstruct_full_disposition(
    simple: SimpleDispositionResult,
    entities_by_value: object = None,
    latent_entities: object = None,
) -> SensitivityDispositionSchema:
    """Build the strict disposition from the loose LLM output + context columns.

    For each SimpleDispositionItem:
      - prefer the model-echoed source/entity_label/entity_value; fall back
        to the id-indexed context lookup if the echo is missing or empty.
      - derive needs_protection from method.
      - keep the LLM protection_reason if it stripped to ≥10 chars, else
        template one from (category, method, sensitivity).

    Orphan simple items (id outside the context range AND no usable echoes)
    are skipped with a warning — better to return a smaller valid schema
    than to drop the whole record.
    Duplicate ids are de-duplicated (first occurrence wins).
    """
    context = _flatten_context(entities_by_value, latent_entities)
    seen_ids: set[int] = set()
    full_items: list[EntityDispositionSchema] = []

    for item in simple.sensitivity_disposition:
        if item.id in seen_ids:
            logger.warning(
                "reconstruct_full_disposition: duplicate id=%s in simple output; keeping first occurrence",
                item.id,
            )
            continue
        seen_ids.add(item.id)

        # Resolve (source, entity_label, entity_value). Context is the
        # AUTHORITATIVE source when the id falls in range — small models
        # (gemma4-e2b) routinely echo garbage in these fields (e.g. the
        # entity_label in the source slot), so trusting the echo there
        # corrupts the strict schema. Fall back to the LLM echo only when
        # there is no context entry for this id (orphan).
        idx = item.id - 1
        if 0 <= idx < len(context):
            ctx = context[idx]
            src = ctx["source"]
            lbl = ctx["entity_label"]
            val = ctx["entity_value"]
        else:
            # Orphan path: id has no context entry. The LLM echoes are the
            # only source of truth, but they may be drifted (gemma4-e4b
            # observed emitting prompt section names in source). Validate
            # the source enum and skip the item if both source and labels
            # are unusable — a skipped orphan is better than a ValidationError
            # that drops the whole record.
            echoed_src = (item.source or "").strip().lower()
            src = echoed_src if echoed_src in {"tagged", "latent"} else ""
            lbl = item.entity_label or ""
            val = item.entity_value or ""

        if not src or not lbl or not val:
            logger.warning(
                "reconstruct_full_disposition: orphan simple item id=%s "
                "(missing or drifted source/label/value, out of context range); skipping",
                item.id,
            )
            continue

        # Default empty LLM-drift slots to sane values so the strict schema
        # doesn't reject the row. category/sensitivity are enums at the
        # internal layer; empty strings would fail.
        category = _normalize_category(item.category, entity_label=lbl)
        sensitivity = (item.sensitivity or "").strip().lower() or "medium"

        # Derive method. When the model omits it, default pessimistically
        # for high-risk entities so a direct_identifier with sensitivity=high
        # never silently slips through as leave_as_is.
        raw_method = (item.protection_method_suggestion or "").strip()
        if raw_method:
            method = raw_method
        elif category in ("direct_identifier", "sensitive_attribute") or sensitivity in ("medium", "high"):
            method = "replace"
        else:
            method = "leave_as_is"
        needs = derive_needs_protection(method)

        # Keep LLM reason if usable, else template.
        raw_reason = (item.protection_reason or "").strip()
        reason = raw_reason if len(raw_reason) >= 10 else template_protection_reason(
            category, method, sensitivity
        )

        full_items.append(
            EntityDispositionSchema(
                id=item.id,
                source=src,
                category=category,       # strict schema coerces via its before-validator
                sensitivity=sensitivity,
                entity_label=lbl,
                entity_value=val,
                needs_protection=needs,
                protection_method_suggestion=method,
                protection_reason=reason,
            )
        )

    return SensitivityDispositionSchema(sensitivity_disposition=full_items)
