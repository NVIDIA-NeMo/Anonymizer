# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Server-side reconstruction of the strict EntityDispositionSchema from the
loose wire-contract SimpleDispositionResult + the per-entity context columns.

The disposition_analyzer LLM emits a minimal ``SimpleDispositionResult``
(8 optional/loose fields per item). This module rebuilds the strict form
deterministically: pair each simple item with its entity context (by id,
with entity_label/value echoes as belt-and-braces), normalize category and
method drift, derive ``combined_risk_level`` consistent with the chosen
method, and template ``protection_reason`` when the model did not provide
one.

No LLM calls; no I/O. Pure python for the reconstruction column.

Why server-side reconstruction (vs. asking the LLM to emit the strict
schema directly): DataDesigner runs ``jsonschema.validate()`` on the raw
LLM response BEFORE pydantic's before-validators get a chance to coerce.
Strict ``enum`` / ``required`` / ``minLength`` constraints on the wire
schema therefore become un-coercible gates for small-model drift, dropping
the entire record. The loose ``SimpleDispositionResult`` wire contract
(see ``schemas/rewrite.py``) lets drifted output survive that gate; this
module then rebuilds the strict form server-side.
"""

from __future__ import annotations

import logging

from anonymizer.engine.schemas.rewrite import (
    _ENTITY_LABEL_TO_CATEGORY,
    EntityDispositionSchema,
    SensitivityDispositionSchema,
    SimpleDispositionResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Derivation helpers
# ---------------------------------------------------------------------------


_VALID_CATEGORIES: frozenset[str] = frozenset({"direct_identifier", "quasi_identifier", "latent_identifier"})


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
      ``_ENTITY_LABEL_TO_CATEGORY`` and mapped to the most-likely category.
      Source label provenance is preserved by the ``source`` field on the
      strict schema.
    - **Empty / unknown** — falls back to ``"quasi_identifier"`` (the
      conservative default; pessimistic protection rather than dropping
      the row).

    Note: ``"sensitive_attribute"`` is no longer a valid EntityCategory
    (collapsed into ``quasi_identifier`` by #150), so the substring branch
    that used to map it to a dedicated category now falls through to the
    quasi_identifier fallback via the entity-label lookup.
    """
    if not isinstance(raw, str) or not raw.strip():
        return "quasi_identifier"
    normalized = raw.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in _VALID_CATEGORIES:
        return normalized
    if normalized.endswith("s") and normalized[:-1] in _VALID_CATEGORIES:
        return normalized[:-1]
    # Merged-enum hallucination: order = strongest protection wins so that
    # "latent_direct_identifier" maps to direct_identifier rather than
    # latent_identifier. ``"sensitive"`` and ``"quasi"`` both fold into
    # quasi_identifier (the conservative protect-cautiously bucket).
    for sub, target in (
        ("direct", "direct_identifier"),
        ("quasi", "quasi_identifier"),
        ("sensitive", "quasi_identifier"),
        ("latent", "latent_identifier"),
    ):
        if sub in normalized:
            return target
    # Entity-label confusion: model wrote an entity_label value in the slot.
    mapped = _ENTITY_LABEL_TO_CATEGORY.get(normalized)
    if mapped is not None:
        return mapped
    if entity_label and normalized == entity_label.strip().lower():
        return "quasi_identifier"
    return "quasi_identifier"


_VALID_METHODS: frozenset[str] = frozenset({"replace", "generalize", "remove", "suppress_inference", "leave_as_is"})


def _normalize_method(raw: str) -> str:
    """Resolve a (potentially case-drifted) protection_method_suggestion
    string into a valid ProtectionMethod enum value.

    Returns ``""`` when no recognizable choice can be extracted, signaling
    the caller should apply a pessimistic default.

    Strategy mirrors ``_normalize_category``:
      * Empty / non-string -> ``""``.
      * Exact match -> as-is.
      * Substring match in priority order ``suppress_inference ->
        leave_as_is -> generalize -> replace -> remove`` so e.g.
        ``"replace_with_surrogate"`` resolves to ``"replace"``,
        ``"leave_as_is_for_now"`` resolves to ``"leave_as_is"``.
    """
    if not raw or not isinstance(raw, str):
        return ""
    cleaned = raw.strip().lower()
    if cleaned in _VALID_METHODS:
        return cleaned
    for choice in ("suppress_inference", "leave_as_is", "generalize", "replace", "remove"):
        if choice in cleaned:
            return choice
    return ""


def derive_combined_risk_level(category: str, method: str, sensitivity: str) -> str:
    """Pick a ``CombinedRiskLevel`` consistent with ``EntityDispositionSchema._validate_protection_consistency``.

    The post-#163 invariant is:
      * ``low`` requires ``method == "leave_as_is"``
      * ``high`` requires ``method != "leave_as_is"``
      * ``medium`` is permissive (either method)

    We always have ``method`` already (either echoed by the LLM or derived
    pessimistically by the reconstructor), so we pick the risk level that
    *passes the invariant* AND best reflects the inputs:

      * ``method == "leave_as_is"`` -> ``"low"``. This matches the spirit
        of leaving a value alone: the model judged the row low risk, or
        the entity is too utility-critical to mask.
      * ``method != "leave_as_is"`` -> ``"high"`` if the entity is a direct
        identifier of any sensitivity, or has ``sensitivity == "high"``,
        else ``"medium"``. Direct identifiers are always high re-id risk
        on their own; explicit ``high`` sensitivity from the LLM is
        respected; otherwise we don't claim high without evidence.

    Returning ``"medium"`` rather than ``"high"`` for borderline cases
    keeps the strict schema's leave_as_is_when_low rule from accidentally
    constraining downstream re-validation if the disposition is
    re-evaluated upstream.
    """
    method = (method or "").strip()
    if method == "leave_as_is":
        return "low"
    sens = (sensitivity or "").strip().lower()
    cat = (category or "").strip().lower()
    if cat == "direct_identifier":
        return "high"
    if sens == "high":
        return "high"
    return "medium"


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
    ("latent_identifier", "suppress_inference"): "latent inference — suppressed to prevent deduction",
    ("latent_identifier", "remove"): "latent identifier — removed to prevent inference",
    ("latent_identifier", "generalize"): "latent identifier — generalized to reduce inference",
    ("latent_identifier", "replace"): "latent identifier — replaced with a less specific surrogate",
}

_SENSITIVITY_PREFIX = {"low": "", "medium": "moderate-risk ", "high": "high-risk "}

# Upper bound for a passthrough (model-authored) protection_reason. The schema
# no longer enforces max_length, so the reconstructor caps length here to keep
# rewrite prompts and parquet bounded without ever failing validation.
_MAX_PROTECTION_REASON_CHARS = 500


def template_protection_reason(category: str, method: str, sensitivity: str) -> str:
    """Build a reason string guaranteed >=10 chars.

    Used when the LLM omits or emits a too-short ``protection_reason``
    (the schema no longer enforces a min_length; this keeps a sensible
    human-readable floor for the rewrite-context line).
    Strong models that provide their own document-specific reason have
    theirs kept verbatim by the reconstructor.
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
        raw_dict: dict = raw
        for key in ("entities_by_value", "latent_entities", "entities", "items"):
            inner = raw_dict.get(key)
            if isinstance(inner, list):
                raw = inner
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
            try:
                parsed = json.loads(item)
                if isinstance(parsed, dict):
                    out.append(parsed)
            except Exception:
                continue
    return out


def _flatten_context(entities_by_value: object, latent_entities: object) -> list[dict]:
    """Produce a flat, ordered list of ``{source, entity_label, entity_value}``.

    Order matches how the disposition prompt enumerates entities: tagged
    entries from ``entities_by_value`` (one slot per ``(value, label)``
    pair) followed by latent entries. The returned list index+1 is the
    expected id.
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
        flat.append(
            {
                "source": "latent",
                "entity_label": le.get("label", ""),
                "entity_value": le.get("value", ""),
            }
        )
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

    For each ``SimpleDispositionItem``:
      - prefer the model-echoed ``source/entity_label/entity_value`` only
        when the id falls outside the context range (orphan path); use
        the trusted context entry when in range, since small models
        routinely echo garbage.
      - normalize ``category`` and ``method`` drift; pessimistically
        default ``method`` to ``replace`` for high-risk entities when the
        LLM omits it.
      - derive ``combined_risk_level`` from category/sensitivity/method
        such that ``EntityDispositionSchema._validate_protection_consistency``
        passes (``leave_as_is`` -> low; otherwise medium/high based on
        category and sensitivity). ``needs_protection`` is the strict
        schema's ``@property`` and falls out of method automatically.
      - keep the LLM ``protection_reason`` if it stripped to >=10 chars,
        else template one from (category, method, sensitivity).

    Orphan simple items (id outside the context range AND no usable
    echoes) are skipped with a warning — better to return a smaller valid
    schema than to drop the whole record. Duplicate ids are de-duplicated
    (first occurrence wins).

    Raises ``ValidationError`` only when ``full_items`` is empty (the
    strict schema requires ``min_length=1``); the workflow column wraps
    this case with a try/except and emits an empty disposition rather
    than dropping the row.
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

        idx = item.id - 1
        if 0 <= idx < len(context):
            ctx = context[idx]
            src = ctx["source"]
            lbl = ctx["entity_label"]
            val = ctx["entity_value"]
        else:
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

        category = _normalize_category(item.category, entity_label=lbl)
        sensitivity = (item.sensitivity or "").strip().lower() or "medium"

        raw_method = (item.protection_method_suggestion or "").strip().lower()
        method = _normalize_method(raw_method)
        if not method:
            # Pessimistic default for omitted method: high-risk entities
            # default to "replace"; everything else to "leave_as_is".
            if category == "direct_identifier" or sensitivity in ("medium", "high"):
                method = "replace"
            else:
                method = "leave_as_is"

        combined_risk = derive_combined_risk_level(category, method, sensitivity)

        raw_reason = (item.protection_reason or "").strip()
        if len(raw_reason) >= 10:
            # Passthrough from the model: cap length here (the schema no longer
            # enforces max_length, so a rambling small-model reason would
            # otherwise flow unbounded into the rewrite prompt and parquet).
            reason = (
                raw_reason[: _MAX_PROTECTION_REASON_CHARS - 3].rstrip() + "..."
                if (len(raw_reason) > _MAX_PROTECTION_REASON_CHARS)
                else raw_reason
            )
        else:
            reason = template_protection_reason(category, method, sensitivity)

        full_items.append(
            EntityDispositionSchema(
                id=item.id,
                source=src,
                category=category,
                sensitivity=sensitivity,
                entity_label=lbl,
                entity_value=val,
                protection_method_suggestion=method,
                combined_risk_level=combined_risk,
                protection_reason=reason,
            )
        )

    return SensitivityDispositionSchema(sensitivity_disposition=full_items)
