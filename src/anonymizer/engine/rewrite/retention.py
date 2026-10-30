# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic checks on the sensitivity disposition and the rewritten text.

``disposition_coverage`` lists detected and latent entities that the disposition
does not mention; the rewrite prompt only handles listed entities, so those rows
need review. ``retained_protected`` lists entities the disposition marks for
protection whose original value still appears verbatim in the rewrite.

Matching is exact after normalization (NFKC, dash and quote variants folded,
whitespace collapsed, case-folded) and bounded by non-word characters. Values
that are common words (for example "History" or "moderate") are reported as
retained whenever the rewrite uses the same word; this false-positive class is
accepted because the signal only triggers repair and review, never data loss.

Cells are read as plain payloads (dicts, lists, numpy arrays, JSON strings);
an absent or unparseable disposition yields empty results.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

import pandas as pd

from anonymizer.engine.constants import (
    COL_DISPOSITION_MISSING_ENTITIES,
    COL_ENTITIES_BY_VALUE,
    COL_LATENT_ENTITIES,
    COL_NEEDS_HUMAN_REVIEW,
    COL_RETAINED_PROTECTED_ENTITIES,
    COL_SENSITIVITY_DISPOSITION,
)
from anonymizer.engine.rewrite.parsers import normalize_payload

logger = logging.getLogger("anonymizer.rewrite.retention")

_LEAVE_AS_IS = "leave_as_is"
_METHOD = "protection_method_suggestion"
# Dash variants (U+2010-2015, U+2212) fold to "-", single quotes to "'", double quotes to '"'.
_CHAR_MAP = str.maketrans("\u2010\u2011\u2012\u2013\u2014\u2015\u2212\u2018\u2019\u02bc\u201c\u201d", "-------'''\"\"")


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).translate(_CHAR_MAP).casefold()


def _key(value: object) -> str:
    return " ".join(_normalize(str(value or "")).split())


def _pattern(value: str) -> re.Pattern[str]:
    # Lookarounds instead of \b so values edged by punctuation ("B.S.", "C++") still match.
    return re.compile(r"(?<!\w)" + r"\s+".join(map(re.escape, _key(value).split())) + r"(?!\w)")


def _scannable(value: str) -> bool:
    """Skip one-character and punctuation-only values; "C++" and "B.S." are scanned."""
    compact = "".join(value.split())
    return len(compact) >= 2 and any(char.isalnum() for char in compact)


def _items(raw: Any, key: str) -> list[dict[str, Any]] | None:
    """Return the dict items of a wrapped (``{key: [...]}``) or bare list payload, or None if unparseable."""
    payload = normalize_payload(raw)
    if isinstance(payload, dict):
        payload = payload.get(key)
    if not isinstance(payload, list):
        return None
    return [item for item in payload if isinstance(item, dict)]


def has_items(raw: Any) -> bool:
    payload = normalize_payload(raw)
    return isinstance(payload, list) and len(payload) > 0


def disposition_coverage(entities_by_value: Any, latent_entities: Any, disposition: Any) -> list[tuple[str, str]]:
    """Return ``(value, label)`` for each entity given to the disposition step that has no disposition entry.

    Detected entities are matched on value alone (a value carrying two labels is
    one obligation); latent entities are matched on value and label.
    """
    entries = _items(disposition, "sensitivity_disposition")
    if entries is None:
        return []
    values = {_key(entry.get("entity_value")) for entry in entries}
    pairs = {(_key(entry.get("entity_value")), _key(entry.get("entity_label"))) for entry in entries}
    missing: list[tuple[str, str]] = []
    seen: set[object] = set()
    for item in _items(entities_by_value, "entities_by_value") or []:
        key = _key(item.get("value"))
        if key and key not in values and key not in seen:
            seen.add(key)
            missing.append((str(item.get("value")), str((item.get("labels") or [""])[0])))
    for item in _items(latent_entities, "latent_entities") or []:
        value, label = str(item.get("value") or ""), str(item.get("label") or "")
        pair = (_key(value), _key(label))
        if pair[0] and pair not in pairs and pair not in seen:
            seen.add(pair)
            missing.append((value, label))
    return missing


def retained_protected(disposition: Any, rewritten_text: Any) -> list[tuple[str, str, str]]:
    """Return ``(value, label, method)`` for each protected disposition entry whose value survives verbatim.

    Entries marked ``leave_as_is`` are never reported, and a kept value that
    contains a protected match exempts that match, so a kept "Florida State
    University" does not count as a retained "Florida". Matches that extend
    beyond a kept span are still reported. Values shorter than two
    non-space characters or without any alphanumeric character are skipped.
    """
    entries = _items(disposition, "sensitivity_disposition")
    if entries is None or not isinstance(rewritten_text, str):
        return []
    text = _normalize(rewritten_text)
    kept = {_key(e.get("entity_value")) for e in entries if e.get(_METHOD) == _LEAVE_AS_IS}
    protected = [
        (_pattern(str(e.get("entity_value"))), e)
        for e in entries
        if e.get(_METHOD) != _LEAVE_AS_IS and _scannable(str(e.get("entity_value") or ""))
    ]
    kept_spans = [
        (value, match.start(), match.end()) for value in kept if value for match in _pattern(value).finditer(text)
    ]
    retained: list[tuple[str, str, str]] = []
    for pattern, entry in protected:
        value, label, method = (str(entry.get(k) or "") for k in ("entity_value", "entity_label", _METHOD))
        item = (value, label, method)
        if item in retained:
            continue
        for match in pattern.finditer(text):
            if not any(
                kept_value != _key(value) and start <= match.start() and match.end() <= end
                for kept_value, start, end in kept_spans
            ):
                retained.append(item)
                break
    return retained


def apply_review_gates(df: pd.DataFrame) -> pd.DataFrame:
    """Record disposition coverage gaps and flag rows with gaps or retained values for human review.

    Runs on the final rewrite frame, after ``COL_NEEDS_HUMAN_REVIEW`` is set.
    Logs counts only.
    """
    if df.empty:
        return df
    empty = pd.Series(None, index=df.index, dtype=object)
    dispositions = df.get(COL_SENSITIVITY_DISPOSITION, empty)
    missing = [
        disposition_coverage(entities, latent, disposition)
        for entities, latent, disposition in zip(
            df.get(COL_ENTITIES_BY_VALUE, empty), df.get(COL_LATENT_ENTITIES, empty), dispositions
        )
    ]
    df[COL_DISPOSITION_MISSING_ENTITIES] = [[list(item) for item in row] for row in missing]
    retained = df.get(COL_RETAINED_PROTECTED_ENTITIES, empty)
    # Set flags in place so the column keeps its dtype (bool[pyarrow] on the combined path).
    df.loc[[bool(row) or has_items(raw) for row, raw in zip(missing, retained)], COL_NEEDS_HUMAN_REVIEW] = True

    unparseable = sum(_items(raw, "sensitivity_disposition") is None for raw in dispositions)
    if unparseable:
        logger.warning("Disposition unavailable for %d of %d row(s); checks skipped there.", unparseable, len(df))
    return df
