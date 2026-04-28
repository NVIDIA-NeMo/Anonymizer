# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

_logger = logging.getLogger(__name__)


def _parse_raw_wrapper(
    model_cls: type[T],
    raw: object,
    key: str,
    fallback_keys: tuple[str, ...] = (),
) -> T:
    """Parse raw DataFrame cell value into a wrapper schema."""

    def _safe_validate(candidate_list: list[object]) -> T:
        try:
            return model_cls.model_validate({key: candidate_list})
        except ValidationError:
            return model_cls()

    if isinstance(raw, model_cls):
        return raw
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return model_cls()
    if isinstance(raw, BaseModel):
        raw = raw.model_dump(mode="python")

    if isinstance(raw, dict):
        candidate = raw.get(key)
        if candidate is None:
            for fk in fallback_keys:
                candidate = raw.get(fk)
                if candidate is not None:
                    break
        if isinstance(candidate, model_cls):
            return candidate
        if isinstance(candidate, BaseModel):
            candidate = candidate.model_dump(mode="python")
        if isinstance(candidate, list):
            return _safe_validate(candidate)
        if hasattr(candidate, "tolist"):
            as_list = candidate.tolist()
            if isinstance(as_list, list):
                return _safe_validate(as_list)
    return model_cls()


# ---------------------------------------------------------------------------
# Loose-list-wrapper helpers
#
# Used by wire-shape schemas whose LLM emitters sometimes drop the wrapper
# key, returning a bare list at the top level (observed: nemotron-3-nano:4b
# on rewrite-mode disposition; qwen3.5:4b on legal-court meaning units).
# Two helpers, one for the JSON Schema widening (consumed by DataDesigner's
# pre-validate gate) and one for the runtime before-validator.
# ---------------------------------------------------------------------------


def loose_list_wrapper_json_schema(handler, schema, *, list_field: str) -> dict:
    """Widen a wrapper-style pydantic schema to ``oneOf({wrapper}, {array})``.

    DataDesigner runs ``jsonschema.validate()`` on raw LLM output BEFORE
    pydantic's before-validators run. If a small model returns
    ``[item, ...]`` instead of ``{list_field: [item, ...]}``, the strict
    ``type: object`` pre-check rejects the row and the record is dropped.
    Widening to a ``oneOf`` of the wrapper-object and the bare-array shape
    lets both pass the pre-check; the runtime ``accept_bare_list``
    validator then normalizes to the canonical wrapper form.

    Falls back gracefully to the unwidened wrapper if pydantic ever
    restructures so the inline schema for ``list_field`` is not directly
    accessible (e.g. moves behind a ``$ref``). Logs a warning so future
    regressions are visible at runtime.
    """
    wrapped = handler(schema)
    items = wrapped.get("properties", {}).get(list_field)
    # Degrade gracefully if the property is missing entirely OR if it's
    # only a $ref pointer (a future pydantic refactor moving the inline
    # array schema behind $defs would do this — we can't safely use a
    # ref pointer as the standalone bare-list branch of oneOf because
    # DD's jsonschema gate would resolve it against the wrapper's $defs
    # and the semantics get murky).
    if not isinstance(items, dict) or set(items.keys()) <= {"$ref"}:
        _logger.warning(
            "loose_list_wrapper_json_schema: inline schema for %r unavailable in %r "
            "(items=%r); skipping oneOf widening (DD pre-validate may reject bare-list shape)",
            list_field,
            wrapped.get("title"),
            items,
        )
        return wrapped
    return {"oneOf": [wrapped, items]}


def accept_bare_list(*, list_field: str):
    """Build a ``mode="before"`` validator that wraps a top-level bare list.

    Returns a ``classmethod`` suitable as a ``model_validator(mode="before")``
    callable: maps ``[item, ...]`` -> ``{list_field: [item, ...]}`` and
    passes anything else through unchanged.
    """

    def _wrap(cls, data):
        if isinstance(data, list):
            return {list_field: data}
        return data

    return classmethod(_wrap)
