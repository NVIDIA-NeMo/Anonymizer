# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import TypeVar, cast

from pydantic import BaseModel, ValidationError

from anonymizer.engine.cells import is_list_convertible as _is_list_convertible
from anonymizer.engine.cells import normalize_payload as normalize_payload
from anonymizer.engine.cells import to_python as to_python

T = TypeVar("T", bound=BaseModel)


def _parse_raw_wrapper(
    model_cls: type[T],
    raw: object,
    key: str,
    fallback_keys: tuple[str, ...] = (),
    *,
    strict: bool = False,
) -> T:
    """Parse raw DataFrame cell value into a wrapper schema.

    Lenient by default: malformed values parse as an empty model. With ``strict=True``,
    a malformed non-empty value raises ``ValueError`` (message never includes the value).
    """

    def _empty(cause: str) -> T:
        if strict and not _is_empty(raw):
            raise ValueError(cause)
        return model_cls()

    def _safe_validate(candidate_list: list[object]) -> T:
        try:
            return model_cls.model_validate({key: candidate_list})
        except ValidationError:
            if strict:
                raise ValueError("payload failed schema validation") from None
            return model_cls()

    if isinstance(raw, model_cls):
        return raw
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return _empty("payload is not valid JSON")
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
            return _safe_validate(cast(list[object], candidate))
        if _is_list_convertible(candidate):
            as_list = candidate.tolist()
            if isinstance(as_list, list):
                return _safe_validate(cast(list[object], as_list))
    return _empty("payload has an unexpected shape")


def _is_empty(raw: object) -> bool:
    # Kept apart from ``cells.is_missing`` on purpose: an empty string, dict or list is an empty payload here.
    if raw is None:
        return True
    if isinstance(raw, (str, dict, list)):
        return not raw
    return isinstance(raw, float) and raw != raw


def read_cell(model: type[T], raw: object, *, error: type[Exception] = TypeError) -> T:
    """Read one nested cell (model, dict, JSON string or parquet-reloaded value) as *model*.

    A model instance is dumped and revalidated, not passed through. Raises *error*
    when the cell does not normalize to a dict.
    """
    payload = normalize_payload(raw)
    if not isinstance(payload, dict):
        raise error(f"Expected {model.__name__} or dict, got {type(payload).__name__}")
    return model.model_validate(payload)
