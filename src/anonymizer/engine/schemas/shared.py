# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Protocol, TypeGuard, TypeVar, cast

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class _ListConvertible(Protocol):
    def tolist(self) -> object: ...


def _is_list_convertible(value: object) -> TypeGuard[_ListConvertible]:
    return callable(getattr(value, "tolist", None))


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
            return _safe_validate(cast(list[object], candidate))
        if _is_list_convertible(candidate):
            as_list = candidate.tolist()
            if isinstance(as_list, list):
                return _safe_validate(cast(list[object], as_list))
    return model_cls()


def to_python(raw: object) -> object:
    """Recursively normalize nested model/array/scalar payloads to plain Python."""
    if isinstance(raw, BaseModel):
        return to_python(raw.model_dump(mode="python"))
    if isinstance(raw, dict):
        return {key: to_python(value) for key, value in raw.items()}
    if isinstance(raw, (list, tuple)):
        return [to_python(value) for value in raw]
    if _is_list_convertible(raw):
        return to_python(raw.tolist())
    item = getattr(raw, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    return raw
