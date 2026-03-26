# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


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
