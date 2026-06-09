# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from numbers import Integral
from typing import Any, cast


def _safe_row_index(row_index: object) -> int | None:
    if isinstance(row_index, bool):
        return None
    if isinstance(row_index, Integral):
        return int(row_index)
    return None


def _count_items(raw: object, *, primary_key: str, fallback_keys: tuple[str, ...] = ()) -> int:
    payload = _coerce_payload(raw)
    if isinstance(payload, Mapping):
        payload_map = cast(Mapping[str, Any], payload)
        for key in (primary_key, *fallback_keys):
            items = payload_map.get(key)
            if isinstance(items, list):
                return len(items)
        return 0
    if isinstance(payload, list):
        return len(payload)
    return 0


def _coerce_payload(raw: object) -> object:
    model_dump = getattr(raw, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="python")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if raw is None:
        return {}
    return raw


def _coerce_int(raw: object, *, default: int) -> int:
    try:
        return int(cast(Any, raw))
    except (TypeError, ValueError):
        return default


def _coerce_float(raw: object) -> float | None:
    try:
        value = float(cast(Any, raw))
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) else value


def _coerce_bool(raw: object) -> bool | None:
    if raw is None:
        return None
    if isinstance(raw, float) and math.isnan(raw):
        return None
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
        return None
    try:
        return bool(cast(Any, raw))
    except (TypeError, ValueError):
        return None


def _safe_rate(numerator: int | float | None, elapsed_sec: float) -> float | None:
    if numerator is None or elapsed_sec <= 0:
        return None
    return float(numerator) / elapsed_sec


def _safe_ratio(numerator: int | float | None, denominator: int | float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def _size_bucket(value: int) -> str:
    if value == 0:
        return "0"
    for upper in (128, 512, 2048, 8192):
        if value < upper:
            return f"1-{upper - 1}" if upper == 128 else f"{upper // 4}-{upper - 1}"
    return "8192+"


def _count_text_tokens(text: str) -> int:
    try:
        import tiktoken

        tokenizer = tiktoken.get_encoding("cl100k_base")
        return len(tokenizer.encode(text, disallowed_special=()))
    except Exception:
        return len(text.split())


def _json_safe(value: object) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return sorted((_json_safe(v) for v in value), key=str)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
