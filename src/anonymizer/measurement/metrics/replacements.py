# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any, cast

from anonymizer.measurement._coerce import _coerce_payload


def _replacement_map_metrics(raw: object) -> dict[str, Any]:
    replacement_maps = _replacement_maps_from_raw(raw)
    synthetic_values = []
    for item in replacement_maps:
        synthetic = item.get("replacement", item.get("synthetic"))
        if synthetic is not None:
            synthetic_values.append(str(synthetic))
    return {
        "replacement_count": len(replacement_maps),
        "replacement_label_counts": dict(
            sorted(Counter(item.get("label", "") for item in replacement_maps if item.get("label")).items())
        ),
        "replacement_duplicate_value_count": max(0, len(synthetic_values) - len(set(synthetic_values))),
    }


def _replacement_coverage_metrics(raw: object, final_entities: list[dict[str, Any]]) -> dict[str, Any]:
    replacement_original_values = {
        str(original)
        for item in _replacement_maps_from_raw(raw)
        if (original := item.get("original")) is not None and str(original)
    }
    missing_entities = [
        entity
        for entity in final_entities
        if entity.get("value") and str(entity.get("value")) not in replacement_original_values
    ]
    missing_values = {str(entity.get("value")) for entity in missing_entities if entity.get("value")}
    return {
        "replacement_missing_final_entity_count": len(missing_entities),
        "replacement_missing_final_entity_label_counts": dict(
            sorted(
                Counter(str(entity.get("label") or "") for entity in missing_entities if entity.get("label")).items()
            )
        ),
        "replacement_missing_final_value_count": len(missing_values),
    }


def _replacement_collision_metrics(raw: object, final_entities: list[dict[str, Any]]) -> dict[str, Any]:
    synthetic_values = {
        str(synthetic)
        for item in _replacement_maps_from_raw(raw)
        if (synthetic := item.get("replacement", item.get("synthetic"))) is not None and str(synthetic)
    }
    collided_entities = [
        entity for entity in final_entities if entity.get("value") and str(entity.get("value")) in synthetic_values
    ]
    collided_values = {str(entity.get("value")) for entity in collided_entities if entity.get("value")}
    return {
        "replacement_synthetic_original_collision_count": len(collided_entities),
        "replacement_synthetic_original_collision_label_counts": dict(
            sorted(
                Counter(str(entity.get("label") or "") for entity in collided_entities if entity.get("label")).items()
            )
        ),
        "replacement_synthetic_original_collision_value_count": len(collided_values),
    }


def _replacement_maps_from_raw(raw: object) -> list[Mapping[str, Any]]:
    payload = _coerce_payload(raw)
    if isinstance(payload, Mapping):
        replacements_raw = cast(Mapping[str, Any], payload).get("replacements")
        tolist = getattr(replacements_raw, "tolist", None)
        if callable(tolist):
            replacements_raw = tolist()
        replacements = replacements_raw if isinstance(replacements_raw, list) else []
    elif isinstance(payload, list):
        replacements = payload
    else:
        replacements = []
    return [cast(Mapping[str, Any], item) for item in replacements if isinstance(item, Mapping)]
