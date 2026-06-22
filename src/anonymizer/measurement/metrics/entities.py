# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any, cast

from anonymizer.measurement._coerce import _coerce_float, _coerce_payload, _f1, _safe_ratio

_GROUND_TRUTH_ENTITY_COLUMNS = ("ground_truth_entities", "gt_entities", "expected_entities")
_ENTITY_LABEL_EQUIVALENCE_CLASSES = (
    frozenset(
        {
            "access_token",
            "api_key",
            "auth_token",
            "bearer_token",
            "password",
            "secret_key",
            "session_id",
            "unique_id",
            "user_id",
        }
    ),
    frozenset({"full_name", "person_name", "user", "user_name", "username"}),
    frozenset({"phone", "phone_number", "telephone"}),
    frozenset({"email", "email_address"}),
    frozenset({"cookie", "http_cookie", "session_cookie"}),
)
_ENTITY_LABEL_EQUIVALENCE: dict[str, str] = {
    label: sorted(labels)[0] for labels in _ENTITY_LABEL_EQUIVALENCE_CLASSES for label in labels
}


def _entities_from_raw(raw: object) -> list[dict[str, Any]]:
    payload = _coerce_payload(raw)
    if isinstance(payload, Mapping):
        items = cast(Mapping[str, Any], payload).get("entities", [])
    elif isinstance(payload, list):
        items = payload
    else:
        items = []
    return [dict(cast(Mapping[str, Any], item)) for item in items if isinstance(item, Mapping)]


def _entity_ground_truth_metrics(
    final_entities: list[dict[str, Any]],
    ground_truth_entities: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    if ground_truth_entities is None:
        return {
            "ground_truth_entity_count": None,
            "ground_truth_entity_label_counts": None,
            "entity_true_positive_count": None,
            "entity_false_positive_count": None,
            "entity_false_negative_count": None,
            "entity_precision": None,
            "entity_recall": None,
            "entity_f1": None,
            "entity_relaxed_gt_found_count": None,
            "entity_relaxed_detected_tp_count": None,
            "entity_relaxed_label_compatible_gt_found_count": None,
            "entity_relaxed_label_compatible_detected_tp_count": None,
            "entity_relaxed_precision": None,
            "entity_relaxed_recall": None,
            "entity_relaxed_f1": None,
            "entity_relaxed_label_compatible_precision": None,
            "entity_relaxed_label_compatible_recall": None,
            "entity_relaxed_label_compatible_f1": None,
        }

    predicted = _entity_identity_counts(final_entities)
    expected = _entity_identity_counts(ground_truth_entities)
    true_positive = sum((predicted & expected).values())
    false_positive = sum((predicted - expected).values())
    false_negative = sum((expected - predicted).values())
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    return {
        "ground_truth_entity_count": len(ground_truth_entities),
        "ground_truth_entity_label_counts": dict(
            sorted(Counter(e.get("label", "") for e in ground_truth_entities if e.get("label")).items())
        ),
        "entity_true_positive_count": true_positive,
        "entity_false_positive_count": false_positive,
        "entity_false_negative_count": false_negative,
        "entity_precision": precision,
        "entity_recall": recall,
        "entity_f1": _f1(precision, recall),
        **_entity_relaxed_ground_truth_metrics(final_entities, ground_truth_entities),
    }


def _entity_identity_counts(entities: list[dict[str, Any]]) -> Counter[tuple[str, str]]:
    identities: Counter[tuple[str, str]] = Counter()
    for entity in entities:
        label = entity.get("label")
        value = entity.get("value")
        if label is None or value is None:
            continue
        identities[(str(value), str(label))] += 1
    return identities


def _entity_relaxed_ground_truth_metrics(
    final_entities: list[dict[str, Any]],
    ground_truth_entities: list[dict[str, Any]],
) -> dict[str, Any]:
    relaxed_match_count = _relaxed_entity_match_count(final_entities, ground_truth_entities)
    label_compatible_match_count = _relaxed_entity_match_count(
        final_entities,
        ground_truth_entities,
        require_label_compatible=True,
    )
    gt_found = relaxed_match_count
    detected_tp = relaxed_match_count
    label_compatible_gt_found = label_compatible_match_count
    label_compatible_detected_tp = label_compatible_match_count
    precision = _safe_ratio(detected_tp, len(final_entities))
    recall = _safe_ratio(gt_found, len(ground_truth_entities))
    label_compatible_precision = _safe_ratio(label_compatible_detected_tp, len(final_entities))
    label_compatible_recall = _safe_ratio(label_compatible_gt_found, len(ground_truth_entities))
    return {
        "entity_relaxed_gt_found_count": gt_found,
        "entity_relaxed_detected_tp_count": detected_tp,
        "entity_relaxed_label_compatible_gt_found_count": label_compatible_gt_found,
        "entity_relaxed_label_compatible_detected_tp_count": label_compatible_detected_tp,
        "entity_relaxed_precision": precision,
        "entity_relaxed_recall": recall,
        "entity_relaxed_f1": _f1(precision, recall),
        "entity_relaxed_label_compatible_precision": label_compatible_precision,
        "entity_relaxed_label_compatible_recall": label_compatible_recall,
        "entity_relaxed_label_compatible_f1": _f1(label_compatible_precision, label_compatible_recall),
    }


def _relaxed_entity_match_count(
    final_entities: list[dict[str, Any]],
    ground_truth_entities: list[dict[str, Any]],
    *,
    require_label_compatible: bool = False,
) -> int:
    matches_by_ground_truth = [
        [
            final_index
            for final_index, final_entity in enumerate(final_entities)
            if _entities_match_relaxed(
                final_entity,
                ground_truth_entity,
                require_label_compatible=require_label_compatible,
            )
        ]
        for ground_truth_entity in ground_truth_entities
    ]
    matched_ground_truth_by_final: dict[int, int] = {}

    def assign(ground_truth_index: int, seen: set[int]) -> bool:
        for final_index in matches_by_ground_truth[ground_truth_index]:
            if final_index in seen:
                continue
            seen.add(final_index)
            if final_index not in matched_ground_truth_by_final or assign(
                matched_ground_truth_by_final[final_index],
                seen,
            ):
                matched_ground_truth_by_final[final_index] = ground_truth_index
                return True
        return False

    return sum(1 for ground_truth_index in range(len(ground_truth_entities)) if assign(ground_truth_index, set()))


def _entities_match_relaxed(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    require_label_compatible: bool,
) -> bool:
    if require_label_compatible and not _entity_labels_compatible(left.get("label"), right.get("label")):
        return False
    left_span = _entity_span(left)
    right_span = _entity_span(right)
    if left_span is not None and right_span is not None:
        return left_span[0] < right_span[1] and right_span[0] < left_span[1]
    left_value = left.get("value")
    right_value = right.get("value")
    return left_value is not None and right_value is not None and str(left_value) == str(right_value)


def _entity_span(entity: dict[str, Any]) -> tuple[int, int] | None:
    start = _coerce_float(entity.get("start_position", entity.get("start")))
    end = _coerce_float(entity.get("end_position", entity.get("end")))
    if start is None or end is None:
        return None
    start_int = int(start)
    end_int = int(end)
    if start_int < 0 or end_int <= start_int:
        return None
    return start_int, end_int


def _entity_labels_compatible(left: object, right: object) -> bool:
    left_key = _entity_label_key(left)
    right_key = _entity_label_key(right)
    return left_key is not None and right_key is not None and left_key == right_key


def _entity_label_key(label: object) -> str | None:
    if label is None:
        return None
    normalized = str(label).strip().lower()
    if not normalized:
        return None
    return _ENTITY_LABEL_EQUIVALENCE.get(normalized, normalized)
