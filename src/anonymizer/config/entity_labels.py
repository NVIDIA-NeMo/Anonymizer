# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entity-label normalization and the merge of built-in and configured detection examples."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from anonymizer.engine.constants import ENTITY_LABEL_EXAMPLES


def normalize_entity_label(label: str) -> str:
    """Return the canonical identity used for entity-label comparisons."""
    return label.strip().casefold()


def is_reserved_entity_label(label: str) -> bool:
    """Return whether a label is reserved for internal prompt markup."""
    return re.fullmatch(r"nagroup_g[1-9][0-9]*", normalize_entity_label(label)) is not None


def reject_reserved_entity_label(label: str) -> str:
    """Return ``label``, or raise when it is reserved for internal prompt markup."""
    if is_reserved_entity_label(label):
        raise ValueError(f"Entity label {label!r} is reserved for internal prompt markup.")
    return label


def normalize_entity_label_examples(
    entity_label_examples: Mapping[Any, Any],
) -> tuple[dict[str, list[str]], list[str], list[str]]:
    """Normalize, merge and dedupe a raw ``entity_label_examples`` mapping, raising on invalid entries.

    Returns the normalized mapping, the normalized labels that were merged as duplicates, and the labels that had
    duplicate example values removed.
    """
    normalized: dict[str, list[str]] = {}
    duplicate_keys: list[str] = []
    duplicate_value_labels: list[str] = []
    for raw_label, raw_examples in entity_label_examples.items():
        if not isinstance(raw_label, str) or not normalize_entity_label(raw_label):
            raise ValueError("Labels in entity_label_examples must be non-blank strings.")
        label = reject_reserved_entity_label(normalize_entity_label(raw_label))
        if not isinstance(raw_examples, list) or not raw_examples:
            raise ValueError(f"entity_label_examples[{label!r}] must be a non-empty list of strings.")

        if label in normalized:
            duplicate_keys.append(label)
        examples = normalized.setdefault(label, [])
        for index, raw_example in enumerate(raw_examples):
            if not isinstance(raw_example, str) or not raw_example.strip():
                raise ValueError(f"entity_label_examples[{label!r}][{index}] must be a non-blank string.")
            example = raw_example.strip()
            if example in examples:
                duplicate_value_labels.append(label)
                continue
            examples.append(example)

    return normalized, sorted(set(duplicate_keys)), sorted(set(duplicate_value_labels))


def merge_entity_label_examples(
    labels: Sequence[str], configured: Mapping[str, Sequence[str]]
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    """Return the validator and augmenter examples for ``labels``, in label order.

    The validator gets each label's built-in examples followed by its configured ones; the augmenter gets only the
    configured ones, for the labels that have them. Configured labels outside ``labels`` are ignored.
    """
    validator: dict[str, tuple[str, ...]] = {}
    for label in labels:
        examples = list(ENTITY_LABEL_EXAMPLES.get(label, []))
        for example in configured.get(label, ()):
            if example not in examples:
                examples.append(example)
        validator[label] = tuple(examples)
    augmenter = {label: tuple(configured[label]) for label in labels if label in configured}
    return validator, augmenter
