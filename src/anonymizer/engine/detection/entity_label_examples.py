# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run-local resolution of entity labels and positive detection examples."""

from __future__ import annotations

from dataclasses import dataclass

from anonymizer.engine.constants import DEFAULT_ENTITY_LABELS, ENTITY_LABEL_EXAMPLES
from anonymizer.engine.detection.postprocess import normalize_label, normalize_labels


@dataclass
class ResolvedEntityOntology:
    """Effective detection labels and stage-specific example mappings."""

    labels: list[str]
    validator_examples: dict[str, list[str]]
    augmenter_examples: dict[str, list[str]]
    strict_labels: bool


def normalize_entity_label_examples(
    entity_label_examples: dict[str, list[str]] | None,
) -> tuple[dict[str, list[str]], list[str], list[str]]:
    """Normalize, merge-dedupe, and validate a raw ``entity_label_examples`` mapping.

    Single source of truth for these rules, shared by ``Detect``'s pydantic
    validation (`anonymizer.config.anonymizer_config`), which reports the
    returned duplicate info as warnings, and :func:`resolve_entity_ontology`,
    which validates direct engine callers (e.g. distributed export paths) that
    bypass ``Detect``. Returns a fresh, defensively copied mapping plus the
    normalized keys that were merged as duplicates and the labels that had
    duplicate example values removed.
    """
    normalized: dict[str, list[str]] = {}
    duplicate_keys: list[str] = []
    duplicate_value_labels: list[str] = []
    for raw_label, raw_examples in (entity_label_examples or {}).items():
        if not isinstance(raw_label, str):
            raise ValueError("entity_label_examples keys must be non-blank strings.")
        label = normalize_label(raw_label)
        if not label:
            raise ValueError("entity_label_examples keys must be non-blank strings.")
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


def resolve_entity_ontology(
    *,
    entity_labels: list[str] | None,
    excluded_entity_labels: list[str] | set[str] | None = None,
    entity_label_examples: dict[str, list[str]] | None = None,
) -> ResolvedEntityOntology:
    """Resolve effective labels and fresh stage-specific example mappings."""
    strict_labels = entity_labels is not None
    labels = list(DEFAULT_ENTITY_LABELS) if entity_labels is None else list(entity_labels)
    excluded = normalize_labels(excluded_entity_labels)
    configured, _, _ = normalize_entity_label_examples(entity_label_examples)

    normalized_labels: list[str] = []
    seen_labels: set[str] = set()
    for raw_label in labels:
        label = normalize_label(raw_label)
        if label and label not in seen_labels and label not in excluded:
            normalized_labels.append(label)
            seen_labels.add(label)

    active_configured = {
        label: list(examples)
        for label, examples in configured.items()
        if label not in excluded
    }
    if strict_labels:
        unknown = sorted(set(active_configured) - seen_labels)
        if unknown:
            raise ValueError(
                "entity_label_examples contains labels outside the explicit entity_labels allowlist: "
                f"{unknown}"
            )
    else:
        for label in active_configured:
            if label not in seen_labels:
                normalized_labels.append(label)
                seen_labels.add(label)

    if not normalized_labels:
        raise ValueError("The effective detection label set is empty.")

    validator_examples: dict[str, list[str]] = {}
    for label in normalized_labels:
        examples = list(ENTITY_LABEL_EXAMPLES.get(label, []))
        for example in active_configured.get(label, []):
            if example not in examples:
                examples.append(example)
        validator_examples[label] = examples

    return ResolvedEntityOntology(
        labels=normalized_labels,
        validator_examples=validator_examples,
        augmenter_examples={
            label: list(active_configured[label])
            for label in normalized_labels
            if label in active_configured
        },
        strict_labels=strict_labels,
    )
