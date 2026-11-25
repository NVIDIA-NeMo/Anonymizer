# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable


def normalize_entity_label(label: str) -> str:
    """Return the canonical identity used for entity-label comparisons: the one label fold."""
    return label.strip().casefold()


def label_keys(labels: Iterable[str] | None) -> set[str]:
    """The keys of ``labels``, dropping empty and whitespace-only ones."""
    return {key for label in labels or () if (key := normalize_entity_label(label))}


def in_label_scope(label: str, *, excluded: set[str], allowed: set[str] | None = None) -> bool:
    """Whether ``label``'s key is not ``excluded`` and, when ``allowed`` is given, is allowed. Both hold keys."""
    key = normalize_entity_label(label)
    return key not in excluded and (allowed is None or key in allowed)
