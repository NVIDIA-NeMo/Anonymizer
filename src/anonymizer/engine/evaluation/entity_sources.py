# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve entity provenance from saved span annotations without new detection."""

from __future__ import annotations

from typing import Any

import pandas as pd

from anonymizer.engine.constants import COL_DETECTED_ENTITIES, COL_FINAL_ENTITIES
from anonymizer.engine.rewrite.parsers import normalize_payload


def entity_source(row: pd.Series, value: Any, label: Any, *, fallback: Any = None) -> str:
    """Join sources for an exact value/label pair across occurrences.

    Detector denotes GLiNER in the standard pipeline. Tagged is intentionally
    kept distinct when older assessments lack span-level provenance.
    """
    if fallback == "latent":
        return "latent"
    names = {
        "detector": "detector",
        "initial_detector": "detector",
        "augmenter": "augment",
        "augmentation": "augment",
        "latent": "latent",
    }
    sources: set[str] = set()
    for column in (COL_FINAL_ENTITIES, COL_DETECTED_ENTITIES):
        payload = normalize_payload(row.get(column))
        entities = payload.get("entities", []) if isinstance(payload, dict) else payload
        for entity in entities if isinstance(entities, list) else []:
            if not isinstance(entity, dict) or entity.get("value") != value or entity.get("label") != label:
                continue
            source = entity.get("source")
            if isinstance(source, str) and source:
                sources.add(names.get(source, source))
        if sources:
            break
    if not sources and isinstance(fallback, str) and fallback:
        sources.add(names.get(fallback, fallback))
    return ", ".join(sorted(sources)) if sources else "Unknown"
