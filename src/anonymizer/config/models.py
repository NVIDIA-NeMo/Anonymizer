# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)


class DetectionModelSelection(BaseModel):
    """Model aliases for the entity detection pipeline.

    ``entity_validator`` accepts either a single alias or a list of aliases.
    A list forms a validator *pool*: chunked validation rotates calls
    across the pool in round-robin order, which is useful for bypassing
    per-alias TPM/RPM limits. A single scalar is normalized to a
    one-element list.
    """

    entity_detector: str
    entity_validator: list[str]
    entity_augmenter: str
    latent_detector: str

    @field_validator("entity_validator", mode="before")
    @classmethod
    def normalize_entity_validator(cls, value: Any) -> list[str]:
        """Accept a scalar alias, a list of aliases, or a tuple of aliases; return a non-empty deduplicated list.

        Normalizing at parse time keeps every downstream consumer on the
        same shape (``list[str]``) regardless of whether the user wrote
        ``entity_validator: some-alias`` or
        ``entity_validator: [alias-a, alias-b]``. Tuples are accepted for
        parity with Pydantic v2's default coercion for ``list[str]`` fields,
        which lets programmatic callers pass either
        ``DetectionModelSelection(entity_validator=["a", "b"])`` or
        ``DetectionModelSelection(entity_validator=("a", "b"))`` without
        caring about the concrete sequence type. Any other input type
        raises ``TypeError``.

        Duplicate aliases are collapsed to the first occurrence (order
        preserved) and a warning is logged. A duplicate in the pool would
        burn a failover attempt on an already-exhausted endpoint, which
        almost certainly isn't what the user wants.
        """
        if isinstance(value, str):
            aliases: list[str] = [value]
        elif isinstance(value, (list, tuple)):
            aliases = [str(item) for item in value]
        else:
            raise TypeError(f"entity_validator must be a string or list of strings, got {type(value).__name__}")
        cleaned = [alias.strip() for alias in aliases if alias.strip()]
        if not cleaned:
            raise ValueError("entity_validator must name at least one model alias.")
        seen: set[str] = set()
        deduped: list[str] = []
        for alias in cleaned:
            if alias in seen:
                continue
            seen.add(alias)
            deduped.append(alias)
        if len(deduped) != len(cleaned):
            removed = [alias for alias in cleaned if cleaned.count(alias) > 1]
            logger.warning(
                "entity_validator pool contained duplicate aliases %s; collapsing to %s. "
                "Duplicates burn a failover attempt on an already-exhausted endpoint.",
                sorted(set(removed)),
                deduped,
            )
        return deduped


class ReplaceModelSelection(BaseModel):
    """Model aliases for the replacement pipeline."""

    replacement_generator: str


class RewriteModelSelection(BaseModel):
    """Model aliases for the rewrite pipeline."""

    domain_classifier: str
    disposition_analyzer: str
    meaning_extractor: str
    qa_generator: str
    rewriter: str
    evaluator: str
    repairer: str
    judge: str


class ModelSelection(BaseModel):
    """Model alias selections for all pipelines, loaded from YAML defaults via ``load_default_model_selection()``."""

    detection: DetectionModelSelection
    replace: ReplaceModelSelection
    rewrite: RewriteModelSelection
