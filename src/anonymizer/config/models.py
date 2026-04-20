# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, field_validator


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
        """Accept either a scalar alias or a list, return a non-empty list.

        Normalizing at parse time keeps every downstream consumer on the
        same shape (``list[str]``) regardless of whether the user wrote
        ``entity_validator: some-alias`` or
        ``entity_validator: [alias-a, alias-b]``.
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
        return cleaned


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
