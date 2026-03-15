# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pydantic import BaseModel


class DetectionModelSelection(BaseModel):
    """Model aliases for the entity detection pipeline."""

    entity_detector: str
    entity_validator: str
    entity_augmenter: str
    latent_detector: str


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
