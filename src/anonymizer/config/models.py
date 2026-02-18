# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pydantic import BaseModel, Field


class DetectionModelSelection(BaseModel):
    """Role-level model aliases for entity detection."""

    entity_detector: str = Field(default="gliner-pii-detector")
    entity_validator: str = Field(default="build-gpt-oss-120b")
    entity_augmenter: str = Field(default="build-gpt-oss-120b")
    latent_detector: str = Field(default="nemotron-30b-thinking")


class ReplaceModelSelection(BaseModel):
    """Role-level model aliases for replacement workflows."""

    replacement_generator: str = Field(default="build-gpt-oss-120b")


class RewriteModelSelection(BaseModel):
    """Role-level model aliases for rewrite workflows."""

    rewriter: str = Field(default="build-gpt-oss-120b")
    evaluator: str = Field(default="nemotron-30b-thinking")


class ModelSelection(BaseModel):
    """Model alias selection grouped by workflow and role."""

    detection: DetectionModelSelection = Field(default_factory=DetectionModelSelection)
    replace: ReplaceModelSelection = Field(default_factory=ReplaceModelSelection)
    rewrite: RewriteModelSelection = Field(default_factory=RewriteModelSelection)
