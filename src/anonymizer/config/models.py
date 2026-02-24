# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pydantic import BaseModel


class DetectionModelSelection(BaseModel):
    """Role-level model aliases for entity detection.

    Defaults are loaded from ``default_model_configs/entity_detection.yaml``
    via ``load_default_model_selection()``. Users override individual roles
    through ``AnonymizerConfig(selected_models=...)``.
    """

    entity_detector: str
    entity_validator: str
    entity_augmenter: str
    latent_detector: str


class ReplaceModelSelection(BaseModel):
    """Role-level model aliases for replacement workflows."""

    replacement_generator: str


class RewriteModelSelection(BaseModel):
    """Role-level model aliases for rewrite workflows."""

    rewriter: str
    evaluator: str


class ModelSelection(BaseModel):
    """Model alias selection grouped by workflow and role.

    Constructed with defaults from YAML via ``load_default_model_selection()``.
    """

    detection: DetectionModelSelection
    replace: ReplaceModelSelection
    rewrite: RewriteModelSelection
