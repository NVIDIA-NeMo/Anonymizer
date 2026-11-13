# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic privacy question construction shared by Replace and Rewrite."""

from __future__ import annotations

from typing import Literal

from anonymizer.engine.schemas import EntityCategory, PrivacyQuestionSchema, SensitivityLevel


def build_privacy_question(
    *,
    question_id: int,
    entity_label: str,
    entity_value: str,
    sensitivity: SensitivityLevel,
    category: EntityCategory,
    text_kind: Literal["replaced", "rewritten"],
) -> PrivacyQuestionSchema:
    """Build a question about recovery of an original value from anonymized text.

    The caller selects targets and assigns sequential IDs; this builder never
    filters entities by sensitivity or protection decisions.
    """
    return PrivacyQuestionSchema(
        id=question_id,
        question=f"Can the {entity_label} '{entity_value}' be deduced from the {text_kind} text?",
        sensitivity=sensitivity,
        entity_label=entity_label,
        entity_value=entity_value,
        category=category,
    )
