# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private, non-serializable description of one normalized execution."""

from __future__ import annotations

from dataclasses import dataclass

from data_designer.config.models import ModelConfig

from anonymizer.config.anonymizer_config import AnonymizerConfig
from anonymizer.config.models import ModelSelection
from anonymizer.config.replace_strategies import ReplaceMethod
from anonymizer.config.rewrite import EvaluationCriteria, PrivacyGoal


@dataclass(frozen=True)
class _CompiledRewrite:
    """Execution-only rewrite choices, detached from the public config object."""

    privacy_goal: PrivacyGoal
    evaluation: EvaluationCriteria
    use_combined_graph: bool
    strict_entity_protection: bool


@dataclass(frozen=True)
class _CompiledInvocation:
    """Immutable, private execution plan; deliberately excludes input and runtime state."""

    model_configs: tuple[ModelConfig, ...]
    selected_models: ModelSelection
    gliner_detection_threshold: float
    validation_max_entities_per_call: int
    validation_excerpt_window_chars: int
    entity_labels: tuple[str, ...] | None
    replace_method: ReplaceMethod | None
    rewrite: _CompiledRewrite | None

    @classmethod
    def compile(
        cls,
        config: AnonymizerConfig,
        selected_models: ModelSelection,
        model_configs: list[ModelConfig] | None = None,
    ) -> _CompiledInvocation:
        """Capture only workflow inputs after the public request has been normalized."""
        rewrite = config.rewrite
        compiled_rewrite = None
        if rewrite is not None:
            privacy_goal = rewrite.privacy_goal
            if privacy_goal is None:
                raise ValueError("rewrite.privacy_goal must not be None")
            compiled_rewrite = _CompiledRewrite(
                privacy_goal=privacy_goal.model_copy(deep=True),
                evaluation=rewrite.evaluation,
                use_combined_graph=rewrite.use_combined_graph,
                strict_entity_protection=rewrite.strict_entity_protection,
            )
        return cls(
            model_configs=tuple(model_config.model_copy(deep=True) for model_config in model_configs or ()),
            selected_models=selected_models.model_copy(deep=True),
            gliner_detection_threshold=config.detect.gliner_threshold,
            validation_max_entities_per_call=config.detect.validation_max_entities_per_call,
            validation_excerpt_window_chars=config.detect.validation_excerpt_window_chars,
            entity_labels=tuple(config.detect.entity_labels) if config.detect.entity_labels is not None else None,
            replace_method=config.replace.model_copy(deep=True) if config.replace is not None else None,
            rewrite=compiled_rewrite,
        )

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("compiled invocation is not serializable")
