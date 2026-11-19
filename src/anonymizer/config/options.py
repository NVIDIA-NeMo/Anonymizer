# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The values a workflow receives for one call: the resolved detection options and the run context."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from anonymizer.config.models import ModelSelection
from anonymizer.config.regex import BuiltinRegex, RegexLimits, RegexRule

if TYPE_CHECKING:
    from data_designer.config.models import ModelConfig


@dataclass(frozen=True, kw_only=True)
class DetectionOptions:
    """What detection reads for one call: `Detect` resolved by `Detect.resolve()`."""

    entity_labels: tuple[str, ...] | None
    """The configured labels; None when `Detect.entity_labels` is unset."""
    ordered_labels: tuple[str, ...]
    """The effective labels, in the order sent to the detector and the detection prompts."""
    excluded_labels: tuple[str, ...]
    validator_examples: Mapping[str, tuple[str, ...]]
    """Built-in plus configured examples for every effective label."""
    augmenter_examples: Mapping[str, tuple[str, ...]]
    """Configured examples, for the effective labels that have them."""
    gliner_threshold: float
    validation_max_entities_per_call: int
    validation_excerpt_window_chars: int
    builtin_regexes: bool
    regex_rules: tuple[BuiltinRegex | RegexRule, ...]
    regex_limits: RegexLimits

    @property
    def strict_labels(self) -> bool:
        """True when final entities are restricted to the effective labels (entity_labels is set)."""
        return self.entity_labels is not None


@dataclass(frozen=True, kw_only=True)
class RunContext:
    """Facts of one call that are not user config. Built by the facade per call; never shared between calls."""

    model_configs: tuple[ModelConfig, ...]
    models: ModelSelection
    data_summary: str | None = None
    preview_num_records: int | None = None
