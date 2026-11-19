# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only views of the resolved options a workflow reads from a config."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from anonymizer.config.anonymizer_config import DetectionScope
from anonymizer.config.models import ModelSelection
from anonymizer.config.regex import BuiltinRegex, RegexLimits, RegexRule

if TYPE_CHECKING:
    from data_designer.config.models import ModelConfig


class DetectionOptions(Protocol):
    """What detection reads from `Detect`: resolved values, not the raw label fields."""

    @property
    def scope(self) -> DetectionScope: ...

    @property
    def ordered_labels(self) -> tuple[str, ...]:
        """The effective labels in the order sent to the detector and the detection prompts."""
        ...

    @property
    def strict_labels(self) -> bool: ...

    @property
    def validator_examples(self) -> Mapping[str, tuple[str, ...]]: ...

    @property
    def augmenter_examples(self) -> Mapping[str, tuple[str, ...]]: ...

    @property
    def validation_max_entities_per_call(self) -> int: ...

    @property
    def validation_excerpt_window_chars(self) -> int: ...

    @property
    def builtin_regexes(self) -> bool: ...

    @property
    def regex_rules(self) -> Sequence[BuiltinRegex | RegexRule]: ...

    @property
    def regex_limits(self) -> RegexLimits: ...


@dataclass(frozen=True, kw_only=True)
class RunContext:
    """Facts of one call that are not user config. Built by the facade per call; never shared between calls."""

    model_configs: tuple[ModelConfig, ...]
    models: ModelSelection
    data_summary: str | None = None
    preview_num_records: int | None = None
