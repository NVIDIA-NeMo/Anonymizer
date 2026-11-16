# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only views of the resolved options a workflow reads from a config."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from anonymizer.config.anonymizer_config import DetectionScope
from anonymizer.config.rewrite import EvaluationCriteria, PrivacyGoal


class DetectionOptions(Protocol):
    """What detection reads from `Detect`: resolved values, not the raw label fields."""

    @property
    def scope(self) -> DetectionScope: ...

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


class RewriteOptions(Protocol):
    """What rewrite reads from `Rewrite`."""

    @property
    def privacy_goal(self) -> PrivacyGoal: ...

    @property
    def evaluation(self) -> EvaluationCriteria: ...

    @property
    def strict_entity_protection(self) -> bool: ...
