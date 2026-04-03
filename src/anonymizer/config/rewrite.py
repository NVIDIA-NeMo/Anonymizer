# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class RiskTolerance(str, Enum):
    """Risk tolerance presets for leakage mass thresholds.

    Each preset bundles a coherent set of repair and review thresholds:

    - **minimal** — Tight leakage threshold (0.6), flags for review aggressively.
      Good for medical, legal, and financial data.
    - **low** — Default. Moderate leakage threshold (1.0).
      Good for most privacy-sensitive data.
    - **moderate** — Relaxed leakage threshold (1.5), lower review bar.
    - **high** — High leakage threshold (2.0), does not auto-repair
      individual high-sensitivity leaks.
    """

    minimal = "minimal"
    low = "low"
    moderate = "moderate"
    high = "high"


SENSITIVITY_WEIGHTS: dict[str, float] = {
    "high": 1.0,
    "medium": 0.6,
    "low": 0.3,
}

DEFAULT_PROTECT_TEXT = (
    "Direct identifiers, quasi-identifier combinations, and latent inferences that could enable re-identification"
)
DEFAULT_PRESERVE_TEXT = "General utility, content quality, and semantic meaning of the original text"


@dataclass(frozen=True)
class _RiskToleranceBundle:
    """Internal: all derived evaluation parameters for a risk_tolerance preset."""

    repair_threshold: float
    repair_any_high_leak: bool
    flag_utility_below: float
    flag_leakage_above: float


_RISK_TOLERANCE_BUNDLES: dict[RiskTolerance, _RiskToleranceBundle] = {
    RiskTolerance.minimal: _RiskToleranceBundle(
        repair_threshold=0.6,
        repair_any_high_leak=True,
        flag_utility_below=0.6,
        flag_leakage_above=1.0,
    ),
    RiskTolerance.low: _RiskToleranceBundle(
        repair_threshold=1.0,
        repair_any_high_leak=True,
        flag_utility_below=0.5,
        flag_leakage_above=2.0,
    ),
    RiskTolerance.moderate: _RiskToleranceBundle(
        repair_threshold=1.5,
        repair_any_high_leak=True,
        flag_utility_below=0.4,
        flag_leakage_above=2.5,
    ),
    RiskTolerance.high: _RiskToleranceBundle(
        repair_threshold=2.0,
        repair_any_high_leak=False,
        flag_utility_below=0.3,
        flag_leakage_above=3.0,
    ),
}


class PrivacyGoal(BaseModel):
    """Structured privacy and utility goal for rewrite mode."""

    protect: str = Field(
        min_length=10, max_length=1000, description="What to protect (e.g. direct identifiers, quasi-identifiers)."
    )
    preserve: str = Field(
        min_length=10, max_length=1000, description="What to preserve (e.g. utility, semantic meaning)."
    )

    @field_validator("protect", "preserve")
    @classmethod
    def validate_min_words(cls, value: str) -> str:
        cleaned = value.strip()
        if len(cleaned.split()) < 3:
            raise ValueError("privacy goal sections must contain at least 3 words")
        return cleaned

    def to_prompt_string(self) -> str:
        """Serialize goal into prompt-ready text."""
        return f"PROTECT: {self.protect}\nPRESERVE: {self.preserve}"


class EvaluationCriteria(BaseModel):
    """Criteria for privacy leakage evaluation and repair.

    ``risk_tolerance`` controls the leakage threshold that triggers repair,
    whether individual high-sensitivity leaks trigger repair, and the
    thresholds for flagging records for human review. See ``RiskTolerance``
    for preset descriptions.

    ``max_repair_iterations`` caps how many repair rounds are attempted
    (each round = one LLM call per failing row). Set to 0 to disable repair
    while still producing evaluation metrics.
    """

    risk_tolerance: RiskTolerance = Field(
        default=RiskTolerance.low,
        description="Preset controlling repair and review thresholds.",
    )
    max_repair_iterations: int = Field(
        default=2,
        ge=0,
        description="Maximum repair rounds. Set to 0 to disable repair.",
    )

    @property
    def _bundle(self) -> _RiskToleranceBundle:
        return _RISK_TOLERANCE_BUNDLES[self.risk_tolerance]

    @property
    def repair_threshold(self) -> float:
        """Leakage mass above which a row is sent for repair."""
        return self._bundle.repair_threshold

    @property
    def repair_any_high_leak(self) -> bool:
        """Whether any single high-sensitivity leak triggers repair."""
        return self._bundle.repair_any_high_leak

    @property
    def flag_utility_below(self) -> float:
        """Flag for human review if utility score is below this."""
        return self._bundle.flag_utility_below

    @property
    def flag_leakage_above(self) -> float:
        """Flag for human review if leakage mass exceeds this."""
        return self._bundle.flag_leakage_above

    @property
    def sensitivity_weights(self) -> dict[str, float]:
        """Weights for high/medium/low sensitivity levels in leakage mass computation."""
        return dict(SENSITIVITY_WEIGHTS)
