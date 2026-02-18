# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# This file is forward looking, and will support future rewrite mode.

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class RiskTolerance(str, Enum):
    """Risk tolerance presets for leakage mass thresholds."""

    strict = "strict"
    conservative = "conservative"
    moderate = "moderate"
    permissive = "permissive"


RISK_TOLERANCE_THRESHOLDS: dict[RiskTolerance, float] = {
    RiskTolerance.strict: 0.6,
    RiskTolerance.conservative: 1.0,
    RiskTolerance.moderate: 1.5,
    RiskTolerance.permissive: 2.0,
}

DOMAIN_RISK_TOLERANCE: dict[str, RiskTolerance] = {
    "medical": RiskTolerance.strict,
    "clinical": RiskTolerance.strict,
    "healthcare": RiskTolerance.strict,
    "legal": RiskTolerance.strict,
    "financial": RiskTolerance.strict,
    "banking": RiskTolerance.strict,
    "insurance": RiskTolerance.strict,
    "hr": RiskTolerance.conservative,
    "human_resources": RiskTolerance.conservative,
    "education": RiskTolerance.conservative,
    "customer_service": RiskTolerance.conservative,
    "general": RiskTolerance.moderate,
    "social_media": RiskTolerance.permissive,
    "entertainment": RiskTolerance.permissive,
}

SENSITIVITY_WEIGHTS: dict[str, float] = {
    "high": 1.0,
    "medium": 0.6,
    "low": 0.3,
}

DEFAULT_PROTECT_TEXT = (
    "Direct identifiers, quasi-identifier combinations, and latent inferences that could enable re-identification"
)
DEFAULT_PRESERVE_TEXT = "General utility, content quality, and semantic meaning of the original text"


class PrivacyGoal(BaseModel):
    """Structured privacy and utility goal for rewrite mode."""

    protect: str = Field(min_length=10, max_length=1000)
    preserve: str = Field(min_length=10, max_length=1000)

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


class RewriteParams(BaseModel):
    """Configuration for rewrite-mode execution."""

    instructions: str | None = None
    skip_low_sensitivity_pii: bool = False


class EvaluationCriteria(BaseModel):
    """Criteria and thresholds for privacy leakage and utility scoring."""

    risk_tolerance: RiskTolerance = RiskTolerance.conservative
    max_leakage_mass: float | None = Field(default=None, ge=0.0)
    auto_adjust_by_domain: bool = False

    repair_any_high_leak: bool = True
    max_repair_iterations: int = Field(default=2, ge=0)
    auto_repair_privacy: bool = True

    flag_utility_below: float | None = Field(default=0.50, ge=0.0, le=1.0)
    flag_leakage_mass_above: float | None = Field(default=2.0, ge=0.0)
    sensitivity_weights: dict[str, float] = Field(default_factory=lambda: dict(SENSITIVITY_WEIGHTS))

    @field_validator("sensitivity_weights")
    @classmethod
    def validate_sensitivity_weights(cls, value: dict[str, float]) -> dict[str, float]:
        required_levels = {"high", "medium", "low"}
        missing_levels = required_levels - set(value.keys())
        if missing_levels:
            missing = ", ".join(sorted(missing_levels))
            raise ValueError(f"sensitivity_weights is missing required levels: {missing}")

        negative_weights = [level for level, weight in value.items() if weight < 0]
        if negative_weights:
            labels = ", ".join(sorted(negative_weights))
            raise ValueError(f"sensitivity_weights must be non-negative for all levels, invalid: {labels}")
        return value

    def get_effective_threshold(self, domain: str | None = None) -> float:
        """Return the effective leakage-mass threshold for a run."""
        if self.max_leakage_mass is not None:
            return self.max_leakage_mass

        if self.auto_adjust_by_domain and domain:
            domain_key = domain.lower().replace(" ", "_").replace("-", "_")
            domain_tolerance = DOMAIN_RISK_TOLERANCE.get(domain_key, RiskTolerance.moderate)
            return RISK_TOLERANCE_THRESHOLDS[domain_tolerance]

        return RISK_TOLERANCE_THRESHOLDS[self.risk_tolerance]
