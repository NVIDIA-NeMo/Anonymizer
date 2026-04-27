# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from importlib.metadata import version

__version__ = version("nemo-anonymizer")

# Re-exports from Data Designer so users don't import it directly.
from data_designer.config.models import ModelProvider as ModelProvider

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput, Detect, Rewrite, RiskTolerance
from anonymizer.config.replace_strategies import Annotate, Hash, Redact, Substitute
from anonymizer.engine.constants import DEFAULT_ENTITY_LABELS as _DEFAULT_ENTITY_LABELS
from anonymizer.interface.anonymizer import Anonymizer
from anonymizer.interface.errors import (
    AnonymizerError,
    AnonymizerIOError,
    InvalidConfigError,
    InvalidInputError,
)
from anonymizer.logging import LoggingConfig, configure_logging

# Export as an immutable public constant so callers can inspect defaults
# without mutating the internal source-of-truth list.
DEFAULT_ENTITY_LABELS: tuple[str, ...] = tuple(_DEFAULT_ENTITY_LABELS)

__all__ = [
    "Anonymizer",
    "AnonymizerConfig",
    "AnonymizerError",
    "AnonymizerInput",
    "AnonymizerIOError",
    "Annotate",
    "DEFAULT_ENTITY_LABELS",
    "Detect",
    "Hash",
    "InvalidConfigError",
    "InvalidInputError",
    "LoggingConfig",
    "ModelProvider",
    "Redact",
    "Rewrite",
    "RiskTolerance",
    "Substitute",
    "configure_logging",
]
