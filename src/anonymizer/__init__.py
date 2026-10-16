# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from importlib import import_module
from importlib.metadata import version
from typing import TYPE_CHECKING

from anonymizer.engine.constants import DEFAULT_ENTITY_LABELS as _DEFAULT_ENTITY_LABELS

__version__ = version("nemo-anonymizer")
DEFAULT_ENTITY_LABELS: tuple[str, ...] = tuple(_DEFAULT_ENTITY_LABELS)

if TYPE_CHECKING:
    from data_designer.config.models import ModelProvider as ModelProvider
    from data_designer.config.run_config import RunConfig as RunConfig

    from anonymizer import graph as graph
    from anonymizer.config.anonymizer_config import (
        AnonymizerConfig,
        AnonymizerInput,
        Detect,
        EvaluateConfig,
        Rewrite,
        RiskTolerance,
    )
    from anonymizer.config.replace_strategies import Annotate, Hash, Redact, Substitute
    from anonymizer.config.rewrite import PrivacyGoal
    from anonymizer.interface.anonymizer import Anonymizer as Anonymizer
    from anonymizer.interface.errors import (
        AnonymizerError,
        AnonymizerIOError,
        InvalidConfigError,
        InvalidInputError,
    )
    from anonymizer.logging import LoggingConfig, configure_logging

_EXPORTS = {
    "AnonymizerConfig": "anonymizer.config.anonymizer_config",
    "AnonymizerInput": "anonymizer.config.anonymizer_config",
    "Detect": "anonymizer.config.anonymizer_config",
    "EvaluateConfig": "anonymizer.config.anonymizer_config",
    "Rewrite": "anonymizer.config.anonymizer_config",
    "RiskTolerance": "anonymizer.config.anonymizer_config",
    "Annotate": "anonymizer.config.replace_strategies",
    "Hash": "anonymizer.config.replace_strategies",
    "Redact": "anonymizer.config.replace_strategies",
    "Substitute": "anonymizer.config.replace_strategies",
    "PrivacyGoal": "anonymizer.config.rewrite",
    "AnonymizerError": "anonymizer.interface.errors",
    "AnonymizerIOError": "anonymizer.interface.errors",
    "InvalidConfigError": "anonymizer.interface.errors",
    "InvalidInputError": "anonymizer.interface.errors",
    "LoggingConfig": "anonymizer.logging",
    "configure_logging": "anonymizer.logging",
    "ModelProvider": "data_designer.config.models",
    "RunConfig": "data_designer.config.run_config",
    "Anonymizer": "anonymizer.interface.anonymizer",
}


def __getattr__(name: str) -> object:
    if name == "graph":
        value = import_module("anonymizer.graph")
    elif name in _EXPORTS:
        value = getattr(import_module(_EXPORTS[name]), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "Anonymizer",
    "AnonymizerConfig",
    "AnonymizerError",
    "AnonymizerInput",
    "AnonymizerIOError",
    "Annotate",
    "DEFAULT_ENTITY_LABELS",
    "Detect",
    "EvaluateConfig",
    "Hash",
    "InvalidConfigError",
    "InvalidInputError",
    "LoggingConfig",
    "ModelProvider",
    "PrivacyGoal",
    "Redact",
    "Rewrite",
    "RiskTolerance",
    "RunConfig",
    "Substitute",
    "configure_logging",
    "graph",
]
