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
        TextRecord,
        TextRecordsInput,
    )
    from anonymizer.config.regex import (
        BUILTIN_REGEX_LABELS,
        BuiltinRegex,
        RegexCandidate,
        RegexLimits,
        RegexRule,
        RegexValidationResult,
    )
    from anonymizer.config.replace_strategies import Annotate, Hash, Redact, Substitute
    from anonymizer.config.rewrite import PrivacyGoal
    from anonymizer.engine.detection.result import DetectedSpan, DetectorResult, DetectorRow
    from anonymizer.interface.anonymizer import Anonymizer as Anonymizer
    from anonymizer.interface.errors import (
        AnonymizerError,
        AnonymizerIOError,
        InvalidConfigError,
        InvalidDetectorResultError,
        InvalidInputError,
    )
    from anonymizer.interface.factory import (
        GlinerEndpoint,
        NativeGliner,
        NativeRuntimeSession,
        create_anonymizer,
        native_runtime,
    )
    from anonymizer.logging import LoggingConfig, configure_logging

_EXPORTS = {
    "AnonymizerConfig": "anonymizer.config.anonymizer_config",
    "AnonymizerInput": "anonymizer.config.anonymizer_config",
    "Detect": "anonymizer.config.anonymizer_config",
    "EvaluateConfig": "anonymizer.config.anonymizer_config",
    "Rewrite": "anonymizer.config.anonymizer_config",
    "RiskTolerance": "anonymizer.config.anonymizer_config",
    "TextRecord": "anonymizer.config.anonymizer_config",
    "TextRecordsInput": "anonymizer.config.anonymizer_config",
    "BUILTIN_REGEX_LABELS": "anonymizer.config.regex",
    "BuiltinRegex": "anonymizer.config.regex",
    "RegexCandidate": "anonymizer.config.regex",
    "RegexLimits": "anonymizer.config.regex",
    "RegexRule": "anonymizer.config.regex",
    "RegexValidationResult": "anonymizer.config.regex",
    "Annotate": "anonymizer.config.replace_strategies",
    "Hash": "anonymizer.config.replace_strategies",
    "Redact": "anonymizer.config.replace_strategies",
    "Substitute": "anonymizer.config.replace_strategies",
    "PrivacyGoal": "anonymizer.config.rewrite",
    "AnonymizerError": "anonymizer.interface.errors",
    "AnonymizerIOError": "anonymizer.interface.errors",
    "InvalidConfigError": "anonymizer.interface.errors",
    "InvalidInputError": "anonymizer.interface.errors",
    "InvalidDetectorResultError": "anonymizer.interface.errors",
    "DetectedSpan": "anonymizer.engine.detection.result",
    "DetectorResult": "anonymizer.engine.detection.result",
    "DetectorRow": "anonymizer.engine.detection.result",
    "LoggingConfig": "anonymizer.logging",
    "configure_logging": "anonymizer.logging",
    "ModelProvider": "data_designer.config.models",
    "RunConfig": "data_designer.config.run_config",
    "Anonymizer": "anonymizer.interface.anonymizer",
    "GlinerEndpoint": "anonymizer.interface.factory",
    "NativeGliner": "anonymizer.interface.factory",
    "NativeRuntimeSession": "anonymizer.interface.factory",
    "create_anonymizer": "anonymizer.interface.factory",
    "native_runtime": "anonymizer.interface.factory",
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
    "BUILTIN_REGEX_LABELS",
    "BuiltinRegex",
    "DEFAULT_ENTITY_LABELS",
    "Detect",
    "DetectedSpan",
    "DetectorResult",
    "DetectorRow",
    "EvaluateConfig",
    "GlinerEndpoint",
    "Hash",
    "InvalidConfigError",
    "InvalidDetectorResultError",
    "InvalidInputError",
    "LoggingConfig",
    "ModelProvider",
    "NativeGliner",
    "NativeRuntimeSession",
    "PrivacyGoal",
    "Redact",
    "RegexCandidate",
    "RegexLimits",
    "RegexRule",
    "RegexValidationResult",
    "Rewrite",
    "RiskTolerance",
    "RunConfig",
    "Substitute",
    "TextRecord",
    "TextRecordsInput",
    "configure_logging",
    "create_anonymizer",
    "graph",
    "native_runtime",
]
