# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from anonymizer.measurement.constants import MEASUREMENT_SCHEMA_VERSION


def _detect_config_metadata(detect: Any | None) -> dict[str, Any]:
    entity_labels = getattr(detect, "entity_labels", None)
    if entity_labels is None:
        from anonymizer.engine.constants import DEFAULT_ENTITY_LABELS

        entity_label_count = len(DEFAULT_ENTITY_LABELS)
    else:
        entity_label_count = len(entity_labels)
    return {
        "gliner_threshold": getattr(detect, "gliner_threshold", None),
        "entity_label_source": "custom" if entity_labels is not None else "default",
        "entity_label_count": entity_label_count,
        "entity_labels": list(entity_labels) if entity_labels is not None else None,
        "validation_max_entities_per_call": getattr(detect, "validation_max_entities_per_call", None),
        "validation_excerpt_window_chars": getattr(detect, "validation_excerpt_window_chars", None),
    }


def _source_metadata(source: str) -> dict[str, Any]:
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        return {
            "kind": "remote_file",
            "scheme": parsed.scheme,
            "suffix": Path(parsed.path).suffix.lower() or None,
        }
    if parsed.scheme == "file":
        return {
            "kind": "local_file",
            "scheme": "file",
            "suffix": Path(parsed.path).suffix.lower() or None,
        }
    return {
        "kind": "local_file" if source else "unknown",
        "scheme": None,
        "suffix": Path(source).suffix.lower() or None,
    }


def _replace_config_metadata(replace_config: Any | None) -> dict[str, Any] | None:
    if replace_config is None:
        return None

    metadata: dict[str, Any] = {
        "strategy": type(replace_config).__name__,
        "has_instructions": bool(getattr(replace_config, "instructions", None)),
    }
    for attr in ("normalize_label", "algorithm", "digest_length"):
        if hasattr(replace_config, attr):
            metadata[attr] = getattr(replace_config, attr)
    if hasattr(replace_config, "format_template"):
        metadata["has_format_template"] = True
    return metadata


def _rewrite_config_metadata(rewrite_config: Any | None) -> dict[str, Any] | None:
    if rewrite_config is None:
        return None
    return {
        "risk_tolerance": _enum_value(getattr(rewrite_config, "risk_tolerance", None)),
        "max_repair_iterations": getattr(rewrite_config, "max_repair_iterations", None),
        "strict_entity_protection": getattr(rewrite_config, "strict_entity_protection", None),
        "has_privacy_goal": bool(getattr(rewrite_config, "privacy_goal", None)),
        "has_instructions": bool(getattr(rewrite_config, "instructions", None)),
    }


def _model_config_metadata(model_config: Any) -> dict[str, Any]:
    inference_parameters = getattr(model_config, "inference_parameters", None)
    return {
        "alias": getattr(model_config, "alias", None),
        "model": getattr(model_config, "model", None),
        "provider": _enum_value(getattr(model_config, "provider", None)),
        "base_url": bool(getattr(model_config, "base_url", None)),
        "max_parallel_requests": getattr(inference_parameters, "max_parallel_requests", None),
    }


def _runtime_metadata() -> dict[str, Any]:
    try:
        anonymizer_version = version("nemo-anonymizer")
    except PackageNotFoundError:
        anonymizer_version = None
    return {
        "anonymizer_version": anonymizer_version,
        "measurement_schema_version": MEASUREMENT_SCHEMA_VERSION,
        "platform_machine": platform.machine(),
        "platform_system": platform.system(),
        "python_version": platform.python_version(),
    }


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)
