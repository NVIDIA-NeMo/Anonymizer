# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Freeze and restore the local replace strategies (Redact, Annotate, Hash) as admitted graph configuration."""

from __future__ import annotations

import json

from pydantic import TypeAdapter

from anonymizer.config.replace_strategies import LocalReplaceMethod, Redact
from anonymizer.engine.graph_sdk.capabilities import (
    ConfigBoolean,
    ConfigField,
    ConfigSequence,
    ConfigText,
    FrozenConfig,
)


def redact_configuration(strategy: Redact) -> FrozenConfig:
    """Snapshot existing Redact fields; an empty sequence preserves an empty template."""
    return FrozenConfig(
        fields=(
            ConfigField(
                name="format_template",
                value=ConfigSequence(
                    items=(ConfigText(value=strategy.format_template),) if strategy.format_template else ()
                ),
            ),
            ConfigField(name="normalize_label", value=ConfigBoolean(value=strategy.normalize_label)),
        )
    )


def redact_strategy(configuration: FrozenConfig) -> Redact:
    """Reconstruct the strategy from its admitted, immutable configuration."""
    fields = {field.name: field.value for field in configuration.fields}
    template, normalize = fields.get("format_template"), fields.get("normalize_label")
    if (
        set(fields) != {"format_template", "normalize_label"}
        or type(template) is not ConfigSequence
        or len(template.items) > 1
        or any(type(item) is not ConfigText for item in template.items)
        or type(normalize) is not ConfigBoolean
    ):
        raise ValueError("Invalid admitted Redact configuration")
    return Redact(
        format_template="".join(item.value for item in template.items if isinstance(item, ConfigText)),
        normalize_label=normalize.value,
    )


def local_configuration(strategy: LocalReplaceMethod) -> FrozenConfig:
    """Freeze local options with the public discriminator; retain Redact's established shape."""
    if isinstance(strategy, Redact):
        return redact_configuration(strategy)
    return FrozenConfig(
        fields=(ConfigField(name=type(strategy).__name__.lower(), value=ConfigText(value=strategy.model_dump_json())),)
    )


def local_strategy(configuration: FrozenConfig) -> LocalReplaceMethod:
    """Reconstruct the existing public model from the single admitted configuration."""
    if len(configuration.fields) == 1:
        (field,) = configuration.fields
        if field.name in {"hash", "annotate"} and type(field.value) is ConfigText:
            return TypeAdapter(LocalReplaceMethod).validate_python(
                {**json.loads(field.value.value), "kind": field.name}
            )
    return redact_strategy(configuration)
