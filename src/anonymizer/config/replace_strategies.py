# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import re
from string import Formatter
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator


class LabelReplace(BaseModel):
    """Replace each entity with a readable label token."""

    kind: Literal["label"] = "label"
    format_template: str = Field(
        default="<{text}, {label}>", description="Template with {text} and {label} placeholders."
    )

    @field_validator("format_template")
    @classmethod
    def validate_format_template(cls, value: str) -> str:
        _validate_template(
            value=value,
            required_fields={"text", "label"},
            allowed_fields={"text", "label"},
        )
        return value

    def replace(self, text: str, label: str) -> str:
        return _render_template(
            template=self.format_template,
            text=text,
            label=label,
        )


class RedactReplace(BaseModel):
    """Replace each entity with a configurable redaction template."""

    kind: Literal["redact"] = "redact"
    format_template: str = Field(
        default="[REDACTED_{label}]", description="Template with optional {text} and {label} placeholders."
    )
    normalize_label: bool = Field(default=True, description="Uppercase and clean label before substitution.")

    @field_validator("format_template")
    @classmethod
    def validate_format_template(cls, value: str) -> str:
        _validate_template(
            value=value,
            required_fields=set(),
            allowed_fields={"text", "label"},
        )
        return value

    def replace(self, text: str, label: str) -> str:
        normalized_label = _format_label_for_redaction(label) if self.normalize_label else label
        return _render_template(
            template=self.format_template,
            text=text,
            label=normalized_label,
        )


class HashReplace(BaseModel):
    """Replace each entity with a deterministic hash token."""

    kind: Literal["hash"] = "hash"
    algorithm: Literal["sha256", "sha1", "md5"] = Field(default="sha256", description="Hash algorithm.")
    digest_length: int = Field(
        default=12, ge=6, le=64, description="Number of hex characters to keep from the hash digest."
    )
    format_template: str = Field(
        default="<HASH_{label}_{digest}>", description="Template with {digest} required; {text} and {label} optional."
    )

    @field_validator("format_template")
    @classmethod
    def validate_format_template(cls, value: str) -> str:
        _validate_template(
            value=value,
            required_fields={"digest"},
            allowed_fields={"text", "label", "digest"},
        )
        return value

    def replace(self, text: str, label: str) -> str:
        digest = hashlib.new(self.algorithm, text.encode("utf-8")).hexdigest()[: self.digest_length]
        return self.format_template.format(text=text, label=label.upper(), digest=digest)


class LLMReplace(BaseModel):
    """Marker config for LLM-backed replacement workflow execution."""

    kind: Literal["llm"] = "llm"
    instructions: str | None = Field(
        default=None, description="Additional instructions for the LLM replacement generator."
    )


ReplaceStrategy = Annotated[
    LabelReplace | RedactReplace | HashReplace | LLMReplace,
    Field(discriminator="kind"),
]

LocalReplaceStrategy = Annotated[
    LabelReplace | RedactReplace | HashReplace,
    Field(discriminator="kind"),
]


def _format_label_for_redaction(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", label.strip()).strip("_")
    if cleaned == "":
        return "UNKNOWN"
    return cleaned.upper()


def _validate_template(
    value: str,
    required_fields: set[str],
    allowed_fields: set[str] | None = None,
) -> None:
    fields = {field_name for _, field_name, _, _ in Formatter().parse(value) if field_name}
    allowed = allowed_fields or {"text", "label"}
    unexpected = fields - allowed
    if unexpected:
        invalid = ", ".join(sorted(unexpected))
        raise ValueError(f"template contains unsupported placeholders: {invalid}")
    if not required_fields.issubset(fields):
        missing = ", ".join(sorted(required_fields - fields))
        raise ValueError(f"template is missing required placeholders: {missing}")


def _render_template(template: str, *, text: str, label: str) -> str:
    return template.format(text=text, label=label)
