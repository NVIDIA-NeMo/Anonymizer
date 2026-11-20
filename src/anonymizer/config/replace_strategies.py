# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import hmac
import re
from string import Formatter
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, SecretStr, field_validator, model_validator


def _normalize_replace_kind(value: Any) -> Any:
    """Lowercase the ``kind`` key of dict input so ``{"kind": "Redact"}`` is accepted."""
    if isinstance(value, dict) and isinstance(value.get("kind"), str):
        return {**value, "kind": value["kind"].lower()}
    return value


class ReplaceMethodBase(BaseModel):
    """Shared configuration for all replacement methods."""

    def _render_template(self, template: str, *, text: str, label: str, **extra: str) -> str:
        return template.format(text=text, label=label, **extra)

    @staticmethod
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


class Substitute(ReplaceMethodBase):
    """Replace entities with LLM-generated synthetic values."""

    kind: Literal["substitute"] = "substitute"
    instructions: str | None = Field(
        default=None, description="Additional instructions for the LLM replacement generator."
    )
    consistency: Literal["record", "run"] = Field(
        default="record",
        description=(
            "'record' gives each value (with its case and whitespace variants) one substitute within its record;"
            " two different values of a record can get the same substitute. 'run' gives each value one substitute"
            " across the whole run, and two different values never share one. A substitute equal to a real value"
            " of the record is replaced by [SUBSTITUTE_<LABEL>_<n>]; in 'run', so is a substitute another value"
            " already uses. A mention can still get a different substitute where the shared one would clash with"
            " another value, or another value's substitute, in the same record; in 'record' this happens only to"
            " case or whitespace variants."
        ),
    )


class Redact(ReplaceMethodBase):
    """Replace each entity with a configurable redaction template."""

    kind: Literal["redact"] = "redact"
    format_template: str = Field(
        default="[REDACTED_{label}]", description="Nonempty redaction template with optional {label} placeholder."
    )
    normalize_label: bool = Field(default=True, description="Uppercase and clean label before substitution.")

    @field_validator("format_template")
    @classmethod
    def validate_format_template(cls, value: str) -> str:
        cls._validate_template(
            value=value,
            required_fields=set(),
            allowed_fields={"label"},
        )
        # Nested format specs depend on the actual label; validate their render below.
        if not any(spec and "{" in spec for _, _, spec, _ in Formatter().parse(value)):
            try:
                rendered = value.format(label="label")
            except (ValueError, IndexError, KeyError) as exc:
                raise ValueError(f"Invalid redaction template {value!r}: {exc}") from exc
            cls._require_nonempty(rendered, value)
        return value

    @staticmethod
    def _require_nonempty(rendered: str, template: str) -> str:
        if rendered == "":
            raise ValueError(f"Redaction template {template!r} must produce a nonempty replacement")
        return rendered

    def replace(self, text: str, label: str) -> str:
        """Apply the redaction template to a single entity occurrence.

        Args:
            text: Original entity text (e.g. ``"Alice"``).
            label: Entity label (e.g. ``"first_name"``).
        """
        normalized_label = _format_label_for_redaction(label) if self.normalize_label else label
        rendered = self._render_template(
            template=self.format_template,
            text=text,
            label=normalized_label,
        )
        return self._require_nonempty(rendered, self.format_template)


class Annotate(ReplaceMethodBase):
    """Tag each entity with a readable label token."""

    kind: Literal["annotate"] = "annotate"
    format_template: str = Field(
        default="<{text}, {label}>", description="Template with {text} and {label} placeholders."
    )

    @field_validator("format_template")
    @classmethod
    def validate_format_template(cls, value: str) -> str:
        cls._validate_template(
            value=value,
            required_fields={"text", "label"},
            allowed_fields={"text", "label"},
        )
        return value

    def replace(self, text: str, label: str) -> str:
        """Apply the annotation template to a single entity occurrence.

        Args:
            text: Original entity text (e.g. ``"Alice"``).
            label: Entity label (e.g. ``"first_name"``).
        """
        return self._render_template(
            template=self.format_template,
            text=text,
            label=label,
        )


_MISSING_KEY_MESSAGE = (
    'Hash requires a secret key: pass Hash(key=...), for example Hash(key=os.environ["ANONYMIZER_HASH_KEY"]) '
    "(the CLI reads ANONYMIZER_HASH_KEY). Unkeyed hashes of low-entropy values are reversible, so there is no "
    "unkeyed mode; use the same key across runs to get the same tokens."
)


class Hash(ReplaceMethodBase):
    """Replace each entity with a keyed hash token (HMAC-SHA256 of the case- and whitespace-folded text)."""

    # Validation errors would otherwise echo the plaintext key.
    model_config = ConfigDict(hide_input_in_errors=True)

    kind: Literal["hash"] = "hash"
    algorithm: Literal["sha256", "sha1", "md5"] = Field(
        default="sha256", description="Hash algorithm. Only 'sha256' is accepted: the digest is always HMAC-SHA256."
    )
    digest_length: int = Field(
        default=12, ge=6, le=64, description="Number of hex characters to keep from the hash digest."
    )
    format_template: str = Field(
        default="<HASH_{label}_{digest}>", description="Template with {digest} required and optional {label}."
    )
    key: SecretStr = Field(
        description="Required secret key. The digest is HMAC-SHA256 of the case- and whitespace-folded text; "
        "unkeyed digests of low-entropy values are reversible, so there is no unkeyed mode.",
    )

    @model_validator(mode="before")
    @classmethod
    def require_key(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("key") is None:
            raise ValueError(_MISSING_KEY_MESSAGE)
        return data

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: SecretStr) -> SecretStr:
        secret = value.get_secret_value()
        if not secret.strip():
            raise ValueError("key must not be empty or whitespace")
        if set(secret) == {"*"}:
            raise ValueError(
                "key is a masked placeholder; a keyed Hash does not survive serialization, so pass the real key"
            )
        return value

    @model_validator(mode="after")
    def validate_keyed_algorithm(self) -> Hash:
        if self.algorithm != "sha256":
            raise ValueError("a keyed Hash always uses HMAC-SHA256; leave algorithm at its default 'sha256'")
        return self

    @field_validator("format_template")
    @classmethod
    def validate_format_template(cls, value: str) -> str:
        cls._validate_template(
            value=value,
            required_fields={"digest"},
            allowed_fields={"label", "digest"},
        )
        return value

    def replace(self, text: str, label: str) -> str:
        """Apply the hash template to a single entity occurrence.

        Args:
            text: Original entity text (e.g. ``"Alice"``).
            label: Entity label (e.g. ``"first_name"``).
        """
        if self.key is None:
            # Only a result's stored copy has no key (see ``without_key``); it never hashes.
            raise ValueError(_MISSING_KEY_MESSAGE)
        # Same fold as Substitute's value matching ("casefold-whitespace"): casefold, then collapse whitespace.
        # Changing it changes every token, so keep the two in step.
        folded = " ".join(text.casefold().split()).encode("utf-8")
        key = self.key.get_secret_value().encode("utf-8")
        digest = hmac.new(key, folded, hashlib.sha256).hexdigest()[: self.digest_length]
        return self._render_template(
            template=self.format_template,
            text=text,
            label=label.upper(),
            digest=digest,
        )

    def without_key(self) -> Hash:
        """A copy with the key removed, for results that keep the strategy. It cannot hash."""
        return self.model_copy(update={"key": None})


ReplaceMethod = Annotated[
    Annotate | Redact | Hash | Substitute,
    Field(discriminator="kind"),
    BeforeValidator(_normalize_replace_kind),
]

LocalReplaceMethod = Annotated[
    Annotate | Redact | Hash,
    Field(discriminator="kind"),
    BeforeValidator(_normalize_replace_kind),
]


def _format_label_for_redaction(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", label.strip()).strip("_")
    if cleaned == "":
        return "UNKNOWN"
    return cleaned.upper()
