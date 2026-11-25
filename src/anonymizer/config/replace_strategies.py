# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import hmac
import re
from string import Formatter
from typing import Annotated, Any, Literal

import cyclopts
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, SecretStr, field_validator, model_validator


def _normalize_replace_kind(value: Any) -> Any:
    """Lowercase the ``kind`` key of dict input so ``{"kind": "Redact"}`` is accepted.

    Input that will not reach ``Hash`` has any ``key`` masked, and a list or other collection is replaced by its type
    name: a union or discriminator error carries its whole input, so either would otherwise echo a Hash key.
    """
    if isinstance(value, dict):
        kind = value.get("kind")
        if isinstance(kind, str):
            value = {**value, "kind": kind.lower()}
        if value.get("kind") != "hash" and "key" in value:
            value = {**value, "key": "**********"}
        return value
    if isinstance(value, (list, tuple, set, frozenset)):
        return f"<{type(value).__name__}>"
    return value


# A strategy's kind is the CLI subcommand that picks it (``anonymizer run redact``), not a flag.
_KIND_IS_THE_SUBCOMMAND = cyclopts.Parameter(parse=False)


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

    kind: Annotated[Literal["substitute"], _KIND_IS_THE_SUBCOMMAND] = "substitute"
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

    kind: Annotated[Literal["redact"], _KIND_IS_THE_SUBCOMMAND] = "redact"
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

    kind: Annotated[Literal["annotate"], _KIND_IS_THE_SUBCOMMAND] = "annotate"
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


# The environment variable the CLI reads the hash key from. The key has no CLI flag, so it stays out of argv and
# shell history.
HASH_KEY_ENV = "ANONYMIZER_HASH_KEY"

HASH_KEY_REQUIRED_MESSAGE = (
    'Hash requires a secret key: pass Hash(key=...), for example Hash(key=os.environ["ANONYMIZER_HASH_KEY"]) '
    "(the CLI reads ANONYMIZER_HASH_KEY). Unkeyed hashes of low-entropy values are reversible, so there is no "
    "unkeyed mode; use the same key across runs to get the same tokens."
)


class _RejectedKey:
    """Stands in for a rejected key, so the error raised for it carries no key material."""

    def __init__(self, message: str) -> None:
        self.message = message

    def __repr__(self) -> str:
        return "<rejected key>"


_UNENCODABLE_KEY = _RejectedKey("key must be text that encodes as UTF-8")
_NOT_A_STRING_KEY = _RejectedKey("key must be a string")


class Hash(ReplaceMethodBase):
    """Replace each entity with a keyed hash token (HMAC-SHA256 of the case- and whitespace-folded text)."""

    # Validation errors would otherwise echo the plaintext key.
    model_config = ConfigDict(hide_input_in_errors=True)

    kind: Annotated[Literal["hash"], _KIND_IS_THE_SUBCOMMAND] = "hash"
    algorithm: Literal["sha256"] = Field(
        default="sha256", description="Hash algorithm. Only 'sha256' is accepted: the digest is always HMAC-SHA256."
    )
    digest_length: int = Field(
        default=12, ge=6, le=64, description="Number of hex characters to keep from the hash digest."
    )
    format_template: str = Field(
        default="<HASH_{label}_{digest}>", description="Template with {digest} required and optional {label}."
    )
    # Required (no default). ``None`` only on a copy made by ``without_key``, which cannot hash.
    key: Annotated[SecretStr | None, cyclopts.Parameter(env_var=HASH_KEY_ENV, parse=False)] = Field(
        description="Required secret key. The digest is HMAC-SHA256 of the case- and whitespace-folded text; "
        "unkeyed digests of low-entropy values are reversible, so there is no unkeyed mode.",
    )

    # Every check below is field-level, or sees input without the key: ``hide_input_in_errors`` hides input only
    # from ``str(error)``, and a model-level error would carry the whole input, key included, in ``errors()``.

    @model_validator(mode="before")
    @classmethod
    def require_key(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        key = data.get("key")
        if key is None:
            raise ValueError(HASH_KEY_REQUIRED_MESSAGE)
        secret = key.get_secret_value() if isinstance(key, SecretStr) else key
        # Swap a rejected key for a marker so the error raised for it carries no key material.
        if isinstance(secret, bytes):
            try:
                secret.decode("utf-8")
            except UnicodeDecodeError:
                return {**data, "key": _UNENCODABLE_KEY}
            return data
        if not isinstance(secret, str):
            return {**data, "key": _NOT_A_STRING_KEY}
        try:
            secret.encode("utf-8")
        except UnicodeEncodeError:
            return {**data, "key": _UNENCODABLE_KEY}
        return data

    @field_validator("key", mode="before")
    @classmethod
    def reject_unencodable_key(cls, value: Any) -> Any:
        if isinstance(value, _RejectedKey):
            raise ValueError(value.message)
        return value

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: SecretStr | None) -> SecretStr:
        if value is None:
            raise ValueError(HASH_KEY_REQUIRED_MESSAGE)
        secret = value.get_secret_value()
        if not secret.strip():
            raise ValueError("key must not be empty or whitespace")
        if set(secret) == {"*"}:
            raise ValueError(
                "key is a masked placeholder; a keyed Hash does not survive serialization, so pass the real key"
            )
        return value

    @field_validator("algorithm", mode="before")
    @classmethod
    def validate_algorithm(cls, value: Any) -> Any:
        if value != "sha256":
            raise ValueError("a keyed Hash always uses HMAC-SHA256; leave algorithm at its default 'sha256'")
        return value

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
            raise ValueError(HASH_KEY_REQUIRED_MESSAGE)
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
