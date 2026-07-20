# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolved W&B settings and tag safety."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from enum import StrEnum
from ipaddress import ip_address
from typing import Any
from urllib.parse import urlsplit

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from measurement_tools.validation import RedactedStrictFrozenModel

__all__ = [
    "DEFAULT_WANDB_PROJECT",
    "WANDB_TAG_MAX_LENGTH",
    "ResolvedWandbConfig",
    "WandbInputs",
    "WandbMode",
    "generated_wandb_tag",
    "is_safe_wandb_tag",
    "wandb_tag_value_is_sensitive",
]

DEFAULT_WANDB_PROJECT = "nemo-anonymizer-benchmarks"
WANDB_TAG_MAX_LENGTH = 64
_WANDB_TAG_DIGEST_LENGTH = 12
_SENSITIVE_WANDB_TAG_PARTS = frozenset({"api_key", "credential", "credentials", "password", "secret", "token"})
_SENSITIVE_WANDB_TAG_VALUE_PREFIXES = ("sk-", "ghp_", "github_pat_", "glpat-", "xoxb-", "xoxp-", "xoxa-")


class WandbMode(StrEnum):
    online = "online"
    offline = "offline"
    disabled = "disabled"


class WandbInputs(BaseSettings):
    """Optional operator inputs; dedicated measurement variables only."""

    wandb_mode: WandbMode | None = None
    wandb_project: str | None = None
    wandb_base_url: str | None = None
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_job_type: str | None = None
    wandb_run_name: str | None = None
    wandb_tags: str | None = None
    wandb_log_tables: bool | None = None

    model_config = SettingsConfigDict(
        env_prefix="ANONYMIZER_MEASUREMENT_",
        extra="forbid",
        env_ignore_empty=True,
        populate_by_name=True,
        hide_input_in_errors=True,
    )


class ResolvedWandbConfig(RedactedStrictFrozenModel):
    """Immutable, fully resolved publisher configuration."""

    wandb_mode: WandbMode = WandbMode.disabled
    wandb_project: str = DEFAULT_WANDB_PROJECT
    wandb_base_url: str | None = None
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_job_type: str | None = None
    wandb_run_name: str | None = None
    wandb_tags: str = ""
    wandb_log_tables: bool = False

    @field_validator(
        "wandb_project",
        "wandb_base_url",
        "wandb_entity",
        "wandb_group",
        "wandb_job_type",
        "wandb_run_name",
    )
    @classmethod
    def nonempty_optional_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("W&B string settings cannot be empty")
        return stripped

    @field_validator("wandb_base_url")
    @classmethod
    def validate_base_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("W&B base URL must be a credential-free HTTP(S) URL")
        if parsed.scheme == "http" and not _is_loopback_host(parsed.hostname):
            raise ValueError("W&B base URL must use HTTPS unless it targets loopback")
        return value.rstrip("/")

    @field_validator("wandb_tags")
    @classmethod
    def validate_tags(cls, value: str) -> str:
        tags = [tag for tag in (part.strip() for part in value.split(",")) if tag]
        if any(not is_safe_wandb_tag(tag) for tag in tags):
            raise ValueError("W&B tags must be safe identifiers between 1 and 64 characters")
        return value

    @property
    def enabled(self) -> bool:
        return self.wandb_mode != WandbMode.disabled

    @property
    def effective_wandb_project(self) -> str:
        return self.wandb_project

    @property
    def effective_wandb_tags(self) -> list[str]:
        return [tag for tag in (part.strip() for part in self.wandb_tags.split(",")) if tag]

    @classmethod
    def from_env_and_overrides(
        cls,
        *,
        defaults: Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> ResolvedWandbConfig:
        source = WandbInputs()
        values = dict(defaults or {})
        values.update(source.model_dump(exclude_none=True))
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls.model_validate(values)

    def validated_update(self, **updates: Any) -> ResolvedWandbConfig:
        values = self.model_dump()
        values.update(updates)
        return type(self).model_validate(values)


def _is_loopback_host(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True
    try:
        return ip_address(hostname).is_loopback
    except ValueError:
        return False


def is_safe_wandb_tag(value: str) -> bool:
    if not value or len(value) > WANDB_TAG_MAX_LENGTH or "://" in value or value.startswith("/"):
        return False
    return not wandb_tag_value_is_sensitive(value)


def generated_wandb_tag(namespace: str, value: str) -> str | None:
    candidate = f"{namespace}:{value}"
    if wandb_tag_value_is_sensitive(candidate) or "://" in candidate or candidate.startswith("/"):
        return None
    if len(candidate) <= WANDB_TAG_MAX_LENGTH:
        return candidate
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:_WANDB_TAG_DIGEST_LENGTH]
    prefix_length = WANDB_TAG_MAX_LENGTH - len(namespace) - len(digest) - 2
    return f"{namespace}:{value[:prefix_length]}-{digest}"


def wandb_tag_value_is_sensitive(value: str) -> bool:
    normalized = value.lower()
    if normalized.startswith(_SENSITIVE_WANDB_TAG_VALUE_PREFIXES):
        return True
    parts = {part for part in re.split(r"[^a-z0-9]+", normalized) if part}
    if "api" in parts and "key" in parts:
        return True
    return bool(parts & _SENSITIVE_WANDB_TAG_PARTS)
