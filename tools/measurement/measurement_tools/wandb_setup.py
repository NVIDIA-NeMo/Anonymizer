# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""W&B setup for Anonymizer benchmark measurement tooling.

Benchmark runners call :func:`initialize_benchmark_wandb_run` explicitly. The
Anonymizer SDK and product CLI do not auto-initialize W&B.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger("measurement.wandb")

DEFAULT_WANDB_PROJECT = "nemo-anonymizer-benchmarks"
WANDB_SANITIZER_VERSION = 1
_WANDB_INSTALL_HINT = "Install the optional measurement dependency group: uv sync --group measurement"
_SENSITIVE_RUN_TAG_PARTS = frozenset({"api_key", "credential", "credentials", "password", "secret", "token"})


class WandbMode(StrEnum):
    online = "online"
    offline = "offline"
    disabled = "disabled"


class WandbSettings(BaseSettings):
    """W&B configuration for benchmark measurement tooling."""

    wandb_mode: WandbMode = Field(
        default=WandbMode.disabled,
        description="Run mode: online, offline, or disabled.",
        validation_alias=AliasChoices(
            "ANONYMIZER_MEASUREMENT_WANDB_MODE",
            "WANDB_MODE",
        ),
    )
    wandb_project: str | None = Field(
        default=None,
        description="W&B project name override.",
        validation_alias=AliasChoices(
            "ANONYMIZER_MEASUREMENT_WANDB_PROJECT",
            "WANDB_PROJECT",
        ),
    )
    wandb_entity: str | None = Field(
        default=None,
        description="W&B entity override.",
        validation_alias=AliasChoices(
            "ANONYMIZER_MEASUREMENT_WANDB_ENTITY",
            "WANDB_ENTITY",
        ),
    )
    wandb_group: str | None = Field(
        default=None,
        description="W&B run group override.",
        validation_alias=AliasChoices(
            "ANONYMIZER_MEASUREMENT_WANDB_GROUP",
            "WANDB_GROUP",
        ),
    )
    wandb_job_type: str | None = Field(
        default=None,
        description="W&B job type override.",
        validation_alias=AliasChoices(
            "ANONYMIZER_MEASUREMENT_WANDB_JOB_TYPE",
            "WANDB_JOB_TYPE",
        ),
    )
    wandb_run_name: str | None = Field(
        default=None,
        description="W&B display run name override.",
        validation_alias=AliasChoices(
            "ANONYMIZER_MEASUREMENT_WANDB_RUN_NAME",
            "WANDB_NAME",
        ),
    )
    wandb_tags: str | None = Field(
        default=None,
        description="Comma-separated W&B run tags.",
        validation_alias=AliasChoices(
            "ANONYMIZER_MEASUREMENT_WANDB_TAGS",
            "WANDB_TAGS",
        ),
    )
    wandb_log_tables: bool = Field(
        default=False,
        description="Upload sanitized measurement tables to W&B.",
        validation_alias=AliasChoices(
            "ANONYMIZER_MEASUREMENT_WANDB_LOG_TABLES",
        ),
    )

    model_config = {"env_prefix": "ANONYMIZER_MEASUREMENT_", "extra": "ignore", "populate_by_name": True}

    @field_validator("wandb_mode", mode="before")
    @classmethod
    def validate_wandb_mode(cls, value: str | WandbMode | None) -> WandbMode:
        if value is None:
            return WandbMode.disabled
        if isinstance(value, WandbMode):
            return value
        return WandbMode(value)

    @property
    def enabled(self) -> bool:
        return self.wandb_mode != WandbMode.disabled

    @property
    def effective_wandb_project(self) -> str:
        return self.wandb_project or DEFAULT_WANDB_PROJECT

    @property
    def effective_wandb_tags(self) -> list[str]:
        if self.wandb_tags is None:
            return []
        return [tag for tag in (part.strip() for part in self.wandb_tags.split(",")) if tag]


def require_wandb() -> Any:
    """Import wandb when W&B logging is enabled."""
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            f"W&B logging is enabled but the wandb package is not installed. {_WANDB_INSTALL_HINT}"
        ) from exc
    return wandb


def initialize_benchmark_wandb_run(
    settings: WandbSettings,
    *,
    suite_id: str,
    output_dir: Path,
    run_tags: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Initialize a W&B run for a benchmark suite when mode is not disabled."""
    if not settings.enabled:
        return False

    wandb = require_wandb()
    if wandb.run is not None:
        logger.info("Reusing existing W&B run: %s", wandb.run.id)
        return False

    project = settings.effective_wandb_project
    logger.info("WANDB_MODE: %s", settings.wandb_mode.value)
    logger.info("WANDB_PROJECT: %s", project)

    wandb_settings = wandb.Settings(console="wrap")
    init_kwargs = {
        "project": project,
        "name": settings.wandb_run_name or _default_run_name(suite_id, metadata),
        "mode": settings.wandb_mode.value,
        "settings": wandb_settings,
        "dir": str(output_dir),
        "group": settings.wandb_group or suite_id,
        "job_type": settings.wandb_job_type or "benchmark",
    }
    if settings.wandb_entity:
        init_kwargs["entity"] = settings.wandb_entity
    tags = _effective_wandb_tags(settings, suite_id=suite_id, metadata=metadata)
    if tags:
        init_kwargs["tags"] = tags
    wandb.init(**init_kwargs)
    _define_benchmark_metrics(wandb)
    config = {
        "suite_id": suite_id,
        "wandb_mode": settings.wandb_mode.value,
        "wandb_log_tables": settings.wandb_log_tables,
    }
    if metadata:
        config.update(_sanitize_wandb_config(metadata))
    if run_tags:
        config["run_tags"] = _scalar_run_tags(run_tags)
    wandb.config.update(config, allow_val_change=True)

    if wandb.run is not None:
        logger.info("W&B run id: %s", wandb.run.id)
    return True


def finish_benchmark_wandb_run() -> None:
    """Finish the active W&B run, if any."""
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    try:
        wandb.finish()
    except Exception as exc:  # noqa: BLE001 -- best-effort cleanup
        logger.warning("Failed to finish W&B run: %s", exc)


def _scalar_run_tags(run_tags: dict[str, Any]) -> dict[str, str | int | float | bool]:
    sanitized: dict[str, str | int | float | bool] = {}
    for key, value in run_tags.items():
        if _run_tag_key_is_sensitive(key):
            continue
        if isinstance(value, bool | int | float):
            sanitized[key] = value
        elif isinstance(value, str) and _run_tag_string_is_safe(value):
            sanitized[key] = value
    return sanitized


def _run_tag_key_is_sensitive(key: str) -> bool:
    normalized = key.lower().replace("-", "_").replace(".", "_")
    parts = {part for part in normalized.split("_") if part}
    if "api" in parts and "key" in parts:
        return True
    return bool(parts & _SENSITIVE_RUN_TAG_PARTS)


def _run_tag_string_is_safe(value: str) -> bool:
    return len(value) <= 256 and "://" not in value and not value.startswith("/")


def _default_run_name(suite_id: str, metadata: dict[str, Any] | None) -> str:
    git = metadata.get("git", {}) if isinstance(metadata, dict) else {}
    if not isinstance(git, dict):
        return suite_id
    commit = git.get("commit")
    branch = git.get("branch")
    if isinstance(commit, str) and commit:
        suffix = commit[:7]
        if isinstance(branch, str) and branch:
            return f"{suite_id} {branch}@{suffix}"
        return f"{suite_id} @{suffix}"
    if isinstance(branch, str) and branch:
        return f"{suite_id} {branch}"
    return suite_id


def _effective_wandb_tags(settings: WandbSettings, *, suite_id: str, metadata: dict[str, Any] | None) -> list[str]:
    tags = list(settings.effective_wandb_tags)
    tags.append(f"suite:{suite_id}")
    git = metadata.get("git", {}) if isinstance(metadata, dict) else {}
    if isinstance(git, dict):
        branch = git.get("branch")
        dirty = git.get("dirty")
        if isinstance(branch, str) and branch:
            tags.append(f"branch:{branch}")
        if isinstance(dirty, bool):
            tags.append("dirty" if dirty else "clean")
    return [tag for tag in tags if _run_tag_string_is_safe(tag)]


def _define_benchmark_metrics(wandb: Any) -> None:
    define_metric = getattr(wandb, "define_metric", None)
    if not callable(define_metric):
        return
    for metric_name in ("benchmark/*", "measurement/*"):
        try:
            define_metric(metric_name, summary="last")
        except Exception as exc:  # noqa: BLE001 -- presentation polish is best-effort
            logger.warning("Failed to define W&B metric %s: %s", metric_name, exc)


def _sanitize_wandb_config(value: Any, *, key: str = "") -> Any:
    if _config_key_is_sensitive(key):
        return None
    if isinstance(value, dict):
        sanitized = {
            item_key: _sanitize_wandb_config(item_value, key=item_key)
            for item_key, item_value in value.items()
            if not _config_key_is_sensitive(item_key)
        }
        return {item_key: item_value for item_key, item_value in sanitized.items() if item_value is not None}
    if isinstance(value, list):
        return [item for item in (_sanitize_wandb_config(item, key=key) for item in value) if item is not None]
    if isinstance(value, str):
        if _config_key_is_path_like(key):
            return None
        return value if _run_tag_string_is_safe(value) else None
    if value is None or isinstance(value, bool | int | float):
        return value
    return str(value)


def _config_key_is_sensitive(key: str) -> bool:
    if not key:
        return False
    normalized = key.lower().replace("-", "_").replace(".", "_")
    parts = {part for part in normalized.split("_") if part}
    if "api" in parts and "key" in parts:
        return True
    return bool(parts & _SENSITIVE_RUN_TAG_PARTS)


def _config_key_is_path_like(key: str) -> bool:
    normalized = key.lower().replace("-", "_").replace(".", "_")
    parts = {part for part in normalized.split("_") if part}
    return any(part in {"path", "url"} or part.endswith(("_path", "_url")) for part in parts)
