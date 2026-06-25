# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""W&B setup for Anonymizer benchmark measurement tooling.

Benchmark runners call :func:`initialize_benchmark_wandb_run` explicitly. The
Anonymizer SDK and product CLI do not auto-initialize W&B.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
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
_SENSITIVE_RUN_TAG_VALUE_PREFIXES = ("sk-", "ghp_", "github_pat_", "glpat-", "xoxb-", "xoxp-", "xoxa-")


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


@dataclass(frozen=True)
class BenchmarkWandbFinalization:
    """Artifacts needed to finalize benchmark W&B logging."""

    measurement_path: Path
    cases: Sequence[Any]
    table_dir: Path | None


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

    init_kwargs = _wandb_init_kwargs(wandb, settings, suite_id=suite_id, output_dir=output_dir, metadata=metadata)
    try:
        wandb.init(**init_kwargs)
        _define_benchmark_metrics(wandb)
        wandb.config.update(
            _benchmark_wandb_config(settings, suite_id=suite_id, run_tags=run_tags, metadata=metadata),
            allow_val_change=True,
        )
    except Exception:
        _finish_wandb_run(wandb)
        raise

    if wandb.run is not None:
        logger.info("W&B run id: %s", wandb.run.id)
    return True


def _wandb_init_kwargs(
    wandb: Any,
    settings: WandbSettings,
    *,
    suite_id: str,
    output_dir: Path,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    init_kwargs = {
        "project": settings.effective_wandb_project,
        "name": settings.wandb_run_name or _default_run_name(suite_id, metadata),
        "mode": settings.wandb_mode.value,
        "settings": wandb.Settings(console="wrap"),
        "dir": str(output_dir),
        "group": settings.wandb_group or suite_id,
        "job_type": settings.wandb_job_type or "benchmark",
    }
    if settings.wandb_entity:
        init_kwargs["entity"] = settings.wandb_entity
    tags = _effective_wandb_tags(settings, suite_id=suite_id, metadata=metadata)
    if tags:
        init_kwargs["tags"] = tags
    return init_kwargs


def _benchmark_wandb_config(
    settings: WandbSettings,
    *,
    suite_id: str,
    run_tags: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    config = {
        "suite_id": suite_id,
        "wandb_mode": settings.wandb_mode.value,
        "wandb_log_tables": settings.wandb_log_tables,
    }
    if metadata:
        sanitized_metadata = _sanitize_wandb_config(metadata)
        config.update(_wandb_comparer_config(sanitized_metadata))
        config.update(sanitized_metadata)
    if run_tags:
        config["run_tags"] = _scalar_run_tags(run_tags)
    return config


def finish_benchmark_wandb_run() -> None:
    """Finish the active W&B run, if any."""
    try:
        import wandb
    except ImportError:
        return
    _finish_wandb_run(wandb)


def _finish_wandb_run(wandb: Any) -> None:
    """Best-effort W&B run finalization for already-imported modules."""
    if wandb.run is None:
        return
    try:
        wandb.finish()
    except Exception as exc:  # noqa: BLE001 -- best-effort cleanup
        logger.warning("Failed to finish W&B run: %s", exc)


def finalize_benchmark_wandb_run(
    settings: WandbSettings,
    *,
    finalization: BenchmarkWandbFinalization,
    run_created: bool,
    wandb: Any | None = None,
) -> None:
    """Log benchmark measurements and finish the W&B run created by benchmark tooling."""
    try:
        if settings.enabled:
            active_wandb = wandb if wandb is not None else require_wandb()
            from measurement_tools.wandb_logging import log_benchmark_measurements  # noqa: PLC0415

            log_benchmark_measurements(
                active_wandb,
                measurement_path=finalization.measurement_path,
                cases=list(finalization.cases),
                log_tables=settings.wandb_log_tables,
                table_dir=finalization.table_dir,
            )
    finally:
        if run_created:
            finish_benchmark_wandb_run()


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
    if len(value) > 256 or "://" in value or value.startswith("/"):
        return False
    return not _run_tag_value_is_sensitive(value)


def _run_tag_value_is_sensitive(value: str) -> bool:
    normalized = value.lower()
    if normalized.startswith(_SENSITIVE_RUN_TAG_VALUE_PREFIXES):
        return True
    parts = {part for part in re.split(r"[^a-z0-9]+", normalized) if part}
    if "api" in parts and "key" in parts:
        return True
    return bool(parts & _SENSITIVE_RUN_TAG_PARTS)


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


def _wandb_comparer_config(metadata: Any) -> dict[str, str | int | float | bool]:
    """Build flat W&B config fields for Run Comparer and workspace panels."""
    source = _metadata_dict(metadata)
    benchmark = _metadata_dict(source.get("benchmark"))
    workloads = _metadata_list(source.get("workloads"))
    configs = _metadata_list(source.get("configs"))
    flat: dict[str, str | int | float | bool] = {}
    _set_flat(flat, "benchmark_suite_id", benchmark.get("suite_id"))
    _set_flat(flat, "benchmark_case_count", benchmark.get("case_count"))
    _set_flat(flat, "benchmark_workload_ids", _compact_values(item.get("id") for item in workloads))
    _set_flat(flat, "benchmark_workload_row_limits", _compact_values(item.get("row_limit") for item in workloads))
    _set_flat(
        flat,
        "benchmark_workload_source_kinds",
        _compact_values(_metadata_dict(item.get("source")).get("kind") for item in workloads),
    )
    _set_flat(
        flat,
        "benchmark_workload_source_suffixes",
        _compact_values(_metadata_dict(item.get("source")).get("suffix") for item in workloads),
    )
    _set_flat(flat, "benchmark_config_ids", _compact_values(item.get("id") for item in configs))
    _set_flat(flat, "benchmark_modes", _compact_values(item.get("mode") for item in configs))
    _set_flat(flat, "benchmark_strategies", _compact_values(_config_strategy(item) for item in configs))
    _set_flat(
        flat,
        "benchmark_gliner_thresholds",
        _compact_values(_detect_value(item, "gliner_threshold") for item in configs),
    )
    _set_flat(
        flat,
        "benchmark_entity_label_counts",
        _compact_values(_detect_value(item, "entity_label_count") for item in configs),
    )
    _set_flat(
        flat, "benchmark_risk_tolerances", _compact_values(_rewrite_value(item, "risk_tolerance") for item in configs)
    )
    return flat


def _set_flat(mapping: dict[str, str | int | float | bool], key: str, value: Any) -> None:
    if isinstance(value, bool | int | float | str):
        mapping[key] = value


def _compact_values(values: Any) -> str | int | float | bool | None:
    compacted: list[str | int | float | bool] = []
    seen: set[tuple[str, str | int | float | bool]] = set()
    for value in values:
        if not isinstance(value, bool | int | float | str):
            continue
        marker = (type(value).__name__, value)
        if marker in seen:
            continue
        compacted.append(value)
        seen.add(marker)
    if len(compacted) == 1:
        return compacted[0]
    if len(compacted) > 1:
        return ",".join(str(value) for value in compacted)
    return None


def _config_strategy(config: dict[str, Any]) -> str | None:
    replace = _metadata_dict(config.get("replace"))
    rewrite = _metadata_dict(config.get("rewrite"))
    strategy = replace.get("strategy")
    if isinstance(strategy, str):
        return strategy
    return "rewrite" if rewrite else None


def _detect_value(config: dict[str, Any], key: str) -> Any:
    return _metadata_dict(config.get("detect")).get(key)


def _rewrite_value(config: dict[str, Any], key: str) -> Any:
    return _metadata_dict(config.get("rewrite")).get(key)


def _metadata_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _metadata_list(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


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
