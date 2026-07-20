# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility facade for secure native W&B publication."""

from __future__ import annotations

import logging
import os
import stat as stat
from pathlib import Path
from typing import Any, Callable

from measurement_tools import wandb_sdk_environment as _sdk_environment
from measurement_tools.wandb_models import DEFAULT_WANDB_PROJECT as DEFAULT_WANDB_PROJECT
from measurement_tools.wandb_models import BenchmarkMetadata as BenchmarkMetadata
from measurement_tools.wandb_models import ResolvedWandbConfig as ResolvedWandbConfig
from measurement_tools.wandb_models import WandbConfigPayload as WandbConfigPayload
from measurement_tools.wandb_models import WandbInitPayload as WandbInitPayload
from measurement_tools.wandb_models import WandbInputs as WandbInputs
from measurement_tools.wandb_models import WandbMode as WandbMode
from measurement_tools.wandb_models import WandbPublishPayload as WandbPublishPayload
from measurement_tools.wandb_models import WandbPublishResult as WandbPublishResult
from measurement_tools.wandb_models import WandbRunMetadata as WandbRunMetadata
from measurement_tools.wandb_payload import BenchmarkWandbFinalization as BenchmarkWandbFinalization
from measurement_tools.wandb_payload import build_publish_payload
from measurement_tools.wandb_publisher import WandbPublisher as WandbPublisher
from measurement_tools.wandb_publisher import (
    define_benchmark_metrics,
    publication_already_complete,
    publication_state,
    raise_lifecycle_failures,
    sdk_init_kwargs,
    wandb_run_url,
)
from measurement_tools.wandb_run_identity import default_run_name, effective_wandb_tags
from measurement_tools.wandb_sdk_environment import WandbSdkEnvironment as WandbSdkEnvironment
from measurement_tools.wandb_staging import open_directory_no_follow, validate_directory_metadata
from measurement_tools.wandb_staging import prepare_wandb_staging_dir as _prepare_wandb_staging_dir

__all__ = [
    "DEFAULT_WANDB_PROJECT",
    "ResolvedWandbConfig",
    "WandbInputs",
    "WandbMode",
    "WandbRunMetadata",
]

logger = logging.getLogger("measurement.wandb")

WANDB_SANITIZER_VERSION = 2
_WANDB_INSTALL_HINT = _sdk_environment._WANDB_INSTALL_HINT
_PUBLISHER_WANDB_MODULE: Any | None = _sdk_environment._PUBLISHER_WANDB_MODULE
_WANDB_AMBIENT_ALLOWLIST = _sdk_environment._WANDB_AMBIENT_ALLOWLIST
_PROCESS_ENVIRONMENT_ALLOWLIST = _sdk_environment._PROCESS_ENVIRONMENT_ALLOWLIST
_WANDB_ENVIRONMENT_LOCK = _sdk_environment._WANDB_ENVIRONMENT_LOCK
_WANDB_ENVIRONMENT_OWNER = _sdk_environment._WANDB_ENVIRONMENT_OWNER

_default_run_name = default_run_name
_effective_wandb_tags = effective_wandb_tags
_publisher_environment = _sdk_environment.publisher_environment
_publication_already_complete = publication_already_complete
_publication_state = publication_state
_wandb_run_url = wandb_run_url
_raise_lifecycle_failures = raise_lifecycle_failures
_sdk_init_kwargs = sdk_init_kwargs
_define_benchmark_metrics = define_benchmark_metrics


def require_wandb() -> Any:
    """Use the canonical SDK loader while preserving the legacy patch sentinel."""
    global _PUBLISHER_WANDB_MODULE  # noqa: PLW0603

    _sdk_environment._PUBLISHER_WANDB_MODULE = _PUBLISHER_WANDB_MODULE
    try:
        return _sdk_environment.require_wandb()
    finally:
        _PUBLISHER_WANDB_MODULE = _sdk_environment._PUBLISHER_WANDB_MODULE


def publish_benchmark_wandb_best_effort(
    settings: ResolvedWandbConfig,
    *,
    suite_id: str,
    output_dir: Path,
    finalization: BenchmarkWandbFinalization,
    metadata: WandbRunMetadata | None = None,
    metadata_factory: Callable[[], WandbRunMetadata] | None = None,
) -> WandbPublishResult:
    """Publish after native execution without changing benchmark status."""
    if not settings.enabled:
        return WandbPublishResult(published=False)
    try:
        if metadata is not None and metadata_factory is not None:
            raise ValueError("provide metadata or metadata_factory, not both")
        resolved_metadata = metadata_factory() if metadata_factory is not None else metadata
        return WandbPublisher().publish(
            settings,
            suite_id=suite_id,
            output_dir=output_dir,
            finalization=finalization,
            metadata=resolved_metadata,
        )
    except Exception as exc:  # noqa: BLE001 -- native observability is explicitly best-effort
        logger.warning("Failed to publish benchmark measurements to W&B (%s)", type(exc).__name__)
        return WandbPublishResult(published=False)


def _build_publish_payload(
    settings: ResolvedWandbConfig,
    *,
    suite_id: str,
    output_dir: Path,
    finalization: BenchmarkWandbFinalization,
    metadata: WandbRunMetadata | None,
) -> tuple[WandbPublishPayload, str, int]:
    return build_publish_payload(
        settings,
        suite_id=suite_id,
        output_dir=output_dir,
        finalization=finalization,
        metadata=metadata,
    )


def prepare_wandb_staging_dir(output_dir: Path) -> Path:
    """Prepare secure W&B staging through the canonical filesystem owner."""
    return _prepare_wandb_staging_dir(output_dir)


def _open_directory_no_follow(path: Path) -> int:
    """Compatibility alias for the canonical no-follow traversal."""
    return open_directory_no_follow(path)


def _validate_directory_metadata(metadata: os.stat_result, *, final: bool) -> None:
    """Compatibility alias for canonical directory metadata validation."""
    validate_directory_metadata(metadata, final=final)
