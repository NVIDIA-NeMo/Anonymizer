# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""W&B presentation operations for completed benchmark sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from measurement_tools.benchmark_sweep_models import SweepSpec
from measurement_tools.wandb_report_models import WandbProjectPath
from measurement_tools.wandb_settings import ResolvedWandbConfig


class WandbViewCreationError(RuntimeError):
    """A post-benchmark W&B presentation operation failed."""


@dataclass(frozen=True)
class WandbViewOperation:
    enabled: bool
    label: str
    url_attribute: str
    create: Callable[..., Any]


def create_view(
    spec: SweepSpec,
    *,
    wandb_settings: ResolvedWandbConfig,
    dry_run: bool,
    operation: WandbViewOperation,
) -> str | None:
    if not operation.enabled or dry_run or not wandb_settings.enabled:
        return None
    entity = wandb_settings.wandb_entity
    if not entity:
        return None
    try:
        project = WandbProjectPath(entity=entity, project=wandb_settings.effective_wandb_project)
        result = operation.create(
            project,
            settings=wandb_settings,
            group=wandb_settings.wandb_group or spec.sweep_id,
            expected_run_kind="sweep_arm",
        )
    except Exception as exc:  # noqa: BLE001 -- remote SDK errors must not expose response contents
        raise WandbViewCreationError(f"W&B {operation.label} creation failed ({type(exc).__name__})") from None
    return getattr(result, operation.url_attribute)


__all__ = ["WandbViewCreationError", "WandbViewOperation", "create_view"]
