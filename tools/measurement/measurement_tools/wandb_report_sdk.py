# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""W&B SDK loading and remote read plumbing for reports and workspaces."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from measurement_tools.wandb_models import ResolvedWandbConfig, WandbMode
from measurement_tools.wandb_report_catalog import all_report_metrics
from measurement_tools.wandb_report_models import (
    WandbProjectPath,
    WandbRunPath,
    WandbRunView,
    parse_wandb_run_view,
)
from measurement_tools.wandb_setup import require_wandb

_WANDB_REPORT_INSTALL_HINT = "Install the optional measurement dependency group: uv sync --group measurement"


def require_wandb_report_sdk() -> tuple[Any, Any, Any]:
    """Import W&B report APIs when report generation is requested."""
    wandb = require_wandb()
    try:
        from wandb.apis.reports import v2 as wr
        from wandb_workspaces import expr
    except ImportError as exc:
        raise ImportError(f"W&B report generation requires wandb[workspaces]. {_WANDB_REPORT_INSTALL_HINT}") from exc
    return wandb, wr, expr


def require_wandb_workspace_sdk() -> tuple[Any, Any]:
    """Import W&B workspace APIs when workspace generation is requested."""
    require_wandb()
    try:
        import wandb_workspaces.reports.v2 as wr
        import wandb_workspaces.workspaces as ws
    except ImportError as exc:
        raise ImportError(f"W&B workspace generation requires wandb[workspaces]. {_WANDB_REPORT_INSTALL_HINT}") from exc
    return ws, wr


def read_group_views(wandb: Any, *, project_path: WandbProjectPath, group: str) -> list[WandbRunView]:
    runs = wandb.Api(timeout=60).runs(project_path.path, filters={"group": group})
    views: list[WandbRunView] = []
    for run in runs:
        try:
            run_id = getattr(run, "id", None)
            if not isinstance(run_id, str):
                raise ValueError("W&B group run is missing a string identity")
            view = parse_wandb_run_view(
                run,
                run_path=WandbRunPath(
                    entity=project_path.entity,
                    project=project_path.project,
                    run_id=run_id,
                    base_url=project_path.base_url,
                ),
                allowed_metrics=frozenset(all_report_metrics()),
            )
        except (ValueError, ValidationError):
            raise ValueError("W&B group contains invalid run metadata") from None
        views.append(view)
    return views


def sweep_param_columns(views: list[WandbRunView]) -> list[str]:
    keys = sorted({key for view in views if view.metadata.sweep is not None for key in view.metadata.sweep.params})
    return [f"config:sweep_param_{key}" for key in keys]


def report_settings(
    settings: ResolvedWandbConfig | None,
    target: WandbRunPath | WandbProjectPath,
) -> ResolvedWandbConfig:
    target_base_url = target.base_url if target.base_url != "https://wandb.ai" else None
    if settings is None:
        return ResolvedWandbConfig.from_env_and_overrides(
            wandb_mode=WandbMode.online,
            wandb_entity=target.entity,
            wandb_project=target.project,
            wandb_base_url=target_base_url,
        )
    if target_base_url is not None and settings.wandb_base_url not in {None, target_base_url}:
        raise ValueError("W&B target URL conflicts with resolved base URL")
    return settings.validated_update(
        wandb_entity=target.entity,
        wandb_project=target.project,
        wandb_base_url=target_base_url or settings.wandb_base_url,
    )


__all__ = [
    "read_group_views",
    "report_settings",
    "require_wandb_report_sdk",
    "require_wandb_workspace_sdk",
    "sweep_param_columns",
]
