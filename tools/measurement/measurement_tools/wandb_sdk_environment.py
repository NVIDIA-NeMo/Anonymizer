# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guarded process environment and SDK loading for W&B operations."""

from __future__ import annotations

import os
import sys
import threading
from typing import Any

from measurement_tools.wandb_settings import ResolvedWandbConfig

__all__ = ["WandbSdkEnvironment", "publisher_environment", "require_wandb"]

_WANDB_INSTALL_HINT = "Install the optional measurement dependency group: uv sync --group measurement"
_PUBLISHER_WANDB_MODULE: Any | None = None


def require_wandb() -> Any:
    """Import wandb inside the publisher's guarded SDK environment."""
    global _PUBLISHER_WANDB_MODULE  # noqa: PLW0603

    if _WANDB_ENVIRONMENT_OWNER != threading.get_ident():
        raise RuntimeError("wandb must be imported inside the guarded SDK environment")
    preloaded = sys.modules.get("wandb")
    if _PUBLISHER_WANDB_MODULE is None and preloaded is not None:
        raise RuntimeError("wandb must not be imported before the guarded publisher")
    if _PUBLISHER_WANDB_MODULE is not None and preloaded is not _PUBLISHER_WANDB_MODULE:
        raise RuntimeError("the loaded wandb module changed outside the guarded publisher")
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            f"W&B logging is enabled but the wandb package is not installed. {_WANDB_INSTALL_HINT}"
        ) from exc
    _PUBLISHER_WANDB_MODULE = wandb
    return wandb


_WANDB_AMBIENT_ALLOWLIST = frozenset(
    {
        "WANDB_HTTP_TIMEOUT",
        "WANDB_INIT_TIMEOUT",
        "WANDB__SERVICE_WAIT",
    }
)
_PROCESS_ENVIRONMENT_ALLOWLIST = frozenset(
    {
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "PATH",
        "TEMP",
        "TMP",
        "TMPDIR",
        "TZ",
    }
)
_WANDB_ENVIRONMENT_LOCK = threading.Lock()
_WANDB_ENVIRONMENT_OWNER: int | None = None


class WandbSdkEnvironment:
    """Process-wide exact environment transaction around W&B SDK use."""

    def __init__(self, settings: ResolvedWandbConfig) -> None:
        self._settings = settings
        self._snapshot: dict[str, str] | None = None

    def __enter__(self) -> WandbSdkEnvironment:
        global _WANDB_ENVIRONMENT_OWNER  # noqa: PLW0603

        if not _WANDB_ENVIRONMENT_LOCK.acquire(blocking=False):
            raise RuntimeError("nested or concurrent W&B publisher use is not allowed")
        _WANDB_ENVIRONMENT_OWNER = threading.get_ident()
        self._snapshot = dict(os.environ)
        try:
            preserved = {
                key: value
                for key, value in self._snapshot.items()
                if key in _PROCESS_ENVIRONMENT_ALLOWLIST or key in _WANDB_AMBIENT_ALLOWLIST
            }
            os.environ.clear()
            os.environ.update(preserved)
            os.environ.update(publisher_environment(self._settings))
        except BaseException:
            self._restore()
            raise
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self._restore()

    def _restore(self) -> None:
        global _WANDB_ENVIRONMENT_OWNER  # noqa: PLW0603

        if self._snapshot is None:
            return
        os.environ.clear()
        os.environ.update(self._snapshot)
        self._snapshot = None
        _WANDB_ENVIRONMENT_OWNER = None
        _WANDB_ENVIRONMENT_LOCK.release()


def publisher_environment(settings: ResolvedWandbConfig) -> dict[str, str]:
    environment = {
        "WANDB_ERROR_REPORTING": "false",
        "WANDB_MODE": settings.wandb_mode.value,
        "WANDB_PROJECT": settings.wandb_project,
        "WANDB_SILENT": "true",
    }
    optional = {
        "WANDB_BASE_URL": settings.wandb_base_url,
        "WANDB_ENTITY": settings.wandb_entity,
        "WANDB_GROUP": settings.wandb_group,
        "WANDB_JOB_TYPE": settings.wandb_job_type,
        "WANDB_NAME": settings.wandb_run_name,
        "WANDB_TAGS": settings.wandb_tags or None,
    }
    environment.update({key: value for key, value in optional.items() if value is not None})
    return environment
