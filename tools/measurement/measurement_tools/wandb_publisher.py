# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict W&B publication lifecycle and remote-state handling."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib.parse import quote

from measurement_tools.wandb_metadata import WandbRunMetadata
from measurement_tools.wandb_payload import BenchmarkWandbFinalization, build_publish_payload
from measurement_tools.wandb_publication import (
    PUBLICATION_COMPLETE_KEY,
    PUBLICATION_SEAL_DIGEST_KEY,
    WandbInitPayload,
    WandbPublicationState,
    WandbPublishPayload,
    WandbPublishResult,
)
from measurement_tools.wandb_sdk_environment import WandbSdkEnvironment, require_wandb
from measurement_tools.wandb_settings import ResolvedWandbConfig, WandbMode

__all__ = [
    "WandbPublisher",
    "define_benchmark_metrics",
    "publication_already_complete",
    "publication_state",
    "raise_lifecycle_failures",
    "sdk_init_kwargs",
    "wandb_run_url",
]

logger = logging.getLogger("measurement.wandb")


class WandbPublisher:
    """Own the complete strict native W&B publication lifecycle."""

    def publish(
        self,
        settings: ResolvedWandbConfig,
        *,
        suite_id: str,
        output_dir: Path,
        finalization: BenchmarkWandbFinalization,
        metadata: WandbRunMetadata | None = None,
    ) -> WandbPublishResult:
        if not settings.enabled:
            return WandbPublishResult(published=False)
        payload, snapshot_sha256, record_count = build_publish_payload(
            settings,
            suite_id=suite_id,
            output_dir=output_dir,
            finalization=finalization,
            metadata=metadata,
        )
        return self.publish_payload(
            settings,
            payload=payload,
            measurement_sha256=snapshot_sha256,
            record_count=record_count,
        )

    def publish_payload(
        self,
        settings: ResolvedWandbConfig,
        *,
        payload: WandbPublishPayload,
        measurement_sha256: str,
        record_count: int,
    ) -> WandbPublishResult:
        """Publish a complete typed payload without access to source artifacts."""
        if not settings.enabled:
            return WandbPublishResult(published=False)
        with WandbSdkEnvironment(settings):
            wandb = require_wandb()
            if getattr(wandb, "run", None) is not None:
                raise RuntimeError("an ambient W&B run is active")
            run: Any | None = None
            result: WandbPublishResult | None = None
            primary_error: BaseException | None = None
            try:
                run = wandb.init(**sdk_init_kwargs(wandb, payload.init))
                if run is None:
                    raise RuntimeError("wandb.init did not return an explicit run handle")
                run_id = str(getattr(run, "id", ""))
                if run_id != payload.init.run_id:
                    raise RuntimeError("wandb.init returned a different run identity")
                already_complete = publication_already_complete(run, payload)
                resolved_publication_state = publication_state(run, already_complete=already_complete)
                if not already_complete:
                    define_benchmark_metrics(run)
                    run.config.update(payload.config.sdk_values(), allow_val_change=True)
                    tables = {
                        table.name: wandb.Table(columns=list(table.columns), data=[list(row) for row in table.data])
                        for table in payload.tables
                    }
                    logged = {**payload.history.metrics, **tables}
                    if payload.init.resume == "allow":
                        run.log(logged, step=0)
                    else:
                        run.log(logged)
                    run.summary.update(payload.summary.metrics)
                logger.info("W&B run id: %s", run_id)
                result = WandbPublishResult(
                    published=True,
                    run_id=run_id,
                    entity=settings.wandb_entity,
                    project=settings.wandb_project,
                    run_url=wandb_run_url(settings, run_id=run_id),
                    publication_state=resolved_publication_state,
                    measurement_sha256=measurement_sha256,
                    record_count=record_count,
                )
            except BaseException as exc:
                primary_error = exc
            finish_error: BaseException | None = None
            if run is not None:
                try:
                    run.finish()
                except BaseException as exc:
                    finish_error = exc
            raise_lifecycle_failures(primary_error, finish_error)
            if result is None:
                raise RuntimeError("W&B publisher completed without a result")
            return result


def publication_already_complete(run: Any, payload: WandbPublishPayload) -> bool:
    if payload.init.resume != "allow":
        return False
    expected_digest = payload.summary.metrics[PUBLICATION_SEAL_DIGEST_KEY]
    summary = getattr(run, "summary", None)
    get_value = getattr(summary, "get", None)
    config = getattr(run, "config", None)
    get_config = getattr(config, "get", None)
    if not callable(get_value) or not callable(get_config):
        raise RuntimeError("resumed W&B run does not expose readable publication state")
    observed_complete = get_value(PUBLICATION_COMPLETE_KEY)
    observed_digest = get_value(PUBLICATION_SEAL_DIGEST_KEY)
    observed_imported = get_config("imported")
    observed_config_digest = (
        observed_imported.get("completion_seal_sha256") if isinstance(observed_imported, dict) else None
    )
    if observed_config_digest not in {None, expected_digest}:
        raise RuntimeError("resumed W&B run config contains different sealed content")
    if observed_complete is True:
        if observed_digest != expected_digest or observed_config_digest != expected_digest:
            raise RuntimeError("resumed W&B run is complete for different sealed content")
        return True
    if observed_complete not in {None, False}:
        raise RuntimeError("resumed W&B run has an invalid publication marker")
    if observed_digest not in {None, expected_digest}:
        raise RuntimeError("resumed W&B run contains different sealed content")
    return False


def publication_state(run: Any, *, already_complete: bool) -> WandbPublicationState:
    if already_complete:
        return WandbPublicationState.already_complete
    return WandbPublicationState.resumed if bool(getattr(run, "resumed", False)) else WandbPublicationState.created


def wandb_run_url(settings: ResolvedWandbConfig, *, run_id: str) -> str | None:
    if settings.wandb_mode != WandbMode.online or settings.wandb_entity is None:
        return None
    base_url = settings.wandb_base_url or "https://wandb.ai"
    if base_url == "https://api.wandb.ai":
        base_url = "https://wandb.ai"
    segments = (settings.wandb_entity, settings.wandb_project, "runs", run_id)
    return f"{base_url}/{'/'.join(quote(segment, safe='') for segment in segments)}"


def raise_lifecycle_failures(primary: BaseException | None, finish: BaseException | None) -> None:
    if primary is not None and finish is not None:
        if isinstance(primary, Exception) and isinstance(finish, Exception):
            raise ExceptionGroup("W&B publication and finish both failed", [primary, finish]) from primary
        raise BaseExceptionGroup("W&B publication and finish both failed", [primary, finish]) from primary
    if primary is not None:
        raise primary.with_traceback(primary.__traceback__)
    if finish is not None:
        raise finish.with_traceback(finish.__traceback__)


def sdk_init_kwargs(wandb: Any, payload: WandbInitPayload) -> dict[str, Any]:
    values: dict[str, Any] = {
        "project": payload.project,
        "id": payload.run_id,
        "resume": payload.resume,
        "name": payload.name,
        "mode": payload.mode.value,
        "settings": wandb.Settings(
            console="off",
            disable_code=True,
            disable_git=True,
            host="redacted",
            save_code=False,
            x_disable_machine_info=True,
            x_disable_meta=True,
            x_disable_stats=True,
            x_save_requirements=False,
        ),
        "dir": str(payload.directory),
        "group": payload.group,
        "job_type": payload.job_type,
    }
    if payload.entity is not None:
        values["entity"] = payload.entity
    if payload.tags:
        values["tags"] = list(payload.tags)
    return values


def define_benchmark_metrics(run: Any) -> None:
    define_metric = getattr(run, "define_metric", None)
    if not callable(define_metric):
        return
    for metric_name in ("benchmark/*", "measurement/*"):
        try:
            define_metric(metric_name, summary="last")
        except Exception as exc:  # noqa: BLE001 -- presentation polish is best-effort
            logger.warning("Failed to define W&B metric %s (%s)", metric_name, type(exc).__name__)
