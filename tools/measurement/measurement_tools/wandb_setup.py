# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Secure native W&B publishing for Anonymizer benchmark tooling."""

from __future__ import annotations

import logging
import os
import secrets
import stat
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from measurement_tools.wandb_ingress import read_measurement_snapshot
from measurement_tools.wandb_models import (
    DEFAULT_WANDB_PROJECT,
    PUBLICATION_COMPLETE_KEY,
    PUBLICATION_SEAL_DIGEST_KEY,
    BenchmarkMetadata,
    ResolvedWandbConfig,
    WandbConfigPayload,
    WandbInitPayload,
    WandbInputs,
    WandbMode,
    WandbPublishPayload,
    WandbPublishResult,
    WandbRunMetadata,
    generated_wandb_tag,
    is_safe_wandb_tag,
)

__all__ = [
    "DEFAULT_WANDB_PROJECT",
    "ResolvedWandbConfig",
    "WandbInputs",
    "WandbMode",
    "WandbRunMetadata",
]

logger = logging.getLogger("measurement.wandb")

WANDB_SANITIZER_VERSION = 2
_WANDB_INSTALL_HINT = "Install the optional measurement dependency group: uv sync --group measurement"


@dataclass(frozen=True)
class BenchmarkWandbFinalization:
    """Artifacts needed to finalize benchmark W&B logging."""

    measurement_path: Path
    cases: Sequence[Any]


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
            os.environ.update(_publisher_environment(self._settings))
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


def _publisher_environment(settings: ResolvedWandbConfig) -> dict[str, str]:
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
        payload, snapshot_sha256, record_count = _build_publish_payload(
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
                run = wandb.init(**_sdk_init_kwargs(wandb, payload.init))
                if run is None:
                    raise RuntimeError("wandb.init did not return an explicit run handle")
                run_id = str(getattr(run, "id", ""))
                if run_id != payload.init.run_id:
                    raise RuntimeError("wandb.init returned a different run identity")
                already_complete = _publication_already_complete(run, payload)
                if not already_complete:
                    _define_benchmark_metrics(run)
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
            _raise_lifecycle_failures(primary_error, finish_error)
            if result is None:
                raise RuntimeError("W&B publisher completed without a result")
            return result


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
    from measurement_tools.wandb_logging import build_outbound_measurements  # noqa: PLC0415

    cases = list(finalization.cases)
    expected_statuses = {str(case.case_id): str(case.status.value) for case in cases}
    snapshot = read_measurement_snapshot(finalization.measurement_path, expected_statuses=expected_statuses)
    history, summary, tables = build_outbound_measurements(
        snapshot,
        cases=cases,
        log_tables=settings.wandb_log_tables,
    )
    staging_dir = prepare_wandb_staging_dir(output_dir)
    resolved_metadata = metadata or WandbRunMetadata(
        benchmark=BenchmarkMetadata(suite_id=suite_id),
    )
    init = WandbInitPayload(
        run_id=secrets.token_hex(16),
        project=settings.wandb_project,
        name=settings.wandb_run_name or _default_run_name(suite_id, resolved_metadata),
        mode=settings.wandb_mode,
        directory=staging_dir,
        group=settings.wandb_group or suite_id,
        job_type=settings.wandb_job_type or "benchmark",
        entity=settings.wandb_entity,
        tags=tuple(_effective_wandb_tags(settings, suite_id=suite_id, metadata=resolved_metadata)),
    )
    config = WandbConfigPayload.from_run_metadata(settings, suite_id=suite_id, metadata=resolved_metadata)
    payload = WandbPublishPayload(
        init=init,
        config=config,
        history=history,
        summary=summary,
        tables=tables,
    )
    return payload, snapshot.sha256, len(snapshot.records)


def _publication_already_complete(run: Any, payload: WandbPublishPayload) -> bool:
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


def _raise_lifecycle_failures(primary: BaseException | None, finish: BaseException | None) -> None:
    if primary is not None and finish is not None:
        if isinstance(primary, Exception) and isinstance(finish, Exception):
            raise ExceptionGroup("W&B publication and finish both failed", [primary, finish]) from primary
        raise BaseExceptionGroup("W&B publication and finish both failed", [primary, finish]) from primary
    if primary is not None:
        raise primary.with_traceback(primary.__traceback__)
    if finish is not None:
        raise finish.with_traceback(finish.__traceback__)


def _sdk_init_kwargs(wandb: Any, payload: WandbInitPayload) -> dict[str, Any]:
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


def prepare_wandb_staging_dir(output_dir: Path) -> Path:
    output_descriptor = _open_directory_no_follow(output_dir)
    staging_dir = output_dir / ".wandb-private"
    try:
        try:
            os.mkdir(".wandb-private", mode=0o700, dir_fd=output_descriptor)
        except FileExistsError:
            pass
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
        staging_descriptor = os.open(
            ".wandb-private",
            flags | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=output_descriptor,
        )
        try:
            metadata = os.fstat(staging_descriptor)
            if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
                raise ValueError("W&B staging directory must be an owned directory")
            os.fchmod(staging_descriptor, 0o700)
        finally:
            os.close(staging_descriptor)
    except OSError as exc:
        raise ValueError("W&B staging directory cannot contain symlinks or special files") from exc
    finally:
        os.close(output_descriptor)
    return staging_dir


def _open_directory_no_follow(path: Path) -> int:
    absolute = path.absolute()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute.anchor, flags)
        for part in absolute.parts[1:]:
            child = os.open(part, flags | no_follow, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
            _validate_directory_metadata(os.fstat(descriptor), final=part == absolute.name)
        return descriptor
    except (OSError, ValueError) as exc:
        if "descriptor" in locals():
            os.close(descriptor)
        raise ValueError("W&B output path cannot contain symlinks, untrusted directories, or special files") from exc


def _validate_directory_metadata(metadata: os.stat_result, *, final: bool) -> None:
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("W&B output path must contain only directories")
    if final and metadata.st_uid != os.geteuid():
        raise ValueError("W&B output directory must be owned by the current user")
    writable_by_others = metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    if writable_by_others and not metadata.st_mode & stat.S_ISVTX:
        raise ValueError("W&B output path contains an untrusted writable directory")


def _default_run_name(suite_id: str, metadata: WandbRunMetadata | None) -> str:
    git = metadata.git if metadata is not None else None
    if git is None:
        return suite_id
    commit = git.commit
    branch = git.branch
    if isinstance(commit, str) and commit:
        suffix = commit[:7]
        if isinstance(branch, str) and branch:
            return f"{suite_id} {branch}@{suffix}"
        return f"{suite_id} @{suffix}"
    if isinstance(branch, str) and branch:
        return f"{suite_id} {branch}"
    return suite_id


def _effective_wandb_tags(
    settings: ResolvedWandbConfig,
    *,
    suite_id: str,
    metadata: WandbRunMetadata | None,
) -> list[str]:
    tags = [tag for tag in settings.effective_wandb_tags if is_safe_wandb_tag(tag)]
    suite_tag = generated_wandb_tag("suite", suite_id)
    if suite_tag is not None:
        tags.append(suite_tag)
    git = metadata.git if metadata is not None else None
    if git is not None:
        branch = git.branch
        dirty = git.dirty
        if isinstance(branch, str) and branch:
            branch_tag = generated_wandb_tag("branch", branch)
            if branch_tag is not None:
                tags.append(branch_tag)
        if isinstance(dirty, bool):
            tags.append("dirty" if dirty else "clean")
    return tags


def _define_benchmark_metrics(run: Any) -> None:
    define_metric = getattr(run, "define_metric", None)
    if not callable(define_metric):
        return
    for metric_name in ("benchmark/*", "measurement/*"):
        try:
            define_metric(metric_name, summary="last")
        except Exception as exc:  # noqa: BLE001 -- presentation polish is best-effort
            logger.warning("Failed to define W&B metric %s (%s)", metric_name, type(exc).__name__)
