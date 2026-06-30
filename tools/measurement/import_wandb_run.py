#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strictly import one sealed external benchmark case into W&B."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import cyclopts
from measurement_tools.cli import LogFormat, configure_logging, log_bad_input, summarize_validation_error
from measurement_tools.wandb_completion import (
    COMPLETION_SEAL_FILENAME,
    CompletionSealSnapshot,
    read_completion_seal,
    verify_completion_seal,
)
from measurement_tools.wandb_ingress import MeasurementSnapshot, read_measurement_snapshot
from measurement_tools.wandb_logging import build_outbound_measurements
from measurement_tools.wandb_models import (
    PUBLICATION_COMPLETE_KEY,
    PUBLICATION_SEAL_DIGEST_KEY,
    BenchmarkMetadata,
    ConfigMetadata,
    ExecutionMetadata,
    GitMetadata,
    ImportedRunMetadata,
    MatrixMetadata,
    ResolvedWandbConfig,
    SlurmMetadata,
    WandbConfigPayload,
    WandbInitPayload,
    WandbMode,
    WandbPublishPayload,
    WandbPublishResult,
    WandbRunMetadata,
    WorkloadMetadata,
)
from measurement_tools.wandb_setup import (
    WANDB_SANITIZER_VERSION,
    WandbPublisher,
    _effective_wandb_tags,
    prepare_wandb_staging_dir,
)
from pydantic import ValidationError

app = cyclopts.App(help=__doc__)
logger = logging.getLogger("measurement.wandb_import")

IMPORT_RUN_ID_NAMESPACE_VERSION = 1
WANDB_METADATA_SCHEMA_VERSION = 2


class _ImportedStatus(StrEnum):
    completed = "completed"


@dataclass(frozen=True)
class _ImportedCaseSummary:
    status: _ImportedStatus
    elapsed_sec: float


@dataclass(frozen=True)
class PreparedWandbImport:
    payload: WandbPublishPayload
    measurement_sha256: str
    record_count: int


class ImportInputError(ValueError):
    """A sealed import input failed before any W&B SDK operation."""


def stable_import_run_id(
    settings: ResolvedWandbConfig,
    *,
    seal_snapshot: CompletionSealSnapshot,
) -> str:
    """Derive one destination-scoped 128-bit identity for a sealed case."""
    seal = seal_snapshot.seal
    identity = {
        "namespace_version": IMPORT_RUN_ID_NAMESPACE_VERSION,
        "sanitizer_version": WANDB_SANITIZER_VERSION,
        "destination": {
            "base_url": settings.wandb_base_url or "https://api.wandb.ai",
            "entity": settings.wandb_entity,
            "project": settings.wandb_project,
        },
        "case": seal.case.model_dump(mode="json"),
        "completion_seal_sha256": seal_snapshot.sha256,
    }
    canonical = json.dumps(identity, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()[:32]


def build_import_metadata(seal_snapshot: CompletionSealSnapshot) -> WandbRunMetadata:
    seal = seal_snapshot.seal
    return WandbRunMetadata(
        run_kind="imported_case",
        benchmark=BenchmarkMetadata(
            metadata_schema_version=WANDB_METADATA_SCHEMA_VERSION,
            wandb_sanitizer_version=WANDB_SANITIZER_VERSION,
            measurement_schema_version=seal.measurement_schema_version,
            suite_id=seal.case.case_id,
            workload_count=1,
            config_count=1,
            matrix_entry_count=1,
            case_count=1,
        ),
        execution=ExecutionMetadata(
            backend="slurm",
            export=True,
            slurm=SlurmMetadata(
                job_id=seal.slurm.job_id,
                array_job_id=seal.slurm.array_job_id,
                array_task_id=seal.slurm.array_task_id,
            ),
        ),
        git=GitMetadata(commit=seal.producer.commit),
        workloads=(WorkloadMetadata(id=seal.case.workload_id),),
        configs=(ConfigMetadata(id=seal.case.config_id),),
        matrix=(
            MatrixMetadata(
                workload=seal.case.workload_id,
                config=seal.case.config_id,
                repetitions=1,
            ),
        ),
        imported=ImportedRunMetadata(
            completion_seal_schema_version=seal.seal_schema_version,
            completion_seal_sha256=seal_snapshot.sha256,
            producer_repository=seal.producer.repository,
            producer_commit=seal.producer.commit,
            phase=seal.slurm.phase,
            case_id=seal.case.case_id,
        ),
    )


def build_import_payload(
    settings: ResolvedWandbConfig,
    *,
    snapshot: MeasurementSnapshot,
    seal_snapshot: CompletionSealSnapshot,
    staging_dir: Path,
) -> WandbPublishPayload:
    """Construct the complete SDK-bound payload before W&B initialization."""
    verify_completion_seal(snapshot, seal_snapshot.seal)
    metadata = build_import_metadata(seal_snapshot)
    terminal = snapshot.terminal_stage()
    history, summary, tables = build_outbound_measurements(
        snapshot,
        cases=[_ImportedCaseSummary(status=_ImportedStatus.completed, elapsed_sec=terminal.elapsed_sec)],
        log_tables=settings.wandb_log_tables,
    )
    case = seal_snapshot.seal.case
    run_id = stable_import_run_id(settings, seal_snapshot=seal_snapshot)
    summary = type(summary).model_validate(
        {
            "metrics": {
                **summary.metrics,
                PUBLICATION_COMPLETE_KEY: True,
                PUBLICATION_SEAL_DIGEST_KEY: seal_snapshot.sha256,
            }
        },
        strict=True,
    )
    init = WandbInitPayload(
        run_id=run_id,
        resume="allow",
        project=settings.wandb_project,
        name=settings.wandb_run_name or f"import-{case.case_id}",
        mode=settings.wandb_mode,
        directory=staging_dir,
        group=settings.wandb_group or case.workload_id,
        job_type=settings.wandb_job_type or "benchmark-import",
        entity=settings.wandb_entity,
        tags=tuple(_effective_wandb_tags(settings, suite_id=case.case_id, metadata=metadata) + ["imported"]),
    )
    return WandbPublishPayload(
        init=init,
        config=WandbConfigPayload.from_run_metadata(settings, suite_id=case.case_id, metadata=metadata),
        history=history,
        summary=summary,
        tables=tables,
    )


def prepare_sealed_import(
    measurement_path: Path,
    *,
    seal_path: Path,
    settings: ResolvedWandbConfig,
) -> PreparedWandbImport:
    """Capture and validate one sealed case without importing the W&B SDK."""
    if not settings.enabled:
        raise ImportInputError("strict W&B import requires online or offline mode")
    try:
        seal_snapshot = read_completion_seal(seal_path)
        seal = seal_snapshot.seal
        snapshot = read_measurement_snapshot(
            measurement_path,
            expected_statuses={seal.expected_run_id: seal.terminal_status},
        )
        verify_completion_seal(snapshot, seal)
        staging_dir = prepare_wandb_staging_dir(seal_path.parent)
        payload = build_import_payload(
            settings,
            snapshot=snapshot,
            seal_snapshot=seal_snapshot,
            staging_dir=staging_dir,
        )
    except (OSError, ValueError) as exc:
        raise ImportInputError(str(exc)) from None
    return PreparedWandbImport(
        payload=payload,
        measurement_sha256=snapshot.sha256,
        record_count=len(snapshot.records),
    )


def import_sealed_run(
    measurement_path: Path,
    *,
    seal_path: Path,
    settings: ResolvedWandbConfig,
) -> WandbPublishResult:
    """Capture, verify, and strictly publish one sealed external case."""
    prepared = prepare_sealed_import(measurement_path, seal_path=seal_path, settings=settings)
    return WandbPublisher().publish_payload(
        settings,
        payload=prepared.payload,
        measurement_sha256=prepared.measurement_sha256,
        record_count=prepared.record_count,
    )


@app.default
def main(
    measurement_path: Path,
    *,
    seal_path: Annotated[Path | None, cyclopts.Parameter("--seal-path")] = None,
    wandb_mode: Annotated[WandbMode | None, cyclopts.Parameter("--wandb-mode")] = None,
    wandb_project: Annotated[str | None, cyclopts.Parameter("--wandb-project")] = None,
    wandb_base_url: Annotated[str | None, cyclopts.Parameter("--wandb-base-url")] = None,
    wandb_entity: Annotated[str | None, cyclopts.Parameter("--wandb-entity")] = None,
    wandb_group: Annotated[str | None, cyclopts.Parameter("--wandb-group")] = None,
    wandb_job_type: Annotated[str | None, cyclopts.Parameter("--wandb-job-type")] = None,
    wandb_run_name: Annotated[str | None, cyclopts.Parameter("--wandb-run-name")] = None,
    wandb_tags: Annotated[str | None, cyclopts.Parameter("--wandb-tags")] = None,
    wandb_log_tables: Annotated[bool | None, cyclopts.Parameter("--wandb-log-tables")] = None,
    json_output: Annotated[bool, cyclopts.Parameter("--json")] = False,
    log_format: Annotated[LogFormat, cyclopts.Parameter("--log-format")] = LogFormat.plain,
) -> None:
    configure_logging(log_format)
    resolved_seal_path = seal_path or measurement_path.with_name(COMPLETION_SEAL_FILENAME)
    try:
        settings = ResolvedWandbConfig.from_env_and_overrides(
            defaults={"wandb_mode": WandbMode.online},
            wandb_mode=wandb_mode,
            wandb_project=wandb_project,
            wandb_base_url=wandb_base_url,
            wandb_entity=wandb_entity,
            wandb_group=wandb_group,
            wandb_job_type=wandb_job_type,
            wandb_run_name=wandb_run_name,
            wandb_tags=wandb_tags,
            wandb_log_tables=wandb_log_tables,
        )
        prepared = prepare_sealed_import(measurement_path, seal_path=resolved_seal_path, settings=settings)
    except ValidationError as exc:
        log_bad_input(logger, summarize_validation_error(exc))
        raise SystemExit(125) from exc
    except ImportInputError as exc:
        log_bad_input(logger, str(exc))
        raise SystemExit(125) from exc
    try:
        result = WandbPublisher().publish_payload(
            settings,
            payload=prepared.payload,
            measurement_sha256=prepared.measurement_sha256,
            record_count=prepared.record_count,
        )
    except Exception as exc:
        logger.error("Strict W&B import failed (%s)", type(exc).__name__)
        raise SystemExit(1) from None
    output = result.model_dump(mode="json", exclude_none=True)
    if json_output:
        sys.stdout.write(json.dumps(output, sort_keys=True) + "\n")
    else:
        sys.stdout.write(f"Imported W&B run: {result.run_id}\n")


if __name__ == "__main__":
    app()
