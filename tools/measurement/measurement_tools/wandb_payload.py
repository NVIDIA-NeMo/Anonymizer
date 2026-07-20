# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed W&B payload assembly from completed benchmark artifacts."""

from __future__ import annotations

import secrets
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from measurement_tools.wandb_ingress import read_measurement_snapshot
from measurement_tools.wandb_metadata import BenchmarkMetadata, WandbConfigPayload, WandbRunMetadata
from measurement_tools.wandb_publication import WandbInitPayload, WandbPublishPayload
from measurement_tools.wandb_run_identity import default_run_name, effective_wandb_tags
from measurement_tools.wandb_settings import ResolvedWandbConfig
from measurement_tools.wandb_staging import prepare_wandb_staging_dir

__all__ = ["BenchmarkWandbFinalization", "build_publish_payload"]


@dataclass(frozen=True)
class BenchmarkWandbFinalization:
    """Artifacts needed to finalize benchmark W&B logging."""

    measurement_path: Path
    cases: Sequence[Any]


def build_publish_payload(
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
        name=settings.wandb_run_name or default_run_name(suite_id, resolved_metadata),
        mode=settings.wandb_mode,
        directory=staging_dir,
        group=settings.wandb_group or suite_id,
        job_type=settings.wandb_job_type or "benchmark",
        entity=settings.wandb_entity,
        tags=tuple(effective_wandb_tags(settings, suite_id=suite_id, metadata=resolved_metadata)),
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
