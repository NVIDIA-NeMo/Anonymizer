#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write an atomic completion seal for one finished external benchmark case."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Annotated

import cyclopts
from measurement_tools.cli import LogFormat, configure_logging, log_bad_input
from measurement_tools.wandb_completion import (
    COMPLETION_SEAL_FILENAME,
    CompletionSealProducer,
    ImportedCaseIdentity,
    SlurmCaseProvenance,
    build_completion_seal,
    read_measurement_snapshot,
    write_completion_seal,
)

app = cyclopts.App(help=__doc__)
logger = logging.getLogger("measurement.completion_seal")


@app.default
def main(
    measurement_path: Path,
    *,
    case_id: Annotated[str, cyclopts.Parameter("--case-id")],
    workload_id: Annotated[str, cyclopts.Parameter("--workload-id")],
    config_id: Annotated[str, cyclopts.Parameter("--config-id")],
    repetition: Annotated[int, cyclopts.Parameter("--repetition")],
    phase: Annotated[str, cyclopts.Parameter("--phase")],
    case_index: Annotated[int, cyclopts.Parameter("--case-index")],
    producer_repository: Annotated[str, cyclopts.Parameter("--producer-repository")],
    producer_commit: Annotated[str, cyclopts.Parameter("--producer-commit")],
    seal_path: Annotated[Path | None, cyclopts.Parameter("--seal-path")] = None,
    slurm_job_id: Annotated[str | None, cyclopts.Parameter("--slurm-job-id")] = None,
    slurm_array_job_id: Annotated[str | None, cyclopts.Parameter("--slurm-array-job-id")] = None,
    slurm_array_task_id: Annotated[str | None, cyclopts.Parameter("--slurm-array-task-id")] = None,
    json_output: Annotated[bool, cyclopts.Parameter("--json")] = False,
    log_format: Annotated[LogFormat, cyclopts.Parameter("--log-format")] = LogFormat.plain,
) -> None:
    configure_logging(log_format)
    resolved_seal_path = seal_path or measurement_path.with_name(COMPLETION_SEAL_FILENAME)
    try:
        snapshot = read_measurement_snapshot(measurement_path)
        seal = build_completion_seal(
            snapshot,
            case=ImportedCaseIdentity(
                case_id=case_id,
                workload_id=workload_id,
                config_id=config_id,
                repetition=repetition,
            ),
            slurm=SlurmCaseProvenance(
                phase=phase,
                case_index=case_index,
                job_id=slurm_job_id,
                array_job_id=slurm_array_job_id,
                array_task_id=slurm_array_task_id,
            ),
            producer=CompletionSealProducer(repository=producer_repository, commit=producer_commit),
        )
        write_completion_seal(resolved_seal_path, seal)
    except (OSError, ValueError) as exc:
        log_bad_input(logger, str(exc))
        raise SystemExit(125) from exc
    result = {
        "seal_path": str(resolved_seal_path),
        "measurement_sha256": seal.measurement_sha256,
        "record_count": seal.measurement_record_count,
    }
    if json_output:
        sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")
    else:
        sys.stdout.write(f"Wrote completion seal: {resolved_seal_path}\n")


if __name__ == "__main__":
    app()
