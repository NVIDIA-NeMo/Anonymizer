# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed completion seals for strict W&B import of external benchmark cases."""

from __future__ import annotations

import json
import os
import secrets
import stat
from contextlib import suppress
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

from measurement_tools.validation import (
    NonNegativeInt,
    RedactedStrictFrozenModel,
    VisibleIdentifier,
    VisibleSlurmIdentifier,
)
from measurement_tools.wandb_ingress import (
    MeasurementSnapshot,
    _reject_json_constant,
    _validate_json_nesting_bytes,
    capture_ingress_bytes,
)
from measurement_tools.wandb_ingress import (
    read_measurement_snapshot as read_measurement_snapshot,
)

COMPLETION_SEAL_FILENAME = "completion-seal.json"
COMPLETION_SEAL_SCHEMA_VERSION = 1
DEFAULT_MAX_COMPLETION_SEAL_BYTES = 64 * 1024

Sha256Digest = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]
GitCommit = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{40,64}$")]


class ImportedCaseIdentity(RedactedStrictFrozenModel):
    case_id: VisibleIdentifier
    workload_id: VisibleIdentifier
    config_id: VisibleIdentifier
    repetition: NonNegativeInt


class SlurmJobProvenance(RedactedStrictFrozenModel):
    role: VisibleSlurmIdentifier
    job_id: VisibleSlurmIdentifier


class SlurmCaseProvenance(RedactedStrictFrozenModel):
    backend: Literal["slurm"] = "slurm"
    phase: VisibleIdentifier
    case_index: NonNegativeInt
    job_id: VisibleIdentifier | None = None
    array_job_id: VisibleIdentifier | None = None
    array_task_id: VisibleIdentifier | None = None
    jobs: Annotated[
        tuple[SlurmJobProvenance, ...],
        Field(max_length=64, exclude_if=lambda value: not value),
    ] = ()

    @field_validator("jobs", mode="before")
    @classmethod
    def parse_json_jobs(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def roles_are_unique(self) -> SlurmCaseProvenance:
        roles = [job.role for job in self.jobs]
        if len(roles) != len(set(roles)):
            raise ValueError("Slurm job roles must be unique")
        return self


def parse_slurm_jobs(values: list[str]) -> tuple[SlurmJobProvenance, ...]:
    """Parse repeatable redaction-safe ``role=job_id`` CLI values."""
    jobs: list[SlurmJobProvenance] = []
    for value in values:
        role, separator, job_id = value.partition("=")
        if not separator:
            raise ValueError("invalid --slurm-job: expected role=job_id")
        jobs.append(SlurmJobProvenance(role=role, job_id=job_id))
    return tuple(jobs)


class CompletionSealProducer(RedactedStrictFrozenModel):
    repository: VisibleIdentifier
    commit: GitCommit


class CompletionSeal(RedactedStrictFrozenModel):
    seal_schema_version: Literal[1] = COMPLETION_SEAL_SCHEMA_VERSION
    terminal_status: Literal["completed"]
    measurement_schema_version: Literal[1]
    expected_run_id: VisibleIdentifier
    measurement_byte_count: NonNegativeInt
    measurement_record_count: NonNegativeInt
    measurement_sha256: Sha256Digest
    case: ImportedCaseIdentity
    slurm: SlurmCaseProvenance
    producer: CompletionSealProducer


class CompletionSealSnapshot(BaseModel):
    path: Path
    sha256: Sha256Digest
    byte_count: NonNegativeInt
    seal: CompletionSeal

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def build_completion_seal(
    snapshot: MeasurementSnapshot,
    *,
    case: ImportedCaseIdentity,
    slurm: SlurmCaseProvenance,
    producer: CompletionSealProducer,
) -> CompletionSeal:
    """Build a seal only for one authoritatively completed measurement run."""
    terminal = snapshot.terminal_stage()
    _validate_case_identity(snapshot, case)
    return CompletionSeal(
        terminal_status="completed",
        measurement_schema_version=1,
        expected_run_id=terminal.run_id,
        measurement_byte_count=snapshot.byte_count,
        measurement_record_count=len(snapshot.records),
        measurement_sha256=snapshot.sha256,
        case=case,
        slurm=slurm,
        producer=producer,
    )


def read_completion_seal(
    path: Path,
    *,
    max_bytes: int = DEFAULT_MAX_COMPLETION_SEAL_BYTES,
) -> CompletionSealSnapshot:
    """Capture and strictly parse one bounded completion seal."""
    captured = capture_ingress_bytes(path, max_bytes=max_bytes)
    try:
        _validate_json_nesting_bytes(captured.payload, max_nesting=8)
        raw = json.loads(captured.payload, parse_constant=_reject_json_constant)
        seal = CompletionSeal.model_validate(raw, strict=True)
    except ValidationError as exc:
        locations = sorted({".".join(str(part) for part in error["loc"]) for error in exc.errors()})
        summary = ", ".join(location for location in locations[:8] if location) or "seal"
        raise ValueError(f"invalid completion seal: schema violation at {summary}") from None
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid completion seal: {exc}") from exc
    return CompletionSealSnapshot(
        path=path,
        sha256=captured.sha256,
        byte_count=len(captured.payload),
        seal=seal,
    )


def verify_completion_seal(snapshot: MeasurementSnapshot, seal: CompletionSeal) -> None:
    """Verify that a captured measurement snapshot is exactly the sealed case."""
    terminal = snapshot.terminal_stage()
    _validate_case_identity(snapshot, seal.case)
    expected = (
        seal.measurement_byte_count,
        seal.measurement_record_count,
        seal.measurement_sha256,
        seal.expected_run_id,
        seal.terminal_status,
    )
    observed = (
        snapshot.byte_count,
        len(snapshot.records),
        snapshot.sha256,
        terminal.run_id,
        "completed",
    )
    if observed != expected:
        raise ValueError("completion seal does not match measurement snapshot")


def write_completion_seal(path: Path, seal: CompletionSeal) -> None:
    """Atomically replace a completion seal in its existing case directory."""
    payload = (
        json.dumps(seal.model_dump(mode="json"), ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    directory_descriptor = _open_owned_directory_no_follow(path.parent)
    temporary_name = f".{path.name}.{secrets.token_hex(8)}.tmp"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_descriptor,
        )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("completion seal write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary_name, path.name, src_dir_fd=directory_descriptor, dst_dir_fd=directory_descriptor)
        os.fsync(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            os.unlink(temporary_name, dir_fd=directory_descriptor)
        os.close(directory_descriptor)


def _validate_case_identity(snapshot: MeasurementSnapshot, case: ImportedCaseIdentity) -> None:
    expected_tags = {
        "case_id": case.case_id,
        "workload_id": case.workload_id,
        "config_id": case.config_id,
        "repetition": case.repetition,
    }
    for record in snapshot.records:
        if record.run_id != case.case_id or any(
            record.run_tags.get(key) != value for key, value in expected_tags.items()
        ):
            raise ValueError("completion seal case identity does not match measurement records")


def _open_owned_directory_no_follow(path: Path) -> int:
    absolute = path.absolute()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(absolute.anchor, flags)
        for index, part in enumerate(absolute.parts[1:], start=1):
            child = os.open(part, flags | no_follow, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
            metadata = os.fstat(descriptor)
            final = index == len(absolute.parts) - 1
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError("completion seal path must contain only directories")
            if final and metadata.st_uid != os.geteuid():
                raise ValueError("completion seal directory must be owned by the current user")
            writable_by_others = metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
            if writable_by_others and not metadata.st_mode & stat.S_ISVTX:
                raise ValueError("completion seal path contains an untrusted writable directory")
        return descriptor
    except (OSError, ValueError) as exc:
        if descriptor >= 0:
            os.close(descriptor)
        raise ValueError("cannot safely open completion seal directory") from exc
