# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import socket
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from tests.tools.measurement_test_support import (
    _strict_record_payload,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WRITE_COMPLETION_SEAL_PATH = REPO_ROOT / "tools/measurement/write_completion_seal.py"


def _rewrite_run_payload(*, include_replace_key: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "record_type": "run",
        "run_id": "run-a",
        "run_tags": {
            "case_id": "run-a",
            "workload_id": "rat-diff1",
            "config_id": "rewrite",
            "repetition": 0,
        },
        "timestamp_unix_sec": 1.0,
        "mode": "rewrite",
        "strategy": "Rewrite",
        "input_row_count": 1,
        "preview_num_records": None,
        "source_hash": "a" * 64,
        "input_source": {"kind": "local_file", "scheme": None, "suffix": ".csv"},
        "input_text_column": "text",
        "input_has_id_column": False,
        "input_has_data_summary": False,
        "detect": {"entity_label_source": "default", "entity_label_count": 2},
        "rewrite": {
            "risk_tolerance": "low",
            "max_repair_iterations": 2,
            "strict_entity_protection": True,
        },
        "models": [],
        "runtime": {},
    }
    if include_replace_key:
        payload["replace"] = None
    return payload


def _terminal_stage_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "record_type": "stage",
        "run_id": "run-a",
        "run_tags": {
            "case_id": "run-a",
            "workload_id": "rat-diff1",
            "config_id": "rewrite",
            "repetition": 0,
        },
        "timestamp_unix_sec": 2.0,
        "stage": "Anonymizer._run_internal",
        "status": "completed",
        "elapsed_sec": 0.5,
    }


def test_wandb_publisher_validates_before_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    wandb_setup_tool: ModuleType,
) -> None:
    measurements = tmp_path / "measurements.jsonl"
    measurements.write_text('{"record_type":"record","unknown":"secret"}\n', encoding="utf-8")
    monkeypatch.setattr(wandb_setup_tool, "require_wandb", lambda: pytest.fail("SDK import must be post-validation"))

    with pytest.raises(ValueError, match="invalid measurement record"):
        wandb_setup_tool.WandbPublisher().publish(
            wandb_setup_tool.ResolvedWandbConfig(wandb_mode=wandb_setup_tool.WandbMode.offline),
            suite_id="suite",
            output_dir=tmp_path,
            finalization=wandb_setup_tool.BenchmarkWandbFinalization(
                measurement_path=measurements,
                cases=[],
            ),
        )


@pytest.mark.parametrize("include_replace_key", [True, False])
def test_wandb_snapshot_accepts_rewrite_run_without_replace_metadata(
    tmp_path: Path,
    wandb_ingress_tool: ModuleType,
    include_replace_key: bool,
) -> None:
    path = tmp_path / "measurements.jsonl"
    records = [
        _rewrite_run_payload(include_replace_key=include_replace_key),
        _terminal_stage_payload(),
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    snapshot = wandb_ingress_tool.read_measurement_snapshot(path, expected_statuses={"run-a": "completed"})

    assert len(snapshot.records) == 2
    assert snapshot.records[0].record_type == "run"
    assert snapshot.records[0].mode == "rewrite"
    assert snapshot.records[0].replace is None


@pytest.mark.parametrize(
    ("mode", "missing_key"),
    [
        ("replace", "replace"),
        ("rewrite", "rewrite"),
    ],
)
def test_wandb_snapshot_requires_active_mode_metadata(
    tmp_path: Path,
    wandb_ingress_tool: ModuleType,
    mode: str,
    missing_key: str,
) -> None:
    path = tmp_path / "measurements.jsonl"
    run = _rewrite_run_payload(include_replace_key=False)
    run["mode"] = mode
    run["strategy"] = "Substitute" if mode == "replace" else "Rewrite"
    run.pop(missing_key, None)
    if mode == "replace":
        run.pop("rewrite", None)
    records = [run, _terminal_stage_payload()]
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    with pytest.raises(ValueError, match="schema violation at run"):
        wandb_ingress_tool.read_measurement_snapshot(path, expected_statuses={"run-a": "completed"})


def test_wandb_snapshot_uses_one_descriptor_and_enforces_limits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    wandb_ingress_tool: ModuleType,
) -> None:
    path = tmp_path / "measurements.jsonl"
    record = _strict_record_payload()
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    reads = 0
    real_read = wandb_ingress_tool.os.read

    def counted_read(descriptor: int, count: int) -> bytes:
        nonlocal reads
        reads += 1
        return real_read(descriptor, count)

    monkeypatch.setattr(wandb_ingress_tool.os, "read", counted_read)
    snapshot = wandb_ingress_tool.read_measurement_snapshot(path)

    assert reads >= 1
    assert len(snapshot.records) == 1
    with pytest.raises(ValueError, match="byte limit"):
        wandb_ingress_tool.read_measurement_snapshot(path, max_bytes=1)
    with pytest.raises(ValueError, match="record limit"):
        wandb_ingress_tool.read_measurement_snapshot(path, max_records=0)


def test_wandb_snapshot_rejects_symlink_and_special_file(
    tmp_path: Path,
    wandb_ingress_tool: ModuleType,
) -> None:
    target = tmp_path / "target.jsonl"
    target.write_text("", encoding="utf-8")
    symlink = tmp_path / "link.jsonl"
    symlink.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        wandb_ingress_tool.read_measurement_snapshot(symlink)
    with pytest.raises(ValueError, match="regular file"):
        wandb_ingress_tool.read_measurement_snapshot(tmp_path)


def test_wandb_snapshot_rejects_fifo_socket_and_device(
    tmp_path: Path,
    wandb_ingress_tool: ModuleType,
) -> None:
    fifo = tmp_path / "measurements.fifo"
    os.mkfifo(fifo)
    unix_socket = tmp_path / "measurements.sock"
    listener = socket.socket(socket.AF_UNIX)
    listener.bind(str(unix_socket))
    try:
        for path in (fifo, unix_socket, Path("/dev/null")):
            with pytest.raises(ValueError, match="regular file|safely open"):
                wandb_ingress_tool.read_measurement_snapshot(path)
    finally:
        listener.close()


def test_wandb_snapshot_rejects_parent_symlink_and_hard_link(
    tmp_path: Path,
    wandb_ingress_tool: ModuleType,
) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    source = real_dir / "measurements.jsonl"
    source.write_text(json.dumps(_strict_record_payload()) + "\n", encoding="utf-8")
    redirected = tmp_path / "redirected"
    redirected.symlink_to(real_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        wandb_ingress_tool.read_measurement_snapshot(redirected / source.name)

    hard_link = tmp_path / "hard-link.jsonl"
    os.link(source, hard_link)
    with pytest.raises(ValueError, match="hard links"):
        wandb_ingress_tool.read_measurement_snapshot(source)


def test_wandb_snapshot_uses_pinned_parent_descriptor_during_path_swap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    wandb_ingress_tool: ModuleType,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "measurements.jsonl"
    source.write_text(json.dumps(_strict_record_payload(run_id="trusted")) + "\n", encoding="utf-8")
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    (replacement / source.name).write_text(
        json.dumps(_strict_record_payload(run_id="redirected")) + "\n",
        encoding="utf-8",
    )
    moved = tmp_path / "source-moved"
    real_open = wandb_ingress_tool.os.open
    swapped = False

    def swapping_open(path: Any, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        nonlocal swapped
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if path == source_dir.name and dir_fd is not None and not swapped:
            source_dir.rename(moved)
            source_dir.symlink_to(replacement, target_is_directory=True)
            swapped = True
        return descriptor

    monkeypatch.setattr(wandb_ingress_tool.os, "open", swapping_open)

    snapshot = wandb_ingress_tool.read_measurement_snapshot(source)

    assert snapshot.records[0].run_id == "trusted"


def test_wandb_staging_rejects_parent_symlink(tmp_path: Path, wandb_setup_tool: ModuleType) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    redirected = tmp_path / "redirected"
    redirected.symlink_to(real_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="cannot contain symlinks"):
        wandb_setup_tool.prepare_wandb_staging_dir(redirected)


def test_wandb_staging_rejects_untrusted_writable_output(tmp_path: Path, wandb_setup_tool: ModuleType) -> None:
    output = tmp_path / "world-writable"
    output.mkdir(mode=0o777)
    output.chmod(0o777)

    with pytest.raises(ValueError, match="untrusted directories"):
        wandb_setup_tool.prepare_wandb_staging_dir(output)


def test_wandb_staging_allows_repeated_basename_in_foreign_intermediate_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    wandb_setup_tool: ModuleType,
) -> None:
    temp_root = Path(tempfile.gettempdir()).resolve()
    assert temp_root in tmp_path.parents
    output = tmp_path / temp_root.name
    output.mkdir()
    real_fstat = wandb_setup_tool.os.fstat
    temp_root_metadata = temp_root.stat()

    def fstat_with_foreign_temp_root(descriptor: int) -> Any:
        metadata = real_fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino) == (temp_root_metadata.st_dev, temp_root_metadata.st_ino):
            return SimpleNamespace(st_mode=metadata.st_mode, st_uid=metadata.st_uid + 1)
        return metadata

    monkeypatch.setattr(wandb_setup_tool.os, "fstat", fstat_with_foreign_temp_root)

    assert wandb_setup_tool.prepare_wandb_staging_dir(output) == output / ".wandb-private"


def test_wandb_snapshot_rejects_line_nesting_negative_and_wrong_shape(
    tmp_path: Path,
    wandb_ingress_tool: ModuleType,
) -> None:
    path = tmp_path / "measurements.jsonl"
    payload = _strict_record_payload(run_tags={"nested": {"deeper": {"value": 1}}})
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line byte limit"):
        wandb_ingress_tool.read_measurement_snapshot(path, max_line_bytes=8)
    with pytest.raises(ValueError, match="nesting limit"):
        wandb_ingress_tool.read_measurement_snapshot(path, max_nesting=2)

    path.write_text(json.dumps(_strict_record_payload(final_entity_count=-1)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="record.final_entity_count"):
        wandb_ingress_tool.read_measurement_snapshot(path)

    path.write_text(json.dumps(_strict_record_payload(stage="RewriteWorkflow.run")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema violation"):
        wandb_ingress_tool.read_measurement_snapshot(path)


def test_wandb_snapshot_schema_error_does_not_echo_rejected_value(
    tmp_path: Path,
    wandb_ingress_tool: ModuleType,
) -> None:
    canary = "alice@example.com/private/input"
    path = tmp_path / "measurements.jsonl"
    path.write_text(json.dumps(_strict_record_payload(unknown=canary)) + "\n", encoding="utf-8")

    with pytest.raises(ValueError) as raised:
        wandb_ingress_tool.read_measurement_snapshot(path)

    assert canary not in str(raised.value)
    assert "record.unknown" in str(raised.value)


def test_wandb_snapshot_rejects_source_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    wandb_ingress_tool: ModuleType,
) -> None:
    path = tmp_path / "measurements.jsonl"
    path.write_text(json.dumps(_strict_record_payload()) + "\n", encoding="utf-8")
    real_read = wandb_ingress_tool.os.read

    def mutate_after_read(descriptor: int, count: int) -> bytes:
        payload = real_read(descriptor, count)
        path.write_bytes(payload + b" ")
        return payload

    monkeypatch.setattr(wandb_ingress_tool.os, "read", mutate_after_read)
    with pytest.raises(ValueError, match="changed while being read"):
        wandb_ingress_tool.read_measurement_snapshot(path)


def test_wandb_snapshot_rejects_source_truncation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    wandb_ingress_tool: ModuleType,
) -> None:
    path = tmp_path / "measurements.jsonl"
    path.write_text(json.dumps(_strict_record_payload()) + "\n", encoding="utf-8")
    real_read = wandb_ingress_tool.os.read
    truncated = False

    def truncate_after_read(descriptor: int, count: int) -> bytes:
        nonlocal truncated
        payload = real_read(descriptor, count)
        if payload and not truncated:
            path.write_bytes(b"")
            truncated = True
        return payload

    monkeypatch.setattr(wandb_ingress_tool.os, "read", truncate_after_read)
    with pytest.raises(ValueError, match="changed while being read"):
        wandb_ingress_tool.read_measurement_snapshot(path)


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (
            _strict_record_payload(unknown="secret"),
            "schema violation",
        ),
        (
            {
                "schema_version": 1,
                "record_type": "dd_message_trace",
                "run_id": "run-a",
                "run_tags": {},
                "timestamp_unix_sec": 1.0,
            },
            "record_type",
        ),
        (
            _strict_record_payload(final_entity_count="1"),
            "schema violation",
        ),
    ],
)
def test_wandb_snapshot_rejects_unknown_and_trace_records(
    tmp_path: Path,
    wandb_ingress_tool: ModuleType,
    payload: dict[str, Any],
    match: str,
) -> None:
    path = tmp_path / "measurements.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        wandb_ingress_tool.read_measurement_snapshot(path)


def test_wandb_snapshot_rejects_nonfinite_and_mixed_runs(
    tmp_path: Path,
    wandb_ingress_tool: ModuleType,
) -> None:
    path = tmp_path / "measurements.jsonl"
    path.write_text(
        json.dumps(_strict_record_payload(utility_score=float("nan"))) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-finite"):
        wandb_ingress_tool.read_measurement_snapshot(path)

    path.write_text(
        json.dumps(_strict_record_payload(run_id="run-a"))
        + "\n"
        + json.dumps(_strict_record_payload(run_id="run-b"))
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mixed run identities"):
        wandb_ingress_tool.read_measurement_snapshot(path)


def test_wandb_snapshot_rejects_case_status_mismatch(tmp_path: Path, wandb_ingress_tool: ModuleType) -> None:
    path = tmp_path / "measurements.jsonl"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "stage",
                "run_id": "run-a",
                "run_tags": {},
                "timestamp_unix_sec": 1.0,
                "stage": "Anonymizer._run_internal",
                "status": "error",
                "elapsed_sec": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="status does not match"):
        wandb_ingress_tool.read_measurement_snapshot(path, expected_statuses={"run-a": "completed"})


def test_wandb_snapshot_requires_exactly_one_terminal_stage(tmp_path: Path, wandb_ingress_tool: ModuleType) -> None:
    path = tmp_path / "measurements.jsonl"
    path.write_text(json.dumps(_strict_record_payload()) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one terminal stage"):
        wandb_ingress_tool.read_measurement_snapshot(path, expected_statuses={"run-a": "completed"})


def test_completion_seal_allows_root_owned_group_writable_intermediate_directory(
    monkeypatch: pytest.MonkeyPatch,
    wandb_completion_tool: ModuleType,
) -> None:
    descriptor_paths: dict[int, str] = {}

    def fake_open(path: Any, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        descriptor = 100 + len(descriptor_paths)
        descriptor_paths[descriptor] = str(path)
        return descriptor

    def fake_close(descriptor: int) -> None:
        pass

    def fake_fstat(descriptor: int) -> Any:
        name = descriptor_paths[descriptor]
        directory_mode = stat.S_IFDIR | 0o755
        uid = 0
        if name == "shared-project":
            directory_mode = stat.S_IFDIR | 0o2770
        if name == "case":
            uid = wandb_completion_tool.os.geteuid()
        return SimpleNamespace(st_mode=directory_mode, st_uid=uid)

    monkeypatch.setattr(wandb_completion_tool.os, "open", fake_open)
    monkeypatch.setattr(wandb_completion_tool.os, "close", fake_close)
    monkeypatch.setattr(wandb_completion_tool.os, "fstat", fake_fstat)

    descriptor = wandb_completion_tool._open_owned_directory_no_follow(
        Path("/shared/projects/shared-project/users/test-user/case")
    )

    assert descriptor_paths[descriptor] == "case"
    assert any(path == "shared-project" for path in descriptor_paths.values())


def test_completion_seal_round_trip_and_digest_verification(
    tmp_path: Path,
    wandb_completion_tool: ModuleType,
) -> None:
    measurement_path = tmp_path / "measurements.jsonl"
    measurement_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "stage",
                "run_id": "dataset__config__r000",
                "run_tags": {
                    "case_id": "dataset__config__r000",
                    "workload_id": "dataset-split",
                    "config_id": "config",
                    "repetition": 0,
                },
                "timestamp_unix_sec": 1.0,
                "stage": "Anonymizer._run_internal",
                "status": "completed",
                "elapsed_sec": 0.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    snapshot = wandb_completion_tool.read_measurement_snapshot(measurement_path)
    seal = wandb_completion_tool.build_completion_seal(
        snapshot,
        case=wandb_completion_tool.ImportedCaseIdentity(
            case_id="dataset__config__r000",
            workload_id="dataset-split",
            config_id="config",
            repetition=0,
        ),
        slurm=wandb_completion_tool.SlurmCaseProvenance(
            phase="baseline",
            case_index=3,
            job_id="12345",
        ),
        producer=wandb_completion_tool.CompletionSealProducer(
            repository="anonymizer-experiments",
            commit="a" * 40,
        ),
    )
    seal_path = tmp_path / wandb_completion_tool.COMPLETION_SEAL_FILENAME

    wandb_completion_tool.write_completion_seal(seal_path, seal)
    first_bytes = seal_path.read_bytes()
    captured_seal = wandb_completion_tool.read_completion_seal(seal_path)
    wandb_completion_tool.verify_completion_seal(snapshot, captured_seal.seal)
    wandb_completion_tool.write_completion_seal(seal_path, seal)

    assert seal_path.read_bytes() == first_bytes
    assert captured_seal.seal.case.config_id == "config"
    assert "jobs" not in json.loads(first_bytes)["slurm"]
    assert not list(tmp_path.glob(".*.tmp"))

    real_parent = tmp_path / "real-parent"
    case_dir = real_parent / "case"
    case_dir.mkdir(parents=True)
    redirected_parent = tmp_path / "redirected-parent"
    redirected_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="safely open completion seal directory"):
        wandb_completion_tool.write_completion_seal(
            redirected_parent / "case" / wandb_completion_tool.COMPLETION_SEAL_FILENAME,
            seal,
        )

    changed_record = json.loads(measurement_path.read_text(encoding="utf-8"))
    changed_record["elapsed_sec"] = 0.75
    measurement_path.write_text(json.dumps(changed_record) + "\n", encoding="utf-8")
    changed_snapshot = wandb_completion_tool.read_measurement_snapshot(measurement_path)
    with pytest.raises(ValueError, match="does not match"):
        wandb_completion_tool.verify_completion_seal(changed_snapshot, captured_seal.seal)


def test_completion_seal_rejects_failed_or_missing_terminal_stage(
    tmp_path: Path,
    wandb_completion_tool: ModuleType,
) -> None:
    measurement_path = tmp_path / "measurements.jsonl"
    measurement_path.write_text(json.dumps(_strict_record_payload()) + "\n", encoding="utf-8")
    snapshot = wandb_completion_tool.read_measurement_snapshot(measurement_path)

    with pytest.raises(ValueError, match="terminal stage record"):
        wandb_completion_tool.build_completion_seal(
            snapshot,
            case=wandb_completion_tool.ImportedCaseIdentity(
                case_id="case",
                workload_id="workload",
                config_id="config",
                repetition=0,
            ),
            slurm=wandb_completion_tool.SlurmCaseProvenance(phase="phase", case_index=0),
            producer=wandb_completion_tool.CompletionSealProducer(
                repository="anonymizer-experiments",
                commit="b" * 40,
            ),
        )


def test_completion_seal_cli_writes_typed_multi_job_provenance_and_redacts_invalid_input(
    tmp_path: Path,
) -> None:
    measurement_path = tmp_path / "measurements.jsonl"
    measurement_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "stage",
                "run_id": "case",
                "run_tags": {
                    "case_id": "case",
                    "workload_id": "workload",
                    "config_id": "config",
                    "repetition": 0,
                },
                "timestamp_unix_sec": 1.0,
                "stage": "Anonymizer._run_internal",
                "status": "completed",
                "elapsed_sec": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    base_command = [
        sys.executable,
        str(WRITE_COMPLETION_SEAL_PATH),
        str(measurement_path),
        "--case-id",
        "case",
        "--workload-id",
        "workload",
        "--config-id",
        "config",
        "--repetition",
        "0",
        "--phase",
        "baseline",
        "--case-index",
        "0",
        "--producer-repository",
        "anonymizer-experiments",
        "--producer-commit",
        "a" * 40,
    ]

    completed = subprocess.run(
        [*base_command, "--slurm-job", "detect=123", "--slurm-job", "replace=456"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads((tmp_path / "completion-seal.json").read_text(encoding="ascii"))

    assert completed.returncode == 0
    assert payload["slurm"]["jobs"] == [
        {"job_id": "123", "role": "detect"},
        {"job_id": "456", "role": "replace"},
    ]

    invalid_value = "private/value"
    rejected = subprocess.run(
        [*base_command, "--slurm-job", f"detect={invalid_value}"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert rejected.returncode == 125
    assert invalid_value not in rejected.stderr

    duplicate = subprocess.run(
        [*base_command, "--slurm-job", "same=123", "--slurm-job", "same=456"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert duplicate.returncode == 125
    assert "123" not in duplicate.stderr
    assert "456" not in duplicate.stderr


def test_completion_seal_rejects_case_identity_mismatch_and_hidden_workflow_error(
    tmp_path: Path,
    wandb_completion_tool: ModuleType,
) -> None:
    case = wandb_completion_tool.ImportedCaseIdentity(
        case_id="case",
        workload_id="workload",
        config_id="config",
        repetition=0,
    )
    tags = {
        "case_id": "case",
        "workload_id": "workload",
        "config_id": "config",
        "repetition": 0,
    }
    records = [
        {
            "schema_version": 1,
            "record_type": "stage",
            "run_id": "case",
            "run_tags": tags,
            "timestamp_unix_sec": 1.0,
            "stage": "EntityDetectionWorkflow.run",
            "status": "error",
            "elapsed_sec": 0.25,
        },
        {
            "schema_version": 1,
            "record_type": "stage",
            "run_id": "case",
            "run_tags": tags,
            "timestamp_unix_sec": 2.0,
            "stage": "Anonymizer._run_internal",
            "status": "completed",
            "elapsed_sec": 0.5,
        },
    ]
    measurement_path = tmp_path / "measurements.jsonl"
    measurement_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    snapshot = wandb_completion_tool.read_measurement_snapshot(measurement_path)
    seal_kwargs = {
        "slurm": wandb_completion_tool.SlurmCaseProvenance(phase="phase", case_index=0),
        "producer": wandb_completion_tool.CompletionSealProducer(
            repository="anonymizer-experiments",
            commit="b" * 40,
        ),
    }

    with pytest.raises(ValueError, match="status does not match"):
        wandb_completion_tool.build_completion_seal(snapshot, case=case, **seal_kwargs)

    records.pop(0)
    records[0]["run_tags"] = {**tags, "config_id": "other"}
    measurement_path.write_text(json.dumps(records[0]) + "\n", encoding="utf-8")
    mismatched_snapshot = wandb_completion_tool.read_measurement_snapshot(measurement_path)
    with pytest.raises(ValueError, match="case identity"):
        wandb_completion_tool.build_completion_seal(mismatched_snapshot, case=case, **seal_kwargs)

    failed_terminal = {
        "schema_version": 1,
        "record_type": "stage",
        "run_id": "case",
        "run_tags": {
            "case_id": "case",
            "workload_id": "workload",
            "config_id": "config",
            "repetition": 0,
        },
        "timestamp_unix_sec": 1.0,
        "stage": "Anonymizer._run_internal",
        "status": "error",
        "elapsed_sec": 0.5,
    }
    measurement_path.write_text(json.dumps(failed_terminal) + "\n", encoding="utf-8")
    failed_snapshot = wandb_completion_tool.read_measurement_snapshot(measurement_path)
    with pytest.raises(ValueError, match="terminal status does not match"):
        wandb_completion_tool.build_completion_seal(
            failed_snapshot,
            case=wandb_completion_tool.ImportedCaseIdentity(
                case_id="case",
                workload_id="workload",
                config_id="config",
                repetition=0,
            ),
            slurm=wandb_completion_tool.SlurmCaseProvenance(phase="phase", case_index=0),
            producer=wandb_completion_tool.CompletionSealProducer(
                repository="anonymizer-experiments",
                commit="b" * 40,
            ),
        )


def test_wandb_parser_rejects_deep_input_before_json_load(
    monkeypatch: pytest.MonkeyPatch,
    wandb_ingress_tool: ModuleType,
) -> None:
    payload = b"[" * 5_000 + b"0" + b"]" * 5_000
    monkeypatch.setattr(wandb_ingress_tool.json, "loads", lambda *_args, **_kwargs: pytest.fail("parser called"))

    with pytest.raises(ValueError, match="nesting limit"):
        wandb_ingress_tool._parse_records(payload, max_records=1, max_line_bytes=len(payload), max_nesting=16)


def test_wandb_ingress_accepts_collector_model_workflow(tmp_path: Path, wandb_ingress_tool: ModuleType) -> None:
    from anonymizer.measurement import (
        MeasurementCollector,
        MeasurementConfig,
        measurement_session,
        record_model_workflow,
    )

    path = tmp_path / "measurements.jsonl"
    canary = "alice@example.com/private/input"
    collector = MeasurementCollector(run_id="collector-run", record_hash_key="test-key")
    with measurement_session(collector):
        record_model_workflow(
            workflow_name="entity-detection-native-rules-router",
            model_aliases=["native-direct"],
            input_row_count=1,
            output_row_count=1,
            failed_record_count=0,
            elapsed_sec=0.25,
            extra_fields={"operator_note": canary},
        )
    MeasurementConfig(output_path=path).write_collector(collector)

    snapshot = wandb_ingress_tool.read_measurement_snapshot(path)

    assert len(snapshot.records) == 1
    assert snapshot.records[0].record_type == "model_workflow"
    source_record = json.loads(path.read_text(encoding="utf-8"))
    assert source_record["local_fields"]["operator_note"] == canary
    assert canary not in snapshot.records[0].model_dump_json()

    source_record["operator_note"] = source_record["local_fields"].pop("operator_note")
    path.write_text(json.dumps(source_record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="operator_note"):
        wandb_ingress_tool.read_measurement_snapshot(path)


def test_wandb_ingress_accepts_collector_ndd_workflow(tmp_path: Path, wandb_ingress_tool: ModuleType) -> None:
    from anonymizer.measurement import (
        MeasurementCollector,
        MeasurementConfig,
        measurement_session,
        record_ndd_workflow,
    )

    path = tmp_path / "measurements.jsonl"
    collector = MeasurementCollector(run_id="collector-run", record_hash_key="test-key")
    with measurement_session(collector):
        record_ndd_workflow(
            workflow_name="entity-detection",
            model_aliases=["detector"],
            input_row_count=1,
            output_row_count=1,
            failed_record_count=0,
            elapsed_sec=0.25,
        )
    MeasurementConfig(output_path=path).write_collector(collector)

    snapshot = wandb_ingress_tool.read_measurement_snapshot(path)

    assert len(snapshot.records) == 1
    assert snapshot.records[0].record_type == "ndd_workflow"
    source_record = json.loads(path.read_text(encoding="utf-8"))
    assert "local_fields" not in source_record


def test_wandb_ingress_accepts_rat_bench_reidentification_summary(
    tmp_path: Path,
    wandb_ingress_tool: ModuleType,
) -> None:
    path = tmp_path / "measurements.jsonl"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "rat_bench_reidentification",
                "run_id": "run-a",
                "run_tags": {},
                "timestamp_unix_sec": 1.0,
                "rows_processed": 10,
                "rows_failed": 1,
                "reidentified_rows": 2,
                "direct_reidentified_rows": 1,
                "correctmatch_reidentified_rows": 1,
                "reid_error_rows": 0,
                "reidentification_rate_pct": 20.0,
                "coverage_pct": 55.0,
                "correct_guess_count": 11,
                "incorrect_guess_count": 9,
                "total_guess_count": 20,
                "mean_reid_score": 0.12,
                "reid_threshold": 0.2,
                "missing_output_rows": 0,
                "elapsed_sec": 3.5,
                "attacker_model": "openai/gpt-oss-120b",
                "attacker_endpoint_kind": "bigiron",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    snapshot = wandb_ingress_tool.read_measurement_snapshot(path)

    assert len(snapshot.records) == 1
    assert snapshot.records[0].record_type == "rat_bench_reidentification"
    source_record = json.loads(path.read_text(encoding="utf-8"))
    assert "local_fields" not in source_record


def test_wandb_ingress_rejects_rat_bench_percentages_above_100(
    wandb_ingress_tool: ModuleType,
) -> None:
    payload = {
        "schema_version": 1,
        "record_type": "rat_bench_reidentification",
        "run_id": "run-a",
        "run_tags": {},
        "timestamp_unix_sec": 1.0,
        "rows_processed": 10,
        "rows_failed": 0,
        "reidentified_rows": 2,
        "direct_reidentified_rows": 1,
        "correctmatch_reidentified_rows": 1,
        "reid_error_rows": 0,
        "reidentification_rate_pct": 100.1,
        "coverage_pct": 55.0,
        "correct_guess_count": 11,
        "incorrect_guess_count": 9,
        "total_guess_count": 20,
        "mean_reid_score": 0.12,
        "reid_threshold": 0.2,
    }

    with pytest.raises(wandb_ingress_tool.ValidationError):
        wandb_ingress_tool.RatBenchReidentificationMeasurement.model_validate(payload, strict=True)

    payload["reidentification_rate_pct"] = 20.0
    payload["coverage_pct"] = 100.1
    with pytest.raises(wandb_ingress_tool.ValidationError):
        wandb_ingress_tool.RatBenchReidentificationMeasurement.model_validate(payload, strict=True)
