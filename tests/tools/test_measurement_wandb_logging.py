# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from tests.tools.measurement_test_support import (
    _minimal_benchmark_spec,
    _strict_record_payload,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _wandb_metadata_spec(tool: ModuleType) -> Any:
    return tool.BenchmarkSpec(
        suite_id="suite-a",
        model_configs="models.yaml",
        model_providers="providers.yaml",
        artifact_path="artifacts",
        run_tags={"ci_job": "123", "api_key": "sk-secret-token"},
        case_retries=2,
        case_retry_backoff_sec=3.5,
        workloads=[
            tool.WorkloadSpec(
                id="workload-a",
                source="/private/path/input.csv",
                text_column="body",
                id_column="id",
                data_summary="contains Alice and raw secret",
                row_limit=5,
                row_offset=10,
            )
        ],
        configs=[_hash_config(tool), _rewrite_config(tool)],
        matrix=[tool.MatrixEntry(workload="workload-a", config="hash-a", repetitions=2)],
    )


def _hash_config(tool: ModuleType) -> Any:
    return tool.ConfigSpec(
        id="hash-a",
        detect={"entity_labels": ["person", "api_key"], "gliner_threshold": 0.4},
        replace=tool.ReplaceSpec(strategy=tool.ReplaceKind.hash, digest_length=12, instructions="raw secret"),
        evaluate=True,
    )


def _rewrite_config(tool: ModuleType) -> Any:
    return tool.ConfigSpec(
        id="rewrite-a",
        rewrite=tool.RewriteSpec(
            protect="protect Alice",
            preserve="preserve raw secret",
            instructions="raw prompt",
            risk_tolerance=tool.RiskTolerance.minimal,
            max_repair_iterations=1,
            strict_entity_protection=True,
        ),
    )


def _patch_stable_wandb_metadata(monkeypatch: pytest.MonkeyPatch, tool: ModuleType) -> None:
    monkeypatch.setattr(tool, "_git_metadata", lambda _cwd: {"commit": "abc123", "branch": "main", "dirty": False})
    monkeypatch.setattr(
        tool,
        "_package_versions",
        lambda: {"anonymizer_version": "1.2.3", "datadesigner_version": "4.5.6", "wandb_version": "7.8.9"},
    )


def _assert_wandb_metadata(metadata: dict[str, Any]) -> None:
    assert metadata["benchmark"]["suite_id"] == "suite-a"
    assert metadata["benchmark"]["case_count"] == 2
    assert metadata["git"] == {"commit": "abc123", "branch": "main", "dirty": False}
    assert metadata["runtime"]["anonymizer_version"] == "1.2.3"
    assert metadata["model_sources"] == {
        "has_model_configs": True,
        "has_model_providers": True,
        "has_artifact_path": True,
    }
    assert metadata["workloads"] == [_expected_wandb_workload(metadata)]
    assert metadata["configs"][0]["replace"] == {
        "strategy": "hash",
        "digest_length": 12,
        "has_instructions": True,
    }
    assert metadata["configs"][1]["rewrite"] == {
        "risk_tolerance": "minimal",
        "max_repair_iterations": 1,
        "strict_entity_protection": True,
        "has_privacy_goal": True,
        "has_protect": True,
        "has_preserve": True,
        "has_instructions": True,
    }


def _expected_wandb_workload(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "workload-a",
        "source": {
            "kind": "local_file",
            "suffix": ".csv",
        },
        "text_column": "body",
        "has_id_column": True,
        "has_data_summary": True,
        "row_limit": 5,
        "row_offset": 10,
    }


def test_build_wandb_metadata_projects_sweep_run_tags(tmp_path: Path, run_benchmarks_tool: ModuleType) -> None:
    spec = _minimal_benchmark_spec(
        run_benchmarks_tool,
        run_tags={
            "wandb_sweep": {
                "id": "threshold",
                "arm_id": "arm-000",
                "params": {"configs_all_detect_gliner_threshold": 0.2},
            },
        },
    )
    spec_path = tmp_path / "suite.yaml"
    spec_path.write_text("suite_id: suite\n", encoding="utf-8")

    metadata = run_benchmarks_tool.build_wandb_metadata(
        spec,
        spec_path=spec_path,
        output_dir=tmp_path / "out",
        export=True,
        fail_fast=False,
        dd_trace=run_benchmarks_tool.DDTraceMode.none,
        dd_task_trace=False,
    )

    assert metadata.sweep is not None
    assert metadata.sweep.id == "threshold"
    assert metadata.sweep.arm_id == "arm-000"
    assert metadata.sweep.params["configs_all_detect_gliner_threshold"] == 0.2


def test_build_wandb_metadata_includes_slurm_execution_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run_benchmarks_tool: ModuleType,
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "98765")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "3")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "private-node")
    spec = _minimal_benchmark_spec(run_benchmarks_tool)
    spec_path = tmp_path / "suite.yaml"
    spec_path.write_text("suite_id: suite\n", encoding="utf-8")

    metadata = run_benchmarks_tool.build_wandb_metadata(
        spec,
        spec_path=spec_path,
        output_dir=tmp_path / "private-output",
        export=True,
        fail_fast=True,
        dd_trace=run_benchmarks_tool.DDTraceMode.last_message,
        dd_task_trace=True,
    )
    serialized = metadata.model_dump_json()

    assert metadata.execution is not None
    assert metadata.execution.backend == "slurm"
    assert metadata.execution.dd_trace == "last_message"
    assert metadata.execution.slurm is not None
    assert metadata.execution.slurm.job_id == "98765"
    assert metadata.execution.slurm.array_task_id == "3"
    assert str(tmp_path) not in serialized
    assert "private-node" not in serialized


def test_wandb_typed_metrics_and_tables_exclude_local_fields(
    wandb_ingress_tool: ModuleType,
    wandb_logging_tool: ModuleType,
) -> None:
    canary = "alice@example.com/private/input"
    record = wandb_ingress_tool.RecordMeasurement.model_validate(
        _strict_record_payload(run_tags={"customer": canary}),
        strict=True,
    )

    metrics = wandb_logging_tool.extract_scalar_metrics(record)
    tables = wandb_logging_tool._typed_tables((record,))
    serialized = json.dumps(
        {"metrics": metrics, "tables": [table.model_dump(mode="json") for table in tables]},
        sort_keys=True,
    )

    assert metrics["measurement/record/final_entity_count"] == 1
    assert "text_length_chars" not in tables[0].columns
    assert "text_length_chars_bucket" in tables[0].columns
    assert canary not in serialized


@pytest.mark.parametrize("field", ["strategy", "row_index"])
def test_wandb_typed_ingress_rejects_arbitrary_table_strings(
    field: str,
    wandb_ingress_tool: ModuleType,
) -> None:
    with pytest.raises(ValidationError):
        wandb_ingress_tool.RecordMeasurement.model_validate(
            _strict_record_payload(**{field: "alice@example.com/private/input"}),
            strict=True,
        )


def test_wandb_aggregates_with_explicit_metric_policy(
    wandb_ingress_tool: ModuleType,
    wandb_logging_tool: ModuleType,
) -> None:
    records = tuple(
        wandb_ingress_tool.RecordMeasurement.model_validate(payload, strict=True)
        for payload in (
            _strict_record_payload(
                final_entity_count=2,
                utility_score=0.8,
                leakage_mass=0.4,
                weighted_leakage_rate=0.1,
                entity_precision=0.5,
                entity_recall=0.25,
                entity_f1=1 / 3,
            ),
            _strict_record_payload(
                timestamp_unix_sec=2.0,
                final_entity_count=1,
                utility_score=1.0,
                leakage_mass=0.2,
                weighted_leakage_rate=0.3,
                entity_precision=1.0,
                entity_recall=0.75,
                entity_f1=6 / 7,
            ),
        )
    )
    aggregated = wandb_logging_tool.aggregate_measurement_scalars(records)

    assert aggregated["measurement/record/final_entity_count"] == 3.0
    assert aggregated["measurement/record/utility_score_mean"] == pytest.approx(0.9)
    assert aggregated["measurement/record/leakage_mass_mean"] == pytest.approx(0.3)
    assert aggregated["measurement/record/weighted_leakage_rate_mean"] == pytest.approx(0.2)
    assert aggregated["measurement/record/entity_precision_mean"] == pytest.approx(0.75)
    assert aggregated["measurement/record/entity_recall_mean"] == pytest.approx(0.5)
    assert aggregated["measurement/record/entity_f1_mean"] == pytest.approx(25 / 42)
    assert not any("schema_version" in key for key in aggregated)
    assert not any("timestamp_unix_sec" in key for key in aggregated)
    assert not any("input_has_id_column" in key for key in aggregated)


def test_wandb_stage_summary_uses_only_terminal_stage_but_table_preserves_all(
    wandb_ingress_tool: ModuleType,
    wandb_logging_tool: ModuleType,
) -> None:
    records = tuple(
        wandb_ingress_tool.StageMeasurement.model_validate(
            {
                "schema_version": 1,
                "record_type": "stage",
                "run_id": "run-a",
                "run_tags": {},
                "timestamp_unix_sec": float(index),
                "stage": stage,
                "status": "completed",
                "elapsed_sec": elapsed_sec,
                "input_row_count": 10,
                "output_row_count": 10,
            },
            strict=True,
        )
        for index, (stage, elapsed_sec) in enumerate(
            (
                ("EntityDetectionWorkflow.run", 60.0),
                ("ReplacementWorkflow.run", 30.0),
                ("Anonymizer._run_internal", 100.0),
            ),
            start=1,
        )
    )

    aggregated = wandb_logging_tool.aggregate_measurement_scalars(records)
    tables = wandb_logging_tool._typed_tables(records)

    assert aggregated["measurement/stage/elapsed_sec"] == 100.0
    assert aggregated["measurement/stage/input_row_count"] == 10.0
    assert aggregated["measurement/stage/output_row_count"] == 10.0
    assert len(tables) == 1
    assert [row.stage for row in tables[0].rows] == [
        "EntityDetectionWorkflow.run",
        "ReplacementWorkflow.run",
        "Anonymizer._run_internal",
    ]


def test_wandb_scalar_registry_matches_package_field_catalog(wandb_logging_tool: ModuleType) -> None:
    from measurement_tools.wandb_metric_schema import (
        AGGREGATED_MEASUREMENT_FIELDS,
        SCALAR_AGGREGATION_BY_FIELD,
    )
    from measurement_tools.wandb_models import WandbHistoryPayload

    from anonymizer.measurement.fields import (
        SCALAR_ADDITIVE_FIELDS,
        SCALAR_AVERAGED_FIELDS,
        SCALAR_LAST_VALUE_FIELDS,
    )

    field_groups = (SCALAR_LAST_VALUE_FIELDS, SCALAR_ADDITIVE_FIELDS, SCALAR_AVERAGED_FIELDS)
    expected_fields = frozenset().union(*field_groups)

    assert sum(map(len, field_groups)) == len(expected_fields)
    assert frozenset(SCALAR_AGGREGATION_BY_FIELD) == expected_fields
    for field_name in AGGREGATED_MEASUREMENT_FIELDS:
        WandbHistoryPayload(metrics={f"measurement/record/{field_name}": 0})


def test_wandb_aggregates_rat_bench_reidentification_record(
    wandb_ingress_tool: ModuleType,
    wandb_logging_tool: ModuleType,
) -> None:
    record = wandb_ingress_tool.RatBenchReidentificationMeasurement.model_validate(
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
        },
        strict=True,
    )

    aggregated = wandb_logging_tool.aggregate_measurement_scalars((record,))

    assert aggregated["measurement/rat_bench_reidentification/rows_processed"] == 10.0
    assert aggregated["measurement/rat_bench_reidentification/reidentified_rows"] == 2.0
    assert aggregated["measurement/rat_bench_reidentification/reidentification_rate_pct_mean"] == pytest.approx(20.0)
    assert aggregated["measurement/rat_bench_reidentification/coverage_pct_mean"] == pytest.approx(55.0)
    assert aggregated["measurement/rat_bench_reidentification/mean_reid_score_mean"] == pytest.approx(0.12)
    assert aggregated["measurement/rat_bench_reidentification/reid_threshold"] == 0.2
    assert "measurement/rat_bench_reidentification/reid_threshold_mean" not in aggregated
    assert aggregated["measurement/rat_bench_reidentification/attacker_model"] == "openai/gpt-oss-120b"


def test_wandb_report_metric_names_match_aggregated_names(
    wandb_ingress_tool: ModuleType,
    wandb_logging_tool: ModuleType,
    wandb_report_tool: ModuleType,
) -> None:
    record = wandb_ingress_tool.RecordMeasurement.model_validate(
        _strict_record_payload(
            entity_precision=0.9,
            entity_recall=0.8,
            entity_f1=0.85,
            leakage_mass=0.1,
            repair_iterations=2,
        ),
        strict=True,
    )
    aggregated = wandb_logging_tool.aggregate_measurement_scalars((record,))

    expected = {
        "measurement/record/entity_precision_mean",
        "measurement/record/entity_recall_mean",
        "measurement/record/entity_f1_mean",
        "measurement/record/leakage_mass_mean",
        "measurement/record/repair_iterations_mean",
    }
    assert expected <= set(aggregated)
    assert expected <= set(wandb_report_tool._all_report_metrics())


def test_wandb_config_excludes_suite_run_tags(wandb_setup_tool: ModuleType) -> None:
    config = wandb_setup_tool.WandbConfigPayload.from_run_metadata(
        wandb_setup_tool.ResolvedWandbConfig(wandb_mode=wandb_setup_tool.WandbMode.offline),
        suite_id="suite-a",
        metadata=wandb_setup_tool.WandbRunMetadata(benchmark=wandb_setup_tool.BenchmarkMetadata(suite_id="suite-a")),
    )

    assert "run_tags" not in config.sdk_values()


def test_wandb_config_projects_only_declared_metadata(wandb_setup_tool: ModuleType) -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        wandb_setup_tool.WandbRunMetadata.model_validate(
            {"benchmark": {"suite_id": "suite-a", "owner_email": "alice@example.com"}},
            strict=True,
        )

    metadata = wandb_setup_tool.WandbRunMetadata.model_validate(
        {
            "run_kind": "sweep_arm",
            "benchmark": {"suite_id": "suite-a", "case_count": 2, "suite_file_hash": "content-hash"},
            "execution": {"backend": "slurm", "export": True, "slurm": {"job_id": "123"}},
            "workloads": [{"id": "workload-a", "row_limit": 5, "source": {"kind": "local_file", "suffix": ".csv"}}],
            "configs": [
                {
                    "id": "redact",
                    "mode": "replace",
                    "detect": {"gliner_threshold": 0.3, "entity_label_count": 4},
                    "replace": {"strategy": "redact"},
                }
            ],
            "matrix": [{"workload": "workload-a", "config": "redact", "repetitions": 1}],
            "sweep": {
                "id": "threshold-sweep",
                "arm_id": "arm-001",
                "params": {"configs_all_detect_gliner_threshold": 0.3},
            },
        },
        strict=True,
    )
    config = wandb_setup_tool.WandbConfigPayload.from_run_metadata(
        wandb_setup_tool.ResolvedWandbConfig(wandb_mode=wandb_setup_tool.WandbMode.offline),
        suite_id="suite-a",
        metadata=metadata,
    )

    assert config.benchmark == metadata.benchmark
    assert config.execution == metadata.execution
    assert config.sweep_params == {"configs_all_detect_gliner_threshold": 0.3}
    assert config.sdk_values()["sweep_param_configs_all_detect_gliner_threshold"] == 0.3


def test_wandb_run_tags_filter_sensitive_generated_values(wandb_setup_tool: ModuleType) -> None:
    metadata = wandb_setup_tool.WandbRunMetadata.model_validate(
        {
            "benchmark": {"suite_id": "suite-a"},
            "git": {"branch": "feature/token-fix", "dirty": False},
        },
        strict=True,
    )
    tags = wandb_setup_tool._effective_wandb_tags(
        wandb_setup_tool.ResolvedWandbConfig(
            wandb_mode=wandb_setup_tool.WandbMode.offline,
            wandb_tags="tag-a,tag-b",
        ),
        suite_id="suite-a",
        metadata=metadata,
    )

    assert tags == ["tag-a", "tag-b", "suite:suite-a", "clean"]


def test_wandb_run_tags_bound_long_slurm_case_identity(wandb_setup_tool: ModuleType) -> None:
    case_id = "rat_bench-diff1__val-gpt-oss-120b-low__aug-gpt-oss-120b-medium__rep-gpt-oss-120b-low__r000"
    metadata = wandb_setup_tool.WandbRunMetadata.model_validate(
        {
            "run_kind": "imported_case",
            "benchmark": {"suite_id": case_id},
            "configs": [{"id": "config-a"}],
            "imported": {
                "completion_seal_schema_version": 1,
                "completion_seal_sha256": "a" * 64,
                "producer_repository": "anonymizer-experiments",
                "producer_commit": "b" * 40,
                "phase": "baseline",
                "case_id": case_id,
            },
        },
        strict=True,
    )

    tags = wandb_setup_tool._effective_wandb_tags(
        wandb_setup_tool.ResolvedWandbConfig(
            wandb_mode=wandb_setup_tool.WandbMode.offline,
            wandb_tags="release",
        ),
        suite_id=case_id,
        metadata=metadata,
    )

    assert tags == ["release", "suite:rat_bench-diff1__val-gpt-oss-120b-low__aug-gp-cc0ea432483a"]
    assert all(1 <= len(tag) <= 64 for tag in tags)


def test_wandb_init_payload_rejects_sdk_invalid_tag_length(
    tmp_path: Path,
    wandb_setup_tool: ModuleType,
) -> None:
    with pytest.raises(ValidationError, match="tags.0"):
        wandb_setup_tool.WandbInitPayload(
            run_id="run-a",
            project="project-a",
            name="run-a",
            mode=wandb_setup_tool.WandbMode.offline,
            directory=tmp_path,
            group="group-a",
            job_type="benchmark",
            tags=("x" * 65,),
        )


def test_wandb_environment_isolates_routing_and_restores_exactly(
    monkeypatch: pytest.MonkeyPatch,
    wandb_setup_tool: ModuleType,
) -> None:
    monkeypatch.setenv("WANDB_GROUP", "ambient-group")
    monkeypatch.setenv("WANDB_API_KEY", "auth-token")
    monkeypatch.setenv("WANDB_ERROR_REPORTING", "true")
    monkeypatch.setenv("UNRELATED", "unchanged")
    before = dict(os.environ)
    settings = wandb_setup_tool.ResolvedWandbConfig(
        wandb_mode=wandb_setup_tool.WandbMode.offline,
        wandb_project="resolved-project",
        wandb_group="resolved-group",
    )

    with wandb_setup_tool.WandbSdkEnvironment(settings):
        assert os.environ["WANDB_GROUP"] == "resolved-group"
        assert os.environ["WANDB_PROJECT"] == "resolved-project"
        assert "WANDB_API_KEY" not in os.environ
        assert os.environ["WANDB_ERROR_REPORTING"] == "false"
        assert "UNRELATED" not in os.environ
        with pytest.raises(RuntimeError, match="nested or concurrent"):
            with wandb_setup_tool.WandbSdkEnvironment(settings):
                pass

    assert dict(os.environ) == before


def test_wandb_output_allows_root_owned_group_writable_intermediate_directory(
    monkeypatch: pytest.MonkeyPatch,
    wandb_setup_tool: ModuleType,
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
        directory_mode = wandb_setup_tool.stat.S_IFDIR | 0o755
        uid = 0
        if name == "shared-project":
            directory_mode = wandb_setup_tool.stat.S_IFDIR | 0o2770
        if name == "wandb-output":
            uid = wandb_setup_tool.os.geteuid()
        return SimpleNamespace(st_mode=directory_mode, st_uid=uid)

    monkeypatch.setattr(wandb_setup_tool.os, "open", fake_open)
    monkeypatch.setattr(wandb_setup_tool.os, "close", fake_close)
    monkeypatch.setattr(wandb_setup_tool.os, "fstat", fake_fstat)

    descriptor = wandb_setup_tool._open_directory_no_follow(
        Path("/shared/projects/shared-project/users/test-user/wandb-output")
    )

    assert descriptor_paths[descriptor] == "wandb-output"
    assert any(path == "shared-project" for path in descriptor_paths.values())


def test_wandb_environment_uses_namespaced_base_url(
    monkeypatch: pytest.MonkeyPatch,
    wandb_setup_tool: ModuleType,
) -> None:
    monkeypatch.setenv("WANDB_BASE_URL", "https://ambient.invalid")
    monkeypatch.setenv("ANONYMIZER_MEASUREMENT_WANDB_BASE_URL", "https://resolved.example")
    settings = wandb_setup_tool.ResolvedWandbConfig.from_env_and_overrides()

    with wandb_setup_tool.WandbSdkEnvironment(settings):
        assert os.environ["WANDB_BASE_URL"] == "https://resolved.example"


def test_wandb_settings_validation_does_not_echo_credential_url(wandb_setup_tool: ModuleType) -> None:
    canary = "alice:super-secret"

    with pytest.raises(ValidationError) as raised:
        wandb_setup_tool.ResolvedWandbConfig(wandb_base_url=f"https://{canary}@example.com")

    assert canary not in str(raised.value)


def test_wandb_cli_validation_does_not_log_credential_url(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    run_benchmarks_tool: ModuleType,
) -> None:
    canary = "alice:super-secret"

    with caplog.at_level(logging.ERROR, logger="measurement.benchmark"), pytest.raises(SystemExit) as raised:
        run_benchmarks_tool.main(
            tmp_path / "unused-suite.yaml",
            wandb_base_url=f"https://{canary}@example.com",
        )

    assert raised.value.code == 125
    assert canary not in caplog.text
    assert "wandb_base_url:value_error" in caplog.text


def test_wandb_native_failure_log_does_not_echo_exception(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
    wandb_setup_tool: ModuleType,
) -> None:
    canary = "alice@example.com/private/input"

    def fail_publish(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError(canary)

    monkeypatch.setattr(wandb_setup_tool.WandbPublisher, "publish", fail_publish)
    with caplog.at_level(logging.WARNING, logger="measurement.wandb"):
        result = wandb_setup_tool.publish_benchmark_wandb_best_effort(
            wandb_setup_tool.ResolvedWandbConfig(wandb_mode=wandb_setup_tool.WandbMode.offline),
            suite_id="suite",
            output_dir=tmp_path,
            finalization=wandb_setup_tool.BenchmarkWandbFinalization(
                measurement_path=tmp_path / "missing.jsonl",
                cases=[],
            ),
        )

    assert result.published is False
    assert canary not in caplog.text
    assert "ValueError" in caplog.text


def test_wandb_outbound_models_structurally_exclude_sensitive_fields(wandb_setup_tool: ModuleType) -> None:
    models = sys.modules[wandb_setup_tool.WandbPublishPayload.__module__]

    with pytest.raises(ValidationError, match="Extra inputs"):
        models.WandbHistoryPayload(metrics={}, text="Alice")
    with pytest.raises(ValidationError, match="no aggregate exposure policy"):
        models.WandbHistoryPayload(metrics={"measurement/record/text": "Alice"})
    with pytest.raises(ValidationError, match="Extra inputs"):
        models.WandbConfigPayload(
            suite_id="suite",
            wandb_mode=models.WandbMode.offline,
            wandb_log_tables=False,
            benchmark={"prompt": "Alice"},
        )
    assert all(
        policy.exposure != models.Exposure.never
        for policies in models.OUTBOUND_FIELD_POLICIES.values()
        for policy in policies.values()
    )
    assert set(models._METRIC_TABLE_POLICY_FIELDS) == set(models._MetricTableRow.model_fields)
    with pytest.raises(ValidationError):
        models.BenchmarkMetadata(case_retry_backoff_sec=-1.0)
    with pytest.raises(ValidationError):
        models.DetectMetadata(gliner_threshold=1.1)
    with pytest.raises(ValidationError):
        models.SweepMetadata(
            id="sweep",
            arm_id="arm",
            params={"configs_all_detect_gliner_threshold": -0.1},
        )


def test_build_wandb_metadata_includes_sanitized_configs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run_benchmarks_tool: ModuleType,
) -> None:
    spec = _wandb_metadata_spec(run_benchmarks_tool)
    _patch_stable_wandb_metadata(monkeypatch, run_benchmarks_tool)

    metadata = run_benchmarks_tool.build_wandb_metadata(
        spec,
        spec_path=tmp_path / "suite.yaml",
        output_dir=tmp_path / "output",
        export=True,
        fail_fast=True,
        dd_trace=run_benchmarks_tool.DDTraceMode.none,
        dd_task_trace=False,
    )
    serialized = metadata.model_dump_json()

    _assert_wandb_metadata(metadata.model_dump(mode="json", exclude_none=True))
    for forbidden in ("Alice", "raw secret", "raw prompt", "/private/path", "models.yaml", "providers.yaml"):
        assert forbidden not in serialized


def test_wandb_extract_scalar_metrics_rejects_trace_record_types(wandb_logging_tool: ModuleType) -> None:
    metrics = wandb_logging_tool.extract_scalar_metrics(
        {
            "record_type": "dd_message_trace",
            "workflow_name": "entity-detection",
            "messages": [{"role": "user", "content": "secret prompt"}],
        }
    )

    assert metrics == {}


def test_wandb_case_elapsed_metrics_reject_negative_and_nonfinite(wandb_logging_tool: ModuleType) -> None:
    status = SimpleNamespace(value="completed")
    for elapsed_sec in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and non-negative"):
            wandb_logging_tool.summarize_benchmark_cases([SimpleNamespace(status=status, elapsed_sec=elapsed_sec)])
