#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, Redact, Rewrite, __version__, configure_logging

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "benchmarks" / "configs" / "smoke.json"
SYNC_MEAN_RE = re.compile(r"^\s*sync\s+mean:\s+(?P<seconds>[0-9.]+)s", re.MULTILINE)
ASYNC_MEAN_RE = re.compile(r"^\s*async\s+mean:\s+(?P<seconds>[0-9.]+)s", re.MULTILINE)
SPEEDUP_RE = re.compile(r"^\s*speedup:\s+(?P<speedup>[0-9.]+)x", re.MULTILINE)


MetricValue = bool | int | float | str | None


@dataclass
class ExperimentResult:
    name: str
    status: str
    duration_seconds: float
    metrics: dict[str, MetricValue] = field(default_factory=dict)
    reason: str | None = None


@dataclass(frozen=True)
class DatasetInput:
    name: str
    path: Path
    source: str
    text_column: str
    data_summary: str
    sha256: str
    rows: int | None

    def metadata(self) -> dict[str, MetricValue]:
        return {
            "name": self.name,
            "source": self.source,
            "path": str(self.path),
            "text_column": self.text_column,
            "data_summary": self.data_summary,
            "sha256": self.sha256,
            "rows": self.rows,
        }


@dataclass(frozen=True)
class ModelCallRecord:
    experiment: str | None
    alias: str
    model: str
    provider: str
    endpoint: str
    modality: str
    latency_seconds: float
    success: bool
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    error: str | None = None


class ModelLatencyRecorder:
    def __init__(self) -> None:
        self._records: list[ModelCallRecord] = []
        self._lock = threading.Lock()
        self._active_experiment: str | None = None

    @contextmanager
    def experiment(self, name: str) -> Iterator[None]:
        previous = self._active_experiment
        self._active_experiment = name
        try:
            yield
        finally:
            self._active_experiment = previous

    def record(
        self,
        model: Any,
        *,
        modality: str,
        latency_seconds: float,
        success: bool,
        usage: Any | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            self._records.append(
                ModelCallRecord(
                    experiment=self._active_experiment,
                    alias=str(getattr(model, "model_alias", "unknown")),
                    model=str(getattr(model, "model_name", "unknown")),
                    provider=str(getattr(model, "model_provider_name", "unknown")),
                    endpoint=str(getattr(getattr(model, "model_provider", None), "endpoint", "unknown")),
                    modality=modality,
                    latency_seconds=latency_seconds,
                    success=success,
                    input_tokens=getattr(usage, "input_tokens", None),
                    output_tokens=getattr(usage, "output_tokens", None),
                    total_tokens=getattr(usage, "total_tokens", None),
                    error=error,
                )
            )

    def records(self) -> list[dict[str, Any]]:
        with self._lock:
            return [asdict(record) for record in self._records]

    def experiment_metrics(self, experiment: str) -> dict[str, MetricValue]:
        records = [record for record in self._records if record.experiment == experiment]
        if not records:
            return {"model_calls": 0}

        successful = [record for record in records if record.success]
        if not successful:
            return {
                "model_calls": len(records),
                "model_failures": len(records),
            }

        durations = [record.latency_seconds for record in successful]
        slowest = max(successful, key=lambda record: record.latency_seconds)
        return {
            "model_calls": len(records),
            "model_failures": sum(not record.success for record in records),
            "model_latency_mean_seconds": sum(durations) / len(durations),
            "model_latency_p50_seconds": percentile(durations, 50),
            "model_latency_p95_seconds": percentile(durations, 95),
            "model_latency_max_seconds": slowest.latency_seconds,
            "slowest_model_alias": slowest.alias,
            "model_input_tokens": sum_optional(record.input_tokens for record in successful),
            "model_output_tokens": sum_optional(record.output_tokens for record in successful),
        }

    def aggregate(self) -> list[dict[str, Any]]:
        groups: dict[tuple[str | None, str, str, str, str, str], list[ModelCallRecord]] = {}
        with self._lock:
            records = list(self._records)

        for record in records:
            key = (record.experiment, record.alias, record.model, record.provider, record.endpoint, record.modality)
            groups.setdefault(key, []).append(record)

        summary = []
        for (experiment, alias, model, provider, endpoint, modality), group in sorted(
            groups.items(), key=lambda item: tuple(str(part) for part in item[0])
        ):
            successful = [record for record in group if record.success]
            durations = [record.latency_seconds for record in successful]
            summary.append(
                {
                    "experiment": experiment,
                    "alias": alias,
                    "model": model,
                    "provider": provider,
                    "endpoint": endpoint,
                    "modality": modality,
                    "calls": len(group),
                    "failures": sum(not record.success for record in group),
                    "mean_seconds": sum(durations) / len(durations) if durations else 0.0,
                    "p50_seconds": percentile(durations, 50),
                    "p95_seconds": percentile(durations, 95),
                    "max_seconds": max(durations) if durations else 0.0,
                    "input_tokens": sum_optional(record.input_tokens for record in successful),
                    "output_tokens": sum_optional(record.output_tokens for record in successful),
                    "total_tokens": sum_optional(record.total_tokens for record in successful),
                }
            )
        return summary


def parse_sync_async_output(output: str) -> dict[str, float]:
    sync_match = SYNC_MEAN_RE.search(output)
    async_match = ASYNC_MEAN_RE.search(output)
    speedup_match = SPEEDUP_RE.search(output)
    if not sync_match or not async_match or not speedup_match:
        raise ValueError("Could not parse sync/async benchmark output.")
    return {
        "sync_mean_seconds": float(sync_match.group("seconds")),
        "async_mean_seconds": float(async_match.group("seconds")),
        "speedup": float(speedup_match.group("speedup")),
    }


def build_markdown_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Anonymizer benchmark CI",
        "",
        f"- SUT ref: `{payload['meta']['sut_ref']}`",
        f"- SUT commit: `{payload['meta']['sut_commit']}`",
        f"- SUT version: `{payload['meta']['sut_version']}`",
        f"- Harness commit: `{payload['meta']['harness_commit']}`",
        f"- Config: `{payload['meta'].get('config_name', 'n/a')}`",
        f"- Config hash: `{payload['meta'].get('config_sha256', 'n/a')}`",
        f"- DD trace: `{payload['meta']['dd_trace']}`",
        "",
        "| Experiment | Status | Duration | Key metrics |",
        "| --- | --- | ---: | --- |",
    ]
    for experiment in payload["experiments"]:
        metric_text = ", ".join(
            f"{key}={format_metric(value)}" for key, value in experiment["metrics"].items() if value is not None
        )
        if experiment.get("reason"):
            metric_text = experiment["reason"] if not metric_text else f"{metric_text}, reason={experiment['reason']}"
        lines.append(
            f"| `{experiment['name']}` | {experiment['status']} | "
            f"{experiment['duration_seconds']:.2f}s | {metric_text or 'n/a'} |"
        )
    lines.append("")
    if payload.get("datasets"):
        lines.extend(
            [
                "## Datasets",
                "",
                "| Name | Rows | Text column | SHA-256 | Source |",
                "| --- | ---: | --- | --- | --- |",
            ]
        )
        for dataset in payload["datasets"]:
            lines.append(
                f"| `{dataset['name']}` | {dataset['rows'] or 'n/a'} | `{dataset['text_column']}` | "
                f"`{str(dataset['sha256'])[:12]}` | `{dataset['source']}` |"
            )
        lines.append("")
    if payload.get("model_latency", {}).get("summary"):
        lines.extend(
            [
                "## Model latency",
                "",
                "| Experiment | Alias | Model | Provider | Endpoint | Calls | Failures | Mean | P50 | P95 | Max | Tokens |",
                "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in payload["model_latency"]["summary"]:
            lines.append(
                f"| `{row['experiment']}` | `{row['alias']}` | `{row['model']}` | `{row['provider']}` | "
                f"`{row['endpoint']}` | "
                f"{row['calls']} | {row['failures']} | {row['mean_seconds']:.2f}s | "
                f"{row['p50_seconds']:.2f}s | "
                f"{row['p95_seconds']:.2f}s | {row['max_seconds']:.2f}s | {row['total_tokens'] or 'n/a'} |"
            )
        lines.append("")
    return "\n".join(lines)


def format_metric(value: MetricValue) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def load_benchmark_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    validate_benchmark_config(config)
    return config


def validate_benchmark_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise ValueError("Benchmark config must be an object.")
    if not isinstance(config.get("datasets"), dict) or not config["datasets"]:
        raise ValueError("Benchmark config must define at least one dataset.")
    if not isinstance(config.get("experiments"), list) or not config["experiments"]:
        raise ValueError("Benchmark config must define at least one experiment.")

    for name, dataset in config["datasets"].items():
        if not isinstance(dataset, dict):
            raise ValueError(f"Dataset {name!r} must be an object.")
        has_path = bool(dataset.get("path"))
        has_url = bool(dataset.get("url"))
        if has_path == has_url:
            raise ValueError(f"Dataset {name!r} must define exactly one of path or url.")
        if has_url and not dataset.get("sha256"):
            raise ValueError(f"Remote dataset {name!r} must define sha256.")
        for key in ("text_column", "data_summary"):
            if not dataset.get(key):
                raise ValueError(f"Dataset {name!r} must define {key}.")

    dataset_names = set(config["datasets"])
    for experiment in config["experiments"]:
        if not isinstance(experiment, dict):
            raise ValueError("Benchmark experiments must be objects.")
        for key in ("name", "pipeline", "dataset"):
            if not experiment.get(key):
                raise ValueError(f"Benchmark experiment must define {key}.")
        if experiment["pipeline"] not in {"redact", "rewrite"}:
            raise ValueError(f"Unsupported pipeline {experiment['pipeline']!r}.")
        if experiment["dataset"] not in dataset_names:
            raise ValueError(f"Unknown dataset {experiment['dataset']!r}.")


def select_experiment_specs(config: dict[str, Any], experiment_filter: str) -> list[dict[str, Any]]:
    experiments = config["experiments"]
    if experiment_filter == "all":
        return list(experiments)
    return [experiment for experiment in experiments if experiment["pipeline"] == experiment_filter]


def resolve_benchmark_datasets(config: dict[str, Any], cache_dir: Path) -> dict[str, DatasetInput]:
    return {
        name: resolve_dataset(name=name, spec=spec, cache_dir=cache_dir)
        for name, spec in sorted(config["datasets"].items())
    }


def resolve_dataset(name: str, spec: dict[str, Any], cache_dir: Path) -> DatasetInput:
    if spec.get("url"):
        path = download_dataset(name=name, url=str(spec["url"]), sha256=str(spec["sha256"]), cache_dir=cache_dir)
        source = str(spec["url"])
    else:
        path = resolve_repo_path(str(spec["path"]))
        source = str(spec["path"])

    expected_sha256 = spec.get("sha256")
    actual_sha256 = file_sha256(path)
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise ValueError(f"Dataset {name!r} SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}.")

    return DatasetInput(
        name=name,
        path=path,
        source=source,
        text_column=str(spec["text_column"]),
        data_summary=str(spec["data_summary"]),
        sha256=actual_sha256,
        rows=count_csv_rows(path) if path.suffix == ".csv" else None,
    )


def resolve_repo_path(path: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {resolved}")
    return resolved


def download_dataset(name: str, url: str, sha256: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(urllib.parse.urlparse(url).path).suffix or ".data"
    path = cache_dir / f"{name}-{sha256[:12]}{suffix}"
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_csv_rows(path: Path) -> int:
    with path.open(newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def run_preview_experiment(
    *,
    name: str,
    pipeline: str,
    dataset: DatasetInput,
    num_records: int,
    require_api_key: bool,
    trace_dir: Path | None,
) -> ExperimentResult:
    if not os.getenv("NVIDIA_API_KEY"):
        reason = "NVIDIA_API_KEY is not set"
        if require_api_key:
            raise RuntimeError(reason)
        return ExperimentResult(name=name, status="skipped", duration_seconds=0.0, reason=reason)

    configure_logging()
    input_data = AnonymizerInput(
        source=str(dataset.path),
        text_column=dataset.text_column,
        data_summary=dataset.data_summary,
    )
    config = build_anonymizer_config(pipeline)

    start = time.perf_counter()
    result = Anonymizer().preview(config=config, data=input_data, num_records=num_records)
    duration = time.perf_counter() - start

    frame = result.dataframe
    metrics = {
        "dataset": dataset.name,
        "rows": len(frame),
        "failed_records": len(result.failed_records),
    }
    if pipeline == "rewrite":
        metrics.update(
            {
                "utility_score_mean": mean_column(frame, "utility_score"),
                "leakage_mass_mean": mean_column(frame, "leakage_mass"),
                "weighted_leakage_rate_mean": mean_column(frame, "weighted_leakage_rate"),
                "any_high_leaked": sum_bool_column(frame, "any_high_leaked"),
                "needs_human_review": sum_bool_column(frame, "needs_human_review"),
            }
        )
    else:
        output_column = f"{result.resolved_text_column}_replaced"
        changed_rows = None
        if output_column in frame.columns and result.resolved_text_column in frame.columns:
            changed_rows = int((frame[output_column] != frame[result.resolved_text_column]).sum())
        metrics["changed_rows"] = changed_rows
    metrics.update(trace_metrics(result.trace_dataframe, experiment_name=name, trace_dir=trace_dir))
    return ExperimentResult(name=name, status="completed", duration_seconds=duration, metrics=metrics)


def build_anonymizer_config(pipeline: str) -> AnonymizerConfig:
    if pipeline == "rewrite":
        return AnonymizerConfig(rewrite=Rewrite(max_repair_iterations=1))
    if pipeline == "redact":
        return AnonymizerConfig(replace=Redact())
    raise ValueError(f"Unsupported pipeline {pipeline!r}.")


def run_sync_async_benchmark(num_records: int, iterations: int, timeout_seconds: int) -> ExperimentResult:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "benchmark_sync_vs_async.py"),
        "--num-records",
        str(num_records),
        "--iterations",
        str(iterations),
    ]
    start = time.perf_counter()
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"sync/async benchmark timed out after {timeout_seconds}s") from exc
    duration = time.perf_counter() - start
    if completed.returncode != 0:
        tail = completed.stderr[-1000:] or completed.stdout[-1000:]
        raise RuntimeError(f"sync/async benchmark failed:\n{tail}")

    metrics = parse_sync_async_output(completed.stdout)
    metrics["records"] = float(num_records)
    metrics["iterations"] = float(iterations)
    return ExperimentResult(
        name="mock_sync_async",
        status="completed",
        duration_seconds=duration,
        metrics=metrics,
    )


def mean_column(frame: Any, column: str) -> float | None:
    if column not in frame.columns:
        return None
    return float(frame[column].mean())


def sum_bool_column(frame: Any, column: str) -> int | None:
    if column not in frame.columns:
        return None
    return int(frame[column].fillna(False).astype(bool).sum())


def percentile(values: list[float], percentile_value: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile_value / 100)
    return ordered[index]


def sum_optional(values: Iterator[int | None]) -> int | None:
    total = 0
    found = False
    for value in values:
        if value is None:
            continue
        total += value
        found = True
    return total if found else None


def trace_metrics(frame: Any, *, experiment_name: str, trace_dir: Path | None) -> dict[str, MetricValue]:
    trace_columns = [column for column in frame.columns if str(column).endswith("__trace")]
    metrics: dict[str, MetricValue] = {"trace_columns": len(trace_columns)}
    if trace_dir is None:
        return metrics

    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"{experiment_name}.json"
    frame.to_json(path, orient="records", indent=2, default_handler=str)
    metrics["trace_dataframe_path"] = str(path)
    return metrics


def git_output(args: list[str]) -> str:
    completed = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return completed.stdout.strip()


@contextmanager
def data_designer_trace(trace_mode: str) -> Iterator[None]:
    if trace_mode == "none":
        yield
        return

    from data_designer.config.utils.trace_type import TraceType

    from anonymizer.engine.ndd.adapter import NddAdapter

    trace_type = TraceType(trace_mode)
    original = NddAdapter.run_workflow

    def run_workflow_with_trace(self: Any, df: Any, **kwargs: Any) -> Any:
        kwargs["columns"] = [copy_with_trace(column, trace_type) for column in kwargs["columns"]]
        return original(self, df, **kwargs)

    NddAdapter.run_workflow = run_workflow_with_trace
    try:
        yield
    finally:
        NddAdapter.run_workflow = original


def copy_with_trace(column: Any, trace_type: Any) -> Any:
    if hasattr(column, "with_trace") and hasattr(column, "model_copy"):
        return column.model_copy(update={"with_trace": trace_type})
    return column


@contextmanager
def model_latency_probe(recorder: ModelLatencyRecorder) -> Iterator[None]:
    from data_designer.engine.models.facade import ModelFacade

    original_completion = ModelFacade.completion
    original_acompletion = ModelFacade.acompletion

    def completion_with_latency(self: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        response = None
        error = None
        ignore_record = False
        try:
            response = original_completion(self, *args, **kwargs)
            return response
        except Exception as exc:
            if type(exc).__name__ == "SyncClientUnavailableError":
                ignore_record = True
                raise
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if not ignore_record:
                recorder.record(
                    self,
                    modality="chat",
                    latency_seconds=time.perf_counter() - start,
                    success=response is not None,
                    usage=getattr(response, "usage", None),
                    error=error,
                )

    async def acompletion_with_latency(self: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        response = None
        error = None
        ignore_record = False
        try:
            response = await original_acompletion(self, *args, **kwargs)
            return response
        except Exception as exc:
            if type(exc).__name__ == "SyncClientUnavailableError":
                ignore_record = True
                raise
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if not ignore_record:
                recorder.record(
                    self,
                    modality="chat",
                    latency_seconds=time.perf_counter() - start,
                    success=response is not None,
                    usage=getattr(response, "usage", None),
                    error=error,
                )

    ModelFacade.completion = completion_with_latency
    ModelFacade.acompletion = acompletion_with_latency
    try:
        yield
    finally:
        ModelFacade.completion = original_completion
        ModelFacade.acompletion = original_acompletion


def capture_experiment(
    name: str,
    runner: Callable[[], ExperimentResult],
    recorder: ModelLatencyRecorder,
) -> ExperimentResult:
    start = time.perf_counter()
    with recorder.experiment(name):
        try:
            result = runner()
        except Exception as exc:
            result = ExperimentResult(
                name=name,
                status="failed",
                duration_seconds=time.perf_counter() - start,
                reason=f"{type(exc).__name__}: {exc}",
            )
    result.metrics.update(recorder.experiment_metrics(name))
    return result


def build_payload(args: argparse.Namespace, recorder: ModelLatencyRecorder) -> dict[str, Any]:
    started_at = datetime.now(UTC)
    config = load_benchmark_config(args.config)
    datasets = resolve_benchmark_datasets(config, args.data_cache_dir)
    experiments = []
    experiment_specs = select_experiment_specs(config, args.experiments)
    if not experiment_specs and not args.run_sync_async:
        raise ValueError(f"No experiments matched filter {args.experiments!r}.")

    for experiment in experiment_specs:
        num_records = args.num_records if args.num_records is not None else int(experiment.get("num_records", 1))
        dataset = datasets[experiment["dataset"]]
        name = str(experiment["name"])
        experiments.append(
            capture_experiment(
                name,
                lambda experiment=experiment, dataset=dataset, name=name, num_records=num_records: (
                    run_preview_experiment(
                        name=name,
                        pipeline=str(experiment["pipeline"]),
                        dataset=dataset,
                        num_records=num_records,
                        require_api_key=args.require_api_key,
                        trace_dir=args.trace_dir,
                    )
                ),
                recorder,
            )
        )
    if args.run_sync_async:
        experiments.append(
            capture_experiment(
                "mock_sync_async",
                lambda: run_sync_async_benchmark(
                    num_records=args.sync_async_records,
                    iterations=args.sync_async_iterations,
                    timeout_seconds=args.sync_async_timeout_seconds,
                ),
                recorder,
            )
        )
    status = "failed" if any(experiment.status == "failed" for experiment in experiments) else "completed"
    return {
        "meta": {
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "sut_ref": args.sut_ref,
            "sut_commit": args.sut_commit,
            "sut_version": __version__,
            "harness_commit": git_output(["rev-parse", "HEAD"]),
            "python": sys.version.split()[0],
            "status": status,
            "config_path": str(args.config),
            "config_name": str(config.get("name", args.config.stem)),
            "config_sha256": file_sha256(args.config),
            "dd_trace": args.dd_trace,
            "experiments": args.experiments,
        },
        "datasets": [dataset.metadata() for dataset in datasets.values()],
        "experiments": [asdict(experiment) for experiment in experiments],
        "model_latency": {
            "summary": recorder.aggregate(),
            "calls": recorder.records(),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the benchmark CI scaffold.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data-cache-dir", type=Path, default=Path(".benchmark-data-cache"))
    parser.add_argument("--num-records", type=int, default=None)
    parser.add_argument("--experiments", choices=["all", "rewrite", "redact"], default="all")
    parser.add_argument("--run-sync-async", action="store_true")
    parser.add_argument("--sync-async-records", type=int, default=20)
    parser.add_argument("--sync-async-iterations", type=int, default=1)
    parser.add_argument("--sync-async-timeout-seconds", type=int, default=120)
    parser.add_argument("--sut-ref", default="HEAD")
    parser.add_argument("--sut-commit", default="HEAD")
    parser.add_argument("--out", type=Path, default=Path("benchmark-results/results.json"))
    parser.add_argument("--summary-out", type=Path, default=Path("benchmark-results/summary.md"))
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--dd-trace", choices=["none", "last_message", "all_messages"], default="none")
    parser.add_argument("--require-api-key", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recorder = ModelLatencyRecorder()
    with data_designer_trace(args.dd_trace), model_latency_probe(recorder):
        payload = build_payload(args, recorder)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.summary_out.write_text(build_markdown_summary(payload))
    if payload["meta"]["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
