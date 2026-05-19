# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("benchmark_ci", REPO_ROOT / "scripts" / "benchmark_ci.py")
assert SPEC is not None
benchmark_ci = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = benchmark_ci
SPEC.loader.exec_module(benchmark_ci)


def test_parse_sync_async_output() -> None:
    output = """
      sync  mean: 1.234s
      async mean: 0.456s
      speedup:    2.71x
    """

    assert benchmark_ci.parse_sync_async_output(output) == {
        "sync_mean_seconds": 1.234,
        "async_mean_seconds": 0.456,
        "speedup": 2.71,
    }


def test_build_markdown_summary() -> None:
    summary = benchmark_ci.build_markdown_summary(
        {
            "meta": {
                "sut_ref": "abc123",
                "sut_commit": "abc123",
                "sut_version": "0.1.0",
                "harness_commit": "def456",
                "dd_trace": "none",
            },
            "experiments": [
                {
                    "name": "rewrite_preview",
                    "status": "completed",
                    "duration_seconds": 1.25,
                    "metrics": {"rows": 3, "utility_score_mean": 0.9},
                }
            ],
        }
    )

    assert "SUT ref: `abc123`" in summary
    assert "| `rewrite_preview` | completed | 1.25s | rows=3, utility_score_mean=0.9 |" in summary


def test_model_latency_recorder_separates_failures() -> None:
    recorder = benchmark_ci.ModelLatencyRecorder()
    model = SimpleNamespace(model_alias="alias", model_name="model", model_provider_name="provider")

    with recorder.experiment("experiment"):
        recorder.record(
            model,
            modality="chat",
            latency_seconds=2.0,
            success=True,
            usage=SimpleNamespace(input_tokens=3, output_tokens=4, total_tokens=7),
        )
        recorder.record(model, modality="chat", latency_seconds=0.1, success=False)

    assert recorder.experiment_metrics("experiment") == {
        "model_calls": 2,
        "model_failures": 1,
        "model_latency_mean_seconds": 2.0,
        "model_latency_p50_seconds": 2.0,
        "model_latency_p95_seconds": 2.0,
        "model_latency_max_seconds": 2.0,
        "slowest_model_alias": "alias",
        "model_input_tokens": 3,
        "model_output_tokens": 4,
    }
    assert recorder.aggregate()[0]["failures"] == 1


def test_model_latency_probe_ignores_async_bridge_sentinel(monkeypatch: pytest.MonkeyPatch) -> None:
    from data_designer.engine.models.clients.errors import SyncClientUnavailableError
    from data_designer.engine.models.facade import ModelFacade

    def raise_sentinel(self: object, *args: object, **kwargs: object) -> None:
        raise SyncClientUnavailableError("sentinel")

    monkeypatch.setattr(ModelFacade, "completion", raise_sentinel)
    recorder = benchmark_ci.ModelLatencyRecorder()

    with recorder.experiment("experiment"), benchmark_ci.model_latency_probe(recorder):
        with pytest.raises(SyncClientUnavailableError):
            ModelFacade.completion(object())

    assert recorder.experiment_metrics("experiment") == {"model_calls": 0}
    assert recorder.aggregate() == []
