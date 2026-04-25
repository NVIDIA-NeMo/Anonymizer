#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark sync vs async engine on an Anonymizer-shaped detection pipeline.

Runs the detection pipeline (with mock LLM responses) through both the sync
and async DataDesigner engines and reports wall-clock times.

Because DATA_DESIGNER_ASYNC_ENGINE is read at module load time, each engine
mode runs in a subprocess with the appropriate environment variable.

Usage:
    python scripts/benchmark_sync_vs_async.py
    python scripts/benchmark_sync_vs_async.py --num-records 50 --iterations 3
    python scripts/benchmark_sync_vs_async.py --latency  # simulate 50ms LLM latency
    python scripts/benchmark_sync_vs_async.py --real --num-records 5  # real NIM endpoint
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any

RESULT_PREFIX = "BENCH="
MODEL_ALIAS = "mock-detector"
REAL_MODEL_ALIAS = "nim-detector"
REAL_MODEL = "nvidia/nemotron-3-nano-30b-a3b"
NVIDIA_ENDPOINT = "https://integrate.api.nvidia.com/v1"
ENV_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
# Also check a common parent location
ENV_FILE_ALT = os.path.expanduser("~/Code/.env")


# ---------------------------------------------------------------------------
# .env loader (no external deps)
# ---------------------------------------------------------------------------


def _load_dotenv() -> None:
    """Load key=value pairs from .env file into os.environ."""
    for path in (ENV_FILE, ENV_FILE_ALT):
        if os.path.isfile(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    value = value.strip().strip('"').strip("'")
                    os.environ.setdefault(key.strip(), value)
            return


# ---------------------------------------------------------------------------
# Mock LLM patches
# ---------------------------------------------------------------------------


def _patch_llm(*, latency_ms: float = 0.0) -> Any:
    """Context manager that replaces LLM completion with mock responses."""
    import contextlib

    from data_designer.engine.models.clients.adapters.openai_compatible import OpenAICompatibleClient
    from data_designer.engine.models.clients.types import AssistantMessage, ChatCompletionResponse

    original_completion = OpenAICompatibleClient.completion
    original_acompletion = OpenAICompatibleClient.acompletion

    mock_entities = json.dumps(
        {
            "entities": [
                {"text": "Alice", "label": "first_name", "start": 0, "end": 5, "score": 0.95},
                {"text": "Acme Corp", "label": "company_name", "start": 15, "end": 24, "score": 0.90},
                {"text": "Seattle", "label": "city", "start": 28, "end": 35, "score": 0.92},
            ]
        }
    )
    mock_augmented = json.dumps({"entities": [{"value": "engineer", "label": "occupation", "reason": "job title"}]})
    mock_validation = json.dumps(
        {
            "decisions": [
                {"id": "first_name_0_5", "decision": "keep", "proposed_label": "", "reason": "direct identifier"},
                {"id": "company_name_15_24", "decision": "keep", "proposed_label": "", "reason": "employer"},
                {"id": "city_28_35", "decision": "keep", "proposed_label": "", "reason": "location"},
            ]
        }
    )

    def _pick_response(messages: list[dict[str, Any]]) -> str:
        prompt_text = " ".join(str(m.get("content", "")) for m in messages).lower()
        if "validate" in prompt_text or "template" in prompt_text:
            return mock_validation
        if "augment" in prompt_text or "untagged" in prompt_text:
            return mock_augmented
        return mock_entities

    def fake_completion(self: Any, request: Any) -> ChatCompletionResponse:
        text = _pick_response(request.messages)
        if latency_ms > 0:
            time.sleep(latency_ms / 1000.0)
        return ChatCompletionResponse(message=AssistantMessage(content=text))

    async def fake_acompletion(self: Any, request: Any) -> ChatCompletionResponse:
        text = _pick_response(request.messages)
        if latency_ms > 0:
            await asyncio.sleep(latency_ms / 1000.0)
        return ChatCompletionResponse(message=AssistantMessage(content=text))

    @contextlib.contextmanager
    def patch():
        OpenAICompatibleClient.completion = fake_completion
        OpenAICompatibleClient.acompletion = fake_acompletion
        try:
            yield
        finally:
            OpenAICompatibleClient.completion = original_completion
            OpenAICompatibleClient.acompletion = original_acompletion

    return patch()


# ---------------------------------------------------------------------------
# Pipeline builder (mirrors Anonymizer detection shape)
# ---------------------------------------------------------------------------


def _build_pipeline(num_records: int, latency_ms: float, *, real: bool = False) -> float:
    from data_designer.config.column_configs import (
        CustomColumnConfig,
        LLMTextColumnConfig,
        SamplerColumnConfig,
    )
    from data_designer.config.config_builder import DataDesignerConfigBuilder
    from data_designer.config.custom_column import custom_column_generator
    from data_designer.config.models import ChatCompletionInferenceParams, ModelConfig, ModelProvider
    from data_designer.config.run_config import RunConfig
    from data_designer.config.sampler_params import SamplerType
    from data_designer.engine.dataset_builders.dataset_builder import DatasetBuilder
    from data_designer.engine.model_provider import resolve_model_provider_registry
    from data_designer.engine.resources.resource_provider import create_resource_provider
    from data_designer.engine.resources.seed_reader import SeedReaderRegistry
    from data_designer.engine.secret_resolver import CompositeResolver, EnvironmentResolver, PlaintextResolver
    from data_designer.engine.storage.artifact_storage import ArtifactStorage

    # -- custom column generators (mimic Anonymizer detection steps) --

    @custom_column_generator(required_columns=["raw_detected"], side_effect_columns=["tag_notation"])
    def parse_entities(row: dict[str, Any]) -> dict[str, Any]:
        raw = row.get("raw_detected", "")
        row["seed_entities"] = raw  # pass through
        row["tag_notation"] = "xml"
        return row

    @custom_column_generator(required_columns=["seed_entities"], side_effect_columns=["seed_tagged_text"])
    def prepare_validation(row: dict[str, Any]) -> dict[str, Any]:
        row["seed_tagged_text"] = f"<tagged>{row.get('seed_entities', '')}</tagged>"
        row["seed_candidates"] = row.get("seed_entities", "")
        return row

    @custom_column_generator(required_columns=["validation_decisions", "seed_candidates"])
    def enrich_decisions(row: dict[str, Any]) -> dict[str, Any]:
        row["validated_entities"] = row.get("validation_decisions", "")
        return row

    @custom_column_generator(
        required_columns=["seed_entities", "augmented_entities"],
        side_effect_columns=["merged_tagged_text", "validation_candidates"],
    )
    def merge_entities(row: dict[str, Any]) -> dict[str, Any]:
        row["merged_entities"] = f"{row.get('seed_entities', '')}+{row.get('augmented_entities', '')}"
        row["merged_tagged_text"] = f"<merged>{row.get('seed_entities', '')}</merged>"
        row["validation_candidates"] = row.get("seed_entities", "")
        return row

    @custom_column_generator(
        required_columns=["merged_entities", "validated_entities"],
        side_effect_columns=["tagged_text"],
    )
    def finalize(row: dict[str, Any]) -> dict[str, Any]:
        row["detected_entities"] = row.get("merged_entities", "")
        row["tagged_text"] = f"<final>{row.get('merged_entities', '')}</final>"
        return row

    # -- config --

    if real:
        alias = REAL_MODEL_ALIAS
        model_configs = [
            ModelConfig(
                alias=alias,
                model=REAL_MODEL,
                provider="nvidia",
                inference_parameters=ChatCompletionInferenceParams(
                    max_parallel_requests=16, max_tokens=4096, temperature=0.3
                ),
            ),
        ]
        provider = ModelProvider(
            name="nvidia", endpoint=NVIDIA_ENDPOINT, provider_type="openai", api_key="NVIDIA_API_KEY"
        )
    else:
        alias = MODEL_ALIAS
        model_configs = [
            ModelConfig(
                alias=alias,
                model="mock-model",
                provider="mock-provider",
                inference_parameters=ChatCompletionInferenceParams(max_parallel_requests=16),
                skip_health_check=True,
            ),
        ]
        provider = ModelProvider(
            name="mock-provider", endpoint="https://mock.local", provider_type="openai", api_key="mock-key"
        )

    builder = DataDesignerConfigBuilder(model_configs=model_configs)
    # Pipeline: sampler -> LLM detect -> parse -> prepare -> LLM validate ->
    #           enrich -> LLM augment -> merge -> finalize
    builder.add_column(
        SamplerColumnConfig(
            name="text_input",
            sampler_type=SamplerType.CATEGORY,
            params={
                "values": [
                    "Alice works at Acme Corp in Seattle.",
                    "Bob visited Paris last summer.",
                    "Carol, 42, lives in Portland.",
                ]
            },
        )
    )
    builder.add_column(
        LLMTextColumnConfig(name="raw_detected", model_alias=alias, prompt="Detect entities in: {{ text_input }}")
    )
    builder.add_column(CustomColumnConfig(name="seed_entities", generator_function=parse_entities))
    builder.add_column(CustomColumnConfig(name="seed_candidates", generator_function=prepare_validation))
    builder.add_column(
        LLMTextColumnConfig(name="validation_decisions", model_alias=alias, prompt="Validate: {{ seed_tagged_text }}")
    )
    builder.add_column(CustomColumnConfig(name="validated_entities", generator_function=enrich_decisions))
    builder.add_column(
        LLMTextColumnConfig(
            name="augmented_entities", model_alias=alias, prompt="Find untagged entities in: {{ seed_tagged_text }}"
        )
    )
    builder.add_column(CustomColumnConfig(name="merged_entities", generator_function=merge_entities))
    builder.add_column(CustomColumnConfig(name="detected_entities", generator_function=finalize))

    run_config = RunConfig(
        buffer_size=num_records,
        disable_early_shutdown=True,
        max_conversation_restarts=0,
        max_conversation_correction_steps=0,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_storage = ArtifactStorage(artifact_path=temp_dir, dataset_name="benchmark")
        resource_provider = create_resource_provider(
            artifact_storage=artifact_storage,
            model_configs=builder.model_configs,
            secret_resolver=CompositeResolver([EnvironmentResolver(), PlaintextResolver()]),
            model_provider_registry=resolve_model_provider_registry([provider], default_provider_name=provider.name),
            seed_reader_registry=SeedReaderRegistry(readers=[]),
            seed_dataset_source=None,
            run_config=run_config,
        )
        dataset_builder = DatasetBuilder(data_designer_config=builder.build(), resource_provider=resource_provider)

        if real:
            start = time.perf_counter()
            dataset_builder.build(num_records=num_records)
            elapsed = time.perf_counter() - start
        else:
            with _patch_llm(latency_ms=latency_ms):
                start = time.perf_counter()
                dataset_builder.build(num_records=num_records)
                elapsed = time.perf_counter() - start

    return elapsed


# ---------------------------------------------------------------------------
# Subprocess runner (engine mode requires fresh process)
# ---------------------------------------------------------------------------


def _run_subprocess(engine: str, num_records: int, latency_ms: float, *, real: bool = False) -> float:
    env = os.environ.copy()
    if engine == "async":
        env["DATA_DESIGNER_ASYNC_ENGINE"] = "1"
    else:
        env.pop("DATA_DESIGNER_ASYNC_ENGINE", None)

    # Use the same Python interpreter that launched this script.
    python = sys.executable
    cmd = [
        python,
        __file__,
        "--mode",
        "run",
        "--engine",
        engine,
        "--num-records",
        str(num_records),
        "--latency-ms",
        str(latency_ms),
    ]
    if real:
        cmd.append("--real")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)

    if result.returncode != 0:
        print(f"  FAILED ({engine}):", file=sys.stderr)
        print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr, file=sys.stderr)
        raise RuntimeError(f"{engine} subprocess failed")

    for line in reversed(result.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return float(line.removeprefix(RESULT_PREFIX))

    raise RuntimeError(f"No result from {engine} subprocess.\nstdout:\n{result.stdout[-300:]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark sync vs async engine")
    parser.add_argument("--mode", choices=["compare", "run"], default="compare")
    parser.add_argument("--engine", choices=["sync", "async"], default="sync")
    parser.add_argument("--num-records", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--latency", action="store_true", help="Simulate 50ms LLM latency per call")
    parser.add_argument("--latency-ms", type=float, default=0.0)
    parser.add_argument("--real", action="store_true", help="Use real NIM endpoint instead of mock LLM")
    args = parser.parse_args()

    if args.mode == "run":
        if args.real:
            _load_dotenv()
        elapsed = _build_pipeline(args.num_records, args.latency_ms, real=args.real)
        print(f"{RESULT_PREFIX}{elapsed:.6f}")
        return

    # -- compare mode --
    latency_ms = 50.0 if args.latency else 0.0
    use_real = args.real
    n = args.num_records
    iters = args.iterations

    mode_label = (
        "real NIM endpoint" if use_real else ("mock LLM" + (f", {latency_ms:.0f}ms latency" if latency_ms else ""))
    )
    print(f"Benchmarking {n} records, {iters} iterations, {mode_label}")
    print()

    sync_times: list[float] = []
    async_times: list[float] = []

    for i in range(1, iters + 1):
        print(f"  [{i}/{iters}] sync  ...", end="", flush=True)
        t = _run_subprocess("sync", n, latency_ms, real=use_real)
        sync_times.append(t)
        print(f" {t:.3f}s")

        print(f"  [{i}/{iters}] async ...", end="", flush=True)
        t = _run_subprocess("async", n, latency_ms, real=use_real)
        async_times.append(t)
        print(f" {t:.3f}s")

    sync_mean = statistics.mean(sync_times)
    async_mean = statistics.mean(async_times)
    speedup = sync_mean / async_mean if async_mean > 0 else float("inf")

    print()
    print(f"  sync  mean: {sync_mean:.3f}s" + (f" (stdev {statistics.stdev(sync_times):.3f}s)" if iters > 1 else ""))
    print(f"  async mean: {async_mean:.3f}s" + (f" (stdev {statistics.stdev(async_times):.3f}s)" if iters > 1 else ""))
    print(f"  speedup:    {speedup:.2f}x")


if __name__ == "__main__":
    main()
