# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ctrl-C during Rewrite and transient provider errors during Substitute, compared with 0.4.0.

Only the HTTP transport is scripted; DataDesigner and the graph run for real. The expected rows, failed
records and HTTP call counts were recorded by running the same stubs against main (0.4.0 behavior): DataDesigner
retries a 5xx, 429 or timeout on the map request and keeps the record; after three failed tries of one task it
drops only that record. Runtime: about 2 s per interrupt case (a subprocess) and under 4 s per retry case.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest
import yaml
from data_designer.config.models import ModelConfig, ModelProvider
from data_designer.engine.dataset_builders.utils.async_concurrency import ensure_async_engine_loop
from data_designer.interface.data_designer import DataDesigner

from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, Substitute

pytestmark = pytest.mark.compat

_INTERRUPTED_REWRITE = r"""
import asyncio, json, signal, sys, tempfile, time
from pathlib import Path

import httpx, pandas as pd, yaml
from data_designer.config.models import ModelConfig, ModelProvider
from data_designer.interface.data_designer import DataDesigner

from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, Rewrite

signal.signal(signal.SIGINT, signal.default_int_handler)
log = open(sys.argv[1], "a", buffering=1)
stage, combined = sys.argv[2], sys.argv[3] == "combined"


def emit(**fields):
    log.write(json.dumps({"t": time.monotonic(), **fields}) + "\n")


def completion(model, content):
    choice = {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
    usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    body = {"id": "s", "object": "chat.completion", "created": 0, "model": model, "choices": [choice], "usage": usage}
    return httpx.Response(200, json=body)


def fenced(value):
    return "```json\n" + json.dumps(value) + "\n```"


names = ["Alice", "Bob", "Carol"]


async def slow(model):
    emit(event="request")
    try:
        await asyncio.sleep(60)
    except asyncio.CancelledError:
        emit(event="cancelled")
        raise
    emit(event="answered")
    return completion(model, "{}")


async def handler(request):
    body = json.loads(request.content)
    model, messages = body["model"], json.dumps(body["messages"])
    if model == "detector":
        name = next(name for name in names if name + " waved." in messages)
        entity = {"text": name, "label": "first_name", "start": 0, "end": len(name), "score": 0.9}
        return completion(model, json.dumps({"entities": [entity]}))
    if model == "latent":
        return completion(model, fenced({"latent_entities": []}))
    if model == "rewriter":
        return await slow(model) if stage == "rewrite" else completion(model, "{}")
    if "Validate entity tags" in messages:
        ids = dict.fromkeys(f"first_name_0_{len(name)}" for name in names if name in messages)
        return completion(model, fenced({"decisions": [{"id": item, "decision": "keep"} for item in ids]}))
    if "Find untagged sensitive entities" in messages:
        return completion(model, fenced({"entities": []}))
    if "Generate synthetic replacements" in messages:
        return await slow(model)
    return completion(model, "Hello!")


root = Path(tempfile.mkdtemp())
designer = DataDesigner(
    artifact_path=root / "artifacts",
    model_providers=[ModelProvider(name="s", endpoint="http://scripted.invalid/v1", provider_type="openai", api_key="x")],
    auto_configure_logging=False,
)
original = designer._create_resource_provider


def resources(*args, **kwargs):
    provider = original(*args, **kwargs)
    factory = provider.model_registry._model_facade_factory

    def facade(*args, **kwargs):
        result = factory(*args, **kwargs)
        result._client._inner._transport = httpx.MockTransport(handler)
        return result

    provider.model_registry._model_facade_factory = facade
    return provider


designer._create_resource_provider = resources
aliases = ("detector", "llm", "rewriter", "latent")
models = [ModelConfig(alias=n, model=n, provider="s", skip_health_check=n != "llm") for n in aliases]
roles = {"entity_detector": "detector", "entity_validator": ["llm"], "entity_augmenter": "llm", "latent_detector": "latent"}
steps = ("domain_classifier", "disposition_analyzer", "meaning_extractor", "qa_generator", "rewriter", "evaluator", "repairer")
configs = {
    "model_configs": [model.model_dump(mode="json", exclude_none=True) for model in models],
    "selected_models": {
        "detection": roles,
        "replace": {"replacement_generator": "llm"},
        "rewrite": {step: "rewriter" for step in steps},
    },
}
source = root / "input.csv"
pd.DataFrame({"text": [name + " waved." for name in names]}).to_csv(source, index=False)
anonymizer = Anonymizer(data_designer=designer, model_configs=yaml.safe_dump(configs))
config = AnonymizerConfig(rewrite=Rewrite(use_combined_graph=combined))
try:
    anonymizer.run(config=config, data=AnonymizerInput(source=str(source), text_column="text"))
    emit(event="finished")
except KeyboardInterrupt as error:
    from anonymizer.engine.harness.kernel import retained_execution

    execution = retained_execution(error)
    emit(
        event="interrupted",
        requests=None if execution is None else sorted(t.category for t in execution.requests.terminals),
        tasks=None if execution is None else sorted(t.category for t in execution.record.terminals),
    )
    raise
"""


@pytest.mark.parametrize(
    "stage,schedule",
    [("map", "staged"), ("rewrite", "staged"), ("rewrite", "combined")],
)
def test_interrupt_during_rewrite_cancels_in_flight_requests_promptly(
    tmp_path: Path, stage: str, schedule: str
) -> None:
    script, log = tmp_path / "interrupted_rewrite.py", tmp_path / "events.jsonl"
    script.write_text(_INTERRUPTED_REWRITE)
    environment = {**os.environ, "NEMO_TELEMETRY_ENABLED": "false"}
    child = subprocess.Popen(
        [sys.executable, str(script), str(log), stage, schedule],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    def events() -> list[dict[str, Any]]:
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    try:
        deadline = time.monotonic() + 60
        while "request" not in [item["event"] for item in events()] and child.poll() is None:
            assert time.monotonic() < deadline, "no slow request started"
            time.sleep(0.02)
        time.sleep(0.3)
        assert child.poll() is None, child.stderr.read().decode() if child.stderr else ""
        interrupted = time.monotonic()
        child.send_signal(signal.SIGINT)
        while child.poll() is None and time.monotonic() - interrupted < 10:
            time.sleep(0.01)
        elapsed = time.monotonic() - interrupted
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()

    recorded = events()
    names = [item["event"] for item in recorded]
    # 0.4.0 stops within about half a second; the graph route used to wait for, or ignore, the interrupt.
    assert elapsed < 2, f"run ended {elapsed:.2f}s after SIGINT"
    assert child.returncode == -signal.SIGINT
    assert names.count("request") >= 1 and names.count("request") == names.count("cancelled")
    assert "answered" not in names and "finished" not in names
    (outcome,) = [item for item in recorded if item["event"] == "interrupted"]
    if outcome["tasks"] is not None:  # 0.4.0 attaches no execution record
        # Rewrite's model requests are DataDesigner-owned, not graph physical requests: none is claimed as
        # settled, and each of the three interrupted records' tasks closes cancelled.
        assert "success" not in outcome["requests"]
        assert outcome["tasks"].count("cancelled") == 3


_NAMES = ("Zorba", "Zorbb", "Zorbc")


def _completion(model: str, content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "scripted",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


def _fenced(value: dict[str, Any]) -> str:
    return "```json\n" + json.dumps(value) + "\n```"


class _TransientMap:
    """Fails the first ``failures`` map requests for the first record with ``kind``; everything else succeeds."""

    def __init__(self, kind: str, failures: int) -> None:
        self.kind = kind
        self.failures = failures
        self.calls: Counter[str] = Counter()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        model, messages = body["model"], json.dumps(body["messages"])
        if model == "detector":
            self.calls["detector"] += 1
            name = next(name for name in _NAMES if f"{name} waved." in messages)
            entity = {"text": name, "label": "first_name", "start": 0, "end": len(name), "score": 0.9}
            return _completion(model, json.dumps({"entities": [entity]}))
        if "Validate entity tags" in messages:
            self.calls["validator"] += 1
            ids = dict.fromkeys(f"first_name_0_{len(name)}" for name in _NAMES if name in messages)
            return _completion(model, _fenced({"decisions": [{"id": item, "decision": "keep"} for item in ids]}))
        if "Find untagged sensitive entities" in messages:
            self.calls["augmenter"] += 1
            return _completion(model, _fenced({"entities": []}))
        if "Generate synthetic replacements" not in messages:
            self.calls["health_check"] += 1
            return _completion(model, "Hello!")
        self.calls["map"] += 1
        if _NAMES[0] in messages and self.calls["map_error"] < self.failures:
            self.calls["map_error"] += 1
            if self.kind == "timeout":
                raise httpx.ReadTimeout("scripted timeout", request=request)
            # A short Retry-After keeps DataDesigner's rate-limit cooldown (2 s by default) fast for the test.
            headers = {"retry-after": "0.01"} if self.kind == "429" else {}
            return httpx.Response(int(self.kind), json={"error": {"message": "scripted"}}, headers=headers)
        replacements = [
            {"original": name, "label": "first_name", "synthetic": "Maya"} for name in _NAMES if name in messages
        ]
        return _completion(model, _fenced({"replacements": replacements}))


@pytest.fixture
def transient_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[DataDesigner, list[_TransientMap]]]:
    scripts: list[_TransientMap] = []
    clients: list[Any] = []
    designer = DataDesigner(
        artifact_path=tmp_path / "artifacts",
        model_providers=[
            ModelProvider(name="scripted", endpoint="http://scripted.invalid/v1", provider_type="openai", api_key="x")
        ],
        auto_configure_logging=False,
    )
    original = designer._create_resource_provider

    def resources(*args: Any, **kwargs: Any) -> Any:
        provider: Any = original(*args, **kwargs)
        factory = provider.model_registry._model_facade_factory

        def facade(*args: Any, **kwargs: Any) -> Any:
            result: Any = factory(*args, **kwargs)
            result._client._inner._transport = httpx.MockTransport(lambda request: scripts[0](request))
            clients.append(result._client)
            return result

        provider.model_registry._model_facade_factory = facade
        return provider

    monkeypatch.setattr(designer, "_create_resource_provider", resources)
    try:
        yield designer, scripts
    finally:

        async def close() -> None:
            for client in clients:
                await client.aclose()

        asyncio.run_coroutine_threadsafe(close(), ensure_async_engine_loop()).result(timeout=30)


@pytest.mark.parametrize(
    "kind,failures,kept,map_calls",
    [
        ("500", 1, 3, 4),  # one retry, the record is kept
        ("429", 1, 3, 4),
        ("timeout", 1, 3, 4),
        ("500", 3, 2, 5),  # three failed tries of one task: DataDesigner drops only that record
        ("timeout", 3, 2, 5),
        # 429 is never dropped: storms past the admitted leaves (6 parser attempts + 11 retries) keep the record.
        ("429", 17, 3, 20),
        ("429", 25, 3, 28),
        ("429", 60, 3, 63),
    ],
)
def test_substitute_map_retries_transient_errors_like_release(
    transient_map: tuple[DataDesigner, list[_TransientMap]],
    tmp_path: Path,
    kind: str,
    failures: int,
    kept: int,
    map_calls: int,
) -> None:
    designer, scripts = transient_map
    script = _TransientMap(kind, failures)
    scripts.append(script)
    models = [
        ModelConfig(alias=name, model=name, provider="scripted", skip_health_check=name == "detector")
        for name in ("detector", "llm")
    ]
    roles = {
        "entity_detector": "detector",
        "entity_validator": ["llm"],
        "entity_augmenter": "llm",
        "latent_detector": "llm",
    }
    configs = {
        "model_configs": [model.model_dump(mode="json", exclude_none=True) for model in models],
        "selected_models": {"detection": roles, "replace": {"replacement_generator": "llm"}},
    }
    source = tmp_path / "input.csv"
    pd.DataFrame({"text": [f"{name} waved." for name in _NAMES]}).to_csv(source, index=False)

    result = Anonymizer(data_designer=designer, model_configs=yaml.safe_dump(configs)).run(
        config=AnonymizerConfig(replace=Substitute()), data=AnonymizerInput(source=str(source), text_column="text")
    )

    assert result.dataframe["text_replaced"].tolist() == ["Maya waved."] * kept
    assert [(record.step, record.reason) for record in result.failed_records] == [
        ("replace-map-generation", "Record missing from workflow output")
    ] * (3 - kept)
    assert script.calls == {
        "health_check": 2,
        "detector": 3,
        "validator": 3,
        "augmenter": 3,
        "map": map_calls,
        "map_error": failures,
    }
    execution = getattr(result, "_execution", None)
    if execution is not None:
        # Each attempt is its own dispatch and settlement; a retry follows its failed predecessor on the chain.
        # Attempts beyond the record's admitted leaves (17) are DataDesigner-owned and unaccounted, which is
        # honest only because the invocation declares no request upper bound.
        accounted = min(failures, 17)
        maps = [
            item for item in execution.requests.dispatches if item.child is not None and item.child.role == "generation"
        ]
        # The two other records' requests, each accounted failure, and the record's successful final attempt
        # unless that attempt fell past the allowance (DataDesigner-owned) or the record was dropped.
        assert len(maps) == 2 + accounted + (kept == 3 and failures < 17)
        assert sorted(item.child.source_kind for item in maps) == sorted(["initial"] * 3 + ["retry"] * (len(maps) - 3))
        terminals = Counter(item.category for item in execution.requests.terminals)
        assert terminals["failure"] == accounted and not execution.requests.defects
        assert execution._execution.context.prepared.declared_request_upper_bound is None
