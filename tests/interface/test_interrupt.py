# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ctrl-C during ``run()`` and ``preview()`` stops promptly and cancels in-flight model requests, as in 0.4.0.

Each case runs a real Anonymizer and DataDesigner in a subprocess whose HTTP transport is scripted to hold one
kind of model request open, sends it real SIGINTs, and checks the event sequence. The script uses only the public
API and DataDesigner's transport hook, so it runs unchanged against 0.4.0; the expected sequences were recorded
there. Runtime: about 3 s per case.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.compat

_SCRIPT = r"""
import asyncio, json, signal, sys, tempfile, time
from pathlib import Path

import httpx, pandas as pd, yaml
from data_designer.config.models import ModelConfig, ModelProvider
from data_designer.interface.data_designer import DataDesigner

from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, Redact, Rewrite, Substitute

signal.signal(signal.SIGINT, signal.default_int_handler)
log = open(sys.argv[1], "a", buffering=1)
case = sys.argv[2]
held = {"plain": "detector", "asyncio_run": "map", "second": "rewriter"}[case]


def emit(event):
    log.write(json.dumps({"t": time.monotonic(), "event": event}) + "\n")


def completion(model, content):
    choice = {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
    usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    body = {"id": "s", "object": "chat.completion", "created": 0, "model": model, "choices": [choice], "usage": usage}
    return httpx.Response(200, json=body)


def fenced(value):
    return "```json\n" + json.dumps(value) + "\n```"


names = ["Alice", "Bob", "Carol"]


async def hold(model):
    emit("request")
    while True:
        try:
            await asyncio.sleep(60)
            break
        except asyncio.CancelledError:
            if case != "second":
                emit("cancelled")
                raise
            emit("cancel_ignored")  # a provider call that does not stop: only a second Ctrl-C ends the wait
    emit("answered")
    return completion(model, "{}")


async def handler(request):
    body = json.loads(request.content)
    model, messages = body["model"], json.dumps(body["messages"])
    if model == "detector":
        if held == "detector":
            return await hold(model)
        name = next(name for name in names if name + " waved." in messages)
        entity = {"text": name, "label": "first_name", "start": 0, "end": len(name), "score": 0.9}
        return completion(model, json.dumps({"entities": [entity]}))
    if model == "latent":
        return completion(model, fenced({"latent_entities": []}))
    if model == "rewriter":
        return await hold(model) if held == "rewriter" else completion(model, "{}")
    if "Validate entity tags" in messages:
        ids = dict.fromkeys(f"first_name_0_{len(name)}" for name in names if name in messages)
        return completion(model, fenced({"decisions": [{"id": item, "decision": "keep"} for item in ids]}))
    if "Find untagged sensitive entities" in messages:
        return completion(model, fenced({"entities": []}))
    if "Generate synthetic replacements" in messages:
        return await hold(model)
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
models = [ModelConfig(alias=n, model=n, provider="s", skip_health_check=True) for n in aliases]
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
data = AnonymizerInput(source=str(source), text_column="text")


def call():
    if case == "plain":
        return anonymizer.run(config=AnonymizerConfig(replace=Redact(), emit_telemetry=False), data=data)
    if case == "asyncio_run":
        config = AnonymizerConfig(replace=Substitute(), emit_telemetry=False)
        return anonymizer.preview(config=config, data=data, num_records=len(names))
    return anonymizer.run(config=AnonymizerConfig(rewrite=Rewrite(), emit_telemetry=False), data=data)


async def handler_task():
    # A service handler that calls the synchronous API from inside its running event loop.
    return call()


try:
    asyncio.run(handler_task()) if case == "asyncio_run" else call()
    emit("finished")
except KeyboardInterrupt:
    emit("interrupted")
    raise
"""

# What 0.4.0 does (recorded against main f4f2b73d). The first Ctrl-C reaches DataDesigner's wait on the caller's
# thread, which cancels its scheduler, waits for the in-flight requests to unwind and re-raises. Inside a caller's
# asyncio.run, the Runner's own SIGINT handler takes the first Ctrl-C and only cancels the caller's task, which is
# blocked in the synchronous call, so a second one is needed. A request that ignores cancellation keeps
# DataDesigner's post-cancel wait open until a second Ctrl-C ends it.
_EXPECTED = {
    "plain": ["request", "sigint", "cancelled", "interrupted"],
    "asyncio_run": ["request", "sigint", "sigint", "cancelled", "interrupted"],
    "second": ["request", "sigint", "cancel_ignored", "sigint", "interrupted"],
}


def _collapse(events: list[str]) -> list[str]:
    """Concurrent requests for several rows log the same event once per row; each SIGINT stays its own step."""
    return [
        event for index, event in enumerate(events) if index == 0 or event == "sigint" or events[index - 1] != event
    ]


@pytest.mark.parametrize("case", ["plain", "asyncio_run", "second"])
def test_ctrl_c_stops_the_run_and_cancels_in_flight_requests(tmp_path: Path, case: str) -> None:
    script, log = tmp_path / "interrupted.py", tmp_path / "events.jsonl"
    script.write_text(_SCRIPT)
    child = subprocess.Popen(
        [sys.executable, str(script), str(log), case],
        env={**os.environ, "NEMO_TELEMETRY_ENABLED": "false"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    def events() -> list[str]:
        return [json.loads(line)["event"] for line in log.read_text().splitlines()] if log.exists() else []

    def interrupt() -> float:
        with log.open("a") as handle:
            handle.write(json.dumps({"t": time.monotonic(), "event": "sigint"}) + "\n")
        child.send_signal(signal.SIGINT)
        return time.monotonic()

    def wait(seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)

    try:
        deadline = time.monotonic() + 60
        while "request" not in events() and child.poll() is None:
            assert time.monotonic() < deadline, "no held request started"
            time.sleep(0.02)
        time.sleep(0.3)
        assert child.poll() is None, child.stderr.read().decode() if child.stderr else ""
        sent = interrupt()
        if case != "plain":
            wait(1.0)
            assert child.poll() is None, "the first Ctrl-C ended the run"
            sent = interrupt()
        wait(10)
        elapsed = time.monotonic() - sent
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()

    assert elapsed < 2, f"run ended {elapsed:.2f}s after the last SIGINT"
    assert child.returncode == -signal.SIGINT
    recorded = events()
    assert _collapse(recorded) == _EXPECTED[case]
    if case != "second":
        assert recorded.count("request") == recorded.count("cancelled")
