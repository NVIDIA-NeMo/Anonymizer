# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading

import pytest

from nemo_anonymizer_relay.backend import RedactionSpan
from nemo_anonymizer_relay.sanitizer import ObservationSanitizer


class FakeDetector:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def detect(self, texts: list[str]) -> list[list[RedactionSpan]]:
        self.calls.append(texts)
        decisions: list[list[RedactionSpan]] = []
        for text in texts:
            start = text.find("ana@example.com")
            decisions.append([] if start < 0 else [RedactionSpan(start, start + len("ana@example.com"), "email")])
        return decisions


def test_sanitizer_batches_unique_leaves_and_caches_span_decisions() -> None:
    detector = FakeDetector()
    sanitizer = ObservationSanitizer(detector)
    value = {"prompt": "Email ana@example.com", "history": ["Email ana@example.com"]}

    first = asyncio.run(sanitizer.sanitize(value))
    second = asyncio.run(sanitizer.sanitize(value))

    assert (
        first
        == second
        == {
            "prompt": "Email [REDACTED]",
            "history": ["Email [REDACTED]"],
        }
    )
    assert detector.calls == [
        ["prompt", "Email ana@example.com", "history"],
        ["prompt", "history"],
    ]


def test_secret_patterns_cover_preserved_protocol_identifiers_and_keys() -> None:
    detector = FakeDetector()
    sanitizer = ObservationSanitizer(detector)
    openai_secret = "sk-proj-abcdefghijklmnop"
    nvidia_secret = "nvapi-abcdefghijklmnop"

    result = asyncio.run(
        sanitizer.sanitize(
            {
                "category_profile": {
                    "annotated_request": {
                        "model": openai_secret,
                        "provider": nvidia_secret,
                        "messages": [{"role": "user", "content": "safe"}],
                        openai_secret: "structural key",
                    }
                }
            }
        )
    )

    request = result["category_profile"]["annotated_request"]
    assert request["model"] == "[REDACTED_SECRET]"
    assert request["provider"] == "[REDACTED_SECRET]"
    assert "[REDACTED_SECRET]" in request
    observed = [text for call in detector.calls for text in call]
    assert openai_secret not in observed
    assert nvidia_secret not in observed


def test_secret_bearing_keys_redact_values_without_substring_matches() -> None:
    detector = FakeDetector()
    sanitizer = ObservationSanitizer(detector)
    secret_values = {
        "Authorization": "opaque-value",
        "api-key": "another-opaque-value",
        "x-api-key": "provider-specific-value",
        "OPENAI_API_KEY": "opaque-openai-value",
        "aws_secret_access_key": "opaque-aws-value",
        "refresh_token": "opaque-refresh-value",
        "openaiApiKey": "opaque-camel-openai-value",
        "awsSecretAccessKey": "opaque-camel-aws-value",
        "sessionToken": "opaque-session-value",
    }
    benign = {
        "password_hint": "benign structural label",
        "passwordHint": "another benign structural label",
    }

    result = asyncio.run(
        sanitizer.sanitize(
            {
                "metadata": {
                    **secret_values,
                    "clientSecret": {"nested": "must not survive"},
                    **benign,
                }
            }
        )
    )

    assert result["metadata"] == {
        **dict.fromkeys(secret_values, "[REDACTED_SECRET]"),
        "clientSecret": "[REDACTED_SECRET]",
        **benign,
    }
    observed = [text for call in detector.calls for text in call]
    assert (set(secret_values.values()) | {"must not survive"}).isdisjoint(observed)


def test_empty_decisions_are_rechecked_in_later_contexts() -> None:
    detector = FakeDetector()
    sanitizer = ObservationSanitizer(detector)

    asyncio.run(sanitizer.sanitize({"prompt": "ordinary label"}))
    asyncio.run(sanitizer.sanitize({"context": "private customer", "prompt": "ordinary label"}))

    assert detector.calls == [
        ["prompt", "ordinary label"],
        ["context", "private customer", "prompt", "ordinary label"],
    ]


async def test_cancelled_detection_does_not_queue_more_executor_work() -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    class BlockingDetector:
        def detect(self, texts: list[str]) -> list[list[RedactionSpan]]:
            started.set()
            release.wait(timeout=5)
            finished.set()
            return [[] for _ in texts]

    sanitizer = ObservationSanitizer(BlockingDetector())
    first = asyncio.create_task(sanitizer.sanitize({"prompt": "first"}))
    assert await asyncio.to_thread(started.wait, 2)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    with pytest.raises(RuntimeError, match="still completing prior work"):
        await sanitizer.sanitize({"prompt": "second"})

    release.set()
    assert await asyncio.to_thread(finished.wait, 2)
    await asyncio.sleep(0)
    assert await sanitizer.sanitize({"prompt": "third"}) == {"prompt": "third"}
    sanitizer.close()
