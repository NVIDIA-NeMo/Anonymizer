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


@pytest.mark.parametrize(
    "token",
    [
        "sk-proj-abcdefghijklmnop",
        "nvapi-abcdefghijklmnop",
        "ghp_01234567890123456789",
        "github_pat_0123456789_abcdefghijklmnop",
        "glpat-01234567890123456789",
        "hf_01234567890123456789",
        "npm_01234567890123456789",
        "xoxb-0123456789",
        f"AIza{'0' * 35}",
        f"AKIA{'0' * 16}",
        "Bearer abcdefghijklmnop",
    ],
)
def test_known_credentials_are_removed_before_detection(token: str) -> None:
    detector = FakeDetector()

    result = asyncio.run(ObservationSanitizer(detector).sanitize({"value": token}))

    assert result == {"value": "[REDACTED_SECRET]"}
    assert token not in [text for call in detector.calls for text in call]


def test_sanitizer_batches_unique_leaves() -> None:
    detector = FakeDetector()
    sanitizer = ObservationSanitizer(detector)
    value = {"prompt": "Email ana@example.com", "history": ["Email ana@example.com"]}

    result = asyncio.run(sanitizer.sanitize(value))

    assert result == {
        "prompt": "Email [REDACTED]",
        "history": ["Email [REDACTED]"],
    }
    assert detector.calls == [["prompt", "Email ana@example.com", "history"]]


def test_sensitive_application_keys_fail_closed() -> None:
    detector = FakeDetector()
    sanitizer = ObservationSanitizer(detector)
    value = {"ana@example.com": "Email ana@example.com"}

    with pytest.raises(RuntimeError, match="sensitive mapping key"):
        asyncio.run(sanitizer.sanitize(value))


def test_secret_patterns_cover_preserved_protocol_identifiers() -> None:
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
                    }
                }
            },
            preserve_protocol_values=True,
        )
    )

    request = result["category_profile"]["annotated_request"]
    assert request["model"] == "[REDACTED_SECRET]"
    assert request["provider"] == "[REDACTED_SECRET]"
    observed = [text for call in detector.calls for text in call]
    assert openai_secret not in observed
    assert nvidia_secret not in observed


def test_secret_shaped_mapping_key_fails_before_detection() -> None:
    detector = FakeDetector()
    sanitizer = ObservationSanitizer(detector)
    token = "sk-proj-abcdefghijklmnop"

    with pytest.raises(RuntimeError, match="secret-shaped mapping key"):
        asyncio.run(sanitizer.sanitize({token: "value"}))

    assert detector.calls == []


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


def test_context_dependent_positive_decisions_are_not_reused() -> None:
    class ContextDetector:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def detect(self, texts: list[str]) -> list[list[RedactionSpan]]:
            self.calls.append(texts)
            include_bob = "private customer" in texts
            decisions: list[list[RedactionSpan]] = []
            for text in texts:
                spans = [RedactionSpan(0, 5, "person")] if text == "Alice and Bob" else []
                if text == "Alice and Bob" and include_bob:
                    spans.append(RedactionSpan(10, 13, "person"))
                decisions.append(spans)
            return decisions

    detector = ContextDetector()
    sanitizer = ObservationSanitizer(detector)

    first = asyncio.run(sanitizer.sanitize({"prompt": "Alice and Bob"}))
    second = asyncio.run(sanitizer.sanitize({"context": "private customer", "prompt": "Alice and Bob"}))

    assert first == {"prompt": "[REDACTED] and Bob"}
    assert second == {"context": "private customer", "prompt": "[REDACTED] and [REDACTED]"}
    assert detector.calls == [
        ["prompt", "Alice and Bob"],
        ["context", "private customer", "prompt", "Alice and Bob"],
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


async def test_close_drains_cancelled_detection_before_closing_backend() -> None:
    started = threading.Event()
    release = threading.Event()
    backend_closed = threading.Event()

    class ClosableDetector:
        def detect(self, texts: list[str]) -> list[list[RedactionSpan]]:
            started.set()
            release.wait(timeout=5)
            return [[] for _ in texts]

        def close(self) -> None:
            backend_closed.set()

    sanitizer = ObservationSanitizer(ClosableDetector())
    pending = asyncio.create_task(sanitizer.sanitize({"prompt": "copied event"}))
    assert await asyncio.to_thread(started.wait, 2)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    closing = asyncio.create_task(asyncio.to_thread(sanitizer.close))
    await asyncio.sleep(0.05)
    assert not closing.done()
    assert not backend_closed.is_set()

    release.set()
    await closing
    assert backend_closed.is_set()
