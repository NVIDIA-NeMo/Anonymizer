# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from nemo_relay_plugin import PluginContext

from nemo_anonymizer_relay import worker
from nemo_anonymizer_relay.backend import RedactionSpan

SURFACES = (
    "register_mark_sanitize_guardrail",
    "register_scope_sanitize_start_guardrail",
    "register_scope_sanitize_end_guardrail",
    "register_tool_sanitize_request_guardrail",
    "register_tool_sanitize_response_guardrail",
    "register_llm_sanitize_request_guardrail",
    "register_llm_sanitize_response_guardrail",
)


class Context:
    def __init__(self) -> None:
        self.registrations: dict[str, tuple[str, Any, int]] = {}

    def __getattr__(self, surface: str) -> Any:
        if surface not in SURFACES:
            raise AttributeError(surface)

        def register(name: str, callback: Any, *, priority: int = 0) -> None:
            self.registrations[surface] = (name, callback, priority)

        return register

    def callback(self, surface: str) -> Any:
        return self.registrations[surface][1]


def valid_config() -> dict[str, Any]:
    return {
        "evaluator_api_key_env": "TEST_EVALUATOR_KEY",
        "secret_env": {"TEST_EVALUATOR_KEY": "private-value"},
    }


def install_runtime_fakes(monkeypatch: pytest.MonkeyPatch) -> Any:
    async def passthrough(value: Any, **_kwargs: Any) -> Any:
        return deepcopy(value)

    sanitizer = SimpleNamespace(sanitize=AsyncMock(side_effect=passthrough), close=MagicMock())
    monkeypatch.setattr(worker, "AnonymizerBackend", MagicMock())
    monkeypatch.setattr(worker, "ObservationSanitizer", MagicMock(return_value=sanitizer))
    monkeypatch.setattr(worker, "require_api_keys", MagicMock())
    return sanitizer


def test_normalized_config_rejects_unknown_or_unsafe_values() -> None:
    with pytest.raises(TypeError, match="configuration must be"):
        worker.normalized_config(None)
    with pytest.raises(ValueError, match="unknown configuration"):
        worker.normalized_config({"surprise": True})
    with pytest.raises(TypeError, match="priority"):
        worker.normalized_config({"priority": True})
    with pytest.raises(TypeError, match="positive integer"):
        worker.normalized_config({"max_text_bytes": 0})
    with pytest.raises(ValueError, match="threshold"):
        worker.normalized_config({"threshold": 1.1})
    with pytest.raises(ValueError, match="secret_env"):
        worker.normalized_config({"secret_env": {"BAD-NAME": "value"}})


async def test_worker_registers_rampart_shaped_sanitizer_surfaces(monkeypatch) -> None:
    sanitizer = install_runtime_fakes(monkeypatch)
    context = Context()
    plugin = worker.NemoAnonymizerWorker()

    await plugin.register(cast(PluginContext, context), valid_config())

    assert set(context.registrations) == set(SURFACES)
    assert {registration[2] for registration in context.registrations.values()} == {100}
    fields = {"data": {"owner": "Marisol Vega"}, "category_profile": None, "metadata": {}}
    result = await context.callback("register_mark_sanitize_guardrail")(
        {"kind": "mark", "name": "custom", "category": "custom"},
        fields,
    )
    assert result == fields
    sanitizer.sanitize.assert_awaited()

    await plugin.close()
    sanitizer.close.assert_called_once_with()


async def test_secret_environment_is_worker_scoped(monkeypatch) -> None:
    install_runtime_fakes(monkeypatch)
    monkeypatch.setenv("TEST_EVALUATOR_KEY", "prior-value")
    plugin = worker.NemoAnonymizerWorker()

    await plugin.register(cast(PluginContext, Context()), valid_config())
    assert os.environ["TEST_EVALUATOR_KEY"] == "private-value"
    await plugin.close()
    assert os.environ["TEST_EVALUATOR_KEY"] == "prior-value"


async def test_all_observability_surfaces_return_redacted_copies(monkeypatch) -> None:
    class Detector:
        def detect(self, texts: list[str]) -> list[list[RedactionSpan]]:
            results: list[list[RedactionSpan]] = []
            for text in texts:
                start = text.find("Marisol Vega")
                results.append([RedactionSpan(start, start + 12, "person")] if start >= 0 else [])
            return results

    monkeypatch.setattr(worker, "AnonymizerBackend", MagicMock(return_value=Detector()))
    monkeypatch.setattr(worker, "require_api_keys", MagicMock())
    plugin = worker.NemoAnonymizerWorker()
    context = Context()
    await plugin.register(cast(PluginContext, context), valid_config())

    source = {"owner": "Marisol Vega"}
    tool_result = await context.callback("register_tool_sanitize_response_guardrail")("lookup", source)
    mark_result = await context.callback("register_mark_sanitize_guardrail")(
        {"kind": "mark", "name": "custom", "category": "custom"},
        {"data": source, "category_profile": None, "metadata": {}},
    )
    generic_llm_result = await context.callback("register_scope_sanitize_start_guardrail")(
        {"kind": "scope", "name": "manual", "category": "llm"},
        {"data": source, "category_profile": {"model_name": "Marisol Vega"}, "metadata": {}},
    )

    assert source == {"owner": "Marisol Vega"}
    assert tool_result == {"owner": "[REDACTED]"}
    assert mark_result["data"] == {"owner": "[REDACTED]"}
    assert generic_llm_result["data"] == {"owner": "[REDACTED]"}
    assert generic_llm_result["category_profile"] == {"model_name": "[REDACTED]"}
    await plugin.close()


async def test_llm_request_uses_active_codec_without_mutating_provider_request(monkeypatch) -> None:
    class Detector:
        def detect(self, texts: list[str]) -> list[list[RedactionSpan]]:
            return [
                [RedactionSpan(start, start + 12, "person")] if (start := text.find("Marisol Vega")) >= 0 else []
                for text in texts
            ]

    monkeypatch.setattr(worker, "AnonymizerBackend", MagicMock(return_value=Detector()))
    monkeypatch.setattr(worker, "require_api_keys", MagicMock())
    plugin = worker.NemoAnonymizerWorker()
    context = Context()
    await plugin.register(cast(PluginContext, context), valid_config())

    request = {"headers": {}, "content": {"messages": [{"role": "user", "content": "Marisol Vega"}]}}
    codec = SimpleNamespace(
        decode=AsyncMock(return_value={"messages": [{"role": "user", "content": "Marisol Vega"}]}),
        encode=AsyncMock(side_effect=lambda _annotated, original: original),
    )
    codec_context = SimpleNamespace(resolve_codec=lambda: codec)
    result = await context.callback("register_llm_sanitize_request_guardrail")(request, codec_context)

    assert request["content"]["messages"][0]["content"] == "Marisol Vega"
    assert result["content"]["messages"][0]["content"] == "[REDACTED]"
    encoded_annotation, encoded_original = codec.encode.await_args.args
    assert encoded_annotation["messages"][0]["content"] == "[REDACTED]"
    assert encoded_original["content"]["messages"][0]["content"] == "[REDACTED]"
    await plugin.close()


async def test_scope_sanitizer_covers_all_mutable_fields(monkeypatch) -> None:
    sanitizer = install_runtime_fakes(monkeypatch)
    context = Context()
    plugin = worker.NemoAnonymizerWorker()
    await plugin.register(cast(PluginContext, context), valid_config())
    fields = {
        "data": {"already": "handled"},
        "category_profile": {"annotated_request": {"already": "handled"}},
        "metadata": {"owner": "Marisol Vega"},
    }

    result = await context.callback("register_scope_sanitize_start_guardrail")(
        {"kind": "scope", "category": "llm"}, fields
    )

    assert sanitizer.sanitize.await_args.args == (fields,)
    assert result == fields

    await context.callback("register_scope_sanitize_end_guardrail")({"kind": "scope", "category": "tool"}, fields)
    assert sanitizer.sanitize.await_args.args == (fields,)
    await plugin.close()


async def test_llm_response_validates_the_sanitized_provider_copy(monkeypatch) -> None:
    sanitizer = install_runtime_fakes(monkeypatch)
    sanitizer.sanitize.side_effect = None
    sanitizer.sanitize.return_value = {"content": "[REDACTED]"}
    context = Context()
    plugin = worker.NemoAnonymizerWorker()
    await plugin.register(cast(PluginContext, context), valid_config())
    codec = SimpleNamespace(decode=AsyncMock(return_value={"message": "[REDACTED]"}))

    result = await context.callback("register_llm_sanitize_response_guardrail")(
        {"content": "Marisol Vega"},
        SimpleNamespace(resolve_codec=lambda: codec),
    )

    assert result == {"content": "[REDACTED]"}
    codec.decode.assert_awaited_once_with(result)
    await plugin.close()


async def test_stream_chunks_are_omitted_without_anonymizer_call(monkeypatch) -> None:
    sanitizer = install_runtime_fakes(monkeypatch)
    context = Context()
    plugin = worker.NemoAnonymizerWorker()
    await plugin.register(cast(PluginContext, context), valid_config())

    result = await context.callback("register_mark_sanitize_guardrail")(
        {"kind": "mark", "name": "llm.chunk", "category": "llm"},
        {"data": {"delta": "Marisol Vega"}, "category_profile": {}, "metadata": {}},
    )

    assert result == {
        "data": None,
        "category_profile": None,
        "metadata": {"nemo_anonymizer.coverage": "stream_chunk_payload_omitted"},
    }
    sanitizer.sanitize.assert_not_awaited()
    await plugin.close()


async def test_failed_registration_restores_secrets(monkeypatch) -> None:
    monkeypatch.setattr(worker, "require_api_keys", MagicMock(side_effect=ValueError("missing credential")))
    plugin = worker.NemoAnonymizerWorker()

    with pytest.raises(ValueError, match="missing credential"):
        await plugin.register(cast(PluginContext, Context()), valid_config())

    assert "TEST_EVALUATOR_KEY" not in os.environ


async def test_worker_errors_never_echo_source_text(monkeypatch) -> None:
    class FailingDetector:
        def detect(self, _texts: list[str]) -> list[list[RedactionSpan]]:
            raise RuntimeError("provider rejected Marisol Vega")

    monkeypatch.setattr(worker, "AnonymizerBackend", MagicMock(return_value=FailingDetector()))
    monkeypatch.setattr(worker, "require_api_keys", MagicMock())
    plugin = worker.NemoAnonymizerWorker()
    context = Context()
    await plugin.register(cast(PluginContext, context), valid_config())

    with pytest.raises(RuntimeError) as error:
        await context.callback("register_tool_sanitize_request_guardrail")("lookup", {"owner": "Marisol Vega"})

    assert str(error.value) == "nemo_anonymizer.tool_sanitization_failed"
    await plugin.close()


async def test_main_closes_worker_after_host_shutdown(monkeypatch) -> None:
    plugin = SimpleNamespace(close=AsyncMock())
    serve = AsyncMock()
    monkeypatch.setattr(worker, "NemoAnonymizerWorker", MagicMock(return_value=plugin))
    monkeypatch.setattr(worker, "serve_plugin", serve)
    monkeypatch.setattr(worker, "sys", SimpleNamespace(platform="linux"))

    await worker.main()

    serve.assert_awaited_once_with(plugin)
    plugin.close.assert_awaited_once_with()
