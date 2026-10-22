# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from pathlib import Path
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


class RedactingDetector:
    def detect(self, texts: list[str]) -> list[list[RedactionSpan]]:
        return [
            [RedactionSpan(start, start + 12, "person")] if (start := text.find("Marisol Vega")) >= 0 else []
            for text in texts
        ]


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
    with pytest.raises(ValueError, match="must not exceed max_text_bytes"):
        worker.normalized_config({"max_text_bytes": 10, "max_text_leaf_bytes": 11})
    with pytest.raises(ValueError, match="must not exceed 29"):
        worker.normalized_config({"provider_timeout_seconds": 30})
    with pytest.raises(ValueError, match="threshold"):
        worker.normalized_config({"threshold": 1.1})
    with pytest.raises(ValueError, match="secret_env"):
        worker.normalized_config({"secret_env": {"BAD-NAME": "value"}})
    with pytest.raises(ValueError, match="configured provider credential"):
        worker.normalized_config({"secret_env": {"UNUSED_KEY": "value"}})
    with pytest.raises(ValueError, match="must not exceed 65536"):
        worker.normalized_config({"max_text_leaf_bytes": 65_537})


def test_schema_defaults_match_runtime_defaults() -> None:
    schema_path = Path(__file__).resolve().parents[1] / "config.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert {name: field["default"] for name, field in schema["properties"].items()} == worker.DEFAULT_CONFIG
    for field in ("detector_endpoint", "detector_model", "evaluator_endpoint", "evaluator_model"):
        assert re.fullmatch(schema["properties"][field]["pattern"], "   ") is None
    for field in ("detector_api_key_env", "evaluator_api_key_env"):
        pattern = schema["properties"][field]["pattern"]
        assert re.fullmatch(pattern, "NVIDIA_API_KEY") is not None
        assert re.fullmatch(pattern, "BAD-NAME") is None


async def test_worker_registers_event_sanitizer_surfaces(monkeypatch: pytest.MonkeyPatch) -> None:
    sanitizer = install_runtime_fakes(monkeypatch)
    context = Context()
    plugin = worker.NemoAnonymizerWorker()

    await plugin.register(cast(PluginContext, context), valid_config())

    assert set(context.registrations) == set(SURFACES)
    assert {registration[2] for registration in context.registrations.values()} == {100}

    await plugin.close()
    sanitizer.close.assert_called_once_with()


async def test_secret_environment_is_worker_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    install_runtime_fakes(monkeypatch)
    monkeypatch.setenv("TEST_EVALUATOR_KEY", "prior-value")
    plugin = worker.NemoAnonymizerWorker()

    await plugin.register(cast(PluginContext, Context()), valid_config())
    assert os.environ["TEST_EVALUATOR_KEY"] == "private-value"
    await plugin.close()
    assert os.environ["TEST_EVALUATOR_KEY"] == "prior-value"


async def test_marks_and_scopes_return_redacted_copies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker, "AnonymizerBackend", MagicMock(return_value=RedactingDetector()))
    monkeypatch.setattr(worker, "require_api_keys", MagicMock())
    plugin = worker.NemoAnonymizerWorker()
    context = Context()
    await plugin.register(cast(PluginContext, context), valid_config())

    source = {"owner": "Marisol Vega"}
    mark_result = await context.callback("register_mark_sanitize_guardrail")(
        {"kind": "mark", "name": "custom", "category": "custom"},
        {"data": source, "category_profile": None, "metadata": {}},
    )
    custom_scope_result = await context.callback("register_scope_sanitize_start_guardrail")(
        {"kind": "scope", "name": "manual", "category": "custom"},
        {"data": source, "category_profile": None, "metadata": {}},
    )
    llm_scope_result = await context.callback("register_scope_sanitize_start_guardrail")(
        {"kind": "scope", "name": "generic", "category": "llm", "scope_category": "start"},
        {
            "data": {"content": {"input": [{"role": "user", "content": "Marisol Vega"}]}},
            "category_profile": {
                "model_name": "Marisol Vega",
                "annotated_request": {"messages": [{"role": "user", "content": "Marisol Vega"}]},
            },
            "metadata": source,
        },
    )
    tool_scope_result = await context.callback("register_scope_sanitize_end_guardrail")(
        {"kind": "scope", "name": "generic", "category": "tool", "scope_category": "end"},
        {
            "data": source,
            "category_profile": {
                "tool_call_id": "Marisol Vega",
                "tool_result_annotation": {"owner": "Marisol Vega"},
            },
            "metadata": source,
        },
    )

    assert source == {"owner": "Marisol Vega"}
    assert mark_result["data"] == {"owner": "[REDACTED]"}
    assert custom_scope_result["data"] == {"owner": "[REDACTED]"}
    assert llm_scope_result == {
        "data": {"content": {"input": [{"role": "user", "content": "[REDACTED]"}]}},
        "category_profile": {
            "model_name": "[REDACTED]",
            "annotated_request": {"messages": [{"role": "user", "content": "[REDACTED]"}]},
        },
        "metadata": {"owner": "[REDACTED]"},
    }
    assert tool_scope_result == {
        "data": {"owner": "[REDACTED]"},
        "category_profile": {
            "tool_call_id": "[REDACTED]",
            "tool_result_annotation": {"owner": "[REDACTED]"},
        },
        "metadata": {"owner": "[REDACTED]"},
    }
    await plugin.close()


async def test_stream_chunks_are_omitted_without_anonymizer_call(monkeypatch: pytest.MonkeyPatch) -> None:
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


async def test_failed_registration_restores_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEST_EVALUATOR_KEY", raising=False)
    monkeypatch.setattr(worker, "require_api_keys", MagicMock(side_effect=ValueError("missing credential")))
    plugin = worker.NemoAnonymizerWorker()

    with pytest.raises(ValueError, match="missing credential"):
        await plugin.register(cast(PluginContext, Context()), valid_config())

    assert "TEST_EVALUATOR_KEY" not in os.environ


async def test_failed_registration_closes_created_sanitizer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEST_EVALUATOR_KEY", raising=False)
    sanitizer = install_runtime_fakes(monkeypatch)

    class FailingContext(Context):
        def __getattr__(self, surface: str) -> Any:
            if surface == "register_scope_sanitize_start_guardrail":

                def fail(*_args: Any, **_kwargs: Any) -> None:
                    raise RuntimeError("registration failed")

                return fail
            return super().__getattr__(surface)

    plugin = worker.NemoAnonymizerWorker()
    with pytest.raises(RuntimeError, match="registration failed"):
        await plugin.register(cast(PluginContext, FailingContext()), valid_config())

    sanitizer.close.assert_called_once_with()
    assert "TEST_EVALUATOR_KEY" not in os.environ


async def test_worker_errors_never_echo_source_text(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingDetector:
        def detect(self, _texts: list[str]) -> list[list[RedactionSpan]]:
            raise RuntimeError("provider rejected Marisol Vega")

    monkeypatch.setattr(worker, "AnonymizerBackend", MagicMock(return_value=FailingDetector()))
    monkeypatch.setattr(worker, "require_api_keys", MagicMock())
    plugin = worker.NemoAnonymizerWorker()
    context = Context()
    await plugin.register(cast(PluginContext, context), valid_config())

    with pytest.raises(RuntimeError) as error:
        await context.callback("register_scope_sanitize_start_guardrail")(
            {"kind": "scope", "name": "lookup", "category": "tool"},
            {"data": {"owner": "Marisol Vega"}, "category_profile": None, "metadata": None},
        )

    assert str(error.value) == "nemo_anonymizer.event_sanitization_failed"
    await plugin.close()


async def test_main_closes_worker_after_host_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = SimpleNamespace(close=AsyncMock())
    serve = AsyncMock()
    monkeypatch.setattr(worker, "NemoAnonymizerWorker", MagicMock(return_value=plugin))
    monkeypatch.setattr(worker, "serve_plugin", serve)
    monkeypatch.setattr(worker, "sys", SimpleNamespace(platform="linux"))

    await worker.main()

    serve.assert_awaited_once_with(plugin)
    plugin.close.assert_awaited_once_with()
