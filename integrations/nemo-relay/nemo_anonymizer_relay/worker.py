# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo Anonymizer observability-sanitizer worker for NeMo Relay."""

from __future__ import annotations

import asyncio
import os
import re
import signal
import sys
from copy import deepcopy
from typing import Any, cast

from nemo_relay_plugin import (
    ConfigDiagnostic,
    DiagnosticLevel,
    EventSanitizeFields,
    Json,
    PluginContext,
    WorkerPlugin,
    serve_plugin,
)

from nemo_anonymizer_relay.backend import AnonymizerBackend, BackendConfig, require_api_keys
from nemo_anonymizer_relay.sanitizer import ObservationSanitizer

PLUGIN_ID = "nvidia.nemo_anonymizer"
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_MISSING = object()

DEFAULT_CONFIG: dict[str, Json] = {
    "version": 1,
    "priority": 100,
    "max_text_leaves": 8192,
    "max_text_bytes": 4 * 1024 * 1024,
    "cache_entries": 16384,
    "detector_endpoint": "http://127.0.0.1:8001/v1",
    "detector_model": "fastino/gliner2-privacy-filter-PII-multi",
    "detector_api_key_env": "EMPTY",
    "evaluator_endpoint": "https://integrate.api.nvidia.com/v1",
    "evaluator_model": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "evaluator_api_key_env": "NVIDIA_API_KEY",
    "threshold": 0.3,
    "provider_timeout_seconds": 300,
    "data_summary": "Copied AI-agent observability containing prompts, model responses, and tool traffic.",
    "secret_env": {},
}


class _SecretEnvironment:
    """Temporarily install validated secrets for this worker activation."""

    def __init__(self) -> None:
        self._previous: dict[str, str | object] | None = None

    def install(self, values: dict[str, str]) -> None:
        if self._previous is not None:
            raise RuntimeError("secret environment is already installed")
        previous: dict[str, str | object] = {}
        try:
            for name, value in values.items():
                previous[name] = os.environ.get(name, _MISSING)
                os.environ[name] = value
        except BaseException:
            self._restore(previous)
            raise RuntimeError("secret environment could not be installed") from None
        self._previous = previous

    @staticmethod
    def _restore(previous: dict[str, str | object]) -> None:
        for name, value in previous.items():
            if value is _MISSING:
                os.environ.pop(name, None)
            else:
                os.environ[name] = cast(str, value)

    def restore(self) -> None:
        previous = self._previous
        self._previous = None
        if previous is not None:
            self._restore(previous)


class NemoAnonymizerWorker(WorkerPlugin):
    """Return sanitized observability copies to Relay for normal fan-out."""

    plugin_id = PLUGIN_ID
    allows_multiple_components = False

    def __init__(self) -> None:
        self._sanitizer: ObservationSanitizer | None = None
        self._secret_environment = _SecretEnvironment()

    def validate(self, config: Json) -> list[ConfigDiagnostic | dict[str, Any]]:
        try:
            normalized_config(config)
        except (TypeError, ValueError) as error:
            return [
                ConfigDiagnostic(
                    level=DiagnosticLevel.ERROR,
                    code="nemo_anonymizer.invalid_config",
                    message=str(error),
                    component=PLUGIN_ID,
                )
            ]
        return []

    async def register(self, ctx: PluginContext, config: Json) -> None:
        settings = normalized_config(config)
        if self._sanitizer is not None:
            raise RuntimeError("NeMo Anonymizer worker is already registered")

        try:
            self._secret_environment.install(cast(dict[str, str], settings["secret_env"]))
            backend_config = BackendConfig(
                detector_endpoint=cast(str, settings["detector_endpoint"]),
                detector_model=cast(str, settings["detector_model"]),
                detector_api_key_env=cast(str, settings["detector_api_key_env"]),
                evaluator_endpoint=cast(str, settings["evaluator_endpoint"]),
                evaluator_model=cast(str, settings["evaluator_model"]),
                evaluator_api_key_env=cast(str, settings["evaluator_api_key_env"]),
                threshold=cast(float, settings["threshold"]),
                timeout_seconds=cast(int, settings["provider_timeout_seconds"]),
                data_summary=cast(str | None, settings["data_summary"]),
            )
            require_api_keys(backend_config)
            sanitizer = ObservationSanitizer(
                AnonymizerBackend(backend_config),
                max_leaves=cast(int, settings["max_text_leaves"]),
                max_bytes=cast(int, settings["max_text_bytes"]),
                cache_entries=cast(int, settings["cache_entries"]),
            )
            priority = cast(int, settings["priority"])

            async def sanitize_event(event: dict[str, Any], fields: EventSanitizeFields) -> EventSanitizeFields:
                try:
                    return await _sanitize_event_fields(sanitizer, event, fields)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    raise RuntimeError("nemo_anonymizer.event_sanitization_failed") from None

            async def sanitize_tool(_name: str, value: Json) -> Json:
                return await _sanitize_value(sanitizer, value, failure_code="tool_sanitization_failed")

            async def sanitize_llm_request(request: dict[str, Any], context: Any) -> dict[str, Any]:
                try:
                    codec = context.resolve_codec()
                    if codec is None:
                        sanitized = await sanitizer.sanitize(request, preserve_protocol_values=True)
                    else:
                        annotated = await codec.decode(request)
                        values = await sanitizer.sanitize(
                            {"annotation": annotated, "request": request},
                            preserve_protocol_values=True,
                        )
                        if not isinstance(values, dict):
                            raise TypeError
                        sanitized = await codec.encode(values["annotation"], values["request"])
                    if not isinstance(sanitized, dict):
                        raise TypeError
                    return sanitized
                except asyncio.CancelledError:
                    raise
                except Exception:
                    raise RuntimeError("nemo_anonymizer.llm_request_sanitization_failed") from None

            async def sanitize_llm_response(response: Json, context: Any) -> Json:
                try:
                    codec = context.resolve_codec()
                    sanitized = await sanitizer.sanitize(response, preserve_protocol_values=True)
                    if codec is not None:
                        # The response proxy cannot encode. Decode the returned
                        # copy so a rewrite that invalidates the active provider
                        # shape fails closed before subscriber fan-out.
                        await codec.decode(sanitized)
                    return sanitized
                except asyncio.CancelledError:
                    raise
                except Exception:
                    raise RuntimeError("nemo_anonymizer.llm_response_sanitization_failed") from None

            ctx.register_mark_sanitize_guardrail("mark", sanitize_event, priority=priority)
            ctx.register_scope_sanitize_start_guardrail("scope_start", sanitize_event, priority=priority)
            ctx.register_scope_sanitize_end_guardrail("scope_end", sanitize_event, priority=priority)
            ctx.register_tool_sanitize_request_guardrail("tool_input", sanitize_tool, priority=priority)
            ctx.register_tool_sanitize_response_guardrail("tool_output", sanitize_tool, priority=priority)
            ctx.register_llm_sanitize_request_guardrail("llm_input", sanitize_llm_request, priority=priority)
            ctx.register_llm_sanitize_response_guardrail("llm_output", sanitize_llm_response, priority=priority)
            self._sanitizer = sanitizer
        except BaseException:
            self._secret_environment.restore()
            raise

    async def close(self) -> None:
        sanitizer = self._sanitizer
        self._sanitizer = None
        failed = False
        if sanitizer is not None:
            try:
                sanitizer.close()
            except Exception:
                failed = True
        try:
            self._secret_environment.restore()
        except Exception:
            failed = True
        if failed:
            raise RuntimeError("NeMo Anonymizer worker resources could not be closed")


def normalized_config(raw: Json) -> dict[str, Json]:
    if not isinstance(raw, dict):
        raise TypeError("configuration must be an object")
    unknown = set(raw) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"unknown configuration field(s): {', '.join(sorted(unknown))}")
    config = deepcopy(DEFAULT_CONFIG)
    config.update(raw)
    if config["version"] != 1 or isinstance(config["version"], bool):
        raise ValueError("version must be 1")

    priority = config["priority"]
    if not isinstance(priority, int) or isinstance(priority, bool) or not -(2**31) <= priority < 2**31:
        raise TypeError("priority must be a signed 32-bit integer")
    for field in ("max_text_leaves", "max_text_bytes", "cache_entries"):
        _positive_integer(config, field)
    _positive_integer(config, "provider_timeout_seconds", maximum=3600)
    for field in ("detector_endpoint", "detector_model", "evaluator_endpoint", "evaluator_model"):
        _nonempty_string(config, field)
    for field in ("detector_api_key_env", "evaluator_api_key_env"):
        value = _nonempty_string(config, field)
        if value != "EMPTY" and _ENV_NAME.fullmatch(value) is None:
            raise ValueError(f"{field} must be EMPTY or a valid environment variable name")
    threshold = config["threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be a number between 0 and 1")
    config["threshold"] = float(threshold)
    summary = config["data_summary"]
    if summary is not None and (not isinstance(summary, str) or len(summary) > 16_384):
        raise TypeError("data_summary must be null or a string no longer than 16384 characters")
    _validate_secret_env(config["secret_env"])
    return config


async def _sanitize_event_fields(
    sanitizer: ObservationSanitizer,
    event: dict[str, Any],
    fields: EventSanitizeFields,
) -> EventSanitizeFields:
    """Sanitize every mutable field on the copied event."""

    if event.get("kind") == "mark" and event.get("name") == "llm.chunk":
        return {
            "data": None,
            "category_profile": None,
            "metadata": {"nemo_anonymizer.coverage": "stream_chunk_payload_omitted"},
        }

    # Typed tool and LLM sanitizers protect Relay-managed lifecycle payloads,
    # but callers may also emit generic scopes with those categories. Relay
    # does not expose provenance here, so inspect every mutable field again.
    # This also covers compatibility fallbacks and future profile fields.
    sanitized = await sanitizer.sanitize(dict(fields), preserve_protocol_values=True)
    if not isinstance(sanitized, dict):
        raise RuntimeError("Anonymizer returned invalid event sanitizer fields")
    result = dict(fields)
    result.update(sanitized)
    return cast(EventSanitizeFields, result)


async def _sanitize_value(
    sanitizer: ObservationSanitizer,
    value: Json,
    *,
    failure_code: str,
    preserve_protocol_values: bool = False,
) -> Json:
    try:
        return await sanitizer.sanitize(value, preserve_protocol_values=preserve_protocol_values)
    except asyncio.CancelledError:
        raise
    except Exception:
        raise RuntimeError(f"nemo_anonymizer.{failure_code}") from None


def _positive_integer(config: dict[str, Json], field: str, *, maximum: int | None = None) -> None:
    value = config[field]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TypeError(f"{field} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field} must not exceed {maximum}")


def _nonempty_string(config: dict[str, Json], field: str) -> str:
    value = config[field]
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{field} must be a non-empty string")
    return value


def _validate_secret_env(value: Json) -> None:
    if not isinstance(value, dict) or len(value) > 32:
        raise TypeError("secret_env must be an object with at most 32 entries")
    for name, secret in value.items():
        if not isinstance(name, str) or _ENV_NAME.fullmatch(name) is None:
            raise ValueError("secret_env contains an invalid environment variable name")
        if not isinstance(secret, str) or not secret or len(secret) > 65_536 or "\x00" in secret:
            raise ValueError("secret_env values must be non-empty strings within the supported size limit")


async def main() -> None:
    """Serve the worker until Relay requests shutdown."""

    plugin = NemoAnonymizerWorker()
    try:
        if sys.platform != "win32":
            await serve_plugin(plugin)
            return
        previous_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            await serve_plugin(plugin)
        finally:
            signal.signal(signal.SIGINT, previous_handler)
    finally:
        await plugin.close()


if __name__ == "__main__":
    asyncio.run(main())
