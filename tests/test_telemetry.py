# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from data_designer.config.models import ModelProvider

from anonymizer.telemetry import (
    DEFAULT_ENDPOINT,
    NOT_APPLICABLE,
    AnonymizerEvent,
    DeploymentTypeEnum,
    ModelHostEnum,
    NemoSourceEnum,
    QueuedEvent,
    TaskEnum,
    TaskStatusEnum,
    TelemetryHandler,
    _deployment_type,
    _telemetry_enabled,
    _telemetry_endpoint,
    avg_tokens_per_record,
    build_payload,
    classify_model_host,
    collect_model_hosts,
    sort_join_aliases,
)

# =============================================================================
# avg_tokens_per_record
# =============================================================================


class TestAvgTokensPerRecord:
    def test_empty_input_returns_negative_one(self) -> None:
        assert avg_tokens_per_record([]) == -1

    def test_single_string_returns_token_count(self) -> None:
        # tiktoken cl100k_base: "hello world" tokenizes to 2 tokens
        result = avg_tokens_per_record(["hello world"])
        assert result >= 1

    def test_averages_across_records(self) -> None:
        short = "hi"
        long_ = "the quick brown fox jumps over the lazy dog"
        avg = avg_tokens_per_record([short, long_])
        only_short = avg_tokens_per_record([short])
        only_long = avg_tokens_per_record([long_])
        assert only_short <= avg <= only_long

    def test_pandas_series_input(self) -> None:
        import pandas as pd

        s = pd.Series(["alpha", "beta gamma delta"])
        result = avg_tokens_per_record(s)
        assert result >= 1


# =============================================================================
# classify_model_host / collect_model_hosts
# =============================================================================


def _provider(endpoint: str) -> ModelProvider:
    return ModelProvider(name="test", endpoint=endpoint)


class TestClassifyModelHost:
    @pytest.mark.parametrize(
        "endpoint,expected",
        [
            ("https://build.nvidia.com/v1/chat/completions", ModelHostEnum.NVIDIA_BUILD),
            ("https://integrate.api.nvidia.com/v1", ModelHostEnum.NVIDIA_BUILD),
            ("https://Build.NVIDIA.com/v1", ModelHostEnum.NVIDIA_BUILD),
            ("https://inference-api.nvidia.com/v1", ModelHostEnum.NVIDIA_INTERNAL),
            ("https://openrouter.ai/api/v1", ModelHostEnum.OPENROUTER),
            ("http://localhost:8000/v1", ModelHostEnum.LOCAL),
            ("http://127.0.0.1:11434/v1", ModelHostEnum.LOCAL),
            ("http://0.0.0.0/v1", ModelHostEnum.LOCAL),
            ("https://gliner-qaqtckhiy.brevlab.com/v1", ModelHostEnum.OTHER),
            ("https://some-random-host.example/v1", ModelHostEnum.OTHER),
        ],
    )
    def test_known_hosts(self, endpoint: str, expected: ModelHostEnum) -> None:
        assert classify_model_host(_provider(endpoint)) == expected

    def test_none_provider(self) -> None:
        assert classify_model_host(None) == ModelHostEnum.OTHER


class TestCollectModelHosts:
    def test_empty_returns_other(self) -> None:
        assert collect_model_hosts([]) == ["other"]

    def test_single_unique_host(self) -> None:
        assert collect_model_hosts([ModelHostEnum.NVIDIA_BUILD]) == ["nvidia-build"]

    def test_duplicates_collapse(self) -> None:
        hosts = [ModelHostEnum.NVIDIA_BUILD, ModelHostEnum.NVIDIA_BUILD]
        assert collect_model_hosts(hosts) == ["nvidia-build"]

    def test_mixed_hosts_sorted(self) -> None:
        # Sorted alphabetically for canonical wire format
        hosts = [ModelHostEnum.OPENROUTER, ModelHostEnum.NVIDIA_BUILD]
        assert collect_model_hosts(hosts) == ["nvidia-build", "openrouter"]


# =============================================================================
# sort_join_aliases
# =============================================================================


class TestSortJoinAliases:
    def test_single_alias_is_unwrapped_string(self) -> None:
        assert sort_join_aliases(["alias-a"]) == "alias-a"

    def test_multiple_aliases_are_sorted_and_joined(self) -> None:
        # Pool member set is canonicalized — independent of input order.
        assert sort_join_aliases(["alias-b", "alias-a"]) == "alias-a,alias-b"
        assert sort_join_aliases(["alias-a", "alias-b"]) == "alias-a,alias-b"

    def test_empty_list_is_not_applicable(self) -> None:
        assert sort_join_aliases([]) == NOT_APPLICABLE

    def test_strips_whitespace(self) -> None:
        assert sort_join_aliases(["  alias-a  ", "alias-b"]) == "alias-a,alias-b"

    def test_filters_empty_strings(self) -> None:
        assert sort_join_aliases(["alias-a", "  ", ""]) == "alias-a"


# =============================================================================
# Env helpers
# =============================================================================


class TestEnvHelpers:
    def test_telemetry_enabled_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NEMO_TELEMETRY_ENABLED", raising=False)
        assert _telemetry_enabled() is True

    @pytest.mark.parametrize("value", ["false", "0", "no", "FALSE"])
    def test_telemetry_enabled_disable_values(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", value)
        assert _telemetry_enabled() is False

    def test_telemetry_endpoint_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NEMO_TELEMETRY_ENDPOINT", raising=False)
        assert _telemetry_endpoint() == DEFAULT_ENDPOINT

    def test_telemetry_endpoint_override_preserves_case(self, monkeypatch: pytest.MonkeyPatch) -> None:
        custom = "https://Events.example.COM/v1/Events?Token=ABC"
        monkeypatch.setenv("NEMO_TELEMETRY_ENDPOINT", custom)
        assert _telemetry_endpoint() == custom

    def test_deployment_type_default_is_sdk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NEMO_DEPLOYMENT_TYPE", raising=False)
        assert _deployment_type() == DeploymentTypeEnum.SDK

    def test_deployment_type_invalid_value_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEMO_DEPLOYMENT_TYPE", "definitely-not-real")
        # Must not raise — telemetry must never block runtime on a misconfigured env var.
        assert _deployment_type() == DeploymentTypeEnum.UNDEFINED


# =============================================================================
# AnonymizerEvent
# =============================================================================


def _minimal_event(**overrides) -> AnonymizerEvent:
    """Construct an event with the minimum required fields, allowing overrides."""
    defaults = dict(
        task=TaskEnum.BATCH,
        task_status=TaskStatusEnum.COMPLETED,
        transformation_type="redact",
        entity_detector_model="gliner-pii",
        entity_validator_model="some-validator",
        entity_augmenter_model="some-augmenter",
    )
    defaults.update(overrides)
    return AnonymizerEvent(**defaults)


class TestAnonymizerEvent:
    def test_minimal_event_populates_required(self) -> None:
        event = _minimal_event()
        assert event.nemo_source == NemoSourceEnum.ANONYMIZER
        assert event.deployment_type == DeploymentTypeEnum.SDK
        assert event.job_duration_sec == -1.0
        assert event.transformation_type == "redact"
        # model_hosts defaults to empty list when not provided
        assert event.model_hosts == []
        # Counts default to -1 sentinels
        assert event.num_input_records == -1
        assert event.num_success_records == -1
        assert event.num_failure_records == -1
        assert event.avg_tokens_per_record == -1
        # Rewrite-only model fields default to not_applicable
        assert event.rewriter_model == NOT_APPLICABLE

    def test_transformation_type_is_required(self) -> None:
        """transformationType has no default — every run has one."""
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            AnonymizerEvent(
                task=TaskEnum.BATCH,
                task_status=TaskStatusEnum.COMPLETED,
                entity_detector_model="x",
                entity_validator_model="x",
                entity_augmenter_model="x",
            )

    def test_detector_validator_augmenter_required(self) -> None:
        """The always-run detection roles have no defaults."""
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            AnonymizerEvent(
                task=TaskEnum.BATCH,
                task_status=TaskStatusEnum.COMPLETED,
                transformation_type="redact",
            )

    def test_model_dump_uses_camelcase_aliases(self) -> None:
        event = _minimal_event(
            task=TaskEnum.PREVIEW,
            task_status=TaskStatusEnum.ERROR,
            max_repair_iterations=3,
            strict_entity_protection=True,
            num_input_records=42,
            avg_tokens_per_record=128,
            model_hosts=["nvidia-build", "openrouter"],
        )
        dumped = event.model_dump(by_alias=True)
        assert dumped["nemoSource"] == "anonymizer"
        assert dumped["task"] == "preview"
        assert dumped["taskStatus"] == "error"
        assert dumped["maxRepairIterations"] == 3
        assert dumped["strictEntityProtection"] is True
        assert dumped["numInputRecords"] == 42
        assert dumped["avgTokensPerRecord"] == 128
        assert dumped["modelHosts"] == ["nvidia-build", "openrouter"]
        # dominantFailureStep is gone — must not appear
        assert "dominantFailureStep" not in dumped

    def test_payload_is_json_serializable(self) -> None:
        """Regression: enum-valued fields must encode as their string values."""
        event = _minimal_event()
        dumped = event.model_dump(by_alias=True, mode="json")
        json.dumps(dumped)  # should not raise

    def test_all_task_statuses(self) -> None:
        for status in TaskStatusEnum:
            event = _minimal_event(task_status=status)
            assert event.task_status == status


# =============================================================================
# build_payload
# =============================================================================


class TestBuildPayload:
    def _make_queued(self, *, task: TaskEnum = TaskEnum.BATCH) -> QueuedEvent:
        event = _minimal_event(task=task)
        return QueuedEvent(event=event, timestamp=datetime(2026, 5, 11, 12, 0, 0, tzinfo=timezone.utc))

    def test_envelope_shape(self) -> None:
        payload = build_payload([self._make_queued()], source_client_version="1.2.3", session_id="anonymizer-abc")
        assert payload["clientId"] == "184482118588404"
        assert payload["clientVer"] == "1.2.3"
        assert payload["sessionId"] == "anonymizer-abc"
        assert payload["eventSchemaVer"] == "1.7"
        assert len(payload["events"]) == 1
        assert payload["events"][0]["name"] == "anonymizer_event"
        assert payload["events"][0]["ts"] == "2026-05-11T12:00:00.000Z"

    def test_event_parameters_are_camelcase_strings(self) -> None:
        payload = build_payload([self._make_queued()], source_client_version="1.0")
        params = payload["events"][0]["parameters"]
        assert params["nemoSource"] == "anonymizer"
        assert params["task"] == "batch"
        assert params["taskStatus"] == "completed"
        assert params["deploymentType"] == "sdk"

    def test_payload_is_json_dumpable(self) -> None:
        payload = build_payload([self._make_queued()], source_client_version="1.0")
        json.dumps(payload)  # round-trip-safe

    def test_empty_events_raises(self) -> None:
        with pytest.raises(ValueError):
            build_payload([], source_client_version="1.0")


# =============================================================================
# TelemetryHandler — opt-out and queue behavior
# =============================================================================


class TestTelemetryHandlerOptOut:
    def test_enqueue_noop_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", "false")
        handler = TelemetryHandler()
        handler.enqueue(_minimal_event())
        assert handler._events == []

    def test_enqueue_noop_for_non_event(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", "true")
        handler = TelemetryHandler()
        handler.enqueue("not an event")  # type: ignore[arg-type]
        assert handler._events == []

    def test_enqueue_when_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", "true")
        handler = TelemetryHandler()
        event = _minimal_event()
        handler.enqueue(event)
        assert len(handler._events) == 1
        assert handler._events[0].event is event


class TestSessionPrefix:
    def test_no_prefix_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NEMO_SESSION_PREFIX", raising=False)
        handler = TelemetryHandler(session_id="abc")
        assert handler._session_id == "abc"

    def test_prefix_applied_when_env_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEMO_SESSION_PREFIX", "anonymizer-")
        handler = TelemetryHandler(session_id="abc")
        assert handler._session_id == "anonymizer-abc"


# =============================================================================
# TelemetryHandler — send semantics (retry / DLQ / split)
# =============================================================================


class TestSendSemantics:
    def _make(self) -> tuple[TelemetryHandler, QueuedEvent]:
        handler = TelemetryHandler(source_client_version="1.0", session_id="s1")
        event = _minimal_event()
        return handler, QueuedEvent(event=event, timestamp=datetime.now(timezone.utc))

    def test_successful_send_does_not_dlq(self) -> None:
        handler, q = self._make()
        mock_resp = MagicMock(status_code=200, is_success=True)
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp

        asyncio.run(handler._send_events_with_client(mock_client, [q]))
        mock_client.post.assert_awaited_once()
        assert handler._dlq == []

    def test_500_adds_to_dlq(self) -> None:
        handler, q = self._make()
        mock_resp = MagicMock(status_code=500, is_success=False)
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp

        asyncio.run(handler._send_events_with_client(mock_client, [q]))
        assert len(handler._dlq) == 1
        assert handler._dlq[0].retry_count == 1

    def test_400_does_not_dlq(self) -> None:
        """4xx (other than 413) means bad payload — no retry."""
        handler, q = self._make()
        mock_resp = MagicMock(status_code=400, is_success=False)
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp

        asyncio.run(handler._send_events_with_client(mock_client, [q]))
        assert handler._dlq == []

    def test_413_splits_and_retries(self) -> None:
        handler = TelemetryHandler(source_client_version="1.0", session_id="s1")
        events = [QueuedEvent(event=_minimal_event(), timestamp=datetime.now(timezone.utc)) for _ in range(2)]
        too_large = MagicMock(status_code=413, is_success=False)
        success = MagicMock(status_code=200, is_success=True)
        mock_client = AsyncMock()
        mock_client.post.side_effect = [too_large, success, success]

        asyncio.run(handler._send_events_with_client(mock_client, events))
        assert mock_client.post.await_count == 3  # 1 original + 2 splits

    def test_exceeds_max_retries_drops(self) -> None:
        handler, q = self._make()
        q.retry_count = handler._max_retries  # already at the cap
        mock_resp = MagicMock(status_code=500, is_success=False)
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp

        asyncio.run(handler._send_events_with_client(mock_client, [q]))
        assert handler._dlq == []  # event was dropped, not re-queued

    def test_client_setup_failure_routes_to_dlq(self) -> None:
        """If httpx.AsyncClient construction fails, events must land in DLQ rather than vanish."""
        handler, q = self._make()
        handler._events.append(q)
        with patch("httpx.AsyncClient", side_effect=RuntimeError("boom")):
            asyncio.run(handler._flush_events())
        assert len(handler._dlq) == 1


# =============================================================================
# TelemetryHandler — context manager
# =============================================================================


class TestContextManager:
    def test_context_manager_flushes_on_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", "true")
        sent: list[int] = []

        async def fake_send(events):
            sent.append(len(events))

        with patch.object(TelemetryHandler, "_send_events_with_client", new=AsyncMock(side_effect=fake_send)):
            with TelemetryHandler(source_client_version="1.0") as handler:
                handler.enqueue(_minimal_event())
        # On exit, flush was called; with httpx mocked at the client level it would still
        # exercise the path, so verify the queue drained.
        assert handler._events == []

    def test_context_manager_with_no_events_is_noop(self) -> None:
        with TelemetryHandler(source_client_version="1.0") as handler:
            pass
        assert handler._events == []
