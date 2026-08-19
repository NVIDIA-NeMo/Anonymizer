# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in dogfood against an operator-owned NeMo Platform Intake service.

Set ``ANONYMIZER_INTAKE_DOGFOOD_BASE_URL`` to the service origin, for example
``http://127.0.0.1:8080``. The default profile protects each request and checks
its declared synthetic PII before crossing the HTTP boundary. Set
``ANONYMIZER_INTAKE_DOGFOOD_ALLOW_RAW=1`` only for isolated characterization
that intentionally sends raw synthetic fixtures. These tests never start,
configure, stop, or clean up Intake or its storage.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import uuid4

import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from anonymizer.config.anonymizer_config import AnonymizerConfig
from anonymizer.config.replace_strategies import Redact
from tests.streaming.intake_format_validation import (
    IntakeValidationError,
    build_local_otlp_request,
    protect_atif,
    protect_chat_completion,
    protect_otlp_request,
)
from tests.streaming.sandbox_session_export import export_codex_session_to_atif
from tests.streaming.structured_trace_prototype import build_synthetic_anonymizer

FIXTURES = Path(__file__).parents[1] / "fixtures" / "streaming"
_BASE_URL_ENV = "ANONYMIZER_INTAKE_DOGFOOD_BASE_URL"
_ALLOW_RAW_ENV = "ANONYMIZER_INTAKE_DOGFOOD_ALLOW_RAW"
_SANDBOX_RUN_DIR_ENV = "ANONYMIZER_SANDBOX_DOGFOOD_RUN_DIR"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def intake_base_url() -> str:
    base_url = os.environ.get(_BASE_URL_ENV)
    if base_url is None:
        pytest.skip(f"set {_BASE_URL_ENV} to run Intake dogfood")
    origin = base_url.rstrip("/")
    status, body = _request(origin, "/health/ready")
    if status != 200 or json.loads(body) != {"status": "ready"}:
        pytest.fail("configured Intake service is not ready")
    return f"{origin}/apis/intake/v2/workspaces/default"


@pytest.fixture(scope="module")
def dogfood_run_id() -> str:
    return uuid4().hex[:12]


def _flow(entities: Mapping[str, str]):
    anonymizer = build_synthetic_anonymizer(dict(entities))
    plan = anonymizer._compile_protection_plan(AnonymizerConfig(replace=Redact(), emit_telemetry=False))
    return anonymizer._open_protection_flow(plan)


def _dogfood_variants() -> tuple[str, ...]:
    if os.environ.get(_ALLOW_RAW_ENV) == "1":
        return ("raw", "protected")
    return ("protected",)


def _assert_safe_outbound(body: bytes, sensitive: tuple[str, ...]) -> None:
    assert all(value.encode() not in body for value in sensitive)
    assert b"[REDACTED_" in body


def _request(
    base_url: str,
    path: str,
    *,
    body: bytes | None = None,
    content_type: str | None = None,
) -> tuple[int, bytes]:
    headers = {"Content-Type": content_type} if content_type else {}
    method = "POST" if body is not None else "GET"
    request = Request(f"{base_url}{path}", data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=20) as response:
            return response.status, response.read()
    except HTTPError as error:
        pytest.fail(f"Intake {method} {path} returned HTTP {error.code}")


class _IntakeEmitter:
    def __init__(self, base_url: str, path: str, content_type: str) -> None:
        self._base_url = base_url
        self._path = path
        self._content_type = content_type
        self.calls = 0

    def __call__(self, body: bytes) -> None:
        self.calls += 1
        status, _ = _request(
            self._base_url,
            self._path,
            body=body,
            content_type=self._content_type,
        )
        assert status == 200


class _DogfoodDeliveryError(RuntimeError):
    """Sanitized test-only protected-payload delivery failure."""


def _deliver(
    base_url: str,
    path: str,
    *,
    body: bytes,
    content_type: str,
) -> tuple[int, bytes]:
    result: tuple[int, bytes] | None = None
    failed = False
    try:
        result = _request(base_url, path, body=body, content_type=content_type)
    except OSError:
        failed = True
    if failed:
        raise _DogfoodDeliveryError("Protected payload delivery failed") from None
    if result is None:  # pragma: no cover - defensive typing guard
        raise RuntimeError("Protected payload delivery returned no result")
    return result


def _deliver_and_forget(
    base_url: str,
    path: str,
    *,
    body: bytes,
    content_type: str,
) -> None:
    _deliver(base_url, path, body=body, content_type=content_type)


def _query_spans(base_url: str, filters: Mapping[str, str]) -> list[dict[str, object]]:
    query_values = {f"filter[{key}]": value for key, value in filters.items()}
    query_values.update({"page": "1", "page_size": "100"})
    query = urlencode(query_values)
    status, body = _request(base_url, f"/spans?{query}")
    if status != 200:
        pytest.fail("Intake span query did not return HTTP 200")
    return json.loads(body)["data"]


def _stored_spans(base_url: str, filters: Mapping[str, str]) -> list[dict[str, object]]:
    for _ in range(30):
        data = _query_spans(base_url, filters)
        if data:
            return data
        time.sleep(0.2)
    pytest.fail("Intake did not expose the ingested synthetic spans")


def _stable_spans(
    base_url: str,
    filters: Mapping[str, str],
    *,
    expected_count: int,
) -> list[dict[str, object]]:
    previous = ""
    stable_reads = 0
    for _ in range(30):
        data = _query_spans(base_url, filters)
        if len(data) > expected_count:
            pytest.fail("Intake exposed duplicate synthetic spans")
        snapshot = json.dumps(data, sort_keys=True)
        if len(data) == expected_count and snapshot == previous:
            stable_reads += 1
        else:
            stable_reads = 1 if len(data) == expected_count else 0
        if stable_reads == 3:
            return data
        previous = snapshot
        time.sleep(0.2)
    pytest.fail("Intake read model did not reach the expected stable state")


def _assert_visibility(
    base_url: str,
    session_id: str,
    sensitive: tuple[str, ...],
    *,
    expect_sensitive: bool,
) -> list[dict[str, object]]:
    stored = _stored_spans(base_url, {"session_id": session_id})
    rendered = json.dumps(stored)
    present = [value for value in sensitive if value in rendered]
    assert bool(present) is expect_sensitive
    if not expect_sensitive:
        assert "[REDACTED_" in rendered
    return stored


@pytest.mark.parametrize(
    ("fixture_name", "version", "entities"),
    [
        (
            "intake_atif_v10.json",
            "v10",
            {"Alice": "person", "alice@example.test": "email", "Acme": "organization"},
        ),
        (
            "intake_atif_v17.json",
            "v17",
            {"Bob": "person", "bob@example.test": "email", "Acme": "organization"},
        ),
    ],
)
def test_atif_round_trips_through_intake(
    intake_base_url: str,
    dogfood_run_id: str,
    fixture_name: str,
    version: str,
    entities: dict[str, str],
) -> None:
    original = json.loads((FIXTURES / fixture_name).read_bytes())
    for variant in _dogfood_variants():
        document = json.loads(json.dumps(original))
        session_id = f"sdk-dogfood-{dogfood_run_id}-atif-{version}-{variant}"
        document["session_id"] = session_id
        document["trajectory_id"] = f"trajectory-{dogfood_run_id}-{version}-{variant}"
        body = json.dumps(document, separators=(",", ":")).encode()
        if variant == "protected":
            emitted: list[bytes] = []
            body = protect_atif(body, flow=_flow(entities), emit=emitted.append)
            assert emitted == [body]
            _assert_safe_outbound(body, tuple(entities))

        status, _ = _request(
            intake_base_url,
            "/ingest/atif",
            body=body,
            content_type="application/json",
        )

        assert status == 201
        stored = _assert_visibility(
            intake_base_url,
            session_id,
            tuple(entities),
            expect_sensitive=variant == "raw",
        )
        assert len({span["trace_id"] for span in stored}) == 1
        assert any(span.get("parent_span_id") is not None for span in stored)


def test_chat_completion_round_trips_through_intake(
    intake_base_url: str,
    dogfood_run_id: str,
) -> None:
    original = json.loads((FIXTURES / "intake_chat_completion.json").read_bytes())
    entities = {"Carol": "person", "carol@example.test": "email", "Acme": "organization"}
    for variant in _dogfood_variants():
        document = json.loads(json.dumps(original))
        session_id = f"sdk-dogfood-{dogfood_run_id}-chat-{variant}"
        document["session_id"] = session_id
        document["trace_id"] = f"sdk-dogfood-{dogfood_run_id}-chat-{variant}"
        document["response"]["created"] = int(time.time())
        body = json.dumps(document, separators=(",", ":")).encode()
        if variant == "protected":
            emitted: list[bytes] = []
            body = protect_chat_completion(body, flow=_flow(entities), emit=emitted.append)
            assert emitted == [body]
            _assert_safe_outbound(body, tuple(entities))

        status, _ = _request(
            intake_base_url,
            "/ingest/chat-completions",
            body=body,
            content_type="application/json",
        )

        assert status == 201
        stored = _assert_visibility(
            intake_base_url,
            session_id,
            tuple(entities),
            expect_sensitive=variant == "raw",
        )
        assert len(stored) == 1


def _otlp_variant(
    source: bytes,
    *,
    variant: int,
    session_id: str,
    agent_name: str,
) -> bytes:
    message = ExportTraceServiceRequest.FromString(source)
    spans = message.resource_spans[0].scope_spans[0].spans
    trace_id = variant.to_bytes(16, "big")
    new_ids = {span.span_id: (variant * 16 + index).to_bytes(8, "big") for index, span in enumerate(spans, start=1)}
    for span in spans:
        parent = bytes(span.parent_span_id)
        old_id = bytes(span.span_id)
        span.trace_id = trace_id
        span.span_id = new_ids[old_id]
        if parent:
            span.parent_span_id = new_ids[parent]
        for attribute in span.attributes:
            if attribute.key == "gen_ai.conversation.id":
                attribute.value.string_value = session_id
            elif attribute.key == "gen_ai.agent.name":
                attribute.value.string_value = agent_name
    return message.SerializeToString(deterministic=True)


def test_otlp_round_trip_and_agent_name_filter(
    intake_base_url: str,
    dogfood_run_id: str,
) -> None:
    source = build_local_otlp_request(FIXTURES / "intake_local_otlp_trace.json")
    entities = {"Dave": "person", "dave@example.test": "email", "Acme": "organization"}
    protected_session = ""
    protected_agent = ""
    for index, variant in enumerate(_dogfood_variants(), start=1):
        session_id = f"sdk-dogfood-{dogfood_run_id}-otlp-{variant}"
        agent_name = f"sdk-validation-agent-{dogfood_run_id}-{variant}"
        body = _otlp_variant(
            source,
            variant=int(dogfood_run_id[:8], 16) + index,
            session_id=session_id,
            agent_name=agent_name,
        )
        if variant == "protected":
            emitted: list[bytes] = []
            body = protect_otlp_request(body, flow=_flow(entities), emit=emitted.append)
            assert emitted == [body]
            _assert_safe_outbound(body, tuple(entities))
            protected_session = session_id
            protected_agent = agent_name

        status, response_body = _request(
            intake_base_url,
            "/ingest/otlp/v1/traces",
            body=body,
            content_type="application/x-protobuf",
        )

        assert status == 200
        assert json.loads(response_body)["errors"] == []
        stored = _assert_visibility(
            intake_base_url,
            session_id,
            tuple(entities),
            expect_sensitive=variant == "raw",
        )
        assert len(stored) == 2
        assert sum(span.get("parent_span_id") is not None for span in stored) == 1

    filtered = _stored_spans(intake_base_url, {"agent_name": protected_agent})
    assert len(filtered) == 1
    assert filtered[0]["agent_name"] == protected_agent
    assert filtered[0]["session_id"] == protected_session


def test_invalid_otlp_exposes_adapter_intake_atomicity_mismatch(
    intake_base_url: str,
    dogfood_run_id: str,
) -> None:
    if os.environ.get(_ALLOW_RAW_ENV) != "1":
        pytest.skip(f"set {_ALLOW_RAW_ENV}=1 only for isolated raw characterization")
    source = build_local_otlp_request(FIXTURES / "intake_local_otlp_trace.json")
    session_id = f"sdk-dogfood-{dogfood_run_id}-otlp-partial"
    variant = int(dogfood_run_id[:8], 16) + 3
    request = ExportTraceServiceRequest.FromString(
        _otlp_variant(
            source,
            variant=variant,
            session_id=session_id,
            agent_name=f"sdk-validation-agent-{dogfood_run_id}-partial",
        )
    )
    invalid = request.resource_spans[0].scope_spans[0].spans.add()
    invalid.span_id = (variant * 16 + 3).to_bytes(8, "big")
    invalid.name = "missing-trace-id"
    body = request.SerializeToString(deterministic=True)
    emitted: list[bytes] = []

    with pytest.raises(IntakeValidationError, match="OTLP batch rejected"):
        protect_otlp_request(body, flow=_flow({}), emit=emitted.append)

    assert emitted == []
    status, response_body = _request(
        intake_base_url,
        "/ingest/otlp/v1/traces",
        body=body,
        content_type="application/x-protobuf",
    )
    stored = _stored_spans(intake_base_url, {"session_id": session_id})

    assert status == 200
    assert len(json.loads(response_body)["errors"]) == 1
    assert len(stored) == 2


def test_invalid_otlp_does_not_cross_intake_boundary(
    intake_base_url: str,
    dogfood_run_id: str,
) -> None:
    source = build_local_otlp_request(FIXTURES / "intake_local_otlp_trace.json")
    session_id = f"sdk-dogfood-{dogfood_run_id}-otlp-withheld"
    variant = int(dogfood_run_id[:8], 16) + 4
    request = ExportTraceServiceRequest.FromString(
        _otlp_variant(
            source,
            variant=variant,
            session_id=session_id,
            agent_name=f"sdk-validation-agent-{dogfood_run_id}-withheld",
        )
    )
    invalid = request.resource_spans[0].scope_spans[0].spans.add()
    invalid.span_id = (variant * 16 + 3).to_bytes(8, "big")
    invalid.name = "missing-trace-id"
    emitter = _IntakeEmitter(
        intake_base_url,
        "/ingest/otlp/v1/traces",
        "application/x-protobuf",
    )

    with pytest.raises(IntakeValidationError, match="OTLP batch rejected"):
        protect_otlp_request(
            request.SerializeToString(deterministic=True),
            flow=_flow({}),
            emit=emitter,
        )

    assert emitter.calls == 0
    assert _query_spans(intake_base_url, {"session_id": session_id}) == []


def test_protected_delivery_failure_is_sanitized_and_retryable(
    intake_base_url: str,
    dogfood_run_id: str,
) -> None:
    document = json.loads((FIXTURES / "intake_chat_completion.json").read_bytes())
    entities = {"Carol": "person", "carol@example.test": "email", "Acme": "organization"}
    session_id = f"sdk-dogfood-{dogfood_run_id}-delivery-unavailable"
    document["session_id"] = session_id
    document["trace_id"] = f"sdk-dogfood-{dogfood_run_id}-delivery-unavailable"
    document["response"]["created"] = int(time.time())
    emitted: list[bytes] = []
    protected = protect_chat_completion(
        json.dumps(document, separators=(",", ":")).encode(),
        flow=_flow(entities),
        emit=emitted.append,
    )
    _assert_safe_outbound(protected, tuple(entities))

    with pytest.raises(_DogfoodDeliveryError, match="Protected payload delivery failed") as exc_info:
        _deliver(
            "http://127.0.0.1:1/apis/intake/v2/workspaces/default",
            "/ingest/chat-completions",
            body=protected,
            content_type="application/json",
        )

    assert emitted == [protected]
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert _query_spans(intake_base_url, {"session_id": session_id}) == []

    status, _ = _deliver(
        intake_base_url,
        "/ingest/chat-completions",
        body=protected,
        content_type="application/json",
    )
    assert status == 201
    stored = _stable_spans(intake_base_url, {"session_id": session_id}, expected_count=1)
    assert "[REDACTED_" in json.dumps(stored)


def _assert_exact_retry_deduplicates(
    intake_base_url: str,
    *,
    session_id: str,
    path: str,
    content_type: str,
    body: bytes,
    expected_status: int,
    expected_count: int,
    expected_parent_count: int,
) -> None:
    _deliver_and_forget(intake_base_url, path, body=body, content_type=content_type)
    after_commit = _stable_spans(
        intake_base_url,
        {"session_id": session_id},
        expected_count=expected_count,
    )
    status, _ = _deliver(intake_base_url, path, body=body, content_type=content_type)
    assert status == expected_status
    after_retry = _stable_spans(
        intake_base_url,
        {"session_id": session_id},
        expected_count=expected_count,
    )

    assert len(after_retry) == len(after_commit) == expected_count
    assert {span["span_id"] for span in after_retry} == {span["span_id"] for span in after_commit}
    assert sum(span.get("parent_span_id") is not None for span in after_retry) == expected_parent_count


def test_atif_exact_retry_deduplicates(
    intake_base_url: str,
    dogfood_run_id: str,
) -> None:
    document = json.loads((FIXTURES / "intake_atif_v10.json").read_bytes())
    entities = {"Alice": "person", "alice@example.test": "email", "Acme": "organization"}
    session_id = f"sdk-dogfood-{dogfood_run_id}-retry-atif"
    document["session_id"] = session_id
    document["trajectory_id"] = f"trajectory-{dogfood_run_id}-retry-atif"
    body = protect_atif(
        json.dumps(document, separators=(",", ":")).encode(),
        flow=_flow(entities),
        emit=lambda _: None,
    )
    _assert_safe_outbound(body, tuple(entities))
    _assert_exact_retry_deduplicates(
        intake_base_url,
        session_id=session_id,
        path="/ingest/atif",
        content_type="application/json",
        body=body,
        expected_status=201,
        expected_count=4,
        expected_parent_count=3,
    )


def test_completed_sandbox_session_is_protected_before_intake(
    intake_base_url: str,
    dogfood_run_id: str,
) -> None:
    run_dir_value = os.environ.get(_SANDBOX_RUN_DIR_ENV)
    if run_dir_value is None:
        pytest.skip(f"set {_SANDBOX_RUN_DIR_ENV} to export a completed Sandbox Codex run")
    entities = {
        "Mira Testperson": "person",
        "mira.sandbox@example.test": "email",
        "555-0109": "phone_number",
        "Acme Validation Lab": "organization",
    }
    session_id = f"sdk-dogfood-{dogfood_run_id}-sandbox"
    source = export_codex_session_to_atif(Path(run_dir_value), session_id=session_id)
    assert all(value.encode() in source for value in entities)
    body = protect_atif(source, flow=_flow(entities), emit=lambda _: None)
    _assert_safe_outbound(body, tuple(entities))

    status, _ = _deliver(
        intake_base_url,
        "/ingest/atif",
        body=body,
        content_type="application/json",
    )

    assert status == 201
    stored = _assert_visibility(intake_base_url, session_id, tuple(entities), expect_sensitive=False)
    assert len(stored) >= 2
    assert any(span.get("parent_span_id") is not None for span in stored)


def test_chat_exact_retry_deduplicates(
    intake_base_url: str,
    dogfood_run_id: str,
) -> None:
    document = json.loads((FIXTURES / "intake_chat_completion.json").read_bytes())
    entities = {"Carol": "person", "carol@example.test": "email", "Acme": "organization"}
    session_id = f"sdk-dogfood-{dogfood_run_id}-retry-chat"
    document["session_id"] = session_id
    document["trace_id"] = f"sdk-dogfood-{dogfood_run_id}-retry-chat"
    document["response"]["created"] = int(time.time())
    body = protect_chat_completion(
        json.dumps(document, separators=(",", ":")).encode(),
        flow=_flow(entities),
        emit=lambda _: None,
    )
    _assert_safe_outbound(body, tuple(entities))
    _assert_exact_retry_deduplicates(
        intake_base_url,
        session_id=session_id,
        path="/ingest/chat-completions",
        content_type="application/json",
        body=body,
        expected_status=201,
        expected_count=1,
        expected_parent_count=0,
    )


def test_otlp_exact_retry_deduplicates(
    intake_base_url: str,
    dogfood_run_id: str,
) -> None:
    source = build_local_otlp_request(FIXTURES / "intake_local_otlp_trace.json")
    entities = {"Dave": "person", "dave@example.test": "email", "Acme": "organization"}
    session_id = f"sdk-dogfood-{dogfood_run_id}-retry-otlp"
    body = protect_otlp_request(
        _otlp_variant(
            source,
            variant=int(dogfood_run_id[:8], 16) + 5,
            session_id=session_id,
            agent_name=f"sdk-validation-agent-{dogfood_run_id}-retry",
        ),
        flow=_flow(entities),
        emit=lambda _: None,
    )
    _assert_safe_outbound(body, tuple(entities))
    _assert_exact_retry_deduplicates(
        intake_base_url,
        session_id=session_id,
        path="/ingest/otlp/v1/traces",
        content_type="application/x-protobuf",
        body=body,
        expected_status=200,
        expected_count=2,
        expected_parent_count=1,
    )
