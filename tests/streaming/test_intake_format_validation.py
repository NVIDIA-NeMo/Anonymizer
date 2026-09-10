# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from tests.streaming.intake_format_validation import (
    IntakeValidationError,
    build_local_otlp_request,
    otlp_string_attributes,
    otlp_topology,
    protect_atif,
    protect_chat_completion,
    protect_otlp_request,
)
from tests.streaming.structured_trace_prototype import build_synthetic_protection_flow

FIXTURES = Path(__file__).parents[1] / "fixtures" / "streaming"


def _flow(sensitive_entities: dict[str, str]):
    return build_synthetic_protection_flow(sensitive_entities)


@pytest.mark.parametrize(
    ("fixture_name", "sensitive_entities", "expected_version"),
    [
        (
            "intake_atif_v10.json",
            {"Alice": "person", "alice@example.test": "email", "Acme": "organization"},
            "ATIF-v1.0",
        ),
        (
            "intake_atif_v17.json",
            {"Bob": "person", "bob@example.test": "email", "Acme": "organization"},
            "ATIF-v1.7",
        ),
    ],
)
def test_atif_boundary_versions_round_trip_through_plan_a(
    fixture_name: str,
    sensitive_entities: dict[str, str],
    expected_version: str,
) -> None:
    source = (FIXTURES / fixture_name).read_bytes()
    original = json.loads(source)
    emitted: list[bytes] = []

    protected = protect_atif(source, flow=_flow(sensitive_entities), emit=emitted.append)

    assert protected
    assert emitted == [protected]
    result = json.loads(protected)
    assert result["schema_version"] == expected_version
    assert result["session_id"] == original["session_id"]
    assert result["trajectory_id"] == original["trajectory_id"]
    assert result["agent"] == original["agent"]
    assert [step["step_id"] for step in result["steps"]] == [1, 2]
    assert [step["source"] for step in result["steps"]] == ["user", "agent"]
    if expected_version == "ATIF-v1.7":
        assert result["steps"][1]["tool_calls"][0]["arguments"]["limit"] == 5
    rendered = json.dumps(result)
    assert all(value not in rendered for value in sensitive_entities)
    assert "[REDACTED]" in rendered


def test_chat_completion_preserves_extensions_and_protects_declared_content() -> None:
    source = (FIXTURES / "intake_chat_completion.json").read_bytes()
    original = json.loads(source)
    emitted: list[bytes] = []
    entities = {"Carol": "person", "carol@example.test": "email", "Acme": "organization"}

    protected = protect_chat_completion(source, flow=_flow(entities), emit=emitted.append)

    assert emitted == [protected]
    result = json.loads(protected)
    assert result["request"]["model"] == original["request"]["model"]
    assert result["request"]["provider_extension"] == original["request"]["provider_extension"]
    assert result["response"]["provider_response_id"] == original["response"]["provider_response_id"]
    assert result["response"]["created"] == original["response"]["created"]
    assert result["response"]["usage"] == original["response"]["usage"]
    assert result["session_id"] == original["session_id"]
    rendered = json.dumps(result)
    assert all(value not in rendered for value in entities)


def test_local_chain_llm_otlp_protobuf_round_trips_through_plan_a() -> None:
    source = build_local_otlp_request(FIXTURES / "intake_local_otlp_trace.json")
    original_topology = otlp_topology(source)
    original_attributes = otlp_string_attributes(source)
    emitted: list[bytes] = []
    entities = {"Dave": "person", "dave@example.test": "email", "Acme": "organization"}

    protected = protect_otlp_request(source, flow=_flow(entities), emit=emitted.append)

    assert emitted == [protected]
    assert otlp_topology(protected) == original_topology
    result_attributes = otlp_string_attributes(protected)
    assert result_attributes["0000000000000001"] == original_attributes["0000000000000001"]
    assert result_attributes["0000000000000002"]["gen_ai.agent.name"] == "sdk-validation-agent"
    assert result_attributes["0000000000000002"]["gen_ai.request.model"] == "gpt-validation"
    rendered = json.dumps(result_attributes)
    assert all(value not in rendered for value in entities)
    assert "[REDACTED]" in rendered


def test_invalid_otlp_span_withholds_complete_batch() -> None:
    source = build_local_otlp_request(FIXTURES / "intake_local_otlp_trace.json")
    request = ExportTraceServiceRequest.FromString(source)
    invalid = request.resource_spans[0].scope_spans[0].spans.add()
    invalid.span_id = bytes.fromhex("0000000000000003")
    invalid.name = "missing-trace-id"
    emitted: list[bytes] = []

    with pytest.raises(IntakeValidationError, match="OTLP batch rejected"):
        protect_otlp_request(request.SerializeToString(), flow=_flow({}), emit=emitted.append)

    assert emitted == []


def test_non_success_plan_a_outcome_withholds_json_item() -> None:
    source = (FIXTURES / "intake_atif_v10.json").read_bytes()
    flow = _flow({"Alice": "person"})
    flow.close()
    emitted: list[bytes] = []

    with pytest.raises(IntakeValidationError, match="protection failed"):
        protect_atif(source, flow=flow, emit=emitted.append)

    assert emitted == []


def test_atif_unknown_top_level_field_is_rejected() -> None:
    document: dict[str, Any] = json.loads((FIXTURES / "intake_atif_v10.json").read_bytes())
    document["undeclared_content"] = "Alice"

    with pytest.raises(IntakeValidationError, match="ATIF item rejected"):
        protect_atif(json.dumps(document).encode(), flow=_flow({"Alice": "person"}), emit=lambda _: None)


def test_chat_completion_requires_exactly_one_response_variant() -> None:
    document: dict[str, Any] = json.loads((FIXTURES / "intake_chat_completion.json").read_bytes())
    document["response"]["error"] = {"message": "Carol failed"}

    with pytest.raises(IntakeValidationError, match="chat completion rejected"):
        protect_chat_completion(json.dumps(document).encode(), flow=_flow({"Carol": "person"}), emit=lambda _: None)


def test_chat_completion_rejects_unreviewed_nested_provider_content() -> None:
    document: dict[str, Any] = json.loads((FIXTURES / "intake_chat_completion.json").read_bytes())
    document["response"]["provider_secret"] = "eve@example.test"

    with pytest.raises(IntakeValidationError, match="chat completion rejected"):
        protect_chat_completion(json.dumps(document).encode(), flow=_flow({}), emit=lambda _: None)


def test_chat_completion_requires_stable_creation_time() -> None:
    document: dict[str, Any] = json.loads((FIXTURES / "intake_chat_completion.json").read_bytes())
    del document["response"]["created"]

    with pytest.raises(IntakeValidationError, match="chat completion rejected"):
        protect_chat_completion(json.dumps(document).encode(), flow=_flow({}), emit=lambda _: None)


@pytest.mark.parametrize("created", [True, 0, 10**100])
def test_chat_completion_rejects_invalid_creation_time(created: object) -> None:
    document: dict[str, Any] = json.loads((FIXTURES / "intake_chat_completion.json").read_bytes())
    document["response"]["created"] = created

    with pytest.raises(IntakeValidationError, match="chat completion rejected"):
        protect_chat_completion(json.dumps(document).encode(), flow=_flow({}), emit=lambda _: None)


def test_chat_completion_rejects_future_creation_time() -> None:
    document: dict[str, Any] = json.loads((FIXTURES / "intake_chat_completion.json").read_bytes())
    document["response"]["created"] = int(time.time()) + 3_600

    with pytest.raises(IntakeValidationError, match="chat completion rejected"):
        protect_chat_completion(json.dumps(document).encode(), flow=_flow({}), emit=lambda _: None)


def test_atif_rejects_unreviewed_image_content_part() -> None:
    document: dict[str, Any] = json.loads((FIXTURES / "intake_atif_v17.json").read_bytes())
    document["steps"][0]["message"].append(
        {"type": "image", "source": {"media_type": "image/png", "path": "/private/alice.png"}}
    )

    with pytest.raises(IntakeValidationError, match="ATIF item rejected"):
        protect_atif(json.dumps(document).encode(), flow=_flow({}), emit=lambda _: None)


def test_otlp_rejects_unreviewed_resource_and_event_content() -> None:
    source = build_local_otlp_request(FIXTURES / "intake_local_otlp_trace.json")
    request = ExportTraceServiceRequest.FromString(source)
    resource_attribute = request.resource_spans[0].resource.attributes.add()
    resource_attribute.key = "deployment.secret"
    resource_attribute.value.string_value = "alice@example.test"

    with pytest.raises(IntakeValidationError, match="OTLP batch rejected"):
        protect_otlp_request(request.SerializeToString(), flow=_flow({}), emit=lambda _: None)

    request = ExportTraceServiceRequest.FromString(source)
    event = request.resource_spans[0].scope_spans[0].spans[1].events.add()
    event.name = "exception"
    event_attribute = event.attributes.add()
    event_attribute.key = "exception.message"
    event_attribute.value.string_value = "alice@example.test"

    with pytest.raises(IntakeValidationError, match="OTLP batch rejected"):
        protect_otlp_request(request.SerializeToString(), flow=_flow({}), emit=lambda _: None)


@pytest.mark.parametrize("invalid_value", ["", False])
def test_otlp_rejects_invalid_agent_name(invalid_value: str | bool) -> None:
    request = ExportTraceServiceRequest.FromString(build_local_otlp_request(FIXTURES / "intake_local_otlp_trace.json"))
    attributes = request.resource_spans[0].scope_spans[0].spans[1].attributes
    agent_name = next(attribute for attribute in attributes if attribute.key == "gen_ai.agent.name")
    if isinstance(invalid_value, str):
        agent_name.value.string_value = invalid_value
    else:
        agent_name.value.bool_value = invalid_value

    with pytest.raises(IntakeValidationError, match="OTLP batch rejected"):
        protect_otlp_request(request.SerializeToString(), flow=_flow({}), emit=lambda _: None)


def test_otlp_rejects_duplicate_attribute_keys() -> None:
    request = ExportTraceServiceRequest.FromString(build_local_otlp_request(FIXTURES / "intake_local_otlp_trace.json"))
    attributes = request.resource_spans[0].scope_spans[0].spans[1].attributes
    duplicate = attributes.add()
    duplicate.key = "input.value"
    duplicate.value.string_value = "duplicate"

    with pytest.raises(IntakeValidationError, match="OTLP batch rejected"):
        protect_otlp_request(request.SerializeToString(), flow=_flow({}), emit=lambda _: None)
