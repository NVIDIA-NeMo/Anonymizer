# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test-only Intake format validation probes."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol, cast

import pandas as pd
from google.protobuf.message import DecodeError
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from anonymizer.engine.constants import COL_FINAL_ENTITIES, COL_TEXT
from anonymizer.interface._protection import (
    _ProtectionRecord,
    _ProtectionRunRecord,
    _ProtectionSucceeded,
    _RecordRef,
    _TextSegment,
)
from tests.streaming.structured_trace_prototype import (
    PROTECTED_TEXT_COLUMN,
    SEGMENT_KEY_COLUMN,
    CodecBounds,
    FieldRole,
    ProjectedItem,
    SourceFormat,
    StructuredItemError,
    TraceMapping,
    project_complete_item,
    reconstruct_complete_item,
)

Emitter = Callable[[bytes], None]
_JSON_BOUNDS = CodecBounds(
    max_bytes=1_048_576,
    max_depth=32,
    max_targets=128,
    max_scalars=512,
    max_scalar_bytes=65_536,
    max_events=2_048,
)
_ATIF_VERSIONS = {"ATIF-v1.0", "ATIF-v1.7"}
_CHAT_ROLES = {"user", "system", "assistant", "developer", "tool", "function"}
_OTLP_TARGET_ATTRIBUTES = {"input.value", "output.value", "exception.message"}
_OTLP_STRUCTURAL_ATTRIBUTES = {
    "openinference.span.kind",
    "gen_ai.conversation.id",
    "gen_ai.system",
    "gen_ai.request.model",
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens",
    "gen_ai.usage.total_tokens",
    "nemo.evaluation.name",
    "nemo.test_case.id",
}


class _PlanAFlow(Protocol):
    def protect(self, records: tuple[_ProtectionRecord, ...]) -> _ProtectionRunRecord: ...


class IntakeValidationError(RuntimeError):
    """Sanitized validation-only complete-item rejection."""


def protect_atif(source: bytes, *, flow: _PlanAFlow, emit: Emitter) -> bytes:
    try:
        document = _json_object(source)
        _validate_atif(document)
        mapping = _mapping(document, version=f"intake-{document['schema_version']}", kind="atif")
        return _protect_json(source, mapping=mapping, flow=flow, emit=emit)
    except IntakeValidationError:
        raise
    except Exception:
        raise IntakeValidationError("ATIF item rejected") from None


def protect_chat_completion(source: bytes, *, flow: _PlanAFlow, emit: Emitter) -> bytes:
    try:
        document = _json_object(source)
        _validate_chat_completion(document)
        mapping = _mapping(document, version="intake-chat-completion-v1", kind="chat")
        return _protect_json(source, mapping=mapping, flow=flow, emit=emit)
    except IntakeValidationError:
        raise
    except Exception:
        raise IntakeValidationError("chat completion rejected") from None


def _protect_json(source: bytes, *, mapping: TraceMapping, flow: _PlanAFlow, emit: Emitter) -> bytes:
    try:
        projected = project_complete_item(
            source,
            source_format=SourceFormat.JSON,
            mapping=mapping,
            bounds=_JSON_BOUNDS,
        )
        protected = _protect_projected(projected, flow=flow)
    except IntakeValidationError:
        raise
    except StructuredItemError:
        raise IntakeValidationError("structured item rejected") from None
    emit(protected)
    return protected


def _protect_projected(projected: ProjectedItem, *, flow: _PlanAFlow) -> bytes:
    frame = projected.dataframe
    records = tuple(
        _ProtectionRecord(_RecordRef(str(row[SEGMENT_KEY_COLUMN])), (_TextSegment(str(row[COL_TEXT])),))
        for _, row in frame.iterrows()
    )
    run = flow.protect(records)
    if len(run.outcomes) != len(records) or not all(isinstance(item, _ProtectionSucceeded) for item in run.outcomes):
        raise IntakeValidationError("protection failed")
    result = _outcome_dataframe(cast(tuple[_ProtectionSucceeded, ...], run.outcomes))
    try:
        return reconstruct_complete_item(projected, result)
    except StructuredItemError:
        raise IntakeValidationError("protection failed") from None


def _outcome_dataframe(outcomes: tuple[_ProtectionSucceeded, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            SEGMENT_KEY_COLUMN: [outcome.ref.value for outcome in outcomes],
            PROTECTED_TEXT_COLUMN: [outcome.output for outcome in outcomes],
            COL_FINAL_ENTITIES: [{"entities": []} for _ in outcomes],
        }
    )


def _json_object(source: bytes) -> dict[str, Any]:
    value = json.loads(source)
    if not isinstance(value, dict):
        raise ValueError("object required")
    return cast(dict[str, Any], value)


def _mapping(document: dict[str, Any], *, version: str, kind: str) -> TraceMapping:
    roles: dict[str, FieldRole] = {}
    for pointer, value in _scalars(document):
        roles[pointer] = _atif_role(pointer, value) if kind == "atif" else _chat_role(pointer, value)
    identity = "/session_id"
    ordered = ("/trajectory_id",) if kind == "atif" else ("/trace_id",)
    return TraceMapping(
        version=version, fields=roles, source_identity_pointer=identity, ordered_identity_pointers=ordered
    )


def _scalars(value: object, pointer: str = "") -> list[tuple[str, object]]:
    if isinstance(value, dict):
        return [item for key, child in value.items() for item in _scalars(child, f"{pointer}/{_escape(key)}")]
    if isinstance(value, list):
        return [item for index, child in enumerate(value) for item in _scalars(child, f"{pointer}/{index}")]
    return [(pointer, value)]


def _escape(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _atif_role(pointer: str, value: object) -> FieldRole:
    tokens = pointer.split("/")[1:]
    if pointer in {"/session_id", "/trajectory_id"}:
        return FieldRole.STRUCTURAL
    if isinstance(value, str) and _atif_target(tokens):
        return FieldRole.TARGET
    if "extra" in tokens and isinstance(value, str):
        raise ValueError("string extension lacks reviewed field policy")
    return FieldRole.PRESERVE


def _atif_target(tokens: list[str]) -> bool:
    if tokens == ["notes"]:
        return True
    if len(tokens) >= 3 and tokens[0] == "steps" and tokens[2] == "message":
        return tokens[-1] in {"message", "text"}
    if len(tokens) == 3 and tokens[0] == "steps" and tokens[2] == "reasoning_content":
        return True
    if "tool_calls" in tokens and "arguments" in tokens:
        return True
    return "observation" in tokens and "content" in tokens and tokens[-1] in {"content", "text"}


def _chat_role(pointer: str, value: object) -> FieldRole:
    tokens = pointer.split("/")[1:]
    if pointer in {"/session_id", "/trace_id"}:
        return FieldRole.STRUCTURAL
    if _chat_target(tokens, value):
        return FieldRole.TARGET
    return FieldRole.PRESERVE


def _chat_target(tokens: list[str], value: object) -> bool:
    if value is None:
        return False
    if len(tokens) == 4 and tokens[:2] == ["request", "messages"] and tokens[-1] == "content":
        return True
    if tokens[:2] == ["request", "messages"] and tokens[-1] == "arguments":
        return True
    if tokens[:2] == ["response", "choices"] and tokens[-1] == "content":
        return True
    return tokens[:2] == ["response", "error"] and tokens[-1] == "message"


def _validate_atif(document: dict[str, Any]) -> None:
    _require_keys(document, {"schema_version", "session_id", "trajectory_id", "agent", "steps"})
    allowed = {
        "schema_version",
        "session_id",
        "trajectory_id",
        "agent",
        "steps",
        "notes",
        "final_metrics",
        "continued_trajectory_ref",
        "extra",
        "subagent_trajectories",
        "evaluation_context",
    }
    if set(document) - allowed or document["schema_version"] not in _ATIF_VERSIONS:
        raise ValueError("unsupported ATIF shape")
    _validate_atif_agent(document["agent"])
    steps = document["steps"]
    if not isinstance(steps, list):
        raise ValueError("steps must be a list")
    for index, step in enumerate(steps, start=1):
        _validate_atif_step(step, expected_id=index)


def _validate_atif_agent(value: object) -> None:
    agent = _as_dict(value)
    _require_keys(agent, {"name", "version"})
    if set(agent) - {"name", "version", "model_name", "tool_definitions", "extra"}:
        raise ValueError("unknown agent field")


def _validate_atif_step(value: object, *, expected_id: int) -> None:
    step = _as_dict(value)
    _require_keys(step, {"step_id", "source", "message"})
    common = {
        "step_id",
        "timestamp",
        "message",
        "is_copied_context",
        "extra",
        "llm_call_count",
        "observation",
        "source",
    }
    agent = {"model_name", "reasoning_effort", "reasoning_content", "tool_calls", "metrics"}
    if step["step_id"] != expected_id or step["source"] not in {"system", "user", "agent"}:
        raise ValueError("invalid ATIF step identity")
    if set(step) - (common | (agent if step["source"] == "agent" else set())):
        raise ValueError("unknown step field")
    _validate_atif_content(step["message"])
    _validate_atif_calls(step)


def _validate_atif_calls(step: dict[str, Any]) -> None:
    calls = step.get("tool_calls") or []
    call_ids: set[str] = set()
    for value in calls:
        call = _as_dict(value)
        _require_keys(call, {"tool_call_id", "function_name"})
        if set(call) - {"tool_call_id", "function_name", "arguments", "extra"} or call["tool_call_id"] in call_ids:
            raise ValueError("invalid tool call")
        call_ids.add(call["tool_call_id"])
    observation = _as_dict(step["observation"]) if step.get("observation") is not None else {"results": []}
    if set(observation) - {"results"}:
        raise ValueError("invalid observation")
    for value in observation.get("results", []):
        result = _as_dict(value)
        if set(result) - {"source_call_id", "content", "subagent_trajectory_ref", "extra"}:
            raise ValueError("invalid observation")
        if result.get("source_call_id") is not None and result["source_call_id"] not in call_ids:
            raise ValueError("unresolved observation")
        if result.get("content") is not None:
            _validate_atif_content(result["content"])


def _validate_atif_content(value: object) -> None:
    if isinstance(value, str):
        return
    if not isinstance(value, list):
        raise ValueError("invalid ATIF content")
    for part_value in value:
        part = _as_dict(part_value)
        if part.get("type") != "text" or set(part) != {"type", "text"} or not isinstance(part["text"], str):
            raise ValueError("unreviewed ATIF content part")


def _validate_chat_completion(document: dict[str, Any]) -> None:
    _require_keys(document, {"request", "response", "session_id", "trace_id"})
    allowed = {
        "request",
        "response",
        "session_id",
        "trace_id",
        "provider",
        "cost_usd",
        "cost_input_usd",
        "cost_output_usd",
        "cost_details",
        "evaluation_context",
    }
    if set(document) - allowed:
        raise ValueError("unknown top-level field")
    request = _as_dict(document["request"])
    response = _as_dict(document["response"])
    _validate_chat_request(request)
    _validate_chat_response(response)
    if ("choices" in response) == ("error" in response):
        raise ValueError("response requires exactly one variant")


def _validate_chat_request(request: dict[str, Any]) -> None:
    _require_keys(request, {"model", "messages"})
    if set(request) - {"model", "messages", "temperature", "provider_extension"}:
        raise ValueError("unreviewed request extension")
    extension = _as_dict(request["provider_extension"]) if "provider_extension" in request else {}
    if set(extension) - {"region", "request_class"}:
        raise ValueError("unreviewed provider extension")
    for value in request["messages"]:
        _validate_chat_message(_as_dict(value))


def _validate_chat_message(message: dict[str, Any]) -> None:
    if message.get("role") not in _CHAT_ROLES:
        raise ValueError("invalid chat role")
    if set(message) - {"role", "content", "tool_calls", "tool_call_id", "name"}:
        raise ValueError("unreviewed message extension")
    for value in message.get("tool_calls") or []:
        call = _as_dict(value)
        if set(call) != {"id", "type", "function"}:
            raise ValueError("invalid tool call")
        function = _as_dict(call["function"])
        if set(function) != {"name", "arguments"} or not isinstance(function["arguments"], str):
            raise ValueError("invalid tool function")


def _validate_chat_response(response: dict[str, Any]) -> None:
    allowed = {"id", "object", "model", "choices", "error", "usage", "provider_response_id"}
    if set(response) - allowed:
        raise ValueError("unreviewed response extension")
    for value in response.get("choices") or []:
        choice = _as_dict(value)
        if set(choice) - {"index", "message", "finish_reason"}:
            raise ValueError("invalid choice")
        message = _as_dict(choice["message"])
        if set(message) - {"role", "content"} or message.get("role") not in _CHAT_ROLES:
            raise ValueError("invalid response message")
    if "usage" in response and set(_as_dict(response["usage"])) - {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    }:
        raise ValueError("unreviewed usage field")


def _require_keys(value: Mapping[str, object], required: set[str]) -> None:
    if not required.issubset(value):
        raise ValueError("required field missing")


def _as_dict(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("object required")
    return cast(dict[str, Any], value)


def build_local_otlp_request(spec_path: Path) -> bytes:
    spec = _json_object(spec_path.read_bytes())
    request = ExportTraceServiceRequest()
    resource_spans = request.resource_spans.add()
    _add_attributes(resource_spans.resource.attributes, _as_dict(spec["resource_attributes"]))
    scope_spans = resource_spans.scope_spans.add()
    scope_spans.scope.name = str(spec["scope"]["name"])
    scope_spans.scope.version = str(spec["scope"]["version"])
    for span_spec in cast(list[dict[str, Any]], spec["spans"]):
        _add_span(scope_spans.spans.add(), span_spec)
    return request.SerializeToString(deterministic=True)


def _add_span(span: Any, spec: dict[str, Any]) -> None:
    span.trace_id = bytes.fromhex(spec["trace_id"])
    span.span_id = bytes.fromhex(spec["span_id"])
    if spec.get("parent_span_id"):
        span.parent_span_id = bytes.fromhex(spec["parent_span_id"])
    span.name = spec["name"]
    span.start_time_unix_nano = spec["start_time_unix_nano"]
    span.end_time_unix_nano = spec["end_time_unix_nano"]
    _add_attributes(span.attributes, spec["attributes"])


def _add_attributes(target: Any, values: Mapping[str, object]) -> None:
    for key, value in values.items():
        attribute = target.add()
        attribute.key = key
        if isinstance(value, bool):
            attribute.value.bool_value = value
        elif isinstance(value, int):
            attribute.value.int_value = value
        elif isinstance(value, float):
            attribute.value.double_value = value
        else:
            attribute.value.string_value = str(value)


def protect_otlp_request(source: bytes, *, flow: _PlanAFlow, emit: Emitter) -> bytes:
    request = _parse_otlp(source)
    targets = _otlp_targets(request)
    records = tuple(_ProtectionRecord(_RecordRef(ref), (_TextSegment(value),)) for ref, value in targets.items())
    run = flow.protect(records)
    if len(run.outcomes) != len(records) or not all(isinstance(item, _ProtectionSucceeded) for item in run.outcomes):
        raise IntakeValidationError("protection failed")
    replacements = {
        cast(_ProtectionSucceeded, item).ref.value: cast(_ProtectionSucceeded, item).output for item in run.outcomes
    }
    protected_request = ExportTraceServiceRequest()
    protected_request.CopyFrom(request)
    _apply_otlp_replacements(protected_request, replacements)
    protected = protected_request.SerializeToString(deterministic=True)
    emit(protected)
    return protected


def _parse_otlp(source: bytes) -> ExportTraceServiceRequest:
    try:
        request = ExportTraceServiceRequest.FromString(source)
    except DecodeError:
        raise IntakeValidationError("OTLP batch rejected") from None
    spans = list(_spans(request))
    if not spans or any(not _valid_span_identity(span) for span in spans) or not _valid_otlp_envelope(request):
        raise IntakeValidationError("OTLP batch rejected")
    return request


def _valid_otlp_envelope(request: ExportTraceServiceRequest) -> bool:
    for resource_spans in request.resource_spans:
        resource_attributes = resource_spans.resource.attributes
        if any(
            item.key != "service.name" or item.value.WhichOneof("value") != "string_value"
            for item in resource_attributes
        ):
            return False
        for scope_spans in resource_spans.scope_spans:
            if scope_spans.scope.attributes:
                return False
            if any(span.events or span.links for span in scope_spans.spans):
                return False
    return True


def _valid_span_identity(span: Any) -> bool:
    return (
        len(span.trace_id) == 16
        and any(span.trace_id)
        and len(span.span_id) == 8
        and any(span.span_id)
        and (not span.parent_span_id or (len(span.parent_span_id) == 8 and any(span.parent_span_id)))
    )


def _otlp_targets(request: ExportTraceServiceRequest) -> dict[str, str]:
    targets: dict[str, str] = {}
    for span in _spans(request):
        span_id = span.span_id.hex()
        for attribute in span.attributes:
            if attribute.key not in _OTLP_TARGET_ATTRIBUTES | _OTLP_STRUCTURAL_ATTRIBUTES:
                raise IntakeValidationError("OTLP batch rejected")
            if attribute.key in _OTLP_TARGET_ATTRIBUTES:
                if attribute.value.WhichOneof("value") != "string_value":
                    raise IntakeValidationError("OTLP batch rejected")
                targets[f"{span_id}:{attribute.key}"] = attribute.value.string_value
        if span.status.message:
            targets[f"{span_id}:status.message"] = span.status.message
    return targets


def _apply_otlp_replacements(request: ExportTraceServiceRequest, replacements: Mapping[str, str]) -> None:
    found: set[str] = set()
    for span in _spans(request):
        span_id = span.span_id.hex()
        for attribute in span.attributes:
            ref = f"{span_id}:{attribute.key}"
            if ref in replacements:
                attribute.value.string_value = replacements[ref]
                found.add(ref)
        status_ref = f"{span_id}:status.message"
        if status_ref in replacements:
            span.status.message = replacements[status_ref]
            found.add(status_ref)
    if found != set(replacements):
        raise IntakeValidationError("OTLP batch rejected")


def _spans(request: ExportTraceServiceRequest):
    for resource_spans in request.resource_spans:
        for scope_spans in resource_spans.scope_spans:
            yield from scope_spans.spans


def otlp_topology(source: bytes) -> tuple[tuple[str, str, str], ...]:
    request = _parse_otlp(source)
    return tuple((span.trace_id.hex(), span.span_id.hex(), span.parent_span_id.hex()) for span in _spans(request))


def otlp_string_attributes(source: bytes) -> Mapping[str, Mapping[str, str]]:
    request = _parse_otlp(source)
    return {
        span.span_id.hex(): {
            attribute.key: attribute.value.string_value
            for attribute in span.attributes
            if attribute.value.WhichOneof("value") == "string_value"
        }
        for span in _spans(request)
    }
