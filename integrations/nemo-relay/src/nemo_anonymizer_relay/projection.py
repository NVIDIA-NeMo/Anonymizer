# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project text out of copied Relay events and put it back safely."""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, TypeAlias

Json = Any
PathPart: TypeAlias = str | int
JsonPath: TypeAlias = tuple[PathPart, ...]


class ProjectionLimitError(ValueError):
    """The selected observability text exceeded the configured work budget."""


@dataclass(frozen=True)
class ExportTextLeaf:
    """One replaceable value, or one data-shaped mapping key."""

    path: JsonPath
    text: str
    key: str | None = None


# These values describe a provider protocol rather than user content. Rewriting
# them can invalidate otherwise-valid copied requests and responses.
_PROTOCOL_VALUE_LITERALS = {
    "finish_reason": frozenset({"content_filter", "error", "length", "stop", "tool_calls"}),
    "object": frozenset({"chat.completion", "chat.completion.chunk", "response"}),
    "role": frozenset({"assistant", "developer", "function", "model", "system", "tool", "user"}),
    "stop_reason": frozenset({"end_turn", "max_tokens", "stop_sequence", "tool_use"}),
    "type": frozenset(
        {
            "function",
            "function_call",
            "function_call_output",
            "input_text",
            "json_schema",
            "message",
            "output_text",
            "text",
        }
    ),
}

_STRUCTURAL_MAPPING_KEYS = frozenset(
    {
        "annotated_request",
        "annotated_response",
        "api",
        "api_specific",
        "arguments",
        "atof_version",
        "attributes",
        "category",
        "category_profile",
        "content",
        "created",
        "data",
        "data_schema",
        "description",
        "finish_reason",
        "function",
        "id",
        "index",
        "input",
        "input_schema",
        "kind",
        "max_tokens",
        "message",
        "messages",
        "metadata",
        "model",
        "model_name",
        "name",
        "object",
        "output",
        "parameters",
        "parent_uuid",
        "params",
        "provider",
        "propagation_root_uuid",
        "role",
        "scope_category",
        "stop_reason",
        "stream",
        "stream_options",
        "subtype",
        "text",
        "timestamp",
        "tool_call_id",
        "tool_calls",
        "tools",
        "type",
        "usage",
        "uuid",
    }
)

_EVENT_ENVELOPE_KEYS = (
    "atof_version",
    "attributes",
    "category",
    "data_schema",
    "kind",
    "name",
    "parent_uuid",
    "propagation_root_uuid",
    "scope_category",
    "timestamp",
    "uuid",
)

# These fields can contain producer-selected strings and must pass through the
# detector even though they sit outside Relay's mutable event-sanitizer fields.
EXPORT_SANITIZE_FIELDS = (
    "name",
    "category",
    "data_schema",
    "data",
    "category_profile",
    "metadata",
)

_FAIL_CLOSED_IDENTITY_KEYS = (
    "atof_version",
    "kind",
    "parent_uuid",
    "propagation_root_uuid",
    "scope_category",
    "timestamp",
    "uuid",
)

_KNOWN_SCOPE_ATTRIBUTES = frozenset(
    {
        "parallel",
        "relocatable",
        "remote",
        "stateful",
        "streaming",
    }
)

_EXPORT_METADATA_KEYS = frozenset(
    {
        "nemo_anonymizer.coverage",
        "nemo_anonymizer.failure",
        "nemo_anonymizer.provider_body_omitted",
        "nemo_anonymizer.sequence",
        "nemo_anonymizer.unknown_attributes_omitted",
        "nemo_anonymizer.unsupported_media_omitted",
    }
)

_MEDIA_TYPES = frozenset(
    {
        "audio",
        "computer_screenshot",
        "file",
        "image",
        "image_url",
        "input_audio",
        "input_file",
        "input_image",
        "output_audio",
        "output_file",
        "output_image",
        "screenshot",
        "video",
    }
)
_ENCODED_BINARY = re.compile(r"^[A-Za-z0-9+/_-]{4096,}={0,2}$")
_OMITTED_MEDIA = "[UNSUPPORTED MEDIA OMITTED]"


def prepare_event_for_export(event: Json) -> Json:
    """Build the only event copy the protected exporter is allowed to publish.

    Codec-backed LLM events already carry Relay's annotated request or response.
    In that case the exact provider body in ``data`` is deliberately omitted so
    the exporter does not maintain a second provider codec or publish an
    unchecked duplicate. Provider-native fragments retained inside Relay's
    annotation are treated as observable JSON and every data-shaped string is
    inspected; they are never re-encoded for execution.
    """

    if not isinstance(event, dict):
        raise TypeError("Relay subscriber event must be an object")
    protected = {key: deepcopy(event.get(key)) for key in _EVENT_ENVELOPE_KEYS if key in event}
    safe_attributes, omitted_attributes = _safe_scope_attributes(event.get("attributes"))
    if "attributes" in event:
        protected["attributes"] = safe_attributes
    if event.get("kind") == "mark" and event.get("name") == "llm.chunk":
        # Relay intentionally emits compact receipts for every provider chunk.
        # Their payload adds no final model text and can dominate a coding run,
        # so retain trace identity while omitting all caller-controlled fields.
        protected.update(
            {
                "category": "llm",
                "data_schema": None,
                "data": None,
                "category_profile": None,
                "metadata": {
                    "nemo_anonymizer.coverage": "stream_chunk_payload_omitted",
                    "nemo_anonymizer.provider_body_omitted": True,
                    **(
                        {"nemo_anonymizer.unknown_attributes_omitted": omitted_attributes} if omitted_attributes else {}
                    ),
                },
            }
        )
        return protected
    category = event.get("category")
    profile = event.get("category_profile")
    annotation_available = (
        category == "llm"
        and isinstance(profile, dict)
        and any(key in profile for key in ("annotated_request", "annotated_response"))
    )
    protected["data"] = None if annotation_available else deepcopy(event.get("data"))
    protected["category_profile"] = deepcopy(profile)
    source_metadata = event.get("metadata")
    protected["metadata"] = (
        {
            key: deepcopy(item)
            for key, item in source_metadata.items()
            if not (isinstance(key, str) and key.startswith("nemo_anonymizer."))
        }
        if isinstance(source_metadata, dict)
        else {}
    )
    omitted_media = 0
    for field in ("data", "category_profile", "metadata"):
        protected[field], field_omissions = _omit_unsupported_media(protected[field])
        omitted_media += field_omissions
    metadata = protected["metadata"]
    if not isinstance(metadata, dict):
        metadata = {}
        protected["metadata"] = metadata
    coverage = "relay_annotation" if annotation_available else "generic_json_text"
    metadata["nemo_anonymizer.provider_body_omitted"] = annotation_available
    if omitted_media:
        metadata["nemo_anonymizer.unsupported_media_omitted"] = omitted_media
    if omitted_attributes:
        metadata["nemo_anonymizer.unknown_attributes_omitted"] = omitted_attributes
    if omitted_media or omitted_attributes:
        coverage += "_partial"
    metadata["nemo_anonymizer.coverage"] = coverage
    return protected


def requires_sanitization(event: Json) -> bool:
    """Return whether a prepared event still contains inspectable payload."""

    if not isinstance(event, dict):
        return True
    metadata = event.get("metadata")
    return not (
        isinstance(metadata, dict) and metadata.get("nemo_anonymizer.coverage") == "stream_chunk_payload_omitted"
    )


def omitted_event(event: Json, reason: str) -> Json:
    """Return a fail-closed envelope containing no caller-controlled payload."""

    if not isinstance(event, dict):
        event = {}
    protected = {key: event.get(key) for key in _FAIL_CLOSED_IDENTITY_KEYS if key in event}
    if "attributes" in event:
        protected["attributes"], _ = _safe_scope_attributes(event.get("attributes"))
    protected.update(
        {
            "name": "nemo_anonymizer.omitted",
            "category": "unknown",
            "data_schema": None,
            "data": None,
            "category_profile": None,
            "metadata": {
                "nemo_anonymizer.coverage": "omitted",
                "nemo_anonymizer.failure": reason,
            },
        }
    )
    return protected


def _safe_scope_attributes(value: Json) -> tuple[list[str], int]:
    """Preserve Relay's structural flags and omit unrecognized strings."""

    if not isinstance(value, list):
        return [], int(value is not None)
    safe = [item for item in value if isinstance(item, str) and item in _KNOWN_SCOPE_ATTRIBUTES]
    return safe, len(value) - len(safe)


def collect_export_text_leaves(
    value: Json,
    *,
    max_leaves: int,
    max_bytes: int,
) -> list[ExportTextLeaf]:
    """Select observable strings and every non-protocol mapping key.

    Applications can place personal data in identifiers as well as values, so
    only a closed vocabulary of Relay and provider structural keys is exempt.
    Known provider vocabulary is preserved only inside Relay's semantic LLM
    profile; similarly named values in generic events are inspected.
    """

    leaves: list[ExportTextLeaf] = []
    selected_bytes = 0

    def add(leaf: ExportTextLeaf) -> None:
        nonlocal selected_bytes
        encoded_bytes = len(leaf.text.encode("utf-8"))
        if len(leaves) >= max_leaves or selected_bytes + encoded_bytes > max_bytes:
            raise ProjectionLimitError(f"selected text exceeds {max_leaves} leaves or {max_bytes} UTF-8 bytes")
        leaves.append(leaf)
        selected_bytes += encoded_bytes

    def visit(current: Json, path: JsonPath, field: str | None) -> None:
        if isinstance(current, str):
            if current and not _preserve_export_string(path, field, current):
                add(ExportTextLeaf(path=path, text=current))
            return
        if isinstance(current, list):
            for index, item in enumerate(current):
                visit(item, (*path, index), field)
            return
        if isinstance(current, dict):
            for key, item in current.items():
                if _inspect_mapping_key(path, key):
                    add(ExportTextLeaf(path=path, text=key, key=key))
                visit(item, (*path, key), key)

    visit(value, (), None)
    return leaves


def replace_export_text_leaves(
    value: Json,
    replacements: dict[tuple[JsonPath, str | None], str],
) -> Json:
    """Copy a projected event while applying value and mapping-key decisions."""

    def visit(current: Json, path: JsonPath) -> Json:
        value_replacement = replacements.get((path, None))
        if value_replacement is not None:
            return value_replacement
        if isinstance(current, list):
            return [visit(item, (*path, index)) for index, item in enumerate(current)]
        if isinstance(current, dict):
            rewritten: dict[str, Json] = {}
            for key, item in current.items():
                candidate = replacements.get((path, key), key)
                unique = candidate
                suffix = 2
                while unique in rewritten:
                    unique = f"{candidate}__{suffix}"
                    suffix += 1
                rewritten[unique] = visit(item, (*path, key))
            return rewritten
        return current

    return visit(value, ())


def _preserve_export_string(path: JsonPath, field: str | None, value: str) -> bool:
    """Preserve only closed protocol vocabulary inside Relay's semantic profile.

    Generic tool/mark ``data`` and ``metadata`` are application-controlled. A
    value named ``role`` or ``model`` there can contain PII and must not gain a
    global exemption merely because provider protocols use the same key.
    """

    if field in _EXPORT_METADATA_KEYS and _is_top_level_metadata_path(path[:-1]):
        return True
    profile_index = 1 if path and isinstance(path[0], int) else 0
    if len(path) <= profile_index or path[profile_index] != "category_profile":
        return False
    return field is not None and value in _PROTOCOL_VALUE_LITERALS.get(field, ())


def _inspect_mapping_key(path: JsonPath, key: Any) -> bool:
    return (
        isinstance(key, str)
        and key not in _STRUCTURAL_MAPPING_KEYS
        and not (key in _EXPORT_METADATA_KEYS and _is_top_level_metadata_path(path))
    )


def _is_top_level_metadata_path(path: JsonPath) -> bool:
    return path == ("metadata",) or (len(path) == 2 and isinstance(path[0], int) and path[1] == "metadata")


def _omit_unsupported_media(value: Json) -> tuple[Json, int]:
    """Remove media/binary bodies before any detector or sink can observe them."""

    if isinstance(value, str):
        if value.startswith("data:") or _ENCODED_BINARY.fullmatch(value):
            return _OMITTED_MEDIA, 1
        return value, 0
    if isinstance(value, list):
        output: list[Json] = []
        omitted = 0
        for item in value:
            safe_item, item_omissions = _omit_unsupported_media(item)
            output.append(safe_item)
            omitted += item_omissions
        return output, omitted
    if not isinstance(value, dict):
        return value, 0

    media_type = value.get("type")
    normalized_type = media_type.lower() if isinstance(media_type, str) else ""
    if normalized_type in _MEDIA_TYPES or normalized_type.startswith(("audio/", "image/", "video/")):
        return {"type": media_type, "content": _OMITTED_MEDIA}, 1
    if isinstance(value.get("mime_type"), str) and "data" in value:
        return {"mime_type": value["mime_type"], "data": _OMITTED_MEDIA}, 1

    output: dict[str, Json] = {}
    omitted = 0
    for key, item in value.items():
        safe_item, item_omissions = _omit_unsupported_media(item)
        output[key] = safe_item
        omitted += item_omissions
    return output, omitted
