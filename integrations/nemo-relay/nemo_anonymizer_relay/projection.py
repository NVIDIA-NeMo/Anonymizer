# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select replaceable text from copied Relay observability values."""

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
class TextLeaf:
    """One replaceable string value or application-defined mapping key."""

    path: JsonPath
    text: str
    key: str | None = None


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

# Closed Relay/provider vocabulary is structural. Other keys are application
# data and are inspected because identifiers can themselves contain PII.
_STRUCTURAL_MAPPING_KEYS = frozenset(
    {
        "annotated_request",
        "annotated_response",
        "annotation",
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
        "headers",
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
        "request",
        "response",
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


def omit_unsupported_media(value: Json) -> Json:
    """Copy a value while replacing recognized media and large binary bodies."""

    return _omit_unsupported_media(deepcopy(value))


def collect_text_leaves(
    value: Json,
    *,
    max_leaves: int,
    max_bytes: int,
    preserve_protocol_values: bool = False,
) -> list[TextLeaf]:
    """Select caller-controlled strings within a copied observability value."""

    leaves: list[TextLeaf] = []
    selected_bytes = 0

    def add(leaf: TextLeaf) -> None:
        nonlocal selected_bytes
        encoded_bytes = len(leaf.text.encode("utf-8"))
        if len(leaves) >= max_leaves or selected_bytes + encoded_bytes > max_bytes:
            raise ProjectionLimitError(f"selected text exceeds {max_leaves} leaves or {max_bytes} UTF-8 bytes")
        leaves.append(leaf)
        selected_bytes += encoded_bytes

    def visit(current: Json, path: JsonPath, field: str | None) -> None:
        if isinstance(current, str):
            if current and not _is_protocol_literal(field, current, preserve_protocol_values):
                add(TextLeaf(path=path, text=current))
            return
        if isinstance(current, list):
            for index, item in enumerate(current):
                visit(item, (*path, index), field)
            return
        if isinstance(current, dict):
            for key, item in current.items():
                if isinstance(key, str) and key not in _STRUCTURAL_MAPPING_KEYS:
                    add(TextLeaf(path=path, text=key, key=key))
                visit(item, (*path, key), key)

    visit(value, (), None)
    return leaves


def replace_text_leaves(
    value: Json,
    replacements: dict[tuple[JsonPath, str | None], str],
) -> Json:
    """Copy a value while applying decisions to string values and mapping keys."""

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


def _is_protocol_literal(field: str | None, value: str, preserve: bool) -> bool:
    return preserve and field is not None and value in _PROTOCOL_VALUE_LITERALS.get(field, ())


def _omit_unsupported_media(value: Json) -> Json:
    if isinstance(value, str):
        if value.startswith("data:") or _ENCODED_BINARY.fullmatch(value):
            return _OMITTED_MEDIA
        return value
    if isinstance(value, list):
        return [_omit_unsupported_media(item) for item in value]
    if not isinstance(value, dict):
        return value

    media_type = value.get("type")
    normalized_type = media_type.lower() if isinstance(media_type, str) else ""
    if normalized_type in _MEDIA_TYPES or normalized_type.startswith(("audio/", "image/", "video/")):
        return {"type": media_type, "content": _OMITTED_MEDIA}
    if isinstance(value.get("mime_type"), str) and "data" in value:
        return {"mime_type": value["mime_type"], "data": _OMITTED_MEDIA}
    return {key: _omit_unsupported_media(item) for key, item in value.items()}


__all__ = [
    "ProjectionLimitError",
    "TextLeaf",
    "collect_text_leaves",
    "omit_unsupported_media",
    "replace_text_leaves",
]
