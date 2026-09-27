# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select replaceable text from copied Relay observability values."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, TypeAlias

Json = Any
PathPart: TypeAlias = str | int
JsonPath: TypeAlias = tuple[PathPart, ...]


class ProjectionLimitError(ValueError):
    """The selected observability text exceeded the configured work budget."""


@dataclass(frozen=True)
class TextLeaf:
    """One replaceable string value or mapping key."""

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
            "file",
            "image",
            "image_url",
            "input_audio",
            "input_file",
            "input_image",
            "input_text",
            "json_schema",
            "message",
            "output_audio",
            "output_file",
            "output_image",
            "output_text",
            "screenshot",
            "text",
            "video",
        }
    ),
}
_DATA_URI = re.compile(r"^\s*data:", re.IGNORECASE)
_ENCODED_BINARY = re.compile(r"^[A-Za-z0-9+/_-]{4096,}={0,2}$")
_OMITTED_MEDIA = "[UNSUPPORTED MEDIA OMITTED]"
_INTERNAL_MARKERS = frozenset({_OMITTED_MEDIA, "[REDACTED]", "[REDACTED_SECRET]"})


def omit_unsupported_media(value: Json) -> Json:
    """Copy a value while replacing inline media and large encoded bodies.

    Mapping shape and field names are retained for downstream consumers of
    copied provider and annotation schemas. URLs, file IDs, and ordinary text
    remain available for text inspection; only data URIs and binary-looking
    strings are omitted.
    """

    return _omit_unsupported_media(value)


def collect_text_leaves(
    value: Json,
    *,
    max_leaves: int,
    max_bytes: int,
    max_leaf_bytes: int,
    preserve_protocol_values: bool = False,
) -> list[TextLeaf]:
    """Select caller-controlled strings within a copied observability value."""

    leaves: list[TextLeaf] = []
    selected_bytes = 0

    def add(leaf: TextLeaf) -> None:
        nonlocal selected_bytes
        encoded_bytes = len(leaf.text.encode("utf-8"))
        if encoded_bytes > max_leaf_bytes:
            raise ProjectionLimitError(f"one selected text leaf exceeds {max_leaf_bytes} UTF-8 bytes")
        if len(leaves) >= max_leaves or selected_bytes + encoded_bytes > max_bytes:
            raise ProjectionLimitError(f"selected text exceeds {max_leaves} leaves or {max_bytes} UTF-8 bytes")
        leaves.append(leaf)
        selected_bytes += encoded_bytes

    def visit(current: Json, path: JsonPath, field: str | None) -> None:
        if isinstance(current, str):
            if (
                current
                and current not in _INTERNAL_MARKERS
                and not _is_protocol_literal(field, current, preserve_protocol_values)
            ):
                add(TextLeaf(path=path, text=current))
            return
        if isinstance(current, list):
            for index, item in enumerate(current):
                visit(item, (*path, index), field)
            return
        if isinstance(current, dict):
            for key, item in current.items():
                if isinstance(key, str):
                    add(TextLeaf(path=path, text=key, key=key))
                visit(item, (*path, key), key)

    visit(value, (), None)
    return leaves


def replace_text_leaves(
    value: Json,
    replacements: dict[JsonPath, str],
) -> Json:
    """Copy a value while applying decisions to string values."""

    def visit(current: Json, path: JsonPath) -> Json:
        value_replacement = replacements.get(path)
        if value_replacement is not None:
            return value_replacement
        if isinstance(current, list):
            return [visit(item, (*path, index)) for index, item in enumerate(current)]
        if isinstance(current, dict):
            return {key: visit(item, (*path, key)) for key, item in current.items()}
        return current

    return visit(value, ())


def _is_protocol_literal(field: str | None, value: str, preserve: bool) -> bool:
    return preserve and field is not None and value in _PROTOCOL_VALUE_LITERALS.get(field, ())


def _omit_unsupported_media(value: Json) -> Json:
    if isinstance(value, str):
        if _DATA_URI.match(value) or _ENCODED_BINARY.fullmatch(value):
            return _OMITTED_MEDIA
        return value
    if isinstance(value, list):
        return [_omit_unsupported_media(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {key: _omit_unsupported_media(item) for key, item in value.items()}


__all__ = [
    "ProjectionLimitError",
    "TextLeaf",
    "collect_text_leaves",
    "omit_unsupported_media",
    "replace_text_leaves",
]
