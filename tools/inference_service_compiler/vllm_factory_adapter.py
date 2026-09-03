# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate Anonymizer detector requests into vLLM Factory pooling calls."""

from __future__ import annotations

import asyncio
import importlib
import json
import math
import os
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, assert_never, cast

from inference_service_compiler.models import FactoryPlugin, parse_factory_plugin

DEFAULT_CHUNK_LENGTH = 384
DEFAULT_OVERLAP = 128
MAX_CHUNKS_PER_REQUEST = 256
MAX_CONCURRENT_POOLING_CALLS = 8
POOLING_LIMITER_STATE_ATTRIBUTE = "_anonymizer_pooling_limiter"


@dataclass(frozen=True, slots=True)
class DetectionRequest:
    """Validated Anonymizer detector request."""

    model: str
    text: str
    labels: tuple[str, ...]
    threshold: float
    chunk_length: int
    overlap: int
    flat_ner: bool


@dataclass(frozen=True, slots=True)
class Entity:
    """One entity in Anonymizer's detector response shape."""

    text: str
    label: str
    start: int
    end: int
    score: float

    def as_dict(self) -> dict[str, str | int | float]:
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "score": self.score,
        }


@dataclass(frozen=True, slots=True)
class TextChunk:
    """One text segment and its original document offset."""

    text: str
    offset: int


async def anonymizer_chat_compatibility(
    request: Any,
    call_next: Callable[[Any], Awaitable[Any]],
) -> Any:
    """Serve Anonymizer's chat contract through the active factory IOProcessor."""
    if request.url.path != "/v1/chat/completions":
        return await call_next(request)

    responses = importlib.import_module("starlette.responses")
    try:
        detection = parse_detection_request(await request.json())
        plugin = parse_factory_plugin(os.environ["ANONYMIZER_VLLM_FACTORY_PLUGIN"])
        entities: list[Entity] = []
        if detection.labels:
            chunks = split_text(detection.text, detection.chunk_length, detection.overlap)
            handler = request.app.state.serving_pooling
            if handler is None:
                raise RuntimeError("vLLM pooling handler is unavailable")
            results = await invoke_pooling_chunks(
                handler=handler,
                plugin=plugin,
                detection=detection,
                chunks=chunks,
                limiter=_pooling_limiter(request.app.state),
            )
            entities = merge_entities(
                plugin=plugin,
                chunks=chunks,
                results=results,
                flat_ner=detection.flat_ner,
            )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        return responses.JSONResponse(
            status_code=400,
            content={"error": {"message": str(exc), "type": "invalid_request_error"}},
        )

    content = json.dumps({"entities": [entity.as_dict() for entity in entities]})
    return responses.JSONResponse(
        content={
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "model": detection.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }
    )


def parse_detection_request(value: object) -> DetectionRequest:
    """Validate the bounded chat-completions request used by Anonymizer."""
    body = require_mapping(value, "request body")
    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("model must be a non-empty string")
    labels_value = body.get("labels", [])
    if not isinstance(labels_value, list) or not all(isinstance(label, str) and label for label in labels_value):
        raise ValueError("labels must be a list of strings")
    threshold = require_number(body.get("threshold", 0.3), "threshold")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    chunk_length = require_integer(body.get("chunk_length", DEFAULT_CHUNK_LENGTH), "chunk_length")
    overlap = require_integer(body.get("overlap", DEFAULT_OVERLAP), "overlap")
    if chunk_length < 1:
        raise ValueError("chunk_length must be >= 1")
    if overlap < 0 or overlap >= chunk_length:
        raise ValueError("overlap must be >= 0 and less than chunk_length")
    flat_ner = body.get("flat_ner", False)
    if not isinstance(flat_ner, bool):
        raise ValueError("flat_ner must be a boolean")
    text = extract_text(body.get("messages"))
    chunk_count = _count_text_chunks(len(text), chunk_length, overlap)
    if labels_value and chunk_count > MAX_CHUNKS_PER_REQUEST:
        raise ValueError(f"detector requests may contain at most {MAX_CHUNKS_PER_REQUEST} chunks")
    return DetectionRequest(
        model=model,
        text=text,
        labels=tuple(cast(list[str], labels_value)),
        threshold=threshold,
        chunk_length=chunk_length,
        overlap=overlap,
        flat_ner=flat_ner,
    )


def extract_text(messages: object) -> str:
    """Extract text from the final user message."""
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    if not messages:
        return ""
    message = require_mapping(messages[-1], "message")
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            text = require_mapping(part, "message part").get("text", "")
            if not isinstance(text, str):
                raise ValueError("message part text must be a string")
            parts.append(text)
        return "".join(parts)
    raise ValueError("message content must be a string or list")


def _count_text_chunks(text_length: int, chunk_length: int, overlap: int) -> int:
    """Calculate the number of chunks without materializing their text."""
    if text_length <= chunk_length:
        return 1
    stride = chunk_length - overlap
    return 1 + (text_length - chunk_length + stride - 1) // stride


def split_text(text: str, chunk_length: int, overlap: int) -> list[TextChunk]:
    """Split text with the same character-offset contract as the native runtime."""
    if not text:
        return [TextChunk("", 0)]
    chunks: list[TextChunk] = []
    start = 0
    while start < len(text):
        chunks.append(TextChunk(text[start : start + chunk_length], start))
        if start + chunk_length >= len(text):
            break
        start += chunk_length - overlap
    return chunks


async def invoke_pooling_chunks(
    *,
    handler: Any,
    plugin: FactoryPlugin,
    detection: DetectionRequest,
    chunks: Sequence[TextChunk],
    limiter: asyncio.Semaphore,
) -> list[object]:
    """Process admitted chunks within the server-wide pooling budget."""
    results = [object() for _ in chunks]
    pending = iter(enumerate(chunks))

    async def worker() -> None:
        for index, chunk in pending:
            async with limiter:
                results[index] = await invoke_pooling(
                    handler=handler,
                    model=detection.model,
                    plugin=plugin,
                    text=chunk.text,
                    labels=detection.labels,
                    threshold=detection.threshold,
                    flat_ner=detection.flat_ner,
                )

    worker_count = min(len(chunks), MAX_CONCURRENT_POOLING_CALLS)
    await asyncio.gather(*(worker() for _ in range(worker_count)))
    return results


def _pooling_limiter(app_state: Any) -> asyncio.Semaphore:
    limiter = getattr(app_state, POOLING_LIMITER_STATE_ATTRIBUTE, None)
    if limiter is None:
        limiter = asyncio.Semaphore(MAX_CONCURRENT_POOLING_CALLS)
        setattr(app_state, POOLING_LIMITER_STATE_ATTRIBUTE, limiter)
    return cast(asyncio.Semaphore, limiter)


async def invoke_pooling(
    *,
    handler: Any,
    model: str,
    plugin: FactoryPlugin,
    text: str,
    labels: tuple[str, ...],
    threshold: float,
    flat_ner: bool,
) -> object:
    """Call vLLM's in-process pooling handler once for one text chunk."""
    protocol = importlib.import_module("vllm.entrypoints.pooling.pooling.protocol")
    data: dict[str, object] = {
        "text": text,
        "labels": list(labels),
        "threshold": threshold,
    }
    match plugin:
        case "deberta_gliner":
            data["flat_ner"] = flat_ner
        case "deberta_gliner2":
            data["include_confidence"] = True
            data["include_spans"] = True
        case _:
            assert_never(plugin)
    response = await handler(protocol.IOProcessorRequest(model=model, data=data), None)
    if response.status_code != 200:
        raise RuntimeError(f"vLLM Factory pooling request returned {response.status_code}")
    body = getattr(response, "body", None)
    if not isinstance(body, bytes):
        raise RuntimeError("vLLM Factory pooling response did not contain a JSON body")
    payload = require_mapping(json.loads(body), "pooling response")
    if "data" not in payload:
        raise ValueError("pooling response is missing data")
    return payload["data"]


def merge_entities(
    *,
    plugin: FactoryPlugin,
    chunks: Sequence[TextChunk],
    results: Sequence[object],
    flat_ner: bool,
) -> list[Entity]:
    """Normalize factory outputs, restore document offsets, and deduplicate overlap."""
    if len(chunks) != len(results):
        raise ValueError("vLLM Factory returned an unexpected result count")
    entities: list[Entity] = []
    for chunk, result in zip(chunks, results, strict=True):
        match plugin:
            case "deberta_gliner":
                normalized = normalize_gliner(result)
            case "deberta_gliner2":
                normalized = normalize_gliner2(result)
            case _:
                assert_never(plugin)
        for entity in normalized:
            if entity.start < 0 or entity.end < entity.start or entity.end > len(chunk.text):
                raise ValueError("vLLM Factory returned an invalid entity span")
            entities.append(
                Entity(
                    text=entity.text,
                    label=entity.label,
                    start=entity.start + chunk.offset,
                    end=entity.end + chunk.offset,
                    score=entity.score,
                )
            )
    if not flat_ner:
        entities = remove_subset_entities(entities)
    unique: dict[tuple[str, str, int, int], Entity] = {}
    for entity in entities:
        key = (entity.text.strip().casefold(), entity.label, entity.start, entity.end)
        previous = unique.get(key)
        if previous is None or entity.score > previous.score:
            unique[key] = entity
    return sorted(unique.values(), key=lambda item: (item.start, item.end, item.label))


def normalize_gliner(value: object) -> list[Entity]:
    """Normalize the vLLM Factory DeBERTa GLiNER IOProcessor output."""
    if not isinstance(value, list):
        raise ValueError("GLiNER pooling data must be a list")
    entities: list[Entity] = []
    for item in value:
        record = require_mapping(item, "GLiNER entity")
        entities.append(
            Entity(
                text=require_string(record.get("text"), "entity text"),
                label=require_string(record.get("label"), "entity label"),
                start=require_integer(record.get("start"), "entity start"),
                end=require_integer(record.get("end"), "entity end"),
                score=require_number(record.get("score"), "entity score"),
            )
        )
    return entities


def normalize_gliner2(value: object) -> list[Entity]:
    """Normalize the vLLM Factory DeBERTa GLiNER2 IOProcessor output."""
    payload = require_mapping(value, "GLiNER2 pooling data")
    by_label = require_mapping(payload.get("entities"), "GLiNER2 entities")
    entities: list[Entity] = []
    for label, records in by_label.items():
        if not isinstance(records, list):
            raise ValueError("GLiNER2 entity values must be lists")
        for item in records:
            record = require_mapping(item, "GLiNER2 entity")
            entities.append(
                Entity(
                    text=require_string(record.get("text"), "entity text"),
                    label=label,
                    start=require_integer(record.get("start"), "entity start"),
                    end=require_integer(record.get("end"), "entity end"),
                    score=require_number(record.get("confidence"), "entity confidence"),
                )
            )
    return entities


def remove_subset_entities(entities: list[Entity]) -> list[Entity]:
    """Remove spans strictly contained by another detected span."""
    return [
        entity
        for entity in entities
        if not any(
            other is not entity
            and other.start <= entity.start
            and other.end >= entity.end
            and (other.start < entity.start or other.end > entity.end)
            for other in entities
        )
    ]


def require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"unexpected {name} shape")
    return cast(Mapping[str, object], value)


def require_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def require_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def require_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number
