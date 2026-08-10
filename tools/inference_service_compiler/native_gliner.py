#!/usr/bin/env -S uv run --script
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "cyclopts>=3.0",
#   "fastapi>=0.115",
#   "gliner>=0.2.21",
#   "gliner2[local]>=1.3",
#   "structlog>=24.4",
#   "uvicorn>=0.30",
# ]
# ///
"""Serve local GLiNER PII detection through Anonymizer's OpenAI wire contract.

This server is launched by ``tools/inference_service.py`` from a compiled run
plan. The default ``nvidia-gliner`` family loads ``nvidia/gliner-pii``;
``gliner2`` loads Fastino's local PII checkpoint.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import math
import os
import sys
import time
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, cast

import structlog  # type: ignore[unresolved-import]
import uvicorn
from cyclopts import App
from fastapi import FastAPI, HTTPException, Request  # type: ignore[unresolved-import]

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8001
DEFAULT_CHUNK_LENGTH = 384
DEFAULT_OVERLAP = 128
DEFAULT_FLAT_NER = False
DEFAULT_INFERENCE_BATCH_SIZE = 8
NVIDIA_GLINER_CHECKPOINT = "nvidia/gliner-pii"
GLINER2_CHECKPOINT = "fastino/gliner2-privacy-filter-PII-multi"
BATCH_MODE = os.getenv("GLINER_BATCH_MODE", "true").lower() not in {"0", "false", "no"}
MAX_BATCH_REQUESTS = int(os.getenv("GLINER_MAX_BATCH_REQUESTS", "32"))
BATCH_WAIT_SECONDS = float(os.getenv("GLINER_BATCH_WAIT_MS", "10")) / 1000


class ModelFamily(StrEnum):
    """Supported local model runtime families."""

    NVIDIA_GLINER = "nvidia-gliner"
    GLINER2 = "gliner2"


class LogFormat(StrEnum):
    """Supported server log renderers."""

    PLAIN = "plain"
    JSON = "json"


class RequestValidationError(ValueError):
    """A malformed detector request at the pure JSON boundary."""


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """Immutable server configuration chosen by the CLI."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    model: ModelFamily = ModelFamily.NVIDIA_GLINER
    checkpoint: str | None = None
    revision: str | None = None

    def __post_init__(self) -> None:
        """Reject invalid transport values before model side effects."""
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")

    @property
    def resolved_checkpoint(self) -> str:
        """Return the family default unless the user supplied an override."""
        if self.checkpoint:
            return self.checkpoint
        match self.model:
            case ModelFamily.NVIDIA_GLINER:
                return NVIDIA_GLINER_CHECKPOINT
            case ModelFamily.GLINER2:
                return GLINER2_CHECKPOINT


@dataclass(frozen=True, slots=True)
class Entity:
    """One normalized entity in Anonymizer's detector response shape."""

    text: str
    label: str
    start: int
    end: int
    score: float

    def as_dict(self) -> dict[str, str | int | float]:
        """Serialize the entity in Anonymizer's required flat schema."""
        return {"text": self.text, "label": self.label, "start": self.start, "end": self.end, "score": self.score}


@dataclass(frozen=True, slots=True)
class DetectParams:
    """Per-request inference settings that determine batching compatibility."""

    labels: tuple[str, ...]
    threshold: float
    chunk_length: int
    overlap: int
    flat_ner: bool
    inference_batch_size: int


class LocalRuntime(Protocol):
    """Narrow common contract for local inference adapters."""

    def infer(self, chunks: list[str], params: DetectParams) -> list[list[Entity]]:
        """Detect normalized entities for each chunk."""


class NvidiaGlinerRuntime:
    """Adapter for the original `gliner` local inference API."""

    def __init__(self, model: object) -> None:
        self._model = model

    def infer(self, chunks: list[str], params: DetectParams) -> list[list[Entity]]:
        raw = cast(
            object,
            self._model.inference(  # type: ignore[attr-defined]
                texts=chunks,
                labels=list(params.labels),
                threshold=params.threshold,
                flat_ner=params.flat_ner,
                relations=[],
                batch_size=params.inference_batch_size,
            ),
        )
        return normalize_nvidia_output(raw)


class Gliner2Runtime:
    """Adapter for GLiNER2's local batch extraction API."""

    def __init__(self, model: object) -> None:
        self._model = model

    def infer(self, chunks: list[str], params: DetectParams) -> list[list[Entity]]:
        raw = cast(
            object,
            self._model.batch_extract_entities(  # type: ignore[attr-defined]
                chunks,
                list(params.labels),
                threshold=params.threshold,
                include_confidence=True,
                include_spans=True,
                batch_size=params.inference_batch_size,
            ),
        )
        return normalize_gliner2_output(raw)


def create_text_chunks(text: str, chunk_length: int, overlap: int) -> tuple[list[str], list[int]]:
    """Split text into overlapping chunks while retaining their global offsets."""
    chunks: list[str] = []
    offsets: list[int] = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_length])
        offsets.append(start)
        if start + chunk_length >= len(text):
            break
        start += chunk_length - overlap
    return chunks, offsets


def normalize_nvidia_output(raw: object) -> list[list[Entity]]:
    """Convert original GLiNER dictionaries at the runtime boundary.

    Args:
        raw: The `GLiNER.inference` batch result.

    Returns:
        One normalized entity list per input chunk.

    Raises:
        ValueError: If the runtime did not return GLiNER's documented batch shape.
    """
    chunks = require_list(raw, "nvidia-gliner inference batch")
    return [
        [normalize_nvidia_entity(entity) for entity in require_list(chunk, "nvidia-gliner chunk")] for chunk in chunks
    ]


def normalize_gliner2_output(raw: object) -> list[list[Entity]]:
    """Convert GLiNER2 entity records, including confidence and character spans.

    Args:
        raw: The `GLiNER2.batch_extract_entities` result.

    Returns:
        One normalized entity list per input chunk.

    Raises:
        ValueError: If the runtime did not return a list of result mappings.
    """
    return [
        normalize_gliner2_result(require_mapping(result, "gliner2 result"))
        for result in require_list(raw, "gliner2 inference batch")
    ]


def load_runtime(config: ServerConfig, device: str) -> LocalRuntime:
    """Load the selected local runtime at the sole heavyweight side-effect boundary.

    Args:
        config: Selected model family and checkpoint.
        device: Local Torch device name.

    Returns:
        A runtime adapter for inference.
    """
    match config.model:
        case ModelFamily.NVIDIA_GLINER:
            gliner = importlib.import_module("gliner")
            model = gliner.GLiNER.from_pretrained(
                config.resolved_checkpoint,
                map_location=device,
                revision=config.revision,
            )
            return NvidiaGlinerRuntime(model)
        case ModelFamily.GLINER2:
            gliner2 = importlib.import_module("gliner2")
            checkpoint = config.resolved_checkpoint
            if config.revision is not None:
                hub = importlib.import_module("huggingface_hub")
                checkpoint = hub.snapshot_download(repo_id=checkpoint, revision=config.revision)
            model = gliner2.GLiNER2.from_pretrained(checkpoint, map_location=device)
            return Gliner2Runtime(model)


def resolve_device() -> str:
    """Choose an explicit DEVICE override or the best available local accelerator."""
    requested = os.getenv("DEVICE", "auto")
    if requested != "auto":
        return requested
    torch = importlib.import_module("torch")
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def finalize_entities(entities: list[Entity], *, flat_ner: bool) -> list[Entity]:
    """Deduplicate overlap artifacts and optionally remove nested spans."""
    candidates = entities if flat_ner else remove_subset_entities(entities)
    best: dict[tuple[str, str, int, int], Entity] = {}
    for entity in candidates:
        key = (entity.label, entity.text.strip().lower(), entity.start, entity.end)
        if key not in best or entity.score > best[key].score:
            best[key] = entity
    return list(best.values())


def remove_subset_entities(entities: list[Entity]) -> list[Entity]:
    """Discard an entity wholly contained by a distinct larger entity."""
    return [
        entity
        for entity in entities
        if not any(
            other != entity
            and other.start <= entity.start
            and other.end >= entity.end
            and (other.start < entity.start or other.end > entity.end)
            for other in entities
        )
    ]


def detect_entities_for_texts(runtime: LocalRuntime, texts: list[str], params: DetectParams) -> list[list[Entity]]:
    """Run all text chunks through one runtime batch and restore global offsets."""
    if not params.labels:
        return [[] for _ in texts]
    records = [
        (text_index, offset, chunk)
        for text_index, text in enumerate(texts)
        if text
        for chunk, offset in zip(*create_text_chunks(text, params.chunk_length, params.overlap), strict=True)
    ]
    output = [[] for _ in texts]
    if not records:
        return output
    inferred = runtime.infer([chunk for _, _, chunk in records], params)
    for (text_index, offset, _), entities in zip(records, inferred, strict=True):
        output[text_index].extend(
            Entity(entity.text, entity.label, entity.start + offset, entity.end + offset, entity.score)
            for entity in entities
        )
    return [finalize_entities(entities, flat_ner=params.flat_ner) for entities in output]


@dataclass(slots=True)
class DetectJob:
    """Queued request awaiting a shared inference call."""

    text: str
    params: DetectParams
    future: asyncio.Future[list[Entity]]


class BatchDetector:
    """Coalesce compatible requests while retaining one inference executor."""

    def __init__(self, runtime: LocalRuntime) -> None:
        self._runtime = runtime
        self._queue: asyncio.Queue[DetectJob | None] = asyncio.Queue()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gliner-infer")
        self._worker_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Start the request-coalescing worker."""
        self._worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        """Drain and stop the worker and its dedicated inference executor."""
        if self._worker_task is not None:
            await self._queue.put(None)
            await self._worker_task
        self._executor.shutdown(wait=True)

    async def detect(self, text: str, params: DetectParams) -> list[Entity]:
        """Queue one detection request or execute it serially when batching is off."""
        loop = asyncio.get_running_loop()
        if not BATCH_MODE:
            return (
                await loop.run_in_executor(self._executor, detect_entities_for_texts, self._runtime, [text], params)
            )[0]
        future: asyncio.Future[list[Entity]] = loop.create_future()
        await self._queue.put(DetectJob(text, params, future))
        return await future

    async def _worker(self) -> None:
        while first := await self._queue.get():
            jobs = [first]
            deadline = asyncio.get_running_loop().time() + BATCH_WAIT_SECONDS
            while len(jobs) < MAX_BATCH_REQUESTS:
                try:
                    queued = await asyncio.wait_for(
                        self._queue.get(), max(0, deadline - asyncio.get_running_loop().time())
                    )
                except TimeoutError:
                    break
                if queued is None:
                    await self._queue.put(None)
                    break
                jobs.append(queued)
            await self._dispatch(jobs)

    async def _dispatch(self, jobs: list[DetectJob]) -> None:
        groups: dict[DetectParams, list[DetectJob]] = {}
        for job in jobs:
            groups.setdefault(job.params, []).append(job)
        loop = asyncio.get_running_loop()
        for params, group in groups.items():
            try:
                results = await loop.run_in_executor(
                    self._executor, detect_entities_for_texts, self._runtime, [job.text for job in group], params
                )
            except Exception as exc:
                for job in group:
                    if not job.future.done():
                        job.future.set_exception(exc)
            else:
                for job, entities in zip(group, results, strict=True):
                    if not job.future.done():
                        job.future.set_result(entities)


def normalize_nvidia_entity(raw: object) -> Entity:
    """Normalize one original GLiNER entity dictionary."""
    mapping = require_mapping(raw, "nvidia-gliner entity")
    return Entity(
        str(mapping["text"]),
        str(mapping["label"]),
        coerce_int(mapping["start"], "nvidia-gliner start"),
        coerce_int(mapping["end"], "nvidia-gliner end"),
        coerce_float(mapping["score"], "nvidia-gliner score"),
    )


def normalize_gliner2_result(raw: Mapping[str, object]) -> list[Entity]:
    """Normalize one GLiNER2 result mapping keyed by entity label."""
    entities = require_mapping(raw.get("entities"), "gliner2 entities")
    return [
        normalize_gliner2_entity(label, item)
        for label, values in entities.items()
        for item in require_sequence(values, "gliner2 entity values")
    ]


def normalize_gliner2_entity(label: object, raw: object) -> Entity:
    """Normalize GLiNER2's span/confidence record for a named label."""
    entity = require_mapping(raw, "gliner2 entity")
    span = entity.get("span")
    if isinstance(span, Sequence) and not isinstance(span, str) and len(span) == 2:
        start, end = span
    else:
        start, end = entity["start"], entity["end"]
    confidence = entity.get("confidence", entity.get("score"))
    if confidence is None:
        raise ValueError("gliner2 entity is missing confidence")
    return Entity(
        str(entity["text"]),
        str(label),
        coerce_int(start, "gliner2 start"),
        coerce_int(end, "gliner2 end"),
        coerce_float(confidence, "gliner2 confidence"),
    )


def require_mapping(value: object, name: str) -> Mapping[str, object]:
    """Validate an untyped runtime mapping at the adapter boundary."""
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"unexpected {name} shape")
    return cast(Mapping[str, object], value)


def require_sequence(value: object, name: str) -> Sequence[object]:
    """Validate an untyped runtime sequence at the adapter boundary."""
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError(f"unexpected {name} shape")
    return value


def require_list(value: object, name: str) -> list[object]:
    """Validate a runtime list at the adapter boundary."""
    if not isinstance(value, list):
        raise ValueError(f"unexpected {name} shape")
    return cast(list[object], value)


def coerce_int(value: object, name: str) -> int:
    """Convert a runtime numeric value or raise a boundary-specific error."""
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        raise ValueError(f"unexpected {name} value")
    return int(value)


def coerce_float(value: object, name: str) -> float:
    """Convert a runtime numeric value or raise a boundary-specific error."""
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        raise ValueError(f"unexpected {name} value")
    return float(value)


def require_request_int(value: object, name: str) -> int:
    """Accept only a JSON integer, excluding booleans and lossy coercions."""
    match value:
        case bool():
            raise RequestValidationError(f"{name} must be an integer")
        case int():
            return value
        case _:
            raise RequestValidationError(f"{name} must be an integer")


def require_request_float(value: object, name: str) -> float:
    """Accept a finite JSON number without parsing strings or booleans."""
    match value:
        case bool():
            raise RequestValidationError(f"{name} must be a number")
        case int() | float():
            number = float(value)
        case _:
            raise RequestValidationError(f"{name} must be a number")
    if not math.isfinite(number):
        raise RequestValidationError(f"{name} must be finite")
    return number


def require_request_bool(value: object, name: str) -> bool:
    """Accept a JSON boolean without truthiness coercion."""
    match value:
        case bool():
            return value
        case _:
            raise RequestValidationError(f"{name} must be a boolean")


def parse_detect_params(body: Mapping[str, object]) -> DetectParams:
    """Parse and validate request options without performing I/O."""
    raw_labels = body.get("labels", [])
    match raw_labels:
        case list() if all(isinstance(label, str) for label in raw_labels):
            labels = tuple(cast(str, label) for label in raw_labels)
        case _:
            raise RequestValidationError("labels must be a list of strings")

    threshold = require_request_float(body.get("threshold", 0.3), "threshold")
    if not 0 <= threshold <= 1:
        raise RequestValidationError("threshold must be between 0 and 1")

    chunk_length = require_request_int(body.get("chunk_length", DEFAULT_CHUNK_LENGTH), "chunk_length")
    overlap = require_request_int(body.get("overlap", DEFAULT_OVERLAP), "overlap")
    flat_ner = require_request_bool(body.get("flat_ner", DEFAULT_FLAT_NER), "flat_ner")
    inference_batch_size = require_request_int(body.get("batch_size", DEFAULT_INFERENCE_BATCH_SIZE), "batch_size")
    if inference_batch_size < 1:
        raise RequestValidationError("batch_size must be >= 1")
    validate_chunk_params(chunk_length, overlap)
    return DetectParams(labels, threshold, chunk_length, overlap, flat_ner, inference_batch_size)


def extract_text(messages: object) -> str:
    """Extract text from the final user message's string or multipart content."""
    if not isinstance(messages, list):
        raise RequestValidationError("messages must be a list")
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
                raise RequestValidationError("message part text must be a string")
            parts.append(text)
        return "".join(parts)
    raise RequestValidationError("message content must be a string or list")


def validate_chunk_params(chunk_length: int, overlap: int) -> None:
    """Reject invalid chunk settings before inference."""
    if chunk_length < 1:
        raise RequestValidationError("chunk_length must be >= 1")
    if overlap < 0 or overlap >= chunk_length:
        raise RequestValidationError("overlap must be >= 0 and less than chunk_length")


state: ServerConfig | None = None
runtime: LocalRuntime | None = None
detector: BatchDetector | None = None
log = structlog.get_logger("gliner-server")


def configure_logging(log_format: LogFormat) -> None:
    """Configure the selected human-readable or structured log renderer."""
    match log_format:
        case LogFormat.PLAIN:
            renderer = structlog.dev.ConsoleRenderer()
        case LogFormat.JSON:
            renderer = structlog.processors.JSONRenderer()
    structlog.configure(processors=[renderer], wrapper_class=structlog.make_filtering_bound_logger(20))


@asynccontextmanager
async def lifespan(_api: FastAPI) -> AsyncIterator[None]:
    """Own the local runtime and its single inference worker for API lifetime."""
    global runtime, detector
    if state is None:
        raise RuntimeError("server configuration is not initialized")
    device = resolve_device()
    runtime = await asyncio.to_thread(load_runtime, state, device)
    detector = BatchDetector(runtime)
    detector.start()
    log.info("server_ready", model=state.model, checkpoint=state.resolved_checkpoint, device=device)
    try:
        yield
    finally:
        if detector is not None:
            await detector.stop()


api = FastAPI(lifespan=lifespan)
app = api


@api.get("/v1/models")
def list_models() -> dict[str, object]:
    """Return the selected local checkpoint in OpenAI's model-list shape."""
    checkpoint = state.resolved_checkpoint if state else NVIDIA_GLINER_CHECKPOINT
    return {"object": "list", "data": [{"id": checkpoint, "object": "model"}]}


@api.post("/v1/chat/completions")
async def chat_completions(request: Request) -> dict[str, object]:
    """Detect requested entity labels and return Anonymizer's JSON-string content."""
    if detector is None:
        raise HTTPException(status_code=503, detail="GLiNER model is not loaded")
    try:
        body = require_mapping(await request.json(), "request")
        params = parse_detect_params(body)
        text = extract_text(body.get("messages", []))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    entities = await detector.detect(text, params)
    content = json.dumps({"entities": [entity.as_dict() for entity in entities]})
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": str(body.get("model", state.resolved_checkpoint if state else NVIDIA_GLINER_CHECKPOINT)),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


cli = App(help="OpenAI-compatible local GLiNER server for Anonymizer.")


@cli.default
def main(
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    model: ModelFamily = ModelFamily.NVIDIA_GLINER,
    checkpoint: str | None = None,
    revision: str | None = None,
    log_format: LogFormat = LogFormat.PLAIN,
) -> None:
    """Run the server without contacting any remote inference service.

    Args:
        host: Bind address; use a private address unless protected by a proxy.
        port: TCP listen port.
        model: `nvidia-gliner` or `gliner2` local model family.
        checkpoint: Optional Hugging Face checkpoint override for that family.
        revision: Optional immutable Hugging Face model revision.
        log_format: Human-readable `plain` logs or newline-delimited `json`.
    """
    global state
    try:
        state = ServerConfig(host=host, port=port, model=model, checkpoint=checkpoint, revision=revision)
    except ValueError as exc:
        sys.stderr.write(f"error: {exc}\n")
        raise SystemExit(125) from exc
    configure_logging(log_format)
    uvicorn.run(api, host=host, port=port)


if __name__ == "__main__":
    cli()
