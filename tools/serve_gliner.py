# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal OpenAI-compatible GLiNER server for Anonymizer detection.

Fakes /v1/chat/completions so Anonymizer routes its `entity_detector` role
here instead of hitting build.nvidia.com. The response's message.content is
a JSON string of shape ``{"entities": [...]}`` as expected by
``anonymizer.engine.detection.postprocess.parse_raw_entities``.

Run:
    python tools/serve_gliner.py    # binds 0.0.0.0:8001

Uses GPU when ``torch.cuda.is_available()`` returns True; otherwise CPU.
Batch mode coalesces concurrent requests (typical when Anonymizer/DataDesigner
runs with ``max_parallel_requests`` > 1) and uses GLiNER's batched ``inference()``
with optional sequence packing for chunk throughput.

See ``docs/concepts/self-hosting-gliner.md`` for full usage.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, Request
from gliner import GLiNER

try:
    from gliner import InferencePackingConfig
except ImportError:  # pragma: no cover - older gliner releases
    InferencePackingConfig = None  # type: ignore[misc, assignment]

MODEL_NAME = "nvidia/gliner-pii"
HOST = "0.0.0.0"
PORT = 8001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_CHUNK_LENGTH = 384
DEFAULT_OVERLAP = 128
DEFAULT_FLAT_NER = False
DEFAULT_INFERENCE_BATCH_SIZE = 8

BATCH_MODE = os.getenv("GLINER_BATCH_MODE", "true").lower() not in {"0", "false", "no"}
MAX_BATCH_REQUESTS = int(os.getenv("GLINER_MAX_BATCH_REQUESTS", "32"))
BATCH_WAIT_MS = float(os.getenv("GLINER_BATCH_WAIT_MS", "10")) / 1000.0
ENABLE_PACKING = os.getenv("GLINER_ENABLE_PACKING", "true").lower() not in {"0", "false", "no"}
PACKING_STREAMS = int(os.getenv("GLINER_PACKING_STREAMS", "4"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gliner-server")

app = FastAPI()
log.info("loading %s on %s", MODEL_NAME, DEVICE)
model = GLiNER.from_pretrained(MODEL_NAME).to(DEVICE)
if ENABLE_PACKING and InferencePackingConfig is not None:
    model.configure_inference_packing(
        InferencePackingConfig(max_length=DEFAULT_CHUNK_LENGTH, streams_per_batch=PACKING_STREAMS)
    )
    log.info("sequence packing enabled (max_length=%d streams=%d)", DEFAULT_CHUNK_LENGTH, PACKING_STREAMS)
elif ENABLE_PACKING:
    log.warning("GLINER_ENABLE_PACKING=true but InferencePackingConfig is unavailable; skipping packing")


@dataclass(frozen=True)
class DetectParams:
    labels: tuple[str, ...]
    threshold: float
    chunk_length: int
    overlap: int
    flat_ner: bool
    inference_batch_size: int


@dataclass
class DetectJob:
    text: str
    params: DetectParams
    future: asyncio.Future[list[dict[str, Any]]]


def _extract_text(messages: list[dict[str, Any]]) -> str:
    """Pull the user message text. Handles both string and multi-part content."""
    if not messages:
        return ""
    content = messages[-1].get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


def _create_text_chunks(text: str, chunk_length: int, overlap: int) -> tuple[list[str], list[int]]:
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


def _shift_offsets(entities: list[dict[str, Any]], offset: int) -> None:
    for entity in entities:
        entity["start"] = int(entity["start"]) + offset
        entity["end"] = int(entity["end"]) + offset


def _dedupe_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for entity in entities:
        label = str(entity.get("label", ""))
        text = str(entity.get("text", ""))
        key = (label, text.strip().lower())
        score = float(entity.get("score", 0.0))
        if key not in best or score > float(best[key].get("score", 0.0)):
            best[key] = entity
    return list(best.values())


def _format_entities(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "text": e["text"],
            "label": e["label"],
            "start": e["start"],
            "end": e["end"],
            "score": e["score"],
        }
        for e in raw
    ]


def _run_chunk_inference(
    chunks: list[str],
    *,
    labels: list[str],
    threshold: float,
    flat_ner: bool,
    inference_batch_size: int,
) -> list[list[dict[str, Any]]]:
    if not chunks:
        return []
    batch = model.inference(
        texts=chunks,
        labels=labels,
        threshold=threshold,
        flat_ner=flat_ner,
        relations=[],
        batch_size=inference_batch_size,
    )
    if not isinstance(batch, list) or (batch and not isinstance(batch[0], list)):
        raise ValueError("unexpected GLiNER inference batch shape")
    return batch


def _detect_entities_for_text(
    text: str,
    labels: list[str],
    *,
    threshold: float,
    chunk_length: int,
    overlap: int,
    flat_ner: bool,
    inference_batch_size: int,
) -> list[dict[str, Any]]:
    if not labels or not text:
        return []

    chunks, offsets = _create_text_chunks(text, chunk_length, overlap)
    batch = _run_chunk_inference(
        chunks,
        labels=labels,
        threshold=threshold,
        flat_ner=flat_ner,
        inference_batch_size=inference_batch_size,
    )

    merged: list[dict[str, Any]] = []
    for chunk_entities, offset in zip(batch, offsets, strict=True):
        adjusted = [dict(entity) for entity in chunk_entities]
        _shift_offsets(adjusted, offset)
        merged.extend(adjusted)

    return _format_entities(_dedupe_entities(merged))


def _detect_entities_for_texts(
    texts: list[str],
    labels: list[str],
    *,
    threshold: float,
    chunk_length: int,
    overlap: int,
    flat_ner: bool,
    inference_batch_size: int,
) -> list[list[dict[str, Any]]]:
    if not labels:
        return [[] for _ in texts]

    chunk_records: list[tuple[int, int, str]] = []
    for text_idx, text in enumerate(texts):
        if not text:
            continue
        chunks, offsets = _create_text_chunks(text, chunk_length, overlap)
        for chunk, offset in zip(chunks, offsets, strict=True):
            chunk_records.append((text_idx, offset, chunk))

    per_text: list[list[dict[str, Any]]] = [[] for _ in texts]
    if not chunk_records:
        return per_text

    flat_chunks = [chunk for _, _, chunk in chunk_records]
    batch = _run_chunk_inference(
        flat_chunks,
        labels=labels,
        threshold=threshold,
        flat_ner=flat_ner,
        inference_batch_size=inference_batch_size,
    )

    for (text_idx, offset, _), chunk_entities in zip(chunk_records, batch, strict=True):
        adjusted = [dict(entity) for entity in chunk_entities]
        _shift_offsets(adjusted, offset)
        per_text[text_idx].extend(adjusted)

    return [_format_entities(_dedupe_entities(entities)) for entities in per_text]


class BatchDetector:
    """Coalesce concurrent detect requests into shared GLiNER inference calls."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[DetectJob | None] = asyncio.Queue()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gliner-infer")
        self._worker_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        if self._worker_task is None:
            return
        await self._queue.put(None)
        await self._worker_task
        self._executor.shutdown(wait=True)

    async def detect(self, text: str, params: DetectParams) -> list[dict[str, Any]]:
        if not BATCH_MODE:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor,
                lambda: _detect_entities_for_text(
                    text,
                    list(params.labels),
                    threshold=params.threshold,
                    chunk_length=params.chunk_length,
                    overlap=params.overlap,
                    flat_ner=params.flat_ner,
                    inference_batch_size=params.inference_batch_size,
                ),
            )

        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[dict[str, Any]]] = loop.create_future()
        await self._queue.put(DetectJob(text=text, params=params, future=future))
        return await future

    async def _worker(self) -> None:
        while True:
            first = await self._queue.get()
            if first is None:
                break

            batch = [first]
            deadline = asyncio.get_running_loop().time() + BATCH_WAIT_MS
            while len(batch) < MAX_BATCH_REQUESTS:
                timeout = deadline - asyncio.get_running_loop().time()
                if timeout <= 0:
                    break
                try:
                    job = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                except TimeoutError:
                    break
                if job is None:
                    await self._queue.put(None)
                    break
                batch.append(job)

            await self._dispatch(batch)

    async def _dispatch(self, jobs: list[DetectJob]) -> None:
        grouped: dict[DetectParams, list[DetectJob]] = {}
        for job in jobs:
            grouped.setdefault(job.params, []).append(job)

        loop = asyncio.get_running_loop()
        for params, group in grouped.items():
            texts = [job.text for job in group]
            try:
                results = await loop.run_in_executor(
                    self._executor,
                    lambda p=params, t=texts: _detect_entities_for_texts(
                        t,
                        list(p.labels),
                        threshold=p.threshold,
                        chunk_length=p.chunk_length,
                        overlap=p.overlap,
                        flat_ner=p.flat_ner,
                        inference_batch_size=p.inference_batch_size,
                    ),
                )
            except Exception as exc:
                for job in group:
                    if not job.future.done():
                        job.future.set_exception(exc)
                continue

            for job, entities in zip(group, results, strict=True):
                if not job.future.done():
                    job.future.set_result(entities)


detector = BatchDetector()


@app.on_event("startup")
async def startup() -> None:
    log.info(
        "batch mode=%s max_requests=%d wait_ms=%.0f inference_batch_size default=%d",
        BATCH_MODE,
        MAX_BATCH_REQUESTS,
        BATCH_WAIT_MS * 1000,
        DEFAULT_INFERENCE_BATCH_SIZE,
    )
    detector.start()


@app.on_event("shutdown")
async def shutdown() -> None:
    await detector.stop()


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> dict[str, Any]:
    body = await request.json()
    text = _extract_text(body.get("messages", []))
    labels = body.get("labels") or []
    threshold = float(body.get("threshold", 0.3))
    chunk_length = int(body.get("chunk_length", DEFAULT_CHUNK_LENGTH))
    overlap = int(body.get("overlap", DEFAULT_OVERLAP))
    flat_ner = bool(body.get("flat_ner", DEFAULT_FLAT_NER))
    inference_batch_size = int(body.get("batch_size", DEFAULT_INFERENCE_BATCH_SIZE))

    params = DetectParams(
        labels=tuple(labels),
        threshold=threshold,
        chunk_length=chunk_length,
        overlap=overlap,
        flat_ner=flat_ner,
        inference_batch_size=inference_batch_size,
    )

    log.info(
        "detect: labels=%d threshold=%.2f chunk=%d overlap=%d flat_ner=%s batch_size=%d text_len=%d",
        len(labels),
        threshold,
        chunk_length,
        overlap,
        flat_ner,
        inference_batch_size,
        len(text),
    )
    entities = await detector.detect(text, params)
    content = json.dumps({"entities": entities})
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", MODEL_NAME),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
