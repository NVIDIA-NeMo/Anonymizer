# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible GLiNER2 server for notebook and development use."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import socket
import time
import uuid
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request

try:
    from .backend import (  # noqa: TID252 - direct-script fallback keeps server dependencies isolated
        MODEL_ID,
        MODEL_REVISION,
        detect_entities_for_texts,
        load_model,
        resolve_device,
    )
except ImportError:  # Direct execution inside the isolated notebook-server environment.
    from backend import (  # ty: ignore[unresolved-import] -- direct-script import path
        MODEL_ID,
        MODEL_REVISION,
        detect_entities_for_texts,
        load_model,
        resolve_device,
    )

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8001
DEFAULT_CHUNK_LENGTH = 384
DEFAULT_OVERLAP = 128
DEFAULT_FLAT_NER = False
DEFAULT_INFERENCE_BATCH_SIZE = 8
TOKEN_ENV = "ANONYMIZER_LOCAL_GLINER2_TOKEN"
DEVICE_ENV = "ANONYMIZER_LOCAL_GLINER2_DEVICE"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("anonymizer-local-gliner2")

model: Any | None = None
selected_device = "cpu"


@dataclass(frozen=True)
class DetectParams:
    """Inference settings used to group compatible concurrent requests."""

    labels: tuple[str, ...]
    threshold: float
    chunk_length: int
    overlap: int
    flat_ner: bool
    inference_batch_size: int


@dataclass
class DetectJob:
    """One queued detection request."""

    text: str
    params: DetectParams
    future: asyncio.Future[list[dict[str, Any]]]


class BatchDetector:
    """Coalesce requests and serialize model access through one worker."""

    def __init__(self, *, max_requests: int = 32, wait_seconds: float = 0.01) -> None:
        self._max_requests = max_requests
        self._wait_seconds = wait_seconds
        self._queue: asyncio.Queue[DetectJob | None] = asyncio.Queue()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gliner2-infer")
        self._worker_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        if self._worker_task is None:
            return
        await self._queue.put(None)
        await self._worker_task
        self._executor.shutdown(wait=True)
        self._worker_task = None

    async def detect(self, text: str, params: DetectParams) -> list[dict[str, Any]]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[dict[str, Any]]] = loop.create_future()
        await self._queue.put(DetectJob(text=text, params=params, future=future))
        return await future

    async def _worker(self) -> None:
        while True:
            first = await self._queue.get()
            if first is None:
                return
            jobs = [first]
            deadline = asyncio.get_running_loop().time() + self._wait_seconds
            while len(jobs) < self._max_requests:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    job = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                except TimeoutError:
                    break
                if job is None:
                    await self._queue.put(None)
                    break
                jobs.append(job)
            await self._dispatch(jobs)

    async def _dispatch(self, jobs: list[DetectJob]) -> None:
        grouped: dict[DetectParams, list[DetectJob]] = {}
        for job in jobs:
            grouped.setdefault(job.params, []).append(job)
        loop = asyncio.get_running_loop()
        for params, group in grouped.items():
            try:
                results = await loop.run_in_executor(
                    self._executor,
                    lambda p=params, g=group: detect_entities_for_texts(
                        model,
                        [job.text for job in g],
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


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load the model and own the batching worker for the server lifespan."""
    global model, selected_device
    selected_device = resolve_device(os.getenv(DEVICE_ENV, "auto"))
    logger.info("Loading %s at revision %s on %s", MODEL_ID, MODEL_REVISION, selected_device)
    model = await asyncio.to_thread(load_model, selected_device)
    detector.start()
    logger.info("Local GLiNER2 is ready")
    try:
        yield
    finally:
        await detector.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/v1/models")
def list_models(request: Request) -> dict[str, Any]:
    """Return readiness and immutable checkpoint identity."""
    _authorize(request)
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "revision": MODEL_REVISION,
                "device": selected_device,
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> dict[str, Any]:
    """Run GLiNER2 through the detector chat-completion contract."""
    _authorize(request)
    if model is None:
        raise HTTPException(status_code=503, detail="GLiNER2 is not loaded")
    body = await request.json()
    text = _extract_text(body.get("messages", []))
    labels = body.get("labels") or []
    params = DetectParams(
        labels=tuple(str(label) for label in labels),
        threshold=float(body.get("threshold", 0.3)),
        chunk_length=int(body.get("chunk_length", DEFAULT_CHUNK_LENGTH)),
        overlap=int(body.get("overlap", DEFAULT_OVERLAP)),
        flat_ner=bool(body.get("flat_ner", DEFAULT_FLAT_NER)),
        inference_batch_size=int(body.get("batch_size", DEFAULT_INFERENCE_BATCH_SIZE)),
    )
    try:
        entities = await detector.detect(text, params)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    content = json.dumps({"entities": entities})
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", MODEL_ID),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _authorize(request: Request) -> None:
    expected = os.getenv(TOKEN_ENV)
    if expected and request.headers.get("authorization") != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Invalid local runtime token")


def _extract_text(messages: object) -> str:
    if not isinstance(messages, list) or not messages:
        return ""
    last = messages[-1]
    if not isinstance(last, dict):
        return ""
    content = last.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return str(content)


def main() -> None:
    """Run the lightweight notebook/development server."""
    parser = argparse.ArgumentParser(description="Notebook/development GLiNER2 server for Anonymizer.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--fd", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.fd is None:
        uvicorn.run(app, host=args.host, port=args.port)
        return
    with socket.socket(fileno=args.fd) as inherited_listener:
        config = uvicorn.Config(app)
        uvicorn.Server(config).run(sockets=[inherited_listener])


if __name__ == "__main__":
    main()
