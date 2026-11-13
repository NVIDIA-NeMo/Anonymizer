# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible GLiNER2 server for native and development use."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import secrets
import socket
import time
import uuid
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

try:
    from .backend import (  # noqa: TID252 - direct-script fallback keeps server dependencies isolated
        ATTENTION_ENV,
        DTYPE_ENV,
        MODEL_ID,
        MODEL_REVISION,
        GLiNER2Like,
        LoadedModel,
        detect_entities_for_texts,
        load_model,
        resolve_device,
    )
    from .contract import (  # noqa: TID252 - direct-script fallback
        ChatChoice,
        ChatMessage,
        DetectedEntity,
        DetectionContent,
        DetectionRequest,
        DetectionResponse,
        ModelCard,
        ModelList,
    )
except ImportError:  # Direct execution inside the isolated server environment.
    from backend import (  # ty: ignore[unresolved-import] -- direct-script import path
        ATTENTION_ENV,
        DTYPE_ENV,
        MODEL_ID,
        MODEL_REVISION,
        GLiNER2Like,
        LoadedModel,
        detect_entities_for_texts,
        load_model,
        resolve_device,
    )
    from contract import (  # ty: ignore[unresolved-import] -- direct-script import path
        ChatChoice,
        ChatMessage,
        DetectedEntity,
        DetectionContent,
        DetectionRequest,
        DetectionResponse,
        ModelCard,
        ModelList,
    )

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8001
TOKEN_ENV = "ANONYMIZER_LOCAL_GLINER2_TOKEN"
DEVICE_ENV = "ANONYMIZER_LOCAL_GLINER2_DEVICE"
# Launch-ownership contract shared with tools/inference_service_compiler; keep these strings in sync.
LAUNCH_TOKEN_ENVIRONMENT_VARIABLE = "ANONYMIZER_INFERENCE_LAUNCH_TOKEN"
LAUNCH_OWNERSHIP_HEADER = "X-Anonymizer-Launch-Token"
LAUNCH_OWNERSHIP_PATH = "/_anonymizer/launch-ownership"
LAUNCH_OWNERSHIP_PROOF_FIELD = "launch_token_sha256"

logger = logging.getLogger("anonymizer-local-gliner2")

loaded: LoadedModel | None = None


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
    future: asyncio.Future[list[DetectedEntity]]


class BatchDetector:
    """Coalesce requests and serialize model access through one worker."""

    def __init__(self, *, max_requests: int = 32, wait_seconds: float = 0.01) -> None:
        self._max_requests = max_requests
        self._wait_seconds = wait_seconds
        self._queue: asyncio.Queue[DetectJob | None] = asyncio.Queue()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gliner2-infer")
        self._worker_task: asyncio.Task[None] | None = None

    def start(self, model: GLiNER2Like) -> None:
        self._worker_task = asyncio.create_task(self._worker(model))

    async def stop(self) -> None:
        if self._worker_task is None:
            return
        await self._queue.put(None)
        await self._worker_task
        self._executor.shutdown(wait=True)
        self._worker_task = None

    async def detect(self, text: str, params: DetectParams) -> list[DetectedEntity]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[DetectedEntity]] = loop.create_future()
        await self._queue.put(DetectJob(text=text, params=params, future=future))
        return await future

    async def _worker(self, model: GLiNER2Like) -> None:
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
            await self._dispatch(jobs, model)

    async def _dispatch(self, jobs: list[DetectJob], model: GLiNER2Like) -> None:
        grouped: dict[DetectParams, list[DetectJob]] = {}
        for job in jobs:
            grouped.setdefault(job.params, []).append(job)
        loop = asyncio.get_running_loop()
        for params, group in grouped.items():
            try:
                results = await loop.run_in_executor(
                    self._executor,
                    partial(
                        detect_entities_for_texts,
                        model,
                        [job.text for job in group],
                        list(params.labels),
                        threshold=params.threshold,
                        chunk_length=params.chunk_length,
                        overlap=params.overlap,
                        flat_ner=params.flat_ner,
                        inference_batch_size=params.inference_batch_size,
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
    global loaded
    device = resolve_device(os.getenv(DEVICE_ENV, "auto"))
    logger.info("Loading %s at revision %s on %s", MODEL_ID, MODEL_REVISION, device)
    loaded = await asyncio.to_thread(
        load_model,
        device,
        dtype=os.getenv(DTYPE_ENV, "auto"),
        attention=os.getenv(ATTENTION_ENV, "auto"),
    )
    detector.start(loaded.model)
    logger.info(
        "Local GLiNER2 is ready: device=%s dtype=%s attention=%s",
        loaded.device,
        loaded.dtype,
        loaded.attention,
    )
    try:
        yield
    finally:
        await detector.stop()


app = FastAPI(lifespan=lifespan)


@app.get(LAUNCH_OWNERSHIP_PATH, include_in_schema=False)
def launch_ownership(request: Request) -> JSONResponse:
    """Prove that this server inherited the launching controller's token."""
    expected = os.getenv(LAUNCH_TOKEN_ENVIRONMENT_VARIABLE)
    supplied = request.headers.get(LAUNCH_OWNERSHIP_HEADER)
    if not expected or supplied is None or not secrets.compare_digest(supplied.encode(), expected.encode()):
        return JSONResponse(status_code=404, content={"detail": "not found"})
    proof = hashlib.sha256(expected.encode()).hexdigest()
    return JSONResponse(content={LAUNCH_OWNERSHIP_PROOF_FIELD: proof})


@app.get("/v1/models")
def list_models(request: Request) -> ModelList:
    """Return readiness and immutable checkpoint identity."""
    _authorize(request)
    if loaded is None:
        raise HTTPException(status_code=503, detail="GLiNER2 is not loaded")
    card = ModelCard(
        id=MODEL_ID,
        revision=MODEL_REVISION,
        device=loaded.device,
        dtype=loaded.dtype,
        attention=loaded.attention,
    )
    return ModelList(data=[card])


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> DetectionResponse:
    """Run GLiNER2 through the detector chat-completion contract."""
    _authorize(request)
    if loaded is None:
        raise HTTPException(status_code=503, detail="GLiNER2 is not loaded")
    try:
        detection = DetectionRequest.model_validate_json(await request.body())
    except ValidationError as exc:
        raise HTTPException(
            status_code=422, detail=exc.errors(include_url=False, include_context=False, include_input=False)
        ) from exc
    params = DetectParams(
        labels=tuple(detection.labels),
        threshold=detection.threshold,
        chunk_length=detection.chunk_length,
        overlap=detection.overlap,
        flat_ner=detection.flat_ner,
        inference_batch_size=detection.batch_size,
    )
    try:
        entities = await detector.detect(detection.text(), params)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    content = json.dumps(DetectionContent.from_entities(entities).model_dump())
    return DetectionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=MODEL_ID if detection.model is None else detection.model,
        choices=[ChatChoice(message=ChatMessage(content=content))],
    )


def _authorize(request: Request) -> None:
    expected = os.getenv(TOKEN_ENV)
    if expected and request.headers.get("authorization") != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Invalid local runtime token")


def main() -> None:
    """Run the lightweight native/development server."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
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
