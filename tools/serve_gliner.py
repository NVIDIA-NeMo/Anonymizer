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
See ``docs/concepts/self-hosting-gliner.md`` for full usage.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, Request
from gliner import GLiNER

MODEL_NAME = "nvidia/gliner-pii"
HOST = "0.0.0.0"
PORT = 8001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gliner-server")

app = FastAPI()
log.info("loading %s on %s", MODEL_NAME, DEVICE)
model = GLiNER.from_pretrained(MODEL_NAME).to(DEVICE)


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


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> dict[str, Any]:
    body = await request.json()
    text = _extract_text(body.get("messages", []))
    labels = body.get("labels") or []
    threshold = float(body.get("threshold", 0.3))

    log.info("detect: labels=%d threshold=%.2f text_len=%d", len(labels), threshold, len(text))
    raw = model.predict_entities(text, labels, threshold=threshold) if (labels and text) else []
    entities = [
        {"text": e["text"], "label": e["label"], "start": e["start"], "end": e["end"], "score": e["score"]}
        for e in raw
    ]
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
