<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Self-hosting GLiNER

By default, Anonymizer's entity detection stage calls the hosted `nvidia/gliner-pii` model on `build.nvidia.com`. For PHI-sensitive workloads that cannot leave the host, or latency-critical setups, you can serve GLiNER locally instead.

The model is small (~500 MB) and runs comfortably on CPU — making it a good fit to run alongside a local LLM without competing for GPU memory. It also runs on GPU if one is available, which cuts detection latency on long documents.

---

## How it works

Anonymizer's detection workflow calls the `entity_detector` role via an OpenAI-compatible `POST /v1/chat/completions` endpoint, passing extra parameters through `extra_body`:

```json
{
    "model": "nvidia/gliner-pii",
    "messages": [{"role": "user", "content": "<the input text>"}],
    "labels": ["first_name", "last_name", "email", ...],
    "threshold": 0.3,
    "chunk_length": 384,
    "overlap": 128,
    "flat_ner": false
}
```

The server must respond with the chat-completion JSON shape, where `message.content` is a JSON string of the form `{"entities": [...]}`:

```json
{
    "id": "chatcmpl-...",
    "object": "chat.completion",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "{\"entities\": [{\"text\": \"Alice\", \"label\": \"first_name\", \"start\": 0, \"end\": 5, \"score\": 0.94}, ...]}"
        },
        "finish_reason": "stop"
    }]
}
```

Each entity has `text`, `label`, `start`, `end`, `score`. The request fields above come from `anonymizer.engine.detection.detection_workflow._inject_detector_params` (`labels`, `threshold`, `chunk_length`, `overlap`, `flat_ner`); the response is parsed by `anonymizer.engine.detection.postprocess.parse_raw_entities`.

Long inputs are split into overlapping chunks before inference. A self-hosted server should honor `chunk_length` and `overlap` so detection matches the hosted `build.nvidia.com` path, while keeping the chat-completion adapter expected by Anonymizer.

---

## Reference implementation

A minimal FastAPI reference server at `tools/serve_gliner.py` implements the contract above. It loads `nvidia/gliner-pii`, exposes `POST /v1/chat/completions` (and `GET /v1/models`), and uses two levels of batching:

1. **Chunk batching** — long text is split into overlapping windows; all chunks are passed to one `model.inference(...)` call.
2. **Request coalescing** (optional, on by default) — concurrent HTTP requests from DataDesigner are grouped briefly, then all their chunks are inferred together.

```python title="tools/serve_gliner.py (excerpt)"
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    text = _extract_text(body.get("messages", []))
    params = DetectParams(
        labels=tuple(body.get("labels") or []),
        threshold=float(body.get("threshold", 0.3)),
        chunk_length=int(body.get("chunk_length", 384)),
        overlap=int(body.get("overlap", 128)),
        flat_ner=bool(body.get("flat_ner", False)),
        inference_batch_size=int(body.get("batch_size", 8)),
    )
    entities = await detector.detect(text, params)
    ...
```

When `flat_ner` is `false` (Anonymizer's default), the server removes nested subset spans before score-based deduplication across chunk overlaps.

| Environment variable | Default | Purpose |
|---|---|---|
| `DEVICE` | `auto` | `auto`, `cuda`, `cpu`, or `mps` (Apple Silicon GPU) |
| `GLINER_BATCH_MODE` | `true` | Coalesce concurrent HTTP requests before inference |
| `GLINER_MAX_BATCH_REQUESTS` | `32` | Max requests per coalesced batch |
| `GLINER_BATCH_WAIT_MS` | `10` | Max wait time to fill a batch (milliseconds) |

Set `GLINER_BATCH_MODE=false` to disable request coalescing; chunk batching still runs per request.

---

## Running it

### Dependencies

```bash
pip install fastapi uvicorn gliner
# or with uv
uv pip install fastapi uvicorn gliner
```

On first launch the `gliner` package will download `nvidia/gliner-pii` from HuggingFace and cache it under `~/.cache/huggingface/`. No HuggingFace token is required (public model).

### Start the server

```bash
python tools/serve_gliner.py
# INFO     Uvicorn running on http://0.0.0.0:8001

# Optional: pick device explicitly (auto prefers mps, then cuda, then cpu)
DEVICE=cuda python tools/serve_gliner.py
```

Verify the server is reachable:

```bash
curl -sf http://localhost:8001/v1/models | python -m json.tool
# {
#     "object": "list",
#     "data": [{"id": "nvidia/gliner-pii", "object": "model"}]
# }
```

Run a real detection call — this is exactly what Anonymizer sends at the `entity_detector` role:

```bash
curl -s http://localhost:8001/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
        "model": "nvidia/gliner-pii",
        "messages": [{"role": "user", "content": "Hi support, I can'\''t log in! My account username is '\''johndoe88'\''. Every time I try, it says '\''invalid credentials'\''. Please reset my password. You can reach me at (555) 123-4567 or johnd@example.com"}],
        "labels": ["user_name", "phone_number", "email"],
        "threshold": 0.3
    }' | jq -r '.choices[0].message.content' | jq
```

The first `jq` unwraps `choices[0].message.content` (an escaped JSON string); the second pretty-prints the decoded payload. Expected output:

```json
{
  "entities": [
    { "text": "johndoe88",          "label": "user_name",    "start":  52, "end":  61, "score": 0.95 },
    { "text": "(555) 123-4567",     "label": "phone_number", "start": 159, "end": 173, "score": 1.00 },
    { "text": "johnd@example.com",  "label": "email",        "start": 177, "end": 194, "score": 1.00 }
  ]
}
```

An empty `"entities": []` means either no `labels` in the request matched real PII in the text, or the `threshold` is too high.

---

## Pointing Anonymizer at the local server

Add a provider and detector alias that route the `entity_detector` role to your local instance:

```yaml
providers:
  - name: local-gliner
    endpoint: http://localhost:8001/v1
    provider_type: openai
    api_key: EMPTY  # ignored; the reference server does not check auth

model_configs:
  - alias: gliner-pii-detector
    model: nvidia/gliner-pii
    provider: local-gliner
    skip_health_check: true   # the default health check sends no `labels`, which GLiNER can't handle
    inference_parameters:
      max_parallel_requests: 8   # send concurrent rows; the reference server batches them
```

Set `skip_health_check: true`: Anonymizer's default probe sends `prompt="Hello!"` with no `labels` field, which is not a valid GLiNER request.

---

## Performance notes

- **Batch mode**: The reference server coalesces concurrent detector requests by default. Pair it with a higher `max_parallel_requests` on the `gliner-pii-detector` alias (see YAML above) so DataDesigner sends multiple rows at once and the server fills GPU batches efficiently.
- On CPU, detection of a ~1000-character note with ~30 candidate labels takes **5–20 ms** per request on a modern x86 core. For typical Anonymizer workflows this is a rounding error compared to the LLM roles that follow, and keeping GLiNER on CPU frees GPU memory for the LLM.
- On GPU the same request drops to roughly **1–3 ms** — worth it when you're processing tens of thousands of documents in a batch workflow, or when the host has spare GPU memory next to the LLM.
- Choose device with the `DEVICE` environment variable (`auto`, `cuda`, `mps`, `cpu`). `auto` prefers Apple Silicon GPU (MPS), then NVIDIA CUDA, then CPU.
- The default GLiNER threshold is `0.3`. Lower values detect more spans (higher recall, more false positives); higher values improve precision but miss edge cases. Tune via `Detect(gliner_threshold=...)`.
- Each request loads the FULL list of candidate labels passed from `Detect.entity_labels`. If you only need a subset (e.g. a clinical-only deployment), narrowing that list materially speeds up detection.
