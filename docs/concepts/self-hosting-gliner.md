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
    "threshold": 0.3
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

Each entity has `text`, `label`, `start`, `end`, `score`. The `labels` and `threshold` fields on the request come from `anonymizer.engine.detection.detection_workflow._inject_detector_params`; the response is parsed by `anonymizer.engine.detection.postprocess.parse_raw_entities`.

---

## Reference implementation

A minimal 55-line FastAPI reference server that implements the contract above is included in the repo at `tools/serve_gliner.py`. It uses the `gliner` Python package directly, loads `nvidia/gliner-pii`, and exposes the required endpoints on `0.0.0.0:8001`.

```python title="tools/serve_gliner.py (excerpt)"
from gliner import GLiNER
from fastapi import FastAPI, Request
import torch, uvicorn, json, time, uuid

MODEL_NAME = "nvidia/gliner-pii"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
app = FastAPI()
model = GLiNER.from_pretrained(MODEL_NAME).to(DEVICE)

@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    text = body["messages"][-1]["content"]  # also handles multi-part
    labels = body.get("labels") or []
    threshold = float(body.get("threshold", 0.3))
    raw = model.predict_entities(text, labels, threshold=threshold) if (labels and text) else []
    entities = [{"text": e["text"], "label": e["label"], "start": e["start"],
                 "end": e["end"], "score": e["score"]} for e in raw]
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": json.dumps({"entities": entities})}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

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
```

Set `skip_health_check: true`: Anonymizer's default probe sends `prompt="Hello!"` with no `labels` field, which is not a valid GLiNER request.

---

## Performance notes

- On CPU, detection of a ~1000-character note with ~30 candidate labels takes **5–20 ms** per request on a modern x86 core. For typical Anonymizer workflows this is a rounding error compared to the LLM roles that follow, and keeping GLiNER on CPU frees GPU memory for the LLM.
- On GPU the same request drops to roughly **1–3 ms** — worth it when you're processing tens of thousands of documents in a batch workflow, or when the host has spare GPU memory next to the LLM.
- Choose device by passing `device="cuda"` / `"cpu"` to `GLiNER.from_pretrained(...)` or calling `.to("cuda")` after load. The reference server auto-selects GPU when `torch.cuda.is_available()` returns `True`.
- The default GLiNER threshold is `0.3`. Lower values detect more spans (higher recall, more false positives); higher values improve precision but miss edge cases. Tune via `Detect(gliner_threshold=...)`.
- Each request loads the FULL list of candidate labels passed from `Detect.entity_labels`. If you only need a subset (e.g. a clinical-only deployment), narrowing that list materially speeds up detection.
