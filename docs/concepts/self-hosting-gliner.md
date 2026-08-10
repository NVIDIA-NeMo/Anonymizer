<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Self-hosting GLiNER

By default, Anonymizer's entity detection stage calls the hosted `nvidia/gliner-pii` model on `build.nvidia.com`. For PHI-sensitive workloads that cannot leave the host, or latency-critical setups, you can serve GLiNER locally instead.

The default NVIDIA GLiNER model is small (~500 MB) and runs comfortably on CPU — making it a good fit to run alongside a local LLM without competing for GPU memory. It also runs on GPU if one is available, which cuts detection latency on long documents. The optional GLiNER2 PII model is also fully local and supports GPU or CPU inference.

The characterized native server lives inside the source-tree [inference service compiler](inference-services.md). It is **not** installed with `pip install nemo-anonymizer`; compile and launch it from a source checkout.

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

The native GLiNER runtime compiled by `tools/inference_service.py` implements the contract above. Its internal server module defaults to `nvidia/gliner-pii`, also supports the local PII-capable `fastino/gliner2-privacy-filter-PII-multi` model, exposes `POST /v1/chat/completions` (and `GET /v1/models`), and uses two levels of batching:

1. **Chunk batching** — long text is split into overlapping windows; all chunks are passed to one runtime batch call.
2. **Request coalescing** (optional, on by default) — concurrent HTTP requests from DataDesigner are grouped briefly, then all their chunks are inferred together.

```python title="tools/inference_service_compiler/native_gliner.py (excerpt)"
@api.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = require_mapping(await request.json(), "request")
    params = parse_detect_params(body)
    text = extract_text(body.get("messages", []))
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

!!! note "Source checkout only"

    `tools/inference_service.py` ships in the [Anonymizer GitHub repository](https://github.com/NVIDIA-NeMo/Anonymizer), not in the `nemo-anonymizer` wheel. Clone the repository and run the compiler from its root.

### Dependencies

The managed native server is a [PEP 723](https://peps.python.org/pep-0723/) uv script and declares its own Python 3.13+ dependencies. Install [uv](https://docs.astral.sh/uv/); launch resolves the isolated environment for the selected local runtime. The first environment setup can be large because the runtime packages include Torch and its platform dependencies. No package installation in the Anonymizer environment is required.

On first launch, the selected public checkpoint is downloaded from Hugging Face and cached under `~/.cache/huggingface/`. No Hugging Face token is required. Package and checkpoint setup use the network; inference stays local and the server does not call a remote inference service after setup.

### Start the server

Create a native GLiNER intent, compile it, and launch the resulting plan as
shown in [Run local inference services](inference-services.md#native-gliner).
The intent keeps the model checkpoint, engine family, device, placement,
access, and managed lifecycle separate. `nvidia-gliner` is the default engine
family and `nvidia/gliner-pii` is the default model; GLiNER2 uses the
`fastino/gliner2-privacy-filter-PII-multi` checkpoint.

Launch writes a versioned receipt only after the model-list and detection
contract probes pass. Use that receipt with the compiler's `inspect` and
`cancel` commands instead of supervising the internal server module directly.

The model families do not use identical label vocabularies. The request example below targets the default NVIDIA model and uses `user_name`; the default GLiNER2 PII checkpoint uses `username` for that category.

The reference server has **no authentication**. The default bind address is `127.0.0.1` so detection traffic stays on localhost. Use `--host 0.0.0.0` only when Anonymizer runs on another host in a trusted environment, ideally behind authentication and TLS termination.

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

Pass separate `model_providers` and `model_configs` files to `Anonymizer`. **`model_configs` replaces the entire model pool** — it is not merged with defaults. Copy the bundled [`models.yaml`](https://github.com/NVIDIA-NeMo/Anonymizer/blob/main/src/anonymizer/config/default_model_configs/models.yaml), change only the `gliner-pii-detector` entry's `provider`, and keep the other default aliases (`gpt-oss-120b`, `nemotron-30b-thinking`). Default role→alias mappings still apply unless you override `selected_models` (see [Custom models](models.md#custom-models)).

Custom `model_providers` also replaces the provider list, so include both your local GLiNER endpoint and the `nvidia` provider used by the LLM roles:

```yaml title="providers.yaml"
providers:
  - name: local-gliner
    endpoint: http://localhost:8001/v1
    provider_type: openai
    api_key: EMPTY  # ignored; the reference server does not check auth

  - name: nvidia
    endpoint: https://integrate.api.nvidia.com/v1
    provider_type: openai
    api_key: NVIDIA_API_KEY
```

```bash
export NVIDIA_API_KEY="your-nvidia-api-key"
```

```yaml title="models.yaml"
model_configs:
  - alias: gliner-pii-detector
    model: nvidia/gliner-pii
    provider: local-gliner
    inference_parameters:
      max_parallel_requests: 8   # send concurrent rows; the reference server batches them
      timeout: 120

  - alias: gpt-oss-120b
    model: openai/gpt-oss-120b
    provider: nvidia
    inference_parameters:
      max_parallel_requests: 16
      max_tokens: 16384
      temperature: 0.3
      top_p: 0.95
      timeout: 300

  - alias: nemotron-30b-thinking
    model: nvidia/nemotron-3-nano-30b-a3b
    provider: nvidia
    inference_parameters:
      max_parallel_requests: 16
      max_tokens: 8192
      temperature: 0.4
      top_p: 1.0
      timeout: 300
```

```python
from anonymizer import Anonymizer

anonymizer = Anonymizer(
    model_providers="providers.yaml",
    model_configs="models.yaml",
)
```

---

## Performance notes

- **Batch mode**: The reference server coalesces concurrent detector requests by default. Pair it with a higher `max_parallel_requests` on the `gliner-pii-detector` alias (see YAML above) so DataDesigner sends multiple rows at once and the server fills GPU batches efficiently.
- On CPU, detection of a ~1000-character note with ~30 candidate labels takes **5–20 ms** per request on a modern x86 core. For typical Anonymizer workflows this is a rounding error compared to the LLM roles that follow, and keeping GLiNER on CPU frees GPU memory for the LLM.
- On GPU the same request drops to roughly **1–3 ms** — worth it when you're processing tens of thousands of documents in a batch workflow, or when the host has spare GPU memory next to the LLM.
- Choose device with the `DEVICE` environment variable (`auto`, `cuda`, `mps`, `cpu`). `auto` prefers Apple Silicon GPU (MPS), then NVIDIA CUDA, then CPU.
- The default GLiNER threshold is `0.3`. Lower values detect more spans (higher recall, more false positives); higher values improve precision but miss edge cases. Tune via `Detect(gliner_threshold=...)`.
- Each request loads the FULL list of candidate labels passed from `Detect.entity_labels`. If you only need a subset (e.g. a clinical-only deployment), narrowing that list materially speeds up detection.
