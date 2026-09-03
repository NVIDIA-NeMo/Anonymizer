<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Self-hosting GLiNER

By default, Anonymizer's entity detection stage calls the hosted `nvidia/gliner-pii` model on `build.nvidia.com`. For PHI-sensitive workloads that cannot leave the host, or latency-critical setups, you can serve GLiNER locally instead.

The default NVIDIA GLiNER model is small enough to share a GPU with a local
LLM. The optional GLiNER2 PII model is also fully local. The pinned reference
profiles serve both models through vLLM 0.27.1 and the external
[vLLM Factory](https://github.com/latenceainew/vllm-factory) project.

The characterized services live inside the source-tree
[local-model deployment tool](inference-services.md). They are **not** installed
with `pip install nemo-anonymizer`; compile and launch them from a source
checkout.

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

The pinned profiles compile a vLLM Factory integration. The external project
prepares each GLiNER checkpoint for vLLM, registers the model implementation,
preprocesses requests through an IOProcessor, schedules pooling inference, and
decodes the output. Its native endpoint is `POST /pooling`.

Anonymizer adds a thin middleware function inside the same vLLM process. It
translates `POST /v1/chat/completions` into in-process pooling calls and restores
the detector response shape. The adapter does not load model weights or run
inference itself. It handles two wire-level responsibilities:

1. **Chunk submission**: long text is split into overlapping character windows,
   then each window enters vLLM's scheduler.
2. **Response normalization**: GLiNER and GLiNER2 results become one list of
   entities with document offsets and overlap deduplication.

```python title="tools/inference_service_compiler/vllm_factory_adapter.py (excerpt)"
results = await invoke_pooling_chunks(
    handler=handler,
    detection=detection,
    chunks=chunks,
    ...
)
entities = merge_entities(plugin=plugin, chunks=chunks, results=results, ...)
```

The adapter rejects a labeled request that would produce more than 256 chunks
before it materializes chunk text or calls vLLM. The service runs at most eight
pooling calls concurrently across all accepted requests; the remaining
admitted chunks wait for capacity.

When `flat_ner` is `false` (Anonymizer's default), the adapter removes nested
subset spans before score-based deduplication across chunk overlaps. A request
without `labels` returns an empty entity list so DataDesigner's generic model
health check can validate the endpoint without running meaningless inference.

The pinned vLLM Factory profiles are the only supported local detector runtime.
Native Transformers GLiNER fallback serving is intentionally not included.

---

## Deploying it

!!! note "Source checkout only"

    `tools/inference_service.py` ships in the [Anonymizer GitHub repository](https://github.com/NVIDIA-NeMo/Anonymizer), not in the `nemo-anonymizer` wheel. Clone the repository and run the compiler from its root.

### Dependencies

Install [uv](https://docs.astral.sh/uv/) and sync the local model group on a
Linux GPU host:

```bash
uv sync --python 3.12 --group dev --group local-models
python -m vllm_factory.compat.doctor
```

The group pins `vllm==0.27.1` and vLLM Factory at the exact source revision
recorded in `pyproject.toml` and `uv.lock`. The doctor must report the general
plugins group, IOProcessor plugins group, and native IO mode. Local vLLM 0.27.1
serving requires Python 3.12 or later.

On first launch, the selected public checkpoint is downloaded from Hugging Face
and cached under `~/.cache/huggingface/`. The integration supplies the TOML
profile's immutable revision to vLLM Factory's preparation API and writes a
provenance record beside the prepared model. Package and checkpoint setup use
the network; inference stays local after setup.

### Start the server

Compile the pinned vLLM Factory GLiNER TOML profile and launch the resulting
plan as shown in [Deploy local models](inference-services.md). That guide has
complete host and GPU container workflows. NVIDIA GLiNER uses
`deberta_gliner`; GLiNER2 uses `deberta_gliner2`.

Launch writes a versioned receipt only after the new process proves launch
ownership and the model-list and detection contract probes pass. Use that
receipt with the compiler's `status` and `stop` commands instead of supervising
the internal server module directly.

The model families do not use identical label vocabularies. The request example below targets the default NVIDIA model and uses `user_name`; the default GLiNER2 PII checkpoint uses `username` for that category.

The reference profiles have **no authentication**. The default bind address is
`127.0.0.1` so detection traffic stays on localhost. Set `api_key_env` in the
`[vllm]` section or place the endpoint behind authenticated TLS before exposing
it to another host.

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
      max_parallel_requests: 8   # vLLM continuously batches concurrent detector calls
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

- **Continuous batching**: Pair vLLM Factory with a higher
  `max_parallel_requests` on the detector alias so DataDesigner supplies enough
  concurrent work for vLLM's scheduler.
- **GPU memory**: `gpu_memory_utilization` is a vLLM memory budget. Size it with
  any colocated generation service in mind.
- **Native fallback**: Use the native engine when CPU or MPS deployment matters
  more than vLLM scheduling.
- The default GLiNER threshold is `0.3`. Lower values detect more spans (higher recall, more false positives); higher values improve precision but miss edge cases. Tune via `Detect(gliner_threshold=...)`.
- Each request loads the FULL list of candidate labels passed from `Detect.entity_labels`. If you only need a subset (e.g. a clinical-only deployment), narrowing that list materially speeds up detection.
