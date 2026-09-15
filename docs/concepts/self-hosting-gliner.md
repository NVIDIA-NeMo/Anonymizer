<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Self-hosting GLiNER2

Anonymizer uses [`fastino/gliner2-privacy-filter-PII-multi`](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi)
for first-pass entity detection. The bundled model configuration sends detector requests to an
OpenAI-compatible endpoint at `http://127.0.0.1:8001/v1`; it does not start a server automatically.

This keeps plain `Anonymizer()` suitable for applications that manage their own services. Start a
compatible GLiNER2 endpoint before calling `run()`, or use the notebook helper below for an ephemeral
in-kernel development server.

## Server contract

The detection workflow calls `POST /v1/chat/completions` and passes detector-specific fields alongside
the normal OpenAI-compatible request:

```json
{
  "model": "fastino/gliner2-privacy-filter-PII-multi",
  "messages": [{"role": "user", "content": "<input text>"}],
  "labels": ["first_name", "last_name", "email"],
  "threshold": 0.3,
  "chunk_length": 384,
  "overlap": 128,
  "flat_ner": false
}
```

The response must use the chat-completion shape. `message.content` is a JSON string containing an
`entities` list. Each entity has `text`, `label`, `start`, `end`, and `score` fields:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "{\"entities\": [{\"text\": \"Alice\", \"label\": \"first_name\", \"start\": 0, \"end\": 5, \"score\": 0.94}]}"
    },
    "finish_reason": "stop"
  }]
}
```

A production deployment should provide this contract through a separately managed, authenticated
service and configure its provider endpoint in `providers.yaml`. The native PyTorch server bundled
with Anonymizer is intended only for notebooks and development. A vLLM-based production serving path
is planned separately; this release does not depend on it.

## Local notebook runtime

Install the notebook extra and create the client through `create_anonymizer()`:

```bash
pip install "nemo-anonymizer[notebooks]"
```

```python
from anonymizer.notebooks import create_anonymizer, stop_local_runtime

anonymizer = create_anonymizer()

# Run notebook work here.

stop_local_runtime()
```

The helper:

- creates a cached, dependency-isolated server environment containing `gliner2[local]` on first use;
- downloads the exact Hugging Face revision
  `59894c087cb2923b01f337d4ee72f6ff84d5bdd6` of
  `fastino/gliner2-privacy-filter-PII-multi`;
- starts an authenticated server on an ephemeral loopback port;
- selects CUDA, MPS, or CPU automatically;
- injects only the detector provider and model selection, leaving configured LLM roles unchanged;
- reuses the process for compatible calls and stops it at interpreter exit.

The isolated server environment is necessary because GLiNER2 2.0's local inference stack and the
current Data Designer release require incompatible `huggingface-hub` major versions. The helper hides
that packaging detail from the notebook kernel. It also creates a local adapter copy of the pinned
snapshot that translates the checkpoint's Transformers v5 tokenizer field name to the Transformers
v4 name required by GLiNER2 2.0; model weights and source cache files remain unchanged. The revision
pin applies only to this packaged notebook runtime. Production services control and pin their own
model artifact independently.

To request a device explicitly:

```python
anonymizer = create_anonymizer(gliner_device="cuda")  # also: "mps", "cpu", or "auto"
```

Calling the helper again with an incompatible device or configuration replaces the previous local
runtime. The most recent request wins.

!!! warning "Local detection is not an all-local pipeline"

    Only GLiNER2 detection runs in the notebook process. Validation, augmentation, replacement,
    rewriting, repair, and evaluation still use the configured LLM providers and may send them
    original or tagged input text.

## Running the development server directly

A compatibility entry point remains available from a source checkout:

```bash
uv sync --extra notebooks
uv run python tools/serve_gliner.py --host 127.0.0.1 --port 8001
```

The wrapper prepares the same isolated dependency set used by the notebook helper, then starts the
packaged server. It uses the pinned model but does not add production hardening. Keep it bound to
loopback unless you provide an authenticated network boundary.

## Configuring a managed endpoint

Custom provider and model files replace the bundled lists, so include the detector plus every LLM
provider and model alias your selected workflow uses.

```yaml title="providers.yaml"
providers:
  - name: my-gliner2-service
    endpoint: https://gliner2.internal.example/v1
    provider_type: openai
    api_key: GLINER2_API_KEY

  - name: nvidia
    endpoint: https://integrate.api.nvidia.com/v1
    provider_type: openai
    api_key: NVIDIA_API_KEY
```

```yaml title="models.yaml"
model_configs:
  - alias: gliner-pii-detector
    model: fastino/gliner2-privacy-filter-PII-multi
    provider: my-gliner2-service
    skip_health_check: true
    inference_parameters:
      max_parallel_requests: 8
      timeout: 120

  # Include the LLM aliases used by your selected detection, replacement,
  # rewrite, and evaluation roles here too.
```

```python
from anonymizer import Anonymizer

anonymizer = Anonymizer(
    model_providers="providers.yaml",
    model_configs="models.yaml",
)
```

Set `skip_health_check: true` for the detector alias because a generic text-generation health probe
does not contain the required GLiNER2 `labels` field. Anonymizer performs a detector-specific endpoint
check immediately before the default detection workflow runs.

Tune recall and precision with `Detect(gliner_threshold=...)`. The default threshold is `0.3`; lower
values increase recall, while higher values generally increase precision.
