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

Anonymizer does not perform a separate model-discovery probe for externally managed providers. The
configured service is responsible for serving the requested detector model; connection and response
errors surface through the normal Data Designer model request. The notebook-managed runtime is
different: its owner uses authenticated `GET /v1/models` readiness checks to verify the exact model
ID and pinned revision before returning an `Anonymizer`.

The detection workflow calls `POST /v1/chat/completions` and passes detector-specific fields alongside
the normal OpenAI-compatible request:

```json
{
  "model": "fastino/gliner2-privacy-filter-PII-multi",
  "messages": [{"role": "user", "content": "<input text>"}],
  "labels": ["first_name", "last_name", "email"],
  "threshold": 0.3,
  "chunk_length": 128,
  "overlap": 32,
  "flat_ner": false
}
```

!!! note "Chunk defaults and units"

    The bundled GLiNER2 server defaults to **128 native word tokens per chunk**
    with **32 tokens of overlap**. These tokens include punctuation; they are not
    characters or encoder subword tokens.

Both the notebook helper and plain `Anonymizer()` use these defaults with the
bundled server. For a custom service, check that service's units and defaults.
Chunk values set in the detector model's `inference_parameters.extra_body` pass
through unchanged; omitted fields use the server's defaults.

The native splitter preserves source offsets and avoids cutting words. Unspaced
Chinese or Japanese text can form a single long token, so word-window size does
not guarantee a subword-token limit; evaluate chunk sizes for your language.
Returned spans must match the original source. The adapter removes GLiNER2's
synthetic terminal period where needed; unexplained text/offset mismatches remain
errors.

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

A production deployment should provide the chat-completion contract through a separately managed,
authenticated service and configure its provider endpoint in `providers.yaml`. The native PyTorch
server bundled with Anonymizer is intended only for notebooks and development. A vLLM-based
production serving path is planned separately; this release does not depend on it.

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

Notebook CI runs on a CPU runner and constrains its isolated runtime to the CPU
build of PyTorch through `.github/constraints/notebook-ci.txt`. It caches only
Hugging Face model files; platform-specific runtime packages are rebuilt per job
and reused across tutorials within that job. These CI installation settings do
not change the local helper's automatic device selection or GPU support.

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

Plain `Anonymizer()` uses this server's defaults without a manual override. To
request explicit chunk settings for a custom endpoint, add them to the detector
entry in your complete model configuration:

```yaml
inference_parameters:
  extra_body:
    chunk_length: 128
    overlap: 32
```

The notebook helper supplies these settings automatically. The threshold still
comes from `Detect(gliner_threshold=...)`.

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
