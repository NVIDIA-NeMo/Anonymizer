<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Self-hosting GLiNER2

Anonymizer uses [`fastino/gliner2-privacy-filter-PII-multi`](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi)
for first-pass entity detection. The public `create_anonymizer()` factory supports two lifecycle
models. `NativeGliner` starts and reuses the library-managed GLiNER2 runtime. `GlinerEndpoint`
connects to a service that you already manage. Plain `Anonymizer()` remains unchanged and uses the
bundled model and provider files.

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
  "chunk_length": 384,
  "overlap": 128,
  "flat_ner": false
}
```

The native server and vLLM Factory adapter implement this HTTP contract, including `overlap` and
`flat_ner`. The contract does not promise identical predictions across runtimes. Model versions,
inference engines, and overlap resolution can produce different entity lists.

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

For Linux/NVIDIA GPU production deployments, use the managed vLLM Factory path described in
[Run local inference services](inference-services.md). Other production deployments should provide
the chat-completion contract through a separately managed, authenticated service and configure its
provider endpoint in `providers.yaml`. The native PyTorch server bundled with Anonymizer is intended
only for notebooks and development.

## Native Runtime

Install the notebook extra, then request the library-managed runtime:

```bash
pip install "nemo-anonymizer[notebooks]"
```

```python
from anonymizer import NativeGliner, create_anonymizer

anonymizer = create_anonymizer(gliner=NativeGliner(device="auto"))
```

`device="auto"` selects CUDA, MPS, or CPU. You can also pass `"cuda"`, `"mps"`, or `"cpu"`.
The native runtime is a process-global singleton and stops at interpreter exit.

## Notebook Compatibility Wrapper

```python
from anonymizer.notebooks import create_anonymizer, stop_local_runtime

anonymizer = create_anonymizer()

# Run notebook work here.

stop_local_runtime()
```

The notebook helper wraps `create_anonymizer(gliner=NativeGliner(...))` and preserves its original
`gliner_device` argument. It:

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

## Connect to a Managed Endpoint

Start the service through its own deployment system, then pass its connection values to
`GlinerEndpoint`:

```bash
export GLINER2_API_KEY="your-service-key"
```

```python
from anonymizer import GlinerEndpoint, create_anonymizer

anonymizer = create_anonymizer(
    gliner=GlinerEndpoint(
        url="https://gliner2.internal.example/v1",
        model="fastino/gliner2-privacy-filter-PII-multi",
        api_key_env="GLINER2_API_KEY",
    )
)
```

The factory adds one detector model and provider, selects that detector, and preserves all other
model configs, providers, and selected roles. It sets the detector's `skip_health_check` flag because
a generic text-generation probe lacks the required `labels` field. Reserved factory alias and
provider names cause a configuration error instead of replacing caller entries.

`GlinerEndpoint` only creates the Anonymizer client. It does not launch, stop, probe, or otherwise own
the service. Keep service shutdown in the deployment system that started it. The installed-library
path does not import vLLM, GLiNER2, Torch, FastAPI, or source-tree tools.

Tune recall and precision with `Detect(gliner_threshold=...)`. The default threshold is `0.3`; lower
values increase recall, while higher values generally increase precision.
