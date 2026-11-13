<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Self-hosting GLiNER2

Anonymizer uses [`fastino/gliner2-privacy-filter-PII-multi`](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi)
for first-pass entity detection. Use `native_runtime()` to manage the lifetime of a local GLiNER2 runtime, or use
`GlinerEndpoint` with `create_anonymizer()` to connect to a service that you already manage. Plain `Anonymizer()` remains unchanged and uses the
bundled model and provider files.

## Server contract

Anonymizer does not perform a separate model-discovery probe for externally managed providers. The
configured service is responsible for serving the requested detector model; connection and response
errors surface through the normal Data Designer model request. The library-managed native runtime is
different: its owner uses authenticated `GET /v1/models` readiness checks to verify the exact model
ID and pinned revision before returning an `Anonymizer`.

The detection workflow calls `POST /v1/chat/completions` and passes detector-specific fields alongside
the normal OpenAI-compatible request. Anonymizer always sends `labels`, `threshold`, and `flat_ner`.
`chunk_length` and `overlap` are optional: they are sent only when the detector model's
`inference_parameters.extra_body` sets them, so servers must apply their own defaults when they are absent:

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

The native server implements this HTTP contract, including `overlap` and `flat_ner`. The contract
does not promise identical predictions across runtimes. Model versions,
inference engines, and overlap resolution can produce different entity lists.

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
Returned entities use character offsets into the original source text, and each entity's text must match that span.

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

On Linux/NVIDIA GPU hosts, the managed tool described in
[Deploy local models](inference-services.md) launches this native server with launch
receipts, ownership checks, and bounded cleanup. Other production deployments should provide the
chat-completion contract through a separately managed, authenticated service and configure its
provider endpoint in `providers.yaml`.

## Native Runtime

Install the notebook extra, then request the library-managed runtime:

```bash
pip install "nemo-anonymizer[notebooks]"
```

```python
from anonymizer import native_runtime

with native_runtime(device="auto") as session:
    anonymizer = session.create_anonymizer()
    # Run preview or anonymization here.
```

`device="auto"` selects CUDA, MPS, or CPU. You can also pass `"cuda"`, `"mps"`, or `"cpu"`.
Use the client while the context is open; a client backed by a context-owned process is no longer valid after it closes. `session.endpoint` exposes the `GlinerEndpoint` connection. Nested contexts share the runtime until the last context exits. A context can borrow an existing singleton without stopping it on exit when its device, dtype, and attention all match; any mismatch raises an error, and explicit shutdown is unavailable while a context is active.

If the process exits while a context is active, close the active contexts before
acquiring a new runtime. Further acquisitions fail instead of silently replacing
the endpoint used by existing sessions.

`create_anonymizer(gliner=NativeGliner(...))` uses a process-global singleton that stops at interpreter exit.

## Notebook Compatibility Wrapper

```python
from anonymizer.notebooks import create_anonymizer, stop_local_runtime

anonymizer = create_anonymizer()

# Run notebook work here.

stop_local_runtime()
```

The notebook helper wraps `create_anonymizer(gliner=NativeGliner(...))` and preserves its original
`gliner_device` argument. It:

- creates a cached, dependency-isolated server environment containing `gliner2[local,disentangled-flash]` on first use;
- downloads the exact Hugging Face revision
  `59894c087cb2923b01f337d4ee72f6ff84d5bdd6` of
  `fastino/gliner2-privacy-filter-PII-multi`;
- starts an authenticated server on an ephemeral loopback port;
- selects CUDA, MPS, or CPU automatically, and runs float32 with standard attention everywhere (pass
  `dtype="bfloat16"` to `NativeGliner` or `native_runtime` for higher batch throughput at slightly different
  scores, and `attention="disentangled_flash"` to opt in to DisentangledFlash on CUDA);
- injects only the detector provider and model selection, leaving configured LLM roles unchanged;
- reuses the process for compatible calls and stops it at interpreter exit.

The isolated server environment is necessary because GLiNER2's local inference stack and the
current Data Designer release require incompatible `huggingface-hub` major versions. The helper hides
that packaging detail from the notebook kernel. The revision pin applies to the native server (the notebook helper, `native_runtime`, and the
`gliner2-native` profile). Production services control and pin their own model artifact independently.

The server reads these environment variables at startup:

| Variable | Values | Default |
|---|---|---|
| `ANONYMIZER_LOCAL_GLINER2_DEVICE` | `auto`, `cuda`, `mps`, `cpu` | `auto` |
| `ANONYMIZER_LOCAL_GLINER2_DTYPE` | `auto`, `float32`, `bfloat16` | `auto`: `float32` on every device; `bfloat16` is opt-in for higher batch throughput at slightly different scores |
| `ANONYMIZER_LOCAL_GLINER2_ATTENTION` | `auto`, `standard`, `disentangled_flash` | `auto`: `standard` on every device; `disentangled_flash` is opt-in |

A `disentangled_flash` request that cannot load stops startup with an error. In gliner2 2.1.0 the kernel
prepares position plans for every distinct input length and keeps them, so latency and GPU memory grow
with the number of distinct lengths seen (the 65th distinct length took 3.7 s and memory grew about 2 GB
over 65 lengths in our measurements). Use it only for fixed-length batch workloads; float32 output is
identical to standard attention. Requests use an inference batch size of 32 unless they pass `batch_size`. The
startup log and `GET /v1/models` report the device, dtype, and attention in use.

A library-managed runtime takes its precision and attention from the `dtype` and `attention` arguments of
`NativeGliner` and `native_runtime`, which override `ANONYMIZER_LOCAL_GLINER2_DTYPE` and
`ANONYMIZER_LOCAL_GLINER2_ATTENTION`; with the default `"auto"` the variable still applies.
Asking for a different device, dtype or attention than a running runtime was started with replaces it (or is an error
inside an active `native_runtime` context).

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

A development entry point is available from a source checkout:

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
