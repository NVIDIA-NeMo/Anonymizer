<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Run Local Inference Services

Anonymizer's source tree includes an inference service compiler for development
and controlled internal deployments. It compiles typed JSON intent into an
immutable plan before it starts a process or container. Launch, probe, inspect,
and cancel operations return versioned JSON receipts with the external identity
and known effects of the operation.

The tool is source-owned under `tools/` and is not included in the
`nemo-anonymizer` wheel. It currently supports:

- entity detection with the native NVIDIA GLiNER or GLiNER2 runtime;
- entity detection or generation through a vLLM-compatible model;
- managed local processes and managed local Docker containers; and
- direct HTTP access to the resulting OpenAI-compatible endpoint.

It does not attach to an existing endpoint or manage remote compute. Configure
an existing provider URL directly in Anonymizer when another system owns that
service.

## Lifecycle

The CLI keeps compilation separate from runtime effects:

```text
intent.json -> compile -> plan.json -> launch -> launch.json
                                      inspect <-+
                                      cancel  <-+
```

Plans use the `inference-service.run-plan/v1` schema. Launch, capability-probe,
status, and cancellation receipts have their own v1 schemas. A SHA-256 digest
binds each plan to its exact intent, command, endpoint contract, compatibility
evidence, and source revision. Runtime commands reject a changed plan before
performing effects.

## Native GLiNER

Create `gliner-intent.json`:

```json
{
  "schema_version": "inference-service.intent/v1",
  "task": {
    "kind": "entity-detection",
    "dynamic_labels": true,
    "offsets": true,
    "scores": true
  },
  "model": {
    "kind": "hugging-face",
    "model_id": "nvidia/gliner-pii"
  },
  "engine": {
    "kind": "native-gliner",
    "family": "nvidia-gliner",
    "device": "auto"
  },
  "placement": {
    "kind": "local-process",
    "host": "127.0.0.1",
    "port": 8001
  },
  "access": {"kind": "direct"},
  "lifecycle": {
    "kind": "managed",
    "startup_timeout_seconds": 120,
    "shutdown_timeout_seconds": 30
  }
}
```

Compile and launch it from the repository root. Replace the example source
revision with the revision of your checkout:

```bash
uv run tools/inference_service.py compile \
  --intent gliner-intent.json \
  --source-revision 3f68c145 \
  --output gliner-plan.json

uv run tools/inference_service.py launch \
  --plan gliner-plan.json \
  --output gliner-launch.json
```

The launch returns only after `/v1/models` and an entity-detection contract
probe succeed. The receipt records the process ID plus its Linux start marker
when available, which lets a later invocation guard against PID reuse.

To use GLiNER2, change the engine family to `gliner2` and select a compatible
checkpoint such as `fastino/gliner2-privacy-filter-PII-multi` in the model
field. `nvidia-gliner` and `nvidia/gliner-pii` remain the defaults.

## Local vLLM Process

On a Linux GPU host, install the optional source-tree dependency group:

```bash
uv sync --group dev --group local-models
nvidia-smi
```

Create an intent with a generation task, vLLM engine, and local-process
placement:

```json
{
  "schema_version": "inference-service.intent/v1",
  "task": {"kind": "generation", "chat": true},
  "model": {
    "kind": "hugging-face",
    "model_id": "openai/gpt-oss-20b",
    "revision": "PIN_A_MODEL_REVISION_HERE"
  },
  "engine": {
    "kind": "vllm",
    "executable": ".venv/bin/vllm",
    "served_model_name": "anonymizer-local",
    "gpu_memory_utilization": 0.85,
    "max_model_len": 8192
  },
  "placement": {
    "kind": "local-process",
    "host": "127.0.0.1",
    "port": 8000
  },
  "access": {"kind": "direct"},
  "lifecycle": {
    "kind": "managed",
    "startup_timeout_seconds": 600,
    "shutdown_timeout_seconds": 30
  }
}
```

Use the same `compile` and `launch` commands shown for GLiNER. The model ID may
cause vLLM to download weights. List existing Hugging Face cache snapshots
without downloading anything:

```bash
uv run tools/inference_service.py models --output cached-models.json
```

Add a LoRA artifact to the model when needed:

```json
"adapter": {
  "path": "/models/privacy-adapter",
  "name": "privacy"
}
```

The compiler renders the corresponding vLLM `--lora-modules` arguments.

## Docker vLLM

Change the placement to Docker to use vLLM's official OpenAI-compatible image:

```json
"placement": {
  "kind": "docker",
  "host": "127.0.0.1",
  "port": 8000,
  "image": "vllm/vllm-openai:v0.20.0",
  "runtime": "docker",
  "gpus": "all",
  "hugging_face_cache": "/home/user/.cache/huggingface"
}
```

The plan records the exact image and complete `docker run` argv. Launch receipts
record the container ID, and `inspect` and `cancel` reconnect through that ID.
Pin an image version appropriate for the host's driver and CUDA compatibility.

## Secrets

Set `api_key_env` on the vLLM engine to reference a named environment variable:

```json
"api_key_env": "LOCAL_VLLM_API_KEY"
```

Plans serialize only that source name and render the service environment value
as `<secret:LOCAL_VLLM_API_KEY>`. `launch` maps it to `VLLM_API_KEY` without
putting the value in process arguments or Docker command metadata. Docker uses
`--env VLLM_API_KEY` to inherit the resolved value. Launch fails before starting
the process or container when the source variable is absent. Do not commit local
secret files.

## Inspect, Probe, and Cancel

Use the plan to collect a fresh capability receipt, or the launch receipt to
inspect and stop the managed service:

```bash
uv run tools/inference_service.py probe \
  --plan gliner-plan.json \
  --output gliner-probe.json

uv run tools/inference_service.py inspect \
  --receipt gliner-launch.json \
  --output gliner-status.json

uv run tools/inference_service.py cancel \
  --receipt gliner-launch.json \
  --output gliner-cancellation.json
```

Local-process logs are written under `.inference-service-runs/` by default.
Use `launch --log-directory PATH` to select another location.

## Connect Anonymizer

Both native GLiNER and vLLM expose OpenAI-compatible URLs. Add the compiled
endpoint to a custom provider file:

```yaml title="providers.yaml"
providers:
  - name: local-inference
    endpoint: http://127.0.0.1:8000/v1
    provider_type: openai
    api_key: EMPTY
```

Set the selected model configuration's `provider` to `local-inference` and its
`model` to the served model name. Custom `model_configs` replaces Anonymizer's
entire bundled model pool, so retain every alias required by the roles you use.
See [Custom models](models.md#custom-models) for the role map and validation
command.

Compilation proves only static compatibility. The launch probe proves the
observed endpoint shape. Neither proves that a model meets your privacy or
utility requirements; run Anonymizer preview and evaluation before trusting a
new model or engine combination.
