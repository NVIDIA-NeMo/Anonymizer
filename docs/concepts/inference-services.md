<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Run Local Inference Services

Anonymizer's source tree includes an inference service compiler for development
and controlled internal deployments. It compiles a typed TOML profile into an
immutable plan before it starts a process or container. Launch, probe, inspect,
and cancel operations return versioned JSON receipts with the external identity
and known effects of the operation.

The tool is source-owned under `tools/` and is not included in the
`nemo-anonymizer` wheel. It currently supports:

- entity detection with the native NVIDIA GLiNER or GLiNER2 runtime;
- generation through a vLLM-compatible model;
- managed local processes and managed local Docker containers; and
- direct HTTP access to the resulting OpenAI-compatible endpoint.

It does not attach to an existing endpoint or manage remote compute. Configure
an existing provider URL directly in Anonymizer when another system owns that
service.

## Lifecycle

The CLI keeps compilation separate from runtime effects:

```text
profile.toml -> compile -> plan.json -> launch -> launch.json
                                        inspect <-+
                                        cancel  <-+
```

Plans use the `inference-service.run-plan/v1` schema. Launch, capability-probe,
status, and cancellation receipts have their own v1 schemas. A SHA-256 digest
binds each plan to its exact intent, command, endpoint contract, compatibility
evidence, and source revision. Runtime commands reject a changed plan before
performing effects.

## Native GLiNER

The source tree includes pinned profiles for NVIDIA GLiNER and GLiNER2 under
`tools/inference_service_profiles/`. Compile and launch one from the repository
root. Replace the example source revision with the revision of your checkout:

```bash
uv run tools/inference_service.py compile \
  --profile tools/inference_service_profiles/nvidia-gliner.toml \
  --source-revision 3f68c145 \
  --output gliner-plan.json

uv run tools/inference_service.py launch \
  --plan gliner-plan.json \
  --output gliner-launch.json
```

The launch returns only after `/v1/models` and an entity-detection contract
probe succeed. The receipt records the process ID plus its Linux start marker
when available, which lets a later invocation guard against PID reuse.

Use `tools/inference_service_profiles/gliner2.toml` for the pinned GLiNER2
checkpoint. Entity detection is intentionally native. The compiler rejects a
GLiNER profile paired with vLLM because vLLM 0.26 does not expose GLiNER's
dynamic-label, offset, and score contract.

## Local vLLM Process

On a Linux GPU host, install the optional source-tree dependency group:

```bash
uv sync --group dev --group local-models
nvidia-smi
```

The `local-models` group pins vLLM 0.26.0. The local plan starts
`tools/inference_service_compiler/vllm_server.py`, which constructs vLLM's
frontend and async engine through its Python API. It does not invoke `vllm
serve` or inherit vLLM's full CLI surface.

Compile `tools/inference_service_profiles/vllm-local.toml`, or copy it and pin
the model revision and sizing fields for your workload:

```bash
uv run tools/inference_service.py compile \
  --profile tools/inference_service_profiles/vllm-local.toml \
  --source-revision 3f68c145 \
  --output vllm-plan.json

uv run tools/inference_service.py launch \
  --plan vllm-plan.json \
  --output vllm-launch.json
```

The model ID may cause vLLM to download weights. List existing Hugging Face
cache snapshots without downloading anything:

```bash
uv run tools/inference_service.py models --output cached-models.json
```

Add a LoRA artifact to the model when needed:

```toml
[model.adapter]
path = "/models/privacy-adapter"
name = "privacy"
```

The compiler renders the corresponding vLLM `--lora-modules` arguments.

## Docker vLLM

Change the placement to Docker to use vLLM's official OpenAI-compatible image:

```toml
[placement]
kind = "docker"
host = "127.0.0.1"
port = 8000
image = "vllm/vllm-openai:v0.26.0"
runtime = "docker"
gpus = "all"
hugging_face_cache = "/home/user/.cache/huggingface"
```

The plan records the exact image and complete `docker run` argv. Launch receipts
record the container ID, and `inspect` and `cancel` reconnect through that ID.
Pin an image version appropriate for the host's driver and CUDA compatibility.

## Secrets

Set `api_key_env` on the vLLM engine to reference a named environment variable:

```toml
[engine]
kind = "vllm"
api_key_env = "LOCAL_VLLM_API_KEY"
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
