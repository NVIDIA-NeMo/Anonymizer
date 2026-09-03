<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Deploy Local Models

NeMo Anonymizer can run its detector and generation roles against models on
your own NVIDIA GPU. The repository includes pinned profiles and a lifecycle
tool for this purpose. The tool compiles a profile into a frozen, checksummed
plan, starts the service, checks readiness, and records the local process handle
it observed so it can later report status or request cleanup.

This path suits development workstations, on-premises servers, and isolated
environments where text must stay on local infrastructure. It manages one
deployment shape: a vLLM process on the machine where the tool runs. You can
run that process directly on the host or inside the supplied GPU container.

The tool lives under `tools/` and is not installed with the
`nemo-anonymizer` wheel. Start from a source checkout.

## Choose models

Nine profiles ship in `tools/inference_service_profiles/`:

| Profile | Endpoint model name | Role and hardware guidance |
| --- | --- | --- |
| `nvidia-gliner.toml` | `nvidia/gliner-pii` | Default PII detector; can share an 80 GB GPU with a 20B or 30B generator |
| `gliner2.toml` | `fastino/gliner2-privacy-filter-PII-multi` | Alternative multilingual PII detector |
| `vllm-local.toml` | `anonymizer-local` | TinyLlama lifecycle smoke test |
| `gemma-4-12b-it.toml` | `gemma-4-12b-it-local` | Instruction-tuned Gemma 4 generation; requires an NVIDIA GPU with at least 40 GB of memory |
| `gpt-oss-20b.toml` | `gpt-oss-20b-local` | Compact GPT-OSS generation |
| `gpt-oss-120b.toml` | `gpt-oss-120b-local` | GPT-OSS generation on a dedicated 80 GB GPU |
| `qwen3-30b-a3b-instruct.toml` | `qwen3-30b-a3b-instruct-local` | Multilingual generation |
| `nemotron-3.5-lightning.toml` | `nemotron-3.5-lightning-local` | High-throughput generation on an 80 GB GPU |
| `nemotron-3.5-lightning-nvfp4.toml` | `nemotron-3.5-lightning-nvfp4-local` | Official NVFP4 checkpoint with a conservative single-H100 configuration |

Each profile pins its Hugging Face revision. Generation profiles use stock
vLLM. The two detector profiles use the pinned external
[vLLM Factory](https://github.com/latenceainew/vllm-factory) integration and
Anonymizer's OpenAI-compatible detector adapter.

The Gemma profile uses the instruction-tuned
[`google/gemma-4-12B-it`](https://huggingface.co/google/gemma-4-12B-it)
checkpoint and bounds its larger native context window to 8,192 tokens. The
Nemotron NVFP4 profile uses NVIDIA's
[`NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4)
checkpoint. vLLM detects its ModelOpt quantization from the checkpoint, so the
profile does not request online quantization.

GPU memory also depends on the architecture, context length, concurrency, and
other processes. Lower `max_model_len`, `max_num_seqs`, or
`gpu_memory_utilization` when vLLM cannot reserve its KV cache. Do not co-host
the GPT-OSS 120B profile with another model on an 80 GB GPU.

## Understand the lifecycle

The workflow has two durable files:

```text
profile.toml -> compile -> plan.json -> launch -> launch.json
                              |                    |
                              +-> probe            +-> status
                                                   +-> stop
```

`compile` has no runtime effects. The plan contains the full command, endpoint,
dependencies, compatibility assessments, and a SHA-256 digest. Runtime commands
verify that digest before acting.

The digest detects accidental changes while a plan is stored or transported.
This developer tool trusts whoever can write the profile and plan: it does not
authenticate the plan's author, sign the plan, or recompile the embedded spec to
prove that every derived field is semantically consistent.

`launch` first requires a launch-scoped ownership proof from the new process,
then waits for `/v1/models` and a task-specific request. The ownership check
prevents an existing compatible service on the target port from satisfying a
new launch. A generation service must return a chat completion. A detector
must demonstrate dynamic labels, offsets, and scores. The launch receipt
records the process group and Linux start marker so later commands do not
signal a reused PID or an unrelated process group.

The v2 profile schema has four sections:

- `[task]` selects generation or entity detection and its required capabilities.
- `[model]` pins the Hugging Face model and optional LoRA adapter.
- `[vllm]` sets the served name, memory limits, parallelism, caching, Mamba controls, authentication source, and optional Factory plugin.
- `[local]` sets the bind address, port, startup timeout, and shutdown timeout.

## Deploy on the GPU host

### Install

Install [uv](https://docs.astral.sh/uv/) and sync the local-model dependency
group with Python 3.12:

```bash
uv sync --python 3.12 --group dev --group local-models
uv run --python 3.12 python -m vllm_factory.compat.doctor
nvidia-smi
```

The lockfile pins vLLM 0.27.1, vLLM Factory, and the CUDA compiler wheels used
by the Nemotron FlashInfer profile.

### Compile and launch

Run all commands from the repository root. This example starts NVIDIA GLiNER
on `127.0.0.1:8001`:

```bash
uv run --python 3.12 python tools/inference_service.py compile \
  --profile tools/inference_service_profiles/nvidia-gliner.toml \
  --source-revision "$(git rev-parse HEAD)" \
  --output gliner-plan.json

uv run --python 3.12 python tools/inference_service.py launch \
  --plan gliner-plan.json \
  --output gliner-launch.json \
  --log-directory .inference-service-runs
```

The command returns after the ownership and readiness checks pass. Keep
`gliner-launch.json`: it is an unsigned durable operation record containing the
observed handle and its consistency fingerprint. The tool validates the output
destination before launch and publishes file receipts with atomic replacement.
If publication still fails, it makes a bounded attempt to stop the launched
process. Later status and stop operations re-check the exact PID, Linux start
marker, and process group.

### Operate the service

```bash
uv run --python 3.12 python tools/inference_service.py status \
  --receipt gliner-launch.json

uv run --python 3.12 python tools/inference_service.py probe \
  --plan gliner-plan.json

curl -sf http://127.0.0.1:8001/v1/models | python -m json.tool

uv run --python 3.12 python tools/inference_service.py stop \
  --receipt gliner-launch.json
```

`stop` checks the recorded PID, start marker, and live process group before
sending `SIGTERM`, waits for the profile's shutdown timeout, and uses `SIGKILL`
if the process remains alive.

## Deploy in a GPU container

The supplied container image installs the same locked local-model environment.
The container stays alive as the deployment boundary. The lifecycle tool still
manages a local process inside that boundary, so plans and receipts keep the
same schema as a host deployment.

### Prerequisites

Install Docker, the NVIDIA driver, and
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Confirm that Docker can expose the GPU:

```bash
docker run --rm --gpus all \
  nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi
```

### Build the image

From the repository root:

```bash
docker build \
  --file tools/inference_service.Dockerfile \
  --tag nemo-anonymizer-local-models:dev \
  .
```

The build uses the repository lockfile. It downloads several large GPU wheels,
so allow enough disk space for the image and model cache.

### Start the deployment container

Create host directories for receipts, logs, prepared Factory models, and the
Hugging Face cache:

```bash
mkdir -p .local-model-deployment/state
mkdir -p .local-model-deployment/factory
mkdir -p "$HOME/.cache/huggingface"

docker run --detach \
  --name anonymizer-local-models \
  --gpus all \
  --ipc host \
  --network host \
  --volume "$PWD/.local-model-deployment/state:/state" \
  --volume "$PWD/.local-model-deployment/factory:/tmp/anonymizer-vllm-factory" \
  --volume "$HOME/.cache/huggingface:/models/huggingface" \
  nemo-anonymizer-local-models:dev
```

Host networking is required because the pinned profiles bind to `127.0.0.1`.
It is supported by Docker Engine on Linux. The profile port must be free on the
host. The default profiles do not authenticate requests, so keep the bind local
or place the service behind authenticated TLS before exposing it.

### Compile and launch inside the container

```bash
SOURCE_REVISION="$(git rev-parse HEAD)"
docker exec anonymizer-local-models \
  python tools/inference_service.py compile \
  --profile tools/inference_service_profiles/nvidia-gliner.toml \
  --source-revision "$SOURCE_REVISION" \
  --output /state/gliner-plan.json

docker exec anonymizer-local-models \
  python tools/inference_service.py launch \
  --plan /state/gliner-plan.json \
  --output /state/gliner-launch.json \
  --log-directory /state/logs
```

The mounted `/state` directory makes the plan, launch receipt, and logs visible
on the host. Operate the service through `docker exec`:

```bash
docker exec anonymizer-local-models \
  python tools/inference_service.py status \
  --receipt /state/gliner-launch.json

docker exec anonymizer-local-models \
  python tools/inference_service.py probe \
  --plan /state/gliner-plan.json

curl -sf http://127.0.0.1:8001/v1/models | python -m json.tool
```

### Stop cleanly

Stop each managed service before removing the container:

```bash
docker exec anonymizer-local-models \
  python tools/inference_service.py stop \
  --receipt /state/gliner-launch.json

docker stop anonymizer-local-models
docker rm anonymizer-local-models
```

Stopping the container first removes the runtime boundary before the tool can
write a stop receipt. Use `stop` first when lifecycle evidence or
graceful model shutdown matters.

## Connect Anonymizer

Add the local endpoint to a provider file:

```yaml title="providers.yaml"
providers:
  - name: local-gliner
    endpoint: http://127.0.0.1:8001/v1
    provider_type: openai
    api_key: EMPTY
```

Then route the detector alias to that provider in your model configuration:

```yaml title="models.yaml"
model_configs:
  - alias: gliner-pii-detector
    model: nvidia/gliner-pii
    provider: local-gliner
    inference_parameters:
      max_parallel_requests: 8
      timeout: 120
```

Custom `model_configs` replaces the full bundled model pool. Copy the bundled
configuration and change only the roles you are deploying locally, or define
every alias needed by the selected pipeline. See
[Custom models](models.md#custom-models) for role mapping and complete examples.

For the detector request and response contract, chunking behavior, and a live
PII request, see [Self-hosting GLiNER](self-hosting-gliner.md).

Compilation records static compatibility assessments. The launch probe observes
the endpoint shape and required task behavior. Run Anonymizer preview and
evaluation before accepting a model for a privacy-sensitive workload.
