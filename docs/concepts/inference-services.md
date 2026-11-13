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
deployment shape: a local process on the machine where the tool runs, either
vLLM for generation or Anonymizer's native GLiNER2 server for detection. You can
run vLLM directly on the host or inside the supplied GPU container.

The tool lives under `tools/` and is not installed with the
`nemo-anonymizer` wheel. Start from a source checkout.

## Choose models

Eleven profiles ship in `tools/inference_service_profiles/`:

| Profile | Port | Endpoint model name | Role and hardware guidance |
| --- | --- | --- | --- |
| `gliner2-native.toml` | 8001 | `fastino/gliner2-privacy-filter-PII-multi` | Multilingual PII detector served by Anonymizer's native GLiNER2 server |
| `vllm-local-test.toml` | 8000 | `anonymizer-local` | TinyLlama lifecycle smoke test |
| `gemma-4-12b-it.toml` | 8010 | `gemma-4-12b-it` | Instruction-tuned Gemma 4 generation; requires an NVIDIA GPU with at least 40 GB of memory |
| `gpt-oss-20b.toml` | 8012 | `gpt-oss-20b` | Compact GPT-OSS generation |
| `gpt-oss-120b.toml` | 8013 | `gpt-oss-120b` | GPT-OSS generation on a dedicated 80 GB GPU |
| `qwen3-30b-a3b-instruct.toml` | 8017 | `qwen3-30b-a3b-instruct` | Multilingual generation |
| `nemotron-3.5-lightning.toml` | 8014 | `nemotron-3.5-lightning` | High-throughput generation on an 80 GB GPU |
| `nemotron-3.5-lightning-nvfp4.toml` | 8015 | `nemotron-3.5-lightning-nvfp4` | Official NVFP4 checkpoint with a conservative single-H100 configuration |
| `gemma-4-26b-a4b-it.toml` | 8011 | `gemma-4-26b-a4b-it` | Gemma 4 26B-A4B MoE generation, text only, 48K context; one 80 GB GPU |
| `qwen3.8-27b.toml` | 8018 | `qwen3.8-27b` | Qwen3.8 27B generation, text only, 32K context; one 80 GB GPU |
| `nemotron-3.5-super-vl.toml` | 8016 | `nemotron-3.5-super-vl` | Nemotron 3.5 Super VL 120B-A12B (BF16), text only, 48K context; four 80 GB GPUs with tensor parallelism and `trust-remote-code` |

Each generation profile pins its Hugging Face revision and runs stock vLLM.
The detector profile starts from the main project environment
(`.venv/bin/python -m anonymizer.engine.local_inference.serve`), which runs
Anonymizer's native GLiNER2 server from a cached, dependency-isolated
environment; the server pins the GLiNER2 checkpoint and revision itself.
`validate` builds or reuses that environment (a one-time `pip` install), so run
it before the first `launch`.

The generation profiles select vLLM reasoning parsers through
`[backend.options].reasoning-parser`: `openai_gptoss` for GPT-OSS, `nemotron_v3` for
Nemotron, and `gemma4` for Gemma. These parsers separate reasoning from the
answer and allow JSON-schema constraints to apply to the answer after
reasoning ends. The Qwen3 Instruct profile supports only non-thinking mode
and does not set a reasoning parser.
Recompile saved plans after updating the tool or changing a profile.

GPT-OSS clients set `reasoning_effort` at the top level of each chat-completion
request, for example `reasoning_effort="low"` in the OpenAI client. Putting
this field inside `chat_template_kwargs` does not control GPT-OSS effort.
The readiness probe sends only the fields in the profile's
`[backend.probe_extra_body]`; the GPT-OSS profiles request low effort there and
the Nemotron, Gemma, and Qwen3.8 profiles disable thinking. This does not set a default for
subsequent client requests. vLLM ignores OpenRouter's `reasoning` field; to
switch thinking per request, set `chat_template_kwargs.enable_thinking` in the
model's `inference_parameters.extra_body`.

The `gpt-oss-120b`, `nemotron-3.5-lightning`, Gemma 4 26B-A4B, Qwen3.8, and
Nemotron Super VL profiles allow 32K-49K tokens of context so that requests
with large `max_tokens` budgets (16,384 to 32,768) fit; vLLM rejects a request
whose prompt plus `max_tokens` exceeds the profile's `max-model-len`.

The Gemma 4 12B profile uses the instruction-tuned
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
profile.toml -> compile -> plan.json -> validate -> launch -> launch.json
                              |                                |
                              +-> probe                        +-> status / stop
```

`compile` has no runtime effects. The plan contains the profile's spec, the
process arguments compiled from it, and a SHA-256 digest. Runtime commands
verify the digest and that compiling the embedded spec reproduces the plan
before acting. This developer tool trusts whoever can write the profile and
plan: it does not authenticate the plan's author or sign the plan.

`validate` checks the serving environment without starting a server. For vLLM
it runs vLLM's own argument parser and pre-load checks in the serving
interpreter; for GLiNER2 it imports the native server there. It does not
establish model or hardware compatibility; startup and task probes provide
additional checks.

`launch` starts the process, then requires a launch-scoped ownership proof from
it and waits for `/v1/models` and a task-specific request. The ownership check
prevents an existing compatible service on the target port from satisfying a
new launch. A generation service must return a chat completion. The detector
must return a labelled entity with character offsets and a score. The launch
receipt records the process group, the Linux start marker, and a digest of the
launch token so later commands do not signal a reused PID or an unrelated
process group.

A v3 profile sets the endpoint at the top level and the serving backend in
`[backend]`:

- Top level: `schema_version = "inference-service.spec/v3"`, `port` (1024 or
  higher), optional `host` (default `127.0.0.1`), `python_executable`,
  `api_key_env`, `startup_timeout_seconds`, and `shutdown_timeout_seconds`.
- `[backend]` with `kind = "vllm"`: `served_model_name`, `[backend.model]`
  (model ID, pinned revision, optional LoRA adapter), `[backend.options]`, and
  `[backend.probe_extra_body]`.
- `[backend]` with `kind = "gliner2"`: `device` (`auto`, `cuda`, `mps`, or `cpu`).

`[backend.options]` supplies vLLM serve options by their long flag names, such
as `reasoning-parser` and `gpu-memory-utilization`. `true` passes the flag,
`false` passes its `--no-` form, a list passes several values, and a table
passes one JSON value. vLLM's parser matches each name exactly, so an
abbreviated or misspelled option fails validation. Omitted options use the
pinned vLLM defaults. Names must be lowercase kebab-case, and empty lists are
rejected.

The tool derives `model`, `revision`, `tokenizer-revision`,
`served-model-name`, `host`, `port`, `uds`, `middleware`, and the LoRA options
from the rest of the profile, so `[backend.options]` cannot set them. Supply the
endpoint bearer token through the environment variable named by `api_key_env`;
the launcher passes it to the child as `VLLM_API_KEY` or
`ANONYMIZER_LOCAL_GLINER2_TOKEN`. Use `HF_TOKEN` for Hugging Face access.
Literal `api-key` and `hf-token` options are rejected so credentials do not
enter plans or process arguments. Recompile saved plans after migrating a
profile to v3.

## Deploy on the GPU host

### Install

Install [uv](https://docs.astral.sh/uv/) and sync the independent serving project
with Python 3.12. Run these commands from the repository root:

```bash
uv sync --project tools/inference-service-runtime --locked
nvidia-smi
```

The serving project's lockfile pins vLLM 0.33.0 and the CUDA compiler wheels
that FlashInfer uses to JIT-compile kernels (sampling for GPT-OSS, Mamba for
Nemotron). Its environment does
not install Anonymizer or DataDesigner. Install Anonymizer separately for client
code; the native notebook runtime continues to use the notebook extra.

vLLM profiles default to `tools/inference-service-runtime/.venv/bin/python`
and the GLiNER2 profile to the root `.venv/bin/python`. You can run the
controller from either environment; it starts the serving interpreter directly.
To use another serving environment, set `python_executable` in your profile. Recompile existing plans after migrating
from the root `local-models` group. The launcher clears inherited `PYTHONHOME`,
`PYTHONPATH`, and `VIRTUAL_ENV` before starting the child; GPU, Hugging Face, and
other runtime settings remain inherited. Avoid setting `UV_PROJECT_ENVIRONMENT`
to the root `.venv` when syncing the serving project.

### Compile and launch

Run all commands from the repository root. This example starts the native
GLiNER2 detector on `127.0.0.1:8001`:

```bash
uv run --project tools/inference-service-runtime --locked python tools/inference_service.py compile \
  --profile tools/inference_service_profiles/gliner2-native.toml \
  --output gliner-plan.json

uv run --project tools/inference-service-runtime --locked python tools/inference_service.py validate \
  --plan gliner-plan.json

uv run --project tools/inference-service-runtime --locked python tools/inference_service.py launch \
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
marker, and process group. If the leader has exited but workers remain, `stop`
signals the group only when every remaining process carries the launch token;
otherwise it refuses with `process-group-ownership-unverified` and `status`
reports `ownership-unverified`.

Read the non-secret client values from the compiled plan, then connect through the public factory:

```bash
uv run --project tools/inference-service-runtime --locked python tools/inference_service.py connection \
  --plan gliner-plan.json
```

```python
from anonymizer import GlinerEndpoint, create_anonymizer

anonymizer = create_anonymizer(
    gliner=GlinerEndpoint(
        url="http://127.0.0.1:8001/v1",
        model="fastino/gliner2-privacy-filter-PII-multi",
        api_key_env=None,
    )
)
```

Use the URL, model, and credential environment-variable name printed by `connection`; profile values
can differ from this unauthenticated loopback example. The command never prints a credential value or
a process handle. Pass `api_key_env=None` when `connection` prints `null`.
`create_anonymizer()` connects to the endpoint but does not own its process. Use the receipt commands
below for status and shutdown.

### Operate the service

```bash
uv run --project tools/inference-service-runtime --locked python tools/inference_service.py status \
  --receipt gliner-launch.json

uv run --project tools/inference-service-runtime --locked python tools/inference_service.py probe \
  --plan gliner-plan.json

curl -sf http://127.0.0.1:8001/v1/models | python -m json.tool

uv run --project tools/inference-service-runtime --locked python tools/inference_service.py stop \
  --receipt gliner-launch.json
```

`stop` checks the recorded PID, start marker, and live process group before
sending `SIGTERM`, waits for the profile's shutdown timeout, and uses `SIGKILL`
if the process remains alive.

## Deploy in a GPU container

The supplied container image installs the same locked vLLM serving environment.
It does not install Anonymizer, so it serves the vLLM generation profiles; run
the GLiNER2 detector on the host. The examples below use `gpt-oss-20b.toml`.
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

Create host directories for receipts, logs, and the Hugging Face cache:

```bash
mkdir -p .local-model-deployment/state
mkdir -p "$HOME/.cache/huggingface"

docker run --detach \
  --name anonymizer-local-models \
  --gpus all \
  --ipc host \
  --network host \
  --volume "$PWD/.local-model-deployment/state:/state" \
  --volume "$HOME/.cache/huggingface:/models/huggingface" \
  nemo-anonymizer-local-models:dev
```

Host networking is required because the pinned profiles bind to `127.0.0.1`.
It is supported by Docker Engine on Linux. The profile port must be free on the
host. The default profiles do not authenticate requests, so keep the bind local
or place the service behind authenticated TLS before exposing it.

### Compile and launch inside the container

```bash
docker exec anonymizer-local-models \
  python tools/inference_service.py compile \
  --profile tools/inference_service_profiles/gpt-oss-20b.toml \
  --output /state/gpt-oss-plan.json

docker exec anonymizer-local-models \
  python tools/inference_service.py validate \
  --plan /state/gpt-oss-plan.json

docker exec anonymizer-local-models \
  python tools/inference_service.py launch \
  --plan /state/gpt-oss-plan.json \
  --output /state/gpt-oss-launch.json \
  --log-directory /state/logs
```

The mounted `/state` directory makes the plan, launch receipt, and logs visible
on the host. Operate the service through `docker exec`:

```bash
docker exec anonymizer-local-models \
  python tools/inference_service.py status \
  --receipt /state/gpt-oss-launch.json

docker exec anonymizer-local-models \
  python tools/inference_service.py probe \
  --plan /state/gpt-oss-plan.json

curl -sf http://127.0.0.1:8012/v1/models | python -m json.tool
```

### Stop cleanly

Stop each managed service before removing the container:

```bash
docker exec anonymizer-local-models \
  python tools/inference_service.py stop \
  --receipt /state/gpt-oss-launch.json

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
  - name: local-gliner2
    endpoint: http://127.0.0.1:8001/v1
    provider_type: openai
    api_key: EMPTY
```

Then route the detector alias to that provider in your model configuration:

```yaml title="models.yaml"
model_configs:
  - alias: gliner-pii-detector
    model: fastino/gliner2-privacy-filter-PII-multi
    provider: local-gliner2
    inference_parameters:
      max_parallel_requests: 8
      timeout: 120
```

Custom `model_configs` replaces the full bundled model pool. Copy the bundled
configuration and change only the roles you are deploying locally, or define
every alias needed by the selected pipeline. See
[Custom models](models.md#custom-models) for role mapping and complete examples.

[`tools/inference_service_profiles/examples/local-models.yaml`](https://github.com/NVIDIA-NeMo/Anonymizer/blob/main/tools/inference_service_profiles/examples/local-models.yaml)
is a complete role configuration that uses only the shipped profiles: GLiNER2
detection with Gemma 4 26B-A4B, Qwen3.8 27B, Nemotron 3.5 Lightning, and
Nemotron 3.5 Super VL for the LLM roles.
[`local-providers.yaml`](https://github.com/NVIDIA-NeMo/Anonymizer/blob/main/tools/inference_service_profiles/examples/local-providers.yaml)
beside it points each provider at its profile's port. Each profile listens on its
own port, so the servers can run side by side on one host.

For the detector request and response contract, chunking behavior, and a live
PII request, see [Self-hosting GLiNER](self-hosting-gliner.md).

Validation checks the serving arguments. The launch probe observes
the endpoint shape and required task behavior. Run Anonymizer preview and
evaluation before accepting a model for a privacy-sensitive workload.
