<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Run Local vLLM Models

This guide is for contributors who operate an Anonymizer source checkout on a
Linux host with NVIDIA GPUs. It starts a local, OpenAI-compatible [vLLM](https://docs.vllm.ai/)
endpoint for development, evaluation, or an internal deployment. It does not
download a model or configure production networking for you.

The helper script, [`tools/vllm_debug.py`](https://github.com/NVIDIA-NeMo/Anonymizer/blob/main/tools/vllm_debug.py), is a source-tree development tool. It is not included in the `nemo-anonymizer` package.

## Host Requirements

Before installing the local-model dependencies, confirm that the host has:

- Linux, Python 3.11 or later, and an NVIDIA GPU that has enough free VRAM for the selected model and its context length.
- An NVIDIA driver that can communicate with the GPU. Verify this with `nvidia-smi`.
- Access to the model weights, either through the Hugging Face cache or a local model directory. Observe the model license and access controls.

The dependency group installs vLLM and its CUDA-compatible dependencies. Driver, GPU, model-size, and CUDA compatibility remain properties of the host. Start with a small model and a short context length before adopting a model for a benchmark or workflow.

## Install the Local-Model Environment

From a source checkout, install the development and local-model groups:

```bash
uv sync --group dev --group local-models
nvidia-smi
uv run python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

If vLLM is kept in a separate virtual environment, the helper can use that interpreter with `--vllm-python /path/to/python`. This is useful when the model-serving environment has different CUDA constraints from the Anonymizer development environment.

## Select a Cached Model

The helper discovers snapshots already in the Hugging Face hub cache. It does not fetch model weights:

```bash
uv run python tools/vllm_debug.py models --cached --json
```

Pass a `snapshot_path` from that output to `serve`. A Hugging Face model ID is also accepted by vLLM, but may trigger a download.

## Start and Verify a Server

Run the server from a terminal that will remain open:

```bash
uv run python tools/vllm_debug.py serve /path/to/cached/snapshot \
  --served-model-name anonymizer-local \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192
```

The default bind address, `127.0.0.1:8000`, keeps the endpoint local to the GPU host. `--served-model-name` gives clients a stable model ID, independent of the cache directory name. Stop the server with `Ctrl-C`.

From another terminal, verify both the model registration and one completion:

```bash
uv run python tools/vllm_debug.py models
uv run python tools/vllm_debug.py call \
  --model anonymizer-local \
  --prompt 'Reply with the word ready.' \
  --timeout-seconds 120
```

Use `--dry-run` with `serve` to print the vLLM command before reserving GPU memory. For multi-GPU models, add `--tensor-parallel-size N`. For a LoRA adapter, add `--adapter /path/to/adapter` and, if needed, `--adapter-name NAME`.

## Connect Anonymizer

vLLM presents an OpenAI-compatible endpoint. Add it to a custom provider file:

```yaml title="providers.yaml"
providers:
  - name: local-vllm
    endpoint: http://127.0.0.1:8000/v1
    provider_type: openai
    api_key: EMPTY  # vLLM has no API key unless one is configured below
```

In a custom `models.yaml`, set each local model configuration's `provider` to `local-vllm` and its `model` to the server's served-model name, `anonymizer-local` in the example above. Then pass both files to `Anonymizer`:

```python
from anonymizer import Anonymizer

anonymizer = Anonymizer(
    model_providers="providers.yaml",
    model_configs="models.yaml",
)
```

`model_configs` replaces Anonymizer's entire bundled model pool. Copy the bundled [`models.yaml`](https://github.com/NVIDIA-NeMo/Anonymizer/blob/main/src/anonymizer/config/default_model_configs/models.yaml), retain every alias required by the roles you use, and change only the models that should route to vLLM. See [Custom models](models.md#custom-models) for the role map and validation command.

Use `anonymizer.validate_config(config)` before processing data. A successful HTTP probe only confirms that the server responds. It does not establish that a model is suitable for detection, replacement, or privacy-preserving rewrite quality.

## Access From Another Host

Keep the default loopback binding whenever Anonymizer and vLLM run on the same machine. If a trusted internal client needs the endpoint, use a network policy or firewall as well as a server API key:

```bash
# .env.local is ignored by this repository. Do not commit it.
LOCAL_VLLM_API_KEY='replace-with-a-long-random-secret'

# Load it in the shell that starts the server and the client.
set -a; source .env.local; set +a

uv run python tools/vllm_debug.py serve /path/to/cached/snapshot \
  --host 0.0.0.0 \
  --served-model-name anonymizer-local \
  --api-key-env LOCAL_VLLM_API_KEY
```

Set the same secret's environment-variable name in the Anonymizer provider configuration:

```yaml title="providers.yaml"
providers:
  - name: local-vllm
    endpoint: http://gpu-host.internal:8000/v1
    provider_type: openai
    api_key: LOCAL_VLLM_API_KEY
```

Use an ignored `.env.local` file, or an ignored `.mise.local.toml` file when your checkout uses Mise, to keep local endpoint credentials out of version control. Load the secret only in the shell or task runner that starts the server and client. Do not expose a raw vLLM endpoint to the public internet. For production access, place it behind your organization's authenticated TLS-enabled network boundary.

## Operating Notes

- List the server's registered model IDs with `models` after every model or adapter change. Client `model` values must match one of those IDs.
- Start conservatively with `--gpu-memory-utilization` and `--max-model-len`; increase them only after observing stable GPU memory use and latency.
- Keep GLiNER separate from the LLM when GPU memory is constrained. The [self-hosted GLiNER guide](self-hosting-gliner.md) describes the detection endpoint.
- Treat local-model output as untrusted until it has passed the same Anonymizer preview, evaluation, and privacy review used for any other provider.
