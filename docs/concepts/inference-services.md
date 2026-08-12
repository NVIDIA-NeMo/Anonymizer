<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Run Local Inference Services

The source tree provides `tools/inference_service.py` for compiling and managing
one domain: a local-process vLLM server. Compilation is pure and produces a
versioned, digest-protected plan; launch, probe, inspect, and cancel consume that
plan rather than re-reading a profile.

Install the optional local-model dependencies on a Linux GPU host:

```bash
uv sync --python 3.12 --group dev --group local-models
python -m vllm_factory.compat.doctor
nvidia-smi
```

The group pins vLLM 0.27.1, the external vLLM Factory revision, and the CUDA
compiler wheels needed by the Nemotron profile.

## Profiles

Seven pinned profiles ship in `tools/inference_service_profiles/`. Generation
profiles use Hugging Face vLLM. `nvidia-gliner.toml` and `gliner2.toml` use the
pinned NVIDIA vLLM Factory integration for entity detection. Detection requires
the Factory section; stock vLLM generation must not include it.

| Profile | Served model | Use |
| --- | --- | --- |
| `vllm-local.toml` | `anonymizer-local` | Small lifecycle smoke tests |
| `gpt-oss-20b.toml` | `gpt-oss-20b-local` | Compact GPT-OSS development |
| `gpt-oss-120b.toml` | `gpt-oss-120b-local` | Dedicated 80 GB GPU |
| `qwen3-30b-a3b-instruct.toml` | `qwen3-30b-a3b-instruct-local` | Multilingual generation |
| `nemotron-3.5-lightning.toml` | `nemotron-3.5-lightning-local` | High-throughput generation |
| `nvidia-gliner.toml` | model ID | NVIDIA GLiNER detection |
| `gliner2.toml` | model ID | GLiNER2 detection |

Memory requirements also depend on the GPU, driver, context length, and
concurrency. Reduce `max_model_len` or `max_num_seqs` if vLLM cannot reserve its
KV cache. Do not co-host the 120B GPT-OSS profile on the GPU used for detection.

## Lifecycle

```bash
uv run tools/inference_service.py compile \
  --profile tools/inference_service_profiles/nvidia-gliner.toml \
  --source-revision "$(git rev-parse HEAD)" --output plan.json
uv run tools/inference_service.py launch --plan plan.json --output launch.json
uv run tools/inference_service.py inspect --receipt launch.json
uv run tools/inference_service.py probe --plan plan.json
uv run tools/inference_service.py cancel --receipt launch.json
```

The `[local]` section holds host, port, and bounded startup/shutdown timeouts.
The `[vllm]` section holds tensor parallelism, memory and model limits, API-key
environment source, LoRA, eager/prefix/async controls, and Mamba controls.
Secrets remain symbolic in plans and are read from their named environment
variable only at launch. Probes use `/v1/models` plus a task-aware chat payload;
the generation probe accepts reasoning-aware GPT-OSS responses.

## GLiNER through vLLM Factory

Factory-backed detection keeps vLLM Factory's pooling endpoint and adds the
OpenAI-compatible chat contract used by Anonymizer. The adapter preserves
dynamic labels, offsets, scores, and overlapping character chunks. Model
preparation uses the profile's pinned Hugging Face revision.

## Connect Anonymizer

Use the plan endpoint and served model name in a custom DataDesigner provider
and model configuration. Custom model configuration replaces Anonymizer's
bundled model pool, so retain every alias required by the roles you use. See
[Custom models](models.md#custom-models) for the role map.

Compilation proves static compatibility. The launch probe proves the observed
endpoint contract. Run Anonymizer preview and evaluation before trusting a new
model for privacy or utility.

Docker placement, native Transformers GLiNER serving, and cache discovery are
not supported by this tool.
