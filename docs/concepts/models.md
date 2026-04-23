<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Models

Anonymizer uses LLMs for entity detection, replacement, and rewriting. Models are configured via YAML and mapped to workflow roles.

---

## Defaults

Set your API key for Anonymizer to use models hosted on [build.nvidia.com](https://build.nvidia.com).

```bash
export NVIDIA_API_KEY="your-nvidia-api-key"
```

!!! note "Provider data handling"

    Anonymizer sends prompts and text snippets to your configured model provider. If data must stay in your trusted environment, use a trusted/local provider endpoint.

| Alias | Model | Used by |
|-------|-------|---------|
| `gliner-pii-detector` | [`nvidia/gliner-pii`](https://build.nvidia.com/nvidia/gliner-pii) | Entity detection (NER) |
| `gpt-oss-120b` | [`openai/gpt-oss-120b`](https://build.nvidia.com/openai/gpt-oss-120b) | Detection validation & augmentation, replacement, rewriting |
| `nemotron-30b-thinking` | [`nvidia/nemotron-3-nano-30b-a3b`](https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b) | Latent detection, evaluation, final judge |

Each pipeline stage has a **role** mapped to one of these aliases. See the full role list in the default configs: [`detection.yaml`](https://github.com/NVIDIA-NeMo/Anonymizer/blob/main/src/anonymizer/config/default_model_configs/detection.yaml), [`replace.yaml`](https://github.com/NVIDIA-NeMo/Anonymizer/blob/main/src/anonymizer/config/default_model_configs/replace.yaml), [`rewrite.yaml`](https://github.com/NVIDIA-NeMo/Anonymizer/blob/main/src/anonymizer/config/default_model_configs/rewrite.yaml).

---

## Custom providers

To use models from a different provider (OpenAI, OpenRouter, etc.), define a providers YAML:

```yaml
# my_providers.yaml
providers:
  - name: openai
    base_url: https://api.openai.com/v1
    api_key_env_var: OPENAI_API_KEY
  - name: openrouter
    base_url: https://openrouter.ai/api/v1
    api_key_env_var: OPENROUTER_API_KEY
```

Make sure the environment variables are set:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export OPENROUTER_API_KEY="your-openrouter-api-key"
```

```python
anonymizer = Anonymizer(model_providers="my_providers.yaml")
```

---

## Custom models

Override specific roles by passing a unified YAML path to `Anonymizer(model_configs=...)`. The `provider` field references a provider by name -- use `nvidia` for build.nvidia.com, or a custom provider defined above.

```yaml
# my_models.yaml
selected_models:
  detection:
    entity_detector: gliner-pii-detector
    entity_validator: gpt5
    entity_augmenter: gpt5
    latent_detector: claude-sonnet
  replace:
    replacement_generator: gpt5

model_configs:
  - alias: gliner-pii-detector
    model: nvidia/gliner-pii
    provider: nvidia
    inference_parameters:
      max_parallel_requests: 16
      timeout: 120
  - alias: gpt5
    model: gpt-5
    provider: openai
    inference_parameters:
      max_tokens: 4096
      temperature: 0.3
  - alias: claude-sonnet
    model: anthropic/claude-sonnet-4
    provider: openrouter
    inference_parameters:
      max_tokens: 8192
      temperature: 0.3
```

```python
anonymizer = Anonymizer(
    model_configs="my_models.yaml",
    model_providers="my_providers.yaml",
)
```

You can pass `model_configs` as either a YAML file path or a YAML string.

Roles you don't override keep their default alias selections, but those aliases must still exist in your `model_configs` pool.

!!! tip "Validate your config"

    Use [`anonymizer.validate_config(config)`](../reference/anonymizer/interface/anonymizer.md) (or [`anonymizer validate`](../reference/anonymizer/interface/cli/main.md) from the CLI) after changing model configs to catch alias mismatches before processing data.


### Validator pools

`entity_validator` accepts either a single alias (shown above) or a list of aliases. A list forms a **validator pool** with two jobs:

1. **Load spreading.** [Chunked validation](detection.md#chunked-validation) dispatches each chunk to the next alias in round-robin order, aggregating quota across equivalent endpoints when a single alias would hit the provider's rate limits (tokens-per-minute or requests-per-minute quotas).
2. **Failover.** If a chunk's assigned alias can't complete the call (an unrecoverable rate limit, a 5xx that didn't clear on retry, a malformed response), the same chunk is automatically retried against the other aliases in your pool before the row is given up on. A row is only dropped when *every* alias in the pool has failed for the same chunk. Single-alias pools have nothing to fall back to, so they behave the same as not using a pool.

```yaml
selected_models:
  detection:
    entity_detector: gliner-pii-detector
    entity_validator:
      - gpt5-primary
      - gpt5-secondary
    entity_augmenter: gpt5-primary
    latent_detector: claude-sonnet
```

Every alias in the pool must also appear in `model_configs`; `anonymizer validate` flags unknown aliases by index. A scalar value remains valid and is equivalent to a one-element list.

!!! warning "`max_parallel_requests` is enforced per alias"

    A pool with N aliases effectively allows up to `sum(max_parallel_requests for alias in pool)` concurrent validator calls per row when chunks exist. Budget your provider rate limits accordingly — the whole point of pooling is to multiply in-flight requests, but the multiplication is real.

    Pool aliases should target **equivalent models** (same model family, similar quality). Mixing heterogeneous models produces inconsistent validation across chunks in the same row and is almost always a misconfiguration.


### Choosing custom models

For Anonymizer, the best overall leaderboard model is not always the best default for every role.
Some roles are simple classification or constrained JSON generation tasks, while others require deeper
reasoning about privacy risk, long-context rewriting, and leakage repair (see [Risk tolerance](rewrite.md#risk-tolerance)).

Use benchmarks as signals for role fit, not as a single global ranking.

#### Most useful benchmark signals

| Benchmark | What it predicts well in Anonymizer |
|--------------------|-------------------------------------|
| `IFBench` | Following detailed instructions, producing constrained outputs, and obeying prompt rules. |
| `AA-Omniscience Accuracy` | Recovering the right facts without dropping important information. |
| `AA-Omniscience Non-Hallucination` | Avoiding invented entities, facts, or unsupported claims. |
| `AA-LCR` | Handling long prompts with tagged text, domain guidance, replacement maps, and evaluation context. |
| `Humanity's Last Exam` / `GPQA Diamond` | General reasoning depth for privacy-sensitive planning and rewriting. |

Also consider operational constraints like latency, output speed, and verbosity, since they drive cost and practical throughput.

#### Practical guidance

- Use your strongest models for `latent_detector`, `disposition_analyzer`, `rewriter`, and `repairer`.
- Use mid-tier models for `entity_augmenter`, `meaning_extractor`, and `replacement_generator`.
- Use smaller or faster models for `entity_validator`, `domain_classifier`, `qa_generator`, and `evaluator`.
- Do not optimize every role for peak leaderboard rank. Optimize the hard-to-recover privacy and rewrite steps for quality. Optimize bounded steps for reliability per token.
