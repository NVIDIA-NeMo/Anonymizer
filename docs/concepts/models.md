# Models

Anonymizer uses LLMs for entity detection, replacement, and rewriting. Models are configured via YAML and mapped to workflow roles.

---

## Defaults

Set your API key for Anonymizer to use models hosted on [build.nvidia.com](https://build.nvidia.com).

```bash
export NVIDIA_API_KEY="your-nvidia-api-key"
```

| Alias | Model | Used by |
|-------|-------|---------|
| `gliner-pii-detector` | [`nvidia/gliner-pii`](https://build.nvidia.com/nvidia/gliner-pii) | Entity detection (NER) |
| `gpt-oss-120b` | [`openai/gpt-oss-120b`](https://build.nvidia.com/openai/gpt-oss-120b) | Detection validation & augmentation, replacement, rewriting |
| `nemotron-30b-thinking` | [`nvidia/nemotron-3-nano-30b-a3b`](https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b) | Latent detection, evaluation, final judge |

Each pipeline stage has a **role** mapped to one of these aliases. See the full role list in the default configs: [`detection.yaml`](https://github.com/NVIDIA-NeMo/Anonymizer/blob/main/src/anonymizer/config/default_model_configs/detection.yaml), [`replace.yaml`](https://github.com/NVIDIA-NeMo/Anonymizer/blob/main/src/anonymizer/config/default_model_configs/replace.yaml), [`rewrite.yaml`](https://github.com/NVIDIA-NeMo/Anonymizer/blob/main/src/anonymizer/config/default_model_configs/rewrite.yaml).

---

## Custom providers

To use models from a non-default provider (OpenAI, OpenRouter, etc.), define a providers YAML:

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

Override specific roles by passing a unified YAML to `Anonymizer(model_configs=...)`. The `provider` field references a provider by name -- use `nvidia` for build.nvidia.com, or a custom provider defined above.

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

Roles you don't override keep their default alias selections, but those aliases must still exist in your `model_configs` pool.

!!! tip "Validate your config"

    Use [`anonymizer.validate_config(config)`](../reference/anonymizer/interface/anonymizer.md) (or [`anonymizer validate`](../reference/anonymizer/interface/cli/main.md) from the CLI) after changing model configs to catch alias mismatches before processing data.


### Choosing custom models

For Anonymizer, the best overall leaderboard model is not always the best default for every role.
Some roles are simple classification or constrained JSON generation tasks, while others require deeper
reasoning about privacy risk, long-context rewriting, and leakage repair.

Use benchmarks as signals for role fit, not as a single global ranking.

#### Most useful benchmark signals

| Benchmark / metric | What it predicts well in Anonymizer |
|--------------------|--------------------------------------|
| `IFBench` | Following detailed instructions, producing constrained outputs, and obeying prompt rules. |
| `AA-Omniscience Accuracy` | Recovering the right facts without dropping important information. |
| `AA-Omniscience Non-Hallucination` | Avoiding invented entities, facts, or unsupported claims. |
| `AA-LCR` | Handling long prompts with tagged text, domain guidance, replacement maps, and evaluation context. |
| `Humanity's Last Exam` / `GPQA Diamond` | General reasoning depth for privacy-sensitive planning and rewriting. |
| Latency / output speed / verbosity | Notebook UX, evaluation-loop cost, and practical throughput. |

#### Practical guidance

- Use your strongest models for `latent_detector`, `disposition_analyzer`, `rewriter`, and often `repairer`.
- Use mid-tier models for `entity_augmenter`, `meaning_extractor`, and `replacement_generator`.
- Use smaller or faster models for `entity_validator`, `domain_classifier`, `qa_generator`, and often `evaluator`.
- Do not optimize every role for peak leaderboard rank. Optimize the hard-to-recover privacy and rewrite steps for quality, and the bounded steps for reliability per token.