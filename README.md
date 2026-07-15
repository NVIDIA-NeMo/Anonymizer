# NeMo Anonymizer

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)

**Detect and replace sensitive entities in text using LLM-powered workflows.**

---

## What can you do with Anonymizer?

- **Detect entities** using GLiNER-PII and LLM-based augmentation and validation
- **Replace with 4 strategies** — LLM-generated substitute, redact, annotate, or hash (deterministic, local)
- **Preview results** before full runs with `display_record()` visualization

---

## Quick Start

### 1. Install
```bash
pip install nemo-anonymizer
```
Or install from source: 
```bash
git clone https://github.com/NVIDIA-NeMo/Anonymizer.git
cd Anonymizer
make setup
mise run install
```

### 2. Set up model providers

By default, Anonymizer uses models hosted on [build.nvidia.com](https://build.nvidia.com/models) — GLiNER-PII for entity detection and a text LLM for augmentation/validation. You can also bring your own models via custom provider configs.

The default build.nvidia.com (NVIDIA Build) setup is a convenient way to try Anonymizer and iterate on previews. Use of NVIDIA Build is subject to NVIDIA Build's own terms of service and privacy practices, which are separate from and independent of the NeMo Framework library. NVIDIA Build is intended for evaluation and testing purposes only and may not be used in production environments. Do not upload any confidential information or personal data when using NVIDIA Build. Your use of NVIDIA Build is logged for security purposes and to improve NVIDIA products and services.

Request and token rate limits on build.nvidia.com vary by account and model access, and lower-volume development access can be slow for full-dataset runs. Start with preview() on a small sample, then move to your own endpoint for production data and usage.

```bash
export NVIDIA_API_KEY="your-nvidia-api-key"
```

### 3. Anonymize text

#### CLI
> Tip: All examples below use `uv run` to invoke commands. If you prefer, activate the venv with `source .venv/bin/activate` and run commands directly.
```bash
DATA_URL="https://raw.githubusercontent.com/NVIDIA-NeMo/Anonymizer/refs/heads/main/docs/data/NVIDIA_synthetic_biographies.csv"

# Preview on a small sample
uv run anonymizer preview --source $DATA_URL --text-column biography --replace redact --num_records 3

# Full run with output file
uv run anonymizer run --source $DATA_URL --text-column biography --replace redact --output result.csv 

# Validate config without running
uv run anonymizer validate --source $DATA_URL --text-column biography --replace hash
```

Run `anonymizer --help` or `anonymizer <subcommand> --help` for all options.

#### Python API

```python
from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, Redact
DATA_URL = "https://raw.githubusercontent.com/NVIDIA-NeMo/Anonymizer/refs/heads/main/docs/data/NVIDIA_synthetic_biographies.csv"

# Uses Anonymizer's bundled model providers (see src/anonymizer/config/default_model_configs/providers.yaml)
anonymizer = Anonymizer()

config = AnonymizerConfig(replace=Redact())

preview = anonymizer.preview(
    config=config,
    data=AnonymizerInput(source=DATA_URL, text_column="biography"),
    num_records=3,
)

# Visualize with entity highlights and replacement map
preview.display_record()

# Most important columns only
preview.dataframe

# Full pipeline trace, including internal underscore-prefixed columns
preview.trace_dataframe
```

For custom model endpoints, pass a providers YAML:

```python
anonymizer = Anonymizer(model_providers="path/to/model_providers.yaml")
```

## Language And Regional Coverage

Anonymizer has been tested most extensively on English-language data. Multilingual quality has not yet been evaluated systematically across languages, domains, and models.

Although testing so far has been primarily in English, the supported entity set is not limited to U.S.-specific identifiers. Detection and anonymization can also apply to international formats such as non-U.S. phone numbers, addresses, legal references, and national or regional identification numbers, though coverage will vary by language, region, and model configuration.

If you are working with another language, we encourage you to experiment on a small sample first with `preview()`, validate detected entities and transformed output carefully, and adjust your model providers and model configs as needed.

---

## Replacement Strategies

| Strategy | Output for `"Alice"` (first_name) | Configurable |
|----------|----------------------------------|-------------|
| **Substitute** | `Maya` | `instructions` |
| **Redact** | `[REDACTED_FIRST_NAME]` | `format_template` |
| **Annotate** | `<Alice, first_name>` | `format_template` |
| **Hash** | `<HASH_FIRST_NAME_3bc51062973c>` | `format_template`, `algorithm`, `digest_length` |

```python
from anonymizer import Redact, Annotate, Hash, Substitute

# LLM-generated contextual replacements
AnonymizerConfig(replace=Substitute())

# Constant redaction
AnonymizerConfig(replace=Redact(format_template="****"))

# Annotation with entities tagging
AnonymizerConfig(replace=Annotate(format_template="<{text}-|-{label}>"))

# Deterministic hash with short digest
AnonymizerConfig(replace=Hash(algorithm="sha256", digest_length=8))
```

---

## Using with Claude Code

This repo ships a Claude Code skill at [`skills/anonymizer/`](skills/anonymizer/SKILL.md) that elicits your dataset's privacy requirements, recommends Rewrite or Replace with a strategy, and drafts a runnable script for you to iterate on. While the skill should work with other coding agents that support skills, development and testing has focused on Claude Code at this stage.

Install via [skills.sh](https://skills.sh):

```bash
npx skills add NVIDIA-NeMo/Anonymizer
```

After installation, invoke it with `/anonymizer` from within Claude Code, or describe what you want to anonymize and let it auto-trigger.

---

## Development

```bash
make setup                # Install the pinned Mise toolchain and create .venv
mise run bootstrap        # Install development dependencies
mise run test             # Run tests
mise run coverage         # Run with coverage report
mise run format-check     # Lint + format check (read-only)
anonymizer --help         # CLI usage
mise run install-pre-commit  # Install pre-commit hooks
```

Run `mise tasks` to list the available developer commands. Tasks live in `.mise/tasks/` and are used by CI. The Makefile retains deprecated compatibility aliases for existing scripts.

### Local endpoint credentials

Mise loads `.env` and `.env.local` from the repository root. Keep endpoint credentials in `.env.local`, which is ignored by Git:

```bash
# .env.local
NVIDIA_API_KEY=your-nvidia-build-key
OPENAI_API_KEY=your-openai-key
OPENROUTER_API_KEY=your-openrouter-key
```

Use `.mise.local.toml` for local Mise-specific overrides. It is ignored by Git, so it can hold redacted environment values when that layout is more convenient:

```toml
# .mise.local.toml
[env]
NVIDIA_API_KEY = { value = "your-nvidia-build-key", redact = true }
```

Do not commit credentials. Provider YAML should refer to environment variable names, never raw keys.

### Benchmark profiles

The benchmark task loads the same local credentials and accepts runner options after `--`:

```bash
mise run benchmark smoke
mise run benchmark smoke-traces
mise run benchmark smoke -- --dry-run
```

`smoke` runs the repository data smoke suite. `smoke-traces` adds DataDesigner message and scheduler task traces. Set `BENCHMARK_OUTPUT_DIR` in `.env.local` when you need a different output directory.

---

## Requirements

- Python 3.11+
- [NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner) (installed as dependency)
- [NVIDIA API key](https://build.nvidia.com) for default model providers (GLiNER-PII + text LLM), or custom model endpoints

---

## Telemetry and Privacy

NeMo Anonymizer collects anonymous run-level telemetry to help prioritize product improvements. One event is sent per `Anonymizer.run()` / `Anonymizer.preview()` call, containing only technical metadata: the replacement strategy in use, models used, model hosts (e.g. `nvidia-build`, `openrouter`, `other`), input-record counts, run duration, and failure attribution by pipeline step. **No user data, record contents, prompts, or model outputs are collected.** See the [Telemetry and Privacy docs](https://nvidia-nemo.github.io/Anonymizer/latest/#telemetry-and-privacy) for the full field list.

You may opt out of telemetry at any time:

- **For one CLI invocation**: pass `--no-emit-telemetry`
  ```bash
  uv run anonymizer run --source data.csv --text-column text --replace redact --no-emit-telemetry
  ```
- **In the SDK**: set `emit_telemetry=False` on `AnonymizerConfig`
  ```python
  config = AnonymizerConfig(replace=Redact(), emit_telemetry=False)
  ```
- **For the current shell**: set the environment variable
  ```bash
  export NEMO_TELEMETRY_ENABLED=false
  ```

Aggregate usage data (such as which models are most popular) will be shared back with the community. It is not used to track any individual user behavior.

**Use of third-party endpoints, including NVIDIA Build:** Anonymizer can be configured to use various inference endpoints, including [build.nvidia.com](https://build.nvidia.com), [OpenRouter](https://openrouter.ai), or local model servers. If you choose to use a third-party endpoint, that endpoint's own terms of service and privacy practices apply independently of this library. Any opt-out you exercise within Anonymizer does not extend to data collection by your chosen endpoint.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
