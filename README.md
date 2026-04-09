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
git clone https://github.com/NVIDIA-NeMo/Anonymizer.git
cd Anonymizer
make install
```

### 2. Set up model providers

By default, Anonymizer uses models hosted on [build.nvidia.com](https://build.nvidia.com/models) — GLiNER-PII for entity detection and a text LLM for augmentation/validation. You can also bring your own models via custom provider configs.

Use the default build.nvidia.com setup as a convenient way to experiment with Anonymizer and iterate on small samples. For privacy-sensitive or production data, point Anonymizer at a secure endpoint you trust and to which you are comfortable sending data. Request and token rate limits on build.nvidia.com vary by account and model access, and lower-volume development access can be slow for full-dataset runs.

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

# Uses default model providers (build.nvidia.com) via NVIDIA_API_KEY env var
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

## Development

```bash
make install-dev          # Install with dev dependencies
make test                 # Run tests
make coverage             # Run with coverage report
make format-check         # Lint + format check (read-only)
anonymizer --help         # CLI usage
make install-pre-commit   # Install pre-commit hooks
```

---

## Requirements

- Python 3.11+
- [NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner) (installed as dependency)
- [NVIDIA API key](https://build.nvidia.com) for default model providers (GLiNER-PII + text LLM), or custom model endpoints

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
