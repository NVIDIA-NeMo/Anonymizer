# NeMo Anonymizer

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)

**Detect and replace sensitive entities in text using LLM-powered workflows.**

---

## What can you do with Anonymizer?

- **Detect entities** using NemotronPII and LLM-based augmentation and validation
- **Replace with 4 strategies** — redact, annotate, hash (deterministic, local) or LLM-generated substitute values
- **Preview results** before full runs with `display_record()` visualization

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/NVIDIA-NeMo/Anonymizer.git
cd Anonymizer
make install
source .venv/bin/activate
```

### 2. Set up model providers

By default, Anonymizer uses models hosted on [build.nvidia.com](https://build.nvidia.com/models) — NemotronPII for entity detection and a text LLM for augmentation/validation. You can also bring your own models via custom provider configs.

```bash
export NVIDIA_API_KEY="your-nvidia-api-key"
```

### 3. Anonymize text

#### CLI

```bash
# Preview on a small sample
anonymizer preview --source data.csv --replace redact

# Full run with output file
anonymizer run --source data.csv --replace redact --output result.csv

# Validate config without running
anonymizer validate --source data.csv --replace hash
```

Run `anonymizer --help` or `anonymizer <subcommand> --help` for all options.

#### Python API

```python
from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, Redact

# Uses default model providers (build.nvidia.com) via NVIDIA_API_KEY env var
anonymizer = Anonymizer()

config = AnonymizerConfig(replace=Redact())

preview = anonymizer.preview(
    config=config,
    data=AnonymizerInput(source="data.csv", text_column="text"),
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

---

## Replacement Strategies

| Strategy | Output for `"Alice"` (first_name) | Configurable |
|----------|----------------------------------|-------------|
| **Redact** | `[REDACTED_FIRST_NAME]` | `format_template` |
| **Annotate** | `<Alice, first_name>` | `format_template` |
| **Hash** | `<HASH_FIRST_NAME_3bc51062973c>` | `format_template`, `algorithm`, `digest_length` |
| **Substitute** | `Maya` | `instructions` |

```python
from anonymizer import Redact, Annotate, Hash, Substitute

# Constant redaction
AnonymizerConfig(replace=Redact(format_template="****"))

# Deterministic hash with short digest
AnonymizerConfig(replace=Hash(algorithm="sha256", digest_length=8))

# LLM-generated contextual replacements
AnonymizerConfig(replace=Substitute())
```

---

## Development

```bash
make install-dev          # Install with dev dependencies
make test                 # Run tests
make coverage             # Run with coverage report
make check-all            # Lint + format check
anonymizer --help         # CLI usage
make install-pre-commit   # Install pre-commit hooks
```

---

## Requirements

- Python 3.11+
- [NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner) (installed as dependency)
- [NVIDIA API key](https://build.nvidia.com) for default model providers (NemotronPII + text LLM), or custom model endpoints

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
