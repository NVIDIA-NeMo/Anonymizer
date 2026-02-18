# NeMo Anonymizer

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)

**Detect and replace sensitive entities in text using LLM-powered workflows.**

---

## What can you do with Anonymizer?

- **Detect entities** using NemotronPII and LLM-based augmentation and validation
- **Replace with 4 strategies** — redact, label, hash (deterministic, local) or LLM-generated synthetic values
- **Preview results** before full runs with `display_record()` visualization

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/NVIDIA/Anonymizer.git
cd Anonymizer
make install
```

### 2. Set up model providers

By default, Anonymizer uses models hosted on [build.nvidia.com](https://build.nvidia.com) — NemotronPII for entity detection and a text LLM for augmentation/validation. You can also bring your own models via custom provider configs. See [model configuration docs](docs/concepts/models/model-provider-config.md) for details.

```bash
export NIM_API_KEY="your-nvidia-api-key"
```

### 3. Anonymize text

```python
from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput, InputSourceType
from anonymizer.config.replace_strategies import RedactReplace
from anonymizer.interface.anonymizer import Anonymizer

# Uses default model providers (build.nvidia.com) via NIM_API_KEY env var
anonymizer = Anonymizer()

config = AnonymizerConfig(replace=RedactReplace())

preview = anonymizer.preview(
    config=config,
    data=AnonymizerInput(
        source="data.csv",
        source_type=InputSourceType.csv,
        text_column="text",
    ),
    num_records=3,
)

# Visualize with entity highlights and replacement map
preview.display_record()
```

For custom model endpoints, pass a providers YAML:

```python
anonymizer = Anonymizer(model_providers="path/to/model_providers.yaml")
```

---

## Replacement Strategies

| Strategy | Output for `"Alice"` (first_name) | Configurable |
|----------|----------------------------------|-------------|
| **RedactReplace** | `[REDACTED_FIRST_NAME]` | `redact_template` |
| **LabelReplace** | `<Alice, first_name>` | `format_template` |
| **HashReplace** | `<HASH_FIRST_NAME_3bc51062973c>` | `format_template`, `algorithm`, `digest_length` |
| **LLMReplace** | `Maya` | `instructions` |

```python
from anonymizer.config.replace_strategies import RedactReplace, LabelReplace, HashReplace, LLMReplace

# Constant redaction
AnonymizerConfig(replace=RedactReplace(redact_template="***"))

# Deterministic hash with short digest
AnonymizerConfig(replace=HashReplace(algorithm="sha256", digest_length=8))

# LLM-generated contextual replacements
AnonymizerConfig(replace=LLMReplace(), data_summary="Describe your data...")
```

---

## Development

```bash
make install-dev          # Install with dev dependencies
make test                 # Run tests
make coverage             # Run with coverage report
make check-all            # Lint + format check
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
