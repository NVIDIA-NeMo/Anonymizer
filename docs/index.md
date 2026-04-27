<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 🕵️ NeMo Anonymizer

[![GitHub](https://img.shields.io/badge/github-repo-952fc6?logo=github)](https://github.com/NVIDIA-NeMo/Anonymizer) [![License](https://img.shields.io/badge/License-Apache_2.0-0074df.svg)](https://opensource.org/licenses/Apache-2.0) [![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)

NeMo Anonymizer detects and protects PII through context-aware replacement and rewriting. It offers high-quality user-guided entity detection, followed by modification options that maintain context while inducing privacy. You can review what sensitive information was found, adjust your masking strategy, and generate anonymized text.

Pick a strategy:
> Alice met with Bob and his daughter to review kindergarten application #A9349.

=== "Substitute"

    Maya met with Daniel and his daughter to review kindergarten application #B5821.

=== "Redact"

    [REDACTED_FIRST_NAME] met with [REDACTED_FIRST_NAME] and his daughter to review kindergarten application #[REDACTED_ID].

=== "Annotate"

    &lt;Alice, first_name&gt; met with &lt;Bob, first_name&gt; and his daughter to review kindergarten application #&lt;A9349, id&gt;.

=== "Hash"

    &lt;HASH_FIRST_NAME_3bc51062973c&gt; met with &lt;HASH_FIRST_NAME_cd9fb1e148cc&gt; and his daughter to review kindergarten application #&lt;HASH_ID_f2a5f83e2a4c&gt;.

=== "Rewrite"

    The family met with the admissions counselor to review their school application.

---

# Get Started

## Install

```bash
pip install nemo-anonymizer
```
## Setup 

```bash
# Get an API key from build.nvidia.com
export NVIDIA_API_KEY="your-nvidia-api-key"
```
By default, Anonymizer uses NVIDIA-hosted models for detection and LLM-based anonymization. You can also [bring your own models](concepts/models.md).

!!! warning "Default hosted models are best for experimentation"

    The default `build.nvidia.com` (NVIDIA Build) setup is a convenient way to try Anonymizer and iterate on previews. Use of NVIDIA Build is subject to NVIDIA Build's own terms of service and privacy practices, which are separate from and independent of the NeMo Framework library. NVIDIA Build is intended for evaluation and testing purposes only and may not be used in production environments. Do not upload any confidential information or personal data when using NVIDIA Build. Your use of NVIDIA Build is logged for security purposes and to improve NVIDIA products and services.

    Request and token rate limits on `build.nvidia.com` vary by account and model access, and lower-volume development access can be slow for full-dataset runs. Start with `preview()` on a small sample, then move to your own endpoint for production data and usage.

!!! info "Record length"

    Records up to 2,000 tokens each work with the default model configs. Longer text will require adjustment of model providers and model configs. 

## Anonymize

=== "Python"

    ```python
    from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, Substitute

    input_data = AnonymizerInput(
        source="https://raw.githubusercontent.com/NVIDIA-NeMo/Anonymizer/refs/heads/main/docs/data/NVIDIA_synthetic_biographies.csv",
        text_column="biography",
        data_summary="Biographical profiles of individuals",
    )
    anonymizer = Anonymizer()
    config = AnonymizerConfig(replace=Substitute())

    anonymizer.validate_config(config)

    preview = anonymizer.preview(
        config=config,
        data=input_data,
        num_records=5,
    )
    preview.display_record()

    # Inspect the preview, adjust parameters if needed, then run full data.
    output = anonymizer.run(
        config=config,
        data=input_data,
    )
    ```

=== "CLI"

    ```bash
    # Preview a few rows first
    anonymizer preview \
      --source "https://raw.githubusercontent.com/NVIDIA-NeMo/Anonymizer/refs/heads/main/docs/data/NVIDIA_synthetic_biographies.csv" \
      --text-column biography \
      --data-summary "Biographical profiles of individuals" \
      --replace substitute \
      --num-records 5

    # Then run the full dataset
    anonymizer run \
      --source "https://raw.githubusercontent.com/NVIDIA-NeMo/Anonymizer/refs/heads/main/docs/data/NVIDIA_synthetic_biographies.csv" \
      --text-column biography \
      --data-summary "Biographical profiles of individuals" \
      --replace substitute \
      --output biographies_anonymized.csv
    ```

!!! note "`data_summary` improves detection"

    `data_summary` is optional but recommended for domain-specific data. It helps the LLM find more entities and reduce false drops.

## Language And Regional Coverage

Anonymizer has been tested most extensively on English-language data. Multilingual quality has not yet been evaluated systematically across languages, domains, and models.

Although testing so far has been primarily in English, the supported entity set is not limited to U.S.-specific identifiers. Detection and anonymization can also apply to international formats such as non-U.S. phone numbers, addresses, legal references, and national or regional identification numbers, though coverage will vary by language, region, and model configuration.

If you are working with another language, we encourage you to experiment on a small sample first with `preview()`, validate detected entities and transformed output carefully, and adjust your [model providers and model configs](concepts/models.md) as needed.

## Inspect
View an interactive visualization with entity highlights.
```python
preview.display_record()
```
Access the main results -- original text, entities, and transformed text.
```python
preview.dataframe
```
Access the full pipeline trace with all internal columns.
```python
preview.trace_dataframe
```
---
## Next up

<div class="grid cards" markdown>

-   :material-text-search: [**Detect**](concepts/detection.md)

    Refine how to search for entities.

-   :material-find-replace: [**Replace**](concepts/replace.md)

    Customize how to replace entities -- substitute, redact, annotate, or hash.

-   :material-auto-fix: [**Rewrite**](concepts/rewrite.md)

    Generate a privacy-safe paraphrase of the entire text.

-   :material-book-open-variant: [**Tutorials**](tutorials/index.md)

    End-to-end notebooks for replace and rewrite.


</div>
