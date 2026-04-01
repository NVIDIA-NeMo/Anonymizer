# :material-incognito:{ .nvidia-green } NeMo Anonymizer

[![GitHub](https://img.shields.io/badge/github-repo-952fc6?logo=github)](https://github.com/NVIDIA-NeMo/Anonymizer) [![License](https://img.shields.io/badge/License-Apache_2.0-0074df.svg)](https://opensource.org/licenses/Apache-2.0) [![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)

**Detect and protect PII through context-aware replacement and rewriting.**

Anonymizer is a privacy-preserving text processing library built for high-quality entity detection with user guidance. Review what was found, tune the protection strategy, then generate anonymized text.

Pick a strategy:
> Alice met with Bob and his daughter to review kindergarten application #A9349.

=== "Annotate"

    &lt;Alice, first_name&gt; met with &lt;Bob, first_name&gt; and his daughter to review kindergarten application &lt;A9349, id&gt;.

=== "Redact"

    [REDACTED_FIRST_NAME] met with [REDACTED_FIRST_NAME] and his daughter to review kindergarten application [REDACTED_ID].

=== "Hash"

    &lt;HASH_FIRST_NAME_3bc51062973c&gt; met with &lt;HASH_FIRST_NAME_cd9fb1e148cc&gt; and his daughter to review kindergarten application &lt;HASH_ID_f2a5f83e2a4c&gt;.

=== "Substitute"

    Maya met with Daniel and his daughter to review kindergarten application #B5821.

=== "Rewrite"

    The family met with the admissions counselor to review their school application.

---

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

## Anonymize

```python
from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, Redact

input_data = AnonymizerInput(source="patient_data.csv", text_column="notes", data_summary="Records containing detailed notes on patient encounters")
anonymizer = Anonymizer()
config = AnonymizerConfig(replace=Redact())

anonymizer.validate_config(config)

preview = anonymizer.preview(
    config=config,
    data=input_data,
    num_records=5,
)

preview.display_record()
```
Inspect the preview, adjust parameters if needed, then run the full dataset.
```python
output = anonymizer.run(
    config=config, 
    data=input_data
)
```

!!! note "`data_summary` improves detection"

    `data_summary` is optional but recommended for domain-specific data. It helps the LLM find more entities and reduce false drops.

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

    Customize how to replace entities -- redact, annotate, hash, or substitute.

-   :material-auto-fix: [**Rewrite**](concepts/rewrite.md)

    Generate a privacy-safe paraphrase of the entire text.

-   :material-book-open-variant: [**Tutorials**](notebooks/01_your_first_anonymization.ipynb)

    End-to-end notebooks for replace and rewrite.


</div>
