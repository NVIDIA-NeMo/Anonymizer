<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Detect

Entity detection is the first stage of every Anonymizer pipeline. Both replace and rewrite modes depend on it.

---

## How it works

Detection combines a lightweight NER model (GLiNER-PII) with LLM-based refinement. GLiNER PII produces an initial set of entity spans, then an LLM augments it with entities the NER missed and validates each detection -- keeping, reclassifying, or dropping entities based on context. 

When rewrite is configured, an additional step identifies **latent entities** -- sensitive information inferable from context but not explicitly stated in the text.

### Example: standard vs. latent entities

Consider this short passage:

> Sarah described her appointment. She's looking forward to ringing the bell soon and said the care team has been wonderful.

| Type | Value | Description |
| --- | --- | --- |
| Standard entity | Sarah | A directly stated first name. |
| Latent entity | cancer treatment | Inferred from context. The passage never explicitly says "cancer," but "ringing the bell" can imply nearing the end of cancer treatment. |

---

## Configuration

Detection is configured via the `Detect` object on `AnonymizerConfig`:

```python
from anonymizer import AnonymizerConfig, Detect, Redact

config = AnonymizerConfig(
    detect=Detect(),
    replace=Redact(),
)
```

### `Detect` fields

| Field | Default | Description |
|-------|---------|-------------|
| `entity_labels` | `None` (all defaults) | List of labels to detect. Leave unset (or pass `None`) to use the full default set. |
| `gliner_threshold` | `0.3` | GLiNER confidence threshold (0.0--1.0). Lower values detect more entities but may increase false positives. |
| `validation_max_entities_per_call` | `100` | Maximum candidate entities per validator LLM call. Rows with more candidates are split into chunks. See [Chunked validation](#chunked-validation). |
| `validation_excerpt_window_chars` | `500` | Characters of context included before and after a chunk's entity spans in the validator prompt. Bounds per-chunk prompt size; not the model's context-window limit. |

---

## Chunked validation

When a row yields many entity candidates, validating them in a single LLM call can exceed the model's context window or the provider's rate limits (the tokens-per-minute or requests-per-minute quota most hosted models enforce). Anonymizer automatically splits validation for such rows: candidates are grouped in position order into chunks of at most `validation_max_entities_per_call`, and each chunk is validated independently with its own bounded text excerpt (`validation_excerpt_window_chars` before and after the chunk's span). Decisions are merged back into a single per-row set.

The chunked path is always on; if a row has fewer candidates than the limit, it runs as a single call and is exactly equivalent to the unchunked behavior. Tuning guidance:

- **Raise `validation_max_entities_per_call`** if your validator has a large context window and you want fewer, larger calls.
- **Lower it** if you hit provider rate limits or want more uniform per-call latency.
- **Raise `validation_excerpt_window_chars`** when short windows hide the context needed to disambiguate entities (e.g., `"John"` as first name vs. last name depends on surrounding text).
- **Lower it** to reduce per-chunk prompt tokens, at the risk of lower validation quality on context-sensitive labels.

### Validator pools

`entity_validator` can be a single alias (the default) or a list of aliases — a **pool**. When multiple aliases are configured, each chunk in a row is dispatched to the next alias in round-robin order, which lets you work around per-alias rate limits by spreading requests across equivalent endpoints.

See [Validator pools](models.md#validator-pools) for the YAML syntax and caveats.


## Entity labels

Anonymizer ships with a comprehensive default label set covering:

- **Direct identifiers** (e.g. `first_name`, `last_name`, `email`, `ssn`, `date_of_birth`, `street_address`)
- **Quasi-identifiers** (e.g. `age`, `city`, `state`, `country`, `occupation`, `company_name`, `date`)
- **Technical data** (e.g. `api_key`, `password`, `url`, `ipv4`, `ipv6`, `device_identifier`)
- **Demographics** (e.g. `gender`, `race_ethnicity`, `religious_belief`, `political_view`, `language`)
- **Financial** (e.g. `credit_debit_card`, `account_number`, `bank_routing_number`, `tax_id`)

To inspect the full list:

```python
from anonymizer import DEFAULT_ENTITY_LABELS
print(DEFAULT_ENTITY_LABELS)
```

### Custom labels

When you pass `entity_labels` explicitly, the augmenter operates in **strict mode** -- it only outputs entities matching your list. When `entity_labels=None`, the augmenter can create additional labels beyond the defaults (e.g., `clinic_name`, `server_name`).

```python
# Strict: only detect these 3 labels
Detect(entity_labels=["first_name", "last_name", "email"])

# Permissive: detect all defaults + LLM can infer new label types
Detect()  # entity_labels=None
```
## Tuning the threshold

For `gliner_threshold`, start with the default `0.3`. If you're seeing too many false positives, raise it to `0.5`. If entities are being missed, try lowering to `0.2`. The LLM validation step catches many false positives, so erring on the side of lower thresholds is usually safe.

---

## Model roles

The detection pipeline uses three model roles, each mapped to a model alias in the default config:

| Role | Default alias | Purpose |
|------|--------------|---------|
| `entity_detector` | [`gliner-pii-detector`](https://build.nvidia.com/nvidia/gliner-pii) | GLiNER-PII NER model. |
| `entity_validator` | [`gpt-oss-120b`](https://build.nvidia.com/openai/gpt-oss-120b) | Validates and reclassifies detected entities. |
| `entity_augmenter` | [`gpt-oss-120b`](https://build.nvidia.com/openai/gpt-oss-120b) | Finds entities the NER model missed. |
| `latent_detector` | [`nemotron-30b-thinking`](https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b) | Identifies inferable entities (rewrite only). |

See [Models](models.md) for how to override these.
