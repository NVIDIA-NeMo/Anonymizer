<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Detect

Entity detection is the first stage of every Anonymizer pipeline. Both replace and rewrite modes depend on it.

---

## How it works

Detection combines a lightweight GLiNER2 PII model with LLM-based refinement. GLiNER2 produces an initial set of entity spans, an LLM validates those candidates by keeping, reclassifying, or dropping them based on context, and then an augmenter finds entities GLiNER2 missed. Augmented findings are merged directly and are not independently revalidated.

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
| `entity_label_examples` | `{}` | User-configured positive examples keyed by label. Examples for default labels extend their built-in examples; non-default keys must also appear in an explicit `entity_labels` set. |
| `excluded_entity_labels` | `None` | List of labels to exclude from detection, even if present in `entity_labels` or the default set. Excluded labels are removed from the active detection scope and filtered from the final entity output. |
| `gliner_threshold` | `0.3` | GLiNER confidence threshold (0.0--1.0). Lower values detect more entities but may increase false positives. |
| `validation_max_entities_per_call` | `100` | Maximum candidate entities per validator LLM call. Rows with more candidates are split into chunks. See [Chunked validation](#chunked-validation). |
| `validation_excerpt_window_chars` | `500` | Characters of context included before and after a chunk's entity spans in the validator prompt. Bounds per-chunk prompt size; not the model's context-window limit. |

---

## Chunked validation

When a row yields many entity candidates, validating them in a single LLM call can often exceed the model's context window or the provider's rate limits (tokens-per-minute or requests-per-minute quotas that many hosted models enforce). Anonymizer automatically splits validation for such rows: candidates are grouped in position order into chunks of at most `validation_max_entities_per_call`, and each chunk is validated independently with its own bounded text excerpt (`validation_excerpt_window_chars` before and after the chunk's span). Decisions are merged back into a single per-row set.

The chunked path is always on; if a row has fewer candidates than the limit, it runs as a single call and is exactly equivalent to the unchunked behavior. Tuning guidance:

- **Raise `validation_max_entities_per_call`** if your validator has a large context window and you want fewer, larger calls.
- **Lower it** if you hit provider rate limits or want more uniform per-call latency.
- **Raise `validation_excerpt_window_chars`** when short windows hide the context needed to disambiguate entities (e.g., `"John"` as first name vs. last name depends on surrounding text).
- **Lower it** to reduce per-chunk prompt tokens, at the risk of lower validation quality on context-sensitive labels.

Each validator call includes the full resolved label/example pairs. Lowering `validation_max_entities_per_call` reduces the candidates in each call, but repeats that complete label/example section across more calls, which can increase total input tokens and cost—especially when many examples are configured.

### Validator pools

`entity_validator` can be a single alias (the default) or a list of aliases — a **pool**. When multiple aliases are configured, each chunk in a row is dispatched to the next alias in round-robin order, which lets you work around per-alias rate limits by spreading requests across equivalent endpoints.

Pools also act as **failover**. If a chunk's assigned alias can't complete the call (an unrecoverable rate limit, a 5xx that didn't clear on retry, a malformed response), the same chunk is automatically retried against the other aliases in your pool before the row is given up on. A chunk only fails once every alias in the pool has failed for it. This is a cheap way to harden validation against any one endpoint having a bad day, on top of the load-spreading role.

#### What happens when a row can't be validated

If validation can't get a complete answer for a row — every alias in the pool has failed on at least one of that row's chunks — the row is **dropped from the output** rather than passed through with some entities unvalidated. This is deliberate: the alternative would be writing the original text back out with those entities still un-scrubbed, which is an undesired outcome.

Dropped rows show up on `result.failed_records` with `step="detection"`, so you can tell which inputs didn't make it through by comparing input IDs against output IDs and reprocess those on a follow-up pass.

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

The label settings serve different purposes:

- `entity_labels` defines **which entity types** are in scope, such as `api_key`.
- `entity_label_examples` provides **representative values** for those types, such as `sk-ant-api03-abc123`.

The following terms distinguish label scope from example origin:

- A **default label** is present in `DEFAULT_ENTITY_LABELS`; a **non-default label** is absent from it.
- An **explicit label set** is any set supplied through `entity_labels` and may contain default labels, non-default labels, or both.
- **Built-in examples** ship with Anonymizer in `ENTITY_LABEL_EXAMPLES`.
- **Configured examples** are values you supply through `entity_label_examples`.

### Label scope

When you pass `entity_labels` explicitly, the augmenter operates in **strict mode** -- it only outputs entities matching your list. When `entity_labels=None`, the augmenter can create additional labels beyond the defaults (e.g., `clinic_name`, `server_name`).

```python
# Strict: only detect these 3 labels
Detect(entity_labels=["first_name", "last_name", "email"])

# Permissive: detect all defaults + LLM can infer new label types
Detect()  # entity_labels=None
```

### Positive examples

Use `entity_label_examples` to help detection recognize dataset- or domain-specific value formats, such as vendor-prefixed API keys, account handles, or organization-specific identifiers. These are positive examples of what a label may look like—not format allowlists, guaranteed matches, negative examples, or replacement templates.

To deterministically exclude an entire label type, use `excluded_entity_labels`. Configured examples cannot express value-level negative examples.

#### Examples for default labels

Configured examples for a default label are appended to its built-in examples:

```python
Detect(entity_label_examples={"api_key": ["sk-ant-api03-abc123"]})
```

Here, `api_key` is already active through `DEFAULT_ENTITY_LABELS`; its built-in examples remain active alongside the configured example.

#### Examples for non-default labels

A configured example does not activate a non-default label. Declare the label explicitly:

```python
Detect(
    entity_labels=["vendor_api_key"],
    entity_label_examples={"vendor_api_key": ["acme_live_abc123"]},
)
```

To detect all defaults plus a non-default label, include both in the explicit label set:

```python
from anonymizer import DEFAULT_ENTITY_LABELS

Detect(
    entity_labels=[*DEFAULT_ENTITY_LABELS, "vendor_api_key"],
    entity_label_examples={"vendor_api_key": ["acme_live_abc123"]},
)
```

Every non-excluded configured example key must appear in the explicit label set. A missing or misspelled key raises a validation error. Because the label set is explicit, the augmenter is strict.

!!! warning "Examples are sent to model providers"
    Configured examples are embedded in prompts and exported detection builders. Use synthetic patterns, not production credentials, secrets, or real PII. Explicitly enabled raw DataDesigner message traces also contain the rendered prompts.

Configured examples affect detection only; they do not guide substitution or evaluation.

The validator receives the full resolved examples: built-in plus configured examples for default labels, and configured examples for non-default labels. The augmenter receives all active label names but only configured examples, which show it the intended value shape beyond the label name alone. Built-in examples are omitted from the augmenter to limit prompt growth. Keep configured lists short because the validator's full example set is repeated for each validation chunk.

`entity_label_examples` is currently configured through the Python `Detect` API; the CLI does not provide a mapping syntax for this field.

### Excluding entity labels

Use `excluded_entity_labels` to omit specific labels from detection without having to enumerate the entire label set. Excluded labels are removed from the active label set before GLiNER runs. If a model still emits an excluded label, final filtering prevents it from appearing in detection results.

```python
# Detect all defaults except occupation and gender
Detect(excluded_entity_labels=["occupation", "gender"])

# Combine with an explicit label set — exclusions always win
Detect(entity_labels=["first_name", "email", "city"], excluded_entity_labels=["city"])
```

!!! warning
    Exclusions always win. Configured examples for an excluded label are ignored with a warning that names the label but never the example values. A total overlap with the default or explicit label set raises a `ValueError` instead of silently detecting nothing.

## Tuning the threshold

For `gliner_threshold`, start with the default `0.3`. If you're seeing too many false positives, raise it to `0.5`. If entities are being missed, try lowering to `0.2`. The LLM validation step catches many false positives, so erring on the side of lower thresholds is usually safe.

---

## Model roles

The detection pipeline uses three model roles, each mapped to a model alias in the default config:

| Role | Default alias | Purpose |
|------|--------------|---------|
| `entity_detector` | [`gliner-pii-detector`](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi) | GLiNER2 PII model, served through a compatible local endpoint. |
| `entity_validator` | [`gpt-oss-120b`](https://openrouter.ai/openai/gpt-oss-120b) | Validates and reclassifies detected entities. |
| `entity_augmenter` | [`gpt-oss-120b`](https://openrouter.ai/openai/gpt-oss-120b) | Finds entities the NER model missed. |
| `latent_detector` | [`nemotron-30b-thinking`](https://openrouter.ai/nvidia/nemotron-3-nano-30b-a3b) | Identifies inferable entities (rewrite only). |

See [Models](models.md) for how to override these.
