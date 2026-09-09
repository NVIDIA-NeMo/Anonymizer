<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Detect

Entity detection is the first stage of every Anonymizer pipeline. Both replace and rewrite modes depend on it.

---

## How it works

Detection combines built-in regex recognizers, a lightweight NER model (GLiNER-PII), and LLM-based refinement. Regex and GLiNER candidates are merged, then an LLM augments them with entities the other detectors missed and validates each candidate -- keeping, reclassifying, or dropping entities based on context.

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
| `builtin_regexes` | `True` | Run built-in regex recognizers when their labels are in the effective detection label set. |
| `regex_rules` | `[]` | Per-label `BuiltinRegex` settings and user-defined `RegexRule` recognizers. |

## Regex recognition

Built-in regex recognition is enabled by default. Users normally do not write `builtin_regexes=True`; selecting a supported label is enough:

```python
Detect(entity_labels=["email", "url"])
```

The initial built-in labels are `credit_debit_card`, `email`, `ipv4`, `ipv6`, `mac_address`, and `url`. These are jurisdiction-neutral technical and payment formats rather than country-issued identifiers. Email and URL matching supports Unicode domains, including IDNA-compatible and CJK domains. Each recognizer combines a regex candidate pattern with structural validation, such as Luhn for payment cards and address parsing for IP values.

Built-in matches receive the same contextual LLM validation as GLiNER matches by default. Disable it for a specific built-in when its deterministic checks are sufficient for your application:

```python
from anonymizer import BuiltinRegex, Detect

detect = Detect(
    entity_labels=["email", "ipv4"],
    regex_rules=[BuiltinRegex(label="ipv4", validate_with_llm=False)],
)
```

Set `builtin_regexes=False` on `Detect` to disable all built-in recognizers while retaining GLiNER and LLM detection.

To replace one built-in while keeping the others, disable that label and add a
custom rule with the same label:

```python
from anonymizer import BuiltinRegex, Detect, RegexRule

detect = Detect(
    regex_rules=[
        BuiltinRegex(label="email", enabled=False),
        RegexRule(
            label="email",
            pattern=MY_EMAIL_PATTERN,
            validator=my_email_validator,
        )
    ],
)
```

### Custom regex rules and validators

`regex_rules` is the single collection for built-in settings and custom recognizers. Use `BuiltinRegex` to configure one curated recognizer and `RegexRule` for domain identifiers. `validate_with_llm` defaults to `True` on both types, so a regex match still receives contextual review unless explicitly disabled.

```python
from anonymizer import Detect, RegexCandidate, RegexRule, RegexValidationResult


def validate_support_case(candidate: RegexCandidate) -> RegexValidationResult:
    number = candidate.groups["number"]
    return RegexValidationResult(valid=not number.startswith("000"))


detect = Detect(
    entity_labels=["support_case"],
    regex_rules=[
        RegexRule(
            label="support_case",
            pattern=r"CASE-(?P<number>\d{6})",
            validator=validate_support_case,
            # validate_with_llm=True is the default
        )
    ],
)
```

A validator receives the matched value, character offsets, named capture groups, nearby context, and the rule ID. It returns `bool` or `RegexValidationResult`. Direct callables work for in-process `run()` and `preview()` calls. Exported detection configurations require a validator package registered under the `nemo_anonymizer.regex_validators` Python entry-point group; pass that entry-point name as `validator` so every worker resolves the same code.

When `entity_labels` is explicit, it must include every custom rule label. With `entity_labels=None`, custom rule labels are added to the default detection scope automatically.

---

## Chunked validation

When a row yields many entity candidates, validating them in a single LLM call can often exceed the model's context window or the provider's rate limits (tokens-per-minute or requests-per-minute quotas that many hosted models enforce). Anonymizer automatically splits validation for such rows: candidates are grouped in position order into chunks of at most `validation_max_entities_per_call`, and each chunk is validated independently with its own bounded text excerpt (`validation_excerpt_window_chars` before and after the chunk's span). Decisions are merged back into a single per-row set.

The chunked path is always on; if a row has fewer candidates than the limit, it runs as a single call and is exactly equivalent to the unchunked behavior. Tuning guidance:

- **Raise `validation_max_entities_per_call`** if your validator has a large context window and you want fewer, larger calls.
- **Lower it** if you hit provider rate limits or want more uniform per-call latency.
- **Raise `validation_excerpt_window_chars`** when short windows hide the context needed to disambiguate entities (e.g., `"John"` as first name vs. last name depends on surrounding text).
- **Lower it** to reduce per-chunk prompt tokens, at the risk of lower validation quality on context-sensitive labels.

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
