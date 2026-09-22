<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Detect

Entity detection is the first stage of every Anonymizer pipeline. Both replace and rewrite modes depend on it.

---

## How it works

Detection combines built-in regex recognizers, a lightweight GLiNER2 PII model, and LLM-based refinement. Regex and GLiNER2 candidates are merged and validated by an LLM. A second LLM step adds entities the other detectors missed.

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
| `excluded_entity_labels` | `None` | List of labels to **never** detect, even if present in `entity_labels` or the default set. Excluded labels are removed before GLiNER and the LLM prompts run, and are also filtered from the final entity output as a safety net. |
| `gliner_threshold` | `0.3` | GLiNER confidence threshold (0.0--1.0). Lower values detect more entities but may increase false positives. |
| `validation_max_entities_per_call` | `100` | Maximum candidate entities per validator LLM call. Rows with more candidates are split into chunks. See [Chunked validation](#chunked-validation). |
| `validation_excerpt_window_chars` | `500` | Characters of context included before and after a chunk's entity spans in the validator prompt. Bounds per-chunk prompt size; not the model's context-window limit. |
| `builtin_regexes` | `True` | Run built-in regex recognizers when their labels are in the effective detection label set. |
| `regex_rules` | `[]` | Per-label `BuiltinRegex` settings and user-defined `RegexRule` recognizers. |

## Regex recognition

Built-in regex recognition for `credit_debit_card`, `email`, `ipv4`, `ipv6`, `mac_address`, and `url` is enabled by default. These are jurisdiction-neutral technical and payment formats rather than country-issued identifiers. Each recognizer combines a regex candidate pattern with structural validation.

### Built-in deterministic validators

Each built-in recognizer performs two local steps:

1. A regex finds text with the expected shape.
2. A deterministic validator rejects invalid matches.

Matches that pass these checks go to the contextual LLM validator by default. The LLM uses the surrounding text to decide whether the match is sensitive. Set `validate_with_llm=False` to accept locally validated matches without this step.

Set `regex_only=True` when regex recognition should be authoritative for a label. This accepts locally valid matches without LLM validation and removes that label from GLiNER and LLM augmentation. Other labels in the same record can still use model-based detection. If every active label is regex-only, the detector and augmenter calls are skipped. Enabled rules that share a label must use the same `regex_only` value.

| Entity label | Checks | Accepted forms | Rejected forms |
| --- | --- | --- | --- |
| `credit_debit_card` | Removes spaces and hyphens, requires 13--19 digits, and verifies the Luhn checksum. | Contiguous digits and digits separated by spaces or hyphens. A card issuer prefix is not required. | Incorrect length, repeated identical digits, or an invalid Luhn checksum. |
| `email` | Requires one `@`, a non-empty local part of at most 64 UTF-8 bytes, and a total value of at most 254 characters. The domain is NFC-normalized, converted through IDNA, and checked for total and per-label length. | Common unquoted local parts and multi-label Unicode domains, including CJK, Devanagari, and decomposed Latin input. | Local parts with leading, trailing, or consecutive dots; quoted local parts; domain literals such as `user@[192.0.2.1]`; single-label domains; and domain labels with leading or trailing hyphens. |
| `ipv4` | Parses the complete candidate as an IPv4 address. | Four decimal octets in the range 0--255. | Extra octets, out-of-range octets, and ambiguous leading-zero forms. |
| `ipv6` | Parses the complete candidate as an IPv6 address. | Full and compressed IPv6, plus dotted IPv4 tails such as `::ffff:192.0.2.128`. | Malformed compression, invalid hexadecimal groups, and invalid IPv4 tails. |
| `mac_address` | Requires either six two-digit hexadecimal groups using one consistent `:` or `-` separator, or three four-digit groups separated by dots. | Forms such as `00:1A:2B:3C:4D:5E`, `00-1A-2B-3C-4D-5E`, and `001A.2B3C.4D5E`. | Mixed separators, missing groups, and non-hexadecimal digits. |
| `url` | Parses only HTTP(S) and `www.` candidates, requires a host, validates ports in the range 1--65535, and validates the host as either an IP address or an NFC-normalized IDNA domain. DNS names require multiple labels with valid lengths and characters. | HTTP(S) URLs, case-insensitive `www.` prefixes, Unicode domains, IPv4 hosts, bracketed IPv6 hosts, paths, and query strings. Balanced closing delimiters in paths are retained while surrounding sentence punctuation is trimmed. | Malformed hosts, IPv4-shaped invalid hosts, single-label hosts such as `localhost`, and invalid ports. |

Entity provenance includes the built-in rule ID, for example `regex_builtin:nemo-anonymizer.email.v1`. Users do not set these IDs.

To skip LLM validation for one built-in:

```python
from anonymizer import BuiltinRegex, Detect

detect = Detect(
    entity_labels=["email", "ipv4"],
    regex_rules=[BuiltinRegex(label="ipv4", validate_with_llm=False)],
)
```

Set `builtin_regexes=False` to disable all built-in regex recognizers. GLiNER and LLM detection remain enabled.

To replace one built-in while keeping the others, disable it and add a custom rule with the same label:

```python
from anonymizer import BuiltinRegex, Detect, RegexRule

detect = Detect(
    regex_rules=[
        BuiltinRegex(label="email", enabled=False),
        RegexRule(
            label="email",
            pattern=MY_EMAIL_PATTERN,
            validator=my_email_validator,
        ),
    ],
)
```

### Custom regex rules and validators

`regex_rules` accepts both built-in settings and custom rules. Use `BuiltinRegex` to configure a built-in recognizer. Use `RegexRule` to add a regex for any entity label. `validate_with_llm` defaults to `True` for both.

```python
from anonymizer import Detect, RegexCandidate, RegexRule


def validate_support_case(candidate: RegexCandidate) -> bool:
    number = candidate.groups["number"]
    return not number.startswith("000")


detect = Detect(
    entity_labels=["support_case"],
    regex_rules=[
        RegexRule(
            label="support_case",
            pattern=r"CASE-(?P<number>\d{6})",
            validator=validate_support_case,
            regex_only=True,
        )
    ],
)
```

A custom validator receives a `RegexCandidate` with the matched value, character offsets, named capture groups, nearby context, and rule ID. It returns `bool` or `RegexValidationResult`. The validator is optional; without one, every regex match passes local validation.

Pass a callable directly when using `run()` or `preview()`. For exported configurations, package the validator under the `nemo_anonymizer.regex_validators` Python entry-point group and pass its registered name instead.

If you provide `entity_labels`, include the label of every enabled custom regex rule. Disabled custom rules are ignored. If you leave `entity_labels` unset, Anonymizer adds labels from enabled custom rules automatically.

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

### Excluding entity labels

Use `excluded_entity_labels` to omit specific labels from detection without having to enumerate the entire allowlist. Excluded labels are removed before GLiNER runs and before the LLM prompts are built, so they are never detected or augmented.

```python
# Detect all defaults except occupation and gender
Detect(excluded_entity_labels=["occupation", "gender"])

# Combine with an explicit allowlist — exclusions always win
Detect(entity_labels=["first_name", "email", "city"], excluded_entity_labels=["city"])
```

!!! warning
    `excluded_entity_labels` is always checked against the effective allowlist — `entity_labels` if set, otherwise `DEFAULT_ENTITY_LABELS`. A total overlap raises a `ValueError` at config time instead of silently detecting nothing. A partial overlap logs a warning only when `entity_labels` is explicit; against the default label set, it's silent.

## Tuning the threshold

For `gliner_threshold`, start with the default `0.3`. If you're seeing too many false positives, raise it to `0.5`. If entities are being missed, try lowering to `0.2`. The LLM validation step catches many false positives, so erring on the side of lower thresholds is usually safe.

---

## Model roles

The detection pipeline uses three model roles, each mapped to a model alias in the default config:

| Role | Default alias | Purpose |
|------|--------------|---------|
| `entity_detector` | [`gliner-pii-detector`](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi) | GLiNER2 PII model, served through a compatible local endpoint. |
| `entity_validator` | [`gpt-oss-120b`](https://build.nvidia.com/openai/gpt-oss-120b) | Validates and reclassifies detected entities. |
| `entity_augmenter` | [`gpt-oss-120b`](https://build.nvidia.com/openai/gpt-oss-120b) | Finds entities the NER model missed. |
| `latent_detector` | [`nemotron-30b-thinking`](https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b) | Identifies inferable entities (rewrite only). |

See [Models](models.md) for how to override these.
