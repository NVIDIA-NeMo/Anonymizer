<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Detect

Entity detection is the first stage of every Anonymizer pipeline. Both replace and rewrite modes depend on it.

---

## How it works

Detection combines local regex recognizers, a lightweight GLiNER2 PII model, and LLM-based refinement. Regex and GLiNER2 findings retain independent source identities and admission routes. An LLM keeps, reclassifies, or drops candidates requiring contextual validation; locally accepted regex matches bypass that call but still obey label exclusions. An augmenter then independently finds sensitive entities in the canonical plain source text. It receives no detector tags or seed-entity list. Grounded findings, including rediscoveries of existing detections, use the shared admission and final reconciliation policy and are not independently revalidated.

After admission, identical label/span candidates are combined with their accepted
provenance intact. Overlaps are resolved once: longer spans win first; source
priority breaks equal-span ties (custom regex, built-in regex, GLiNER, augmenter).
A rejected claim never removes an independently accepted match at the same span.
Detections are occurrence-specific: accepting one occurrence does not copy its
label to every occurrence of the same text.

Augmenter suggestions must match the source text and their supplied context exactly. If some do not, Anonymizer keeps the matching suggestions and asks the augmenter to correct only the others once. Suggestions that still do not match are dropped; the record and its other detections are retained. Transport errors or responses that do not match the required schema still fail the record.

Unknown fields in augmentation responses are ignored, but required fields and their types are still checked. Passing those checks does not guarantee that a suggestion is grounded in the source or correctly classified.

When rewrite is configured, an additional step identifies **latent entities** -- sensitive information inferable from context but not explicitly stated in the text.

### Occurrences and boundaries

Each detection refers to a specific source span. Accepting one occurrence does not automatically accept every occurrence of the same text: in `May visited in May`, the two occurrences can have different meanings. Accepting `John Smith` also does not synthesize separate `first_name` and `last_name` detections.

An augmenter suggestion can identify multiple occurrences when its exact value and surrounding context match at multiple positions. This is contextual matching, not propagation of an accepted value throughout the document. Other occurrences need their own detector observation or grounded augmentation suggestion; the augmenter is not guaranteed to recover them.

Published values are sliced from the original source at their character offsets. Augmenter values and context must be verbatim; leading or trailing whitespace in a suggested value is moved into its context rather than included in the entity span. Exact grounding prevents invented text or mismatched offsets, but does not establish the right semantic boundary. Ambiguous boundaries still require review against the user's intended policy; Anonymizer does not trim emails to a list of familiar domain endings.

### Overlapping detections

Label eligibility is checked before overlap selection. Eligible detections with the same normalized label and source bounds are combined while retaining their independently accepted origins. Labels are normalized by trimming surrounding whitespace and case-folding.

The final selection is deterministic and greedy: longer spans are considered first, with earlier starts breaking equal-length ties, and any later candidate overlapping a selected span is discarded. This is not an optimizer for maximum total covered characters. For identical bounds, explicit acceptance outranks fallback protection, then source priority applies; native confidence scores break ties only within a shared source/model calibration domain. Touching spans do not overlap.

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

### In-memory records

Applications can pass typed records without first writing a CSV or Parquet file:

```python
from anonymizer import Anonymizer, AnonymizerConfig, Redact, TextRecord, TextRecordsInput

result = Anonymizer().run(
    config=AnonymizerConfig(replace=Redact()),
    data=TextRecordsInput(
        records=[
            TextRecord(id="request-1", text="Email Alice at alice@example.com"),
            TextRecord(id="request-2", text="Bob works at Acme"),
        ],
        data_summary="Application request logs",
    ),
)

entities_by_id = dict(zip(result.dataframe["id"], result.dataframe["final_entities"], strict=True))
```

`result.dataframe` is the public result surface for validated entities. It
retains IDs and input order for records that complete the pipeline. Dropped
records use the same caller-provided IDs in `result.failed_records`. Use
`result.trace_dataframe` only when you need internal pipeline diagnostics.

### `Detect` fields

| Field | Default | Description |
|-------|---------|-------------|
| `entity_labels` | `None` (all defaults) | List of labels to detect. Leave unset (or pass `None`) to use the full default set. |
| `entity_label_examples` | `{}` | User-configured positive examples organized by label. Examples for default labels extend their built-in examples; every referenced non-default label must also appear in an explicit `entity_labels` set. |
| `excluded_entity_labels` | `None` | List of labels to exclude from detection, even if present in `entity_labels` or the default set. Excluded labels are removed from the active detection scope and filtered from the final entity output. |
| `gliner_threshold` | `0.3` | GLiNER confidence threshold (0.0--1.0). Lower values detect more entities but may increase false positives. |
| `builtin_regexes` | `True` | Enable built-in recognizers for supported labels in the effective detection scope. |
| `regex_rules` | `[]` | Custom `RegexRule` patterns and per-label `BuiltinRegex` settings. |
| `validation_max_entities_per_call` | `100` | Number of detected entities to validate per LLM call. This limit may be exceeded to keep detections covering the same text together. See [Chunked validation](#chunked-validation). |
| `validation_excerpt_window_chars` | `500` | Characters of context included before and after a chunk's entity spans in the validator prompt. Bounds per-chunk prompt size; not the model's context-window limit. |

---

## Regex recognition

Built-ins cover `credit_debit_card`, `email`, `ipv4`, `ipv6`,
`mac_address`, and `url`. They run only for active labels and use local
structural checks (such as IP parsing and the card-number Luhn checksum).
Format validity is not proof that a value identifies a person; contextual
LLM validation is enabled by default.

Import the immutable `BUILTIN_REGEX_LABELS` tuple from `anonymizer` to discover
supported built-in labels programmatically:

```python
from anonymizer import BUILTIN_REGEX_LABELS, BuiltinRegex

rules = [BuiltinRegex(label=label) for label in BUILTIN_REGEX_LABELS]
```

This lists supported recognizers, not the labels enabled by a particular config;
the effective detection scope and exclusions still apply.

The URL recognizer supports explicit `http://`, `https://`, `ftp://`, `ftps://`,
`sftp://`, `ws://`, and `wss://` addresses, plus `www.` forms. Explicit-scheme
addresses may use internal single-label hosts or `localhost`, as well as DNS
names and IP addresses. Recognition and structural validation are entirely
local: no URL is fetched or resolved. Arbitrary bare domains, scheme-relative
references (`//host/path`), JSON-escaped addresses, `file://`, and cloud-storage
locators such as `s3://` are not covered by this built-in. Custom rules can
cover these forms. When `detect_additional_matches=False`, this scope is also
the label's detection limit; neither GLiNER nor augmentation fills its gaps.
Surrounding prose punctuation is trimmed while balanced delimiters are kept;
ambiguous literal trailing punctuation can require a custom rule with exact
match boundaries.

A standalone, case-insensitive `url:` field prefix is supported without a space
(for example, `URL:https://example.org/path`). The prefix is not part of the
detected span or replacement. Other immediately adjacent colon prefixes, such
as `Visit:` or `custom:`, and embedded prefixes such as `myurl:` remain outside
the built-in's scope; add whitespace or use a custom rule for those forms.

Use the same `regex_rules` list to configure built-ins or add custom patterns:

```python
from anonymizer import BuiltinRegex, Detect, RegexRule

Detect(
    regex_rules=[
        BuiltinRegex(label="email", validate_matches_with_llm=False),
        BuiltinRegex(label="url", enabled=False),
        RegexRule(label="support_case", pattern=r"CASE-[0-9]{8}"),
    ],
)
```

`Detect(builtin_regexes=False)` disables all built-ins without disabling custom
rules. With `entity_labels=None`, enabled custom rules extend the default label
set. With an explicit `entity_labels` list, every enabled custom rule's label
must appear in it. Exclusions always win, including for directly accepted rules.

Each enabled rule has two independent controls:

| Setting | Default | Effect when false |
| --- | --- | --- |
| `validate_matches_with_llm` | `True` | Locally valid matches are admitted directly, without contextual validation. |
| `detect_additional_matches` | `True` | Regex supplies every candidate span for the label; GLiNER and augmentation cannot introduce other spans for it. |

Enabled rules sharing a label must agree on `detect_additional_matches`.
When it is false, a model reclassification or augmentation suggestion needs
accepted, unchanged-label regex evidence at the exact source span. A rejected
regex match cannot authorize another claim. If a validator omits a decision,
the original regex span can retain fallback protection, but grants no accepted
authority to other claims.

Custom rules may include a synchronous `validator` callable accepting
`RegexCandidate` and returning `bool` or `RegexValidationResult`. It receives
the match value, offsets, named groups, adjacent context, and rule ID. Result
`reason` is optional diagnostic metadata; never put source values or other PII
in it. A false result discards only that match. Exceptions fail the record;
validators must not be asynchronous.
`RegexValidationResult.valid` must be a Boolean and `reason` must be a string
or `None`; malformed results fail the record rather than being coerced.

The built-in payment-card recognizer prefers the longest checksum-valid match.
If a trailing numeric code invalidates that match, it tries shorter candidates
only at complete digit-group boundaries, never inside a contiguous digit run.
Custom patterns retain their explicitly matched spans.

For exported builders, use a validator name registered by a trusted installed
package under `nemo_anonymizer.regex_validators`; direct callables are
in-process only. Disabled or excluded rules do not impose this export restriction.
Names must be unique. Imported validators execute trusted
Python code, so do not load registrations from untrusted packages.

Patterns use the third-party `regex` dialect, except that fuzzy matching groups
and match reset (`\K`) are unsupported. Invalid, zero-width, or oversized
patterns are rejected during configuration, as are constructs whose consumed
width cannot be established conservatively. Runtime matching has a per-rule
timeout and match-count limit; exceeding either fails the record rather than
silently omitting detections. This is not a sandbox for arbitrary validator
code.

## Chunked validation

When a row has many detected entities, Anonymizer splits validation across LLM calls using `validation_max_entities_per_call`. Detections that cover some of the same text stay together so the validator can compare them—for example, `San Diego` detected as a city and `Diego` detected as a first name. If keeping related detections together requires exceeding the limit, Anonymizer sends them in a separate call and logs a warning. The call must still fit the model's context window.

Each response is bound to its request inventory. Multi-chunk requests receive a source excerpt with `validation_excerpt_window_chars` before and after the assigned spans. Only candidates assigned to that request are tagged; neighboring detections remain plain text. Single-chunk requests receive the full text supplied to detection by default (currently the whole document).

The chunked path is always on; if a row has fewer candidates than the limit, it runs as a single call and is exactly equivalent to the unchunked behavior. Tuning guidance:

- **Raise `validation_max_entities_per_call`** if your validator has a large context window and you want fewer, larger calls.
- **Lower it** if you hit provider rate limits or want more uniform per-call latency.
- **Raise `validation_excerpt_window_chars`** when short windows hide the context needed to disambiguate entities (e.g., `"John"` as first name vs. last name depends on surrounding text).
- **Lower it** to reduce per-chunk prompt tokens, at the risk of lower validation quality on context-sensitive labels.

Each validator call includes the full resolved label/example pairs. Lowering `validation_max_entities_per_call` reduces the candidates in each call, but repeats that complete label/example section across more calls, which can increase total input tokens and cost—especially when many examples are configured.

### Validator pools

`entity_validator` can be a single alias (the default) or a list of aliases — a **pool**. When multiple aliases are configured, each chunk in a row is dispatched to the next alias in round-robin order, which lets you work around per-alias rate limits by spreading requests across equivalent endpoints.

Pools also act as **failover**. If a chunk's assigned alias can't complete the call (an unrecoverable rate limit, a 5xx that didn't clear on retry, a malformed response), the same chunk is automatically retried against the other aliases in your pool before the row is given up on. A chunk only fails once every alias in the pool has failed for it. This is a cheap way to harden validation against any one endpoint having a bad day, on top of the load-spreading role.

Validation aliases identify observations only within one request. Decisions are consumed once and cannot veto independently admitted augmenter findings. Unknown response aliases are ignored. Returning an assigned alias more than once invalidates the response, even if the decisions agree: a single-alias pool gets one bounded retry, while a multi-alias pool moves to the next member. Unresolved duplicate aliases fail the record rather than becoming fallback protection.

In an otherwise successful response, a missing/unusable decision or an empty reclassification label can retain an eligible candidate's original exact span and label as `fallback_protected`. This is a conservative protection measure, not explicit model acceptance, and grants no accepted provenance. It does not override an explicit drop or configured label exclusions.

#### What happens when a row can't be validated

If every alias fails on any required validation chunk, the row is **dropped from the output** rather than returned as a successful partial result. This differs from an omitted decision in a successful response, which can use the exact-span fallback described above. Transport failures, invalid required response fields, and unresolved duplicate assigned aliases are failures, not omissions.

A detector text/source mismatch also fails the record instead of silently skipping the observation. This check allows case-folded comparison and surrounding-whitespace normalization of detector-echoed text, then publishes the actual source substring. It does not mean every malformed detector payload is rejected: malformed entries and unusable bounds can still be ignored by the parser.

Dropped rows show up on `result.failed_records` with `step="entity-detection"`, so you can tell which inputs didn't make it through by comparing input IDs against output IDs and reprocess those on a follow-up pass.

See [Validator pools](models.md#validator-pools) for the YAML syntax and caveats.

### Inspecting detection diagnostics

Use `result.dataframe["final_entities"]` for the public entity output. Its representative `source` can be `detector`, `augmenter`, or `fallback`; it is not a full provenance ledger. `score` is the available native score, or `0.0` when none is available, not a universal probability of correctness. Augmenter-only findings are no longer assigned synthetic confidence `1.0`.

For internal debugging, `result.trace_dataframe` can include `_seed_detection_result`, `_validation_request_inventories`, `_validation_decisions`, `_admission_results`, `_admitted_candidates`, `_augmentation_detection_result`, `_augmentation_admission_results`, `_augmentation_attempts`, `_augmented_candidates`, and `_reconciled_candidates`. These underscore-prefixed structures are internal and may change between releases. `_augmentation_attempts` records attempt outcomes and grounding-failure categories without embedding suggested values or context; other trace columns and explicitly enabled raw message traces can contain sensitive source text.


## Entity labels

Anonymizer ships with a comprehensive default label set covering:

- **Direct identifiers** (e.g. `first_name`, `last_name`, `email`, `ssn`, `date_of_birth`, `street_address`)
- **Quasi-identifiers** (e.g. `age`, `city`, `state`, `country`, `occupation`, `company_name`, `date`)
- **Technical data** (e.g. `api_key`, `password`, `url`, `ipv4`, `ipv6`, `device_identifier`)
- **Demographics** (e.g. `gender`, `race_ethnicity`, `religious_belief`, `political_view`, `language`)
- **Financial** (e.g. `credit_debit_card`, `account_number`, `bank_routing_number`, `tax_id`)

The `date` label includes complete dates and partial calendar references, such as a month (`March`), year (`2024`), month and year (`March 2024`), or day of the month (`the 15th`), when the surrounding text establishes their date meaning. Standalone numbers without date context and vague time words such as `today`, `now`, and `soon` are not dates.

To inspect the full list:

```python
from anonymizer import DEFAULT_ENTITY_LABELS
print(DEFAULT_ENTITY_LABELS)
```

Validation and augmentation use context to recognize personal names, including short names, lowercase or uppercase names, and personal initials with or without a following surname, such as initials that stand in for a person in running text. Short name values in structured fields (for example, `"surname": "Ng"`) are recognized the same way. A letter used as a variable, grade, unit, or list marker is not a personal name. Initials are never expanded into inferred names; labels still follow your configured label scope.

The label settings serve different purposes:

- `entity_labels` defines **which entity types** are in scope, such as `api_key`.
- `entity_label_examples` provides **representative values** for those types, such as `sk-ant-api03-abc123`.

The following terms distinguish label scope from example origin:

- A **default label** is present in `DEFAULT_ENTITY_LABELS`; a **non-default label** is absent from it.
- An **explicit label set** is any set supplied through `entity_labels` and may contain default labels, non-default labels, or both.
- **Built-in examples** ship with Anonymizer in `ENTITY_LABEL_EXAMPLES`.
- **Configured examples** are values you supply through `entity_label_examples`.

### Label scope

When you pass `entity_labels` explicitly, the augmenter operates in **strict mode** -- only suggestions whose normalized labels match your list are admitted. When `entity_labels=None`, the augmenter can create additional labels beyond the defaults (e.g., `clinic_name`, `server_name`).

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

Every non-excluded label referenced by `entity_label_examples` must appear in the explicit label set. A missing or misspelled label raises a validation error. Because the label set is explicit, the augmenter is strict.

!!! warning "Examples are sent to model providers"
    Configured examples are embedded in prompts and exported detection builders. Use synthetic patterns, not production credentials, secrets, or real PII. Explicitly enabled raw DataDesigner message traces also contain the rendered prompts.

Configured examples affect detection only; they do not guide substitution or evaluation.

The validator receives the full resolved examples: built-in plus configured examples for default labels, and configured examples for non-default labels. The augmenter receives all active label names but only configured examples, which show it the intended value shape beyond the label name alone. Built-in examples are omitted from the augmenter to limit prompt growth. Keep configured lists short because the validator's full example set is repeated for each validation chunk.

`entity_label_examples` is currently configured through the Python `Detect` API; the CLI does not provide a mapping syntax for this field.

When injecting a custom `detection_workflow` into `Anonymizer`, older methods without
an `entity_label_examples` keyword continue to work when no examples are configured.
If examples are configured, `run()`, `preview()`, and the detection export methods raise
`InvalidConfigError` when the corresponding detector method cannot accept them. Update
the custom method to accept the keyword or remove `Detect.entity_label_examples`.

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
