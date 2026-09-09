<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Hybrid Regex Detection

## Status

Implementation plan for
[issue #262](https://github.com/NVIDIA-NeMo/Anonymizer/issues/262). This
document scopes the deterministic regex and validator feature as a focused
slice of the broader
[Multi-Pole Detection](../detection-poles/multi-pole-detection.md) design.

## Summary

Add deterministic entity detection as a second seed source alongside GLiNER.
Anonymizer will provide curated built-in regex rules for structured entity
types and allow users to configure a custom entity label and regex. Built-in
rules may apply trusted local structural checks such as IP parsing or a Luhn
checksum. Each rule controls whether surviving candidates enter the existing
chunked LLM validation path or are accepted deterministically before
augmentation and finalization.

The target user experience is:

```python
from anonymizer import AnonymizerConfig, Detect, Redact, RegexRule

config = AnonymizerConfig(
    detect=Detect(
        entity_labels=["email", "support_case_id"],
        regex_rules=[
            RegexRule(
                label="support_case_id",
                pattern=r"(?<![A-Z0-9])CASE-[0-9]{8}(?![A-Z0-9])",
            )
        ],
    ),
    replace=Redact(),
)
```

Built-in rules are enabled by default and run only when their output label is
in the effective detection label set.

## Motivation

GLiNER and LLM augmentation are valuable for contextual and weakly structured
entities, but several entity types have strong syntactic structure and, in some
cases, deterministic validity checks. Examples include email addresses, URLs,
IP addresses, MAC addresses, and payment-card numbers.

A deterministic seed source can:

- improve recall for well-structured values;
- reduce dependence on model behavior for exact formats;
- provide explainable local rejection of structurally invalid candidates;
- support organization-specific identifiers without retraining a model;
- preserve contextual review by routing candidates through the existing LLM
  validator;
- create a stable recognizer boundary for deterministic detector integrations.

## Goals

1. Detect a conservative initial set of structured entity types locally.
2. Apply parser, checksum, or structural guards to built-in candidates where a
   stable validation algorithm exists.
3. Let users add a custom entity label and regex through serializable public
   configuration.
4. Let users attach a local validator callable and select contextual LLM
   validation independently for each rule.
5. Preserve identical behavior between in-process execution and exported
   DataDesigner workflows for built-ins and registered custom validators.
6. Make candidate provenance and stable rule identity visible in trace artifacts.
7. Fail safely on invalid, pathological, or excessively productive regexes.
8. Preserve current behavior when deterministic built-ins are disabled and no
   custom rules are configured.

## Product Semantics

### Built-In Activation

Built-in rules run only for labels in the effective detection scope:

```python
Detect()
# Uses the default entity labels and activates matching built-in rules.

Detect(entity_labels=["email", "first_name"])
# Activates the built-in email rule, but not unrelated built-ins.

Detect(entity_labels=["first_name"], builtin_regexes=False)
# Runs no built-in regex rules.
```

`builtin_regexes` controls curated built-ins only. User-provided
`regex_rules` remain active when `builtin_regexes=False`.

### Custom Label Semantics

- When `entity_labels=None`, custom regex labels are added to the effective
  label set used for GLiNER and validator/augmenter prompts.
- When `entity_labels` is explicit, each custom regex label must appear in the
  list. Configuration fails if it does not.
- Explicit labels retain the existing strict final-label and augmenter
  semantics.
- A custom regex matches the entire regular-expression match. Named capture
  groups are exposed to custom validators without changing the entity span.

### Relationship With GLiNER

The first release keeps regex-covered labels in the GLiNER request. This gives
model detection a chance to recover valid but unusual formats and provides
disagreement data for measurement. Exact duplicate candidates are removed
before validation.

### Validation Layers

The feature has two independent validation layers:

1. **Local guard:** a deterministic structural, parser, or checksum check for a
   built-in rule or a user-provided validator callable. Invalid candidates are
   rejected before an LLM call.
2. **Contextual validation:** rules with `validate_with_llm=True` pass surviving
   candidates to the existing chunked LLM validator for keep/drop/reclass.

`validate_with_llm` defaults to `True` for every rule. When it is `False`, a
candidate that passes its local validator is accepted directly. With no local
validator, the regex match itself is authoritative.

## Public API

Add a public Pydantic model next to `Detect`:

```python
class RegexRule(BaseModel):
    """A user-defined regex entity candidate rule."""

    label: str
    pattern: str
    validator: RegexValidatorCallable | str | None = None
    enabled: bool = True
    validate_with_llm: bool = True


class BuiltinRegex(BaseModel):
    label: str
    enabled: bool = True
    validate_with_llm: bool = True


class Detect(BaseModel):
    # Existing fields omitted.
    builtin_regexes: bool = True
    regex_rules: list[BuiltinRegex | RegexRule] = Field(default_factory=list)
```

`regex_rules` is the single collection for per-label built-in settings
and custom recognizers. Users can disable one built-in and provide a custom rule
with the same label while the rest of the built-in registry remains active.

`RegexRule` validation must:

- trim and lowercase `label` using the same rules as `entity_labels`;
- reject an empty label or pattern;
- compile the pattern during configuration validation;
- reject patterns that match an empty string;
- preserve the pattern string, not a compiled regex object, for serialization;
- accept a direct callable for local use or a stable installed validator name;
- report the offending label and pattern in configuration errors.

`Detect` validation must:

- reject duplicate rules with the same normalized label and pattern;
- reject custom labels omitted from an explicit `entity_labels` list;
- preserve the current `None` versus explicit-label contract.

Re-export `BuiltinRegex`, `RegexRule`, `RegexCandidate`, and `RegexValidationResult` from
`anonymizer.__init__`. A custom validator accepts `RegexCandidate` and returns
either `bool` or `RegexValidationResult`. Because this changes the public
detection surface, update the bundled agent skill template and detection
documentation in the same release.

Direct callables are the primary Python API. A callable used by an exported
workflow must resolve to a stable name supplied by an installed validator
package on every worker. Whether a decorator should provide optional naming and
version metadata remains a PR review question; it is not required for local
callable use and must not be the primary documented path.

## Built-In Rule Model

Use a richer private rule definition than the public custom-rule model:

```python
class ResolvedRegexRule(BaseModel):
    rule_id: str
    label: str
    pattern: str
    validator_id: str | None = None
    source: Literal["regex_builtin", "regex_user"]
    validate_with_llm: bool = True
```

The built-in rule registry should live under `anonymizer.engine.detection`, not
public configuration. Built-in validator IDs resolve through a closed internal
registry. Direct user callables are registered for in-process runs. Exported
workflows require stable names exposed by an installed validator package through
the `nemo_anonymizer.regex_validators` entry-point group, avoiding arbitrary
imports from serialized configuration.

The initial built-ins are jurisdiction-neutral and do not require a geography
selector.

## Initial Built-In Scope

### Initial Default Candidates

| Output label | Scope | Candidate strategy | Local guard |
| --- | --- | --- | --- |
| `email` | Global-format | Conservative address pattern with safe boundaries | Address syntax validation |
| `url` | Global-format | HTTP(S) or `www.` candidate pattern | Host/scheme validation and trailing-punctuation trimming |
| `ipv4` | Global | IPv4-like token | `ipaddress.IPv4Address` |
| `ipv6` | Global | IPv6-like token | `ipaddress.IPv6Address` |
| `mac_address` | Global | Six hexadecimal octets | Consistent separator and octet validation |
| `credit_debit_card` | International | 13-19 digits with permitted separators | Luhn and repeated-digit rejection |
Use the existing canonical label `credit_debit_card`, not the shorter
`credit_debit` wording used in the source presentation.

## Runtime Representation

Accepted matches become the existing canonical `EntitySpan` shape:

```python
EntitySpan(
    entity_id=f"{label}_{start}_{end}",
    value=text[start:end],
    label=label,
    start_position=start,
    end_position=end,
    score=1.0,
    source=f"regex_builtin:{rule_id}",  # or regex_user:{rule_id}
)
```

The score is source-local evidence, not a calibrated probability across
GLiNER, regex, and augmentation. Source and rule ID are the authoritative
provenance fields for measurement.

The matcher should use a private result shape capable of representing span
normalization and rejection explanations:

```python
@dataclass(frozen=True)
class GuardResult:
    accepted: bool
    start_position: int
    end_position: int
    reason: str | None = None
```

This is needed for rules such as URLs, where trailing punctuation may need to
be removed while preserving exact source offsets.

## Workflow Architecture

The current seed path parses GLiNER output directly into `COL_SEED_ENTITIES`.
Split candidate generation from seed fan-in:

```text
COL_TEXT
  -> COL_RAW_DETECTED                 # GLiNER via LLMTextColumnConfig
  -> COL_GLINER_ENTITIES              # parse GLiNER response

COL_TEXT
  -> COL_REGEX_ENTITIES               # rules requiring LLM validation
  -> COL_REGEX_ACCEPTED_ENTITIES      # rules accepting local validation

COL_GLINER_ENTITIES + COL_REGEX_ENTITIES
  -> COL_SEED_ENTITIES                # deduplicate + resolve overlaps
  -> COL_SEED_VALIDATION_CANDIDATES
  -> COL_VALIDATION_DECISIONS          # existing chunked LLM validation
  -> COL_VALIDATED_SEED_ENTITIES

COL_VALIDATED_SEED_ENTITIES + COL_REGEX_ACCEPTED_ENTITIES
  -> COL_ACCEPTED_SEED_ENTITIES        # source-aware deduplication
  -> COL_AUGMENTED_ENTITIES            # existing LLM augmentation
  -> COL_MERGED_ENTITIES
  -> COL_DETECTED_ENTITIES
  -> COL_FINAL_ENTITIES
```

Add `COL_GLINER_ENTITIES`, `COL_REGEX_ENTITIES`,
`COL_REGEX_ACCEPTED_ENTITIES`, and `COL_ACCEPTED_SEED_ENTITIES` to
`engine/constants.py`. Keep the current public final-entity schema unchanged.

### DataDesigner Plugin

Implement a new serializable workflow column rather than preprocessing the
DataFrame outside `NddAdapter`:

```python
class RegexDetectionConfig(SingleColumnConfig):
    column_type: Literal["anonymizer-regex-detection"]
    rules: list[ResolvedRegexRule]
    timeout_seconds: float
    max_matches_per_rule: int
```

Add an `anonymizer-regex-detection` DataDesigner entry point. The plugin
generator should compile resolved patterns once when it is initialized on the
worker and reuse them across rows. Configuration contains pattern strings,
validation-route flags, and validator IDs. Direct local callables resolve in
the invoking runtime; exported builders require a stable validator name
provided by an installed package.

Keep matching and guard functions pure and independently testable. The plugin
generator should only adapt row input/output and error behavior.

### Workflow Integration Points

Thread the resolved rules through all current construction paths:

- `EntityDetectionWorkflow.detect_and_validate_entities()`;
- `_build_detection_spec()`;
- `build_detection_config()`;
- `build_detection_builder_for_seed()`;
- `EntityDetectionWorkflow.run()`;
- `Anonymizer` run and preview paths;
- exported builder interfaces.

This plumbing is required for local/exported parity and must not be implemented
only in the interface layer.

## Safe Regex Execution

Use a regex engine that supports execution timeouts. The Python standard
library `re` module does not provide a per-match timeout. The third-party
`regex` package is a reasonable candidate and is also used by Presidio, but the
dependency and supported syntax must be reviewed before implementation.

Required safeguards:

1. Compile patterns during Pydantic validation and again when reconstructing a
   worker-side generator.
2. Enforce a per-rule/per-record timeout.
3. Enforce a maximum match count per rule and record.
4. Reject empty-string matches during configuration and ignore them
   defensively at runtime.
5. Bound pattern length and document the supported regex dialect.
6. Avoid silently skipping a rule after timeout or match-limit exhaustion.
7. Convert runtime failures into the normal failed-record path for detection.

Failing the row is safer than treating a timed-out privacy rule as though it
found no sensitive data.

## Merge and Overlap Policy

Add a source-aware `merge_detection_sources()` helper rather than globally
changing `resolve_overlaps()`, because the latter also affects augmentation,
name splitting, and occurrence propagation.

Merge policy:

1. Reject malformed or out-of-bounds spans.
2. Deduplicate identical `(label, start, end)` candidates.
3. For identical boundaries with conflicting labels, prefer:

   ```text
   regex_user > regex_builtin > detector
   ```

4. Resolve remaining partial overlaps with the existing longest-span, then
   earliest-position behavior.
5. Preserve the winning source and rule identity for tracing.
6. Send candidates requiring contextual validation through the normal LLM
   path and merge deterministically accepted candidates afterward.

Explicit user regexes receive highest same-span precedence because they encode
direct user intent. Contextual validation can still drop or reclassify rules
configured with `validate_with_llm=True`. When an exact same-label/span GLiNER
candidate duplicates a rule configured with `False`, deterministic acceptance
is preserved and provenance records both sources.

## Error Semantics

### Configuration Errors

Raise Pydantic validation errors for:

- invalid regex syntax;
- empty or zero-width patterns;
- empty labels;
- duplicate custom rules;
- custom labels missing from explicit `entity_labels`;
- unsupported pattern length or options.

### Runtime Errors

The following conditions should fail the affected row through the existing
detection `FailedRecord` mechanism:

- regex timeout;
- match-count limit exceeded;
- internal validator failure;
- malformed rule configuration reconstructed on a worker.

Do not log full matched sensitive values in warning or error messages. Include
the rule ID, label, text length, and safe exception details.

## Provenance and Measurement

Preserve `COL_REGEX_ENTITIES` in the trace DataFrame and record:

- candidate count by rule ID and source;
- local-guard acceptance and rejection counts;
- exact duplicates against GLiNER;
- overlap losses by source and label;
- LLM keep/drop/reclass counts for regex candidates;
- accepted candidate counts by `validate_with_llm` route;
- final entity counts by source and label;
- local detection duration;
- regex timeout and match-limit failures;
- validation token/call changes relative to baseline.

Avoid logging raw entity values in aggregate telemetry.

The source presentation asks whether deterministic and model detection can run
in parallel. Treat actual scheduler parallelism as a later optimization. The
local regex pass should be small relative to a remote GLiNER call, and
correctness plus distributed portability are more important in the first
release.

## Testing Strategy

### Configuration Tests

- Defaults and `builtin_regexes=False`.
- Label and pattern normalization.
- Invalid syntax and zero-width patterns.
- Duplicate rules.
- Custom labels under `entity_labels=None`.
- Missing custom labels under explicit `entity_labels`.
- Public serialization and re-export.
- `validate_with_llm=True` by default and explicit `False`.
- Direct callable and installed-name validator forms.

### Matcher and Guard Tests

For each built-in rule:

- canonical valid examples;
- international variants supported by the rule;
- invalid checksums and impossible structured values;
- substring and boundary false positives;
- punctuation at document and sentence boundaries;
- mixed case and Unicode surrounding text;
- values directly adjacent to Han characters without whitespace;
- named capture groups passed to user validators;
- boolean and `RegexValidationResult` return values;
- custom validator failures without logging candidate values;
- repeated-digit and test-number cases where relevant;
- large and adversarial input;
- timeout and maximum-match behavior.

### Merge Tests

- GLiNER-only candidate.
- Regex-only candidate.
- Exact duplicate from GLiNER and a built-in rule.
- Exact duplicate whose regex route bypasses LLM validation.
- Exact-span conflict between model, built-in, and user rule.
- Partial overlap, including an IP address inside a URL.
- Stable ordering and IDs.
- Source and rule provenance after deduplication.

### Workflow Tests

- Regex candidates appear in seed validation candidates.
- Deterministically accepted candidates bypass the LLM candidate payload.
- LLM decisions can keep, drop, and reclass regex candidates.
- Locally invalid candidates never reach the LLM validator.
- Augmentation receives tagged, validated regex entities.
- Replace and rewrite modes consume unchanged final schemas.
- No-regex configurations preserve baseline behavior.

### Serialization Tests

- Detection builder JSON contains the new plugin type and resolved rules.
- Reconstructing the builder restores user and built-in patterns.
- No compiled regex or Python callback is serialized.
- Local callable validators resolve in-process.
- Installed validator names reconstruct on workers and missing names fail
  preflight.
- In-process and reconstructed/exported workflows produce equivalent spans.
- Plugin discovery works when `nemo-anonymizer` is installed on a worker.

### Quality Evaluation

Compare current and hybrid detection on representative positive and hard
negative datasets. Report per label:

- precision, recall, and F1;
- regex-only recoveries;
- GLiNER/regex agreement;
- locally rejected candidates;
- validator reversals;
- latency, validation candidate count, and token usage.
- language, script, region, and locale slices where source metadata permits.

Default-enable a built-in rule only when it demonstrates very high precision
(target at least 99% on the agreed evaluation sets), useful recall, and no
material end-to-end privacy regression. Evaluate each rule independently;
passing one rule does not justify enabling the whole registry.

## Documentation

Update:

- `docs/concepts/detection.md` with the hybrid pipeline and configuration;
- the generated API reference for `Detect` and `RegexRule`;
- `skills/anonymizer/SKILL.md` because the public detection surface changes;
- examples showing default built-ins, an explicit label subset, and a custom
  organization identifier;
- security guidance for regex timeouts, match limits, and trusted
  configuration.

Document the distinction between regex-only, locally validated, and
contextually LLM-validated matches, including the effect of setting
`validate_with_llm=False`.

## Implementation Map

Expected files and responsibilities:

| Area | Expected changes |
| --- | --- |
| `config/anonymizer_config.py` | Add `RegexRule`, `Detect` fields, and cross-field validation |
| `anonymizer/__init__.py` | Re-export regex rule and validator public types |
| `engine/constants.py` | Add GLiNER and regex intermediate column constants |
| `engine/detection/regex_rules.py` | Built-in registry, rule resolution, and activation |
| `engine/detection/regex_validators.py` | Built-in validators and custom callable contract |
| `engine/detection/regex_detection.py` | Safe matching and canonical candidate creation |
| `engine/detection/postprocess.py` | Source-aware seed fan-in helper |
| `engine/detection/custom_columns.py` | GLiNER parse/fan-in transform integration |
| `engine/detection/detection_workflow.py` | Add regex column and thread configuration |
| `engine/workflow_columns/detection/` | Add regex config, generator, and plugin |
| `interface/anonymizer.py` | Thread public configuration through all run/export paths |
| `pyproject.toml` | Add regex runtime dependency and DataDesigner plugin entry point |
| `tests/config/` | Public configuration tests |
| `tests/engine/` | Matcher, guards, merge, workflow, and serialization tests |
| `tests/interface/` | Replace/rewrite and preview/run plumbing tests |
| `docs/` and `skills/anonymizer/` | Public documentation and skill updates |

Exact module names may change during implementation, but matching, validation,
rule resolution, workflow adaptation, and merging should remain separate
responsibilities.

## Delivery Plan

### Stage 0: Contract and Measurement

1. Finalize public names (`RegexRule`, `builtin_regexes`, `regex_rules`).
2. Add source-aware baseline measurement needed for comparison.
3. Assemble per-rule positive, negative, and adversarial evaluation fixtures.
4. Confirm dependency and regex-dialect choice.

Exit criteria:

- The public configuration and activation truth table are approved.
- Baseline metrics can attribute current candidates and final entities.
- Evaluation datasets and rule-level success criteria are agreed.

### Stage 1: Safe Custom Regex Path

1. Add `RegexRule` and configuration validation.
2. Implement safe matching, timeouts, and match limits.
3. Add direct callable and installed-name custom validator resolution.
4. Add the serializable DataDesigner regex column.
5. Merge custom candidates with GLiNER seeds according to each rule's
   `validate_with_llm` value.
6. Add local/exported parity and CJK-boundary tests.

Exit criteria:

- A user-provided label and regex work in replace and rewrite modes.
- Invalid and pathological patterns fail safely.
- Exported workflow reconstruction preserves the rule.
- Baseline behavior is unchanged with no rules and built-ins disabled.

### Stage 2: Initial Built-In Registry

1. Add the six benchmark-backed initial rules and pure validators.
2. Add rule provenance and aggregate measurement.
3. Run per-rule precision/recall and performance evaluation.
4. Revise patterns and guards based on hard negatives.
5. Enable only rules that meet their individual quality gates.

Exit criteria:

- Every enabled built-in passes its unit, adversarial, serialization, and
  quality gates.
- Hybrid detection does not materially regress end-to-end privacy outcomes.
- Default behavior and migration impact are documented.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Catastrophic regex runtime | Timeout-capable engine, pattern validation, and match limits |
| False confidence from syntax alone | Default `validate_with_llm=True` and document deterministic bypass |
| Numeric false positives | Conservative patterns, checksums, hard-negative datasets, per-rule rollout |
| Duplicate model and regex candidates | Explicit deduplication and source precedence |
| Exported workflow divergence | Dedicated serializable plugin and reconstruction tests |
| Hidden PII in logs | Aggregate telemetry without raw values |
| Public API growth | Keep one `RegexRule` object for pattern and validation policy |
| Validator code injection | Resolve serialized names only from trusted installed registrations |
| Default behavior change | Per-rule quality gates and staged default-on rollout |

## Decisions Made for Version 1

1. Implement deterministic detection natively rather than depending on
   Presidio.
2. Provide custom `label` + `pattern` with an optional direct validator
   callable or trusted installed validator name.
3. Keep GLiNER active for regex-covered labels.
4. Default `validate_with_llm=True` and allow an explicit per-rule `False`.
5. Use internal named validators for curated built-ins.
6. Fail a row on regex timeout or match-limit exhaustion.
7. Preserve the current public final-entity schema.
8. Activate built-in rules by default only when their labels are in the
   effective detection scope; `Detect(builtin_regexes=False)` opts out.
9. Start with `credit_debit_card`, `email`, `ipv4`, `ipv6`, `mac_address`, and
   `url`.
10. Derive versioned custom rule IDs from normalized label and pattern content,
    so reordering configuration does not change provenance or result ordering.
11. Keep per-label built-in settings and custom recognizers in one
    `regex_rules` collection.

## Open Questions

1. Should an optional decorator attach stable name/version metadata to reusable
   validators, or should installed package registration be the only naming
   mechanism?

## External Design References

- [Presidio recognizers](https://presidio.dataprivacystack.org/analyzer/adding_recognizers/):
  regex patterns, context, confidence, country metadata, and code-based
  validator hooks.
- [Microsoft Purview sensitive information types](https://learn.microsoft.com/en-us/purview/sit-sensitive-information-type-learn-about):
  primary matches, checksums/functions, supporting evidence, proximity, and
  confidence tiers.
- [Google Sensitive Data Protection hotword rules](https://cloud.google.com/sensitive-data-protection/docs/creating-custom-infotypes-likelihood):
  proximity-aware contextual likelihood adjustment.
- [Amazon Macie custom data identifiers](https://docs.aws.amazon.com/macie/latest/user/cdis-options.html):
  regex safety checks, keywords, proximity, exclusions, and pre-deployment
  testing.
- [TruffleHog custom detectors](https://github.com/trufflesecurity/trufflehog/blob/main/pkg/custom_detectors/CUSTOM_DETECTORS.md):
  regex candidates, keywords, exclusions, entropy, validations, and optional
  verification for machine credentials.
