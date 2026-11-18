<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Hybrid Regex Detection

## Scope

[Issue #262](https://github.com/NVIDIA-NeMo/Anonymizer/issues/262) covers the
detection foundation delivered in [PR #305](https://github.com/NVIDIA-NeMo/Anonymizer/pull/305)
and regex recognition in [PR #265](https://github.com/NVIDIA-NeMo/Anonymizer/pull/265).
This consolidated design replaces the separate foundation, integration, GLiNER
recall, resource-limit, and evaluation-scope plans. It describes the contract,
not a claim that every review or quality gate is complete. See the
[detection guide](../../docs/concepts/detection.md) for user-facing details.

Provide built-in and custom deterministic recognition alongside GLiNER, with one
shared admission/reconciliation lifecycle and equivalent local/exported behavior.
The later graph/SDK implementation should preserve these semantics rather than
introduce a second detection or merge policy.

## Configuration and scope

```python
from anonymizer import AnonymizerConfig, Detect, Redact, RegexRule

config = AnonymizerConfig(
    detect=Detect(
        regex_rules=[RegexRule(label="ticket", pattern=r"TICKET-[0-9]{8}")],
    ),
    replace=Redact(),
)
```

- Built-ins are enabled by default, only for labels in the effective scope.
  `builtin_regexes=False` disables built-ins, not custom rules.
- With `entity_labels=None`, start from defaults and add enabled custom labels.
  An explicit label list is authoritative; enabled custom labels must belong to it.
  Apply normalized exclusions last using `resolve_effective_detection_labels()`.
- `regex_rules` holds custom `RegexRule` entries and per-label `BuiltinRegex`
  overrides. `BUILTIN_REGEX_LABELS` exposes the built-in labels.
- `validate_matches_with_llm` selects contextual validation or direct acceptance
  after local checks. `detect_additional_matches` allows or excludes additional
  model-origin claims for that label. Both default to `True`.
- A regex-only label requires accepted, unchanged-label regex evidence at the
  exact span. Reclassification, fallback, and another occurrence cannot grant it.

Export the public regex types from `anonymizer`; keep configuration, documentation,
and the bundled agent skill consistent.

## Shared detection lifecycle

```text
GLiNER observations + locally valid regex observations
  -> request-scoped validation of LLM-route observations
  -> admission and label/route eligibility
  -> independently grounded augmentation
  -> exact-candidate coalescing
  -> one final overlap reconciliation
  -> final entities and downstream replacement/rewrite
```

The foundation corrects decision replay, premature overlap suppression, and lost
provenance. Preserve these invariants:

1. **Canonical evidence:** values equal `text[start:end]`. Half-open Unicode
   character offsets refer to a canonical record field (`SourceTextRef`);
   `SourceView` translates view-relative offsets into that source.
2. **Separate identities:** immutable result/observation references preserve
   lineage, request-local aliases correlate validation responses, and normalized
   source/label/bounds identify equivalent candidates. Reclassification does not
   change observation identity. Serialized lifecycle state is frozen and strict.
3. **Independent admission:** retain overlaps and duplicates until each completes
   its configured route. Consume each decision once, against its dispatched
   inventory. A detector drop cannot veto independent direct regex acceptance or
   grounded augmentation. Rejected origins never contribute accepted provenance.
4. **Fallback is not acceptance:** an omitted/unusable decision in a successful,
   semantically valid response may protect the original eligible span only.
   Terminal request failures remain `FailedRecord`s, not implicit acceptance.
5. **Eligibility before suppression:** allowed/excluded labels and route restrictions
   apply before coalescing or overlap selection, including direct-only rows.
6. **Occurrence-bound protection:** no global value propagation or name splitting.
   An accepted `full_name` cannot authorize standalone name parts elsewhere.
   Every protected occurrence requires source-bound evidence.

### Validation, augmentation, and final selection

Connected overlapping LLM-route candidates form atomic validation groups. Render
each region once with neutral `NAGROUP_gN` markup and an authoritative inventory.
Pack whole groups toward `validation_max_entities_per_call` (default 100); send
oversized groups alone with a value-free warning. Unknown/wrong-request aliases
have no authority. Duplicate assigned aliases invalidate the entire response;
bounded retries retain the same inventory, aliases, context, and policy. Exhausted
attempts fail the record rather than applying partial or arbitrary decisions.

Augmentation admission is independent of seed-aware versus plain-text prompting.
Ground each suggestion against copied source value and adjacent context; never
expand by value alone. Retain grounded items while retrying unsuccessful suggestions
once, then drop remaining ungroundable items. Transport/schema failures remain
record failures. Prompt changes such as [PR #267](https://github.com/NVIDIA-NeMo/Anonymizer/pull/267)
need separate quality evidence; they do not redefine admission.

After admission, coalesce exact source/normalized-label/bounds identities, retaining
accepted origins and fallback lineage separately. Run one deterministic greedy
overlap selection: longer spans first, then earlier positions. At identical bounds,
explicit acceptance outranks fallback, then
`regex_user > regex_builtin > initial_detector > augmenter`. Compare native scores
only within their calibration domain; use stable label/lineage tie-breakers.
Touching intervals survive. Partial overlaps do not create union spans, and this
is not an optimizer for maximum aggregate covered characters.

## Built-ins and validators

| Label | Recognition and local checks |
| --- | --- |
| `email` | Conservative Unicode-aware address syntax and normalized IDNA domains; search at each `@` using the complete local token, without artificial truncation windows. |
| `url` | HTTP(S), FTP(S), SFTP, WS(S), and `www.`; scheme/host/port validation and balance-aware, linear-time punctuation trimming. No network verification. |
| `ipv4` | Complete tokens validated by `IPv4Address`. |
| `ipv6` | IPv6 including dotted IPv4 tails, validated by `IPv6Address`; Unicode boundaries prevent matches inside identifiers such as `std::vector`. |
| `mac_address` | Six hexadecimal octets with consistent supported separators. |
| `credit_debit_card` | 13–19 contiguous digits, consistently space/hyphen-separated 4-4-4-n (n=1–7), or 4-6-4/5 layouts; Luhn and repeated-digit rejection. |

URL candidates stop at backticks and literal backslashes, preserving Markdown
wrappers and JSON quote escapes. Percent-encoded backslashes remain supported.
A standalone case-insensitive `url:` prefix is allowed without whitespace, not
arbitrary colon prefixes. Bare domains, slash-escaped schemes (`https:\/\/`), file
URLs, and cloud-storage schemes remain outside built-in scope. Custom rules can
broaden recognition; regex-only mode cannot rely on model recovery.

Only the built-in card matcher searches overlapping starts, recovering a card
after a leading numeric ID without starting inside a contiguous digit run.
Explicit layouts replace prefix trimming and exclude adjacent codes/expiry dates.
Luhn does not prove a number is a payment card. A custom rule can support irregular
layouts with the exportable `validator="nemo-anonymizer.credit-card-luhn.v1"`.

Custom validators receive `RegexCandidate` (including named groups) and return
`bool` or `RegexValidationResult`; reject malformed return types and asynchronous
validators. Direct callables use process-local object identity. Exports require
unique stable names from trusted installed `nemo_anonymizer.regex_validators`
entry points. Check exportability only for active, in-scope rules. Validator code
is trusted code, not sandboxed configuration.

## Resource limits and failures

Use the timeout-capable `regex` engine with non-concurrent scans. Validate syntax,
pattern length, and non-empty match width at configuration time; reject unsupported
fuzzy constructs and retain a runtime zero-width guard.

`Detect.regex_limits=RegexLimits(...)` applies to built-in and custom rules:

- `max_matches_per_rule=10_000` locally accepted candidates before LLM validation,
  not rejected raw matches or final reconciled entities;
- `min_timeout_seconds=0.05`;
- `timeout_seconds_per_million_characters=1.0`.

The per-rule/per-record search budget, in seconds, is:

```text
max(min_timeout_seconds,
    len(text) / 1_000_000 * timeout_seconds_per_million_characters)
```

Length means Unicode code points, not bytes or model tokens. Collect bounded
batches under one cumulative search-time budget; Python validators and time between
batches are excluded. This does not bound arbitrary callbacks. Repeated cards can
produce Luhn-valid overlapping rotations, so candidate counts can exceed card counts.

Timeouts, cap exhaustion, zero-width matches, and validator exceptions fail the
record without returning partial regex results. Diagnostics identify the rule,
limit, and exact configuration knob without source text. Public failed-record
reasons use fixed, sanitized guidance, not arbitrary validator/trace text.

## Execution and evaluation integration

Declare recognition as a serializable `anonymizer-regex-detection` workflow column;
compile patterns on workers and execute through `NddAdapter`, not interface-only
preprocessing. Forward rules, routes, limits, and provenance through run, preview,
and both exports. Serialize pattern strings and stable validator names, not
compiled objects or Python callables.

Injected workflows receive optional regex keywords when supported. Unsupported
implicit defaults retain existing behavior; explicit regex requests or non-default
limits they cannot honor raise `InvalidConfigError`. Before adding further detector
backends/alternate models, replace these compatibility shims with a documented
detector/options contract and deliberate breaking-change migration guidance.

Persist `effective_entity_labels` on run/preview results and forward it through
replace/rewrite evaluation into coverage taxonomy guidance. Preserve the original
`entity_labels` as scope policy: `None` remains permissive, the taxonomy is explicitly
non-exhaustive, and novel non-excluded categories remain eligible. Explicit lists
stay strict; old/manual results without this metadata keep default guidance.

GLiNER recall remains a foundation concern: retain threshold 0.3, native word
windows (128 words, 32 overlap), overlapping alternatives, and canonical offsets.
Preserve explicit window settings; otherwise let the server choose defaults.
Translate equivalent labels only at the native server boundary and restore requested
labels on return. Older separately managed servers require their own update;
client changes cannot recover candidates discarded during inference.

## Validation and delivery

Use synthetic, behavior-focused tests for configuration/normalization, active scope,
public exports and JSON parity; built-in valid/invalid values and exact spans through
redaction; independent admission, fallback, reclassification and deterministic
overlaps; bounded matching and sanitized failures; and run/preview/evaluation scope
forwarding. Cover native GLiNER window boundaries, source offsets, and label mapping.

Run focused suites, full unit tests for behavioral changes, `make check`, and strict
docs validation where relevant. Use offline scaling probes rather than fragile
wall-clock CI thresholds. Keep docs/skill source aligned and regenerate/sign skill
artifacts through trusted CI.

Quality evidence requires fixed records, models/providers/settings, and independent
annotations for repeated mentions, standalone name parts, and ambiguous homographs.
Report missed/exposed mentions, false positives, record/schema/grounding failures,
candidate counts, tokens, and latency by dataset/model. Distinguish seed rediscovery
from new recovery, and measure augmentation rediscovery of validator-dropped claims.

Removing propagation can reduce false positives but lose repeated-mention coverage.
Targeted diagnostics and incomplete known-target annotations do not establish
precision, exhaustive recall, or safety of that removal. Historical benchmark targets
and exploratory timings are not release guarantees. This plan does not authorize
experiment launches, model changes, or benchmark reruns.
