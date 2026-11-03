<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Rebuild regex detection on PR #305

## Goal

Retain PR #265's public regex configuration, bounded matcher, local validators,
and exported-workflow support while making PR #305's observation admission and
candidate reconciliation the only detection lifecycle. The old PR head is
preserved at `39c315151a3bb76d9988d296905f7b8b9e5bbad9`.

## Integration

- Keep the regex matcher and public configuration, including built-in rules,
  custom validators, contextual-validation routing, and regex-only labels.
- Convert each valid regex match into a source-bound observation. Preserve
  rule identity and independent direct/LLM admission routes before coalescing.
- Combine regex and GLiNER observations before request inventories are built.
  Only LLM-route observations enter validator requests; direct matches still
  pass the same explicit effective-label policy.
- Preserve #305's occurrence-specific semantics: regex matches do not grant
  document-wide copying, and old propagation fields/helpers are not restored.
- Require accepted, unchanged-label regex evidence at the exact span for any
  regex-only label, including model reclassifications. Apply this gate before
  reconciliation, without re-reading validation decisions.
- Keep #305's augmentation grounding, trace setup, overlap handling, final
  publication, and source priority unchanged.
- Port tests by contract rather than copying tests for removed implementations.
  Keep existing regex guard/configuration/serialization coverage.

## Validation and delivery

Use synthetic offline tests for independent admission routes, overlapping
matches, missing/drop/reclass decisions, excluded labels without inventories,
regex-only evidence, source provenance, exported builders and all regex modes.
Run the matcher/config tests, detection lifecycle and interface tests, then the
full suite, lint and type checks. No new live inference is needed for integration.
Do not include native schema enforcement or PR #267 prompt changes. Review the
diff against #305 before replacing #265's branch; retain the backup and use a
lease for any eventual history rewrite.

## Remaining review fixes

- Recover built-in payment cards at complete digit-group boundaries when a
  trailing numeric code makes the longest candidate fail Luhn; never truncate
  inside a contiguous digit run, and prefer a valid full-length candidate.
- Trim URL punctuation in linear time with one delimiter count and one final
  slice, preserving existing balanced-delimiter behavior.
- Validate the runtime types of local validator result fields without coercion.
- Apply exportability checks only to enabled rules in the effective label scope.
- Explicitly reject fuzzy matching constructs, including nested forms, with an
  unsupported-syntax error; document this conservative width-analysis boundary.

Add failing regressions first, including both export builders and both regex
admission routes. Run all tests and repository checks after the fixes, and use
an offline URL scaling probe rather than timing-sensitive CI assertions.

The review regressions reproduced 23 failures before the implementation fixes.
The URL probe with 4k/8k/16k/32k trailing punctuation pairs took median
0.69/1.34/2.68/5.41 ms locally, consistent with linear scaling. These are offline
diagnostics, not performance guarantees or CI thresholds. End-to-end preview
and create tests also verify that the recovered card is redacted without
consuming its trailing numeric code; unexpected model calls fail those tests.

## Network URL coverage

Extend the built-in URL recognizer to HTTP(S), FTP(S), SFTP, and WS(S), using
one scheme allowlist for extraction and local validation. Accept well-formed
single-label internal hosts when an explicit scheme makes the address
unambiguous. Preserve source spans, existing punctuation behavior, custom-rule
ownership, timeout limits, and port/IP/DNS structural checks. No DNS lookup or
HTTP verification is performed. Django's scheme/structure separation informs
this design; neither its code nor Presidio's TLD-list regex is copied.

Keep bare domains, scheme-relative references, escaped JSON representations,
file URLs and cloud-storage locators outside this bounded patch. Document that
regex-only URL mode cannot rely on models to fill unsupported formats.

Add failing-before regressions for every scheme, credentials, Unicode and IP
hosts, internal hosts, prose boundaries, unsupported-prefix salvage and malformed
addresses. Verify both direct admission and the default validation route, then
run the full offline suite and repository checks. The user requested stopping
the old live benchmarks, preserving their artifacts, and rerunning all eight
frozen workloads for Luna, Super and Gemma with this patch and the timeout fix.
