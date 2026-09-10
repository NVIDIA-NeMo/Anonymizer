<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Phase 6 design — anchored mentions, resolution, and local verification

Status: reviewed phase-specific design and test strategy; private branch-local implementation
and hardening landed on 2026-08-27 in `5bf61c6` through `29bddad`. This document is
subordinate to the complete development and research plan in the
[graph-native SDK RFC](graph-native-anonymizer-sdk-rfc.md), the
[phase 4 terminal-accounting design](phase-4-hierarchical-terminal-accounting-design.md),
and the [phase 5 target/context design](phase-5-target-context-workframe-design.md). RFC
acceptance remains the project decision on the plan. The branch-local implementation does
not authorize a public graph or session API, production Intake or OpenShell integration, or
stable promotion.

Review status: independent architecture and test-strategy council review completed on
2026-08-20 with zero unresolved Critical or Warning findings. Focused re-review after the
Phase 5 ownership-boundary revision also completed with zero unresolved Critical or Warning
findings. The design review is complete, and the branch now contains the Phase 6
implementation, frozen independent reference model, and focused mention, resolution, role,
Redact, backend, lifecycle, privacy, and compatibility tests. The 2026-08-31 PR checks pass;
this evidence remains private to the branch and does not establish Substitute, product, or
publication support.

## Decision

Phase 6 should convert untrusted detection evidence into immutable mentions anchored only to
authoritative target datum offsets, resolve those mentions into deterministic clusters from
explicit typed same-subject evidence, classify their replacement roles through a versioned
closed policy, and qualify one private Redact profile through exact mention-keyed patch and
atomic-group verification.

Text equality, label equality, source identity, DataFrame position, context membership, and
model confidence never establish mention or cluster identity. Every mention starts in a
singleton cluster. A merge occurs only when accepted same-subject evidence names exact
mention tokens; the deterministic resolver computes the same clusters from the same evidence
regardless of row, response, or declaration order.

The first local profile is Redact only. Public Redact, Annotate, Hash, and Substitute retain
their current DataFrame behavior. Adding a private Annotate, Hash, Substitute, or Rewrite
profile requires its own reviewed release predicate and capability.

## Preconditions and non-goals

Phase 6 assumes:

- phase 4 supplies exhaustive task, datum, dependency, group, stage, and invocation
  accounting plus cancellation, loss, reconciliation, and release;
- phase 5 supplies immutable target identity and separately framed declared context;
- the graph compiler has accepted the target, dependency, atomic-group, context, and
  capability declarations before phase-6-owned effects; and
- every DataDesigner workflow executes through `NddAdapter.run_workflow()`.

Phase 6 does not:

- claim detection is exhaustive or output contains no PII;
- anchor a mention to context-only text;
- infer aliases from equal or normalized content;
- create replacement slots, replacement maps, synthetic values, or a planning ledger;
- qualify public or private Substitute, grouped Rewrite, evaluation, or repair;
- move codecs, source identity, field selection, reconstruction, persistence, retries,
  deduplication, cleanup, retention, or delivery into Anonymizer;
- select an Intake field policy, source commit unit, or privacy boundary; or
- make the private graph, mentions, clusters, evidence, or roles serializable or public.

## Phase boundaries and typed graph values

Phase 6 makes the proposal's detection and resolution phases concrete:

```text
context-framed ProtectionGraph
  -> DetectedGraph  # accepted target-anchored mentions
  -> ResolvedGraph  # deterministic clusters, evidence, and role results
  -> TransformedGraph  # Redact patches applied once by mention identity
  -> VerifiedGraph  # datum and atomic-group predicates ready for phase 4 release
```

Each graph phase is an immutable, closed, private value. The executor consumes only the
immediately preceding phase and must not reread the original authoring graph or unvalidated
backend payload after hydration.

The private model contains:

```text
MentionId
AnchoredMention(
  id,
  target_datum_id,
  start,
  end,
  exact_source_slice,
  detector_label,
  provenance_kind,
)

ClusterId
SameSubjectEvidence(left_mention_id, right_mention_id, kind, version)
DistinctSubjectEvidence(left_mention_id, right_mention_id, kind, version)
EntityCluster(id, ordered_mention_ids, accepted_evidence)

ReplacementRoleResult = classified(role, policy_version) | unsupported(reason_code)
ResolvedMention(mention, cluster_id, role_result)
```

All IDs are compiler- or executor-issued opaque graph-scoped values. They are never source
IDs, legacy entity IDs, model-returned IDs, content hashes, labels, offsets encoded as a
string, or DataFrame values. Exact source slices are private evidence, not identity.

## Target-anchored mention contract

Offsets use Python string character indexes and half-open intervals `[start, end)`. A
candidate mention is accepted only when:

1. its owning target task and datum are known and current;
2. `start` and `end` are integers with `0 <= start < end <= len(target_text)`;
3. `target_text[start:end]` exactly equals the returned source slice;
4. its detector label and closed provenance kind are non-empty and supported;
5. its target-attribution token is current and unambiguous; and
6. it does not overlap another accepted mention in the same target datum.

Context frames may inform a reviewed candidate, validation, augmentation, or resolution task,
but every accepted mention must still point to an exact target slice. A value found only in
context is not a target mention. A model-returned value without offsets is insufficient for
the private graph profile.

Exact duplicate evidence for one `(datum, start, end, label, provenance)` tuple may collapse
after candidate finalization when its complete authoritative tuple and lineage are
byte-equivalent. Different final labels, slices, provenance, attribution, or terminal
decisions for one span are contradictory. Partial overlap, strict containment, duplicate
non-identical final evidence, source-slice mismatch, and an offset into context close the
affected target task as `inconsistent` or `failed` according to the phase 4 attribution
rule; no heuristic tie-break is permitted.

The current public detector may continue its existing value-based augmentation, occurrence
expansion, overlap resolution, and legacy entity IDs. The private graph profile must use a
separate keyed schema with explicit offsets. It must not import `group_entities_by_value()`
or `expand_entity_occurrences()` as graph identity or clustering rules.

## Detection, validation, and context use

The first phase 6 profile keeps detector and target boundaries explicit:

- GLiNER or another span detector receives target text only and returns target offsets.
- Validation may receive the target candidate plus the phase 5 compiled context frame.
- Augmentation may receive target and compiled context, but it must return exact target
  offsets and a source slice; value-only suggestions fail graph hydration.
- A resolution task may receive accepted mention tokens, bounded target excerpts, and
  separately compiled context. It returns only closed pairwise evidence keyed by mention
  tokens.

Candidate refinement has one closed lineage. Detector and augmenter records are provisional,
target-task-keyed candidate evidence. After exact offset and source-slice validation, the
executor issues a fresh candidate token. A validator returns exactly one of `keep`,
`reclass(label)`, or `drop` for that token. `keep` creates one final candidate with the
provisional label, `reclass` creates one with the accepted validated label, and `drop`
creates none. Missing, duplicate, foreign, stale, or contradictory terminal decisions are
not defaults. Only finalized candidates enter duplicate collapse, overlap validation, and
mention ID issuance. Competing final candidates without one valid lineage are inconsistent;
legitimate reclassification is not contradictory evidence.

Each effectful step is a compiler-owned semantic stage or task accounted by the phase 4
ledger. DataDesigner columns may declare the model operations, but a DataDesigner column,
row, or `FailedRecord.step` is not a semantic task. The compiler freezes which steps exist
for the selected profile.

No-context execution must preserve the public compatibility workflow. The private graph
profile may use stricter schemas and fail-closed behavior without changing the public trace
or legacy detection path.

## Resolver task cardinality and readiness

The first profile compiles exactly one phase 4 resolver logical task for each target datum.
Its eligible endpoint set contains the finalized mentions owned by that target plus the
finalized mentions of target datums explicitly named in its compiled phase 5 context
scope. Context-only datums may inform the resolver request, but they never supply a mention
endpoint. Every returned edge must have at least one endpoint owned by the resolver task's
target; its other endpoint must belong to the eligible set.

The compiler enumerates the resolver's exact input bindings and mention-finalization
predecessors. A resolver becomes ready only after mention-finalization tasks for its owner
and every referenced target datum have terminated successfully. All mention finalization is
therefore a stage before resolution. A read-only phase 5 context cycle creates symmetric
resolver predecessors after that stage and cannot create a scheduling cycle. These are
phase 6 task predecessors, not phase 4 datum dependencies, and do not change release
propagation.

A failed, cancelled, lost, inconsistent, or blocked mention-finalization predecessor blocks
the resolver. Phase 6 neither dispatches it nor substitutes singleton clusters for requested
but incomplete resolution. After every dispatched resolver task terminates, a deterministic
reducer combines the evidence. Byte-identical duplicate evidence for one ordered endpoint
pair and kind may collapse. Conflicting `same_subject` and `distinct_subject` evidence for
one pair, or any contradiction that cannot be localized to the involved targets, follows the
fail-closed attribution rules below. There is no invocation-global or atomic-group resolver
task; phase 4 accounts for exactly one resolver task per owning target.

## Deterministic clustering from explicit evidence

Every mention begins in a singleton cluster. Phase 6 accepts a versioned closed evidence
grammar:

```text
same_subject     — the two mentions refer to one semantic subject
distinct_subject — the two mentions must not be in one cluster
```

Evidence in the first profile comes only from a separately accounted invocation-private
resolver task that receives mention tokens after finalization. The evidence origin is a
closed private provenance kind. Source adapters cannot name or observe mention tokens and do
not author alias evidence. A later declaration path would require a separate private
post-detection binding design; content or source-ID joins are never an alternative.

The resolver:

1. validates every evidence endpoint against current mention IDs;
2. rejects self-edges, duplicates, foreign or stale tokens, and unsupported evidence
   versions;
3. sorts accepted `same_subject` edges by opaque mention declaration position only as a
   deterministic implementation order;
4. computes connected components with deterministic union-find;
5. rejects the evidence set if both endpoints of any `distinct_subject` edge belong to the
   same component after all `same_subject` unions; a distinct edge whose endpoints remain in
   different components is valid separation evidence; and
6. issues a fresh opaque cluster ID for each resulting component.

Mention declaration position affects only deterministic presentation. Renaming opaque IDs or
permuting evidence and workframe rows yields an isomorphic cluster graph. Equal text and
labels remain separate without an accepted edge. An unresolved or absent edge means separate
clusters, not guessed sameness.

Phase 6 does not own or validate coherence-scope membership. It emits cluster membership in
mention and datum identities only. Phase 7 validates those clusters against its compiled
flat coherence partition and rejects any cross-scope cluster at its documented admission
tier. Context, dependencies, atomic groups, and phase 6 clusters remain independent.

## Replacement-role classification

Detector labels and replacement roles are separate axes. A versioned closed policy maps an
accepted detector label and allowed private profile to either:

```text
classified(replacement_role, policy_version)
unsupported(reason_code)
```

An unsupported role does not invalidate a mention for Redact: Redact needs only the
authoritative span and a local patch. It does block any later phase 7 Substitute task that
requires a type-appropriate slot. Phase 7 may rely only on `classified` roles from the exact
policy version named by its frozen semantic contract.

For the Redact-only phase 6 profile, the Anonymizer semantic owner freezes only a versioned
label/provenance admission policy and the structural `classified | unsupported` result
grammar. Phase 6 carries that result without inventing a generic role, copying an arbitrary
custom label into the role grammar, or treating `unsupported` as a silent default.

Phase 7 separately freezes and selects the broader replacement-role and relational-constraint
vocabulary, distinct-slot matrix, host ceilings, and cleanup-observability contract described
in its branch decision record. It validates compatibility with the Phase 6 result version
at admission. Phase 6 does not freeze that phase 7 contract early.

A cluster represents one semantic subject; a replacement role describes one mention's
replacement function. Phase 6 creates no slot. One cluster may therefore contain mentions
with several classified roles, which phase 7 may later map to distinct type-appropriate
slots.

## Private Redact transformation profile

The first qualified transform is deterministic local Redact. After mention finalization, a
pure phase refinement creates an immutable patch manifest with exactly one expected entry per
accepted mention and no invocation token or replacement payload. After the phase 4 invocation
opens, the executor binds every manifest entry exactly once to a fresh mention-private patch
token and materializes the closed Redact operation. Rejected compilation or failed mention
refinement creates neither patch tokens nor patch workframes.

Each runtime patch contains:

- the owning target-task token;
- the exact authoritative `[start, end)` interval;
- a closed Redact replacement value or operation selected by the reviewed profile; and
- no source, graph, cluster, or public identifier.

The transform validates the complete patch set before application. It requires one patch per
mention and no missing, duplicate, foreign, stale, cross-target, or extra patch. Patches are
applied once in ascending source offsets by copying untouched source intervals and inserting
the closed replacement. Application never searches the evolving output and never falls back
to `(value, label)` or value-only lookup.

The public offset replacement primitive may be reused only below a private adapter that
disables value fallback and proves exact mention-token coverage. Its current legacy
value/label map and unambiguous value fallback remain public compatibility behavior, not the
graph contract.

Private Annotate is not qualified because its intended output retains the source value and
needs a different release claim. Private Hash is not qualified because content-derived
output, normalization, collision, and leakage policy require separate review. Substitute
remains phase 7.

## Datum and atomic-group verification

A target datum is locally qualified for the private Redact profile only when:

1. its detection and resolution tasks closed successfully;
2. every accepted mention has one valid authoritative span and exactly one patch;
3. the patch set is non-overlapping and applied exactly once without skips;
4. the returned text exactly equals the output reconstructed from authoritative untouched
   source intervals and the accepted patch set;
5. no replacement payload contains its protected source slice; and
6. the datum-level strategy predicate passes.

A target with zero accepted final mentions requires an empty patch manifest, zero patch
tokens and applications, and output exactly equal to its input. It still closes through the
normal verified no-work datum and atomic-group predicates. Unchanged input is never used as
a fallback for a failed mention-bearing target.

Exact reconstruction is the primary proof. A global substring search is not: the same
source text may legitimately occur at an unprotected location. Tests may retain substring
checks as a conservative secondary canary, but they must not replace mention conservation.

The group predicate verifies that every member target has one exact locally qualified
result, every expected mention and patch belongs to exactly one member, and no context-only
datum, partial result, token, or raw fallback enters the output. Phase 4 then applies
dependency and atomic-group fixed-point withholding and remains the only release authority.

A localizable target verification failure withholds its atomic group and explicit dependent
closure. Missing or contradictory evidence that destroys global attribution closes the
invocation as `inconsistent` and withholds all groups. A successful cluster or patch never
overrides a phase 4 cancellation, loss, or embargo.

## Failure, cancellation, retry, and cleanup

Phase 6 uses phase 4 terminal outcomes and precedence. A known detector, resolver,
classification, transform, or predicate error maps to `failed`. Trusted missing keyed output
may be localized as `inconsistent(missing)`. Foreign, duplicate, swapped, stale-at-admission,
or contradictory evidence that prevents attribution is `inconsistent`. Dispatch without a
trusted terminal or stop record is `lost`.

The phase 6 task and fault mapping is closed:

| Semantic task or evidence | Local terminal result | Release effect |
| --- | --- | --- |
| detector/augmenter attempt returns one attributable known failure | task `failed` | owning target and phase 4 closure withheld |
| validator raises or returns one attributable known failure | task `failed` | owning target and phase 4 closure withheld |
| validator returns `drop` for one current candidate | task success; no final candidate | no failure by itself |
| validator returns one current `keep` or `reclass` | task success; one final candidate | continue mention refinement |
| validator omits, duplicates, or contradicts one expected decision | task or invocation `inconsistent` | localize when ownership remains provable; otherwise global embargo |
| finalization receives contradictory candidate lineage | task or invocation `inconsistent` | localize when ownership remains provable; otherwise global embargo |
| finalization finds one attributable invalid span, overlap, or source-slice mismatch | task `failed` | owning target and phase 4 closure withheld |
| resolver returns one valid complete evidence set | task success | publish immutable clusters privately |
| known role-policy or transform failure | task `failed` | owning target and phase 4 closure withheld |
| trusted batch omits one expected current token with all other bijections proven | task `inconsistent(missing)` | localize to owning target |
| foreign, duplicate, swapped, plan-mismatch, or contradictory attribution | invocation `inconsistent` | global release embargo |
| dispatched attempt has no trusted terminal or stop evidence | task or invocation `lost` | affected output withheld |
| datum or group predicate rejects exact accounted evidence | datum `failed(release_predicate_failed)` | group withheld |

Every semantic stage declares its fixed predecessor tasks in the compiled phase 4 plan:
candidate generation precedes validation/finalization, finalization precedes resolution and
role classification, and mention refinement precedes patch transformation and verification.
Missing or blocked predecessors cause the accepted phase 4 `blocked` outcome without
dispatch. Competing terminal evidence follows phase 4 precedence and never rewrites an
accepted terminal record.

Cancellation before dispatch causes no provider or transform call. After dispatch,
cancellation requires trusted stop evidence; otherwise the task is `lost`. Accepted terminal
evidence wins over later cancellation, and accepted cancellation makes later completion
stale. Phase 6 performs no automatic retry.

Before phase 4 release, cleanup closes all mention, evidence, cluster, role, patch, excerpt,
and token stores to mutation and verifies that no partial graph phase is observable. A
publication-critical cleanup failure closes the invocation as `inconsistent` and withholds
all output. Post-acceptance host teardown cannot rewrite an already accepted result. Python
reference release is not secure erasure.

## Privacy and diagnostic boundary

Mention source slices, target/context text, excerpts, prompts, resolver evidence, labels,
roles, clusters, patches, graph IDs, and content-derived hashes are private. Active bounded
workframes may contain only the projection compiled for that task. Withheld and nonterminal
results expose none of them.

Logs, metrics, exceptions, tracebacks, receipts, diagnostic views, cleanup errors, and
serialized artifacts may expose only allowlisted content-free reason codes and bounded
counts. Public `trace_dataframe` and `FailedRecord` retain their current compatibility shape;
phase 6 must not inject mention, cluster, context, or correlation identity into them.

The private rejection grammar is closed and versioned. Its classes are `unknown_target`,
`invalid_offset`, `source_slice_mismatch`, `unsupported_provenance`, `missing_decision`,
`duplicate_decision`, `overlap`, `foreign_token`, `stale_token`, `contradictory_candidate`,
`invalid_evidence`, `evidence_contradiction`, `unsupported_role`, `invalid_patch`, and
`release_predicate_failed`. More specific subcodes may be added only within a reviewed
version. Multiply-invalid candidate evidence follows target attribution, token integrity,
offset bounds, source-slice equality, lineage, duplicate, overlap, evidence, role, then patch
precedence. Reason codes and counts contain no content, label, offset, or identifier.

## Phase 7 handoff contract

For phase 7 admission, phase 6 supplies an immutable complete input containing:

- target-anchored mentions with exact source offsets and closed provenance;
- compiler-issued mention and cluster IDs;
- deterministic cluster membership and accepted evidence version;
- one replacement-role result per mention;
- terminal phase 4 task and target outcomes; and
- content-free bounded reason codes for non-success.

Phase 7 compiles its own flat coherence partition, validates every phase 6 cluster against
that partition, and rejects a cross-scope cluster. It must also reject an unsupported role,
incomplete mention set, unaccepted evidence version, nonterminal predecessor, or missing
group-verification input. It must not repair phase 6 data, regroup by text, or fall back to
row-local planning.

Phase 7 still owns replacement slots, its scope-planning task and ledger, bundle validation,
collision and relational policy, assignments, and Substitute transformation. Phase 6 does
not precompute or expose those values.

## Pure reference model

Build a pure model independent of pandas, DataDesigner, the production resolver, the patch
implementation, and the phase 4 ledger. Its inputs are:

- target texts, compiled phase 5 context frames, dependencies, and atomic groups;
- closed detection, evidence, role-policy, Redact, capability, and limit versions;
- target-keyed detector, validator, augmenter, and resolver observations;
- mention-keyed patch and transformed-output observations; and
- timestamp-free dispatch, `FailedRecord`, exception, cancellation, trusted-stop, loss,
  finalization, and teardown observations.

The model derives accepted mentions, clusters, role results, readiness, reconciliation,
patches, exact transformed output, datum qualification, group verification, terminal
outcomes, and the only legal release set. Production overlap selection, clustering,
classification, patch application, verification, or release decisions are never model
inputs.

The central oracle is:

```text
accepted mention iff
  one current finalized candidate survived exactly one keep or reclass decision
  and its complete lineage is valid
  and it has an exact in-range non-overlapping target source span

same cluster iff
  mentions are connected by accepted same-subject evidence
  and no accepted distinct-subject evidence contradicts the component

qualified target iff
  every required phase-6 task succeeded
  and every accepted mention has one exact applied patch
  and returned output equals authoritative patch reconstruction

released group iff
  every member target remains phase-4 release eligible
  and the phase-6 group predicate passes
  and no invocation embargo applies
```

## Finite conformance envelope

The exhaustive envelope contains:

- one through four target datums with zero through three compiled context frames each;
- a finite symbolic text domain with alphabet classes `{ASCII, multibyte BMP, astral,
  combining, whitespace}` and length classes `{0, 1, exact span, exact byte limit,
  one-over-limit}`; concrete fixtures freeze one representative per class and bound target,
  context, excerpt, label, and source-slice bytes;
- zero through six candidate mentions, with zero through four accepted non-overlapping
  mentions per target;
- zero through six same-subject or distinct-subject evidence edges;
- every evidence graph over up to four accepted mentions, including contradictory
  components;
- zero through three entity clusters and one role result per accepted mention;
- the phase 4 DAGs and flat atomic partitions over target datums;
- one dispatch per required detector, validation, augmentation, or resolver task;
- exactly one resolver task per target, with the compiled eligible endpoint set and every
  required mention-finalization predecessor represented;
- one patch and transform observation per accepted mention/target;
- one primary terminal observation and at most one missing, duplicate, foreign, stale,
  swapped, plan-mismatch, or contradictory observation per attempt; and
- a computed event bound that includes every required task dispatch/terminal pair, candidate
  decision, evidence result, patch, transformation, group verification, cancellation/loss,
  finalization, and teardown observation for the admitted graph.

Canonicalization orders commuting events for independent targets and evidence edges while
retaining every race around dispatch, terminal acceptance, cancellation, transformation,
group verification, finalization, and release. Freeze the model/generator versions, exact
symbolic domain cardinalities, computed maximum event count, exact graph count, exact
canonical trace count, and SHA-256 manifest digest before executor comparison. The checked
generator publishes its machine-readable event alphabet and independence relation; tests
prove commuting schedules collapse while dispatch/terminal, cancellation/terminal,
verification/release, finalization/release, and teardown/acceptance races remain distinct.
Larger seeded state-machine tests supplement the finite envelope.

## Mention, resolution, and role tests

Mention tests cover exact boundaries; empty and whole-string spans; adjacent spans; repeated
equal text; equal labels at different offsets; astral Unicode, combining characters, emoji,
newlines, and mixed scripts under Python character indexing; negative, reversed, zero-width,
and out-of-range offsets; source-slice mismatch; context-only offsets; exact duplicate and
non-identical duplicate evidence; containment and partial overlap; and maximum count/byte
limits.

Independently permute candidate records and workframe rows. The same accepted evidence must
produce isomorphic mentions. A model-returned ID, public record ID, text, label, DataFrame
index, and response position must never satisfy mention correlation.

Resolution tests enumerate singleton, chain, star, diamond, and disconnected evidence
graphs; duplicate and reversed edges; unknown, stale, foreign, and cross-scope endpoints;
self-edges; a direct distinct edge between singleton components; a distinct edge whose
endpoints become transitively same; a valid distinct edge across separate final components;
equal text in distinct clusters; different text in one explicitly evidenced cluster; and
evidence-order permutations. Cluster IDs may change under opaque renaming, but membership
must be isomorphic. Cross-scope validation is exercised in phase 7 admission tests, not as a
phase 6 resolver rule.

Resolver-readiness tests cover empty and non-empty eligible endpoint sets; one target using
another target as context; permitted two-way and three-way context cycles; a context cycle
overlapping a phase 4 dependency edge; every mention-finalization predecessor outcome; an
edge with neither endpoint owned by the resolver target; a context-only endpoint; and
duplicate resolver evidence across owners. They prove that all finalization precedes
resolution, context cycles cannot deadlock, non-success predecessors block without singleton
fallback, and one local resolver failure changes only its attributable phase 4 closure.

Role tests cover every frozen label mapping, every role, unsupported custom labels, policy
version mismatch, missing mapping, declaration permutations, and the difference between
Redact eligibility and phase 7 Substitute readiness. No unknown label becomes a generic role.

The role-policy manifest is a required versioned input artifact with a content-free version
and digest plus positive and negative fixtures for every admitted mapping. Before that
artifact is frozen, tests exercise only the structural `classified | unsupported` grammar,
unknown-version rejection, and fail-closed absence behavior; they do not invent the eventual
phase 7 vocabulary.

## Transformation and verification tests

Generate valid and invalid patch sets across zero, one, adjacent, repeated-text, and maximum
mention cases. Cover missing, duplicate, extra, foreign, stale, swapped, cross-target,
overlapping, out-of-range, wrong-source, wrong-replacement, and wrong-order patch evidence.
The implementation must reconstruct expected output from the original target and accepted
patches; response order and evolving-output search are irrelevant.

Tests prove:

- every accepted mention is applied exactly once;
- adjacent and repeated equal source slices remain distinct;
- replacements do not cascade into later source matching;
- value/label and value-only fallback are impossible in the graph profile;
- a source value repeated at an unprotected location does not cause false success or become
  an implicit patch;
- empty-mention targets retain unchanged text only through a verified no-work path;
- a datum failure withholds the exact atomic/dependency closure; and
- no withheld group exposes protected or raw fallback text.

Group tests cover one scope across several atomic groups, several clusters in one group,
independent disconnected groups, a target used as another target's context, fixed-point
dependency propagation, one localizable patch failure, and one global attribution fault.

## Schedule, fault, and lifecycle tests

Use deterministic barriers around candidate receipt, mention acceptance, context-informed
validation, evidence receipt, cluster publication, role classification, patch construction,
transformation, group verification, release, and cleanup.

Required races include cancellation on both sides of every dispatch and terminal acceptance;
late candidates or evidence after cancellation/loss; duplicate resolver completion; patch
application racing with a contradictory record; one target failure while an independent
group succeeds; invocation cancellation after verification but before release;
publication-critical cleanup failure; and teardown failure after immutable result acceptance.

At least one process-kill test uses a test-only crashable backend at the existing execution
seam. It compares worker death with the pure model's `lost` outcome and empty affected release
set. It does not add a production process API or satisfy the materially different runtime
gate.

## Required properties and mutation tests

Required properties are:

- **Mention conservation:** every canonical finalized candidate equivalence class yields
  exactly one mention. Every dropped or rejected candidate lineage has one bounded reason;
  byte-equivalent duplicate records in one class do not create extra mentions.
- **Anchor integrity:** every mention slice equals its authoritative target substring.
- **Non-overlap:** accepted mention intervals in one target are disjoint.
- **Cluster partition:** every accepted mention belongs to exactly one cluster.
- **Evidence determinism:** the same accepted evidence yields isomorphic clusters under
  declaration and response permutation.
- **Content non-identity:** equal or normalized text and labels do not merge clusters.
- **Role completeness:** every mention has exactly one classified or unsupported result.
- **Patch conservation:** every qualified mention has exactly one applied patch and no other
  patch exists.
- **Exact reconstruction:** released target text equals the pure source-plus-patches result.
- **Non-cascading application:** inserted text is never reconsidered as a source span.
- **Identity invariance:** opaque ID renaming yields an isomorphic result.
- **Monotone withholding:** replacing accepted evidence with non-success cannot enlarge the
  release set.
- **Independent isolation:** a localizable fault cannot change an unrelated component;
  global attribution faults are excluded.
- **Attempt isolation:** stale or foreign evidence cannot satisfy a current task.
- **No partial visibility:** nonterminal, failed, or withheld phases expose no private values.
- **Boundedness and confidentiality:** all retained state and diagnostics respect ceilings
  and allowlists.

Mutation tests must catch value-only mention admission, skipped slice validation, heuristic
overlap selection, text/label clustering, transitive distinct-edge contradiction ignored,
cluster IDs derived from content, unknown role defaulting, positional patch joins, value
fallback, evolving-output replacement, skipped or duplicate patch acceptance, row-local
release before group verification, raw fallback, late-result resurrection, and incomplete
cleanup.

## Compatibility, boundary, and privacy tests

Compatibility tests exercise public `run()`, `preview()`, `evaluate()`, display, validation,
and CLI behavior; duplicate and non-monotonic indices; duplicate text; filtered, reordered,
concatenated, and reset frames; result columns and attributes; `trace_dataframe`; exact
`FailedRecord` ID, step, order, and shape; existing public detection expansion and overlap
behavior; local Redact/Annotate/Hash behavior; legacy Substitute fallback; and no-entity
provider bypass.

Paired tests show that the unchanged public path may retain a legacy value-only or
value-expanded behavior while the private graph profile rejects the same unverifiable input.
Spies prove all DataDesigner execution passes through `NddAdapter.run_workflow()` and no
source-format type enters Anonymizer core.

Privacy tests inject separate high-entropy canaries for target/context text, mention slices,
labels, evidence, roles, graph/source IDs, tokens, prompts, patches, protected output, and
known digests. Structural allowlists inspect private active state, withheld results, public
results, traces, logs, metrics, exceptions and cause/context, tracebacks, receipts,
serialization, and cleanup errors. Substring and digest scans supplement the structural
inspection.

## Ownership and promotion gates

| Decision | Required authority |
| --- | --- |
| Mention grammar, evidence algebra, clustering, roles, and Redact predicates | Project and Anonymizer semantic owner |
| Context semantics, model/provider capability, limits, and cleanup observability | Anonymizer semantic and execution owners |
| Source projection, field policy, reconstruction, retry, and destination checks | Source-adapter and adopter owners |
| Source access, field authorization, privacy boundary, and residual risk | Customer or consuming-product owner |
| Any public mention, cluster, role, graph, receipt, or endpoint | Public-API and Platform owners |

The implemented private Phase 6 profile is governed by these prerequisites, which remain
regression and promotion gates:

1. Phases 4 and 5 are implemented and their evidence qualifies;
2. reviewers accept the exact target-anchor and no-heuristic-overlap rules;
3. reviewers accept singleton-by-default clustering and the closed evidence algebra;
4. the Anonymizer semantic owner freezes the Redact-only phase-6 label/provenance admission
   policy and structural role-result version needed by the selected private profile;
5. reviewers accept the Redact-only patch and group predicates;
6. reviewers accept context use, limits, provider capability, and cleanup behavior;
7. reviewers accept the reference model, exhaustive envelope, mutations, and privacy
   canaries; and
8. reviewers accept unchanged public DataFrame behavior, the NDD boundary, and downstream
   ownership.

Completion of Phase 6 does not authorize the Phase 7 branch checkpoint automatically, Substitute or
Rewrite graph execution, public graph/session/mention APIs, production Intake or OpenShell
support, durable state, an accepted privacy boundary, a `zero PII` claim, or stable promotion.

## Evidence and unresolved gates

Current branch code provides graph mention IDs, exact keyed augmentation spans, typed alias
evidence, deterministic evidence-based clusters, closed role results, mention-keyed Redact
patches, exact reconstruction, and group verification. The frozen Phase 6 reference model and
focused tests cover admission, resolution, role policy, backend evidence, lifecycle, privacy,
and public compatibility. They do not qualify graph Substitute or Rewrite.

The branch freezes the Redact-only label/provenance admission policy and structural
`phase6-role-result/v1` contract. Its intentionally empty Redact role mapping remains
fail-closed rather than inventing Phase 7 roles. Phase 7 separately owns the broader role and
relational contract. Intake field roles, context exposure, source
mappings, atomic commit units, and privacy objectives remain adopter or customer decisions
and do not move into either policy.

The authoritative inputs are the parent
[technical proposal](graph-native-anonymizer-sdk-technical-proposal.md), the reviewed
[phase 4 design](phase-4-hierarchical-terminal-accounting-design.md), the reviewed
[phase 5 design](phase-5-target-context-workframe-design.md), the reviewed future
[phase 7 design](phase-7-stable-substitute-design.md), the separate
[Intake evidence](intake-workload-validation-evidence.md), and the branch-local detection,
replacement, graph runtime, adapter, and private release code at
`88b59e2a4366be09aa7af802fa0a8f81afa8440d`.
