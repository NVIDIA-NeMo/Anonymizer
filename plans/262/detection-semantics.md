<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Detection Candidate Semantics

[GitHub issue #262](https://github.com/NVIDIA-NeMo/Anonymizer/issues/262) is the
tracking issue for this foundational detection work and the subsequent regex
implementation in [PR #265](https://github.com/NVIDIA-NeMo/Anonymizer/pull/265).
No separate detection-semantics issue is required.

## Goal

Define one internal lifecycle for entity candidates before adding another
detection source. The lifecycle must make candidate identity, validation,
eligibility, provenance, propagation, derivation, and overlap reconciliation
explicit and deterministic.

This work should land before the graph/SDK refactor. It defines detection
semantics that a later graph can schedule and compose; it does not introduce a
second orchestration framework.

The intended sequence is:

1. **Completed:** [PR #282](https://github.com/NVIDIA-NeMo/Anonymizer/pull/282)
   makes validated source offsets, rather than detector-echoed text, determine
   entity values.
2. Implement the lifecycle in this plan on `main` without changing the public
   `Detect` API.
3. Rebase [PR #267](https://github.com/NVIDIA-NeMo/Anonymizer/pull/267) so its
   clean-slate augmentation prompt uses the shared augmentation-admission and
   final-reconciliation contracts rather than a route-specific overlap filter.
4. Rebase or reconstruct PR #265 on the resulting internal contracts.
5. Preserve these contracts when detection is later represented as graph/SDK
   stages.

## Why This Is a Separate Change

The graph/SDK work will decide how operations are represented, scheduled, and
composed. It will make a chain such as `regex -> GLiNER -> reconcile` easier to
express, but it does not by itself answer the following questions:

- What identifies an observation after its label is reclassified?
- When has a validation decision been consumed?
- Which origins actually survived their validation route?
- When may one accepted occurrence authorize protection of other occurrences?
- Which candidates are eligible to participate in an overlap conflict?
- How are equivalent observations coalesced without losing accepted provenance?
- What deterministic rule selects among incompatible accepted spans?

Those are detection semantics. Establishing them now fixes existing correctness
issues and gives the graph refactor a stable contract to preserve.

## Current Lifecycle and Failure Modes

The current `main` workflow is approximately:

```text
GLiNER response
  -> parse spans and resolve some overlaps
  -> construct validation candidates
  -> ask the validator for decisions
  -> enrich decisions with candidate values and labels
  -> apply decisions to detector spans
  -> remove excluded detector spans
  -> ask the augmenter for missed entities
  -> materialize augmented values at source offsets
  -> split full names and resolve overlaps
  -> apply the original validation decisions to the merged candidates again
  -> apply allowed/excluded label filters
  -> propagate accepted values to other occurrences and resolve overlaps again
  -> final entities
```

The same logical candidate can therefore be represented by several mutable
records, and destructive conflict resolution occurs at several points.

| Current behavior | Failure it permits | Concrete example | Proposed resolution |
|---|---|---|---|
| IDs are constructed from label and offsets | Reclassification or exact-candidate coalescing can change the representative identity used to find a decision | `Seattle` is detected as `country_0_7` and reclassified as `city`; its ID still encodes `country`, while a later reconstructed `city_0_7` cannot correlate to the original decision. | Give every original observation an immutable source-batch reference independent of mutable semantic fields, then use short request-local aliases only for validation. |
| The same decision payload is applied before and after augmentation | A decision can be replayed against a different merged or derived candidate that happens to carry the same constructed ID | The validator decides on detector candidate `token_10_20`; after augmentation and merging, the same ID may identify the representative of a different set of origins, but the old decision is applied again without checking that lineage. | Consume a decision exactly once and persist the resulting admission state; downstream transforms never receive the raw decision map. |
| Chunk responses are merged against a row-wide candidate inventory | A response can decide a candidate assigned to a different validation request | Chunk A owns `c1` but returns `c2=drop`; chunk B owns `c2` and returns `c2=keep`. Row-wide lookup plus first-wins merging allows chunk A to override chunk B. | Bind aliases and decisions to the exact dispatched request inventory; wrong-request aliases have no authority, and response ordering is independent of completion order. |
| Raw origins and accepted origins share the same entity representation | A rejected or undecided origin can accidentally contribute provenance or propagation permission | If the validator omits a detector candidate, the current fallback keeps the same `EntitySpan`; final occurrence expansion can then treat it like an explicitly accepted candidate and protect unrelated occurrences. | Keep observations, validation results, and admitted candidates as distinct states; fallback protection preserves only the exact span and grants no accepted authority. |
| Exact identity is compared inconsistently | Equivalent labels may fail to coalesce and lose accepted provenance | Detector label `token` and augmenter label `TOKEN` cover the same bounds, but case-sensitive comparison treats them as competing spans and retains only one representative. | Define semantic identity as normalized label plus exact source bounds, then coalesce equivalent eligible claims without discarding accepted origins. |
| Provenance is collapsed into a representative source string | Coalescing can discard an independently accepted origin, while unconditional permission union can let a disabled route regain authority | The detector and augmenter independently accept the same email span, but overlap resolution retains one `source` string. With regex added, blindly combining origins could also let a model duplicate contribute even when `detect_additional_matches=False`. | Apply configured source and route eligibility first, then aggregate eligible accepted origins and combine permissions only within configuration limits. |
| Overlapping detector candidates are suppressed before validation | A longer candidate that is later dropped or excluded can erase a valid shorter candidate before either decision is known | GLiNER returns `John Doe` as `full_name` and `John` as `first_name`; parsing keeps only the longer span. If validation drops `John Doe`, the original `John` candidate is already gone. | Retain every conflicting observation through validation, represent the overlapping region once with a compact conflict group, and reconcile only admitted candidates. |
| Allowed/excluded checks can occur after an overlap winner is selected | An ineligible longer candidate can suppress an eligible shorter candidate and then be removed | An excluded `full_name` span wins over an allowed overlapping `first_name` during parsing; exclusion then removes the winner, leaving neither span protected. | Complete admission and normalized label/source-route eligibility before any candidate can suppress another. |
| Parsing, augmentation, name splitting, and propagation resolve overlaps independently | Results can depend on which route produced a candidate and when it entered the workflow | Identical GLiNER spans can be ranked by native confidence during parsing, while an identical GLiNER/augmenter conflict ignores scores and can fall through to label ordering; a candidate discarded in either pass never reaches propagation. | Collect all admitted and permitted derived candidates, then apply one documented final reconciliation policy with source-aware score comparison. |

## Terminology and Internal Model

The implementation should use different types, or equivalently strict state
objects, for the following concepts. These remain internal to
`anonymizer.engine`.

### Observation

An observation is one detector source's claim about a source span before that
claim has passed its configured validation route.

Required fields:

- `observation_id`: immutable identifier allocated by the producing backend
  before its batch is published;
- `source_batch_id`: immutable identifier for that backend result;
- exact source `value` taken from `text[start:end]`;
- original `label`, `start_position`, and `end_position`;
- source kind and source instance identity;
- source-native score when one exists;
- validation route and initial propagation policy.

The pair `(source_batch_id, observation_id)` is the immutable observation
reference. It is source-scoped, survives collection and downstream reuse, and
must be allocated independently of backend completion order. It does not
promise stability across separate anonymizer runs and must not be rebuilt after
reclassification, coalescing, derivation, or overlap resolution.

Validation uses a second identity: a short alias such as `c1`, assigned while
building one validation request. That alias is unique only inside its request
inventory and maps to exactly one immutable observation reference. Retries of
the same logical request retain the same inventory and aliases. Short aliases
are never used as provenance or regenerated during merging.

### Validation request and decision

Each validation request carries an immutable request inventory that binds:

- its request ID, source record, validation route, and effective label policy;
- every request-local alias to one immutable observation reference;
- the observation's authoritative label, value, and absolute source bounds;
- the source-context window and its absolute start and end offsets.

A validation decision refers to an observation through the short alias in that
request inventory. The same alias in another request has no authority over the
first request's observations.

This PR preserves the current validator response schema, model pool, and
keep/reclass/drop behavior. The prompt changes only to represent overlapping
candidates compactly. The admission layer's responsibility is to consume the
existing response safely:

- `keep` accepts the observation under its original label;
- `reclass` accepts it under a usable proposed label;
- `drop` rejects it.

Structured output enforces the response shape and the keep/reclass/drop enum,
but it cannot enforce row-dependent relationships between the response and the
input candidates. Admission therefore applies the following semantic policy:

- ignore and trace aliases that are not present in the response's own
  dispatched request inventory, even if that alias or observation exists
  elsewhere in the row;
- for duplicate aliases within one successful authoritative response, use the
  first usable decision in serialized response order and trace later entries;
- treat `keep` and `drop` as usable, and treat `reclass` as usable only when it
  contains a non-empty proposed label;
- when a successful response has no usable decision for one of its assigned
  observations, preserve only that exact observed span as fallback protection;
- when a validation request exhausts its attempts or fails terminally, preserve
  the workflow's existing failed-record behavior instead of converting every
  assigned observation into fallback protection.

Observation identity must never be inferred by matching echoed values, labels,
offsets, or aliases from another request.

A `reclass` decision without a proposed label preserves the exact original span
under its original eligible label as fallback protection. It does not grant
occurrence-propagation permission. When a non-empty proposed label is excluded
or falls outside an explicit allowed-label set, normal eligibility rejects the
reclassified candidate; the engine must not restore it under a knowingly
incorrect original label.

### Admission result

Applying validation produces one terminal admission result per observation:

- `accepted`: the route explicitly accepted the observation, possibly with a
  successfully resolved reclassified label;
- `rejected`: the route explicitly rejected it;
- `fallback_protected`: a successful validator response did not contain a
  usable decision for its assigned observation, so the existing fail-safe may
  preserve the observed span without treating the origin as validated, or a
  `reclass` decision did not contain a usable proposed label and the original
  span must remain protected.

This distinction preserves privacy-safe exact-span behavior while ensuring that
an undecided observation cannot grant accepted provenance or authorize broader
occurrence propagation. A decision is consumed while producing this result and
is never consulted again downstream.

`fallback_protected` is internal lifecycle state, not an entity label or a new
public result field. When such a candidate survives final reconciliation, the
public entity retains its ordinary label and exact bounds. The fallback status
and reason remain available only in internal admitted-candidate state and trace
diagnostics.

### Admitted candidate

An admitted candidate is eligible for protection after its validation route,
label scope, and configured source/route checks have completed. It contains:

- immutable lineage back to its originating observation references;
- the current label and exact source bounds;
- accepted provenance references containing the source batch ID, observation
  ID, source kind, and source instance only for origins that survived an
  enabled route;
- whether any configuration-eligible accepted origin permits occurrence
  propagation within the effective policy limits;
- whether it is explicitly accepted or retained only by fallback protection;
- source-native confidence retained only for comparisons where that score is
  meaningful.

Accepted origins and lineage are serialized in a deterministic order, not as
an unordered set. This keeps exported workflow state and trace comparisons
stable without collapsing distinct origins into one source string or requiring
downstream code to parse source-name prefixes.

An admitted candidate's semantic identity is:

```text
(normalize_label(label), start_position, end_position)
```

This identity is used for coalescing equivalent accepted claims. It is separate
from the immutable observation reference and the request-local alias used for
decision correlation.

## Semantic Invariants

The implementation must make the following properties true:

1. **Source offsets are authoritative.** Every materialized span value equals
   `text[start_position:end_position]`.
2. **Observation identity is immutable.** Backends assign source-batch-scoped
   observation references before publication. Reclassification and collection
   never replace them, and allocation never depends on completion order.
3. **Validation authority is request-scoped.** A decision may affect only the
   observation bound to its alias in that dispatched request inventory.
4. **Decisions are single-use.** Each selected usable decision is applied to
   its original observation exactly once.
5. **Unknown output is non-authoritative.** Unknown or wrong-request aliases
   and later duplicates in the same response cannot mutate a candidate.
6. **Omission is not execution failure.** A missing or unusable decision in a
   successful response may produce exact-span fallback protection; terminal
   request failure preserves failed-record behavior.
7. **Every retained observation reaches validation.** After redundant
   same-source emissions are collapsed, overlapping alternatives remain
   independent candidates, and the prompt represents each candidate exactly
   once without duplicating the source excerpt.
8. **Only accepted, configuration-eligible origins grant authority.** Rejected,
   undecided, or disabled source routes cannot contribute accepted provenance
   or propagation permission.
9. **Eligibility precedes suppression.** Excluded, disallowed, unsupported,
   disabled-source, or route-rejected candidates cannot remove an otherwise
   eligible candidate.
10. **Exact coalescing is normalized and lossless.** Equivalent label/span
   candidates coalesce while retaining eligible accepted origins and the most
   protective propagation behavior permitted by configuration.
11. **Derivation cannot broaden authority.** Name splitting and occurrence
   propagation inherit permissions from an accepted parent; a span-restricted
   parent cannot create independently protected occurrences elsewhere.
12. **Augmentation admission is prompt-independent.** Augmenter outputs never
    consume seed-validator decisions, whether the augmenter prompt uses seed
    context or plain source text.
13. **Seed protection precedes augmentation.** An admitted seed or authorized
    seed-derived candidate cannot be displaced by an augmentation-only
    candidate; this ordering does not grant fallback protection accepted
    provenance or propagation permission.
14. **Scores are compared only within their calibration domain.** A GLiNER
   confidence may rank otherwise equivalent GLiNER alternatives, but it is not
   compared with synthetic `1.0` values from augmentation or derived spans.
15. **Reconciliation is deterministic.** Input ordering, validation chunking,
    or detector execution order cannot change the final result for the same set
    of observations and decisions.
16. **Final spans are valid and non-overlapping.** Every output span is within
    source bounds, contains the exact source value, and survives one final
    overlap policy.

## Proposed Lifecycle

```text
source text
  |
  +-> detector backend -> observation batch ---------------+
  |       assign immutable source-batch references          |
  |       retain all conflicting alternatives              |
  |                                                        |
  |       build request inventory + short aliases          |
  |       route observations for validation                |
  |       -> consume existing keep / drop / reclass output |
  |       -> apply each decision once                       |
  |       -> accepted / rejected / fallback-protected       |
  |                                                        |
  |       apply label + source/route eligibility            |
  |       -> admitted seed candidates                       |
  |                                                        v
  +-> augmentation model ------------------------------ candidate pool
          prompt construction is independent of admission   |
          -> materialize exact source occurrences           |
          -> assign immutable observation references         |
          -> apply label + source/route eligibility          |
          -> admitted augmentation candidates                |
                                                             |
                 coalesce exact semantic identities <--------+
                 preserve eligible accepted provenance
                 combine permissions within config limits
                                  |
                 split permitted names, then propagate values
                                  |
                 apply eligibility to derived candidates
                                  |
                 coalesce exact derived identities
                                  |
                 one deterministic final overlap policy
                                  |
                 final entities + tagged text + trace state
```

### Step 1: Produce observations through a narrow backend contract

Introduce a private `_EntityDetectionBackend` contract whose successful result
is an ordered observation batch. A batch contains its immutable source batch
ID, source record reference, backend kind and instance, ordered observations,
and an explicit successful status. A successful empty batch is distinct from a
terminal backend failure. The initial request contract contains source text and
the effective label set; any later dependency on upstream detector context must
be represented as an explicit typed input rather than inferred from execution
order or silently ignored.

The backend owns detector-specific execution and translation into source-bound
observations. The shared lifecycle owns validation admission, label and route
eligibility, provenance, propagation, derivation, and reconciliation. Adapt
GLiNER first while preserving the existing DataDesigner execution and export
path. PR #265 adds regex as another backend producer. Collection retains each
backend's original observation references; forwarding observations through a
later backend cannot manufacture a second supporting origin.

For this PR, GLiNER consumes source text and the effective label set; it does
not consume an earlier backend's observations merely because it executes later.
When PR #265 defines a regex-to-GLiNER context mode, that dependency must be a
typed input. GLiNER then publishes only its new claims under GLiNER observation
references, while collection preserves the original regex batch alongside the
new GLiNER batch. Forwarded regex observations retain their regex identity and
route restrictions and are not republished as GLiNER support.

The private contract is synchronous or declarative as required by the existing
DataDesigner workflow. It does not prescribe an asynchronous graph execution
API. A later graph binding may own dependencies, retries, request association,
resource limits, and execution artifacts while consuming the same observation
batches and lifecycle semantics.

### Step 2: Parse without destructive conflict resolution

Validate backend observations, preserve their immutable references, and derive
exact values from source bounds. Exact source slicing follows PR #282. Do not
discard overlapping alternatives while parsing: a longer candidate that is
later rejected or excluded must not have already removed a valid shorter
candidate.

Only redundant emissions with the same source instance, normalized label,
bounds, validation route, and propagation policy may be collapsed before
validation. When calibrated scores differ within that otherwise identical
same-source group, retain the highest score. Exact candidates from different
sources or validation routes remain independent through admission so
pre-validation coalescing cannot discard an independently meaningful decision
or permission. General semantic coalescing happens only after admission.

### Step 3: Preserve the current validation contract

Validation input continues to include the text context needed to make a
decision. The response contract and model roles remain unchanged in this PR:

```json
{
  "decisions": [
    {
      "id": "c1",
      "value": "Alice",
      "label": "first_name",
      "decision": "keep",
      "proposed_label": "",
      "reason": "person name in context"
    },
    {
      "id": "c2",
      "value": "Seattle",
      "label": "country",
      "decision": "reclass",
      "proposed_label": "city",
      "reason": "Seattle is a city"
    }
  ]
}
```

The engine resolves each response against its original observation. Values and
labels echoed by the model are not used as authority for source spans or
candidate identity.

Because overlapping candidates can no longer be suppressed before validation,
the prompt input gains a compact conflict representation. The source excerpt is
included once. Non-overlapping candidates keep their existing tags; each
connected overlapping region receives one neutral candidate-group tag, and the
validation template maps that group ID to the candidates assigned to the
request. Every inventory entry includes its short alias, label, value, and
absolute character offsets into the original source text. The request also
records the absolute start and end of its context window, so coordinates remain
unambiguous when the excerpt is clipped. Repeated values such as the two `Ann`
occurrences in `Ann Ann` are therefore distinguishable without encoding
geometry into the alias.

If one connected overlap group crosses a validation chunk boundary, every
request may show the same neutral source region but exposes only the inventory
entries that request is authorized to decide. The model still returns one
existing-schema decision per assigned candidate. This avoids duplicating tagged
text for every overlap alternative or inflating the prompt with multiple copies
of the same source context.

Decision-only validation and conditional reclassification change the response
schema, model roles, cost, latency, and quality behavior. That work is tracked
separately in [issue #304](https://github.com/NVIDIA-NeMo/Anonymizer/issues/304)
rather than added to issue #262.

Each response is interpreted only against its dispatched request inventory.
Wrong-request aliases are ignored even when they happen to name a valid alias
or observation in another chunk. Within one successful response, the first
usable decision in serialized response order wins and later entries for that
alias are counted in trace diagnostics. Completion order never selects a
decision. If a successful response omits an assigned observation or supplies no
usable decision for it, that observation becomes `fallback_protected`,
preserving the exact observed span without granting accepted provenance or
propagation. Exhausted attempts or a terminal request failure continue through
the existing failed-record path instead.

### Step 4: Apply decisions once and admit candidates

The admission transform owns validation consumption. It:

1. resolves the response alias through its authoritative request inventory to
   an immutable observation reference;
2. computes the resulting label once, using the current proposed label for a
   valid `reclass` decision;
3. records the terminal admission state;
4. applies normalized label eligibility and configured source/route policy to
   the resulting candidate;
5. emits an admitted candidate or a rejection trace record.

To preserve current behavior deliberately, an explicitly accepted detector
candidate permits occurrence propagation, while a fallback-protected detector
candidate does not. A valid reclassification is checked under its proposed
label before it can enter reconciliation.

Configuration is an authority ceiling, not another origin to merge. A source
or detection route disabled for a label does not enter coalescing and cannot
contribute provenance or propagation permission through an exact duplicate.
When PR #265 is rebuilt on this lifecycle,
`detect_additional_matches=False` for a regex entity therefore makes
model-origin candidates for that label ineligible;
`validate_matches_with_llm` independently selects the admission route for the
regex observation itself.

No later transform receives the raw decision map as an input. In particular,
the finalization step must not reapply detector decisions to augmented,
coalesced, propagated, or name-derived entities.

### Step 5: Admit augmentation through a separate route

Augmenter prompt construction is independent of augmentation admission. This
foundational PR preserves the current seed-aware prompt. While that prompt
remains, admitted seed candidates are rendered without silently omitting
overlapping alternatives: use one neutral candidate-group tag for the source
region and keep the group-to-candidate mapping in the structured seed
inventory. PR #267 can later switch the prompt to plain source text and remove
seed hints without changing the output-admission contract described here.

Unlike seed validation, augmentation is open-ended structured extraction: it
does not answer an expected inventory of validation aliases. Unknown-alias,
duplicate-decision, and missing-decision policies therefore do not apply. Each
augmenter suggestion is instead materialized at its exact source occurrences,
and each materialized occurrence receives a new immutable observation reference
and augmenter origin. It then passes bounds, exact-value, label-normalization,
and configured source/route eligibility checks before joining the candidate
pool.

Augmentation output is already a model decision rather than a high-recall
detector observation, so it does not reuse the seed validator decision map.
Exact duplicate suggestions are handled by semantic coalescing, not by an
augmentation-specific overlap pass. Materialized augmentation candidates do
not receive a second, implicit propagation pass.

### Step 6: Coalesce exact semantic identities

Group admitted candidates by normalized label and exact bounds. Coalescing:

- keeps the source value from authoritative offsets;
- retains lineage to all contributing observations;
- retains provenance only from accepted origins whose source and route remain
  enabled by configuration;
- allows propagation when at least one such eligible accepted origin
  independently permits it and configuration does not prohibit it;
- does not let fallback-only, rejected, or configuration-disabled origins add
  permissions;
- retains calibrated score information by source rather than manufacturing a
  cross-source score.

Same bounds with different normalized labels are not exact duplicates. They
remain alternatives for final reconciliation.

### Step 7: Derive in one bounded pass without broadening permissions

Occurrence propagation and full-name splitting become one bounded derivation
pass over admitted candidates:

1. split eligible accepted full-name candidates into supported name-part
   candidates at the observed occurrence;
2. propagate the exact values of eligible accepted original candidates and
   newly split name parts when their supporting origin permits propagation;
3. apply label and route eligibility to every derived candidate;
4. coalesce exact semantic identities across admitted and derived candidates,
   retaining every eligible supporting origin;
5. send the resulting pool to final reconciliation.

Propagation outputs are not split again, and propagation is not rerun over its
own outputs, so this is bounded rather than a fixed-point engine. Splitting
before value propagation means an accepted `John Smith` can still protect the
standalone `John` in `John Smith met John` when the accepted origin authorizes
that behavior.

A fallback-protected or span-restricted origin protects its observed bounds but
cannot authorize splitting or other occurrences. Authority remains attached to
the specific supporting origin across coalescing and derivation. If an exact
fallback seed coalesces with an accepted augmentation origin, a permitted
derivation is attributable only to the accepted augmentation origin; the
fallback origin does not become an accepted seed ancestor or grant the derived
candidate seed-supported precedence. Derivation itself performs no overlap
suppression.

### Step 8: Reconcile once

Run one pure final reconciliation function after all candidates are admitted,
coalesced, and derived. The comparator must be documented in one place and
must provide a total, deterministic order.

A seed-supported candidate is backed by a pre-augmentation detector or regex
observation that was admitted as either accepted or fallback-protected, or is
an authorized derivation of an accepted seed. This reconciliation classification
does not turn fallback protection into accepted provenance or grant propagation
permission; it only prevents a later augmentation result from displacing an
already protected seed span. It is derived from lineage and admission state,
not stored as a separate boolean.

The initial policy is:

1. prefer a seed-supported candidate over an augmentation-only candidate;
2. within the same seed-support tier, prefer the longer source span;
3. prefer the earlier start, then earlier end;
4. for identical bounds with incompatible labels, prefer an explicitly
   accepted candidate over a fallback-protected candidate;
5. use the centrally defined internal source priority;
6. within the same calibrated detector source and identical bounds, prefer the
   higher native confidence;
7. use normalized label and immutable lineage as stable final tie-breakers.

Step 1 preserves the precedence intended by PR #267: an admitted seed such as
validated `Seattle`/`city` beats an overlapping augmentation such as
`near downtown Seattle`/`address`, even though the augmenter span is longer.
Fallback-protected seeds receive the same reconciliation precedence so their
exact protected span cannot be displaced, but they still contribute no
accepted provenance or propagation permission. For conflicts within the same
seed-support tier, span coverage applies; for identical bounds, explicit
acceptance wins at step 4.

This precedence is a settled internal policy, not a user-facing option. A
fallback-protected seed wins over overlapping augmentation because allowing an
augmentation-only candidate to replace it would break the exact-span privacy
guarantee when the augmenter emits a narrower span. When the augmenter emits a
broader overlap, the minimal-span contract used by PR #267 treats that output
as over-broad rather than attempting to split residual text around the seed.

Source priority is an internal reconciliation policy, not a user-facing
configuration knob. It must not depend on which route happens to be passed
first by a caller. For the sources on `main`, the initial order is detector,
augmenter, then derived observations. Seed support is derived from lineage and
admission state before this source tie-break, so an authorized derivation
inherits its accepted parent's precedence while an augmentation-derived
candidate does not. Within the seed-supported tier, fallback protection is
ranked by span coverage and step 4 rather than being treated as accepted
provenance. PR #265 can extend the centralized source table for custom regex
and built-in regex without adding another merge implementation.

## Affected Subsystems

- `anonymizer.engine.detection`: candidate construction, validation input,
  private detector backend boundary, admission, augmentation merge, derivation,
  and final reconciliation;
- `anonymizer.engine.schemas`: JSON-compatible internal observation, admission,
  provenance, and diagnostic payloads;
- `anonymizer.engine.workflow_columns.detection`: serialized transform
  operations for the revised lifecycle;
- `anonymizer.engine.constants`: lifecycle and trace column names;
- detection workflow, serialization, replace-mode, rewrite-mode, and trace
  tests.

`anonymizer.config` and `anonymizer.interface` retain their existing public
signatures and result schemas. The bundled anonymizer skill does not require a
change because this PR does not alter the public `Detect` surface.

## Implementation Shape

The exact class names can follow repository conventions, but responsibilities
should be separated as follows.

### Schemas

Update `src/anonymizer/engine/schemas/detection.py` to represent:

- immutable source-batch and observation IDs;
- validation request inventories and their request-local aliases;
- explicit successful-empty and terminal-failure backend results;
- the existing validator response associated with its authoritative request;
- terminal admission state;
- accepted provenance and propagation permission;
- optional trace diagnostics for ignored or unusable decisions.

Serialized workflow columns must remain JSON-compatible and portable through
the existing DataDesigner plugin boundary.

### Detector backend boundary

Add a private `_EntityDetectionBackend` contract and observation-batch result
under `src/anonymizer/engine/detection/`. The first adapter wraps the existing
GLiNER declaration and parsing path without bypassing `NddAdapter.run_workflow()`
or changing the public `Detect` API. The boundary publishes source-bound
observations and success/failure state; it does not own validation, eligibility,
coalescing, propagation, or reconciliation.

The contract must distinguish a successful empty result from failure, assign
observation references before publishing the batch, and preserve references
when batches are collected. Its initial inputs are explicit source text and the
effective label set. Any future backend reuse adds a typed input and defined
behavior; unsupported input fails clearly rather than being silently ignored.

### Pure detection semantics

Refactor `src/anonymizer/engine/detection/postprocess.py` around pure helpers for:

- source-bound observation construction;
- request-inventory construction, decision indexing, and single-use application;
- compact validation conflict groups for overlapping candidates;
- augmentation-output materialization without overlap suppression or seed-
  decision replay;
- normalized eligibility;
- exact-identity coalescing;
- permission-preserving derivation;
- deterministic final reconciliation.

Each helper should accept and return explicit lifecycle states. Avoid helpers
that accept a raw decision payload after admission has completed.

### Workflow transforms

Update `src/anonymizer/engine/detection/custom_columns.py` and the corresponding
workflow-column operation enum so the column sequence reflects the lifecycle:

```text
parse observations
-> prepare conflict-aware validation
-> validate
-> admit seed candidates
-> augment
-> admit augmentation candidates
-> coalesce admitted candidates
-> split, propagate, and coalesce derived candidates
-> reconcile and finalize
```

Intermediate column names should describe lifecycle state rather than retain
historical names such as "merged" when the payload has not yet completed final
reconciliation. Column constants remain centralized in
`src/anonymizer/engine/constants.py`.

### Workflow assembly

Update `src/anonymizer/engine/detection/detection_workflow.py` to wire the new
transforms while preserving the current `EntityDetectionWorkflow` boundary,
DataDesigner execution path, latent-entity stage, final entity schema, and
public result shape.

The eventual graph/SDK implementation can replace this linear assembly while
reusing the lifecycle types and pure semantics.

### Trace diagnostics

Persist JSON-compatible admission diagnostics in an internal workflow trace
column. At minimum, record request identity, ignored unknown or wrong-request
aliases, duplicate decisions within a response, assigned candidates missing a
usable decision from a successful response, fallback reasons, terminal request
failure, reclassification fallback, and the accepted origin references
retained after coalescing. Keep these diagnostics out of the public
`COL_FINAL_ENTITIES` schema and final user dataframe; they are available
through the existing trace dataframe for debugging and evaluation.

## Validation Strategy

### Unit and composed tests

Add behavior-focused tests using fabricated data for at least the following
cases:

| Scenario | Required result |
|---|---|
| Detector succeeds with no observations | Backend emits a successful empty batch rather than a failure |
| Detector fails terminally | Backend failure remains distinguishable from a successful empty batch and follows failed-record behavior |
| GLiNER executes after another backend without a typed data binding | It consumes only source text and effective labels; execution order alone does not alter its observations |
| An upstream observation batch is explicitly reused by a later backend | Collection preserves the original batch and adds only the later backend's new claims; forwarded observations do not gain another supporting origin |
| Observation batches and request inventories are serialized and restored | Immutable references, alias bindings, geometry, policy, ordering, and success/failure state survive the round trip |
| Redundant exact emissions from one source and route | One observation reaches validation, retaining the highest score from that calibrated source |
| Exact emissions from different sources or routes | They remain independent through admission so each decision and permission is preserved |
| Observation batches complete in different orders | Their preassigned observation references and collected result are unchanged |
| Candidate is reclassified | Original observation reference remains valid; resulting semantic identity uses the new normalized label |
| Reclass decision has no proposed label | Original exact span remains fallback-protected under its original eligible label and propagation is disabled |
| Reclass decision proposes an excluded or disallowed label | Candidate is rejected by eligibility before reconciliation and is not restored under its old label |
| Detector candidates overlap | Every candidate is represented once in the validation template; the source excerpt contains one neutral group tag rather than duplicated alternative tags |
| Repeated candidate values occur in one conflict group | Absolute bounds distinguish each occurrence even though values and labels match |
| Validation uses a clipped source excerpt | Context-window bounds and absolute candidate bounds map every alias unambiguously to the original source |
| A longer overlapping candidate is dropped | An independently accepted shorter candidate remains eligible for final reconciliation |
| One overlap group crosses a validation chunk boundary | Each chunk maps the neutral group only to its assigned aliases and absolute geometry; every observation belongs to one authoritative request inventory |
| Accepted seed candidates still overlap while the current seed-aware augmenter prompt remains | The augmenter sees one neutral group tag plus every seed in the structured inventory, without duplicated source text; PR #267 can later remove this prompt dependency without changing admission |
| Augmented candidate constructs the same label/bounds as a seed | The seed decision is not replayed against the augmented candidate |
| Augmenter returns duplicate suggestions for the same label/value | Materialized exact identities coalesce deterministically without an augmentation-specific overlap pass |
| Longer augmentation candidate overlaps an admitted seed | The seed-supported candidate wins regardless of span length, preserving the precedence required by PR #267 |
| Fallback-protected seed overlaps an augmentation candidate | The seed span keeps reconciliation precedence without gaining accepted provenance or propagation permission |
| Candidate derived from an accepted seed overlaps augmentation | The derived candidate inherits seed support through lineage and uses the same final reconciler |
| Unknown decision alias | No observation is mutated; diagnostic is recorded |
| Chunk A returns an alias assigned to chunk B | Chunk A cannot mutate chunk B's observation; the wrong-request alias is traced |
| Duplicate decision alias within one response | First usable entry in serialized response order wins deterministically |
| Assigned candidate is absent from a successful structured response | Exact detector span remains fallback-protected; it grants no accepted provenance or propagation |
| Validation chunk fails terminally | Its row follows existing failed-record behavior rather than converting assigned candidates to fallback protection |
| Labels differ only by case or surrounding whitespace | Exact spans coalesce under normalized identity |
| One exact origin is accepted and another is rejected | Only the accepted origin contributes provenance and permissions |
| One exact accepted origin permits propagation and another does not | When both routes are configuration-eligible, the coalesced candidate permits propagation without losing either accepted origin |
| A model origin duplicates a regex entity whose model-detection route is disabled | The model origin contributes no provenance or permission; the regex policy remains authoritative |
| A model origin duplicates a regex entity whose model-detection route is enabled | Independently accepted origins may coalesce and combine behavior within the configured limits |
| Excluded longer span overlaps allowed shorter span | Shorter candidate survives because eligibility runs first |
| Same-source GLiNER alternatives have identical bounds | Native detector confidence breaks the label tie |
| Cross-source candidates carry synthetic and native scores | Scores are not compared across calibration domains |
| Span-restricted full name is split | Derived parts do not appear at unrelated occurrences |
| Propagating full name contains a repeated standalone part | Splitting occurs before propagation, so the eligible standalone part is protected |
| Two accepted parents derive the same exact identity | The derived candidates coalesce while retaining both supporting origins |
| Fallback seed coalesces with accepted augmentation | Only the accepted augmentation origin can authorize derivation; fallback lineage does not become accepted-seed authority |
| Propagated, augmented, and seed spans conflict | All use the same final comparator |
| Candidate input order is permuted | Final entities and provenance are unchanged |
| Accepted and fallback-protected candidates have identical bounds | The explicitly accepted candidate wins before source priority is considered |

Retain end-to-end assertions for `COL_DETECTED_ENTITIES`, `COL_FINAL_ENTITIES`,
tagged text, grouped entities, strict labels, excluded labels, replace mode,
rewrite mode, workflow export, and chunked validator pools.

### Behavioral validation

Run the existing detection benchmark before and after the lifecycle change.
This PR preserves model roles and the response schema; its prompt change is
limited to compactly representing overlapping candidates that were previously
discarded. The comparison is a regression check rather than a model-contract
experiment. Measure:

- candidates missing a usable decision;
- unknown, wrong-request, and duplicate aliases;
- successful omissions versus terminal validation-request failures;
- structured-output call or schema-validation failures at the workflow boundary;
- keep/drop/reclass agreement and unresolved-decision handling;
- per-label precision and recall;
- candidate coverage;
- prompt and completion tokens;
- latency and failed-record rate.

The lifecycle change advances only if privacy quality does not regress
materially. Any increase in candidates, tokens, or model calls must be
attributable to validating previously suppressed conflicts, quantified in the
results, and accepted explicitly. Output differences must correspond to an
intentional correction captured by a focused test.

### Repository checks

Run the focused detection suite while iterating, followed by the repository
checks required by `DEVELOPMENT.md`:

```bash
uv run --group dev pytest \
  tests/engine/test_detection_postprocess.py \
  tests/engine/test_detection_custom_columns.py \
  tests/engine/test_chunked_validation.py \
  tests/engine/test_detection_workflow.py
make check
make test
```

## Trade-offs

### Validate overlapping alternatives before suppression

Retaining conflicts until admission may send more GLiNER observations to the
validator than early overlap pruning. The additional cost is necessary for
correct admission ordering: an unvalidated or ineligible candidate cannot be
allowed to erase another candidate. Benchmark candidate and token counts to
quantify the change.

### Separate observation, request, and semantic identity

Three identities add terminology, but they serve different lifetimes. An
immutable source-batch observation reference preserves detector lineage; a
short request-local alias constrains one validation response; normalized
label/bounds identity makes coalescing correct after reclassification. Reusing
one identity across those jobs causes decision replay, cross-chunk authority,
and mutable provenance.

### Preserve fallback protection without granting authority

Keeping an undecided detector span is safer than silently exposing it, but
treating it as fully accepted can broaden protection to unrelated occurrences.
The explicit fallback state preserves the exact span while preventing that
authority leak.

### Centralize source priority

A fixed internal priority is less flexible than user-configurable precedence,
but it gives stable behavior and keeps public configuration small. The policy
can evolve in one place as new detector sources are added.

### Preserve seed spans across clean-slate augmentation

PR #267 intentionally makes augmentation operate on plain text and relies on a
minimal-span extraction contract. That increases duplicate and overlapping
model discoveries. Giving admitted seed spans reconciliation precedence over
augmentation-only candidates prevents an unvalidated relabeling from replacing
an already protected seed, but it can discard a broader augmentation span that
contains the seed. Preserve this policy centrally, test the minimal-span prompt
contract in PR #267, and measure missed sensitive coverage after its rebase.

## Rollout Plan

1. **Completed:** land PR #282 and base this work on the resulting `main`.
2. Add lifecycle types, the private backend contract, and characterization
   tests for current behavior.
3. Adapt GLiNER to publish ordered observation batches with immutable references
   while preserving the existing DataDesigner execution and export path.
4. Add validation request inventories and compact conflict-group input with
   explicit source geometry while preserving the existing response schema and
   model roles.
5. Introduce request-scoped, single-use admission and remove downstream decision
   replay.
6. Move eligibility before exact coalescing and overlap reconciliation.
7. Move provenance, bounded name splitting and occurrence propagation, and
   post-derivation coalescing onto the admitted-candidate contract.
8. Replace route-specific overlap calls with the single final reconciler.
9. Run focused, full, export, and benchmark validation.
10. Rebase PR #267 so it changes the augmenter prompt to plain source text and
   removes seed hints while reusing shared augmentation admission and final
   reconciliation; drop its route-specific `resolve_overlaps()` filtering.
11. Rebase or reconstruct PR #265 so regex contributes observation batches and
   route policy through these shared contracts rather than owning separate merge
   and decision-replay machinery.
12. Document these semantics as invariants for the later graph/SDK detection
    stage and preserve them when its execution assembly changes.

## Review Checklist

Maintainer review should verify that the implementation preserves these
settled invariants:

1. Backends allocate immutable source-batch observation references before
   publication; validation uses separate short aliases scoped to one dispatched
   request inventory, and semantic identity remains normalized label plus bounds.
2. The current validation response schema and model roles remain unchanged;
   prompt input changes only to represent overlapping candidates compactly.
3. A `reclass` decision without a proposed label retains exact-span fallback
   protection under the original eligible label and cannot authorize occurrence
   propagation; a non-empty excluded or disallowed proposed label is rejected
   by eligibility rather than restored under the old label.
4. Missing or unusable decisions in a successful response retain exact-span
   fallback protection without authority; terminal request failure preserves
   existing failed-record behavior.
5. A response can decide only its dispatched inventory; first usable result in
   serialized response order wins for duplicates within that response.
6. Conflict-group inventories include absolute source geometry and expose only
   candidates assigned to that request.
7. Eligibility completes before any candidate can suppress another.
8. Exact coalescing retains accepted provenance and combines propagation
   permission only among configuration-eligible origins; disabled routes cannot
   regain authority during merging.
9. Name splitting precedes eligible value propagation in one bounded derivation
   pass; exact derived duplicates coalesce before reconciliation without
   transferring authority between supporting origins.
10. Seed-supported candidates receive precedence over augmentation-only
   candidates in the final reconciler, without converting fallback protection
   into accepted provenance or propagation authority.
11. Augmenter prompt construction is independent of output admission: this PR
   preserves the current seed-aware prompt, and PR #267 may later switch it to
   plain text without adding another merge path.
12. All admitted and derived candidates use one final reconciliation policy.
13. Source priority remains an internal centralized policy rather than a public
   configuration option.
14. The graph/SDK refactor consumes the private backend and lifecycle contracts
    without redefining their authority semantics.
