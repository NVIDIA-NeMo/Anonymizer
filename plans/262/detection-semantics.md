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
3. Rebase or reconstruct PR #265 on the resulting internal contracts.
4. Preserve these contracts when detection is later represented as graph/SDK
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

| Current behavior | Failure it permits | Required semantic change |
|---|---|---|
| IDs are constructed from label and offsets | Reclassification or exact-candidate coalescing can change the representative identity used to find a decision | Give every original observation an immutable row-local correlation ID independent of mutable semantic fields |
| The same decision payload is applied before and after augmentation | A decision can be replayed against a different merged or derived candidate that happens to carry the same constructed ID | Consume a decision exactly once and persist the resulting admission state |
| Raw origins and accepted origins share the same entity representation | A rejected, malformed, or undecided origin can accidentally contribute provenance or propagation permission | Keep observations, validation results, and admitted candidates as distinct states |
| Exact identity is compared inconsistently | Equivalent labels such as `token` and `TOKEN` may fail to coalesce and lose accepted provenance | Define semantic identity as normalized label plus exact source bounds |
| Provenance is collapsed into a representative source string | Coalescing can discard an independently accepted origin or its propagation permission | Aggregate accepted origins explicitly and combine permissions monotonically |
| Overlapping detector candidates are suppressed before validation | A longer candidate that is later dropped or excluded can erase a valid shorter candidate before either decision is known | Retain every conflicting observation through validation and represent overlap compactly in the prompt |
| Allowed/excluded checks can occur after an overlap winner is selected | An ineligible longer candidate can suppress an eligible shorter candidate and then be removed | Complete admission and eligibility checks before destructive reconciliation |
| Parsing, augmentation, name splitting, and propagation resolve overlaps independently | Results can depend on which route produced a candidate and when it entered the workflow | Use one final reconciliation policy for all admitted and derived candidates |

## Terminology and Internal Model

The implementation should use different types, or equivalently strict state
objects, for the following concepts. These remain internal to
`anonymizer.engine`.

### Observation

An observation is one detector source's claim about a source span before that
claim has passed its configured validation route.

Required fields:

- `candidate_id`: immutable, short, row-local correlation token such as `c0001`;
- exact source `value` taken from `text[start:end]`;
- original `label`, `start_position`, and `end_position`;
- source kind and source instance identity;
- source-native score when one exists;
- validation route and initial propagation policy.

`candidate_id` is not a public entity identity and does not promise stability
across runs. It exists only to correlate one model decision with one original
observation. It must not be rebuilt after reclassification, coalescing,
derivation, or overlap resolution.

### Validation decision

A validation decision refers to exactly one observation by `candidate_id`.
This PR preserves the current validator response schema, model pool, and
keep/reclass/drop behavior. The prompt changes only to represent overlapping
candidates compactly. The admission layer's responsibility is to consume the
existing response safely:

- `keep` accepts the observation under its original label;
- `reclass` accepts it under a usable proposed label;
- `drop` rejects it.

Unknown IDs, duplicate decisions, missing decisions, and null decisions are
handled by one documented policy and recorded in trace metadata. They must
never be interpreted by matching mutable values, labels, or offsets.

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
- `fallback_protected`: the route did not produce a usable decision, so the
  existing fail-safe may preserve the observed span without treating the
  origin as validated, or a `reclass` decision did not contain a usable
  proposed label and the original span must remain protected.

This distinction preserves privacy-safe exact-span behavior while ensuring that
an undecided observation cannot grant accepted provenance or authorize broader
occurrence propagation. A decision is consumed while producing this result and
is never consulted again downstream.

### Admitted candidate

An admitted candidate is eligible for protection after its validation route and
label-scope checks have completed. It contains:

- immutable lineage back to its originating observation IDs;
- the current label and exact source bounds;
- accepted provenance references containing the observation ID, source kind,
  and source instance only for origins that survived their route;
- whether any accepted origin permits occurrence propagation;
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
from `candidate_id`, which is used only for decision correlation and lineage.

## Semantic Invariants

The implementation must make the following properties true:

1. **Source offsets are authoritative.** Every materialized span value equals
   `text[start_position:end_position]`.
2. **Observation identity is immutable.** Reclassification changes the semantic
   label but never changes the ID used to correlate the original decision.
3. **Decisions are single-use.** Each valid decision is applied to its original
   observation exactly once.
4. **Unknown output is non-authoritative.** Unknown IDs and duplicate or
   malformed validation entries cannot mutate a candidate.
5. **Every retained observation reaches validation.** After redundant
   same-source emissions are collapsed, overlapping alternatives remain
   independent candidates, and the prompt represents each candidate exactly
   once without duplicating the source excerpt.
6. **Only accepted origins grant authority.** Rejected or undecided origins
   cannot contribute accepted provenance or propagation permission.
7. **Eligibility precedes suppression.** Excluded, disallowed, unsupported, or
   route-rejected candidates cannot remove an otherwise eligible candidate.
8. **Exact coalescing is normalized and lossless.** Equivalent label/span
   candidates coalesce while retaining the union of accepted origins and the
   most protective permitted propagation behavior.
9. **Derivation cannot broaden authority.** Name splitting and occurrence
   propagation inherit permissions from an accepted parent; a span-restricted
   parent cannot create independently protected occurrences elsewhere.
10. **Scores are compared only within their calibration domain.** A GLiNER
   confidence may rank otherwise equivalent GLiNER alternatives, but it is not
   compared with synthetic `1.0` values from augmentation or derived spans.
11. **Reconciliation is deterministic.** Input ordering, validation chunking,
    or detector execution order cannot change the final result for the same set
    of observations and decisions.
12. **Final spans are valid and non-overlapping.** Every output span is within
    source bounds, contains the exact source value, and survives one final
    overlap policy.

## Proposed Lifecycle

```text
source text
  |
  +-> detector observations ------------------------------+
  |       assign immutable row-local candidate IDs         |
  |       retain all conflicting alternatives              |
  |                                                        |
  |       route observations for validation                |
  |       -> consume existing keep / drop / reclass output |
  |       -> apply each decision once                       |
  |       -> accepted / rejected / fallback-protected       |
  |                                                        |
  |       apply label eligibility                           |
  |       -> admitted seed candidates                       |
  |                                                        v
  +-> augmentation using admitted seed context        candidate pool
          -> materialize exact source occurrences           |
          -> apply label eligibility                         |
          -> admitted augmentation candidates                |
                                                             |
                 coalesce exact semantic identities <--------+
                 preserve accepted provenance + permissions
                                  |
                 derive permitted name parts and occurrences
                                  |
                 apply eligibility to derived candidates
                                  |
                 one deterministic final overlap policy
                                  |
                 final entities + tagged text + trace state
```

### Step 1: Parse without destructive conflict resolution

Parse detector output into observations, validate bounds, and assign candidate
IDs. Exact source slicing follows PR #282. Do not discard overlapping
alternatives while parsing: a longer candidate that is later rejected or
excluded must not have already removed a valid shorter candidate.

Only redundant emissions with the same source instance, normalized label,
bounds, validation route, and propagation policy may be collapsed before
validation. When calibrated scores differ within that otherwise identical
same-source group, retain the highest score. Exact candidates from different
sources or validation routes remain independent through admission so
pre-validation coalescing cannot discard an independently meaningful decision
or permission. General semantic coalescing happens only after admission.

### Step 2: Preserve the current validation contract

Validation input continues to include the text context needed to make a
decision. The response contract and model roles remain unchanged in this PR:

```json
{
  "decisions": [
    {
      "id": "c0001",
      "value": "Alice",
      "label": "first_name",
      "decision": "keep",
      "proposed_label": "",
      "reason": "person name in context"
    },
    {
      "id": "c0002",
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
validation template maps that group ID to the candidate IDs that overlap there.
The model still returns one existing-schema decision per candidate. This avoids
duplicating tagged text for every overlap alternative or inflating the prompt
with multiple copies of the same source context.

Decision-only validation and conditional reclassification change the response
schema, model roles, cost, latency, and quality behavior. That work is tracked
separately in [issue #304](https://github.com/NVIDIA-NeMo/Anonymizer/issues/304)
rather than added to issue #262.

Chunk merging and decision application must share one duplicate policy. The
proposed rule is first valid decision wins; later decisions for the same ID are
ignored and counted in trace diagnostics. Missing or unusable decisions produce
`fallback_protected`, preserving the exact observed span without granting
accepted provenance or propagation.

### Step 3: Apply decisions once and admit candidates

The admission transform owns validation consumption. It:

1. looks up a decision by immutable candidate ID;
2. computes the resulting label once, using the current proposed label for a
   valid `reclass` decision;
3. records the terminal admission state;
4. applies normalized allowed/excluded label checks to the resulting label;
5. emits an admitted candidate or a rejection trace record.

To preserve current behavior deliberately, an explicitly accepted detector
candidate permits occurrence propagation, while a fallback-protected detector
candidate does not. A valid reclassification is checked under its proposed
label before it can enter reconciliation.

No later transform receives the raw decision map as an input. In particular,
the finalization step must not reapply detector decisions to augmented,
coalesced, propagated, or name-derived entities.

### Step 4: Augment from admitted seed context

The augmentation prompt continues to see the accepted/fallback-protected seed
spans needed to avoid duplicate discovery. When admitted seed candidates still
overlap, reuse the neutral candidate-group rendering so the source text appears
once and no accepted seed is silently omitted. The structured seed inventory
includes the group-to-candidate mapping and remains authoritative for the
augmenter. Augmenter output is materialized against the original source text
and goes through the same bounds, exact-value, label-normalization, and
eligibility checks before joining the candidate pool.

Augmentation is already a model decision rather than a high-recall detector
observation, so it does not reuse the seed validator decision map. The
augmenter value is materialized at all exact source occurrences in this step;
those materialized candidates do not receive a second, implicit propagation
pass.

### Step 5: Coalesce exact semantic identities

Group admitted candidates by normalized label and exact bounds. Coalescing:

- keeps the source value from authoritative offsets;
- retains lineage to all contributing observations;
- retains only accepted provenance;
- allows propagation when at least one accepted origin independently permits
  it;
- does not let fallback-only or rejected origins add permissions;
- retains calibrated score information by source rather than manufacturing a
  cross-source score.

Same bounds with different normalized labels are not exact duplicates. They
remain alternatives for final reconciliation.

### Step 6: Derive without broadening permissions

Occurrence propagation and full-name splitting become explicit derivation
operations over admitted candidates.

- An accepted parent that permits propagation may create candidates at other
  exact occurrences.
- A fallback-protected or span-restricted parent protects its observed bounds
  but cannot authorize other occurrences.
- A derived candidate records its parent lineage and inherited permissions.
- Derived candidates pass label eligibility before final reconciliation.
- Derivation does not perform overlap suppression.

### Step 7: Reconcile once

Run one pure final reconciliation function after all candidates are admitted,
coalesced, and derived. The comparator must be documented in one place and
must provide a total, deterministic order.

The initial policy is:

1. prefer the longer source span;
2. prefer the earlier start, then earlier end;
3. for identical bounds with incompatible labels, prefer an explicitly
   accepted candidate over a fallback-protected candidate;
4. use the centrally defined internal source priority;
5. within the same calibrated detector source and identical bounds, prefer the
   higher native confidence;
6. use normalized label and immutable lineage as stable final tie-breakers.

Source priority is an internal reconciliation policy, not a user-facing
configuration knob. It must not depend on which route happens to be passed
first by a caller. For the sources on `main`, the initial order is detector,
augmenter, then derived observations. Fallback protection is ranked by step 3
rather than being allowed to inherit detector priority. PR #265 can extend the
centralized source table for custom regex and built-in regex without adding
another merge implementation.

## Affected Subsystems

- `anonymizer.engine.detection`: candidate construction, validation input,
  admission, augmentation merge, derivation, and final reconciliation;
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

- immutable observation/candidate correlation IDs;
- the existing validator response associated with immutable candidate IDs;
- terminal admission state;
- accepted provenance and propagation permission;
- optional trace diagnostics for ignored or unusable decisions.

Serialized workflow columns must remain JSON-compatible and portable through
the existing DataDesigner plugin boundary.

### Pure detection semantics

Refactor `src/anonymizer/engine/detection/postprocess.py` around pure helpers for:

- source-bound observation construction;
- decision indexing and single-use application;
- compact validation conflict groups for overlapping candidates;
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
-> coalesce and derive
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
column. At minimum, record ignored unknown IDs, duplicate decisions, missing or
malformed decisions, fallback reasons, reclassification fallback, and the
accepted origin references retained after coalescing. Keep these diagnostics
out of the public `COL_FINAL_ENTITIES` schema and final user dataframe; they are
available through the existing trace dataframe for debugging and evaluation.

## Validation Strategy

### Unit and composed tests

Add behavior-focused tests using fabricated data for at least the following
cases:

| Scenario | Required result |
|---|---|
| Redundant exact emissions from one source and route | One observation reaches validation, retaining the highest score from that calibrated source |
| Exact emissions from different sources or routes | They remain independent through admission so each decision and permission is preserved |
| Candidate is reclassified | Original correlation ID remains valid; resulting semantic identity uses the new normalized label |
| Reclass decision has no proposed label | Original exact span remains fallback-protected under its original eligible label and propagation is disabled |
| Reclass decision proposes an excluded or disallowed label | Candidate is rejected by eligibility before reconciliation and is not restored under its old label |
| Detector candidates overlap | Every candidate is represented once in the validation template; the source excerpt contains one neutral group tag rather than duplicated alternative tags |
| A longer overlapping candidate is dropped | An independently accepted shorter candidate remains eligible for final reconciliation |
| One overlap group crosses a validation chunk boundary | Each chunk maps the neutral group only to its pending candidate IDs; every candidate receives at most one merged decision |
| Accepted seed candidates still overlap before augmentation | The augmenter sees one neutral group tag plus every seed in the structured inventory, without duplicated source text |
| Augmented candidate constructs the same label/bounds as a seed | The seed decision is not replayed against the augmented candidate |
| Unknown decision ID | No candidate is mutated; diagnostic is recorded |
| Duplicate decision ID across or within chunks | First valid decision wins deterministically |
| Missing, null, or malformed decision | Exact detector span remains fallback-protected; it grants no accepted provenance or propagation |
| Labels differ only by case or surrounding whitespace | Exact spans coalesce under normalized identity |
| One exact origin is accepted and another is rejected | Only the accepted origin contributes provenance and permissions |
| One exact accepted origin permits propagation and another does not | Coalesced candidate permits propagation without losing either accepted origin |
| Excluded longer span overlaps allowed shorter span | Shorter candidate survives because eligibility runs first |
| Same-source GLiNER alternatives have identical bounds | Native detector confidence breaks the label tie |
| Cross-source candidates carry synthetic and native scores | Scores are not compared across calibration domains |
| Span-restricted full name is split | Derived parts do not appear at unrelated occurrences |
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

- missing or null decisions;
- unknown and duplicate IDs;
- malformed structured outputs;
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

### Separate correlation identity from semantic identity

Two identities add terminology, but they serve different purposes. An immutable
correlation ID prevents decision replay; normalized label/bounds identity makes
coalescing correct after reclassification. Reusing one identity for both is the
source of several current edge cases.

### Preserve fallback protection without granting authority

Keeping an undecided detector span is safer than silently exposing it, but
treating it as fully accepted can broaden protection to unrelated occurrences.
The explicit fallback state preserves the exact span while preventing that
authority leak.

### Centralize source priority

A fixed internal priority is less flexible than user-configurable precedence,
but it gives stable behavior and keeps public configuration small. The policy
can evolve in one place as new detector sources are added.

## Rollout Plan

1. **Completed:** land PR #282 and base this work on the resulting `main`.
2. Add lifecycle types and characterization tests for current behavior.
3. Introduce immutable candidate IDs while preserving the existing validator
   response schema and model roles; add only the compact conflict-group prompt
   input required to represent overlapping candidates.
4. Introduce single-use admission and remove downstream decision replay.
5. Move eligibility before exact coalescing and overlap reconciliation.
6. Move provenance, propagation, name splitting, and occurrence expansion onto
   the admitted-candidate contract.
7. Replace route-specific overlap calls with the single final reconciler.
8. Run focused, full, export, and benchmark validation.
9. Rebase or reconstruct PR #265 so regex contributes observations and route
   policy through these shared contracts rather than owning separate merge and
   decision-replay machinery.
10. Document these semantics as invariants for the later graph/SDK detection
    stage and preserve them when its execution assembly changes.

## Review Decisions Requested

Maintainer review should confirm:

1. Candidate IDs are short, immutable, row-local correlation tokens and are
   separate from normalized semantic identity.
2. The current validation response schema and model roles remain unchanged;
   prompt input changes only to represent overlapping candidates compactly.
3. A `reclass` decision without a proposed label retains exact-span fallback
   protection under the original eligible label and cannot authorize occurrence
   propagation; a non-empty excluded or disallowed proposed label is rejected
   by eligibility rather than restored under the old label.
4. Missing or unusable validator decisions retain exact-span fallback
   protection but grant no accepted provenance or propagation permission.
5. First valid result wins when duplicate validation decisions are returned.
6. Eligibility completes before any candidate can suppress another.
7. Exact coalescing unions accepted provenance and propagation permission.
8. All admitted and derived candidates use one final reconciliation policy.
9. Source priority remains an internal centralized policy rather than a public
   configuration option.
10. The graph/SDK refactor consumes these semantics instead of redefining them.
