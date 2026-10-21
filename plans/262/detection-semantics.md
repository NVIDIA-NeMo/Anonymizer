<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Detection Candidate Semantics

[GitHub issue #262](https://github.com/NVIDIA-NeMo/Anonymizer/issues/262) is the
tracking issue for this foundational detection work and the subsequent regex
implementation in [PR #265](https://github.com/NVIDIA-NeMo/Anonymizer/pull/265).
No separate detection-semantics issue is required.

> [!IMPORTANT]
> This is a draft implementation plan. Inline **Review decision needed**
> callouts identify passages that are not yet settled. They are intentionally
> left beside the relevant design text so reviewers can comment in context.

## Goal

Define one internal lifecycle for entity candidates before adding another
detection source. The lifecycle must make candidate identity, validation,
eligibility, provenance, augmentation grounding, and overlap reconciliation
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
- Which origins actually survived their configured admission route?
- How is every accepted claim tied to the specific source occurrence that
produced it?
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
  -> materialize augmented values at every matching source offset
  -> split full names and resolve overlaps (legacy heuristic)
  -> apply the original validation decisions to the merged candidates again
  -> apply allowed/excluded label filters
  -> propagate accepted values to other occurrences and resolve overlaps again
  -> final entities
```

The same logical candidate can therefore be represented by several mutable
records, and destructive conflict resolution occurs at several points.


| Current behavior                                                                                  | Failure it permits                                                                                                                                                                                                                                                                               | Concrete example                                                                                                                                                                                                                                                   | Proposed resolution                                                                                                                                                                                                                                  |
| ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| IDs are constructed from label and offsets                                                        | Reclassification or exact-candidate coalescing can change the representative identity used to find a decision                                                                                                                                                                                    | `Seattle` is detected as `country_0_7` and reclassified as `city`; its ID still encodes `country`, while a later reconstructed `city_0_7` cannot correlate to the original decision.                                                                               | Give every original observation an immutable reference within its `DetectionResult`, independent of mutable semantic fields, then use short request-local aliases only for validation.                                                               |
| The same decision payload is looked up again after augmentation by a constructed entity ID        | A decision can govern an observation that was not included in the validation request when a later origin produces the same constructed ID. Its production frequency is unmeasured; it changes current augmenter behavior as well as independently configured detection routes. | A GLiNER observation is dropped by validation. A later regex backend independently emits the same label and bounds under a rule configured to bypass LLM validation. Replaying the GLiNER decision would incorrectly veto the accepted regex observation.          | Consume each decision exactly once into the admission result of the observation in its request. A later observation follows its own admission route. Exact equivalent accepted claims may coalesce only after those independent admission decisions. |
| Chunk responses are merged against a row-wide candidate inventory                                 | A response can decide a candidate assigned to a different validation request                                                                                                                                                                                                                     | Chunk A owns `c1` but returns `c2=drop`; chunk B owns `c2` and returns `c2=keep`. Row-wide lookup plus first-wins merging allows chunk A to override chunk B.                                                                                                      | Bind aliases and decisions to the exact dispatched request inventory; wrong-request aliases have no authority, and response ordering is independent of completion order.                                                                             |
| Raw origins and accepted origins share the same entity representation                             | A rejected or undecided origin can accidentally contribute accepted provenance                                                                                                                                                                                                                   | If the validator omits a detector candidate, the current fallback keeps the same `EntitySpan`; later merging cannot tell that it was retained only as a privacy-safe fallback.                                                                                     | Keep observations, validation results, and admitted candidates as distinct states; fallback protection preserves only the exact observed span and grants no accepted provenance.                                                                     |
| Exact identity is compared inconsistently                                                         | Equivalent labels may fail to coalesce and lose accepted provenance                                                                                                                                                                                                                              | Detector label `token` and augmenter label `TOKEN` cover the same bounds, but case-sensitive comparison treats them as competing spans and retains only one representative.                                                                                        | Define semantic identity as normalized label plus exact source bounds, then coalesce equivalent eligible claims without discarding accepted origins.                                                                                                 |
| Provenance is collapsed into a representative source string                                       | Coalescing can discard an independently accepted origin, while a disabled route can appear to support the retained claim                                                                                                                                                                         | The detector and augmenter independently accept the same email span, but overlap resolution retains one `source` string. With regex added, blindly combining origins could also retain a model origin even when model detection is disabled for that regex entity. | Apply configured source and route eligibility first, then aggregate only independently admitted, configuration-eligible origins.                                                                                                                     |
| Overlapping detector candidates are suppressed before validation                                  | A longer candidate that is later dropped or excluded can erase a valid shorter candidate before either decision is known                                                                                                                                                                         | GLiNER returns `John Doe` as `full_name` and `John` as `first_name`; parsing keeps only the longer span. If validation drops `John Doe`, the original `John` candidate is already gone.                                                                            | Retain every conflicting observation through validation, represent the overlapping region once with a compact conflict group, and reconcile only admitted candidates.                                                                                |
| Allowed/excluded checks can occur after an overlap winner is selected                             | An ineligible longer candidate can suppress an eligible shorter candidate and then be removed                                                                                                                                                                                                    | An excluded `full_name` span wins over an allowed overlapping `first_name` during parsing; exclusion then removes the winner, leaving neither span protected.                                                                                                      | Complete admission and normalized label/source-route eligibility before any candidate can suppress another.                                                                                                                                          |
| Accepted `full_name` values are split into name-part candidates and matched throughout the source | A full-name decision silently creates new `first_name`, `middle_name`, and `last_name` claims that no detector or model evaluated                                                                                                                                                                | Accepting `John Smith` also creates a `first_name` claim for every standalone `John`, even when those occurrences refer to another person.                                                                                                                         | Remove `_split_full_names()` and treat `full_name` as an ordinary detected label; no name-part candidates are synthesized.                                                                                                                           |
| Parsing and augmentation resolve overlaps independently                                           | Results can depend on which route produced a candidate and when it entered the workflow                                                                                                                                                                                                          | Identical GLiNER spans can be ranked by native confidence during parsing, while a GLiNER/augmenter conflict can ignore scores and fall through to label ordering. A candidate discarded in either pass never reaches the final comparison.                         | Collect every admitted candidate, coalesce exact equivalent claims, then apply one documented final reconciliation policy with source-aware score comparison.                                                                                        |


## Terminology and Internal Model

The implementation should use different types, or equivalently strict state
objects, for the following concepts. These remain internal to
`anonymizer.engine`.

Lifecycle state that crosses a workflow-column or DataDesigner serialization
boundary uses Pydantic models. Identity-bearing models are frozen, reject
unknown fields, and use tuples instead of mutable lists. Definitions below use
this common base:

```python
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenStrictModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
```

Pure helper-local state that never crosses a workflow or serialization boundary
may use a frozen dataclass, but it must be converted into these lifecycle models
before publication. This PR introduces only the concrete GLiNER translation
boundary; a shared backend protocol should be added only when common
orchestration consumes more than one backend through that contract.

DataDesigner does not need to mutate or store live frozen model instances. At a
workflow-column boundary, Anonymizer writes JSON-compatible dictionaries from
`model_dump(mode="json")`, as the current detection custom columns already do.
The receiving Anonymizer column reconstructs and validates the corresponding
model with `model_validate()`. `frozen=True` prevents accidental in-process
assignment while the object is live; `extra="forbid"` detects an unexpected
serialized shape when it is reconstructed. Neither setting changes
DataDesigner's scheduling or DataFrame transport.

### Source text and view

`SourceTextRef` identifies the canonical text against which entity offsets are
interpreted. It is a structured internal reference containing:

- an opaque record ID supplied by the execution boundary; and
- an opaque field ID identifying the text within that record.

On the current DataFrame path these values are `RECORD_ID_COLUMN` and
`COL_TEXT`. The domain type does not depend on pandas or expose either value to
users. It does not hash source text: identical text in different records remains
distinct, while different views of the same field retain the same
`SourceTextRef`.

```python
class SourceTextRef(FrozenStrictModel):
    record_id: str = Field(min_length=1)
    field_id: str = Field(min_length=1)
```

The serialized shape is deliberately plain and stable:

```json
{
  "record_id": "<opaque execution record ID>",
  "field_id": "__nemo_anonymizer_text_input__"
}
```

The current DataFrame adapter constructs that value as
`SourceTextRef(record_id=str(row[RECORD_ID_COLUMN]), field_id=COL_TEXT)` after
record IDs have been attached. This is an execution-scoped source reference,
not a content-derived ID or a new public input field.

`SourceView` identifies the contiguous portion of canonical text examined by a
detection component. GLiNER and regex report spans relative to this view; the
augmentation grounding step uses the same view when converting contextual
matches into canonical offsets. The view contains the `SourceTextRef` and
canonical start and end offsets. The current full-text path uses `[0, len(text))`. A
detection component operating on a view reports `view_relative_start` and
`view_relative_end` relative to the beginning of the exact text in that view.
Before publishing an observation, the component converts those view-relative
offsets into canonical source offsets:

```python
class SourceView(FrozenStrictModel):
    source_text_ref: SourceTextRef
    start_position: int = Field(ge=0)
    end_position: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_geometry(self) -> SourceView:
        if self.end_position < self.start_position:
            raise ValueError("source view end must not precede its start")
        return self
```

```text
canonical_start = source_view.start_position + view_relative_start
canonical_end   = source_view.start_position + view_relative_end
```

For example, if a view begins at canonical offset `13` and a detection component
reports the span `[0, 5)` within that view, the published observation uses
canonical bounds `[13, 18)`. On the current full-text path the view begins at
`0`, so the view-relative and canonical offsets are identical.

Only canonical offsets enter observations, admission, coalescing, or
reconciliation. This lets overlapping views of one source describe the same
occurrence while preventing equal label/bounds pairs from different fields or
records from coalescing.

### Observation

An observation is one detection component's claim about a source span before
that claim has passed its configured admission route.

Required fields:

- `observation_id`: immutable identifier allocated by the detection component
before its result is published;
- `detection_result_id`: immutable identifier for that component invocation;
- the canonical `SourceTextRef` inherited from the `DetectionResult`;
- exact source `value` taken from canonical
`text[start_position:end_position]`;
- original `label`, `start_position`, and `end_position`;
- source kind and source instance identity;
- source-native score when one exists;
- admission route.

```python
class AdmissionRoute(str, Enum):
    llm_validation = "llm_validation"
    direct_acceptance = "direct_acceptance"


class ObservationRef(FrozenStrictModel):
    detection_result_id: str = Field(min_length=1)
    observation_id: str = Field(min_length=1)


class Observation(FrozenStrictModel):
    detection_result_id: str = Field(min_length=1)
    observation_id: str = Field(min_length=1)
    source_text_ref: SourceTextRef
    value: str = Field(min_length=1)
    label: str = Field(min_length=1)
    start_position: int = Field(ge=0)
    end_position: int = Field(gt=0)
    source_kind: str = Field(min_length=1)
    source_instance: str = Field(min_length=1)
    admission_route: AdmissionRoute
    native_score: float | None = Field(default=None, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_geometry(self) -> Observation:
        if self.end_position <= self.start_position:
            raise ValueError("observation must cover a non-empty span")
        if len(self.value) != self.end_position - self.start_position:
            raise ValueError("observation value length must match its source bounds")
        return self
```

`source_kind` identifies the recognized lifecycle and precedence role, such as
`initial_detector` or `augmenter`; it is not tied to a specific model
implementation. PR #265 adds the distinct roles `regex_user` and
`regex_builtin` so equal-geometry regex conflicts preserve the reviewed user-
rule precedence without depending on rule or route ordering.
`source_instance` identifies the configured implementation or component
instance, such as the current GLiNER detector or a future span-tagging backend.
They remain strings rather than a closed enum so another backend can implement
the private contract without changing the serialized schema. Source priority
is keyed by the recognized `source_kind` values in the central reconciliation
table; arbitrary source strings do not create precedence.

Origin and admission are independent fields. Observations from the current
initial detector use `source_kind="initial_detector"` with `llm_validation`;
regex observations added by PR #265 use `source_kind="regex_user"` or
`source_kind="regex_builtin"` with `llm_validation` or `direct_acceptance`
according to their rule; and grounded augmenter observations use
`source_kind="augmenter"` with `direct_acceptance`. Their admission reason is
`augmenter_output`, which explains why the direct route was used without
encoding the origin into `AdmissionRoute`.

The pair `(detection_result_id, observation_id)` is the immutable observation
reference. It is source-scoped, survives collection and downstream reuse, and
must be allocated independently of backend completion order. It does not
promise stability across separate anonymizer runs and must not be rebuilt after
reclassification, coalescing, or overlap resolution.

ID allocation follows one order. `EntityDetectionWorkflow`, before invoking a
detection component, allocates an opaque execution-local identifier such as `r1`,
unique within the current row's detection execution. This coordinator is the
"execution boundary"; GLiNER is one invoked component and does not choose the
result ID. After that invocation succeeds, its adapter
validates each raw claim and sorts the valid claims deterministically by
canonical bounds, normalized label, admission route, and descending native
score. It then assigns opaque result-local observation IDs (`o1`, `o2`, ...)
and publishes the frozen `DetectionResult`. Neither ID value encodes a label or
source offset, and model completion order cannot change an already allocated
result ID.

Validation uses a second identity: a short alias such as `c1`, assigned while
building one validation request. That alias is unique only inside its request
inventory and maps to exactly one immutable observation reference. Retries of
the same logical request retain the same inventory and aliases. Short aliases
are never used as provenance or regenerated during merging.

The scopes deliberately nest rather than compete. For example, validation alias
`c1` can map to `ObservationRef(detection_result_id="r1", observation_id="o2")`. `r1` identifies the component invocation and its source
view, `o2` identifies one observation within that result, and `c1` is only the
compact name shown to the model in one request. The mapping prevents aliases
from colliding across requests, avoids asking the model to reproduce internal
lineage IDs, and lets the original observation remain identifiable after
reclassification. Downstream state retains the observation reference and drops
the request alias.

### Detection result

A `DetectionResult` is the successful output of one detection-component
invocation over one `SourceView`. It owns the ordered observations. An empty
observation tuple represents a successful invocation that found nothing.
Terminal component failures do not publish a `DetectionResult`; they continue
through the existing `NddAdapter` failure path and surface as `FailedRecord`.

```python
class DetectionResult(FrozenStrictModel):
    detection_result_id: str = Field(min_length=1)
    source_view: SourceView
    source_kind: str = Field(min_length=1)
    source_instance: str = Field(min_length=1)
    observations: tuple[Observation, ...] = ()

    @model_validator(mode="after")
    def validate_result(self) -> DetectionResult:
        observation_ids = [item.observation_id for item in self.observations]
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("observation IDs must be unique within a result")
        for observation in self.observations:
            if observation.detection_result_id != self.detection_result_id:
                raise ValueError("observation belongs to another result")
            if observation.source_text_ref != self.source_view.source_text_ref:
                raise ValueError("observation belongs to another source text")
            if (
                observation.source_kind != self.source_kind
                or observation.source_instance != self.source_instance
            ):
                raise ValueError("observation belongs to another component")
            if (
                observation.start_position < self.source_view.start_position
                or observation.end_position > self.source_view.end_position
            ):
                raise ValueError("observation falls outside its source view")
        return self
```

### Validation request and decision

Each validation request carries an immutable request inventory that binds:

- its request ID, source record, admission route, and effective label policy;
- every request-local alias to one immutable observation reference;
- the observation's authoritative label, value, and absolute source bounds;
- the source-context window and its absolute start and end offsets.

```python
class EffectiveLabelPolicy(FrozenStrictModel):
    allowed_labels: tuple[str, ...] | None = None
    excluded_labels: tuple[str, ...] = ()


class ValidationInventoryEntry(FrozenStrictModel):
    alias: str = Field(pattern=r"^c[1-9][0-9]*$")
    conflict_group_id: str | None = Field(
        default=None,
        pattern=r"^g[1-9][0-9]*$",
    )
    observation_ref: ObservationRef
    value: str = Field(min_length=1)
    label: str = Field(min_length=1)
    start_position: int = Field(ge=0)
    end_position: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_geometry(self) -> ValidationInventoryEntry:
        if self.end_position <= self.start_position:
            raise ValueError("inventory entry must cover a non-empty span")
        if len(self.value) != self.end_position - self.start_position:
            raise ValueError("inventory entry value length must match its source bounds")
        return self


class ValidationRequestInventory(FrozenStrictModel):
    request_id: str = Field(min_length=1)
    source_text_ref: SourceTextRef
    context_view: SourceView
    source_context: str
    effective_label_policy: EffectiveLabelPolicy
    entries: tuple[ValidationInventoryEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_inventory(self) -> ValidationRequestInventory:
        aliases = [entry.alias for entry in self.entries]
        if len(aliases) != len(set(aliases)):
            raise ValueError("request-local aliases must be unique")
        observation_refs = [entry.observation_ref for entry in self.entries]
        if len(observation_refs) != len(set(observation_refs)):
            raise ValueError("request-local observation references must be unique")
        if self.context_view.source_text_ref != self.source_text_ref:
            raise ValueError("context view belongs to another source text")
        if len(self.source_context) != (
            self.context_view.end_position - self.context_view.start_position
        ):
            raise ValueError("context text does not match its source view")
        for entry in self.entries:
            if (
                entry.start_position < self.context_view.start_position
                or entry.end_position > self.context_view.end_position
            ):
                raise ValueError("inventory entry falls outside its context")
        return self
```

`source_context` is the unmodified canonical substring described by
`context_view`; neutral conflict tags are derived only while rendering the
prompt. The inventory deliberately copies the authoritative value, label, and
geometry used to build that prompt. The observation reference remains the
identity used to apply the response. Echoed model fields never replace either
source.

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
elsewhere in the detection execution;
- before normalizing or evaluating any decision, reject a parsed response when
an alias assigned to that request appears more than once, even when the
duplicate entries are identical;
- discard every decision from a response rejected for duplicate assigned
aliases and retry the same logical request with the same immutable inventory,
aliases, source context, and effective label policy;
- treat `keep` and `drop` as usable, and treat `reclass` as usable only when it
contains a non-empty proposed label;
- when a semantically valid response has no usable decision for one of its
assigned observations, preserve only that exact observed span as fallback
protection;
- when a validation request exhausts its attempts or fails terminally, preserve
the workflow's existing failed-record behavior instead of converting every
assigned observation into fallback protection.

A duplicate assigned alias is a semantic response failure, not an omission. It
is detected after structured parsing but before proposed-label normalization or
usability checks, so response ordering can never choose between conflicting
entries. No partial decisions from the invalid attempt are applied. With a
validator pool, semantic retry proceeds to the next configured alias in pool
order. With a single validator alias, the dispatcher permits one bounded
same-alias semantic retry. If no attempt returns a semantically valid response,
the request follows the existing terminal `FailedRecord` path.

This semantic retry does not create a new validation request: the request ID,
candidate inventory, request-local aliases, source excerpt, and policy remain
unchanged. Attempt identity and model alias may change and are recorded in trace
diagnostics without recording sensitive entity values. Only a response that
passes the one-entry-per-assigned-alias check may treat an omitted alias or one
unique unusable entry as exact-span fallback protection.

Keep the current `validation_max_entities_per_call` default of `100` in this PR.
Duplicate detection, conflict-group atomicity, and retry semantics must remain
correct for every supported positive chunk size. Track a possible reduction to
`25` as a follow-up issue: smaller inventories may reduce correlation errors and
the blast radius of an invalid response, but they also increase calls, repeated
prompt context, latency, and cost. Benchmark those effects before changing the
public default.

Observation identity must never be inferred by matching echoed values, labels,
offsets, or aliases from another request.

A `reclass` decision without a proposed label preserves the exact original span
under its original eligible label as fallback protection. When a non-empty
proposed label is excluded or falls outside an explicit allowed-label set,
normal eligibility rejects the reclassified candidate; the engine must not
restore it under a knowingly incorrect original label.

### Admission result

Applying validation produces one terminal admission result per observation in
the current detection execution:

- `accepted`: the route explicitly accepted the observation, possibly with a
successfully resolved reclassified label;
- `rejected`: the route explicitly rejected it;
- `fallback_protected`: a successful validator response did not contain a
usable decision for its assigned observation, so the existing fail-safe may
preserve the observed span without treating the origin as validated, or a
`reclass` decision did not contain a usable proposed label and the original
span must remain protected.

```python
class AdmissionStatus(str, Enum):
    accepted = "accepted"
    rejected = "rejected"
    fallback_protected = "fallback_protected"


class AdmissionReason(str, Enum):
    validator_keep = "validator_keep"
    validator_reclass = "validator_reclass"
    validator_drop = "validator_drop"
    missing_decision_fallback = "missing_decision_fallback"
    unusable_reclass_fallback = "unusable_reclass_fallback"
    direct_acceptance = "direct_acceptance"
    augmenter_output = "augmenter_output"
    ineligible_label = "ineligible_label"


class AdmissionResult(FrozenStrictModel):
    observation_ref: ObservationRef
    route: AdmissionRoute
    status: AdmissionStatus
    effective_label: str | None = None
    reason: AdmissionReason

    @model_validator(mode="after")
    def validate_result(self) -> AdmissionResult:
        accepted_reasons = {
            AdmissionReason.validator_keep,
            AdmissionReason.validator_reclass,
            AdmissionReason.direct_acceptance,
            AdmissionReason.augmenter_output,
        }
        rejected_reasons = {
            AdmissionReason.validator_drop,
            AdmissionReason.ineligible_label,
        }
        fallback_reasons = {
            AdmissionReason.missing_decision_fallback,
            AdmissionReason.unusable_reclass_fallback,
        }
        llm_reasons = {
            AdmissionReason.validator_keep,
            AdmissionReason.validator_reclass,
            AdmissionReason.validator_drop,
            *fallback_reasons,
        }
        direct_reasons = {
            AdmissionReason.direct_acceptance,
            AdmissionReason.augmenter_output,
        }
        if self.status is AdmissionStatus.rejected:
            if self.effective_label is not None:
                raise ValueError("a rejected result has no effective label")
            valid_reasons = rejected_reasons
        elif not self.effective_label:
            raise ValueError("a retained result requires an effective label")
        elif self.status is AdmissionStatus.accepted:
            valid_reasons = accepted_reasons
        else:
            valid_reasons = fallback_reasons
        if self.reason not in valid_reasons:
            raise ValueError("admission reason does not match its status")
        if (
            self.reason in llm_reasons
            and self.route is not AdmissionRoute.llm_validation
        ):
            raise ValueError("validator reason requires LLM validation")
        if (
            self.reason in direct_reasons
            and self.route is not AdmissionRoute.direct_acceptance
        ):
            raise ValueError("direct reason requires direct acceptance")
        return self
```

This distinction preserves privacy-safe exact-span behavior while ensuring that
an undecided observation cannot grant accepted provenance. A decision is
consumed while producing this result and is never consulted again downstream.

Admission is observation-scoped. A decision governs only the observation in
its request, not every past or future observation that happens to produce the
same normalized label and bounds. This is primarily a lifecycle invariant, not
a claim that one model frequently contradicts itself across validation and
augmentation prompts. It matters whenever observations have independent
admission routes, especially when a regex rule bypasses LLM validation.

Different prompts can still lead the same model to disagree, and deployments
may intentionally use different models for validation and augmentation. In
either case, an earlier rejection is not a global veto over a later observation
that the rejecting model never evaluated.

Validation chunks, retries, and validator-pool calls collectively implement
this one logical validation pass; they do not create additional admission runs.
Admission state remains a separate object that references the immutable
observation rather than being encoded into or mutating the observation.

`fallback_protected` is internal lifecycle state, not an entity label or a new
public result field. When such a candidate survives final reconciliation, the
public entity retains its ordinary label and exact bounds. The fallback status
and reason remain available only in internal admitted-candidate state and trace
diagnostics.

### Admitted candidate

An admitted candidate is eligible for protection after its admission route,
label scope, and configured source/route checks have completed. It contains:

- immutable lineage back to its originating observation references;
- the canonical source reference, current label, and exact canonical bounds;
- accepted provenance references containing the detection result ID,
observation ID, source kind, and source instance only for origins that
survived an enabled route;
- whether it is explicitly accepted or retained only by fallback protection;
- source-native confidence retained only for comparisons where that score is
meaningful.

```python
class AcceptedOrigin(FrozenStrictModel):
    observation_ref: ObservationRef
    source_kind: str = Field(min_length=1)
    source_instance: str = Field(min_length=1)
    route: AdmissionRoute


class NativeScore(FrozenStrictModel):
    observation_ref: ObservationRef
    source_kind: str = Field(min_length=1)
    source_instance: str = Field(min_length=1)
    score: float = Field(allow_inf_nan=False)


class AdmittedCandidate(FrozenStrictModel):
    source_text_ref: SourceTextRef
    value: str = Field(min_length=1)
    label: str = Field(min_length=1)
    start_position: int = Field(ge=0)
    end_position: int = Field(gt=0)
    lineage: tuple[ObservationRef, ...] = Field(min_length=1)
    accepted_origins: tuple[AcceptedOrigin, ...] = ()
    native_scores: tuple[NativeScore, ...] = ()
    has_fallback_protection: bool

    @property
    def has_explicit_acceptance(self) -> bool:
        return bool(self.accepted_origins)

    @model_validator(mode="after")
    def validate_candidate(self) -> AdmittedCandidate:
        if self.end_position <= self.start_position:
            raise ValueError("candidate must cover a non-empty span")
        if len(self.value) != self.end_position - self.start_position:
            raise ValueError("candidate value length must match its source bounds")
        if not (self.has_explicit_acceptance or self.has_fallback_protection):
            raise ValueError("candidate requires an admitted protection state")
        lineage = set(self.lineage)
        accepted_origin_refs = {origin.observation_ref for origin in self.accepted_origins}
        if not accepted_origin_refs <= lineage:
            raise ValueError("accepted origins must reference candidate lineage")
        native_score_refs = {score.observation_ref for score in self.native_scores}
        if not native_score_refs <= accepted_origin_refs:
            raise ValueError("native scores must reference accepted origins")
        return self
```

Before coalescing, an admitted candidate normally has one lineage entry. Exact
coalescing may produce multiple lineage entries and accepted origins, ordered
deterministically. Rejected observations never enter `lineage` or
`accepted_origins`. A fallback-only candidate has
`has_fallback_protection=True` and no accepted origins, so its derived
`has_explicit_acceptance` property is false. A candidate supported by both
states retains fallback state while its explicitly accepted observations appear
in `accepted_origins` and make `has_explicit_acceptance` true.

Accepted origins and lineage are serialized in a deterministic order, not as
an unordered set. This keeps exported workflow state and trace comparisons
stable without collapsing distinct origins into one source string or requiring
downstream code to parse source-name prefixes.

An admitted candidate's semantic identity is:

```text
(source_text_ref, normalize_label(label), start_position, end_position)
```

This identity is used for coalescing equivalent accepted claims. It is separate
from the immutable observation reference and the request-local alias used for
decision correlation.

## Semantic Invariants

The implementation must make the following properties true:

1. **Source identity and offsets are authoritative.** Every observation names
  one canonical `SourceTextRef`; every view-relative span is mapped through its
   `SourceView`; and every materialized value equals canonical
   `text[start_position:end_position]`.
2. **Observation identity is immutable.** Backends assign observation IDs
  within an immutable `DetectionResult` before publication. Reclassification
   and collection never replace them, and allocation never depends on
   completion order.
3. **Validation authority is request-scoped.** A decision may affect only the
  observation bound to its alias in that dispatched request inventory.
4. **Decisions are single-use.** Each selected usable decision is applied to
  its original observation exactly once.
5. **Unknown or duplicate output is non-authoritative.** Unknown or
  wrong-request aliases cannot mutate a candidate. A repeated assigned alias
   invalidates the response before any of its decisions are applied.
6. **Omission is not execution failure.** A missing or unusable decision in a
  semantically valid response may produce exact-span fallback protection;
   terminal request failure preserves failed-record behavior.
7. **Every retained observation completes its configured admission route.**
  Overlapping and exact-duplicate observations remain independent candidates
   through admission. Observations assigned to LLM validation appear exactly
   once in that prompt; a future directly accepted regex observation bypasses
   the validator by configuration rather than by accident.
8. **Conflict groups are validation-atomic.** Every LLM-validated candidate in
  one connected overlap group is assigned to the same validation request, so
   the validator never decides a validated alternative without seeing the rest
   of that validated group. Direct-admission routes remain outside the request
   and join the common candidate pool afterward.
9. **Only accepted, configuration-eligible origins grant authority.** Rejected,
  undecided, or disabled source routes cannot contribute accepted provenance.
10. **Eligibility precedes suppression.** Excluded, disallowed, unsupported,
  disabled-source, or route-rejected candidates cannot remove an otherwise
   eligible candidate.
11. **Exact coalescing follows admission.** Equivalent source/label/span claims
  coalesce only after each observation completes its own admission route.
    Rejected observations do not contribute provenance to the result.
12. **Every protected occurrence has source-bound evidence.** It must come from
  a detector observation, an exact augmenter value-and-context match, or exact-
    span fallback protection. The shared lifecycle never performs blind value-
    only expansion and never synthesizes name parts from `full_name`.
13. **Augmentation admission is prompt-independent.** Augmenter outputs never
  consume seed-validator decisions, whether the augmenter prompt uses seed
    context or plain source text.
14. **Coverage precedes source.** For different overlapping bounds, prefer the
  longer protected span regardless of origin. Source priority may break ties
    only after span geometry.
15. **Scores are compared only within their calibration domain.** A GLiNER
  confidence may rank otherwise equivalent GLiNER alternatives, but it is not
    compared with synthetic `1.0` values from augmentation.
16. **Reconciliation is deterministic.** Input ordering, validation chunking,
  or detector execution order cannot change the final result for the same set
    of observations and decisions.
17. **Final spans are valid and non-overlapping per source.** Every output span
  is within its canonical source bounds, contains the exact canonical source
    value, and survives one final overlap policy for that `SourceTextRef`.

## Proposed Lifecycle

```text
source text
  |
  +-> detector backend -> DetectionResult -----------------+
  |       assign immutable result-scoped references         |
  |       retain all conflicting alternatives              |
  |                                                        |
  |       build request inventory + short aliases          |
  |       route observations by configured admission       |
  |       -> validate: keep / drop / reclass once          |
  |       -> or direct admission for an enabled route      |
  |       -> accepted / rejected / fallback-protected       |
  |                                                        |
  |       apply label + source/route eligibility            |
  |       -> admitted seed candidates                       |
  |                                                        v
  +-> augmentation model ------------------------------ candidate pool
          output admission is independent of prompt form    |
          -> emit label + value + adjacent context          |
          -> ground each item to one canonical span         |
          -> assign immutable observation references         |
          -> apply label + source/route eligibility          |
          -> admitted augmentation candidates                |
                                                             |
                 coalesce exact semantic identities <--------+
                 preserve eligible accepted provenance
                                  |
                 one deterministic final overlap policy
                                  |
                 final entities + tagged text + trace state
```

### Step 1: Produce observations through a narrow backend contract

Introduce a private `_EntityDetectionBackend` contract whose successful result
is one ordered `DetectionResult` for a detector invocation on a `SourceView`.
The result contains its immutable detection result ID, canonical
`SourceTextRef`, view bounds, backend kind and instance, ordered observations,
and may contain an empty observation tuple. A terminal backend failure publishes
no result and continues through the existing `NddAdapter`/`FailedRecord` path.
The initial request contract contains the source view text and effective label
set; any later dependency on upstream detector context must be represented as
an explicit typed input rather than inferred from execution order or silently
ignored.

The backend owns detector-specific execution and translation into source-bound
observations. The shared lifecycle owns validation admission, label and route
eligibility, provenance, augmentation grounding, and reconciliation. Adapt
GLiNER first while preserving the existing DataDesigner execution and export
path. PR #265 adds regex as another detection backend. Each backend must publish
the specific occurrences it found. Collection retains each backend's original
`DetectionResult` and observation references; forwarding observations through
a later backend cannot manufacture a second supporting origin.

For this PR, GLiNER consumes source text and the effective label set; it does
not consume an earlier backend's observations merely because it executes later.
When PR #265 defines a regex-to-GLiNER context mode, that dependency must be a
typed input. GLiNER then publishes only its new claims under GLiNER observation
references, while collection preserves the original regex result alongside the
new GLiNER result. Forwarded regex observations retain their regex identity and
route restrictions and are not republished as GLiNER support.

The private contract is synchronous or declarative as required by the existing
DataDesigner workflow. It does not prescribe an asynchronous graph execution
API. A later graph binding may own dependencies, retries, request association,
resource limits, and execution artifacts while consuming the same observation
results and lifecycle semantics.

### Step 2: Parse without destructive conflict resolution

Validate backend observations, convert view-relative offsets to canonical source
offsets, preserve their immutable references, and derive exact values from the
canonical source. Exact source slicing follows PR #282. Do not discard
overlapping alternatives while parsing: a longer candidate that is later
rejected or excluded must not have already removed a valid shorter candidate.

Every structurally valid raw detector item becomes its own `Observation`,
including an exact duplicate emitted by the same component invocation. Each
observation follows its configured admission route independently. Exact claims
from the same or different sources remain separate through admission; only
admitted candidates with the same normalized label and canonical bounds
coalesce afterward.

### Step 3: Preserve the current validator contract

Validation input continues to include the text context needed to make a
decision. The validator response contract and validator model role remain
unchanged in this PR:

```json
{
  "decisions": [
    {
      "id": "c1",
      "decision": "keep",
      "proposed_label": "",
      "reason": "person name in context"
    },
    {
      "id": "c2",
      "decision": "reclass",
      "proposed_label": "city",
      "reason": "Seattle is a city"
    }
  ]
}
```

The request skeleton and immutable inventory supply each candidate's value and
label. Skeleton `start_position` and `end_position` are zero-based character
offsets within the untagged source excerpt: start inclusive, end exclusive,
excluding inserted tag markup. They are calculated by subtracting
`context_view.start_position` from the inventory's canonical bounds. Inventory
offsets remain absolute positions in the original source text, including for a
full-text request over a source view that begins at a nonzero position.

The engine resolves each response against its original observation; the
model does not supply authoritative value, label, source span, or candidate
identity beyond its request-local alias. Any extra echoed fields are ignored.

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

Conflict-group IDs are request-local aliases (`g1`, `g2`, ...) allocated in
source order. The renderer wraps the union of each connected overlap group once
with the prompt-only label `NAGROUP_<group_id>` through the same
`_format_entity_tag()` and row `TagNotation` used elsewhere. For `g1`, the four
renderings are:

- `xml`: `<NAGROUP_g1>{region}</NAGROUP_g1>`
- `bracket`: `[[{region}|NAGROUP_g1]]`
- `paren`: `((SENSITIVE:NAGROUP_g1|{region}))`
- `sentinel`: `<<SENSITIVE:NAGROUP_g1>>{region}<</SENSITIVE:NAGROUP_g1>>`

There is exactly one group tag around each connected overlapping source region,
not one tag per candidate. For example, if `John Smith` has candidates
`full_name [0, 10)`, `first_name [0, 4)`, and `last_name [5, 10)`, the prompt
contains one `<NAGROUP_g1>John Smith</NAGROUP_g1>` wrapper and the structured
inventory lists all three candidates under `g1`. A request contains multiple
group tags only when it contains separate, non-overlapping conflict regions;
for example, overlapping alternatives for `John Smith` may be `g1`, while
overlapping alternatives for `New York` later in the same text are `g2`.

`NAGROUP_g1` is not an entity label, observation, or provenance source. The
structured inventory remains authoritative for every candidate in `g1`; the
wrapper only identifies the region in the prompt without duplicating source
text. Entity labels matching the internal `NAGROUP_g<positive integer>` form
case-insensitively are reserved and rejected during label configuration or
model-response validation.

The connected conflict group is the atomic unit for validation chunking. Build
groups first, retain deterministic source order, and pack whole groups into
requests targeting at most `validation_max_entities_per_call` candidates. Every
candidate in a group appears in one authoritative request, and the model
returns decisions for all overlapping alternatives together. A group is never
split across requests.

`validation_max_entities_per_call` is a batching target. If one conflict group
alone exceeds it, send that group intact in its own request and log a warning
containing only the group size and configured target. Do not add other groups
to that request, split the group, or silently omit alternatives. Provider
context limits still apply, and a terminal model-call failure retains the
existing `FailedRecord` behavior.

Tag only the observations assigned to the current request. Other observations
inside its source excerpt remain plain text, retaining surrounding context
without presenting unvalidated detector labels as hints or suggesting the
request can decide those neighboring observations. This differs from the old
excerpt builder, which tagged every detected entity contained in the window.

Decision-only validation and conditional reclassification change the response
schema, model roles, cost, latency, and quality behavior. That work is tracked
separately in [issue #304](https://github.com/NVIDIA-NeMo/Anonymizer/issues/304) rather than added to issue #262.

Each response is interpreted only against its dispatched request inventory.
Wrong-request aliases are ignored even when they happen to name a valid alias
or observation in another chunk. A parsed response containing the same assigned
alias more than once is discarded in full and retried; normalization, usability
checks, and decision application occur only after the response passes this
cardinality check. If a semantically valid response omits an assigned
observation or supplies one unusable decision for it, that observation becomes
`fallback_protected`, preserving the exact observed span without granting
accepted provenance. Exhausted attempts or a terminal request
failure continue through the existing failed-record path instead.

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

A valid reclassification is checked under its proposed label before it can
enter reconciliation. Neither explicit acceptance nor fallback protection
creates claims at other occurrences of the same value.

Configuration is an authority ceiling, not another origin to merge. A source
or detection route disabled for a label does not enter coalescing and cannot
contribute provenance through an exact duplicate. When PR #265 is rebuilt on
this lifecycle,
`detect_additional_matches=False` for a regex entity therefore makes
model-origin candidates for that label ineligible;
`validate_matches_with_llm` independently selects the admission route for the
regex observation itself.

No later transform receives the raw decision map as an input. In particular,
the finalization step must not reapply detector decisions to augmented,
coalesced, or independently admitted entities.

### Step 5: Admit augmentation through a separate route

Augmenter prompt construction is independent of augmentation admission. This foundational PR preserves the current seed-aware prompt. While that prompt remains, admitted seed candidates are rendered without silently omitting overlapping alternatives: use one neutral candidate-group tag for the source region and keep the group-to-candidate mapping in the structured seed inventory. PR #267 can later switch the prompt to plain source text and remove seed hints without changing the output-admission contract described here but will require extensive testing.

Unlike seed validation, augmentation is open-ended structured extraction: it
does not answer an expected inventory of validation aliases. Unknown-alias,
duplicate-decision, and missing-decision policies therefore do not apply. Each
augmenter item defines one exact contextual match with this conceptual payload:

```json
{
  "value": "May",
  "context_before": "told ",
  "context_after": " to stop laughing",
  "label": "first_name",
  "reason": "May is the person being addressed"
}
```

The corresponding LLM-output schema is defined here because it belongs to the
augmentation contract, not the frozen lifecycle state:

```python
class AugmentedEntitySchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str = Field(min_length=1)
    context_before: str = ""
    context_after: str = ""
    label: str = Field(min_length=1)
    reason: str | None = None
```

`value` is the exact source substring to protect. `context_before` and
`context_after` are exact, immediately adjacent source substrings. Prompt the
model to copy enough surrounding text to establish why the label applies:
approximately five words on each side for whitespace-delimited text, or a
comparably informative phrase for text without whitespace boundaries. Near a
source boundary, use whatever context is available. The model must not invent,
normalize, or paraphrase context.

This is prompt guidance, not a schema-enforced word, token, or character count.
Token counts are model-specific, word counts are not language-universal, and a
hard minimum fails near source boundaries. At least one context field must be
non-empty unless `value` is the entire `SourceView`. Both fields may be empty
only when the exact source view equals `value`.

These fields contain canonical source text, not the seed tags shown in the
transitional `COL_INITIAL_TAGGED_TEXT` prompt. The prompt must state this
explicitly and provide canonical-source examples. This avoids introducing an
internal marker label, marker parsing, or another `TagNotation` collision rule.

The grounding helper finds every exact occurrence of `value` in the
`SourceView`, then retains every occurrence for which the immediately preceding
source ends with `context_before` and the immediately following source starts
with `context_after`. One item may therefore materialize several observations
when the same exact local context occurs more than once. For each retained
match, the helper computes authoritative canonical offsets and value:

```text
view_relative_start = contextual_match_value_start
view_relative_end   = view_relative_start + len(value)
canonical_start     = source_view.start_position + view_relative_start
canonical_end       = source_view.start_position + view_relative_end
value               = canonical_text[canonical_start:canonical_end]
```

> [!WARNING]
> **Review decision needed — transitional grounding:** The augmenter still sees
> seed-tagged text until PR #267, while grounding must resolve against canonical
> source text. Confirm the smallest prompt change that reliably asks the model
> to copy `value`, `context_before`, and `context_after` from the underlying
> source without copying engine-inserted seed tags. PR #267 removes this
> transitional mismatch by showing plain source text.
>
> It is technically possible to match the returned context against tagged text,
> but those offsets are not canonical source offsets. The engine would then need
> a deterministic rendered-text-to-source offset map and rules for context that
> crosses inserted tags. Stripping tag-looking text afterward is unsafe because
> equivalent text may have existed in the user's source. The simpler contract is
> for the model to return canonical source substrings; what remains unsettled is
> whether this PR should also show canonical text to the augmenter, or retain the
> transitional tagged prompt and rely on an instruction to omit inserted tags.
>
> Benchmark both variants on the same records. In particular, measure whether
> the plain-text prompt rediscovers already supplied seed entities rather than
> finding genuinely missed entities.

The model never supplies offsets. Its `value` and context locate matching
occurrences; for every match, the helper derives the authoritative value from
`canonical_text[start_position:end_position]` while constructing a frozen
augmentation `Observation`.

Multiple contextual matches are valid and do not trigger retry. For example,
`context_before="told "`, `value="May"`, and
`context_after=" to stop laughing"` materializes both matching occurrences in
`She told May to stop laughing. They told May to stop laughing again.` If
different occurrences require different adjacent context to establish the
label, the augmenter returns separate items.

Missing or non-verbatim values or context, a non-empty-source item with both
contexts empty, or a combination that matches zero occurrences makes the
augmentation response semantically invalid. Discard the entire response; do
not admit the otherwise valid items from that attempt. Retry the same logical
augmentation request through the existing model alias once; augmentation has
one configured `entity_augmenter` alias rather than a model pool. The retry
prompt includes a value-free repair message that identifies each failing item
by its response index and failure category, restates that `value`,
`context_before`, and `context_after` must be copied exactly from the canonical
source, and requires a complete replacement response rather than only the
corrected items. For example: `Items 1 and 3 could not be grounded
(non_verbatim_context, zero_contextual_matches). Return the complete entity
list again, correcting those items and copying all grounding text exactly from
the source.` It must not echo sensitive values in the repair message. If the
second semantic attempt also fails, follow the existing terminal
`FailedRecord` path. Do not fuzzy-match, guess an occurrence, or fall back to
value-only expansion.

Implement augmentation as a custom column that owns the model call, structured
schema parsing, exact grounding checks, and the one bounded same-alias semantic
retry through the existing private model facade. Provider-level transport
retries remain separate from this semantic retry. Record both semantic attempts
under the augmenter alias with attempt index, outcome, and a value-free failure
category in the existing trace path. A terminal schema or grounding failure
raises from the custom column so `NddAdapter` reports the row through its
existing `FailedRecord` behavior.

Each successfully grounded item receives a new immutable observation reference
and augmenter origin. It then passes exact-value, bounds, label-normalization,
and configured source/route eligibility checks before joining the candidate
pool.

Augmentation output is already a model decision rather than a high-recall
detector observation, so it does not reuse the seed validator decision map.
Exact duplicate suggestions are admitted independently and then handled by
semantic coalescing, not by an augmentation-specific overlap pass.

Because augmentation performs open-ended multilingual extraction, must copy
source text exactly, and is not followed by another validation stage,
deployments should use a model demonstrated to perform this task reliably. The
public API does not attempt to classify or reject models as "strong": that term
is not enforceable. Instead, benchmark schema validity, contextual-grounding
rate, precision, recall, multilingual behavior, and output completeness for the
chosen model, including records with many requested entities. A more capable
model does not remove output-token limits, so completeness remains an explicit
benchmark dimension.

### Step 6: Coalesce exact semantic identities

Group admitted candidates by normalized label and exact bounds. Coalescing:

- keeps the source value from authoritative offsets;
- retains lineage to all contributing observations;
- retains provenance only from accepted origins whose source and route remain
enabled by configuration;
- does not let rejected or configuration-disabled origins add provenance;
- retains calibrated score information by source rather than manufacturing a
cross-source score.

Coalescing occurs only after every contributing observation has completed its
own admission route. A rejected GLiNER observation therefore does not veto an
independently accepted augmenter observation or a regex observation configured
to bypass LLM validation, even when the claims have the same normalized label
and bounds. This is an observation-lifecycle rule; it is not based on an
expectation that such cross-route disagreement is frequent.

Same bounds with different normalized labels are not exact duplicates. They
remain alternatives for final reconciliation.

### Step 7: Remove generic value propagation and name splitting

Delete `_split_full_names()` and `expand_entity_occurrences()` from the shared
detection lifecycle. A `full_name` observation remains a `full_name` at its
observed bounds; the engine does not synthesize `first_name`, `middle_name`, or
`last_name` candidates from it.

The shared lifecycle does not infer that identical text has the same meaning at
another offset. GLiNER must emit each occurrence it detects. Regex must emit
each pattern match. The augmenter may materialize one or more occurrences only
when each has the exact value and adjacent context supplied by its suggestion.
Exact-span fallback protection applies only to the original observation.
Replacement may still assign a consistent substitute to equal accepted values,
but that later grouping does not create detection claims.

### Step 8: Reconcile once

Run one pure final reconciliation function after all candidates are admitted
and coalesced. The comparator must be documented in one place and
must provide a total, deterministic order.

The initial policy is:

1. prefer the longer canonical source span;
2. prefer the earlier start, then earlier end;
3. for identical bounds with incompatible labels, prefer an explicitly
  accepted candidate over a fallback-protected candidate;
4. use the centrally defined internal source priority;
5. within the same calibrated detector source and identical bounds, prefer the
  higher native confidence;
6. use normalized label and immutable lineage as stable final tie-breakers.

Apply that policy with one deterministic greedy pass per `SourceTextRef`:

1. sort every eligible, coalesced candidate by the comparator above;
2. visit candidates in that order;
3. accept a candidate when its half-open interval does not overlap any already
  accepted interval, otherwise discard it from final output; and
4. sort the accepted candidates by canonical source position for publication.

Touching intervals such as `[0, 5)` and `[5, 10)` do not overlap and both
survive. The algorithm is intentionally longest-first rather than an optimizer
for the greatest sum of protected character counts. It mirrors the selection
shape of the current `resolve_overlaps()` helper while moving selection after
validation, eligibility, and exact coalescing and applying it only once.

The current helper's greedy loop is not itself the failure being corrected.
The current pipeline invokes it on different partial candidate sets with
different score behavior. Raw GLiNER parsing can discard a shorter alternative
before a longer alternative is later dropped by validation; mixed detector and
augmenter conflicts run without comparable-score handling and can fall through
to alphabetical label order; and later augmentation, name splitting, and
propagation passes resolve already-reduced sets again. The proposed lifecycle
retains all observations until their admission and eligibility are known, then
runs the greedy loop once over the complete admitted set.

For an exact coalesced candidate with multiple independently accepted origins,
its effective source priority is the highest-priority recognized
`source_kind` among its eligible accepted origins. Rejected or disabled origins
are absent and cannot improve that priority. A fallback-only candidate has no
accepted origin and therefore gains no source priority; after the explicit-
acceptance comparison, deterministic label and lineage tie-breakers resolve a
fallback-only tie. Native confidence is compared only when the candidates'
highest-priority origins belong to the same calibrated `source_kind` and
`source_instance`; otherwise the score step is skipped.

Coverage is evaluated before provenance. A complete `john.smith@example.com`
augmentation therefore beats an overlapping detector claim for `Smith`. The
same rule applies when a detector directly emits a narrower span inside a
broader accepted augmentation. Choosing the longer span does not expose the
narrower text; it protects at least the same source region.

For partial overlap, longer-first reconciliation does not synthesize a union.
If `[0, 5)` and `[3, 10)` conflict, `[3, 10)` wins and `[0, 3)` is not retained
unless another admitted candidate covers it. All alternatives are validated
together before this decision, and the behavior is documented and tested.

For identical bounds, coverage is equal. Explicit acceptance, centralized
source priority, same-source calibrated confidence, normalized label, and
immutable lineage then provide deterministic tie-breakers. This preserves
detector authority for an exact-bound detector/augmenter disagreement without
allowing source lineage to defeat greater protected coverage.

Source priority is an internal reconciliation policy, not a user-facing
configuration knob. It must not depend on which route happens to be passed
first by a caller. For the sources on `main`, the initial order for otherwise
identical geometry is `initial_detector`, then `augmenter`. The current GLiNER
implementation is recorded in `source_instance`, not encoded into the priority
key. PR #265 extends the equal-geometry order to `regex_user`,
`regex_builtin`, `initial_detector`, then `augmenter`, without adding another
merge implementation. `source_instance` continues to identify the particular
rule or configured implementation for trace detail.

## Affected Subsystems

- `anonymizer.engine.detection`: candidate construction, validation input,
private detector backend boundary, admission, augmentation grounding, exact
coalescing, and final reconciliation, including removal of
`_split_full_names()` and `expand_entity_occurrences()`;
- `anonymizer.engine.schemas`: JSON-compatible internal observation, admission,
provenance, and diagnostic payloads;
- `anonymizer.engine.workflow_columns.detection`: serialized transform
operations for the revised lifecycle;
- `anonymizer.engine.constants`: lifecycle and trace column names;
- detection workflow, serialization, replace-mode, rewrite-mode, and trace
tests.

`anonymizer.config` and `anonymizer.interface` retain their existing public
signatures, defaults, and result schemas. The bundled anonymizer skill therefore
does not require a public-API update for this PR.

## Implementation Shape

The exact class names can follow repository conventions, but responsibilities
should be separated as follows.

### Schemas

Implement the concrete frozen Pydantic lifecycle models defined above in
`src/anonymizer/engine/schemas/detection.py`. That module represents:

- structured `SourceTextRef` and contiguous `SourceView` geometry;
- immutable detection-result and observation IDs;
- validation request inventories and their request-local aliases;
- successful backend results, including an explicit empty observation tuple;
- the existing validator response associated with its authoritative request;
- terminal admission state;
- accepted provenance;
- optional trace diagnostics for ignored or unusable decisions.

Serialized workflow columns must remain JSON-compatible and portable through
the existing DataDesigner plugin boundary.

### Detector backend boundary

Add a private `_EntityDetectionBackend` contract and `DetectionResult` schema
under `src/anonymizer/engine/detection/`. The first adapter wraps the existing
GLiNER declaration and parsing path without bypassing `NddAdapter.run_workflow()`
or adding new detector-backend configuration to the public `Detect` API. The
boundary publishes source-bound observations after successful execution. It
does not own validation, eligibility, coalescing, reconciliation, or the
existing `FailedRecord` path.

The contract represents a successful result with no observations using an empty
tuple. It assigns observation references before publishing a successful result
and preserves those references when results are collected. Terminal execution
failure publishes no `DetectionResult` and remains owned by
`NddAdapter`/`FailedRecord`. Its initial inputs are explicit source text and the
effective label set. Any future backend reuse adds a typed input and defined
behavior; unsupported input fails clearly rather than being silently ignored.

### Pure detection semantics

Refactor `src/anonymizer/engine/detection/postprocess.py` around pure helpers for:

- source-bound observation construction;
- view-relative to canonical source-offset conversion;
- request-inventory construction, decision indexing, and single-use application;
- compact validation conflict groups for overlapping candidates;
- context-constrained augmentation value-and-context grounding without overlap suppression
or seed-decision replay;
- normalized eligibility;
- exact-identity coalescing;
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
column. At minimum, record request identity, attempt identity, validator alias,
ignored unknown or wrong-request aliases, duplicate assigned-alias response
failures, assigned candidates missing a usable decision from a semantically
valid response, fallback reasons, terminal request failure, reclassification
fallback, augmentation grounding failures, and the accepted origin references
retained after coalescing. Do not record sensitive entity values in retry
diagnostics. Keep these diagnostics out of the public `COL_FINAL_ENTITIES`
schema and final user dataframe; they are available through the existing trace
dataframe for debugging and evaluation.

## Validation Strategy

### Unit and composed tests

Add behavior-focused tests using fabricated data for at least the following
cases:


| Scenario                                                                                          | Required result                                                                                                                                                                                     |
| ------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Lifecycle models are mutated after construction                                                   | Frozen Pydantic models reject assignment; tuple fields prevent mutation of ordered collections                                                                                                      |
| Serialized lifecycle state contains an unknown field                                              | Parsing fails because lifecycle models use `extra="forbid"`                                                                                                                                         |
| A `SourceView`, observation, inventory entry, or admitted candidate has invalid geometry          | Model construction fails before the state reaches workflow transforms                                                                                                                               |
| Lifecycle models serialize and restore through JSON                                               | Identity, source geometry, routes, ordered lineage, and provenance round-trip without type loss                                                                                                     |
| Detector succeeds with no observations                                                            | Backend emits a successful `DetectionResult` with an empty observation list rather than a failure                                                                                                   |
| Detector fails terminally                                                                         | No `DetectionResult` is published; the failure follows the existing `NddAdapter`/`FailedRecord` behavior                                                                                            |
| A `DetectionResult` contains repeated observation IDs                                             | Model construction fails because observation IDs must be unique within their result                                                                                                                 |
| Equal label and bounds occur in different records or fields                                       | Distinct `SourceTextRef` values keep the observations and semantic identities separate                                                                                                              |
| Overlapping views detect the same source occurrence                                               | View-relative offsets map to the same canonical bounds and exact semantic identity                                                                                                                  |
| A backend emits bounds outside its declared view                                                  | Observation construction rejects the invalid geometry before admission                                                                                                                              |
| GLiNER executes after another backend without a typed data binding                                | It consumes only source text and effective labels; execution order alone does not alter its observations                                                                                            |
| An upstream `DetectionResult` is explicitly reused by a later backend                             | Collection preserves the original result and adds only the later backend's new claims; forwarded observations do not gain another supporting origin                                                 |
| Detection results and request inventories are serialized and restored                             | Source references, view mappings, immutable references, alias bindings, canonical geometry, policy, and ordering survive the round trip                                                             |
| A detector emits the same semantic claim twice                                                    | Both valid items receive distinct observation IDs and admission results; accepted equivalents coalesce only after admission                                                                         |
| Raw emissions arrive in a different order                                                         | Deterministic sorting produces the same result-local observation-ID assignment before publication                                                                                                   |
| Exact emissions from different sources or routes                                                  | They remain independent through admission so each route applies its own decision before accepted claims coalesce                                                                                    |
| Detection results complete in different orders                                                    | Their preassigned observation references and collected output are unchanged                                                                                                                         |
| Candidate is reclassified                                                                         | Original observation reference remains valid; resulting semantic identity uses the new normalized label                                                                                             |
| Reclass decision has no proposed label                                                            | Original exact span remains fallback-protected under its original eligible label                                                                                                                    |
| Reclass decision proposes an excluded or disallowed label                                         | Candidate is rejected by eligibility before reconciliation and is not restored under its old label                                                                                                  |
| Detector candidates overlap                                                                       | Every candidate is represented once in the validation template; the source excerpt contains one `NAGROUP_g<id>` tag rendered with the row's active notation rather than duplicated alternative tags |
| A conflict group is rendered under each supported `TagNotation`                                   | XML, bracket, paren, and sentinel output use the documented `NAGROUP_g<id>` form and preserve the source region exactly once                                                                        |
| An entity label matches `NAGROUP_g<positive integer>` case-insensitively                          | Configuration or model-response validation rejects the reserved prompt-only label form                                                                                                              |
| Repeated candidate values occur in one conflict group                                             | Excerpt-relative bounds distinguish occurrences in the prompt; canonical bounds remain authoritative in the inventory                                                                                |
| Validation uses a clipped source excerpt or a source view with a nonzero start                     | Skeleton offsets index the untagged excerpt; inventory offsets still index the canonical source, and aliases bind responses to that inventory                                                      |
| A longer overlapping candidate is dropped                                                         | An independently accepted shorter candidate remains eligible for final reconciliation                                                                                                               |
| Adding the next conflict group would cross a validation chunk boundary                            | The entire group moves to the next request; no group is split and source-order packing is deterministic                                                                                             |
| One conflict group exceeds `validation_max_entities_per_call`                                     | Send the group intact in its own request and log its size and the configured target without entity values; no unrelated group joins that request                                                    |
| Accepted seed candidates still overlap while the current seed-aware augmenter prompt remains      | The augmenter sees one `NAGROUP_g<id>` tag plus every seed in the structured inventory, without duplicated source text; PR #267 can later remove this prompt dependency without changing admission  |
| Augmented candidate constructs the same label/bounds as a seed                                    | The augmentation observation follows its own admission route; the seed decision is not replayed against it; accepted equivalent claims coalesce afterward                                           |
| Rejected detector observation and independently accepted exact augmenter observation              | The rejected observation contributes no provenance, while the augmenter observation remains admitted                                                                                                |
| Rejected detector observation and exact regex observation configured to bypass LLM validation     | The detector decision cannot veto the regex observation; the accepted regex claim remains eligible                                                                                                  |
| Augmenter returns empty context and `value` is the entire source view                             | The engine accepts the full-view value and derives its canonical offsets                                                                                                                             |
| Augmenter returns empty context for a value embedded in a larger source view                      | The response is semantically invalid and retried; the engine does not perform value-only expansion                                                                                                  |
| Augmenter returns an exact value with adjacent context that matches one occurrence                | The engine materializes that occurrence and derives canonical offsets from the source                                                                                                                |
| Augmenter value and adjacent context match several occurrences                                   | Every exact contextual match becomes its own observation; multiple matches do not trigger retry                                                                                                     |
| Augmenter returns non-verbatim `value`, `context_before`, or `context_after`                      | The response is semantically invalid and retried without fuzzy matching or rewriting model output                                                                                                   |
| The same sensitive value appears under different adjacent contexts                                | The augmenter returns separate items for the different contexts; each item materializes all of its exact contextual matches                                                                          |
| Augmenter value and context match zero occurrences                                                | The response is semantically invalid and retried without fuzzy matching; exhausted attempts follow failed-record behavior                                                                            |
| Prompt guidance requests surrounding context in whitespace and non-whitespace languages           | It asks for approximately five words per side or a comparably informative phrase, but schema validation imposes no language-specific length minimum                                                   |
| An augmenter response contains many contextual suggestions or matches                             | Benchmark output completeness, materialized-observation count, and token-limit behavior rather than imposing an arbitrary context-length cap                                                         |
| Augmenter returns duplicate grounded suggestions for the same label/bounds                        | Independently admitted exact identities coalesce deterministically without an augmentation-specific overlap pass                                                                                    |
| Broader augmentation candidate overlaps an admitted detector candidate                            | The broader candidate wins by protected coverage regardless of source                                                                                                                               |
| Two candidates partially overlap without containment                                              | Longer-first reconciliation selects one admitted span and does not synthesize a union; the uncovered unique prefix or suffix is documented in the expected result                                   |
| Candidates `[0, 6)`, `[0, 4)`, and `[4, 8)` are all admitted                                      | Greedy longest-first reconciliation retains `[0, 6)` only; it does not replace the winner with two shorter spans to maximize aggregate covered characters                                             |
| A short bridge candidate overlaps two longer candidates that do not overlap each other             | The two longer non-overlapping candidates survive and the bridge is discarded; reconciliation does not select only one winner for the entire connected conflict component                           |
| Two admitted candidates touch at one boundary                                                      | Half-open intervals do not overlap, so both candidates survive                                                                                                                                       |
| Detector and augmenter candidates have identical bounds and incompatible labels                   | Explicit admission and centralized source priority resolve the equal-geometry tie deterministically                                                                                                 |
| User and built-in regex candidates have identical bounds and incompatible labels                  | After PR #265 adds its source roles, `regex_user` wins over `regex_builtin` regardless of admission route or input order                                                                              |
| Built-in regex and initial-detector candidates have identical bounds and incompatible labels      | After PR #265 adds its source roles, the centralized table applies the documented `regex_builtin` versus `initial_detector` ordering                                                                 |
| An exact coalesced candidate has accepted initial-detector and augmenter origins                   | Its effective source priority is `initial_detector`; both eligible accepted origins remain in provenance                                                                                            |
| A coalesced candidate also has a rejected or configuration-disabled higher-priority origin         | That origin does not contribute provenance or improve effective source priority                                                                                                                      |
| Fallback-protected detector candidate overlaps an augmentation candidate                          | Coverage wins for different bounds; for identical bounds, explicit acceptance wins without granting fallback authority                                                                              |
| Unknown decision alias                                                                            | No observation is mutated; diagnostic is recorded                                                                                                                                                   |
| Chunk A returns an alias assigned to chunk B                                                      | Chunk A cannot mutate chunk B's observation; the wrong-request alias is traced                                                                                                                      |
| Identical duplicate decision aliases within one response                                          | The entire response is discarded and retried; no decision from the invalid attempt is applied                                                                                                       |
| Conflicting duplicate decision aliases within one response                                        | The entire response is discarded and retried without selecting by normalization, usability, or response order                                                                                       |
| Duplicate `reclass` entries contain whitespace and a usable label                                 | Cardinality validation runs first, so the response is retried rather than selecting either proposed label                                                                                           |
| Validator pool receives a response with a duplicate assigned alias                                | Retry proceeds to the next configured validator alias with the same request inventory, aliases, source context, and policy                                                                          |
| A single validator alias returns a duplicate assigned alias                                       | The same alias receives one bounded semantic retry for the unchanged logical request                                                                                                                |
| Every semantic attempt is invalid                                                                 | No partial decision is applied and the request follows the existing terminal `FailedRecord` path                                                                                                    |
| Semantic retry succeeds                                                                           | Admission uses only the valid retry response and retains the original request inventory and aliases                                                                                                 |
| Assigned candidate is absent from a semantically valid structured response                        | Exact detector span remains fallback-protected; it grants no accepted provenance                                                                                                                    |
| Validation chunk fails terminally                                                                 | Its row follows existing failed-record behavior rather than converting assigned candidates to fallback protection                                                                                   |
| Validation is chunked, retried, or distributed across a model pool                                | The calls collectively produce one terminal admission result per observation in the current detection execution                                                                                     |
| Default validation chunk configuration is constructed                                             | The existing `validation_max_entities_per_call=100` default is unchanged; explicit positive overrides remain supported                                                                              |
| Labels differ only by case or surrounding whitespace                                              | Exact spans coalesce under normalized identity                                                                                                                                                      |
| One exact origin is accepted and another is rejected                                              | Only the accepted origin contributes provenance                                                                                                                                                     |
| A model origin duplicates a regex entity whose model-detection route is disabled                  | The model origin contributes no provenance; the regex policy remains authoritative                                                                                                                  |
| A model origin duplicates a regex entity whose model-detection route is enabled                   | Independently accepted origins coalesce without erasing either origin                                                                                                                               |
| Excluded longer span overlaps allowed shorter span                                                | Shorter candidate survives because eligibility runs first                                                                                                                                           |
| Same-source GLiNER alternatives have identical bounds                                             | Native detector confidence breaks the label tie                                                                                                                                                     |
| Cross-source candidates carry synthetic and native scores                                         | Scores are not compared across calibration domains                                                                                                                                                  |
| The same admitted candidate set is supplied in different orders                                   | The greedy winner set and source-ordered final output are identical                                                                                                                                 |
| Accepted multi-token `full_name` is finalized                                                     | It remains one `full_name` entity; no `first_name`, `middle_name`, or `last_name` candidates are synthesized at its bounds or elsewhere                                                             |
| Standalone name part repeats outside an accepted `full_name`                                      | The standalone value is not protected unless a detector or augmenter independently supplies that occurrence                                                                                         |
| GLiNER detects one of two identical strings                                                       | Only the observed occurrence is admitted; the shared lifecycle does not create a claim at the second offset                                                                                         |
| Regex matches two occurrences                                                                     | The backend publishes two occurrence-specific observations, both of which follow the configured regex admission route                                                                               |
| Fallback seed coalesces with accepted augmentation                                                | The accepted augmentation origin supplies accepted provenance; fallback lineage remains distinguishable and does not become accepted authority                                                      |
| Augmented and detector spans conflict                                                             | Both use the same final comparator                                                                                                                                                                  |
| Candidate input order is permuted                                                                 | Final entities and provenance are unchanged                                                                                                                                                         |
| Accepted and fallback-protected candidates have identical bounds                                  | The explicitly accepted candidate wins before source priority is considered                                                                                                                         |


Retain end-to-end assertions for `COL_DETECTED_ENTITIES`, `COL_FINAL_ENTITIES`,
tagged text, grouped entities, strict labels, excluded labels, replace mode,
rewrite mode, workflow export, and chunked validator pools.

### Behavioral validation

Compare `main` with this branch on RAT-Bench, Nemotron-PII, and OpenPII,
using the same sampled records, model configuration, prompts except where this
plan intentionally changes them, and scoring procedure for each comparison.
For the unresolved augmenter-input decision, run both the seed-tagged and
canonical-plain-text prompt variants on those same samples. Report duplicate
rediscovery of already supplied seeds separately from newly recovered entities;
do not count rediscovered seeds as additional recall.
This PR preserves the validator's model role and response schema. Validation
prompt changes are limited to compactly representing overlapping candidates
that were previously discarded. The augmenter schema adds occurrence-level
`value`, `context_before`, and `context_after` fields used for grounding, so
augmentation quality and response validity must also be measured. Measure:

- candidates missing a usable decision;
- unknown and wrong-request aliases, plus responses rejected for duplicate
assigned aliases;
- successful omissions versus terminal validation-request failures;
- structured-output call or schema-validation failures at the workflow boundary;
- augmentation contextual-grounding success, matches materialized per item,
  semantic retries, and terminal grounding failures;
- mixed-validity augmentation responses, including whether the bounded repair
  returns a complete corrected response and whether any item from the rejected
  first attempt leaks into admission;
- augmentation output completeness and token usage as requested entity count
increases;
- keep/drop/reclass agreement and unresolved-decision handling;
- overall and per-label precision and recall;
- validator-drop rediscovery: count final entities whose only accepted origins
  are from the augmenter and whose normalized label and exact canonical source
  bounds match a GLiNER observation rejected with `validator_drop` in the same
  record field. Count each final entity once, not each supporting observation;
  fallback lineage does not count as an accepted origin. Report the count and
  its share of final entities and distinct validator-dropped seed claims;
- precision of those rediscovered entities against ground truth, alongside the
  overall precision and recall comparison with `main`;
- candidate coverage;
- occurrence-level recall for repeated sensitive values;
- false positives on ambiguous repeated strings such as person-name and month
uses of `May`;
- prompt and completion tokens;
- latency and failed-record rate.

Validator drops are no longer replayed after augmentation. An independently
admitted augmenter match of a dropped GLiNER claim can therefore survive
finalization, subject to label eligibility and final overlap resolution. This
can recover validator false negatives or reintroduce false positives. Its
frequency and benchmark impact are not yet measured; do not assume it is rare.
Use the measurements above to decide whether to show the augmenter rejected
spans as already reviewed or introduce an explicit policy for exact matches
between rejected detector claims and accepted augmenter claims. A high count
alone does not establish a precision problem; score those entities first.
Changing which source wins such disagreements is a deliberate policy decision,
not a reason to restore implicit decision replay. Carry this behavior note and
the benchmark results into PR #305's description when the implementation is
pushed.

The lifecycle change advances only if privacy quality does not regress
materially. Any increase in candidates, tokens, or model calls must be
attributable to validating previously suppressed conflicts, quantified in the
results, and accepted explicitly. Report RAT-Bench, Nemotron-PII, and OpenPII
separately as well as in aggregate so a gain on one dataset cannot conceal a
regression on another. Output differences must correspond to an intentional
correction captured by a focused test.

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
immutable `(detection_result_id, observation_id)` reference preserves detector
lineage; a short request-local alias constrains one validation response;
normalized label/bounds identity makes coalescing correct after
reclassification. Reusing one identity across those jobs causes decision
replay, cross-chunk authority, and mutable provenance.

### Preserve fallback protection without granting authority

Keeping an undecided detector span is safer than silently exposing it, but
treating it as fully accepted would misstate its provenance. The explicit
fallback state preserves only the observed span and keeps that distinction.

### Require every detection source to identify occurrences

An audit of 12 completed trace artifacts (986 full-run records across Gemma,
GPT-OSS, and Qwen on RAT-Bench, Nemotron-PII, and OpenPII) showed that blind
global value propagation (`expand_entity_occurrences()`) affected 74.5% of
RAT-Bench records while being dominated by broad open-class categories
(`occupation`, `field_of_study`, `employment_status`, and `education_level`
accounted for 64.4% of propagated spans, versus 5.4% for `first_name`), with
38%–44% manual precision due to case-insensitive homograph collisions (such as
`OR` -> `"or"`, `US` -> `"us"`, and `IT` -> `"It"`). In augmentation, 15.6% of
unique augmenter suggestions matched multiple non-seed occurrences and generated
42.1% of all non-seed augmenter materializations.

Removing global value propagation eliminates the assumption that equal strings
have equal meaning. GLiNER already runs at the configured low default threshold
of `0.3`; regex naturally emits every pattern match; and augmentation gains
context-constrained value-and-context grounding. Benchmark repeated-mention recall
and ambiguous homographs before accepting the change rather than preserving
blind expansion.

### Centralize source priority

A fixed internal priority is less flexible than user-configurable precedence,
but it gives stable behavior and keeps public configuration small. The policy
can evolve in one place as new detector sources are added.

### Prefer protected coverage before source

PR #267 intentionally makes augmentation operate on plain text, which can
increase duplicate and overlapping model discoveries. A blanket preference for
detector candidates would let a narrow detector span suppress a broader
accepted augmentation candidate. The final reconciler therefore ranks
different bounds by protected coverage first and uses source priority only for
otherwise identical geometry. When PR #267 is rebased, remove its any-overlap
seed filter, route its candidates through this reconciler, and measure coverage
changes caused by the clean-slate prompt.

## Rollout Plan

1. **Completed:** land PR #282 and base this work on the resulting `main`.
2. Add source-reference and view types, lifecycle types, the private backend
  contract, and characterization tests for current behavior.
3. Adapt GLiNER to publish an ordered `DetectionResult` with canonical source
  geometry and immutable references while preserving the existing DataDesigner
   execution and export path.
4. Add validation request inventories and compact conflict-group input with
  explicit source geometry while preserving the existing response schema and
   model roles.
5. Introduce request-scoped semantic response validation, bounded retry, and
  single-use admission; remove downstream decision replay while retaining the
   current validation chunk default.
6. Move eligibility before exact coalescing and overlap reconciliation.
7. Add context-constrained augmentation value-and-context grounding, then remove
  `_split_full_names()` and `expand_entity_occurrences()` so the shared
   lifecycle no longer synthesizes unobserved occurrences.
8. Replace route-specific overlap calls with the single final reconciler.
9. Run focused, full, export, and benchmark validation.
10. Rebase PR #267 so it changes the augmenter prompt to plain source text and
  removes seed hints while reusing shared augmentation admission and final
   reconciliation; drop its route-specific `resolve_overlaps()` filtering.
11. Rebase or reconstruct PR #265 so regex contributes detection results and
  route policy through these shared contracts rather than owning separate merge
   and decision-replay machinery.
12. Document these semantics as invariants for the later graph/SDK detection
  stage and preserve them when its execution assembly changes.

## Review Checklist

Maintainer review should verify that the implementation preserves these
settled invariants:

1. Serialized lifecycle state uses the documented frozen Pydantic models,
  rejects unknown fields, and preserves ordered tuple fields through JSON
   round trips. The backend interface itself is not a Pydantic model.
2. `SourceTextRef` identifies a canonical record field, `SourceView` maps
  view-relative offsets into that source, and semantic identity includes the
   source reference with normalized label and canonical bounds.
3. Backends allocate immutable observation references within a
  `DetectionResult` before publication; validation uses separate short aliases
   scoped to one dispatched request inventory.
4. The current validation response schema and model roles remain unchanged;
  prompt input changes only to represent overlapping candidates compactly.
5. A `reclass` decision without a proposed label retains exact-span fallback
  protection under the original eligible label; a non-empty excluded or
   disallowed proposed label is rejected by eligibility rather than restored
   under the old label.
6. Missing or unusable decisions in a semantically valid response retain
  exact-span fallback protection without authority; terminal request failure
   preserves existing failed-record behavior.
7. A response can decide only its dispatched inventory; a repeated assigned
  alias invalidates the entire parsed response before normalization or
   usability checks. The unchanged logical request is retried through the
   validator pool, or once with the same alias when no pool is configured, and
   exhausted semantic attempts preserve terminal failed-record behavior.
8. Conflict-group inventories include absolute source geometry and expose only
  candidates assigned to that request. Connected conflict groups are atomic:
   every overlapping alternative is assigned to and decided by the same
   request, and an oversized group is sent alone with a value-free warning.
9. Eligibility completes before any candidate can suppress another.
10. Exact claims from independent routes remain separate through admission and
  coalesce only afterward; rejected and disabled origins cannot regain
   provenance during merging.
11. Every protected occurrence comes from a backend observation, a grounded
  augmenter observation (`value` plus adjacent context), or exact-span fallback
  protection.
    The lifecycle performs neither global value propagation nor `full_name`
    splitting.
12. Different overlapping bounds are reconciled by protected coverage before
  source. Partial overlap does not synthesize a union span.
13. Augmenter prompt construction is independent of output admission: this PR
  preserves the current seed-aware prompt, and PR #267 may later switch it to
   plain text without adding another merge path.
14. All admitted candidates use one final reconciliation policy.
15. Source priority remains an internal centralized policy rather than a public
  configuration option.
16. The graph/SDK refactor consumes the private backend and lifecycle contracts
  without redefining their authority semantics.
