<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Phase 7 design — stable Substitute planning

Status: reviewed phase-specific design and test strategy; branch implementation authorization
is pending and cannot be requested before its prerequisites pass. This
document is subordinate to the complete development and research plan in the
[graph-native SDK RFC](graph-native-anonymizer-sdk-rfc.md). The checkpoint is not a separate
project acceptance decision and does not authorize a public graph or session API, production
Intake or OpenShell integration, durable replacement state, or stable promotion.

Review status: independent architecture and test-strategy council review completed on
2026-08-20 with zero unresolved Critical or Warning findings. The decision record below
preserves that reviewed design; it does not authorize implementation. The phase is not
implemented or tested. Work remains sequenced after the
Phase 4–6 gates, and its owned pre-implementation contract must be frozen before Phase 7
implementation begins and receives a separate operator checkpoint.

## Decision

Phase 7 should plan Substitute assignments once per declared coherence scope and store the
complete assignment bundle in a bounded invocation-private ledger. Every mention resolves
through an immutable replacement slot; no row generates or owns an independent replacement
map.

The first qualified profile is invocation-bounded. It guarantees consistent
assignments only among datums admitted to one coherence scope in one invocation. It does
not guarantee the same replacement across invocations, processes, workers, restarts, or
delivery attempts. A longer-lived private session or durable state backend requires a
separate capability and state-authority review.

Stable Substitute is a semantic guarantee about assignment reuse inside the admitted
scope. It is not a claim of durable idempotency, transactional persistence, exhaustive
detection, or absence of all PII.

## Branch decision record

The first phase 7 profile is invocation-bounded, and it is the only stability profile in
phase 7. Session-bounded consistency, durable consistency, cross-worker consistency, and a
governed state backend are deferred to a separately authorized design. They are not phase 7
implementation options.

The graph profile performs no automatic candidate regeneration. An invalid, failed,
cancelled, lost, inconsistent, partial, or unverifiable planning attempt fails closed under
the outcome and release rules below. Adding bounded regeneration would require a new review
of attempt identity, limits, cancellation, and stale-result behavior.

Before phase 7 implementation begins, the Anonymizer semantic and execution owners must
jointly publish and freeze one versioned implementation contract containing:

- the closed replacement-role and relational-constraint vocabulary, owned by the
  Anonymizer semantic owner;
- the matrix of slot pairs that must use distinct synthetic values, owned by the
  Anonymizer semantic owner;
- accepted count, byte, concurrency, and lifetime ceilings, owned by the execution owner;
  and
- the cleanup-observability contract for every terminal path, owned by the execution owner.

The contract is an input to compilation, candidate validation, the independent reference
model, and the frozen conformance manifest. Unknown versions, roles, constraints, or missing
host limits fail admission. This deliverable may be prepared while earlier phases proceed,
but it cannot be frozen against an implementation until phases 4–6 establish the accounting,
workframe, mention, cluster, and slot inputs it depends on.

## Contract and current behavior to preserve

Current public Substitute behavior is DataFrame-oriented. `LlmReplaceWorkflow` generates a
replacement map per row through `NddAdapter.run_workflow()`, filters unrequested entries,
and repairs a synthetic value that equals another protected original in the same row.
`ReplacementWorkflow` then applies the map with the existing offset replacement primitive.
The public trace compatibility surface retains replacement maps while the result DataFrame
does not expose them.

Phase 7 must not recast this behavior as graph-wide coherence. It adds a private graph
profile while preserving public constructors, configuration, `run()`, `preview()`,
`evaluate()`, result columns and attributes, `trace_dataframe`, `failed_records`, CLI
behavior, and errors. DataFrames remain temporary workframes, datum identity remains graph
identity, and `NddAdapter.run_workflow()` remains the sole DataDesigner execution boundary.

Until a public change is separately reviewed, the compatibility facade retains row-local
Substitute semantics on the legacy path, including its current same-row collision repair.
Phase 7 does not lower public Substitute through the stricter graph profile. A future
compatibility migration must define and review a distinct semantic profile before changing
that boundary. Cluster, slot, scope, reservation, and planning identities never appear in
public DataFrames or artifacts.

Phase 7 qualifies Substitute only. Current rewrite mode may use a replacement map in its
prompt, but graph-wide grouped rewrite, evaluation, and repair remain phase 8 work.

## Preconditions and non-goals

Phase 7 assumes the earlier gates have supplied:

- phase 4 exhaustive task, datum, dependency, atomic-group, and invocation accounting;
- phase 5 separately framed target and bounded context workframes; and
- phase 6 datum-anchored mentions, deterministic entity clusters, replacement-role
  classification, and group verification for local strategies.

This phase does not:

- infer entity clusters from repeated text, labels, DataFrame position, or source IDs;
- infer coherence scopes from context, dependencies, atomic groups, or source hierarchy;
- make the graph or ledger public or serializable;
- provide durable, restart, or cross-worker replacement consistency;
- move codecs, source reconstruction, persistence, retries, deduplication, retention,
  cleanup, or delivery into Anonymizer;
- retry replacement planning automatically;
- qualify grouped rewrite or independent-row fallback; or
- claim secure memory erasure when Python releases ledger references.

## Typed semantic model

Phase 7 keeps four identities distinct:

| Identity | Meaning | Must not be derived from |
| --- | --- | --- |
| Datum ID | Immutable graph datum | DataFrame index, row order, text, or source ID |
| Entity cluster ID | Mentions that refer to one semantic subject | Raw entity text alone |
| Replacement slot ID | One type-appropriate assignment reused by its mentions | Original or synthetic content, or a content hash |
| Coherence scope ID | The boundary inside which slot assignments are stable | Context, dependency, or atomic membership |

One entity cluster may own multiple replacement slots. For example, one person cluster can
have name, email, and phone slots whose values must be relationally consistent. Mentions
that must share one literal synthetic value reference the same slot. Mentions that refer to
the same subject but require different types reference different slots in the same cluster.

Slot and scope IDs are compiler-issued opaque values. Source identity remains in the
adapter's reconstruction state. Runtime workframes use invocation-private correlation
tokens and must not contain source IDs or public graph IDs.

The phase 4 terminal-accounting ledger and the phase 7 replacement-planning ledger are
separate state machines. The first proves exhaustive task, datum, dependency, group, and
invocation closure. The second retains only provisional slot assignments for the bounded
invocation. Phase 4 remains the only authority for releasing output.

The owned internal grammar should use immutable closed variants. Expected scope-planning
outcomes are:

```text
planned(bundle_ref)
blocked(reason_code)
failed(reason_code)
cancelled
lost
inconsistent(reason_code)
```

Only `planned` contains a private reference to a complete immutable assignment bundle.
`blocked` records that an admitted prerequisite prevented planning from running. All other
terminal outcomes contain no replacement values. Pending, reserved, and running are
internal states and cannot appear in a terminal invocation result. Outcome precedence and
fixed-point propagation follow the accepted phase 4 accounting model.

## Coherence-scope admission policy

For the first qualified profile, coherence scopes form a flat exact partition of target
datums:

- every target datum belongs to exactly one non-empty scope;
- every member resolves to one declared target datum;
- duplicate members and duplicate scopes are invalid;
- coverage gaps and implicit singleton completion are invalid;
- nesting and partial overlap are recognized but unsupported; and
- every entity cluster and replacement slot belongs to exactly one scope.

Reject a cluster that crosses scopes. Reject a mention whose datum, cluster, and slot do not
agree on the same scope. Scope membership order has no semantic meaning; declaration order
is only a deterministic presentation tie-break.

The compiler validates scope shape, cluster and slot references, declared bounds, strategy,
and required runtime capabilities before any additional graph-invocation effect: ledger or
workframe creation, graph telemetry, candidate dispatch, transformation, or publication. On
the current compatibility facade, providers and other host resources may already be bound
before the internally generated graph is compiled; phase 7 does not claim otherwise.
Unsupported semantics never degrade to row-local planning.

Phase 7 extends the phase 4 deterministic rejection order rather than creating a second
validator order:

1. malformed outer graph, type, or version;
2. total count and byte limits, including phase 7 scope, cluster, slot, mention, and bundle
   limits;
3. malformed or duplicate phase 4 datum, dependency, and atomic-group declarations;
4. unknown or wrong-purpose phase 4 references, invalid atomic partition, and dependency
   cycle;
5. empty or duplicate coherence declarations, duplicate members, coverage gaps, and partial
   overlap;
6. unknown, duplicate, mismatched, or cross-scope mention, cluster, and slot declarations;
7. recognized but unsupported coherence nesting, cardinality, role, or constraint grammar;
8. unsupported Substitute profile or missing runtime capability.

Compiler-issued scope, cluster, and slot IDs cannot collide by authored input. Tests instead
use duplicate semantic declarations or force a compiler invariant failure through a test
seam. Multiply-invalid tests cover adjacent tiers, declaration permutations, and exact-limit
versus one-over-limit cases.

Context scope, coherence scope, dependency, and atomic group remain independent. A datum
can use related context without sharing replacements, share replacements without depending
on another datum, or share a coherence scope across several atomic groups.

Two scopes have no assignment-sharing or cross-scope uniqueness guarantee. Equal original
text does not join them, and an accidentally equal synthetic value does not create a shared
slot. Any broader anti-linkability or global uniqueness policy would expand the privacy and
state boundary and requires separate authorization.

## Replacement-planning bundle

The compiler derives one closed slot manifest per coherence scope. Each manifest contains
only typed identities, role constraints, and the mention bindings needed to verify coverage.
The planner receives only an allowlisted bounded projection: original values for admitted
  slot-bound mentions, closed slot roles and relational constraints, and target or context
  fragments contained in the exact compiled context projection.
Coherence membership alone grants no context access. Workframes exclude source IDs,
unrelated fields, datums outside the compiled target/context projection, and reconstruction
state. Public receipts and diagnostics see only counts and allowlisted reason codes.

Candidate workframes correlate expected slots with invocation-private opaque tokens. A
returned `original` value, label, row position, or DataDesigner record ID is content or
backend evidence, not slot identity. Missing, duplicate, foreign, or contradictory slot
tokens make the scope inconsistent.

A candidate bundle is valid only when:

1. every expected slot appears exactly once;
2. no unknown, missing, or duplicate slot appears;
3. every synthetic value is non-empty and differs from every original bound to that slot;
4. no synthetic value equals any protected original in the coherence scope;
5. slots declared distinct do not share a synthetic value;
6. mentions declared to share a slot have exactly one assignment;
7. type, format, wildcard, and closed relational constraints pass; and
8. the bundle remains within declared count and byte limits.

The graph profile fails closed on an invalid candidate bundle. It does not use the current
row-local placeholder repair as graph semantics, regenerate candidates automatically, or
accept a partial bundle. Any future bounded regeneration policy would need explicit attempt
identity, limits, cancellation, stale-result handling, and separate review.

A scope with no admitted Substitute mentions completes as `planned` with an empty bundle and
does not call a provider or DataDesigner. A qualified no-work datum may legitimately retain
its unchanged text. That is distinct from substituting raw input for a failed or withheld
result.

## Phase 4 accounting integration

The compiler creates one scope-planning stage task for every admitted coherence scope. This
is the phase 7 review of the grouped-task cardinality deferred by phase 4: one planning task
may read bounded inputs from several datums but produces one private scope plan, not a datum
output. It remains a compiler-owned Anonymizer task rather than a DataDesigner scheduler
task. An empty manifest succeeds as verified no-work with zero dispatched attempts.

Every scope-planning task and dispatched attempt appears in the phase 4 one-shot terminal
ledger. It receives a task ID, attempt ID, and invocation-private correlation tokens before
dispatch. Candidate generation executes through `NddAdapter.run_workflow()`. Reconciliation
requires exactly one current terminal attempt record and the exact expected slot-token set;
missing, duplicate, foreign, stale, extra, contradictory, or unmapped failure evidence uses
the phase 4 `failed`, `cancelled`, `lost`, or `inconsistent` outcome rules.

The terminal scope outcome is a pure reduction of the phase 4 planning-task outcome plus
candidate validation. A successful verified task yields `planned`; a prerequisite that
prevents dispatch yields `blocked`; all other task outcomes map to the same named non-success
scope outcome. An empty manifest yields `planned` through a verified-no-work task with no
backend attempt.

Datum transformation tasks for Substitute are not ready until their scope-planning task is
terminal `planned`. A non-planned scope blocks those tasks and withholds every affected
datum and atomic group through the phase 4 fixed-point rules. No provider or backend dispatch
exists outside phase 4 conservation.

Scope-planning readiness depends only on phase 6 cluster, role-result, mention, and compiled
context inputs for every member datum. A non-terminal phase 6 input leaves the
planning task unresolved and causes no dispatch; a terminal non-success input closes the
planning task and scope as `blocked`. Execution dependencies among member
datums do not gate the scope planner; applying datum dependency readiness to this grouped
task could deadlock a scope that contains both a prerequisite and its dependent. Normal
datum dependencies gate transformation and release after planning.

## Ledger and linearization

One invocation owns one bounded ledger. The ledger is created only after pure compilation
succeeds and is closed with the invocation. It is not serialized, returned, logged, or used
as downstream retry identity.

Planning uses a scope-level all-or-none reservation:

```text
absent
  -> reserved(planning_attempt)
  -> planned(immutable_bundle)
     or aborted
     or poisoned
```

`reserved`, `aborted`, and `poisoned` are private ledger states. The terminal accounting
model exposes their closed result as `planned`, `blocked`, `failed`, `cancelled`, `lost`,
or `inconsistent`.

The planning linearization point is the atomic transition from a reservation owned by
the current planning attempt to the complete validated immutable bundle. Before that point,
no transformation task may read any candidate. After that point, all readers observe the
same bundle and no task may replace, merge, or partially update it.

The executor schedules at most one active planner for a coherence scope. Different scopes
may plan concurrently. Compare-and-set ownership still guards every transition so duplicate
dispatch, late results, and implementation defects fail closed rather than overwrite state.

An identical replay received before planning acceptance may be treated as an idempotent
notification only if no second effect or publication occurs. Conflicting evidence observed
before that linearization makes the scope inconsistent. After `planned` is accepted, the
terminal outcome is absorbing: later duplicate, different, stale, or foreign evidence is
rejected and cannot rewrite the scope. A result from another invocation is foreign even when
source data and declared scope are identical.

`planned` is not an externally committed assignment. The bundle remains provisional and
invocation-private until phase 4 releases a qualified atomic-group output that uses it.
Rollback in this design means abandoning or poisoning an unpublished reservation or
candidate before `planned` is accepted. A planned bundle remains immutable and private until
phase 4 either releases qualified output or withholds and discards it during finalization.
Rollback does not undo provider execution, persistence, delivery, or any other downstream
effect.

The closed transition table is:

| From | Observation and guard | To | Phase 4 effect |
| --- | --- | --- | --- |
| `absent` | Admitted non-empty manifest; owner CAS succeeds | `reserved` | Planning task becomes dispatchable |
| `absent` | Verified empty manifest | `planned(empty)` | Task succeeds as verified no-work; zero dispatches |
| `absent` or `reserved` | Prerequisite prevents planning | `blocked` | Task and scope close blocked |
| `reserved` | Exactly one current result reconciles and validates | `planned(bundle)` | Task succeeds with private plan |
| `absent` or `reserved` | Local validation or attributable backend failure | `failed` | Task and scope close failed |
| `absent` or `reserved` | Cancellation before dispatch, or trusted stop before accepted result | `cancelled` | Task and scope close cancelled |
| `reserved` | Dispatch occurred without trusted stop or terminal run evidence | `lost` | Task and scope close lost |
| `reserved` | Contradictory, foreign, non-identical duplicate, or ambiguous evidence before acceptance | `poisoned` then `inconsistent` | Task and scope close inconsistent |
| `planned` | Any later result or terminal signal | `planned` | Evidence is stale and rejected; no state rewrite |
| Any terminal scope outcome | Repeated transition request | Same terminal outcome | Inert or rejected; no redispatch or release |

Publication-critical finalization failure does not rewrite an absorbing scope outcome. It
closes the invocation as inconsistent and withholds every output under the phase 4 global
embargo. Post-acceptance host teardown failure likewise cannot rewrite scope or invocation
results already accepted for return.

## Failure, cancellation, and lost execution

Planning follows the phase 4 cancellation and lost-execution rules:

- cancellation before dispatch aborts the reservation and closes the scope as `cancelled`;
- cancellation after dispatch closes as `cancelled` only with trusted evidence that the
  planning execution stopped without producing an assignment;
- without trusted stop evidence, post-dispatch cancellation closes as `lost` and poisons
  the scope;
- a transport break or missing trusted run record closes as `lost`;
- a late result from an aborted, lost, or superseded attempt cannot assign the scope; and
- duplicate, foreign, contradictory, or partial results close as `inconsistent`.

Phase 7 performs no automatic planning retry. A downstream retry of delivery reuses the
exact already-protected payload when safe; it does not rerun Anonymizer. A new Anonymizer
invocation receives fresh correlation, scope, slot, and attempt identities and may produce
different assignments.

A non-planned scope withholds every atomic group that contains a mention bound to that
scope. Dependency and group withholding then follow the phase 4 fixed-point rules. A scope
planning failure does not affect a graph component that neither uses nor depends on that
scope.

## Transformation and fail-closed release

Transformation reads only planned bundles. Each datum-local replacement application must
prove that:

- every admitted Substitute mention resolves to one expected slot;
- every expected span is in range, non-overlapping, and still matches its anchored source
  value;
- every targeted mention is applied exactly once;
- no unplanned value-only fallback selects an assignment; and
- the resulting datum passes the declared phase 6 and Substitute release predicates.

Skipped, ambiguous, stale, overlapping, missing, or extra applications are not qualified
successes. They withhold the affected atomic group; raw input is never substituted for a
withheld output.

Coherence and atomic release have different boundaries. A valid planned bundle may serve
several atomic groups. A precisely accounted transformation failure in one group does not
invalidate the assignment or automatically fail another group in the same coherence scope.
The other group can release only if its own datums, dependencies, and predicates qualify and
the invocation has no global accounting embargo.

No replacement map becomes externally visible before exhaustive invocation reconciliation.
The public DataFrame compatibility path may continue to materialize its current trace
columns, but the private graph profile must not expose a partial scope bundle through a
result, exception, callback, log, receipt, or diagnostic view.

## Bounds, cleanup, and privacy

Compilation requires ceilings for scopes, clusters, slots, mentions, candidate bytes, and
total ledger bytes. The executor rejects unsupported or excessive declarations before
creating the ledger. It retains only the content needed for the active invocation and
releases ledger references on every terminal path.

Cleanup must cover successful closure, validation failure, planner failure, cancellation,
lost execution, inconsistent accounting, transformation failure, and publication failure.
Before phase 4 releases output, phase 7 must abandon or poison every unused reservation,
close the planning ledger to mutation, and verify that no provisional bundle can become
observable. Failure in this publication-critical finalization makes the invocation
inconsistent and withholds output.

Host-resource teardown after immutable result acceptance is a separate lifecycle step. Its
failure is reported through the owning host surface and cannot retroactively retract an
already accepted result. Tests must therefore inject pre-release finalization faults and
post-acceptance teardown faults as different events. Python reference release does not prove
physical zeroization, so the design makes no secure-erasure claim.

Replacement values, original values, prompts, candidate bundles, and content-derived hashes
must not enter logs, metrics, exceptions, public receipts, or unbounded traces. Diagnostic
surfaces may expose only allowlisted reason codes, opaque invocation-private test identities,
and bounded counts by non-sensitive category.

## Reference model and test oracle

Build a pure reference model independent of pandas, DataDesigner, and the production ledger.
Its input is:

- datums, mentions, clusters, slots, coherence scopes, dependencies, and atomic groups;
- closed slot and relational constraints;
- declared limits and capabilities; and
- a timestamp-free sequence of exogenous observations: dispatch accepted or rejected,
  keyed candidate rows, `FailedRecord` evidence, backend exception, cancellation request,
  trusted stop acknowledgement, transport or process loss, anchored transformation evidence,
  verification-task evidence, publication-critical finalization success or failure, and
  post-acceptance teardown success or failure.

The model derives admission, reservation eligibility and transitions, candidate validation,
task and scope reconciliation, dependency propagation, terminal outcomes, finalization
consequences, datum and group eligibility, and the only legal release set. Production
reconciliation or release decisions are never model inputs. Production outcomes and release
sets must equal the reference result for every generated case.

The central oracle is:

```text
planned(scope) iff
  its empty manifest completed as verified no-work with zero attempts
  or exactly one current planning attempt produced one complete valid bundle

qualified(datum) iff
  its scope is planned
  and every admitted mention resolved and applied exactly once
  and every required predicate passed

released(group) iff
  phase-4 exhaustive reconciliation completed
  and every member datum remains release-eligible
  and no dependency, cancellation, loss, or accounting embargo applies
```

The finite exhaustive conformance envelope contains:

- zero through four datums;
- zero through two coherence scopes;
- zero through three clusters, zero through four slots, and zero through six mentions;
- zero through three flat atomic groups and the phase 4 DAGs over the admitted datums;
- exactly one scope-planning task per admitted scope, with zero attempts for an empty
  manifest and zero or one attempt for a non-empty manifest; a non-empty manifest has
  exactly one attempt only after dispatch is accepted;
- at most one primary terminal observation and one late, duplicate, stale, foreign, or
  contradictory observation per planning attempt;
- at most one cancellation request, one trusted-stop or loss observation, one transformation
  observation per datum, one verification observation per group, one publication-critical
  finalization observation, and one post-acceptance teardown observation; and
- at most 16 exogenous observations in one canonical trace.

Canonicalization orders commuting events from independent scopes, tasks, and datums by
opaque declaration position while retaining every non-commuting race around dispatch,
planning acceptance, cancellation, finalization, and release. Before executor comparison,
freeze the model and generator versions, exact graph count, canonical schedule count, and a
SHA-256 digest of the manifest. Larger seeded state-machine tests supplement this envelope;
they do not replace it.

## Admission and state-machine tests

Admission tests cover empty scopes, gaps, duplicate membership, duplicate semantic scope
declarations, unknown datums, cross-scope clusters, unknown slots, duplicate slot
declarations, nesting, overlap, unsupported strategy, missing capabilities, and every
declared limit. Every rejection asserts zero
additional graph-owned ledger, workframe, graph telemetry, DataDesigner, transformation,
reconstruction, and publication effects. Provider or host construction that preceded
internal compilation on the current facade is outside this assertion.

Spies also assert that rejection creates no invocation, attempt, reservation, or runtime
correlation token; performs no credential or resource lookup beyond already-bound host
construction; and emits no content-bearing telemetry. Adjacent-tier multiply-invalid cases
must return the earlier deterministic code under declaration and capability permutations.

Separately test a non-empty scope whose datums contain no admitted Substitute mentions. It
must plan an empty bundle without provider or DataDesigner work and may release unchanged
text only after the normal qualified no-work accounting path.

Readiness tests place a prerequisite and its dependent in one coherence scope. A
non-terminal phase 6 member leaves the planning task unresolved with zero planner
dispatches. After all required phase 6 inputs succeed, the planner dispatches without
waiting for either datum's transformation; transformations then follow the normal
dependency order. A terminal non-success phase 6 member closes the planning task and scope
as `blocked`, with zero planner dispatches.

State-machine tests cover every allowed transition and reject:

- read before planning completes;
- plan publication without the owning reservation;
- partial-bundle plan publication;
- second active reservation for one scope;
- overwrite or merge after planning;
- planning completion after abort, poison, cancellation, or loss;
- stale or foreign attempt results;
- duplicate or contradictory terminal signals;
- transformation before planning; and
- any terminal invocation containing an active reservation.

Terminal states are absorbing. Replaying an accepted notification cannot duplicate an effect
or release. A late completion cannot resurrect a cancelled, lost, inconsistent, or closed
scope.

## Candidate and collision tests

Generate complete and malformed candidate bundles across cluster and slot shapes. Cover:

- missing, extra, duplicate, empty, and unchanged assignments;
- synthetic/original collisions at the same slot and across the complete scope;
- duplicate synthetics for slots declared distinct;
- intentional reuse through one shared slot;
- one cluster with several type-appropriate related slots;
- type, format, wildcard, geographic, temporal, and contact consistency constraints;
- duplicate original text belonging to different clusters;
- identical labels with different slot identities;
- the same original or synthetic value in isolated scopes without shared state;
- candidate order and backend row-order permutations; and
- maximum allowed bundle size and one-over-limit rejection.

Graph-profile tests assert that invalid bundles fail closed without placeholder repair or
automatic regeneration. Separate public compatibility tests preserve the existing row-local
collision behavior until a public change is separately reviewed.

Before freezing the reference-model manifest, freeze a versioned closed slot-role and
relational-constraint vocabulary with positive and negative conformance fixtures. This
includes every supported type, format, wildcard, geographic, temporal, and contact rule.
An unknown role or constraint rejects at admission; an independent implementation is not
expected to infer a verdict from prompt prose.

Planner reconciliation tests take the cross-product of current, missing, duplicate, stale,
and foreign attempt and slot tokens with success rows and `FailedRecord` evidence. One
failure attributed by trusted correlation evidence to exactly one planning task closes that
task failed; lack of slot-level attribution within that proven task invalidates its complete
scope. Unknown, duplicate, stale, success-plus-failure, or otherwise contradictory failure
evidence closes the invocation inconsistent. A planner `FailedRecord` that cannot be
attributed to exactly one expected task by an invocation-private task or attempt token also
closes the invocation inconsistent and triggers the phase 4 global release embargo, unless
trusted batch or call evidence independently proves that it belongs to exactly one planning
task. Content-derived record IDs never establish attribution.

## Concurrency, cancellation, and fault injection

Use a deterministic scheduler and barriers around reservation, dispatch, result receipt,
validation, plan publication, transformation, reconciliation, output release, and cleanup.
Explore the frozen finite envelope exhaustively and use seeded schedules for larger graphs.

Required races include:

- two planners attempting one scope;
- duplicate dispatch of one attempt;
- cancellation on both sides of dispatch and assignment;
- transport loss before and after the planner may have produced a candidate;
- late success after cancellation or loss;
- assignment concurrent with a foreign or contradictory result;
- one scope failing while independent scopes complete;
- transformation failure in one of several atomic groups sharing a scope;
- invocation cancellation after planning but before release;
- publication-critical finalization interrupted before release; and
- host teardown failure after immutable result acceptance.

The race oracle is:

| Accepted evidence order | Scope result | Dispatch count | Release effect |
| --- | --- | ---: | --- |
| Cancellation before dispatch | `cancelled` | 0 | No output |
| Dispatch, then cancellation without trusted stop | `lost` | 1 | Global phase 4 embargo |
| Dispatch, then trusted stop before candidate acceptance | `cancelled` | 1 | No output |
| Valid candidate accepted, then trusted stop or cancellation | `planned` remains absorbing | 1 | Invocation cancellation still withholds if accepted before output release |
| Trusted stop accepted, then candidate arrives | `cancelled`; candidate is stale | 1 | No output |
| Contradictory evidence before planning acceptance | `inconsistent` | 1 | Global phase 4 embargo |
| Byte-identical replay | Existing outcome | 1 | Inert; no duplicate effect |
| Output release accepted, then cancellation | Existing released result | 1 | No retroactive retraction |

No race causes an automatic redispatch. Every dispatched planning attempt corresponds to
exactly one Anonymizer dispatch even if the provider or DataDesigner performs opaque work
internally.

Inject malformed, missing, extra, duplicate, stale, and foreign workframe results. At least
one mandatory process-kill test must use a test-only crashable backend at the existing
execution seam, with no production process or session API. It asserts only `lost`, poisoned
or abandoned private state, complete cleanup handling, and an empty release set. This is
lifecycle evidence for phase 7, not the materially different semantic runtime required for
stable public promotion.

Run sequential and concurrent fresh invocations with identical declarations and content.
They must not reuse ledger, reservation, scope, slot, attempt, correlation-token, or bundle
object identity; an old result is foreign. Each scope with a non-empty manifest dispatches
a new planner, while an empty manifest creates a fresh verified-no-work task with zero
attempts. Equal generated literals are allowed but do not prove shared state. Ledger access
must fail after every invocation terminal path.

## Property, privacy, and compatibility tests

Required properties include:

- **Conservation:** every admitted scope has exactly one terminal outcome; every expected
  slot appears exactly once in its planned bundle or is covered by the scope's non-planned
  outcome.
- **Dispatch conservation:** every admitted scope has exactly one phase 4 planning-task
  record; an empty manifest has zero attempts, and every accepted non-empty planner dispatch
  has exactly one current attempt-terminal record or an explicit `lost` classification.
- **Completeness:** `planned` contains every expected slot and no other slot.
- **Stability:** every mention of one slot observes the same assignment.
- **Collision safety:** no distinct slots in one uniqueness namespace receive the same
  forbidden or canonicalized synthetic value.
- **Declared reuse:** every mention of a shared slot receives one value, while slots declared
  distinct never collapse through equal text or labels.
- **Application conservation:** every admitted mention is applied exactly once at its
  authoritative span; released output retains no admitted original at that span.
- **Non-cascading application:** a synthetic value is never treated as a second source span
  during the same transformation.
- **Isolation:** changing or failing one independent scope does not change another scope.
- **Permutation invariance:** declaration, scheduling, workframe, and candidate order do not
  change semantic results.
- **Identity invariance:** opaque identity renaming yields an isomorphic outcome.
- **Content non-identity:** equal original text does not merge clusters or slots.
- **Monotonic withholding:** replacing a valid event with blocking, failure, cancellation,
  loss, or inconsistency cannot increase the release set.
- **Attempt isolation:** an old or foreign attempt cannot assign the current scope.
- **No partial visibility:** a non-planned bundle and a withheld group expose no replacement
  values through any observable surface.
- **Boundedness:** retained states and diagnostics stay within declared ceilings.

Use separate high-entropy canaries for original values, synthetic values, prompts, source
identity, and known digests of each. Enforce a structural surface-and-lifecycle allowlist:

| Surface | Original values | Synthetic values |
| --- | --- | --- |
| Active private planner input, prompt, ledger, and workframe | Only the compiled projection | Only the active candidate or plan channel |
| Released private graph output | Not at admitted anchored spans | Allowed only as protected text |
| Grandfathered public compatibility trace | Allowed only in existing trace fields | Allowed only in existing replacement-map and protected-text fields |
| Withheld or non-planned result | Forbidden | Forbidden |
| Logs, metrics, exceptions, receipts, IDs, diagnostics, and cleanup errors | Forbidden, including known digests | Forbidden, including known digests |

Structural allowlist assertions are primary; substring and known-digest scans supplement
them. Serialized results are inspected field by field rather than rejected merely because a
qualified protected output contains its synthetic value.

Compatibility tests preserve duplicate and non-monotonic DataFrame indexes, duplicate text,
row reordering, existing public replacement-map trace behavior, result columns and
attributes, CLI behavior, and current Substitute configuration. The matrix explicitly
covers no-entity provider bypass, filtering of unrequested entries, same-row placeholder
collision repair, custom instructions in the generator prompt, preview limiting after the
entity-row partition, non-cascading offset application, the legacy value-only fallback,
`FailedRecord` passthrough with exact ID, step, order, and shape, and the separate
`evaluate()` judge path.

Exercise those contracts through `run()`, `preview()`, `evaluate()`, trace materialization,
display, validation, and CLI paths. In paired cases, the unchanged public facade retains
legacy filtering and repair while the private graph profile rejects the same invalid bundle
without partial output. Spies assert that every DataDesigner execution still passes through
`NddAdapter.run_workflow()` and that no source-format type enters Anonymizer core.

Mutation tests must catch at least premature bundle visibility, slot joins by original text,
row-local regeneration inside a coherence scope, accepted partial bundles, collision checks
limited to one datum, distinct-slot aliasing, shared-slot divergence, last-writer-wins
planning, stale-attempt adoption, cancellation resurrection, value-only fallback, cascading
replacement, missing or duplicate application, released output retaining an admitted
original at its anchored span, raw-input fallback, and incomplete cleanup.

## Ownership and promotion gates

| Decision | Required authority |
| --- | --- |
| Scope grammar, slot semantics, collision policy, outcome algebra, and release predicates | Project and Anonymizer semantic owner |
| Invocation lifetime, ceilings, concurrency, provider authority, and cleanup observability | Host authority |
| Source-to-scope mapping, reconstruction, retry identity, and destination postconditions | Source-adapter and adopter owners |
| Trust boundary, allowed linkability, leakage criteria, and residual risk | Customer or consuming-product owner |
| Any public session, graph, ledger, receipt, artifact, or endpoint | Public-API and Platform owners |
| Any durable or cross-worker assignment backend | Separately named state-service and governance owners |

No explanatory SDK or type-safety guidance assigns these product decisions. RFC acceptance
does not replace a named owner's later product, public-API, deployment, or promotion decision.

## Review gates

Phase 7 is ready for implementation only after reviewers accept:

1. the invocation-bounded stability claim and explicit non-durability;
2. the flat exact coherence-scope partition and cross-scope cluster rejection;
3. cluster-to-slot modeling and the closed relational policy;
4. all-or-none provisional scope bundles and planning linearization;
5. fail-closed collision handling with no automatic repair or retry in the graph profile;
6. cancellation, loss, late-result, and cleanup behavior;
7. interaction between coherence scopes, dependencies, and atomic groups;
8. the pure reference model, finite schedule exploration, mutation set, and privacy canaries;
9. unchanged public DataFrame behavior and the `NddAdapter.run_workflow()` boundary; and
10. the retained downstream ownership of durable state, retries, deduplication,
    reconstruction, persistence, and delivery.

The 2026-08-20 review accepted this architecture as the design candidate for a future branch
checkpoint; it did not authorize implementation. Phase 7 remains blocked until Phases 4–6 pass and the versioned semantic and
execution contract in the decision record is reviewed and frozen; that contract supplies
the concrete closed relational policy and execution limits required by gates 3, 6, and 8.

Any future operator checkpoint must not bypass these prerequisites. Passing Phase 7 tests later
would not authorize a public graph or session API, durable replacement state, production
Intake or OpenShell support, a `zero PII` claim, grouped rewrite, or stable public promotion.

## Evidence and deferred decisions

The current implementation and tests establish row-local map generation, filtering,
same-row synthetic/original collision repair, offset-based non-cascading application,
PII-free log summaries for tested paths, and public trace compatibility. They do not
establish coherence scopes, cluster-to-slot identity, graph-wide collision policy,
concurrent ledger semantics, rollback, or cross-call stability.

The branch decision selected invocation-bounded stability as the only Phase 7 profile and retained
the no-regeneration baseline. The versioned role and constraint vocabulary, distinct-slot
matrix, ceilings, and cleanup-observability contract are owned pre-implementation inputs as
recorded above; they are not unclassified open decisions.

Whether a future governed state backend belongs in Anonymizer, the execution host, or a separate
service remains deferred. No durable backend is selected by this design, and that later
ownership decision does not block the invocation-bounded phase 7 profile.
