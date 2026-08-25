<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Phase 5 design — target and bounded-context workframes

Status: reviewed phase-specific design and test strategy; branch implementation authorization
is pending. This document is subordinate to the complete development and research plan in
the [graph-native SDK RFC](graph-native-anonymizer-sdk-rfc.md) and the
[phase 4 terminal-accounting design](phase-4-hierarchical-terminal-accounting-design.md).
RFC acceptance is the project decision on the plan; a later operator checkpoint controls
when Phase 5 work begins on this branch. Neither decision authorizes a public graph or
session API, production Intake or OpenShell integration, or a privacy-boundary decision.

Review status: independent architecture and test-strategy council review completed on
2026-08-20 with zero unresolved Critical or Warning findings for the prior draft. The
2026-08-21 revision moves product authorization and field policy outside Anonymizer, defines
preflight and private work-ID terminology, and adds the content-free observation and cleanup
contracts while retaining typed context limits and backend compatibility. Focused
architecture and test-strategy re-review completed with zero unresolved Critical or Warning
findings. The design review is complete, but Phase 5 is not implemented or tested. Branch
implementation authorization remains pending and is sequenced after Phase 4 is implemented
and passes its evidence gates.

## Decision

Phase 5 should add one private compiled projection for every output-bearing target datum.
The projection keeps target text and explicitly declared context datums in separate,
bounded frames and binds both to one phase 4 logical target task with private,
invocation-local work IDs.

Phase 5 qualifies framing, minimization, lowering, reconciliation, and
cleanup. It does not allow context to create entity mentions, change detection decisions,
join entity clusters, grant replacement coherence, or become output. Phase 6 must separately
review and qualify any context-informed detection or resolution semantics.

Context scope, dependency, coherence scope, and atomic group remain independent. A declared
relationship, dependency, common group, source adjacency, equal text, or shared context does
not grant context-read authority or replacement sharing.

## Preconditions and preserved boundaries

Phase 5 assumes phase 4 has implemented and qualified:

- pure graph compilation before graph-invocation effects;
- an immutable compiled plan and one-shot ledger;
- exhaustive task, target datum, dependency, atomic-group, stage, and invocation outcomes;
- work-ID-keyed backend reconciliation, cancellation, lost-execution, and no-retry rules;
  and
- fixed-point dependency and atomic-group withholding.

The implementation must preserve these boundaries:

- `_DatumId` remains immutable and graph-scoped. Text, source identity, DataFrame index,
  row order, context position, and content-derived hashes are not datum identity.
- Current public constructors, configuration, `run()`, `preview()`, `evaluate()`,
  `validate_config()`, result columns and attributes, `trace_dataframe`, `failed_records`,
  CLI behavior, and errors remain compatible.
- DataFrames are temporary workframes. `NddAdapter.run_workflow()` remains the sole boundary
  for executing DataDesigner workflows.
- The graph remains authoritative before lowering and after hydration. A row, batch, or
  context fragment is not a semantic output unit.
- Anonymizer owns source-neutral context grammar, validation, projection, correlation,
  accounting, and release qualification. Source adapters retain codecs, source identity,
  field policy, projection proposals, reconstruction, persistence, retries, deduplication,
  cleanup, retention, and delivery. The integrating product authorizes source access and
  field use before its adapter constructs the graph. Provider access and credentials remain
  outside Anonymizer's graph semantics.

The existing public DataFrame path continues to compile an empty-context profile. Phase 5
must not add an empty context section to legacy prompts, change public model inputs, or alter
row-local behavior merely because the private framing machinery exists.

## Closed semantic model

Phase 5 extends the private graph grammar with two datum purposes:

```text
target datum       — output-bearing text processed by phase 4 tasks
context-only datum — read-only text that may inform a declared target task
```

A target datum may also be referenced as read-only context for another target. A datum has
one immutable purpose, but a target reference used as context does not change the referenced
datum's own task, dependency, atomic-group, or output semantics.

A target reused as context contributes the immutable source text captured in the admitted
graph and compiled projection. It never contributes transformed output or live task state.
Its own success, failure, cancellation, loss, group eligibility, or release cannot change an
already compiled context binding. A context reference creates no scheduling edge in either
direction. Read-only context-reference cycles are therefore permitted when every binding is
declared and bounded; they are not dependency cycles. Temporal, recursive, hierarchical,
or transitive context expansion remains unsupported.

Every target has exactly one `_ContextScope`. A scope contains one target and an ordered
tuple of zero or more context datum IDs. The order inside the tuple is explicit prompt or
presentation order and is therefore semantic. Context-scope declaration order is not
semantic. A compiler-issued opaque `_ContextScopeId` identifies the compiled scope; member
text, tuple position, source IDs, and graph declaration order do not.

The first profile permits:

- one scope for every target datum;
- an empty context tuple;
- one context datum reused by several targets;
- one target used as context for another target; and
- ordered context without inferring chronology, hierarchy, or dependency.

It rejects:

- a missing or duplicate scope for a target;
- an unknown target or context datum;
- a context-only datum used as a scope target or placed in an atomic group;
- an unreferenced context-only datum;
- target self-context;
- a duplicate context member in one scope;
- implicit context derived from links, dependencies, coherence, atomic membership, source
  order, or equal content; and
- any unsupported relation, nesting, wildcard expansion, or unbounded context rule.

Atomic groups remain a flat exact partition of target datums only. Dependencies continue to
connect target datums only. Context-only datums are immutable task inputs: they do not own
logical protection tasks, dependencies, atomic-group membership, output candidates, or
datum-release outcomes.

To preserve exhaustive accounting, the compiled plan expands each target task into a fixed
set of private context-binding records. Every expected binding closes as `available` or
`invalid` before dispatch. These records are child input evidence for the owning task, not
new graph datums or group-scoped tasks. An invalid binding deterministically fails the owning
undispatched task; it is never treated as absent context.

Compilation creates one immutable binding identity for every `(owning target-task identity,
context-scope ID, ordinal, context datum ID)` tuple before any runtime work ID exists.
Lowering must bind each identity exactly once to a fresh `context_binding_id`.
Reconciliation starts from the compiled binding set, not from work IDs or rows observed
after lowering, and proves the full bijection from compiled binding identity to lowered work
ID to consumed evidence.

Phase 5 uses three private correlation names: `target_work_id` identifies one lowered target
row, `context_binding_id` identifies one lowered use of context by one target, and the phase 4
`attempt_id` identifies the dispatch attempt. They are random, invocation-local work IDs,
not credentials, permissions, public graph identifiers, or caller-supplied trace IDs. Text,
source identifiers, DataFrame indexes, row order, and the work IDs themselves never replace
the compiled identities that remain authoritative.

A context-only datum is one graph value. A context binding is one consumer-specific,
ordinal-bearing compiled use of a datum. Reusing one datum creates distinct bindings; it
does not copy the datum or create a dependency among consumers.

`available` and `invalid` are private binding-evidence states, not phase 4 task outcomes. A
declaration, limit, or execution-contract defect is rejected during preparation. Failure to
construct a known compiled binding before dispatch yields local task `failed`. Evidence that makes
binding attribution unsafe yields phase 4 `inconsistent` and follows its localization or
global-embargo rules. Lack of a trusted terminal record after dispatch yields `lost`.

## Pure admission and context execution contract

Compilation remains the only graph admission boundary. It consumes the proposed graph,
phase 4 plan inputs, context limits, strategy profile, and declared backend capabilities. It
returns one immutable plan or one bounded content-free rejection before phase-5-owned
ledger records, work IDs, workframes, provider access, or backend dispatch. The compiler
remains pure. The public wrapper must expose a versioned content-free preflight observation
through the existing opt-in measurement surface. Observation cannot change the compiler
input, decision, rejection precedence, or prepared value.

Phase 5 extends the phase 4 rejection order:

1. Reject a malformed outer graph or unsupported private schema version.
2. Reject global datum, identifier, and byte-limit violations.
3. Reject malformed, duplicate, or purpose-invalid datum declarations.
4. Apply the accepted phase 4 dependency and atomic-partition checks.
5. Reject malformed context scopes, missing or duplicate target coverage, unknown
   references, orphan context-only datums, self-context, and duplicate members.
6. Reject per-scope context count or byte excess, total context-reference excess, and total
   lowered-frame expansion excess.
7. Reject recognized but unsupported context relation, ordering, nesting, wildcard, or
   cardinality semantics.
8. Reject a missing or incompatible context-workframe capability, strategy, or provider
   retention posture.

The same invalid graph must select the same safe code under declaration-order and capability
permutations. A structural error must not be presented as a backend-compatibility decision.

The first context execution contract is a private, immutable, versioned value bound to the
compiled plan. It declares:

- the permitted private profile and schema version;
- maximum context members and UTF-8 bytes per target;
- maximum total context references and expanded workframe bytes;
- whether target datums may be reused as context;
- the permitted context direction and ordering grammar; and
- the closed execution-boundary artifact classes and versioned closure attestations required
  before release; and
- the provider-retention posture, which must be `retention_disabled` for the first profile.

The contract is not an access credential and has no product identity, expiry,
renewal, or revocation semantics. The integrating product authorizes source access and field
use before graph construction. Public `prepare()` is the preflight boundary: it validates
the exact declared projection against the frozen contract without model or provider work.
Immediately before invocation open, `protect()` verifies that the selected backend still
provides the required profile, limits, and retention posture. A missing or incompatible
backend raises the typed pre-invocation rejection and never widens the plan, removes context,
or falls back to independent rows. Provider or credential binding already performed by the
legacy public facade remains outside this narrower no-additional-effects guarantee.

## Compiled projection and separate frames

The immutable compiled plan contains one projection manifest per target task:

```text
target task identity
  -> target datum identity
  -> context scope identity
  -> ordered context datum identities
  -> compiled byte and count ceilings
  -> permitted private consumer profile
```

The manifest contains graph identities but is never passed directly to DataDesigner. At
runtime, lowering creates fresh private work IDs and two logical projections:

| Projection | Required fields | Forbidden fields |
| --- | --- | --- |
| Target frame | `target_work_id`, phase 4 task and `attempt_id`, target text, reviewed target-local fields | source IDs, graph IDs, context text, reconstruction state, public index |
| Context frame | `context_binding_id`, owning `target_work_id`, context ordinal, context text | source IDs, graph IDs, target output columns, unrelated fields, reconstruction state |

Implementation may encode the context projection as a bounded structured internal column or
as a separate temporary DataFrame. If a backend needs one tabular call, a private adapter may
construct a third ephemeral request payload from the two validated logical frames. That
payload is not the target frame, must preserve both work-ID namespaces and every context
ordinal, and is discarded before hydration and release. Concatenating target and context
into one prose string, using a context row as an output row, or deriving correlation from
position is not conformant.

All internal column names must use `COL_*` constants. Shared prompt references must use
`_jinja()`. A workflow may consume context only after a later phase defines its exact prompt
and semantic role. The no-context path must reuse the existing prompt and workflow shape
byte-for-byte where current compatibility tests require it.

## Lowering, reconciliation, and hydration

A target task becomes dispatchable only after every expected context binding is available,
the projection is within its compiled limits, and all phase 4 readiness rules pass. Lowering
must not fetch source data, consult reconstruction state, infer missing fields, or expand the
scope after compilation.

The executor may batch ready target tasks. Each batch remains transport-only. Reconciliation
must prove:

1. compiled target-task identities, lowered `target_work_id` values, and terminal target
   records form exact bijections;
2. compiled binding identities, lowered `context_binding_id` values, and consumed input
   evidence form exact bijections;
3. each context ordinal, datum, scope, and owner equals its compiled binding identity; and
4. no foreign, stale, duplicated, missing, cross-target, or extra binding influenced the
   task.

A trusted response missing one target work ID may be localized under the accepted phase 4
rules. A context-binding defect is local only when the complete unaffected target and
context bijections remain provable. An unknown work ID, cross-target binding, plan mismatch,
or attribution contradiction that destroys the batch bijection closes the invocation as
`inconsistent` and triggers the phase 4 global embargo.

Hydration produces results only for target tasks. It never creates a mention anchored to
context, emits context text, or promotes context-only datums into target outcomes. Phase 5
returns a private context-framed graph capability for phase 6; it does not change protected
text by itself.

## Phase 4 integration and release

Phase 5 does not add an output-producing semantic stage. It extends the input projection and
reconciliation obligations of existing per-target phase 4 tasks. Synthetic ledger tests may
exercise context-binding children, but a backend batch or context row is never a task.

Known projection construction or binding failure before dispatch closes only the owning
task as `failed`. Cancellation before dispatch closes it as `cancelled` with no backend call.
After dispatch, phase 4 cancellation, trusted-stop, lost-execution, terminal precedence, and
no-retry rules apply unchanged.

A context-local fault affects every target task whose compiled projection contains that
binding. Phase 4 then propagates target ineligibility through explicit dependencies and
atomic groups. Reuse of the same context datum does not create an implicit dependency among
its consumers. A global attribution fault withholds all groups.

The phase 4 release barrier remains the only output publication point. Released atomic
groups contain complete target outputs only. Withheld groups, failed projections, context
frames, callbacks, errors, diagnostics, and traces expose no context text and never substitute
raw target input.

The binding transition and fault mapping is closed:

| Point | Observation | Binding evidence | Phase 4 effect |
| --- | --- | --- | --- |
| Preparation | malformed, over-limit, or backend-incompatible declaration | none | reject; no invocation |
| Before dispatch | exact compiled binding constructed | `available` | task may become ready |
| Before dispatch | known compiled binding cannot be constructed | `invalid` | owning task `failed`; no dispatch |
| At dispatch | compiled binding-to-work-ID bijection committed | `available` | one task attempt |
| Reconciliation | one exact consumed binding record | `available` | no additional terminal effect |
| Reconciliation | localizable missing or malformed binding evidence | `invalid` | owning task `inconsistent` |
| Reconciliation | foreign, cross-target, contradictory, or plan-mismatch evidence | `invalid` | invocation `inconsistent`; global embargo |
| After dispatch | no trusted task run record | unchanged | owning task or invocation `lost` under phase 4 |

Cancellation before binding construction wins with no frame or dispatch. A known construction
failure accepted first remains `failed`; a later cancellation request cannot rewrite it.
After dispatch, phase 4 terminal acceptance and cancellation precedence remains authoritative.

## Cancellation, cleanup, and privacy

Cancellation is an event, not proof that backend work stopped. Before dispatch it causes no
workframe or backend effect. After dispatch, a target task is `cancelled` only with trusted
stop evidence; otherwise it is `lost`. A late result cannot reopen a terminal task. Phase 5
adds no automatic retry.

Before release, an owned bounded lifecycle seam must report that every context binding is
terminal, every Anonymizer-owned context or joined workframe is closed, every work-ID map
rejects further access or mutation, and every backend-owned ephemeral artifact class named
by the frozen contract has one trusted closure attestation. Tests inspect that seam, the
bounded artifact paths owned by the invocation, and the versioned execution-boundary
attestations. Cleanup runs after task and datum terminal evidence is accepted but before
group release and public target-outcome materialization. It never rewrites an absorbing task
or datum outcome. A confirmed Anonymizer cleanup failure or trusted backend closure failure
closes the invocation as `failed` with reason `cleanup_failed`. Missing, foreign, or
contradictory closure evidence closes it as `inconsistent` with reason
`cleanup_unconfirmed`. Both apply a global release embargo; every public target receives the
corresponding non-success outcome and no group exposes output. This proves logical lifecycle
closure and required execution-boundary attestation only; it does not prove physical memory
erasure, provider deletion, or absence from an unowned provider trace. Host teardown after
immutable result acceptance is separate and cannot retroactively change the result.

Context text, target text, source identity, prompts, graph IDs, context-scope IDs, and
content-derived hashes are forbidden in Anonymizer-owned logs, metrics, exceptions, public
receipts, diagnostics, cleanup errors, and bounded traces. Private active workframes may
contain only the compiled projection. Public compatibility traces retain only
their existing contract and gain no context fields. Provider-side retention and tracing are
integration and provider-governance controls that cross a separately owned privacy boundary.
The first profile requires a backend-attested `retention_disabled` posture; this is not proof
of provider behavior. Provider tracing is not part of this capability and Phase 5 makes no
claim about arbitrary external traces. A later retention-enabled profile would require a
separately reviewed customer or consuming-product privacy contract outside Anonymizer.

## Observability and profiling

Phase 5 exposes versioned, content-free observations around preflight, workframe
construction, dispatch, backend execution, reconciliation, cleanup, and release. When the
existing opt-in measurement surface is active, the observation schema records monotonic
duration, bounded or bucketed target/context counts and sizes, selected semantic and
implementation profile versions, route, terminal outcome, reason code, reconciliation
status, cleanup status, and allowlisted numeric or bucketed provider-usage fields when
available. Instrumentation around
DataDesigner work remains outside and immediately around `NddAdapter.run_workflow()`; it
does not create another execution boundary.

Anonymizer may join a caller's distributed trace as a child operation, but a caller trace ID
is never datum identity, a work ID, or a metric label. Target/context text, prompts, entities,
replacements, source IDs, graph IDs, private work IDs, endpoints, credentials, and unbounded
or content-derived dimensions are forbidden in observations. Instrumentation is
non-authoritative: it cannot change graph semantics, supply missing terminal evidence, or
turn measurement failure into protection success.

## Phase 6 handoff

Phase 6 may consume the private framed projection only after it defines a closed use for
context. Any detected mention remains anchored to target datum offsets. Context may inform a
reviewed target decision, but it cannot supply a replacement span, become a target by
inference, or establish alias, dependency, coherence, or release semantics merely by being
visible.

The handoff supplies:

- immutable target datum identity and text;
- ordered, declared, bounded context frames;
- exact target/context binding evidence;
- the phase 4 task and attempt identities that account for execution; and
- content-free participation counts and reason codes.

It does not supply entity mentions, clusters, replacement roles, replacement slots,
synthetic values, source reconstruction state, or durable provenance.

## Reference model and test oracle

Build a pure reference model before the production framing runtime. It is independent of
pandas, DataDesigner, the production compiler, and the phase 4 ledger. Its inputs are:

- target and context-only datums, purposes, context scopes, dependencies, and atomic groups;
- every concrete count/byte ceiling, profile, immutable context execution contract, the
  capability snapshot accepted at preflight, and the capability snapshot observed
  immediately before invocation open;
- the fixed phase 4 readiness and terminal rules; and
- a timestamp-free sequence of binding construction, binding-work-ID commitment, consumed
  binding evidence, dispatch, keyed task terminal, cancellation, trusted stop, transport
  loss, cleanup, publication, and post-acceptance teardown observations.

The model derives admission, projection manifests, binding records, readiness, correlation,
terminal target outcomes, fixed-point withholding, cleanup evidence, invocation outcome,
public target outcomes, and the only legal release set. Production framing, localization,
or release decisions are not model inputs.

Cleanup evidence is closed: `verified` means every Anonymizer-owned frame, artifact, and
work-ID map reports closed and every required backend artifact has one trusted compatible
closure attestation; `failed` is one definitive Anonymizer cleanup failure or trusted backend
closure failure; and `unconfirmed` is missing, duplicated, foreign, incompatible, or
contradictory closure evidence. Cleanup does not rewrite terminal task or datum evidence. It
derives invocation-level
`failed(cleanup_failed)` or `inconsistent(cleanup_unconfirmed)` and the corresponding public
target outcomes before publication.

The central oracle is:

```text
projection valid(target) iff
  exactly one admitted scope names the target
  and the exact frozen projection satisfies the context execution contract
  and the preflight capability snapshot satisfies the required contract
  and every ordered context binding is declared, unique, known, and within limits

invocation opens iff
  the immediately-before-open capability snapshot still satisfies the complete frozen
  profile, limit, schema, and retention requirements

task dispatchable(target) iff
  projection valid(target)
  and every context binding is available
  and the phase 4 task-readiness predicate passes

group released iff
  phase 4 accounting is exhaustive and globally consistent
  and every member target remains release eligible
  and cleanup evidence is verified
  and no cancellation or publication embargo applies
```

The finite exhaustive envelope contains:

- one through four target datums and zero through three context-only datums;
- zero through three context members per target, including target-as-context cases;
- the phase 4 DAGs and flat atomic partitions over target datums;
- exact, over-limit, and backend-incompatible scope projections;
- ordered capability pairs whose preflight snapshot is `compatible` and whose
  immediately-before-open snapshot is `compatible`, `missing`, `incompatible`, `weakened`,
  or `retention_enabled`; only a runtime snapshot that still satisfies the complete frozen
  contract may open an invocation;
- a finite ceiling domain `{0, exact, exact + 1}` for datum bytes, ID bytes, context members,
  context bytes, total references, and expanded-frame bytes, using symbolic payload classes
  `{empty, one-byte, multibyte, exact-limit, one-over-limit}`;
- one dispatch per ready logical target task;
- one primary binding construction/consumption observation and at most one missing,
  duplicate, wrong-ordinal, foreign, cross-target, or contradictory observation per binding;
- one primary terminal observation and at most one missing, duplicate, foreign, stale,
  cross-target, plan-mismatch, or contradictory observation per task;
- cleanup evidence classes `verified`, definitive `failed`, missing, duplicate, foreign,
  incompatible, and contradictory, with at most one primary cleanup observation and one
  competing observation per invocation;
- at most one cancellation, trusted-stop, transport-loss, publication, and post-acceptance
  teardown event per invocation; and
- context-reference topology classes with no cycle, rejected self-context, every two-target
  cycle, every three-target cycle within the datum bound, and each permitted cycle both
  disjoint from and overlapping a phase 4 dependency edge; and
- the deterministic trace bound `E_max = 4B + 3T + 7`, where `B` is the admitted binding
  count and `T` is the admitted target-task count. The four binding slots cover construction,
  work-ID commitment, consumption, and one corruption; the three task slots cover dispatch,
  terminal acceptance, and one corruption; the seven invocation slots cover cancellation,
  trusted stop, transport loss, one primary cleanup observation, one competing cleanup
  observation, publication, and post-acceptance teardown.

Schedules that differ only by commuting events for independent target tasks share one
canonical representative. Context order within one scope never commutes because it is
semantic. Freeze the model and generator versions, exact graph count, exact canonical trace
count by target/context cardinality, context-order class, and context-cycle class, the full
finite ceiling domain, `T`, `B`, `E_max`, the actual event count, and a SHA-256 manifest digest
before comparing the implementation. The generator rejects any trace that exceeds its
computed bound; it never truncates a legal trace at a fixed number.

## Admission and framing tests

Admission tests cover every rejection tier, declaration permutations, adjacent-tier
multiply-invalid inputs, exact-limit and one-over-limit cases, empty context, missing and
duplicate target scopes, orphan context-only datums, unknown and duplicate members,
self-context, unsupported target-as-context use, invalid purpose, unsupported relation,
missing or incompatible backend capability, and unsupported strategy profile.

Effect spies start before compilation and require zero phase-5-owned invocation, ledger,
work ID, workframe, provider, credential, backend, NDD, reconstruction, and publication
effects on rejection. When opt-in measurement is active, the wrapper records exactly one
versioned preflight start/terminal pair with duration and the closed rejection code; it may
not contain graph content or create invocation identity. Pre-existing public-facade
construction and request-start telemetry remain outside this guarantee.

Context-contract tests distinguish structural, limit, profile, and backend-compatibility
failure and prove the documented rejection precedence. They enumerate ordered capability
snapshots at preflight and immediately before invocation open. A runtime snapshot that still
satisfies every frozen profile, limit, schema, and retention requirement opens normally;
missing, unknown, incompatible, weakened, or retention-enabled runtime posture raises the
typed pre-invocation rejection with zero invocation, workframe, provider-call, or NDD
effects. Mutating the graph, limits, or prepared value cannot widen the plan from live
backend state.

Framing tests prove:

- target and context schemas remain separate under every transport encoding;
- declared context order is preserved exactly;
- equal or repeated text remains distinct by opaque identity;
- target-as-context creates a read-only binding, not a second output;
- reused context creates independent bindings for each consumer;
- no source ID, graph ID, public index, reconstruction state, or unrelated field enters a
  workframe; and
- empty-context lowering preserves the existing private/public workflow shape.

The last assertion uses frozen prompt text, column configuration, provider-call count, and
workflow-schema snapshots for the current no-context `run()` and `preview()` profiles. A
structural digest changes only through an explicitly reviewed compatibility update.

## Reconciliation, fault, and schedule tests

Independently permute target rows, context rows, batches, and terminal records. Cover missing,
duplicate, foreign, stale, cross-target, extra, and swapped work IDs; correct work IDs with
wrong ordinals; a valid work ID bound to the wrong task; one localizable missing binding; and a
global attribution contradiction. Duplicate and null-like public DataFrame indices, equal
target/context text, and identical source-shaped fixture fields must not affect attribution.

Paired oracle fixtures use the same admitted graph and differ by one observation. In the
local fixture, one binding defect is attributable to one owning task and changes only that
target's explicit dependency and atomic-group closure. In the global fixture, a contradictory
binding destroys ownership attribution and embargos every group. The production executor
must match both release sets and reason classes exactly.

Metamorphic cases reset, duplicate, reorder, and replace DataFrame indexes; inject colliding
source-shaped IDs and row labels; and permute source-field order. Work-ID-keyed outcomes remain
isomorphic, no source field enters either logical frame, and no mutated value can satisfy a
compiled binding.

Cancellation tests retain the phase 4 linearization point: pre-dispatch cancellation causes
zero frames and dispatches; trusted stop accepted before a result yields `cancelled`; a
post-dispatch request without stop evidence yields `lost`; accepted terminal evidence wins
over later cancellation; late evidence cannot resurrect output.

Inject projection construction failure, backend failure, trusted partial results, missing
run records, worker death through a test-only backend, cleanup-before-release failure, result
construction failure, and post-acceptance teardown failure. The process-kill test supplies
only phase-5 failure-boundary evidence and does not create a production process API or satisfy
the second-runtime gate.

Cleanup linearization fixtures first accept all task and datum terminal evidence, then vary
the cleanup evidence before publication. They cover owned frame and map closure plus every
declared backend artifact's successful, failed, missing, duplicate, foreign, incompatible,
and contradictory attestation. `verified` preserves private terminal evidence and permits
normal release qualification. Definitive owned or trusted backend failure produces
invocation `failed(cleanup_failed)` and public `Failed` outcomes without rewriting task or
datum states. Missing, duplicate, foreign, incompatible, or contradictory cleanup evidence
produces invocation `inconsistent(cleanup_unconfirmed)` and public `Inconsistent` outcomes.
Every non-verified case applies a global embargo and exposes no output.

Compiled-plan tests mutate every reachable source declaration, context tuple, adapter
projection, limit object, and available backend capability after preparation. The lowered
frames and binding identities remain those of the immutable compiled snapshot. Nested tampering through
a test seam fails before workframe construction. NDD request inspection proves that the
compiled manifest itself and graph/source IDs never enter the backend payload.

Lifecycle tests inspect the bounded private-state seam and ephemeral artifact directory on
success, rejection, preparation failure, cancellation, lost execution, inconsistency,
publication failure, and post-acceptance teardown. Pre-release paths have no live frames or
mutable work-ID maps at publication; teardown faults are reported separately. Confirmed
cleanup failure and unconfirmed cleanup receive distinct closed reasons. These tests make no
secure-erasure claim.

Observation-contract tests require one versioned start/terminal pair for each entered
lifecycle boundary and verify monotonic durations, closed routes and outcomes, bounded
dimensions, and stable profile versions. Batch, row, datum-ID, work-ID, and caller trace-ID
permutations must not change semantic outcomes or create high-cardinality metric labels.
Fault injection covers unavailable, throwing, delayed, duplicate, and reentrant measurement
sinks. No observer behavior may fabricate terminal evidence, alter event or cleanup
linearization, leak content, change the release set, or create unbounded labels.

## Required properties and mutation tests

Required properties are:

- **Target conservation:** every admitted target has exactly one terminal phase 4 outcome.
- **Binding conservation:** every compiled context reference has exactly one terminal private
  binding record for its owning task.
- **Projection exactness:** workframe content equals the compiled manifest and contains no
  other datum or field.
- **Frame separation:** target and context cannot exchange roles under row or batch
  permutation.
- **Identity invariance:** opaque datum and work-ID renaming yields an isomorphic result.
- **Content non-identity:** equal text never merges datums or satisfies a missing binding.
- **Declaration invariance:** scope-declaration and unrelated datum order do not change
  semantics; explicit context order is preserved.
- **Contract monotonicity:** reducing limits or backend capability cannot enlarge the
  dispatch or release set.
- **Snapshot isolation:** referenced target success, failure, cancellation, transformation,
  withholding, and context-reference cycles cannot change another task's compiled original
  context snapshot or readiness absent an explicit phase 4 dependency.
- **Isolation:** a localizable context fault changes only its consumers and their explicit
  dependency/group closure; global attribution faults are excluded.
- **Batch invariance:** ready-frontier batch size does not change outcomes.
- **No context visibility:** context never appears in output, withheld results, or public
  surfaces.
- **Boundedness:** retained frames, work-ID maps, observations, and diagnostics stay within
  declared limits.

Mutation tests must catch target/context concatenation, positional joins, missing binding
acceptance, fallback to no context, implicit source-order context, text-based deduplication,
target-as-context output, cross-target work-ID adoption, ignored ordinals, limit checks after
lowering, premature output, context leakage, incomplete cleanup, and raw-target fallback.
They also catch live-plan rereads, transformed-output context, target-status-derived context
dependencies, work-ID issuance from observed rows, and acceptance of a weakened or
incompatible backend capability.

## Compatibility and privacy tests

Compatibility tests exercise `run()`, `preview()`, `evaluate()`, display, validation, and CLI
paths with unique, duplicate, non-monotonic, string, and null-like indices; duplicate text;
row filtering, concatenation, and reset; mixed success/failure; `trace_dataframe`; and exact
`FailedRecord` shape and ordering. Public calls continue to use empty context and must not
gain prompt sections, columns, diagnostics, or behavior changes.

Boundary spies prove every DataDesigner execution still passes through
`NddAdapter.run_workflow()`. Source-shaped ATIF, OTLP, and chat fixtures may exercise
projection pressure only after their owners approve field roles; source types and
reconstruction state never enter Anonymizer core.

Privacy tests use separate high-entropy canaries for target text, every context datum,
source identity, graph identity, private work IDs, prompts, and known digests. The owned-sink
inventory covers target, context, and joined request workframes; work-ID maps; the bounded
ephemeral artifact directory; `repr` and `str`; exceptions and their cause/context; tracebacks;
Anonymizer-owned logs and metrics; public results and compatibility traces; receipts;
diagnostics; serialization; and cleanup errors. A structural allowlist inspects every owned
sink. The expected visibility matrix is:

| Surface and lifecycle | Permitted canary visibility |
| --- | --- |
| Active target frame | Target canary only in the exact compiled target-text field |
| Active context frame | Context canary only in the exact compiled context-text field |
| Active joined request payload | Only the exact compiled target/context projection in its reviewed fields |
| Work-ID maps and accounting state | Private work-ID canaries only; no target or context content |
| Any owned surface after verified cleanup | No target, context, prompt, entity, replacement, source-ID, graph-ID, or work-ID canary |
| Observations, logs, exceptions, results, receipts, diagnostics, traces, and cleanup errors | No protected canary or known digest at any lifecycle point |

Substring and known-digest scans supplement these field- and lifecycle-specific assertions;
they do not reject content in the exact active private fields that are required to execute the
compiled projection. Capability tests verify the backend's attested `retention_disabled`
posture; they do not claim to prove provider behavior. An absent, unknown, incompatible, or
retention-enabled posture rejects before invocation open. A future retention-enabled profile
remains unavailable until a separate customer-owned, versioned privacy-boundary contract is
reviewed outside Anonymizer.

## Ownership and promotion gates

| Decision | Required authority |
| --- | --- |
| Context grammar, framing, correlation, and failure semantics | Project and Anonymizer semantic owner |
| Context count/byte ceilings and backend capability contract | Anonymizer semantic and execution owners |
| Observation schema, privacy allowlist, and measurement non-interference | Anonymizer semantic and measurement owners |
| Backend artifact classes and trusted closure-attestation schema | Anonymizer semantic and execution owners |
| Source field roles and source-to-context projection | Source-adapter and adopter owners |
| Source access, field authorization, and earliest boundary where unprotected target/context may cross | Customer or consuming-product owner |
| Any public graph, context, receipt, artifact, or endpoint | Public-API and Platform owners |

Phase 5 is ready for an operator-authorized branch implementation checkpoint only after:

1. Phase 4 is implemented and its evidence qualifies;
2. reviewers accept the closed purpose and context-scope grammar;
3. reviewers accept semantic context ordering and target-as-context behavior;
4. reviewers accept the bounded context contract and backend compatibility checks;
5. reviewers accept separate framing, binding reconciliation, and failure localization;
6. reviewers accept the reference model, finite envelope, mutations, privacy canaries, and
   observation-contract tests; and
7. reviewers accept the unchanged public, DataFrame, NDD, measurement, and downstream
   ownership boundaries.

Passing Phase 5 later does not authorize context-informed entity decisions, the Phase 6 or 7
branch checkpoints, a public graph/session API, source mappings, production Intake or OpenShell
support, durable state, a privacy boundary, a `zero PII` claim, or stable promotion.

## Evidence and unresolved gates

Current phases 1–3 prove only empty-context singleton lowering. They do not establish
multi-datum framing, context correlation, cleanup, or context-aware model
semantics. A Phase 4 branch implementation candidate and evidence suite exist, but its
completion review and repository checks remain pending at this
checkpoint.

Intake hierarchy and structured fields motivate bounded context but do not approve which
fields may be exposed, the earliest protection boundary, or a source-to-context mapping.
Those remain adopter and customer decisions. Phase 5 must fail admission without the
required context execution contract and compatible backend, and it must not encode Intake
policy in Anonymizer core.

The authoritative inputs are the parent
[technical proposal](graph-native-anonymizer-sdk-technical-proposal.md), the authorized
[phase 4 design](phase-4-hierarchical-terminal-accounting-design.md), the separate
[Intake evidence](intake-workload-validation-evidence.md), and the branch-local graph,
runtime, adapter, and private release code at
`702f43a988cf3673d16f40be5c59bc784737e1a3`.
