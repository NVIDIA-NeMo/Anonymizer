<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Phase 4 design — Hierarchical terminal accounting

Status: branch implementation checkpoint complete. The private branch-local implementation
and its evidence gates passed on 2026-08-25. This document is subordinate to the complete
development and research plan in the
[graph-native SDK RFC](graph-native-anonymizer-sdk-rfc.md). The checkpoint permits Phase 4
work on this branch, but it is not a separate project acceptance decision and does not
authorize a public graph API, production Intake or OpenShell integration, or stable
promotion.

Review status: independent architecture and test-strategy council review completed on
2026-08-20 with zero unresolved Critical or Warning findings. On 2026-08-20, the operator
authorized the Phase 4 branch checkpoint after review of its outcome algebra, policies,
boundaries, and test strategy. This is development authorization, not an approving GitHub
review or RFC acceptance. A separate implementation-remediation council completed on
2026-08-25 with zero unresolved material findings.

## Decision

**[Branch-local implementation]** Phase 4 adds a private, source-neutral execution ledger that accounts for
every admitted invocation, stage, logical task, datum, dependency, and atomic group. The
ledger permits output only after exhaustive reconciliation and atomic-group qualification.
It does not turn a DataFrame row, a backend response, or a source-format item into the
semantic identity of a datum.

**[Branch-local implementation]** The first supported related-record profile uses explicit datum dependencies
that form a directed acyclic graph and atomic groups that form a flat, exact partition of
the target datums. Cycles, nesting, overlap, incomplete group coverage, implicit
dependencies, and unsupported task cardinalities are rejected before effects. Phases 5–10
remain sequenced after the implemented Phase 4 passes the evidence gates below and their
branch checkpoints are authorized.

## Scope and preserved boundaries

The current branch-local phases 1–3 provide immutable private graph values, a compiler for
independent datums, temporary DataFrame lowering, invocation-private row correlation, and a
fail-closed Redact release profile. Phase 4 extends that seam; it does not replace the
published facade or claim that the graph architecture is current public behavior.

The implementation must preserve these boundaries:

- `_DatumId` remains immutable and graph-scoped through every graph phase. Source identity
  stays in the adapter's private reconstruction map; DataFrame index, row order, text, and
  caller identifiers are not datum identity.
- The public constructors, config types, defaults, `run()`, `preview()`, `evaluate()`,
  `validate_config()`, CLI behavior, result columns and attributes, `trace_dataframe`, and
  `failed_records` shape remain compatible.
- A stage may use a temporary DataFrame workframe and declare DataDesigner columns, but
  `NddAdapter.run_workflow()` remains the sole boundary for executing DataDesigner
  workflows.
- Anonymizer owns source-neutral validation, scheduling, terminal accounting, protection
  verification, and release qualification. Downstream owners retain codecs, closed field
  policy, source identity, projection, reconstruction, persistence, retries,
  deduplication, cleanup, retention, and delivery.
- New phase-4 runtime tokens and diagnostics remain opaque, invocation-local, bounded, and
  content-free. They must not expose source content, detected values, prompts, source IDs,
  or content-derived hashes. The existing public `FailedRecord` representation is a
  grandfathered compatibility surface, not phase-4 identity or a proposed diagnostic
  design.

Phase 4 does not add context or coherence semantics, grouped rewrite, durable state,
delivery, or an adapter-accessible provenance artifact. Those capabilities remain later
phases or unresolved gates.

## Pure compilation before effects

Compilation is the graph admission boundary. It consumes a proposed graph, limits, strategy,
and declared runtime capabilities and either returns one immutable compiled plan or one
bounded, content-free rejection. The compiler is pure: it does not use providers,
credentials, runtime resources, telemetry, workframes, or an execution context. It must
finish before phase-4 invocation identity, ledger or row-token creation, content-bearing
workframe lowering, backend dispatch, or any `NddAdapter.run_workflow()` call.

The existing public facade is already an open host: construction initializes logging,
selects providers, and creates the DataDesigner adapter, and public call telemetry may begin
before private graph compilation. Phase 4 does not move or repeat those compatibility
effects. Its guarantee is that an internally generated compatibility graph is compiled
before any graph-invocation or protection effect and that rejection causes no additional
provider use, credential access, resource acquisition, content-bearing telemetry,
workframe, reconstruction, or publication. A future raw-graph/session entry would need a
separate reviewed pre-binding API; this phase does not introduce one.

```text
proposed graph + limits + declared capabilities
  -> pure validation and compilation
  -> rejected                         # no invocation and no effects
     or
  -> immutable compiled plan
  -> invocation-local execution
  -> temporary ready-frontier workframes
  -> existing pandas workflows
  -> NddAdapter.run_workflow()
  -> reconcile, verify, close ledger
  -> atomic-group release decision
```

The graph runtime accepts only a compiled plan. It must not discover unsupported graph
semantics after it opens an execution context. After public input loading and preview
selection, the compatibility facade compiles its internally generated independent-row
workload to a no-dependency graph with singleton atomic groups.

Validation uses deterministic precedence so malformed structure does not appear as a
capability decision:

1. Reject a malformed outer value or unsupported private schema version.
2. Reject count and byte-limit violations.
3. Reject malformed or duplicate datum, dependency, task, stage, or group declarations.
4. Reject unknown references, wrong-purpose references, empty groups, duplicate members,
   and atomic coverage gaps.
5. Reject self-dependencies, duplicate dependency edges, partial group overlap, and cycles.
6. Reject recognized but unsupported atomic nesting, relation, context, coherence, or task
   cardinality.
7. Reject an unsupported strategy or runtime capability.

The implementation may use more specific safe codes within these classes. The same invalid
input must select the same code regardless of declaration order or runtime configuration.

## Dependency and atomic-group policy

Dependencies are first-class values with `prerequisite` and `dependent` endpoints. They
express execution and release prerequisites between target datums. They are never inferred
from datum declaration order, a source parent-child relation, context, coherence, or atomic
membership. `_DatumLink.RELATED` must not be reinterpreted as a dependency.

Every endpoint resolves to exactly one admitted target datum. Self-edges, duplicate edges,
dangling endpoints, and every directed cycle are invalid. Declaration order is only a
deterministic tie-break among ready tasks and a presentation-order input to the compatibility
adapter; it has no dependency meaning.

Atomic groups form a flat exact partition:

- Every target datum belongs to exactly one explicitly declared, non-empty group.
- Compilation assigns each group an opaque graph-scoped identity; tuple position and member
  contents are not group identity.
- Members are unique target datum IDs. Equal duplicate groups and implicit singleton groups
  are invalid.
- Partial overlap, such as `{a, b}` and `{b, c}`, is invalid.
- Strict nesting, such as `{a}` inside `{a, b}`, is recognized but unsupported in phase 4.
- Member order and group declaration order have no semantic effect.

Flat partitioning can express the current design choices between independently releasable
datums and a whole-set fail-together boundary. Nested source commit units remain a downstream
concern unless a later Anonymizer capability defines their protection semantics. Intake's
mixed-validity OTLP behavior is evidence that an adapter owner must choose the mapping; it
is not an Anonymizer default.

## Execution hierarchy and identity

One invocation owns an immutable compiled plan and a one-shot ledger. A stage is a
compiler-owned Anonymizer accounting boundary with a fixed per-datum predecessor rule and
typed result; it is not a global dispatch barrier, DataDesigner task, column, scheduler
step, or `FailedRecord.step`. A later-stage task may become ready after its own fixed
predecessor succeeds while an unrelated earlier-stage task remains open. Requiring global
stage closure would deadlock a multi-stage plan whose dependent datum waits for its
prerequisite datum to become locally qualified across every stage. The production
phase-4 Redact profile has one effectful `protect` stage that wraps the existing complete
pandas detection-and-replacement path. Its release checks are pure reducers after that
stage. Synthetic ledger plans may use up to three semantic stages to prove hierarchy. Later
phases may add reviewed semantic stages, but changing stage inventory is a compiler decision,
not an inference from backend traces.

Each stage expands to one logical task for every datum that requires it. A logical task is
identified by an opaque `(invocation, stage, datum)` identity; later phases must separately
review group-scoped tasks before admitting them.

The executor derives a task DAG from fixed stage ordering and explicit datum dependencies.
It may batch any ready frontier into a DataFrame call, but a batch and its rows are transport
details, not accounting units. Each lowered row carries a fresh opaque task token. Hydration
must prove a bijection between the expected ready tasks and the returned terminal records;
it never joins by position, index, text, or caller identity.

The hierarchy has these obligations:

```text
invocation
  stage
    logical task: exactly one target datum and one terminal task outcome
  datum: exactly one terminal datum outcome after all required tasks close
  dependency: unresolved until its prerequisite closes, then satisfied or unsatisfied
  atomic group: exactly one released or withheld decision
```

A stage result may close only after every logical task declared for it has a terminal outcome. A
datum may close only after every required task has a terminal outcome or has been
deterministically blocked. An invocation may return a result only after every admitted unit
is terminal and every dispatched backend attempt has either an accepted task-terminal
classification grounded in trusted run evidence—including localized
`inconsistent(missing)`—or a `lost` classification.

## Closed outcome algebra

Admission has two outcomes: `compiled`, which creates no execution evidence yet, and
`rejected`, which creates no invocation and performs no effect. Runtime states such as
`pending`, `ready`, `running`, and `cancellation_requested` are not terminal outcomes and
must never appear in a returned terminal result.

Logical task outcomes are closed and mutually exclusive:

| Outcome | Meaning | Output candidate |
| --- | --- | --- |
| `succeeded` | Exactly one trusted result passed task-level schema and provenance checks | May contribute |
| `failed` | Execution returned a known failure or task-level verification failed | None |
| `cancelled` | The task did not run, or the execution boundary proved that it stopped without a usable result | None |
| `lost` | Dispatch may have occurred but no trusted terminal record proves success, failure, or cancellation | None |
| `blocked` | A prerequisite closed without local qualification before this task was dispatched | None |
| `inconsistent` | Returned accounting was missing, duplicated, foreign, stale, contradictory, or otherwise unverifiable | None |

A datum is `locally_qualified` only when every required logical task succeeded exactly once
and the datum-level release predicate passed. A datum-level predicate rejection is
`failed(release_predicate_failed)`. Other non-successes produce the corresponding terminal
`failed`, `cancelled`, `lost`, `blocked`, or `inconsistent` outcome. A locally qualified
datum holds an internal output candidate, but local qualification alone does not expose
output.

A dependency is nonterminal `unresolved` until its prerequisite datum closes. It then has
the execution outcome `satisfied` when that datum is `locally_qualified` before atomic-group
propagation, or `unsatisfied` with only the prerequisite's bounded cause classes otherwise.
Dependencies never contain protected content. An unsatisfied dependency blocks an
undispatched dependent task. Release propagation separately follows prerequisite datum
eligibility, so a later atomic-peer failure can withhold an already-run dependent without
rewriting the dependency's execution outcome.

An atomic group is either `released` or `withheld`. Propagation derives a separate
`release_eligible` bit without rewriting the datum's terminal outcome. A group is released
only when every member datum is `locally_qualified` and remains `release_eligible`, and the
group release predicate passes. `released` is the only terminal outcome that contains
outputs, and it contains the complete group in graph datum presentation order. `withheld`
contains no output; raw input is never substituted for a withheld result.

Stages use the same non-success evidence classes as tasks and otherwise close as
`succeeded`. Their outcome summarizes child evidence and does not alter it. An invocation
closes as:

- `completed` when exhaustive, internally consistent accounting produced a trusted group
  result, even if some groups were withheld;
- `failed` when a known invocation-scoped failure prevented a normal group result;
- `cancelled` when invocation cancellation closed all unfinished work but no normal group
  result is returned;
- `lost` when the execution boundary cannot establish a trusted invocation run record; or
- `inconsistent` when global reconciliation cannot prove which declared units the evidence
  describes.

Invocation-level `failed`, `cancelled`, `lost`, and `inconsistent` outcomes expose no graph
output. They still close every admitted child unit with a terminal outcome for private
accounting.

When a datum or stage has several non-success children, its terminal variant follows
`inconsistent` → `lost` → `cancelled` → `failed` → `blocked`, while retaining an ordered,
deduplicated tuple of every bounded cause. Atomic groups retain the same cause tuple under
the single `withheld` variant. The precedence gives each scope one mutually exclusive
terminal variant without erasing concurrent evidence.

The logical-task transition relation is closed:

| Current state | Accepted event or guard | Next state |
| --- | --- | --- |
| `planned` | No earlier stage exists, or it succeeded; every direct dependency is `satisfied` | `ready` |
| `planned` | Any direct dependency closes as `unsatisfied` | terminal `blocked` |
| `planned` or `ready` | Cancellation is accepted before dispatch | terminal `cancelled` |
| `ready` | One dispatch is committed with a fresh attempt and row token | `dispatched` |
| `dispatched` | One exact keyed result passes reconciliation | terminal `succeeded` |
| `dispatched` | A definitive execution or verification error is accepted | terminal `failed` |
| `dispatched` | A trusted stop acknowledgement is accepted first | terminal `cancelled` |
| `dispatched` | Completion and stop become unobservable | terminal `lost` |
| Any nonterminal state | Evidence makes safe attribution impossible | terminal `inconsistent` |

No other transition is valid. `cancellation_requested` is an invocation flag, not a task
state; it changes a task only through the cancellation rows above. Terminal task states are
absorbing. Dependencies and parent scopes are deterministic reducers over terminal child
evidence rather than independently mutable state.

## Terminal precedence and reconciliation scope

Terminal states are absorbing. A later completion cannot reopen a task that is already
`cancelled`, `lost`, `blocked`, or `inconsistent`, and a terminal attempt token cannot be
reused. When several observations compete, the ledger applies this precedence:

1. An integrity contradiction that prevents safe attribution is `inconsistent`.
2. A previously accepted trusted terminal record remains authoritative.
3. A dispatched execution without a trusted terminal or stop record is `lost`.
4. Proven cancellation is `cancelled`.
5. A known execution or verification failure is `failed`.
6. An unscheduled task with a terminal unsatisfied prerequisite is `blocked`.

The ledger localizes a reconciliation fault only when it can still prove the complete
expected-to-observed bijection for every unaffected task. After a trusted batch run record
arrives, a missing expected token is local `inconsistent(missing)`; it is not `lost`. When no
trusted run record exists for a dispatched task, the task is `lost`. An unknown token,
duplicate token in one reconciliation set, plan mismatch, or contradictory initial record
destroys the batch or invocation bijection and closes the invocation as `inconsistent`,
withholding all groups. Implementations must not guess a narrower scope to preserve output.

The ledger serializes terminal acceptance. A byte-identical replay received after its
terminal record was accepted is an idempotent stale observation; any other late record for
that closed attempt is rejected as stale and cannot change state. A duplicate, foreign,
stale-at-admission, or contradictory record present before the first terminal acceptance is
an accounting inconsistency. Thus cancellation acknowledged before a late success remains
`cancelled`, while competing unsequenced evidence is `inconsistent`.

Existing `FailedRecord` values are evidence about dropped backend rows. The phase-4 adapter
must reconcile each value to one expected task by opaque token before it maps it to a task
failure. It must preserve the published `failed_records` compatibility shape at the facade;
unattributable or contradictory failures are invocation inconsistency, not anonymous row
loss. Existing content-derived public record IDs retain their published behavior only at
that facade. They must not be copied into graph outcomes or receipts, used for correlation,
or treated as proof of datum identity; exact public `step`, ID, ordering, and shape behavior
requires compatibility tests.

## Dependency propagation and release

Scheduling and release use different predicates. A task becomes ready when its fixed
earlier-stage task succeeded and every direct datum prerequisite is locally qualified: all
of that prerequisite's required tasks and datum-level release predicate passed, before
atomic-group propagation. This rule prevents known-bad prerequisites from causing more
effects. A dependent that already finished may later become release-ineligible when an
atomic peer of its prerequisite fails; its successful task evidence remains unchanged.

After every task is terminal, release qualification computes the least fixed point of these
monotone rules:

1. Mark every datum that is not `locally_qualified` ineligible.
2. Mark every member of an atomic group that contains an ineligible datum ineligible for
   release. Preserve each member's underlying datum outcome.
3. Mark every transitive dependent of an ineligible datum ineligible for release.
4. Repeat group and dependency propagation until no eligibility changes.

This fixed point is necessary because an atomic peer may feed a dependent in another group,
and that dependent's group may contain further peers. Failure affects an unrelated group
only through an explicit dependency path. Precisely accounted independent groups may be
released together in the final result even when another group is withheld. Any
invocation-global inconsistency withholds all groups.

## Cancellation, lost execution, and retry

A cancellation request is an event, not proof that execution stopped. Before dispatch, it
closes the task as `cancelled` without a backend call. After dispatch, a task is `cancelled`
only when the execution boundary provides a trusted stop acknowledgement. If a caller
abandons a waiter, a transport breaks, or a worker disappears without that evidence, the
task or invocation is `lost`. Likely completion is not evidence of success.

The terminal-record acceptance point is the cancellation linearization point. A trusted
terminal record accepted first wins. If proven cancellation closes the task first, a late
result is stale and cannot resurrect output. Cancellation after a task or invocation is
terminal has no semantic effect. Phase 4 must not claim that cancelling the public caller's
waiter stops synchronous DataDesigner or model work.

An invocation cancellation request accepted before the final publication point sets a
release embargo. The ledger still closes every admitted child, but the invocation terminates
as `cancelled` and exposes no group output, even if all effectful tasks had already
succeeded. Publication accepted first wins; a later cancellation request has no effect.

Phase 4 performs no automatic task retry. One dispatch is one attempt in the ledger;
provider- or client-internal behavior below `NddAdapter.run_workflow()` is opaque and creates
no Anonymizer retry guarantee. A downstream owner may start a new Anonymizer invocation,
which receives fresh opaque identities and reprocesses the declared input. The new call does
not reopen the lost attempt, reuse its output, deduplicate an effect, or imply durable
idempotency. Protected-byte retention, destination postcondition checks, and delivery retry
remain downstream responsibilities.

## Fail-closed release algorithm

The release barrier runs once, after exhaustive terminal accounting:

1. Verify that the compiled plan identity and all expected invocation, stage, task, datum,
   dependency, and group identities have exactly one compatible terminal record.
2. Reject or classify every missing, extra, duplicated, stale, or contradictory record
   according to the terminal-acceptance rules above.
3. Derive datum qualification without trusting workframe order or contents as identity.
4. Compute the dependency-and-group ineligibility fixed point.
5. Run the strategy-specific datum and group release predicates.
6. Construct complete outputs only for released groups.
7. Remove all invocation-private tokens and verify that no withheld output, raw fallback,
   or private diagnostic enters the returned compatibility result.
8. Publish one immutable terminal result. If result construction or publication
   verification fails, expose no graph output.

No iterator, callback, trace view, exception, partial DataFrame, or diagnostic may expose a
member output before its group passes this barrier. Atomic release here is a Python result
qualification rule, not a transaction over Platform artifacts, Intake persistence,
providers, telemetry, or delivery.

## Test strategy

### Independent reference model

Implement a small pure model before the executor. Its inputs are a compiled declaration and
a timestamp-free sequence containing only exogenous observations: dispatch acceptance,
terminal evidence, cancellation request, stop acknowledgement, and transport loss. The
model derives readiness, reconciliation, propagation, release, and the only legal task,
datum, dependency, stage, group, and invocation outcomes. Runtime scheduling or release
decisions are not model inputs. Review this oracle independently from the executor.

The model must enforce these equations:

```text
task ready = no earlier-stage task exists or it succeeded
             and every direct datum dependency is satisfied

datum locally qualified = every required task has exactly one accepted success
                          and the datum release predicate passes

group released = every member is locally qualified and release eligible
                 and the group release predicate passes

invocation completed = every admitted unit is terminal
                       and every dispatch has a trusted-evidence terminal classification
                           or is lost
                       and reconciliation is globally consistent
```

### Example and transition tests

Unit tests must cover every permitted transition and reject every transition out of a
terminal state. The minimum graph cases are empty and malformed input, singleton
compatibility, wide and deep DAGs, a diamond DAG, disconnected components, declaration
order different from topological order, and maximum accepted limits.

An empty datum set is `malformed_graph`, matching the phases 1–3 compiler; it is not a
vacuous completed invocation. Admission tests include one family for every validation tier,
exact-limit and limit-plus-one cases for every count and byte bound, adjacent-tier
multiply-invalid inputs, declaration permutations, and capability permutations. They must
prove the documented rejection precedence.

Admission tests must cover dangling endpoints, duplicate and self dependencies, cycles of
different lengths, empty groups, duplicate members, duplicate groups, coverage gaps,
partial overlap, strict nesting, and every still-unsupported context, coherence, relation,
strategy, and task cardinality. Effect spies start at graph compilation and require zero
invocation/ledger/token creation, execution-context opening, phase-4 resource acquisition,
provider or credential use, content-bearing workframe or telemetry, backend or NDD call,
reconstruction, and publication. Content-free compiler intermediates and pre-existing public
facade construction or request-start telemetry are outside this spy window.

Plan and ledger tests mutate source declarations after compilation, attempt nested plan
tampering, replay a compiled plan where disallowed, reuse a closed ledger, concurrently open
the same one-shot invocation twice, accept a second terminal record, and publish twice. The
immutable snapshot must remain detached, and every second one-shot action must fail closed.

Execution tests must inject known failure, cancellation before and after dispatch, stop
acknowledgement, transport loss, worker death, partial backend results, missing rows,
duplicate and foreign tokens, stale attempt results, contradictory records, hydration
failure, release-predicate failure, and result-construction failure. They must prove exact
terminal conservation, dependency blocking, fixed-point withholding, and no raw fallback.
Independent-group isolation applies only to localizable evidence in a globally consistent
invocation; a global attribution fault must withhold every group.

An identity cross-product uses equal text, equal structured values, duplicate and null-like
public indices, distinct datum IDs, multiple synthetic stages, and fresh invocations. It
independently permutes rows, terminal records, and `FailedRecord` evidence and covers a valid
token bound to the wrong stage or datum, swapped valid tokens, token reuse or collision,
plan mismatch, one attributable failure, and duplicate, unknown, stale, or
success-plus-failure evidence.

Cancellation tests fix the linearization expectations: accepted terminal evidence before
stop acknowledgement wins; accepted stop acknowledgement before a late result is
`cancelled`; a request after closure changes nothing; a post-dispatch request without
acknowledgement, waiter abandonment, or transport loss is `lost`. Pre-dispatch cancellation
or blocking causes zero dispatches; every task that reached `dispatched` causes exactly one,
including tasks that later fail, cancel, or become lost. A downstream re-invocation uses
fresh identities.

A minimal alternating group/dependency graph must require more than one fixed-point round.
One schedule lets a dependent finish before an atomic peer of its prerequisite fails. The
dependent remains locally successful but is withheld, and a disconnected group is
unchanged.

At least one process-kill smoke test uses a test-only crashable backend at the existing
execution seam and compares worker death with the pure model's `lost` result and empty
release set. It must not add a production process/session API. This supplies phase-4
failure-boundary evidence only; it is not a materially different semantic runtime or the
lifecycle proof reserved for phase 11.

### Generative and schedule tests

The exhaustive conformance envelope is finite: zero through four declared datums, one
through three synthetic semantic stages for non-empty graphs, one through the datum count
atomic groups with every flat partition enumerated, one dispatch per logical task, one
primary terminal observation per dispatch, at most one late
or contradictory observation per dispatch, and at most one cancellation request, stop
acknowledgement, and transport-loss event per invocation. Reconciliation corruption is one
of the closed missing, duplicate, unknown, foreign, stale, swapped, plan-mismatch, or
contradictory classes. Schedules that differ only by commuting events for independent tasks
share one canonical representative. The harness must emit a checked manifest containing
the generator version, exact graph count, exact canonical event-trace count, and a digest;
review freezes that manifest before executor comparison. Use seeded property tests beyond
this envelope and retain minimized failing graphs and traces.

The required properties are:

- **Conservation:** every admitted unit has exactly one terminal outcome.
- **Output conservation:** released output keys equal the group member set exactly once,
  follow the compatibility presentation order, and belong to no other group or invocation;
  withheld groups contain no output.
- **Reference equivalence:** the implementation equals the pure model.
- **Permutation invariance:** declaration, ready-frontier, workframe-row, and response order
  preserve ID-keyed/isomorphic semantics; compatibility presentation order may follow the
  admitted input order.
- **Identity invariance:** an opaque identity renaming yields an isomorphic result.
- **Content non-identity:** equal text and structurally equal values remain distinct datums.
- **Monotone withholding:** replacing any required success with a non-success never enlarges
  the release set.
- **Independent isolation:** a localizable failure in a globally consistent disconnected
  component cannot change another component; global attribution faults are excluded.
- **Batch invariance:** different ready-frontier batch sizes produce the same result.
- **Attempt isolation:** stale, foreign, or prior-invocation records cannot satisfy a task.
- **Boundedness and confidentiality:** accounting respects limits and emits no content or
  content-derived diagnostic.

Mutation tests must demonstrate that the suite detects positional joins, omitted terminal
records, premature group release, skipped propagation, accepted duplicate results, stale
attempt adoption, raw-input fallback, and late-result resurrection.

### Compatibility and boundary tests

Compatibility tests must exercise duplicate text; reordered, filtered, concatenated, and
index-reset workframes; unique, duplicate, non-monotonic, string, and null-like public
DataFrame indices; mixed successful and failed rows; and private-column collision. They must
verify original public presentation order, user columns, result columns and attributes,
`trace_dataframe`, `failed_records`, CLI behavior, and the sanitized public error surface.
Invocation tokens must not appear in public DataFrames, receipts, exceptions, tracebacks,
logs, or persisted fixtures. Preview selection must occur before graph compilation, and a
graph executor must never silently omit an admitted datum to reproduce preview sampling.

Privacy tests define a closed allowlist of public fields and bounded values. They inject
independent canaries for content, source identity, graph identity, row tokens, and their
known digests; capture `repr` and `str`, exception cause/context and traceback, logs,
telemetry, `trace_dataframe`, receipts, and serialization sinks; and reject every field not
on the allowlist. Canary substring scans supplement this structural check but do not stand
alone as proof that arbitrary content-derived values are absent. The grandfathered public
`FailedRecord` fields are checked against their compatibility contract separately.

Boundary tests must prove that all DataDesigner execution still passes through
`NddAdapter.run_workflow()` and that compilation performs no effects. Source-shaped ATIF,
OTLP, and chat-completion fixtures may test adapter projection pressure only after their
owners review the relevant field and atomic mappings. The tests must not encode complete
request withholding as current Intake behavior, claim byte-exact source fidelity, or import
source types into Anonymizer core.

## Review and implementation gates

The operator authorized implementation after reviewers completed review of:

1. the closed outcome algebra and precedence;
2. the DAG-only dependency policy and fixed-point propagation;
3. the flat exact atomic partition and rejection of nesting and overlap;
4. the cancellation linearization point, `lost` semantics, and no-retry decision;
5. the pure compile-before-effects barrier and global-integrity scope;
6. the independent reference model and test matrix; and
7. the preserved public, DataFrame, NDD adapter, privacy, and downstream ownership
   boundaries.

The 2026-08-20 review and operator checkpoint satisfied the seven pre-implementation
decisions above.
The Phase 4 completion gate requires the reference-model, transition, fault-injection, bounded schedule,
property, mutation, compatibility, privacy-canary, and full repository suites pass and an
independent reviewer accepts the evidence. Before later Platform adoption or stable
promotion, the NeMo Platform plugin owner must turn the current call-site compatibility
inference into executed public-facade smoke-test evidence. That cross-repository test is not
a prerequisite for starting the private phase 4 implementation.

The branch-local implementation passed these gates on 2026-08-25. The frozen `phase4-stream-v4`
reference corpus covers 8,278 admitted graphs and 397,542 canonical traces; its manifest digest is
`e778147bf77909ddb94117fe7e6c230de57e46a722fad49c563b36f0b5660efa`. The full repository run
reported 1,508 passed and 11 skipped tests with one pre-existing deprecation warning. Formatting,
type checking, documentation, privacy-canary, compatibility, process-loss, concurrency, and
mutation checks passed. The independent remediation verifier accepted all nine council
remediations with no remaining material findings. The maintained authenticated review-only
Arc then accepted the focused remediation with zero findings and passed every configured
host validation.

Passing Phase 4 does not authorize the Phase 5–10 branch checkpoints automatically, a public
graph or session surface, production Intake or OpenShell support, a wire protocol, transactional persistence,
an accepted privacy boundary, a `zero PII` claim, or stable promotion. It only satisfies the
phase-4 prerequisite after its evidence is reviewed.

## Evidence and unresolved gates

The workload evidence motivates the design but does not approve source mappings. ATIF
hierarchy, OTLP partial acceptance, and structured chat request/response fields demonstrate
why datum identity, dependencies, and atomic release cannot be reduced to DataFrame rows.
Intake owners still need to approve closed field roles, source commit units, retry identity,
and partial reconstruction. The customer or consuming-product owner still needs to select
the PII trust boundary and acceptance objective. Opaque provenance across adapter, process,
and artifact boundaries also remains unresolved. These are future adapter-adoption and
promotion gates, not prerequisites for implementing the authorized private Phase 4 design.

The authoritative evidence and constraints for this phase are the parent
[technical proposal](graph-native-anonymizer-sdk-technical-proposal.md), the separate
[Intake workload evidence](intake-workload-validation-evidence.md), the published facade and
`NddAdapter.run_workflow()` contracts cited there, and the branch-local phases 1–3 graph,
runtime, release, and verification tests at `702f43a988cf3673d16f40be5c59bc784737e1a3`.
