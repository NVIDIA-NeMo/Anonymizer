<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Graph SDK

Use `anonymizer.graph` to declare a workflow, provide its implementations, execute it locally, and inspect or project saved results. The SDK validates declared relationships and reconciles evidence before Protected release. A successful callback alone does not establish protection.

The [provider-free example](../examples/graph_sdk/local_workflows.py) uses only public imports. Save the file and run it with an environment containing `nemo-anonymizer`:

```bash
python local_workflows.py
```

It runs a custom text transform, then a separate transform-and-verifier workflow. The transform changes `Alice` to `[NAME]`; the custom verifier checks that exact name on the resulting candidate. This deliberately narrow example leaves the email address unchanged. Its `met` result applies to the declared name check, not general anonymization quality.

| Example | Callback counts | Qualification | Protected items |
|---|---|---|---|
| Custom transform | Transform: 1 | `not_assessed` | 0 |
| Transform with separate verifier | Transform: 1, verifier: 1 | `met` | 1 |

Both examples project the saved result twice and assert that callback counts do not increase. The same file exercises synchronous execution, rejection inside an existing event loop, admission failure before callbacks, a same-process decision wait/resume, and two calls to a `RemoteContext` subclass.

## Declare and execute

The example's `build()` function shows the complete declaration path:

1. Create text datums with `DataGraph.new()` and `add_text()`, then validate explicit targets and relationships under `DataLimits`.
2. Declare operation inputs, outputs, dependencies, outcomes and resource ceilings. Admit the static workflow, then its activation structure.
3. Declare implementation capabilities and select implementations during `prepare()`. The prepared plan binds the data, workflow, configuration, revisions and bounded activation inventory.
4. Admit context and execution policies. Supply `ExecutionServices` with the corresponding callback handles.
5. Call `await start_execution(...)`, then `await running.wait()`. Alternatively, call `execute_sync(...)` outside an active event loop.
6. Admit qualification and construct an explicit `evidence_revision_view`. Call `qualify(admitted=..., result=..., current=...)` with the saved execution result.
7. Bind a projection source. `project_protected` releases qualified current candidates; `project_content` selects retained artifacts for an explicit content view.

The admission functions return factory-protected values. Keep those values and pass them to their corresponding next boundary. Do not reconstruct an admitted plan or execution result from a dictionary. `ContractViolation`, `PreparationRejected`, and `EffectRejected` carry closed rejection codes; bounds and ownership checks occur before the affected work starts.

`LocalCallable.run` receives typed `AssociationInput` values and returns typed results. `ImplementationHandle` binds the callback to its admitted implementation, operation and configuration. An `AssessmentFinding` must come from an admitted evidence production with a supported finding and the exact evidence port. The example's verifier is a separate callback; the transform cannot turn its execution-only result into Protected content by selecting a projection name.

## Internal candidate-span artifacts

The engine also supports the closed artifact type `anonymizer.candidate-spans`,
revision 1. Its `EntitySpansValue` contains an immutable tuple of the existing
`EntitySpan` records and the exact source `ArtifactRef`. These are candidate
detections, not accepted subject identity or protection evidence. The local
witness produces spans, filters them, and renders a text candidate through the
existing replacement algorithm; ordinary Anonymizer execution is unchanged.

Publication checks exact offsets and substrings against a declared text input,
including the source artifact version. An identity-preserving span alias keeps
its source and its separate producing occurrence. Candidate payloads are
validated at callback and saved-result boundaries for closed shapes, bounded
content, and consistent cached counts.
Every artifact carries exact content-byte and collection-item counts; counts
cover all span string fields and span members, not Python object memory. Retained
reads and inspection revalidate cached counts for text, text collections (including
children), and spans. This consistency check preserves the separate admission
budgets for root data, bound context, and callback results.
Malformed Unicode, non-finite scores, mutable containers and forged counts are
rejected. Publication and inspection consume the same counts.

Candidate spans are transient: they cannot be root text bindings, retrieved text
materializations, map item/membership schemas, or continuity decisions. Continuity
admission rejects span decisions before execution, while existing text and text
collection snapshots retain their format. Initial binding rejects the reserved
candidate-span type name at every revision before allocating binding identities,
acquiring resources, or calling providers; custom text type names remain supported.
Protected projections still require text candidates. Arbitrary domain schema registration and DataDesigner request
accounting are separate implementation work.

## Saved results and current evidence

Ordinary public `qualify` consumes all retained accepted assessments. There is no public assessment-subset selector. It does not execute the graph again. You can construct another `evidence_revision_view` for the same saved result to describe which artifacts, absences, configurations and state revisions remain current, then qualify again. Selective artifact requalification can withhold a candidate that was previously releasable. Restoring the same valid current view can restore qualification without another callback.

Use `StateRevisionView.from_revisions(revisions=raw_revisions)` for an external tuple of state revisions. It checks the raw sequence before canonicalization, including duplicate effects. Constructing a frozenset first can discard information that validation needs.

A `CandidateRef` identifies an exact final artifact. Replacing it with a newer artifact is a different claim. A numerical version comparison cannot establish external freshness or transfer evidence to a different candidate.

## Decisions, continuity and snapshots

`RunningExecution.pending_decisions()` exposes installed waits. Submit a `DecisionResponse` containing that wait's identity, workflow and artifact through `submit_decision()`, then await `wait()`. `request_cancel()` requests cancellation. These operations belong to one process-local running task; the handle is not serializable and does not provide durable task resumption.

Continuity is explicit. Create or restore a `ContinuityState`, begin a bounded session, admit the workflow's continuity declarations, and commit qualified support with `commit_qualified`. Bind the resulting commit when projection requires it. Snapshot operations (`capture_snapshot`, `encode_snapshot`, `write_snapshot`, `load_snapshot`, `restore_snapshot`) use the bounded versioned continuity format. `write_snapshot` replaces the destination atomically. `fork_continuity`, `create_handoff` and `accept_handoff` operate on continuity state, not running execution handles. Load snapshots under explicit limits and supply the required expected revision when capturing state or creating a handoff.

## Context providers and blocking callbacks

Subclass `RemoteContext` and implement asynchronous `retrieve` and `cancel`, using typed request, association, selector, bounds and result values. The example's `MemoryContext` makes two initial-binding calls against a caller-owned provider. Each call retains its own response versions and request receipt. Production providers must return the matching request settlement and association identities; a returned object of the wrong shape is not successful retrieval.

`InitialContextDecl.version_selection` defaults to `"exact_one"`. With `"latest"`, an initial scalar binding can retain multiple versions of one source key and select their numeric maximum. This selection does not assert that the source is fresh. Adaptive retrieval has no `version_selection` field. Resource ownership is explicit: caller-owned resources remain the caller's responsibility; SDK-owned cleanup is recorded in the result.

Use `BlockingContextProvider` or `BlockingTransport` with a `BlockingPool` for supported synchronous callbacks. Declare `BlockingAffinity` for the callback and its captured handles. A queued call does not count as a physical dispatch. Cancellation does not terminate an ordinary Python worker; capacity remains occupied until that worker actually completes. A late return can settle usage without replacing the request's first semantic terminal.

## Rows, groups and map items

A datum is an explicit graph identity. For rows or a DataFrame, create a datum for each target text and retain your own mapping from row keys to `DatumId` values. Select the target IDs during validation. Declare `DatumDependency`, `CoherenceScope`, and `AtomicGroup` when records depend on one another, share a coherence scope, or require atomic release. A shared DataFrame index or nearby row position does not create these relationships.

A `MapItemPort` names an exact expander, expansion outcome, item input, member and nested path. `EvidencePort` permits either a scalar port name or this map-item reference. A map-item protection requirement applies to each actual member. Its `ProtectionRequirement.candidate_port` names the final candidate output whose ancestry must include that member; evidence on an unrelated branch cannot release it. A zero-member expansion supplies no member evidence and cannot invent a member to satisfy a candidate claim.

`VerifiedEvidence.subject` can be a scalar `CandidateRef` or a `MapItemSubjectRef`. The latter retains the item artifact and its exact producing occurrence. Currentness follows that artifact's identity and version, not a position in a collection. Nested paths, exact outcomes and ports remain part of the obligation. A scalar subject may project through an identity-preserving output alias; changing the content creates a different subject.

## Bounded observations and inspection

`begin_observations` creates a session with explicit count, depth, byte and duration limits. Binding and execution reserve their complete observation scope before effects. Explicit `observed_*` wrappers cover pure preparation, admission, qualification, continuity and projection boundaries. `close()` returns a cached receipt or an `ObservationFailure`; it never returns a partial receipt. Clock or sink failures do not change the semantic execution result.

Inspection binds the exact owners through `bind_inspection_source`, or `bind_binding_inspection_source` for initial binding alone. `explain`, `inspect` and `diagnose` provide bounded closed summaries; `encode_inspection` emits canonical JSON. Forecasts distinguish exact counts, known upper bounds and unknown values. Missing phases and unknown usage remain unknown. Content projection is separate from this content-free inspection encoder. Provider logging and arbitrary host callback side effects are outside the observation privacy boundary.

## Staged delegated request receipts

`DelegatedRequestContract` declares qualified or opaque coverage, fixed request
slots from the prepared activation recipe, parser routes and follow-up kinds.
Its usage requirement defaults to `exact`; `unknown_allowed` preserves unknown
usage as unknown while still requiring complete request and remote-stop facts.
Ordinary external operations retain exact-usage qualification.

Qualified structural admission currently supports fixed sibling operation slots
in one lexical container and target. `RequestChild.leaf` identifies a physical
attempt separately from its semantic association. Qualification joins every used
leaf to its admitted slot and canonical terminal, checks explicit predecessor and
parser evidence, and confines recovery to the validated chain. A recovered leaf
failure does not erase an unrelated semantic failure or a provenance defect.
Record-only request reducer fixtures may omit `leaf`; protected child receipt
qualification requires it.

A bounded runtime subset now submits one generation fragment once through an
`ImplementationHandle.delegated` callable declaring `execution="same_loop"`.
The invocation supplies restricted request-slot access to `submit(inputs, requests)`.
Its `request(slot, ...)` method uses the declared leaf transport, and
`parser_rejected(request)` accepts evidence for a subsequent declared correction
after completed settlement and an initial malformed-response terminal. It cannot
reclassify a successful terminal.
The same invocation owns slot registration, Start, Dispatch, terminal publication
and the request ledger. The parent makes no physical request and adds no charge.

Service admission currently permits one delegated owner and one target, one
parser route, correction-only follow-ups, stateless asynchronous transports, and
empty input/output ports on the physical leaves. The parent uses normal typed
input resolution and output publication; a local assessor can qualify its
produced candidate. The fixed physical
slots are mandatory: returning before all slots execute fails the submission.
Other external operations, adaptive retrieval and concurrent borrowed requests
remain gated in this subset. The ordinary borrowed owner rejects other loops/tasks
and expires when submission finishes. These are executable graph-owner fixtures;
no DD provider route is enabled yet.

An optional `DelegatedReadiness` descriptor names three additional fixed sibling
slots: a semantic readiness obligation, its distinct physical request leaf, and
a local capability probe. `requests.readiness()` starts and closes the semantic
obligation through that one physical request; `requests.probe()` executes the
actual local callable after readiness succeeds. Generation requires the declared
readiness result and a declared successful probe continuation, such as
`available` or `async_required`. Discovering that a synchronous client is
unavailable can therefore permit asynchronous generation without recording a
physical probe request. Readiness failure or a probe failure stops generation.

Readiness requires exact usage and has one route with no retries. Its descriptor
is the only authorization for a semantic external node to borrow the submitting
owner rather than supply an independent transport handle. Qualification joins
that node's canonical attempt to its declared request leaf and rejects extra
ordinary charges involving the semantic owner, physical leaves or local probe.
The local probe uses the existing local completion path; self-cancellation closes
its canonical attempt, and cancellation of its observer retains owned work until
the callback drains. This adds no shared readiness across targets.

The internal `WorkerDelegatedSubmitter` adds one bounded synchronous submission
through the existing caller-owned `BlockingPool`. Its parent capability declares
`resource_lifetime="caller_owned"`, and its `ImplementationHandle.resource` is a
caller-owned `ResourceLease` for that exact submitter, with detachment forbidden.
Missing, mismatched, SDK-owned or closed resources reject before submission.
Cleanup records the caller lease as `left_open`; it does not close the pool or
loop. The submitter declares one borrowed async loop, which the caller must keep
running through drain. The worker receives an
owner-issued capability whose context propagates through `asyncio.to_thread` and
`run_coroutine_threadsafe`. Both the exact issued object and its context must match
at the invocation owner; an expired capability, foreign context, undeclared loop
or simultaneous callback is rejected before that request's effects. Ordinary
same-loop submission retains its submitting-task restriction.

The worker and each registered child remain retained through caught, propagated
and repeated observer cancellation. Physical transports, classification, Start,
terminal facts and publication still execute on the invocation loop. A worker
submission is not a physical request. Cancellation before worker entry submits
nothing; cancellation after physical entry preserves the first terminal and
usable late settlement while draining local work. The caller owns the pool and
borrowed loop; the executor neither closes nor replaces them. The pool must be
open at service admission. These are live execution resources, not stateless
handles: this invocation drains its retained worker and children without waiting
for unrelated caller-pool work. Other captured values must satisfy the declared
cross-thread/stateless profile. Arbitrary synchronous
code cannot be stopped by cancelling an asyncio observer. Foreign-loop physical
transports and owned-client close remain unsupported pending separate entry and
resource-lifetime conformance. This does not enable a DataDesigner route.

The shared asynchronous request driver now admits Dispatch inside the retained
call task immediately before callable entry. The delegated path preflights Start,
then commits Dispatch and the prepared Start without an intervening await.
Cancellation before that entry cancels the unused reservation without a charge;
post-entry cancellation retains the existing known-stop or Lost accounting.
Blocking transports retain their existing worker-entry boundary.

Opaque structural admission remains unsupported because the capability schema
cannot yet express unknown physical request bounds truthfully. Hard-budget
preparation retains its implementation-owned-retry rejection. Optional unused
slots, foreign-transport entry and resource draining, multi-target
shared ownership, dynamic recipes and provider-error exhaustion require later
bounded integration units. Ordinary injected-runner compatibility must gain its
explicit opaque path before production cutover.
