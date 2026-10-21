<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Graph SDK

Use `anonymizer.graph` to declare a workflow, provide its implementations, execute it locally, and inspect or project saved results. The SDK validates declared relationships and reconciles evidence before Protected release. A successful callback alone does not establish protection.

Until the SDK is published, its surface is not a compatibility contract. This covers `anonymizer.graph` and the engine modules behind it: names, export order, signatures, defaults, exception constructors, pickles and import cost may all change. Tests that pin them are refactor safety nets and are re-recorded in the same commit as the change they cover. The `Anonymizer` facade keeps its own compatibility promises.

The [provider-free example](../examples/graph_sdk/local_workflows.py) uses only public imports. Save the file and run it with an environment containing `nemo-anonymizer`:

```bash
python local_workflows.py
```

It runs a custom text transform, then a separate transform-and-verifier workflow. The transform changes `Alice` to `[NAME]`; the custom verifier checks that exact name on the resulting candidate. This deliberately narrow example leaves the email address unchanged. Its `met` result applies to the declared name check, not general anonymization quality.

| Example | Callback counts | Qualification | Protected items |
|---|---|---|---|
| Custom transform | Transform: 1 | `not_assessed` | 0 |
| Transform with separate verifier | Transform: 1, verifier: 1 | `met` | 1 |

Both examples project the saved result twice and assert that callback counts do not increase. The same file exercises synchronous execution, rejection inside an existing event loop, admission failure before callbacks, and two calls to a `RemoteContext` subclass.

## Declare and execute

The example's `build()` function shows the complete declaration path:

1. Create text datums with `DataGraph.new()` and `add_text()`, then validate explicit targets and relationships under `DataLimits`.
2. Declare operation inputs, outputs, dependencies, outcomes and resource ceilings. Compile the complete graph with `compile_graph()`. Its `nodes` tuple contains operations, subgraphs and any choice, map, loop or keyed-join controls; supply `dynamic_limits` when those controls expand work.
3. Declare implementation capabilities and select implementations during `prepare()`. The prepared plan binds the data, workflow, configuration, revisions and bounded activation inventory.
4. Admit context and execution policies. Supply `ExecutionServices` with the corresponding callback handles.
5. Call `await start_execution(...)`, then `await running.wait()`. Alternatively, call `execute_sync(...)` outside an active event loop.
6. Admit qualification and construct an explicit `evidence_revision_view`. Call `qualify(admitted=..., result=..., current=...)` with the saved execution result.
7. Bind a projection source. `project_protected` releases qualified current candidates; `project_content` selects retained artifacts for an explicit content view.

`Graph` is the shared admitted representation consumed by activation, preparation
and execution. `OperationNode` and `SubgraphNode` describe executable work;
`ChoiceDecl`, `MapDecl`, `LoopDecl` and `KeyedJoinDecl` are control nodes in the
same body. `Graph.operations` selects executable nodes and `Graph.bodies()`
walks the declared topology, including shared subgraphs once. Compilation checks
port and outcome compatibility, protection summaries, control ownership and
finite expansion bounds before returning the graph.

The admission functions return factory-protected values. Keep those values and pass them to their corresponding next boundary. Do not reconstruct an admitted plan or execution result from a dictionary. `ContractViolation`, `PreparationRejected`, and `EffectRejected` carry closed rejection codes; bounds and ownership checks occur before the affected work starts.

`LocalCallable.run` receives typed `AssociationInput` values and returns typed results. `ImplementationHandle` binds the callback to its admitted implementation, operation and configuration. An `AssessmentFinding` must come from an admitted evidence production with a supported finding and the exact evidence port. The example's verifier is a separate callback; the transform cannot turn its execution-only result into Protected content by selecting a projection name.

## Internal candidate-span artifacts

The engine also supports the closed artifact type `anonymizer.candidate-spans`,
revision 1. Its `EntitySpansValue` contains an immutable tuple of the existing
`EntitySpan` records and the exact source `ArtifactRef`. These are candidate
detections, not accepted subject identity or protection evidence. The local
witness produces spans, filters them, and renders a text candidate through the
existing replacement algorithm. Ordinary Redact uses the same typed artifacts as described below.

Publication checks exact offsets and substrings against a declared text input,
including the source artifact version. An identity-preserving span alias keeps
its source and its separate producing occurrence. Candidate payloads are
validated at callback and saved-result boundaries for closed shapes, selected
resource limits, and consistent cached counts.
Every artifact carries exact content-byte and collection-item counts; counts
cover all span string fields and span members, not Python object memory. Retained
reads and inspection revalidate cached counts for text, text collections (including
children), and spans. This consistency check preserves the separate admission
budgets for root data, bound context, and callback results.
Malformed Unicode, non-finite scores, mutable containers and forged counts are
rejected. Publication and inspection consume the same counts.

Candidate spans are transient: they cannot be root text bindings, retrieved text
materializations, or map item/membership schemas. Initial binding rejects the reserved
candidate-span type name at every revision before allocating binding identities,
acquiring resources, or calling providers; custom text type names remain supported.
Qualification is not limited to single text values: a text-collection subject
qualifies under the same evidence and coverage checks. Protected projections still
require text candidates and reject other values as unsupported. Arbitrary domain
schema registration and DataDesigner request accounting are separate implementation work.

## Count and byte policies

Artifact, collection-item, activation and pending-work limits remain finite.
Physical request bounds may be explicitly unknown for opaque execution as described below. `ResourceCeiling.max_input_bytes`, `max_output_bytes`,
`ExecutionLimits.max_runtime_artifact_bytes`, and the text/metadata byte fields
of `IngressLimits` accept `None` when no hard byte ceiling is selected.
`IngressLimits` defaults to that byte policy; explicit finite ceilings still
reject overflow. A workflow or replacement with no byte ceiling cannot satisfy
a finite parent byte promise. The policy is fixed before effects; the executor
does not enlarge it after receiving an output.

Admission accepts `IngressLimits` and each delegated request contract
(`DelegatedRequestContract`, its `PhysicalRequestPolicy`, `DelegatedReadiness`,
`ChildRoute` and `DelegatedGenerationSelection` members) only as exact types;
subclasses reject with `invalid_type`. The issued `IngressLimits` is a copy, so
later changes to the caller's object never reach a run.

Actual byte counts and independent UTF-8/shape checks remain mandatory. An absent
ceiling does not claim bounded memory usage or a hard transport-response size.
Context retrieval, observations and other explicitly bounded contracts retain
their separate limits.

The qualified development detection/Redact profile derives count capacity from the selected
source and declared work when no execution limits are supplied. Five retained
artifacts per source is an upper bound: one root, three producer outputs and one
assessment output. Final span count is bounded by source character count;
validation membership is bounded by the admitted chunk capacity. Actual counts
can be lower. Caller-supplied execution limits remain authoritative. The separate qualified development profile retains this five-artifact inventory.

This counts-first policy records the 2026-10-13 operator clarification that
ordinary runs should not impose incidental byte or per-string ceilings. It
qualifies the RFC's earlier mandatory-retained-byte-bound wording; it does not
claim the absent ceiling meets that earlier hard-byte interpretation. Other
admission, accounting, qualification and public-compatibility requirements remain.

## Ordinary local replacement execution

Ordinary production-owned Redact, Hash and Annotate runs, previews and CLI commands select the
execution-only graph path before detection. Detection still runs once on the
whole frame through `NddAdapter.run_workflow()`, with the existing DataDesigner
retry, chunk scheduling and provider configuration. Authenticated source positions,
typed spans and canonical text artifacts support the existing `PublishedFrame`;
there is no inference from text, indexes or row order. This path makes no
Protected claim.

The ordinary profile reserves at most three semantic artifacts per selected
source: initial text, final spans and transformed text. Span membership is bounded
by each source character count. Ingress reserves their sum across the selected
batch; publication checks each span artifact against the largest selected source.
These are upper bounds; actual returned counts are recorded. Physical calls,
usage and remote outstanding work are unknown to this path. An absent
`ExecutionLimits.max_remote_outstanding` ceiling is expressed as `None`.
Opaque execution rejects every explicit finite remote ceiling before callbacks,
including zero, and rejects a selected hard request budget. A selected hard
retained-byte ceiling is likewise rejected for opaque execution because arbitrary
frame object cells cannot be completely measured. Finite ceilings on
visible paths retain their existing enforcement.

## Substitute and injected replacement callbacks

Substitute and injected detection or replacement runners execute once on the whole
frame through the same graph executor. The compatibility result retains their
DataFrames, failed records and original errors. It does not infer source
associations from returned row positions, indexes or text, and it makes no
Protected claim. Saved public results retain their ordinary content and exclude
live execution resources. Substitute keeps its row-local map generation, instructions,
collision handling and no-entity bypass. Its map workflow may drop rows; the
compatibility frame does not infer their source associations. Hash retains its
existing text-based UTF-8 digest and Annotate retains its original-text rendering.
Empty-render behavior for Hash and Annotate is unchanged.

This path declares external work with unknown physical request visibility, usage
and upper bound. Its recorded dispatch inventory contains only observed SDK
requests; an empty inventory does not mean the callback performed no external
work. Inspection reports unknown totals and rejects hard-request-budget support.
A selected hard request budget is rejected before effects. The private profile
also rejects supplied execution limits it cannot guarantee. Its one callback
uses existing worker admission and draining. The synchronous facade runs injected
callbacks on the thread that called the public entry point, with a copy of its
context variables, including when that thread already has an event loop;
caller-owned thread-affine resources stay valid.
Cancellation does not establish
that hidden remote work stopped.

The operation owns one compatibility result containing detection and replacement
frames. Their row counts are observed after return, without assuming they match
the input cardinality. Direct string-cell byte sizes are measured; sizes of
arbitrary object cells remain unavailable. These opaque frames are separate from per-source semantic
artifacts. The separate qualified DataDesigner development route retains its existing source,
physical-request and evidence checks.

The registered chunked-validation plugin admits each synchronous or asynchronous
row callback as its own opaque invocation. Its unchanged dictionary payload is
retained in the existing compatibility frame and projected only after completion.
The async path keeps model calls on the caller's borrowed DD loop and preserves
its context; neither entry point closes caller-owned clients or loops. Failed
chunks retain their live siblings until they drain. A synchronous bridge timeout
still raises its original timeout error, but waits for actual coroutine completion;
it cannot promise finite drain when a borrowed loop or request never finishes.
Each call has an independent scope, including calls made inside an enclosing
Anonymizer invocation; original context preserves any already-registered request
owner without inventing a shared parent from rows or model registries.
This covers the owned validation plugin only. Native external readiness, detection,
augmentation and the enclosing create/preview launch remain outside that scope.
The native builder, nine columns, plugin kinds, JSON and partition contract remain
unchanged. Validation completion supplies no Protected or complete-request claim.

Public Rewrite run and preview use the same invocation owner for detection and
one selected existing Rewrite runner. Both the sequential and combined DD
schedules, including their repair loops, retain their existing behavior. Injected
runners keep their original whole-frame arguments and caller thread. The result
retains ordered detection and Rewrite failures and a transient execution owner.
This compatibility path has unknown physical request counts and no per-row
semantic lineage or Protected claim; the existing stage controllers remain in use.
When an injected runner delegates to a builtin Rewrite runner (for example
`super().run(...)` or a wrapper calling a builtin's `run`), that builtin runs as a
separate, nested invocation with its own owner, as on main, rather than as part of
the enclosing invocation. Its body, including its DataDesigner calls, runs on the
caller's thread like every other synchronous body.

Public `Anonymizer.evaluate()` admits a fresh execution-only invocation for its
whole-frame judge sequence, including injected evaluation runners. It preserves
the caller thread, judge selection, optional detection-validity checks and existing
noncritical failure metrics. Its canonical frame retains the pre-judge trace,
returned content and any prior live execution owner. A pickled public result
contains content and strategy configuration, not execution resources; evaluating
it creates fresh authority without reusing historical runtime identities. Judge
calls remain opaque, and evaluation does not establish a Protected claim.

Native diagnostic joins use validated adapter source positions, so repeated record
IDs cannot transfer judgments between rows. Dropped judge outputs leave their
source rows in place with unavailable scores. Standalone judges compose positions
through passthrough partitions; saved-result evaluation uses the same joins.
These positions describe host row correspondence, not graph artifact identity.
Legacy injected results may explicitly omit positions and retain their existing
unassociated compatibility path.

## Threads and context variables

`Anonymizer.run()`, `preview()` and `evaluate()` run any code you supply (subclass
overrides, injected runners and detectors, adapter `run_workflow` overrides) on the
thread that called them, at every depth, including built-in code reached through
`super()` that calls your code again. Built-in detection, replacement and Rewrite
bodies, and every DataDesigner `create()`/`preview()` call and wait, run there too.
Your code therefore sees that thread's thread-locals and can install signal handlers
exactly when that thread is the main thread, and Ctrl-C reaches the run as in 0.4.0.
The graph's own bookkeeping runs on that thread as well, on a private event loop that
leaves a notebook's or `asyncio.run` caller's running loop untouched; only
DataDesigner's engine threads run Anonymizer's column generators and send model
requests, as in 0.4.0, and none of them runs your code. Each call runs in a copy of the caller's
context variables: values set before the call are visible inside, and values set
inside are not visible to the caller afterwards.

## Saved results and current evidence

Ordinary public `qualify` consumes all retained accepted assessments. There is no public assessment-subset selector. It does not execute the graph again. You can construct another `evidence_revision_view` for the same saved result to describe which artifacts and configurations remain current, then qualify again. Selective artifact requalification can withhold a candidate that was previously releasable. Restoring the same valid current view can restore qualification without another callback.

A `CandidateRef` identifies an exact final artifact. Replacing it with a newer artifact is a different claim. A numerical version comparison cannot establish external freshness or transfer evidence to a different candidate.

## Running executions

`start_execution()` returns a `RunningExecution`. Await `wait()` for the result; `request_cancel()` requests cancellation. These operations belong to one process-local running task; the handle is not serializable and does not provide durable task resumption. Operations cannot pause for an external decision.

## Context providers and blocking callbacks

Subclass `RemoteContext` and implement asynchronous `retrieve` and `cancel`, using typed request, association, selector, bounds and result values. The example's `MemoryContext` makes two initial-binding calls against a caller-owned provider. Each call retains its own response versions and request receipt. Production providers must return the matching request settlement and association identities; a returned object of the wrong shape is not successful retrieval.

`InitialContextDecl.version_selection` defaults to `"exact_one"`. With `"latest"`, an initial scalar binding can retain multiple versions of one source key and select their numeric maximum. This selection does not assert that the source is fresh. Resource ownership is explicit: caller-owned resources remain the caller's responsibility; SDK-owned cleanup is recorded in the result.

Use `BlockingContextProvider` or `BlockingTransport` with a `BlockingPool` for supported synchronous callbacks. Declare `BlockingAffinity` for the callback and its captured handles. A queued call does not count as a physical dispatch. Cancellation does not terminate an ordinary Python worker; capacity remains occupied until that worker actually completes. A late return can settle usage without replacing the request's first semantic terminal.

## Rows, groups and map items

A datum is an explicit graph identity. For rows or a DataFrame, create a datum for each target text and retain your own mapping from row keys to `DatumId` values. Select the target IDs during validation. Declare `DatumDependency`, `CoherenceScope`, and `AtomicGroup` when records depend on one another, share a coherence scope, or require atomic release. A shared DataFrame index or nearby row position does not create these relationships.

A `MapItemPort` names an exact expander, expansion outcome, item input, member and nested path. `EvidencePort` permits either a scalar port name or this map-item reference. A map-item protection requirement applies to each actual member. Its `ProtectionRequirement.candidate_port` names the final candidate output whose ancestry must include that member; evidence on an unrelated branch cannot release it. A zero-member expansion supplies no member evidence and cannot invent a member to satisfy a candidate claim.

`VerifiedEvidence.subject` can be a scalar `CandidateRef` or a `MapItemSubjectRef`. The latter retains the item artifact and its exact producing occurrence. Currentness follows that artifact's identity and version, not a position in a collection. Nested paths, exact outcomes and ports remain part of the obligation. A scalar subject may project through an identity-preserving output alias; changing the content creates a different subject.

## Bounded observations and inspection

`begin_observations` creates a session with explicit count, depth, byte and duration limits. Binding and execution reserve their complete observation scope before effects. Explicit `observed_*` wrappers cover pure preparation, admission, qualification and projection boundaries. `close()` returns a cached receipt or an `ObservationFailure`; it never returns a partial receipt. Clock or sink failures do not change the semantic execution result.

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
parser route, correction follow-ups (plus retry follow-ups when the observed
request replay is idempotent), stateless asynchronous transports, and empty
input/output ports on the physical leaves. The parent uses normal typed
input resolution and output publication; a local assessor can qualify its
produced candidate. Fixed physical slots are mandatory by default: returning
before all slots execute fails the submission.

Same-loop and worker submissions, including the registered foreign-client profile,
may declare `optional_tail=(Q,)` for exactly
two generation slots `(G, Q)`, one route with two parse attempts and correction-only
follow-ups. G must be initial; Q may execute only as its authenticated correction.
Successful G can leave Q blocked without a Start, attempt or physical request.
Only after D's normal output staging and structural publication commit successfully
does Q's canonical terminal gain `unused_retry_capacity` and a typed
`UnusedRetryCapacity(owner=D_attempt, after=G_attempt)` cause. Cancellation,
rejected borrowing, denial, malformed return, failed publication or used Q cannot
acquire this disposition. A later independent assessment failure still withholds
protection. Qualification checks the declared Q occurrence, actual D/G attempts,
published D output, complete request history and settlement; it never fabricates
Q evidence. Ordinary omitted-context semantics remain unchanged.
Other external operations and concurrent borrowed requests
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
at the invocation owner; an expired capability, foreign context or undeclared
loop is rejected before that request's effects. Physical requests on different
request chains run concurrently: the owner serializes their checks and
reservation, then releases its gate while each request is in flight, so
DataDesigner's `max_parallel_requests` bounds them. A second request on a chain
whose previous request is still in flight is rejected, and a declared
remote-outstanding limit queues new requests behind in-flight ones. Any other simultaneous callback is
queued until the in-flight requests settle. Ordinary same-loop submission
retains its submitting-task restriction and one borrowed child at a time.

The worker and each registered child remain retained through caught, propagated
and repeated observer cancellation. Physical transports, classification, Start,
terminal facts and publication still execute on the invocation loop. A worker
submission is not a physical request. Cancellation before worker entry submits
nothing; cancellation after physical entry preserves the first terminal and
usable late settlement while draining local work.

The Substitute map, like every other facade stage, runs on DataDesigner's own
transport, retry policy and scheduling, as in 0.4.0; the graph accounts it as one
opaque invocation with unknown physical requests. A Ctrl-C cancels DataDesigner's
runs through DataDesigner's own scheduler cancellation, as in 0.4.0. The caller owns
the pool and borrowed loop; the executor neither closes nor replaces them. The
pool must be open at service admission. These are live execution resources, not stateless
handles: this invocation drains its retained worker and children without waiting
for unrelated caller-pool work. Other captured values must satisfy the declared
cross-thread/stateless profile. Arbitrary synchronous
code cannot be stopped by cancelling an asyncio observer. The registered foreign
client profile below adds loop-affine physical transport and cleanup. This does
not enable a DataDesigner route.

The shared asynchronous request driver now admits Dispatch inside the retained
call task immediately before callable entry. The delegated path preflights Start,
then commits Dispatch and the prepared Start without an intervening await.
Cancellation before that entry cancels the unused reservation without a charge;
post-entry cancellation retains the existing known-stop or Lost accounting.
Blocking transports retain their existing worker-entry boundary.

Opaque structural admission remains unsupported because the capability schema
cannot yet express unknown physical request bounds truthfully. Hard-budget
preparation retains its implementation-owned-retry rejection. Broader optional
tails, multi-target shared ownership,
dynamic recipes and provider-error exhaustion require later
bounded integration units. Ordinary injected-runner compatibility must gain its
explicit opaque path before production cutover.

The internal X2 foreign-entry driver witness binds a transport and cancellation
hook to one exact borrowed loop. Its retained task reaches the final send boundary
on that loop and synchronously waits for a non-awaiting invocation-loop gate.
That gate preflights the registered Start, admits Dispatch through the existing
request authority, and commits Start. Acceptance returns directly into the bound
transport coroutine without another asynchronous suspension. The foreign thread
must hold no host, client, request or adapter lock while waiting; the invocation
loop must never synchronously wait for that thread.

A timeout or cancellation that wins before acceptance invalidates the queued gate;
a later callback cannot start or charge the request. If acceptance wins, the
retained stack enters the original transport once despite observer cancellation.
The driver retains classification and late settlement through repeated observer
cancellation. Loop stop/resume tests keep the same borrowed loop and demonstrate
that public completion waits for its retained work. Permanently losing that loop
violates the required lifetime and cannot establish successful completion or
confirmed remote stop. Neither cancellation nor request accounting closes or
replaces caller resources.

An unregistered `ForeignLoopTransport` remains rejected before invocation identity
is allocated. Its driver witnesses cover entry and cancellation without granting
service admission. The registered client profile below adds public graph execution
and resource cleanup; it does not qualify a DataDesigner route.

The shared request driver retains an active cancellation hook through observer
cancellation. A completed hook that self-cancels or raises an ordinary exception
provides no stop confirmation: retained requests continue through Lost, drain and
late settlement. The hook cannot cancel its caller by cancelling its own task.
This uses the existing request authority and preserves the first terminal; it
does not expand foreign service admission or ordinary call-lifetime profiles.


The registered foreign-client profile uses one worker owner, one target, one shared
readiness obligation, one local probe and either one mandatory generation leaf
or the declared `(G, Q)` optional-correction shape. Every
physical leaf uses a `ForeignLoopTransport` registered on the worker's declared
borrowed loop. `ForeignLoopTransport.create(factory=..., owner=...,
exclusive=True, gate_timeout_ns=...)` runs the client factory on its actual loop
and issues the exact `ResourceLease` used by its implementation handle. The
capability declares `caller_owned` or `executor_owned`, with detachment forbidden.
The factory must create a fresh client confined to that loop; the caller grants
this adapter exclusive use until the invocation drains. Registration does not
establish safety for aliases or arbitrary concurrent raw-client users.

After all pre-effect validation, execution atomically claims the unique adapters
before allocating its invocation identity. A failed multi-adapter claim consumes
none. Each adapter serves one admitted invocation; stale, mismatched, closed,
wrong-loop and repeated handles reject before dispatch. The client identity and
creation loop remain pinned through request entry and cleanup.

The adapter retains the actual scheduled send and stop-hook Futures. A Lost or
confirmed-stop request fact cannot erase these local resource users. The existing
owner drains its child classification, callbacks and worker before cleanup; the
resource boundary also checks the retained users directly. SDK-owned clients
close once on their recorded creation loop. Close remains retained through
observer cancellation; an exception records `close_failed`, and a self-cancelled
close records `close_unknown`. Existing cleanup associations withhold the affected
target from qualification. A permanently unavailable borrowed loop cannot prove
successful drain or confirmed remote stop.

Caller-owned clients, pools and loops stay open. Only this invocation's users
drain; unrelated caller-pool work does not delay its result. The adapter is spent
after that invocation, while the caller's client remains available for subsequent
caller use. The public optional-tail fixture executes two sends when G succeeds
or three when Q corrects malformed G, publishes a typed candidate, and qualifies
it through the existing local assessment path. All registered SDK-owned clients,
including an unsent Q client, close once on their creation loop after actual users
drain. Its cleanup association remains required even without a Q request.
Execution captures an invocation-local registration for every resource-bearing
service handle before running operations.
Qualification derives that complete handle-key domain from the admitted
nonstateless implementations before joining resource owners and cleanup
facts. Removing a cleanup fact and its association cannot hide a declared handle
resource; removing its registration also fails the domain check. Shared leases
still produce one cleanup fact.
Initial-binding acquisition membership remains a separate retained-proof obligation;
normal runtime cleanup includes those resources.
Broader recipes, shared targets, additional optional shapes, opaque execution and actual DD
provider routes remain separate admission work.
