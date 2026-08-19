<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Extensible SDK companion plans

Status: Proposed design report, updated 2026-08-19. This document does not authorize a
public API, a production Intake integration, a Relay integration, or a release
claim. Names are provisional.

Except where a sentence cites an online repository revision as current
behavior, every type, operation, ownership boundary, gate, and lifecycle
statement below is proposed.

## Recorded adoption decisions

As of 2026-08-19, the Intake team is the named Plan A adopter. The project owner
accepted the canonical error contract implemented by this design spike:
pipeline failures that cross the private row-verification boundary become a
cause-free `AnonymizerWorkflowError` at public `run()` and `preview()` entry
points. Preflight, configuration, and input failures raised before that
boundary retain their established behavior.

The same review records a narrow atomicity contract for those public calls.
`run()` and `preview()` publish one complete Python result or raise without a
partial result attached to the exception. `failed_records` are explicit dropped
records in an otherwise successful result. This is result-publication
atomicity, not a transaction over model providers, telemetry, source ingestion,
or durable storage.

For current design and validation planning, use **before Intake** as the
provisional PII boundary: no unprotected payload may enter the Intake service
process. This is a working assumption, not a customer-approved requirement.
The Intake team still owns routing the final decision to the consuming-product
owner. Raw dogfood requests remain synthetic characterization evidence and do
not validate a deployment against this assumed boundary.

## Decision

Develop one semantic protection core through two companion plans:

1. **Plan A: private Intake-first protection runtime.** Implement the smallest
   synchronous, bounded, source-neutral contract that Intake and Python jobs
   can validate through the current Anonymizer engine.
2. **Plan B: composable multi-host SDK.** Design the fuller intent, plan,
   runtime, record, receipt, capability, wire, and agent-task language now, but
   promote only semantics demonstrated by Plan A and a materially different
   runtime.

Plan A is the implementation target. Plan B is the architecture and promotion
target. Plan A must leave a clean expansion path, but it must not contain empty
protocols, speculative wire models, or optional fields that impersonate future
semantics.

This proposal sets these design constraints:

- Existing `Anonymizer.run()`, `preview()`, `evaluate()`, and CLI behavior
  remains the compatibility baseline.
- Every Python execution path converges on one normalized dataframe pipeline
  and the existing `NddAdapter.run_workflow()` boundary.
- ATIF, OTLP, chat-completion, direct-span, OCSF, Intake, and Relay types remain
  outside Anonymizer core.
- Source adapters own codecs, closed field policy, projection, reconstruction,
  schema validation, and item atomicity.
- Intake owns durable staging, replay, leases, commits, retention, purge, and
  delivery. Relay owns queueing, saturation, omission, plugin or worker
  lifecycle, and subscriber delivery.
- Public portable-contract shipment remains gated on an independently
  implemented second semantic runtime and reviewed provenance and PII-boundary
  decisions. A process-backed Python host supplies lifecycle evidence only.
- Current OpenShell telemetry is architecture evidence only. The Anonymizer
  OCSF adapter remains test-only; neither artifact is a second runtime or
  authority to change OpenShell.

## Why the design is layered

The public boundary, trusted domain, runtime, and engine need different types:

```text
dynamic or compatibility input
        |
        v
boundary parser
        |
        v
immutable intent and compiled plan
        |
        v
effectful protection flow
        |
        v
bounded public run record
        |
        v
compatibility, wire, or bounded agent projection
```

Pydantic remains appropriate for compatibility parsing and schema publication.
Frozen dataclasses and closed unions carry the trusted internal contract.
Pandas, DataDesigner, model clients, credentials, and source models stay at
effect or adapter boundaries.

The architecture has two nested compilation chains:

```text
ProtectionSpec
  -> compile against declared capabilities
  -> ProtectionPlan
  -> open under host authority
  -> ProtectionFlow

ProtectionBatch
  -> validate within plan ceilings
  -> OperationPlan
  -> execute through ProtectionFlow
  -> ProtectionRunRecord
```

The second chain is necessary because record bounds, deadlines, and approved
invocation context can vary between uses of one reusable flow. Anything that
changes prompts or protection semantics must be immutable, invocation-private
input bound to the operation plan. A receipt records only allowlisted public
identifiers and non-content verification claims; it never copies or hashes raw
summaries, caller context, or record content.

## Shared domain rules

### Protection is narrower than transformation

The new `protect` operation is protection-only. A release predicate is a
machine-checkable condition for marking output eligible under a named strategy
and profile; it is not proof of exhaustive PII detection. A plan may compile
only when it has a reviewed release predicate for that strategy and profile.
`Annotate`, an unsupported release predicate, or another non-releasing
configuration is rejected during compilation. The existing general-purpose
`run()` surface continues to support its established transformation behavior.

A successful protection outcome means that the configured strategy completed,
terminal accounting and accepted-detection integrity checks passed, and the
named release predicate passed. It does not mean that detection was exhaustive
or that the output contains zero PII.

### Closed terminal outcomes

Both plans target the same four exhaustive submission outcomes:

```python
RecordOutcome = (
    ProtectionSucceeded
    | Rejected
    | Failed
    | Cancelled
)
```

- `ProtectionSucceeded` contains the `RecordRef`, policy-qualified output, an
  allowlisted protection receipt, and exactly one `SuccessDisposition`:
  `ProtectionApplied` or `NoAcceptedDetections`. Success establishes execution
  and policy-postcondition completion, not release authority.
- `Rejected` means the submitted record was not accepted for execution.
- `Failed` means execution was accepted but produced no policy-qualified
  output.
- `Cancelled` means accepted work was observed to stop under the runtime's
  documented cancellation capability. Cleanup is reported independently as
  `CleanupComplete` or `CleanupIncomplete`.

`Rejected` is pre-execution; the other three variants are execution-terminal
for accepted work. Cancellation before admission is
`Rejected(reason=cancelled_before_admission)`. Only success contains output.
The current Plan A design spike does not implement accepted-work cancellation;
between-stage cancellation remains a requirement for later validation.
If the outer envelope is invalid, no records are admitted and the call returns
or raises one sanitized batch error. In every returned run record, each valid,
unique submitted `RecordRef` has exactly one outcome. A crash or transport loss
returns no run record, so the adapter must withhold the complete source item or
retry under its own policy. `RecordRef`, not output order, is authoritative.

`Omitted` is not an Anonymizer outcome. A source adapter or Relay maps a
rejected, failed, or cancelled outcome—and a transport-lost invocation—to its
own item-rejection or omission behavior. That preserves ownership of
complete-item atomicity and fail-closed publication.

### Safe failures

Expected failures are immutable domain data with:

- a closed code;
- coarse stage and scope;
- closed `unknown` retry safety and `Unassigned` retry ownership until taxonomy
  review assigns more specific semantics;
- an optional bounded static message.

They exclude raw input, protected output, caller references in rendered
messages, provider text, prompts, detections, workflow names, engine IDs,
tracebacks, exception causes, and arbitrary detail dictionaries. At the
proposed SDK boundary, unexpected defects must become
`AnonymizerWorkflowError` with a bounded static message and no arbitrary backend
cause. The branch-local public `run()` and `preview()` paths now implement that
mapping after leaving the active private exception handler, so the public error
has neither an accessible `__cause__` nor `__context__`. Preflight errors raised
before private row verification retain their established canonical types.

That mapping aborts result publication for the invocation. It does not roll
back provider calls or telemetry already attempted, and it does not define a
source adapter's commit unit. A successfully returned `AnonymizerResult` or
`PreviewResult` may contain `failed_records`; those are explicit NDD row drops,
not a partial result recovered from an exception. The caller owns whether such
a degraded result is publishable, withheld, or retried.

### Proposed private provenance

Plan A proposes a one-shot invocation-private verification
envelope:

```text
RecordRef outside dataframe
  -> random invocation row token
  -> private accepted-detection evidence
  -> verified terminal outcome
  -> private evidence cleared
```

`RecordRef`, adapter item or segment keys, invocation-private `RowToken`, and
`ReceiptId` remain distinct. `RecordRef` is opaque but may still contain PII;
Anonymizer bounds and echoes them only where required and never logs,
interprets, or sends them to a model.

Receipts may identify contract, policy, execution profile, implementation, and
verification claims. They do not contain raw-derived public hashes, detected
values, offsets, prompts, replacement maps, private correlations, or a general
`trace_dataframe`.

### External authority

Successful execution does not prove that the caller was authorized to provide
raw content or disclose output. The design distinguishes execution authority
(providers, credentials, files, and resources), data-handling authorization
(which actors may receive raw content), and release or commit authority (which
output may cross a selected boundary). An external product or deployment
authority grants and enforces those permissions. Anonymizer may record an
opaque `AuthorityReference`; it cannot mint, validate, or reinterpret one.

## Plan A: private Intake-first protection runtime

### Objective

Prove a useful vertical slice for Intake and Python jobs without committing to
a public record API or cross-runtime protocol.

### Private vocabulary

```text
_ProtectionSpec
_ProtectionPlan
_ProtectionFlow
_ProtectionRecord
_TextSegment
_OperationPlan
_ProtectionRunRecord
_ProtectionReceipt
```

`AnonymizerConfig` remains the compatibility input to the private intent.
Compilation copies and normalizes it, resolves model selections, validates a
static support matrix, attaches explicit limits, and produces an immutable,
content-free, non-serializable plan and digest.

The implementation may use a private runtime-local lowering. It is not the
portable flow plan and must never become a wire contract.

### Input and output

Both plans share one atomic record model:

```python
@dataclass(frozen=True, slots=True, repr=False)
class ProtectionRecord:
    ref: RecordRef
    segments: tuple[TextSegment, ...]
```

Plan A admits exactly one segment per record; Plan B may broaden supported
cardinality without changing outcome atomicity. The source adapter projects
each declared text leaf into a segment and retains
its complete source item, grouping, alias destinations, and reconstruction
manifest privately. A valid batch is bounded before dataframe construction by
record count, UTF-8 bytes per record, aggregate bytes, and admission deadline.

The batch result is a complete `ProtectionRunRecord`, not a convenience list of
successes. It contains ordered terminal outcomes, reviewed aggregate counts,
plan and runtime versions, attempt identity, `ProtectionReceipt` records, and
an `OperationReceipt`. It does not expose
dataframes or `FailedRecord`.

### Compilation and execution

Use separate names for pure compilation and effectful sampling:

- `compile` or `describe_plan` performs no network, filesystem, environment,
  logging, telemetry, model, or DataDesigner work.
- The existing `Anonymizer.preview()` remains an effectful compatibility
  operation.
- A future bounded SDK `preview` runs the same protection engine as `protect`
  but produces a noncommitting result type that an adapter cannot pass to its
  commit function.

Compilation returns a closed result: `PlanInvalid` for malformed or internally
contradictory specifications, `PlanUnsupported` for valid semantics absent
from declared capabilities, or `PlanRejected` for technically supported
semantics forbidden by the closed release policy. External authorization is
checked during effectful opening or admission, not compilation. Compilation
never silently falls back to a weaker
strategy, detector, evaluation mode, model, or release predicate.

### Runtime and lifecycle

`_ProtectionFlow` is an identity-bearing lifecycle boundary, not a frozen
value. It holds an Anonymizer-created runtime or an exclusively leased caller
runtime. Exclusivity is the caller's obligation. It must provide:

- idempotent `close()` and context-manager behavior;
- one whole-invocation admission guard;
- default capacity of one active invocation;
- explicit artifact, telemetry, logging, and provider policies;
- invocation-local usage accounting;
- no process-environment or host-logging mutation in the new private path;
- honest owned, borrowed, and partially cleanable resource status.

`close()` must stop new admission, use a documented drain-or-reject policy, and
release only demonstrably owned resources. Borrowed resources are never closed.
Dependency cleanup and owned-artifact cleanup are separate effects; unsupported
DataDesigner teardown must be reported as incomplete rather than implied.

Concurrent callers may wait only through a bounded host-selected admission
policy or receive a sanitized busy rejection. The flow does not maintain an
unbounded queue. Hosts scale with independently owned flows.

Plan A is synchronous. A caller may withdraw before admission. Between-stage
cancellation is an implementation requirement, not current behavior. An
in-flight DataDesigner or model call must not be described as hard-cancellable
unless the runtime can observe that execution stopped. Waiter cancellation must
not masquerade as operation cancellation.

### Intake composition

Intake compiles a separate adapter plan:

```text
Anonymizer protection policy -> compiled protection plan
Intake source mapping        -> compiled adapter plan
both                         -> Intake protection plan
```

The Intake plan records protection and mapping digests, declared item boundary,
semantic fidelity, and selected trust boundary. Its initial source policy
assigns each relevant field `target`, `preserve`, `structural`, or `reject`.
Content-bearing prompt context is deferred until its exposure and output
semantics receive separate review.

Under the provisional before-Intake assumption, an in-process seam after Intake
normalization is not a valid deployment placement. Protection, reconstruction,
and source-schema validation must complete at the producer or an authorized
edge component before the request crosses into the Intake service process.
The exact producer or edge component remains unselected, as do its deployment,
credential, retry, logging, and retention contracts.

### Plan A milestones

1. Freeze current compatibility behavior and record exact public schemas,
   errors, logging, environment mutation, artifacts, and concurrency behavior.
2. Define the private closed outcomes, safe failure taxonomy, acceptance point,
   release predicate, bounds, receipt claims, and one-shot verification
   envelope.
3. Add pure private compilation and a private runtime-local NDD lowering.
4. Add the private synchronous flow with whole-invocation admission and
   host-neutral logging, environment, telemetry, and artifact behavior.
5. Route private single-segment records through one dataframe and the existing
   engine;
   keep legacy `run()` and `preview()` compatible.
6. Validate ATIF v1.0 and v1.7 as boundary probes, then v1.1 through v1.6
   before claiming Intake's accepted v1.0–v1.7 range. Then validate chat
   completions and OTLP/HTTP protobuf through Intake-owned adapters. Treat
   direct spans as follow-on evidence.
7. Exercise reordering, drops, duplicates, unknown rows, provider failures,
   cancellation requests, overload, shutdown, artifact cleanup, resource soak,
   PII-canary diagnostics, reconstruction, and no-raw-fallback behavior.
8. Obtain independent architecture and privacy review. Keep the result private
   even if Plan A passes.

### Plan A exit gate

Plan A is complete when the consuming-product owner has selected the boundary
and a reviewed validation deployment dispatches approved synthetic or governed
workloads through the private path, every source item is reconstructed or
withheld according to its declared atomicity, resource behavior is bounded,
diagnostics remain content-free, and the reviewed provenance contract is
implemented. This does not satisfy the public shipment gate.

## Plan B: composable multi-host SDK

### Objective

Expose a small, agent-legible SDK language for Intake, Python jobs, a
process-backed Relay client, and eventually a native conforming implementation,
while preserving a single Python/DataDesigner execution path inside
Anonymizer. A future native conforming implementation remains a separately
owned runtime, not a second execution path inside Anonymizer.

### Public semantic layers

Candidate roles are:

```text
ProtectionSpec
ProtectionPlan
PlanPreview
CapabilityReport
ProtectionFlow
ProtectionRecord and TextSegment
ProtectionBatch
OperationPlan
ProtectionRunRecord
RecordOutcome
ProtectionReceipt, OperationReceipt, and ProjectionReceipt
BoundedRecordView
```

Names remain provisional. Public Pydantic or wire models parse once into a
closed immutable domain. Materializers lower the domain plan into a local
runtime plan. The Python materializer produces a private runtime-local lowering;
another runtime may produce a different local representation.

Plan B broadens the shared `ProtectionRecord` from exactly one `TextSegment` to
bounded multiple segments. A record remains the Anonymizer atomic outcome unit.
The source adapter still chooses
the source commit unit and performs source reconstruction. It may map one source
item to one record or withhold a larger item when any of several record outcomes
fails.

### Applicability algebra

Optional analysis and projections do not add nullable fields to terminal
outcomes. They use a separate closed algebra:

```python
type Projection[T] = (
    Available[T]
    | NotRequested
    | Inapplicable[InapplicableReason]
    | Unavailable[SafeFailure]
)
```

This distinguishes a value, an omitted request, a valid mode where the value
has no meaning, and a requested computation that failed. It is suitable for
evaluation, diagnostics, release assessments on general transformations, and
other companion views. It is not the terminal execution state. A mandatory
protection release check can never be `NotRequested` or `Inapplicable`; failure
to perform it prevents `ProtectionSucceeded`.

### Agent task surface

The fuller SDK exposes a small task language over the same semantic handlers:

- `compile`: create a complete immutable plan or a closed compile outcome.
- `preview`: run a bounded, noncommitting sample through the same executor.
- `protect`: execute a declared batch and return the bounded public run record.
- `explain`: explain plan resolution or an outcome from allowlisted evidence;
  it does not rerun protection.
- `inspect`: expose versions, effective policy, provenance, limits,
  capabilities, lifecycle, and bounded record structure.
- `diagnose`: classify a failure, retry safety, likely owner, and next checks;
  it does not retry or probe without separately authorized effects.

Each task returns structured records before prose rendering. Agent-facing views
are deterministic bounded projections over the bounded public record. They
declare included and omitted material, counts, continuation state, and content
disposition. Official reconstruction and commit functions reject these view
types. This cannot stop a holder from copying included content. Protected text
is excluded unless the projection layer receives an externally validated
disclosure scope.

### Capabilities and cross-runtime design

Static requirements and live readiness remain separate:

```text
compile(spec, declared capabilities)    # pure
open(plan, host authority)               # effect
check readiness                          # effect
```

Do not introduce an open `ProtectionEngine` protocol until two real semantic
implementations prove substitutability over an explicit capability subset.
Before that point, capabilities are closed versioned data consumed by concrete
implementations.

The cross-runtime design separates portable and local state:

```text
portable policy bundle
  -> runtime-local compilation
  -> opened local plan
  -> bounded batch execution
  -> exhaustive wire outcomes and receipts
```

The capability vocabulary covers operation, detection class, taxonomy,
language and platform qualification, single- or multi-segment support, context,
evaluation, verification, bounds, cancellation, artifact handling, provider
handling, and receipt versions.

Contract schema, policy schema, capability vocabulary, implementation, and
model versions remain distinct. Opening selects an exact compatible
contract and policy digest. Unknown required capabilities, incompatible major
versions, policy drift, or silent model substitution fail opening.

Wire encoding is not selected until Python and Rust implementations measure a
real boundary. Internal frozen unions do not double as Pydantic, protobuf, or
other wire types. An unknown wire outcome must fail closed and cannot be treated
as success.

### Flow and operation lifecycle

`ProtectionFlow` is the only public object that may own live resources. The
full design may add readiness, degradation, drain, rolling rotation, pools, and
local, borrowed, or remote ownership states.

A public operation handle is justified only when work can outlive a blocking
method and callers need observable state:

```text
PendingAdmission
Running
CancelRequested
Settled
Aborted
```

`CancelRequested` is not terminal. Cancelling an async waiter does not claim
the operation stopped. Reconnection belongs only to a durable service runtime;
an in-process Python handle cannot survive process loss.

Cancellation capability is negotiated explicitly: unsupported, between-stage,
cooperative in-flight, or process termination. Async waiting follows proven
operation semantics rather than preceding them.

### Relay realization

This proposal does not require Relay to embed Python. NeMo Relay exposes a
Rust-centered multi-language runtime with middleware, plugin, event, and
observability surfaces. Relay continues to own runtime, queue, subscriber,
export, and saturation lifecycle. Shared policy and conformance artifacts are a
future boundary blocked on accepted owners, repository, schema authority,
corpus governance, and release thresholds. If Relay later needs the full
Anonymizer engine, a supervised bounded local worker is one candidate, not a
selected implementation.

A Rust client to a Python worker validates hosting, IPC, crash isolation,
backpressure, version negotiation, and Relay lifecycle. It does not by itself
prove a second semantic protection implementation or satisfy that shipment
gate. An independently implemented native conforming runtime
justifies an open implementation protocol only when it consumes the same policy
bundle, advertises the same capability vocabulary, passes shared conformance
fixtures for its claimed subset, and produces the same outcome algebra.

Any Relay integration must map unresolved or failed selected content under its
reviewed omission policy. A
transport break produces no policy-qualified output and never permits raw fallback.

### Plan B milestones

1. Complete Plan A and preserve its four terminal variants and versioned
   discriminators.
2. Validate a process-backed production-intent host for transport and lifecycle
   evidence. Separately validate an independent native semantic implementation.
3. Compare at least two semantic implementations before extracting an open runtime or
   adapter protocol.
4. Freeze the smallest demonstrated policy, capability, record, outcome,
   receipt, and wire schemas.
5. Add bounded public run records and deterministic bounded views.
6. Add `explain`, `inspect`, and `diagnose` over those records without a second
   analysis engine.
7. Add operation handles and async waiting only where cancellation, ownership,
   and cleanup behavior are real and tested.
8. Review names, compatibility models, generated bindings, documentation, and
   the bundled Anonymizer skill before public export.

### Plan B exit gate

Plan B may ship publicly only after Intake and a materially different runtime
use the same reviewed semantic contract, the provenance and PII-boundary
records are accepted, resource and diagnostic behavior is bounded, and each
public capability is supported by real dispatch evidence rather than a stub or
test-only adapter.

## Comparison

| Concern | Plan A: private and simpler | Plan B: fuller and composable |
| --- | --- | --- |
| Immediate purpose | Validate Intake and Python jobs | Stable multi-host and agent-legible SDK |
| Public status | No new public protection API | Public only after all shipment gates |
| Input | One bounded text segment per record | The same record with bounded multiple segments |
| Operations | Pure compile; synchronous protect; private bounded preview | Compile, preview, protect, explain, inspect, diagnose |
| Plan | Frozen, content-free, local, non-serializable | Canonical portable semantics plus runtime-local activation |
| Runtime | One private Python flow, capacity one | Local, pooled, process-backed, or remote flows |
| Outcomes | Success, rejected, failed, cancelled | Same stable terminal core |
| Optional results | Deferred | Separate applicability/projection algebra |
| Provenance | Private envelope and allowlisted receipt | Protection, operation, projection, and authority-use receipts |
| Authority | Explicit non-claim | Reference to externally validated authority only |
| Capabilities | Static reviewed support matrix | Required/offered negotiation and versioning |
| Protocol | None | Extracted only after two substitutable implementations |
| Wire format | None | Selected after real Python and Rust measurements |
| Agent support | Structured tags, safe failures, retry ownership | Task-shaped operations and bounded views |
| Async | None | Waiting over explicit operation semantics only |
| Cancellation | Exact supported capability must be measured | Capability-negotiated cooperative or process cancellation |
| Main risk | Under-specifying release and provenance semantics | Over-modeling an unvalidated language |
| Latest bounded stop | Reviewed private Intake vertical slice | Reviewed public contract backed by two semantic runtimes |

Plan A is intentionally not a reduced version of every Plan B feature. It is a
complete narrow operation. Plan B grows through sibling algebras and adapters,
not by turning Plan A's fields into ambiguous option bags.

## Intake workload validation

The current-behavior column is established by the immutable Intake references
linked from each format; the proposed protection target is design work.

| Format | Current Intake behavior | Proposed protection target |
| --- | --- | --- |
| [ATIF](https://github.com/NVIDIA-NeMo/nemo-platform/blob/e1057736703bb8b167a4bd9013cea0caae2df63a/packages/nemo_platform_ext/src/nemo_platform_ext/skills/nemo-intake/references/ingest-formats.md#L32-L72) | Accepts v1.0–v1.7 | Probe v1.0 and v1.7 first, then validate every accepted version; protect and reconstruct complete trajectories while preserving hierarchy and ordering |
| [OpenAI-compatible chat completion](https://github.com/NVIDIA-NeMo/nemo-platform/blob/e1057736703bb8b167a4bd9013cea0caae2df63a/packages/nemo_platform_ext/src/nemo_platform_ext/skills/nemo-intake/references/ingest-formats.md#L80-L87) | Accepts extension-tolerant nested request and response models | Classify provider extensions and tool or content leaves under a closed policy while preserving pre-protection identities |
| [OTLP/HTTP protobuf](https://github.com/NVIDIA-NeMo/nemo-platform/blob/e1057736703bb8b167a4bd9013cea0caae2df63a/packages/nemo_platform_ext/src/nemo_platform_ext/skills/nemo-intake/references/ingest-formats.md#L120-L127) | Collects per-span errors and persists valid spans from a mixed-validity request | Resolve whether the source adapter preserves per-span acceptance or declares a stricter complete-request commit unit; use sanitized adapter-owned errors either way |
| [Provider-neutral direct spans](https://github.com/NVIDIA-NeMo/nemo-platform/blob/e1057736703bb8b167a4bd9013cea0caae2df63a/packages/nemo_platform_ext/src/nemo_platform_ext/skills/nemo-intake/references/ingest-formats.md#L91-L118) | Validates batches of 1–1,000 structured spans before conversion and write | Use a request-level protection unit without claiming transactional persistence atomicity |

Several formats in one Intake process remain one semantic runtime. They validate
shape and adapter semantics, not the two-semantic-implementation public gate.
OpenShell's OTLP/gRPC exporter is not directly equivalent to Intake's OTLP/HTTP
protobuf receiver; any collector bridge remains outside these plans.

The branch-local 2026-08-19 validation corpus exercises synthetic ATIF v1.0 and
v1.7, extension-bearing chat-completion JSON, real OTLP protobuf bytes, and an
Intake-shaped local CHAIN-to-LLM trace through the private Plan A flow and local
`Redact`. Its hermetic tests verify complete-item reconstruction, topology
preservation, closed field policy, and withholding on invalid spans or
non-success outcomes. They do not run provider-backed detection, durable
commit, or customer data, and they do not claim production format support.

An opt-in test-only dogfood also ran against a real local Intake service backed
by ClickHouse 26.3. Raw and protected ATIF v1.0, ATIF v1.7, chat-completion JSON,
and OTLP/protobuf round-tripped through public Intake routes. Protected read
models contained none of the declared synthetic PII and retained the asserted
topology and semantic fields. The OTLP fixture now preserves
`gen_ai.agent.name`, and Intake's exact `agent_name` filter returned the matching
protected LLM span.

A second opt-in path consumed a completed, successful Codex session from an
isolated Sandbox run. The real agent handled a synthetic prompt, created and
read a file, and emitted messages and shell-command output containing the
declared synthetic PII. A test-only, closed Sandbox adapter mapped the copied
prompt and reviewed completed-item types to ATIF v1.0. The private Plan A flow
protected that document before its HTTP request crossed into Intake; Intake's
read model exposed the protected trajectory with parent-child topology and none
of the declared values. The adapter rejects unreviewed completed-item types and
does not launch or manage Sandbox. Raw Sandbox artifacts remain on the producer
side of the provisional boundary. This run used deterministic local detection,
so it validates a real agent execution and the protection boundary, not
provider-backed detection quality or production Sandbox support.

The live test also established a material atomicity mismatch. For one request
with two valid spans and one invalid span, the test adapter rejected the whole
request and emitted no bytes. Intake accepted the same source request with HTTP
200, returned one per-span error, and persisted the two valid spans. This does
not violate the public `run()` or `preview()` contract: those APIs govern Python
result publication, while the source adapter and Intake own their respective
commit units. It does mean complete-request withholding is only a proposed
test-adapter policy and requires adopter review before an Intake integration
contract can be claimed.

The checked-in dogfood owns neither service nor storage lifecycle. It is skipped
unless `ANONYMIZER_INTAKE_DOGFOOD_BASE_URL` names an operator-owned deployment,
uses only synthetic unique identifiers, and does not provision, configure,
stop, or clean up Intake, ClickHouse, containers, or persisted rows. Its default
before-Intake profile sends only protected requests after checking that the
declared synthetic PII is absent from the outbound bytes. Sending raw fixtures
or reproducing Intake's mixed-validity OTLP partial write requires the separate
`ANONYMIZER_INTAKE_DOGFOOD_ALLOW_RAW=1` opt-in on an approved isolated
deployment; those cases characterize Intake and do not satisfy the assumed
deployment boundary. The protected-only failure probe wires invalid OTLP to a
real Intake emitter, verifies that adapter rejection prevents the emitter call,
and confirms that Intake stores no span for the withheld session.

Protected-delivery dogfood further separated protection success from transport
outcomes. A pre-connect failure produced a bounded, cause-free test error,
created no Intake row, and left the exact protected bytes available to the
test adapter for retry. The probe then sent that same byte string successfully
and observed one stable public row. A commit-then-forget approximation modeled
a request that may have committed before the caller could act on its response,
then sent the same protected bytes again. The test waited for stable read-model
snapshots after both deliveries. Intake's public read model collapsed the exact
ATIF and OTLP retries for these fixtures. The original chat-completion fixture
lacked a source timestamp, so its retry appeared as two rows with the same
external span ID because Intake assigned a new start time at each ingestion.

Adopter review resolved that duplicate chat rows are unacceptable. The
test-only chat contract therefore requires a positive, representable,
non-future integer `response.created`, preserves it unchanged through
protection, and rejects a chat item when Intake would fall back from that value.
Intake uses the accepted source value as the span start time. The live exact-byte
retry then collapsed to one public chat row. The dogfood assigns the timestamp
once when it constructs each unique synthetic capture so the record remains
inside Intake's default query window; it does not regenerate the value for a
retry.

This is format-specific read-model behavior, not evidence of transactional
idempotency, rollback, or one physical ClickHouse write. Anonymizer protects one
in-memory dataframe invocation and publishes one result or error; it does not
own delivery state, retry scheduling, or adopter idempotency keys. A source
adapter must retain protected bytes when retry is allowed, define stable
identity fields, classify retryable transport failures, and treat any duplicate
public row as a failed idempotency postcondition. Intake owns the persistence
and read-model semantics after it accepts a request.

## Proposed ownership and trust boundary

| Owner | Responsibility this SDK leaves outside Anonymizer core |
| --- | --- |
| Anonymizer | Python execution, protection-plan semantics, sanitized outcomes, and private verification; cross-runtime taxonomy and quality governance only if formally accepted |
| Source adapter | Codec, closed field roles, projection, aliases, manifest, fidelity, reconstruction, schema validation, output mapping |
| Intake | Authentication, staging, quotas, leases, replay, durable commit, retention, purge, delivery |
| Relay | Event selection, queue and drain, saturation, omission, activation, worker or plugin lifecycle, subscriber delivery |
| Host authority | Providers, credentials, endpoints, allowed paths, model provisioning, resource ceilings, audit sinks |
| Shared governance | Policy schema, conformance corpora, thresholds, support matrix, compliance-facing claims; owner remains unresolved |

The customer-required PII boundary is not yet confirmed. Until the
consuming-product owner decides otherwise, this report uses a before-Intake
working posture: only protected payloads and policy-approved preserved fields
may enter the Intake service process. Protection therefore runs at the producer
or an authorized edge component, not inside Intake or downstream of raw Intake
storage. The deployment record must still name every actor that can access raw
data before that boundary, including model providers, artifacts, logs, and
crash dumps. A receipt reports execution under a policy; it does not prove that
the deployment placed or enforced the boundary correctly.

A successful protection outcome attests only that the named implementation
completed the receipt's checks under the named policy and execution profile. It
does not establish exhaustive detection, absence of PII, authority to disclose
the output, or compliance with a deployment boundary. After the
consuming-product owner confirms or replaces the working boundary and accepts
residual risk, that deployment owns withholding every non-success outcome at
the selected boundary.

## Rejected shortcuts

- Public `trace_dataframe`, `include_internal=True`, or generic engine-row
  callbacks.
- Caller keys used as engine correlation.
- Public detected values, entity inventories, evidence quotes, or raw-derived
  hashes.
- A Pydantic result with a status string and mutually ambiguous optional
  output, error, receipt, and metrics fields.
- `ProtectedText = NewType(...)` or caller-constructible authorization markers
  presented as runtime proof.
- A provider, source-adapter, or runtime protocol with one implementation.
- A frozen flow that secretly owns mutable clients, threads, or cleanup duties.
- Generic protection of every string in arbitrary JSON.
- Default-open environment inheritance or agent-selectable raw diagnostics.
- Async wrappers that imply cancellation unsupported by the underlying work.
- Durable Intake or Relay lifecycle state inside Anonymizer.

## Decision gates

| Decision | Blocks Plan A implementation | Blocks Intake validation | Blocks Plan B public shipment | Decision authority |
| --- | ---: | ---: | ---: | --- |
| Earliest raw-PII boundary | No for private type work | Yes | Yes | Working assumption: before Intake; Intake team routes final approval to the consuming-product owner |
| Release predicate and first profile | Yes | Yes | Yes | Anonymizer interface owner and Intake team |
| Failure and retry taxonomy | Yes | Yes | Yes | Anonymizer interface owner plus adopter review |
| Resource limits and cancellation | Yes for the flow | Yes | Yes | Runtime owners |
| Policy, corpus, and compliance ownership | No | No | Yes | Unresolved |
| Second semantic implementation | No | No | Yes | Anonymizer and adopter architecture owners |

Further gated decisions include stable public projections, authenticated
cross-process receipt binding, and the capability subset of the second semantic
implementation. Unresolved owners are not inferred.

## Public implementation references

These immutable repository sources establish current implementation behavior;
the ownership split, shipment gates, provenance contract, trust boundary, and
companion plans remain proposals pending review.

- [Anonymizer facade and `run()` path](https://github.com/NVIDIA-NeMo/Anonymizer/blob/3eab7d1b6005b85e7d415b704e27a20dc41ba71e/src/anonymizer/interface/anonymizer.py#L151-L257)
- [Anonymizer DataDesigner execution boundary](https://github.com/NVIDIA-NeMo/Anonymizer/blob/3eab7d1b6005b85e7d415b704e27a20dc41ba71e/src/anonymizer/engine/ndd/adapter.py#L267-L328)
- [DataDesigner runtime interface](https://github.com/NVIDIA-NeMo/DataDesigner/blob/90c14379e78285be2baa4bc0f233eff8a3e1340e/packages/data-designer/src/data_designer/interface/data_designer.py)
- [Intake formats](https://github.com/NVIDIA-NeMo/nemo-platform/blob/e1057736703bb8b167a4bd9013cea0caae2df63a/packages/nemo_platform_ext/src/nemo_platform_ext/skills/nemo-intake/references/ingest-formats.md#L32-L161)
- [Intake normalized span model](https://github.com/NVIDIA-NeMo/nemo-platform/blob/e1057736703bb8b167a4bd9013cea0caae2df63a/services/intake/src/nmp/intake/spans/domain.py#L41-L61)
- [Intake ATIF normalization boundary](https://github.com/NVIDIA-NeMo/nemo-platform/blob/e1057736703bb8b167a4bd9013cea0caae2df63a/services/intake/src/nmp/intake/spans/ingest/atif.py#L100-L138)
- [Intake OTLP/HTTP receiver](https://github.com/NVIDIA-NeMo/nemo-platform/blob/e1057736703bb8b167a4bd9013cea0caae2df63a/services/intake/src/nmp/intake/spans/ingest/otlp.py#L57-L122)
- [Intake chat-completion normalization](https://github.com/NVIDIA-NeMo/nemo-platform/blob/e1057736703bb8b167a4bd9013cea0caae2df63a/services/intake/src/nmp/intake/spans/ingest/chat_completions.py#L143-L163)
- [NeMo Relay runtime and observability surfaces](https://github.com/NVIDIA/NeMo-Relay/blob/c37b551b98f0d3e890c32503bb2edf69445ad3c4/README.md)
- [OpenShell OTLP/gRPC export](https://github.com/NVIDIA/OpenShell/blob/600bbae845f96c3ef94222c5531965227c65dfcc/docs/reference/gateway-config.mdx#L213-L258)
- [OpenShell OCSF inference events](https://github.com/NVIDIA/OpenShell/blob/600bbae845f96c3ef94222c5531965227c65dfcc/docs/observability/ocsf-json-export.mdx#L145-L169)

The branch-local error mapping and structured projection prototype are not
public evidence until their commits are reachable in an online Git repository.
Unpublished ownership records and design guides informed this proposal but are
neither normative dependencies nor cited evidence.
