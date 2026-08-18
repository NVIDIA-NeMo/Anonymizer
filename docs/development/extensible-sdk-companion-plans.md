# Extensible SDK companion plans

Status: Proposed design report, 2026-08-18. This document does not authorize a
public API, a production Intake integration, a Relay integration, or a release
claim. Names are provisional.

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

Both plans preserve these invariants:

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
- Public API shipment remains gated on two materially different validated
  runtimes and reviewed provenance and PII-boundary decisions.
- Current OpenShell telemetry is test evidence, not the second runtime and not
  authorization to change OpenShell.

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
effectful protection runner
        |
        v
complete safe run record
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
ProtectionIntent
  -> compile against declared capabilities
  -> ProtectionPlan
  -> open under host authority
  -> ProtectionRunner

ProtectionRequest
  -> validate and lower
  -> OperationPlan
  -> execute through ProtectionRunner
  -> ProtectionRunRecord
```

The second chain is necessary because record bounds, data summaries, deadlines,
and approved invocation context can vary between uses of one reusable runner.
Anything that changes prompts or protection semantics must appear in the
operation plan and receipt rather than ambient mutable state.

## Shared domain rules

### Protection is narrower than transformation

The new `protect` operation is protection-only. A plan may compile only when it
has a reviewed release predicate for the chosen strategy and profile.
`Annotate`, an unsupported release predicate, or another non-releasing
configuration is rejected during compilation. The existing general-purpose
`run()` surface continues to support its established transformation behavior.

A successful protection outcome means that the configured strategy completed,
terminal accounting and accepted-detection integrity checks passed, and the
named release predicate passed. It does not mean that detection was exhaustive
or that the output contains zero PII.

### Closed terminal outcomes

Both plans use the same four terminal alternatives:

```python
RecordOutcome = (
    ProtectionSucceeded
    | Rejected
    | Failed
    | Cancelled
)
```

- `ProtectionSucceeded` contains the caller reference, releasable output, and
  a content-safe protection receipt.
- `Rejected` means the submitted record was not accepted for execution.
- `Failed` means execution was accepted but produced no verified releasable
  output.
- `Cancelled` means accepted work stopped under the runtime's documented
  cancellation capability.

Only success contains output. Each submitted reference in a valid batch gets
exactly one outcome; each accepted reference gets exactly one execution
terminal state. Caller references, not output order, are authoritative.

`Omitted` is not an Anonymizer outcome. A source adapter or Relay maps a
rejected, failed, cancelled, missing, or transport-lost result to its own item
rejection or omission behavior. That preserves ownership of complete-item
atomicity and fail-closed publication.

### Safe failures

Expected failures are immutable domain data with:

- a closed code;
- coarse stage and scope;
- retry safety and retry owner;
- an optional bounded static message.

They exclude raw input, protected output, caller references in rendered
messages, provider text, prompts, detections, workflow names, engine IDs,
tracebacks, exception causes, and arbitrary detail dictionaries. Unexpected
SDK defects raise the canonical safe public exception.

### Private provenance

The first provenance mechanism is a one-shot invocation-private verification
envelope:

```text
caller reference outside dataframe
  -> random invocation row token
  -> private accepted-detection evidence
  -> verified terminal outcome
  -> private evidence cleared
```

Caller references, adapter item or segment keys, engine row tokens, and receipt
IDs remain distinct. Caller references are opaque but may still contain PII;
Anonymizer bounds and echoes them only where required and never logs,
interprets, or sends them to a model.

Receipts may identify contract, policy, execution profile, implementation, and
verification claims. They do not contain raw-derived public hashes, detected
values, offsets, prompts, replacement maps, private correlations, or a general
`trace_dataframe`.

### External authority

Successful execution does not prove that the caller was authorized to provide
raw content. Processing authority originates in Intake, Relay, the host, or a
deployment control plane. Plan A makes this an explicit non-claim. Plan B may
refer to an externally validated authority-use receipt, but Anonymizer cannot
mint authority or let portable configuration widen it.

## Plan A: private Intake-first protection runtime

### Objective

Prove a useful vertical slice for Intake and Python jobs without committing to
a public record API or cross-runtime protocol.

### Private vocabulary

```text
_ProtectionIntent
_ProtectionPlan
_ProtectionRunner
_TextRecord
_OperationPlan
_ProtectionRunRecord
_ProtectionReceipt
```

`AnonymizerConfig` remains the compatibility input to the private intent.
Compilation copies and normalizes it, resolves model selections, validates a
static support matrix, attaches explicit limits, and produces an immutable,
content-free, non-serializable plan and digest.

The current `_CompiledInvocation` remains the lower, NDD-specific operation
description. It is not the portable flow plan and must not be serialized as a
wire contract.

### Input and output

Plan A accepts a finite sequence of scalar records:

```python
@dataclass(frozen=True, slots=True, repr=False)
class TextRecord:
    ref: RecordRef
    text: str
```

The source adapter projects each declared text leaf into one record and retains
its complete source item, grouping, alias destinations, and reconstruction
manifest privately. A valid batch is bounded before dataframe construction by
record count, UTF-8 bytes per record, aggregate bytes, and admission deadline.

The batch result is a complete `ProtectionRunRecord`, not a convenience list of
successes. It contains ordered terminal outcomes, safe counts, plan and runtime
versions, attempt identity, and coarse effect receipts. It does not expose
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

Compilation returns a closed result such as `PlanReady`, `PlanInvalid`,
`PlanUnsupported`, or `PlanRejected`. It never silently falls back to a weaker
strategy, detector, evaluation mode, model, or release predicate.

### Runtime and lifecycle

`ProtectionRunner` is an identity-bearing resource owner, not a frozen value.
It owns or exclusively borrows one DataDesigner-backed runtime and provides:

- explicit `close()` and context-manager behavior;
- one whole-invocation admission guard;
- default capacity of one active invocation;
- explicit artifact, telemetry, logging, and provider policies;
- invocation-local usage accounting;
- no process environment or host logging mutation;
- honest owned, borrowed, and partially cleanable resource status.

Concurrent callers may wait only through a bounded host-selected admission
policy or receive a safe busy rejection. The runner does not maintain an
unbounded queue. Hosts scale with independently owned runners.

Plan A is synchronous. It supports cancellation before admission and checks
between stages. It does not claim hard cancellation of an in-flight
DataDesigner or model call. `Cancelled` is terminal only after work stops and
cleanup completes.

### Intake composition

Intake compiles a separate adapter plan:

```text
Anonymizer protection policy -> compiled protection plan
Intake source mapping        -> compiled adapter plan
both                         -> Intake protection plan
```

The Intake plan records protection and mapping digests, declared item boundary,
semantic fidelity, and selected trust boundary. Its source policy assigns each
relevant field `target`, `context`, `preserve`, `structural`, or `reject`.

The recommended in-process placement is after existing Intake normalization
and identity derivation but immediately before `ingest_batch()`. This preserves
current trace and span identity while supporting only the posture in which raw
content may exist in Intake memory but not cross the selected persistence or
read boundary. A requirement to protect before Intake needs the adapter at a
producer or edge process.

### Plan A milestones

1. Freeze current compatibility behavior and record exact public schemas,
   errors, logging, environment mutation, artifacts, and concurrency behavior.
2. Define the private closed outcomes, safe failure taxonomy, acceptance point,
   release predicate, bounds, receipt claims, and one-shot verification
   envelope.
3. Add pure private compilation and retain `_CompiledInvocation` as the
   NDD-specific lowering.
4. Add the private synchronous runner with whole-invocation admission and
   host-neutral logging, environment, telemetry, and artifact behavior.
5. Route private scalar records through one dataframe and the existing engine;
   keep legacy `run()` and `preview()` compatible.
6. Validate ATIF v1.0 and v1.7 first, then chat completions and OTLP/HTTP
   protobuf through Intake-owned adapters. Treat direct spans as follow-on
   evidence.
7. Exercise reordering, drops, duplicates, unknown rows, provider failures,
   cancellation requests, overload, shutdown, artifact cleanup, resource soak,
   PII-canary diagnostics, reconstruction, and no-raw-fallback behavior.
8. Obtain independent architecture and privacy review. Keep the result private
   even if Plan A passes.

### Plan A exit gate

Plan A is complete when Intake dispatches representative workloads through the
private path, every source item is reconstructed or withheld according to its
declared atomicity, resource behavior is bounded, diagnostics remain
content-free, and the reviewed provenance contract is implemented. This does
not satisfy the public shipment gate.

## Plan B: composable multi-host SDK

### Objective

Expose a small, agent-legible SDK language for Intake, Python jobs, a
process-backed Relay client, and eventually a native conforming implementation,
without creating a second semantic engine.

### Public semantic layers

Candidate roles are:

```text
ProtectionIntent
ProtectionPlan
PlanPreview
CapabilityReport
ProtectionFlow
Record and Scalar
ProtectionRequest
OperationPlan
ProtectionRunRecord
RecordOutcome
EffectReceipt
BoundedRecordView
```

Names remain provisional. Public Pydantic or wire models parse once into a
closed immutable domain. Materializers lower the domain plan into a local
runtime plan. The Python materializer produces `_CompiledInvocation`; another
runtime may produce a different local representation.

Plan B adds bounded multi-scalar records as a sibling to Plan A's scalar
operation rather than silently changing `TextRecord`:

```text
Record
  ref
  tuple[TextScalar, ...]
```

A record is the Anonymizer atomic outcome unit. The source adapter still chooses
the source commit unit and performs source reconstruction. It may map one source
item to one record or withhold a larger item when any of several record outcomes
fails.

### Applicability algebra

Optional analysis and projections do not add nullable fields to terminal
outcomes. They use a separate closed algebra:

```python
Projection[T] = Available[T] | NotRequested | Inapplicable | Unavailable
```

This distinguishes a value, an omitted request, a valid mode where the value
has no meaning, and a requested computation that failed. It is suitable for
evaluation, diagnostics, release assessments on general transformations, and
other companion views. It is not the terminal execution state.

### Agent task surface

The fuller SDK exposes a small task language over the same semantic handlers:

- `compile`: create a complete immutable plan or a closed compile outcome.
- `preview`: run a bounded, noncommitting sample through the same executor.
- `protect`: execute a declared batch and return the complete safe run record.
- `explain`: explain plan resolution or an outcome from existing safe evidence;
  it does not rerun protection.
- `inspect`: expose versions, effective policy, provenance, limits,
  capabilities, lifecycle, and safe record structure.
- `diagnose`: classify a failure, retry safety, likely owner, and next checks;
  it does not retry or probe without separately authorized effects.

Each task returns structured records before prose rendering. Agent-facing views
are deterministic bounded projections over the complete safe record. They
declare included and omitted material, counts, continuation state, and content
disposition. They cannot be used for reconstruction or commit, and protected
text is excluded by default unless caller authority and policy permit it.

### Capabilities and cross-runtime design

Static requirements and live readiness remain separate:

```text
compile(intent, declared capabilities)  # pure
negotiate(plan, runtime snapshot)        # pure
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
  -> activated local plan
  -> bounded batch execution
  -> exhaustive wire outcomes and receipts
```

The capability vocabulary covers operation, detection class, taxonomy,
language and platform qualification, scalar or multi-scalar support, context,
evaluation, verification, bounds, cancellation, artifact handling, provider
handling, and receipt versions.

Contract schema, policy schema, capability vocabulary, implementation, and
model versions remain distinct. Activation selects an exact compatible
contract and policy digest. Unknown required capabilities, incompatible major
versions, policy drift, or silent model substitution fail activation.

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
Succeeded
Failed
Cancelled
```

`CancelRequested` is not terminal. Cancelling an async waiter does not claim
the operation stopped. Reconnection belongs only to a durable service runtime;
an in-process Python handle cannot survive process loss.

Cancellation capability is negotiated explicitly: unsupported, between-stage,
cooperative in-flight, or process termination. Async waiting follows proven
operation semantics rather than preceding them.

### Relay realization

Relay should not embed Python. The current path remains Relay-owned native
Rampart consuming shared policy and conformance artifacts. If Relay later needs
the full Anonymizer engine, the preferred host shape is a supervised local
worker with bounded IPC.

A Rust client to a Python worker validates transport, crash isolation,
backpressure, version negotiation, and Relay lifecycle. It does not by itself
prove a second semantic protection implementation. A native conforming runtime
justifies an open implementation protocol only when it consumes the same policy
bundle, advertises the same capability vocabulary, passes shared conformance
fixtures for its claimed subset, and produces the same outcome algebra.

Relay continues to map unresolved or failed selected content to omission. A
transport break produces no safe output and never permits raw fallback.

### Plan B milestones

1. Complete Plan A and preserve its four terminal variants and versioned
   discriminators.
2. Validate a process-backed or native production-intent runtime with bounded
   admission, crash, timeout, malformed response, version drift, shutdown, and
   no-raw-fallback tests.
3. Compare at least two implementations before extracting an open runtime or
   adapter protocol.
4. Freeze the smallest demonstrated policy, capability, record, outcome,
   receipt, and wire schemas.
5. Add full safe run records and deterministic bounded views.
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
| Public status | No new public API | Public only after all shipment gates |
| Input | One bounded text scalar per record | Bounded multi-scalar records plus scalar compatibility |
| Operations | Pure compile; synchronous protect; private bounded preview | Compile, preview, protect, explain, inspect, diagnose |
| Plan | Frozen, content-free, local, non-serializable | Canonical portable semantics plus runtime-local activation |
| Runtime | One concrete Python runner, capacity one | Local, pooled, process-backed, or remote flows |
| Outcomes | Success, rejected, failed, cancelled | Same stable terminal core |
| Optional results | Deferred | Separate applicability/projection algebra |
| Provenance | Private envelope and minimal receipt | Safe plan, effect, projection, and authority-use receipts |
| Authority | Explicit non-claim | Reference to externally validated authority only |
| Capabilities | Static reviewed support matrix | Required/offered negotiation and versioning |
| Protocol | None | Extracted only after two substitutable implementations |
| Wire format | None | Selected after real Python and Rust measurements |
| Agent support | Structured tags, safe failures, retry ownership | Task-shaped operations and bounded views |
| Async | None | Waiting over explicit operation semantics only |
| Cancellation | Before admission and between stages | Capability-negotiated cooperative or process cancellation |
| Main risk | Under-specifying release and provenance semantics | Over-modeling an unvalidated language |
| Latest safe stop | Reviewed private Intake vertical slice | Reviewed public contract backed by two runtimes |

Plan A is intentionally not a reduced version of every Plan B feature. It is a
complete narrow operation. Plan B grows through sibling algebras and adapters,
not by turning Plan A's fields into ambiguous option bags.

## Intake workload validation

| Format | Initial role | Contract pressure |
| --- | --- | --- |
| ATIF v1.0 and v1.7 | First complete adapter validation | Recursive item atomicity, nested text, aliases, hierarchy, ordering |
| OpenAI-compatible chat completion | Second initial adapter | Simple item path, tool calls, provider-extension rejection, identity preservation |
| OTLP/HTTP protobuf | Third initial adapter | Per-span partial acceptance, semantic precedence, sanitized failures |
| Provider-neutral direct spans | Follow-on evidence | Open-ended source profiles, arbitrary JSON attributes, request atomicity |

Several formats in one Intake process remain one runtime. They validate shape
and adapter semantics, not the two-runtime public gate.

## Ownership and trust boundary

| Owner | Responsibility |
| --- | --- |
| Anonymizer | Protection intent and plan semantics, Python execution, safe outcomes, private verification, taxonomy, quality methods |
| Source adapter | Codec, closed field roles, projection, aliases, manifest, fidelity, reconstruction, schema validation, output mapping |
| Intake | Authentication, staging, quotas, leases, replay, durable commit, retention, purge, delivery |
| Relay | Event selection, queue and drain, saturation, omission, activation, worker or plugin lifecycle, subscriber delivery |
| Host authority | Providers, credentials, endpoints, allowed paths, model provisioning, resource ceilings, audit sinks |
| Shared governance | Policy schema, conformance corpora, thresholds, support matrix, compliance-facing claims; owner remains unresolved |

The customer-required PII boundary remains unknown. The same SDK may run at the
source or edge, inside Intake before persistence, or downstream of controlled
raw storage. The deployment record must name every raw-data actor, including
model providers, artifacts, logs, and crash dumps. A receipt reports execution
under a policy; it does not assert where the deployment boundary sits.

An absolute zero-PII guarantee is not supported. The enforceable claim is that
no content violating a reviewed release predicate under a named closed policy
crosses the selected boundary, subject to documented residual risk and
unsupported-content behavior.

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

## Open decisions

1. The customer-selected earliest boundary that unprotected PII may cross and
   the accepted residual-risk definition.
2. The reviewed release predicates and first supported protection profiles.
3. The final safe failure codes, stage vocabulary, retry classification, and
   retry owners.
4. Canonical byte, memory, call, token, repair, wall-time, and artifact limits.
5. Deterministic DataDesigner teardown and the exact cancellation capability by
   supported version.
6. Which metrics are stable enough for safe public projections.
7. Whether a cross-process receipt needs authenticated-channel binding or a
   durable signature.
8. The named owner or co-owners for portable policy artifacts, qualification
   corpora, and compliance-facing claims.
9. Which production-intent runtime supplies the second validation and which
   narrower capability subset it implements.

## Evidence and authority

This report follows the current Anonymizer and Rampart ownership decision and
the streaming continuation plan. It incorporates the current Intake main
behavior, the current OpenShell trace-export evidence, the reviewed canonical
public error mapping, the test-only structured projection prototype, the
Scala-like Python type-safety guide, and the agent-accessible inference SDK
design note.

The ownership decision remains authoritative. Its historical branch snapshot
does not override the current Anonymizer branch state. Work Packet 1 remains in
progress until the opaque provenance mechanism and PII boundary receive the
required reviews.
