<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Graph-native Anonymizer SDK

*AIRE Engineering RFC*

Author(s): TBD

Status: Under Review — acceptance of the complete RFC plan requested

Category: Architecture / SDK

Draft Date: 2026-08-20

Review Date: In progress

Target Closing Date: Not set

Implementation: https://github.com/NVIDIA-NeMo/Anonymizer/pull/253

Implementation baseline: `codex/anonymizer-openshell-intake` at `29bddad51fdc879c5e5c677857c1d2561f4528ee`. The review candidate includes this RFC, the phase designs, and the private Phases 1–6 source and tests. Phase 7 implementation remains unauthorized. This tab is the review mirror.

## Decision Requested

Accept, request revisions to, or reject this RFC as one development and research plan for the graph-native Anonymizer SDK. The decision covers the proposed semantic architecture, the ordered branch-development phases, the strongly typed graph SDK direction, and the separate performance and experimentation program.

Acceptance records the project decision on the complete plan. It permits iterative implementation, experiments, and evidence gathering on this development branch, subject to the phase order, prerequisites, and operator checkpoints in the plan. It does not approve the proposed public API for publication, authorize production Intake or OpenShell integration, establish a product SLO or deployment profile, permit stable promotion, or make a privacy-boundary or “zero PII” claim.

Phase checkpoints are branch execution controls, not separate project acceptance decisions. A checkpoint records that the operator has authorized the next bounded implementation or research step on this branch. Evidence review determines whether its prerequisites have passed; later product, public-API, adopter, customer, and release gates remain with the owners named below.

* Phase 4: the authorized branch-local implementation and its evidence gates completed on 2026-08-25; RFC acceptance, public-API approval, production integration, and promotion remain pending their separate decisions.
* Phase 5: the private branch-local implementation and hardening landed on 2026-08-26 in `bb79cda` through `53ef74f`; its frozen reference model and focused context, lifecycle, privacy, and compatibility evidence are present on the branch.
* Phase 6: the private branch-local implementation and hardening landed on 2026-08-27 in `5bf61c6` through `29bddad`; its frozen reference model and focused mention, resolution, role, Redact, lifecycle, privacy, and compatibility evidence are present on the branch.
* Phase 7: reviewed design only; branch implementation authorization remains pending and sequenced after Phases 4–6 and its versioned semantic and execution contract.

## Revision History

* 0.1 — 2026-08-20 — Recorded the graph-native proposal, branch-local Phases 1–3, and the ordered migration.
* 0.2 — 2026-08-20 — Added independently reviewed Phase 5 and Phase 6 designs, synchronized Phase 4 and Phase 7 acceptance status, and reorganized the review mirror as a complete RFC.
* 0.3 — 2026-08-21 — Proposed the strongly typed graph SDK, added benchmark and trace examples, and moved product authorization and field policy outside Anonymizer.
* 0.4 — 2026-08-21 — Defined preflight, private work-ID terminology, content-free lifecycle observations, cleanup outcomes, and the retention-capability boundary.
* 0.5 — 2026-08-21 — Made acceptance apply to the complete RFC plan, distinguished project acceptance from branch execution checkpoints, and recorded implementation and test status.
* 0.6 — 2026-08-25 — Recorded the completed private Phase 4 implementation, frozen conformance corpus, repository verification, and remediation-council closeout without changing RFC or later-phase authority.
* 0.7 — 2026-08-25 — Recorded authenticated Arc acceptance, bounded rejection of non-UTF-8 datum values, and the explicit per-datum stage-predecessor contract without changing later-phase or publication authority.
* 0.8 — 2026-08-31 — Reconciled Phase 5 and Phase 6 status with the private branch implementation, reference-model, test, and CI evidence without changing public, production, Phase 7, or promotion authority.

## Review History

* Phase 5 architecture and test-strategy councils — Complete — 2026-08-21 focused re-review closed with zero unresolved Critical or Warning findings.
* Phase 6 architecture and test-strategy councils — Complete — 2026-08-20 — Zero unresolved Critical or Warning findings.
* Phase 4 implementation-remediation council — Complete — 2026-08-25 — All nine remediation claims independently verified; zero unresolved material findings.
* Phase 4 authenticated review-only Arc — Complete — 2026-08-25 — Reviewer accepted the focused remediation with zero findings; formatting, type, full repository, and strict documentation validations passed.
* Project and Anonymizer semantic owner — Pending — Acceptance of the complete RFC development and research plan requested.
* Additional adopter, customer, public-API, and runtime approvals remain scoped to the gates that name them.

## Development Status

PR 253 is an active development and research PR. Its protected branch contains the private Phases 1–6 implementation plus the evolving RFC and phase designs; it is not an implementation of the complete RFC and does not expose a public graph SDK.

| Phase | Branch state | Verification state | Next branch checkpoint |
| --- | --- | --- | --- |
| 1–3 | Implemented privately | 38 focused graph and private-protection tests passed; the 2026-08-20 repository run reported 1,408 passed and 11 opt-in Intake tests skipped | Preserve compatibility while later phases replace the internal system of record |
| 4 | Implemented privately under its authorized branch checkpoint | Frozen `phase4-stream-v4` corpus passed 397,542 canonical traces over 8,278 admitted graphs; focused accounting, race, process-loss, privacy, compatibility, formatting, type, documentation, and full repository checks passed; remediation council closed with zero unresolved material findings | Preserve the verified boundary; later adoption, publication, and promotion require their separate gates |
| 5 | Implemented privately and hardened | Frozen Phase 5 reference-model evidence and focused context admission, execution, reconciliation, cleanup, privacy, and public-compatibility tests are present; 2026-08-31 PR checks pass | Preserve the qualified private boundary; publication and production use require separate gates |
| 6 | Implemented privately and hardened | Frozen Phase 6 reference-model evidence and focused mention, resolution, role-policy, Redact, backend, lifecycle, privacy, and public-compatibility tests are present; 2026-08-31 PR checks pass | Preserve the qualified private Redact boundary; Phase 7 requires its own contract and authorization |
| 7 | Designed and independently reviewed; implementation authorization pending | Test strategy reviewed, but implementation evidence does not yet exist | Freeze the Phase 7 semantic and execution contract and obtain separate authorization |
| 8–11 | RFC plan only | No phase implementation evidence | Refine and authorize bounded branch checkpoints in order |

For phases that remain proposals, “test strategy reviewed” means reviewers found the proposed evidence plan sufficient to begin the corresponding branch work when authorized. It does not mean that phase has been implemented or its tests have passed.

## Problem

The current Anonymizer API treats a DataFrame row as the practical unit of protection and accepts source-formatted data only after callers flatten it into tabular text. That boundary works for independent records, but it cannot faithfully express related-record workloads in which context, replacement coherence, dependencies, and atomic release differ.

External workloads such as trace trees, trajectories, and structured request/response records need source-neutral protection semantics without moving codecs, source identity, persistence, retries, deduplication, reconstruction, cleanup, or delivery into Anonymizer. Unsupported relationships must fail closed; silently treating related datums as independent rows would erase the property the integration needs to preserve.

## Background

Published Anonymizer exposes DataFrame-oriented `run()`, `preview()`, and `evaluate()` paths. `NddAdapter.run_workflow()` is the engine boundary for executing DataDesigner workflows. The migration must preserve those public contracts while replacing the row-local internal system of record.

Private branch-local Phases 1–3 define immutable, non-serializable graph values, validate a trivial independent-datum profile, lower it through the existing pandas runtime, hydrate graph outcomes, and map them back through the private compatibility flow. Phase 4 extends that seam with explicit dependency DAGs, flat exact atomic partitions, and exhaustive terminal accounting. Phase 5 adds bounded target/context workframes, and Phase 6 adds anchored mentions, explicit-evidence resolution, versioned role results, and exact local Redact verification. Coherence planning and links remain unsupported and fail closed.

Verification on 2026-08-20 reported 1,408 passed and 11 skipped tests with one warning. The focused graph and private-protection suites passed 38 tests. The 11 opt-in Intake dogfood tests skipped because their external environment was not enabled; historical operator runs remain bounded observations, not product guarantees.

Branch-local Phase 4 verification on 2026-08-25 reported 1,508 passed and 11 skipped tests with one pre-existing deprecation warning. The frozen `phase4-stream-v4` corpus passed 397,542 canonical traces over 8,278 admitted graphs with manifest digest `e778147bf77909ddb94117fe7e6c230de57e46a722fad49c563b36f0b5660efa`. Formatting, type checking, documentation, privacy-canary, compatibility, fault, concurrency, and mutation checks passed. An independent remediation verifier closed all nine council findings with no remaining material findings, and the maintained authenticated review-only Arc accepted the focused remediation with zero findings.

Admission now converts non-UTF-8 datum IDs and text into the bounded `malformed_graph` rejection rather than allowing an encoding exception to escape. Synthetic multi-stage plans use a fixed per-datum earlier-stage predecessor, matching the independent reference model; a stage result still closes only after all of its tasks are terminal. This is an accounting contract, not authorization for later semantic stages.

The separate [Intake workload evidence](intake-workload-validation-evidence.md) records ATIF, OTLP, and chat-completion ingestion findings. Those formats demonstrate different hierarchy, structure, and partial-acceptance pressures, but they remain one Intake runtime and do not satisfy the independent-semantic-runtime gate.

## Goals

* Make an immutable ProtectionGraph the private semantic system of record and define a small strongly typed public authoring contract for review.
* Represent context, replacement coherence, dependencies, and atomic release independently.
* Preserve the current DataFrame API and DataDesigner execution boundary during migration.
* Reconcile backend work through opaque invocation-local identity rather than source IDs, content, indexes, or row position.
* Fail closed when requested semantics, backend compatibility, attribution, terminal evidence, or release verification are incomplete.
* Keep source codecs and durable operational lifecycle responsibilities downstream.
* Qualify each semantic capability through an independent reference model, bounded conformance envelope, mutation tests, and privacy checks.

### Non-Goals

* Implement or publish the proposed graph API before its public-API and implementation-evidence gates pass.
* Claim exhaustive detection, absence of all PII, or an approved “before Intake” boundary.
* Move source codecs, platform jobs, storage, durable retries, deduplication, retention, purge, reconstruction, or delivery into Anonymizer.
* Treat multiple Intake formats, a process-backed Python host, test adapters, or OpenShell telemetry as a materially different semantic runtime.
* Permit unsupported related-record semantics to degrade to independent rows.
* Authorize production Intake or OpenShell changes, stable promotion, or revisions to “samples.”

## Terminology and Definitions

* Atomic group — A flat, exact set of target datums whose qualified outputs are released or withheld together.
* Coherence scope — The set within which aliases and replacement assignments must remain consistent.
* Context scope — The ordered, explicitly declared set of related datums that may inform one target decision.
* Datum — An immutable source-neutral text value with graph-scoped identity.
* Entity cluster — A deterministic set of accepted mentions joined only by explicit same-subject evidence.
* Mention — A detected entity anchored to an exact half-open character interval in one authoritative target datum.
* NDD — NVIDIA DataDesigner, used for LLM column generation through `NddAdapter.run_workflow()`.
* ProtectionGraph — The immutable semantic input containing source-neutral datums and reviewed relationships.
* PreparedProtection — The immutable process-local result of successful graph admission; the only value accepted by public `protect()`.
* ProtectionResult — The immutable mapping from every admitted target key to one closed terminal target outcome.
* Replacement role — A versioned classification of a mention’s replacement function; distinct from detector label, cluster identity, and replacement slot.
* Workframe — A temporary bounded DataFrame projection used to execute one graph stage; not a semantic output unit.
* Bounded invocation — One finite request whose datums, relationships, limits, and selected protection contract are known at compilation.
* Inline integration — A future integration pattern in which an existing system submits a bounded invocation and waits for qualified output or a typed non-success outcome.
* Bounded microbatch — A finite host-formed execution group used to explore throughput and queueing trade-offs; it does not define graph identity, context, atomic groups, or release units.
* Streaming transport — A long-lived or unbounded flow with ordering, backpressure, checkpointing, delivery, retry, recovery, and partial-result semantics. This RFC does not propose one.
* Execution profile — A closed, versioned internal selection of semantic tasks, authorized implementations, schemas, limits, conditional routes, failure behavior, release predicate, and release claim, fixed before invocation effects.
* Deployment form — In-process SDK, service, container, or hosted packaging. Deployment form is not a protection semantic and remains separately governed.

## Requirements

### REQ 1 Preserve Public Compatibility

The migration MUST preserve current public constructors, configuration types, signatures, defaults, result columns and attributes, failed\_records shape, trace\_dataframe behavior, `evaluate()` behavior, CLI behavior, and canonical errors unless a separate public-API review approves a change.

### REQ 2 Use the Graph as the Semantic Record

The graph MUST remain authoritative before lowering and after hydration. DataFrame rows, batches, provider responses, and context fragments MUST NOT become semantic identity or release units.

### REQ 3 Preserve the DataDesigner Boundary

Every DataDesigner workflow MUST execute through `NddAdapter.run_workflow()`. Graph stages MAY declare DataDesigner column configurations and use temporary DataFrames, but MUST NOT call DataDesigner execution APIs directly.

### REQ 4 Keep Scopes Independent

Context scope, coherence scope, dependency, and atomic group MUST remain separate declarations. Source adjacency, equal content, common context, or membership in one relationship MUST NOT imply another.

### REQ 5 Preserve Datum and Attempt Identity

Datum identity MUST be immutable and graph-scoped. Private runtime work IDs MUST be random and invocation-local. Text, labels, offsets, source IDs, DataFrame indexes, row order, model-returned IDs, and content-derived hashes MUST NOT satisfy graph or attempt correlation.

### REQ 6 Compile Before Effects

The complete requested graph and its declared capabilities MUST be validated into one immutable compiled plan before graph-invocation effects. The executor MUST consume that plan and MUST NOT widen it from live authoring input, host state, source reconstruction state, or observed rows.

### REQ 7 Account for Every Terminal Outcome

Every expected task, stage, target datum, dependency, atomic group, and invocation MUST close with an exhaustive typed outcome. Missing, failed, cancelled, blocked, lost, duplicated, stale, foreign, or inconsistent evidence MUST NOT be represented as success or raw-input fallback.

### REQ 8 Release Only Qualified Complete Groups

Only policy-qualified success MAY contain output. Release MUST occur after exact reconciliation, strategy verification, publication-critical cleanup, and fixed-point dependency and atomic-group withholding.

### REQ 9 Bound Context and Preserve Product Ownership

Context MUST use a private immutable compiled projection with separate target and context frames, exact binding reconciliation, explicit count and byte ceilings, and provider retention disabled for the first profile. The integrating product MUST authorize source access and field use before graph construction; Anonymizer MUST NOT accept or interpret product authorization tokens.

### REQ 10 Anchor Mentions and Patches Exactly

Private graph mentions MUST name exact target offsets and source slices. Context MAY inform reviewed decisions but MUST NOT supply a mention endpoint or replacement span. Private Redact MUST apply exactly one mention-keyed patch per accepted mention and verify output by authoritative source-plus-patches reconstruction.

### REQ 11 Keep Resolution Evidence-Based

Every mention MUST begin in a singleton cluster. Clusters MAY merge only through accepted versioned same-subject evidence keyed by current mention tokens. Equal text, equal labels, source identity, context membership, and response order MUST NOT merge clusters.

### REQ 12 Preserve Ownership Boundaries

Anonymizer MUST own source-neutral protection semantics and sanitized outcomes. Source adapters and downstream systems MUST retain codecs, source identity, field policy, reconstruction, persistence, durable retry, deduplication, retention, cleanup, and delivery.

### REQ 13 Gate Public Promotion

Any public graph or session surface MUST receive separate public-API authorization. Stable promotion MUST additionally pass the privacy, provenance, lifecycle, capability, artifact, governance, and materially different runtime gates.

### REQ 14 Keep Integration Form Separate from Protection Semantics

A DataFrame adapter and any future bounded-invocation, inline, bounded-microbatch, service, or transport adapter MUST lower once into the same source-neutral graph model and typed outcome model for the selected contract. Packaging and transport MUST NOT redefine datum identity, context declarations, task accounting, atomic groups, release semantics, or downstream ownership. A new public integration form requires separate public-API review.

### REQ 15 Compile Closed Conditional Profiles

Before invocation effects, the compiler MUST fix one immutable versioned profile and closed conditional graph, including semantic tasks, declared applicability and routing edges, dependencies, authorized implementations, schemas, limits, failure semantics, release predicate, and release claim. Runtime MUST NOT introduce an undeclared stage, implementation, fallback, or claim from latency, provider availability, row content, or other live conditions.

### REQ 16 Preserve Accounting Across Physical Optimization

Batching, concurrency, local execution, provider aggregation, and physical call consolidation MAY change effect scheduling but MUST preserve compiled dependencies, keyed outcomes, failure attribution, terminal accounting, strategy verification, and Phase 4 dependency and atomic-group release behavior. Every DataDesigner-backed task MUST continue to execute through `NddAdapter.run_workflow()`.

### REQ 17 Make Execution Observable and Measurable

Every graph phase and execution boundary MUST expose versioned, content-free observations sufficient to measure latency, throughput, resource use, route selection, terminal accounting, reconciliation, cleanup, and protection-quality proxies. Instrumentation MUST preserve semantic behavior and MUST NOT place content, source identity, private work IDs, credentials, endpoints, or unbounded dimensions in telemetry.

## Proposal

### Overview

Adopt an immutable ProtectionGraph as Anonymizer’s private semantic system of record. Keep current public APIs as compatibility facades, keep `NddAdapter.run_workflow()` as the sole DataDesigner execution boundary, and use DataFrames only as temporary stage workframes.

This separates protection semantics from the accidental boundaries of a DataFrame row and a source format. It allows context, replacement coherence, dependencies, and atomic release to vary independently while preserving downstream ownership of source and operational concerns.

### Semantic Model

ProtectionGraph contains source-neutral datums plus independently declared context scopes, coherence scopes, dependencies, and atomic groups. The semantic phases are immutable and explicit:

`ProtectionGraph → DetectedGraph → ResolvedGraph → PlannedGraph → TransformedGraph → VerifiedGraph`

DetectedGraph contains accepted datum-anchored mentions. ResolvedGraph contains deterministic entity clusters and evidence. PlannedGraph contains dispositions and replacement assignments. TransformedGraph contains mention-keyed patches or keyed group revisions. VerifiedGraph contains exhaustive datum and atomic-group outcomes.

Each executor stage consumes only the immediately preceding immutable phase. A grouped operation must return a complete keyed group result or fail; it cannot silently execute as unrelated row operations.

### Execution Architecture

The execution path is:

`source-owned value → source adapter validation and projection → ProtectionGraph → temporary graph workframe → existing pandas workflows → NddAdapter.run_workflow() → exact reconciliation and typed hydration → source adapter reconstruction → downstream persistence and delivery`

Compilation is pure against declared capabilities. Opening an invocation binds providers, credentials, resources, and private invocation-local work IDs. Runtime correlation never reuses caller or source identifiers.

Only allowlisted content-free identifiers, bounded counts, reason codes, and verification statements may enter public receipts or diagnostics. Raw target or context text, entities, prompts, replacement values, graph IDs, and content-derived hashes remain private.

### Product Experimentation and Integration

The graph-native design creates one stable semantic trunk for product experimentation and future integration. Identity, exact context projection, semantic-task outcomes, terminal accounting, verification, and release remain graph-defined while qualified implementations and execution envelopes may vary within a compiled contract.

An existing system may eventually submit one bounded invocation and receive qualified atomic-group outputs or typed non-success outcomes through an adapter. The integrating product authorizes source access and field use before that adapter constructs a graph. The adapter and integrating system retain source decoding, field policy, reconstruction, persistence, retries, deduplication, retention, cleanup, and delivery. This RFC does not select an in-process, service, container, or Platform deployment form.

#### Independent Axes

* Strategy and release claim: Redact, Substitute, or Rewrite.
* Qualified pipeline profile: the current full path or a separately reviewed alternative implementation or restricted contract.
* Execution envelope and backend: single datum, bounded microbatch, or batch; local or DataDesigner-backed task.
* Deployment form: in-process SDK, local service, container, or hosted service, subject to separate public and operational decisions.

A bounded microbatch normally changes scheduling, not graph identity, context declarations, atomic groups, or release semantics. Streaming transport remains out of scope because it would require its own ordering, backpressure, checkpointing, recovery, and delivery contract.

#### Closed Conditional Profiles

Compilation fixes one closed conditional semantic graph and its authorized implementations before effects. Runtime may follow only predeclared applicability, skip, failover, fallback, or retry edges and must record the selected route and terminal outcome. This rule does not add retry, failover, or fallback to phases whose accepted design excludes them.

Profiles remain an Anonymizer-owned closed catalog. Narrow internal capabilities may support independently implemented stages where real substitution pressure exists, but this RFC does not create a user-extensible Anonymizer plugin surface or a public profile selector.

#### Semantic Tasks and Physical Effects

Semantic tasks are accounting and verification units, not necessarily provider calls. One physical call may serve several tasks only when every task outcome remains keyed, attributable, schema-valid, and independently reconcilable. If physical consolidation prevents safe fault localization, it is not a compatible optimization and must withhold the wider affected scope under the compiled contract.

Every DataDesigner-backed task continues to execute through `NddAdapter.run_workflow()`. Pure local implementations participate in the same graph accounting even when they do not call DataDesigner.

#### Performance Program

The separate Performance and Experimentation tab owns benchmark methodology, experiment arms, candidate implementations, workload evidence, comparison results, and promotion recommendations. Candidate examples such as regex-first detection, restricted rule-based detection, smaller or specialized models, physical call consolidation, batching, and deployment prototypes are not approved behavior. Performance evidence cannot authorize a privacy claim, product profile, deployment, public API, or SLO.

#### Observability and Profiling

The graph lifecycle exposes versioned observations through the existing opt-in measurement surface for preflight, workframe construction, semantic stages, backend calls, reconciliation, cleanup, and release. When measurement is active, each entered boundary records a start/terminal pair with monotonic duration. Observations may also carry bounded or bucketed workload dimensions, profile and implementation versions, selected route, terminal outcomes, reason codes, reconciliation and cleanup status, and allowlisted numeric or bucketed provider-usage fields. This common contract allows full-path and alternative implementations to be compared without changing graph semantics.

Public results and observability remain separate surfaces. Anonymizer may join a caller's distributed trace as a child operation, but caller trace IDs never become datum identity, private work IDs, or metric labels. Measurement failure cannot supply missing terminal evidence, fabricate success, or change the release set. The separate Performance and Experimentation tab owns benchmark interpretation and promotion evidence; the SDK redesign owns the lifecycle events and privacy boundary that make those measurements possible.

### Phase 4: Hierarchical Terminal Accounting

Phase 4 adds a private one-shot ledger for stage tasks, target datums, dependencies, invocations, and atomic groups. Dependencies form an explicit DAG. Atomic groups form a flat exact partition. Cycles, nesting, overlap, coverage gaps, unknown references, and implicit singleton completion are rejected before graph-invocation effects.

The ledger reconciles keyed terminal evidence and distinguishes failed, cancelled, blocked, lost, and inconsistent outcomes. Post-dispatch cancellation without trusted stop evidence is lost. Phase 4 performs no automatic retry. Release occurs only after exhaustive reconciliation and fixed-point dependency and group withholding.

Status: the authorized private branch-local implementation and its evidence gates completed on 2026-08-25. RFC acceptance, public-API approval, production integration, later-phase authorization, and promotion remain pending their separate decisions.

### Phase 5: Target and Bounded-Context Workframes

Phase 5 compiles one bounded projection for every output-bearing target datum. Target text and explicitly declared context remain in separate logical frames and bind to one Phase 4 target task through immutable compiled binding identities and fresh private `target_work_id`, `context_binding_id`, and `attempt_id` values. These work IDs are invocation-local correlation values, not credentials or public graph identifiers.

Context is an immutable original-text snapshot. It creates no dependency, output, mention, cluster, coherence scope, or release rule. Context-reference cycles are permitted when explicitly declared and bounded because they are not scheduling edges. An invalid binding, exceeded limit, or incompatible backend fails closed rather than dropping context or falling back to an independent row.

The first profile requires an immutable typed context execution contract with explicit limits, closed backend artifact classes and closure attestations, and a backend-attested `retention_disabled` posture. It is a backend-compatibility contract, not product authorization, proof of provider behavior, or a claim that Anonymizer controls provider retention. Reconciliation proves exact target-task, target-row, context-binding, context-row, ordinal, owner, and consumed-evidence bijections. Publication-critical cleanup closes every Anonymizer-owned frame and work-ID map and reconciles every required execution-boundary closure attestation after private task and datum outcomes are terminal but before release. It does not rewrite those absorbing outcomes. Confirmed owned or attested backend cleanup failure closes the invocation as `failed(cleanup_failed)`; missing or contradictory cleanup evidence closes it as `inconsistent(cleanup_unconfirmed)`. Both map every public target to the corresponding non-success outcome and embargo all output.

Status: reviewed and implemented privately on the branch. Commits `bb79cda` through `53ef74f` landed and hardened bounded context admission, separate workframes, exact reconciliation, cleanup, reference-model, privacy, and compatibility evidence on 2026-08-26. This branch-local completion does not authorize public APIs, production integration, or promotion.

### Phase 6: Anchored Mentions and Resolution

Phase 6 converts untrusted candidate evidence into immutable mentions anchored only to exact target offsets. Candidate lineage is closed: each current candidate receives exactly one keep, reclass, or drop decision before finalization. Missing, duplicate, stale, foreign, contradictory, value-only, overlapping, or source-slice-mismatched evidence fails closed.

The compiler creates exactly one resolver task per target. Resolution waits for the complete set of required mention-finalization predecessors, then accepts only versioned same-subject or distinct-subject evidence keyed by eligible current mention tokens. Every mention starts as a singleton; deterministic connected components merge only accepted same-subject evidence and reject contradictions.

The first transform profile is private Redact only. It creates one exact patch per accepted mention, applies patches once against authoritative source offsets without evolving-output search or value fallback, and verifies the returned text through source-plus-patches reconstruction and atomic-group predicates.

Status: reviewed and implemented privately on the branch. Commits `5bf61c6` through `29bddad` landed and hardened anchored mention admission, explicit-evidence resolution, the frozen Redact role policy, mention-keyed verification, reference-model, privacy, and compatibility evidence on 2026-08-27. This branch-local completion does not authorize Substitute, public APIs, production integration, or promotion.

### Phase 7: Stable Substitute

Phase 7 adds a flat coherence partition, replacement-slot planning, and a bounded ephemeral ledger. An entity cluster may map to several type-appropriate replacement slots. Each slot keeps one assignment within one explicitly declared coherence scope and invocation.

Stable means invocation-bounded consistency, not global permanence, restart recovery, cross-worker consistency, transactional delivery, or durable idempotency. Planning creates one complete provisional bundle per scope and exposes no replacement map until qualified atomic-group release. Collision, concurrency, rollback, leakage, lifetime, cleanup, and state authority must fail closed.

Status: reviewed design; branch implementation authorization pending. Execution remains sequenced after Phases 4–6 and after its owned versioned semantic and execution contract is frozen.

### Ownership

* Anonymizer — Source-neutral graph validation; detection, resolution, planning, transformation, verification, dependency and group accounting; sanitized outcomes.
* Source adapter — Codec; closed field policy; source identity; graph projection; bounded reconstruction state; schema validation; reconstruction and output mapping.
* Execution host — Providers, credentials, endpoints, model provisioning, resource ceilings, backend capabilities, and ephemeral state lifetime.
* NeMo Platform Anonymizer plugin — Current public-facade integration, authentication, filesets, providers, jobs, storage, cancellation, artifacts, and delivery lifecycle.
* Future Intake integration — Ingress, partial acceptance, source-item persistence, durable retries, retention, cleanup, destination deduplication, and delivery.
* Integrating event-driven host (proposed boundary) — Event selection, queueing, saturation and omission policy, worker or plugin lifecycle, and subscriber delivery.
* Shared governance — Policy schemas, conformance corpora, thresholds, support matrix, and compliance-facing claims; owner remains unresolved.

### Failure, Cancellation, and Release

Cancellation is an ordered event, not proof that backend work stopped. Pre-dispatch cancellation performs no provider work. After dispatch, a task is cancelled only with trusted stop evidence; otherwise it is lost. Accepted terminal evidence cannot be rewritten by later cancellation or late results.

Attribution defects localize only when the complete unaffected bijection remains provable. Foreign, swapped, cross-target, plan-mismatched, or contradictory evidence that destroys attribution closes the invocation as inconsistent and withholds all groups.

Publication-critical finalization must close all owned workframes, stores, artifacts, and work-ID maps before release. Cleanup qualification occurs after private task and datum terminal acceptance and does not rewrite those outcomes. A cleanup failure or inability to prove cleanup instead closes the invocation with the specified global non-success reason and exposes no public target text. This proves logical lifecycle closure, not secure memory erasure or absence from unowned provider traces.

### Verification Strategy

Each phase must first define a pure reference model independent of pandas, DataDesigner, production reducers, and the Phase 4 ledger. A frozen finite conformance generator must publish its event alphabet, independence relation, exact graph and trace counts, computed bounds, versions, and SHA-256 manifest digest before comparing executor outcomes.

Tests must cover exact-limit and one-over-limit admission; declaration and row permutations; duplicate, missing, stale, foreign, cross-target, and contradictory evidence; cancellation and terminal races; worker death; cleanup and publication failure; opaque-ID renaming; DataFrame compatibility; FailedRecord shape; privacy canaries; and mutations that violate every critical invariant.

Passing the Python reference model and process-backed lifecycle tests does not satisfy the materially different semantic runtime gate.

## Open Questions

* Which customer or consuming-product owner will approve the earliest boundary that unprotected target and context data may cross?
* What closed field roles, representative bounds, and source commit units will Intake owners approve for ATIF, OTLP, and chat-completion workloads before graph construction?
* What opaque provenance and receipt contract will span source items, graph datums, invocations, processes, and artifacts without exposing private correlation or content?
* Who owns policy schemas, conformance corpora, support thresholds, and compliance-facing claims?
* Which materially different semantic runtime will implement the agreed conformance subset before stable promotion?
* When cancellation and cleanup become observable, what async operation surface—if any—should receive separate public review?
* Which parts of the proposed graph SDK should remain experimental until private semantics, lifecycle, diagnostics, and privacy gates pass?

## User Experience Impact

### Overall UX

The private migration changes no current Anonymizer or NeMo Platform user workflow. Existing DataFrame, file, preview, run, evaluation, CLI, trace, and failed-record behavior remains the compatibility contract.

The proposed additive graph SDK lets adopters preserve hierarchy, bounded context, stable substitutions, dependencies, and complete-output requirements without flattening source records. Publication remains gated on private implementation evidence, public-API approval, adapter ownership, cancellation review, and the stable-promotion gates.

### Public API Changes

Status: included in the RFC plan; not implemented or approved for publication. Publication requires separate public-API review.

The public contract uses three immutable generic values and one mandatory state transition:

```text
ProtectionGraph[TargetKey]
    -> PreparedProtection[TargetKey]
    -> ProtectionResult[TargetKey]
```

`prepare()` is the public preflight operation and the only graph admission boundary. It validates and freezes the exact graph, configuration, profile, schemas, routes, dependencies, limits, context projection, stable-substitution partition, and complete-output partition without model calls or other invocation effects. `protect()` accepts only a prepared value:

```python
prepared = anonymizer.prepare(
    graph,
    config=config,
)

result = anonymizer.protect(prepared)
```

`PreparedProtection` is immutable, process-local, and non-serializable. It contains no credentials, live provider session, attempt, or result. The same prepared value may open more than one independent invocation, but reuse does not imply caching, idempotency, or identical generated substitutes.

Invalid or unsupported input raises a typed `ProtectionRejected` before an invocation exists. An admitted invocation returns exactly one terminal outcome per target. The public target outcome is a closed union:

```python
TargetOutcome = (
    Protected
    | Withheld
    | Failed
    | Cancelled
    | Lost
    | Blocked
    | Inconsistent
)
```

Only `Protected` contains text. A locally successful target whose `require_complete` peer does not qualify becomes `Withheld`; raw input is never returned as fallback. Bounded enums or closed values carry public reason codes, and exhaustive pattern matching distinguishes known failure, proven cancellation, uncertain lost execution, dependency blocking, and inconsistent evidence.

Immediately before invocation effects, `protect()` verifies that the bound backend still satisfies the capabilities frozen during preflight. A mismatch raises `ProtectionRejected`; it creates no invocation and never drops context or selects a context-free fallback. After execution begins, exact reconciliation uses private invocation-local `target_work_id`, `context_binding_id`, and `attempt_id` values. Missing or contradictory execution evidence produces typed target outcomes rather than a preflight rejection.

The graph authoring vocabulary is:

* `texts` — immutable text datums keyed by the caller's target-key type;
* `targets` — the ordered output-bearing datum keys;
* `context` — ordered, explicitly declared read-only context per target;
* `depends_on` — explicit target scheduling dependencies;
* `stable_substitutions` — exact target partitions within which the same resolved entity receives the same invocation-bounded substitute; and
* `require_complete` — exact target partitions for which protected text is exposed only when every member qualifies.

Context, dependencies, stable substitutions, and complete-output requirements never imply one another. Direct graph construction requires complete valid partitions. Source adapters may materialize reviewed defaults before compilation.

#### DataFrame benchmark

The checked-in `repo-data-smoke` benchmark selects the `biography` column from `docs/data/NVIDIA_synthetic_biographies.csv`. Its rows are independent source records, so the DataFrame adapter creates one target and singleton partitions per selected row:

```python
import pandas as pd

from anonymizer import Anonymizer, AnonymizerConfig, Redact
from anonymizer.graph import ProtectionGraph


rows = pd.read_csv(
    "docs/data/NVIDIA_synthetic_biographies.csv"
).head(5)

graph = ProtectionGraph.from_dataframe(
    rows,
    target_column="biography",
)

anonymizer = Anonymizer()
prepared = anonymizer.prepare(
    graph,
    config=AnonymizerConfig(replace=Redact()),
)
result = anonymizer.protect(prepared)
```

`target_column` names the selected field's graph role. Existing `AnonymizerInput.text_column` and benchmark `text_column` retain their published names. The factory does not infer context, dependencies, or cross-row relationships. Equal text and duplicate or null DataFrame indexes never merge datums.

Existing callers should continue to use `run()` for the simplest DataFrame-in/DataFrame-out workflow. A graph-native caller that needs exact DataFrame reconstruction uses a `DataFrameAdapter`; the adapter retains source identity and reconstruction state outside `ProtectionGraph`.

#### Whole-trace protection

One graph may represent an entire bounded trace. This example uses parent text as context, keeps substitutions stable across the trace, and exposes no protected span text unless every target qualifies:

```python
span_ids = (
    "trace:7/span:root",
    "trace:7/span:http",
    "trace:7/span:database",
)

graph = ProtectionGraph(
    texts={
        "trace:7/span:root": root_text,
        "trace:7/span:http": http_text,
        "trace:7/span:database": database_text,
    },
    targets=span_ids,
    context={
        "trace:7/span:http": (
            "trace:7/span:root",
        ),
        "trace:7/span:database": (
            "trace:7/span:root",
            "trace:7/span:http",
        ),
    },
    stable_substitutions=(
        span_ids,
    ),
    require_complete=(
        span_ids,
    ),
)
```

Putting the trace in one graph does not require all-or-none output. An adapter may instead declare one singleton `require_complete` set per span when its reconstruction contract permits partial protected traces. Trace hierarchy alone implies neither context nor completeness.

The integrating product authorizes source access and field use before constructing either graph. Anonymizer receives no product authorization token. Preflight enforces exact bounded context handling, and execution rejects an incompatible backend posture before effects. A `retention_disabled` profile is a checked backend requirement, not an Anonymizer guarantee of provider behavior or physical deletion.

### Documentation Changes

Repository development documents remain canonical. This Google document remains a review mirror. Public product documentation and the bundled Anonymizer skill change only if a separately approved public API or behavior changes.

## Implementation Details

### Private Types and Validation

Use immutable closed values for the proposed public graph, prepared protection, target outcomes, and rejection reasons, and for private graph phases, compiled plans, capabilities, identities, outcomes, and evidence. Dynamic DataFrames, dictionaries, serialized payloads, and compatibility models cross one validation boundary before becoming trusted graph values.

Use Pydantic where user-facing validation or serialization is required and frozen dataclasses for immutable engine values. Keep `PreparedProtection` and private graph phases non-serializable. A portable graph artifact requires a separate versioned-schema decision.

### Workframes and NDD

Use COL\_\* constants for every internal column. Use `_jinja()` for shared DataFrame column references and `substitute_placeholders()` for dynamic prompt values. Preserve the no-context legacy workflow and prompt shape where compatibility tests require it.

Graph stages may declare DataDesigner column configurations, but `NddAdapter.run_workflow()` remains the sole execution boundary. Reconciliation starts from the immutable compiled plan, not from rows or work IDs observed after lowering. Instrumentation may measure immediately around that boundary but cannot bypass it or change its semantic result.

### Compatibility

The public facade continues to compile only profiles that have been qualified for compatibility. Public Substitute and Rewrite remain on their legacy semantic paths until separately reviewed graph profiles preserve their filtering, collision repair, prompt, trace, evaluation, and repair behavior.

### Deferred to Implementation

Exact private module and class placement, helper decomposition, batch sizing within frozen ceilings, and performance tuning may be resolved during implementation and code review.

Semantic grammar, requirement strength, rejection precedence, capability versions, privacy boundaries, role policies, release predicates, and ownership gates are not implementation details. Their named owners must freeze or approve them before the phase that consumes them.

## Implementation Phases

### Phases 1–3: Private Compatibility Foundation

Status: branch-local implementation. Phases 1–3 established immutable trivial graphs, independent-datum validation, temporary pandas lowering, typed hydration, and private Redact compatibility. Phase 4 added dependency DAGs, flat exact multi-datum atomic groups, and exhaustive terminal accounting. Later branch-local phases now add bounded context and anchored mention resolution; coherence planning, grouped rewrite, links, and public graph APIs remain unsupported.

### Phase 4: Hierarchical Terminal Accounting

Status: the authorized private branch implementation and evidence gates completed on 2026-08-25. It delivers an immutable compiled accounting plan, one-shot ledger, dependency DAG, flat atomic partition, exhaustive outcomes, exact reconciliation, cancellation/loss rules, and fixed-point release. RFC acceptance, public-API approval, production integration, later-phase authorization, and promotion remain separate gates.

### Phase 5: Target and Context Workframes

Status: reviewed and implemented privately on 2026-08-26 in `bb79cda` through `53ef74f`. The branch delivers immutable context scopes, bounded separate frames, original-text snapshots, a typed context execution contract, exact work-ID reconciliation, a retention-disabled first profile, observable lifecycle cleanup, and the versioned content-free observation contract. Product authorization and field policy remain outside Anonymizer.

### Phase 6: Anchored Mentions and Private Redact

Status: reviewed and implemented privately on 2026-08-27 in `5bf61c6` through `29bddad`. The branch delivers exact target-offset mentions, closed candidate lineage, one resolver task per target, deterministic evidence-based clustering, versioned role results, mention-keyed Redact patches, and exact reconstruction.

### Phase 7: Stable Substitute

Status: reviewed design; branch implementation authorization pending. If separately authorized after Phases 4–6 and contract freeze, deliver flat coherence planning, type-appropriate replacement slots, bounded ephemeral ledger, deterministic collision handling, complete provisional bundles, and qualified release.

### Phase 8: Grouped Rewrite

Status: proposed. Add keyed group rewrite, evaluation, and repair with no independent-row fallback.

### Phase 9: Result Compatibility

Status: proposed. Move legacy result materialization behind compatibility adapters while retaining public behavior.

### Phase 10: Bounded Inspection

Status: proposed. Add bounded explain, inspect, and diagnose views, then prepare graph/session records for separate review.

### Phase 11: Lifecycle and Independent Runtime

Status: proposed. Validate lifecycle behavior through a process-backed host and the agreed conformance subset through a materially different semantic runtime. The Python host supplies lifecycle evidence only.

## Related Proposals

* Implementation and review PR: https://github.com/NVIDIA-NeMo/Anonymizer/pull/253
* Phase 4 hierarchical terminal-accounting design: repository file docs/development/phase-4-hierarchical-terminal-accounting-design.md
* Phase 5 target/context workframe design: repository file docs/development/phase-5-target-context-workframe-design.md
* Phase 6 anchored-mention resolution design: repository file docs/development/phase-6-anchored-mention-resolution-design.md
* Phase 7 stable Substitute design: repository file docs/development/phase-7-stable-substitute-design.md
* [Intake workload evidence](intake-workload-validation-evidence.md).
* [Performance and experimentation program](anonymizer-performance-and-experimentation.md) — proposed cross-context program; no candidate profile or product SLO is approved.

## Alternate Solutions

### Alternative 1: Keep DataFrame Rows as the Semantic Unit

Pros: smallest internal change; preserves the current mental model.

Cons: conflates context, coherence, dependency, and atomic release; cannot faithfully represent related-record workloads.

Reason rejected: flattening would silently weaken requested semantics and make source format or row position an accidental part of protection identity.

### Alternative 2: Put Source Formats and Delivery Lifecycle in Anonymizer

Pros: one component could appear to own end-to-end processing.

Cons: couples protection semantics to ATIF, OTLP, chat, Intake, Relay, and future formats; duplicates platform responsibilities; expands the privacy and reliability boundary.

Reason rejected: Anonymizer should own source-neutral protection semantics, while adapters and downstream systems retain codecs and durable effects.

### Alternative 3: Implement the Proposed Graph API Before Private Qualification

Pros: adopters could integrate against the reviewed graph concepts immediately.

Cons: would freeze semantics, lifecycle, diagnostics, provenance, capability, and artifact contracts before implementation and conformance evidence exist.

Reason rejected: this RFC may settle the candidate public contract, but private implementation and evidence must precede publication and compatibility commitments.

### Alternative 4: Start with Durable Cross-Worker Replacement State

Pros: could provide restart and distributed consistency earlier.

Cons: requires governed storage, transactional semantics, recovery, idempotency, retention, and operational ownership beyond the current SDK boundary.

Reason rejected for the first profile: Phase 7 deliberately qualifies invocation-bounded stability. Durable consistency remains a separate architecture decision.

## Appendix A: Status Labels

* Published current behavior — Supported by an immutable public repository revision.
* Branch-local implementation — Present on the draft research branch or explicitly identified working tree; not published behavior or a public contract.
* Dated dogfood observation — A bounded result from a named test or environment on a stated date; not a product guarantee.
* Proposal — Architecture to adopt or work to perform; not current behavior.
* Unresolved gate — A decision, assumption, authority, or proof still required. A provisional assumption is not approval.

## Appendix B: Gates Before Stable Public Promotion

* Materially different runtime — Independent semantic implementation of a declared capability subset and shared conformance outcomes — Anonymizer and adopter architecture review.
* Privacy boundary — Earliest boundary unprotected content may cross, named actors, residual risk, and release criteria — Customer or consuming-product owner routed by the Intake team.
* Provenance — Opaque identity and receipt contract across invocation, process, and artifact boundaries — Anonymizer and adopter architecture review.
* Related-record semantics — Hierarchical accounting, no silent flattening, and conformance for context, coherence, dependency, and atomic groups — Anonymizer architecture review.
* Stable Substitute — Collision, concurrency, rollback, leakage, scope lifetime, cleanup, and state-lifetime evidence — Anonymizer semantic and execution owners.
* Lifecycle — Bounded resources, readiness, cancellation, cleanup, crash, and version behavior — Runtime owners.
* Observability — Versioned content-free lifecycle events, bounded dimensions, privacy allowlist, non-interference, and profiling coverage — Anonymizer semantic and measurement owners.
* Public graph or session surface — Separate publication review and explicit authorization — Public-API owners.
* Stable public artifacts — Versioned schemas, capability negotiation, reconstruction, OpenAPI and SDK regeneration, and cross-repository tests — Anonymizer and Platform owners.
* Governance — Owners for policy, corpus, thresholds, support matrix, and compliance-facing claims — Unresolved.

## Appendix C: Evidence Base

* Published Anonymizer facade: https://github.com/NVIDIA-NeMo/Anonymizer/blob/3eab7d1b6005b85e7d415b704e27a20dc41ba71e/src/anonymizer/interface/anonymizer.py
* Published `NddAdapter.run_workflow()` boundary: https://github.com/NVIDIA-NeMo/Anonymizer/blob/3eab7d1b6005b85e7d415b704e27a20dc41ba71e/src/anonymizer/engine/ndd/adapter.py
* Branch-local graph model: src/anonymizer/engine/execution/graph.py
* Branch-local graph runtime: src/anonymizer/engine/execution/graph\_runtime.py
* Branch-local protection service: src/anonymizer/engine/execution/protection\_service.py
* Branch-local compatibility flow: src/anonymizer/interface/\_protection.py
* [Separate Intake workload evidence](intake-workload-validation-evidence.md)

## Next Decision

Project reviewers should accept the complete RFC development and research plan or identify required revisions. Acceptance permits branch-local implementation and experimentation only through the ordered prerequisites and operator checkpoints described above; it does not approve public API publication or any production integration. Private Phases 4–6 are implemented with branch-local evidence. Phase 7 remains after those completed prerequisites and still requires its contract freeze, semantic and execution owner approval, and separate operator authorization.
