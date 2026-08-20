<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Technical Proposal — Graph-native Anonymizer SDK

Status: proposed architecture, with a private branch-local implementation checkpoint as of 2026-08-20. This proposal supersedes the Architecture RFC and SDK Companion Plans tabs in the [external integration design record](https://docs.google.com/document/d/15lQESfIU0wD5ysvdaILR-TtXCWfp5gySv59-NW6NeV8/edit); it does not authorize any public graph API, a production Intake integration, or a release claim. “Samples” are outside this rewrite and require separate review.

## Decision

**[Proposal]** Adopt an immutable `ProtectionGraph` as Anonymizer’s semantic system of record. Keep the current public APIs as compatibility facades, keep `NddAdapter.run_workflow()` as the sole boundary for executing DataDesigner workflows, and use DataFrames only as temporary stage workframes.

**[Proposal]** This decision separates protection semantics from two accidental boundaries: a DataFrame row and a source format. Context, replacement coherence, and atomic release can then vary independently without moving codecs, platform jobs, persistence, retries, deduplication, or delivery into Anonymizer.

**[Proposal and unresolved gate]** The private architecture may advance now. Any public graph or session exposure requires separate review and authorization. Stable promotion additionally requires a materially different semantic runtime to implement the reviewed contract and acceptance of the privacy and provenance decisions described below.

## How to read the status labels

Every consequential claim uses one of these labels:

- **[Published current behavior]** Behavior supported by an immutable public repository revision.
- **[Branch-local implementation]** Uncommitted behavior in this worktree at `d39e74f17ef090e84ca9c4fb86f47e6cee2ecd4d` plus the listed working-tree changes.
- **[Dated dogfood observation]** A bounded observation from a named test or environment on a stated date; not a product guarantee.
- **[Proposal]** Architecture to adopt or work to perform; not current behavior.
- **[Unresolved gate]** A decision, assumption, or proof still required. A provisional assumption is not approval.

## Contract to preserve

**[Published current behavior]** Anonymizer exposes DataFrame-oriented `run()`, `preview()`, and `evaluate()` paths, and `NddAdapter.run_workflow()` is the engine boundary that executes DataDesigner workflows. The replace and rewrite pipelines use this adapter rather than creating or previewing DataDesigner workflows directly. [Facade and `run()` path](https://github.com/NVIDIA-NeMo/Anonymizer/blob/3eab7d1b6005b85e7d415b704e27a20dc41ba71e/src/anonymizer/interface/anonymizer.py#L151-L257) · [DataDesigner execution boundary](https://github.com/NVIDIA-NeMo/Anonymizer/blob/3eab7d1b6005b85e7d415b704e27a20dc41ba71e/src/anonymizer/engine/ndd/adapter.py#L267-L328)

**[Proposal]** Preserve the signatures, constructors and config types, defaults, result columns and attributes, `failed_records` shape, and CLI behavior of the current public facade throughout the internal migration. `trace_dataframe` and current `evaluate()` behavior remain pandas compatibility contracts until an additive graph-level design is separately reviewed.

**[Proposal]** Preserve `NddAdapter.run_workflow()` as the sole DataDesigner execution boundary. A graph stage may declare DataDesigner column configurations and may batch work in a DataFrame, but it must execute through the adapter.

**[Proposal]** Treat a successful protection outcome as a qualified execution result, not proof that detection was exhaustive or that output contains no PII. A release predicate can verify only its reviewed strategy and policy postconditions.

## Semantic model

**[Proposal]** `ProtectionGraph` contains source-neutral text datums, links, and three independent scopes:

```text
context scope    — which related datums may inform a decision
coherence scope  — where aliases and replacements must remain consistent
atomic group     — which protected outputs succeed or fail together
```

A DataFrame row must not stand in for all three. For example, a trace can use parent spans as context, a trace-wide coherence scope, and per-span atomic groups. A trajectory can instead require bounded turn context, trajectory-wide replacement coherence, and a whole-trajectory atomic group. Unsupported related-record semantics fail closed; they never fall back to independent rows.

**[Proposal]** The semantic phases are immutable and explicit:

```text
ProtectionGraph
  -> DetectedGraph       # accepted, datum-anchored mentions
  -> ResolvedGraph       # entity clusters and alias evidence
  -> PlannedGraph        # dispositions and replacement assignments
  -> TransformedGraph    # patches or keyed group revisions
  -> VerifiedGraph       # exhaustive datum and atomic-group outcomes
```

Each phase should have its own closed type. Detection offsets remain anchored to the target datum. Entity clusters and replacement slots remain distinct, so related aliases can refer to one subject while using type-appropriate replacements. A grouped rewrite must return a keyed group result or fail; it cannot silently run as unrelated row rewrites.

## Execution architecture

**[Proposal]** The graph is authoritative before and after each stage. A private executor lowers the ready frontier to a temporary workframe, calls the existing pandas/DataDesigner backend, reconciles invocation-private tokens, and hydrates typed results into the next graph phase.

```text
source-owned value
  -> source adapter: validate, project, retain reconstruction state
  -> ProtectionGraph
  -> graph stage: lower to temporary DataFrame workframe
  -> existing pandas workflows
  -> NddAdapter.run_workflow()
  -> graph stage: verify and hydrate typed outcomes
  -> source adapter: reconstruct
  -> downstream persistence and delivery
```

The text equivalent is: adapters project source data into graph semantics; Anonymizer runs source-neutral protection through temporary DataFrames; adapters reconstruct protected source data; downstream systems own durable effects.

**[Proposal]** Caller or source identifiers never become engine correlation tokens. Runtime correlation stays opaque and invocation-local. Public receipts contain only allowlisted, content-free identifiers and verification statements; they do not include raw entities, prompts, source content, or content-derived hashes.

**[Proposal]** Terminal accounting is exhaustive by datum and atomic group. Only a policy-qualified success contains output. Rejected, failed, cancelled, unknown, missing, duplicated, or inconsistent outcomes withhold the affected atomic group. A transport break that returns no run record also withholds output; raw input is never the fallback.

## Compatibility migration

**[Branch-local implementation]** Private phases 1–3 exist in `src/anonymizer/engine/execution/graph.py`, `src/anonymizer/engine/execution/graph_runtime.py`, `src/anonymizer/engine/execution/protection_service.py`, and `src/anonymizer/interface/_protection.py`. The implementation defines immutable, non-serializable graph values; validates a trivial independent-datum profile; lowers it through the existing pandas runtime; hydrates graph outcomes; and maps those outcomes back to the private compatibility flow.

**[Branch-local implementation]** The trivial compiler accepts only singleton context, coherence, and atomic scopes. It rejects links, added context, multi-datum coherence, and multi-datum atomic groups with explicit private validation codes. This establishes a real extension boundary without claiming related-record support or weakening requested semantics. The branch-local tests are in `tests/engine/execution/test_graph_runtime.py`.

**[Branch-local implementation]** No public graph API exists. The graph slice qualifies only its private Redact release profile; it does not qualify Substitute or Rewrite for graph execution and does not change current public signatures.

**[Branch-local verification — 2026-08-20]** The verifier reran `uv run --frozen pytest -q` against the current uncommitted worktree: **1,408 passed and 11 skipped** with one warning. The focused graph and private-protection tests passed **38 tests**. The 11 opt-in Intake dogfood tests skipped because their external environment was not enabled; the dated observations below remain historical operator-run evidence, not results from this suite invocation.

**[Proposal]** The next architectural phase is hierarchical datum, dependency, and atomic-group accounting. Complete it before adding related context, replacement coherence, or grouped rewrite so that every later semantic feature inherits correct failure and release boundaries.

The ordered migration is:

1. **[Branch-local implementation — phases 1–3]** Encapsulate the pandas backend, extract source-neutral protection services, and prove trivial-graph compatibility. The current private slice implements only these phases.
2. **[Proposal — phase 4 hard gate]** Add hierarchical stage-task, datum, dependency, and atomic-group terminal accounting. Define and test DAG versus cycle policy; atomic-group partition, nesting, and overlap rules; dependency-failure propagation; group withholding; exhaustive outcomes for datum, stage, invocation, and lost execution; and pre-effect rejection of unsupported semantics with no independent-row fallback. Phases 5–10 do not begin or qualify until this gate passes.
3. **[Proposal — phase 5]** Add separately framed target and bounded context workframes.
4. **[Proposal — phase 6]** Add datum-anchored mentions, deterministic entity grouping, and group verification for local Redact-like strategies.
5. **[Proposal — phase 7]** Add coherence-scope replacement planning and a bounded ephemeral ledger. Qualify stable Substitute only after collision, concurrency, rollback, and leakage testing.
6. **[Proposal — phase 8]** Add keyed group rewrite, evaluation, and repair with no independent-row fallback.
7. **[Proposal — phase 9]** Move legacy result materialization behind compatibility adapters while retaining public behavior.
8. **[Proposal — phase 10]** Add bounded explain, inspect, and diagnose views. Prepare bounded graph/session records for review after their semantics, diagnostics, cancellation, and cleanup are verified.
9. **[Proposal — phase 11]** Validate lifecycle behavior through a process-backed host and validate the agreed conformance subset through a materially different semantic runtime. The process-backed Python host supplies lifecycle evidence only; it does not satisfy the second-runtime gate.

**[Unresolved gate]** Public exposure, including an experimental graph or session surface, requires separate authorization. Freeze and promote a stable portable contract only after that authority exists, phase 11 passes, and the privacy, provenance, lifecycle, and capability gates below pass.

## SDK and runtime boundary

**[Proposal]** Subject to separate public-surface review, the eventual small SDK would consist of a constrained graph or builder, an immutable compiled plan, a synchronous protection session, closed datum and group outcomes, and bounded content-free capabilities, failures, and receipts. Compilation is pure against declared capabilities; opening a session binds providers, credentials, resources, and other host authority.

**[Proposal]** An in-process session may provide bounded process-local replacement consistency. It does not provide durable idempotency, restart recovery, cross-worker consistency, transactional output-and-ledger commit, or platform job recovery. Those claims require a durable service and a governed state backend.

**[Unresolved gate]** Async operation handles are deferred until cancellation and cleanup are observable in the underlying runtime. Cancelling a caller’s waiter does not by itself prove that model or workflow execution stopped.

**[Unresolved gate]** Do not introduce an open engine protocol or select a wire format from the Python implementation alone. A process-backed Python host can validate lifecycle, IPC, crash, and cancellation behavior, but it is not a materially different semantic runtime.

## Ownership

**[Proposal]** The boundary follows this allocation:

| Owner | Responsibility |
| --- | --- |
| Anonymizer | Source-neutral graph validation; detection, resolution, transformation, and release semantics; context, coherence, dependency, and atomic-group accounting; sanitized outcomes |
| Source adapter | Codec; closed field policy; projection; bounded reconstruction state; source identity; schema validation; reconstruction and output mapping |
| Host authority | Providers, credentials, endpoints, model provisioning, resource ceilings, context exposure, replacement-scope authorization, and state lifetime |
| NeMo Platform Anonymizer plugin | Published public-facade integration; authentication, filesets, provider resolution, jobs, storage, cancellation, artifacts, and delivery lifecycle |
| Future Intake integration | Ingress, source-item persistence and partial acceptance, durable retries, retention, cleanup, destination deduplication, and delivery |
| Relay | Event selection, queueing, saturation and omission policy, worker or plugin lifecycle, and subscriber delivery |
| Shared governance | Policy schemas, conformance corpora, thresholds, support matrix, and compliance claims; owner remains unresolved |

**[Proposal]** Source formats stay downstream. ATIF, OTLP, chat-completion, direct-span, OCSF, Intake, Relay, and OpenShell types do not enter Anonymizer core. Multiple formats handled by Intake exercise distinct workload shapes but remain one Intake runtime, not a second semantic implementation.

**[Proposal]** For target graph/session adoption, source adapters may propose graph relations and atomic groups under a closed policy; Anonymizer validates support for the requested source-neutral semantics. The adapter maps the source-owned commit unit to graph atomic groups and retains the reconstruction manifest. Anonymizer does not select a codec, commit durable data, schedule retries, or deduplicate destination writes. Legacy `run()` and `preview()` retain their published file/DataFrame input behavior during migration.

## NeMo Platform compatibility and future adoption

**[Published current behavior]** At NeMo Platform commit `e1057736703bb8b167a4bd9013cea0caae2df63a`, the current Anonymizer plugin pins `nemo-anonymizer==0.3.3`, constructs the public `Anonymizer` facade, and calls public `run()`, `preview()`, and `validate_config()`. It persists and reconstructs DataFrame-shaped artifacts and owns the service, job, fileset, provider, storage, cancellation, and delivery lifecycle. This released plugin is distinct from the proposed Intake workload adapter; no current plugin-to-Intake graph integration exists. [Plugin dependency](https://github.com/NVIDIA-NeMo/nemo-platform/blob/e1057736703bb8b167a4bd9013cea0caae2df63a/plugins/nemo-anonymizer/pyproject.toml#L10-L20) · [Run job](https://github.com/NVIDIA-NeMo/nemo-platform/blob/e1057736703bb8b167a4bd9013cea0caae2df63a/plugins/nemo-anonymizer/src/nemo_anonymizer_plugin/jobs/run.py#L41-L141) · [Preview worker](https://github.com/NVIDIA-NeMo/nemo-platform/blob/e1057736703bb8b167a4bd9013cea0caae2df63a/plugins/nemo-anonymizer/src/nemo_anonymizer_plugin/functions/_preview_worker.py#L31-L73)

**[Proposal, inferred compatibility]** Private phases 1–3 should require no Platform change if current constructors and config types, `run()`, `preview()`, `validate_config()`, result columns and attributes, and `failed_records` shape remain stable, because the plugin does not call private protection or graph symbols. This is a call-site inference, not an executed cross-repository compatibility test.

**[Proposal]** Future graph/session adoption requires a Platform-owned adapter for cell-to-datum projection, outcome-to-row reconstruction, and source identity; capability negotiation; versioned artifacts and readers; cancellation and cleanup review; OpenAPI and SDK regeneration for any new endpoint schema; and cross-repository tests. The legacy endpoints remain compatibility adapters until that work is complete.

**[Published current behavior]** The Platform plugin writes `dataset.parquet`, `trace.parquet`, `metadata.json`, and optional `failed_records.json` sequentially before `ctx.results.save`; no atomic artifact-bundle contract has been established. Preview uses `abandon_on_cancel=True`, so abandoning the async wait may leave the synchronous worker running. A future session design must therefore establish cancellation and cleanup behavior rather than inherit it by assumption.

## Atomicity, retry, and deduplication

**[Published current behavior]** Current Anonymizer `run()` and `preview()` return one Python result or raise; `failed_records` identify records dropped by a workflow in an otherwise returned result. This is Python result-publication behavior. It is not artifact atomicity—the Platform plugin writes result files sequentially before publication—and it is not a transaction over providers, telemetry, source ingestion, storage, or delivery.

**[Proposal]** Anonymizer atomicity is scoped to declared graph atomic groups and the publication of their qualified outcomes. A source adapter maps source commit units to those groups and withholds or reconstructs accordingly. Intake decides its persistence and partial-acceptance behavior.

**[Proposal]** Retry ownership remains downstream. A source adapter that retries must retain the exact protected payload when safe, preserve stable source identity, classify transport uncertainty, and verify the destination postcondition. Neither a repeated successful Anonymizer call nor one collapsed read-model row establishes transactional Intake idempotency.

## Gates before stable public promotion

| Gate | Evidence required | Authority or routing surface |
| --- | --- | --- |
| Materially different runtime | Independent semantic implementation of a declared capability subset; shared conformance outcomes | Anonymizer and adopter architecture review |
| Privacy boundary | Earliest boundary unprotected content may cross; named actors; residual-risk and release criteria | Customer or consuming-product owner, routed by the Intake team |
| Provenance | Reviewed opaque identity and receipt contract across invocation, process, and artifact boundaries | Anonymizer and adopter architecture review |
| Related-record semantics | Hierarchical accounting, no silent flattening, and conformance for context, coherence, dependency, and atomic groups | Anonymizer architecture review |
| Stable Substitute | Collision, concurrency, rollback, leakage, scope-lifetime, and state-authority evidence | Anonymizer semantic owner and host authority |
| Lifecycle | Bounded resources, readiness, cancellation, cleanup, crash, and version behavior | Runtime owners |
| Any public graph/session surface | Separate review and explicit authorization | Project and public-API owners |
| Stable public artifacts | Versioned schemas, capability negotiation, reconstruction rules, OpenAPI/SDK regeneration, and cross-repository tests | Anonymizer and Platform owners |
| Governance | Owners for policy, corpus, support thresholds, and compliance-facing claims | Unresolved |

**[Unresolved gate]** “Before Intake” is only a provisional planning assumption. It is not customer-approved. The accepted decision must distinguish the source adapter, optional edge component, Intake process, durable storage, operator-facing APIs and UI, and downstream consumers.

**[Unresolved gate]** “Zero PII” has no accepted contract meaning. The design must not claim absence of all PII. A review may instead define which content, detection policy, release predicate, residual risk, and trust boundary make an output eligible for a particular use.

## Explicit non-goals

This proposal does not:

- make a public graph API current behavior;
- treat multiple Intake formats, a process-backed host, a test adapter, or current OpenShell telemetry as a second semantic runtime;
- move source codecs, platform jobs, storage, retries, deduplication, retention, purge, or delivery into Anonymizer;
- claim transactional Intake idempotency or storage rollback;
- claim that “before Intake” is approved;
- permit unsupported related-record semantics to degrade to independent rows;
- claim exhaustive detection or absence of all PII;
- authorize production OpenShell changes; or
- revise or approve “samples.”

## Evidence base

- [Published Anonymizer facade](https://github.com/NVIDIA-NeMo/Anonymizer/blob/3eab7d1b6005b85e7d415b704e27a20dc41ba71e/src/anonymizer/interface/anonymizer.py#L151-L257)
- [Published `NddAdapter.run_workflow()` boundary](https://github.com/NVIDIA-NeMo/Anonymizer/blob/3eab7d1b6005b85e7d415b704e27a20dc41ba71e/src/anonymizer/engine/ndd/adapter.py#L267-L328)
- Branch-local graph model: `src/anonymizer/engine/execution/graph.py`
- Branch-local graph runtime: `src/anonymizer/engine/execution/graph_runtime.py`
- Branch-local protection service: `src/anonymizer/engine/execution/protection_service.py`
- Branch-local compatibility flow: `src/anonymizer/interface/_protection.py`
- [Separate Intake workload evidence](intake-workload-validation-evidence.md)

## Next decision

Review and accept the graph semantics, ownership split, migration order, and stable-promotion gates. If accepted, begin hierarchical datum/dependency/atomic-group accounting; do not publish a stable graph surface or infer the unresolved privacy and provenance decisions.
