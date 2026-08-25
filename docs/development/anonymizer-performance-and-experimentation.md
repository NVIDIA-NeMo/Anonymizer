<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Anonymizer Performance and Experimentation

*Cross-context optimization program*

Status: Proposed as part of the complete RFC development and research plan. RFC acceptance permits branch-local experiments under this program; it does not qualify an optimization, approve an execution profile, establish a product SLO, select a deployment form, or authorize production use.

Canonical architecture: SDK Redesign tab and repository development documents.

## Decision

Establish a repeatable performance and experimentation program for Anonymizer across batch, bounded-invocation, inline, and bounded-microbatch workloads. Use the graph-native semantic contracts to test alternative implementations and execution choices without silently changing identity, authorization, accounting, verification, release, privacy claims, or downstream ownership.

This tab owns hypotheses, workload definitions, measurement methods, experiment records, comparison evidence, and promotion recommendations. The SDK Redesign RFC owns semantic contracts, compilation, task accounting, release rules, compatibility, and the permitted internal extension mechanism. Performance evidence may recommend promotion; it does not authorize a product profile, deployment, privacy claim, public API, or product SLO.

## Why This Work Is Separate

Performance is a property of the complete Anonymizer system, not a streaming-only feature. Improvements to detection, models, prompts, stage execution, batching, serialization, resource reuse, or deployment may benefit DataFrame batch calls, one-at-a-time inline calls, bounded microbatches, and future integrations wherever the measured workload assumptions hold.

The term streaming is intentionally avoided here. A bounded invocation is one finite request whose datums, relationships, limits, and selected contract are known at compilation. A streaming transport would additionally require ordering, backpressure, checkpointing, delivery, retry, recovery, and partial-result semantics; this program does not define those behaviors.

## Goals

* Measure end-to-end and per-stage latency, throughput, predictability, resource use, reliability, model use, and quality.
* Support controlled experiments with detectors, models, prompts, parsers, physical stage execution, batching, runtime implementation, and deployment prototypes.
* Make results reproducible against pinned inputs, environments, revisions, profiles, and evidence schemas.
* Require correctness, privacy, compatibility, and failure-behavior gates alongside performance evidence.
* Promote only changes whose supported scope, claim, owner, rollout, rollback, and requalification rules are explicit.
* Reuse improvements across workload contexts only where their measured assumptions and qualified contract apply.

### Non-Goals

* Approve a fast profile, regex detector, smaller model, stage consolidation, batching policy, or deployment form in advance.
* Promise a universal latency, throughput, cost, or availability target.
* Treat baseline parity as proof of privacy or detection completeness.
* Create a user-extensible Anonymizer plugin surface or permit live undeclared stage substitution.
* Design a streaming transport or move queues, retries, persistence, reconstruction, retention, or delivery into Anonymizer.
* Replace the existing Anonymizer measurement contract with a second benchmark system.

## Vocabulary and Claim Classes

Experiment arm — A benchmark-only configuration used to test a hypothesis. It carries no support or deployment claim.

Candidate implementation — An implemented alternative that has not passed promotion gates.

Supported execution profile — A versioned product contract with accepted semantic, correctness, privacy, compatibility, failure, performance, ownership, rollout, rollback, and requalification evidence.

Same-profile optimization — A change that claims the same supported behavior. It must prove compatibility and non-inferiority plus absolute correctness and privacy gates.

Equivalent alternative implementation — A different implementation of the same semantic tasks. It requires conformance evidence before substitution.

Restricted profile — A deliberately narrower entity, input, strategy, quality, or release contract. It must declare unsupported cases and cannot inherit another profile’s privacy claim.

Router profile — A profile whose compiled graph predeclares eligibility, routing, fallback, and terminal behavior among supported paths and records the actual route.

Deployment or scheduling experiment — A change to placement, batching, concurrency, or transport overhead that does not by itself change the protection contract.

Research probe — An investigation that produces evidence but is not eligible for product use.

## Architecture Guardrails

Compilation fixes one closed conditional semantic graph and its authorized implementations before invocation effects. Runtime may traverse only predeclared applicability, skip, failover, fallback, or retry edges and must record the selected route and terminal outcome. This statement does not authorize retry, failover, or fallback in profiles whose accepted design excludes them.

Semantic tasks remain accounting and verification units even when one physical call implements several tasks. Consolidation is eligible only when every task outcome remains keyed, attributable, schema-valid, and independently reconcilable. If safe localization is impossible, failure withholds the wider affected scope under the compiled release contract.

Every DataDesigner-backed task executes through NddAdapter.run\_workflow(). Pure local implementations do not need DataDesigner, but they participate in the same compiled task, terminal-outcome, verification, and release accounting.

Phase 4 release semantics remain authoritative: every expected task and datum reaches a terminal outcome; dependencies and flat atomic groups apply fixed-point withholding; group predicates qualify publication; and only released atomic groups contain output. Public failed\_records remains a compatibility surface where applicable and is not graph identity.

Single-datum, bounded-microbatch, and batch execution normally change scheduling only. They must preserve datum identity, authorized context, semantic outcomes, failure policy, and release scope. Provider output position and DataFrame row order are never identity.

## Workload Model

Every benchmark must identify the workload rather than reporting one aggregate performance number. Required dimensions include:

* Strategy and qualified contract, such as Redact, Substitute, or Rewrite.
* Entity-bearing and no-entity inputs, because no-entity paths may bypass model work.
* Text-size, entity-density, label, language, Unicode, context-size, and relationship strata.
* Single datum, bounded microbatch, or batch execution.
* Batch size, offered concurrency, achieved concurrency, and provider rate limits.
* Local and provider-backed stages.
* Cold-start and warm steady-state execution.
* Success, unsupported, inapplicable, failed, timed-out, cancelled, lost, and inconsistent outcomes where the selected contract defines them.
* Deployment prototype, if any, with transport time reported separately from core processing.

## Existing Measurement Surface

The anonymizer.measurement package, docs/development/observability.md, and tools/measurement are the canonical measurement and benchmark surfaces. They already record runs, stages, row throughput, DataDesigner workflows, direct benchmark model workflows, requests and tokens, failures, safety and quality fields, parameter sweeps, sealed artifacts, and sanitized W\&B summaries.

This program extends that contract. It does not assume that current aggregate and median reports establish per-datum tail latency, queue delay, concurrency saturation, open-loop load behavior, or deployment SLOs. New metrics and analysis require schema, privacy, completion-seal, and compatibility review.

## Required Graph Observation Contract

The SDK redesign must expose versioned, content-free observations through the existing opt-in measurement surface for preflight, workframe construction, semantic stages, backend calls, reconciliation, cleanup, and release. When measurement is active, each entered boundary records a monotonic duration and closed outcome. Where available and privacy-reviewed, observations also carry bounded or bucketed workload dimensions, resource use, allowlisted numeric or bucketed provider usage, selected route, semantic and implementation profile versions, reason codes, and protection-quality proxies.

This is the common measurement surface for the current path and experimental alternatives such as regex-first detection, specialized models, physical call consolidation, and batching. An experiment may change implementation and scheduling, but it must preserve the semantic accounting events required by its claimed profile. Missing lifecycle observations are missing evidence, not zero cost or success.

Public results and observations remain separate. Caller trace IDs may establish distributed-trace parentage, but they do not become datum identity, private work IDs, or metric labels. Target/context text, prompts, entities, replacements, source identifiers, graph identifiers, private work IDs, endpoints, credentials, and unbounded or content-derived dimensions are forbidden. Measurement failure cannot fabricate terminal evidence or change the release set.

## Measurement Method

Use controlled component probes to isolate detector, model, prompt, parser, DataFrame, serialization, graph-building, initialization, resource-reuse, and transport overhead. Use end-to-end benchmarks to determine whether a component improvement survives the complete Anonymizer path. A microbenchmark win is not promotion evidence by itself.

Freeze before execution:

* Baseline and candidate code revisions, configurations, profile manifests, prompt and parser versions, rule sets, thresholds, and exact model or endpoint revisions.
* Materialized corpus and workload version, including provenance and held-out role.
* Hardware, runtime, dependency versions, provider, endpoint, region, and resource limits.
* Warmup policy, cold/warm posture, batching, concurrency, measurement boundary, timeouts, retry posture, repetition count, and stopping rule.
* Primary performance hypothesis and whether the candidate claims equivalence, restriction, routing, deployment-only change, or research evidence.
* Mandatory correctness, privacy, quality, reliability, compatibility, and resource gates.

Remote-model comparisons should use identical materialized inputs and a paired, interleaved, or counterbalanced schedule where practical. Report uncertainty and environmental limitations. Do not publish tail percentiles without a defined sampling unit and enough observations to support them.

## Metrics

* End-to-end latency and per-stage or per-workflow service time, using a named sampling unit.
* Queue wait, service time, and total datum latency separately for bounded microbatches.
* Throughput, offered load, achieved load, saturation, latency variance, and timeout behavior.
* CPU, memory, accelerator, initialization, serialization, and instrumentation overhead.
* Provider request count, token use, failures, and rate-limit behavior.
* Detection, transformation, utility, leakage, release-conformance, and FailedRecord evidence appropriate to the selected strategy.
* Report resource consumption before monetary cost. A currency estimate must name the currency, date, rate card, provider and model revision, excluded costs, retry treatment, and calculation method.

## Correctness and Privacy Gates

Performance and privacy are not interchangeable scores. A faster candidate does not pass by compensating for worse privacy or correctness with lower latency.

Baseline parity is necessary for a same-profile claim but is not sufficient: a baseline comparison can preserve every baseline miss. Qualification also requires absolute gates on an independently labelled held-out corpus. Missing or inadequate ground truth produces insufficient evidence, not success.

Evidence should include exact and relaxed span results, per-label floors, transformation coverage, original-value leakage, unsupported and inapplicable cases, adversarial formats, Unicode and normalization cases, overlap behavior, and strategy-specific utility or relational consistency. LLM judges may supplement but do not replace independently adjudicated labels.

## Candidate Experiment Families

The following are unapproved experiment families, not a roadmap commitment or supported behavior.

### Regex-first hybrid detection

A candidate experiment may use deterministic rules to seed or replace the initial candidate detector while retaining declared downstream validation, augmentation, finalization, verification, accounting, and release behavior. It must prove exact target offsets, rule provenance, Unicode and normalization handling, overlap behavior, adversarial complexity bounds, per-label quality, end-to-end leakage, and failure semantics.

This description does not assert that regex is faster, sufficiently complete, safe for any entity class, or equivalent to the current detector.

### Regex-only restricted detection

A separate research arm may test a locally executable path for a narrow declared label and input domain. It would be a restricted contract, not a fast form of the full pipeline. It must return unsupported or otherwise fail closed outside its qualified scope and cannot treat no rule match as proof that no PII exists.

No regex-only profile is approved by this tab.

### Smaller or specialized models

Candidate experiments may bind a smaller, specialized, local, fine-tuned, or alternate provider model to a declared semantic task. Model identity, prompt, parser, label set, thresholds, provider policy, and actual route must be recorded. Performance evidence must be paired with stage and end-to-end quality, failure, and privacy evidence.

### Physical stage consolidation

Experiments may reduce orchestration or provider overhead by combining physical calls. Consolidation does not remove semantic tasks. Promotion requires preserved keyed outputs, terminal outcomes, failure attribution, cancellation behavior, FailedRecord reconciliation, and measurement visibility. The existing combined rewrite work is evidence that fewer workflows do not by themselves prove a speedup or compatibility.

### Batching, concurrency, and runtime

Experiments may explore bounded batching, concurrency, DataFrame construction, serialization, tokenization, graph compilation, initialization, connection reuse, and instrumentation overhead. They must measure queueing, saturation, fairness, memory, provider limits, partial failure, and cancellation without using row position as identity or exposing one datum as unauthorized context for another.

### Deployment prototypes

The program may measure externally supplied in-process, local-service, container, or hosted prototypes. Such measurements do not select a deployment form or assign hosting, autoscaling, transport retry, persistence, backpressure, or delivery ownership.

## Reported Product Interest and Latency Notes

The separate Streaming Mode discovery note lists OpenShell, Switchyard, Relay, CrowdStrike, and Fortinet. Unless stronger evidence is linked, these names indicate reported interest, not accepted requirements, supported integrations, or product commitments.

The same note records “30 ms latency expectation” under CrowdStrike and “A few hundred ms” under Fortinet. These are unvalidated discovery notes, not SLOs, benchmark results, or Anonymizer requirements. They lack a named accountable owner, percentile, start and stop boundary, payload and entity density, concurrency, environment, warm/cold posture, model and provider, error budget, quality and privacy objective, and accepted failure policy.

Do not use these figures as targets until the relevant owner supplies a measurable requirement and approves its context. Until then, experiments may investigate representative low-latency workloads without claiming that they satisfy either note.

## Evidence Status for Product Inputs

Reported interest — Preliminary or unattributed input; not a requirement.

Candidate workload — A named owner supplies workflow placement, request shape, representative bounds, and intended outcome.

Validated workload evidence — A pinned corpus and environment produce dated, reproducible observations.

Accepted requirement — The accountable owner approves measurable performance, quality, privacy, and failure criteria.

Supported integration — Implementation, semantic conformance, operational, public-surface, and deployment gates pass.

A use-case record should name its source, product owner, technical owner, workflow insertion point, request shape and bounds, measurement definition, throughput and concurrency, quality and privacy objective, failure and fallback policy, trust boundary, deployment constraints, representative corpus, validation date and revision, and unresolved decisions.

## Experiment Lifecycle

* Classify the experiment arm and claim before implementation.
* Freeze the hypothesis, baseline, candidate, corpus, environment, measurement method, and acceptance gates.
* Validate instrumentation and run correctness and privacy preflight checks.
* Run controlled component and end-to-end comparisons.
* Analyze distributions, strata, failures, uncertainty, quality, privacy, and resource use.
* Record a result of rejected, inconclusive, further research, candidate for qualification, or qualified for a separately authorized rollout.
* For promotion, name the approving authorities, supported scope, rollout, rollback, monitoring, drift, and requalification triggers.

## Promotion Evidence

A promotion package should include:

* Immutable profile or candidate manifest and distinct version dimensions for code, semantic graph, prompts, parsers, rules, thresholds, exact models or endpoints, provider policy, schemas, retry or fallback policy, and release predicate.
* Pinned benchmark corpus, environment, raw sealed measurements, analysis, uncertainty, and limitations.
* Component and end-to-end performance results.
* Baseline comparison plus absolute correctness, privacy, quality, compatibility, failure-attribution, cancellation, and release evidence.
* Named product, semantic, privacy, compatibility, architecture or API, deployment, and operational authorities where applicable.
* Rollout, rollback, monitoring, drift detection, and requalification plan.

If an authority or evidence class is missing, the candidate remains an experiment or candidate implementation.

## Initial Experiment Backlog

* Establish reproducible baselines for current Redact, Substitute, and Rewrite paths where each path is already qualified.
* Measure single-datum non-model overhead and compare it with complete end-to-end time.
* Prototype an unapproved regex-first hybrid arm for declared labels and exact target offsets.
* Prototype a separate unapproved regex-only restricted Redact research arm.
* Evaluate smaller or specialized models for individual semantic tasks.
* Measure physical stage consolidation while preserving semantic accounting and failure attribution.
* Explore bounded batching and concurrency across the latency, throughput, memory, and failure frontier.
* Measure deployment overhead only after a deployment owner supplies a prototype and measurement boundary.

Backlog order does not imply priority, approval, implementation authorization, or product commitment.

## Open Decisions

* Who owns performance, quality, privacy, and product-promotion thresholds?
* Which corpus is authoritative, how is it governed and versioned, and where may sensitive examples live?
* What constitutes equivalence for nondeterministic Rewrite output?
* Which labels, languages, and adversarial cases require independent critical gates?
* What harness and sampling rules are required before inline tail-latency or load-saturation claims are permitted?
* Which deployment forms are supported, observed, or out of scope, and who owns each?
* May caching or deduplication be investigated, and under what isolation, retention, invalidation, and version-binding policy?
* How are model, provider, prompt, dependency, hardware, and workload drift detected and requalified?
* Should a restricted path reject, declare unsupported, or route to a broader profile when its declared scope does not apply? No behavior is selected here.

## References

* [Graph-native Anonymizer SDK RFC](graph-native-anonymizer-sdk-rfc.md).
* [Intake workload validation evidence](intake-workload-validation-evidence.md).
* Streaming Mode discovery document: https://docs.google.com/document/d/1eYsTD49wBbIrE\_321JuFeptZTzs-zMz4VotDvQG44Uk/edit
* Repository observability contract: docs/development/observability.md
* Repository measurement tools: tools/measurement/README.md
* Combined rewrite experiment plan: plans/237/combined-rewrite-graph.md

## Next Action

Review this program boundary and identify owners for the first baseline and workload corpus. Do not select or implement a candidate profile, adopt a latency target, or infer adopter acceptance from this tab.
