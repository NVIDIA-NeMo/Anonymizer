<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Evidence — Intake workload validation

Status: evidence record as of 2026-08-20. This tab validates workload pressure on the graph-native SDK proposal; it does not define Anonymizer architecture, claim production format support, approve a PII boundary, or make “samples” part of this rewrite.

## Scope and interpretation

The evidence covers NeMo Platform Intake at commit `e1057736703bb8b167a4bd9013cea0caae2df63a`, the Anonymizer draft research branch at `702f43a988cf3673d16f40be5c59bc784737e1a3`, and dated synthetic dogfood anchored to the revisions named below. Consequential claims use the same labels as the proposal: **[Published current behavior]**, **[Branch-local implementation]**, **[Dated dogfood observation]**, **[Proposal]**, and **[Unresolved gate]**.

**[Proposal]** Several formats passing through one Intake service provide varied workload evidence. They do not constitute a second semantic runtime for Anonymizer’s stable-public-API gate.

## Current Intake workload

**[Published current behavior]** Intake exposes ATIF, OTLP/HTTP protobuf, and OpenAI-compatible chat-completion ingress. These routes normalize source data into `IntakeSpan` records. Content that may require protection occurs in `input`, `output`, and retained raw attributes, often as JSON serializations of objects, arrays, or scalars. [Format reference](https://github.com/NVIDIA-NeMo/nemo-platform/blob/e1057736703bb8b167a4bd9013cea0caae2df63a/packages/nemo_platform_ext/src/nemo_platform_ext/skills/nemo-intake/references/ingest-formats.md#L32-L161) · [Normalized span model](https://github.com/NVIDIA-NeMo/nemo-platform/blob/e1057736703bb8b167a4bd9013cea0caae2df63a/services/intake/src/nmp/intake/spans/domain.py#L41-L61)

| Format | Published current behavior | Design pressure, not current support |
| --- | --- | --- |
| ATIF | Accepts ATIF v1.0–v1.7; one trajectory can normalize into a recursive span tree | Preserve hierarchy and order while testing bounded context, trajectory coherence, and reviewed atomic groups |
| OTLP | Accepts OTLP/HTTP protobuf at the traces endpoint; a mixed-validity request can report per-span errors while retaining valid spans | Preserve trace and parent identity; reconcile per-span acceptance with source-item reconstruction and atomic-group policy |
| Chat completion | Accepts a captured request/response and creates one LLM span | Preserve structured request/response fields, tool content, source identity, and stable retry fields under a closed policy |

**[Published current behavior]** Intake currently provides semantic or model-dumped fidelity, not byte-for-byte source fidelity. ATIF retains validated fragments under `atif.raw`; OTLP retains selected unconsumed attributes, events, and instrumentation scope; chat-completion ingress stores parsed request and response objects as JSON strings. Therefore, any protection adapter must preserve parsed structure and field identity rather than treat every payload as prose. [ATIF boundary](https://github.com/NVIDIA-NeMo/nemo-platform/blob/e1057736703bb8b167a4bd9013cea0caae2df63a/services/intake/src/nmp/intake/spans/ingest/atif.py#L100-L138) · [OTLP receiver](https://github.com/NVIDIA-NeMo/nemo-platform/blob/e1057736703bb8b167a4bd9013cea0caae2df63a/services/intake/src/nmp/intake/spans/ingest/otlp.py#L57-L122) · [Chat normalization](https://github.com/NVIDIA-NeMo/nemo-platform/blob/e1057736703bb8b167a4bd9013cea0caae2df63a/services/intake/src/nmp/intake/spans/ingest/chat_completions.py#L143-L163)

**[Published current behavior]** Provider-neutral direct-span ingress and importers for MLflow, LangSmith, Phoenix, and Braintrust also exist. They remain outside this initial three-format validation scope until separately reviewed. Their existence supports an open-ended source vocabulary, not an Anonymizer enum of supported sources.

## Platform plugin impact

**[Published current behavior]** The current NeMo Platform Anonymizer plugin pins `nemo-anonymizer==0.3.3` and calls only the public `Anonymizer` facade: `run()`, `preview()`, and `validate_config()`. It accepts CSV/Parquet and fileset-backed inputs, resolves providers and secrets, runs jobs, and publishes DataFrame-shaped result artifacts. The plugin—not Anonymizer—owns authentication, filesets, provider resolution, job state, storage, cancellation, artifact publication, download, and delivery. This released plugin is distinct from the proposed Intake workload adapter; there is no current plugin-to-Intake graph integration. [Plugin dependency](https://github.com/NVIDIA-NeMo/nemo-platform/blob/e1057736703bb8b167a4bd9013cea0caae2df63a/plugins/nemo-anonymizer/pyproject.toml#L10-L20) · [Run job](https://github.com/NVIDIA-NeMo/nemo-platform/blob/e1057736703bb8b167a4bd9013cea0caae2df63a/plugins/nemo-anonymizer/src/nemo_anonymizer_plugin/jobs/run.py#L41-L141) · [Preview worker](https://github.com/NVIDIA-NeMo/nemo-platform/blob/e1057736703bb8b167a4bd9013cea0caae2df63a/plugins/nemo-anonymizer/src/nemo_anonymizer_plugin/functions/_preview_worker.py#L31-L73)

**[Proposal, inferred compatibility]** The private phases 1–3 should be compatible with the current plugin if current constructors and config types, `run()`, `preview()`, `validate_config()`, result columns and attributes, and `failed_records` shape do not change. This follows from inspected call sites; it has not yet been established by a cross-repository test against the uncommitted graph slice.

**[Proposal]** A future public graph/session integration, if separately authorized, is not a drop-in replacement. Platform would need projection and reconstruction, capability negotiation, versioned artifacts, cancellation and cleanup review, OpenAPI and SDK regeneration for new schemas, and cross-repository tests. Platform must retain its existing lifecycle responsibilities. Preview cancellation is specifically unresolved because abandoning an async wait may leave the synchronous worker running.

## Dated validation observations

The operator-backed observations in this section are historical run evidence retained in the checked-in runbook and test contracts. In the 2026-08-20 verification run, all 11 opt-in Intake dogfood tests skipped because the external Intake/ClickHouse/Sandbox environment was not enabled.

**[Dated dogfood observation — 2026-08-19]** The branch-local hermetic corpus exercised synthetic ATIF v1.0 and v1.7, extension-bearing chat-completion JSON, real OTLP protobuf bytes, and an Intake-shaped CHAIN-to-LLM trace through the private Redact profile. Tests checked complete-item reconstruction, topology preservation, closed field policy, and withholding for invalid spans or non-success outcomes. They did not use provider-backed detection, customer data, or durable Intake commit and do not establish production format support. [Hermetic adapter tests](https://github.com/NVIDIA-NeMo/Anonymizer/tree/d39e74f17ef090e84ca9c4fb86f47e6cee2ecd4d/tests/streaming)

**[Dated dogfood observation — 2026-08-19]** An opt-in test-only run used an operator-owned local Intake service backed by ClickHouse 26.3. Raw and protected ATIF v1.0, ATIF v1.7, chat-completion JSON, and OTLP/protobuf traversed public Intake routes. The protected read models omitted the specifically declared synthetic test values and retained the asserted topology and semantic fields. This bounded observation is not evidence that all PII was absent. The checked-in runbook retains the validated revisions, environment, and local instance identity; the opt-in tests retain the asserted route behavior. [Dogfood run record](https://github.com/NVIDIA-NeMo/Anonymizer/blob/d39e74f17ef090e84ca9c4fb86f47e6cee2ecd4d/tools/intake_dogfood_runbook.md#L15-L34) · [Validated instance](https://github.com/NVIDIA-NeMo/Anonymizer/blob/d39e74f17ef090e84ca9c4fb86f47e6cee2ecd4d/tools/intake_dogfood_runbook.md#L274-L291) · [Opt-in dogfood tests](https://github.com/NVIDIA-NeMo/Anonymizer/blob/d39e74f17ef090e84ca9c4fb86f47e6cee2ecd4d/tests/streaming/test_intake_dogfood.py#L211-L636)

**[Dated dogfood observation — 2026-08-19]** A completed isolated Sandbox session was mapped by a closed test-only adapter to ATIF v1.0. Deterministic local detection protected the declared synthetic values before the test request reached Intake, and the resulting read model retained the asserted parent-child topology. This validates the tested execution and provisional boundary path; it does not validate provider-backed detection quality, production Sandbox support, or a customer-approved “before Intake” boundary. [Sandbox export test](https://github.com/NVIDIA-NeMo/Anonymizer/blob/d39e74f17ef090e84ca9c4fb86f47e6cee2ecd4d/tests/streaming/test_intake_dogfood.py#L549-L580)

**[Dated dogfood observation — 2026-08-19]** A mixed-validity OTLP request exposed an atomicity mismatch. The test adapter rejected the complete request and emitted no bytes, whereas Intake accepted the same raw request, returned one per-span error, and persisted the two valid spans. Complete-request withholding is therefore only a test-adapter policy pending adopter review. It is not current Intake behavior and does not alter current Anonymizer result-publication behavior. [Atomicity-mismatch test](https://github.com/NVIDIA-NeMo/Anonymizer/blob/d39e74f17ef090e84ca9c4fb86f47e6cee2ecd4d/tests/streaming/test_intake_dogfood.py#L375-L449)

**[Dated dogfood observation — 2026-08-19]** A pre-connect delivery failure left the exact protected bytes available for a test retry and created no observed Intake row. Exact-byte retries collapsed to one public read-model row for the tested ATIF and OTLP fixtures. The initial chat fixture produced two public rows because it omitted a source timestamp; preserving one positive, non-future integer `response.created` caused the tested exact-byte retry to collapse to one public chat row. [Delivery-failure and retry tests](https://github.com/NVIDIA-NeMo/Anonymizer/blob/d39e74f17ef090e84ca9c4fb86f47e6cee2ecd4d/tests/streaming/test_intake_dogfood.py#L450-L636)

**[Dated dogfood observation — 2026-08-19]** That last observation is format-specific read-model behavior. It is not evidence of transactional Intake idempotency, rollback, one physical write, or atomic persistence. A duplicate that is hidden by the read model can still represent more than one underlying write.

## Findings for the proposal

**[Proposal]** The evidence supports modeling target data, related context, replacement coherence, and atomic release independently. ATIF hierarchy, OTLP parentage and partial acceptance, and chat request/response structure cannot be represented faithfully by assuming that every independent DataFrame row is the complete semantic unit.

**[Proposal]** A source adapter should own the codec, a closed target/context/preserve/reject field policy, stable source identity, graph projection, reconstruction state, output mapping, protected-byte retention for safe retries, and destination postcondition checks. It must pass only source-neutral graph semantics into Anonymizer.

**[Proposal]** One complete ingestion item is the candidate buffered output-commit unit for adapter design, but Intake’s mixed-validity OTLP behavior proves that this cannot be stated as accepted Intake policy. The adapter and Intake owners must review each format’s partial-success and reconstruction contract.

**[Proposal]** Unsupported context, coherence, atomicity, or dependency semantics must be rejected. Treating related datums as independent rows would erase the workload property being validated.

**[Proposal]** Phase 4 selects a flat exact atomic partition as Anonymizer's first supported release model. That capability does not choose Intake's source-item or per-span grouping: the adapter and Intake owners must still review that mapping, and nested or overlapping groups remain unsupported.

**[Proposal]** The reviewed phase 5 design keeps target and context in separate source-neutral
frames under an immutable bounded capability. It does not select which ATIF, OTLP, or chat
fields become targets or context; the source adapter and adopter owners retain that field
policy. The first profile also requires provider retention to be disabled. Any future
retention-enabled profile needs separate customer-owned privacy-boundary authorization.

**[Proposal]** The reviewed phase 6 design anchors every mention and patch to authoritative
target offsets. Context may inform a reviewed validation, augmentation, or resolution task,
but it cannot supply a mention endpoint or replacement span. Source-field mapping,
reconstruction, source commit units, persistence, retries, destination deduplication, and
delivery therefore remain downstream responsibilities.

## Open gates

**[Unresolved gate]** The customer or consuming-product owner has not selected the earliest boundary that unprotected content may cross. “Before Intake” remains a provisional test posture, not approval. The decision must name the source adapter, optional edge component, Intake process, durable storage, operator-facing APIs and UI, and downstream consumers.

**[Unresolved gate]** No accepted “zero PII” definition exists. An enforceable contract would need a reviewed closed detection and transformation policy, a selected trust boundary, release criteria, and residual-risk treatment. Current evidence supports no claim that all PII is absent.

**[Unresolved gate]** Intake owners must define representative bounds and closed field roles for every relevant `input`, `output`, and raw-attribute surface in ATIF, OTLP, and chat-completion workloads.

**[Unresolved gate]** The adapter contract still needs a reviewed provenance mechanism across source item, graph datum, atomic group, process, and persisted artifact without exposing raw detected entities or internal correlation tokens.

**[Unresolved gate]** Stable public promotion still needs a materially different semantic runtime. Multiple Intake formats, hermetic fixtures, a process-backed Python host, and test-only Sandbox or OpenShell adapters do not satisfy that gate.

**[Unresolved gate]** Future Platform adoption needs versioned projection and artifact contracts, capability negotiation, cancellation and cleanup behavior, OpenAPI/SDK regeneration, and cross-repository tests.

## Ownership retained downstream

| Owner | Work that remains outside Anonymizer |
| --- | --- |
| Source adapter | Format validation; closed field policy; source IDs; projection; reconstruction; protected-byte retention; retry classification; destination postcondition |
| NeMo Platform Anonymizer plugin | Current public-facade integration; authentication; filesets; provider resolution; jobs; cancellation; storage; artifacts; delivery |
| Future Intake integration | Ingress; source-item persistence and partial acceptance; durable retries; retention; cleanup; destination deduplication; delivery |
| Customer or consuming-product owner | Trust boundary and accepted privacy objective |

## Evidence references

- [NeMo Platform formats at `e1057736703bb8b167a4bd9013cea0caae2df63a`](https://github.com/NVIDIA-NeMo/nemo-platform/blob/e1057736703bb8b167a4bd9013cea0caae2df63a/packages/nemo_platform_ext/src/nemo_platform_ext/skills/nemo-intake/references/ingest-formats.md#L32-L161)
- [Normalized `IntakeSpan`](https://github.com/NVIDIA-NeMo/nemo-platform/blob/e1057736703bb8b167a4bd9013cea0caae2df63a/services/intake/src/nmp/intake/spans/domain.py#L41-L61)
- [ATIF normalization](https://github.com/NVIDIA-NeMo/nemo-platform/blob/e1057736703bb8b167a4bd9013cea0caae2df63a/services/intake/src/nmp/intake/spans/ingest/atif.py#L100-L138)
- [OTLP/HTTP protobuf receiver](https://github.com/NVIDIA-NeMo/nemo-platform/blob/e1057736703bb8b167a4bd9013cea0caae2df63a/services/intake/src/nmp/intake/spans/ingest/otlp.py#L57-L122)
- [Chat-completion normalization](https://github.com/NVIDIA-NeMo/nemo-platform/blob/e1057736703bb8b167a4bd9013cea0caae2df63a/services/intake/src/nmp/intake/spans/ingest/chat_completions.py#L143-L163)
- [Branch-local validation tests at `d39e74f17ef090e84ca9c4fb86f47e6cee2ecd4d`](https://github.com/NVIDIA-NeMo/Anonymizer/tree/d39e74f17ef090e84ca9c4fb86f47e6cee2ecd4d/tests/streaming)
- [Retained local dogfood run record](https://github.com/NVIDIA-NeMo/Anonymizer/blob/d39e74f17ef090e84ca9c4fb86f47e6cee2ecd4d/tools/intake_dogfood_runbook.md#L274-L291)
- [Technical proposal](graph-native-anonymizer-sdk-technical-proposal.md)
- [Phase 5 target and bounded-context workframe design](phase-5-target-context-workframe-design.md)
- [Phase 6 anchored-mention, resolution, and local-verification design](phase-6-anchored-mention-resolution-design.md)

## Next evidence action

Have the Intake owner review bounded workloads, field roles, source commit units, retry identity, and partial reconstruction for the three initial formats. Then run cross-repository compatibility tests and rerun opt-in dogfood only in an operator-owned environment under the selected privacy boundary.
These are future adapter-adoption gates; they do not block the authorized private Phase 4 branch implementation. The RFC and its public or production adoption remain under review.
