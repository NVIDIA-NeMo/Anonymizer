<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Graph SDK production cutover

M0 characterizes a small part of the existing production contract before migration. The graph components and their conformance tests do not yet establish that every production route uses one graph authority. This inventory records evidence and remaining work; it is not a complete RFC audit or publication approval.

The reference revision is `9f824d95b1cc74e71eb4a152a033afabbfd84ac0`. Normative authority remains the SDK RFC main text and Appendices A–E, snapshot SHA256 `55da6300f42404673fbd94999fc0763390532917403e37b74ac674000bdffc8f`. The requirement summaries below are navigation aids, not replacements for the normative clauses. Before activating a successor unit, bind its exact predecessor revision, relevant normative sections, concrete objective and test selectors. Appendix E's open decisions remain open.

## Stage ownership

| Stage | Responsibility |
| --- | --- |
| M1 | Typed domain artifacts, backend capabilities, private correlation, physical request admission and settlement |
| M2 | Detection, exports/plugins and local Redact/Annotate/Hash migration |
| M3 | Informational evaluation, result projections and saved-result compatibility |
| M4 | Substitute, evidence-based identity, replacement slots and continuity |
| M5 | Rewrite, grouped assessment, bounded repair and both DataDesigner schedules |
| M6 | Switch every production entry point and retire displaced orchestration |
| M7 | Integrated compatibility, lifecycle, privacy, packaging and cutover evidence; separate publication review |

M2 and M3 can proceed independently after their M1 prerequisites. The campaign currently serializes their execution. Minimization checkpoints surround migration; moving code or retaining forwarding wrappers is not retirement.

## Requirement map

Paths under `engine/graph_sdk` below refer to `src/anonymizer/engine/graph_sdk`; test names refer to `tests/graph_sdk`. Existing evidence identifies a bounded implementation/test surface, not full requirement acceptance. All rows retain a production-integration obligation.

| Requirement | Existing implementation / test evidence | Missing work and owner stage |
| --- | --- | --- |
| REQ 1 — Public compatibility | `interface/anonymizer.py`, `interface/results.py`; `tests/interface/test_anonymizer_interface.py`, M0 production witnesses | Preserve signatures, defaults, canonical errors, columns, failures, traces, CLI and saved results across all migrated routes — M2–M6; final evidence M7 |
| REQ 2 — Graph semantic record | `graph/workflow.py`, `engine/graph_sdk/data.py`, `preparation.py`; `test_data_graph.py`, `test_workflow_static.py` | Normalize production authoring into the same graph and hydrate one canonical execution record — M1, M6 |
| REQ 3 — DataDesigner boundary | `engine/ndd/adapter.py:NddAdapter.run_workflow`; `tests/engine/test_ndd_adapter.py`, M0 retry witness | Preserve this boundary in every migrated DD path, including combined scheduling and extensions — M1–M6 |
| REQ 4 — Independent scopes | `data.py` distinct context/coherence/dependency/atomic declarations; `test_data_graph.py` | Preserve distinctions in production lowering; never infer relationships from rows or batches — M2, M4, M5 |
| REQ 5 — Datum and attempt identity | `graph/_values.py`, `requests.py`; `test_record_values.py`, `test_requests.py` | Bind invocation-private opaque identities through DD workers and hydrate keyed results; legacy content-derived record IDs are not graph identity — M1 |
| REQ 6 — Compile before effects | `preparation.py`, `capabilities.py`, `_execution_admission.py`; `test_preparation.py`, `test_effects_production_conformance.py` | Admit actual selected backend visibility, retry ownership, resources and settlement before production effects — M1, M6 |
| REQ 7 — Exhaustive terminal outcomes | `records.py`, `executor.py`, `requests.py`; `test_external_executor.py`, `test_requests.py` | Reconcile dropped/reordered/foreign DD rows and all attempts into typed task/datum/group outcomes — M1, M3, M6 |
| REQ 8 — Qualified complete release | `qualification.py`, `_execution_publication.py`; `test_qualification_production_conformance.py`, `test_publication_reference_conformance.py` | Derive legacy/protected/annotation/review projections from one record; keep completion, availability and criteria satisfaction distinct — M3–M6 |
| REQ 9 — Bounded context | `context.py`, `binding.py`; `test_context_source_execution.py`, `test_ingress_bounds.py` | Carry admitted context and failures into detection and grouped rewriting without treating failed binding as empty context — M1, M2, M5 |
| REQ 10 — Exact mentions and patches | `data.py`, `engine/replace/strategies.py`; `test_data_graph.py`, `tests/engine/test_replace_strategies.py` | Connect accepted mention identities to verified source patches; reject unsupported overlapping output ownership — M2 |
| REQ 11 — Evidence-based resolution | `evidence.py`, `continuity.py`; `test_evidence.py`, `test_continuity.py` | Wire real detection mentions, singleton initialization and accepted versioned identity evidence to Substitute — M2, M4 |
| REQ 12 — Ownership and privacy | `observations.py`, `resources.py`; `test_observations.py`, `test_cleanup_scope_witnesses.py` | Verify production/dependency log and exporter privacy; preserve caller-owned clients/state and host-owned codecs/delivery — M1, M6, M7 |
| REQ 13 — Promotion gates | `tests/graph_sdk/test_public_api.py` exercises public exports and import isolation | Complete library/API, lifecycle, privacy, component and packaging gates; no M0 publication claim — M7 |
| REQ 14 — Integration versus semantics | `binding.py`, `continuity.py`; `test_binding.py`, `test_continuity_snapshots.py` | Validate bounded production adapters and explicit continuity; external hosts keep stream transport/lifecycle — M4, M6, M7 |
| REQ 15 — Closed conditional profiles | `graph/_workflow_admission.py`, `preparation.py`; `test_compiler_reference_conformance.py`, `test_workflow_static.py` | Compile built-in selected implementations, corrections, failover, repair bounds and qualification coverage — M1, M2, M4, M5 |
| REQ 16 — Physical optimization accounting | `requests.py`, `_request_driver.py`; `test_requests.py`; M0 real DD retry witness | Connect every physical dispatch to admission and settlement; reject requested hard budgets on opaque paths; validate each schedule separately — M1, M5 |
| REQ 17 — Bounded observations | `observations.py`, `inspection.py`; `test_observation_reference_conformance.py`, `test_observations.py` | Integrate production latency/resources and planned/conditional/actual request reporting with the versioned content-free allowlist — M1, M6, M7 |
| REQ 18 — Typed relations | `data.py`, `graph/_workflow_values.py`; `test_data_graph.py`, `test_workflow_analysis.py` | Preserve value, context, subject, readiness, datum-release and atomic relations through lowering; DD column dependencies cannot create protection relations — M1, M2, M4, M5 |

## Production route inventory

| Route | Current implementation / existing evidence | Cutover obligation |
| --- | --- | --- |
| `run()` and `preview()` | [Anonymizer](../../src/anonymizer/interface/anonymizer.py), [interface tests](../../tests/interface/test_anonymizer_interface.py), [M0 witnesses](../../tests/graph_sdk/test_production_contracts.py) | M6: preserve record selection, text-column renaming, public result types and failure aggregation |
| Redact / Annotate / Hash | [ReplacementWorkflow](../../src/anonymizer/engine/replace/replace_runner.py), [local strategies](../../src/anonymizer/engine/replace/strategies.py), [strategy tests](../../tests/engine/test_replace_strategies.py) | M2: preserve configured markers, sensitive-text annotation and deterministic text-based hashing; do not infer subject identity |
| Substitute | [LlmReplaceWorkflow](../../src/anonymizer/engine/replace/llm_replace_workflow.py), [tests](../../tests/engine/test_llm_replace_workflow.py) | M4: preserve instructions, parsing, collisions, per-row maps and failures while adding explicit identity/slot continuity |
| Sequential Rewrite | [RewriteWorkflow](../../src/anonymizer/engine/rewrite/rewrite_workflow.py), [tests](../../tests/engine/test_rewrite_workflow.py) | M5: preserve prompts, initial evaluation, repair, review flags and unchanged zero-entity output; separately qualify related-text/latent risks |
| Combined DD Rewrite | [CombinedRewriteWorkflow](../../src/anonymizer/engine/rewrite/combined_rewrite_workflow.py), [tests](../../tests/engine/test_combined_rewrite_workflow.py) | M5: independent request/correlation conformance for `Rewrite.use_combined_graph`; physical consolidation cannot change release semantics |
| Detection and plugins | [EntityDetectionWorkflow](../../src/anonymizer/engine/detection/detection_workflow.py), [detection tests](../../tests/engine/test_detection_workflow.py) | M2: preserve validation-before-augmentation, filters, chunking, latent detection and plugin transforms; M7: fresh-process installed-wheel discovery |
| Detection export | `Anonymizer.export_detection_config`, `export_detection_builder_for_seed`; `test_build_detection_builder_for_seed_threads_validation_single_chunk_full_text` | M2/M6: retain export/seed configuration contract; exported construction is distinct from executing a workflow |
| Injected workflows/runners/backend | `Anonymizer.__init__`; interface injection tests and M0 local route | M1/M6: retain injection compatibility; declare/reject unsupported graph capability combinations before effects without claiming visibility inside opaque runners |
| Post-hoc `evaluate()` | `ReplacementWorkflow.evaluate`, `RewriteWorkflow.evaluate`, coverage/detection/final judges; [evaluation tests](../../tests/engine/test_evaluate.py) | M3: keep optional informational evaluation separate from Rewrite's automatic evaluate/repair and graph qualification |
| Saved-result pickle evaluation | `Anonymizer.evaluate` restores internal column names; M0 round-trip to a fresh evaluator | M3: retain stored method, text column, detection context and trace; establish version scope separately; characterization covers same-revision round trips |
| CLI | [CLI package](../../src/anonymizer/interface/cli), [CLI tests](../../tests/interface/cli) | M6: preserve argument parsing, configuration, output files and canonical failures through the shared authority |
| Failures, traces and result projections | [results](../../src/anonymizer/interface/results.py), [result tests](../../tests/interface/test_results.py), M0 witnesses | M3/M6: preserve `FailedRecord(record_id, step, reason)`, hidden trace columns and evaluation's unavailable rows; M0 records evaluation-local failures, not accumulated run failures |

M0 reuses `tests/conftest.py` synthetic entity/model fixtures. It checks real local replacement and public projection through both run/preview, same-revision pickle evaluation with real judge postprocessing, and retry success/exhaustion through the real DD retry executor. Detection, DD orchestration and provider output are mocked. Expected public names and text are fixed literals, independent of production column-renaming and replacement implementations. These tests do not validate detector/LLM quality or graph cutover.

## DataDesigner request seam and next witness

`NddAdapter.run_workflow()` calls DD `create()` or `preview()` once under a run lock. A workflow can contain several rows, model columns, corrections and retries. `_DataDesignerUsageProbe` and task/message tracing observe work; they do not reserve a graph request budget before dispatch.

The M0 seam witness uses locked DataDesigner 0.9.1's `ModelRequestExecutor` with `RetryConfig(max_retries=1)` and a mocked `ModelClient`. One adapter call produces two model-client calls, two distinct request-attempt IDs and two admission/release pairs, both after success-on-retry and retry exhaustion. It also checks canonical `AnonymizerWorkflowError` wrapping and that the caller-owned client is left open. DD orchestration is a boundary double; this is not a complete installed-backend or HTTP-wire conformance test. There is no hard-budget claim.

M1's next concrete witness should submit one real DD LLM column for two synthetic seed rows, return their responses out of order, and trigger a retry for one row. Keep all model transport mocked. Connect graph request reservation/dispatch/settlement to DD `ModelRequestExecutor._execute_sync_attempt` / `_execute_async_attempt` through its `RequestAdmissionController.acquire_sync` / `acquire_async` and `release` seam. With an invocation hard limit of two, assert that the third model-client dispatch is prevented **before** the call, every dispatched request retains its private semantic associations, and both targets receive truthful terminal outcomes. Add a late-response case in the subsequent settlement unit; a Lost terminal must not be rewritten and unknown usage must stay unknown.

The unresolved authority is the supported way to inject or compose that admission controller for each resource provider and preserve graph association context across DD workers, rows, corrections and failovers. DD admission leases regulate backend resources; they are not yet the graph's physical-request ledger. Event sinks and post-call usage cannot substitute for a veto. The retry owner, replay safety, local versus uncertain remote capacity, cancellation evidence and client lifetime require explicit selected-path contracts. Backend exceptions and physical dispatches must not be confused with confirmed provider acceptance or billing. Opaque paths must disclose unknown bounds and reject a selected hard request budget. M1 remains pending.

Bounded navigation used astnav `refs NddAdapter.run_workflow`, lspgate inspection and direct producer/consumer reads. Astnav returned zero indexed references; lspgate found production detection calls and 22 references with a bounded preview, but unresolved dependency types in the initially unconfigured checkout. Neither result establishes absence. Source reads confirm the adapter call sites and DD retry/admission seam used above.

## Size accounting

Count tracked Python added minus deleted against `30b6d82af74953949ce654c57cd42ed3e4e71041`, including examples and tooling. Test/helper/reference Python and non-Python fixtures are separate. At M0's input revision, the measured net growth is **21,443 non-test Python lines** and **42,715 test/helper/reference Python lines** (38,517 tests/helpers plus 4,198 reference Python). Non-Python fixture/provenance growth is 42,370 text lines plus five archives. M0 changes no production Python and adds only its bounded test module; the campaign receipt records its exact measured delta and resulting totals.

No retirement occurs in M0. The 10K non-test target remains a stretch; 5K remains unsupported. The proposed 9,801–15,726 future range is an unproven planning estimate, not a measured result. M7 must validate the same built wheel on Python 3.11–3.13 outside the source tree, all production routes, plugin discovery and applicable lifecycle/privacy contracts before claiming cutover evidence.
