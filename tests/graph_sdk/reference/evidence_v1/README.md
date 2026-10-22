<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# P7 finite evidence/currentness/release reference v1

This candidate derives evidence support from raw declared and retained rows, then enumerates feasible release subsets under directed datum dependencies and an exact atomic partition. It is independently authored reference evidence for review, not P7 acceptance, an SDK implementation, or a replacement for all historical tests. Attempt 417; the coordinator owns accounting. No continuity/P8 support or commitment semantics are added.

`inputs.json` pins PLAN, accepted S0 v2, D08 v2, the retained qualification support corrections, the neutral legacy corpus/manifests and the 372-row coverage inventory. The author previously wrote the publication references and has seen their independent reviews. For this task the author read the named contracts and neutral legacy case declarations/support notes, including the baseline legacy expected projection and selected legacy status labels. No product implementation/tests or product execution output were inspected. New expected values are computed from the new predicates; the generator imports neither the legacy corpus nor a product module.

## Model and exact identities

`p7_reference/model.py` has three separate predicates:

1. `verify_rows` reconciles retained assessment owner inventory against the successful occurrence/production pairs, authenticates each submission by Python object identity, and joins the exact occurrence, node, target, subject/evidence ports and artifact invocation/key/version. It checks captured environment against the declaration. Coverage and consumed ports come from the production; subject, evidence and consumed refs come from exact port rows. A copied assessment with identical fields fails authentication.
2. `classify_validity` compares only those derived subject/consumed refs and captured absence, configuration and state dependencies with explicit current revision rows. Equal keys and values are current; omitted keys are unknown; unequal values at the same key are stale. Stale precedes unknown. Configuration comparison is type-sensitive. An unconsumed evidence-output identity is not independently a currentness dependency; the retained integrity join still requires its exact output artifact. The legacy evidence-output/consumption coupling remains a separate integration obligation.
3. `evaluate` requires a current exact selected final candidate plus current satisfied evidence matching meaning, consumed ports and complete coverage for each requirement. `enumerate_release` enumerates subsets of the three targets and takes the union of all locally eligible subsets closed under prerequisite implication and whole atomic groups. Each constraint preserves unions, so their feasible union is the greatest release set; this does not assume a product-supplied eligibility verdict.

Required decisions derive from exact final-producer ancestry and supporting evidence's consumed decision refs. The ancestry predicate enumerates bounded paths through raw provenance, checks each parent tuple against its separately declared source relation, and retains distinct occurrence keys even when an alias forwards the same artifact identity. It rejects missing/duplicate joins or cyclic ancestry and excludes unrelated provisional decisions. Wholly withheld targets contribute no decisions. This does not implement A3 continuity support closure.

Neutral identity tokens are opaque symbols. Ref equality preserves `(invocation,key,version)` and occurrence equality preserves `(invocation,target,node,ordinal)`; no text, ordinal-only or key-only matching substitutes for these tuples. `candidate_owners`, `selected_outputs` and `declared_parents` are neutral declaration relations, not proposed new SDK fields. They stand for separately admitted ownership, selected interface binding and dependency declarations. Product integration must construct corresponding facts from actual owners without inventing these fields on existing retained SDK types.

The JSON schema follows the typed source: `Raw` and `Assessment` have named fields; NamedTuple rows preserve their declared field order. Submission entries are indices into the retained assessment list, preserving exact object references on reconstruction. Negative copy/foreign-join tests construct actual distinct Python objects locally; they are not encoded as successful corpus submissions. Result arrays follow `Result(verified, eligible, released, decisions)`, with `Verified` field order declared in the source. Identity normalization must not erase any row coordinate.

## Finite alphabet and counts

The universe always has three targets A/B/C. It is not a multiplied generator axis. Each target has one successful `ok` occurrence, one privacy requirement and one retained assessment. A's consumed ports select XA-only, XB-only, XA+XB or decision D; B and C consume their own XB/XC. Assessment subject candidates A/B/C remain separate from these contextual dependencies.

The generator crosses **4 consumption selections × 5 topology templates × 7 revision axes × 3 revision states × 4 submission selections = 1,680 cases**, P7E-00000 through P7E-01679. The seven revision axes are subject A, consumed XB, decision D, absence QA, configuration NA, state SA and unrelated artifact Z. States are unchanged, changed or omitted. Submission selection is all, A-only, B-only or none; retained assessments remain present regardless of submission selection.

The five templates are independent, A→B→C, A→B with atomic {B,C}, B→A, and independent datum targets with A's final provenance forwarding through an identity alias to decision D. They form **4 distinct datum/atomic graphs** and **20 complete declaration graphs** when consumption declarations and provenance are included. Each case is one raw snapshot/qualification trace, not an execution simulator or a schedule. The manifest records **1,680 traces and 99,720 raw event/fact rows**. This event-row count sums ports, terminals, retained assessments, ordered submissions, artifacts, finals, provenance and current revisions across cases; it does not count runtime dispatches or declaration rows.

Bounds are three targets, three assessments, at most three submissions, two versions per artifact key, 11 baseline artifacts (12 when a changed version is retained), 13 ports, six provenance rows and 20 current revision rows. There are no dynamic memberships. The model uses one success outcome, fixed subject/evidence/result port names, one promise per occurrence, one requirement per target and coverage atoms K0/K1. Small malformed fixtures and a sixth atomic-then-datum graph are separately calibrated in tests; they are not included in generated graph/case/event counts.

## Calibration and reproduction

Literal known answers establish all-target release, B-only release after selective submission, no release without submissions, XA/XB selective invalidation, subject/absence/configuration/state currentness, stale-before-unknown, unrelated revision independence, incomplete coverage, unsatisfied/unknown findings, both directed dependency directions, datum→atomic and atomic→datum withholding, transitive identity-alias ancestry, consumed decisions and exclusion of withheld/unrelated decisions.

Exact-join negatives change each occurrence coordinate, artifact owner/key/version, subject port name, target ownership, final producer and declared provenance parent. Twelve source-semantic mutants compile and execute, then fail behavioral assertions: copied-fact acceptance, occurrence-insensitive joins, ignored declared parents, stale-as-unknown, ignored absence, ignored coverage, ignored findings, ignored dependencies, ignored atomicity, lost ancestry, included unrelated decisions and lost consumed decisions. These calibrate the reference; they do not establish product mutation resistance.

From this directory:

```sh
/root/dev/wt/Anonymizer-graph-sdk-contracts/.venv/bin/python -m unittest -v p7_reference.test_model
/root/dev/wt/Anonymizer-graph-sdk-contracts/.venv/bin/python -m p7_reference.fixtures regenerated
/root/dev/wt/Anonymizer-graph-sdk-contracts/.venv/bin/ruff check --config /root/dev/wt/Anonymizer-graph-sdk-contracts/ruff.toml p7_reference/*.py
/root/dev/wt/Anonymizer-graph-sdk-contracts/.venv/bin/ruff format --check --config /root/dev/wt/Anonymizer-graph-sdk-contracts/ruff.toml p7_reference/*.py
/root/dev/wt/Anonymizer-graph-sdk-contracts/.venv/bin/ty check --project /root/dev/wt/Anonymizer-graph-sdk-contracts --extra-search-path . --error-on-warning p7_reference/*.py
```

`manifest.json` binds source/test and corpus hashes. `evidence.json` retains command results and two fresh CLI regeneration comparisons. `frozen.json` binds the complete review bundle. Source, generator and tests count together; this is one 279-line predicate module, not a second production-shaped interpreter. The generator and calibration code are included in the combined size measurement.

## Draft mapping and limits

`coverage-draft.json` preserves every historical ID and status. Fifty-one rows identify a relevant predicate; all 372 retain an explicit integration obligation. No row is accepted, superseded or closed, and no product selector/candidate/verdict is invented. Even mapped rows still require their exact old discriminator, real constructible owner boundary, typed error/status projection and actual graph witness. The two latest-binding supplement obligations and all unmapped dynamic/provenance/boundary families remain with the campaign; this bundle does not replace their inventory.

The model assumes admitted declarations and a reconciled, closed successful static execution shape. It does not implement public outer-type/count/error precedence, all duplicate/foreign view checks, preparation/execution identity authentication, full retained unsubmitted-fact validation, optional/interface binding projection, intermediate-candidate generation, failed/missing final outputs, request/cleanup localization, selected/unselected choices, maps/loops/subgraphs, final-loop/member provenance, dynamic assessment completeness, bounded parser behavior, callback science, canonical SDK ordering or record/status projections. It checks supported raw-join faults but is not a general malformed-SDK-record validator. Absence/configuration/state current rows are explicit claims, not observations of an external world. Opaque SDK object factories and all those boundaries require product integration tests.

The only commutation claim exercised is independent state-SA and absence-QB revision updates; parent/child provenance relations remain causal. No full event permutation or interleaving theorem is claimed. Feasible-subset union closure applies to P7 datum/atomic constraints only and is not evidence for A3 continuity decision support.

Ownership: stay; owner=p7_reference.model; evidence=raw declaration/fact joins, currentness comparison and finite subset/path predicates with no SDK imports; reason=independently authored specification model.

Ownership: stay; owner=p7_reference.fixtures and p7_reference.test_model; evidence=closed declarative alphabet and literal known answers; reason=reference generation/calibration without product-derived expectations.
