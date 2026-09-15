<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Phase 10 Bounded Inspection Closeout

The private Phase 10 implementation and conformance evidence completed on
2026-09-15 on `codex/anonymizer-bounded-inspection-p11`, through commit
`3ed8d02417737a6363d5363494c9e182762f8458`. Two independent council reviewers
accepted those exact implementation and test bytes, each at 0 Critical,
0 Warning, and 0 Nit. This record supports branch integration review; integration
and public API publication retain separate approval gates.

## Implemented Scope

The private module `src/anonymizer/engine/execution/phase10_inspection.py`
provides grant-authorized, fixed-arity explain, inspect, and diagnose operations.
Owner capture reducers produce bounded detached summaries. They read the
inspection limit table at execution time and reject construction overages before
publishing a view. Encoding measures detached payloads before admitting their
schema, requires the complete ordered declared-limit table, and returns either
complete canonical JSON or a cause-free rejection.

The conformance evidence includes independent expectations for 25 real
owner-operation witnesses, deterministic permutations and four hash seeds,
retention and non-interference probes, and designated mutation assertions.
The source-built wheel omission test applies the same acceptance witness to
baseline and omitted-resource wheels.

Public exports, CLI behavior, P9 result compatibility, telemetry schema, and
protection execution routing remain unchanged by the Phase 10 remediation.
This checkpoint makes no claim of stable cross-process provenance or portable
graph/session artifacts.

## Commit and Contract Pins

| Boundary | Commit |
| --- | --- |
| Remediation base | `26b147268d9184b7542936d3193cf9be1cfb628b` |
| Tier 1 contract correction | `8c2a2f7ec92634baa3644f39b4271628d53162bf` |
| Tier 2 enforcement correction | `bfe73562dc21da1b70de2b3282d0fe87f35fd004` |
| Tier 3 conformance evidence | `3ed8d02417737a6363d5363494c9e182762f8458` |

The three remediation commits carry verified Git signatures and DCO signoffs.
The original Tier 2 commit `ed461f4dcbf83ef8887484304a4f30dc72fb2ad1` was replaced
by the Tier 2 commit above; it is not an additional integration input.

The owner contract at
`src/anonymizer/engine/execution/phase10_bounded_inspection_contract.json`
retains version `anonymizer-phase10-bounded-inspection/v1`:

- Contract-member SHA-256: `0d6e189bf3d89472a6880a76367ed99b5462b6c5d460818282e403e4a285eb95`.
- Raw-resource SHA-256: `1b2bf397cfaed7d74b4ec2e0bb5db428d59a9ea24679faa7e2056705f00528bc`.

The frozen manifest at
`tests/engine/execution/phase10_reference_manifest.json` records 207 traces, 1,055 events, 83 production mutation instances, 31 reference
mutation instances, and 20 stable mutation classes. The canonical corpus SHA-256
is `4712ce2edbf95dc9cf998be5feb193394e5f4075c836205db51957f1126e3148`.

## Validation at the Tier 3 Checkpoint

These are completed local runs against the Tier 3 candidate. They do not report
remote CI or acceptance of a later integration result.

| Command or check | Result |
| --- | --- |
| `uv run --frozen pytest -q tests/engine/execution/test_phase10_*.py` | 851 passed |
| `uv run --frozen pytest -q tests/engine/execution/test_phase8_*.py tests/engine/execution/test_phase10_*.py` | 1,189 passed |
| `uv run --frozen pytest -q` | 5,600 passed, 11 skipped |
| `make check` | Passed |
| Fresh dependency-installed wheel | Expected contract and raw-resource digests admitted |
| Baseline versus P11 CLI help | Byte-identical |
| P9 result contract | Unchanged approved digest |

The full suite emitted one W&B dependency deprecation warning, also observed
before the final candidate. The independent council verdicts describe review
findings separately from that test-run warning.

The two final reviews bound the seven-file Tier 3 binary diff to SHA-256
`85dfc381db8cd566f5bbb1872fa331e4e6020c4926b4df464959dd628ee30d05`.
Final review also closed two test-evidence gaps: admission-view oracle domains
and full-encoder tests with lowered measurement ceilings. Twelve encoder cases
cover four ceilings across all three view kinds.

On the originating worktree, `.agent-work/p11/` retains the validation logs,
per-file hashes, review reports, and recovery evidence. That ignored temporary
directory is not distributed with this tracked record; transfer its evidence
bundle separately when another reviewer needs the original run logs. The
tracked tests and manifest provide the reproducible conformance inputs.

## Remaining Gates

Before a push, PR, or integration action, the operator must identify the candidate
commit, target repository and branch, target base commit, and permitted operation.
Recheck ancestry and the reviewed diff against that target; the remediation base
above is an evidence boundary, not an assumed integration target.

The next SDK development phase is [Phase 11: Lifecycle and Independent Runtime](graph-native-anonymizer-sdk-rfc.md#phase-11-lifecycle-and-independent-runtime).
It requires process-backed lifecycle evidence and the agreed conformance subset
on a materially different semantic runtime. The Python host supplies lifecycle
evidence only. That work requires its own scoped authorization.

[Phase 12](graph-native-anonymizer-sdk-rfc.md#phase-12-public-surface-qualification-and-publication)
starts only after reviewers accept both Phase 10 private inspection evidence and
Phase 11 lifecycle and independent-runtime evidence. Public graph, session, and
inspection APIs require a new owner contract, an exact reviewed plan, explicit
public-API authorization, and the named promotion gates.
