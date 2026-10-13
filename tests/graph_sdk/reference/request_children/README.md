<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Child request contract: first reducer unit

This is a pure contract fixture. No delegated execution, registered child activation,
DD integration, or qualification support is enabled by these records. The next unit
must independently verify retained child receipts against the admitted execution
plan and canonical activation record.

A request owner binds chains before requests for their associations begin. Each
chain declares its role, ordered fallback routes, parser-attempt bounds, and allowed
follow-up sources. Chain count is bounded by the admitted physical attempt limit
per association; declaring a chain consumes no request. Ordinary requests continue
to use the existing rules. An association bound to a child contract cannot evade it
by omitting child provenance.

A completed transport settlement creates no acceptance terminal. Parser rejection
is the first `malformed_response` terminal. Accepted parser evidence then permits
correction or a declared conversation restart. Exhaustion evidence requires every
admitted parser attempt on that route to have rejected; failover can then select
only the next declared route. The failed terminal remains unchanged. Retries retain
the ordinary failure-class and replay restrictions. Provider-error exhaustion and
other failover policies are outside this first contract.

Interleaved chains use exact predecessors, not the latest request sharing a parent
association. Child role, source kind, predecessor, and evidence persist in the
request receipt. They do not certify that an activation exists. An initial label
cannot restart a dispatched chain. An unused reservation may be cancelled without
a dispatch charge; retrying initial admission after such cancellation uses a fresh
request ID. Shared child reservations stop as a whole when any associated task's
attempt budget is exhausted.

The independent reference recognizes prefixes of two explicit words:
`IPCPEV` and `IPRPEV`. It imports no production code and implements no ledger.

| Letter | Event macro |
|---|---|
| I | Initial dispatch, completed settlement, malformed first terminal |
| P | Accept parser-rejection evidence for that request |
| C | Correction dispatch, completed settlement, malformed first terminal |
| R | Conversation-restart dispatch, same settlement and first terminal |
| E | Accept generation-exhaustion evidence |
| V | Declared fallback dispatch, completed settlement, successful first terminal |

The first route permits two parser attempts; the fallback permits one. The alphabet
is `IPCREV`, with no independent events within this single chain. Corpus version 2
retains all 11 valid prefixes and the 24 first-invalid prefixes of length at most
four. The generator derives both sets directly from the two complete words,
without production imports. `manifest.json` freezes these 35 cases before product
comparison: length at most six and at most three dispatched requests. Its SHA-256
covers the compact JSON encoding of the sorted case tuple.

Version 1 enumerated all 1,555 words of length zero through four plus four longer
valid prefixes. Every removed word maps to its shortest invalid prefix, since the
test stops at that rejection and never executes the remaining suffix. All valid
words remain. The previous generator and manifest are retained in Git revision
`5b921b15e327dbc6177732fcbf75283b64303ea5`; the new manifest records its corpus
digest. This reduction applies only to the fixed three-request budget and
two-attempt first route. Seven literal controls vary the request and parser bounds;
they do not use this fixed-language oracle.

Every retained trace checks dispatch, terminal, settlement and accepted-evidence
counts. Rejected traces must fail at the first illegal transition without changing
request state; earlier accepted terminals remain visible. Separate tests still
cover foreign owners, interleaved chains, evidence forgery, binding order/count
bounds, undeclared routes and source kinds, budgets, unused reservations,
conflicting support, and immutable terminals. The change reduces repeated
executions, not the amount of product functionality or the maintained Python size.

The existing ordinary reference oracles and their case sets remain unchanged.
