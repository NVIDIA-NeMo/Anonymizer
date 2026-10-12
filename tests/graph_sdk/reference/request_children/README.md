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
is `IPCREV`. Before comparing production behavior, `manifest.json` freezes all
1,555 words of length zero through four, plus four longer valid prefixes: 1,559
cases, length at most six, at most three dispatched requests on an accepted trace.
Its SHA-256 covers the compact JSON encoding of the sorted case tuple. Accepted
traces must agree on dispatch and terminal counts without defects. Rejected traces
must raise a content-free effect rejection. Separate tests cover foreign owners,
interleaved chains, evidence forgery, binding order/count bounds, undeclared routes
and source kinds, budgets, unused reservations, and immutable terminals.

The existing ordinary reference oracles and their case sets remain unchanged.
