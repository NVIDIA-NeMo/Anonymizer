# P9 finite reference, attempt 457

This reference is contract-derived and frozen before product observation/inspection consumers. It imports only the standard library and its local reference modules; tests additionally use pytest. No product implementation or production encoder supplied expected results. Historical P7 implementation exposure during retirement work is disclosed. The current task read only P9 v3, PLAN, the parent owner-seam report and independently approved bound-count clarification. No product files, campaign counters or acceptance records change.

Ownership: stay; owner=reference.py; evidence=typed neutral facts and pure bounded predicates; reason=independent reference, without callbacks, an executor, owner joins or product state. Cases and tests are separate from the predicate owner. All helpers and source lines are counted in the manifest.

## Neutral scope and counting

The base observation alphabet uses execution, operation, physical_request, cleanup, qualification and release; all eight outcomes and seventeen closed reasons are independently varied. Full v3 boundary constants exist only to validate the closed schema. Correlations are 0/1/2 in focused controls, with parent absent/0/1/foreign2. Raw finite Pair facts have clock readings and explicit presence of each half; closed PublicPair values have only ordinal, parent ordinal, closed enums, closed reason tuples and elapsed duration. Raw clock readings, owner tokens, private work IDs and callback objects do not enter the receipt or sink projection. Receipt pairs are presented in entry-ordinal order; this is the neutral adapter's canonical presentation, not a claim that real nested boundaries finish in entry order.

Observation phases carry a whole compiled ceiling. The independent forecast is fixed wrappers + two pairs per activation slot + admitted adaptive requests + cumulative physical request ceiling + decision waits + cleanup. Every pair reserves exactly two events. Requirements are calculated from the worst-case closed receipt grammar, including all reasons, widest admitted integers and the optional measurement object's full fields. Actual generated measurement is absent; real measurement handle/route validation is an integration obligation. Repeated phase envelope overhead is conservatively reserved. This is a sufficient finite upper bound, not a minimality claim or the actual production compiler's count table.

The reviewed common codec convention applies: each JSON container or scalar counts as one node; keys do not count; root depth is 1 and each child value adds 1; null, bool and empty containers count as one. `requirements` and `encode` use that same convention. Byte maxima use fixed schema keys, bounded enum strings and decimal widths; nullable parent ordinals reserve whichever is wider, null or the maximum integer. No content is inspected to forecast storage. Tests cover exact and one-less pairs/events/nodes/depth/bytes/duration, decimal-width transitions and domination of every finite accepted encoding by its forecast.

The finite phase reducer checks all cumulative ceilings before entering a measured phase. Even an empty actual phase fails observation when its declared whole ceiling cannot fit. No lazy pair funding is permitted. Underforecast, foreign producer, invalid pairing, clock reversal or elapsed overflow yields a closed failure rather than a partial receipt. The reducer models already-completed neutral facts; it does not execute semantic work. Product noninterference, automatic owner hooks and `finally` closure are real integration obligations.

Sink failure is represented by a finite failure index, not an arbitrary callback. Only complete PublicPair values are delivered. Failure at zero or after one delivered pair stops later delivery but leaves the complete internal receipt. Cached-close identity and source ownership are a predicate on exact opaque object identity: a copied equal receipt, foreign owner and None/None match reject. The model does not implement or purport to test a live session's caching behavior.

Request scope facts list exact distinct physical dispatches and typed purposes. The seven purposes are initial, initial_binding, adaptive_retrieval, retry, correction, repair and failover. A missing owning receipt makes total and usage unknown; affected per-purpose fields are unknown according to the scope's explicitly supplied admissible purpose set. Complete other purpose scopes retain exact counts. This possible-purpose set must come from authoritative owner facts, never names, lifecycle events or inferred non-entry. An empty exact receipt proves zero only for its own phase. Any missing or unknown usage makes both aggregate unit fields unknown; unknown-request count is exact only for complete scope. Sharing semantic associations never multiplies the one physical dispatch row. Duplicate physical ownership rejects rather than silently deduplicating two receipts.

No fixture acceptance boolean or expected error feeds a predicate. Corpus input facts and calculated expectations occupy separate JSON fields. Literal known answers calibrate seven-purpose totals/usage, missing ownership, whole-phase reservation, pairing, privacy and encoding. The whole corpus is regenerated deterministically, without product comparison.

## Retained family mapping

| v3 family | Executed reference discriminator | Actual integration still required |
|---|---|---|
| 1 plan-only | exact/unknown planned and conditional; known-upper ceiling; supported/unsupported/unknown control tuples | Derive from actual prepared grammar and admitted capability paths |
| 2 complete lifecycle | independent six boundaries/eight outcomes; nested parent; exactly two halves | Automatic owner hooks across every supported entry point and `finally` closure |
| 3 failure paths | all seventeen closed reasons; negative clock/pair/producer paths | Actual callback/provider/cancellation/cleanup/qualification/continuity mapping without exception text |
| 4 requests/usage | every purpose, all seven together, two owners, unknown and partial usage, missing binding | Real shared associations, retry/failover/repair and hard-budget stop receipts |
| 5 limits | whole phase and closed encoding exact/one-less; root counting; integer widths | Production pre-allocation bounds and measured compiler ceilings, no content-dependent forecast |
| 6 malformed | absent half, duplicate/foreign correlation, mismatched member, version, clock, delivery totals | Actual immutable owner receipt provenance and full diagnostic precedence |
| 7 sink failure | failure before delivery/after one pair; internal pairs retained | Real exception-throwing sink, disabled later calls, unchanged semantic result |
| 8 canary inputs | hostile reason, arbitrary encoder mapping, opaque owner with exploding repr | Place every specified content category in actual graph inputs and private owners |
| 9 privacy channels | traverse closed dataclass fields via asdict; exact encoded key sets; sink PublicPairs | SDK-owned logs/errors and controlled fixture logs; host logging explicitly outside guarantee |
| 10 qualification | met/unmet/unknown/not_assessed count tuple | Actual P7 criteria proxy, withheld rows and verified evidence counts |
| 11 pure repeated inspection | repeated encoding equality; pure predicates with no service imports | Actual repeated explain/inspect/diagnose, zero provider/callback/qualification/continuity/projection calls |
| 12 early binding | exact binding-purpose scope versus missing receipt and owner mismatch | Actual required binding failure/cancellation/Lost before preparation with retained resource facts |

The receipt encoder is independent and closed, but covers only ObservationReceipt and ObservationFailure, with measurement absent in concrete receipts. It is not a second full InspectionValue encoder. Other summary formats, full resource/route/terminal proxies and error precedence remain integration obligations. The finite encoder builds its tiny tree before checking it; it does not prove production pre-allocation behavior. Forecasts reserve optional measurements conservatively without implementing their callback. No real SDK log, public canary or dependency logger coverage is claimed here. No provider retention, durable telemetry or host callback sandbox claim is made.

Independence: outcome, reason, route/control, count-width, purpose and boundary axes are varied separately, with selected causal combinations for nested pairs, underfunded second phases, missing binding with retained execution, and mixed known/unknown usage. Whole-session reservation and ordinals couple phases, so their order is never quotiented. Distinct physical request rows commute under accounting; no traces are suppressed using an unproved independence equivalence.

## Reproduction

From this directory, use the product environment only as an interpreter/tool installation:

```
/root/dev/wt/Anonymizer-graph-sdk-contracts/.venv/bin/python -m pytest -q -o addopts='' test_reference.py
/root/dev/wt/Anonymizer-graph-sdk-contracts/.venv/bin/python cases.py /tmp/p9-a.json
/root/dev/wt/Anonymizer-graph-sdk-contracts/.venv/bin/python cases.py /tmp/p9-b.json
cmp /tmp/p9-a.json /tmp/p9-b.json
cmp corpus.json /tmp/p9-a.json
/root/dev/wt/Anonymizer-graph-sdk-contracts/.venv/bin/ruff check --config /root/dev/wt/Anonymizer-graph-sdk-contracts/pyproject.toml *.py
/root/dev/wt/Anonymizer-graph-sdk-contracts/.venv/bin/ruff format --check --config /root/dev/wt/Anonymizer-graph-sdk-contracts/pyproject.toml *.py
/root/dev/wt/Anonymizer-graph-sdk-contracts/.venv/bin/ty check --project /root/dev/wt/Anonymizer-graph-sdk-contracts --extra-search-path . *.py
```

Source-semantic mutants remove input and output halves, swap a terminal parent, inject a private identity, admit arbitrary/exception reasons, coerce unknown usage to a partial exact sum, double-charge physical or repair dispatches, truncate over-limit encoding, or disable whole-phase reservation. Each must fail a behavioral assertion. Mutation harnesses and generation helpers are included in size accounting. Independent review must approve this exact manifest before comparison; passing reference tests does not close product acceptance rows.

## Successor v2: reviewed pre-consumer depth and envelope amendments

The first manifest and its files remain immutable one directory above; its author report is preserved as `../author-v1.md`. This successor is the consumer candidate. Independent 458 addenda arrived immediately after the first manifest was written, before review or product comparison.

Requirements now take the exact immutable Limits value used to derive closed-schema integer widths and retain it privately. Reservation accepts an equal copied value but rejects a differing envelope before that phase with capacity_contract. Requirement envelopes never enter receipts or encoding. This models the reviewed unshipped builder/session interface; no product output supplied expected values.

Observation depth bounds both logical parent-chain depth (root 1, empty chain 0) and JSON value-tree depth (root 1). Required depth is their maximum. Inspection encoding checks JSON depth only. The bounded alphabet extension is one twelve-pair chain with ordinals 0–11, deeper than its flat JSON receipt. Literal exact-depth and one-less cases distinguish pre-phase forecast rejection from an underforecasted producer's depth defect; encoding the resulting flat receipt at JSON depth 11 remains legal. Source mutants separately remove the logical forecast maximum, actual chain-depth guard and envelope equality guard. All prior pairing, accounting and encoding discriminators remain.

## Successor v3: bounded review 458 corrections

V2 remains immutable in `../v2/`; its author report is preserved there. Review identified two reference defects before any product comparison. First, a forecast could fit pair/event capacity while its possible delivered_pairs count exceeded max_count, producing a receipt the encoder would reject. Whole-phase reservation now conservatively requires cumulative forecast pairs ≤ max_count, the worst possible delivery count. This is a sufficient, nonminimal envelope independent of actual sink delivery. Begin count 0, exact counts 1/2, one-less count and cumulative two-phase controls cover it; already delivered whole pairs survive a later phase rejection.

Second, Python bool/integer equality could admit a terminal ordinal or terminal parent equal to a valid integer. Both terminal fields now require exact integers (or None for absent parent), nonnegative and count-bounded, before equality checks. Literal True near-neighbors and separate source omission mutants calibrate both guards. Seven new corpus cases and three mutants are added; previous neutral pair facts and expected outcomes are unchanged. The corrected generator calculates each requirement with its actual session envelope rather than rebinding a width envelope after construction.
