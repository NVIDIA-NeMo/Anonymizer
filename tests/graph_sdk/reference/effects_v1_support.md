<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Effects v1 reference support

This reference implements the adopted D06/D07 v4 contract identified by
SHA-256 `9b0ab07b8c0212ffd954dc37eb37540141da6899753fc778ad26238494aeaeca`.
It is pure Python under `tests/graph_sdk/reference`, imports no product graph
SDK module, and derives every committed expectation by reducing the committed
declaration and event sequence.

The finite identity universe is two semantic tasks (`T0`, `T1`), two physical
requests plus one follow-up (`R0`, `R1`, `R2`), two binding declarations (`D0`, `D1`), two sources
(`S0`, `S1`), two implementation policies (`P0`, `P1`), two workflow nodes,
two artifacts, two waits, and one resource. Hard budgets are `0`, `1`, and `2`;
request attempt bounds are `1`, `2`, and `3`.

The generator enumerates these declared products and named boundary families:

- budgets cross `0/1/2` with zero, partial, exact, one-over, and cancelled
  reservation outcomes;
- keyed requests cover `T0`, `T1`, and `{T0,T1}`, reordered success, and each
  missing, duplicate, extra, and foreign association defect;
- retry paths cross the three replay levels with their boundary failure class,
  plus correction, repair, failover, cross-policy use, cumulative attempt
  exhaustion, mixed exhausted/eligible sharing, graph-budget denial, and a
  cross-association predecessor negative. Retry, correction, and failover use
  the immutable terminal of the latest dispatched request for each exact
  association and its bound policy. A prior failure cannot authorize another
  attempt while the latest request is pending or after it succeeded. Their
  permitted failure classes are separate, with cross-purpose and
  late-conflicting-terminal negatives;
- failover also crosses replay authority: an eligible failure class still
  requires idempotent replay, or before-acceptance replay when that exact
  predecessor was rejected before acceptance;
- request races enumerate success and failure with settlement before and after,
  cancel before result, result before cancel, trusted stop, unknown stop/Lost,
  late result, identical and conflicting terminal, identical and conflicting
  settlement, and scope cancellation;
- binding covers required/optional failure, reordered sources, omitted optional,
  oversize rejection, pre/post-dispatch cancellation, two sources returning the
  same key/version, and two declarations sharing one source and resource while
  retaining one request per declaration. Result and failure events name the
  dispatched one-declaration request; unsolicited, wrong-source, empty,
  duplicate, and foreign-association results are separate negatives;
- cleanup crosses caller/SDK ownership, local and remote outstanding work,
  forbidden and independently safe detachment, trusted stop, close failure,
  and close uncertainty;
- bridges enumerate every runtime condition, all six failure classes, abnormal
  `None`, named result, and the no-attempt unstarted closure. Each condition is
  resolved through an admitted mapping rather than through event-provided
  outcome/category values. The shared two-task request is integrated with both
  reordered success and every keyed-result defect before both bridge emits;
  the task/request causal link remains valid when bridge start precedes or
  follows dispatch. The immutable request fact, including a task's reported
  result, must match the observed bridge condition;
- decisions cover matching resume while unrelated work advances, deadline,
  cancellation, implementation failure, and foreign, stale, unknown, duplicate,
  and late identity classes;
- admission negatives cover outer type and aggregate precedence, member/value,
  owner, duplicate, missing, unsupported, contradictory, outcome mapping,
  category, bounded catalog, identical failover policy, and one-implementation
  local/decision rules.

Admission also enumerates the exact local, external, and decision condition
products. Runtime mapping keys are unique triples of condition, reported
outcome, and failure, every row has the exact closed fields, and missing,
duplicate, extra, unknown, and category-incompatible mutations reject before
effects.

The admission cases contain the malformed declaration itself. There is no
fixture-only expected-error field or reducer shortcut. Settlement disposition,
remote-stop certainty, and exact-or-unknown usage form one closed grammar:
completed/rejected/stopped require trusted remote stop, while unknown retains
remote uncertainty; trusted cancellation stop also carries usage.
`remote_stopped` is exactly `bool | None`. Late generic results/failures and
binding results/failures after Lost preserve the first terminal and remote
uncertainty; only a compatible trusted settlement can clear it. Late binding
responses after Lost or cancellation create no artifacts or binding state.
A whole oversize binding response still closes
the charged physical request while failing the binding without truncation.

## Materialization extension

The extension implements the adopted context-materialization addendum identified
by SHA-256
`b1a5651ee2649b01c89e80bd1442e3f03209292b48ce846d27714698f57eb07c`.
The first 156 cases, 161 traces, and 679 events are the accepted R2 corpus in
their original order and bytes. Its corpus SHA-256 is
`c62f2cc7e7237ea030451ac8d35a3c30b39f71766b4275348597949d560a7a6a`.

The added finite universe has initial and adaptive paths, single and collection
shapes, collection sizes `0/1/2/3/4` against an admitted maximum of three, two
nodes sharing the port name `context`, a scalar selector root, and bounded UTF-8 values.
Single results enumerate exact-one success and multiple-result rejection.
Collection results enumerate one, two, exact-three, one-over-four, empty,
duplicate key/version, wrong scalar type, nested value, and reversed input
order. Reversed input is normalized by ascending numeric `(key, version)`.

Admission derives one artifact-type schema from all initial and adaptive
declarations. It rejects single/output mismatch, a single maximum other than
one, empty or over-ceiling collection bounds, a scalar output reused as a
collection, conflicting item types for one collection type, a collection type
used as an item type, a collection-typed root input, duplicate declaration
identity, and binding identity on an adaptive declaration. These are mutated
declarations evaluated by the admission function; there is no fixture-provided
error label.

Every result is tied to one dispatched physical request and its exact admitted
association. Initial collections retain `N` scalar items plus one collection
artifact, count each item byte once in the scalar and once in the collection,
and record `N` edges from one scoped `InitialCollectionKey` to exact
`BoundInputKey(target,node,port,declaration,key,version)` parents. Each scalar
item retains its item type and an empty-parent root fact. Adaptive traces first
retain and count the selector root through a separate `root_input` event.
The materialization adds exactly one output and its bytes, using
`OperationOutputKey(activation,target,port)` with the declared existing
`RootInputKey(target,port)` as parent; it never introduces a binding identity.
Missing, foreign, and invented parents reject. A late result after Lost or
trusted cancellation cannot create output, replace the first terminal, or clear
remote uncertainty. Scalar materialization succeeds with zero collection
capacity. Adaptive success retains the exact reported request outcome, and an
integrated bridge witness closes the semantic task through that fact. Whole-response rejection occurs before any materialization
state is committed. Separate witnesses exercise collection, artifact-count,
logical-byte, and provenance-edge ceilings and two nodes with the same port.
The dispatched-attribution addendum changes no neutral R2 event: adaptive
results use the association already recorded by reserve and dispatch.

Only causal sequences are emitted. Policy binding precedes reservation;
reservation precedes dispatch; dispatch precedes terminal and settlement;
trusted stop follows cancellation; decision submit/deadline follows open; and
cleanup follows the last locally known use. The alternate traces commute only
orthogonal terminal and settlement observations. Static invalidity is recorded
at admission and is never disguised as a runtime provider event.
