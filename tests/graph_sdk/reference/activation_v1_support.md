<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Activation reference static-support expansion

This document is normative for translating `activation_v1_cases.json` to the
adopted D03 and D04 public boundaries. It defines static workflow support only;
it does not contribute activations, events, counters, capacity, state, or
completion obligations. An adapter must construct these declarations through
the real public constructors and admission functions. It must not infer an
expected error or fabricate an admitted workflow after admission rejects.

## Identity and common declarations

A template identity is `(scope, template)`. `scope` is the ordered activation
key path serialized on every seed. The empty path is the top-level workflow.
A subgraph body's path is its parent's scope followed by the parent activation
key. Renaming rewrites every key in the path as well as every template and
activation field. Equal labels in different scopes create distinct D03
`WorkflowId` and `NodeId` values. Distinct ordinary nodes in one scope always
have distinct labels. Repeated map or loop member reservations refer to one
declared member node in their scope.

Each scope has an interface outcome named `ok` and, where exercised, `fail`,
`again`, and `stop`, with the category and `produced_ports` stated by the
neutral outcome records. The artifact type is `reference-artifact` revision 1.
An operation declares output `result` when any outcome can produce `result`,
output `carry` when any outcome can produce `carry`, and input `input` when it
is the destination of an `input_dependencies`, initial, or carried binding.
Every output has an `OutputDependency` on `input` when that input exists and an
empty dependency otherwise; identity input is always null. Ceilings are the
smallest nonnegative values that admit the declaration. Context, evidence,
state effects, model requirements, protection requirements, choices not named
by the case, and unused interface ports are empty.

An occurrence edge in `edges` becomes a D03 `SequenceEdge` between the two
scoped template identities. An edge also gets `before.result -> after.input`
only when the occurrence pair appears in `input_dependencies`. Thus the common
sink in the independent-sibling family has two ordering edges and no data
dependency. Outcome bindings cover every reachable sink outcome exactly once.
Limits are the exact structural counts produced by this expansion.

Malformed reservation and event cases use the valid support workflow of their
family's nearest base case. Reservation mutations never alter that admitted
support. This separation is required: foreign, duplicate, missing-parent,
wrong-iteration, and event defects reach initialization or transition rather
than being moved to D03 admission. The `boundary` field selects the real public
boundary to call.

The exact support selection for negatives is:

| Cases | Valid support constructed before the mutation |
| --- | --- |
| `sequence_mutations/*` | single top-level N0 workflow |
| choice coordinates 004-007 | selector N0 with N1/N2 branches |
| subgraph coordinates 004-007 and 009 | top-level N0 subgraph with one-node `[A0]`/N1 body |
| map coordinates 012-024 | bound-two N0/N1/N2 map support |
| join coordinates 021-025 | bound-two N0/N1/N2 map-and-join support |
| loop coordinates 008-016 | bound-two N0/N1/N2 loop support; coordinates 008 and 009 then omit only the named D04 binding |
| nested coordinates 009-010 | the ordinary bound-two map or loop support named by the case |
| precedence 000 | bound-two map support before constructing the malformed event |
| precedence 001 and 003 | single N0 support |
| precedence 002 | valid N0 -> N1 support; only reservation ownership/key facts are mutated |
| precedence 004 | valid bound-one map support; only required reservation and parent-role facts are mutated |
| terminal coverage coordinates 012-014 | bound-two N0/N1/N2 map support |

Precedence coordinates 005 and 006 intentionally have no admitted support;
their complete malformed D03 declarations are specified below and must be sent
to static admission.

## Family expansion

- `sequence_single` and `sequence_mutations` declare top-level N0. Linked pairs
  declare N0 -> N1 and the matching result-to-input binding. Independent
  siblings declare N0 -> N2 and N1 -> N2, with no input bindings; N2 is the
  sole sink. Mutation cases use the single-node support even when their
  reservation or event names an absent key.
- `choice` declares top-level N0 as selector, N1 and N2 as branch members, the
  recorded choice branches, and selector-before-member sequence edges. The
  foreign, unknown, select-both, and abnormal cases reuse this valid support.
- `subgraph` declares top-level N0 as a `SubgraphNode`. Its body scope is
  `[A0]`. A one-node body declares N1. A two-node body declares N1 -> N2 with
  N2 as sole sink. Nested bodies repeat this rule at `[A0,A1]`. The parent's
  black-box operation matches the admitted body interface. Reservation
  negatives reuse the corresponding valid one-node support.
- `map` and `join` declare top-level N0 -> N1 -> N2. N0 is the expander, N1 the
  member template, and N2 the keyed join. Bound zero still declares N1 in D03;
  it creates no member occurrence. D04 map declarations and accepted child
  categories come directly from `aggregates`. Map mutations reuse this valid
  support and change only the stated reservation or event input.
- `loop` declares top-level N0 -> N1 -> N2. N0 is the starter, N1 the member,
  and N2 the join. N1.input has a real static workflow-input binding. A present
  `initial_binding` maps that workflow input to N1.input. A present
  `carried_binding` maps N1.carry to N1.input for the next iteration. Null means
  that exact D04 declaration is omitted and must reject at `dynamic_admission`.
  In `missing_carried_output`, both bindings are present, `again` does not
  produce `carry`, and `stop` does; the failure therefore occurs at runtime.
- `nested_map_loop` uses top-level N0/N1/N2 for the map support. Each observed
  N1 map member is a distinct subgraph occurrence but refers to the top-level
  N1 template. Its body scope `[child]` independently declares loop support
  N0 -> N1 -> N2 with the explicit initial and carried bindings. No node or edge
  is shared across those body workflows. The `(2,2)` case consequently has 12
  activation occurrences while using three labels per scope.
- `precedence` cases 000 through 004 use valid support as described above and
  reach event construction, transition, or initialization according to their
  `boundary`. Case 005 constructs two overlapping D03 choice branches plus the
  recorded sequence cycle and calls `admit_static_workflow`. Case 006 constructs
  the recorded sequence cycle plus one result-to-input binding whose source and
  destination use different artifact revisions, then calls
  `admit_static_workflow`. No dynamic admitted object is created for either
  static negative.
- `terminal_coverage` uses the same valid map support as `map`; only terminal
  observations differ.

The adapter may allocate opaque identities and translate neutral names, but
must preserve scope, declaration topology, ports, bindings, outcomes, case
boundary, event order, and every expected state field. Support nodes are static
declarations. They are never silently added to activation selection, capacity,
event counts, or completion.
