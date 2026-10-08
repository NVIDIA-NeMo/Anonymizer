<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qualification v1 R3 map-item evidence reference extension v8

## Canonical verified-evidence order

V8 authenticates submissions in caller order for duplicate and ownership
errors, then orders the resulting verified tuple by D08's canonical key:
target, activation occurrence, node, evidence port, and artifact ordinal. The
finite model now gives its map occurrences an explicit reservation recipe. For
one map, `MAP`, `M0`, `M1`, `ROOT:A`, and `JOIN` have ordinals 0 through 4.
The second map uses `MAP2=5`, `Z0=6`, and `JOIN2=8`; nested occurrences have a
separate explicit route-local recipe. `two_independent_maps` therefore emits
`M0`, `ROOT:A`, `Z0`, rather than submission order `M0`, `Z0`, `ROOT:A`.

Neutral symbolic identities and runtime identities belong to different plans.
The runtime reservation recipe iterates admitted node storage and fresh typed
IDs need not assign the same numeric occurrence to the same neutral name.
Cross-model comparison must first assert that each raw tuple is ordered by its
own retained numeric ordinals, then compare complete rows after identity
normalization. It cannot require raw symbolic order to coincide across plans.
The reference self-test checks every multi-fact result against its complete
finite canonical key.

`map_item_evidence/member_blocked_unreached` remains a neutral terminal-failure
witness but is now `neutral_only`: its declaration supplies the mapped item and
contains no unavailable prerequisite capable of blocking the member. The
production obligation is a real map member with an additional failed ordinary
input, a retained item owner, a blocked unstarted member, no assessment
callback/fact, and terminal-failure withholding.

V8 also aligns two admission negatives with their earliest public boundary.
`max_fixed_point_steps=0` violates the positive `QualificationLimits` scalar
invariant and is `invalid_value`; it never reaches the later target-count
limit check. A protection requirement whose meaning has no matching projected
promise is rejected during workflow preparation as `protection_ineligible`;
it cannot reach P7 as a missing production. Both cases remain callback-free.

## Retained map-owner validation precedence

V7 corrects six corruption results at the retained provenance owner boundary.
An item key or version absent from the expander's exact retained collection is
`missing`. A `MapItemKey` whose expander or member occurrence is absent is also
`missing`. A key naming a different workflow target is `foreign_owner`.
These checks precede relationship checks because the relationship can be
authenticated only after its owners and collection item exist. A present
non-map producer remains `contradictory`, and an artifact from another
invocation remains `foreign_owner`.

Typed subject and consumed endpoints use this same ordering. In particular, a
consumed endpoint whose captured `MapItemKey` names an absent member occurrence
is `missing`; the neutral evaluator no longer collapses that absence into a
generic endpoint contradiction. The eight corruption witnesses assert this
precedence directly.

The retained `expansion_failed`, `expansion_overflow`, and `expansion_open`
records remain useful finite qualification-state witnesses, but V7 classifies
them as `neutral_only`. The runtime publishes a successful collection and its
membership transition atomically: an empty successful collection closes the
expansion, overflow requires an actually over-limit retained collection, and a
failed expander cannot publish the successful collection consumed by the
candidate operation. Production comparison needs separate records with those
real topologies; these neutral traces do not claim to be constructible P5
records.

V7 also aligns the nested record with public structural projection. A
`SubgraphNode` input is an internal binding used to materialize body operation
inputs; it is not an `ExecutionPortFact` occurrence and does not appear in the
public operation-input-parent inventory. The nested trace therefore retains
the body expander's `context` port and producer plus the wrapper's projected
`nested_members` output and provenance, while omitting the synthetic wrapper
`context` port and input producer.

## Typed admission precedence and route ownership

V6 corrects two typed admission codes to the public owner boundary. A
`ProtectionRequirement` first establishes that all typed subject and consumed
endpoints name one exact map domain. Therefore `invalid_path_owner`, whose
subject path is `MN` while its consumed path is empty, rejects
`contradictory` before either path is validated as a container route.

For `valid_container_wrong_route`, `UNRELATED` is a genuine declared container,
but the retained `map_routes` owner record places the same expander and member
at the root route. Entering the unrelated container changes workflow ownership;
the root-owned expander/member cannot belong to that body. The case therefore
rejects `foreign_owner`, rather than treating the map declaration as merely
missing. The evaluator derives this result by matching the endpoint's map
identity against its retained route owner; it does not switch on a case ID or
stored expected error.

`map_item_evidence/wrong_subject_artifact` remains in the corpus with its
`contradictory` result, but is tagged `neutral_only`. A real
`ExecutionAssessmentFact` has no caller-supplied subject-artifact field; P7
derives that artifact from the admitted promise subject port and its retained
occurrence port. Production comparison must therefore corrupt the actual
retained item owner instead. `map_item_evidence/wrong_item_owner` is the
corresponding production-boundary witness: the member item port/input parent no
longer resolves to its exact `MapItemKey` and rejects `contradictory`. This
mapping preserves the neutral family without adding an SDK field or concealing
the real owner check.

`map_item_evidence/two_maps_cross_owner` has the same representation limit: its
neutral mutation replaces the derived assessment subject artifact. V7 retains
the scenario and result but classifies it as `neutral_only`. A production
witness remains required that changes a real captured member port or
`MapItemKey` to the sibling map while retaining both map owner domains; no
caller-supplied assessment field may stand in for that test.

## Root-input candidate passthrough output role

V5 corrects one retained port role in
`map_item_evidence/candidate_passthrough_unrelated`. Its final workflow output
is bound directly from the root `subject` input. That root input and `N`'s
assessment-subject occurrence are candidates. The separately executed but
unused `N.result` output is neither an assessment subject nor the source of a
`NodeOutputRef` workflow output binding, so its occurrence role is `artifact`.

The result port remains retained with its operation provenance; only its role
changes from v4's incorrect `candidate` to `artifact`. The final candidate is
still the root subject, its ancestry is still unrelated to the membership
producer, and the case continues to withhold `missing_assessment`. No verifier
normalization or product behavior change is implied.

## Blocked keyed joins

V4 corrects the four typed records in which a keyed join cannot start because
its expansion or member obligations are already impossible:
`member_non_success`, `member_blocked_unreached`, `expansion_failed`, and
`expansion_overflow`. Their `JOIN` entry is closed unstarted with category
`blocked` and no outcome. The corresponding nonstructural terminal has no task
attempt, category `blocked`, no outcome, and the canonical `prerequisite`
reason. This is the public D04/P5 projection of `CloseUnstarted`; it replaces
v3's synthetic started failure and `TASK:JOIN`.

The nonempty reason is required by the public `TerminalFact` invariant. An
empty reason set for any non-success terminal is contradictory. These records
continue to contribute `terminal_failure`, retain their independently produced
candidate, and preserve every qualification and withholding result from v3.

## Public P3/P5 record compatibility

V3 retains every v19 object and every v2 case ID while replacing the shared
typed-case occurrence skeleton with publicly constructible records. Flat map
expanders are nonstructural `EXP` operations with real attempts and directly
own their `OperationOutputKey` membership collections. Every map declaration
also retains one exact control-only keyed join declaration: source expander,
distinct join operation, accepted terminal categories, and `all_by_key`
reduction. The join occurrence is present in the same workflow scope as its
source. Successful, failed, and incomplete aggregate paths retain corresponding
join terminal history instead of projecting the control occurrence away.

The ordinary final operation `N` remains separate from the keyed join. It
consumes the accepted membership collection and publishes the scalar candidate.
This permits failure, overflow, open, and member-failure cases to retain their
existing candidate and qualification semantics while the keyed join records the
aggregate control result. No candidate, assessment, or provenance fact is
invented to bridge those roles.

The nested case now retains the full structural boundary. `SG` is an outer
structural occurrence; inner `EXP`, member, and keyed join occurrences belong
to its body membership. The wrapper input and inner body input retain their
actual common source. The wrapper output port/provenance is a structural
projection whose sole parent is the inner `EXP.members` producer, and outer
`N` consumes that structural output. Inner operation provenance is never wired
directly across the wrapper.

The two-map cases retain two distinct operation expanders and two distinct
keyed joins. Their final combiner consumes both collection producers, while
each typed requirement still quantifies only over its own map. All 44 typed
case verdicts, verified facts, withholding codes, currentness behavior,
submission behavior, and deliberate corruptions are unchanged from v2; only
their complete record/declaration scaffolding changes.

## Adopted typed map-item projection

This extension implements the neutral reference semantics of the adopted
`D03-D04-D08-map-item-evidence-proposal-v2.md` while preserving every R3 v19
case object. Typed requirements are an optional declaration surface, absent
from all predecessor cases. Each endpoint names its exact container path,
expander node, member node, member item input, expansion outcome, and membership
output port. A requirement retains one candidate interface output, one inner
promise, and a subject endpoint and/or consumed endpoints from one exact map
domain. Different domains require separate requirements; the reducer never
forms a cross-product.

Admission joins each endpoint to one map input, one operation production, one
admitted candidate output, and one exact `map_routes` record. That record
retains the ordered outer-to-inner container path and its ownership transition
to the named expander/member map domain. Container kind alone is insufficient:
a path made only from valid but unrelated containers rejects. Every typed
declaration, including the previously published flat cases, now carries this
route metadata; declarations without typed requirements remain byte-for-byte
unchanged. Runtime authentication joins the actual member
input port to its exact input-producer and `MapItemKey`, then to the exact
expander activation, membership producer, member occurrence, target,
invocation, item key, and version. A typed subject must retain the artifact
role. Its verified projection is the evidence-owned
`{artifact, producer: MapItemKey}` pair; it is never converted to the final
candidate. Typed consumed inputs retain ordinary artifact references but pass
the same endpoint-owner check.

The final candidate remains independently selected by `candidate_port` and the
ordinary final-output fact. Its transitive provenance must contain the exact
membership-output producer for every matching map occurrence. Evidence from a
sibling member, another map, another target or invocation, or an unrelated
candidate ancestry cannot satisfy the scoped requirement. Each corroborated
successful member needs its own retained fact and submission. Zero actual
members is vacuous for the per-member evidence condition, while candidate
ancestry, final currentness, and ordinary membership accounting still apply.

Map-item currentness uses the item's artifact invocation/key/version in the
artifact revision view. It does not compare the item to the target candidate.
The frozen cases cover zero/one/two members, direct subject and typed consumed
routes, nested path lifting, a valid-container/wrong-route rejection, distinct
outcome/port endpoints, two independent maps, duplicate typed requirements,
missing/foreign/repeated submissions, stale/unknown items, exact owner and
version corruptions (including a consumed-only wrong `MapItemKey` owner),
non-success members, failed/open/overflow expansion, candidate passthrough and
unrelated ancestry, and cross-map substitution.
Existing occurrence, assessment, submission, port, artifact, provenance, and
verified-evidence limits bound the finite model; no fixed-point or implicit
map-product allowance was added.

The adopted compatibility rule for a scalar subject used beside a typed
consumed item is shared projection behavior. Projection first walks an input
subject backward to a workflow input. If that path is absent, it may use one
unique identity-connected produced workflow output. Zero or multiple such
outputs reject. Scalar consumed ports remain backward-only and never use this
output alias. The neutral consumed-only cases retain the ordinary backward
workflow-input form; the approved output-alias fallback is recorded here for
the actual typed-consumed fixture and does not weaken typed item owner checks.

This is an independent neutral model. Product implementation and complete
actual-reference comparison remain separate acceptance obligations, as do P10
public API and documentation review.

## Retained v19 history

## Possible and required assessment owners (v19)

V19 separates retained assessment history from terminal-corroborated
completeness. A possible owner comes from an actual successful nonstructural
operation entry whose node and outcome match an admitted production. A required
owner is a possible owner with the exact matching successful terminal. Duplicate
retained owners reject first; every required owner must be retained; and a
retained owner outside the possible set remains `unsupported`.

An unsubmitted retained fact in `possible - required` is therefore preserved
when its terminal is missing. Membership reconciliation records incomplete
accounting and release remains withheld. Selecting that same fact cannot be
authenticated and rejects `missing`. The root and representable dynamic-member
families each contain direct unsubmitted/submitted missing-terminal witnesses.
Explicit non-success, closed-unstarted, absent-activation, and foreign-owner
facts remain invalid.

The adopted map-item evidence v2 contract remains implementation-pending. V19
does not narrow it and does not claim direct map-item or full P7 coverage.

## Retained v18 history

## Occurrence and submission closure (v18)

V18 retains all V17 IDs and adds direct witnesses for the negative side of
occurrence-derived assessment ownership. A selected reservation suppressed by
a failed closed expansion, a closed-unstarted blocked assessed member, and a
started failed assessed member contribute no required fact. Injecting an
assessment for each such occurrence rejects as `unsupported`, because none is
in the reconciled successful occurrence inventory.

The one- and two-member submission positives select the exact retained
`F:M0:P_MEMBER` and `F:M1:P_MEMBER` references after the root reference. The
verified inventory consequently contains the root fact followed by the exact
dynamic facts. Repeating a dynamic reference rejects `duplicate`; selecting an
unknown dynamic reference rejects `foreign_owner`. Retention and ordered P7
selection remain separate operations.

These cases use the constructible separate scalar `subject` shape documented
in V17. They do not claim that D03/D04/D08 currently admit a promise whose
subject or consumed input is the directly projected map item. That per-item
typed projection and requirement scope remain an open contract clarification;
V18 makes no full-P7 compatibility claim for it.

## Retained v17 history

This directory contains the seventeenth independent reference candidate for D08 v3, SHA-256 `239bdaf97eda6b90caeb13d29826abead08e2beff6297460c26409b3e1f5d87c`. It also consumes the adopted structural-accounting addendum, SHA-256 `88c0ef075b225847f1b2668d2a749307c220eceb50215718db722119d828dc6b`, and materialized-version selection addendum v4, SHA-256 `165c7c95bce31a7c5808860f28d012bbe1986bf0712cebdc86fb08ad07afcb21`. It preserves R3 v1 through v16 as review history. The generator imports no product reducer and computes each expected result from admitted declarations and neutral execution facts.

The finite alphabet has targets A/B/C, invocation I0 plus foreign I1, plan P0, graph G0, operation nodes N/N0/N1/BN, container nodes EXP/SG, binding BIND0 with declaration identities D0/D1/D2/DVER0 and source revisions, materialized artifact versions 1/2/3, context and decision artifacts, Q0 absence revisions 1/2, K0/K1 coverage, request attempts R0/R1/R2, resources Q0/QC, maps with 0/1/2 members, and one nested map. It covers these values and products; it does not claim arbitrary graph-size coverage or product conformance.

## Review289 closure

| Finding | Reference rule | Direct families |
| --- | --- | --- |
| 1. Exact assessment authentication | Authentication joins a started entry, exact success terminal node/target/outcome, subject/evidence/consumed port occurrences, artifact target/invocation, declared absence queries, selected configuration, and read-state projection. | `authentication/*`, `joins/*`, `final_output/producer_*` |
| 2. Requirement projection | Admission requires the projected subject port and every required consumed port to occur in the matching production. Qualification requires the same fields on supporting verified evidence. | `admission/requirement_subject_port`, `requirement_consumed_port`; `selective/*` |
| 3. Causal request and cleanup facts | Recovery requires the failure-specific purpose (`retryable→retry`, `malformed→correction`, `permanent→failover`), the same policy and association, known settlement, and a successful latest authority. SDK verification/accounting resources must close; only caller-owned `left_open` and transport-only cleanup are nonblocking. | `request/*`, `cleanup/*` |
| 4. Expansion reconciliation | A closed membership requires its expander terminal's exact declared expansion outcome. Entries, reservations, members and terminals retain distinct identities. | `membership/closed_0|1|2`, `nested_closed`, `wrong_expansion_outcome`, open/nested-open, selected/unselected, duplicate/foreign, and four terminal categories |
| 5. Decision ownership | A collected decision must be a retained same-invocation decision artifact and belong to the traversed target or shared identity. A withheld target contributes nothing unless a released candidate has its own occurrence of that decision in producer ancestry. | `decisions/direct`, `transitive`, `consumed`, `shared_deduplicated`, `missing_artifact`, `foreign_target`, `withheld_unrelated`, `withheld_linked` |
| 6. Canonical owner and execution-only | Independent result-owner facts are compared with the admitted plan/graph/invocation. Execution-only skips assessment authentication, returns no verified or qualified output, and still reconciles execution. | `record/plan|graph|invocation`, `record/open_execution_only`, `release/execution_only`, `execution_only_empty` |
| 7. Restored coverage and bounds | Every `QualificationLimits` collection has a concrete boundary case. Production, coverage, verified evidence, required decisions and fixed-point limits have exact and one-over pairs; duplicate revision inputs remain visible before normalization. | `bounds/*`, `validity/*`, `selective/*`, `assessment/*`, `provenance/*`, `final_output/*` |

## Adopted structural-accounting extension

Every terminal fact carries an actual Boolean `structural` field, an optional attempt, its category/outcome and bounded reasons. The default operation form is `structural=False`. Success, failure, cancelled and lost operation terminals require an attempt; blocked and inconsistent operation terminals retain their admitted no-attempt form. Structural terminals require `attempt=None`. The closed P0 vocabulary uses terminal category `failure`, while expansion status remains `failed`. Generated non-success facts use `execution_failed`, `cancel_requested`, `transport_lost`, `prerequisite`, or `contradictory` as appropriate.

The declaration independently maps admitted node IDs to `operation` or `container`. Reconciliation compares an entry with that admitted kind, then compares the terminal's structural flag, target, category and outcome with the actual entry state. Assessment authentication additionally requires an operation entry, a non-structural successful terminal and a real attempt. A container fact cannot authenticate evidence.

`structural/valid_container` proves a successful container terminal and an explicitly closed empty membership. Direct negatives cover a structural attempt, structural flag on an operation, operation flag on a container, missing operation attempts for success/failure/cancelled/lost, category mismatch and admitted node-kind mismatch. `membership/missing_expander_terminal` and `missing_member_terminal` retain the two missing-terminal boundaries. `structural/failed_actual_members` and `overflow_actual_members` keep M0 as the sole actual member while M1 remains reserved but uninstantiated; the failed or overflowed expansion withholds through its real terminal without inventing M1.

## Review293 closure

| Finding | V5 correction | Direct cases |
| --- | --- | --- |
| P0 terminal vocabulary | The reducer accepts only the six real terminal categories and nine real reason codes. Generated category/reason pairs use the P5 mappings while expansion `failed` remains separate. | `structural/*`, `membership/terminal_*` |
| Root and parent partition | Every trace has one explicit root membership. Entries carry an actual parent and must occur in exactly one matching membership; every entry has one terminal, and no terminal may exist without an entry. Dynamic parent/member targets must agree. | `membership/missing_root`, `orphan_entry`, `stray_terminal`, `cross_target_member`, plus all map/nested cases |
| No invented revision facts | Artifact selections must name a retained `(key, version)`, absence keys must be admitted queries, configuration nodes must belong to the plan, and state keys must be declared reads. The commutation trace swaps retained A and admitted Q0 revision facts. | `revision/*`, `commutation/independent_revisions` |
| Occurrence roles and provenance | Global artifact roles are ignored. Evidence, subject and consumed meanings come from exact port occurrences. Every provenance fact names a retained same-owner artifact and one of the five actual P5 source kinds; operation/root sources join an exact port occurrence, and decision facts join a decision occurrence. | `roles/*`, `provenance/missing_parent_artifact`, `mismatched_source_artifact`, all prior provenance/decision cases |
| Supporting evidence only | Qualification retains all authenticated evidence in the canonical record, but each released output and consumed-decision seed uses only evidence selected for its applicable requirements. Intermediate evidence can authenticate without releasing a different final candidate. | `selection/unrelated_assessment`, `selective/intermediate_a0_final_a1` |
| Association set identity | Request attempts normalize association identities as sets before predecessor/successor comparison. | `request/association_order_invariant`, `changed_association` |

## Review296 and clarification297 closure

The provenance event is a closed discriminated representation of the five accepted P5 key variants. A bound input retains a distinct structured P6 `BindingArtifactRef(declaration,key,version)` and maps it to its invocation `ArtifactRef`; substituting the invocation reference for that binding reference rejects. An initial collection retains its admitted declaration, target, node and port, and its parents must equal every actual bound-item key for that declaration and occurrence. A map item retains the exact expander, selected child, target, admitted `item_input`, nonnegative item key and positive item version. Its retained collection's canonical item at the child's occurrence order must have the same key and version. The reducer checks the child/parent membership, target, port fact and producer fact together. Direct mutations cover each field, omitted or multiple parents, incomplete initial-parent coverage, and wrong declaration or reference kind.

Clarification297 retracts global candidate/decision artifact disjointness. `roles/candidate_decision_distinct_occurrence_alias` proves that one artifact may carry decision authority at one occurrence and become the final candidate at a later distinct occurrence. `roles/subject_decision_collision` continues to reject a decision role on the exact assessment subject occurrence.

Atomic declarations are stored as P1's exact partition. The constructor adds singleton groups for every target absent from an explicit multi-target group; admission rejects missing, overlapping or foreign partition members. Singleton groups add no peer withholding. `propagation/atomic_ab` and `atomic_bc` expose their normalized singleton complements, and `admission/incomplete_atomic_partition` rejects an unnormalized declaration.

## Review299 closure

P6 authority is retained independently as one binding receipt with ordered source facts and its exact bound-artifact inventory. Each admitted site has a unique declaration identity and source revision. Bound provenance must join one inventory item by declaration, key, version, target, node and port; an invented key/version or a foreign receipt source rejects. Initial collection parents are computed from that inventory and then matched to the derived bound provenance, so adding an arbitrary bound event cannot enlarge the authoritative set. Direct cases cover an unretained scalar ref, an unretained collection parent, a foreign source, and duplicate declaration identity.

Map support retains the admitted expander node, expansion outcome, membership port and item input. The only valid collection parent is derived as `OperationOutputKey(actual expander activation, target, membership_port)`; no declaration field accepts a producer key. The positive producer is the actual `MAP` expander occurrence on `members`. `provenance/map_item_two_members` binds ordered occurrences M0/M1 to distinct item identities 0/1, while wrong outcome, membership port, producer activation, unrelated producer and swapped item assignment reject.

An explicit empty atomic group rejects at the P1 `invalid_value` boundary before exact partition checks. Normalized nonempty multi-target groups and singleton no-peer behavior remain unchanged.

## Review302 closure

The declaration now retains every admitted output dependency as `(node, outcome, port, inputs, identity_input)`. Runtime `input_producer` facts map each executed input occurrence to its exact provenance key. An operation output must have exactly the ordered parents selected by those mappings; missing mappings and missing, extra, or swapped parents reject. Root outputs independently select either a node output or a direct workflow input. `provenance/root` uses `ROOT:A:subject` as the final producer, `identity_alias` uses a distinct identity-preserving output occurrence, and `subgraph` contains a structural SG wrapper, an operation BN body, a projected body input sourced from the root input, and an exact body-output-to-wrapper-output projection. The corresponding parent mutations reject.

Completion records exhaustive accounting. Exactly reconciled non-success terminals and failed or overflowed expansions are closed and unmet. Missing terminals or open membership remain incomplete. Request normalization separately retains uncertainty: a fully settled final failure is unmet, while Lost, unknown usage, missing settlement, or inconsistent attribution is unknown. The public withholding code remains `request_accounting`.

An empty cleanup association is valid only for `transport_only`. Empty verification or accounting associations create global `inconsistent_attribution`; nonempty unrelated-target localization remains unchanged. Evidence occurrence roles apply root-output precedence before evidence-port fallback, so `roles/root_bound_distinct_evidence_candidate` authenticates a distinct evidence output selected as the root candidate while its assessment subject remains the input occurrence.

Map admission permits operation and container expanders. `provenance/map_item_two_members_operation_expander` uses a nonstructural terminal with a real task attempt, while the existing two-member case retains the structural container form and exact M0/M1 ordering.

## Review306 closure

Output dependency inputs and provenance parents are set-valued. The generator emits deterministic lists, rejects duplicate members, and compares sets. `provenance/parent_order_invariant` reverses serialized parent order and remains met. The historical `swapped_dependency_parents` ID now swaps the subject and context producer assignments themselves and rejects because each producer artifact must equal the exact executed input-port artifact.

Input-producer facts form a complete occurrence inventory. For every actual port occurrence selected by an admitted output dependency or structural input projection, there is exactly one same-activation, same-node, same-target producer fact whose provenance owns the port artifact. Missing, extra, foreign, and wrong mappings reject. `provenance/extra_input_producer` covers an otherwise valid producer on a port that was never executed.

Every reused output artifact has an explicit admitted identity input. The baseline result reuses the subject artifact and therefore declares `identity_input="subject"`; decision, map, binding, collection, and shared-context variants retain the same explicit identity edge. `selective/intermediate_a0_final_a1` produces a distinct result artifact and declares no identity input. Assessment facts consume exactly their named promise ports and carry exactly the factory-derived promise coverage. The baseline promise and fact consume only `context` and carry `K0/K1`; decision-consuming families declare and retain their decision input occurrence explicitly. Missing, extra, and duplicate retained coverage reject instead of being treated as partial factory output. Partial evidence is represented by a distinct admitted partial promise.

Subgraphs retain a closed body source kind. `node_output` projects the deepest body operation output as before. `workflow_input` projects the wrapper's captured input directly without fabricating a body operation. `provenance/subgraph_nested_workflow_input_passthrough` proves root input capture through the structural wrapper; missing and wrong capture mutations reject.

Request uncertainty includes Lost and inconsistent terminal outcomes even with known usage and confirmed remote stop. A settled permanent exhausted failure remains unmet. Fresh `initial_binding`, `adaptive_retrieval`, and `repair` requests require no retry predecessor and remain independently admitted starts.

Requirements select a promise meaning and projected subject, consumed-port, outcome, and coverage semantics. They do not select one promise ID. All eligible authenticated assessments are considered. Only current, satisfied, coverage-complete evidence enters a target's supporting set; partial evidence remains in the canonical authenticated inventory. The partial-only, complete-only, and combined two-promise cases assert these projections directly.

Initial-binding cleanup uses a distinct neutral ledger with the same closed reconciliation as execution cleanup. Every cleanup has exactly one pre-close association and every association has exactly one cleanup. The A/C cases cover normal close, A-local accounting failure, and global missing, foreign, extra, or duplicate association defects.

## Adopted Materialized-Version Extension

A repeated runtime artifact key is valid only when every retained version comes from one admitted materialization lineage. Initial versions share one binding declaration and source item key. Map versions share one expander occurrence, target, and item key. Each version has a positive version number, a distinct `ArtifactRef`, its own artifact fact, and one exact `BoundInputKey` or `MapItemKey` provenance fact. Different declarations, expanders, targets, or invocations reject. An invented selected version and two selected current versions reject before qualification.

`validity/candidate_stale` uses one initial `latest` scalar lineage: the operation assesses and identity-forwards the selected maximum version as its exact final candidate, while the current view selects the retained older sibling. The final reference is therefore unavailable and qualification is unknown; the sibling never becomes a candidate. The evidence and consumed stale families use the same closed initial `latest` inventory and selected scalar port. `provenance/version_edge` now contains two bound versions and keeps the original one-version identity alias in `provenance/identity_alias`. `lineage/map_two_versions` uses one closed map expander with two members for source item `(0,1)` and `(0,2)`. Its crossover mutations change the expander, target, or invocation.

Artifact and byte limits count each retained version. Provenance limits count every parent edge. The initial `latest` scalar route has one parent-free `BoundInputKey` fact per retained version and supplies the maximum directly, without an `InitialCollection` or extraction edge. The exact and one-over cases exercise all three limits. Publication rollback is proved by the R2 v2 operational trace; the historical `lineage/rejected_publication_rollback` ID now consumes only the resulting retained execution facts and contains no trusted rejection marker.

Every initial `latest` positive also retains its exact binding request association, retrieved terminal, known settlement, SDK accounting cleanup association, and closed cleanup. Missing or foreign request and cleanup ownership reject at their typed boundaries. The destination input producer must name the maximum retained version for the admitted declaration and source key; `lineage/latest_older_selected` consistently rewires every downstream fact to the older retained sibling and still rejects.

Assessment-production admission joins each evidence output dependency to its exact node, outcome, and evidence port, then requires its input set to equal the promise's consumed-port set. The baseline uses a fresh evidence output depending only on `context`; its identity-forwarded result depends on `subject` and `context`. `roles/candidate_evidence_alias` is a separate constructible shape whose promise consumes both ports and whose evidence output explicitly identity-forwards `subject`. No fixture obtains an artifact alias from an undeclared dependency.

The P5 assessment seam requires an evidence output's declared inputs to equal its promise's consumed ports. An identity-forwarded versioned evidence output therefore makes the same version a consumed dependency. The neutral evidence stale family retains an authentic initial `latest` lineage and the stale result. It is explicitly a combined evidence-output and consumed-reference stale witness, not evidence-output-only staleness. The reference does not create an ownerless alias to simulate isolation.

## Full finite family inventory

- `validity/*` changes candidate, evidence output, consumed artifact or decision, absence, configuration and state. Missing gives `unknown`; a known unequal version gives `stale`; stale wins over simultaneous unknown. `selective/a_only|b_only|a_b|decision_{current,stale,unknown}` provides the dependency products. The materialized evidence replacement retains the exact evidence/consumption coupling described above. `intermediate_a0_final_a1` uses distinct ordinary output keys and withholds for a missing assessment of the exact final candidate; it does not pretend unrelated producer lineages are versions of one key.
- `assessment/*` covers satisfied baseline, unsatisfied, unknown, missing, incomplete/extra/wrong-kind coverage, unsupported finding, duplicate submission, caller replacement, absence-query mismatch, undeclared consumed port, and foreign target/evidence/subject. `authentication/*` contains every review289 minimal join mutation.
- `membership/*` covers selected and unselected choice accounting, zero/one/two and nested closed maps, open and nested-open membership, missing or duplicate terminal/member facts, exact expander outcome, foreign ownership, and blocked/cancelled/lost/inconsistent structural expander terminals. `structural/*` adds the admitted node-kind, terminal-kind, attempt and failed/overflow actual-member products.
- `request/*` keeps immutable physical attempts, terminal failures, settlement and predecessor links. It separates valid retry/correction/failover from wrong-purpose, changed-policy, changed-association, later failure, Lost, unknown usage and foreign attribution.
- `cleanup/*` separates verification, accounting, transport-only, caller-owned, SDK-owned, unrelated C, missing association and foreign-target localization.
- `propagation/*` covers independent A/B/C, A→B, reverse B→A, and distinct `{A,B}` and `{B,C}` atomic groups. The default declaration retains P1's actual singleton atomic groups for otherwise ungrouped targets. A singleton adds no peer relation, so fixed-point propagation does not add `atomic_group` withholding for it.
- `provenance/*` uses the actual P5 source kinds: operation output, root input, bound input, initial collection and map item. N0/N1 scoped bound identities join an independently retained P6 receipt and artifact inventory; initial collections cover every authoritative bound-item parent, while initial `latest` declarations retain all same-key versions and supply only their maximum scalar artifact; map items derive the selected expander output and join ordered members to canonical collection items. Shared identity, mismatched ownership, invalid source fields, missing and duplicate final producers remain distinct facts.
- `lineage/*` covers initial `latest` and map two-version positives, maximum-selection enforcement, duplicate and invented selections, declaration/expander/target/invocation crossover, exact and one-over artifact/byte/provenance limits, duplicate source pairs, and consumption of the retained post-rollback state proved in R2.
- `decisions/*` covers direct, transitive, consumed, unrelated, shared-deduplicated, unowned artifact, foreign target and both wholly-withheld outcomes.
- `release/*`, `record/*` and `final_output/*` preserve execution identity, artifacts, memberships, terminals, target statuses, exact final candidate, verified evidence and decision projections without rerunning work.
- `admission/*` covers production identity, production count, fixed-point capacity, foreign dependencies, exact atomic partitioning, missing promise, unsupported outcome, requirement projection, binding occurrence ownership, initial-collection support, and unique map support.

`commutation/independent_revisions` swaps the retained A artifact revision and admitted Q0 absence revision and requires byte-identical normalized results. Provenance parents must exist before children, so parent-child order is causal and is not treated as commuting.

The reference does not authenticate product objects, verifier science, serialization, continuity, or delivery. A later adapter must construct the same facts through accepted P5/P6 and P7 public boundaries; this corpus cannot mint factory-authenticated assessments or grant product acceptance.

## Production projection correction (v13)

The neutral reducer keeps candidate-current uncertainty as private state. Its public projection uses the closed `missing_candidate` withholding code, retains the exact final candidate identity, and reports `artifact_available=false`; `validity/candidate_stale_no_assessment` independently adds `missing_assessment`. Execution-only cases have no admitted qualification productions, assessment callback, assessment fact, or absence query. They retain the final execution artifact while projecting `candidate=None`, `artifact_available=true`, `qualification=not_assessed`, and only `execution_only` withholding.

Public target rows omit the reducer's pre-propagation eligibility bit because the production result cannot reconstruct it. Open accounting projects the closed `Completion` value `pending`. Rejections expose exact `EffectCode` values: foreign assessment ownership is `foreign_owner`, duplicate membership is `duplicate`, and absent provenance is `missing`. The self-test carries a canonical frozen v10 ID inventory and digest inside this directory, so a copied reference has no external predecessor-path dependency.

## Retained fact and submission correction (v14)

The neutral assessment envelope now has an explicit `submitted` selector that is not an SDK fact field. The reducer always retains the P5 assessment fact and separately selects the exact retained fact for P7 submission. `validity/candidate_stale_no_assessment` therefore keeps its successful P5 fact while submitting none; its canonical record has no verified evidence and withholds for `missing_assessment` and `missing_candidate`.

Retained-result corruption uses the product boundary codes: unmatched assessment node/outcome/promise tuples are `unsupported`; a removed consumed port with its retained input-parent fact is `contradictory`; missing absence environment entries are `missing`; foreign evidence-port targets and entry nodes are `foreign_owner`; and a wrong final-output outcome is `contradictory`.

Cases tagged `comparison_scope="neutral_only"` mutate coverage, subject, consumed-set, or evidence-port values that are derived by P7 or absent from the real retained types. They preserve the finite neutral family and its IDs but are not public positive execution paths and must not be implemented by adding fields to `ExecutionAssessmentFact`, `ExecutionTerminalFact`, or `FinalOutputFact`. Production comparison for those semantics belongs at declaration admission, P5 callback completeness, or retained port/provenance integrity.

## Explicit retained facts and ordered submissions (v15)

Assessment events now retain one fact under a neutral fact identity. Separate ordered `assessment_submission` events reference those retained identities. The identity is reference grammar, not a new SDK field. `assessment/missing` and `validity/candidate_stale_no_assessment` retain the required P5 fact and submit zero references. `assessment/duplicate` retains one fact and submits its exact reference twice. Submission exact/one-over limits count the ordered references. Missing references are foreign ownership; duplicate exact references reject as `duplicate`.

For protection qualification, the retained inventory contains exactly one fact for every target and admitted production. That completeness check is independent of the ordered submission list and runs after typed integrity adjudication, preserving the exact codes of retained-owner corruptions. Execution-only qualification admits no productions and requires no assessment facts.

`assessment/foreign_evidence` is a production-boundary retained corruption because `ExecutionAssessmentFact.evidence_artifact` is real. It is no longer tagged neutral-only and rejects `contradictory` when the retained artifact differs from the exact evidence port.

The finite neutral cases retain these concrete production witness obligations:

| Neutral case | Actual owner boundary and required witness |
| --- | --- |
| `assessment/consumed_port` | Admission requires `EvidencePromise.consumed_ports` to equal evidence output dependency inputs and name real operation inputs. P7 derives consumed refs from exact retained ports and input parents. |
| `assessment/duplicate_coverage` | Coverage is a typed `frozenset`; duplicate atoms cannot survive construction. Verify the closed set type plus distinct-atom exact/one-over coverage limits. |
| `assessment/extra_coverage` | Coverage belongs to the admitted promise. Verify admission bounds and requirement/promise coverage matching; callbacks cannot add atoms. |
| `assessment/foreign_consumed` | Mutate a real retained consumed input port, artifact owner, or captured provenance parent and require the exact P7 integrity rejection. |
| `assessment/foreign_evidence` | Mutate the real retained `ExecutionAssessmentFact.evidence_artifact`; exact evidence-port equality must reject `contradictory`. This case is production-boundary. |
| `assessment/foreign_subject` | Derive subject from the admitted promise subject port and exact occurrence port. Cover invalid declarations at admission and actual subject port, role, or artifact corruption at P7. |
| `assessment/foreign_target` | Derive target from activation and entry ownership. Cover foreign activation, entry, or port ownership using a real retained result. |
| `assessment/incomplete_coverage` | Use real partial and complete admitted promises. Partial-only support withholds `incomplete_coverage`; complete support releases. |
| `assessment/wrong_kind_coverage` | Construct admitted promise and requirement coverage with real `CoverageAtom` values and test admission and matching. |
| `joins/evidence_port_swap` | Return a wrong `LocalAssessmentResult.evidence_port` through P5 and require callback capture to fail without a retained fact. |
| `joins/subject_port_swap` | Mutate the static promise/requirement subject declaration at admission or corrupt the real retained subject port fact at P7. |

These obligations preserve the full D08 family without adding coverage, consumed-set, subject, target, or evidence-port fields to the real assessment fact.

## Exact retained owners and representable revisions (v16)

Retained assessment completeness compares the exact `(activation, target, node, outcome, promise)` owner inventory with every root target and admitted production. Neutral fact identifiers cannot distinguish two retained facts for the same owner. A repeated owner rejects `duplicate`; a missing owner rejects `missing`, so an unsubmitted duplicate A fact cannot mask a missing B fact.

`revision/invented_configuration` represents a node in the same workflow that is absent from the selected implementation plan and therefore rejects `missing`. Foreign-workflow ownership remains a separate product boundary and is not inferred from absence in the neutral `node_kinds` map.

`bounds/duplicate_state_key` uses two distinct revisions, 1 and 2, for the same state effect. Both survive the actual `frozenset[StateRevision]` boundary and reject `duplicate` by effect key. Literal repetition of one identical `StateRevision` collapses at the canonical view type; D08's identical-repeat validation remains an explicit raw-input or factory-boundary obligation pending reviewed disposition. The neutral corpus does not claim that the typed P7 boundary observes an identical repeat.

An operation terminal in `success`, `failure`, `cancelled`, or `lost` state without its attempt is `contradictory`. This preserves the accepted P0 terminal-construction rule and the structural addendum; the missing-attempt cases are retained-owner integrity negatives rather than absence lookups.

## Occurrence-derived assessment inventory (v17)

The required retained assessment owners are derived after entry, terminal, and membership reconciliation. Every nonstructural operation occurrence whose entry and terminal are successful and whose node and outcome match an admitted production contributes one exact `(activation, target, node, outcome, promise)` owner. Structural containers, closed-unstarted entries, unsuccessful occurrences, and operations without a matching production contribute none. A repeated retained owner rejects `duplicate`; a missing matching occurrence rejects `missing`.

The historical membership and map-provenance cases now name their ordinary member operation `MEM`, which has no admitted evidence production. This matches the actual map fixture: the expander emits the member collection, while ordinary members consume projected items without returning assessments. `membership/closed_1` and `provenance/map_item_two_members` therefore remain genuine positives rather than hiding assessment omissions or being converted into rejection cases.

`assessment/dynamic_occurrences_0|1|2` add a separate assessed member operation `MN`, modeled on the actual representable P5 map shape. Each member retains its exact dynamic `MapItemKey` input and separately receives the root candidate as its `subject` input. Its promise consumes that scalar subject, and its fresh occurrence-owned evidence output depends on the same subject. The member assessment carries the operation configuration and no absence or state revisions, matching the actual callback. The map item is not silently claimed as assessment consumption: dynamic item projection has no scalar identity to the root input and is a separate contract question. Each successful member retains one `P_MEMBER` assessment fact independently of P7 selection. `dynamic_occurrence_missing` removes the required member fact and rejects `missing`; `dynamic_occurrence_duplicate` adds the same owner under a fresh neutral fact reference and rejects `duplicate`.
