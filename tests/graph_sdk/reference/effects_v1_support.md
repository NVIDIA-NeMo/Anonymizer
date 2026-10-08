# R2 binding/map reference v6

V6 adds the missing manifest binding to request-boundary addendum SHA 6344f7f1bdbb14a4c9e08f546928d31c89f26d3ed26e0c50362d9537d891c3a5.
The corpus is byte-identical to v5. This version retains the three v4 constructor-boundary corrections under
D06-request-boundary-addendum-v1.md. It changes no other case and removes none.
Counts remain 296 cases, 301 traces, 1008 events.

The existing IDs source_result_wrong_outcome, source_result_outputs_present and
source_result_consumed_present now use binding_result events representing direct
AssociationResult construction. Expected rejection is CONTRADICTORY before any
AcceptResult/settlement or source fact. The fields cannot be supplied through
SourceResponse; generic invalid provider dictionaries are not equivalent tests.
All genuine provider responses and cancellation-race witnesses remain unchanged.

46 isolated tests, Ruff and ty are checked for this version. All earlier
reference bytes and reports remain historical. The materialization-v2 prefix
and its 20 corrected cases are unchanged from v4.

## Historical v4 support

The following describes v4 and its predecessor comparison. V5 differs only in
the three cases above; use the v5 manifest for its corpus digest.

# R2 binding/map reference v4

Independent neutral reference; no production imports or output-derived expectations.
This root-authored correction follows independent review281 of frozen v3.
Product author276 did not author this reference. V3 and all prior bytes remain
retained in sibling directories.

The reference covers the original request, retry, cancellation, settlement,
admission, binding, resources, decision and bridge families, the adopted context
materialization extension, and the reviewed map execution and binding addenda.
Its finite alphabet and bounds are defined by effects_v1.py; the corpus records
every generated trace and normalized expected state. Case IDs expose the named
families; the manifest gives actual counts and contract hashes.

V4 restores optional failure/oversize aggregate partial while preserving source
failed/oversize and physical request facts. Actual map overflow retains valid
expander output, assessment, producer provenance and physical request success,
but creates no child items or successful membership. Storage overflow publishes
none of these. The independent collection-item ceiling distinguishes the two.

Collection output and other operation output producer facts now exist before
MapItemKey parents reference them. A prospective membership rejection derives
from an explicit cancelled parent state, reached through close_parent; injected
p3_accept verdicts reject. The adapter must reach the corresponding real
terminal P3 state instead of passing an expected-error switch to production.

A map fact represents this bounded local expander transaction, not an entire
execution receipt. Production comparisons must project the corresponding exact
facts and must separately exercise the actual dynamic/subgraph/loop execution.
No test claims a streaming provider interface or universal proof.

Counts: 296 cases, 301 traces, 1008 events.
Corpus SHA-256: 60504bb0c3e45efccb459f661fd275c35391c60537255eab29dfebe59cb5a9d8.

Of the 214 materialization-v2 predecessor cases, 20 are corrected and 194 unchanged. The original corpus digest remains b509a2e1dc697ae5e16c8f2f9652aac74d862e4b449f64958fbba99abda87e93; the corrected prefix digest is d15f65e7a559d4990911f067cfa310419881813319d678d532c9db2d1f659987.

## Corrected predecessor cases

| Case | Original digest | V4 digest |
| --- | --- | --- |
| binding/two_sources_same_key | 1dfa65b743eeea338608e4f099da3856fc43836cf9276c590511a870b5b95445 | 24d8f3d91681ae13f537efe84ae8b526f6dd22d04feb539b74d38af718484097 |
| binding/one_source_two_declarations | f6afa24e173334c5959d83760d62a64467c5bd9e13172fb2dd5bd3df62a19c79 | 97816a388325c4bc086751089b7d04fe208dbd355a63e9bee29ba19dd6bfa6ed |
| binding/required_failure_preserves_prior | 0f19130c5ef55591d7254db9f4ca901ee98cf68d3c033f0aaa3e2fdb0eeaebb4 | 53ab562db17363fc463e73c7760e2bf7ffed664f57bc41f149a6b9e08bda35ed |
| binding/optional_failure_partial | b39883bc40802d3b425ad62fd4fb982f8bbd19da272f47005ca3b591d6a058fd | 06ea43146028fc5adbcc6b57a724b6f46a12f8965a4ceee66ea19ec351331e32 |
| binding/omitted_optional | bc6152acaf8501eefb1a427dfb9990fbe627996fbe10f0eaebc24b80233646da | 0a68d3c565d4aaf811059aa509abd7c351338558579ed21e41c5ada7909ddd2e |
| binding/response_reordered | ce82c70827478ea75bfbf75e4ce2ef5249a2fd6f865ae7ba264cfa7f07ed357f | a1105e77b805e9523698e299a1bbce9369b0402dcafce0e4161e581429afe538 |
| binding/oversize_no_truncation | 4f10fbbbae569ac658eeef2c34e661fae822827c1dd2a14b1c616521ba6e2d8a | 7ff1280b63b73ea1a289032eb47ee933f0f6e90d2957a22b64b971f14bf31cbc |
| binding/exact_item_byte_bounds | bab3fd873ce1ecb260bd55272c5289d797930bea8e7cb90ab19610098336808f | 2c82276ca41eddfb6f02041cee297d6becd6ea94e64dd6163b5d03a19e165674 |
| binding/one_over_byte_bound | 6ec35b067941e753ca4dfaf969571b8e80588b26c7e0dd7abea65c6b631e6a20 | 46ae00b3cbefbdb9026ac46d28b130034870f786a933d8f500c4f67599b58532 |
| binding/one_over_item_bound | 2d00c984873c7f482c4822d9389490330901e27338e34aeae39368c749047026 | e208b4bd85aab7c97adc7f07cd62ce4a7594d88087715a9d846f18483305681d |
| binding/unsolicited_source_result | 40407944db0c8bf5a98b5fe41ce6b9e5222eed057066e94874eeae6ff62b4f1b | 2a037c2d83a92cde2de2b07de86efd97a3f56a8bd0135113d26e3b10a4653e10 |
| binding/wrong_source | 7725b2dc7a9b493af4cf1cfd2c28ff6e5ae9809a7bfb0b17d597c97a6903997e | fd1603cb16fa4c45ae94e17456340823b02f5648bc5492cd771b189bdb066539 |
| binding/missing_result | 6f54c48a35b18730d8eabcdfbeaf6f751e34664bb5c185a64cdf21cbe1464534 | 4534f45330a013a8c6b8dc63aaf3a167b70dc035ddc275cdc298131dd7385c53 |
| binding/duplicate_result | 0f04fee815a5ade3be4eba8478881ff0371f881dac43414670bad2006766a89a | 4bba74569ff97d173ac9858adfdfcab110eb9da8809862f167f51a747f242655 |
| binding/foreign_result_association | d46e1c22ba40e2ca1f96c5fba8002f66ff2acab7a12aa001734be949a211d37f | 6487926b107d749ce957c4cb62a0e1b948273ca910a006105fc78c83d1d8b4b9 |
| binding/lost_late_source_result | 38b6189704d6ebd89ad6f9d314f910baed57e3bb165444dc76b44d27f807f0a1 | 95c7ec13c3ffde1754a2a681886f44a1caf55342fecc503eaf3bca60e32f3379 |
| binding/lost_late_source_failure | 16cd62470f1ca327295bdf8228a1680188f102aeb8be102ab52bb2d8cef5bd28 | a5c07310e49f48eeb0df2e8b06f376429fecd8c5d9316a1be6ca14155eb589e9 |
| binding/cancelled_late_source_result | 9411f93366e7cf3709561a30bb55dec02edf670526ad479792fc6ac87ec6de4c | b85520b883212f1d78b8570225edda3e5d6b7e787cb968f5709d7d29a0adc9aa |
| binding/cancelled_late_source_failure | 289716e0d92c2b98a268bba2653e8654482ca245041c5a6c7f53f4f36a479e26 | b524a49ca2809c0c10c74b5004a7791154899d84d5c3ae271a6f0e5dc3f644c2 |
| materialization/initial_unmaterialized_source_result | c2753b0acdc17c8344806286e200622741f060f0f0cca5ba23d1de1e1268ee89 | dfcc049a80d3db17f85433611e44dec5a9572d5a7266be27955f576da42cb95f |

## Changes from frozen v3

60 existing cases changed; 7 cases added.

- binding/optional_failure_partial
- map/admit_operation
- map/admit_subgraph
- map/admit_control_only
- map/default_override_dependency
- map/max_zero_scalar_join
- map/max_one_scalar_ordinary
- map/max_one_scalar_workflow
- map/max_two_scalar_join
- map/max_two_scalar_ordinary
- map/max_two_scalar_workflow_output
- map/missing_expansion
- map/duplicate_map_source
- map/map_loop_duplicate_source
- map/duplicate_loop_source
- map/context_minimum_conflict
- map/collection_item_schema_conflict
- map/item_type_mismatch
- map/context_override_conflict
- map/dependency_summary_mismatch
- map/false_identity_summary
- map/foreign_before_duplicate
- map/membership_with_0_other_outputs
- map/membership_with_1_other_outputs
- map/membership_with_2_other_outputs
- map/membership_0
- map/membership_1
- map/membership_2
- map/membership_one_over
- map/missing_membership_port
- map/wrong_membership_type
- map/wrong_parent
- map/duplicate_membership_port
- map/duplicate_item
- map/noncanonical_items
- map/subgraph_item_binding
- map/control_only_members
- map/default_override_no_fallback
- map/prospective_transition_rejected
- map/artifact_count_one_over
- map/artifact_bytes_one_over
- map/provenance_one_over
- map/bounds_exact
- map/resolve_join_max_0
- map/resolve_ordinary_max_0
- map/resolve_workflow_output_max_0
- map/resolve_join_max_1
- map/resolve_ordinary_max_1
- map/resolve_workflow_output_max_1
- map/resolve_join_max_2
- map/resolve_ordinary_max_2
- map/resolve_workflow_output_max_2
- map/resolve_join_max_1_empty
- map/resolve_ordinary_max_1_empty
- map/resolve_workflow_output_max_1_empty
- map/loop_exit
- map/loop_bypass
- map/loop_prior_continue
- map/loop_failure
- map/loop_overflow
- map/collection_items_exact
- map/collection_items_one_over
- map/overflow_collection_storage_one_over
- map/overflow_collection_storage_exact
- map/collection_items_invalid_limit
- map/caller_transition_verdict_rejected
- binding/optional_oversize_partial

Checks before freeze: 44 isolated self-tests, Ruff, and ty passed; two final regenerations matched bytes. No product files changed. Independent review and production comparison remain pending.
