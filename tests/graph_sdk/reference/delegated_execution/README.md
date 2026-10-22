# Delegated execution reference

The manifest was frozen before production comparison. It has two explicit
single-target graphs, two positive traces, no commuting event pairs, at most
four activations, two physical dispatches and one submission. Its nine named
negative controls define rejection boundaries; additional regressions cover
scope cancellation and stale proof rejection after activation preflight.

Manifest SHA-256: `9f0b2caa673efd8870161070df0079e9c62f27701276b945aeb1d737843ff830`.

The separately frozen driver table records eight traces: four initiation
boundaries for ordinary and prepared-leaf execution. Its expected dispatch,
callable-entry and cancel-hook counts are independent of the driver/reducer.

Driver table SHA-256: `854fe2251d2327f1eff41b2a1c303e72a6ce2d06b6a50827926d9a2dd197c2d9`.

The live fixture reuses admitted graph construction but executes through
`start_execution`; canonical facts come from the production runtime. Expected
protected release is explicit in the table and checked through full `qualify`.
Neither table imports the request reducer or terminal ledger.

The publication table was frozen before its comparison. Two vertical traces
consume `target-0`, publish `candidate:target-0` from the delegated parent, and
qualify that candidate through an independent local assessment. It bounds one
parent artifact and two dispatches. The separate assessment evidence artifact
is produced through ordinary local publication.

Publication SHA-256: `fc5a6aabe0642d0a161b68e679975d843bd0a5f7575edf47549974cddb59457f`.

A fourth one-trace guard preserves an already-successful request terminal when
late parser rejection is attempted. The accepted contract requires an initial
malformed-response terminal plus completed settlement; it rejects a correction
after terminal success, with one dispatch and no contradictory facts.

Parser guard SHA-256: `b6654120a76f119fa9f0d0c4ddbad2f05921460e6c7463c26254a32c13681f5e`.

The cancellation correction adds `cancellation.json`: one admitted graph and
13 event-driven traces. Eight live-call cases cross propagated/caught submission
cancellation, confirmed/unknown stop, and repeated owner cancellation. Two cases
retain already-available completion; two reject pre-entry cancellation without
charge; one preserves rejection of another task's borrowed request. The table
bounds three activations and one physical request. Live cases retain the first
terminal, drain before public return, incorporate late settlement and withhold
protected release. Observer cancellation does not detach the call task.

Cancellation SHA-256: `d9da9134ef10251824264b0b5b8571ece41069b85a3ad529bce8c76e5b8725e7`.

The initial table spelled the confirmed stop settlement `cancelled`; that is a
terminal category, not a valid settlement disposition. It was corrected to the
existing `stopped` disposition before the successful comparison. The original
table and failing run remain in campaign evidence; prior reference tables are
unchanged.
