# Context-source reference fixtures

These independent fixtures specify the adopted D03/D06 extension. They are a draft for review, not a claim of executable production conformance or a replacement for the frozen R1 corpus. Abstract names map to real typed workflow values in the production comparator.

Every valid fixture uses a single identity-preserving operation (or the stated exact wrapper) whose outcome consumes the named context input and produces an output depending on and preserving that input. Static scaffolding must include the matching interface context/outcome/output declarations. Negative fixtures change only the named declaration from that valid scaffold; they must not fail earlier from omitted scaffolding. No arbitrary source tags may bypass real ContextInputRef/WorkflowInputRef constructors.

Collection accounting describes initial materialization only, before operation output publication: two two-byte bound scalars plus their four-byte collection yield three artifacts, eight logical bytes and two parent edges. Execution assertions must additionally check actual input/output port facts, exact final producer, wrapper provenance, and absence of fabricated caller roots. Optional omission uses a valid permanent failure with explicit settlement and must block the selected consumer.

The collection WorkflowInputRef fixture preserves D03 type-compatible static admission but must be rejected by root/context admission before effects; review must pin its established exact validation code. Substitution must use the real public substitution API and retain source-kind and checked-interface semantics. Preserve all old R1 cases and add these fixtures separately.

## Fixture notation

All local names resolve under one owned workflow and selected target. FOREIGN names resolve under a distinct real owner. BoundInputKey notation includes target, destination, item key and version; it is an abstract identity label to map to the actual public key, not a new constructor signature. InitialCollectionKey similarly names the exact declared destination. Compare all expected lineage fields by actual key and artifact equality.

The two-outcome fixture declares both input context ports but each outcome consumes only its stated port; the interface mirrors both outcomes and their separately identity-preserving outputs. Initial declaration coverage is their union, while callback/result consumption is the actual outcome subset. The evidence fixture adds a declared promise with exact input subject and consumed port and checks static projection; it makes no claim of runtime qualification.
