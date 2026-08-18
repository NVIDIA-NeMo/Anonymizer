<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Private Plan A design-spike record

This record covers the private, synchronous protection slice only. It does not
authorize a public SDK, Intake integration, release decision, or alternate
engine path.

## Ownership verdicts

Ownership: stay; owner=anonymizer.interface._protection._compile_protection_plan; evidence=the helper consumes AnonymizerConfig, ModelSelection, and ModelConfig and returns a distinct private plan; reason=cross-domain release-policy compilation is not a same-type config transform.

Ownership: stay; owner=anonymizer.interface._protection._build_operation_plan; evidence=the helper consumes a compiled plan and protection records and returns an invocation-private operation plan; reason=batch admission bounds and correlation span several domain values.

Ownership: stay; owner=anonymizer.interface._protection._ProtectionFlow._execute; evidence=the lifecycle caller coordinates the pandas runtime, invocation verifier, and terminal accounting; reason=effectful runtime coordination does not belong on a frozen domain value.

Ownership: stay; owner=anonymizer.interface._protection._failure; evidence=the helper constructs one closed safe-failure domain value from static enum and stage inputs; reason=the private protection domain module owns its stable failure taxonomy.

Ownership: stay; owner=anonymizer.interface._protection release-policy helpers; evidence=_has_accepted_detections and _redact_release_passed inspect verified engine entity values to derive a protection disposition and enforce the compiled Redact predicate; reason=these helpers coordinate engine schema values with protection-domain policy rather than transform one config model.

Ownership: stay; owner=anonymizer.interface._protection._SafeRepr; evidence=all new private domain values inherit the content-free rendering mixin; reason=one domain-local rendering policy prevents record content and references from entering repr, logs, or errors.

## Test strategy

Focused contract tests cover private value bounds, closed compilation outcomes,
pre-admission batch rejection, terminal accounting, lifecycle overlap and close,
safe rendering, and adversarial engine results. A focused integration test uses
the synthetic detector and the real local Redact/pandas runtime seam. Tests
assert returned outcomes and absence of private data, not internal call counts.

The flow borrows the facade runtime. Closing it rejects new admission and lets
an already admitted synchronous invocation drain; it never closes the borrowed
DataDesigner or provider resources. The current runtime does not establish hard
cancellation or deterministic dependency teardown, and this spike makes no such
claim. Execution completion remains distinct from external data-handling and
release or commit authority.
