<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Local serving reduction implementation

Implement the operator-reviewed reductions for PR #212 (`a54f43eb`) and #301 (`bdb8a715`) while preserving current main (`f294dc1e`). Work is isolated from existing PR branches and unrelated main-checkout edits. The agreed decisions supersede the experiment's original plan.

## Scope and decisions

- Move the entire packaged native GLiNER serving controller, server, and backend under `engine/local_inference`; retain compatibility wrappers and add an explicit-lifetime public context manager. Interface owns client construction; engine owns processes and credentials. Preserve native 128/32 word chunks and source character offsets.
- Delegate vLLM configuration decoding, defaults, argument validation, and server execution to pinned upstream abstractions. Omitted options use upstream defaults; explicit false remains explicit. Keep pure compilation and digest-protected explicit configuration. Validate using the configured serving interpreter before startup.
- Keep `connection`, structured diagnostics, separate lifecycle receipt types, Factory detection, the independent serving project, ownership proof, PID protection, authentication, bounded cleanup, and task-aware probes.
- Remove unconsumed plan metadata and `--source-revision`; simplify credential mapping and obsolete CLI invocation metadata. Consolidate tests only with equivalent or stronger assembled coverage. Do not replace runner invocation checks with imports or simplify the adapter worker pool without a concrete design.
- Preserve model preparation and revision pinning. Native context resources must surround actual client use; exiting a context must not silently stop a borrowed singleton.

## Execution and ownership

1. Parent integrates current main with #301, resolving the four conflicts and carrying main's native chunk settings. Run baseline focused tests and repository checks.
2. Terra native agent owns packaged native code, interface factory/exports, `tools/serve_gliner.py`, and native/interface tests in its isolated branch.
3. Terra vLLM agent owns serving compiler/runtime/profiles/container and tool tests in its isolated branch.
4. Terra documentation agent owns relevant documentation, bundled skill, and necessary Makefile guidance, coordinating API/schema names with producers.
5. Parent integrates pinned commits, reviews interoperation, and runs the assembled focused tests, full unit suite, repository checks, and strict documentation build. No push, PR merge, GPU/model execution, or paid service calls are part of this task.

## Verification

Use meaningful lifetime, startup-failure, credential restoration, provider preservation, and lightweight-import tests for native serving. Use real CLI subprocesses and cross-interpreter stubs for compile/validate/launch/status/probe/stop, digest tampering, ownership, auth, cleanup, explicit false, and omitted options. Preserve runner invocation coverage. Root checks must not require installing vLLM/CUDA. If the real pinned serving environment is unavailable, state that parser/model execution remains unverified instead of treating stubs as that evidence.

## Implementation log

- Created isolated integration and three producer branches from #301. Existing main and PR worktrees remain untouched.
- Main merge conflicts occur in self-hosting documentation, navigation, notebook model configuration, and its tests. Retain new factory architecture and main's native chunk settings; combine documentation/navigation.
- Producers inspected their scopes and proposed `native_runtime(device=...)` with an interface session, and `[vllm.config]` native option mapping with `validate --plan`.

- Baseline merge committed at `b8526c28`; 33 focused tests and repository hooks passed. Existing MkDocs custom tags required the YAML hook's syntax-only mode; no hooks were bypassed.
- Terra producer tips: native `cba3eace`, vLLM `954bb425`, docs `6bd9aa30`. Mirador prepared and promoted their signed commits to `c0b76cbd` on the isolated implementation branch; producer branches were retained.
- Native controller/server/backend now live under the engine. `native_runtime(device=...)` exposes an interface session; nesting, borrowing, credential restoration, startup failure, and interpreter-exit cleanup are covered. Notebook entry points remain wrappers.
- vLLM options are an explicit `[vllm.config]` mapping. The serving child uses the pinned upstream YAML loader/parser/validation/runner; compilation stays pure. `validate --plan` does not start a server. Structured diagnostics, ownership proof, distinct receipts, and Factory revision preparation remain.
- Combined revision `c0b76cbd`: full unit suite **1,679 passed**, `make check` passed, strict docs build passed. Cross-interpreter tests require host process access here: the baseline's missing `/proc` start marker disappears outside filesystem isolation.
- Existing vLLM 0.29.0/Python 3.12 environment validated all nine migrated profiles with a test-only CPU platform override, plus omitted defaults and explicit booleans. GPU execution and model feasibility were not exercised. Production does not override platform detection.
- Review fixes preserve native caller configuration and 128/32 chunk settings, prevent false-valued unknown options from disappearing in the upstream loader, and prohibit literal credential options in serialized native config. Inherited `HF_TOKEN` remains supported.
- Final integration cleanup fixes a renamed documentation anchor and test annotations, removes an unrelated ignore entry, and excludes the scorecard section from this serving change. Its exact standalone patch is retained in the durable experiment record for separate work.

## Ownership review

Source search was used because astnav was unavailable. Scope is the merged implementation relative to `b8526c28`, not unrelated changes inherited from main.

- Ownership: move; owner=anonymizer.engine.local_inference; evidence=native controller/server/backend relocation and absence of interface/notebook imports; reason=process, listener, environment, and shutdown ownership belongs to the engine.
- Ownership: stay; owner=anonymizer.interface.factory; evidence=create_anonymizer/native_runtime/NativeRuntimeSession call paths; reason=public input validation, client construction, and the lifetime-facing API coordinate the engine with the public facade.
- Ownership: stay; owner=anonymizer.interface.factory; evidence=_create_endpoint_anonymizer/_build_gliner_model_configuration/_parse_gliner_model_inputs; reason=the GLiNER factory injects its reserved detector into external DataDesigner model/provider inputs and constructs the facade, rather than transforming one owned config type.
- Ownership: stay; owner=inference_service_compiler.compiler; evidence=_compile_vllm/_validate_factory_detection; reason=coordinate task, model, endpoint, credential routing, pinned integration, and serialized process command across domains.
- Ownership: stay; owner=inference_service_compiler.vllm_runtime; evidence=load_and_validate_arguments/run_server; reason=third-party parser/config-file IO, plugin initialization, model preparation, and upstream server execution belong to the serving child.

## Remaining limits

The new validation command performs upstream argument and preliminary cross-option checks. It does not construct a complete model engine configuration or establish GPU/model compatibility. Native runtime tests use synthetic process/model fixtures; no model download or paid service call was performed.

## Astra review follow-up

The operator requested one Terra worker per reproduced correctness finding, followed by parent implementation of the three streamlining opportunities. The independent review targets `e765d5d2`; its report and reproductions are retained outside the repository under `/stable-cache/anonymizer-serving-reduction/astra-review/`.

1. Preserve Factory model identity and the full middleware sequence through the real upstream parser. A native YAML list cannot be assumed equivalent to repeated append-action flags.
2. Reject accepted abbreviations of Anonymizer-owned credential/configuration options before plan serialization, without importing vLLM during compilation or copying its option schema.
3. Reject restart/replacement while native contexts are active, including when the process has died. Preserve healthy borrowing, nesting, and cleanup; do not add reconnection machinery.
4. Preserve representable explicit empty lists, and clearly reject unknown or unrepresentable values instead of silently using defaults. Keep explicit false and omitted-default behavior.

Four isolated Terra producers own these findings. The empty-list producer follows the middleware producer to avoid concurrent edits to the same parser path; credentials and native leases proceed independently. Each producer records failing regressions before its fix, targeted passing tests afterward, signed commits, and applicable real-parser evidence.

After integration, streamline caller configuration parsing into one validated input snapshot before native acquisition, move validation subprocess/environment preparation into the controller runtime, and inline the extra private facade-constructor forwarding layer while preserving lazy imports. Retain the typed configuration result. No additional manager hierarchy, generic command abstraction, or upstream option schema is authorized by this follow-up.

Final verification will rerun all four review reproductions against the assembled source, check effective parser values for both Factory profiles and generation middleware, and run the full unit suite, repository gates, and relevant docs checks. No GPU/model serving, weight downloads, or paid requests are needed.

### Fix and streamlining implementation

- Terra fixes: middleware `aeafcb7c`, empty-list transport `d4f718e2` on that middleware base, native leases `cc29f6d0`, and credential admission `b15b7cf5` (including dotted root spellings). Pinned fixes integrated through Mirador at `08679670` after 56 assembled regressions, the real vLLM empty/false/default and Factory model/middleware probe, and repository checks passed.
- Credential verification additionally reproduced accepted dotted spellings targeting owned actions; the worker extended the same small pure root/prefix guard rather than importing vLLM or inspecting arbitrary values.
- Factory construction now resolves and validates raw inputs once, before native acquisition, and passes the parsed model pool and a copied provider list through injection. The typed output record remains, and caller provider instances remain intact. A regression removes input files during simulated runtime acquisition: it failed on the old second read and now passes using the validated snapshot.
- Inlined the lazy facade import/constructor into `_create_endpoint_anonymizer`; removed `_new_anonymizer`. The external-endpoint test constructs a real client and retains its no-process assertion; the installed-wheel test still checks import isolation.
- Moved validation subprocess execution into controller `validate_plan()`, which verifies the digest and shares `_child_environment()` with launch. The CLI retains plan loading and rendering; credentials are still injected only by launch as before.
- Streamlining focused verification: 102 factory, native lifetime, CLI, and parser tests passed. The controller extraction is a behavior-preserving refactor covered by the existing real CLI and cross-interpreter tests; no new command framework or receipt type was added.

Ownership: stay; owner=anonymizer.interface.factory; helper=_parse_gliner_model_inputs; evidence=public factory/session call paths; reason=reserved detector-name policy and raw caller input IO belong to factory injection.
Ownership: stay; owner=anonymizer.interface.factory; helper=_create_endpoint_anonymizer; evidence=typed parsed inputs and facade construction; reason=coordinates the public client construction boundary while retaining lazy imports.
Ownership: stay; owner=anonymizer.interface.factory; helper=_build_gliner_model_configuration; evidence=parsed model pool, providers, selections, endpoint; reason=coordinates third-party model/provider domains with factory injection.
Ownership: stay; owner=inference_service_compiler.runtime; helper=validate_plan; evidence=CLI delegation and configured command/cwd subprocess; reason=controller owns execution effects and structured failures.
Ownership: stay; owner=inference_service_compiler.runtime; helper=_child_environment; evidence=validate_plan and launch_plan callers; reason=shared serving-child environment policy belongs beside process execution.
