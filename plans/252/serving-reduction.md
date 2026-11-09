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
