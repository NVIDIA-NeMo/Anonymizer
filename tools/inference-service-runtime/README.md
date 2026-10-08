<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Inference service runtime

This independent uv project supplies the Python 3.12 environment for the source
checkout's vLLM serving tool. It has its own lockfile and does not install
Anonymizer or DataDesigner. It is not a uv workspace member or a published package.

Run from the repository root:

```bash
uv sync --project tools/inference-service-runtime --locked
uv run --project tools/inference-service-runtime --locked python -m vllm_factory.compat.doctor
```

The shipped profiles launch `tools/inference-service-runtime/.venv/bin/python`
directly. See the [deployment guide](../../docs/concepts/inference-services.md)
for supported hardware, model profiles, compilation, launch, and cleanup.
See [development instructions](../../DEVELOPMENT.md#local-inference-service-environment)
for tests.
