<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Inference service runtime

This independent uv project supplies the Python 3.12 environment for the source
checkout's vLLM serving tool. It has its own lockfile and does not install
Anonymizer or DataDesigner. It is not a uv workspace member or a published package.

Run from the repository root:

```bash
uv sync --project tools/inference-service-runtime --locked
```

The shipped vLLM profiles launch `tools/inference-service-runtime/.venv/bin/python`
directly; the GLiNER2 detector profile runs from the main project environment. See the [deployment guide](../../docs/concepts/inference-services.md)
for supported hardware, model profiles, compilation, validation, launch, and cleanup.
Compilation does not import vLLM; `validate --plan` runs vLLM's own argument
parser in the configured serving interpreter without starting a server. Every
vLLM import path the serving child uses lives in
`tools/inference_service_compiler/vllm_compat.py`.
See [development instructions](../../DEVELOPMENT.md#local-inference-service-environment)
for tests.
