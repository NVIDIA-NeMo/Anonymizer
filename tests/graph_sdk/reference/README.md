<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Graph SDK reference cases

The reference implementations generate cases independently of the product SDK.
The data, workflow, activation, effects, and qualification corpora are generated in memory during tests;
their expanded JSON is not checked in. Each corpus is checked against the
SHA-256 digest in its committed manifest before a consumer receives it.
Generation is cached as immutable JSON bytes, and each consumer gets its own
decoded copy so mutation tests cannot affect other tests.

Review changes in the generators and their tests. A changed corpus digest means
case contents changed and requires reviewing those changes before updating the
manifest. Moving code between modules should preserve the digest.

To inspect or compare the complete JSON, export it outside the source tree:

```bash
.venv/bin/python -m tests.graph_sdk.reference.corpora effects /tmp/effects.json
.venv/bin/python -m tests.graph_sdk.reference.corpora qualification /tmp/qualification.json
```

Run the reference self-tests with:

```bash
.venv/bin/python -m pytest tests/graph_sdk/reference
```

The small context-source fixtures remain checked in. The export command also accepts
`data`, `workflow_static`, and `activation`.
