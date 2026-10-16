<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Graph SDK reference cases

Active compact reference implementations generate answers independently of the product SDK. Their self-tests, mutation controls and product consumers are retained alongside the reviewed manifests and corpora. The foundation, compiler, evidence, publication and observation models have distinct bounded scopes; a reference self-test is not by itself a product acceptance test.

The historical data, workflow, activation, effects and qualification JSON corpora now live in `frozen/*.json.gz`. `corpora.py` verifies both compressed and original decoded digests before returning cached immutable bytes. Consumers receive separate decoded objects. The retired broad interpreters are not executed to regenerate these historical answers. `data_shapes.py` and `workflow_fixture_builders.py` retain the construction support needed by actual consumers.

Review changes to active generators and their tests before changing a corpus digest. Preserve original frozen provenance when relocating code. [Compact adoption](COMPACT_ADOPTION.md) records the initial bounded adoption; its historical counts do not grant additional product acceptance.

To inspect or compare the complete JSON, export it outside the source tree:

```bash
.venv/bin/python -m tests.graph_sdk.reference.corpora effects /tmp/effects.json
.venv/bin/python -m tests.graph_sdk.reference.corpora qualification /tmp/qualification.json
```

Run the reference self-tests with:

```bash
.venv/bin/python -m pytest tests/graph_sdk/reference
```

The context-source fixtures remain checked in. The export command also accepts
`data`, `workflow_static`, and `activation`.

The publication model keeps its finite predicate and generator definitions. `publication_v3/retirement.json` and `retirement.patch` record removal of the unused standalone manifest CLI, which depended on a calibration module absent from this relocated package. `test_compact_reference_provenance.py` regenerates both publication corpora and checks their original frozen digests. Historical manifest source/test hashes continue to describe the original standalone bundle.
