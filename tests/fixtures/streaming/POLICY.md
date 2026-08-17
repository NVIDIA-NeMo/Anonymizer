<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Synthetic streaming fixtures

Fixtures in this directory are synthetic test data only. Do not commit customer
data, private captures, production corpus excerpts, credentials, or provider
responses. Provider-backed characterization requires separate written approval.

Streaming tests and the internal characterization runner use local `Redact`,
`Annotate`, or `Hash` strategies only. Reports must be aggregate and
privacy-safe: they may contain counts, byte totals, durations, and boolean
outcomes, but never source content, protected content, entity values, prompts,
provider text, engine identifiers, or per-row traces.
