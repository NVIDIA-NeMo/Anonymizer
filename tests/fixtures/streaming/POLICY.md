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

## Intake validation corpus

The `intake_*` fixtures are synthetic validation probes, not captured Intake
traffic and not claims of production format support. They exercise only the
closed fields declared by the test adapter. Unknown content-bearing fields fail
closed.

Their shapes are derived from immutable NeMo Platform sources:

- [Intake ingest-format reference](https://github.com/NVIDIA-NeMo/nemo-platform/blob/e1057736703bb8b167a4bd9013cea0caae2df63a/packages/nemo_platform_ext/src/nemo_platform_ext/skills/nemo-intake/references/ingest-formats.md)
- [ATIF domain and validation rules](https://github.com/NVIDIA-NeMo/nemo-platform/blob/e1057736703bb8b167a4bd9013cea0caae2df63a/services/intake/src/nmp/intake/spans/ingest/atif_domain.py)
- [Chat-completion ingest model](https://github.com/NVIDIA-NeMo/nemo-platform/blob/e1057736703bb8b167a4bd9013cea0caae2df63a/services/intake/src/nmp/intake/spans/ingest/chat_completions.py)
- [OTLP/HTTP protobuf receiver](https://github.com/NVIDIA-NeMo/nemo-platform/blob/e1057736703bb8b167a4bd9013cea0caae2df63a/services/intake/src/nmp/intake/spans/ingest/otlp.py)
- [Local CHAIN-to-LLM OTLP example](https://github.com/NVIDIA-NeMo/nemo-platform/blob/e1057736703bb8b167a4bd9013cea0caae2df63a/services/intake/examples/send_otel_sample.py)

OTLP validation uses real `ExportTraceServiceRequest` protobuf bytes. The probe
withholds the complete batch if any span is invalid or any selected segment
does not produce a successful Plan A outcome. This is stricter than Intake's
current HTTP-200 response with per-span errors and is a proposed adapter policy,
not established Intake behavior.
