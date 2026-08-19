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

`tests/streaming/test_intake_dogfood.py` is an opt-in integration test for an
operator-owned Intake deployment. Set `ANONYMIZER_INTAKE_DOGFOOD_BASE_URL` to
the service origin to enable it. Its default before-Intake profile sends only
protected requests and checks that every declared synthetic PII value is absent
from the outbound bytes. Raw-format and partial-write characterization requires
the additional explicit `ANONYMIZER_INTAKE_DOGFOOD_ALLOW_RAW=1` opt-in and is
valid only for an isolated deployment approved to receive raw synthetic data.
The protected-only profile also wires invalid OTLP to a real Intake emitter and
checks that adapter rejection prevents the emitter call and leaves no stored
session.

The protected-delivery probes keep retry ownership outside Anonymizer. A
pre-connect failure raises a bounded, cause-free test error; it does not alter
the protected bytes or create an Intake row. The probe then sends that same
protected byte string successfully and verifies one stable public row. A
commit-then-forget approximation separately resends exact protected bytes to
characterize an ambiguous delivery.
Intake's public read model collapses repeated ATIF, chat-completion, and OTLP
records for the fixed fixture identities. The chat contract requires a
positive, representable, non-future integer `response.created` and preserves it
unchanged because Intake uses that value as the span start time; if Intake
cannot accept it, Intake assigns a new ingestion time and exposes an exact
retry as another row. This does not establish
transactional idempotency or describe the number of physical ClickHouse writes.
A source adapter must retain the exact protected bytes for retry and must reject
a chat item that lacks a stable creation time.

The test owns only its synthetic requests and unique identifiers: it does not
provision, configure, stop, or clean up Intake, ClickHouse, containers, or data.
Its persisted synthetic rows remain under the operator's retention policy.
