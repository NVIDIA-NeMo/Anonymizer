<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Anonymizer integration for NeMo Relay

This integration sanitizes copies of NeMo Relay observability events before
writing them to a protected destination. It does not inspect or change the LLM
requests, LLM responses, or tool traffic used by the application.

```text
application traffic ───────────────► provider and tools (unchanged)
        │
        └─ Relay event copy ─► thin subscriber worker ─► bounded service queue
                                                         │
                                                         └─ Anonymizer ─► protected ATOF JSONL
```

The process split is deliberate. The Relay-managed worker performs a health
check and quickly forwards copied events to a separately managed service. The
service owns the larger Anonymizer environment, batching, sanitization, queue
lifetime, explicit drain, and destination output. Anonymization therefore stays
off the application request path and outside Relay's worker callback lifetime.

Selected raw observability text is sent to both the configured entity detector
and LLM evaluator. Both endpoints must be inside the deployment's approved data
boundary. The defaults use a local detector and NVIDIA's hosted evaluator; a
fully local deployment must also configure a local evaluator. Detection is
probabilistic and can produce both false positives and false negatives.

## Protection boundary

Only output written by this integration is protected. Relay subscribers
configured beside it—including ATOF, OTLP, and Phoenix exporters—continue to
receive the original unsanitized event copy. Do not enable those destinations
for data that must pass through Anonymizer first.

The exporter retains trace identifiers and relationships while inspecting
caller-controlled text in supported Relay event fields. It omits exact provider
bodies when a normalized LLM annotation is available, removes streaming chunk
payloads, and emits an envelope-only omission record when projection or
anonymization fails. It never falls back to exporting the original payload.

This first version uses Anonymizer's full detection pipeline: GLiNER candidate
detection followed by LLM validation and augmentation. A detector-only profile
is not part of the initial supported behavior.

| Event content | Behavior |
|---|---|
| Relay trace IDs, timing, lifecycle, and known scope flags | Preserved |
| Normalized LLM requests and responses | Text is inspected; duplicate exact provider bodies are omitted |
| Tool and custom event JSON | String values and data-shaped keys are inspected |
| Credentials with known token shapes or secret-bearing keys | Removed before detector or evaluator calls |
| Streaming chunk payloads | Omitted; the trace receipt is preserved |
| Recognized media objects, data URLs, and long encoded blobs | Omitted rather than claimed as inspected |
| Projection or Anonymizer failure | Envelope-only omission record; original payload is never exported |

## Operation

Start the exporter service before enabling the Relay worker. The worker refuses
activation when the service health check fails. The initial integration accepts
only an owner-only Unix socket in an owner-only directory and therefore targets
Linux and macOS. TCP and Windows are not supported in this first version.

The service queue is bounded by event count and total bytes. Admission returns
after an event is placed in memory, before anonymization or destination output
finishes. Operators must explicitly flush and stop the service during graceful
shutdown. Relay does not currently own that service lifecycle.

### Run from a source checkout

Install the worker/test environment and the larger service environment
separately:

```bash
uv sync --python 3.11 --project integrations/nemo-relay --group test --locked
uv sync --python 3.11 --project integrations/nemo-relay/service --no-dev --locked
```

Start a compatible GLiNER endpoint as described in the
[self-hosting guide](https://github.com/NVIDIA-NeMo/Anonymizer/blob/main/docs/concepts/self-hosting-gliner.md),
then start the exporter. Create a private runtime directory first. The
evaluator credential exists only in the service process:

```bash
install -d -m 700 "$HOME/.cache/nemo-anonymizer-relay"
NVIDIA_API_KEY="..." \
uv run --project integrations/nemo-relay/service --no-dev --locked \
  python -m nemo_anonymizer_relay.service serve \
  --endpoint "unix://$HOME/.cache/nemo-anonymizer-relay/exporter.sock" \
  --output /absolute/path/protected-events.jsonl
```

Materialize and install the thin Relay worker bundle:

```bash
uv run --project integrations/nemo-relay --group test --locked \
  python integrations/nemo-relay/scripts/package_bundle.py \
  --output /tmp/nemo-anonymizer-relay-bundle
nemo-relay plugins add --user /tmp/nemo-anonymizer-relay-bundle/relay-plugin.toml
nemo-relay plugins edit
```

Configure the generated plugin entry before enabling it:

```toml
[plugins.dynamic.config]
version = 1
endpoint = "unix:///absolute/path/to/.cache/nemo-anonymizer-relay/exporter.sock"
```

```bash
nemo-relay plugins enable nvidia.nemo_anonymizer
nemo-relay plugins validate nvidia.nemo_anonymizer
```

Before stopping the service, drain and shut it down through the same private
socket:

```bash
uv run --project integrations/nemo-relay/service --no-dev --locked \
  python -m nemo_anonymizer_relay.service flush \
  --endpoint "unix://$HOME/.cache/nemo-anonymizer-relay/exporter.sock"
uv run --project integrations/nemo-relay/service --no-dev --locked \
  python -m nemo_anonymizer_relay.service shutdown \
  --endpoint "unix://$HOME/.cache/nemo-anonymizer-relay/exporter.sock"
```

## Current limitations

- Accepted events are held in memory. A service or machine failure can lose
  events that were acknowledged but not yet exported.
- The destination is protected only when competing raw Relay subscribers are
  disabled.
- Service health proves queue and sink readiness, but does not yet preflight
  detector or evaluator credentials and connectivity.
- Recognized media and long encoded blobs are omitted. Provider-native
  representations without a Relay annotation do not have a complete
  inspection guarantee.
- Short opaque encodings without a media or encoding marker are treated as
  visible text. Their decoded contents are not inspected; applications should
  label attachments or omit them before publication.
- The service has no durable spool, restart recovery, or destination retry.
- Relay worker shutdown does not flush or stop the external service.
- The current protected destination is ATOF JSONL; OTLP and Phoenix export are
  not implemented by this service.
- The thin Relay worker can be packaged independently, but the heavy service
  environment currently resolves this Anonymizer source checkout. A release
  needs a separately installable service artifact or container with a
  relocatable lock.
- The manifest requires Relay 0.9.1 or newer because packaged Python workers
  depend on the Linux environment-attestation fix in that release line.

This integration is an implementation spike. It is not yet a production
readiness claim.

## Development

The integration is an independent Python project so Relay-specific dependencies
do not enter the main `nemo-anonymizer` wheel.

```bash
uv sync --python 3.11 --project integrations/nemo-relay --group test --locked
uv run --project integrations/nemo-relay --group test --locked \
  pytest -q integrations/nemo-relay/tests
uv sync --python 3.11 --project integrations/nemo-relay/service --group dev --locked
uv run --project integrations/nemo-relay/service --group dev --locked \
  ty check --project integrations/nemo-relay
```

Behavioral tests live with this source. Cross-platform release assembly,
attribution generation, and installed-archive smoke tests belong to the
NeMo Relay Plugins release repository.
