<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Use NeMo Anonymizer with NeMo Relay

This integration removes sensitive information from Relay telemetry before it
is sent to ATOF, OpenTelemetry, Phoenix, or another subscriber. It works only on
Relay's copy of the event. The request sent to the model, the model response,
and tool input and output used by the application are not changed.

> [!WARNING]
> This integration is experimental and is not ready for production use. The
> current version is too slow and too large for Relay's worker limits. Relay
> clears slow event data instead of risking unredacted telemetry.
> [Issue #289](https://github.com/NVIDIA-NeMo/Anonymizer/issues/289) tracks the
> faster Anonymizer API needed before release.

```text
request, response, or tool data ─────────────────► application (unchanged)
               │
               └─ Relay telemetry copy
                          ↓
                  Anonymizer worker
                          ↓
                 redacted telemetry
                          ↓
              ATOF / OTLP / Phoenix / subscribers
```

## How it works

Relay runs the integration as a managed Python worker. The worker handles mark,
scope-start, and scope-end events. Relay has already added the LLM or tool
details by the time the worker receives an event, so the worker can sanitize
the event's `data`, `category_profile`, and `metadata` together in one call.

The worker does not intercept application traffic, export telemetry itself, or
run a separate queue or service.

## What is covered

| Data | Behavior |
|---|---|
| LLM requests and responses | Checks the copied provider payload, Relay's normalized view, and metadata. |
| Tool calls and results | Checks copied arguments, results, tool annotations, and metadata. |
| Marks and custom scopes | Checks the event's `data`, `category_profile`, and `metadata` fields. |
| Streaming chunks | Keeps the trace event but removes the chunk payload. The final LLM response is checked normally. |
| Common credentials | Removes recognized API keys, bearer tokens, passwords, cookies, and similar secret values before sending text to Anonymizer. |
| Inline media | Removes data URIs and large encoded bodies. Image, audio, and file contents are not inspected. |
| Sensitive JSON keys | Rejects the event payload if a mapping key contains PII or a credential. Keys are never renamed because that could break the event schema. |
| Errors and timeouts | Relay keeps trace timing and IDs but clears the fields that may contain sensitive data. |

Relay does not allow sanitizers to change event names, categories, schema names,
timestamps, UUIDs, lifecycle phases, or parent relationships. Do not place PII
in those fields.

PII detection is probabilistic. It can miss sensitive data or redact ordinary
text by mistake.

## Current backend and limitations

The current backend uses Anonymizer's full pipeline:

1. GLiNER finds possible PII.
2. An LLM validates and expands those findings.
3. The worker uses the returned character positions to redact the copied event.

Both configured model endpoints receive selected telemetry text. Only use
endpoints that are approved to receive this telemetry.

The worker loads one Anonymizer pipeline when it is first needed and reuses it
for later events. Calls are processed one at a time because the current
Anonymizer pipeline does not support concurrent use. Temporary artifacts are
removed after a completed call.

This backend is useful for testing the integration, but it cannot ship in its
current form:

- It exceeds Relay's 30-second callback limit.
- Its last packaged environment was about 541 MiB, above Relay's 512 MiB limit.
- A forced worker shutdown may happen before a synchronous Anonymizer call has
  removed its temporary files.
- Some internal Anonymizer or provider errors may log source text.

A slow sanitizer does not slow the model or tool call. It delays delivery of
that telemetry event. If the callback times out, Relay clears `data`,
`category_profile`, and `metadata` before sending the event to subscribers.

## Try the integration

Use this only for development until the warning above is removed.

You need:

- NeMo Relay 0.9.1 or later, but earlier than 1.0;
- Python 3.11;
- an OpenAI-compatible GLiNER endpoint;
- an OpenAI-compatible evaluator endpoint; and
- an extracted development bundle of this plugin.

Validate and add an extracted plugin bundle:

```bash
nemo-relay plugins validate ./nemo-anonymizer-relay/relay-plugin.toml
nemo-relay plugins add --user ./nemo-anonymizer-relay/relay-plugin.toml
nemo-relay plugins edit
```

Add the configuration below to the generated `[[plugins.dynamic]]` entry. Keep
the generated entry itself because it also records the worker environment.

```toml
[plugins.dynamic.config]
version = 1
priority = 100
detector_endpoint = "http://127.0.0.1:8001/v1"
detector_model = "fastino/gliner2-privacy-filter-PII-multi"
detector_api_key_env = "EMPTY"
evaluator_endpoint = "https://integrate.api.nvidia.com/v1"
evaluator_model = "nvidia/nemotron-3.5-lightning-30b-a3b"
evaluator_api_key_env = "NVIDIA_API_KEY"

[plugins.dynamic.config.secret_env]
NVIDIA_API_KEY = "..."
```

Relay does not pass the host environment directly to workers. Values under
`secret_env` are copied into this worker as literal environment variables; they
are not secret-store references. Protect the plugin configuration file. The
worker accepts only the credential variable names selected by
`detector_api_key_env` and `evaluator_api_key_env`.

Enable the plugin and validate the saved configuration:

```bash
nemo-relay plugins enable nvidia.nemo_anonymizer
nemo-relay plugins validate nvidia.nemo_anonymizer
```

See [`config.schema.json`](config.schema.json) for the complete configuration.

## Develop and test

The integration is a separate Python project so its Relay dependencies do not
become dependencies of the main Anonymizer package.

```bash
uv sync --python 3.11 --project integrations/nemo-relay --group test --locked
uv run --project integrations/nemo-relay --group test --locked \
  pytest -q integrations/nemo-relay/tests
uv run --project integrations/nemo-relay --group test --locked \
  ruff check integrations/nemo-relay
uv run --project integrations/nemo-relay --group test --locked \
  ty check --project integrations/nemo-relay
```

`scripts/package_bundle.py` prepares the Anonymizer source used by a release
build. It does not create the final Relay plugin archive or collect its full
dependency set and license notices.

Local development uses this Anonymizer checkout. The staged bundle currently
requires `nemo-anonymizer>0.4.0,<1` because version 0.4.0 does not include the
required in-memory input API. Do not publish the bundle until that API is
released and the dependency is pinned to an exact supported version.

## Packaging and release

The Anonymizer repository owns this source, its tests, configuration schema,
and manifest template. The
[NeMo Relay Plugins](https://github.com/NVIDIA/NeMo-Relay-Plugins) repository
will pin an exact Anonymizer commit and build the final archives, dependency
locks, license notices, checksums, and platform tests.

Before release, this integration still needs:

- the faster Anonymizer API from #289;
- an exact released Anonymizer dependency;
- tests covering forced shutdown and source-free error logs;
- a worker environment comfortably below Relay's 512 MiB limit; and
- a final Relay and Codex test showing that application traffic keeps its
  original data while subscribers receive useful redacted telemetry.
