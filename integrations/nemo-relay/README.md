<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Anonymizer integration for NeMo Relay

This integration runs NeMo Anonymizer inside one Relay-managed Python worker.
It sanitizes Relay's copied observability values before Relay sends the event
to its normal subscribers. Provider requests, tool calls, and application
results remain unchanged.

```text
application traffic ─────────────────────────────► provider/tools (unchanged)
        │
        └─ copied observability value
                    ↓
           Anonymizer worker sanitizer
                    ↓ returns protected copy
                Relay runtime
                    ↓
          ATOF / OTLP / Phoenix / custom subscribers
```

The worker does not run a private service, socket, queue, or exporter. Relay
owns its process and authenticated transport. The integration follows the
same middleware shape as Rampart: it registers mark, scope-start, scope-end,
tool request/response, and LLM request/response sanitizers.

## What it protects

| Surface | Behavior |
|---|---|
| LLM request | Uses Relay's active request codec when available, sanitizes the semantic annotation and exact copied request, then returns a provider-shaped observability copy |
| LLM response | Sanitizes the copied provider response, then validates the returned shape with Relay's active response codec when available |
| Tool request and response | Sanitizes copied JSON values and application-defined keys |
| Marks and ordinary scopes | Sanitizes `data`, `category_profile`, and `metadata` |
| LLM and tool scopes | Rechecks every mutable field so generic scopes, compatibility fallbacks, and future profile fields cannot bypass the typed sanitizers |
| Streaming chunk marks | Omits the chunk payload while retaining Relay's trace envelope |
| Credentials and recognized media | Removes known secret shapes before model calls; replaces media and large encoded bodies rather than claiming inspection |
| Sanitizer failure or timeout | Relay clears the mutable observability fields before subscriber fan-out |

Relay intentionally keeps event identity immutable. Names, categories, schema
names, UUIDs, timestamps, lifecycle phases, and parentage are not writable by
sanitizers and must not contain PII.

Detection uses Anonymizer's full pipeline: GLiNER candidate detection followed
by LLM validation and augmentation. Both configured endpoints must be inside
the deployment's approved data boundary. Detection is probabilistic and can
produce false positives and false negatives.

## Runtime behavior

Relay queues copied observability work off the provider and tool execution
path. Its observability dispatcher then waits for Anonymizer before delivering
that event to subscribers. `flush_subscribers` therefore covers both
sanitization and downstream delivery; there is no hidden worker backlog.

The worker creates one Anonymizer pipeline lazily, serializes calls through it,
and removes its per-call artifact tree after each completed run. This avoids
reloading the pipeline for every event. Relay can still terminate a worker
whose synchronous model call outlives shutdown, so cleanup after forced
termination remains a release gate.

Each dynamic-worker callback has a 30-second host timeout. Current full
Anonymizer measurements exceed that budget, so this remains an implementation
spike. A slow callback does not delay the application call, but it does delay
the serial observability dispatcher and will eventually produce an event with
its mutable fields cleared.

## Configuration

Relay scrubs the worker's inherited environment. `secret_env` contains literal
values copied into this worker process; it is not a secret-reference mechanism.
Protect the plugin configuration file.

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

See `config.schema.json` for text budgets, cache controls, and provider timeout
settings.

## Development

The integration is a separate Python project so Relay dependencies do not
enter the main Anonymizer wheel.

```bash
uv sync --python 3.11 --project integrations/nemo-relay --group test --locked
uv run --project integrations/nemo-relay --group test --locked \
  pytest -q integrations/nemo-relay/tests
uv run --project integrations/nemo-relay --group test --locked \
  ruff check integrations/nemo-relay
uv run --project integrations/nemo-relay --group test --locked \
  ty check --project integrations/nemo-relay
```

The development lock resolves this Anonymizer checkout. The bundle generator
removes that local source override and deliberately requires a release newer
than 0.4.0, which is the current release without the required in-memory API.
Do not publish or install the bundle until the prerequisite release exists and
the dependency is replaced with an exact supported pin.

## Release gates

- Provide an online Anonymizer profile whose per-event latency stays
  comfortably below Relay's 30-second callback limit.
- Prove that copied artifacts are removed when Relay terminates a worker with
  an in-flight model call; normal callback and plugin cleanup already pass.
- Build a managed environment below Relay's 512 MiB closure limit on every
  supported platform.
- Publish the required Anonymizer API and replace the bundle's temporary
  `>0.4.0,<1` requirement with an exact supported version.
- From the final checksummed archive, prove that provider/tool inputs retain
  their original PII while ordinary ATOF, OTLP, and Phoenix subscribers receive
  only the sanitized copy.

This integration is not yet a production-readiness claim.
