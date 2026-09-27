<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Anonymizer integration for NeMo Relay

> [!WARNING]
> This is an implementation draft, not a production-ready integration. The
> last full-pipeline qualification measured about 117 seconds cold and 40
> seconds warm against Relay's 30-second callback limit. [Issue
> #289](https://github.com/NVIDIA-NeMo/Anonymizer/issues/289) proposes the
> faster detector API; that API is not part of this branch.

The integration protects **copied observability data** before Relay sends it to
ATOF, OpenTelemetry, Phoenix, or another subscriber. It never changes the
provider request, model response, tool arguments, or tool result used by the
application.

```text
application traffic ─────────────────────────────► provider/tools (unchanged)
        │
        └─ copied observability value
                    ↓
           Relay-managed Python worker
                    ↓ protected copy
                Relay runtime
                    ↓
          ATOF / OTLP / Phoenix / custom subscribers
```

The worker registers Relay's existing mark and scope event sanitizers. Relay
first assembles the final copied LLM or tool event; Anonymizer then checks its
data, normalized profile, and metadata together in one pass. The integration
adds no application-path interceptor, exporter, private queue, or service.

## Coverage

| Surface | Behavior |
|---|---|
| LLM request and response events | Sanitizes the final copied native payload, normalized annotation, and metadata together after Relay constructs the event. |
| Tool request and response events | Sanitizes copied arguments or results, tool annotations, and metadata together. |
| Marks and custom scopes | Sanitizes the mutable `data`, `category_profile`, and `metadata` fields. |
| Streaming chunks | Retains the trace envelope but deliberately omits each chunk payload. |
| Secrets and inline media | Removes recognized credential values before model calls and omits data URIs or large encoded bodies without changing the surrounding JSON shape. |
| Sensitive JSON keys | Inspects mapping keys. If a key itself contains PII or a credential, the event fails closed instead of changing its schema or leaking the key. |
| Failure, unsafe output, or timeout | Fails closed; Relay clears the event's mutable observability fields before subscriber fan-out. |

Image, audio, and file contents are not inspected; inline encoded bodies are
omitted. Detection is probabilistic and can produce false positives and false
negatives.

Relay keeps event identity immutable. Names, categories, schema names, UUIDs,
timestamps, lifecycle phases, and parentage cannot be rewritten by a sanitizer
and must not contain PII.

## Current runtime

The committed backend runs Anonymizer's complete detection pipeline: GLiNER
candidate detection followed by LLM validation and augmentation. Both model
endpoints therefore receive selected copied text and must be inside the
deployment's approved data boundary.

The worker creates one pipeline lazily, serializes calls through it, and removes
completed-call artifacts. Relay performs observability work away from provider
and tool execution, but its subscriber dispatcher still waits for sanitization
before delivering that event. A slow callback does not delay the application
call; it delays observability delivery and ultimately causes fail-closed
omission.

Relay limits each worker callback to 30 seconds. Provider requests are capped
below that limit, but the full multi-stage pipeline can make several calls and
still exceed the outer deadline. Relay may also terminate a worker before a
cancelled synchronous call finishes cleaning temporary data. Both behaviors
remain release blockers. The last packaged environment was also approximately
541 MiB, above Relay's 512 MiB managed-environment limit.

## Configuration

Validate and add the extracted plugin bundle. `add` creates a disabled entry in
the user plugin registry:

```bash
nemo-relay plugins validate ./nemo-anonymizer-relay/relay-plugin.toml
nemo-relay plugins add --user ./nemo-anonymizer-relay/relay-plugin.toml
nemo-relay plugins edit
```

Add the following table under the generated `[[plugins.dynamic]]` entry. Do not
replace that entry by hand; it also records the managed worker environment.

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

Relay scrubs the worker's inherited environment. `secret_env` contains literal
values copied into this worker process; it is not a secret-reference mechanism.
Protect the plugin configuration file. See `config.schema.json` for all work
budgets and timeouts.

Then enable and validate the configured plugin:

```bash
nemo-relay plugins enable nvidia.nemo_anonymizer
nemo-relay plugins validate nvidia.nemo_anonymizer
```

## Development

Development is independently locked so Relay dependencies do not enter the main
Anonymizer wheel.

```bash
uv sync --python 3.11 --project integrations/nemo-relay --group test --locked
uv run --project integrations/nemo-relay --group test --locked \
  pytest -q integrations/nemo-relay/tests
uv run --project integrations/nemo-relay --group test --locked \
  ruff check integrations/nemo-relay
uv run --project integrations/nemo-relay --group test --locked \
  ty check --project integrations/nemo-relay
```

`scripts/package_bundle.py` stages the upstream runtime and source payload for
downstream release packaging. It does not create the final archive, dependency
closure, attribution files, or qualification evidence. The CI wheel build only
checks that the Python source can be packaged; the wheel by itself is not the
Relay plugin.

Development resolves this Anonymizer checkout locally. Do not publish the
staged bundle until the required APIs have shipped and the temporary
`nemo-anonymizer>0.4.0,<1` dependency has been replaced by an exact supported
version.

## Source and release ownership

Anonymizer owns this source, its tests, schema, and manifest template. The
[NeMo Relay Plugins](https://github.com/NVIDIA/NeMo-Relay-Plugins) repository
should pin this directory at an exact Anonymizer commit and own the reproducible
dependency closure, platform builds, attributions, final archives, checksums,
and installed-bundle smoke tests.

The manifest's integrity field covers the worker entry point, not every imported
module. Release integrity therefore also depends on the pinned source commit and
the checksum of the complete archive.

## Release gates

- Replace the full pipeline with an Anonymizer-owned online profile that stays
  comfortably below Relay's callback limit and has an explicit accuracy policy.
- Publish and exactly pin the required Anonymizer API.
- Prove cleanup after forced worker termination.
- Prove that worker and provider logs cannot echo copied source text.
- Keep the managed environment below Relay's 512 MiB limit on every supported
  platform with reasonable headroom and a reproducible dependency closure.
- Install the final checksummed archive and repeat the Relay/Codex qualification,
  proving that provider/tool traffic keeps its original PII while subscribers
  receive useful sanitized copies rather than empty fail-closed payloads.
