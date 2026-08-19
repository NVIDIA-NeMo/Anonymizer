<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Local Intake dogfood runbook

Use this runbook to start an isolated NeMo Platform Intake service with managed
ClickHouse, run Anonymizer's protected-only dogfood, and inspect the resulting
spans. This is a local development procedure, not a production deployment or a
production Intake integration.

The current Anonymizer design uses a provisional **before-Intake** boundary.
Only protected payloads may enter the Intake service process. Do not enable the
raw characterization cases unless the deployment is isolated and the operator
has explicitly approved receiving raw synthetic fixtures.

## Prerequisites

You need:

- a NeMo Platform source checkout with its Python environment bootstrapped;
- Docker with a reachable daemon;
- this Anonymizer worktree and its development environment; and
- an unused loopback port, normally `8080`.

The procedure was validated with:

- NeMo Platform revision `e1057736703bb8b167a4bd9013cea0caae2df63a`;
- ClickHouse 26.3;
- NeMo Platform at `/root/dev/nemo-platform`; and
- Anonymizer at `/root/dev/wt/Anonymizer-openshell-intake`.

Other revisions may change CLI options, endpoint behavior, or storage layout.
Use NeMo Platform's `SETUP.md` and `services/intake/README.md` as the authority
when updating this procedure.

If the NeMo Platform checkout is not bootstrapped, follow its canonical setup
guide. For a Python-only source environment, the relevant bootstrap target is:

```bash
cd /root/dev/nemo-platform
make bootstrap-python
```

## 1. Check for an existing service

Do not start a second platform blindly on the same port or data directory.
Check the ready endpoint and running commands first:

```bash
curl --fail --silent --show-error \
  http://127.0.0.1:8080/health/ready || true

pgrep -af '[n]emo services run' || true
```

If the ready endpoint returns `{"status":"ready"}`, either reuse that approved
deployment or choose another port and a separate data directory. Do not stop an
existing service or remove its container or data without operator approval.

## 2. Create isolated local state

Create a unique root and print it so another terminal or later session can
reuse the same deployment:

```bash
INTAKE_DOGFOOD_ROOT="$(mktemp -d /tmp/nemo-intake-dogfood.XXXXXX)"
mkdir -p \
  "$INTAKE_DOGFOOD_ROOT/data" \
  "$INTAKE_DOGFOOD_ROOT/state" \
  "$INTAKE_DOGFOOD_ROOT/cache/uv"
printf 'Intake dogfood root: %s\n' "$INTAKE_DOGFOOD_ROOT"
```

The paths have distinct ownership:

```text
$INTAKE_DOGFOOD_ROOT/
├── data/       NeMo Platform SQLite state and managed ClickHouse data
├── state/nmp/  local-service state and runtime metadata
└── cache/uv/   uv cache for this isolated run
```

Retain the printed root for restarts. A new root creates a different managed
ClickHouse identity and an empty Intake deployment.

## 3. Start Intake and managed ClickHouse

Run the platform in the foreground from the NeMo Platform repository:

```bash
cd /root/dev/nemo-platform

env \
  NMP_DATA_DIR="$INTAKE_DOGFOOD_ROOT/data" \
  XDG_STATE_HOME="$INTAKE_DOGFOOD_ROOT/state" \
  UV_CACHE_DIR="$INTAKE_DOGFOOD_ROOT/cache/uv" \
  NMP_INTAKE_CLICKHOUSE_IMAGE=clickhouse/clickhouse-server:26.3 \
  uv run nemo services run \
    --services auth,entities,intake \
    --host 127.0.0.1 \
    --port 8080
```

When `NMP_INTAKE_CLICKHOUSE_URL` is unset, Intake provisions a managed
ClickHouse container. It binds the ClickHouse HTTP port to a Docker-assigned
loopback port and stores data under
`$INTAKE_DOGFOOD_ROOT/data/intake-clickhouse/`. Run only one local platform
process against a given data directory.

Keep this terminal open. Use another terminal for the remaining commands.

## 4. Verify both service and storage

First check aggregate readiness:

```bash
curl --fail --silent --show-error \
  http://127.0.0.1:8080/health/ready
```

Expected response:

```json
{"status":"ready"}
```

Then exercise Intake's ClickHouse-backed read path:

```bash
curl --fail-with-body --silent --show-error --get \
  'http://127.0.0.1:8080/apis/intake/v2/workspaces/default/spans' \
  --data-urlencode 'page=1' \
  --data-urlencode 'page_size=1' \
  | jq .
```

Continue only after this request returns HTTP 200. An empty `data` list is
healthy. HTTP 503 means Intake cannot reach ClickHouse even if the aggregate
ready endpoint succeeds.

Inspect managed ClickHouse without changing it:

```bash
docker ps --all \
  --filter 'label=nmp.nvidia.com/component=intake-clickhouse' \
  --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'
```

If several managed containers exist, use their bind mounts and the printed
dogfood root to identify the current one:

```bash
docker inspect \
  --format '{{.Name}} {{range .Mounts}}{{.Source}} -> {{.Destination}}{{end}}' \
  CONTAINER_NAME
```

## 5. Run protected-only Anonymizer dogfood

Run the checked-in integration test from the Anonymizer worktree:

```bash
cd /root/dev/wt/Anonymizer-openshell-intake

env \
  PYTHONPATH=. \
  ANONYMIZER_INTAKE_DOGFOOD_BASE_URL=http://127.0.0.1:8080 \
  uv run --frozen pytest tests/streaming/test_intake_dogfood.py -q
```

This default profile sends only protected payloads and leaves its synthetic
rows under the deployment's retention policy. It does not provision, stop, or
clean up Intake or ClickHouse.

To include a completed Sandbox Codex session created from
[`sandbox_agent_prompt.md`](../tests/fixtures/streaming/sandbox_agent_prompt.md),
add its run directory:

```bash
env \
  PYTHONPATH=. \
  ANONYMIZER_INTAKE_DOGFOOD_BASE_URL=http://127.0.0.1:8080 \
  ANONYMIZER_SANDBOX_DOGFOOD_RUN_DIR=/stable-cache/sandbox/runs/RUN_NAME \
  uv run --frozen pytest tests/streaming/test_intake_dogfood.py -q
```

The Sandbox run must be completed successfully and must use the declared
synthetic values. The test-only adapter rejects unsupported completed event
types and never launches or manages Sandbox.

Do not set `ANONYMIZER_INTAKE_DOGFOOD_ALLOW_RAW=1` during normal dogfood. That
flag intentionally sends raw synthetic fixtures and exists only to characterize
Intake on a separately approved isolated deployment.

## 6. Query a stored trace

List the newest distinct session IDs from the isolated deployment:

```bash
curl --fail --silent --show-error --get \
  'http://127.0.0.1:8080/apis/intake/v2/workspaces/default/spans' \
  --data-urlencode 'page=1' \
  --data-urlencode 'page_size=100' \
  | jq -r '
      .data
      | sort_by(.ingested_at)
      | reverse
      | .[]
      | [.ingested_at, .session_id]
      | @tsv
    ' \
  | awk '!seen[$2]++'
```

Replace `SESSION_ID` below with the relevant value:

```bash
curl --fail --silent --show-error --get \
  'http://127.0.0.1:8080/apis/intake/v2/workspaces/default/spans' \
  --data-urlencode 'filter[session_id]=SESSION_ID' \
  --data-urlencode 'page=1' \
  --data-urlencode 'page_size=100' \
  | jq .
```

For a compact topology and content view:

```bash
curl --fail --silent --show-error --get \
  'http://127.0.0.1:8080/apis/intake/v2/workspaces/default/spans' \
  --data-urlencode 'filter[session_id]=SESSION_ID' \
  --data-urlencode 'page=1' \
  --data-urlencode 'page_size=100' \
  | jq '.data[] | {
      span_id,
      parent_span_id,
      kind,
      name,
      tool_name,
      input,
      output,
      raw_attributes,
      ingested_at
    }'
```

In Intake's read model, `raw_attributes` means retained source attributes. For
the protected dogfood it contains the protected ATIF representation, not the
unprotected Sandbox trace.

## 7. Stop without deleting data

Return to the foreground service terminal and press `Ctrl-C`. A graceful stop
stops the managed ClickHouse container but does not remove the container or its
bind-mounted data. Restart with the same dogfood root and the command from step
3 to reuse the deployment.

After stopping, verify the platform process is gone and record the container
state:

```bash
pgrep -af '[n]emo services run' || true

docker ps --all \
  --filter 'label=nmp.nvidia.com/component=intake-clickhouse' \
  --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
```

Removing the managed container or deleting `$INTAKE_DOGFOOD_ROOT` is a separate,
destructive operation. It requires explicit operator approval and must follow
NeMo Platform's managed teardown procedure. Do not use an unscoped `docker rm`
or recursive deletion command from this runbook.

## Validated local instance

The 2026-08-19 dogfood used this instance. These identifiers are evidence, not
reusable configuration:

```text
Intake:         http://127.0.0.1:8080
Dogfood root:   /tmp/nemo-intake-dogfood.AgO959
ClickHouse:     nmp-intake-clickhouse-b02569d8fb94
Image:          clickhouse/clickhouse-server:26.3
ClickHouse HTTP 127.0.0.1:32768 -> 8123/tcp
```

The deployment stored protected ATIF, chat-completion, and OTLP/protobuf
fixtures, plus a protected ATIF trajectory derived from a real Sandbox Codex
session. See the
[`extensible SDK companion report`](../docs/development/extensible-sdk-companion-plans.md#intake-workload-validation)
for the validated behavior and remaining contract limits.
