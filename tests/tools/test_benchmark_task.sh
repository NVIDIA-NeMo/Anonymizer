#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
task="$repo_root/.mise/tasks/benchmark"
tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/anonymizer-benchmark-task.XXXXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT

cat >"$tmpdir/uv" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$@" >"${CAPTURE_FILE:?}"
EOF
chmod +x "$tmpdir/uv"

run_task() {
    local profile="$1"
    local runner_args="${2:-}"
    local capture_file="$tmpdir/${profile}.actual"

    PATH="$tmpdir:$PATH" \
        CAPTURE_FILE="$capture_file" \
        usage_profile="$profile" \
        usage_runner_args="$runner_args" \
        bash "$task"
}

run_task smoke
cat >"$tmpdir/smoke.expected" <<'EOF'
run
--locked
--group
dev
python
tools/measurement/run_benchmarks.py
tools/measurement/examples/repo-data-smoke.yaml
--output
benchmark-runs/smoke
EOF
diff -u "$tmpdir/smoke.expected" "$tmpdir/smoke.actual"

run_task smoke-traces
cat >"$tmpdir/smoke-traces.expected" <<'EOF'
run
--locked
--group
dev
python
tools/measurement/run_benchmarks.py
tools/measurement/examples/repo-data-smoke.yaml
--output
benchmark-runs/smoke-traces
--dd-trace
last_message
--dd-task-trace
EOF
diff -u "$tmpdir/smoke-traces.expected" "$tmpdir/smoke-traces.actual"

run_task smoke "--wandb-run-name 'test run'"
cat >>"$tmpdir/smoke.expected" <<'EOF'
--wandb-run-name
test run
EOF
diff -u "$tmpdir/smoke.expected" "$tmpdir/smoke.actual"
