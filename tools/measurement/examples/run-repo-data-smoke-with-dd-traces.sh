#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

output_dir="${1:-/tmp/anonymizer-repo-data-smoke-dd-traces}"
trace_mode="${DD_TRACE_MODE:-last_message}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
suite_file="${BENCHMARK_SUITE:-${script_dir}/repo-data-smoke.yaml}"

cd "${repo_root}"

uv run python tools/measurement/run_benchmarks.py \
  "${suite_file}" \
  --output "${output_dir}" \
  --overwrite \
  --dd-trace "${trace_mode}" \
  --trace-dir "${output_dir}/traces"
