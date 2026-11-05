#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Agent traces: anonymize with the NeMo Anonymizer CLI, upscale with the NeMo Data Designer CLI.
# Configs sit next to this script; the LLM is vLLM on localhost:8000 (edit model_providers.yaml to change).
# Needs curl, jq and nemo-anonymizer[notebooks] on PATH (or VENV=/path/to/venv). Sizes: N_TRACES=10 N_SYNTHETIC=1000.
set -euo pipefail
CFG="$(cd "$(dirname "$0")" && pwd)"
export PATH="${VENV:+$VENV/bin:}$PATH" DATA_DESIGNER_HOME="${OUT:=$CFG/out}/.data-designer"
mkdir -p "$DATA_DESIGNER_HOME" && cp "$CFG/model_providers.yaml" "$DATA_DESIGNER_HOME/" && cd "$OUT"
LOG="$CFG/run_$(date +%Y%m%d_%H%M%S).log"; exec > >(tee "$LOG") 2>&1     # everything below is also saved to $LOG
log() { printf '\n\033[1m[%s] %s\033[0m\n' "$(date +%T)" "$*"; }

log "log file: $LOG"
log "1/4 load ${N_TRACES:-10} tau-bench retail traces"
curl -fsS "https://datasets-server.huggingface.co/rows?dataset=lefft/tau-dev-task-retail-v1&config=default&split=train&length=${N_TRACES:-10}" \
  | jq '[.rows[].row]' > raw.json
jq -r -f "$CFG/transcript.jq" raw.json > traces.csv
log "loaded $(jq length raw.json) traces, one transcript per trace -> traces.csv"

log "2/4 anonymize (Redact: GLiNER2 detects, the LLM validates and augments) with the Anonymizer CLI"
curl -fs localhost:8001/v1/models >/dev/null || {
  log "starting the local GLiNER2 detector on :8001 (log: $CFG/gliner2.log)"
  PYTHONPATH="$(python -c 'from anonymizer.notebooks._runtime import _ensure_server_environment as e; print(e())')" \
    python "$(python -c 'import anonymizer.notebooks.local_inference.gliner2 as g; print(g.__path__[0])')/server.py" > "$CFG/gliner2.log" 2>&1 &
  trap "kill $!" EXIT
  until curl -fs localhost:8001/v1/models >/dev/null; do sleep 2; done
}
anonymizer run --source traces.csv --id-column trace_id --replace redact \
  --excluded-entity-labels monetary_amount --excluded-entity-labels country \
  --data-summary "Retail support agent traces. Each turn starts with a chat-template role marker (<|user|>, <|assistant|>, <|tool_call:NAME|>, <|tool_result|>); markers are structure, never entities. Tool results are JSON customer profiles with addresses, emails, payment methods and order IDs." \
  --model-configs "$CFG/anonymizer_models.yaml" --model-providers "$CFG/model_providers.yaml" --output anonymized.parquet

log "3/4 generate ${N_SYNTHETIC:-1000} synthetic traces from the anonymized seeds with the Data Designer CLI"
data-designer create "$CFG/upscale.yaml" -n "${N_SYNTHETIC:-1000}" -d synthetic -o artifacts -f jsonl --no-tui
generated="$(ls -t artifacts/synthetic*/*.jsonl | head -1)"
rm -rf artifacts/synthetic*/dropped-columns-parquet-files     # Data Designer keeps dropped columns here: the original text

log "4/4 filter: judge scores >= 4 and every tool result follows its call; then check for leaked emails"
jq -c '.trace.messages as $m
  | select([.trace_quality[].score | tonumber] | min >= 4)
  | select(all(range(1; $m | length) | select($m[.].role == "tool"); $m[. - 1].tool_calls[0].id? == $m[.].tool_call_id))
  | {messages: $m | map(with_entries(select(.value != null and .value != []))), metadata: {scenario, temperament, auth_path, outcome}}' \
  "$generated" > synthetic.jsonl
grep -owFf <(jq -r '.[].messages[] | select(.role != "system") | .. | strings | scan("[\\w.+-]+@[\\w-]+\\.[a-z]+")' raw.json | sort -u) synthetic.jsonl \
  && { log "LEAK: the original emails above appear in synthetic.jsonl" >&2; exit 1; }
log "done: kept $(wc -l < synthetic.jsonl | tr -d ' ') / $(wc -l < "$generated" | tr -d ' ') synthetic traces, no original emails -> $OUT/synthetic.jsonl"
