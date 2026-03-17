#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# _lib.sh -- shared helpers for lint/format/typecheck scripts
#
# Source this file; do not execute it directly.
#
# Provides:
#   REPO_ROOT                    -- absolute path to the repo root
#   require_tool <name>          -- die if <name> is not on PATH; warn on version mismatch
#   collect_py_files "$@"        -- populate PY_FILES array and CHECK_MODE
#
# collect_py_files recognises --check and exports CHECK_MODE=true/false
# so callers (format.sh) can inspect it.  Results are stored in the
# PY_FILES array -- call the function directly (not inside $(...)) so
# both PY_FILES and CHECK_MODE survive in the parent shell.
#

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# check_tool_version -- warn when installed version differs from tools.yaml
# ---------------------------------------------------------------------------
check_tool_version() {
    local name="$1"
    local tools_yaml="$REPO_ROOT/tools/binaries/tools.yaml"
    [[ ! -f "$tools_yaml" ]] && return 0
    command -v yq >/dev/null 2>&1 || return 0

    local expected
    expected=$(yq ".tools.${name}.version" "$tools_yaml" 2>/dev/null)
    [[ -z "$expected" || "$expected" == "null" ]] && return 0

    local actual
    actual=$("$name" --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+[^ ]*' | head -1)
    [[ -z "$actual" ]] && return 0

    if [[ "$actual" != "$expected" ]]; then
        echo "WARNING: $name version $actual differs from expected $expected (from tools.yaml)" >&2
        echo "         Run: make bootstrap-tools && export PATH=\"\$HOME/.local/bin:\$PATH\"" >&2
    fi
}

# ---------------------------------------------------------------------------
# require_tool -- die if tool is missing; warn on version mismatch
# ---------------------------------------------------------------------------
require_tool() {
    local name="$1"
    if ! command -v "$name" >/dev/null 2>&1; then
        echo "ERROR: $name not found on PATH." >&2
        echo "       Run: make bootstrap-tools && export PATH=\"\$HOME/.local/bin:\$PATH\"" >&2
        exit 1
    fi
    check_tool_version "$name"
}

# ---------------------------------------------------------------------------
# collect_py_files -- parse args and populate PY_FILES array
#
# Modes:
#   (no args)                    → all tracked .py files
#   file1.py file2.py ...        → those exact files
#
# Also strips --check from the arg list and exports CHECK_MODE.
#
# IMPORTANT: call directly -- not inside $(...) -- so globals survive.
# ---------------------------------------------------------------------------
PY_FILES=()
CHECK_MODE=false
export CHECK_MODE

collect_py_files() {
    local -a rest=()
    CHECK_MODE=false

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --check) CHECK_MODE=true; shift ;;
            *)       rest+=("$1"); shift ;;
        esac
    done
    export CHECK_MODE

    if [[ ${#rest[@]} -gt 0 ]]; then
        PY_FILES=("${rest[@]}")
    else
        local IFS=$'\n'
        # shellcheck disable=SC2207
        PY_FILES=($(git ls-files '*.py'))
        unset IFS
    fi

    if [[ ${#PY_FILES[@]} -eq 0 ]]; then
        echo "No Python files to check" >&2
        return 0
    fi
}
