#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# ruff_check.sh -- check Python files for lint-rule violations (read-only)
#
# This is the read-only counterpart to `ruff check --fix` in format.sh.
# ruff exposes format and check as separate commands; this script wraps
# the read-only check so CI and `make format-check` can verify lint rules
# without modifying files.
#
# Usage:
#   ./ruff_check.sh                      # all tracked .py files
#   ./ruff_check.sh src/foo.py bar.py    # specific files
#

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
# shellcheck source=_lib.sh
source "$REPO_ROOT/tools/codestyle/_lib.sh"

require_tool ruff

collect_py_files "$@"
[[ ${#PY_FILES[@]} -eq 0 ]] && exit 0

ruff check "${PY_FILES[@]}"
