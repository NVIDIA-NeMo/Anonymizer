#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# format.sh -- format (or check formatting of) Python files with ruff
#
# Usage:
#   ./format.sh                          # fix mode: ruff format + ruff check --fix
#   ./format.sh --check                  # check mode: ruff format --check (exit 1 if unformatted)
#   ./format.sh src/foo.py bar.py        # fix specific files
#   ./format.sh --check src/foo.py       # check mode on specific files
#
# Lint-rule violations (ruff check without --fix) are handled by ruff_check.sh.
# Copyright headers are handled separately by copyright_fixer.py.
#

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
# shellcheck source=_lib.sh
source "$REPO_ROOT/tools/codestyle/_lib.sh"

require_tool ruff

collect_py_files "$@"
[[ ${#PY_FILES[@]} -eq 0 ]] && exit 0

if [[ "$CHECK_MODE" == true ]]; then
    ruff format --check "${PY_FILES[@]}"
else
    ruff format "${PY_FILES[@]}"
    # this does import sorting and autofixes
    ruff check --fix "${PY_FILES[@]}"
fi
