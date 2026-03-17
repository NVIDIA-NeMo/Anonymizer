#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# typecheck.sh -- run ty type checks on Python files
#
# Usage:
#   ./typecheck.sh                        # all files (ty discovers and excludes via pyproject.toml)
#   ./typecheck.sh src/foo.py bar.py      # specific files
#

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
# shellcheck source=_lib.sh
source "$REPO_ROOT/tools/codestyle/_lib.sh"

require_tool ty

if [[ $# -eq 0 ]]; then
    ty check
else
    ty check "$@"
fi
