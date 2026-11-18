#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Runs a tool at the version this repository's Mise config pins: sh .house-tool.sh TOOL [ARGUMENTS ...]
# The pre-commit hooks call it, so a commit from a shell without Mise activated, or from an IDE, runs
# the pinned tool rather than whatever is on PATH.
#   HOUSE_TOOLCHAIN=system  the TOOL on PATH, for CI and machines that install the pinned tools themselves
#   in a Mise task here     the TOOL on PATH, which Mise has already set up for this repository
#   otherwise               mise exec -- TOOL
# house writes this file and reports any change to it.
set -eu

if [ "$#" -eq 0 ]; then
    echo "usage: sh .house-tool.sh TOOL [ARGUMENTS ...]" >&2
    exit 2
fi
tool=$1
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)

case ${HOUSE_TOOLCHAIN:-} in
system)
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "HOUSE_TOOLCHAIN=system runs $tool from PATH, and PATH has none." >&2
        echo "Install $tool at the version the Mise config pins, or unset HOUSE_TOOLCHAIN to use Mise." >&2
        exit 127
    fi
    exec "$@"
    ;;
"") ;;
*)
    echo "HOUSE_TOOLCHAIN must be system or unset, not $HOUSE_TOOLCHAIN." >&2
    exit 2
    ;;
esac

if [ -n "${MISE_PROJECT_ROOT:-}" ] && [ "$(CDPATH= cd -- "$MISE_PROJECT_ROOT" 2>/dev/null && pwd -P)" = "$root" ]; then
    exec "$@"
fi

if ! command -v mise >/dev/null 2>&1; then
    echo "Mise runs $tool at the version this repository pins, and PATH has no mise." >&2
    echo "Install Mise (https://mise.jdx.dev), or set HOUSE_TOOLCHAIN=system to run $tool from PATH." >&2
    exit 127
fi
exec mise exec -- "$@"
