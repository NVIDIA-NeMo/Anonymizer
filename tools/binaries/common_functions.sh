#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


REPO_ROOT=${REPO_ROOT:-$(git rev-parse --show-toplevel)}
source "${REPO_ROOT}/tools/binaries/defs.sh"

maybe_install_brew_dep() {
    dep=$1
    if ! brew list -1 | grep -q "$dep"; then
        brew install "$dep"
    fi
}

set -ue
## adds a directory to the PATH if it exists and is not already in the PATH
# If the second argument is "after", it adds the directory to the end of the PATH
# Otherwise, it adds the directory to the beginning of the PATH.
add_to_path() {
  new_path=${1%/}
  if [ -d "$1" ] && ! echo "$PATH" | grep -E -q "(^|:)$new_path($|:)" ; then
      if [ "$2" = "after" ] ; then
          PATH="${PATH:+${PATH}:}$new_path"
      else
          PATH="$new_path:${PATH:+${PATH}:}"
      fi
  fi
}

set +ue
