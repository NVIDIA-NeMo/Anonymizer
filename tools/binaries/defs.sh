#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# shellcheck disable=SC2034
DEV="${DEV:-"$HOME/dev"}"
# shellcheck disable=SC2034
OS="$(uname | tr '[:upper:]' '[:lower:]')"
# shellcheck disable=SC2034
ARCH="$(uname -m | sed -e 's/x86_64/amd64/' -e 's/\(arm\)\(64\)\?.*/\1\2/' -e 's/aarch64$/arm64/')"
