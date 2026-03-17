#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Installs yq - needed to parse tools.yaml
# This script is called by bootstrap_tools.sh before parsing YAML
set -eu

REPO_ROOT=${REPO_ROOT:-$(git rev-parse --show-toplevel)}
source "${REPO_ROOT}/tools/binaries/defs.sh"

YQ_VERSION="v4.44.1"

install_yq() {
    if command -v yq >/dev/null 2>&1; then
        echo "yq is already installed at $(command -v yq)"
        yq --version
        return 0
    fi

    echo "Installing yq ${YQ_VERSION}..."

    # Validate supported platforms (darwin/linux, amd64/arm64 only)
    if [[ "$OS" != "darwin" && "$OS" != "linux" ]]; then
        echo "Error: Unsupported OS: $OS (only darwin and linux are supported)"
        exit 1
    fi
    if [[ "$ARCH" != "amd64" && "$ARCH" != "arm64" ]]; then
        echo "Error: Unsupported architecture: $ARCH (only amd64 and arm64 are supported)"
        exit 1
    fi

    mkdir -p "$HOME/.local/bin"

    local url="https://github.com/mikefarah/yq/releases/download/${YQ_VERSION}/yq_${OS}_${ARCH}"
    echo "Downloading from: $url"

    curl -sSL "$url" -o "$HOME/.local/bin/yq"
    chmod +x "$HOME/.local/bin/yq"

    echo "yq installed successfully to $HOME/.local/bin/yq"
    "$HOME/.local/bin/yq" --version
}

install_yq
