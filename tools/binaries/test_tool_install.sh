#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -eu

REPO_ROOT=${REPO_ROOT:-$(git rev-parse --show-toplevel)}
test_image="python:3.11"

echo "Testing tool installation and package install in container..."
docker run \
    --rm \
    --interactive \
    --name test_tool_install \
    --volume "$REPO_ROOT":/workspace \
    -e DEBIAN_FRONTEND=noninteractive \
    -e REPO_ROOT=/workspace \
    --platform linux/amd64 \
    "$test_image" \
    bash -c '
        set -eux
        apt-get update && apt-get install -y curl git
        cd /workspace

        # 1. Bootstrap tools (standalone binaries: uv, ty, yq, gh -- NOT ruff)
        make bootstrap-tools
        export PATH=$HOME/.local/bin:${PATH:+${PATH}:}

        # 2. Verify standalone tools are on PATH
        echo "=== Verifying installed tools ==="
        uv --version
        ty --version
        yq --version
        gh --version

        # 3. Install package into venv via bootstrap
        make bootstrap
        echo "=== Verifying venv and package ==="
        uv run ruff --version
        uv run python -c "import anonymizer; print(f\"anonymizer {anonymizer.__version__} installed successfully\")"

        echo "=== All checks passed ==="
    '
echo "Container test succeeded"
