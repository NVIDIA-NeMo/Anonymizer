#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu

REPO_ROOT=${REPO_ROOT:-$(git rev-parse --show-toplevel)}
TOOL_BASE="${REPO_ROOT}/tools/binaries"
source "${TOOL_BASE}/defs.sh"
source "${TOOL_BASE}/common_functions.sh"

TOOLS_YAML="${TOOL_BASE}/tools.yaml"

# Parse command line arguments
BOOTSTRAP_ONLY=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --bootstrap-only)
            BOOTSTRAP_ONLY=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --bootstrap-only  Install only bootstrap tools (yq, ty, uv)"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Non-interactive mode for CI (skip prompts)
if [[ "${BOOTSTRAP_NON_INTERACTIVE:-0}" == "1" ]]; then
    export DEBIAN_FRONTEND=noninteractive
fi

# Validate supported platforms (darwin/linux, amd64/arm64 only)
validate_platform() {
    if [[ "$OS" != "darwin" && "$OS" != "linux" ]]; then
        echo "Error: Unsupported OS: $OS (only darwin and linux are supported)"
        exit 1
    fi
    if [[ "$ARCH" != "amd64" && "$ARCH" != "arm64" ]]; then
        echo "Error: Unsupported architecture: $ARCH (only amd64 and arm64 are supported)"
        exit 1
    fi
}

# Ensure yq is available (bootstrap dependency)
ensure_yq() {
    if command -v yq >/dev/null 2>&1; then
        return 0
    fi
    echo "yq not found, installing..."
    bash "${TOOL_BASE}/install_yq.sh"
    # Add to path for this session
    export PATH="$HOME/.local/bin:$PATH"
}

# Build URL from template with placeholders
build_url() {
    local name=$1
    local url version os_val arch_val

    url=$(yq -r ".tools.${name}.url" "$TOOLS_YAML")
    version=$(yq -r ".tools.${name}.version" "$TOOLS_YAML")

    # Check for OS mapping (e.g., darwin -> macos)
    os_val=$(yq -r ".tools.${name}.os_map.${OS} // \"${OS}\"" "$TOOLS_YAML")
    # Check for arch mapping
    arch_val=$(yq -r ".tools.${name}.arch_map.${ARCH} // \"${ARCH}\"" "$TOOLS_YAML")

    # Substitute placeholders
    url="${url//\{version\}/$version}"
    url="${url//\{os\}/$os_val}"
    url="${url//\{arch\}/$arch_val}"

    echo "$url"
}

# Generic installer for simple binary tools
install_binary_tool() {
    local name=$1
    local check_command version url extract darwin_brew post_install install_path

    echo "----------------------------------------"
    echo "Processing: $name"

    check_command=$(yq -r ".tools.${name}.check_command" "$TOOLS_YAML")
    version=$(yq -r ".tools.${name}.version" "$TOOLS_YAML")

    # Check if already installed
    if eval "$check_command" >/dev/null 2>&1; then
        echo "$name is already installed"
        eval "$check_command" || true
        return 0
    fi

    # Check for darwin_brew option (use brew on macOS)
    darwin_brew=$(yq -r ".tools.${name}.darwin_brew // \"\"" "$TOOLS_YAML")
    if [[ "$OS" == "darwin" && -n "$darwin_brew" ]]; then
        if command -v brew >/dev/null 2>&1; then
            echo "Installing $name via Homebrew ($darwin_brew)..."
            brew install "$darwin_brew"
            # Run post_install if present
            post_install=$(yq -r ".tools.${name}.post_install // \"\"" "$TOOLS_YAML")
            if [[ -n "$post_install" ]]; then
                eval "$post_install"
            fi
            return 0
        fi
    fi

    # Build download URL
    url=$(build_url "$name")
    echo "Installing $name ${version}..."
    echo "Downloading from: $url"

    # Determine install path (default to ~/.local/bin/<name>)
    install_path="$HOME/.local/bin/$name"

    # Check if we need to extract from archive
    extract=$(yq -r ".tools.${name}.extract // \"\"" "$TOOLS_YAML")

    if [[ -n "$extract" ]]; then
        # Download and extract from tarball
        local workdir os_val arch_val
        workdir=$(mktemp -d)

        # Get mapped OS/arch values for extract path substitution
        os_val=$(yq -r ".tools.${name}.os_map.${OS} // \"${OS}\"" "$TOOLS_YAML")
        arch_val=$(yq -r ".tools.${name}.arch_map.${ARCH} // \"${ARCH}\"" "$TOOLS_YAML")

        # Substitute placeholders in extract path
        extract="${extract//\{version\}/$version}"
        extract="${extract//\{os\}/$os_val}"
        extract="${extract//\{arch\}/$arch_val}"

        (
            cd "$workdir"
            curl -sSL "$url" -o archive.tar.gz
            tar -xzf archive.tar.gz
            mv "$extract" "$install_path"
        )
        rm -rf "$workdir"
    else
        # Direct binary download
        curl -sSL "$url" -o "$install_path"
    fi

    chmod +x "$install_path"

    # Verify installation
    if eval "$check_command" >/dev/null 2>&1; then
        eval "$check_command" || true
    else
        echo "Warning: $name installed but check command failed"
        echo "installed from url: $url"
        echo "Make sure $HOME/.local/bin is in your PATH"
        exit 1
    fi
    echo "$name installed successfully to $install_path"

    # Run post_install if present
    post_install=$(yq -r ".tools.${name}.post_install // \"\"" "$TOOLS_YAML")
    if [[ -n "$post_install" ]]; then
        eval "$post_install"
    fi
}

# Run custom install scripts
install_custom_script() {
    local name=$1
    local script
    script=$(yq -r ".custom_scripts[] | select(.name == \"${name}\") | .script" "$TOOLS_YAML")

    if [[ -z "$script" || "$script" == "null" ]]; then
        echo "Error: No script found for custom tool: $name"
        return 1
    fi

    echo "----------------------------------------"
    echo "Running custom script for: $name"
    bash "${TOOL_BASE}/${script}" || echo "Failed to install $name - continuing with other tools"
}

bootstrap_tools() {
    validate_platform
    mkdir -p "$HOME/.local/bin"
    add_to_path "$HOME/.local/bin"

    ensure_yq

    # Get list of tools marked as bootstrap (install first)
    local bootstrap_tool_list
    bootstrap_tool_list=$(yq -r '.tools | to_entries | .[] | select(.value.bootstrap == true) | .key' "$TOOLS_YAML")

    # Install bootstrap tools first (like yq, ty)
    for tool in $bootstrap_tool_list; do
        install_binary_tool "$tool" || echo "Failed to install $tool - continuing with other tools"
    done

    # Get custom scripts marked as bootstrap
    local bootstrap_custom_scripts
    bootstrap_custom_scripts=$(yq -r '.custom_scripts[] | select(.bootstrap == true) | .name' "$TOOLS_YAML")

    for name in $bootstrap_custom_scripts; do
        install_custom_script "$name"
    done

    # If --bootstrap-only flag is set, stop here
    if [[ "$BOOTSTRAP_ONLY" == "true" ]]; then
        echo "----------------------------------------"
        echo "Bootstrap tools installed (--bootstrap-only mode)"
        return 0
    fi

    # Get list of regular tools (not bootstrap)
    local regular_tools
    regular_tools=$(yq -r '.tools | to_entries | .[] | select(.value.bootstrap != true) | .key' "$TOOLS_YAML")

    # Install regular binary tools from YAML
    for tool in $regular_tools; do
        install_binary_tool "$tool" || echo "Failed to install $tool - continuing with other tools"
    done

    # Get custom scripts that are NOT bootstrap
    local regular_custom_scripts
    regular_custom_scripts=$(yq -r '.custom_scripts[] | select(.bootstrap != true) | .name' "$TOOLS_YAML")

    for name in $regular_custom_scripts; do
        install_custom_script "$name"
    done

    echo "----------------------------------------"
    echo "Bootstrap complete!"
}

bootstrap_tools
