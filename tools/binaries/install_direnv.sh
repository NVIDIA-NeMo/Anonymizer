#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu

DIRENV_VERSION="2.37.1"
REPO_ROOT=${REPO_ROOT:-$(git rev-parse --show-toplevel)}
source "${REPO_ROOT}/tools/binaries/defs.sh"

# Parse arguments
CONFIGURE_SHELL=false
for arg in "$@"; do
    case "$arg" in
    --configure-shell)
        CONFIGURE_SHELL=true
        ;;
    --help | -h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Install direnv and create a default .envrc file"
        echo ""
        echo "Options:"
        echo "  --configure-shell    Automatically add direnv hook to your shell config"
        echo "  --help, -h           Show this help message"
        exit 0
        ;;
    *)
        echo "Unknown option: $arg"
        echo "Run '$0 --help' for usage information"
        exit 1
        ;;
    esac
done

install_direnv() {
    if command -v direnv >/dev/null; then
        echo "direnv is already installed at $(command -v direnv)"
    else
        echo "Installing direnv..."
        if [ "$OS" == "darwin" ]; then
            if command -v brew >/dev/null; then
                echo "Installing direnv via Homebrew..."
                brew install direnv
            else
                echo "Homebrew not found. Installing direnv binary..."
                mkdir -p "$HOME/.local/bin"
                wget -q "https://github.com/direnv/direnv/releases/download/v${DIRENV_VERSION}/direnv.darwin-arm64" -O direnv
                chmod +x direnv
                mv direnv "$HOME/.local/bin/direnv"
                echo "direnv installed to $HOME/.local/bin/direnv"
            fi
        elif [ "$OS" == "linux" ]; then
            echo "apt-get not found. Installing direnv binary..."
            mkdir -p "$HOME/.local/bin"
            direnv_arch=$(uname -m)
            if [ "$direnv_arch" == "x86_64" ]; then
                direnv_arch="amd64"
            elif [ "$direnv_arch" == "aarch64" ]; then
                direnv_arch="arm64"
            fi
            wget -q "https://github.com/direnv/direnv/releases/download/v${DIRENV_VERSION}/direnv.linux-${direnv_arch}" -O direnv
            chmod +x direnv
            mv direnv "$HOME/.local/bin/direnv"
            echo "direnv installed to $HOME/.local/bin/direnv"
        else
            echo "Error: Unsupported OS: $OS"
            exit 1
        fi
    fi

    echo ""
    echo "direnv installed successfully!"
    echo ""
    if [ "$CONFIGURE_SHELL" = false ]; then
        echo "To use direnv, add the following to your shell config file:"
        echo "  For bash (~/.bashrc):"
        echo "    eval \"\$(direnv hook bash)\""
        echo "  For zsh (~/.zshrc):"
        echo "    eval \"\$(direnv hook zsh)\""
        echo ""
        echo "Tip: Run this script with --configure-shell to do this automatically"
        echo ""
    fi
}

create_envrc() {
    envrc_path="${REPO_ROOT}/.envrc"

    if [ -f "$envrc_path" ]; then
        echo ".envrc file already exists at $envrc_path"
        echo "Skipping creation."
    else
        echo "Creating .envrc file at $envrc_path..."
        cat >"$envrc_path" <<'EOF'
# See https://github.com/direnv/direnv/wiki/Python
# This has to be before our PATH_adds as well
export VIRTUAL_ENV=.venv
layout python3
# this should make it so you don't have to source the venv directly
PATH_add $VIRTUAL_ENV/bin
# ensure local bin is ahead of the venv, mostly for UV. we want to use the
PATH_add ./script
EOF
        echo ".envrc file created successfully!"
        echo ""
        echo "To activate direnv in this directory, run:"
        echo "  cd ${REPO_ROOT} && direnv allow"
        echo ""
    fi
}

configure_shell() {
    echo ""
    echo "Configuring shell for direnv..."

    shell_rc=""
    hook_cmd=""
    shell_name="$(basename "$SHELL")"

    case "$shell_name" in
    bash)
        shell_rc="$HOME/.bashrc"
        hook_cmd='eval "$(direnv hook bash)"'
        ;;
    zsh)
        shell_rc="$HOME/.zshrc"
        hook_cmd='eval "$(direnv hook zsh)"'
        ;;
    *)
        echo "Unsupported shell: $shell_name"
        echo "Please manually add the direnv hook to your shell config."
        return 1
        ;;
    esac

    # Check if already configured
    if [ -f "$shell_rc" ] && grep -q "direnv hook" "$shell_rc"; then
        echo "direnv hook already configured in $shell_rc"
        return 0
    fi

    # Add hook to shell config
    echo "Adding direnv hook to $shell_rc..."
    {
        echo ""
        echo "# direnv hook (added by install_direnv.sh)"
        echo "$hook_cmd"
    } >>"$shell_rc"

    echo "✓ Successfully added direnv hook to $shell_rc"
    echo ""
    echo "To apply changes, either:"
    echo "  1. Restart your shell, or"
    echo "  2. Run: source $shell_rc"
    echo ""
}

install_direnv
create_envrc

if [ "$CONFIGURE_SHELL" = true ]; then
    configure_shell
fi
