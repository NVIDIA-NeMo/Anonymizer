# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Mise discovery and bootstrap entry points.
# Developer commands live in .mise/tasks/ and run via `mise run`.

SHELL := /bin/bash
export PATH := $(HOME)/.local/share/mise/shims:$(HOME)/.local/bin:$(PATH)

MISE_GPG_KEY := 24853EC9F655CE80B48E6C3A8B81C9D17413A06D

.PHONY: help
help:
	@mise tasks

.PHONY: install-mise
install-mise:
	@MISE_GPG_KEY=$(MISE_GPG_KEY) bash tools/install-mise.sh

.PHONY: setup
setup: install-mise
	@MISE_YES=1 mise trust
	@MISE_YES=1 mise run setup
