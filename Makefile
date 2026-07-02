# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Compatibility entry points for contributors and scripts that still invoke
# make. New commands belong in .mise/tasks/ and should run via `mise run`.

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

.PHONY: run
run:
	@if [ -z "$(TASK)" ]; then \
		echo "Error: missing TASK. Usage: make run TASK=test [ARGS='...']" >&2; \
		exit 1; \
	fi
	@mise run "$(TASK)" $(ARGS)

define deprecated_target
.PHONY: $(1)
$(1):
	@printf '%s\n' 'deprecated; forwarding to `mise run $(2)`' >&2
	@mise run $(2)
endef

$(eval $(call deprecated_target,bootstrap,bootstrap))
$(eval $(call deprecated_target,install,install))
$(eval $(call deprecated_target,install-dev,install-dev))
$(eval $(call deprecated_target,install-dev-notebooks,install-dev-notebooks))
$(eval $(call deprecated_target,install-pre-commit,install-pre-commit))
$(eval $(call deprecated_target,format,format))
$(eval $(call deprecated_target,format-check,format-check))
$(eval $(call deprecated_target,typecheck,typecheck))
$(eval $(call deprecated_target,copyright,copyright))
$(eval $(call deprecated_target,copyright-check,copyright-check))
$(eval $(call deprecated_target,check,check))
$(eval $(call deprecated_target,lock-check,lock-check))
$(eval $(call deprecated_target,test,test))
$(eval $(call deprecated_target,test-e2e,test:e2e))
$(eval $(call deprecated_target,coverage,coverage))
$(eval $(call deprecated_target,build-wheel,build-wheel))
$(eval $(call deprecated_target,publish-pypi,publish:pypi))
$(eval $(call deprecated_target,install-dev-docs,install-dev-docs))
$(eval $(call deprecated_target,docs-serve,docs:serve))
$(eval $(call deprecated_target,docs-build,docs:build))
$(eval $(call deprecated_target,convert-notebooks,convert-notebooks))
$(eval $(call deprecated_target,clean-pycache,clean-pycache))
$(eval $(call deprecated_target,clean,clean))
$(eval $(call deprecated_target,clean-merged-branches,clean-merged-branches))
