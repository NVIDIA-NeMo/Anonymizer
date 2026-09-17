<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Default OpenRouter provider

## Goal

Route the bundled external LLM aliases through OpenRouter while keeping the existing model IDs,
inference parameters, role selections, and local GLiNER2 detector unchanged.

## Changes

1. Replace the bundled NVIDIA provider with OpenRouter and use `OPENROUTER_API_KEY`.
2. Point the three hosted model aliases at the OpenRouter provider without changing their model IDs.
3. Add regression coverage for the provider contract and notebook configuration.
4. Update default-provider guidance, notebook sources, generated notebooks, and notebook CI.
5. Keep explicit custom NVIDIA provider and benchmark examples intact.

## Validation

- Run model-loader and notebook configuration tests.
- Run formatting, type, lock, copyright, and documentation checks.
- Verify generated notebooks match their source guidance.
