<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Detector-Only Mode

Tracks [GitHub issue #280](https://github.com/NVIDIA-NeMo/Anonymizer/issues/280).

## Goal

Offer an explicit lower-latency detection path for callers that accept the
accuracy trade-off of using the configured GLiNER detector without LLM
validation or augmentation.

## Design

Add an opt-in `gliner_only` detection setting. When enabled, Anonymizer:

1. runs the configured GLiNER detector;
2. skips validator and augmenter model calls;
3. finalizes the same public entity and result schemas; and
4. records the skipped refinement stages as not applicable in telemetry.

The default pipeline is unchanged. Detector-only mode still honors configured
entity labels, exclusions, score thresholds, and source-offset validation.

## Compatibility and Safety

The setting is additive and defaults to `false`. It is a speed-versus-quality
choice, not a privacy guarantee. GLiNER can miss sensitive data or over-redact
benign text, so callers must benchmark representative data before deployment.

The mode should consume the canonical source-offset fix tracked by
[issue #279](https://github.com/NVIDIA-NeMo/Anonymizer/issues/279) before merge.

## Validation

- workflow tests proving validator and augmenter are not scheduled;
- configuration, serialization, label-filtering, and public API tests;
- measurement and W&B telemetry tests, including one detector call and no
  refinement calls; and
- the complete repository test suite.

## Non-Goals

- making detector-only mode the default;
- claiming parity with the full detection pipeline; or
- defining production thresholds before representative accuracy benchmarks.
