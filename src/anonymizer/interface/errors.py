# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


class AnonymizerError(Exception):
    """Base error for Anonymizer interface operations."""


class InvalidInputError(AnonymizerError):
    """Raised when input data or configuration is invalid."""


class InvalidConfigError(AnonymizerError):
    """Raised when model, provider, alias, or semantic configuration is invalid."""


class AnonymizerIOError(AnonymizerError):
    """Raised when file IO operations fail."""


class AnonymizerWorkflowError(AnonymizerError):
    """Raised when an underlying workflow step (preview, execution, or dataset load) fails.

    The original backend exception is preserved as ``__cause__`` via ``raise ... from exc``
    so callers can inspect the exact failure without the anonymizer layer leaking backend
    exception types into its own public error hierarchy.
    """


class InvalidDetectorResultError(AnonymizerError):
    """Raised when a ``detection_workflow`` returns a result that cannot be placed against its input.

    A custom detector's result is not a ``DetectorResult``, holds a malformed span, extra value, latent entity or
    failure, names a record id that is not an input id, or names one twice; or the built-in detector returned
    something other than its frame. No record of that batch is released, and no model is called after detection.
    """
