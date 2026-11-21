# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The error types the engine raises and users catch, defined below both layers.

Users import them from ``anonymizer`` or ``anonymizer.interface.errors``, which re-export these classes; each keeps
``anonymizer.interface.errors`` as its ``__module__``, so tracebacks and pickles name them as before. The cost:
``inspect.getsource`` on these classes looks in the wrong module, so it fails or returns the wrong lines.
"""

from __future__ import annotations

_PUBLIC_MODULE = "anonymizer.interface.errors"


class AnonymizerError(Exception):
    """Base error for Anonymizer interface operations."""

    __module__ = _PUBLIC_MODULE


class InvalidInputError(AnonymizerError):
    """Raised when input data or configuration is invalid."""

    __module__ = _PUBLIC_MODULE


class InvalidConfigError(AnonymizerError):
    """Raised when model, provider, alias, or semantic configuration is invalid."""

    __module__ = _PUBLIC_MODULE


class AnonymizerIOError(AnonymizerError):
    """Raised when file IO operations fail."""

    __module__ = _PUBLIC_MODULE


class AnonymizerWorkflowError(AnonymizerError):
    """Raised when an underlying workflow step (preview, execution, or dataset load) fails.

    The original backend exception is preserved as ``__cause__`` via ``raise ... from exc``
    so callers can inspect the exact failure without the anonymizer layer leaking backend
    exception types into its own public error hierarchy.
    """

    __module__ = _PUBLIC_MODULE


class InvalidDetectorResultError(AnonymizerError):
    """Raised when a ``detection_workflow`` returns a result that cannot be placed against its input.

    A custom detector's result is not a ``DetectorResult``, holds a malformed span, extra value, latent entity or
    failure, names a record id that is not an input id, or names one twice. No record of that batch is released, and
    no model is called after detection.
    """

    __module__ = _PUBLIC_MODULE
